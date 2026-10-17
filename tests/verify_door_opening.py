"""Verify physical grasping, handle depression and door opening, with controls.

uv run --no-sync python tests/verify_door_opening.py
uv run --no-sync python tests/verify_door_opening.py --record
uv run --no-sync python tests/verify_door_opening.py --render

Only robot actuator targets are commanded. Door coordinates are written solely
in a separate reference MjData used to plan the desired end-effector trajectory;
the simulated door is passive and moves through collision/friction forces.
"""

import argparse
import csv
import json
from pathlib import Path
import time

import mujoco
import numpy as np


MODEL = Path(__file__).resolve().parents[1] / "models" / "door_scene.xml"
CONTROL_DT = 0.02
PRESS_ANGLE = 1.3
OPEN_ANGLE = 0.35
GRIP_COMMAND = 170.0


def rotation_x(angle: float) -> np.ndarray:
    c, s = np.cos(angle), np.sin(angle)
    return np.array([[1, 0, 0], [0, c, -s], [0, s, c]])


def camera() -> mujoco.MjvCamera:
    cam = mujoco.MjvCamera()
    mujoco.mjv_defaultCamera(cam)
    cam.lookat[:] = [0.4, 0, 0.65]
    cam.distance, cam.azimuth, cam.elevation = 2.3, 45, -20
    return cam


class DoorTrial:
    def __init__(self, scenario: str, model_path=MODEL):
        self.scenario = scenario
        self.model = m = mujoco.MjModel.from_xml_path(str(model_path))
        self.data = d = mujoco.MjData(m)
        mujoco.mj_resetDataKeyframe(m, d, m.key("home").id)
        mujoco.mj_forward(m, d)
        self.ik = mujoco.MjData(m)
        self.reference = mujoco.MjData(m)
        self.joints = np.array([m.joint(f"joint_{i}").id for i in range(1, 8)])
        self.qadr = m.jnt_qposadr[self.joints]
        self.vadr = m.jnt_dofadr[self.joints]
        self.actuators = np.array([m.actuator(f"joint_{i}").id for i in range(1, 8)])
        self.gripper = m.actuator("fingers_actuator").id
        self.ee = m.site("pinch_site").id
        self.handle_site = m.site("door_handle_site").id
        self.handle_geom = m.geom("door_handle").id
        self.hinge = m.jnt_qposadr[m.joint("door_hinge").id]
        self.latch = m.jnt_qposadr[m.joint("door_latch_joint").id]
        self.door_dofs = m.jnt_dofadr[[m.joint("door_hinge").id, m.joint("door_latch_joint").id]]
        self.door_bodies = [m.body("door_panel_body").id, m.body("door_handle_body").id]
        door_joint_ids = {m.joint("door_hinge").id, m.joint("door_latch_joint").id}
        for actuator in range(m.nu):
            if m.actuator_trntype[actuator] == mujoco.mjtTrn.mjTRN_JOINT:
                if int(m.actuator_trnid[actuator, 0]) in door_joint_ids:
                    raise RuntimeError("Verification requires a passive, unactuated door")
        self.pad_sides = {
            m.geom(f"{side}_pad{i}").id: side for side in ("left", "right") for i in (1, 2)
        }
        self.last_target = d.qpos[self.qadr].copy()
        self.jacp = np.zeros((3, m.nv))
        self.jacr = np.zeros((3, m.nv))
        self.home_position = d.site_xpos[self.ee].copy()
        self.home_rotation = d.site_xmat[self.ee].reshape(3, 3).copy()
        self.grasp_rotation = rotation_x(np.pi / 2) @ self.home_rotation
        # Roll the gripper so its fingers close across handle thickness, rather
        # than along the lever's length (which exceeds the gripper opening).
        self.relative_rotation = d.site_xmat[self.handle_site].reshape(3, 3).T @ self.grasp_rotation
        self.relative_position = np.array([0, -0.01, 0])
        self.rows = []
        self.phases = {}
        self.snapshots = {}
        self.max_base_displacement = 0.0
        self.max_door_angle = 0.0
        self.door_forces_applied = False

    def handle_pose(self, door_angle: float, handle_angle: float) -> tuple:
        m, ref = self.model, self.reference
        ref.qpos[:] = m.key("home").qpos
        ref.qpos[self.hinge], ref.qpos[self.latch] = door_angle, handle_angle
        mujoco.mj_forward(m, ref)
        rotation = ref.site_xmat[self.handle_site].reshape(3, 3)
        return (ref.site_xpos[self.handle_site] + rotation @ self.relative_position,
                rotation @ self.relative_rotation)

    def step(self, position: np.ndarray, rotation: np.ndarray, grip: float, phase: str) -> dict:
        m, d, ik = self.model, self.data, self.ik
        ik.qpos[:] = d.qpos
        ik.qpos[self.qadr] = self.last_target
        for _ in range(12):
            mujoco.mj_forward(m, ik)
            current_rotation = ik.site_xmat[self.ee].reshape(3, 3)
            position_error = position - ik.site_xpos[self.ee]
            rotation_error = 0.5 * sum(
                np.cross(current_rotation[:,i], rotation[:,i]) for i in range(3)
            )
            if np.linalg.norm(position_error) < 1e-4 and np.linalg.norm(rotation_error) < 1e-3:
                break
            mujoco.mj_jacSite(m, ik, self.jacp, self.jacr, self.ee)
            jac = np.vstack((self.jacp[:, self.vadr], 0.3 * self.jacr[:, self.vadr]))
            error = np.r_[position_error, 0.3 * rotation_error]
            delta = jac.T @ np.linalg.solve(jac @ jac.T + 1e-5 * np.eye(6), error)
            ik.qpos[self.qadr] += np.clip(delta, -0.04, 0.04)
            for joint, adr in zip(self.joints, self.qadr):
                if m.jnt_limited[joint]:
                    ik.qpos[adr] = np.clip(ik.qpos[adr], *m.jnt_range[joint])
        self.last_target = ik.qpos[self.qadr].copy()
        # Feed forward robot bias torque via position targets; the existing
        # actuator force limits still apply. No forces are applied to the door.
        targets = self.last_target + d.qfrc_bias[self.vadr] / m.actuator_gainprm[self.actuators, 0]
        for i, actuator in enumerate(self.actuators):
            if m.actuator_ctrllimited[actuator]:
                targets[i] = np.clip(targets[i], *m.actuator_ctrlrange[actuator])
        d.ctrl[self.actuators] = targets
        d.ctrl[self.gripper] = grip

        normal = {"left": 0.0, "right": 0.0}
        penetration = 0.0
        max_force = 0.0
        worst_pair = ""
        # Inspect every physics step, including contact transients between
        # control ticks; final-tick-only checks can miss peak penetration.
        for _ in range(round(CONTROL_DT / m.opt.timestep)):
            mujoco.mj_step(m, d)
            mujoco.mj_forward(m, d)
            if not (np.isfinite(d.qpos).all() and np.isfinite(d.qvel).all()):
                raise RuntimeError(f"{self.scenario}/{phase}: non-finite state")
            if any(w.number for w in d.warning):
                raise RuntimeError(f"{self.scenario}/{phase}: MuJoCo warning")
            self.max_base_displacement = max(self.max_base_displacement, float(np.linalg.norm(d.qpos[:2])))
            self.max_door_angle = max(self.max_door_angle, float(d.qpos[self.hinge]))
            self.door_forces_applied |= bool(
                np.any(d.qfrc_applied[self.door_dofs]) or np.any(d.xfrc_applied[self.door_bodies])
            )
            normal = {"left": 0.0, "right": 0.0}
            for index, contact in enumerate(d.contact):
                force = np.zeros(6)
                mujoco.mj_contactForce(m, d, index, force)
                max_force = max(max_force, float(force[0]))
                if -contact.dist > penetration:
                    penetration = float(-contact.dist)
                    worst_pair = " / ".join(
                        m.geom(g).name or m.body(int(m.geom_bodyid[g])).name
                        for g in (contact.geom1, contact.geom2)
                    )
                pair = {int(contact.geom1), int(contact.geom2)}
                if self.handle_geom in pair:
                    for geom in pair:
                        if geom in self.pad_sides:
                            normal[self.pad_sides[geom]] += max(0.0, float(force[0]))
        row = {
            "scenario": self.scenario, "phase": phase, "time_s": float(d.time),
            "door_rad": float(d.qpos[self.hinge]), "handle_rad": float(d.qpos[self.latch]),
            "left_pad_normal_N": normal["left"], "right_pad_normal_N": normal["right"],
            "bilateral_grasp": normal["left"] > 0.05 and normal["right"] > 0.05,
            "penetration_m": penetration, "worst_pair": worst_pair,
            "max_contact_normal_N": max_force,
            "tracking_error_m": float(np.linalg.norm(position - d.site_xpos[self.ee])),
        }
        self.rows.append(row)
        return row

    def segment(self, name: str, seconds: float, trajectory, grip: float, on_tick=None) -> None:
        count = round(seconds / CONTROL_DT)
        start = len(self.rows)
        for i in range(count):
            tick_start = time.monotonic()
            t = (i + 1) / count
            t = t * t * (3 - 2 * t)
            position, rotation = trajectory(t)
            self.step(position, rotation, grip, name)
            if on_tick:
                on_tick(self, tick_start)
        rows = self.rows[start:]
        self.phases[name] = {
            "door_rad": rows[-1]["door_rad"], "handle_rad": rows[-1]["handle_rad"],
            "bilateral_fraction": float(np.mean([r["bilateral_grasp"] for r in rows])),
            "tail_bilateral_fraction": float(np.mean([r["bilateral_grasp"] for r in rows[-25:]])),
        }
        self.snapshots[name] = (self.data.qpos.copy(), self.data.ctrl.copy())
        print(f"{self.scenario}/{name}: door={np.rad2deg(rows[-1]['door_rad']):.2f} deg, "
              f"handle={np.rad2deg(rows[-1]['handle_rad']):.2f} deg, "
              f"bilateral={self.phases[name]['bilateral_fraction']:.0%}", flush=True)

    def run(self, on_tick=None) -> dict:
        grip = 0.0 if self.scenario == "no_grip" else GRIP_COMMAND
        press = 0.0 if self.scenario == "no_press" else PRESS_ANGLE
        p, r = self.handle_pose(0, 0)
        self.segment("align", 2, lambda t: (self.home_position, rotation_x(t*np.pi/2) @ self.home_rotation), 0, on_tick)
        self.segment("approach", 2, lambda t: (self.home_position+t*(p-self.home_position), r), 0, on_tick)
        self.segment("grasp", 2, lambda t: (p, r), grip, on_tick)
        self.segment("press", 3, lambda t: self.handle_pose(0, press*t), grip, on_tick)
        self.segment("pull", 4, lambda t: self.handle_pose(OPEN_ANGLE*t, press), grip, on_tick)
        self.segment("hold", 1, lambda t: self.handle_pose(OPEN_ANGLE, press), grip, on_tick)
        worst = max(self.rows, key=lambda row: row["penetration_m"])
        return {
            "scenario": self.scenario, "phases": self.phases,
            "final_door_deg": float(np.rad2deg(self.data.qpos[self.hinge])),
            "final_handle_deg": float(np.rad2deg(self.data.qpos[self.latch])),
            "max_penetration_m": worst["penetration_m"], "worst_pair": worst["worst_pair"],
            "max_contact_normal_N": max(row["max_contact_normal_N"] for row in self.rows),
            "max_door_rad": self.max_door_angle,
            "max_base_displacement_m": self.max_base_displacement,
            "warnings": sum(int(w.number) for w in self.data.warning),
            "door_forces_applied": self.door_forces_applied,
        }


def assess(results: list[dict]) -> list[str]:
    normal, no_press, no_grip = results
    problems = []
    if normal["phases"]["grasp"]["tail_bilateral_fraction"] < 0.95:
        problems.append("Normal trial did not establish a bilateral fingertip grasp")
    if normal["phases"]["press"]["handle_rad"] < 1.1:
        problems.append("Handle did not reach the release angle")
    if not 17 <= normal["final_door_deg"] <= 24:
        problems.append("Normal trial did not open the door within its existing range")
    for phase in ("pull", "hold"):
        if normal["phases"][phase]["bilateral_fraction"] < 0.95:
            problems.append(f"Fingertip grasp was lost during {phase}")
    for result in (no_press, no_grip):
        if result["max_door_rad"] >= 0.05:
            problems.append(f"{result['scenario']} unexpectedly opened the door")
    for result in results:
        if result["max_penetration_m"] > 0.0015:
            problems.append(f"{result['scenario']} exceeded 1.5 mm contact penetration")
        if result["warnings"] or result["door_forces_applied"]:
            problems.append(f"{result['scenario']} has warnings or directly applied door forces")
        if result["max_base_displacement_m"] > 0.001:
            problems.append(f"{result['scenario']} did not hold the base position")
    return problems


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--render", action="store_true", help="Show the normal trial in real time.")
    parser.add_argument("--record", action="store_true", help="Save a GIF and phase PNGs for the normal trial.")
    parser.add_argument("--output-dir", type=Path, default=MODEL.parents[1] / "runs" / "door_opening")
    args = parser.parse_args()
    args.output_dir.mkdir(parents=True, exist_ok=True)
    trials, results, frames = [], [], []
    normal = DoorTrial("normal")
    renderer = mujoco.Renderer(normal.model, height=480, width=640) if args.record else None
    viewer = None
    if args.render:
        from mujoco import viewer as mj_viewer
        viewer = mj_viewer.launch_passive(normal.model, normal.data)
        cam = camera()
        viewer.cam.lookat[:] = cam.lookat
        viewer.cam.distance, viewer.cam.azimuth, viewer.cam.elevation = cam.distance, cam.azimuth, cam.elevation
    try:
        def on_tick(trial, tick_start):
            if renderer and len(trial.rows) % 5 == 0:
                from PIL import Image
                renderer.update_scene(trial.data, camera=camera())
                frames.append(Image.fromarray(renderer.render()).convert("RGB"))
            if viewer and viewer.is_running():
                viewer.sync()
                time.sleep(max(0.0, CONTROL_DT - (time.monotonic() - tick_start)))

        results.append(normal.run(on_tick))
        trials.append(normal)
        if renderer:
            from PIL import Image
            frames[0].save(args.output_dir / "demo.gif", save_all=True, append_images=frames[1:], duration=100, loop=0)
            snapshot_data = mujoco.MjData(normal.model)
            for phase in ("grasp", "press", "hold"):
                snapshot_data.qpos[:], snapshot_data.ctrl[:] = normal.snapshots[phase]
                mujoco.mj_forward(normal.model, snapshot_data)
                renderer.update_scene(snapshot_data, camera=camera())
                Image.fromarray(renderer.render()).save(args.output_dir / f"{phase}.png")
    finally:
        if renderer:
            renderer.close()
        if viewer:
            viewer.close()
    for scenario in ("no_press", "no_grip"):
        trial = DoorTrial(scenario)
        results.append(trial.run())
        trials.append(trial)
    failures = assess(results)
    report = {
        "passed": not failures, "failures": failures, "model": str(MODEL),
        "method": "Robot position actuators + bias-torque feedforward + contact/friction; passive door",
        "gripper_command": GRIP_COMMAND, "target_handle_rad": PRESS_ANGLE,
        "target_door_rad": OPEN_ANGLE, "contact_penetration_limit_m": 0.0015,
        "results": results,
    }
    (args.output_dir / "validation.json").write_text(json.dumps(report, indent=2), encoding="utf-8")
    with (args.output_dir / "trajectory.csv").open("w", newline="", encoding="utf-8") as stream:
        writer = csv.DictWriter(stream, fieldnames=list(normal.rows[0]))
        writer.writeheader()
        for trial in trials:
            writer.writerows(trial.rows)
    print(f"{'PASS' if not failures else 'FAIL'}: {args.output_dir.resolve() / 'validation.json'}")
    if failures:
        raise SystemExit("; ".join(failures))


if __name__ == "__main__":
    main()
