"""View and validate the TidyBot + door scene.

uv run --no-sync python tests/view_door_scene.py
uv run --no-sync python tests/view_door_scene.py --check
uv run --no-sync python tests/view_door_scene.py --check --screenshot runs/door_scene/preview.png
"""

import argparse
import json
from pathlib import Path
import time

import mujoco
import numpy as np


MODEL = Path(__file__).resolve().parents[1] / "models" / "door_scene.xml"


def reset_home(model: mujoco.MjModel, data: mujoco.MjData) -> None:
    mujoco.mj_resetDataKeyframe(model, data, model.key("home").id)
    mujoco.mj_forward(model, data)


def penetrating_contacts(model: mujoco.MjModel, data: mujoco.MjData) -> list:
    return [
        {"geom1": model.geom(c.geom1).name or str(c.geom1),
         "geom2": model.geom(c.geom2).name or str(c.geom2), "distance_m": float(c.dist)}
        for c in data.contact if c.dist < -1e-6
    ]


def collision_geoms(model: mujoco.MjModel, root_name: str) -> list[int]:
    root_id = model.body(root_name).id
    result = []
    for geom_id in range(model.ngeom):
        if not (model.geom_contype[geom_id] or model.geom_conaffinity[geom_id]):
            continue
        body_id = int(model.geom_bodyid[geom_id])
        while body_id:
            if body_id == root_id:
                result.append(geom_id)
                break
            body_id = int(model.body_parentid[body_id])
    return result


def validate(model: mujoco.MjModel, data: mujoco.MjData) -> dict:
    # Also check the raw XML default state, before applying a keyframe.
    mujoco.mj_resetData(model, data)
    mujoco.mj_forward(model, data)
    problems = penetrating_contacts(model, data)
    if problems:
        raise RuntimeError(f"Default pose penetrates: {problems}")

    reset_home(model, data)
    problems = penetrating_contacts(model, data)
    if problems:
        raise RuntimeError(f"Home pose penetrates: {problems}")
    if model.body("door_frame").dofnum or model.body("door_fixture").dofnum:
        raise RuntimeError("Door frame must be fixed")
    door_joint_ids = {model.joint(name).id for name in ("door_hinge", "door_latch_joint")}
    for actuator_id in range(model.nu):
        if model.actuator_trntype[actuator_id] == mujoco.mjtTrn.mjTRN_JOINT:
            if int(model.actuator_trnid[actuator_id, 0]) in door_joint_ids:
                raise RuntimeError("Door joints must be passive")

    robot = collision_geoms(model, "base_link")
    door = collision_geoms(model, "door_fixture")
    # Explicit pair distances check robot/door even independently of contact filtering.
    distance, robot_geom, door_geom = min(
        (float(mujoco.mj_geomDistance(model, data, a, b, 10, None)), a, b)
        for a in robot for b in door
    )
    if distance <= 0:
        raise RuntimeError(f"Robot and door overlap: {distance} m")
    floor_id = model.geom("floor").id
    floor_clearance = min(
        float(mujoco.mj_geomDistance(model, data, g, floor_id, 10, None))
        for g in robot + door
    )
    if floor_clearance < -1e-6:
        raise RuntimeError(f"Geometry penetrates the floor: {floor_clearance} m")

    ee = data.site("pinch_site").xpos.copy()
    handle = data.site("door_handle_site").xpos.copy()
    result = {
        "model": str(MODEL), "nq": model.nq, "nv": model.nv, "nu": model.nu,
        "default_penetrations": 0, "home_penetrations": 0,
        "end_effector_m": ee.tolist(), "handle_m": handle.tolist(),
        "end_effector_to_handle_m": float(np.linalg.norm(handle - ee)),
        "robot_door_clearance_m": distance,
        "closest_geoms": [model.geom(robot_geom).name or str(robot_geom), model.geom(door_geom).name],
        "floor_clearance_m": floor_clearance,
    }
    # Hold the existing home actuator targets for two seconds; no door actuation.
    duration = 2.0
    steps = int(np.ceil(duration / model.opt.timestep))
    max_contacts = 0
    for step in range(steps):
        mujoco.mj_step(model, data)
        mujoco.mj_forward(model, data)
        if not (np.isfinite(data.qpos).all() and np.isfinite(data.qvel).all()):
            raise RuntimeError(f"Non-finite state at step {step}")
        if any(w.number for w in data.warning):
            raise RuntimeError(f"MuJoCo warning at step {step}")
        problems = penetrating_contacts(model, data)
        if problems:
            raise RuntimeError(f"Penetration at step {step}: {problems}")
        max_contacts = max(max_contacts, data.ncon)
    result.update(simulated_seconds=float(data.time), simulation_steps=steps,
                  max_contacts=max_contacts, simulation_penetrations=0, warnings=0)
    reset_home(model, data)
    return result


def camera() -> mujoco.MjvCamera:
    cam = mujoco.MjvCamera()
    mujoco.mjv_defaultCamera(cam)
    cam.lookat[:] = [0.4, 0, 0.65]
    cam.distance = 2.3
    cam.azimuth = 45
    cam.elevation = -20
    return cam


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--check", action="store_true", help="Validate loading, clearances and two seconds of physics.")
    parser.add_argument("--screenshot", type=Path, help="Save the home pose as a PNG instead of opening a window.")
    args = parser.parse_args()
    model = mujoco.MjModel.from_xml_path(str(MODEL))
    data = mujoco.MjData(model)
    reset_home(model, data)
    if args.check:
        print(json.dumps(validate(model, data), indent=2))
    if args.screenshot:
        # Pillow is available through the project's matplotlib dependency.
        from PIL import Image

        args.screenshot.parent.mkdir(parents=True, exist_ok=True)
        with mujoco.Renderer(model, height=720, width=960) as renderer:
            renderer.update_scene(data, camera=camera())
            Image.fromarray(renderer.render()).save(args.screenshot)
        print(f"Saved {args.screenshot.resolve()}")
    if args.check or args.screenshot:
        return

    from mujoco import viewer as mj_viewer

    with mj_viewer.launch_passive(model, data) as viewer:
        mujoco.mjv_defaultCamera(viewer.cam)
        viewer.cam.lookat[:] = camera().lookat
        viewer.cam.distance = 2.3
        viewer.cam.azimuth = 45
        viewer.cam.elevation = -20
        while viewer.is_running():
            viewer.sync()
            time.sleep(1 / 60)


if __name__ == "__main__":
    main()
