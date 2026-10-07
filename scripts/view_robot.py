"""Load the teaching MJCF, or run a deterministic headless smoke check."""

import argparse
from pathlib import Path
import time

import mujoco
import numpy as np


MODEL = Path(__file__).resolve().parents[1] / "models" / "mobile_manipulator.xml"


def set_controls(model: mujoco.MjModel, data: mujoco.MjData, demo: bool) -> None:
    # Named lookup avoids confusing actuator indices with qpos indices.
    for i in range(1, 7):
        target = 0.0
        if demo and i == 2:
            target = 0.25 * np.sin(0.8 * data.time)
        if demo and i == 3:
            target = -0.35 * np.sin(0.8 * data.time)
        data.ctrl[model.actuator(f"arm_a{i}").id] = target
    opening = 0.0175 * (1 + np.cos(data.time)) if demo else 0.02
    data.ctrl[model.actuator("gripper").id] = opening
    # Keep the base parked initially. Positive wheel speeds drive toward +x.
    for wheel in ("fl", "fr", "rl", "rr"):
        data.ctrl[model.actuator(f"drive_{wheel}").id] = 0


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--model", type=Path, default=MODEL)
    parser.add_argument("--headless", action="store_true")
    parser.add_argument("--seconds", type=float, default=10)
    parser.add_argument("--demo", action="store_true")
    args = parser.parse_args()
    if not np.isfinite(args.seconds) or args.seconds <= 0:
        parser.error("--seconds must be positive and finite")

    model = mujoco.MjModel.from_xml_path(str(args.model.resolve()))
    data = mujoco.MjData(model)
    for finger in ("left", "right"):
        data.joint(f"finger_{finger}_joint").qpos[0] = 0.02
    set_controls(model, data, args.demo)
    mujoco.mj_forward(model, data)
    print(f"MuJoCo {mujoco.__version__}: nq={model.nq}, nv={model.nv}, nu={model.nu}")

    def step() -> None:
        set_controls(model, data, args.demo)
        mujoco.mj_step(model, data)
        if not np.isfinite(data.qpos).all() or not np.isfinite(data.qvel).all():
            raise RuntimeError("Non-finite simulation state")

    if args.headless:
        for _ in range(int(np.ceil(args.seconds / model.opt.timestep))):
            step()
        warnings = {mujoco.mjtWarning(i).name: int(w.number)
                    for i, w in enumerate(data.warning) if w.number}
        if warnings:
            raise RuntimeError(f"MuJoCo warnings: {warnings}")
        error = abs(data.joint("finger_left_joint").qpos[0]
                    - data.joint("finger_right_joint").qpos[0])
        base_height = data.body("base").xpos[2]
        print(f"time={data.time:.3f}s, contacts={data.ncon}, "
              f"base_z={base_height:.4f}m, finger_sync_error={error:.6f}m")
    else:
        from mujoco import viewer as mj_viewer

        with mj_viewer.launch_passive(model, data) as viewer:
            viewer.cam.lookat[:] = [0, 0, 0.65]
            viewer.cam.distance = 2.8
            viewer.cam.azimuth = 135
            viewer.cam.elevation = -20
            while viewer.is_running():
                start = time.perf_counter()
                step()
                viewer.sync()
                time.sleep(max(0.0, model.opt.timestep - (time.perf_counter() - start)))


if __name__ == "__main__":
    main()
