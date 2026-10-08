"""Display an MJCF model in its initial pose without advancing physics."""

import argparse
from pathlib import Path
import time

import mujoco
from mujoco import viewer as mj_viewer


MODEL = Path(__file__).resolve().parents[1] / "models" / "mobile_manipulator.xml"


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--model", type=Path, default=MODEL)
    args = parser.parse_args()

    model = mujoco.MjModel.from_xml_path(str(args.model.resolve()))
    data = mujoco.MjData(model)
    # Compute geometry transforms for rendering the XML's initial pose.
    mujoco.mj_forward(model, data)

    with mj_viewer.launch_passive(model, data) as viewer:
        viewer.cam.lookat[:] = [0, 0, 0.65]
        viewer.cam.distance = 2.8
        viewer.cam.azimuth = 135
        viewer.cam.elevation = -20
        while viewer.is_running():
            viewer.sync()
            time.sleep(1 / 60)


if __name__ == "__main__":
    main()
