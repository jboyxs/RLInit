"""Preview the downloaded door model without advancing physics.

Run from the repository root:
    uv run --no-sync python tests/view_door.py
    uv run --no-sync python tests/view_door.py --check
"""

import argparse
from pathlib import Path
import time
import xml.etree.ElementTree as ET

import mujoco
from mujoco import viewer as mj_viewer


MODEL = Path(__file__).resolve().parents[1] / "models" / "objects" / "door_lock.xml"


def load_model(path: Path) -> mujoco.MjModel:
    path = path.resolve()
    root = ET.parse(path).getroot()
    # The robosuite door asset uses radians, but standalone MJCF defaults to degrees.
    compiler = root.find("compiler")
    if compiler is None:
        compiler = ET.SubElement(root, "compiler")
    if "angle" not in compiler.attrib:
        compiler.set("angle", "radian")

    asset = root.find("asset")
    if asset is not None:
        missing = set()
        for texture in list(asset.findall("texture")):
            filename = texture.get("file")
            if filename is None:
                continue
            texture_path = path.parent / compiler.get("texturedir", "") / filename
            if texture_path.is_file():
                texture.set("file", str(texture_path.resolve()))
            else:
                missing.add(texture.get("name"))
                asset.remove(texture)
                print(f"Missing texture; using solid color: {texture_path}")

        colors = {
            "MatMetal": "0.65 0.55 0.35 1",
            "MatDarkWood": "0.35 0.18 0.08 1",
            "MatLightWood": "0.65 0.45 0.25 1",
        }
        for material in asset.findall("material"):
            if material.get("texture") in missing:
                material.attrib.pop("texture")
                material.set("rgba", colors.get(material.get("name"), "0.6 0.6 0.6 1"))

    worldbody = root.find("worldbody")
    if worldbody is not None:
        ET.SubElement(
            worldbody, "light", pos="-1 1 2", dir="1 -1 -2",
            directional="true", diffuse="0.8 0.8 0.8", ambient="0.3 0.3 0.3",
        )

    return mujoco.MjModel.from_xml_string(ET.tostring(root, encoding="unicode"))


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--model", type=Path, default=MODEL)
    parser.add_argument("--check", action="store_true", help="Load the model without opening a window.")
    args = parser.parse_args()

    model = load_model(args.model)
    data = mujoco.MjData(model)
    mujoco.mj_forward(model, data)
    print(f"Loaded {args.model.resolve()} ({model.njnt} joints, {model.ngeom} geoms)")
    if args.check:
        return

    with mj_viewer.launch_passive(model, data) as viewer:
        panel_id = mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_GEOM, "panel")
        if panel_id >= 0:
            viewer.cam.lookat[:] = data.geom_xpos[panel_id]
        viewer.cam.distance = 1.5
        viewer.cam.azimuth = 135
        viewer.cam.elevation = -15
        while viewer.is_running():
            viewer.sync()
            time.sleep(1 / 60)


if __name__ == "__main__":
    main()
