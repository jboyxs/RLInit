"""Render an unmodified deterministic policy rollout from noisy home."""
from pathlib import Path
import sys
import json
import hashlib

import mujoco
import numpy as np
from PIL import Image
import torch

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT / "tests"))
from trainp2_sac import load_agent


def main():
    torch.set_num_threads(1)
    output = Path(__file__).resolve().parent
    checkpoint = output / "sac_door.zip"
    agent, env = load_agent(checkpoint, device="cpu", model_path=ROOT / "models/door_scene.xml")
    frames = []
    seed = 70000
    try:
        obs, reset_info = env.reset(seed=seed)
        camera = mujoco.MjvCamera()
        camera.lookat[:] = env.data.geom_xpos[env.handle_geom] + np.array([-0.3, 0, -0.2])
        camera.distance, camera.azimuth, camera.elevation = 1.8, 45, -15
        with mujoco.Renderer(env.model, height=480, width=640) as renderer:
            steps = 0
            while True:
                renderer.update_scene(env.data, camera)
                if steps % 4 == 0:
                    frames.append(Image.fromarray(renderer.render().copy()))
                action, _ = agent.predict(obs, deterministic=True)
                obs, _, terminated, truncated, info = env.step(action)
                steps += 1
                if terminated or truncated:
                    renderer.update_scene(env.data, camera)
                    frames.append(Image.fromarray(renderer.render().copy()))
                    break
        frames[0].save(output / "home.png")
        frames[-1].save(output / "opened.png")
        frames[0].save(output / "opening.gif", save_all=True, append_images=frames[1:],
                       duration=round(env.dt * 4 * 1000), loop=0)
        metrics = dict(seed=seed, reset_origin=reset_info.get("reset_origin", "home"),
                       deterministic=True, exploration_noise=False, steps=steps,
                       simulation_seconds=steps * env.dt, success=info["is_success"],
                       failure=info["failure_reason"], final_door_rad=info["door_angle"],
                       final_handle_rad=info["handle_angle"],
                       grasped=info["grasped"], success_hold_seconds=info["success_hold_seconds"],
                       checkpoint_sha256=hashlib.sha256(checkpoint.read_bytes()).hexdigest())
        (output / "video_metrics.json").write_text(json.dumps(metrics, indent=2) + "\n")
        print(json.dumps(metrics, indent=2))
    finally:
        env.close()


if __name__ == "__main__":
    main()
