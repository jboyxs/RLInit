"""Record one complete deterministic rollout, from home reset to task success.

uv run --no-sync python assets/door_sac_2026-10-21/record_gif.py
The learned checkpoint is local; no scripted actions or modified door physics.
"""
import argparse
import hashlib
import json
from pathlib import Path
import sys

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT))


def main():
    from train.sac_staged.evaluation import load_agent
    import mujoco
    import numpy as np
    from PIL import Image, ImageDraw, ImageFont
    import torch

    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--checkpoint", type=Path, default=ROOT / "runs/door_sac_staged/best/best_model.zip")
    parser.add_argument("--seed", type=int, default=20000)
    parser.add_argument("--output", type=Path, default=Path(__file__).with_name("opening.gif"))
    args = parser.parse_args()
    torch.set_num_threads(1)
    agent, env = load_agent(args.checkpoint, device="cpu")
    renderer = None
    frames = []
    try:
        obs, info = env.reset(seed=args.seed)
        env.model.vis.global_.offwidth = 800
        env.model.vis.global_.offheight = 600
        renderer = mujoco.Renderer(env.model, height=600, width=800)
        camera = mujoco.MjvCamera()
        mujoco.mjv_defaultCamera(camera)
        camera.lookat[:] = [0.45, 0., 0.75]
        camera.distance, camera.azimuth, camera.elevation = 2.1, 45, -20
        font = ImageFont.load_default(size=19)

        def capture(step, info):
            renderer.update_scene(env.data, camera=camera)
            frame = Image.fromarray(renderer.render().copy())
            draw = ImageDraw.Draw(frame)
            draw.rectangle((0, 0, 800, 58), fill=(22, 27, 34))
            draw.text((14, 8), f"SAC | t={step * env.dt:.2f}s | seed={args.seed}", font=font, fill="white")
            status = "SUCCESS" if info["is_success"] else "OPENING" if info["press_seen"] else "APPROACH / GRASP / PRESS"
            draw.text((14, 33), f"{status} | handle={np.rad2deg(info['handle_angle']):.1f} deg | door={np.rad2deg(info['door_angle']):.1f} deg",
                      font=font, fill=(137, 220, 170))
            frames.append(frame)

        capture(0, info)
        total = 0.
        for step in range(1, env.config.max_steps + 1):
            action, _ = agent.predict(obs, deterministic=True)
            obs, reward, done, timeout, info = env.step(action)
            total += reward
            if step % 2 == 0 or done or timeout:
                capture(step, info)
            if done or timeout:
                break
        if not info["is_success"]:
            raise RuntimeError(f"Episode did not succeed: seed={args.seed}, failure={info['failure_reason']}")
        args.output.parent.mkdir(parents=True, exist_ok=True)
        durations = [40] * len(frames)
        if step % 2:
            durations[-2] = 20
        durations[-1] = 1000  # Pause on success; no extra simulation or actions.
        palette = frames[0].quantize(colors=96, dither=Image.Dither.NONE)
        encoded = [frame.quantize(palette=palette, dither=Image.Dither.NONE) for frame in frames]
        encoded[0].save(args.output, save_all=True, append_images=encoded[1:], duration=durations,
                       loop=0, optimize=True, disposal=2)
        metadata = dict(checkpoint="runs/door_sac_staged/best/best_model.zip",
            checkpoint_sha256=hashlib.sha256(args.checkpoint.read_bytes()).hexdigest(),
            trained_timesteps=agent.num_timesteps, seed=args.seed, deterministic=True,
            reset_origin="home", success=bool(info["is_success"]), steps=step,
            simulation_seconds=step * env.dt, frames=len(frames), frame_interval_seconds=2 * env.dt,
            final_pause_seconds=1., return_=total, grasp=bool(info["grasp_seen"]), press=bool(info["press_seen"]),
            final_handle_deg=float(np.rad2deg(info["handle_angle"])),
            final_door_deg=float(np.rad2deg(info["door_angle"])),
            success_hold_seconds=info["success_hold_seconds"], failure=info["failure_reason"],
            gif_sha256=hashlib.sha256(args.output.read_bytes()).hexdigest())
        args.output.with_suffix(".json").write_text(json.dumps(metadata, indent=2) + "\n", encoding="utf-8")
        preview = ROOT / "runs/door_sac_staged/verification"
        preview.mkdir(parents=True, exist_ok=True)
        for name, index in (("start", 0), ("middle", len(frames) // 2), ("final", -1)):
            frames[index].save(preview / f"{name}.png")
        print(json.dumps(metadata, indent=2))
    finally:
        if renderer is not None:
            renderer.close()
        env.close()


if __name__ == "__main__":
    main()
