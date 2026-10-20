"""Trace a learned policy with no action modification or controller.

The evaluator recomputes the success sequence from per-step physical contacts
and coordinates, and saves evidence independently from training metrics.
"""

import argparse
import hashlib
import json
from pathlib import Path

import numpy as np
import torch

from train_sac_learn import load_agent


def inspect(checkpoint, output, *, episodes=5, seed=30000, device="cpu", render=False, loader=load_agent):
    torch.set_num_threads(1)
    agent, env = loader(checkpoint, device=device, render=render)
    records = []
    try:
        for offset in range(episodes):
            obs, _ = env.reset(seed=seed + offset)
            trajectory = []
            total = 0.
            grasp_steps = longest = door_hold_steps = 0
            pressed_step = None
            for step in range(1, env.config.max_steps + 1):
                action, _ = agent.predict(obs, deterministic=True)
                with torch.no_grad():
                    tensor, _ = agent.policy.obs_to_tensor(obs)
                    action_tensor = torch.as_tensor(action[None], device=agent.device)
                    critic_q = min(float(q.item()) for q in agent.critic(tensor, action_tensor))
                    _, latent_log_std, _ = agent.actor.get_action_dist_params(tensor)
                obs, reward, done, timeout, info = env.step(action)
                total += reward
                grasp_steps = grasp_steps + 1 if info["grasped"] and not info["failure_reason"] else 0
                longest = max(longest, grasp_steps)
                if (grasp_steps * env.dt + 1e-9 >= env.reward_config.stable_grasp_seconds
                        and info["handle_angle"] >= env.reward_config.press_target_angle):
                    pressed_step = pressed_step or step
                if (pressed_step is not None and info["grasped"] and not info["failure_reason"]
                        and info["door_angle"] >= env.config.success_angle):
                    door_hold_steps += 1
                else:
                    door_hold_steps = 0
                trajectory.append(dict(step=step, action=action.tolist(), reward=reward, critic_q=critic_q,
                    latent_log_std=latent_log_std.cpu().numpy().ravel().tolist(),
                    joint_tracking_error=(env.arm_target - env.data.qpos[env.qadr]).tolist(),
                    components=info["reward_components"], grasp=info["grasped"],
                    handle_deg=float(np.rad2deg(info["handle_angle"])),
                    door_deg=float(np.rad2deg(info["door_angle"])),
                    left_N=info["left_contact_N"], right_N=info["right_contact_N"],
                    penetration_m=info["penetration_m"], bad_contact_N=info["bad_contact_N"],
                    alignment=info["alignment"], center_m=info["center_distance_m"],
                    failure=info["failure_reason"]))
                if done or timeout:
                    break
            verified = (door_hold_steps * env.dt + 1e-9 >= env.config.success_hold_seconds
                        and not info["failure_reason"])
            if verified != info["is_success"]:
                raise RuntimeError("Environment and independent trace success disagree")
            finite_return = 0.
            for sample in reversed(trajectory):
                finite_return = sample["reward"] + agent.gamma * finite_return
                sample["finite_discounted_return"] = finite_return
            records.append(dict(seed=seed + offset, success=bool(verified), steps=step,
                return_=total, pressed_step=pressed_step, longest_grasp_steps=longest,
                initial_critic_q=trajectory[0]["critic_q"],
                finite_discounted_return=trajectory[0]["finite_discounted_return"],
                final_door_hold_seconds=door_hold_steps * env.dt, failure=info["failure_reason"],
                max_handle_deg=max(t["handle_deg"] for t in trajectory),
                max_door_deg=max(t["door_deg"] for t in trajectory), trajectory=trajectory))
    finally:
        env.close()
    result = dict(checkpoint=str(Path(checkpoint).resolve()),
        task=agent.door_environment["task"],
        trained_timesteps=agent.num_timesteps,
        checkpoint_sha256=hashlib.sha256(Path(checkpoint).read_bytes()).hexdigest(),
        episodes=episodes, first_seed=seed, deterministic=True, reset_origin="home",
        success_rate=float(np.mean([r["success"] for r in records])),
        critic_comparison="Soft stochastic SAC critic versus finite deterministic rollout return; not an equality test",
        records=records)
    destination = Path(output)
    destination.parent.mkdir(parents=True, exist_ok=True)
    destination.write_text(json.dumps(result, indent=2) + "\n", encoding="utf-8")
    print(json.dumps({k: v for k, v in result.items() if k != "records"}, indent=2))
    return result


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("checkpoint", type=Path)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--episodes", type=int, default=5)
    parser.add_argument("--seed", type=int, default=30000)
    parser.add_argument("--device", choices=("cpu", "cuda"), default="cpu")
    parser.add_argument("--render", action="store_true")
    args = parser.parse_args()
    if args.episodes < 1:
        parser.error("--episodes must be positive")
    inspect(args.checkpoint, args.output, episodes=args.episodes, seed=args.seed,
            device=args.device, render=args.render)
