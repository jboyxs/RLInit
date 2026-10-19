"""Record independent SAC rollouts; report grasp geometry and plot one trace.

uv run --no-sync python tests/inspect_door_sac_direct.py CHECKPOINT --output-dir RUN/inspection
"""

import argparse
import csv
import json
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import torch
import mujoco

from trainp2_sac import load_agent


def inspect(checkpoint, *, output_dir, episodes=5, seed=20_000, stochastic=False, frames=False,
            full_task=False, probe_forces=False, force_input_clip=None):
    if probe_forces and stochastic:
        raise ValueError("force observation probing uses deterministic policy comparisons")
    if force_input_clip is not None and force_input_clip <= 0:
        raise ValueError("force_input_clip must be positive")
    torch.set_num_threads(1)
    output = Path(output_dir)
    output.mkdir(parents=True, exist_ok=True)
    agent, env = load_agent(checkpoint, device="cpu", full_task=full_task)
    records = []
    keys = ("distance", "upper_finger_distance_m", "lower_finger_distance_m", "alignment",
            "center_distance_m", "finger_imbalance_m", "finger_gap_m", "grasp_readiness",
            "left_contact_N", "right_contact_N", "handle_angle", "door_angle",
            "gripper_action", "grasped", "penetration_m")
    try:
        for episode in range(episodes):
            obs, reset_info = env.reset(seed=seed + episode)
            agent.set_random_seed(seed + episode)
            rows = []
            initial_qpos = env.data.qpos.copy()
            nearest_qpos = initial_qpos.copy()
            first_contact_qpos = None
            minimum_distance = float("inf")
            grasp_streak = longest_grasp = 0
            while True:
                policy_obs = obs.copy()
                if force_input_clip is not None:
                    policy_obs[62:64] = np.clip(policy_obs[62:64], 0, force_input_clip)
                action, _ = agent.predict(policy_obs, deterministic=not stochastic)
                if probe_forces:
                    bounded = obs.copy()
                    bounded[62:64] = np.clip(bounded[62:64], 0, 1)
                    alternate, _ = agent.predict(bounded, deterministic=True)
                    force_action_delta = float(np.linalg.norm(action[:7] - alternate[:7]))
                obs, reward, terminated, truncated, info = env.step(action)
                rows.append(dict(step=len(rows), reward=reward,
                                 **{f"policy_action_{i}": float(value) for i, value in enumerate(action)},
                                 **{f"joint_position_{i}": float(value) for i, value in enumerate(env.data.qpos[env.qadr])},
                                 **{f"joint_target_{i}": float(value) for i, value in enumerate(env.arm_target)},
                                 target_error_rad=float(np.max(np.abs(env.arm_target - env.data.qpos[env.qadr]))),
                                 **{k: info.get(k, float("nan")) for k in keys}))
                if probe_forces:
                    rows[-1]["force_action_delta"] = force_action_delta
                if info["distance"] < minimum_distance:
                    minimum_distance = info["distance"]
                    nearest_qpos = env.data.qpos.copy()
                grasp_streak = grasp_streak + 1 if info["grasped"] else 0
                longest_grasp = max(longest_grasp, grasp_streak)
                if info["grasped"] and first_contact_qpos is None:
                    first_contact_qpos = env.data.qpos.copy()
                if terminated or truncated:
                    break
            with (output / f"seed_{seed + episode}.csv").open("w", newline="", encoding="utf-8") as stream:
                writer = csv.DictWriter(stream, fieldnames=rows[0].keys())
                writer.writeheader()
                writer.writerows(rows)
            near = min(rows, key=lambda r: r["distance"])
            contacts = []
            for index, contact in enumerate(env.data.contact):
                force = np.zeros(6)
                mujoco.mj_contactForce(env.model, env.data, index, force)
                contacts.append(dict(geom1=env.model.geom(int(contact.geom1)).name,
                                     geom2=env.model.geom(int(contact.geom2)).name,
                                     body1=env.model.body(int(env.model.geom_bodyid[int(contact.geom1)])).name,
                                     body2=env.model.body(int(env.model.geom_bodyid[int(contact.geom2)])).name,
                                     penetration_m=max(0.0, float(-contact.dist)),
                                     normal_force_N=float(force[0])))
            contacts.sort(key=lambda item: item["normal_force_N"], reverse=True)
            records.append(dict(seed=seed + episode, steps=len(rows),
                                reset_origin=reset_info.get("reset_origin", "home"),
                                success=info["is_success"], grasp=info["grasp_seen"],
                                stage_success=info.get("stage_success", info["is_success"]),
                                press=info["press_seen"], failure=info["failure_reason"],
                                return_=sum(r["reward"] for r in rows),
                                minimum_distance=near["distance"], nearest_geometry=near,
                                max_handle_deg=float(np.rad2deg(max(r["handle_angle"] for r in rows))),
                                max_door_deg=float(np.rad2deg(max(r["door_angle"] for r in rows))),
                                grasp_steps=sum(r["grasped"] for r in rows),
                                terminal_contacts=contacts,
                                longest_grasp_seconds=longest_grasp * env.dt))
            if episode == 0:
                fig, axes = plt.subplots(4, 1, figsize=(10, 10), sharex=True)
                for key in ("upper_finger_distance_m", "lower_finger_distance_m", "center_distance_m"):
                    axes[0].plot([r[key] for r in rows], label=key)
                for key in ("alignment", "grasp_readiness", "gripper_action"):
                    axes[1].plot([r[key] for r in rows], label=key)
                for key in ("handle_angle", "door_angle"):
                    axes[2].plot(np.rad2deg([r[key] for r in rows]), label=key)
                for key in ("left_contact_N", "right_contact_N"):
                    axes[3].plot([r[key] for r in rows], label=key)
                for ax, label in zip(axes, ("Distance (m)", "Alignment / action", "Angle (deg)", "Force (N)")):
                    ax.set_ylabel(label)
                    ax.legend(loc="upper right")
                    ax.grid(alpha=0.25)
                axes[-1].set_xlabel("Policy step")
                fig.suptitle(f"SAC rollout seed {seed}, success={info['is_success']}")
                fig.tight_layout()
                fig.savefig(output / "trace.png")
                plt.close(fig)
                if frames:
                    camera = mujoco.MjvCamera()
                    camera.lookat[:] = env.data.geom_xpos[env.handle_geom]
                    camera.distance, camera.azimuth, camera.elevation = 0.8, 45, -15
                    poses = {"initial": initial_qpos, "nearest": nearest_qpos,
                             "final": env.data.qpos.copy()}
                    if first_contact_qpos is not None:
                        poses["first_contact"] = first_contact_qpos
                    with mujoco.Renderer(env.model, height=480, width=640) as renderer:
                        for name, pose in poses.items():
                            env.data.qpos[:] = pose
                            mujoco.mj_forward(env.model, env.data)
                            renderer.update_scene(env.data, camera)
                            plt.imsave(output / f"{name}.png", renderer.render())
    finally:
        env.close()
    summary = dict(checkpoint=str(Path(checkpoint).resolve()), deterministic=not stochastic,
                   training_timesteps=agent.num_timesteps,
                   force_input_clip=force_input_clip,
                   full_task_override=full_task,
                   reset_origins=sorted({r["reset_origin"] for r in records}),
                   episodes=episodes, success_rate=float(np.mean([r["success"] for r in records])),
                   training_stage=getattr(getattr(env, "training_config", None), "training_stage", "full"),
                   stage_success_rate=float(np.mean([r["stage_success"] for r in records])),
                   grasp_rate=float(np.mean([r["grasp"] for r in records])),
                   press_rate=float(np.mean([r["press"] for r in records])), records=records)
    (output / "summary.json").write_text(json.dumps(summary, indent=2) + "\n", encoding="utf-8")
    print(json.dumps(summary, indent=2), flush=True)
    return summary


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("checkpoint", type=Path)
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--episodes", type=int, default=5)
    parser.add_argument("--seed", type=int, default=20_000)
    parser.add_argument("--stochastic", action="store_true")
    parser.add_argument("--frames", action="store_true", help="Render initial, nearest and final robot poses.")
    parser.add_argument("--full-task", action="store_true")
    parser.add_argument("--probe-forces", action="store_true", help="Read-only comparison with bounded force inputs; actual actions stay unchanged.")
    parser.add_argument("--force-input-clip", type=float, help="Diagnostic policy input saturation; does not change physics or task criteria.")
    args = parser.parse_args()
    inspect(args.checkpoint, output_dir=args.output_dir, episodes=args.episodes,
            seed=args.seed, stochastic=args.stochastic, frames=args.frames,
            full_task=args.full_task, probe_forces=args.probe_forces,
            force_input_clip=args.force_input_clip)
