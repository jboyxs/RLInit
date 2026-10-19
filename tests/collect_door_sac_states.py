"""Collect real simulator states reached by a trained joint-action SAC actor.

No optimizer, expert action, inverse kinematics, or synthetic pose is used.
The state bank is for an optional training curriculum, never final evaluation.
"""

import argparse
import json
from pathlib import Path

import numpy as np
import torch

from rlinit.algorithmhand.door_sac_grasp_env import TASK
from rlinit.algorithmhand.door_sac_state_bank import capture_state, save_bank
from trainp2_sac import load_agent


def collect(checkpoint, output, *, episodes=30, seed=50_000, per_episode=25, phase="approach"):
    if min(episodes, per_episode) < 1:
        raise ValueError("episodes and per_episode must be positive")
    if phase not in ("approach", "press"):
        raise ValueError("phase must be approach or press")
    torch.set_num_threads(1)
    agent, env = load_agent(checkpoint, device="cpu")
    states, records = [], []
    try:
        if env.specification()["task"] != TASK:
            raise ValueError("Only direct-joint SAC rollouts may populate this state bank")
        for episode in range(episodes):
            observation, _ = env.reset(seed=seed + episode)
            candidates = []
            while True:
                action, _ = agent.predict(observation, deterministic=True)
                observation, _, terminated, truncated, info = env.step(action)
                # Select actual visits before pressing/opening, not an ideal
                # pose or action supplied by a conventional controller.
                approach = (info["center_distance_m"] < 0.03
                        and info["alignment"] >= 0.9 and abs(info["handle_angle"]) < 0.2
                        and abs(info["door_angle"]) < 0.02 and info["penetration_m"] < 0.001
                        and info["bad_contact_N"] < 10
                        and np.max(np.abs(env.arm_target - env.data.qpos[env.qadr])) < 0.08)
                pressed = (info["grasped"] and 1.1 <= info["handle_angle"] <= 1.35
                           and info["alignment"] >= 0.85 and info["center_distance_m"] < 0.04
                           and abs(info["door_angle"]) < 0.05 and info["penetration_m"] < 0.001
                           and info["bad_contact_N"] < 10
                           and np.max(np.abs(env.arm_target - env.data.qpos[env.qadr])) < 0.12)
                if not terminated and not truncated and (approach if phase == "approach" else pressed):
                    candidates.append(capture_state(env))
                if terminated or truncated:
                    break
            selected = np.linspace(0, len(candidates) - 1, min(per_episode, len(candidates)), dtype=int)
            states.extend(candidates[index] for index in selected)
            records.append(dict(seed=seed + episode, eligible_states=len(candidates), saved_states=len(selected)))
        metadata = save_bank(output, env, states, checkpoint=checkpoint,
                             collection=dict(phase=phase, episodes=episodes, first_seed=seed))
        report = dict(states=len(states), metadata=metadata, episodes=records)
        Path(output).with_suffix(".json").write_text(json.dumps(report, indent=2) + "\n", encoding="utf-8")
        print(json.dumps(dict(states=len(states), episodes=episodes, output=str(Path(output).resolve())), indent=2))
        return report
    finally:
        env.close()


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("checkpoint", type=Path)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--episodes", type=int, default=30)
    parser.add_argument("--seed", type=int, default=50_000)
    parser.add_argument("--per-episode", type=int, default=25)
    parser.add_argument("--phase", choices=("approach", "press"), default="approach")
    arguments = parser.parse_args()
    collect(arguments.checkpoint, arguments.output, episodes=arguments.episodes,
            seed=arguments.seed, per_episode=arguments.per_episode, phase=arguments.phase)
