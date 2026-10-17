"""Prefill large-network SAC with physical demonstrations from verify_door_opening.

uv run --no-sync python tests/trainp2_sac_reference_large_demo.py --demo-episodes 10 --timesteps 300000
uv run --no-sync python tests/trainp2_sac_reference_large_demo.py --n-envs 4 --timesteps 300000
uv run --no-sync python tests/trainp2_sac_reference_large_demo.py --evaluate runs/door_sac_reference_large_demo/sac_door.zip --render

Only targets come from the scripted robot. Every saved transition is generated
by ReferenceRewardDoorEnv.step(), with the same actions/rewards as online SAC.
Demonstrations use ordinary uniform replay and can be overwritten as it fills.
"""

import json
import multiprocessing
from pathlib import Path
import time

import numpy as np

from rlinit.algorithmhand.door_env import DEFAULT_MODEL, DoorConfig
from rlinit.algorithmhand.door_reference_env import ReferenceRewardConfig, ReferenceRewardDoorEnv
from trainp2_sac import build_parser, evaluate
from trainp2_sac_reference_large import train as train_large
from verify_door_opening import CONTROL_DT, DoorTrial


def collect_demonstrations(*, model_path=DEFAULT_MODEL, env_config=None, reward_config=None,
                           episodes=10, seed=30_000, initial_noise=None):
    """Replay scripted targets in the actual fixed-base training MDP.

    Run DoorTrial once to get its target sequence, then use independent noisy
    resets for the demonstrations. Never transplant its different-sized qpos,
    observations, rewards or actuator commands into the training replay buffer.
    """
    if not isinstance(episodes, int) or episodes < 1:
        raise ValueError("demo_episodes must be a positive integer")
    config = env_config or DoorConfig()
    noise = config.initial_noise if initial_noise is None else initial_noise
    if not np.isfinite(noise) or noise < 0:
        raise ValueError("demo_initial_noise must be nonnegative and finite")
    env = ReferenceRewardDoorEnv(model_path, config=config, reward_config=reward_config)
    started = time.perf_counter()
    try:
        if not np.isclose(env.dt, CONTROL_DT, rtol=0, atol=1e-12):
            raise ValueError("Demonstrations require the script's 0.02 s control interval")
        targets = []

        def record_target(trial, _):
            targets.append((trial.last_target.copy(), float(trial.data.ctrl[trial.gripper]),
                            trial.rows[-1]["phase"]))

        print("Generating scripted robot target trajectory...", flush=True)
        reference_result = DoorTrial("normal", model_path=model_path).run(record_target)
        data = {key: [] for key in ("observations", "next_observations", "actions", "rewards",
                                    "terminated", "truncated", "episode_ids", "phases")}
        records = []
        for episode in range(episodes):
            obs, _ = env.reset(seed=seed + episode, options={"initial_noise": noise})
            episode_return = 0.0
            clipped_steps = 0
            for step, (target, grip, phase) in enumerate(targets):
                raw_action = np.r_[(target - env.arm_target) / config.action_scale,
                                   2 * grip / config.grip_max - 1]
                clipped_steps += int(np.any(np.abs(raw_action) > 1))
                action = np.clip(raw_action, -1, 1).astype(np.float32)
                next_obs, reward, terminated, truncated, info = env.step(action)
                collector_timeout = step == len(targets) - 1 and not (terminated or truncated)
                truncated = bool(truncated or collector_timeout)
                for key, value in (
                    ("observations", obs.copy()), ("next_observations", next_obs.copy()),
                    ("actions", action), ("rewards", reward), ("terminated", terminated),
                    ("truncated", truncated), ("episode_ids", episode), ("phases", phase),
                ):
                    data[key].append(value)
                episode_return += reward
                obs = next_obs
                if terminated or truncated:
                    break
            record = {
                "seed": seed + episode, "length": step + 1, "return": episode_return,
                "is_success": info["is_success"], "grasp_seen": info["grasp_seen"],
                "press_seen": info["press_seen"], "failure_reason": info["failure_reason"],
                "final_door_deg": float(np.rad2deg(info["door_angle"])),
                "terminated": bool(terminated), "truncated": truncated,
                "collector_time_limit": collector_timeout, "clipped_action_steps": clipped_steps,
            }
            records.append(record)
            print(f"Demo {episode + 1}/{episodes}: steps={record['length']}, "
                  f"success={record['is_success']}, door={record['final_door_deg']:.2f} deg, "
                  f"failure={record['failure_reason'] or 'none'}", flush=True)
        for key in data:
            dtype = (np.bool_ if key in ("terminated", "truncated") else
                     np.int64 if key == "episode_ids" else str if key == "phases" else np.float32)
            data[key] = np.asarray(data[key], dtype=dtype)
        report = {
            "source": "tests/verify_door_opening.py:DoorTrial.run",
            "method": "scripted joint targets -> clipped normalized actions -> training env.step",
            "environment": env.specification(), "episodes": episodes, "first_seed": seed,
            "initial_noise": noise, "transitions": len(data["rewards"]),
            "success_rate": float(np.mean([r["is_success"] for r in records])),
            "collection_seconds": time.perf_counter() - started,
            "reference_final_door_deg": reference_result["final_door_deg"], "records": records,
        }
        if not any(r["is_success"] for r in records):
            print("Warning: no successful demonstrations; see demonstrations.json for failures/timeouts.", flush=True)
        return data, report
    finally:
        env.close()


def prefill_replay_buffer(agent, data):
    """Group independent transitions into SB3's n_envs-wide replay rows.

    A final incomplete row repeats at most n_envs-1 entries from the beginning.
    No episode boundary is fabricated: each entry keeps its own next state and
    timeout mask. SAC stores actions scaled to [-1, 1].
    """
    replay = agent.replay_buffer
    count = len(data["rewards"])
    if not count:
        raise ValueError("Cannot prefill replay with an empty demonstration dataset")
    if replay.size():
        raise ValueError("Demonstration prefill expects an empty replay buffer")
    rows = (count + replay.n_envs - 1) // replay.n_envs
    if rows > replay.buffer_size:
        raise ValueError(f"Replay buffer needs at least {rows * replay.n_envs} transitions "
                         "to retain all demonstrations; increase --buffer-size or reduce --demo-episodes")
    for start in range(0, rows * replay.n_envs, replay.n_envs):
        indices = np.arange(start, start + replay.n_envs) % count
        terminated, truncated = data["terminated"][indices], data["truncated"][indices]
        replay.add(
            data["observations"][indices], data["next_observations"][indices],
            agent.policy.scale_action(data["actions"][indices]), data["rewards"][indices],
            terminated | truncated,
            [{"TimeLimit.truncated": bool(timeout and not terminal)}
             for terminal, timeout in zip(terminated, truncated)],
        )
    return {"inserted_transitions": rows * replay.n_envs,
            "padding_transitions": rows * replay.n_envs - count,
            "replay_rows": rows, "sampling": "uniform; online transitions can overwrite demonstrations"}


def train(*, demo_episodes=10, demo_seed=None, demo_initial_noise=None,
          output_dir="runs/door_sac_reference_large_demo", seed=0, **kwargs):
    output = Path(output_dir).resolve()

    def initialize(agent):
        # Use checkpoint metadata so gamma, model and reward settings cannot
        # diverge between demonstration collection and the online learner.
        specification = agent.door_environment
        data, report = collect_demonstrations(
            model_path=specification["model_path"], env_config=DoorConfig(**specification["config"]),
            reward_config=ReferenceRewardConfig(**specification["reward_config"]),
            episodes=demo_episodes, seed=seed + 30_000 if demo_seed is None else demo_seed,
            initial_noise=demo_initial_noise,
        )
        data_path, report_path = output / "demonstrations.npz", output / "demonstrations.json"
        np.savez_compressed(data_path, **data)
        report_path.write_text(json.dumps(report, indent=2) + "\n", encoding="utf-8")
        metadata = prefill_replay_buffer(agent, data)
        metadata.update(dataset_path=str(data_path), report_path=str(report_path),
                        transitions=report["transitions"], episodes=report["episodes"],
                        first_seed=report["first_seed"], initial_noise=report["initial_noise"],
                        success_rate=report["success_rate"])
        # Small provenance metadata is embedded in best and final checkpoints.
        agent.door_demonstrations = metadata
        print(f"Prefilled {metadata['inserted_transitions']} transitions; "
              f"demo success={report['success_rate']:.1%}", flush=True)
        return metadata

    return train_large(output_dir=output, seed=seed, replay_buffer_initializer=initialize,
                       save_replay_buffer=True, **kwargs)


def main():
    parser = build_parser()
    parser.description = __doc__
    parser.set_defaults(output_dir=Path("runs/door_sac_reference_large_demo"), gradient_steps=-1)
    parser.add_argument("--n-envs", type=int, default=1)
    parser.add_argument("--torch-threads", type=int, default=1)
    parser.add_argument("--reward-scale", type=float, default=1.0)
    parser.add_argument("--sparse", action="store_true")
    parser.add_argument("--demo-episodes", type=int, default=10, help="Scripted episodes collected before learning.")
    parser.add_argument("--demo-seed", type=int, help="First demonstration reset seed (default seed + 30000).")
    parser.add_argument("--demo-initial-noise", type=float,
                        help="Demonstration reset noise (default same as --initial-noise; 0 for home).")
    args = parser.parse_args()
    if args.evaluate:
        metrics = evaluate(args.evaluate, episodes=args.episodes, seed=args.seed,
                           device=args.device, render=args.render, model_path=args.model)
        print(json.dumps(metrics, indent=2))
    else:
        train(total_timesteps=args.timesteps, output_dir=args.output_dir, seed=args.seed,
              n_envs=args.n_envs, torch_threads=args.torch_threads,
              model_path=args.model or DEFAULT_MODEL,
              env_config=DoorConfig(max_steps=args.max_steps, action_scale=args.action_scale,
                                    initial_noise=args.initial_noise),
              reward_config=ReferenceRewardConfig(reward_shaping=not args.sparse, reward_scale=args.reward_scale),
              demo_episodes=args.demo_episodes, demo_seed=args.demo_seed, demo_initial_noise=args.demo_initial_noise,
              learning_rate=args.learning_rate, buffer_size=args.buffer_size,
              learning_starts=args.learning_starts, batch_size=args.batch_size,
              train_freq=args.train_freq, gradient_steps=args.gradient_steps,
              gamma=args.gamma, tau=args.tau, ent_coef=args.ent_coef,
              eval_freq=args.eval_freq, eval_episodes=args.episodes, render=args.render, device=args.device)


if __name__ == "__main__":
    multiprocessing.freeze_support()
    main()
