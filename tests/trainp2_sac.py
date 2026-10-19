"""Train/evaluate the fixed-base door task with Stable-Baselines3 SAC.

uv run --no-sync python tests/trainp2_sac.py --timesteps 300000
uv run --no-sync python tests/trainp2_sac.py --evaluate runs/door_sac/sac_door.zip --seed 20000 --render

Use SB3's default MlpPolicy and algorithm settings unless explicitly overridden.
Reward-shaping gamma follows the actual SAC gamma, including during evaluation.
"""

from dataclasses import replace
import argparse
from functools import partial
import inspect
import json
from pathlib import Path
import time

import numpy as np
import torch
from stable_baselines3 import SAC
from stable_baselines3.common.callbacks import EvalCallback
from stable_baselines3.common.logger import configure
from stable_baselines3.common.monitor import Monitor
from stable_baselines3.common.vec_env import SubprocVecEnv

from rlinit.algorithmhand.device import resolve_device
from rlinit.algorithmhand.door_direct_env import TASK as DIRECT_TASK, DirectRewardDoorEnv
from rlinit.algorithmhand.door_env import DEFAULT_MODEL, DoorConfig, DoorEnv
from rlinit.algorithmhand.door_reference_env import (
    TASK as REFERENCE_TASK, ReferenceRewardConfig, ReferenceRewardDoorEnv,
)
from rlinit.algorithmhand.evaluation import final_eval_seed


FORMAT = "rlinit-door-sac-v1"
DEFAULT_MAX_STEPS = 1000
MONITOR_KEYS = ("distance", "is_success", "grasp_seen", "press_seen",
                "door_angle", "max_door_angle", "failure_reason")


def make_environment(model_path, config, *, render=False, reference_reward=None, direct_reward=False):
    if direct_reward and reference_reward is not None:
        raise ValueError("direct_reward and reference_reward cannot be combined")
    kwargs = {"config": config, "render_mode": "human" if render else None}
    if direct_reward:
        # New training uses the reward defaults defined in door_direct_env.py.
        return DirectRewardDoorEnv(model_path, **kwargs)
    if reference_reward is not None:
        return ReferenceRewardDoorEnv(model_path, reward_config=reference_reward, **kwargs)
    return DoorEnv(model_path, **kwargs)


def make_worker(model_path, config, monitor_path, reference_reward=None, direct_reward=False):
    """Construct physics state inside the spawned process, never in the parent."""
    torch.set_num_threads(1)
    return Monitor(make_environment(model_path, config, reference_reward=reference_reward, direct_reward=direct_reward),
                   str(monitor_path) if monitor_path is not None else None, info_keywords=MONITOR_KEYS)


def entropy_coefficient(value):
    """Accept a fixed nonnegative alpha, 'auto', or 'auto_<positive initial>'."""
    if value == "auto":
        return value
    if isinstance(value, str) and value.startswith("auto_"):
        initial = float(value.removeprefix("auto_"))
        if not np.isfinite(initial) or initial <= 0:
            raise ValueError("Automatic entropy coefficient needs a positive finite initial value")
        return value
    fixed = float(value)
    if not np.isfinite(fixed) or fixed < 0:
        raise ValueError("ent_coef must be nonnegative and finite, 'auto', or 'auto_<positive>'")
    return fixed


def sac_settings(**values):
    defaults = inspect.signature(SAC).parameters
    keys = ("learning_rate", "buffer_size", "learning_starts", "batch_size", "train_freq",
            "gradient_steps", "gamma", "tau", "ent_coef")
    unknown = values.keys() - set(keys)
    if unknown:
        raise ValueError(f"Unknown SAC settings: {sorted(unknown)}")
    overrides = {key: value for key, value in values.items() if value is not None}
    if "ent_coef" in overrides:
        overrides["ent_coef"] = entropy_coefficient(overrides["ent_coef"])
    actual = {key: overrides.get(key, defaults[key].default) for key in keys}
    for key in ("buffer_size", "batch_size", "train_freq"):
        if not isinstance(actual[key], int) or actual[key] < 1:
            raise ValueError(f"{key} must be a positive integer")
    if not isinstance(actual["learning_starts"], int) or actual["learning_starts"] < 0:
        raise ValueError("learning_starts must be a nonnegative integer")
    gradient_steps = actual["gradient_steps"]
    if not isinstance(gradient_steps, int) or (gradient_steps != -1 and gradient_steps < 1):
        raise ValueError("gradient_steps must be positive or -1 (one update per collected step)")
    for key in ("learning_rate", "gamma", "tau"):
        if not np.isfinite(actual[key]) or actual[key] <= 0:
            raise ValueError(f"{key} must be positive and finite")
    if actual["gamma"] > 1 or actual["tau"] > 1:
        raise ValueError("gamma and tau must be in (0, 1]")
    return overrides, actual


def load_agent(checkpoint, *, device="auto", render=False, model_path=None):
    agent = SAC.load(checkpoint, device=resolve_device(device))
    if getattr(agent, "door_task_format", None) != FORMAT:
        raise ValueError("Expected a door-task .zip checkpoint from trainp2_sac.py")
    specification = agent.door_environment
    direct_reward = specification["task"] == DIRECT_TASK
    if specification["task"] == REFERENCE_TASK:
        reference_reward = ReferenceRewardConfig(**specification["reward_config"])
    elif specification["task"] in ("fixed_base_door_v1", DIRECT_TASK):
        reference_reward = None
    else:
        raise ValueError(f"Unsupported SAC door task: {specification['task']}")
    path = model_path or specification["model_path"]
    config = DoorConfig(**specification["config"])
    if direct_reward:
        # Evaluation restores checkpoint values rather than new-training defaults.
        env = DirectRewardDoorEnv(
            path, config=config, render_mode="human" if render else None,
            position_reward=specification.get("position_reward", "exponential_reach"),
            orientation_reward=specification.get("orientation_reward", "distance_weighted_alignment"),
            distance_metric=specification.get("distance_metric", "grasp_target"),
            distance_weight=specification.get("distance_weight", 5.0),
            orientation_weight=specification.get("orientation_weight", 0.25),
        )
    else:
        env = make_environment(path, config, render=render, reference_reward=reference_reward)
    if (agent.observation_space != env.observation_space
            or agent.action_space != env.action_space or agent.gamma != env.config.gamma):
        env.close()
        raise ValueError("Checkpoint spaces/gamma do not match its environment")
    return agent, env


def evaluate(checkpoint, *, episodes=10, seed=20_000, device="auto", render=False, model_path=None):
    if episodes < 1:
        raise ValueError("episodes must be positive")
    agent, env = load_agent(checkpoint, device=device, render=render, model_path=model_path)
    records = []
    try:
        for episode in range(episodes):
            observation, _ = env.reset(seed=seed + episode)
            episode_return = 0.0
            length = 0
            while True:
                action, _ = agent.predict(observation, deterministic=True)
                observation, reward, terminated, truncated, info = env.step(action)
                episode_return += reward
                length += 1
                if terminated or truncated:
                    break
            records.append({
                "seed": seed + episode, "return": episode_return, "length": length,
                "is_success": info["is_success"], "grasp_seen": info["grasp_seen"],
                "press_seen": info["press_seen"], "final_door_deg": float(np.rad2deg(info["door_angle"])),
                "max_door_deg": float(np.rad2deg(info["max_door_angle"])),
                "final_distance_m": info["distance"], "failure_reason": info["failure_reason"],
                "truncated": bool(truncated),
            })
        return {
            "episodes": episodes, "first_seed": seed,
            "success_rate": float(np.mean([r["is_success"] for r in records])),
            "grasp_rate": float(np.mean([r["grasp_seen"] for r in records])),
            "press_rate": float(np.mean([r["press_seen"] for r in records])),
            "mean_return": float(np.mean([r["return"] for r in records])),
            "mean_final_door_deg": float(np.mean([r["final_door_deg"] for r in records])),
            "mean_max_door_deg": float(np.mean([r["max_door_deg"] for r in records])),
            "records": records,
        }
    finally:
        env.close()


def train(*, total_timesteps=300_000, output_dir="runs/door_sac", seed=0,
          model_path=DEFAULT_MODEL, env_config=None, learning_rate=None, buffer_size=None,
          learning_starts=None, batch_size=None, train_freq=None, gradient_steps=None,
          gamma=None, tau=None, ent_coef=None, eval_freq=10_000, eval_episodes=10,
          render=False, device="auto", n_envs=1, reference_reward=None, policy_kwargs=None,
          replay_buffer_initializer=None, save_replay_buffer=False, direct_reward=False):
    if direct_reward and reference_reward is not None:
        raise ValueError("direct_reward and reference_reward cannot be combined")
    if min(total_timesteps, eval_freq, eval_episodes) < 1:
        raise ValueError("timesteps/eval_freq/eval_episodes must be positive")
    if not isinstance(n_envs, int) or n_envs < 1:
        raise ValueError("n_envs must be a positive integer")
    if render and n_envs > 1:
        raise ValueError("Parallel training runs headless; use --evaluate CHECKPOINT --render to view the policy")
    overrides, actual = sac_settings(
        learning_rate=learning_rate, buffer_size=buffer_size, learning_starts=learning_starts,
        batch_size=batch_size, train_freq=train_freq, gradient_steps=gradient_steps,
        gamma=gamma, tau=tau, ent_coef=ent_coef,
    )
    if actual["buffer_size"] < n_envs:
        raise ValueError("buffer_size must hold at least one transition per environment")
    task_config = replace(env_config or DoorConfig(max_steps=DEFAULT_MAX_STEPS), gamma=actual["gamma"])
    selected_device = resolve_device(device)
    output = Path(output_dir).resolve()
    output.mkdir(parents=True, exist_ok=True)
    eval_seed = final_eval_seed(seed)
    env = eval_env = logger = None
    try:
        if n_envs == 1:
            env = Monitor(
                make_environment(model_path, task_config, render=render,
                                 reference_reward=reference_reward, direct_reward=direct_reward),
                str(output / "train"), info_keywords=MONITOR_KEYS,
            )
            specification = env.unwrapped.specification()
        else:
            workers = output / "workers"
            workers.mkdir(exist_ok=True)
            env = SubprocVecEnv([
                partial(make_worker, str(Path(model_path).resolve()), task_config,
                        workers / str(rank), reference_reward, direct_reward)
                for rank in range(n_envs)
            ], start_method="spawn")
            specification = env.env_method("specification", indices=0)[0]
        if n_envs == 1:
            eval_env = Monitor(make_environment(model_path, task_config,
                                                reference_reward=reference_reward, direct_reward=direct_reward))
            eval_env.reset(seed=seed + 10_000)
        else:
            eval_env = SubprocVecEnv([
                partial(make_worker, str(Path(model_path).resolve()), task_config, None, reference_reward, direct_reward)
            ], start_method="spawn")
            eval_env.seed(seed + 10_000)
        agent = SAC("MlpPolicy", env, seed=seed, device=selected_device, verbose=1,
                    policy_kwargs=policy_kwargs, **overrides)
        # These attributes are embedded in final and best checkpoints, allowing
        # evaluation without config.json and with a model path override on a new host.
        agent.door_task_format = FORMAT
        agent.door_environment = specification
        logger = configure(str(output / "logs"), ["stdout", "csv"])
        agent.set_logger(logger)
        replay_metadata = replay_buffer_initializer(agent) if replay_buffer_initializer is not None else None
        settings = dict(actual, net_arch=agent.policy.net_arch,
                        activation=agent.policy.activation_fn.__name__,
                        optimizer=agent.policy.optimizer_class.__name__,
                        n_critics=agent.critic.n_critics, target_entropy=agent.target_entropy)
        run_config = {
            "algorithm": "stable_baselines3.SAC", "environment": agent.door_environment,
            "agent": settings, "algorithm_overrides": overrides, "seed": seed,
            "final_eval_seed": eval_seed, "requested_timesteps": total_timesteps,
            "actual_timesteps": 0, "eval_freq": eval_freq, "eval_episodes": eval_episodes,
            "render": render, "device": str(selected_device),
            "n_envs": n_envs, "torch_threads": torch.get_num_threads(),
            "worker_seeds": [seed + rank for rank in range(n_envs)],
            "vector_backend": "SubprocVecEnv(spawn)" if n_envs > 1 else "DummyVecEnv",
        }
        # A callback is invoked once per vector step, which yields n_envs
        # transitions. Round up so evaluation never runs earlier than requested.
        callback_freq = max(1, (eval_freq + n_envs - 1) // n_envs)
        run_config["effective_eval_freq"] = callback_freq * n_envs
        if replay_metadata is not None:
            run_config["demonstrations"] = replay_metadata
        config_path = output / "config.json"
        config_path.write_text(json.dumps(run_config, indent=2) + "\n", encoding="utf-8")
        print(f"SAC n_envs={n_envs}, network={settings['net_arch']}, activation={settings['activation']}, "
              f"gamma={agent.gamma}, ent_coef={agent.ent_coef}; overrides={overrides}", flush=True)
        if total_timesteps <= agent.learning_starts:
            print("Budget does not exceed learning_starts: this run collects data without SAC updates.", flush=True)
        callback = EvalCallback(
            eval_env, best_model_save_path=str(output / "best"), log_path=str(output / "eval"),
            eval_freq=callback_freq, n_eval_episodes=eval_episodes, deterministic=True,
        )
        started = time.perf_counter()
        agent.learn(total_timesteps=total_timesteps, callback=callback)
        training_seconds = time.perf_counter() - started
        # Flush the last update's diagnostics even if no episode just ended.
        agent.logger.dump(step=agent.num_timesteps)
        checkpoint = output / "sac_door.zip"
        agent.save(checkpoint)
        if save_replay_buffer:
            replay_path = output / "replay_buffer.pkl"
            agent.save_replay_buffer(replay_path)
            run_config["replay_buffer_path"] = str(replay_path)
        run_config.update(actual_timesteps=agent.num_timesteps, gradient_updates=agent._n_updates,
                          training_seconds=training_seconds,
                          transitions_per_second=agent.num_timesteps / max(training_seconds, 1e-9))
        config_path.write_text(json.dumps(run_config, indent=2) + "\n", encoding="utf-8")
    finally:
        if logger is not None:
            logger.close()
        if env is not None:
            env.close()
        if eval_env is not None:
            eval_env.close()
    print(f"Final evaluation seed: {eval_seed}", flush=True)
    metrics = evaluate(checkpoint, episodes=eval_episodes, seed=eval_seed,
                       model_path=model_path, device=selected_device)
    (output / "metrics.json").write_text(json.dumps(metrics, indent=2) + "\n", encoding="utf-8")
    print(f"Saved: {checkpoint}; success={metrics['success_rate']:.1%}, grasp={metrics['grasp_rate']:.1%}")
    return checkpoint


def build_parser():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--timesteps", type=int, default=300_000)
    parser.add_argument("--output-dir", type=Path, default=Path("runs/door_sac"))
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--model", type=Path)
    parser.add_argument("--learning-rate", type=float)
    parser.add_argument("--buffer-size", type=int)
    parser.add_argument("--learning-starts", type=int, help="Random-action warmup steps (SB3 default 100).")
    parser.add_argument("--batch-size", type=int)
    parser.add_argument("--train-freq", type=int, help="Environment steps collected between updates.")
    parser.add_argument("--gradient-steps", type=int, help="Updates per collection; -1 matches collected steps.")
    parser.add_argument("--gamma", type=float, help="SAC discount factor (SB3 default 0.99).")
    parser.add_argument("--tau", type=float)
    parser.add_argument("--ent-coef", help="auto (default), auto_0.1, or a fixed nonnegative alpha.")
    parser.add_argument("--max-steps", type=int, default=DEFAULT_MAX_STEPS,
                        help="Maximum actions per episode (default 1000).")
    parser.add_argument("--action-scale", type=float, default=0.03)
    parser.add_argument("--initial-noise", type=float, default=0.02)
    parser.add_argument("--eval-freq", type=int, default=10_000)
    parser.add_argument("--episodes", type=int, default=10)
    parser.add_argument("--evaluate", type=Path)
    parser.add_argument("--render", action="store_true")
    parser.add_argument("--device", choices=("auto", "cpu", "cuda"), default="auto")
    return parser


def main():
    args = build_parser().parse_args()
    if args.evaluate:
        metrics = evaluate(args.evaluate, episodes=args.episodes, seed=args.seed,
                           device=args.device, render=args.render, model_path=args.model)
        print(json.dumps(metrics, indent=2))
    else:
        train(total_timesteps=args.timesteps, output_dir=args.output_dir, seed=args.seed,
              model_path=args.model or DEFAULT_MODEL,
              env_config=DoorConfig(max_steps=args.max_steps, action_scale=args.action_scale,
                                    initial_noise=args.initial_noise),
              learning_rate=args.learning_rate, buffer_size=args.buffer_size,
              learning_starts=args.learning_starts, batch_size=args.batch_size,
              train_freq=args.train_freq, gradient_steps=args.gradient_steps,
              gamma=args.gamma, tau=args.tau, ent_coef=args.ent_coef,
              eval_freq=args.eval_freq, eval_episodes=args.episodes, render=args.render, device=args.device)


if __name__ == "__main__":
    main()
