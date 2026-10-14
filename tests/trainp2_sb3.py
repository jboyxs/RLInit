"""Train/evaluate the door MDP with Stable-Baselines3's default PPO policy.

uv run --no-sync python tests/trainp2_sb3.py --timesteps 300000
uv run --no-sync python tests/trainp2_sb3.py --evaluate runs/door_sb3/ppo_door.zip --seed 20000 --render

Network, initialization, optimizer and PPO settings use installed SB3 defaults.
Only explicitly provided CLI overrides change algorithm hyperparameters.
The environment's reward-shaping gamma always matches PPO's actual gamma.
"""

from dataclasses import replace
import argparse
import inspect
import json
from pathlib import Path

import numpy as np
from stable_baselines3 import PPO
from stable_baselines3.common.callbacks import EvalCallback
from stable_baselines3.common.logger import configure
from stable_baselines3.common.monitor import Monitor

from rlinit.algorithmhand.device import resolve_device
from rlinit.algorithmhand.door_env import DEFAULT_MODEL, DoorConfig, DoorEnv
from rlinit.algorithmhand.evaluation import final_eval_seed


FORMAT = "rlinit-door-sb3-v1"


def ppo_settings(*, n_steps=None, batch_size=None, epochs=None, gamma=None):
    overrides = {key: value for key, value in {
        "n_steps": n_steps, "batch_size": batch_size, "n_epochs": epochs, "gamma": gamma,
    }.items() if value is not None}
    defaults = inspect.signature(PPO).parameters
    actual = {key: overrides.get(key, defaults[key].default)
              for key in ("n_steps", "batch_size", "n_epochs", "gamma")}
    if actual["n_steps"] < 2 or actual["batch_size"] < 2 or actual["n_epochs"] < 1:
        raise ValueError("Require n_steps/batch_size >= 2 and epochs >= 1")
    if not 0 < actual["gamma"] <= 1:
        raise ValueError("gamma must be in (0, 1]")
    return overrides, actual


def load_agent(checkpoint, *, device="auto", render=False, model_path=None):
    agent = PPO.load(checkpoint, device=resolve_device(device))
    if getattr(agent, "door_task_format", None) != FORMAT:
        raise ValueError("Expected a door-task .zip checkpoint from trainp2_sb3.py")
    specification = agent.door_environment
    env = DoorEnv(model_path or specification["model_path"], config=DoorConfig(**specification["config"]),
                  render_mode="human" if render else None)
    if (agent.observation_space.shape != env.observation_space.shape
            or agent.action_space.shape != env.action_space.shape
            or agent.gamma != env.config.gamma):
        env.close()
        raise ValueError("Checkpoint dimensions/gamma do not match its environment")
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


def train(*, total_timesteps=300_000, output_dir="runs/door_sb3", seed=0,
          model_path=DEFAULT_MODEL, env_config=None, n_steps=None, batch_size=None,
          epochs=None, gamma=None, eval_freq=10_000, eval_episodes=10,
          render=False, device="auto"):
    if total_timesteps < 1 or eval_freq < 1 or eval_episodes < 1:
        raise ValueError("timesteps/eval_freq/eval_episodes must be positive")
    overrides, actual = ppo_settings(n_steps=n_steps, batch_size=batch_size, epochs=epochs, gamma=gamma)
    task_config = replace(env_config or DoorConfig(), gamma=actual["gamma"])
    output = Path(output_dir).resolve()
    output.mkdir(parents=True, exist_ok=True)
    selected_device = resolve_device(device)
    eval_seed = final_eval_seed(seed)
    env = eval_env = logger = None
    try:
        env = Monitor(
            DoorEnv(model_path, config=task_config, render_mode="human" if render else None),
            str(output / "train"),
            info_keywords=("distance", "is_success", "grasp_seen", "press_seen",
                           "door_angle", "max_door_angle", "failure_reason"),
        )
        eval_env = Monitor(DoorEnv(model_path, config=task_config))
        eval_env.reset(seed=seed + 10_000)
        # Deliberately omit policy_kwargs: use SB3's own network, activation,
        # initialization, Gaussian policy and optimizer without customization.
        agent = PPO("MlpPolicy", env, seed=seed, device=selected_device, verbose=1, **overrides)
        # SB3 saves these serializable attributes inside both final/best .zip
        # files, so evaluation restores task settings without an adjacent JSON.
        agent.door_task_format = FORMAT
        agent.door_environment = env.unwrapped.specification()
        logger = configure(str(output / "logs"), ["stdout", "csv"])
        agent.set_logger(logger)
        settings = {
            "n_steps": agent.n_steps, "batch_size": agent.batch_size, "n_epochs": agent.n_epochs,
            "gamma": agent.gamma, "gae_lambda": agent.gae_lambda,
            "learning_rate": float(agent.lr_schedule(1.0)), "clip_range": float(agent.clip_range(1.0)),
            "ent_coef": agent.ent_coef, "vf_coef": agent.vf_coef,
            "max_grad_norm": agent.max_grad_norm, "target_kl": agent.target_kl,
            "net_arch": agent.policy.net_arch,
            "activation": agent.policy.activation_fn.__name__,
            "optimizer": agent.policy.optimizer_class.__name__,
            "ortho_init": agent.policy.ortho_init,
        }
        run_config = {
            "algorithm": "stable_baselines3.PPO", "environment": agent.door_environment,
            "agent": settings, "algorithm_overrides": overrides, "seed": seed,
            "final_eval_seed": eval_seed, "requested_timesteps": total_timesteps,
            "actual_timesteps": 0, "eval_freq": eval_freq, "eval_episodes": eval_episodes,
            "render": render, "device": str(selected_device),
        }
        config_path = output / "config.json"
        config_path.write_text(json.dumps(run_config, indent=2) + "\n", encoding="utf-8")
        print(f"SB3 defaults: network={settings['net_arch']}, activation={settings['activation']}, "
              f"gamma={agent.gamma}; explicit overrides={overrides}", flush=True)
        callback = EvalCallback(
            eval_env, best_model_save_path=str(output / "best"), log_path=str(output / "eval"),
            eval_freq=eval_freq, n_eval_episodes=eval_episodes, deterministic=True,
        )
        agent.learn(total_timesteps=total_timesteps, callback=callback)
        # SB3 normally emits update diagnostics at the next rollout; flush the
        # final update too, including when a smoke run contains only one rollout.
        agent.logger.dump(step=agent.num_timesteps)
        checkpoint = output / "ppo_door.zip"
        agent.save(checkpoint)
        run_config["actual_timesteps"] = agent.num_timesteps
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


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--timesteps", type=int, default=300_000)
    parser.add_argument("--output-dir", type=Path, default=Path("runs/door_sb3"))
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--model", type=Path)
    parser.add_argument("--n-steps", type=int, help="Override SB3's rollout length (default 2048).")
    parser.add_argument("--batch-size", type=int, help="Override SB3's minibatch size (default 64).")
    parser.add_argument("--epochs", type=int, help="Override SB3's update epochs (default 10).")
    parser.add_argument("--gamma", type=float, help="Override SB3's gamma (default 0.99); also used in reward shaping.")
    parser.add_argument("--max-steps", type=int, default=1000)
    parser.add_argument("--action-scale", type=float, default=0.03)
    parser.add_argument("--initial-noise", type=float, default=0.02)
    parser.add_argument("--eval-freq", type=int, default=10_000)
    parser.add_argument("--episodes", type=int, default=10)
    parser.add_argument("--evaluate", type=Path)
    parser.add_argument("--render", action="store_true")
    parser.add_argument("--device", choices=("auto", "cpu", "cuda"), default="auto")
    args = parser.parse_args()
    if args.evaluate:
        metrics = evaluate(args.evaluate, episodes=args.episodes, seed=args.seed,
                           device=args.device, render=args.render, model_path=args.model)
        print(json.dumps(metrics, indent=2))
    else:
        env_config = DoorConfig(max_steps=args.max_steps, action_scale=args.action_scale,
                                initial_noise=args.initial_noise)
        train(total_timesteps=args.timesteps, output_dir=args.output_dir, seed=args.seed,
              model_path=args.model or DEFAULT_MODEL, env_config=env_config, n_steps=args.n_steps,
              batch_size=args.batch_size, epochs=args.epochs, gamma=args.gamma,
              eval_freq=args.eval_freq, eval_episodes=args.episodes, render=args.render, device=args.device)


if __name__ == "__main__":
    main()
