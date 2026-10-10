"""Train/evaluate the first reaching task using Stable-Baselines3 PPO."""

import json
from pathlib import Path

from stable_baselines3 import PPO
from stable_baselines3.common.callbacks import EvalCallback
from stable_baselines3.common.monitor import Monitor

from .reach_env import DEFAULT_MODEL, ReachEnv
from .device import resolve_device


def train_ppo(
    total_timesteps=300_000,
    output_dir="runs/reach_ppo",
    seed=0,
    model_path=DEFAULT_MODEL,
    n_steps=2048,
    batch_size=64,
    render=False,
    device="auto",
):
    """Train PPO, save checkpoints/logs, and return the final checkpoint path.

    Training automatically selects CUDA/CPU; render=True opens a paced viewer.
    Timesteps are rounded up to full rollouts by
    SB3; short runs verify the pipeline but do not establish convergence.
    """
    if total_timesteps < 1 or n_steps < 2 or batch_size < 2:
        raise ValueError("Require timesteps > 0, n_steps >= 2, batch_size >= 2")
    if n_steps % batch_size:
        raise ValueError("batch_size must divide n_steps")
    selected_device = resolve_device(device)
    output = Path(output_dir).resolve()
    output.mkdir(parents=True, exist_ok=True)
    env = Monitor(ReachEnv(model_path, render_mode="human" if render else None), str(output / "train"), info_keywords=("distance", "is_success"))
    eval_env = Monitor(ReachEnv(model_path))
    try:
        eval_env.reset(seed=seed + 10_000)
        agent = PPO(
            "MlpPolicy", env, learning_rate=3e-4, n_steps=n_steps,
            batch_size=batch_size, n_epochs=10, gamma=0.99, gae_lambda=0.95,
            clip_range=0.2, policy_kwargs={"net_arch": dict(pi=[64, 64], vf=[64, 64])},
            seed=seed, device=selected_device, verbose=1,
        )
        callback = EvalCallback(
            eval_env, best_model_save_path=str(output / "best"),
            log_path=str(output / "eval"), eval_freq=max(10_000, n_steps),
            n_eval_episodes=10, deterministic=True,
        )
        agent.learn(total_timesteps=total_timesteps, callback=callback)
        checkpoint = output / "ppo_reach.zip"
        agent.save(checkpoint)
        config = {
            "model_path": str(Path(model_path).resolve()), "seed": seed,
            "requested_timesteps": total_timesteps, "actual_timesteps": agent.num_timesteps,
            "n_steps": n_steps, "batch_size": batch_size,
            "render": render,
            "device": str(selected_device),
        }
        (output / "config.json").write_text(json.dumps(config, indent=2) + "\n")
        metrics = evaluate_ppo(checkpoint, episodes=10, seed=seed + 20_000, model_path=model_path, device=selected_device)
        (output / "metrics.json").write_text(json.dumps(metrics, indent=2) + "\n")
        print(f"Saved: {checkpoint}\nEvaluation: {metrics}")
        return checkpoint
    finally:
        env.close()
        eval_env.close()


def evaluate_ppo(checkpoint, episodes=20, seed=10_000, model_path=DEFAULT_MODEL, render=False, device="auto"):
    """Evaluate deterministic actions on held-out seeded episodes."""
    if episodes < 1:
        raise ValueError("episodes must be positive")
    env = ReachEnv(model_path, render_mode="human" if render else None)
    try:
        agent = PPO.load(checkpoint, device=resolve_device(device))
        returns, distances, successes = [], [], []
        for episode in range(episodes):
            obs, info = env.reset(seed=seed + episode)
            episode_return = 0.0
            while True:
                action, _ = agent.predict(obs, deterministic=True)
                obs, reward, terminated, truncated, info = env.step(action)
                episode_return += reward
                if terminated or truncated:
                    break
            returns.append(episode_return)
            distances.append(info["distance"])
            successes.append(info["is_success"])
        return {
            "episodes": episodes, "success_rate": sum(successes) / episodes,
            "mean_return": sum(returns) / episodes,
            "mean_final_distance_m": sum(distances) / episodes,
        }
    finally:
        env.close()
