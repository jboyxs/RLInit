"""Train/evaluate the shared ReachEnv using Stable-Baselines3 PPO."""
import argparse
import json
from pathlib import Path
import torch
from stable_baselines3 import PPO
from stable_baselines3.common.callbacks import EvalCallback
from stable_baselines3.common.monitor import Monitor
from rlinit.algorithmhand.reach_env import DEFAULT_MODEL, ReachEnv

def resolve_device(device="auto"):
    """Auto prefers CUDA; explicit CUDA fails clearly when unavailable."""
    if str(device) == "auto":
        return torch.device("cuda" if torch.cuda.is_available() else "cpu")
    selected = torch.device(device)
    if selected.type not in ("cpu", "cuda"):
        raise ValueError("device 必须是 auto、cpu 或 cuda（可指定 cuda:0）")
    if selected.type == "cuda" and not torch.cuda.is_available():
        raise RuntimeError("CUDA 不可用：请安装 CUDA 版 PyTorch 并检查 NVIDIA 驱动；或使用 --device cpu")
    return selected


FINAL_EVAL_SEED_OFFSET = 20_000


def final_eval_seed(training_seed):
    """Return the first episode seed for every trainer's final evaluation."""
    return training_seed + FINAL_EVAL_SEED_OFFSET


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
    eval_seed = final_eval_seed(seed)
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
            "final_eval_seed": eval_seed,
            "requested_timesteps": total_timesteps, "actual_timesteps": agent.num_timesteps,
            "n_steps": n_steps, "batch_size": batch_size,
            "render": render,
            "device": str(selected_device),
        }
        (output / "config.json").write_text(json.dumps(config, indent=2) + "\n")
        print(f"Final evaluation seed: {eval_seed}", flush=True)
        metrics = evaluate_ppo(checkpoint, episodes=10, seed=eval_seed, model_path=model_path, device=selected_device)
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


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--timesteps", type=int, default=300_000)
    parser.add_argument("--output-dir", type=Path, default=Path("runs/reach_ppo"))
    parser.add_argument("--seed", type=int, default=0,
                        help="Training seed; final evaluation starts at seed + 20000. In evaluation mode, the first episode seed.")
    parser.add_argument("--model", type=Path, default=DEFAULT_MODEL)
    parser.add_argument("--n-steps", type=int, default=2048)
    parser.add_argument("--batch-size", type=int, default=64)
    parser.add_argument("--evaluate", type=Path, help="Evaluate a saved .zip instead of training")
    parser.add_argument("--episodes", type=int, default=20)
    parser.add_argument("--render", action="store_true", help="Show training or evaluation in the MuJoCo viewer")
    parser.add_argument("--device", default="auto", choices=("auto", "cpu", "cuda"))
    args = parser.parse_args()
    if args.evaluate:
        print(json.dumps(evaluate_ppo(args.evaluate, args.episodes, args.seed, args.model, args.render, device=args.device), indent=2))
    else:
        train_ppo(args.timesteps, args.output_dir, args.seed, args.model, args.n_steps, args.batch_size, render=args.render, device=args.device)


if __name__ == "__main__":
    main()
