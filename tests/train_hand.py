"""使用 PPOhand.py 训练或评估固定底盘 TidyBot 到达任务。

在仓库根目录同步 cpu/cuda extra 后运行：uv run --no-sync python tests/train_hand.py --render
"""

import argparse
import csv
import json
from pathlib import Path

import numpy as np
import torch

from rlinit.algorithmhand.PPOhand import PPO
from rlinit.algorithmhand.reach_env import DEFAULT_MODEL, ReachEnv


ROLLOUT_KEYS = (
    "states", "actions", "rewards", "next_states",
    "terminated", "truncated", "log_probs",
)


def save_agent(agent, path, config, timesteps):
    """保存网络、优化器和构造参数；格式独立于 SB3 的 .zip。"""
    torch.save({
        "format": "rlinit-ppohand-v1", "config": config, "timesteps": timesteps,
        "actor": agent.actor.state_dict(), "critic": agent.critic.state_dict(),
        "actor_optimizer": agent.actor_optimizer.state_dict(),
        "critic_optimizer": agent.critic_optimizer.state_dict(),
    }, path)


def load_agent(path, env, device="auto"):
    checkpoint = torch.load(path, map_location="cpu", weights_only=True)
    if checkpoint.get("format") != "rlinit-ppohand-v1":
        raise ValueError("需要此脚本生成的 PPOhand .pt 模型")
    config = dict(checkpoint["config"])
    config["device"] = device
    if (config["state_dim"] != env.observation_space.shape[0]
            or config["action_dim"] != env.action_space.shape[0]):
        raise ValueError("模型与环境的观测/动作维度不匹配")
    agent = PPO(**config)
    agent.actor.load_state_dict(checkpoint["actor"])
    agent.critic.load_state_dict(checkpoint["critic"])
    agent.actor.eval()
    agent.critic.eval()
    return agent


def evaluate(checkpoint, episodes=20, seed=10_000, model_path=DEFAULT_MODEL, render=False, device="auto"):
    if episodes < 1:
        raise ValueError("episodes 必须为正整数")
    env = ReachEnv(model_path, render_mode="human" if render else None)
    try:
        agent = load_agent(checkpoint, env, device=device)
        returns, distances, successes = [], [], []
        for episode in range(episodes):
            state, _ = env.reset(seed=seed + episode)
            episode_return = 0.0
            while True:
                action = agent.take_action(state, deterministic=True)
                state, reward, terminated, truncated, info = env.step(action)
                episode_return += reward
                if terminated or truncated:
                    break
            returns.append(episode_return)
            distances.append(info["distance"])
            successes.append(info["is_success"])
        return {
            "episodes": episodes, "success_rate": float(np.mean(successes)),
            "mean_return": float(np.mean(returns)),
            "mean_final_distance_m": float(np.mean(distances)),
        }
    finally:
        env.close()


def train(total_timesteps=300_000, output_dir="runs/reach_ppohand", seed=0,
          model_path=DEFAULT_MODEL, n_steps=2048, epochs=10, hidden_dim=128,
          render=False, device="auto"):
    if total_timesteps < 1 or n_steps < 2:
        raise ValueError("timesteps 须为正数，n_steps 须至少为 2")
    torch.manual_seed(seed)
    np.random.seed(seed)
    output = Path(output_dir).resolve()
    output.mkdir(parents=True, exist_ok=True)
    env = ReachEnv(model_path, render_mode="human" if render else None)
    try:
        # 在这里调用 from_env：根据当前模型创建 20 输入、7 输出的策略。
        agent = PPO.from_env(env, hidden_dim=hidden_dim, epochs=epochs, device=device)
        print(f"Using device: {agent.device}", flush=True)
        config = {
            "state_dim": agent.state_dim, "action_dim": agent.action_dim,
            "hidden_dim": hidden_dim, "epochs": epochs, "actor_lr": 3e-4,
            "critic_lr": 1e-3, "gamma": agent.gamma, "lmbda": agent.lmbda,
            "eps": agent.eps, "device": str(agent.device),
        }
        run_config = {
            "agent": config, "model_path": str(Path(model_path).resolve()),
            "seed": seed, "total_timesteps": total_timesteps,
            "n_steps": n_steps, "render": render,
        }
        (output / "config.json").write_text(json.dumps(run_config, indent=2) + "\n")
        state, _ = env.reset(seed=seed)
        steps = episode_count = episode_length = 0
        episode_return = 0.0
        with (output / "episodes.csv").open("w", newline="") as episode_file, \
                (output / "updates.csv").open("w", newline="") as update_file:
            episode_log = csv.writer(episode_file)
            update_log = csv.writer(update_file)
            episode_log.writerow(["episode", "timesteps", "return", "length", "distance", "is_success"])
            update_log.writerow(["timesteps", "actor_loss", "critic_loss"])
            while steps < total_timesteps:
                rollout = {key: [] for key in ROLLOUT_KEYS}
                for _ in range(min(n_steps, total_timesteps - steps)):
                    action, log_prob = agent.take_action(state, return_log_prob=True)
                    next_state, reward, terminated, truncated, info = env.step(action)
                    # 保存 step 的 next_state，再 reset，保留超时 bootstrap 所需状态。
                    for key, value in zip(ROLLOUT_KEYS, (
                        state, action, reward, next_state, terminated, truncated, log_prob
                    )):
                        rollout[key].append(value)
                    state = next_state
                    steps += 1
                    episode_length += 1
                    episode_return += reward
                    if terminated or truncated:
                        episode_count += 1
                        episode_log.writerow([episode_count, steps, episode_return,
                                              episode_length, info["distance"], info["is_success"]])
                        episode_file.flush()
                        state, _ = env.reset()
                        episode_length, episode_return = 0, 0.0
                # 同一段 rollout 的全部动作采样完成后才更新，固定旧策略 log_probs。
                losses = agent.update(rollout)
                update_log.writerow([steps, losses["actor_loss"], losses["critic_loss"]])
                update_file.flush()
                print(f"steps={steps}/{total_timesteps} episodes={episode_count} "
                      f"actor_loss={losses['actor_loss']:.4f} critic_loss={losses['critic_loss']:.4f}",
                      flush=True)
        checkpoint = output / "ppo_hand.pt"
        save_agent(agent, checkpoint, config, steps)
    finally:
        env.close()
    metrics = evaluate(checkpoint, episodes=10, seed=seed + 20_000, model_path=model_path, device=device)
    (output / "metrics.json").write_text(json.dumps(metrics, indent=2) + "\n")
    print(f"Saved: {checkpoint}\nEvaluation: {metrics}")
    return checkpoint


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--timesteps", type=int, default=300_000)
    parser.add_argument("--output-dir", type=Path, default=Path("runs/reach_ppohand"))
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--model", type=Path, default=DEFAULT_MODEL)
    parser.add_argument("--n-steps", type=int, default=2048, help="每次更新前收集的步数；手写实现使用整段 rollout 更新")
    parser.add_argument("--epochs", type=int, default=10)
    parser.add_argument("--hidden-dim", type=int, default=128)
    parser.add_argument("--evaluate", type=Path, help="加载 PPOhand .pt 模型评估")
    parser.add_argument("--episodes", type=int, default=20)
    parser.add_argument("--render", action="store_true", help="显示训练或评估窗口")
    parser.add_argument("--device", default="auto", choices=("auto", "cpu", "cuda"))
    args = parser.parse_args()
    if args.evaluate:
        metrics = evaluate(args.evaluate, args.episodes, args.seed, args.model, args.render, device=args.device)
        print(json.dumps(metrics, indent=2))
    else:
        train(args.timesteps, args.output_dir, args.seed, args.model,
              args.n_steps, args.epochs, args.hidden_dim, args.render, device=args.device)


if __name__ == "__main__":
    main()
