"""Train/evaluate the fixed-base door MDP using the original PPOhand.py.

uv run --no-sync python tests/trainp2_hand.py --timesteps 300000
uv run --no-sync python tests/trainp2_hand.py --evaluate runs/door_ppohand/ppo_hand.pt --render
"""

import argparse
import csv
import json
from pathlib import Path

import numpy as np
import torch

from rlinit.algorithmhand.PPOhand import PPO
from rlinit.algorithmhand.door_env import DEFAULT_MODEL, DoorConfig, DoorEnv
from rlinit.algorithmhand.evaluation import final_eval_seed


ROLLOUT_KEYS = ("states", "actions", "rewards", "next_states", "terminated", "truncated", "log_probs")
FORMAT = "rlinit-door-ppohand-v1"


def save_agent(agent, env, path, agent_config, timesteps):
    torch.save({
        "format": FORMAT, "agent_config": {**agent_config, "hidden_depth": agent.hidden_depth}, "environment": env.specification(),
        "timesteps": timesteps, "actor": agent.actor.state_dict(), "critic": agent.critic.state_dict(),
        "actor_optimizer": agent.actor_optimizer.state_dict(),
        "critic_optimizer": agent.critic_optimizer.state_dict(),
    }, path)


def load_agent(path, *, device="auto", render=False, model_path=None):
    checkpoint = torch.load(path, map_location="cpu", weights_only=True)
    if checkpoint.get("format") != FORMAT:
        raise ValueError("Expected a door-task checkpoint from trainp2_hand.py")
    specification = checkpoint["environment"]
    env = DoorEnv(model_path or specification["model_path"],
                  config=DoorConfig(**specification["config"]), render_mode="human" if render else None)
    try:
        config = dict(checkpoint["agent_config"])
        config.setdefault("hidden_depth", 2 if "fc3.weight" in checkpoint["actor"] else 1)
        if config["state_dim"] != env.observation_space.shape[0] or config["action_dim"] != 8:
            raise ValueError("Checkpoint observation/action dimensions do not match the environment")
        config["device"] = device
        agent = PPO(**config)
        agent.actor.load_state_dict(checkpoint["actor"])
        agent.critic.load_state_dict(checkpoint["critic"])
        agent.actor.eval()
        agent.critic.eval()
        return agent, env
    except Exception:
        env.close()
        raise


def evaluate(checkpoint, *, episodes=10, seed=20_000, device="auto", render=False, model_path=None):
    if episodes < 1:
        raise ValueError("episodes must be positive")
    agent, env = load_agent(checkpoint, device=device, render=render, model_path=model_path)
    records = []
    try:
        for episode in range(episodes):
            state, _ = env.reset(seed=seed + episode)
            episode_return = 0.0
            length = 0
            while True:
                action = agent.take_action(state, deterministic=True)
                state, reward, terminated, truncated, info = env.step(action)
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
                "truncated": truncated,
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


def train(*, total_timesteps=300_000, output_dir="runs/door_ppohand", seed=0,
          model_path=DEFAULT_MODEL, n_steps=2048, epochs=10, hidden_dim=128,
          config=None, render=False, device="auto", eval_episodes=10, batch_size=64):
    if total_timesteps < 1 or n_steps < 2 or eval_episodes < 1:
        raise ValueError("timesteps/eval_episodes must be positive; n_steps must be >= 2")
    config = config or DoorConfig()
    torch.manual_seed(seed)
    np.random.seed(seed)
    output = Path(output_dir).resolve()
    output.mkdir(parents=True, exist_ok=True)
    env = DoorEnv(model_path, config=config, render_mode="human" if render else None)
    eval_seed = final_eval_seed(seed)
    try:
        agent = PPO.from_env(env, hidden_dim=hidden_dim, epochs=epochs, gamma=config.gamma,
                             device=device, batch_size=batch_size)
        agent_config = {
            "state_dim": agent.state_dim, "action_dim": agent.action_dim, "hidden_dim": hidden_dim,
            "hidden_depth": agent.hidden_depth,
            "epochs": epochs, "actor_lr": 3e-4, "critic_lr": 1e-3,
            "gamma": agent.gamma, "lmbda": agent.lmbda, "eps": agent.eps, "device": str(agent.device),
            "batch_size": agent.batch_size,
        }
        run_config = {
            "agent": agent_config, "environment": env.specification(), "seed": seed,
            "final_eval_seed": eval_seed, "total_timesteps": total_timesteps,
            "n_steps": n_steps, "render": render, "eval_episodes": eval_episodes,
        }
        (output / "config.json").write_text(json.dumps(run_config, indent=2) + "\n", encoding="utf-8")
        print(f"Using {agent.device}; obs={agent.state_dim}, action={agent.action_dim}, gamma={agent.gamma}", flush=True)
        state, _ = env.reset(seed=seed)
        steps = episodes = length = 0
        episode_return = 0.0
        checkpoint = output / "ppo_hand.pt"
        with (output / "episodes.csv").open("w", newline="", encoding="utf-8") as episode_file, \
                (output / "updates.csv").open("w", newline="", encoding="utf-8") as update_file:
            episode_log, update_log = csv.writer(episode_file), csv.writer(update_file)
            episode_log.writerow(["episode", "timesteps", "return", "length", "is_success", "grasp_seen",
                                  "press_seen", "door_deg", "max_door_deg", "distance_m", "failure_reason", "truncated"])
            update_log.writerow(["timesteps", "actor_loss", "critic_loss"])
            while steps < total_timesteps:
                rollout = {key: [] for key in ROLLOUT_KEYS}
                for _ in range(min(n_steps, total_timesteps - steps)):
                    action, log_prob = agent.take_action(state, return_log_prob=True)
                    next_state, reward, terminated, truncated, info = env.step(action)
                    for key, value in zip(ROLLOUT_KEYS, (state, action, reward, next_state, terminated, truncated, log_prob)):
                        rollout[key].append(value)
                    state = next_state
                    steps += 1
                    length += 1
                    episode_return += reward
                    if terminated or truncated:
                        episodes += 1
                        episode_log.writerow([
                            episodes, steps, episode_return, length, info["is_success"], info["grasp_seen"],
                            info["press_seen"], np.rad2deg(info["door_angle"]), np.rad2deg(info["max_door_angle"]),
                            info["distance"], info["failure_reason"], truncated,
                        ])
                        episode_file.flush()
                        state, _ = env.reset()
                        length, episode_return = 0, 0.0
                # GAE is prepared in time order; PPOhand then shuffles minibatches.
                losses = agent.update(rollout)
                update_log.writerow([steps, losses["actor_loss"], losses["critic_loss"]])
                update_file.flush()
                save_agent(agent, env, checkpoint, agent_config, steps)
                print(f"steps={steps}/{total_timesteps} episodes={episodes} "
                      f"actor={losses['actor_loss']:.4f} critic={losses['critic_loss']:.4f}", flush=True)
    finally:
        env.close()
    print(f"Final evaluation seed: {eval_seed}", flush=True)
    metrics = evaluate(checkpoint, episodes=eval_episodes, seed=eval_seed, device=device, model_path=model_path)
    (output / "metrics.json").write_text(json.dumps(metrics, indent=2) + "\n", encoding="utf-8")
    print(f"Saved: {checkpoint}; success={metrics['success_rate']:.1%}, grasp={metrics['grasp_rate']:.1%}")
    return checkpoint


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--timesteps", type=int, default=300_000)
    parser.add_argument("--output-dir", type=Path, default=Path("runs/door_ppohand"))
    parser.add_argument("--seed", type=int, default=0, help="Training seed; manual evaluation uses this as its first seed.")
    parser.add_argument("--model", type=Path, help="Override the scene XML; evaluation otherwise uses the checkpoint's model.")
    parser.add_argument("--n-steps", type=int, default=2048)
    parser.add_argument("--batch-size", type=int, default=64)
    parser.add_argument("--epochs", type=int, default=10)
    parser.add_argument("--hidden-dim", type=int, default=128)
    parser.add_argument("--gamma", type=float, default=0.999)
    parser.add_argument("--max-steps", type=int, default=1000)
    parser.add_argument("--action-scale", type=float, default=0.03)
    parser.add_argument("--initial-noise", type=float, default=0.02)
    parser.add_argument("--evaluate", type=Path)
    parser.add_argument("--episodes", type=int, default=10, help="Number of evaluation episodes.")
    parser.add_argument("--render", action="store_true")
    parser.add_argument("--device", choices=("auto", "cpu", "cuda"), default="auto")
    args = parser.parse_args()
    if args.evaluate:
        metrics = evaluate(args.evaluate, episodes=args.episodes, seed=args.seed,
                           device=args.device, render=args.render, model_path=args.model)
        print(json.dumps(metrics, indent=2))
    else:
        config = DoorConfig(gamma=args.gamma, max_steps=args.max_steps,
                            action_scale=args.action_scale, initial_noise=args.initial_noise)
        train(total_timesteps=args.timesteps, output_dir=args.output_dir, seed=args.seed,
              model_path=args.model or DEFAULT_MODEL, n_steps=args.n_steps, epochs=args.epochs,
              hidden_dim=args.hidden_dim, config=config, render=args.render,
              device=args.device, eval_episodes=args.episodes, batch_size=args.batch_size)


if __name__ == "__main__":
    main()
