"""Door curriculum using the handwritten PPOhand.PPO and shuffled minibatches.

uv run --no-sync python tests/train_curriculum_hand.py --timesteps 1000000
uv run --no-sync python tests/train_curriculum_hand.py --evaluate runs/door_curriculum_hand/ppo_curriculum_hand.pt --full-task --render
"""

from dataclasses import asdict
import argparse
import csv
import json
from pathlib import Path

import numpy as np
import torch

from rlinit.algorithmhand.PPOhand import PPO
from rlinit.algorithmhand.door_curriculum import CurriculumProgress, evaluate_policy
from rlinit.algorithmhand.door_curriculum_env import CurriculumConfig, DoorCurriculumEnv, STAGES
from rlinit.algorithmhand.door_env import DEFAULT_MODEL, DoorConfig
from rlinit.algorithmhand.evaluation import final_eval_seed


FORMAT = "rlinit-door-curriculum-ppohand-v1"
ROLLOUT_KEYS = ("states", "actions", "rewards", "next_states", "terminated", "truncated", "log_probs")


def save_agent(agent, env, path, agent_config, progress, timesteps):
    torch.save({
        "format": FORMAT, "agent_config": agent_config, "environment": env.specification(),
        "progress": asdict(progress), "timesteps": timesteps,
        "actor": agent.actor.state_dict(), "critic": agent.critic.state_dict(),
        "actor_optimizer": agent.actor_optimizer.state_dict(),
        "critic_optimizer": agent.critic_optimizer.state_dict(),
    }, path)


def load_agent(path, *, device="auto", render=False, model_path=None,
               full_task=False, stage=None, home_probability=None):
    checkpoint = torch.load(path, map_location="cpu", weights_only=True)
    if checkpoint.get("format") != FORMAT:
        raise ValueError("Expected a curriculum .pt checkpoint from train_curriculum_hand.py")
    spec = checkpoint["environment"]
    env = DoorCurriculumEnv(
        model_path or spec["model_path"], config=DoorConfig(**spec["config"]),
        curriculum=CurriculumConfig(**spec["curriculum"]),
        stage="full" if full_task else (spec["stage"] if stage is None else stage),
        home_probability=1.0 if full_task else (spec["home_probability"] if home_probability is None else home_probability),
        render_mode="human" if render else None,
    )
    try:
        config = {**checkpoint["agent_config"], "device": device}
        if (config["state_dim"] != env.observation_space.shape[0]
                or config["action_dim"] != env.action_space.shape[0]):
            raise ValueError("Checkpoint observation/action dimensions do not match the curriculum environment")
        agent = PPO(**config)
        agent.actor.load_state_dict(checkpoint["actor"])
        agent.critic.load_state_dict(checkpoint["critic"])
        agent.actor.eval()
        agent.critic.eval()
        return agent, env
    except Exception:
        env.close()
        raise


def evaluate_agent(agent, specification, **kwargs):
    return evaluate_policy(lambda obs: agent.take_action(obs, deterministic=True), specification, **kwargs)


def evaluate(checkpoint, *, episodes=10, seed=20_000, device="auto", render=False,
             model_path=None, full_task=False, stage=None, home_probability=None):
    if episodes < 1:
        raise ValueError("episodes must be positive")
    agent, env = load_agent(checkpoint, device=device, model_path=model_path,
                            full_task=full_task, stage=stage, home_probability=home_probability)
    try:
        return evaluate_agent(agent, env.specification(), episodes=episodes, seed=seed, render=render)
    finally:
        env.close()


def train(*, total_timesteps=1_000_000, output_dir="runs/door_curriculum_hand", seed=0,
          model_path=DEFAULT_MODEL, config=None, curriculum=None,
          start_stage="reach", home_probability=0.0, eval_freq=10_000,
          eval_episodes=10, threshold=0.8, patience=2, home_step=0.25,
          n_steps=2048, batch_size=64, epochs=10, hidden_dim=128,
          render=False, device="auto"):
    if min(total_timesteps, eval_freq, eval_episodes, patience) < 1 or n_steps < 2:
        raise ValueError("Budgets, evaluation count and patience must be positive; n_steps must be >= 2")
    if not 0 < threshold <= 1 or not 0 < home_step <= 1:
        raise ValueError("threshold and home_step must be in (0, 1]")
    # Match the SB3 curriculum discount while retaining the handwritten PPO's
    # two-layer 128 ReLU networks, tanh Gaussian and separate optimizers.
    core = config or DoorConfig(gamma=0.99)
    lesson = curriculum or CurriculumConfig()
    progress = CurriculumProgress(stage=start_stage, home_probability=home_probability)
    torch.manual_seed(seed)
    np.random.seed(seed)
    output = Path(output_dir).resolve()
    output.mkdir(parents=True, exist_ok=True)
    (output / "best").mkdir(exist_ok=True)
    checkpoint = output / "ppo_curriculum_hand.pt"
    env = DoorCurriculumEnv(model_path, config=core, curriculum=lesson,
                            stage=start_stage, home_probability=home_probability,
                            render_mode="human" if render else None)
    try:
        agent = PPO.from_env(env, hidden_dim=hidden_dim, gamma=core.gamma,
                             epochs=epochs, batch_size=batch_size, device=device)
        agent_config = {
            "state_dim": agent.state_dim, "action_dim": agent.action_dim,
            "hidden_dim": hidden_dim, "hidden_depth": agent.hidden_depth,
            "actor_lr": 3e-4, "critic_lr": 1e-3, "lmbda": agent.lmbda,
            "epochs": agent.epochs, "eps": agent.eps, "gamma": agent.gamma,
            "device": str(agent.device), "batch_size": agent.batch_size,
        }
        run_config = {
            "agent": agent_config, "seed": seed, "final_eval_seed": final_eval_seed(seed),
            "requested_timesteps": total_timesteps, "n_steps": n_steps,
            "eval_freq": eval_freq, "eval_episodes": eval_episodes,
            "threshold": threshold, "patience": patience, "home_step": home_step,
        }
        steps = episodes = length = 0
        episode_return = 0.0
        next_eval = eval_freq
        best_scores = {}

        def save_progress():
            save_agent(agent, env, checkpoint, agent_config, progress, steps)
            run_config.update(environment=env.specification(), progress=asdict(progress), actual_timesteps=steps)
            (output / "config.json").write_text(json.dumps(run_config, indent=2) + "\n", encoding="utf-8")

        print(f"Using {agent.device}; obs={agent.state_dim}, action={agent.action_dim}, gamma={agent.gamma}", flush=True)
        state, info = env.reset(seed=seed)
        with (output / "episodes.csv").open("w", newline="", encoding="utf-8") as episode_file, \
                (output / "updates.csv").open("w", newline="", encoding="utf-8") as update_file, \
                (output / "curriculum.csv").open("w", newline="", encoding="utf-8") as curriculum_file:
            episode_log = csv.writer(episode_file)
            update_log = csv.writer(update_file)
            curriculum_log = csv.writer(curriculum_file)
            episode_log.writerow([
                "episode", "timesteps", "stage", "home_probability", "episode_start", "return", "length",
                "is_success", "full_task_success", "grasp_seen", "press_seen", "door_deg",
                "distance_m", "alignment", "failure_reason", "truncated", "cutoff",
            ])
            update_log.writerow(["timesteps", "stage", "home_probability", "actor_loss", "critic_loss"])
            curriculum_log.writerow([
                "timesteps", "stage", "home_probability", "success_rate", "home_episodes",
                "home_success_rate", "mean_return", "min_distance_m", "promoted",
            ])

            def log_episode(cutoff=""):
                nonlocal episodes, length, episode_return
                if not length:
                    return
                episodes += 1
                episode_log.writerow([
                    episodes, steps, env.stage, env.home_probability, info["episode_start"], episode_return, length,
                    info["is_success"], info["full_task_success"], info["grasp_seen"], info["press_seen"],
                    np.rad2deg(info["door_angle"]), info["distance"], info["alignment"],
                    info["failure_reason"], truncated, cutoff,
                ])
                episode_file.flush()
                length, episode_return = 0, 0.0

            while steps < total_timesteps and not progress.completed:
                rollout = {key: [] for key in ROLLOUT_KEYS}
                for _ in range(min(n_steps, total_timesteps - steps)):
                    action, log_prob = agent.take_action(state, return_log_prob=True)
                    next_state, reward, terminated, truncated, info = env.step(action)
                    # Record the sampled action and its own log probability,
                    # including the gripper dimension ignored by reach.
                    for key, value in zip(ROLLOUT_KEYS, (state, action, reward, next_state, terminated, truncated, log_prob)):
                        rollout[key].append(value)
                    state = next_state
                    steps += 1
                    length += 1
                    episode_return += reward
                    if terminated or truncated:
                        log_episode()
                        state, info = env.reset()
                # Consume all old-stage transitions before changing stages.
                # GAE sees time order; PPOhand shuffles minibatches afterwards.
                losses = agent.update(rollout)
                update_log.writerow([steps, env.stage, env.home_probability,
                                     losses["actor_loss"], losses["critic_loss"]])
                update_file.flush()
                print(f"steps={steps}/{total_timesteps}, {env.stage}, home={env.home_probability:.0%}; "
                      f"actor={losses['actor_loss']:.4f}, critic={losses['critic_loss']:.4f}", flush=True)

                if steps >= next_eval or steps >= total_timesteps:
                    spec = env.specification()
                    metrics = evaluate_agent(agent, spec, episodes=eval_episodes, seed=seed + 10_000)
                    key = f"{progress.stage}_home_{progress.home_probability:.2f}"
                    score = (metrics["success_rate"], metrics["mean_return"])
                    if key not in best_scores or score > best_scores[key]:
                        best_scores[key] = score
                        save_agent(agent, env, output / "best" / f"{key}.pt", agent_config, progress, steps)
                    previous = (progress.stage, progress.home_probability)
                    promoted = progress.advance(metrics["success_rate"], threshold, patience, home_step)
                    curriculum_log.writerow([
                        steps, *previous, metrics["success_rate"], metrics["home_episodes"],
                        metrics["home_success_rate"], metrics["mean_return"], metrics["mean_min_distance_m"], promoted,
                    ])
                    curriculum_file.flush()
                    (output / f"evaluation_{key}.json").write_text(json.dumps(metrics, indent=2) + "\n", encoding="utf-8")
                    print(f"Evaluation: success={metrics['success_rate']:.0%}, "
                          f"min distance={metrics['mean_min_distance_m']:.3f} m; promoted={promoted}", flush=True)
                    if previous != (progress.stage, progress.home_probability):
                        log_episode(cutoff="stage_switch")
                        env.set_stage(progress.stage, progress.home_probability)
                        state, info = env.reset()
                        # The same actor, critic and optimizer states continue.
                    next_eval = steps + eval_freq
                save_progress()
            log_episode(cutoff="budget_end" if not progress.completed else "curriculum_complete")

        print(f"Final evaluation seed: {final_eval_seed(seed)}", flush=True)
        metrics = evaluate_agent(agent, env.specification(), episodes=eval_episodes,
                                 seed=final_eval_seed(seed), stage="full", home_probability=1.0)
        metrics["curriculum_completed"] = progress.completed
        (output / "metrics.json").write_text(json.dumps(metrics, indent=2) + "\n", encoding="utf-8")
        print(f"Saved: {checkpoint}; full-task success={metrics['full_task_success_rate']:.0%}; "
              f"curriculum completed={progress.completed}", flush=True)
        return checkpoint
    finally:
        env.close()


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--timesteps", type=int, default=1_000_000)
    parser.add_argument("--output-dir", type=Path, default=Path("runs/door_curriculum_hand"))
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--model", type=Path)
    parser.add_argument("--eval-freq", type=int, default=10_000)
    parser.add_argument("--episodes", type=int, default=10)
    parser.add_argument("--success-threshold", type=float, default=0.8)
    parser.add_argument("--patience", type=int, default=2)
    parser.add_argument("--home-step", type=float, default=0.25)
    parser.add_argument("--stage", choices=STAGES)
    parser.add_argument("--home-probability", type=float)
    parser.add_argument("--n-steps", type=int, default=2048)
    parser.add_argument("--batch-size", type=int, default=64)
    parser.add_argument("--epochs", type=int, default=10)
    parser.add_argument("--hidden-dim", type=int, default=128)
    parser.add_argument("--gamma", type=float, default=0.99)
    parser.add_argument("--max-steps", type=int, default=1000)
    parser.add_argument("--action-scale", type=float, default=0.03)
    parser.add_argument("--initial-noise", type=float, default=0.02)
    parser.add_argument("--evaluate", type=Path)
    parser.add_argument("--full-task", action="store_true")
    parser.add_argument("--render", action="store_true")
    parser.add_argument("--device", choices=("auto", "cpu", "cuda"), default="auto")
    args = parser.parse_args()
    if args.evaluate:
        metrics = evaluate(args.evaluate, episodes=args.episodes, seed=args.seed, device=args.device,
                           render=args.render, model_path=args.model, full_task=args.full_task,
                           stage=args.stage, home_probability=args.home_probability)
        print(json.dumps(metrics, indent=2))
    else:
        train(total_timesteps=args.timesteps, output_dir=args.output_dir, seed=args.seed,
              model_path=args.model or DEFAULT_MODEL,
              config=DoorConfig(gamma=args.gamma, max_steps=args.max_steps,
                                action_scale=args.action_scale, initial_noise=args.initial_noise),
              start_stage=args.stage or "reach", home_probability=args.home_probability or 0.0,
              eval_freq=args.eval_freq, eval_episodes=args.episodes, threshold=args.success_threshold,
              patience=args.patience, home_step=args.home_step, n_steps=args.n_steps,
              batch_size=args.batch_size, epochs=args.epochs, hidden_dim=args.hidden_dim,
              render=args.render, device=args.device)


if __name__ == "__main__":
    main()
