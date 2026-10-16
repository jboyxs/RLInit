"""Multiprocess door curriculum with batched handwritten PPO inference.

uv run --no-sync python tests/train_curriculum_hand_parallel.py --num-envs 8 --device cuda
uv run --no-sync python tests/train_curriculum_hand_parallel.py --num-envs 16 --device cuda

--n-steps is per environment; --timesteps and --eval-freq count total transitions.
Training is headless. Existing curriculum .pt evaluation supports these models.
"""

from dataclasses import asdict
from functools import partial
import argparse
import csv
import os
from pathlib import Path
import time
import json

from gymnasium.vector import AsyncVectorEnv, AutoresetMode
import numpy as np
import torch

from rlinit.algorithmhand.PPOhand import PPO
from rlinit.algorithmhand.door_curriculum import CurriculumProgress
from rlinit.algorithmhand.door_curriculum_env import CurriculumConfig, DoorCurriculumEnv, STAGES
from rlinit.algorithmhand.door_env import DEFAULT_MODEL, DoorConfig
from rlinit.algorithmhand.evaluation import final_eval_seed
from train_curriculum_hand import ROLLOUT_KEYS, evaluate, evaluate_agent, save_agent


def make_parallel_envs(num_envs, model_path, config, curriculum, stage, home_probability,
                       start_method="spawn"):
    """Each worker owns one CPU simulation; only the parent runs the policy."""
    factory = partial(DoorCurriculumEnv, model_path, config=config, curriculum=curriculum,
                      stage=stage, home_probability=home_probability)
    # Set these before spawning, so NumPy/PyTorch imports in workers cannot
    # create dozens of BLAS threads per simulation. Restore the parent's env.
    names = ("OMP_NUM_THREADS", "MKL_NUM_THREADS", "OPENBLAS_NUM_THREADS")
    previous = {name: os.environ.get(name) for name in names}
    try:
        os.environ.update(dict.fromkeys(names, "1"))
        return AsyncVectorEnv([factory] * num_envs, context=start_method,
                              shared_memory=True, autoreset_mode=AutoresetMode.SAME_STEP)
    finally:
        for name, value in previous.items():
            if value is None:
                os.environ.pop(name, None)
            else:
                os.environ[name] = value


def info_at(infos, index):
    """Unpack Gymnasium's dict of arrays, including nested final_info masks."""
    return {
        key: info_at(value, index) if isinstance(value, dict) else value[index]
        for key, value in infos.items()
        if not key.startswith("_") and (f"_{key}" not in infos or infos[f"_{key}"][index])
    }


def transition_observations(observations, terminated, truncated, infos):
    """Bootstrap timeouts from the final observation, never the reset state."""
    result = observations.copy()
    done = terminated | truncated
    if done.any():
        if "_final_obs" not in infos or not np.all(infos["_final_obs"][done]):
            raise RuntimeError("SAME_STEP autoreset must provide every completed transition's final_obs")
        for index in np.flatnonzero(done):
            result[index] = infos["final_obs"][index]
    return result


def train(*, total_timesteps=1_000_000, output_dir="runs/door_curriculum_hand_parallel", seed=0,
          num_envs=8, n_steps=256, batch_size=512, epochs=10, hidden_dim=256,
          model_path=DEFAULT_MODEL, config=None, curriculum=None,
          start_stage="reach", home_probability=0.0, eval_freq=10_000,
          eval_episodes=10, threshold=0.8, patience=2, home_step=0.25,
          device="auto", torch_threads=1, start_method="spawn"):
    if min(total_timesteps, num_envs, n_steps, eval_freq, eval_episodes, patience, torch_threads) < 1:
        raise ValueError("Budgets, environment/thread counts and patience must be positive")
    if not 0 < threshold <= 1 or not 0 < home_step <= 1:
        raise ValueError("threshold and home_step must be in (0, 1]")
    core = config or DoorConfig(gamma=0.99)
    lesson = curriculum or CurriculumConfig()
    progress = CurriculumProgress(stage=start_stage, home_probability=home_probability)
    output = Path(output_dir).resolve()
    output.mkdir(parents=True, exist_ok=True)
    (output / "best").mkdir(exist_ok=True)
    checkpoint = output / "ppo_curriculum_hand.pt"
    previous_threads = torch.get_num_threads()
    torch.set_num_threads(torch_threads)
    torch.manual_seed(seed)
    np.random.seed(seed)
    envs = None
    try:
        print(f"Starting {num_envs} simulation workers ({start_method})...", flush=True)
        envs = make_parallel_envs(num_envs, model_path, core, lesson, start_stage,
                                  home_probability, start_method)
        specification = envs.call("specification")[0]
        agent = PPO(state_dim=envs.single_observation_space.shape[0],
                    action_dim=envs.single_action_space.shape[0], hidden_dim=hidden_dim,
                    hidden_depth=3, gamma=core.gamma, epochs=epochs,
                    batch_size=batch_size, device=device)
        agent_config = {
            "state_dim": agent.state_dim, "action_dim": agent.action_dim,
            "hidden_dim": hidden_dim, "hidden_depth": agent.hidden_depth,
            "actor_lr": 3e-4, "critic_lr": 1e-3, "lmbda": agent.lmbda,
            "epochs": epochs, "eps": agent.eps, "gamma": agent.gamma,
            "device": str(agent.device), "batch_size": batch_size,
        }
        run_config = {
            "agent": agent_config, "seed": seed, "final_eval_seed": final_eval_seed(seed),
            "requested_timesteps": total_timesteps, "num_envs": num_envs,
            "n_steps_per_env": n_steps, "rollout_size": n_steps * num_envs,
            "eval_freq": eval_freq, "eval_episodes": eval_episodes,
            "threshold": threshold, "patience": patience, "home_step": home_step,
            "torch_threads": torch_threads, "start_method": start_method,
        }
        steps = episodes = 0
        lengths = np.zeros(num_envs, dtype=np.int64)
        episode_returns = np.zeros(num_envs)
        next_eval = eval_freq
        best_scores = {}
        states, infos = envs.reset(seed=seed)
        last_infos = [info_at(infos, index) for index in range(num_envs)]
        last_truncated = np.zeros(num_envs, dtype=bool)
        print(f"Using {agent.device}; {num_envs} envs, network={[hidden_dim] * 3}, "
              f"rollout={n_steps} x {num_envs}, batch={batch_size}", flush=True)

        def save_progress():
            save_agent(agent, specification, checkpoint, agent_config, progress, steps)
            run_config.update(environment=specification, progress=asdict(progress), actual_timesteps=steps)
            (output / "config.json").write_text(json.dumps(run_config, indent=2) + "\n", encoding="utf-8")

        with (output / "episodes.csv").open("w", newline="", encoding="utf-8") as episode_file, \
                (output / "updates.csv").open("w", newline="", encoding="utf-8") as update_file, \
                (output / "curriculum.csv").open("w", newline="", encoding="utf-8") as curriculum_file:
            episode_log, update_log, curriculum_log = map(csv.writer, (episode_file, update_file, curriculum_file))
            episode_log.writerow([
                "episode", "env_id", "timesteps", "stage", "home_probability", "episode_start",
                "return", "length", "is_success", "full_task_success", "grasp_seen", "press_seen",
                "door_deg", "distance_m", "alignment", "failure_reason", "truncated", "cutoff",
            ])
            update_log.writerow(["timesteps", "stage", "home_probability", "actor_loss", "critic_loss",
                                 "samples", "sample_seconds", "sample_fps", "update_seconds"])
            curriculum_log.writerow([
                "timesteps", "stage", "home_probability", "success_rate", "home_episodes",
                "home_success_rate", "mean_return", "min_distance_m", "promoted",
            ])

            def log_episode(index, cutoff=""):
                nonlocal episodes
                if not lengths[index]:
                    return
                episodes += 1
                info = last_infos[index]
                episode_log.writerow([
                    episodes, index, steps, specification["stage"], specification["home_probability"], info["episode_start"],
                    episode_returns[index], lengths[index], info["is_success"], info["full_task_success"],
                    info["grasp_seen"], info["press_seen"], np.rad2deg(info["door_angle"]),
                    info["distance"], info["alignment"], info["failure_reason"], last_truncated[index], cutoff,
                ])
                lengths[index], episode_returns[index] = 0, 0.0

            while steps < total_timesteps and not progress.completed:
                rollout = {key: [] for key in ROLLOUT_KEYS}
                # A vector step always advances every environment, so the final
                # budget rounds up by at most num_envs - 1 total transitions.
                vector_steps = min(n_steps, (total_timesteps - steps + num_envs - 1) // num_envs)
                started = time.perf_counter()
                for _ in range(vector_steps):
                    actions, log_probs = agent.take_actions(states, return_log_prob=True)
                    observations, rewards, terminated, truncated, infos = envs.step(actions)
                    next_states = transition_observations(observations, terminated, truncated, infos)
                    for key, value in zip(ROLLOUT_KEYS, (states, actions, rewards, next_states,
                                                        terminated, truncated, log_probs)):
                        rollout[key].append(value)
                    states = observations
                    steps += num_envs
                    lengths += 1
                    episode_returns += rewards
                    last_truncated = truncated
                    for index in range(num_envs):
                        info = info_at(infos, index)
                        done = terminated[index] or truncated[index]
                        last_infos[index] = info["final_info"] if done else info
                        if done:
                            log_episode(index)
                sample_seconds = time.perf_counter() - started
                episode_file.flush()
                # PPO computes GAE on (time, env), then flattens for minibatches.
                started = time.perf_counter()
                losses = agent.update(rollout)
                update_seconds = time.perf_counter() - started
                samples = vector_steps * num_envs
                sample_fps = samples / sample_seconds
                update_log.writerow([steps, progress.stage, progress.home_probability,
                                     losses["actor_loss"], losses["critic_loss"], samples,
                                     sample_seconds, sample_fps, update_seconds])
                update_file.flush()
                print(f"steps={steps}/{total_timesteps}, {progress.stage}, home={progress.home_probability:.0%}; "
                      f"sample={sample_fps:.0f} steps/s, update={update_seconds:.2f}s; "
                      f"actor={losses['actor_loss']:.4f}, critic={losses['critic_loss']:.4f}", flush=True)

                if steps >= next_eval or steps >= total_timesteps:
                    metrics = evaluate_agent(agent, specification, episodes=eval_episodes, seed=seed + 10_000)
                    key = f"{progress.stage}_home_{progress.home_probability:.2f}"
                    score = (metrics["success_rate"], metrics["mean_return"])
                    if key not in best_scores or score > best_scores[key]:
                        best_scores[key] = score
                        save_agent(agent, specification, output / "best" / f"{key}.pt", agent_config, progress, steps)
                    previous = (progress.stage, progress.home_probability)
                    promoted = progress.advance(metrics["success_rate"], threshold, patience, home_step)
                    curriculum_log.writerow([
                        steps, *previous, metrics["success_rate"], metrics["home_episodes"],
                        metrics["home_success_rate"], metrics["mean_return"], metrics["mean_min_distance_m"], promoted,
                    ])
                    curriculum_file.flush()
                    (output / f"evaluation_{key}.json").write_text(json.dumps(metrics, indent=2) + "\n", encoding="utf-8")
                    print(f"Evaluation: success={metrics['success_rate']:.0%}; promoted={promoted}", flush=True)
                    if previous != (progress.stage, progress.home_probability):
                        # specification still describes the stage that produced these episodes.
                        for index in range(num_envs):
                            log_episode(index, cutoff="stage_switch")
                        envs.call("set_stage", progress.stage, progress.home_probability)
                        specification = envs.call("specification")[0]
                        states, infos = envs.reset()
                        last_infos = [info_at(infos, index) for index in range(num_envs)]
                        last_truncated = np.zeros(num_envs, dtype=bool)
                    next_eval = steps + eval_freq
                save_progress()
            for index in range(num_envs):
                log_episode(index, cutoff="curriculum_complete" if progress.completed else "budget_end")

        eval_seed = final_eval_seed(seed)
        print(f"Final evaluation seed: {eval_seed}", flush=True)
        metrics = evaluate_agent(agent, specification, episodes=eval_episodes, seed=eval_seed,
                                 stage="full", home_probability=1.0)
        metrics["curriculum_completed"] = progress.completed
        (output / "metrics.json").write_text(json.dumps(metrics, indent=2) + "\n", encoding="utf-8")
        print(f"Saved: {checkpoint}; full-task success={metrics['full_task_success_rate']:.0%}; "
              f"curriculum completed={progress.completed}", flush=True)
        return checkpoint
    finally:
        if envs is not None:
            envs.close(timeout=5)
        torch.set_num_threads(previous_threads)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--timesteps", type=int, default=1_000_000)
    parser.add_argument("--output-dir", type=Path, default=Path("runs/door_curriculum_hand_parallel"))
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--num-envs", type=int, default=8)
    parser.add_argument("--n-steps", type=int, default=256, help="Steps per environment per rollout.")
    parser.add_argument("--batch-size", type=int, default=512)
    parser.add_argument("--epochs", type=int, default=10)
    parser.add_argument("--hidden-dim", type=int, default=256, help="Width of each of the three hidden layers.")
    parser.add_argument("--torch-threads", type=int, default=1)
    parser.add_argument("--start-method", choices=("spawn", "forkserver"), default="spawn")
    parser.add_argument("--model", type=Path)
    parser.add_argument("--gamma", type=float, default=0.99)
    parser.add_argument("--max-steps", type=int, default=1000)
    parser.add_argument("--action-scale", type=float, default=0.03)
    parser.add_argument("--initial-noise", type=float, default=0.02)
    parser.add_argument("--eval-freq", type=int, default=10_000)
    parser.add_argument("--episodes", type=int, default=10)
    parser.add_argument("--success-threshold", type=float, default=0.8)
    parser.add_argument("--patience", type=int, default=2)
    parser.add_argument("--home-step", type=float, default=0.25)
    parser.add_argument("--stage", choices=STAGES)
    parser.add_argument("--home-probability", type=float)
    parser.add_argument("--device", choices=("auto", "cpu", "cuda"), default="auto")
    parser.add_argument("--evaluate", type=Path)
    parser.add_argument("--full-task", action="store_true")
    parser.add_argument("--render", action="store_true", help="Render evaluation only.")
    args = parser.parse_args()
    if args.evaluate:
        metrics = evaluate(args.evaluate, episodes=args.episodes, seed=args.seed, device=args.device,
                           render=args.render, model_path=args.model, full_task=args.full_task,
                           stage=args.stage, home_probability=args.home_probability)
        print(json.dumps(metrics, indent=2))
    else:
        if args.render:
            parser.error("--render is only available with --evaluate; parallel training is headless")
        train(total_timesteps=args.timesteps, output_dir=args.output_dir, seed=args.seed,
              num_envs=args.num_envs, n_steps=args.n_steps, batch_size=args.batch_size,
              epochs=args.epochs, hidden_dim=args.hidden_dim, torch_threads=args.torch_threads,
              start_method=args.start_method, device=args.device, model_path=args.model or DEFAULT_MODEL,
              config=DoorConfig(gamma=args.gamma, max_steps=args.max_steps,
                                action_scale=args.action_scale, initial_noise=args.initial_noise),
              start_stage=args.stage or "reach", home_probability=args.home_probability or 0.0,
              eval_freq=args.eval_freq, eval_episodes=args.episodes, threshold=args.success_threshold,
              patience=args.patience, home_step=args.home_step)


if __name__ == "__main__":
    main()
