"""SB3 PPO curriculum: reach -> grasp -> press -> open -> full from home.

uv run --no-sync python tests/train_curriculum.py --timesteps 1000000
uv run --no-sync python tests/train_curriculum.py --evaluate runs/door_curriculum/ppo_curriculum.zip --full-task --render
"""

from dataclasses import asdict, replace
import argparse
import csv
import inspect
import json
from pathlib import Path

from stable_baselines3 import PPO
from stable_baselines3.common.logger import configure
from stable_baselines3.common.monitor import Monitor

from rlinit.algorithmhand.device import resolve_device
from rlinit.algorithmhand.door_env import DEFAULT_MODEL, DoorConfig
from rlinit.algorithmhand.door_curriculum_env import CurriculumConfig, DoorCurriculumEnv, STAGES
from rlinit.algorithmhand.door_curriculum import CurriculumProgress, evaluate_policy
from rlinit.algorithmhand.evaluation import final_eval_seed


FORMAT = "rlinit-door-curriculum-sb3-v1"


def evaluate_agent(agent, specification, *, episodes=10, seed=20_000,
                   render=False, stage=None, home_probability=None):
    return evaluate_policy(lambda obs: agent.predict(obs, deterministic=True)[0], specification,
                           episodes=episodes, seed=seed, render=render, stage=stage,
                           home_probability=home_probability)


def evaluate(checkpoint, *, episodes=10, seed=20_000, device="auto", render=False,
             full_task=False, stage=None, home_probability=None):
    agent = PPO.load(checkpoint, device=resolve_device(device))
    if getattr(agent, "curriculum_format", None) != FORMAT:
        raise ValueError("Expected a .zip checkpoint from train_curriculum.py")
    return evaluate_agent(agent, agent.curriculum_environment, episodes=episodes, seed=seed,
                          render=render, stage="full" if full_task else stage,
                          home_probability=1.0 if full_task else home_probability)


def train(*, total_timesteps=1_000_000, output_dir="runs/door_curriculum", seed=0,
          model_path=DEFAULT_MODEL, config=None, curriculum=None,
          start_stage="reach", home_probability=0.0, eval_freq=10_000,
          eval_episodes=10, threshold=0.8, patience=2, home_step=0.25,
          n_steps=None, batch_size=None, epochs=None, gamma=None,
          render=False, device="auto"):
    if min(total_timesteps, eval_freq, eval_episodes, patience) < 1:
        raise ValueError("Budgets, evaluation count and patience must be positive")
    if not 0 < threshold <= 1 or not 0 < home_step <= 1:
        raise ValueError("threshold and home_step must be in (0, 1]")
    overrides = {k: v for k, v in {"n_steps": n_steps, "batch_size": batch_size,
                                  "n_epochs": epochs, "gamma": gamma}.items() if v is not None}
    actual_gamma = gamma if gamma is not None else inspect.signature(PPO).parameters["gamma"].default
    core = replace(config or DoorConfig(), gamma=actual_gamma)
    lesson = curriculum or CurriculumConfig()
    progress = CurriculumProgress(stage=start_stage, home_probability=home_probability)
    output = Path(output_dir).resolve()
    output.mkdir(parents=True, exist_ok=True)
    (output / "best").mkdir(exist_ok=True)
    selected_device = resolve_device(device)
    logger = env = None
    serial = 0

    def make_env():
        nonlocal serial
        name = f"stage_{serial:02d}_{progress.stage}_home_{progress.home_probability:.2f}"
        serial += 1
        return Monitor(
            DoorCurriculumEnv(model_path, config=core, curriculum=lesson,
                              stage=progress.stage, home_probability=progress.home_probability,
                              render_mode="human" if render else None),
            str(output / name),
            info_keywords=("stage", "episode_start", "is_success", "full_task_success",
                           "distance", "alignment", "door_angle", "failure_reason"),
        )

    best_scores = {}
    config_path = output / "config.json"
    checkpoint = output / "ppo_curriculum.zip"
    try:
        env = make_env()
        agent = PPO("MlpPolicy", env, seed=seed, device=selected_device, verbose=1, **overrides)
        logger = configure(str(output / "logs"), ["stdout", "csv"])
        agent.set_logger(logger)
        agent.curriculum_format = FORMAT
        run_config = {
            "requested_timesteps": total_timesteps, "seed": seed,
            "final_eval_seed": final_eval_seed(seed), "eval_freq": eval_freq,
            "eval_episodes": eval_episodes, "threshold": threshold, "patience": patience,
            "home_step": home_step, "algorithm_overrides": overrides,
            "agent": {"net_arch": agent.policy.net_arch, "activation": agent.policy.activation_fn.__name__,
                      "gamma": agent.gamma, "n_steps": agent.n_steps, "batch_size": agent.batch_size,
                      "n_epochs": agent.n_epochs},
            "device": str(selected_device),
        }

        def save_progress():
            agent.curriculum_environment = env.unwrapped.specification()
            agent.curriculum_progress = asdict(progress)
            agent.save(checkpoint)
            run_config.update(environment=agent.curriculum_environment, progress=asdict(progress),
                              actual_timesteps=agent.num_timesteps)
            config_path.write_text(json.dumps(run_config, indent=2) + "\n", encoding="utf-8")

        with (output / "curriculum.csv").open("w", newline="", encoding="utf-8") as stream:
            writer = csv.writer(stream)
            writer.writerow(["timesteps", "stage", "home_probability", "success_rate", "home_episodes",
                             "home_success_rate", "mean_return", "min_distance_m", "promoted"])
            while agent.num_timesteps < total_timesteps and not progress.completed:
                print(f"Curriculum: {progress.stage}, home={progress.home_probability:.0%}", flush=True)
                agent.learn(total_timesteps=min(eval_freq, total_timesteps - agent.num_timesteps), reset_num_timesteps=False)
                agent.logger.dump(step=agent.num_timesteps)
                spec = env.unwrapped.specification()
                metrics = evaluate_agent(agent, spec, episodes=eval_episodes, seed=seed + 10_000)
                key = f"{progress.stage}_home_{progress.home_probability:.2f}"
                score = (metrics["success_rate"], metrics["mean_return"])
                agent.curriculum_environment = spec
                agent.curriculum_progress = asdict(progress)
                if key not in best_scores or score > best_scores[key]:
                    best_scores[key] = score
                    agent.save(output / "best" / f"{key}.zip")
                previous = (progress.stage, progress.home_probability)
                promoted = progress.advance(metrics["success_rate"], threshold, patience, home_step)
                writer.writerow([agent.num_timesteps, *previous, metrics["success_rate"], metrics["home_episodes"],
                                 metrics["home_success_rate"], metrics["mean_return"], metrics["mean_min_distance_m"], promoted])
                stream.flush()
                print(f"Evaluation: success={metrics['success_rate']:.0%}, "
                      f"min distance={metrics['mean_min_distance_m']:.3f} m; promoted={promoted}", flush=True)
                (output / f"evaluation_{key}.json").write_text(json.dumps(metrics, indent=2) + "\n", encoding="utf-8")
                if previous != (progress.stage, progress.home_probability):
                    old = env
                    env = make_env()
                    agent.set_env(env, force_reset=True)
                    old.close()
                save_progress()
            print(f"Final evaluation seed: {final_eval_seed(seed)}", flush=True)
            # Full-task metrics always start at home, even if the budget ended
            # before graduation. Subtask success never counts as final success.
            metrics = evaluate_agent(agent, env.unwrapped.specification(), episodes=eval_episodes,
                                     seed=final_eval_seed(seed), stage="full", home_probability=1.0)
            metrics["curriculum_completed"] = progress.completed
            (output / "metrics.json").write_text(json.dumps(metrics, indent=2) + "\n", encoding="utf-8")
            print(f"Saved: {checkpoint}; full-task success={metrics['full_task_success_rate']:.0%}; "
                  f"curriculum completed={progress.completed}", flush=True)
            return checkpoint
    finally:
        if logger is not None:
            logger.close()
        if env is not None:
            env.close()


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--timesteps", type=int, default=1_000_000)
    parser.add_argument("--output-dir", type=Path, default=Path("runs/door_curriculum"))
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--model", type=Path, default=DEFAULT_MODEL)
    parser.add_argument("--eval-freq", type=int, default=10_000)
    parser.add_argument("--episodes", type=int, default=10)
    parser.add_argument("--success-threshold", type=float, default=0.8)
    parser.add_argument("--patience", type=int, default=2)
    parser.add_argument("--home-step", type=float, default=0.25)
    parser.add_argument("--stage", choices=STAGES)
    parser.add_argument("--home-probability", type=float)
    parser.add_argument("--max-steps", type=int, default=1000)
    parser.add_argument("--initial-noise", type=float, default=0.02)
    parser.add_argument("--n-steps", type=int)
    parser.add_argument("--batch-size", type=int)
    parser.add_argument("--epochs", type=int)
    parser.add_argument("--gamma", type=float)
    parser.add_argument("--evaluate", type=Path)
    parser.add_argument("--full-task", action="store_true", help="Evaluate complete opening from home.")
    parser.add_argument("--render", action="store_true")
    parser.add_argument("--device", choices=("auto", "cpu", "cuda"), default="auto")
    args = parser.parse_args()
    if args.evaluate:
        metrics = evaluate(args.evaluate, episodes=args.episodes, seed=args.seed, device=args.device,
                           render=args.render, full_task=args.full_task, stage=args.stage,
                           home_probability=args.home_probability)
        print(json.dumps(metrics, indent=2))
    else:
        train(total_timesteps=args.timesteps, output_dir=args.output_dir, seed=args.seed, model_path=args.model,
              config=DoorConfig(max_steps=args.max_steps, initial_noise=args.initial_noise),
              start_stage=args.stage or "reach", home_probability=args.home_probability or 0.0,
              eval_freq=args.eval_freq, eval_episodes=args.episodes, threshold=args.success_threshold,
              patience=args.patience, home_step=args.home_step, n_steps=args.n_steps,
              batch_size=args.batch_size, epochs=args.epochs, gamma=args.gamma,
              render=args.render, device=args.device)


if __name__ == "__main__":
    main()
