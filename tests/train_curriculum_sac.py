"""SAC defaults from trainp2_sac, with train_curriculum_hand's rewards and starts.

uv run --no-sync python tests/train_curriculum_sac.py --timesteps 300000
uv run --no-sync python tests/train_curriculum_sac.py --evaluate runs/door_curriculum_sac/sac_curriculum.zip --full-task --render

Reuse DoorCurriculumEnv, CurriculumProgress and deterministic evaluation. SAC's
default MlpPolicy/optimizers/replay/warmup/update settings remain unchanged.
"""

from dataclasses import asdict, replace
import csv
import json
from pathlib import Path
import time

import torch
from stable_baselines3 import SAC
from stable_baselines3.common.logger import configure
from stable_baselines3.common.monitor import Monitor

from rlinit.algorithmhand.device import resolve_device
from rlinit.algorithmhand.door_curriculum import CurriculumProgress, evaluate_policy
from rlinit.algorithmhand.door_curriculum_env import CurriculumConfig, DoorCurriculumEnv, STAGES
from rlinit.algorithmhand.door_env import DEFAULT_MODEL, DoorConfig
from rlinit.algorithmhand.evaluation import final_eval_seed
from trainp2_sac import build_parser as sac_parser, sac_settings


FORMAT = "rlinit-door-curriculum-sac-v1"
DEFAULT_OUTPUT = "runs/door_curriculum_sac"
MONITOR_KEYS = ("stage", "episode_start", "is_success", "full_task_success",
                "distance", "alignment", "grasp_seen", "press_seen", "door_angle", "failure_reason")


def evaluate_agent(agent, specification, **kwargs):
    return evaluate_policy(lambda obs: agent.predict(obs, deterministic=True)[0], specification, **kwargs)


def load_agent(checkpoint, *, device="auto", render=False, model_path=None,
               full_task=False, stage=None, home_probability=None):
    agent = SAC.load(checkpoint, device=resolve_device(device))
    if getattr(agent, "curriculum_format", None) != FORMAT:
        raise ValueError("Expected a curriculum SAC .zip checkpoint from train_curriculum_sac.py")
    spec = agent.curriculum_environment
    if spec["task"] != "door_curriculum_v1":
        raise ValueError("Checkpoint does not describe a door curriculum task")
    env = DoorCurriculumEnv(
        model_path or spec["model_path"], config=DoorConfig(**spec["config"]),
        curriculum=CurriculumConfig(**spec["curriculum"]),
        stage="full" if full_task else (spec["stage"] if stage is None else stage),
        home_probability=1.0 if full_task else (spec["home_probability"] if home_probability is None else home_probability),
        render_mode="human" if render else None,
    )
    if (agent.observation_space != env.observation_space or agent.action_space != env.action_space
            or agent.gamma != env.base_config.gamma):
        env.close()
        raise ValueError("Checkpoint spaces/gamma do not match its curriculum environment")
    return agent, env


def evaluate(checkpoint, *, episodes=10, seed=20_000, device="auto", render=False,
             model_path=None, full_task=False, stage=None, home_probability=None):
    if episodes < 1:
        raise ValueError("episodes must be positive")
    agent, env = load_agent(checkpoint, device=device, model_path=model_path, full_task=full_task,
                            stage=stage, home_probability=home_probability)
    try:
        return evaluate_agent(agent, env.specification(), episodes=episodes, seed=seed, render=render)
    finally:
        env.close()


def train(*, total_timesteps=300_000, output_dir=DEFAULT_OUTPUT, seed=0,
          model_path=DEFAULT_MODEL, env_config=None, curriculum=None,
          start_stage="reach", home_probability=0.0, eval_freq=10_000, eval_episodes=10,
          threshold=0.8, patience=2, home_step=0.25,
          learning_rate=None, buffer_size=None, learning_starts=None, batch_size=None,
          train_freq=None, gradient_steps=None, gamma=None, tau=None, ent_coef=None,
          render=False, device="auto"):
    if any(not isinstance(value, int) or value < 1
           for value in (total_timesteps, eval_freq, eval_episodes, patience)):
        raise ValueError("Budgets, evaluation count and patience must be positive integers")
    if not 0 < threshold <= 1 or not 0 < home_step <= 1:
        raise ValueError("threshold and home_step must be in (0, 1]")
    if start_stage not in STAGES or not 0 <= home_probability <= 1:
        raise ValueError("Unknown stage or home_probability outside [0, 1]")
    overrides, actual = sac_settings(
        learning_rate=learning_rate, buffer_size=buffer_size, learning_starts=learning_starts,
        batch_size=batch_size, train_freq=train_freq, gradient_steps=gradient_steps,
        gamma=gamma, tau=tau, ent_coef=ent_coef,
    )
    core = replace(env_config or DoorConfig(), gamma=actual["gamma"])
    lesson = curriculum or CurriculumConfig()
    progress = CurriculumProgress(stage=start_stage, home_probability=home_probability)
    selected_device = resolve_device(device)
    output = Path(output_dir).resolve()
    output.mkdir(parents=True, exist_ok=True)
    (output / "best").mkdir(exist_ok=True)
    checkpoint, config_path = output / "sac_curriculum.zip", output / "config.json"
    env = logger = None
    serial = 0

    def make_env():
        nonlocal serial
        name = f"stage_{serial:02d}_{progress.stage}_home_{progress.home_probability:.2f}"
        serial += 1
        return Monitor(
            DoorCurriculumEnv(model_path, config=core, curriculum=lesson, stage=progress.stage,
                              home_probability=progress.home_probability,
                              render_mode="human" if render else None),
            str(output / name), info_keywords=MONITOR_KEYS,
        )

    try:
        env = make_env()
        # No policy_kwargs: preserve trainp2_sac.py's default Actor and twin Qs.
        agent = SAC("MlpPolicy", env, seed=seed, device=selected_device, verbose=1, **overrides)
        agent.curriculum_format = FORMAT
        logger = configure(str(output / "logs"), ["stdout", "csv"])
        agent.set_logger(logger)
        settings = dict(actual, net_arch=agent.policy.net_arch,
                        activation=agent.policy.activation_fn.__name__,
                        optimizer=agent.policy.optimizer_class.__name__,
                        n_critics=agent.critic.n_critics, target_entropy=agent.target_entropy)
        run_config = {
            "algorithm": "stable_baselines3.SAC", "agent": settings, "algorithm_overrides": overrides,
            "seed": seed, "final_eval_seed": final_eval_seed(seed), "requested_timesteps": total_timesteps,
            "eval_freq": eval_freq, "eval_episodes": eval_episodes,
            "threshold": threshold, "patience": patience, "home_step": home_step,
            "device": str(selected_device), "render": render, "n_envs": 1,
            "torch_threads": torch.get_num_threads(),
            "replay_on_stage_change": "retain; stage and home probability are part of each observation",
        }
        best_scores = {}
        started = time.perf_counter()

        def save_progress():
            agent.curriculum_environment = env.unwrapped.specification()
            agent.curriculum_progress = asdict(progress)
            agent.save(checkpoint)
            seconds = time.perf_counter() - started
            run_config.update(environment=agent.curriculum_environment, progress=asdict(progress),
                              actual_timesteps=agent.num_timesteps, gradient_updates=agent._n_updates,
                              replay_transitions=agent.replay_buffer.size(), training_seconds=seconds,
                              transitions_per_second=agent.num_timesteps / max(seconds, 1e-9))
            config_path.write_text(json.dumps(run_config, indent=2) + "\n", encoding="utf-8")

        print(f"SAC network={settings['net_arch']}, activation={settings['activation']}, "
              f"gamma={agent.gamma}, ent_coef={agent.ent_coef}; overrides={overrides}", flush=True)
        if total_timesteps <= agent.learning_starts:
            print("Budget does not exceed learning_starts: no SAC updates in this run.", flush=True)
        save_progress()
        with (output / "curriculum.csv").open("w", newline="", encoding="utf-8") as stream:
            writer = csv.writer(stream)
            writer.writerow(["timesteps", "stage", "home_probability", "success_rate", "home_episodes",
                             "home_success_rate", "mean_return", "min_distance_m", "promoted"])
            while agent.num_timesteps < total_timesteps and not progress.completed:
                print(f"Curriculum: {progress.stage}, home={progress.home_probability:.0%}", flush=True)
                # Evaluate/promote only after a complete SAC collection/update,
                # so the last old-stage transition is already in replay.
                agent.learn(total_timesteps=min(eval_freq, total_timesteps - agent.num_timesteps),
                            reset_num_timesteps=False)
                specification = env.unwrapped.specification()
                metrics = evaluate_agent(agent, specification, episodes=eval_episodes, seed=seed + 10_000)
                agent.logger.record("curriculum/stage", progress.stage)
                agent.logger.record("curriculum/home_probability", progress.home_probability)
                agent.logger.record("eval/success_rate", metrics["success_rate"])
                agent.logger.record("eval/mean_reward", metrics["mean_return"])
                agent.logger.record("eval/min_distance_m", metrics["mean_min_distance_m"])
                agent.logger.dump(step=agent.num_timesteps)
                key = f"{progress.stage}_home_{progress.home_probability:.2f}"
                score = (metrics["success_rate"], metrics["mean_return"])
                agent.curriculum_environment = specification
                agent.curriculum_progress = asdict(progress)
                if key not in best_scores or score > best_scores[key]:
                    best_scores[key] = score
                    agent.save(output / "best" / f"{key}.zip")
                previous = (progress.stage, progress.home_probability)
                promoted = progress.advance(metrics["success_rate"], threshold, patience, home_step)
                writer.writerow([agent.num_timesteps, *previous, metrics["success_rate"], metrics["home_episodes"],
                                 metrics["home_success_rate"], metrics["mean_return"],
                                 metrics["mean_min_distance_m"], promoted])
                stream.flush()
                (output / f"evaluation_{key}.json").write_text(json.dumps(metrics, indent=2) + "\n", encoding="utf-8")
                print(f"Evaluation: success={metrics['success_rate']:.0%}, "
                      f"min distance={metrics['mean_min_distance_m']:.3f} m; promoted={promoted}", flush=True)
                if previous != (progress.stage, progress.home_probability):
                    old = env
                    env = make_env()
                    # Keep the learner, optimizer states and contextual replay.
                    # Force a reset to prevent an old observation/new stage pair.
                    agent.set_env(env, force_reset=True)
                    agent.get_env().seed(seed + serial - 1)
                    old.close()
                save_progress()
        print(f"Final evaluation seed: {final_eval_seed(seed)}", flush=True)
        metrics = evaluate(checkpoint, episodes=eval_episodes, seed=final_eval_seed(seed),
                           device=selected_device, model_path=model_path, full_task=True)
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


def build_parser():
    parser = sac_parser()
    parser.description = __doc__
    parser.set_defaults(output_dir=Path(DEFAULT_OUTPUT))
    parser.add_argument("--success-threshold", type=float, default=0.8)
    parser.add_argument("--patience", type=int, default=2)
    parser.add_argument("--home-step", type=float, default=0.25)
    parser.add_argument("--stage", choices=STAGES)
    parser.add_argument("--home-probability", type=float)
    parser.add_argument("--full-task", action="store_true", help="Evaluate complete opening from home.")
    return parser


def main():
    args = build_parser().parse_args()
    if args.evaluate:
        metrics = evaluate(args.evaluate, episodes=args.episodes, seed=args.seed, device=args.device,
                           render=args.render, model_path=args.model, full_task=args.full_task,
                           stage=args.stage, home_probability=args.home_probability)
        print(json.dumps(metrics, indent=2))
    else:
        if args.full_task:
            raise ValueError("--full-task is an evaluation option; train with --stage full --home-probability 1")
        train(total_timesteps=args.timesteps, output_dir=args.output_dir, seed=args.seed,
              model_path=args.model or DEFAULT_MODEL,
              env_config=DoorConfig(max_steps=args.max_steps, action_scale=args.action_scale,
                                    initial_noise=args.initial_noise),
              start_stage=args.stage or "reach",
              home_probability=0.0 if args.home_probability is None else args.home_probability,
              eval_freq=args.eval_freq, eval_episodes=args.episodes, threshold=args.success_threshold,
              patience=args.patience, home_step=args.home_step,
              learning_rate=args.learning_rate, buffer_size=args.buffer_size,
              learning_starts=args.learning_starts, batch_size=args.batch_size,
              train_freq=args.train_freq, gradient_steps=args.gradient_steps,
              gamma=args.gamma, tau=args.tau, ent_coef=args.ent_coef, render=args.render, device=args.device)


if __name__ == "__main__":
    main()
