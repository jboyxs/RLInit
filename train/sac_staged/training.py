"""One SAC stage: environment collection, updates, saving and final evaluation."""
from dataclasses import asdict, replace
from functools import partial
import json
from pathlib import Path
import platform
import random
import time

import mujoco
import numpy as np
import stable_baselines3 as sb3
from stable_baselines3 import SAC
from stable_baselines3.common.logger import configure
from stable_baselines3.common.monitor import Monitor
from stable_baselines3.common.noise import OrnsteinUhlenbeckActionNoise
from stable_baselines3.common.vec_env import DummyVecEnv, SubprocVecEnv
import torch

from rlinit.algorithmhand.door_sac_learn_env import DEFAULT_MODEL, LearnRewardConfig, SACLearnDoorEnv, default_door_config
from .callbacks import LearnEvalCallback
from .config import ROOT, FORMAT, MONITOR_KEYS, SACLearnSettings, resolve_device, _file_hash, recipe_sources
from .evaluation import load_agent, _validate_source, evaluate
from .replay import _load_replay

def _worker(model_path, config, reward, monitor_path=None, *, render=False):
    torch.set_num_threads(1)
    return Monitor(SACLearnDoorEnv(model_path, config=config, reward_config=reward,
                                 render_mode="human" if render else None),
                   str(monitor_path) if monitor_path else None, info_keywords=MONITOR_KEYS)


def train(*, output_dir=ROOT / "runs/door_sac_learn", total_timesteps=1200000, seed=67, n_envs=None,
          device="auto", model_path=DEFAULT_MODEL, settings=None, env_config=None, reward_config=None,
          eval_freq=10000, eval_episodes=5, initialize_from=None, replay_buffer_path=None,
          resume_checkpoint=None, save_replay_buffer=True, render=False, upgrade_press_progress=False):
    if initialize_from is not None:
        raise ValueError("Start from random weights, or resume this experiment with its own replay")
    n_envs = n_envs if n_envs is not None else (1 if render else 4)
    if min(total_timesteps, n_envs, eval_freq, eval_episodes) < 1:
        raise ValueError("Step counts, environments and episodes must be positive")
    if render and n_envs != 1:
        raise ValueError("Training rendering requires --n-envs 1; omit --render for parallel training")
    if initialize_from is not None and resume_checkpoint is not None:
        raise ValueError("Choose initialization or resume")
    if replay_buffer_path is not None and initialize_from is None and resume_checkpoint is None:
        raise ValueError("Replay requires an online initialization or resume checkpoint")
    if resume_checkpoint is not None and replay_buffer_path is None:
        raise ValueError("Resume requires the saved replay buffer")
    if upgrade_press_progress and resume_checkpoint is None:
        raise ValueError("Press-progress upgrade requires this task's own checkpoint and replay")
    output = Path(output_dir).resolve()
    if output.exists() and (not output.is_dir() or any(output.iterdir())):
        raise ValueError("Output directory is not empty; choose a new directory for this run")
    selected = resolve_device(device)
    source = None
    if initialize_from is not None or resume_checkpoint is not None:
        source, source_env = load_agent(initialize_from or resume_checkpoint, device=selected, model_path=model_path)
        try:
            if resume_checkpoint is not None:
                if source.door_task_format != FORMAT:
                    raise ValueError("Resume needs a checkpoint from train_sac_learn.py; use --init-from for older policies")
                settings = settings or SACLearnSettings(**source.open_settings)
                env_config = env_config or source_env.config
                reward_config = reward_config or source_env.reward_config
        finally:
            source_env.close()
    recipe = settings or SACLearnSettings()
    config, reward = env_config or default_door_config(), reward_config or LearnRewardConfig.linear()
    if upgrade_press_progress:
        if not reward.dense_guidance or reward.press_progress_only:
            raise ValueError("Upgrade requires original guided continuing-press rewards")
        reward = replace(reward, press_progress_only=True)
    if config.gamma != recipe.gamma or recipe.buffer_size // n_envs < recipe.n_steps:
        raise ValueError("Environment gamma or replay capacity does not match SAC settings")
    probe = SACLearnDoorEnv(model_path, config=config, reward_config=reward)
    try:
        if source is not None:
            _validate_source(source, probe, allow_gamma_change=initialize_from is not None,
                             upgrade_press_progress=upgrade_press_progress)
        specification = probe.specification()
        latch_index = probe.latch
    finally:
        probe.close()
    previous_threads = torch.get_num_threads()
    torch.set_num_threads(1)
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)
    torch.backends.cudnn.benchmark = False
    torch.backends.cudnn.deterministic = True
    output.mkdir(parents=True, exist_ok=True)
    env = eval_env = logger = None
    try:
        factories = [partial(_worker, str(Path(model_path).resolve()), config, reward, output / f"worker_{i}",
                             render=render)
                     for i in range(n_envs)]
        env = DummyVecEnv(factories) if n_envs == 1 else SubprocVecEnv(factories, start_method="spawn")
        eval_env = DummyVecEnv([partial(_worker, str(Path(model_path).resolve()), config, reward)])
        sigma = np.zeros(8)
        sigma[7] = np.sqrt(2) * recipe.gripper_exploration_std
        noise = OrnsteinUhlenbeckActionNoise(np.zeros(8), sigma, theta=1.0, dt=0.02)
        learning_starts = recipe.scratch_learning_starts if source is None else 0
        if resume_checkpoint is None:
            agent = SAC("MlpPolicy", env, seed=seed, device=selected, verbose=1,
                        learning_rate=recipe.learning_rate, buffer_size=recipe.buffer_size,
                        batch_size=recipe.batch_size, learning_starts=learning_starts, train_freq=1,
                        gradient_steps=-1, gamma=recipe.gamma, tau=recipe.tau, ent_coef=recipe.ent_coef,
                        n_steps=recipe.n_steps, action_noise=noise,
                        policy_kwargs=dict(net_arch=[256, 256], activation_fn=torch.nn.ReLU))
            if source is not None:
                agent.policy.load_state_dict(source.policy.state_dict())
        else:
            if recipe != SACLearnSettings(**source.open_settings):
                raise ValueError("Resume must preserve saved SAC settings")
            agent = SAC.load(resume_checkpoint, env=env, seed=seed, device=selected, action_noise=noise)
        replay_conversion = dict(relabelled_transitions=0, resumed_timeout_boundaries=0)
        if replay_buffer_path is not None:
            replay_conversion = _load_replay(agent, replay_buffer_path, source.door_environment, specification,
                                            upgrade_press_progress=upgrade_press_progress, latch_index=latch_index)
        source_timesteps = source.num_timesteps if source is not None else None
        del source
        agent.door_task_format, agent.door_environment = FORMAT, specification
        agent.open_settings = asdict(recipe)
        agent.door_training_reset_curriculum = None
        agent.door_success_replay_fraction = 0.0
        logger = configure(str(output / "logs"), ["stdout", "csv"])
        agent.set_logger(logger)
        run = dict(algorithm="stable_baselines3.SAC", environment=specification, settings=asdict(recipe),
                   mode="resume" if resume_checkpoint else "online-initialization" if initialize_from else "scratch",
                   seed=seed, worker_seeds=[seed + i for i in range(n_envs)], eval_seed=10000,
                   n_envs=n_envs, learning_starts=agent.learning_starts, requested_timesteps=total_timesteps,
                   save_replay_buffer=save_replay_buffer, render=render,
                   train_freq=1, gradient_steps=-1, net_arch=[256, 256], activation="ReLU", torch_threads=1,
                   eval_freq=eval_freq, eval_episodes=eval_episodes,
                   initialization=str(initialize_from) if initialize_from else None,
                   source_checkpoint_timesteps=source_timesteps,
                   resume=str(resume_checkpoint) if resume_checkpoint else None,
                   replay_source=str(replay_buffer_path) if replay_buffer_path else None,
                   press_progress_upgrade=upgrade_press_progress, replay_conversion=replay_conversion,
                   training_reset_curriculum=None, success_replay_fraction=0.0,
                   versions=dict(python=platform.python_version(), sb3=sb3.__version__, torch=torch.__version__,
                                 mujoco=mujoco.__version__, numpy=np.__version__), device=str(selected))
        run["input_sha256"] = {name: _file_hash(path) for name, path in (
            ("initialization", initialize_from), ("resume", resume_checkpoint), ("replay", replay_buffer_path))
            if path is not None}
        run["recipe_sha256"] = {str(path.relative_to(ROOT)): _file_hash(path) for path in recipe_sources()}
        (output / "config.json").write_text(json.dumps(run, indent=2) + "\n", encoding="utf-8")
        callback = LearnEvalCallback(eval_env, best_model_save_path=output / "best", log_path=output / "eval",
                                          eval_freq=max(1, (eval_freq + n_envs - 1) // n_envs),
                                          n_eval_episodes=eval_episodes, eval_seed=10000,
                                          replay_save_freq=20000 if save_replay_buffer else float("inf"))
        if resume_checkpoint is None:
            agent.save(output / "initial_random_model.zip")
        started = time.perf_counter()
        if agent.num_timesteps + total_timesteps <= agent.learning_starts:
            print("This budget collects initial exploration only; SAC updates start after learning_starts.")
        agent.learn(total_timesteps=total_timesteps, callback=callback, reset_num_timesteps=resume_checkpoint is None)
        final = output / "final_model.zip"
        agent.save(final)
        if save_replay_buffer:
            agent.save_replay_buffer(output / "replay_buffer.pkl")
        run.update(actual_timesteps=agent.num_timesteps, gradient_updates=agent._n_updates,
                   training_seconds=time.perf_counter() - started)
        (output / "config.json").write_text(json.dumps(run, indent=2) + "\n", encoding="utf-8")
    finally:
        if logger is not None:
            logger.close()
        if env is not None:
            env.close()
        if eval_env is not None:
            eval_env.close()
        torch.set_num_threads(previous_threads)
    best = output / "best/best_model.zip"
    candidate = best if best.is_file() else final
    metrics = evaluate(candidate, episodes=eval_episodes, seed=20000, device="cpu", model_path=model_path)
    metrics["checkpoint"] = str(candidate)
    (output / "evaluation.json").write_text(json.dumps(metrics, indent=2) + "\n", encoding="utf-8")
    print(f"Selected policy: {candidate}; independent success={metrics['success_rate']:.1%}")
    return final
