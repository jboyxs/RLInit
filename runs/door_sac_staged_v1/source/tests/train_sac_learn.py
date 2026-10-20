"""Standalone home-only, raw-eight-action SAC training from random weights.

  uv run --no-sync python tests/train_sac_learn.py --timesteps 1200000
  uv run --no-sync python tests/train_sac_learn.py --evaluate runs/door_sac_learn/best/best_model.zip --episodes 100 --seed 70000

The first 400k transitions use one-step targets; three-step targets then reuse
only this run's own network and uniform online replay. No scripted controller,
demonstrations, modified reset states, or external pretrained weights are used.
"""

import argparse
from dataclasses import asdict, dataclass, replace
from functools import partial
import hashlib
import json
import multiprocessing
import os
from pathlib import Path
import platform
import random
import shutil
import time

os.environ.setdefault("CUBLAS_WORKSPACE_CONFIG", ":4096:8")
import mujoco
import numpy as np
import stable_baselines3 as sb3
from stable_baselines3 import SAC
from stable_baselines3.common.buffers import ReplayBuffer, NStepReplayBuffer
from stable_baselines3.common.logger import configure
from stable_baselines3.common.monitor import Monitor
from stable_baselines3.common.noise import OrnsteinUhlenbeckActionNoise
from stable_baselines3.common.vec_env import DummyVecEnv, SubprocVecEnv
import torch

from rlinit.algorithmhand.device import resolve_device
from rlinit.algorithmhand.door_env import DEFAULT_MODEL, DoorConfig
from rlinit.algorithmhand.door_sac_callback import DoorSuccessEvalCallback
from rlinit.algorithmhand.door_sac_learn_env import (
    TASK, LearnRewardConfig, SACLearnDoorEnv, default_door_config,
)
from rlinit.algorithmhand.door_sac_learn_replay import close_resume_segments, relabel_press_progress


FORMAT = "rlinit-sac-learn-v1"
ROOT = Path(__file__).resolve().parents[1]
MONITOR_KEYS = ("is_success", "grasp_seen", "press_seen", "door_angle", "failure_reason")


@dataclass(frozen=True)
class SACLearnSettings:
    learning_rate: float = 0.0001
    buffer_size: int = 300000
    batch_size: int = 256
    gamma: float = 0.995
    tau: float = 0.005
    ent_coef: float = 0.005
    n_steps: int = 1
    multistep_after: int = 400000
    final_n_steps: int = 3
    gripper_exploration_std: float = 0.3
    scratch_learning_starts: int = 5000

    def __post_init__(self):
        if self.multistep_after < 0 or self.final_n_steps < 1:
            raise ValueError("Invalid multistep schedule")
        for key in ("buffer_size", "batch_size", "n_steps"):
            if not isinstance(getattr(self, key), int) or getattr(self, key) < 1:
                raise ValueError(f"{key} must be a positive integer")
        if not isinstance(self.scratch_learning_starts, int) or self.scratch_learning_starts < 0:
            raise ValueError("scratch_learning_starts must be nonnegative")
        for key in ("learning_rate", "gamma", "tau", "ent_coef", "gripper_exploration_std"):
            if not np.isfinite(getattr(self, key)) or getattr(self, key) < 0:
                raise ValueError(f"{key} must be finite and nonnegative")
        if self.learning_rate <= 0 or not 0 < self.gamma <= 1 or not 0 < self.tau <= 1:
            raise ValueError("Invalid learning rate, gamma or tau")


def _file_hash(path):
    digest = hashlib.sha256()
    with Path(path).open("rb") as stream:
        for chunk in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def _saved_reward(spec):
    if spec.get("task") != TASK:
        raise ValueError("Only this standalone home-only SAC task is supported")
    return LearnRewardConfig(**spec["reward_config"])


def _reward_matches(source, target, upgrade_press_progress):
    if upgrade_press_progress:
        if not source.dense_guidance or source.press_progress_only:
            return False
        source = replace(source, press_progress_only=True)
    return source == target


def _validate_source(agent, env, *, allow_gamma_change=False, upgrade_press_progress=False):
    spec = agent.door_environment
    if spec.get("guidance_base_reward") != env.specification().get("guidance_base_reward"):
        raise ValueError("The original guidance reward changed since this checkpoint")
    if spec.get("model_source_sha256") != env.specification()["model_source_sha256"]:
        raise ValueError("Door or robot XML changed since this checkpoint")
    old_config, new_config = dict(spec["config"]), asdict(env.config)
    if allow_gamma_change:
        old_config.pop("gamma"); new_config.pop("gamma")
    if (old_config != new_config or not _reward_matches(_saved_reward(spec), env.reward_config, upgrade_press_progress)
            or agent.observation_space != env.observation_space or agent.action_space != env.action_space):
        raise ValueError("Checkpoint physics, observation or reward configuration differs")
    if getattr(agent, "door_training_reset_curriculum", None) is not None:
        raise ValueError("This recipe accepts home-only online checkpoints")
    if getattr(agent, "door_success_replay_fraction", 0):
        raise ValueError("This recipe uses uniform online replay")


def load_agent(checkpoint, *, device="cpu", model_path=DEFAULT_MODEL, render=False):
    agent = SAC.load(checkpoint, device=resolve_device(device), verbose=0)
    spec = getattr(agent, "door_environment", {})
    expected_format = FORMAT
    if getattr(agent, "door_task_format", None) != expected_format:
        raise ValueError("Unsupported SAC checkpoint format")
    reward = _saved_reward(spec)  # Reject controller, subtask and incompatible experiments.
    env = SACLearnDoorEnv(model_path, config=DoorConfig(**spec["config"]), reward_config=reward,
                         render_mode="human" if render else None)
    try:
        _validate_source(agent, env)
        if agent.gamma != env.config.gamma:
            raise ValueError("Checkpoint gamma differs from saved environment")
        expected_hash = spec.get("scene_sha256")
        if expected_hash is None:
            manifest = Path(checkpoint).parent / "manifest.json"
            if manifest.is_file():
                provenance = json.loads(manifest.read_text(encoding="utf-8"))
                if provenance.get("checkpoint_sha256") == hashlib.sha256(Path(checkpoint).read_bytes()).hexdigest():
                    expected_hash = provenance.get("source_sha256", {}).get("models/door_scene.xml")
        old_path = Path(spec["model_path"])
        if expected_hash is None and old_path.is_file():
            expected_hash = hashlib.sha256(old_path.read_bytes()).hexdigest()
        if expected_hash != env.specification()["scene_sha256"]:
            raise ValueError("Scene XML differs or its source hash cannot be verified")
        return agent, env
    except Exception:
        env.close()
        raise


def evaluate(checkpoint, *, episodes=20, seed=20000, device="cpu", model_path=DEFAULT_MODEL, render=False):
    if episodes < 1:
        raise ValueError("episodes must be positive")
    previous_threads = torch.get_num_threads()
    torch.set_num_threads(1)
    env = None
    records = []
    try:
        agent, env = load_agent(checkpoint, device=device, model_path=model_path, render=render)
        for offset in range(episodes):
            obs, _ = env.reset(seed=seed + offset)
            total, length = 0.0, 0
            while True:
                action, _ = agent.predict(obs, deterministic=True)
                obs, reward, done, timeout, info = env.step(action)
                total += reward
                length += 1
                if done or timeout:
                    break
            records.append(dict(seed=seed + offset, success=info["is_success"],
                                grasp=info["grasp_seen"], press=info["press_seen"],
                                pressed_with_stable_grasp=info["pressed_with_stable_grasp"],
                                failure=info["failure_reason"], steps=length, return_=total,
                                final_door_deg=float(np.rad2deg(info["door_angle"])),
                                success_hold_seconds=info["success_hold_seconds"], reset_origin="home"))
    finally:
        if env is not None:
            env.close()
        torch.set_num_threads(previous_threads)
    return dict(episodes=episodes, first_seed=seed, deterministic=True, reset_origins=["home"],
                success_rate=float(np.mean([r["success"] for r in records])),
                grasp_rate=float(np.mean([r["grasp"] for r in records])),
                press_rate=float(np.mean([r["press"] for r in records])), records=records)


def _worker(model_path, config, reward, monitor_path=None, *, render=False):
    torch.set_num_threads(1)
    return Monitor(SACLearnDoorEnv(model_path, config=config, reward_config=reward,
                                 render_mode="human" if render else None),
                   str(monitor_path) if monitor_path else None, info_keywords=MONITOR_KEYS)


def _load_replay(agent, path, source_spec, target_spec, *, upgrade_press_progress=False, latch_index=None):
    metadata = json.loads((Path(path).parent / "config.json").read_text(encoding="utf-8"))
    spec = metadata["environment"]
    physics, target = dict(spec["config"]), dict(target_spec["config"])
    physics.pop("gamma")
    target.pop("gamma")
    if (metadata.get("algorithm") != "stable_baselines3.SAC" or "demonstrations" in metadata
            or metadata.get("training_reset_curriculum") is not None
            or metadata.get("success_replay_fraction", 0)
            or physics != target or not _reward_matches(_saved_reward(spec), LearnRewardConfig(**target_spec["reward_config"]), upgrade_press_progress)
            or spec["model_path"] != source_spec["model_path"]):
        raise ValueError("Replay must be matching home-only, uniform, online SAC experience")
    agent.load_replay_buffer(path)
    buffer = agent.replay_buffer
    if (type(buffer) not in (ReplayBuffer, NStepReplayBuffer) or buffer.optimize_memory_usage
            or buffer.n_envs != agent.n_envs or buffer.buffer_size < agent.n_steps
            or buffer.observation_space != agent.observation_space or buffer.action_space != agent.action_space):
        raise ValueError("Replay type, vector count or spaces differ")
    relabelled = 0
    if upgrade_press_progress:
        relabelled = relabel_press_progress(buffer, reward_config=_saved_reward(spec),
                                            gamma=agent.gamma, latch_index=latch_index)
    boundaries = close_resume_segments(buffer)
    cls = NStepReplayBuffer if agent.n_steps > 1 else ReplayBuffer
    converted = cls.__new__(cls)
    converted.__dict__.update(buffer.__dict__)
    if agent.n_steps > 1:
        converted.n_steps, converted.gamma = agent.n_steps, agent.gamma
    agent.replay_buffer = converted
    agent.replay_buffer_class = cls
    agent.replay_buffer_kwargs = dict(n_steps=agent.n_steps, gamma=agent.gamma) if agent.n_steps > 1 else {}
    return dict(relabelled_transitions=relabelled, resumed_timeout_boundaries=boundaries)


class LearnEvalCallback(DoorSuccessEvalCallback):
    """One SAC run: preserve its own uniform replay when switching return horizon."""

    def _on_step(self):
        recipe = SACLearnSettings(**self.model.open_settings)
        if self.num_timesteps >= recipe.multistep_after and self.model.n_steps != recipe.final_n_steps:
            buffer = self.model.replay_buffer
            cls = NStepReplayBuffer if recipe.final_n_steps > 1 else ReplayBuffer
            converted = cls.__new__(cls)
            converted.__dict__.update(buffer.__dict__)
            converted.n_steps, converted.gamma = recipe.final_n_steps, self.model.gamma
            self.model.replay_buffer = converted
            self.model.replay_buffer_class = cls
            self.model.replay_buffer_kwargs = dict(n_steps=recipe.final_n_steps, gamma=self.model.gamma) if recipe.final_n_steps > 1 else {}
            self.model.n_steps = recipe.final_n_steps
            print(f"Own online SAC replay retained; return horizon -> {recipe.final_n_steps} at {self.num_timesteps}", flush=True)
        for info in self.locals.get("infos", []):
            if "episode" in info:
                record = {key: info[key] for key in ("is_success", "press_seen", "grasp_seen", "failure_reason", "door_angle", "pressed_with_stable_grasp", "stable_grasp_seconds")}
                record.update(timesteps=self.num_timesteps, episode=info["episode"])
                with (self.best_path.parent / "episodes.jsonl").open("a", encoding="utf-8") as stream:
                    stream.write(json.dumps(record) + "\n")
        return super()._on_step()


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
        run["recipe_sha256"] = {str(path.relative_to(ROOT)): _file_hash(path) for path in (
            Path(__file__).resolve(), ROOT / "src/rlinit/algorithmhand/door_sac_learn_env.py",
            ROOT / "src/rlinit/algorithmhand/door_sac_open_env.py",
            ROOT / "src/rlinit/algorithmhand/door_sac_learn_replay.py",
            ROOT / "src/rlinit/algorithmhand/door_env.py", ROOT / "src/rlinit/algorithmhand/door_direct_env.py",
            ROOT / "src/rlinit/algorithmhand/door_sac_callback.py")}
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


def train_two_phases(*, output_dir, total_timesteps, press_progress_after, **kwargs):
    """A complete random-initialization run, retaining only its own experience."""
    root = Path(output_dir).resolve()
    if root.exists() and (not root.is_dir() or any(root.iterdir())):
        raise ValueError("Pipeline output directory must be empty")
    if not 0 < press_progress_after < total_timesteps:
        raise ValueError("Press-progress switch must fall inside the training budget")
    for key in ("resume_checkpoint", "initialize_from", "replay_buffer_path", "upgrade_press_progress"):
        if kwargs.pop(key, None):
            raise ValueError("The two-phase pipeline starts from random weights")
    if not kwargs.get("save_replay_buffer", True):
        raise ValueError("Two-phase training requires its own replay")
    reward = kwargs.get("reward_config") or LearnRewardConfig.guided()
    if not reward.dense_guidance or reward.press_progress_only:
        raise ValueError("Two-phase training starts with guided pressing rewards")
    kwargs["reward_config"] = reward
    kwargs["settings"] = kwargs.get("settings") or replace(SACLearnSettings(), n_steps=3)
    root.mkdir(parents=True, exist_ok=True)
    manifest = dict(mode="scratch-two-phase-SAC", external_pretraining=False, demonstrations=False,
                    requested_timesteps=total_timesteps, press_progress_after=press_progress_after,
                    phase="acquiring_grasp_and_press")
    manifest_path = root / "pipeline.json"
    manifest_path.write_text(json.dumps(manifest, indent=2) + "\n", encoding="utf-8")
    first = train(output_dir=root / "phase1", total_timesteps=press_progress_after, **kwargs)
    acquired = json.loads((root / "phase1/config.json").read_text(encoding="utf-8"))["actual_timesteps"]
    remaining = total_timesteps - acquired
    if remaining < 1:
        raise ValueError("The first phase consumed the budget after vector-step rounding")
    manifest.update(phase="learning_opening", phase1_timesteps=acquired,
                    phase1_checkpoint_sha256=_file_hash(first))
    manifest_path.write_text(json.dumps(manifest, indent=2) + "\n", encoding="utf-8")
    final = train(output_dir=root / "phase2", total_timesteps=remaining,
                  resume_checkpoint=first, replay_buffer_path=root / "phase1/replay_buffer.pkl",
                  upgrade_press_progress=True, **kwargs)
    selected = root / "phase2/best/best_model.zip"
    selected = selected if selected.is_file() else final
    destination = root / "best/best_model.zip"
    destination.parent.mkdir(parents=True, exist_ok=True)
    shutil.copyfile(selected, destination)
    selection = root / "phase2/best/selection.json"
    if selection.is_file():
        shutil.copyfile(selection, destination.parent / "selection.json")
    metrics = json.loads((root / "phase2/evaluation.json").read_text(encoding="utf-8"))
    metrics["checkpoint"] = str(destination)
    (root / "evaluation.json").write_text(json.dumps(metrics, indent=2) + "\n", encoding="utf-8")
    manifest.update(phase="training_finished", selected_checkpoint=str(destination),
                    selected_sha256=_file_hash(destination),
                    actual_timesteps=json.loads((root / "phase2/config.json").read_text(encoding="utf-8"))["actual_timesteps"])
    manifest_path.write_text(json.dumps(manifest, indent=2) + "\n", encoding="utf-8")
    print(f"Two-phase selected policy: {destination}; independent success={metrics['success_rate']:.1%}")
    return destination


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    mode = parser.add_mutually_exclusive_group()
    mode.add_argument("--evaluate", type=Path)
    mode.add_argument("--resume", type=Path)
    parser.add_argument("--replay-buffer", type=Path)
    parser.add_argument("--output-dir", type=Path, default=ROOT / "runs/door_sac_learn")
    parser.add_argument("--timesteps", type=int, default=1200000)
    parser.add_argument("--seed", type=int)
    parser.add_argument("--n-envs", type=int, help="Training environments (default: 4, or 1 with --render)")
    parser.add_argument("--eval-freq", type=int, default=10000)
    parser.add_argument("--episodes", type=int)
    parser.add_argument("--model", type=Path, default=DEFAULT_MODEL)
    parser.add_argument("--device", choices=("auto", "cpu", "cuda"))
    parser.add_argument("--render", action="store_true",
                        help="Watch training (single environment) or evaluation; requires a graphical desktop")
    parser.add_argument("--no-save-replay", action="store_true")
    parser.add_argument("--reward-profile", choices=("linear", "potential", "guided"), default=None,
                        help="New scratch run: linear distance costs (default), or original potential pilot")
    parser.add_argument("--n-steps", type=int, choices=(1, 3),
                        help="Scratch SAC return horizon (default: 1, switching to 3 at 400k)")
    parser.add_argument("--upgrade-press-progress", action="store_true",
                        help="Resume own guided SAC, recomputing its own replay for progress-only pressing")
    parser.add_argument("--press-progress-after", type=int,
                        help="Start random three-step guided SAC, then switch own replay to press progress at this step")
    args = parser.parse_args()
    torch.set_num_threads(1)
    if args.evaluate:
        if args.replay_buffer is not None:
            parser.error("Evaluation does not use a replay buffer")
        metrics = evaluate(args.evaluate, episodes=args.episodes or 20, seed=args.seed if args.seed is not None else 20000,
                           model_path=args.model, device=args.device or "cpu", render=args.render)
        print(json.dumps(metrics, indent=2))
    else:
        if args.resume and args.reward_profile is not None:
            parser.error("Resume preserves the checkpoint reward; omit --reward-profile")
        if args.resume and args.n_steps is not None:
            parser.error("Resume preserves saved SAC settings; omit --n-steps")
        if args.render and args.n_envs not in (None, 1):
            parser.error("Training rendering requires --n-envs 1; omit --render for parallel training")
        options = dict(output_dir=args.output_dir, total_timesteps=args.timesteps, seed=args.seed if args.seed is not None else 67,
              n_envs=args.n_envs, device=args.device or "auto", model_path=args.model, eval_freq=args.eval_freq,
              eval_episodes=args.episodes or 5, resume_checkpoint=args.resume,
              settings=replace(SACLearnSettings(), n_steps=args.n_steps) if args.n_steps else None,
              upgrade_press_progress=args.upgrade_press_progress,
              replay_buffer_path=args.replay_buffer, save_replay_buffer=not args.no_save_replay, render=args.render,
              reward_config=(LearnRewardConfig() if args.reward_profile == "potential" else
                             LearnRewardConfig.guided() if args.reward_profile == "guided" or
                             (args.reward_profile is None and args.press_progress_after is not None) else LearnRewardConfig.linear())
              if not args.resume else None)
        if args.press_progress_after is not None:
            train_two_phases(press_progress_after=args.press_progress_after, **options)
        else:
            train(**options)


if __name__ == "__main__":
    multiprocessing.freeze_support()
    main()
