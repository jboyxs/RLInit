"""Two-phase cold-start SAC with its single standalone door environment.

The original SAC updates, reward switch, paired periodic snapshots and model
selection are retained. Only project dependencies and provenance paths moved.

  uv run --no-sync python train/train_sac_staged.py --device cuda --opening-device cpu
  uv run --no-sync python train/train_sac_staged.py --evaluate CHECKPOINT --render
"""
import argparse
from dataclasses import asdict, dataclass, replace
from functools import partial
import hashlib
import json
import multiprocessing
import os
from pathlib import Path
import pickle
import platform
import random
import shutil
import time
import zipfile

os.environ.setdefault("CUBLAS_WORKSPACE_CONFIG", ":4096:8")
import mujoco
import numpy as np
import stable_baselines3 as sb3
from stable_baselines3 import SAC
from stable_baselines3.common.buffers import ReplayBuffer, NStepReplayBuffer
from stable_baselines3.common.callbacks import BaseCallback
from stable_baselines3.common.evaluation import evaluate_policy
from stable_baselines3.common.logger import configure
from stable_baselines3.common.monitor import Monitor
from stable_baselines3.common.noise import OrnsteinUhlenbeckActionNoise
from stable_baselines3.common.vec_env import DummyVecEnv, SubprocVecEnv
import torch

from rlinit.algorithmhand.door_sac_learn_env import (
    TASK, DEFAULT_MODEL, DoorConfig, LearnRewardConfig, SACLearnDoorEnv, default_door_config,
)

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


class DoorSuccessEvalCallback(BaseCallback):
    def __init__(self, eval_env, *, best_model_save_path, log_path, eval_freq,
                 n_eval_episodes=5, deterministic=True, replay_save_freq=20_000,
                 eval_seed=10_000):
        super().__init__()
        self.eval_env = eval_env
        self.best_path = Path(best_model_save_path)
        self.log_path = Path(log_path)
        self.eval_freq = eval_freq
        self.n_eval_episodes = n_eval_episodes
        self.deterministic = deterministic
        self.replay_save_freq = replay_save_freq
        self.eval_seed = eval_seed
        self.last_replay_step = 0
        self.best_rank = (-1.0, -1.0, -1.0, -1.0, -1.0, -1.0, -np.inf)
        self.records = []

    def _init_callback(self):
        self.best_path.mkdir(parents=True, exist_ok=True)
        self.log_path.mkdir(parents=True, exist_ok=True)
        history = self.log_path / "task_evaluations.json"
        if history.exists():
            self.records = json.loads(history.read_text(encoding="utf-8"))
            if self.records:
                self.best_rank = max(self._rank(r) for r in self.records)

    @staticmethod
    def _rank(record):
        stage = record.get("training_stage", "full")
        return (record["success_rate"], record.get("stage_success_rate", record["success_rate"]),
                record["press_rate"] if stage in ("full", "press") else 0.0,
                record.get("mean_max_grasped_door_deg", 0.0) if stage == "full" else 0.0,
                record.get("mean_longest_grasp_steps", 0.0), record["grasp_rate"], record["mean_return"])

    def _on_step(self):
        if self.n_calls % 250 == 0 and (self.best_path.parent / "STOP_TRAINING").exists():
            print("STOP_TRAINING requested: saving final online SAC checkpoint.", flush=True)
            return False
        if self.n_calls % self.eval_freq:
            return True
        terminal = []
        nearest = []
        grasp_lengths = []
        grasped_openings = []
        current_nearest = float("inf")
        grasp_streak = longest_grasp = 0
        maximum_grasped_opening = 0.0

        if hasattr(self.eval_env, "seed"):
            self.eval_env.seed(self.eval_seed)
        else:
            self.eval_env.reset(seed=self.eval_seed)

        def collect(locals_, globals_):
            nonlocal current_nearest, grasp_streak, longest_grasp, maximum_grasped_opening
            info = locals_["info"]
            current_nearest = min(current_nearest, info["distance"])
            grasp_streak = grasp_streak + 1 if info["grasped"] else 0
            longest_grasp = max(longest_grasp, grasp_streak)
            if grasp_streak >= 5 and not info["failure_reason"]:
                maximum_grasped_opening = max(maximum_grasped_opening, info["door_angle"])
            if locals_["done"]:
                terminal.append(dict(info))
                nearest.append(current_nearest)
                grasp_lengths.append(longest_grasp)
                grasped_openings.append(maximum_grasped_opening)
                current_nearest = float("inf")
                grasp_streak = longest_grasp = 0
                maximum_grasped_opening = 0.0

        rewards, lengths = evaluate_policy(
            self.model, self.eval_env, n_eval_episodes=self.n_eval_episodes,
            deterministic=self.deterministic, return_episode_rewards=True, callback=collect,
        )
        success = float(np.mean([r["is_success"] for r in terminal]))
        stage_success = float(np.mean([r.get("stage_success", r["is_success"]) for r in terminal]))
        press = float(np.mean([r["press_seen"] for r in terminal]))
        grasp = float(np.mean([r["grasp_seen"] for r in terminal]))
        record = {
            "timesteps": self.num_timesteps, "success_rate": success,
            "stage_success_rate": stage_success,
            "training_stage": terminal[0].get("training_stage", "full"),
            "evaluation_reset_origins": sorted({r.get("reset_origin", "home") for r in terminal}),
            "press_rate": press, "grasp_rate": grasp, "mean_return": float(np.mean(rewards)),
            "mean_length": float(np.mean(lengths)),
            "mean_distance_m": float(np.mean([r["distance"] for r in terminal])),
            "mean_minimum_distance_m": float(np.mean(nearest)),
            "mean_final_alignment": float(np.mean([r["alignment"] for r in terminal])),
            "mean_longest_grasp_steps": float(np.mean(grasp_lengths)),
            "mean_max_grasped_door_deg": float(np.rad2deg(np.mean(grasped_openings))),
            "failures": sum(bool(r["failure_reason"]) for r in terminal),
        }
        rank = self._rank(record)
        record["rank"] = list(rank)
        self.records.append(record)
        (self.log_path / "task_evaluations.json").write_text(
            json.dumps(self.records, indent=2) + "\n", encoding="utf-8")
        for key in ("success_rate", "stage_success_rate", "press_rate", "grasp_rate", "mean_return", "mean_distance_m"):
            self.logger.record(f"eval/{key}", record[key])
        self.logger.dump(self.num_timesteps)
        output = self.best_path.parent
        curriculum = getattr(self.model, "door_training_reset_curriculum", None)
        if curriculum is not None:
            self.model.door_training_reset_curriculum = dict(
                curriculum, current_workers=self.training_env.env_method("training_reset_specification"))
        self.model.save(output / "latest_model.zip")
        if rank > self.best_rank:
            self.best_rank = rank
            self.model.save(self.best_path / "best_model.zip")
            (self.best_path / "selection.json").write_text(json.dumps(record, indent=2) + "\n", encoding="utf-8")
        if self.num_timesteps - self.last_replay_step >= self.replay_save_freq:
            self.model.save_replay_buffer(output / "latest_replay_buffer.pkl")
            self.last_replay_step = self.num_timesteps
        print(f"Task evaluation @{self.num_timesteps}: success={success:.0%}, press={press:.0%}, "
              f"grasp={grasp:.0%}, stage={record['training_stage']}:{stage_success:.0%}, "
              f"longest_grasp={record['mean_longest_grasp_steps']:.1f} steps, return={rank[-1]:.3f}", flush=True)
        return True


def press_progress_delta(obs, next_obs, terminal, *, reward_config, gamma, latch_index):
    obs, next_obs = np.asarray(obs), np.asarray(next_obs)
    if obs.shape != next_obs.shape or obs.shape[-1] != 80:
        raise ValueError("Expected matching 80-D observations")

    def values(state):
        fraction = np.clip(state[..., -2].astype(np.float64), 0, 1)
        angle = state[..., latch_index].astype(np.float64) * 1.3
        press = np.clip(angle / reward_config.press_target_angle, 0, 1)
        # Positive duration implies real bilateral contact in this task.
        continuing_press = reward_config.guided_press_weight * fraction * press
        potential = reward_config.press_potential * np.where(state[..., -1] > .5, 1., fraction * press)
        return continuing_press, potential

    _, previous = values(obs)
    old_press, following = values(next_obs)
    return -old_press + gamma * np.where(terminal, 0., following) - previous


def relabel_press_progress(buffer, *, reward_config, gamma, latch_index):
    if not reward_config.dense_guidance or reward_config.press_progress_only:
        raise ValueError("Expected original guided continuing-press rewards")
    if not buffer.handle_timeout_termination:
        raise ValueError("Timeout metadata is required for reward conversion")
    size = buffer.size()
    terminal = (buffer.dones[:size] > .5) & (buffer.timeouts[:size] < .5)
    delta = press_progress_delta(buffer.observations[:size], buffer.next_observations[:size], terminal,
        reward_config=reward_config, gamma=gamma, latch_index=latch_index)
    buffer.rewards[:size] += delta.astype(buffer.rewards.dtype)
    return int(size * buffer.n_envs)


def close_resume_segments(buffer):
    """A reset after resume must not join n-step returns across unrelated homes.

Mark unfinished last transitions as timeouts, preserving their recorded next
state and bootstrap value. Real terminal/success transitions stay unchanged.
"""
    if not buffer.handle_timeout_termination:
        raise ValueError("Resume requires timeout-aware replay")
    if not buffer.size():
        return 0
    last = (buffer.pos - 1) % buffer.buffer_size
    unfinished = buffer.dones[last] < .5
    buffer.dones[last, unfinished] = 1.
    buffer.timeouts[last, unfinished] = 1.
    return int(unfinished.sum())


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
            Path(__file__).resolve(), ROOT / "src/rlinit/algorithmhand/door_sac_learn_env.py")}
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


def inspect(checkpoint, output, *, episodes=5, seed=30000, device="cpu", render=False, loader=load_agent):
    torch.set_num_threads(1)
    agent, env = loader(checkpoint, device=device, render=render)
    records = []
    try:
        for offset in range(episodes):
            obs, _ = env.reset(seed=seed + offset)
            trajectory = []
            total = 0.
            grasp_steps = longest = door_hold_steps = 0
            pressed_step = None
            for step in range(1, env.config.max_steps + 1):
                action, _ = agent.predict(obs, deterministic=True)
                with torch.no_grad():
                    tensor, _ = agent.policy.obs_to_tensor(obs)
                    action_tensor = torch.as_tensor(action[None], device=agent.device)
                    critic_q = min(float(q.item()) for q in agent.critic(tensor, action_tensor))
                    _, latent_log_std, _ = agent.actor.get_action_dist_params(tensor)
                obs, reward, done, timeout, info = env.step(action)
                total += reward
                grasp_steps = grasp_steps + 1 if info["grasped"] and not info["failure_reason"] else 0
                longest = max(longest, grasp_steps)
                if (grasp_steps * env.dt + 1e-9 >= env.reward_config.stable_grasp_seconds
                        and info["handle_angle"] >= env.reward_config.press_target_angle):
                    pressed_step = pressed_step or step
                if (pressed_step is not None and info["grasped"] and not info["failure_reason"]
                        and info["door_angle"] >= env.config.success_angle):
                    door_hold_steps += 1
                else:
                    door_hold_steps = 0
                trajectory.append(dict(step=step, action=action.tolist(), reward=reward, critic_q=critic_q,
                    latent_log_std=latent_log_std.cpu().numpy().ravel().tolist(),
                    joint_tracking_error=(env.arm_target - env.data.qpos[env.qadr]).tolist(),
                    components=info["reward_components"], grasp=info["grasped"],
                    handle_deg=float(np.rad2deg(info["handle_angle"])),
                    door_deg=float(np.rad2deg(info["door_angle"])),
                    left_N=info["left_contact_N"], right_N=info["right_contact_N"],
                    penetration_m=info["penetration_m"], bad_contact_N=info["bad_contact_N"],
                    alignment=info["alignment"], center_m=info["center_distance_m"],
                    failure=info["failure_reason"]))
                if done or timeout:
                    break
            verified = (door_hold_steps * env.dt + 1e-9 >= env.config.success_hold_seconds
                        and not info["failure_reason"])
            if verified != info["is_success"]:
                raise RuntimeError("Environment and independent trace success disagree")
            finite_return = 0.
            for sample in reversed(trajectory):
                finite_return = sample["reward"] + agent.gamma * finite_return
                sample["finite_discounted_return"] = finite_return
            records.append(dict(seed=seed + offset, success=bool(verified), steps=step,
                return_=total, pressed_step=pressed_step, longest_grasp_steps=longest,
                initial_critic_q=trajectory[0]["critic_q"],
                finite_discounted_return=trajectory[0]["finite_discounted_return"],
                final_door_hold_seconds=door_hold_steps * env.dt, failure=info["failure_reason"],
                max_handle_deg=max(t["handle_deg"] for t in trajectory),
                max_door_deg=max(t["door_deg"] for t in trajectory), trajectory=trajectory))
    finally:
        env.close()
    result = dict(checkpoint=str(Path(checkpoint).resolve()),
        task=agent.door_environment["task"],
        trained_timesteps=agent.num_timesteps,
        checkpoint_sha256=hashlib.sha256(Path(checkpoint).read_bytes()).hexdigest(),
        episodes=episodes, first_seed=seed, deterministic=True, reset_origin="home",
        success_rate=float(np.mean([r["success"] for r in records])),
        critic_comparison="Soft stochastic SAC critic versus finite deterministic rollout return; not an equality test",
        records=records)
    destination = Path(output)
    destination.parent.mkdir(parents=True, exist_ok=True)
    destination.write_text(json.dumps(result, indent=2) + "\n", encoding="utf-8")
    print(json.dumps({k: v for k, v in result.items() if k != "records"}, indent=2))
    return result


def file_hash(path):
    with Path(path).open("rb") as stream:
        return hashlib.file_digest(stream, "sha256").hexdigest()


def periodic_pair(phase, *, timesteps, n_envs):
    """Reject stale/mismatched callback snapshots before reusing own experience.

    SB3 calls on_step before storing the latest vector transition and before its
    gradient updates. A normal final_model is therefore a different boundary.
    """
    phase = Path(phase)
    model = phase / "latest_model.zip"
    replay = phase / "latest_replay_buffer.pkl"
    with zipfile.ZipFile(model) as archive:
        if archive.testzip() is not None:
            raise ValueError("Incomplete periodic model snapshot")
        data = json.loads(archive.read("data"))
    if data["num_timesteps"] != timesteps:
        raise ValueError("Periodic model does not match the phase boundary")
    with replay.open("rb") as stream:
        buffer = pickle.load(stream)
    count = (buffer.buffer_size if buffer.full else buffer.pos) * buffer.n_envs
    if buffer.n_envs != n_envs or buffer.full or count != timesteps - n_envs:
        raise ValueError("Periodic replay does not match the model's callback boundary")
    metadata = dict(model=str(model), model_sha256=file_hash(model),
                    replay=str(replay), replay_sha256=file_hash(replay),
                    model_timesteps=data["num_timesteps"],
                    gradient_updates=data["_n_updates"], replay_transitions=count)
    return model, replay, metadata


def run(*, output_dir, total_timesteps=500000, press_progress_after=100000,
        seed=67, n_envs=4, device="auto", opening_device="cpu", eval_episodes=5):
    root = Path(output_dir).resolve()
    if root.exists() and (not root.is_dir() or any(root.iterdir())):
        raise ValueError("Output directory must be empty")
    # Both periodic files are written at 20k boundaries. Keeping the prefix
    # inside the replay capacity also makes the snapshot pairing check exact.
    if (n_envs not in (1, 2, 4, 8) or not 0 < press_progress_after < total_timesteps
            or press_progress_after % 20000 or press_progress_after > 300000
            or total_timesteps % n_envs or eval_episodes < 1):
        raise ValueError("Use 1/2/4/8 envs, a 20k-multiple prefix <=300k, and a larger vector-aligned budget")
    root.mkdir(parents=True, exist_ok=True)
    sources = [Path(__file__).resolve(),
               ROOT / "src/rlinit/algorithmhand/door_sac_learn_env.py",
               ROOT / "models/door_scene.xml", ROOT / "models/objects/door_lock_fixture.xml",
               ROOT / "models/stanford_tidybot/tidybot.xml"]
    source_hashes = {}
    for source in sources:
        relative = source.relative_to(ROOT)
        target = root / "source" / relative
        target.parent.mkdir(parents=True, exist_ok=True)
        shutil.copyfile(source, target)
        source_hashes[str(relative)] = file_hash(target)
    manifest = dict(mode="scratch-two-phase-SAC-periodic-pair", external_pretraining=False,
                    demonstrations=False, scripted_actions=False, seed=seed, n_envs=n_envs,
                    requested_timesteps=total_timesteps, press_progress_after=press_progress_after,
                    phase1_requested_device=device, phase2_requested_device=opening_device,
                    source_sha256=source_hashes, phase="learning_grasp_and_press")
    manifest_path = root / "pipeline.json"

    def save_manifest():
        manifest_path.write_text(json.dumps(manifest, indent=2) + "\n", encoding="utf-8")

    save_manifest()
    options = dict(seed=seed, n_envs=n_envs, eval_freq=10000, eval_episodes=eval_episodes,
                   settings=replace(SACLearnSettings(), n_steps=3),
                   reward_config=LearnRewardConfig.guided(), save_replay_buffer=True)
    train(output_dir=root / "phase1", total_timesteps=press_progress_after,
          device=device, **options)
    first_config = json.loads((root / "phase1/config.json").read_text(encoding="utf-8"))
    if first_config["actual_timesteps"] != press_progress_after:
        manifest.update(phase="interrupted_in_phase1", actual_timesteps=first_config["actual_timesteps"])
        save_manifest()
        return None
    model, replay, snapshot = periodic_pair(root / "phase1", timesteps=press_progress_after, n_envs=n_envs)
    manifest.update(phase="learning_opening", phase1_snapshot=snapshot)
    save_manifest()
    train(output_dir=root / "phase2", total_timesteps=total_timesteps - press_progress_after,
          device=opening_device, resume_checkpoint=model, replay_buffer_path=replay,
          upgrade_press_progress=True, **options)
    choices = []
    for phase in ("phase1", "phase2"):
        folder = root / phase
        selection = folder / "best/selection.json"
        metrics = json.loads(selection.read_text(encoding="utf-8")) if selection.exists() else {}
        candidate = folder / "best/best_model.zip"
        if not candidate.exists():
            candidate = folder / "final_model.zip"
        # Full task success wins. Prefer phase two on ties across reward scales.
        choices.append((metrics.get("success_rate", 0), phase == "phase2", phase, candidate, selection))
    _, _, selected_phase, selected, selection = max(choices, key=lambda x: x[:2])
    destination = root / "best/best_model.zip"
    destination.parent.mkdir(parents=True, exist_ok=True)
    shutil.copyfile(selected, destination)
    if selection.exists():
        shutil.copyfile(selection, destination.parent / "selection.json")
    metrics = json.loads((root / selected_phase / "evaluation.json").read_text(encoding="utf-8"))
    metrics["checkpoint"] = str(destination)
    (root / "evaluation.json").write_text(json.dumps(metrics, indent=2) + "\n", encoding="utf-8")
    actual = json.loads((root / "phase2/config.json").read_text(encoding="utf-8"))["actual_timesteps"]
    manifest.update(phase="training_finished" if actual >= total_timesteps else "interrupted_in_phase2",
                    actual_timesteps=actual, selected_phase=selected_phase,
                    selected_checkpoint=str(destination), selected_sha256=file_hash(destination))
    save_manifest()
    print(f"Selected own SAC policy: {destination}; independent success={metrics['success_rate']:.1%}")
    return destination


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--evaluate", type=Path)
    parser.add_argument("--output-dir", type=Path, default=ROOT / "runs/door_sac_staged")
    parser.add_argument("--timesteps", type=int, default=500000)
    parser.add_argument("--press-progress-after", type=int, default=100000)
    parser.add_argument("--seed", type=int)
    parser.add_argument("--n-envs", type=int, choices=(1, 2, 4, 8), default=4)
    parser.add_argument("--device", choices=("auto", "cpu", "cuda"))
    parser.add_argument("--opening-device", choices=("auto", "cpu", "cuda"), default="cpu")
    parser.add_argument("--episodes", type=int)
    parser.add_argument("--render", action="store_true", help="Render evaluation")
    parser.add_argument("--trace-output", type=Path, help="Save independent per-step evaluation traces")
    args = parser.parse_args()
    if args.trace_output is not None:
        if args.evaluate is None:
            parser.error("--trace-output requires --evaluate CHECKPOINT")
        inspect(args.evaluate, args.trace_output, episodes=args.episodes or 20,
                seed=args.seed if args.seed is not None else 20000,
                device=args.device or "cpu", render=args.render)
        return
    if args.evaluate:
        metrics = evaluate(args.evaluate, episodes=args.episodes or 20,
                           seed=args.seed if args.seed is not None else 20000,
                           device=args.device or "cpu", render=args.render)
        print(json.dumps(metrics, indent=2))
    else:
        if args.render:
            parser.error("Use --evaluate CHECKPOINT --render to inspect this parallel recipe")
        run(output_dir=args.output_dir, total_timesteps=args.timesteps,
            press_progress_after=args.press_progress_after, seed=args.seed if args.seed is not None else 67,
            n_envs=args.n_envs, device=args.device or "auto", opening_device=args.opening_device,
            eval_episodes=args.episodes or 5)


if __name__ == "__main__":
    multiprocessing.freeze_support()
    main()
