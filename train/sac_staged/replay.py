"""Uniform own-online replay conversion and resume boundary handling."""
import json
from pathlib import Path
import numpy as np
from stable_baselines3.common.buffers import ReplayBuffer, NStepReplayBuffer

from rlinit.algorithmhand.door_sac_learn_env import LearnRewardConfig
from .evaluation import _saved_reward, _reward_matches

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
