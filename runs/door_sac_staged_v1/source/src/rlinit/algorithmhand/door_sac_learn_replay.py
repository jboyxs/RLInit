"""Uniform own-online SAC replay reward conversion; no action modification.

The 80-D observation contains latch angle, grasp duration and the latched
stable-press flag. These suffice to replace the old continuing press payment
with its discounted progress potential, without inventing transitions.
"""

import numpy as np


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
