"""Reward-based replay sampling of the learner's own complete home episodes.

Only replay sampling changes. Actor and critic updates remain SB3 SAC, with
no behavior cloning, expert actions, or action-label objective.
"""

from collections import deque

import numpy as np
from stable_baselines3.common.buffers import ReplayBuffer, NStepReplayBuffer
from stable_baselines3.common.type_aliases import ReplayBufferSamples


def configure_multistep_replay(agent):
    """Use SB3's n-step targets with the same chronological online transitions.

    Loaded replay files can be ordinary one-step buffers. Reuse their arrays
    without altering rewards, observations, actions or vector/ring ordering.
    SB3 handles terminations, timeouts and the unfinished newest trajectory.
    """
    steps = agent.n_steps
    buffer = agent.replay_buffer
    if steps == 1 and not isinstance(buffer, NStepReplayBuffer):
        return
    if type(buffer) not in (ReplayBuffer, NStepReplayBuffer):
        raise ValueError("Multi-step replay requires standard uniform online replay")
    if buffer.optimize_memory_usage or buffer.buffer_size < steps:
        raise ValueError("Multi-step replay needs separate next observations and sufficient capacity")
    cls = NStepReplayBuffer if steps > 1 else ReplayBuffer
    converted = cls.__new__(cls)
    converted.__dict__.update(buffer.__dict__)
    settings = dict(agent.replay_buffer_kwargs or {})
    settings.pop("n_steps", None)
    settings.pop("gamma", None)
    if steps > 1:
        converted.n_steps, converted.gamma = steps, agent.gamma
        settings.update(n_steps=steps, gamma=agent.gamma)
    agent.replay_buffer = converted
    agent.replay_buffer_class = cls
    agent.replay_buffer_kwargs = settings


class OwnSuccessReplayBuffer(ReplayBuffer):
    def __init__(self, *args, success_fraction=0.25, door_index=15,
                 door_threshold=0.75, grasp_threshold=0.025, **kwargs):
        super().__init__(*args, **kwargs)
        self.configure(success_fraction, door_index, door_threshold, grasp_threshold, rebuild=True)

    @classmethod
    def from_buffer(cls, buffer, **settings):
        if isinstance(buffer, cls):
            buffer.configure(**settings, rebuild=False)
            return buffer
        result = cls.__new__(cls)
        result.__dict__.update(buffer.__dict__)
        result.configure(**settings, rebuild=True)
        return result

    def configure(self, success_fraction, door_index, door_threshold, grasp_threshold, *, rebuild):
        if not np.isfinite(success_fraction) or not 0 <= success_fraction <= 0.5:
            raise ValueError("success_fraction must be in [0, 0.5]")
        if self.observation_space.shape != (78,) or self.action_dim != 8 or self.optimize_memory_usage:
            raise ValueError("Own-success replay requires the precision joint task with separate next observations")
        self.success_fraction = float(success_fraction)
        self.door_index = int(door_index)
        self.door_threshold = float(door_threshold)
        self.grasp_threshold = float(grasp_threshold)
        self._pending = [deque(maxlen=self.buffer_size) for _ in range(self.n_envs)]
        if rebuild:
            self.success_mask = np.zeros((self.buffer_size, self.n_envs), dtype=bool)
            self.successful_episodes = 0
            count = self.buffer_size if self.full else self.pos
            order = (np.arange(count) + (self.pos if self.full else 0)) % self.buffer_size
            for environment in range(self.n_envs):
                episode = []
                for index in order:
                    episode.append(int(index))
                    if self.dones[index, environment]:
                        observation = self.next_observations[index, environment]
                        # The original success timer reaches 1 only after the
                        # full hold. Exclude timeouts and failed transitions.
                        success = (not self.timeouts[index, environment]
                                   and observation[51] >= 1 - 1e-6
                                   and observation[self.door_index] >= self.door_threshold
                                   and observation[77] > self.grasp_threshold)
                        if success:
                            self.success_mask[episode, environment] = True
                            self.successful_episodes += 1
                        episode = []

    def add(self, obs, next_obs, action, reward, done, infos):
        slot = self.pos
        self.success_mask[slot, :] = False
        super().add(obs, next_obs, action, reward, done, infos)
        for environment, finished in enumerate(done):
            self._pending[environment].append(slot)
            if finished:
                info = infos[environment]
                if (info.get("is_success", False) and info.get("reset_origin", "home") == "home"
                        and not info.get("TimeLimit.truncated", False)):
                    self.success_mask[list(self._pending[environment]), environment] = True
                    self.successful_episodes += 1
                self._pending[environment].clear()

    def sample(self, batch_size, env=None):
        successes = np.flatnonzero(self.success_mask)
        preferred_count = int(batch_size * self.success_fraction) if len(successes) else 0
        if not preferred_count:
            return super().sample(batch_size, env=env)
        count = self.buffer_size if self.full else self.pos
        uniform = np.random.randint(count * self.n_envs, size=batch_size - preferred_count)
        preferred = np.random.choice(successes, size=preferred_count, replace=True)
        selected = np.concatenate((uniform, preferred))
        np.random.shuffle(selected)
        slots, environments = selected // self.n_envs, selected % self.n_envs
        data = (
            self._normalize_obs(self.observations[slots, environments], env),
            self.actions[slots, environments],
            self._normalize_obs(self.next_observations[slots, environments], env),
            (self.dones[slots, environments] * (1 - self.timeouts[slots, environments])).reshape(-1, 1),
            self._normalize_reward(self.rewards[slots, environments].reshape(-1, 1), env),
        )
        return ReplayBufferSamples(*tuple(map(self.to_torch, data)))

    def specification(self):
        return dict(kind="own-success episode sampling; unchanged SAC losses",
                    fraction=self.success_fraction, successful_episodes=self.successful_episodes,
                    retained_success_transitions=int(self.success_mask.sum()))
