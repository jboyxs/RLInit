"""Own-success sampling preserves vector/ring indexing and SAC targets."""

import pickle
import unittest
from types import SimpleNamespace
from unittest.mock import patch

from gymnasium import spaces
import numpy as np
import torch
from stable_baselines3.common.buffers import ReplayBuffer, NStepReplayBuffer

from rlinit.algorithmhand.door_sac_replay import OwnSuccessReplayBuffer, configure_multistep_replay


class SuccessReplayTests(unittest.TestCase):
    def test_multistep_online_replay_respects_timeouts_terminal_and_newest_boundary(self):
        base = self.make_buffer(ReplayBuffer)
        for step in range(4):
            self.add(base, step, successful=step == 2)
        original_timeouts = base.timeouts.copy()
        agent = SimpleNamespace(n_steps=3, gamma=0.9, replay_buffer=base,
                                replay_buffer_kwargs={}, replay_buffer_class=ReplayBuffer)
        configure_multistep_replay(agent)
        self.assertIsInstance(agent.replay_buffer, NStepReplayBuffer)
        self.assertIs(agent.replay_buffer.observations, base.observations)
        with patch("numpy.random.randint", return_value=np.array([0, 1, 0, 1, 0])):
            batch = agent.replay_buffer._get_samples(np.array([0, 0, 2, 2, 3]))
        np.testing.assert_allclose(batch.rewards.numpy().ravel(), [9, 27.91, 47, 21, 30], rtol=1e-6)
        np.testing.assert_allclose(batch.discounts.numpy().ravel(), [.81, .729, .81, .9, .9], rtol=1e-6)
        np.testing.assert_array_equal(batch.dones.numpy().ravel(), [0, 1, 0, 1, 0])
        np.testing.assert_array_equal(batch.next_observations.numpy()[:, 0], [1010, 1021, 1030, 1021, 1030])
        np.testing.assert_array_equal(agent.replay_buffer.timeouts, original_timeouts)
        restored = pickle.loads(pickle.dumps(agent.replay_buffer))
        self.assertEqual(restored.n_steps, 3)
        self.assertEqual(restored.gamma, 0.9)
        agent.n_steps = 1
        configure_multistep_replay(agent)
        self.assertIs(type(agent.replay_buffer), ReplayBuffer)
        self.assertEqual(agent.replay_buffer_kwargs, {})

    def test_multistep_full_ring_does_not_cross_newest_transition(self):
        base = self.make_buffer(ReplayBuffer, capacity=10)
        for step in range(7):
            self.add(base, step)
        agent = SimpleNamespace(n_steps=3, gamma=0.9, replay_buffer=base,
                                replay_buffer_kwargs={}, replay_buffer_class=ReplayBuffer)
        configure_multistep_replay(agent)
        with patch("numpy.random.randint", return_value=np.array([0, 1])):
            batch = agent.replay_buffer._get_samples(np.array([0, 1]))
        np.testing.assert_allclose(batch.rewards.numpy().ravel(), [104, 61])
        np.testing.assert_allclose(batch.discounts.numpy().ravel(), [.81, .9])
        np.testing.assert_array_equal(batch.next_observations.numpy()[:, 0], [1060, 1061])

    def make_buffer(self, cls=OwnSuccessReplayBuffer, capacity=24):
        return cls(capacity, spaces.Box(-np.inf, np.inf, (78,), dtype=np.float32),
                   spaces.Box(-1, 1, (8,), dtype=np.float32), n_envs=2, device="cpu")

    def add(self, buffer, step, *, successful=False):
        observation = np.zeros((2, 78), np.float32)
        observation[:, 0] = [10 * step, 10 * step + 1]
        next_observation = observation.copy()
        next_observation[:, 0] += 1000
        action = np.zeros((2, 8), np.float32)
        action[:, 0] = observation[:, 0] / 100
        done = np.array([step == 1, successful])
        infos = [{"is_success": False, "TimeLimit.truncated": step == 1},
                 {"is_success": successful, "reset_origin": "home"}]
        if step == 1:
            # A timeout looking geometrically successful is not completion.
            next_observation[0, [15, 51, 77]] = [.8, 1, 1]
        if successful:
            next_observation[1, [15, 51, 77]] = [.8, 1, 1]
        buffer.add(observation, next_observation, action, observation[:, 0], done, infos)

    def test_vector_success_samples_keep_matched_actions_rewards_and_next_states(self):
        buffer = self.make_buffer()
        for step in range(4):
            self.add(buffer, step, successful=step == 2)
        self.assertEqual(buffer.specification()["retained_success_transitions"], 3)
        np.testing.assert_array_equal(buffer.success_mask[:3, 1], True)
        self.assertFalse(buffer.success_mask[:, 0].any())
        buffer.success_fraction = .5
        np.random.seed(0)
        batch = buffer.sample(4000)
        marker = batch.observations[:, 0]
        fraction = torch.isin(marker, torch.tensor([1, 11, 21])).float().mean().item()
        self.assertGreater(fraction, .64)
        self.assertLess(fraction, .74)
        torch.testing.assert_close(batch.actions[:, 0] * 100, marker)
        torch.testing.assert_close(batch.rewards[:, 0], marker)
        torch.testing.assert_close(batch.next_observations[:, 0] - 1000, marker)
        # Timeout done is masked out in Bellman targets, as in SB3.
        self.assertTrue(torch.all(batch.dones[marker == 10] == 0))

    def test_existing_online_data_is_rebuilt_without_counting_timeouts(self):
        base = self.make_buffer(ReplayBuffer)
        for step in range(4):
            self.add(base, step, successful=step == 2)
        upgraded = OwnSuccessReplayBuffer.from_buffer(
            base, success_fraction=.25, door_index=15, door_threshold=.75, grasp_threshold=.025)
        self.assertEqual(upgraded.successful_episodes, 1)
        self.assertEqual(upgraded.success_mask.sum(), 3)
        np.testing.assert_array_equal(upgraded.observations, base.observations)
        np.testing.assert_array_equal(upgraded.rewards, base.rewards)
        restored = pickle.loads(pickle.dumps(upgraded))
        np.testing.assert_array_equal(restored.success_mask, upgraded.success_mask)
        self.assertEqual(restored.sample(16).actions.shape, (16, 8))

    def test_circular_overwrites_remove_stale_successes(self):
        buffer = self.make_buffer(capacity=8)
        for step in range(3):
            self.add(buffer, step, successful=step == 2)
        self.assertEqual(buffer.success_mask.sum(), 3)
        for step in range(3, 12):
            self.add(buffer, step)
        self.assertTrue(buffer.full)
        self.assertEqual(buffer.success_mask.sum(), 0)
        self.assertEqual(buffer.sample(32).observations.shape, (32, 78))

    def test_zero_fraction_has_identical_uniform_sampling(self):
        base = self.make_buffer(ReplayBuffer)
        for step in range(4):
            self.add(base, step, successful=step == 2)
        upgraded = OwnSuccessReplayBuffer.from_buffer(
            base, success_fraction=0, door_index=15, door_threshold=.75, grasp_threshold=.025)
        np.random.seed(10)
        expected = base.sample(50)
        np.random.seed(10)
        actual = upgraded.sample(50)
        for before, after in zip(expected, actual):
            if before is not None:
                torch.testing.assert_close(before, after)


if __name__ == "__main__":
    unittest.main()
