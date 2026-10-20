"""Check press-only relabelling against the environment and resume boundaries."""

from dataclasses import replace
import unittest

from gymnasium import spaces
import numpy as np
from stable_baselines3.common.buffers import NStepReplayBuffer, ReplayBuffer

from rlinit.algorithmhand.door_sac_learn_env import LearnRewardConfig, SACLearnDoorEnv
from rlinit.algorithmhand.door_sac_learn_replay import (
    close_resume_segments, press_progress_delta, relabel_press_progress,
)


class LearnReplayTests(unittest.TestCase):
    def test_delta_matches_environment_for_press_hold_release_and_termination(self):
        cfg = LearnRewardConfig.guided()
        old, new = SACLearnDoorEnv(reward_config=cfg), SACLearnDoorEnv(
            reward_config=replace(cfg, press_progress_only=True))
        self.addCleanup(old.close)
        self.addCleanup(new.close)
        old.reset(seed=67)
        new.reset(seed=67)
        cases = [((.08, .7, False), (.1, .8, True), False),
                 ((.1, .8, True), (.1, .9, True), False),
                 ((.1, .8, True), (0., .2, True), False),
                 ((.08, .7, False), (0., .2, False), True),
                 ((.1, .8, True), (0., .2, True), True)]
        for before, after, terminal in cases:
            observations = []
            for state in (before, after):
                hold, angle, latched = state
                obs = np.zeros(80, np.float32)
                obs[new.latch], obs[-2], obs[-1] = angle / 1.3, hold / .1, latched
                observations.append(obs)
            new.grasp_hold, new.data.qpos[new.latch], new.pressed_with_stable_grasp = before
            potential_before = new._potential(new._features())
            for env in (old, new):
                env.grasp_hold, env.data.qpos[env.latch], env.pressed_with_stable_grasp = after
            f = old._features()
            grasp = after[0] > 0
            features = (*f[:3], {"left": 20., "right": 20.} if grasp else {"left": 0., "right": 0.},
                        grasp, 0., 0.)
            args = dict(features=features, terminated=terminal, newly_grasped=False,
                        newly_pressed=False, success=False, reason="unsafe_collision" if terminal else "",
                        peak_bad_force=0., action=np.zeros(8))
            old_reward = sum(old._reward_components(potential_before=0., **args).values())
            new_reward = sum(new._reward_components(potential_before=potential_before, **args).values())
            delta = press_progress_delta(*observations, terminal, reward_config=cfg,
                                        gamma=new.config.gamma, latch_index=new.latch)
            self.assertAlmostEqual(new_reward - old_reward, float(delta), places=5)

    @staticmethod
    def buffer():
        return ReplayBuffer(4, spaces.Box(-np.inf, np.inf, (80,), np.float32),
                            spaces.Box(-1, 1, (8,), np.float32), n_envs=2, device="cpu")

    def test_resume_closes_only_unfinished_segments_with_bootstrap_preserved(self):
        buffer = self.buffer()
        obs = np.zeros((2, 80), np.float32)
        buffer.add(obs, obs + .1, np.zeros((2, 8)), np.zeros(2), np.array([False, True]), [{}, {}])
        following = buffer.next_observations.copy()
        self.assertEqual(close_resume_segments(buffer), 1)
        np.testing.assert_array_equal(buffer.dones[0], [1, 1])
        np.testing.assert_array_equal(buffer.timeouts[0], [1, 0])
        np.testing.assert_array_equal(buffer.next_observations, following)

    def test_conversion_handles_ring_buffer_and_timeouts_without_changing_transitions(self):
        buffer = self.buffer()
        for i in range(3):
            obs = np.zeros((2, 80), np.float32)
            obs[:, 16] = .8 / 1.3
            obs[:, -2:] = 1.
            buffer.add(obs, obs, np.ones((2, 8)) * .2, np.ones(2) * 3.,
                       np.array([True, True]), [{"TimeLimit.truncated": True}, {}])
        self.assertTrue(buffer.full)
        original_obs, original_actions = buffer.observations.copy(), buffer.actions.copy()
        cfg = LearnRewardConfig.guided()
        count = relabel_press_progress(buffer, reward_config=cfg, gamma=.995, latch_index=16)
        self.assertEqual(count, 4)
        # Timeout bootstraps the latched potential; a real terminal zeros it.
        np.testing.assert_allclose(buffer.rewards[:, 0], -.06, atol=1e-5)
        np.testing.assert_allclose(buffer.rewards[:, 1], -12., atol=1e-5)
        np.testing.assert_array_equal(buffer.observations, original_obs)
        np.testing.assert_array_equal(buffer.actions, original_actions)

    def test_three_step_sampling_does_not_cross_a_resumed_home_reset(self):
        buffer = NStepReplayBuffer(16, spaces.Box(-np.inf, np.inf, (80,), np.float32),
            spaces.Box(-1, 1, (8,), np.float32), n_envs=1, device="cpu", n_steps=3, gamma=.9)
        for i in range(3):
            buffer.add(np.full((1, 80), i), np.full((1, 80), i + 1), np.zeros((1, 8)),
                       np.array([1.]), np.array([False]), [{}])
        close_resume_segments(buffer)
        for i in range(2):
            buffer.add(np.full((1, 80), 100 + i), np.full((1, 80), 101 + i), np.zeros((1, 8)),
                       np.array([100.]), np.array([False]), [{}])
        sample = buffer._get_samples(np.array([1]))
        self.assertAlmostEqual(float(sample.rewards.item()), 1.9, places=5)
        self.assertAlmostEqual(float(sample.discounts.item()), .81, places=5)
        self.assertEqual(float(sample.dones.item()), 0.)
        np.testing.assert_array_equal(sample.next_observations.numpy(), np.full((1, 80), 3))


if __name__ == "__main__":
    unittest.main()
