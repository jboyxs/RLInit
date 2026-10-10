"""Behavioral checks for the fixed-base reaching MDP."""

import unittest

import mujoco
import numpy as np

from rlinit.algorithmhand.reach_env import ReachEnv


class ReachEnvTests(unittest.TestCase):
    def setUp(self):
        self.env = ReachEnv()
        self.addCleanup(self.env.close)

    def test_only_arm_moves_and_reset_is_reproducible(self):
        env = self.env
        self.assertEqual((env.model.nq, env.model.nv, env.model.nu), (7, 7, 7))
        first, info = env.reset(seed=42)
        obs, _ = env.reset(seed=42)
        np.testing.assert_array_equal(first, obs)
        self.assertFalse(info["is_success"])
        base_id = env.model.body("base_link").id
        base_position = env.data.xpos[base_id].copy()
        for _ in range(20):
            obs, reward, _, _, _ = env.step(np.ones(7))
            self.assertTrue(env.observation_space.contains(obs))
            self.assertTrue(np.isfinite(reward))
            self.assertTrue(np.all(env.data.ctrl >= env.lower))
            self.assertTrue(np.all(env.data.ctrl <= env.upper))
        np.testing.assert_array_equal(env.data.xpos[base_id], base_position)

    def test_success_reward_and_time_limit(self):
        env = self.env
        env.reset(seed=7)
        # Predict the next zero-action state so the target is exactly reached.
        clone = mujoco.MjData(env.model)
        mujoco.mj_copyData(clone, env.model, env.data)
        clone.ctrl[env.actuators] = clone.qpos[env.qadr]
        mujoco.mj_step(env.model, clone, nstep=env.frame_skip)
        mujoco.mj_forward(env.model, clone)
        env.goal = clone.site_xpos[env.ee_id].copy()
        _, reward, terminated, truncated, info = env.step(np.zeros(7))
        self.assertTrue(terminated)
        self.assertFalse(truncated)
        self.assertTrue(info["is_success"])
        self.assertAlmostEqual(reward, 5.0)
        env.reset(seed=7)
        env.goal = np.array([10.0, 10.0, 10.0])
        env.max_steps = 1
        _, reward, terminated, truncated, info = env.step(np.zeros(7))
        self.assertFalse(terminated)
        self.assertTrue(truncated)
        self.assertAlmostEqual(reward, -info["distance"])

    def test_reject_invalid_actions(self):
        self.env.reset(seed=0)
        for action in (np.zeros(6), np.full(7, np.nan)):
            with self.assertRaises(ValueError):
                self.env.step(action)


if __name__ == "__main__":
    unittest.main()
