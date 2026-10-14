"""Behavioral checks for the door MDP and its original hand-written PPO."""

import tempfile
import unittest
from pathlib import Path

import mujoco
import numpy as np
import torch

from rlinit.algorithmhand.door_env import DoorConfig, DoorEnv
from rlinit.algorithmhand.PPOhand import PPO
from trainp2_hand import load_agent, save_agent
from verify_door_opening import DoorTrial


class DoorEnvTests(unittest.TestCase):
    def test_reset_fixed_base_timeout_and_control_limits(self):
        env = DoorEnv(config=DoorConfig(max_steps=8))
        self.addCleanup(env.close)
        self.assertEqual((env.model.nq, env.model.nv, env.model.nu), (17, 17, 8))
        first, _ = env.reset(seed=42)
        second, _ = env.reset(seed=42)
        np.testing.assert_array_equal(first, second)
        self.assertEqual(first.shape, (64,))
        base = env.data.body("base_link").xpos.copy()
        for index in range(8):
            obs, reward, terminated, truncated, info = env.step(np.r_[np.zeros(7), -1.0])
            self.assertTrue(env.observation_space.contains(obs))
            self.assertTrue(np.isfinite(reward))
            self.assertFalse(terminated)
            self.assertEqual(truncated, index == 7)
            self.assertFalse(info["is_success"])
            np.testing.assert_array_equal(env.data.body("base_link").xpos, base)
            self.assertTrue(np.all(env.arm_target >= env.lower))
            self.assertTrue(np.all(env.arm_target <= env.upper))
        with self.assertRaises(RuntimeError):
            env.step(np.zeros(8))
        env.reset()
        for action in (np.zeros(7), np.full(8, np.nan)):
            with self.assertRaises(ValueError):
                env.step(action)

    def test_reference_robot_actions_reach_success_in_the_actual_mdp(self):
        env = DoorEnv(config=DoorConfig(initial_noise=0))
        self.addCleanup(env.close)
        env.reset(seed=0)
        reference = DoorTrial("normal")
        result = {}
        milestone_counts = {"grasp_milestone": 0, "press_milestone": 0}
        peak_depth = 0.0

        def apply_reference(trial, _):
            nonlocal peak_depth
            if result.get("is_success"):
                return
            action = np.r_[(trial.last_target - env.arm_target) / env.config.action_scale,
                           2 * trial.data.ctrl[trial.gripper] / env.config.grip_max - 1]
            self.assertLessEqual(np.max(np.abs(action)), 1.00001)
            obs, reward, terminated, truncated, info = env.step(action)
            self.assertTrue(env.observation_space.contains(obs))
            self.assertTrue(np.isfinite(reward))
            peak_depth = max(peak_depth, info["penetration_m"])
            for key in milestone_counts:
                milestone_counts[key] += info["reward_components"][key] > 0
            self.assertFalse(truncated)
            self.assertEqual(info["failure_reason"], "")
            if terminated:
                self.assertTrue(info["is_success"])
                self.assertEqual(info["reward_components"]["success"], 50)
                result.update(info)

        reference.run(apply_reference)
        self.assertTrue(result.get("is_success"), "The verified contact trajectory must succeed through env.step")
        self.assertGreaterEqual(result["door_angle"], env.config.success_angle)
        self.assertGreaterEqual(result["success_hold_seconds"], 0.30)
        self.assertEqual(milestone_counts, {"grasp_milestone": 1, "press_milestone": 1})
        self.assertLess(peak_depth, 0.0015)
        # This test commands only the robot, including through DoorEnv's action
        # limits. Neither door coordinates nor door forces are injected.
        np.testing.assert_array_equal(env.data.qfrc_applied, 0)
        np.testing.assert_array_equal(env.data.xfrc_applied, 0)

    def test_rollout_updates_ppo_and_checkpoint_preserves_task(self):
        torch.manual_seed(0)
        env = DoorEnv(config=DoorConfig(max_steps=8, gamma=0.995))
        self.addCleanup(env.close)
        agent = PPO.from_env(env, hidden_dim=32, epochs=2, gamma=env.config.gamma, device="cpu")
        self.assertEqual((agent.state_dim, agent.action_dim), (64, 8))
        keys = ("states", "actions", "rewards", "next_states", "terminated", "truncated", "log_probs")
        rollout = {key: [] for key in keys}
        state, _ = env.reset(seed=0)
        before = [p.detach().clone() for p in agent.actor.parameters()]
        for _ in range(32):
            action, log_prob = agent.take_action(state, return_log_prob=True)
            next_state, reward, terminated, truncated, _ = env.step(action)
            for key, value in zip(keys, (state, action, reward, next_state, terminated, truncated, log_prob)):
                rollout[key].append(value)
            state = next_state
            if terminated or truncated:
                state, _ = env.reset()
        losses = agent.update(rollout)
        self.assertTrue(all(np.isfinite(v) for v in losses.values()))
        self.assertTrue(any(not torch.equal(a, b) for a, b in zip(before, agent.actor.parameters())))
        config = dict(state_dim=64, action_dim=8, hidden_dim=32, epochs=2, gamma=0.995, device="cpu")
        with tempfile.TemporaryDirectory() as directory:
            path = Path(directory) / "door.pt"
            save_agent(agent, env, path, config, 32)
            loaded, restored_env = load_agent(path, device="cpu")
            self.addCleanup(restored_env.close)
            self.assertEqual(restored_env.config, env.config)
            self.assertEqual(loaded.hidden_depth, 2)
            np.testing.assert_array_equal(agent.take_action(state, deterministic=True),
                                          loaded.take_action(state, deterministic=True))

    def test_old_single_layer_door_checkpoint_remains_evaluable(self):
        env = DoorEnv()
        self.addCleanup(env.close)
        state, _ = env.reset(seed=0)
        config = dict(state_dim=64, action_dim=8, hidden_dim=16, hidden_depth=1, device="cpu")
        original = PPO(**config)
        with tempfile.TemporaryDirectory() as directory:
            path = Path(directory) / "old_door.pt"
            save_agent(original, env, path, config, 123)
            checkpoint = torch.load(path, weights_only=True)
            del checkpoint["agent_config"]["hidden_depth"]
            torch.save(checkpoint, path)
            loaded, restored_env = load_agent(path, device="cpu")
            self.addCleanup(restored_env.close)
            self.assertEqual(loaded.hidden_depth, 1)
            np.testing.assert_array_equal(original.take_action(state, deterministic=True),
                                          loaded.take_action(state, deterministic=True))


if __name__ == "__main__":
    unittest.main()
