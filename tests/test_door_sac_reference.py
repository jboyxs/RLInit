"""Validate the supplied reward formula, physical success and SAC restoration."""

import ast
import csv
import json
from pathlib import Path
import tempfile
from types import SimpleNamespace
import unittest
from unittest.mock import patch

import numpy as np

from rlinit.algorithmhand.door_env import DoorConfig
from rlinit.algorithmhand.door_reference_env import TASK, ReferenceRewardConfig, ReferenceRewardDoorEnv
from trainp2_sac import evaluate, load_agent
from trainp2_sac_reference import train
from verify_door_opening import DoorTrial


REFERENCE = Path(__file__).resolve().parents[1] / "reference" / "door.py"


def reference_reward_function():
    """Extract only Door.reward, avoiding reference's optional robosuite imports."""
    source = ast.parse(REFERENCE.read_text(encoding="utf-8"))
    door = next(node for node in source.body if isinstance(node, ast.ClassDef) and node.name == "Door")
    reward = next(node for node in door.body if isinstance(node, ast.FunctionDef) and node.name == "reward")
    namespace = {"np": np}
    exec(compile(ast.Module(body=[reward], type_ignores=[]), str(REFERENCE), "exec"), namespace)
    return namespace["reward"]


class ReferenceRewardTests(unittest.TestCase):
    @unittest.skipUnless(REFERENCE.is_file(), "Optional reference/door.py was not provided")
    def test_formula_matches_supplied_source_including_success_and_sparse_mode(self):
        oracle = reference_reward_function()
        env = ReferenceRewardDoorEnv(config=DoorConfig(initial_noise=0))
        self.addCleanup(env.close)
        env.reset(seed=0)
        for shaping in (True, False):
            for scale in (1.0, 2.5):
                env.reward_config = ReferenceRewardConfig(reward_shaping=shaping, reward_scale=scale)
                for distance in (0.0, 0.1, 1.0):
                    for handle_angle in (-2.0, -0.8, 0.0, 0.8, 2.0):
                        for door_angle in (0.0, 0.3, 0.300001):
                            # Synthetic reward inputs only; no physics rollout uses these states.
                            env.data.site_xpos[env.ee] = env.data.site_xpos[env.handle_site] + [distance, 0, 0]
                            env.data.qpos[env.hinge] = door_angle
                            env.data.qpos[env.latch] = handle_angle
                            success = env._is_success(grasp=False, reason="")
                            reference = SimpleNamespace(
                                _check_success=lambda: door_angle > 0.3,
                                _gripper_to_handle=distance, reward_shaping=shaping,
                                use_latch=True, reward_scale=scale,
                                sim=SimpleNamespace(data=env.data), handle_qpos_addr=env.latch,
                            )
                            actual = sum(env._reward_components(success=success).values())
                            self.assertAlmostEqual(actual, oracle(reference), places=12)
                            self.assertEqual(success, door_angle > 0.3)

    def test_actual_steps_have_only_reference_rewards_and_keep_failure_checks(self):
        env = ReferenceRewardDoorEnv(config=DoorConfig(max_steps=4, initial_noise=0))
        self.addCleanup(env.close)
        env.reset(seed=0)
        action = np.r_[np.zeros(7), -1.0]
        for index in range(4):
            obs, reward, terminated, truncated, info = env.step(action)
            distance = np.linalg.norm(env.data.site_xpos[env.handle_site] - env.data.site_xpos[env.ee])
            expected = 0.25 * (1 - np.tanh(10 * distance))
            expected += np.clip(0.25 * abs(info["handle_angle"] / (0.5 * np.pi)), -0.25, 0.25)
            self.assertAlmostEqual(reward, expected)
            self.assertAlmostEqual(info["distance"], distance)
            self.assertEqual(set(info["reward_components"]), {"reaching", "rotating", "success"})
            self.assertTrue(env.observation_space.contains(obs))
            self.assertFalse(terminated)
            self.assertEqual(truncated, index == 3)
        with self.assertRaises(RuntimeError):
            env.step(action)
        env.reset(seed=0)
        with patch.object(env, "_contacts", return_value=({"left": 0.0, "right": 0.0}, 301.0, 0.0)):
            _, reward, terminated, truncated, info = env.step(action)
        self.assertTrue(terminated)
        self.assertFalse(truncated)
        self.assertEqual(info["failure_reason"], "unsafe_collision")
        self.assertFalse(info["is_success"])
        self.assertGreaterEqual(reward, 0.0)  # Reference does not add a failure penalty.

    def test_real_robot_actions_open_the_passive_door_and_receive_completion_reward(self):
        env = ReferenceRewardDoorEnv(config=DoorConfig(initial_noise=0))
        self.addCleanup(env.close)
        env.reset(seed=0)
        reference = DoorTrial("normal")
        result = {}

        def apply_reference(trial, _):
            if result:
                return
            action = np.r_[(trial.last_target - env.arm_target) / env.config.action_scale,
                           2 * trial.data.ctrl[trial.gripper] / env.config.grip_max - 1]
            obs, reward, terminated, truncated, info = env.step(action)
            self.assertTrue(env.observation_space.contains(obs))
            self.assertFalse(truncated)
            self.assertEqual(info["failure_reason"], "")
            if terminated:
                self.assertTrue(info["is_success"])
                self.assertGreater(info["door_angle"], 0.3)
                self.assertEqual(reward, 1.0)
                self.assertEqual(info["reward_components"], {"reaching": 0.0, "rotating": 0.0, "success": 1.0})
                self.assertEqual(info["success_hold_seconds"], 0.0)
                result.update(info)

        reference.run(apply_reference)
        self.assertTrue(result)
        np.testing.assert_array_equal(env.data.qfrc_applied, 0)
        np.testing.assert_array_equal(env.data.xfrc_applied, 0)

    def check_training(self, n_envs, reward_config):
        with tempfile.TemporaryDirectory() as directory:
            output = Path(directory)
            checkpoint = train(
                total_timesteps=24, n_envs=n_envs, output_dir=output, device="cpu",
                env_config=DoorConfig(max_steps=4), reward_config=reward_config,
                learning_starts=4, buffer_size=64, batch_size=8, gamma=0.995,
                eval_freq=12, eval_episodes=1,
            )
            config = json.loads((output / "config.json").read_text())
            self.assertEqual(config["environment"]["task"], TASK)
            self.assertEqual(config["environment"]["reward_config"]["reward_scale"], reward_config.reward_scale)
            self.assertEqual(config["gradient_updates"], 20)
            with (output / "logs" / "progress.csv").open(newline="") as stream:
                rows = list(csv.DictReader(stream))
            for key in ("train/actor_loss", "train/critic_loss"):
                values = [float(row[key]) for row in rows if row.get(key)]
                self.assertTrue(values)
                self.assertTrue(np.isfinite(values).all())
            (output / "config.json").unlink()
            for path in (checkpoint, output / "best" / "best_model.zip"):
                agent, env = load_agent(path, device="cpu")
                try:
                    self.assertIsInstance(env, ReferenceRewardDoorEnv)
                    self.assertEqual(env.reward_config, reward_config)
                    self.assertEqual(env.config.max_steps, 4)
                    self.assertEqual(env.config.gamma, agent.gamma)
                finally:
                    env.close()
                metrics = evaluate(path, episodes=1, seed=123, device="cpu")
                self.assertGreaterEqual(metrics["mean_return"], 0.0)
                if not reward_config.reward_shaping:
                    self.assertEqual(metrics["mean_return"], 0.0)

    def test_single_environment_checkpoint_restores_dense_reward(self):
        self.check_training(1, ReferenceRewardConfig(reward_scale=2.5))

    def test_parallel_checkpoint_restores_sparse_reward(self):
        self.check_training(2, ReferenceRewardConfig(reward_shaping=False, reward_scale=1.2))


if __name__ == "__main__":
    unittest.main()
