"""Check direct state rewards and SAC checkpoint restoration in both backends."""

from dataclasses import replace
import json
from pathlib import Path
import tempfile
import unittest
from unittest.mock import patch

import numpy as np
import mujoco
import torch

from rlinit.algorithmhand.door_direct_env import TASK, DirectRewardDoorEnv
from rlinit.algorithmhand.door_env import DEFAULT_MODEL, DoorConfig
from rlinit.algorithmhand.door_reference_env import ReferenceRewardConfig
from trainp2_sac import evaluate, load_agent, make_environment
from trainp2_sac_direct import train


class DirectRewardTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.previous_threads = torch.get_num_threads()
        torch.set_num_threads(1)

    @classmethod
    def tearDownClass(cls):
        torch.set_num_threads(cls.previous_threads)

    def test_state_reward_ignores_potential_discount_and_terminal_subtraction(self):
        env = DirectRewardDoorEnv(config=DoorConfig(initial_noise=0))
        self.addCleanup(env.close)
        env.reset(seed=0)
        # Halfway rotation and opening, with 60% alignment and 5 cm total error.
        features = (np.array([0.08 * np.log(2), 0, 0]), None, 0.6, None, True, 0, 0)
        distances = patch.object(env, "_finger_surface_distances", return_value=np.array([0.02, 0.03]))
        distances.start()
        self.addCleanup(distances.stop)
        env.data.qpos[env.latch] = 0.65
        env.data.qpos[env.hinge] = 0.15
        action = env.previous_action.copy()
        for gamma in (0.5, 0.999):
            env.config = replace(env.config, gamma=gamma)
            for terminated in (False, True):
                for potential_before in (-100.0, 100.0):
                    parts = env._reward_components(
                        features=features, terminated=terminated, potential_before=potential_before,
                        newly_grasped=False, newly_pressed=False, success=False, reason="",
                        peak_bad_force=0, action=action,
                    )
                    self.assertNotIn("progress", parts)
                    expected = 5.99 - env.distance_weight * 0.05
                    expected += env.orientation_weight * 0.6 * np.exp(-0.05 / 0.08)
                    self.assertAlmostEqual(sum(parts.values()), expected)
                    self.assertEqual(env._potential(features), 0.0)
        ungrasped = (*features[:4], False, *features[5:])
        parts = env._reward_components(
            features=ungrasped, newly_grasped=True, newly_pressed=True,
            success=True, reason="", peak_bad_force=0, action=action,
        )
        self.assertEqual(parts["pressing"], 0.0)
        self.assertEqual(parts["grasp_milestone"], 2.0)
        self.assertEqual(parts["press_milestone"], 3.0)
        self.assertEqual(parts["success"], 50.0)
        self.assertAlmostEqual(parts["opening"], 5.0)

    def test_position_cost_uses_sum_not_mean_and_does_not_saturate(self):
        env = DirectRewardDoorEnv(config=DoorConfig(initial_noise=0), distance_weight=2.5)
        self.addCleanup(env.close)
        env.reset(seed=0)
        for distance, expected in ((0.0, 0.0), (0.02, -0.05), (0.05, -0.125),
                                   (0.1, -0.25), (0.3, -0.75)):
            features = (np.array([distance, 0, 0]), None, 0.6, None, False, 0, 0)
            with patch.object(env, "_finger_surface_distances", return_value=np.array([0.4, 0.6]) * distance):
                parts = env._reward_components(
                    features=features, newly_grasped=False, newly_pressed=False,
                    success=False, reason="", peak_bad_force=0, action=env.previous_action,
                )
            self.assertAlmostEqual(parts["distance"], expected)
            self.assertNotIn("reaching", parts)
            self.assertNotIn("alignment", parts)
            self.assertAlmostEqual(parts["orientation"], env.orientation_weight * 0.6 * np.exp(-distance / 0.08))

    def test_surface_points_use_inner_faces_and_follow_body_rotations(self):
        env = DirectRewardDoorEnv(config=DoorConfig(initial_noise=0))
        self.addCleanup(env.close)
        env.reset(seed=0)
        for door_angle, latch_angle in ((0.0, 0.0), (0.2, 0.9)):
            # Kinematics-only checks; no rollout uses these synthetic poses.
            env.data.qpos[env.hinge] = door_angle
            env.data.qpos[env.latch] = latch_angle
            env.data.qpos[env.qadr[-1]] += 0.3
            mujoco.mj_forward(env.model, env.data)
            fingers, faces = env._surface_distance_points()
            for index, side in enumerate(("left", "right")):
                body = env.model.body(f"{side}_pad").id
                rotation = env.data.xmat[body].reshape(3, 3)
                # From XML: two adjoining 18.75 mm pad boxes, inward Y=-6.6 mm.
                expected = env.data.xpos[body] + rotation @ np.array([0.0, -0.0066, 0.01875])
                np.testing.assert_allclose(fingers[index], expected, atol=1e-12)
            body = env.model.body("door_handle_body").id
            rotation = env.data.xmat[body].reshape(3, 3)
            for index, height in enumerate((0.02, -0.02)):
                expected = env.data.xpos[body] + rotation @ np.array([0.075, -0.10, height])
                np.testing.assert_allclose(faces[index], expected, atol=1e-12)
            np.testing.assert_allclose(env._finger_surface_distances(),
                                       np.linalg.norm(fingers - faces, axis=1), atol=1e-12)
        # At the desired grasp rotation left is upper, right is lower.
        rotation = env.data.site_xmat[env.handle_site].reshape(3, 3)
        ideal = rotation @ env.grasp_relative_rotation
        local_gripper_x = ideal[:, 0]
        self.assertGreater(np.dot(local_gripper_x, rotation[:, 2]), 0.99)

    def test_only_position_and_orientation_depend_on_distance_metric(self):
        config = DoorConfig(max_steps=3, initial_noise=0)
        env = DirectRewardDoorEnv(config=config)
        previous = DirectRewardDoorEnv(config=config, distance_metric="grasp_target")
        self.addCleanup(env.close)
        self.addCleanup(previous.close)
        obs, _ = env.reset(seed=0)
        previous_obs, _ = previous.reset(seed=0)
        np.testing.assert_array_equal(obs, previous_obs)
        for _ in range(3):
            action = np.r_[np.full(7, 0.01), -0.5]
            obs, reward, terminated, truncated, info = env.step(action)
            old_obs, old_reward, old_terminated, old_truncated, old_info = previous.step(action)
            np.testing.assert_array_equal(obs, old_obs)
            self.assertEqual((terminated, truncated), (old_terminated, old_truncated))
            self.assertAlmostEqual(info["distance"],
                                   info["upper_finger_distance_m"] + info["lower_finger_distance_m"])
            self.assertAlmostEqual(info["grasp_target_distance_m"], old_info["distance"])
            new_parts, old_parts = info["reward_components"], old_info["reward_components"]
            for key in new_parts.keys() - {"distance", "orientation"}:
                self.assertEqual(new_parts[key], old_parts[key], key)
            delta = sum(new_parts[key] - old_parts[key] for key in ("distance", "orientation"))
            self.assertAlmostEqual(reward - old_reward, delta)

    def test_orientation_cost_has_matching_weight_at_any_distance(self):
        env = DirectRewardDoorEnv(config=DoorConfig(initial_noise=0), orientation_reward="orientation_cost",
                                  distance_weight=2.5, orientation_weight=2.5)
        self.addCleanup(env.close)
        env.reset(seed=0)
        self.assertEqual(env.orientation_weight, env.distance_weight)
        for distance in (0.0, 0.1, 0.3):
            for alignment, expected in ((0.0, -2.5), (0.5, -1.25), (0.98, -0.05), (1.0, 0.0)):
                features = (np.array([distance, 0, 0]), None, alignment, None, False, 0, 0)
                parts = env._reward_components(
                    features=features, newly_grasped=False, newly_pressed=False,
                    success=False, reason="", peak_bad_force=0, action=env.previous_action,
                )
                self.assertAlmostEqual(parts["orientation"], expected)

    def test_proximity_alignment_uses_surface_distance_and_zero_alignment_gives_zero(self):
        env = DirectRewardDoorEnv(config=DoorConfig(initial_noise=0), orientation_weight=1.0)
        self.addCleanup(env.close)
        env.reset(seed=0)
        self.assertEqual(env.orientation_reward, "proximity_alignment")
        for distance, decay in ((0.0, 1.0), (0.08, np.exp(-1.0)), (0.16, np.exp(-2.0))):
            for alignment in (0.0, 0.5, 1.0):
                # Deliberately unrelated legacy delta: d must come from both fingers.
                features = (np.array([10.0, 0, 0]), None, alignment, None, False, 0, 0)
                with patch.object(env, "_finger_surface_distances", return_value=np.array([0.25, 0.75]) * distance):
                    parts = env._reward_components(
                        features=features, newly_grasped=False, newly_pressed=False,
                        success=False, reason="", peak_bad_force=0, action=env.previous_action,
                    )
                self.assertAlmostEqual(parts["orientation"], alignment * decay)
                self.assertGreaterEqual(parts["orientation"], 0.0)

    def test_steps_keep_dense_rewards_on_timeout_and_physical_failure(self):
        env = DirectRewardDoorEnv(config=DoorConfig(max_steps=2, initial_noise=0))
        self.addCleanup(env.close)
        action = np.r_[np.zeros(7), -1.0]
        rewards = []
        for gamma in (0.5, 0.999):
            env.config = replace(env.config, gamma=gamma)
            env.reset(seed=0)
            episode_rewards = []
            for index in range(2):
                obs, reward, terminated, truncated, info = env.step(action)
                self.assertTrue(env.observation_space.contains(obs))
                self.assertFalse(terminated)
                self.assertEqual(truncated, index == 1)
                self.assertAlmostEqual(info["reward_components"]["distance"], -env.distance_weight * info["distance"])
                self.assertAlmostEqual(reward, sum(info["reward_components"].values()))
                episode_rewards.append(reward)
            rewards.append(episode_rewards)
        np.testing.assert_array_equal(*rewards)
        env.reset(seed=0)
        with patch.object(env, "_contacts", return_value=({"left": 0.0, "right": 0.0}, 301.0, 0.0)):
            _, reward, terminated, truncated, info = env.step(action)
        self.assertTrue(terminated)
        self.assertFalse(truncated)
        self.assertEqual(info["failure_reason"], "unsafe_collision")
        self.assertEqual(info["reward_components"]["failure"], -20.0)
        self.assertEqual(info["reward_components"]["collision"], -0.2)
        self.assertAlmostEqual(info["reward_components"]["distance"], -env.distance_weight * info["distance"])
        self.assertTrue(np.isfinite(reward))

    def test_success_still_requires_grasp_and_hold(self):
        env = DirectRewardDoorEnv(config=DoorConfig(initial_noise=0))
        self.addCleanup(env.close)
        env.reset(seed=0)
        env.data.qpos[env.hinge] = env.config.success_angle
        self.assertFalse(env._is_success(grasp=False, reason=""))
        self.assertFalse(env._is_success(grasp=True, reason=""))
        self.assertFalse(env._is_success(grasp=True, reason="unsafe_collision"))
        self.assertEqual(env.success_time, 0.0)
        for _ in range(100):
            if env._is_success(grasp=True, reason=""):
                break
        self.assertEqual(env.success_time, env.config.success_hold_seconds)

    def test_reward_modes_cannot_be_combined(self):
        with self.assertRaisesRegex(ValueError, "cannot be combined"):
            make_environment(DEFAULT_MODEL, DoorConfig(), direct_reward=True,
                             reference_reward=ReferenceRewardConfig())

    def test_training_factory_follows_environment_weight_defaults(self):
        # Simulate editing only the constructor defaults in door_direct_env.py.
        defaults = dict(DirectRewardDoorEnv.__init__.__kwdefaults__,
                        distance_weight=3.75, orientation_weight=1.25)
        with patch.object(DirectRewardDoorEnv.__init__, "__kwdefaults__", defaults):
            env = make_environment(DEFAULT_MODEL, DoorConfig(initial_noise=0), direct_reward=True)
        self.addCleanup(env.close)
        self.assertEqual(env.distance_weight, 3.75)
        self.assertEqual(env.orientation_weight, 1.25)
        self.assertEqual(env.specification()["distance_weight"], 3.75)
        self.assertEqual(env.specification()["orientation_weight"], 1.25)
        env.reset(seed=0)
        _, _, _, _, info = env.step(np.r_[np.zeros(7), -1.0])
        self.assertAlmostEqual(info["reward_components"]["distance"], -3.75 * info["distance"])
        self.assertAlmostEqual(info["reward_components"]["orientation"],
                               1.25 * info["alignment"] * np.exp(-info["distance"] / 0.08))

    def check_training(self, n_envs):
        defaults = DirectRewardDoorEnv.__init__.__kwdefaults__
        distance_weight = defaults["distance_weight"]
        orientation_weight = defaults["orientation_weight"]
        with tempfile.TemporaryDirectory() as directory:
            output = Path(directory)
            checkpoint = train(
                total_timesteps=16, n_envs=n_envs, output_dir=output, device="cpu",
                env_config=DoorConfig(max_steps=4), learning_starts=4,
                buffer_size=64, batch_size=8, gamma=0.995,
                eval_freq=8, eval_episodes=1,
            )
            config = json.loads((output / "config.json").read_text())
            self.assertEqual(config["environment"]["task"], TASK)
            self.assertEqual(config["environment"]["position_reward"], "linear_distance")
            self.assertEqual(config["environment"]["orientation_reward"], "proximity_alignment")
            self.assertEqual(config["environment"]["distance_metric"], "finger_surface_sum")
            self.assertEqual(config["environment"]["distance_weight"], distance_weight)
            self.assertEqual(config["environment"]["orientation_weight"], orientation_weight)
            self.assertEqual(config["actual_timesteps"], 16)
            self.assertGreater(config["gradient_updates"], 0)
            self.assertEqual(config["agent"]["net_arch"], [256, 256])
            self.assertEqual(config["agent"]["gradient_steps"], 1)
            (output / "config.json").unlink()
            for path in (checkpoint, output / "best" / "best_model.zip"):
                agent, env = load_agent(path, device="cpu")
                try:
                    self.assertIsInstance(env, DirectRewardDoorEnv)
                    self.assertEqual(env.position_reward, "linear_distance")
                    self.assertEqual(env.orientation_reward, "proximity_alignment")
                    self.assertEqual(env.distance_metric, "finger_surface_sum")
                    self.assertEqual(env.distance_weight, distance_weight)
                    self.assertEqual(env.orientation_weight, orientation_weight)
                    self.assertEqual(env.config.gamma, agent.gamma)
                    obs, _ = env.reset(seed=123)
                    action, _ = agent.predict(obs, deterministic=True)
                    _, reward, _, _, info = env.step(action)
                    self.assertNotIn("progress", info["reward_components"])
                    self.assertNotIn("reaching", info["reward_components"])
                    self.assertAlmostEqual(info["reward_components"]["distance"], -distance_weight * info["distance"])
                    self.assertNotIn("alignment", info["reward_components"])
                    self.assertAlmostEqual(info["reward_components"]["orientation"],
                                           orientation_weight * info["alignment"] * np.exp(-info["distance"] / 0.08))
                    self.assertTrue(np.isfinite(reward))
                finally:
                    env.close()
            metrics = evaluate(checkpoint, episodes=1, seed=123, device="cpu")
            self.assertEqual(metrics, evaluate(checkpoint, episodes=1, seed=123, device="cpu"))
            # Previous checkpoints used a negative orientation error, with
            # their saved weight, even though new training uses a positive term.
            agent.door_environment["orientation_reward"] = "orientation_cost"
            legacy_cost_path = output / "legacy_cost.zip"
            agent.save(legacy_cost_path)
            _, legacy_cost_env = load_agent(legacy_cost_path, device="cpu")
            try:
                self.assertEqual(legacy_cost_env.orientation_reward, "orientation_cost")
                self.assertEqual(legacy_cost_env.orientation_weight, orientation_weight)
                legacy_cost_env.reset(seed=123)
                _, _, _, _, info = legacy_cost_env.step(action)
                self.assertAlmostEqual(info["reward_components"]["orientation"],
                                       -orientation_weight * (1.0 - info["alignment"]))
            finally:
                legacy_cost_env.close()
            # Before the orientation weight field, its coefficient was 0.25.
            agent.door_environment.pop("orientation_weight")
            legacy_orientation_path = output / "legacy_orientation.zip"
            agent.save(legacy_orientation_path)
            _, legacy_orientation_env = load_agent(legacy_orientation_path, device="cpu")
            try:
                self.assertEqual(legacy_orientation_env.distance_weight, distance_weight)
                self.assertEqual(legacy_orientation_env.orientation_weight, 0.25)
                legacy_orientation_env.reset(seed=123)
                _, _, _, _, info = legacy_orientation_env.step(action)
                self.assertAlmostEqual(info["reward_components"]["orientation"],
                                       -0.25 * (1.0 - info["alignment"]))
            finally:
                legacy_orientation_env.close()
            # Previous surface-distance checkpoints used weight 5, without a field.
            agent.door_environment.pop("distance_weight")
            legacy_surface_path = output / "legacy_surface.zip"
            agent.save(legacy_surface_path)
            _, legacy_surface_env = load_agent(legacy_surface_path, device="cpu")
            try:
                self.assertEqual(legacy_surface_env.distance_metric, "finger_surface_sum")
                self.assertEqual(legacy_surface_env.distance_weight, 5.0)
                legacy_surface_env.reset(seed=123)
                _, _, _, _, info = legacy_surface_env.step(action)
                self.assertAlmostEqual(info["reward_components"]["distance"], -5.0 * info["distance"])
            finally:
                legacy_surface_env.close()
            # Earlier checkpoints all used the pinch-to-target distance. Older
            # ones additionally used distance-weighted alignment / reaching.
            agent.door_environment.pop("distance_metric")
            for position_reward, orientation_reward in (
                    ("linear_distance", "orientation_cost"),
                    ("linear_distance", "distance_weighted_alignment"),
                    ("exponential_reach", "distance_weighted_alignment")):
                if orientation_reward == "distance_weighted_alignment":
                    agent.door_environment.pop("orientation_reward", None)
                if position_reward == "exponential_reach":
                    agent.door_environment.pop("position_reward")
                legacy_path = output / f"legacy_{position_reward}_{orientation_reward}.zip"
                agent.save(legacy_path)
                _, legacy_env = load_agent(legacy_path, device="cpu")
                try:
                    self.assertEqual(legacy_env.position_reward, position_reward)
                    self.assertEqual(legacy_env.orientation_reward, orientation_reward)
                    self.assertEqual(legacy_env.distance_metric, "grasp_target")
                    self.assertEqual(legacy_env.distance_weight, 5.0)
                    legacy_env.reset(seed=123)
                    _, reward, _, _, info = legacy_env.step(action)
                    self.assertAlmostEqual(info["distance"], np.linalg.norm(legacy_env._features()[0]))
                    if orientation_reward == "distance_weighted_alignment":
                        self.assertNotIn("orientation", info["reward_components"])
                        self.assertAlmostEqual(info["reward_components"]["alignment"],
                                               info["alignment"] * np.exp(-info["distance"] / 0.08))
                    else:
                        self.assertAlmostEqual(info["reward_components"]["orientation"],
                                               -0.25 * (1.0 - info["alignment"]))
                    if position_reward == "exponential_reach":
                        self.assertNotIn("distance", info["reward_components"])
                        self.assertAlmostEqual(info["reward_components"]["reaching"],
                                               2.0 * np.exp(-info["distance"] / 0.08))
                    else:
                        self.assertAlmostEqual(info["reward_components"]["distance"], -5.0 * info["distance"])
                    self.assertTrue(np.isfinite(reward))
                finally:
                    legacy_env.close()

    def test_single_environment_training_and_checkpoint(self):
        self.check_training(1)

    def test_parallel_training_and_checkpoint(self):
        self.check_training(2)


if __name__ == "__main__":
    unittest.main()
