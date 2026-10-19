"""Validate the online-only precision SAC task and resumable checkpoints."""

import json
import pickle
from pathlib import Path
from dataclasses import replace
import tempfile
import unittest
from unittest.mock import patch

import numpy as np
import torch

from rlinit.algorithmhand.door_env import DoorConfig
from rlinit.algorithmhand.door_direct_env import DirectRewardDoorEnv
from rlinit.algorithmhand.door_sac_grasp_env import GraspTrainingConfig, PrecisionGraspDoorEnv, TASK, LEGACY_TASK
from rlinit.algorithmhand.door_sac_state_bank import (
    AdaptiveProgressStates, capture_state, eligible_progress_state, save_bank,
)
from trainp2_sac import load_agent
from trainp2_sac_direct import train


class PrecisionSACTests(unittest.TestCase):
    def test_safe_opening_costs_warn_before_original_failures(self):
        env = PrecisionGraspDoorEnv(training_config=GraspTrainingConfig.safe_opening())
        self.addCleanup(env.close)
        env.reset(seed=0)
        features = list(env._features())
        context = dict(newly_grasped=False, newly_pressed=False, success=False,
                       reason="", peak_bad_force=0, action=np.zeros(8))
        env.arm_target[:] = env.data.qpos[env.qadr]
        env.data.qpos[env.latch] = 1.2
        features[6] = 0.001
        safe = env._reward_components(features=features, **context)
        for key in ("target_error", "penetration", "overpress"):
            self.assertAlmostEqual(safe[key], 0)
        env.arm_target[0] += 0.125
        env.data.qpos[env.latch] = 1.4
        features[6] = 0.003
        risky = env._reward_components(features=features, **context)
        self.assertAlmostEqual(risky["target_error"], -1.0)
        self.assertAlmostEqual(risky["penetration"], -1.0)
        self.assertAlmostEqual(risky["overpress"], -0.5)
        features[6] = 1.0
        env.arm_target[0] += 10
        env.data.qpos[env.latch] = 10
        bounded = env._reward_components(features=features, **context)
        self.assertEqual(bounded["target_error"], -2.0)
        self.assertEqual(bounded["penetration"], -2.0)
        self.assertEqual(bounded["overpress"], -1.0)

    def test_safe_opening_profile_keeps_joint_actions_and_physics_exact(self):
        baseline = PrecisionGraspDoorEnv(training_config=GraspTrainingConfig.joint())
        safer = PrecisionGraspDoorEnv(training_config=GraspTrainingConfig.safe_opening())
        self.addCleanup(baseline.close)
        self.addCleanup(safer.close)
        baseline.reset(seed=5)
        safer.reset(seed=5)
        for action in np.random.default_rng(5).uniform(-1, 1, (20, 8)):
            original = baseline.step(action)
            changed = safer.step(action)
            np.testing.assert_array_equal(baseline.data.qpos, safer.data.qpos)
            np.testing.assert_array_equal(baseline.data.ctrl, safer.data.ctrl)
            np.testing.assert_array_equal(original[0], changed[0])
            self.assertEqual(original[2:4], changed[2:4])
            self.assertEqual(original[4]["is_success"], changed[4]["is_success"])
            self.assertEqual(original[4]["failure_reason"], changed[4]["failure_reason"])

    def test_precision_reward_is_independent_of_algorithm_discount(self):
        environments = [PrecisionGraspDoorEnv(config=DoorConfig(gamma=gamma),
                                             training_config=GraspTrainingConfig.joint())
                        for gamma in (0.99, 0.995)]
        for env in environments:
            self.addCleanup(env.close)
            env.reset(seed=0)
        for success, reason in ((False, ""), (True, ""), (False, "unsafe_collision")):
            components = [env._reward_components(features=env._features(), newly_grasped=True,
                                                newly_pressed=True, success=success, reason=reason,
                                                peak_bad_force=20, action=np.ones(8)) for env in environments]
            self.assertEqual(components[0], components[1])

    def test_progress_pool_keeps_earlier_levels_when_later_visits_dominate(self):
        pool = AdaptiveProgressStates()
        pool.add({"id": "early"}, 0.03)
        for index in range(500):
            pool.add({"id": index}, 0.25)
        self.assertIn(0.03, pool.angles)
        self.assertLessEqual(len(pool.states), pool.capacity)
        rng = np.random.default_rng(0)
        early = sum(pool.angles[pool.sample_index(rng)] < 0.1 for _ in range(500))
        self.assertGreater(early, 175)
        self.assertLess(early, 325)

    def test_adaptive_reset_candidates_exclude_terminal_near_success_and_unsafe_states(self):
        config = DoorConfig()
        info = dict(failure_reason="", grasped=True, left_contact_N=5, right_contact_N=5,
                    door_angle=0.1, handle_angle=1.2, success_hold_seconds=0,
                    penetration_m=0.0005, bad_contact_N=0)
        arguments = dict(terminated=False, truncated=False, target_error=0.05, steps=200, config=config)
        self.assertTrue(eligible_progress_state(info, **arguments))
        for replacement in (dict(door_angle=config.success_angle), dict(success_hold_seconds=0.2),
                            dict(grasped=False), dict(left_contact_N=0), dict(failure_reason="unsafe_collision"),
                            dict(penetration_m=config.failure_penetration), dict(bad_contact_N=50)):
            self.assertFalse(eligible_progress_state(dict(info, **replacement), **arguments))
        for replacement in (dict(terminated=True), dict(truncated=True), dict(target_error=0.2),
                            dict(steps=config.max_steps - 10)):
            self.assertFalse(eligible_progress_state(info, **dict(arguments, **replacement)))

    def test_home_initialization_and_physics_are_unchanged(self):
        base = DirectRewardDoorEnv()
        env = PrecisionGraspDoorEnv()
        self.addCleanup(base.close)
        self.addCleanup(env.close)
        base_obs, _ = base.reset(seed=17)
        obs, info = env.reset(seed=17)
        np.testing.assert_array_equal(base.data.qpos, env.data.qpos)
        np.testing.assert_array_equal(base_obs, obs[:64])
        self.assertEqual(obs.shape, (78,))
        self.assertTrue(env.observation_space.contains(obs))
        self.assertFalse(info["grasped"])
        self.assertEqual(env.config.success_angle, base.config.success_angle)

    def test_policy_actions_use_original_joint_mapping_even_near_handle(self):
        config = DoorConfig(max_steps=30, action_scale=0.01)
        base = DirectRewardDoorEnv(config=config)
        env = PrecisionGraspDoorEnv(config=config)
        self.addCleanup(base.close)
        self.addCleanup(env.close)
        geometry = (np.zeros((2, 3)), np.zeros(3), np.zeros(2), 0.04)
        for seed in (0, 11, 20000):
            base.reset(seed=seed)
            env.reset(seed=seed)
            rng = np.random.default_rng(seed)
            with patch.object(env, "_grasp_geometry", return_value=geometry):
                for _ in range(10):
                    action = rng.uniform(-1, 1, 8)
                    action[7] = 1.0  # Full closure must not be replaced with a slew command.
                    base.step(action)
                    obs, reward, _, _, _ = env.step(action)
                    np.testing.assert_array_equal(env.previous_action, action)
                    np.testing.assert_array_equal(env.arm_target, base.arm_target)
                    np.testing.assert_array_equal(env.data.ctrl, base.data.ctrl)
                    np.testing.assert_array_equal(env.data.qpos, base.data.qpos)
                    np.testing.assert_array_equal(env.data.qvel, base.data.qvel)
                    self.assertTrue(env.observation_space.contains(obs))
                    self.assertTrue(np.isfinite(reward))

    def test_off_center_and_single_sided_contact_are_costly(self):
        env = PrecisionGraspDoorEnv(config=DoorConfig(gamma=0.99))
        self.addCleanup(env.close)
        env.reset(seed=0)
        features = (None, None, 1.0, {"left": 20.0, "right": 0.0}, False, 0, 0)
        geometry = (np.zeros((2, 3)), np.array([0.1, 0, 0]), np.array([0.01, 0.09]), 0.04)
        with patch.object(env, "_grasp_geometry", return_value=geometry):
            parts = env._reward_components(features=features, newly_grasped=False,
                                           newly_pressed=False, success=False, reason="",
                                           peak_bad_force=0, action=np.ones(8))
        self.assertAlmostEqual(parts["distance"], -0.05)
        self.assertAlmostEqual(parts["balance"], -0.04)
        self.assertAlmostEqual(parts["centering"], -0.1)
        self.assertLess(parts["early_close"], -0.099)
        self.assertEqual(parts["contact"], 0.0)
        cfg = env.training_config
        maximum_dense = sum((cfg.orientation_weight, cfg.contact_weight, cfg.grasp_weight,
                             cfg.press_weight, cfg.opening_weight))
        self.assertLess(maximum_dense / (1 - env.config.gamma), cfg.success_bonus)

    def test_log_force_encoding_changes_only_two_observation_entries(self):
        linear = PrecisionGraspDoorEnv()
        logged = PrecisionGraspDoorEnv(training_config=GraspTrainingConfig(contact_observation="log"))
        self.addCleanup(linear.close)
        self.addCleanup(logged.close)
        linear.reset(seed=0)
        logged.reset(seed=0)
        contacts = ({"left": 2000.0, "right": 10.0}, 0.0, 0.0)
        with patch.object(linear, "_contacts", return_value=contacts), patch.object(logged, "_contacts", return_value=contacts):
            original, transformed = linear._obs(), logged._obs()
        expected = original.copy()
        expected[62:64] = np.log1p(original[62:64])
        np.testing.assert_array_equal(transformed, expected)
        self.assertLess(transformed[62], 5.0)

    def test_removed_controller_options_cannot_start_training(self):
        for option in ("cartesian_actions", "handle_pose_actions"):
            with self.assertRaisesRegex(ValueError, "removed"):
                train(precision_grasp=True, **{option: True})
        self.assertFalse(hasattr(PrecisionGraspDoorEnv, "_cartesian_to_joints"))

    def test_subtask_requires_stable_contact_and_is_not_full_opening(self):
        env = PrecisionGraspDoorEnv(training_config=GraspTrainingConfig(training_stage="grasp"))
        self.addCleanup(env.close)
        env.reset(seed=0)
        features = list(env._features())
        features[2:5] = [1.0, {"left": 1.0, "right": 1.0}, True]
        geometry = (np.zeros((2, 3)), np.zeros(3), np.array([0.01, 0.01]), 0.04)
        with patch.object(env, "_grasp_geometry", return_value=geometry), patch.object(env, "_features", return_value=tuple(features)):
            self.assertFalse(env._is_success(grasp=True, reason=""))
            for _ in range(20):
                if env._is_success(grasp=True, reason=""):
                    break
            else:
                self.fail("stable contact did not complete grasp subtask")
            info = env._info(success=True)
            self.assertTrue(info["stage_success"])
            self.assertFalse(info["is_success"])
            self.assertFalse(env._is_success(grasp=False, reason=""))
            self.assertEqual(env.success_time, 0.0)

    def test_strong_reward_penalizes_grasping_while_turning_the_wrong_way(self):
        env = PrecisionGraspDoorEnv(training_config=GraspTrainingConfig.strong())
        self.addCleanup(env.close)
        env.reset(seed=0)
        env.data.qpos[env.latch] = -0.65
        features = list(env._features())
        features[2] = 1.0
        features[4] = True
        geometry = (np.zeros((2, 3)), np.zeros(3), np.array([0.01, 0.01]), 0.04)
        with patch.object(env, "_grasp_geometry", return_value=geometry):
            parts = env._reward_components(features=features, newly_grasped=False, newly_pressed=False,
                                           success=False, reason="", peak_bad_force=0,
                                           action=env.previous_action)
        self.assertAlmostEqual(parts["pressing"], -0.25)
        features[3] = {"left": 2.0, "right": 2.0}
        parts = env._reward_components(features=features, newly_grasped=False, newly_pressed=False,
                                       success=False, reason="", peak_bad_force=0,
                                       action=np.ones(8))
        self.assertEqual(parts["early_close"], 0.0)
        features[3] = {"left": 2.0, "right": 0.0}
        features[4] = False
        parts = env._reward_components(features=features, newly_grasped=False, newly_pressed=False,
                                       success=False, reason="", peak_bad_force=0,
                                       action=np.ones(8))
        self.assertLess(parts["early_close"], -0.49)

    def test_ready_open_gripper_can_close_before_making_contact(self):
        env = PrecisionGraspDoorEnv(training_config=GraspTrainingConfig.strong())
        self.addCleanup(env.close)
        env.reset(seed=0)
        features = (None, None, 0.91, {"left": 0.0, "right": 0.0}, False, 0, 0)
        geometry = (np.zeros((2, 3)), np.array([0.018, 0, 0]), np.array([0.01, 0.04]), 0.078)
        with patch.object(env, "_grasp_geometry", return_value=geometry):
            parts = env._reward_components(features=features, newly_grasped=False,
                                           newly_pressed=False, success=False, reason="",
                                           peak_bad_force=0, action=np.ones(8))
        self.assertEqual(parts["early_close"], 0.0)

    def test_closure_reward_requires_geometric_readiness_without_overriding_actions(self):
        env = PrecisionGraspDoorEnv(training_config=GraspTrainingConfig.joint())
        self.addCleanup(env.close)
        env.reset(seed=0)
        features = (None, None, 0.97, {"left": 0.0, "right": 0.0}, False, 0, 0)
        geometry = (np.zeros((2, 3)), np.array([0.017, 0, 0]), np.array([0.027, 0.027]), 0.084)
        context = dict(features=features, newly_grasped=False, newly_pressed=False,
                       success=False, reason="", peak_bad_force=0)
        with patch.object(env, "_grasp_geometry", return_value=geometry):
            closed = env._reward_components(action=np.ones(8), **context)
            opened = env._reward_components(action=np.r_[np.ones(7), -1], **context)
        self.assertEqual(closed["closure"], 0.5)
        self.assertEqual(opened["closure"], 0.0)
        far = (np.zeros((2, 3)), np.array([0.2, 0, 0]), np.array([0.2, 0.2]), 0.084)
        with patch.object(env, "_grasp_geometry", return_value=far):
            premature = env._reward_components(action=np.ones(8), **context)
        self.assertLess(premature["closure"], 1e-20)
        self.assertLess(premature["early_close"], -0.49)

    def test_reach_goal_is_precise_open_and_noncontact(self):
        env = PrecisionGraspDoorEnv(training_config=GraspTrainingConfig.strong(training_stage="reach"))
        self.addCleanup(env.close)
        env.reset(seed=0)
        features = list(env._features())
        features[2] = 1.0
        geometry = (np.zeros((2, 3)), np.zeros(3), np.array([0.02, 0.02]), 0.08)
        with patch.object(env, "_grasp_geometry", return_value=geometry), patch.object(env, "_features", return_value=tuple(features)):
            self.assertFalse(env._is_success(grasp=False, reason=""))
            for _ in range(10):
                if env._is_success(grasp=False, reason=""):
                    break
            else:
                self.fail("precise open pose did not complete reach subtask")
            env.data.ctrl[env.gripper] = env.config.grip_max
            self.assertFalse(env._is_success(grasp=False, reason=""))

    def test_parallel_train_checkpoint_evaluation_and_resume(self):
        previous_threads = torch.get_num_threads()
        self.addCleanup(torch.set_num_threads, previous_threads)
        with tempfile.TemporaryDirectory() as folder:
            common = dict(precision_grasp=True, orientation_cost_weight=0.5,
                          gripper_exploration_std=0.8,
                          grasp_reward_profile="safe_opening",
                          output_dir=folder, env_config=DoorConfig(max_steps=4),
                          learning_starts=2, buffer_size=64, batch_size=4, n_envs=2,
                          device="cpu", total_timesteps=8, eval_freq=8, eval_episodes=1,
                          save_replay_buffer=True, policy_kwargs={"net_arch": [16, 16]})
            checkpoint = train(**common)
            agent, env = load_agent(checkpoint, device="cpu")
            self.addCleanup(env.close)
            self.assertEqual(env.specification()["task"], TASK)
            self.assertEqual(agent.num_timesteps, 8)
            self.assertIsNotNone(agent.action_noise)
            self.assertEqual(env.observation_space.shape, (78,))
            self.assertIn("policy joint increments", env.specification()["action_source"])
            self.assertEqual(env.training_config.success_bonus, 1000.0)
            self.assertEqual(env.training_config.target_error_weight, 2.0)
            self.assertEqual(env.training_config.penetration_weight, 2.0)
            self.assertEqual(env.training_config.overpress_weight, 0.5)
            self.assertTrue((Path(folder) / "best" / "best_model.zip").exists())
            common.update(resume_checkpoint=checkpoint,
                          replay_buffer_path=Path(folder) / "replay_buffer.pkl",
                          grasp_reward_profile="balanced")
            train(**common)
            config = json.loads((Path(folder) / "config.json").read_text())
            self.assertEqual(config["actual_timesteps"], 16)
            self.assertNotIn("demonstrations", config)
            source, source_env = load_agent(checkpoint, device="cpu")
            self.assertEqual(source_env.training_config.success_bonus, 1000.0)
            self.assertEqual(source_env.training_config.target_error_weight, 2.0)
            source_env.close()
            initialization = dict(common, output_dir=Path(folder) / "initialized", n_envs=1,
                                  total_timesteps=4, learning_starts=100,
                                  initialize_from=checkpoint, training_stage="grasp")
            initialization.pop("resume_checkpoint")
            initialization.pop("replay_buffer_path")
            with (Path(folder) / "replay_buffer.pkl").open("rb") as stream:
                old_replay = pickle.load(stream)
            reuse = dict(initialization, output_dir=Path(folder) / "reuse_online",
                         n_envs=2, training_stage="full", grasp_reward_profile="safe_opening",
                         gamma=0.995, total_timesteps=8, policy_warmup_steps=4,
                         success_replay_fraction=0.25,
                         replay_buffer_path=Path(folder) / "replay_buffer.pkl")
            reused_checkpoint = train(**reuse)
            reused, reused_environment = load_agent(reused_checkpoint, device="cpu")
            self.assertEqual(reused.gamma, 0.995)
            self.assertEqual(reused.num_timesteps, 8)
            self.assertEqual(reused.door_success_replay_fraction, 0.25)
            self.assertEqual(reused_environment.config.gamma, 0.995)
            reused_environment.close()
            with (Path(folder) / "reuse_online" / "replay_buffer.pkl").open("rb") as stream:
                new_replay = pickle.load(stream)
            self.assertEqual(new_replay.pos, old_replay.pos + 4)
            np.testing.assert_array_equal(new_replay.observations[0], old_replay.observations[0])
            np.testing.assert_array_equal(new_replay.rewards[0], old_replay.rewards[0])
            multistep = dict(reuse, output_dir=Path(folder) / "multistep",
                             success_replay_fraction=0, n_steps=3, learning_starts=0)
            multistep_checkpoint = train(**multistep)
            multistep_agent, multistep_env = load_agent(multistep_checkpoint, device="cpu")
            multistep_env.close()
            self.assertEqual(multistep_agent.n_steps, 3)
            self.assertEqual(multistep_agent._n_updates, 2)
            with (Path(folder) / "multistep" / "replay_buffer.pkl").open("rb") as stream:
                multistep_buffer = pickle.load(stream)
            self.assertEqual(multistep_buffer.n_steps, 3)
            np.testing.assert_array_equal(multistep_buffer.rewards[0], old_replay.rewards[0])
            continuation = dict(common, output_dir=Path(folder) / "multistep_resume",
                                resume_checkpoint=multistep_checkpoint,
                                replay_buffer_path=Path(folder) / "multistep" / "replay_buffer.pkl",
                                gamma=0.995)
            continued = train(**continuation)
            continued_agent, continued_env = load_agent(continued, device="cpu")
            continued_env.close()
            self.assertEqual(continued_agent.n_steps, 3)
            self.assertEqual(continued_agent.num_timesteps, 16)
            with self.assertRaisesRegex(ValueError, "matching online precision SAC"):
                train(**dict(reuse, training_stage="grasp", success_replay_fraction=0))
            initialized = train(**initialization)
            target, target_env = load_agent(initialized, device="cpu")
            self.assertEqual(target_env.training_config.training_stage, "grasp")
            target_env.close()
            _, full_env = load_agent(initialized, device="cpu", full_task=True)
            self.assertEqual(full_env.training_config.training_stage, "full")
            self.assertFalse(full_env._is_success(grasp=True, reason=""))
            full_env.close()
            self.assertEqual(target._n_updates, 0)
            for key, value in source.policy.state_dict().items():
                torch.testing.assert_close(value, target.policy.state_dict()[key])
            initial_config = json.loads((Path(folder) / "initialized" / "config.json").read_text())
            self.assertNotIn("demonstrations", initial_config)
            self.assertEqual(initial_config["policy_initialization"], str(checkpoint))
            legacy_checkpoint = Path(folder) / "legacy_controller.zip"
            source.door_environment = dict(source.door_environment, task=LEGACY_TASK)
            source.save(legacy_checkpoint)
            with self.assertRaisesRegex(ValueError, "Legacy precision checkpoint"):
                load_agent(legacy_checkpoint, device="cpu")
            with self.assertRaisesRegex(ValueError, "Legacy precision checkpoint"):
                train(**dict(common, resume_checkpoint=legacy_checkpoint))
            with self.assertRaisesRegex(ValueError, "Legacy precision checkpoint"):
                train(**dict(initialization, initialize_from=legacy_checkpoint))
            initialization.update(output_dir=Path(folder) / "rescaled", initialize_critic_scale=0.1)
            scaled_checkpoint = train(**initialization)
            scaled, scaled_env = load_agent(scaled_checkpoint, device="cpu")
            scaled_env.close()
            observation = torch.zeros((1, 78))
            action = torch.zeros((1, 8))
            with torch.no_grad():
                for original_value, scaled_value in zip(source.critic(observation, action), scaled.critic(observation, action)):
                    torch.testing.assert_close(scaled_value, original_value * 0.1)
            for key, value in source.actor.state_dict().items():
                torch.testing.assert_close(value, scaled.actor.state_dict()[key])
            initialization.update(output_dir=Path(folder) / "log_forces", initialize_critic_scale=1.0,
                                  contact_observation="log")
            with self.assertRaisesRegex(ValueError, "matching physics/actions"):
                train(**initialization)
            initialization["initialize_without_force_inputs"] = True
            logged_checkpoint = train(**initialization)
            logged, logged_env = load_agent(logged_checkpoint, device="cpu")
            self.assertEqual(logged_env.training_config.contact_observation, "log")
            logged_env.close()
            with torch.no_grad():
                torch.testing.assert_close(source.actor(observation, deterministic=True),
                                           logged.actor(observation, deterministic=True))
                for before, after in zip(source.critic(observation, action), logged.critic(observation, action)):
                    torch.testing.assert_close(before, after)
            for network in (logged.actor.latent_pi, *logged.critic.q_networks,
                            *logged.critic_target.q_networks):
                first = next(layer for layer in network.modules() if isinstance(layer, torch.nn.Linear))
                self.assertEqual(torch.count_nonzero(first.weight[:, 62:64]).item(), 0)

            # Real simulator-state replay changes only training starts; the
            # learned checkpoint remains independently evaluable from home.
            _, origin = load_agent(logged_checkpoint, device="cpu")
            self.addCleanup(origin.close)
            observation, _ = origin.reset(seed=71)
            policy_action, _ = logged.predict(observation, deterministic=True)
            observation, _, _, _, _ = origin.step(policy_action)
            snapshot = capture_state(origin)
            bank = Path(folder) / "own_states.npz"
            save_bank(bank, origin, [snapshot], checkpoint=logged_checkpoint)
            replay = PrecisionGraspDoorEnv(config=replace(origin.config, gamma=0.995), training_config=origin.training_config)
            self.addCleanup(replay.close)
            replay.enable_training_resets(bank, 1.0, adaptive=True)
            restored, info = replay.reset(seed=99)
            self.assertEqual(info["reset_origin"], "own_online_sac_state")
            np.testing.assert_allclose(restored, observation, atol=1e-6)
            np.testing.assert_array_equal(replay.arm_target, origin.arm_target)
            self.assertEqual(replay.steps, origin.steps)
            for simulator in (origin, replay):
                simulator.step(np.zeros(8))
            np.testing.assert_allclose(origin.data.qpos, replay.data.qpos, atol=1e-10)
            self.assertTrue(replay.training_reset_specification()["adaptive"]["enabled"])
            bank_run = dict(common, output_dir=Path(folder) / "state_bank_run", contact_observation="log",
                            training_reset_bank=bank, training_reset_probability=1.0,
                            adaptive_training_resets=True)
            bank_run.pop("resume_checkpoint")
            bank_run.pop("replay_buffer_path")
            bank_checkpoint = train(**bank_run)
            bank.rename(bank.with_suffix(".hidden"))
            bank_agent, independent = load_agent(bank_checkpoint, device="cpu")
            self.addCleanup(independent.close)
            _, independent_info = independent.reset(seed=99)
            self.assertEqual(independent_info["reset_origin"], "home")
            self.assertEqual(independent.steps, 0)
            self.assertIsNone(independent.training_reset_specification())
            self.assertEqual(bank_agent.door_training_reset_curriculum["states"], 1)


if __name__ == "__main__":
    unittest.main()
