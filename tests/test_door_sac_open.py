"""Check frozen-recipe fidelity and real train/save/load/resume behavior."""

from contextlib import ExitStack
from dataclasses import asdict, replace
import json
from pathlib import Path
import pickle
import sys
import tempfile
import unittest
from unittest.mock import patch

import numpy as np
import torch
import train_sac_open

from rlinit.algorithmhand.door_env import DoorConfig
from rlinit.algorithmhand.door_sac_grasp_env import GraspTrainingConfig, PrecisionGraspDoorEnv
from rlinit.algorithmhand.door_sac_open_env import SACOpenDoorEnv, OpenRewardConfig, default_door_config
from train_sac_open import SACOpenSettings, evaluate, load_agent, train


class SACOpenTests(unittest.TestCase):
    def test_render_training_updates_saves_and_closes_one_viewer(self):
        previous_threads = torch.get_num_threads()
        self.addCleanup(torch.set_num_threads, previous_threads)
        settings = replace(SACOpenSettings(), buffer_size=64, batch_size=4, scratch_learning_starts=2)
        with tempfile.TemporaryDirectory() as directory, patch("mujoco.viewer.launch_passive") as launch:
            viewer = launch.return_value
            # Closing the window midway must not interrupt learning or open another window.
            viewer.is_running.side_effect = [True, True] + [False] * 100
            checkpoint = train(
                output_dir=directory, total_timesteps=8, render=True, device="cpu", settings=settings,
                env_config=replace(default_door_config(), max_steps=4), eval_freq=4, eval_episodes=1,
            )
            launch.assert_called_once()
            viewer.sync.assert_called()
            viewer.close.assert_called_once()
            metadata = json.loads((Path(directory) / "config.json").read_text())
            self.assertEqual(metadata["n_envs"], 1)
            self.assertTrue(metadata["render"])
            self.assertEqual(metadata["actual_timesteps"], 8)
            self.assertGreater(metadata["gradient_updates"], 0)
            self.assertTrue(checkpoint.is_file())
            self.assertTrue((Path(directory) / "best/best_model.zip").is_file())
            self.assertTrue((Path(directory) / "evaluation.json").is_file())
        with self.assertRaisesRegex(ValueError, "--n-envs 1"):
            train(render=True, n_envs=2)

    def test_cli_accepts_training_render_and_preserves_evaluation_render(self):
        previous_threads = torch.get_num_threads()
        self.addCleanup(torch.set_num_threads, previous_threads)
        with patch.object(sys, "argv", ["train_sac_open.py", "--timesteps", "1000000", "--render"]), \
                patch.object(train_sac_open, "train") as training:
            train_sac_open.main()
            self.assertTrue(training.call_args.kwargs["render"])
            self.assertIsNone(training.call_args.kwargs["n_envs"])
            self.assertEqual(training.call_args.kwargs["total_timesteps"], 1000000)
        with patch.object(sys, "argv", ["train_sac_open.py", "--evaluate", "model.zip", "--render"]), \
                patch.object(train_sac_open, "evaluate", return_value={}) as evaluation:
            train_sac_open.main()
            self.assertTrue(evaluation.call_args.kwargs["render"])

    def test_matches_previous_recipe_with_explicit_legacy_press_angles(self):
        config = default_door_config()
        reference = PrecisionGraspDoorEnv(config=config, training_config=GraspTrainingConfig.joint(
            orientation_cost_weight=0.5, contact_observation="log"))
        frozen = SACOpenDoorEnv(reward_config=OpenRewardConfig(press_target_angle=1.3, press_milestone_angle=1.1))
        self.addCleanup(reference.close)
        self.addCleanup(frozen.close)
        for seed in (0, 17):
            first = reference.reset(seed=seed)[0]
            second = frozen.reset(seed=seed)[0]
            np.testing.assert_array_equal(first, second)
            for action in np.random.default_rng(seed).uniform(-1, 1, (20, 8)):
                old, new = reference.step(action), frozen.step(action)
                np.testing.assert_array_equal(old[0], new[0])
                np.testing.assert_array_equal(reference.data.qpos, frozen.data.qpos)
                np.testing.assert_array_equal(reference.data.ctrl, frozen.data.ctrl)
                self.assertAlmostEqual(old[1], new[1], places=12)
                self.assertEqual(old[2:4], new[2:4])
            # Exercise contact/press/open rewards and terminal bonuses, not just home states.
            reference.data.qpos[reference.latch] = frozen.data.qpos[frozen.latch] = 1.2
            reference.data.qpos[reference.hinge] = frozen.data.qpos[frozen.hinge] = 0.2
            features = (None, None, .93, {"left": 50., "right": 20.}, True, 0., .001)
            for success, reason in ((False, ""), (True, ""), (False, "unsafe_collision")):
                args = dict(features=features, newly_grasped=True, newly_pressed=True, success=success,
                            reason=reason, peak_bad_force=40, action=np.ones(8))
                old, new = reference._reward_components(**args), frozen._reward_components(**args)
                self.assertEqual(old.keys(), new.keys())
                for key in old:
                    self.assertAlmostEqual(old[key], new[key], places=12)
        self.assertFalse(hasattr(frozen, "enable_training_resets"))

    def test_45_degree_press_reward_and_once_per_episode_milestone(self):
        env = SACOpenDoorEnv()
        self.addCleanup(env.close)
        env.reset(seed=0)
        self.assertAlmostEqual(env.reward_config.press_target_angle, np.pi / 4)
        self.assertAlmostEqual(env.reward_config.press_milestone_angle, np.pi / 4)
        self.assertLess(env.reward_config.press_target_angle, env.model.joint("door_latch_joint").range[1])
        quality = .93 ** 4
        press_index = env.model.nq + env.model.nv + 7 + 8 + 1
        # Hold exact synthetic contact states to check the reward boundary;
        # separate integration tests exercise real physics and SAC updates.
        with ExitStack() as stack:
            stack.enter_context(patch("mujoco.mj_step"))
            stack.enter_context(patch("mujoco.mj_forward"))
            features = stack.enter_context(patch.object(env, "_features"))
            stack.enter_context(patch.object(env, "_contacts", return_value=({"left": 2., "right": 2.}, 0., 0.)))
            stack.enter_context(patch.object(env, "_grasp_geometry", return_value=(
                np.zeros((2, 3)), np.zeros(3), np.zeros(2), .04)))
            for angle, grasp, expected_press, expected_milestone in (
                    (0., True, 0., 0.),
                    (np.pi / 8, True, .25, 0.),
                    (np.nextafter(np.pi / 4, 0), True, .5, 0.),
                    (np.pi / 4, False, 0., 0.),
                    (np.pi / 4, True, .5, 3.),
                    (np.pi / 3, True, .5, 0.),
                    (0., True, 0., 0.),
                    (np.pi / 4, True, .5, 0.)):
                with self.subTest(angle=angle, grasp=grasp):
                    features.return_value = (np.zeros(3), np.eye(3), .93, {"left": 2., "right": 2.}, grasp, 0., 0.)
                    env.data.qpos[env.latch] = angle
                    obs, _, _, _, info = env.step(np.zeros(8))
                    self.assertAlmostEqual(info["reward_components"]["pressing"], expected_press * quality)
                    self.assertAlmostEqual(info["reward_components"]["press_milestone"], expected_milestone * quality)
                    self.assertEqual(obs[press_index], info["press_seen"])
            env.reset(seed=0)
            self.assertFalse(env.press_seen)
            env.data.qpos[env.latch] = np.pi / 4
            self.assertAlmostEqual(env.step(np.zeros(8))[4]["reward_components"]["press_milestone"], 3 * quality)

    def test_saved_press_angles_restore_without_reinterpreting_old_rewards(self):
        env = SACOpenDoorEnv()
        self.addCleanup(env.close)
        new_spec = env.specification()
        self.assertEqual(train_sac_open._saved_reward(new_spec), OpenRewardConfig())
        new_spec["reward_config"].pop("press_target_angle")
        new_spec["reward_config"].pop("press_milestone_angle")
        saved = train_sac_open._saved_reward(new_spec)
        self.assertEqual(saved.press_target_angle, 1.3)
        self.assertEqual(saved.press_milestone_angle, 1.1)
        self.assertNotEqual(saved, OpenRewardConfig())
        legacy_training = dict(new_spec["reward_config"], training_stage="full", contact_observation="log",
                               signed_press_reward=True, contact_releases_close_cost=True,
                               quality_gated_reward=True, reward_scale=1.0)
        self.assertEqual(train_sac_open._saved_reward(dict(task=train_sac_open.LEGACY_TASK,
                                                          training_config=legacy_training)), saved)
        for field in ("press_target_angle", "press_milestone_angle"):
            for invalid in (0, -1, float("nan"), float("inf")):
                with self.subTest(field=field, invalid=invalid), self.assertRaises(ValueError):
                    OpenRewardConfig(**{field: invalid})

    def test_train_restore_resume_initialize_and_guard_replay(self):
        previous_threads = torch.get_num_threads()
        self.addCleanup(torch.set_num_threads, previous_threads)
        torch.set_num_threads(1)
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            settings = replace(SACOpenSettings(), buffer_size=64, batch_size=4, scratch_learning_starts=2)
            common = dict(total_timesteps=8, n_envs=2, device="cpu", settings=settings,
                          env_config=replace(default_door_config(), max_steps=4), eval_freq=8, eval_episodes=1)
            first = train(output_dir=root / "scratch", **common)
            agent, env = load_agent(first)
            env.close()
            self.assertGreater(agent._n_updates, 0)
            self.assertEqual(agent.n_steps, 3)
            self.assertEqual(agent.open_settings, asdict(settings))
            self.assertEqual(agent.door_environment["reward_config"]["press_target_angle"], np.pi / 4)
            self.assertEqual(agent.door_environment["reward_config"]["press_milestone_angle"], np.pi / 4)
            self.assertTrue((root / "scratch/best/best_model.zip").is_file())
            replay = root / "scratch/replay_buffer.pkl"
            with replay.open("rb") as stream:
                old_replay = pickle.load(stream)
            resumed = train(output_dir=root / "resumed", resume_checkpoint=first,
                            replay_buffer_path=replay, **common)
            resumed_agent, resumed_env = load_agent(resumed)
            resumed_env.close()
            self.assertEqual(resumed_agent.num_timesteps, 16)
            self.assertGreater(resumed_agent._n_updates, agent._n_updates)
            self.assertEqual(resumed_agent.door_environment["reward_config"], agent.door_environment["reward_config"])
            initialized = train(output_dir=root / "initialized", initialize_from=first,
                                replay_buffer_path=replay, **common)
            with (root / "initialized/replay_buffer.pkl").open("rb") as stream:
                new_replay = pickle.load(stream)
            self.assertEqual(new_replay.n_steps, 3)
            np.testing.assert_array_equal(new_replay.rewards[0], old_replay.rewards[0])
            # Evaluation needs only the checkpoint and scene, not the adjacent config/replay.
            (root / "initialized/config.json").unlink()
            (root / "initialized/replay_buffer.pkl").unlink()
            self.assertEqual(evaluate(initialized, episodes=2, seed=123), evaluate(initialized, episodes=2, seed=123))
            with self.assertRaisesRegex(ValueError, "not empty"):
                train(output_dir=root / "scratch", **common)
            with self.assertRaisesRegex(ValueError, "saved replay"):
                train(output_dir=root / "bad_resume", resume_checkpoint=first, **common)
            metadata = json.loads((root / "scratch/config.json").read_text())
            legacy_metadata = json.loads(json.dumps(metadata))
            legacy_metadata["environment"]["reward_config"].pop("press_target_angle")
            legacy_metadata["environment"]["reward_config"].pop("press_milestone_angle")
            (root / "scratch/config.json").write_text(json.dumps(legacy_metadata))
            with self.assertRaisesRegex(ValueError, "matching home-only"):
                train(output_dir=root / "old_press_replay", initialize_from=first, replay_buffer_path=replay, **common)
            metadata["demonstrations"] = {"unexpected": True}
            (root / "scratch/config.json").write_text(json.dumps(metadata))
            with self.assertRaisesRegex(ValueError, "matching home-only"):
                train(output_dir=root / "bad_replay", initialize_from=first, replay_buffer_path=replay, **common)

    def test_coefficients_validate_and_do_not_override_original_success(self):
        with self.assertRaises(ValueError):
            OpenRewardConfig(ready_distance=0)
        with self.assertRaises(ValueError):
            SACOpenSettings(n_steps=0)
        env = SACOpenDoorEnv()
        self.addCleanup(env.close)
        env.reset(seed=0)
        env.data.qpos[env.hinge] = env.config.success_angle
        for _ in range(14):
            self.assertFalse(env._is_success(grasp=True, reason=""))
        self.assertTrue(env._is_success(grasp=True, reason=""))
        self.assertFalse(env._is_success(grasp=True, reason="unsafe_collision"))
        self.assertEqual(env.success_time, 0)


if __name__ == "__main__":
    unittest.main()
