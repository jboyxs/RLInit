"""Check the press-only difference and real training/checkpoint lifecycle."""

from contextlib import ExitStack
from dataclasses import asdict, replace
import json
from pathlib import Path
import tempfile
import unittest
from unittest.mock import patch

import numpy as np
import torch

from rlinit.algorithmhand.door_sac_open_env import OpenRewardConfig, SACOpenDoorEnv, default_door_config
from rlinit.algorithmhand.door_sac_open_press_env import TASK, OpenPressRewardConfig, SACOpenPressDoorEnv
import train_sac_open as original
from train_sac_open_press import FORMAT, SACOpenSettings, evaluate, load_agent, train


class SACOpenPressTests(unittest.TestCase):
    def test_defaults_and_real_physics_match_original(self):
        reward = asdict(OpenPressRewardConfig())
        self.assertEqual(reward.pop("press_milestone_angle"), 1.25)
        self.assertEqual(reward, asdict(OpenRewardConfig()))
        self.assertEqual(asdict(SACOpenSettings()), asdict(original.SACOpenSettings()))
        old, new = SACOpenDoorEnv(), SACOpenPressDoorEnv()
        self.addCleanup(old.close)
        self.addCleanup(new.close)
        self.assertEqual(old.config, new.config)
        self.assertEqual(old.action_space, new.action_space)
        self.assertEqual(old.observation_space, new.observation_space)
        for seed in (0, 17):
            np.testing.assert_array_equal(old.reset(seed=seed)[0], new.reset(seed=seed)[0])
            for action in np.random.default_rng(seed).uniform(-1, 1, (20, 8)):
                first, second = old.step(action), new.step(action)
                np.testing.assert_array_equal(first[0], second[0])
                np.testing.assert_array_equal(old.data.qpos, new.data.qpos)
                np.testing.assert_array_equal(old.data.qvel, new.data.qvel)
                np.testing.assert_array_equal(old.data.ctrl, new.data.ctrl)
                self.assertEqual(first[1:4], second[1:4])
                self.assertEqual(first[4]["reward_components"], second[4]["reward_components"])
        self.assertEqual(new.specification()["reward_config"]["press_milestone_angle"], 1.25)
        self.assertEqual(new.specification()["task"], TASK)

    def test_only_press_milestone_changes_at_boundary_and_pays_once(self):
        old, new = SACOpenDoorEnv(), SACOpenPressDoorEnv()
        self.addCleanup(old.close)
        self.addCleanup(new.close)
        old.reset(seed=0)
        new.reset(seed=0)
        quality = 0.93 ** 4
        press_index = new.model.nq + new.model.nv + 7 + 8 + 1
        # Isolate reward/flag bookkeeping at exact angles. Physics equivalence is
        # checked above; synthetic contact here avoids controller tracking error.
        with ExitStack() as stack:
            stack.enter_context(patch("mujoco.mj_step"))
            stack.enter_context(patch("mujoco.mj_forward"))
            features = []
            for env in (old, new):
                features.append(stack.enter_context(patch.object(env, "_features")))
                stack.enter_context(patch.object(env, "_contacts", return_value=({"left": 2., "right": 2.}, 0., 0.)))
                stack.enter_context(patch.object(env, "_grasp_geometry", return_value=(
                    np.zeros((2, 3)), np.zeros(3), np.zeros(2), 0.04)))
            for angle, grasp, old_paid, new_paid in (
                    (1.1, False, False, False),
                    (1.1, True, True, False),
                    (1.249999, True, False, False),
                    (1.25, False, False, False),
                    (1.25, True, False, True),
                    (1.3, True, False, False),
                    (1.0, True, False, False),
                    (1.25, True, False, False)):
                with self.subTest(angle=angle, grasp=grasp):
                    for env, mock in zip((old, new), features):
                        env.data.qpos[env.latch] = angle
                        mock.return_value = (np.zeros(3), np.eye(3), .93, {"left": 2., "right": 2.}, grasp, 0., 0.)
                    first, second = old.step(np.zeros(8)), new.step(np.zeros(8))
                    old_components = dict(first[4]["reward_components"])
                    new_components = dict(second[4]["reward_components"])
                    self.assertAlmostEqual(old_components.pop("press_milestone"), 3 * quality * old_paid)
                    self.assertAlmostEqual(new_components.pop("press_milestone"), 3 * quality * new_paid)
                    self.assertEqual(old_components, new_components)
                    self.assertAlmostEqual(new_components["pressing"], .5 * grasp * np.clip(angle / 1.3, -1, 1) * quality)
                    np.testing.assert_array_equal(np.delete(first[0], press_index), np.delete(second[0], press_index))
                    self.assertEqual(second[0][press_index], second[4]["press_seen"])
            new.reset(seed=0)
            self.assertFalse(new.press_seen)
            new.data.qpos[new.latch] = 1.25
            self.assertAlmostEqual(new.step(np.zeros(8))[4]["reward_components"]["press_milestone"], 3 * quality)

    def test_success_and_failure_rules_are_inherited(self):
        env = SACOpenPressDoorEnv()
        self.addCleanup(env.close)
        env.reset(seed=0)
        env.data.qpos[env.latch] = 1.3
        self.assertFalse(env._is_success(grasp=True, reason=""))
        env.data.qpos[env.hinge] = env.config.success_angle
        for _ in range(14):
            self.assertFalse(env._is_success(grasp=True, reason=""))
        self.assertTrue(env._is_success(grasp=True, reason=""))
        self.assertFalse(env._is_success(grasp=True, reason="unsafe_collision"))
        self.assertFalse(env._is_success(grasp=False, reason=""))
        for invalid in (0, -1, float("nan"), float("inf")):
            with self.subTest(invalid=invalid), self.assertRaises(ValueError):
                OpenPressRewardConfig(press_milestone_angle=invalid)

    def test_parallel_train_save_evaluate_resume_and_initialize(self):
        previous_threads = torch.get_num_threads()
        self.addCleanup(torch.set_num_threads, previous_threads)
        torch.set_num_threads(1)
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            settings = replace(SACOpenSettings(), buffer_size=64, batch_size=4, scratch_learning_starts=2)
            reward = OpenPressRewardConfig(press_milestone_angle=1.27)
            common = dict(total_timesteps=8, n_envs=2, device="cpu", settings=settings,
                          env_config=replace(default_door_config(), max_steps=4), eval_freq=8, eval_episodes=1)
            first = train(output_dir=root / "scratch", reward_config=reward, **common)
            agent, env = load_agent(first)
            self.addCleanup(env.close)
            self.assertEqual(agent.door_task_format, FORMAT)
            self.assertEqual(env.reward_config, reward)
            self.assertGreater(agent._n_updates, 0)
            self.assertTrue((first.parent / "best/best_model.zip").is_file())
            metadata = json.loads((first.parent / "config.json").read_text())
            self.assertEqual(metadata["environment"]["reward_config"]["press_milestone_angle"], 1.27)
            self.assertIn(str(Path("src/rlinit/algorithmhand/door_sac_open_press_env.py")), metadata["recipe_sha256"])
            replay = first.parent / "replay_buffer.pkl"
            resumed = train(output_dir=root / "resumed", resume_checkpoint=first, replay_buffer_path=replay, **common)
            restored, restored_env = load_agent(resumed)
            self.addCleanup(restored_env.close)
            self.assertEqual(restored.num_timesteps, 16)
            self.assertGreater(restored._n_updates, agent._n_updates)
            self.assertEqual(restored_env.reward_config, reward)
            initialized = train(output_dir=root / "initialized", initialize_from=first, replay_buffer_path=replay,
                                reward_config=reward, **common)
            self.assertEqual(evaluate(initialized, episodes=1, seed=123), evaluate(initialized, episodes=1, seed=123))
            with self.assertRaisesRegex(ValueError, "checkpoint format"):
                original.load_agent(first)
            # Do not accept replay whose rewards were recorded under the old task.
            metadata["environment"]["task"] = "fixed_base_door_sac_open_v1"
            (first.parent / "config.json").write_text(json.dumps(metadata))
            with self.assertRaisesRegex(ValueError, "press-threshold"):
                train(output_dir=root / "bad_replay", initialize_from=first, replay_buffer_path=replay,
                      reward_config=reward, **common)

    def test_render_training_still_uses_one_viewer_and_finishes(self):
        previous_threads = torch.get_num_threads()
        self.addCleanup(torch.set_num_threads, previous_threads)
        settings = replace(SACOpenSettings(), buffer_size=64, batch_size=4, scratch_learning_starts=2)
        with tempfile.TemporaryDirectory() as directory, patch("mujoco.viewer.launch_passive") as launch:
            launch.return_value.is_running.side_effect = [True, True] + [False] * 100
            checkpoint = train(output_dir=directory, total_timesteps=8, render=True, device="cpu", settings=settings,
                               env_config=replace(default_door_config(), max_steps=4), eval_freq=4, eval_episodes=1)
            self.assertTrue(checkpoint.is_file())
            launch.assert_called_once()
            launch.return_value.sync.assert_called()
            launch.return_value.close.assert_called_once()
            metadata = json.loads((Path(directory) / "config.json").read_text())
            self.assertEqual(metadata["n_envs"], 1)
            self.assertTrue(metadata["render"])


if __name__ == "__main__":
    unittest.main()
