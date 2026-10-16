"""Physical subtask, graduation and checkpoint checks for the curriculum."""

import json
from pathlib import Path
import tempfile
import unittest

import mujoco
import numpy as np
from stable_baselines3 import PPO as SB3PPO

from rlinit.algorithmhand.door_curriculum_env import DoorCurriculumEnv, STAGES, _ResetStateBuilder
from rlinit.algorithmhand.door_env import DoorConfig
from train_curriculum import CurriculumProgress, evaluate, train
from verify_door_opening import DoorTrial


class CurriculumTests(unittest.TestCase):
    def test_reset_states_and_stage_changes_are_reproducible(self):
        env = DoorCurriculumEnv(config=DoorConfig(initial_noise=0))
        self.addCleanup(env.close)
        for stage in STAGES:
            env.set_stage(stage, 0)
            first, info = env.reset(seed=42)
            second, _ = env.reset(seed=42)
            np.testing.assert_array_equal(first, second)
            self.assertEqual(first.shape, (70,))
            self.assertFalse(env._goal_reached(env._features()))
            self.assertLess(info["door_angle"], 0.05)
            if stage in ("press", "open"):
                self.assertTrue(info["grasped"])
            if stage == "open":
                self.assertGreaterEqual(info["handle_angle"], 1.1)
        env.set_stage("full", 1)
        _, info = env.reset(seed=42)
        self.assertEqual(info["episode_start"], "home")
        self.assertFalse(info["grasped"])
        env.set_stage("reach", 0)
        with self.assertRaises(RuntimeError):
            env.step(np.zeros(8))

    def test_all_subtasks_are_physically_solvable_through_robot_actions(self):
        env = DoorCurriculumEnv(config=DoorConfig(initial_noise=0))
        self.addCleanup(env.close)
        for stage in STAGES[:-1]:
            env.set_stage(stage, 0)
            env.reset(seed=0)
            controller = _ResetStateBuilder(env)
            mujoco.mj_copyData(controller.data, env.model, env.data)
            controller.target = env.arm_target.copy()
            start = env.data.site_xpos[env.ee].copy()
            position, rotation = controller.pose(1.3 if stage == "open" else 0)
            if stage == "reach":
                axis = env.data.site_xmat[env.handle_site].reshape(3, 3)[:, 1]
                position -= env.curriculum.pregrasp_offset * axis
            succeeded = False
            for tick in range(250):
                t = min(1.0, (tick + 1) / 100)
                t = t * t * (3 - 2 * t)
                if stage == "press":
                    p, r = controller.pose(1.3 * t)
                elif stage == "open":
                    p, r = controller.pose(1.3, 0.35 * t)
                else:
                    p, r = start + t * (position - start), rotation
                grip = 0 if stage == "reach" or (stage == "grasp" and tick < 100) else 170
                controller.tick(p, r, grip)
                action = np.r_[(controller.target - env.arm_target) / env.config.action_scale,
                               2 * grip / env.config.grip_max - 1]
                self.assertLessEqual(np.max(np.abs(action)), 1.0001)
                _, reward, terminated, truncated, info = env.step(action)
                self.assertEqual(info["failure_reason"], "", (stage, info))
                self.assertFalse(truncated)
                self.assertTrue(np.isfinite(reward))
                if terminated:
                    self.assertTrue(info["is_success"])
                    self.assertFalse(info["full_task_success"])
                    succeeded = True
                    break
            self.assertTrue(succeeded, stage)
            np.testing.assert_array_equal(env.data.qfrc_applied, 0)
            np.testing.assert_array_equal(env.data.xfrc_applied, 0)

    def test_full_task_requires_real_opening_from_home(self):
        env = DoorCurriculumEnv(stage="full", home_probability=1, config=DoorConfig(initial_noise=0))
        self.addCleanup(env.close)
        env.reset(seed=0)
        reference = DoorTrial("normal")
        result = {}

        def apply(trial, _):
            if result:
                return
            action = np.r_[(trial.last_target - env.arm_target) / env.config.action_scale,
                           2 * trial.data.ctrl[trial.gripper] / env.config.grip_max - 1]
            _, _, terminated, truncated, info = env.step(action)
            self.assertFalse(truncated)
            self.assertEqual(info["failure_reason"], "")
            if terminated:
                result.update(info)

        reference.run(apply)
        self.assertTrue(result["full_task_success"])
        self.assertGreaterEqual(result["door_angle"], 0.30)
        self.assertGreaterEqual(result["success_hold_seconds"], 0.30)

    def test_graduation_needs_consecutive_checks_and_full_home(self):
        progress = CurriculumProgress(stage="open")
        self.assertFalse(progress.advance(1, 0.8, 2, 0.25))
        self.assertFalse(progress.advance(0.7, 0.8, 2, 0.25))
        self.assertEqual(progress.passed_checks, 0)
        self.assertFalse(progress.advance(1, 0.8, 2, 0.25))
        self.assertTrue(progress.advance(1, 0.8, 2, 0.25))
        self.assertEqual((progress.stage, progress.home_probability), ("full", 0.25))
        self.assertFalse(progress.completed)
        for _ in range(3):
            self.assertFalse(progress.advance(1, 0.8, 2, 0.25))
            self.assertTrue(progress.advance(1, 0.8, 2, 0.25))
        self.assertEqual(progress.home_probability, 1)
        self.assertFalse(progress.completed)
        progress.advance(1, 0.8, 2, 0.25)
        progress.advance(1, 0.8, 2, 0.25)
        self.assertTrue(progress.completed)

    def test_native_sb3_checkpoint_records_stage_and_full_task_metrics(self):
        with tempfile.TemporaryDirectory() as directory:
            path = train(total_timesteps=32, output_dir=directory, n_steps=16, batch_size=8,
                         epochs=1, eval_freq=16, eval_episodes=1, device="cpu",
                         config=DoorConfig(max_steps=16, initial_noise=0))
            config = json.loads((Path(directory) / "config.json").read_text())
            self.assertEqual(config["environment"]["observation_dim"], 70)
            self.assertEqual(config["agent"]["net_arch"], {"pi": [256, 256, 256], "vf": [256, 256, 256]})
            restored = SB3PPO.load(path, device="cpu")
            self.assertEqual(restored.policy.net_arch, config["agent"]["net_arch"])
            for network in (restored.policy.mlp_extractor.policy_net, restored.policy.mlp_extractor.value_net):
                self.assertEqual(network[0].weight.shape, (256, 70))
                self.assertEqual(network[2].weight.shape, (256, 256))
                self.assertEqual(network[4].weight.shape, (256, 256))
            self.assertFalse(config["progress"]["completed"])
            current = evaluate(path, episodes=1, seed=123, device="cpu")
            full = evaluate(path, episodes=1, seed=123, device="cpu", full_task=True)
            self.assertEqual(current["stage"], "reach")
            self.assertEqual(full["stage"], "full")
            self.assertEqual(full["home_probability"], 1)
            self.assertEqual(full["records"][0]["episode_start"], "home")
            self.assertEqual(full["full_task_success_rate"], 0)


if __name__ == "__main__":
    unittest.main()
