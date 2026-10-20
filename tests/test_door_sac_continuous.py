"""Regress native continuous joints, unchanged 8-D actions and cold SAC pipeline."""

from dataclasses import replace
import json
from pathlib import Path
import tempfile
import unittest
from unittest.mock import patch

import mujoco
import numpy as np
from stable_baselines3 import SAC

from rlinit.algorithmhand.door_sac_learn_env import LearnRewardConfig, SACLearnDoorEnv
from rlinit.algorithmhand.door_sac_continuous_env import ContinuousJointSACDoorEnv, default_door_config
from train_sac_continuous import SACContinuousSettings, load_agent, train_two_phases


class ContinuousSACDoorTests(unittest.TestCase):
    def test_only_extra_target_limits_change_not_physical_model(self):
        old, new = SACLearnDoorEnv(), ContinuousJointSACDoorEnv()
        self.addCleanup(old.close)
        self.addCleanup(new.close)
        for field in ("jnt_limited", "jnt_range", "actuator_ctrllimited", "actuator_ctrlrange",
                      "actuator_forcerange", "actuator_gainprm", "actuator_biasprm"):
            np.testing.assert_array_equal(getattr(old.model, field), getattr(new.model, field))
        for i, joint in enumerate(new.joints):
            if new.model.jnt_limited[joint]:
                self.assertEqual(new.lower[i], old.lower[i])
                self.assertEqual(new.upper[i], old.upper[i])
            else:
                self.assertTrue(np.isneginf(new.lower[i]))
                self.assertTrue(np.isposinf(new.upper[i]))
        self.assertEqual(new.action_space.shape, (8,))
        self.assertEqual(new.config.action_scale, old.config.action_scale)
        json.dumps(new.specification(), allow_nan=False)

    def test_physics_and_action_mapping_equal_away_from_old_cut(self):
        cfg = LearnRewardConfig.guided()
        old, new = SACLearnDoorEnv(reward_config=cfg), ContinuousJointSACDoorEnv(reward_config=cfg)
        self.addCleanup(old.close)
        self.addCleanup(new.close)
        old.reset(seed=31)
        new.reset(seed=31)
        for action in np.random.default_rng(31).uniform(-1, 1, (25, 8)):
            original, changed = old.step(action), new.step(action)
            np.testing.assert_array_equal(old.data.qpos, new.data.qpos)
            np.testing.assert_array_equal(old.data.ctrl, new.data.ctrl)
            np.testing.assert_array_equal(original[0], changed[0])
            self.assertEqual(original[1:4], changed[1:4])

    def test_joint7_action_can_pass_old_software_cut(self):
        old, new = SACLearnDoorEnv(), ContinuousJointSACDoorEnv()
        self.addCleanup(old.close)
        self.addCleanup(new.close)
        cut = old.lower[6]
        for env in (old, new):
            env.reset(seed=31)
            env.arm_target[6] = cut + .002
            env.data.qpos[env.qadr[6]] = cut + .002
            mujoco.mj_forward(env.model, env.data)
            env.step(np.r_[np.zeros(6), -1., -1.])
        self.assertAlmostEqual(old.arm_target[6], cut)
        self.assertAlmostEqual(new.arm_target[6], cut - .008)
        self.assertAlmostEqual(new.data.ctrl[new.actuators[6]] - old.data.ctrl[old.actuators[6]], -.008)

    def test_cold_two_stage_updates_noise_save_load_and_rejects_different_limits(self):
        settings = replace(SACContinuousSettings(), buffer_size=64, batch_size=4, scratch_learning_starts=2)
        with tempfile.TemporaryDirectory() as folder:
            checkpoint = train_two_phases(output_dir=folder, total_timesteps=12, press_progress_after=8,
                opening_joint_exploration_std=.1, settings=settings, n_envs=1, device="cpu",
                env_config=replace(default_door_config(), max_steps=4), eval_freq=4, eval_episodes=1)
            agent, env = load_agent(checkpoint)
            self.addCleanup(env.close)
            self.assertEqual(agent.num_timesteps, 12)
            self.assertTrue(np.isneginf(env.lower[6]))
            self.assertTrue(env.reward_config.press_progress_only)
            noise = agent.action_noise.noises[0] if hasattr(agent.action_noise, "noises") else agent.action_noise
            np.testing.assert_allclose(noise._sigma[:7], np.sqrt(2) * .1)
            metadata = json.loads((Path(folder) / "phase1/config.json").read_text())
            self.assertEqual(metadata["mode"], "scratch")
            self.assertIsNone(metadata["environment"]["joint_target_lower"][6])
            agent.door_environment["joint_target_lower"][6] = -np.pi / 2
            modified = Path(folder) / "wrong_limits.zip"
            agent.save(modified)
            with self.assertRaisesRegex(ValueError, "Joint target limits"):
                load_agent(modified)
            agent.door_task_format = "rlinit-sac-learn-v1"
            agent.save(modified)
            with self.assertRaisesRegex(ValueError, "format"):
                load_agent(modified)

    def test_pipeline_keeps_a_successful_first_phase_if_continuation_regresses(self):
        # Synthetic files isolate checkpoint selection; real SAC is exercised above.
        def stage(*, output_dir, **kwargs):
            path = Path(output_dir)
            path.mkdir()
            (path / "best").mkdir()
            first = path.name == "phase1"
            content = b"first-policy" if first else b"later-policy"
            (path / "final_model.zip").write_bytes(content)
            (path / "best/best_model.zip").write_bytes(content)
            (path / "best/selection.json").write_text(json.dumps(dict(success_rate=1. if first else .2, timesteps=8 if first else 12)))
            (path / "evaluation.json").write_text(json.dumps(dict(success_rate=1. if first else .2)))
            (path / "config.json").write_text(json.dumps(dict(actual_timesteps=8 if first else 12)))
            return path / "final_model.zip"

        with tempfile.TemporaryDirectory() as folder, patch("train_sac_continuous.train", side_effect=stage), \
                patch("builtins.print"):
            selected = train_two_phases(output_dir=folder, total_timesteps=12, press_progress_after=8)
            self.assertEqual(selected.read_bytes(), b"first-policy")
            manifest = json.loads((Path(folder) / "pipeline.json").read_text())
            self.assertEqual(manifest["selected_model_phase"], "phase1")


if __name__ == "__main__":
    unittest.main()
