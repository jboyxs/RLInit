"""Guard physical/action fidelity, reward loopholes and real SAC continuation."""

from dataclasses import replace
import json
from pathlib import Path
import tempfile
import unittest
from unittest.mock import patch

import numpy as np
from stable_baselines3.common.buffers import NStepReplayBuffer

from rlinit.algorithmhand.door_sac_open_env import OpenRewardConfig, SACOpenDoorEnv, default_door_config
from rlinit.algorithmhand.door_sac_learn_env import LearnRewardConfig, SACLearnDoorEnv
from train_sac_learn import SACLearnSettings, load_agent, train, train_two_phases


class SACLearnTests(unittest.TestCase):
    def test_same_physics_and_original_eight_actions(self):
        original, new = SACOpenDoorEnv(), SACLearnDoorEnv()
        self.addCleanup(original.close)
        self.addCleanup(new.close)
        original.reset(seed=19)
        obs, _ = new.reset(seed=19)
        self.assertEqual(obs.shape, (80,))
        self.assertEqual(new.action_space.shape, (8,))
        for action in np.random.default_rng(0).uniform(-1, 1, (30, 8)):
            original.step(action)
            new.step(action)
            np.testing.assert_array_equal(original.data.qpos, new.data.qpos)
            np.testing.assert_array_equal(original.data.ctrl, new.data.ctrl)
        spec = new.specification()
        self.assertIn("objects\\door_lock_fixture.xml", spec["model_source_sha256"])
        self.assertFalse(spec["scripted_actions"])
        self.assertIsNone(spec["reset_curriculum"])

    def test_stationary_noncontact_cannot_collect_positive_reward(self):
        env = SACLearnDoorEnv()
        self.addCleanup(env.close)
        env.reset(seed=67)
        features = env._features()
        self.assertFalse(features[4])
        # Arbitrary stationary poses, including nearly perfect alignment/enclosure,
        # cannot earn positive rewards without improving state or reaching a milestone.
        for alignment in (0., .9, 1.):
            candidate = (*features[:2], alignment, {"left": 0., "right": 0.}, False, 0., 0.)
            for closure in (-1., 1.):
                action = np.r_[np.zeros(7), closure]
                parts = env._reward_components(features=candidate, terminated=False,
                    potential_before=env._potential(candidate), newly_grasped=False,
                    newly_pressed=False, success=False, reason="", peak_bad_force=0., action=action)
                self.assertLess(sum(parts.values()), 0.)

    def test_success_needs_stable_downward_press_before_door_hold(self):
        env = SACLearnDoorEnv()
        self.addCleanup(env.close)
        env.reset(seed=67)
        # Synthetic coordinates test the success state machine, not physical ability.
        env.data.qpos[env.hinge] = .31
        for _ in range(25):
            self.assertFalse(env._is_success(grasp=True, reason=""))
        env.data.qpos[env.latch] = np.pi / 4 + .001
        for _ in range(15):
            success = env._is_success(grasp=True, reason="")
        self.assertTrue(success)
        self.assertTrue(env.pressed_with_stable_grasp)
        self.assertFalse(env._is_success(grasp=False, reason=""))
        obs, _ = env.reset(seed=67)
        np.testing.assert_array_equal(obs[-2:], [0., 0.])

    def test_farther_noncontact_pose_keeps_linear_cost_gradient(self):
        env = SACLearnDoorEnv()
        self.addCleanup(env.close)
        env.reset(seed=67)
        features = env._features()
        costs = []
        for distance in (.5, 1.):
            geometry = (np.zeros((2, 3)), np.array([distance, 0., 0.]),
                        np.array([distance, distance]), .08)
            with patch.object(env, "_grasp_geometry", return_value=geometry):
                parts = env._reward_components(features=features, terminated=False,
                    potential_before=env._potential(features), newly_grasped=False,
                    newly_pressed=False, success=False, reason="", peak_bad_force=0.,
                    action=np.r_[np.zeros(7), -1.])
            costs.append(parts["distance"] + parts["centering"])
        self.assertLess(costs[1] - costs[0], -.5)

    def test_guided_reward_closes_free_pose_loophole_and_rewards_real_press(self):
        cfg = LearnRewardConfig.guided()
        env = SACLearnDoorEnv(reward_config=cfg)
        self.addCleanup(env.close)
        env.reset(seed=67)
        env.grasp_hold = .1
        env.data.qpos[env.latch] = np.pi / 4
        geometry = (np.zeros((2, 3)), np.zeros(3), np.array([.005, .005]), .04)
        with patch.object(env, "_grasp_geometry", return_value=geometry):
            for grasp in (False, True):
                for alignment in (.3, 1.):
                    features = (None, None, alignment, {"left": 20., "right": 20.}
                                if grasp else {"left": 0., "right": 0.}, grasp, 0., 0.)
                    parts = env._reward_components(features=features, terminated=False,
                        potential_before=0., newly_grasped=False, newly_pressed=False,
                        success=False, reason="", peak_bad_force=0., action=np.ones(8))
                    if grasp:
                        self.assertAlmostEqual(parts["pressing"], 3.)
                    else:
                        self.assertLess(sum(parts.values()), 0.)
        base = OpenRewardConfig()
        upper = (base.orientation_weight + base.closure_weight + base.contact_weight
                 + base.grasp_weight + cfg.guided_press_weight + base.opening_weight)
        # Holding just short of termination must not beat the terminal reward.
        self.assertGreater(cfg.guided_success_bonus, upper / (1 - env.config.gamma))

    def test_real_updates_horizon_switch_and_own_replay_resume(self):
        settings = replace(SACLearnSettings(), buffer_size=64, batch_size=4,
                           scratch_learning_starts=2, multistep_after=8)
        config = replace(default_door_config(), max_steps=4)
        with tempfile.TemporaryDirectory() as folder:
            root = Path(folder)
            checkpoint = train(output_dir=root / "first", total_timesteps=12, n_envs=1,
                device="cpu", settings=settings, env_config=config, eval_freq=4, eval_episodes=1)
            agent, env = load_agent(checkpoint)
            env.close()
            self.assertEqual(agent.n_steps, 3)
            self.assertGreater(agent._n_updates, 0)
            self.assertTrue((root / "first/initial_random_model.zip").is_file())
            first_updates = agent._n_updates
            resumed = train(output_dir=root / "second", total_timesteps=8, n_envs=1,
                device="cpu", resume_checkpoint=checkpoint,
                replay_buffer_path=root / "first/replay_buffer.pkl", eval_freq=4, eval_episodes=1)
            agent, env = load_agent(resumed)
            env.close()
            self.assertEqual(agent.num_timesteps, 20)
            self.assertGreater(agent._n_updates, first_updates)
            self.assertEqual(agent.n_steps, 3)
            agent.load_replay_buffer(root / "second/replay_buffer.pkl")
            self.assertIs(type(agent.replay_buffer), NStepReplayBuffer)
            self.assertGreater(agent.replay_buffer.size(), 12)
            run = json.loads((root / "first/config.json").read_text())
            self.assertEqual(run["mode"], "scratch")

    def test_guided_three_step_sac_really_updates_and_loads(self):
        recipe = replace(SACLearnSettings(), n_steps=3, buffer_size=64, batch_size=4,
                         scratch_learning_starts=2)
        with tempfile.TemporaryDirectory() as folder:
            checkpoint = train(output_dir=folder, total_timesteps=8, n_envs=1,
                settings=recipe, reward_config=LearnRewardConfig.guided(), device="cpu",
                env_config=replace(default_door_config(), max_steps=4), eval_freq=4, eval_episodes=1)
            agent, env = load_agent(checkpoint)
            self.addCleanup(env.close)
            self.assertTrue(env.reward_config.dense_guidance)
            self.assertEqual(agent.n_steps, 3)
            self.assertGreater(agent._n_updates, 0)
            self.assertIn("guidance_base_reward", agent.door_environment)
            with patch("rlinit.algorithmhand.door_sac_learn_env.OpenRewardConfig",
                       return_value=replace(OpenRewardConfig(), orientation_weight=.7)):
                with self.assertRaisesRegex(ValueError, "guidance reward changed"):
                    load_agent(checkpoint)
            upgraded = train(output_dir=Path(folder) / "progress", total_timesteps=4, n_envs=1,
                device="cpu", resume_checkpoint=checkpoint,
                replay_buffer_path=Path(folder) / "replay_buffer.pkl", upgrade_press_progress=True,
                eval_freq=4, eval_episodes=1)
            resumed, progress_env = load_agent(upgraded)
            self.addCleanup(progress_env.close)
            self.assertTrue(progress_env.reward_config.press_progress_only)
            self.assertGreater(resumed._n_updates, agent._n_updates)
            metadata = json.loads((Path(folder) / "progress/config.json").read_text())
            self.assertEqual(metadata["replay_conversion"]["relabelled_transitions"], 8)

    def test_single_entry_runs_both_phases_from_random_weights(self):
        recipe = replace(SACLearnSettings(), n_steps=3, buffer_size=64, batch_size=4,
                         scratch_learning_starts=2)
        with tempfile.TemporaryDirectory() as folder:
            checkpoint = train_two_phases(output_dir=folder, total_timesteps=12,
                press_progress_after=8, settings=recipe, n_envs=1, device="cpu",
                env_config=replace(default_door_config(), max_steps=4), eval_freq=4, eval_episodes=1)
            root = Path(folder)
            self.assertTrue((root / "phase1/initial_random_model.zip").is_file())
            manifest = json.loads((root / "pipeline.json").read_text())
            self.assertFalse(manifest["external_pretraining"])
            self.assertEqual(manifest["actual_timesteps"], 12)
            second = json.loads((root / "phase2/config.json").read_text())
            self.assertEqual(second["source_checkpoint_timesteps"], 8)
            self.assertEqual(second["replay_conversion"]["relabelled_transitions"], 8)
            agent, env = load_agent(checkpoint)
            self.addCleanup(env.close)
            self.assertEqual(agent.num_timesteps, 12)
            self.assertTrue(env.reward_config.press_progress_only)


if __name__ == "__main__":
    unittest.main()
