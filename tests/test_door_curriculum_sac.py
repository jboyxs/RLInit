"""SAC defaults, curriculum rewards, stage transitions and checkpoint checks."""

import csv
import inspect
import json
from pathlib import Path
import tempfile
import unittest
from unittest.mock import patch

import numpy as np
import torch
from stable_baselines3 import SAC

from rlinit.algorithmhand.door_curriculum_env import CurriculumConfig, DoorCurriculumEnv
from rlinit.algorithmhand.door_env import DoorConfig
from train_curriculum_hand import evaluate_policy as hand_evaluator
from train_curriculum_sac import build_parser, evaluate, evaluate_agent, load_agent, train
from trainp2_sac import build_parser as baseline_parser


class CurriculumSACTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.previous_threads = torch.get_num_threads()
        torch.set_num_threads(1)

    @classmethod
    def tearDownClass(cls):
        torch.set_num_threads(cls.previous_threads)

    def test_sac_baseline_defaults_checkpoint_and_both_evaluation_modes(self):
        baseline = vars(baseline_parser().parse_args([]))
        current = vars(build_parser().parse_args([]))
        for key, value in baseline.items():
            if key != "output_dir":
                self.assertEqual(current[key], value, key)
        with tempfile.TemporaryDirectory() as directory:
            output = Path(directory)
            checkpoint = train(
                total_timesteps=8, output_dir=output, env_config=DoorConfig(max_steps=4, initial_noise=0),
                buffer_size=64, eval_freq=4, eval_episodes=1, device="cpu",
            )
            config = json.loads((output / "config.json").read_text())
            self.assertEqual(config["gradient_updates"], 0)  # Preserve the default 100-step warmup.
            for key in ("learning_rate", "learning_starts", "batch_size", "train_freq",
                        "gradient_steps", "gamma", "tau", "ent_coef"):
                self.assertEqual(config["agent"][key], inspect.signature(SAC).parameters[key].default)
            self.assertEqual(config["agent"]["net_arch"], [256, 256])
            self.assertEqual(config["agent"]["activation"], "ReLU")
            self.assertEqual(config["agent"]["optimizer"], "Adam")
            self.assertEqual(config["agent"]["n_critics"], 2)
            self.assertEqual(config["environment"]["observation_dim"], 70)
            self.assertEqual(config["environment"]["action_dim"], 8)
            self.assertEqual(config["environment"]["curriculum"], json.loads(json.dumps(CurriculumConfig().__dict__)))
            (output / "config.json").unlink()
            agent, env = load_agent(checkpoint, device="cpu")
            try:
                metrics = evaluate(checkpoint, episodes=1, seed=123, device="cpu")
                expected = hand_evaluator(lambda obs: agent.predict(obs, deterministic=True)[0],
                                          env.specification(), episodes=1, seed=123)
                self.assertEqual(metrics, expected)
                self.assertEqual(metrics["stage"], "reach")
                self.assertEqual(agent.gamma, env.base_config.gamma)
            finally:
                env.close()
            full = evaluate(checkpoint, episodes=1, seed=123, device="cpu", full_task=True)
            self.assertEqual(full["stage"], "full")
            self.assertEqual(full["home_probability"], 1)
            self.assertTrue(all(r["episode_start"] == "home" for r in full["records"]))
            final = json.loads((output / "metrics.json").read_text())
            self.assertFalse(final["curriculum_completed"])
            self.assertEqual(final["full_task_success_rate"], 0)

    def test_switches_preserve_optimizers_and_replay_without_cross_stage_transitions(self):
        real_train, real_evaluate = SAC.train, evaluate_agent
        learners, update_stages = [], []

        def record_train(agent, *args, **kwargs):
            learners.append(agent)
            update_stages.append(agent.env.envs[0].unwrapped.stage)
            return real_train(agent, *args, **kwargs)

        def pass_stage(agent, specification, **kwargs):
            metrics = real_evaluate(agent, specification, **kwargs)
            # Synthetic graduation scores test control flow; physical success
            # is still measured honestly in the final home-only evaluation.
            if kwargs.get("seed") == 10_000:
                metrics["success_rate"] = 1.0
            return metrics

        with tempfile.TemporaryDirectory() as directory:
            output = Path(directory)
            with patch.object(SAC, "train", record_train), \
                    patch("train_curriculum_sac.evaluate_agent", side_effect=pass_stage):
                checkpoint = train(
                    total_timesteps=12, output_dir=output, start_stage="press", patience=1, home_step=1,
                    env_config=DoorConfig(max_steps=4, initial_noise=0), gamma=0.995,
                    buffer_size=64, batch_size=4, learning_starts=0, train_freq=2, gradient_steps=-1,
                    eval_freq=4, eval_episodes=1, device="cpu",
                )
            self.assertEqual(len({id(a) for a in learners}), 1)
            agent = learners[0]
            self.assertEqual(agent._n_updates, 12)
            self.assertEqual(update_stages, ["press", "press", "open", "open", "full", "full"])
            for optimizer in (agent.actor.optimizer, agent.critic.optimizer, agent.ent_coef_optimizer):
                self.assertTrue(optimizer.state)
                self.assertTrue(all(s["step"].item() == 12 for s in optimizer.state.values()))
            replay = agent.replay_buffer
            self.assertEqual(replay.size(), 12)
            for start, stage_index in ((0, 2), (4, 3), (8, 4)):
                expected = np.zeros((4, 5))
                expected[:, stage_index] = 1
                np.testing.assert_array_equal(replay.observations[start:start + 4, 0, 64:69], expected)
                np.testing.assert_array_equal(replay.next_observations[start:start + 4, 0, 64:69], expected)
                self.assertEqual(replay.timeouts[start + 3, 0], 1)
            np.testing.assert_array_equal(replay.observations[:8, 0, 69], 0)
            np.testing.assert_array_equal(replay.observations[8:12, 0, 69], 1)
            for name in ("actor_loss", "critic_loss", "ent_coef"):
                with (output / "logs" / "progress.csv").open(newline="") as file:
                    values = [float(r[f"train/{name}"]) for r in csv.DictReader(file) if r.get(f"train/{name}")]
                self.assertTrue(values)
                self.assertTrue(np.isfinite(values).all())
            config = json.loads((output / "config.json").read_text())
            self.assertTrue(config["progress"]["completed"])
            self.assertEqual(config["environment"]["stage"], "full")
            self.assertEqual(config["agent"]["gamma"], config["environment"]["config"]["gamma"])
            for path in (checkpoint, output / "best" / "press_home_0.00.zip",
                         output / "best" / "open_home_0.00.zip", output / "best" / "full_home_1.00.zip"):
                restored, env = load_agent(path, device="cpu")
                try:
                    self.assertEqual(restored.gamma, 0.995)
                    self.assertEqual(restored.curriculum_environment["stage"], env.stage)
                finally:
                    env.close()
            final = json.loads((output / "metrics.json").read_text())
            self.assertTrue(final["curriculum_completed"])
            self.assertTrue(all(r["episode_start"] == "home" for r in final["records"]))
            self.assertEqual(final["full_task_success_rate"], 0)

    def test_stage_change_mid_episode_resets_last_observation(self):
        real_train, real_evaluate = SAC.train, evaluate_agent
        learners = []

        def record_train(agent, *args, **kwargs):
            learners.append(agent)
            return real_train(agent, *args, **kwargs)

        def pass_stage(agent, specification, **kwargs):
            metrics = real_evaluate(agent, specification, **kwargs)
            if kwargs.get("seed") == 10_000:
                metrics["success_rate"] = 1.0
            return metrics

        with tempfile.TemporaryDirectory() as directory, \
                patch.object(SAC, "train", record_train), \
                patch("train_curriculum_sac.evaluate_agent", side_effect=pass_stage):
            train(total_timesteps=4, output_dir=directory, start_stage="reach", patience=1, home_step=1,
                  env_config=DoorConfig(max_steps=8, initial_noise=0), buffer_size=32,
                  batch_size=2, learning_starts=0, eval_freq=2, eval_episodes=1, device="cpu")
            replay = learners[0].replay_buffer
            np.testing.assert_array_equal(replay.observations[:2, 0, 69], 0)
            np.testing.assert_array_equal(replay.next_observations[:2, 0, 69], 0)
            np.testing.assert_array_equal(replay.observations[2:4, 0, 69], 1)
            np.testing.assert_array_equal(replay.next_observations[2:4, 0, 69], 1)
            self.assertTrue(np.all(replay.dones[:4] == 0))  # Both lessons ended at collection boundaries.

    def test_original_sac_checkpoint_and_bad_gamma_are_rejected(self):
        env = DoorCurriculumEnv(config=DoorConfig(gamma=0.99))
        self.addCleanup(env.close)
        agent = SAC("MlpPolicy", env, buffer_size=8, device="cpu")
        with tempfile.TemporaryDirectory() as directory:
            path = Path(directory) / "bad.zip"
            agent.door_task_format = "rlinit-door-sac-v1"
            agent.save(path)
            with self.assertRaisesRegex(ValueError, "curriculum SAC"):
                load_agent(path, device="cpu")
            from train_curriculum_sac import FORMAT
            agent.curriculum_format = FORMAT
            agent.curriculum_environment = env.specification()
            agent.curriculum_environment["config"]["gamma"] = 0.9
            agent.save(path)
            with self.assertRaisesRegex(ValueError, "gamma"):
                load_agent(path, device="cpu")


if __name__ == "__main__":
    unittest.main()
