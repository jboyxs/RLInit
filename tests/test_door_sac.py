"""Exercise SAC updates and portable door-task checkpoint evaluation."""

import csv
import inspect
import json
from pathlib import Path
import tempfile
import unittest

import numpy as np
from stable_baselines3 import SAC

from rlinit.algorithmhand.door_env import DoorConfig
from trainp2_sac import evaluate, load_agent, sac_settings, train


class DoorSACTests(unittest.TestCase):
    def test_defaults_and_invalid_settings(self):
        overrides, actual = sac_settings()
        self.assertEqual(overrides, {})
        for key, value in actual.items():
            self.assertEqual(value, inspect.signature(SAC).parameters[key].default)
        self.assertEqual(sac_settings(ent_coef="0.2")[0]["ent_coef"], 0.2)
        self.assertEqual(sac_settings(ent_coef="auto_0.1")[0]["ent_coef"], "auto_0.1")
        for invalid in ({"buffer_size": 0}, {"batch_size": 0}, {"learning_starts": -1},
                        {"train_freq": 0}, {"gradient_steps": 0}, {"gradient_steps": -2},
                        {"gamma": 1.1}, {"tau": 0}, {"learning_rate": float("nan")},
                        {"ent_coef": "auto_0"}, {"ent_coef": "auto_nan"}, {"ent_coef": -1}):
            with self.subTest(invalid=invalid), self.assertRaises(ValueError):
                sac_settings(**invalid)

    def test_updates_best_checkpoint_and_standalone_evaluation(self):
        with tempfile.TemporaryDirectory() as directory:
            output = Path(directory)
            checkpoint = train(
                total_timesteps=25, output_dir=output, seed=0, device="cpu",
                env_config=DoorConfig(max_steps=4), learning_starts=4,
                buffer_size=64, batch_size=8, train_freq=4, gradient_steps=-1,
                gamma=0.999, ent_coef="auto_0.1", eval_freq=12, eval_episodes=1,
            )
            config = json.loads((output / "config.json").read_text())
            self.assertEqual(config["actual_timesteps"], 28)
            self.assertGreater(config["gradient_updates"], 0)
            self.assertEqual(config["final_eval_seed"], 20_000)
            self.assertEqual(config["environment"]["config"]["gamma"], 0.999)
            self.assertTrue((output / "train.monitor.csv").is_file())
            self.assertTrue((output / "eval" / "evaluations.npz").is_file())
            with (output / "logs" / "progress.csv").open(newline="") as stream:
                rows = list(csv.DictReader(stream))
            for key in ("train/actor_loss", "train/critic_loss", "train/ent_coef"):
                values = [float(row[key]) for row in rows if row.get(key)]
                self.assertTrue(values, key)
                self.assertTrue(np.isfinite(values).all(), key)
            metrics = json.loads((output / "metrics.json").read_text())
            self.assertEqual(metrics["first_seed"], 20_000)
            self.assertLessEqual(metrics["records"][0]["length"], 4)
            for path in (checkpoint, output / "best" / "best_model.zip"):
                agent, env = load_agent(path, device="cpu")
                try:
                    self.assertIsInstance(agent, SAC)
                    self.assertGreater(agent._n_updates, 0)
                    self.assertEqual(env.config.max_steps, 4)
                    self.assertEqual(env.config.gamma, agent.gamma)
                    obs, _ = env.reset(seed=42)
                    action, _ = agent.predict(obs, deterministic=True)
                    self.assertTrue(env.action_space.contains(action))
                finally:
                    env.close()
            # Model metadata must suffice even when config.json is missing.
            (output / "config.json").unlink()
            metrics = evaluate(checkpoint, episodes=2, seed=123, device="cpu")
            repeated = evaluate(checkpoint, episodes=2, seed=123, device="cpu")
            self.assertEqual(metrics, repeated)
            self.assertEqual([r["seed"] for r in metrics["records"]], [123, 124])
            # Reject task metadata with an inconsistent shaping discount.
            agent.door_environment["config"]["gamma"] = 0.9
            bad_checkpoint = output / "bad_gamma.zip"
            agent.save(bad_checkpoint)
            with self.assertRaisesRegex(ValueError, "gamma"):
                load_agent(bad_checkpoint, device="cpu")


if __name__ == "__main__":
    unittest.main()
