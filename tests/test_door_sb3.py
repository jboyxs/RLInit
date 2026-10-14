"""Check the native SB3 baseline and task restoration in .zip checkpoints."""

import inspect
import json
from pathlib import Path
import tempfile
import unittest

from stable_baselines3 import PPO

from rlinit.algorithmhand.door_env import DoorConfig
from trainp2_sb3 import evaluate, load_agent, ppo_settings, train


class DoorSB3Tests(unittest.TestCase):
    def test_no_algorithm_overrides_by_default(self):
        overrides, actual = ppo_settings()
        self.assertEqual(overrides, {})
        signature = inspect.signature(PPO)
        for key, value in actual.items():
            self.assertEqual(value, signature.parameters[key].default)
        _, actual = ppo_settings(gamma=0.999)
        self.assertEqual(actual["gamma"], 0.999)
        with self.assertRaises(ValueError):
            ppo_settings(n_steps=1)

    def test_training_best_checkpoint_and_standalone_evaluation(self):
        with tempfile.TemporaryDirectory() as directory:
            output = Path(directory)
            checkpoint = train(
                total_timesteps=17, output_dir=output, seed=0, device="cpu",
                env_config=DoorConfig(max_steps=8), n_steps=16, batch_size=8,
                epochs=1, eval_freq=16, eval_episodes=1,
            )
            config = json.loads((output / "config.json").read_text())
            self.assertEqual(config["actual_timesteps"], 32)
            self.assertEqual(config["agent"]["gamma"], 0.99)
            self.assertEqual(config["environment"]["config"]["gamma"], 0.99)
            self.assertEqual(config["agent"]["net_arch"], {"pi": [64, 64], "vf": [64, 64]})
            self.assertEqual(config["agent"]["activation"], "Tanh")
            self.assertTrue(config["agent"]["ortho_init"])
            self.assertTrue((output / "train.monitor.csv").is_file())
            self.assertTrue((output / "logs" / "progress.csv").is_file())
            self.assertTrue((output / "metrics.json").is_file())
            for path in (checkpoint, output / "best" / "best_model.zip"):
                agent, env = load_agent(path, device="cpu")
                try:
                    self.assertIsInstance(agent, PPO)
                    self.assertEqual(env.config.max_steps, 8)
                    self.assertEqual(env.config.gamma, agent.gamma)
                    self.assertEqual(env.observation_space.shape, (64,))
                    self.assertEqual(env.action_space.shape, (8,))
                finally:
                    env.close()
            # Evaluation must use the metadata inside the .zip, not config.json.
            (output / "config.json").unlink()
            metrics = evaluate(checkpoint, episodes=1, seed=123, device="cpu")
            self.assertEqual(metrics["first_seed"], 123)
            self.assertLessEqual(metrics["records"][0]["length"], 8)


if __name__ == "__main__":
    unittest.main()
