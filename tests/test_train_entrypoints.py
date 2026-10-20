"""Real saves, reloads and subprocess collection through the new entry points."""
from contextlib import redirect_stdout
from dataclasses import replace
import io
import json
from pathlib import Path
import tempfile
import unittest

import torch

from train import train_hand, train_sb3, train_sac_staged as staged


class TrainingEntryTests(unittest.TestCase):
    def test_handwritten_ppo_entry_updates_saves_and_evaluates(self):
        previous = torch.get_num_threads()
        torch.set_num_threads(1)
        self.addCleanup(torch.set_num_threads, previous)
        with tempfile.TemporaryDirectory() as folder, redirect_stdout(io.StringIO()):
            checkpoint = train_hand.train(total_timesteps=8, output_dir=folder, n_steps=8,
                                          epochs=1, hidden_dim=16, batch_size=4, device="cpu")
            self.assertTrue(checkpoint.is_file())
            config = json.loads((Path(folder) / "config.json").read_text())
            self.assertEqual(config["agent"]["batch_size"], 4)
            self.assertEqual(config["agent"]["state_dim"], 20)
            self.assertEqual(config["agent"]["action_dim"], 7)
            self.assertEqual(json.loads((Path(folder) / "metrics.json").read_text())["episodes"], 10)

    def test_sb3_ppo_entry_updates_saves_and_evaluates(self):
        previous = torch.get_num_threads()
        torch.set_num_threads(1)
        self.addCleanup(torch.set_num_threads, previous)
        with tempfile.TemporaryDirectory() as folder, redirect_stdout(io.StringIO()):
            checkpoint = train_sb3.train_ppo(total_timesteps=8, output_dir=folder, n_steps=8,
                                            batch_size=4, device="cpu")
            self.assertTrue(checkpoint.is_file())
            self.assertEqual(json.loads((Path(folder) / "config.json").read_text())["actual_timesteps"], 8)
            self.assertEqual(json.loads((Path(folder) / "metrics.json").read_text())["episodes"], 10)

    def test_standalone_sac_collects_with_four_spawned_environments(self):
        recipe = replace(staged.SACLearnSettings(), n_steps=3, buffer_size=64,
                         batch_size=4, scratch_learning_starts=4)
        with tempfile.TemporaryDirectory() as folder, redirect_stdout(io.StringIO()):
            checkpoint = staged.train(output_dir=folder, total_timesteps=16, n_envs=4,
                settings=recipe, reward_config=staged.LearnRewardConfig.guided(), device="cpu",
                env_config=replace(staged.default_door_config(), max_steps=4), eval_freq=8, eval_episodes=1)
            agent, env = staged.load_agent(checkpoint)
            self.addCleanup(env.close)
            self.assertEqual(agent.num_timesteps, 16)
            self.assertEqual(agent.n_envs, 4)
            self.assertGreater(agent._n_updates, 0)
            spec = json.loads((Path(folder) / "config.json").read_text())
            self.assertEqual(spec["environment"]["action_dim"], 8)
            self.assertEqual(set(spec["recipe_sha256"]), {
                str(Path("train/train_sac_staged.py")),
                str(Path("src/rlinit/algorithmhand/door_sac_learn_env.py")),
            })


if __name__ == "__main__":
    unittest.main()
