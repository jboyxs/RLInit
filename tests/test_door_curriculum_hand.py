"""Handwritten PPO curriculum checkpoint and stage-transition integration checks."""

import csv
import json
from pathlib import Path
import tempfile
import unittest
from unittest.mock import patch

import numpy as np
import torch

from rlinit.algorithmhand.PPOhand import PPO
from rlinit.algorithmhand.door_env import DoorConfig
from train_curriculum import evaluate_agent as evaluate_sb3_agent
from train_curriculum_hand import FORMAT, evaluate, evaluate_agent, load_agent, train


class HandCurriculumTests(unittest.TestCase):
    def test_training_checkpoint_and_both_evaluation_modes(self):
        with tempfile.TemporaryDirectory() as directory:
            path = train(total_timesteps=32, output_dir=directory, n_steps=16, batch_size=8,
                         epochs=1, eval_freq=16, eval_episodes=1, device="cpu",
                         config=DoorConfig(gamma=0.99, max_steps=16, initial_noise=0))
            saved = torch.load(path, map_location="cpu", weights_only=True)
            config = json.loads((Path(directory) / "config.json").read_text())
            self.assertEqual(saved["format"], FORMAT)
            self.assertEqual(saved["actor"]["fc1.weight"].shape, (128, 70))
            self.assertEqual(saved["actor"]["fc2.weight"].shape, (128, 128))
            self.assertEqual(saved["actor"]["fc3.weight"].shape, (8, 128))
            self.assertEqual(config["actual_timesteps"], 32)
            self.assertEqual(config["agent"]["gamma"], config["environment"]["config"]["gamma"])
            self.assertFalse(saved["progress"]["completed"])
            for optimizer in ("actor_optimizer", "critic_optimizer"):
                self.assertTrue(saved[optimizer]["state"])
                self.assertTrue(all(s["step"].item() == 4 for s in saved[optimizer]["state"].values()))
            agent, env = load_agent(path, device="cpu")
            try:
                for key, value in agent.actor.state_dict().items():
                    torch.testing.assert_close(value, saved["actor"][key], rtol=0, atol=0)
                for key, value in agent.critic.state_dict().items():
                    torch.testing.assert_close(value, saved["critic"][key], rtol=0, atol=0)

                class SB3Interface:
                    def predict(self, obs, deterministic):
                        return agent.take_action(obs, deterministic=deterministic), None

                current = evaluate(path, episodes=1, seed=123, device="cpu")
                via_sb3 = evaluate_sb3_agent(SB3Interface(), env.specification(), episodes=1, seed=123)
                self.assertEqual(current, via_sb3)
                self.assertEqual(current["stage"], "reach")
            finally:
                env.close()
            full = evaluate(path, episodes=1, seed=123, device="cpu", full_task=True)
            self.assertEqual(full["stage"], "full")
            self.assertEqual(full["home_probability"], 1)
            self.assertEqual(full["records"][0]["episode_start"], "home")
            self.assertEqual(full["full_task_success_rate"], 0)

    def test_stage_switches_keep_optimizers_and_separate_rollouts(self):
        # Controlled evaluation scores exercise graduation without asserting
        # that a 24-step PPO run has learned these physical tasks.
        real_update = PPO.update
        real_evaluate = evaluate_agent
        rollouts = []
        agent_ids = []

        def record_update(agent, rollout):
            rollouts.append(np.asarray(rollout["states"]).copy())
            agent_ids.append(id(agent))
            return real_update(agent, rollout)

        def pass_stage(agent, specification, **kwargs):
            result = real_evaluate(agent, specification, **kwargs)
            if kwargs.get("stage") != "full":
                result["success_rate"] = 1.0
            return result

        with tempfile.TemporaryDirectory() as directory:
            with patch.object(PPO, "update", record_update), \
                    patch("train_curriculum_hand.evaluate_agent", side_effect=pass_stage):
                path = train(total_timesteps=24, output_dir=directory, n_steps=8, batch_size=4,
                             epochs=1, eval_freq=8, eval_episodes=1, device="cpu",
                             start_stage="press", patience=1, home_step=1,
                             config=DoorConfig(gamma=0.99, max_steps=16, initial_noise=0))
            self.assertEqual(len(set(agent_ids)), 1)
            self.assertEqual(len(rollouts), 3)
            for states, index in zip(rollouts, (2, 3, 4)):
                expected = np.zeros(5)
                expected[index] = 1
                np.testing.assert_array_equal(states[:, 64:69], np.tile(expected, (8, 1)))
            saved = torch.load(path, map_location="cpu", weights_only=True)
            self.assertTrue(saved["progress"]["completed"])
            self.assertEqual(saved["environment"]["stage"], "full")
            self.assertEqual(saved["environment"]["home_probability"], 1)
            for optimizer in ("actor_optimizer", "critic_optimizer"):
                self.assertTrue(all(s["step"].item() == 6 for s in saved[optimizer]["state"].values()))
            with (Path(directory) / "episodes.csv").open(newline="", encoding="utf-8") as file:
                records = list(csv.DictReader(file))
            switches = [r for r in records if r["cutoff"] == "stage_switch"]
            self.assertEqual([r["stage"] for r in switches], ["press", "open"])
            final = json.loads((Path(directory) / "metrics.json").read_text())
            self.assertEqual(final["home_probability"], 1)
            self.assertTrue(all(r["episode_start"] == "home" for r in final["records"]))
            self.assertEqual(final["full_task_success_rate"], 0)

    def test_original_hand_checkpoint_is_rejected(self):
        with tempfile.TemporaryDirectory() as directory:
            path = Path(directory) / "original.pt"
            torch.save({"format": "rlinit-door-ppohand-v1"}, path)
            with self.assertRaisesRegex(ValueError, "curriculum .pt"):
                load_agent(path, device="cpu")


if __name__ == "__main__":
    unittest.main()
