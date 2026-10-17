"""Verify physical demonstrations, timeout masks and SAC prefill/updates."""

import json
from pathlib import Path
import tempfile
import unittest

import numpy as np
from stable_baselines3 import SAC
from stable_baselines3.common.buffers import ReplayBuffer

from rlinit.algorithmhand.door_env import DoorConfig
from rlinit.algorithmhand.door_reference_env import ReferenceRewardConfig, ReferenceRewardDoorEnv
from trainp2_sac import evaluate, load_agent
from trainp2_sac_reference_large_demo import collect_demonstrations, prefill_replay_buffer, train


class DemonstrationSACTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.config = DoorConfig(initial_noise=0.02, gamma=0.99)
        cls.reward_config = ReferenceRewardConfig(reward_scale=2.5)
        cls.data, cls.report = collect_demonstrations(
            episodes=2, seed=30_000, env_config=cls.config, reward_config=cls.reward_config)

    def test_demonstrations_are_reproducible_training_env_transitions(self):
        self.assertEqual(self.report["success_rate"], 1.0)
        env = ReferenceRewardDoorEnv(config=self.config, reward_config=self.reward_config)
        self.addCleanup(env.close)
        for episode, record in enumerate(self.report["records"]):
            obs, _ = env.reset(seed=record["seed"])
            indices = np.flatnonzero(self.data["episode_ids"] == episode)
            self.assertEqual(len(indices), record["length"])
            for index in indices:
                np.testing.assert_array_equal(obs, self.data["observations"][index])
                action = self.data["actions"][index]
                self.assertTrue(env.action_space.contains(action))
                obs, reward, terminated, truncated, info = env.step(action)
                np.testing.assert_array_equal(obs, self.data["next_observations"][index])
                self.assertAlmostEqual(reward, self.data["rewards"][index], places=6)
                self.assertEqual(terminated, self.data["terminated"][index])
                self.assertEqual(truncated, self.data["truncated"][index])
            self.assertTrue(info["is_success"])
            self.assertEqual(reward, 2.5)
            np.testing.assert_array_equal(env.data.qfrc_applied, 0)
            np.testing.assert_array_equal(env.data.xfrc_applied, 0)
        # Noisy resets actually produce different initial observations.
        starts = [np.flatnonzero(self.data["episode_ids"] == i)[0] for i in range(2)]
        self.assertFalse(np.array_equal(*self.data["observations"][starts]))

    def test_prefill_preserves_all_transitions_and_parallel_padding(self):
        env = ReferenceRewardDoorEnv(config=self.config)
        self.addCleanup(env.close)
        agent = SAC("MlpPolicy", env, buffer_size=2048, device="cpu")
        first_episode = self.data["episode_ids"] == 0
        data = {key: values[first_episode].copy() for key, values in self.data.items()}
        count = len(data["rewards"])
        agent.replay_buffer = ReplayBuffer(2048, env.observation_space, env.action_space, n_envs=2)
        metadata = prefill_replay_buffer(agent, data)
        inserted = metadata["inserted_transitions"]
        self.assertEqual(metadata["padding_transitions"], count % 2)
        replay = agent.replay_buffer
        for field, values in (("observations", data["observations"]),
                              ("next_observations", data["next_observations"]),
                              ("actions", data["actions"]), ("rewards", data["rewards"]),
                              ("dones", data["terminated"] | data["truncated"])):
            actual = getattr(replay, field)[:metadata["replay_rows"]].reshape((inserted, *values.shape[1:]))
            if field == "actions":
                # SB3 rescales even an already normalized Box, adding float32 rounding.
                np.testing.assert_allclose(actual[:count], values, rtol=0, atol=np.finfo(np.float32).eps)
                np.testing.assert_allclose(actual[count:], values[:inserted - count],
                                           rtol=0, atol=np.finfo(np.float32).eps)
            else:
                np.testing.assert_array_equal(actual[:count], values)
                np.testing.assert_array_equal(actual[count:], values[:inserted - count])
        self.assertEqual(replay.timeouts.sum(), 0)
        agent.replay_buffer = ReplayBuffer(4, env.observation_space, env.action_space, n_envs=2)
        with self.assertRaisesRegex(ValueError, "increase --buffer-size"):
            prefill_replay_buffer(agent, data)

    def test_collector_timeouts_bootstrap_but_success_does_not(self):
        env = ReferenceRewardDoorEnv(config=self.config)
        self.addCleanup(env.close)
        agent = SAC("MlpPolicy", env, buffer_size=8, device="cpu")
        terminal_index = np.flatnonzero(self.data["terminated"])[0]
        data = {key: values[[0, terminal_index]].copy() for key, values in self.data.items()}
        # A collection stopped early has a time limit, whereas success is terminal.
        data["truncated"][0] = True
        prefill_replay_buffer(agent, data)
        samples = agent.replay_buffer._get_samples(np.array([0, 1]))
        np.testing.assert_array_equal(samples.dones.cpu().numpy().ravel(), [0, 1])
        np.testing.assert_array_equal(samples.next_observations.cpu().numpy(), data["next_observations"])

    def test_single_and_parallel_training_save_dataset_replay_and_portable_models(self):
        for n_envs in (1, 2):
            with self.subTest(n_envs=n_envs), tempfile.TemporaryDirectory() as directory:
                output = Path(directory)
                checkpoint = train(
                    n_envs=n_envs, torch_threads=1, total_timesteps=8, output_dir=output,
                    device="cpu", env_config=DoorConfig(max_steps=4), demo_episodes=2,
                    reward_config=ReferenceRewardConfig(reward_shaping=False, reward_scale=1.2),
                    learning_starts=0, buffer_size=64, batch_size=4,
                    eval_freq=4, eval_episodes=1,
                )
                config = json.loads((output / "config.json").read_text())
                self.assertEqual(config["gradient_updates"], 8)
                self.assertEqual(config["actual_timesteps"], 8)  # Demos do not consume online budget.
                self.assertEqual(config["agent"]["net_arch"], [1024, 1024])
                self.assertEqual(config["demonstrations"]["transitions"], 8)
                report = json.loads((output / "demonstrations.json").read_text())
                self.assertTrue(all(r["truncated"] for r in report["records"]))
                self.assertEqual(report["environment"], config["environment"])
                with np.load(output / "demonstrations.npz", allow_pickle=False) as data:
                    self.assertEqual(data["actions"].shape, (8, 8))
                    self.assertEqual(data["observations"].shape, (8, 64))
                    self.assertEqual(data["rewards"].sum(), 0)
                agent, env = load_agent(checkpoint, device="cpu")
                try:
                    self.assertEqual(agent.door_demonstrations, config["demonstrations"])
                    agent.load_replay_buffer(output / "replay_buffer.pkl")
                    replay = agent.replay_buffer
                    self.assertEqual(replay.size() * n_envs, 16)
                    demo_rows = 8 // n_envs
                    with np.load(output / "demonstrations.npz") as data:
                        np.testing.assert_array_equal(
                            replay.observations[:demo_rows].reshape(8, 64), data["observations"])
                        np.testing.assert_array_equal(
                            replay.next_observations[:demo_rows].reshape(8, 64), data["next_observations"])
                        np.testing.assert_array_equal(
                            replay.timeouts[:demo_rows].ravel(), data["truncated"])
                    losses = (output / "logs" / "progress.csv").read_text()
                    self.assertIn("train/critic_loss", losses)
                finally:
                    env.close()
                (output / "config.json").unlink()
                metrics = evaluate(output / "best" / "best_model.zip", episodes=1, device="cpu")
                self.assertEqual(metrics["mean_return"], 0)


if __name__ == "__main__":
    unittest.main()
