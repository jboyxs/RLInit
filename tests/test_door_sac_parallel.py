"""Check real spawned physics workers and parallel SAC update accounting."""

from functools import partial
import json
import multiprocessing
import os
from pathlib import Path
import tempfile
import unittest

import numpy as np
import torch
from stable_baselines3.common.vec_env import SubprocVecEnv

from rlinit.algorithmhand.door_env import DEFAULT_MODEL, DoorConfig
from trainp2_sac import evaluate, load_agent, make_worker
from trainp2_sac_parallel import train


class ParallelSACTests(unittest.TestCase):
    def test_independent_spawned_workers_timeout_and_cleanup(self):
        with tempfile.TemporaryDirectory() as directory:
            env = SubprocVecEnv([
                partial(make_worker, DEFAULT_MODEL, DoorConfig(max_steps=2), Path(directory) / str(rank))
                for rank in range(2)
            ], start_method="spawn")
            processes = env.processes
            try:
                self.assertEqual(len({process.pid for process in processes}), 2)
                self.assertTrue(all(process.pid != os.getpid() for process in processes))
                self.assertEqual(env.seed(42), [42, 43])
                first = env.reset()
                self.assertFalse(np.array_equal(first[0], first[1]))
                env.seed(42)
                np.testing.assert_array_equal(first, env.reset())
                actions = np.zeros((2, 8), dtype=np.float32)
                actions[:, -1] = -1
                _, _, dones, _ = env.step(actions)
                self.assertFalse(dones.any())
                obs, _, dones, infos = env.step(actions)
                self.assertTrue(dones.all())
                self.assertEqual(obs.shape, (2, 64))
                for info in infos:
                    self.assertTrue(info["TimeLimit.truncated"])
                    self.assertEqual(info["terminal_observation"].shape, (64,))
            finally:
                env.close()
            self.assertTrue(all(not process.is_alive() for process in processes))
            for rank in range(2):
                self.assertTrue((Path(directory) / f"{rank}.monitor.csv").is_file())

    def test_parallel_updates_eval_frequency_and_checkpoint_compatibility(self):
        with tempfile.TemporaryDirectory() as directory:
            output = Path(directory)
            previous_threads = torch.get_num_threads()
            previous_children = {child.pid for child in multiprocessing.active_children()}
            checkpoint = train(
                n_envs=2, torch_threads=1, total_timesteps=25, output_dir=output,
                seed=5, device="cpu", env_config=DoorConfig(max_steps=4),
                learning_starts=4, buffer_size=64, batch_size=8, train_freq=2,
                gamma=0.999, eval_freq=5, eval_episodes=1,
            )
            self.assertEqual(torch.get_num_threads(), previous_threads)
            self.assertEqual({child.pid for child in multiprocessing.active_children()}, previous_children)
            config = json.loads((output / "config.json").read_text())
            self.assertEqual(config["n_envs"], 2)
            self.assertEqual(config["worker_seeds"], [5, 6])
            self.assertEqual(config["torch_threads"], 1)
            self.assertEqual(config["agent"]["gradient_steps"], -1)
            self.assertEqual(config["actual_timesteps"], 28)
            self.assertEqual(config["gradient_updates"], 24)
            self.assertEqual(config["effective_eval_freq"], 6)
            self.assertGreater(config["transitions_per_second"], 0)
            with np.load(output / "eval" / "evaluations.npz") as evaluations:
                np.testing.assert_array_equal(evaluations["timesteps"], [6, 12, 18, 24])
            (output / "config.json").unlink()
            for path in (checkpoint, output / "best" / "best_model.zip"):
                agent, env = load_agent(path, device="cpu")
                try:
                    self.assertEqual(agent.gamma, env.config.gamma)
                    self.assertEqual(env.config.max_steps, 4)
                    obs, _ = env.reset(seed=123)
                    action, _ = agent.predict(obs, deterministic=True)
                    self.assertTrue(env.action_space.contains(action))
                finally:
                    env.close()
                metrics = evaluate(path, episodes=1, seed=123, device="cpu")
                self.assertEqual(metrics["first_seed"], 123)

    def test_invalid_parallel_settings_restore_threads(self):
        previous_threads = torch.get_num_threads()
        for invalid in ({"n_envs": 0}, {"torch_threads": 0},
                        {"n_envs": 2, "buffer_size": 1}, {"n_envs": 2, "render": True}):
            with self.subTest(invalid=invalid), self.assertRaises(ValueError):
                train(**invalid)
            self.assertEqual(torch.get_num_threads(), previous_threads)


if __name__ == "__main__":
    unittest.main()
