"""Parallel rollout boundaries, batched inference, and subprocess curriculum checks."""

import csv
import json
from pathlib import Path
import tempfile
import unittest
from unittest.mock import patch

import numpy as np
import torch

from rlinit.algorithmhand.PPOhand import PPO, checkpoint_hidden_depth
from rlinit.algorithmhand.door_env import DoorConfig
from train_curriculum_hand import evaluate, load_agent
import train_curriculum_hand_parallel as parallel


class ParallelPPOTests(unittest.TestCase):
    def test_default_network_and_batch_actions(self):
        torch.manual_seed(42)
        agent = PPO(state_dim=70, action_dim=8, device="cpu")
        for network, output in ((agent.actor, 8), (agent.critic, 1)):
            layers = [module for module in network.modules() if isinstance(module, torch.nn.Linear)]
            self.assertEqual([(layer.in_features, layer.out_features) for layer in layers],
                             [(70, 256), (256, 256), (256, 256), (256, output)])
        states = np.random.default_rng(42).normal(size=(8, 70)).astype(np.float32)
        with patch.object(agent.actor, "forward", wraps=agent.actor.forward) as forward:
            actions, log_probs = agent.take_actions(states, deterministic=True, return_log_prob=True)
            forward.assert_called_once()
        self.assertEqual(actions.shape, (8, 8))
        self.assertEqual(log_probs.shape, (8,))
        self.assertTrue(np.isfinite(log_probs).all())
        self.assertTrue(np.all(np.abs(actions) < 1))
        for index, state in enumerate(states):
            np.testing.assert_allclose(actions[index], agent.take_action(state, deterministic=True), atol=1e-6)
        with torch.no_grad():
            expected = agent.actor.log_prob(torch.as_tensor(states), torch.as_tensor(actions))
        np.testing.assert_allclose(log_probs, expected.numpy(), atol=1e-6)
        for invalid in (np.zeros(70), np.zeros((0, 70)), np.full((8, 70), np.nan)):
            with self.assertRaises(ValueError):
                agent.take_actions(invalid)

    def test_vector_gae_keeps_environments_episodes_and_timeouts_separate(self):
        agent = PPO(state_dim=2, action_dim=1, hidden_dim=8, device="cpu", epochs=1, batch_size=4,
                    gamma=1., lmbda=1.)
        with torch.no_grad():
            for parameter in agent.critic.parameters():
                parameter.zero_()
            agent.critic.fc4.bias.fill_(2.)
        states = np.arange(12, dtype=np.float32).reshape(3, 2, 2) / 10
        rollout = dict(states=states, actions=np.zeros((3, 2, 1)), next_states=states + 0.1,
                       rewards=[[1., 10.], [2., 20.], [3., 30.]],
                       terminated=[[False, False], [True, False], [False, False]],
                       truncated=[[False, True], [False, False], [True, False]])
        batch = agent._prepare_rollout(rollout, normalize_advantage=False)
        torch.testing.assert_close(batch[0], torch.as_tensor(states.reshape(6, 2)))
        torch.testing.assert_close(batch[2], torch.tensor([1., 10., 0., 50., 3., 30.]))
        torch.testing.assert_close(batch[3], torch.tensor([3., 12., 2., 52., 5., 32.]))
        before_actor = [p.detach().clone() for p in agent.actor.parameters()]
        before_critic = [p.detach().clone() for p in agent.critic.parameters()]
        self.assertTrue(all(np.isfinite(value) for value in agent.update(rollout).values()))
        for before, network in ((before_actor, agent.actor), (before_critic, agent.critic)):
            self.assertTrue(any(not torch.equal(a, b) for a, b in zip(before, network.parameters())))
        with self.assertRaises(ValueError):
            agent.update({**rollout, "rewards": np.zeros(6)})

    def test_same_step_autoreset_uses_terminal_observations(self):
        reset_observations = np.array([[100., 100.], [200., 200.], [300., 300.]])
        infos = {
            "final_obs": np.array([np.array([1., 2.]), None, np.array([3., 4.])], dtype=object),
            "_final_obs": np.array([True, False, True]),
            "final_info": {"is_success": np.array([True, False, False]),
                           "_is_success": np.array([True, False, True])},
            "_final_info": np.array([True, False, True]),
        }
        terminated, truncated = np.array([True, False, False]), np.array([False, False, True])
        actual = parallel.transition_observations(reset_observations, terminated, truncated, infos)
        np.testing.assert_array_equal(actual, [[1., 2.], [200., 200.], [3., 4.]])
        np.testing.assert_array_equal(reset_observations, [[100., 100.], [200., 200.], [300., 300.]])
        self.assertTrue(parallel.info_at(infos, 0)["final_info"]["is_success"])
        self.assertNotIn("final_info", parallel.info_at(infos, 1))
        with self.assertRaises(RuntimeError):
            parallel.transition_observations(reset_observations, terminated, truncated, {})

    def test_old_network_state_dicts_keep_original_architecture(self):
        for depth in (1, 2, 3):
            with self.subTest(depth=depth):
                original = PPO(hidden_dim=16, hidden_depth=depth, device="cpu")
                inferred = checkpoint_hidden_depth(original.actor.state_dict())
                self.assertEqual(inferred, depth)
                restored = PPO(hidden_dim=16, hidden_depth=inferred, device="cpu")
                restored.actor.load_state_dict(original.actor.state_dict())
                restored.critic.load_state_dict(original.critic.state_dict())
                state = np.zeros(20, dtype=np.float32)
                np.testing.assert_array_equal(original.take_action(state, deterministic=True),
                                              restored.take_action(state, deterministic=True))

    def test_subprocess_training_autoresets_checkpoint_and_evaluation(self):
        observed = []
        processes = []
        real_update, make_envs = PPO.update, parallel.make_parallel_envs

        def record(agent, rollout):
            observed.append({key: np.asarray(value) for key, value in rollout.items()})
            return real_update(agent, rollout)

        def record_workers(*args, **kwargs):
            envs = make_envs(*args, **kwargs)
            processes.extend(envs.processes)
            return envs

        with tempfile.TemporaryDirectory() as directory:
            with patch.object(PPO, "update", record), patch.object(parallel, "make_parallel_envs", record_workers):
                path = parallel.train(total_timesteps=7, output_dir=directory, num_envs=2, n_steps=2,
                                      batch_size=4, epochs=1, eval_episodes=1, device="cpu",
                                      config=DoorConfig(gamma=0.99, max_steps=1, initial_noise=0))
            self.assertEqual(len({process.pid for process in processes}), 2)
            self.assertFalse(any(process.is_alive() for process in processes))
            self.assertEqual(len(observed), 2)
            for rollout in observed:
                self.assertEqual(rollout["states"].shape, (2, 2, 70))
                self.assertEqual(rollout["actions"].shape, (2, 2, 8))
                self.assertTrue(rollout["truncated"].all())
            saved = torch.load(path, map_location="cpu", weights_only=True)
            config = json.loads((Path(directory) / "config.json").read_text())
            self.assertEqual(saved["timesteps"], 8)
            self.assertEqual(config["requested_timesteps"], 7)
            self.assertEqual(config["actual_timesteps"], 8)
            self.assertEqual(config["num_envs"], 2)
            self.assertEqual(config["agent"]["hidden_depth"], 3)
            self.assertEqual(saved["actor"]["fc4.weight"].shape, (8, 256))
            with (Path(directory) / "episodes.csv").open() as stream:
                episodes = list(csv.DictReader(stream))
            self.assertEqual(len(episodes), 8)
            self.assertEqual({row["env_id"] for row in episodes}, {"0", "1"})
            self.assertTrue(all(row["length"] == "1" for row in episodes))
            loaded, env = load_agent(path, device="cpu")
            try:
                self.assertEqual(loaded.hidden_depth, 3)
                for key, value in loaded.actor.state_dict().items():
                    torch.testing.assert_close(value, saved["actor"][key], atol=0, rtol=0)
            finally:
                env.close()
            self.assertEqual(evaluate(path, episodes=1, device="cpu", full_task=True)["stage"], "full")

    def test_all_workers_switch_stage_with_one_shared_policy(self):
        rollouts, identities = [], []
        real_update = PPO.update

        def record(agent, rollout):
            rollouts.append(np.asarray(rollout["states"]))
            identities.append(id(agent))
            return real_update(agent, rollout)

        def pass_stage(agent, specification, **kwargs):
            return dict(success_rate=1., home_episodes=0, home_success_rate=None,
                        mean_return=0., mean_min_distance_m=0., full_task_success_rate=0.)

        with tempfile.TemporaryDirectory() as directory:
            with patch.object(PPO, "update", record), patch.object(parallel, "evaluate_agent", pass_stage):
                path = parallel.train(total_timesteps=12, output_dir=directory, num_envs=2, n_steps=2,
                                      batch_size=4, epochs=1, eval_freq=4, eval_episodes=1,
                                      device="cpu", start_stage="press", patience=1, home_step=1,
                                      config=DoorConfig(gamma=0.99, max_steps=4, initial_noise=0))
            self.assertEqual(len(set(identities)), 1)
            self.assertEqual(len(rollouts), 3)
            for states, stage in zip(rollouts, (2, 3, 4)):
                expected = np.zeros((2, 2, 5), dtype=np.float32)
                expected[:, :, stage] = 1
                np.testing.assert_array_equal(states[:, :, 64:69], expected)
            saved = torch.load(path, map_location="cpu", weights_only=True)
            self.assertTrue(saved["progress"]["completed"])
            self.assertEqual(saved["environment"]["stage"], "full")
            for optimizer in ("actor_optimizer", "critic_optimizer"):
                self.assertTrue(all(state["step"].item() == 3 for state in saved[optimizer]["state"].values()))
            with (Path(directory) / "episodes.csv").open() as stream:
                cutoffs = [row for row in csv.DictReader(stream) if row["cutoff"] == "stage_switch"]
            self.assertEqual([row["stage"] for row in cutoffs], ["press", "press", "open", "open"])


if __name__ == "__main__":
    unittest.main()
