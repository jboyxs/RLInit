"""Continuous PPO checks against the current reaching environment."""

import unittest
from pathlib import Path
import tempfile
from unittest.mock import patch

import numpy as np
import torch

from rlinit.algorithmhand.PPOhand import PPO, compute_gae
from rlinit.algorithmhand.reach_env import ReachEnv
from train_hand import load_agent, save_agent


def make_rollout(agent, count=9):
    rng = np.random.default_rng(42)
    states = rng.normal(size=(count, agent.state_dim)).astype(np.float32)
    actions, log_probs = zip(*(agent.take_action(state, return_log_prob=True) for state in states))
    return dict(states=states, actions=actions, log_probs=log_probs,
                next_states=states * 0.95, rewards=np.linspace(-0.2, 0.3, count),
                terminated=np.arange(count) == count - 1,
                truncated=np.arange(count) == count // 2)



class PPOHandTests(unittest.TestCase):
    def test_gae_bootstraps_timeout_but_does_not_cross_episodes(self):
        advantages, returns = compute_gae(
            torch.tensor([1., 1.]), torch.tensor([2., 2.]),
            torch.tensor([4., 4.]), torch.tensor([0., 1.]),
            torch.tensor([1., 0.]), gamma=0.9, lmbda=0.95,
        )
        torch.testing.assert_close(advantages, torch.tensor([2.6, -1.]))
        torch.testing.assert_close(returns, torch.tensor([4.6, 1.]))

    def test_real_rollout_updates_both_networks(self):
        torch.manual_seed(42)
        env = ReachEnv(max_steps=8)
        self.addCleanup(env.close)
        agent = PPO.from_env(env, hidden_dim=32, epochs=2, device="cpu")
        self.assertEqual((agent.state_dim, agent.action_dim), (20, 7))
        rollout = {key: [] for key in (
            'states', 'actions', 'rewards', 'next_states',
            'terminated', 'truncated', 'log_probs',
        )}
        state, _ = env.reset(seed=42)
        for _ in range(32):
            action, log_prob = agent.take_action(state, return_log_prob=True)
            self.assertTrue(env.action_space.contains(action))
            next_state, reward, terminated, truncated, _ = env.step(action)
            for key, value in zip(rollout, (
                state, action, reward, next_state, terminated, truncated, log_prob
            )):
                rollout[key].append(value)
            state = next_state
            if terminated or truncated:
                state, _ = env.reset()
        before_actor = [p.detach().clone() for p in agent.actor.parameters()]
        before_critic = [p.detach().clone() for p in agent.critic.parameters()]
        metrics = agent.update(rollout)
        self.assertTrue(all(np.isfinite(v) for v in metrics.values()))
        self.assertTrue(any(not torch.equal(a, b) for a, b in zip(before_actor, agent.actor.parameters())))
        self.assertTrue(any(not torch.equal(a, b) for a, b in zip(before_critic, agent.critic.parameters())))
        first = agent.take_action(state, deterministic=True)
        np.testing.assert_array_equal(first, agent.take_action(state, deterministic=True))

    def test_single_transition_and_legacy_dones(self):
        agent = PPO(hidden_dim=16, epochs=1, device="cpu")
        state = np.zeros(20, dtype=np.float32)
        action = agent.take_action(state)
        metrics = agent.update(dict(states=[state], actions=[action], rewards=[1.],
                                    next_states=[state], dones=[True]))
        self.assertTrue(all(np.isfinite(v) for v in metrics.values()))

    def test_log_probability_is_finite_near_action_boundaries(self):
        agent = PPO(hidden_dim=16, device="cpu")
        value = agent.actor.log_prob(torch.zeros(2, 20), torch.tensor([[1.] * 7, [-1.] * 7]))
        self.assertEqual(value.shape, (2,))
        self.assertTrue(torch.isfinite(value).all())

    def test_checkpoint_roundtrip_and_legacy_metadata(self):
        env = ReachEnv()
        self.addCleanup(env.close)
        state, _ = env.reset(seed=42)
        config = dict(state_dim=20, action_dim=7, hidden_dim=16, epochs=2, device="cpu")
        agent = PPO(**config)
        with tempfile.TemporaryDirectory() as directory:
            path = Path(directory) / "ppo.pt"
            save_agent(agent, path, config, 123)
            checkpoint = torch.load(path, weights_only=True)
            for marker in ("original", "minibatch", None):
                if marker is None:
                    checkpoint.pop("variant", None)
                else:
                    checkpoint["variant"] = marker
                torch.save(checkpoint, path)
                loaded = load_agent(path, env, device="cpu")
                self.assertIs(type(loaded), PPO)
                np.testing.assert_array_equal(agent.take_action(state, deterministic=True),
                                              loaded.take_action(state, deterministic=True))
            checkpoint["variant"] = "network"
            torch.save(checkpoint, path)
            with self.assertRaises(ValueError):
                load_agent(path, env, device="cpu")


    def test_minibatches_cover_rollout_keep_targets_fixed_and_update_both_networks(self):
        np.random.seed(7)
        agent = PPO(hidden_dim=16, epochs=2, batch_size=4, device="cpu")
        rollout = make_rollout(agent)
        before = [p.detach().clone() for p in (*agent.actor.parameters(), *agent.critic.parameters())]
        seen = []
        update_batch = agent._update_batch

        def record_batch(states, actions, advantage, returns, old_log_probs):
            seen.append((states.clone(), returns.clone(), old_log_probs.clone()))
            if len(advantage) > 1:
                self.assertAlmostEqual(float(advantage.mean()), 0.0, places=5)
                self.assertAlmostEqual(float(advantage.std()), 1.0, places=5)
            return update_batch(states, actions, advantage, returns, old_log_probs)

        with patch.object(agent, "_prepare_rollout", wraps=agent._prepare_rollout) as prepare, \
                patch.object(agent, "_update_batch", side_effect=record_batch), \
                patch.object(agent.actor_optimizer, "step", wraps=agent.actor_optimizer.step) as actor_step, \
                patch.object(agent.critic_optimizer, "step", wraps=agent.critic_optimizer.step) as critic_step:
            losses = agent.update(rollout)
            prepare.assert_called_once()
            self.assertEqual((actor_step.call_count, critic_step.call_count), (6, 6))
        self.assertTrue(all(np.isfinite(value) for value in losses.values()))
        targets = {}
        for epoch in range(2):
            covered = []
            for states, returns, probabilities in seen[epoch * 3:(epoch + 1) * 3]:
                for state, target, probability in zip(states, returns, probabilities):
                    sample = int(np.flatnonzero(np.all(rollout['states'] == state.numpy(), axis=1))[0])
                    covered.append(sample)
                    if epoch == 0:
                        targets[sample] = (target, probability)
                    else:
                        torch.testing.assert_close(target, targets[sample][0], rtol=0, atol=0)
                        torch.testing.assert_close(probability, targets[sample][1], rtol=0, atol=0)
            self.assertEqual(sorted(covered), list(range(9)))
        self.assertFalse(torch.equal(seen[0][0], seen[3][0]))
        after = [p for p in (*agent.actor.parameters(), *agent.critic.parameters())]
        split = len(list(agent.actor.parameters()))
        self.assertTrue(any(not torch.equal(a, b) for a, b in zip(before[:split], after[:split])))
        self.assertTrue(any(not torch.equal(a, b) for a, b in zip(before[split:], after[split:])))


    def test_minibatch_rejects_invalid_size_and_handles_single_transition(self):
        for batch_size in (0, 1, 2.5):
            with self.assertRaises(ValueError):
                PPO(batch_size=batch_size)
        agent = PPO(hidden_dim=16, epochs=1, device="cpu")
        losses = agent.update(make_rollout(agent, count=1))
        self.assertTrue(all(np.isfinite(value) for value in losses.values()))


    def test_single_hidden_layer_checkpoint_without_depth_metadata(self):
        env = ReachEnv()
        self.addCleanup(env.close)
        state, _ = env.reset(seed=42)
        config = dict(state_dim=20, action_dim=7, hidden_dim=16, hidden_depth=1, device="cpu")
        original = PPO(**config)
        with tempfile.TemporaryDirectory() as directory:
            path = Path(directory) / "single_layer.pt"
            save_agent(original, path, config, 123)
            checkpoint = torch.load(path, weights_only=True)
            del checkpoint["config"]["hidden_depth"]
            torch.save(checkpoint, path)
            loaded = load_agent(path, env, device="cpu")
            self.assertEqual(loaded.hidden_depth, 1)
            np.testing.assert_array_equal(original.take_action(state, deterministic=True),
                                          loaded.take_action(state, deterministic=True))



if __name__ == '__main__':
    unittest.main()
