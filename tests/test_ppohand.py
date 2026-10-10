"""Continuous PPO checks against the current reaching environment."""

import unittest

import numpy as np
import torch

from rlinit.algorithmhand.PPOhand import PPO, compute_gae
from rlinit.algorithmhand.reach_env import ReachEnv


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


if __name__ == '__main__':
    unittest.main()
