"""Checks for isolated network/update experiments and checkpoint compatibility."""

from pathlib import Path
import tempfile
import unittest
from unittest.mock import patch

import numpy as np
import torch
from stable_baselines3.common.policies import ActorCriticPolicy

from rlinit.algorithmhand.PPOhand import PPO
from rlinit.algorithmhand.PPOhand_network import PPO as NetworkPPO
from rlinit.algorithmhand.PPOhand_minibatch import PPO as MinibatchPPO
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


class PPOVariantTests(unittest.TestCase):
    def test_network_matches_sb3_mlp_forward_and_initialization(self):
        env = ReachEnv()
        self.addCleanup(env.close)
        agent = NetworkPPO.from_env(env, device="cpu")
        policy = ActorCriticPolicy(env.observation_space, env.action_space,
                                  lr_schedule=lambda _: 3e-4,
                                  net_arch=dict(pi=[64, 64], vf=[64, 64]))
        # Compare the actual SB3 MLP with identical weights, independent of the
        # intentionally retained original action distribution and optimizers.
        for branch, network, output in (
            (policy.mlp_extractor.policy_net, agent.actor, policy.action_net),
            (policy.mlp_extractor.value_net, agent.critic, policy.value_net),
        ):
            branch[0].load_state_dict(network.fc1.state_dict())
            branch[2].load_state_dict(network.fc2.state_dict())
            output.load_state_dict(network.fc3.state_dict())
        states = torch.randn(5, 20)
        latent_pi, latent_vf = policy.mlp_extractor(states)
        torch.testing.assert_close(agent.actor(states)[0], policy.action_net(latent_pi))
        torch.testing.assert_close(agent.critic(states), policy.value_net(latent_vf))
        for layer, gain in ((agent.actor.fc3, 0.01), (agent.critic.fc3, 1.0)):
            torch.testing.assert_close(layer.weight @ layer.weight.T,
                                       torch.eye(layer.out_features) * gain ** 2)
            torch.testing.assert_close(layer.bias, torch.zeros_like(layer.bias))
        self.assertIs(NetworkPPO.update, PPO.update)
        torch.testing.assert_close(agent.actor.log_std, torch.full((7,), -0.5))

    def test_minibatch_keeps_original_networks_and_optimizer_settings(self):
        torch.manual_seed(42)
        original = PPO(device="cpu")
        torch.manual_seed(42)
        minibatch = MinibatchPPO(device="cpu")
        for left, right in ((original.actor, minibatch.actor), (original.critic, minibatch.critic)):
            for name, value in left.state_dict().items():
                torch.testing.assert_close(value, right.state_dict()[name], rtol=0, atol=0)
        for name in ("actor_optimizer", "critic_optimizer"):
            self.assertEqual(getattr(original, name).param_groups[0]["lr"],
                             getattr(minibatch, name).param_groups[0]["lr"])

    def test_minibatches_cover_rollout_keep_targets_fixed_and_update_both_networks(self):
        np.random.seed(7)
        agent = MinibatchPPO(hidden_dim=16, epochs=2, batch_size=4, device="cpu")
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

    def test_checkpoint_roundtrip_and_legacy_checkpoint(self):
        env = ReachEnv()
        self.addCleanup(env.close)
        state, _ = env.reset(seed=42)
        with tempfile.TemporaryDirectory() as directory:
            for agent_class in (PPO, NetworkPPO, MinibatchPPO):
                config = dict(state_dim=20, action_dim=7, hidden_dim=16, epochs=2, device="cpu")
                if agent_class is MinibatchPPO:
                    config['batch_size'] = 4
                agent = agent_class(**config)
                path = Path(directory) / f'{agent.variant}.pt'
                save_agent(agent, path, config, 123)
                loaded = load_agent(path, env, device="cpu")
                self.assertIs(type(loaded), agent_class)
                np.testing.assert_array_equal(agent.take_action(state, deterministic=True),
                                              loaded.take_action(state, deterministic=True))
                if agent_class is PPO:
                    checkpoint = torch.load(path, weights_only=True)
                    del checkpoint['variant']
                    torch.save(checkpoint, path)
                    self.assertIs(type(load_agent(path, env, device="cpu")), PPO)

    def test_minibatch_rejects_invalid_size_and_handles_single_transition(self):
        for batch_size in (0, 1, 2.5):
            with self.assertRaises(ValueError):
                MinibatchPPO(batch_size=batch_size)
        agent = MinibatchPPO(hidden_dim=16, epochs=1, device="cpu")
        losses = agent.update(make_rollout(agent, count=1))
        self.assertTrue(all(np.isfinite(value) for value in losses.values()))


if __name__ == '__main__':
    unittest.main()
