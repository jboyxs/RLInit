"""网络对照版：仅对齐 SB3 的双层 Tanh MLP 与正交权重初始化。

默认 Actor 为 20→64→64→7，Critic 为 20→64→64→1。
动作分布（含 log_std=-0.5）、tanh 动作变换、学习率和小批次更新沿用原版。
"""

import math

import torch

from .PPOhand import PPO as BasePPO, PolicyNet as BasePolicyNet


def _init_layers(fc1, fc2, output, output_gain):
    for layer, gain in ((fc1, math.sqrt(2)), (fc2, math.sqrt(2)), (output, output_gain)):
        torch.nn.init.orthogonal_(layer.weight, gain=gain)
        torch.nn.init.zeros_(layer.bias)


class PolicyNet(BasePolicyNet):
    def __init__(self, state_dim, hidden_dim, action_dim):
        torch.nn.Module.__init__(self)
        self.fc1 = torch.nn.Linear(state_dim, hidden_dim)
        self.fc2 = torch.nn.Linear(hidden_dim, hidden_dim)
        self.fc3 = torch.nn.Linear(hidden_dim, action_dim)
        self.log_std = torch.nn.Parameter(torch.full((action_dim,), -0.5))
        _init_layers(self.fc1, self.fc2, self.fc3, output_gain=0.01)

    def forward(self, x):
        mean = self.fc3(torch.tanh(self.fc2(torch.tanh(self.fc1(x)))))
        std = self.log_std.clamp(-5, 2).exp().expand_as(mean)
        return mean, std


class ValueNet(torch.nn.Module):
    def __init__(self, state_dim, hidden_dim):
        super().__init__()
        self.fc1 = torch.nn.Linear(state_dim, hidden_dim)
        self.fc2 = torch.nn.Linear(hidden_dim, hidden_dim)
        self.fc3 = torch.nn.Linear(hidden_dim, 1)
        _init_layers(self.fc1, self.fc2, self.fc3, output_gain=1.0)

    def forward(self, x):
        return self.fc3(torch.tanh(self.fc2(torch.tanh(self.fc1(x)))))


class PPO(BasePPO):
    variant = "network"
    policy_net_class = PolicyNet
    value_net_class = ValueNet

    def __init__(self, state_dim=20, hidden_dim=64, action_dim=7,
                 actor_lr=3e-4, critic_lr=1e-3, lmbda=0.95,
                 epochs=10, eps=0.2, gamma=0.99, device="auto", batch_size=64):
        super().__init__(state_dim, hidden_dim, action_dim, actor_lr, critic_lr,
                         lmbda, epochs, eps, gamma, device, batch_size=batch_size)
