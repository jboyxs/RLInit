"""手写连续动作 PPO，适配 ReachEnv 的 20 维观测和 7 维动作。

动作是 [-1, 1] 内的关节目标增量，由环境映射为位置执行器输入。
采样期间不要更新策略；update 接收单环境按时间顺序排列的 rollout。
"""

import numpy as np
import torch
import torch.nn.functional as F
from torch.distributions import Normal

from .device import resolve_device


class PolicyNet(torch.nn.Module):
    def __init__(self, state_dim, hidden_dim, action_dim):
        super().__init__()
        self.fc1 = torch.nn.Linear(state_dim, hidden_dim)
        self.fc2 = torch.nn.Linear(hidden_dim, action_dim)
        self.log_std = torch.nn.Parameter(torch.full((action_dim,), -0.5))

    def forward(self, x):
        """返回高斯分布的均值和标准差，动作随后通过 tanh 限幅。"""
        mean = self.fc2(F.relu(self.fc1(x)))
        std = self.log_std.clamp(-5, 2).exp().expand_as(mean)
        return mean, std

    def log_prob(self, states, actions):
        mean, std = self(states)
        actions = actions.clamp(-1 + 1e-6, 1 - 1e-6)
        raw_actions = torch.atanh(actions)
        # tanh 变换的 Jacobian 修正；七个关节共同组成一个动作。
        return (Normal(mean, std).log_prob(raw_actions)
                - torch.log1p(-actions.square())).sum(dim=-1)


class ValueNet(torch.nn.Module):
    def __init__(self, state_dim, hidden_dim):
        super().__init__()
        self.fc1 = torch.nn.Linear(state_dim, hidden_dim)
        self.fc2 = torch.nn.Linear(hidden_dim, 1)

    def forward(self, x):
        return self.fc2(F.relu(self.fc1(x)))


def compute_gae(rewards, values, next_values, terminated, truncated, gamma, lmbda):
    """终止不 bootstrap；超时仍 bootstrap，但两者都切断跨回合 GAE。"""
    deltas = rewards + gamma * next_values * (1 - terminated) - values
    advantages = torch.zeros_like(deltas)
    running = torch.zeros((), device=deltas.device)
    for t in reversed(range(len(deltas))):
        continuation = (1 - terminated[t]) * (1 - truncated[t])
        running = deltas[t] + gamma * lmbda * continuation * running
        advantages[t] = running
    return advantages, advantages + values


class PPO:
    """采用 tanh 高斯策略和截断目标的 PPO。

    保留原构造参数顺序，默认适配当前固定底盘到达任务。
    推荐通过 from_env(env) 自动读取维度。
    """

    def __init__(self, state_dim=20, hidden_dim=128, action_dim=7,
                 actor_lr=3e-4, critic_lr=1e-3, lmbda=0.95,
                 epochs=10, eps=0.2, gamma=0.99, device="auto"):
        if min(state_dim, hidden_dim, action_dim, epochs) < 1:
            raise ValueError("网络维度和 epochs 必须为正整数")
        if not (0 <= gamma <= 1 and 0 <= lmbda <= 1 and 0 < eps < 1):
            raise ValueError("gamma/lmbda 应在 [0, 1]，eps 应在 (0, 1)")
        self.device = resolve_device(device)
        self.state_dim, self.action_dim = state_dim, action_dim
        self.actor = PolicyNet(state_dim, hidden_dim, action_dim).to(self.device)
        self.critic = ValueNet(state_dim, hidden_dim).to(self.device)
        self.actor_optimizer = torch.optim.Adam(self.actor.parameters(), lr=actor_lr)
        self.critic_optimizer = torch.optim.Adam(self.critic.parameters(), lr=critic_lr)
        self.gamma, self.lmbda, self.epochs, self.eps = gamma, lmbda, epochs, eps

    @classmethod
    def from_env(cls, env, **kwargs):
        if (len(env.observation_space.shape) != 1 or len(env.action_space.shape) != 1
                or not np.all(env.action_space.low == -1)
                or not np.all(env.action_space.high == 1)):
            raise ValueError("需要一维向量观测和 [-1, 1] 连续动作空间")
        return cls(state_dim=env.observation_space.shape[0],
                   action_dim=env.action_space.shape[0], **kwargs)

    @torch.no_grad()
    def take_action(self, state, deterministic=False, return_log_prob=False):
        """返回 (7,) float32 动作；可同时返回采样策略的联合 log_prob。"""
        state = torch.as_tensor(np.asarray(state), dtype=torch.float32, device=self.device)
        if state.shape != (self.state_dim,) or not torch.isfinite(state).all():
            raise ValueError(f"state 必须是 {self.state_dim} 维有限向量")
        mean, std = self.actor(state)
        raw_action = mean if deterministic else Normal(mean, std).sample()
        action = raw_action.tanh().clamp(-1 + 1e-6, 1 - 1e-6)
        result = action.cpu().numpy().copy()
        if return_log_prob:
            return result, float(self.actor.log_prob(state, action).item())
        return result

    def update(self, transition_dict):
        """更新一段新采样的 rollout，返回损失指标。

        必填 states/actions/rewards/next_states/terminated/truncated，推荐
        提供 take_action(return_log_prob=True) 返回的 log_probs。
        next_states 必须是 step 返回的观测，不能替换成 reset 后的观测。
        兼容旧 dones 字段，但此时将所有 done 视为真正终止，无法区分超时。
        """
        def tensor(key):
            result = torch.as_tensor(np.asarray(transition_dict[key]),
                                     dtype=torch.float32, device=self.device)
            if not torch.isfinite(result).all():
                raise ValueError(f"{key} 含非有限值")
            return result

        states, actions = tensor("states"), tensor("actions")
        next_states, rewards = tensor("next_states"), tensor("rewards").flatten()
        count = len(rewards)
        if (count == 0 or states.shape != (count, self.state_dim)
                or next_states.shape != states.shape
                or actions.shape != (count, self.action_dim)):
            raise ValueError("rollout 不能为空，状态/动作维度及样本数须匹配")
        if torch.any(actions.abs() > 1):
            raise ValueError("动作必须在 [-1, 1] 内")
        if "terminated" in transition_dict:
            terminated = tensor("terminated").flatten()
            truncated = tensor("truncated").flatten()
        else:
            terminated = tensor("dones").flatten()
            truncated = torch.zeros_like(terminated)
        for flags in (terminated, truncated):
            if flags.shape != (count,) or not torch.all((flags == 0) | (flags == 1)):
                raise ValueError("结束标志须为每个样本对应的布尔值")

        with torch.no_grad():
            values = self.critic(states).squeeze(-1)
            next_values = self.critic(next_states).squeeze(-1)
            advantage, returns = compute_gae(rewards, values, next_values,
                                            terminated, truncated, self.gamma, self.lmbda)
            old_log_probs = (tensor("log_probs").flatten() if "log_probs" in transition_dict
                             else self.actor.log_prob(states, actions))
            if old_log_probs.shape != (count,):
                raise ValueError("log_probs 须为每个动作的联合对数概率")
            if count > 1:
                advantage = (advantage - advantage.mean()) / (advantage.std(unbiased=False) + 1e-8)

        for _ in range(self.epochs):
            log_probs = self.actor.log_prob(states, actions)
            ratio = torch.exp(log_probs - old_log_probs)
            surrogate = ratio * advantage
            clipped = ratio.clamp(1 - self.eps, 1 + self.eps) * advantage
            actor_loss = -torch.minimum(surrogate, clipped).mean()
            critic_loss = F.mse_loss(self.critic(states).squeeze(-1), returns)
            if not torch.isfinite(actor_loss) or not torch.isfinite(critic_loss):
                raise RuntimeError("PPO 损失非有限，请检查 rollout 或学习率")
            self.actor_optimizer.zero_grad()
            actor_loss.backward()
            torch.nn.utils.clip_grad_norm_(self.actor.parameters(), 0.5)
            self.actor_optimizer.step()
            self.critic_optimizer.zero_grad()
            critic_loss.backward()
            torch.nn.utils.clip_grad_norm_(self.critic.parameters(), 0.5)
            self.critic_optimizer.step()
        return {"actor_loss": actor_loss.item(), "critic_loss": critic_loss.item()}
