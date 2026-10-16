"""手写连续动作 PPO，通过 from_env 适配到达或开门任务的维度。

动作是 [-1, 1] 内的关节目标增量，由环境映射为位置执行器输入。
采样期间不要更新策略；update 接收按时间排列的单环境或并行环境 rollout。
"""

import numpy as np
import torch
import torch.nn.functional as F
from torch.distributions import Normal

from .device import resolve_device


def _squashed_log_prob(mean, std, actions):
    actions = actions.clamp(-1 + 1e-6, 1 - 1e-6)
    raw_actions = torch.atanh(actions)
    # tanh 变换的 Jacobian 修正；所有动作分量共同组成一个动作。
    return (Normal(mean, std).log_prob(raw_actions)
            - torch.log1p(-actions.square())).sum(dim=-1)


def checkpoint_hidden_depth(actor_state):
    """Infer the architecture of checkpoints that predate depth metadata."""
    return 3 if "fc4.weight" in actor_state else 2 if "fc3.weight" in actor_state else 1


class PolicyNet(torch.nn.Module):
    def __init__(self, state_dim, hidden_dim, action_dim, hidden_depth=3):
        super().__init__()
        self.hidden_depth = hidden_depth
        for layer in range(1, hidden_depth + 1):
            setattr(self, f"fc{layer}", torch.nn.Linear(state_dim if layer == 1 else hidden_dim, hidden_dim))
        setattr(self, f"fc{hidden_depth + 1}", torch.nn.Linear(hidden_dim, action_dim))
        self.log_std = torch.nn.Parameter(torch.full((action_dim,), -0.5))

    def forward(self, x):
        """返回高斯分布的均值和标准差，动作随后通过 tanh 限幅。"""
        hidden = x
        for layer in range(1, self.hidden_depth + 1):
            hidden = F.relu(getattr(self, f"fc{layer}")(hidden))
        mean = getattr(self, f"fc{self.hidden_depth + 1}")(hidden)
        std = self.log_std.clamp(-5, 2).exp().expand_as(mean)
        return mean, std

    def log_prob(self, states, actions):
        mean, std = self(states)
        return _squashed_log_prob(mean, std, actions)


class ValueNet(torch.nn.Module):
    def __init__(self, state_dim, hidden_dim, hidden_depth=3):
        super().__init__()
        self.hidden_depth = hidden_depth
        for layer in range(1, hidden_depth + 1):
            setattr(self, f"fc{layer}", torch.nn.Linear(state_dim if layer == 1 else hidden_dim, hidden_dim))
        setattr(self, f"fc{hidden_depth + 1}", torch.nn.Linear(hidden_dim, 1))

    def forward(self, x):
        hidden = x
        for layer in range(1, self.hidden_depth + 1):
            hidden = F.relu(getattr(self, f"fc{layer}")(hidden))
        return getattr(self, f"fc{self.hidden_depth + 1}")(hidden)


def compute_gae(rewards, values, next_values, terminated, truncated, gamma, lmbda):
    """终止不 bootstrap；超时仍 bootstrap，但两者都切断跨回合 GAE。"""
    deltas = rewards + gamma * next_values * (1 - terminated) - values
    advantages = torch.zeros_like(deltas)
    # A separate accumulator for each environment; recurse only through time.
    running = torch.zeros_like(deltas[0])
    for t in reversed(range(len(deltas))):
        continuation = (1 - terminated[t]) * (1 - truncated[t])
        running = deltas[t] + gamma * lmbda * continuation * running
        advantages[t] = running
    return advantages, advantages + values


class PPO:
    """采用 tanh 高斯策略和截断目标的 PPO。

    保留原构造参数顺序，默认适配当前固定底盘到达任务。
    推荐通过 from_env(env) 自动读取维度。默认三层 256 ReLU；
    hidden_dim 指定每层宽度，hidden_depth=1/2 用于恢复旧检查点。
    """

    variant = "original"
    policy_net_class = PolicyNet
    value_net_class = ValueNet
    supports_minibatches = True

    def __init__(self, state_dim=20, hidden_dim=256, action_dim=7,
                 actor_lr=3e-4, critic_lr=1e-3, lmbda=0.95,
                 epochs=10, eps=0.2, gamma=0.99, device="auto", batch_size=64, hidden_depth=3):
        if min(state_dim, hidden_dim, action_dim, epochs) < 1:
            raise ValueError("网络维度和 epochs 必须为正整数")
        if not (0 <= gamma <= 1 and 0 <= lmbda <= 1 and 0 < eps < 1):
            raise ValueError("gamma/lmbda 应在 [0, 1]，eps 应在 (0, 1)")
        if not isinstance(batch_size, int) or batch_size < 2:
            raise ValueError("batch_size 必须是至少为 2 的整数")
        if hidden_depth not in (1, 2, 3):
            raise ValueError("hidden_depth 必须为 1、2 或 3")
        self.device = resolve_device(device)
        self.state_dim, self.action_dim = state_dim, action_dim
        self.hidden_dim = hidden_dim
        self.hidden_depth = hidden_depth
        self.actor = self.policy_net_class(state_dim, hidden_dim, action_dim, hidden_depth=hidden_depth).to(self.device)
        self.critic = self.value_net_class(state_dim, hidden_dim, hidden_depth=hidden_depth).to(self.device)
        self.actor_optimizer = torch.optim.Adam(self.actor.parameters(), lr=actor_lr)
        self.critic_optimizer = torch.optim.Adam(self.critic.parameters(), lr=critic_lr)
        self.gamma, self.lmbda, self.epochs, self.eps = gamma, lmbda, epochs, eps
        self.batch_size = batch_size

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
        """返回 (action_dim,) float32 动作；可返回联合 log_prob。"""
        state = np.asarray(state, dtype=np.float32)
        if state.shape != (self.state_dim,):
            raise ValueError(f"state 必须是 {self.state_dim} 维有限向量")
        result = self.take_actions(state[None], deterministic=deterministic, return_log_prob=return_log_prob)
        return (result[0][0], float(result[1][0])) if return_log_prob else result[0]

    @torch.no_grad()
    def take_actions(self, states, deterministic=False, return_log_prob=False):
        """Batch inference: (n_envs, state_dim) -> actions and optional log_probs."""
        states = np.asarray(states, dtype=np.float32)
        if (states.ndim != 2 or states.shape[0] == 0 or states.shape[1] != self.state_dim
                or not np.isfinite(states).all()):
            raise ValueError(f"states 必须是 (n_envs, {self.state_dim}) 有限数组")
        states = torch.as_tensor(states, dtype=torch.float32, device=self.device)
        mean, std = self.actor(states)
        raw_action = mean if deterministic else Normal(mean, std).sample()
        action = raw_action.tanh().clamp(-1 + 1e-6, 1 - 1e-6)
        if return_log_prob:
            # Reuse this forward pass and transfer actions/probabilities together.
            log_probs = _squashed_log_prob(mean, std, action)
            result = torch.cat((action, log_probs[:, None]), dim=-1).cpu().numpy()
            return result[:, :-1].copy(), result[:, -1].copy()
        return action.cpu().numpy().copy()

    def update(self, transition_dict):
        """更新一段新采样的 rollout，返回损失指标。

        必填 states/actions/rewards/next_states/terminated/truncated，推荐
        提供 take_action(return_log_prob=True) 返回的 log_probs。
        next_states 必须是 step 返回的观测，不能替换成 reset 后的观测。
        并行 rollout 使用 (time, n_envs, dim) 状态/动作和 (time, n_envs)
        奖励/标志/概率；各环境单独计算 GAE 后才展平、打乱。
        兼容旧 dones 字段，但此时将所有 done 视为真正终止，无法区分超时。
        """
        # 先在完整时间序列上计算一次 GAE、returns 和旧策略概率。
        batch = self._prepare_rollout(transition_dict, normalize_advantage=False)
        count = len(batch[0])
        losses = []
        for _ in range(self.epochs):
            indices = torch.as_tensor(np.random.permutation(count), device=self.device)
            for start in range(0, count, self.batch_size):
                selected = indices[start:start + self.batch_size]
                states, actions, advantage, returns, old_log_probs = (x[selected] for x in batch)
                # 每批单独归一化；单条尾批跳过标准差计算。
                if len(advantage) > 1:
                    advantage = (advantage - advantage.mean()) / (advantage.std() + 1e-8)
                losses.append(self._update_batch(states, actions, advantage, returns, old_log_probs))
        return {key: float(np.mean([loss[key] for loss in losses])) for key in losses[0]}

    def _prepare_rollout(self, transition_dict, normalize_advantage=True):
        """在时间顺序不变的完整 rollout 上计算固定的 GAE 和旧策略概率。"""
        def tensor(key):
            result = torch.as_tensor(np.asarray(transition_dict[key]),
                                     dtype=torch.float32, device=self.device)
            if not torch.isfinite(result).all():
                raise ValueError(f"{key} 含非有限值")
            return result

        states, actions = tensor("states"), tensor("actions")
        next_states, rewards = tensor("next_states"), tensor("rewards")
        if states.ndim == 2:
            rewards = rewards.flatten()
        if (states.ndim not in (2, 3) or rewards.numel() == 0
                or states.shape[-1] != self.state_dim or rewards.shape != states.shape[:-1]
                or next_states.shape != states.shape
                or actions.shape != (*states.shape[:-1], self.action_dim)):
            raise ValueError("rollout 不能为空，状态/动作维度及样本数须匹配")
        if torch.any(actions.abs() > 1):
            raise ValueError("动作必须在 [-1, 1] 内")
        if "terminated" in transition_dict:
            terminated = tensor("terminated")
            truncated = tensor("truncated")
        else:
            terminated = tensor("dones")
            truncated = torch.zeros_like(terminated)
        if states.ndim == 2:
            terminated, truncated = terminated.flatten(), truncated.flatten()
        for flags in (terminated, truncated):
            if flags.shape != rewards.shape or not torch.all((flags == 0) | (flags == 1)):
                raise ValueError("结束标志须为每个样本对应的布尔值")

        with torch.no_grad():
            values = self.critic(states).squeeze(-1)
            next_values = self.critic(next_states).squeeze(-1)
            advantage, returns = compute_gae(rewards, values, next_values,
                                            terminated, truncated, self.gamma, self.lmbda)
            old_log_probs = (tensor("log_probs") if "log_probs" in transition_dict
                             else self.actor.log_prob(states, actions))
            if states.ndim == 2:
                old_log_probs = old_log_probs.flatten()
            if old_log_probs.shape != rewards.shape:
                raise ValueError("log_probs 须为每个动作的联合对数概率")
            for name, value in (("advantages", advantage), ("returns", returns),
                                ("log_probs", old_log_probs)):
                if not torch.isfinite(value).all():
                    raise RuntimeError(f"PPO {name} 含非有限值，请检查 rollout 或价值网络")
            if normalize_advantage and rewards.numel() > 1:
                advantage = (advantage - advantage.mean()) / (advantage.std(unbiased=False) + 1e-8)
        return (states.reshape(-1, self.state_dim), actions.reshape(-1, self.action_dim),
                advantage.reshape(-1), returns.reshape(-1), old_log_probs.reshape(-1))

    def _update_batch(self, states, actions, advantage, returns, old_log_probs):
        """一次梯度更新；三种实验共用原版损失、优化器和梯度裁剪。"""
        log_probs = self.actor.log_prob(states, actions)
        # Bound the exponent before exp: PPO's epsilon clipping can hide an
        # infinite ratio in the loss, while backward still computes 0 * inf
        # and corrupts the policy with NaN gradients. This numerical guard is
        # separate from the usual [1-eps, 1+eps] PPO clipping below.
        ratio = (log_probs - old_log_probs).clamp(-20, 20).exp()
        surrogate = ratio * advantage
        clipped = ratio.clamp(1 - self.eps, 1 + self.eps) * advantage
        actor_loss = -torch.minimum(surrogate, clipped).mean()
        critic_loss = F.mse_loss(self.critic(states).squeeze(-1), returns)
        if not torch.isfinite(actor_loss) or not torch.isfinite(critic_loss):
            raise RuntimeError("PPO 损失非有限，请检查 rollout 或学习率")
        self.actor_optimizer.zero_grad()
        self.critic_optimizer.zero_grad()
        actor_loss.backward()
        critic_loss.backward()
        # Validate both networks before either optimizer changes parameters or
        # Adam state; a finite loss alone does not guarantee finite gradients.
        torch.nn.utils.clip_grad_norm_(self.actor.parameters(), 0.5, error_if_nonfinite=True)
        torch.nn.utils.clip_grad_norm_(self.critic.parameters(), 0.5, error_if_nonfinite=True)
        self.actor_optimizer.step()
        self.critic_optimizer.step()
        return {"actor_loss": actor_loss.item(), "critic_loss": critic_loss.item()}
