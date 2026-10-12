"""更新方式对照版：仅采用 SB3 式打乱小批次及批内优势归一化。

每轮打乱完整 rollout，每批默认 64 条，更新 10 轮。
网络、动作分布、独立优化器、学习率和损失沿用原版。
"""

import numpy as np
import torch

from .PPOhand import PPO as BasePPO


class PPO(BasePPO):
    variant = "minibatch"
    supports_minibatches = True

    def __init__(self, *args, batch_size=64, **kwargs):
        if not isinstance(batch_size, int) or batch_size < 2:
            raise ValueError("batch_size 必须是至少为 2 的整数")
        super().__init__(*args, **kwargs)
        self.batch_size = batch_size

    def update(self, transition_dict):
        # GAE 必须先按时间顺序计算；不能在打乱后的小批次中重新计算。
        batch = self._prepare_rollout(transition_dict, normalize_advantage=False)
        count = len(batch[0])
        losses = []
        for _ in range(self.epochs):
            indices = torch.as_tensor(np.random.permutation(count), device=self.device)
            for start in range(0, count, self.batch_size):
                selected = indices[start:start + self.batch_size]
                states, actions, advantage, returns, old_log_probs = (x[selected] for x in batch)
                # SB3 在每批内归一化，使用样本标准差；单条尾批跳过归一化。
                if len(advantage) > 1:
                    advantage = (advantage - advantage.mean()) / (advantage.std() + 1e-8)
                losses.append(self._update_batch(states, actions, advantage, returns, old_log_probs))
        # 与 SB3 一样报告本次全部小批次的损失均值。
        return {key: float(np.mean([loss[key] for loss in losses])) for key in losses[0]}
