"""保留旧小批次实验入口和检查点版本名，更新逻辑共用 PPOhand。

每轮打乱完整 rollout，每批默认 64 条，更新 10 轮。
原版现在也采用同样的小批次更新，两者训练实现一致。
"""

from .PPOhand import PPO as BasePPO


class PPO(BasePPO):
    variant = "minibatch"
