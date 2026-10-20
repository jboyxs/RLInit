"""Raw-joint SAC door task respecting XML joint/actuator target limits.

The baseline adds home +/- pi bounds even to continuous joints. Keep finite
model limits, but remove that extra turn limit on joints declared unlimited.
Action dimensions and increment scaling stay unchanged. Native servos and
physics stay unchanged; only the extra software target cuts are removed.
"""

import numpy as np

from .door_env import DEFAULT_MODEL
from .door_sac_learn_env import LearnRewardConfig, SACLearnDoorEnv, default_door_config


TASK = "fixed_base_door_sac_continuous_v1"


class ContinuousJointSACDoorEnv(SACLearnDoorEnv):
    def __init__(self, model_path=DEFAULT_MODEL, *, config=None, reward_config=None, render_mode=None):
        super().__init__(model_path, config=config, reward_config=reward_config or LearnRewardConfig.guided(),
                         render_mode=render_mode)
        for i, (joint, actuator) in enumerate(zip(self.joints, self.actuators)):
            if not self.model.jnt_limited[joint]:
                self.lower[i], self.upper[i] = -np.inf, np.inf
                if self.model.actuator_ctrllimited[actuator]:
                    self.lower[i], self.upper[i] = self.model.actuator_ctrlrange[actuator]

    def specification(self):
        spec = super().specification()
        spec.update(task=TASK, target_limits_source="XML joint and actuator limits only",
                    joint_target_lower=[float(x) if np.isfinite(x) else None for x in self.lower],
                    joint_target_upper=[float(x) if np.isfinite(x) else None for x in self.upper])
        return spec
