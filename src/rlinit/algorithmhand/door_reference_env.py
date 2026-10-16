"""Apply reference/door.py's Door.reward to the project's TidyBot physics."""

from dataclasses import asdict, dataclass

import numpy as np

from .door_env import DEFAULT_MODEL, DoorEnv


TASK = "fixed_base_door_reference_reward_v1"


@dataclass(frozen=True)
class ReferenceRewardConfig:
    reward_shaping: bool = True
    reward_scale: float = 1.0

    def __post_init__(self):
        if not isinstance(self.reward_shaping, bool):
            raise ValueError("reward_shaping must be a boolean")
        if not np.isfinite(self.reward_scale) or self.reward_scale <= 0:
            raise ValueError("reward_scale must be positive and finite")


class ReferenceRewardDoorEnv(DoorEnv):
    """Dense reaching + handle rotation, or sparse completion-only reward.

    The supplied reference uses hinge > 0.3 as its success predicate. Retain
    this project's termination on success/failure and its 1000-step horizon,
    while dropping the original grasp-and-hold success requirement. Physical
    collision/numerical failure checks still end episodes, without adding
    penalties to the reference reward. No robosuite dependency is required.
    """

    def __init__(self, model_path=DEFAULT_MODEL, *, config=None, reward_config=None, render_mode=None):
        self.reward_config = reward_config or ReferenceRewardConfig()
        super().__init__(model_path, config=config, render_mode=render_mode)

    def _handle_distance(self):
        return float(np.linalg.norm(self.data.site_xpos[self.handle_site] - self.data.site_xpos[self.ee]))

    def _potential(self, features):
        # This reward is based on current state, not a potential difference.
        return 0.0

    def _is_success(self, *, grasp, reason):
        self.success_time = 0.0
        return bool(not reason and self.data.qpos[self.hinge] > self.config.success_angle)

    def _reward_components(self, *, success, **context):
        reaching = rotating = 0.0
        if not success and self.reward_config.reward_shaping:
            reaching = 0.25 * (1.0 - np.tanh(10.0 * self._handle_distance()))
            # The project's door has a physical spring-loaded latch (use_latch=True).
            rotating = float(np.clip(0.25 * abs(self.data.qpos[self.latch] / (0.5 * np.pi)), -0.25, 0.25))
        scale = self.reward_config.reward_scale
        return {"reaching": scale * float(reaching), "rotating": scale * rotating,
                "success": scale * float(success)}

    def _info(self, **kwargs):
        info = super()._info(**kwargs)
        info["grasp_target_distance_m"] = info["distance"]
        info["distance"] = self._handle_distance()
        info["reward_shaping"] = self.reward_config.reward_shaping
        return info

    def specification(self):
        return dict(super().specification(), task=TASK, reward_config=asdict(self.reward_config),
                    reward_source="reference/door.py:Door.reward (use_latch=True)",
                    success_rule="door_angle > success_angle; no grasp/hold requirement")
