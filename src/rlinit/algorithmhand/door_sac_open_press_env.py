"""SAC open-door variant changing only the first-press milestone to 1.25 rad.

All continuous reward terms, including pressing / 1.3, come from SACOpenDoorEnv.
The existing press_seen observation/info flag follows the new milestone so the
one-time reward remains observable; dimensions and all other features are kept.
"""

from dataclasses import dataclass
import time

import mujoco
import numpy as np

from .door_env import DEFAULT_MODEL
from .door_sac_open_env import OpenRewardConfig, SACOpenDoorEnv, default_door_config


TASK = "fixed_base_door_sac_open_press_v1"


@dataclass(frozen=True)
class OpenPressRewardConfig(OpenRewardConfig):
    press_milestone_angle: float = 1.25

    def __post_init__(self):
        super().__post_init__()
        if self.press_milestone_angle <= 0:
            raise ValueError("press_milestone_angle must be positive")


class SACOpenPressDoorEnv(SACOpenDoorEnv):
    def __init__(self, model_path=DEFAULT_MODEL, *, config=None, reward_config=None, render_mode=None):
        super().__init__(model_path, config=config, reward_config=reward_config or OpenPressRewardConfig(),
                         render_mode=render_mode)

    def step(self, action):
        # Keep DoorEnv.step identical except for the first-press angle below.
        # A local copy leaves the original environment and recipe untouched.
        if self._ended:
            raise RuntimeError("Call reset() after a terminated/truncated episode")
        action = np.asarray(action, dtype=np.float64)
        if action.shape != (8,) or not np.isfinite(action).all():
            raise ValueError("action must contain eight finite numbers")
        action = np.clip(action, -1, 1)
        start = time.monotonic()
        potential_before = self._potential(self._features())
        self.arm_target = np.clip(self.arm_target + self.config.action_scale * action[:7], self.lower, self.upper)
        self._apply_controls()
        self.data.ctrl[self.gripper] = (action[7] + 1) * 0.5 * self.config.grip_max
        peak_depth = peak_bad_force = 0.0
        reason = ""
        for _ in range(self.config.frame_skip):
            mujoco.mj_step(self.model, self.data)
            mujoco.mj_forward(self.model, self.data)
            if not (np.isfinite(self.data.qpos).all() and np.isfinite(self.data.qvel).all()) or any(w.number for w in self.data.warning):
                reason = "numerical_instability"
                break
            _, bad_force, depth = self._contacts()
            peak_depth, peak_bad_force = max(peak_depth, depth), max(peak_bad_force, bad_force)
            self.max_door_angle = max(self.max_door_angle, float(self.data.qpos[self.hinge]))
            if depth > self.config.failure_penetration:
                reason = "excessive_penetration"
                break
            if bad_force > 300:
                reason = "unsafe_collision"
                break
        self.steps += 1
        features = self._features()
        grasp = features[4]
        newly_grasped = grasp and not self.grasp_seen
        self.grasp_seen |= grasp
        newly_pressed = grasp and self.data.qpos[self.latch] >= self.reward_config.press_milestone_angle and not self.press_seen
        self.press_seen |= newly_pressed
        success = self._is_success(grasp=grasp, reason=reason)
        terminated = bool(success or reason)
        truncated = bool(self.steps >= self.config.max_steps and not terminated)
        components = self._reward_components(
            features=features, terminated=terminated, potential_before=potential_before,
            newly_grasped=newly_grasped, newly_pressed=newly_pressed, success=success,
            reason=reason, peak_bad_force=peak_bad_force, action=action,
        )
        self.previous_action = action.copy()
        self._ended = terminated or truncated
        info = self._info(reason=reason, success=success, peak_depth=peak_depth)
        info["reward_components"] = components
        obs = self._obs()
        reward = float(sum(components.values()))
        if not np.isfinite(reward):
            raise RuntimeError("Non-finite reward")
        if self.render_mode == "human":
            self.render()
            if self._viewer.is_running():
                time.sleep(max(0.0, self.dt - (time.monotonic() - start)))
        return obs, reward, terminated, truncated, info

    def specification(self):
        return dict(super().specification(), task=TASK)
