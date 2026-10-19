"""Fixed full-door SAC environment matching the independently verified recipe.

This module freezes the successful observation/reward design, without the
experimental stages, reset banks or action controllers. Physics and all eight
actions remain those of DoorEnv. Change new-training reward weights here.
"""

from dataclasses import asdict, dataclass
import hashlib

from gymnasium import spaces
import numpy as np

from .door_direct_env import DirectRewardDoorEnv
from .door_env import DEFAULT_MODEL, DoorConfig


TASK = "fixed_base_door_sac_open_v1"


@dataclass(frozen=True)
class OpenRewardConfig:
    distance_weight: float = 2.5
    balance_weight: float = 2.5
    center_weight: float = 1.0
    orientation_weight: float = 0.5
    orientation_cost_weight: float = 0.5
    early_close_weight: float = 0.5
    closure_weight: float = 0.5
    contact_weight: float = 0.5
    grasp_weight: float = 0.5
    press_weight: float = 0.5
    opening_weight: float = 2.0
    success_bonus: float = 1000.0
    failure_penalty: float = 100.0
    grasp_milestone: float = 2.0
    press_milestone: float = 3.0
    ready_distance: float = 0.025
    ready_alignment: float = 0.85
    contact_reference: float = 2.0

    def __post_init__(self):
        for name, value in asdict(self).items():
            if not np.isfinite(value) or value < 0:
                raise ValueError(f"{name} must be finite and nonnegative")
        if min(self.distance_weight, self.orientation_weight, self.ready_distance, self.contact_reference) <= 0:
            raise ValueError("Distance/orientation scales and contact reference must be positive")
        if not 0.05 < self.ready_alignment <= 1:
            raise ValueError("ready_alignment must be in (0.05, 1]")


def default_door_config():
    return DoorConfig(gamma=0.995, action_scale=0.01)


class SACOpenDoorEnv(DirectRewardDoorEnv):
    """78 observations, raw joint/gripper actions, full task, noisy home only."""

    def __init__(self, model_path=DEFAULT_MODEL, *, config=None, reward_config=None, render_mode=None):
        self.reward_config = reward_config or OpenRewardConfig()
        super().__init__(model_path, config=config or default_door_config(), render_mode=render_mode,
                         distance_weight=self.reward_config.distance_weight,
                         orientation_weight=self.reward_config.orientation_weight)
        self.observation_space = spaces.Box(-np.inf, np.inf, shape=(78,), dtype=np.float32)

    def _grasp_geometry(self):
        fingers, faces = self._surface_distance_points()
        rotation = self.data.geom_xmat[self.handle_geom].reshape(3, 3)
        errors = (faces - fingers) @ rotation
        center = np.mean(errors, axis=0)
        distances = np.linalg.norm(errors, axis=1)
        gap = float(np.dot(fingers[0] - fingers[1], rotation[:, 2]))
        return errors, center, distances, gap

    def _readiness(self, center, alignment):
        return float(np.exp(-(np.linalg.norm(center) / self.reward_config.ready_distance) ** 2) * alignment ** 4)

    def _obs(self):
        base = super()._obs()
        base[62:64] = np.log1p(np.maximum(base[62:64], 0))
        errors, center, distances, gap = self._grasp_geometry()
        features = self._features()
        contact = min(1.0, min(features[3].values()) / self.reward_config.contact_reference)
        extra = np.r_[errors.ravel() / 0.1, center / 0.1, distances / 0.1,
                      gap / 0.08, self._readiness(center, features[2]), contact]
        return np.nan_to_num(np.r_[base, extra], nan=0, posinf=1e6, neginf=-1e6).astype(np.float32)

    def _reward_components(self, *, features, newly_grasped, newly_pressed, success,
                           reason, peak_bad_force, action, **context):
        cfg = self.reward_config
        alignment, normal, grasp = features[2:5]
        _, center, distances, _ = self._grasp_geometry()
        distance = float(np.sum(distances))
        center_distance = np.linalg.norm(center)
        contact = min(1.0, min(normal.values()) / cfg.contact_reference)
        quality = float(alignment ** 4 * np.exp(-center_distance / 0.05))
        aligned = float(np.clip((alignment - cfg.ready_alignment + 0.05) / 0.05, 0, 1))
        centered = float(np.clip((2 * cfg.ready_distance - center_distance) / cfg.ready_distance, 0, 1))
        close_readiness = max(self._readiness(center, alignment), contact * alignment ** 4, aligned * centered)
        closure = float((action[7] + 1) / 2)
        press = float(np.clip(self.data.qpos[self.latch] / 1.3, -1, 1))
        opening = float(np.clip(self.data.qpos[self.hinge] / self.config.success_angle, 0, 1))
        return {
            "distance": -cfg.distance_weight * distance,
            "balance": -cfg.balance_weight * float(abs(distances[0] - distances[1])),
            "centering": -cfg.center_weight * float(center_distance),
            "orientation": cfg.orientation_weight * alignment * float(np.exp(-distance / 0.08)),
            "orientation_cost": -cfg.orientation_cost_weight * (1 - alignment),
            "early_close": -cfg.early_close_weight * (1 - close_readiness) * closure,
            "closure": cfg.closure_weight * close_readiness * closure,
            "contact": cfg.contact_weight * contact * quality,
            "grasp": cfg.grasp_weight * float(grasp) * quality,
            "pressing": cfg.press_weight * float(grasp) * press * quality,
            "opening": cfg.opening_weight * opening * float(grasp) * quality,
            "grasp_milestone": cfg.grasp_milestone * newly_grasped * quality,
            "press_milestone": cfg.press_milestone * newly_pressed * quality,
            "success": cfg.success_bonus * success,
            "failure": -cfg.failure_penalty * bool(reason),
            "collision": -0.0005 * min(peak_bad_force, 100.0),
            "action": -0.0005 * float(np.mean(action[:7] ** 2)),
            "smoothness": -0.0002 * float(np.mean((action - self.previous_action) ** 2)),
            "time": -0.005,
        }

    def _info(self, **kwargs):
        info = super()._info(**kwargs)
        _, center, distances, gap = self._grasp_geometry()
        info.update(training_stage="full", reset_origin="home", stage_success=info["is_success"],
                    center_distance_m=float(np.linalg.norm(center)),
                    finger_imbalance_m=float(abs(distances[0] - distances[1])), finger_gap_m=gap,
                    grasp_readiness=self._readiness(center, info["alignment"]),
                    gripper_action=float(self.previous_action[7]))
        return info

    def specification(self):
        return dict(task=TASK, model_path=str(self.model_path),
                    scene_sha256=hashlib.sha256(self.model_path.read_bytes()).hexdigest(),
                    config=asdict(self.config), reward_config=asdict(self.reward_config),
                    observation_dim=78, action_dim=8,
                    action_source="original DoorEnv joint increments and gripper",
                    initialization="original noisy home only", contact_observation="log")
