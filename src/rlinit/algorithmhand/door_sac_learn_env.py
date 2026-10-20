"""Home-only SAC door task with direct joint actions and progress rewards.

No IK, planner, scripted policy, demonstration/reset bank or door actuation is
used. DoorEnv owns all physics and the original eight-action robot interface.
"""

from dataclasses import asdict, dataclass
import hashlib

from gymnasium import spaces
import numpy as np

from .door_env import DEFAULT_MODEL
from .door_sac_open_env import OpenRewardConfig, SACOpenDoorEnv, default_door_config


TASK = "fixed_base_door_sac_learn_v1"


@dataclass(frozen=True)
class LearnRewardConfig:
    dense_guidance: bool = False
    press_progress_only: bool = False
    guided_press_weight: float = 3.0
    guided_success_bonus: float = 2000.0
    distance_weight: float = 0.0
    balance_weight: float = 0.0
    center_weight: float = 0.0
    orientation_cost_weight: float = 0.0
    early_close_weight: float = 0.0
    ready_distance: float = 0.025
    contact_reference: float = 2.0
    press_target_angle: float = np.pi / 4
    press_milestone_angle: float = np.pi / 4
    stable_grasp_seconds: float = 0.10
    reach_scale: float = 0.10
    surface_scale: float = 0.06
    reach_potential: float = 6.0
    alignment_potential: float = 4.0
    enclosure_potential: float = 6.0
    contact_potential: float = 6.0
    press_potential: float = 12.0
    opening_potential: float = 25.0
    reach_cost: float = 0.05
    alignment_cost: float = 0.02
    ungrasped_cost: float = 0.02
    unpressed_cost: float = 0.02
    unopened_cost: float = 0.04
    time_cost: float = 0.01
    grasp_milestone: float = 1.0
    press_milestone: float = 3.0
    success_bonus: float = 100.0
    failure_penalty: float = 50.0

    @classmethod
    def linear(cls):
        """Retain a distance gradient outside the local potential's reach."""
        return cls(distance_weight=0.5, balance_weight=0.5, center_weight=0.25,
                   orientation_cost_weight=0.1, early_close_weight=0.1,
                   failure_penalty=100.0)

    @classmethod
    def guided(cls):
        return cls(dense_guidance=True, failure_penalty=100.0)

    def __post_init__(self):
        if self.press_progress_only and not self.dense_guidance:
            raise ValueError("Press-progress mode requires guided rewards")
        for name, value in asdict(self).items():
            if not np.isfinite(value) or value < 0:
                raise ValueError(f"{name} must be finite and nonnegative")
        for name in ("ready_distance", "contact_reference", "press_target_angle", "press_milestone_angle",
                     "stable_grasp_seconds", "reach_scale", "surface_scale", "time_cost"):
            if getattr(self, name) <= 0:
                raise ValueError(f"{name} must be positive")


class SACLearnDoorEnv(SACOpenDoorEnv):
    """80 observed values; unchanged eight joint/gripper actions and physics."""

    def __init__(self, model_path=DEFAULT_MODEL, *, config=None, reward_config=None, render_mode=None):
        self.grasp_hold = 0.0
        self.pressed_with_stable_grasp = False
        super().__init__(model_path, config=config or default_door_config(), render_mode=render_mode)
        self.reward_config = reward_config or LearnRewardConfig.linear()
        upper = self.model.joint("door_latch_joint").range[1]
        if max(self.reward_config.press_target_angle, self.reward_config.press_milestone_angle) > upper:
            raise ValueError("Press target exceeds the physical handle range")
        self.observation_space = spaces.Box(-np.inf, np.inf, shape=(80,), dtype=np.float32)

    def reset(self, *, seed=None, options=None):
        self.grasp_hold = 0.0
        self.pressed_with_stable_grasp = False
        return super().reset(seed=seed, options=options)

    def _task_features(self, features):
        cfg = self.reward_config
        alignment, normal, grasp = features[2:5]
        _, center, distances, _ = self._grasp_geometry()
        center_distance = float(np.linalg.norm(center))
        reach = float(np.exp(-center_distance / cfg.reach_scale))
        quality = float(alignment ** 4 * np.exp(-center_distance / 0.05))
        enclosure = float(np.exp(-float(np.sum(distances)) / cfg.surface_scale) * alignment ** 4)
        contact = float(np.clip(min(normal.values()) / cfg.contact_reference, 0, 1)) * quality
        stable = min(self.grasp_hold / cfg.stable_grasp_seconds, 1.0)
        press = float(np.clip(self.data.qpos[self.latch] / cfg.press_target_angle, 0, 1))
        pressed = float(grasp) * stable * quality * press
        opening = float(np.clip(self.data.qpos[self.hinge] / self.config.success_angle, 0, 1))
        readiness = max(self._readiness(center, alignment), contact)
        return dict(reach=reach, alignment=alignment, quality=quality, enclosure=enclosure,
                    contact=contact, pressed=pressed, opening=opening,
                    distance=float(np.sum(distances)), center_distance=center_distance,
                    imbalance=float(abs(distances[0] - distances[1])), readiness=readiness)

    def _potential(self, features):
        if self.reward_config.dense_guidance:
            if self.reward_config.press_progress_only:
                fraction = min(self.grasp_hold / self.reward_config.stable_grasp_seconds, 1.)
                press = float(np.clip(self.data.qpos[self.latch] / self.reward_config.press_target_angle, 0, 1))
                return self.reward_config.press_potential * (1. if self.pressed_with_stable_grasp else fraction * press)
            return 0.0
        cfg, f = self.reward_config, self._task_features(features)
        return float(cfg.reach_potential * f["reach"]
                     + cfg.alignment_potential * f["reach"] * f["alignment"] ** 2
                     + cfg.enclosure_potential * f["enclosure"]
                     + cfg.contact_potential * f["contact"]
                     + cfg.press_potential * f["pressed"]
                     + cfg.opening_potential * f["opening"] * self.pressed_with_stable_grasp)

    def _is_success(self, *, grasp, reason):
        cfg = self.reward_config
        self.grasp_hold = min(cfg.stable_grasp_seconds, self.grasp_hold + self.dt) if grasp and not reason else 0.0
        if self.grasp_hold >= cfg.stable_grasp_seconds and self.data.qpos[self.latch] >= cfg.press_target_angle:
            self.pressed_with_stable_grasp = True
        return super()._is_success(grasp=grasp and self.pressed_with_stable_grasp, reason=reason)

    def _reward_components(self, *, features, terminated, potential_before, newly_grasped,
                           newly_pressed, success, reason, peak_bad_force, action, **context):
        cfg, f = self.reward_config, self._task_features(features)
        if cfg.dense_guidance:
            return self._guided_components(features=features, f=f, newly_grasped=newly_grasped,
                newly_pressed=newly_pressed, success=success, reason=reason,
                peak_bad_force=peak_bad_force, action=action,
                terminated=terminated, potential_before=potential_before)
        return {
            "progress": self.config.gamma * (0.0 if terminated else self._potential(features)) - potential_before,
            "distance": -cfg.distance_weight * f["distance"],
            "balance": -cfg.balance_weight * f["imbalance"],
            "centering": -cfg.center_weight * f["center_distance"],
            "orientation_cost": -cfg.orientation_cost_weight * (1 - f["alignment"]),
            "early_close": -cfg.early_close_weight * (1 - f["readiness"]) * (action[7] + 1) / 2,
            "reach_cost": -cfg.reach_cost * (1 - f["reach"]),
            "alignment_cost": -cfg.alignment_cost * f["reach"] * (1 - f["alignment"]),
            "ungrasped_cost": -cfg.ungrasped_cost * (1 - f["contact"]),
            "unpressed_cost": -cfg.unpressed_cost * (1 - f["pressed"]),
            "unopened_cost": -cfg.unopened_cost * (1 - f["opening"]),
            "grasp_milestone": cfg.grasp_milestone * newly_grasped * f["quality"],
            "press_milestone": cfg.press_milestone * newly_pressed * f["quality"],
            "success": cfg.success_bonus * success,
            "failure": -cfg.failure_penalty * bool(reason),
            "collision": -0.0005 * min(peak_bad_force, 100.0),
            "action": -0.0001 * float(np.mean(action[:7] ** 2)),
            "smoothness": -0.0001 * float(np.mean((action - self.previous_action) ** 2)),
            "time": -cfg.time_cost,
        }

    def _guided_components(self, *, features, f, newly_grasped, newly_pressed, success,
                           reason, peak_bad_force, action, terminated, potential_before):
        # Preserve the original reaching/grasping signals. Post-grasp rotation
        # uses real bilateral contact and duration, rather than A**4 quality.
        base, cfg = OpenRewardConfig(), self.reward_config
        alignment, normal, grasp = features[2:5]
        _, center, _, _ = self._grasp_geometry()
        contact = float(np.clip(min(normal.values()) / base.contact_reference, 0, 1))
        aligned = float(np.clip((alignment - base.ready_alignment + .05) / .05, 0, 1))
        centered = float(np.clip((2 * base.ready_distance - f["center_distance"]) / base.ready_distance, 0, 1))
        ready = max(self._readiness(center, alignment), contact * alignment ** 4, aligned * centered)
        closure = float((action[7] + 1) / 2)
        press = float(np.clip(self.data.qpos[self.latch] / cfg.press_target_angle, 0, 1))
        stable = min(self.grasp_hold / cfg.stable_grasp_seconds, 1.0)
        parts = {
            "distance": -base.distance_weight * f["distance"],
            "balance": -base.balance_weight * f["imbalance"],
            "centering": -base.center_weight * f["center_distance"],
            "orientation": base.orientation_weight * alignment * float(np.exp(-f["distance"] / .08)),
            "orientation_cost": -base.orientation_cost_weight * (1 - alignment),
            "early_close": -base.early_close_weight * (1 - ready) * closure,
            "closure": base.closure_weight * ready * closure,
            "contact": base.contact_weight * contact * f["quality"],
            "grasp": base.grasp_weight * float(grasp) * f["quality"],
            "pressing": 0. if cfg.press_progress_only else cfg.guided_press_weight * float(grasp) * stable * press,
            "opening": base.opening_weight * f["opening"] * float(grasp) * self.pressed_with_stable_grasp,
            "grasp_milestone": base.grasp_milestone * newly_grasped * f["quality"],
            "press_milestone": cfg.press_milestone * newly_pressed,
            "success": cfg.guided_success_bonus * success,
            "failure": -cfg.failure_penalty * bool(reason),
            "collision": -.0005 * min(peak_bad_force, 100.),
            "action": -.0005 * float(np.mean(action[:7] ** 2)),
            "smoothness": -.0002 * float(np.mean((action - self.previous_action) ** 2)),
            "time": -.005,
        }
        # Remove the demonstrated free-pose reward loophole while retaining
        # the original negative distance/alignment/early-closure gradients.
        parts["noncontact_cap"] = min(0., -.01 - sum(parts.values())) if not grasp else 0.
        if cfg.press_progress_only:
            # Outside the cap: this is a discounted potential difference, with
            # no continuing payment for staying pressed or cycling the handle.
            parts["press_progress"] = self.config.gamma * (0. if terminated else self._potential(features)) - potential_before
        return parts

    def _obs(self):
        base = super()._obs()
        extra = [min(self.grasp_hold / self.reward_config.stable_grasp_seconds, 1.0),
                 float(self.pressed_with_stable_grasp)]
        return np.r_[base, extra].astype(np.float32)

    def _info(self, **kwargs):
        info = super()._info(**kwargs)
        info.update(stable_grasp_seconds=self.grasp_hold,
                    pressed_with_stable_grasp=self.pressed_with_stable_grasp,
                    reward_potential=self._potential(self._features()))
        return info

    def specification(self):
        spec = super().specification()
        # Include imported XMLs: changing a fixture must invalidate provenance.
        sources = [self.model_path, self.model_path.parent / "objects/door_lock_fixture.xml",
                   self.model_path.parent / "stanford_tidybot/tidybot.xml"]
        spec.update(task=TASK, observation_dim=80,
                    model_source_sha256={str(p.relative_to(self.model_path.parent)): hashlib.sha256(p.read_bytes()).hexdigest()
                                         for p in sources if p.is_file()},
                    success_rule="home -> stable bilateral grasp -> >=45deg handle depression -> door>=0.30rad with grasp for 0.30s",
                    reset_curriculum=None, demonstrations=False, scripted_actions=False)
        if self.reward_config.dense_guidance:
            spec["guidance_base_reward"] = asdict(OpenRewardConfig())
        return spec
