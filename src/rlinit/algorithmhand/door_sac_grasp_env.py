"""Online SAC door task with explicit fingertip geometry and bounded shaping.

No scripted policy or demonstrations are used. Evaluation begins with
DoorEnv's noisy home reset. Optional training resets replay the learner's
own previously visited physical states. The policy directly
commands seven joint increments and the gripper, using DoorEnv's action mapping.
"""

from dataclasses import asdict, dataclass

from gymnasium import spaces
import numpy as np

from .door_direct_env import DirectRewardDoorEnv
from .door_env import DEFAULT_MODEL
from .door_sac_state_bank import AdaptiveProgressStates, OnlineStateBank, save_bank


TASK = "fixed_base_door_precision_joint_sac_v2"
LEGACY_TASK = "fixed_base_door_precision_sac_v1"


@dataclass(frozen=True)
class GraspTrainingConfig:
    distance_weight: float = 0.5
    balance_weight: float = 0.5
    center_weight: float = 1.0
    orientation_weight: float = 0.05
    orientation_cost_weight: float = 0.0
    early_close_weight: float = 0.10
    closure_weight: float = 0.0
    contact_weight: float = 0.10
    grasp_weight: float = 0.10
    press_weight: float = 0.10
    opening_weight: float = 0.15
    success_bonus: float = 100.0
    failure_penalty: float = 200.0
    ready_distance: float = 0.025
    contact_reference: float = 2.0
    contact_observation: str = "linear"
    grasp_milestone: float = 0.5
    press_milestone: float = 1.0
    training_stage: str = "full"
    signed_press_reward: bool = False
    contact_releases_close_cost: bool = False
    quality_gated_reward: bool = False
    ready_alignment: float | None = None
    reward_scale: float = 1.0
    target_error_weight: float = 0.0
    penetration_weight: float = 0.0
    overpress_weight: float = 0.0

    @classmethod
    def strong(cls, **overrides):
        """Preserve clear alignment/contact signals relative to geometric costs."""
        values = dict(distance_weight=2.5, balance_weight=2.5, orientation_weight=0.5,
                      early_close_weight=0.5, contact_weight=0.5, grasp_weight=0.5,
                      press_weight=0.5, opening_weight=2.0, success_bonus=1000.0,
                      failure_penalty=100.0, grasp_milestone=2.0, press_milestone=3.0)
        values["signed_press_reward"] = True
        values["contact_releases_close_cost"] = True
        values["quality_gated_reward"] = True
        values["ready_alignment"] = 0.85
        if overrides.get("training_stage") == "press":
            values["press_weight"] = 2.0
        values.update(overrides)
        return cls(**values)

    @classmethod
    def opening(cls, **overrides):
        values = dict(press_weight=2.0, opening_weight=10.0, success_bonus=3000.0, reward_scale=0.1)
        values.update(overrides)
        return cls.strong(**values)

    @classmethod
    def joint(cls, **overrides):
        """Give the joint policy a dense incentive to close after alignment.

        This only scores its chosen action; it never changes that action.
        """
        values = dict(closure_weight=0.5)
        values.update(overrides)
        return cls.strong(**values)

    @classmethod
    def safe_opening(cls, **overrides):
        """Score contact precursors to failure without filtering policy actions."""
        values = dict(target_error_weight=2.0, penetration_weight=2.0,
                      overpress_weight=0.5)
        values.update(overrides)
        return cls.joint(**values)

    def __post_init__(self):
        for name, value in asdict(self).items():
            if name == "contact_observation":
                if value not in ("linear", "log"):
                    raise ValueError("contact_observation must be linear or log")
                continue
            if name == "ready_alignment":
                if value is not None and (not np.isfinite(value) or not 0.05 < value <= 1):
                    raise ValueError("ready_alignment must be None or in (0.05, 1]")
                continue
            if name == "training_stage":
                if value not in ("full", "reach", "grasp", "press"):
                    raise ValueError("training_stage must be full, reach, grasp or press")
                continue
            if name in ("signed_press_reward", "contact_releases_close_cost", "quality_gated_reward"):
                if not isinstance(value, bool):
                    raise ValueError(f"{name} must be a bool")
                continue
            if name in ("orientation_cost_weight", "closure_weight", "target_error_weight",
                        "penetration_weight", "overpress_weight"):
                if not np.isfinite(value) or value < 0:
                    raise ValueError(f"{name} must be nonnegative and finite")
                continue
            if not np.isfinite(value) or value <= 0:
                raise ValueError(f"{name} must be positive and finite")


class PrecisionGraspDoorEnv(DirectRewardDoorEnv):
    """The policy controls all seven arm joints and the gripper, as before.

    Fourteen additional observations expose both fingertip errors, centering,
    gap, readiness and balanced contact. Physics, noisy home starts and the
    original bilateral grasp + opening + hold success predicate are retained.
    """

    def __init__(self, model_path=DEFAULT_MODEL, *, config=None, training_config=None, render_mode=None):
        self.training_config = training_config or GraspTrainingConfig()
        super().__init__(model_path, config=config, render_mode=render_mode,
                         distance_weight=self.training_config.distance_weight,
                         orientation_weight=self.training_config.orientation_weight)
        self.observation_space = spaces.Box(-np.inf, np.inf, shape=(78,), dtype=np.float32)
        self._training_state_bank = None
        self._training_reset_probability = 0.0
        self._adaptive_training_states = None
        self._reset_origin = "home"

    def enable_training_resets(self, path, probability, adaptive=False):
        if not np.isfinite(probability) or not 0 < probability <= 1:
            raise ValueError("training reset probability must be in (0, 1]")
        self._training_state_bank = OnlineStateBank(path, self, expected_task=TASK)
        self._training_reset_probability = probability
        self._adaptive_training_states = AdaptiveProgressStates() if adaptive else None
        if self._adaptive_training_states is not None:
            self._adaptive_training_states.seed_from_bank(self, self._training_state_bank)

    def training_reset_specification(self):
        if self._training_state_bank is None:
            return None
        specification = self._training_state_bank.specification(self._training_reset_probability)
        progress = self._adaptive_training_states
        specification["adaptive"] = dict(enabled=progress is not None,
                                         states=len(progress.states) if progress is not None else 0,
                                         best_door_angle=progress.best_angle if progress is not None else 0.0,
                                         covered_angle_bins=sorted({int(value / 0.02) for value in progress.angles})
                                         if progress is not None else [])
        return specification

    def save_adaptive_training_states(self, path, checkpoint):
        progress = self._adaptive_training_states
        if progress is None or not progress.states:
            return None
        save_bank(path, self, progress.states, checkpoint=checkpoint,
                  collection=dict(kind="own_online_training_progress",
                                  initial_bank=str(self._training_state_bank.path),
                                  best_door_angle=progress.best_angle))
        return str(path)

    def reset(self, *, seed=None, options=None):
        self._reset_origin = "home"
        observation, info = super().reset(seed=seed, options=options)
        bank = self._training_state_bank
        if bank is not None and self.np_random.random() < self._training_reset_probability:
            progress = self._adaptive_training_states
            if progress is not None and progress.states and self.np_random.random() < 0.5:
                progress.restore(self, self.np_random)
                self._reset_origin = "own_online_sac_progress"
            else:
                bank.restore(self, int(self.np_random.integers(bank.size)))
                self._reset_origin = "own_online_sac_state"
            observation, info = self._obs(), self._info()
        return observation, info

    def step(self, action):
        result = super().step(action)
        if self._adaptive_training_states is not None:
            _, _, terminated, truncated, info = result
            self._adaptive_training_states.record(self, info, terminated=terminated, truncated=truncated)
        return result

    def _grasp_geometry(self):
        fingers, faces = self._surface_distance_points()
        rotation = self.data.geom_xmat[self.handle_geom].reshape(3, 3)
        errors = (faces - fingers) @ rotation
        center_error = np.mean(errors, axis=0)
        distances = np.linalg.norm(errors, axis=1)
        gap = float(np.dot(fingers[0] - fingers[1], rotation[:, 2]))
        return errors, center_error, distances, gap

    def _readiness(self, center_error, alignment):
        return float(np.exp(-(np.linalg.norm(center_error) / self.training_config.ready_distance) ** 2)
                     * alignment ** 4)

    def _obs(self):
        base = super()._obs()
        if self.training_config.contact_observation == "log":
            base[62:64] = np.log1p(np.maximum(base[62:64], 0))
        errors, center, distances, gap = self._grasp_geometry()
        features = self._features()
        normal, alignment = features[3], features[2]
        balanced_contact = min(1.0, min(normal.values()) / self.training_config.contact_reference)
        extra = np.r_[errors.ravel() / 0.1, center / 0.1,
                      distances / 0.1, gap / 0.08, self._readiness(center, alignment), balanced_contact]
        return np.nan_to_num(np.r_[base, extra], nan=0, posinf=1e6, neginf=-1e6).astype(np.float32)

    def _reward_components(self, *, features, newly_grasped, newly_pressed, success,
                           reason, peak_bad_force, action, **context):
        cfg = self.training_config
        _, _, alignment, normal, grasp, _, _ = features
        _, center, distances, _ = self._grasp_geometry()
        distance = float(np.sum(distances))
        readiness = self._readiness(center, alignment)
        contact = min(1.0, min(normal.values()) / cfg.contact_reference)
        quality = float(alignment ** 4 * np.exp(-np.linalg.norm(center) / 0.05)) if cfg.quality_gated_reward else 1.0
        contact_readiness = contact * alignment ** 4 if cfg.quality_gated_reward else contact
        close_readiness = max(readiness, contact_readiness) if cfg.contact_releases_close_cost else readiness
        if cfg.ready_alignment is not None:
            aligned = float(np.clip((alignment - cfg.ready_alignment + 0.05) / 0.05, 0, 1))
            centered = float(np.clip((2 * cfg.ready_distance - np.linalg.norm(center)) / cfg.ready_distance, 0, 1))
            close_readiness = max(close_readiness, aligned * centered)
        phase_allows_progress = cfg.training_stage != "grasp" or not cfg.quality_gated_reward
        press = float(np.clip(self.data.qpos[self.latch] / 1.3,
                              -1 if cfg.signed_press_reward else 0, 1))
        opening = float(np.clip(self.data.qpos[self.hinge] / self.config.success_angle, 0, 1))
        common = {
            "success": cfg.success_bonus * success,
            "failure": -cfg.failure_penalty * bool(reason),
            "collision": -0.0005 * min(peak_bad_force, 100.0),
            "action": -0.0005 * float(np.mean(action[:7] ** 2)),
            "smoothness": -0.0002 * float(np.mean((action - self.previous_action) ** 2)),
            "time": -0.005,
        }
        # Soft costs begin before the original hard failures. They only change
        # rewards: joint targets, actuator commands and termination stay intact.
        if cfg.target_error_weight:
            error = float(np.max(np.abs(self.arm_target - self.data.qpos[self.qadr])))
            common["target_error"] = -cfg.target_error_weight * float(np.clip((error - 0.05) / 0.15, 0, 1))
        if cfg.penetration_weight:
            depth = features[6]
            common["penetration"] = -cfg.penetration_weight * float(np.clip(
                (depth / self.config.failure_penetration - 0.2) / 0.8, 0, 1))
        if cfg.overpress_weight:
            common["overpress"] = -cfg.overpress_weight * float(np.clip(
                (self.data.qpos[self.latch] - 1.2) / 0.2, 0, 2))
        if cfg.training_stage == "reach":
            # A non-contact pose goal gives SAC a smooth precision objective
            # before learning closure, pressure and pulling through contact.
            angle = float(np.arccos(np.clip(2 * alignment - 1, -1, 1)))
            components = {
                "centering": -cfg.distance_weight * float(np.linalg.norm(center)),
                "orientation": -cfg.orientation_weight * angle / np.pi,
                "early_close": -cfg.early_close_weight * float((action[7] + 1) / 2),
                **common,
            }
            return {name: value * cfg.reward_scale for name, value in components.items()}
        components = {
            "distance": -cfg.distance_weight * distance,
            "balance": -cfg.balance_weight * float(abs(distances[0] - distances[1])),
            "centering": -cfg.center_weight * float(np.linalg.norm(center)),
            "orientation": cfg.orientation_weight * alignment * float(np.exp(-distance / 0.08)),
            "orientation_cost": -cfg.orientation_cost_weight * (1 - alignment),
            "early_close": -cfg.early_close_weight * (1 - close_readiness) * float((action[7] + 1) / 2),
            "closure": cfg.closure_weight * close_readiness * float((action[7] + 1) / 2),
            "contact": cfg.contact_weight * contact * quality,
            "grasp": cfg.grasp_weight * float(grasp) * quality,
            "pressing": (cfg.press_weight * float(grasp) * press * quality
                         if phase_allows_progress else 0.0),
            "opening": (cfg.opening_weight * opening * (float(grasp) * quality if cfg.quality_gated_reward else 1)
                        if phase_allows_progress else 0.0),
            "grasp_milestone": cfg.grasp_milestone * newly_grasped * quality,
            "press_milestone": (cfg.press_milestone * newly_pressed * quality
                                if phase_allows_progress else 0.0),
            **common,
        }
        return {name: value * cfg.reward_scale for name, value in components.items()}

    def _is_success(self, *, grasp, reason):
        stage = self.training_config.training_stage
        if stage == "full":
            return super()._is_success(grasp=grasp, reason=reason)
        _, center, distances, gap = self._grasp_geometry()
        if stage == "reach":
            condition = (float(np.linalg.norm(center)) <= 0.02 and self._features()[2] >= 0.98
                         and gap >= 0.06 and self.data.ctrl[self.gripper] <= 0.1 * self.config.grip_max)
            hold = 0.1
        elif stage == "grasp":
            condition = (grasp and self._features()[2] >= 0.9
                         and float(np.max(distances)) <= 0.03)
            hold = 0.2
        else:
            condition = grasp and self.data.qpos[self.latch] >= 1.1
            if self.training_config.quality_gated_reward:
                condition = condition and self._features()[2] >= 0.85 and float(np.max(distances)) <= 0.04
            hold = 0.1
        if condition and not reason:
            self.success_time = min(hold, self.success_time + self.dt)
        else:
            self.success_time = 0.0
        return self.success_time >= hold

    def _info(self, **kwargs):
        info = super()._info(**kwargs)
        info["training_stage"] = self.training_config.training_stage
        info["reset_origin"] = self._reset_origin
        info["stage_success"] = info["is_success"]
        if self.training_config.training_stage != "full":
            # Completing a training subtask is never reported as opening a door.
            info["is_success"] = False
        _, center, distances, gap = self._grasp_geometry()
        info.update(center_distance_m=float(np.linalg.norm(center)),
                    finger_imbalance_m=float(abs(distances[0] - distances[1])),
                    finger_gap_m=gap, grasp_readiness=self._readiness(center, info["alignment"]),
                    gripper_action=float(self.previous_action[7]))
        return info

    def specification(self):
        specification = dict(super().specification(), task=TASK, training_config=asdict(self.training_config),
                    action_source="policy joint increments and gripper; unchanged DoorEnv mapping",
                    observation_dim=78, reward_source="bounded online SAC precision-grasp shaping",
                    initialization="noisy home; no demonstrations or scripted training actions")
        if self.training_config.training_stage != "full":
            specification["success_rule"] = f"training subtask {self.training_config.training_stage}; not full door opening"
        return specification
