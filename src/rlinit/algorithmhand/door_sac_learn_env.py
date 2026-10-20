"""Standalone raw-eight-action SAC door environment.

All original physical stepping, geometry, rewards and success checks live in
this file. There are no imports from another project environment or trainer.
"""

from dataclasses import asdict, dataclass
import hashlib
from pathlib import Path
import threading
import time

import gymnasium as gym
from gymnasium import spaces
import mujoco
import numpy as np

DEFAULT_MODEL = Path(__file__).resolve().parents[3] / "models" / "door_scene.xml"


ARM_NAMES = tuple(f"joint_{i}" for i in range(1, 8))


@dataclass(frozen=True)
class DoorConfig:
    gamma: float = 0.999
    action_scale: float = 0.03
    max_steps: int = 1000
    frame_skip: int = 10
    initial_noise: float = 0.02
    grip_max: float = 170.0
    success_angle: float = 0.30
    success_hold_seconds: float = 0.30
    contact_threshold: float = 0.05
    failure_penetration: float = 0.005

    def __post_init__(self):
        if not 0 < self.gamma <= 1:
            raise ValueError("gamma must be in (0, 1]")
        if min(self.action_scale, self.success_angle, self.success_hold_seconds,
               self.contact_threshold, self.failure_penetration, self.grip_max) <= 0:
            raise ValueError("Scales, thresholds and grip_max must be positive")
        if self.max_steps < 1 or self.frame_skip < 1 or self.initial_noise < 0:
            raise ValueError("Invalid step count or initial_noise")
        if self.grip_max > 170:
            raise ValueError("First version limits grip commands to the validated 0–170 range")


def fixed_base_model(path: Path) -> mujoco.MjModel:
    """Modify an in-memory spec; original robot and scene XMLs remain reusable."""
    spec = mujoco.MjSpec.from_file(str(path.resolve()))
    for key in spec.keys:
        key.qpos = list(key.qpos)[3:]
        key.ctrl = list(key.ctrl)[3:]
    for name in ("joint_x", "joint_y", "joint_th"):
        spec.delete(spec.actuator(name))
        spec.delete(spec.joint(name))
    # Welding the base changes automatic world/parent collision filtering.
    spec.add_exclude(bodyname1="gen3/base_link", bodyname2="shoulder_link")
    model = spec.compile()
    model.opt.timestep = 0.002
    return model


class DoorEnv(gym.Env):
    """Reach, grasp, depress the handle and pull a passive door through contact.

    State core: all retained qpos/qvel, arm controller targets, previous action,
    two milestone flags, success timer and elapsed episode time. Relative grasp
    pose and fingertip normal forces are additional observation features.
    """

    metadata = {"render_modes": ["human"], "render_fps": 50}

    def __init__(self, model_path=DEFAULT_MODEL, *, config=None, render_mode=None):
        super().__init__()
        if render_mode not in (None, "human"):
            raise ValueError("render_mode must be None or 'human'")
        self.config = config or DoorConfig()
        self.model_path = Path(model_path).resolve()
        self.render_mode = render_mode
        self.model = m = fixed_base_model(self.model_path)
        self.data = d = mujoco.MjData(m)
        self.joints = np.array([m.joint(name).id for name in ARM_NAMES])
        self.qadr = m.jnt_qposadr[self.joints]
        self.vadr = m.jnt_dofadr[self.joints]
        self.actuators = np.array([m.actuator(name).id for name in ARM_NAMES])
        self.gripper = m.actuator("fingers_actuator").id
        self.ee = m.site("pinch_site").id
        self.handle_site = m.site("door_handle_site").id
        self.hinge = int(m.jnt_qposadr[m.joint("door_hinge").id])
        self.latch = int(m.jnt_qposadr[m.joint("door_latch_joint").id])
        self.pad_sides = {m.geom(f"{side}_pad{i}").id: side
                          for side in ("left", "right") for i in (1, 2)}
        self.handle_geom = m.geom("door_handle").id
        self.handle_geoms = {self.handle_geom, m.geom("door_handle_base").id}
        self.floor = m.geom("floor").id
        self.robot_geoms = self._descendant_geoms("base_link")
        self.gripper_geoms = self._descendant_geoms("base")
        self.door_geoms = self._descendant_geoms("door_fixture")
        self.dt = m.opt.timestep * self.config.frame_skip
        if self.config.success_angle >= m.joint("door_hinge").range[1]:
            raise ValueError("success_angle must be below the door joint upper limit")
        mujoco.mj_resetDataKeyframe(m, d, m.key("home").id)
        mujoco.mj_forward(m, d)
        self.home = d.qpos[self.qadr].copy()
        self.lower, self.upper = self.home - np.pi, self.home + np.pi
        for i, (joint, actuator) in enumerate(zip(self.joints, self.actuators)):
            if m.jnt_limited[joint]:
                self.lower[i], self.upper[i] = m.jnt_range[joint]
            if m.actuator_ctrllimited[actuator]:
                lo, hi = m.actuator_ctrlrange[actuator]
                self.lower[i], self.upper[i] = max(self.lower[i], lo), min(self.upper[i], hi)
        roll90 = np.array([[1, 0, 0], [0, 0, -1], [0, 1, 0]])
        ideal_grasp = roll90 @ d.site_xmat[self.ee].reshape(3, 3)
        self.grasp_relative_rotation = d.site_xmat[self.handle_site].reshape(3, 3).T @ ideal_grasp
        # Center along the handle bar: its local X is 0.075 vs the site's 0.125.
        # Keep the existing 1 cm approach offset along local -Y.
        #修改成了门把手把柄的中间
        self.grasp_relative_position = np.array([-0.05, -0.01, 0])
        self.action_space = spaces.Box(-1.0, 1.0, shape=(8,), dtype=np.float32)
        # 34 physical + 7 targets + 8 action history + 4 flags/timers/time
        # + 3 relative position + 6 relative rotation + 2 contact forces = 64.
        self.observation_space = spaces.Box(-np.inf, np.inf,
                                            shape=(m.nq + m.nv + 30,), dtype=np.float32)
        self.arm_target = self.home.copy()
        self.previous_action = np.zeros(8)
        self.grasp_seen = self.press_seen = False
        self.success_time = 0.0
        self.steps = 0
        self.max_door_angle = 0.0
        self._ended = False
        self._viewer = None
        self._viewer_threads = []

    def _descendant_geoms(self, name):
        root_id = self.model.body(name).id
        result = set()
        for geom in range(self.model.ngeom):
            if not (self.model.geom_contype[geom] or self.model.geom_conaffinity[geom]):
                continue
            body = int(self.model.geom_bodyid[geom])
            while body:
                if body == root_id:
                    result.add(geom)
                    break
                body = int(self.model.body_parentid[body])
        return result

    def _contacts(self):
        normal = {"left": 0.0, "right": 0.0}
        bad_force = depth = 0.0
        for index, contact in enumerate(self.data.contact):
            force = np.zeros(6)
            mujoco.mj_contactForce(self.model, self.data, index, force)
            depth = max(depth, float(-contact.dist))
            pair = {int(contact.geom1), int(contact.geom2)}
            if self.handle_geom in pair:
                for geom in pair:
                    if geom in self.pad_sides:
                        normal[self.pad_sides[geom]] += max(0.0, float(force[0]))
            robot = pair & self.robot_geoms
            allowed_grasp = bool(pair & self.gripper_geoms and pair & self.handle_geoms)
            bad = bool(robot and (
                self.floor in pair or len(pair & self.robot_geoms) == 2
                or (pair & self.door_geoms and not allowed_grasp)
            ))
            if bad:
                bad_force += max(0.0, float(force[0]))
        return normal, bad_force, depth

    def _features(self):
        rotation = self.data.site_xmat[self.handle_site].reshape(3, 3)
        target = self.data.site_xpos[self.handle_site] + rotation @ self.grasp_relative_position
        delta = target - self.data.site_xpos[self.ee]
        relative_rotation = (rotation @ self.grasp_relative_rotation).T @ self.data.site_xmat[self.ee].reshape(3, 3)
        alignment = float(np.clip((np.trace(relative_rotation) + 1) / 4, 0, 1))
        normal, bad_force, depth = self._contacts()
        grasp = min(normal.values()) > self.config.contact_threshold
        return delta, relative_rotation, alignment, normal, grasp, bad_force, depth

    def _potential(self, features):
        delta, _, alignment, _, grasp, _, _ = features
        reach = np.exp(-np.linalg.norm(delta) / 0.08)
        press = np.clip(self.data.qpos[self.latch] / 1.3, 0, 1)
        opening = np.clip(self.data.qpos[self.hinge] / self.config.success_angle, 0, 1)
        return float(2 * reach + alignment * reach + 2 * grasp * press + 10 * opening)

    def _press_milestone_threshold(self):
        return 1.1

    def _is_success(self, *, grasp, reason):
        if grasp and self.data.qpos[self.hinge] >= self.config.success_angle and not reason:
            self.success_time = min(self.config.success_hold_seconds, self.success_time + self.dt)
        else:
            self.success_time = 0.0
        return self.success_time >= self.config.success_hold_seconds

    def _reward_components(self, *, features, terminated, potential_before, newly_grasped,
                           newly_pressed, success, reason, peak_bad_force, action):
        return {
            # Discounted potential differences avoid repeatedly paying for a fixed pose.
            "progress": self.config.gamma * (0.0 if terminated else self._potential(features)) - potential_before,
            "grasp_milestone": 2.0 * newly_grasped,
            "press_milestone": 3.0 * newly_pressed,
            "success": 50.0 * success,
            "failure": -20.0 * bool(reason),
            "collision": -0.002 * min(peak_bad_force, 100.0),
            "action": -0.001 * float(np.mean(action[:7] ** 2)),
            "smoothness": -0.0005 * float(np.mean((action - self.previous_action) ** 2)),
            "time": -0.01,
        }

    def _obs(self):
        delta, rotation, _, normal, _, _, _ = self._features()
        qpos = self.data.qpos.copy() / np.pi
        qpos[self.hinge] = self.data.qpos[self.hinge] / 0.4
        qpos[self.latch] = self.data.qpos[self.latch] / 1.3
        result = np.concatenate((
            qpos, self.data.qvel / 2, self.arm_target / np.pi, self.previous_action,
            [self.grasp_seen, self.press_seen,
             self.success_time / self.config.success_hold_seconds, self.steps / self.config.max_steps],
            delta / 0.3, rotation[:, :2].reshape(-1),
            [normal["left"] / 20, normal["right"] / 20],
        ))
        return np.nan_to_num(result, nan=0, posinf=1e6, neginf=-1e6).astype(np.float32)

    def _info(self, *, reason="", success=False, peak_depth=None):
        delta, _, alignment, normal, grasp, bad_force, depth = self._features()
        return {
            "distance": float(np.linalg.norm(delta)), "alignment": alignment,
            "door_angle": float(self.data.qpos[self.hinge]),
            "handle_angle": float(self.data.qpos[self.latch]),
            "max_door_angle": self.max_door_angle, "grasped": bool(grasp),
            "grasp_seen": bool(self.grasp_seen), "press_seen": bool(self.press_seen),
            "left_contact_N": normal["left"], "right_contact_N": normal["right"],
            "bad_contact_N": bad_force, "penetration_m": depth if peak_depth is None else peak_depth,
            "success_hold_seconds": self.success_time,
            "is_success": bool(success), "failure_reason": reason,
        }

    def _apply_controls(self):
        # Bias-torque feedforward goes through existing position actuators and
        # their force limits. Door coordinates/forces are never commanded.
        target = self.arm_target + self.data.qfrc_bias[self.vadr] / self.model.actuator_gainprm[self.actuators, 0]
        for i, actuator in enumerate(self.actuators):
            if self.model.actuator_ctrllimited[actuator]:
                target[i] = np.clip(target[i], *self.model.actuator_ctrlrange[actuator])
        self.data.ctrl[self.actuators] = target

    def reset(self, *, seed=None, options=None):
        super().reset(seed=seed)
        noise = (options or {}).get("initial_noise", self.config.initial_noise)
        if not np.isfinite(noise) or noise < 0:
            raise ValueError("initial_noise must be nonnegative and finite")
        for _ in range(100):
            mujoco.mj_resetDataKeyframe(self.model, self.data, self.model.key("home").id)
            self.data.qpos[self.qadr] = np.clip(self.home + self.np_random.uniform(-noise, noise, 7), self.lower, self.upper)
            mujoco.mj_forward(self.model, self.data)
            if self._contacts()[2] < 1e-6:
                break
        else:
            raise RuntimeError("Could not sample a collision-free initial pose")
        self.arm_target = self.data.qpos[self.qadr].copy()
        self.previous_action = np.zeros(8)
        self.previous_action[-1] = -1  # Initial gripper command is fully open.
        self.grasp_seen = self.press_seen = False
        self.success_time = 0.0
        self.steps = 0
        self.max_door_angle = 0.0
        self._ended = False
        self._apply_controls()
        self.data.ctrl[self.gripper] = 0
        mujoco.mj_forward(self.model, self.data)
        if self.render_mode == "human":
            self.render()
        return self._obs(), self._info()

    def step(self, action):
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
        newly_pressed = grasp and self.data.qpos[self.latch] >= self._press_milestone_threshold() and not self.press_seen
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
        return {"task": "fixed_base_door_v1", "model_path": str(self.model_path),
                "config": asdict(self.config), "observation_dim": self.observation_space.shape[0],
                "action_dim": self.action_space.shape[0]}

    def render(self):
        if self.render_mode != "human":
            return
        if self._viewer is None:
            from mujoco import viewer
            existing = set(threading.enumerate())
            self._viewer = viewer.launch_passive(self.model, self.data)
            self._viewer_threads = [thread for thread in threading.enumerate()
                                    if thread not in existing and thread.daemon]
            self._viewer.cam.lookat[:] = [0.4, 0, 0.65]
            self._viewer.cam.distance, self._viewer.cam.azimuth, self._viewer.cam.elevation = 2.3, 45, -20
        if self._viewer.is_running():
            self._viewer.sync()

    def close(self):
        if self._viewer is not None:
            self._viewer.close()
            for thread in self._viewer_threads:
                thread.join(timeout=5)
            self._viewer_threads = []
            self._viewer = None


# door_direct_env.py: definitions retained in this standalone module.

DIRECT_TASK = "fixed_base_door_direct_reward_v1"


class DirectRewardDoorEnv(DoorEnv):
    """Penalize distance and reward nearby alignment, pressing and opening.

    Like ReferenceRewardDoorEnv, compute reward directly from the current
    state. Penalize the summed upper/lower inner pad distances to the handle
    face centers and reward alignment weighted by exp(-distance / 0.08).
    Distance/orientation weights are configured here; retain DoorEnv's other
    rewards and termination.
    Holding a useful pose earns dense reward repeatedly, including on the
    final step; neither discounting nor terminal potential subtraction applies.
    """

    def __init__(self, model_path=DEFAULT_MODEL, *, config=None, render_mode=None,
                 position_reward="linear_distance", orientation_reward="proximity_alignment",
                 distance_metric="finger_surface_sum", distance_weight=2.5, orientation_weight=2):
        if position_reward not in ("linear_distance", "exponential_reach"):
            raise ValueError("Unknown direct position reward")
        if orientation_reward not in ("proximity_alignment", "orientation_cost", "distance_weighted_alignment"):
            raise ValueError("Unknown direct orientation reward")
        if distance_metric not in ("finger_surface_sum", "grasp_target"):
            raise ValueError("Unknown direct distance metric")
        if not np.isfinite(distance_weight) or distance_weight <= 0:
            raise ValueError("distance_weight must be positive and finite")
        if not np.isfinite(orientation_weight) or orientation_weight <= 0:
            raise ValueError("orientation_weight must be positive and finite")
        # Earlier reward modes are retained to restore pre-change checkpoints.
        self.position_reward = position_reward
        self.orientation_reward = orientation_reward
        self.distance_metric = distance_metric
        self.distance_weight = float(distance_weight)
        self.orientation_weight = float(orientation_weight)
        super().__init__(model_path, config=config, render_mode=render_mode)
        # Under the desired grasp rotation, left is the upper finger and right
        # the lower. Keep their identities fixed as the robot/handle rotate.
        self._finger_pad_geoms = tuple(
            np.array([self.model.geom(f"{side}_pad{i}").id for i in (1, 2)])
            for side in ("left", "right")
        )

    def _surface_distance_points(self):
        """World coordinates of upper/lower inner pad and handle face centers."""
        fingers = []
        for pads in self._finger_pad_geoms:
            rotations = self.data.geom_xmat[pads].reshape(2, 3, 3)
            # Each pad box's inward face is its local -Y face. The two boxes
            # are equal-area adjoining halves of the complete contact pad.
            inner_faces = (self.data.geom_xpos[pads]
                           - rotations[:, :, 1] * self.model.geom_size[pads, 1][:, None])
            fingers.append(np.mean(inner_faces, axis=0))
        center = self.data.geom_xpos[self.handle_geom]
        rotation = self.data.geom_xmat[self.handle_geom].reshape(3, 3)
        half_height = self.model.geom_size[self.handle_geom, 2]
        offset = rotation[:, 2] * half_height
        # Actual surface centers, with no extra approach offset. Local +/-Z
        # follows the handle's hinge/latch motion rather than world height.
        handle_faces = np.array([center + offset, center - offset])
        return np.array(fingers), handle_faces

    def _finger_surface_distances(self):
        fingers, handle_faces = self._surface_distance_points()
        return np.linalg.norm(fingers - handle_faces, axis=1)

    def _potential(self, features):
        # DoorEnv.step requests this before simulation; no potential is needed.
        return 0.0

    def _reward_components(self, *, features, newly_grasped, newly_pressed,
                           success, reason, peak_bad_force, action, **context):
        delta, _, alignment, _, grasp, _, _ = features
        grasp_target_distance = float(np.linalg.norm(delta))
        distance = (float(np.sum(self._finger_surface_distances()))
                    if self.distance_metric == "finger_surface_sum" else grasp_target_distance)
        # Preserve the original alignment term for older checkpoints.
        reach = float(np.exp(-grasp_target_distance / 0.08))
        press = float(np.clip(self.data.qpos[self.latch] / 1.3, 0, 1))
        opening = float(np.clip(self.data.qpos[self.hinge] / self.config.success_angle, 0, 1))
        position = ({"distance": -self.distance_weight * distance} if self.position_reward == "linear_distance"
                    else {"reaching": 2.0 * reach})
        if self.orientation_reward == "proximity_alignment":
            orientation = {"orientation": self.orientation_weight * alignment * float(np.exp(-distance / 0.08))}
        elif self.orientation_reward == "orientation_cost":
            orientation = {"orientation": -self.orientation_weight * (1.0 - alignment)}
        else:
            orientation = {"alignment": alignment * reach}
        return {
            **position,
            **orientation,
            "pressing": 2.0 * grasp * press,
            "opening": 10.0 * opening,
            "grasp_milestone": 2.0 * newly_grasped,
            "press_milestone": 3.0 * newly_pressed,
            "success": 50.0 * success,
            "failure": -20.0 * bool(reason),
            "collision": -0.002 * min(peak_bad_force, 100.0),
            "action": -0.001 * float(np.mean(action[:7] ** 2)),
            "smoothness": -0.0005 * float(np.mean((action - self.previous_action) ** 2)),
            "time": -0.01,
        }

    def _info(self, **kwargs):
        info = super()._info(**kwargs)
        if self.distance_metric == "finger_surface_sum":
            upper, lower = self._finger_surface_distances()
            info.update(grasp_target_distance_m=info["distance"],
                        distance=float(upper + lower),
                        upper_finger_distance_m=float(upper), lower_finger_distance_m=float(lower))
        return info

    def specification(self):
        position_source = (f"-{self.distance_weight:g} * distance ({self.distance_metric})"
                           if self.position_reward == "linear_distance" else "door_env.py:_potential reach")
        if self.orientation_reward == "proximity_alignment":
            orientation_source = f"{self.orientation_weight:g} * alignment * exp(-distance / 0.08) ({self.distance_metric})"
        elif self.orientation_reward == "orientation_cost":
            orientation_source = f"-{self.orientation_weight:g} * (1 - alignment)"
        else:
            orientation_source = "door_env.py:_potential alignment"
        return dict(super().specification(), task=DIRECT_TASK, position_reward=self.position_reward,
                    orientation_reward=self.orientation_reward,
                    distance_metric=self.distance_metric,
                    distance_weight=self.distance_weight,
                    orientation_weight=self.orientation_weight,
                    reward_source=f"{position_source}; {orientation_source}; other door_env.py features/weights; "
                                  "direct current-state reward",
                    success_rule="grasp and door_angle >= success_angle for success_hold_seconds")


# door_sac_open_env.py: definitions retained in this standalone module.

OPEN_TASK = "fixed_base_door_sac_open_v1"


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
    press_target_angle: float = np.pi / 4
    press_milestone_angle: float = np.pi / 4
    ready_distance: float = 0.025
    ready_alignment: float = 0.85
    contact_reference: float = 2.0

    def __post_init__(self):
        for name, value in asdict(self).items():
            if not np.isfinite(value) or value < 0:
                raise ValueError(f"{name} must be finite and nonnegative")
        if min(self.press_target_angle, self.press_milestone_angle) <= 0:
            raise ValueError("Press target and milestone angles must be positive")
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

    def _press_milestone_threshold(self):
        return self.reward_config.press_milestone_angle

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
        press = float(np.clip(self.data.qpos[self.latch] / cfg.press_target_angle, -1, 1))
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
        return dict(task=OPEN_TASK, model_path=str(self.model_path),
                    scene_sha256=hashlib.sha256(self.model_path.read_bytes()).hexdigest(),
                    config=asdict(self.config), reward_config=asdict(self.reward_config),
                    observation_dim=78, action_dim=8,
                    action_source="original DoorEnv joint increments and gripper",
                    initialization="original noisy home only", contact_observation="log")


# door_sac_learn_env.py: definitions retained in this standalone module.

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
