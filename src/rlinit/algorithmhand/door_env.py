"""Fixed-base door-opening MDP with eight normalized robot actions."""

from dataclasses import asdict, dataclass
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
        self.grasp_relative_position = np.array([0, -0.01, 0])
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
        newly_pressed = grasp and self.data.qpos[self.latch] >= 1.1 and not self.press_seen
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
