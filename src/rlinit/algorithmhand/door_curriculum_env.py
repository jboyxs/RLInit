"""A separate door-opening curriculum with physical reset states and dense costs."""

from dataclasses import asdict, dataclass, replace

import mujoco
import numpy as np
from gymnasium import spaces

from .door_env import DEFAULT_MODEL, DoorConfig, DoorEnv


STAGES = ("reach", "grasp", "press", "open", "full")


@dataclass(frozen=True)
class CurriculumConfig:
    pregrasp_offset: float = 0.05
    reach_tolerance: float = 0.02
    alignment_threshold: float = 0.98
    subtask_hold_seconds: float = 0.10
    failure_penalty: float = 50.0
    stage_max_steps: tuple = (250, 350, 400, 500, 1000)

    def __post_init__(self):
        object.__setattr__(self, "stage_max_steps", tuple(self.stage_max_steps))
        if min(self.pregrasp_offset, self.reach_tolerance, self.subtask_hold_seconds, self.failure_penalty) <= 0:
            raise ValueError("Curriculum distances, hold duration and failure penalty must be positive")
        if not 0 < self.alignment_threshold <= 1:
            raise ValueError("alignment_threshold must be in (0, 1]")
        if len(self.stage_max_steps) != len(STAGES) or any(n < 1 for n in self.stage_max_steps):
            raise ValueError("Provide five positive stage time limits")


class _ResetStateBuilder:
    """Drive only the robot offline to obtain feasible, contact-consistent starts.

    This controller is never called by the policy's step(). Its reference data
    contains hypothetical door poses for IK; the physical reset-state simulation
    leaves the door unactuated throughout.
    """

    def __init__(self, env):
        self.env = env
        self.model = m = env.model
        self.data = mujoco.MjData(m)
        self.ik = mujoco.MjData(m)
        self.reference = mujoco.MjData(m)
        mujoco.mj_resetDataKeyframe(m, self.data, m.key("home").id)
        mujoco.mj_forward(m, self.data)
        self.target = self.data.qpos[env.qadr].copy()
        self.jacp = np.zeros((3, m.nv))
        self.jacr = np.zeros((3, m.nv))

    def pose(self, angle=0.0, door_angle=0.0):
        e, m, ref = self.env, self.model, self.reference
        ref.qpos[:] = m.key("home").qpos
        ref.qpos[e.hinge] = door_angle
        ref.qpos[e.latch] = angle
        mujoco.mj_forward(m, ref)
        rotation = ref.site_xmat[e.handle_site].reshape(3, 3)
        return (ref.site_xpos[e.handle_site] + rotation @ e.grasp_relative_position,
                rotation @ e.grasp_relative_rotation)

    def tick(self, position, rotation, grip):
        e, m, d, ik = self.env, self.model, self.data, self.ik
        ik.qpos[:] = d.qpos
        ik.qpos[e.qadr] = self.target
        for _ in range(12):
            mujoco.mj_forward(m, ik)
            current = ik.site_xmat[e.ee].reshape(3, 3)
            ep = position - ik.site_xpos[e.ee]
            er = 0.5 * sum(np.cross(current[:, k], rotation[:, k]) for k in range(3))
            if np.linalg.norm(ep) < 1e-4 and np.linalg.norm(er) < 1e-3:
                break
            mujoco.mj_jacSite(m, ik, self.jacp, self.jacr, e.ee)
            jac = np.vstack((self.jacp[:, e.vadr], 0.3 * self.jacr[:, e.vadr]))
            delta = jac.T @ np.linalg.solve(jac @ jac.T + 1e-5 * np.eye(6), np.r_[ep, 0.3 * er])
            ik.qpos[e.qadr] = np.clip(ik.qpos[e.qadr] + np.clip(delta, -0.04, 0.04), e.lower, e.upper)
        self.target = ik.qpos[e.qadr].copy()
        controls = self.target + d.qfrc_bias[e.vadr] / m.actuator_gainprm[e.actuators, 0]
        for i, actuator in enumerate(e.actuators):
            if m.actuator_ctrllimited[actuator]:
                controls[i] = np.clip(controls[i], *m.actuator_ctrlrange[actuator])
        d.ctrl[e.actuators] = controls
        d.ctrl[e.gripper] = grip
        for _ in range(e.config.frame_skip):
            mujoco.mj_step(m, d)
            mujoco.mj_forward(m, d)
            if (not np.isfinite(d.qpos).all() or not np.isfinite(d.qvel).all()
                    or any(w.number for w in d.warning)
                    or any(c.dist < -0.0015 for c in d.contact)):
                raise RuntimeError("Could not generate a stable physical curriculum start")

    def segment(self, seconds, trajectory, grip):
        count = round(seconds / self.env.dt)
        for i in range(count):
            t = (i + 1) / count
            t = t * t * (3 - 2 * t)
            self.tick(*trajectory(t), grip)

    def snapshot(self, grip):
        copy = mujoco.MjData(self.model)
        mujoco.mj_copyData(copy, self.model, self.data)
        previous = np.r_[np.zeros(7), 2 * grip / self.env.config.grip_max - 1]
        return copy, self.target.copy(), previous

    def build(self):
        e, d = self.env, self.data
        initial_position = d.site_xpos[e.ee].copy()
        initial_rotation = d.site_xmat[e.ee].reshape(3, 3).copy()
        grasp_position, grasp_rotation = self.pose()
        approach_axis = d.site_xmat[e.handle_site].reshape(3, 3)[:, 1].copy()
        pregrasp = grasp_position - e.curriculum.pregrasp_offset * approach_axis
        easy_start = pregrasp - 0.04 * approach_axis

        def roll(t):
            c, s = np.cos(t * np.pi / 2), np.sin(t * np.pi / 2)
            return initial_position, np.array([[1, 0, 0], [0, c, -s], [0, s, c]]) @ initial_rotation

        self.segment(2, roll, 0)
        self.segment(1, lambda t: (initial_position + t * (easy_start - initial_position), grasp_rotation), 0)
        anchors = {"aligned": self.snapshot(0)}
        self.segment(1, lambda t: (easy_start + t * (pregrasp - easy_start), grasp_rotation), 0)
        anchors["pregrasp"] = self.snapshot(0)
        self.segment(1, lambda t: (pregrasp + t * (grasp_position - pregrasp), grasp_rotation), 0)
        self.segment(2, lambda t: (grasp_position, grasp_rotation), e.config.grip_max)
        anchors["grasped"] = self.snapshot(e.config.grip_max)
        self.segment(3, lambda t: self.pose(1.3 * t), e.config.grip_max)
        anchors["pressed"] = self.snapshot(e.config.grip_max)
        return anchors


class DoorCurriculumEnv(DoorEnv):
    """Five subtasks, 70 observations and the same eight robot actions.

    The six extra observations identify the stage and home-start probability.
    Later stages start from real scripted contact states; full/home=1 starts
    entirely from home and has the original complete door-opening success goal.
    """

    def __init__(self, model_path=DEFAULT_MODEL, *, config=None, curriculum=None,
                 stage="reach", home_probability=0.0, render_mode=None):
        self.curriculum = curriculum or CurriculumConfig()
        self.base_config = config or DoorConfig()
        self.stage = stage
        self.home_probability = home_probability
        self._anchors = None
        self._stage_hold = 0.0
        self._episode_start = "home"
        self._needs_reset = True
        super().__init__(model_path, config=self.base_config, render_mode=render_mode)
        self.observation_space = spaces.Box(-np.inf, np.inf,
                                            shape=(self.observation_space.shape[0] + 6,), dtype=np.float32)
        self.set_stage(stage, home_probability)

    def set_stage(self, stage, home_probability=0.0):
        if stage not in STAGES or not 0 <= home_probability <= 1:
            raise ValueError("Unknown stage or home_probability outside [0, 1]")
        self.stage = stage
        self.home_probability = float(home_probability)
        self.config = replace(self.base_config, max_steps=min(
            self.base_config.max_steps, self.curriculum.stage_max_steps[STAGES.index(stage)]))
        self._needs_reset = True

    def _features(self):
        result = list(super()._features())
        if self.stage == "reach":
            axis = self.data.site_xmat[self.handle_site].reshape(3, 3)[:, 1]
            result[0] = result[0] - self.curriculum.pregrasp_offset * axis
        return tuple(result)

    def _obs(self):
        stage = np.zeros(len(STAGES))
        stage[STAGES.index(self.stage)] = 1
        return np.r_[super()._obs(), stage, self.home_probability].astype(np.float32)

    def _goal_reached(self, features):
        delta, _, alignment, _, grasp, _, _ = features
        if self.stage == "reach":
            return np.linalg.norm(delta) <= self.curriculum.reach_tolerance and alignment >= self.curriculum.alignment_threshold
        if self.stage == "grasp":
            return grasp
        if self.stage == "press":
            return grasp and self.data.qpos[self.latch] >= 1.1
        return grasp and self.data.qpos[self.hinge] >= self.config.success_angle

    def _dense_costs(self, features):
        delta, _, alignment, _, grasp, _, _ = features
        costs = {"distance": -5 * float(np.linalg.norm(delta)), "orientation": -0.25 * (1 - alignment)}
        if self.stage != "reach":
            costs["grasp"] = -0.5 * (not grasp)
        if self.stage in ("press", "open", "full"):
            # Keep unfinished-goal costs when contact is lost; dropping the
            # handle must not remove the cost of pressing/opening it.
            costs["press"] = -float(np.clip((1.3 - self.data.qpos[self.latch]) / 1.3, 0, 2))
            if self.stage in ("open", "full"):
                costs["opening"] = -float(np.clip(
                    (self.config.success_angle - self.data.qpos[self.hinge]) / self.config.success_angle, 0, 2))
        return costs

    def reset(self, *, seed=None, options=None):
        super().reset(seed=seed, options=options)
        self._episode_start = "home"
        if self.np_random.random() >= self.home_probability:
            if self._anchors is None:
                self._anchors = _ResetStateBuilder(self).build()
            name = {"reach": "aligned", "grasp": "pregrasp", "press": "grasped", "open": "pressed"}.get(self.stage)
            if name is None:
                name = str(self.np_random.choice(("pregrasp", "grasped", "pressed")))
            snapshot, target, previous = self._anchors[name]
            noise = (options or {}).get("initial_noise", self.config.initial_noise)
            accepted = False
            for attempt in range(41):
                mujoco.mj_copyData(self.data, self.model, snapshot)
                perturbation = self.np_random.uniform(-noise, noise, 7) if attempt < 40 else np.zeros(7)
                self.data.qpos[self.qadr] = np.clip(self.data.qpos[self.qadr] + perturbation, self.lower, self.upper)
                self.arm_target = np.clip(target + perturbation, self.lower, self.upper)
                self.previous_action = previous.copy()
                mujoco.mj_forward(self.model, self.data)
                self._apply_controls()
                mujoco.mj_forward(self.model, self.data)
                features = self._features()
                physical_grasp = features[4]
                if (features[6] <= 0.0015 and self.data.qpos[self.hinge] < 0.05
                        and (name not in ("grasped", "pressed") or physical_grasp)
                        and (name != "pressed" or self.data.qpos[self.latch] >= 1.1)
                        and not self._goal_reached(features)):
                    accepted = True
                    break
            if not accepted:
                raise RuntimeError(f"Invalid curriculum reset state: {name}")
            self._episode_start = name
        self.data.time = 0
        self.steps = 0
        self._stage_hold = self.success_time = 0.0
        self.max_door_angle = float(self.data.qpos[self.hinge])
        self.grasp_seen = bool(self._features()[4])
        self.press_seen = bool(self.grasp_seen and self.data.qpos[self.latch] >= 1.1)
        self._ended = self._needs_reset = False
        if self.render_mode == "human":
            self.render()
        info = self._info()
        info.update(stage=self.stage, episode_start=self._episode_start, full_task_success=False)
        return self._obs(), info

    def step(self, action):
        if self._needs_reset:
            raise RuntimeError("Call reset() after changing the curriculum stage")
        effective = np.asarray(action, dtype=np.float64).copy()
        if effective.shape != (8,) or not np.isfinite(effective).all():
            raise ValueError("action must contain eight finite numbers")
        effective = np.clip(effective, -1, 1)
        if self.stage == "reach":
            effective[-1] = -1  # Open gripper while learning the pregrasp pose.
        _, _, _, _, info = super().step(effective)
        reason = info["failure_reason"]
        goal = self._goal_reached(self._features()) and not reason
        hold = self.config.success_hold_seconds if self.stage in ("open", "full") else self.curriculum.subtask_hold_seconds
        self._stage_hold = min(hold, self._stage_hold + self.dt) if goal else 0.0
        self.success_time = self._stage_hold
        success = self._stage_hold >= hold
        terminated = bool(success or reason)
        truncated = bool(self.steps >= self.config.max_steps and not terminated)
        inherited = info["reward_components"]
        components = self._dense_costs(self._features())
        components.update({key: inherited[key] for key in ("collision", "action", "smoothness", "time")})
        components["success"] = (50.0 if self.stage == "full" else 20.0) * success
        components["failure"] = -self.curriculum.failure_penalty * bool(reason)
        self._ended = terminated or truncated
        info.update(self._info(reason=reason, success=success, peak_depth=info["penetration_m"]))
        info.update(stage=self.stage, episode_start=self._episode_start, goal_reached=bool(goal),
                    full_task_success=bool(success and self.stage == "full" and self._episode_start == "home"),
                    reward_components=components)
        return self._obs(), float(sum(components.values())), terminated, truncated, info

    def specification(self):
        return {"task": "door_curriculum_v1", "model_path": str(self.model_path),
                "config": asdict(self.base_config), "curriculum": asdict(self.curriculum),
                "stage": self.stage, "home_probability": self.home_probability,
                "observation_dim": self.observation_space.shape[0], "action_dim": 8}
