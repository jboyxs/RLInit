"""Fixed-base TidyBot reaching task with seven joint-position actions."""

from pathlib import Path
import threading
import time
import xml.etree.ElementTree as ET

import gymnasium as gym
from gymnasium import spaces
import mujoco
import numpy as np


DEFAULT_MODEL = Path(__file__).resolve().parents[3] / "models/stanford_tidybot/tidybot.xml"
ARM_NAMES = tuple(f"joint_{i}" for i in range(1, 8))


def _fixed_arm_model(path: Path) -> tuple[mujoco.MjModel, np.ndarray]:
    """Remove base/gripper DOFs in memory, leaving the source XML untouched."""
    root = ET.parse(path).getroot()
    root.find("compiler").set("meshdir", str(path.parent / "assets"))
    home = root.find("keyframe/key[@name='home']")
    controls = np.fromstring(home.attrib["ctrl"], sep=" ")
    actuator_elements = list(root.find("actuator"))
    home_controls = dict(zip((e.attrib["name"] for e in actuator_elements), controls, strict=True))
    # Gripper linkage constraints/transmission are unnecessary after all fingers
    # become rigidly attached at their zero-angle (open) configuration.
    for tag in ("keyframe", "equality", "tendon"):
        for element in root.findall(tag):
            root.remove(element)
    for body in root.findall(".//worldbody//body"):
        for joint in list(body.findall("joint")):
            if joint.attrib.get("name") not in ARM_NAMES:
                body.remove(joint)
    for element in actuator_elements:
        if element.attrib["name"] not in ARM_NAMES:
            root.find("actuator").remove(element)
    # Fixing the base welds it to the world, changing automatic parent collision
    # filtering. Preserve the original exclusion at the shoulder attachment.
    contact = root.find("contact")
    if contact is None:
        contact = ET.SubElement(root, "contact")
    ET.SubElement(contact, "exclude", body1="gen3/base_link", body2="shoulder_link")
    world = root.find("worldbody")
    ET.SubElement(world, "geom", name="floor", type="plane", size="3 3 0.1", rgba="0.3 0.3 0.3 1")
    ET.SubElement(world, "site", name="target", type="sphere", size="0.02", group="0", rgba="0.2 0.9 0.2 0.7")
    return (
        mujoco.MjModel.from_xml_string(ET.tostring(root, encoding="unicode")),
        np.array([home_controls[name] for name in ARM_NAMES]),
    )


class ReachEnv(gym.Env):
    """Reach a nearby feasible target; success is position error below 2 cm."""

    metadata = {"render_modes": ["human"], "render_fps": 50}

    def __init__(self, model_path=DEFAULT_MODEL, render_mode=None, max_steps=250):
        super().__init__()
        if render_mode not in (None, "human"):
            raise ValueError("render_mode must be None or 'human'")
        if max_steps < 1:
            raise ValueError("max_steps must be positive")
        self.render_mode = render_mode
        self.max_steps = max_steps
        self.model, self.home = _fixed_arm_model(Path(model_path).resolve())
        self.model.opt.timestep = 0.002
        self.frame_skip = 10
        self.data = mujoco.MjData(self.model)
        self._sample_data = mujoco.MjData(self.model)
        self.joints = np.array([self.model.joint(name).id for name in ARM_NAMES])
        self.qadr = self.model.jnt_qposadr[self.joints]
        self.vadr = self.model.jnt_dofadr[self.joints]
        self.actuators = np.array([self.model.actuator(name).id for name in ARM_NAMES])
        self.ee_id = self.model.site("pinch_site").id
        self.target_id = self.model.site("target").id
        self.lower = np.full(7, -np.inf)
        self.upper = np.full(7, np.inf)
        for i, (joint, actuator) in enumerate(zip(self.joints, self.actuators)):
            if self.model.jnt_limited[joint]:
                self.lower[i], self.upper[i] = self.model.jnt_range[joint]
            if self.model.actuator_ctrllimited[actuator]:
                lo, hi = self.model.actuator_ctrlrange[actuator]
                self.lower[i] = max(self.lower[i], lo)
                self.upper[i] = min(self.upper[i], hi)
        self.action_space = spaces.Box(-1.0, 1.0, shape=(7,), dtype=np.float32)
        self.observation_space = spaces.Box(-np.inf, np.inf, shape=(20,), dtype=np.float32)
        self.goal = np.zeros(3)
        self.steps = 0
        self._viewer = None
        self._viewer_threads = []

    def _obs(self):
        ee = self.data.site_xpos[self.ee_id]
        return np.concatenate((self.data.qpos[self.qadr], self.data.qvel[self.vadr], ee, self.goal - ee)).astype(np.float32)

    def _info(self):
        distance = float(np.linalg.norm(self.data.site_xpos[self.ee_id] - self.goal))
        return {"distance": distance, "is_success": distance < 0.02}

    def reset(self, *, seed=None, options=None):
        super().reset(seed=seed)
        mujoco.mj_resetData(self.model, self.data)
        self.data.qpos[self.qadr] = self.home
        self.data.ctrl[self.actuators] = self.home
        mujoco.mj_forward(self.model, self.data)
        initial_ee = self.data.site_xpos[self.ee_id].copy()
        # FK of nearby joint configurations guarantees kinematic reachability;
        # reject penetrating poses and targets already within success tolerance.
        for _ in range(500):
            mujoco.mj_resetData(self.model, self._sample_data)
            self._sample_data.qpos[self.qadr] = np.clip(
                self.home + self.np_random.uniform(-0.25, 0.25, 7), self.lower, self.upper
            )
            mujoco.mj_forward(self.model, self._sample_data)
            candidate = self._sample_data.site_xpos[self.ee_id]
            distance = np.linalg.norm(candidate - initial_ee)
            penetrating = any(c.dist < -1e-4 for c in self._sample_data.contact)
            if 0.04 <= distance <= 0.15 and not penetrating:
                self.goal = candidate.copy()
                break
        else:
            raise RuntimeError("Could not sample a collision-free nearby target")
        self.model.site_pos[self.target_id] = self.goal
        mujoco.mj_forward(self.model, self.data)
        self.steps = 0
        if self.render_mode == "human":
            self.render()
        return self._obs(), self._info()

    def step(self, action):
        start = time.monotonic() if self.render_mode == "human" else None
        action = np.asarray(action, dtype=np.float64)
        if action.shape != (7,) or not np.isfinite(action).all():
            raise ValueError("action must contain seven finite numbers")
        action = np.clip(action, -1, 1)
        self.data.ctrl[self.actuators] = np.clip(
            self.data.qpos[self.qadr] + 0.03 * action, self.lower, self.upper
        )
        mujoco.mj_step(self.model, self.data, nstep=self.frame_skip)
        # Refresh site positions for the final integrated state.
        mujoco.mj_forward(self.model, self.data)
        self.steps += 1
        if not np.isfinite(self.data.qpos).all() or not np.isfinite(self.data.qvel).all():
            raise RuntimeError("Non-finite MuJoCo state")
        info = self._info()
        reward = -info["distance"] - 0.001 * float(action @ action)
        reward += 5.0 * info["is_success"]
        if self.render_mode == "human":
            self.render()
            if self._viewer.is_running():
                time.sleep(max(0.0, self.frame_skip * self.model.opt.timestep - (time.monotonic() - start)))
        return self._obs(), reward, info["is_success"], self.steps >= self.max_steps, info

    def render(self):
        if self.render_mode != "human":
            return
        if self._viewer is None:
            from mujoco import viewer

            existing_threads = set(threading.enumerate())
            self._viewer = viewer.launch_passive(self.model, self.data)
            self._viewer_threads = [
                thread for thread in threading.enumerate()
                if thread not in existing_threads and thread.daemon
            ]
            self._viewer.cam.lookat[:] = [0, 0, 0.7]
            self._viewer.cam.distance = 2.5
        if self._viewer.is_running():
            self._viewer.sync()

    def close(self):
        if self._viewer is not None:
            self._viewer.close()
            # close() only requests exit. Wait for GUI resource cleanup before
            # Python's exit handlers terminate GLFW underneath the render loop.
            for thread in self._viewer_threads:
                thread.join(timeout=5)
            self._viewer_threads = []
            self._viewer = None
