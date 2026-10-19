"""Replay initial physical states visited by an online joint-action SAC policy.

This stores simulator states, not demonstrations or action targets. Evaluation
never enables this training-only reset distribution.
"""

from dataclasses import asdict
import hashlib
import json
from pathlib import Path

import mujoco
import numpy as np


FORMAT = "online_joint_sac_state_bank_v1"
STATE = mujoco.mjtState.mjSTATE_INTEGRATION


def eligible_progress_state(info, *, terminated, truncated, target_error, steps, config):
    """Only genuine nonterminal grasped progress, with no near-success timer."""
    return (not terminated and not truncated and not info["failure_reason"]
            and info["grasped"] and min(info["left_contact_N"], info["right_contact_N"]) >= 2.0
            and 0.005 <= info["door_angle"] < config.success_angle - 0.02
            and (info["handle_angle"] >= 1.1 or info["door_angle"] >= 0.05)
            and info["success_hold_seconds"] == 0 and info["penetration_m"] < 0.0015
            and info["bad_contact_N"] < 10 and target_error < 0.15
            and steps < config.max_steps - 50)


class AdaptiveProgressStates:
    """Keep actual visits across opening levels, so earlier skills get practice."""

    def __init__(self, capacity=128):
        self.capacity = capacity
        self.states = []
        self.angles = []
        self.best_angle = 0.0

    def record(self, env, info, *, terminated, truncated):
        if env.steps % 2 or not eligible_progress_state(
                info, terminated=terminated, truncated=truncated,
                target_error=float(np.max(np.abs(env.arm_target - env.data.qpos[env.qadr]))),
                steps=env.steps, config=env.config):
            return
        self.add(capture_state(env), info["door_angle"])

    def add(self, snapshot, angle):
        self.best_angle = max(self.best_angle, angle)
        self.states.append(snapshot)
        self.angles.append(angle)
        groups = {}
        for index, value in enumerate(self.angles):
            groups.setdefault(int(value / 0.02), []).append(index)
        per_group = max(1, min(8, self.capacity // len(groups)))
        keep = sorted(index for indices in groups.values() for index in indices[-per_group:])
        self.states = [self.states[i] for i in keep]
        self.angles = [self.angles[i] for i in keep]

    def sample_index(self, rng):
        groups = {}
        for index, value in enumerate(self.angles):
            groups.setdefault(int(value / 0.02), []).append(index)
        indices = groups[sorted(groups)[int(rng.integers(len(groups)))]]
        return indices[int(rng.integers(len(indices)))]

    def restore(self, env, rng):
        OnlineStateBank.restore_snapshot(env, self.states[self.sample_index(rng)])

    def seed_from_bank(self, env, bank):
        original = capture_state(env)
        try:
            for index in range(bank.size):
                bank.restore(env, index)
                info = env._info()
                if eligible_progress_state(
                        info, terminated=False, truncated=False, steps=env.steps, config=env.config,
                        target_error=float(np.max(np.abs(env.arm_target - env.data.qpos[env.qadr])))):
                    self.add(capture_state(env), info["door_angle"])
        finally:
            OnlineStateBank.restore_snapshot(env, original)


def capture_state(env):
    integration = np.empty(mujoco.mj_stateSize(env.model, STATE))
    mujoco.mj_getState(env.model, env.data, integration, STATE)
    return dict(integration=integration, arm_target=env.arm_target.copy(),
                previous_action=env.previous_action.copy(),
                episode=np.array([env.grasp_seen, env.press_seen, env.success_time,
                                  env.steps, env.max_door_angle], dtype=np.float64))


def save_bank(path, env, states, *, checkpoint, collection=None):
    if not states:
        raise ValueError("No eligible online SAC states were collected")
    checkpoint = Path(checkpoint).resolve()
    metadata = dict(format=FORMAT, source_kind="online_joint_sac_rollouts",
                    checkpoint=str(checkpoint),
                    checkpoint_sha256=hashlib.sha256(checkpoint.read_bytes()).hexdigest(),
                    source_environment=env.specification(),
                    model_sha256=hashlib.sha256(env.model_path.read_bytes()).hexdigest(),
                    collection=collection,
                    action_labels_used_for_learning=False)
    path = Path(path).resolve()
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("wb") as stream:
        np.savez_compressed(stream, metadata=json.dumps(metadata),
                            **{key: np.stack([state[key] for state in states]) for key in states[0]})
    return metadata


class OnlineStateBank:
    def __init__(self, path, env, *, expected_task):
        self.path = Path(path).resolve()
        with np.load(self.path, allow_pickle=False) as archive:
            self.metadata = json.loads(str(archive["metadata"].item()))
            self.arrays = {key: archive[key].copy()
                           for key in ("integration", "arm_target", "previous_action", "episode")}
        source = self.metadata.get("source_environment", {})
        source_physics, target_physics = dict(source.get("config", {})), asdict(env.config)
        # Precision rewards are direct state rewards. Discounting changes SAC's
        # planning horizon, not these simulator states or their physics.
        source_physics.pop("gamma", None)
        target_physics.pop("gamma", None)
        if (self.metadata.get("format") != FORMAT
                or self.metadata.get("source_kind") != "online_joint_sac_rollouts"
                or source.get("task") != expected_task
                or source_physics != target_physics
                or self.metadata.get("model_sha256") != hashlib.sha256(env.model_path.read_bytes()).hexdigest()):
            raise ValueError("Reset bank must contain online joint SAC states with matching physics/config")
        self.size = len(self.arrays["integration"])
        sizes = dict(integration=mujoco.mj_stateSize(env.model, STATE), arm_target=7,
                     previous_action=8, episode=5)
        if self.size < 1 or any(value.shape != (self.size, sizes[key]) or not np.isfinite(value).all()
                                for key, value in self.arrays.items()):
            raise ValueError("Invalid reset bank array shapes or values")
        if (np.any(np.abs(self.arrays["previous_action"]) > 1)
                or np.any(self.arrays["episode"][:, 3] >= env.config.max_steps)):
            raise ValueError("Reset bank contains invalid action history or expired episodes")

    def restore(self, env, index):
        self.restore_snapshot(env, {key: values[index] for key, values in self.arrays.items()})

    @staticmethod
    def restore_snapshot(env, snapshot):
        # Restore exactly what the learner visited, including actuator commands
        # and warm-start state. No IK, pose servo or external force is computed.
        mujoco.mj_setState(env.model, env.data, snapshot["integration"], STATE)
        if np.any(env.data.qfrc_applied) or np.any(env.data.xfrc_applied):
            raise ValueError("Training reset states must not inject external forces")
        env.arm_target = snapshot["arm_target"].copy()
        env.previous_action = snapshot["previous_action"].copy()
        grasp_seen, press_seen, timer, steps, max_angle = snapshot["episode"]
        env.grasp_seen, env.press_seen = bool(grasp_seen), bool(press_seen)
        env.success_time, env.steps = float(timer), int(steps)
        env.max_door_angle, env._ended = float(max_angle), False
        mujoco.mj_forward(env.model, env.data)
        if env._contacts()[2] > env.config.failure_penetration:
            raise ValueError("Reset bank state violates the original penetration limit")

    def specification(self, probability):
        return dict(path=str(self.path), sha256=hashlib.sha256(self.path.read_bytes()).hexdigest(),
                    states=self.size, probability=probability,
                    source=self.metadata, evaluation_initialization="original noisy home only")
