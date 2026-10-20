"""Direct door rewards with paired finger-to-handle surface distances."""

import numpy as np

from .door_env import DEFAULT_MODEL, DoorEnv


TASK = "fixed_base_door_direct_reward_v1"


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
        return dict(super().specification(), task=TASK, position_reward=self.position_reward,
                    orientation_reward=self.orientation_reward,
                    distance_metric=self.distance_metric,
                    distance_weight=self.distance_weight,
                    orientation_weight=self.orientation_weight,
                    reward_source=f"{position_source}; {orientation_source}; other door_env.py features/weights; "
                                  "direct current-state reward",
                    success_rule="grasp and door_angle >= success_angle for success_hold_seconds")
