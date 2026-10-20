"""Physical checks for a downward-only handle that releases by 45 degrees."""

import unittest

import mujoco
import numpy as np

from rlinit.algorithmhand.door_env import DEFAULT_MODEL, fixed_base_model


class DoorModelTests(unittest.TestCase):
    def test_downward_only_limit_and_passive_joints(self):
        model = fixed_base_model(DEFAULT_MODEL)
        handle = model.joint("door_latch_joint").id
        door = model.joint("door_hinge").id
        np.testing.assert_allclose(model.jnt_range[handle], [0, np.pi / 3], rtol=0, atol=1e-15)
        np.testing.assert_array_equal(model.jnt_range[door], [0, 0.4])
        self.assertTrue(model.jnt_limited[handle])
        self.assertEqual(model.jnt_stiffness[handle], 1.)
        for actuator in range(model.nu):
            if model.actuator_trntype[actuator] == mujoco.mjtTrn.mjTRN_JOINT:
                self.assertNotIn(int(model.actuator_trnid[actuator, 0]), (handle, door))

    def test_latch_stays_locked_at_zero_and_clears_by_45_degrees(self):
        model = fixed_base_model(DEFAULT_MODEL)
        reference = mujoco.MjData(model)
        mujoco.mj_resetDataKeyframe(model, reference, model.key("home").id)
        handle = int(model.jnt_qposadr[model.joint("door_latch_joint").id])
        hinge = int(model.jnt_qposadr[model.joint("door_hinge").id])
        pairs = [(model.geom(first).id, model.geom(second).id)
                 for first in ("door_latch", "door_latch_tip")
                 for second in ("door_left_frame", "door_right_frame", "door_left_support", "door_right_support")]

        def minimum_clearance(angle):
            reference.qpos[handle] = angle
            distances = []
            for door_angle in np.linspace(0, 0.4, 401):
                reference.qpos[hinge] = door_angle
                mujoco.mj_forward(model, reference)
                distances.extend(mujoco.mj_geomDistance(model, reference, a, b, 1., None) for a, b in pairs)
            return min(distances)

        self.assertLess(minimum_clearance(0), -0.01)
        self.assertLess(minimum_clearance(np.deg2rad(40)), -0.001)
        self.assertGreater(minimum_clearance(np.pi / 4), 0.0015)

    @staticmethod
    def isolated_fixture():
        # Isolate door mechanics from the robot. Applied torques in these tests
        # measure constraints; they are never used by the training environment.
        fixture = (DEFAULT_MODEL.parent / "objects/door_lock_fixture.xml").as_posix()
        return mujoco.MjModel.from_xml_string(
            '<mujoco><compiler angle="radian"/>'
            '<option timestep="0.002" integrator="implicitfast" cone="elliptic" impratio="10"/>'
            f'<include file="{fixture}"/></mujoco>')

    def test_upward_torque_is_stopped_near_horizontal(self):
        model = self.isolated_fixture()
        joint = model.joint("door_latch_joint").id
        position, dof = int(model.jnt_qposadr[joint]), int(model.jnt_dofadr[joint])
        for torque in (-0.5, -3., -10.):
            with self.subTest(torque=torque):
                data = mujoco.MjData(model)
                minimum = 0.
                for _ in range(1000):
                    data.qfrc_applied[dof] = torque
                    mujoco.mj_step(model, data)
                    minimum = min(minimum, data.qpos[position])
                # MuJoCo limits are soft constraints; allow small solver deflection.
                self.assertGreater(minimum, np.deg2rad(-0.2))
                self.assertFalse(any(w.number for w in data.warning))

    def test_pull_is_blocked_at_zero_and_opens_with_45_degree_depression(self):
        model = self.isolated_fixture()
        handle = model.joint("door_latch_joint").id
        hinge = model.joint("door_hinge").id
        hp, hd = int(model.jnt_qposadr[handle]), int(model.jnt_dofadr[handle])
        dp, dd = int(model.jnt_qposadr[hinge]), int(model.jnt_dofadr[hinge])
        maximum_openings = []
        for target in (0., np.pi / 4):
            data = mujoco.MjData(model)
            maximum = 0.
            for tick in range(2500):
                data.qfrc_applied[hd] = 30 * (target - data.qpos[hp]) - 3 * data.qvel[hd] + data.qpos[hp]
                data.qfrc_applied[dd] = 2. if tick >= 1000 else 0.
                mujoco.mj_step(model, data)
                maximum = max(maximum, data.qpos[dp])
            maximum_openings.append(maximum)
            self.assertFalse(any(w.number for w in data.warning))
        self.assertLess(maximum_openings[0], 0.05)
        self.assertGreater(maximum_openings[1], 0.3)

    def test_downward_torque_settles_at_60_degree_stop(self):
        model = self.isolated_fixture()
        joint = model.joint("door_latch_joint").id
        position, dof = int(model.jnt_qposadr[joint]), int(model.jnt_dofadr[joint])
        for torque in (3., 10.):
            with self.subTest(torque=torque):
                data = mujoco.MjData(model)
                for _ in range(1500):
                    data.qfrc_applied[dof] = torque
                    mujoco.mj_step(model, data)
                # Test the sustained stop; an impulsive load can temporarily
                # overshoot a soft joint limit during deceleration.
                self.assertAlmostEqual(np.rad2deg(data.qpos[position]), 60., delta=0.01)
                self.assertFalse(any(w.number for w in data.warning))


if __name__ == "__main__":
    unittest.main()
