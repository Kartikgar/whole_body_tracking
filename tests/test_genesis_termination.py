"""Isaac batch failure criteria used by the modular Genesis evaluator."""
from pathlib import Path
import sys
import unittest
import tempfile

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "scripts"))
from sim2sim_genesis.metrics import TrackingMetricsEvaluator
from sim2sim_genesis.config import load_logged_initial_states
from sim2sim_genesis.scene import GenesisSceneAdapter
from sim2sim_genesis.math_utils import quat_rotate_wxyz


class GenesisTerminationTests(unittest.TestCase):
    def test_logged_root_angular_velocity_frame(self):
        scene=object.__new__(GenesisSceneAdapter)
        scene.joint_names=['joint'];scene.joint_qs_indices=[7];scene.joint_dof_indices=[6]
        quaternion=np.array([np.sqrt(.5),0.,0.,np.sqrt(.5)],dtype=np.float32)
        world_ang=np.array([1.,0.,0.],dtype=np.float32)
        _,vel=scene._build_qpos_qvel_from_logged(logged_joint_names=['joint'],
            initial_joint_pos=np.zeros(1),initial_joint_vel=np.zeros(1),
            initial_body_pos_w=np.zeros((1,3)),initial_body_quat_w=quaternion[None],
            initial_body_lin_vel_w=np.array([[.1,.2,.3]]),initial_body_ang_vel_w=world_ang[None],root_body_index=0)
        np.testing.assert_allclose(vel[3:6],[0.,-1.,0.],atol=1e-6)
        np.testing.assert_allclose(quat_rotate_wxyz(quaternion,vel[3:6]),world_ang,atol=1e-6)
        np.testing.assert_allclose(vel[:3],[.1,.2,.3],atol=1e-6)

    def test_paired_startup_removes_placement_offsets(self):
        with tempfile.TemporaryDirectory() as folder:
            path=Path(folder)/'initial.npz'
            origins=np.array([[10.,20.,0.],[-10.,-20.,0.]])
            np.savez(path, joint_names=['joint'],body_names=['pelvis'],
                     initial_joint_pos=np.zeros((2,1)), initial_joint_vel=np.zeros((2,1)),
                     initial_body_pos_w=origins[:,None,:]+np.array([0.,0.,.8]),
                     initial_body_quat_w=np.tile([1.,0.,0.,0.],(2,1,1)),env_origins=origins)
            initial=load_logged_initial_states(path,2)
            np.testing.assert_allclose(initial['initial_body_pos_w'],np.tile([0.,0.,.8],(2,1,1)))
            with self.assertRaises(ValueError):load_logged_initial_states(path,1)

    def setUp(self):
        self.names = ["pelvis", "left_ankle_roll_link", "right_ankle_roll_link",
                      "left_wrist_yaw_link", "right_wrist_yaw_link"]
        self.evaluator = TrackingMetricsEvaluator(self.names, 0, 0, "isaac_batch")
        self.reference = dict(body_pos_w=np.zeros((5, 3)),
                              body_quat_w=np.tile([1., 0., 0., 0.], (5, 1)),
                              joint_pos=np.zeros(29), joint_vel=np.zeros(29))

    def check_state(self, modify, reason):
        state = {key: value.copy() for key, value in self.reference.items()}
        modify(state)
        self.assertEqual(self.evaluator.termination_check(state, self.reference, None),
                         (reason != "completed", reason))

    def test_height_threshold_is_strict(self):
        self.check_state(lambda s: s["body_pos_w"].__setitem__((0, 2), .25), "completed")
        self.check_state(lambda s: s["body_pos_w"].__setitem__((0, 2), .251), "anchor_pos_z")

    def test_all_four_endpoints(self):
        for i in range(1, 5):
            with self.subTest(endpoint=self.names[i]):
                self.check_state(lambda s: s["body_pos_w"].__setitem__((i, 2), -.251),
                                 "end_effector_vertical_error")

    def test_orientation_and_nonfinite(self):
        self.check_state(lambda s: s["body_quat_w"].__setitem__(0, [0., 1., 0., 0.]), "anchor_ori")
        self.check_state(lambda s: s["joint_vel"].__setitem__(0, np.nan), "nonfinite_state")

    def test_missing_endpoint_rejected(self):
        with self.assertRaises(ValueError):
            TrackingMetricsEvaluator(["pelvis"], 0, 0, "isaac_batch")

    def test_other_callers_keep_legacy_profile(self):
        evaluator = TrackingMetricsEvaluator(self.names, 0, 0)
        state = {key: value.copy() for key, value in self.reference.items()}
        state["body_pos_w"][0, 2] = .3
        self.assertEqual(evaluator.termination_check(state, self.reference, None), (False, "completed"))


if __name__ == "__main__":
    unittest.main()
