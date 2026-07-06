"""PD control for MuJoCo sim2sim rollouts."""

from __future__ import annotations

import numpy as np

from sim2sim_genesis.onnx_policy import PolicyMeta
from sim2sim_mujoco.scene import MujocoSceneAdapter


class MujocoPdController:
    """Convert ONNX actions to joint targets and apply explicit PD torques."""

    def __init__(
        self,
        *,
        scene: MujocoSceneAdapter,
        meta: PolicyMeta,
        control_dt: float,
        sim_dt: float,
        torque_limit: float | None,
    ):
        ratio = float(control_dt) / float(sim_dt)
        if abs(ratio - round(ratio)) > 1.0e-6:
            raise ValueError(f"control_dt/sim_dt must be an integer. Got {control_dt=} {sim_dt=}.")
        self.scene = scene
        self.meta = meta
        self.torque_limit = torque_limit
        self.sim_decimation = int(round(ratio))

    def compute_joint_target(self, raw_action: np.ndarray) -> np.ndarray:
        """Convert policy output into desired joint positions."""

        action = np.asarray(raw_action, dtype=np.float32)
        if action.ndim == 2:
            if action.shape[0] != 1:
                raise ValueError(f"MuJoCo v0 supports one env, got action batch {action.shape}.")
            action = action[0]
        if action.shape[0] != len(self.meta.joint_names):
            raise ValueError(f"Action dim {action.shape[0]} does not match joint count {len(self.meta.joint_names)}.")
        return (self.meta.default_joint_pos + self.meta.action_scale * action).astype(np.float32)

    def step(self, joint_target: np.ndarray) -> None:
        """Advance MuJoCo one policy control step."""

        target = np.asarray(joint_target, dtype=np.float64).reshape(-1)
        for _ in range(self.sim_decimation):
            joint_pos = self.scene.data.qpos[self.scene.joint_qpos_indices]
            joint_vel = self.scene.data.qvel[self.scene.joint_dof_indices]
            torque = (target - joint_pos) * self.meta.joint_stiffness - joint_vel * self.meta.joint_damping
            if self.torque_limit is not None:
                torque = np.clip(torque, -self.torque_limit, self.torque_limit)
            self.scene.data.qfrc_applied[:] = 0.0
            self.scene.data.qfrc_applied[self.scene.joint_dof_indices] = torque
            self.scene.mujoco.mj_step(self.scene.model, self.scene.data)

