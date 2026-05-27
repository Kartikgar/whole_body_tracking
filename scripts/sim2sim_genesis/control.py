"""Control logic for the modular Genesis sim2sim evaluator."""

from __future__ import annotations

import numpy as np

from sim2sim_genesis.observations import ObservationBuilder
from sim2sim_genesis.onnx_policy import PolicyMeta
from sim2sim_genesis.scene import GenesisSceneAdapter


class PdController:
    """Convert policy actions into joint targets and explicit PD torques."""

    def __init__(
        self,
        scene: GenesisSceneAdapter,
        meta: PolicyMeta,
        observation_builder: ObservationBuilder,
        control_dt: float,
        sim_dt: float,
        torque_limit: float | None,
    ):
        """Create the PD controller used by the modular evaluator."""

        ratio = control_dt / sim_dt
        if abs(ratio - round(ratio)) > 1e-6:
            raise ValueError(f"control_dt/sim_dt must be an integer. Got control_dt={control_dt}, sim_dt={sim_dt}")

        self.scene = scene
        self.meta = meta
        self.observation_builder = observation_builder
        self.torque_limit = torque_limit
        self.sim_decimation = int(round(ratio))

    def compute_joint_target(self, raw_action: np.ndarray, time_step: int) -> np.ndarray:
        """Convert policy output into desired joint positions."""

        processed_action = np.asarray(raw_action, dtype=np.float32).copy()
        if processed_action.ndim == 1:
            processed_action = processed_action.reshape(1, -1)

        if self.observation_builder.requires_motion_action:
            processed_action += self.observation_builder.motion_action_at(
                time_step, batch_size=processed_action.shape[0]
            )

        default_joint_pos = self.observation_builder.default_joint_pos_for_batch(processed_action.shape[0])
        joint_target = default_joint_pos + self.meta.action_scale[None, :] * processed_action
        if self.scene.num_envs == 1 and joint_target.shape[0] == 1:
            return joint_target[0].astype(np.float32)
        return joint_target.astype(np.float32)

    def step(self, joint_target: np.ndarray) -> None:
        """Advance Genesis for one control step using explicit PD torques."""

        for _ in range(self.sim_decimation):
            joint_pos = self.scene.to_numpy(self.scene.robot.get_dofs_position(self.scene.joint_dof_indices))
            joint_vel = self.scene.to_numpy(self.scene.robot.get_dofs_velocity(self.scene.joint_dof_indices))
            torque = (joint_target - joint_pos) * self.meta.joint_stiffness - joint_vel * self.meta.joint_damping
            if self.torque_limit is not None:
                torque = np.clip(torque, -self.torque_limit, self.torque_limit)
            self.scene.call_genesis(
                self.scene.robot.control_dofs_force,
                torque.astype(np.float32),
                self.scene.joint_dof_indices,
            )
            self.scene.scene.step()
