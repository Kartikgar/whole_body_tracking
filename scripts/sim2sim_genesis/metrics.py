"""Tracking and rollout metrics for the modular Genesis sim2sim evaluator."""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np

from sim2sim_genesis.math_utils import (
    quat_conjugate_wxyz,
    quat_error_magnitude_wxyz,
    quat_mul_wxyz,
    quat_rotate_inverse_wxyz,
    quat_rotate_wxyz,
    yaw_quat_wxyz,
)


def compute_error_vel(reference_positions: np.ndarray, predicted_positions: np.ndarray) -> np.ndarray:
    """Compute mean per-body velocity error across consecutive frames."""

    if reference_positions.shape[0] < 2:
        return np.empty((0,), dtype=np.float32)
    reference_vel = reference_positions[1:] - reference_positions[:-1]
    predicted_vel = predicted_positions[1:] - predicted_positions[:-1]
    normed = np.linalg.norm(predicted_vel - reference_vel, axis=2)
    return np.mean(normed, axis=1).astype(np.float32)


def compute_error_accel(reference_positions: np.ndarray, predicted_positions: np.ndarray) -> np.ndarray:
    """Compute mean per-body acceleration error across consecutive frames."""

    if reference_positions.shape[0] < 3:
        return np.empty((0,), dtype=np.float32)
    reference_accel = reference_positions[:-2] - 2.0 * reference_positions[1:-1] + reference_positions[2:]
    predicted_accel = predicted_positions[:-2] - 2.0 * predicted_positions[1:-1] + predicted_positions[2:]
    normed = np.linalg.norm(predicted_accel - reference_accel, axis=2)
    return np.mean(normed, axis=1).astype(np.float32)


def compute_metrics_lite(predicted_positions: np.ndarray, reference_positions: np.ndarray, root_idx: int = 0) -> dict[str, np.ndarray]:
    """Compute ASAP-style MPJPE, velocity, and acceleration metrics."""

    if predicted_positions.shape != reference_positions.shape:
        raise ValueError(
            f"pred/gt shape mismatch: {predicted_positions.shape} vs {reference_positions.shape}"
        )
    if predicted_positions.ndim != 3:
        raise ValueError(f"Expected [T, B, 3] body positions, got shape={predicted_positions.shape}")

    mpjpe_global = np.linalg.norm(reference_positions - predicted_positions, axis=2) * 1000.0
    velocity_distance = compute_error_vel(predicted_positions, reference_positions) * 1000.0
    accel_distance = compute_error_accel(predicted_positions, reference_positions) * 1000.0

    predicted_local = predicted_positions - predicted_positions[:, [root_idx]]
    reference_local = reference_positions - reference_positions[:, [root_idx]]
    mpjpe_local = np.linalg.norm(reference_local - predicted_local, axis=2) * 1000.0

    return {
        "mpjpe_g": mpjpe_global.astype(np.float32),
        "mpjpe_l": mpjpe_local.astype(np.float32),
        "vel_dist": velocity_distance.astype(np.float32),
        "accel_dist": accel_distance.astype(np.float32),
    }


@dataclass(slots=True)
class TrackingMetricsEvaluator:
    """Evaluate rollout tracking quality and early termination conditions."""

    body_names: list[str]
    anchor_idx: int
    root_idx: int

    def compute_body_relative_targets(
        self, state: dict[str, np.ndarray], reference: dict[str, np.ndarray]
    ) -> tuple[np.ndarray, np.ndarray]:
        """Build yaw-aligned relative target body poses for reward-style metrics."""

        robot_anchor_pos = state["body_pos_w"][self.anchor_idx]
        robot_anchor_quat = state["body_quat_w"][self.anchor_idx]
        ref_anchor_pos = reference["body_pos_w"][self.anchor_idx]
        ref_anchor_quat = reference["body_quat_w"][self.anchor_idx]

        delta_pos = np.repeat(robot_anchor_pos[None, :], len(self.body_names), axis=0)
        delta_pos[:, 2] = ref_anchor_pos[2]
        delta_ori = yaw_quat_wxyz(quat_mul_wxyz(robot_anchor_quat, quat_conjugate_wxyz(ref_anchor_quat)))
        delta_ori_batch = np.repeat(delta_ori[None, :], len(self.body_names), axis=0)

        ref_body_pos = reference["body_pos_w"]
        ref_body_quat = reference["body_quat_w"]
        ref_anchor_repeat = np.repeat(ref_anchor_pos[None, :], len(self.body_names), axis=0)

        body_pos_relative = delta_pos + quat_rotate_wxyz(delta_ori, ref_body_pos - ref_anchor_repeat)
        body_quat_relative = quat_mul_wxyz(delta_ori_batch, ref_body_quat)
        return body_pos_relative, body_quat_relative

    def termination_check(
        self, state: dict[str, np.ndarray], reference: dict[str, np.ndarray], body_pos_relative: np.ndarray
    ) -> tuple[bool, str]:
        """Return whether the rollout should terminate and the associated reason."""

        del body_pos_relative

        robot_anchor_pos = state["body_pos_w"][self.anchor_idx]
        robot_anchor_quat = state["body_quat_w"][self.anchor_idx]
        ref_anchor_pos = reference["body_pos_w"][self.anchor_idx]
        ref_anchor_quat = reference["body_quat_w"][self.anchor_idx]

        if abs(ref_anchor_pos[2] - robot_anchor_pos[2]) > 0.5:
            return True, "anchor_pos_z"

        gravity = np.array([0.0, 0.0, -1.0], dtype=np.float32)
        motion_projected_gravity = quat_rotate_inverse_wxyz(ref_anchor_quat, gravity)
        robot_projected_gravity = quat_rotate_inverse_wxyz(robot_anchor_quat, gravity)
        if abs(motion_projected_gravity[2] - robot_projected_gravity[2]) > 0.8:
            return True, "anchor_ori"

        return False, "completed"

    def tracking_metrics(self, state: dict[str, np.ndarray], reference: dict[str, np.ndarray]) -> dict[str, float | str]:
        """Compute stepwise tracking errors and reward-style terms."""

        robot_anchor_pos = state["body_pos_w"][self.anchor_idx]
        robot_anchor_quat = state["body_quat_w"][self.anchor_idx]
        ref_anchor_pos = reference["body_pos_w"][self.anchor_idx]
        ref_anchor_quat = reference["body_quat_w"][self.anchor_idx]

        body_pos_relative, body_quat_relative = self.compute_body_relative_targets(state, reference)

        error_anchor_pos = float(np.linalg.norm(ref_anchor_pos - robot_anchor_pos))
        error_anchor_rot = float(quat_error_magnitude_wxyz(ref_anchor_quat, robot_anchor_quat))
        error_body_pos = float(np.linalg.norm(body_pos_relative - state["body_pos_w"], axis=-1).mean())
        error_body_rot = float(quat_error_magnitude_wxyz(body_quat_relative, state["body_quat_w"]).mean())
        error_joint_pos = float(np.linalg.norm(reference["joint_pos"] - state["joint_pos"]))
        error_joint_vel = float(np.linalg.norm(reference["joint_vel"] - state["joint_vel"]))

        reward_anchor_pos = float(np.exp(-(np.square(ref_anchor_pos - robot_anchor_pos).sum()) / (0.3**2)))
        reward_anchor_ori = float(np.exp(-(error_anchor_rot**2) / (0.4**2)))

        body_pos_sq = np.square(body_pos_relative - state["body_pos_w"]).sum(axis=-1).mean()
        reward_body_pos = float(np.exp(-body_pos_sq / (0.3**2)))

        body_rot_sq = np.square(quat_error_magnitude_wxyz(body_quat_relative, state["body_quat_w"])).mean()
        reward_body_ori = float(np.exp(-body_rot_sq / (0.4**2)))

        body_lin_sq = np.square(reference["body_lin_vel_w"] - state["body_lin_vel_w"]).sum(axis=-1).mean()
        reward_body_lin_vel = float(np.exp(-body_lin_sq / (1.0**2)))

        body_ang_sq = np.square(reference["body_ang_vel_w"] - state["body_ang_vel_w"]).sum(axis=-1).mean()
        reward_body_ang_vel = float(np.exp(-body_ang_sq / (3.14**2)))

        tracking_reward_total = (
            0.5 * reward_anchor_pos
            + 0.5 * reward_anchor_ori
            + 1.0 * reward_body_pos
            + 1.0 * reward_body_ori
            + 1.0 * reward_body_lin_vel
            + 1.0 * reward_body_ang_vel
        )

        terminated, reason = self.termination_check(state, reference, body_pos_relative)
        return {
            "error_anchor_pos": error_anchor_pos,
            "error_anchor_rot": error_anchor_rot,
            "error_body_pos": error_body_pos,
            "error_body_rot": error_body_rot,
            "error_joint_pos": error_joint_pos,
            "error_joint_vel": error_joint_vel,
            "rew_tracking_anchor_pos": reward_anchor_pos,
            "rew_tracking_anchor_ori": reward_anchor_ori,
            "rew_tracking_body_pos": reward_body_pos,
            "rew_tracking_body_ori": reward_body_ori,
            "rew_tracking_body_lin_vel": reward_body_lin_vel,
            "rew_tracking_body_ang_vel": reward_body_ang_vel,
            "tracking_reward_total": tracking_reward_total,
            "terminated": float(1.0 if terminated else 0.0),
            "termination_reason": reason,
        }
