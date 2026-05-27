"""Trajectory recording helpers for the modular Genesis sim2sim evaluator."""

from __future__ import annotations

import numpy as np
import torch

from rsl_rl.utils import StateActionTrajectoryRecorder


class TrajectoryRecorder:
    """Wrap the shared state-action recorder for Genesis rollout data collection."""

    def __init__(self, num_envs: int, fps: float, target_trajectories: int, output_path: str | None):
        """Create a trajectory recorder for optional motion dataset export."""

        self.output_path = output_path
        self.recorder = StateActionTrajectoryRecorder(
            num_envs=num_envs,
            fps=fps,
            target_trajectories=target_trajectories,
            state_keys=(
                "joint_pos",
                "joint_vel",
                "body_pos_w",
                "body_quat_w",
                "body_lin_vel_w",
                "body_ang_vel_w",
            ),
            action_key="action",
        )

    @property
    def collected_count(self) -> int:
        """Return the number of finalized trajectories currently buffered."""

        return self.recorder.collected_count

    def append_step(self, state_batch: dict[str, np.ndarray], action_batch: np.ndarray) -> None:
        """Append one rollout step worth of state and action data."""

        state_tensors = {
            "joint_pos": torch.from_numpy(state_batch["joint_pos"]),
            "joint_vel": torch.from_numpy(state_batch["joint_vel"]),
            "body_pos_w": torch.from_numpy(state_batch["log_body_pos_w"]),
            "body_quat_w": torch.from_numpy(state_batch["log_body_quat_w"]),
            "body_lin_vel_w": torch.from_numpy(state_batch["log_body_lin_vel_w"]),
            "body_ang_vel_w": torch.from_numpy(state_batch["log_body_ang_vel_w"]),
        }
        action_tensor = torch.from_numpy(np.asarray(action_batch, dtype=np.float32))
        self.recorder.append_step(state_tensors, action_tensor)

    def finalize_rollout(self) -> int:
        """Finalize all open per-environment trajectories for the current rollout."""

        previous_count = self.recorder.collected_count
        self.recorder.finalize_open()
        return self.recorder.collected_count - previous_count

    def has_reached_target(self) -> bool:
        """Return whether the requested number of trajectories has been collected."""

        return self.recorder.has_reached_target()

    def save(self) -> str | None:
        """Persist the recorded motion dataset if output was configured."""

        if self.output_path is None:
            return None

        saved_path = self.recorder.save_dataset(self.output_path, include_partial=False)
        if saved_path is None:
            return None

        payload = dict(np.load(saved_path))
        payload["actions"] = payload["action"]
        np.savez(saved_path, **payload)
        return saved_path
