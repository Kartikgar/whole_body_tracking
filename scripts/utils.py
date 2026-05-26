"""Shared utilities for whole_body_tracking scripts."""

from __future__ import annotations

import os
from typing import Sequence

import numpy as np
import torch

DEFAULT_STATE_ACTION_KEYS: tuple[str, ...] = (
    "joint_pos",
    "joint_vel",
    "body_pos_w",
    "body_quat_w",
    "body_lin_vel_w",
    "body_ang_vel_w",
)


class StateActionTrajectoryRecorder:
    """Record robot state tensors and applied actions into padded [num_traj, T, D] NPZ payloads."""

    def __init__(
        self,
        num_envs: int,
        fps: float,
        target_trajectories: int,
        state_keys: Sequence[str] = DEFAULT_STATE_ACTION_KEYS,
        action_key: str = "actions",
    ):
        self.num_envs = int(num_envs)
        self.fps = float(fps)
        self.target_trajectories = max(int(target_trajectories), 0)
        self.state_keys = tuple(state_keys)
        if len(self.state_keys) == 0:
            raise ValueError("state_keys must contain at least one key.")
        self.action_key = str(action_key)
        self.component_keys = list(self.state_keys)
        self.traj_buffers: list[dict[str, list[np.ndarray]]] = [
            {self.action_key: []} for _ in range(self.num_envs)
        ]
        for env_id in range(self.num_envs):
            for key in self.component_keys:
                self.traj_buffers[env_id][key] = []
        self.collected_trajectories: list[dict[str, np.ndarray]] = []
        self.saved_count = 0
        self.saved_motion_length = 0

    @property
    def collected_count(self) -> int:
        return len(self.collected_trajectories)

    def has_reached_target(self) -> bool:
        return self.target_trajectories > 0 and self.collected_count >= self.target_trajectories

    def _validate_state_components(self, components: dict[str, torch.Tensor]):
        missing = [key for key in self.component_keys if key not in components]
        if missing:
            raise AssertionError(
                "Missing required state components for trajectory logging: "
                f"{missing}. Expected keys={self.component_keys}."
            )
        extra = [key for key in components if key not in self.component_keys]
        if extra:
            raise AssertionError(
                "Unexpected state components for trajectory logging: "
                f"{extra}. Expected keys={self.component_keys}."
            )

    def append_step(self, state_components: dict[str, torch.Tensor], action_batch: torch.Tensor):
        self._validate_state_components(state_components)
        action_np = action_batch.detach().to("cpu", dtype=torch.float32).numpy()
        if action_np.ndim != 2 or action_np.shape[0] != self.num_envs:
            raise AssertionError(f"Expected action shape [num_envs, A], got {action_np.shape}.")

        component_np_map: dict[str, np.ndarray] = {}
        for key in self.component_keys:
            component_np = state_components[key].detach().to("cpu", dtype=torch.float32).numpy()
            if component_np.ndim < 2 or component_np.shape[0] != self.num_envs:
                raise AssertionError(
                    f"Expected state component '{key}' shape [num_envs, ...], got {component_np.shape}."
                )
            component_np_map[key] = component_np

        for env_id in range(self.num_envs):
            for key in self.component_keys:
                self.traj_buffers[env_id][key].append(component_np_map[key][env_id].astype(np.float32).copy())
            self.traj_buffers[env_id][self.action_key].append(action_np[env_id].astype(np.float32).copy())

    def finalize_done(self, dones: torch.Tensor) -> int:
        done_mask = dones.detach().to(device="cpu", dtype=torch.bool)
        if done_mask.ndim == 0:
            done_mask = done_mask.reshape(1)
        elif done_mask.ndim > 1:
            done_mask = done_mask.reshape(done_mask.shape[0], -1).any(dim=1)
        if done_mask.ndim != 1 or done_mask.shape[0] != self.num_envs:
            raise AssertionError(f"Expected done mask shape [num_envs], got {tuple(done_mask.shape)}.")

        gained = 0
        done_ids = torch.nonzero(done_mask, as_tuple=False).flatten().tolist()
        for env_id in done_ids:
            if self.has_reached_target():
                break
            gained += self._finalize_env_traj(env_id)
        return gained

    def finalize_open(self):
        for env_id in range(self.num_envs):
            if self.has_reached_target():
                break
            self._finalize_env_traj(env_id)

    def _finalize_env_traj(self, env_id: int) -> int:
        if self.has_reached_target():
            return 0
        if len(self.traj_buffers[env_id][self.action_key]) == 0:
            return 0
        keys = [*self.component_keys, self.action_key]
        traj = {key: np.stack(self.traj_buffers[env_id][key], axis=0).astype(np.float32) for key in keys}
        self.collected_trajectories.append(traj)
        self.traj_buffers[env_id] = {key: [] for key in keys}
        return 1

    def save_dataset(self, output_path: str, include_partial: bool) -> str | None:
        if include_partial:
            self.finalize_open()

        if len(self.collected_trajectories) == 0:
            self.saved_count = 0
            self.saved_motion_length = 0
            return None

        trajs = (
            self.collected_trajectories
            if self.target_trajectories <= 0
            else self.collected_trajectories[: self.target_trajectories]
        )
        motion_length = max(int(traj[self.action_key].shape[0]) for traj in trajs)
        payload_keys = [*self.component_keys, self.action_key]
        feature_shapes = {key: tuple(trajs[0][key].shape[1:]) for key in payload_keys}

        payload: dict[str, np.ndarray] = {
            "fps": np.array([self.fps], dtype=np.float32),
        }

        for key in payload_keys:
            stacked: list[np.ndarray] = []
            expected_feature_shape = feature_shapes[key]
            for traj in trajs:
                seq = traj[key].astype(np.float32)
                if seq.ndim < 2:
                    raise AssertionError(
                        f"Trajectory key '{key}' must include a time and feature dimension, got {seq.shape}."
                    )
                if tuple(seq.shape[1:]) != expected_feature_shape:
                    raise AssertionError(
                        f"Trajectory key '{key}' shape mismatch: expected tail {expected_feature_shape}, "
                        f"got {tuple(seq.shape[1:])}."
                    )
                if seq.shape[0] <= 0:
                    raise AssertionError(f"Trajectory key '{key}' has empty time dimension.")
                if seq.shape[0] < motion_length:
                    pad = np.repeat(seq[-1:, ...], motion_length - seq.shape[0], axis=0)
                    seq = np.concatenate([seq, pad], axis=0)
                elif seq.shape[0] > motion_length:
                    seq = seq[:motion_length]
                stacked.append(seq)
            payload[key] = np.stack(stacked, axis=0).astype(np.float32)

        expected_num_traj = len(trajs)
        expected_shapes = {
            key: (expected_num_traj, motion_length, *feature_shape) for key, feature_shape in feature_shapes.items()
        }
        fps = payload["fps"]
        if fps.ndim != 1 or fps.shape[0] != 1 or not np.isfinite(fps).all() or float(fps[0]) <= 0.0:
            raise AssertionError(f"Invalid `fps` payload shape/value: shape={fps.shape}, value={fps}")
        for key, value in payload.items():
            if key == "fps":
                continue
            if value.shape != expected_shapes[key]:
                raise AssertionError(
                    f"Payload key '{key}' shape mismatch: expected {expected_shapes[key]}, got {value.shape}"
                )

        os.makedirs(os.path.dirname(output_path) or ".", exist_ok=True)
        np.savez(output_path, **payload)
        self.saved_count = expected_num_traj
        self.saved_motion_length = motion_length
        return output_path
