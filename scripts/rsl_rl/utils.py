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

INITIAL_STATE_KEYS: tuple[str, ...] = tuple(f"initial_{key}" for key in DEFAULT_STATE_ACTION_KEYS)


def _infer_valid_lengths_from_states(states: np.ndarray, *, atol: float = 1.0e-8, rtol: float = 1.0e-8) -> np.ndarray:
    """Infer unpadded trajectory lengths from trailing repeated frames."""

    if states.ndim != 3:
        raise ValueError(f"Expected 3D array [num_traj, T, D], got {states.shape}.")
    num_traj, total_steps, _ = states.shape
    lengths = np.ones(num_traj, dtype=np.int32)
    for traj_idx in range(num_traj):
        traj = states[traj_idx]
        last_change = 0
        for step_idx in range(1, total_steps):
            if not np.allclose(traj[step_idx], traj[step_idx - 1], atol=atol, rtol=rtol):
                last_change = step_idx
        lengths[traj_idx] = last_change + 1
    return lengths


class StateActionTrajectoryRecorder:
    """Record robot state tensors and applied actions into padded [num_traj, T, D] NPZ payloads."""

    def __init__(
        self,
        num_envs: int,
        fps: float,
        target_trajectories: int,
        state_keys: Sequence[str] = DEFAULT_STATE_ACTION_KEYS,
        action_key: str = "actions",
        secondary_action_keys: Sequence[str] = (),
        record_initial_state: bool = True,
    ):
        self.num_envs = int(num_envs)
        self.fps = float(fps)
        self.target_trajectories = max(int(target_trajectories), 0)
        self.state_keys = tuple(state_keys)
        if len(self.state_keys) == 0:
            raise ValueError("state_keys must contain at least one key.")
        self.action_key = str(action_key)
        self.secondary_action_keys = tuple(secondary_action_keys)
        self.all_action_keys = (self.action_key, *self.secondary_action_keys)
        self.record_initial_state = bool(record_initial_state)
        self.component_keys = list(self.state_keys)
        self.traj_buffers: list[dict[str, list[np.ndarray]]] = [
            {key: [] for key in self.all_action_keys} for _ in range(self.num_envs)
        ]
        for env_id in range(self.num_envs):
            for key in self.component_keys:
                self.traj_buffers[env_id][key] = []
        self.pending_initial_states: list[dict[str, np.ndarray] | None] = [None for _ in range(self.num_envs)]
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

    def _snapshot_state_components(self, components: dict[str, torch.Tensor]) -> dict[str, np.ndarray]:
        snapshot: dict[str, np.ndarray] = {}
        for key in self.component_keys:
            component_np = components[key].detach().to("cpu", dtype=torch.float32).numpy()
            if component_np.ndim < 2 or component_np.shape[0] != self.num_envs:
                raise AssertionError(
                    f"Expected state component '{key}' shape [num_envs, ...], got {component_np.shape}."
                )
            snapshot[key] = component_np
        return snapshot

    def capture_initial_if_new_traj(self, state_components: dict[str, torch.Tensor]) -> None:
        """Capture pre-action state for envs that are starting a new trajectory."""

        if not self.record_initial_state:
            return
        self._validate_state_components(state_components)
        snapshot = self._snapshot_state_components(state_components)
        for env_id in range(self.num_envs):
            if len(self.traj_buffers[env_id][self.action_key]) == 0:
                self.pending_initial_states[env_id] = {
                    key: snapshot[key][env_id].astype(np.float32).copy() for key in self.component_keys
                }

    def append_step(
        self,
        state_components: dict[str, torch.Tensor],
        action_batch: torch.Tensor,
        *,
        auxiliary_actions: dict[str, torch.Tensor] | None = None,
    ):
        self._validate_state_components(state_components)
        action_np = action_batch.detach().to("cpu", dtype=torch.float32).numpy()
        if action_np.ndim != 2 or action_np.shape[0] != self.num_envs:
            raise AssertionError(f"Expected action shape [num_envs, A], got {action_np.shape}.")

        auxiliary_np: dict[str, np.ndarray] = {}
        if auxiliary_actions:
            for key in self.secondary_action_keys:
                if key not in auxiliary_actions:
                    raise AssertionError(f"Missing auxiliary action key '{key}' for trajectory logging.")
                aux = auxiliary_actions[key].detach().to("cpu", dtype=torch.float32).numpy()
                if aux.ndim != 2 or aux.shape[0] != self.num_envs:
                    raise AssertionError(f"Expected auxiliary action '{key}' shape [num_envs, A], got {aux.shape}.")
                auxiliary_np[key] = aux

        component_np_map = self._snapshot_state_components(state_components)

        for env_id in range(self.num_envs):
            for key in self.component_keys:
                self.traj_buffers[env_id][key].append(component_np_map[key][env_id].astype(np.float32).copy())
            self.traj_buffers[env_id][self.action_key].append(action_np[env_id].astype(np.float32).copy())
            for key in self.secondary_action_keys:
                self.traj_buffers[env_id][key].append(auxiliary_np[key][env_id].astype(np.float32).copy())

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
            self.pending_initial_states[env_id] = None
            return 0

        keys = [*self.component_keys, *self.all_action_keys]
        traj = {key: np.stack(self.traj_buffers[env_id][key], axis=0).astype(np.float32) for key in keys}
        if self.record_initial_state:
            pending = self.pending_initial_states[env_id]
            if pending is None:
                raise AssertionError(
                    f"Missing initial state snapshot for env {env_id}. "
                    "Call capture_initial_if_new_traj() before the first env.step()."
                )
            for state_key in self.component_keys:
                traj[f"initial_{state_key}"] = pending[state_key].astype(np.float32).copy()
        self.collected_trajectories.append(traj)
        self.traj_buffers[env_id] = {key: [] for key in keys}
        self.pending_initial_states[env_id] = None
        return 1

    def save_dataset(
        self,
        output_path: str,
        include_partial: bool,
        metadata: dict[str, object] | None = None,
    ) -> str | None:
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
        time_series_keys = [*self.component_keys, *self.all_action_keys]
        feature_shapes = {key: tuple(trajs[0][key].shape[1:]) for key in time_series_keys}

        payload: dict[str, np.ndarray] = {
            "fps": np.array([self.fps], dtype=np.float32),
        }

        for key in time_series_keys:
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

        if self.record_initial_state:
            for state_key in self.component_keys:
                initial_key = f"initial_{state_key}"
                initial_rows = [traj[initial_key] for traj in trajs]
                payload[initial_key] = np.stack(initial_rows, axis=0).astype(np.float32)

        raw_valid_lengths = np.asarray([int(traj[self.action_key].shape[0]) for traj in trajs], dtype=np.int32)
        payload["valid_lengths"] = raw_valid_lengths

        if metadata:
            for key, value in metadata.items():
                if isinstance(value, str):
                    payload[key] = np.array(value, dtype=np.object_)
                elif isinstance(value, (list, tuple)):
                    payload[key] = np.asarray(list(value), dtype=np.object_)
                elif isinstance(value, np.ndarray):
                    payload[key] = np.asarray(value)
                else:
                    payload[key] = np.asarray(value, dtype=np.float32)

        expected_num_traj = len(trajs)
        expected_shapes = {
            key: (expected_num_traj, motion_length, *feature_shape) for key, feature_shape in feature_shapes.items()
        }
        fps = payload["fps"]
        if fps.ndim != 1 or fps.shape[0] != 1 or not np.isfinite(fps).all() or float(fps[0]) <= 0.0:
            raise AssertionError(f"Invalid `fps` payload shape/value: shape={fps.shape}, value={fps}")
        for key, value in payload.items():
            if key in {"fps", "valid_lengths"} or key.startswith("initial_"):
                continue
            if key in metadata if metadata else {}:
                continue
            if key in expected_shapes and value.shape != expected_shapes[key]:
                raise AssertionError(
                    f"Payload key '{key}' shape mismatch: expected {expected_shapes[key]}, got {value.shape}"
                )

        os.makedirs(os.path.dirname(output_path) or ".", exist_ok=True)
        np.savez(output_path, **payload)
        self.saved_count = expected_num_traj
        self.saved_motion_length = motion_length
        return output_path
