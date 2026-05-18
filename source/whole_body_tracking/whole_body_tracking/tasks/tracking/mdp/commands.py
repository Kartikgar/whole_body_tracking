from __future__ import annotations

import math
import numpy as np
import os
import torch
from collections.abc import Sequence
from dataclasses import MISSING
from typing import TYPE_CHECKING

from isaaclab.assets import Articulation
from isaaclab.managers import CommandTerm, CommandTermCfg
from isaaclab.markers import VisualizationMarkers, VisualizationMarkersCfg
from isaaclab.markers.config import FRAME_MARKER_CFG
from isaaclab.utils import configclass
from isaaclab.utils.math import (
    quat_apply,
    quat_error_magnitude,
    quat_from_euler_xyz,
    quat_inv,
    quat_mul,
    sample_uniform,
    yaw_quat,
)

if TYPE_CHECKING:
    from isaaclab.envs import ManagerBasedRLEnv


class MotionLoader:
    _ACTION_TRAJ_KEYS = ("actions", "action", "joint_actions", "joint_action")
    _VECTOR_TRAJ_KEYS = ("joint_pos", "joint_vel") + _ACTION_TRAJ_KEYS
    _BODY_TRAJ_KEYS = ("body_pos_w", "body_quat_w", "body_lin_vel_w", "body_ang_vel_w")
    _TOP_LEVEL_METADATA_KEYS = {"fps", "format", "motion_keys", "num_motions", "source_files"}
    _REQUIRED_KEYS = ("joint_pos", "joint_vel", "body_pos_w", "body_quat_w", "body_lin_vel_w", "body_ang_vel_w")

    def __init__(self, motion_file: str, body_indexes: Sequence[int], device: str = "cpu"):
        if not os.path.isfile(motion_file):
            raise FileNotFoundError(f"Invalid file path: {motion_file}")

        self.device = torch.device(device)
        self._body_indexes = (
            body_indexes.to(device=self.device, dtype=torch.long)
            if isinstance(body_indexes, torch.Tensor)
            else torch.tensor(body_indexes, dtype=torch.long, device=self.device)
        )

        with np.load(motion_file, allow_pickle=True) as data:
            if "joint_pos" in data.files:
                self._load_stacked_motion(data, motion_file)
            else:
                motion_entries, fps_values = self._load_motion_entries_from_data(data)
                self._load_motion_entries_as_flat_buffers(motion_entries, fps_values, motion_file)

        self.time_step_total = int(self.trajectory_time_step_total.max().item())

    def _set_fps(self, fps_values: list[float], motion_file: str):
        if len(fps_values) == 0:
            self.fps = torch.zeros(self.num_trajectories, dtype=torch.float32, device=self.device)
        elif len(fps_values) == 1 and self.num_trajectories > 1:
            self.fps = torch.full(
                (self.num_trajectories,), float(fps_values[0]), dtype=torch.float32, device=self.device
            )
        elif len(fps_values) == self.num_trajectories:
            self.fps = torch.tensor(fps_values, dtype=torch.float32, device=self.device)
        else:
            raise ValueError(
                f"Unexpected `fps` shape for '{motion_file}'. Received {len(fps_values)} values for "
                f"{self.num_trajectories} trajectories."
            )

    def _set_trajectory_lengths(self, trajectory_lengths: Sequence[int], motion_file: str):
        self.num_trajectories = len(trajectory_lengths)
        if self.num_trajectories == 0:
            raise ValueError(f"Motion file '{motion_file}' did not contain any trajectories.")

        self.trajectory_time_step_total = torch.tensor(trajectory_lengths, dtype=torch.long, device=self.device)
        if torch.any(self.trajectory_time_step_total <= 0):
            raise ValueError(f"All trajectories in '{motion_file}' must have at least one frame.")

        offsets = [0]
        for length in self.trajectory_time_step_total[:-1].tolist():
            offsets.append(offsets[-1] + int(length))
        self.trajectory_start_time_step = torch.tensor(offsets, dtype=torch.long, device=self.device)

    @staticmethod
    def _as_float32_array(value: np.ndarray) -> np.ndarray:
        return np.asarray(value, dtype=np.float32)

    @classmethod
    def _normalize_stacked_vector_array(cls, key: str, value: np.ndarray) -> np.ndarray:
        arr = cls._as_float32_array(value)
        if arr.ndim == 2:
            arr = arr[None, ...]
        if arr.ndim != 3:
            raise ValueError(f"Expected `{key}` shape [T, D] or [N_traj, T, D], got {arr.shape}.")
        return arr

    @classmethod
    def _normalize_stacked_body_array(cls, key: str, value: np.ndarray, expected_tail_dim: int) -> np.ndarray:
        arr = cls._as_float32_array(value)
        if arr.ndim == 3:
            arr = arr[None, ...]
        if arr.ndim != 4 or arr.shape[-1] != expected_tail_dim:
            raise ValueError(
                f"Expected `{key}` shape [T, B, {expected_tail_dim}] or [N_traj, T, B, {expected_tail_dim}], "
                f"got {arr.shape}."
            )
        return arr

    def _motion_body_indexes_for_count(self, num_motion_bodies: int) -> np.ndarray:
        if num_motion_bodies == int(self._body_indexes.numel()):
            return np.arange(num_motion_bodies, dtype=np.int64)

        body_indexes = self._body_indexes.detach().to(device="cpu", dtype=torch.long).numpy()
        max_body_index = int(body_indexes.max()) if body_indexes.size > 0 else -1
        if max_body_index >= num_motion_bodies:
            raise ValueError(
                f"Motion body count mismatch. Requested body index {max_body_index}, but motion has only "
                f"{num_motion_bodies} bodies."
            )
        return body_indexes

    def _to_device_tensor(self, arr: np.ndarray) -> torch.Tensor:
        return torch.tensor(np.ascontiguousarray(arr), dtype=torch.float32, device=self.device)

    def _load_stacked_motion(self, data: np.lib.npyio.NpzFile, motion_file: str):
        joint_pos = self._normalize_stacked_vector_array("joint_pos", data["joint_pos"])
        num_trajectories, time_steps, joint_dim = map(int, joint_pos.shape)
        self._set_trajectory_lengths([time_steps] * num_trajectories, motion_file)
        self._set_fps(self._extract_fps_values_from_stacked(data, num_trajectories), motion_file)
        self.joint_pos = self._to_device_tensor(joint_pos.reshape(-1, joint_dim))
        del joint_pos

        joint_vel = self._normalize_stacked_vector_array("joint_vel", data["joint_vel"])
        if tuple(joint_vel.shape) != (num_trajectories, time_steps, joint_dim):
            raise ValueError(
                f"`joint_vel` shape {joint_vel.shape} must match `joint_pos` shape "
                f"({num_trajectories}, {time_steps}, {joint_dim})."
            )
        self.joint_vel = self._to_device_tensor(joint_vel.reshape(-1, joint_dim))
        del joint_vel

        self._body_pos_w = self._load_stacked_body_tensor(data, "body_pos_w", (num_trajectories, time_steps), 3)
        self._body_quat_w = self._load_stacked_body_tensor(data, "body_quat_w", (num_trajectories, time_steps), 4)
        self._body_lin_vel_w = self._load_stacked_body_tensor(
            data, "body_lin_vel_w", (num_trajectories, time_steps), 3
        )
        self._body_ang_vel_w = self._load_stacked_body_tensor(
            data, "body_ang_vel_w", (num_trajectories, time_steps), 3
        )

        action_key = next((key for key in self._ACTION_TRAJ_KEYS if key in data.files), None)
        self.has_joint_action = action_key is not None
        if action_key is None:
            self.joint_action = torch.zeros_like(self.joint_pos)
            return

        action = self._normalize_stacked_vector_array(action_key, data[action_key])
        if tuple(action.shape) != (num_trajectories, time_steps, joint_dim):
            raise ValueError(
                f"`{action_key}` shape {action.shape} must match `joint_pos` shape "
                f"({num_trajectories}, {time_steps}, {joint_dim})."
            )
        self.joint_action = self._to_device_tensor(action.reshape(-1, joint_dim))

    def _load_stacked_body_tensor(
        self,
        data: np.lib.npyio.NpzFile,
        key: str,
        expected_shape: tuple[int, int],
        expected_tail_dim: int,
    ) -> torch.Tensor:
        arr = self._normalize_stacked_body_array(key, data[key], expected_tail_dim=expected_tail_dim)
        num_trajectories, time_steps = expected_shape
        if tuple(arr.shape[:2]) != expected_shape:
            raise ValueError(
                f"`{key}` has leading shape {arr.shape[:2]}, expected ({num_trajectories}, {time_steps})."
            )
        motion_body_indexes = self._motion_body_indexes_for_count(int(arr.shape[2]))
        arr = arr[:, :, motion_body_indexes, :]
        return self._to_device_tensor(
            arr.reshape(num_trajectories * time_steps, len(motion_body_indexes), expected_tail_dim)
        )

    def _build_flat_buffers(self):
        offsets = [0]
        for length in self.trajectory_time_step_total[:-1].tolist():
            offsets.append(offsets[-1] + int(length))
        self.trajectory_start_time_step = torch.tensor(offsets, dtype=torch.long, device=self.device)
        # import ipdb;ipdb.set_trace()
        self.joint_pos = torch.cat([entry["joint_pos"] for entry in self._trajectory_data], dim=0)
        self.joint_vel = torch.cat([entry["joint_vel"] for entry in self._trajectory_data], dim=0)
        self._body_pos_w = torch.cat([entry["body_pos_w"] for entry in self._trajectory_data], dim=0)
        self._body_quat_w = torch.cat([entry["body_quat_w"] for entry in self._trajectory_data], dim=0)
        self._body_lin_vel_w = torch.cat([entry["body_lin_vel_w"] for entry in self._trajectory_data], dim=0)
        self._body_ang_vel_w = torch.cat([entry["body_ang_vel_w"] for entry in self._trajectory_data], dim=0)

        joint_dim = int(self.joint_pos.shape[1])
        if self.has_joint_action:
            action_chunks = []
            for entry in self._trajectory_data:
                if "joint_action" in entry:
                    chunk = entry["joint_action"]
                    if chunk.shape[1] != joint_dim:
                        raise ValueError(
                            f"Motion action dimension mismatch: expected {joint_dim}, got {int(chunk.shape[1])}."
                        )
                else:
                    chunk = torch.zeros(entry["joint_pos"].shape[0], joint_dim, dtype=torch.float32, device=self.device)
                action_chunks.append(chunk)
            self.joint_action = torch.cat(action_chunks, dim=0)
        else:
            self.joint_action = torch.zeros_like(self.joint_pos)

    @classmethod
    def _infer_num_trajectories_from_stacked(cls, data: np.lib.npyio.NpzFile) -> int:
        candidates: list[int] = []
        for key in cls._REQUIRED_KEYS:
            if key not in data.files:
                continue
            arr = np.asarray(data[key])
            if key in cls._VECTOR_TRAJ_KEYS:
                if arr.ndim == 3:
                    candidates.append(int(arr.shape[0]))
                elif arr.ndim == 2:
                    candidates.append(1)
            elif key in cls._BODY_TRAJ_KEYS:
                if arr.ndim == 4:
                    candidates.append(int(arr.shape[0]))
                elif arr.ndim == 3:
                    candidates.append(1)

        if len(candidates) == 0:
            raise ValueError(
                "Could not infer number of trajectories from stacked format. "
                f"Expected at least one of keys: {sorted(cls._REQUIRED_KEYS)}"
            )
        if any(candidate != candidates[0] for candidate in candidates[1:]):
            raise ValueError(f"Inconsistent trajectory counts in stacked arrays: {candidates}")
        return candidates[0]

    @classmethod
    def _select_stacked_trajectory_array(
        cls, key: str, arr: np.ndarray, num_trajectories: int, trajectory_idx: int
    ) -> np.ndarray | None:
        value = np.asarray(arr)

        if key in cls._VECTOR_TRAJ_KEYS:
            if value.ndim == 2 and num_trajectories == 1:
                return np.array(value, copy=True)
            if value.ndim == 3 and int(value.shape[0]) == int(num_trajectories):
                return np.array(value[trajectory_idx], copy=True)
            return None

        if key in cls._BODY_TRAJ_KEYS:
            if value.ndim == 3 and num_trajectories == 1:
                return np.array(value, copy=True)
            if value.ndim == 4 and int(value.shape[0]) == int(num_trajectories):
                return np.array(value[trajectory_idx], copy=True)
            return None

        return None

    @classmethod
    def _resolve_motion_keys(cls, data: np.lib.npyio.NpzFile) -> list[str]:
        if "motion_keys" in data.files:
            keys = [str(key) for key in np.asarray(data["motion_keys"]).reshape(-1).tolist()]
            if len(keys) == 0:
                raise ValueError("`motion_keys` is present but empty.")
            return keys

        motion_keys = [key for key in data.files if key.startswith("motion")]
        if len(motion_keys) == 0:
            raise ValueError("Could not find any `motion{i}` keys in the dataset.")

        def sort_key(key: str) -> tuple[int, str]:
            suffix = key[len("motion") :]
            return (int(suffix), key) if suffix.isdigit() else (10**9, key)

        return sorted(motion_keys, key=sort_key)

    @staticmethod
    def _extract_motion_dict(key: str, raw_value: np.ndarray) -> dict:
        value = raw_value
        if isinstance(value, np.ndarray):
            if value.dtype != object:
                raise ValueError(f"Expected `{key}` to be an object array containing a dict, got dtype={value.dtype}.")
            if value.size != 1:
                raise ValueError(f"Expected `{key}` object array size=1, got size={value.size}.")
            value = value.reshape(()).item()
        if not isinstance(value, dict):
            raise ValueError(f"Expected `{key}` payload to be a dict, got {type(value)}.")
        return value

    @classmethod
    def _resolve_action_key(cls, motion: dict[str, np.ndarray]) -> str | None:
        for key in cls._ACTION_TRAJ_KEYS:
            if key in motion:
                return key
        return None

    @classmethod
    def _validate_motion_dict(cls, motion: dict[str, np.ndarray], source_key: str):
        missing = [key for key in cls._REQUIRED_KEYS if key not in motion]
        if missing:
            raise ValueError(f"Motion entry `{source_key}` is missing required keys: {missing}")

        joint_pos = np.asarray(motion["joint_pos"])
        joint_vel = np.asarray(motion["joint_vel"])
        if joint_pos.ndim != 2:
            raise ValueError(f"`{source_key}.joint_pos` must be rank-2 [T, joints], got shape={joint_pos.shape}.")
        if joint_vel.ndim != 2:
            raise ValueError(f"`{source_key}.joint_vel` must be rank-2 [T, joints], got shape={joint_vel.shape}.")
        if joint_pos.shape != joint_vel.shape:
            raise ValueError(
                f"`{source_key}` has mismatched joint shapes: joint_pos={joint_pos.shape}, joint_vel={joint_vel.shape}."
            )

        time_steps = int(joint_pos.shape[0])
        for key in cls._BODY_TRAJ_KEYS:
            arr = np.asarray(motion[key])
            if arr.ndim != 3:
                raise ValueError(f"`{source_key}.{key}` must be rank-3 [T, bodies, ...], got shape={arr.shape}.")
            if int(arr.shape[0]) != time_steps:
                raise ValueError(
                    f"`{source_key}.{key}` has time length {arr.shape[0]} but expected {time_steps} to match `joint_pos`."
                )

        action_key = cls._resolve_action_key(motion)
        if action_key is not None:
            action = np.asarray(motion[action_key])
            if action.ndim != 2:
                raise ValueError(
                    f"`{source_key}.{action_key}` must be rank-2 [T, joints], got shape={action.shape}."
                )
            if int(action.shape[0]) != time_steps:
                raise ValueError(
                    f"`{source_key}.{action_key}` has time length {action.shape[0]} but expected {time_steps}."
                )

    @classmethod
    def _extract_fps_values_from_stacked(cls, data: np.lib.npyio.NpzFile, num_trajectories: int) -> list[float]:
        if "fps" not in data.files:
            return []
        fps = np.asarray(data["fps"]).reshape(-1)
        if fps.size == 1:
            return [float(fps[0])] * num_trajectories
        if fps.size == num_trajectories:
            return [float(v) for v in fps.tolist()]
        raise ValueError(
            f"Invalid `fps` array shape for stacked motion file: got {fps.shape}, "
            f"expected scalar or length {num_trajectories}."
        )

    @classmethod
    def _load_motion_entries(cls, motion_file: str) -> tuple[list[dict[str, np.ndarray]], list[float]]:
        with np.load(motion_file, allow_pickle=True) as data:
            if "joint_pos" in data.files:
                num_trajectories = cls._infer_num_trajectories_from_stacked(data)
                fps_values = cls._extract_fps_values_from_stacked(data, num_trajectories)
                entries = []
                # Compatibility path for callers that still want per-trajectory numpy dictionaries.
                # MotionLoader itself uses _load_stacked_motion() to avoid repeatedly materializing large arrays.
                stacked_arrays = {
                    key: np.asarray(data[key]) for key in cls._REQUIRED_KEYS + cls._ACTION_TRAJ_KEYS if key in data.files
                }
                for trajectory_idx in range(num_trajectories):
                    motion = {}
                    for key, arr in stacked_arrays.items():
                        selected = cls._select_stacked_trajectory_array(
                            key=key,
                            arr=arr,
                            num_trajectories=num_trajectories,
                            trajectory_idx=trajectory_idx,
                        )
                        if selected is not None:
                            motion[key] = selected
                    cls._validate_motion_dict(motion, source_key=f"trajectory{trajectory_idx}")
                    entries.append(motion)
                return entries, fps_values

            return cls._load_motion_entries_from_data(data)

    @classmethod
    def _load_motion_entries_from_data(cls, data: np.lib.npyio.NpzFile) -> tuple[list[dict[str, np.ndarray]], list[float]]:
        motion_keys = cls._resolve_motion_keys(data)
        entries = []
        fps_values = []
        top_level_fps = float(np.asarray(data["fps"]).reshape(-1)[0]) if "fps" in data.files else None
        for motion_key in motion_keys:
            motion_dict = cls._extract_motion_dict(motion_key, data[motion_key])
            motion = {key: np.asarray(value) for key, value in motion_dict.items()}
            cls._validate_motion_dict(motion, source_key=motion_key)
            entries.append(motion)
            if "fps" in motion:
                fps_values.append(float(np.asarray(motion["fps"]).reshape(-1)[0]))
            elif top_level_fps is not None:
                fps_values.append(top_level_fps)
        return entries, fps_values

    def _load_motion_entries_as_flat_buffers(
        self, motion_entries: list[dict[str, np.ndarray]], fps_values: list[float], motion_file: str
    ):
        trajectory_data = [self._to_trajectory_tensors(motion) for motion in motion_entries]
        self._set_trajectory_lengths([int(entry["joint_pos"].shape[0]) for entry in trajectory_data], motion_file)
        self._set_fps(fps_values, motion_file)

        self.joint_pos = torch.cat([entry["joint_pos"] for entry in trajectory_data], dim=0)
        self.joint_vel = torch.cat([entry["joint_vel"] for entry in trajectory_data], dim=0)
        self._body_pos_w = torch.cat([entry["body_pos_w"] for entry in trajectory_data], dim=0)
        self._body_quat_w = torch.cat([entry["body_quat_w"] for entry in trajectory_data], dim=0)
        self._body_lin_vel_w = torch.cat([entry["body_lin_vel_w"] for entry in trajectory_data], dim=0)
        self._body_ang_vel_w = torch.cat([entry["body_ang_vel_w"] for entry in trajectory_data], dim=0)

        joint_dim = int(self.joint_pos.shape[1])
        self.has_joint_action = any("joint_action" in entry for entry in trajectory_data)
        if self.has_joint_action:
            action_chunks = []
            for entry in trajectory_data:
                if "joint_action" in entry:
                    chunk = entry["joint_action"]
                    if chunk.shape[1] != joint_dim:
                        raise ValueError(
                            f"Motion action dimension mismatch: expected {joint_dim}, got {int(chunk.shape[1])}."
                        )
                else:
                    chunk = torch.zeros(entry["joint_pos"].shape[0], joint_dim, dtype=torch.float32, device=self.device)
                action_chunks.append(chunk)
            self.joint_action = torch.cat(action_chunks, dim=0)
        else:
            self.joint_action = torch.zeros_like(self.joint_pos)

    def _to_trajectory_tensors(self, motion: dict[str, np.ndarray]) -> dict[str, torch.Tensor]:
        joint_pos = torch.tensor(np.asarray(motion["joint_pos"]), dtype=torch.float32, device=self.device)
        joint_vel = torch.tensor(np.asarray(motion["joint_vel"]), dtype=torch.float32, device=self.device)
        body_pos_w = torch.tensor(np.asarray(motion["body_pos_w"]), dtype=torch.float32, device=self.device)
        body_quat_w = torch.tensor(np.asarray(motion["body_quat_w"]), dtype=torch.float32, device=self.device)
        body_lin_vel_w = torch.tensor(np.asarray(motion["body_lin_vel_w"]), dtype=torch.float32, device=self.device)
        body_ang_vel_w = torch.tensor(np.asarray(motion["body_ang_vel_w"]), dtype=torch.float32, device=self.device)
        num_motion_bodies = int(body_pos_w.shape[1])
        if (
            int(body_quat_w.shape[1]) != num_motion_bodies
            or int(body_lin_vel_w.shape[1]) != num_motion_bodies
            or int(body_ang_vel_w.shape[1]) != num_motion_bodies
        ):
            raise ValueError(
                "Inconsistent body count across motion tensors: "
                f"body_pos_w={tuple(body_pos_w.shape)}, body_quat_w={tuple(body_quat_w.shape)}, "
                f"body_lin_vel_w={tuple(body_lin_vel_w.shape)}, body_ang_vel_w={tuple(body_ang_vel_w.shape)}."
            )

        # If the motion already stores exactly the configured body subset/order, index directly by [0..N-1].
        # Otherwise, keep legacy behavior that indexes a larger motion body set using robot body indices.
        if num_motion_bodies == int(self._body_indexes.numel()):
            motion_body_indexes = torch.arange(num_motion_bodies, dtype=torch.long, device=self.device)
        else:
            max_body_index = int(self._body_indexes.max().item()) if self._body_indexes.numel() > 0 else -1
            if max_body_index >= num_motion_bodies:
                raise ValueError(
                    f"Motion body count mismatch. Requested body index {max_body_index}, but motion has only "
                    f"{num_motion_bodies} bodies."
                )
            motion_body_indexes = self._body_indexes

        trajectory = {
            "joint_pos": joint_pos,
            "joint_vel": joint_vel,
            "body_pos_w": body_pos_w.index_select(dim=1, index=motion_body_indexes),
            "body_quat_w": body_quat_w.index_select(dim=1, index=motion_body_indexes),
            "body_lin_vel_w": body_lin_vel_w.index_select(dim=1, index=motion_body_indexes),
            "body_ang_vel_w": body_ang_vel_w.index_select(dim=1, index=motion_body_indexes),
        }

        action_key = self._resolve_action_key(motion)
        if action_key is not None:
            trajectory["joint_action"] = torch.tensor(
                np.asarray(motion[action_key]), dtype=torch.float32, device=self.device
            )

        return trajectory

    def _get_flat_indexes(self, trajectory_ids: torch.Tensor, time_steps: torch.Tensor) -> torch.Tensor:
        traj_ids = trajectory_ids.to(device=self.device, dtype=torch.long)
        clamped_time_steps = torch.clamp(time_steps.to(device=self.device, dtype=torch.long), min=0)
        lengths = self.trajectory_time_step_total[traj_ids]
        clamped_time_steps = torch.minimum(clamped_time_steps, torch.clamp(lengths - 1, min=0))
        return self.trajectory_start_time_step[traj_ids] + clamped_time_steps

    def get_joint_pos(self, trajectory_ids: torch.Tensor, time_steps: torch.Tensor) -> torch.Tensor:
        return self.joint_pos[self._get_flat_indexes(trajectory_ids, time_steps)]

    def get_joint_vel(self, trajectory_ids: torch.Tensor, time_steps: torch.Tensor) -> torch.Tensor:
        return self.joint_vel[self._get_flat_indexes(trajectory_ids, time_steps)]

    def get_body_pos_w(self, trajectory_ids: torch.Tensor, time_steps: torch.Tensor) -> torch.Tensor:
        return self._body_pos_w[self._get_flat_indexes(trajectory_ids, time_steps)]

    def get_body_quat_w(self, trajectory_ids: torch.Tensor, time_steps: torch.Tensor) -> torch.Tensor:
        return self._body_quat_w[self._get_flat_indexes(trajectory_ids, time_steps)]

    def get_body_lin_vel_w(self, trajectory_ids: torch.Tensor, time_steps: torch.Tensor) -> torch.Tensor:
        return self._body_lin_vel_w[self._get_flat_indexes(trajectory_ids, time_steps)]

    def get_body_ang_vel_w(self, trajectory_ids: torch.Tensor, time_steps: torch.Tensor) -> torch.Tensor:
        return self._body_ang_vel_w[self._get_flat_indexes(trajectory_ids, time_steps)]

    def get_joint_action(self, trajectory_ids: torch.Tensor, time_steps: torch.Tensor) -> torch.Tensor:
        if not self.has_joint_action:
            raise RuntimeError(
                "Motion file does not contain action keys (`action`, `actions`, `joint_action`, `joint_actions`)."
            )
        return self.joint_action[self._get_flat_indexes(trajectory_ids, time_steps)]

    def get_trajectory_data(self, trajectory_idx: int) -> dict[str, torch.Tensor]:
        idx = int(trajectory_idx)
        if idx < 0:
            idx += self.num_trajectories
        if idx < 0 or idx >= self.num_trajectories:
            raise IndexError(
                f"Invalid trajectory index {trajectory_idx}. Expected in [0, {self.num_trajectories - 1}] "
                "(or negative equivalent)."
            )
        start = int(self.trajectory_start_time_step[idx].item())
        length = int(self.trajectory_time_step_total[idx].item())
        end = start + length
        data = {
            "joint_pos": self.joint_pos[start:end],
            "joint_vel": self.joint_vel[start:end],
            "body_pos_w": self._body_pos_w[start:end],
            "body_quat_w": self._body_quat_w[start:end],
            "body_lin_vel_w": self._body_lin_vel_w[start:end],
            "body_ang_vel_w": self._body_ang_vel_w[start:end],
        }
        if self.has_joint_action:
            data["actions"] = self.joint_action[start:end]
            data["action"] = self.joint_action[start:end]
            data["joint_action"] = self.joint_action[start:end]
        return data


class MotionCommand(CommandTerm):
    cfg: MotionCommandCfg

    def __init__(self, cfg: MotionCommandCfg, env: ManagerBasedRLEnv):
        super().__init__(cfg, env)

        if len(self.cfg.body_names) == 0:
            raise ValueError("`MotionCommandCfg.body_names` must contain at least one body.")

        self.robot: Articulation = env.scene[cfg.asset_name]
        self.robot_anchor_body_index = self.robot.body_names.index(self.cfg.anchor_body_name)
        self.motion_anchor_body_index = self.cfg.body_names.index(self.cfg.anchor_body_name)
        self.heading_body_name = (
            self.cfg.relative_heading_body_name if self.cfg.relative_heading_body_name is not None else self.cfg.body_names[0]
        )
        if self.heading_body_name not in self.cfg.body_names:
            raise ValueError(f"`relative_heading_body_name={self.heading_body_name}` must exist in `body_names`.")
        self.robot_heading_body_index = self.robot.body_names.index(self.heading_body_name)
        self.motion_heading_body_index = self.cfg.body_names.index(self.heading_body_name)
        self.body_indexes = torch.tensor(
            self.robot.find_bodies(self.cfg.body_names, preserve_order=True)[0], dtype=torch.long, device=self.device
        )

        self.motion = MotionLoader(self.cfg.motion_file, self.body_indexes, device=self.device)
        self.trajectory_ids = torch.zeros(self.num_envs, dtype=torch.long, device=self.device)
        self.time_steps = torch.zeros(self.num_envs, dtype=torch.long, device=self.device)
        self.body_pos_relative_w = torch.zeros(self.num_envs, len(cfg.body_names), 3, device=self.device)
        self.body_quat_relative_w = torch.zeros(self.num_envs, len(cfg.body_names), 4, device=self.device)
        self.body_quat_relative_w[:, :, 0] = 1.0

        self.bin_count = int(self.motion.time_step_total // (1 / (env.cfg.decimation * env.cfg.sim.dt))) + 1
        self.bin_failed_count = torch.zeros(self.bin_count, dtype=torch.float, device=self.device)
        self._current_bin_failed = torch.zeros(self.bin_count, dtype=torch.float, device=self.device)
        self.kernel = torch.tensor(
            [self.cfg.adaptive_lambda**i for i in range(self.cfg.adaptive_kernel_size)], device=self.device
        )
        self.kernel = self.kernel / self.kernel.sum()

        self.metrics["error_anchor_pos"] = torch.zeros(self.num_envs, device=self.device)
        self.metrics["error_anchor_rot"] = torch.zeros(self.num_envs, device=self.device)
        self.metrics["error_anchor_lin_vel"] = torch.zeros(self.num_envs, device=self.device)
        self.metrics["error_anchor_ang_vel"] = torch.zeros(self.num_envs, device=self.device)
        self.metrics["error_body_pos"] = torch.zeros(self.num_envs, device=self.device)
        self.metrics["error_body_rot"] = torch.zeros(self.num_envs, device=self.device)
        self.metrics["error_joint_pos"] = torch.zeros(self.num_envs, device=self.device)
        self.metrics["error_joint_vel"] = torch.zeros(self.num_envs, device=self.device)
        self.metrics["sampling_entropy"] = torch.zeros(self.num_envs, device=self.device)
        self.metrics["sampling_top1_prob"] = torch.zeros(self.num_envs, device=self.device)
        self.metrics["sampling_top1_bin"] = torch.zeros(self.num_envs, device=self.device)

    @property
    def command(self) -> torch.Tensor:  # TODO Consider again if this is the best observation
        return torch.cat([self.joint_pos, self.joint_vel], dim=1)

    @property
    def joint_pos(self) -> torch.Tensor:
        return self.motion.get_joint_pos(self.trajectory_ids, self.time_steps)

    @property
    def joint_vel(self) -> torch.Tensor:
        return self.motion.get_joint_vel(self.trajectory_ids, self.time_steps)

    @property
    def body_pos_w(self) -> torch.Tensor:
        return self.motion.get_body_pos_w(self.trajectory_ids, self.time_steps) + self._env.scene.env_origins[:, None, :]

    @property
    def body_quat_w(self) -> torch.Tensor:
        return self.motion.get_body_quat_w(self.trajectory_ids, self.time_steps)

    @property
    def body_lin_vel_w(self) -> torch.Tensor:
        return self.motion.get_body_lin_vel_w(self.trajectory_ids, self.time_steps)

    @property
    def body_ang_vel_w(self) -> torch.Tensor:
        return self.motion.get_body_ang_vel_w(self.trajectory_ids, self.time_steps)

    @property
    def anchor_pos_w(self) -> torch.Tensor:
        body_pos = self.motion.get_body_pos_w(self.trajectory_ids, self.time_steps)
        return body_pos[:, self.motion_anchor_body_index] + self._env.scene.env_origins

    @property
    def anchor_quat_w(self) -> torch.Tensor:
        body_quat = self.motion.get_body_quat_w(self.trajectory_ids, self.time_steps)
        return body_quat[:, self.motion_anchor_body_index]

    @property
    def heading_quat_w(self) -> torch.Tensor:
        body_quat = self.motion.get_body_quat_w(self.trajectory_ids, self.time_steps)
        return body_quat[:, self.motion_heading_body_index]

    @property
    def anchor_lin_vel_w(self) -> torch.Tensor:
        body_lin_vel = self.motion.get_body_lin_vel_w(self.trajectory_ids, self.time_steps)
        return body_lin_vel[:, self.motion_anchor_body_index]

    @property
    def anchor_ang_vel_w(self) -> torch.Tensor:
        body_ang_vel = self.motion.get_body_ang_vel_w(self.trajectory_ids, self.time_steps)
        return body_ang_vel[:, self.motion_anchor_body_index]

    @property
    def has_joint_action(self) -> bool:
        return self.motion.has_joint_action

    @property
    def joint_action(self) -> torch.Tensor:
        if not self.has_joint_action:
            raise RuntimeError(
                "Motion file does not contain action keys (`action`, `actions`, `joint_action`, `joint_actions`)."
            )
        return self.motion.get_joint_action(self.trajectory_ids, self.time_steps)

    @property
    def robot_joint_pos(self) -> torch.Tensor:
        return self.robot.data.joint_pos

    @property
    def robot_joint_vel(self) -> torch.Tensor:
        return self.robot.data.joint_vel

    @property
    def robot_body_pos_w(self) -> torch.Tensor:
        return self.robot.data.body_pos_w[:, self.body_indexes]

    @property
    def robot_body_quat_w(self) -> torch.Tensor:
        return self.robot.data.body_quat_w[:, self.body_indexes]

    @property
    def robot_body_lin_vel_w(self) -> torch.Tensor:
        return self.robot.data.body_lin_vel_w[:, self.body_indexes]

    @property
    def robot_body_ang_vel_w(self) -> torch.Tensor:
        return self.robot.data.body_ang_vel_w[:, self.body_indexes]

    @property
    def robot_anchor_pos_w(self) -> torch.Tensor:
        return self.robot.data.body_pos_w[:, self.robot_anchor_body_index]

    @property
    def robot_anchor_quat_w(self) -> torch.Tensor:
        return self.robot.data.body_quat_w[:, self.robot_anchor_body_index]

    @property
    def robot_heading_quat_w(self) -> torch.Tensor:
        return self.robot.data.body_quat_w[:, self.robot_heading_body_index]

    @property
    def robot_anchor_lin_vel_w(self) -> torch.Tensor:
        return self.robot.data.body_lin_vel_w[:, self.robot_anchor_body_index]

    @property
    def robot_anchor_ang_vel_w(self) -> torch.Tensor:
        return self.robot.data.body_ang_vel_w[:, self.robot_anchor_body_index]

    def _update_metrics(self):
        self.metrics["error_anchor_pos"] = torch.norm(self.anchor_pos_w - self.robot_anchor_pos_w, dim=-1)
        self.metrics["error_anchor_rot"] = quat_error_magnitude(self.anchor_quat_w, self.robot_anchor_quat_w)
        self.metrics["error_anchor_lin_vel"] = torch.norm(self.anchor_lin_vel_w - self.robot_anchor_lin_vel_w, dim=-1)
        self.metrics["error_anchor_ang_vel"] = torch.norm(self.anchor_ang_vel_w - self.robot_anchor_ang_vel_w, dim=-1)

        self.metrics["error_body_pos"] = torch.norm(self.body_pos_relative_w - self.robot_body_pos_w, dim=-1).mean(
            dim=-1
        )
        self.metrics["error_body_rot"] = quat_error_magnitude(self.body_quat_relative_w, self.robot_body_quat_w).mean(
            dim=-1
        )

        self.metrics["error_body_lin_vel"] = torch.norm(self.body_lin_vel_w - self.robot_body_lin_vel_w, dim=-1).mean(
            dim=-1
        )
        self.metrics["error_body_ang_vel"] = torch.norm(self.body_ang_vel_w - self.robot_body_ang_vel_w, dim=-1).mean(
            dim=-1
        )

        self.metrics["error_joint_pos"] = torch.norm(self.joint_pos - self.robot_joint_pos, dim=-1)
        self.metrics["error_joint_vel"] = torch.norm(self.joint_vel - self.robot_joint_vel, dim=-1)

    def _update_sampling_metrics(self, sampling_probabilities: torch.Tensor):
        if sampling_probabilities.numel() == 0:
            return

        H = -(sampling_probabilities * (sampling_probabilities + 1e-12).log()).sum()
        if sampling_probabilities.numel() > 1:
            H_norm = H / math.log(sampling_probabilities.numel())
        else:
            H_norm = torch.zeros((), dtype=sampling_probabilities.dtype, device=sampling_probabilities.device)
        pmax, imax = sampling_probabilities.max(dim=0)
        self.metrics["sampling_entropy"][:] = H_norm
        self.metrics["sampling_top1_prob"][:] = pmax
        self.metrics["sampling_top1_bin"][:] = imax.float() / max(float(sampling_probabilities.numel()), 1.0)

    def _sample_trajectories(self, env_ids: Sequence[int]):
        if len(env_ids) == 0:
            return

        if (not self.cfg.sample_trajectories) or (self.motion.num_trajectories == 1):
            self.trajectory_ids[env_ids] = 0
            probabilities = torch.zeros(self.motion.num_trajectories, dtype=torch.float, device=self.device)
            probabilities[0] = 1.0
            self._update_sampling_metrics(probabilities)
            return

        if self.cfg.equal_trajectory_sampling:
            probabilities = torch.full(
                (self.motion.num_trajectories,),
                1.0 / float(self.motion.num_trajectories),
                dtype=torch.float,
                device=self.device,
            )
        else:
            probabilities = self.motion.trajectory_time_step_total.float()
            probabilities = probabilities / probabilities.sum()

        self.trajectory_ids[env_ids] = torch.multinomial(probabilities, len(env_ids), replacement=True)
        self._update_sampling_metrics(probabilities)

    def _sample_time_steps(self, env_ids: Sequence[int]):
        if len(env_ids) == 0:
            return
        if not self.cfg.sample_time_steps:
            self.time_steps[env_ids] = 0
            return
        lengths = self.motion.trajectory_time_step_total[self.trajectory_ids[env_ids]]
        sampled = (sample_uniform(0.0, 1.0, (len(env_ids),), device=self.device) * lengths.float()).long()
        self.time_steps[env_ids] = torch.minimum(sampled, torch.clamp(lengths - 1, min=0))

    def _adaptive_sampling(self, env_ids: Sequence[int]):
        episode_failed = self._env.termination_manager.terminated[env_ids]
        if torch.any(episode_failed):
            trajectory_lengths = torch.clamp(self.motion.trajectory_time_step_total[self.trajectory_ids], min=1)
            current_bin_index = torch.clamp(
                (self.time_steps * self.bin_count) // trajectory_lengths, 0, self.bin_count - 1
            )
            fail_bins = current_bin_index[env_ids][episode_failed]
            self._current_bin_failed[:] = torch.bincount(fail_bins, minlength=self.bin_count)

        # Sample
        sampling_probabilities = self.bin_failed_count + self.cfg.adaptive_uniform_ratio / float(self.bin_count)
        sampling_probabilities = torch.nn.functional.pad(
            sampling_probabilities.unsqueeze(0).unsqueeze(0),
            (0, self.cfg.adaptive_kernel_size - 1),  # Non-causal kernel
            mode="replicate",
        )
        sampling_probabilities = torch.nn.functional.conv1d(sampling_probabilities, self.kernel.view(1, 1, -1)).view(-1)

        sampling_probabilities = sampling_probabilities / sampling_probabilities.sum()

        sampled_bins = torch.multinomial(sampling_probabilities, len(env_ids), replacement=True)
        trajectory_lengths = torch.clamp(self.motion.trajectory_time_step_total[self.trajectory_ids[env_ids]], min=1)
        sampled = (
            (sampled_bins + sample_uniform(0.0, 1.0, (len(env_ids),), device=self.device))
            / self.bin_count
            * trajectory_lengths.float()
        ).long()
        self.time_steps[env_ids] = torch.minimum(sampled, torch.clamp(trajectory_lengths - 1, min=0))

        # Metrics
        H = -(sampling_probabilities * (sampling_probabilities + 1e-12).log()).sum()
        if self.bin_count > 1:
            H_norm = H / math.log(self.bin_count)
        else:
            H_norm = torch.zeros((), dtype=sampling_probabilities.dtype, device=sampling_probabilities.device)
        pmax, imax = sampling_probabilities.max(dim=0)
        self.metrics["sampling_entropy"][:] = H_norm
        self.metrics["sampling_top1_prob"][:] = pmax
        self.metrics["sampling_top1_bin"][:] = imax.float() / self.bin_count

    def _resample_command(self, env_ids: Sequence[int]):
        if len(env_ids) == 0:
            return
        self._sample_trajectories(env_ids)
        if self.cfg.sample_time_steps and self.cfg.adaptive_alpha > 0.0:
            self._adaptive_sampling(env_ids)
        else:
            self._sample_time_steps(env_ids)

        root_pos = self.body_pos_w[:, 0].clone()
        root_ori = self.body_quat_w[:, 0].clone()
        root_lin_vel = self.body_lin_vel_w[:, 0].clone()
        root_ang_vel = self.body_ang_vel_w[:, 0].clone()

        range_list = [self.cfg.pose_range.get(key, (0.0, 0.0)) for key in ["x", "y", "z", "roll", "pitch", "yaw"]]
        ranges = torch.tensor(range_list, device=self.device)
        rand_samples = sample_uniform(ranges[:, 0], ranges[:, 1], (len(env_ids), 6), device=self.device)
        root_pos[env_ids] += rand_samples[:, 0:3]
        orientations_delta = quat_from_euler_xyz(rand_samples[:, 3], rand_samples[:, 4], rand_samples[:, 5])
        root_ori[env_ids] = quat_mul(orientations_delta, root_ori[env_ids])
        range_list = [self.cfg.velocity_range.get(key, (0.0, 0.0)) for key in ["x", "y", "z", "roll", "pitch", "yaw"]]
        ranges = torch.tensor(range_list, device=self.device)
        rand_samples = sample_uniform(ranges[:, 0], ranges[:, 1], (len(env_ids), 6), device=self.device)
        root_lin_vel[env_ids] += rand_samples[:, :3]
        root_ang_vel[env_ids] += rand_samples[:, 3:]

        joint_pos = self.joint_pos.clone()
        joint_vel = self.joint_vel.clone()

        joint_pos += sample_uniform(*self.cfg.joint_position_range, joint_pos.shape, joint_pos.device)
        soft_joint_pos_limits = self.robot.data.soft_joint_pos_limits[env_ids]
        joint_pos[env_ids] = torch.clip(
            joint_pos[env_ids], soft_joint_pos_limits[:, :, 0], soft_joint_pos_limits[:, :, 1]
        )
        self.robot.write_joint_state_to_sim(joint_pos[env_ids], joint_vel[env_ids], env_ids=env_ids)
        self.robot.write_root_state_to_sim(
            torch.cat([root_pos[env_ids], root_ori[env_ids], root_lin_vel[env_ids], root_ang_vel[env_ids]], dim=-1),
            env_ids=env_ids,
        )

    def _update_command(self):
        self.time_steps += 1
        trajectory_lengths = self.motion.trajectory_time_step_total[self.trajectory_ids]
        env_ids = torch.where(self.time_steps >= trajectory_lengths)[0]
        self._resample_command(env_ids)

        anchor_pos_w_repeat = self.anchor_pos_w[:, None, :].repeat(1, len(self.cfg.body_names), 1)
        heading_quat_w_repeat = self.heading_quat_w[:, None, :].repeat(1, len(self.cfg.body_names), 1)
        robot_anchor_pos_w_repeat = self.robot_anchor_pos_w[:, None, :].repeat(1, len(self.cfg.body_names), 1)
        robot_heading_quat_w_repeat = self.robot_heading_quat_w[:, None, :].repeat(1, len(self.cfg.body_names), 1)

        delta_pos_w = robot_anchor_pos_w_repeat
        delta_pos_w[..., 2] = anchor_pos_w_repeat[..., 2]
        # Compute yaw-only alignment using a heading body (pelvis by default).
        # This avoids torso-bend artifacts from extracting yaw directly from the anchor.
        robot_heading_yaw_quat_w_repeat = yaw_quat(robot_heading_quat_w_repeat)
        heading_yaw_quat_w_repeat = yaw_quat(heading_quat_w_repeat)
        delta_ori_w = quat_mul(robot_heading_yaw_quat_w_repeat, quat_inv(heading_yaw_quat_w_repeat))

        self.body_quat_relative_w = quat_mul(delta_ori_w, self.body_quat_w)
        self.body_pos_relative_w = delta_pos_w + quat_apply(delta_ori_w, self.body_pos_w - anchor_pos_w_repeat)

        self.bin_failed_count = (
            self.cfg.adaptive_alpha * self._current_bin_failed + (1 - self.cfg.adaptive_alpha) * self.bin_failed_count
        )
        self._current_bin_failed.zero_()

    def _set_debug_vis_impl(self, debug_vis: bool):
        if debug_vis:
            if not hasattr(self, "current_anchor_visualizer"):
                self.current_anchor_visualizer = VisualizationMarkers(
                    self.cfg.anchor_visualizer_cfg.replace(prim_path="/Visuals/Command/current/anchor")
                )
                self.goal_anchor_visualizer = VisualizationMarkers(
                    self.cfg.anchor_visualizer_cfg.replace(prim_path="/Visuals/Command/goal/anchor")
                )

                self.current_body_visualizers = []
                self.goal_body_visualizers = []
                for name in self.cfg.body_names:
                    self.current_body_visualizers.append(
                        VisualizationMarkers(
                            self.cfg.body_visualizer_cfg.replace(prim_path="/Visuals/Command/current/" + name)
                        )
                    )
                    self.goal_body_visualizers.append(
                        VisualizationMarkers(
                            self.cfg.body_visualizer_cfg.replace(prim_path="/Visuals/Command/goal/" + name)
                        )
                    )

            self.current_anchor_visualizer.set_visibility(self.cfg.debug_vis_show_current)
            self.goal_anchor_visualizer.set_visibility(self.cfg.debug_vis_show_goal)
            for i in range(len(self.cfg.body_names)):
                self.current_body_visualizers[i].set_visibility(self.cfg.debug_vis_show_current)
                self.goal_body_visualizers[i].set_visibility(self.cfg.debug_vis_show_goal)

        else:
            if hasattr(self, "current_anchor_visualizer"):
                self.current_anchor_visualizer.set_visibility(False)
                self.goal_anchor_visualizer.set_visibility(False)
                for i in range(len(self.cfg.body_names)):
                    self.current_body_visualizers[i].set_visibility(False)
                    self.goal_body_visualizers[i].set_visibility(False)

    def _debug_vis_callback(self, event):
        if not self.robot.is_initialized:
            return

        if self.cfg.debug_vis_show_current:
            self.current_anchor_visualizer.visualize(self.robot_anchor_pos_w, self.robot_anchor_quat_w)
        if self.cfg.debug_vis_show_goal:
            self.goal_anchor_visualizer.visualize(self.anchor_pos_w, self.anchor_quat_w)

        if self.cfg.debug_vis_goal_relative_to_robot:
            goal_body_pos = self.body_pos_relative_w
            goal_body_quat = self.body_quat_relative_w
        else:
            goal_body_pos = self.body_pos_w
            goal_body_quat = self.body_quat_w

        for i in range(len(self.cfg.body_names)):
            if self.cfg.debug_vis_show_current:
                self.current_body_visualizers[i].visualize(self.robot_body_pos_w[:, i], self.robot_body_quat_w[:, i])
            if self.cfg.debug_vis_show_goal:
                self.goal_body_visualizers[i].visualize(goal_body_pos[:, i], goal_body_quat[:, i])


@configclass
class MotionCommandCfg(CommandTermCfg):
    """Configuration for the motion command."""

    class_type: type = MotionCommand

    asset_name: str = MISSING

    motion_file: str = MISSING
    anchor_body_name: str = MISSING
    body_names: list[str] = MISSING

    pose_range: dict[str, tuple[float, float]] = {}
    velocity_range: dict[str, tuple[float, float]] = {}

    joint_position_range: tuple[float, float] = (-0.52, 0.52)
    sample_trajectories: bool = False
    equal_trajectory_sampling: bool = True
    sample_time_steps: bool = True

    adaptive_kernel_size: int = 1
    adaptive_lambda: float = 0.8
    adaptive_uniform_ratio: float = 0.1
    adaptive_alpha: float = 0.001
    debug_vis_goal_relative_to_robot: bool = True
    debug_vis_show_current: bool = True
    debug_vis_show_goal: bool = True
    # Body used for yaw-only heading alignment in relative body targets.
    # If None, defaults to body_names[0] (typically pelvis/root).
    relative_heading_body_name: str | None = None

    anchor_visualizer_cfg: VisualizationMarkersCfg = FRAME_MARKER_CFG.replace(prim_path="/Visuals/Command/pose")
    anchor_visualizer_cfg.markers["frame"].scale = (0.2, 0.2, 0.2)

    body_visualizer_cfg: VisualizationMarkersCfg = FRAME_MARKER_CFG.replace(prim_path="/Visuals/Command/pose")
    body_visualizer_cfg.markers["frame"].scale = (0.1, 0.1, 0.1)
