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
    _REQUIRED_MOTION_KEYS = (
        "joint_pos",
        "joint_vel",
        "body_pos_w",
        "body_quat_w",
        "body_lin_vel_w",
        "body_ang_vel_w",
    )
    _BODY_KEYS = {"body_pos_w", "body_quat_w", "body_lin_vel_w", "body_ang_vel_w"}

    def __init__(self, motion_file: str, body_indexes: Sequence[int], device: str = "cpu"):
        assert os.path.isfile(motion_file), f"Invalid file path: {motion_file}"
        self._body_indexes = torch.as_tensor(body_indexes, dtype=torch.long, device=device)
        self._max_body_index = int(self._body_indexes.max().item())

        with np.load(motion_file, allow_pickle=True) as data:
            if "joint_pos" in data.files:
                parsed = self._parse_stacked_format(data)
            else:
                parsed = self._parse_per_motion_format(data)

        self.fps = float(parsed["fps"])
        trajectory_lengths = np.asarray(parsed["trajectory_lengths"], dtype=np.int64)
        if trajectory_lengths.ndim != 1 or trajectory_lengths.size == 0:
            raise ValueError(f"Invalid trajectory length metadata: shape={trajectory_lengths.shape}")

        self.num_trajectories = int(trajectory_lengths.shape[0])
        self.trajectory_time_step_total = torch.tensor(trajectory_lengths, dtype=torch.long, device=device)
        self.time_step_total = int(self.trajectory_time_step_total.max().item())

        self._trajectory_start_index = torch.zeros(self.num_trajectories, dtype=torch.long, device=device)
        if self.num_trajectories > 1:
            self._trajectory_start_index[1:] = torch.cumsum(self.trajectory_time_step_total[:-1], dim=0)

        self._joint_pos_flat = self._concat_trajectory_tensors(parsed["joint_pos"], device=device)
        self._joint_vel_flat = self._concat_trajectory_tensors(parsed["joint_vel"], device=device)
        self._body_pos_w_flat = self._concat_trajectory_tensors(parsed["body_pos_w"], device=device)
        self._body_quat_w_flat = self._concat_trajectory_tensors(parsed["body_quat_w"], device=device)
        self._body_lin_vel_w_flat = self._concat_trajectory_tensors(parsed["body_lin_vel_w"], device=device)
        self._body_ang_vel_w_flat = self._concat_trajectory_tensors(parsed["body_ang_vel_w"], device=device)

        self._joint_action_flat = None
        if parsed["action"] is not None:
            self._joint_action_flat = self._concat_trajectory_tensors(parsed["action"], device=device)

    @staticmethod
    def _concat_trajectory_tensors(values: list[np.ndarray], device: str) -> torch.Tensor:
        tensors = [torch.tensor(np.asarray(value), dtype=torch.float32, device=device) for value in values]
        if len(tensors) == 0:
            raise ValueError("Cannot concatenate empty trajectory list.")
        return torch.cat(tensors, dim=0)

    def _normalize_stacked_vector_key(self, key: str, value: np.ndarray) -> np.ndarray:
        arr = np.asarray(value, dtype=np.float32)
        if arr.ndim == 2:
            arr = arr[None, ...]
        if arr.ndim != 3:
            raise ValueError(f"Expected `{key}` shape [T, D] or [N_traj, T, D], got {arr.shape}.")
        return arr

    def _normalize_stacked_body_key(self, key: str, value: np.ndarray, expected_tail_dim: int) -> np.ndarray:
        arr = np.asarray(value, dtype=np.float32)
        if arr.ndim == 3:
            arr = arr[None, ...]
        if arr.ndim != 4 or arr.shape[-1] != expected_tail_dim:
            raise ValueError(
                f"Expected `{key}` shape [T, B, {expected_tail_dim}] or [N_traj, T, B, {expected_tail_dim}], "
                f"got {arr.shape}."
            )
        if arr.shape[2] <= self._max_body_index:
            raise ValueError(
                f"`{key}` has only {arr.shape[2]} bodies, but requested body index {self._max_body_index} exists."
            )
        return arr

    def _parse_stacked_format(self, data: np.lib.npyio.NpzFile) -> dict[str, list[np.ndarray] | np.ndarray | float | None]:
        if "fps" not in data.files:
            raise ValueError("Motion file is missing required key `fps`.")
        fps = float(np.asarray(data["fps"]).reshape(-1)[0])

        joint_pos = self._normalize_stacked_vector_key("joint_pos", data["joint_pos"])
        joint_vel = self._normalize_stacked_vector_key("joint_vel", data["joint_vel"])
        if joint_vel.shape != joint_pos.shape:
            raise ValueError(f"`joint_vel` shape {joint_vel.shape} must match `joint_pos` shape {joint_pos.shape}.")

        body_pos_w = self._normalize_stacked_body_key("body_pos_w", data["body_pos_w"], expected_tail_dim=3)
        body_quat_w = self._normalize_stacked_body_key("body_quat_w", data["body_quat_w"], expected_tail_dim=4)
        body_lin_vel_w = self._normalize_stacked_body_key("body_lin_vel_w", data["body_lin_vel_w"], expected_tail_dim=3)
        body_ang_vel_w = self._normalize_stacked_body_key("body_ang_vel_w", data["body_ang_vel_w"], expected_tail_dim=3)

        num_traj = int(joint_pos.shape[0])
        time_step_total = int(joint_pos.shape[1])
        for key, arr in (
            ("body_pos_w", body_pos_w),
            ("body_quat_w", body_quat_w),
            ("body_lin_vel_w", body_lin_vel_w),
            ("body_ang_vel_w", body_ang_vel_w),
        ):
            if arr.shape[0] != num_traj:
                raise ValueError(f"`{key}` has {arr.shape[0]} trajectories, expected {num_traj}.")
            if arr.shape[1] != time_step_total:
                raise ValueError(f"`{key}` has {arr.shape[1]} timesteps, expected {time_step_total}.")

        action = None
        if "action" in data.files:
            action = self._normalize_stacked_vector_key("action", data["action"])
        elif "actions" in data.files:
            action = self._normalize_stacked_vector_key("actions", data["actions"])
        if action is not None:
            if action.shape[0] != num_traj:
                raise ValueError(f"Motion action trajectory mismatch. Expected {num_traj}, got {action.shape[0]}.")
            if action.shape[1] != time_step_total:
                raise ValueError(
                    f"Motion action length mismatch. Expected {time_step_total}, got {action.shape[1]}."
                )
            if action.shape[2] != joint_pos.shape[2]:
                raise ValueError(
                    f"Motion action dim mismatch. joint_pos has dim {joint_pos.shape[2]}, action has dim {action.shape[2]}."
                )

        trajectory_lengths = np.full((num_traj,), time_step_total, dtype=np.int64)
        return {
            "fps": fps,
            "joint_pos": [joint_pos[i] for i in range(num_traj)],
            "joint_vel": [joint_vel[i] for i in range(num_traj)],
            "body_pos_w": [body_pos_w[i] for i in range(num_traj)],
            "body_quat_w": [body_quat_w[i] for i in range(num_traj)],
            "body_lin_vel_w": [body_lin_vel_w[i] for i in range(num_traj)],
            "body_ang_vel_w": [body_ang_vel_w[i] for i in range(num_traj)],
            "action": [action[i] for i in range(num_traj)] if action is not None else None,
            "trajectory_lengths": trajectory_lengths,
        }

    @staticmethod
    def _resolve_motion_keys(data: np.lib.npyio.NpzFile) -> list[str]:
        if "motion_keys" in data.files:
            raw_keys = [str(key) for key in np.asarray(data["motion_keys"]).reshape(-1).tolist()]
            if len(raw_keys) == 0:
                raise ValueError("`motion_keys` is present but empty.")
            return raw_keys

        motion_keys = [key for key in data.files if key.startswith("motion")]
        if len(motion_keys) == 0:
            raise ValueError(
                "Could not find motion data. Expected stacked keys (`joint_pos`, ...) or `motion{i}` keys."
            )

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

    @staticmethod
    def _normalize_single_vector_key(key: str, value: np.ndarray, motion_key: str) -> np.ndarray:
        arr = np.asarray(value, dtype=np.float32)
        if arr.ndim == 3 and arr.shape[0] == 1:
            arr = arr[0]
        if arr.ndim != 2:
            raise ValueError(f"`{motion_key}`:`{key}` must be [T, D], got {arr.shape}.")
        if arr.shape[0] <= 0:
            raise ValueError(f"`{motion_key}`:`{key}` has empty time dimension.")
        return arr

    def _normalize_single_body_key(
        self, key: str, value: np.ndarray, motion_key: str, expected_tail_dim: int
    ) -> np.ndarray:
        arr = np.asarray(value, dtype=np.float32)
        if arr.ndim == 4 and arr.shape[0] == 1:
            arr = arr[0]
        if arr.ndim != 3 or arr.shape[-1] != expected_tail_dim:
            raise ValueError(
                f"`{motion_key}`:`{key}` must be [T, B, {expected_tail_dim}], got {arr.shape}."
            )
        if arr.shape[0] <= 0:
            raise ValueError(f"`{motion_key}`:`{key}` has empty time dimension.")
        if arr.shape[1] <= self._max_body_index:
            raise ValueError(
                f"`{motion_key}`:`{key}` has {arr.shape[1]} bodies, but requested body index {self._max_body_index} exists."
            )
        return arr

    def _parse_per_motion_format(
        self, data: np.lib.npyio.NpzFile
    ) -> dict[str, list[np.ndarray] | np.ndarray | float | None]:
        motion_keys = self._resolve_motion_keys(data)

        parsed_motions: list[dict[str, np.ndarray]] = []
        action_presence: bool | None = None
        fps_candidates: list[float] = []
        joint_dim: int | None = None
        body_count: int | None = None

        for motion_key in motion_keys:
            motion_dict = self._extract_motion_dict(motion_key, data[motion_key])

            for required_key in self._REQUIRED_MOTION_KEYS:
                if required_key not in motion_dict:
                    raise ValueError(f"`{motion_key}` is missing required key `{required_key}`.")

            motion_joint_pos = self._normalize_single_vector_key("joint_pos", motion_dict["joint_pos"], motion_key)
            motion_joint_vel = self._normalize_single_vector_key("joint_vel", motion_dict["joint_vel"], motion_key)
            if motion_joint_vel.shape != motion_joint_pos.shape:
                raise ValueError(
                    f"`{motion_key}`:`joint_vel` shape {motion_joint_vel.shape} must match `joint_pos` shape {motion_joint_pos.shape}."
                )

            motion_body_pos_w = self._normalize_single_body_key("body_pos_w", motion_dict["body_pos_w"], motion_key, 3)
            motion_body_quat_w = self._normalize_single_body_key("body_quat_w", motion_dict["body_quat_w"], motion_key, 4)
            motion_body_lin_vel_w = self._normalize_single_body_key(
                "body_lin_vel_w", motion_dict["body_lin_vel_w"], motion_key, 3
            )
            motion_body_ang_vel_w = self._normalize_single_body_key(
                "body_ang_vel_w", motion_dict["body_ang_vel_w"], motion_key, 3
            )

            time_len = motion_joint_pos.shape[0]
            for key, arr in (
                ("body_pos_w", motion_body_pos_w),
                ("body_quat_w", motion_body_quat_w),
                ("body_lin_vel_w", motion_body_lin_vel_w),
                ("body_ang_vel_w", motion_body_ang_vel_w),
            ):
                if arr.shape[0] != time_len:
                    raise ValueError(
                        f"`{motion_key}`:`{key}` has {arr.shape[0]} timesteps, expected {time_len}."
                    )

            if joint_dim is None:
                joint_dim = int(motion_joint_pos.shape[1])
            elif int(motion_joint_pos.shape[1]) != joint_dim:
                raise ValueError(
                    f"`{motion_key}`:`joint_pos` has dim {motion_joint_pos.shape[1]}, expected {joint_dim}."
                )
            if body_count is None:
                body_count = int(motion_body_pos_w.shape[1])
            elif int(motion_body_pos_w.shape[1]) != body_count:
                raise ValueError(
                    f"`{motion_key}` body count {motion_body_pos_w.shape[1]} does not match expected {body_count}."
                )

            motion_action = None
            if "action" in motion_dict:
                motion_action = self._normalize_single_vector_key("action", motion_dict["action"], motion_key)
            elif "actions" in motion_dict:
                motion_action = self._normalize_single_vector_key("actions", motion_dict["actions"], motion_key)
            has_action = motion_action is not None
            if action_presence is None:
                action_presence = has_action
            elif action_presence != has_action:
                raise ValueError(
                    "Inconsistent action coverage across motions. Either all motions must provide "
                    "`action`/`actions`, or none."
                )
            if motion_action is not None:
                if motion_action.shape[0] != time_len:
                    raise ValueError(
                        f"`{motion_key}` action has {motion_action.shape[0]} timesteps, expected {time_len}."
                    )
                if motion_action.shape[1] != joint_dim:
                    raise ValueError(
                        f"`{motion_key}` action dim {motion_action.shape[1]} does not match joint dim {joint_dim}."
                    )

            if "fps" in motion_dict:
                fps_candidates.append(float(np.asarray(motion_dict["fps"]).reshape(-1)[0]))

            parsed_motions.append(
                {
                    "joint_pos": motion_joint_pos,
                    "joint_vel": motion_joint_vel,
                    "body_pos_w": motion_body_pos_w,
                    "body_quat_w": motion_body_quat_w,
                    "body_lin_vel_w": motion_body_lin_vel_w,
                    "body_ang_vel_w": motion_body_ang_vel_w,
                    "action": motion_action,
                }
            )

        if len(parsed_motions) == 0:
            raise ValueError("No motions were found in per-motion dataset.")

        fps = float(np.asarray(data["fps"]).reshape(-1)[0]) if "fps" in data.files else None
        if fps is None:
            if len(fps_candidates) == 0:
                raise ValueError("Motion file has no `fps` key at top-level or per-motion level.")
            fps = fps_candidates[0]
        elif len(fps_candidates) > 0:
            ref = fps
            max_delta = max(abs(val - ref) for val in fps_candidates)
            if max_delta > 1e-5:
                raise ValueError(f"Inconsistent per-motion fps values detected (max delta {max_delta}).")

        num_traj = len(parsed_motions)
        trajectory_lengths = np.asarray([motion["joint_pos"].shape[0] for motion in parsed_motions], dtype=np.int64)
        if np.any(trajectory_lengths <= 0):
            raise ValueError(f"Invalid trajectory lengths found: {trajectory_lengths.tolist()}")

        return {
            "fps": float(fps),
            "joint_pos": [motion["joint_pos"] for motion in parsed_motions],
            "joint_vel": [motion["joint_vel"] for motion in parsed_motions],
            "body_pos_w": [motion["body_pos_w"] for motion in parsed_motions],
            "body_quat_w": [motion["body_quat_w"] for motion in parsed_motions],
            "body_lin_vel_w": [motion["body_lin_vel_w"] for motion in parsed_motions],
            "body_ang_vel_w": [motion["body_ang_vel_w"] for motion in parsed_motions],
            "action": [motion["action"] for motion in parsed_motions] if bool(action_presence) else None,
            "trajectory_lengths": trajectory_lengths,
        }

    def _resolve_frame_indices(self, trajectory_ids: torch.Tensor, time_steps: torch.Tensor) -> torch.Tensor:
        traj = torch.as_tensor(trajectory_ids, dtype=torch.long, device=self.trajectory_time_step_total.device)
        step = torch.as_tensor(time_steps, dtype=torch.long, device=self.trajectory_time_step_total.device)
        if traj.shape != step.shape:
            raise ValueError(
                f"trajectory_ids and time_steps must have the same shape. Got {tuple(traj.shape)} and {tuple(step.shape)}."
            )
        max_step = torch.clamp(self.trajectory_time_step_total[traj] - 1, min=0)
        step = torch.clamp(step, min=0)
        step = torch.minimum(step, max_step)
        return self._trajectory_start_index[traj] + step

    def get_joint_pos(self, trajectory_ids: torch.Tensor, time_steps: torch.Tensor) -> torch.Tensor:
        flat_idx = self._resolve_frame_indices(trajectory_ids, time_steps)
        return self._joint_pos_flat[flat_idx]

    def get_joint_vel(self, trajectory_ids: torch.Tensor, time_steps: torch.Tensor) -> torch.Tensor:
        flat_idx = self._resolve_frame_indices(trajectory_ids, time_steps)
        return self._joint_vel_flat[flat_idx]

    def get_joint_action(self, trajectory_ids: torch.Tensor, time_steps: torch.Tensor) -> torch.Tensor | None:
        if self._joint_action_flat is None:
            return None
        flat_idx = self._resolve_frame_indices(trajectory_ids, time_steps)
        return self._joint_action_flat[flat_idx]

    def get_body_pos_w_full(self, trajectory_ids: torch.Tensor, time_steps: torch.Tensor) -> torch.Tensor:
        flat_idx = self._resolve_frame_indices(trajectory_ids, time_steps)
        return self._body_pos_w_flat[flat_idx]

    def get_body_quat_w_full(self, trajectory_ids: torch.Tensor, time_steps: torch.Tensor) -> torch.Tensor:
        flat_idx = self._resolve_frame_indices(trajectory_ids, time_steps)
        return self._body_quat_w_flat[flat_idx]

    def get_body_lin_vel_w_full(self, trajectory_ids: torch.Tensor, time_steps: torch.Tensor) -> torch.Tensor:
        flat_idx = self._resolve_frame_indices(trajectory_ids, time_steps)
        return self._body_lin_vel_w_flat[flat_idx]

    def get_body_ang_vel_w_full(self, trajectory_ids: torch.Tensor, time_steps: torch.Tensor) -> torch.Tensor:
        flat_idx = self._resolve_frame_indices(trajectory_ids, time_steps)
        return self._body_ang_vel_w_flat[flat_idx]

    def get_body_pos_w(self, trajectory_ids: torch.Tensor, time_steps: torch.Tensor) -> torch.Tensor:
        body_pos = self.get_body_pos_w_full(trajectory_ids, time_steps)
        return body_pos.index_select(-2, self._body_indexes)

    def get_body_quat_w(self, trajectory_ids: torch.Tensor, time_steps: torch.Tensor) -> torch.Tensor:
        body_quat = self.get_body_quat_w_full(trajectory_ids, time_steps)
        return body_quat.index_select(-2, self._body_indexes)

    def get_body_lin_vel_w(self, trajectory_ids: torch.Tensor, time_steps: torch.Tensor) -> torch.Tensor:
        body_lin_vel = self.get_body_lin_vel_w_full(trajectory_ids, time_steps)
        return body_lin_vel.index_select(-2, self._body_indexes)

    def get_body_ang_vel_w(self, trajectory_ids: torch.Tensor, time_steps: torch.Tensor) -> torch.Tensor:
        body_ang_vel = self.get_body_ang_vel_w_full(trajectory_ids, time_steps)
        return body_ang_vel.index_select(-2, self._body_indexes)

    def get_trajectory_data(self, trajectory_id: int) -> dict[str, torch.Tensor]:
        if trajectory_id < 0:
            trajectory_id += self.num_trajectories
        if trajectory_id < 0 or trajectory_id >= self.num_trajectories:
            raise IndexError(f"Invalid trajectory id {trajectory_id}. Expected [0, {self.num_trajectories - 1}].")

        start = int(self._trajectory_start_index[trajectory_id].item())
        length = int(self.trajectory_time_step_total[trajectory_id].item())
        end = start + length
        out = {
            "joint_pos": self._joint_pos_flat[start:end],
            "joint_vel": self._joint_vel_flat[start:end],
            "body_pos_w": self._body_pos_w_flat[start:end].index_select(1, self._body_indexes),
            "body_quat_w": self._body_quat_w_flat[start:end].index_select(1, self._body_indexes),
            "body_lin_vel_w": self._body_lin_vel_w_flat[start:end].index_select(1, self._body_indexes),
            "body_ang_vel_w": self._body_ang_vel_w_flat[start:end].index_select(1, self._body_indexes),
        }
        if self._joint_action_flat is not None:
            out["joint_action"] = self._joint_action_flat[start:end]
        return out

    @property
    def has_joint_action(self) -> bool:
        return self._joint_action_flat is not None

    @property
    def joint_action(self) -> torch.Tensor | None:
        return self._joint_action_flat


class MotionCommand(CommandTerm):
    cfg: MotionCommandCfg

    def __init__(self, cfg: MotionCommandCfg, env: ManagerBasedRLEnv):
        super().__init__(cfg, env)
        self.robot: Articulation = env.scene[cfg.asset_name]
        self.robot_anchor_body_index = self.robot.body_names.index(self.cfg.anchor_body_name)
        self.motion_anchor_body_index = self.cfg.body_names.index(self.cfg.anchor_body_name)
        self.body_indexes = torch.tensor(
            self.robot.find_bodies(self.cfg.body_names, preserve_order=True)[0], dtype=torch.long, device=self.device
        )

        self.motion = MotionLoader(self.cfg.motion_file, self.body_indexes, device=self.device)
        self.trajectory_ids = torch.zeros(self.num_envs, dtype=torch.long, device=self.device)
        self._trajectory_sampling_offset = 0
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
    def has_joint_action(self) -> bool:
        return self.motion.has_joint_action

    @property
    def joint_action(self) -> torch.Tensor:
        if not self.motion.has_joint_action or self.motion.joint_action is None:
            raise RuntimeError("Motion file does not contain `action`/`actions`, but motion_joint_action was requested.")
        action = self.motion.get_joint_action(self.trajectory_ids, self.time_steps)
        if action is None:
            raise RuntimeError("Motion file does not contain `action`/`actions`, but motion_joint_action was requested.")
        return action

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
        return (
            self.motion.get_body_pos_w_full(self.trajectory_ids, self.time_steps)[:, self.motion_anchor_body_index]
            + self._env.scene.env_origins
        )

    @property
    def anchor_quat_w(self) -> torch.Tensor:
        return self.motion.get_body_quat_w_full(self.trajectory_ids, self.time_steps)[:, self.motion_anchor_body_index]

    @property
    def anchor_lin_vel_w(self) -> torch.Tensor:
        return self.motion.get_body_lin_vel_w_full(self.trajectory_ids, self.time_steps)[:, self.motion_anchor_body_index]

    @property
    def anchor_ang_vel_w(self) -> torch.Tensor:
        return self.motion.get_body_ang_vel_w_full(self.trajectory_ids, self.time_steps)[:, self.motion_anchor_body_index]

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

    def _adaptive_sampling(self, env_ids: Sequence[int]):
        if not self.cfg.sample_time_steps:
            # Deterministic replay mode: always restart selected trajectories from the first frame.
            self.time_steps[env_ids] = 0
            self.metrics["sampling_entropy"][:] = 0.0
            self.metrics["sampling_top1_prob"][:] = 1.0
            self.metrics["sampling_top1_bin"][:] = 0.0
            return

        episode_failed = self._env.termination_manager.terminated[env_ids]
        if torch.any(episode_failed):
            env_lengths = torch.clamp(self.motion.trajectory_time_step_total[self.trajectory_ids], min=1)
            current_bin_index = torch.clamp(
                (self.time_steps * self.bin_count) // env_lengths, 0, self.bin_count - 1
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
        sampled_lengths = torch.clamp(self.motion.trajectory_time_step_total[self.trajectory_ids[env_ids]], min=1)
        sampled_time_steps = (
            (sampled_bins + sample_uniform(0.0, 1.0, (len(env_ids),), device=self.device))
            / self.bin_count
            * (sampled_lengths - 1)
        ).long()
        self.time_steps[env_ids] = torch.clamp(sampled_time_steps, min=0)

        # Metrics
        H = -(sampling_probabilities * (sampling_probabilities + 1e-12).log()).sum()
        H_norm = H / math.log(self.bin_count)
        pmax, imax = sampling_probabilities.max(dim=0)
        self.metrics["sampling_entropy"][:] = H_norm
        self.metrics["sampling_top1_prob"][:] = pmax
        self.metrics["sampling_top1_bin"][:] = imax.float() / self.bin_count

    def _sample_trajectory_ids(self, env_ids: Sequence[int]):
        if len(env_ids) == 0:
            return
        if self.motion.num_trajectories <= 1:
            self.trajectory_ids[env_ids] = 0
            return
        if not self.cfg.sample_trajectories:
            self.trajectory_ids[env_ids] = 0
            return

        num_to_sample = len(env_ids)
        num_traj = self.motion.num_trajectories

        if self.cfg.equal_trajectory_sampling:
            # Balanced assignment with random order: each trajectory appears floor/ceil equally often.
            base = (torch.arange(num_to_sample, device=self.device) + self._trajectory_sampling_offset) % num_traj
            self._trajectory_sampling_offset = int((self._trajectory_sampling_offset + num_to_sample) % num_traj)
            sampled = base[torch.randperm(num_to_sample, device=self.device)]
        else:
            sampled = torch.randint(0, num_traj, (num_to_sample,), device=self.device)

        self.trajectory_ids[env_ids] = sampled.long()

    def _resample_command(self, env_ids: Sequence[int]):
        if len(env_ids) == 0:
            return
        self._sample_trajectory_ids(env_ids)
        self._adaptive_sampling(env_ids)

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
        env_lengths = torch.clamp(self.motion.trajectory_time_step_total[self.trajectory_ids], min=1)
        env_ids = torch.where(self.time_steps >= env_lengths)[0]
        self._resample_command(env_ids)

        anchor_pos_w_repeat = self.anchor_pos_w[:, None, :].repeat(1, len(self.cfg.body_names), 1)
        anchor_quat_w_repeat = self.anchor_quat_w[:, None, :].repeat(1, len(self.cfg.body_names), 1)
        robot_anchor_pos_w_repeat = self.robot_anchor_pos_w[:, None, :].repeat(1, len(self.cfg.body_names), 1)
        robot_anchor_quat_w_repeat = self.robot_anchor_quat_w[:, None, :].repeat(1, len(self.cfg.body_names), 1)

        delta_pos_w = robot_anchor_pos_w_repeat
        delta_pos_w[..., 2] = anchor_pos_w_repeat[..., 2]
        delta_ori_w = yaw_quat(quat_mul(robot_anchor_quat_w_repeat, quat_inv(anchor_quat_w_repeat)))

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

    adaptive_kernel_size: int = 1
    adaptive_lambda: float = 0.8
    adaptive_uniform_ratio: float = 0.1
    adaptive_alpha: float = 0.001

    # If True, sample random reference time-steps on command resample.
    # If False, restart at time-step 0 for deterministic start-to-finish replay.
    sample_time_steps: bool = True

    # Multi-trajectory sampling controls.
    sample_trajectories: bool = False
    equal_trajectory_sampling: bool = False

    # If True, goal/reference body markers are yaw-aligned to the robot anchor for easier shape comparison.
    # If False, goal/reference body markers stay in world frame and don't rotate with robot turns/falls.
    debug_vis_goal_relative_to_robot: bool = True
    # If True, show robot-current debug markers.
    debug_vis_show_current: bool = True
    # If True, show reference/goal debug markers.
    debug_vis_show_goal: bool = True

    anchor_visualizer_cfg: VisualizationMarkersCfg = FRAME_MARKER_CFG.replace(prim_path="/Visuals/Command/pose")
    anchor_visualizer_cfg.markers["frame"].scale = (0.2, 0.2, 0.2)

    body_visualizer_cfg: VisualizationMarkersCfg = FRAME_MARKER_CFG.replace(prim_path="/Visuals/Command/pose")
    body_visualizer_cfg.markers["frame"].scale = (0.1, 0.1, 0.1)
