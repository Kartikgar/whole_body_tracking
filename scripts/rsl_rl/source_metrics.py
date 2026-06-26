"""Source-simulator tracking metrics for ``play.py`` rollouts."""

from __future__ import annotations

import json
import os
from collections import defaultdict
from dataclasses import dataclass, field
from typing import Literal

import numpy as np
import torch
from isaaclab.utils.math import quat_apply, quat_error_magnitude, quat_inv, quat_mul, yaw_quat

SourceMetricMode = Literal["base_tracking", "delta_openloop"]

SOURCE_METRIC_STATE_KEYS: tuple[str, ...] = (
    "joint_pos",
    "joint_vel",
    "body_pos_w",
    "body_quat_w",
    "body_lin_vel_w",
    "body_ang_vel_w",
)


def infer_source_metric_mode(task_name: str | None) -> SourceMetricMode:
    """Infer post-step reference timing from the play task name."""

    normalized = (task_name or "").lower()
    if "deltaa-openloop" in normalized or "delta-openloop" in normalized or "openloop" in normalized:
        return "delta_openloop"
    return "base_tracking"


def capture_motion_reference(env) -> dict[str, torch.Tensor]:
    """Snapshot the current motion command tensors for metric comparison."""

    command_manager = getattr(env, "command_manager", None)
    if command_manager is None:
        raise RuntimeError("Source metrics require an env with a command manager.")
    motion_term = command_manager.get_term("motion")
    reference = {
        "joint_pos": motion_term.joint_pos,
        "joint_vel": motion_term.joint_vel,
        "body_pos_w": motion_term.body_pos_w,
        "body_quat_w": motion_term.body_quat_w,
        "body_lin_vel_w": motion_term.body_lin_vel_w,
        "body_ang_vel_w": motion_term.body_ang_vel_w,
    }
    return {key: value.detach().clone() for key, value in reference.items()}


def resolve_motion_body_names(env) -> list[str]:
    """Return the body order used by the motion command."""

    command_manager = getattr(env, "command_manager", None)
    if command_manager is None:
        return []
    try:
        motion_term = command_manager.get_term("motion")
    except Exception:
        return []
    return list(getattr(motion_term.cfg, "body_names", []))


def resolve_motion_anchor_index(env, body_names: list[str]) -> int:
    """Resolve the motion anchor body index, falling back to root index 0."""

    command_manager = getattr(env, "command_manager", None)
    if command_manager is None:
        return 0
    try:
        motion_term = command_manager.get_term("motion")
    except Exception:
        return 0
    anchor_name = getattr(motion_term.cfg, "anchor_body_name", None)
    if anchor_name in body_names:
        return int(body_names.index(anchor_name))
    return int(getattr(motion_term, "motion_anchor_body_index", 0))


def _finite_mean_std(values: list[float]) -> tuple[float, float]:
    array = np.asarray(values, dtype=np.float32)
    array = array[np.isfinite(array)]
    if array.size == 0:
        return float("nan"), float("nan")
    return float(array.mean()), float(array.std())


def _trajectory_position_metrics(
    predicted_positions: np.ndarray,
    reference_positions: np.ndarray,
    *,
    root_idx: int,
) -> dict[str, float]:
    if predicted_positions.shape != reference_positions.shape:
        raise ValueError(
            "Trajectory metric shape mismatch: "
            f"predicted={predicted_positions.shape}, reference={reference_positions.shape}."
        )
    if predicted_positions.ndim != 3:
        raise ValueError(f"Expected body position trajectory [T, B, 3], got {predicted_positions.shape}.")

    mpjpe = np.linalg.norm(reference_positions - predicted_positions, axis=2) * 1000.0

    predicted_local = predicted_positions - predicted_positions[:, [root_idx]]
    reference_local = reference_positions - reference_positions[:, [root_idx]]
    mpjpe_l = np.linalg.norm(reference_local - predicted_local, axis=2) * 1000.0

    if predicted_positions.shape[0] >= 2:
        ref_vel = reference_positions[1:] - reference_positions[:-1]
        pred_vel = predicted_positions[1:] - predicted_positions[:-1]
        vel_dist = np.linalg.norm(pred_vel - ref_vel, axis=2).mean(axis=1) * 1000.0
    else:
        vel_dist = np.empty((0,), dtype=np.float32)

    if predicted_positions.shape[0] >= 3:
        ref_acc = reference_positions[:-2] - 2.0 * reference_positions[1:-1] + reference_positions[2:]
        pred_acc = predicted_positions[:-2] - 2.0 * predicted_positions[1:-1] + predicted_positions[2:]
        accel_dist = np.linalg.norm(pred_acc - ref_acc, axis=2).mean(axis=1) * 1000.0
    else:
        accel_dist = np.empty((0,), dtype=np.float32)

    return {
        "mpjpe": float(mpjpe.mean()) if mpjpe.size > 0 else float("nan"),
        "mpjpe_l": float(mpjpe_l.mean()) if mpjpe_l.size > 0 else float("nan"),
        "vel_dist": float(vel_dist.mean()) if vel_dist.size > 0 else float("nan"),
        "accel_dist": float(accel_dist.mean()) if accel_dist.size > 0 else float("nan"),
    }


@dataclass
class SourceSimMetricsTracker:
    """Accumulate source-simulator tracking metrics over ``play.py`` rollouts."""

    num_envs: int
    mode: SourceMetricMode
    body_names: list[str]
    anchor_idx: int
    root_idx: int = 0
    step_metric_sums: dict[str, float] = field(default_factory=lambda: defaultdict(float))
    step_sample_count: int = 0
    trajectory_metric_values: dict[str, list[float]] = field(default_factory=lambda: defaultdict(list))
    trajectory_count: int = 0
    _pred_body_pos_buffers: list[list[np.ndarray]] = field(default_factory=list)
    _ref_body_pos_buffers: list[list[np.ndarray]] = field(default_factory=list)

    def __post_init__(self) -> None:
        self.num_envs = int(self.num_envs)
        self.anchor_idx = int(self.anchor_idx)
        self.root_idx = int(self.root_idx)
        if self.num_envs <= 0:
            raise ValueError(f"num_envs must be positive, got {self.num_envs}.")
        if not self.body_names:
            raise ValueError("Source metrics require motion body names.")
        if not (0 <= self.anchor_idx < len(self.body_names)):
            raise ValueError(f"anchor_idx={self.anchor_idx} out of range for {len(self.body_names)} bodies.")
        if not (0 <= self.root_idx < len(self.body_names)):
            raise ValueError(f"root_idx={self.root_idx} out of range for {len(self.body_names)} bodies.")
        self._pred_body_pos_buffers = [[] for _ in range(self.num_envs)]
        self._ref_body_pos_buffers = [[] for _ in range(self.num_envs)]

    @property
    def reference_timing_description(self) -> str:
        if self.mode == "delta_openloop":
            return "post-step state vs advanced motion reference (m[t+1])"
        return "post-step state vs current commanded reference (m[t])"

    def _compute_body_relative_targets(
        self, state: dict[str, torch.Tensor], reference: dict[str, torch.Tensor]
    ) -> tuple[torch.Tensor, torch.Tensor]:
        robot_anchor_pos = state["body_pos_w"][:, self.anchor_idx]
        robot_heading_quat = state["body_quat_w"][:, self.root_idx]
        ref_anchor_pos = reference["body_pos_w"][:, self.anchor_idx]
        ref_heading_quat = reference["body_quat_w"][:, self.root_idx]

        delta_pos = robot_anchor_pos[:, None, :].repeat(1, len(self.body_names), 1)
        delta_pos[..., 2] = ref_anchor_pos[:, None, 2]
        delta_ori = quat_mul(yaw_quat(robot_heading_quat), quat_inv(yaw_quat(ref_heading_quat)))
        delta_ori_batch = delta_ori[:, None, :].repeat(1, len(self.body_names), 1)

        ref_anchor_repeat = ref_anchor_pos[:, None, :].repeat(1, len(self.body_names), 1)
        body_pos_relative = delta_pos + quat_apply(delta_ori_batch, reference["body_pos_w"] - ref_anchor_repeat)
        body_quat_relative = quat_mul(delta_ori_batch, reference["body_quat_w"])
        return body_pos_relative, body_quat_relative

    def append_step(
        self,
        state: dict[str, torch.Tensor],
        reference: dict[str, torch.Tensor],
        *,
        active_mask: torch.Tensor | None = None,
    ) -> None:
        if set(state.keys()) != set(SOURCE_METRIC_STATE_KEYS):
            raise AssertionError(f"State keys mismatch: got {sorted(state.keys())}.")
        if set(reference.keys()) != set(SOURCE_METRIC_STATE_KEYS):
            raise AssertionError(f"Reference keys mismatch: got {sorted(reference.keys())}.")

        device = state["joint_pos"].device
        if active_mask is None:
            active_mask = torch.ones(self.num_envs, dtype=torch.bool, device=device)
        else:
            active_mask = active_mask.to(device=device, dtype=torch.bool).reshape(-1)
        if active_mask.numel() != self.num_envs:
            raise AssertionError(f"Expected active_mask [{self.num_envs}], got {tuple(active_mask.shape)}.")
        if not torch.any(active_mask):
            return

        body_pos_relative, body_quat_relative = self._compute_body_relative_targets(state, reference)

        error_anchor_pos = torch.linalg.norm(
            reference["body_pos_w"][:, self.anchor_idx] - state["body_pos_w"][:, self.anchor_idx], dim=-1
        )
        error_anchor_rot = quat_error_magnitude(
            reference["body_quat_w"][:, self.anchor_idx], state["body_quat_w"][:, self.anchor_idx]
        )
        error_body_pos = torch.linalg.norm(body_pos_relative - state["body_pos_w"], dim=-1).mean(dim=-1)
        error_body_rot = quat_error_magnitude(body_quat_relative, state["body_quat_w"]).mean(dim=-1)
        error_joint_pos = torch.linalg.norm(reference["joint_pos"] - state["joint_pos"], dim=-1)
        error_joint_vel = torch.linalg.norm(reference["joint_vel"] - state["joint_vel"], dim=-1)

        reward_anchor_pos = torch.exp(-(error_anchor_pos**2) / (0.3**2))
        reward_anchor_ori = torch.exp(-(error_anchor_rot**2) / (0.4**2))
        body_pos_sq = torch.square(body_pos_relative - state["body_pos_w"]).sum(dim=-1).mean(dim=-1)
        body_rot_sq = torch.square(quat_error_magnitude(body_quat_relative, state["body_quat_w"])).mean(dim=-1)
        reward_body_pos = torch.exp(-body_pos_sq / (0.3**2))
        reward_body_ori = torch.exp(-body_rot_sq / (0.4**2))

        body_lin_sq = torch.square(reference["body_lin_vel_w"] - state["body_lin_vel_w"]).sum(dim=-1).mean(dim=-1)
        body_ang_sq = torch.square(reference["body_ang_vel_w"] - state["body_ang_vel_w"]).sum(dim=-1).mean(dim=-1)
        reward_body_lin_vel = torch.exp(-body_lin_sq / (1.0**2))
        reward_body_ang_vel = torch.exp(-body_ang_sq / (3.14**2))
        tracking_reward_total = (
            0.5 * reward_anchor_pos
            + 0.5 * reward_anchor_ori
            + reward_body_pos
            + reward_body_ori
            + reward_body_lin_vel
            + reward_body_ang_vel
        )

        metrics = {
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
        }

        sample_count = int(active_mask.sum().item())
        for name, values in metrics.items():
            self.step_metric_sums[name] += float(values[active_mask].sum().item())
        self.step_sample_count += sample_count

        pred_body_pos = state["body_pos_w"].detach().to("cpu", dtype=torch.float32).numpy()
        ref_body_pos = reference["body_pos_w"].detach().to("cpu", dtype=torch.float32).numpy()
        active_ids = torch.nonzero(active_mask, as_tuple=False).flatten().tolist()
        for env_id in active_ids:
            self._pred_body_pos_buffers[env_id].append(pred_body_pos[env_id].copy())
            self._ref_body_pos_buffers[env_id].append(ref_body_pos[env_id].copy())

    def finalize_done(self, done_mask: torch.Tensor) -> int:
        done = done_mask.detach().to(device="cpu", dtype=torch.bool)
        if done.ndim == 0:
            done = done.reshape(1)
        elif done.ndim > 1:
            done = done.reshape(done.shape[0], -1).any(dim=1)
        if done.ndim != 1 or done.shape[0] != self.num_envs:
            raise AssertionError(f"Expected done mask [{self.num_envs}], got {tuple(done.shape)}.")

        gained = 0
        for env_id in torch.nonzero(done, as_tuple=False).flatten().tolist():
            gained += self._finalize_env(env_id)
        return gained

    def finalize_open(self) -> int:
        gained = 0
        for env_id in range(self.num_envs):
            gained += self._finalize_env(env_id)
        return gained

    def _finalize_env(self, env_id: int) -> int:
        pred_buffer = self._pred_body_pos_buffers[env_id]
        ref_buffer = self._ref_body_pos_buffers[env_id]
        if len(pred_buffer) == 0:
            return 0

        predicted = np.stack(pred_buffer, axis=0).astype(np.float32)
        reference = np.stack(ref_buffer, axis=0).astype(np.float32)
        metrics = _trajectory_position_metrics(predicted, reference, root_idx=self.root_idx)
        for key, value in metrics.items():
            self.trajectory_metric_values[key].append(value)
        self.trajectory_count += 1
        self._pred_body_pos_buffers[env_id] = []
        self._ref_body_pos_buffers[env_id] = []
        return 1

    def summary(self) -> dict[str, object]:
        output: dict[str, object] = {
            "source_metric_mode": self.mode,
            "source_metric_reference_timing": self.reference_timing_description,
            "source_metric_samples": self.step_sample_count,
            "source_metric_trajectories": self.trajectory_count,
        }

        if self.step_sample_count > 0:
            for key, value in sorted(self.step_metric_sums.items()):
                output[f"{key}_mean"] = float(value / float(self.step_sample_count))

        for key in ("mpjpe", "mpjpe_l", "vel_dist", "accel_dist"):
            mean, std = _finite_mean_std(self.trajectory_metric_values[key])
            output[f"{key}_mean"] = mean
            output[f"{key}_std"] = std

        return output

    def print_summary(self) -> None:
        summary = self.summary()
        print("\n=== Source Sim Tracking Metrics ===")
        print(f"mode: {summary['source_metric_mode']}")
        print(f"reference_timing: {summary['source_metric_reference_timing']}")
        print(f"samples: {summary['source_metric_samples']}")
        print(f"trajectories: {summary['source_metric_trajectories']}")
        for key in (
            "tracking_reward_total_mean",
            "error_anchor_pos_mean",
            "error_body_pos_mean",
            "error_joint_pos_mean",
            "mpjpe_mean",
            "mpjpe_l_mean",
            "vel_dist_mean",
            "accel_dist_mean",
        ):
            value = summary.get(key)
            if isinstance(value, (float, int)) and np.isfinite(float(value)):
                print(f"{key}: {float(value):.6f}")

    def save_summary(self, output_path: str) -> str:
        os.makedirs(os.path.dirname(output_path) or ".", exist_ok=True)
        summary = self.summary()
        with open(output_path, "w", encoding="utf-8") as handle:
            json.dump(summary, handle, indent=2, sort_keys=True)
        return output_path
