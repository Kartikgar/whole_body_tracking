"""Configuration helpers for the modular Genesis sim2sim evaluator."""

from __future__ import annotations

import os
from dataclasses import dataclass

from sim2sim_genesis.constants import DEFAULT_NEXT_LAB_DATE


@dataclass(slots=True)
class EvalConfig:
    """Normalized runtime options for a Genesis sim2sim rollout."""

    policy_path: str
    urdf_file: str | None
    xml_file: str | None
    motion_file: str | None
    backend: str
    policy_device: str
    num_envs: int
    sim_dt: float
    control_dt: float
    start_timestep: int
    max_steps: int | None
    torque_limit: float | None
    record_video: bool
    viewer: bool
    show_reference: bool
    reference_marker_radius: float
    add_noise: bool
    domain_randomization: bool
    randomize_startup_qpos: bool
    startup_qpos_joint_range: tuple[float, float]
    compute_metrics: bool
    metric_num_envs: int
    record_motion: bool
    target_trajectories: int
    output_motion_npz: str | None
    video_name: str | None
    seed: int | None


@dataclass(slots=True)
class OutputTargets:
    """Resolved output paths for summary artifacts."""

    output_csv: str | None
    output_json: str | None
    output_motion_npz: str | None


def merged_policy_path_tag(policy_path: str) -> str:
    """Sanitize ``policy_path`` for artifact filenames (matches video naming)."""

    return policy_path.replace("/", "_")[:-5]


def default_eval_artifact_path(
    run_timestamp: str,
    policy_path: str,
    extension: str,
    *,
    output_dir: str | None = None,
) -> str:
    """Build a default sim2sim eval artifact path using the video filename convention."""

    directory = output_dir or os.path.join("logs", "sim2sim_eval", DEFAULT_NEXT_LAB_DATE)
    ext = extension if extension.startswith(".") else f".{extension}"
    stem = f"{run_timestamp}_{merged_policy_path_tag(policy_path)}"
    return os.path.join(directory, f"{stem}{ext}")
