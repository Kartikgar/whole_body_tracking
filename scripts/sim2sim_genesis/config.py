"""Configuration helpers for the modular Genesis sim2sim evaluator."""

from __future__ import annotations

from dataclasses import dataclass


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
