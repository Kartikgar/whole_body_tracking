"""Configuration helpers for the modular Genesis sim2sim evaluator."""

from __future__ import annotations

import os
from dataclasses import dataclass

from sim2sim_genesis.constants import DEFAULT_NEXT_LAB_DATE
from sim2sim_genesis.experiment import ExperimentConfig


def load_logged_initial_states(path, num_envs):
    """Load one paired startup batch and remove Isaac placement offsets, if saved."""
    import numpy as np
    with np.load(path, allow_pickle=True) as archive:
        initial = {key:archive[key] for key in archive.files if key.startswith('initial_')}
        initial['joint_names'] = archive['joint_names'].tolist()
        initial['body_names'] = archive['body_names'].tolist()
        if 'env_origins' in archive:
            origins = archive['env_origins']
            if origins.shape != (num_envs,3):
                raise ValueError('env_origins must have shape [num_envs, 3]')
            initial['initial_body_pos_w'] = initial['initial_body_pos_w'] - origins[:,None,:]
    for key in ('initial_joint_pos','initial_joint_vel','initial_body_pos_w','initial_body_quat_w'):
        if key not in initial or initial[key].shape[0] != num_envs:
            raise ValueError(f'{key} must provide one state per environment')
    if any(not np.isfinite(value).all() for key,value in initial.items() if key.startswith('initial_')):
        raise ValueError('Nonfinite logged initial state')
    if 'pelvis' not in initial['body_names']:
        raise ValueError('Logged initial states must include pelvis')
    return initial


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
    initial_state_npz: str | None = None
    endpoint_height_error_tolerance_m: float = 0.25
    experiment: ExperimentConfig | None = None
    policy_type: str = "beyondmimic"
    sonic_model_dir: str | None = None
    trajectory_index: int = 0


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
