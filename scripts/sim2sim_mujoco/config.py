"""Configuration helpers for the MuJoCo sim2sim evaluator."""

from __future__ import annotations

import os
from dataclasses import dataclass

from sim2sim_genesis.config import default_eval_artifact_path
from sim2sim_mujoco.domain_randomization import MujocoDRConfig


@dataclass(slots=True)
class MujocoEvalConfig:
    """Normalized options for one batched MuJoCo evaluation."""

    policy_path: str
    xml_file: str
    policy_device: str
    num_envs: int
    num_threads: int | None
    sim_dt: float
    control_dt: float
    start_timestep: int
    max_steps: int | None
    torque_limit: float | None
    compute_metrics: bool
    record_video: bool
    video_env_ids: list[int] | None
    show_reference: bool
    reference_marker_radius: float
    video_width: int
    video_height: int
    video_name: str | None
    output_json: str
    seed: int | None
    domain_randomization: bool
    dr: MujocoDRConfig
    randomize_startup_qpos: bool
    startup_qpos_joint_range: tuple[float, float]
    disable_termination: bool


def append_timestamp_to_path(path: str, timestamp: str) -> str:
    """Append a timestamp to ``path`` while preserving the extension."""

    directory = os.path.dirname(path)
    stem, ext = os.path.splitext(os.path.basename(path))
    return os.path.join(directory, f"{stem}_{timestamp}{ext}")


def default_json_path(run_timestamp: str, policy_path: str) -> str:
    return default_eval_artifact_path(run_timestamp, _artifact_policy_path(policy_path), ".json")


def default_video_path(run_timestamp: str, policy_path: str) -> str:
    return default_eval_artifact_path(run_timestamp, _artifact_policy_path(policy_path), ".mp4")


def _artifact_policy_path(policy_path: str) -> str:
    """Use a cwd-relative policy path for readable artifact names when possible."""

    abs_path = os.path.abspath(policy_path)
    try:
        relative = os.path.relpath(abs_path, os.getcwd())
    except ValueError:
        return policy_path
    return policy_path if relative.startswith("..") else relative
