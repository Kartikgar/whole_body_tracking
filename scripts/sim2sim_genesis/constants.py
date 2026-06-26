"""Constants shared by the modular Genesis sim2sim evaluator."""

from __future__ import annotations

import os
import xml.etree.ElementTree as ET

OBSERVATION_NOISE_RANGES: dict[str, tuple[float, float]] = {
    "motion_anchor_pos_b": (-0.25, 0.25),
    "motion_anchor_ori_b": (-0.05, 0.05),
    "base_lin_vel": (-0.5, 0.5),
    "base_ang_vel": (-0.2, 0.2),
    "joint_pos": (-0.01, 0.01),
    "joint_vel": (-0.5, 0.5),
}

DOMAIN_RAND_FRICTION_RANGE = (0.3, 1.6)
DOMAIN_RAND_FRICTION_NUM_BUCKETS = 64
DOMAIN_RAND_JOINT_DEFAULT_POS_RANGE = (-0.00, 0.00)
DOMAIN_RAND_BASE_COM_RANGE = {
    "x": (-0.025, 0.025),
    "y": (-0.05, 0.05),
    "z": (-0.05, 0.05),
}
DOMAIN_RAND_PUSH_INTERVAL_RANGE_S = (1.0, 3.0)
DOMAIN_RAND_PUSH_VELOCITY_RANGE = {
    "x": (-0.5, 0.5),
    "y": (-0.5, 0.5),
    "z": (-0.2, 0.2),
    "roll": (-0.52, 0.52),
    "pitch": (-0.52, 0.52),
    "yaw": (-0.78, 0.78),
}

# Uniform noise on actuated joint qpos after reset-to-reference (radians); root pose unchanged.
DEFAULT_STARTUP_QPOS_JOINT_RANGE = (-0.10, 0.10)

# Kp/Kd perturbation uses this seed only, not `--seed` / evaluation seed.
EVAL_KP_KD_PERTURB_RNG_SEED = 913_571
KP_KD_PERTURB_SCALE = 0.0

SUPPORTED_OBSERVATION_TERMS: tuple[str, ...] = (
    "command",
    "motion_anchor_pos_b",
    "motion_anchor_ori_b",
    "base_lin_vel",
    "base_ang_vel",
    "joint_pos",
    "joint_vel",
    "actions",
    "motion_joint_action",
)

SUPPORTED_TERM_DIMS: dict[str, str] = {
    "command": "double_joint_count",
    "motion_anchor_pos_b": "3",
    "motion_anchor_ori_b": "6",
    "base_lin_vel": "3",
    "base_ang_vel": "3",
    "joint_pos": "joint_count",
    "joint_vel": "joint_count",
    "actions": "action_count",
    "motion_joint_action": "joint_count",
}

G1_ALL_BODY_JOINT_NAMES: tuple[str, ...] = (
    "root_joint",
    "left_hip_pitch_joint",
    "left_hip_roll_joint",
    "left_hip_yaw_joint",
    "left_knee_joint",
    "left_ankle_pitch_joint",
    "left_ankle_roll_joint",
    "right_hip_pitch_joint",
    "right_hip_roll_joint",
    "right_hip_yaw_joint",
    "right_knee_joint",
    "right_ankle_pitch_joint",
    "right_ankle_roll_joint",
    "waist_yaw_joint",
    "waist_roll_joint",
    "waist_pitch_joint",
    "left_shoulder_pitch_joint",
    "left_shoulder_roll_joint",
    "left_shoulder_yaw_joint",
    "left_elbow_joint",
    "left_wrist_roll_joint",
    "left_wrist_pitch_joint",
    "left_wrist_yaw_joint",
    "right_shoulder_pitch_joint",
    "right_shoulder_roll_joint",
    "right_shoulder_yaw_joint",
    "right_elbow_joint",
    "right_wrist_roll_joint",
    "right_wrist_pitch_joint",
    "right_wrist_yaw_joint",
)

DEFAULT_NEXT_LAB_DATE = "2026.07.01"
DEFAULT_REFERENCE_MARKER_RADIUS = 0.05

DEFAULT_G1_URDF = os.path.join(
    os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__)))),
    "source",
    "whole_body_tracking",
    "whole_body_tracking",
    "assets",
    "unitree_description",
    "urdf",
    "g1",
    "main.urdf",
)


def bucketize_uniform(values: object, low: float, high: float, num_buckets: int):
    """Bucketize uniformly sampled values to the nearest discrete bin."""

    import numpy as np

    array = np.asarray(values, dtype=np.float32)
    if num_buckets <= 1 or high <= low:
        return array.astype(np.float32)
    idx = np.round((array - low) / (high - low) * (num_buckets - 1)).astype(np.int32)
    idx = np.clip(idx, 0, num_buckets - 1)
    bucket_values = np.linspace(low, high, num_buckets, dtype=np.float32)
    return bucket_values[idx]


def resolve_urdf_child_link_names(urdf_file: str | None, joint_names: tuple[str, ...]) -> list[str] | None:
    """Resolve unique URDF child link names for an ordered joint list."""

    if urdf_file is None or not os.path.isfile(urdf_file):
        return None

    try:
        robot_root = ET.parse(urdf_file).getroot()
    except Exception:
        return None

    child_link_by_joint: dict[str, str] = {}
    all_link_names: list[str] = []
    child_link_names: set[str] = set()

    for link_el in robot_root.findall("link"):
        link_name = link_el.get("name")
        if link_name is not None:
            all_link_names.append(link_name)

    for joint_el in robot_root.findall("joint"):
        joint_name = joint_el.get("name")
        child_el = joint_el.find("child")
        if joint_name is None or child_el is None:
            continue
        child_name = child_el.get("link")
        if child_name is not None:
            child_link_by_joint[joint_name] = child_name
            child_link_names.add(child_name)

    root_link_name = None
    for link_name in all_link_names:
        if link_name not in child_link_names:
            root_link_name = link_name
            break

    resolved_links: list[str] = []
    for joint_name in joint_names:
        child_name = child_link_by_joint.get(joint_name)
        if child_name is None and joint_name == "root_joint":
            child_name = root_link_name
        if child_name is None:
            return None
        if child_name not in resolved_links:
            resolved_links.append(child_name)
    return resolved_links
