"""Simulator-independent specification of the Fauna Sprout humanoid.

Single source of truth for the Isaac Lab articulation config and the Genesis / MuJoCo
probes. This module imports no simulator, so tests and CPU tools can use it.

Sources:
- Robot description: ``sprout-model-v0.1.4/`` (``sprout.urdf``, ``sprout.xml``).
  Joint names, order and limits are parsed from the URDF at import time.
- PD gains, armature and passive damping: ``sprout.xml`` default classes
  (lower_body kp 32.5, upper_body kp 10, kv 1, armature 0.01, damping 0.1).
- Hardware joint IDs: ``fauna-sdk/.../fauna_msgs/msg/BipedJointNames.msg``.
- Motor models: ``fauna-sdk/fauna-sdk/ansible/files/robot_config.json``.
"""

from __future__ import annotations

import json
import os
import xml.etree.ElementTree as ET
from pathlib import Path

_WORKSPACE_ROOT = Path(__file__).resolve().parents[5]

SPROUT_ASSET_DIR = Path(os.environ.get("SPROUT_ASSET_DIR", _WORKSPACE_ROOT / "sprout-model-v0.1.4"))
SPROUT_URDF_PATH = SPROUT_ASSET_DIR / "sprout.urdf"
SPROUT_MJCF_PATH = SPROUT_ASSET_DIR / "sprout.xml"
SPROUT_SCENE_PATH = SPROUT_ASSET_DIR / "scene.xml"
SPROUT_STAND_POSE_PATH = Path(__file__).with_name("sprout_stand_pose.json")
# Render-fixed copy for URDF-based simulators (Isaac, Genesis): byte-identical URDF, symlinked
# meshes, only the visual .mtl texture paths/colour patched (scripts/sprout/make_render_asset.py).
SPROUT_RENDER_URDF_PATH = Path(__file__).resolve().parents[1] / "assets/sprout_description/sprout.urdf"

# ---------------------------------------------------------------------------
# Joints (sim order = URDF tree order, identical to the MJCF actuator order is NOT
# guaranteed; always index by name).
# ---------------------------------------------------------------------------


def _parse_urdf_joints(path: Path) -> list[tuple[str, float, float, float, float]]:
    """Return (name, lower, upper, effort, velocity) for each revolute joint in file order."""
    root = ET.parse(path).getroot()
    joints = []
    for joint in root.findall("joint"):
        if joint.get("type") != "revolute":
            continue
        limit = joint.find("limit")
        joints.append(
            (
                joint.get("name"),
                float(limit.get("lower")),
                float(limit.get("upper")),
                float(limit.get("effort")),
                float(limit.get("velocity")),
            )
        )
    return joints


_URDF_JOINTS = _parse_urdf_joints(SPROUT_URDF_PATH)

JOINT_NAMES: tuple[str, ...] = tuple(j[0] for j in _URDF_JOINTS)
NUM_JOINTS = len(JOINT_NAMES)  # 27
JOINT_LIMITS: dict[str, tuple[float, float]] = {j[0]: (j[1], j[2]) for j in _URDF_JOINTS}

# URDF effort/velocity values are uniform placeholders (80 Nm, 5 rad/s). They stay
# PROVISIONAL until replaced with motor datasheet values per MOTOR_MODEL.
EFFORT_LIMIT: dict[str, float] = {j[0]: j[3] for j in _URDF_JOINTS}
VELOCITY_LIMIT: dict[str, float] = {j[0]: j[4] for j in _URDF_JOINTS}
EFFORT_VELOCITY_LIMITS_PROVISIONAL = True

LOWER_BODY_JOINTS: tuple[str, ...] = (
    "waist_yaw_joint",
    "left_hip_pitch_joint",
    "left_hip_roll_joint",
    "left_hip_yaw_joint",
    "left_knee_joint",
    "left_ankle_joint",
    "right_hip_pitch_joint",
    "right_hip_roll_joint",
    "right_hip_yaw_joint",
    "right_knee_joint",
    "right_ankle_joint",
)
UPPER_BODY_JOINTS: tuple[str, ...] = tuple(n for n in JOINT_NAMES if n not in LOWER_BODY_JOINTS)

# ---------------------------------------------------------------------------
# Actuation (MJCF defaults)
# ---------------------------------------------------------------------------

KP_LOWER_BODY = 32.5
KP_UPPER_BODY = 10.0
KD = 1.0
ARMATURE = 0.01
PASSIVE_DAMPING = 0.1

JOINT_STIFFNESS: dict[str, float] = {
    n: (KP_LOWER_BODY if n in LOWER_BODY_JOINTS else KP_UPPER_BODY) for n in JOINT_NAMES
}
JOINT_DAMPING: dict[str, float] = {n: KD for n in JOINT_NAMES}

# Known discrepancy: the SDK example walk policy (examples/onnx_motor_control) commands
# kp = 10 on waist_yaw_joint, whereas the MJCF puts the waist in lower_body (kp 32.5).
# Sims use the MJCF value; this records what that hardware policy actually sends.
SDK_EXAMPLE_POLICY_KP_OVERRIDES: dict[str, float] = {"waist_yaw_joint": 10.0}

MOTOR_MODEL: dict[str, str] = {
    "waist_yaw_joint": "EC_A4310_P2_36",
    "left_hip_pitch_joint": "EC_A6408_P2_25",
    "left_hip_roll_joint": "EC_A6408_P2_25",
    "left_hip_yaw_joint": "EC_A4310_P2_36",
    "left_knee_joint": "EC_A6408_P2_25",
    "left_ankle_joint": "EC_A6408_P2_25",
    "right_hip_pitch_joint": "EC_A6408_P2_25",
    "right_hip_roll_joint": "EC_A6408_P2_25",
    "right_hip_yaw_joint": "EC_A4310_P2_36",
    "right_knee_joint": "EC_A6408_P2_25",
    "right_ankle_joint": "EC_A6408_P2_25",
    "neck_yaw_joint": "HT4438",
    "neck_pitch_joint": "HT3536",
    "left_shoulder_pitch_joint": "EC_A4310_P2_36",
    "left_shoulder_roll_joint": "HT5047",
    "left_shoulder_yaw_joint": "HT5047",
    "left_elbow_joint": "HT5047",
    "left_wrist_roll_joint": "HT4438",
    "left_wrist_pitch_joint": "HT3536",
    "left_gripper_joint": "HT3536",
    "right_shoulder_pitch_joint": "EC_A4310_P2_36",
    "right_shoulder_roll_joint": "HT5047",
    "right_shoulder_yaw_joint": "HT5047",
    "right_elbow_joint": "HT5047",
    "right_wrist_roll_joint": "HT4438",
    "right_wrist_pitch_joint": "HT3536",
    "right_gripper_joint": "HT3536",
}

# ---------------------------------------------------------------------------
# Hardware ("control") order from BipedJointNames.msg. THIGH = hip yaw and
# WRIST = wrist roll: confirmed by the example policy's per-ID kp pattern.
# ---------------------------------------------------------------------------

HARDWARE_JOINT_ORDER: tuple[str, ...] = (
    "left_hip_pitch_joint",  # 0 L_HIP_PITCH
    "left_hip_roll_joint",  # 1 L_HIP_ROLL
    "left_knee_joint",  # 2 L_KNEE
    "right_hip_pitch_joint",  # 3 R_HIP_PITCH
    "right_hip_roll_joint",  # 4 R_HIP_ROLL
    "right_knee_joint",  # 5 R_KNEE
    "left_ankle_joint",  # 6 L_ANKLE
    "right_ankle_joint",  # 7 R_ANKLE
    "waist_yaw_joint",  # 8 WAIST
    "left_shoulder_pitch_joint",  # 9 L_SHOULDER_PITCH
    "left_shoulder_roll_joint",  # 10 L_SHOULDER_ROLL
    "right_shoulder_pitch_joint",  # 11 R_SHOULDER_PITCH
    "right_shoulder_roll_joint",  # 12 R_SHOULDER_ROLL
    "left_hip_yaw_joint",  # 13 L_THIGH
    "right_hip_yaw_joint",  # 14 R_THIGH
    "left_shoulder_yaw_joint",  # 15 L_SHOULDER_YAW
    "left_elbow_joint",  # 16 L_ELBOW
    "left_wrist_roll_joint",  # 17 L_WRIST
    "left_gripper_joint",  # 18 L_GRIPPER
    "right_shoulder_yaw_joint",  # 19 R_SHOULDER_YAW
    "right_elbow_joint",  # 20 R_ELBOW
    "right_wrist_roll_joint",  # 21 R_WRIST
    "right_gripper_joint",  # 22 R_GRIPPER
    "neck_pitch_joint",  # 23 NECK_PITCH
    "neck_yaw_joint",  # 24 NECK_YAW
    "left_wrist_pitch_joint",  # 25 L_WRIST_PITCH
    "right_wrist_pitch_joint",  # 26 R_WRIST_PITCH
)

# hw_array = sim_array[SIM_TO_HW]; sim_array = hw_array[HW_TO_SIM]
SIM_TO_HW: tuple[int, ...] = tuple(JOINT_NAMES.index(n) for n in HARDWARE_JOINT_ORDER)
HW_TO_SIM: tuple[int, ...] = tuple(HARDWARE_JOINT_ORDER.index(n) for n in JOINT_NAMES)

# MJCF (sprout.xml) hinge order: the joint order of BlinkMimic policies (their actions and observations). The
# BlinkMimic task configs use it as the action term's joint order (preserve_order=True). Checked against the MJCF
# in tests/test_sprout_spec.py.
MJCF_JOINT_ORDER: tuple[str, ...] = (
    "waist_yaw_joint",
    "left_hip_pitch_joint", "left_hip_roll_joint", "left_hip_yaw_joint", "left_knee_joint", "left_ankle_joint",
    "right_hip_pitch_joint", "right_hip_roll_joint", "right_hip_yaw_joint", "right_knee_joint", "right_ankle_joint",
    "neck_yaw_joint", "neck_pitch_joint",
    "left_shoulder_pitch_joint", "left_shoulder_roll_joint", "left_shoulder_yaw_joint", "left_elbow_joint",
    "left_wrist_roll_joint", "left_wrist_pitch_joint", "left_gripper_joint",
    "right_shoulder_pitch_joint", "right_shoulder_roll_joint", "right_shoulder_yaw_joint", "right_elbow_joint",
    "right_wrist_roll_joint", "right_wrist_pitch_joint", "right_gripper_joint",
)  # fmt: skip

# ---------------------------------------------------------------------------
# Bodies and frames
# ---------------------------------------------------------------------------

ROOT_BODY = "torso_link"  # floating base; pelvis hangs below it via waist_yaw_joint
PELVIS_BODY = "pelvis_link"
# Frame links (URDF fixed joints ref_* -> these child links; same body names in the MJCF).
LEFT_FOOT_BODY = "left_foot_link"  # carries the foot collision meshes
RIGHT_FOOT_BODY = "right_foot_link"
LEFT_SOLE_FRAME = "left_sole_link"
RIGHT_SOLE_FRAME = "right_sole_link"
FOOT_FRAMES: tuple[str, ...] = ("left_toe_link", "left_heel_link", "right_toe_link", "right_heel_link")
IMU_FRAME = "imu_link"  # MJCF also has site "imu_site"
TOTAL_MASS_KG = 24.940353  # sum of URDF link masses

# ---------------------------------------------------------------------------
# Default standing pose (written by scripts/sprout/derive_stand_pose.py)
# ---------------------------------------------------------------------------


def load_stand_pose(path: Path = SPROUT_STAND_POSE_PATH) -> tuple[dict[str, float], float] | None:
    """Return (joint_pos, root_height) or None if the pose has not been derived yet."""
    if not path.exists():
        return None
    data = json.loads(path.read_text())
    return {k: float(v) for k, v in data["joint_pos"].items()}, float(data["root_height"])


_stand = load_stand_pose()
DEFAULT_STAND_POSE: dict[str, float] = _stand[0] if _stand else {n: 0.0 for n in JOINT_NAMES}
STAND_ROOT_HEIGHT: float | None = _stand[1] if _stand else None
