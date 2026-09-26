"""Default SONIC G1 deployment contract.

Parameters reproduced from NVIDIA's Apache-2.0 policy_parameters.hpp and
gear_sonic/envs/manager_env/robots/g1.py. See NOTICE.md for provenance.
"""
from dataclasses import dataclass, field
from pathlib import Path

import numpy as np

SOURCE_REVISION = "b042411fae38ee4d1af9aac82a37a1f8d14d6dd0"
MODEL_REVISION = "6733128a3d8a523b1418b06bca3cdf61c8b0987f"
CONTROL_DT = 0.02
# Isaac Lab order expected by both SONIC ONNX models. This is deliberately not
# URDF/MuJoCo order: the policy interleaves left, right, and waist joints.
JOINT_NAMES = [
    "left_hip_pitch_joint", "right_hip_pitch_joint", "waist_yaw_joint",
    "left_hip_roll_joint", "right_hip_roll_joint", "waist_roll_joint",
    "left_hip_yaw_joint", "right_hip_yaw_joint", "waist_pitch_joint",
    "left_knee_joint", "right_knee_joint",
    "left_shoulder_pitch_joint", "right_shoulder_pitch_joint",
    "left_ankle_pitch_joint", "right_ankle_pitch_joint",
    "left_shoulder_roll_joint", "right_shoulder_roll_joint",
    "left_ankle_roll_joint", "right_ankle_roll_joint",
    "left_shoulder_yaw_joint", "right_shoulder_yaw_joint",
    "left_elbow_joint", "right_elbow_joint",
    "left_wrist_roll_joint", "right_wrist_roll_joint",
    "left_wrist_pitch_joint", "right_wrist_pitch_joint",
    "left_wrist_yaw_joint", "right_wrist_yaw_joint",
]
BODY_NAMES = [
    "pelvis", "left_hip_pitch_link", "right_hip_pitch_link", "waist_yaw_link",
    "left_hip_roll_link", "right_hip_roll_link", "waist_roll_link",
    "left_hip_yaw_link", "right_hip_yaw_link", "torso_link", "left_knee_link", "right_knee_link",
    "left_shoulder_pitch_link", "right_shoulder_pitch_link", "left_ankle_pitch_link", "right_ankle_pitch_link",
    "left_shoulder_roll_link", "right_shoulder_roll_link", "left_ankle_roll_link", "right_ankle_roll_link",
    "left_shoulder_yaw_link", "right_shoulder_yaw_link", "left_elbow_link", "right_elbow_link",
    "left_wrist_roll_link", "right_wrist_roll_link", "left_wrist_pitch_link", "right_wrist_pitch_link",
    "left_wrist_yaw_link", "right_wrist_yaw_link",
]


@dataclass
class SonicSpec:
    joint_names: list[str] = field(default_factory=lambda: list(JOINT_NAMES))
    body_names: list[str] = field(default_factory=lambda: list(BODY_NAMES))
    anchor_body_name: str = "pelvis"
    observation_names: list[str] = field(default_factory=list)
    observation_history_lengths: list[int] = field(default_factory=list)

    def __post_init__(self):
        arms, efforts, defaults, scale_stiffness = [], [], [], []
        for name in self.joint_names:
            if any(s in name for s in ("hip_pitch", "hip_roll", "knee")):
                arm, effort, scale_kp = 0.025101925, 139.0, 0.025101925
            elif "hip_yaw" in name or "waist_yaw" in name:
                arm, effort, scale_kp = 0.010177520, 88.0, 0.010177520
            elif "wrist_pitch" in name or "wrist_yaw" in name:
                arm, effort, scale_kp = 0.00425, 5.0, 0.00425
            elif "ankle" in name or name in ("waist_roll_joint", "waist_pitch_joint"):
                # NVIDIA's Isaac environment doubles armature, PD gains, and the
                # simulation effort limit. Its action scale consequently remains
                # equal to 0.25 * 25 / undoubled_5020_stiffness.
                arm, effort, scale_kp = 2 * 0.003609725, 50.0, 2 * 0.003609725
            else:
                arm, effort, scale_kp = 0.003609725, 25.0, 0.003609725
            angle = 0.0
            for key, value in (("hip_pitch", -0.312), ("knee", 0.669), ("ankle_pitch", -0.363),
                               ("shoulder_pitch", 0.2), ("elbow", 0.6)):
                if key in name:
                    angle = value
            if "shoulder_roll" in name:
                angle = 0.2 if name.startswith("left") else -0.2
            arms.append(arm)
            efforts.append(effort)
            defaults.append(angle)
            scale_stiffness.append(scale_kp)
        self.armature = np.asarray(arms, dtype=np.float32)
        self.effort_limits = np.asarray(efforts, dtype=np.float32)
        self.default_joint_pos = np.asarray(defaults, dtype=np.float32)
        omega = 10 * 2 * np.pi
        self.joint_stiffness = self.armature * omega**2
        self.joint_damping = self.armature * 4 * omega
        self.action_scale = (
            0.25 * self.effort_limits / (np.asarray(scale_stiffness, dtype=np.float32) * omega**2)
        )

    def joint_targets(self, actions):
        return self.default_joint_pos + np.asarray(actions, dtype=np.float32) * self.action_scale


def robot_path(model_dir):
    path = Path(model_dir) / "robot_description/urdf/g1/main.urdf"
    if not path.is_file():
        raise FileNotFoundError(f"SONIC G1 asset missing: {path}. Run scripts/setup_sonic.py.")
    return str(path.resolve())
