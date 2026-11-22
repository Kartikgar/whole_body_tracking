"""Fauna Sprout RobotSpec for the Isaac Lab port of BlinkMimic.

Same definition as the mjlab entity (blinkmimic/src/blinkmimic/sprout/sprout_constants.py): joint order = MJCF order
(the torch FK is built from Fauna's sprout.xml), 14 tracked bodies with torso_link as root, feet = foot_link,
contact height 0.103 m, mirror signs from the joint axes, frax-retargeted LAFAN1 motions. The simulator is Isaac Lab
with ``robots.sprout.SPROUT_CFG`` (URDF; kp 32.5/10, kd 1.0 + 0.1 passive, armature 0.01, 80 Nm, no velocity
limit, no self-collision).
"""

from __future__ import annotations

import os

import mujoco
import numpy as np

from whole_body_tracking.robots import sprout_spec as SS

from .robot import MotionArrays, RobotSpec

WBT_DIR = os.path.abspath(os.path.join(os.path.dirname(__file__), "..", "..", "..", ".."))
# The 40 LAFAN1 clips of the BlinkMimic G1 benchmark, retargeted to Sprout (symlinks into lafan1_frax/sprout).
MOTION_DIR = os.path.join(WBT_DIR, "data", "lafan1_frax", "sprout_bm40")
MJCF_PATH = str(SS.SPROUT_MJCF_PATH)


def _mjcf_joint_names() -> list[str]:
    m = mujoco.MjModel.from_xml_path(MJCF_PATH)
    return [m.joint(i).name for i in range(m.njnt) if m.jnt_type[i] == mujoco.mjtJoint.mjJNT_HINGE]


JOINT_NAMES = _mjcf_joint_names()
assert sorted(JOINT_NAMES) == sorted(SS.JOINT_NAMES), "MJCF and sprout_spec joints differ"

TRACKED_BODIES = [
    "torso_link",  # root / anchor (floating base)
    "pelvis_link",
    "left_femur_upper_link", "left_tibia_link", "left_foot_link",
    "right_femur_upper_link", "right_tibia_link", "right_foot_link",
    "left_humerus_upper_link", "left_radius_link", "left_gripper_lower_link",
    "right_humerus_upper_link", "right_radius_link", "right_gripper_lower_link",
]  # fmt: skip

FOOT_ORIGIN_HEIGHT = 0.093  # ankle joint (foot_link origin) above the sole on flat ground


def _mirror_signs() -> dict[str, float]:
    """+1 if the mirror of the left axis is the negated right axis, -1 if equal (reflection reverses rotation)."""
    m = mujoco.MjModel.from_xml_path(MJCF_PATH)
    d = mujoco.MjData(m)
    mujoco.mj_forward(m, d)
    M = np.diag([1.0, -1.0, 1.0])
    out = {}
    for n in JOINT_NAMES:
        p = n.replace("left_", "@").replace("right_", "left_").replace("@", "right_")
        a, b = M @ d.xaxis[m.joint(n).id], d.xaxis[m.joint(p).id]
        if np.allclose(a, b, atol=1e-3):
            out[n] = -1.0
        elif np.allclose(a, -b, atol=1e-3):
            out[n] = 1.0
        else:
            raise ValueError(f"{n}: axis is not mirror-symmetric with {p}")
    return out


def load_frax_npz(path: str) -> MotionArrays:
    """frax retarget: q = [root pos, root quat wxyz, joints in npz joint_names order] -> MJCF joint order."""
    d = np.load(path)
    q, names = d["q"], [str(n) for n in d["joint_names"]]
    j = q[:, 7:][:, [names.index(n) for n in JOINT_NAMES]]
    return q[:, :3], q[:, 3:7], j, float(d["fps"])


def _robot_cfg():
    from whole_body_tracking.robots.sprout import SPROUT_CFG  # needs a running Isaac Sim app

    return SPROUT_CFG


def _action_scale() -> dict[str, float]:
    # = robots.sprout.SPROUT_ACTION_SCALE (0.25 * 80 Nm / kp), without importing isaaclab here
    return {n: 0.25 * 80.0 / SS.JOINT_STIFFNESS[n] for n in JOINT_NAMES}


SPEC = RobotSpec(
    name="sprout",
    mjcf_path=MJCF_PATH,
    joint_names=tuple(JOINT_NAMES),
    tracked_bodies=tuple(TRACKED_BODIES),
    default_joint_pos={n: float(v) for n, v in SS.DEFAULT_STAND_POSE.items()},
    action_scale=_action_scale(),
    com_body="torso_link",
    com_range=((-0.025, 0.025), (-0.05, 0.05), (-0.05, 0.05)),
    robot_cfg=_robot_cfg,
    motion_dir=MOTION_DIR,
    motion_glob="*.npz",
    load_motion=load_frax_npz,
    feet_bodies=("left_foot_link", "right_foot_link"),
    contact_z=FOOT_ORIGIN_HEIGHT + 0.010,  # as G1: flat-foot origin height + 1 cm
    mirror_sign=_mirror_signs(),
)
