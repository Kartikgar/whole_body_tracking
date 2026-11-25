"""frax retargeting configuration for LAFAN1 -> Fauna Sprout.

Ported from frax's examples/ik/lafan1/lafan1_g1.py (whose tracked bones, weights and
rotation offsets come from GMR). Differences for Sprout:

- Robot: loaded from sprout.urdf; the floating-base root is torso_link (the pelvis hangs
  below it via waist_yaw_joint and shares its origin). Hips -> pelvis_link (position +
  rotation), Spine2 -> torso_link (rotation only), as in the G1 table.
- Rotation offsets: G1's offsets assume its link frames at its zero pose. For each tracked
  link, OFF_sprout = OFF_g1 @ R_g1(ref)^T @ R_sprout(ref), where both robots are put in the
  same reference pose: legs straight, upper arms down, forearms forward (G1's zero pose;
  Sprout: shoulder roll -/+pi/2, elbow -/+pi/2, see SPROUT_REFERENCE_ARMS).
- Scales, pelvis offset and ankle height are computed from Sprout's kinematics and its
  spherized collision model (spherize_sprout.py), with frax's definitions.
- No ankle roll on Sprout: the feet keep GMR's low rotation weight (10), so the foot roll
  target is tracked only as far as the hip allows.
- Grippers are not tracked and are held closed by a narrow retargeting limit.

Run in the frax env; see retarget_lafan1.py.
"""

from __future__ import annotations

import importlib.util
import os
import sys
import xml.etree.ElementTree as ET
from pathlib import Path

os.environ.setdefault("JAX_PLATFORMS", "cpu")
os.environ.setdefault("JAX_ENABLE_X64", "1")

import jax  # noqa: E402
import jax.numpy as jnp  # noqa: E402
import numpy as np  # noqa: E402
import yaml  # noqa: E402
from jax import Array  # noqa: E402
from scipy.spatial.transform import Rotation  # noqa: E402

from frax import load_g1  # noqa: E402
from frax.core.humanoid import Humanoid  # noqa: E402
from frax.ik import (  # noqa: E402
    DiffIKSolver,
    FloorConstraint,
    FrameTask,
    JointLimits,
    Retargeter,
    load_bvh,
)
from frax.utils.collision_utils import bubblify_to_mine  # noqa: E402
from frax.utils.rotation_utils import quat_wxyz_to_rmat, rmat_to_quat_wxyz  # noqa: E402

HERE = Path(__file__).resolve().parent
REPO = HERE.parents[2]
_spec_file = REPO / "source/whole_body_tracking/whole_body_tracking/robots/sprout_spec.py"
_s = importlib.util.spec_from_file_location("sprout_spec", _spec_file)
spec = importlib.util.module_from_spec(_s)
sys.modules.setdefault("sprout_spec", spec)
_s.loader.exec_module(spec)

SPROUT_URDF = spec.SPROUT_URDF_PATH
COLLISION_MODEL = HERE / "assets/sprout_spherized.yml"
LAFAN1_DIR = REPO / "data/lafan1_bvh"
OUT_DIR = REPO / "data/lafan1_frax/sprout"
LAFAN1_FPS = 30
Y_UP_TO_Z_UP = np.array([[1, 0, 0], [0, 0, -1], [0, 1, 0]])

# Sprout joint order used everywhere in this repo (URDF tree order, parents first).
JOINT_ORDERING = tuple(spec.JOINT_NAMES)
ROOT_LINK = spec.ROOT_BODY  # torso_link


def _child_map(urdf: Path) -> dict[str, str]:
    root = ET.parse(urdf).getroot()
    child = {j.get("name"): j.find("child").get("link") for j in root.findall("joint") if j.get("type") == "revolute"}
    return {n: child[n] for n in JOINT_ORDERING}


JOINT_TO_CHILD = _child_map(SPROUT_URDF)

# Upper arms down, forearms forward: the pose G1 has at zero (see module docstring).
SPROUT_REFERENCE_ARMS = {
    "left_shoulder_roll_joint": -np.pi / 2,
    "left_elbow_joint": -np.pi / 2,
    "right_shoulder_roll_joint": np.pi / 2,
    "right_elbow_joint": np.pi / 2,
}


def _sole_height_in_foot_frame() -> float:
    """Distance from the ankle joint (foot_link origin) down to the sole plane."""
    spheres = yaml.safe_load(COLLISION_MODEL.read_text())["collision_spheres"]["left_foot_link"]
    return float(-min(s["center"][2] - s["radius"] for s in spheres))


ANKLE_HEIGHT = _sole_height_in_foot_frame()


def _joint_q(robot: Humanoid, values: dict[str, float], root_z: float = 0.0) -> np.ndarray:
    q = np.zeros(robot.nq)
    q[2] = root_z
    q[3] = 1.0  # identity quaternion (wxyz)
    for name, v in values.items():
        q[7 + robot.joint_name_to_index[name] - robot.nv_floating] = v
    return q


# ---------------------------------------------------------------------------
# Self-collision pairs (left side; mirrored), in the style of frax's G1 table. Without them
# walk1_subject1 had leg-leg interpenetration in 24% of frames (up to 8 cm: Sprout's legs are
# thick for their length) and forearm/hand-in-torso in 77% (Sprout's torso is wide).
# Neighbouring links that always touch are left out.
# ---------------------------------------------------------------------------
_LEG = ("femur_upper", "femur_lower", "tibia", "foot")  # hip roll, hip yaw, knee, ankle children
_FOREARM = ("radius", "wrist", "gripper_lower", "gripper_upper")
_THIGH = ("femur_upper", "femur_lower")


def _side(side: str, names) -> list[str]:
    return [f"{side}_{n}_link" for n in names]


def _mirror(link: str) -> str:
    for a, b in (("left_", "right_"), ("right_", "left_")):
        if link.startswith(a):
            return b + link[len(a):]
    return link


_LEFT_SC_PAIRS = (
    [(a, b) for a in _side("left", _LEG) for b in _side("right", _LEG)]
    + [(a, b) for a in _side("left", _FOREARM) for b in ("torso_link", "pelvis_link")]
    + [(a, b) for a in _side("left", _FOREARM) for b in _side("left", _THIGH) + _side("right", _THIGH)]
    + [(a, b) for a in _side("left", _FOREARM) for b in _side("right", ("humerus_lower",) + _FOREARM)]
)
SELF_COLLISION_LINK_PAIRS = tuple(dict.fromkeys(
    tuple(sorted(p)) for a, b in _LEFT_SC_PAIRS for p in ((a, b), (_mirror(a), _mirror(b)))
))  # fmt: skip
SELF_COLLISION_ROWS = 64  # closest sphere pairs kept per QP step (frax's G1 value)
SELF_COLLISION_MARGIN = 0.005  # m
# Map the human's neutral stance (feet under the hips) to Sprout's (hip roll 0, feet ~24 cm apart):
# each foot target is moved outward along the human hip axis by half the width difference.
STANCE_WIDEN = True


def sprout_sc_data() -> tuple:
    from frax.utils.collision_utils import link_pairs_to_sc_data

    return link_pairs_to_sc_data(COLLISION_MODEL, SELF_COLLISION_LINK_PAIRS)


def load_sprout(with_collision: bool = True, self_collision: bool = True) -> Humanoid:
    """Sprout with a quaternion floating base, standing on the floor with arms in the reference pose."""
    collision = None
    if with_collision:
        collision = bubblify_to_mine(
            COLLISION_MODEL, dict(JOINT_TO_CHILD), root_link_name=ROOT_LINK, add_floating_base=True,
            sc_data=sprout_sc_data() if self_collision else None, verbose=False,
        )
    robot = Humanoid(
        SPROUT_URDF,
        "left_wrist_pitch_joint", "right_wrist_pitch_joint",
        "left_ankle_joint", "right_ankle_joint",
        collision_data=collision,
        joint_ordering=list(JOINT_ORDERING),
        floating_base="quaternion",
    )
    # Root height that puts the soles on the floor with straight legs.
    q0 = _joint_q(robot, SPROUT_REFERENCE_ARMS)
    T = np.asarray(robot.joint_to_world_transforms(q0))
    ankle_z = T[robot.joint_name_to_index["left_ankle_joint"], 2, 3]
    q0[2] = ANKLE_HEIGHT - ankle_z
    robot.default_configuration = jnp.asarray(q0)
    return robot


# ---------------------------------------------------------------------------
# Retargeting table: (Sprout joint whose child link is tracked, LAFAN1 bone, position
# weight, rotation weight, G1 joint used for the G1 offset, G1 rotation offset WXYZ).
# Hips first: preprocess treats row 0 as the pelvis. Weights and G1 offsets are GMR's.
# ---------------------------------------------------------------------------
# fmt: off
_TABLE = (
    ("waist_yaw_joint",            "Hips",         50,  50, "yaw_joint",                (0.5, 0.5, 0.5, 0.5)),
    ("left_hip_yaw_joint",         "LeftUpLeg",    0,   10, "left_hip_yaw_joint",       (-0.5, 0.5, 0.5, -0.5)),
    ("left_knee_joint",            "LeftLeg",      0,   10, "left_knee_joint",          (-0.5, 0.5, 0.5, -0.5)),
    ("left_ankle_joint",           "LeftFootMod",  100, 10, "left_ankle_roll_joint",    (-0.70710678, 0.70710678, 0.0, 0.0)),
    ("right_hip_yaw_joint",        "RightUpLeg",   0,   10, "right_hip_yaw_joint",      (-0.5, 0.5, 0.5, -0.5)),
    ("right_knee_joint",           "RightLeg",     0,   10, "right_knee_joint",         (-0.5, 0.5, 0.5, -0.5)),
    ("right_ankle_joint",          "RightFootMod", 100, 10, "right_ankle_roll_joint",   (-0.70710678, 0.70710678, 0.0, 0.0)),
    ("yaw_joint",                  "Spine2",       0,   50, "waist_pitch_joint",        (0.5, 0.5, 0.5, 0.5)),
    ("left_shoulder_yaw_joint",    "LeftArm",      0,   50, "left_shoulder_yaw_joint",  (0.5, 0.5, -0.5, -0.5)),
    ("left_elbow_joint",           "LeftForeArm",  0,   10, "left_elbow_joint",         (0.70710678, 0.70710678, 0.0, 0.0)),
    ("left_wrist_pitch_joint",     "LeftHand",     10,  10, "left_wrist_yaw_joint",     (0.70710678, 0.70710678, 0.0, 0.0)),
    ("right_shoulder_yaw_joint",   "RightArm",     0,   50, "right_shoulder_yaw_joint", (0.5, 0.5, -0.5, -0.5)),
    ("right_elbow_joint",          "RightForeArm", 0,   10, "right_elbow_joint",        (0.70710678, 0.70710678, 0.0, 0.0)),
    ("right_wrist_pitch_joint",    "RightHand",    10,  10, "right_wrist_yaw_joint",    (0.70710678, 0.70710678, 0.0, 0.0)),
)
# fmt: on
SPROUT_TRACKED_JOINTS = tuple(r[0] for r in _TABLE)
LAFAN1_BONES = tuple(r[1] for r in _TABLE)
WEIGHT_POS = np.array([r[2] for r in _TABLE], dtype=float)
WEIGHT_ROT = np.array([r[3] for r in _TABLE], dtype=float)

# Gripper held near closed (not tracked); a zero-width interval would make the QP degenerate.
RETARGETING_JOINT_LIMITS = {
    "left_gripper_joint": (-0.01, 0.01),
    "right_gripper_joint": (-0.01, 0.01),
}
# Joint velocity cap used while retargeting (rad/s). Sprout's URDF gives every joint a 5 rad/s
# placeholder; frax's JointLimits would enforce it and throttle fast motions (19% of LAFAN1
# frames saturated at 5 rad/s in a first run). 30 rad/s is G1-like (its URDF: 20-37 rad/s),
# so retargeting reproduces the human motion and feasibility is left to the tracker.
RETARGET_V_MAX = 30.0
GAIN = 30.0
DAMPING = 10.0
SHOULDER_DAMPING = 1000.0
WRIST_DAMPING = 100.0
# Hip yaw (thigh axial rotation) is weakly observed by the targets; without extra damping it
# can flip ~1 rad in one frame (seen in fightAndSports1_subject1 once v_max was raised).
HIP_YAW_DAMPING = 10.0  # = DAMPING, i.e. no extra damping; tuned below if needed
HIP_YAW_JOINTS = ("left_hip_yaw_joint", "right_hip_yaw_joint")
SMOOTH_ROT = 0.05
SHOULDER_JOINTS = tuple(f"{s}_shoulder_{a}_joint" for s in ("left", "right") for a in ("pitch", "roll", "yaw"))
WRIST_JOINTS = tuple(f"{s}_wrist_{a}_joint" for s in ("left", "right") for a in ("roll", "pitch"))

# LAFAN1 proportions (frax's values, m).
LAFAN1_LEG = 0.4350 + 0.4237
LAFAN1_UPPER = 0.5375 + 0.3300 + 0.2520
LAFAN1_ANKLE_HEIGHT = 0.077


def _geometry(robot: Humanoid) -> dict:
    """Sprout lengths/offsets with frax's G1 definitions, plus rotation offsets."""
    T = np.asarray(robot.joint_to_world_transforms(np.asarray(robot.default_configuration)))
    p = lambda n: T[robot.joint_name_to_index[n], :3, 3]  # noqa: E731
    hip_l, hip_r = p("left_hip_pitch_joint"), p("right_hip_pitch_joint")
    hip_center = 0.5 * (hip_l + hip_r)
    leg = np.linalg.norm(hip_l - p("left_knee_joint")) + np.linalg.norm(p("left_knee_joint") - p("left_ankle_joint"))
    upper = (
        np.linalg.norm(hip_center - p("left_shoulder_roll_joint"))
        + np.linalg.norm(p("left_shoulder_roll_joint") - p("left_elbow_joint"))
        + np.linalg.norm(p("left_elbow_joint") - p("left_wrist_pitch_joint"))
    )
    pelvis = T[robot.joint_name_to_index["waist_yaw_joint"]]
    pelvis_offset = pelvis[:3, :3].T @ (pelvis[:3, 3] - hip_center)
    stance_width = float(np.linalg.norm(p("left_ankle_joint") - p("right_ankle_joint")))  # legs at zero

    g1 = load_g1()
    T_g1 = np.asarray(g1.joint_to_world_transforms(np.asarray(g1.default_configuration)))
    offsets = []
    for s_joint, _, _, _, g1_joint, off_wxyz in _TABLE:
        off_g1 = Rotation.from_quat(off_wxyz, scalar_first=True).as_matrix()
        R_g1 = T_g1[g1.joint_name_to_index[g1_joint], :3, :3]
        R_s = T[robot.joint_name_to_index[s_joint], :3, :3]
        offsets.append(off_g1 @ R_g1.T @ R_s)
    return {
        "leg": leg, "upper": upper, "pelvis_offset": pelvis_offset, "rot_offsets": np.stack(offsets),
        "stance_width": stance_width,
    }


def make_retargeter(
    robot: Humanoid,
    floor: bool = True,
    v_max: float | None = RETARGET_V_MAX,
    hip_yaw_damping: float = HIP_YAW_DAMPING,
    self_collision: bool = True,
    stance_widen: bool = STANCE_WIDEN,
) -> tuple[Retargeter, dict]:
    """Human bone poses (n, 7), in LAFAN1_BONES order -> Sprout configuration.

    v_max: uniform joint velocity cap (rad/s); None = the URDF's (5 rad/s placeholder).
    self_collision: keep SELF_COLLISION_LINK_PAIRS apart (needs load_sprout(self_collision=True)).
    stance_widen: move the foot targets outward to Sprout's neutral stance width (STANCE_WIDEN)."""
    geo = _geometry(robot)
    leg_scale = geo["leg"] / LAFAN1_LEG
    upper_scale = geo["upper"] / LAFAN1_UPPER
    floor_offset = np.array([0.0, 0.0, leg_scale * LAFAN1_ANKLE_HEIGHT - ANKLE_HEIGHT])
    rot_offsets = geo["rot_offsets"]
    pelvis_offset = geo["pelvis_offset"]
    hips_idx = np.array([LAFAN1_BONES.index(b) for b in ("LeftUpLeg", "RightUpLeg")])
    feet_idx = np.array([LAFAN1_BONES.index(b) for b in ("LeftFootMod", "RightFootMod")])
    stance_width = geo["stance_width"]

    def preprocess(human_poses: Array) -> dict:
        pos = human_poses[:, :3]
        rmats = jax.vmap(quat_wxyz_to_rmat)(human_poses[:, 3:]) @ rot_offsets
        hips = jnp.mean(pos[hips_idx], axis=0)
        targets = leg_scale * hips + upper_scale * (pos - hips)
        targets = targets.at[0].set(leg_scale * hips + rmats[0] @ pelvis_offset)
        feet = leg_scale * pos[feet_idx]
        if stance_widen:
            axis = (pos[hips_idx[0]] - pos[hips_idx[1]]).at[2].set(0.0)  # right hip -> left hip, horizontal
            width = jnp.linalg.norm(axis)
            widen = 0.5 * jnp.maximum(stance_width - leg_scale * width, 0.0)
            shift = widen * axis / jnp.maximum(width, 1e-6)
            feet = feet + jnp.stack((shift, -shift))
        targets = targets.at[feet_idx].set(feet)
        return {"frame_pos": targets - floor_offset, "frame_rmat": rmats}

    def initial_q(human_poses: Array) -> Array:
        ref = preprocess(human_poses)
        q = jnp.asarray(robot.default_configuration).at[:3].set(ref["frame_pos"][0])
        return q.at[3:7].set(rmat_to_quat_wxyz(ref["frame_rmat"][0]))

    task = FrameTask(
        robot, [robot.joint_name_to_index[n] for n in SPROUT_TRACKED_JOINTS],
        gain_pos=GAIN, gain_rot=GAIN, weight_pos=WEIGHT_POS, weight_rot=WEIGHT_ROT,
    )
    q_min = np.array(robot.actuated_joint_lower_limits, dtype=float)
    q_max = np.array(robot.actuated_joint_upper_limits, dtype=float)
    for name, (lo, hi) in RETARGETING_JOINT_LIMITS.items():
        i = robot.joint_name_to_index[name] - robot.nv_floating
        q_min[i], q_max[i] = lo, hi
    v_lim = "robot" if v_max is None else np.full(len(q_min), float(v_max))
    constraints = [JointLimits(robot, q_min=q_min, q_max=q_max, v_max=v_lim)]
    if floor:
        constraints.append(FloorConstraint(robot, gain=GAIN))
    if self_collision:
        from frax.ik import SelfCollisionConstraint

        constraints.append(SelfCollisionConstraint(
            robot, max_rows=SELF_COLLISION_ROWS, gain=GAIN, margin=SELF_COLLISION_MARGIN,
        ))  # fmt: skip
    reg = np.full(robot.nv, DAMPING)
    for joints, value in (
        (SHOULDER_JOINTS, SHOULDER_DAMPING),
        (WRIST_JOINTS, WRIST_DAMPING),
        (HIP_YAW_JOINTS, hip_yaw_damping),
    ):
        reg[[robot.joint_name_to_index[n] for n in joints]] = value
    solver = DiffIKSolver(robot, [task], constraints, regularization=reg)
    info = {
        "leg_scale": leg_scale, "upper_scale": upper_scale, "floor_offset": floor_offset,
        "pelvis_offset": pelvis_offset, "ankle_height": ANKLE_HEIGHT, "leg": geo["leg"], "upper": geo["upper"],
        "v_max": v_max, "hip_yaw_damping": hip_yaw_damping,
        "self_collision": self_collision, "stance_widen": stance_widen, "stance_width": stance_width,
    }
    return Retargeter(solver, preprocess, initial_q), info


def load_lafan1(filepath: str | Path) -> tuple[np.ndarray, list[str]]:
    """Global bone poses (T, B, 7) [pos, wxyz] with GMR's Left/RightFootMod bones added."""
    poses, names = load_bvh(filepath, Y_UP_TO_Z_UP)
    idx = {n: i for i, n in enumerate(names)}
    feet = [
        np.concatenate([poses[:, idx[foot], :3], poses[:, idx[toe], 3:]], axis=-1)
        for foot, toe in (("LeftFoot", "LeftToe"), ("RightFoot", "RightToe"))
    ]
    return np.concatenate([poses, np.stack(feet, axis=1)], axis=1), names + ["LeftFootMod", "RightFootMod"]


def load_clip(path: str | Path) -> np.ndarray:
    """LAFAN1_BONES poses (T, n, 7) from one BVH file, orientation-smoothed (offline, no lag)."""
    from frax.ik import smooth_poses

    poses, names = load_lafan1(path)
    frames = poses[:, [names.index(b) for b in LAFAN1_BONES]]
    return smooth_poses(frames, 1.0 / LAFAN1_FPS, sigma_rot=SMOOTH_ROT)
