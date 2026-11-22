"""Consistency checks between the Sprout spec, its URDF/MJCF and the SDK hardware order."""

from __future__ import annotations

import importlib.util
import re
import sys
import xml.etree.ElementTree as ET
from pathlib import Path

import pytest

_SPEC_PATH = (
    Path(__file__).resolve().parents[1]
    / "source/whole_body_tracking/whole_body_tracking/robots/sprout_spec.py"
)
_spec = importlib.util.spec_from_file_location("sprout_spec", _SPEC_PATH)
sprout = importlib.util.module_from_spec(_spec)
sys.modules["sprout_spec"] = sprout
_spec.loader.exec_module(sprout)

_WS = Path(__file__).resolve().parents[2]
_JOINT_ENUM = _WS / "fauna-sdk/fauna-sdk/libraries/ros2/src/fauna_msgs/msg/BipedJointNames.msg"
_ROBOT_CONFIG = _WS / "fauna-sdk/fauna-sdk/ansible/files/robot_config.json"


def _mjcf():
    # sprout.xml declares XML version '1' (not '1.0'), which expat rejects; MuJoCo ignores it.
    text = re.sub(r"^<\?xml[^>]*\?>", "", sprout.SPROUT_MJCF_PATH.read_text())
    root = ET.fromstring(text)
    joints = {j.get("name"): j for j in root.iter("joint") if j.get("type") == "hinge" and j.get("name")}
    defaults = {d.get("class"): d for d in root.iter("default") if d.get("class")}
    return root, joints, defaults


def test_joint_count_and_uniqueness():
    assert sprout.NUM_JOINTS == 27
    assert len(set(sprout.JOINT_NAMES)) == 27
    assert set(sprout.LOWER_BODY_JOINTS) | set(sprout.UPPER_BODY_JOINTS) == set(sprout.JOINT_NAMES)


def test_mjcf_joints_limits_and_classes_match():
    _, joints, _ = _mjcf()
    assert set(joints) == set(sprout.JOINT_NAMES)
    for name, (lo, hi) in sprout.JOINT_LIMITS.items():
        mlo, mhi = map(float, joints[name].get("range").split())
        assert mlo == pytest.approx(lo, abs=1e-4) and mhi == pytest.approx(hi, abs=1e-4), name
        expected = "lower_body" if name in sprout.LOWER_BODY_JOINTS else "upper_body"
        assert joints[name].get("class") == expected, name


def test_mjcf_joint_order_matches():
    root, _, _ = _mjcf()
    order = tuple(j.get("name") for j in root.iter("joint") if j.get("type") == "hinge" and j.get("name"))
    assert order == sprout.MJCF_JOINT_ORDER


def test_mjcf_gains_match():
    _, _, defaults = _mjcf()
    robot_joint = defaults["robot"].find("joint")
    assert float(robot_joint.get("armature")) == sprout.ARMATURE
    assert float(robot_joint.get("damping")) == sprout.PASSIVE_DAMPING
    for cls, kp in (("lower_body", sprout.KP_LOWER_BODY), ("upper_body", sprout.KP_UPPER_BODY)):
        pos = defaults[cls].find("position")
        assert float(pos.get("kp")) == kp
        assert float(pos.get("kv")) == sprout.KD


def test_hardware_order_matches_sdk_enum_and_is_permutation():
    ids = [int(m.group(1)) for m in re.finditer(r"^uint8 \w+=(\d+)", _JOINT_ENUM.read_text(), re.M)]
    assert ids == list(range(27))
    assert sorted(sprout.SIM_TO_HW) == list(range(27))
    for sim_i, hw_i in enumerate(sprout.HW_TO_SIM):
        assert sprout.SIM_TO_HW[hw_i] == sim_i


def test_motor_models_match_robot_config():
    import json

    joint_info = json.loads(_ROBOT_CONFIG.read_text())["joint_info"]
    assert {k: v["motor_model"] for k, v in joint_info.items()} == sprout.MOTOR_MODEL


def test_total_mass():
    root = ET.parse(sprout.SPROUT_URDF_PATH).getroot()
    total = sum(float(m.get("value")) for m in root.iter("mass"))
    assert total == pytest.approx(sprout.TOTAL_MASS_KG, abs=1e-4)


def test_stand_pose_within_soft_limits():
    for name, q in sprout.DEFAULT_STAND_POSE.items():
        lo, hi = sprout.JOINT_LIMITS[name]
        mid, half = 0.5 * (lo + hi), 0.5 * (hi - lo) * 0.9
        assert mid - half <= q <= mid + half, name


def test_render_urdf_is_physics_identical():
    """The render-fixed URDF used by Isaac/Genesis must equal Fauna's byte-for-byte."""
    import filecmp

    assert sprout.SPROUT_RENDER_URDF_PATH.exists(), "run scripts/sprout/make_render_asset.py"
    assert filecmp.cmp(sprout.SPROUT_URDF_PATH, sprout.SPROUT_RENDER_URDF_PATH, shallow=False)
