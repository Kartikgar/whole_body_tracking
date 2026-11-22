"""Convert a Sprout Foxglove recording (sensors.jsonl) into a compact .npz in MJCF joint order.

    python convert.py recordings/20261121T230359Z            # writes <dir>/robot_rec.npz

Arrays (matched stream = /motor_control/policy_input/matched_sensor_data, 50 Hz):
    t                (N,)     s, from message stamp, relative to the first message
    q, qd, effort, current, temperature, stale   (N, 27)  MJCF hinge order (JOINT_NAMES)
    imu_quat         (N, 4)   wxyz, imu_link orientation (as published)
    imu_gyro, imu_acc (N, 3)  imu_link frame
    timestamp_diff_ms, has_stale_states (N,)
EKF stream (/sensor/imu/torso/ekf, ~500 Hz): ekf_t, ekf_quat, ekf_gyro, ekf_acc.
"""

from __future__ import annotations

import argparse
import json
import os
from pathlib import Path

import numpy as np

# Fauna joint_id (fauna_msgs/BipedJointNames) -> MJCF joint name (sprout-model-v0.1.4).
# THIGH -> hip_yaw and WRIST -> wrist_roll are inferred from names; confirm with a moved-joint recording.
FAUNA_TO_MJCF = {
    0: "left_hip_pitch_joint", 1: "left_hip_roll_joint", 2: "left_knee_joint",
    3: "right_hip_pitch_joint", 4: "right_hip_roll_joint", 5: "right_knee_joint",
    6: "left_ankle_joint", 7: "right_ankle_joint", 8: "waist_yaw_joint",
    9: "left_shoulder_pitch_joint", 10: "left_shoulder_roll_joint",
    11: "right_shoulder_pitch_joint", 12: "right_shoulder_roll_joint",
    13: "left_hip_yaw_joint", 14: "right_hip_yaw_joint",
    15: "left_shoulder_yaw_joint", 16: "left_elbow_joint", 17: "left_wrist_roll_joint", 18: "left_gripper_joint",
    19: "right_shoulder_yaw_joint", 20: "right_elbow_joint", 21: "right_wrist_roll_joint", 22: "right_gripper_joint",
    23: "neck_pitch_joint", 24: "neck_yaw_joint", 25: "left_wrist_pitch_joint", 26: "right_wrist_pitch_joint",
}  # fmt: skip

# MJCF hinge order of sprout.xml
JOINT_NAMES = [
    "waist_yaw_joint", "left_hip_pitch_joint", "left_hip_roll_joint", "left_hip_yaw_joint", "left_knee_joint",
    "left_ankle_joint", "right_hip_pitch_joint", "right_hip_roll_joint", "right_hip_yaw_joint", "right_knee_joint",
    "right_ankle_joint", "neck_yaw_joint", "neck_pitch_joint", "left_shoulder_pitch_joint",
    "left_shoulder_roll_joint", "left_shoulder_yaw_joint", "left_elbow_joint", "left_wrist_roll_joint",
    "left_wrist_pitch_joint", "left_gripper_joint", "right_shoulder_pitch_joint", "right_shoulder_roll_joint",
    "right_shoulder_yaw_joint", "right_elbow_joint", "right_wrist_roll_joint", "right_wrist_pitch_joint",
    "right_gripper_joint",
]  # fmt: skip
COL = {jid: JOINT_NAMES.index(name) for jid, name in FAUNA_TO_MJCF.items()}

MATCHED = "/motor_control/policy_input/matched_sensor_data"
EKF = "/sensor/imu/torso/ekf"


def _t(stamp) -> float:
    return stamp["sec"] + stamp["nanosec"] * 1e-9


def _quat(o):
    return [o["w"], o["x"], o["y"], o["z"]]


def _vec(v):
    return [v["x"], v["y"], v["z"]]


def convert(rec_dir: str, out: str | None = None) -> str:
    recording = Path(rec_dir).expanduser()
    # Recordings live at the workspace root, one level above whole_body_tracking.
    if not recording.is_absolute() and not recording.exists():
        workspace_recording = Path(__file__).resolve().parents[4] / recording
        if workspace_recording.is_dir():
            recording = workspace_recording
    sensor_path = recording / "sensors.jsonl"
    if not sensor_path.is_file():
        raise FileNotFoundError(
            f"Recording sensor file not found: {sensor_path.resolve()}. "
            "Pass an absolute recording directory or a path relative to the current directory "
            "or HumSim2Real workspace root."
        )
    rec_dir = str(recording)
    fields = ("q", "qd", "effort", "current", "temperature", "stale")
    m = {k: [] for k in ("t", *fields, "imu_quat", "imu_gyro", "imu_acc", "timestamp_diff_ms", "has_stale_states")}
    e = {k: [] for k in ("ekf_t", "ekf_quat", "ekf_gyro", "ekf_acc")}
    with open(os.path.join(rec_dir, "sensors.jsonl")) as f:
        for line in f:
            d = json.loads(line)
            if d["topic"] == MATCHED:
                msg = d["data"]
                rows = {k: np.full(27, np.nan) for k in fields}
                for j in msg["joint_states"]["joint_states"]:
                    c = COL[j["joint_id"]]
                    rows["q"][c], rows["qd"][c] = j["position_rad"], j["velocity_rads"]
                    rows["effort"][c], rows["current"][c] = j["effort_nm"], j["current_amp"]
                    rows["temperature"][c], rows["stale"][c] = j["temperature_celsius"], float(j["is_stale"])
                for k in fields:
                    m[k].append(rows[k])
                m["t"].append(_t(msg["stamp"]))
                imu = msg["imu"]
                m["imu_quat"].append(_quat(imu["orientation"]))
                m["imu_gyro"].append(_vec(imu["angular_velocity"]))
                m["imu_acc"].append(_vec(imu["linear_acceleration"]))
                m["timestamp_diff_ms"].append(msg["timestamp_diff_ms"])
                m["has_stale_states"].append(float(msg["has_stale_states"]))
            elif d["topic"] == EKF:
                msg = d["data"]
                e["ekf_t"].append(_t(msg["header"]["stamp"]))
                e["ekf_quat"].append(_quat(msg["orientation"]))
                e["ekf_gyro"].append(_vec(msg["angular_velocity"]))
                e["ekf_acc"].append(_vec(msg["linear_acceleration"]))
    arrs = {k: np.asarray(v, dtype=np.float64) for k, v in {**m, **e}.items()}
    t0 = arrs["t"][0]
    arrs["t"] -= t0
    if len(arrs["ekf_t"]):
        arrs["ekf_t"] -= t0
    out = out or os.path.join(rec_dir, "robot_rec.npz")
    np.savez_compressed(out, joint_names=np.array(JOINT_NAMES), **arrs)
    print(f"{out}: {len(arrs['t'])} matched frames ({arrs['t'][-1]:.1f} s), {len(arrs['ekf_t'])} EKF samples")
    return out


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("rec_dir")
    ap.add_argument("--out", default=None)
    a = ap.parse_args()
    convert(a.rec_dir, a.out)
