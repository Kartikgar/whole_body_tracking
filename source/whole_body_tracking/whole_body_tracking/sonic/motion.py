"""Read repository G1 NPZ motions without importing a simulator."""
from pathlib import Path

import numpy as np

from .spec import BODY_NAMES, CONTROL_DT, JOINT_NAMES

FIELDS = ("joint_pos", "joint_vel", "body_pos_w", "body_quat_w", "body_lin_vel_w", "body_ang_vel_w")


def slerp(q0, q1, weight):
    """Shortest-arc quaternion interpolation, with wxyz component order."""
    dot = np.sum(q0 * q1, axis=-1, keepdims=True)
    q1 = np.where(dot < 0, -q1, q1)
    angle = np.arccos(np.clip(np.abs(dot), 0, 1))
    sine = np.sin(angle)
    a = np.sin((1 - weight) * angle) / np.maximum(sine, 1e-8)
    b = np.sin(weight * angle) / np.maximum(sine, 1e-8)
    out = np.where(sine > 1e-6, a * q0 + b * q1, (1 - weight) * q0 + weight * q1)
    return out / np.linalg.norm(out, axis=-1, keepdims=True)


class SonicMotion:
    def __init__(self, path, trajectory_index=0):
        self.path = str(Path(path).resolve())
        self.trajectory_index = int(trajectory_index)
        with np.load(path, allow_pickle=True) as data:
            missing = set(FIELDS) - set(data.files)
            if missing:
                raise ValueError(f"SONIC requires a G1 motion NPZ; missing fields: {sorted(missing)}")
            stacked = data["joint_pos"].ndim == 3
            count = data["joint_pos"].shape[0] if stacked else 1
            if not 0 <= self.trajectory_index < count:
                raise ValueError(f"trajectory_index must be in [0, {count}), got {trajectory_index}")
            self.data = {k: np.asarray(data[k][self.trajectory_index] if stacked else data[k], dtype=np.float32)
                         for k in FIELDS}
            total = len(self.data["joint_pos"])
            lengths = np.asarray(data.get("valid_lengths", [total] * count)).reshape(-1)
            if len(lengths) != count or np.any(lengths != lengths.astype(int)) or np.any(lengths < 1) or np.any(lengths > total):
                raise ValueError("Invalid valid_lengths in motion NPZ")
            length = int(lengths[self.trajectory_index])
            fps = np.asarray(data.get("fps", [])).reshape(-1)
            if len(fps) not in (1, count) or not np.all(np.isfinite(fps)) or np.any(fps <= 0):
                raise ValueError("Motion NPZ requires positive fps (scalar or per trajectory)")
            self.source_fps = float(fps[0 if len(fps) == 1 else self.trajectory_index])
            joints = self._names(data, "joint_names", JOINT_NAMES, self.data["joint_pos"].shape[-1])
            bodies = self._names(data, "body_names", BODY_NAMES, self.data["body_pos_w"].shape[-2])
            jidx = [joints.index(name) for name in JOINT_NAMES]
            bidx = [bodies.index(name) for name in BODY_NAMES]
            for key in FIELDS:
                arr = self.data[key]
                tail = (len(joints),) if key.startswith("joint") else (len(bodies), 4 if key == "body_quat_w" else 3)
                if arr.shape != (total, *tail) or not np.all(np.isfinite(arr[:length])):
                    raise ValueError(f"Invalid {key} shape/values: {arr.shape}")
                self.data[key] = arr[:length, jidx] if key.startswith("joint") else arr[:length, bidx]
        quat = self.data["body_quat_w"]
        norm = np.linalg.norm(quat, axis=-1, keepdims=True)
        if np.any(norm < 1e-6):
            raise ValueError("Motion contains zero quaternions")
        self.data["body_quat_w"] = quat / norm
        duration = (length - 1) / self.source_fps
        # Include only ticks lying within the valid source interval.
        times = np.arange(int(np.floor(duration / CONTROL_DT + 1e-6)) + 1) * CONTROL_DT * self.source_fps
        lo = np.minimum(np.floor(times).astype(int), length - 1)
        hi = np.minimum(lo + 1, length - 1)
        for key, arr in self.data.items():
            w = (times - lo).reshape((-1,) + (1,) * (arr.ndim - 1))
            self.data[key] = (slerp(arr[lo], arr[hi], w) if key == "body_quat_w"
                              else arr[lo] * (1 - w) + arr[hi] * w).astype(np.float32)
        self.length = len(times)

    @staticmethod
    def _names(data, key, canonical, size):
        if key not in data.files:
            if size != len(canonical):
                raise ValueError(f"Unnamed {key} has {size} entries; expected legacy repository G1 layout ({len(canonical)})")
            return list(canonical)
        names = [v.decode() if isinstance(v, bytes) else str(v) for v in data[key].reshape(-1)]
        if len(names) != size or len(set(names)) != len(names) or not set(canonical).issubset(names):
            raise ValueError(f"{key} does not contain a complete, unique G1 mapping")
        return names

    def reference_at(self, step, batch_size=1):
        index = int(np.clip(step, 0, self.length - 1))
        return {k: np.repeat(v[index:index + 1], batch_size, axis=0) for k, v in self.data.items()}

    def future(self, key, step, frames=10, stride=5):
        indices = np.clip(int(step) + np.arange(frames) * stride, 0, self.length - 1)
        return self.data[key][indices]
