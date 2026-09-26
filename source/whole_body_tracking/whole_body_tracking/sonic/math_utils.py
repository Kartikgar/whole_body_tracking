"""Numerical helpers for the modular Genesis sim2sim evaluator."""

from __future__ import annotations

import numpy as np


def quat_conjugate_wxyz(quaternion: np.ndarray) -> np.ndarray:
    """Return the quaternion conjugate assuming ``wxyz`` storage."""

    output = quaternion.copy()
    output[..., 1:] *= -1.0
    return output


def quat_normalize_wxyz(quaternion: np.ndarray) -> np.ndarray:
    """Normalize a quaternion array stored in ``wxyz`` order."""

    norm = np.linalg.norm(quaternion, axis=-1, keepdims=True)
    norm = np.maximum(norm, 1e-8)
    return quaternion / norm


def quat_mul_wxyz(lhs: np.ndarray, rhs: np.ndarray) -> np.ndarray:
    """Multiply two ``wxyz`` quaternion arrays elementwise."""

    w1, x1, y1, z1 = np.moveaxis(lhs, -1, 0)
    w2, x2, y2, z2 = np.moveaxis(rhs, -1, 0)
    w = w1 * w2 - x1 * x2 - y1 * y2 - z1 * z2
    x = w1 * x2 + x1 * w2 + y1 * z2 - z1 * y2
    y = w1 * y2 - x1 * z2 + y1 * w2 + z1 * x2
    z = w1 * z2 + x1 * y2 - y1 * x2 + z1 * w2
    return np.stack([w, x, y, z], axis=-1)


def quat_rotate_wxyz(quaternion: np.ndarray, vector: np.ndarray) -> np.ndarray:
    """Rotate a vector by a ``wxyz`` quaternion."""

    quaternion_xyz = quaternion[..., 1:]
    quaternion_w = quaternion[..., :1]
    temp = 2.0 * np.cross(quaternion_xyz, vector)
    return vector + quaternion_w * temp + np.cross(quaternion_xyz, temp)


def quat_rotate_inverse_wxyz(quaternion: np.ndarray, vector: np.ndarray) -> np.ndarray:
    """Rotate a vector by the inverse of a ``wxyz`` quaternion."""

    return quat_rotate_wxyz(quat_conjugate_wxyz(quaternion), vector)


def quat_to_matrix_wxyz(quaternion: np.ndarray) -> np.ndarray:
    """Convert a ``wxyz`` quaternion array to rotation matrices."""

    normalized = quat_normalize_wxyz(quaternion)
    w, x, y, z = np.moveaxis(normalized, -1, 0)

    ww = w * w
    xx = x * x
    yy = y * y
    zz = z * z
    wx = w * x
    wy = w * y
    wz = w * z
    xy = x * y
    xz = x * z
    yz = y * z

    return np.stack(
        [
            np.stack([ww + xx - yy - zz, 2.0 * (xy - wz), 2.0 * (xz + wy)], axis=-1),
            np.stack([2.0 * (xy + wz), ww - xx + yy - zz, 2.0 * (yz - wx)], axis=-1),
            np.stack([2.0 * (xz - wy), 2.0 * (yz + wx), ww - xx - yy + zz], axis=-1),
        ],
        axis=-2,
    )


def quat_error_magnitude_wxyz(reference: np.ndarray, current: np.ndarray) -> np.ndarray:
    """Return the angular distance between two quaternion arrays."""

    reference_normalized = quat_normalize_wxyz(reference)
    current_normalized = quat_normalize_wxyz(current)
    dot = np.sum(reference_normalized * current_normalized, axis=-1)
    dot = np.clip(np.abs(dot), -1.0, 1.0)
    return 2.0 * np.arccos(dot)


def yaw_quat_wxyz(quaternion: np.ndarray) -> np.ndarray:
    """Extract the yaw-only quaternion from a single ``wxyz`` quaternion."""

    w, x, y, z = quaternion
    yaw = np.arctan2(2.0 * (w * z + x * y), 1.0 - 2.0 * (y * y + z * z))
    return np.array([np.cos(yaw * 0.5), 0.0, 0.0, np.sin(yaw * 0.5)], dtype=np.float32)
