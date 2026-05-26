"""Shared helpers for plotting joint positions in policy-relative space."""

from __future__ import annotations

import re
from pathlib import Path

import numpy as np

JOINT_POS_REL_YLABEL = "Relative joint position (rad)"
JOINT_POS_REL_TITLE = "Joint positions (relative to default)"


def load_default_joint_pos_from_onnx(onnx_path: Path) -> np.ndarray:
    import onnx

    model = onnx.load(str(onnx_path))
    meta = {entry.key: entry.value for entry in model.metadata_props}
    if "default_joint_pos" not in meta:
        raise KeyError(
            f"ONNX metadata missing 'default_joint_pos' in {onnx_path}. "
            f"Available keys: {sorted(meta.keys())}"
        )
    values = [float(item) for item in meta["default_joint_pos"].split(",") if item != ""]
    return np.asarray(values, dtype=np.float32)


def infer_onnx_path_from_npz(npz_path: Path) -> Path | None:
    """Best-effort lookup for `<run_dir>/exported/model_<step>.onnx` from a rollout NPZ name."""
    match = re.match(r"(model_\d+)", npz_path.stem)
    if match is None:
        return None

    checkpoint_stem = match.group(1)
    run_dir = npz_path.parent.parent
    candidates = [
        run_dir / "exported" / f"{checkpoint_stem}.onnx",
        run_dir / "export" / f"{checkpoint_stem}.onnx",
        run_dir / f"{checkpoint_stem}.onnx",
    ]
    for candidate in candidates:
        if candidate.is_file():
            return candidate
    return None


def resolve_default_joint_pos(npz_path: Path, onnx_path: Path | None) -> tuple[np.ndarray, Path]:
    if onnx_path is not None:
        resolved = onnx_path.expanduser().resolve()
        if not resolved.is_file():
            raise FileNotFoundError(f"ONNX file not found: {resolved}")
        return load_default_joint_pos_from_onnx(resolved), resolved

    inferred = infer_onnx_path_from_npz(npz_path)
    if inferred is None:
        raise FileNotFoundError(
            f"Could not infer exported ONNX path from {npz_path.name}. "
            "Pass --default-joint-pos-onnx explicitly."
        )
    return load_default_joint_pos_from_onnx(inferred), inferred


def to_relative_joint_pos(joint_pos: np.ndarray, default_joint_pos: np.ndarray) -> np.ndarray:
    """Convert absolute sim joint positions to IsaacLab ``joint_pos_rel`` coordinates."""
    if joint_pos.ndim != 3:
        raise ValueError(f"Expected joint_pos shape [num_traj, T, D], got {joint_pos.shape}.")
    default = np.asarray(default_joint_pos, dtype=np.float32).reshape(-1)
    if default.shape[0] != joint_pos.shape[-1]:
        raise ValueError(
            f"default_joint_pos length {default.shape[0]} does not match joint dim {joint_pos.shape[-1]}."
        )
    return (joint_pos - default.reshape(1, 1, -1)).astype(np.float32)
