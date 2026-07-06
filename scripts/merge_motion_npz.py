#!/usr/bin/env python3
"""Merge stacked motion NPZ files without materializing the full dataset in RAM.

The output is the stacked format consumed by ``MotionLoader``:

    joint_pos:       [N_total, T_max, D]
    body_pos_w:     [N_total, T_max, B, 3]
    valid_lengths:  [N_total]

Inputs may be single-clip ``[T, ...]`` files or stacked ``[N, T, ...]`` files.
If an input has ``valid_lengths``, only each trajectory's valid prefix is copied;
the merged output is padded by repeating the last valid frame.
"""

from __future__ import annotations

import argparse
import shutil
import tempfile
from dataclasses import dataclass
from pathlib import Path

import numpy as np

REQUIRED_KEYS = (
    "joint_pos",
    "joint_vel",
    "body_pos_w",
    "body_quat_w",
    "body_lin_vel_w",
    "body_ang_vel_w",
)
ACTION_KEYS = ("actions", "action", "joint_action", "joint_actions")
INITIAL_KEYS = tuple(f"initial_{key}" for key in REQUIRED_KEYS)

STATIC_METADATA_KEYS = (
    "joint_names",
    "body_names",
    "action_scale",
    "default_joint_pos",
    "action_mode",
    "backend",
    "control_dt",
    "sim_dt",
    "startup_qpos_joint_range",
)


@dataclass(frozen=True)
class InputInfo:
    path: Path
    num_traj: int
    time_steps: int
    valid_lengths: np.ndarray
    fps: float | None
    action_key: str | None
    metadata: dict[str, np.ndarray]


def _load_key(path: Path, key: str) -> np.ndarray:
    """Load one key from an NPZ and immediately close the archive."""

    with np.load(path, allow_pickle=True) as data:
        return np.asarray(data[key])


def _action_key(files: list[str]) -> str | None:
    for key in ACTION_KEYS:
        if key in files:
            return key
    return None


def _shape_num_traj_and_steps(joint_pos_shape: tuple[int, ...], path: Path) -> tuple[int, int]:
    if len(joint_pos_shape) == 2:
        return 1, int(joint_pos_shape[0])
    if len(joint_pos_shape) == 3:
        return int(joint_pos_shape[0]), int(joint_pos_shape[1])
    raise ValueError(
        f"Unsupported `joint_pos` shape {joint_pos_shape} in '{path}'. "
        "Expected [T, D] or [N_traj, T, D]."
    )


def _valid_lengths(path: Path, num_traj: int, time_steps: int, files: list[str]) -> np.ndarray:
    if "valid_lengths" not in files:
        return np.full(num_traj, time_steps, dtype=np.int64)
    lengths = np.asarray(_load_key(path, "valid_lengths"), dtype=np.int64).reshape(-1)
    if lengths.shape[0] != num_traj:
        raise ValueError(
            f"`valid_lengths` length {lengths.shape[0]} in '{path}' does not match "
            f"trajectory count {num_traj}."
        )
    if np.any(lengths <= 0) or np.any(lengths > time_steps):
        raise ValueError(
            f"Invalid `valid_lengths` in '{path}': expected values in [1, {time_steps}], "
            f"got min={int(lengths.min())}, max={int(lengths.max())}."
        )
    return lengths


def _fps(path: Path, files: list[str]) -> float | None:
    if "fps" not in files:
        return None
    values = np.asarray(_load_key(path, "fps")).reshape(-1)
    return None if values.size == 0 else float(values[0])


def _scan_input(path: Path) -> InputInfo:
    path = path.expanduser().resolve()
    if not path.is_file():
        raise FileNotFoundError(f"Motion file not found: {path}")

    with np.load(path, allow_pickle=True) as data:
        files = list(data.files)
        missing = [key for key in REQUIRED_KEYS if key not in files]
        if missing:
            raise KeyError(f"Motion file '{path}' is missing required keys: {missing}")
        joint_pos_shape = tuple(np.asarray(data["joint_pos"]).shape)
        metadata = {key: np.asarray(data[key]) for key in STATIC_METADATA_KEYS if key in files}

    num_traj, time_steps = _shape_num_traj_and_steps(joint_pos_shape, path)
    return InputInfo(
        path=path,
        num_traj=num_traj,
        time_steps=time_steps,
        valid_lengths=_valid_lengths(path, num_traj, time_steps, files),
        fps=_fps(path, files),
        action_key=_action_key(files),
        metadata=metadata,
    )


def _select_traj(arr: np.ndarray, input_info: InputInfo, traj_idx: int, valid_len: int) -> np.ndarray:
    if input_info.num_traj == 1:
        return np.asarray(arr[:valid_len], dtype=np.float32)
    return np.asarray(arr[traj_idx, :valid_len], dtype=np.float32)


def _copy_time_series_key(
    key: str,
    inputs: list[InputInfo],
    out_dir: Path,
    total_traj: int,
    max_steps: int,
) -> np.memmap:
    first = _load_key(inputs[0].path, key)
    first_valid = _select_traj(first, inputs[0], 0, int(inputs[0].valid_lengths[0]))
    tail_shape = first_valid.shape[1:]
    out_shape = (total_traj, max_steps, *tail_shape)
    out = np.lib.format.open_memmap(out_dir / f"{key}.npy", mode="w+", dtype=np.float32, shape=out_shape)

    out_idx = 0
    for input_info in inputs:
        arr = _load_key(input_info.path, key)
        for traj_idx, valid_len_raw in enumerate(input_info.valid_lengths.tolist()):
            valid_len = int(valid_len_raw)
            seq = _select_traj(arr, input_info, traj_idx, valid_len)
            if seq.shape[1:] != tail_shape:
                raise ValueError(
                    f"`{key}` feature shape mismatch in '{input_info.path}': "
                    f"expected {tail_shape}, got {seq.shape[1:]}."
                )
            out[out_idx, :valid_len] = seq
            if valid_len < max_steps:
                out[out_idx, valid_len:max_steps] = seq[-1]
            out_idx += 1

    out.flush()
    return out


def _copy_initial_key(
    key: str,
    inputs: list[InputInfo],
    out_dir: Path,
    total_traj: int,
) -> np.memmap | None:
    for info in inputs:
        with np.load(info.path, allow_pickle=True) as data:
            if key not in data.files:
                return None

    first = _load_key(inputs[0].path, key)
    first_item = np.asarray(first[0] if inputs[0].num_traj > 1 else first, dtype=np.float32)
    out_shape = (total_traj, *first_item.shape)
    out = np.lib.format.open_memmap(out_dir / f"{key}.npy", mode="w+", dtype=np.float32, shape=out_shape)

    out_idx = 0
    for input_info in inputs:
        arr = _load_key(input_info.path, key)
        for traj_idx in range(input_info.num_traj):
            item = np.asarray(arr[traj_idx] if input_info.num_traj > 1 else arr, dtype=np.float32)
            if item.shape != first_item.shape:
                raise ValueError(
                    f"`{key}` feature shape mismatch in '{input_info.path}': "
                    f"expected {first_item.shape}, got {item.shape}."
                )
            out[out_idx] = item
            out_idx += 1

    out.flush()
    return out


def _common_metadata(inputs: list[InputInfo]) -> tuple[dict[str, np.ndarray], list[str]]:
    kept: dict[str, np.ndarray] = {}
    dropped: list[str] = []
    for key in STATIC_METADATA_KEYS:
        values = [info.metadata[key] for info in inputs if key in info.metadata]
        if len(values) != len(inputs):
            continue
        first = values[0]
        if all(np.array_equal(first, value) for value in values[1:]):
            kept[key] = first
        else:
            dropped.append(key)
    return kept, dropped


def merge_motion_files(input_paths: list[Path], output_path: Path, fps: float | None = None) -> Path:
    if not input_paths:
        raise ValueError("At least one input file is required.")

    inputs = [_scan_input(path) for path in input_paths]
    total_traj = sum(info.num_traj for info in inputs)
    valid_lengths = np.concatenate([info.valid_lengths for info in inputs]).astype(np.int64)
    max_steps = int(valid_lengths.max())

    action_keys = [info.action_key for info in inputs]
    action_key = next((key for key in ACTION_KEYS if key in action_keys), None)
    if action_key is not None and any(key != action_key for key in action_keys):
        raise ValueError(
            f"Inconsistent action keys across inputs: {action_keys}. "
            "Use inputs that all provide the same action key or no action key."
        )

    fps_values = [fps if fps is not None else info.fps for info in inputs for _ in range(info.num_traj)]
    if any(value is None or not np.isfinite(float(value)) for value in fps_values):
        raise ValueError("Could not resolve fps for all inputs. Pass --fps or ensure each input has `fps`.")
    unique_fps = sorted({float(value) for value in fps_values})
    fps_payload = np.array([unique_fps[0]], dtype=np.float32) if len(unique_fps) == 1 else np.asarray(fps_values, dtype=np.float32)

    source_files = np.asarray(
        [f"{info.path.name}#{traj_idx}" for info in inputs for traj_idx in range(info.num_traj)],
        dtype=np.object_,
    )
    metadata, dropped_metadata = _common_metadata(inputs)

    output_path = output_path.expanduser().resolve()
    output_path.parent.mkdir(parents=True, exist_ok=True)

    temp_dir = Path(tempfile.mkdtemp(prefix=f".{output_path.stem}_merge_", dir=output_path.parent))
    try:
        payload: dict[str, np.ndarray] = {
            "fps": fps_payload,
            "valid_lengths": valid_lengths,
            "source_files": source_files,
            "num_motions": np.array([total_traj], dtype=np.int32),
            **metadata,
        }

        time_keys = list(REQUIRED_KEYS)
        if action_key is not None:
            time_keys.append(action_key)

        for key in time_keys:
            print(f"[INFO] Copying {key} ...", flush=True)
            payload[key] = _copy_time_series_key(key, inputs, temp_dir, total_traj, max_steps)

        for key in INITIAL_KEYS:
            initial = _copy_initial_key(key, inputs, temp_dir, total_traj)
            if initial is not None:
                payload[key] = initial

        print("[INFO] Writing final NPZ ...", flush=True)
        np.savez(output_path, **payload)
    finally:
        shutil.rmtree(temp_dir, ignore_errors=True)

    print(f"[INFO] Inputs:        {len(inputs)} file(s)")
    print(f"[INFO] Trajectories:  {total_traj}")
    print(f"[INFO] Max steps:     {max_steps}")
    print(f"[INFO] valid_lengths: min={int(valid_lengths.min())}, max={int(valid_lengths.max())}")
    print(f"[INFO] fps:           {fps_payload.reshape(-1).tolist()}")
    if metadata:
        print(f"[INFO] Metadata kept: {', '.join(sorted(metadata.keys()))}")
    if dropped_metadata:
        print(f"[INFO] Metadata dropped (not identical): {', '.join(sorted(dropped_metadata))}")
    print(f"[INFO] Output:        {output_path}")
    return output_path


def main() -> None:
    parser = argparse.ArgumentParser(description="Merge stacked motion NPZ files into one stacked dataset.")
    parser.add_argument("input_files", nargs="+", type=Path, help="Input stacked motion NPZ files.")
    parser.add_argument("--output", type=Path, required=True, help="Output merged NPZ path.")
    parser.add_argument("--fps", type=float, default=None, help="Optional FPS override.")
    args = parser.parse_args()
    merge_motion_files(args.input_files, args.output, fps=args.fps)


if __name__ == "__main__":
    main()
