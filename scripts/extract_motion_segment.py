#!/usr/bin/env python3
"""Extract a time slice from a motion NPZ clip.

Slices all time-varying arrays along the motion timeline using Python slice
semantics: ``--step_range a:b`` selects frames ``a`` (inclusive) through ``b``
(exclusive), i.e. ``original[a:b]``.

Output is written next to the input file as ``<stem>_[a|b].npz``.

Example:
    python scripts/extract_motion_segment.py \
        --motion_file data/LAFAN1_Retargeting_Dataset/g1/jumps1_subject2.npz \
        --step_range 1100:1700
"""

from __future__ import annotations

import argparse
from pathlib import Path

import numpy as np

_TIME_VARYING_KEYS = frozenset(
    {
        "joint_pos",
        "joint_vel",
        "body_pos_w",
        "body_quat_w",
        "body_lin_vel_w",
        "body_ang_vel_w",
        "action",
        "actions",
        "joint_action",
        "joint_actions",
    }
)
_PASSTHROUGH_KEYS = frozenset({"fps"})


def _parse_step_range(value: str) -> tuple[int, int]:
    if ":" not in value:
        raise argparse.ArgumentTypeError(f"Expected 'start:end' slice syntax, got '{value}'.")
    start_str, end_str = value.split(":", 1)
    try:
        start = int(start_str)
        end = int(end_str)
    except ValueError as exc:
        raise argparse.ArgumentTypeError(f"Slice bounds must be integers, got '{value}'.") from exc
    if start < 0 or end < 0:
        raise argparse.ArgumentTypeError(f"Slice bounds must be non-negative, got '{value}'.")
    if start >= end:
        raise argparse.ArgumentTypeError(f"Slice start must be < end, got '{value}'.")
    return start, end


def _infer_time_axis(arr: np.ndarray, reference_time_steps: int, time_axis: int | None) -> int | None:
    if arr.ndim == 0:
        return None
    if time_axis is not None:
        if arr.shape[time_axis] == reference_time_steps:
            return time_axis
        return None
    for axis in range(arr.ndim):
        if arr.shape[axis] == reference_time_steps:
            return axis
    return None


def _slice_array(arr: np.ndarray, start: int, end: int, time_axis: int) -> np.ndarray:
    slices = [slice(None)] * arr.ndim
    slices[time_axis] = slice(start, end)
    return np.ascontiguousarray(arr[tuple(slices)])


def _reference_time_steps(data: np.lib.npyio.NpzFile) -> tuple[int, int | None]:
    if "joint_pos" not in data.files:
        raise KeyError("Motion NPZ must contain `joint_pos` to infer timeline length.")
    joint_pos = np.asarray(data["joint_pos"])
    if joint_pos.ndim == 2:
        return int(joint_pos.shape[0]), 0
    if joint_pos.ndim == 3:
        return int(joint_pos.shape[1]), 1
    raise ValueError(
        f"Unsupported `joint_pos` rank {joint_pos.ndim}. Expected [T, D] or [N_traj, T, D]."
    )


def extract_motion_segment(motion_file: Path, start: int, end: int) -> Path:
    motion_file = motion_file.expanduser().resolve()
    if not motion_file.is_file():
        raise FileNotFoundError(f"Motion file not found: {motion_file}")

    with np.load(motion_file, allow_pickle=True) as data:
        time_steps, time_axis = _reference_time_steps(data)
        if end > time_steps:
            raise ValueError(
                f"Slice end {end} exceeds motion length {time_steps} in '{motion_file.name}'."
            )

        output: dict[str, np.ndarray] = {}
        unknown_time_varying: list[str] = []

        for key in data.files:
            arr = np.asarray(data[key])

            if key in _PASSTHROUGH_KEYS:
                output[key] = arr
                continue

            if key in _TIME_VARYING_KEYS:
                axis = _infer_time_axis(arr, time_steps, time_axis)
                if axis is None:
                    raise ValueError(
                        f"Could not align `{key}` (shape={arr.shape}) with timeline length {time_steps}."
                    )
                output[key] = _slice_array(arr, start, end, axis)
                continue

            axis = _infer_time_axis(arr, time_steps, time_axis)
            if axis is not None:
                unknown_time_varying.append(key)
                output[key] = _slice_array(arr, start, end, axis)
            else:
                output[key] = arr

        if unknown_time_varying:
            print(
                "[INFO] Also sliced unrecognized time-varying keys: "
                + ", ".join(sorted(unknown_time_varying))
            )

    out_name = f"{motion_file.stem}_[{start}|{end}].npz"
    out_path = motion_file.with_name(out_name)
    np.savez(out_path, **output)

    extracted_steps = end - start
    print(f"[INFO] Input:  {motion_file}")
    print(f"[INFO] Slice:  [{start}:{end}) -> {extracted_steps} frames")
    print(f"[INFO] Output: {out_path}")
    return out_path


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Extract a motion NPZ segment using Python slice semantics (start inclusive, end exclusive)."
    )
    parser.add_argument(
        "--motion_file",
        type=Path,
        required=True,
        help="Path to the input motion .npz file.",
    )
    parser.add_argument(
        "--step_range",
        type=_parse_step_range,
        required=True,
        help="Frame slice as start:end (0-based, end exclusive). Example: 1100:1700.",
    )
    args = parser.parse_args()
    start, end = args.step_range
    extract_motion_segment(args.motion_file, start, end)


if __name__ == "__main__":
    main()
