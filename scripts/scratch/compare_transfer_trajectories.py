"""Compare Isaac source states (base+delta) vs Genesis base-only open-loop replay."""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

_SCRATCH_DIR = Path(__file__).resolve().parent
if str(_SCRATCH_DIR) not in sys.path:
    sys.path.insert(0, str(_SCRATCH_DIR))

import matplotlib.pyplot as plt
import numpy as np
from joint_plot_utils import (
    JOINT_POS_REL_YLABEL,
    JOINT_POS_REL_TITLE,
    resolve_default_joint_pos,
    to_relative_joint_pos,
)
from matplotlib.lines import Line2D
from matplotlib.patches import Patch

SOURCE_COLOR = "#2563eb"
TARGET_COLOR = "#ea580c"
BAND_ALPHA = 0.22
MEAN_LW = 1.8

AXIS_LABELS = ("x", "y", "z")


def _parse_step_range(text: str) -> tuple[int, int | None]:
    if ":" not in text:
        raise argparse.ArgumentTypeError(f"Expected start:end step range, got {text!r}.")
    start_text, end_text = text.split(":", 1)
    start = int(start_text) if start_text else 0
    end = int(end_text) if end_text else None
    if start < 0:
        raise argparse.ArgumentTypeError("Step range start must be >= 0.")
    if end is not None and end <= start:
        raise argparse.ArgumentTypeError(f"Step range end must be > start ({start}).")
    return start, end


def _read_string_array(value: np.ndarray) -> list[str]:
    array = np.asarray(value)
    if array.ndim == 0:
        return [str(array.item())]
    return [str(item) for item in array.reshape(-1).tolist()]


def _load_npz_context(
    npz_path: Path,
) -> tuple[set[str], np.ndarray | None, list[str] | None, list[str] | None, np.ndarray | None]:
    with np.load(npz_path, allow_pickle=True) as data:
        files = set(data.files)
        default_joint_pos = None
        if "default_joint_pos" in files:
            default_joint_pos = np.asarray(data["default_joint_pos"], dtype=np.float32).reshape(-1)
        joint_names = _read_string_array(data["joint_names"]) if "joint_names" in files else None
        body_names = _read_string_array(data["body_names"]) if "body_names" in files else None
        valid_lengths = None
        if "valid_lengths" in files:
            valid_lengths = np.asarray(data["valid_lengths"], dtype=np.int32).reshape(-1)
    return files, default_joint_pos, joint_names, body_names, valid_lengths


def _load_series(
    npz_path: Path,
    key: str,
    step_start: int,
    step_end: int | None,
) -> tuple[np.ndarray, np.ndarray | None]:
    with np.load(npz_path, allow_pickle=True) as data:
        if key not in data.files:
            raise KeyError(f"Missing {key!r} in {npz_path}")
        series = np.asarray(data[key], dtype=np.float32)
        valid_lengths = None
        if "valid_lengths" in data.files:
            valid_lengths = np.asarray(data["valid_lengths"], dtype=np.int32).reshape(-1)

    if series.ndim != 3:
        raise ValueError(f"Expected {key} [num_traj, T, D] in {npz_path}, got {series.shape}.")
    _, total_steps, _ = series.shape
    end = total_steps if step_end is None else min(step_end, total_steps)
    if step_start >= end:
        raise ValueError(f"Invalid step slice [{step_start}:{end}] for {npz_path}.")
    sliced = series[:, step_start:end, :]
    if valid_lengths is not None:
        valid_lengths = np.clip(valid_lengths - step_start, 0, end - step_start)
    return sliced, valid_lengths


def _load_body_pos(
    npz_path: Path,
    step_start: int,
    step_end: int | None,
) -> tuple[np.ndarray, list[str] | None, np.ndarray | None]:
    with np.load(npz_path, allow_pickle=True) as data:
        if "body_pos_w" not in data.files:
            raise KeyError(f"Missing body_pos_w in {npz_path}")
        series = np.asarray(data["body_pos_w"], dtype=np.float32)
        body_names = _read_string_array(data["body_names"]) if "body_names" in data.files else None
        valid_lengths = None
        if "valid_lengths" in data.files:
            valid_lengths = np.asarray(data["valid_lengths"], dtype=np.int32).reshape(-1)

    if series.ndim != 4:
        raise ValueError(f"Expected body_pos_w [num_traj, T, B, 3] in {npz_path}, got {series.shape}.")
    _, total_steps, _, _ = series.shape
    end = total_steps if step_end is None else min(step_end, total_steps)
    if step_start >= end:
        raise ValueError(f"Invalid step slice [{step_start}:{end}] for {npz_path}.")
    sliced = series[:, step_start:end, :, :]
    if valid_lengths is not None:
        valid_lengths = np.clip(valid_lengths - step_start, 0, end - step_start)
    return sliced, body_names, valid_lengths


def _align_trajectory_batches(
    source: np.ndarray,
    target: np.ndarray,
    source_valid: np.ndarray | None,
    target_valid: np.ndarray | None,
) -> tuple[np.ndarray, np.ndarray, np.ndarray | None]:
    if source.shape[0] != target.shape[0]:
        count = min(source.shape[0], target.shape[0])
        print(f"[WARN] Trajectory count mismatch ({source.shape[0]} vs {target.shape[0]}); using first {count}.")
        source = source[:count]
        target = target[:count]
        if source_valid is not None:
            source_valid = source_valid[:count]
        if target_valid is not None:
            target_valid = target_valid[:count]

    min_steps = min(source.shape[1], target.shape[1])
    source = source[:, :min_steps, ...]
    target = target[:, :min_steps, ...]
    valid_lengths = source_valid
    if valid_lengths is None:
        valid_lengths = target_valid
    elif target_valid is not None:
        valid_lengths = np.minimum(valid_lengths, target_valid)
    return source, target, valid_lengths


def _align_joint_series_by_name(
    source: np.ndarray,
    target: np.ndarray,
    source_names: list[str] | None,
    target_names: list[str] | None,
) -> tuple[np.ndarray, np.ndarray, list[str]]:
    """Reorder joint dimension to a shared joint-name list."""

    if source.ndim != 3 or target.ndim != 3:
        raise ValueError("Expected joint arrays with shape [num_traj, T, J].")

    if source_names is None and target_names is None:
        count = min(source.shape[2], target.shape[2])
        if source.shape[2] != target.shape[2]:
            print(
                f"[WARN] Joint count mismatch without joint_names metadata ({source.shape[2]} vs {target.shape[2]}); "
                f"using first {count} by index."
            )
        return source[:, :, :count], target[:, :, :count], [f"joint_{idx}" for idx in range(count)]

    reference_names = source_names or target_names
    assert reference_names is not None

    def select_by_names(array: np.ndarray, names: list[str] | None) -> np.ndarray:
        if names is None:
            if array.shape[2] < len(reference_names):
                raise ValueError(
                    f"NPZ has {array.shape[2]} joints but comparison expects {len(reference_names)} named joints."
                )
            return array[:, :, : len(reference_names)]
        name_to_index = {name: idx for idx, name in enumerate(names)}
        missing = [name for name in reference_names if name not in name_to_index]
        if missing:
            raise KeyError(f"Joint names missing from NPZ: {missing}. Available: {names}.")
        indices = [name_to_index[name] for name in reference_names]
        return array[:, :, indices]

    aligned_source = select_by_names(source, source_names)
    aligned_target = select_by_names(target, target_names)
    return aligned_source, aligned_target, list(reference_names)


def _align_default_joint_pos(
    default_joint_pos: np.ndarray | None,
    default_names: list[str] | None,
    joint_names: list[str],
) -> np.ndarray:
    """Select default pose entries to match an aligned joint-name list."""

    if default_joint_pos is None:
        raise ValueError("default_joint_pos is required for relative joint plotting.")
    default_joint_pos = np.asarray(default_joint_pos, dtype=np.float32).reshape(-1)
    if default_names is None:
        if default_joint_pos.shape[0] != len(joint_names):
            raise ValueError(
                f"default_joint_pos length {default_joint_pos.shape[0]} does not match joint count {len(joint_names)}."
            )
        return default_joint_pos

    name_to_index = {name: idx for idx, name in enumerate(default_names)}
    missing = [name for name in joint_names if name not in name_to_index]
    if missing:
        raise KeyError(f"default_joint_pos names missing joints: {missing}. Available: {default_names}.")
    return np.asarray([default_joint_pos[name_to_index[name]] for name in joint_names], dtype=np.float32)


def _align_body_series_by_name(
    source: np.ndarray,
    target: np.ndarray,
    source_names: list[str] | None,
    target_names: list[str] | None,
) -> tuple[np.ndarray, np.ndarray, list[str]]:
    """Reorder body dimension to a shared body-name list."""

    if source.ndim != 4 or target.ndim != 4:
        raise ValueError("Expected body arrays with shape [num_traj, T, B, 3].")

    if source_names is None and target_names is None:
        count = min(source.shape[2], target.shape[2])
        if source.shape[2] != target.shape[2]:
            print(
                f"[WARN] Body count mismatch without body_names metadata ({source.shape[2]} vs {target.shape[2]}); "
                f"using first {count} by index."
            )
        return source[:, :, :count, :], target[:, :, :count, :], [f"body_{idx}" for idx in range(count)]

    reference_names = source_names or target_names
    assert reference_names is not None

    def select_by_names(array: np.ndarray, names: list[str] | None) -> np.ndarray:
        if names is None:
            if array.shape[2] < len(reference_names):
                raise ValueError(
                    f"NPZ has {array.shape[2]} bodies but comparison expects {len(reference_names)} named bodies."
                )
            return array[:, :, : len(reference_names), :]
        name_to_index = {name: idx for idx, name in enumerate(names)}
        missing = [name for name in reference_names if name not in name_to_index]
        if missing:
            raise KeyError(f"Body names missing from NPZ: {missing}. Available: {names}.")
        indices = [name_to_index[name] for name in reference_names]
        return array[:, :, indices, :]

    aligned_source = select_by_names(source, source_names)
    aligned_target = select_by_names(target, target_names)
    return aligned_source, aligned_target, list(reference_names)


def _flatten_body_pos(body_pos: np.ndarray, body_names: list[str]) -> tuple[np.ndarray, list[str]]:
    """Flatten [num_traj, T, B, 3] into [num_traj, T, B*3] with dim labels."""

    num_traj, num_steps, num_bodies, _ = body_pos.shape
    flat = body_pos.reshape(num_traj, num_steps, num_bodies * 3)
    labels = [f"{body}/{axis}" for body in body_names for axis in AXIS_LABELS]
    return flat, labels


def _masked_stats(data: np.ndarray, valid_lengths: np.ndarray | None) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    num_traj, num_steps, num_dims = data.shape
    mean = np.zeros((num_steps, num_dims), dtype=np.float32)
    vmin = np.zeros((num_steps, num_dims), dtype=np.float32)
    vmax = np.zeros((num_steps, num_dims), dtype=np.float32)

    for step_idx in range(num_steps):
        active = []
        for traj_idx in range(num_traj):
            if valid_lengths is not None and step_idx >= valid_lengths[traj_idx]:
                continue
            active.append(data[traj_idx, step_idx])
        if not active:
            continue
        stack = np.stack(active, axis=0)
        mean[step_idx] = stack.mean(axis=0)
        vmin[step_idx] = stack.min(axis=0)
        vmax[step_idx] = stack.max(axis=0)
    return mean, vmin, vmax


def plot_transfer_overlay(
    source: np.ndarray,
    target: np.ndarray,
    *,
    output_path: Path,
    step_start: int,
    source_label: str,
    target_label: str,
    valid_lengths: np.ndarray | None,
    ylabel: str,
    title: str,
    dim_labels: list[str] | None,
    ncols: int,
    dpi: int,
    transform=None,
) -> Path:
    if transform is not None:
        source = transform(source)
        target = transform(target)

    source_mean, source_min, source_max = _masked_stats(source, valid_lengths)
    target_mean, target_min, target_max = _masked_stats(target, valid_lengths)

    num_steps, num_dims = source_mean.shape
    steps = np.arange(step_start, step_start + num_steps)
    nrows = int(np.ceil(num_dims / ncols))
    fig, axes = plt.subplots(nrows, ncols, figsize=(2.8 * ncols, 2.5 * nrows), dpi=dpi, sharex=True)
    fig.patch.set_facecolor("white")
    axes_flat = np.atleast_1d(axes).flatten()

    for dim in range(num_dims):
        ax = axes_flat[dim]
        ax.fill_between(steps, source_min[:, dim], source_max[:, dim], color=SOURCE_COLOR, alpha=BAND_ALPHA, linewidth=0)
        ax.plot(steps, source_mean[:, dim], color=SOURCE_COLOR, linewidth=MEAN_LW)
        ax.fill_between(steps, target_min[:, dim], target_max[:, dim], color=TARGET_COLOR, alpha=BAND_ALPHA, linewidth=0)
        ax.plot(steps, target_mean[:, dim], color=TARGET_COLOR, linewidth=MEAN_LW)
        if dim_labels is not None and dim < len(dim_labels):
            ax.set_title(dim_labels[dim], fontsize=8, pad=2)
        else:
            ax.set_title(f"dim {dim}", fontsize=9, pad=2)
        ax.grid(True, color="#e8e8e8", linewidth=0.6)
        ax.tick_params(labelsize=7)

    for idx in range(num_dims, len(axes_flat)):
        axes_flat[idx].axis("off")

    fig.supxlabel("Step", fontsize=11)
    fig.supylabel(ylabel, fontsize=11)
    fig.suptitle(title, fontsize=13, y=0.995)
    legend_handles = [
        Line2D([0], [0], color=SOURCE_COLOR, linewidth=MEAN_LW, label=f"{source_label} mean"),
        Patch(facecolor=SOURCE_COLOR, edgecolor="none", alpha=BAND_ALPHA, label=f"{source_label} min–max"),
        Line2D([0], [0], color=TARGET_COLOR, linewidth=MEAN_LW, label=f"{target_label} mean"),
        Patch(facecolor=TARGET_COLOR, edgecolor="none", alpha=BAND_ALPHA, label=f"{target_label} min–max"),
    ]
    fig.legend(handles=legend_handles, loc="upper center", bbox_to_anchor=(0.5, 0.965), ncol=4, framealpha=0.95, fontsize=8)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    fig.tight_layout(rect=[0, 0, 1, 0.93])
    fig.savefig(output_path, bbox_inches="tight", facecolor="white")
    plt.close(fig)
    return output_path


def build_arg_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="Compare Isaac source vs Genesis base-only replay trajectories.")
    parser.add_argument("source_npz", type=Path, help="Isaac source state-action NPZ.")
    parser.add_argument("target_npz", type=Path, help="Genesis replay NPZ from replay_openloop_genesis.py.")
    parser.add_argument(
        "--compare",
        choices=("joints", "bodies", "both"),
        default="both",
        help="Which state trajectories to plot. Default: both.",
    )
    parser.add_argument("--joint-output", type=Path, default=None, help="Joint comparison PNG path.")
    parser.add_argument("--body-output", type=Path, default=None, help="Body comparison PNG path.")
    parser.add_argument("--step-range", type=_parse_step_range, default=(0, None), metavar="START:END")
    parser.add_argument("--default-joint-pos-onnx", type=Path, default=None, help="Fallback default_joint_pos ONNX.")
    parser.add_argument("--source-label", type=str, default="Isaac source")
    parser.add_argument("--target-label", type=str, default="Genesis replay")
    parser.add_argument("--ncols", type=int, default=6)
    parser.add_argument("--dpi", type=int, default=150)
    return parser


def main(argv: list[str] | None = None) -> int:
    args = build_arg_parser().parse_args(argv)
    source_path = args.source_npz.expanduser().resolve()
    target_path = args.target_npz.expanduser().resolve()
    if not source_path.is_file():
        print(f"[ERROR] Source NPZ not found: {source_path}", file=sys.stderr)
        return 1
    if not target_path.is_file():
        print(f"[ERROR] Target NPZ not found: {target_path}", file=sys.stderr)
        return 1

    step_start, step_end = args.step_range
    saved_paths: list[Path] = []

    if args.compare in ("joints", "both"):
        source, source_valid = _load_series(source_path, "joint_pos", step_start, step_end)
        target, target_valid = _load_series(target_path, "joint_pos", step_start, step_end)
        source, target, valid_lengths = _align_trajectory_batches(source, target, source_valid, target_valid)

        _, source_default, source_joint_names, _, _ = _load_npz_context(source_path)
        _, target_default, target_joint_names, _, _ = _load_npz_context(target_path)
        source, target, joint_names = _align_joint_series_by_name(
            source, target, source_joint_names, target_joint_names
        )
        if source_default is not None:
            default_joint_pos = _align_default_joint_pos(source_default, source_joint_names, joint_names)
        elif target_default is not None:
            default_joint_pos = _align_default_joint_pos(target_default, target_joint_names, joint_names)
        else:
            default_joint_pos, _ = resolve_default_joint_pos(source_path, args.default_joint_pos_onnx)
            default_joint_pos = _align_default_joint_pos(default_joint_pos, None, joint_names)

        joint_output = args.joint_output or source_path.with_name(
            f"{source_path.stem}_vs_{target_path.stem}_transfer_joint_pos_rel.png"
        )
        saved = plot_transfer_overlay(
            source,
            target,
            output_path=joint_output.expanduser().resolve(),
            step_start=step_start,
            source_label=args.source_label,
            target_label=args.target_label,
            valid_lengths=valid_lengths,
            ylabel=JOINT_POS_REL_YLABEL,
            title=f"{JOINT_POS_REL_TITLE}\nIsaac source (base+δ) vs Genesis replay (base only)",
            dim_labels=joint_names,
            ncols=args.ncols,
            dpi=args.dpi,
            transform=lambda data: to_relative_joint_pos(data, default_joint_pos),
        )
        saved_paths.append(saved)
        print(f"[INFO] Saved joint transfer comparison ({len(joint_names)} joints): {saved}")

    if args.compare in ("bodies", "both"):
        source_body, source_body_names, source_valid = _load_body_pos(source_path, step_start, step_end)
        target_body, target_body_names, target_valid = _load_body_pos(target_path, step_start, step_end)
        source_body, target_body, valid_lengths = _align_trajectory_batches(
            source_body, target_body, source_valid, target_valid
        )
        source_body, target_body, body_names = _align_body_series_by_name(
            source_body, target_body, source_body_names, target_body_names
        )
        source_flat, body_dim_labels = _flatten_body_pos(source_body, body_names)
        target_flat, _ = _flatten_body_pos(target_body, body_names)

        body_output = args.body_output or source_path.with_name(
            f"{source_path.stem}_vs_{target_path.stem}_transfer_body_pos_w.png"
        )
        saved = plot_transfer_overlay(
            source_flat,
            target_flat,
            output_path=body_output.expanduser().resolve(),
            step_start=step_start,
            source_label=args.source_label,
            target_label=args.target_label,
            valid_lengths=valid_lengths,
            ylabel="Body position in world frame (m)",
            title="Body world positions\nIsaac source (base+δ) vs Genesis replay (base only)",
            dim_labels=body_dim_labels,
            ncols=args.ncols,
            dpi=args.dpi,
        )
        saved_paths.append(saved)
        print(f"[INFO] Saved body transfer comparison ({len(body_names)} bodies): {saved}")

    if not saved_paths:
        print("[ERROR] No comparisons were generated.", file=sys.stderr)
        return 1
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
