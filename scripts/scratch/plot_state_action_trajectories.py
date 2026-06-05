"""Plot aggregate mean / min-max bands from play.py state-action rollout NPZ files.

Expects NPZ output from ``play.py --record_state_action_trajectories`` with keys
such as ``joint_pos`` and ``actions`` shaped ``[num_traj, T, num_dofs]``.

``joint_pos`` is stored as absolute sim coordinates; this script converts it to
policy-relative space (``joint_pos - default_joint_pos``) before plotting so it
matches ``obs_joint_pos`` in delta-finetune rollout plots.
"""

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
    JOINT_POS_REL_TITLE,
    JOINT_POS_REL_YLABEL,
    resolve_default_joint_pos,
    to_relative_joint_pos,
)
from matplotlib.lines import Line2D
from matplotlib.patches import Patch

SERIES_COLOR = "#2563eb"
BAND_ALPHA = 0.30
MEAN_LW = 1.8

DEFAULT_KEYS = ("joint_pos", "actions")


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


def _load_series(
    npz_path: Path,
    key: str,
    step_start: int,
    step_end: int | None,
) -> tuple[np.ndarray, float | None, np.ndarray]:
    with np.load(npz_path) as data:
        if key not in data.files:
            raise KeyError(f"Key {key!r} not found in {npz_path}. Available keys: {sorted(data.files)}")
        series = np.asarray(data[key], dtype=np.float32)
        fps = float(np.asarray(data["fps"]).reshape(-1)[0]) if "fps" in data.files else None

    if series.ndim != 3:
        raise ValueError(f"Expected 3D array [num_traj, T, D] for key {key!r}; got {series.shape}.")
    num_traj, total_steps, _ = series.shape
    end = total_steps if step_end is None else min(step_end, total_steps)
    if step_start >= end:
        raise ValueError(f"Invalid step slice [{step_start}:{end}] for trajectory length {total_steps}.")
    valid_lengths = _infer_valid_lengths(series)
    valid_lengths = np.clip(valid_lengths - step_start, 0, end - step_start)
    return series[:, step_start:end, :], fps, valid_lengths


def _infer_valid_lengths(data: np.ndarray, *, atol: float = 1.0e-8, rtol: float = 1.0e-8) -> np.ndarray:
    """Infer per-trajectory valid lengths by dropping trailing constant-value padding.

    Recorder padding repeats the final valid frame to the rollout max length. We keep a
    trajectory active until the last timestep whose value differs from its predecessor.
    """

    if data.ndim != 3:
        raise ValueError(f"Expected 3D array [num_traj, T, D], got {data.shape}.")
    num_traj, total_steps, _ = data.shape
    lengths = np.ones(num_traj, dtype=np.int32)
    for traj_idx in range(num_traj):
        traj = data[traj_idx]
        last_change = 0
        for step_idx in range(1, total_steps):
            if not np.allclose(traj[step_idx], traj[step_idx - 1], atol=atol, rtol=rtol):
                last_change = step_idx
        lengths[traj_idx] = last_change + 1
    return lengths


def _masked_stats(data: np.ndarray, valid_lengths: np.ndarray) -> tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
    if data.ndim != 3:
        raise ValueError(f"Expected 3D array [num_traj, T, D], got {data.shape}.")
    if valid_lengths.ndim != 1 or valid_lengths.shape[0] != data.shape[0]:
        raise ValueError(
            f"Expected valid_lengths shape ({data.shape[0]},), got {tuple(valid_lengths.shape)}."
        )

    num_traj, num_steps, _ = data.shape
    step_ids = np.arange(num_steps, dtype=np.int32)
    valid_mask = step_ids[None, :] < valid_lengths[:, None]
    masked = np.where(valid_mask[:, :, None], data, np.nan)
    mean = np.nanmean(masked, axis=0)
    min_vals = np.nanmin(masked, axis=0)
    max_vals = np.nanmax(masked, axis=0)
    counts = valid_mask.sum(axis=0).astype(np.int32)
    return mean, min_vals, max_vals, counts


def plot_aggregate_series(
    data: np.ndarray,
    *,
    valid_lengths: np.ndarray,
    output_path: Path,
    step_start: int,
    ylabel: str,
    title_prefix: str,
    ncols: int = 6,
    dpi: int = 150,
    dim_labels: list[str] | None = None,
    share_y: bool = False,
) -> Path:
    num_traj, num_steps, num_dims = data.shape
    steps = np.arange(step_start, step_start + num_steps)
    nrows = int(np.ceil(num_dims / ncols))
    mean, min_vals, max_vals, valid_counts = _masked_stats(data, valid_lengths)

    fig, axes = plt.subplots(nrows, ncols, figsize=(2.8 * ncols, 2.4 * nrows), dpi=dpi, sharex=True)
    fig.patch.set_facecolor("white")
    axes_flat = np.atleast_1d(axes).flatten()

    ylim = None
    if share_y:
        finite = np.concatenate(
            [min_vals[np.isfinite(min_vals)].reshape(-1), max_vals[np.isfinite(max_vals)].reshape(-1)]
        )
        if finite.size > 0:
            pad = 0.05 * max(float(finite.max() - finite.min()), 1e-6)
            ylim = (float(finite.min()) - pad, float(finite.max()) + pad)

    for dim in range(num_dims):
        ax = axes_flat[dim]
        dim_mean = mean[:, dim]
        dim_min = min_vals[:, dim]
        dim_max = max_vals[:, dim]
        ax.fill_between(steps, dim_min, dim_max, color=SERIES_COLOR, alpha=BAND_ALPHA, linewidth=0)
        ax.plot(steps, dim_mean, color=SERIES_COLOR, linewidth=MEAN_LW)
        if dim_labels is not None and dim < len(dim_labels):
            ax.set_title(dim_labels[dim], fontsize=8, pad=2)
        else:
            ax.set_title(f"dof {dim}", fontsize=9, pad=2)
        ax.grid(True, color="#e8e8e8", linewidth=0.6)
        ax.tick_params(labelsize=7)
        if ylim is not None:
            ax.set_ylim(ylim)

    for idx in range(num_dims, len(axes_flat)):
        axes_flat[idx].axis("off")

    step_end = step_start + num_steps
    fig.supxlabel("Step", fontsize=11)
    fig.supylabel(ylabel, fontsize=11)
    valid_min = int(valid_counts.min()) if valid_counts.size > 0 else 0
    valid_max = int(valid_counts.max()) if valid_counts.size > 0 else 0
    fig.suptitle(
        f"{title_prefix} — all DOFs (steps {step_start}–{step_end - 1})\n"
        f"{num_traj} trajectories, valid per step: {valid_min}–{valid_max}",
        fontsize=13,
        y=0.995,
    )
    legend_handles = [
        Line2D([0], [0], color=SERIES_COLOR, linewidth=MEAN_LW, label="Mean"),
        Patch(facecolor=SERIES_COLOR, edgecolor="none", alpha=BAND_ALPHA, label="Min–max"),
    ]
    fig.legend(handles=legend_handles, loc="upper center", bbox_to_anchor=(0.5, 0.965), ncol=2, framealpha=0.95, fontsize=9)

    output_path.parent.mkdir(parents=True, exist_ok=True)
    fig.tight_layout(rect=[0, 0, 1, 0.94])
    fig.savefig(output_path, bbox_inches="tight", facecolor="white")
    plt.close(fig)
    return output_path


def build_arg_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description="Plot mean/min-max aggregate bands for state-action rollout NPZ keys."
    )
    parser.add_argument("npz_path", type=Path, help="State-action rollout NPZ from play.py.")
    parser.add_argument(
        "--output-dir",
        type=Path,
        default=None,
        help="Directory for PNG outputs. Default: same directory as the NPZ.",
    )
    parser.add_argument(
        "--step-range",
        type=_parse_step_range,
        default=(0, None),
        metavar="START:END",
        help="Step slice to plot (end exclusive). Default: 0: (all steps).",
    )
    parser.add_argument(
        "--keys",
        nargs="+",
        default=list(DEFAULT_KEYS),
        help=f"NPZ keys to plot. Default: {' '.join(DEFAULT_KEYS)}.",
    )
    parser.add_argument("--ncols", type=int, default=6, help="Subplot columns. Default: 6.")
    parser.add_argument("--dpi", type=int, default=150, help="Figure DPI. Default: 150.")
    parser.add_argument("--share-y", action="store_true", help="Share y-axis across subplots.")
    parser.add_argument(
        "--default-joint-pos-onnx",
        type=Path,
        default=None,
        help=(
            "ONNX policy with metadata `default_joint_pos` for converting logged absolute "
            "`joint_pos` to relative coordinates. Default: infer `<run_dir>/exported/model_<step>.onnx`."
        ),
    )
    parser.add_argument(
        "--joint-pos-space",
        choices=("relative", "absolute"),
        default="relative",
        help="Plot joint_pos in policy-relative or raw sim coordinates. Default: relative.",
    )
    return parser


def _default_output_path(npz_path: Path, key: str, *, joint_pos_space: str = "relative") -> Path:
    if key == "joint_pos" and joint_pos_space == "relative":
        return npz_path.with_name(f"{npz_path.stem}_joint_pos_rel.png")
    return npz_path.with_name(f"{npz_path.stem}_{key}.png")


def _ylabel_for_key(key: str, *, joint_pos_space: str) -> str:
    if key == "joint_pos":
        return JOINT_POS_REL_YLABEL if joint_pos_space == "relative" else "Joint position (rad)"
    if key == "actions":
        return "Action"
    return key


def _title_for_key(key: str, *, joint_pos_space: str) -> str:
    if key == "joint_pos":
        return JOINT_POS_REL_TITLE if joint_pos_space == "relative" else "Joint positions (absolute)"
    if key == "actions":
        return "Policy actions"
    return key.replace("_", " ").title()


def main(argv: list[str] | None = None) -> int:
    args = build_arg_parser().parse_args(argv)
    npz_path = args.npz_path.expanduser().resolve()
    if not npz_path.is_file():
        print(f"[ERROR] NPZ file not found: {npz_path}", file=sys.stderr)
        return 1

    output_dir = (args.output_dir or npz_path.parent).expanduser().resolve()
    step_start, step_end = args.step_range

    output_dir = (args.output_dir or npz_path.parent).expanduser().resolve()
    step_start, step_end = args.step_range
    default_joint_pos = None
    default_joint_pos_source = None

    saved_paths: list[Path] = []
    for key in args.keys:
        try:
            data, fps, valid_lengths = _load_series(npz_path, key, step_start, step_end)
        except (KeyError, ValueError) as exc:
            print(f"[ERROR] {exc}", file=sys.stderr)
            return 1

        if key == "joint_pos" and args.joint_pos_space == "relative":
            try:
                default_joint_pos, default_joint_pos_source = resolve_default_joint_pos(
                    npz_path=npz_path,
                    onnx_path=args.default_joint_pos_onnx,
                )
            except (FileNotFoundError, KeyError, ValueError) as exc:
                print(f"[ERROR] {exc}", file=sys.stderr)
                return 1
            data = to_relative_joint_pos(data, default_joint_pos)

        output_path = output_dir / _default_output_path(npz_path, key, joint_pos_space=args.joint_pos_space).name
        saved = plot_aggregate_series(
            data,
            valid_lengths=valid_lengths,
            output_path=output_path,
            step_start=step_start,
            ylabel=_ylabel_for_key(key, joint_pos_space=args.joint_pos_space),
            title_prefix=_title_for_key(key, joint_pos_space=args.joint_pos_space),
            ncols=args.ncols,
            dpi=args.dpi,
            share_y=args.share_y,
        )
        saved_paths.append(saved)
        num_traj, num_steps, num_dims = data.shape
        fps_text = f", fps={fps:.3f}" if fps is not None else ""
        valid_min = int(valid_lengths.min()) if valid_lengths.size > 0 else 0
        valid_max = int(valid_lengths.max()) if valid_lengths.size > 0 else 0
        space_text = ""
        if key == "joint_pos":
            space_text = f", space={args.joint_pos_space}"
            if args.joint_pos_space == "relative" and default_joint_pos_source is not None:
                space_text += f", default_from={default_joint_pos_source.name}"
        print(
            f"[INFO] {key}: trajectories={num_traj}, steps={num_steps}, dims={num_dims}, "
            f"valid_lengths={valid_min}-{valid_max}{fps_text}{space_text} "
            f"-> {saved}"
        )

    return 0


if __name__ == "__main__":
    raise SystemExit(main())
