"""Overlay rollout state aggregate bands with a single reference motion series.

Compares a multi-trajectory state-action / motion dataset NPZ (e.g. Genesis sim2sim
recordings) against a reference motion NPZ extracted from an exported policy ONNX.
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
from matplotlib.lines import Line2D
from matplotlib.patches import Patch
from plot_state_action_trajectories import _infer_valid_lengths, _load_series, _masked_stats

ROLLOUT_COLOR = "#2563eb"
REFERENCE_COLOR = "#16a34a"
BAND_ALPHA = 0.30
MEAN_LW = 1.8
REFERENCE_LW = 1.6

DEFAULT_STATE_KEYS = ("joint_pos", "joint_vel")


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


def _load_reference_series(
    npz_path: Path,
    key: str,
    step_start: int,
    step_end: int | None,
) -> tuple[np.ndarray, float | None]:
    with np.load(npz_path) as data:
        if key not in data.files:
            raise KeyError(f"Key {key!r} not found in {npz_path}. Available keys: {sorted(data.files)}")
        series = np.asarray(data[key], dtype=np.float32)
        fps = float(np.asarray(data["fps"]).reshape(-1)[0]) if "fps" in data.files else None

    if series.ndim != 2:
        raise ValueError(f"Expected 2D array [T, D] for reference key {key!r}; got {series.shape}.")
    total_steps, _ = series.shape
    end = total_steps if step_end is None else min(step_end, total_steps)
    if step_start >= end:
        raise ValueError(f"Invalid step slice [{step_start}:{end}] for reference length {total_steps}.")
    return series[step_start:end, :], fps


def _ylabel_for_key(key: str) -> str:
    if key == "joint_pos":
        return "Joint position (rad)"
    if key == "joint_vel":
        return "Joint velocity (rad/s)"
    return key.replace("_", " ")


def _title_for_key(key: str) -> str:
    if key == "joint_pos":
        return "Joint positions"
    if key == "joint_vel":
        return "Joint velocities"
    return key.replace("_", " ").title()


def plot_rollout_vs_reference_aggregate(
    rollout_data: np.ndarray,
    reference_data: np.ndarray,
    *,
    rollout_valid_lengths: np.ndarray,
    output_path: Path,
    step_start: int,
    ylabel: str,
    title_prefix: str,
    rollout_label: str = "Rollout",
    reference_label: str = "Reference",
    ncols: int = 6,
    dpi: int = 150,
    share_y: bool = False,
) -> Path:
    if rollout_data.ndim != 3:
        raise ValueError(f"Expected rollout array [num_traj, T, D], got {rollout_data.shape}.")
    if reference_data.ndim != 2:
        raise ValueError(f"Expected reference array [T, D], got {reference_data.shape}.")
    if rollout_data.shape[1:] != reference_data.shape:
        raise ValueError(
            f"Shape mismatch: rollout {rollout_data.shape[1:]} vs reference {reference_data.shape}."
        )

    num_traj, num_steps, num_dims = rollout_data.shape
    steps = np.arange(step_start, step_start + num_steps)
    nrows = int(np.ceil(num_dims / ncols))
    rollout_mean, rollout_min, rollout_max, rollout_counts = _masked_stats(rollout_data, rollout_valid_lengths)

    fig, axes = plt.subplots(nrows, ncols, figsize=(2.8 * ncols, 2.4 * nrows), dpi=dpi, sharex=True)
    fig.patch.set_facecolor("white")
    axes_flat = np.atleast_1d(axes).flatten()

    ylim = None
    if share_y:
        finite = np.concatenate(
            [
                rollout_min[np.isfinite(rollout_min)].reshape(-1),
                rollout_max[np.isfinite(rollout_max)].reshape(-1),
                reference_data[np.isfinite(reference_data)].reshape(-1),
            ]
        )
        if finite.size > 0:
            pad = 0.05 * max(float(finite.max() - finite.min()), 1e-6)
            ylim = (float(finite.min()) - pad, float(finite.max()) + pad)

    for dim in range(num_dims):
        ax = axes_flat[dim]
        ax.fill_between(
            steps,
            rollout_min[:, dim],
            rollout_max[:, dim],
            color=ROLLOUT_COLOR,
            alpha=BAND_ALPHA,
            linewidth=0,
        )
        ax.plot(steps, rollout_mean[:, dim], color=ROLLOUT_COLOR, linewidth=MEAN_LW)
        ax.plot(steps, reference_data[:, dim], color=REFERENCE_COLOR, linewidth=REFERENCE_LW)
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
    rollout_valid_min = int(rollout_counts.min()) if rollout_counts.size > 0 else 0
    rollout_valid_max = int(rollout_counts.max()) if rollout_counts.size > 0 else 0
    fig.suptitle(
        f"{title_prefix} — rollout vs reference (steps {step_start}–{step_end - 1})\n"
        f"{num_traj} trajectories, rollout valid per step: {rollout_valid_min}–{rollout_valid_max}",
        fontsize=13,
        y=0.995,
    )

    legend_handles = [
        Line2D([0], [0], color=ROLLOUT_COLOR, linewidth=MEAN_LW, label=f"{rollout_label} mean"),
        Patch(facecolor=ROLLOUT_COLOR, edgecolor="none", alpha=BAND_ALPHA, label=f"{rollout_label} min–max"),
        Line2D([0], [0], color=REFERENCE_COLOR, linewidth=REFERENCE_LW, label=reference_label),
    ]
    fig.legend(handles=legend_handles, loc="upper center", bbox_to_anchor=(0.5, 0.965), ncol=3, framealpha=0.95, fontsize=9)

    output_path.parent.mkdir(parents=True, exist_ok=True)
    fig.tight_layout(rect=[0, 0, 1, 0.94])
    fig.savefig(output_path, bbox_inches="tight", facecolor="white")
    plt.close(fig)
    return output_path


def build_arg_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description="Plot rollout state aggregate bands overlaid with a reference motion NPZ."
    )
    parser.add_argument("rollout_npz", type=Path, help="Multi-trajectory rollout / motion dataset NPZ.")
    parser.add_argument("reference_npz", type=Path, help="Single reference motion NPZ (e.g. from ONNX export).")
    parser.add_argument(
        "--output-dir",
        type=Path,
        default=None,
        help="Directory for PNG outputs. Default: same directory as rollout_npz.",
    )
    parser.add_argument(
        "--output-stem",
        type=str,
        default=None,
        help="Filename stem for outputs (without extension). Default: <rollout_stem>_vs_<reference_stem>.",
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
        default=list(DEFAULT_STATE_KEYS),
        help=f"State keys to plot. Default: {' '.join(DEFAULT_STATE_KEYS)}.",
    )
    parser.add_argument("--ncols", type=int, default=6, help="Subplot columns. Default: 6.")
    parser.add_argument("--dpi", type=int, default=150, help="Figure DPI. Default: 150.")
    parser.add_argument("--share-y", action="store_true", help="Share y-axis across subplots.")
    parser.add_argument(
        "--rollout-label",
        type=str,
        default="Rollout",
        help="Legend label for rollout aggregate. Default: Rollout.",
    )
    parser.add_argument(
        "--reference-label",
        type=str,
        default="Reference",
        help="Legend label for reference line. Default: Reference.",
    )
    return parser


def main(argv: list[str] | None = None) -> int:
    args = build_arg_parser().parse_args(argv)
    rollout_npz = args.rollout_npz.expanduser().resolve()
    reference_npz = args.reference_npz.expanduser().resolve()
    for path in (rollout_npz, reference_npz):
        if not path.is_file():
            print(f"[ERROR] NPZ file not found: {path}", file=sys.stderr)
            return 1

    output_dir = (args.output_dir or rollout_npz.parent).expanduser().resolve()
    step_start, step_end = args.step_range
    output_stem = args.output_stem or f"{rollout_npz.stem}_vs_{reference_npz.stem}"

    saved_paths: list[Path] = []
    for key in args.keys:
        try:
            rollout_data, rollout_fps, rollout_valid_lengths = _load_series(
                rollout_npz, key, step_start, step_end
            )
            reference_data, reference_fps = _load_reference_series(
                reference_npz, key, step_start, step_end
            )
        except (KeyError, ValueError) as exc:
            print(f"[ERROR] {key}: {exc}", file=sys.stderr)
            return 1

        output_path = output_dir / f"{output_stem}_{key}.png"
        saved = plot_rollout_vs_reference_aggregate(
            rollout_data,
            reference_data,
            rollout_valid_lengths=rollout_valid_lengths,
            output_path=output_path,
            step_start=step_start,
            ylabel=_ylabel_for_key(key),
            title_prefix=_title_for_key(key),
            rollout_label=args.rollout_label,
            reference_label=args.reference_label,
            ncols=args.ncols,
            dpi=args.dpi,
            share_y=args.share_y,
        )
        saved_paths.append(saved)

        num_traj, num_steps, num_dims = rollout_data.shape
        fps_parts = []
        if rollout_fps is not None:
            fps_parts.append(f"rollout_fps={rollout_fps:.3f}")
        if reference_fps is not None:
            fps_parts.append(f"reference_fps={reference_fps:.3f}")
        fps_text = f", {', '.join(fps_parts)}" if fps_parts else ""
        valid_min = int(rollout_valid_lengths.min()) if rollout_valid_lengths.size > 0 else 0
        valid_max = int(rollout_valid_lengths.max()) if rollout_valid_lengths.size > 0 else 0
        print(
            f"[INFO] {key}: trajectories={num_traj}, steps={num_steps}, dims={num_dims}, "
            f"valid_lengths={valid_min}-{valid_max}{fps_text} -> {saved}"
        )

    print(f"[INFO] rollout_npz={rollout_npz}")
    print(f"[INFO] reference_npz={reference_npz}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
