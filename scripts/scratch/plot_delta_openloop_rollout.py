"""Compare open-loop delta rollout NPZ series with aggregate mean/min-max bands.

Expects NPZ output from ``play.py --record_delta_model_dataset`` on
``Tracking-Flat-G1-DeltaA-OpenLoop-v0``.

Figure 1 (actions): ``actions`` vs ``obs_motion_joint_action`` vs ``motion_joint_action``.
Figure 2 (joints): ``obs_joint_pos`` vs ``motion_joint_pos`` (reference converted to
policy-relative coordinates via ONNX ``default_joint_pos``).

.. code-block:: bash

    python scripts/scratch/plot_delta_openloop_rollout.py \\
        logs/rsl_rl/.../delta_model_datasets/model_23000_2026-05-24_21-06-05.npz
"""

from __future__ import annotations

import argparse
import sys
from dataclasses import dataclass
from pathlib import Path

_SCRATCH_DIR = Path(__file__).resolve().parent
if str(_SCRATCH_DIR) not in sys.path:
    sys.path.insert(0, str(_SCRATCH_DIR))

import matplotlib.pyplot as plt
import numpy as np
from joint_plot_utils import (
    JOINT_POS_REL_YLABEL,
    resolve_default_joint_pos,
    to_relative_joint_pos,
)
from matplotlib.lines import Line2D
from matplotlib.patches import Patch

BAND_ALPHA = 0.28
MEAN_LW = 2.0

DEFAULT_ACTION_KEYS = ("actions", "motion_joint_action")
DEFAULT_JOINT_KEYS = ("obs_joint_pos", "motion_joint_pos")

ACTION_SERIES = {
    "actions": ("#2563eb", "Delta Action (policy output)"),
    # "obs_motion_joint_action": ("#ea580c", "obs_motion_joint_action"),
    "motion_joint_action": ("#16a34a", "Base Action"),
}

JOINT_SERIES = {
    "obs_joint_pos": ("#7c3aed", "Observed Joint Pos (robot relative)"),
    "motion_joint_pos": ("#0d9488", "Target Joint Pos (reference relative)"),
}


@dataclass(frozen=True)
class SeriesSpec:
    key: str
    label: str
    color: str
    data: np.ndarray
    valid_lengths: np.ndarray


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


def _load_3d_array(
    npz_path: Path,
    key: str,
    step_start: int,
    step_end: int | None,
) -> tuple[np.ndarray, int, np.ndarray]:
    with np.load(npz_path) as data:
        if key not in data.files:
            raise KeyError(f"Key {key!r} not found in {npz_path}. Available keys: {sorted(data.files)}")
        array = np.asarray(data[key], dtype=np.float32)

    if array.ndim != 3:
        raise ValueError(f"Expected 3D array [num_traj, T, D] for key {key!r}; got {array.shape}.")
    num_traj, total_steps, _ = array.shape
    end = total_steps if step_end is None else min(step_end, total_steps)
    if step_start >= end:
        raise ValueError(f"Invalid step slice [{step_start}:{end}] for trajectory length {total_steps}.")
    valid_lengths = _infer_valid_lengths(array)
    valid_lengths = np.clip(valid_lengths - step_start, 0, end - step_start)
    return array[:, step_start:end, :], num_traj, valid_lengths


def _infer_valid_lengths(data: np.ndarray, *, atol: float = 1.0e-8, rtol: float = 1.0e-8) -> np.ndarray:
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


def _masked_stats(data: np.ndarray, valid_lengths: np.ndarray) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    if data.ndim != 3:
        raise ValueError(f"Expected 3D array [num_traj, T, D], got {data.shape}.")
    if valid_lengths.ndim != 1 or valid_lengths.shape[0] != data.shape[0]:
        raise ValueError(
            f"Expected valid_lengths shape ({data.shape[0]},), got {tuple(valid_lengths.shape)}."
        )

    num_steps = data.shape[1]
    step_ids = np.arange(num_steps, dtype=np.int32)
    valid_mask = step_ids[None, :] < valid_lengths[:, None]
    masked = np.where(valid_mask[:, :, None], data, np.nan)
    mean = np.nanmean(masked, axis=0)
    min_vals = np.nanmin(masked, axis=0)
    max_vals = np.nanmax(masked, axis=0)
    return mean, min_vals, max_vals


def _load_series_specs(
    npz_path: Path,
    keys: tuple[str, ...],
    palette: dict[str, tuple[str, str]],
    step_start: int,
    step_end: int | None,
) -> tuple[list[SeriesSpec], int]:
    specs: list[SeriesSpec] = []
    num_traj: int | None = None
    for key in keys:
        data, traj_count, valid_lengths = _load_3d_array(npz_path, key, step_start, step_end)
        if num_traj is None:
            num_traj = traj_count
        elif num_traj != traj_count:
            raise ValueError(f"Trajectory count mismatch for key {key!r}: expected {num_traj}, got {traj_count}.")
        if specs and data.shape != specs[0].data.shape:
            raise ValueError(
                f"Shape mismatch for key {key!r}: expected {specs[0].data.shape}, got {data.shape}."
            )
        color, label = palette[key]
        specs.append(SeriesSpec(key=key, label=label, color=color, data=data, valid_lengths=valid_lengths))
    if num_traj is None:
        raise ValueError("No series loaded.")
    return specs, num_traj


def plot_multi_series_aggregate(
    series_specs: list[SeriesSpec],
    *,
    output_path: Path,
    step_start: int,
    num_traj: int,
    ylabel: str,
    title_prefix: str,
    ncols: int = 6,
    dpi: int = 150,
    share_y: bool = False,
) -> Path:
    if len(series_specs) == 0:
        raise ValueError("series_specs must not be empty.")

    num_traj_data, num_steps, num_dims = series_specs[0].data.shape
    if num_traj_data != num_traj:
        raise ValueError("num_traj argument does not match data.")

    steps = np.arange(step_start, step_start + num_steps)
    nrows = int(np.ceil(num_dims / ncols))
    fig, axes = plt.subplots(nrows, ncols, figsize=(2.8 * ncols, 2.5 * nrows), dpi=dpi, sharex=True)
    fig.patch.set_facecolor("white")
    axes_flat = np.atleast_1d(axes).flatten()

    ylim = None
    if share_y:
        finite_chunks: list[np.ndarray] = []
        for spec in series_specs:
            _, min_vals, max_vals = _masked_stats(spec.data, spec.valid_lengths)
            finite_chunks.append(min_vals[np.isfinite(min_vals)].reshape(-1))
            finite_chunks.append(max_vals[np.isfinite(max_vals)].reshape(-1))
        finite = np.concatenate([chunk for chunk in finite_chunks if chunk.size > 0]) if finite_chunks else np.array([])
        if finite.size > 0:
            pad = 0.05 * max(float(finite.max() - finite.min()), 1e-6)
            ylim = (float(finite.min()) - pad, float(finite.max()) + pad)

    for dim in range(num_dims):
        ax = axes_flat[dim]
        for spec in series_specs:
            mean, min_vals, max_vals = _masked_stats(spec.data, spec.valid_lengths)
            ax.fill_between(
                steps,
                min_vals[:, dim],
                max_vals[:, dim],
                color=spec.color,
                alpha=BAND_ALPHA,
                linewidth=0,
            )
            ax.plot(steps, mean[:, dim], color=spec.color, linewidth=MEAN_LW)
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
    valid_min = min(int(spec.valid_lengths.min()) for spec in series_specs)
    valid_max = max(int(spec.valid_lengths.max()) for spec in series_specs)
    fig.suptitle(
        f"{title_prefix} — all DOFs (steps {step_start}–{step_end - 1})\n"
        f"{num_traj} trajectories, valid lengths across series: {valid_min}–{valid_max}",
        fontsize=13,
        y=0.995,
    )

    legend_handles: list = []
    for spec in series_specs:
        legend_handles.extend(
            [
                Line2D([0], [0], color=spec.color, linewidth=MEAN_LW, label=f"{spec.label} mean"),
                Patch(facecolor=spec.color, edgecolor="none", alpha=BAND_ALPHA, label=f"{spec.label} min–max"),
            ]
        )
    fig.legend(
        handles=legend_handles,
        loc="upper center",
        bbox_to_anchor=(0.5, 0.965),
        ncol=min(6, len(legend_handles)),
        framealpha=0.95,
        fontsize=8,
    )

    output_path.parent.mkdir(parents=True, exist_ok=True)
    fig.tight_layout(rect=[0, 0, 1, 0.93])
    fig.savefig(output_path, bbox_inches="tight", facecolor="white")
    plt.close(fig)
    return output_path


def _default_actions_output(npz_path: Path) -> Path:
    return npz_path.with_name(f"{npz_path.stem}_openloop_actions_compare_all_dims.png")


def _default_joint_pos_output(npz_path: Path) -> Path:
    return npz_path.with_name(f"{npz_path.stem}_openloop_joint_pos_compare_all_dims.png")


def build_arg_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description="Plot open-loop delta rollout action/joint comparisons from NPZ logs."
    )
    parser.add_argument(
        "npz_path",
        type=Path,
        help="Rollout NPZ from play.py --record_delta_model_dataset on DeltaA-OpenLoop.",
    )
    parser.add_argument(
        "--step-range",
        type=_parse_step_range,
        default=(0, None),
        metavar="START:END",
        help="Step slice to plot (end exclusive). Default: 0: (all steps).",
    )
    parser.add_argument(
        "--actions-output",
        type=Path,
        default=None,
        help="Output PNG for action comparison. Default: <npz_stem>_openloop_actions_compare_all_dims.png",
    )
    parser.add_argument(
        "--joint-pos-output",
        type=Path,
        default=None,
        help="Output PNG for joint comparison. Default: <npz_stem>_openloop_joint_pos_compare_all_dims.png",
    )
    parser.add_argument(
        "--default-joint-pos-onnx",
        type=Path,
        default=None,
        help="ONNX with metadata default_joint_pos for converting motion_joint_pos to relative coords.",
    )
    parser.add_argument("--ncols", type=int, default=6, help="Subplot columns. Default: 6.")
    parser.add_argument("--dpi", type=int, default=150, help="Figure DPI. Default: 150.")
    parser.add_argument("--share-y", action="store_true", help="Share y-axis across subplots.")
    parser.add_argument(
        "--skip-actions",
        action="store_true",
        help="Skip the actions / motion-action comparison figure.",
    )
    parser.add_argument(
        "--skip-joint-pos",
        action="store_true",
        help="Skip the obs_joint_pos / motion_joint_pos comparison figure.",
    )
    return parser


def main(argv: list[str] | None = None) -> int:
    args = build_arg_parser().parse_args(argv)
    npz_path = args.npz_path.expanduser().resolve()
    if not npz_path.is_file():
        print(f"[ERROR] NPZ file not found: {npz_path}", file=sys.stderr)
        return 1

    step_start, step_end = args.step_range

    if not args.skip_actions:
        try:
            action_specs, num_traj = _load_series_specs(
                npz_path=npz_path,
                keys=DEFAULT_ACTION_KEYS,
                palette=ACTION_SERIES,
                step_start=step_start,
                step_end=step_end,
            )
            actions_output = (args.actions_output or _default_actions_output(npz_path)).expanduser().resolve()
            saved = plot_multi_series_aggregate(
                action_specs,
                output_path=actions_output,
                step_start=step_start,
                num_traj=num_traj,
                ylabel="Action",
                title_prefix="Open-loop delta actions vs motion actions",
                ncols=args.ncols,
                dpi=args.dpi,
                share_y=args.share_y,
            )
        except (KeyError, ValueError) as exc:
            print(f"[ERROR] Action plot failed: {exc}", file=sys.stderr)
            return 1
        print(f"[INFO] Saved action comparison: {saved}")

    if not args.skip_joint_pos:
        try:
            joint_specs, num_traj = _load_series_specs(
                npz_path=npz_path,
                keys=DEFAULT_JOINT_KEYS,
                palette=JOINT_SERIES,
                step_start=step_start,
                step_end=step_end,
            )
            default_joint_pos, default_source = resolve_default_joint_pos(
                npz_path=npz_path,
                onnx_path=args.default_joint_pos_onnx,
            )
            converted_specs: list[SeriesSpec] = []
            for spec in joint_specs:
                data = spec.data
                if spec.key == "motion_joint_pos":
                    data = to_relative_joint_pos(data, default_joint_pos)
                converted_specs.append(
                    SeriesSpec(
                        key=spec.key,
                        label=spec.label,
                        color=spec.color,
                        data=data,
                        valid_lengths=spec.valid_lengths,
                    )
                )

            joint_output = (args.joint_pos_output or _default_joint_pos_output(npz_path)).expanduser().resolve()
            saved = plot_multi_series_aggregate(
                converted_specs,
                output_path=joint_output,
                step_start=step_start,
                num_traj=num_traj,
                ylabel=JOINT_POS_REL_YLABEL,
                title_prefix="Open-loop obs_joint_pos vs motion_joint_pos",
                ncols=args.ncols,
                dpi=args.dpi,
                share_y=args.share_y,
            )
        except (KeyError, ValueError, FileNotFoundError) as exc:
            print(f"[ERROR] Joint-position plot failed: {exc}", file=sys.stderr)
            return 1
        print(f"[INFO] motion_joint_pos converted with default_joint_pos from {default_source.name}")
        print(f"[INFO] Saved joint-position comparison: {saved}")

    return 0


if __name__ == "__main__":
    raise SystemExit(main())
