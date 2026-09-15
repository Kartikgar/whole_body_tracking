"""Compare open-loop delta rollout NPZ series with aggregate bands.

Default bands are mean/min-max. Pass ``--band p10_90`` for median with a
p10–p90 envelope (more robust to a few fallen rollouts).

Expects NPZ output from ``play.py --record_delta_model_dataset`` on
``Tracking-Flat-G1-DeltaA-OpenLoop-v0``.

Figure 1 (actions): ``actions`` vs ``obs_motion_joint_action`` vs ``motion_joint_action``.
Figure 2 (joints): ``obs_joint_pos`` (source Isaac) vs ``motion_joint_pos``
(target/reference, converted to policy-relative coords via ONNX ``default_joint_pos``).
Figure 3 (bodies): ``body_pos_w`` vs ``motion_body_pos_w`` for selected links
(default: ``torso_link`` x/y/z).

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
MEAN_LW = 2.4
SUBPLOT_TITLE_FS = 12
TICK_FS = 10
AXIS_LABEL_FS = 16
SUPTITLE_FS = 18
LEGEND_FS = 16

DEFAULT_ACTION_KEYS = ("actions", "motion_joint_action")
DEFAULT_JOINT_KEYS = ("obs_joint_pos", "motion_joint_pos")
DEFAULT_BODY_KEYS = ("body_pos_w", "motion_body_pos_w")
DEFAULT_BODY_NAMES = ("torso_link",)
XYZ = ("x", "y", "z")

# Motion-command body order from G1FlatDeltaAOpenLoopEnvCfg.
OPENLOOP_BODY_NAMES = [
    "pelvis",
    "left_hip_pitch_link",
    "left_hip_roll_link",
    "left_hip_yaw_link",
    "left_knee_link",
    "left_ankle_pitch_link",
    "left_ankle_roll_link",
    "right_hip_pitch_link",
    "right_hip_roll_link",
    "right_hip_yaw_link",
    "right_knee_link",
    "right_ankle_pitch_link",
    "right_ankle_roll_link",
    "waist_yaw_link",
    "waist_roll_link",
    "torso_link",
    "left_shoulder_pitch_link",
    "left_shoulder_roll_link",
    "left_shoulder_yaw_link",
    "left_elbow_link",
    "left_wrist_roll_link",
    "left_wrist_pitch_link",
    "left_wrist_yaw_link",
    "right_shoulder_pitch_link",
    "right_shoulder_roll_link",
    "right_shoulder_yaw_link",
    "right_elbow_link",
    "right_wrist_roll_link",
    "right_wrist_pitch_link",
    "right_wrist_yaw_link",
]

ACTION_SERIES = {
    "actions": ("#2563eb", "Delta Action (policy output)"),
    # "obs_motion_joint_action": ("#ea580c", "obs_motion_joint_action"),
    "motion_joint_action": ("#16a34a", "Base Action"),
}

JOINT_SERIES = {
    "obs_joint_pos": ("#7c3aed", "Source (Isaac)"),
    "motion_joint_pos": ("#0d9488", "Target (Genesis)"),
}

BODY_SERIES = {
    "body_pos_w": ("#7c3aed", "Source (Isaac)"),
    "motion_body_pos_w": ("#0d9488", "Target (Genesis)"),
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

    if array.ndim == 4:
        array = array.reshape(array.shape[0], array.shape[1], -1)
    if array.ndim != 3:
        raise ValueError(f"Expected 3D array [num_traj, T, D] for key {key!r}; got {array.shape}.")
    num_traj, total_steps, _ = array.shape
    end = total_steps if step_end is None else min(step_end, total_steps)
    if step_start >= end:
        raise ValueError(f"Invalid step slice [{step_start}:{end}] for trajectory length {total_steps}.")
    valid_lengths = _infer_valid_lengths(array)
    valid_lengths = np.clip(valid_lengths - step_start, 0, end - step_start)
    return array[:, step_start:end, :], num_traj, valid_lengths


def _body_dim_slice(body_names: tuple[str, ...] | list[str], num_dims: int) -> tuple[np.ndarray, list[str]]:
    name_to_idx = {name: idx for idx, name in enumerate(OPENLOOP_BODY_NAMES)}
    dim_indices: list[int] = []
    labels: list[str] = []
    for body_name in body_names:
        if body_name not in name_to_idx:
            raise ValueError(
                f"Unknown body {body_name!r}. Expected one of {list(OPENLOOP_BODY_NAMES)}."
            )
        start = name_to_idx[body_name] * len(XYZ)
        stop = start + len(XYZ)
        if stop > num_dims:
            raise ValueError(
                f"Body {body_name!r} maps to dims [{start}:{stop}), but array only has {num_dims} dims."
            )
        dim_indices.extend(range(start, stop))
        labels.extend(f"{body_name} {axis}" for axis in XYZ)
    return np.asarray(dim_indices, dtype=np.int32), labels


def _slice_series_dims(specs: list[SeriesSpec], dim_indices: np.ndarray) -> list[SeriesSpec]:
    return [
        SeriesSpec(
            key=spec.key,
            label=spec.label,
            color=spec.color,
            data=spec.data[:, :, dim_indices],
            valid_lengths=spec.valid_lengths,
        )
        for spec in specs
    ]


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


def _masked_array(data: np.ndarray, valid_lengths: np.ndarray) -> np.ndarray:
    if data.ndim != 3:
        raise ValueError(f"Expected 3D array [num_traj, T, D], got {data.shape}.")
    if valid_lengths.ndim != 1 or valid_lengths.shape[0] != data.shape[0]:
        raise ValueError(
            f"Expected valid_lengths shape ({data.shape[0]},), got {tuple(valid_lengths.shape)}."
        )
    num_steps = data.shape[1]
    step_ids = np.arange(num_steps, dtype=np.int32)
    valid_mask = step_ids[None, :] < valid_lengths[:, None]
    return np.where(valid_mask[:, :, None], data, np.nan)


def _masked_stats(
    data: np.ndarray,
    valid_lengths: np.ndarray,
    band: str = "minmax",
) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    masked = _masked_array(data, valid_lengths)
    if band == "p10_90":
        with np.errstate(all="ignore"):
            center = np.nanmedian(masked, axis=0)
            low = np.nanquantile(masked, 0.10, axis=0)
            high = np.nanquantile(masked, 0.90, axis=0)
        return center, low, high
    if band != "minmax":
        raise ValueError(f"Unknown band {band!r}. Expected 'minmax' or 'p10_90'.")
    mean = np.nanmean(masked, axis=0)
    min_vals = np.nanmin(masked, axis=0)
    max_vals = np.nanmax(masked, axis=0)
    return mean, min_vals, max_vals


def _band_labels(band: str) -> tuple[str, str]:
    if band == "p10_90":
        return "median", "p10–p90"
    return "mean", "min–max"


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
    dim_labels: list[str] | None = None,
    band: str = "minmax",
) -> Path:
    if len(series_specs) == 0:
        raise ValueError("series_specs must not be empty.")

    num_traj_data, num_steps, num_dims = series_specs[0].data.shape
    if num_traj_data != num_traj:
        raise ValueError("num_traj argument does not match data.")

    steps = np.arange(step_start, step_start + num_steps)
    nrows = int(np.ceil(num_dims / ncols))
    fig, axes = plt.subplots(
        nrows,
        ncols,
        figsize=(3.6 * ncols, 3.1 * nrows + 2.4),
        dpi=dpi,
        sharex=True,
    )
    fig.patch.set_facecolor("white")
    axes_flat = np.atleast_1d(axes).flatten()

    center_label, band_label = _band_labels(band)
    series_stats = [
        (spec, *_masked_stats(spec.data, spec.valid_lengths, band=band)) for spec in series_specs
    ]
    ylim = None
    if share_y:
        finite_chunks: list[np.ndarray] = []
        for _, _, low_vals, high_vals in series_stats:
            finite_chunks.append(low_vals[np.isfinite(low_vals)].reshape(-1))
            finite_chunks.append(high_vals[np.isfinite(high_vals)].reshape(-1))
        finite = np.concatenate([chunk for chunk in finite_chunks if chunk.size > 0]) if finite_chunks else np.array([])
        if finite.size > 0:
            pad = 0.05 * max(float(finite.max() - finite.min()), 1e-6)
            ylim = (float(finite.min()) - pad, float(finite.max()) + pad)

    for dim in range(num_dims):
        ax = axes_flat[dim]
        for spec, center, low_vals, high_vals in series_stats:
            ax.fill_between(
                steps,
                low_vals[:, dim],
                high_vals[:, dim],
                color=spec.color,
                alpha=BAND_ALPHA,
                linewidth=0,
            )
            ax.plot(steps, center[:, dim], color=spec.color, linewidth=MEAN_LW)
        title = dim_labels[dim] if dim_labels is not None and dim < len(dim_labels) else f"dof {dim}"
        ax.set_title(title, fontsize=SUBPLOT_TITLE_FS, pad=4)
        ax.grid(True, color="#e8e8e8", linewidth=0.6)
        ax.tick_params(labelsize=TICK_FS)
        if ylim is not None:
            ax.set_ylim(ylim)

    for idx in range(num_dims, len(axes_flat)):
        axes_flat[idx].axis("off")

    step_end = step_start + num_steps
    fig.supxlabel("Step", fontsize=AXIS_LABEL_FS)
    fig.supylabel(ylabel, fontsize=AXIS_LABEL_FS)
    valid_min = min(int(spec.valid_lengths.min()) for spec in series_specs)
    valid_max = max(int(spec.valid_lengths.max()) for spec in series_specs)
    fig.suptitle(
        f"{title_prefix} — all DOFs (steps {step_start}–{step_end - 1})\n"
        f"{num_traj} trajectories, {center_label} + {band_label}, "
        f"valid lengths {valid_min}–{valid_max}",
        fontsize=SUPTITLE_FS,
        y=0.995,
    )

    output_path.parent.mkdir(parents=True, exist_ok=True)
    fig.tight_layout(rect=[0.02, 0.02, 0.98, 0.84])

    legend_handles: list = []
    for spec in series_specs:
        legend_handles.extend(
            [
                Line2D([0], [0], color=spec.color, linewidth=MEAN_LW, label=f"{spec.label} {center_label}"),
                Patch(facecolor=spec.color, edgecolor="none", alpha=BAND_ALPHA, label=f"{spec.label} {band_label}"),
            ]
        )
    fig.legend(
        handles=legend_handles,
        loc="upper center",
        bbox_to_anchor=(0.5, 0.935),
        ncol=min(2, len(legend_handles)),
        framealpha=0.95,
        fontsize=LEGEND_FS,
        columnspacing=1.6,
        handlelength=2.6,
        borderpad=0.7,
        labelspacing=0.5,
    )
    fig.savefig(output_path, bbox_inches="tight", facecolor="white")
    plt.close(fig)
    return output_path


def _default_actions_output(npz_path: Path) -> Path:
    return npz_path.with_name(f"{npz_path.stem}_openloop_actions_compare_all_dims.png")


def _default_joint_pos_output(npz_path: Path) -> Path:
    return npz_path.with_name(f"{npz_path.stem}_openloop_joint_pos_compare_all_dims.png")


def _default_body_pos_output(npz_path: Path) -> Path:
    return npz_path.with_name(f"{npz_path.stem}_openloop_body_pos_w_compare.png")


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
        "--body-pos-output",
        type=Path,
        default=None,
        help="Output PNG for body-position comparison. Default: <npz_stem>_openloop_body_pos_w_compare.png",
    )
    parser.add_argument(
        "--body-names",
        nargs="+",
        default=list(DEFAULT_BODY_NAMES),
        metavar="NAME",
        help="Body links to plot (xyz each). Default: torso_link.",
    )
    parser.add_argument(
        "--default-joint-pos-onnx",
        type=Path,
        default=None,
        help="ONNX with metadata default_joint_pos for converting motion_joint_pos to relative coords.",
    )
    parser.add_argument(
        "--band",
        choices=("minmax", "p10_90"),
        default="minmax",
        help="Envelope around the center line. minmax uses mean+min/max; p10_90 uses median+p10/p90.",
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
    parser.add_argument(
        "--skip-body-pos",
        action="store_true",
        help="Skip the body_pos_w / motion_body_pos_w comparison figure.",
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
                band=args.band,
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
                title_prefix="Source vs target env joint positions",
                ncols=args.ncols,
                dpi=args.dpi,
                share_y=args.share_y,
                band=args.band,
            )
        except (KeyError, ValueError, FileNotFoundError) as exc:
            print(f"[ERROR] Joint-position plot failed: {exc}", file=sys.stderr)
            return 1
        print(f"[INFO] motion_joint_pos converted with default_joint_pos from {default_source.name}")
        print(f"[INFO] Saved joint-position comparison: {saved}")

    if not args.skip_body_pos:
        try:
            body_specs, num_traj = _load_series_specs(
                npz_path=npz_path,
                keys=DEFAULT_BODY_KEYS,
                palette=BODY_SERIES,
                step_start=step_start,
                step_end=step_end,
            )
            dim_indices, dim_labels = _body_dim_slice(args.body_names, body_specs[0].data.shape[-1])
            body_specs = _slice_series_dims(body_specs, dim_indices)
            body_ncols = min(args.ncols, body_specs[0].data.shape[-1])
            body_title = (
                "Source vs target torso position"
                if tuple(args.body_names) == DEFAULT_BODY_NAMES
                else "Source vs target body positions"
            )
            body_output = (args.body_pos_output or _default_body_pos_output(npz_path)).expanduser().resolve()
            saved = plot_multi_series_aggregate(
                body_specs,
                output_path=body_output,
                step_start=step_start,
                num_traj=num_traj,
                ylabel="Body position in world frame (m)",
                title_prefix=body_title,
                ncols=body_ncols,
                dpi=args.dpi,
                share_y=args.share_y,
                dim_labels=dim_labels,
                band=args.band,
            )
        except (KeyError, ValueError) as exc:
            print(f"[ERROR] Body-position plot failed: {exc}", file=sys.stderr)
            return 1
        print(f"[INFO] Saved body-position comparison: {saved}")

    return 0


if __name__ == "__main__":
    raise SystemExit(main())
