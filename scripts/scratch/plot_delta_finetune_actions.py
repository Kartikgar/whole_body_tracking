"""Plot base, delta, and composed actions from delta-finetune rollout NPZ files.

Expects NPZ output from ``play.py --record_delta_model_dataset`` on a
``Tracking-Flat-G1-DeltaA-Finetune-v0`` run. In that format:

- ``actions`` stores frozen delta-policy outputs
- ``obs_current_action`` stores same-step base-policy actions
- ``obs_joint_pos`` stores relative joint-position observation terms

Each action subplot shows mean and min-max across trajectories for base (blue) and
delta (orange). Optionally overlay composed mean ``base + delta`` as a green dotted
line via ``--plot-composed-mean``.
``obs_joint_pos`` is plotted separately as a single-series mean/min-max grid.

.. code-block:: bash

    python whole_body_tracking/scripts/scratch/plot_delta_finetune_actions.py \\
        whole_body_tracking/logs/rsl_rl/2026.05.20/g1_deltaa_finetune/.../model_39500_2026-05-18_19-20-28.npz

    python whole_body_tracking/scripts/scratch/plot_delta_finetune_actions.py rollout.npz \\
        --step-range 0:250 \\
        --output /tmp/combined_actions.png
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
from joint_plot_utils import JOINT_POS_REL_TITLE, JOINT_POS_REL_YLABEL
from matplotlib.lines import Line2D
from matplotlib.patches import Patch


DEFAULT_BASE_KEY = "obs_current_action"
DEFAULT_DELTA_KEY = "actions"
DEFAULT_OBS_JOINT_POS_KEY = "obs_joint_pos"

BASE_COLOR = "#2563eb"
DELTA_COLOR = "#ea580c"
COMPOSED_COLOR = "#16a34a"
JOINT_POS_COLOR = "#7c3aed"
BAND_ALPHA = 0.30
MEAN_LW = 2.2


def _parse_step_range(text: str) -> tuple[int, int | None]:
    """Parse ``start:end`` (end exclusive), ``start:``, or ``:end``."""
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


def _default_output_path(npz_path: Path) -> Path:
    return npz_path.with_name(f"{npz_path.stem}_base_plus_delta_all_dims.png")


def _default_obs_joint_pos_output_path(npz_path: Path) -> Path:
    return npz_path.with_name(f"{npz_path.stem}_joint_pos_rel_all_dims.png")


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
    """Infer per-trajectory valid lengths by dropping trailing constant-value padding.

    Recorder padding repeats the final valid frame to the rollout max length. We keep a
    frame as valid while any dimension changes from the previous timestep.
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


def _load_action_arrays(
    npz_path: Path,
    base_key: str,
    delta_key: str,
    step_start: int,
    step_end: int | None,
) -> tuple[np.ndarray, np.ndarray, int, np.ndarray, np.ndarray]:
    base, num_traj, base_valid_lengths = _load_3d_array(npz_path, base_key, step_start, step_end)
    delta, delta_num_traj, delta_valid_lengths = _load_3d_array(npz_path, delta_key, step_start, step_end)
    if num_traj != delta_num_traj:
        raise ValueError(f"Trajectory count mismatch: base={num_traj}, delta={delta_num_traj}.")
    if base.shape != delta.shape:
        raise ValueError(f"Base/delta shape mismatch: base={base.shape}, delta={delta.shape}.")
    return base, delta, num_traj, base_valid_lengths, delta_valid_lengths


def plot_combined_actions(
    base: np.ndarray,
    delta: np.ndarray,
    *,
    output_path: Path,
    step_start: int,
    num_traj: int,
    base_valid_lengths: np.ndarray,
    delta_valid_lengths: np.ndarray,
    share_y: bool = False,
    plot_composed_mean: bool = False,
    ncols: int = 6,
    dpi: int = 150,
    title_prefix: str | None = None,
    dim_labels: list[str] | None = None,
) -> Path:
    num_traj_data, num_steps, num_dims = base.shape
    if num_traj_data != num_traj:
        raise ValueError("num_traj argument does not match data.")

    composed_valid_lengths = None
    composed = None
    if plot_composed_mean:
        composed_valid_lengths = np.minimum(base_valid_lengths, delta_valid_lengths)
        composed = base + delta

    steps = np.arange(step_start, step_start + num_steps)
    nrows = int(np.ceil(num_dims / ncols))

    fig, axes = plt.subplots(nrows, ncols, figsize=(2.8 * ncols, 2.4 * nrows), dpi=dpi, sharex=True)
    fig.patch.set_facecolor("white")
    axes_flat = np.atleast_1d(axes).flatten()

    ylim = None
    if share_y:
        finite_chunks: list[np.ndarray] = []
        series_specs: list[tuple[np.ndarray, np.ndarray]] = [
            (base, base_valid_lengths),
            (delta, delta_valid_lengths),
        ]
        if plot_composed_mean and composed is not None and composed_valid_lengths is not None:
            series_specs.append((composed, composed_valid_lengths))
        for data, valid_lengths in series_specs:
            _, min_vals, max_vals = _masked_stats(data, valid_lengths)
            finite_chunks.append(min_vals[np.isfinite(min_vals)].reshape(-1))
            finite_chunks.append(max_vals[np.isfinite(max_vals)].reshape(-1))
        finite = np.concatenate([chunk for chunk in finite_chunks if chunk.size > 0]) if finite_chunks else np.array([])
        if finite.size > 0:
            pad = 0.05 * max(float(finite.max() - finite.min()), 1e-6)
            ylim = (float(finite.min()) - pad, float(finite.max()) + pad)

    base_mean, base_min, base_max = _masked_stats(base, base_valid_lengths)
    delta_mean, delta_min, delta_max = _masked_stats(delta, delta_valid_lengths)
    composed_mean = None
    if plot_composed_mean and composed is not None and composed_valid_lengths is not None:
        composed_mean, _, _ = _masked_stats(composed, composed_valid_lengths)

    for dim in range(num_dims):
        ax = axes_flat[dim]

        ax.fill_between(
            steps,
            base_min[:, dim],
            base_max[:, dim],
            color=BASE_COLOR,
            alpha=BAND_ALPHA,
            linewidth=0,
        )
        ax.plot(steps, base_mean[:, dim], color=BASE_COLOR, linewidth=MEAN_LW)

        ax.fill_between(
            steps,
            delta_min[:, dim],
            delta_max[:, dim],
            color=DELTA_COLOR,
            alpha=BAND_ALPHA,
            linewidth=0,
        )
        ax.plot(steps, delta_mean[:, dim], color=DELTA_COLOR, linewidth=MEAN_LW)
        if composed_mean is not None:
            ax.plot(steps, composed_mean[:, dim], color=COMPOSED_COLOR, linewidth=MEAN_LW, linestyle=":")

        if dim_labels is not None and dim < len(dim_labels):
            ax.set_title(dim_labels[dim], fontsize=8, pad=2)
        else:
            ax.set_title(f"dim {dim}", fontsize=9, pad=2)
        ax.grid(True, color="#e8e8e8", linewidth=0.6)
        ax.tick_params(labelsize=7)
        if ylim is not None:
            ax.set_ylim(ylim)

    for idx in range(num_dims, len(axes_flat)):
        axes_flat[idx].axis("off")

    step_end = step_start + num_steps
    if title_prefix is not None:
        prefix = title_prefix
    elif plot_composed_mean:
        prefix = "Base + delta + composed actions"
    else:
        prefix = "Base + delta actions"
    valid_mins = [int(base_valid_lengths.min()), int(delta_valid_lengths.min())]
    valid_maxs = [int(base_valid_lengths.max()), int(delta_valid_lengths.max())]
    if plot_composed_mean and composed_valid_lengths is not None:
        valid_mins.append(int(composed_valid_lengths.min()))
        valid_maxs.append(int(composed_valid_lengths.max()))
    valid_min = min(valid_mins)
    valid_max = max(valid_maxs)
    fig.supxlabel("Step", fontsize=11)
    fig.supylabel("Action", fontsize=11)
    fig.suptitle(
        f"{prefix} — all dimensions (steps {step_start}–{step_end - 1})\n"
        f"{num_traj} trajectories, valid lengths: {valid_min}–{valid_max}",
        fontsize=13,
        y=0.995,
    )

    legend_handles = [
        Line2D([0], [0], color=BASE_COLOR, linewidth=MEAN_LW, label="Base mean"),
        Patch(facecolor=BASE_COLOR, edgecolor="none", alpha=BAND_ALPHA, label="Base min–max"),
        Line2D([0], [0], color=DELTA_COLOR, linewidth=MEAN_LW, label="Delta mean"),
        Patch(facecolor=DELTA_COLOR, edgecolor="none", alpha=BAND_ALPHA, label="Delta min–max"),
    ]
    if plot_composed_mean:
        legend_handles.append(
            Line2D(
                [0],
                [0],
                color=COMPOSED_COLOR,
                linewidth=MEAN_LW,
                linestyle=":",
                label="Composed mean (base + delta)",
            )
        )
    fig.legend(
        handles=legend_handles,
        loc="upper center",
        bbox_to_anchor=(0.5, 0.965),
        ncol=len(legend_handles),
        framealpha=0.95,
        fontsize=9,
    )

    output_path.parent.mkdir(parents=True, exist_ok=True)
    fig.tight_layout(rect=[0, 0, 1, 0.94])
    fig.savefig(output_path, bbox_inches="tight", facecolor="white")
    plt.close(fig)
    return output_path


def plot_aggregate_series(
    data: np.ndarray,
    *,
    output_path: Path,
    step_start: int,
    num_traj: int,
    valid_lengths: np.ndarray,
    series_color: str,
    ylabel: str,
    title_prefix: str,
    share_y: bool = False,
    ncols: int = 6,
    dpi: int = 150,
    dim_labels: list[str] | None = None,
) -> Path:
    num_traj_data, num_steps, num_dims = data.shape
    if num_traj_data != num_traj:
        raise ValueError("num_traj argument does not match data.")

    steps = np.arange(step_start, step_start + num_steps)
    nrows = int(np.ceil(num_dims / ncols))

    fig, axes = plt.subplots(nrows, ncols, figsize=(2.8 * ncols, 2.4 * nrows), dpi=dpi, sharex=True)
    fig.patch.set_facecolor("white")
    axes_flat = np.atleast_1d(axes).flatten()

    ylim = None
    if share_y:
        _, min_vals, max_vals = _masked_stats(data, valid_lengths)
        finite = np.concatenate(
            [
                min_vals[np.isfinite(min_vals)].reshape(-1),
                max_vals[np.isfinite(max_vals)].reshape(-1),
            ]
        )
        if finite.size > 0:
            pad = 0.05 * max(float(finite.max() - finite.min()), 1e-6)
            ylim = (float(finite.min()) - pad, float(finite.max()) + pad)

    for dim in range(num_dims):
        ax = axes_flat[dim]
        mean, min_vals, max_vals = _masked_stats(data, valid_lengths)
        ax.fill_between(
            steps,
            min_vals[:, dim],
            max_vals[:, dim],
            color=series_color,
            alpha=BAND_ALPHA,
            linewidth=0,
        )
        ax.plot(steps, mean[:, dim], color=series_color, linewidth=MEAN_LW)
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
    valid_min = int(valid_lengths.min()) if valid_lengths.size > 0 else 0
    valid_max = int(valid_lengths.max()) if valid_lengths.size > 0 else 0
    fig.supxlabel("Step", fontsize=11)
    fig.supylabel(ylabel, fontsize=11)
    fig.suptitle(
        f"{title_prefix} — all DOFs (steps {step_start}–{step_end - 1})\n"
        f"{num_traj} trajectories, valid lengths: {valid_min}–{valid_max}",
        fontsize=13,
        y=0.995,
    )
    legend_handles = [
        Line2D([0], [0], color=series_color, linewidth=MEAN_LW, label="Mean"),
        Patch(facecolor=series_color, edgecolor="none", alpha=BAND_ALPHA, label="Min–max"),
    ]
    fig.legend(handles=legend_handles, loc="upper center", bbox_to_anchor=(0.5, 0.965), ncol=2, framealpha=0.95, fontsize=9)

    output_path.parent.mkdir(parents=True, exist_ok=True)
    fig.tight_layout(rect=[0, 0, 1, 0.94])
    fig.savefig(output_path, bbox_inches="tight", facecolor="white")
    plt.close(fig)
    return output_path


def build_arg_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description=(
            "Plot base, delta, and composed action means/min-max from a delta-finetune rollout NPZ."
        )
    )
    parser.add_argument(
        "npz_path",
        type=Path,
        help="Rollout NPZ from play.py --record_delta_model_dataset on a Delta-A finetune task.",
    )
    parser.add_argument(
        "--output",
        "-o",
        type=Path,
        default=None,
        help="Output PNG path. Default: <npz_stem>_base_plus_delta_all_dims.png beside the NPZ.",
    )
    parser.add_argument(
        "--step-range",
        type=_parse_step_range,
        default=(0, 250),
        metavar="START:END",
        help="Step slice to plot (end exclusive). Default: 0:250. Use e.g. 0: or :500 for open bounds.",
    )
    parser.add_argument(
        "--base-key",
        default=DEFAULT_BASE_KEY,
        help=f"NPZ key for base-policy actions. Default: {DEFAULT_BASE_KEY}.",
    )
    parser.add_argument(
        "--delta-key",
        default=DEFAULT_DELTA_KEY,
        help=f"NPZ key for frozen delta actions. Default: {DEFAULT_DELTA_KEY}.",
    )
    parser.add_argument(
        "--plot-composed-mean",
        action="store_true",
        default=False,
        help="Overlay composed mean (base + delta) as a green dotted line. Default: off.",
    )
    parser.add_argument(
        "--share-y",
        action="store_true",
        help="Use one shared y-axis across all subplots (can flatten small-magnitude joints).",
    )
    parser.add_argument(
        "--ncols",
        type=int,
        default=6,
        help="Number of subplot columns. Default: 6.",
    )
    parser.add_argument(
        "--dpi",
        type=int,
        default=150,
        help="Figure DPI. Default: 150.",
    )
    parser.add_argument(
        "--title",
        default=None,
        help="Optional custom title prefix.",
    )
    parser.add_argument(
        "--plot-obs-joint-pos",
        action=argparse.BooleanOptionalAction,
        default=True,
        help="Also plot obs_joint_pos mean/min-max grid when the key exists. Default: true.",
    )
    parser.add_argument(
        "--obs-joint-pos-key",
        default=DEFAULT_OBS_JOINT_POS_KEY,
        help=f"NPZ key for relative joint-position observations. Default: {DEFAULT_OBS_JOINT_POS_KEY}.",
    )
    parser.add_argument(
        "--obs-joint-pos-output",
        type=Path,
        default=None,
        help="Output PNG for obs_joint_pos plot. Default: <npz_stem>_joint_pos_rel_all_dims.png beside the NPZ.",
    )
    return parser


def main(argv: list[str] | None = None) -> int:
    args = build_arg_parser().parse_args(argv)

    npz_path = args.npz_path.expanduser().resolve()
    if not npz_path.is_file():
        print(f"[ERROR] NPZ file not found: {npz_path}", file=sys.stderr)
        return 1

    output_path = (args.output or _default_output_path(npz_path)).expanduser().resolve()
    step_start, step_end = args.step_range

    try:
        base, delta, num_traj, base_valid_lengths, delta_valid_lengths = _load_action_arrays(
            npz_path=npz_path,
            base_key=args.base_key,
            delta_key=args.delta_key,
            step_start=step_start,
            step_end=step_end,
        )
        saved_path = plot_combined_actions(
            base,
            delta,
            output_path=output_path,
            step_start=step_start,
            num_traj=num_traj,
            base_valid_lengths=base_valid_lengths,
            delta_valid_lengths=delta_valid_lengths,
            share_y=args.share_y,
            plot_composed_mean=args.plot_composed_mean,
            ncols=args.ncols,
            dpi=args.dpi,
            title_prefix=args.title,
        )
    except (KeyError, ValueError) as exc:
        print(f"[ERROR] {exc}", file=sys.stderr)
        return 1

    _, num_steps, num_dims = base.shape
    print(f"[INFO] Loaded: {npz_path.name}")
    print(f"[INFO] Trajectories: {num_traj}, steps: {num_steps}, action dims: {num_dims}")
    print(f"[INFO] Step slice: [{step_start}:{step_start + num_steps})")
    valid_lengths_msg = (
        f"base {int(base_valid_lengths.min())}-{int(base_valid_lengths.max())}, "
        f"delta {int(delta_valid_lengths.min())}-{int(delta_valid_lengths.max())}"
    )
    if args.plot_composed_mean:
        composed_valid_lengths = np.minimum(base_valid_lengths, delta_valid_lengths)
        valid_lengths_msg += (
            f", composed {int(composed_valid_lengths.min())}-{int(composed_valid_lengths.max())}"
        )
    print(f"[INFO] Valid lengths (masked padding): {valid_lengths_msg}")
    print(f"[INFO] Saved action plot: {saved_path}")

    if args.plot_obs_joint_pos:
        obs_joint_pos_output = (
            args.obs_joint_pos_output or _default_obs_joint_pos_output_path(npz_path)
        ).expanduser().resolve()
        try:
            obs_joint_pos, obs_num_traj, obs_valid_lengths = _load_3d_array(
                npz_path=npz_path,
                key=args.obs_joint_pos_key,
                step_start=step_start,
                step_end=step_end,
            )
            obs_saved_path = plot_aggregate_series(
                obs_joint_pos,
                output_path=obs_joint_pos_output,
                step_start=step_start,
                num_traj=obs_num_traj,
                valid_lengths=obs_valid_lengths,
                series_color=JOINT_POS_COLOR,
                ylabel=JOINT_POS_REL_YLABEL,
                title_prefix=JOINT_POS_REL_TITLE,
                share_y=args.share_y,
                ncols=args.ncols,
                dpi=args.dpi,
            )
        except KeyError as exc:
            print(f"[WARN] Skipping obs_joint_pos plot: {exc}", file=sys.stderr)
        except ValueError as exc:
            print(f"[ERROR] {exc}", file=sys.stderr)
            return 1
        else:
            _, obs_steps, obs_dims = obs_joint_pos.shape
            print(
                f"[INFO] obs_joint_pos dims: {obs_dims}, steps: {obs_steps}, "
                f"valid lengths: {int(obs_valid_lengths.min())}-{int(obs_valid_lengths.max())}"
            )
            print(f"[INFO] Saved obs_joint_pos plot: {obs_saved_path}")

    return 0


if __name__ == "__main__":
    raise SystemExit(main())
