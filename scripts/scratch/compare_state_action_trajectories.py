"""Compare Genesis vs Isaac Lab state-action rollout NPZ files on aggregate plots.

Produces two overlay figures (state = policy-relative joint positions, action = policy
actions) with Genesis and Isaac Lab distinguished by color. Intended for NPZ outputs from
``evaluate_sim2sim_genesis.py`` / modular sim2sim and ``play.py --record_state_action_trajectories``.
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

GENESIS_COLOR = "#ea580c"
ISAACLAB_COLOR = "#2563eb"
BAND_ALPHA = 0.22
MEAN_LW = 1.8

STATE_KEY = "joint_pos"
ACTION_KEYS = ("actions", "action")


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


def _resolve_action_key(npz_path: Path) -> str:
    with np.load(npz_path) as data:
        for key in ACTION_KEYS:
            if key in data.files:
                return key
    raise KeyError(
        f"No action key found in {npz_path}. Expected one of {ACTION_KEYS}. "
        f"Available keys: {sorted(data.files)}"
    )


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
        raise ValueError(f"Expected 3D array [num_traj, T, D] for key {key!r} in {npz_path}; got {series.shape}.")
    _, total_steps, _ = series.shape
    end = total_steps if step_end is None else min(step_end, total_steps)
    if step_start >= end:
        raise ValueError(f"Invalid step slice [{step_start}:{end}] for trajectory length {total_steps} in {npz_path}.")
    valid_lengths = _infer_valid_lengths(series)
    valid_lengths = np.clip(valid_lengths - step_start, 0, end - step_start)
    return series[:, step_start:end, :], fps, valid_lengths


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


def _masked_stats(data: np.ndarray, valid_lengths: np.ndarray) -> tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
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
    counts = valid_mask.sum(axis=0).astype(np.int32)
    return mean, min_vals, max_vals, counts


def _validate_compatible_shapes(genesis: np.ndarray, isaaclab: np.ndarray, *, label: str) -> None:
    if genesis.ndim != 3 or isaaclab.ndim != 3:
        raise ValueError(f"Expected 3D arrays for {label}, got genesis={genesis.shape}, isaaclab={isaaclab.shape}.")
    if genesis.shape[1:] != isaaclab.shape[1:]:
        raise ValueError(
            f"Shape mismatch for {label}: genesis {genesis.shape} vs isaaclab {isaaclab.shape}. "
            "Expected matching [T, D] after any step slicing."
        )


def plot_compare_aggregate_series(
    genesis_data: np.ndarray,
    isaaclab_data: np.ndarray,
    *,
    genesis_valid_lengths: np.ndarray,
    isaaclab_valid_lengths: np.ndarray,
    output_path: Path,
    step_start: int,
    ylabel: str,
    title_prefix: str,
    genesis_label: str = "Genesis",
    isaaclab_label: str = "Isaac Lab",
    ncols: int = 6,
    dpi: int = 150,
    share_y: bool = False,
) -> Path:
    _validate_compatible_shapes(genesis_data, isaaclab_data, label=title_prefix)

    num_steps, num_dims = genesis_data.shape[1:]
    steps = np.arange(step_start, step_start + num_steps)
    nrows = int(np.ceil(num_dims / ncols))
    genesis_mean, genesis_min, genesis_max, genesis_counts = _masked_stats(genesis_data, genesis_valid_lengths)
    isaaclab_mean, isaaclab_min, isaaclab_max, isaaclab_counts = _masked_stats(isaaclab_data, isaaclab_valid_lengths)

    fig, axes = plt.subplots(nrows, ncols, figsize=(2.8 * ncols, 2.4 * nrows), dpi=dpi, sharex=True)
    fig.patch.set_facecolor("white")
    axes_flat = np.atleast_1d(axes).flatten()

    ylim = None
    if share_y:
        finite = np.concatenate(
            [
                genesis_min[np.isfinite(genesis_min)].reshape(-1),
                genesis_max[np.isfinite(genesis_max)].reshape(-1),
                isaaclab_min[np.isfinite(isaaclab_min)].reshape(-1),
                isaaclab_max[np.isfinite(isaaclab_max)].reshape(-1),
            ]
        )
        if finite.size > 0:
            pad = 0.05 * max(float(finite.max() - finite.min()), 1e-6)
            ylim = (float(finite.min()) - pad, float(finite.max()) + pad)

    for dim in range(num_dims):
        ax = axes_flat[dim]
        ax.fill_between(steps, genesis_min[:, dim], genesis_max[:, dim], color=GENESIS_COLOR, alpha=BAND_ALPHA, linewidth=0)
        ax.plot(steps, genesis_mean[:, dim], color=GENESIS_COLOR, linewidth=MEAN_LW)
        ax.fill_between(steps, isaaclab_min[:, dim], isaaclab_max[:, dim], color=ISAACLAB_COLOR, alpha=BAND_ALPHA, linewidth=0)
        ax.plot(steps, isaaclab_mean[:, dim], color=ISAACLAB_COLOR, linewidth=MEAN_LW)
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
    genesis_valid_min = int(genesis_counts.min()) if genesis_counts.size > 0 else 0
    genesis_valid_max = int(genesis_counts.max()) if genesis_counts.size > 0 else 0
    isaaclab_valid_min = int(isaaclab_counts.min()) if isaaclab_counts.size > 0 else 0
    isaaclab_valid_max = int(isaaclab_counts.max()) if isaaclab_counts.size > 0 else 0
    fig.suptitle(
        f"{title_prefix} — Genesis vs Isaac Lab (steps {step_start}–{step_end - 1})\n"
        f"Genesis valid per step: {genesis_valid_min}–{genesis_valid_max}; "
        f"Isaac Lab valid per step: {isaaclab_valid_min}–{isaaclab_valid_max}",
        fontsize=13,
        y=0.995,
    )

    legend_handles = [
        Line2D([0], [0], color=GENESIS_COLOR, linewidth=MEAN_LW, label=f"{genesis_label} mean"),
        Patch(facecolor=GENESIS_COLOR, edgecolor="none", alpha=BAND_ALPHA, label=f"{genesis_label} min–max"),
        Line2D([0], [0], color=ISAACLAB_COLOR, linewidth=MEAN_LW, label=f"{isaaclab_label} mean"),
        Patch(facecolor=ISAACLAB_COLOR, edgecolor="none", alpha=BAND_ALPHA, label=f"{isaaclab_label} min–max"),
    ]
    fig.legend(handles=legend_handles, loc="upper center", bbox_to_anchor=(0.5, 0.965), ncol=2, framealpha=0.95, fontsize=9)

    output_path.parent.mkdir(parents=True, exist_ok=True)
    fig.tight_layout(rect=[0, 0, 1, 0.94])
    fig.savefig(output_path, bbox_inches="tight", facecolor="white")
    plt.close(fig)
    return output_path


def build_arg_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description="Overlay Genesis and Isaac Lab state-action rollout NPZ aggregate plots."
    )
    parser.add_argument(
        "genesis_npz",
        type=Path,
        help="Genesis / sim2sim state-action NPZ (e.g. under logs/sim2sim_eval/).",
    )
    parser.add_argument(
        "isaaclab_npz",
        type=Path,
        help="Isaac Lab play.py state-action NPZ (e.g. under state_action_datasets/).",
    )
    parser.add_argument(
        "--output-dir",
        type=Path,
        default=None,
        help="Directory for PNG outputs. Default: same directory as genesis_npz.",
    )
    parser.add_argument(
        "--output-stem",
        type=str,
        default=None,
        help="Filename stem for outputs (without extension). Default: <genesis_stem>_vs_<isaac_stem>.",
    )
    parser.add_argument(
        "--step-range",
        type=_parse_step_range,
        default=(0, None),
        metavar="START:END",
        help="Step slice to plot (end exclusive). Default: 0: (all steps).",
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
            "`joint_pos` to relative coordinates. Default: infer from isaaclab_npz run directory."
        ),
    )
    return parser


def main(argv: list[str] | None = None) -> int:
    args = build_arg_parser().parse_args(argv)
    genesis_npz = args.genesis_npz.expanduser().resolve()
    isaaclab_npz = args.isaaclab_npz.expanduser().resolve()
    for path in (genesis_npz, isaaclab_npz):
        if not path.is_file():
            print(f"[ERROR] NPZ file not found: {path}", file=sys.stderr)
            return 1

    output_dir = (args.output_dir or genesis_npz.parent).expanduser().resolve()
    step_start, step_end = args.step_range
    output_stem = args.output_stem
    if output_stem is None:
        output_stem = f"{genesis_npz.stem}_vs_{isaaclab_npz.stem}"

    try:
        genesis_action_key = _resolve_action_key(genesis_npz)
        isaaclab_action_key = _resolve_action_key(isaaclab_npz)
        genesis_state, genesis_fps, genesis_state_lengths = _load_series(genesis_npz, STATE_KEY, step_start, step_end)
        isaaclab_state, isaaclab_fps, isaaclab_state_lengths = _load_series(isaaclab_npz, STATE_KEY, step_start, step_end)
        genesis_actions, _, genesis_action_lengths = _load_series(genesis_npz, genesis_action_key, step_start, step_end)
        isaaclab_actions, _, isaaclab_action_lengths = _load_series(isaaclab_npz, isaaclab_action_key, step_start, step_end)
        default_joint_pos, default_source = resolve_default_joint_pos(
            npz_path=isaaclab_npz,
            onnx_path=args.default_joint_pos_onnx,
        )
    except (KeyError, ValueError, FileNotFoundError) as exc:
        print(f"[ERROR] {exc}", file=sys.stderr)
        return 1

    genesis_state = to_relative_joint_pos(genesis_state, default_joint_pos)
    isaaclab_state = to_relative_joint_pos(isaaclab_state, default_joint_pos)

    state_path = output_dir / f"{output_stem}_state_joint_pos_rel.png"
    action_path = output_dir / f"{output_stem}_actions.png"

    saved_state = plot_compare_aggregate_series(
        genesis_state,
        isaaclab_state,
        genesis_valid_lengths=genesis_state_lengths,
        isaaclab_valid_lengths=isaaclab_state_lengths,
        output_path=state_path,
        step_start=step_start,
        ylabel=JOINT_POS_REL_YLABEL,
        title_prefix=JOINT_POS_REL_TITLE,
        ncols=args.ncols,
        dpi=args.dpi,
        share_y=args.share_y,
    )
    saved_actions = plot_compare_aggregate_series(
        genesis_actions,
        isaaclab_actions,
        genesis_valid_lengths=genesis_action_lengths,
        isaaclab_valid_lengths=isaaclab_action_lengths,
        output_path=action_path,
        step_start=step_start,
        ylabel="Action",
        title_prefix="Policy actions",
        ncols=args.ncols,
        dpi=args.dpi,
        share_y=args.share_y,
    )

    fps_parts = []
    if genesis_fps is not None:
        fps_parts.append(f"genesis_fps={genesis_fps:.3f}")
    if isaaclab_fps is not None:
        fps_parts.append(f"isaaclab_fps={isaaclab_fps:.3f}")
    fps_text = f" ({', '.join(fps_parts)})" if fps_parts else ""

    print(f"[INFO] genesis_npz={genesis_npz}")
    print(f"[INFO] isaaclab_npz={isaaclab_npz}")
    print(f"[INFO] genesis_action_key={genesis_action_key!r}, isaaclab_action_key={isaaclab_action_key!r}")
    print(f"[INFO] default_joint_pos from {default_source}")
    print(
        f"[INFO] trajectories={genesis_state.shape[0]}, steps={genesis_state.shape[1]}, "
        f"dims={genesis_state.shape[2]}{fps_text}"
    )
    print(f"[INFO] state plot -> {saved_state}")
    print(f"[INFO] action plot -> {saved_actions}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
