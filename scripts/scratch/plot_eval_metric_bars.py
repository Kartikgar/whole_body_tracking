"""Bar chart of eval JSON metrics (mean ± std).

Prefers ``*_succ_*`` keys when both files have them. Otherwise uses the overall mean/std.
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np

# (panel title, succ mean, succ std, overall mean, overall std, unit)
METRICS = (
    ("MPJPE_l", "mpjpe_l_succ_mean", "mpjpe_l_succ_std", "mpjpe_l_mean", "mpjpe_l_std", "mm"),
    ("MPJPE", "mpjpe_succ_mean", "mpjpe_succ_std", "mpjpe_mean", "mpjpe_std", "mm"),
    ("accn_dist", "accel_dist_succ_mean", "accel_dist_succ_std", "accel_dist_mean", "accel_dist_std", "mm / frame²"),
    ("vel_dist", "vel_dist_succ_mean", "vel_dist_succ_std", "vel_dist_mean", "vel_dist_std", "mm / frame"),
)

BASE_COLOR = "#77AADD"
FINETUNE_COLOR = "#EE8866"
BASE_EDGE = "#4477AA"
FINETUNE_EDGE = "#CC6677"
ERROR_COLOR = "#4A4A4A"
TITLE_FS = 18
AXIS_FS = 14
TICK_FS = 12


def _load_json(path: Path) -> dict:
    return json.loads(path.read_text())


def _metric_pair(payload: dict, succ_key: str, overall_key: str, *, use_succ: bool) -> float:
    key = succ_key if use_succ else overall_key
    if key not in payload:
        raise KeyError(f"Missing metric key {key!r}.")
    return float(payload[key])


def plot_metric_bars(
    left: dict,
    right: dict,
    *,
    output_path: Path,
    left_label: str,
    right_label: str,
    figure_title: str,
    use_succ: bool,
) -> Path:
    fig, axes = plt.subplots(1, 4, figsize=(16, 5.6), dpi=150)
    fig.patch.set_facecolor("white")
    x = np.array([0.0, 1.0])
    width = 0.62

    for ax, (panel_title, succ_mean, succ_std, mean_key, std_key, unit) in zip(axes, METRICS):
        means = np.array(
            [
                _metric_pair(left, succ_mean, mean_key, use_succ=use_succ),
                _metric_pair(right, succ_mean, mean_key, use_succ=use_succ),
            ]
        )
        stds = np.array(
            [
                _metric_pair(left, succ_std, std_key, use_succ=use_succ),
                _metric_pair(right, succ_std, std_key, use_succ=use_succ),
            ]
        )
        ax.bar(
            x,
            means,
            width=width,
            color=[BASE_COLOR, FINETUNE_COLOR],
            edgecolor=[BASE_EDGE, FINETUNE_EDGE],
            linewidth=0.8,
            yerr=stds,
            capsize=5,
            error_kw={"elinewidth": 1.2, "ecolor": ERROR_COLOR, "capthick": 1.2},
        )
        ax.set_title(panel_title, fontsize=TITLE_FS, pad=8)
        ax.set_ylabel(unit, fontsize=AXIS_FS)
        ax.set_xticks(x)
        ax.set_xticklabels([left_label, right_label], fontsize=TICK_FS)
        ax.tick_params(axis="y", labelsize=TICK_FS)
        ax.grid(True, axis="y", color="#E6E6E6", linewidth=0.7)
        ax.set_axisbelow(True)
        ax.spines["top"].set_visible(False)
        ax.spines["right"].set_visible(False)
        ax.spines["left"].set_color("#8A8A8A")
        ax.spines["bottom"].set_color("#8A8A8A")
        ymax = float((means + stds).max())
        ax.set_ylim(0.0, ymax * 1.22)
        for xi, mean, std in zip(x, means, stds):
            ax.text(xi, mean + std + 0.03 * ymax, f"{mean:.2f}", ha="center", va="bottom", fontsize=11)

    fig.tight_layout(rect=[0.0, 0.0, 1.0, 0.88]) # type: ignore
    fig.suptitle(figure_title, fontsize=TITLE_FS)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(output_path, facecolor="white")
    plt.close(fig)
    return output_path


def _has_succ(payload: dict) -> bool:
    keys = [key for _, mean_key, std_key, _, _, _ in METRICS for key in (mean_key, std_key)]
    return all(key in payload for key in keys)


def build_arg_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="Plot two eval JSON metric summaries (mean ± std).")
    parser.add_argument("left_json", type=Path, help="Left bar JSON (baseline).")
    parser.add_argument("right_json", type=Path, help="Right bar JSON (comparison).")
    parser.add_argument("--output", type=Path, default=None)
    parser.add_argument("--left-label", default="Base motion")
    parser.add_argument("--right-label", default="Base+delta")
    parser.add_argument("--title", default="Base motion vs base+delta (mean ± std)")
    parser.add_argument(
        "--succ-only",
        action="store_true",
        help="Require *_succ_* keys. Default uses them when both files have them.",
    )
    return parser


def main(argv: list[str] | None = None) -> int:
    args = build_arg_parser().parse_args(argv)
    left_path = args.left_json.expanduser().resolve()
    right_path = args.right_json.expanduser().resolve()
    left = _load_json(left_path)
    right = _load_json(right_path)
    use_succ = args.succ_only or (_has_succ(left) and _has_succ(right))
    if args.succ_only and not use_succ:
        raise SystemExit("Requested --succ-only but *_succ_* keys are missing.")
    output = (
        args.output if args.output is not None else right_path.with_name("base_vs_base_plus_delta_metrics.png")
    ).expanduser().resolve()
    saved = plot_metric_bars(
        left,
        right,
        output_path=output,
        left_label=args.left_label,
        right_label=args.right_label,
        figure_title=args.title,
        use_succ=use_succ,
    )
    print(f"[INFO] use_succ={use_succ}")
    print(f"[INFO] Saved {saved}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
