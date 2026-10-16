#!/usr/bin/env python3
"""Plot decreasing cumulative trajectory support from valid recorded lengths."""
import argparse
import csv
import json
from pathlib import Path

import numpy as np


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--dataset", type=Path, required=True)
    parser.add_argument("--output_dir", type=Path, required=True)
    parser.add_argument("--max_time_s", type=float, default=20.0)
    args = parser.parse_args()
    with np.load(args.dataset, allow_pickle=False) as archive:
        fps = float(np.asarray(archive["fps"]).reshape(-1)[0])
        lengths = np.asarray(archive["valid_lengths"], dtype=np.int64).reshape(-1)
        start = int(np.asarray(archive["start_timestep"]).reshape(-1)[0]) if "start_timestep" in archive else 0
    if not np.isfinite(fps) or fps <= 0 or lengths.size == 0 or np.any(lengths <= 0):
        raise ValueError("Expected positive FPS and nonempty positive valid_lengths")
    if start != 0:
        raise ValueError("Reference-time support requires trajectories starting at frame zero")
    if not np.isfinite(args.max_time_s) or args.max_time_s <= 0:
        raise ValueError("max_time_s must be finite and positive")

    durations = lengths / fps
    # Duplicate each event time to show the exact drop immediately after L.
    # At the final reference endpoint retain trajectories with L == max_time_s.
    events = np.unique(durations[durations < args.max_time_s])
    x, counts = [0.0], [len(lengths)]
    for event in events:
        x.extend([float(event), float(event)])
        counts.extend([int(np.sum(durations >= event)), int(np.sum(durations > event))])
    x.append(args.max_time_s)
    counts.append(int(np.sum(durations >= args.max_time_s)))
    x, counts = np.asarray(x), np.asarray(counts)
    percentages = 100.0 * counts / len(lengths)
    out = args.output_dir
    out.mkdir(parents=True, exist_ok=True)

    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    plt.rcParams.update({"font.size": 16, "axes.titlesize": 19,
                         "axes.labelsize": 17, "pdf.fonttype": 42,
                         "axes.spines.top": False, "axes.spines.right": False})
    fig, ax = plt.subplots(figsize=(10, 5.8))
    color = "#237b9a"
    ax.fill_between(x, percentages, color=color, alpha=0.13)
    ax.plot(x, percentages, color=color, linewidth=2.7)
    for time in np.arange(5.0, args.max_time_s + 1e-9, 5.0):
        n = int(np.sum(durations >= time))
        y = 100.0 * n / len(lengths)
        ax.plot(time, y, "o", color=color, markersize=6, clip_on=False)
        ax.annotate(f"{y:.0f}%", (time, y), xytext=(-7 if time == args.max_time_s else 0, 11),
                    textcoords="offset points", ha="right" if time == args.max_time_s else "center",
                    fontsize=15, color=color, fontweight="semibold")
    ax.text(0.97, 0.94, f"{len(lengths)} trajectories\n$S(t)=P(L\\geq t)$", transform=ax.transAxes,
            ha="right", va="top", fontsize=15)
    ax.set(xlim=(0, args.max_time_s), ylim=(0, 105), xlabel="Clip time (s)",
           ylabel="Trajectories reaching time (%)", title="Dance training data: trajectory support")
    ax.set_xticks(np.arange(0, args.max_time_s + 1e-9, 2.0))
    ax.set_yticks(np.arange(0, 101, 20))
    ax.grid(alpha=0.2)
    ax.set_axisbelow(True)
    fig.text(0.5, 0.025, "Support for an interval [a, b]: read the curve at b; all trajectories start at 0 s.",
             ha="center", fontsize=12.5, color="#444444")
    fig.tight_layout(rect=(0, 0.065, 1, 1))
    for suffix in ("png", "pdf"):
        fig.savefig(out / f"trajectory_support.{suffix}", dpi=220)
    plt.close(fig)

    times = np.arange(int(np.floor(args.max_time_s * fps)) + 1) / fps
    with (out / "trajectory_support.csv").open("w", newline="") as handle:
        writer = csv.writer(handle)
        writer.writerow(["clip_time_s", "trajectories_reaching_time", "fraction"])
        for time in times:
            n = int(np.sum(durations >= time))
            writer.writerow([float(time), n, n / len(lengths)])
    summary = dict(dataset=str(args.dataset.resolve()), fps=fps, num_trajectories=len(lengths),
                   min_duration_s=float(durations.min()), median_duration_s=float(np.median(durations)),
                   mean_duration_s=float(durations.mean()), max_duration_s=float(durations.max()),
                   total_valid_frames=int(lengths.sum()),
                   method="S(t) = fraction with valid_lengths/fps >= t; padding excluded. All trajectories start at frame 0.")
    (out / "summary.json").write_text(json.dumps(summary, indent=2) + "\n")
    print(out / "trajectory_support.png")


if __name__ == "__main__":
    main()
