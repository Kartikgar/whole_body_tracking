#!/usr/bin/env python3
"""Rank checkpoint batch clips by tracking difficulty and safety, then plot both.

Tracking errors are averaged over full-clip rollouts only. Safety values use the
90th percentile across all rollouts, with contact time divided by each rollout's
observed duration. Safety ranks put sustained non-support contact last. Missing
successful tracking metrics are shown as a dash.
"""

from __future__ import annotations

import argparse
import bisect
import csv
import json
import math
import re
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np


TRACKING = ("mpjpe", "mpjpe_l", "vel_dist", "accel_dist")
SAFETY = {
    "contact_p90": "safety_non_support_contact_time_s",
    "hard_limit_p90": "safety_joint_position_violation_fraction",
    "soft_limit_p90": "safety_joint_soft_position_violation_fraction",
    "effort_p90": "safety_actuator_effort_near_limit_fraction",
    "slip_p90": "safety_foot_slip_time_fraction",
}
SAFETY_WEIGHTS = {"contact_p90": 0.50, "soft_limit_p90": 0.30, "effort_p90": 0.20}
CLIP_PATTERN = re.compile(r"(?P<parent>.+)__(?P<start>\d+(?:\.\d+)?)s-(?P<end>\d+(?:\.\d+)?)s")


def finite(values):
    return [float(value) for value in values if value is not None and math.isfinite(float(value))]


def clip_label(name: str) -> tuple[str, str, float]:
    stem = name.split("_", 1)[1] if re.match(r"\d+_", name) else name
    match = CLIP_PATTERN.fullmatch(stem)
    if match is None:
        return stem, stem, float("nan")
    parent = match.group("parent")
    start, end = float(match.group("start")), float(match.group("end"))
    head = re.match(r"(.+)_subject(\d+)$", parent)
    motion_type = head.group(1).replace("jumps", "jump") if head else ""
    short_parent = f"{motion_type.title()} S{head.group(2)}" if head else parent
    short = f"{short_parent}  {start:g}–{end:g}s"
    if end - start < 19.99:
        short += " *"
    return short, parent, start


def load_rows(run_dir: Path) -> list[dict]:
    with (run_dir / "summary.csv").open(newline="", encoding="utf-8") as stream:
        summaries = list(csv.DictReader(stream))
    if not summaries:
        raise ValueError("summary.csv has no clip rows")
    result = []
    for summary in summaries:
        name = summary["motion"]
        json_path = run_dir / f"{name}.json"
        data = json.loads(json_path.read_text(encoding="utf-8"))
        rollouts = data["rollouts"]
        if not rollouts:
            raise ValueError(f"{json_path} has no rollouts")
        duration = float(data["reference_duration_s"])
        if duration <= 0:
            raise ValueError(f"{json_path} has invalid reference duration")
        successful = [item for item in rollouts if not item["terminated"]]
        label, parent, start = clip_label(name)
        row = {
            "motion": name,
            "label": label,
            "parent": parent,
            "start_s": start,
            "reference_duration_s": duration,
            "num_rollouts": len(rollouts),
            "completed_rollouts": len(successful),
            "completion_rate": len(successful) / len(rollouts),
            "survival_fraction": float(np.mean([
                min(float(item["completed_duration_s"]) / duration, 1.0) for item in rollouts
            ])),
        }
        for key in TRACKING:
            values = finite(item["metrics"].get(key) for item in successful)
            row[f"{key}_success_mean"] = float(np.mean(values)) if values else float("nan")
        for output_key, metric_key in SAFETY.items():
            if output_key == "contact_p90":
                values = finite(
                    float(item["metrics"][metric_key]) / float(item["completed_duration_s"])
                    for item in rollouts if float(item["completed_duration_s"]) > 0
                )
            else:
                values = finite(item["metrics"].get(metric_key) for item in rollouts)
            row[output_key] = float(np.percentile(values, 90)) if values else float("nan")
        result.append(row)
    return result


def percentile_rank(value: float, sorted_values: list[float]) -> float:
    if not math.isfinite(value):
        return 1.0
    lo = bisect.bisect_left(sorted_values, value)
    hi = bisect.bisect_right(sorted_values, value)
    return (lo + hi) / (2 * len(sorted_values))


def rank_rows(rows: list[dict], sustained_contact_threshold: float = 0.10) -> tuple[list[dict], list[dict]]:
    distributions = {
        key: sorted(finite(row[key] for row in rows)) for key in SAFETY_WEIGHTS
    }
    for row in rows:
        contact = row["contact_p90"]
        row["sustained_contact"] = not math.isfinite(contact) or contact >= sustained_contact_threshold
        row["sustained_contact_threshold"] = sustained_contact_threshold
        row["safety_score"] = sum(
            weight * percentile_rank(row[key], distributions[key])
            for key, weight in SAFETY_WEIGHTS.items()
        )
    difficulty = sorted(rows, key=lambda row: (
        -row["completion_rate"], -row["survival_fraction"],
        *(row[f"{key}_success_mean"] if math.isfinite(row[f"{key}_success_mean"]) else math.inf
          for key in TRACKING), row["motion"],
    ))
    safety = sorted(rows, key=lambda row: (
        row["sustained_contact"],
        # Within the sustained-contact group, time in contact remains the first priority.
        row["contact_p90"] if row["sustained_contact"] and math.isfinite(row["contact_p90"])
        else row["safety_score"],
        row["safety_score"] if row["sustained_contact"] else row["contact_p90"],
        row["soft_limit_p90"],
        row["effort_p90"], row["motion"],
    ))
    return difficulty, safety


def write_ranking(path: Path, rows: list[dict]) -> None:
    columns = ["rank", "motion", "label", "parent", "start_s", "reference_duration_s",
               "num_rollouts", "completed_rollouts", "completion_rate", "survival_fraction",
               *(f"{key}_success_mean" for key in TRACKING), *SAFETY,
               "sustained_contact", "sustained_contact_threshold", "safety_score"]
    with path.open("w", newline="", encoding="utf-8") as stream:
        writer = csv.DictWriter(stream, fieldnames=columns)
        writer.writeheader()
        for rank, row in enumerate(rows, start=1):
            writer.writerow({"rank": rank, **{key: row[key] for key in columns if key != "rank"}})


def plot_rows(rows: list[dict], *, kind: str, path: Path, total: int, page: str,
              dpi: int, sustained_contact_threshold: float) -> None:
    if kind == "difficulty":
        panels = [
            ("completion_rate", "Full-clip completion", "%", "#218D7A", 100),
            ("survival_fraction", "Clip survival", "%", "#6F9883", 100),
            ("mpjpe_success_mean", "Global MPJPE", "mm", "#3C78A6", 1),
            ("mpjpe_l_success_mean", "Local MPJPE", "mm", "#6097BA", 1),
            ("vel_dist_success_mean", "Velocity distance", "mm/frame", "#8465A3", 1),
            ("accel_dist_success_mean", "Acceleration distance", "mm/frame²", "#A385B9", 1),
        ]
        subtitle = "Easy → hard  ·  tracking bars use completed rollouts only  ·  – = no completed rollout"
    else:
        panels = [
            ("contact_p90", "Non-support contact", "% of observed time", "#C65A4A", 100),
            ("soft_limit_p90", "Soft joint limit", "% of samples", "#E0A65D", 100),
            ("effort_p90", "Effort near limit", "% of samples", "#8567A5", 100),
            ("hard_limit_p90", "Hard joint limit", "% of samples", "#D48A45", 100),
            ("slip_p90", "Foot slip proxy", "% of stance", "#4B87B1", 100),
        ]
        subtitle = ("Safe → risky  ·  values are rollout p90  ·  sustained non-support contact "
                    "ranks after lower-contact clips")

    n = len(rows)
    height = max(5.5, 1.7 + 0.35 * n)
    fig, axes = plt.subplots(1, len(panels), figsize=(19 if kind == "safety" else 21, height), sharey=True)
    fig.patch.set_facecolor("white")
    y = np.arange(n)
    for ax, (key, title, unit, color, scale) in zip(axes, panels):
        values = np.array([row[key] * scale for row in rows], dtype=float)
        good = np.isfinite(values)
        maximum = max(1.0 if scale == 100 else 0.0, float(np.nanmax(values)) if good.any() else 0.0)
        limit = (100.0 if key in ("completion_rate", "survival_fraction", "contact_p90", "slip_p90")
                 else maximum * 1.18 if maximum else 1.0)
        ax.barh(y[good], values[good], height=0.71, color=color, alpha=0.93)
        ax.set_xlim(0, limit * 1.14)
        ax.set_title(title, fontsize=14, fontweight="bold", pad=11)
        ax.set_xlabel(unit, fontsize=11)
        ax.set_yticks(y)
        ax.grid(axis="x", color="#E4E9ED", linewidth=0.8)
        ax.set_axisbelow(True)
        ax.spines[["top", "right"]].set_visible(False)
        ax.tick_params(axis="x", labelsize=10)
        ax.tick_params(axis="y", length=0, labelsize=10)
        for index, value in enumerate(values):
            if not math.isfinite(value):
                ax.text(limit * 0.025, index, "–", va="center", fontsize=11, color="#737B83")
            elif n <= 30:
                if scale == 100:
                    label = ("0%" if value == 0 else "<0.01%" if value < 0.01 else
                             f"{value:.0f}%" if value >= 1 else f"{value:.2f}%")
                else:
                    label = f"{value:.1f}"
                ax.text(min(value + limit * 0.012, limit * 1.01), index, label,
                        va="center", fontsize=9.5, color="#2A333B")
    axes[0].set_yticklabels([row["label"] for row in rows], fontsize=10)
    for ax in axes[1:]:
        ax.tick_params(axis="y", labelleft=False)
    axes[0].invert_yaxis()
    fig.suptitle(f"PgS2R-mini checkpoint batch: {kind.title()} ranking ({page}; {total} clips)",
                 fontsize=20, fontweight="bold", y=1 - 0.15 / height)
    fig.text(0.5, 1 - 0.56 / height, subtitle, ha="center", va="top", fontsize=12, color="#4E5964")
    if kind == "safety":
        fig.text(0.5, 0.52 / height,
                 f"Order: contact < {sustained_contact_threshold:.0%} first, then weighted percentile score; "
                 f"contact ≥ {sustained_contact_threshold:.0%} ordered by contact time. "
                 "Score = 0.50 × contact + 0.30 × soft limit + 0.20 × effort percentiles.",
                 ha="center", fontsize=11, color="#394B59")
    fig.text(0.5, 0.23 / height, "* final parent-motion segment is shorter than 20 s",
             ha="center", fontsize=11, color="#4E5964")
    fig.subplots_adjust(left=0.11 if kind == "safety" else 0.12, right=0.985,
                        top=1 - 1.24 / height, bottom=(1.25 if kind == "safety" else 1.05) / height,
                        wspace=0.26)
    fig.savefig(path, dpi=dpi, facecolor="white")
    plt.close(fig)


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--run_dir", type=Path, required=True, help="Batch directory with summary.csv and per-clip JSON files")
    parser.add_argument("--output_dir", type=Path, help="Default: <run_dir>/analysis/rankings")
    parser.add_argument("--min_duration_s", type=float, default=10.0,
                        help="Exclude clips shorter than this duration from plots and rankings (default: 10 s)")
    parser.add_argument("--sustained_contact_threshold", type=float, default=0.10,
                        help="Contact p90 fraction that marks sustained non-support contact (default: 0.10)")
    parser.add_argument("--page_size", type=int, default=24)
    parser.add_argument("--dpi", type=int, default=150)
    args = parser.parse_args()
    if (args.page_size < 1 or args.dpi < 72 or not math.isfinite(args.min_duration_s)
            or args.min_duration_s < 0 or not math.isfinite(args.sustained_contact_threshold)
            or not 0 < args.sustained_contact_threshold <= 1):
        parser.error("page_size must be positive, dpi at least 72, min_duration_s finite and nonnegative, "
                     "and sustained_contact_threshold in (0, 1]")
    run_dir = args.run_dir.resolve()
    output_dir = (args.output_dir or run_dir / "analysis" / "rankings").resolve()
    output_dir.mkdir(parents=True, exist_ok=True)
    all_rows = load_rows(run_dir)
    rows = [row for row in all_rows if row["reference_duration_s"] >= args.min_duration_s]
    if not rows:
        parser.error(f"No clips are at least {args.min_duration_s:g} seconds long")
    difficulty, safety = rank_rows(rows, args.sustained_contact_threshold)
    for kind, ordered in (("difficulty", difficulty), ("safety", safety)):
        write_ranking(output_dir / f"{kind}_ranking.csv", ordered)
        plot_rows(ordered, kind=kind, path=output_dir / f"{kind}_all.png",
                  total=len(rows), page="all", dpi=args.dpi,
                  sustained_contact_threshold=args.sustained_contact_threshold)
        for offset in range(0, len(ordered), args.page_size):
            page = offset // args.page_size + 1
            subset = ordered[offset:offset + args.page_size]
            plot_rows(subset, kind=kind, path=output_dir / f"{kind}_{page:02d}.png",
                      total=len(rows), page=f"ranks {offset + 1}–{offset + len(subset)}", dpi=args.dpi,
                      sustained_contact_threshold=args.sustained_contact_threshold)
    print(f"Wrote rankings and plots for {len(rows)} clips to {output_dir} "
          f"({len(all_rows) - len(rows)} shorter than {args.min_duration_s:g} s excluded)")


if __name__ == "__main__":
    main()
