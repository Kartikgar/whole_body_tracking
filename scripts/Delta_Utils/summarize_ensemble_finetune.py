#!/usr/bin/env python3
"""Aggregate a finetuning sweep and plot Genesis results versus gating lambda."""
from __future__ import annotations

import argparse
import csv
import json
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np


def summarize(directory):
    directory = Path(directory).resolve()
    plan = json.loads((directory / "plan.json").read_text())
    rows = []
    for stage in plan["stages"]:
        if stage["kind"] != "genesis":
            continue
        marker = directory / "stages" / (stage["id"].replace("/", "__") + ".json")
        if not marker.is_file():
            continue
        record = json.loads(marker.read_text())
        result = json.loads(Path(record["result_file"]).read_text())
        rollouts = result["rollout_metrics"]
        completed = [r for r in rollouts if r["completed"]]
        row = dict(label=stage["label"], lambda_value=stage["lambda_value"],
                   finetune_iteration=stage["finetune_iteration"], seed=stage["seed"],
                   num_rollouts=len(rollouts), completion_pct=result["full_clip_completion_pct"],
                   success_pct=result["success"], survival_pct=result["clip_survival_pct"],
                   wall_seconds=record["seconds"], completed_rollouts=len(completed))
        for metric in ("mpjpe", "mpjpe_l", "vel_dist", "accel_dist"):
            values = np.array([r[metric] for r in completed], dtype=float)
            row[f"{metric}_completed_mean"] = float(values.mean()) if len(values) else float("nan")
            row[f"{metric}_all_mean"] = result[f"{metric}_mean"]
        rows.append(row)
    if not rows:
        raise ValueError("No completed Genesis evaluations found")
    analysis = directory / "analysis"
    analysis.mkdir(exist_ok=True)
    with (analysis / "results.csv").open("w", newline="") as stream:
        writer = csv.DictWriter(stream, fieldnames=list(rows[0]))
        writer.writeheader()
        writer.writerows(rows)
    plt.rcParams.update({"font.size": 15, "axes.titlesize": 17, "axes.labelsize": 15,
                         "xtick.labelsize": 13, "ytick.labelsize": 13,
                         "pdf.fonttype": 42, "ps.fonttype": 42})
    fig, axes = plt.subplots(2, 3, figsize=(16, 9), layout="constrained")
    panels = [("completion_pct", "Full-clip completion", "%"), ("survival_pct", "Clip survival", "%"),
              ("mpjpe_completed_mean", "World body position error", "mm"),
              ("mpjpe_l_completed_mean", "Local body position error", "mm"),
              ("vel_dist_completed_mean", "Body velocity distance", "mm/frame"),
              ("accel_dist_completed_mean", "Body acceleration distance", "mm/frame²")]
    ages = plan["config"]["export_iterations"]
    lambdas = plan["config"]["lambdas"]
    def stats(selected, key):
        values = np.array([r[key] for r in selected], dtype=float)
        values = values[np.isfinite(values)]
        return (float(values.mean()), float(values.std(ddof=1)) if len(values) > 1 else 0) if len(values) else (np.nan, np.nan)
    for ax, (key, title, unit) in zip(axes.flat, panels):
        for age, color in zip(ages, plt.cm.viridis(np.linspace(.1, .8, len(ages)))):
            means, spreads = [], []
            for lam in lambdas:
                selected = [r for r in rows if r["lambda_value"] == lam and r["finetune_iteration"] == age]
                mean, spread = stats(selected, key)
                means.append(mean)
                spreads.append(spread)
            ax.errorbar(range(len(lambdas)), means, yerr=spreads, marker="o", capsize=3,
                        color=color, label=f"PPO iteration {age}")
        for label, style, color, text in (("original_base", "--", "0.35", "Original base"),
                                         ("no_delta_finetune", ":", "#b74b4b", "No-delta finetune")):
            mean, spread = stats([r for r in rows if r["label"] == label], key)
            if np.isfinite(mean):
                ax.axhline(mean, linestyle=style, color=color, label=text)
                if spread:
                    ax.axhspan(mean-spread, mean+spread, color=color, alpha=.08)
        ax.set(title=title, ylabel=unit, xlabel="Gating λ", xticks=range(len(lambdas)), xticklabels=[f"{x:g}" for x in lambdas])
        ax.spines[["top", "right"]].set_visible(False)
        ax.grid(axis="y", alpha=.2)
        if key in ("completion_pct", "survival_pct"):
            ax.set_ylim(0, 105)
    handles, labels = axes[0, 0].get_legend_handles_labels()
    fig.legend(handles, labels, loc="outside upper center", ncol=3, frameon=False)
    name = "smoke_pipeline_metrics" if plan["config"].get("smoke") else "genesis_gate_sweep"
    for extension in ("png", "pdf"):
        fig.savefig(analysis / f"{name}.{extension}", dpi=200)
    plt.close(fig)
    (analysis / "README.md").write_text(
        "# Gating sweep results\n\n"
        "Tracking panels use only strictly completed rollouts; absent completions produce gaps. "
        "Whiskers show one standard deviation of the seed-level means, not a confidence interval. "
        "Each seed is weighted equally. Survival measures the fraction of the evaluated horizon before termination. "
        "The CSV also preserves all-rollout tracking errors and the existing five-step-tolerance success metric. "
        "Worker wall time includes startup/compilation and is recorded in results.csv. "
        "Smoke results use a short horizon and do not establish clip-level performance.\n")
    print(f"Aggregated {len(rows)} seed-level evaluations: {analysis}")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--sweep_dir", required=True)
    summarize(parser.parse_args().sweep_dir)
