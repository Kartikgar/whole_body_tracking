#!/usr/bin/env python3
"""Validate a gap-model ensemble: accuracy, distribution-shift detection and error ranking of its uncertainty.

Datasets are one-step gap NPZs from build_gap_dataset_isaac.py, passed as name=path. The in-distribution reference
is the model's own validation trajectories of its training dataset (recorded in model.pt).
"""
from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import torch
from scipy.stats import mannwhitneyu, spearmanr

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT / "source/whole_body_tracking"))
from whole_body_tracking.utils.gap_model import GapEnsemble  # noqa: E402

COLORS = ["#2a78d6", "#eb6834", "#1baf7a", "#eda100", "#e87ba4"]


def auroc(negative, positive):
    """P(score_pos > score_neg) via the Mann-Whitney U statistic."""
    return float(mannwhitneyu(positive, negative).statistic / (len(positive) * len(negative)))


def evaluate(model, path, device, trajectories=None):
    d = np.load(path, allow_pickle=True)
    keep = d["valid"] & (np.isin(d["trajectory"], trajectories) if trajectories is not None else True)
    x = torch.as_tensor(d["x"][keep], device=device); y = torch.as_tensor(d["y"][keep], device=device)
    outs = [model.predict(x[i:i + 65536]) for i in range(0, len(x), 65536)]
    pred = {k: torch.cat([o[k] for o in outs]) for k in outs[0]}
    yn = model.normalize_y(y)
    err = (pred["mean_std_units"] - yn) ** 2
    groups = json.loads(str(d["target_groups"]))
    bounds = np.cumsum([0, *groups.values()])
    per_group = {}
    for (g, _), s, e in zip(groups.items(), bounds[:-1], bounds[1:]):
        resid = (pred["mean"][:, s:e] - y[:, s:e]) ** 2
        per_group[g] = dict(rmse=float(resid.mean().sqrt()), zero_rmse=float((y[:, s:e] ** 2).mean().sqrt()),
                            r2=float(1 - resid.mean() / y[:, s:e].var(0).mean().clamp_min(1e-12)))
    return dict(u=pred["u"].cpu().numpy(), err=err.mean(-1).sqrt().cpu().numpy(),
                z2=(err / pred["total"]).mean(-1).cpu().numpy(), per_group=per_group, n=int(keep.sum()))


def main():
    p = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    p.add_argument("--model", type=Path, required=True)
    p.add_argument("--train_dataset", type=Path, required=True, help="Dataset the model was trained on (validation split used)")
    p.add_argument("--datasets", nargs="+", required=True, help="name=path one-step gap datasets to evaluate")
    p.add_argument("--output_dir", type=Path, required=True)
    p.add_argument("--device", default="cuda:0")
    args = p.parse_args()
    args.output_dir.mkdir(parents=True, exist_ok=True)
    payload = torch.load(args.model, map_location="cpu", weights_only=False)
    model = GapEnsemble.load(args.model).to(args.device)
    results = {"in_dist_val": evaluate(model, args.train_dataset, args.device, payload["extra"]["val_trajectories"])}
    for item in args.datasets:
        name, path = item.split("=", 1)
        results[name] = evaluate(model, Path(path), args.device)

    ref = results["in_dist_val"]
    summary = {}
    for name, r in results.items():
        order = np.argsort(r["u"])
        keep = [1.0, 0.75, 0.5, 0.25]
        summary[name] = dict(
            n=r["n"], per_group=r["per_group"], u_median=float(np.median(r["u"])), u_p90=float(np.percentile(r["u"], 90)),
            err_median_std_units=float(np.median(r["err"])),
            auroc_vs_in_dist=None if name == "in_dist_val" else auroc(ref["u"], r["u"]),
            spearman_u_vs_error=float(spearmanr(r["u"], r["err"]).statistic),
            mean_err_when_keeping_most_confident={f"{int(f * 100)}%": float(r["err"][order[:max(1, int(f * len(order)))]].mean()) for f in keep},
            mean_standardized_sq_residual=float(np.mean(r["z2"])))
    (args.output_dir / "summary.json").write_text(json.dumps(summary, indent=2))

    fig, axes = plt.subplots(1, 3, figsize=(17, 4.6), constrained_layout=True)
    hi = np.percentile(np.concatenate([r["u"] for r in results.values()]), 99.5)
    bins = np.linspace(0, hi, 60)
    for (name, r), c in zip(results.items(), COLORS):
        axes[0].hist(r["u"], bins=bins, density=True, histtype="step", lw=2, color=c,
                     label=f"{name} (median {np.median(r['u']):.3f})")
        order = np.argsort(r["u"]); fr = np.linspace(0.05, 1, 20)
        axes[1].plot(fr * 100, [r["err"][order[:int(f * len(order))]].mean() for f in fr], "-o", ms=3, color=c, label=name)
        q = np.quantile(r["u"], np.linspace(0, 1, 11)); idx = np.clip(np.digitize(r["u"], q[1:-1]), 0, 9)
        axes[2].plot([r["u"][idx == b].mean() for b in range(10)], [r["err"][idx == b].mean() for b in range(10)],
                     "-o", ms=4, color=c, label=name)
    axes[0].set_title("Epistemic uncertainty u (ensemble disagreement)"); axes[0].set_xlabel("u (target std units)")
    axes[1].set_title("Error when keeping the most confident fraction"); axes[1].set_xlabel("% of transitions kept")
    axes[1].set_ylabel("mean prediction error (std units)")
    axes[2].set_title("Error vs uncertainty (deciles of u)"); axes[2].set_xlabel("mean u in decile")
    axes[2].set_ylabel("mean prediction error (std units)")
    for ax in axes:
        ax.grid(alpha=.3); ax.legend(fontsize=8)
    fig.suptitle(f"Gap-model ensemble {args.model.parent.name}: one-step Isaac→Genesis Δ")
    fig.savefig(args.output_dir / "uncertainty_validation.png", dpi=130)

    for name, s in summary.items():
        g = s["per_group"]
        print(f"{name:18s} n={s['n']:6d} u_med={s['u_median']:.3f} AUROC={s['auroc_vs_in_dist'] if s['auroc_vs_in_dist'] is None else round(s['auroc_vs_in_dist'], 3)} "
              f"rho(u,err)={s['spearman_u_vs_error']:.3f} | R2 jv={g['joint_vel']['r2']:.2f} angv={g['root_ang_vel_b']['r2']:.2f} "
              f"linv={g['root_lin_vel_b']['r2']:.2f} | err keep100/25%={s['mean_err_when_keeping_most_confident']['100%']:.3f}/"
              f"{s['mean_err_when_keeping_most_confident']['25%']:.3f}")


if __name__ == "__main__":
    main()
