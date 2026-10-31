#!/usr/bin/env python3
"""Step 3.4 go/no-go: does gap-model uncertainty predict when the delta is worse than zero delta?

For each replay window (trajectory, start) the gap ensemble's epistemic uncertainty u is evaluated on the logged
Genesis states of that window (teacher-forced inputs from the one-step gap dataset): at the start frame and as the
mean over the first k frames. The outcome is "delta loses" = delta error > zero-delta error at a horizon, from paired
open-loop replays. AUROC(u -> delta loses) is reported with a trajectory-bootstrap 95% CI.
"""
from __future__ import annotations

import argparse
import ast
import csv
import json
import sys
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import torch
from scipy.stats import mannwhitneyu

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT / "source/whole_body_tracking"))
from whole_body_tracking.utils.gap_model import GapEnsemble  # noqa: E402

METRICS = ("body_pos_error_m", "joint_pos_rmse_rad", "body_orientation_error_rad")
U_FEATURES = {"u_start": 1, "u_mean_0.25s": 13, "u_mean_0.5s": 25}


def auroc(score, label):
    pos, neg = score[label], score[~label]
    if len(pos) == 0 or len(neg) == 0:
        return float("nan")
    return float(mannwhitneyu(pos, neg).statistic / (len(pos) * len(neg)))


def load_windows(path):
    rows = {}
    with open(path / "windows.csv") as f:
        for r in csv.DictReader(f):
            if r["status"] == "complete":
                rows[(int(r["trajectory"]), int(r["start"]))] = {c["requested_time_s"]: c for c in ast.literal_eval(r["error_curve"])}
    return rows


def frame_uncertainty(model, gap_path, device):
    d = np.load(gap_path, allow_pickle=True)
    x = torch.as_tensor(d["x"], device=device)
    u = torch.cat([model.predict(x[i:i + 65536])["u"] for i in range(0, len(x), 65536)]).cpu().numpy()
    u[~d["valid"]] = np.nan
    return {(int(t), int(f)): v for t, f, v in zip(d["trajectory"], d["frame"], u)}


def analyse(name, model, gap_path, zero_dir, delta_dir, horizons, device, rng):
    u_frame = frame_uncertainty(model, gap_path, device)
    zero, delta = load_windows(zero_dir), load_windows(delta_dir)
    keys = sorted(set(zero) & set(delta))
    feats = {f: np.array([np.nanmean([u_frame.get((t, s + i), np.nan) for i in range(k)]) for t, s in keys])
             for f, k in U_FEATURES.items()}
    traj = np.array([t for t, _ in keys])
    out = dict(windows=len(keys), trajectories=int(len(np.unique(traj))), results={})
    for h in horizons:
        for m in METRICS:
            lose = np.array([delta[k][h][m] > zero[k][h][m] for k in keys])
            cell = dict(delta_loss_rate=float(lose.mean()))
            for f, u in feats.items():
                ok = np.isfinite(u)
                point = auroc(u[ok], lose[ok])
                uniq = np.unique(traj[ok]); boots = []
                for _ in range(500):
                    pick = rng.choice(uniq, len(uniq), replace=True)
                    idx = np.concatenate([np.flatnonzero((traj == t) & ok) for t in pick])
                    boots.append(auroc(u[idx], lose[idx]))
                cell[f] = dict(auroc=point, ci95=np.nanpercentile(boots, [2.5, 97.5]).tolist())
                # Gate preview: delta win rate among the most confident half / quarter of windows.
                order = np.argsort(u[ok])
                cell[f]["win_rate_keep_50%"] = float(1 - lose[ok][order[: len(order) // 2]].mean())
                cell[f]["win_rate_keep_25%"] = float(1 - lose[ok][order[: len(order) // 4]].mean())
            out["results"][f"{h:g}s/{m}"] = cell
    return out, feats, keys, zero, delta


def main():
    p = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    p.add_argument("--model", type=Path, required=True)
    p.add_argument("--set", nargs=4, action="append", metavar=("NAME", "GAP_NPZ", "ZERO_DIR", "DELTA_DIR"), required=True)
    p.add_argument("--horizons", type=float, nargs="+", default=[0.25, 0.5, 1.0, 1.5])
    p.add_argument("--output_dir", type=Path, required=True)
    p.add_argument("--device", default="cuda:0")
    args = p.parse_args()
    args.output_dir.mkdir(parents=True, exist_ok=True)
    model = GapEnsemble.load(args.model).to(args.device)
    rng = np.random.default_rng(0)
    summary, cache = {}, {}
    for name, gap, zero_dir, delta_dir in args.set:
        summary[name], *cache[name] = analyse(name, model, Path(gap), Path(zero_dir), Path(delta_dir),
                                              args.horizons, args.device, rng)
    (args.output_dir / "harm_auroc.json").write_text(json.dumps(summary, indent=2))

    fig, axes = plt.subplots(1, len(METRICS), figsize=(16, 4.4), constrained_layout=True, sharey=True)
    for ax, m in zip(axes, METRICS):
        for (name, s), ls in zip(summary.items(), ("-", "--")):
            for f, c in zip(U_FEATURES, ("#2a78d6", "#eb6834", "#1baf7a")):
                vals = [s["results"][f"{h:g}s/{m}"][f] for h in args.horizons]
                ax.errorbar(args.horizons, [v["auroc"] for v in vals],
                            yerr=np.abs(np.array([v["ci95"] for v in vals]).T - [v["auroc"] for v in vals]),
                            ls=ls, marker="o", ms=4, capsize=3, color=c, label=f"{name}: {f}")
        ax.axhline(.5, color="k", lw=1, ls=":"); ax.set_ylim(.2, .9); ax.grid(alpha=.3)
        ax.set_title(f"AUROC(u → delta loses), {m}"); ax.set_xlabel("Horizon (s)")
    axes[0].set_ylabel("AUROC (0.5 = no signal)"); axes[-1].legend(fontsize=7)
    fig.savefig(args.output_dir / "harm_auroc.png", dpi=130)

    for name, s in summary.items():
        print(f"== {name}: {s['windows']} windows, {s['trajectories']} trajectories")
        for key, cell in s["results"].items():
            parts = " | ".join(f"{f} {cell[f]['auroc']:.2f} [{cell[f]['ci95'][0]:.2f},{cell[f]['ci95'][1]:.2f}]" for f in U_FEATURES)
            print(f"  {key:32s} loss {cell['delta_loss_rate']:.2f} | {parts} | win keep50/25 (u_mean_0.25s) "
                  f"{cell['u_mean_0.25s']['win_rate_keep_50%']:.2f}/{cell['u_mean_0.25s']['win_rate_keep_25%']:.2f}")


if __name__ == "__main__":
    main()
