#!/usr/bin/env python3
"""Gap-model uncertainty u along closed-loop rollouts (Isaac batch-eval rollout.npz or Genesis --record_motion NPZ).

u is evaluated every --interval_s at clip time t on (state at t, base action applied at t), using the same offline
input rebuild as gap_uncertainty_vs_horizon.py (pelvis COM-velocity correction fitted from the gap dataset). Isaac
archives store post-action states (index k = frame k+1, frame 0 = initial_*); Genesis archives store pre-action
states (index k = frame k). Frames at or after a rollout's failure are excluded.
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

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT / "source/whole_body_tracking"))
sys.path.insert(0, str(Path(__file__).resolve().parent))
from whole_body_tracking.utils.gap_model import GapEnsemble  # noqa: E402
from gap_uncertainty_vs_horizon import build_inputs, fit_com_offset, rotate_inverse  # noqa: E402
from eval_delta_replay_isaac import load_dataset  # noqa: E402

STATE = ("joint_pos", "joint_vel", "body_pos_w", "body_quat_w", "body_lin_vel_w", "body_ang_vel_w")
COLORS = ["#2a78d6", "#6b7280", "#eb6834", "#1baf7a"]


def frames_of(path: Path, kind: str):
    """Per-frame states [N, T, ...] aligned so index k is the state at clip frame k, actions[k] applied there."""
    d = np.load(path, allow_pickle=True)
    if kind == "isaac":
        origins = d["env_origins"][:, None, None, :]
        s = {}
        for k in STATE:
            arr, ini = d[k], d["initial_" + k]
            if k == "body_pos_w":
                arr, ini = arr - origins, ini - d["env_origins"][:, None, :]
            s[k] = np.concatenate([ini[:, None], arr[:, :-1]], 1)
        valid = d["valid_lengths"].astype(int)
    else:
        s = {k: d[k] for k in STATE}
        valid = d["valid_lengths"].astype(int)
    return s, d["actions"] if "actions" in d.files else d["action"], valid, list(map(str, d["joint_names"])), \
        list(map(str, d["body_names"])), d["default_joint_pos"]


def main():
    p = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    p.add_argument("--model", type=Path, required=True)
    p.add_argument("--gap_dataset", type=Path, required=True, help="One-step gap dataset (for joint order / COM fit)")
    p.add_argument("--gap_source_dataset", type=Path, required=True, help="Genesis dataset the gap dataset was built from")
    p.add_argument("--rollouts", nargs="+", required=True, help="name=kind:path with kind isaac|genesis")
    p.add_argument("--interval_s", type=float, default=0.25)
    p.add_argument("--output_dir", type=Path, required=True)
    p.add_argument("--device", default="cuda:0")
    args = p.parse_args()
    args.output_dir.mkdir(parents=True, exist_ok=True)
    model = GapEnsemble.load(args.model).to(args.device)
    gap = np.load(args.gap_dataset, allow_pickle=True)
    sim_names = [n[len("joint_vel["):-1] for n in map(str, gap["y_names"]) if n.startswith("joint_vel[")]
    src = load_dataset(args.gap_source_dataset)
    root_src = list(map(str, src["body_names"])).index("pelvis")
    ok = gap["valid"]; t, f = gap["trajectory"][ok][:5000], gap["frame"][ok][:5000]
    q = src["body_quat_w"][t, f, root_src]
    lin_cols = [i for i, n in enumerate(map(str, gap["x_names"])) if n.startswith("base_lin_vel[")]
    com = fit_com_offset(rotate_inverse(q, src["body_lin_vel_w"][t, f, root_src]),
                         rotate_inverse(q, src["body_ang_vel_w"][t, f, root_src]), gap["x"][ok][:5000][:, lin_cols] / 2.0)

    def u_of(x):
        x = torch.as_tensor(x, dtype=torch.float32, device=args.device)
        return torch.cat([model.predict(x[i:i + 65536])["u"] for i in range(0, len(x), 65536)]).cpu().numpy()

    results = {}
    for item in args.rollouts:
        name, spec = item.split("=", 1)
        kind, path = spec.split(":", 1)
        s, actions, valid, jn, bn, default = frames_of(Path(path), kind)
        sim_from = np.array([jn.index(n) for n in sim_names]); root = bn.index("pelvis")
        fps = 50.0
        times = np.arange(0, actions.shape[1], int(round(args.interval_s * fps)))
        U = np.full((len(valid), len(times)), np.nan)
        for j, k in enumerate(times):
            alive = np.flatnonzero(valid > k)
            if len(alive) == 0:
                continue
            x = build_inputs(s["joint_pos"][alive, k], s["joint_vel"][alive, k], s["body_pos_w"][alive, k, root],
                             s["body_quat_w"][alive, k, root], s["body_lin_vel_w"][alive, k, root],
                             s["body_ang_vel_w"][alive, k, root], actions[alive, k], sim_from, default[sim_from], com)
            U[alive, j] = u_of(x)
        results[name] = dict(t=times / fps, u=U, alive=np.isfinite(U))
    summary = {n: dict(times_s=r["t"].tolist(), u_median=np.nanmedian(r["u"], 0).tolist(),
                       u_p25=np.nanpercentile(r["u"], 25, 0).tolist(), u_p75=np.nanpercentile(r["u"], 75, 0).tolist(),
                       alive_fraction=r["alive"].mean(0).tolist()) for n, r in results.items()}
    summary["pelvis_com_offset_b"] = com.round(5).tolist()
    (args.output_dir / "u_closed_loop.json").write_text(json.dumps(summary, indent=1))

    fig, axes = plt.subplots(1, 2, figsize=(15, 4.8), constrained_layout=True, gridspec_kw=dict(width_ratios=[2, 1]))
    for (n, r), c in zip(results.items(), COLORS):
        keep = r["alive"].mean(0) >= 0.1  # only plot while >= 10% of rollouts are still running
        med = np.where(keep, np.nanmedian(r["u"], 0), np.nan)
        axes[0].fill_between(r["t"], np.where(keep, np.nanpercentile(r["u"], 25, 0), np.nan),
                             np.where(keep, np.nanpercentile(r["u"], 75, 0), np.nan), color=c, alpha=.15, lw=0)
        axes[0].plot(r["t"], med, "-", lw=2, color=c, label=n)
        axes[1].plot(r["t"], 100 * r["alive"].mean(0), "-", lw=2, color=c, label=n)
    axes[0].axvspan(0, 5, color="k", alpha=.04, lw=0); axes[0].text(2.5, .98, "first 5 s", transform=axes[0].get_xaxis_transform(),
                                                                   ha="center", va="top", fontsize=9, color="#52514e")
    axes[0].set_yscale("log"); axes[0].set_ylabel("u (log scale, target std units)")
    axes[0].set_title("Gap-model uncertainty u along closed-loop rollouts (median, IQR; shown while ≥10% alive)")
    axes[1].set_ylabel("% rollouts not yet failed"); axes[1].set_ylim(0, 102); axes[1].set_title("Rollouts still running")
    for ax in axes:
        ax.set_xlabel("Clip time (s)"); ax.grid(alpha=.3, which="both"); ax.legend(fontsize=8)
    fig.suptitle("m4, 100 paired closed-loop rollouts from identical start states; u every 0.25 s")
    fig.savefig(args.output_dir / "u_closed_loop.png", dpi=130)
    for n, r in summary.items():
        if n == "pelvis_com_offset_b":
            continue
        print(n, " ".join(f"{t:g}s:{u:.3f}({a:.0%})" for t, u, a in zip(r["times_s"][::8], r["u_median"][::8], r["alive_fraction"][::8])))


if __name__ == "__main__":
    main()
