#!/usr/bin/env python3
"""Train a deep ensemble of gap models g(s, a) -> Δs on a one-step gap dataset (build_gap_dataset_isaac.py).

The train/validation split and each member's bootstrap resample are drawn over trajectories, not frames, so
correlated neighbouring frames never straddle the split. Members share the normalization statistics of the
training split and are early-stopped on validation Student-t NLL.
"""
from __future__ import annotations

import argparse
import json
import sys
import time
from pathlib import Path

import numpy as np
import torch

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT / "source/whole_body_tracking"))
from whole_body_tracking.utils.gap_model import GapEnsemble, student_t_nll  # noqa: E402


def parse_args():
    p = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    p.add_argument("--dataset", type=Path, required=True)
    p.add_argument("--output_dir", type=Path, required=True)
    p.add_argument("--members", type=int, default=5)
    p.add_argument("--hidden", type=int, nargs="+", default=[256, 256, 256])
    p.add_argument("--dof", type=float, default=4.0, help="Student-t degrees of freedom")
    p.add_argument("--val_fraction", type=float, default=0.1)
    p.add_argument("--epochs", type=int, default=200)
    p.add_argument("--patience", type=int, default=15)
    p.add_argument("--batch_size", type=int, default=4096)
    p.add_argument("--lr", type=float, default=1e-3)
    p.add_argument("--weight_decay", type=float, default=1e-4)
    p.add_argument("--seed", type=int, default=0)
    p.add_argument("--device", default="cuda:0")
    return p.parse_args()


def main():
    args = parse_args()
    args.output_dir.mkdir(parents=True, exist_ok=True)
    rng = np.random.default_rng(args.seed)
    torch.manual_seed(args.seed)
    d = np.load(args.dataset, allow_pickle=True)
    valid = d["valid"]
    x, y, traj = d["x"][valid], d["y"][valid], d["trajectory"][valid]
    trajectories = np.unique(traj)
    val_traj = rng.choice(trajectories, max(1, int(round(args.val_fraction * len(trajectories)))), replace=False)
    train_traj = np.setdiff1d(trajectories, val_traj)
    is_val = np.isin(traj, val_traj)
    dev = torch.device(args.device)
    X = torch.as_tensor(x, device=dev); Y = torch.as_tensor(y, device=dev)
    tr_idx = np.flatnonzero(~is_val); va_idx = torch.as_tensor(np.flatnonzero(is_val), device=dev)

    model = GapEnsemble(list(map(str, d["x_names"])), list(map(str, d["y_names"])), args.members,
                        tuple(args.hidden), args.dof).to(dev)
    model.set_normalization(X[tr_idx], Y[tr_idx])
    Yn = model.normalize_y(Y)
    log = dict(dataset=str(args.dataset), train_trajectories=train_traj.tolist(), val_trajectories=val_traj.tolist(),
               train_samples=int(len(tr_idx)), val_samples=int(len(va_idx)), args={k: str(v) for k, v in vars(args).items()},
               members=[])
    for k, member in enumerate(model.members):
        # Bootstrap over training trajectories; frames inherit their trajectory's multiplicity.
        boot = rng.choice(train_traj, len(train_traj), replace=True)
        counts = {t: c for t, c in zip(*np.unique(boot, return_counts=True))}
        member_idx = np.concatenate([np.repeat(np.flatnonzero((traj == t) & ~is_val), c) for t, c in counts.items()])
        member_idx = torch.as_tensor(member_idx, device=dev)
        opt = torch.optim.AdamW(member.parameters(), lr=args.lr, weight_decay=args.weight_decay)
        sched = torch.optim.lr_scheduler.CosineAnnealingLR(opt, args.epochs)
        best, best_state, stale, history = float("inf"), None, 0, []
        start = time.time()
        for epoch in range(args.epochs):
            member.train()
            perm = member_idx[torch.randperm(len(member_idx), device=dev)]
            for i in range(0, len(perm), args.batch_size):
                b = perm[i:i + args.batch_size]
                loc, log_scale = member(model.normalize_x(X[b]))
                loss = student_t_nll(loc, log_scale, Yn[b], args.dof).mean()
                opt.zero_grad(set_to_none=True); loss.backward(); opt.step()
            sched.step()
            member.eval()
            with torch.no_grad():
                loc, log_scale = member(model.normalize_x(X[va_idx]))
                val_nll = float(student_t_nll(loc, log_scale, Yn[va_idx], args.dof).mean())
                val_rmse = float(((loc - Yn[va_idx]) ** 2).mean().sqrt())
            history.append(dict(epoch=epoch, train_nll=float(loss), val_nll=val_nll, val_rmse_std_units=val_rmse))
            if val_nll < best - 1e-4:
                best, stale = val_nll, 0
                best_state = {n: t.detach().clone() for n, t in member.state_dict().items()}
            else:
                stale += 1
                if stale >= args.patience:
                    break
        member.load_state_dict(best_state)
        log["members"].append(dict(member=k, best_val_nll=best, epochs=len(history), seconds=time.time() - start,
                                   history=history))
        print(f"[gap-train] member {k}: best val NLL {best:.4f} after {len(history)} epochs "
              f"(val RMSE {min(h['val_rmse_std_units'] for h in history):.3f} std units)", flush=True)

    model.eval()
    with torch.no_grad():
        pred = model.predict(X[va_idx])
        resid = pred["mean"] - Y[va_idx]
        r2 = 1 - (resid ** 2).mean(0) / Y[va_idx].var(0).clamp_min(1e-12)
    log["ensemble_val"] = dict(rmse_std_units=float(((pred["mean_std_units"] - Yn[va_idx]) ** 2).mean().sqrt()),
                               r2_mean=float(r2.mean()), u_median=float(pred["u"].median()))
    model.save(args.output_dir / "model.pt", extra=dict(dataset=str(args.dataset), val_trajectories=val_traj.tolist()))
    (args.output_dir / "config.json").write_text(json.dumps(model.config(), indent=2))
    (args.output_dir / "train_log.json").write_text(json.dumps(log, indent=1))
    print(json.dumps(log["ensemble_val"], indent=2))


if __name__ == "__main__":
    main()
