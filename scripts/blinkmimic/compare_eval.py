"""Compare two BlinkMimic evaluation JSONs (per-clip normalized return, failures), e.g. Isaac vs mjlab (MuJoCo Warp).

    python scripts/blinkmimic/compare_eval.py A.json B.json --labels Isaac mjlab --out cmp.png

Prints overall / per-motion-type / per-clip numbers and saves a matplotlib bar chart of the per-clip returns.
"""

from __future__ import annotations

import argparse
import json
import re
from collections import defaultdict

import numpy as np


def motion_type(clip: str) -> str:
    return re.sub(r"\d+$", "", clip.split("_")[0])


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("a")
    ap.add_argument("b")
    ap.add_argument("--labels", nargs=2, default=("A", "B"))
    ap.add_argument("--out", default=None)
    args = ap.parse_args()
    A, B = json.load(open(args.a)), json.load(open(args.b))
    la, lb = args.labels
    clips = sorted(set(A["per_clip"]) & set(B["per_clip"]))
    ra = np.array([A["per_clip"][c] for c in clips])
    rb = np.array([B["per_clip"][c] for c in clips])
    print(f"{'':28s} {la:>10s} {lb:>10s} {'diff':>8s}")
    print(f"{'mean normalized return':28s} {ra.mean():10.3f} {rb.mean():10.3f} {ra.mean() - rb.mean():+8.3f}")
    print(f"{'median':28s} {np.median(ra):10.3f} {np.median(rb):10.3f}")
    print(f"{'clips >= 0.82':28s} {(ra >= 0.82).sum():10d} {(rb >= 0.82).sum():10d}")
    print(f"{'failed clips':28s} {A['n_failed']:10d} {B['n_failed']:10d}   {la}: {A['failed']}  {lb}: {B['failed']}")
    groups = defaultdict(list)
    for c, x, y in zip(clips, ra, rb):
        groups[motion_type(c)].append((x, y))
    print("\nby motion type:")
    for t, v in sorted(groups.items(), key=lambda kv: np.mean([y for _, y in kv[1]])):
        x, y = np.array(v).T
        print(f"  {t:16s} n={len(v):2d}  {x.mean():.3f}  {y.mean():.3f}  {x.mean() - y.mean():+.3f}")
    print("\nper clip (largest gaps first):")
    for i in np.argsort(ra - rb):
        print(f"  {clips[i]:28s} {ra[i]:.3f}  {rb[i]:.3f}  {ra[i] - rb[i]:+.3f}")
    if args.out:
        import matplotlib

        matplotlib.use("Agg")
        import matplotlib.pyplot as plt

        order = np.argsort(rb)
        x = np.arange(len(clips))
        fig, ax = plt.subplots(figsize=(14, 5))
        ax.bar(x - 0.2, ra[order], 0.4, label=f"{la} (mean {ra.mean():.3f}, {A['n_failed']} failed)")
        ax.bar(x + 0.2, rb[order], 0.4, label=f"{lb} (mean {rb.mean():.3f}, {B['n_failed']} failed)")
        ax.axhline(0.82, color="k", lw=0.8, ls="--", label="0.82 target")
        ax.set_xticks(x)
        ax.set_xticklabels([clips[i] for i in order], rotation=75, ha="right", fontsize=7)
        ax.set_ylabel("normalized return")
        ax.set_ylim(0, 1)
        ax.set_title("Per-clip normalized return, final checkpoint, 1 replica")
        ax.legend(loc="lower right", fontsize=8)
        fig.tight_layout()
        fig.savefig(args.out, dpi=150)
        print(f"\nwrote {args.out}")


if __name__ == "__main__":
    main()
