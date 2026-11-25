"""Retarget LAFAN1 BVH clips to Fauna Sprout with frax (all CPU cores), and score them.

    python scripts/sprout/retarget/retarget_lafan1.py            # frax env; all 77 clips
    python scripts/sprout/retarget/retarget_lafan1.py --clips walk1_subject1 dance1_subject1

Output per clip, <out_dir>/<clip>.npz:
    q            (T, 34)  [root pos (3), root quat wxyz (4), 27 joints in sprout_spec order]
    fps          30
    joint_names  (27,)    sprout_spec.JOINT_NAMES
plus <out_dir>/metrics.csv (one row per clip) and <out_dir>/run_info.json.

Metrics (per clip):
    pos_err_mean_cm / pos_err_p95_cm   tracked-link position error vs scaled targets, over the
                                       links with a position weight (pelvis, feet, hands)
    feet_err_p95_cm                    the same, feet only
    sole_min_cm                        lowest sole-sphere bottom (negative = floor penetration)
    contact_slide_cm_s                 mean horizontal speed of a foot while its sole is within
                                       2 cm of the floor (foot skating)
    limit_frac                         fraction of joint samples within 0.02 rad of a limit
    spike_frames                       frames where any joint moves > 15 rad/s (IK flips)
    leg_collision_frac                 fraction of frames where the left and right legs interpenetrate
    self_collision_frac                same, over all SELF_COLLISION_LINK_PAIRS (legs, forearms vs torso/thighs/arms)
    self_collision_min_cm              deepest interpenetration over the clip (negative = penetration)
"""

from __future__ import annotations

import os

os.environ["JAX_PLATFORMS"] = "cpu"
os.environ["JAX_ENABLE_X64"] = "1"
os.environ["XLA_FLAGS"] = "--xla_cpu_multi_thread_eigen=false --xla_cpu_scheduler_type=CPU_SCHEDULER_TYPE_MEMORY_OPTIMIZED"
os.environ["OPENBLAS_NUM_THREADS"] = "1"

import jax  # noqa: E402

jax.config.update("jax_num_cpu_devices", os.cpu_count())

import argparse  # noqa: E402
import csv  # noqa: E402
import json  # noqa: E402
import multiprocessing  # noqa: E402
import sys  # noqa: E402
import time  # noqa: E402
from concurrent.futures import ProcessPoolExecutor  # noqa: E402
from pathlib import Path  # noqa: E402

import jax.numpy as jnp  # noqa: E402
import numpy as np  # noqa: E402

sys.path.insert(0, str(Path(__file__).resolve().parent))
import sprout_frax as S  # noqa: E402
from frax.ik import OfflineRetargeter  # noqa: E402

CONTACT_HEIGHT = 0.02
LIMIT_MARGIN = 0.02
SPIKE_RAD_S = 15.0  # a joint moving faster than this between frames is treated as an IK flip


load_clip = S.load_clip  # module-level name so the spawn-based process pool can pickle it


def clip_metrics(robot, retargeter, frames: np.ndarray, q: np.ndarray, fk, coll, lo, hi, scd, leg_pairs) -> dict:
    T = np.asarray(fk(jnp.asarray(q)))  # (T, nv, 4, 4)
    D = np.asarray(scd(jnp.asarray(q)))  # (T, sphere pairs) self-collision distances
    ref = jax.vmap(retargeter.preprocess)(jnp.asarray(frames))
    tracked = np.array([robot.joint_name_to_index[n] for n in S.SPROUT_TRACKED_JOINTS])
    err = np.linalg.norm(T[:, tracked][..., :3, 3] - np.asarray(ref["frame_pos"]), axis=-1)  # (T, n)
    pos_rows = S.WEIGHT_POS > 0
    feet_rows = np.array([b.endswith("FootMod") for b in S.LAFAN1_BONES])

    # Sole spheres from the spherized model (foot frame = ankle joint's child link frame).
    sole_bottom, slide = [], []
    for side in ("left", "right"):
        j = robot.joint_name_to_index[f"{side}_ankle_joint"]
        c, r = coll[f"{side}_foot_link"]
        world = np.einsum("tij,kj->tki", T[:, j, :3, :3], c) + T[:, j, None, :3, 3]
        bottom = (world[..., 2] - r).min(axis=1)  # (T,)
        sole_bottom.append(bottom)
        foot_xy = T[:, j, :2, 3]
        speed = np.linalg.norm(np.diff(foot_xy, axis=0), axis=-1) * S.LAFAN1_FPS
        contact = (bottom[1:] < CONTACT_HEIGHT) & (bottom[:-1] < CONTACT_HEIGHT)
        if contact.any():
            slide.append(speed[contact])
    joints = q[:, 7:]
    near = (joints - lo < LIMIT_MARGIN) | (hi - joints < LIMIT_MARGIN)
    qd = np.abs(np.diff(joints, axis=0)) * S.LAFAN1_FPS
    return {
        "frames": len(q),
        "pos_err_mean_cm": 100 * float(err[:, pos_rows].mean()),
        "pos_err_p95_cm": 100 * float(np.percentile(err[:, pos_rows], 95)),
        "feet_err_p95_cm": 100 * float(np.percentile(err[:, feet_rows], 95)),
        "sole_min_cm": 100 * float(np.min(sole_bottom)),
        "contact_slide_cm_s": 100 * float(np.concatenate(slide).mean()) if slide else float("nan"),
        "limit_frac": float(near.mean()),
        "spike_frames": int((qd > SPIKE_RAD_S).any(axis=1).sum()),
        "leg_collision_frac": float((D[:, leg_pairs] < 0).any(axis=1).mean()),
        "self_collision_frac": float((D < 0).any(axis=1).mean()),
        "self_collision_min_cm": 100 * float(D.min()),
    }


def main(args) -> None:
    paths = sorted(Path(args.data_dir).glob("*.bvh"))
    if args.clips:
        paths = [p for p in paths if p.stem in set(args.clips)]
    assert paths, f"no BVH files selected in {args.data_dir}"
    t0 = time.perf_counter()
    with ProcessPoolExecutor(mp_context=multiprocessing.get_context("spawn")) as pool:
        clips = list(pool.map(load_clip, paths))
    t_load = time.perf_counter() - t0
    print(f"Loaded {len(clips)} clips ({sum(map(len, clips))} frames) in {t_load:.1f} s")

    robot = S.load_sprout()
    v_max = None if args.v_max <= 0 else args.v_max
    retargeter, info = S.make_retargeter(
        robot, floor=not args.no_floor, v_max=v_max, hip_yaw_damping=args.hip_yaw_damping,
        self_collision=not args.no_self_collision, stance_widen=not args.no_stance_widen,
    )
    offline = OfflineRetargeter(retargeter, 1.0 / S.LAFAN1_FPS)
    t0 = time.perf_counter()
    qs = offline(clips)
    t_solve = time.perf_counter() - t0
    print(f"Retargeted {len(clips)} clips on {min(len(clips), len(offline.devices))} devices in {t_solve:.1f} s "
          f"({sum(map(len, clips)) / t_solve:,.0f} frames/s; compile {offline.compile_time:.1f} s in parallel)")

    out = Path(args.out_dir)
    out.mkdir(parents=True, exist_ok=True)
    fk = jax.jit(jax.vmap(robot.joint_to_world_transforms))
    import yaml

    spheres = yaml.safe_load(S.COLLISION_MODEL.read_text())["collision_spheres"]
    coll = {
        link: (np.array([s["center"] for s in spheres[link]]), np.array([s["radius"] for s in spheres[link]]))
        for link in ("left_foot_link", "right_foot_link")
    }
    scd = jax.jit(jax.vmap(robot.self_collision_distances))
    leg_links = {f"{s}_{n}_link" for s in ("left", "right") for n in S._LEG}
    leg_pairs = np.array([a in leg_links and b in leg_links for a, _, b, _, _ in S.sprout_sc_data()])
    lo = np.asarray(robot.actuated_joint_lower_limits)
    hi = np.asarray(robot.actuated_joint_upper_limits)
    rows = []
    t0 = time.perf_counter()
    for path, frames, q in zip(paths, clips, qs):
        np.savez(out / f"{path.stem}.npz", q=q, fps=S.LAFAN1_FPS, joint_names=np.array(S.JOINT_ORDERING))
        rows.append({"clip": path.stem, **clip_metrics(robot, retargeter, frames, q, fk, coll, lo, hi, scd, leg_pairs)})
    print(f"Saved + scored in {time.perf_counter() - t0:.1f} s -> {out}")

    with open(out / "metrics.csv", "w", newline="") as f:
        w = csv.DictWriter(f, fieldnames=list(rows[0]))
        w.writeheader()
        w.writerows(rows)
    (out / "run_info.json").write_text(json.dumps(
        {"clips": len(clips), "frames": int(sum(map(len, clips))), "load_s": t_load, "solve_s": t_solve,
         "compile_s": offline.compile_time, "floor": not args.no_floor,
         **{k: (v.tolist() if isinstance(v, np.ndarray) else v) for k, v in info.items()}}, indent=2))

    keys = [k for k in rows[0] if k not in ("clip", "frames")]
    print(f"\n{'clip':26s} " + " ".join(f"{k:>18s}" for k in keys))
    for r in rows:
        print(f"{r['clip']:26s} " + " ".join(f"{r[k]:18.3f}" for k in keys))


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--data-dir", default=str(S.LAFAN1_DIR))
    ap.add_argument("--out-dir", default=str(S.OUT_DIR.parent / "sprout_v2"),
                    help="default data/lafan1_frax/sprout_v2 (data/lafan1_frax/sprout is the original run)")
    ap.add_argument("--clips", nargs="*", help="clip stems (default: all)")
    ap.add_argument("--no-floor", action="store_true", help="disable the floor constraint")
    ap.add_argument("--v-max", type=float, default=S.RETARGET_V_MAX,
                    help="joint velocity cap in rad/s; <= 0 uses the URDF's 5 rad/s placeholder")
    ap.add_argument("--hip-yaw-damping", type=float, default=S.HIP_YAW_DAMPING)
    ap.add_argument("--no-self-collision", action="store_true", help="disable the self-collision constraint")
    ap.add_argument("--no-stance-widen", action="store_true", help="keep the human foot spacing (scaled)")
    main(ap.parse_args())
