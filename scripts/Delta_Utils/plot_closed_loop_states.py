#!/usr/bin/env python3
"""Plot closed-loop root, joint and body state evolution for paired Isaac conditions and Genesis.

Inputs follow the closed-loop experiment layout: ``<experiment>/<condition>/rollout.npz`` from
``play.py --batch_eval`` (post-action states) and a Genesis ``--record_motion`` NPZ (pre-action states,
state[0] restored from the same Isaac initial states). Both are placed on one time axis: index k is the
state at k / fps seconds, plotted against reference frame k from the exported policy ONNX. Isaac positions
have their scene origins removed. Rollouts stop at their first failure; statistics use rollouts still alive.
"""
import argparse
import json
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import onnxruntime as ort

# Validated default categorical slots 1-5 in fixed order; reference is neutral ink.
COLORS = ["#2a78d6", "#eb6834", "#1baf7a", "#eda100", "#e87ba4"]
INK, MUTED, GRID, SURFACE = "#3d3d3a", "#6b6a63", "#e4e3dd", "#fcfcfb"
T_MAX = None
plt.rcParams.update({"figure.facecolor": SURFACE, "axes.facecolor": SURFACE, "axes.edgecolor": GRID,
                     "axes.labelcolor": INK, "text.color": INK, "xtick.color": MUTED, "ytick.color": MUTED,
                     "axes.grid": True, "grid.color": GRID, "grid.linewidth": 0.6, "axes.titlesize": 9,
                     "axes.labelsize": 8, "xtick.labelsize": 7, "ytick.labelsize": 7, "font.size": 8})


def label(key):
    if key.startswith("single_"):
        return f"Isaac single delta {key.split('_')[-1].upper()}"
    return {"zero_delta": "Isaac zero delta", "genesis": "Genesis"}.get(key, f"Isaac λ={key.split('_')[-1]}")


def load_isaac(path):
    with np.load(path, allow_pickle=True) as d:
        origins = d["env_origins"][:, None, None, :]
        lengths = d["valid_lengths"].astype(int)
        def series(key, initial_key):
            return np.concatenate([d[initial_key][:, None], d[key]], axis=1)
        out = dict(joint_pos=series("joint_pos", "initial_joint_pos"),
                   body_pos=series("body_pos_w", "initial_body_pos_w") - origins,
                   body_quat=series("body_quat_w", "initial_body_quat_w"),
                   joint_names=list(map(str, d["joint_names"])), body_names=list(map(str, d["body_names"])),
                   fps=float(d["fps"][0]))
    out["alive"] = np.arange(out["joint_pos"].shape[1])[None, :] <= lengths[:, None]
    return out


def load_genesis(path, joint_names, body_names):
    with np.load(path, allow_pickle=True) as d:
        jn, bn = list(map(str, d["joint_names"])), list(map(str, d["body_names"]))
        ji, bi = [jn.index(n) for n in joint_names], [bn.index(n) for n in body_names]
        lengths = d["valid_lengths"].astype(int)
        out = dict(joint_pos=d["joint_pos"][:, :, ji], body_pos=d["body_pos_w"][:, :, bi],
                   body_quat=d["body_quat_w"][:, :, bi])
    out["alive"] = np.arange(out["joint_pos"].shape[1])[None, :] < lengths[:, None]
    return out


def load_reference(onnx_path, joint_names, body_names, frames):
    session = ort.InferenceSession(str(onnx_path), providers=["CPUExecutionProvider"])
    meta = session.get_modelmeta().custom_metadata_map
    jn, bn = meta["joint_names"].split(","), meta["body_names"].split(",")
    ji, bi = [jn.index(n) for n in joint_names], [bn.index(n) for n in body_names]
    obs = np.zeros((1, session.get_inputs()[0].shape[1]), np.float32)
    rows = [session.run(["joint_pos", "body_pos_w", "body_quat_w"], {"obs": obs, "time_step": np.array([[t]], np.float32)})
            for t in range(frames)]
    return dict(joint_pos=np.stack([r[0][0] for r in rows])[:, ji], body_pos=np.stack([r[1][0] for r in rows])[:, bi],
                body_quat=np.stack([r[2][0] for r in rows])[:, bi])


def euler_wxyz(q):
    w, x, y, z = np.moveaxis(q, -1, 0)
    roll = np.arctan2(2 * (w * x + y * z), 1 - 2 * (x * x + y * y))
    pitch = np.arcsin(np.clip(2 * (w * y - z * x), -1, 1))
    yaw = np.arctan2(2 * (w * z + x * y), 1 - 2 * (y * y + z * z))
    return roll, pitch, yaw


def unwrap_alive(angle, alive):
    out = np.full(angle.shape, np.nan)
    for i in range(angle.shape[0]):
        n = int(alive[i].sum())
        out[i, :n] = np.unwrap(angle[i, :n])
    return out


def quat_angle(a, b):
    return 2 * np.arccos(np.clip(np.abs(np.sum(a * b, axis=-1)), 0, 1))


def masked(values, alive):
    shape = alive.shape + (1,) * (values.ndim - 2)
    return np.where(alive.reshape(shape), values, np.nan)


def band(values, alive, min_alive):
    """Median and IQR over alive rollouts at each time; NaN where fewer than min_alive remain."""
    v = masked(values, alive)
    enough = alive.sum(0) >= min_alive
    with np.errstate(all="ignore"), __import__("warnings").catch_warnings():
        __import__("warnings").simplefilter("ignore", RuntimeWarning)
        q = np.nanpercentile(v, [25, 50, 75], axis=0)
    q[:, ~enough] = np.nan
    return q


def finish(fig, axes, keys, colors, path, title, reference=True):
    handles = [plt.Line2D([], [], color=colors[k], lw=2.2 if k == "genesis" else 1.4,
                          ls=(0, (1.2, 1.2)) if k == "genesis" else "-", label=label(k)) for k in keys]
    if reference:
        handles.append(plt.Line2D([], [], color=INK, lw=1.0, ls="--", label="Reference"))
    fig.legend(handles=handles, loc="upper center", ncol=len(handles), frameon=False, bbox_to_anchor=(0.5, 0.995))
    fig.suptitle(title, y=1.035, fontsize=11)
    fig.subplots_adjust(top=1 - 0.55 / fig.get_figheight())
    for ax in np.ravel(axes):
        if T_MAX is not None:
            ax.set_xlim(0, T_MAX)
        for side in ("top", "right"):
            ax.spines[side].set_visible(False)
    if T_MAX is not None:
        path = path.with_name(f"{path.name}_0-{T_MAX:g}s")
    fig.savefig(path.with_suffix(".png"), dpi=160, bbox_inches="tight")
    fig.savefig(path.with_suffix(".pdf"), bbox_inches="tight")
    plt.close(fig)


def draw(ax, t, q, color, key, show_band):
    genesis = key == "genesis"
    if show_band:
        ax.fill_between(t, q[0], q[2], color=color, alpha=0.14, lw=0)
    ax.plot(t, q[1], color=color, lw=2.0 if genesis else 1.2, ls=(0, (1.2, 1.2)) if genesis else "-")


def main():
    p = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    p.add_argument("--experiment_dir", type=Path, required=True)
    p.add_argument("--conditions", nargs="+", default=["zero_delta", "lambda_12", "lambda_14", "lambda_16"])
    p.add_argument("--genesis_states", type=Path, help="Defaults to the single states*.npz under <experiment>/genesis.")
    p.add_argument("--policy_onnx", type=Path, help="Defaults to <experiment>/policy.onnx.")
    p.add_argument("--output_dir", type=Path, help="Defaults to <experiment>/analysis_states.")
    p.add_argument("--min_alive", type=int, default=5, help="Hide statistics when fewer rollouts remain alive.")
    p.add_argument("--t_max", type=float, help="Limit the time axis; figures get a _0-<t_max>s suffix.")
    args = p.parse_args()
    global T_MAX
    T_MAX = args.t_max
    root = args.experiment_dir
    out = args.output_dir or root / "analysis_states"
    out.mkdir(parents=True, exist_ok=True)

    data = {k: load_isaac(root / k / "rollout.npz") for k in args.conditions}
    first = data[args.conditions[0]]
    joint_names, body_names, fps = first["joint_names"], first["body_names"], first["fps"]
    genesis_path = args.genesis_states or next((root / "genesis").glob("states*.npz"))
    data["genesis"] = load_genesis(genesis_path, joint_names, body_names)
    keys = [*args.conditions, "genesis"]
    colors = dict(zip(keys, COLORS))
    T = max(d["joint_pos"].shape[1] for d in data.values())
    for d in data.values():  # pad to a common horizon (Genesis has no post-action state at the final frame)
        pad = T - d["joint_pos"].shape[1]
        for name in ("joint_pos", "body_pos", "body_quat"):
            d[name] = np.concatenate([d[name], np.repeat(d[name][:, -1:], pad, 1)], 1) if pad else d[name]
        d["alive"] = np.concatenate([d["alive"], np.zeros((d["alive"].shape[0], pad), bool)], 1) if pad else d["alive"]
    ref = load_reference(args.policy_onnx or root / "policy.onnx", joint_names, body_names, T - 1)
    for name in ref:  # reference frame k for k < T-1; hold the last frame for the final post-action state
        ref[name] = np.concatenate([ref[name], ref[name][-1:]], 0)
    t = np.arange(T) / fps
    root_i = body_names.index("pelvis")

    checks = {}
    for k in keys:
        checks[k] = dict(
            initial_joint_max_abs_diff_vs_first=float(np.abs(data[k]["joint_pos"][:, 0] - first["joint_pos"][:, 0]).max()),
            initial_body_pos_max_abs_diff_vs_first=float(np.abs(data[k]["body_pos"][:, 0] - first["body_pos"][:, 0]).max()))
    if checks["genesis"]["initial_body_pos_max_abs_diff_vs_first"] > 1e-3:
        raise RuntimeError(f"Genesis initial states do not match Isaac after origin removal: {checks['genesis']}")

    derived = {}
    for k, d in data.items():
        roll, pitch, yaw = euler_wxyz(d["body_quat"][:, :, root_i])
        derived[k] = dict(
            root_pos=d["body_pos"][:, :, root_i], roll=unwrap_alive(roll, d["alive"]), pitch=pitch, yaw=unwrap_alive(yaw, d["alive"]),
            root_pos_err=np.linalg.norm(d["body_pos"][:, :, root_i] - ref["body_pos"][None, :, root_i], axis=-1),
            root_ori_err=quat_angle(d["body_quat"][:, :, root_i], ref["body_quat"][None, :, root_i]),
            body_err=np.linalg.norm(d["body_pos"] - ref["body_pos"][None], axis=-1),
            joint_rmse=np.sqrt(np.mean((d["joint_pos"] - ref["joint_pos"][None]) ** 2, axis=-1)))
    r_roll, r_pitch, r_yaw = euler_wxyz(ref["body_quat"][:, root_i])
    r_roll, r_yaw = np.unwrap(r_roll), np.unwrap(r_yaw)

    # Root
    fig, axes = plt.subplots(3, 3, figsize=(13, 8.5), sharex=True)
    panels = [("Pelvis x (m)", lambda k: derived[k]["root_pos"][..., 0], ref["body_pos"][:, root_i, 0]),
              ("Pelvis y (m)", lambda k: derived[k]["root_pos"][..., 1], ref["body_pos"][:, root_i, 1]),
              ("Pelvis height z (m)", lambda k: derived[k]["root_pos"][..., 2], ref["body_pos"][:, root_i, 2]),
              ("Pelvis roll, unwrapped (rad)", lambda k: derived[k]["roll"], r_roll),
              ("Pelvis pitch (rad)", lambda k: derived[k]["pitch"], r_pitch),
              ("Pelvis yaw, unwrapped (rad)", lambda k: derived[k]["yaw"], r_yaw),
              ("Pelvis position error (m)", lambda k: derived[k]["root_pos_err"], None),
              ("Pelvis orientation error (rad)", lambda k: derived[k]["root_ori_err"], None)]
    for ax, (title, fn, reference) in zip(axes.flat, panels):
        for k in keys:
            draw(ax, t, band(fn(k), data[k]["alive"], args.min_alive), colors[k], k, show_band=True)
        if reference is not None:
            ax.plot(t, reference, color=INK, lw=1.0, ls="--")
        ax.set_title(title)
    ax = axes.flat[-1]
    for k in keys:  # Genesis records no post-action state for the final step
        n = T - 1 if k == "genesis" else T
        ax.plot(t[:n], data[k]["alive"].sum(0)[:n], color=colors[k], lw=2.0 if k == "genesis" else 1.2,
                ls=(0, (1.2, 1.2)) if k == "genesis" else "-")
    ax.set_title("Rollouts still alive (of 100)")
    for ax in axes[-1]:
        ax.set_xlabel("Time (s)")
    finish(fig, axes, keys, colors, out / "root_states",
           "Root (pelvis) state: median over alive rollouts, shaded 25–75%")

    # Joints
    rows, cols = 6, 5
    fig, axes = plt.subplots(rows, cols, figsize=(15, 15), sharex=True)
    for j, ax in enumerate(axes.flat):
        if j >= len(joint_names):
            ax.axis("off")
            continue
        for k in keys:
            draw(ax, t, band(data[k]["joint_pos"][..., j], data[k]["alive"], args.min_alive), colors[k], k, False)
        ax.plot(t, ref["joint_pos"][:, j], color=INK, lw=0.9, ls="--")
        ax.set_title(joint_names[j].removesuffix("_joint"), fontsize=8)
    for ax in axes[-1]:
        ax.set_xlabel("Time (s)")
    finish(fig, axes, keys, colors, out / "joint_states", "Joint positions (rad): median over alive rollouts")

    # Bodies: height and position error
    for name, fn, ref_fn, title in (
            ("body_heights", lambda k, b: data[k]["body_pos"][..., b, 2], lambda b: ref["body_pos"][:, b, 2],
             "Body height z (m): median over alive rollouts"),
            ("body_position_errors", lambda k, b: derived[k]["body_err"][..., b], None,
             "Body world-position error vs reference (m): median over alive rollouts, shaded 25–75%")):
        fig, axes = plt.subplots(4, 4, figsize=(14, 11), sharex=True)
        for b, ax in enumerate(axes.flat):
            if b >= len(body_names):
                ax.axis("off")
                continue
            for k in keys:
                draw(ax, t, band(fn(k, b), data[k]["alive"], args.min_alive), colors[k], k, ref_fn is None)
            if ref_fn is not None:
                ax.plot(t, ref_fn(b), color=INK, lw=0.9, ls="--")
            ax.set_title(body_names[b].removesuffix("_link"), fontsize=8)
        for ax in axes[-1]:
            ax.set_xlabel("Time (s)")
        finish(fig, axes, keys, colors, out / name, title, reference=ref_fn is not None)

    # Paired distance from each Isaac condition to Genesis (same initial state, both alive)
    g = data["genesis"]
    measures = [("Joint-position RMS difference (rad)",
                 lambda d: np.sqrt(np.mean((d["joint_pos"] - g["joint_pos"]) ** 2, -1))),
                ("Pelvis position difference (m)",
                 lambda d: np.linalg.norm(d["body_pos"][:, :, root_i] - g["body_pos"][:, :, root_i], axis=-1)),
                ("Pelvis height difference, Isaac − Genesis (m)",
                 lambda d: d["body_pos"][:, :, root_i, 2] - g["body_pos"][:, :, root_i, 2]),
                ("Mean body position difference (m)",
                 lambda d: np.linalg.norm(d["body_pos"] - g["body_pos"], axis=-1).mean(-1))]
    fig, axes = plt.subplots(2, 2, figsize=(12, 7.5), sharex=True)
    paired_summary = {}
    for ax, (title, fn) in zip(axes.flat, measures):
        for k in args.conditions:
            both = data[k]["alive"] & g["alive"]
            draw(ax, t, band(fn(data[k]), both, args.min_alive), colors[k], k, True)
            values = masked(fn(data[k]), both)
            for window in (1.0, 2.0, 5.0, 20.0):
                n = int(window * fps) + 1
                with np.errstate(all="ignore"):
                    paired_summary.setdefault(k, {}).setdefault(title, {})[f"mean_0_to_{window:g}s"] = float(
                        np.nanmean(values[:, :n]))
        if "height" in title:
            ax.axhline(0, color=MUTED, lw=0.8)
        ax.set_title(title)
    for ax in axes[-1]:
        ax.set_xlabel("Time (s)")
    finish(fig, axes, args.conditions, colors, out / "paired_distance_to_genesis",
           "Paired distance to Genesis from the same initial state (both rollouts alive): median, shaded 25–75%",
           reference=False)

    summary = dict(
        conditions=keys, fps=fps, min_alive=args.min_alive, genesis_states=str(genesis_path),
        time_alignment="index k = state at k/fps vs reference frame k; Isaac post-action arrays shifted by one, "
                       "prefixed with the initial state; Genesis pre-action recording used directly",
        initial_state_checks=checks,
        completion={k: int(data[k]["alive"][:, -1].sum()) if k != "genesis" else int(data[k]["alive"][:, T - 2].sum())
                    for k in keys},
        median_failure_time_s={k: float(np.median(data[k]["alive"].sum(1) / fps)) for k in keys},
        paired_distance_to_genesis=paired_summary)
    (out / "summary.json").write_text(json.dumps(summary, indent=2))
    print(json.dumps({k: summary[k] for k in ("initial_state_checks", "completion", "median_failure_time_s")}, indent=1))


if __name__ == "__main__":
    main()
