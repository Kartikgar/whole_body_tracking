#!/usr/bin/env python3
"""Gap-model uncertainty u along open-loop replay windows, on Isaac replay states and on the logged Genesis states.

g's inputs (the open-loop delta actor observation without contact forces) are rebuilt offline from recorded
states: base height, base-frame root linear (x2) / angular (x0.25) velocity, projected gravity, joint position
relative to default, joint velocity (x0.05) and the logged raw base action at that frame. The rebuild is checked
against the inputs stored in the one-step gap dataset (computed by Isaac's observation manager) before use.
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
from eval_delta_replay_isaac import load_dataset  # noqa: E402

FALL_HEIGHT_M = 0.5


def rotate_inverse(q, v):
    """Rotate world vectors v into the frame of quaternions q (w, x, y, z)."""
    w, xyz = q[..., :1], q[..., 1:]
    t = 2 * np.cross(xyz, v)
    return v - w * t + np.cross(xyz, t)


def build_inputs(jp, jv, root_pos, root_quat, root_lin, root_ang, action, sim_from_data, default_sim, com_offset_b):
    """Inputs of g in its stored column order, from dataset-order joints (rows = samples).

    Recorded velocities are root-link velocities; Isaac's base_lin_vel is the pelvis centre-of-mass velocity,
    v_com_b = v_link_b + w_b x r_com_b, with r_com_b the COM offset in the pelvis frame.
    """
    gravity = np.broadcast_to([0.0, 0.0, -1.0], root_lin.shape)
    ang_b = rotate_inverse(root_quat, root_ang)
    lin_b = rotate_inverse(root_quat, root_lin) + np.cross(ang_b, com_offset_b)
    return np.concatenate([root_pos[:, 2:3], 2.0 * lin_b, 0.25 * ang_b, rotate_inverse(root_quat, gravity),
                           jp[:, sim_from_data] - default_sim, 0.05 * jv[:, sim_from_data], action[:, sim_from_data]], 1)


def fit_com_offset(lin_link_b, ang_b, lin_com_b):
    """Least-squares pelvis COM offset r from v_com = v_link + w x r (w x r = -[r]x w = skew(w) r)."""
    w = ang_b
    skew = np.zeros((len(w), 3, 3))
    skew[:, 0, 1], skew[:, 0, 2], skew[:, 1, 0] = -w[:, 2], w[:, 1], w[:, 2]
    skew[:, 1, 2], skew[:, 2, 0], skew[:, 2, 1] = -w[:, 0], -w[:, 1], w[:, 0]
    return np.linalg.lstsq(skew.reshape(-1, 3), (lin_com_b - lin_link_b).reshape(-1), rcond=None)[0]


def main():
    p = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    p.add_argument("--model", type=Path, required=True)
    p.add_argument("--dataset", type=Path, required=True, help="Genesis dataset that was replayed")
    p.add_argument("--gap_dataset", type=Path, required=True, help="One-step gap dataset of the same Genesis data")
    p.add_argument("--replays", nargs="+", required=True, help="name=replay_dir (run with --record_rollouts)")
    p.add_argument("--interval_s", type=float, default=0.25)
    p.add_argument("--output_dir", type=Path, required=True)
    p.add_argument("--device", default="cuda:0")
    args = p.parse_args()
    args.output_dir.mkdir(parents=True, exist_ok=True)
    model = GapEnsemble.load(args.model).to(args.device)
    data = load_dataset(args.dataset)
    gap = np.load(args.gap_dataset, allow_pickle=True)
    sim_names = [n[len("joint_vel["):-1] for n in map(str, gap["y_names"]) if n.startswith("joint_vel[")]
    data_names = list(map(str, data["joint_names"]))
    sim_from_data = np.array([data_names.index(n) for n in sim_names])
    default_sim = data["default_joint_pos"][sim_from_data]
    root = data_names and list(map(str, data["body_names"])).index("pelvis")
    fps = float(data["fps"].ravel()[0])

    def u_of(x):
        x = torch.as_tensor(x, dtype=torch.float32, device=args.device)
        return torch.cat([model.predict(x[i:i + 65536])["u"] for i in range(0, len(x), 65536)]).cpu().numpy()

    # Pelvis COM offset, estimated from Isaac's own observations in the gap dataset.
    ok = gap["valid"]
    t, f = gap["trajectory"][ok][:5000], gap["frame"][ok][:5000]
    q = data["body_quat_w"][t, f, root]
    lin_cols = [i for i, n in enumerate(map(str, gap["x_names"])) if n.startswith("base_lin_vel[")]
    com_offset_b = fit_com_offset(rotate_inverse(q, data["body_lin_vel_w"][t, f, root]),
                                  rotate_inverse(q, data["body_ang_vel_w"][t, f, root]), gap["x"][ok][:5000][:, lin_cols] / 2.0)

    def genesis_inputs(traj, frames):
        return build_inputs(data["joint_pos"][traj, frames], data["joint_vel"][traj, frames],
                            data["body_pos_w"][traj, frames, root], data["body_quat_w"][traj, frames, root],
                            data["body_lin_vel_w"][traj, frames, root], data["body_ang_vel_w"][traj, frames, root],
                            data["actions"][traj, frames], sim_from_data, default_sim, com_offset_b)

    # Check the offline rebuild against Isaac's observation manager (stored x of the gap dataset).
    t, f = gap["trajectory"][ok][:2000], gap["frame"][ok][:2000]
    rebuilt = genesis_inputs(t, f)
    stored = gap["x"][ok][:2000]
    check = dict(pelvis_com_offset_b=com_offset_b.round(5).tolist(), max_abs_input_error=float(np.abs(rebuilt - stored).max()),
                 max_abs_u_error=float(np.abs(u_of(rebuilt) - u_of(stored)).max()))
    print("rebuild check:", check, flush=True)
    if check["max_abs_u_error"] > 1e-3:
        raise SystemExit("Offline input rebuild does not match the stored gap inputs")

    step = int(round(args.interval_s * fps))
    curves = {}
    for item in args.replays:
        name, directory = item.split("=", 1)
        directory = Path(directory)
        windows = json.loads((directory / "windows.json").read_text())
        rec = np.load(directory / "rollouts.npz")
        ids = sorted({int(k.split("_")[1]) for k in rec.files})
        horizon = windows[ids[0]]["steps"]
        sample = np.arange(0, horizon + 1, step)
        U, Ug, Z = [], [], []
        for idx in ids:
            w = windows[idx]; traj, start = w["trajectory"], w["start"]
            g = lambda k: rec[f"window_{idx}_{k}"]  # noqa: E731
            # Recorded step j is the state after j+1 actions, i.e. frame start+j+1; prepend the start (Genesis) state.
            states = {k: np.concatenate([data[k][traj, start][None], g(k)]) for k in
                      ("joint_pos", "joint_vel", "body_pos_w", "body_quat_w", "body_lin_vel_w", "body_ang_vel_w")}
            frames = start + sample
            x = build_inputs(states["joint_pos"][sample], states["joint_vel"][sample], states["body_pos_w"][sample, root],
                             states["body_quat_w"][sample, root], states["body_lin_vel_w"][sample, root],
                             states["body_ang_vel_w"][sample, root], data["actions"][traj, frames], sim_from_data, default_sim, com_offset_b)
            U.append(u_of(x)); Ug.append(u_of(genesis_inputs(np.full(len(frames), traj), frames)))
            Z.append(states["body_pos_w"][sample, root, 2])
        curves[name] = dict(t=sample / fps, u=np.array(U), u_genesis=np.array(Ug), upright=np.array(Z) > FALL_HEIGHT_M)

    any_name = next(iter(curves))
    summary = dict(rebuild_check=check, windows={n: int(len(c["u"])) for n, c in curves.items()}, times_s=curves[any_name]["t"].tolist(),
                   u_genesis_median=np.median(curves[any_name]["u_genesis"], 0).tolist())
    for n, c in curves.items():
        summary[n] = dict(u_median=np.median(c["u"], 0).tolist(), u_p25=np.percentile(c["u"], 25, 0).tolist(),
                          u_p75=np.percentile(c["u"], 75, 0).tolist(), upright_fraction=c["upright"].mean(0).tolist(),
                          u_median_upright_only=[float(np.median(c["u"][c["upright"][:, j], j])) if c["upright"][:, j].any()
                                                 else None for j in range(c["u"].shape[1])])
    (args.output_dir / "u_vs_horizon.json").write_text(json.dumps(summary, indent=1))

    colors = {"zero": "#6b7280", "m1m2m3": "#eb6834"}
    fig, axes = plt.subplots(1, 2, figsize=(14, 4.8), constrained_layout=True)
    c0 = curves[any_name]; t = c0["t"]
    ax = axes[0]
    ax.fill_between(t, *np.percentile(c0["u_genesis"], [25, 75], 0), color="#2a78d6", alpha=.15, lw=0)
    ax.plot(t, np.median(c0["u_genesis"], 0), "-o", ms=3.5, lw=2, color="#2a78d6", label="Genesis logged states (teacher-forced)")
    for n, c in curves.items():
        col = colors.get(n, "#1baf7a")
        ax.fill_between(t, np.percentile(c["u"], 25, 0), np.percentile(c["u"], 75, 0), color=col, alpha=.15, lw=0)
        ax.plot(t, np.median(c["u"], 0), "-o", ms=3.5, lw=2, color=col, label=f"Isaac replay, {n}")
        ax.plot(t, summary[n]["u_median_upright_only"], ":", lw=1.5, color=col, label=f"Isaac replay, {n} (upright windows only)")
    ax.axhline(summary["u_genesis_median"][0], color="#2a78d6", lw=.8, ls=":")
    ax.set_yscale("log"); ax.set_xlabel("Horizon (s)"); ax.set_ylabel("u (log scale; target std units)")
    ax.set_title("Gap-model uncertainty along 5 s open-loop windows (median, IQR)"); ax.grid(alpha=.3, which="both"); ax.legend(fontsize=8)
    ax = axes[1]
    for n, c in curves.items():
        ax.plot(t, 100 * c["upright"].mean(0), "-o", ms=3.5, color=colors.get(n, "#1baf7a"), label=f"Isaac replay, {n}")
    ax.set_ylim(0, 102); ax.set_xlabel("Horizon (s)"); ax.set_ylabel("% windows upright (pelvis > 0.5 m)")
    ax.set_title("Isaac replay windows still upright"); ax.grid(alpha=.3); ax.legend(fontsize=8)
    fig.suptitle(f"m4 calibration set (seed 0, 25 traj): {next(iter(summary['windows'].values()))} paired 5 s windows; gap ensemble {args.model.parent.name}")
    fig.savefig(args.output_dir / "u_vs_horizon.png", dpi=130)
    print(json.dumps({k: v for k, v in summary.items() if k != "times_s"}, indent=1)[:3000])


if __name__ == "__main__":
    main()
