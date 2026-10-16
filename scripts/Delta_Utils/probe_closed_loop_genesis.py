#!/usr/bin/env python3
"""Probe Genesis ground truth along closed-loop rollouts: reset to logged states, replay the logged actions.

For each start frame k on a grid along each rollout (k + horizon within the valid length), Genesis is reset to the
logged state k and stepped with the rollout's own base actions a_k ... a_{k+H-1}. The resulting Genesis states are
written as a stacked recording (one short trajectory per probe window, frame 0 = Genesis state after reset) that
``eval_delta_replay_isaac.py`` accepts, so any Isaac model can be scored against Genesis on these closed-loop states.

Sources: ``isaac`` (``play.py --batch_eval`` rollout.npz: post-action arrays, prefixed here with initial_* and
with scene origins removed) or ``genesis`` (``--record_motion`` NPZ: pre-action arrays; used as a reset
self-consistency check because Genesis should reproduce its own continuation).
Run in the Genesis environment from ``whole_body_tracking/``.
"""
import argparse
import json
import sys
from datetime import datetime
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT / "scripts"))
STATE_KEYS = ("joint_pos", "joint_vel", "body_pos_w", "body_quat_w", "body_lin_vel_w", "body_ang_vel_w")


def load_source(path, kind):
    """Return per-rollout state arrays indexed by frame k (k=0 initial) plus actions[k] applied at frame k."""
    with np.load(path, allow_pickle=True) as d:
        data = {k: d[k] for k in d.files}
    if kind == "isaac":
        origins = data["env_origins"][:, None, :]
        states = {}
        for key in STATE_KEYS:
            initial, arr = data[f"initial_{key}"], data[key]
            if key == "body_pos_w":
                initial, arr = initial - origins, arr - origins[:, None]
            states[key] = np.concatenate([initial[:, None], arr], axis=1)
        # Frames 0..L are known (L post-action states recorded); actions exist for frames 0..L-1.
        last_frame = data["valid_lengths"].astype(int)
        actions = data["actions"]
    else:
        states = {key: data[key] for key in STATE_KEYS}
        # Pre-action recording: frames 0..L-1 known; each has its action.
        last_frame = data["valid_lengths"].astype(int) - 1
        actions = data.get("actions", data.get("action"))
    return dict(states=states, actions=actions, last_frame=last_frame,
                joint_names=list(map(str, data["joint_names"])), body_names=list(map(str, data["body_names"])))


def main():
    p = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    p.add_argument("--source", type=Path, required=True)
    p.add_argument("--source_kind", choices=("isaac", "genesis"), required=True)
    p.add_argument("--policy_onnx", type=Path, required=True, help="Exported base policy; supplies PD gains and defaults.")
    p.add_argument("--motion_file", type=Path, required=True)
    p.add_argument("--horizon_steps", type=int, default=25)
    p.add_argument("--stride_steps", type=int, default=25)
    p.add_argument("--first_frame", type=int, default=0)
    p.add_argument("--num_envs", type=int, default=500)
    p.add_argument("--seed", type=int, default=0)
    p.add_argument("--output", type=Path, required=True)
    args = p.parse_args()
    if args.output.exists():
        raise FileExistsError(args.output)

    src = load_source(args.source, args.source_kind)
    H = args.horizon_steps
    windows = [(r, k) for r in range(len(src["last_frame"]))
               for k in range(args.first_frame, int(src["last_frame"][r]) - H + 1, args.stride_steps)]
    if not windows:
        raise ValueError("No eligible probe windows")
    print(f"[probe] {len(windows)} windows from {args.source}", flush=True)

    import eval_sim2sim_genesis as ev
    sys.argv = [sys.argv[0], "--policy_path", str(args.policy_onnx), "--motion_file", str(args.motion_file),
                "--num_envs", str(args.num_envs), "--no-domain_randomization", "--no-add_noise", "--seed", str(args.seed)]
    eargs = ev.parse_args()
    ev.validate_inputs(eargs)  # resolves the default G1 URDF exactly as the evaluator does
    stamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    runner = ev.build_runner(ev.build_eval_config(eargs, ev.resolve_output_targets(eargs, stamp, eargs.policy_path)))
    scene, controller = runner.scene, runner.controller
    log_bodies = list(scene.log_body_names)
    root_index = src["body_names"].index("pelvis")
    probe_root = log_bodies.index("pelvis")
    src_joint = [src["joint_names"].index(n) for n in runner.policy.meta.joint_names]

    n, J, B = len(windows), len(src_joint), len(log_bodies)
    out = {k: np.zeros((n, H + 1) + shape, np.float32) for k, shape in
           [("joint_pos", (J,)), ("joint_vel", (J,)), ("body_pos_w", (B, 3)), ("body_quat_w", (B, 4)),
            ("body_lin_vel_w", (B, 3)), ("body_ang_vel_w", (B, 3)), ("actions", (J,))]}
    reset_err = {k: np.zeros(n, np.float32) for k in
                 ("joint_pos", "joint_vel", "root_pos", "root_quat", "root_lin_vel", "root_ang_vel")}
    source_cont = {k: np.zeros((n, H + 1, J), np.float32) for k in ("joint_pos",)}
    for b0 in range(0, n, args.num_envs):
        batch = windows[b0:b0 + args.num_envs]
        m = len(batch)
        rs, ks = np.array([w[0] for w in batch]), np.array([w[1] for w in batch])
        start = {key: src["states"][key][rs, ks] for key in STATE_KEYS}
        scene.reset_from_logged_state_batch(
            logged_joint_names=src["joint_names"], initial_joint_pos=start["joint_pos"],
            initial_joint_vel=start["joint_vel"], initial_body_pos_w=start["body_pos_w"],
            initial_body_quat_w=start["body_quat_w"], initial_body_lin_vel_w=start["body_lin_vel_w"],
            initial_body_ang_vel_w=start["body_ang_vel_w"], root_body_index=root_index, batch_size=m)
        state = scene.extract_state_batch()
        for step in range(H + 1):
            sl = slice(b0, b0 + m)
            out["joint_pos"][sl, step] = state["joint_pos"][:m]
            out["joint_vel"][sl, step] = state["joint_vel"][:m]
            for key in ("body_pos_w", "body_quat_w", "body_lin_vel_w", "body_ang_vel_w"):
                out[key][sl, step] = state[f"log_{key}"][:m]
            source_cont["joint_pos"][sl, step] = src["states"]["joint_pos"][rs, ks + step][:, src_joint]
            if step == 0:
                reset_err["joint_pos"][sl] = np.abs(state["joint_pos"][:m] - start["joint_pos"][:, src_joint]).max(-1)
                reset_err["joint_vel"][sl] = np.abs(state["joint_vel"][:m] - start["joint_vel"][:, src_joint]).max(-1)
                reset_err["root_pos"][sl] = np.abs(state["log_body_pos_w"][:m, probe_root] - start["body_pos_w"][:, root_index]).max(-1)
                dot = np.abs(np.sum(state["log_body_quat_w"][:m, probe_root] * start["body_quat_w"][:, root_index], -1))
                reset_err["root_quat"][sl] = 2 * np.arccos(np.clip(dot, 0, 1))
                reset_err["root_lin_vel"][sl] = np.abs(state["log_body_lin_vel_w"][:m, probe_root] - start["body_lin_vel_w"][:, root_index]).max(-1)
                reset_err["root_ang_vel"][sl] = np.abs(state["log_body_ang_vel_w"][:m, probe_root] - start["body_ang_vel_w"][:, root_index]).max(-1)
            if step == H:
                break
            action = np.zeros((args.num_envs, J), np.float32)
            action[:m] = src["actions"][rs, ks + step][:, src_joint]
            action[m:] = action[m - 1]
            out["actions"][sl, step] = action[:m]
            controller.step(controller.compute_joint_target(action, int(step)))
            state = scene.extract_state_batch()
        out["actions"][b0:b0 + m, H] = out["actions"][b0:b0 + m, H - 1]
        print(f"[probe] {b0 + m}/{n}", flush=True)

    meta = runner.policy.meta
    payload = dict(out, **{f"initial_{k}": out[k][:, 0] for k in STATE_KEYS})
    payload.update(valid_lengths=np.full(n, H + 1, np.int32), fps=np.array([1.0 / eargs.control_dt], np.float32),
                   joint_names=np.array(meta.joint_names), body_names=np.array(log_bodies),
                   default_joint_pos=meta.default_joint_pos.astype(np.float32),
                   action_scale=meta.action_scale.astype(np.float32), action_mode=np.array("base_policy_raw"),
                   probe_source_rollout=np.array([w[0] for w in windows], np.int32),
                   probe_source_frame=np.array([w[1] for w in windows], np.int32),
                   probe_source_joint_pos=source_cont["joint_pos"],
                   **{f"probe_reset_error_{k}": v for k, v in reset_err.items()})
    args.output.parent.mkdir(parents=True, exist_ok=True)
    np.savez_compressed(args.output, **payload)
    divergence = np.sqrt(np.mean((out["joint_pos"] - source_cont["joint_pos"]) ** 2, axis=-1))
    report = dict(source=str(args.source), source_kind=args.source_kind, windows=n, horizon_steps=H,
                  stride_steps=args.stride_steps, num_envs=args.num_envs,
                  reset_error_max={k: float(v.max()) for k, v in reset_err.items()},
                  reset_error_p99={k: float(np.percentile(v, 99)) for k, v in reset_err.items()},
                  source_vs_genesis_joint_rmse_rad={str(s): float(divergence[:, s].mean()) for s in (0, 1, 5, H)})
    args.output.with_suffix(".json").write_text(json.dumps(report, indent=2))
    print(json.dumps(report, indent=1))


if __name__ == "__main__":
    main()
