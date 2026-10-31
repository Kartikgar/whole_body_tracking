#!/usr/bin/env python3
"""Build a one-step teacher-forced Isaac/Genesis gap dataset for the supervised gap model g(s, a) -> Δs.

For every frame t of every recorded (pre-action) Genesis trajectory, Isaac is reset to the logged state s_t, the
logged base action a_t is applied for one control step with zero pelvis wrench, and the Isaac next state s'_I is
compared with the logged Genesis next state s_{t+1}. The label is Δ = f(s_{t+1}) - f(s'_I) for the feature map f
below; the input is the open-loop delta actor observation at s_t (contact forces stored but excluded from x,
because the contact sensor is zeroed by the reset).

Run in the sim2real (Isaac) environment from ``whole_body_tracking/``. The checkpoint only supplies the
pelvis-wrench open-loop environment configuration (physics, actions, observations); its policy is not used.
"""
from __future__ import annotations

import argparse
import json
import pickle
import sys
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(Path(__file__).resolve().parent))
from eval_delta_replay_isaac import load_dataset, name_indices, sha256  # noqa: E402

CONTACT_TERM = "feet_contact_force"
# Delta targets: base-frame root velocities, joint velocities, root height and projected gravity.
# A one-step Isaac result faster than this is a PhysX blow-up after the teleport, not dynamics (recorded G1 joint speeds
# stay below ~40 rad/s); such transitions are marked invalid instead of entering the labels.
MAX_ISAAC_JOINT_VEL, MAX_ISAAC_ROOT_ANG_VEL = 100.0, 50.0
TARGET_GROUPS = {"root_lin_vel_b": 3, "root_ang_vel_b": 3, "joint_vel": 29, "root_height": 1, "projected_gravity": 3}


def parse_args():
    p = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    p.add_argument("--dataset", type=Path, required=True, help="Genesis pre-action state/action NPZ")
    p.add_argument("--checkpoint", type=Path, required=True,
                   help="Any pelvis-wrench open-loop delta checkpoint (for params/env.pkl only)")
    p.add_argument("--output", type=Path, required=True, help="Output NPZ (a .json metadata file is written beside it)")
    p.add_argument("--num_envs", type=int, default=4096)
    p.add_argument("--device", default="cuda:0")
    p.add_argument("--seed", type=int, default=0)
    p.add_argument("--max_transitions", type=int, help="Smoke-test cap on the number of transitions")
    return p.parse_args()


def features(torch, math, joint_pos, joint_vel, root_pos, root_quat, root_lin_w, root_ang_w):
    """Feature map f(s) used for Δ: base-frame root velocities, joint velocities, root height, projected gravity."""
    gravity = torch.tensor([0.0, 0.0, -1.0], device=root_quat.device).expand(root_quat.shape[0], 3)
    return torch.cat([
        math.quat_rotate_inverse(root_quat, root_lin_w),
        math.quat_rotate_inverse(root_quat, root_ang_w),
        joint_vel,
        root_pos[:, 2:3],
        math.quat_rotate_inverse(root_quat, gravity),
    ], dim=-1)


def main():
    args = parse_args()
    args.output.parent.mkdir(parents=True, exist_ok=True)
    from isaaclab.app import AppLauncher
    app = AppLauncher(headless=True, device=args.device).app
    try:
        import torch
        import gymnasium as gym
        import isaaclab.utils.math as math
        import whole_body_tracking.tasks  # noqa: F401
        from whole_body_tracking.tasks.tracking.mdp.pelvis_wrench import DeltaPelvisWrenchActionCfg

        with (args.checkpoint.parent / "params/env.pkl").open("rb") as f:
            cfg = pickle.load(f)
        if not isinstance(cfg.actions.joint_pos, DeltaPelvisWrenchActionCfg):
            raise ValueError("Checkpoint must use the pelvis-wrench open-loop environment")
        spawn = cfg.scene.robot.spawn
        if hasattr(spawn, "asset_path") and not Path(spawn.asset_path).is_file():
            marker = "/source/whole_body_tracking/"
            spawn.asset_path = str(ROOT / "source/whole_body_tracking" / spawn.asset_path.split(marker, 1)[1])
            spawn.usd_dir = str(args.output.parent / "robot_usd")
        data = load_dataset(args.dataset)
        fps = float(data["fps"].ravel()[0])
        if not np.isclose(cfg.sim.dt * cfg.decimation, 1 / fps):
            raise ValueError("Dataset and environment control rates differ")
        if cfg.observations.policy.history_length > 0:
            raise ValueError("Observation history is not supported")
        torch.manual_seed(args.seed)
        np.random.seed(args.seed)
        # Same deterministic replay settings as eval_delta_replay_isaac.py.
        cfg.seed = args.seed
        cfg.sim.device = args.device
        cfg.scene.num_envs = args.num_envs
        cfg.commands.motion.motion_file = str(args.dataset.resolve())
        cfg.commands.motion.sample_time_steps = False
        cfg.commands.motion.sample_trajectories = False
        cfg.commands.motion.hold_last_frame = True
        cfg.commands.motion.joint_position_range = (0., 0.)
        cfg.commands.motion.pose_range = {k: (0., 0.) for k in cfg.commands.motion.pose_range}
        cfg.commands.motion.velocity_range = {k: (0., 0.) for k in cfg.commands.motion.velocity_range}
        for k in list(vars(cfg.events)):
            if not k.startswith("_"):
                setattr(cfg.events, k, None)
        for group in vars(cfg.observations).values():
            if hasattr(group, "enable_corruption"):
                group.enable_corruption = False
        env = gym.make("Tracking-Flat-G1-DeltaWrench-OpenLoop-v0", cfg=cfg).unwrapped

        term = env.command_manager.get_term("motion")
        robot = env.scene["robot"]
        action = env.action_manager.get_term("joint_pos")
        joints = name_indices(data["joint_names"], robot.joint_names)  # dataset index for each sim joint
        bodies = name_indices(robot.body_names, data["body_names"])  # sim body index for each dataset body
        root = list(map(str, data["body_names"])).index("pelvis")
        if robot.body_names[0] != "pelvis":
            raise ValueError("Robot root must be pelvis")
        action._joint_scale = torch.as_tensor(data["action_scale"][joints], device=args.device).expand(args.num_envs, -1).clone()
        action._offset[:] = torch.as_tensor(data["default_joint_pos"][joints], device=args.device)
        for key in ("joint_pos", "joint_vel", "joint_action"):
            setattr(term.motion, key, getattr(term.motion, key)[:, joints])
        command_bodies = name_indices(list(map(str, data["body_names"])), term.cfg.body_names)
        for key in ("_body_pos_w", "_body_quat_w", "_body_lin_vel_w", "_body_ang_vel_w"):
            setattr(term.motion, key, getattr(term.motion, key)[:, command_bodies])

        # Observation layout of the delta actor ('policy' group) and the input feature columns of g.
        names = env.observation_manager.active_terms["policy"]
        dims = [int(np.prod(d)) for d in env.observation_manager.group_obs_term_dim["policy"]]
        bounds = np.cumsum([0, *dims])
        obs_columns = [f"{n}[{i}]" for n, d in zip(names, dims) for i in range(d)]
        keep = np.concatenate([np.arange(bounds[i], bounds[i + 1]) for i, n in enumerate(names) if n != CONTACT_TERM])
        target_names = [f"{g}[{i}]" for g, d in TARGET_GROUPS.items() for i in range(d)]
        target_names[6:35] = [f"joint_vel[{n}]" for n in robot.joint_names]

        tensor = lambda v: torch.as_tensor(v, device=args.device, dtype=torch.float32)  # noqa: E731
        lengths = np.asarray(data["valid_lengths"], dtype=int)
        pairs = np.array([(i, t) for i, n in enumerate(lengths) for t in range(n - 1)], dtype=np.int64)
        if args.max_transitions:
            pairs = pairs[np.linspace(0, len(pairs) - 1, min(args.max_transitions, len(pairs)), dtype=int)]
        origins = env.scene.env_origins
        jinv = torch.as_tensor(np.argsort(joints), device=args.device)  # sim order -> dataset order
        all_ids = torch.arange(args.num_envs, device=args.device)

        X, OBS, Y, RESET, PLAUSIBLE = [], [], [], [], []
        for offset in range(0, len(pairs), args.num_envs):
            batch = pairs[offset:offset + args.num_envs]
            n = len(batch)
            traj, frame = batch[:, 0], batch[:, 1]
            ids = all_ids[:n]
            env.reset()
            env.action_manager.reset(env_ids=all_ids)
            env.observation_manager.reset(env_ids=all_ids)
            term.trajectory_ids[ids] = torch.as_tensor(traj, device=args.device)
            term.time_steps[ids] = torch.as_tensor(frame, device=args.device)
            robot.write_joint_state_to_sim(tensor(data["joint_pos"][traj, frame][:, joints]),
                                           tensor(data["joint_vel"][traj, frame][:, joints]), env_ids=ids)
            root_state = np.concatenate((data["body_pos_w"][traj, frame, root] + origins[:n].cpu().numpy(),
                                         data["body_quat_w"][traj, frame, root],
                                         data["body_lin_vel_w"][traj, frame, root],
                                         data["body_ang_vel_w"][traj, frame, root]), axis=-1)
            robot.write_root_link_state_to_sim(tensor(root_state), env_ids=ids)
            env.scene["contact_forces"].reset(ids)
            env.scene.write_data_to_sim(); env.sim.forward(); env.scene.update(0.)
            rd = robot.data
            reset_err = torch.stack([
                (rd.joint_pos[:n][:, jinv] - tensor(data["joint_pos"][traj, frame])).abs().amax(-1),
                (rd.joint_vel[:n][:, jinv] - tensor(data["joint_vel"][traj, frame])).abs().amax(-1),
                (rd.body_pos_w[:n, bodies[root]] - origins[:n] - tensor(data["body_pos_w"][traj, frame, root])).norm(dim=-1),
            ], -1)
            obs = env.observation_manager.compute_group("policy")[:n].clone()
            # One control step with zero wrench; the base action comes from the motion command (logged a_t).
            env.action_manager.process_action(torch.zeros((args.num_envs, 6), device=args.device))
            for _ in range(cfg.decimation):
                env.action_manager.apply_action(); env.scene.write_data_to_sim(); env.sim.step(render=False)
                env.scene.update(cfg.sim.dt)
            f_isaac = features(torch, math, rd.joint_pos[:n][:, jinv], rd.joint_vel[:n][:, jinv],
                               rd.body_pos_w[:n, 0] - origins[:n], rd.body_quat_w[:n, 0],
                               rd.body_link_lin_vel_w[:n, 0], rd.body_link_ang_vel_w[:n, 0])
            nxt = frame + 1
            f_gen = features(torch, math, tensor(data["joint_pos"][traj, nxt]), tensor(data["joint_vel"][traj, nxt]),
                             tensor(data["body_pos_w"][traj, nxt, root]), tensor(data["body_quat_w"][traj, nxt, root]),
                             tensor(data["body_lin_vel_w"][traj, nxt, root]), tensor(data["body_ang_vel_w"][traj, nxt, root]))
            delta = f_gen - f_isaac
            # Joint velocities in sim order for the target columns (names above follow robot.joint_names).
            delta[:, 6:35] = delta[:, 6:35][:, torch.as_tensor(joints, device=args.device)]
            OBS.append(obs.cpu().numpy()); X.append(obs[:, keep].cpu().numpy())
            Y.append(delta.cpu().numpy()); RESET.append(reset_err.cpu().numpy())
            PLAUSIBLE.append(((f_isaac[:, 6:35].abs().amax(-1) < MAX_ISAAC_JOINT_VEL)
                              & (f_isaac[:, 3:6].abs().amax(-1) < MAX_ISAAC_ROOT_ANG_VEL)).cpu().numpy())
            print(f"[gap-dataset] {offset + n}/{len(pairs)} transitions", flush=True)

        reset = np.concatenate(RESET)
        plausible = np.concatenate(PLAUSIBLE)
        valid = ((reset[:, 0] < 1e-4) & (reset[:, 1] < 1e-3) & (reset[:, 2] < 1e-4) & np.isfinite(np.concatenate(Y)).all(1)
                 & plausible)
        payload = dict(x=np.concatenate(X).astype(np.float32), y=np.concatenate(Y).astype(np.float32),
                       obs=np.concatenate(OBS).astype(np.float32), trajectory=pairs[:, 0], frame=pairs[:, 1],
                       reset_error=reset.astype(np.float32), valid=valid,
                       x_names=np.array([obs_columns[i] for i in keep]), obs_names=np.array(obs_columns),
                       y_names=np.array(target_names), target_groups=json.dumps(TARGET_GROUPS))
        np.savez_compressed(args.output, **payload)
        meta = dict(dataset=str(args.dataset), dataset_sha256=sha256(args.dataset), checkpoint=str(args.checkpoint),
                    transitions=int(len(pairs)), valid=int(valid.sum()), rejected_isaac_blowups=int((~plausible).sum()), num_envs=args.num_envs, seed=args.seed,
                    fps=fps, observation_terms=dict(zip(names, dims)), excluded_input_terms=[CONTACT_TERM],
                    target_groups=TARGET_GROUPS, label="feature(Genesis s_{t+1}) - feature(Isaac step from s_t, a_t, zero wrench)",
                    reset_error_p99=dict(zip(("joint_pos", "joint_vel", "root_pos"), np.percentile(reset, 99, 0).tolist())),
                    label_abs_mean=dict(zip(TARGET_GROUPS, [float(np.abs(payload["y"][valid][:, s:e]).mean()) for s, e in
                                        zip(np.cumsum([0, *TARGET_GROUPS.values()])[:-1], np.cumsum(list(TARGET_GROUPS.values())))])))
        args.output.with_suffix(".json").write_text(json.dumps(meta, indent=2))
        print(json.dumps(meta, indent=2))
        env.close()
    except Exception:
        # Print before closing: a failing app.close() can otherwise hang and hide the traceback.
        import traceback
        traceback.print_exc()
        sys.stdout.flush()
        raise
    finally:
        app.close()


if __name__ == "__main__":
    main()
