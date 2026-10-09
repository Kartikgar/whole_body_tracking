#!/usr/bin/env python3
"""Replay one pelvis-wrench delta checkpoint from multiple recorded state/action starting points."""
from __future__ import annotations
import argparse
import csv
import hashlib
import json
import os
from pathlib import Path
import signal
import subprocess
import sys
from datetime import datetime
import numpy as np

ROOT = Path(__file__).resolve().parents[1]
STATE_KEYS = ('joint_pos', 'joint_vel', 'body_pos_w', 'body_quat_w', 'body_lin_vel_w', 'body_ang_vel_w')


def sha256(path):
    h = hashlib.sha256()
    with Path(path).open('rb') as f:
        for block in iter(lambda: f.read(1024 * 1024), b''):
            h.update(block)
    return h.hexdigest()


def load_dataset(path):
    with np.load(path, allow_pickle=True) as archive:
        data = {k: archive[k] for k in archive.files}
    for k in (*STATE_KEYS, 'joint_names', 'body_names', 'valid_lengths', 'fps'):
        if k not in data:
            raise ValueError(f'Missing dataset field: {k}')
    data['actions'] = data.get('actions', data.get('action'))
    q = data['joint_pos']
    if q.ndim != 3 or data['actions'] is None or data['actions'].shape != q.shape:
        raise ValueError('Expected stacked [trajectory, frame, joint] states and raw actions')
    n, t, j = q.shape
    if j != len(data['joint_names']):
        raise ValueError('Joint names do not match state dimensions')
    for k in STATE_KEYS:
        expected=(n,t,j) if k.startswith('joint_') else (n,t,len(data['body_names']),4 if k=='body_quat_w' else 3)
        if data[k].shape != expected:
            raise ValueError(f'{k} has shape {data[k].shape}, expected {expected}')
    lengths = np.asarray(data['valid_lengths'], dtype=int)
    if lengths.shape != (n,) or np.any(lengths < 1) or np.any(lengths > t):
        raise ValueError('Invalid trajectory lengths')
    for k in STATE_KEYS:
        if data[k].shape[:2] != (n, t):
            raise ValueError(f'Invalid state shape: {k}')
    for i,length in enumerate(lengths):
        if not all(np.isfinite(data[k][i,:length]).all() for k in (*STATE_KEYS,'actions')):
            raise ValueError(f'Nonfinite source data in trajectory {i}')
    for i,length in enumerate(lengths):
        norms=np.linalg.norm(data['body_quat_w'][i,:length],axis=-1)
        if not np.allclose(norms,1.,atol=1e-3):
            raise ValueError(f'Invalid body quaternions in trajectory {i}')
    fps = np.asarray(data['fps']).ravel()
    if not np.all(np.isfinite(fps)) or np.any(fps <= 0) or not np.allclose(fps, fps[0]):
        raise ValueError('A common positive fps is required')
    for k in ('joint_names', 'body_names'):
        if len(set(map(str, data[k]))) != len(data[k]):
            raise ValueError(f'Duplicate {k}')
    if 'action_mode' not in data or str(data['action_mode'].item()) != 'base_policy_raw':
        raise ValueError('Dataset must contain base_policy_raw joint actions')
    # Pre-action recordings store exactly the reset state in frame zero.
    for k in STATE_KEYS:
        initial = data.get('initial_' + k)
        if initial is None or not np.allclose(initial, data[k][:, 0], atol=1e-5):
            raise ValueError(f'Cannot establish pre-action state alignment for {k}')
    return data


def make_windows(lengths, fps, stride_s=0.5, replay_s=1.0, replay_steps=None,
                 trajectories=None, start_frames=None):
    def frames(seconds):
        value = seconds * fps
        if not np.isfinite(value) or seconds <= 0 or not np.isclose(value, round(value)):
            raise ValueError('Durations must be positive whole numbers of control steps')
        return int(round(value))
    stride = frames(stride_s)
    horizon = int(replay_steps) if replay_steps is not None else frames(replay_s)
    if horizon < 1:
        raise ValueError('Replay length must be positive')
    indices = list(range(len(lengths))) if trajectories is None else trajectories
    if len(set(indices)) != len(indices) or any(i < 0 or i >= len(lengths) for i in indices):
        raise ValueError('Trajectory indices must be unique and in range')
    if start_frames is not None and (len(set(start_frames)) != len(start_frames) or any(t < 0 for t in start_frames)):
        raise ValueError('Start frames must be unique and nonnegative')
    return [dict(trajectory=i, start=t, steps=horizon, horizon_s=horizon / fps)
            for i in indices
            for t in (range(0, int(lengths[i]) - horizon, stride) if start_frames is None else sorted(start_frames))
            if t + horizon < int(lengths[i])]


def name_indices(source, target):
    source, target = list(map(str, source)), list(map(str, target))
    if len(set(source)) != len(source) or len(set(target)) != len(target):
        raise ValueError('Duplicate names in state mapping')
    missing = set(target) - set(source)
    if missing:
        raise ValueError(f'Missing recorded names: {sorted(missing)}')
    return [source.index(name) for name in target]


def errors(actual, target, root_idx=0):
    def rmse(k):
        return float(np.sqrt(np.mean((actual[k] - target[k]) ** 2)))
    q, r = actual['body_quat_w'], target['body_quat_w']
    q = q / np.maximum(np.linalg.norm(q, axis=-1, keepdims=True), 1e-12)
    r = r / np.maximum(np.linalg.norm(r, axis=-1, keepdims=True), 1e-12)
    return dict(joint_pos_rmse_rad=rmse('joint_pos'), joint_vel_rmse_rad_s=rmse('joint_vel'),
                body_pos_error_m=float(np.linalg.norm(actual['body_pos_w'] - target['body_pos_w'], axis=-1).mean()),
                root_pos_error_m=float(np.linalg.norm(actual['body_pos_w'][root_idx] - target['body_pos_w'][root_idx])),
                body_orientation_error_rad=float((2 * np.arccos(np.clip(np.abs((q * r).sum(-1)), 0, 1))).mean()),
                body_lin_vel_rmse_m_s=rmse('body_lin_vel_w'), body_ang_vel_rmse_rad_s=rmse('body_ang_vel_w'))


def summarize(rows):
    result = {}
    for horizon in sorted(set(r['steps'] for r in rows)):
        group = [r for r in rows if r['steps'] == horizon]
        good = [r for r in group if r['status'] == 'complete']
        metrics = {}
        for k in good[0] if good else []:
            if not k.startswith(('endpoint_', 'mean_')):
                continue
            values = [r[k] for r in good]
            trajectory_means = [np.mean([r[k] for r in good if r['trajectory'] == i])
                                for i in sorted(set(r['trajectory'] for r in good))]
            metrics[k] = dict(window_mean=float(np.mean(values)), window_p90=float(np.percentile(values, 90)),
                              trajectory_equal_weight_mean=float(np.mean(trajectory_means)),
                              trajectory_p90=float(np.percentile(trajectory_means, 90)))
        result[str(horizon)] = dict(scheduled=len(group), complete=len(good), failed=len(group)-len(good), metrics=metrics)
    return result


def worker(args):
    args.output_dir = args.output_dir.resolve()
    from isaaclab.app import AppLauncher
    launcher = AppLauncher(headless=True, device=args.device)
    app = launcher.app
    env = None
    try:
        import pickle
        import torch
        import gymnasium as gym
        import whole_body_tracking.tasks  # noqa
        from isaaclab_rl.rsl_rl import RslRlVecEnvWrapper
        from rsl_rl.runners import OnPolicyRunner
        checkpoint = Path(args.checkpoint)
        with (checkpoint.parent / 'params/env.pkl').open('rb') as f:
            cfg = pickle.load(f)
        with (checkpoint.parent / 'params/agent.pkl').open('rb') as f:
            agent = pickle.load(f)
        from whole_body_tracking.tasks.tracking.mdp.pelvis_wrench import DeltaPelvisWrenchActionCfg
        if not isinstance(cfg.actions.joint_pos, DeltaPelvisWrenchActionCfg):
            raise ValueError('Checkpoint must use the pelvis-wrench open-loop environment')
        # Training can occur on another machine; relocate only repository assets.
        spawn = cfg.scene.robot.spawn
        if hasattr(spawn, 'asset_path') and not Path(spawn.asset_path).is_file():
            marker = '/source/whole_body_tracking/'
            if marker not in spawn.asset_path:
                raise FileNotFoundError(spawn.asset_path)
            relocated = ROOT / 'source/whole_body_tracking' / spawn.asset_path.split(marker, 1)[1]
            if not relocated.is_file():
                raise FileNotFoundError(relocated)
            spawn.asset_path = str(relocated)
            spawn.usd_dir = str(Path(args.output_dir) / 'robot_usd')
        data = load_dataset(args.dataset)
        fps = float(data['fps'].ravel()[0])
        if not np.isclose(cfg.sim.dt * cfg.decimation, 1/fps):
            raise ValueError('Dataset and trained control rates differ')
        # All conditions must use the same trained dynamics, observations, and action contract.
        fingerprint = {k: getattr(cfg, k).to_dict() for k in ('sim', 'actions', 'observations')}
        fingerprint['robot'] = cfg.scene.robot.to_dict()
        fingerprint['robot']['spawn'].pop('usd_dir', None)
        contract = hashlib.sha256(json.dumps(fingerprint, default=lambda v: (v.__module__ + "." + v.__qualname__) if callable(v) else str(v), sort_keys=True).encode()).hexdigest()
        torch.manual_seed(args.seed)
        np.random.seed(args.seed)
        if cfg.observations.policy.history_length > 0:
            raise ValueError('Temporal observation history is not supported in this replay protocol')
        cfg.seed = args.seed
        cfg.sim.device = args.device
        cfg.scene.num_envs = args.num_envs
        cfg.commands.motion.motion_file = str(Path(args.dataset).resolve())
        cfg.commands.motion.sample_time_steps = False
        cfg.commands.motion.sample_trajectories = False
        cfg.commands.motion.hold_last_frame = True
        cfg.commands.motion.joint_position_range = (0., 0.)
        cfg.commands.motion.pose_range = {k: (0., 0.) for k in cfg.commands.motion.pose_range}
        cfg.commands.motion.velocity_range = {k: (0., 0.) for k in cfg.commands.motion.velocity_range}
        for k in list(vars(cfg.events)):
            if not k.startswith('_'):
                setattr(cfg.events, k, None)
        for group in vars(cfg.observations).values():
            if hasattr(group, 'enable_corruption'):
                group.enable_corruption = False
        env = gym.make('Tracking-Flat-G1-DeltaWrench-OpenLoop-v0', cfg=cfg).unwrapped
        wrapped = RslRlVecEnvWrapper(env)
        agent.device = args.device
        runner = OnPolicyRunner(wrapped, agent.to_dict(), log_dir=None, device=args.device)
        runner.load(str(checkpoint))
        if getattr(runner.alg.policy, 'is_recurrent', False):
            raise ValueError('Recurrent delta policies require a recorded-history protocol and are not supported')
        policy = runner.get_inference_policy(device=args.device)
        term = env.command_manager.get_term('motion')
        robot = env.scene['robot']
        action = env.action_manager.get_term('joint_pos')
        if action.action_dim != 6:
            raise ValueError('Expected six-component wrench actions')
        if set(map(str,data['joint_names'])) != set(robot.joint_names):
            raise ValueError('Dataset and robot must have the same joint set')
        joints = name_indices(data['joint_names'], robot.joint_names)
        bodies = name_indices(robot.body_names, data['body_names'])
        root = list(map(str, data['body_names'])).index('pelvis')
        if robot.body_names[0] != 'pelvis':
            raise ValueError('Robot root must be pelvis')
        for name, actual in [('action_scale', action._joint_scale[0].cpu().numpy()),
                             ('default_joint_pos', action._offset[0].cpu().numpy())]:
            if name not in data or not np.allclose(actual, data[name][joints], atol=5.1e-4, rtol=0):
                raise ValueError(f'Dataset differs from trained {name}')
        # ONNX metadata uses rounded affine values. Replay the exact logged transform.
        action._joint_scale = torch.as_tensor(data['action_scale'][joints], device=args.device).expand(args.num_envs,-1).clone()
        action._offset[:] = torch.as_tensor(data['default_joint_pos'][joints], device=args.device)
        # MotionLoader buffers assume simulator order; explicitly map recorded names.
        for key in ('joint_pos', 'joint_vel', 'joint_action'):
            setattr(term.motion, key, getattr(term.motion, key)[:, joints])
        source_bodies = list(map(str,data['body_names']))
        command_bodies = name_indices(source_bodies, term.cfg.body_names)
        for key in ('_body_pos_w', '_body_quat_w', '_body_lin_vel_w', '_body_ang_vel_w'):
            setattr(term.motion, key, getattr(term.motion,key)[:,command_bodies])
        out = Path(args.output_dir)
        windows = json.loads((out/'windows.json').read_text())
        rows = []
        recording = []
        def state(i):
            rd = robot.data
            return dict(joint_pos=rd.joint_pos[i].detach().cpu().numpy()[np.argsort(joints)],
                        joint_vel=rd.joint_vel[i].detach().cpu().numpy()[np.argsort(joints)],
                        body_pos_w=rd.body_pos_w[i,bodies].detach().cpu().numpy()-env.scene.env_origins[i].cpu().numpy(),
                        body_quat_w=rd.body_quat_w[i,bodies].detach().cpu().numpy(),
                        body_lin_vel_w=rd.body_link_lin_vel_w[i,bodies].detach().cpu().numpy(),
                        body_ang_vel_w=rd.body_link_ang_vel_w[i,bodies].detach().cpu().numpy())
        def target(traj, frame):
            return {k:data[k][traj,frame] for k in STATE_KEYS}
        env.reset()
        for offset in range(0, len(windows), args.num_envs):
            batch = windows[offset:offset+args.num_envs]
            # Windows are ordered by horizon; do not mix different lengths in a batch.
            # Each env stops contributing after its scheduled horizon.
            env.reset()
            env.action_manager.reset(env_ids=torch.arange(args.num_envs, device=args.device))
            env.observation_manager.reset(env_ids=torch.arange(args.num_envs, device=args.device))
            n = len(batch)
            ids = torch.arange(n, device=args.device)
            traj = np.array([w['trajectory'] for w in batch]); starts = np.array([w['start'] for w in batch])
            term.trajectory_ids[ids] = torch.as_tensor(traj, device=args.device)
            term.time_steps[ids] = torch.as_tensor(starts, device=args.device)
            def tensor(values): return torch.as_tensor(values, device=args.device, dtype=torch.float32)
            robot.write_joint_state_to_sim(tensor(data['joint_pos'][traj,starts][:,joints]),
                                           tensor(data['joint_vel'][traj,starts][:,joints]),env_ids=ids)
            rp=data['body_pos_w'][traj,starts,root]+env.scene.env_origins[:n].cpu().numpy()
            rs=np.concatenate((rp,data['body_quat_w'][traj,starts,root],
                               data['body_lin_vel_w'][traj,starts,root],data['body_ang_vel_w'][traj,starts,root]),axis=-1)
            robot.write_root_link_state_to_sim(tensor(rs),env_ids=ids)
            env.scene["contact_forces"].reset(ids)
            env.scene.write_data_to_sim();env.sim.forward();env.scene.update(0.)
            reset_errors=[errors(state(i),target(traj[i],starts[i]),root) for i in range(n)]
            valid=[max(e['joint_pos_rmse_rad'],e['joint_vel_rmse_rad_s'],e['root_pos_error_m'])<1e-4
                   and e['body_pos_error_m']<1e-3 and e['body_orientation_error_rad']<1e-3
                   and e['body_lin_vel_rmse_m_s']<1e-3 and e['body_ang_vel_rmse_rad_s']<1e-3 for e in reset_errors]
            per_window=[[] for _ in batch];forces=[[] for _ in batch];saved=[[] for _ in batch]
            for step in range(max(w['steps'] for w in batch)):
                obs = env.observation_manager.compute_group('policy')
                with torch.inference_mode():
                    delta = torch.zeros((args.num_envs,6),device=args.device) if args.zero_delta else policy(obs)
                finite_delta = torch.isfinite(delta).all(dim=-1)
                for i in range(n):
                    if not bool(finite_delta[i]):
                        valid[i] = False
                delta = torch.where(finite_delta[:, None], delta, torch.zeros_like(delta))
                env.action_manager.process_action(delta)
                for substep in range(cfg.decimation):
                    env.action_manager.apply_action();env.scene.write_data_to_sim();env.sim.step(render=False)
                    env.scene.update(cfg.sim.dt)
                for i,w in enumerate(batch):
                    if step>=w['steps'] or not valid[i]: continue
                    actual=state(i)
                    if not all(np.isfinite(a).all() for a in actual.values()): valid[i]=False;continue
                    per_window[i].append(errors(actual,target(traj[i],starts[i]+step+1),root))
                    forces[i].append(action._wrench[i].detach().cpu().numpy().copy())
                    if args.record_rollouts:
                        saved[i].append(actual|{'actions':data['actions'][traj[i],starts[i]+step].copy(),
                                               'wrench':forces[i][-1]})
                term.time_steps += 1
            for i,w in enumerate(batch):
                good=valid[i] and len(per_window[i])==w['steps']
                row=w|dict(status='complete' if good else 'reset_or_numerical_failure', reset_errors=reset_errors[i])
                if good:
                    row.update({'endpoint_'+k:v for k,v in per_window[i][-1].items()})
                    row.update({'mean_'+k:float(np.mean([e[k] for e in per_window[i]])) for k in per_window[i][0]})
                    wrench=np.asarray(forces[i]);row['force_mean_n']=float(np.linalg.norm(wrench[:,:3],axis=-1).mean())
                    row['torque_mean_nm']=float(np.linalg.norm(wrench[:,3:],axis=-1).mean())
                rows.append(row)
                if args.record_rollouts and good:
                    recording.append((offset+i,saved[i]))
            (out/'partial_results.json').write_text(json.dumps(rows,indent=2))
            print(f'Evaluated {len(rows)}/{len(windows)} windows',flush=True)
        columns=sorted(set().union(*(r.keys() for r in rows)))
        with (out/'windows.csv').open('w') as f:
            writer=csv.DictWriter(f,fieldnames=columns);writer.writeheader();writer.writerows(rows)
        if args.record_rollouts:
            payload={f'window_{idx}_{k}':np.stack([s[k] for s in seq]) for idx,seq in recording for k in seq[0]}
            np.savez_compressed(out/'rollouts.npz',**payload)
        report=dict(status='complete',configuration_contract=contract,wrench_contract=action.wrench_contract(),
                    checkpoint_sha256=sha256(checkpoint),config_sha256={k:sha256(checkpoint.parent/'params'/k) for k in ('env.pkl','agent.pkl')},
                    reset_tolerances={'joint_and_root':1e-4,'body_pose_and_velocity':1e-3},
                    velocity_convention='body link origins; root link velocity converted to COM by Isaac',
                    summary=summarize(rows),
                    per_trajectory={str(i):summarize([r for r in rows if r['trajectory']==i])
                                    for i in sorted(set(r['trajectory'] for r in rows))})
        (out/'summary.json').write_text(json.dumps(report,indent=2))
    except BaseException:
        import traceback
        traceback.print_exc()
        sys.stdout.flush(); sys.stderr.flush()
        raise
    finally:
        # SimulationApp.close can hang; only the owned child exits.
        import threading
        def close():
            if env is not None:
                env.close()
            app.close()
        t=threading.Thread(target=close,daemon=True);t.start();t.join(10)
        if t.is_alive():os._exit(0 if (Path(args.output_dir)/'summary.json').exists() else 1)


def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument('--dataset', type=Path, required=True, help='Recorded state/action NPZ')
    p.add_argument('--checkpoint', type=Path, required=True, help='Delta checkpoint with params/env.pkl and agent.pkl')
    p.add_argument('--num_envs', type=int, default=100)
    p.add_argument('--device', default='cuda:0')
    p.add_argument('--seed', type=int, default=0)
    p.add_argument('--output_dir', type=Path)
    horizon = p.add_mutually_exclusive_group()
    horizon.add_argument('--replay_length_s', type=float, help='Fixed replay length in seconds; default 1 second')
    horizon.add_argument('--replay_steps', type=int, help='Fixed replay length in control steps; use 1 for one-step evaluation')
    p.add_argument('--start_stride_s', '--stride_s', dest='stride_s', type=float, default=.5)
    p.add_argument('--start_frames', type=int, nargs='+', help='Explicit start frames, applied to each selected trajectory')
    p.add_argument('--trajectory_indices', type=int, nargs='+', help='Subset of recorded trajectories; default all')
    p.add_argument('--max_windows', type=int, help='Evenly subsample the schedule for a smoke test')
    p.add_argument('--record_rollouts', action='store_true')
    p.add_argument('--zero_delta', action='store_true', help='Replay with zero delta using the supplied checkpoint configuration')
    p.add_argument('--validate_only', action='store_true')
    p.add_argument('--timeout_s', type=float, default=3600)
    p.add_argument('--worker', action='store_true', help=argparse.SUPPRESS)
    args = p.parse_args()
    if args.worker:
        worker(args)
        return
    if args.num_envs < 1 or not np.isfinite(args.timeout_s) or args.timeout_s <= 0 or args.seed < 0 or args.max_windows is not None and args.max_windows < 1:
        p.error('Invalid limits or seed')
    args.checkpoint = args.checkpoint.resolve()
    for file in (args.checkpoint, args.checkpoint.parent/'params/env.pkl', args.checkpoint.parent/'params/agent.pkl'):
        if not file.is_file():
            p.error(f'Missing checkpoint/config: {file}')
    args.dataset = args.dataset.resolve()
    data = load_dataset(args.dataset)
    try:
        windows = make_windows(data['valid_lengths'], float(data['fps'].ravel()[0]), args.stride_s,
                               args.replay_length_s if args.replay_length_s is not None else 1., args.replay_steps,
                               args.trajectory_indices, args.start_frames)
    except ValueError as exc:
        p.error(str(exc))
    if args.max_windows and len(windows) > args.max_windows:
        windows = [windows[i] for i in np.linspace(0, len(windows)-1, args.max_windows, dtype=int)]
    if not windows:
        p.error('No eligible windows: replay must have an observed successor within valid_lengths')
    if args.validate_only:
        print(json.dumps(dict(windows=len(windows), checkpoint=str(args.checkpoint),
                              dataset=str(args.dataset), replay_steps=windows[0]['steps']), indent=2))
        return
    out = (args.output_dir or ROOT/'logs/delta_eval'/datetime.now().strftime('%Y%m%d_%H%M%S')).resolve()
    out.mkdir(parents=True, exist_ok=False)
    (out/'windows.json').write_text(json.dumps(windows, indent=2))
    manifest = dict(dataset=str(args.dataset), dataset_sha256=sha256(args.dataset),
                    checkpoint=str(args.checkpoint), checkpoint_sha256=sha256(args.checkpoint),
                    state_timing='pre_action; s[t]+a[t] compared to s[t+1]', seed=args.seed,
                    num_envs=args.num_envs, device=args.device, stride_s=args.stride_s,
                    replay_steps=windows[0]['steps'], replay_length_s=windows[0]['horizon_s'],
                    zero_delta=args.zero_delta, record_rollouts=args.record_rollouts, status='running')
    (out/'manifest.json').write_text(json.dumps(manifest, indent=2))
    command = [sys.executable, str(Path(__file__).resolve()), '--worker', '--dataset', str(args.dataset),
               '--output_dir', str(out), '--checkpoint', str(args.checkpoint), '--num_envs', str(args.num_envs),
               '--device', args.device, '--seed', str(args.seed)]
    if args.zero_delta:
        command.append('--zero_delta')
    if args.record_rollouts:
        command.append('--record_rollouts')
    environment = os.environ.copy()
    environment['PYTHONUNBUFFERED'] = '1'
    environment['PYTHONPATH'] = str(ROOT/'source/whole_body_tracking') + os.pathsep + environment.get('PYTHONPATH', '')
    try:
        with (out/'isaac.log').open('w') as log:
            process = subprocess.Popen(command, cwd=ROOT, env=environment, stdout=log,
                                       stderr=subprocess.STDOUT, start_new_session=True)
            try:
                process.wait(timeout=args.timeout_s)
            finally:
                if process.poll() is None:
                    os.killpg(process.pid, signal.SIGTERM)
                    try:
                        process.wait(timeout=10)
                    except subprocess.TimeoutExpired:
                        os.killpg(process.pid, signal.SIGKILL)
                        process.wait()
        manifest['child_exit_code'] = process.returncode
        if process.returncode or not (out/'summary.json').exists():
            raise RuntimeError(f'Delta replay exited with code {process.returncode}; see {out / "isaac.log"}')
        manifest['status'] = 'complete'
        print(f'Results: {out}')
    except BaseException as exc:
        manifest.update(status='failed', error=str(exc))
        raise
    finally:
        (out/'manifest.json').write_text(json.dumps(manifest, indent=2))


if __name__ == '__main__':
    main()
