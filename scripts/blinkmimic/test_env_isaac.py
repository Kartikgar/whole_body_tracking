"""Isaac port of BlinkMimic's tools/test_env.py: state round-trip, torch FK (MJCF) vs Isaac body frames (URDF), and the
reward of open-loop reference playback.

    python scripts/blinkmimic/test_env_isaac.py --headless [--num_envs 64]
"""

import argparse
import time

from isaaclab.app import AppLauncher

parser = argparse.ArgumentParser()
parser.add_argument("--robot", type=str, default="sprout")
parser.add_argument("--motion_dir", type=str, default=None)
parser.add_argument("--num_envs", type=int, default=64)
parser.add_argument("--shared", action="store_true", help="test two slices of one scene + snapshot/restore")
AppLauncher.add_app_launcher_args(parser)
args = parser.parse_args()
app = AppLauncher(args).app

import faulthandler  # noqa: E402
import os  # noqa: E402
import sys  # noqa: E402

import torch  # noqa: E402

faulthandler.dump_traceback_later(90, repeat=True)

from whole_body_tracking.blinkmimic.dr import XiLayout  # noqa: E402
from whole_body_tracking.blinkmimic.env_isaac import TrackingSim, make_sim_and_scene  # noqa: E402
from whole_body_tracking.blinkmimic.kinematics import Kinematics  # noqa: E402
from whole_body_tracking.blinkmimic.mdp import MDP  # noqa: E402
from whole_body_tracking.blinkmimic.motion import MotionLib  # noqa: E402
from whole_body_tracking.blinkmimic.robot import get_robot  # noqa: E402
from whole_body_tracking.blinkmimic.rotations import quat_to_mat, rot_angle_sq  # noqa: E402

dev = "cuda:0"
robot = get_robot(args.robot)
kin = Kinematics(robot, dev)
t0 = time.time()
motion = MotionLib.from_dir(args.motion_dir or robot.motion_dir, robot, kin, dev)
print(f"motion: {len(motion.names)} clips, {motion.num_frames} frames ({motion.num_frames * 0.02 / 3600:.2f} h), "
      f"bins {motion.num_bins}, load {time.time() - t0:.1f}s", flush=True)  # fmt: skip
mdp = MDP(kin, motion, torch.tensor(robot.default_joint_pos_list(), device=dev), XiLayout(robot, dev).dim, robot.sensing)
t0 = time.time()
if args.shared:
    # training-like slice A (randomized) + eval slice B in one scene; B is stepped while A is snapshotted/restored
    sh = make_sim_and_scene(robot, 2 * args.num_envs, dev, ground_friction=robot.dr.friction_range[0])
    envA = TrackingSim(args.num_envs, robot, motion, mdp, dev, shared=sh)
    envB = TrackingSim(args.num_envs, robot, motion, mdp, dev, eval_mode=True, randomize=False, reset_noise=False,
                       shared=sh, env_offset=args.num_envs)
    for _ in range(20):  # get A moving
        envA.step(torch.zeros(args.num_envs, robot.num_joints, device=dev))
    SA0, snap = envA.state(), envA.snapshot()
    for _ in range(50):
        envB.step(torch.zeros(args.num_envs, robot.num_joints, device=dev))
    SA_moved = envA.state()
    envA.restore(snap)
    SA1 = envA.state()
    for n, a, b, c in zip(["p", "R", "v", "w", "q", "qd"], SA0, SA1, SA_moved):
        print(f"shared: A after B stepped + restore, {n}: max err {(a - b).abs().max().item():.2e} "
              f"(moved by {(a - c).abs().max().item():.2e} before restore)", flush=True)
    # continuing A from the restored state must behave like a normal step (finite, plausible)
    out = envA.step(torch.zeros(args.num_envs, robot.num_joints, device=dev))
    print("shared: A step after restore finite:", bool(torch.isfinite(out["s1"]).all()), flush=True)
    sys.stdout.flush()
    os._exit(0)
env = TrackingSim(args.num_envs, robot, motion, mdp, dev, eval_mode=True)
print(f"isaac env with {args.num_envs} envs in {time.time() - t0:.1f}s", flush=True)
frames = motion.clip_start[torch.arange(args.num_envs, device=dev) % len(motion.names)] + 100
env.reset(torch.arange(args.num_envs, device=dev), frames)
S0 = env.state()
ref0 = motion.ref_state(frames)
for n, a, b in zip(["p", "R", "v", "w", "q", "qd"], S0, ref0):
    print(f"roundtrip (no refresh) {n}: max err {(a - b).abs().max().item():.2e}", flush=True)
d0 = env.robot.data
print("com offset in root frame (max |.|):", env.robot.root_physx_view.get_coms()[:, 0, :3].abs().max().item(), flush=True)
env.refresh()
S = env.state()
ref = motion.ref_state(frames)
for n, a, b in zip(["p", "R", "v", "w", "q", "qd"], S, ref):
    print(f"roundtrip {n}: max err {(a - b).abs().max().item():.2e}", flush=True)
d = env.robot.data
ids = [env.robot.body_names.index(b) for b in robot.tracked_bodies]
sp = d.body_link_pos_w[:, ids] - env.env_origins.unsqueeze(1)
sR = quat_to_mat(d.body_link_quat_w[:, ids])
bp, bR, bv, bw = kin.forward(*S)
per_body = (bp - sp).norm(dim=-1).max(0).values
print("FK pos err (max)", per_body.max().item(), {b: round(e, 5) for b, e in zip(robot.tracked_bodies, per_body.tolist())})
print("FK rot err", rot_angle_sq(bR, sR).sqrt().max().item())
sv = d.body_link_vel_w[:, ids]
print("FK linvel err", (bv - sv[..., :3]).norm(dim=-1).max().item(), " angvel err", (bw - sv[..., 3:]).norm(dim=-1).max().item())
print("ref body vs FK(sim) pos err", (motion.body_p[frames] - bp).norm(dim=-1).max().item(), flush=True)
t0 = time.time()
for t in range(150):
    f1 = env.f + 1
    a = (motion.q[f1] - env.default_q) / env.alpha
    out = env.step(a)
    if t % 25 == 0:
        print(t, "norm reward", out["r"].mean().item() / 0.1, "term frac", env.failed_eval.float().mean().item(),
              "e_body_pos", out["terms"]["e_body_pos"].mean().item(), flush=True)  # fmt: skip
Sx = env.state()
d = env.robot.data
print("after steps: |v_state - root_link_lin_vel_w| max", (Sx[2] - d.root_link_lin_vel_w).abs().max().item(),
      " |w_state - root_link_ang_vel_w| max", (Sx[3] - d.root_link_ang_vel_w).abs().max().item(), flush=True)
torch.cuda.synchronize()
print(f"150 control steps x {args.num_envs} envs: {time.time() - t0:.2f}s", flush=True)
sys.stdout.flush()
os._exit(0)  # Kit's app.close() can hang after a standalone run
