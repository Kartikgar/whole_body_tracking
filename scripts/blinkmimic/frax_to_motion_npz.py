"""Convert frax-retargeted Sprout clips to BeyondMimic motion NPZs, frame-identical to BlinkMimic's reference.

Each clip goes through BlinkMimic's MotionLib (50 Hz resampling, central-difference root/joint velocities), so the
BeyondMimic tracking tasks see exactly the reference the BlinkMimic policy was trained on. Body poses come from Isaac
forward kinematics of SPROUT_CFG (no physics step); body velocities are central differences of those poses, with the
same formulas MotionLib uses for the root. Joint arrays are written in Isaac joint order, as MotionCommand expects.

    python scripts/blinkmimic/frax_to_motion_npz.py --headless \
        --inputs data/lafan1_frax/sprout_bm40/*.npz --out_dir data/lafan1_frax/sprout_bm40_motion
"""

from __future__ import annotations

import argparse
import glob
import os
import sys

from isaaclab.app import AppLauncher

parser = argparse.ArgumentParser()
parser.add_argument("--inputs", nargs="+", required=True, help="frax npz files (globs allowed)")
parser.add_argument("--out_dir", required=True)
parser.add_argument("--chunk", type=int, default=2048, help="frames posed in parallel (one env per frame)")
AppLauncher.add_app_launcher_args(parser)
args = parser.parse_args()
app = AppLauncher(args).app

import numpy as np  # noqa: E402
import torch  # noqa: E402

import isaaclab.sim as sim_utils  # noqa: E402
from isaaclab.assets import Articulation, ArticulationCfg  # noqa: E402
from isaaclab.scene import InteractiveScene, InteractiveSceneCfg  # noqa: E402
from isaaclab.utils import configclass  # noqa: E402

from whole_body_tracking.blinkmimic.kinematics import Kinematics  # noqa: E402
from whole_body_tracking.blinkmimic.motion import CONTROL_DT, MotionLib, _gradient  # noqa: E402
from whole_body_tracking.blinkmimic.robot import get_robot  # noqa: E402
from whole_body_tracking.blinkmimic.rotations import mat_to_quat, quat_to_mat, so3_log  # noqa: E402
from whole_body_tracking.robots.sprout import SPROUT_CFG  # noqa: E402


@configclass
class _SceneCfg(InteractiveSceneCfg):
    robot: ArticulationCfg = SPROUT_CFG.replace(prim_path="{ENV_REGEX_NS}/Robot")


def angular_velocity(R: torch.Tensor) -> torch.Tensor:
    """World angular velocity by central differences of rotations (T, ..., 3, 3), as MotionLib does for the root."""
    w = torch.zeros(R.shape[:-1], dtype=R.dtype)
    w[1:-1] = so3_log(R[2:] @ R[:-2].transpose(-1, -2)) / (2 * CONTROL_DT)
    w[0] = so3_log(R[1] @ R[0].transpose(-1, -2)) / CONTROL_DT
    w[-1] = so3_log(R[-1] @ R[-2].transpose(-1, -2)) / CONTROL_DT
    return w


def main() -> None:
    files = sorted({f for pat in args.inputs for f in glob.glob(pat)})
    assert files, args.inputs
    os.makedirs(args.out_dir, exist_ok=True)
    spec = get_robot("sprout")
    kin = Kinematics(spec, "cpu")

    sim = sim_utils.SimulationContext(sim_utils.SimulationCfg(dt=CONTROL_DT / 4, device="cuda:0"))
    scene = InteractiveScene(_SceneCfg(num_envs=args.chunk, env_spacing=3.0))
    sim.reset()
    robot: Articulation = scene["robot"]
    dev = robot.device
    isaac_joints = list(robot.joint_names)
    to_isaac = [list(spec.joint_names).index(n) for n in isaac_joints]  # spec-order array -> Isaac order
    origins = scene.env_origins
    print(f"[frax_to_motion_npz] {len(files)} clips, {robot.num_bodies} Isaac bodies", flush=True)

    for f in files:
        m = MotionLib([f], kin, "cpu", spec.load_motion)
        T = m.num_frames
        q_isaac = m.q[:, to_isaac]
        qd_isaac = m.qd[:, to_isaac]
        quat = mat_to_quat(m.root_R)
        body_p = torch.zeros(T, robot.num_bodies, 3)
        body_q = torch.zeros(T, robot.num_bodies, 4)
        for s in range(0, T, args.chunk):
            e = min(s + args.chunk, T)
            n = e - s
            ids = torch.arange(n, device=dev)
            pose = torch.cat((m.root_p[s:e].to(dev) + origins[:n], quat[s:e].to(dev)), -1)
            robot.write_root_link_pose_to_sim(pose, env_ids=ids)
            robot.write_root_com_velocity_to_sim(torch.zeros(n, 6, device=dev), env_ids=ids)
            jp = q_isaac[s:e].to(dev)
            robot.write_joint_state_to_sim(jp, torch.zeros_like(jp), env_ids=ids)
            sim.forward()
            robot.update(0.0)
            body_p[s:e] = (robot.data.body_link_pos_w[:n] - origins[:n, None]).cpu()
            body_q[s:e] = robot.data.body_link_quat_w[:n].cpu()
        root_i = robot.body_names.index(spec.tracked_bodies[0])
        err_root = (body_p[:, root_i] - m.root_p).abs().max().item()
        # tracked bodies: Isaac (URDF) FK against BlinkMimic (MJCF) FK
        tb = [robot.body_names.index(b) for b in spec.tracked_bodies]
        err_fk = (body_p[:, tb] - m.body_p).abs().max().item()
        body_v = _gradient(body_p, CONTROL_DT)
        body_w = angular_velocity(quat_to_mat(body_q))
        out = os.path.join(args.out_dir, os.path.basename(f))
        np.savez(
            out, fps=np.array([1.0 / CONTROL_DT]), joint_names=np.array(isaac_joints), body_names=np.array(robot.body_names),
            joint_pos=q_isaac.numpy(), joint_vel=qd_isaac.numpy(), body_pos_w=body_p.numpy(), body_quat_w=body_q.numpy(),
            body_lin_vel_w=body_v.numpy(), body_ang_vel_w=body_w.numpy(),
        )  # fmt: skip
        print(f"  {os.path.basename(f):32s} T={T:6d}  root err {err_root:.1e}  tracked-body FK err {err_fk:.1e} m", flush=True)
        assert err_root < 1e-4 and err_fk < 2e-3, "Isaac FK disagrees with BlinkMimic's MJCF FK"


if __name__ == "__main__":
    main()
    sys.stdout.flush()
    os._exit(0)  # Kit's app.close() can hang after a standalone run
