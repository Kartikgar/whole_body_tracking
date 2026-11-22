"""Nominal-gain stand attempt in Isaac Lab (URDF, implicit PD via SPROUT_CFG).

Headless, logs only (writes logs/sprout_phase1/stand_nominal/isaac.npz):
    python scripts/sprout/stand_nominal_isaac.py --headless
GUI: real-time viewer, the attempt restarts every STAND_DURATION_S until the window closes:
    python scripts/sprout/stand_nominal_isaac.py
"""

from __future__ import annotations

import argparse
import sys
import time
from pathlib import Path

from isaaclab.app import AppLauncher

parser = argparse.ArgumentParser()
parser.add_argument("--video", action="store_true", help="headless: also record isaac.mp4 (needs --enable_cameras)")
AppLauncher.add_app_launcher_args(parser)
args = parser.parse_args()
app = AppLauncher(args).app

import torch  # noqa: E402

import isaaclab.sim as sim_utils  # noqa: E402
from isaaclab.assets import Articulation  # noqa: E402

sys.path.insert(0, str(Path(__file__).resolve().parent))
import stand_common as C  # noqa: E402

from whole_body_tracking.robots import sprout_spec as spec  # noqa: E402
from whole_body_tracking.robots.sprout import SPROUT_CFG  # noqa: E402


def log(msg: str) -> None:
    print(f"[stand_isaac] {msg}", flush=True)


def reset(robot: Articulation) -> torch.Tensor:
    root = robot.data.default_root_state.clone()
    robot.write_root_pose_to_sim(root[:, :7])
    robot.write_root_velocity_to_sim(torch.zeros_like(root[:, 7:]))
    jpos = robot.data.default_joint_pos.clone()
    robot.write_joint_state_to_sim(jpos, torch.zeros_like(jpos))
    robot.set_joint_position_target(jpos)
    robot.write_data_to_sim()
    robot.reset()
    return jpos


def run_attempt(sim, robot, order, realtime: bool, camera=None, frames: list | None = None):
    jpos = reset(robot)
    t, pos, quat, q = [], [], [], []
    steps = int(round(C.STAND_DURATION_S / C.PHYSICS_DT))
    wall0 = time.perf_counter()
    for k in range(steps):
        robot.set_joint_position_target(jpos)
        robot.write_data_to_sim()
        render = realtime or (camera is not None and k % C.RENDER_EVERY == 0)
        sim.step(render=render)
        robot.update(C.PHYSICS_DT)
        if camera is not None and k % C.RENDER_EVERY == 0:
            camera.update(dt=C.PHYSICS_DT * C.RENDER_EVERY)
            frames.append(camera.data.output["rgb"][0, ..., :3].cpu().numpy().astype("uint8"))
        t.append((k + 1) * C.PHYSICS_DT)
        pos.append(robot.data.root_pos_w[0].cpu().numpy())
        quat.append(robot.data.root_quat_w[0].cpu().numpy())
        q.append(robot.data.joint_pos[0, order].cpu().numpy())
        if realtime:
            lag = t[-1] - (time.perf_counter() - wall0)
            if lag > 0:
                time.sleep(lag)
        elif (k + 1) % 100 == 0:
            log(f"step {k + 1}/{steps}  root z {pos[-1][2]:.3f}  tilt {C.tilt_deg(quat[-1]):.1f} deg")
    return t, pos, quat, q


def main() -> None:
    log("building scene")
    sim = sim_utils.SimulationContext(sim_utils.SimulationCfg(dt=C.PHYSICS_DT, device=args.device))
    sim_utils.GroundPlaneCfg().func("/World/ground", sim_utils.GroundPlaneCfg())
    sim_utils.DomeLightCfg(intensity=2000.0).func("/World/light", sim_utils.DomeLightCfg(intensity=2000.0))
    robot = Articulation(SPROUT_CFG.replace(prim_path="/World/Sprout"))
    camera = None
    if args.headless and args.video:
        import numpy as np
        from isaaclab.sensors import Camera, CameraCfg

        # 45 deg vertical FOV at 4:3, as in MuJoCo/Genesis.
        aperture = 20.955
        focal = (aperture * C.VIDEO_RES[1] / C.VIDEO_RES[0]) / (2 * np.tan(np.radians(C.CAMERA_FOV_DEG) / 2))
        camera = Camera(CameraCfg(
            prim_path="/World/cam", update_period=0, width=C.VIDEO_RES[0], height=C.VIDEO_RES[1],
            data_types=["rgb"],
            spawn=sim_utils.PinholeCameraCfg(focal_length=float(focal), horizontal_aperture=aperture,
                                             clipping_range=(0.05, 50.0)),
        ))
    sim.reset()
    if camera is not None:
        camera.set_world_poses_from_view(torch.tensor([C.CAMERA_EYE], device=sim.device),
                                         torch.tensor([C.CAMERA_TARGET], device=sim.device))
    log("scene built")

    names = robot.joint_names
    assert set(names) == set(spec.JOINT_NAMES), names
    order = [names.index(n) for n in spec.JOINT_NAMES]
    log(f"{robot.num_joints} joints, {robot.num_bodies} bodies, "
        f"mass {robot.root_physx_view.get_masses().sum().item():.3f} kg")

    if args.headless:
        frames: list = []
        t, pos, quat, q = run_attempt(sim, robot, order, realtime=False, camera=camera, frames=frames)
        C.save_log("isaac", t, pos, quat, q, frames=frames)
        return

    sim.set_camera_view(eye=C.CAMERA_EYE, target=C.CAMERA_TARGET)
    attempt = 0
    while app.is_running():
        attempt += 1
        t, pos, quat, q = run_attempt(sim, robot, order, realtime=True)
        log(f"attempt {attempt}: final root z {pos[-1][2]:.3f}, max tilt "
            f"{max(C.tilt_deg(x) for x in quat):.1f} deg")


if __name__ == "__main__":
    main()
    # Kit's app.close() can hang indefinitely after a standalone run (observed here);
    # outputs are already written, so flush and hard-exit.
    sys.stdout.flush()
    sys.stderr.flush()
    import os

    os._exit(0)
