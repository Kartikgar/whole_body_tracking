"""Nominal-gain stand attempt in Genesis (URDF, explicit PD torques as in sim2sim_genesis).

    python scripts/sprout/stand_nominal_genesis.py [--cpu]   # genesis env
"""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

import numpy as np
import torch

sys.path.insert(0, str(Path(__file__).resolve().parent))
sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
import stand_common as C  # noqa: E402
from sim2sim_genesis.scene import add_robot_entity  # noqa: E402

spec = C.load_spec()


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--cpu", action="store_true")
    args = ap.parse_args()

    import genesis as gs

    gs.init(backend=gs.cpu if args.cpu else gs.gpu, precision="32", logging_level="warning")
    scene = gs.Scene(
        sim_options=gs.options.SimOptions(dt=C.PHYSICS_DT, substeps=1, gravity=(0.0, 0.0, -9.81)),
        rigid_options=gs.options.RigidOptions(enable_self_collision=False),
        show_viewer=False,
        renderer=gs.renderers.Rasterizer(),
    )
    scene.add_entity(gs.morphs.Plane())
    robot = add_robot_entity(gs, scene, urdf_file=str(spec.SPROUT_RENDER_URDF_PATH), xml_file=None)
    cam = scene.add_camera(res=C.VIDEO_RES, pos=C.CAMERA_EYE, lookat=C.CAMERA_TARGET,
                           fov=C.CAMERA_FOV_DEG, GUI=False)
    scene.build()

    dofs = [int(robot.get_joint(n).dofs_idx_local[0]) for n in spec.JOINT_NAMES]
    n = len(dofs)
    kp = torch.tensor([spec.JOINT_STIFFNESS[j] for j in spec.JOINT_NAMES], dtype=torch.float32)
    kd = torch.tensor([spec.JOINT_DAMPING[j] for j in spec.JOINT_NAMES], dtype=torch.float32)
    effort = np.array([spec.EFFORT_LIMIT[j] for j in spec.JOINT_NAMES], dtype=np.float32)
    target = torch.tensor([spec.DEFAULT_STAND_POSE[j] for j in spec.JOINT_NAMES], dtype=torch.float32)

    robot.set_dofs_armature(torch.full((n,), spec.ARMATURE), dofs)
    robot.set_dofs_damping(torch.full((n,), spec.PASSIVE_DAMPING), dofs)  # MJCF passive damping
    robot.set_dofs_force_range(torch.from_numpy(-effort), torch.from_numpy(effort), dofs)
    robot.set_pos(torch.tensor([0.0, 0.0, spec.STAND_ROOT_HEIGHT]))
    robot.set_quat(torch.tensor([1.0, 0.0, 0.0, 0.0]))
    robot.set_dofs_position(target, dofs)
    robot.zero_all_dofs_velocity()

    def np_(x):
        return x.cpu().numpy().squeeze() if hasattr(x, "cpu") else np.asarray(x).squeeze()

    t, pos, quat, q, frames = [], [], [], [], []
    for k in range(int(round(C.STAND_DURATION_S / C.PHYSICS_DT))):
        if k % C.RENDER_EVERY == 0:
            frames.append(cam.render()[0])
        jq = robot.get_dofs_position(dofs).reshape(-1).to(kp.device)
        jv = robot.get_dofs_velocity(dofs).reshape(-1).to(kp.device)
        tau = kp * (target - jq) - kd * jv
        robot.control_dofs_force(tau, dofs)
        scene.step()
        t.append((k + 1) * C.PHYSICS_DT)
        pos.append(np_(robot.get_pos()))
        quat.append(np_(robot.get_quat()))
        q.append(np_(robot.get_dofs_position(dofs)))
    C.save_log("genesis", t, pos, quat, q, frames)


if __name__ == "__main__":
    main()
