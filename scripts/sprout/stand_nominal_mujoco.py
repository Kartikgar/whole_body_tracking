"""Nominal-gain stand attempt in MuJoCo (Fauna's MJCF position actuators). CPU.

    MUJOCO_GL=glfw python scripts/sprout/stand_nominal_mujoco.py   # genesis env has mujoco
"""

from __future__ import annotations

import sys
from pathlib import Path

import mujoco
import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parent))
import stand_common as C  # noqa: E402

spec = C.load_spec()


def main() -> None:
    m = mujoco.MjModel.from_xml_path(str(spec.SPROUT_SCENE_PATH))
    m.opt.timestep = C.PHYSICS_DT
    d = mujoco.MjData(m)

    d.qpos[0:3] = (0.0, 0.0, spec.STAND_ROOT_HEIGHT)
    d.qpos[3:7] = (1.0, 0.0, 0.0, 0.0)
    qadr = [m.joint(n).qposadr[0] for n in spec.JOINT_NAMES]
    for n, a in zip(spec.JOINT_NAMES, qadr):
        d.qpos[a] = spec.DEFAULT_STAND_POSE[n]
        d.ctrl[m.actuator(n).id] = spec.DEFAULT_STAND_POSE[n]
    mujoco.mj_forward(m, d)

    eye, target = np.array(C.CAMERA_EYE), np.array(C.CAMERA_TARGET)
    fwd = target - eye
    cam = mujoco.MjvCamera()
    cam.type = mujoco.mjtCamera.mjCAMERA_FREE
    cam.lookat[:] = target
    cam.distance = float(np.linalg.norm(fwd))
    cam.azimuth = float(np.degrees(np.arctan2(fwd[1], fwd[0])))
    cam.elevation = float(np.degrees(np.arcsin(fwd[2] / np.linalg.norm(fwd))))
    m.vis.global_.fovy = C.CAMERA_FOV_DEG
    renderer = mujoco.Renderer(m, C.VIDEO_RES[1], C.VIDEO_RES[0])
    opt = mujoco.MjvOption()
    opt.geomgroup[1] = 0  # hide collision geoms (orange, group 1); show textured visuals (group 2)

    t, pos, quat, q, frames = [], [], [], [], []
    for k in range(int(round(C.STAND_DURATION_S / C.PHYSICS_DT))):
        if k % C.RENDER_EVERY == 0:
            renderer.update_scene(d, camera=cam, scene_option=opt)
            frames.append(renderer.render())
        mujoco.mj_step(m, d)
        t.append((k + 1) * C.PHYSICS_DT)
        pos.append(d.qpos[0:3].copy())
        quat.append(d.qpos[3:7].copy())
        q.append(d.qpos[qadr].copy())
    C.save_log("mujoco", t, pos, quat, q, frames)


if __name__ == "__main__":
    main()
