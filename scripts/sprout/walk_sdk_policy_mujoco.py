"""Run Fauna's SDK example walk policy (examples/onnx_motor_control/model/policy.onnx) in MuJoCo.

The policy takes raw sensor data in hardware ("control") joint order -- joint positions/velocities, IMU quaternion
(wxyz, world frame) and IMU angular velocity (IMU frame), and 3 joystick axes [left_y, left_x, right_x] in [-1, 1] --
and returns motor position targets plus per-joint kp/kd. Those gains are written into Fauna's MJCF position actuators
every control step, as the robot's motor driver would apply them.

    # genesis env has mujoco + onnxruntime
    MUJOCO_GL=egl python scripts/sprout/walk_sdk_policy_mujoco.py --schedule "2:0,0,0 6:1,0,0 2:0,0,0" --video out.mp4

--schedule is a list of "<seconds>:<a0>,<a1>,<a2>" segments (joystick axes held for that long).
"""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

import mujoco
import numpy as np
import onnxruntime as ort

sys.path.insert(0, str(Path(__file__).resolve().parent))
import stand_common as C  # noqa: E402

spec = C.load_spec()
REPO = Path(__file__).resolve().parents[3]
POLICY_DIR = REPO / "fauna-sdk-v0.10.0/fauna-sdk/examples/onnx_motor_control/model"


def quat_mul(a, b):
    w1, x1, y1, z1 = a
    w2, x2, y2, z2 = b
    return np.array([w1 * w2 - x1 * x2 - y1 * y2 - z1 * z2, w1 * x2 + x1 * w2 + y1 * z2 - z1 * y2,
                     w1 * y2 - x1 * z2 + y1 * w2 + z1 * x2, w1 * z2 + x1 * y2 - y1 * x2 + z1 * w2])


def describe(axes: np.ndarray) -> str:
    """Human-readable joystick command (axis signs measured in MuJoCo: +a0 forward, +a1 left, +a2 turn left / CCW, ~75 deg/s at 1.0)."""
    parts = []
    for v, pos, neg in zip(axes, ("forward", "step left", "turn left"), ("backward", "step right", "turn right")):
        if abs(v) > 1e-3:
            parts.append(f"{pos if v > 0 else neg} {abs(v):.1f}")
    return ", ".join(parts) if parts else "stand"


def parse_schedule(s: str) -> list[tuple[float, np.ndarray]]:
    out = []
    for seg in s.split():
        dur, ax = seg.split(":")
        out.append((float(dur), np.array([float(v) for v in ax.split(",")], dtype=np.float32)))
    return out


class SdkPolicy:
    def __init__(self, model_dir: Path = POLICY_DIR):
        so = ort.SessionOptions()
        so.log_severity_level = 3
        so.intra_op_num_threads = 1
        so.inter_op_num_threads = 1
        self.sess = ort.InferenceSession(str(model_dir / "policy.onnx"), so, providers=["CPUExecutionProvider"])
        carry = json.load(open(model_dir / "settings.json"))["carry"]
        self.carry0 = {k: np.zeros(v, np.float32) for k, v in carry.items()}
        self.reset()

    def reset(self):
        self.carry = {k: v.copy() for k, v in self.carry0.items()}

    def __call__(self, q_hw, dq_hw, imu_quat, gyro, axes):
        f = {"joint_positions_control": q_hw[None].astype(np.float32),
             "joint_velocities_control": dq_hw[None].astype(np.float32),
             "imu_quat": imu_quat[None].astype(np.float32), "base_ang_vel": gyro[None].astype(np.float32),
             "joystick_axes": axes[None].astype(np.float32), **self.carry}
        names = [o.name for o in self.sess.get_outputs()]
        o = dict(zip(names, self.sess.run(names, f)))
        for k in self.carry:
            self.carry[k] = o[f"updated_{k}"]
        return o["motor_targets_control"][0], o["stiffness_control"][0], o["damping_control"][0], o["est_observations"][0]


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--schedule", default="3:0,0,0 6:1,0,0 3:0,0,0")
    ap.add_argument("--control_hz", type=float, default=50.0)
    ap.add_argument("--physics_dt", type=float, default=0.002)
    ap.add_argument("--imu_frame", choices=("site", "torso"), default="site", help="site = MJCF imu_site (yawed 180 deg)")
    ap.add_argument("--video", default=None)
    ap.add_argument("--fps", type=int, default=50)
    ap.add_argument("--quiet", action="store_true")
    ap.add_argument("--label", action="store_true", help="draw time and joystick command on the video")
    a = ap.parse_args()

    m = mujoco.MjModel.from_xml_path(str(spec.SPROUT_SCENE_PATH))
    m.opt.timestep = a.physics_dt
    d = mujoco.MjData(m)
    decim = int(round(1.0 / (a.control_hz * a.physics_dt)))

    hw = list(spec.HARDWARE_JOINT_ORDER)
    qadr = np.array([m.joint(n).qposadr[0] for n in hw])
    dadr = np.array([m.joint(n).dofadr[0] for n in hw])
    act = np.array([m.actuator(n).id for n in hw])
    site = m.site("imu_site").id
    site_q_local = m.site_quat[site].copy()

    d.qpos[0:3] = (0.0, 0.0, spec.STAND_ROOT_HEIGHT)
    d.qpos[3:7] = (1.0, 0.0, 0.0, 0.0)
    for n, i in zip(hw, qadr):
        d.qpos[i] = spec.DEFAULT_STAND_POSE[n]
    d.ctrl[act] = d.qpos[qadr]
    mujoco.mj_forward(m, d)

    renderer = cam = opt = None
    frames = []
    if a.video:
        m.vis.global_.offwidth, m.vis.global_.offheight = 960, 540
        renderer = mujoco.Renderer(m, 540, 960)
        cam = mujoco.MjvCamera()
        cam.distance, cam.azimuth, cam.elevation = 2.2, 135.0, -12.0
        opt = mujoco.MjvOption()
        opt.geomgroup[1] = 0
    render_every = int(round(1.0 / (a.fps * a.physics_dt)))

    pol = SdkPolicy()
    sched = parse_schedule(a.schedule)
    log = []
    k = 0
    for seg_i, (dur, axes) in enumerate(sched):
        p0, yaw0 = d.qpos[0:3].copy(), None
        n_steps = int(round(dur / a.physics_dt))
        for s in range(n_steps):
            if k % decim == 0:
                q_torso = d.qpos[3:7].copy()
                w_body = d.qvel[3:6].copy()  # free-joint angular velocity is in the body (torso) frame
                if a.imu_frame == "site":
                    imu_q = quat_mul(q_torso, site_q_local)
                    R = np.zeros(9)
                    mujoco.mju_quat2Mat(R, site_q_local)
                    gyro = R.reshape(3, 3).T @ w_body
                else:
                    imu_q, gyro = q_torso, w_body
                tgt, kp, kd, est = pol(d.qpos[qadr], d.qvel[dadr], imu_q, gyro, axes)
                d.ctrl[act] = tgt
                m.actuator_gainprm[act, 0] = kp
                m.actuator_biasprm[act, 1] = -kp
                m.actuator_biasprm[act, 2] = -kd
            if renderer is not None and k % render_every == 0:
                cam.lookat[:] = [d.qpos[0], d.qpos[1], 0.45]
                renderer.update_scene(d, camera=cam, scene_option=opt)
                img = renderer.render()
                if a.label:
                    from PIL import Image, ImageDraw, ImageFont

                    im = Image.fromarray(img)
                    dr = ImageDraw.Draw(im)
                    font = ImageFont.load_default(size=30)
                    txt = f"t={k * a.physics_dt:4.1f}s  {describe(axes)}"
                    bb = dr.textbbox((16, 12), txt, font=font)
                    dr.rectangle((bb[0] - 6, bb[1] - 4, bb[2] + 6, bb[3] + 4), fill=(255, 255, 255))
                    dr.text((16, 12), txt, fill=(20, 20, 20), font=font)
                    img = np.asarray(im)
                frames.append(img)
            mujoco.mj_step(m, d)
            k += 1
        # per-segment summary: displacement in the heading frame at segment start
        tilt = C.tilt_deg(d.qpos[3:7])
        dp = d.qpos[0:3] - p0
        w, x, y, z = d.qpos[3:7]
        yaw = np.degrees(np.arctan2(2 * (w * z + x * y), 1 - 2 * (y * y + z * z)))
        log.append(dict(seg=seg_i, axes=axes.tolist(), dur=dur, dx=float(dp[0]), dy=float(dp[1]), vx=float(dp[0] / dur),
                        vy=float(dp[1] / dur), yaw_deg=float(yaw), height=float(d.qpos[2]), tilt_deg=tilt,
                        est=np.round(est, 3).tolist()))
        if not a.quiet:
            print(f"seg {seg_i} axes={axes.tolist()} {dur:.0f}s: dx={dp[0]:+.2f} dy={dp[1]:+.2f} m  yaw={yaw:+.0f}deg "
                  f"h={d.qpos[2]:.3f} tilt={tilt:.0f}deg est={np.round(est, 2).tolist()}")
        if tilt > 60:
            print("FELL")
            break
    if a.video and frames:
        import imageio

        imageio.mimsave(a.video, frames, fps=a.fps, macro_block_size=1)
        print("wrote", a.video)


if __name__ == "__main__":
    main()
