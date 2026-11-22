"""Rebuild a recorded Sprout state in MuJoCo and run static sim-vs-real checks.

    MUJOCO_GL=egl python reconstruct.py recordings/20261121T230359Z [--frame 0] [--video]

Needs robot_rec.npz from convert.py. Writes into <rec_dir>/analysis/:
    pose_<frame>_{front,side,top}.png   rendered reconstruction (kinematic: no dynamics are simulated)
    contacts_<frame>.txt                which collision geoms touch / penetrate the floor
    imu_check.png, sensors.png          IMU consistency, noise and timing plots
    report.json                         all numbers
    replay.mp4                          (--video) the whole recording replayed kinematically

Pose: joints from the recording; torso orientation from the IMU quaternion through the model's imu_site
(on torso_link, rotated 180 deg about z), i.e. R_torso = R_imu @ R_site^T. The robot is then lowered until
its lowest collision geom touches the floor (z = 0). The IMU yaw is arbitrary, so the heading is not meaningful.
"""

from __future__ import annotations

import argparse
import json
import os

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402
import mujoco  # noqa: E402
import numpy as np  # noqa: E402

HERE = os.path.dirname(os.path.abspath(__file__))
DEFAULT_XML = os.path.normpath(os.path.join(HERE, "../../../../sprout-model-v0.1.4/sprout.xml"))
G = 9.81


def quat_to_mat(q):
    w, x, y, z = q / np.linalg.norm(q)
    return np.array([
        [1 - 2 * (y * y + z * z), 2 * (x * y - w * z), 2 * (x * z + w * y)],
        [2 * (x * y + w * z), 1 - 2 * (x * x + z * z), 2 * (y * z - w * x)],
        [2 * (x * z - w * y), 2 * (y * z + w * x), 1 - 2 * (x * x + y * y)],
    ])  # fmt: skip


def build_model(xml: str) -> mujoco.MjModel:
    spec = mujoco.MjSpec.from_file(xml)
    spec.worldbody.add_geom(name="floor", type=mujoco.mjtGeom.mjGEOM_PLANE, size=[0, 0, 0.01], rgba=[0.6, 0.6, 0.6, 1])
    spec.visual.global_.offwidth, spec.visual.global_.offheight = 960, 720
    return spec.compile()


class Reconstructor:
    def __init__(self, m: mujoco.MjModel, joint_names):
        self.m, self.d = m, mujoco.MjData(m)
        hinge = [m.joint(i).name for i in range(m.njnt) if m.jnt_type[i] == mujoco.mjtJoint.mjJNT_HINGE]
        assert list(hinge) == list(joint_names), "recording joint order differs from the MJCF"
        self.floor = m.geom("floor").id
        self.R_site = quat_to_mat(m.site_quat[m.site("imu_site").id])  # site in torso_link frame
        self.col = [g for g in range(m.ngeom) if g != self.floor and (m.geom_contype[g] or m.geom_conaffinity[g])]

    def set(self, q, imu_quat):
        m, d = self.m, self.d
        R = quat_to_mat(imu_quat) @ self.R_site.T
        qt = np.zeros(4)
        mujoco.mju_mat2Quat(qt, R.reshape(-1))
        d.qpos[:3], d.qpos[3:7], d.qpos[7:] = 0, qt, q
        d.qvel[:] = 0
        mujoco.mj_kinematics(m, d)
        dist = self.floor_distances()
        d.qpos[2] -= min(dist.values())
        mujoco.mj_forward(m, d)
        return self.floor_distances()

    def floor_distances(self):
        m, d = self.m, self.d
        mujoco.mj_kinematics(m, d)
        mujoco.mj_collision(m, d) if False else None
        fromto = np.zeros(6)
        return {g: mujoco.mj_geomDistance(m, d, self.floor, g, 10.0, fromto) for g in self.col}

    def geom_label(self, g):
        m = self.m
        name = m.geom(g).name
        return f"{m.body(m.geom_bodyid[g]).name}" + (f"/{name}" if name else f"/geom{g}")


def render(m, d, path, azimuth, elevation, lookat, distance=1.6):
    r = mujoco.Renderer(m, 720, 960)
    cam = mujoco.MjvCamera()
    cam.azimuth, cam.elevation, cam.distance = azimuth, elevation, distance
    cam.lookat[:] = lookat
    r.update_scene(d, cam)
    plt.imsave(path, r.render())
    r.close()


def imu_checks(rec, out_dir):
    """Accelerometer vs quaternion gravity, gyro bias/noise, timing."""
    t, quat, acc, gyro = rec["t"], rec["imu_quat"], rec["imu_acc"], rec["imu_gyro"]
    g_from_quat = np.stack([quat_to_mat(q).T @ np.array([0, 0, G]) for q in quat])  # specific force at rest, imu frame
    ang = np.degrees(np.arccos(np.clip(np.sum(g_from_quat * acc, 1) / (G * np.linalg.norm(acc, axis=1)), -1, 1)))
    dt = np.diff(t) * 1e3
    rep = {
        "acc_mean_imu_frame": acc.mean(0).round(4).tolist(),
        "acc_norm_mean": float(np.linalg.norm(acc, axis=1).mean()),
        "gravity_from_quat_mean_imu_frame": (g_from_quat.mean(0)).round(4).tolist(),
        "acc_vs_quat_angle_deg_mean": float(ang.mean()),
        "acc_std": acc.std(0).round(5).tolist(),
        "gyro_bias_rad_s": gyro.mean(0).round(5).tolist(),
        "gyro_std_rad_s": gyro.std(0).round(5).tolist(),
        "matched_dt_ms": {"mean": float(dt.mean()), "std": float(dt.std()), "min": float(dt.min()), "max": float(dt.max())},
        "timestamp_diff_ms": {"mean": float(rec["timestamp_diff_ms"].mean()), "max": float(np.abs(rec["timestamp_diff_ms"]).max())},
        "stale_fraction": float(rec["has_stale_states"].mean()),
        "joint_pos_std_rad": float(rec["q"].std(0).max()),
    }
    if len(rec["ekf_t"]):
        rep["ekf_gyro_bias_rad_s"] = rec["ekf_gyro"].mean(0).round(5).tolist()
        rep["ekf_gyro_std_rad_s"] = rec["ekf_gyro"].std(0).round(5).tolist()
        rep["ekf_rate_hz"] = float(1 / np.diff(rec["ekf_t"]).mean())
        rep["ekf_vs_matched_quat_angle_deg"] = float(
            np.degrees(2 * np.arccos(min(1.0, abs(float(np.dot(rec["ekf_quat"][0], quat[0]))))))
        )

    fig, ax = plt.subplots(2, 2, figsize=(12, 7))
    for i, c in enumerate("xyz"):
        ax[0, 0].plot(t, acc[:, i], label=f"acc {c}", lw=0.8)
        ax[0, 0].axhline(g_from_quat.mean(0)[i], ls="--", lw=0.8, color=f"C{i}")
    ax[0, 0].set(title="accelerometer (solid) vs gravity from quaternion (dashed), imu frame", xlabel="s", ylabel="m/s²")
    ax[0, 0].legend(fontsize=8)
    ax[0, 1].plot(t, ang, lw=0.8)
    ax[0, 1].set(title="angle between accelerometer and quaternion gravity", xlabel="s", ylabel="deg")
    src_t, src_g = (rec["ekf_t"], rec["ekf_gyro"]) if len(rec["ekf_t"]) else (t, gyro)
    for i, c in enumerate("xyz"):
        ax[1, 0].plot(src_t, src_g[:, i], label=f"gyro {c}", lw=0.5)
    ax[1, 0].set(title="gyro at rest (EKF stream)" if len(rec["ekf_t"]) else "gyro at rest", xlabel="s", ylabel="rad/s")
    ax[1, 0].legend(fontsize=8)
    ax[1, 1].hist(dt, bins=60)
    ax[1, 1].set(title="matched_sensor_data interval", xlabel="ms", ylabel="count")
    fig.tight_layout()
    fig.savefig(os.path.join(out_dir, "imu_check.png"), dpi=120)
    plt.close(fig)

    fig, ax = plt.subplots(1, 2, figsize=(13, 4.5))
    names = [str(n).replace("_joint", "") for n in rec["joint_names"]]
    ax[0].bar(range(27), rec["q"].std(0) * 1e3)
    ax[0].set(title="joint position std at rest", ylabel="mrad")
    ax[1].bar(range(27), rec["effort"].mean(0))
    ax[1].set(title="mean measured effort (damped, lying)", ylabel="Nm")
    for a in ax:
        a.set_xticks(range(27), names, rotation=90, fontsize=7)
    fig.tight_layout()
    fig.savefig(os.path.join(out_dir, "sensors.png"), dpi=120)
    plt.close(fig)
    return rep


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("rec_dir")
    ap.add_argument("--xml", default=DEFAULT_XML)
    ap.add_argument("--frame", type=int, default=0)
    ap.add_argument("--video", action="store_true")
    ap.add_argument("--viewer", action="store_true", help="interactive viewer only (no files written)")
    ap.add_argument("--loop", action="store_true", help="with --viewer: replay the recording in a loop")
    a = ap.parse_args()

    if a.viewer:
        import time

        from mujoco import viewer as mj_viewer

        rec = dict(np.load(os.path.join(a.rec_dir, "robot_rec.npz")))
        m = build_model(a.xml)
        rc = Reconstructor(m, rec["joint_names"])
        rc.set(rec["q"][a.frame], rec["imu_quat"][a.frame])
        print("Viewer: drag to rotate, right-drag to pan, scroll to zoom; Ctrl+right-click a body to inspect.")
        with mj_viewer.launch_passive(m, rc.d) as v:
            t = rec["t"]
            while v.is_running():
                if a.loop:
                    t0 = time.time()
                    for k in range(len(t)):
                        if not v.is_running():
                            break
                        rc.set(rec["q"][k], rec["imu_quat"][k])
                        v.sync()
                        time.sleep(max(0.0, t0 + t[k] - time.time()))
                else:
                    v.sync()
                    time.sleep(0.05)
        return

    rec = dict(np.load(os.path.join(a.rec_dir, "robot_rec.npz")))
    out = os.path.join(a.rec_dir, "analysis")
    os.makedirs(out, exist_ok=True)
    m = build_model(a.xml)
    rc = Reconstructor(m, rec["joint_names"])

    f = a.frame
    dist = rc.set(rec["q"][f], rec["imu_quat"][f])
    d = rc.d
    lim = m.jnt_range[1:]
    q = rec["q"][f]
    near_limit = {str(n): round(float(v), 3) for n, v, (lo, hi) in zip(rec["joint_names"], q, lim)
                  if v < lo + 0.05 or v > hi - 0.05}  # fmt: skip
    touching = sorted(((v, rc.geom_label(g)) for g, v in dist.items() if v < 0.01))
    with open(os.path.join(out, f"contacts_{f}.txt"), "w") as fh:
        fh.write("signed distance to floor (m) of collision geoms within 1 cm, after lowering to touch\n")
        for v, lab in touching:
            fh.write(f"{v:+.4f}  {lab}\n")
    R_torso = d.xmat[m.body("torso_link").id].reshape(3, 3)
    lookat = d.subtree_com[0].copy()
    for name, az, el in (("front", 180, -15), ("side", 90, -15), ("top", 90, -89)):
        render(m, d, os.path.join(out, f"pose_{f}_{name}.png"), az, el, lookat)

    rep = {
        "frame": f, "t_s": float(rec["t"][f]),
        "torso_height_m": float(d.qpos[2]),
        "torso_up_axis_in_world": R_torso[:, 2].round(3).tolist(),
        "torso_forward_axis_in_world": R_torso[:, 0].round(3).tolist(),
        "joints_within_0.05rad_of_limit": near_limit,
        "geoms_touching_floor": [lab for _, lab in touching],
        **imu_checks(rec, out),
    }  # fmt: skip
    json.dump(rep, open(os.path.join(out, "report.json"), "w"), indent=1)
    print(json.dumps(rep, indent=1))

    if a.video:
        import imageio

        r = mujoco.Renderer(m, 480, 640)
        cam = mujoco.MjvCamera()
        cam.azimuth, cam.elevation, cam.distance = 135, -20, 1.8
        w = imageio.get_writer(os.path.join(out, "replay.mp4"), fps=25)
        for k in range(0, len(rec["t"]), 2):
            rc.set(rec["q"][k], rec["imu_quat"][k])
            cam.lookat[:] = rc.d.subtree_com[0]
            r.update_scene(rc.d, cam)
            w.append_data(r.render())
        w.close()
        r.close()


if __name__ == "__main__":
    main()
