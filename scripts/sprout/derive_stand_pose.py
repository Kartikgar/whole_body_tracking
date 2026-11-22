"""Derive a default standing pose for Fauna Sprout in MuJoCo (CPU).

Pose family: symmetric legs parameterised by (hip_pitch, knee); ankle pitch is solved so
the sole is level with the torso upright; arms hang at the sides (ARM_POSE); all other
joints are 0. Each candidate is placed with its feet just touching the ground and held
for ``--hold_s`` seconds.

Why a stiff hold: the hardware PD gains cannot hold any static pose. The ankle needs
roughly m*g*h = 25 kg * 9.81 * 0.6 m ~ 147 Nm/rad of stiffness, and its kp is 32.5, so
the robot tips like an inverted pendulum within ~1 s. The real robot stands only because
its policy actively balances. Candidates are therefore scored while held with the MJCF
gains scaled by ``--hold_kp_scale`` (default 8). How long each pose survives at nominal
gains is also recorded (``nominal_survival_s``), but it is not part of the score.

The best-scoring candidate from a coarse grid plus a local refinement is written to
``robots/sprout_stand_pose.json``.

Run (genesis env has MuJoCo):
    MUJOCO_GL=glfw python scripts/sprout/derive_stand_pose.py --out_dir logs/sprout_phase1/stand_pose
"""

from __future__ import annotations

import argparse
import importlib.util
import json
import sys
from dataclasses import asdict, dataclass
from pathlib import Path

import mujoco
import numpy as np

_REPO = Path(__file__).resolve().parents[2]
_SPEC_FILE = _REPO / "source/whole_body_tracking/whole_body_tracking/robots/sprout_spec.py"


def _load_spec():
    """Import sprout_spec without importing the Isaac-dependent package."""
    s = importlib.util.spec_from_file_location("sprout_spec", _SPEC_FILE)
    mod = importlib.util.module_from_spec(s)
    sys.modules["sprout_spec"] = mod
    s.loader.exec_module(mod)
    return mod


spec = _load_spec()

PHYSICS_DT = 0.005
FALL_TILT_RAD = np.deg2rad(30.0)
FALL_HEIGHT_DROP_M = 0.15

# Zero shoulder roll is a T-pose; +-1.4 rad lowers the arms to the sides without self-contact.
ARM_POSE: dict[str, float] = {
    "left_shoulder_roll_joint": -1.4,
    "right_shoulder_roll_joint": 1.4,
}


@dataclass
class Result:
    hip_pitch: float
    knee: float
    ankle: float
    root_height_init: float
    root_height_final: float
    height_drift: float
    xy_drift: float
    max_tilt_deg: float
    com_offset: float
    track_err: float
    torque_rms: float
    foot_roll_deg: float
    both_feet_contact: bool
    fell: bool
    nominal_survival_s: float
    score: float


class SproutStandSim:
    def __init__(self) -> None:
        self.m = mujoco.MjModel.from_xml_path(str(spec.SPROUT_SCENE_PATH))
        self.m.opt.timestep = PHYSICS_DT
        self.d = mujoco.MjData(self.m)
        self.qadr = {n: self.m.joint(n).qposadr[0] for n in spec.JOINT_NAMES}
        self.act = {self.m.actuator(i).name: i for i in range(self.m.nu)}
        self.torso = self.m.body(spec.ROOT_BODY).id
        self.feet = [self.m.body(spec.LEFT_FOOT_BODY).id, self.m.body(spec.RIGHT_FOOT_BODY).id]
        self.soles = [self.m.body(spec.LEFT_SOLE_FRAME).id, self.m.body(spec.RIGHT_SOLE_FRAME).id]
        self.floor_geom = 0  # scene.xml floor is geom 0
        assert self.m.geom_bodyid[self.floor_geom] == 0
        self._kp0 = self.m.actuator_gainprm[:, 0].copy()
        self._bias_kp0 = self.m.actuator_biasprm[:, 1].copy()

    def set_kp_scale(self, scale: float) -> None:
        """Scale position-actuator kp (gainprm[0] = kp, biasprm[1] = -kp); kv unchanged."""
        self.m.actuator_gainprm[:, 0] = self._kp0 * scale
        self.m.actuator_biasprm[:, 1] = self._bias_kp0 * scale

    # -- kinematics -------------------------------------------------------
    def set_pose(self, pose: dict[str, float], root_z: float) -> None:
        d = self.d
        d.qpos[:] = self.m.qpos0
        d.qvel[:] = 0.0
        d.qpos[0:3] = (0.0, 0.0, root_z)
        d.qpos[3:7] = (1.0, 0.0, 0.0, 0.0)
        for n, q in pose.items():
            d.qpos[self.qadr[n]] = q
        mujoco.mj_forward(self.m, d)

    def foot_min_z(self) -> float:
        zs = []
        for g in range(self.m.ngeom):
            if self.m.geom_bodyid[g] not in self.feet or self.m.geom_contype[g] == 0:
                continue
            mid = self.m.geom_dataid[g]
            v = self.m.mesh_vert[self.m.mesh_vertadr[mid] : self.m.mesh_vertadr[mid] + self.m.mesh_vertnum[mid]]
            w = v @ self.d.geom_xmat[g].reshape(3, 3).T + self.d.geom_xpos[g]
            zs.append(w[:, 2].min())
        return float(min(zs))

    def sole_rpy(self, side: int) -> tuple[float, float]:
        """Roll and pitch (rad) of a sole frame in the world."""
        R = self.d.xmat[self.soles[side]].reshape(3, 3)
        pitch = float(np.arcsin(-np.clip(R[2, 0], -1, 1)))
        roll = float(np.arctan2(R[2, 1], R[2, 2]))
        return roll, pitch

    def leg_pose(self, hip_pitch: float, knee: float, ankle: float) -> dict[str, float]:
        pose = {n: 0.0 for n in spec.JOINT_NAMES}
        pose.update(ARM_POSE)
        for side in ("left", "right"):
            pose[f"{side}_hip_pitch_joint"] = hip_pitch
            pose[f"{side}_knee_joint"] = knee
            pose[f"{side}_ankle_joint"] = ankle
        return pose

    def solve_level_ankle(self, hip_pitch: float, knee: float) -> float:
        """Ankle angle that zeroes left sole pitch (torso upright), by secant iterations."""
        lo, hi = spec.JOINT_LIMITS["left_ankle_joint"]

        def pitch(a: float) -> float:
            self.set_pose(self.leg_pose(hip_pitch, knee, a), 1.0)
            return self.sole_rpy(0)[1]

        a0, a1 = 0.0, -0.2
        f0, f1 = pitch(a0), pitch(a1)
        for _ in range(20):
            if abs(f1) < 1e-6 or f1 == f0:
                break
            a0, a1, f0 = a1, float(np.clip(a1 - f1 * (a1 - a0) / (f1 - f0), lo, hi)), f1
            f1 = pitch(a1)
        return a1

    def place(self, pose: dict[str, float]) -> float:
        """Root height so the lowest foot point is 1 mm above the floor."""
        self.set_pose(pose, 1.0)
        return 1.0 - self.foot_min_z() + 0.001

    def feet_in_contact(self) -> tuple[bool, bool]:
        hit = [False, False]
        for i in range(self.d.ncon):
            c = self.d.contact[i]
            for g in (c.geom1, c.geom2):
                b = self.m.geom_bodyid[g]
                for s, fb in enumerate(self.feet):
                    if b == fb or self.m.body_parentid[b] == fb:
                        hit[s] = True
        return hit[0], hit[1]

    # -- evaluation -------------------------------------------------------
    def _hold(self, pose: dict[str, float], z0: float, hold_s: float, kp_scale: float,
              frames: list | None = None, renderer=None) -> dict:
        """Place the pose, hold its targets for hold_s with kp scaled by kp_scale, return stats."""
        self.set_kp_scale(kp_scale)
        self.set_pose(pose, z0)
        for n, q in pose.items():
            self.d.ctrl[self.act[n]] = q
        target = np.array([pose[n] for n in spec.JOINT_NAMES])
        idx = np.array([self.qadr[n] for n in spec.JOINT_NAMES])
        xy0 = self.d.qpos[0:2].copy()

        steps = int(round(hold_s / PHYSICS_DT))
        max_tilt, errs, taus, fell, k = 0.0, [], [], False, 0
        render_every = max(1, int(round(1 / (30 * PHYSICS_DT))))
        for k in range(steps):
            mujoco.mj_step(self.m, self.d)
            up_z = self.d.xmat[self.torso].reshape(3, 3)[2, 2]
            tilt = float(np.arccos(np.clip(up_z, -1, 1)))
            max_tilt = max(max_tilt, tilt)
            if k >= steps // 2:  # settled half
                errs.append(np.abs(self.d.qpos[idx] - target).mean())
                taus.append(np.sqrt((self.d.actuator_force**2).mean()))
            if renderer is not None and frames is not None and k % render_every == 0:
                renderer.update_scene(self.d, camera="side_camera")
                frames.append(renderer.render())
            if tilt > FALL_TILT_RAD or self.d.qpos[2] < z0 - FALL_HEIGHT_DROP_M:
                fell = True
                break
        self.set_kp_scale(1.0)
        return {
            "fell": fell,
            "survival_s": (k + 1) * PHYSICS_DT,
            "zf": float(self.d.qpos[2]),
            "xy_drift": float(np.linalg.norm(self.d.qpos[0:2] - xy0)),
            "max_tilt_deg": float(np.rad2deg(max_tilt)),
            "track_err": float(np.mean(errs)) if errs else np.inf,
            "torque_rms": float(np.mean(taus)) if taus else np.inf,
            "contact": all(self.feet_in_contact()),
        }

    def evaluate(self, hip_pitch: float, knee: float, hold_s: float, kp_scale: float,
                 frames: list | None = None, renderer=None) -> Result:
        ankle = self.solve_level_ankle(hip_pitch, knee)
        pose = self.leg_pose(hip_pitch, knee, ankle)
        z0 = self.place(pose)
        self.set_pose(pose, z0)
        foot_roll = np.rad2deg(max(abs(self.sole_rpy(0)[0]), abs(self.sole_rpy(1)[0])))
        sole_xy = 0.5 * (self.d.xpos[self.soles[0], :2] + self.d.xpos[self.soles[1], :2])
        com_offset = float(np.linalg.norm(self.d.subtree_com[self.torso, :2] - sole_xy))

        nominal = self._hold(pose, z0, hold_s, 1.0)
        stiff = self._hold(pose, z0, hold_s, kp_scale, frames=frames, renderer=renderer)
        r = Result(
            hip_pitch=hip_pitch, knee=knee, ankle=ankle,
            root_height_init=z0, root_height_final=stiff["zf"],
            height_drift=abs(stiff["zf"] - z0), xy_drift=stiff["xy_drift"],
            max_tilt_deg=stiff["max_tilt_deg"], com_offset=com_offset,
            track_err=stiff["track_err"], torque_rms=stiff["torque_rms"],
            foot_roll_deg=float(foot_roll), both_feet_contact=stiff["contact"], fell=stiff["fell"],
            nominal_survival_s=nominal["survival_s"], score=np.inf,
        )
        fell, contact = r.fell, r.both_feet_contact
        if not fell and contact:
            # metres and radians weighted to comparable scales; torque term breaks ties
            r.score = (
                r.xy_drift + r.height_drift + np.deg2rad(r.max_tilt_deg) + r.com_offset
                + 2.0 * r.track_err + 0.01 * r.torque_rms
            )
        return r


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--out_dir", default="logs/sprout_phase1/stand_pose")
    ap.add_argument("--hold_s", type=float, default=5.0)
    ap.add_argument("--hip_range", type=float, nargs=2, default=(-0.8, 0.0))
    ap.add_argument("--knee_range", type=float, nargs=2, default=(0.0, 1.4))
    ap.add_argument("--grid", type=int, default=9)
    ap.add_argument("--hold_kp_scale", type=float, default=8.0, help="kp multiplier for the scored hold")
    ap.add_argument("--no_video", action="store_true")
    ap.add_argument("--no_write_spec", action="store_true", help="do not write robots/sprout_stand_pose.json")
    args = ap.parse_args()

    out = Path(args.out_dir)
    out.mkdir(parents=True, exist_ok=True)
    sim = SproutStandSim()

    hips = np.linspace(*args.hip_range, args.grid)
    knees = np.linspace(*args.knee_range, args.grid)
    grid = np.full((len(hips), len(knees)), np.nan)
    results: list[Result] = []
    for i, h in enumerate(hips):
        for j, k in enumerate(knees):
            r = sim.evaluate(float(h), float(k), args.hold_s, args.hold_kp_scale)
            results.append(r)
            grid[i, j] = r.score
            print(f"hip {h:+.3f} knee {k:+.3f} ankle {r.ankle:+.3f} -> "
                  f"{'FELL' if r.fell else ''} score {r.score:.4f} tilt {r.max_tilt_deg:.1f} "
                  f"dz {r.height_drift*100:.1f}cm dxy {r.xy_drift*100:.1f}cm com {r.com_offset*100:.1f}cm nominal {r.nominal_survival_s:.2f}s")

    best = min(results, key=lambda r: r.score)
    if not np.isfinite(best.score):
        raise SystemExit("No candidate stood. Widen --hip_range / --knee_range.")

    # Local refinement: shrinking pattern search around the best grid point.
    step_h = (hips[1] - hips[0]) / 2
    step_k = (knees[1] - knees[0]) / 2
    for _ in range(4):
        improved = True
        while improved:
            improved = False
            for dh, dk in ((step_h, 0), (-step_h, 0), (0, step_k), (0, -step_k)):
                r = sim.evaluate(best.hip_pitch + dh, best.knee + dk, args.hold_s, args.hold_kp_scale)
                results.append(r)
                if r.score < best.score - 1e-6:
                    best, improved = r, True
        step_h, step_k = step_h / 2, step_k / 2
    print("BEST", json.dumps(asdict(best), indent=1))

    # Score landscape PNG.
    import matplotlib

    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    fig, ax = plt.subplots(figsize=(6, 5))
    shown = np.where(np.isfinite(grid), grid, np.nan)
    im = ax.imshow(shown, origin="lower", aspect="auto", cmap="viridis_r",
                   extent=(knees[0], knees[-1], hips[0], hips[-1]))
    fig.colorbar(im, ax=ax, label="score (lower is better; blank = fell / lost contact)")
    ax.plot(best.knee, best.hip_pitch, "r*", ms=14, label="selected")
    ax.set_xlabel("knee [rad]")
    ax.set_ylabel("hip pitch [rad]")
    ax.set_title(f"Sprout stand-pose search (MuJoCo, {args.hold_s:.0f} s hold)")
    ax.legend(loc="upper right")
    fig.tight_layout()
    fig.savefig(out / "stand_pose_score.png", dpi=150)

    pose = sim.leg_pose(best.hip_pitch, best.knee, best.ankle)
    # Isaac/Genesis reset height: settled torso height plus 5 mm clearance.
    record = {
        "joint_pos": {n: round(q, 4) for n, q in pose.items()},
        "root_height": round(best.root_height_final + 0.005, 4),
        "source": "scripts/sprout/derive_stand_pose.py (MuJoCo, scene.xml)",
        "physics_dt": PHYSICS_DT,
        "hold_s": args.hold_s,
        "hold_kp_scale": args.hold_kp_scale,
        "metrics": {k: v for k, v in asdict(best).items() if k not in ("hip_pitch", "knee", "ankle")},
        "note": "Derived in simulation; confirm on hardware.",
    }
    (out / "stand_pose.json").write_text(json.dumps(record, indent=2))
    (out / "search_results.json").write_text(json.dumps([asdict(r) for r in results], indent=1))
    if not args.no_write_spec:
        spec.SPROUT_STAND_POSE_PATH.write_text(json.dumps(record, indent=2) + "\n")
        print(f"wrote {spec.SPROUT_STAND_POSE_PATH}")

    if not args.no_video:
        import imageio

        renderer = mujoco.Renderer(sim.m, 480, 640)
        for tag, scale in (("stiff", args.hold_kp_scale), ("nominal", 1.0)):
            frames: list = []
            sim._hold(pose, best.root_height_init, args.hold_s, scale, frames=frames, renderer=renderer)
            path = out / f"stand_pose_hold_{tag}.mp4"
            imageio.mimsave(path, frames, fps=30)
            print(f"wrote {path}")


if __name__ == "__main__":
    main()
