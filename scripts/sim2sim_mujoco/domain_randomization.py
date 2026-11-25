"""Generic per-env domain randomization for the batched MuJoCo evaluator.

Nothing here targets a specific dynamics gap: friction, per-link mass scale, PD gain scale, torso CoM shift and
interval pushes. Tune the ranges below (MuJoCo-only; initial values copied from the Genesis evaluator's constants).
"""

from __future__ import annotations

from dataclasses import asdict, dataclass, field

import numpy as np

from sim2sim_mujoco.scene import MujocoSceneAdapter

# ---- DR ranges (edit here) ----------------------------------------------------------------------------------------
FRICTION_RANGE = (0.3, 1.6)  # floor and robot geoms, one value per env
MASS_SCALE_RANGE = (0.9, 1.8)  # per link, inertia scaled with it
KP_KD_SCALE_RANGE = (0.9, 1.1)  # one kp scale and one kd scale per env
TORSO_COM_RANGE = {"x": (-0.025, 0.025), "y": (-0.05, 0.05), "z": (-0.05, 0.05)}  # m
PUSH_INTERVAL_RANGE_S = (1.0, 3.0)
PUSH_VELOCITY_RANGE = {  # root velocity set at each push: m/s (x, y, z), rad/s (roll, pitch, yaw)
    "x": (-0.5, 0.5),
    "y": (-0.5, 0.5),
    "z": (-0.25, 0.25),
    "roll": (-0.52, 0.52),
    "pitch": (-0.52, 0.52),
    "yaw": (-0.78, 0.78),
}


@dataclass(slots=True)
class MujocoDRConfig:
    friction_range: tuple[float, float] = FRICTION_RANGE
    mass_scale_range: tuple[float, float] = MASS_SCALE_RANGE
    kp_kd_scale_range: tuple[float, float] = KP_KD_SCALE_RANGE
    torso_com_range: dict = field(default_factory=lambda: dict(TORSO_COM_RANGE))
    push_interval_s: tuple[float, float] = PUSH_INTERVAL_RANGE_S
    push_velocity_range: dict = field(default_factory=lambda: dict(PUSH_VELOCITY_RANGE))
    pushes: bool = True

    def to_dict(self) -> dict:
        return asdict(self)


class MujocoDomainRandomizer:
    """Sample per-env model parameters once (startup) and apply interval pushes during rollouts."""

    def __init__(self, scene: MujocoSceneAdapter, rng: np.random.Generator, control_dt: float, cfg: MujocoDRConfig):
        self.scene, self.rng, self.cfg = scene, rng, cfg
        self.control_dt = float(control_dt)
        self.next_push = np.full(scene.num_envs, np.iinfo(np.int64).max, dtype=np.int64)
        self.env_params: list[dict] = []

    def setup(self) -> list[dict]:
        """Randomize each env's model in place; return the sampled parameters per env."""

        import mujoco

        cfg, rng = self.cfg, self.rng
        torso = mujoco.mj_name2id(self.scene.model, mujoco.mjtObj.mjOBJ_BODY, "torso_link")
        for model in self.scene.models:
            # MuJoCo uses the max friction of a contact pair, so set the floor and robot geoms together.
            friction = rng.uniform(*cfg.friction_range)
            model.geom_friction[:, 0] = friction
            mass_scale = rng.uniform(*cfg.mass_scale_range, size=model.nbody - 1)
            model.body_mass[1:] *= mass_scale
            model.body_inertia[1:] *= mass_scale[:, None]
            gain_scale = rng.uniform(*cfg.kp_kd_scale_range, size=2)
            model.actuator_gainprm[:, 0] *= gain_scale[0]
            model.actuator_biasprm[:, 1] *= gain_scale[0]
            model.actuator_biasprm[:, 2] *= gain_scale[1]
            com_shift = np.array([rng.uniform(*cfg.torso_com_range[k]) for k in ("x", "y", "z")])
            if torso >= 0:
                model.body_ipos[torso] += com_shift
            self.env_params.append(dict(friction=float(friction), kp_scale=float(gain_scale[0]),
                                        kd_scale=float(gain_scale[1]), torso_com_shift=com_shift.tolist(),
                                        mass_scale_mean=float(mass_scale.mean()),
                                        total_mass_kg=float(model.body_mass.sum())))
        return self.env_params

    def reset_rollout(self) -> None:
        if self.cfg.pushes:
            self._schedule(np.arange(self.scene.num_envs), 0)

    def _schedule(self, env_ids: np.ndarray, step: int) -> None:
        seconds = self.rng.uniform(*self.cfg.push_interval_s, size=len(env_ids))
        self.next_push[env_ids] = step + np.maximum(1, np.round(seconds / self.control_dt)).astype(np.int64)

    def apply_interval_push(self, step: int, active: np.ndarray) -> None:
        if not self.cfg.pushes:
            return
        due = np.flatnonzero((step >= self.next_push) & active)
        if due.size == 0:
            return
        r = self.cfg.push_velocity_range
        keys = ("x", "y", "z", "roll", "pitch", "yaw")
        velocity = np.stack([self.rng.uniform(*r[k], size=due.size) for k in keys], axis=1)
        self.scene.set_root_velocity(due, velocity)
        self._schedule(due, step)
