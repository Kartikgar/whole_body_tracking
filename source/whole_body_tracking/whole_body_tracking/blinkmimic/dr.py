"""Simulator domain randomization and the layout of the sim context xi.

Every randomized simulator parameter is part of xi, which the world model and critic see (the
policy does not). The defaults are the paper's G1 setup: friction once per env, joint (encoder)
offsets per episode, and a CoM offset of ``RobotSpec.com_body`` once per env. A robot can switch on
more terms in ``RobotSpec.dr``; each is applied with the corresponding mjlab ``dr`` function where
mjlab has one, and its values are read back from the model into xi.

Isaac port: only DomainRandCfg and XiLayout are kept here; the Isaac randomizer (friction, joint offsets, CoM)
is IsaacDomainRandomizer in env_isaac.py.
"""

from __future__ import annotations

import re
from dataclasses import dataclass, field
from typing import TYPE_CHECKING

import mujoco
import torch

if TYPE_CHECKING:
    from .robot import RobotSpec


@dataclass(frozen=True)
class DomainRandCfg:
    friction_range: tuple[float, float] = (0.3, 1.2)
    """Contact friction, sampled once per env."""
    friction_geoms: str | None = None
    """Regex of the robot geoms that get it (None: every robot geom). Robot geoms win over the
    ground either by MuJoCo's max-of-pair rule (ground at the bottom of the range) or by priority."""
    joint_offset: dict[str, float] = field(default_factory=lambda: {".*": 0.01})
    """Joint regex -> half-range (rad) of the encoder offset, per episode: the policy reads q + bias and
    the PD target moves by -bias, as on a robot whose encoder zero is off."""
    com_mass_range: tuple[float, float] | None = None
    """None: ``com_body`` CoM offset once per env. Otherwise ``com_body`` is randomized per episode by
    pseudo-inertia: mass scale in this range and CoM shift in ``RobotSpec.com_range``."""
    link_mass_range: tuple[float, float] | None = None
    """Per-link mass scale (inertia scaled with it, pseudo-inertia), per episode."""
    link_mass_exclude: tuple[str, ...] = ()
    """Bodies kept out of the link-mass term (``com_body`` always is)."""
    solref_geoms: str | None = None
    solref_timeconst: tuple[float, float] = (0.02, 0.02)
    solref_dampratio: tuple[float, float] = (1.0, 1.0)
    """Contact solref of ``solref_geoms`` (shared across them), per episode."""
    pd_gain_scale: tuple[tuple[tuple[int, ...], tuple[float, float], tuple[float, float]], ...] = ()
    """(actuator ids, kp scale range, kd scale range), per joint and episode."""
    armature_range: tuple[float, float] | None = None
    damping_range: tuple[float, float] | None = None
    """Per-joint scale of the rotor armature / passive joint damping, per episode."""
    push_interval_s: tuple[float, float] | None = None
    push_velocity: tuple[float, ...] = (0.0,) * 6
    """Root velocity kick (x, y, z, roll, pitch, yaw half-ranges) every push_interval_s."""
    reset_pos: tuple[float, float, float] = (0.05, 0.05, 0.01)
    reset_rot: tuple[float, float, float] = (0.1, 0.1, 0.2)
    reset_lin_vel: tuple[float, float, float] = (0.5, 0.5, 0.2)
    reset_ang_vel: tuple[float, float, float] = (0.52, 0.52, 0.78)
    reset_joint_pos: float = 0.1
    reset_joint_vel: float = 0.0
    """Half-ranges of the offsets from the reference state at episode start."""


def _match(patterns: dict[str, float], names, default: float = 0.0) -> list[float]:
    out = []
    for n in names:
        val = default
        for pat, v in patterns.items():
            if re.fullmatch(pat, n):
                val = v
        out.append(val)
    return out


class XiLayout:
    """Slices of xi and its normalization (center, half-range) for the networks."""

    def __init__(self, robot: RobotSpec, device: str):
        c, s = robot.dr, robot.sensing
        J = robot.num_joints
        self.slices: dict[str, slice] = {}
        centers, scales = [], []

        def add(name: str, center: list[float], scale: list[float]):
            n = sum(sl.stop - sl.start for sl in self.slices.values())
            self.slices[name] = slice(n, n + len(center))
            centers.extend(center), scales.extend(scale)

        lo, hi = c.friction_range
        add("friction", [(lo + hi) / 2], [(hi - lo) / 2])
        add("joint_offset", [0.0] * J, [max(h, 1e-6) for h in _match(c.joint_offset, robot.joint_names)])
        add("com", [0.0] * 3, [0.05] * 3)  # fixed scale (the paper setup's normalization)
        if c.com_mass_range is not None:
            add("com_mass", [sum(c.com_mass_range) / 2], [(c.com_mass_range[1] - c.com_mass_range[0]) / 2])
        self.link_bodies: list[str] = []
        if c.link_mass_range is not None:
            m = mujoco.MjModel.from_xml_path(robot.mjcf_path)
            excl = (robot.com_body, *c.link_mass_exclude)
            self.link_bodies = [m.body(b).name for b in range(1, m.nbody)
                                if not any(re.fullmatch(p, m.body(b).name) for p in excl)]  # fmt: skip
            add("link_mass", [sum(c.link_mass_range) / 2] * len(self.link_bodies),
                [(c.link_mass_range[1] - c.link_mass_range[0]) / 2] * len(self.link_bodies))  # fmt: skip
        if c.solref_geoms is not None:
            add("solref", [sum(c.solref_timeconst) / 2, sum(c.solref_dampratio) / 2],
                [max((c.solref_timeconst[1] - c.solref_timeconst[0]) / 2, 1e-6),
                 max((c.solref_dampratio[1] - c.solref_dampratio[0]) / 2, 1e-6)])  # fmt: skip
        if c.pd_gain_scale:
            add("kp", [1.0] * J, [0.2] * J)
            add("kd", [1.0] * J, [0.5] * J)
        if c.armature_range is not None:
            add("armature", [sum(c.armature_range) / 2] * J, [(c.armature_range[1] - c.armature_range[0]) / 2] * J)
        if c.damping_range is not None:
            add("damping", [sum(c.damping_range) / 2] * J, [(c.damping_range[1] - c.damping_range[0]) / 2] * J)
        if s.imu is not None:
            add("imu_tilt", [0.0, 0.0], [max(s.imu.tilt, 1e-6)] * 2)
            add("gyro_bias", [0.0] * 3, [max(s.imu.gyro_bias, 1e-6)] * 3)
        if s.joint_vel_filter is not None:
            lo, hi = s.joint_vel_filter.tau_scale_range
            add("qd_tau_scale", [(lo + hi) / 2], [max((hi - lo) / 2, 1e-6)])
        self.dim = len(centers)
        self.center = torch.tensor(centers, device=device)
        self.scale = torch.tensor(scales, device=device)

    def normalize(self, xi: torch.Tensor) -> torch.Tensor:
        return (xi - self.center) / self.scale

    def get(self, xi: torch.Tensor, name: str) -> torch.Tensor:
        return xi[..., self.slices[name]]
