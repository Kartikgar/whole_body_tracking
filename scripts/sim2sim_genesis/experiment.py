"""Direct, named physical-property overrides for Genesis experiments."""

from dataclasses import asdict, dataclass, field
import math
from pathlib import Path


@dataclass
class ExperimentConfig:
    link_masses_kg: dict[str, float] = field(default_factory=dict)
    joint_stiffness: dict[str, float] = field(default_factory=dict)
    ground_friction: float | None = None
    robot_friction: float | None = None

    def to_dict(self) -> dict:
        return asdict(self)


def load_experiment(path: str | None) -> ExperimentConfig:
    """Load strict YAML; omitted properties retain their existing values."""
    if path is None:
        return ExperimentConfig()
    import yaml

    raw = yaml.safe_load(Path(path).read_text(encoding="utf-8"))
    if not isinstance(raw, dict):
        raise ValueError("Experiment config must be a YAML mapping.")
    unknown = set(raw) - {"link_masses_kg", "joint_stiffness", "ground_friction", "robot_friction"}
    if unknown:
        raise ValueError(f"Unknown experiment fields: {sorted(unknown, key=str)}")

    def number(value, label, positive=False):
        if isinstance(value, bool) or not isinstance(value, (int, float)):
            raise ValueError(f"{label} must be a numeric scalar.")
        value = float(value)
        if not math.isfinite(value) or value < 0 or (positive and value == 0):
            raise ValueError(f"{label} must be finite and {'positive' if positive else 'non-negative'}.")
        return value

    maps = {}
    for key in ("link_masses_kg", "joint_stiffness"):
        values = raw.get(key, {})
        if not isinstance(values, dict):
            raise ValueError(f"{key} must map exact names to values.")
        maps[key] = {}
        for name, value in values.items():
            if not isinstance(name, str) or not name.strip():
                raise ValueError(f"{key} requires non-empty string names.")
            maps[key][name] = number(value, f"{key}.{name}", positive=key == "link_masses_kg")
    friction_values = {}
    for key in ("ground_friction", "robot_friction"):
        value = raw.get(key)
        if value is not None:
            value = number(value, key)
            if not 0.01 <= value <= 5.0:
                raise ValueError(f"{key} must be in [0.01, 5.0], as required by Genesis set_friction.")
        friction_values[key] = value
    return ExperimentConfig(**maps, **friction_values)


def apply_experiment(scene, config: ExperimentConfig) -> None:
    """Apply native physical properties after scene build, identically in all envs."""
    # Resolve every name before mutating any properties. Avoid substring matching.
    links = {link.name: link for link in scene.robot.links}
    joints = {joint.name: joint for joint in scene.robot.joints}
    for name in config.link_masses_kg:
        if name not in links:
            raise ValueError(f"Unknown experiment link {name!r}; available: {sorted(links)}")
        if links[name].is_fixed:
            raise ValueError(f"Cannot override mass of world-fixed link {name!r}.")
    for name in config.joint_stiffness:
        if name not in joints:
            raise ValueError(f"Unknown experiment joint {name!r}; available: {sorted(joints)}")
        if len(joints[name].dofs_idx_local) != 1:
            raise ValueError(f"Passive stiffness requires a single-DoF joint: {name!r}")
    for name, mass in config.link_masses_kg.items():
        links[name].set_mass(mass)
    for name, stiffness in config.joint_stiffness.items():
        scene.robot.set_dofs_stiffness(stiffness, list(joints[name].dofs_idx_local))
    if config.ground_friction is not None:
        scene.ground.set_friction(config.ground_friction)
    if config.robot_friction is not None:
        import numpy as np

        scene.robot.set_friction(config.robot_friction)
        # Startup randomization multiplies geometry friction by per-env ratios.
        # Clear those ratios so the requested absolute coefficient wins everywhere.
        shape = (scene.num_envs, scene.robot.n_links) if scene.num_envs > 1 else (scene.robot.n_links,)
        scene.robot.set_friction_ratio(np.ones(shape, dtype=np.float32))
