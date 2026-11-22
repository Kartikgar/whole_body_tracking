"""Isaac Lab articulation config for the Fauna Sprout humanoid.

All numbers come from :mod:`whole_body_tracking.robots.sprout_spec`.

Damping convention: MJCF applies passive joint damping (0.1) on top of the position
actuator kv (1.0). Isaac's implicit drive has only one damping term, so it gets
kd + passive = 1.1, which gives the same total damping as the other simulators.
"""

from __future__ import annotations

import copy

import isaaclab.sim as sim_utils
from isaaclab.actuators import ImplicitActuatorCfg
from isaaclab.assets.articulation import ArticulationCfg

from whole_body_tracking.robots import sprout_spec as spec

SPROUT_DRIVE_DAMPING = spec.KD + spec.PASSIVE_DAMPING

# No joint velocity limit (decision 2026-11-09, matching the mjlab/BlinkMimic setup and MuJoCo/Genesis): Isaac would
# otherwise fall back to the URDF placeholder (5 rad/s), which clamps ~24% of retargeted LAFAN1 frames. Leaving
# velocity_limit_sim unset does NOT remove it (the converter copies the URDF value), so set a large value explicitly.
SIM_VELOCITY_LIMIT = 1000.0  # rad/s
# Torque limit kept at the URDF placeholder (80 Nm), as in the mjlab/BlinkMimic Sprout entity.

# Root height when no derived stand pose exists yet (MJCF spawns torso_link at 0.7 m).
_FALLBACK_ROOT_HEIGHT = 0.7


def _actuator_group(joint_names: tuple[str, ...]) -> ImplicitActuatorCfg:
    return ImplicitActuatorCfg(
        joint_names_expr=list(joint_names),
        effort_limit_sim={n: spec.EFFORT_LIMIT[n] for n in joint_names},
        velocity_limit_sim={n: SIM_VELOCITY_LIMIT for n in joint_names},
        stiffness={n: spec.JOINT_STIFFNESS[n] for n in joint_names},
        damping={n: SPROUT_DRIVE_DAMPING for n in joint_names},
        armature={n: spec.ARMATURE for n in joint_names},
    )


SPROUT_CFG = ArticulationCfg(
    spawn=sim_utils.UrdfFileCfg(
        fix_base=False,
        asset_path=str(spec.SPROUT_RENDER_URDF_PATH),  # same physics as SPROUT_URDF_PATH; textures fixed
        activate_contact_sensors=True,
        rigid_props=sim_utils.RigidBodyPropertiesCfg(
            disable_gravity=False,
            retain_accelerations=False,
            linear_damping=0.0,
            angular_damping=0.0,
            max_linear_velocity=1000.0,
            max_angular_velocity=1000.0,
            max_depenetration_velocity=1.0,
        ),
        articulation_props=sim_utils.ArticulationRootPropertiesCfg(
            # no self-collision, as in the mjlab/BlinkMimic Sprout entity (collision geoms only touch the ground)
            enabled_self_collisions=False, solver_position_iteration_count=8, solver_velocity_iteration_count=4
        ),
        joint_drive=sim_utils.UrdfConverterCfg.JointDriveCfg(
            gains=sim_utils.UrdfConverterCfg.JointDriveCfg.PDGainsCfg(stiffness=0, damping=0)
        ),
    ),
    init_state=ArticulationCfg.InitialStateCfg(
        pos=(0.0, 0.0, spec.STAND_ROOT_HEIGHT or _FALLBACK_ROOT_HEIGHT),
        joint_pos=dict(spec.DEFAULT_STAND_POSE),
        joint_vel={".*": 0.0},
    ),
    soft_joint_pos_limit_factor=0.9,
    actuators={
        "lower_body": _actuator_group(spec.LOWER_BODY_JOINTS),
        "upper_body": _actuator_group(spec.UPPER_BODY_JOINTS),
    },
)

# Fixed-base variant for hanging PD probes (step/chirp tests): root raised clear of the ground.
SPROUT_FIXED_CFG = copy.deepcopy(SPROUT_CFG)
SPROUT_FIXED_CFG.spawn.fix_base = True
SPROUT_FIXED_CFG.init_state.pos = (0.0, 0.0, 1.2)

# Same formula as G1_ACTION_SCALE (0.25 * effort / stiffness), fixed to the 80 Nm reference effort so that
# changing the simulated torque limit does not change the action scale (0.615 rad lower body, 2.0 rad upper body).
ACTION_SCALE_REF_EFFORT = 80.0
SPROUT_ACTION_SCALE: dict[str, float] = {
    n: 0.25 * ACTION_SCALE_REF_EFFORT / spec.JOINT_STIFFNESS[n] for n in spec.JOINT_NAMES
}
