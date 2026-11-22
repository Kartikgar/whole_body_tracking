"""Sprout tracking tasks that run a BlinkMimic policy (play / evaluate, delta open-loop, delta finetune).

The policy observation group reproduces BlinkMimic's inputs exactly (see ``mdp/blinkmimic.py``) and the action term
uses BlinkMimic's joint order and action scale, so a BlinkMimic checkpoint plays and PPO-finetunes here through
``BlinkMimicActorCritic``. Motions must be NPZs made by ``scripts/blinkmimic/frax_to_motion_npz.py``.
"""

from __future__ import annotations

import torch

from isaaclab.managers import ObservationGroupCfg as ObsGroup
from isaaclab.managers import ObservationTermCfg as ObsTerm
from isaaclab.managers import RewardTermCfg as RewTerm
from isaaclab.managers import SceneEntityCfg
from isaaclab.managers import TerminationTermCfg as DoneTerm
from isaaclab.utils import configclass
from isaaclab.utils.noise import AdditiveUniformNoiseCfg as Unoise
from isaaclab.utils.noise import NoiseCfg

import whole_body_tracking.tasks.tracking.mdp as mdp
from whole_body_tracking.robots import sprout_spec as SS
from whole_body_tracking.robots.sprout import SPROUT_ACTION_SCALE, SPROUT_CFG
from whole_body_tracking.tasks.tracking.tracking_env_cfg import TrackingEnvCfg

# BlinkMimic policy interface (blinkmimic/sprout.py, the training args of the checkpoints in logs/blinkmimic_isaac)
POLICY_JOINTS = list(SS.MJCF_JOINT_ORDER)
HIST = 10
GOAL_FRAMES = 10
TRACKED_BODIES = [
    "torso_link",  # root / anchor (floating base)
    "pelvis_link",
    "left_femur_upper_link", "left_tibia_link", "left_foot_link",
    "right_femur_upper_link", "right_tibia_link", "right_foot_link",
    "left_humerus_upper_link", "left_radius_link", "left_gripper_lower_link",
    "right_humerus_upper_link", "right_radius_link", "right_gripper_lower_link",
]  # fmt: skip
FEET = ["left_foot_link", "right_foot_link"]
END_EFFECTORS = FEET + ["left_gripper_lower_link", "right_gripper_lower_link"]
# BlinkMimic's uniform observation-noise half-widths (sensing.SensingCfg defaults / obs_noise): q, qd, w, g, a_prev, rel
_J = len(POLICY_JOINTS)
PROPRIO_NOISE = (0.01,) * _J + (0.5,) * _J + (0.2,) * 3 + (0.05,) * 3 + (0.0,) * _J + (0.05,) * 6


def _per_dim_uniform_noise(data: torch.Tensor, cfg: "PerDimUniformNoiseCfg") -> torch.Tensor:
    half = torch.tensor(cfg.half_widths, device=data.device, dtype=data.dtype)
    return data + (2.0 * torch.rand_like(data) - 1.0) * half


@configclass
class PerDimUniformNoiseCfg(NoiseCfg):
    """Additive uniform noise with a half-width per observation dimension (BlinkMimic's ``add_noise``)."""

    func = _per_dim_uniform_noise
    half_widths: tuple[float, ...] = PROPRIO_NOISE


@configclass
class BlinkMimicPolicyObsCfg(ObsGroup):
    """[proprio history (HIST x (3J + 12), oldest first), goal (GOAL_FRAMES x (15 + 2J))] = BlinkMimicActor input."""

    proprio = ObsTerm(func=mdp.blinkmimic_proprio, params={"command_name": "motion", "action_name": "joint_pos"},
                      noise=PerDimUniformNoiseCfg(), history_length=HIST, flatten_history_dim=True)  # fmt: skip
    goal = ObsTerm(func=mdp.blinkmimic_goal,
                   params={"command_name": "motion", "action_name": "joint_pos", "num_frames": GOAL_FRAMES})  # fmt: skip

    def __post_init__(self):
        self.enable_corruption = True
        self.concatenate_terms = True


def _feet_force_term(noise: bool) -> ObsTerm:
    return ObsTerm(func=mdp.feet_contact_force, params={"sensor_cfg": SceneEntityCfg("contact_forces", body_names=FEET)},
                   noise=Unoise(n_min=-0.01, n_max=0.01) if noise else None, scale=0.01)  # fmt: skip


@configclass
class SproutDeltaPolicyObsCfg(ObsGroup):
    """Observations of the frozen delta policy during finetuning (same layout as the open-loop actor)."""

    base_pos_z = ObsTerm(func=mdp.base_pos_z, scale=1.0)
    feet_contact_force = _feet_force_term(noise=False)
    base_lin_vel = ObsTerm(func=mdp.base_lin_vel, scale=2.0)
    base_ang_vel = ObsTerm(func=mdp.base_ang_vel, scale=0.25)
    projected_gravity = ObsTerm(func=mdp.projected_gravity, scale=1.0)
    joint_pos = ObsTerm(func=mdp.joint_pos_rel, scale=1.0)
    joint_vel = ObsTerm(func=mdp.joint_vel_rel, scale=0.05)
    current_action = ObsTerm(func=mdp.current_action, params={"action_buffer_name": "delta_base_actions"}, scale=1.0)

    def __post_init__(self):
        self.enable_corruption = False
        self.concatenate_terms = True
        self.history_length = 0


@configclass
class SproutDeltaOpenLoopPolicyObsCfg(ObsGroup):
    """ASAP-style open-loop delta policy observations."""

    base_pos_z = ObsTerm(func=mdp.base_pos_z, noise=Unoise(n_min=-1.0, n_max=1.0), scale=1.0)
    feet_contact_force = _feet_force_term(noise=True)
    base_lin_vel = ObsTerm(func=mdp.base_lin_vel, scale=2.0)
    base_ang_vel = ObsTerm(func=mdp.base_ang_vel, scale=0.25)
    projected_gravity = ObsTerm(func=mdp.projected_gravity, scale=1.0)
    joint_pos = ObsTerm(func=mdp.joint_pos_rel, scale=1.0)
    joint_vel = ObsTerm(func=mdp.joint_vel_rel, scale=0.05)
    motion_joint_action = ObsTerm(func=mdp.motion_joint_action, params={"command_name": "motion"}, scale=1.0)

    def __post_init__(self):
        self.enable_corruption = True
        self.concatenate_terms = True
        self.history_length = 0


@configclass
class SproutDeltaOpenLoopCriticObsCfg(ObsGroup):
    base_pos_z = ObsTerm(func=mdp.base_pos_z, scale=1.0)
    feet_contact_force = _feet_force_term(noise=False)
    command = ObsTerm(func=mdp.generated_commands, params={"command_name": "motion"})
    motion_anchor_pos_b = ObsTerm(func=mdp.motion_anchor_pos_b, params={"command_name": "motion"})
    motion_anchor_ori_b = ObsTerm(func=mdp.motion_anchor_ori_b, params={"command_name": "motion"})
    body_pos = ObsTerm(func=mdp.robot_body_pos_b, params={"command_name": "motion"})
    body_ori = ObsTerm(func=mdp.robot_body_ori_b, params={"command_name": "motion"})
    base_lin_vel = ObsTerm(func=mdp.base_lin_vel, scale=2.0)
    base_ang_vel = ObsTerm(func=mdp.base_ang_vel, scale=0.25)
    projected_gravity = ObsTerm(func=mdp.projected_gravity, scale=1.0)
    joint_pos = ObsTerm(func=mdp.joint_pos_rel, scale=1.0)
    joint_vel = ObsTerm(func=mdp.joint_vel_rel, scale=0.05)
    motion_joint_action = ObsTerm(func=mdp.motion_joint_action, params={"command_name": "motion"}, scale=1.0)

    def __post_init__(self):
        self.enable_corruption = False
        self.concatenate_terms = True
        self.history_length = 0


def _policy_action_kwargs() -> dict:
    return dict(asset_name="robot", joint_names=POLICY_JOINTS, preserve_order=True, use_default_offset=True,
                scale=dict(SPROUT_ACTION_SCALE))  # fmt: skip


@configclass
class SproutBlinkMimicFlatEnvCfg(TrackingEnvCfg):
    """Base task: the BlinkMimic policy tracks one or more Sprout motions (play / evaluate / PPO finetune)."""

    def __post_init__(self):
        super().__post_init__()
        self.scene.robot = SPROUT_CFG.replace(prim_path="{ENV_REGEX_NS}/Robot")
        self.actions.joint_pos = mdp.JointPositionActionCfg(**_policy_action_kwargs())
        self.observations.policy = BlinkMimicPolicyObsCfg()

        self.commands.motion.anchor_body_name = "torso_link"
        self.commands.motion.body_names = list(TRACKED_BODIES)
        self.commands.motion.debug_vis_goal_relative_to_robot = True
        self.commands.motion.debug_vis_show_current = False
        self.commands.motion.debug_vis_show_goal = True

        # The policy was trained with BlinkMimic's termination (root height / tilt error), not BeyondMimic's.
        self.terminations.anchor_pos = None
        self.terminations.anchor_ori = None
        self.terminations.ee_body_pos = None
        self.terminations.blinkmimic_root = DoneTerm(
            func=mdp.blinkmimic_root_error, params={"command_name": "motion", "height": 0.25, "tilt": 0.8}
        )
        self.rewards.motion_body_pos_global = None
        self.rewards.motion_body_ori_global = None
        self.rewards.undesired_contacts.params["sensor_cfg"] = SceneEntityCfg(
            "contact_forces", body_names=[r"^(?!" + r"$)(?!".join(END_EFFECTORS) + r"$).+$"]
        )
        # BlinkMimic's training randomization: friction per env, joint offsets, torso CoM; no pushes.
        self.events.physics_material.params["static_friction_range"] = (0.3, 1.2)
        self.events.physics_material.params["dynamic_friction_range"] = (0.3, 1.2)
        self.events.physics_material.params["restitution_range"] = (0.0, 0.0)
        self.events.push_robot = None
        self.episode_length_s = 10.0


@configclass
class SproutDeltaAOpenLoopEnvCfg(SproutBlinkMimicFlatEnvCfg):
    """Open-loop joint-delta model: recorded actions + delta should reproduce the next recorded frame."""

    def __post_init__(self):
        super().__post_init__()
        zero = {k: (0.0, 0.0) for k in ("x", "y", "z", "roll", "pitch", "yaw")}
        self.commands.motion.pose_range = dict(zero)
        self.commands.motion.velocity_range = dict(zero)
        self.commands.motion.joint_position_range = (0.0, 0.0)
        self.commands.motion.sample_trajectories = True
        self.commands.motion.equal_trajectory_sampling = True
        self.commands.motion.required_future_steps = 1
        self.observations.policy = SproutDeltaOpenLoopPolicyObsCfg()
        self.observations.critic = SproutDeltaOpenLoopCriticObsCfg()
        self.actions.joint_pos = mdp.DeltaJointPositionActionCfg(
            **_policy_action_kwargs(), motion_command_name="motion", require_motion_action=True
        )
        self.rewards.motion_body_pos = None
        self.rewards.motion_body_ori = None
        self.rewards.joint_limit = None
        self.rewards.undesired_contacts = None
        offset = {"command_name": "motion", "time_offset": 1}
        self.rewards.motion_global_anchor_pos = RewTerm(
            func=mdp.motion_global_anchor_position_error_exp_at_offset, weight=0.5, params={**offset, "std": 0.3})
        self.rewards.motion_global_anchor_ori = RewTerm(
            func=mdp.motion_global_anchor_orientation_error_exp_at_offset, weight=0.5, params={**offset, "std": 0.4})
        self.rewards.motion_body_pos_global = RewTerm(
            func=mdp.motion_global_body_position_error_exp_at_offset, weight=1.0, params={**offset, "std": 0.3})
        self.rewards.motion_body_ori_global = RewTerm(
            func=mdp.motion_global_body_orientation_error_exp_at_offset, weight=1.0, params={**offset, "std": 0.4})
        self.rewards.motion_body_lin_vel = RewTerm(
            func=mdp.motion_global_body_linear_velocity_error_exp_at_offset, weight=1.0, params={**offset, "std": 1.0})
        self.rewards.motion_body_ang_vel = RewTerm(
            func=mdp.motion_global_body_angular_velocity_error_exp_at_offset, weight=1.0, params={**offset, "std": 3.14})
        self.rewards.penalty_minimal_action_norm = RewTerm(func=mdp.penalty_minimal_action_norm, weight=+0.1)
        # As the G1 delta task: end-effector height termination only.
        self.terminations.blinkmimic_root = None
        self.terminations.ee_body_pos = DoneTerm(
            func=mdp.bad_motion_body_pos_z_only,
            params={"command_name": "motion", "threshold": 0.25, "body_names": list(END_EFFECTORS)},
        )
        self.events.physics_material = None
        self.events.add_joint_default_pos = None
        self.events.base_com = None
        self.events.push_robot = None
        self.observations.policy.enable_corruption = False
        self.observations.critic.enable_corruption = False


@configclass
class SproutBlinkMimicDeltaAFineTuneEnvCfg(SproutBlinkMimicFlatEnvCfg):
    """PPO-finetune the BlinkMimic policy with a frozen open-loop joint delta added to its actions."""

    def __post_init__(self):
        super().__post_init__()
        self.actions.joint_pos = mdp.ExternalDeltaJointPositionActionCfg(
            **_policy_action_kwargs(), external_action_buffer_name="delta_external_actions",
            external_action_scale=1.0, require_external_action=True,
        )  # fmt: skip
        self.observations.delta_policy = SproutDeltaPolicyObsCfg()
        self.rewards.penalty_minimal_action_norm = None
