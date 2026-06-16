from isaaclab.managers import ObservationGroupCfg as ObsGroup
from isaaclab.managers import SceneEntityCfg
from isaaclab.managers import ObservationTermCfg as ObsTerm
from isaaclab.managers import RewardTermCfg as RewTerm
from isaaclab.utils import configclass
from isaaclab.utils.noise import AdditiveUniformNoiseCfg as Unoise

from whole_body_tracking.robots.g1 import G1_ACTION_SCALE, G1_CYLINDER_CFG
import whole_body_tracking.tasks.tracking.mdp as mdp
from whole_body_tracking.tasks.tracking.config.g1.agents.rsl_rl_ppo_cfg import LOW_FREQ_SCALE
from whole_body_tracking.tasks.tracking.tracking_env_cfg import TrackingEnvCfg


@configclass
class DeltaPolicyObsCfg(ObsGroup):
    """Observation group consumed by the frozen delta policy during finetuning."""

    # Match open-loop delta-policy actor observation layout/scales.
    base_pos_z = ObsTerm(func=mdp.base_pos_z, scale=1.0)
    feet_contact_force = ObsTerm(
        func=mdp.feet_contact_force,
        params={
            "sensor_cfg": SceneEntityCfg(
                "contact_forces", body_names=["left_ankle_roll_link", "right_ankle_roll_link"]
            )
        },
        scale=0.01,
    )
    base_lin_vel = ObsTerm(func=mdp.base_lin_vel, scale=2.0)
    base_ang_vel = ObsTerm(func=mdp.base_ang_vel, scale=0.25)
    projected_gravity = ObsTerm(func=mdp.projected_gravity, scale=1.0)
    joint_pos = ObsTerm(func=mdp.joint_pos_rel, scale=1.0)
    joint_vel = ObsTerm(func=mdp.joint_vel_rel, scale=0.05)
    # `actions`: frozen delta-policy action history channel (autoregressive).
    # actions = ObsTerm(func=mdp.external_delta_action, params={"action_buffer_name": "delta_external_actions"}, scale=1.0)
    # `current_action`: same-step policy-being-finetuned rollout action (injected by runner).
    current_action = ObsTerm(func=mdp.current_action, params={"action_buffer_name": "delta_base_actions"}, scale=1.0)

    def __post_init__(self):
        self.enable_corruption = False
        self.concatenate_terms = True
        self.history_length = 0


@configclass
class DeltaOpenLoopPolicyObsCfg(ObsGroup):
    """ASAP-style open-loop delta policy observations."""

    base_pos_z = ObsTerm(func=mdp.base_pos_z, noise=Unoise(n_min=-1.0, n_max=1.0), scale=1.0)
    feet_contact_force = ObsTerm(
        func=mdp.feet_contact_force,
        params={
            "sensor_cfg": SceneEntityCfg(
                "contact_forces", body_names=["left_ankle_roll_link", "right_ankle_roll_link"]
            )
        },
        noise=Unoise(n_min=-0.01, n_max=0.01),
        scale=0.01,
    )
    base_lin_vel = ObsTerm(func=mdp.base_lin_vel, scale=2.0)
    base_ang_vel = ObsTerm(func=mdp.base_ang_vel, scale=0.25)
    projected_gravity = ObsTerm(func=mdp.projected_gravity, scale=1.0)
    joint_pos = ObsTerm(func=mdp.joint_pos_rel, scale=1.0)
    joint_vel = ObsTerm(func=mdp.joint_vel_rel, scale=0.05)
    # actions = ObsTerm(func=mdp.last_action, scale=1.0)
    motion_joint_action = ObsTerm(func=mdp.motion_joint_action, params={"command_name": "motion"}, scale=1.0)

    def __post_init__(self):
        self.enable_corruption = True
        self.concatenate_terms = True
        self.history_length = 0


@configclass
class DeltaOpenLoopCriticObsCfg(ObsGroup):
    base_pos_z = ObsTerm(func=mdp.base_pos_z, scale=1.0)
    feet_contact_force = ObsTerm(
        func=mdp.feet_contact_force,
        params={
            "sensor_cfg": SceneEntityCfg(
                "contact_forces", body_names=["left_ankle_roll_link", "right_ankle_roll_link"]
            )
        },
        scale=0.01,
    )
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
    # actions = ObsTerm(func=mdp.last_action, scale=1.0)
    motion_joint_action = ObsTerm(func=mdp.motion_joint_action, params={"command_name": "motion"}, scale=1.0)

    def __post_init__(self):
        self.enable_corruption = False
        self.concatenate_terms = True
        self.history_length = 0


@configclass
class G1FlatEnvCfg(TrackingEnvCfg):
    def __post_init__(self):
        super().__post_init__()

        self.scene.robot = G1_CYLINDER_CFG.replace(prim_path="{ENV_REGEX_NS}/Robot")
        self.actions.joint_pos.scale = G1_ACTION_SCALE
        self.commands.motion.debug_vis_goal_relative_to_robot = True
        self.commands.motion.debug_vis_show_current = False
        self.commands.motion.debug_vis_show_goal = True
        self.commands.motion.anchor_body_name = "torso_link"
        self.rewards.motion_body_pos_global = None
        self.rewards.motion_body_ori_global = None
        # self.commands.motion.body_names = [
        #     "pelvis",
        #     "left_hip_pitch_link",
        #     "left_hip_roll_link",
        #     "left_hip_yaw_link",
        #     "left_knee_link",
        #     "left_ankle_pitch_link",
        #     "left_ankle_roll_link",
        #     "right_hip_pitch_link",
        #     "right_hip_roll_link",
        #     "right_hip_yaw_link",
        #     "right_knee_link",
        #     "right_ankle_pitch_link",
        #     "right_ankle_roll_link",
        #     "waist_yaw_link",
        #     "waist_roll_link",
        #     "torso_link",
        #     "left_shoulder_pitch_link",
        #     "left_shoulder_roll_link",
        #     "left_shoulder_yaw_link",
        #     "left_elbow_link",
        #     "left_wrist_roll_link",
        #     "left_wrist_pitch_link",
        #     "left_wrist_yaw_link",
        #     "right_shoulder_pitch_link",
        #     "right_shoulder_roll_link",
        #     "right_shoulder_yaw_link",
        #     "right_elbow_link",
        #     "right_wrist_roll_link",
        #     "right_wrist_pitch_link",
        #     "right_wrist_yaw_link",
        # ]
        self.commands.motion.body_names = [
            "pelvis",
            "left_hip_roll_link",
            "left_knee_link",
            "left_ankle_roll_link",
            "right_hip_roll_link",
            "right_knee_link",
            "right_ankle_roll_link",
            "torso_link",
            "left_shoulder_roll_link",
            "left_elbow_link",
            "left_wrist_yaw_link",
            "right_shoulder_roll_link",
            "right_elbow_link",
            "right_wrist_yaw_link",
        ]
        # self.terminations.ee_body_pos = None
        # self.terminations.anchor_pos = None
        self.episode_length_s = 10.0

@configclass
class G1FlatDeltaAOpenLoopEnvCfg(G1FlatEnvCfg):
    def __post_init__(self):
        super().__post_init__()
        self.commands.motion.pose_range = {
            "x": (0.0, 0.0),
            "y": (0.0, 0.0),
            "z": (0.0, 0.0),
            "roll": (0.0, 0.0),
            "pitch": (0.0, 0.0),
            "yaw": (0.0, 0.0),
        }
        self.commands.motion.velocity_range = {
            "x": (0.0, 0.0),
            "y": (0.0, 0.0),
            "z": (0.0, 0.0),
            "roll": (0.0, 0.0),
            "pitch": (0.0, 0.0),
            "yaw": (0.0, 0.0),
        }
        self.commands.motion.joint_position_range = (0.0, 0.0)
        self.commands.motion.sample_trajectories = True
        self.commands.motion.equal_trajectory_sampling = True
        self.observations.policy = DeltaOpenLoopPolicyObsCfg()
        self.observations.critic = DeltaOpenLoopCriticObsCfg()
        self.actions.joint_pos = mdp.DeltaJointPositionActionCfg(
            asset_name="robot",
            joint_names=[".*"],
            use_default_offset=True,
            motion_command_name="motion",
            require_motion_action=True,
            # clip={".*": (-10.0, 10.0)},
        )
        self.actions.joint_pos.scale = G1_ACTION_SCALE
        self.commands.motion.body_names = [
            "pelvis",
            "left_hip_pitch_link",
            "left_hip_roll_link",
            "left_hip_yaw_link",
            "left_knee_link",
            "left_ankle_pitch_link",
            "left_ankle_roll_link",
            "right_hip_pitch_link",
            "right_hip_roll_link",
            "right_hip_yaw_link",
            "right_knee_link",
            "right_ankle_pitch_link",
            "right_ankle_roll_link",
            "waist_yaw_link",
            "waist_roll_link",
            "torso_link",
            "left_shoulder_pitch_link",
            "left_shoulder_roll_link",
            "left_shoulder_yaw_link",
            "left_elbow_link",
            "left_wrist_roll_link",
            "left_wrist_pitch_link",
            "left_wrist_yaw_link",
            "right_shoulder_pitch_link",
            "right_shoulder_roll_link",
            "right_shoulder_yaw_link",
            "right_elbow_link",
            "right_wrist_roll_link",
            "right_wrist_pitch_link",
            "right_wrist_yaw_link",
        ]
        # Use global body pose rewards instead of relative-body pose rewards.
        self.rewards.motion_body_pos_global = None
        self.rewards.motion_body_ori_global = None
        self.rewards.penalty_minimal_action_norm = RewTerm(func=mdp.penalty_minimal_action_norm, weight=+0.1)
        # self.terminations.ee_body_pos = None
        self.terminations.anchor_pos = None
        self.terminations.anchor_ori=None
        self.episode_length_s = 10.0


@configclass
class G1FlatDeltaAFineTuneEnvCfg(G1FlatEnvCfg):
    def __post_init__(self):
        super().__post_init__()
        self.actions.joint_pos = mdp.ExternalDeltaJointPositionActionCfg(
            asset_name="robot",
            joint_names=[".*"],
            use_default_offset=True,
            external_action_buffer_name="delta_external_actions",
            external_action_scale=1.0,
            require_external_action=True,
            # external_delta_action_clip=(-10.0, 10.0),
        )
        self.actions.joint_pos.scale = G1_ACTION_SCALE

        self.observations.delta_policy = DeltaPolicyObsCfg()
        self.rewards.penalty_minimal_action_norm = None
        self.rewards.motion_body_pos_global = None
        self.rewards.motion_body_ori_global = None
        self.episode_length_s = 10.0 #1.0
        self.terminations.ee_body_pos = None
        self.terminations.anchor_pos = None


@configclass
class G1FlatWoStateEstimationEnvCfg(G1FlatEnvCfg):
    def __post_init__(self):
        super().__post_init__()
        self.observations.policy.motion_anchor_pos_b = None
        self.observations.policy.base_lin_vel = None


@configclass
class G1FlatLowFreqEnvCfg(G1FlatEnvCfg):
    def __post_init__(self):
        super().__post_init__()
        self.decimation = round(self.decimation / LOW_FREQ_SCALE)
        self.rewards.action_rate_l2.weight *= LOW_FREQ_SCALE
