import copy

import isaaclab.terrains as terrain_gen
from isaaclab.managers import EventTermCfg as EventTerm
from isaaclab.managers import ObservationGroupCfg as ObsGroup
from isaaclab.managers import SceneEntityCfg
from isaaclab.managers import ObservationTermCfg as ObsTerm
from isaaclab.managers import RewardTermCfg as RewTerm
from isaaclab.managers import TerminationTermCfg as DoneTerm
from isaaclab.sensors import RayCasterCfg, patterns
from isaaclab.utils import configclass
from isaaclab.utils.noise import AdditiveUniformNoiseCfg as Unoise

from whole_body_tracking.robots.g1 import G1_ACTION_SCALE, G1_CYLINDER_CFG
import whole_body_tracking.tasks.tracking.mdp as mdp
from whole_body_tracking.tasks.tracking.config.g1.agents.rsl_rl_ppo_cfg import LOW_FREQ_SCALE
from whole_body_tracking.tasks.tracking.terrains import GapCourseTerrainCfg
from whole_body_tracking.tasks.tracking.tracking_env_cfg import (
    HighLevelScanCfg,
    TrackingEnvCfg,
    resolve_scan_grid_config,
)

_FEET_CONTACT_SENSOR_CFG = SceneEntityCfg(
    "contact_forces",
    body_names=["left_ankle_roll_link", "right_ankle_roll_link"],
)


def _make_feet_contact_force_obs_term(scale: float = 0.01, noise: Unoise | None = None) -> ObsTerm:
    term_kwargs = {
        "func": mdp.feet_contact_force,
        "params": {"sensor_cfg": _FEET_CONTACT_SENSOR_CFG},
        "scale": scale,
    }
    if noise is not None:
        term_kwargs["noise"] = noise
    return ObsTerm(**term_kwargs)


def _make_low_level_obs_from_flat_policy(
    flat_policy_obs_cfg: ObsGroup,
    action_buffer_name: str,
    motion_command_name: str,
) -> ObsGroup:
    """Reuse flat-policy observation config for frozen low-level policies.

    The `actions` channel is replaced with per-policy external action history so each
    low-level policy receives its own autoregressive action state. Motion-reference
    terms are retargeted to the provided motion command name.
    """
    low_level_obs = copy.deepcopy(flat_policy_obs_cfg)
    low_level_obs.enable_corruption = False
    low_level_obs.history_length = 0

    if not hasattr(low_level_obs, "actions"):
        raise RuntimeError("Flat-policy observation config has no `actions` term to override for low-level policy.")

    low_level_obs.actions.func = mdp.external_delta_action
    low_level_obs.actions.params = {"action_buffer_name": action_buffer_name}

    for obs_term_cfg in vars(low_level_obs).values():
        params = getattr(obs_term_cfg, "params", None)
        if isinstance(params, dict) and params.get("command_name") == "motion":
            updated_params = dict(params)
            updated_params["command_name"] = motion_command_name
            obs_term_cfg.params = updated_params

    return low_level_obs


@configclass
class DeltaPolicyObsCfg(ObsGroup):
    """Observation group consumed by the frozen delta policy during finetuning."""

    # Match open-loop delta-policy actor observation layout/scales.
    base_pos_z = ObsTerm(func=mdp.base_pos_z, scale=1.0)
    feet_contact_force = _make_feet_contact_force_obs_term(scale=0.01)
    base_lin_vel = ObsTerm(func=mdp.base_lin_vel, scale=2.0)
    base_ang_vel = ObsTerm(func=mdp.base_ang_vel, scale=0.25)
    projected_gravity = ObsTerm(func=mdp.projected_gravity, scale=1.0)
    joint_pos = ObsTerm(func=mdp.joint_pos_rel, scale=1.0)
    joint_vel = ObsTerm(func=mdp.joint_vel_rel, scale=0.05)
    # `actions`: frozen delta-policy action history channel (autoregressive).
    actions = ObsTerm(func=mdp.external_delta_action, params={"action_buffer_name": "delta_external_actions"}, scale=1.0)
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
    feet_contact_force = _make_feet_contact_force_obs_term(
        scale=0.01,
        noise=Unoise(n_min=-0.01, n_max=0.01),
    )
    base_lin_vel = ObsTerm(func=mdp.base_lin_vel, scale=2.0)
    base_ang_vel = ObsTerm(func=mdp.base_ang_vel, scale=0.25)
    projected_gravity = ObsTerm(func=mdp.projected_gravity, scale=1.0)
    joint_pos = ObsTerm(func=mdp.joint_pos_rel, scale=1.0)
    joint_vel = ObsTerm(func=mdp.joint_vel_rel, scale=0.05)
    actions = ObsTerm(func=mdp.last_action, scale=1.0)
    motion_joint_action = ObsTerm(func=mdp.motion_joint_action, params={"command_name": "motion"}, scale=1.0)

    def __post_init__(self):
        self.enable_corruption = True
        self.concatenate_terms = True
        self.history_length = 0


@configclass
class DeltaOpenLoopCriticObsCfg(ObsGroup):
    base_pos_z = ObsTerm(func=mdp.base_pos_z, scale=1.0)
    feet_contact_force = _make_feet_contact_force_obs_term(scale=0.01)
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
    actions = ObsTerm(func=mdp.last_action, scale=1.0)
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
        self.commands.motion.debug_vis_goal_relative_to_robot = False
        self.commands.motion.anchor_body_name = "torso_link"
        
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
        self.rewards.motion_body_pos = None
        self.rewards.motion_body_ori = None
        self.rewards.penalty_minimal_action_norm = RewTerm(func=mdp.penalty_minimal_action_norm, weight=-0.1)
        self.terminations.ee_body_pos = None
        self.terminations.anchor_pos = None
        self.terminations.anchor_ori = None
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
        )
        self.actions.joint_pos.scale = G1_ACTION_SCALE

        self.observations.delta_policy = DeltaPolicyObsCfg()
        self.rewards.penalty_minimal_action_norm = None
        self.rewards.motion_body_pos_global = None
        self.rewards.motion_body_ori_global = None
        self.episode_length_s = 10.0
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


@configclass
class G1GapSwitchScanObsCfg(ObsGroup):
    """Observation group for high-level gap-switch policy scan input."""

    scan_points = ObsTerm(
        func=mdp.terrain_scan_points_b,
        params={
            "sensor_cfg": SceneEntityCfg("terrain_scan"),
            "grid_shape": (20, 20),
            "no_hit_value": 5.0,
            "flatten": False,
        },
    )

    def __post_init__(self):
        self.enable_corruption = False
        self.concatenate_terms = False
        self.history_length = 0


@configclass
class G1GapSwitchScanObsFlatCfg(ObsGroup):
    """Flattened scan observation for MLP-based experimentation."""

    scan_points_flat = ObsTerm(
        func=mdp.terrain_scan_points_b_flat,
        params={
            "sensor_cfg": SceneEntityCfg("terrain_scan"),
            "grid_shape": (20, 20),
            "no_hit_value": 5.0,
        },
    )

    def __post_init__(self):
        self.enable_corruption = False
        self.concatenate_terms = True
        self.history_length = 0


@configclass
class G1GapSwitchScanEnvCfg(G1FlatEnvCfg):
    """Scaffold env that adds a configurable local scan matrix around the robot."""

    high_level_scan: HighLevelScanCfg = HighLevelScanCfg()
    gap_course: GapCourseTerrainCfg = GapCourseTerrainCfg(
        proportion=1.0,
        size=(8.0, 4.0),
        num_gaps=1,
        gap_width_range=(0.4, 0.4),
        gap_depth=1.0,
        first_gap_center_x=1.0,
        gap_center_spacing=1.5,
        gap_centers_x=None,
        gap_y_span=None,
        gap_y_center_offset=0.0,
        surface_thickness=1.0,
        floor_thickness=1.0,
    )
    terrain_num_rows: int = 8
    terrain_num_cols: int = 8
    terrain_curriculum: bool = False
    terrain_max_init_level: int | None = 0

    def __post_init__(self):
        super().__post_init__()

        terrain_physics_material = self.scene.terrain.physics_material
        terrain_visual_material = self.scene.terrain.visual_material
        terrain_size = self.gap_course.size
        self.scene.terrain = terrain_gen.TerrainImporterCfg(
            prim_path="/World/ground",
            terrain_type="generator",
            terrain_generator=terrain_gen.TerrainGeneratorCfg(
                curriculum=self.terrain_curriculum,
                size=terrain_size,
                border_width=0.0,
                num_rows=self.terrain_num_rows,
                num_cols=self.terrain_num_cols,
                horizontal_scale=0.1,
                vertical_scale=0.005,
                slope_threshold=0.75,
                use_cache=False,
                sub_terrains={"gap_course": self.gap_course},
            ),
            max_init_terrain_level=self.terrain_max_init_level,
            collision_group=-1,
            physics_material=terrain_physics_material,
            visual_material=terrain_visual_material,
            debug_vis=False,
        )

        grid_shape, pattern_size, offset_xy = resolve_scan_grid_config(self.high_level_scan)

        self.scene.terrain_scan = RayCasterCfg(
            prim_path="{ENV_REGEX_NS}/Robot/torso_link",
            offset=RayCasterCfg.OffsetCfg(
                pos=(offset_xy[0], offset_xy[1], self.high_level_scan.sensor_height),
            ),
            attach_yaw_only=True,
            pattern_cfg=patterns.GridPatternCfg(
                resolution=self.high_level_scan.resolution,
                size=pattern_size,
                ordering="xy",
            ),
            debug_vis=False,
            mesh_prim_paths=["/World/ground"],
            max_distance=self.high_level_scan.max_distance,
        )
        if self.scene.contact_forces is not None:
            self.scene.terrain_scan.update_period = self.scene.contact_forces.update_period
        else:
            self.scene.terrain_scan.update_period = self.sim.dt

        # Keep existing policy/critic groups unchanged for backwards compatibility.
        self.observations.high_level_policy = G1GapSwitchScanObsCfg()
        self.observations.high_level_policy_flat = G1GapSwitchScanObsFlatCfg()
        self.observations.high_level_policy.scan_points.params["grid_shape"] = grid_shape
        self.observations.high_level_policy.scan_points.params["no_hit_value"] = self.high_level_scan.no_hit_value
        self.observations.high_level_policy_flat.scan_points_flat.params["grid_shape"] = grid_shape
        self.observations.high_level_policy_flat.scan_points_flat.params["no_hit_value"] = (
            self.high_level_scan.no_hit_value
        )


@configclass
class G1GapHighLevelPolicyObsCfg(ObsGroup):
    """High-level policy observations: local scan + goal in base frame."""

    scan_points_flat = ObsTerm(
        func=mdp.terrain_scan_points_b_flat,
        params={
            "sensor_cfg": SceneEntityCfg("terrain_scan"),
            "grid_shape": (20, 20),
            "no_hit_value": 5.0,
        },
    )
    goal_pos_b = ObsTerm(
        func=mdp.goal_position_b,
        params={
            "goal_offset": (2.0, 0.0, 0.0),
            "asset_cfg": SceneEntityCfg("robot"),
        },
    )
    # High-level switch policy receives its own previous action for autoregressive context.
    actions = ObsTerm(func=mdp.last_action, scale=1.0)

    def __post_init__(self):
        self.enable_corruption = False
        self.concatenate_terms = True
        self.history_length = 0


@configclass
class G1GapSwitchHierarchicalEnvCfg(G1GapSwitchScanEnvCfg):
    """Gap task with goal reward and high-level switch between two frozen low-level policies."""

    low_level_policy_1_checkpoint: str = ""
    low_level_policy_2_checkpoint: str = ""
    low_level_action_clip: float | None = None

    goal_margin_after_last_gap: float = 1.0
    goal_lateral_offset: float = 0.0
    goal_reward_std: float = 1.0
    goal_reached_threshold: float = 0.35
    base_height_termination_threshold: float = 0.2

    def _resolve_goal_offset(self) -> tuple[float, float, float]:
        if self.gap_course.gap_centers_x is not None and len(self.gap_course.gap_centers_x) > 0:
            last_gap_center_x = float(max(self.gap_course.gap_centers_x[: self.gap_course.num_gaps]))
        else:
            last_gap_center_x = float(
                self.gap_course.first_gap_center_x + (self.gap_course.num_gaps - 1) * self.gap_course.gap_center_spacing
            )

        max_gap_width = float(self.gap_course.gap_width_range[1])
        goal_x = last_gap_center_x + 0.5 * max_gap_width + self.goal_margin_after_last_gap
        max_goal_x = 0.5 * float(self.gap_course.size[0]) - 0.5
        goal_x = float(min(max(goal_x, 0.0), max_goal_x))

        goal_y = float(self.gap_course.gap_y_center_offset + self.goal_lateral_offset)
        max_abs_goal_y = 0.5 * float(self.gap_course.size[1]) - 0.25
        goal_y = float(max(min(goal_y, max_abs_goal_y), -max_abs_goal_y))

        return (goal_x, goal_y, 0.0)

    def __post_init__(self):
        super().__post_init__()

        if self.commands.motion is None:
            raise RuntimeError("Hierarchical switch task requires base `commands.motion` config.")

        self.commands.motion_policy_1 = copy.deepcopy(self.commands.motion)
        self.commands.motion_policy_2 = copy.deepcopy(self.commands.motion)

        goal_offset = self._resolve_goal_offset()
        grid_shape, _, _ = resolve_scan_grid_config(self.high_level_scan)
        flat_policy_obs_cfg = copy.deepcopy(self.observations.policy)

        # High-level observation groups for PPO actor/critic (switch policy).
        self.observations.policy = G1GapHighLevelPolicyObsCfg()
        self.observations.critic = G1GapHighLevelPolicyObsCfg()
        self.observations.policy.scan_points_flat.params["grid_shape"] = grid_shape
        self.observations.policy.scan_points_flat.params["no_hit_value"] = self.high_level_scan.no_hit_value
        self.observations.policy.goal_pos_b.params["goal_offset"] = goal_offset
        self.observations.critic.scan_points_flat.params["grid_shape"] = grid_shape
        self.observations.critic.scan_points_flat.params["no_hit_value"] = self.high_level_scan.no_hit_value
        self.observations.critic.goal_pos_b.params["goal_offset"] = goal_offset

        # Low-level observation groups consumed by frozen policies inside action term.
        # Reuse G1 flat-policy observation structure to match low-level checkpoints.
        low_level_obs_1 = _make_low_level_obs_from_flat_policy(
            flat_policy_obs_cfg,
            action_buffer_name="low_level_policy_1_last_actions",
            motion_command_name="motion_policy_1",
        )
        low_level_obs_2 = _make_low_level_obs_from_flat_policy(
            flat_policy_obs_cfg,
            action_buffer_name="low_level_policy_2_last_actions",
            motion_command_name="motion_policy_2",
        )

        self.actions.joint_pos = mdp.HighLevelPolicySwitchActionCfg(
            asset_name="robot",
            policy_1_checkpoint=self.low_level_policy_1_checkpoint,
            policy_2_checkpoint=self.low_level_policy_2_checkpoint,
            policy_1_observations=low_level_obs_1,
            policy_2_observations=low_level_obs_2,
            low_level_decimation=1,
            switch_threshold=0.5,
            use_sigmoid=True,
            low_level_action_clip=self.low_level_action_clip,
            low_level_actions=mdp.JointPositionActionCfg(
                asset_name="robot",
                joint_names=[".*"],
                use_default_offset=True,
                scale=G1_ACTION_SCALE,
            ),
        )

        # Replace tracking rewards with goal-based task rewards.
        self.rewards.motion_global_anchor_pos = None
        self.rewards.motion_global_anchor_ori = None
        self.rewards.motion_body_pos = None
        self.rewards.motion_body_ori = None
        self.rewards.motion_body_pos_global = None
        self.rewards.motion_body_ori_global = None
        self.rewards.motion_body_lin_vel = None
        self.rewards.motion_body_ang_vel = None
        self.rewards.joint_limit = None
        self.rewards.goal_distance = RewTerm(
            func=mdp.goal_position_error_tanh,
            weight=4.0,
            params={"std": self.goal_reward_std, "goal_offset": goal_offset, "asset_cfg": SceneEntityCfg("robot")},
        )
        self.rewards.goal_reached_bonus = RewTerm(
            func=mdp.goal_reached_bonus,
            weight=6.0,
            params={
                "threshold": self.goal_reached_threshold,
                "goal_offset": goal_offset,
                "asset_cfg": SceneEntityCfg("robot"),
            },
        )
        self.rewards.action_rate_l2.weight = -0.01

        # Replace tracking terminations with goal/fall terminations.
        self.terminations.anchor_pos = None
        self.terminations.anchor_ori = None
        self.terminations.ee_body_pos = None
        self.terminations.goal_reached = DoneTerm(
            func=mdp.goal_reached,
            params={
                "threshold": self.goal_reached_threshold,
                "goal_offset": goal_offset,
                "asset_cfg": SceneEntityCfg("robot"),
            },
        )
        self.terminations.base_fall = DoneTerm(
            func=mdp.base_height_below,
            params={"threshold": self.base_height_termination_threshold, "asset_cfg": SceneEntityCfg("robot")},
        )

        # Stabilize resets for the hierarchical task.
        self.events.push_robot = None
        self.events.reset_base = EventTerm(
            func=mdp.reset_root_state_uniform,
            mode="reset",
            params={
                "asset_cfg": SceneEntityCfg("robot"),
                "pose_range": {"x": (-0.2, 0.2), "y": (-0.2, 0.2), "yaw": (-0.5, 0.5)},
                "velocity_range": {
                    "x": (0.0, 0.0),
                    "y": (0.0, 0.0),
                    "z": (0.0, 0.0),
                    "roll": (0.0, 0.0),
                    "pitch": (0.0, 0.0),
                    "yaw": (0.0, 0.0),
                },
            },
        )
        self.events.reset_robot_joints = EventTerm(
            func=mdp.reset_joints_by_scale,
            mode="reset",
            params={
                "asset_cfg": SceneEntityCfg("robot"),
                "position_range": (0.95, 1.05),
                "velocity_range": (0.0, 0.0),
            },
        )

        self.episode_length_s = 8.0
