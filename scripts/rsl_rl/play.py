"""Script to play a checkpoint if an RL agent from RSL-RL."""

"""Launch Isaac Sim Simulator first."""

import argparse
import sys

from isaaclab.app import AppLauncher

# local imports
import cli_args  # isort: skip

# add argparse arguments
parser = argparse.ArgumentParser(description="Train an RL agent with RSL-RL.")
parser.add_argument("--video", action="store_true", default=False, help="Record videos during training.")
parser.add_argument("--video_length", type=int, default=200, help="Length of the recorded video (in steps).")
parser.add_argument(
    "--disable_fabric", action="store_true", default=False, help="Disable fabric and use USD I/O operations."
)
parser.add_argument("--num_envs", type=int, default=None, help="Number of environments to simulate.")
parser.add_argument("--task", type=str, default=None, help="Name of the task.")
parser.add_argument("--motion_file", type=str, default=None, help="Path to the motion file.")
parser.add_argument(
    "--low_level_policy_1_checkpoint",
    type=str,
    default=None,
    help="Path to frozen low-level policy #1 checkpoint for hierarchical switch tasks.",
)
parser.add_argument(
    "--low_level_policy_2_checkpoint",
    type=str,
    default=None,
    help="Path to frozen low-level policy #2 checkpoint for hierarchical switch tasks.",
)
parser.add_argument(
    "--disable_dr",
    action="store_true",
    default=False,
    help="Disable domain randomization (event randomization and observation corruption/noise).",
)
parser.add_argument(
    "--delta_action_space",
    type=str,
    default="whole_body",
    choices=("whole_body", "ankles", "lower_body", "com_force"),
    help=(
        "Delta-action space mode. `whole_body` keeps current behavior. "
        "`ankles` uses only ankle joints; `lower_body` uses lower-body joints; "
        "`com_force` uses (Fx, Fy, Fz) COM force actions. "
        "Joint-space modes remap outputs into the full-body action vector."
    ),
)
parser.add_argument(
    "--delta_com_force_scale",
    type=float,
    default=1.0,
    help=(
        "Gain for `--delta_action_space com_force`. "
        "Force commands are clipped in action space before scaling."
    ),
)
parser.add_argument(
    "--delta_com_force_clip",
    type=float,
    default=1.0,
    help=(
        "Symmetric clip bound for `--delta_action_space com_force` in action space, applied before scaling. "
        "Set <= 0 to disable clipping."
    ),
)
parser.add_argument(
    "--delta_com_force_debug_vis",
    action=argparse.BooleanOptionalAction,
    default=True,
    help="Enable viewport arrow visualization for applied COM forces in `--delta_action_space com_force` mode.",
)
parser.add_argument(
    "--replay_motion_actions_only",
    action="store_true",
    default=False,
    help=(
        "Ignore learned policy actions by zeroing them before env.step(). "
        "Useful with delta-action tasks to replay only motion NPZ `action`/`actions`."
    ),
)
parser.add_argument(
    "--enable_adaptive_reference_sampling",
    action="store_true",
    default=False,
    help=(
        "Deprecated flag. Adaptive reference time-step sampling is now always disabled in play "
        "for deterministic evaluation."
    ),
)
parser.add_argument(
    "--record_delta_model_dataset",
    action="store_true",
    default=False,
    help=(
        "Record delta-model inference trajectories (obs/actions) to NPZ using Genesis-style padded "
        "[num_traj, T, D] payload."
    ),
)
parser.add_argument(
    "--output_delta_model_npz",
    type=str,
    default=None,
    help=(
        "Output NPZ path for --record_delta_model_dataset. "
        "If omitted, defaults to <checkpoint_name>_<timestamp>.npz under the run directory."
    ),
)
parser.add_argument(
    "--delta_dataset_target_trajectories",
    "--target_trajectories",
    dest="delta_dataset_target_trajectories",
    type=int,
    default=0,
    help=(
        "Number of completed trajectories to record for --record_delta_model_dataset. "
        "Set <= 0 to keep recording until play loop exits. "
        "Alias: --target_trajectories."
    ),
)
parser.add_argument(
    "--delta_dataset_include_partial",
    action=argparse.BooleanOptionalAction,
    default=True,
    help="Include in-progress (non-terminated) trajectories when saving --record_delta_model_dataset.",
)
parser.add_argument(
    "--record_state_action_trajectories",
    action="store_true",
    default=False,
    help=(
        "Record robot state (joint/body world tensors) and applied actions to NPZ using padded "
        "[num_traj, T, D] payload."
    ),
)
parser.add_argument(
    "--output_state_action_npz",
    type=str,
    default=None,
    help=(
        "Output NPZ path for --record_state_action_trajectories. "
        "If omitted, defaults to <checkpoint_name>_state_action_<timestamp>.npz under the run directory."
    ),
)
parser.add_argument(
    "--state_action_target_trajectories",
    type=int,
    default=0,
    help=(
        "Number of completed trajectories to record for --record_state_action_trajectories. "
        "Set <= 0 to keep recording until play loop exits."
    ),
)
parser.add_argument(
    "--state_action_include_partial",
    action=argparse.BooleanOptionalAction,
    default=True,
    help="Include in-progress (non-terminated) trajectories when saving --record_state_action_trajectories.",
)
# append RSL-RL cli arguments
cli_args.add_rsl_rl_args(parser)
# append AppLauncher cli args
AppLauncher.add_app_launcher_args(parser)
args_cli, hydra_args = parser.parse_known_args()
# always enable cameras to record video
if args_cli.video:
    args_cli.enable_cameras = True

# clear out sys.argv for Hydra
sys.argv = [sys.argv[0]] + hydra_args

# launch omniverse app
app_launcher = AppLauncher(args_cli)
simulation_app = app_launcher.app

"""Rest everything follows."""

import gymnasium as gym
import os
import pathlib
from datetime import datetime

import numpy as np
import torch

from isaaclab.envs import (
    DirectMARLEnv,
    DirectMARLEnvCfg,
    DirectRLEnvCfg,
    ManagerBasedRLEnvCfg,
    multi_agent_to_single_agent,
)
from isaaclab.utils.dict import print_dict
from isaaclab_rl.rsl_rl import RslRlOnPolicyRunnerCfg, RslRlVecEnvWrapper
from isaaclab_tasks.utils import get_checkpoint_path
from isaaclab_tasks.utils.hydra import hydra_task_config
from isaaclab.utils.math import quat_apply, quat_inv, quat_mul, yaw_quat

# Import extensions to set up environment tasks
import whole_body_tracking.tasks  # noqa: F401
from whole_body_tracking.utils.exporter import attach_onnx_metadata, export_motion_policy_as_onnx
from whole_body_tracking.utils.my_on_policy_runner import MotionOnPolicyRunner as OnPolicyRunner

from utils import DEFAULT_STATE_ACTION_KEYS, StateActionTrajectoryRecorder

ANKLE_DELTA_ACTION_JOINT_NAMES = [
    "left_ankle_pitch_joint",
    "left_ankle_roll_joint",
    "right_ankle_pitch_joint",
    "right_ankle_roll_joint",
]
LOWER_BODY_DELTA_ACTION_JOINT_NAMES = [
    "left_hip_yaw_joint",
    "left_hip_roll_joint",
    "left_hip_pitch_joint",
    "left_knee_joint",
    "left_ankle_pitch_joint",
    "left_ankle_roll_joint",
    "right_hip_yaw_joint",
    "right_hip_roll_joint",
    "right_hip_pitch_joint",
    "right_knee_joint",
    "right_ankle_pitch_joint",
    "right_ankle_roll_joint",
    "waist_yaw_joint",
    "waist_roll_joint",
    "waist_pitch_joint",
]


def _configure_delta_action_space(env_cfg: ManagerBasedRLEnvCfg | DirectRLEnvCfg | DirectMARLEnvCfg):
    """Apply CLI-selected delta-action space overrides when the env/action cfg supports them."""
    joint_pos_cfg = getattr(getattr(env_cfg, "actions", None), "joint_pos", None)
    if joint_pos_cfg is None:
        if args_cli.delta_action_space != "whole_body":
            print("[WARN]: `--delta_action_space` was provided but this task has no `actions.joint_pos` config.")
        return

    if args_cli.delta_action_space == "com_force":
        if hasattr(joint_pos_cfg, "motion_command_name") and not hasattr(joint_pos_cfg, "external_action_buffer_name"):
            import whole_body_tracking.tasks.tracking.mdp as mdp

            force_body_name = getattr(env_cfg.commands.motion, "anchor_body_name", "torso_link")
            force_clip = args_cli.delta_com_force_clip if args_cli.delta_com_force_clip > 0.0 else None
            com_force_cfg = mdp.DeltaComForceActionCfg(
                asset_name=joint_pos_cfg.asset_name,
                joint_names=joint_pos_cfg.joint_names,
                use_default_offset=getattr(joint_pos_cfg, "use_default_offset", True),
                motion_command_name=getattr(joint_pos_cfg, "motion_command_name", "motion"),
                require_motion_action=getattr(joint_pos_cfg, "require_motion_action", True),
                force_body_name=force_body_name,
                force_scale=args_cli.delta_com_force_scale,
                force_clip=force_clip,
            )
            if hasattr(joint_pos_cfg, "preserve_order"):
                com_force_cfg.preserve_order = joint_pos_cfg.preserve_order
            if hasattr(joint_pos_cfg, "scale"):
                com_force_cfg.scale = joint_pos_cfg.scale
            if hasattr(joint_pos_cfg, "offset"):
                com_force_cfg.offset = joint_pos_cfg.offset
            if hasattr(joint_pos_cfg, "clip"):
                com_force_cfg.clip = joint_pos_cfg.clip
            com_force_cfg.debug_vis = bool(args_cli.delta_com_force_debug_vis)
            env_cfg.actions.joint_pos = com_force_cfg
            # COM-force mode: disable action penalties.
            if hasattr(env_cfg, "rewards"):
                if hasattr(env_cfg.rewards, "action_rate_l2"):
                    env_cfg.rewards.action_rate_l2 = None
                if hasattr(env_cfg.rewards, "penalty_minimal_action_norm"):
                    env_cfg.rewards.penalty_minimal_action_norm = None
            print(
                "[INFO]: Using COM-force delta action space with 3D force actions "
                f"(Fx, Fy, Fz), scale={args_cli.delta_com_force_scale} N, "
                f"clip={force_clip}, body='{force_body_name}', "
                f"debug_vis={com_force_cfg.debug_vis}."
            )
            print("[INFO]: Disabled action-penalty rewards for COM-force mode.")
        elif hasattr(joint_pos_cfg, "external_action_buffer_name"):
            import whole_body_tracking.tasks.tracking.mdp as mdp

            force_body_name = getattr(env_cfg.commands.motion, "anchor_body_name", "torso_link")
            force_clip = args_cli.delta_com_force_clip if args_cli.delta_com_force_clip > 0.0 else None
            com_force_cfg = mdp.ExternalDeltaComForceActionCfg(
                asset_name=joint_pos_cfg.asset_name,
                joint_names=joint_pos_cfg.joint_names,
                use_default_offset=getattr(joint_pos_cfg, "use_default_offset", True),
                motion_command_name=getattr(joint_pos_cfg, "motion_command_name", "motion"),
                external_action_buffer_name=getattr(joint_pos_cfg, "external_action_buffer_name", "delta_external_actions"),
                require_external_action=getattr(joint_pos_cfg, "require_external_action", True),
                force_body_name=force_body_name,
                force_scale=args_cli.delta_com_force_scale,
                force_clip=force_clip,
            )
            if hasattr(joint_pos_cfg, "preserve_order"):
                com_force_cfg.preserve_order = joint_pos_cfg.preserve_order
            if hasattr(joint_pos_cfg, "scale"):
                com_force_cfg.scale = joint_pos_cfg.scale
            if hasattr(joint_pos_cfg, "offset"):
                com_force_cfg.offset = joint_pos_cfg.offset
            if hasattr(joint_pos_cfg, "clip"):
                com_force_cfg.clip = joint_pos_cfg.clip
            com_force_cfg.debug_vis = bool(args_cli.delta_com_force_debug_vis)
            env_cfg.actions.joint_pos = com_force_cfg
            print(
                "[INFO]: Using COM-force delta action space for finetuning. "
                "Base policy remains joint-space while the frozen delta policy outputs "
                f"(Fx, Fy, Fz), scale={args_cli.delta_com_force_scale} N, "
                f"clip={force_clip}, body='{force_body_name}', "
                f"debug_vis={com_force_cfg.debug_vis}."
            )
        else:
            print(
                "[WARN]: `--delta_action_space com_force` is supported only for "
                "delta-action open-loop or finetune tasks with compatible action configs. "
                "Keeping the existing action configuration."
            )
        return

    requested_joint_names = None
    if args_cli.delta_action_space == "ankles":
        requested_joint_names = ANKLE_DELTA_ACTION_JOINT_NAMES.copy()
    elif args_cli.delta_action_space == "lower_body":
        requested_joint_names = LOWER_BODY_DELTA_ACTION_JOINT_NAMES.copy()

    applied = False
    if hasattr(joint_pos_cfg, "delta_action_joint_names"):
        joint_pos_cfg.delta_action_joint_names = requested_joint_names
        applied = True
    if hasattr(joint_pos_cfg, "external_delta_action_joint_names"):
        joint_pos_cfg.external_delta_action_joint_names = requested_joint_names
        applied = True

    if args_cli.delta_action_space == "ankles":
        if applied:
            print(
                "[INFO]: Using ankle-only delta action space. "
                f"Delta joints: {ANKLE_DELTA_ACTION_JOINT_NAMES}."
            )
        else:
            print(
                "[WARN]: `--delta_action_space ankles` has no effect for this task "
                "(delta-action fields are not present in the action config)."
            )
    elif args_cli.delta_action_space == "lower_body":
        if applied:
            print(
                "[INFO]: Using lower-body delta action space. "
                f"Delta joints: {LOWER_BODY_DELTA_ACTION_JOINT_NAMES}."
            )
        else:
            print(
                "[WARN]: `--delta_action_space lower_body` has no effect for this task "
                "(delta-action fields are not present in the action config)."
            )


def _disable_domain_randomization(env_cfg: ManagerBasedRLEnvCfg | DirectRLEnvCfg | DirectMARLEnvCfg):
    """Disable DR sources used by these manager-based tracking tasks."""
    disabled_event_terms = []
    if hasattr(env_cfg, "events"):
        for term_name in ("physics_material", "add_joint_default_pos", "base_com", "push_robot"):
            if hasattr(env_cfg.events, term_name):
                setattr(env_cfg.events, term_name, None)
                disabled_event_terms.append(term_name)

    if hasattr(env_cfg, "observations"):
        for group_name in ("policy", "critic", "delta_policy"):
            group_cfg = getattr(env_cfg.observations, group_name, None)
            if group_cfg is not None and hasattr(group_cfg, "enable_corruption"):
                group_cfg.enable_corruption = False

    print(
        "[INFO]: DR disabled. "
        f"Disabled event terms: {disabled_event_terms if disabled_event_terms else 'none'}; "
        "set observation corruption OFF for available observation groups."
    )


def _configure_hierarchical_switch_policies(env_cfg: ManagerBasedRLEnvCfg | DirectRLEnvCfg | DirectMARLEnvCfg):
    """Apply CLI checkpoint overrides for hierarchical high-level switch action tasks."""
    joint_pos_cfg = getattr(getattr(env_cfg, "actions", None), "joint_pos", None)
    if joint_pos_cfg is None:
        return
    if not (hasattr(joint_pos_cfg, "policy_1_checkpoint") and hasattr(joint_pos_cfg, "policy_2_checkpoint")):
        return

    if args_cli.low_level_policy_1_checkpoint is not None:
        ckpt_1 = os.path.abspath(os.path.expanduser(args_cli.low_level_policy_1_checkpoint))
        if not os.path.isfile(ckpt_1):
            raise FileNotFoundError(f"Low-level policy #1 checkpoint not found: {ckpt_1}")
        joint_pos_cfg.policy_1_checkpoint = ckpt_1

    if args_cli.low_level_policy_2_checkpoint is not None:
        ckpt_2 = os.path.abspath(os.path.expanduser(args_cli.low_level_policy_2_checkpoint))
        if not os.path.isfile(ckpt_2):
            raise FileNotFoundError(f"Low-level policy #2 checkpoint not found: {ckpt_2}")
        joint_pos_cfg.policy_2_checkpoint = ckpt_2

    if not getattr(joint_pos_cfg, "policy_1_checkpoint", "") or not getattr(joint_pos_cfg, "policy_2_checkpoint", ""):
        raise ValueError(
            "Hierarchical switch task requires both low-level checkpoints. Provide "
            "`--low_level_policy_1_checkpoint` and `--low_level_policy_2_checkpoint`."
        )

    print(
        "[INFO]: Hierarchical switch policies configured: "
        f"policy_1='{joint_pos_cfg.policy_1_checkpoint}', "
        f"policy_2='{joint_pos_cfg.policy_2_checkpoint}'."
    )


def _has_motion_command(env_cfg: ManagerBasedRLEnvCfg | DirectRLEnvCfg | DirectMARLEnvCfg) -> bool:
    return bool(
        hasattr(env_cfg, "commands")
        and hasattr(env_cfg.commands, "motion")
        and getattr(env_cfg.commands, "motion", None) is not None
    )


def _append_timestamp_to_path(path: str, timestamp: str) -> str:
    directory = os.path.dirname(path)
    filename = os.path.basename(path)
    stem, ext = os.path.splitext(filename)
    stamped = f"{stem}_{timestamp}{ext}"
    return os.path.join(directory, stamped)


def _resolve_control_dt(env: RslRlVecEnvWrapper) -> float:
    step_dt = getattr(env.unwrapped, "step_dt", None)
    if step_dt is not None:
        step_dt = float(step_dt)
        if step_dt > 0.0:
            return step_dt

    cfg = getattr(env.unwrapped, "cfg", None)
    sim_cfg = getattr(cfg, "sim", None)
    sim_dt = getattr(sim_cfg, "dt", None)
    decimation = getattr(cfg, "decimation", None)
    if sim_dt is not None and decimation is not None:
        control_dt = float(sim_dt) * float(decimation)
        if control_dt > 0.0:
            return control_dt
    return 0.0


def _group_history_lengths(observation_manager, group_name: str, term_names: list[str]) -> list[int]:
    group_cfg = getattr(observation_manager.cfg, group_name)
    group_history_length = getattr(group_cfg, "history_length", None)
    if group_history_length is not None:
        hist = int(group_history_length)
        hist = 1 if hist == 0 else hist
        return [hist] * len(term_names)

    history_lengths: list[int] = []
    for term_name in term_names:
        term_cfg = getattr(group_cfg, term_name, None)
        term_history = int(getattr(term_cfg, "history_length", 0)) if term_cfg is not None else 0
        history_lengths.append(1 if term_history == 0 else term_history)
    return history_lengths


def _infer_group_obs_split_spec(env, group_name: str) -> tuple[list[str], list[int], list[str]]:
    observation_manager = env.observation_manager
    term_names = list(observation_manager.active_terms[group_name])
    history_lengths = _group_history_lengths(observation_manager, group_name, term_names)
    group_cfg = getattr(observation_manager.cfg, group_name)

    split_dims: list[int] = []
    payload_keys: list[str] = []
    for term_name, hist_len in zip(term_names, history_lengths, strict=True):
        term_cfg = getattr(group_cfg, term_name, None)
        if term_cfg is None or getattr(term_cfg, "func", None) is None:
            raise RuntimeError(
                f"Could not resolve observation term config for group='{group_name}', term='{term_name}'."
            )
        params = getattr(term_cfg, "params", None) or {}
        term_tensor = term_cfg.func(env, **params)
        if isinstance(term_tensor, dict):
            term_tensor = torch.cat([term_tensor[key] for key in term_tensor], dim=-1)
        if term_tensor.ndim < 2:
            term_tensor = term_tensor.reshape(term_tensor.shape[0], -1)
        base_dim = int(term_tensor.reshape(term_tensor.shape[0], -1).shape[1])
        split_dims.append(base_dim * int(hist_len))
        # Prefix with `obs_` to avoid collision with action keys (`action`, `actions`).
        payload_keys.append(f"obs_{term_name}")
    return term_names, split_dims, payload_keys


def _split_obs_into_term_components(
    obs_batch: torch.Tensor,
    split_dims: list[int],
    payload_keys: list[str],
) -> dict[str, torch.Tensor]:
    obs_2d = obs_batch
    if obs_2d.ndim == 1:
        obs_2d = obs_2d.reshape(1, -1)
    if obs_2d.ndim != 2:
        raise AssertionError(f"Expected observation tensor shape [B, D], got {tuple(obs_2d.shape)}.")

    total_dim = int(sum(split_dims))
    if int(obs_2d.shape[1]) != total_dim:
        raise AssertionError(
            f"Observation dim mismatch while splitting components: obs_dim={int(obs_2d.shape[1])}, expected={total_dim}."
        )

    components: dict[str, torch.Tensor] = {}
    start = 0
    for key, dim in zip(payload_keys, split_dims, strict=True):
        end = start + int(dim)
        components[key] = obs_2d[:, start:end]
        start = end
    return components


class _DeltaModelTrajectoryRecorder:
    def __init__(self, num_envs: int, fps: float, target_trajectories: int, record_ensemble_stats: bool = False):
        self.num_envs = int(num_envs)
        self.fps = float(fps)
        self.target_trajectories = max(int(target_trajectories), 0)
        self.record_ensemble_stats = bool(record_ensemble_stats)
        self.component_keys: list[str] = []
        self.auxiliary_keys: list[str] = []
        self.action_key = "actions"
        self.ensemble_uncertainty_key = "ensemble_uncertainty"
        self.ensemble_gate_key = "ensemble_gate"
        self.traj_buffers: list[dict[str, list[np.ndarray]]] = [{self.action_key: []} for _ in range(self.num_envs)]
        self.collected_trajectories: list[dict[str, np.ndarray]] = []
        self.saved_count = 0
        self.saved_motion_length = 0

    @property
    def collected_count(self) -> int:
        return len(self.collected_trajectories)

    def has_reached_target(self) -> bool:
        return self.target_trajectories > 0 and self.collected_count >= self.target_trajectories

    def _ensure_component_keys(self, components: dict[str, torch.Tensor]):
        keys = list(components.keys())
        if len(keys) == 0:
            raise AssertionError("Logged component dictionary is empty.")
        if len(self.component_keys) == 0:
            self.component_keys = keys
            for env_id in range(self.num_envs):
                for key in self.component_keys:
                    self.traj_buffers[env_id][key] = []
            return
        if keys != self.component_keys:
            raise AssertionError(
                "Logged component keys changed during rollout: "
                f"expected {self.component_keys}, got {keys}."
            )

    def _ensure_auxiliary_keys(self, auxiliary_components: dict[str, torch.Tensor]):
        keys = list(auxiliary_components.keys())
        if len(keys) == 0:
            return
        if len(self.auxiliary_keys) == 0:
            self.auxiliary_keys = keys
            for env_id in range(self.num_envs):
                for key in self.auxiliary_keys:
                    self.traj_buffers[env_id][key] = []
            return
        if keys != self.auxiliary_keys:
            raise AssertionError(
                "Logged auxiliary keys changed during rollout: "
                f"expected {self.auxiliary_keys}, got {keys}."
            )

    def append_step(
        self,
        components: dict[str, torch.Tensor],
        action_batch: torch.Tensor,
        auxiliary_components: dict[str, torch.Tensor] | None = None,
    ):
        self._ensure_component_keys(components)
        action_np = action_batch.detach().to("cpu", dtype=torch.float32).numpy()
        if action_np.ndim != 2 or action_np.shape[0] != self.num_envs:
            raise AssertionError(f"Expected action shape [num_envs, A], got {action_np.shape}.")

        component_np_map: dict[str, np.ndarray] = {}
        for key in self.component_keys:
            component_np = components[key].detach().to("cpu", dtype=torch.float32).numpy()
            if component_np.ndim < 2 or component_np.shape[0] != self.num_envs:
                raise AssertionError(
                    f"Expected logged component '{key}' shape [num_envs, ...], got {component_np.shape}."
                )
            component_np_map[key] = component_np

        auxiliary_np_map: dict[str, np.ndarray] = {}
        if self.record_ensemble_stats:
            if auxiliary_components is None:
                raise AssertionError(
                    "Ensemble stats logging is enabled but no auxiliary components were provided."
                )
            self._ensure_auxiliary_keys(auxiliary_components)
            for key in self.auxiliary_keys:
                auxiliary_np = auxiliary_components[key].detach().to("cpu", dtype=torch.float32).numpy()
                if auxiliary_np.ndim == 1:
                    if auxiliary_np.shape[0] != self.num_envs:
                        raise AssertionError(
                            f"Expected auxiliary '{key}' shape [num_envs] or [num_envs, D], got {auxiliary_np.shape}."
                        )
                    auxiliary_np = auxiliary_np.reshape(self.num_envs, 1)
                elif auxiliary_np.ndim != 2 or auxiliary_np.shape[0] != self.num_envs:
                    raise AssertionError(
                        f"Expected auxiliary '{key}' shape [num_envs] or [num_envs, D], got {auxiliary_np.shape}."
                    )
                auxiliary_np_map[key] = auxiliary_np

        for env_id in range(self.num_envs):
            for key in self.component_keys:
                self.traj_buffers[env_id][key].append(component_np_map[key][env_id].astype(np.float32).copy())
            for key in self.auxiliary_keys:
                self.traj_buffers[env_id][key].append(auxiliary_np_map[key][env_id].astype(np.float32).copy())
            self.traj_buffers[env_id][self.action_key].append(action_np[env_id].astype(np.float32).copy())

    def finalize_done(self, dones: torch.Tensor) -> int:
        done_mask = dones.detach().to(device="cpu", dtype=torch.bool)
        if done_mask.ndim == 0:
            done_mask = done_mask.reshape(1)
        elif done_mask.ndim > 1:
            done_mask = done_mask.reshape(done_mask.shape[0], -1).any(dim=1)
        if done_mask.ndim != 1 or done_mask.shape[0] != self.num_envs:
            raise AssertionError(f"Expected done mask shape [num_envs], got {tuple(done_mask.shape)}.")

        gained = 0
        done_ids = torch.nonzero(done_mask, as_tuple=False).flatten().tolist()
        for env_id in done_ids:
            if self.has_reached_target():
                break
            gained += self._finalize_env_traj(env_id)
        return gained

    def finalize_open(self):
        for env_id in range(self.num_envs):
            if self.has_reached_target():
                break
            self._finalize_env_traj(env_id)

    def _finalize_env_traj(self, env_id: int) -> int:
        if self.has_reached_target():
            return 0
        if len(self.traj_buffers[env_id][self.action_key]) == 0:
            return 0
        keys = [*self.component_keys, *self.auxiliary_keys, self.action_key]
        traj = {key: np.stack(self.traj_buffers[env_id][key], axis=0).astype(np.float32) for key in keys}
        self.collected_trajectories.append(traj)
        self.traj_buffers[env_id] = {key: [] for key in keys}
        return 1

    def save_dataset(self, output_path: str, include_partial: bool) -> str | None:
        if include_partial:
            self.finalize_open()

        if len(self.collected_trajectories) == 0:
            self.saved_count = 0
            self.saved_motion_length = 0
            return None

        trajs = (
            self.collected_trajectories
            if self.target_trajectories <= 0
            else self.collected_trajectories[: self.target_trajectories]
        )
        motion_length = max(int(traj[self.action_key].shape[0]) for traj in trajs)
        payload_keys = [*self.component_keys, *self.auxiliary_keys, self.action_key]
        feature_shapes = {key: tuple(trajs[0][key].shape[1:]) for key in payload_keys}

        payload: dict[str, np.ndarray] = {
            "fps": np.array([self.fps], dtype=np.float32),
        }

        for key in payload_keys:
            stacked: list[np.ndarray] = []
            expected_feature_shape = feature_shapes[key]
            for traj in trajs:
                seq = traj[key].astype(np.float32)
                if seq.ndim < 2:
                    raise AssertionError(f"Trajectory key '{key}' must include a time and feature dimension, got {seq.shape}.")
                if tuple(seq.shape[1:]) != expected_feature_shape:
                    raise AssertionError(
                        f"Trajectory key '{key}' shape mismatch: expected tail {expected_feature_shape}, "
                        f"got {tuple(seq.shape[1:])}."
                    )
                if seq.shape[0] <= 0:
                    raise AssertionError(f"Trajectory key '{key}' has empty time dimension.")
                if seq.shape[0] < motion_length:
                    pad = np.repeat(seq[-1:, ...], motion_length - seq.shape[0], axis=0)
                    seq = np.concatenate([seq, pad], axis=0)
                elif seq.shape[0] > motion_length:
                    seq = seq[:motion_length]
                stacked.append(seq)
            payload[key] = np.stack(stacked, axis=0).astype(np.float32)

        expected_num_traj = len(trajs)
        expected_shapes = {key: (expected_num_traj, motion_length, *feature_shape) for key, feature_shape in feature_shapes.items()}
        fps = payload["fps"]
        if fps.ndim != 1 or fps.shape[0] != 1 or not np.isfinite(fps).all() or float(fps[0]) <= 0.0:
            raise AssertionError(f"Invalid `fps` payload shape/value: shape={fps.shape}, value={fps}")
        for key, value in payload.items():
            if key == "fps":
                continue
            if value.shape != expected_shapes[key]:
                raise AssertionError(
                    f"Payload key '{key}' shape mismatch: expected {expected_shapes[key]}, got {value.shape}"
                )

        os.makedirs(os.path.dirname(output_path) or ".", exist_ok=True)
        np.savez(output_path, **payload)
        self.saved_count = expected_num_traj
        self.saved_motion_length = motion_length
        return output_path


def _resolve_obs_components_for_logging(
    env,
    group_name: str,
    obs_batch: torch.Tensor,
    split_cache: dict[str, tuple[list[str], list[int], list[str]]],
) -> dict[str, torch.Tensor]:
    if group_name not in split_cache:
        split_cache[group_name] = _infer_group_obs_split_spec(env, group_name)
    _, split_dims, payload_keys = split_cache[group_name]
    return _split_obs_into_term_components(obs_batch=obs_batch, split_dims=split_dims, payload_keys=payload_keys)


def _resolve_robot_body_world_components_for_logging(env) -> dict[str, torch.Tensor]:
    command_manager = getattr(env, "command_manager", None)
    if command_manager is None:
        return {}

    try:
        motion_term = command_manager.get_term("motion")
    except Exception:
        return {}

    body_pos_w = getattr(motion_term, "robot_body_pos_w", None)
    body_quat_w = getattr(motion_term, "robot_body_quat_w", None)
    if not isinstance(body_pos_w, torch.Tensor) or not isinstance(body_quat_w, torch.Tensor):
        return {}

    if body_pos_w.ndim != 3 or int(body_pos_w.shape[-1]) != 3:
        raise AssertionError(f"Expected robot body positions as [num_envs, num_bodies, 3], got {tuple(body_pos_w.shape)}.")
    if body_quat_w.ndim != 3 or int(body_quat_w.shape[-1]) != 4:
        raise AssertionError(
            f"Expected robot body quaternions as [num_envs, num_bodies, 4], got {tuple(body_quat_w.shape)}."
        )

    return {
        "body_pos_w": body_pos_w,
        "body_quat_w": body_quat_w,
    }


def _resolve_robot_state_components_for_logging(env) -> dict[str, torch.Tensor]:
    command_manager = getattr(env, "command_manager", None)
    if command_manager is not None:
        try:
            motion_term = command_manager.get_term("motion")
        except Exception:
            motion_term = None
        if motion_term is not None:
            component_map = {
                "joint_pos": getattr(motion_term, "robot_joint_pos", None),
                "joint_vel": getattr(motion_term, "robot_joint_vel", None),
                "body_pos_w": getattr(motion_term, "robot_body_pos_w", None),
                "body_quat_w": getattr(motion_term, "robot_body_quat_w", None),
                "body_lin_vel_w": getattr(motion_term, "robot_body_lin_vel_w", None),
                "body_ang_vel_w": getattr(motion_term, "robot_body_ang_vel_w", None),
            }
            components: dict[str, torch.Tensor] = {}
            for key, value in component_map.items():
                if not isinstance(value, torch.Tensor):
                    continue
                if int(value.shape[0]) != env.num_envs:
                    raise AssertionError(
                        f"Expected logged robot state '{key}' batch size {env.num_envs}, got {tuple(value.shape)}."
                    )
                components[key] = value
            if set(components.keys()) == set(DEFAULT_STATE_ACTION_KEYS):
                return components

    scene = getattr(env, "scene", None)
    if scene is None or "robot" not in scene:
        return {}

    robot_data = scene["robot"].data
    components = {
        "joint_pos": robot_data.joint_pos,
        "joint_vel": robot_data.joint_vel,
    }
    for key in ("body_pos_w", "body_quat_w", "body_lin_vel_w", "body_ang_vel_w"):
        value = getattr(robot_data, key, None)
        if isinstance(value, torch.Tensor):
            components[key] = value
    if set(components.keys()) != set(DEFAULT_STATE_ACTION_KEYS):
        return {}
    for key, value in components.items():
        if int(value.shape[0]) != env.num_envs:
            raise AssertionError(
                f"Expected logged robot state '{key}' batch size {env.num_envs}, got {tuple(value.shape)}."
            )
    return components


def _resolve_motion_command_components_for_logging(env) -> dict[str, torch.Tensor]:
    command_manager = getattr(env, "command_manager", None)
    if command_manager is None:
        return {}

    try:
        motion_term = command_manager.get_term("motion")
    except Exception:
        return {}

    component_map = {
        "motion_joint_pos": getattr(motion_term, "joint_pos", None),
        "motion_joint_vel": getattr(motion_term, "joint_vel", None),
        "motion_anchor_pos_w": getattr(motion_term, "anchor_pos_w", None),
        "motion_anchor_quat_w": getattr(motion_term, "anchor_quat_w", None),
        "motion_anchor_lin_vel_w": getattr(motion_term, "anchor_lin_vel_w", None),
        "motion_anchor_ang_vel_w": getattr(motion_term, "anchor_ang_vel_w", None),
        "motion_body_pos_w": getattr(motion_term, "body_pos_w", None),
        "motion_body_quat_w": getattr(motion_term, "body_quat_w", None),
        "motion_body_lin_vel_w": getattr(motion_term, "body_lin_vel_w", None),
        "motion_body_ang_vel_w": getattr(motion_term, "body_ang_vel_w", None),
        "motion_body_pos_relative_w": getattr(motion_term, "body_pos_relative_w", None),
        "motion_body_quat_relative_w": getattr(motion_term, "body_quat_relative_w", None),
    }
    if getattr(motion_term, "has_joint_action", False):
        component_map["motion_joint_action"] = motion_term.joint_action

    components: dict[str, torch.Tensor] = {}
    for key, value in component_map.items():
        if not isinstance(value, torch.Tensor):
            continue
        if int(value.shape[0]) != env.num_envs:
            raise AssertionError(
                f"Expected logged motion component '{key}' batch size {env.num_envs}, got {tuple(value.shape)}."
            )
        components[key] = value

    return components


def _get_motion_time_steps(env) -> torch.Tensor | None:
    command_manager = getattr(env, "command_manager", None)
    if command_manager is None:
        return None
    try:
        motion_term = command_manager.get_term("motion")
    except Exception:
        return None
    time_steps = getattr(motion_term, "time_steps", None)
    if time_steps is None:
        return None
    if not isinstance(time_steps, torch.Tensor):
        return None
    return time_steps.detach().clone()

def _bootstrap_motion_reference_startup(env: RslRlVecEnvWrapper):
    """Hacky startup alignment for play-mode motion tasks.

    `env.reset()` resamples the motion command and writes the robot/root state into sim, but
    `MotionCommand.body_pos_relative_w/body_quat_relative_w` are still only refreshed in
    `_update_command()`. Since terminations run before `command_manager.compute()` on the first
    step, we manually refresh those cached relative targets once at startup.
    """

    base_env = env.unwrapped
    command_manager = getattr(base_env, "command_manager", None)
    if command_manager is None:
        return

    try:
        motion_term = command_manager.get_term("motion")
    except Exception:
        return

    env_ids = torch.arange(base_env.num_envs, dtype=torch.int64, device=base_env.device)
    motion_term._resample_command(env_ids)
    base_env.scene.write_data_to_sim()
    base_env.sim.forward()
    base_env.scene.update(dt=0.0)

    anchor_pos_w_repeat = motion_term.anchor_pos_w[:, None, :].repeat(1, len(motion_term.cfg.body_names), 1)
    heading_quat_w_repeat = motion_term.heading_quat_w[:, None, :].repeat(1, len(motion_term.cfg.body_names), 1)
    robot_anchor_pos_w_repeat = motion_term.robot_anchor_pos_w[:, None, :].repeat(1, len(motion_term.cfg.body_names), 1)
    robot_heading_quat_w_repeat = motion_term.robot_heading_quat_w[:, None, :].repeat(
        1, len(motion_term.cfg.body_names), 1
    )

    delta_pos_w = robot_anchor_pos_w_repeat.clone()
    delta_pos_w[..., 2] = anchor_pos_w_repeat[..., 2]
    robot_heading_yaw_quat_w_repeat = yaw_quat(robot_heading_quat_w_repeat)
    heading_yaw_quat_w_repeat = yaw_quat(heading_quat_w_repeat)
    delta_ori_w = quat_mul(robot_heading_yaw_quat_w_repeat, quat_inv(heading_yaw_quat_w_repeat))

    motion_term.body_quat_relative_w = quat_mul(delta_ori_w, motion_term.body_quat_w)
    motion_term.body_pos_relative_w = delta_pos_w + quat_apply(delta_ori_w, motion_term.body_pos_w - anchor_pos_w_repeat)

@hydra_task_config(args_cli.task, "rsl_rl_cfg_entry_point")
def main(env_cfg: ManagerBasedRLEnvCfg | DirectRLEnvCfg | DirectMARLEnvCfg, agent_cfg: RslRlOnPolicyRunnerCfg):
    """Play with RSL-RL agent."""
    agent_cfg: RslRlOnPolicyRunnerCfg = cli_args.parse_rsl_rl_cfg(args_cli.task, args_cli)
    env_cfg.scene.num_envs = args_cli.num_envs if args_cli.num_envs is not None else env_cfg.scene.num_envs
    if _has_motion_command(env_cfg):
        env_cfg.commands.motion.adaptive_alpha = 0.0
        env_cfg.commands.motion.sample_time_steps = False
        print(
            "[INFO]: Adaptive reference sampling is disabled for play "
            "(commands.motion.adaptive_alpha=0.0, commands.motion.sample_time_steps=False)."
        )
        if args_cli.enable_adaptive_reference_sampling:
            print(
                "[WARN]: `--enable_adaptive_reference_sampling` is deprecated and ignored in play; "
                "adaptive sampling remains disabled."
            )
    elif args_cli.enable_adaptive_reference_sampling:
        print(
            "[INFO]: This task has no motion command; `--enable_adaptive_reference_sampling` "
            "is deprecated and ignored."
        )
    motion_file_override = None
    if _has_motion_command(env_cfg) and args_cli.motion_file is not None:
        motion_file_override = os.path.abspath(os.path.expanduser(args_cli.motion_file))
        if not os.path.isfile(motion_file_override):
            raise FileNotFoundError(f"Motion file not found: {motion_file_override}")

    checkpoint_path_override = None
    if args_cli.checkpoint is not None:
        checkpoint_candidate = os.path.abspath(os.path.expanduser(args_cli.checkpoint))
        if os.path.isfile(checkpoint_candidate):
            checkpoint_path_override = checkpoint_candidate
        elif os.path.isabs(args_cli.checkpoint) or os.path.sep in args_cli.checkpoint:
            raise FileNotFoundError(f"Checkpoint file not found: {checkpoint_candidate}")

    # specify directory for logging experiments
    log_root_path = os.path.join("logs", "rsl_rl", agent_cfg.experiment_name)
    log_root_path = os.path.abspath(log_root_path)

    if checkpoint_path_override is not None:
        resume_path = checkpoint_path_override
        print(f"[INFO]: Loading model checkpoint from CLI override: {resume_path}")
    elif args_cli.wandb_path:
        import wandb

        run_path = args_cli.wandb_path

        api = wandb.Api()
        if "model" in args_cli.wandb_path:
            run_path = "/".join(args_cli.wandb_path.split("/")[:-1])
        wandb_run = api.run(run_path)
        # loop over files in the run
        files = [file.name for file in wandb_run.files() if "model" in file.name]
        # files are all model_xxx.pt find the largest filename
        if "model" in args_cli.wandb_path:
            file = args_cli.wandb_path.split("/")[-1]
        else:
            file = max(files, key=lambda x: int(x.split("_")[1].split(".")[0]))

        wandb_file = wandb_run.file(str(file))
        wandb_file.download("./logs/rsl_rl/temp", replace=True)

        print(f"[INFO]: Loading model checkpoint from: {run_path}/{file}")
        resume_path = f"./logs/rsl_rl/temp/{file}"

        if motion_file_override is None and _has_motion_command(env_cfg):
            art = next((a for a in wandb_run.used_artifacts() if a.type == "motions"), None)
            if art is None:
                print("[WARN] No motion artifact found in the run. Using env_cfg.commands.motion.motion_file as-is.")
            else:
                env_cfg.commands.motion.motion_file = str(pathlib.Path(art.download()) / "motion.npz")

    else:
        print(f"[INFO] Loading experiment from directory: {log_root_path}")
        resume_path = get_checkpoint_path(log_root_path, agent_cfg.load_run, agent_cfg.load_checkpoint)
        print(f"[INFO]: Loading model checkpoint from: {resume_path}")

    if _has_motion_command(env_cfg):
        if motion_file_override is not None:
            env_cfg.commands.motion.motion_file = motion_file_override
            print(f"[INFO]: Using motion file from CLI override: {motion_file_override}")
        else:
            print(f"[INFO]: Using motion file from env config: {env_cfg.commands.motion.motion_file}")
    elif args_cli.motion_file is not None:
        print("[INFO]: This task has no motion command; ignoring --motion_file.")
    _configure_delta_action_space(env_cfg)
    _configure_hierarchical_switch_policies(env_cfg)
    if args_cli.disable_dr:
        _disable_domain_randomization(env_cfg)

    # create isaac environment
    env = gym.make(args_cli.task, cfg=env_cfg, render_mode="rgb_array" if args_cli.video else None)

    log_dir = os.path.dirname(resume_path)

    timestamp = datetime.now().strftime("%Y-%m-%d_%H-%M-%S")
    ckpt_stem = os.path.splitext(os.path.basename(resume_path))[0]
    # wrap for video recording
    if args_cli.video:
        video_kwargs = {
            "video_folder": os.path.join(log_dir, "videos", f"play_{timestamp}_{ckpt_stem}"),
            "step_trigger": lambda step: step == 0,
            "video_length": args_cli.video_length,
            "disable_logger": True,
        }
        print("[INFO] Recording videos during playing.")
        print_dict(video_kwargs, nesting=4)
        env = gym.wrappers.RecordVideo(env, **video_kwargs)

    # convert to single-agent instance if required by the RL algorithm
    if isinstance(env.unwrapped, DirectMARLEnv):
        env = multi_agent_to_single_agent(env)

    # wrap around environment for rsl-rl
    env = RslRlVecEnvWrapper(env)

    # load previously trained model
    ppo_runner = OnPolicyRunner(env, agent_cfg.to_dict(), log_dir=None, device=agent_cfg.device)
    ppo_runner.load(resume_path)

    # obtain the trained policy for inference
    policy = ppo_runner.get_inference_policy(device=env.unwrapped.device)
    if args_cli.replay_motion_actions_only:
        print("[INFO]: Replaying motion actions only (policy actions are forced to zero each step).")

    delta_dataset_recorder = None
    delta_dataset_output_path = None
    if args_cli.record_delta_model_dataset:
        control_dt = _resolve_control_dt(env)
        if control_dt <= 0.0:
            print("[WARN]: Could not resolve control dt; using fallback fps=1.0 for delta-model dataset logging.")
            control_dt = 1.0
        delta_fps = 1.0 / control_dt
        record_ensemble_stats = ppo_runner.delta_policy_ensemble_size > 1
        delta_dataset_recorder = _DeltaModelTrajectoryRecorder(
            num_envs=env.num_envs,
            fps=delta_fps,
            target_trajectories=args_cli.delta_dataset_target_trajectories,
            record_ensemble_stats=record_ensemble_stats,
        )
        checkpoint_stem = os.path.splitext(os.path.basename(resume_path))[0]
        default_delta_npz = os.path.join(
            os.path.dirname(resume_path),
            "delta_model_datasets",
            f"{checkpoint_stem if checkpoint_stem else 'policy'}.npz",
        )
        requested_output = args_cli.output_delta_model_npz or default_delta_npz
        requested_output = os.path.abspath(os.path.expanduser(requested_output))
        run_timestamp = datetime.now().strftime("%Y-%m-%d_%H-%M-%S")
        delta_dataset_suffix = args_cli.delta_dataset_suffix.strip() if args_cli.delta_dataset_suffix else None
        delta_dataset_output_path = _append_timestamp_to_path(
            requested_output, run_timestamp, suffix=delta_dataset_suffix
        )

        if ppo_runner.delta_policy is None:
            print(
                "[INFO]: Delta-model dataset logging enabled without frozen delta policy; "
                "recording active policy obs/actions."
            )
        else:
            if record_ensemble_stats:
                print(
                    "[INFO]: Delta-model dataset logging enabled; recording frozen delta-policy obs/actions "
                    "and per-step ensemble uncertainty/gate."
                )
            else:
                print("[INFO]: Delta-model dataset logging enabled; recording frozen delta-policy obs/actions only.")
        if delta_dataset_recorder.target_trajectories > 0:
            print(
                f"[INFO]: Collecting {delta_dataset_recorder.target_trajectories} completed trajectories into "
                f"{delta_dataset_output_path}"
            )
        else:
            print(f"[INFO]: Collecting trajectories until exit into {delta_dataset_output_path}")

    state_action_recorder = None
    state_action_output_path = None
    if args_cli.record_state_action_trajectories:
        control_dt = _resolve_control_dt(env)
        if control_dt <= 0.0:
            print("[WARN]: Could not resolve control dt; using fallback fps=1.0 for state-action dataset logging.")
            control_dt = 1.0
        state_action_fps = 1.0 / control_dt
        state_action_recorder = StateActionTrajectoryRecorder(
            num_envs=env.num_envs,
            fps=state_action_fps,
            target_trajectories=args_cli.state_action_target_trajectories,
        )
        checkpoint_stem = os.path.splitext(os.path.basename(resume_path))[0]
        default_state_action_npz = os.path.join(
            os.path.dirname(resume_path),
            "state_action_datasets",
            f"{checkpoint_stem if checkpoint_stem else 'policy'}_state_action.npz",
        )
        requested_output = args_cli.output_state_action_npz or default_state_action_npz
        requested_output = os.path.abspath(os.path.expanduser(requested_output))
        run_timestamp = datetime.now().strftime("%Y-%m-%d_%H-%M-%S")
        state_action_output_path = _append_timestamp_to_path(requested_output, run_timestamp)
        print("[INFO]: State-action trajectory logging enabled.")
        if state_action_recorder.target_trajectories > 0:
            print(
                f"[INFO]: Collecting {state_action_recorder.target_trajectories} completed trajectories into "
                f"{state_action_output_path}"
            )
        else:
            print(f"[INFO]: Collecting state-action trajectories until exit into {state_action_output_path}")

    # Export ONNX only for motion-command tasks, since exporter metadata expects `commands.motion`.
    if _has_motion_command(env_cfg):
        export_model_dir = os.path.join(os.path.dirname(resume_path), "exported")
        checkpoint_stem = os.path.splitext(os.path.basename(resume_path))[0]
        onnx_filename = f"{checkpoint_stem if checkpoint_stem else 'policy'}.onnx"

        export_motion_policy_as_onnx(
            env.unwrapped,
            ppo_runner.alg.policy,
            normalizer=ppo_runner.obs_normalizer,
            path=export_model_dir,
            filename=onnx_filename,
        )
        attach_onnx_metadata(
            env.unwrapped,
            args_cli.wandb_path if args_cli.wandb_path else "none",
            export_model_dir,
            filename=onnx_filename,
        )
        print(f"[INFO]: Exported ONNX policy to: {os.path.join(export_model_dir, onnx_filename)}")
    else:
        print("[INFO]: Skipping ONNX export for tasks without a motion command.")
    obs_split_cache: dict[str, tuple[list[str], list[int], list[str]]] = {}
    shutdown_requested_by_target = False

    # Force a real env reset before the first policy step, then patch up the motion-command caches
    # that are otherwise only refreshed after the first command-manager compute.
    obs, _ = env.reset()
    _bootstrap_motion_reference_startup(env)
    obs, _ = env.get_observations()
    prev_motion_time_steps = _get_motion_time_steps(env.unwrapped)
    timestep = 0
    # simulate environment
    while simulation_app.is_running():
        # run everything in inference mode
        with torch.inference_mode():
            # agent stepping
            policy_actions = policy(obs)
            actions = policy_actions
            if args_cli.replay_motion_actions_only:
                actions = torch.zeros_like(policy_actions)

            delta_log_obs = obs
            delta_log_obs_group = "policy"
            delta_log_actions = policy_actions
            if ppo_runner.delta_policy is not None:
                # Inject same-step base-policy action for delta-policy current_action input.
                ppo_runner._set_delta_base_action_buffer(actions)
                # For Delta-A finetune tasks, infer and inject frozen delta before stepping env.
                delta_obs = ppo_runner._compute_delta_policy_obs()
                delta_actions = ppo_runner._compute_delta_actions(delta_obs)
                if delta_actions is not None:
                    ppo_runner._set_delta_action_buffer(delta_actions)
                    delta_log_obs = delta_obs
                    delta_log_obs_group = ppo_runner.delta_policy_obs_group
                    delta_log_actions = delta_actions

            if delta_dataset_recorder is not None:
                obs_components = _resolve_obs_components_for_logging(
                    env=env.unwrapped,
                    group_name=delta_log_obs_group,
                    obs_batch=delta_log_obs,
                    split_cache=obs_split_cache,
                )
                logged_components = dict(obs_components)
                logged_components.update(_resolve_robot_body_world_components_for_logging(env.unwrapped))
                logged_components.update(_resolve_motion_command_components_for_logging(env.unwrapped))
                auxiliary_components = None
                if delta_dataset_recorder.record_ensemble_stats:
                    auxiliary_components = {
                        delta_dataset_recorder.ensemble_uncertainty_key: ppo_runner.delta_policy_last_uncertainty,
                        delta_dataset_recorder.ensemble_gate_key: ppo_runner.delta_policy_last_gate,
                    }
                delta_dataset_recorder.append_step(
                    logged_components,
                    delta_log_actions,
                    auxiliary_components=auxiliary_components,
                )
            # env stepping
            # actions.zero_()
            # print(actions)
            # import ipdb;ipdb.set_trace()

            obs, _, dones, _ = env.step(actions)

            if state_action_recorder is not None:
                state_components = _resolve_robot_state_components_for_logging(env.unwrapped)
                if len(state_components) == 0:
                    raise RuntimeError(
                        "State-action trajectory logging is enabled but robot state tensors could not be resolved."
                    )
                state_action_recorder.append_step(state_components, actions)

            if delta_dataset_recorder is not None or state_action_recorder is not None:
                done_source = dones
                termination_manager = getattr(env.unwrapped, "termination_manager", None)
                if termination_manager is not None and hasattr(termination_manager, "terminated"):
                    term_done = termination_manager.terminated
                    try:
                        done_source = torch.logical_or(done_source.to(dtype=torch.bool), term_done.to(dtype=torch.bool))
                    except Exception:
                        done_source = dones

                # Also treat reference-motion rollover as trajectory completion.
                current_motion_time_steps = _get_motion_time_steps(env.unwrapped)
                if prev_motion_time_steps is not None and current_motion_time_steps is not None:
                    try:
                        motion_rollover_done = current_motion_time_steps < prev_motion_time_steps
                        done_source = torch.logical_or(done_source.to(dtype=torch.bool), motion_rollover_done.to(dtype=torch.bool))
                    except Exception:
                        pass
                prev_motion_time_steps = current_motion_time_steps

            if delta_dataset_recorder is not None:
                gained = delta_dataset_recorder.finalize_done(done_source)
                if gained > 0 and delta_dataset_recorder.target_trajectories > 0:
                    shown = min(delta_dataset_recorder.collected_count, delta_dataset_recorder.target_trajectories)
                    print(
                        f"[INFO]: Collected delta trajectories: {shown}/"
                        f"{delta_dataset_recorder.target_trajectories}"
                    )
                if delta_dataset_recorder.has_reached_target():
                    print("[INFO]: Reached requested delta trajectory count. Stopping play loop.")
                    shutdown_requested_by_target = True
                    break

            if state_action_recorder is not None:
                gained = state_action_recorder.finalize_done(done_source)
                if gained > 0 and state_action_recorder.target_trajectories > 0:
                    shown = min(state_action_recorder.collected_count, state_action_recorder.target_trajectories)
                    print(
                        f"[INFO]: Collected state-action trajectories: {shown}/"
                        f"{state_action_recorder.target_trajectories}"
                    )
                if state_action_recorder.has_reached_target():
                    print("[INFO]: Reached requested state-action trajectory count. Stopping play loop.")
                    shutdown_requested_by_target = True
                    break
        if args_cli.video:
            timestep += 1
            print(f"timestep: {timestep}")
            # Exit the play loop after recording one video
            if timestep == args_cli.video_length:
                break

    if delta_dataset_recorder is not None and delta_dataset_output_path is not None:
        saved_path = delta_dataset_recorder.save_dataset(
            output_path=delta_dataset_output_path,
            include_partial=bool(args_cli.delta_dataset_include_partial),
        )
        if saved_path is None:
            print("[WARN]: Delta-model dataset logging was enabled but no samples were collected.")
        else:
            print(
                f"[INFO]: Saved delta-model dataset to {saved_path} "
                f"(trajectories={delta_dataset_recorder.saved_count}, "
                f"motion_length={delta_dataset_recorder.saved_motion_length})."
            )

    if state_action_recorder is not None and state_action_output_path is not None:
        saved_path = state_action_recorder.save_dataset(
            output_path=state_action_output_path,
            include_partial=bool(args_cli.state_action_include_partial),
        )
        if saved_path is None:
            print("[WARN]: State-action trajectory logging was enabled but no samples were collected.")
        else:
            print(
                f"[INFO]: Saved state-action dataset to {saved_path} "
                f"(trajectories={state_action_recorder.saved_count}, "
                f"motion_length={state_action_recorder.saved_motion_length})."
            )

    # Clean up any externally injected action buffers.
    ppo_runner._clear_delta_action_buffer()

    # close the simulator
    env.close()

    if shutdown_requested_by_target:
        print("[INFO]: Target trajectory count met. Shutting down simulator app.")
        simulation_app.close()


if __name__ == "__main__":
    # run the main function
    main()
    # close sim app
    simulation_app.close()
