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
    help="Enable adaptive reference time-step sampling (disabled by default in play).",
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

# Import extensions to set up environment tasks
import whole_body_tracking.tasks  # noqa: F401
from whole_body_tracking.utils.exporter import attach_onnx_metadata, export_motion_policy_as_onnx
from whole_body_tracking.utils.my_on_policy_runner import MotionOnPolicyRunner as OnPolicyRunner

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
        # COM-force mode is currently supported for open-loop delta action tasks.
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
                f"clip={force_clip}, body='{force_body_name}'."
            )
            print("[INFO]: Disabled action-penalty rewards for COM-force mode.")
        else:
            print(
                "[WARN]: `--delta_action_space com_force` is currently supported only for "
                "open-loop delta-action tasks. "
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


@hydra_task_config(args_cli.task, "rsl_rl_cfg_entry_point")
def main(env_cfg: ManagerBasedRLEnvCfg | DirectRLEnvCfg | DirectMARLEnvCfg, agent_cfg: RslRlOnPolicyRunnerCfg):
    """Play with RSL-RL agent."""
    agent_cfg: RslRlOnPolicyRunnerCfg = cli_args.parse_rsl_rl_cfg(args_cli.task, args_cli)
    env_cfg.scene.num_envs = args_cli.num_envs if args_cli.num_envs is not None else env_cfg.scene.num_envs
    if _has_motion_command(env_cfg):
        if not args_cli.enable_adaptive_reference_sampling:
            env_cfg.commands.motion.adaptive_alpha = 0.0
            env_cfg.commands.motion.sample_time_steps = False
            print(
                "[INFO]: Disabled adaptive reference sampling for play "
                "(commands.motion.adaptive_alpha=0.0, commands.motion.sample_time_steps=False)."
            )
        else:
            env_cfg.commands.motion.sample_time_steps = True
            print("[INFO]: Adaptive reference sampling is enabled for play.")
    elif args_cli.enable_adaptive_reference_sampling:
        print("[INFO]: This task has no motion command; ignoring --enable_adaptive_reference_sampling.")
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

    # wrap for video recording
    if args_cli.video:
        video_kwargs = {
            "video_folder": os.path.join(log_dir, "videos", "play"),
            "step_trigger": lambda step: step == 0,
            "video_length": args_cli.video_length,
            "disable_logger": True,
        }
        print("[INFO] Recording videos during training.")
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
    # reset environment
    obs, _ = env.get_observations()
    timestep = 0
    # simulate environment
    while simulation_app.is_running():
        # run everything in inference mode
        with torch.inference_mode():
            # agent stepping
            actions = policy(obs)
            if args_cli.replay_motion_actions_only:
                actions.zero_()
            if ppo_runner.delta_policy is not None:
                # Inject same-step base-policy action for delta-policy current_action input.
                ppo_runner._set_delta_base_action_buffer(actions)
                # For Delta-A finetune tasks, infer and inject frozen delta before stepping env.
                delta_obs = ppo_runner._compute_delta_policy_obs()
                delta_actions = ppo_runner._compute_delta_actions(delta_obs)
                if delta_actions is not None:
                    ppo_runner._set_delta_action_buffer(delta_actions)
            # env stepping
            # actions.zero_()
            # print(actions)
            # import ipdb;ipdb.set_trace()

            obs, _, _, infos = env.step(actions)
        if args_cli.video:
            timestep += 1
            # Exit the play loop after recording one video
            if timestep == args_cli.video_length:
                break

    # Clean up any externally injected action buffers.
    ppo_runner._clear_delta_action_buffer()

    # close the simulator
    env.close()


if __name__ == "__main__":
    # run the main function
    main()
    # close sim app
    simulation_app.close()
