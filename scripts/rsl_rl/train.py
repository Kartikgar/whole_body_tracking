# Copyright (c) 2022-2024, The Isaac Lab Project Developers.
# All rights reserved.
#
# SPDX-License-Identifier: BSD-3-Clause

"""Script to train RL agent with RSL-RL."""

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
parser.add_argument("--video_interval", type=int, default=2000, help="Interval between video recordings (in steps).")
parser.add_argument("--num_envs", type=int, default=None, help="Number of environments to simulate.")
parser.add_argument("--task", type=str, default=None, help="Name of the task.")
parser.add_argument("--seed", type=int, default=None, help="Seed used for the environment")
parser.add_argument("--max_iterations", type=int, default=None, help="RL Policy training iterations.")
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
        "Pre-clamp gain for `--delta_action_space com_force`. "
        "Final applied Fx/Fy/Fz are always clamped to [-1, 1]."
    ),
)
parser.add_argument(
    "--registry_name",
    type=str,
    default=None,
    help="Optional wandb artifact name. Used only when --motion_file is not provided.",
)
parser.add_argument(
    "--motion_file",
    type=str,
    default=None,
    help="Optional local path to motion .npz. If provided, wandb registry is skipped.",
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
import torch
from datetime import datetime

from isaaclab.envs import (
    DirectMARLEnv,
    DirectMARLEnvCfg,
    DirectRLEnvCfg,
    ManagerBasedRLEnvCfg,
    multi_agent_to_single_agent,
)
from isaaclab.utils.dict import print_dict
from isaaclab.utils.io import dump_pickle, dump_yaml
from isaaclab_rl.rsl_rl import RslRlOnPolicyRunnerCfg, RslRlVecEnvWrapper
from isaaclab_tasks.utils.hydra import hydra_task_config

# Import extensions to set up environment tasks
import whole_body_tracking.tasks  # noqa: F401
from whole_body_tracking.utils.my_on_policy_runner import MotionOnPolicyRunner as OnPolicyRunner

torch.backends.cuda.matmul.allow_tf32 = True
torch.backends.cudnn.allow_tf32 = True
torch.backends.cudnn.deterministic = False
torch.backends.cudnn.benchmark = False

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
            com_force_cfg = mdp.DeltaComForceActionCfg(
                asset_name=joint_pos_cfg.asset_name,
                joint_names=joint_pos_cfg.joint_names,
                use_default_offset=getattr(joint_pos_cfg, "use_default_offset", True),
                motion_command_name=getattr(joint_pos_cfg, "motion_command_name", "motion"),
                require_motion_action=getattr(joint_pos_cfg, "require_motion_action", True),
                force_body_name=force_body_name,
                force_scale=args_cli.delta_com_force_scale,
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
                f"(Fx, Fy, Fz), scale={args_cli.delta_com_force_scale} N, body='{force_body_name}'."
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


@hydra_task_config(args_cli.task, "rsl_rl_cfg_entry_point")
def main(env_cfg: ManagerBasedRLEnvCfg | DirectRLEnvCfg | DirectMARLEnvCfg, agent_cfg: RslRlOnPolicyRunnerCfg):
    """Train with RSL-RL agent."""
    # override configurations with non-hydra CLI arguments
    agent_cfg = cli_args.update_rsl_rl_cfg(agent_cfg, args_cli)
    env_cfg.scene.num_envs = args_cli.num_envs if args_cli.num_envs is not None else env_cfg.scene.num_envs
    agent_cfg.max_iterations = (
        args_cli.max_iterations if args_cli.max_iterations is not None else agent_cfg.max_iterations
    )

    # set the environment seed
    # note: certain randomizations occur in the environment initialization so we set the seed here
    env_cfg.seed = agent_cfg.seed
    env_cfg.sim.device = args_cli.device if args_cli.device is not None else env_cfg.sim.device
    _configure_delta_action_space(env_cfg)

    # load motion file from local path or wandb registry
    registry_name = None
    if args_cli.motion_file is not None:
        motion_file = os.path.abspath(os.path.expanduser(args_cli.motion_file))
        if not os.path.isfile(motion_file):
            raise FileNotFoundError(f"Motion file not found: {motion_file}")
        print(f"[INFO]: Using local motion file: {motion_file}")
        env_cfg.commands.motion.motion_file = motion_file
    else:
        if args_cli.registry_name is None:
            raise ValueError("Provide --motion_file, or provide --registry_name to fetch motion.npz from wandb.")
        registry_name = args_cli.registry_name
        if ":" not in registry_name:  # Check if the registry name includes alias, if not, append ":latest"
            registry_name += ":latest"
        import pathlib

        import wandb

        print(f"[INFO]: Downloading motion artifact from wandb: {registry_name}")
        api = wandb.Api()
        artifact = api.artifact(registry_name)
        env_cfg.commands.motion.motion_file = str(pathlib.Path(artifact.download()) / "motion.npz")

    # specify directory for logging experiments
    log_root_path = os.path.join("logs", "rsl_rl", agent_cfg.experiment_name)
    log_root_path = os.path.abspath(log_root_path)
    print(f"[INFO] Logging experiment in directory: {log_root_path}")
    # specify directory for logging runs: {time-stamp}_{run_name}
    log_dir = datetime.now().strftime("%Y-%m-%d_%H-%M-%S")
    if agent_cfg.run_name:
        log_dir += f"_{agent_cfg.run_name}"
    log_dir = os.path.join(log_root_path, log_dir)

    # create isaac environment
    env = gym.make(args_cli.task, cfg=env_cfg, render_mode="rgb_array" if args_cli.video else None)
    # wrap for video recording
    if args_cli.video:
        video_kwargs = {
            "video_folder": os.path.join(log_dir, "videos", "train"),
            "step_trigger": lambda step: step % args_cli.video_interval == 0,
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

    # create runner from rsl-rl
    runner = OnPolicyRunner(
        env, agent_cfg.to_dict(), log_dir=log_dir, device=agent_cfg.device, registry_name=registry_name
    )
    # write git state to logs
    runner.add_git_repo_to_log(__file__)
    # Save resume path before creating a new log_dir.
    # We treat --checkpoint as an explicit file path for resume.
    # Alternatively, --wandb_path can be used to fetch a checkpoint from a wandb run.
    if agent_cfg.resume:
        if args_cli.checkpoint is not None:
            resume_path = os.path.abspath(os.path.expanduser(args_cli.checkpoint))
            if not os.path.isfile(resume_path):
                raise FileNotFoundError(
                    f"Checkpoint file not found: {resume_path}. "
                    "Pass `--checkpoint /abs/path/to/model_x.pt`."
                )
        elif args_cli.wandb_path:
            import wandb

            run_path = args_cli.wandb_path
            api = wandb.Api()
            if "model" in args_cli.wandb_path:
                run_path = "/".join(args_cli.wandb_path.split("/")[:-1])
            wandb_run = api.run(run_path)
            files = [file.name for file in wandb_run.files() if "model" in file.name]
            if len(files) == 0:
                raise FileNotFoundError(f"No model checkpoint files found in wandb run: {run_path}")
            if "model" in args_cli.wandb_path:
                file = args_cli.wandb_path.split("/")[-1]
            else:
                file = max(files, key=lambda x: int(x.split("_")[1].split(".")[0]))

            download_dir = os.path.abspath(os.path.join("logs", "rsl_rl", "temp"))
            os.makedirs(download_dir, exist_ok=True)
            wandb_file = wandb_run.file(str(file))
            wandb_file.download(download_dir, replace=True)
            resume_path = os.path.join(download_dir, file)
        else:
            raise ValueError(
                "`--resume True` requires either `--checkpoint /abs/path/to/model_x.pt` "
                "or `--wandb_path entity/project/run[/model_x.pt]`."
            )
        print(f"[INFO]: Loading model checkpoint from: {resume_path}")
        # load previously trained model
        runner.load(resume_path)

    # dump the configuration into log-directory
    dump_yaml(os.path.join(log_dir, "params", "env.yaml"), env_cfg)
    dump_yaml(os.path.join(log_dir, "params", "agent.yaml"), agent_cfg)
    dump_pickle(os.path.join(log_dir, "params", "env.pkl"), env_cfg)
    dump_pickle(os.path.join(log_dir, "params", "agent.pkl"), agent_cfg)

    # run training
    runner.learn(num_learning_iterations=agent_cfg.max_iterations, init_at_random_ep_len=True)

    # close the simulator
    env.close()


if __name__ == "__main__":
    # run the main function
    main()
    # close sim app
    simulation_app.close()
