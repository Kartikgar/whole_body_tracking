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
    "--manual_switch_after_steps",
    type=int,
    default=None,
    help=(
        "If set, manually override hierarchical switch action in play mode. "
        "Use `--manual_switch_prob_before` before this step and `--manual_switch_prob_after` from this step onward."
    ),
)
parser.add_argument(
    "--manual_switch_prob_before",
    type=float,
    default=0.0,
    help="Manual hierarchical switch probability before --manual_switch_after_steps.",
)
parser.add_argument(
    "--manual_switch_prob_after",
    type=float,
    default=1.0,
    help="Manual hierarchical switch probability at/after --manual_switch_after_steps.",
)
parser.add_argument(
    "--low_level_policy_1_motion_file",
    type=str,
    default=None,
    help="Optional local path to motion .npz for low-level policy #1 in hierarchical switch tasks.",
)
parser.add_argument(
    "--low_level_policy_2_motion_file",
    type=str,
    default=None,
    help="Optional local path to motion .npz for low-level policy #2 in hierarchical switch tasks.",
)
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
# append RSL-RL cli arguments
cli_args.add_rsl_rl_args(parser)
# append AppLauncher cli args
AppLauncher.add_app_launcher_args(parser)
args_cli, hydra_args = parser.parse_known_args()
# Fail fast on unknown CLI-style flags instead of forwarding them to Hydra, which raises
# less actionable lexer errors for tokens like mistyped `--flag` names.
unknown_cli_flags = [arg for arg in hydra_args if arg.startswith("--")]
if unknown_cli_flags:
    parser.error(
        "Unrecognized arguments: "
        + " ".join(unknown_cli_flags)
        + ". If these are Hydra overrides, pass them as key=value (without leading '--')."
    )
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

from rsl_rl.runners import OnPolicyRunner

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


def _resolve_local_motion_file(path: str, arg_name: str) -> str:
    motion_file = os.path.abspath(os.path.expanduser(path))
    if not os.path.isfile(motion_file):
        raise FileNotFoundError(f"{arg_name} not found: {motion_file}")
    return motion_file


def _switch_probability_to_action_value(probability: float, use_sigmoid: bool, device: torch.device) -> torch.Tensor:
    """Convert desired switch probability to the action value expected by HighLevelPolicySwitchAction."""
    probability = float(probability)
    if probability < 0.0 or probability > 1.0:
        raise ValueError(f"Switch probability must be in [0, 1], got {probability}.")
    if use_sigmoid:
        # Inverse sigmoid so the processed action matches the requested probability.
        eps = 1.0e-6
        probability = min(max(probability, eps), 1.0 - eps)
        return torch.logit(torch.tensor(probability, device=device, dtype=torch.float32))
    return torch.tensor(probability, device=device, dtype=torch.float32)


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
            "Hierarchical switch play requires both low-level checkpoints. Provide "
            "`--low_level_policy_1_checkpoint` and `--low_level_policy_2_checkpoint`."
        )

    print(
        "[INFO]: Hierarchical switch policies configured for play: "
        f"policy_1='{joint_pos_cfg.policy_1_checkpoint}', "
        f"policy_2='{joint_pos_cfg.policy_2_checkpoint}'."
    )


def _disable_adaptive_sampling_for_play(env_cfg: ManagerBasedRLEnvCfg | DirectRLEnvCfg | DirectMARLEnvCfg):
    """Disable adaptive resampling for motion commands in play mode."""
    commands_cfg = getattr(env_cfg, "commands", None)
    if commands_cfg is None:
        return

    disabled_terms: list[str] = []
    for command_name in ("motion", "motion_policy_1", "motion_policy_2"):
        command_cfg = getattr(commands_cfg, command_name, None)
        if command_cfg is None:
            continue

        updated = False
        # Optional explicit switch for future compatibility.
        if hasattr(command_cfg, "adaptive_sampling"):
            command_cfg.adaptive_sampling = False
            updated = True
        # Force uniform sampling by freezing failure-bin updates.
        if hasattr(command_cfg, "adaptive_alpha"):
            command_cfg.adaptive_alpha = 0.0
            updated = True
        # Keep a positive uniform prior so probabilities remain well-defined.
        if hasattr(command_cfg, "adaptive_uniform_ratio"):
            command_cfg.adaptive_uniform_ratio = max(float(command_cfg.adaptive_uniform_ratio), 1.0)
            updated = True

        if updated:
            disabled_terms.append(command_name)

    if disabled_terms:
        print(
            "[INFO]: Disabled adaptive sampling for play on motion command(s): "
            f"{', '.join(disabled_terms)}."
        )


@hydra_task_config(args_cli.task, "rsl_rl_cfg_entry_point")
def main(env_cfg: ManagerBasedRLEnvCfg | DirectRLEnvCfg | DirectMARLEnvCfg, agent_cfg: RslRlOnPolicyRunnerCfg):
    """Play with RSL-RL agent."""
    agent_cfg: RslRlOnPolicyRunnerCfg = cli_args.parse_rsl_rl_cfg(args_cli.task, args_cli)
    env_cfg.scene.num_envs = args_cli.num_envs if args_cli.num_envs is not None else env_cfg.scene.num_envs
    _configure_hierarchical_switch_policies(env_cfg)

    # specify directory for logging experiments
    log_root_path = os.path.join("logs", "rsl_rl", agent_cfg.experiment_name)
    log_root_path = os.path.abspath(log_root_path)

    if args_cli.wandb_path:
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

        if args_cli.motion_file is not None:
            print(f"[INFO]: Using motion file from CLI: {args_cli.motion_file}")
            env_cfg.commands.motion.motion_file = args_cli.motion_file

        art = next((a for a in wandb_run.used_artifacts() if a.type == "motions"), None)
        if art is None:
            print("[WARN] No model artifact found in the run.")
        else:
            env_cfg.commands.motion.motion_file = str(pathlib.Path(art.download()) / "motion.npz")

    else:
        # Treat --checkpoint as an explicit checkpoint file path when provided.
        explicit_checkpoint = args_cli.checkpoint or getattr(agent_cfg, "load_checkpoint", None)
        if explicit_checkpoint is not None:
            resume_path = os.path.abspath(os.path.expanduser(str(explicit_checkpoint)))
            if not os.path.isfile(resume_path):
                raise FileNotFoundError(
                    f"Checkpoint file not found: {resume_path}. "
                    "Pass `--checkpoint /abs/path/to/model_x.pt`."
                )
            print(f"[INFO]: Loading model checkpoint from explicit path: {resume_path}")
        else:
            print(f"[INFO] Loading experiment from directory: {log_root_path}")
            resume_path = get_checkpoint_path(log_root_path, agent_cfg.load_run, agent_cfg.load_checkpoint)
            print(f"[INFO]: Loading model checkpoint from: {resume_path}")

    if args_cli.motion_file is not None:
        resolved_motion_file = _resolve_local_motion_file(args_cli.motion_file, "--motion_file")
        print(f"[INFO]: Using motion file from CLI: {resolved_motion_file}")
        env_cfg.commands.motion.motion_file = resolved_motion_file

    motion_policy_1_cfg = getattr(getattr(env_cfg, "commands", None), "motion_policy_1", None)
    motion_policy_2_cfg = getattr(getattr(env_cfg, "commands", None), "motion_policy_2", None)
    has_hierarchical_motion_commands = motion_policy_1_cfg is not None and motion_policy_2_cfg is not None
    if has_hierarchical_motion_commands:
        using_policy_specific_motion_files = (
            args_cli.low_level_policy_1_motion_file is not None
            or args_cli.low_level_policy_2_motion_file is not None
        )

        if using_policy_specific_motion_files:
            if args_cli.low_level_policy_1_motion_file is None or args_cli.low_level_policy_2_motion_file is None:
                raise ValueError(
                    "Provide both --low_level_policy_1_motion_file and --low_level_policy_2_motion_file for "
                    "hierarchical switch tasks."
                )
            motion_file_policy_1 = _resolve_local_motion_file(
                args_cli.low_level_policy_1_motion_file,
                "--low_level_policy_1_motion_file",
            )
            motion_file_policy_2 = _resolve_local_motion_file(
                args_cli.low_level_policy_2_motion_file,
                "--low_level_policy_2_motion_file",
            )
            motion_policy_1_cfg.motion_file = motion_file_policy_1
            motion_policy_2_cfg.motion_file = motion_file_policy_2
            if getattr(env_cfg.commands, "motion", None) is not None:
                env_cfg.commands.motion.motion_file = motion_file_policy_1
            print(
                "[INFO]: Using hierarchical per-policy motion files for play: "
                f"policy_1='{motion_file_policy_1}', policy_2='{motion_file_policy_2}'."
            )
        else:
            shared_motion_cfg = getattr(env_cfg.commands, "motion", None)
            shared_motion_file = getattr(shared_motion_cfg, "motion_file", None) if shared_motion_cfg is not None else None
            if not shared_motion_file:
                raise ValueError(
                    "Hierarchical switch play requires motion references. Provide --motion_file "
                    "or both --low_level_policy_1_motion_file and --low_level_policy_2_motion_file."
                )
            motion_policy_1_cfg.motion_file = shared_motion_file
            motion_policy_2_cfg.motion_file = shared_motion_file
            print(
                "[INFO]: Using shared motion file for both hierarchical low-level policies in play: "
                f"{shared_motion_file}"
            )
    elif args_cli.low_level_policy_1_motion_file is not None or args_cli.low_level_policy_2_motion_file is not None:
        print(
            "[INFO]: Ignoring --low_level_policy_1_motion_file/--low_level_policy_2_motion_file because this "
            "task does not define hierarchical motion commands."
        )

    _disable_adaptive_sampling_for_play(env_cfg)

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

    # export policy to onnx/jit
    export_model_dir = os.path.join(os.path.dirname(resume_path), "exported")

    export_motion_policy_as_onnx(
        env.unwrapped,
        ppo_runner.alg.policy,
        normalizer=ppo_runner.obs_normalizer,
        path=export_model_dir,
        filename="policy.onnx",
    )
    attach_onnx_metadata(env.unwrapped, args_cli.wandb_path if args_cli.wandb_path else "none", export_model_dir)
    # reset environment
    obs, _ = env.get_observations()
    timestep = 0
    manual_switch_enabled = args_cli.manual_switch_after_steps is not None
    manual_switch_action_before = None
    manual_switch_action_after = None
    manual_switch_has_announced_flip = False
    if manual_switch_enabled:
        if args_cli.manual_switch_after_steps < 0:
            raise ValueError("--manual_switch_after_steps must be >= 0.")

        joint_pos_cfg = getattr(getattr(env_cfg, "actions", None), "joint_pos", None)
        if joint_pos_cfg is None or not hasattr(joint_pos_cfg, "switch_threshold"):
            raise ValueError(
                "--manual_switch_* options require a hierarchical switch task (actions.joint_pos with switch_threshold)."
            )

        switch_uses_sigmoid = bool(getattr(joint_pos_cfg, "use_sigmoid", True))
        manual_switch_action_before = _switch_probability_to_action_value(
            args_cli.manual_switch_prob_before,
            use_sigmoid=switch_uses_sigmoid,
            device=env.unwrapped.device,
        )
        manual_switch_action_after = _switch_probability_to_action_value(
            args_cli.manual_switch_prob_after,
            use_sigmoid=switch_uses_sigmoid,
            device=env.unwrapped.device,
        )
        print(
            "[INFO]: Manual switch override enabled: "
            f"p_before={args_cli.manual_switch_prob_before} until step<{args_cli.manual_switch_after_steps}, "
            f"p_after={args_cli.manual_switch_prob_after} from step>={args_cli.manual_switch_after_steps}."
        )

    # simulate environment
    while simulation_app.is_running():
        # run everything in inference mode
        with torch.inference_mode():
            # agent stepping
            actions = policy(obs)
            if manual_switch_enabled:
                if actions.shape[1] < 1:
                    raise RuntimeError(
                        "Manual switch override requested, but action tensor has no switch dimension (shape="
                        f"{tuple(actions.shape)})."
                    )
                if timestep < args_cli.manual_switch_after_steps:
                    actions[:, 0] = manual_switch_action_before
                else:
                    actions[:, 0] = manual_switch_action_after
                    if not manual_switch_has_announced_flip:
                        print(f"[INFO]: Manual switch flipped at env step {timestep}.")
                        manual_switch_has_announced_flip = True
            # env stepping
            obs, _, _, _ = env.step(actions)
        if args_cli.video:
            timestep += 1
            # Exit the play loop after recording one video
            if timestep == args_cli.video_length:
                break
        else:
            timestep += 1

    # close the simulator
    env.close()


if __name__ == "__main__":
    # run the main function
    main()
    # close sim app
    simulation_app.close()
