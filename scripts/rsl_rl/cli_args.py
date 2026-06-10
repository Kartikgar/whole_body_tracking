from __future__ import annotations

import argparse
from typing import TYPE_CHECKING

if TYPE_CHECKING:
    from isaaclab_rl.rsl_rl import RslRlOnPolicyRunnerCfg


def add_rsl_rl_args(parser: argparse.ArgumentParser):
    """Add RSL-RL arguments to the parser.

    Args:
        parser: The parser to add the arguments to.
    """
    # create a new argument group
    arg_group = parser.add_argument_group("rsl_rl", description="Arguments for RSL-RL agent.")
    # -- experiment arguments
    arg_group.add_argument(
        "--experiment_name", type=str, default=None, help="Name of the experiment folder where logs will be stored."
    )
    arg_group.add_argument("--run_name", type=str, default=None, help="Run name suffix to the log directory.")
    # -- load arguments
    arg_group.add_argument("--resume", type=bool, default=None, help="Whether to resume from a checkpoint.")
    arg_group.add_argument(
        "--load_run",
        type=str,
        default=None,
        help="Deprecated for training resume; ignored.",
    )
    arg_group.add_argument(
        "--checkpoint",
        type=str,
        default=None,
        help="Full path to checkpoint file to resume from (for example: /abs/path/to/model_1000.pt).",
    )
    # -- logger arguments
    arg_group.add_argument(
        "--logger", type=str, default=None, choices={"wandb", "tensorboard", "neptune"}, help="Logger module to use."
    )
    arg_group.add_argument(
        "--log_project_name", type=str, default=None, help="Name of the logging project when using wandb or neptune."
    )
    arg_group.add_argument(
        "--wandb_path",
        type=str,
        default=None,
        help="Wandb run path for checkpoint resume (entity/project/run or entity/project/run/model_x.pt).",
    )
    arg_group.add_argument(
        "--delta_policy_checkpoints",
        nargs="+",
        type=str,
        default=None,
        help=(
            "One or more frozen open-loop delta-policy checkpoints for delta-action finetuning runners. "
            "Pass multiple paths to enable uncertainty-gated ensemble inference."
        ),
    )
    arg_group.add_argument(
        "--delta_policy_uncertainty_gate_scale",
        type=float,
        default=None,
        help=(
            "Exponential gating scale applied to ensemble epistemic uncertainty. "
            "Final delta = exp(-scale * uncertainty) * mean(delta_ensemble)."
        ),
    )
    arg_group.add_argument(
        "--delta_policy_clip_actions",
        type=float,
        default=None,
        help=(
            "Symmetric clip bound applied to frozen delta-policy actions during finetune/play rollouts. "
            "If omitted, uses the runner default (env clip_actions when config is None)."
        ),
    )


def parse_rsl_rl_cfg(task_name: str, args_cli: argparse.Namespace) -> RslRlOnPolicyRunnerCfg:
    """Parse configuration for RSL-RL agent based on inputs.

    Args:
        task_name: The name of the environment.
        args_cli: The command line arguments.

    Returns:
        The parsed configuration for RSL-RL agent based on inputs.
    """
    from isaaclab_tasks.utils.parse_cfg import load_cfg_from_registry

    # load the default configuration
    rslrl_cfg: RslRlOnPolicyRunnerCfg = load_cfg_from_registry(task_name, "rsl_rl_cfg_entry_point")
    rslrl_cfg = update_rsl_rl_cfg(rslrl_cfg, args_cli)
    return rslrl_cfg


def update_rsl_rl_cfg(agent_cfg: RslRlOnPolicyRunnerCfg, args_cli: argparse.Namespace):
    """Update configuration for RSL-RL agent based on inputs.

    Args:
        agent_cfg: The configuration for RSL-RL agent.
        args_cli: The command line arguments.

    Returns:
        The updated configuration for RSL-RL agent based on inputs.
    """
    # override the default configuration with CLI arguments
    if hasattr(args_cli, "seed") and args_cli.seed is not None:
        agent_cfg.seed = args_cli.seed
    if args_cli.resume is not None:
        agent_cfg.resume = args_cli.resume
    if args_cli.load_run is not None:
        agent_cfg.load_run = args_cli.load_run
    if args_cli.checkpoint is not None:
        agent_cfg.load_checkpoint = args_cli.checkpoint
    if args_cli.run_name is not None:
        agent_cfg.run_name = args_cli.run_name
    if args_cli.logger is not None:
        agent_cfg.logger = args_cli.logger
    if args_cli.delta_policy_checkpoints is not None and hasattr(agent_cfg, "delta_policy_checkpoints"):
        agent_cfg.delta_policy_checkpoints = list(args_cli.delta_policy_checkpoints)
    if (
        hasattr(args_cli, "delta_policy_uncertainty_gate_scale")
        and args_cli.delta_policy_uncertainty_gate_scale is not None
        and hasattr(agent_cfg, "delta_policy_uncertainty_gate_scale")
    ):
        agent_cfg.delta_policy_uncertainty_gate_scale = args_cli.delta_policy_uncertainty_gate_scale
    if (
        hasattr(args_cli, "delta_policy_clip_actions")
        and args_cli.delta_policy_clip_actions is not None
        and hasattr(agent_cfg, "delta_policy_clip_actions")
    ):
        agent_cfg.delta_policy_clip_actions = args_cli.delta_policy_clip_actions
    # set the project name for wandb and neptune
    if agent_cfg.logger in {"wandb", "neptune"} and args_cli.log_project_name:
        agent_cfg.wandb_project = args_cli.log_project_name
        agent_cfg.neptune_project = args_cli.log_project_name

    return agent_cfg
