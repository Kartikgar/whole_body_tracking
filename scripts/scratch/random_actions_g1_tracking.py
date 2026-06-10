#!/usr/bin/env python3
"""Load G1 in Tracking-Flat-G1-v0 and step with random joint actions.

Useful for sanity-checking env creation, action dimensions, and sim stepping
without a trained policy.

Example:

    python scripts/scratch/random_actions_g1_tracking.py \\
        --motion_file data/LAFAN1_Retargeting_Dataset/g1/walk1_subject1.npz \\
        --num_envs 4 \\
        --num_steps 500
"""

from __future__ import annotations

import argparse
import os
import sys

from isaaclab.app import AppLauncher

parser = argparse.ArgumentParser(description="Step Tracking-Flat-G1-v0 with random actions.")
parser.add_argument(
    "--motion_file",
    type=str,
    required=True,
    help="Path to a motion .npz required by the tracking motion command.",
)
parser.add_argument("--num_envs", type=int, default=1, help="Number of parallel environments.")
parser.add_argument("--num_steps", type=int, default=1500, help="Number of control steps to run.")
parser.add_argument("--seed", type=int, default=42, help="Random seed for action sampling.")
parser.add_argument(
    "--action_low",
    type=float,
    default=-1.0,
    help="Lower bound for uniform random actions (policy action space).",
)
parser.add_argument(
    "--action_high",
    type=float,
    default=1.0,
    help="Upper bound for uniform random actions (policy action space).",
)
parser.add_argument(
    "--disable_dr",
    action="store_true",
    default=False,
    help="Disable domain randomization events and observation corruption/noise.",
)
parser.add_argument("--task", type=str, default="Tracking-Flat-G1-v0", help="Gym task id.")
AppLauncher.add_app_launcher_args(parser)
args_cli = parser.parse_args()

app_launcher = AppLauncher(args_cli)
simulation_app = app_launcher.app

import gymnasium as gym
import torch

import whole_body_tracking.tasks  # noqa: F401  # register gym tasks
from isaaclab.envs import ManagerBasedRLEnvCfg
from isaaclab_tasks.utils.parse_cfg import parse_env_cfg


def _disable_domain_randomization(env_cfg: ManagerBasedRLEnvCfg):
    event_manager = getattr(env_cfg, "events", None)
    if event_manager is not None:
        for term_name in list(vars(event_manager).keys()):
            if term_name.startswith("_"):
                continue
            setattr(event_manager, term_name, None)

    observations = getattr(env_cfg, "observations", None)
    if observations is not None:
        for group_name in list(vars(observations).keys()):
            if group_name.startswith("_"):
                continue
            group_cfg = getattr(observations, group_name, None)
            if group_cfg is not None and hasattr(group_cfg, "enable_corruption"):
                group_cfg.enable_corruption = False


def main() -> int:
    motion_file = os.path.abspath(os.path.expanduser(args_cli.motion_file))
    if not os.path.isfile(motion_file):
        print(f"[ERROR] Motion file not found: {motion_file}", file=sys.stderr)
        return 1

    env_cfg = parse_env_cfg(
        args_cli.task,
        device=args_cli.device if getattr(args_cli, "device", None) is not None else "cuda:0",
        num_envs=args_cli.num_envs,
        use_fabric=not args_cli.disable_fabric if hasattr(args_cli, "disable_fabric") else None,
    )
    env_cfg.commands.motion.motion_file = motion_file
    if args_cli.disable_dr:
        _disable_domain_randomization(env_cfg)

    print(f"[INFO] Task: {args_cli.task}")
    print(f"[INFO] Motion file: {motion_file}")
    print(f"[INFO] num_envs={args_cli.num_envs}, num_steps={args_cli.num_steps}")

    env = gym.make(args_cli.task, cfg=env_cfg)
    unwrapped = env.unwrapped
    device = unwrapped.device
    action_dim = unwrapped.action_manager.total_action_dim

    print(f"[INFO] Device: {device}")
    print(f"[INFO] Action dim: {action_dim}")
    print(f"[INFO] Control dt: {unwrapped.step_dt:.4f}s")

    print("The joint names in order are:")
    print(unwrapped.scene["robot"].joint_names)

    print("The joint names in order are:")
    print(unwrapped.action_manager._terms["joint_pos"]._joint_names)

    torch.manual_seed(args_cli.seed)
    generator = torch.Generator(device=device)
    generator.manual_seed(args_cli.seed)

    obs, _ = env.reset()
    terminated_count = 0
    truncated_count = 0

    for step_idx in range(args_cli.num_steps):
        if not simulation_app.is_running():
            print("[INFO] Simulation app stopped; exiting loop.")
            break

        actions = torch.empty((unwrapped.num_envs, action_dim), device=device)
        actions.uniform_(args_cli.action_low, args_cli.action_high, generator=generator)

        obs, rewards, terminated, truncated, extras = env.step(actions)
        terminated_count += int(terminated.sum().item())
        truncated_count += int(truncated.sum().item())

        if (step_idx + 1) % 100 == 0 or step_idx == 0:
            reward_mean = float(rewards.mean().item())
            print(
                f"[INFO] step={step_idx + 1:5d}  reward_mean={reward_mean:8.4f}  "
                f"terminated={int(terminated.sum().item())}  truncated={int(truncated.sum().item())}"
            )

    print(
        f"[INFO] Finished {min(args_cli.num_steps, step_idx + 1)} steps. "
        f"Total terminated events={terminated_count}, truncated events={truncated_count}."
    )
    env.close()
    simulation_app.close()
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
