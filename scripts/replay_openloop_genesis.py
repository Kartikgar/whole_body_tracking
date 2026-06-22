#!/usr/bin/env python3
"""Open-loop replay of Isaac-logged base actions in Genesis for transfer validation."""

from __future__ import annotations

import argparse
import json
import os
import random
import sys
from datetime import datetime

import numpy as np
import torch

_SCRIPTS_DIR = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
if _SCRIPTS_DIR not in sys.path:
    sys.path.insert(0, _SCRIPTS_DIR)

from sim2sim_genesis.constants import DEFAULT_G1_URDF
from sim2sim_genesis.openloop_replay import build_runner_from_paths


def apply_global_random_seeds(seed: int) -> None:
    seed = int(seed)
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description=(
            "Replay logged state-action NPZ actions open-loop in Genesis from the stored initial states. "
            "Useful for validating replay fidelity and transfer assumptions."
        )
    )
    parser.add_argument(
        "--state_action_npz",
        type=str,
        required=True,
        help="State-action NPZ with initial_* fields and replay actions.",
    )
    parser.add_argument(
        "--policy_path",
        type=str,
        required=True,
        help="Exported base-policy ONNX (metadata only: joint order, action_scale, Kp/Kd).",
    )
    parser.add_argument("--urdf_file", type=str, default=DEFAULT_G1_URDF, help="Genesis G1 URDF path.")
    parser.add_argument("--xml_file", type=str, default=None, help="Optional MJCF fallback.")
    parser.add_argument("--backend", type=str, default="gpu", choices=["cpu", "gpu"], help="Genesis backend.")
    parser.add_argument("--sim_dt", type=float, default=0.001, help="Genesis simulation dt.")
    parser.add_argument("--control_dt", type=float, default=0.02, help="Control dt (must match Isaac logging fps).")
    parser.add_argument(
        "--num_envs",
        type=int,
        default=1,
        help="Parallel Genesis environments for batched open-loop replay (match Isaac --num_envs for speed).",
    )
    parser.add_argument(
        "--trajectory_indices",
        type=str,
        default=None,
        help="Trajectories to replay: '0:10', '3,5,7', or single index. Default: all.",
    )
    parser.add_argument("--output_replay_npz", type=str, default=None, help="Optional Genesis replay NPZ output.")
    parser.add_argument("--report_csv", type=str, default=None, help="Optional per-trajectory RMSE CSV.")
    parser.add_argument("--report_json", type=str, default=None, help="Optional summary JSON output.")
    parser.add_argument("--torque_limit", type=float, default=None, help="Optional symmetric torque clip.")
    parser.add_argument("--seed", type=int, default=None, help="Optional deterministic seed.")
    return parser.parse_args()


def validate_args(args: argparse.Namespace) -> None:
    if not os.path.isfile(args.state_action_npz):
        raise FileNotFoundError(f"State-action NPZ not found: {args.state_action_npz}")
    if not os.path.isfile(args.policy_path):
        raise FileNotFoundError(f"Policy ONNX not found: {args.policy_path}")
    if args.urdf_file is not None and not os.path.isfile(args.urdf_file):
        if args.xml_file is None:
            raise FileNotFoundError(f"URDF not found: {args.urdf_file}")
        args.urdf_file = None
    if args.xml_file is not None and not os.path.isfile(args.xml_file):
        raise FileNotFoundError(f"XML not found: {args.xml_file}")
    if args.num_envs < 1:
        raise ValueError(f"--num_envs must be >= 1, got {args.num_envs}.")


def main() -> int:
    args = parse_args()
    validate_args(args)
    if args.seed is not None:
        apply_global_random_seeds(args.seed)

    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    output_npz = args.output_replay_npz
    if output_npz is None:
        stem = os.path.splitext(os.path.basename(args.state_action_npz))[0]
        output_npz = os.path.join(
            os.path.dirname(args.state_action_npz),
            f"{stem}_genesis_base_replay_{timestamp}.npz",
        )
    report_csv = args.report_csv
    if report_csv is None:
        report_csv = os.path.splitext(output_npz)[0] + "_metrics.csv"

    runner = build_runner_from_paths(
        state_action_npz=os.path.abspath(args.state_action_npz),
        policy_path=os.path.abspath(args.policy_path),
        urdf_file=args.urdf_file,
        xml_file=args.xml_file,
        backend=args.backend,
        sim_dt=args.sim_dt,
        control_dt=args.control_dt,
        trajectory_indices_text=args.trajectory_indices,
        seed=args.seed,
        torque_limit=args.torque_limit,
        num_envs=args.num_envs,
    )
    summary = runner.run(output_npz=os.path.abspath(output_npz), report_csv=os.path.abspath(report_csv))

    if args.report_json is not None:
        os.makedirs(os.path.dirname(args.report_json) or ".", exist_ok=True)
        with open(args.report_json, "w", encoding="utf-8") as handle:
            json.dump(summary, handle, indent=2, sort_keys=True)
        print(f"[INFO] Saved summary JSON to {args.report_json}")

    print(
        "[INFO] Open-loop replay complete: "
        f"trajectories={summary['num_trajectories']} "
        f"mean_rmse_abs={summary['mean_rmse_abs']:.6f} "
        f"mean_rmse_rel={summary['mean_rmse_rel']:.6f}"
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
