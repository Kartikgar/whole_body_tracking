#!/usr/bin/env python3
"""Modular Genesis sim2sim evaluator for exported whole_body_tracking ONNX policies."""

from __future__ import annotations

import argparse
import csv
import json
import os
import random
from datetime import datetime

import numpy as np
import torch

from sim2sim_genesis.config import EvalConfig, OutputTargets
from sim2sim_genesis.constants import DEFAULT_G1_URDF, DEFAULT_NEXT_LAB_DATE, DEFAULT_REFERENCE_MARKER_RADIUS
from sim2sim_genesis.control import PdController
from sim2sim_genesis.domain_randomization import DomainRandomizer
from sim2sim_genesis.metrics import TrackingMetricsEvaluator
from sim2sim_genesis.observations import ObservationBuilder
from sim2sim_genesis.onnx_policy import OnnxMotionPolicy
from sim2sim_genesis.runner import Sim2SimRunner
from sim2sim_genesis.scene import GenesisSceneAdapter
from sim2sim_genesis.trajectory_io import TrajectoryRecorder


def apply_global_random_seeds(seed: int) -> None:
    """Align Python, NumPy, and PyTorch RNG streams for repeatable evaluation."""

    seed = int(seed)
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)
        torch.backends.cudnn.benchmark = False
        torch.backends.cudnn.deterministic = True


def append_timestamp_to_path(path: str, timestamp: str) -> str:
    """Append a timestamp to a filename while preserving its extension."""

    directory = os.path.dirname(path)
    filename = os.path.basename(path)
    stem, ext = os.path.splitext(filename)
    stamped = f"{stem}_{timestamp}{ext}"
    return os.path.join(directory, stamped)


def write_csv(path: str, row: dict[str, object]) -> None:
    """Write a one-row CSV summary."""

    os.makedirs(os.path.dirname(path) or ".", exist_ok=True)
    with open(path, "w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(row.keys()))
        writer.writeheader()
        writer.writerow(row)


def write_json(path: str, row: dict[str, object]) -> None:
    """Write a JSON summary artifact."""

    os.makedirs(os.path.dirname(path) or ".", exist_ok=True)
    with open(path, "w", encoding="utf-8") as handle:
        json.dump(row, handle, indent=2, sort_keys=True)


def parse_args() -> argparse.Namespace:
    """Parse CLI arguments for the modular Genesis sim2sim evaluator."""

    parser = argparse.ArgumentParser(description="Modular Genesis sim2sim evaluator for whole_body_tracking ONNX policies.")
    parser.add_argument("--policy_path", type=str, required=True, help="Path to exported ONNX policy.")
    parser.add_argument("--dataset_yaml", type=str, default=None, help="Deprecated and ignored.")
    parser.add_argument(
        "--urdf_file",
        type=str,
        default=DEFAULT_G1_URDF,
        help="Path to the robot URDF used by Genesis.",
    )
    parser.add_argument("--xml_file", type=str, default=None, help="Optional MJCF XML fallback.")
    parser.add_argument(
        "--motion_file",
        type=str,
        default=None,
        help="Optional motion.npz path. Required if the policy uses `motion_joint_action`.",
    )
    parser.add_argument("--output_csv", type=str, default=None, help="Output CSV path (single-row summary).")
    parser.add_argument("--output_json", type=str, default=None, help="Optional JSON output path.")
    parser.add_argument("--output_motion_npz", type=str, default=None, help="Optional path for motion dataset export.")
    parser.add_argument("--record_motion", action="store_true", help="Record motion trajectories to NPZ.")
    parser.add_argument("--target_trajectories", type=int, default=1, help="Number of trajectories to collect.")
    parser.add_argument("--backend", type=str, default="gpu", choices=["cpu", "gpu"], help="Genesis backend.")
    parser.add_argument("--policy_device", type=str, default="cuda", help="ONNX Runtime device hint.")
    parser.add_argument("--device", dest="policy_device", help="Alias for --policy_device.")
    parser.add_argument("--num_envs", type=int, default=1, help="Number of parallel Genesis environments.")
    parser.add_argument("--sim_dt", type=float, default=0.001, help="Genesis simulation dt.")
    parser.add_argument("--control_dt", type=float, default=0.02, help="Policy control dt.")
    parser.add_argument("--start_timestep", type=int, default=0, help="Reference-motion start timestep.")
    parser.add_argument("--max_steps", type=int, default=None, help="Optional rollout horizon override.")
    parser.add_argument("--torque_limit", type=float, default=None, help="Optional symmetric torque clip.")
    parser.add_argument("--record_video", action="store_true", help="Record evaluation video.")
    parser.add_argument("--viewer", action="store_true", help="Show the Genesis viewer during evaluation.")
    parser.add_argument("--compute_metrics", action="store_true", help="Enable MPJPE/velocity/acceleration metrics.")
    parser.add_argument("--metric_num_envs", type=int, default=1, help="Number of environment trajectories to score.")
    parser.add_argument(
        "--add_noise",
        action=argparse.BooleanOptionalAction,
        default=True,
        help="Add training-style uniform noise to supported observation terms.",
    )
    parser.add_argument(
        "--domain_randomization",
        action=argparse.BooleanOptionalAction,
        default=True,
        help="Apply Isaac-matching domain randomization in Genesis.",
    )
    parser.add_argument("--seed", type=int, default=None, help="Optional global seed for evaluation.")
    parser.add_argument("--show_reference", action="store_true", help="Visualize reference body markers.")
    parser.add_argument(
        "--reference_marker_radius",
        type=float,
        default=DEFAULT_REFERENCE_MARKER_RADIUS,
        help="Radius of the optional white reference marker spheres.",
    )
    parser.add_argument("--video_name", type=str, default=None, help="Optional output video filename.")
    return parser.parse_args()


def validate_inputs(args: argparse.Namespace) -> None:
    """Validate required input paths before constructing the evaluator."""

    if not os.path.isfile(args.policy_path):
        raise FileNotFoundError(f"Policy not found: {args.policy_path}")
    if args.urdf_file is not None and not os.path.isfile(args.urdf_file):
        if args.xml_file is None:
            raise FileNotFoundError(
                f"URDF file not found: {args.urdf_file}\n"
                "Provide --urdf_file pointing to G1 URDF, or provide --xml_file fallback."
            )
        args.urdf_file = None
    if args.xml_file is not None and not os.path.isfile(args.xml_file):
        raise FileNotFoundError(f"XML file not found: {args.xml_file}")
    if args.motion_file is not None and not os.path.isfile(args.motion_file):
        raise FileNotFoundError(f"Motion file not found: {args.motion_file}")


def resolve_output_targets(args: argparse.Namespace, run_timestamp: str, policy_name: str) -> OutputTargets:
    """Resolve CSV, JSON, and NPZ output paths for the current invocation."""

    output_csv = None
    output_json = None
    output_motion_npz = None

    effective_compute_metrics = bool(args.compute_metrics or args.record_motion)
    if effective_compute_metrics:
        output_csv = args.output_csv or os.path.join(
            "logs", "sim2sim_eval", DEFAULT_NEXT_LAB_DATE, f"{policy_name}_genesis_metrics.csv"
        )
        output_json = args.output_json or os.path.join(
            "logs", "sim2sim_eval", DEFAULT_NEXT_LAB_DATE, f"{policy_name}_genesis_metrics.json"
        )
        output_csv = append_timestamp_to_path(output_csv, run_timestamp)
        output_json = append_timestamp_to_path(output_json, run_timestamp)

    if args.record_motion:
        output_motion_npz = args.output_motion_npz or os.path.join(
            "logs", "sim2sim_eval", DEFAULT_NEXT_LAB_DATE, f"{policy_name}_motion_dataset.npz"
        )
        output_motion_npz = append_timestamp_to_path(output_motion_npz, run_timestamp)

    return OutputTargets(
        output_csv=output_csv,
        output_json=output_json,
        output_motion_npz=output_motion_npz,
    )


def build_eval_config(args: argparse.Namespace, outputs: OutputTargets) -> EvalConfig:
    """Convert parsed CLI arguments into a normalized runtime config."""

    effective_compute_metrics = bool(args.compute_metrics or args.record_motion)
    effective_metric_num_envs = args.target_trajectories if args.record_motion else args.metric_num_envs
    if effective_compute_metrics and effective_metric_num_envs <= 0:
        raise ValueError(f"--metric_num_envs must be > 0. Got {effective_metric_num_envs}")

    return EvalConfig(
        policy_path=args.policy_path,
        urdf_file=args.urdf_file,
        xml_file=args.xml_file,
        motion_file=args.motion_file,
        backend=args.backend,
        policy_device=args.policy_device,
        num_envs=max(int(args.num_envs), 1),
        sim_dt=float(args.sim_dt),
        control_dt=float(args.control_dt),
        start_timestep=int(args.start_timestep),
        max_steps=args.max_steps,
        torque_limit=args.torque_limit,
        record_video=bool(args.record_video),
        viewer=bool(args.viewer),
        show_reference=bool(args.show_reference),
        reference_marker_radius=float(args.reference_marker_radius),
        add_noise=bool(args.add_noise),
        domain_randomization=bool(args.domain_randomization),
        compute_metrics=effective_compute_metrics,
        metric_num_envs=max(int(effective_metric_num_envs), 1),
        record_motion=bool(args.record_motion),
        target_trajectories=max(int(args.target_trajectories), 1),
        output_motion_npz=outputs.output_motion_npz,
        video_name=args.video_name,
        seed=args.seed,
    )


def build_runner(config: EvalConfig) -> Sim2SimRunner:
    """Construct the modular evaluator stack from the normalized runtime config."""

    if config.seed is not None:
        apply_global_random_seeds(config.seed)
    rng = np.random.default_rng(config.seed)

    policy = OnnxMotionPolicy(config.policy_path, config.policy_device, seed=config.seed)
    observation_builder = ObservationBuilder(
        policy.meta,
        num_envs=config.num_envs,
        add_noise=config.add_noise,
        rng=rng,
        motion_file=config.motion_file,
    )
    scene = GenesisSceneAdapter(
        backend=config.backend,
        sim_dt=config.sim_dt,
        num_envs=config.num_envs,
        viewer=config.viewer,
        record_video=config.record_video,
        show_reference=config.show_reference,
        reference_marker_radius=config.reference_marker_radius,
        urdf_file=config.urdf_file,
        xml_file=config.xml_file,
        meta_body_names=policy.meta.body_names,
        joint_names=policy.meta.joint_names,
        anchor_body_name=policy.meta.anchor_body_name,
        domain_randomization=config.domain_randomization,
        seed=config.seed,
    )
    controller = PdController(
        scene=scene,
        meta=policy.meta,
        observation_builder=observation_builder,
        control_dt=config.control_dt,
        sim_dt=config.sim_dt,
        torque_limit=config.torque_limit,
    )
    metrics_evaluator = TrackingMetricsEvaluator(
        body_names=policy.meta.body_names,
        anchor_idx=scene.anchor_idx,
        root_idx=scene.root_idx,
    )
    domain_randomizer = DomainRandomizer(
        scene=scene,
        observation_builder=observation_builder,
        rng=rng,
        control_dt=config.control_dt,
        enabled=config.domain_randomization,
    )
    domain_randomizer.setup()
    trajectory_recorder = None
    if config.record_motion:
        trajectory_recorder = TrajectoryRecorder(
            num_envs=config.num_envs,
            fps=1.0 / config.control_dt,
            target_trajectories=config.target_trajectories,
            output_path=config.output_motion_npz,
        )

    return Sim2SimRunner(
        config=config,
        policy=policy,
        scene=scene,
        observation_builder=observation_builder,
        controller=controller,
        metrics_evaluator=metrics_evaluator,
        domain_randomizer=domain_randomizer,
        trajectory_recorder=trajectory_recorder,
    )


def main() -> None:
    """Run the modular Genesis sim2sim evaluator from the command line."""

    args = parse_args()
    validate_inputs(args)

    policy_name = os.path.basename(args.policy_path).replace(".onnx", "")
    run_timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    outputs = resolve_output_targets(args, run_timestamp, policy_name)
    config = build_eval_config(args, outputs)
    runner = build_runner(config)
    result = runner.evaluate(run_timestamp)

    if outputs.output_csv is not None:
        write_csv(outputs.output_csv, result)
    if outputs.output_json is not None:
        write_json(outputs.output_json, result)

    print("\n=== Modular Genesis Sim2Sim Evaluation Summary ===")
    for key in sorted(result.keys()):
        print(f"{key}: {result[key]}")
    if outputs.output_csv is not None:
        print(f"\nSaved CSV: {outputs.output_csv}")
    if outputs.output_json is not None:
        print(f"Saved JSON: {outputs.output_json}")
    if "output_motion_npz" in result:
        print(f"Saved motion NPZ: {result['output_motion_npz']}")


if __name__ == "__main__":
    main()
