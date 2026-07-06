#!/usr/bin/env python3
"""MuJoCo sim2sim evaluator for exported whole_body_tracking ONNX policies."""

from __future__ import annotations

import argparse
import json
import os
import random
from datetime import datetime

import numpy as np

from sim2sim_genesis.metrics import TrackingMetricsEvaluator
from sim2sim_genesis.observations import ObservationBuilder
from sim2sim_genesis.onnx_policy import OnnxMotionPolicy
from sim2sim_mujoco.config import MujocoEvalConfig, append_timestamp_to_path, default_json_path
from sim2sim_mujoco.constants import (
    DEFAULT_G1_MJCF,
    DEFAULT_REFERENCE_MARKER_RADIUS,
    DEFAULT_VIDEO_HEIGHT,
    DEFAULT_VIDEO_WIDTH,
)
from sim2sim_mujoco.control import MujocoPdController
from sim2sim_mujoco.runner import MujocoSim2SimRunner
from sim2sim_mujoco.scene import MujocoSceneAdapter


def apply_global_random_seed(seed: int) -> None:
    """Set Python and NumPy RNGs for repeatable evaluation."""

    random.seed(int(seed))
    np.random.seed(int(seed))


def write_json(path: str, row: dict[str, object]) -> None:
    """Write one JSON summary artifact."""

    os.makedirs(os.path.dirname(path) or ".", exist_ok=True)
    with open(path, "w", encoding="utf-8") as handle:
        json.dump(row, handle, indent=2, sort_keys=True)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Single-env MuJoCo sim2sim evaluator for G1 ONNX policies.")
    parser.add_argument("--policy_path", type=str, required=True, help="Path to exported ONNX policy.")
    parser.add_argument("--xml_file", type=str, default=DEFAULT_G1_MJCF, help="Path to MuJoCo MJCF XML.")
    parser.add_argument("--policy_device", type=str, default="cpu", help="ONNX Runtime device hint.")
    parser.add_argument("--device", dest="policy_device", help="Alias for --policy_device.")
    parser.add_argument("--num_envs", type=int, default=1, help="Must be 1 for MuJoCo v0.")
    parser.add_argument("--sim_dt", type=float, default=0.001, help="MuJoCo simulation timestep.")
    parser.add_argument("--control_dt", type=float, default=0.02, help="Policy control timestep.")
    parser.add_argument("--start_timestep", type=int, default=0, help="Reference-motion start timestep.")
    parser.add_argument("--max_steps", type=int, default=None, help="Optional rollout horizon cap.")
    parser.add_argument("--torque_limit", type=float, default=None, help="Optional symmetric torque clip.")
    parser.add_argument("--compute_metrics", action="store_true", help="Compute MPJPE/velocity/acceleration metrics.")
    parser.add_argument("--record_video", action="store_true", help="Record an MP4 rollout video.")
    parser.add_argument("--show_reference", action="store_true", help="Draw reference body markers in recorded video.")
    parser.add_argument(
        "--reference_marker_radius",
        type=float,
        default=DEFAULT_REFERENCE_MARKER_RADIUS,
        help="Radius of white reference marker spheres.",
    )
    parser.add_argument("--video_width", type=int, default=DEFAULT_VIDEO_WIDTH, help="Recorded video width in pixels.")
    parser.add_argument("--video_height", type=int, default=DEFAULT_VIDEO_HEIGHT, help="Recorded video height in pixels.")
    parser.add_argument("--video_name", type=str, default=None, help="Optional output MP4 path.")
    parser.add_argument("--output_json", type=str, default=None, help="Optional output JSON path.")
    parser.add_argument("--seed", type=int, default=None, help="Optional deterministic seed.")
    return parser.parse_args()


def validate_args(args: argparse.Namespace) -> None:
    if args.num_envs != 1:
        raise ValueError(f"MuJoCo v0 supports exactly one env. Got --num_envs {args.num_envs}.")
    if not os.path.isfile(args.policy_path):
        raise FileNotFoundError(f"Policy not found: {args.policy_path}")
    if not os.path.isfile(args.xml_file):
        raise FileNotFoundError(f"MuJoCo XML not found: {args.xml_file}")
    if args.reference_marker_radius <= 0.0:
        raise ValueError("--reference_marker_radius must be positive.")
    if args.video_width <= 0 or args.video_height <= 0:
        raise ValueError("--video_width and --video_height must be positive.")


def build_config(args: argparse.Namespace, run_timestamp: str) -> MujocoEvalConfig:
    output_json = args.output_json
    if output_json is None:
        output_json = default_json_path(run_timestamp, args.policy_path)
    else:
        output_json = append_timestamp_to_path(output_json, run_timestamp)

    video_name = args.video_name
    if video_name is not None:
        video_name = append_timestamp_to_path(video_name, run_timestamp)

    return MujocoEvalConfig(
        policy_path=os.path.abspath(args.policy_path),
        xml_file=os.path.abspath(args.xml_file),
        policy_device=args.policy_device,
        sim_dt=float(args.sim_dt),
        control_dt=float(args.control_dt),
        start_timestep=int(args.start_timestep),
        max_steps=args.max_steps,
        torque_limit=args.torque_limit,
        compute_metrics=bool(args.compute_metrics),
        record_video=bool(args.record_video),
        show_reference=bool(args.show_reference),
        reference_marker_radius=float(args.reference_marker_radius),
        video_width=int(args.video_width),
        video_height=int(args.video_height),
        video_name=video_name,
        output_json=output_json,
        seed=args.seed,
    )


def build_runner(config: MujocoEvalConfig) -> MujocoSim2SimRunner:
    if config.seed is not None:
        apply_global_random_seed(config.seed)
    rng = np.random.default_rng(config.seed)
    policy = OnnxMotionPolicy(config.policy_path, config.policy_device, seed=config.seed)
    if "motion_joint_action" in policy.meta.observation_names:
        raise RuntimeError(
            "MuJoCo v0 supports deployed base/finetuned-base ONNX policies only. "
            "Policies requiring `motion_joint_action` are not supported yet."
        )

    observation_builder = ObservationBuilder(
        policy.meta,
        num_envs=1,
        add_noise=False,
        rng=rng,
        motion_file=None,
    )
    scene = MujocoSceneAdapter(
        xml_file=config.xml_file,
        sim_dt=config.sim_dt,
        meta_body_names=policy.meta.body_names,
        joint_names=policy.meta.joint_names,
        anchor_body_name=policy.meta.anchor_body_name,
        record_video=config.record_video,
        show_reference=config.show_reference,
        reference_marker_radius=config.reference_marker_radius,
        width=config.video_width,
        height=config.video_height,
    )
    controller = MujocoPdController(
        scene=scene,
        meta=policy.meta,
        control_dt=config.control_dt,
        sim_dt=config.sim_dt,
        torque_limit=config.torque_limit,
    )
    metrics_evaluator = TrackingMetricsEvaluator(
        body_names=policy.meta.body_names,
        anchor_idx=scene.anchor_idx,
        root_idx=scene.root_idx,
    )
    return MujocoSim2SimRunner(
        config=config,
        policy=policy,
        scene=scene,
        observation_builder=observation_builder,
        controller=controller,
        metrics_evaluator=metrics_evaluator,
    )


def main() -> None:
    args = parse_args()
    validate_args(args)
    run_timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    config = build_config(args, run_timestamp)
    runner = build_runner(config)
    try:
        result = runner.evaluate(run_timestamp)
    finally:
        runner.scene.close()
    write_json(config.output_json, result)

    print("\n=== MuJoCo Sim2Sim Evaluation Summary ===")
    for key in sorted(result.keys()):
        print(f"{key}: {result[key]}")
    print(f"\nSaved JSON: {config.output_json}")
    if "video_path" in result:
        print(f"Saved video: {result['video_path']}")


if __name__ == "__main__":
    main()
