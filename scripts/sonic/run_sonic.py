#!/usr/bin/env python3
"""Run the pinned default SONIC actor on a repository G1 NPZ in Isaac Lab."""
import argparse
import json
import os
import threading
import traceback
from datetime import datetime
from dataclasses import MISSING
from pathlib import Path

from isaaclab.app import AppLauncher

parser = argparse.ArgumentParser(description=__doc__)
parser.add_argument("--sonic_model_dir", required=True)
parser.add_argument("--motion_file", required=True)
parser.add_argument("--trajectory_index", type=int, default=0)
parser.add_argument("--start_timestep", type=int, default=0)
parser.add_argument("--duration_s", type=float, default=10.0, help="Requested rollout duration; must complete without an early termination.")
parser.add_argument("--max_steps", type=int)
parser.add_argument("--output_json")
parser.add_argument("--output_npz", help="Save valid-length state and action rollouts.")
parser.add_argument("--num_envs", type=int, default=1)
parser.add_argument("--seed", type=int, default=0)
parser.add_argument("--spawn_joint_noise_rad", type=float, default=0.0)
parser.add_argument("--full_motion", action="store_true", help="Run all reference frames from the start timestep.")
parser.add_argument("--video", action="store_true", default=False, help="Record a video of the rollout.")
parser.add_argument(
    "--video_length",
    type=int,
    default=None,
    help="Recorded control steps. Default is the full rollout.",
)
parser.add_argument("--video_folder", type=str, default=None, help="Directory for the recorded mp4.")
parser.add_argument(
    "--g1_urdf",
    choices=("sonic", "beyondmimic"),
    default="sonic",
    help="G1 URDF to spawn. sonic is the GEAR-SONIC asset; beyondmimic is the repository Unitree description.",
)
AppLauncher.add_app_launcher_args(parser)
args = parser.parse_args()
if args.video:
    args.enable_cameras = True
launcher = AppLauncher(args)
simulation_app = launcher.app

import gymnasium as gym
import numpy as np
import torch
import isaaclab.sim as sim_utils
from isaaclab.actuators import ImplicitActuatorCfg
from isaaclab.assets import Articulation, ArticulationCfg, AssetBaseCfg
from isaaclab.markers import VisualizationMarkers, VisualizationMarkersCfg
from isaaclab.scene import InteractiveScene, InteractiveSceneCfg
from isaaclab.sensors import ContactSensorCfg
from isaaclab.sim import SimulationContext
from isaaclab.utils import configclass

from whole_body_tracking.assets import ASSET_DIR
from whole_body_tracking.sonic.motion import SonicMotion
from whole_body_tracking.sonic.math_utils import quat_rotate_inverse_wxyz
from whole_body_tracking.sonic.policy import SonicPolicy
from whole_body_tracking.sonic.spec import CONTROL_DT, robot_path
from whole_body_tracking.safety_metrics import SAFETY_DEFINITIONS, SafetyMetricsAccumulator, g1_actuator_velocity_limits

KEY_REFERENCE_BODIES = (
    "pelvis",
    "torso_link",
    "left_ankle_roll_link",
    "right_ankle_roll_link",
    "left_wrist_yaw_link",
    "right_wrist_yaw_link",
)


@configclass
class SonicSceneCfg(InteractiveSceneCfg):
    ground = AssetBaseCfg(prim_path="/World/Ground", spawn=sim_utils.GroundPlaneCfg())
    light = AssetBaseCfg(
        prim_path="/World/Light",
        spawn=sim_utils.DomeLightCfg(intensity=750.0),
    )
    robot: ArticulationCfg = MISSING  # type: ignore
    contact_forces = ContactSensorCfg(
        prim_path="{ENV_REGEX_NS}/Robot/.*", history_length=4,
        track_air_time=True, force_threshold=10.0, debug_vis=False,
    )


def to_numpy(tensor):
    return tensor.detach().cpu().numpy().astype(np.float32)


def g1_urdf_path() -> str:
    """Resolve the selected G1 description. Actuator gains stay on the SONIC contract either way."""
    if args.g1_urdf == "beyondmimic":
        path = Path(ASSET_DIR) / "unitree_description/urdf/g1/main.urdf"
        if not path.is_file():
            raise FileNotFoundError(f"BeyondMimic G1 URDF missing: {path}")
        return str(path.resolve())
    return robot_path(args.sonic_model_dir)


def update_reference_markers(visualizer, reference, body_names, device):
    """Place spheres at the reference poses currently used for tracking."""

    if visualizer is None:
        return
    indices = [body_names.index(name) for name in KEY_REFERENCE_BODIES]
    visualizer.visualize(
        torch.as_tensor(reference["body_pos_w"][:, indices].reshape(-1, 3), device=device),
        torch.as_tensor(reference["body_quat_w"][:, indices].reshape(-1, 4), device=device),
    )


class SonicIsaacEnv(gym.Env):
    """One closed-loop SONIC step per Gymnasium step, so RecordVideo can capture it."""

    metadata = {"render_modes": [None, "rgb_array"], "render_fps": int(round(1.0 / CONTROL_DT))}

    def __init__(self, sim, scene, robot, policy, motion, joint_ids, body_ids, reference_visualizer, steps, decimation, sim_dt):
        super().__init__()
        self.sim = sim
        self.scene = scene
        self.robot = robot
        self.policy = policy
        self.motion = motion
        self.joint_ids = joint_ids
        self.body_ids = body_ids
        self.reference_visualizer = reference_visualizer
        self.steps = steps
        self.decimation = decimation
        self.sim_dt = sim_dt
        self.render_mode = "rgb_array" if args.video else None
        self.observation_space = gym.spaces.Box(low=-np.inf, high=np.inf, shape=(1,), dtype=np.float32)
        self.action_space = gym.spaces.Box(low=-1.0, high=1.0, shape=(1,), dtype=np.float32)
        self.offset = -1
        self.num_envs = args.num_envs
        self.origins = to_numpy(scene.env_origins)
        self.termination_reasons = [None] * self.num_envs
        self.valid_lengths = np.zeros(self.num_envs, dtype=np.int32)
        metric_template = {
            "joint_position_error": [],
            "root_position_error": [],
            "root_orientation_error": [],
            "feet_wrist_vertical_error": [],
            "root_vertical_error": [],
            "body_position_error": [],
            "body_velocity_error": [],
            "body_acceleration_error": [],
        }
        self.metrics = [{key: [] for key in metric_template} for _ in range(self.num_envs)]
        self.recordings = {key: [] for key in ("joint_pos", "joint_vel", "body_pos_w", "body_quat_w", "body_lin_vel_w", "body_ang_vel_w", "actions")}
        self.initial_state = self._state()
        self.previous_body_velocity = None
        self.previous_reference_velocity = None
        self.endpoint_ids = [policy.meta.body_names.index(name) for name in (
            "left_ankle_roll_link", "right_ankle_roll_link", "left_wrist_yaw_link", "right_wrist_yaw_link"
        )]
        self.contact_sensor = scene["contact_forces"]
        self.safety = SafetyMetricsAccumulator(
            num_envs=self.num_envs, dt=CONTROL_DT,
            joint_names=list(policy.meta.joint_names), body_names=list(policy.meta.body_names),
            contact_body_names=list(self.contact_sensor.body_names),
            joint_pos_limits=to_numpy(robot.data.joint_pos_limits[0, joint_ids]),
            soft_joint_pos_limits=to_numpy(robot.data.soft_joint_pos_limits[0, joint_ids]),
            joint_vel_limits=g1_actuator_velocity_limits(list(policy.meta.joint_names)),
            joint_effort_limits=to_numpy(robot.data.joint_effort_limits[0, joint_ids]),
        )
        self._rgb_annotator = None

    def reset(self, *, seed=None, options=None):
        super().reset(seed=seed)
        return np.zeros(1, dtype=np.float32), {}

    def _state(self):
        return {
            "joint_pos": to_numpy(self.robot.data.joint_pos[:, self.joint_ids]),
            "joint_vel": to_numpy(self.robot.data.joint_vel[:, self.joint_ids]),
            "body_pos_w": to_numpy(self.robot.data.body_pos_w[:, self.body_ids]),
            "body_quat_w": to_numpy(self.robot.data.body_quat_w[:, self.body_ids]),
            "body_lin_vel_w": to_numpy(self.robot.data.body_lin_vel_w[:, self.body_ids]),
            "body_ang_vel_w": to_numpy(self.robot.data.body_ang_vel_w[:, self.body_ids]),
        }

    def _reference(self, reference):
        reference = {key: value.copy() for key, value in reference.items()}
        reference["body_pos_w"] += self.origins[:, None, :]
        return reference

    def step(self, action):
        del action
        self.offset += 1
        offset = self.offset
        step = args.start_timestep + offset
        state = self._state()
        state["root_quat_w"] = state["body_quat_w"][:, 0]
        encoder_obs = self.policy.observations.build(state, self._reference(self.motion.reference_at(step, self.num_envs)), 0, step)
        output = self.policy.run(encoder_obs, step)
        policy_action = output["actions"]
        target = self.policy.meta.joint_targets(policy_action)
        self.robot.set_joint_position_target(torch.as_tensor(target, device=self.robot.device), joint_ids=self.joint_ids)
        for _ in range(self.decimation):
            self.scene.write_data_to_sim()
            self.sim.step()
            self.scene.update(self.sim_dt)
        self.policy.observations.update_last_action(policy_action)
        reference = self._reference({key: value for key, value in output.items() if key != "actions"})
        update_reference_markers(self.reference_visualizer, reference, self.policy.meta.body_names, self.robot.device)
        next_state = self._state()
        contact_history = self.contact_sensor.data.net_forces_w_history
        if contact_history is None:
            raise RuntimeError("Contact force history is unavailable for safety metrics")
        self.safety.update(
            active=np.asarray([reason is None for reason in self.termination_reasons], dtype=bool),
            joint_pos=next_state["joint_pos"], joint_vel=next_state["joint_vel"],
            applied_torque=to_numpy(self.robot.data.applied_torque[:, self.joint_ids]),
            body_lin_vel_w=next_state["body_lin_vel_w"],
            contact_forces_w_history=to_numpy(contact_history),
        )
        if args.output_npz:
            for key in self.recordings:
                self.recordings[key].append((policy_action if key == "actions" else next_state[key]).copy())
        for env_id in range(self.num_envs):
            if self.termination_reasons[env_id] is not None:
                continue
            q = next_state["joint_pos"][env_id]
            body_pos = next_state["body_pos_w"][env_id]
            body_quat = next_state["body_quat_w"][env_id]
            pos, quat = body_pos[0], body_quat[0]
            metric = self.metrics[env_id]
            metric["joint_position_error"].append(float(np.mean(np.abs(q - reference["joint_pos"][env_id]))))
            metric["root_position_error"].append(float(np.linalg.norm(pos - reference["body_pos_w"][env_id, 0])))
            metric["root_orientation_error"].append(float(2 * np.arccos(np.clip(abs(np.dot(quat, reference["body_quat_w"][env_id, 0])), 0, 1))))
            root_z_error = float(abs(pos[2] - reference["body_pos_w"][env_id, 0, 2]))
            metric["root_vertical_error"].append(root_z_error)
            relative_z = body_pos[:, 2] - pos[2]
            reference_relative_z = reference["body_pos_w"][env_id, :, 2] - reference["body_pos_w"][env_id, 0, 2]
            endpoint_delta = np.abs(relative_z[self.endpoint_ids] - reference_relative_z[self.endpoint_ids])
            endpoint_z_error = float(np.max(endpoint_delta))
            foot_z_error = float(np.max(endpoint_delta[:2]))
            metric["feet_wrist_vertical_error"].append(endpoint_z_error)
            metric["body_position_error"].append(float(np.mean(np.linalg.norm(body_pos - reference["body_pos_w"][env_id], axis=-1))))
            velocity = next_state["body_lin_vel_w"][env_id]
            reference_velocity = reference["body_lin_vel_w"][env_id]
            metric["body_velocity_error"].append(float(np.mean(np.linalg.norm(velocity - reference_velocity, axis=-1))))
            if self.previous_body_velocity is not None:
                acceleration_error = (velocity - self.previous_body_velocity[env_id] - reference_velocity + self.previous_reference_velocity[env_id]) / CONTROL_DT
                metric["body_acceleration_error"].append(float(np.mean(np.linalg.norm(acceleration_error, axis=-1))))
            ref_gravity_z = quat_rotate_inverse_wxyz(reference["body_quat_w"][env_id, 0], np.array([0., 0., -1.], dtype=np.float32))[2]
            robot_gravity_z = quat_rotate_inverse_wxyz(quat, np.array([0., 0., -1.], dtype=np.float32))[2]
            if root_z_error > 0.25:
                self.termination_reasons[env_id] = "anchor_vertical_error"
            elif abs(ref_gravity_z - robot_gravity_z) > 0.8:
                self.termination_reasons[env_id] = "anchor_orientation_error"
            elif foot_z_error > 0.40:
                self.termination_reasons[env_id] = "end_effector_vertical_error"
            self.valid_lengths[env_id] += 1
        self.previous_body_velocity = next_state["body_lin_vel_w"].copy()
        self.previous_reference_velocity = reference["body_lin_vel_w"].copy()
        if offset % 50 == 49:
            print(f"[SONIC] {offset + 1}/{self.steps} steps ({(offset + 1) * CONTROL_DT:.1f}s), "
                  f"active={sum(r is None for r in self.termination_reasons)}/{self.num_envs}", flush=True)
        pos = next_state["body_pos_w"][0, 0]
        eye = pos + np.array([2.0, 2.0, 0.8])
        target = pos + np.array([0.0, 0.0, 0.1])
        self.sim.set_camera_view(eye, target)
        return np.zeros(1, dtype=np.float32), 0.0, all(r is not None for r in self.termination_reasons), False, {}

    def render(self):
        if self.render_mode != "rgb_array":
            return None
        if not self.sim.has_rtx_sensors():
            self.sim.render()
        if self._rgb_annotator is None:
            import omni.replicator.core as rep

            self._render_product = rep.create.render_product("/OmniverseKit_Persp", (1280, 720))
            self._rgb_annotator = rep.AnnotatorRegistry.get_annotator("rgb", device="cpu")
            self._rgb_annotator.attach([self._render_product])
        rgb_data = self._rgb_annotator.get_data()
        rgb_data = np.frombuffer(rgb_data, dtype=np.uint8).reshape(*rgb_data.shape)
        if rgb_data.size == 0:
            return np.zeros((720, 1280, 3), dtype=np.uint8)
        return np.ascontiguousarray(rgb_data[:, :, :3])


def run():
    if args.num_envs < 1 or args.spawn_joint_noise_rad < 0:
        raise ValueError("num_envs must be positive and spawn_joint_noise_rad must be nonnegative")
    motion = SonicMotion(args.motion_file, args.trajectory_index)
    policy = SonicPolicy(args.sonic_model_dir, motion, args.num_envs, args.device, seed=args.seed)
    if not 0 <= args.start_timestep < policy.reference_motion_length_steps:
        raise ValueError(f"start_timestep outside [0, {policy.reference_motion_length_steps})")
    if args.duration_s <= 0:
        raise ValueError("duration_s must be positive")
    steps = motion.length - args.start_timestep if args.full_motion else int(np.ceil(args.duration_s / CONTROL_DT))
    available_steps = policy.reference_motion_length_steps - args.start_timestep
    if available_steps < steps:
        # raise ValueError(f"Motion has only {available_steps * CONTROL_DT:.2f}s remaining; requested {args.duration_s:.2f}s")
        steps = available_steps
    if args.max_steps is not None:
        if args.max_steps <= 0:
            raise ValueError("max_steps must be positive")
        steps = min(steps, args.max_steps)
    urdf_path = g1_urdf_path()
    print(f"[SONIC] Starting {steps} control steps ({steps * CONTROL_DT:.2f}s requested={args.duration_s:.2f}s) on {args.device}", flush=True)
    print(f"[SONIC] G1 URDF ({args.g1_urdf}): {urdf_path}", flush=True)

    robot_cfg = ArticulationCfg(
        prim_path="{ENV_REGEX_NS}/Robot",
        spawn=sim_utils.UrdfFileCfg(
            asset_path=urdf_path, fix_base=False, replace_cylinders_with_capsules=True,
            activate_contact_sensors=True,
            rigid_props=sim_utils.RigidBodyPropertiesCfg(max_depenetration_velocity=1.0),
            articulation_props=sim_utils.ArticulationRootPropertiesCfg(
                enabled_self_collisions=True, solver_position_iteration_count=8, solver_velocity_iteration_count=4
            ),
            joint_drive=sim_utils.UrdfConverterCfg.JointDriveCfg(
                gains=sim_utils.UrdfConverterCfg.JointDriveCfg.PDGainsCfg(stiffness=0, damping=0)
            ),
        ),
        init_state=ArticulationCfg.InitialStateCfg(pos=(0, 0, 0.76)),
        # PhysX integrates these PD gains. Explicit torque with zero drive
        # stiffness does not hold the same reference on this asset.
        actuators={"sonic": ImplicitActuatorCfg(
            joint_names_expr=[".*"],
            effort_limit_sim=dict(zip(policy.meta.joint_names, policy.meta.effort_limits.tolist())),
            velocity_limit_sim=1000.0,
            stiffness=dict(zip(policy.meta.joint_names, policy.meta.joint_stiffness.tolist())),
            damping=dict(zip(policy.meta.joint_names, policy.meta.joint_damping.tolist())),
            armature=dict(zip(policy.meta.joint_names, policy.meta.armature.tolist())),
        )},
    )
    sim_dt = 0.005
    sim = SimulationContext(sim_utils.SimulationCfg(dt=sim_dt, device=args.device))
    scene_cfg = SonicSceneCfg(num_envs=args.num_envs, env_spacing=2.5)
    scene_cfg.robot = robot_cfg
    scene = InteractiveScene(scene_cfg)
    sim.reset()
    print("[SONIC] Isaac scene initialized; starting closed-loop rollout", flush=True)
    robot: Articulation = scene["robot"]
    joint_ids = robot.find_joints(policy.meta.joint_names, preserve_order=True)[0]
    body_ids = robot.find_bodies(policy.meta.body_names, preserve_order=True)[0]
    if len(joint_ids) != 29 or len(body_ids) != 30:
        raise RuntimeError("Downloaded SONIC G1 asset does not match the policy joint/body contract")
    stiffness = to_numpy(robot.data.default_joint_stiffness[:, joint_ids])[0]
    print(f"[SONIC] drive stiffness min/max {stiffness.min():.2f}/{stiffness.max():.2f}", flush=True)
    reference_visualizer = None
    if args.video or not getattr(args, "headless", False):
        reference_visualizer = VisualizationMarkers(
            VisualizationMarkersCfg(
                prim_path="/Visuals/SONIC/reference",
                markers={
                    "reference": sim_utils.SphereCfg(
                        radius=0.035,
                        visual_material=sim_utils.PreviewSurfaceCfg(diffuse_color=(1.0, 1.0, 0.0)),
                    )
                },
            )
        )

    reference0 = motion.reference_at(args.start_timestep, args.num_envs)
    root = robot.data.default_root_state.clone()
    root[:, :3] = torch.as_tensor(reference0["body_pos_w"][:, 0], device=robot.device) + scene.env_origins
    root[:, 3:7] = torch.as_tensor(reference0["body_quat_w"][:, 0], device=robot.device)
    root[:, 7:10] = torch.as_tensor(reference0["body_lin_vel_w"][:, 0], device=robot.device)
    root[:, 10:13] = torch.as_tensor(reference0["body_ang_vel_w"][:, 0], device=robot.device)
    robot.write_root_state_to_sim(root)
    joint_pos = robot.data.default_joint_pos.clone()
    joint_vel = robot.data.default_joint_vel.clone()
    rng = np.random.default_rng(args.seed)
    spawn_joint_pos = reference0["joint_pos"] + rng.uniform(-args.spawn_joint_noise_rad, args.spawn_joint_noise_rad, reference0["joint_pos"].shape).astype(np.float32)
    joint_pos[:, joint_ids] = torch.as_tensor(spawn_joint_pos, device=robot.device)
    joint_vel[:, joint_ids] = torch.as_tensor(reference0["joint_vel"], device=robot.device)
    robot.write_joint_state_to_sim(joint_pos, joint_vel)
    scene.write_data_to_sim()
    sim.forward()
    scene.update(sim_dt)
    policy.observations.reset()
    reference0["body_pos_w"] += to_numpy(scene.env_origins)[:, None, :]
    update_reference_markers(reference_visualizer, reference0, policy.meta.body_names, robot.device)
    root_pos = to_numpy(root[:, :3])[0]
    sim.set_camera_view(root_pos + np.array([2.0, 2.0, 0.8]), root_pos + np.array([0.0, 0.0, 0.3]))
    decimation = round(CONTROL_DT / sim_dt)
    env = SonicIsaacEnv(
        sim, scene, robot, policy, motion, joint_ids, body_ids, reference_visualizer, steps, decimation, sim_dt
    )
    video_length = steps if args.video_length is None else args.video_length
    if args.video and video_length <= 0:
        raise ValueError("video_length must be positive")
    video_writer = None
    video_path = None
    if args.video:
        import imageio.v2 as imageio

        motion_stem = Path(args.motion_file).stem
        video_folder = args.video_folder or str(
            Path("logs/sonic_eval/videos") / f"{datetime.now().strftime('%Y-%m-%d_%H-%M-%S')}_{motion_stem}"
        )
        Path(video_folder).mkdir(parents=True, exist_ok=True)
        video_path = str((Path(video_folder) / f"sonic_{motion_stem}_env0.mp4").resolve())
        video_writer = imageio.get_writer(video_path, fps=round(1 / CONTROL_DT), codec="libx264", quality=7)
        print(f"[SONIC] Recording env 0 video to {video_path} ({video_length} steps).", flush=True)
    termination_reason = None
    try:
        env.reset()
        for frame_index in range(steps):
            if not simulation_app.is_running():
                termination_reason = "isaac_sim_closed"
                break
            _, _, terminated, _, _ = env.step(None)
            if video_writer is not None and frame_index < video_length:
                video_writer.append_data(env.render())
            if terminated:
                break
    finally:
        if video_writer is not None:
            video_writer.close()
        env.close()
    rollout = env.unwrapped
    if termination_reason:
        rollout.termination_reasons = [reason or termination_reason for reason in rollout.termination_reasons]
    rollouts = []
    for env_id, (metric, reason) in enumerate(zip(rollout.metrics, rollout.termination_reasons)):
        length = int(rollout.valid_lengths[env_id])
        safety_metrics, safety_details = rollout.safety.rollout(env_id)
        rollouts.append({"env_id": env_id, "steps": length, "completed_duration_s": length * CONTROL_DT,
                         "terminated": reason is not None, "termination_reason": reason or "reference_complete",
                         "metrics": {name: float(np.mean(values)) if values else float("nan") for name, values in metric.items()} | safety_metrics,
                         "safety": safety_details})
    if args.output_npz:
        output_npz = Path(args.output_npz)
        output_npz.parent.mkdir(parents=True, exist_ok=True)
        payload = {key: np.stack(values, axis=1) for key, values in rollout.recordings.items()}
        payload.update({f"initial_{key}": value for key, value in rollout.initial_state.items()})
        payload.update({"valid_lengths": rollout.valid_lengths, "fps": np.array([1 / CONTROL_DT], dtype=np.float32),
                        "joint_names": np.array(policy.meta.joint_names), "body_names": np.array(policy.meta.body_names)})
        np.savez_compressed(output_npz, **payload)

    result = {"policy_type": "sonic", "g1_urdf": args.g1_urdf, "robot_urdf": urdf_path,
              "motion_file": motion.path, "trajectory_index": motion.trajectory_index,
              "start_timestep": args.start_timestep, "steps": int(max(rollout.valid_lengths)),
              "requested_duration_s": steps * CONTROL_DT if args.full_motion else args.duration_s,
              "reference_duration_s": steps * CONTROL_DT,
              "completed_duration_s": min(r["completed_duration_s"] for r in rollouts),
              "terminated": any(r["terminated"] for r in rollouts),
              "termination_reason": next((r["termination_reason"] for r in rollouts if r["terminated"]), "requested_duration_complete"),
              "num_envs": args.num_envs, "seed": args.seed, "spawn_joint_noise_rad": args.spawn_joint_noise_rad,
              "safety_definitions": SAFETY_DEFINITIONS,
              "rollouts": rollouts,
              **({"recording_npz": str(Path(args.output_npz).resolve())} if args.output_npz else {}),
              **({"video_folder": video_folder, "video_mp4": video_path} if args.video else {}),
              **policy.artifact_metadata,
              **{name + "_mean": float(np.mean([r["metrics"][name] for r in rollouts])) for name in rollouts[0]["metrics"]}}
    output = Path(args.output_json) if args.output_json else Path("logs/sonic_eval") / f"{Path(args.motion_file).stem}_isaac.json"
    output.parent.mkdir(parents=True, exist_ok=True)
    output.write_text(json.dumps(result, indent=2, sort_keys=True) + "\n")
    print(json.dumps(result, indent=2, sort_keys=True), flush=True)
    print(f"Saved JSON: {output}", flush=True)
    if result["terminated"]:
        raise RuntimeError(f"SONIC Isaac rollout failed: {result['termination_reason']}")


def shutdown_isaac(app, exit_code: int) -> None:
    """Leave after the rollout. Kit close() loops while the stage is loading, and OmniHub never finishes."""
    def _close() -> None:
        try:
            app.close(wait_for_replicator=False)
        except Exception:
            traceback.print_exc()

    print("[SONIC] Episode finished; closing Isaac.", flush=True)
    thread = threading.Thread(target=_close, daemon=True)
    thread.start()
    thread.join(timeout=5.0)
    os._exit(exit_code)


if __name__ == "__main__":
    exit_code = 0
    try:
        run()
    except Exception:
        traceback.print_exc()
        exit_code = 1
    finally:
        shutdown_isaac(simulation_app, exit_code)
