"""Bounded, disk-backed full-motion evaluation used by play.py batch mode."""

from __future__ import annotations

import json
import tempfile
from pathlib import Path

import numpy as np
import torch
from isaaclab.utils.math import quat_apply, quat_error_magnitude, quat_inv, quat_mul, quat_rotate_inverse, yaw_quat

from whole_body_tracking.safety_metrics import SAFETY_DEFINITIONS, SafetyMetricsAccumulator, g1_actuator_velocity_limits
from utils import DEFAULT_STATE_ACTION_KEYS


def _numpy_state(state: dict[str, torch.Tensor]) -> dict[str, np.ndarray]:
    return {key: state[key].detach().to("cpu", dtype=torch.float32).numpy() for key in DEFAULT_STATE_ACTION_KEYS}


def _step_metrics(state: dict[str, torch.Tensor], reference: dict[str, torch.Tensor], anchor: int, root: int):
    body_count = reference["body_pos_w"].shape[1]
    robot_anchor_pos = state["body_pos_w"][:, anchor]
    ref_anchor_pos = reference["body_pos_w"][:, anchor]
    delta_pos = robot_anchor_pos[:, None, :].repeat(1, body_count, 1)
    delta_pos[..., 2] = ref_anchor_pos[:, None, 2]
    delta_ori = quat_mul(yaw_quat(state["body_quat_w"][:, root]), quat_inv(yaw_quat(reference["body_quat_w"][:, root])))
    delta_ori_batch = delta_ori[:, None, :].repeat(1, body_count, 1)
    body_pos_relative = delta_pos + quat_apply(
        delta_ori_batch, reference["body_pos_w"] - ref_anchor_pos[:, None, :]
    )
    body_quat_relative = quat_mul(delta_ori_batch, reference["body_quat_w"])
    errors = {
        "error_anchor_pos": torch.linalg.norm(ref_anchor_pos - robot_anchor_pos, dim=-1),
        "error_anchor_rot": quat_error_magnitude(reference["body_quat_w"][:, anchor], state["body_quat_w"][:, anchor]),
        "error_body_pos": torch.linalg.norm(body_pos_relative - state["body_pos_w"], dim=-1).mean(dim=-1),
        "error_body_rot": quat_error_magnitude(body_quat_relative, state["body_quat_w"]).mean(dim=-1),
        "error_joint_pos": torch.linalg.norm(reference["joint_pos"] - state["joint_pos"], dim=-1),
        "error_joint_vel": torch.linalg.norm(reference["joint_vel"] - state["joint_vel"], dim=-1),
    }
    body_lin_sq = torch.square(reference["body_lin_vel_w"] - state["body_lin_vel_w"]).sum(dim=-1).mean(dim=-1)
    body_ang_sq = torch.square(reference["body_ang_vel_w"] - state["body_ang_vel_w"]).sum(dim=-1).mean(dim=-1)
    errors["tracking_reward_total"] = (
        0.5 * torch.exp(-(errors["error_anchor_pos"] ** 2) / (0.3 ** 2))
        + 0.5 * torch.exp(-(errors["error_anchor_rot"] ** 2) / (0.4 ** 2))
        + torch.exp(-torch.square(body_pos_relative - state["body_pos_w"]).sum(dim=-1).mean(dim=-1) / (0.3 ** 2))
        + torch.exp(-torch.square(quat_error_magnitude(body_quat_relative, state["body_quat_w"])).mean(dim=-1) / (0.4 ** 2))
        + torch.exp(-body_lin_sq / (1.0 ** 2))
        + torch.exp(-body_ang_sq / (3.14 ** 2))
    )
    return errors


def _failure_masks(state, reference, anchor: int, endpoint_ids: list[int]):
    nonfinite = torch.zeros(state["joint_pos"].shape[0], dtype=torch.bool, device=state["joint_pos"].device)
    for value in state.values():
        nonfinite |= ~torch.isfinite(value.reshape(value.shape[0], -1)).all(dim=1)
    anchor_z = (state["body_pos_w"][:, anchor, 2] - reference["body_pos_w"][:, anchor, 2]).abs()
    gravity = torch.tensor([0.0, 0.0, -1.0], device=anchor_z.device).expand(state["body_quat_w"].shape[0], -1)
    ref_gravity_z = quat_rotate_inverse(reference["body_quat_w"][:, anchor], gravity)[:, 2]
    robot_gravity_z = quat_rotate_inverse(state["body_quat_w"][:, anchor], gravity)[:, 2]
    ori_error = (ref_gravity_z - robot_gravity_z).abs()
    endpoint_error = (
        state["body_pos_w"][:, endpoint_ids, 2] - reference["body_pos_w"][:, endpoint_ids, 2]
    ).abs().amax(dim=-1)
    return (nonfinite, anchor_z > 0.25, ori_error > 0.8, endpoint_error > 0.25)


def run_checkpoint_batch_eval(
    *, env, policy, simulation_app, checkpoint: str, motion_file: str, motion_frames: int,
    seed: int, spawn_joint_noise_rad: float, output_json: str, output_npz: str,
    video_folder: str | None, capture_reference, capture_state, bootstrap, metadata,
) -> None:
    base_env = env.unwrapped
    motion_term = base_env.command_manager.get_term("motion")
    if motion_term.motion.num_trajectories != 1 or int(motion_term.motion.trajectory_time_step_total[0]) != motion_frames:
        raise ValueError("Batch evaluation requires one full, unpadded motion trajectory")
    body_names = list(motion_term.cfg.body_names)
    anchor = body_names.index(motion_term.cfg.anchor_body_name)
    root = 0
    endpoint_ids = [body_names.index(name) for name in (
        "left_ankle_roll_link", "right_ankle_roll_link", "left_wrist_yaw_link", "right_wrist_yaw_link"
    )]
    fps = 1.0 / float(base_env.step_dt)
    obs, _ = env.reset()
    bootstrap(env)
    obs, _ = env.get_observations()
    if torch.any(motion_term.time_steps != 0):
        raise RuntimeError("Checkpoint batch expected every environment to start at reference frame zero")
    initial = _numpy_state(capture_state(base_env))
    num_envs = env.num_envs
    robot = base_env.scene["robot"]
    contact_sensor = base_env.scene["contact_forces"]
    safety = SafetyMetricsAccumulator(
        num_envs=num_envs, dt=float(base_env.step_dt),
        joint_names=list(robot.data.joint_names), body_names=body_names,
        contact_body_names=list(contact_sensor.body_names),
        joint_pos_limits=robot.data.joint_pos_limits[0].detach().cpu().numpy(),
        soft_joint_pos_limits=robot.data.soft_joint_pos_limits[0].detach().cpu().numpy(),
        joint_vel_limits=g1_actuator_velocity_limits(list(robot.data.joint_names)),
        joint_effort_limits=robot.data.joint_effort_limits[0].detach().cpu().numpy(),
    )
    valid_lengths = np.zeros(num_envs, dtype=np.int32)
    termination_reasons: list[str | None] = [None] * num_envs
    metric_sums: dict[str, np.ndarray] = {}
    metric_counts: dict[str, np.ndarray] = {}
    previous_positions = None
    previous_reference_positions = None
    previous2_positions = None
    previous2_reference_positions = None
    npz_path = Path(output_npz).resolve()
    json_path = Path(output_json).resolve()
    npz_path.parent.mkdir(parents=True, exist_ok=True)
    json_path.parent.mkdir(parents=True, exist_ok=True)
    video_path = None
    writer = None
    if video_folder:
        import imageio.v2 as imageio

        folder = Path(video_folder).resolve()
        folder.mkdir(parents=True, exist_ok=True)
        video_path = folder / f"g1_{Path(motion_file).stem}_env0.mp4"
        writer = imageio.get_writer(str(video_path), fps=round(fps), codec="libx264", quality=7)

    with tempfile.TemporaryDirectory(prefix="checkpoint_recording_", dir=npz_path.parent) as scratch:
        arrays = {}
        for key, value in initial.items():
            arrays[key] = np.lib.format.open_memmap(
                Path(scratch) / f"{key}.npy", mode="w+", dtype=np.float32,
                shape=(num_envs, motion_frames, *value.shape[1:]),
            )
        action_array = None
        steps_run = 0
        interrupted = False
        try:
            for step in range(motion_frames):
                if not simulation_app.is_running():
                    interrupted = True
                    break
                with torch.inference_mode():
                    reference = capture_reference(base_env)
                    actions = policy(obs)
                    if action_array is None:
                        action_array = np.lib.format.open_memmap(
                            Path(scratch) / "actions.npy", mode="w+", dtype=np.float32,
                            shape=(num_envs, motion_frames, actions.shape[1]),
                        )
                    obs, _, dones, _ = env.step(actions)
                    if torch.any(dones):
                        raise RuntimeError("Unexpected Isaac episode reset during bounded batch evaluation")
                    state = capture_state(base_env)
                    active = np.array([reason is None for reason in termination_reasons], dtype=bool)
                    state_np = _numpy_state(state)
                    for key, value in state_np.items():
                        arrays[key][active, step] = value[active]
                    action_np = actions.detach().to("cpu", dtype=torch.float32).numpy()
                    action_array[active, step] = action_np[active]
                    contact_history = contact_sensor.data.net_forces_w_history
                    if contact_history is None:
                        raise RuntimeError("Contact force history is unavailable for safety metrics")
                    safety.update(
                        active=active, joint_pos=state_np["joint_pos"], joint_vel=state_np["joint_vel"],
                        applied_torque=robot.data.applied_torque.detach().cpu().numpy(),
                        body_lin_vel_w=state_np["body_lin_vel_w"],
                        contact_forces_w_history=contact_history.detach().cpu().numpy(),
                    )
                    metrics = _step_metrics(state, reference, anchor, root)
                    for name, values in metrics.items():
                        if name not in metric_sums:
                            metric_sums[name] = np.zeros(num_envs, dtype=np.float64)
                            metric_counts[name] = np.zeros(num_envs, dtype=np.int32)
                        metric_sums[name][active] += values.detach().cpu().numpy()[active]
                        metric_counts[name][active] += 1
                    positions = state_np["body_pos_w"]
                    reference_positions = reference["body_pos_w"].detach().cpu().numpy()
                    position_metrics = {
                        "mpjpe": np.linalg.norm(positions - reference_positions, axis=-1).mean(axis=-1) * 1000.0,
                        "mpjpe_l": np.linalg.norm(
                            (positions - positions[:, [root]]) - (reference_positions - reference_positions[:, [root]]),
                            axis=-1,
                        ).mean(axis=-1) * 1000.0,
                    }
                    if previous_positions is not None:
                        position_metrics["vel_dist"] = np.linalg.norm(
                            (positions - previous_positions) - (reference_positions - previous_reference_positions),
                            axis=-1,
                        ).mean(axis=-1) * 1000.0
                    if previous2_positions is not None:
                        position_metrics["accel_dist"] = np.linalg.norm(
                            (positions - 2 * previous_positions + previous2_positions)
                            - (reference_positions - 2 * previous_reference_positions + previous2_reference_positions),
                            axis=-1,
                        ).mean(axis=-1) * 1000.0
                    for name, values in position_metrics.items():
                        if name not in metric_sums:
                            metric_sums[name] = np.zeros(num_envs, dtype=np.float64)
                            metric_counts[name] = np.zeros(num_envs, dtype=np.int32)
                        metric_sums[name][active] += values[active]
                        metric_counts[name][active] += 1
                    previous2_positions, previous_positions = previous_positions, positions.copy()
                    previous2_reference_positions, previous_reference_positions = (
                        previous_reference_positions, reference_positions.copy()
                    )
                    failure_masks = _failure_masks(state, reference, anchor, endpoint_ids)
                    masks = [mask.detach().cpu().numpy() for mask in failure_masks]
                    for env_id in np.flatnonzero(active):
                        valid_lengths[env_id] += 1
                        for failed, reason in zip(masks, (
                            "nonfinite_state", "anchor_vertical_error", "anchor_orientation_error",
                            "end_effector_vertical_error",
                        )):
                            if failed[env_id]:
                                termination_reasons[env_id] = reason
                                break
                steps_run += 1
                if writer is not None:
                    writer.append_data(base_env.render())
                if step % 50 == 49:
                    print(f"[Checkpoint batch] {step + 1}/{motion_frames} steps, active={sum(r is None for r in termination_reasons)}/{num_envs}", flush=True)
                if all(reason is not None for reason in termination_reasons):
                    break
        finally:
            if writer is not None:
                writer.close()

        if interrupted:
            termination_reasons = [reason or "isaac_sim_closed" for reason in termination_reasons]
        if action_array is None:
            raise RuntimeError("No simulation steps were recorded")
        # Preserve one row per environment and pad each failed rollout through the last simulated step.
        for env_id, length in enumerate(valid_lengths):
            if length == 0:
                continue
            for value in (*arrays.values(), action_array):
                value[env_id, length:steps_run] = value[env_id, length - 1]
        payload = {key: value[:, :steps_run] for key, value in arrays.items()}
        payload["actions"] = action_array[:, :steps_run]
        payload.update({f"initial_{key}": value for key, value in initial.items()})
        payload["valid_lengths"] = valid_lengths
        payload["fps"] = np.array([fps], dtype=np.float32)
        extra = metadata(base_env, has_delta_policy=False)
        for key in ("joint_names", "body_names", "default_joint_pos", "action_scale", "action_mode"):
            if key in extra:
                payload[key] = np.asarray(extra[key])
        np.savez_compressed(npz_path, **payload)

    rollouts = []
    for env_id, reason in enumerate(termination_reasons):
        length = int(valid_lengths[env_id])
        safety_metrics, safety_details = safety.rollout(env_id)
        rollouts.append({
            "env_id": env_id, "steps": length, "completed_duration_s": length / fps,
            "terminated": reason is not None, "termination_reason": reason or "reference_complete",
            "metrics": {
                name: float(values[env_id] / metric_counts[name][env_id])
                if metric_counts[name][env_id] else float("nan")
                for name, values in metric_sums.items()
            } | safety_metrics,
            "safety": safety_details,
        })
    result = {
        "policy_type": "g1_checkpoint", "metric_basis": "g1_14_body_source_tracking",
        "motion_file": str(Path(motion_file).resolve()), "checkpoint": str(Path(checkpoint).resolve()),
        "reference_duration_s": motion_frames / fps, "num_envs": num_envs,
        "seed": seed, "spawn_joint_noise_rad": spawn_joint_noise_rad,
        "steps_run": steps_run, "rollouts": rollouts,
        "terminated": any(item["terminated"] for item in rollouts),
        "termination_reason": next((item["termination_reason"] for item in rollouts if item["terminated"]), "reference_complete"),
        "recording_npz": str(npz_path), "video_mp4": str(video_path) if video_path else "",
        "safety_definitions": SAFETY_DEFINITIONS,
    }
    json_path.write_text(json.dumps(result, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    print(f"[Checkpoint batch] Saved {npz_path} and {json_path}", flush=True)
    if result["terminated"]:
        raise RuntimeError(f"Checkpoint batch rollout failed: {result['termination_reason']}")
