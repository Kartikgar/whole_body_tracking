from __future__ import annotations

import torch
from typing import TYPE_CHECKING

from isaaclab.managers import SceneEntityCfg
from isaaclab.sensors import ContactSensor, RayCaster
from isaaclab.utils.math import matrix_from_quat, quat_rotate_inverse, subtract_frame_transforms

from whole_body_tracking.tasks.tracking.mdp.commands import MotionCommand

if TYPE_CHECKING:
    from isaaclab.envs import ManagerBasedEnv


def robot_anchor_ori_w(env: ManagerBasedEnv, command_name: str) -> torch.Tensor:
    command: MotionCommand = env.command_manager.get_term(command_name)
    mat = matrix_from_quat(command.robot_anchor_quat_w)
    return mat[..., :2].reshape(mat.shape[0], -1)


def robot_anchor_lin_vel_w(env: ManagerBasedEnv, command_name: str) -> torch.Tensor:
    command: MotionCommand = env.command_manager.get_term(command_name)

    return command.robot_anchor_vel_w[:, :3].view(env.num_envs, -1)


def robot_anchor_ang_vel_w(env: ManagerBasedEnv, command_name: str) -> torch.Tensor:
    command: MotionCommand = env.command_manager.get_term(command_name)

    return command.robot_anchor_vel_w[:, 3:6].view(env.num_envs, -1)


def robot_body_pos_b(env: ManagerBasedEnv, command_name: str) -> torch.Tensor:
    command: MotionCommand = env.command_manager.get_term(command_name)

    num_bodies = len(command.cfg.body_names)
    pos_b, _ = subtract_frame_transforms(
        command.robot_anchor_pos_w[:, None, :].repeat(1, num_bodies, 1),
        command.robot_anchor_quat_w[:, None, :].repeat(1, num_bodies, 1),
        command.robot_body_pos_w,
        command.robot_body_quat_w,
    )

    return pos_b.view(env.num_envs, -1)


def robot_body_ori_b(env: ManagerBasedEnv, command_name: str) -> torch.Tensor:
    command: MotionCommand = env.command_manager.get_term(command_name)

    num_bodies = len(command.cfg.body_names)
    _, ori_b = subtract_frame_transforms(
        command.robot_anchor_pos_w[:, None, :].repeat(1, num_bodies, 1),
        command.robot_anchor_quat_w[:, None, :].repeat(1, num_bodies, 1),
        command.robot_body_pos_w,
        command.robot_body_quat_w,
    )
    mat = matrix_from_quat(ori_b)
    return mat[..., :2].reshape(mat.shape[0], -1)


def motion_anchor_pos_b(env: ManagerBasedEnv, command_name: str) -> torch.Tensor:
    command: MotionCommand = env.command_manager.get_term(command_name)

    pos, _ = subtract_frame_transforms(
        command.robot_anchor_pos_w,
        command.robot_anchor_quat_w,
        command.anchor_pos_w,
        command.anchor_quat_w,
    )

    return pos.view(env.num_envs, -1)


def motion_anchor_ori_b(env: ManagerBasedEnv, command_name: str) -> torch.Tensor:
    command: MotionCommand = env.command_manager.get_term(command_name)
    # import ipdb;ipdb.set_trace()
    _, ori = subtract_frame_transforms(
        command.robot_anchor_pos_w,
        command.robot_anchor_quat_w,
        command.anchor_pos_w,
        command.anchor_quat_w,
    )
    mat = matrix_from_quat(ori)
    return mat[..., :2].reshape(mat.shape[0], -1)


def motion_joint_action(env: ManagerBasedEnv, command_name: str) -> torch.Tensor:
    command: MotionCommand = env.command_manager.get_term(command_name)
    if not command.has_joint_action:
        raise RuntimeError(
            "Observation term `motion_joint_action` requires motion files with an `action` or `actions` array."
        )
    return command.joint_action.view(env.num_envs, -1)


def external_delta_action(env: ManagerBasedEnv, action_buffer_name: str = "delta_external_actions") -> torch.Tensor:
    action = getattr(env, action_buffer_name, None)
    if action is None:
        return torch.zeros(env.num_envs, env.action_manager.total_action_dim, device=env.device)
    return action.view(env.num_envs, -1)


def current_action(env: ManagerBasedEnv, action_buffer_name: str = "delta_base_actions") -> torch.Tensor:
    """Current-step policy action injected by the runner before delta-policy inference."""
    action = getattr(env, action_buffer_name, None)
    if action is None:
        return torch.zeros(env.num_envs, env.action_manager.total_action_dim, device=env.device)
    return action.view(env.num_envs, -1)


def feet_contact_force(env: ManagerBasedEnv, sensor_cfg: SceneEntityCfg) -> torch.Tensor:
    """Latest world-frame contact force vector on selected feet, flattened as [B, 3 * num_feet]."""
    contact_sensor: ContactSensor = env.scene.sensors[sensor_cfg.name]
    # net_forces_w_history: [num_envs, history, num_bodies, 3]
    net_forces_w_history = contact_sensor.data.net_forces_w_history[:, :, sensor_cfg.body_ids, :]
    # Use the latest sample from sensor history for observation.
    feet_forces_w = net_forces_w_history[:, -1, :, :]
    return feet_forces_w.reshape(env.num_envs, -1)


def terrain_scan_points_b(
    env: ManagerBasedEnv,
    sensor_cfg: SceneEntityCfg,
    grid_shape: tuple[int, int],
    no_hit_value: float = 10.0,
    flatten: bool = False,
) -> torch.Tensor:
    """Ray-cast hit points in the sensor frame as a (HxWx3) scan matrix.

    Args:
        env: The environment.
        sensor_cfg: Scene entity config for the ray-caster sensor.
        grid_shape: Expected grid shape as (num_x, num_y).
        no_hit_value: Fallback value for rays with no valid hit.
        flatten: If True, return a flattened vector [B, H*W*3].

    Returns:
        Tensor with shape [B, H, W, 3] if ``flatten=False`` else [B, H*W*3].
    """
    sensor: RayCaster = env.scene.sensors[sensor_cfg.name]
    ray_hits_w = sensor.data.ray_hits_w

    num_x, num_y = int(grid_shape[0]), int(grid_shape[1])
    expected_num_rays = num_x * num_y
    if ray_hits_w.shape[1] != expected_num_rays:
        raise RuntimeError(
            "Scan grid shape mismatch with ray-caster pattern: "
            f"expected {expected_num_rays} rays for grid_shape={grid_shape}, got {ray_hits_w.shape[1]} rays."
        )

    sensor_pos_w = sensor.data.pos_w.unsqueeze(1)
    relative_hits_w = ray_hits_w - sensor_pos_w

    # Some rays may miss the terrain and produce non-finite values.
    valid_mask = torch.isfinite(relative_hits_w).all(dim=-1, keepdim=True)
    safe_relative_hits_w = torch.where(valid_mask, relative_hits_w, torch.zeros_like(relative_hits_w))

    num_rays = ray_hits_w.shape[1]
    sensor_quat_w = sensor.data.quat_w.unsqueeze(1).expand(-1, num_rays, -1)
    relative_hits_b = quat_rotate_inverse(
        sensor_quat_w.reshape(-1, 4), safe_relative_hits_w.reshape(-1, 3)
    ).reshape(env.num_envs, num_rays, 3)

    if torch.any(~valid_mask):
        fallback_hits_b = torch.full_like(relative_hits_b, fill_value=no_hit_value)
        fallback_hits_b[..., 2] = 0.0
        relative_hits_b = torch.where(valid_mask.expand_as(relative_hits_b), relative_hits_b, fallback_hits_b)

    scan_matrix = relative_hits_b.view(env.num_envs, num_x, num_y, 3)
    if flatten:
        return scan_matrix.reshape(env.num_envs, -1)
    return scan_matrix


def terrain_scan_points_b_flat(
    env: ManagerBasedEnv,
    sensor_cfg: SceneEntityCfg,
    grid_shape: tuple[int, int],
    no_hit_value: float = 10.0,
) -> torch.Tensor:
    """Flattened version of :func:`terrain_scan_points_b` for MLP-style policies."""
    return terrain_scan_points_b(
        env=env,
        sensor_cfg=sensor_cfg,
        grid_shape=grid_shape,
        no_hit_value=no_hit_value,
        flatten=True,
    )


def goal_position_b(
    env: ManagerBasedEnv,
    goal_offset: tuple[float, float, float],
    asset_cfg: SceneEntityCfg = SceneEntityCfg("robot"),
) -> torch.Tensor:
    """Goal position expressed in the robot base frame.

    The goal is specified as an offset from each environment origin.
    """
    asset = env.scene[asset_cfg.name]
    goal_offset_t = torch.tensor(goal_offset, device=env.device, dtype=asset.data.root_pos_w.dtype).unsqueeze(0)
    goal_pos_w = env.scene.env_origins + goal_offset_t
    goal_vec_w = goal_pos_w - asset.data.root_pos_w
    return quat_rotate_inverse(asset.data.root_quat_w, goal_vec_w)
