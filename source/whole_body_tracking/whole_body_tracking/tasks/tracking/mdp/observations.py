from __future__ import annotations

import torch
from typing import TYPE_CHECKING

from isaaclab.managers import SceneEntityCfg
from isaaclab.sensors import ContactSensor
from isaaclab.utils.math import matrix_from_quat, subtract_frame_transforms

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
