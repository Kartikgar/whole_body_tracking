from __future__ import annotations

import torch
from typing import TYPE_CHECKING, cast

from isaaclab.managers import SceneEntityCfg
from isaaclab.sensors import ContactSensor
from isaaclab.utils.math import quat_apply, quat_error_magnitude, quat_inv, quat_mul, yaw_quat

from whole_body_tracking.tasks.tracking.mdp.commands import MotionCommand

if TYPE_CHECKING:
    from isaaclab.envs import ManagerBasedRLEnv


def _get_body_indexes(command: MotionCommand, body_names: list[str] | None) -> list[int]:
    return [i for i, name in enumerate(command.cfg.body_names) if (body_names is None) or (name in body_names)]


def _get_motion_command(env: ManagerBasedRLEnv, command_name: str) -> MotionCommand:
    return cast(MotionCommand, env.command_manager.get_term(command_name))


def _offset_time_steps(command: MotionCommand, time_offset: int) -> torch.Tensor:
    return command.time_steps + int(time_offset)


def _motion_body_pos_w_at_offset(command: MotionCommand, time_offset: int) -> torch.Tensor:
    time_steps = _offset_time_steps(command, time_offset)
    return (
        command.motion.get_body_pos_w(command.trajectory_ids, time_steps)
        + command._env.scene.env_origins[:, None, :]
    )


def _motion_body_quat_w_at_offset(command: MotionCommand, time_offset: int) -> torch.Tensor:
    time_steps = _offset_time_steps(command, time_offset)
    return command.motion.get_body_quat_w(command.trajectory_ids, time_steps)


def _motion_body_lin_vel_w_at_offset(command: MotionCommand, time_offset: int) -> torch.Tensor:
    time_steps = _offset_time_steps(command, time_offset)
    return command.motion.get_body_lin_vel_w(command.trajectory_ids, time_steps)


def _motion_body_ang_vel_w_at_offset(command: MotionCommand, time_offset: int) -> torch.Tensor:
    time_steps = _offset_time_steps(command, time_offset)
    return command.motion.get_body_ang_vel_w(command.trajectory_ids, time_steps)


def _motion_anchor_pos_w_at_offset(command: MotionCommand, time_offset: int) -> torch.Tensor:
    body_pos_w = _motion_body_pos_w_at_offset(command, time_offset)
    return body_pos_w[:, command.motion_anchor_body_index]


def _motion_anchor_quat_w_at_offset(command: MotionCommand, time_offset: int) -> torch.Tensor:
    body_quat_w = _motion_body_quat_w_at_offset(command, time_offset)
    return body_quat_w[:, command.motion_anchor_body_index]


def _motion_relative_body_targets_at_offset(
    command: MotionCommand, time_offset: int
) -> tuple[torch.Tensor, torch.Tensor]:
    body_pos_w = _motion_body_pos_w_at_offset(command, time_offset)
    body_quat_w = _motion_body_quat_w_at_offset(command, time_offset)
    anchor_pos_w = body_pos_w[:, command.motion_anchor_body_index]
    heading_quat_w = body_quat_w[:, command.motion_heading_body_index]

    anchor_pos_w_repeat = anchor_pos_w[:, None, :].repeat(1, len(command.cfg.body_names), 1)
    heading_quat_w_repeat = heading_quat_w[:, None, :].repeat(1, len(command.cfg.body_names), 1)
    robot_anchor_pos_w_repeat = command.robot_anchor_pos_w[:, None, :].repeat(1, len(command.cfg.body_names), 1)
    robot_heading_quat_w_repeat = command.robot_heading_quat_w[:, None, :].repeat(1, len(command.cfg.body_names), 1)

    delta_pos_w = robot_anchor_pos_w_repeat.clone()
    delta_pos_w[..., 2] = anchor_pos_w_repeat[..., 2]
    delta_ori_w = quat_mul(yaw_quat(robot_heading_quat_w_repeat), quat_inv(yaw_quat(heading_quat_w_repeat)))
    body_quat_relative_w = quat_mul(delta_ori_w, body_quat_w)
    body_pos_relative_w = delta_pos_w + quat_apply(delta_ori_w, body_pos_w - anchor_pos_w_repeat)
    return body_pos_relative_w, body_quat_relative_w


def motion_global_anchor_position_error_exp(env: ManagerBasedRLEnv, command_name: str, std: float) -> torch.Tensor:
    command = _get_motion_command(env, command_name)
    error = torch.sum(torch.square(command.anchor_pos_w - command.robot_anchor_pos_w), dim=-1)
    return torch.exp(-error / std**2)


def motion_global_anchor_orientation_error_exp(env: ManagerBasedRLEnv, command_name: str, std: float) -> torch.Tensor:
    command = _get_motion_command(env, command_name)
    error = quat_error_magnitude(command.anchor_quat_w, command.robot_anchor_quat_w) ** 2
    return torch.exp(-error / std**2)


def motion_relative_body_position_error_exp(
    env: ManagerBasedRLEnv, command_name: str, std: float, body_names: list[str] | None = None
) -> torch.Tensor:
    command = _get_motion_command(env, command_name)
    body_indexes = _get_body_indexes(command, body_names)
    error = torch.sum(
        torch.square(command.body_pos_relative_w[:, body_indexes] - command.robot_body_pos_w[:, body_indexes]), dim=-1
    )
    return torch.exp(-error.mean(-1) / std**2)


def motion_relative_body_orientation_error_exp(
    env: ManagerBasedRLEnv, command_name: str, std: float, body_names: list[str] | None = None
) -> torch.Tensor:
    command = _get_motion_command(env, command_name)
    body_indexes = _get_body_indexes(command, body_names)
    error = (
        quat_error_magnitude(command.body_quat_relative_w[:, body_indexes], command.robot_body_quat_w[:, body_indexes])
        ** 2
    )
    return torch.exp(-error.mean(-1) / std**2)


def motion_global_body_position_error_exp(
    env: ManagerBasedRLEnv, command_name: str, std: float, body_names: list[str] | None = None
) -> torch.Tensor:
    command = _get_motion_command(env, command_name)
    body_indexes = _get_body_indexes(command, body_names)
    error = torch.sum(torch.square(command.body_pos_w[:, body_indexes] - command.robot_body_pos_w[:, body_indexes]), dim=-1)
    return torch.exp(-error.mean(-1) / std**2)


def motion_global_body_orientation_error_exp(
    env: ManagerBasedRLEnv, command_name: str, std: float, body_names: list[str] | None = None
) -> torch.Tensor:
    command = _get_motion_command(env, command_name)
    body_indexes = _get_body_indexes(command, body_names)
    error = quat_error_magnitude(command.body_quat_w[:, body_indexes], command.robot_body_quat_w[:, body_indexes]) ** 2
    return torch.exp(-error.mean(-1) / std**2)


def motion_global_body_linear_velocity_error_exp(
    env: ManagerBasedRLEnv, command_name: str, std: float, body_names: list[str] | None = None
) -> torch.Tensor:
    command = _get_motion_command(env, command_name)
    body_indexes = _get_body_indexes(command, body_names)
    error = torch.sum(
        torch.square(command.body_lin_vel_w[:, body_indexes] - command.robot_body_lin_vel_w[:, body_indexes]), dim=-1
    )
    return torch.exp(-error.mean(-1) / std**2)


def motion_global_body_angular_velocity_error_exp(
    env: ManagerBasedRLEnv, command_name: str, std: float, body_names: list[str] | None = None
) -> torch.Tensor:
    command = _get_motion_command(env, command_name)
    body_indexes = _get_body_indexes(command, body_names)
    error = torch.sum(
        torch.square(command.body_ang_vel_w[:, body_indexes] - command.robot_body_ang_vel_w[:, body_indexes]), dim=-1
    )
    return torch.exp(-error.mean(-1) / std**2)


def motion_global_anchor_position_error_exp_at_offset(
    env: ManagerBasedRLEnv, command_name: str, std: float, time_offset: int
) -> torch.Tensor:
    command = _get_motion_command(env, command_name)
    anchor_pos_w = _motion_anchor_pos_w_at_offset(command, time_offset)
    error = torch.sum(torch.square(anchor_pos_w - command.robot_anchor_pos_w), dim=-1)
    return torch.exp(-error / std**2)


def motion_global_anchor_orientation_error_exp_at_offset(
    env: ManagerBasedRLEnv, command_name: str, std: float, time_offset: int
) -> torch.Tensor:
    command = _get_motion_command(env, command_name)
    anchor_quat_w = _motion_anchor_quat_w_at_offset(command, time_offset)
    error = quat_error_magnitude(anchor_quat_w, command.robot_anchor_quat_w) ** 2
    return torch.exp(-error / std**2)


def motion_relative_body_position_error_exp_at_offset(
    env: ManagerBasedRLEnv,
    command_name: str,
    std: float,
    time_offset: int,
    body_names: list[str] | None = None,
) -> torch.Tensor:
    command = _get_motion_command(env, command_name)
    body_indexes = _get_body_indexes(command, body_names)
    body_pos_relative_w, _ = _motion_relative_body_targets_at_offset(command, time_offset)
    error = torch.sum(
        torch.square(body_pos_relative_w[:, body_indexes] - command.robot_body_pos_w[:, body_indexes]), dim=-1
    )
    return torch.exp(-error.mean(-1) / std**2)


def motion_relative_body_orientation_error_exp_at_offset(
    env: ManagerBasedRLEnv,
    command_name: str,
    std: float,
    time_offset: int,
    body_names: list[str] | None = None,
) -> torch.Tensor:
    command = _get_motion_command(env, command_name)
    body_indexes = _get_body_indexes(command, body_names)
    _, body_quat_relative_w = _motion_relative_body_targets_at_offset(command, time_offset)
    error = (
        quat_error_magnitude(body_quat_relative_w[:, body_indexes], command.robot_body_quat_w[:, body_indexes]) ** 2
    )
    return torch.exp(-error.mean(-1) / std**2)


def motion_global_body_linear_velocity_error_exp_at_offset(
    env: ManagerBasedRLEnv,
    command_name: str,
    std: float,
    time_offset: int,
    body_names: list[str] | None = None,
) -> torch.Tensor:
    command = _get_motion_command(env, command_name)
    body_indexes = _get_body_indexes(command, body_names)
    body_lin_vel_w = _motion_body_lin_vel_w_at_offset(command, time_offset)
    error = torch.sum(
        torch.square(body_lin_vel_w[:, body_indexes] - command.robot_body_lin_vel_w[:, body_indexes]), dim=-1
    )
    return torch.exp(-error.mean(-1) / std**2)


def motion_global_body_angular_velocity_error_exp_at_offset(
    env: ManagerBasedRLEnv,
    command_name: str,
    std: float,
    time_offset: int,
    body_names: list[str] | None = None,
) -> torch.Tensor:
    command = _get_motion_command(env, command_name)
    body_indexes = _get_body_indexes(command, body_names)
    body_ang_vel_w = _motion_body_ang_vel_w_at_offset(command, time_offset)
    error = torch.sum(
        torch.square(body_ang_vel_w[:, body_indexes] - command.robot_body_ang_vel_w[:, body_indexes]), dim=-1
    )
    return torch.exp(-error.mean(-1) / std**2)


def feet_contact_time(env: ManagerBasedRLEnv, sensor_cfg: SceneEntityCfg, threshold: float) -> torch.Tensor:
    contact_sensor = cast(ContactSensor, env.scene.sensors[sensor_cfg.name])
    first_air = contact_sensor.compute_first_air(env.step_dt, env.physics_dt)[:, sensor_cfg.body_ids]
    last_contact_time = contact_sensor.data.last_contact_time
    assert last_contact_time is not None
    last_contact_time = last_contact_time[:, sensor_cfg.body_ids]
    reward = torch.sum((last_contact_time < threshold) * first_air, dim=-1)
    return reward


def penalty_minimal_action_norm(env: ManagerBasedRLEnv) -> torch.Tensor:
    """Delta-action regularizer: exp(-||a_delta||) - 1."""
    return torch.exp(-torch.norm(env.action_manager.action, dim=-1)) - 1.0
