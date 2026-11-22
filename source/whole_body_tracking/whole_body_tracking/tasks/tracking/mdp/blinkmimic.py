"""Observation and termination terms that reproduce BlinkMimic's policy inputs inside BeyondMimic tracking tasks.

BlinkMimic's policy sees (``blinkmimic.mdp.MDP.obs`` and ``blinkmimic.motion.MotionLib.goal``):

* per control step, ``obs_dim = 3J + 12`` values: joint positions relative to the default pose, joint velocities,
  root angular velocity (root frame), projected gravity, the previous action, and the 6D root orientation of the
  reference relative to the robot root (``R^T R_ref``); a history of ``hist`` steps (oldest first) is formed by the
  observation manager (``history_length`` on :func:`blinkmimic_proprio`);
* a goal of ``K`` future reference frames ``f+1..f+K`` (clamped to the clip end), each ``(dp, rot6d, v, w, q,
  0.1 qd)`` with ``dp``/``rot6d`` in the heading frame of the current reference root and ``v``/``w`` in each future
  root frame.

Joints are in the order of the action term (``preserve_order=True`` with BlinkMimic's MJCF joint list), so the same
indices map Isaac joints to the policy. The motion NPZ must come from ``scripts/blinkmimic/frax_to_motion_npz.py``
(frame-identical to BlinkMimic's MotionLib) and the command's anchor body must be the floating-base root.
"""

from __future__ import annotations

from typing import TYPE_CHECKING

import torch

from whole_body_tracking.blinkmimic.rotations import mat_to_6d, mm3, mv3, quat_to_mat, yaw_mat
from whole_body_tracking.tasks.tracking.mdp.commands import MotionCommand

if TYPE_CHECKING:
    from isaaclab.envs import ManagerBasedEnv, ManagerBasedRLEnv


def _policy_joint_ids(env: ManagerBasedEnv, action_name: str) -> list[int] | slice:
    return env.action_manager.get_term(action_name)._joint_ids


def blinkmimic_proprio(env: ManagerBasedEnv, command_name: str = "motion", action_name: str = "joint_pos") -> torch.Tensor:
    """One step of BlinkMimic's proprioceptive observation (3J + 12)."""
    command: MotionCommand = env.command_manager.get_term(command_name)
    robot = command.robot
    ids = _policy_joint_ids(env, action_name)
    q = robot.data.joint_pos[:, ids] - robot.data.default_joint_pos[:, ids]
    qd = robot.data.joint_vel[:, ids]
    R = quat_to_mat(robot.data.root_link_quat_w)
    Rt = R.transpose(-1, -2)
    # CoM angular velocity: equal to the link's, and not stale right after a reset write (Isaac Lab 2.1)
    w_b = mv3(Rt, robot.data.root_com_ang_vel_w)
    g_b = -R[..., 2, :]  # R^T (0, 0, -1)
    a_prev = env.action_manager.get_term(action_name).raw_actions
    rel = mat_to_6d(mm3(Rt, quat_to_mat(command.anchor_quat_w)))
    return torch.cat((q, qd, w_b, g_b, a_prev, rel), -1)


def blinkmimic_goal(
    env: ManagerBasedEnv, command_name: str = "motion", action_name: str = "joint_pos", num_frames: int = 10
) -> torch.Tensor:
    """BlinkMimic goal features of the ``num_frames`` future reference frames (num_frames * (15 + 2J))."""
    command: MotionCommand = env.command_manager.get_term(command_name)
    motion = command.motion
    ids = _policy_joint_ids(env, action_name)
    a = command.motion_anchor_body_index
    tid, t = command.trajectory_ids, command.time_steps
    n, K = t.shape[0], num_frames
    last = motion.trajectory_time_step_total[tid] - 1
    fut = torch.minimum(t[:, None] + torch.arange(1, K + 1, device=t.device), last[:, None])  # (n, K)
    tid_k = tid[:, None].expand(n, K).reshape(-1)
    fut_k = fut.reshape(-1)

    p0 = motion.get_body_pos_w(tid, t)[:, a]
    R0 = quat_to_mat(motion.get_body_quat_w(tid, t)[:, a])
    pk = motion.get_body_pos_w(tid_k, fut_k)[:, a].view(n, K, 3)
    Rk = quat_to_mat(motion.get_body_quat_w(tid_k, fut_k)[:, a]).view(n, K, 3, 3)
    vk = motion.get_body_lin_vel_w(tid_k, fut_k)[:, a].view(n, K, 3)
    wk = motion.get_body_ang_vel_w(tid_k, fut_k)[:, a].view(n, K, 3)
    qk = motion.get_joint_pos(tid_k, fut_k)[:, ids].view(n, K, -1)
    qdk = motion.get_joint_vel(tid_k, fut_k)[:, ids].view(n, K, -1)

    Ht = yaw_mat(R0).transpose(-1, -2).unsqueeze(1)  # (n, 1, 3, 3)
    Rkt = Rk.transpose(-1, -2)
    dp = mv3(Ht, pk - p0.unsqueeze(1))
    rot = mat_to_6d(mm3(Ht, Rk))
    feats = torch.cat((dp, rot, mv3(Rkt, vk), mv3(Rkt, wk), qk, qdk * 0.1), -1)  # (n, K, 15 + 2J)
    return feats.reshape(n, -1)


def blinkmimic_root_error(
    env: ManagerBasedRLEnv, command_name: str = "motion", height: float = 0.25, tilt: float = 0.8
) -> torch.Tensor:
    """BlinkMimic's termination: root height error > ``height`` or |R_ref[2,2] - R[2,2]| > ``tilt``."""
    command: MotionCommand = env.command_manager.get_term(command_name)
    robot = command.robot
    e_z = (command.anchor_pos_w[:, 2] - robot.data.root_link_pos_w[:, 2]).abs()
    up_ref = quat_to_mat(command.anchor_quat_w)[:, 2, 2]
    up = quat_to_mat(robot.data.root_link_quat_w)[:, 2, 2]
    return (e_z > height) | ((up_ref - up).abs() > tilt)
