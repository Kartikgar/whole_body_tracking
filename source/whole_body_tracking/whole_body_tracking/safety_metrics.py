"""Shared rollout safety metrics for the SONIC and G1 checkpoint evaluators."""

from __future__ import annotations

import re

import numpy as np


FOOT_BODIES = ("left_ankle_roll_link", "right_ankle_roll_link")
ALLOWED_CONTACT_BODIES = FOOT_BODIES
FOOT_CONTACT_FORCE_N = 10.0
OTHER_CONTACT_FORCE_N = 1.0
SLIP_SPEED_MPS = 0.1
NEAR_EFFORT_LIMIT_RATIO = 0.9


def g1_actuator_velocity_limits(joint_names: list[str]) -> np.ndarray:
    """Use the G1 actuator configuration as the physical speed reference for both simulators."""
    from whole_body_tracking.robots.g1 import G1_CYLINDER_CFG

    limits = np.full(len(joint_names), np.nan, dtype=np.float32)
    for actuator in G1_CYLINDER_CFG.actuators.values():
        values = actuator.velocity_limit_sim
        for index, name in enumerate(joint_names):
            if not any(re.fullmatch(pattern, name) for pattern in actuator.joint_names_expr):
                continue
            if isinstance(values, dict):
                matched = [float(value) for pattern, value in values.items() if re.fullmatch(pattern, name)]
                if len(matched) != 1:
                    raise ValueError(f"Expected one velocity limit for {name}, got {matched}")
                limits[index] = matched[0]
            else:
                limits[index] = float(values)
    if not np.isfinite(limits).all() or np.any(limits <= 0):
        missing = [name for name, value in zip(joint_names, limits) if not np.isfinite(value) or value <= 0]
        raise ValueError(f"Missing G1 actuator velocity limits for: {missing}")
    return limits


class SafetyMetricsAccumulator:
    """Accumulate valid-prefix contact, slip, and joint-limit measures per environment."""

    def __init__(
        self, *, num_envs: int, dt: float, joint_names: list[str], body_names: list[str],
        contact_body_names: list[str], joint_pos_limits: np.ndarray,
        joint_vel_limits: np.ndarray, joint_effort_limits: np.ndarray,
        soft_joint_pos_limits: np.ndarray | None = None,
    ):
        self.num_envs = int(num_envs)
        self.dt = float(dt)
        self.joint_names = list(joint_names)
        self.body_names = list(body_names)
        self.contact_body_names = list(contact_body_names)
        if self.num_envs < 1 or self.dt <= 0:
            raise ValueError("num_envs and dt must be positive")
        self.foot_body_ids = [self.body_names.index(name) for name in FOOT_BODIES]
        self.foot_contact_ids = [self.contact_body_names.index(name) for name in FOOT_BODIES]
        self.other_contact_ids = [
            index for index, name in enumerate(self.contact_body_names)
            if name not in ALLOWED_CONTACT_BODIES
        ]
        joint_count = len(self.joint_names)
        self.pos_limits = np.asarray(joint_pos_limits, dtype=np.float32)
        self.soft_pos_limits = np.asarray(
            joint_pos_limits if soft_joint_pos_limits is None else soft_joint_pos_limits, dtype=np.float32
        )
        self.vel_limits = np.asarray(joint_vel_limits, dtype=np.float32)
        self.effort_limits = np.asarray(joint_effort_limits, dtype=np.float32)
        if self.pos_limits.shape != (joint_count, 2) or self.soft_pos_limits.shape != (joint_count, 2) or self.vel_limits.shape != (joint_count,) or self.effort_limits.shape != (joint_count,):
            raise ValueError("Joint limit array shapes do not match joint_names")
        self.valid_pos = np.isfinite(self.pos_limits).all(axis=1) & (self.pos_limits[:, 0] < self.pos_limits[:, 1])
        self.valid_soft_pos = np.isfinite(self.soft_pos_limits).all(axis=1) & (self.soft_pos_limits[:, 0] < self.soft_pos_limits[:, 1])
        self.valid_vel = np.isfinite(self.vel_limits) & (self.vel_limits > 0)
        self.valid_effort = np.isfinite(self.effort_limits) & (self.effort_limits > 0)
        if not self.valid_pos.any() or not self.valid_soft_pos.any() or not self.valid_vel.all() or not self.valid_effort.all():
            raise ValueError("Safety metrics require finite position bounds and positive velocity/effort limits")

        n, c = self.num_envs, len(self.contact_body_names)
        self.foot_stance_samples = np.zeros((n, 2), dtype=np.int64)
        self.foot_slip_samples = np.zeros((n, 2), dtype=np.int64)
        self.foot_slip_distance = np.zeros((n, 2), dtype=np.float64)
        self.foot_peak_stance_speed = np.zeros((n, 2), dtype=np.float64)
        self.contact_previous = np.zeros((n, c), dtype=bool)
        self.contact_events = np.zeros((n, c), dtype=np.int64)
        self.contact_samples = np.zeros((n, c), dtype=np.int64)
        self.contact_peak_force = np.zeros((n, c), dtype=np.float64)
        self.non_support_any_samples = np.zeros(n, dtype=np.int64)
        self.joint_sample_count = np.zeros(n, dtype=np.int64)
        self.min_pos_margin = np.full(n, np.inf, dtype=np.float64)
        self.min_pos_margin_by_joint = np.full((n, joint_count), np.inf, dtype=np.float64)
        self.pos_violation_samples = np.zeros(n, dtype=np.int64)
        self.min_soft_pos_margin = np.full(n, np.inf, dtype=np.float64)
        self.soft_pos_violation_samples = np.zeros(n, dtype=np.int64)
        self.max_vel_ratio = np.zeros(n, dtype=np.float64)
        self.max_vel_ratio_by_joint = np.zeros((n, joint_count), dtype=np.float64)
        self.vel_over_samples = np.zeros(n, dtype=np.int64)
        self.max_effort_ratio = np.zeros(n, dtype=np.float64)
        self.max_effort_ratio_by_joint = np.zeros((n, joint_count), dtype=np.float64)
        self.effort_near_samples = np.zeros(n, dtype=np.int64)
        self.effort_over_samples = np.zeros(n, dtype=np.int64)

    def update(
        self, *, active: np.ndarray, joint_pos: np.ndarray, joint_vel: np.ndarray,
        applied_torque: np.ndarray, body_lin_vel_w: np.ndarray, contact_forces_w_history: np.ndarray,
    ) -> None:
        active = np.asarray(active, dtype=bool)
        joint_pos = np.asarray(joint_pos)
        joint_vel = np.asarray(joint_vel)
        applied_torque = np.asarray(applied_torque)
        body_lin_vel_w = np.asarray(body_lin_vel_w)
        forces = np.asarray(contact_forces_w_history)
        n, j, c = self.num_envs, len(self.joint_names), len(self.contact_body_names)
        if active.shape != (n,) or joint_pos.shape != (n, j) or joint_vel.shape != (n, j) or applied_torque.shape != (n, j):
            raise ValueError("Safety joint/state arrays have unexpected shapes")
        if body_lin_vel_w.shape != (n, len(self.body_names), 3) or forces.ndim != 4 or forces.shape[0] != n or forces.shape[2:] != (c, 3):
            raise ValueError("Safety body/contact arrays have unexpected shapes")
        if not active.any():
            return

        peak_force = np.linalg.norm(forces, axis=-1).max(axis=1)
        contact = peak_force > OTHER_CONTACT_FORCE_N
        self.contact_events[active] += (contact & ~self.contact_previous)[active]
        self.contact_samples[active] += contact[active]
        self.contact_peak_force[active] = np.maximum(self.contact_peak_force[active], peak_force[active])
        self.contact_previous[active] = contact[active]
        self.non_support_any_samples[active] += contact[:, self.other_contact_ids].any(axis=1)[active]

        foot_force = peak_force[:, self.foot_contact_ids]
        stance = foot_force > FOOT_CONTACT_FORCE_N
        foot_speed = np.linalg.norm(body_lin_vel_w[:, self.foot_body_ids, :2], axis=-1)
        self.foot_stance_samples[active] += stance[active]
        self.foot_slip_samples[active] += (stance & (foot_speed > SLIP_SPEED_MPS))[active]
        self.foot_slip_distance[active] += (foot_speed * stance * self.dt)[active]
        self.foot_peak_stance_speed[active] = np.maximum(
            self.foot_peak_stance_speed[active], (foot_speed * stance)[active]
        )

        margin = np.minimum(joint_pos - self.pos_limits[:, 0], self.pos_limits[:, 1] - joint_pos)
        margin[:, ~self.valid_pos] = np.inf
        self.min_pos_margin[active] = np.minimum(self.min_pos_margin[active], margin[active].min(axis=1))
        self.min_pos_margin_by_joint[active] = np.minimum(self.min_pos_margin_by_joint[active], margin[active])
        self.pos_violation_samples[active] += (margin[active, :][:, self.valid_pos] < 0).sum(axis=1)
        soft_margin = np.minimum(joint_pos - self.soft_pos_limits[:, 0], self.soft_pos_limits[:, 1] - joint_pos)
        soft_margin[:, ~self.valid_soft_pos] = np.inf
        self.min_soft_pos_margin[active] = np.minimum(self.min_soft_pos_margin[active], soft_margin[active].min(axis=1))
        self.soft_pos_violation_samples[active] += (soft_margin[active, :][:, self.valid_soft_pos] < 0).sum(axis=1)
        vel_ratio = np.abs(joint_vel) / self.vel_limits
        effort_ratio = np.abs(applied_torque) / self.effort_limits
        self.max_vel_ratio[active] = np.maximum(self.max_vel_ratio[active], vel_ratio[active].max(axis=1))
        self.max_vel_ratio_by_joint[active] = np.maximum(self.max_vel_ratio_by_joint[active], vel_ratio[active])
        self.vel_over_samples[active] += (vel_ratio[active] > 1.0).sum(axis=1)
        self.max_effort_ratio[active] = np.maximum(self.max_effort_ratio[active], effort_ratio[active].max(axis=1))
        self.max_effort_ratio_by_joint[active] = np.maximum(self.max_effort_ratio_by_joint[active], effort_ratio[active])
        self.effort_near_samples[active] += (effort_ratio[active] >= NEAR_EFFORT_LIMIT_RATIO).sum(axis=1)
        self.effort_over_samples[active] += (effort_ratio[active] > 1.0).sum(axis=1)
        self.joint_sample_count[active] += 1

    def rollout(self, env_id: int) -> tuple[dict[str, float], dict]:
        stance = self.foot_stance_samples[env_id]
        other = self.other_contact_ids
        sample_count = max(int(self.joint_sample_count[env_id]), 1)
        joint_count = len(self.joint_names)
        metrics = {
            "safety_foot_slip_distance_m": float(self.foot_slip_distance[env_id].max()),
            "safety_foot_slip_time_fraction": float(self.foot_slip_samples[env_id].sum() / max(int(stance.sum()), 1)),
            "safety_foot_peak_stance_speed_mps": float(self.foot_peak_stance_speed[env_id].max()),
            "safety_non_support_contact_event_count": int(self.contact_events[env_id, other].sum()),
            "safety_non_support_contact_time_s": float(self.non_support_any_samples[env_id] * self.dt),
            "safety_non_support_contact_peak_force_n": float(self.contact_peak_force[env_id, other].max(initial=0)),
            "safety_min_joint_position_margin_rad": float(self.min_pos_margin[env_id]) if self.joint_sample_count[env_id] else float("nan"),
            "safety_joint_position_violation_fraction": float(self.pos_violation_samples[env_id] / (sample_count * int(self.valid_pos.sum()))),
            "safety_min_joint_soft_position_margin_rad": float(self.min_soft_pos_margin[env_id]) if self.joint_sample_count[env_id] else float("nan"),
            "safety_joint_soft_position_violation_fraction": float(self.soft_pos_violation_samples[env_id] / (sample_count * max(int(self.valid_soft_pos.sum()), 1))),
            "safety_max_joint_speed_limit_ratio": float(self.max_vel_ratio[env_id]),
            "safety_joint_speed_limit_violation_fraction": float(self.vel_over_samples[env_id] / (sample_count * joint_count)),
            "safety_max_actuator_effort_limit_ratio": float(self.max_effort_ratio[env_id]),
            "safety_actuator_effort_near_limit_fraction": float(self.effort_near_samples[env_id] / (sample_count * joint_count)),
            "safety_actuator_effort_limit_violation_fraction": float(self.effort_over_samples[env_id] / (sample_count * joint_count)),
        }
        details = {
            "worst_joints": {
                "position_margin": self.joint_names[int(np.argmin(self.min_pos_margin_by_joint[env_id]))],
                "speed_limit_ratio": self.joint_names[int(np.argmax(self.max_vel_ratio_by_joint[env_id]))],
                "effort_limit_ratio": self.joint_names[int(np.argmax(self.max_effort_ratio_by_joint[env_id]))],
            },
            "foot_slip_by_body": {
                name: {
                    "stance_time_s": float(stance[index] * self.dt),
                    "slip_distance_m": float(self.foot_slip_distance[env_id, index]),
                    "slip_time_fraction": float(self.foot_slip_samples[env_id, index] / max(int(stance[index]), 1)),
                    "peak_stance_speed_mps": float(self.foot_peak_stance_speed[env_id, index]),
                }
                for index, name in enumerate(FOOT_BODIES)
            },
            "non_support_contacts_by_body": {
                self.contact_body_names[index]: {
                    "event_count": int(self.contact_events[env_id, index]),
                    "contact_time_s": float(self.contact_samples[env_id, index] * self.dt),
                    "peak_force_n": float(self.contact_peak_force[env_id, index]),
                }
                for index in other if self.contact_events[env_id, index] or self.contact_samples[env_id, index]
            },
        }
        return metrics, details


SAFETY_DEFINITIONS = {
    "foot_contact_threshold_n": FOOT_CONTACT_FORCE_N,
    "non_support_contact_threshold_n": OTHER_CONTACT_FORCE_N,
    "foot_slip_speed_threshold_mps": SLIP_SPEED_MPS,
    "allowed_contact_bodies": list(ALLOWED_CONTACT_BODIES),
    "contact_force_kind": "net contact-force magnitude; peak over physics-step history within each control step",
    "slip_kind": "horizontal ankle-link speed during detected stance; proxy for sole slip",
    "actuator_effort_kind": "implicit actuator estimated applied effort after clipping",
    "limits_kind": "robot hard and soft joint position limits; simulator effort limits; G1 actuator-config speed limits",
}
