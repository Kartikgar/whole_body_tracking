"""Observation assembly for the modular Genesis sim2sim evaluator."""

from __future__ import annotations

from collections import deque

import numpy as np

from sim2sim_genesis.constants import OBSERVATION_NOISE_RANGES, SUPPORTED_TERM_DIMS
from sim2sim_genesis.math_utils import (
    quat_conjugate_wxyz,
    quat_mul_wxyz,
    quat_rotate_inverse_wxyz,
    quat_to_matrix_wxyz,
)
from sim2sim_genesis.onnx_policy import PolicyMeta


class ObservationBuilder:
    """Rebuild Isaac-style policy observations from Genesis state and ONNX references."""

    def __init__(
        self,
        meta: PolicyMeta,
        num_envs: int,
        add_noise: bool,
        rng: np.random.Generator,
        motion_file: str | None = None,
    ):
        """Create the observation builder and load optional motion-action payloads."""

        self.meta = meta
        self.num_envs = max(int(num_envs), 1)
        self.add_noise = bool(add_noise)
        self.rng = rng
        self.num_joints = len(meta.joint_names)
        self.num_actions = len(meta.action_scale)
        self.motion_actions: np.ndarray | None = None
        if motion_file is not None:
            payload = np.load(motion_file)
            if "action" in payload.files:
                self.motion_actions = payload["action"].astype(np.float32)
            elif "actions" in payload.files:
                self.motion_actions = payload["actions"].astype(np.float32)

        self.requires_motion_action = "motion_joint_action" in meta.observation_names
        if self.requires_motion_action and self.motion_actions is None:
            raise RuntimeError(
                "Policy observation includes 'motion_joint_action' but motion file with `action` or `actions` "
                "was not provided."
            )

        self.default_joint_pos_by_env = np.repeat(meta.default_joint_pos[None, :], self.num_envs, axis=0).astype(
            np.float32
        )
        self.last_action = np.zeros((self.num_envs, self.num_actions), dtype=np.float32)
        self.term_dims = self.build_term_dims()
        self.term_history: dict[str, deque[np.ndarray]] = {}
        self.reset()

    def build_term_dims(self) -> dict[str, int]:
        """Build the concrete observation term dimensions for the current policy."""

        dims: dict[str, int] = {}
        for term_name, dim_name in SUPPORTED_TERM_DIMS.items():
            if dim_name == "double_joint_count":
                dims[term_name] = self.num_joints * 2
            elif dim_name == "joint_count":
                dims[term_name] = self.num_joints
            elif dim_name == "action_count":
                dims[term_name] = self.num_actions
            else:
                dims[term_name] = int(dim_name)
        return dims

    def compute_obs_dim(self, model_dim: int | str | None) -> int:
        """Compute the flattened observation width and validate it against the ONNX input."""

        obs_dim = 0
        for name, history in zip(self.meta.observation_names, self.meta.observation_history_lengths, strict=True):
            if name not in self.term_dims:
                raise RuntimeError(f"Unsupported observation term '{name}' for modular sim2sim script.")
            obs_dim += self.term_dims[name] * history

        if isinstance(model_dim, int) and model_dim != obs_dim:
            raise RuntimeError(f"Observation dim mismatch: metadata-built={obs_dim}, ONNX-input={model_dim}")
        return obs_dim

    def reset(self) -> None:
        """Reset observation history and previous-action state for a new rollout."""

        self.term_history = {}
        for name, history in zip(self.meta.observation_names, self.meta.observation_history_lengths, strict=True):
            dim = self.term_dims[name]
            self.term_history[name] = deque(
                (np.zeros((self.num_envs, dim), dtype=np.float32) for _ in range(history)),
                maxlen=history,
            )
        self.last_action[:] = 0.0

    def default_joint_pos_for_batch(self, batch_size: int) -> np.ndarray:
        """Return per-environment default joint positions for a batch size."""

        if batch_size <= 0:
            raise ValueError(f"Expected positive batch size, got {batch_size}.")
        if self.default_joint_pos_by_env.shape[0] == batch_size:
            return self.default_joint_pos_by_env
        if self.default_joint_pos_by_env.shape[0] == 1:
            return np.repeat(self.default_joint_pos_by_env, batch_size, axis=0)
        if batch_size == 1:
            return self.default_joint_pos_by_env[:1]
        raise RuntimeError(
            "Joint-default-pos batch mismatch: "
            f"have {self.default_joint_pos_by_env.shape[0]} defaults, requested batch={batch_size}."
        )

    def motion_action_at(self, time_step: int, batch_size: int) -> np.ndarray:
        """Return the recorded motion action for open-loop delta policies."""

        if self.motion_actions is None:
            raise RuntimeError("motion_joint_action was requested but no motion file was loaded.")
        index = min(time_step, self.motion_actions.shape[0] - 1)
        action = self.motion_actions[index].astype(np.float32)
        if action.shape[0] not in (self.num_actions, self.num_joints):
            raise RuntimeError(
                f"Unexpected motion action dimension {action.shape[0]}; expected {self.num_actions} or {self.num_joints}."
            )
        return np.repeat(action[None, :], batch_size, axis=0)

    def update_last_action(self, action_batch: np.ndarray) -> None:
        """Store the most recent policy action for autoregressive observation channels."""

        self.last_action = np.asarray(action_batch, dtype=np.float32).copy()

    def build(
        self,
        state_batch: dict[str, np.ndarray],
        reference_batch: dict[str, np.ndarray],
        anchor_idx: int,
        time_step: int,
        obs_dim_expected: int,
    ) -> np.ndarray:
        """Assemble the flattened observation vector for all active environments."""

        batch_size = int(state_batch["joint_pos"].shape[0])
        if batch_size != self.num_envs:
            raise RuntimeError(f"Observation batch mismatch: got {batch_size}, expected {self.num_envs}.")

        robot_anchor_pos = state_batch["body_pos_w"][:, anchor_idx]
        robot_anchor_quat = state_batch["body_quat_w"][:, anchor_idx]
        ref_anchor_pos = reference_batch["body_pos_w"][:, anchor_idx]
        ref_anchor_quat = reference_batch["body_quat_w"][:, anchor_idx]

        rel_anchor_pos = quat_rotate_inverse_wxyz(robot_anchor_quat, ref_anchor_pos - robot_anchor_pos)
        rel_anchor_quat = quat_mul_wxyz(quat_conjugate_wxyz(robot_anchor_quat), ref_anchor_quat)
        rel_anchor_rotmat = quat_to_matrix_wxyz(rel_anchor_quat)
        rel_anchor_ori = rel_anchor_rotmat[:, :, :2].reshape(batch_size, -1)

        base_lin_vel = quat_rotate_inverse_wxyz(state_batch["root_quat_w"], state_batch["root_lin_vel_w"])
        base_ang_vel = quat_rotate_inverse_wxyz(state_batch["root_quat_w"], state_batch["root_ang_vel_w"])

        terms: dict[str, np.ndarray] = {
            "command": np.concatenate([reference_batch["joint_pos"], reference_batch["joint_vel"]], axis=1),
            "motion_anchor_pos_b": rel_anchor_pos,
            "motion_anchor_ori_b": rel_anchor_ori,
            "base_lin_vel": base_lin_vel,
            "base_ang_vel": base_ang_vel,
            "joint_pos": state_batch["joint_pos"] - self.default_joint_pos_for_batch(batch_size),
            "joint_vel": state_batch["joint_vel"],
            "actions": self.last_action.copy(),
        }
        if self.requires_motion_action:
            terms["motion_joint_action"] = self.motion_action_at(time_step, batch_size)

        obs_parts = []
        for name in self.meta.observation_names:
            current = terms[name].astype(np.float32).reshape(batch_size, -1)
            if self.add_noise and name in OBSERVATION_NOISE_RANGES:
                noise_min, noise_max = OBSERVATION_NOISE_RANGES[name]
                current = current + self.rng.uniform(noise_min, noise_max, size=current.shape).astype(np.float32)
            self.term_history[name].append(current)
            obs_parts.append(np.concatenate(list(self.term_history[name]), axis=1))

        observation = np.concatenate(obs_parts, axis=1).astype(np.float32)
        if observation.shape[1] != obs_dim_expected:
            raise RuntimeError(
                f"Built obs dim {observation.shape[1]} does not match expected {obs_dim_expected}."
            )
        return observation
