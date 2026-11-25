"""Action -> joint-target mapping for the batched MuJoCo evaluator (PD runs inside MuJoCo actuators)."""

from __future__ import annotations

import numpy as np

from sim2sim_genesis.onnx_policy import PolicyMeta


class MujocoPdController:
    """Convert ONNX actions [N, A] into joint position targets [N, J]."""

    def __init__(self, *, meta: PolicyMeta):
        self.meta = meta

    def compute_joint_target(self, raw_action: np.ndarray) -> np.ndarray:
        action = np.asarray(raw_action, dtype=np.float32)
        if action.ndim == 1:
            action = action[None, :]
        if action.shape[1] != len(self.meta.joint_names):
            raise ValueError(f"Action dim {action.shape[1]} does not match joint count {len(self.meta.joint_names)}.")
        return (self.meta.default_joint_pos[None, :] + self.meta.action_scale[None, :] * action).astype(np.float32)
