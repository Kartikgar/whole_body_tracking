from __future__ import annotations

import torch

from isaaclab.envs.mdp.actions.actions_cfg import JointPositionActionCfg
from isaaclab.envs.mdp.actions.joint_actions import JointPositionAction
from isaaclab.managers.action_manager import ActionTerm
import isaaclab.utils.string as string_utils
from isaaclab.utils import configclass


class DeltaJointPositionAction(JointPositionAction):
    """Joint-position action with open-loop motion action added before scaling."""

    cfg: DeltaJointPositionActionCfg

    def __init__(self, cfg: DeltaJointPositionActionCfg, env):
        super().__init__(cfg, env)
        self._motion_command = env.command_manager.get_term(cfg.motion_command_name)
        self._delta_action_indices = self._resolve_delta_action_indices(self.cfg.delta_action_joint_names)
        self._delta_action_dim = len(self._delta_action_indices)
        if self._delta_action_dim != self._num_joints:
            self._raw_actions = torch.zeros(self.num_envs, self._delta_action_dim, device=self.device)
        self._delta_actions_full = torch.zeros(self.num_envs, self._num_joints, device=self.device)

    @property
    def action_dim(self) -> int:
        # During parent initialization, `_raw_actions` is not created yet.
        if hasattr(self, "_raw_actions"):
            return int(self._raw_actions.shape[1])
        return self._num_joints

    def _resolve_delta_action_indices(self, delta_action_joint_names: list[str] | None) -> list[int]:
        if delta_action_joint_names is None:
            return list(range(self._num_joints))
        delta_indices, _ = string_utils.resolve_matching_names(
            delta_action_joint_names, self._joint_names, preserve_order=True
        )
        if len(delta_indices) == 0:
            raise ValueError("`delta_action_joint_names` resolved to zero joints.")
        return list(delta_indices)

    def _expand_delta_actions_to_full(self, delta_actions: torch.Tensor) -> torch.Tensor:
        if delta_actions.shape[1] == self._num_joints:
            return delta_actions
        if delta_actions.shape[1] != self._delta_action_dim:
            raise RuntimeError(
                "Delta action shape mismatch: "
                f"expected {self._delta_action_dim} (delta space) or {self._num_joints} (full body), "
                f"got {delta_actions.shape[1]}."
            )
        self._delta_actions_full.zero_()
        self._delta_actions_full[:, self._delta_action_indices] = delta_actions
        return self._delta_actions_full

    def process_actions(self, actions: torch.Tensor):
        self._raw_actions[:] = actions
        delta_actions_full = self._expand_delta_actions_to_full(self._raw_actions)

        if self._motion_command.has_joint_action:
            motion_action = self._motion_command.joint_action
            if motion_action.shape != self._processed_actions.shape:
                raise RuntimeError(
                    "Motion action shape mismatch: "
                    f"expected {self._processed_actions.shape}, got {motion_action.shape}."
                )
            combined_actions = delta_actions_full + motion_action
        elif self.cfg.require_motion_action:
            raise RuntimeError(
                "DeltaJointPositionAction requires motion files with `action`/`actions` but none was found."
            )
        else:
            combined_actions = delta_actions_full

        self._processed_actions = combined_actions * self._scale + self._offset
        if self.cfg.clip is not None:
            self._processed_actions = torch.clamp(
                self._processed_actions, min=self._clip[:, :, 0], max=self._clip[:, :, 1]
            )


class ExternalDeltaJointPositionAction(JointPositionAction):
    """Joint-position action with an externally provided delta action added before scaling."""

    cfg: ExternalDeltaJointPositionActionCfg

    def __init__(self, cfg: ExternalDeltaJointPositionActionCfg, env):
        super().__init__(cfg, env)
        self._external_delta_action_indices = self._resolve_external_delta_action_indices(
            self.cfg.external_delta_action_joint_names
        )
        self._external_delta_action_dim = len(self._external_delta_action_indices)
        self._external_delta_actions_full = torch.zeros(self.num_envs, self._num_joints, device=self.device)

    def _resolve_external_delta_action_indices(self, joint_names: list[str] | None) -> list[int]:
        if joint_names is None:
            return list(range(self._num_joints))
        delta_indices, _ = string_utils.resolve_matching_names(joint_names, self._joint_names, preserve_order=True)
        if len(delta_indices) == 0:
            raise ValueError("`external_delta_action_joint_names` resolved to zero joints.")
        return list(delta_indices)

    def _expand_external_delta_action(self, external_delta_action: torch.Tensor) -> torch.Tensor:
        if self.cfg.external_delta_action_joint_names is None:
            if external_delta_action.shape[1] != self._num_joints:
                raise RuntimeError(
                    "External delta action shape mismatch: "
                    f"expected {self._num_joints}, got {external_delta_action.shape[1]}. "
                    "Set `external_delta_action_joint_names` (for example ankle joints) to remap "
                    "reduced-dimension delta actions into the full-body action vector."
                )
            return external_delta_action

        if external_delta_action.shape[1] == self._external_delta_action_dim:
            self._external_delta_actions_full.zero_()
            self._external_delta_actions_full[:, self._external_delta_action_indices] = external_delta_action
            return self._external_delta_actions_full

        if external_delta_action.shape[1] == self._num_joints:
            self._external_delta_actions_full.zero_()
            self._external_delta_actions_full[:, self._external_delta_action_indices] = external_delta_action[
                :, self._external_delta_action_indices
            ]
            return self._external_delta_actions_full

        raise RuntimeError(
            "External delta action shape mismatch: "
            f"expected {self._external_delta_action_dim} (configured delta space) "
            f"or {self._num_joints} (full body), got {external_delta_action.shape[1]}."
        )

    def _get_external_delta_action(self) -> torch.Tensor:
        external_delta_action = getattr(self._env, self.cfg.external_action_buffer_name, None)
        if external_delta_action is None:
            if self.cfg.require_external_action:
                raise RuntimeError(
                    f"Expected external delta action buffer '{self.cfg.external_action_buffer_name}' on the env."
                )
            return torch.zeros_like(self._processed_actions)

        external_delta_action = external_delta_action.to(device=self.device, dtype=self._processed_actions.dtype)
        if external_delta_action.ndim != 2 or external_delta_action.shape[0] != self.num_envs:
            raise RuntimeError(
                "External delta action shape mismatch: "
                f"expected batch size {self.num_envs}, got {external_delta_action.shape}."
            )
        return self._expand_external_delta_action(external_delta_action)

    def process_actions(self, actions: torch.Tensor):
        self._raw_actions[:] = actions
        external_delta_action = self._get_external_delta_action()
        combined_actions = self._raw_actions + self.cfg.external_action_scale * external_delta_action

        self._processed_actions = combined_actions * self._scale + self._offset
        if self.cfg.clip is not None:
            self._processed_actions = torch.clamp(
                self._processed_actions, min=self._clip[:, :, 0], max=self._clip[:, :, 1]
            )


@configclass
class DeltaJointPositionActionCfg(JointPositionActionCfg):
    """Config for open-loop delta action training."""

    class_type: type[ActionTerm] = DeltaJointPositionAction

    motion_command_name: str = "motion"
    require_motion_action: bool = True
    delta_action_joint_names: list[str] | None = None


@configclass
class ExternalDeltaJointPositionActionCfg(JointPositionActionCfg):
    """Config for joint control with externally provided delta actions."""

    class_type: type[ActionTerm] = ExternalDeltaJointPositionAction

    external_action_buffer_name: str = "delta_external_actions"
    external_action_scale: float = 1.0
    require_external_action: bool = True
    external_delta_action_joint_names: list[str] | None = None
