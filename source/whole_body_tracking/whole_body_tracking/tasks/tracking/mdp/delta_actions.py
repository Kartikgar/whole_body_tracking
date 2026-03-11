from __future__ import annotations

import torch

from isaaclab.envs.mdp.actions.actions_cfg import JointPositionActionCfg
from isaaclab.envs.mdp.actions.joint_actions import JointPositionAction
from isaaclab.managers.action_manager import ActionTerm
from isaaclab.utils import configclass


class DeltaJointPositionAction(JointPositionAction):
    """Joint-position action with open-loop motion action added before scaling."""

    cfg: DeltaJointPositionActionCfg

    def __init__(self, cfg: DeltaJointPositionActionCfg, env):
        super().__init__(cfg, env)
        self._motion_command = env.command_manager.get_term(cfg.motion_command_name)

    def process_actions(self, actions: torch.Tensor):
        self._raw_actions[:] = actions

        if self._motion_command.has_joint_action:
            motion_action = self._motion_command.joint_action
            if motion_action.shape != self._raw_actions.shape:
                raise RuntimeError(
                    f"Motion action shape mismatch: expected {self._raw_actions.shape}, got {motion_action.shape}."
                )
            combined_actions = self._raw_actions + motion_action
        elif self.cfg.require_motion_action:
            raise RuntimeError(
                "DeltaJointPositionAction requires motion files with `action`/`actions` but none was found."
            )
        else:
            combined_actions = self._raw_actions

        self._processed_actions = combined_actions * self._scale + self._offset
        if self.cfg.clip is not None:
            self._processed_actions = torch.clamp(
                self._processed_actions, min=self._clip[:, :, 0], max=self._clip[:, :, 1]
            )


class ExternalDeltaJointPositionAction(JointPositionAction):
    """Joint-position action with an externally provided delta action added before scaling."""

    cfg: ExternalDeltaJointPositionActionCfg

    def _get_external_delta_action(self) -> torch.Tensor:
        external_delta_action = getattr(self._env, self.cfg.external_action_buffer_name, None)
        if external_delta_action is None:
            if self.cfg.require_external_action:
                raise RuntimeError(
                    f"Expected external delta action buffer '{self.cfg.external_action_buffer_name}' on the env."
                )
            return torch.zeros_like(self._raw_actions)

        external_delta_action = external_delta_action.to(self.device)
        if external_delta_action.shape != self._raw_actions.shape:
            raise RuntimeError(
                "External delta action shape mismatch: "
                f"expected {self._raw_actions.shape}, got {external_delta_action.shape}."
            )
        return external_delta_action

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


@configclass
class ExternalDeltaJointPositionActionCfg(JointPositionActionCfg):
    """Config for joint control with externally provided delta actions."""

    class_type: type[ActionTerm] = ExternalDeltaJointPositionAction

    external_action_buffer_name: str = "delta_external_actions"
    external_action_scale: float = 1.0
    require_external_action: bool = True
