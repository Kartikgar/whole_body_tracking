from __future__ import annotations

import torch
from collections.abc import Sequence

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


class DeltaComForceAction(ActionTerm):
    """Action term for delta training with COM force actions and motion-action joint replay."""

    cfg: DeltaComForceActionCfg

    def __init__(self, cfg: DeltaComForceActionCfg, env):
        super().__init__(cfg, env)
        self._motion_command = env.command_manager.get_term(cfg.motion_command_name)

        # Resolve controlled joints for motion-action replay.
        self._joint_ids, self._joint_names = self._asset.find_joints(
            self.cfg.joint_names, preserve_order=self.cfg.preserve_order
        )
        self._num_joints = len(self._joint_ids)
        if self._num_joints == self._asset.num_joints and not self.cfg.preserve_order:
            self._joint_ids = slice(None)

        # Resolve a single body where COM force is applied.
        self._force_body_ids, self._force_body_names = self._asset.find_bodies(
            self.cfg.force_body_name, preserve_order=True
        )
        if len(self._force_body_ids) != 1:
            raise ValueError(
                f"`force_body_name` must resolve to exactly one body. Got {self._force_body_names} "
                f"for pattern '{self.cfg.force_body_name}'."
            )

        # Action-space buffers (policy output = Fx, Fy, Fz).
        self._raw_actions = torch.zeros(self.num_envs, 3, device=self.device)
        self._processed_actions = torch.zeros_like(self._raw_actions)

        # Joint-target and wrench buffers used during apply_actions().
        self._joint_position_targets = torch.zeros(self.num_envs, self._num_joints, device=self.device)
        self._external_torques = torch.zeros(self.num_envs, 1, 3, device=self.device)

        # Parse force scaling/clipping for action -> force mapping.
        self._force_scale = self._parse_force_scale(self.cfg.force_scale)
        # Keep `_scale` for compatibility with existing metadata exporters.
        self._scale = self._force_scale
        self._force_clip_min, self._force_clip_max = self._parse_force_clip(self.cfg.force_clip)

        # Parse joint replay affine transform (same semantics as JointPositionAction).
        if isinstance(cfg.scale, (float, int)):
            self._joint_scale = float(cfg.scale)
        elif isinstance(cfg.scale, dict):
            self._joint_scale = torch.ones(self.num_envs, self._num_joints, device=self.device)
            index_list, _, value_list = string_utils.resolve_matching_names_values(cfg.scale, self._joint_names)
            self._joint_scale[:, index_list] = torch.tensor(value_list, device=self.device)
        else:
            raise ValueError(f"Unsupported scale type: {type(cfg.scale)}. Supported types are float and dict.")

        if isinstance(cfg.offset, (float, int)):
            self._joint_offset = float(cfg.offset)
        elif isinstance(cfg.offset, dict):
            self._joint_offset = torch.zeros(self.num_envs, self._num_joints, device=self.device)
            index_list, _, value_list = string_utils.resolve_matching_names_values(cfg.offset, self._joint_names)
            self._joint_offset[:, index_list] = torch.tensor(value_list, device=self.device)
        else:
            raise ValueError(f"Unsupported offset type: {type(cfg.offset)}. Supported types are float and dict.")

        if cfg.use_default_offset:
            self._joint_offset = self._asset.data.default_joint_pos[:, self._joint_ids].clone()
        elif isinstance(self._joint_offset, float):
            # Keep compatibility with startup randomization that indexes action-term `_offset`.
            self._joint_offset = torch.full(
                (self.num_envs, self._num_joints), self._joint_offset, dtype=torch.float32, device=self.device
            )

        # Alias used by existing startup events (`randomize_joint_default_pos`) for in-place offset updates.
        self._offset = self._joint_offset

        self._joint_clip = None
        if cfg.clip is not None:
            if isinstance(cfg.clip, dict):
                self._joint_clip = torch.tensor([[-float("inf"), float("inf")]], device=self.device).repeat(
                    self.num_envs, self._num_joints, 1
                )
                index_list, _, value_list = string_utils.resolve_matching_names_values(cfg.clip, self._joint_names)
                self._joint_clip[:, index_list] = torch.tensor(value_list, device=self.device)
            else:
                raise ValueError(f"Unsupported clip type: {type(cfg.clip)}. Supported types are dict.")

    @property
    def action_dim(self) -> int:
        return 3

    @property
    def raw_actions(self) -> torch.Tensor:
        return self._raw_actions

    @property
    def processed_actions(self) -> torch.Tensor:
        return self._processed_actions

    def _parse_force_scale(self, force_scale: float | Sequence[float]) -> torch.Tensor:
        if isinstance(force_scale, (float, int)):
            return torch.full((self.num_envs, 3), float(force_scale), device=self.device)
        if isinstance(force_scale, Sequence):
            if len(force_scale) != 3:
                raise ValueError(
                    f"`force_scale` sequence must have 3 elements (Fx, Fy, Fz). Got {len(force_scale)}."
                )
            return (
                torch.tensor(force_scale, device=self.device, dtype=torch.float32)
                .view(1, 3)
                .repeat(self.num_envs, 1)
            )
        raise ValueError(
            f"Unsupported force_scale type: {type(force_scale)}. Supported types are float and length-3 sequence."
        )

    def _parse_force_clip(
        self, force_clip: float | Sequence[float] | None
    ) -> tuple[torch.Tensor | None, torch.Tensor | None]:
        if force_clip is None:
            return None, None
        if isinstance(force_clip, (float, int)):
            min_val, max_val = -abs(float(force_clip)), abs(float(force_clip))
        elif isinstance(force_clip, Sequence) and len(force_clip) == 2:
            min_val, max_val = float(force_clip[0]), float(force_clip[1])
        else:
            raise ValueError(
                f"Unsupported force_clip value: {force_clip}. Use float (symmetric) or length-2 sequence (min, max)."
            )
        min_tensor = torch.full((self.num_envs, 3), min_val, device=self.device)
        max_tensor = torch.full((self.num_envs, 3), max_val, device=self.device)
        return min_tensor, max_tensor

    def process_actions(self, actions: torch.Tensor):
        # Keep policy force commands bounded per-axis.
        self._raw_actions[:] = torch.clamp(actions, min=-1.0, max=1.0)

        # Delta policy output controls COM force.
        self._processed_actions = self._raw_actions * self._force_scale
        if self._force_clip_min is not None and self._force_clip_max is not None:
            self._processed_actions = torch.clamp(
                self._processed_actions, min=self._force_clip_min, max=self._force_clip_max
            )
        # Hard safety bound required by task setup.
        self._processed_actions = torch.clamp(self._processed_actions, min=-1.0, max=1.0)

        # Joint targets continue to replay motion action (open-loop baseline).
        if self._motion_command.has_joint_action:
            motion_action = self._motion_command.joint_action
            if motion_action.shape != self._joint_position_targets.shape:
                raise RuntimeError(
                    "Motion action shape mismatch: "
                    f"expected {self._joint_position_targets.shape}, got {motion_action.shape}."
                )
            replay_action = motion_action
        elif self.cfg.require_motion_action:
            raise RuntimeError(
                "DeltaComForceAction requires motion files with `action`/`actions` but none was found."
            )
        else:
            replay_action = torch.zeros_like(self._joint_position_targets)

        self._joint_position_targets = replay_action * self._joint_scale + self._offset
        if self._joint_clip is not None:
            self._joint_position_targets = torch.clamp(
                self._joint_position_targets, min=self._joint_clip[:, :, 0], max=self._joint_clip[:, :, 1]
            )

    def apply_actions(self):
        self._asset.set_joint_position_target(self._joint_position_targets, joint_ids=self._joint_ids)

        external_forces = self._processed_actions.view(self.num_envs, 1, 3)
        self._asset.set_external_force_and_torque(
            forces=external_forces,
            torques=self._external_torques,
            body_ids=self._force_body_ids,
        )

    def reset(self, env_ids: Sequence[int] | None = None) -> None:
        self._raw_actions[env_ids] = 0.0
        self._processed_actions[env_ids] = 0.0


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
class DeltaComForceActionCfg(JointPositionActionCfg):
    """Config for delta training where the policy outputs COM forces (Fx, Fy, Fz)."""

    class_type: type[ActionTerm] = DeltaComForceAction

    motion_command_name: str = "motion"
    require_motion_action: bool = True
    force_body_name: str = "torso_link"
    force_scale: float | tuple[float, float, float] = 1.0
    force_clip: float | tuple[float, float] | None = None


@configclass
class ExternalDeltaJointPositionActionCfg(JointPositionActionCfg):
    """Config for joint control with externally provided delta actions."""

    class_type: type[ActionTerm] = ExternalDeltaJointPositionAction

    external_action_buffer_name: str = "delta_external_actions"
    external_action_scale: float = 1.0
    require_external_action: bool = True
    external_delta_action_joint_names: list[str] | None = None
