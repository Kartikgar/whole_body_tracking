from __future__ import annotations

import torch
from collections.abc import Sequence

import isaaclab.sim as sim_utils
from isaaclab.envs.mdp.actions.actions_cfg import JointPositionActionCfg
from isaaclab.envs.mdp.actions.joint_actions import JointPositionAction
from isaaclab.managers.action_manager import ActionTerm
from isaaclab.markers import VisualizationMarkers, VisualizationMarkersCfg
import isaaclab.utils.string as string_utils
from isaaclab.utils import configclass
from isaaclab.utils.math import quat_from_angle_axis, quat_rotate


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
        # Pre-init debug-vis fields because ActionTerm.__init__ may call _set_debug_vis_impl().
        self._force_visualizer: VisualizationMarkers | None = None
        self._force_arrow_x_axis = torch.tensor([[1.0, 0.0, 0.0]], device=env.device, dtype=torch.float32)
        self._force_arrow_fallback_axis = torch.tensor([[0.0, 1.0, 0.0]], device=env.device, dtype=torch.float32)
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
        self._force_body_id = int(self._force_body_ids[0])

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

        # Debug-vis helpers for rendering COM-force arrows.
        self._force_arrow_x_axis = self._force_arrow_x_axis.to(device=self.device)
        self._force_arrow_fallback_axis = self._force_arrow_fallback_axis.to(device=self.device)
        self._reset_force_log_stats()

    @property
    def action_dim(self) -> int:
        return 3

    @property
    def raw_actions(self) -> torch.Tensor:
        return self._raw_actions

    @property
    def processed_actions(self) -> torch.Tensor:
        return self._processed_actions

    def _reset_force_log_stats(self):
        self._log_force_component_sum = torch.zeros(3, device=self.device, dtype=torch.float32)
        self._log_force_norm_sum = torch.zeros(1, device=self.device, dtype=torch.float32)
        self._log_force_sample_count = 0

    def _accumulate_force_log_stats(self, applied_force_b: torch.Tensor):
        force_b = applied_force_b.detach()
        if force_b.ndim != 2 or force_b.shape[1] != 3:
            raise RuntimeError(f"Expected applied COM force shape [N, 3], got {tuple(force_b.shape)}.")
        self._log_force_component_sum += force_b.sum(dim=0)
        self._log_force_norm_sum += torch.linalg.norm(force_b, dim=-1).sum().view(1)
        self._log_force_sample_count += int(force_b.shape[0])

    def consume_applied_force_log_stats(self) -> dict[str, float]:
        if self._log_force_sample_count == 0:
            return {}

        denom = float(self._log_force_sample_count)
        force_component_mean = (self._log_force_component_sum / denom).detach().cpu()
        force_norm_mean = float((self._log_force_norm_sum / denom).item())
        stats = {
            "applied_force_net": force_norm_mean,
            "applied_force_x": float(force_component_mean[0].item()),
            "applied_force_y": float(force_component_mean[1].item()),
            "applied_force_z": float(force_component_mean[2].item()),
        }
        self._reset_force_log_stats()
        return stats

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
        self._raw_actions[:] = actions
        if self._force_clip_min is not None and self._force_clip_max is not None:
            # Optional force clip is applied in action space before scaling.
            self._raw_actions[:] = torch.clamp(self._raw_actions, min=self._force_clip_min, max=self._force_clip_max)

        # Delta policy output controls COM force.
        self._processed_actions = self._raw_actions * self._force_scale
        self._accumulate_force_log_stats(self._processed_actions)

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

    def _set_debug_vis_impl(self, debug_vis: bool):
        force_visualizer = getattr(self, "_force_visualizer", None)
        if debug_vis:
            if force_visualizer is None:
                force_marker_cfg = VisualizationMarkersCfg(
                    prim_path="/Visuals/Actions/com_force",
                    markers={
                        "force": sim_utils.ConeCfg(
                            radius=0.5,
                            height=1.0,
                            axis="X",
                            visual_material=sim_utils.PreviewSurfaceCfg(diffuse_color=(1.0, 1.0, 0.0)),
                        )
                    },
                )
                self._force_visualizer = VisualizationMarkers(force_marker_cfg)
                force_visualizer = self._force_visualizer
            force_visualizer.set_visibility(True)
        elif force_visualizer is not None:
            force_visualizer.set_visibility(False)

    def _compute_force_arrow_orientation(self, force_w: torch.Tensor) -> torch.Tensor:
        num_envs = force_w.shape[0]
        x_axis = self._force_arrow_x_axis.expand(num_envs, -1)
        force_norm = torch.linalg.norm(force_w, dim=-1, keepdim=True)

        direction = torch.where(force_norm > self.cfg.force_debug_vis_min_magnitude, force_w / force_norm, x_axis)
        direction = direction / direction.norm(dim=-1, keepdim=True).clamp_min(1.0e-9)
        dot = torch.clamp(torch.sum(x_axis * direction, dim=-1), -1.0, 1.0)

        cross = torch.cross(x_axis, direction, dim=-1)
        cross_norm = torch.linalg.norm(cross, dim=-1, keepdim=True)
        axis = cross / cross_norm.clamp_min(1.0e-9)
        angle = torch.acos(dot)
        orientations = quat_from_angle_axis(angle, axis)

        near_parallel = cross_norm.squeeze(-1) < 1.0e-6
        if torch.any(near_parallel):
            orientations[near_parallel] = torch.tensor([1.0, 0.0, 0.0, 0.0], device=self.device)
            anti_parallel = near_parallel & (dot < 0.0)
            if torch.any(anti_parallel):
                anti_angle = torch.full((int(anti_parallel.sum()),), torch.pi, device=self.device)
                anti_axis = self._force_arrow_fallback_axis.expand(int(anti_parallel.sum()), -1)
                orientations[anti_parallel] = quat_from_angle_axis(anti_angle, anti_axis)
        return orientations

    def _debug_vis_callback(self, event):
        del event
        if self._force_visualizer is None or not self._asset.is_initialized:
            return

        body_pos_w = self._asset.data.body_pos_w[:, self._force_body_id]
        body_quat_w = self._asset.data.body_quat_w[:, self._force_body_id]

        # Draw only one net-force arrow in world frame.
        force_w = quat_rotate(body_quat_w, self._processed_actions)
        orientations = self._compute_force_arrow_orientation(force_w)

        force_mag = torch.linalg.norm(self._processed_actions, dim=-1)
        arrow_len = force_mag * self.cfg.force_debug_vis_length_scale
        is_nonzero = force_mag > self.cfg.force_debug_vis_min_magnitude
        arrow_len = torch.where(
            is_nonzero,
            arrow_len,
            torch.full_like(arrow_len, self.cfg.force_debug_vis_hidden_arrow_length),
        )
        arrow_thickness = torch.full_like(arrow_len, self.cfg.force_debug_vis_thickness)
        scales = torch.stack((arrow_len, arrow_thickness, arrow_thickness), dim=-1)

        # Shift marker center so cone base stays at COM (instead of centering cone on COM).
        force_dir_w = torch.where(
            force_mag.unsqueeze(-1) > self.cfg.force_debug_vis_min_magnitude,
            force_w / force_mag.unsqueeze(-1).clamp_min(1.0e-9),
            torch.zeros_like(force_w),
        )
        marker_pos_w = body_pos_w + 0.5 * arrow_len.unsqueeze(-1) * force_dir_w
        self._force_visualizer.visualize(translations=marker_pos_w, orientations=orientations, scales=scales)

    def reset(self, env_ids: Sequence[int] | None = None) -> None:
        if env_ids is None:
            self._raw_actions[:] = 0.0
            self._processed_actions[:] = 0.0
            return
        self._raw_actions[env_ids] = 0.0
        self._processed_actions[env_ids] = 0.0


class ExternalDeltaComForceAction(DeltaComForceAction):
    """Joint-position action with an externally provided COM force applied before each sim step."""

    cfg: ExternalDeltaComForceActionCfg

    def __init__(self, cfg: ExternalDeltaComForceActionCfg, env):
        super().__init__(cfg, env)

        # Finetuning policy keeps the standard joint-space action dimension.
        self._raw_actions = torch.zeros(self.num_envs, self._num_joints, device=self.device)
        self._processed_actions = torch.zeros_like(self._raw_actions)
        self._processed_force_actions = torch.zeros(self.num_envs, 3, device=self.device)

        if isinstance(self._joint_scale, (float, int)):
            self._joint_scale = torch.full(
                (self.num_envs, self._num_joints), float(self._joint_scale), dtype=torch.float32, device=self.device
            )
        self._scale = self._joint_scale

    @property
    def action_dim(self) -> int:
        if hasattr(self, "_raw_actions"):
            return int(self._raw_actions.shape[1])
        return getattr(self, "_num_joints", 3)

    def _get_external_force_action(self) -> torch.Tensor:
        external_force_action = getattr(self._env, self.cfg.external_action_buffer_name, None)
        if external_force_action is None:
            if self.cfg.require_external_action:
                raise RuntimeError(
                    f"Expected external COM-force buffer '{self.cfg.external_action_buffer_name}' on the env."
                )
            return torch.zeros(self.num_envs, 3, device=self.device, dtype=self._processed_force_actions.dtype)

        external_force_action = external_force_action.to(device=self.device, dtype=self._processed_force_actions.dtype)
        if external_force_action.ndim != 2 or external_force_action.shape[0] != self.num_envs:
            raise RuntimeError(
                "External COM-force action shape mismatch: "
                f"expected batch size {self.num_envs}, got {external_force_action.shape}."
            )
        if external_force_action.shape[1] != 3:
            raise RuntimeError(
                "External COM-force action shape mismatch: "
                f"expected 3 force channels (Fx, Fy, Fz), got {external_force_action.shape[1]}."
            )
        return external_force_action

    def process_actions(self, actions: torch.Tensor):
        self._raw_actions[:] = actions

        # Base finetuning policy still outputs joint actions.
        self._processed_actions = self._raw_actions * self._joint_scale + self._offset
        if self._joint_clip is not None:
            self._processed_actions = torch.clamp(
                self._processed_actions, min=self._joint_clip[:, :, 0], max=self._joint_clip[:, :, 1]
            )

        # Frozen delta policy output controls COM force.
        external_force_action = self._get_external_force_action()
        if self._force_clip_min is not None and self._force_clip_max is not None:
            external_force_action = torch.clamp(
                external_force_action, min=self._force_clip_min, max=self._force_clip_max
            )
        self._processed_force_actions = external_force_action * self._force_scale
        self._accumulate_force_log_stats(self._processed_force_actions)

    def apply_actions(self):
        self._asset.set_joint_position_target(self._processed_actions, joint_ids=self._joint_ids)

        external_forces = self._processed_force_actions.view(self.num_envs, 1, 3)
        self._asset.set_external_force_and_torque(
            forces=external_forces,
            torques=self._external_torques,
            body_ids=self._force_body_ids,
        )

    def _debug_vis_callback(self, event):
        del event
        if self._force_visualizer is None or not self._asset.is_initialized:
            return

        body_pos_w = self._asset.data.body_pos_w[:, self._force_body_id]
        body_quat_w = self._asset.data.body_quat_w[:, self._force_body_id]

        force_w = quat_rotate(body_quat_w, self._processed_force_actions)
        orientations = self._compute_force_arrow_orientation(force_w)

        force_mag = torch.linalg.norm(self._processed_force_actions, dim=-1)
        arrow_len = force_mag * self.cfg.force_debug_vis_length_scale
        is_nonzero = force_mag > self.cfg.force_debug_vis_min_magnitude
        arrow_len = torch.where(
            is_nonzero,
            arrow_len,
            torch.full_like(arrow_len, self.cfg.force_debug_vis_hidden_arrow_length),
        )
        arrow_thickness = torch.full_like(arrow_len, self.cfg.force_debug_vis_thickness)
        scales = torch.stack((arrow_len, arrow_thickness, arrow_thickness), dim=-1)

        force_dir_w = torch.where(
            force_mag.unsqueeze(-1) > self.cfg.force_debug_vis_min_magnitude,
            force_w / force_mag.unsqueeze(-1).clamp_min(1.0e-9),
            torch.zeros_like(force_w),
        )
        marker_pos_w = body_pos_w + 0.5 * arrow_len.unsqueeze(-1) * force_dir_w
        self._force_visualizer.visualize(translations=marker_pos_w, orientations=orientations, scales=scales)

    def reset(self, env_ids: Sequence[int] | None = None) -> None:
        if env_ids is None:
            self._raw_actions[:] = 0.0
            self._processed_actions[:] = 0.0
            self._processed_force_actions[:] = 0.0
            return
        self._raw_actions[env_ids] = 0.0
        self._processed_actions[env_ids] = 0.0
        self._processed_force_actions[env_ids] = 0.0


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
    force_scale: float | tuple[float, float, float] = 0.1
    force_clip: float | tuple[float, float] | None = None
    force_debug_vis_length_scale: float = 0.02
    force_debug_vis_thickness: float = 0.1
    force_debug_vis_min_magnitude: float = 1.0e-4
    force_debug_vis_min_arrow_length: float = 0.05
    force_debug_vis_max_arrow_length: float = 5.0
    force_debug_vis_hidden_arrow_length: float = 1.0e-4


@configclass
class ExternalDeltaComForceActionCfg(DeltaComForceActionCfg):
    """Config for finetuning with joint-space base actions and external COM-force delta actions."""

    class_type: type[ActionTerm] = ExternalDeltaComForceAction

    external_action_buffer_name: str = "delta_external_actions"
    require_external_action: bool = True


@configclass
class ExternalDeltaJointPositionActionCfg(JointPositionActionCfg):
    """Config for joint control with externally provided delta actions."""

    class_type: type[ActionTerm] = ExternalDeltaJointPositionAction

    external_action_buffer_name: str = "delta_external_actions"
    external_action_scale: float = 1.0
    require_external_action: bool = True
    external_delta_action_joint_names: list[str] | None = None
