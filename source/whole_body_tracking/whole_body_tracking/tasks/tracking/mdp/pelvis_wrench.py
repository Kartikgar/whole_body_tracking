"""Pelvis-local wrench actions for target-dynamics emulation."""

import math
import torch
from isaaclab.utils import configclass
from .delta_actions import DeltaComForceAction, DeltaComForceActionCfg


class DeltaPelvisWrenchAction(DeltaComForceAction):
    """Replay joint actions while learning a six-component pelvis wrench."""

    deployment_export = False

    def __init__(self, cfg, env):
        for name in ("force_scale", "torque_scale", "action_clip"):
            value = getattr(cfg, name)
            if isinstance(value, bool) or not isinstance(value, (int, float)) or not math.isfinite(value) or value <= 0:
                raise ValueError(f"{name} must be a finite positive scalar")
        if cfg.force_body_name != "pelvis" or cfg.force_clip is not None:
            raise ValueError("Pelvis wrench requires force_body_name='pelvis' and force_clip=None; use action_clip")
        super().__init__(cfg, env)
        self._raw_actions = torch.zeros(self.num_envs, self.action_dim, device=self.device)
        self._processed_actions = torch.zeros_like(self._raw_actions)
        self._wrench = torch.zeros(self.num_envs, 6, device=self.device)
        self._wrench_scale = torch.tensor([cfg.force_scale] * 3 + [cfg.torque_scale] * 3, device=self.device)
        self._normalized_wrench = torch.zeros_like(self._wrench)
        self._previous_normalized_wrench = torch.zeros_like(self._wrench)
        self._stats = torch.zeros(14, device=self.device)
        self._samples = 0

    @property
    def action_dim(self):
        return 6

    @property
    def normalized_wrench(self):
        """Return the clipped wrench command before force/torque unit scaling."""
        return self._normalized_wrench

    @property
    def previous_normalized_wrench(self):
        """Return the normalized wrench command applied on the preceding control step."""
        return self._previous_normalized_wrench

    def wrench_contract(self):
        return dict(version=1, body="pelvis", frame="body_local", point="com",
                    components=["Fx", "Fy", "Fz", "Tx", "Ty", "Tz"],
                    force_scale=float(self.cfg.force_scale), torque_scale=float(self.cfg.torque_scale),
                    action_clip=float(self.cfg.action_clip))

    def _process_wrench(self, actions):
        if actions.shape != self._wrench.shape:
            raise ValueError(f"Expected wrench shape {tuple(self._wrench.shape)}, got {tuple(actions.shape)}")
        self._previous_normalized_wrench[:] = self._normalized_wrench
        self._normalized_wrench[:] = actions.clamp(-self.cfg.action_clip, self.cfg.action_clip)
        self._wrench[:] = self._normalized_wrench * self._wrench_scale
        self._stats[:6] += self._wrench.detach().sum(0)
        self._stats[6] += self._wrench[:, :3].detach().norm(dim=-1).sum()
        self._stats[7] += self._wrench[:, 3:].detach().norm(dim=-1).sum()
        self._stats[8:] += (actions.detach().abs() >= self.cfg.action_clip).sum(0)
        self._samples += self.num_envs

    def _set_joint_targets(self, actions):
        if actions.shape != self._joint_position_targets.shape:
            raise ValueError("Joint action shape does not match controlled joints")
        self._joint_position_targets[:] = actions * self._joint_scale + self._offset
        if self._joint_clip is not None:
            self._joint_position_targets[:] = torch.clamp(
                self._joint_position_targets, min=self._joint_clip[:, :, 0], max=self._joint_clip[:, :, 1])

    def process_actions(self, actions):
        self._process_wrench(actions)
        self._raw_actions[:] = actions
        self._processed_actions[:] = self._wrench
        if not self._motion_command.has_joint_action:
            raise RuntimeError("Pelvis-wrench open-loop requires recorded joint actions")
        self._set_joint_targets(self._motion_command.joint_action)

    def apply_actions(self):
        self._asset.set_joint_position_target(self._joint_position_targets, joint_ids=self._joint_ids)
        self._asset.set_external_force_and_torque(
            forces=self._wrench[:, None, :3], torques=self._wrench[:, None, 3:], body_ids=self._force_body_ids)

    def reset(self, env_ids=None):
        ids = slice(None) if env_ids is None else env_ids
        self._raw_actions[ids] = 0
        self._processed_actions[ids] = 0
        self._wrench[ids] = 0
        self._normalized_wrench[ids] = 0
        self._previous_normalized_wrench[ids] = 0
        self._asset.set_external_force_and_torque(
            forces=self._wrench[ids, None, :3], torques=self._wrench[ids, None, 3:],
            body_ids=self._force_body_ids, env_ids=env_ids)
        for name in ("delta_external_actions", "delta_base_actions"):
            buffer = getattr(self._env, name, None)
            if buffer is not None:
                buffer[ids] = 0

    def consume_applied_wrench_log_stats(self):
        if not self._samples:
            return {}
        keys = ["Fx", "Fy", "Fz", "Tx", "Ty", "Tz", "force_norm", "torque_norm"]
        keys += [f"saturation_{k}" for k in keys[:6]]
        result = dict(zip(keys, (self._stats / self._samples).tolist()))
        self._stats.zero_()
        self._samples = 0
        return result

    def _set_debug_vis_impl(self, debug_vis):
        # Existing COM-force arrows assume a three-dimensional action buffer.
        if debug_vis:
            raise ValueError("Pelvis-wrench visualization is not implemented; use logged wrench metrics")


class ExternalDeltaPelvisWrenchAction(DeltaPelvisWrenchAction):
    """Apply base joint targets together with a frozen policy's pelvis wrench."""

    deployment_export = True

    def __init__(self, cfg, env):
        super().__init__(cfg, env)
        self._scale = self._joint_scale
        if not isinstance(self._scale, torch.Tensor):
            self._scale = torch.full_like(self._joint_position_targets, float(self._scale))

    @property
    def action_dim(self):
        return self._num_joints

    def process_actions(self, actions):
        wrench = getattr(self._env, self.cfg.external_action_buffer_name, None)
        if wrench is None:
            raise RuntimeError(f"Missing frozen wrench buffer {self.cfg.external_action_buffer_name!r}")
        self._process_wrench(wrench.to(self.device))
        self._raw_actions[:] = actions
        self._set_joint_targets(actions)
        self._processed_actions[:] = self._joint_position_targets


@configclass
class DeltaPelvisWrenchActionCfg(DeltaComForceActionCfg):
    class_type: type = DeltaPelvisWrenchAction
    force_body_name: str = "pelvis"
    # Sized for a 10--15 kg pelvis payload. At the upper end, 300 N is about
    # twice the payload weight and leaves comparable authority for acceleration.
    force_scale: float = 300.0
    torque_scale: float = 60.0
    action_clip: float = 1.0
    force_clip: None = None


@configclass
class ExternalDeltaPelvisWrenchActionCfg(DeltaPelvisWrenchActionCfg):
    class_type: type = ExternalDeltaPelvisWrenchAction
    external_action_buffer_name: str = "delta_external_actions"
