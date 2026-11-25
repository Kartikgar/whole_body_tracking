"""Arrow visualization of the applied (gated) pelvis wrench in a live Isaac batch eval, plus a per-step log.

Reads the wrench the action term actually applies this step (``ExternalDeltaPelvisWrenchAction._wrench``: pelvis body
frame, at the pelvis COM, N and N*m, already multiplied by the gate), rotates it to world and draws, per env:
  force  arrow: from the pelvis along F, length = |F| * force_len_per_N          (orange)
  torque arrow: from the pelvis along tau (right-hand rule), length = |tau| * torque_len_per_Nm (purple)
All envs are updated in one batched ``VisualizationMarkers.visualize`` call per control step, right before rendering.
The log (``wrench_log.npz``: body/world force and torque, gate, uncertainty per step and env) is rewritten every 50 steps.
"""
import numpy as np
import torch

import isaaclab.sim as sim_utils
from isaaclab.markers import VisualizationMarkers, VisualizationMarkersCfg
from isaaclab.utils.math import quat_apply

FORCE_COLOR = (1.0, 0.45, 0.05)
TORQUE_COLOR = (0.6, 0.2, 0.9)


def _x_to_dir_quat(d):
    """wxyz quaternion rotating +X onto unit vector d (batched); antiparallel case -> 180 deg about Z."""
    q = torch.stack([1.0 + d[:, 0], torch.zeros_like(d[:, 0]), -d[:, 2], d[:, 1]], -1)
    n = q.norm(dim=-1, keepdim=True)
    flip = n.squeeze(-1) < 1e-6
    q = q / n.clamp_min(1e-9)
    if flip.any():
        q[flip] = torch.tensor([0.0, 0.0, 0.0, 1.0], device=d.device)
    return q


def make_wrench_arrow_writer(base_env, ppo_runner, log_path, force_len_per_N=0.004, torque_len_per_Nm=0.02,
                             shaft_radius=0.012, head_len=0.07, head_radius=0.03):
    term = base_env.action_manager.get_term("joint_pos")
    if not hasattr(term, "_wrench"):
        raise RuntimeError("--wrench_arrows needs a pelvis-wrench action term (e.g. Tracking-Flat-G1-DeltaWrench-Finetune-v0)")
    robot = base_env.scene["robot"]
    body = robot.find_bodies("pelvis")[0][0]
    mk = lambda c: sim_utils.PreviewSurfaceCfg(diffuse_color=c, emissive_color=tuple(0.35 * x for x in c))
    markers = VisualizationMarkers(VisualizationMarkersCfg(prim_path="/Visuals/PelvisWrench", markers={
        "force_shaft": sim_utils.CylinderCfg(radius=1.0, height=1.0, axis="X", visual_material=mk(FORCE_COLOR)),
        "force_head": sim_utils.ConeCfg(radius=1.0, height=1.0, axis="X", visual_material=mk(FORCE_COLOR)),
        "torque_shaft": sim_utils.CylinderCfg(radius=1.0, height=1.0, axis="X", visual_material=mk(TORQUE_COLOR)),
        "torque_head": sim_utils.ConeCfg(radius=1.0, height=1.0, axis="X", visual_material=mk(TORQUE_COLOR)),
    }))
    n, dev = base_env.num_envs, robot.device
    idx = torch.arange(4, device=dev).repeat(n)  # per env: force shaft, force head, torque shaft, torque head
    log = {k: [] for k in ("force_body", "torque_body", "force_world", "torque_world", "gate", "uncertainty")}

    def arrow(origin, vec, per_unit):
        mag = vec.norm(dim=-1, keepdim=True)
        d = vec / mag.clamp_min(1e-9)
        length = mag * per_unit
        visible = (length > 1e-3).float()
        q = _x_to_dir_quat(torch.where(mag > 1e-9, d, torch.tensor([1.0, 0.0, 0.0], device=dev)))
        shaft_pos = origin + d * length / 2
        head_pos = origin + d * (length + head_len / 2)
        tiny = 1e-4
        shaft_scale = torch.cat([length.clamp_min(tiny), (shaft_radius * visible).clamp_min(tiny).expand(-1, 2)], -1)
        head_scale = torch.cat([(head_len * visible).clamp_min(tiny), (head_radius * visible).clamp_min(tiny).expand(-1, 2)], -1)
        return (shaft_pos, q, shaft_scale), (head_pos, q, head_scale)

    @torch.inference_mode()
    def update(step):
        w = term._wrench
        quat = robot.data.body_quat_w[:, body]
        origin = robot.data.body_pos_w[:, body]
        f_w, t_w = quat_apply(quat, w[:, :3]), quat_apply(quat, w[:, 3:])
        parts = [*arrow(origin, f_w, force_len_per_N), *arrow(origin, t_w, torque_len_per_Nm)]
        pos = torch.stack([p[0] for p in parts], 1).reshape(-1, 3)
        rot = torch.stack([p[1] for p in parts], 1).reshape(-1, 4)
        scl = torch.stack([p[2] for p in parts], 1).reshape(-1, 3)
        markers.visualize(translations=pos, orientations=rot, scales=scl, marker_indices=idx)
        gate = getattr(ppo_runner, "delta_policy_last_gate", None)
        unc = getattr(ppo_runner, "delta_policy_last_uncertainty", None)
        for k, v in (("force_body", w[:, :3]), ("torque_body", w[:, 3:]), ("force_world", f_w), ("torque_world", t_w),
                     ("gate", gate), ("uncertainty", unc)):
            log[k].append(np.full(n, np.nan, np.float32) if v is None else v.detach().float().reshape(n, -1).squeeze(-1).cpu().numpy())
        if step % 50 == 49:
            np.savez(log_path, **{k: np.stack(v, 1) for k, v in log.items()}, fps=50.0,
                     frame="force/torque *_body: pelvis body frame; *_world: world frame; applied (gated) wrench")

    return update
