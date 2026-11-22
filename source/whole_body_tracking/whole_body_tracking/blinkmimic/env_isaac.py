"""Motion-tracking simulator on Isaac Lab (PhysX): the Isaac counterpart of BlinkMimic's env.py (mjlab / MuJoCo Warp).

Same interface and bookkeeping as blinkmimic.env.TrackingSim (state, observe, reset, step / step_core / step_tail,
return-based bin sampling, fail-reset, evaluation-probe focus), with the simulator calls replaced:

- one Isaac Lab scene (ground plane + ``robot.robot_cfg()`` articulation) per process, cloned num_envs times;
  4 PhysX substeps of 5 ms per 50 Hz control step, implicit PD on joint position targets;
- state(): root link pose / world-frame linear and angular velocity, joints permuted to RobotSpec (MJCF) order;
- domain randomization (IsaacDomainRandomizer): the BlinkMimic default terms only (friction per env, encoder offsets
  per episode, CoM offset of ``robot.com_body`` per env). The ground uses the bottom of the friction range with the
  "max" combine mode, so the robot's per-env friction decides, as MuJoCo's max-of-pair rule does.

Not ported (MuJoCo-Warp specific): reset_masked / step_tail_masked and CUDA-graph capture of the simulator step.
"""

from __future__ import annotations

import torch

import isaaclab.sim as sim_utils
from isaaclab.assets import Articulation, ArticulationCfg, AssetBaseCfg
from isaaclab.scene import InteractiveScene, InteractiveSceneCfg
from isaaclab.utils import configclass

from .dr import XiLayout, _match
from .mdp import MAX_STEP_REWARD, MDP, flatten_state
from .motion import CONTROL_DT, MotionLib
from .robot import RobotSpec
from .rotations import mat_to_quat, quat_to_mat, so3_exp
from .sensing import ObsPipeline

DECIMATION = 4  # 4 physics substeps of 5 ms per 50 Hz control step
PHYSICS_DT = CONTROL_DT / DECIMATION
ENV_SPACING = 3.0


def _bin_ema(stat: torch.Tensor, bins: torch.Tensor, x: torch.Tensor, old: float) -> torch.Tensor:
    sums = torch.zeros_like(stat).index_add_(0, bins, x)
    cnt = torch.zeros_like(stat).index_add_(0, bins, torch.ones_like(x))
    return torch.where(cnt > 0, old * stat + (1 - old) * sums / cnt.clamp(min=1), stat)


def make_sim_and_scene(robot: RobotSpec, num_envs: int, device: str, ground_friction: float):
    """Create the (single) Isaac simulation context and the cloned robot scene."""
    sim_cfg = sim_utils.SimulationCfg(dt=PHYSICS_DT, render_interval=DECIMATION, device=device)
    sim_cfg.physx.gpu_max_rigid_patch_count = 10 * 2**15
    sim_cfg.physx.gpu_max_rigid_contact_count = 2**24
    sim_cfg.physx.gpu_found_lost_pairs_capacity = 2**23
    sim_cfg.physx.gpu_total_aggregate_pairs_capacity = 2**23
    sim = sim_utils.SimulationContext(sim_cfg)

    robot_cfg: ArticulationCfg = robot.robot_cfg().replace(prim_path="{ENV_REGEX_NS}/Robot")

    @configclass
    class _SceneCfg(InteractiveSceneCfg):
        ground = AssetBaseCfg(
            prim_path="/World/ground",
            spawn=sim_utils.GroundPlaneCfg(
                physics_material=sim_utils.RigidBodyMaterialCfg(
                    static_friction=ground_friction, dynamic_friction=ground_friction, restitution=0.0,
                    friction_combine_mode="max", restitution_combine_mode="min",
                )
            ),
        )
        light = AssetBaseCfg(prim_path="/World/light", spawn=sim_utils.DomeLightCfg(intensity=2000.0))
        robot = robot_cfg

    scene = InteractiveScene(_SceneCfg(num_envs=num_envs, env_spacing=ENV_SPACING, replicate_physics=True))
    sim.reset()
    return sim, scene


class IsaacDomainRandomizer:
    """BlinkMimic's default domain randomization (DomainRandCfg defaults) on an Isaac articulation."""

    def __init__(self, robot: RobotSpec, layout: XiLayout, env: "TrackingSim", eval_mode: bool):
        c = robot.dr
        assert c.com_mass_range is None and c.link_mass_range is None and c.solref_geoms is None, "not ported"
        assert not c.pd_gain_scale and c.armature_range is None and c.damping_range is None, "not ported"
        assert robot.sensing.imu is None and robot.sensing.joint_vel_filter is None, "stateful sensing not ported"
        self.cfg, self.spec, self.layout, self.env, self.eval_mode = c, robot, layout, env, eval_mode
        self.J = robot.num_joints
        self.joint_offset_half = torch.tensor(_match(c.joint_offset, robot.joint_names), device=env.device)
        art: Articulation = env.robot
        self.com_body_id = art.find_bodies(robot.com_body)[0][0]

    def startup(self, xi: torch.Tensor):
        c, art, N, dev = self.cfg, self.env.robot, self.env.N, self.env.device
        L = self.layout
        fr = torch.ones(N, 1, device=dev) if self.eval_mode else torch.empty(N, 1, device=dev).uniform_(*c.friction_range)
        xi[:, L.slices["friction"]] = fr
        view = art.root_physx_view
        sl = self.env.sl
        all_ids = torch.arange(sl.start, sl.stop, device="cpu")  # this env's slice of the (possibly shared) scene
        mats = view.get_material_properties().clone()  # (N_total, num_shapes, 3): static, dynamic, restitution
        mats[sl, :, 0] = fr.cpu()
        mats[sl, :, 1] = fr.cpu()
        mats[sl, :, 2] = 0.0
        view.set_material_properties(mats, all_ids)
        if not self.eval_mode:
            com = torch.stack([torch.empty(N, device=dev).uniform_(lo, hi) for lo, hi in self.spec.com_range], -1)
            xi[:, L.slices["com"]] = com
            coms = view.get_coms().clone()
            coms[sl, self.com_body_id, :3] += com.cpu()
            view.set_coms(coms, all_ids)

    def reset(self, xi: torch.Tensor, env_ids: torch.Tensor):
        if self.eval_mode:
            return
        n = len(env_ids)
        xi[env_ids, self.layout.slices["joint_offset"]] = (
            2 * torch.rand(n, self.J, device=xi.device) - 1
        ) * self.joint_offset_half


class TrackingSim:
    """Batched motion-tracking simulator for ``robot`` on Isaac Lab."""

    def __init__(self, num_envs: int, robot: RobotSpec, motion: MotionLib, mdp: MDP, device: str,
                 gamma: float = 0.99, eval_mode: bool = False, max_episode_steps: int = 500, reset_noise: bool = True,
                 randomize: bool | None = None, fail_reset: float = 0.0, bin_fix: bool = False,
                 shared: tuple | None = None, env_offset: int = 0):  # fmt: skip
        """``shared`` = (sim, scene) from make_sim_and_scene: this simulator owns envs [env_offset, env_offset +
        num_envs) of that scene (e.g. training worlds and evaluation-probe worlds in one process). Every PhysX step
        advances all envs of the scene; see snapshot() / restore()."""
        self.device = device
        self.eval_mode = eval_mode
        self.randomize = (not eval_mode) if randomize is None else randomize
        if shared is None:
            ground_friction = robot.dr.friction_range[0] if self.randomize else 1.0
            shared = make_sim_and_scene(robot, num_envs, device, ground_friction)
        self.sim, self.scene = shared
        self.robot: Articulation = self.scene["robot"]
        self.N = num_envs
        self.off = env_offset
        self.sl = slice(env_offset, env_offset + num_envs)
        assert self.sl.stop <= self.scene.num_envs, (self.sl, self.scene.num_envs)
        self.gall = torch.arange(env_offset, env_offset + num_envs, device=device)  # global ids of this slice
        self.J = robot.num_joints
        self.spec = robot
        self.motion, self.mdp, self.gamma = motion, mdp, gamma
        dev = device
        # joint order: spec (MJCF) <-> Isaac
        isaac_names = list(self.robot.joint_names)
        assert sorted(isaac_names) == sorted(robot.joint_names), (isaac_names, robot.joint_names)
        self.to_spec = torch.tensor([isaac_names.index(n) for n in robot.joint_names], device=dev)  # isaac[:, to_spec]
        self.to_isaac = torch.tensor([list(robot.joint_names).index(n) for n in isaac_names], device=dev)
        self.env_origins = self.scene.env_origins[self.sl]
        self.max_episode_steps = max_episode_steps if not eval_mode else 10**9
        self.reset_noise = reset_noise and not eval_mode
        self.default_q = torch.tensor(robot.default_joint_pos_list(), device=dev)
        self.alpha = torch.tensor(robot.action_scale_list(), device=dev)

        self.layout = XiLayout(robot, dev)
        assert self.layout.dim == mdp.xi_dim, (self.layout.dim, mdp.xi_dim)
        self.xi = torch.zeros(self.N, self.layout.dim, device=dev)
        self.randomizer = IsaacDomainRandomizer(robot, self.layout, self, eval_mode=not self.randomize)
        self.randomizer.startup(self.xi)

        self.pipe = ObsPipeline(robot, mdp, self.layout, dev, PHYSICS_DT)
        assert not self.pipe.stateful, "stateful sensing not ported"
        self._obs = None
        self.obs_fresh = torch.ones(self.N, dtype=torch.bool, device=dev)
        self.push_enabled = False
        assert robot.dr.push_interval_s is None, "pushes not ported"

        self.f = torch.zeros(self.N, dtype=torch.long, device=dev)
        self.ep_step = torch.zeros(self.N, dtype=torch.long, device=dev)
        self.a_prev = torch.zeros(self.N, self.J, device=dev)
        self.start_bin = torch.zeros(self.N, dtype=torch.long, device=dev)
        self.ep_disc_ret = torch.zeros(self.N, device=dev)
        self.ep_ret = torch.zeros(self.N, device=dev)
        self.G_max = MAX_STEP_REWARD / (1 - gamma)
        self.bin_G = torch.full((motion.num_bins,), 0.0, device=dev)
        self.bin_w_avg = torch.full((motion.num_bins,), 1.0 / motion.num_bins, device=dev)
        self.fail_reset = fail_reset
        self.bin_fix = bin_fix
        self.focus_q, self.focus_share = None, 0.0
        self.focus_static, self.focus_share_t = False, None
        self.bin_fail = torch.zeros(motion.num_bins, device=dev)
        prev = torch.arange(motion.num_bins, device=dev) - 1
        self.bin_prev = torch.where((prev >= 0) & (motion.bin_clip[prev.clamp(min=0)] == motion.bin_clip), prev, prev + 1)
        self.done_eval = torch.zeros(self.N, dtype=torch.bool, device=dev)
        self.failed_eval = torch.zeros(self.N, dtype=torch.bool, device=dev)
        self.reset(torch.arange(self.N, device=dev))

    @property
    def num_envs(self) -> int:
        return self.N

    # ------------------------------------------------------------------ state
    def state(self):
        d, sl = self.robot.data, self.sl
        p = d.root_link_pos_w[sl] - self.env_origins
        R = quat_to_mat(d.root_link_quat_w[sl])  # wxyz
        # Root link velocity from the CoM velocity: Isaac Lab 2.1's write_root_link_velocity_to_sim updates only the
        # cached CoM velocity, so root_link_vel_w is stale right after a reset (0.2-0.6 error on Sprout, whose torso
        # CoM is 0.18 m off the link origin). v_link = v_com - w x (R com_b); w is the same for both.
        w = d.root_com_ang_vel_w[sl].clone()  # world frame (BlinkMimic convention)
        com_w = (R @ d.com_pos_b[sl, 0, :].unsqueeze(-1)).squeeze(-1)
        v = d.root_com_lin_vel_w[sl] - torch.linalg.cross(w, com_w, dim=-1)  # world-frame velocity of the root link origin
        return (p, R, v, w, d.joint_pos[sl][:, self.to_spec].clone(), d.joint_vel[sl][:, self.to_spec].clone())

    def body_pos_w(self, body_names: list[str]) -> torch.Tensor:
        """World positions (env-local) of named bodies, for FK checks."""
        ids = [self.robot.body_names.index(b) for b in body_names]
        return self.robot.data.body_link_pos_w[self.sl][:, ids] - self.env_origins.unsqueeze(1)

    # ------------------------------------------------------------------ reset
    def sample_frames(self, n: int, gate: torch.Tensor | None = None):
        m = self.motion
        soft = torch.softmax(-25.0 * self.bin_G / self.G_max, 0)
        self.bin_w_avg.mul_(0.999).add_(0.001 * soft)
        probs = 0.5 / m.num_bins + 0.4 * soft + 0.1 * self.bin_w_avg / self.bin_w_avg.sum()
        if self.fail_reset > 0:
            fail = torch.zeros_like(self.bin_fail).index_add_(0, self.bin_prev, self.bin_fail)
            fail = fail / fail.sum().clamp(min=1e-6)
            probs = (1 - self.fail_reset) * probs + self.fail_reset * torch.where(fail.sum() > 0, fail, probs)
        if self.focus_q is not None and self.focus_share > 0:
            probs = (1 - self.focus_share) * probs + self.focus_share * self.focus_q
        bins = torch.multinomial(probs, n, replacement=True)
        start = m.bin_start[bins]
        end = torch.minimum(start + 50, m.clip_end[m.bin_clip[bins]])
        f = start + (torch.rand(n, device=self.device) * (end - start).float()).long()
        return f, bins

    def reset(self, env_ids: torch.Tensor, frames: torch.Tensor | None = None):
        if len(env_ids) == 0:
            return
        n = len(env_ids)
        dev = self.device
        if frames is None:
            frames, bins = self.sample_frames(n)
        else:
            bins = self.motion.frame_bin[frames]
        m, c = self.motion, self.spec.dr
        p, R, v, w, q, qd = (x.clone() for x in m.ref_state(frames))
        if self.reset_noise:
            u = lambda half: torch.stack([torch.empty(n, device=dev).uniform_(-h, h) for h in half], -1)  # noqa: E731
            p = p + u(c.reset_pos)
            R = so3_exp(u(c.reset_rot)) @ R
            v = v + u(c.reset_lin_vel)
            w = w + u(c.reset_ang_vel)
            q = q + torch.empty_like(q).uniform_(-c.reset_joint_pos, c.reset_joint_pos)
            if c.reset_joint_vel > 0:
                qd = qd + torch.empty_like(qd).uniform_(-c.reset_joint_vel, c.reset_joint_vel)
        q = torch.clamp(q, self.mdp.kin.q_lo, self.mdp.kin.q_hi)
        art = self.robot
        gids = env_ids + self.off
        art.reset(gids)
        pose = torch.cat((p + self.env_origins[env_ids], mat_to_quat(R)), -1)
        art.write_root_link_pose_to_sim(pose, env_ids=gids)
        art.write_root_link_velocity_to_sim(torch.cat((v, w), -1), env_ids=gids)
        q_i, qd_i = q[:, self.to_isaac], qd[:, self.to_isaac]
        art.write_joint_state_to_sim(q_i, qd_i, env_ids=gids)
        art.set_joint_position_target(q_i, env_ids=gids)
        self.f[env_ids] = frames
        self.ep_step.index_fill_(0, env_ids, 0)
        self.a_prev.index_fill_(0, env_ids, 0.0)
        self.start_bin[env_ids] = bins
        self.ep_disc_ret.index_fill_(0, env_ids, 0.0)
        self.ep_ret.index_fill_(0, env_ids, 0.0)
        self.randomizer.reset(self.xi, env_ids)
        self.obs_fresh.index_fill_(0, env_ids, True)

    def snapshot(self) -> dict:
        """Exact physical state of this slice (root link pose, CoM velocity, joints, PD targets)."""
        d, sl = self.robot.data, self.sl
        return {"pose": torch.cat((d.root_link_pos_w[sl], d.root_link_quat_w[sl]), -1).clone(),
                "com_vel": torch.cat((d.root_com_lin_vel_w[sl], d.root_com_ang_vel_w[sl]), -1).clone(),
                "q": d.joint_pos[sl].clone(), "qd": d.joint_vel[sl].clone(),
                "target": d.joint_pos_target[sl].clone()}  # fmt: skip

    def restore(self, snap: dict):
        """Write a snapshot() back (e.g. after another slice of the shared scene was stepped)."""
        art, g = self.robot, self.gall
        art.write_root_link_pose_to_sim(snap["pose"], env_ids=g)
        art.write_root_com_velocity_to_sim(snap["com_vel"], env_ids=g)
        art.write_joint_state_to_sim(snap["q"], snap["qd"], env_ids=g)
        art.set_joint_position_target(snap["target"], env_ids=g)

    def use_static_focus(self):
        raise NotImplementedError("reset_masked / static focus is MuJoCo-Warp specific")

    def set_focus(self, q: torch.Tensor, share: float):
        self.focus_q, self.focus_share = q, share

    def refresh(self):
        """Recompute body poses from the written root/joint state (kinematics only, no dynamics step)."""
        self.sim.forward()
        self.robot.update(0.0)

    # ------------------------------------------------------------------ observation
    def observe(self) -> torch.Tensor:
        S = self.state()
        o = self.mdp.obs(S, self.a_prev, self.xi, self.f)
        self._obs = self.mdp.add_noise(o) if self.randomize else o
        return self._obs

    # ------------------------------------------------------------------ step
    def step(self, a: torch.Tensor) -> dict:
        out = self.step_core(a)
        if self.eval_mode:
            self.failed_eval |= out["term"] & ~self.done_eval
            self.done_eval |= out["done"]
            return out
        return self.step_tail(out)

    def step_core(self, a: torch.Tensor) -> dict:
        self._obs = None
        S = self.state()
        f = self.f.clone()
        xi = self.xi.clone()
        a_prev = self.a_prev.clone()
        ep_step = self.ep_step.clone()
        target = self.default_q + xi[:, 1 : 1 + self.J] + self.alpha * a
        self.robot.set_joint_position_target(target[:, self.to_isaac], env_ids=self.gall)
        for _ in range(DECIMATION):
            self.robot.write_data_to_sim()
            self.sim.step(render=False)
            self.robot.update(PHYSICS_DT)
        S1 = self.state()
        f1 = f + 1
        r, terms = self.mdp.reward(S1, f1, a, a_prev, terms=True)
        term = self.mdp.termination(S1, f1)
        clip_end = self.motion.clip_end[self.motion.clip_id[f]]
        timeout = (f1 >= clip_end) | (ep_step + 1 >= self.max_episode_steps)
        done = term | timeout
        out = {
            "s": flatten_state(S), "a": a.clone(), "a_prev": a_prev, "f": f, "s1": flatten_state(S1), "r": r,
            "term": term, "done": done, "ep_step": ep_step, "xi": xi, "terms": terms, "cut": done,
        }  # fmt: skip
        self.ep_disc_ret += (self.gamma ** ep_step.float()) * r
        self.ep_ret += r
        self.f = f1
        self.ep_step = ep_step + 1
        self.a_prev = a.clone()
        return out

    def step_tail(self, out: dict) -> dict:
        done, term, f = out["done"], out["term"], out["f"]
        ids = done.nonzero(as_tuple=False).squeeze(-1)
        if len(ids) > 0:
            T = self.ep_step[ids].float()
            boot = torch.where(term[ids], 0.0, (self.gamma**T) * (self.ep_ret[ids] / T) / (1 - self.gamma))
            G = self.ep_disc_ret[ids] + boot
            b = self.start_bin[ids]
            if self.bin_fix:
                self.bin_G.copy_(_bin_ema(self.bin_G, b, G, 0.8))
            else:
                self.bin_G.index_put_((b,), 0.8 * self.bin_G[b] + 0.2 * G)
            if self.fail_reset > 0:
                fb = self.motion.frame_bin[f[ids].clamp(max=self.motion.num_frames - 1)]
                self.bin_fail.mul_(0.999)
                if self.bin_fix:
                    self.bin_fail.copy_(_bin_ema(self.bin_fail, fb, term[ids].float(), 0.9))
                else:
                    self.bin_fail.index_put_((fb,), 0.9 * self.bin_fail[fb] + 0.1 * term[ids].float())
            out["ep_len"] = T
            out["ep_ret_norm"] = self.ep_ret[ids] / (T * MAX_STEP_REWARD)
            self.reset(ids)
        return out
