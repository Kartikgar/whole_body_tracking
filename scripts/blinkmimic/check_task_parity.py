"""Parity check: the Sprout BlinkMimic task (Isaac Lab manager env) against BlinkMimic's own MDP.

Runs one clip from frame 0 with no randomization or noise, the BlinkMimic EMA policy acting through
``BlinkMimicActorCritic``. Every step it recomputes BlinkMimic's observation history and goal from the same robot
state with BlinkMimic's code (``MDP.obs``, ``MotionLib.goal``) and reports the largest difference to the task's
policy observation, then scores the rollout with BlinkMimic's reward (normalized return, as evaluate_isaac.py).

    python scripts/blinkmimic/check_task_parity.py --headless --clip walk1_subject1
"""

from __future__ import annotations

import argparse
import json
import os
import sys

from isaaclab.app import AppLauncher

parser = argparse.ArgumentParser()
parser.add_argument("--clip", default="walk1_subject1")
parser.add_argument("--frax_dir", default="data/lafan1_frax/sprout_bm40")
parser.add_argument("--motion_dir", default="data/lafan1_frax/sprout_bm40_motion")
parser.add_argument("--ckpt", default="logs/blinkmimic_isaac/sprout_10k_s0/ckpt_010000.pt")
parser.add_argument("--task", default="Tracking-Flat-Sprout-BlinkMimic-v0")
parser.add_argument("--max_steps", type=int, default=None)
AppLauncher.add_app_launcher_args(parser)
args = parser.parse_args()
app = AppLauncher(args).app

import gymnasium as gym  # noqa: E402
import torch  # noqa: E402

import whole_body_tracking.tasks  # noqa: E402,F401
from isaaclab_tasks.utils.parse_cfg import load_cfg_from_registry  # noqa: E402
from whole_body_tracking.blinkmimic.dr import XiLayout  # noqa: E402
from whole_body_tracking.blinkmimic.kinematics import Kinematics  # noqa: E402
from whole_body_tracking.blinkmimic.mdp import MAX_STEP_REWARD, MDP  # noqa: E402
from whole_body_tracking.blinkmimic.motion import MotionLib  # noqa: E402
from whole_body_tracking.blinkmimic.robot import get_robot  # noqa: E402
from whole_body_tracking.blinkmimic.rotations import quat_to_mat  # noqa: E402
from whole_body_tracking.blinkmimic.rsl_actor_critic import BlinkMimicActorCritic  # noqa: E402


def main() -> None:
    dev = "cuda:0"
    env_cfg = load_cfg_from_registry(args.task, "env_cfg_entry_point")
    env_cfg.scene.num_envs = 1
    env_cfg.commands.motion.motion_file = os.path.join(args.motion_dir, f"{args.clip}.npz")
    env_cfg.commands.motion.sample_time_steps = False
    zero = {k: (0.0, 0.0) for k in ("x", "y", "z", "roll", "pitch", "yaw")}
    env_cfg.commands.motion.pose_range = dict(zero)
    env_cfg.commands.motion.velocity_range = dict(zero)
    env_cfg.commands.motion.joint_position_range = (0.0, 0.0)
    for ev in ("physics_material", "add_joint_default_pos", "base_com", "push_robot"):
        setattr(env_cfg.events, ev, None)
    env_cfg.observations.policy.enable_corruption = False
    env_cfg.commands.motion.debug_vis = False

    spec = get_robot("sprout")
    kin = Kinematics(spec, dev)
    motion = MotionLib([os.path.join(args.frax_dir, f"{args.clip}.npz")], kin, dev, spec.load_motion)
    T = motion.num_frames
    env_cfg.episode_length_s = (T + 10) * 0.02
    env = gym.make(args.task, cfg=env_cfg).unwrapped
    robot = env.scene["robot"]
    command = env.command_manager.get_term("motion")
    ids = env.action_manager.get_term("joint_pos")._joint_ids
    assert [robot.joint_names[i] for i in ids] == list(spec.joint_names), "action joint order != BlinkMimic order"

    mdp = MDP(kin, motion, torch.tensor(spec.default_joint_pos_list(), device=dev), XiLayout(spec, dev).dim, spec.sensing)
    xi = torch.zeros(1, XiLayout(spec, dev).dim, device=dev)
    obs, _ = env.reset()
    o = obs["policy"]
    ac = BlinkMimicActorCritic(o.shape[-1], obs["critic"].shape[-1], len(ids), blinkmimic_ckpt=args.ckpt).to(dev).eval()
    H, od = ac.actor.hist, ac.actor.obs_dim

    def state():
        p = robot.data.root_link_pos_w - env.scene.env_origins
        R = quat_to_mat(robot.data.root_link_quat_w)
        w = robot.data.root_com_ang_vel_w
        com_w = (R @ robot.data.com_pos_b[:, 0, :].unsqueeze(-1)).squeeze(-1)  # as env_isaac.TrackingSim.state
        v = robot.data.root_com_lin_vel_w - torch.linalg.cross(w, com_w, dim=-1)
        return p, R, v, w, robot.data.joint_pos[:, ids], robot.data.joint_vel[:, ids]

    hist, a_prev = None, torch.zeros(1, len(ids), device=dev)
    worst = {"hist": 0.0, "goal": 0.0}
    ret, steps, failed = 0.0, 0, False
    n_steps = min(T - 1, args.max_steps or T)
    with torch.no_grad():
        for _ in range(n_steps):
            f = command.time_steps.clone()
            S = state()
            ob = mdp.obs(S, a_prev, xi, f)
            hist = ob.unsqueeze(1).repeat(1, H, 1) if hist is None else torch.cat((hist[:, 1:], ob.unsqueeze(1)), 1)
            goal = motion.goal(f, ac.actor.pi.enc[0].in_features // (15 + 2 * len(ids)))
            worst["hist"] = max(worst["hist"], (o[:, : H * od] - hist.flatten(1)).abs().max().item())
            worst["goal"] = max(worst["goal"], (o[:, H * od :] - goal).abs().max().item())
            a = ac.act_inference(o)
            obs, _, term, trunc, _ = env.step(a)
            o = obs["policy"]
            S1 = state()
            ret += mdp.reward(S1, f + 1, a, a_prev).item()
            a_prev = a.clone()
            steps += 1
            if bool(mdp.termination(S1, f + 1).item()):
                failed = True
                break
    res = {"clip": args.clip, "steps": steps, "clip_frames": T, "failed": failed,
           "norm_return": ret / ((T - 1) * MAX_STEP_REWARD), "max_abs_diff_obs_history": worst["hist"],
           "max_abs_diff_goal": worst["goal"]}  # fmt: skip
    print(json.dumps(res, indent=1), flush=True)


if __name__ == "__main__":
    main()
    sys.stdout.flush()
    os._exit(0)
