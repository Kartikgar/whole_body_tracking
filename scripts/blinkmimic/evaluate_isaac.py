"""Checkpoint evaluation on Isaac Lab: the protocol of BlinkMimic's evaluate.py (SuperMimic paper Appendix F.1).

Each motion plays once (per repeat) from its first frame with friction 1.0, no domain randomization, no observation or
action noise, with the deployed (EMA) policy. A motion fails if the rollout triggers the training termination.
Normalized return = undiscounted reward sum / maximum attainable. Results (incl. per-clip returns) are printed and
written to <ckpt>_eval_isaac_x<repeats>.json. --record saves trajectories in the format of blinkmimic's tools/render.py.

    python scripts/blinkmimic/evaluate_isaac.py --headless --ckpt logs/blinkmimic_isaac/<run>/ckpt_010000.pt
"""

from __future__ import annotations

import argparse
import json
import os
import sys

from isaaclab.app import AppLauncher

parser = argparse.ArgumentParser()
parser.add_argument("--robot", type=str, default="sprout")
parser.add_argument("--ckpt", type=str, required=True)
parser.add_argument("--motion_dir", type=str, default=None)
parser.add_argument("--record", type=str, default=None, help="save trajectories (npz) for rendering")
parser.add_argument("--repeats", type=int, default=1, help="rollouts per clip")
parser.add_argument("--seed", type=int, default=0)
AppLauncher.add_app_launcher_args(parser)
args = parser.parse_args()
app = AppLauncher(args).app

import numpy as np  # noqa: E402
import torch  # noqa: E402

import whole_body_tracking.blinkmimic.networks as networks  # noqa: E402
from whole_body_tracking.blinkmimic.dr import XiLayout  # noqa: E402
from whole_body_tracking.blinkmimic.env_isaac import TrackingSim  # noqa: E402
from whole_body_tracking.blinkmimic.kinematics import Kinematics  # noqa: E402
from whole_body_tracking.blinkmimic.mdp import MAX_STEP_REWARD, MDP  # noqa: E402
from whole_body_tracking.blinkmimic.motion import MotionLib  # noqa: E402
from whole_body_tracking.blinkmimic.networks import Policy, policy_kwargs  # noqa: E402
from whole_body_tracking.blinkmimic.robot import get_robot  # noqa: E402
from whole_body_tracking.blinkmimic.rotations import mat_to_quat  # noqa: E402

torch.backends.cuda.matmul.allow_tf32 = True
torch.manual_seed(args.seed)
dev = "cuda:0"
robot = get_robot(args.robot)
kin = Kinematics(robot, dev)
motion = MotionLib.from_dir(args.motion_dir or robot.motion_dir, robot, kin, dev)
mdp = MDP(kin, motion, torch.tensor(robot.default_joint_pos_list(), device=dev), XiLayout(robot, dev).dim, robot.sensing)
C = len(motion.names)
N = C * args.repeats
clip_of = torch.arange(N, device=dev) % C
env = TrackingSim(N, robot, motion, mdp, dev, eval_mode=True)

ck = torch.load(args.ckpt, map_location=dev, weights_only=False)
a_ = ck["args"]
K, Hh = a_["goal_frames"], a_["hist"]
networks.AMP_DTYPE = torch.bfloat16 if a_.get("amp", "none") == "bf16" else None
widths = lambda key, default: tuple(int(x) for x in a_.get(key, default).split(","))  # noqa: E731
pi = Policy(mdp.obs_dim, motion.goal_dim(K), robot.num_joints, hist=Hh, enc_hidden=widths("enc_hidden", "2048,1024,512,256"),
            dec_hidden=widths("dec_hidden", "2048,1024,512,256"), **policy_kwargs(a_, robot)).to(dev)  # fmt: skip
pi.load_state_dict(ck["pi"])
pi.eval()

with torch.no_grad():
    ids = torch.arange(N, device=dev)
    env.done_eval[:] = False
    env.failed_eval[:] = False
    env.reset(ids, motion.clip_start[clip_of].clone())
    clip_end = motion.clip_end[clip_of]
    ret = torch.zeros(N, device=dev)
    hist = None
    traj = {"root_p": [], "root_quat": [], "q": [], "f": [], "alive": []}
    while not bool(env.done_eval.all()):
        S = env.state()
        o = env.observe()
        hist = o.unsqueeze(1).repeat(1, Hh, 1) if hist is None else torch.cat((hist[:, 1:], o.unsqueeze(1)), 1)
        a = pi(hist, motion.goal(env.f, K))
        alive = ~env.done_eval.clone()
        if args.record:
            traj["root_p"].append(S[0].cpu()), traj["root_quat"].append(mat_to_quat(S[1]).cpu())
            traj["q"].append(S[4].cpu()), traj["f"].append(env.f.cpu()), traj["alive"].append(alive.cpu())
        out = env.step(a)
        ret += torch.where(alive, out["r"], 0.0)
        env.f = torch.minimum(env.f, clip_end - 1)  # finished clips keep stepping; keep f+1 in range

clip_len = motion.clip_len[clip_of]
norm = ret / ((clip_len - 1).float() * MAX_STEP_REWARD)
norm_c = norm.view(args.repeats, C).mean(0)
fail_c = env.failed_eval.view(args.repeats, C).float().mean(0)
res = {
    "ckpt": args.ckpt,
    "iteration": ck.get("iteration"),
    "norm_return": norm.mean().item(),
    "norm_return_framew": (ret.sum() / ((clip_len - 1).sum() * MAX_STEP_REWARD)).item(),
    "n_failed": int(env.failed_eval.sum().item()),
    "fail_rate": env.failed_eval.float().mean().item(),
    "failed": [motion.names[i] for i in (fail_c > 0).nonzero().flatten().tolist()],
    "per_clip": {motion.names[i]: round(norm_c[i].item(), 4) for i in range(C)},
    "per_clip_fail_rate": {motion.names[i]: round(fail_c[i].item(), 3) for i in range(C) if fail_c[i] > 0},
}
print(json.dumps({k: v for k, v in res.items() if k != "per_clip"}, indent=1), flush=True)
json.dump(res, open(args.ckpt.replace(".pt", f"_eval_isaac_x{args.repeats}.json"), "w"), indent=1)
if args.record:
    np.savez_compressed(args.record, names=np.array(motion.names), **{k: torch.stack(v).numpy() for k, v in traj.items()},
                        ref_root_p=motion.root_p.cpu().numpy(), ref_root_R=motion.root_R.cpu().numpy(), ref_q=motion.q.cpu().numpy(),
                        clip_start=motion.clip_start.cpu().numpy(), clip_len=motion.clip_len.cpu().numpy())  # fmt: skip
sys.stdout.flush()
os._exit(0)  # Kit's app.close() can hang after a standalone run
