#!/usr/bin/env python3
"""Render paired closed-loop rollouts in Isaac: the Isaac robot (solid) with the paired Genesis rollout as a translucent ghost.

Kinematic replay only (no physics stepping): every control frame both articulations are written to the logged state and the
scene is rendered. Inputs:
  --isaac_npz    ``play.py --batch_eval --batch_output_npz`` rollout (post-action arrays + initial_*, env_origins)
  --genesis_npz  ``eval_sim2sim_genesis.py --record_motion`` NPZ of the same envs, started via ``--initial_state_npz``
                 from the Isaac rollout (pre-action arrays + initial_*)
After a rollout terminates, its last state is held (and the tile label says so).
One MP4 per selected env is written to --output_dir; tile them with ffmpeg.

Run from ``whole_body_tracking/`` in the sim2real env, e.g.
  python scripts/Delta_Utils/render_ghost_closed_loop.py --isaac_npz E/isaac/rollout.npz --genesis_npz E/genesis/states.npz \
      --env_ids 0 1 2 3 --output_dir E/videos
"""
import argparse
import json
import subprocess
from pathlib import Path

from isaaclab.app import AppLauncher

parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
parser.add_argument("--isaac_npz", required=True)
parser.add_argument("--genesis_npz", required=True)
parser.add_argument("--env_ids", type=int, nargs="+", required=True)
parser.add_argument("--output_dir", required=True)
parser.add_argument("--width", type=int, default=960)
parser.add_argument("--height", type=int, default=720)
parser.add_argument("--max_frames", type=int, default=None)
parser.add_argument("--ghost_opacity", type=float, default=0.35)
parser.add_argument("--ghost_color", type=float, nargs=3, default=(0.25, 0.55, 1.0))
parser.add_argument("--camera_offset", type=float, nargs=3, default=(1.7, -2.0, 0.55), help="eye - target, metres")
AppLauncher.add_app_launcher_args(parser)
args = parser.parse_args()
args.headless = True
args.enable_cameras = True
app = AppLauncher(args).app

import sys  # noqa: E402

import numpy as np  # noqa: E402
import torch  # noqa: E402

if "numpy._core" not in sys.modules and hasattr(np, "core"):  # Genesis NPZs pickle object arrays under numpy 2
    sys.modules["numpy._core"] = np.core
    sys.modules["numpy._core.multiarray"] = np.core.multiarray

import isaaclab.sim as sim_utils  # noqa: E402
from isaaclab.assets import AssetBaseCfg  # noqa: E402
from isaaclab.scene import InteractiveScene, InteractiveSceneCfg  # noqa: E402
from isaaclab.sensors import CameraCfg  # noqa: E402
from isaaclab.utils import configclass  # noqa: E402
from whole_body_tracking.robots.g1 import G1_CYLINDER_CFG  # noqa: E402


def load_isaac(path):
    """Frames 0..L: frame 0 = initial state, frame t = state after action t-1 (scene origins removed)."""
    d = np.load(path, allow_pickle=True)
    o = d["env_origins"][:, None, :]
    pos = np.concatenate([d["initial_body_pos_w"][:, None] - o[:, None], d["body_pos_w"] - o[:, None]], 1)
    quat = np.concatenate([d["initial_body_quat_w"][:, None], d["body_quat_w"]], 1)
    jp = np.concatenate([d["initial_joint_pos"][:, None], d["joint_pos"]], 1)
    return dict(pos=pos, quat=quat, jp=jp, n=d["valid_lengths"].astype(int) + 1, bodies=list(d["body_names"]),
                joints=list(d["joint_names"]))


def load_genesis(path):
    """Frames 0..L-1 are pre-action states (frame 0 = initial state), so frame t matches Isaac frame t."""
    d = np.load(path, allow_pickle=True)
    return dict(pos=d["body_pos_w"], quat=d["body_quat_w"], jp=d["joint_pos"], n=d["valid_lengths"].astype(int),
                bodies=list(d["body_names"]), joints=list(d["joint_names"]))


I, G = load_isaac(args.isaac_npz), load_genesis(args.genesis_npz)
ids = args.env_ids
K = len(ids)
T = int(max(I["n"][ids].max(), G["n"][ids].max()))
if args.max_frames:
    T = min(T, args.max_frames)


@configclass
class GhostSceneCfg(InteractiveSceneCfg):
    ground = AssetBaseCfg(prim_path="/World/ground", spawn=sim_utils.GroundPlaneCfg())
    light = AssetBaseCfg(prim_path="/World/light", spawn=sim_utils.DomeLightCfg(intensity=2500.0, color=(0.9, 0.9, 0.9)))
    robot = G1_CYLINDER_CFG.replace(prim_path="{ENV_REGEX_NS}/Robot")
    ghost = G1_CYLINDER_CFG.replace(
        prim_path="{ENV_REGEX_NS}/Ghost",
        spawn=G1_CYLINDER_CFG.spawn.replace(
            activate_contact_sensors=False,
            collision_props=sim_utils.CollisionPropertiesCfg(collision_enabled=False),
            articulation_props=sim_utils.ArticulationRootPropertiesCfg(enabled_self_collisions=False),
        ),
    )
    camera = CameraCfg(prim_path="{ENV_REGEX_NS}/Cam", update_period=0, height=args.height, width=args.width,
                       data_types=["rgb"], spawn=sim_utils.PinholeCameraCfg(focal_length=18.0, clipping_range=(0.05, 50.0)))


sim = sim_utils.SimulationContext(sim_utils.SimulationCfg(dt=0.02, render_interval=1, device=args.device))
sim.set_setting("/rtx/translucency/enabled", True)
scene = InteractiveScene(GhostSceneCfg(num_envs=K, env_spacing=50.0))  # far apart: one robot pair per camera


def tint_ghosts():
    """Colour the ghosts. The converted URDF uses instanced meshes, which ignore material overrides, so de-instance the
    ghost subtrees on the stage first (converting with make_instanceable=False made the ghost disappear instead)."""
    import omni.usd
    from pxr import Usd
    stage = omni.usd.get_context().get_stage()
    mat = sim_utils.PreviewSurfaceCfg(diffuse_color=tuple(args.ghost_color), opacity=args.ghost_opacity)
    mat.func("/World/Looks/GhostMaterial", mat)
    for i in range(K):
        root = stage.GetPrimAtPath(f"/World/envs/env_{i}/Ghost")
        while True:  # newly exposed children can be instances themselves
            inst = [p for p in Usd.PrimRange(root) if p.IsInstance()]
            if not inst:
                break
            for p in inst:
                p.SetInstanceable(False)
        sim_utils.bind_visual_material(root.GetPath(), "/World/Looks/GhostMaterial", stage=stage)


tint_ghosts()
sim.reset()
robot, ghost, cam = scene["robot"], scene["ghost"], scene["camera"]
dev = robot.device
origins = scene.env_origins


def to_targets(src, k_frames):
    """Per selected env: root pos/quat (pelvis) and joint positions in articulation order, padded by holding the last frame."""
    pelvis = src["bodies"].index("pelvis")
    jmap = [src["joints"].index(j) for j in robot.data.joint_names]
    root_p = np.zeros((K, T, 3), np.float32); root_q = np.zeros((K, T, 4), np.float32); jp = np.zeros((K, T, len(jmap)), np.float32)
    for i, e in enumerate(ids):
        n = min(int(src["n"][e]), src["pos"].shape[1])
        idx = np.minimum(np.arange(T), n - 1)
        root_p[i] = src["pos"][e, idx, pelvis]; root_q[i] = src["quat"][e, idx, pelvis]; jp[i] = src["jp"][e][idx][:, jmap]
    return [torch.tensor(a, device=dev) for a in (root_p, root_q, jp)]


RI, GI = to_targets(I, T), to_targets(G, T)
zeros_v = torch.zeros(K, 6, device=dev); zeros_j = torch.zeros_like(RI[2][:, 0])
out = Path(args.output_dir); out.mkdir(parents=True, exist_ok=True)
writers = [subprocess.Popen(["ffmpeg", "-loglevel", "error", "-y", "-f", "rawvideo", "-pix_fmt", "rgb24", "-s",
                             f"{args.width}x{args.height}", "-r", "50", "-i", "-", "-c:v", "libx264", "-pix_fmt", "yuv420p",
                             "-crf", "20", str(out / f"env{e:03d}.mp4")], stdin=subprocess.PIPE) for e in ids]
offset = torch.tensor(args.camera_offset, device=dev)
for t in range(T):
    for art, (rp, rq, jp) in ((robot, RI), (ghost, GI)):
        art.write_root_pose_to_sim(torch.cat([rp[:, t] + origins, rq[:, t]], -1))
        art.write_root_velocity_to_sim(zeros_v)
        art.write_joint_state_to_sim(jp[:, t], zeros_j)
    target = RI[0][:, t] + origins
    target[:, 2] = 0.8
    cam.set_world_poses_from_view(target + offset, target)
    sim.render()
    cam.update(0.0, force_recompute=True)
    rgb = cam.data.output["rgb"][..., :3].to(torch.uint8).cpu().numpy()
    for i, w in enumerate(writers):
        w.stdin.write(rgb[i].tobytes())
    if t % 100 == 0:
        print(f"frame {t}/{T}", flush=True)
for w in writers:
    w.stdin.close(); w.wait()
meta = {int(e): dict(isaac_frames=int(I["n"][e]), genesis_frames=int(G["n"][e])) for e in ids}
json.dump(dict(frames=T, envs=meta, args={k: v for k, v in vars(args).items() if isinstance(v, (int, float, str, list, tuple))}),
          open(out / "render_meta.json", "w"), indent=1)
print("done", out, flush=True)
import os  # noqa: E402
os._exit(0)  # Isaac app.close() can hang after rendering; all outputs are already written
