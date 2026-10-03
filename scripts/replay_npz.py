"""Replay an NPZ motion in Isaac Sim, optionally recording one pass to MP4.

.. code-block:: bash

    python scripts/replay_npz.py --motion_file path/to/motion.npz
    python scripts/replay_npz.py --motion_file path/to/motion.npz --video
"""

"""Launch Isaac Sim Simulator first."""

import argparse
import os
import sys
import traceback
from contextlib import ExitStack

import numpy as np
import torch

from isaaclab.app import AppLauncher

# add argparse arguments
parser = argparse.ArgumentParser(description="Replay converted motions.")
parser.add_argument("--registry_name", type=str, default=None, help="The name of the wandb registry artifact.")
parser.add_argument("--motion_file", type=str, default=None, help="Path to a local motion .npz file.")
parser.add_argument("--video", action="store_true", help="Record one playback of the first trajectory to an MP4.")
parser.add_argument(
    "--video_file",
    type=str,
    default=None,
    help="Output MP4 path (default: beside the motion NPZ). Implies --video.",
)

# append AppLauncher cli args
AppLauncher.add_app_launcher_args(parser)
# parse the arguments
args_cli = parser.parse_args()
video_enabled = args_cli.video or args_cli.video_file is not None
if video_enabled:
    args_cli.enable_cameras = True

# launch omniverse app
app_launcher = AppLauncher(args_cli)
simulation_app = app_launcher.app

"""Rest everything follows."""

import isaaclab.sim as sim_utils
from isaaclab.assets import Articulation, ArticulationCfg, AssetBaseCfg
from isaaclab.scene import InteractiveScene, InteractiveSceneCfg
from isaaclab.sim import SimulationContext
from isaaclab.utils import configclass
from isaaclab.utils.assets import ISAAC_NUCLEUS_DIR

##
# Pre-defined configs
##
from whole_body_tracking.robots.g1 import G1_CYLINDER_CFG
from whole_body_tracking.tasks.tracking.mdp.commands import MotionLoader


@configclass
class ReplayMotionsSceneCfg(InteractiveSceneCfg):
    """Configuration for a replay motions scene."""

    ground = AssetBaseCfg(prim_path="/World/defaultGroundPlane", spawn=sim_utils.GroundPlaneCfg())

    sky_light = AssetBaseCfg(
        prim_path="/World/skyLight",
        spawn=sim_utils.DomeLightCfg(
            intensity=750.0,
            texture_file=f"{ISAAC_NUCLEUS_DIR}/Materials/Textures/Skies/PolyHaven/kloofendal_43d_clear_puresky_4k.hdr",
        ),
    )

    # articulation
    robot: ArticulationCfg = G1_CYLINDER_CFG.replace(prim_path="{ENV_REGEX_NS}/Robot")


def run_simulator(sim: sim_utils.SimulationContext, scene: InteractiveScene):
    # Extract scene entities
    robot: Articulation = scene["robot"]
    # Define simulation stepping
    sim_dt = sim.get_physics_dt()

    if args_cli.motion_file is not None:
        motion_file = os.path.abspath(os.path.expanduser(args_cli.motion_file))
        if not os.path.isfile(motion_file):
            raise FileNotFoundError(f"Motion file not found: {motion_file}")
        print(f"[INFO]: Using local motion file: {motion_file}")
    else:
        if args_cli.registry_name is None:
            raise ValueError("Provide --motion_file, or provide --registry_name to fetch motion.npz from wandb.")
        registry_name = args_cli.registry_name
        if ":" not in registry_name:  # Check if the registry name includes alias, if not, append ":latest"
            registry_name += ":latest"
        import pathlib

        import wandb

        print(f"[INFO]: Downloading motion artifact from wandb: {registry_name}")
        api = wandb.Api()
        artifact = api.artifact(registry_name)
        motion_file = str(pathlib.Path(artifact.download()) / "motion.npz")

    motion = MotionLoader(
        motion_file,
        [0],
        sim.device,
    )
    motion_length = int(motion.trajectory_time_step_total[0].item())
    trajectory_ids = torch.zeros(scene.num_envs, dtype=torch.long, device=sim.device)

    with ExitStack() as stack:
        if video_enabled:
            import omni.replicator.core as rep

            try:
                import imageio.v2 as imageio
            except ImportError as exc:
                raise RuntimeError("MP4 encoding requires imageio with ffmpeg support (imageio[ffmpeg]).") from exc
            video_fps = float(motion.fps[0].item())
            if video_fps <= 0:
                video_fps = 1.0 / sim_dt
                print(f"[WARN]: Motion has no FPS metadata; recording at {video_fps:g} fps.")
            output_video = os.path.abspath(os.path.expanduser(args_cli.video_file or os.path.splitext(motion_file)[0] + ".mp4"))
            if os.path.exists(output_video):
                raise FileExistsError(f"Video already exists: {output_video}")
            os.makedirs(os.path.dirname(output_video), exist_ok=True)
            render_product = rep.create.render_product("/OmniverseKit_Persp", (1280, 720))
            rgb_annotator = rep.AnnotatorRegistry.get_annotator("rgb", device="cpu")
            rgb_annotator.attach([render_product])
            writer = stack.enter_context(imageio.get_writer(output_video, fps=video_fps, codec="libx264"))
            print(f"[INFO]: Recording {motion_length} frames to: {output_video}", flush=True)

        frame_index = 0
        while simulation_app.is_running() and (not video_enabled or frame_index < motion_length):
            time_steps = torch.full(
                (scene.num_envs,), frame_index % motion_length, dtype=torch.long, device=sim.device
            )
            root_states = robot.data.default_root_state.clone()
            root_states[:, :3] = motion.get_body_pos_w(trajectory_ids, time_steps)[:, 0] + scene.env_origins
            root_states[:, 3:7] = motion.get_body_quat_w(trajectory_ids, time_steps)[:, 0]
            root_states[:, 7:10] = motion.get_body_lin_vel_w(trajectory_ids, time_steps)[:, 0]
            root_states[:, 10:] = motion.get_body_ang_vel_w(trajectory_ids, time_steps)[:, 0]

            robot.write_root_state_to_sim(root_states)
            robot.write_joint_state_to_sim(
                motion.get_joint_pos(trajectory_ids, time_steps),
                motion.get_joint_vel(trajectory_ids, time_steps),
            )
            scene.write_data_to_sim()

            pos_lookat = root_states[0, :3].cpu().numpy()
            eye = (float(pos_lookat[0] + 2.0), float(pos_lookat[1] + 2.0), float(pos_lookat[2] + 0.5))
            target = (float(pos_lookat[0]), float(pos_lookat[1]), float(pos_lookat[2]))
            sim.set_camera_view(eye, target)
            sim.render()  # Replay stored states without stepping physics.
            scene.update(sim_dt)

            if video_enabled:
                rgb_data = rgb_annotator.get_data()
                frame = np.frombuffer(rgb_data, dtype=np.uint8).reshape(*rgb_data.shape)
                if frame.size == 0:
                    raise RuntimeError(f"RGB capture returned an empty frame at index {frame_index}.")
                writer.append_data(np.ascontiguousarray(frame[:, :, :3]))
            frame_index += 1

    if video_enabled:
        print(f"[INFO]: Saved motion video to: {output_video} ({frame_index} frames)", flush=True)


def main():
    sim_cfg = sim_utils.SimulationCfg(device=args_cli.device)
    sim_cfg.dt = 0.02
    sim = SimulationContext(sim_cfg)

    scene_cfg = ReplayMotionsSceneCfg(num_envs=1, env_spacing=2.0)
    scene = InteractiveScene(scene_cfg)
    sim.reset()
    # Run the simulator
    run_simulator(sim, scene)


if __name__ == "__main__":
    if video_enabled:
        try:
            main()
        except BaseException:
            traceback.print_exc()
            sys.stderr.flush()
            os._exit(1)
        # Video runs use separate processes. Isaac Sim can hang during shutdown
        # after the MP4 writer closes, so release it with the process.
        os._exit(0)
    try:
        main()
    finally:
        simulation_app.close()
