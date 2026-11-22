"""Play retargeted Sprout clips in a viser viewer (frax env).

    python scripts/sprout/retarget/view_sprout.py [--results-dir data/lafan1_frax/sprout] [--port 8080]

Open http://localhost:<port>. Optional: --source shows the scaled LAFAN1 targets as dots.
"""

from __future__ import annotations

import argparse
import sys
import time
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parent))
import sprout_frax as S  # noqa: E402

import viser  # noqa: E402
from viser.extras import ViserUrdf  # noqa: E402


def load_textured_urdf(flat_color: bool):
    """Sprout URDF with its visual meshes' texture attached.

    Fauna's visual .mtl files say `map_Kd robot_D.png` (relative to meshes/visual/), but the
    texture lives in textures/, and their fallback diffuse colour is black (Kd 0 0 0). Loaded
    as-is, every link renders solid black. Attach textures/robot_D.png to each mesh (all have
    UVs), or give them a flat grey with --flat-color.
    """
    from functools import partial

    import trimesh
    import yourdfpy
    from PIL import Image

    path = Path(S.SPROUT_URDF)
    urdf = yourdfpy.URDF.load(
        path, build_scene_graph=True, load_meshes=True,
        filename_handler=partial(yourdfpy.filename_handler_magic, dir=path.parent),
    )
    image = Image.open(path.parent / "textures/robot_D.png")
    for mesh in urdf.scene.geometry.values():
        uv = getattr(mesh.visual, "uv", None)
        if flat_color or uv is None:
            mesh.visual = trimesh.visual.ColorVisuals(mesh, face_colors=[170, 170, 170, 255])
        else:
            mesh.visual = trimesh.visual.TextureVisuals(
                uv=uv, material=trimesh.visual.material.SimpleMaterial(image=image)
            )
    return urdf


def main(args) -> None:
    paths = sorted(Path(args.results_dir).glob("*.npz"))
    assert paths, f"No .npz files in {args.results_dir}"
    clips = {p.stem: p for p in paths}
    dt = 1.0 / S.LAFAN1_FPS

    server = viser.ViserServer(port=args.port)
    server.scene.add_grid("/grid", width=6.0, height=6.0)
    root = server.scene.add_frame("/robot", show_axes=False)
    urdf_viz = ViserUrdf(server, load_textured_urdf(args.flat_color), root_node_name="/robot")
    viz_names = list(urdf_viz.get_actuated_joint_names())
    to_viz = None

    retargeter = None
    if args.source:
        import jax

        robot = S.load_sprout()
        retargeter, _ = S.make_retargeter(robot)
        load_clip = S.load_clip

    clip_dropdown = server.gui.add_dropdown("Clip", tuple(clips))
    playing = server.gui.add_checkbox("Playing", True)
    speed = server.gui.add_slider("Speed", min=0.1, max=2.0, step=0.1, initial_value=1.0)
    frame_slider = server.gui.add_slider("Frame", min=0, max=1, step=1, initial_value=0)
    qs, targets = None, None

    def load(_=None):
        nonlocal qs, to_viz, targets
        data = np.load(clips[clip_dropdown.value])
        qs = data["q"]
        names = [str(n) for n in data["joint_names"]]
        to_viz = np.array([names.index(n) for n in viz_names])
        if retargeter is not None:
            frames = load_clip(S.LAFAN1_DIR / f"{clip_dropdown.value}.bvh")
            targets = np.asarray(jax.vmap(retargeter.preprocess)(frames)["frame_pos"])
        frame_slider.max = len(qs) - 1
        frame_slider.value = 0

    clip_dropdown.on_update(load)
    load()
    try:
        while True:
            t0 = time.perf_counter()
            i = min(int(frame_slider.value), len(qs) - 1)
            q = qs[i]
            with server.atomic():
                root.position = q[:3]
                root.wxyz = q[3:7]
                urdf_viz.update_cfg(q[7:][to_viz])
                if targets is not None:
                    server.scene.add_point_cloud("/targets", targets[i], colors=(255, 60, 60), point_size=0.03)
            if playing.value:
                frame_slider.value = (i + 1) % len(qs)
            time.sleep(max(0.0, dt / speed.value - (time.perf_counter() - t0)))
    except KeyboardInterrupt:
        pass
    server.stop()


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--results-dir", default=str(S.OUT_DIR))
    ap.add_argument("--port", type=int, default=8080)
    ap.add_argument("--source", action="store_true", help="overlay the scaled LAFAN1 targets")
    ap.add_argument("--flat-color", action="store_true", help="plain grey meshes instead of the texture")
    main(ap.parse_args())
