"""Settings shared by the nominal-gain stand recorders (MuJoCo, Genesis, Isaac).

Every simulator runs the same test: start in the derived stand pose at the spec root
height, hold the pose targets with the hardware PD gains (no balance controller), and
record STAND_DURATION_S from the same side camera, continuing after a fall.
"""

from __future__ import annotations

import importlib.util
import sys
from pathlib import Path

import numpy as np

REPO = Path(__file__).resolve().parents[2]
_SPEC_FILE = REPO / "source/whole_body_tracking/whole_body_tracking/robots/sprout_spec.py"


def load_spec():
    """Import sprout_spec without importing the Isaac-dependent package."""
    if "sprout_spec" in sys.modules:
        return sys.modules["sprout_spec"]
    s = importlib.util.spec_from_file_location("sprout_spec", _SPEC_FILE)
    mod = importlib.util.module_from_spec(s)
    sys.modules["sprout_spec"] = mod
    s.loader.exec_module(mod)
    return mod


PHYSICS_DT = 0.005
STAND_DURATION_S = 3.0
RENDER_EVERY = 7  # physics steps per video frame
VIDEO_FPS = 1.0 / (RENDER_EVERY * PHYSICS_DT)  # ~28.6, real time
VIDEO_RES = (640, 480)  # width, height
CAMERA_EYE = (0.0, -2.5, 0.7)  # robot faces +x; viewed from its right side
CAMERA_TARGET = (0.0, 0.0, 0.4)
CAMERA_FOV_DEG = 45.0

OUT_DIR = REPO / "logs/sprout_phase1/stand_nominal"


def tilt_deg(quat_wxyz: np.ndarray) -> float:
    """Angle between the body z axis and world z, from a wxyz quaternion."""
    w, x, y, z = quat_wxyz
    up_z = 1.0 - 2.0 * (x * x + y * y)
    return float(np.degrees(np.arccos(np.clip(up_z, -1.0, 1.0))))


def save_log(sim: str, t: list, base_pos: list, base_quat: list, q: list, frames: list) -> Path:
    """Write <sim>.npz and <sim>.mp4 to OUT_DIR and print a one-line summary."""
    import imageio

    OUT_DIR.mkdir(parents=True, exist_ok=True)
    base_pos_a, base_quat_a = np.asarray(base_pos), np.asarray(base_quat)
    tilts = np.array([tilt_deg(qq) for qq in base_quat_a])
    np.savez(OUT_DIR / f"{sim}.npz", t=np.asarray(t), base_pos=base_pos_a, base_quat=base_quat_a,
             joint_pos=np.asarray(q), tilt_deg=tilts)
    video = OUT_DIR / f"{sim}.mp4"
    if frames:
        imageio.mimsave(video, frames, fps=VIDEO_FPS)
    else:
        video = "none"
    fall = np.nonzero(tilts > 30.0)[0]
    t_fall = f"{t[fall[0]]:.2f}s" if len(fall) else "no fall"
    print(f"[{sim}] tilt>30deg at {t_fall}; final height {base_pos_a[-1, 2]:.3f} m; video {video}")
    return video
