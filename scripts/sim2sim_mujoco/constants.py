"""Constants for the MuJoCo sim2sim evaluator."""

from __future__ import annotations

import os

DEFAULT_G1_MJCF = os.path.join(
    os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__)))),
    "source",
    "whole_body_tracking",
    "whole_body_tracking",
    "assets",
    "unitree_description",
    "mjcf",
    "g1.xml",
)

DEFAULT_REFERENCE_MARKER_RADIUS = 0.035
DEFAULT_VIDEO_WIDTH = 640
DEFAULT_VIDEO_HEIGHT = 480
