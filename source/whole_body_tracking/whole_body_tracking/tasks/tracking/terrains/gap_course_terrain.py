from __future__ import annotations

import numpy as np
import trimesh
from dataclasses import MISSING

from isaaclab.terrains import SubTerrainBaseCfg
from isaaclab.utils import configclass


def _resolve_gap_centers_x(cfg: "GapCourseTerrainCfg") -> list[float]:
    if cfg.num_gaps <= 0:
        raise ValueError(f"num_gaps must be > 0. Received: {cfg.num_gaps}.")

    if cfg.gap_centers_x is not None:
        if len(cfg.gap_centers_x) < cfg.num_gaps:
            raise ValueError(
                "gap_centers_x has fewer entries than num_gaps: "
                f"len(gap_centers_x)={len(cfg.gap_centers_x)}, num_gaps={cfg.num_gaps}."
            )
        return list(cfg.gap_centers_x[: cfg.num_gaps])

    return [cfg.first_gap_center_x + i * cfg.gap_center_spacing for i in range(cfg.num_gaps)]


def _add_box(meshes: list[trimesh.Trimesh], dims: tuple[float, float, float], pos: tuple[float, float, float]):
    if dims[0] <= 1.0e-6 or dims[1] <= 1.0e-6 or dims[2] <= 1.0e-6:
        return
    meshes.append(trimesh.creation.box(dims, trimesh.transformations.translation_matrix(pos)))


def gap_course_terrain(difficulty: float, cfg: "GapCourseTerrainCfg") -> tuple[list[trimesh.Trimesh], np.ndarray]:
    """Generate a flat course with configurable rectangular gaps."""
    if cfg.size[0] <= 0.0 or cfg.size[1] <= 0.0:
        raise ValueError(f"Terrain size must be positive. Received: {cfg.size}.")
    if cfg.gap_depth <= 0.0:
        raise ValueError(f"gap_depth must be > 0. Received: {cfg.gap_depth}.")
    if cfg.surface_thickness <= 0.0:
        raise ValueError(f"surface_thickness must be > 0. Received: {cfg.surface_thickness}.")
    if cfg.floor_thickness <= 0.0:
        raise ValueError(f"floor_thickness must be > 0. Received: {cfg.floor_thickness}.")

    gap_width = cfg.gap_width_range[0] + difficulty * (cfg.gap_width_range[1] - cfg.gap_width_range[0])
    if gap_width <= 0.0:
        raise ValueError(f"gap_width must be > 0. Received: {gap_width}.")

    size_x, size_y = float(cfg.size[0]), float(cfg.size[1])
    center_x, center_y = 0.5 * size_x, 0.5 * size_y

    gap_centers_x_rel = _resolve_gap_centers_x(cfg)
    intervals: list[tuple[float, float, float]] = []
    for cx_rel in gap_centers_x_rel:
        cx = center_x + float(cx_rel)
        x0 = cx - 0.5 * gap_width
        x1 = cx + 0.5 * gap_width
        intervals.append((x0, x1, cx))

    intervals.sort(key=lambda item: item[0])

    prev_x1 = 0.0
    for x0, x1, _ in intervals:
        if x0 < 0.0 or x1 > size_x:
            raise ValueError(
                "Gap interval exceeds terrain bounds. "
                f"Interval [{x0:.3f}, {x1:.3f}] for terrain size_x={size_x:.3f}."
            )
        if x0 < prev_x1 - 1.0e-6:
            raise ValueError(
                "Gap intervals overlap. "
                f"Current interval starts at {x0:.3f} before previous ends at {prev_x1:.3f}."
            )
        prev_x1 = x1

    if cfg.gap_y_span is None:
        gap_y0, gap_y1 = 0.0, size_y
    else:
        if cfg.gap_y_span <= 0.0:
            raise ValueError(f"gap_y_span must be > 0 when provided. Received: {cfg.gap_y_span}.")
        if cfg.gap_y_span > size_y:
            raise ValueError(
                f"gap_y_span ({cfg.gap_y_span}) cannot exceed terrain width ({size_y})."
            )
        gy_center = center_y + cfg.gap_y_center_offset
        gap_y0 = gy_center - 0.5 * cfg.gap_y_span
        gap_y1 = gy_center + 0.5 * cfg.gap_y_span
        if gap_y0 < 0.0 or gap_y1 > size_y:
            raise ValueError(
                "gap_y_span with gap_y_center_offset exceeds terrain bounds. "
                f"Computed y interval [{gap_y0:.3f}, {gap_y1:.3f}] in [0, {size_y:.3f}]."
            )

    meshes: list[trimesh.Trimesh] = []

    # Base floor below all gaps so robots don't fall forever.
    floor_center_z = -cfg.gap_depth - 0.5 * cfg.floor_thickness
    _add_box(
        meshes,
        dims=(size_x, size_y, cfg.floor_thickness),
        pos=(center_x, center_y, floor_center_z),
    )

    # Top walkable slabs outside gap x-intervals.
    top_center_z = -0.5 * cfg.surface_thickness
    cursor_x = 0.0
    for x0, x1, _ in intervals:
        if x0 > cursor_x:
            width = x0 - cursor_x
            _add_box(
                meshes,
                dims=(width, size_y, cfg.surface_thickness),
                pos=(cursor_x + 0.5 * width, center_y, top_center_z),
            )
        cursor_x = x1
    if cursor_x < size_x:
        width = size_x - cursor_x
        _add_box(
            meshes,
            dims=(width, size_y, cfg.surface_thickness),
            pos=(cursor_x + 0.5 * width, center_y, top_center_z),
        )

    # If gap does not span full y, fill side strips inside each gap x-interval.
    if gap_y0 > 0.0 or gap_y1 < size_y:
        for x0, x1, cx in intervals:
            gap_x_width = x1 - x0
            if gap_y0 > 0.0:
                _add_box(
                    meshes,
                    dims=(gap_x_width, gap_y0, cfg.surface_thickness),
                    pos=(cx, 0.5 * gap_y0, top_center_z),
                )
            if gap_y1 < size_y:
                side_width = size_y - gap_y1
                _add_box(
                    meshes,
                    dims=(gap_x_width, side_width, cfg.surface_thickness),
                    pos=(cx, gap_y1 + 0.5 * side_width, top_center_z),
                )

    origin = np.array([center_x, center_y, 0.0])
    return meshes, origin


@configclass
class GapCourseTerrainCfg(SubTerrainBaseCfg):
    """Configuration for a course terrain with one or more jump gaps along +x."""

    function = gap_course_terrain

    num_gaps: int = 1
    """Number of gaps to carve in the terrain."""

    gap_width_range: tuple[float, float] = MISSING
    """Gap width along x-axis (min, max) in meters."""

    gap_depth: float = 1.0
    """Depth of the gap below the top surface in meters."""

    # Gap-center placement (x relative to terrain center).
    gap_centers_x: tuple[float, ...] | None = None
    """Optional explicit x-centers for gaps relative to terrain center."""

    first_gap_center_x: float = 1.0
    """First gap center x relative to terrain center when ``gap_centers_x`` is not provided."""

    gap_center_spacing: float = 1.5
    """Spacing between consecutive gap centers in meters."""

    # Gap opening span in y.
    gap_y_span: float | None = None
    """Gap span along y-axis in meters. If None, gap spans full terrain width."""

    gap_y_center_offset: float = 0.0
    """Gap center offset along y relative to terrain center."""

    # Mesh extrusion thicknesses.
    surface_thickness: float = 1.0
    """Thickness of top walkable surface slabs in meters."""

    floor_thickness: float = 1.0
    """Thickness of bottom floor slab in meters."""
