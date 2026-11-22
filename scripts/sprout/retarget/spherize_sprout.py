"""Generate a spherized collision model of Sprout for frax (bubblify YAML format).

frax's FloorConstraint (and optional self-collision) works on spheres attached to each
link, expressed in that link's frame. frax's G1 model was drawn by hand in bubblify; this
script fits one automatically from Sprout's URDF collision meshes:

- Feet: a 3 x 2 grid of small spheres whose bottoms lie exactly on the sole plane, spanning
  the sole's footprint. These decide foot-floor contact, so they are placed precisely.
- Other links: the mesh is split into up to MAX_SPHERES slabs along its principal axis;
  each slab gets a sphere at its bounding-box centre with a radius covering the
  RADIUS_PERCENTILE of its vertices (slightly under-covering; fine for floor avoidance).

Run in the frax env:
    python scripts/sprout/retarget/spherize_sprout.py
"""

from __future__ import annotations

import argparse
import xml.etree.ElementTree as ET
from pathlib import Path

import numpy as np
import trimesh
import yaml

HERE = Path(__file__).resolve().parent
WORKSPACE = HERE.parents[3]
URDF = WORKSPACE / "sprout-model-v0.1.4/sprout.urdf"
OUT = HERE / "assets/sprout_spherized.yml"

MAX_SPHERES = 4
RADIUS_PERCENTILE = 95
FOOT_SPHERE_RADIUS = 0.02
FOOT_LINKS = ("left_foot_link", "right_foot_link")
SKIP_LINKS = ("left_eyebrow_link", "right_eyebrow_link")  # fixed, cosmetic


def link_vertices(urdf_path: Path) -> dict[str, np.ndarray]:
    """Collision-mesh vertices per link, in the link frame (all Sprout collision origins are identity)."""
    root = ET.parse(urdf_path).getroot()
    out = {}
    for link in root.findall("link"):
        name = link.get("name")
        verts = []
        for col in link.findall("collision"):
            origin = col.find("origin")
            assert origin is None or (
                origin.get("xyz", "0 0 0").split() == ["0", "0", "0"]
                and origin.get("rpy", "0 0 0").split() == ["0", "0", "0"]
            ), f"{name}: non-identity collision origin not supported"
            mesh = col.find("geometry/mesh")
            mesh_path = urdf_path.parent / mesh.get("filename")
            loaded = trimesh.load(mesh_path, force="mesh")
            verts.append(np.asarray(loaded.vertices))
        if verts and name not in SKIP_LINKS:
            out[name] = np.concatenate(verts)
    return out


def foot_spheres(v: np.ndarray, r: float = FOOT_SPHERE_RADIUS) -> list[dict]:
    z_min = v[:, 2].min()
    sole = v[v[:, 2] < z_min + 0.01]
    x_lo, x_hi = sole[:, 0].min(), sole[:, 0].max()
    y_lo, y_hi = sole[:, 1].min(), sole[:, 1].max()
    xs = (x_lo + r, 0.5 * (x_lo + x_hi), x_hi - r)
    ys = (y_lo + r, y_hi - r)
    return [{"center": [float(x), float(y), float(z_min + r)], "radius": float(r)} for x in xs for y in ys]


def slab_spheres(v: np.ndarray) -> list[dict]:
    mean = v.mean(0)
    _, _, vt = np.linalg.svd(v - mean, full_matrices=False)
    local = (v - mean) @ vt.T  # principal-axis coordinates
    extent = local.max(0) - local.min(0)
    width = max(extent[1], extent[2], 1e-3)
    k = int(np.clip(round(extent[0] / width), 1, MAX_SPHERES))
    edges = np.linspace(local[:, 0].min(), local[:, 0].max(), k + 1)
    spheres = []
    for a, b in zip(edges[:-1], edges[1:]):
        pts = local[(local[:, 0] >= a) & (local[:, 0] <= b)]
        if len(pts) == 0:
            continue
        c_local = 0.5 * (pts.min(0) + pts.max(0))
        r = np.percentile(np.linalg.norm(pts - c_local, axis=1), RADIUS_PERCENTILE)
        c = mean + c_local @ vt
        spheres.append({"center": [float(x) for x in c], "radius": float(r)})
    return spheres


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--out", type=Path, default=OUT)
    args = ap.parse_args()
    verts = link_vertices(URDF)
    model = {
        name: (foot_spheres(v) if name in FOOT_LINKS else slab_spheres(v)) for name, v in verts.items()
    }
    args.out.parent.mkdir(parents=True, exist_ok=True)
    args.out.write_text(yaml.safe_dump({"collision_spheres": model}, sort_keys=False))
    n = sum(len(s) for s in model.values())
    print(f"{len(model)} links, {n} spheres -> {args.out}")
    for name, s in model.items():
        print(f"  {name:26s} {len(s)} spheres, r = " + ", ".join(f"{x['radius']:.3f}" for x in s))


if __name__ == "__main__":
    main()
