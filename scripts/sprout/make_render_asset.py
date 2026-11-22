"""Build a render-fixed copy of the Fauna Sprout description for URDF-based simulators.

Fauna's visual .mtl files reference `robot_D.png` / `robot_N.png` relative to
meshes/visual/, but the textures live in textures/, and their fallback diffuse colour is
black (Kd 0 0 0). Loaded from the URDF (Isaac, Genesis, viser), every link renders black or
grey. MuJoCo is unaffected (its MJCF points at textures/ directly).

The copy changes rendering only:
- sprout.urdf is copied byte-for-byte (same joints, inertias, collision geometry);
- meshes/collision/, textures/ and every visual .obj are symlinks to Fauna's files;
- only the visual .mtl files are rewritten: texture paths -> ../../textures/, Kd -> 1 1 1.

    python scripts/sprout/make_render_asset.py
"""

from __future__ import annotations

import filecmp
import os
import re
import shutil
from pathlib import Path

HERE = Path(__file__).resolve().parent
REPO = HERE.parents[1]
SRC = REPO.parent / "sprout-model-v0.1.4"
DST = REPO / "source/whole_body_tracking/whole_body_tracking/assets/sprout_description"


def patch_mtl(text: str) -> str:
    text = re.sub(r"^(map_Kd\s+)(\S+)$", lambda m: f"{m.group(1)}../../textures/{Path(m.group(2)).name}", text, flags=re.M)
    text = re.sub(r"^(bump\s+(?:-bm\s+\S+\s+)?)(\S+)$", lambda m: f"{m.group(1)}../../textures/{Path(m.group(2)).name}",
                  text, flags=re.M)
    return re.sub(r"^Kd\s+.*$", "Kd 1.00 1.00 1.00", text, flags=re.M)


def link(src: Path, dst: Path) -> None:
    dst.symlink_to(os.path.relpath(src, dst.parent))


def main() -> None:
    if DST.exists():
        shutil.rmtree(DST)
    (DST / "meshes/visual").mkdir(parents=True)
    shutil.copy2(SRC / "sprout.urdf", DST / "sprout.urdf")
    link(SRC / "meshes/collision", DST / "meshes/collision")
    link(SRC / "textures", DST / "textures")
    n_obj = n_mtl = 0
    for f in sorted((SRC / "meshes/visual").iterdir()):
        if f.suffix == ".mtl":
            (DST / "meshes/visual" / f.name).write_text(patch_mtl(f.read_text()))
            n_mtl += 1
        else:
            link(f, DST / "meshes/visual" / f.name)
            n_obj += 1
    assert filecmp.cmp(SRC / "sprout.urdf", DST / "sprout.urdf", shallow=False)
    print(f"{DST}: urdf copied, {n_obj} visual meshes linked, {n_mtl} .mtl patched")
    print("patched .mtl:\n" + (DST / "meshes/visual/torso_link.mtl").read_text())


if __name__ == "__main__":
    main()
