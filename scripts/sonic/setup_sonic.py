#!/usr/bin/env python3
"""Download the pinned default SONIC models and G1 asset (no simulator needed)."""
import argparse
import hashlib
import json
from pathlib import Path
import sys
import urllib.request
import xml.etree.ElementTree as ET

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "source/whole_body_tracking"))
from whole_body_tracking.sonic.spec import MODEL_REVISION, SOURCE_REVISION


def download(url, path):
    path.parent.mkdir(parents=True, exist_ok=True)
    print(f"Download {path}", flush=True)
    temporary = path.with_suffix(path.suffix + ".part")
    try:
        with urllib.request.urlopen(url, timeout=120) as source, temporary.open("wb") as output:
            while chunk := source.read(1024 * 1024):
                output.write(chunk)
        temporary.replace(path)
    finally:
        temporary.unlink(missing_ok=True)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--model_dir", required=True, type=Path)
    parser.add_argument("--assets_only", action="store_true")
    args = parser.parse_args()
    root = args.model_dir.resolve()
    hf = f"https://huggingface.co/nvidia/GEAR-SONIC/resolve/{MODEL_REVISION}"
    raw = f"https://raw.githubusercontent.com/NVlabs/GR00T-WholeBodyControl/{SOURCE_REVISION}"
    if not args.assets_only:
        for name in ("model_encoder.onnx", "model_decoder.onnx", "observation_config.yaml", "LICENSE"):
            download(f"{hf}/{name}", root / name)
    asset = "gear_sonic/data/assets/robot_description"
    urdf = root / "robot_description/urdf/g1/main.urdf"
    download(f"{raw}/{asset}/urdf/g1/main.urdf", urdf)
    tree = ET.parse(urdf)
    meshes = set()
    for mesh in tree.iter("mesh"):
        name = mesh.attrib["filename"].removeprefix("package://robot_description/")
        if not name.startswith("meshes/g1/") or ".." in Path(name).parts:
            raise ValueError(f"Unexpected mesh path in pinned G1 asset: {name}")
        meshes.add(name)
        mesh.set("filename", "../../" + name)
    for name in sorted(meshes):
        url = f"https://media.githubusercontent.com/media/NVlabs/GR00T-WholeBodyControl/{SOURCE_REVISION}/{asset}/{name}"
        download(url, root / "robot_description" / name)
    tree.write(urdf, encoding="utf-8", xml_declaration=True)
    download(f"{raw}/LICENSE", root / "UPSTREAM_LICENSE")
    manifest = {"source_revision": SOURCE_REVISION, "model_revision": MODEL_REVISION,
                "files": {str(p.relative_to(root)): hashlib.sha256(p.read_bytes()).hexdigest()
                          for p in root.rglob("*") if p.is_file() and p.name != "manifest.json"}}
    (root / "manifest.json").write_text(json.dumps(manifest, indent=2) + "\n")
    print(f"Ready: {root}")


if __name__ == "__main__":
    main()
