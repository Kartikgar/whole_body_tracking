#!/usr/bin/env python3
"""Replay configured checkpoint segments and export Genesis-ready ONNX artifacts.

Example (run in an Isaac Lab Python environment):
    python scripts/prepare_segment_artifacts.py --batch_dir logs/batch_eval/beyondMimic/<run> \
        --motion jumps1_subject2__040.00s-060.00s --motion run1_subject5__200.00s-220.00s
"""

from __future__ import annotations

import argparse
import hashlib
import json
import os
from pathlib import Path
import shutil
import signal
import subprocess
import sys
import time

ROOT = Path(__file__).resolve().parents[1]


def sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for chunk in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def resolve_input(value: str, batch_dir: Path) -> Path:
    path = Path(value).expanduser()
    candidates = [path] if path.is_absolute() else [batch_dir / path, ROOT / path]
    for candidate in candidates:
        if candidate.is_file():
            return candidate.resolve()
    raise FileNotFoundError(f"Input file not found: {value}")


def select_segment(batch_dir: Path, query: str) -> dict:
    config = json.loads((batch_dir / "config.json").read_text())
    name = Path(query).name
    for suffix in (".npz", ".json"):
        name = name.removesuffix(suffix)
    matches = []
    for index, pair in enumerate(config["motions"]):
        stem = Path(pair["path"]).stem
        batch_id = f"{index:03d}_{stem}"
        if name in (stem, batch_id):
            matches.append((index, pair, batch_id))
    if len(matches) != 1:
        raise ValueError(f"Expected one config.json entry for {query!r}, found {len(matches)}")
    index, pair, batch_id = matches[0]
    motion = resolve_input(pair["path"], batch_dir)
    checkpoint = resolve_input(pair["checkpoint"], batch_dir)
    report_path = batch_dir / f"{batch_id}.json"
    if not report_path.is_file():
        raise FileNotFoundError(f"No batch evaluation report for this segment: {report_path}")
    report = json.loads(report_path.read_text())
    if resolve_input(report["checkpoint"], batch_dir) != checkpoint:
        raise ValueError("Saved rollout checkpoint disagrees with config.json")
    if resolve_input(report["motion_file"], batch_dir) != motion:
        raise ValueError("Saved rollout motion disagrees with config.json")
    return {
        "batch_motion_id": batch_id, "motion_file": str(motion), "checkpoint": str(checkpoint),
        "seed": int(config.get("seed", 0)) + index,
        "spawn_joint_noise_rad": float(config.get("spawn_joint_noise_rad", 0.01)),
    }


def stop_child(process: subprocess.Popen) -> None:
    if process.poll() is not None:
        return
    os.killpg(process.pid, signal.SIGTERM)
    try:
        process.wait(timeout=10)
    except subprocess.TimeoutExpired:
        os.killpg(process.pid, signal.SIGKILL)
        process.wait()


def run_isaac(command: list[str], log_path: Path, result_path: Path, timeout_s: float) -> tuple[int, bool]:
    """Bound both simulation runtime and cleanup; terminate the owned process group on interruption."""
    environment = os.environ.copy()
    environment["PYTHONPATH"] = str(ROOT / "source/whole_body_tracking") + os.pathsep + environment.get("PYTHONPATH", "")
    with log_path.open("w") as log:
        process = subprocess.Popen(command, cwd=ROOT, env=environment, stdout=log,
                                   stderr=subprocess.STDOUT, start_new_session=True)
        started = time.monotonic()
        saved_at = None
        try:
            while process.poll() is None:
                now = time.monotonic()
                if result_path.is_file() and saved_at is None:
                    saved_at = now
                if now - started > timeout_s:
                    raise TimeoutError(f"Isaac exceeded {timeout_s:g} seconds; see {log_path}")
                if saved_at is not None and now - saved_at > 30:
                    stop_child(process)
                    return process.returncode, True
                time.sleep(0.5)
        except BaseException:
            stop_child(process)
            raise
    return process.returncode, False


def validate_policy(path: Path, motion_path: Path) -> dict:
    """Use the Genesis loader and compare embedded references with the selected segment."""
    import numpy as np
    import onnx

    sys.path.insert(0, str(ROOT / "scripts"))
    from sim2sim_genesis.onnx_policy import OnnxMotionPolicy

    onnx.checker.check_model(onnx.load(str(path)))
    policy = OnnxMotionPolicy(str(path), "cpu", seed=0)
    with np.load(motion_path, allow_pickle=False) as motion:
        frames = int(motion["joint_pos"].shape[0])
        if policy.reference_motion_length_steps != frames:
            raise ValueError(f"ONNX embeds {policy.reference_motion_length_steps} frames, expected {frames}")
        obs = np.zeros((1, int(policy.get_obs_input_dim())), dtype=np.float32)
        checked = [0, frames // 2, frames - 1]
        for step in checked:
            outputs = policy.run(obs, step)
            for key in ("joint_pos", "joint_vel"):
                np.testing.assert_allclose(outputs[key][0], motion[key][step], atol=1e-5, rtol=1e-5)
            if not all(np.isfinite(value).all() for value in outputs.values()):
                raise ValueError(f"Nonfinite ONNX output at frame {step}")
    return {"genesis_loader": "passed", "reference_frames": frames,
            "reference_frames_checked": checked, "obs_dim": int(policy.get_obs_input_dim()),
            "num_actions": policy.num_actions}


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--batch_dir", type=Path, required=True)
    parser.add_argument("--motion", action="append", required=True, help="Segment stem or batch motion ID; repeat for multiple segments")
    parser.add_argument("--python", default=sys.executable, help="Isaac Lab Python executable")
    parser.add_argument("--device", default="cuda:0")
    parser.add_argument("--num_envs", type=int, default=1)
    parser.add_argument("--headless", action=argparse.BooleanOptionalAction, default=True)
    parser.add_argument("--timeout_s", type=float, default=600)
    parser.add_argument("--validate_only", action="store_true")
    args = parser.parse_args()
    if args.num_envs < 1 or not 0 < args.timeout_s < float("inf"):
        parser.error("num_envs must be positive and timeout_s finite and positive")
    batch_dir = args.batch_dir.resolve()
    selections = [select_segment(batch_dir, query) for query in args.motion]
    if len({item["batch_motion_id"] for item in selections}) != len(selections):
        parser.error("The same segment was selected more than once")
    if args.validate_only:
        print(json.dumps(selections, indent=2))
        return
    for selected in selections:
        motion = Path(selected["motion_file"])
        checkpoint = Path(selected["checkpoint"])
        folder = batch_dir / "segment_artifacts" / motion.stem
        # Never mix fresh outputs with artifacts from another invocation.
        folder.mkdir(parents=True, exist_ok=False)
        local_motion = folder / motion.name
        shutil.copy2(motion, local_motion)
        result_path, policy_path = folder / "rollout.json", folder / "policy.onnx"
        command = [args.python, str(ROOT / "scripts/rsl_rl/play.py"),
                   "--task", "Tracking-Flat-G1-v0", "--checkpoint", str(checkpoint),
                   "--motion_file", str(local_motion), "--num_envs", str(args.num_envs),
                   "--seed", str(selected["seed"]), "--device", args.device,
                   "--batch_eval", "--batch_spawn_joint_noise_rad", str(selected["spawn_joint_noise_rad"]),
                   "--batch_output_json", str(result_path), "--batch_output_npz", str(folder / "rollout.npz"),
                   "--video", "--batch_video_folder", str(folder), "--export_onnx_path", str(policy_path)]
        if args.headless:
            command.append("--headless")
        manifest = selected | {"batch_dir": str(batch_dir), "config_sha256": sha256(batch_dir / "config.json"),
                               "checkpoint_sha256": sha256(checkpoint), "motion_sha256": sha256(local_motion),
                               "command": command, "status": "running"}
        manifest_path = folder / "manifest.json"
        manifest_path.write_text(json.dumps(manifest, indent=2) + "\n")
        print(f"Preparing {motion.stem}: {checkpoint} -> {folder}", flush=True)
        try:
            code, cleanup_timeout = run_isaac(command, folder / "isaac.log", result_path, args.timeout_s)
            manifest.update(child_exit_code=code, cleanup_timeout=cleanup_timeout)
            if not result_path.is_file():
                log_path = folder / "isaac.log"
                log_tail = "\n".join(log_path.read_text(errors="replace").splitlines()[-65:])
                raise RuntimeError(
                    f"Isaac exited with code {code} before saving rollout.json. "
                    f"See {log_path}\n{log_tail}"
                )
            result = json.loads(result_path.read_text())
            if result["checkpoint"] != str(checkpoint) or result["motion_file"] != str(local_motion):
                raise ValueError("Demo provenance does not match the selected checkpoint and motion")
            validation = validate_policy(policy_path, local_motion)
            if result["terminated"] or result["steps_run"] != validation["reference_frames"]:
                raise RuntimeError(f"Demo did not complete the clip: {result['termination_reason']}")
            video = Path(result["video_mp4"])
            if not video.is_file() or not (folder / "rollout.npz").is_file():
                raise FileNotFoundError("Demo video or rollout recording is missing")
            demo = folder / "demo.mp4"
            video.rename(demo)
            result["video_mp4"] = str(demo)
            result_path.write_text(json.dumps(result, indent=2) + "\n")
            import imageio.v2 as imageio
            with imageio.get_reader(demo) as reader:
                video_frames = reader.count_frames()
                video_fps = reader.get_meta_data()["fps"]
            if video_frames != result["steps_run"]:
                raise ValueError(f"Demo contains {video_frames} frames, expected {result['steps_run']}")
            manifest.update(status="complete", onnx_validation=validation, child_exit_code=code,
                            cleanup_timeout=cleanup_timeout, video_frames=video_frames, video_fps=video_fps,
                            policy_sha256=sha256(policy_path), demo_video=str(demo), policy_onnx=str(policy_path))
            print(f"Complete: {demo} and {policy_path}", flush=True)
        except BaseException as exc:
            manifest.update(status="failed", error=str(exc))
            raise
        finally:
            manifest_path.write_text(json.dumps(manifest, indent=2) + "\n")


if __name__ == "__main__":
    main()
