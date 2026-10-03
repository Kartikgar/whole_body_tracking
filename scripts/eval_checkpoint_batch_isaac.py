#!/usr/bin/env python3
"""Evaluate one motion-specific G1 checkpoint per motion in Isaac Lab."""

import argparse
import json
import subprocess
import sys
import time
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "source/whole_body_tracking"))
from whole_body_tracking.checkpoint_batch import (
    checkpoint_summary_row, load_checkpoint_batch_config, run_directory, write_checkpoint_summary,
)


def run_child_with_shutdown_timeout(command, *, cwd, log, result_path, grace_s=30.0):
    """Allow long rollouts, but bound child cleanup after its result is saved."""
    process = subprocess.Popen(command, cwd=cwd, stdout=log, stderr=subprocess.STDOUT)
    result_seen_at = None
    try:
        while process.poll() is None:
            if result_seen_at is None and result_path.is_file():
                result_seen_at = time.monotonic()
            if result_seen_at is not None and time.monotonic() - result_seen_at >= grace_s:
                message = f"Child did not exit within {grace_s:g} s of saving {result_path}; terminating it."
                print(f"[Checkpoint batch] {message}", flush=True)
                log.write(message + "\n")
                log.flush()
                process.terminate()
                try:
                    process.wait(timeout=10.0)
                except subprocess.TimeoutExpired:
                    process.kill()
                    process.wait()
                return process.returncode, True
            time.sleep(0.5)
    except BaseException:
        if process.poll() is None:
            process.terminate()
            try:
                process.wait(timeout=10.0)
            except subprocess.TimeoutExpired:
                process.kill()
                process.wait()
        raise
    return process.returncode, False


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--config", required=True)
    parser.add_argument("--device", default="cuda:0")
    parser.add_argument("--headless", action="store_true")
    parser.add_argument("--validate_only", action="store_true", help="Check YAML and file paths without launching Isaac.")
    args = parser.parse_args()
    config = load_checkpoint_batch_config(args.config)
    if args.validate_only:
        print(f"Validated {len(config['motions'])} motion/checkpoint pairs.")
        return
    run_dir = run_directory(config["output_dir"])
    (run_dir / "config.json").write_text(json.dumps(config, indent=2) + "\n", encoding="utf-8")
    failures = []
    summary_rows = []
    for index, pair in enumerate(config["motions"]):
        motion = Path(pair["path"])
        stem = f"{index:03d}_{motion.stem}"
        json_path = run_dir / f"{stem}.json"
        npz_path = run_dir / f"{stem}.npz"
        log_path = run_dir / f"{stem}.log"
        video_dir = run_dir / f"{stem}_video"
        command = [
            sys.executable, str(ROOT / "scripts/rsl_rl/play.py"),
            "--task", "Tracking-Flat-G1-v0",
            "--checkpoint", pair["checkpoint"], "--motion_file", pair["path"],
            "--num_envs", str(config["num_envs"]), "--seed", str(config["seed"] + index),
            "--batch_eval", "--batch_spawn_joint_noise_rad", str(config["spawn_joint_noise_rad"]),
            "--batch_output_json", str(json_path), "--batch_output_npz", str(npz_path),
            "--device", args.device,
        ]
        if args.headless:
            command.append("--headless")
        if config["record_video"]:
            command.extend(["--video", "--batch_video_folder", str(video_dir)])
        print(f"[Checkpoint batch] {index + 1}/{len(config['motions'])}: {motion.name} -> {log_path}", flush=True)
        with log_path.open("w", encoding="utf-8") as log:
            returncode, shutdown_timed_out = run_child_with_shutdown_timeout(
                command, cwd=ROOT, log=log, result_path=json_path,
            )
        try:
            result = json.loads(json_path.read_text(encoding="utf-8"))
            if "rollouts" not in result:
                raise ValueError("missing rollouts")
        except (OSError, ValueError, KeyError) as exc:
            failure = f"{motion}: process exit {returncode}; missing/invalid result ({exc}); see {log_path}"
            failures.append(failure)
            json_path.write_text(json.dumps({
                "motion_file": pair["path"], "checkpoint": pair["checkpoint"],
                "status": "process_failed", "error": failure,
            }, indent=2) + "\n", encoding="utf-8")
            summary_rows.append({
                "motion": stem, "motion_file": pair["path"], "checkpoint": pair["checkpoint"],
                "status": "process_failed", "error": failure,
            })
            write_checkpoint_summary(run_dir / "summary.csv", summary_rows)
            continue
        row = checkpoint_summary_row(result, stem, npz_path)
        if shutdown_timed_out:
            row["status"] = "shutdown_timeout"
            row["error"] = f"Isaac child did not exit after saving results; see {log_path}"
        elif returncode and not result["terminated"]:
            row["status"] = "process_failed"
            row["error"] = f"process exit {returncode}; see {log_path}"
        missing_outputs = []
        if not npz_path.is_file():
            missing_outputs.append(str(npz_path))
        if config["record_video"] and not Path(result.get("video_mp4", "")).is_file():
            missing_outputs.append("video_mp4")
        if missing_outputs:
            row["status"] = "missing_output"
            row["error"] = f"Missing outputs: {', '.join(missing_outputs)}; see {log_path}"
        summary_rows.append(row)
        write_checkpoint_summary(run_dir / "summary.csv", summary_rows)
        if returncode or result["terminated"] or missing_outputs:
            failures.append(f"{motion}: {row.get('error', result['termination_reason'])}; see {log_path}")
        print(
            f"[Checkpoint batch] {stem}: "
            f"{sum(not r['terminated'] for r in result['rollouts'])}/{config['num_envs']} completed",
            flush=True,
        )
    print(f"[Checkpoint batch] Summary: {run_dir / 'summary.csv'}", flush=True)
    if failures:
        raise SystemExit("\n".join(failures))


if __name__ == "__main__":
    main()
