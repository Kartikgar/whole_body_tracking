#!/usr/bin/env python3
"""Run SONIC on every motion in a YAML file using parallel Isaac Sim environments."""

import argparse
import json
import subprocess
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "source/whole_body_tracking"))
from whole_body_tracking.sonic.batch import append_summary, load_batch_config, run_directory


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--config", required=True, help="YAML paths are relative to this file.")
    parser.add_argument("--device", default="cuda:0")
    parser.add_argument("--headless", action="store_true")
    args = parser.parse_args()
    config = load_batch_config(args.config)
    run_dir = run_directory(config["output_dir"])
    (run_dir / "config.json").write_text(json.dumps(config, indent=2) + "\n")
    csv_path = run_dir / "summary.csv"
    failures = []
    for index, motion in enumerate(config["motions"]):
        path = Path(motion["path"])
        if not path.is_file():
            raise FileNotFoundError(path)
        stem = f"{index:03d}_{path.stem}_trajectory{motion['trajectory_index']}"
        json_path = run_dir / f"{stem}.json"
        npz_path = run_dir / f"{stem}.npz"
        log_path = run_dir / f"{stem}.log"
        command = [sys.executable, str(ROOT / "scripts/run_sonic.py"),
                   "--sonic_model_dir", config["sonic_model_dir"],
                   "--motion_file", motion["path"],
                   "--trajectory_index", str(motion["trajectory_index"]),
                   "--num_envs", str(config["num_envs"]),
                   "--seed", str(config["seed"] + index),
                   "--spawn_joint_noise_rad", str(config["spawn_joint_noise_rad"]),
                   "--output_json", str(json_path), "--output_npz", str(npz_path),
                   "--full_motion", "--device", args.device]
        if args.headless:
            command.append("--headless")
        if config["record_video"]:
            command.extend(["--video", "--video_folder", str(run_dir / f"{stem}_video")])
        print(f"[SONIC batch] {index + 1}/{len(config['motions'])}: {path.name} -> {log_path}", flush=True)
        with log_path.open("w") as log:
            completed = subprocess.run(command, cwd=ROOT, stdout=log, stderr=subprocess.STDOUT, check=False)
        if not json_path.is_file():
            failures.append(f"{path}: process exit {completed.returncode}; see {log_path}")
            continue
        result = json.loads(json_path.read_text())
        append_summary(csv_path, result, stem, npz_path)
        if completed.returncode or result["terminated"]:
            failures.append(f"{path}: {result['termination_reason']}; see {log_path}")
        print(f"[SONIC batch] {stem}: {sum(not r['terminated'] for r in result['rollouts'])}/{config['num_envs']} completed", flush=True)
    print(f"[SONIC batch] Summary: {csv_path}", flush=True)
    if failures:
        raise SystemExit("\n".join(failures))


if __name__ == "__main__":
    main()
