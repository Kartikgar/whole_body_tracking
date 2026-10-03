"""Configuration and result aggregation for Isaac Sim SONIC batches."""

import csv
import math
from datetime import datetime
from pathlib import Path

import yaml


def load_batch_config(path):
    config_path = Path(path).resolve()
    with config_path.open() as stream:
        config = yaml.safe_load(stream)
    if not isinstance(config, dict):
        raise ValueError("Batch YAML must be a mapping")
    base = config_path.parent

    def resolve(value):
        path = Path(value).expanduser()
        return str((base / path).resolve()) if not path.is_absolute() else str(path.resolve())

    for key in ("sonic_model_dir", "output_dir", "num_envs", "motions"):
        if key not in config:
            raise ValueError(f"Batch YAML requires {key}")
    num_envs = int(config["num_envs"])
    if num_envs < 1:
        raise ValueError("num_envs must be positive")
    motions = config["motions"]
    if not isinstance(motions, list) or not motions:
        raise ValueError("motions must be a nonempty list")
    normalized = []
    for motion in motions:
        if not isinstance(motion, dict) or "path" not in motion:
            raise ValueError("Each motion needs a path")
        index = int(motion.get("trajectory_index", 0))
        if index < 0:
            raise ValueError("trajectory_index must be nonnegative")
        normalized.append({"path": resolve(motion["path"]), "trajectory_index": index})
    noise = float(config.get("spawn_joint_noise_rad", 0.01))
    if noise < 0:
        raise ValueError("spawn_joint_noise_rad must be nonnegative")
    return {
        "sonic_model_dir": resolve(config["sonic_model_dir"]),
        "output_dir": resolve(config["output_dir"]),
        "num_envs": num_envs,
        "seed": int(config.get("seed", 0)),
        "spawn_joint_noise_rad": noise,
        "record_video": bool(config.get("record_video", True)),
        "motions": normalized,
    }


def run_directory(output_dir):
    base = Path(output_dir)
    stamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    candidate = base / stamp
    suffix = 1
    while candidate.exists():
        candidate = base / f"{stamp}_{suffix}"
        suffix += 1
    candidate.mkdir(parents=True)
    return candidate


def append_summary(csv_path, result, motion_label, recording_path):
    """One row per motion, including aggregate and individual rollout outcomes."""
    rollouts = result["rollouts"]
    count = len(rollouts)
    row = {
        "motion": motion_label,
        "motion_file": result["motion_file"],
        "trajectory_index": result["trajectory_index"],
        "num_envs": count,
        "reference_duration_s": result["reference_duration_s"],
        "completed_envs": sum(not r["terminated"] for r in rollouts),
        "terminated_envs": sum(r["terminated"] for r in rollouts),
        "mean_duration_s": sum(r["completed_duration_s"] for r in rollouts) / count,
        "min_duration_s": min(r["completed_duration_s"] for r in rollouts),
        "termination_reasons": ";".join(f"{r['env_id']}:{r['termination_reason']}" for r in rollouts if r["terminated"]),
        "recording_npz": str(recording_path),
        "video_mp4": result.get("video_mp4", ""),
    }
    metric_names = sorted(set().union(*(r["metrics"] for r in rollouts)))
    for key in metric_names:
        values = [r["metrics"][key] for r in rollouts if math.isfinite(r["metrics"][key])]
        row[f"{key}_mean"] = sum(values) / len(values) if values else float("nan")
        if key.startswith("safety_"):
            row[f"{key}_worst"] = (
                min(values) if key in ("safety_min_joint_position_margin_rad", "safety_min_joint_soft_position_margin_rad") else max(values)
            ) if values else float("nan")
    path = Path(csv_path)
    exists = path.exists()
    with path.open("a", newline="") as stream:
        writer = csv.DictWriter(stream, fieldnames=list(row))
        if not exists:
            writer.writeheader()
        writer.writerow(row)
