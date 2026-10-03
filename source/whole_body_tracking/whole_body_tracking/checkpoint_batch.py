"""Configuration and result aggregation for motion-specific G1 checkpoint batches."""

from __future__ import annotations

import csv
import math
from datetime import datetime
from pathlib import Path

import yaml


def run_directory(output_dir: str) -> Path:
    base = Path(output_dir)
    stamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    candidate = base / stamp
    suffix = 1
    while candidate.exists():
        candidate = base / f"{stamp}_{suffix}"
        suffix += 1
    candidate.mkdir(parents=True)
    return candidate


def load_checkpoint_batch_config(path: str) -> dict:
    config_path = Path(path).resolve()
    with config_path.open(encoding="utf-8") as stream:
        config = yaml.safe_load(stream)
    if not isinstance(config, dict):
        raise ValueError("Batch YAML must be a mapping")

    def resolve(value: str) -> str:
        candidate = Path(value).expanduser()
        return str((config_path.parent / candidate).resolve())

    motions = config.get("motions")
    if not isinstance(motions, list) or not motions:
        raise ValueError("motions must be a nonempty list")
    pairs = []
    for index, item in enumerate(motions):
        if not isinstance(item, dict) or not item.get("path") or not item.get("checkpoint"):
            raise ValueError(f"Motion {index} requires path and checkpoint")
        motion_path = resolve(item["path"])
        checkpoint_path = resolve(item["checkpoint"])
        if not Path(motion_path).is_file():
            raise FileNotFoundError(motion_path)
        if not Path(checkpoint_path).is_file():
            raise FileNotFoundError(checkpoint_path)
        pairs.append({"path": motion_path, "checkpoint": checkpoint_path})
    num_envs = int(config.get("num_envs", 100))
    noise = float(config.get("spawn_joint_noise_rad", 0.01))
    if num_envs < 1 or noise < 0:
        raise ValueError("num_envs must be positive and spawn_joint_noise_rad must be nonnegative")
    return {
        "output_dir": resolve(config.get("output_dir", "../../logs/checkpoint_batch")),
        "num_envs": num_envs,
        "seed": int(config.get("seed", 42)),
        "spawn_joint_noise_rad": noise,
        "record_video": bool(config.get("record_video", True)),
        "motions": pairs,
    }


def checkpoint_summary_row(result: dict, label: str, npz_path: Path) -> dict:
    rollouts = result["rollouts"]
    row = {
        "motion": label,
        "motion_file": result["motion_file"],
        "checkpoint": result["checkpoint"],
        "metric_basis": result["metric_basis"],
        "num_envs": len(rollouts),
        "reference_duration_s": result["reference_duration_s"],
        "completed_envs": sum(not item["terminated"] for item in rollouts),
        "terminated_envs": sum(item["terminated"] for item in rollouts),
        "mean_duration_s": sum(item["completed_duration_s"] for item in rollouts) / len(rollouts),
        "min_duration_s": min(item["completed_duration_s"] for item in rollouts),
        "termination_reasons": ";".join(
            f"{item['env_id']}:{item['termination_reason']}" for item in rollouts if item["terminated"]
        ),
        "recording_npz": str(npz_path),
        "video_mp4": result.get("video_mp4", ""),
    }
    names = sorted(set().union(*(item["metrics"] for item in rollouts)))
    for name in names:
        values = [item["metrics"][name] for item in rollouts if math.isfinite(item["metrics"][name])]
        row[f"{name}_mean"] = sum(values) / len(values) if values else float("nan")
        if name.startswith("safety_"):
            row[f"{name}_worst"] = (
                min(values) if name in ("safety_min_joint_position_margin_rad", "safety_min_joint_soft_position_margin_rad") else max(values)
            ) if values else float("nan")
    row["status"] = "terminated" if result["terminated"] else "complete"
    return row


def write_checkpoint_summary(path: Path, rows: list[dict]) -> None:
    columns = list(dict.fromkeys(key for row in rows for key in row))
    with path.open("w", newline="", encoding="utf-8") as stream:
        writer = csv.DictWriter(stream, fieldnames=columns)
        writer.writeheader()
        writer.writerows(rows)


__all__ = ["load_checkpoint_batch_config", "checkpoint_summary_row", "write_checkpoint_summary", "run_directory"]
