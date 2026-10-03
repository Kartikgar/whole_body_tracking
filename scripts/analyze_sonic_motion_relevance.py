"""Compare short windows from SONIC rollouts of different motions.

Rows are executed source motions; columns are executed target motions. A match
uses a threshold calibrated from recurrence within completed SONIC rollouts.
This is a motion-overlap hypothesis for transfer, not a measured transfer result.
"""

from __future__ import annotations

import argparse
import csv
import json
from collections import defaultdict
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
from scipy.spatial.distance import cdist
from whole_body_tracking.sonic.spec import BODY_NAMES as CANONICAL_BODY_NAMES
from whole_body_tracking.sonic.spec import JOINT_NAMES as CANONICAL_JOINT_NAMES


BODY_NAMES = (
    "pelvis", "torso_link", "left_knee_link", "right_knee_link",
    "left_ankle_roll_link", "right_ankle_roll_link",
    "left_elbow_link", "right_elbow_link",
    "left_wrist_yaw_link", "right_wrist_yaw_link",
)
LEFT_ANKLE = BODY_NAMES.index("left_ankle_roll_link")
RIGHT_ANKLE = BODY_NAMES.index("right_ankle_roll_link")

# Each group contributes its specified share to the squared window distance.
FEATURE_GROUPS = {
    "joint_pose": (0.22, 0.50),       # rad
    "joint_speed": (0.16, 2.00),      # rad/s; signed mean and RMS
    "body_pose": (0.26, 0.25),        # m, pelvis relative, heading aligned
    "root_speed": (0.12, 0.60),       # m/s, heading aligned
    "pelvis_height": (0.05, 0.20),    # m
    "foot_height": (0.05, 0.15),      # m, continuous contact proxy
    "foot_speed": (0.05, 0.50),       # m/s, continuous contact proxy
    "local_path": (0.07, 0.60),       # m, from window start
    "yaw_change": (0.02, 0.50),      # rad, from window start
}


def motion_label(row: dict[str, str]) -> str:
    name = row["motion"].split("_", 1)[1].removesuffix("_trajectory0")
    aliases = {
        "walk1_subject1_s0e500": "Walk S1",
        "walk1_subject2": "Walk S2",
        "jumps1_subject1": "Jump1 S1",
        "jumps1_subject2": "Jump1 S2",
        "dance1_subject1": "Dance1 S1",
        "dance1_subject2": "Dance1 S2",
        "fallAndGetUp1_subject1": "Fall+GetUp S1",
        "fallAndGetUp1_subject4": "Fall+GetUp S4",
        "run1_subject2": "Run1 S2",
        "run1_subject5": "Run1 S5",
        "sprint1_subject2": "Sprint1 S2",
    }
    return aliases.get(name, name)


def select_envs(rollouts: list[dict], count: int) -> list[int]:
    completed = sorted((r for r in rollouts if not r["terminated"]), key=lambda r: r["env_id"])
    failed = sorted((r for r in rollouts if r["terminated"]),
                    key=lambda r: (r["completed_duration_s"], r["env_id"]))

    def spaced(group: list[dict], size: int) -> list[int]:
        positions = np.linspace(0, len(group) - 1, min(size, len(group)), dtype=int)
        return [group[i]["env_id"] for i in positions]

    if not completed or not failed:
        return spaced(completed or failed, count)
    completed_count = min(len(completed), max(1, count // 2))
    failed_count = min(len(failed), count - completed_count)
    completed_count = min(len(completed), count - failed_count)
    return sorted(spaced(completed, completed_count) + spaced(failed, failed_count))


def load_frames(path: Path, trajectory_index: int | None, env_ids: list[int] | None,
                sample_hz: float) -> tuple[list[dict[str, np.ndarray]], float]:
    with np.load(path, allow_pickle=False) as archive:
        fps = float(np.asarray(archive["fps"]).reshape(-1)[0])
        raw_stride = round(fps / sample_hz)
        if raw_stride < 1 or not np.isclose(fps / raw_stride, sample_hz):
            raise ValueError(f"{path}: sampling rate {sample_hz:g} Hz does not divide {fps:g} Hz")
        names = [str(x) for x in archive["body_names"]] if "body_names" in archive else CANONICAL_BODY_NAMES
        body_ids = [names.index(name) for name in BODY_NAMES]
        joint_names = ([str(x) for x in archive["joint_names"]]
                       if "joint_names" in archive else CANONICAL_JOINT_NAMES)
        joint_ids = [joint_names.index(name) for name in CANONICAL_JOINT_NAMES]
        fields = ("joint_pos", "joint_vel", "body_pos_w", "body_quat_w", "body_lin_vel_w")
        sampled = {}
        for field in fields:
            raw = archive[field]
            if env_ids is not None:
                raw = raw[env_ids]
                raw = raw[:, ::raw_stride]
            else:
                if trajectory_index is not None and raw.ndim == (3 if field.startswith("joint") else 4):
                    raw = raw[trajectory_index]
                raw = raw[::raw_stride]
            if field.startswith("joint"):
                raw = raw[..., joint_ids]
            elif field == "body_quat_w":
                raw = raw[..., [body_ids[0]], :]
            else:
                raw = raw[..., body_ids, :]
            sampled[field] = np.asarray(raw, dtype=np.float32)
        if env_ids is None:
            count = len(sampled["joint_pos"])
            lengths = np.asarray(archive.get("valid_lengths", [count])).reshape(-1)
            if len(lengths) > 1:
                count = min(count, int(lengths[trajectory_index or 0] / raw_stride + 0.999))
            return [{key: value[:count] for key, value in sampled.items()}], fps
        valid_lengths = np.asarray(archive["valid_lengths"], dtype=int)[env_ids]
        result = []
        for env, raw_length in enumerate(valid_lengths):
            length = min(sampled["joint_pos"].shape[1], (int(raw_length) + raw_stride - 1) // raw_stride)
            result.append({key: value[env, :length] for key, value in sampled.items()})
        return result, fps


def yaw_from_quat(quat: np.ndarray) -> np.ndarray:
    w, x, y, z = np.moveaxis(quat, -1, 0)
    return np.unwrap(np.arctan2(2 * (w * z + x * y), 1 - 2 * (y * y + z * z)))


def rotate_xy(vector: np.ndarray, yaw: np.ndarray) -> np.ndarray:
    c, s = np.cos(yaw), np.sin(yaw)
    x, y = vector[..., 0], vector[..., 1]
    return np.stack((c * x + s * y, -s * x + c * y), axis=-1)


def window_features(frames: dict[str, np.ndarray], sample_hz: float,
                    window_samples: int, stride_samples: int,
                    temporal_bins: int) -> tuple[np.ndarray, np.ndarray]:
    length = len(frames["joint_pos"])
    if length < window_samples:
        return np.empty((0, 0), dtype=np.float32), np.empty(0, dtype=int)
    if window_samples % temporal_bins:
        raise ValueError("Window length must divide evenly into temporal bins")
    starts = np.arange(0, length - window_samples + 1, stride_samples, dtype=int)
    root_pos = frames["body_pos_w"][:, 0]
    yaw = yaw_from_quat(frames["body_quat_w"][:, 0])
    body_delta = frames["body_pos_w"] - root_pos[:, None]
    body_xy = rotate_xy(body_delta[..., :2], yaw[:, None])
    body_pose = np.concatenate((body_xy, body_delta[..., 2:]), axis=-1)[:, 1:].reshape(length, -1)
    root_velocity = frames["body_lin_vel_w"][:, 0]
    root_speed = np.concatenate((rotate_xy(root_velocity[:, :2], yaw), root_velocity[:, 2:]), axis=-1)
    feet_pos = frames["body_pos_w"][:, [LEFT_ANKLE, RIGHT_ANKLE]]
    feet_speed = np.linalg.norm(frames["body_lin_vel_w"][:, [LEFT_ANKLE, RIGHT_ANKLE]], axis=-1)
    frame_groups = {
        "joint_pose": frames["joint_pos"],
        "body_pose": body_pose,
        "root_speed": root_speed,
        "pelvis_height": root_pos[:, 2:3],
        "foot_height": feet_pos[..., 2],
        "foot_speed": feet_speed,
    }
    frames_per_bin = window_samples // temporal_bins
    vectors = []
    for start in starts:
        stop = start + window_samples

        def pool(values: np.ndarray) -> np.ndarray:
            return values[start:stop].reshape(temporal_bins, frames_per_bin, -1).mean(axis=1)

        groups = {name: pool(values) for name, values in frame_groups.items()}
        qvel = frames["joint_vel"][start:stop].reshape(temporal_bins, frames_per_bin, -1)
        groups["joint_speed"] = np.concatenate((qvel.mean(axis=1),
                                                  np.sqrt(np.mean(qvel * qvel, axis=1))), axis=-1)
        displacement = rotate_xy(root_pos[start:stop, :2] - root_pos[start, :2], yaw[start])
        groups["local_path"] = displacement.reshape(temporal_bins, frames_per_bin, 2).mean(axis=1)
        groups["yaw_change"] = (yaw[start:stop] - yaw[start]).reshape(temporal_bins, frames_per_bin, 1).mean(axis=1)
        vectors.append(np.concatenate([
            (groups[name] * np.sqrt(weight / groups[name].size) / scale).ravel()
            for name, (weight, scale) in FEATURE_GROUPS.items()
        ]).astype(np.float32))
    return np.stack(vectors), starts


def heatmap(values: np.ndarray, row_labels: list[str], col_labels: list[str],
            title: str, output: Path, *, vmax: float = 100,
            colorbar_label: str = "Target windows with a source match (%)",
            cell_format: str = ".0f", lower_is_better: bool = False) -> None:
    fig_width = max(10.8, 3.8 + 0.83 * len(col_labels))
    fig, ax = plt.subplots(figsize=(fig_width, 9.8))
    colors = plt.get_cmap("YlGnBu_r" if lower_is_better else "YlGnBu").copy()
    colors.set_bad("#e6e8eb")
    image = ax.imshow(np.ma.masked_invalid(values), vmin=0, vmax=vmax, cmap=colors, aspect="auto")
    ax.set_xticks(range(len(col_labels)), col_labels, rotation=45, ha="right", rotation_mode="anchor")
    ax.set_yticks(range(len(row_labels)), row_labels)
    ax.set_xlabel("Recorded target motion")
    ax.set_ylabel("Recorded source motion")
    ax.set_title(title, loc="center", fontsize=21, fontweight="bold", pad=20)
    ax.tick_params(axis="both", labelsize=12, length=0)
    ax.set_xticks(np.arange(-0.5, len(col_labels), 1), minor=True)
    ax.set_yticks(np.arange(-0.5, len(row_labels), 1), minor=True)
    ax.grid(which="minor", color="white", linewidth=1.8)
    ax.tick_params(which="minor", bottom=False, left=False)
    for (i, j), value in np.ndenumerate(values):
        if np.isfinite(value):
            dark = (value <= 0.38 * vmax) if lower_is_better else (value >= 0.62 * vmax)
            ax.text(j, i, format(value, cell_format), ha="center", va="center", fontsize=11,
                    color="white" if dark else "#1f2937")
    for j in range(values.shape[1]):
        if np.all(~np.isfinite(values[:, j])):
            ax.text(j, (values.shape[0] - 1) / 2, "N/A", ha="center", va="center",
                    fontsize=12, color="#58616b", rotation=90)
    colorbar = fig.colorbar(image, ax=ax, shrink=0.83, pad=0.02)
    colorbar.set_label(colorbar_label, fontsize=12)
    colorbar.ax.tick_params(labelsize=11)
    fig.subplots_adjust(left=0.18, right=0.92, top=0.91, bottom=0.19)
    for suffix in ("png", "pdf"):
        fig.savefig(output.with_suffix(f".{suffix}"), dpi=190 if suffix == "png" else None,
                    bbox_inches="tight", pad_inches=0.09)
    plt.close(fig)


def write_matrix(path: Path, labels: list[str], matrix: np.ndarray) -> None:
    with path.open("w", newline="") as stream:
        writer = csv.writer(stream)
        writer.writerow(["recorded_source / recorded_target", *labels])
        for label, values in zip(labels, matrix):
            writer.writerow([label, *(f"{x:.3f}" if np.isfinite(x) else "" for x in values)])


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("run_dir", type=Path)
    parser.add_argument("--output_dir", type=Path)
    parser.add_argument("--sample_hz", type=float, default=50)
    parser.add_argument("--window_s", type=float, default=2)
    parser.add_argument("--temporal_bins", type=int, default=4)
    parser.add_argument("--stride_s", type=float, default=0.5)
    parser.add_argument("--source_envs", "--rollout_envs", dest="source_envs", type=int, default=8)
    parser.add_argument("--min_source_env_support", "--min_rollout_env_support",
                        dest="min_source_env_support", type=int, default=3)
    parser.add_argument("--max_source_windows", "--max_rollout_windows",
                        dest="max_source_windows", type=int, default=600)
    parser.add_argument("--calibration_quantile", type=float, default=0.9)
    parser.add_argument("--failure_radius_s", type=float, default=4)
    args = parser.parse_args()
    run_dir = args.run_dir.resolve()
    output_dir = args.output_dir or run_dir / "analysis" / "relevance_rollout"
    output_dir.mkdir(parents=True, exist_ok=True)
    window_samples = round(args.window_s * args.sample_hz)
    stride_samples = round(args.stride_s * args.sample_hz)
    if (args.temporal_bins < 1 or window_samples < args.temporal_bins
            or window_samples % args.temporal_bins or stride_samples < 1 or args.source_envs < 1
            or not 1 <= args.min_source_env_support <= args.source_envs
            or args.max_source_windows < 1 or not 0 < args.calibration_quantile < 1
            or args.failure_radius_s <= 0):
        raise ValueError("Invalid sampling, temporal bins, rollout support, calibration, or failure-window setting")
    rows = list(csv.DictReader((run_dir / "summary.csv").open(newline="")))
    results = {row["motion"]: json.loads((run_dir / f"{row['motion']}.json").read_text()) for row in rows}

    def mean_survival(row: dict[str, str]) -> float:
        duration = float(row["reference_duration_s"])
        return np.mean([min(1, r["completed_duration_s"] / duration)
                        for r in results[row["motion"]]["rollouts"]])

    rows.sort(key=lambda r: (-int(r["completed_envs"]) / int(r["num_envs"]),
                             -mean_survival(r), float(r["root_position_error_mean"]),
                             float(r["joint_position_error_mean"])))
    labels = [motion_label(row) for row in rows]
    libraries = {}
    library_starts = {}
    failure_targets = {}
    selected_envs = {}
    for row in rows:
        key = row["motion"]
        rollouts = results[key]["rollouts"]
        rollout_by_env = {r["env_id"]: r for r in rollouts}
        env_ids = select_envs(rollouts, args.source_envs)
        selected_envs[key] = env_ids
        recording_path = Path(row["recording_npz"])
        if not recording_path.exists():
            recording_path = run_dir / recording_path.name
        trajectories, _ = load_frames(recording_path, None, env_ids, args.sample_hz)
        by_time = defaultdict(list)
        near_failure = []
        for env_id, trajectory in zip(env_ids, trajectories):
            vectors, starts = window_features(trajectory, args.sample_hz, window_samples,
                                              stride_samples, args.temporal_bins)
            if not len(starts):
                continue
            for start, vector in zip(starts, vectors):
                by_time[int(start)].append(vector)
            rollout = rollout_by_env[env_id]
            if rollout["terminated"]:
                failure_time = float(rollout["completed_duration_s"])
                centers = (starts + window_samples / 2) / args.sample_hz
                mask = (centers >= failure_time - args.failure_radius_s) & (centers <= failure_time)
                near_failure.extend(vectors[mask])
        representatives = []
        representative_starts = []
        for start in sorted(by_time):
            candidates = np.stack(by_time[start])
            if len(candidates) < args.min_source_env_support:
                continue
            median = np.median(candidates, axis=0)
            representatives.append(candidates[np.argmin(np.linalg.norm(candidates - median, axis=1))])
            representative_starts.append(start)
        if len(representatives) > args.max_source_windows:
            selected = np.linspace(0, len(representatives) - 1, args.max_source_windows, dtype=int)
            representatives = [representatives[i] for i in selected]
            representative_starts = [representative_starts[i] for i in selected]
        libraries[key] = np.stack(representatives) if representatives else np.empty((0, 0), dtype=np.float32)
        library_starts[key] = np.asarray(representative_starts, dtype=int)
        failure_targets[key] = np.stack(near_failure) if near_failure else np.empty((0, 0), dtype=np.float32)
        print(f"Recorded {motion_label(row)}: {len(representatives)} consensus windows "
              f"and {len(near_failure)} pre-failure windows from {len(env_ids)} envs", flush=True)

    calibration = []
    for row in rows:
        if int(row["completed_envs"]) != int(row["num_envs"]):
            continue
        key = row["motion"]
        vectors = libraries[key]
        starts = library_starts[key]
        if len(vectors) < 2:
            continue
        distances = cdist(vectors, vectors, metric="euclidean")
        distances[np.abs(starts[:, None] - starts[None, :]) < window_samples] = np.inf
        nearest = distances.min(axis=1)
        nearest = nearest[np.isfinite(nearest)]
        if len(nearest):
            calibration.append(nearest[np.linspace(0, len(nearest) - 1,
                                                       min(250, len(nearest)), dtype=int)])
    if not calibration:
        raise ValueError("No completed rollout has enough distinct-time windows to calibrate matching")
    threshold = float(np.quantile(np.concatenate(calibration), args.calibration_quantile))
    if threshold <= 0:
        raise ValueError("Rollout recurrence produced a nonpositive match threshold")
    print(f"Rollout recurrence match threshold: {threshold:.4f}", flush=True)

    n = len(rows)
    coverage = np.full((n, n), np.nan, dtype=float)
    failure_coverage = np.full((n, n), np.nan, dtype=float)
    failure_distance_ratio = np.full((n, n), np.nan, dtype=float)
    pair_rows = []
    for i, source_row in enumerate(rows):
        source_key = source_row["motion"]
        library = libraries[source_key]
        if not len(library):
            continue
        for j, target_row in enumerate(rows):
            target_key = target_row["motion"]
            target_vectors = libraries[target_key]
            if not len(target_vectors):
                continue
            best = cdist(target_vectors, library, metric="euclidean").min(axis=1)
            matched = best <= threshold
            coverage[i, j] = 100 * matched.mean()
            failure_vectors = failure_targets[target_key]
            if len(failure_vectors):
                failure_best = cdist(failure_vectors, library, metric="euclidean").min(axis=1)
                failure_coverage[i, j] = 100 * (failure_best <= threshold).mean()
                failure_distance_ratio[i, j] = failure_best.mean() / threshold
            pair_rows.append({
                "source": motion_label(source_row), "target": motion_label(target_row),
                "whole_clip_coverage_pct": f"{coverage[i, j]:.3f}",
                "failure_window_coverage_pct": (f"{failure_coverage[i, j]:.3f}"
                                                if np.isfinite(failure_coverage[i, j]) else ""),
                "failure_mean_distance_ratio": (f"{failure_distance_ratio[i, j]:.5f}"
                                                if np.isfinite(failure_distance_ratio[i, j]) else ""),
                "mean_nearest_distance": f"{best.mean():.5f}",
                "source_windows": len(library),
                "target_windows": len(target_vectors),
                "target_pre_failure_windows": len(failure_vectors),
                "source_completion_pct": 100 * int(source_row["completed_envs"]) / int(source_row["num_envs"]),
            })
        print(f"Compared source {motion_label(source_row)}", flush=True)

    write_matrix(output_dir / "whole_clip_coverage.csv", labels, coverage)
    write_matrix(output_dir / "failure_window_coverage.csv", labels, failure_coverage)
    write_matrix(output_dir / "failure_mean_distance_ratio.csv", labels, failure_distance_ratio)
    with (output_dir / "pair_scores.csv").open("w", newline="") as stream:
        if not pair_rows:
            raise ValueError("No rollout motion has enough supported windows for comparison")
        writer = csv.DictWriter(stream, fieldnames=list(pair_rows[0]))
        writer.writeheader()
        writer.writerows(pair_rows)
    plt.rcParams.update({"font.family": "DejaVu Sans", "pdf.fonttype": 42, "ps.fonttype": 42})
    heatmap(coverage, labels, labels, "SONIC rollout behavior overlap",
            output_dir / "whole_clip_coverage")
    heatmap(failure_coverage, labels, labels,
            "Source coverage of target pre-failure behavior", output_dir / "failure_window_coverage")
    finite_failure_distances = failure_distance_ratio[np.isfinite(failure_distance_ratio)]
    heatmap(failure_distance_ratio, labels, labels,
            "Nearest source behavior before target failure", output_dir / "failure_window_distance",
            vmax=max(2.5, float(np.percentile(finite_failure_distances, 95)))
                 if len(finite_failure_distances) else 2.5,
            colorbar_label="Distance / threshold (1 = match cutoff)",
            cell_format=".2f", lower_is_better=True)
    metadata = {
        "method": "Time-pooled kinematic windows from recorded SONIC rollouts on both axes; "
                  "nearest source consensus window to each target consensus window",
        "window_s": args.window_s, "stride_s": args.stride_s, "sample_hz": args.sample_hz,
        "temporal_bins": args.temporal_bins,
        "rollout_envs_max": args.source_envs, "min_rollout_env_support": args.min_source_env_support,
        "rollout_windows_max": args.max_source_windows,
        "feature_groups": FEATURE_GROUPS, "calibration_quantile": args.calibration_quantile,
        "match_threshold": threshold,
        "calibration": "Quantile of nearest nonoverlapping window distances within fully "
                       "completed rollout motions; same-time windows are excluded",
        "failure_radius_s": args.failure_radius_s,
        "failure_windows": "Recorded windows from selected failed envs whose centers fall in the "
                           "last failure_radius_s seconds before each env's termination",
        "selected_envs": selected_envs,
        "rollout_windows_used": {key: len(value) for key, value in libraries.items()},
        "pre_failure_windows_used": {key: len(value) for key, value in failure_targets.items()},
        "note": "A matching window indicates executed kinematic overlap, not successful policy transfer."
    }
    (output_dir / "method.json").write_text(json.dumps(metadata, indent=2) + "\n")
    print(f"Saved heatmaps and pair scores in {output_dir}")


if __name__ == "__main__":
    main()
