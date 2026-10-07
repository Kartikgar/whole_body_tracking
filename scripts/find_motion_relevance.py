"""Compare short windows from recorded rollouts of different motions.

Rows are executed source motions; columns are executed target motions. Values
are mean nearest-window distances, so lower values indicate greater kinematic
similarity. This is a motion-overlap hypothesis, not a measured transfer result.
"""

from __future__ import annotations

import argparse
import csv
import json
import re
from collections import defaultdict
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
from scipy.spatial.distance import cdist


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


def motion_identity(row: dict[str, str]) -> tuple[str, str]:
    """Format arbitrary motion IDs; recognize optional batch, segment and trajectory suffixes."""
    name = re.sub(r"^\d+_", "", row["motion"])
    trajectory = re.search(r"_trajectory(\d+)$", name)
    if trajectory:
        name = name[:trajectory.start()]
    segment = re.fullmatch(r"(.+)__(\d+(?:\.\d+)?)s-(\d+(?:\.\d+)?)s", name)
    parent = segment.group(1) if segment else name
    # Preserve motion numbers and frame-range suffixes; format subject IDs and camel case.
    readable = re.sub(r"([a-z])([A-Z])", r"\1 \2", parent)
    readable = re.sub(r"(?:^|_)subject(\d+)(?=_|$)", r" S\1", readable)
    readable = readable.replace("_", " ").strip()
    readable = readable[:1].upper() + readable[1:]
    label = readable
    if segment:
        label += f" {float(segment.group(2)):g}–{float(segment.group(3)):g}s"
    if trajectory:
        label += f" T{trajectory.group(1)}"
    return label, readable


def motion_label(row: dict[str, str]) -> str:
    return motion_identity(row)[0]


def parent_label(row: dict[str, str]) -> str:
    return motion_identity(row)[1]


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


def load_frames(path: Path, env_ids: list[int], sample_hz: float,
                joint_order: list[str]) -> list[dict[str, np.ndarray]]:
    with np.load(path, allow_pickle=False) as archive:
        fps = float(np.asarray(archive["fps"]).reshape(-1)[0])
        raw_stride = round(fps / sample_hz)
        if raw_stride < 1 or not np.isclose(fps / raw_stride, sample_hz):
            raise ValueError(f"{path}: sampling rate {sample_hz:g} Hz does not divide {fps:g} Hz")
        if "body_names" not in archive or "joint_names" not in archive:
            raise ValueError(f"{path}: rollout recording must contain body_names and joint_names")
        names = [str(x) for x in archive["body_names"]]
        missing = set(BODY_NAMES) - set(names)
        if missing:
            raise ValueError(f"{path}: missing feature bodies: {sorted(missing)}")
        body_ids = [names.index(name) for name in BODY_NAMES]
        joint_names = [str(x) for x in archive["joint_names"]]
        if len(set(joint_names)) != len(joint_names) or set(joint_names) != set(joint_order):
            raise ValueError(f"{path}: joint names differ from the other recordings")
        joint_ids = [joint_names.index(name) for name in joint_order]
        sampled = {}
        for field in ("joint_pos", "joint_vel", "body_pos_w", "body_quat_w", "body_lin_vel_w"):
            raw = archive[field][env_ids, ::raw_stride]
            if field.startswith("joint"):
                raw = raw[..., joint_ids]
            elif field == "body_quat_w":
                raw = raw[..., [body_ids[0]], :]
            else:
                raw = raw[..., body_ids, :]
            sampled[field] = np.asarray(raw, dtype=np.float32)
        valid_lengths = np.asarray(archive["valid_lengths"], dtype=int)[env_ids]
        result = []
        for env, raw_length in enumerate(valid_lengths):
            length = min(sampled["joint_pos"].shape[1], (int(raw_length) + raw_stride - 1) // raw_stride)
            result.append({key: value[env, :length] for key, value in sampled.items()})
        return result


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
            title: str, output: Path, *, vmax: float, safety_ordered: bool = False) -> None:
    large = max(len(row_labels), len(col_labels)) > 30
    fig_width = min(30, max(10.8, 3.8 + (0.24 if large else 0.83) * len(col_labels)))
    fig_height = min(30, max(9.8, 2.8 + 0.23 * len(row_labels))) if large else 9.8
    fig, ax = plt.subplots(figsize=(fig_width, fig_height))
    colors = plt.get_cmap("YlGnBu_r").copy()
    colors.set_bad("#e6e8eb")
    image = ax.imshow(np.ma.masked_invalid(values), vmin=0, vmax=vmax, cmap=colors, aspect="auto")
    ax.set_xticks(range(len(col_labels)), col_labels, rotation=45, ha="right", rotation_mode="anchor")
    ax.set_yticks(range(len(row_labels)), row_labels)
    direction = " (safe → risky)" if safety_ordered else ""
    ax.set_xlabel(f"Recorded target motion{direction}")
    ax.set_ylabel(f"Recorded source motion{direction}")
    ax.set_title(title, loc="center", fontsize=21, fontweight="bold", pad=20)
    ax.tick_params(axis="both", labelsize=6 if large else 12, length=0)
    ax.set_xticks(np.arange(-0.5, len(col_labels), 1), minor=True)
    ax.set_yticks(np.arange(-0.5, len(row_labels), 1), minor=True)
    ax.grid(which="minor", color="white", linewidth=1.8)
    ax.tick_params(which="minor", bottom=False, left=False)
    if not large:
        for (i, j), value in np.ndenumerate(values):
            if np.isfinite(value):
                dark = value <= 0.38 * vmax
                ax.text(j, i, f"{value:.2f}", ha="center", va="center", fontsize=11,
                        color="white" if dark else "#1f2937")
    for j in range(values.shape[1]):
        if np.all(~np.isfinite(values[:, j])):
            ax.text(j, (values.shape[0] - 1) / 2, "N/A", ha="center", va="center",
                    fontsize=12, color="#58616b", rotation=90)
    colorbar = fig.colorbar(image, ax=ax, shrink=0.83, pad=0.02)
    colorbar.set_label("Mean nearest-window distance (lower = more similar)", fontsize=12)
    colorbar.ax.tick_params(labelsize=11)
    fig.subplots_adjust(left=0.22 if large else 0.18, right=0.92,
                        top=0.96 if large else 0.91, bottom=0.22 if large else 0.19)
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
    parser.add_argument("--behavior_name", help="Optional display name; otherwise inferred from evaluation metadata")
    parser.add_argument("--sample_hz", type=float, default=50)
    parser.add_argument("--window_s", type=float, default=2)
    parser.add_argument("--temporal_bins", type=int, default=4)
    parser.add_argument("--stride_s", type=float, default=0.5)
    parser.add_argument("--source_envs", "--rollout_envs", dest="source_envs", type=int, default=8)
    parser.add_argument("--min_source_env_support", "--min_rollout_env_support",
                        dest="min_source_env_support", type=int, default=3)
    parser.add_argument("--max_source_windows", "--max_rollout_windows",
                        dest="max_source_windows", type=int, default=600)
    parser.add_argument("--min_duration_s", type=float, default=0,
                        help="Exclude shorter clips (e.g. 10 for the PgS2R-mini set)")
    parser.add_argument("--safety_ranking_csv", type=Path,
                        help="Order both axes by the rank column in this safety-ranking CSV")
    args = parser.parse_args()
    run_dir = args.run_dir.resolve()
    output_dir = args.output_dir or run_dir / "analysis" / "relevance_rollout"
    output_dir.mkdir(parents=True, exist_ok=True)
    window_samples = round(args.window_s * args.sample_hz)
    stride_samples = round(args.stride_s * args.sample_hz)
    if (args.temporal_bins < 1 or window_samples < args.temporal_bins
            or window_samples % args.temporal_bins or stride_samples < 1 or args.source_envs < 1
            or not 1 <= args.min_source_env_support <= args.source_envs
            or args.max_source_windows < 1 or not np.isfinite(args.min_duration_s)
            or args.min_duration_s < 0):
        raise ValueError("Invalid sampling, temporal bins, rollout support, or minimum duration")
    rows = list(csv.DictReader((run_dir / "summary.csv").open(newline="")))
    rows = [row for row in rows if float(row["reference_duration_s"]) >= args.min_duration_s]
    if not rows:
        raise ValueError(f"No motions last at least {args.min_duration_s:g} seconds")
    results = {row["motion"]: json.loads((run_dir / f"{row['motion']}.json").read_text()) for row in rows}

    def mean_survival(row: dict[str, str]) -> float:
        duration = float(row["reference_duration_s"])
        return np.mean([min(1, r["completed_duration_s"] / duration)
                        for r in results[row["motion"]]["rollouts"]])

    if args.safety_ranking_csv:
        with args.safety_ranking_csv.open(newline="") as stream:
            ranked = list(csv.DictReader(stream))
        ranked.sort(key=lambda row: int(row["rank"]))
        motions = [row["motion"] for row in ranked]
        rows_by_motion = {row["motion"]: row for row in rows}
        if len(set(motions)) != len(motions) or set(motions) != set(rows_by_motion):
            raise ValueError("Safety ranking must contain each included motion exactly once")
        rows = [rows_by_motion[motion] for motion in motions]
    else:
        rows.sort(key=lambda r: (-int(r["completed_envs"]) / int(r["num_envs"]),
                                 -mean_survival(r),
                                 float(r.get("root_position_error_mean") or r.get("error_anchor_pos_mean") or "inf"),
                                 float(r.get("joint_position_error_mean") or r.get("error_joint_pos_mean") or "inf")))
    labels = [motion_label(row) for row in rows]
    if len(set(labels)) != len(labels):
        raise ValueError("Motion IDs produce duplicate display labels; use distinct motion IDs")
    policy_types = {results[row["motion"]].get("policy_type", "") for row in rows}
    if args.behavior_name:
        run_kind = args.behavior_name
    elif policy_types == {"sonic"}:
        run_kind = "SONIC"
    elif policy_types == {"g1_checkpoint"}:
        run_kind = "BeyondMimic checkpoint"
    else:
        run_kind = "Recorded behavior"
    joint_order = None
    libraries = {}
    selected_envs = {}
    for row in rows:
        key = row["motion"]
        rollouts = results[key]["rollouts"]
        env_ids = select_envs(rollouts, args.source_envs)
        selected_envs[key] = env_ids
        recording_path = Path(row.get("recording_npz") or results[key].get("recording_npz") or f"{key}.npz")
        candidates = [recording_path] if recording_path.is_absolute() else [run_dir / recording_path]
        candidates.append(run_dir / recording_path.name)
        recording_path = next((path for path in candidates if path.is_file()), None)
        if recording_path is None:
            raise FileNotFoundError(f"{key}: no rollout recording at {candidates}")
        if joint_order is None:
            with np.load(recording_path, allow_pickle=False) as archive:
                if "joint_names" not in archive:
                    raise ValueError(f"{recording_path}: missing joint_names")
                joint_order = sorted(str(name) for name in archive["joint_names"])
        trajectories = load_frames(recording_path, env_ids, args.sample_hz, joint_order)
        by_time = defaultdict(list)
        for trajectory in trajectories:
            vectors, starts = window_features(trajectory, args.sample_hz, window_samples,
                                              stride_samples, args.temporal_bins)
            if not len(starts):
                continue
            for start, vector in zip(starts, vectors):
                by_time[int(start)].append(vector)
        representatives = []
        for start in sorted(by_time):
            candidates = np.stack(by_time[start])
            if len(candidates) < args.min_source_env_support:
                continue
            median = np.median(candidates, axis=0)
            representatives.append(candidates[np.argmin(np.linalg.norm(candidates - median, axis=1))])
        if len(representatives) > args.max_source_windows:
            selected = np.linspace(0, len(representatives) - 1, args.max_source_windows, dtype=int)
            representatives = [representatives[i] for i in selected]
        libraries[key] = np.stack(representatives) if representatives else np.empty((0, 0), dtype=np.float32)
        print(f"Recorded {motion_label(row)}: {len(representatives)} consensus windows "
              f"from {len(env_ids)} envs", flush=True)

    n = len(rows)
    mean_distance = np.full((n, n), np.nan, dtype=float)
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
            mean_distance[i, j] = best.mean()
            pair_rows.append({
                "source": motion_label(source_row), "target": motion_label(target_row),
                "source_motion": source_key, "target_motion": target_key,
                "mean_nearest_distance": f"{mean_distance[i, j]:.5f}",
                "source_windows": len(library),
                "target_windows": len(target_vectors),
                "source_completion_pct": 100 * int(source_row["completed_envs"]) / int(source_row["num_envs"]),
            })
        print(f"Compared source {motion_label(source_row)}", flush=True)

    write_matrix(output_dir / "mean_nearest_distance.csv", labels, mean_distance)
    parent_names = list(dict.fromkeys(parent_label(row) for row in rows))
    if args.safety_ranking_csv:
        parent_names.sort(key=lambda name: np.median([
            index for index, row in enumerate(rows) if parent_label(row) == name
        ]))
    if len(parent_names) < n:
        parent_ids = [[i for i, row in enumerate(rows) if parent_label(row) == name]
                      for name in parent_names]
        parent_distance = np.full((len(parent_names), len(parent_names)), np.nan)
        for source, source_ids in enumerate(parent_ids):
            for target, target_ids in enumerate(parent_ids):
                pairs = [(i, j) for i in source_ids for j in target_ids if i != j]
                if pairs:
                    parent_distance[source, target] = np.mean([mean_distance[i, j] for i, j in pairs])
        write_matrix(output_dir / "parent_mean_distance.csv", parent_names, parent_distance)
    with (output_dir / "pair_scores.csv").open("w", newline="") as stream:
        if not pair_rows:
            raise ValueError("No rollout motion has enough supported windows for comparison")
        writer = csv.DictWriter(stream, fieldnames=list(pair_rows[0]))
        writer.writeheader()
        writer.writerows(pair_rows)
    plt.rcParams.update({"font.family": "DejaVu Sans", "pdf.fonttype": 42, "ps.fonttype": 42})
    finite_distances = mean_distance[np.isfinite(mean_distance)]
    if not len(finite_distances):
        raise ValueError("No rollout motion has enough supported windows for comparison")
    distance_vmax = max(float(np.percentile(finite_distances, 95)), 1e-9)
    title_prefix = "Safety-ordered " if args.safety_ranking_csv else ""
    heatmap(mean_distance, labels, labels,
            f"{title_prefix}{run_kind} mean nearest-window distance",
            output_dir / "mean_nearest_distance", vmax=distance_vmax,
            safety_ordered=bool(args.safety_ranking_csv))
    if len(parent_names) < n:
        heatmap(parent_distance, parent_names, parent_names,
                f"{title_prefix}{run_kind} parent-motion mean distance (distinct clip pairs)",
                output_dir / "parent_mean_distance", vmax=distance_vmax,
                safety_ordered=bool(args.safety_ranking_csv))
    metadata = {
        "method": "Time-pooled kinematic windows from recorded rollouts on both axes; "
                  "nearest source consensus window to each target consensus window",
        "behavior_name": run_kind, "policy_types": sorted(policy_types),
        "joint_names": joint_order, "feature_body_names": BODY_NAMES,
        "axis_motions": [{"motion": row["motion"], "label": motion_label(row),
                          "parent_label": parent_label(row)} for row in rows],
        "window_s": args.window_s, "stride_s": args.stride_s, "sample_hz": args.sample_hz,
        "temporal_bins": args.temporal_bins,
        "rollout_envs_max": args.source_envs, "min_rollout_env_support": args.min_source_env_support,
        "rollout_windows_max": args.max_source_windows,
        "min_duration_s": args.min_duration_s,
        "axis_order": ("safety_ranking_csv" if args.safety_ranking_csv else "tracking_performance"),
        "safety_ranking_csv": str(args.safety_ranking_csv.resolve()) if args.safety_ranking_csv else None,
        "safety_contact_exempt_bodies_in_saved_rollouts": (
            results[rows[0]["motion"]].get("safety_definitions", {}).get("allowed_contact_bodies")
            if args.safety_ranking_csv else None
        ),
        "parent_axis_order": ("median clip safety rank" if args.safety_ranking_csv else "first clip appearance"),
        "feature_groups": FEATURE_GROUPS,
        "selected_envs": selected_envs,
        "rollout_windows_used": {key: len(value) for key, value in libraries.items()},
        "parent_summary": "Arithmetic mean over distinct source-target clip pairs; directional; "
                          "same-parent diagonal omits self-comparisons",
        "note": "A smaller distance indicates executed kinematic similarity, not successful policy transfer."
    }
    (output_dir / "method.json").write_text(json.dumps(metadata, indent=2) + "\n")
    print(f"Saved heatmaps and pair scores in {output_dir}")


if __name__ == "__main__":
    main()
