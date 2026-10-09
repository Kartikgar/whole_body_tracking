#!/usr/bin/env python3
"""Bootstrap whole trajectories from a trusted Genesis recording for ensemble training.

Only declared trajectory fields are resampled; arbitrary recording metadata is
preserved even when its leading dimension happens to equal the trajectory count.
Object metadata requires pickle loading: use trusted local input files only.
"""

from __future__ import annotations

import argparse
import hashlib
import json
from pathlib import Path
import zipfile

import numpy as np

STATE_KEYS = (
    "joint_pos", "joint_vel", "body_pos_w", "body_quat_w",
    "body_lin_vel_w", "body_ang_vel_w",
)
ACTION_KEYS = ("action", "actions", "joint_action", "joint_actions")
TRAJECTORY_KEYS = set(STATE_KEYS + ACTION_KEYS + tuple(f"initial_{k}" for k in STATE_KEYS))
TRAJECTORY_KEYS.update(("valid_lengths", "source_files", "bootstrap_source_indices"))


def bootstrap_motion_file(
    input_path: Path, output_dir: Path, num_sets: int,
    *, seed: int = 0, sample_size: int | None = None,
    trajectory_keys: tuple[str, ...] = (),
) -> list[Path]:
    """Write independent samples with replacement, plus source/OOB manifests.

    Memory usage is bounded by one input field and its resampled output field.
    The time dimension and padding are copied exactly, without frame resampling.
    """
    if num_sets < 1 or seed < 0 or (sample_size is not None and sample_size < 1):
        raise ValueError("num_sets and sample_size must be positive; seed must be nonnegative")
    input_path = Path(input_path).expanduser().resolve()
    output_dir = Path(output_dir).expanduser().resolve()
    keys = TRAJECTORY_KEYS | set(trajectory_keys)
    with np.load(input_path, allow_pickle=True) as data:
        missing = set(STATE_KEYS) - set(data.files)
        if missing or not any(k in data for k in ACTION_KEYS):
            raise ValueError(f"Expected Genesis states and recorded actions; missing states: {sorted(missing)}")
        shape = data["joint_pos"].shape
        if len(shape) != 3 or any(d < 1 for d in shape):
            raise ValueError("Expected nonempty stacked joint_pos [trajectories, time, joints]")
        count, steps, _ = shape
        size = count if sample_size is None else sample_size
        if set(trajectory_keys) - set(data.files):
            raise ValueError("An explicitly declared trajectory key is absent from the input")
        for key in data.files:
            if key in keys:
                arr = data[key]
                if arr.ndim < 1 or arr.shape[0] != count:
                    raise ValueError(f"{key}: expected leading trajectory dimension {count}, got {arr.shape}")
                if key in STATE_KEYS + ACTION_KEYS and (arr.ndim < 3 or arr.shape[1] != steps):
                    raise ValueError(f"{key}: expected leading dimensions ({count}, {steps})")
        lengths = data["valid_lengths"] if "valid_lengths" in data else np.full(count, steps, dtype=np.int64)
        if lengths.shape != (count,) or lengths.dtype.kind not in "iu" or np.any(lengths < 1) or np.any(lengths > steps):
            raise ValueError("valid_lengths must contain one integer in [1, time_steps] per trajectory")
        outputs = [output_dir / f"{input_path.stem}_btstrap_{i:03d}.npz" for i in range(num_sets)]
        manifests = [p.with_suffix(".json") for p in outputs]
        for path in outputs + manifests:
            if path.exists():
                raise FileExistsError(f"Refusing to overwrite {path}")
        with input_path.open("rb") as stream:
            hasher = hashlib.sha256()
            for chunk in iter(lambda: stream.read(1024 * 1024), b""):
                hasher.update(chunk)
            digest = hasher.hexdigest()
        output_dir.mkdir(parents=True, exist_ok=True)
        children = np.random.SeedSequence(seed).spawn(num_sets)
        for index, (output, manifest, child) in enumerate(zip(outputs, manifests, children)):
            indices = np.random.default_rng(child).integers(0, count, size=size)
            unique = np.unique(indices)
            info = {
                "method": "trajectory bootstrap with replacement",
                "source_path": str(input_path), "source_sha256": digest,
                "seed": seed, "spawn_key": list(child.spawn_key),
                "member": index, "source_trajectory_count": count,
                "sample_size": size, "unique_trajectory_count": int(unique.size),
                "source_indices": indices.tolist(),
                "out_of_bag_indices": np.setdiff1d(np.arange(count), unique).tolist(),
                "extra_trajectory_keys": list(trajectory_keys),
            }
            # Write fields individually instead of keeping all body-state arrays in RAM.
            try:
                with output.open("xb") as stream, zipfile.ZipFile(stream, "w", allowZip64=True) as archive:
                    def write(key, value):
                        with archive.open(f"{key}.npy", "w", force_zip64=True) as member:
                            np.lib.format.write_array(member, np.asarray(value), allow_pickle=True)

                    for key in data.files:
                        if key in ("bootstrap_source_indices", "bootstrap_metadata_json"):
                            continue
                        arr = data[key]
                        if key in keys:
                            arr = arr[indices]
                        elif key == "num_motions":
                            arr = np.full_like(arr, size)
                        elif key == "fps" and arr.shape == (count,) and count > 1:
                            arr = arr[indices]
                        write(key, arr)
                    if "valid_lengths" not in data:
                        write("valid_lengths", lengths[indices])
                    write("bootstrap_source_indices", indices)
                    write("bootstrap_metadata_json", json.dumps(info, sort_keys=True))
                with manifest.open("x") as stream:
                    json.dump(info, stream, indent=2)
                    stream.write("\n")
            except Exception:
                output.unlink(missing_ok=True)
                raise
            print(f"{output}: {size} trajectories, {unique.size} unique, {count - unique.size} out of bag")
    return outputs


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("input_file", type=Path, help="Trusted stacked Genesis NPZ training pool")
    parser.add_argument("--output_dir", type=Path, required=True)
    parser.add_argument("--num_sets", type=int, required=True)
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--sample_size", type=int, help="Trajectories per set (default: input count)")
    parser.add_argument("--trajectory_keys", nargs="*", default=[], help="Additional per-trajectory fields to resample")
    args = parser.parse_args()
    bootstrap_motion_file(args.input_file, args.output_dir, args.num_sets,
                          seed=args.seed, sample_size=args.sample_size,
                          trajectory_keys=tuple(args.trajectory_keys))


if __name__ == "__main__":
    main()
