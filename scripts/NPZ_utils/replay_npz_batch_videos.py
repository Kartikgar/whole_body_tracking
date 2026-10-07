"""Export each NPZ motion in a directory using a separate Isaac Sim process.

Run with the Isaac Lab Python environment, for example::

    python scripts/replay_npz_batch_videos.py \
        --input_dir data/LAFAN1_Retargeting_Dataset/g1/PgS2R-mini

Videos are saved beside the motions. Existing, verified videos are skipped.
"""

import argparse
import json
import os
import signal
import subprocess
import sys
import time
from pathlib import Path

import numpy as np


ROOT = Path(__file__).resolve().parents[1]
REPLAY_SCRIPT = ROOT / "scripts/replay_npz.py"


def video_info(path: Path) -> tuple[int, float] | None:
    if not path.is_file():
        return None
    try:
        result = subprocess.run(
            [
                "ffprobe", "-v", "error", "-select_streams", "v:0",
                "-show_entries", "stream=nb_frames,avg_frame_rate",
                "-of", "json", str(path),
            ],
            capture_output=True, text=True, timeout=10, check=True,
        )
        stream = json.loads(result.stdout)["streams"][0]
        numerator, denominator = map(int, stream["avg_frame_rate"].split("/"))
        return int(stream["nb_frames"]), numerator / denominator
    except (OSError, subprocess.SubprocessError, ValueError, KeyError, IndexError, ZeroDivisionError):
        return None


def motion_info(path: Path) -> tuple[int, float]:
    with np.load(path, allow_pickle=False) as motion:
        return int(motion["joint_pos"].shape[0]), float(np.asarray(motion["fps"]).reshape(-1)[0])


def stop_child(process: subprocess.Popen) -> None:
    if process.poll() is not None:
        return
    os.killpg(process.pid, signal.SIGTERM)
    try:
        process.wait(timeout=10)
    except subprocess.TimeoutExpired:
        os.killpg(process.pid, signal.SIGKILL)
        process.wait()


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--input_dir", type=Path, required=True)
    parser.add_argument("--output_dir", type=Path, help="Defaults to the input directory.")
    parser.add_argument("--limit", type=int, help="Process at most this many motions (for a trial run).")
    parser.add_argument("--shutdown_grace_s", type=float, default=15.0)
    parser.add_argument("--max_seconds_per_motion", type=float, default=1800.0)
    args = parser.parse_args()
    input_dir = args.input_dir.expanduser().resolve()
    output_dir = (args.output_dir or input_dir).expanduser().resolve()
    motions = sorted(input_dir.glob("*.npz"))
    if not motions:
        parser.error(f"No NPZ motions found in {input_dir}")
    if args.limit is not None:
        if args.limit < 1:
            parser.error("--limit must be positive")
        motions = motions[:args.limit]
    output_dir.mkdir(parents=True, exist_ok=True)
    log_dir = output_dir / "_video_logs"
    log_dir.mkdir(exist_ok=True)
    failures = []

    for index, motion in enumerate(motions, 1):
        expected = motion_info(motion)
        video = output_dir / f"{motion.stem}.mp4"
        actual = video_info(video)
        if actual == expected:
            print(f"[{index}/{len(motions)}] Already verified: {video.name}", flush=True)
            continue
        if video.exists():
            failures.append(f"{video}: existing video is incomplete or has the wrong frame rate")
            print(f"[{index}/{len(motions)}] Invalid existing video: {video.name}", flush=True)
            continue

        log_path = log_dir / f"{motion.stem}.log"
        command = [
            sys.executable, str(REPLAY_SCRIPT), "--motion_file", str(motion),
            "--video_file", str(video), "--headless",
        ]
        print(f"[{index}/{len(motions)}] Recording {motion.name}", flush=True)
        with log_path.open("w", encoding="utf-8") as log:
            process = subprocess.Popen(
                command, cwd=ROOT, stdout=log, stderr=subprocess.STDOUT,
                start_new_session=True,
            )
            start = time.monotonic()
            finished_at = None
            forced_shutdown = False
            try:
                while process.poll() is None:
                    now = time.monotonic()
                    if finished_at is None and video_info(video) == expected:
                        finished_at = now
                    if finished_at is not None and now - finished_at >= args.shutdown_grace_s:
                        forced_shutdown = True
                        stop_child(process)
                        break
                    if now - start >= args.max_seconds_per_motion:
                        stop_child(process)
                        break
                    time.sleep(1)
            except BaseException:
                stop_child(process)
                raise

        if video_info(video) == expected:
            note = " (child shutdown timed out)" if forced_shutdown else ""
            print(f"[{index}/{len(motions)}] Saved {video.name}: {expected[0]} frames at {expected[1]:g} fps{note}", flush=True)
        else:
            failures.append(f"{motion.name}: video missing or invalid; see {log_path}")
            print(f"[{index}/{len(motions)}] FAILED: see {log_path}", flush=True)

    if failures:
        raise SystemExit("\n".join(failures))
    print(f"Verified all {len(motions)} videos in {output_dir}", flush=True)


if __name__ == "__main__":
    main()
