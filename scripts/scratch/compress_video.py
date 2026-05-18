"""Compress a video file with ffmpeg (H.264).

.. code-block:: bash

    python whole_body_tracking/scripts/scratch/compress_video.py \\
        whole_body_tracking/logs/rsl_rl/g1_flat/.../rl-video-step-0.mp4

    # Writes rl-video-step-0_compressed.mp4; original is untouched.
    python whole_body_tracking/scripts/scratch/compress_video.py input.mp4 --crf 28
"""

from __future__ import annotations

import argparse
import shutil
import subprocess
import sys
from pathlib import Path


def _format_bytes(num_bytes: int) -> str:
    size = float(num_bytes)
    for unit in ("B", "KB", "MB", "GB"):
        if size < 1024.0 or unit == "GB":
            return f"{size:.1f} {unit}"
        size /= 1024.0
    return f"{size:.1f} GB"


def _default_output_path(input_path: Path) -> Path:
    return input_path.with_name(f"{input_path.stem}_compressed{input_path.suffix}")


def _unique_output_path(output_path: Path) -> Path:
    """Return output_path, or the next unused sibling name if it already exists."""
    if not output_path.exists():
        return output_path

    stem = output_path.stem
    suffix = output_path.suffix
    parent = output_path.parent
    index = 1
    while True:
        candidate = parent / f"{stem}_{index}{suffix}"
        if not candidate.exists():
            return candidate
        index += 1


def compress_video(
    input_path: Path,
    output_path: Path,
    *,
    crf: int = 28,
    preset: str = "medium",
    scale: str | None = None,
) -> None:
    if not shutil.which("ffmpeg"):
        raise RuntimeError("ffmpeg not found on PATH. Install ffmpeg and retry.")

    input_path = input_path.resolve()
    output_path = output_path.resolve()

    if not input_path.is_file():
        raise FileNotFoundError(f"Input video not found: {input_path}")

    if output_path == input_path:
        raise ValueError("Output path must differ from input. The original file is never modified.")

    output_path = _unique_output_path(output_path)
    output_path.parent.mkdir(parents=True, exist_ok=True)

    cmd = [
        "ffmpeg",
        "-hide_banner",
        "-loglevel",
        "error",
        "-y",
        "-i",
        str(input_path),
        "-c:v",
        "libx264",
        "-crf",
        str(crf),
        "-preset",
        preset,
        "-pix_fmt",
        "yuv420p",
        "-movflags",
        "+faststart",
        "-c:a",
        "aac",
        "-b:a",
        "128k",
    ]

    if scale is not None:
        cmd.extend(["-vf", f"scale={scale}"])

    cmd.append(str(output_path))

    print(f"[INFO] Input:  {input_path}")
    print(f"[INFO] Output: {output_path}")
    print(f"[INFO] Running: {' '.join(cmd)}")
    subprocess.run(cmd, check=True)

    input_size = input_path.stat().st_size
    output_size = output_path.stat().st_size
    ratio = 100.0 * (1.0 - output_size / input_size) if input_size else 0.0

    print(f"[INFO] Input size:  {_format_bytes(input_size)}")
    print(f"[INFO] Output size: {_format_bytes(output_size)}")
    print(f"[INFO] Saved ~{ratio:.1f}%")


def main() -> None:
    parser = argparse.ArgumentParser(description="Compress a video with ffmpeg (H.264).")
    parser.add_argument("input", type=Path, help="Path to the input video.")
    parser.add_argument(
        "-o",
        "--output",
        type=Path,
        default=None,
        help="Output path (default: new <input_stem>_compressed<suffix> next to input).",
    )
    parser.add_argument(
        "--crf",
        type=int,
        default=28,
        help="Quality (0=lossless, 51=worst). Lower is better. Default: 28.",
    )
    parser.add_argument(
        "--preset",
        type=str,
        default="medium",
        choices=["ultrafast", "superfast", "veryfast", "faster", "fast", "medium", "slow", "slower", "veryslow"],
        help="x264 speed/efficiency tradeoff. Default: medium.",
    )
    parser.add_argument(
        "--scale",
        type=str,
        default=None,
        help='Optional scale filter, e.g. "1280:-2" to cap width at 1280.',
    )
    args = parser.parse_args()

    input_path: Path = args.input.expanduser()
    output_path: Path = (args.output or _default_output_path(input_path)).expanduser()

    try:
        compress_video(
            input_path,
            output_path,
            crf=args.crf,
            preset=args.preset,
            scale=args.scale,
        )
    except (FileNotFoundError, ValueError, RuntimeError) as exc:
        print(f"[ERROR] {exc}", file=sys.stderr)
        sys.exit(1)
    except subprocess.CalledProcessError as exc:
        print(f"[ERROR] ffmpeg failed with exit code {exc.returncode}", file=sys.stderr)
        sys.exit(exc.returncode)


if __name__ == "__main__":
    main()
