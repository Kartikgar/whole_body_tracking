"""Generate timestamped action descriptions for motion NPZ/MP4 pairs.

Run in a Python environment with PyTorch, Transformers, OpenCV, and a CUDA GPU::

    python scripts/caption_motion_segments.py \
        --input-dir data/LAFAN1_Retargeting_Dataset/g1/PgS2R-mini \
        --model artifacts/models/Qwen3-VL-4B-Instruct

The model stays loaded while clips are processed. Results are one JSON file per
motion under artifacts/motion_captions/<input directory name>/ by default.
"""

from __future__ import annotations

import argparse
import json
import math
import os
import re
from datetime import datetime, timezone
from pathlib import Path

import cv2
import numpy as np


ROOT = Path(__file__).resolve().parents[1]
DEFAULT_MODEL = ROOT / "artifacts/models/Qwen3-VL-4B-Instruct"
PROMPT_VERSION = "motion-events-v5"


def _smooth(values: np.ndarray, radius: int) -> np.ndarray:
    if radius < 1:
        return values
    padded = np.pad(values, (radius, radius), mode="edge")
    return np.convolve(padded, np.ones(2 * radius + 1) / (2 * radius + 1), mode="valid")


def motion_features(npz_path: Path) -> dict:
    """Extract simple, interpretable kinematic signals from one motion clip."""
    with np.load(npz_path, allow_pickle=False) as data:
        fps = float(np.asarray(data["fps"]).reshape(-1)[0])
        root_pos = np.asarray(data["body_pos_w"][:, 0], dtype=np.float64)
        root_quat = np.asarray(data["body_quat_w"][:, 0], dtype=np.float64)
        root_ang_vel = np.asarray(data["body_ang_vel_w"][:, 0], dtype=np.float64)
        joint_vel = np.asarray(data["joint_vel"], dtype=np.float64)
    if not math.isfinite(fps) or fps <= 0 or len(root_pos) < 2:
        raise ValueError(f"Invalid frame rate or motion length in {npz_path}")

    root_speed = np.linalg.norm(np.gradient(root_pos[:, :2], 1 / fps, axis=0), axis=1)
    up_z = np.clip(1 - 2 * (root_quat[:, 1] ** 2 + root_quat[:, 2] ** 2), -1, 1)
    tilt_deg = np.degrees(np.arccos(up_z))
    angular_speed = np.linalg.norm(root_ang_vel, axis=1)
    joint_activity = np.mean(np.abs(joint_vel), axis=1)
    radius = max(1, round(0.15 * fps))
    tracks = np.stack(
        [
            _smooth(root_pos[:, 2], radius),
            _smooth(tilt_deg, radius),
            _smooth(root_speed, radius),
            _smooth(angular_speed, radius),
            _smooth(joint_activity, radius),
        ],
        axis=1,
    )
    # Compare nearby states in units meaningful for humanoid motion.
    scale = np.array([0.18, 25.0, 0.8, 1.5, 0.8])
    look = max(1, round(0.35 * fps))
    before = tracks[np.maximum(np.arange(len(tracks)) - look, 0)]
    after = tracks[np.minimum(np.arange(len(tracks)) + look, len(tracks) - 1)]
    change_score = np.linalg.norm((after - before) / scale, axis=1)
    return {
        "fps": fps,
        "frames": len(root_pos),
        "duration_s": len(root_pos) / fps,
        "change_score": change_score,
        "root_height": root_pos[:, 2],
        "min_root_height_m": float(root_pos[:, 2].min()),
        "max_root_height_m": float(root_pos[:, 2].max()),
        "max_tilt_deg": float(tilt_deg.max()),
        "mean_root_speed_mps": float(root_speed.mean()),
    }


def plan_windows(features: dict, target_s: float = 5.0, context_s: float = 0.75) -> list[dict]:
    """Cover the clip with short cores, shifting boundaries toward kinematic changes."""
    duration = features["duration_s"]
    fps = features["fps"]
    score = features["change_score"]
    boundaries = [0.0]
    number_of_windows = max(1, round(duration / target_s))
    for index in range(1, number_of_windows):
        target = index * duration / number_of_windows
        lo = max(boundaries[-1] + 2.5, target - 1.5)
        hi = min(duration - 2.5, target + 1.5)
        if lo < hi:
            first = math.ceil(lo * fps)
            last = min(len(score), math.floor(hi * fps) + 1)
            best = first + int(np.argmax(score[first:last]))
            candidate = best / fps if score[best] >= 1.0 else target
        else:
            candidate = target
        boundaries.append(round(candidate, 3))
    boundaries.append(round(duration, 3))
    return [
        {
            "start_s": boundaries[i],
            "end_s": boundaries[i + 1],
            "context_start_s": round(max(0.0, boundaries[i] - context_s), 3),
            "context_end_s": round(min(duration, boundaries[i + 1] + context_s), 3),
        }
        for i in range(len(boundaries) - 1)
    ]


def sample_video_frames(video_path: Path, start_s: float, end_s: float, sample_fps: float, max_width: int):
    """Read evenly spaced RGB frames from the matching rendered video."""
    from PIL import Image
    from transformers.video_utils import VideoMetadata

    capture = cv2.VideoCapture(str(video_path))
    if not capture.isOpened():
        raise RuntimeError(f"Cannot open video: {video_path}")
    native_fps = capture.get(cv2.CAP_PROP_FPS)
    if native_fps <= 0:
        capture.release()
        raise ValueError(f"Video has no frame rate: {video_path}")
    frame_count = int(capture.get(cv2.CAP_PROP_FRAME_COUNT))
    if frame_count <= 0:
        capture.release()
        raise ValueError(f"Video has no frames: {video_path}")
    frames = []
    try:
        sample_fps = max(sample_fps, 4 / (end_s - start_s))
        # Sample at interval midpoints; frame zero can show Isaac's previous pose
        # while its render pipeline warms up.
        timestamps = np.arange(start_s + 0.5 / sample_fps, end_s - 1e-6, 1 / sample_fps)
        if len(timestamps) < 4:
            timestamps = np.linspace(start_s, max(start_s, end_s - 1 / native_fps), 4)
        for timestamp in timestamps:
            # A timestamp very near the clip end can round up to frame_count
            # (e.g. 19.99 * 50 -> frame 1000 in a 1000-frame video).
            frame_index = min(frame_count - 1, max(0, int(float(timestamp) * native_fps)))
            capture.set(cv2.CAP_PROP_POS_FRAMES, frame_index)
            ok, frame = capture.read()
            if not ok:
                raise RuntimeError(f"Could not read {video_path} at {timestamp:.2f}s")
            height, width = frame.shape[:2]
            if width > max_width:
                frame = cv2.resize(frame, (max_width, round(height * max_width / width)), interpolation=cv2.INTER_AREA)
            frames.append(Image.fromarray(cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)))
    finally:
        capture.release()
    metadata = VideoMetadata(
        total_num_frames=len(frames), fps=sample_fps, width=frames[0].width,
        height=frames[0].height, duration=len(frames) / sample_fps,
        frames_indices=list(range(len(frames))),
    )
    return frames, metadata


def parse_object(response: str) -> dict:
    """Accept a JSON object even if the model surrounds it with a code fence."""
    first = response.find("{")
    last = response.rfind("}")
    if first < 0 or last <= first:
        raise ValueError("Model response did not contain a JSON object")
    result = json.loads(response[first : last + 1])
    if not isinstance(result, dict):
        raise ValueError("Model response was not a JSON object")
    return result


class VideoCaptioner:
    def __init__(self, model_path: str, max_new_tokens: int):
        import torch
        from transformers import AutoProcessor, Qwen3VLForConditionalGeneration

        if not torch.cuda.is_available():
            raise RuntimeError("Video captioning requires CUDA. Use --plan-only to inspect windows without a GPU.")
        self.torch = torch
        self.processor = AutoProcessor.from_pretrained(model_path)
        self.model = Qwen3VLForConditionalGeneration.from_pretrained(
            model_path, dtype=torch.bfloat16, attn_implementation="sdpa"
        ).to("cuda").eval()
        self.max_new_tokens = max_new_tokens

    def generate(self, messages: list[dict], video_metadata=None, max_new_tokens=None) -> str:
        kwargs = {}
        if video_metadata is not None:
            kwargs["processor_kwargs"] = {
                "video_metadata": [video_metadata], "cap_pixels_per_frame": True,
            }
        inputs = self.processor.apply_chat_template(
            messages, tokenize=True, add_generation_prompt=True,
            return_dict=True, return_tensors="pt", **kwargs,
        ).to(self.model.device)
        with self.torch.inference_mode():
            output = self.model.generate(
                **inputs, max_new_tokens=max_new_tokens or self.max_new_tokens,
                do_sample=False,
            )
        answer = self.processor.batch_decode(
            [output[0, inputs["input_ids"].shape[-1] :]], skip_special_tokens=True
        )[0]
        return answer.strip()

    def caption_window(self, frames, metadata, grounded_hint: bool = False) -> str:
        prompt = (
            "Describe the humanoid robot's visible body movements over this short video in one or two "
            "concrete sentences. Describe posture and changes. If it stays on the ground, say so. "
            "Do not infer intent."
        )
        if grounded_hint:
            prompt += (
                " The accompanying motion data shows its pelvis remains below 0.5 meters. "
                "Check whether a rotation is a ground roll or an inverted floor pose rather than "
                "an airborne flip, jump, or handstand."
            )
        messages = [{"role": "user", "content": [
            {"type": "video", "video": frames}, {"type": "text", "text": prompt},
        ]}]
        return self.generate(messages, video_metadata=metadata, max_new_tokens=120)

    def organize(self, events: list[dict]) -> str:
        observations = [
            f"{event['start_s']:.2f}-{event['end_s']:.2f}s: {event['description']}"
            for event in events
        ]
        prompt = (
            "Use only these observations of a humanoid robot to produce JSON with keys: "
            "'summary' (one or two chronological sentences), 'window_labels' (one short action "
            "phrase for each observation in order), 'tags' (short action names present anywhere), "
            "and 'uncertain' (boolean). Preserve distinct actions such as rolling and getting up. "
            "Use 'get up' only if the robot reaches standing. Do not infer intent or add actions "
            "absent from the observations. Return only JSON.\n"
            + "\n".join(observations)
        )
        return self.generate([{"role": "user", "content": prompt}], max_new_tokens=240)


def source_interval_from_name(path: Path) -> list[float] | None:
    match = re.search(r"__(\d+(?:\.\d+)?)s-(\d+(?:\.\d+)?)s$", path.stem)
    return [float(match.group(1)), float(match.group(2))] if match else None


def caption_motion(motion: Path, video: Path, features: dict, windows: list[dict],
                   captioner: VideoCaptioner, args) -> dict:
    review_reasons = []
    events = []
    for window in windows:
        frames, metadata = sample_video_frames(
            video, window["context_start_s"], window["context_end_s"],
            args.sample_fps, args.max_width,
        )
        description = captioner.caption_window(frames, metadata).strip()
        first_frame = max(0, round(window["start_s"] * features["fps"]))
        last_frame = min(features["frames"], math.ceil(window["end_s"] * features["fps"]))
        window_max_height = float(features["root_height"][first_frame:last_frame].max())
        suspect_aerial = any(
            term in description.lower() for term in
            ("backflip", "somersault", "airborne", "jumps", "jumping", "handstand")
        ) and window_max_height < 0.5
        if suspect_aerial:
            description = captioner.caption_window(frames, metadata, grounded_hint=True).strip()
            if any(term in description.lower() for term in
                   ("backflip", "somersault", "airborne", "jumps", "jumping", "handstand")):
                review_reasons.append("aerial_claim_conflicts_with_low_root_height")
        if not description:
            description = "Movement unclear."
            review_reasons.append("empty_window_caption")
        events.append({"start_s": window["start_s"], "end_s": window["end_s"],
                       "label": "unlabeled", "description": description,
                       "sampled_frames": len(frames)})
    organized_raw = captioner.organize(events)
    try:
        organized = parse_object(organized_raw)
        summary = str(organized["summary"]).strip()
        labels = organized["window_labels"]
        tags = organized["tags"]
        if not summary or not isinstance(labels, list) or len(labels) != len(events):
            raise ValueError("Missing summary or mismatched window labels")
        if not isinstance(tags, list):
            raise ValueError("Tags must be a list")
        for event, label in zip(events, labels):
            event["label"] = str(label).strip().lower()[:80] or "unclear"
        tags = [str(tag).strip().lower()[:80] for tag in tags if str(tag).strip()]
        if organized.get("uncertain"):
            review_reasons.append("model_uncertain")
    except (ValueError, KeyError, TypeError, json.JSONDecodeError) as exc:
        review_reasons.append(f"invalid_summary_json: {exc}")
        summary = " ".join(event["description"] for event in events)
        tags = []
    if features["max_root_height_m"] < 0.5:
        claims = (summary + " " + " ".join(tags)).lower()
        if any(term in claims for term in ("backflip", "somersault", "airborne", "jump", "handstand", "flip")):
            review_reasons.append("aerial_summary_conflicts_with_low_root_height")
        if "get up" in claims or "stands up" in claims:
            review_reasons.append("standing_claim_conflicts_with_low_root_height")
        unsupported_tags = ("get up", "stand", "flip", "jump", "airborne", "somersault")
        tags = [tag for tag in tags if not any(term in tag for term in unsupported_tags)]
    if features["max_tilt_deg"] > 60 or features["min_root_height_m"] < 0.35:
        review_reasons.append("complex_ground_interaction")
        wording = " ".join(event["description"].lower() for event in events)
        if not any(term in wording for term in ("ground", "floor", "roll", "lie", "lying", "fall")):
            review_reasons.append("ground_interaction_not_described")
    return {
        "motion_file": str(motion), "video_file": str(video),
        "source_interval_s": source_interval_from_name(motion),
        "duration_s": round(features["duration_s"], 3), "fps": features["fps"],
        "model": args.model, "prompt_version": PROMPT_VERSION,
        "sample_fps": args.sample_fps, "max_frame_width": args.max_width,
        "created_utc": datetime.now(timezone.utc).isoformat(),
        "motion_signals": {
            key: features[key] for key in
            ("min_root_height_m", "max_root_height_m", "max_tilt_deg", "mean_root_speed_mps")
        },
        "summary": summary, "labels": sorted(set(tags)),
        "events": events, "windows": windows,
        "organization_raw_response": organized_raw,
        "needs_review": bool(review_reasons),
        "review_reasons": sorted(set(review_reasons)),
    }


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--input-dir", type=Path, required=True)
    parser.add_argument("--output-dir", type=Path)
    parser.add_argument("--model", default=str(DEFAULT_MODEL))
    parser.add_argument("--motion", action="append", help="NPZ filename or stem; repeat to select clips.")
    parser.add_argument("--limit", type=int)
    parser.add_argument("--window-s", type=float, default=5.0)
    parser.add_argument("--context-s", type=float, default=0.0)
    parser.add_argument("--sample-fps", type=float, default=2.0)
    parser.add_argument("--max-width", type=int, default=480)
    parser.add_argument("--max-new-tokens", type=int, default=240)
    parser.add_argument("--plan-only", action="store_true")
    parser.add_argument("--overwrite", action="store_true")
    args = parser.parse_args()
    if args.window_s < 3 or args.context_s < 0 or args.sample_fps <= 0 or args.max_width < 64:
        parser.error("Invalid window, context, sampling rate, or frame width")
    input_dir = args.input_dir.expanduser().resolve()
    output_dir = (args.output_dir or ROOT / "artifacts/motion_captions" / input_dir.name).expanduser().resolve()
    motions = sorted(input_dir.glob("*.npz"))
    if args.motion:
        selected = {name[:-4] if name.endswith(".npz") else name for name in args.motion}
        motions = [motion for motion in motions if motion.stem in selected]
        if len(motions) != len(selected):
            parser.error("Some --motion selections were not found in --input-dir")
    if args.limit is not None:
        if args.limit < 1:
            parser.error("--limit must be positive")
        motions = motions[:args.limit]
    if not motions:
        parser.error(f"No NPZ motions found in {input_dir}")

    pending = []
    for motion in motions:
        video = motion.with_suffix(".mp4")
        if not video.is_file():
            raise FileNotFoundError(f"Matching video missing for {motion}: {video}")
        output = output_dir / f"{motion.stem}.json"
        if output.exists() and not args.overwrite and not args.plan_only:
            print(f"Skipping existing: {output}", flush=True)
            continue
        features = motion_features(motion)
        windows = plan_windows(features, args.window_s, args.context_s)
        if args.plan_only:
            print(json.dumps({"motion": motion.name, "duration_s": features["duration_s"],
                              "windows": windows,
                              "min_root_height_m": features["min_root_height_m"],
                              "max_tilt_deg": features["max_tilt_deg"]}), flush=True)
        else:
            pending.append((motion, video, output, features, windows))
    if args.plan_only or not pending:
        return

    output_dir.mkdir(parents=True, exist_ok=True)
    captioner = VideoCaptioner(args.model, args.max_new_tokens)
    for index, (motion, video, output, features, windows) in enumerate(pending, 1):
        print(f"[{index}/{len(pending)}] Captioning {motion.name} ({len(windows)} windows)", flush=True)
        result = caption_motion(motion, video, features, windows, captioner, args)
        temporary = output.with_suffix(".json.tmp")
        temporary.write_text(json.dumps(result, indent=2) + "\n", encoding="utf-8")
        os.replace(temporary, output)
        print(f"Saved {output}: {result['summary']}", flush=True)


if __name__ == "__main__":
    main()
