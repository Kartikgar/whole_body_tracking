#!/usr/bin/env python3
"""Review local motion videos and captions. Uses only the Python standard library."""

import argparse
import json
import math
import mimetypes
import re
import threading
from datetime import datetime, timezone
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer
from pathlib import Path
from urllib.parse import unquote, urlsplit

ROOT = Path(__file__).resolve().parents[1]


class ReviewStore:
    def __init__(self, input_dir, review_file):
        self.input_dir = Path(input_dir).resolve()
        self.review_file = Path(review_file).resolve()
        self.videos = {
            p.stem: p for p in sorted(self.input_dir.glob("*.mp4"))
            if not p.stem.endswith("_compressed")
        }
        if not self.videos:
            raise ValueError(f"No original MP4 videos found in {self.input_dir}")
        # Never replace a generated caption file with a review database.
        if self.review_file in {p.with_suffix(".json") for p in self.videos.values()}:
            raise ValueError("The review file must be separate from generated captions")
        self.reviews = json.loads(self.review_file.read_text()) if self.review_file.exists() else {}
        if not isinstance(self.reviews, dict):
            raise ValueError("Review file must contain a JSON object")
        self.lock = threading.Lock()

    def clip(self, clip_id):
        video = self.videos[clip_id]
        caption = video.with_suffix(".json")
        source = json.loads(caption.read_text()) if caption.exists() else {}
        with self.lock:
            review = self.reviews.get(clip_id, {}).copy()
        return {
            "id": clip_id,
            "duration_s": source.get("duration_s"),
            "source_interval_s": source.get("source_interval_s"),
            "needs_review": source.get("needs_review", False),
            "review_reasons": source.get("review_reasons", []),
            "summary": review.get("summary", source.get("summary", "")),
            "labels": review.get("labels", source.get("labels", [])),
            "events": review.get("events", source.get("events", [])),
            "reviewed": review.get("reviewed", False),
            "updated_at": review.get("updated_at"),
        }

    def catalog(self):
        with self.lock:
            return [{"id": key, "reviewed": self.reviews.get(key, {}).get("reviewed", False)}
                    for key in self.videos]

    def save(self, clip_id, payload):
        if clip_id not in self.videos:
            raise KeyError(clip_id)
        if not isinstance(payload, dict):
            raise ValueError("Expected a JSON object")
        summary, labels, events = (payload.get(key) for key in ("summary", "labels", "events"))
        if not isinstance(summary, str) or not isinstance(labels, list) or not all(isinstance(x, str) for x in labels):
            raise ValueError("Expected a summary string and a list of label strings")
        if not isinstance(events, list) or not isinstance(payload.get("reviewed"), bool):
            raise ValueError("Expected an events list and a reviewed boolean")
        cleaned_events = []
        for event in events:
            if not isinstance(event, dict):
                raise ValueError("Each event must be an object")
            start, end = event.get("start_s"), event.get("end_s")
            if (type(start) not in (int, float) or type(end) not in (int, float)
                    or not math.isfinite(start) or not math.isfinite(end) or not 0 <= start < end):
                raise ValueError("Event times must be finite with 0 <= start < end")
            if not all(isinstance(event.get(key), str) for key in ("label", "description")):
                raise ValueError("Each event needs a label and description string")
            cleaned_events.append({key: event[key] for key in ("start_s", "end_s", "label", "description")})
        review = dict(summary=summary, labels=labels, events=cleaned_events, reviewed=payload["reviewed"],
                      updated_at=datetime.now(timezone.utc).isoformat())
        with self.lock:
            updated = {**self.reviews, clip_id: review}
            self.review_file.parent.mkdir(parents=True, exist_ok=True)
            temporary = self.review_file.with_suffix(self.review_file.suffix + ".tmp")
            temporary.write_text(json.dumps(updated, indent=2, ensure_ascii=False) + "\n")
            temporary.replace(self.review_file)
            self.reviews = updated
        return review


def make_handler(store):
    class Handler(BaseHTTPRequestHandler):
        def json_response(self, value, status=200):
            data = json.dumps(value, allow_nan=False).encode()
            self.send_response(status)
            self.send_header("Content-Type", "application/json; charset=utf-8")
            self.send_header("Content-Length", str(len(data)))
            self.send_header("Cache-Control", "no-store")
            self.end_headers()
            self.wfile.write(data)

        def do_GET(self):
            path = unquote(urlsplit(self.path).path)
            try:
                if path == "/":
                    self.send_file(Path(__file__).with_suffix(".html"))
                elif path == "/api/clips":
                    self.json_response(store.catalog())
                elif path.startswith("/api/clip/"):
                    self.json_response(store.clip(path.removeprefix("/api/clip/")))
                elif path.startswith("/video/"):
                    self.send_file(store.videos[path.removeprefix("/video/")])
                else:
                    self.send_error(404)
            except KeyError:
                self.json_response({"error": "Unknown clip"}, 404)
            except (OSError, ValueError) as exc:
                self.json_response({"error": str(exc)}, 500)

        def do_POST(self):
            path = unquote(urlsplit(self.path).path)
            if not path.startswith("/api/clip/"):
                self.send_error(404)
                return
            # The app is local; reject browser requests originating on another site.
            origin = self.headers.get("Origin")
            if origin and origin != f"http://{self.headers.get('Host')}":
                self.json_response({"error": "Cross-origin writes are not allowed"}, 403)
                return
            try:
                size = int(self.headers.get("Content-Length", "0"))
                if not 0 < size <= 1024 * 1024:
                    raise ValueError("Expected a JSON body of at most 1 MiB")
                payload = json.loads(self.rfile.read(size))
                self.json_response(store.save(path.removeprefix("/api/clip/"), payload))
            except KeyError:
                self.json_response({"error": "Unknown clip"}, 404)
            except (ValueError, TypeError) as exc:
                self.json_response({"error": str(exc)}, 400)
            except OSError as exc:
                self.json_response({"error": f"Could not save: {exc}"}, 500)

        def send_file(self, path):
            # Range requests let the browser seek without downloading an entire video.
            size = path.stat().st_size
            start, end = 0, size - 1
            requested = self.headers.get("Range")
            if requested:
                match = re.fullmatch(r"bytes=(\d*)-(\d*)", requested)
                if not match or not any(match.groups()):
                    self.send_range_error(size)
                    return
                first, last = match.groups()
                if first:
                    start = int(first)
                    end = min(int(last), end) if last else end
                else:
                    start = max(0, size - int(last))
                if start > end or start >= size:
                    self.send_range_error(size)
                    return
            self.send_response(206 if requested else 200)
            self.send_header("Content-Type", mimetypes.guess_type(path.name)[0] or "application/octet-stream")
            self.send_header("Accept-Ranges", "bytes")
            self.send_header("Content-Length", str(end - start + 1))
            if requested:
                self.send_header("Content-Range", f"bytes {start}-{end}/{size}")
            self.end_headers()
            try:
                with path.open("rb") as stream:
                    stream.seek(start)
                    remaining = end - start + 1
                    while remaining:
                        chunk = stream.read(min(remaining, 256 * 1024))
                        if not chunk:
                            break
                        self.wfile.write(chunk)
                        remaining -= len(chunk)
            except (BrokenPipeError, ConnectionResetError):
                pass  # Normal when a video is paused or replaced by the next clip.

        def send_range_error(self, size):
            self.send_response(416)
            self.send_header("Content-Range", f"bytes */{size}")
            self.send_header("Content-Length", "0")
            self.end_headers()

    return Handler


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--input_dir", type=Path,
                        default=ROOT / "data/LAFAN1_Retargeting_Dataset/g1/PgS2R-mini")
    parser.add_argument("--review_file", type=Path, help="Separate JSON file for saved edits.")
    parser.add_argument("--port", type=int, default=8765)
    args = parser.parse_args()
    review_file = args.review_file or ROOT / "artifacts/motion_reviews" / args.input_dir.name / "reviews.json"
    store = ReviewStore(args.input_dir, review_file)
    server = ThreadingHTTPServer(("127.0.0.1", args.port), make_handler(store))
    print(f"Review {len(store.videos)} clips at http://127.0.0.1:{server.server_port}", flush=True)
    print(f"Edits saved to {store.review_file}", flush=True)
    try:
        server.serve_forever()
    except KeyboardInterrupt:
        pass
    finally:
        server.server_close()


if __name__ == "__main__":
    main()
