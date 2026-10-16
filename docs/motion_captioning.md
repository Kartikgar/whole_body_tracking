# Captioning motion segments

[Documentation index](README.md)

`scripts/vlm_caption_motion.py` creates short, chronological descriptions
for NPZ/MP4 pairs such as the clips in `data/LAFAN1_Retargeting_Dataset/g1/PgS2R-mini`.
It uses root height, body tilt, and motion changes from the NPZ to place roughly
five-second windows near movement transitions. A local video VLM describes each
window. A second text pass supplies a clip summary and short action labels.

The script needs a CUDA-enabled Python environment with `torch`, `transformers`
(with Qwen3-VL support), `opencv-python`, `Pillow`, and `numpy`. The matching
MP4 must be beside each NPZ. Qwen3-VL-4B-Instruct is the tested model; model
weights are stored locally under the ignored `artifacts/VLM/` directory.
For example, download the [Apache-2.0 checkpoint](https://huggingface.co/Qwen/Qwen3-VL-4B-Instruct)
with `huggingface_hub.snapshot_download` into
`artifacts/VLM/Qwen3-VL-4B-Instruct`.

From the `whole_body_tracking` directory, inspect the window plan without a GPU:

```bash
python scripts/vlm_caption_motion.py \
  --input-dir data/LAFAN1_Retargeting_Dataset/g1/PgS2R-mini --plan-only
```

Caption selected clips before running the full directory:

```bash
python scripts/vlm_caption_motion.py \
  --input-dir data/LAFAN1_Retargeting_Dataset/g1/PgS2R-mini \
  --motion jumps1_subject2__200.00s-220.00s \
  --motion jumps1_subject2__220.00s-240.00s \
  --motion walk1_subject1__000.00s-020.00s
```

Omit `--motion` to process all clips. Each clip gets a JSON file in
`artifacts/motion_captions/PgS2R-mini/`. Existing results are skipped, so a run
can resume; use `--overwrite` to regenerate them. `--model` accepts another
local or Hugging Face Qwen3-VL checkpoint. The model is loaded once per run.

The JSON contains a chronological `summary`, one `events` entry per planned
window, action `labels`, source-file information, and `needs_review` with
`review_reasons`. Event times are **window bounds**, not verified onset and
offset times for each action. Ground interactions are flagged for review because
floor rolls and inverted poses can be confused with jumps or handstands. The
source filename is not sent to the VLM prompt, avoiding label leakage from
names such as `jumps1`.

## Review videos and edit labels

Run the lightweight review page with any Python 3.10+ interpreter (no extra packages):

```bash
python scripts/review_motion_labels.py
```

Open **http://127.0.0.1:8765** in your browser. The default directory is PgS2R-mini.
Only original MP4s are listed; `*_compressed.mp4` duplicates are excluded.
Watch a video alongside its editable summary, comma-separated labels, and events.
Event play buttons seek to their start time. Event times can also be corrected,
and events can be added or removed. Mark a clip **Reviewed** when finished.

Click **Save** (or Ctrl/Cmd+S); switching clips also saves pending edits. Closing
the tab with unsaved changes prompts you to return and save. Edits and reviewed
status persist across restarts in
`artifacts/motion_reviews/PgS2R-mini/reviews.json`. Generated caption JSONs remain
intact; the viewer loads your saved corrections in preference to generated fields.
The separate review file is not automatically consumed by training or search scripts.

Use `--input_dir /path/to/clips`, `--review_file /path/to/reviews.json`, or
`--port 8766` to change these defaults. Stop the server with Ctrl+C. For a remote
workspace, forward port 8765 through your editor or SSH before opening the page.
