#!/usr/bin/env python3
"""Place a synced time-series panel next to a wrench-arrow video (``play.py --wrench_arrows`` output).

Panel (from ``wrench_log.npz``, env 0): applied (gated) world-frame pelvis force F_z with the payload-weight line -m*g,
the applied torque magnitude, and the gate exp(-lambda*u); a cursor marks the current video frame (frame k = step k).
The static plot is drawn once; per frame only the cursor is painted, so composing 1000 frames takes seconds.
"""
import argparse
import subprocess

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402
import numpy as np  # noqa: E402

p = argparse.ArgumentParser()
p.add_argument("--video", required=True)
p.add_argument("--log", required=True)
p.add_argument("--output", required=True)
p.add_argument("--payload_kg", type=float, default=12.5)
p.add_argument("--title", default="")
p.add_argument("--panel_width", type=int, default=760)
p.add_argument("--required_npz", default=None,
               help="optional required_wrench.npz (F_req, tau_req per step): overlay the exact payload wrench in grey")
p.add_argument("--font_scale", type=float, default=1.0, help="multiply all panel font sizes (e.g. 1.6 for slides)")
a = p.parse_args()

probe = subprocess.run(["ffprobe", "-v", "error", "-select_streams", "v:0", "-show_entries", "stream=width,height,nb_frames",
                        "-of", "csv=p=0", a.video], capture_output=True, text=True, check=True).stdout.strip().split(",")
W, H = int(probe[0]), int(probe[1])
d = np.load(a.log)
fz = d["force_world"][0, :, 2]
tau = np.linalg.norm(d["torque_world"][0], axis=-1)
gate = d["gate"][0]
fps = float(d["fps"])
t = np.arange(len(fz)) / fps
weight = a.payload_kg * 9.81

PW, dpi, fs = a.panel_width, 100, a.font_scale
plt.rcParams.update({"font.size": 15 * fs, "axes.spines.top": False, "axes.spines.right": False})
fig, axes = plt.subplots(3, 1, figsize=(PW / dpi, H / dpi), dpi=dpi, sharex=True,
                         gridspec_kw=dict(height_ratios=[1.6, 1, 1]))
fig.subplots_adjust(left=0.17 + 0.05 * (fs - 1), right=0.97, top=0.86 - 0.18 * (fs - 1), bottom=0.1 + 0.03 * (fs - 1),
                    hspace=0.35 + 0.15 * (fs - 1))
ax_f, ax_t, ax_g = axes
REQ = "#4b5563"
req = np.load(a.required_npz) if a.required_npz else None
if req is not None:  # exact payload wrench along this rollout (same step indexing as the log)
    fz_req = req["F_req"][: len(t), 2]; tau_req = np.linalg.norm(req["tau_req"][: len(t)], axis=-1)
    ax_f.plot(t[: len(fz_req)], fz_req, color=REQ, lw=1.4 * fs, label="required $F_z$ (exact payload)")
    ax_f.axhline(-weight, color="0.6", ls=(0, (6, 3)), lw=1.2 * fs)
else:
    ax_f.axhline(-weight, color="#555555", ls=(0, (6, 3)), lw=2 * fs, label=f"payload weight $-mg$ ({-weight:.0f} N)")
ax_f.plot(t, fz, color="#f07318", lw=2.2 * fs, label="applied $F_z$ (world)")
ax_f.axhline(0, color="0.75", lw=1)
lo = min(-weight * 1.15, np.nanmin(fz) * 1.1, np.nanmin(fz_req) * 1.05 if req is not None else 0)
ax_f.set_ylabel("Force (N)"); ax_f.set_ylim(lo, max(40, np.nanmax(fz) * 1.1))
ax_f.legend(frameon=False, fontsize=12 * fs, loc="lower center", bbox_to_anchor=(0.5, 1.0), ncol=2 if fs < 1.2 else 1)
ax_t.plot(t, tau, color="#9933e6", lw=2.2 * fs, label="applied")
if req is not None:
    ax_t.plot(t[: len(tau_req)], tau_req, color=REQ, lw=1.4 * fs, label="required")
    ax_t.legend(frameon=False, fontsize=11 * fs, loc="upper left", ncol=2, borderaxespad=0.1, handlelength=1.2)
ax_t.set_ylabel("|torque|\n(N·m)"); ax_t.set_ylim(0, max(np.nanmax(tau) * (1.35 if req is not None else 1.05), 1))
ax_g.plot(t, gate, color="#2563eb", lw=2.2 * fs); ax_g.set_ylabel("gate"); ax_g.set_ylim(0, 1.05); ax_g.set_xlabel("time (s)")
for ax in axes:
    ax.grid(alpha=0.3); ax.set_xlim(0, t[-1])
fig.suptitle(a.title or "Applied pelvis wrench (after gate)", fontsize=16 * fs)
fig.canvas.draw()
bg = np.asarray(fig.canvas.buffer_rgba())[..., :3].copy()
ph, pw = bg.shape[:2]
# pixel geometry of each axes (matplotlib's origin is bottom-left; image rows are top-down)
boxes = [ax.get_window_extent() for ax in axes]
x0, x1 = boxes[0].x0, boxes[0].x1


def cursor_x(sec):
    return int(round(x0 + (x1 - x0) * sec / t[-1]))


dec = subprocess.Popen(["ffmpeg", "-loglevel", "error", "-i", a.video, "-f", "rawvideo", "-pix_fmt", "rgb24", "-"],
                       stdout=subprocess.PIPE)
enc = subprocess.Popen(["ffmpeg", "-loglevel", "error", "-y", "-f", "rawvideo", "-pix_fmt", "rgb24", "-s", f"{W + pw}x{H}",
                        "-r", str(fps), "-i", "-", "-c:v", "libx264", "-pix_fmt", "yuv420p", "-crf", "20", a.output],
                       stdin=subprocess.PIPE)
k = 0
while True:
    buf = dec.stdout.read(W * H * 3)
    if len(buf) < W * H * 3:
        break
    frame = np.frombuffer(buf, np.uint8).reshape(H, W, 3)
    panel = bg.copy()
    cx = cursor_x(min(k, len(t) - 1) / fps)
    for b in boxes:
        r0, r1 = int(ph - b.y1), int(ph - b.y0)
        panel[r0:r1, max(cx - 1, 0):cx + 2] = (30, 30, 30)
    out = np.concatenate([frame, np.full((H, pw, 3), 255, np.uint8) if panel.shape[0] != H else panel], 1)
    enc.stdin.write(out.tobytes())
    k += 1
enc.stdin.close(); enc.wait(); dec.wait()
print(f"wrote {a.output} ({k} frames, {W + pw}x{H})")
