#!/usr/bin/env python3
"""Render stress / drift test: hundreds of random shots on the synthetic demo.

Cuts the demo between cam A / cam B / wide every few frames (random), renders with the
production renderer and checks every sampled frame shows the planned camera and the
correct reference time (barcode) -> proves the per-shot + concat approach has no drift.

Usage: python scripts/stress_render.py --demo-dir demo --shots 300
"""

import argparse
import json
import math
import os
import random
import subprocess
import sys
import time

import numpy as np

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from podcast_autoeditor import CAM_A, CAM_B, WIDE  # noqa: E402
from podcast_autoeditor.config import RenderConfig  # noqa: E402
from podcast_autoeditor.datatypes import Shot  # noqa: E402
from podcast_autoeditor.ffmpeg_utils import find_ffmpeg, probe  # noqa: E402
from podcast_autoeditor.render import AudioSource, SourceClip, render_final, render_segments  # noqa: E402

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from verify_demo import classify_camera, decode_barcode  # noqa: E402


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--demo-dir", default="demo")
    ap.add_argument("--shots", type=int, default=300)
    ap.add_argument("--out", default="out/stress.mp4")
    ap.add_argument("--samples", type=int, default=60)
    ap.add_argument("--seed", type=int, default=0)
    a = ap.parse_args()
    gt = json.load(open(os.path.join(a.demo_dir, "ground_truth.json")))
    ff = find_ffmpeg()
    fps = 30.0
    dur = gt["duration"] - 0.2
    n = int(math.ceil(dur * fps - 1e-6))
    rng = random.Random(a.seed)
    cuts = sorted(rng.sample(range(1, n), a.shots - 1))
    bounds = [0] + cuts + [n]
    cams, prev = [], None
    for _ in range(a.shots):
        c = rng.choice([x for x in (CAM_A, CAM_B, WIDE) if x != prev])
        cams.append(c)
        prev = c
    shots = [Shot(s / fps, e / fps if i < a.shots - 1 else dur, c, "stress", s, e)
             for i, (s, e, c) in enumerate(zip(bounds, bounds[1:], cams))]
    files = {CAM_A: "cam_a.mp4", CAM_B: "cam_b.mp4", WIDE: "wide.mp4"}
    keys = {CAM_A: "cam_a", CAM_B: "cam_b", WIDE: "wide"}
    srcs = {c: SourceClip(os.path.join(a.demo_dir, f), gt["offsets"][keys[c]],
                          probe(ff, os.path.join(a.demo_dir, f))) for c, f in files.items()}
    cfg = RenderConfig(preset="veryfast")
    work = os.path.splitext(a.out)[0] + "_work"
    os.makedirs(work, exist_ok=True)
    t0 = time.time()
    segs = render_segments(ff, shots, srcs, cfg, os.path.join(work, "segments"))
    t1 = time.time()
    render_final(ff, segs, AudioSource(os.path.abspath(os.path.join(a.demo_dir, "wide.mp4"))), dur, cfg,
                 work, a.out)
    t2 = time.time()
    pr = json.loads(subprocess.run([ff.ffprobe, "-v", "error", "-count_frames", "-select_streams", "v:0",
                                    "-print_format", "json", "-show_streams", a.out],
                                   capture_output=True, text=True).stdout)["streams"][0]
    nb = int(pr["nb_read_frames"])
    print(f"{a.shots} shots: segments {t1 - t0:.1f}s, final {t2 - t1:.1f}s; frames {nb} (expected {n})")
    ok = nb == n
    picks = sorted(rng.sample(range(a.shots), min(a.samples, a.shots)))
    bad = 0
    worst = 0.0
    cam_name = {"cam_a": CAM_A, "cam_b": CAM_B, "wide": WIDE}
    for i in picks:
        s = shots[i]
        f = (s.start_frame + s.end_frame - 1) // 2
        # -ss picks the first frame with pts >= t, so seek a quarter frame early to land on f
        t = max(0.0, (f - 0.25) / fps)
        raw = subprocess.run([ff.ffmpeg, "-hide_banner", "-loglevel", "error", "-ss", f"{t:.4f}", "-i", a.out,
                              "-frames:v", "1", "-f", "rawvideo", "-pix_fmt", "rgb24", "-"],
                             capture_output=True, check=True).stdout
        rgb = np.frombuffer(raw, np.uint8).reshape(1080, 1920, 3)
        got, _ = classify_camera(rgb, gt["clips"])
        ref_t = decode_barcode(rgb, gt["barcode"]) / 100.0
        err = ref_t - f / fps
        worst = max(worst, abs(err))
        src_fps = gt["clips"][got]["fps"]
        if cam_name[got] != s.camera or abs(err) > 1.0 / src_fps + 0.011:
            bad += 1
            print(f"  shot {i} frame {f}: planned {s.camera} got {cam_name[got]}, time err {err * 1000:+.0f} ms")
    print(f"sampled {len(picks)} shots: {len(picks) - bad} correct; worst |time error| {worst * 1000:.0f} ms")
    ok = ok and bad == 0
    print("PASS" if ok else "FAIL")
    return 0 if ok else 1


if __name__ == "__main__":
    sys.exit(main())
