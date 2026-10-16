#!/usr/bin/env python3
"""Build a synthetic multicam podcast for testing (macOS ``say`` voices).

Produces in ``--out-dir`` (default ``demo/``):
  cam_a.mp4   close-up of Speaker A  (1280x720 @ 30 fps; A loud, B bleed; starts 0.6 s EARLY)
  cam_b.mp4   close-up of Speaker B  ( 960x540 @ 25 fps; B loud, A bleed; starts 1.3 s LATE)
  wide.mp4    wide shot of both      (1920x1080 @ 30 fps; balanced mix; timeline reference)
  master.wav  clean stereo-free mix (optional external master audio)
  ground_truth.json  line timings / speakers / offsets / frame-code layout

Every frame carries a 16-bit barcode of the *reference-timeline* time (centiseconds)
so the edited output can be checked for camera choice and sync automatically.
"""

from __future__ import annotations

import argparse
import json
import os
import shutil
import subprocess
import sys
import tempfile

import numpy as np

SR = 48000

# (speaker, text, gap_before_seconds).  A negative gap = overlap with previous line.
SCRIPT = [
    ("A", "Welcome back to Circuit Hour. Today we are talking about electric vehicles, "
          "batteries, and the charging network. My guest has spent ten years designing battery packs.", 2.0),
    ("A", "So, what got you interested in batteries in the first place?", 0.6),
    ("B", "Honestly, it started in my garage. I took apart an old laptop battery, and I was amazed "
          "by how much energy was packed into such a small space. After that, I was hooked.", 0.7),
    ("A", "That is a great story. How long does a typical battery pack last today?", 0.8),
    ("B", "Most modern packs keep about eighty percent of their capacity after ten years, "
          "which is much better than people expect.", 0.6),
    # rapid back-and-forth (no question marks)
    ("A", "Ten years, wow.", 0.4),
    ("B", "Easily.", 0.3),
    ("A", "Even in the cold.", 0.3),
    ("B", "Even in the cold.", 0.3),
    ("A", "Impressive.", 0.3),
    ("B", "The chemistry has improved a lot. Engineers now manage temperature very carefully, "
          "and the software protects the cells from stress.", 0.5),
    # long silence then B asks A
    ("B", "Can I ask you something now?", 3.5),
    ("A", "Of course, go ahead.", 0.6),
    ("B", "Do you drive an electric car yourself?", 0.5),
    ("A", "I do. I switched three years ago, and I would never go back. Charging at home "
          "overnight is so convenient, and the car is quiet and fast.", 0.7),
    ("B", "Exactly, yes.", -0.9),   # interjection overlapping the end of A's line
    ("A", "Thanks for listening, everyone. See you next week on Circuit Hour.", 0.8),
]
TAIL = 2.5
# --extra-overlaps: interjections spoken OVER the other person's line
# (speaker, text, index of the line they overlap, seconds after that line's start)
EXTRA_OVERLAPS = [("B", "Mm, absolutely, yes.", 0, 1.3), ("A", "Oh wow, that is amazing.", 2, 7.0)]
# --monologue A|B: a single person talks (no second voice at all)
MONOLOGUE = {
    "short": ["Today I want to share one quick idea about saving energy at home."],
    "long": ["Let me tell you how this whole project started.",
             "A few years ago I was renovating an old farmhouse with almost no insulation.",
             "The heating bills were enormous, and every winter the place felt freezing.",
             "So I started reading everything I could find about heat pumps and solar panels.",
             "It took months of planning, but the results were better than I ever expected.",
             "Now the house stays warm, the bills are tiny, and I learned a lot along the way."],
}
VOICES = {"A": "Samantha", "B": "Daniel"}
OFFSETS = {"cam_a": -0.6, "cam_b": 1.3, "wide": 0.0}
CLIPS = {
    "cam_a": dict(size=(1280, 720), fps=30, bg=(150, 40, 40), label="CAM A", faces="A"),
    "cam_b": dict(size=(960, 540), fps=25, bg=(30, 120, 60), label="CAM B", faces="B"),
    "wide": dict(size=(1920, 1080), fps=30, bg=(90, 60, 140), label="WIDE", faces="AB"),
}
# barcode layout in relative coordinates (same for every clip; all are 16:9)
CODE_BITS = 16
CODE_X0, CODE_DX, CODE_Y, CODE_SIZE = 0.05, 0.035, 0.05, 0.025


def run(cmd, **kw):
    r = subprocess.run(cmd, capture_output=True, text=True, **kw)
    if r.returncode != 0:
        sys.exit(f"command failed: {' '.join(cmd)}\n{r.stderr[-2000:]}")
    return r


def tts(text: str, voice: str, tmp: str, idx: int, ffmpeg: str) -> np.ndarray:
    aiff = os.path.join(tmp, f"l{idx}.aiff")
    wav = os.path.join(tmp, f"l{idx}.wav")
    run(["say", "-v", voice, "-o", aiff, text])
    run([ffmpeg, "-hide_banner", "-loglevel", "error", "-y", "-i", aiff, "-ac", "1", "-ar", str(SR),
         "-c:a", "pcm_f32le", wav])
    import soundfile as sf

    x, _ = sf.read(wav, dtype="float32")
    # trim leading/trailing silence so ground-truth timings are tight
    thr = 0.01 * np.abs(x).max()
    nz = np.flatnonzero(np.abs(x) > thr)
    x = x[max(0, nz[0] - int(0.01 * SR)): nz[-1] + int(0.02 * SR)]
    return 0.5 * x / (np.abs(x).max() + 1e-9)


def envelope(x: np.ndarray, fps: float, n_frames: int, offset_s: float) -> np.ndarray:
    """RMS of ``x`` (reference timeline) around each clip frame's reference time."""
    win = int(0.06 * SR)
    out = np.zeros(n_frames)
    for k in range(n_frames):
        t = k / fps + offset_s
        c = int(t * SR)
        seg = x[max(0, c - win): max(0, c + win)]
        out[k] = np.sqrt(np.mean(seg ** 2)) if seg.size else 0.0
    m = out.max() or 1.0
    return np.clip(out / (0.6 * m), 0, 1)


def load_photo_faces():
    """Two real face photos (Olivetti dataset via scikit-learn, downloaded once)."""
    from sklearn.datasets import fetch_olivetti_faces

    try:
        import certifi

        os.environ.setdefault("SSL_CERT_FILE", certifi.where())
    except ImportError:
        pass
    d = fetch_olivetti_faces(data_home=os.path.expanduser("~/.cache/podcast_autoeditor/sklearn"))
    return {"A": (d.images[0] * 255).astype(np.uint8), "B": (d.images[10] * 255).astype(np.uint8)}


PHOTOS = None


# --single-wide: ONE wide clip with two people (real face photos) for single-clip mode
SINGLE = None                       # {"seat": {"A": side, "B": side}, "hide": set, "move": (spk, t, dx)}
SEAT_X = {"left": 0.28, "right": 0.72}   # face centre (fraction of width)
SEAT_Y, FACE_R = 0.52, 0.10         # face centre y and half-size (fractions of height)
STRIP_DY, STRIP_H = 0.30, 0.07       # per-face time-code strip above each photo (fractions of photo size)


def single_geometry(W, H, spk, ref_t):
    """(x0, y0, side_px) of a person's photo square at reference time ref_t."""
    side = int(2 * FACE_R * H)
    cx = SEAT_X[SINGLE["seat"][spk]] * W
    mv = SINGLE.get("move")
    if mv and mv[0] == spk and ref_t >= mv[1]:
        cx += mv[2] * W
    return int(cx - side / 2), int(SEAT_Y * H - side / 2), side


def draw_single(cv2, base, photos, env, k, ref_t, W, H):
    """One frame of the single wide clip: both people, mouths driven by their own voice, a
    16-bit time-code strip above each face (so cropped close-ups can be checked for sync)
    and the usual global barcode."""
    img = base.copy()
    v = max(0, int(round(ref_t * 100)))
    for spk in "AB":
        x0, y0, side = single_geometry(W, H, spk, ref_t)
        cx, cy, r = x0 + side // 2, y0 + side // 2, side / 2.0
        o = env[spk][k]
        if spk in SINGLE["hide"]:   # a cartoon the face detector will not find
            cv2.circle(img, (cx, cy), int(r), (200, 170, 140), -1, cv2.LINE_AA)
            cv2.ellipse(img, (cx, int(cy + 0.42 * r)), (int(0.35 * r), max(2, int(0.04 * r + 0.25 * r * o))),
                        0, 0, 360, (90, 20, 30), -1, cv2.LINE_AA)
        else:
            img[y0:y0 + side, x0:x0 + side] = photos[spk][:, :, None]
            cv2.ellipse(img, (cx, int(y0 + 0.80 * side)),
                        (int(0.15 * side), max(1, int((0.005 + 0.06 * o) * side))),
                        0, 0, 360, (35, 25, 25), -1, cv2.LINE_AA)
        sy0, sh, cw = int(y0 - STRIP_DY * side), max(4, int(STRIP_H * side)), side / 16.0
        cv2.rectangle(img, (x0 - 4, sy0 - 4), (x0 + side + 4, sy0 + sh + 4), (128, 128, 128), -1)
        for b in range(16):
            bit = (v >> (15 - b)) & 1
            img[sy0:sy0 + sh, int(x0 + b * cw) + 1:int(x0 + (b + 1) * cw) - 1] = 255 if bit else 0
    sz = int(CODE_SIZE * W)
    y0 = int(CODE_Y * H)
    cv2.rectangle(img, (int(CODE_X0 * W) - 6, y0 - 6),
                  (int((CODE_X0 + CODE_DX * (CODE_BITS - 1)) * W) + sz + 6, y0 + sz + 6), (128, 128, 128), -1)
    for b in range(CODE_BITS):
        bit = (v >> (CODE_BITS - 1 - b)) & 1
        x0 = int((CODE_X0 + CODE_DX * b) * W)
        img[y0:y0 + sz, x0:x0 + sz] = 255 if bit else 0
    return img


def draw_static(cv2, W, H, bg, label, faces):
    img = np.zeros((H, W, 3), np.uint8)
    img[:] = bg
    s = H / 1080.0
    font = cv2.FONT_HERSHEY_DUPLEX
    (tw, th), _ = cv2.getTextSize(label, font, 4.0 * s, int(8 * s))
    cv2.putText(img, label, ((W - tw) // 2, int(0.22 * H)), font, 4.0 * s, (255, 255, 255),
                int(8 * s), cv2.LINE_AA)
    sub = {"A": "Speaker A", "B": "Speaker B", "AB": "A  +  B"}[faces]
    (tw, th), _ = cv2.getTextSize(sub, font, 1.6 * s, int(3 * s))
    cv2.putText(img, sub, ((W - tw) // 2, int(0.30 * H)), font, 1.6 * s, (230, 230, 230),
                int(3 * s), cv2.LINE_AA)
    centers = []
    if faces == "AB":
        centers = [("A", int(0.32 * W), int(0.55 * H), 0.16 * H), ("B", int(0.68 * W), int(0.55 * H), 0.16 * H)]
    else:
        centers = [(faces, W // 2, int(0.56 * H), 0.2 * H)]
    for spk, cx, cy, r in centers:
        if PHOTOS is not None:
            side = int(2 * r)
            ph = cv2.resize(PHOTOS[spk], (side, side), interpolation=cv2.INTER_CUBIC)
            y0, x0 = cy - side // 2, cx - side // 2
            img[y0:y0 + side, x0:x0 + side] = ph[:, :, None]
            continue
        skin = (230, 190, 160) if spk == "A" else (200, 160, 120)
        cv2.circle(img, (cx, cy), int(r), skin, -1, cv2.LINE_AA)
        for ex in (-0.38, 0.38):
            cv2.circle(img, (int(cx + ex * r), int(cy - 0.25 * r)), int(0.1 * r), (40, 40, 40), -1, cv2.LINE_AA)
    return img, centers


def draw_frame(cv2, base, centers, env_by_spk, k, ref_cs, W, H):
    img = base.copy()
    for spk, cx, cy, r in centers:
        o = env_by_spk[spk][k]
        if PHOTOS is not None:  # open the photo's mouth with the speaker's loudness
            cv2.ellipse(img, (cx, int(cy - r + 0.80 * 2 * r)),
                        (int(0.15 * 2 * r), max(1, int((0.005 + 0.06 * o) * 2 * r))),
                        0, 0, 360, (35, 25, 25), -1, cv2.LINE_AA)
            continue
        cv2.ellipse(img, (cx, int(cy + 0.42 * r)), (int(0.35 * r), max(2, int(0.04 * r + 0.25 * r * o))),
                    0, 0, 360, (90, 20, 30), -1, cv2.LINE_AA)
    # barcode of reference time (centiseconds), MSB first
    sz = int(CODE_SIZE * W)
    y0 = int(CODE_Y * H)
    cv2.rectangle(img, (int(CODE_X0 * W) - 6, y0 - 6),
                  (int((CODE_X0 + CODE_DX * (CODE_BITS - 1)) * W) + sz + 6, y0 + sz + 6), (128, 128, 128), -1)
    v = max(0, int(round(ref_cs)))
    for b in range(CODE_BITS):
        bit = (v >> (CODE_BITS - 1 - b)) & 1
        x0 = int((CODE_X0 + CODE_DX * b) * W)
        img[y0:y0 + sz, x0:x0 + sz] = 255 if bit else 0
    return img


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--out-dir", default="demo")
    ap.add_argument("--ffmpeg", default=shutil.which("ffmpeg") or "ffmpeg")
    ap.add_argument("--voice-a", default=None, help="macOS voice for Speaker A (default Samantha)")
    ap.add_argument("--voice-b", default=None, help="macOS voice for Speaker B (default Daniel)")
    ap.add_argument("--bleed", type=float, default=0.18,
                    help="level of the other speaker in each close-up mic (0.18 ~ -15 dB)")
    ap.add_argument("--extra-overlaps", action="store_true",
                    help="add two interjections spoken over the other person's line")
    ap.add_argument("--monologue", choices=["A", "B"], default=None,
                    help="single-speaker scenario: only this person talks")
    ap.add_argument("--monologue-len", choices=["short", "long"], default="long",
                    help="short ~5 s, long ~30 s of speech")
    ap.add_argument("--single-wide", action="store_true",
                    help="single-clip scenario: write ONLY wide.mp4 (two real face photos, left and "
                         "right, mouths driven by who speaks, mixed audio) for --mode single")
    ap.add_argument("--first-speaker-side", choices=["left", "right"], default="left",
                    help="--single-wide: where the script's first speaker (voice A) sits")
    ap.add_argument("--hide-face", choices=["left", "right"], default=None,
                    help="--single-wide: draw a cartoon instead of a photo on that side (1-face case)")
    ap.add_argument("--move-at", type=float, default=None,
                    help="--single-wide: the right-hand person shifts sideways at this time (s)")
    ap.add_argument("--move-dx", type=float, default=0.09, help="... by this fraction of the width")
    ap.add_argument("--face-size", type=float, default=None,
                    help="--single-wide: half-size of the face photos as a fraction of the frame "
                         "height (default 0.10; 0.07 gives more realistic, smaller faces)")
    ap.add_argument("--photo-faces", action="store_true",
                    help="use real face photos (Olivetti, downloaded via scikit-learn) with animated "
                         "mouths instead of cartoon faces - exercises the visual mapping strategy")
    args = ap.parse_args()
    if args.voice_a:
        VOICES["A"] = args.voice_a
    if args.voice_b:
        VOICES["B"] = args.voice_b
    global PHOTOS, SINGLE, FACE_R
    if args.face_size:
        FACE_R = args.face_size
    if args.photo_faces or args.single_wide:
        PHOTOS = load_photo_faces()
    if args.single_wide:
        other = "right" if args.first_speaker_side == "left" else "left"
        seat = {"A": args.first_speaker_side, "B": other}
        by_side = {v: k for k, v in seat.items()}
        SINGLE = {"seat": seat, "hide": {by_side[args.hide_face]} if args.hide_face else set(),
                  "move": (by_side["right"], args.move_at, args.move_dx) if args.move_at else None}
    import cv2
    import soundfile as sf

    if shutil.which("say") is None:
        sys.exit("macOS `say` is required to synthesise the demo voices")
    os.makedirs(args.out_dir, exist_ok=True)
    tmp = tempfile.mkdtemp(prefix="podcast_demo_")
    rng = np.random.default_rng(42)

    # ---------------------------------------------------------------- audio timeline
    script = SCRIPT
    if args.monologue:
        script = [(args.monologue, txt, 2.0 if i == 0 else 0.5 + 0.15 * (i % 3))
                  for i, txt in enumerate(MONOLOGUE[args.monologue_len])]
    clips = [tts(text, VOICES[spk], tmp, i, args.ffmpeg) for i, (spk, text, _) in enumerate(script)]
    t = 0.0
    lines = []
    for (spk, text, gap), x in zip(script, clips):
        start = t + gap if gap >= 0 else lines[-1]["end"] + gap
        end = start + x.size / SR
        lines.append({"speaker": spk, "text": text, "start": round(start, 3), "end": round(end, 3)})
        t = max(t, end)
    overlap_lines = [15] if not args.monologue else []
    if args.extra_overlaps and not args.monologue:
        for j, (spk, text, anchor, off) in enumerate(EXTRA_OVERLAPS):
            x = tts(text, VOICES[spk], tmp, 100 + j, args.ffmpeg)
            st = lines[anchor]["start"] + off
            lines.append({"speaker": spk, "text": text, "start": round(st, 3),
                          "end": round(st + x.size / SR, 3), "interjection": True})
            clips.append(x)
            overlap_lines.append(len(lines) - 1)
    total = t + TAIL
    n = int(np.ceil(total * SR))
    sA, sB = np.zeros(n, np.float32), np.zeros(n, np.float32)
    for ln, x in zip(lines, clips):
        i0 = int(round(ln["start"] * SR))
        tgt = sA if ln["speaker"] == "A" else sB
        tgt[i0:i0 + x.size] += x

    def room(level_db: float) -> np.ndarray:
        w = rng.standard_normal(n).astype(np.float32)
        w = np.convolve(w, np.ones(6) / 6, mode="same").astype(np.float32)
        return w * (10 ** (level_db / 20))

    bleed = args.bleed  # 0.18 ~ -15 dB
    mixes = {
        "cam_a": sA + bleed * sB + room(-52),
        "cam_b": bleed * sA + sB + room(-52),
        "wide": 0.6 * sA + 0.6 * sB + room(-48),
    }
    sf.write(os.path.join(args.out_dir, "master.wav"), np.clip(0.7 * sA + 0.7 * sB, -1, 1), SR)
    # per-clip audio with start offsets (clip t=0 at reference time = offset)
    clip_audio = {}
    for name, mix in mixes.items():
        off = OFFSETS[name]
        k = int(round(off * SR))
        if k >= 0:
            a = mix[k:]
        else:
            a = np.concatenate([room(-52)[: -k], mix])
        extra = {"cam_a": 1.0, "cam_b": 0.4, "wide": 0.0}[name]
        a = np.concatenate([a, room(-52)[: int(extra * SR)]])
        clip_audio[name] = np.clip(a, -1, 1).astype(np.float32)

    # ---------------------------------------------------------------- video
    clip_specs = CLIPS
    if SINGLE:  # one wide clip that IS the timeline reference
        clip_specs = {"wide": dict(CLIPS["wide"], bg=(60, 70, 95))}
    for name, spec in clip_specs.items():
        W, H = spec["size"]
        fps = spec["fps"]
        audio = clip_audio[name]
        dur = audio.size / SR
        nf = int(round(dur * fps))
        wav = os.path.join(tmp, f"{name}.wav")
        sf.write(wav, audio, SR)
        base, centers = draw_static(cv2, W, H, spec["bg"][::-1], spec["label"], spec["faces"])  # RGB->BGR
        env = {s: envelope(sA if s == "A" else sB, fps, nf, OFFSETS[name]) for s in spec["faces"]}
        if SINGLE:
            base = np.zeros((H, W, 3), np.uint8)
            base[:] = spec["bg"][::-1]
            side_px = int(2 * FACE_R * H)
            small = {k: cv2.resize(v, (side_px, side_px), interpolation=cv2.INTER_CUBIC)
                     for k, v in PHOTOS.items()}
        out = os.path.join(args.out_dir, f"{name}.mp4")
        cmd = [args.ffmpeg, "-hide_banner", "-loglevel", "error", "-y",
               "-f", "rawvideo", "-pix_fmt", "bgr24", "-s", f"{W}x{H}", "-r", str(fps), "-i", "-",
               "-i", wav, "-map", "0:v", "-map", "1:a", "-c:v", "libx264", "-preset", "veryfast",
               "-crf", "20", "-pix_fmt", "yuv420p", "-g", str(fps * 2), "-c:a", "aac", "-b:a", "160k",
               "-shortest", out]
        p = subprocess.Popen(cmd, stdin=subprocess.PIPE, stderr=subprocess.PIPE)
        assert p.stdin is not None
        for k in range(nf):
            ref_t = k / fps + OFFSETS[name]
            if SINGLE:
                p.stdin.write(draw_single(cv2, base, small, env, k, ref_t, W, H).tobytes())
            else:
                p.stdin.write(draw_frame(cv2, base, centers, env, k, ref_t * 100, W, H).tobytes())
        p.stdin.close()
        err = p.stderr.read().decode() if p.stderr else ""
        if p.wait() != 0:
            sys.exit(f"ffmpeg failed for {name}: {err[-1500:]}")
        print(f"wrote {out}: {W}x{H}@{fps} {dur:.2f}s offset {OFFSETS[name]:+.2f}s")

    gt = {
        "description": "synthetic multicam podcast; times are on the reference (wide) timeline",
        "voices": VOICES,
        "reference": "wide",
        "duration": round(clip_audio["wide"].size / SR, 3),
        "offsets": OFFSETS if not SINGLE else {"wide": 0.0},
        "clips": {k: {"size": v["size"], "fps": v["fps"], "bg_rgb": v["bg"], "label": v["label"]}
                  for k, v in clip_specs.items()},
        "barcode": {"bits": CODE_BITS, "x0": CODE_X0, "dx": CODE_DX, "y": CODE_Y, "size": CODE_SIZE,
                    "unit": "centiseconds of reference time"},
        "lines": lines,
        "expected": ({"single_speaker": args.monologue} if args.monologue else {
            "qa_pairs": [[1, 2], [3, 4], [11, 12], [13, 14]],
            "rapid_lines": [5, 6, 7, 8, 9],
            "long_silence": [lines[10]["end"], lines[11]["start"]],
            "overlap_line": 15,
            "overlap_lines": overlap_lines,
            "monologues": {"A": [0, 14], "B": [2, 10]},
        }),
    }
    if SINGLE:
        W, H = CLIPS["wide"]["size"]
        for ln in lines:
            ln["side"] = SINGLE["seat"][ln["speaker"]]
        ppl = {}
        for spk in "AB":
            x0, y0, side_px = single_geometry(W, H, spk, 0.0)
            ppl[SINGLE["seat"][spk]] = {
                "voice": VOICES[spk], "script_speaker": spk, "hidden": spk in SINGLE["hide"],
                "photo_box": [x0, y0, x0 + side_px, y0 + side_px],
                "strip": {"dy": STRIP_DY, "h": STRIP_H, "bits": 16},
            }
        mv = SINGLE.get("move")
        gt["single_wide"] = {"frame": [W, H], "people": ppl,
                             "move": ({"side": "right", "t": mv[1], "dx_px": int(mv[2] * W)} if mv else None)}
    with open(os.path.join(args.out_dir, "ground_truth.json"), "w") as f:
        json.dump(gt, f, indent=1)
    shutil.rmtree(tmp, ignore_errors=True)
    print(f"ground truth: {os.path.join(args.out_dir, 'ground_truth.json')}  (duration {gt['duration']:.1f}s)")


if __name__ == "__main__":
    main()
