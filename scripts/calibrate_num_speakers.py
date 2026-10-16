#!/usr/bin/env python3
"""Calibrate the 1-vs-2 speaker test (DiarizationConfig.single_centroid_cos etc.).

Synthesises single-voice monologues (~5 s and ~30 s, many macOS voices) and two-voice
dialogues (~10 s and ~30 s, including same-gender pairs), runs the real ASR +
diarization path (ECAPA and MFCC) on each, and prints/saves the cluster statistics.
Optionally also measures existing recordings (``--extra label:path.wav`` where label
is 1 or 2).

Usage: python scripts/calibrate_num_speakers.py [--out out/calibration_num_speakers.json]
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

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from podcast_autoeditor.config import AsrConfig, DiarizationConfig  # noqa: E402
from podcast_autoeditor.diarize import diarize  # noqa: E402
from podcast_autoeditor.transcribe import transcribe, words_from_transcript  # noqa: E402

SR = 16000
VOICES = ["Samantha", "Daniel", "Karen", "Rishi", "Moira", "Shelley (English (US))", "Fred",
          "Reed (English (US))", "Flo (English (US))", "Eddy (English (US))", "Sandy (English (US))",
          "Grandma (English (US))", "Ralph", "Kathy", "Albert", "Rocko (English (US))"]
PAIRS = [("Samantha", "Daniel"), ("Samantha", "Shelley (English (US))"), ("Moira", "Karen"),
         ("Karen", "Rishi"), ("Fred", "Ralph"), ("Flo (English (US))", "Sandy (English (US))"),
         ("Reed (English (US))", "Eddy (English (US))"), ("Daniel", "Rocko (English (US))")]
SHORT = "Today I want to share one quick idea about saving energy at home."
LONG = ["Let me tell you how this whole project started.",
        "A few years ago I was renovating an old farmhouse with almost no insulation.",
        "The heating bills were enormous, and every winter the place felt freezing.",
        "So I started reading everything I could find about heat pumps and solar panels.",
        "It took months of planning, but the results were better than I ever expected.",
        "Now the house stays warm, the bills are tiny, and I have learned a great deal along the way."]
DIALOG_SHORT = [("A", "So what do you think about electric cars these days?"),
                ("B", "Honestly, I think they are finally ready for most families.")]
DIALOG_LONG = [("A", "Welcome back to the show. Today we are talking about heat pumps and home energy."),
               ("B", "Thanks for having me. It is a topic I really love."),
               ("A", "So how does a heat pump actually work?"),
               ("B", "It moves heat instead of making it, a bit like a fridge running in reverse."),
               ("A", "Right."),
               ("B", "That is why it can deliver three or four units of heat for every unit of electricity."),
               ("A", "That sounds almost too good to be true, to be honest.")]


def say(text: str, voice: str, tmp: str, tag: str) -> np.ndarray:
    import soundfile as sf

    aiff, wav = os.path.join(tmp, tag + ".aiff"), os.path.join(tmp, tag + ".wav")
    subprocess.run(["say", "-v", voice, "-o", aiff, text], check=True)
    subprocess.run(["ffmpeg", "-hide_banner", "-loglevel", "error", "-y", "-i", aiff, "-ac", "1",
                    "-ar", str(SR), wav], check=True)
    x, _ = sf.read(wav, dtype="float32")
    nz = np.flatnonzero(np.abs(x) > 0.01 * np.abs(x).max())
    return x[nz[0]: nz[-1] + 1] / (np.abs(x).max() + 1e-9) * 0.5


def assemble(parts, rng) -> np.ndarray:
    out = [np.zeros(int(1.5 * SR), np.float32)]
    for x in parts:
        out += [x, np.zeros(int(rng.uniform(0.35, 0.8) * SR), np.float32)]
    out.append(np.zeros(int(1.5 * SR), np.float32))
    y = 0.6 * np.concatenate(out)
    noise = np.convolve(rng.standard_normal(y.size), np.ones(6) / 6, mode="same") * 10 ** (-48 / 20)
    return (y + noise).astype(np.float32)


def measure(wav: str, label: int, name: str, asr: AsrConfig) -> list:
    tr = transcribe(wav, asr)
    words = words_from_transcript(tr)
    rows = []
    for backend in ("ecapa", "mfcc"):
        d = diarize(wav, words, DiarizationConfig(backend=backend, num_speakers="auto"))
        st = d.stats
        rows.append({"case": name, "true": label, "backend": backend, "words": len(words),
                     "centroid_cos": st.get("centroid_cos"), "silhouette": st.get("silhouette"),
                     "speech_s": st.get("speech_s"), "verdict": st.get("embedding_verdict"),
                     "reasons": st.get("reasons")})
        r = rows[-1]
        print(f"{name:42s} true={label} {backend:5s} cos={r['centroid_cos']:.3f} "
              f"sil={r['silhouette']:.3f} speech={r['speech_s']} -> {r['verdict']}", flush=True)
    return rows


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--out", default="out/calibration_num_speakers.json")
    ap.add_argument("--extra", action="append", default=[], help="label:path.wav (label 1 or 2)")
    ap.add_argument("--model", default="small")
    ap.add_argument("--quick", action="store_true", help="fewer voices/pairs")
    a = ap.parse_args()
    import soundfile as sf

    rng = np.random.default_rng(3)
    tmp = tempfile.mkdtemp(prefix="calib_")
    asr = AsrConfig(model_size=a.model)
    rows = []
    voices = VOICES[:6] if a.quick else VOICES
    pairs = PAIRS[:3] if a.quick else PAIRS
    try:
        for v in voices:
            tag = v.split()[0]
            for kind, texts in (("5s", [SHORT]), ("30s", LONG)):
                parts = [say(t, v, tmp, f"{tag}_{kind}_{i}") for i, t in enumerate(texts)]
                wav = os.path.join(tmp, f"mono_{tag}_{kind}.wav")
                sf.write(wav, assemble(parts, rng), SR)
                rows += measure(wav, 1, f"mono {tag} {kind}", asr)
        for va, vb in pairs:
            ta, tb = va.split()[0], vb.split()[0]
            for kind, script in (("10s", DIALOG_SHORT), ("30s", DIALOG_LONG)):
                parts = [say(t, va if s == "A" else vb, tmp, f"{ta}{tb}_{kind}_{i}")
                         for i, (s, t) in enumerate(script)]
                wav = os.path.join(tmp, f"dia_{ta}_{tb}_{kind}.wav")
                sf.write(wav, assemble(parts, rng), SR)
                rows += measure(wav, 2, f"dialog {ta}/{tb} {kind}", asr)
        for spec in a.extra:
            lab, path = spec.split(":", 1)
            rows += measure(path, int(lab), f"extra {os.path.basename(os.path.dirname(path))}/"
                                             f"{os.path.basename(path)}", asr)
    finally:
        shutil.rmtree(tmp, ignore_errors=True)
    os.makedirs(os.path.dirname(os.path.abspath(a.out)), exist_ok=True)
    with open(a.out, "w") as f:
        json.dump(rows, f, indent=1)
    for backend, key in (("ecapa", "centroid_cos"), ("mfcc", "silhouette")):
        one = [r[key] for r in rows if r["backend"] == backend and r["true"] == 1 and r[key] == r[key]]
        two = [r[key] for r in rows if r["backend"] == backend and r["true"] == 2 and r[key] == r[key]]
        if one and two:
            print(f"{backend} {key}: single-voice min/median/max {min(one):.3f}/{np.median(one):.3f}/"
                  f"{max(one):.3f}; two-voice min/median/max {min(two):.3f}/{np.median(two):.3f}/{max(two):.3f}")
        errs = [r["case"] for r in rows if r["backend"] == backend and r["verdict"] != r["true"]]
        print(f"{backend}: {len(errs)} wrong verdicts with current thresholds: {errs}")
    print("saved", a.out)
    return 0


if __name__ == "__main__":
    sys.exit(main())
