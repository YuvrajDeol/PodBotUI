#!/usr/bin/env python3
"""Verify an auto-edit of the synthetic demo against its ground truth.

Checks: sync offsets, speaker mapping / word attribution accuracy, editing rules in
the EDL (Q->A, rapid exchange, silence, overlap, monologues), and - if the video was
rendered - container properties, per-frame camera identity (background colour),
frame-level sync (barcode of reference time) and caption colours.

Usage: python scripts/verify_demo.py --demo-dir demo --out out/demo_edit.mp4
"""

from __future__ import annotations

import argparse
import json
import os
import shutil
import subprocess
import sys
from typing import Dict, List, Optional, Tuple

import numpy as np

RESULTS: List[Tuple[str, bool, str]] = []


def check(name: str, ok: bool, detail: str = "") -> bool:
    RESULTS.append((name, bool(ok), detail))
    print(f"[{'PASS' if ok else 'FAIL'}] {name}{(' - ' + detail) if detail else ''}")
    return ok


def find_ffmpeg() -> Tuple[str, str]:
    ff = shutil.which("ffmpeg") or "/opt/homebrew/bin/ffmpeg"
    fp = shutil.which("ffprobe") or os.path.join(os.path.dirname(ff), "ffprobe")
    return ff, fp


def cam_at(shots: List[Dict], t: float) -> str:
    for s in shots:
        if s["start"] <= t < s["end"]:
            return s["camera"]
    return shots[-1]["camera"]


def gt_speaker(lines: List[Dict], t: float, tol: float = 0.15) -> Optional[str]:
    hits = {l["speaker"] for l in lines if l["start"] - tol <= t <= l["end"] + tol}
    if len(hits) == 1:
        return hits.pop()
    return None  # silence or overlapping speech -> ambiguous


def grab(ff: str, video: str, t: float, out_png: str, fps: float = 30.0) -> np.ndarray:
    """Extract the output frame that is on screen at time t (frame floor(t*fps)).
    ffmpeg's -ss selects the first frame with pts >= t, so seek a quarter frame early."""
    import cv2

    f = int(np.floor(t * fps + 1e-9))
    ts = max(0.0, (f - 0.25) / fps)
    subprocess.run([ff, "-hide_banner", "-loglevel", "error", "-y", "-ss", f"{ts:.4f}", "-i", video,
                    "-frames:v", "1", out_png], check=True)
    img = cv2.imread(out_png)
    if img is None:
        raise RuntimeError(f"could not read frame at {t}")
    return img[:, :, ::-1]  # RGB


def classify_camera(rgb: np.ndarray, clips: Dict) -> Tuple[str, float]:
    H, W = rgb.shape[:2]
    patch = rgb[int(0.40 * H):int(0.60 * H), int(0.93 * W):int(0.98 * W)].reshape(-1, 3)
    med = np.median(patch, axis=0)
    best, bd = "?", 1e9
    for name, c in clips.items():
        d = float(np.linalg.norm(med - np.array(c["bg_rgb"], float)))
        if d < bd:
            best, bd = name, d
    return best, bd


def decode_barcode(rgb: np.ndarray, bc: Dict) -> int:
    H, W = rgb.shape[:2]
    gray = rgb.mean(axis=2)
    sz = bc["size"] * W
    v = 0
    for b in range(bc["bits"]):
        x = int((bc["x0"] + bc["dx"] * b) * W + sz / 2)
        y = int(bc["y"] * H + sz / 2)
        val = gray[y - 2:y + 3, x - 2:x + 3].mean()
        v = (v << 1) | int(val > 128)
    return v


def caption_pixels(rgb: np.ndarray, color_a: Tuple[int, int, int]) -> Tuple[int, int]:
    """Count caption-coloured pixels in the lower part of the frame: (blue-ish A, white B)."""
    H = rgb.shape[0]
    reg = rgb[int(0.62 * H):int(0.97 * H)].astype(int)
    r, g, b = reg[..., 0], reg[..., 1], reg[..., 2]
    ca = np.array(color_a)
    dist_a = np.sqrt(((reg - ca) ** 2).sum(-1))
    a_cnt = int((dist_a < 45).sum())
    w_cnt = int(((r > 225) & (g > 225) & (b > 225)).sum())
    return a_cnt, w_cnt


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--demo-dir", default="demo")
    ap.add_argument("--out", default="out/demo_edit.mp4")
    ap.add_argument("--frames-dir", default=None)
    ap.add_argument("--min-word-acc", type=float, default=0.90)
    ap.add_argument("--report", default=None)
    ap.add_argument("--a-side", choices=["left", "right"], default="left",
                    help="single-wide demos: the --a-side the edit was made with")
    args = ap.parse_args()
    ff, fp = find_ffmpeg()

    gt = json.load(open(os.path.join(args.demo_dir, "ground_truth.json")))
    sw = gt.get("single_wide")
    if sw:
        # In single-clip mode Speaker A is whoever SITS on --a-side, not "the script's first
        # voice": relabel the ground truth by seat before comparing.
        letter = {args.a_side: "A", ("right" if args.a_side == "left" else "left"): "B"}
        script_to_letter = {v["script_speaker"]: letter[side] for side, v in sw["people"].items()}
        for ln in gt["lines"]:
            ln["speaker"] = letter[ln["side"]]
        if "monologues" in gt["expected"]:
            gt["expected"]["monologues"] = {script_to_letter[k]: v
                                            for k, v in gt["expected"]["monologues"].items()}
    stem = os.path.splitext(args.out)[0]
    edl = json.load(open(f"{stem}_edit_decision_list.json"))
    tr = json.load(open(f"{stem}_transcript.json"))
    meta, shots = edl["meta"], edl["shots"]
    lines = gt["lines"]
    fps = meta["fps"]
    report: Dict = {"files": {"edl": f"{stem}_edit_decision_list.json"}}

    # ---------------------------------------------------------------- sync
    errs = {}
    for k, true in gt["offsets"].items():
        est = meta["offsets"][k]["offset"]
        errs[k] = est - true
        check(f"sync offset {k}", abs(est - true) <= 1.0 / fps,
              f"estimated {est:+.4f}s vs true {true:+.2f}s (error {1000 * (est - true):+.1f} ms)")
    report["sync_error_ms"] = {k: round(1000 * v, 2) for k, v in errs.items()}

    # ---------------------------------------------------------------- words
    words = tr["words"]
    matched = correct = 0
    per_line: Dict[int, List[str]] = {}
    for w in words:
        mid = 0.5 * (w["start"] + w["end"])
        g = gt_speaker(lines, mid)
        if g is None:
            continue
        matched += 1
        correct += int(w["speaker"] == g)
        for i, l in enumerate(lines):
            if l["start"] - 0.15 <= mid <= l["end"] + 0.15:
                per_line.setdefault(i, []).append(w["speaker"])
    acc = correct / max(1, matched)
    report["word_accuracy"] = round(acc, 4)
    report["words_matched"] = matched
    report["words_total"] = len(words)
    check("word->speaker attribution", acc >= args.min_word_acc,
          f"{correct}/{matched} words correct ({100 * acc:.1f}%), {len(words)} ASR words")
    line_ok = judged = 0
    for i, l in enumerate(lines):
        labs = per_line.get(i, [])
        if not labs:  # e.g. an interjection spoken over the other person: no unambiguous words
            continue
        judged += 1
        maj = max(set(labs), key=labs.count)
        line_ok += int(maj == l["speaker"])
    report["line_majority_accuracy"] = f"{line_ok}/{judged} (of {len(lines)} lines)"
    check("speaker mapping (majority label per line)", judged > 0 and line_ok >= judged - 1,
          f"{line_ok}/{judged} judged lines ({len(lines) - judged} fully overlapped lines skipped); "
          f"mapping via {meta['mapping']['method']}")

    # ---------------------------------------------------------------- EDL rules
    exp = gt["expected"]
    single = exp.get("single_speaker")
    if single:
        spk_ok = all(w["speaker"] == single for w in words)
        check(f"single speaker: every word labelled {single}", spk_ok and len(words) > 0,
              f"{sum(w['speaker'] == single for w in words)}/{len(words)} words; "
              f"num_speakers={meta.get('stats', {}).get('num_speakers')}")
        t0, t1 = lines[0]["start"] + 0.05, lines[-1]["end"] - 0.05
        bad = [(s["camera"], round(s["start"], 2), round(s["end"], 2)) for s in shots
               if s["end"] > t0 and s["start"] < t1 and s["camera"] != f"CAM_{single}"]
        check(f"single speaker: whole speech span on CAM_{single}", not bad, f"other shots: {bad}")
    else:
        n_spk = meta.get("stats", {}).get("num_speakers", 2)
        check("two speakers detected", n_spk == 2, f"num_speakers={n_spk}")
    for q, a in ([] if single else exp["qa_pairs"]):
        t1, t2 = lines[q]["end"] - 0.3, lines[a]["start"] + 0.5
        c1, c2 = cam_at(shots, t1), cam_at(shots, t2)
        check(f"Q->A lines {q}->{a} is WIDE", c1 == "WIDE" and c2 == "WIDE",
              f"t={t1:.2f}:{c1}, t={t2:.2f}:{c2}")
    if not single:
        rap = [cam_at(shots, 0.5 * (lines[i]["start"] + lines[i]["end"])) for i in exp["rapid_lines"]]
        check("rapid back-and-forth is WIDE", all(c == "WIDE" for c in rap), str(rap))
        s0, s1 = exp["long_silence"]
        check("long silence is WIDE", cam_at(shots, 0.5 * (s0 + s1)) == "WIDE")
    ov_spans = []
    for ov in ([] if single else exp.get("overlap_lines", [exp.get("overlap_line")])):
        # overlapped part = intersection with any other-speaker line
        L = lines[ov]
        others = [o for j, o in enumerate(lines) if j != ov and o["speaker"] != L["speaker"]
                  and o["start"] < L["end"] and L["start"] < o["end"]]
        if not others:
            continue
        s_, e_ = max(L["start"], others[0]["start"]), min(L["end"], others[0]["end"])
        ov_spans.append((s_, e_))
        t_ov = 0.5 * (s_ + e_)
        if sw:  # no per-camera mics in single-clip mode: only overlapping ASR words count
            print(f"[INFO] overlapping speech (line {ov}) {s_:.2f}-{e_:.2f}: {cam_at(shots, t_ov)} "
                  "(not checked: mic-based overlap detection needs separate close-up mics)")
            report.setdefault("not_checked", []).append(f"overlap line {ov}: {cam_at(shots, t_ov)}")
            continue
        check(f"overlapping speech (line {ov}) is WIDE", cam_at(shots, t_ov) == "WIDE",
              f"overlap {s_:.2f}-{e_:.2f}, t={t_ov:.2f}:{cam_at(shots, t_ov)}")
    for spk, idxs in ({} if single else exp["monologues"]).items():
        for i in idxs:
            t = 0.5 * (lines[i]["start"] + lines[i]["end"]) + 1.0
            dist = lambda x: min((max(a_ - x, x - b_, 0.0) for a_, b_ in ov_spans), default=99.0)  # noqa: E731
            if dist(t) < 1.5:  # sample away from a deliberate interjection
                L = lines[i]
                t = max((L["start"] + f * (L["end"] - L["start"]) for f in (0.3, 0.45, 0.6, 0.75)), key=dist)
            c = cam_at(shots, t)
            if sw and any(v["hidden"] and letter[sd] == spk for sd, v in sw["people"].items()):
                # that person's face is not detectable: no virtual close-up -> stays WIDE
                check(f"monologue line {i} ({spk}, no face found) stays WIDE", c == "WIDE", f"t={t:.2f}:{c}")
                continue
            check(f"monologue line {i} ({spk}) on CAM_{spk}", c == f"CAM_{spk}", f"t={t:.2f}:{c}")
    check("intro is WIDE", cam_at(shots, 0.5) == "WIDE")
    check("outro is WIDE", cam_at(shots, gt["duration"] - 0.5) == "WIDE")
    check("min shot length respected", min(s["duration"] for s in shots) >= 1.2 - 1e-6,
          f"min {min(s['duration'] for s in shots):.3f}s over {len(shots)} shots")
    cont = all(abs(a["end"] - b["start"]) < 1e-6 for a, b in zip(shots, shots[1:]))
    check("EDL covers timeline without gaps", cont and shots[0]["start"] == 0
          and abs(shots[-1]["end"] - meta["duration"]) < 1e-6)

    # ---------------------------------------------------------------- single-clip reframing
    if sw:
        rfm = meta.get("reframe", {})
        FWs, FHs = sw["frame"]
        check("single-clip mode used", meta.get("mode") == "single", f"mode={meta.get('mode')}")
        shown = {side: v for side, v in sw["people"].items() if not v["hidden"]}
        for side, v in sw["people"].items():
            L = letter[side]
            x0, y0, x1, y1 = v["photo_box"]
            person, crop = (rfm.get("people") or {}).get(L), (rfm.get("crops") or {}).get(L)
            if v["hidden"]:
                check(f"hidden face on the {side}: Speaker {L} has no virtual close-up",
                      person is None and crop is None, f"person={person} crop={crop}")
                continue
            mv_ = sw.get("move") or {}
            dxs = (0, mv_.get("dx_px", 0)) if mv_.get("side") == side else (0,)   # before / after a move
            ok_face = bool(person) and y0 <= person["cy"] <= y1 and any(
                x0 + d_ <= person["cx"] <= x1 + d_ for d_ in dxs)
            check(f"face track for the {side} person = Speaker {L}", ok_face,
                  f"track at {(person or {}).get('cx')},{(person or {}).get('cy')} vs photo {v['photo_box']}")
            if not crop:
                check(f"Speaker {L} crop exists", False)
                continue
            cx_, cy_, cw_, ch_ = crop
            inside = cx_ >= 0 and cy_ >= 0 and cx_ + cw_ <= FWs and cy_ + ch_ <= FHs
            even = all(val % 2 == 0 for val in crop)
            aspect = cw_ * meta["height"] == ch_ * meta["width"]
            contains = any(cx_ <= x0 + d_ and cy_ <= y0 and cx_ + cw_ >= x1 + d_ and cy_ + ch_ >= y1
                           for d_ in dxs)
            # the other person's photo at every position they take (before and after a move)
            others = []
            for sd, o in shown.items():
                if sd != side:
                    for d_ in ((0, mv_.get("dx_px", 0)) if mv_.get("side") == sd else (0,)):
                        others.append([o["photo_box"][0] + d_, 0, o["photo_box"][2] + d_, 0])
            excl = all(not (cx_ < o[2] and cx_ + cw_ > o[0]) for o in others)
            eye_frac = (person["eye_y"] - cy_) / ch_
            check(f"Speaker {L} crop {crop}: inside frame, even, output aspect, contains own face, "
                  "excludes the other, eyes near the upper third",
                  inside and even and aspect and contains and excl and 0.25 <= eye_frac <= 0.42,
                  f"inside={inside} even={even} aspect={aspect} contains={contains} excludes={excl} "
                  f"eye line at {eye_frac:.2f} of the crop height, zoom {meta['height'] / ch_:.2f}x")
        crop_rows = [s_ for s_ in shots if s_["camera"] != "WIDE"]
        check("EDL records a crop rectangle for every close-up and none for WIDE",
              all(s_.get("crop_w") not in ("", None) for s_ in crop_rows)
              and all(s_.get("crop_w") in ("", None) for s_ in shots if s_["camera"] == "WIDE"),
              f"{len(crop_rows)} close-up shots")
        report["reframe"] = rfm

    # ---------------------------------------------------------------- rendered video
    if os.path.isfile(args.out):
        pr = json.loads(subprocess.run([fp, "-v", "error", "-print_format", "json", "-show_streams",
                                        "-show_format", args.out], capture_output=True, text=True).stdout)
        vs = next(s for s in pr["streams"] if s["codec_type"] == "video")
        as_ = [s for s in pr["streams"] if s["codec_type"] == "audio"]
        vdur = float(vs.get("duration", pr["format"]["duration"]))
        adur = float(as_[0]["duration"]) if as_ else 0.0
        report["video"] = {"codec": vs["codec_name"], "size": f"{vs['width']}x{vs['height']}",
                           "fps": vs["avg_frame_rate"], "video_duration": vdur, "audio_duration": adur,
                           "nb_frames": vs.get("nb_frames")}
        check("output is H.264 1920x1080 + AAC", vs["codec_name"] == "h264" and vs["width"] == 1920
              and vs["height"] == 1080 and bool(as_) and as_[0]["codec_name"] == "aac",
              str(report["video"]))
        check("output duration ~= master audio", abs(vdur - gt["duration"]) < 0.1 and abs(adur - gt["duration"]) < 0.1,
              f"video {vdur:.3f}s audio {adur:.3f}s master {gt['duration']:.3f}s")
        frames_dir = args.frames_dir or (stem + "_frames")
        os.makedirs(frames_dir, exist_ok=True)
        cam_map = {"cam_a": "CAM_A", "cam_b": "CAM_B", "wide": "WIDE"}
        clips = gt["clips"]
        # sample: middle of every shot + some fixed checkpoints
        times = sorted({round(0.5 * (s["start"] + s["end"]), 2) for s in shots}
                       | {round(s["start"] + 0.2, 2) for s in shots if s["duration"] > 0.6})
        cam_ok = sync_ok = 0
        worst = 0.0
        samples = []
        if sw:
            import cv2

            src = os.path.join(args.demo_dir, "wide.mp4")
            OW, OH = meta["width"], meta["height"]
            top = int(0.60 * OH)                      # compare above the caption band
            base = {L: (meta["reframe"]["crops"] or {}).get(L) for L in ("A", "B")}
            worst_mad = 0.0
            strip_ok = strip_n = 0
            for t in times:
                f = int(np.floor(t * fps + 1e-9))
                out_rgb = grab(ff, args.out, t, os.path.join(frames_dir, f"t{t:07.2f}.png"))
                src_rgb = grab(ff, src, f / fps - meta["offsets"]["wide"]["offset"],
                               os.path.join(frames_dir, "_src.png"))
                row = next(s_ for s_ in shots if s_["start"] <= f / fps < s_["end"] or s_ is shots[-1])
                want = row["camera"]
                hyp = {"WIDE": None}
                for L in ("A", "B"):
                    c = ([row["crop_x"], row["crop_y"], row["crop_w"], row["crop_h"]]
                         if want == f"CAM_{L}" else base[L])
                    if c:
                        hyp[f"CAM_{L}"] = c
                mads = {}
                for cam, c in hyp.items():
                    exp = src_rgb if c is None else src_rgb[c[1]:c[1] + c[3], c[0]:c[0] + c[2]]
                    exp = cv2.resize(exp, (OW, OH), interpolation=cv2.INTER_LANCZOS4)
                    mads[cam] = float(np.abs(exp[:top].astype(int) - out_rgb[:top].astype(int)).mean())
                got = min(mads, key=mads.get)
                ok_cam = got == want and mads[want] < 6.0
                cam_ok += ok_cam
                worst_mad = max(worst_mad, mads.get(want, 99.0))
                # frame-accurate sync from the time code: global barcode on WIDE, per-face strip
                # (mapped through the crop) on close-ups
                if want == "WIDE":
                    code_t = decode_barcode(out_rgb, gt["barcode"]) / 100.0
                else:
                    L = want[-1]
                    side = next(sd for sd, lt in letter.items() if lt == L)
                    pb = sw["people"][side]["photo_box"]
                    S_ = pb[2] - pb[0]
                    mv = sw.get("move")
                    dx = mv["dx_px"] if (mv and mv["side"] == side and f / fps >= mv["t"]) else 0
                    c = hyp[want]
                    sy = pb[1] - sw["people"][side]["strip"]["dy"] * S_ + 0.5 * max(4, int(sw["people"][side]["strip"]["h"] * S_))
                    gray = out_rgb.mean(axis=2)
                    v = 0
                    for b_ in range(16):
                        sx = pb[0] + dx + (b_ + 0.5) * S_ / 16.0
                        X = int((sx - c[0]) * OW / c[2])
                        Y = int((sy - c[1]) * OH / c[3])
                        ok_px = 2 <= X < OW - 2 and 2 <= Y < OH - 2
                        v = (v << 1) | int(ok_px and gray[Y - 1:Y + 2, X - 1:X + 2].mean() > 128)
                    code_t = v / 100.0
                    strip_n += 1
                    strip_ok += abs(code_t - f / fps) <= 1.0 / fps + 0.011
                err = code_t - f / fps
                sync_ok += abs(err) <= 1.0 / fps + 0.011
                worst = max(worst, abs(err))
                samples.append({"t": t, "edl": want, "frame": got, "mad": {k: round(v_, 2) for k, v_ in mads.items()},
                                "code_t": code_t})
            report["frame_samples"] = samples
            check("output frames match the expected source crop (A/B close-up or full wide) at sampled "
                  "frames", cam_ok == len(times),
                  f"{cam_ok}/{len(times)}; worst mean abs diff to the expected render {worst_mad:.2f}/255")
            check("frame-level sync (time code read through the crop)", sync_ok == len(times),
                  f"{sync_ok}/{len(times)} within 1 frame ({strip_ok}/{strip_n} on cropped close-ups); "
                  f"worst |err| {1000 * worst:.0f} ms")
            times = []
        for t in times:
            rgb = grab(ff, args.out, t, os.path.join(frames_dir, f"t{t:07.2f}.png"))
            got, dist = classify_camera(rgb, clips)
            want = cam_at(shots, t)
            ok_cam = cam_map[got] == want
            cam_ok += ok_cam
            ref_cs = decode_barcode(rgb, gt["barcode"])
            err = ref_cs / 100.0 - np.floor(t * fps + 1e-9) / fps  # vs the frame's own time
            src_fps = clips[got]["fps"]
            ok_sync = abs(err) <= 1.0 / src_fps + 0.011  # within one source frame
            sync_ok += ok_sync
            worst = max(worst, abs(err))
            samples.append({"t": t, "edl": want, "frame": cam_map[got], "barcode_t": ref_cs / 100.0})
        if not sw:
            report["frame_samples"] = samples
            check("rendered camera matches EDL at sampled frames", cam_ok == len(times), f"{cam_ok}/{len(times)}")
            check("frame-level sync (barcode of reference time)", sync_ok == len(times),
                  f"{sync_ok}/{len(times)} within 1 source frame; worst |err| {1000 * worst:.0f} ms")
        # caption colours
        col_a = tuple(int(tr["speakers"]["A"]["caption_color"].lstrip("#")[i:i + 2], 16) for i in (0, 2, 4))
        caps = tr["captions"]
        cap_res = []
        good = 0
        chosen = [c for c in caps if c["end"] - c["start"] > 0.5]
        chosen = chosen if single else chosen[::2]
        for c in chosen:
            t = round(0.5 * (c["start"] + c["end"]), 2)
            active = {x["speaker"] for x in caps if x["start"] <= t < x["end"]}
            rgb = grab(ff, args.out, t, os.path.join(frames_dir, f"cap_t{t:07.2f}.png"))
            a_cnt, w_cnt = caption_pixels(rgb, col_a)
            ok = (("A" in active) == (a_cnt > 400)) and (("B" in active) == (w_cnt > 400))
            good += ok
            cap_res.append({"t": t, "active": sorted(active), "blue_px": a_cnt, "white_px": w_cnt, "ok": ok})
        report["caption_samples"] = cap_res
        check("caption colours (A blue / B white) at sampled frames", good == len(chosen),
              f"{good}/{len(chosen)} frames")
        if single:  # ground truth: only the one speaker's colour may ever appear
            other = "white_px" if single == "A" else "blue_px"
            own = "blue_px" if single == "A" else "white_px"
            check(f"single speaker: only Speaker {single}'s caption colour in sampled frames",
                  bool(cap_res) and all(c[other] <= 400 and c[own] > 400 for c in cap_res),
                  f"{len(cap_res)} frames; max other-colour px {max((c[other] for c in cap_res), default=0)}")
        report["frames_dir"] = frames_dir
    else:
        print(f"(no rendered video at {args.out}; skipped frame checks)")

    n_fail = sum(1 for _, ok, _ in RESULTS if not ok)
    report["checks"] = [{"name": n, "ok": ok, "detail": d} for n, ok, d in RESULTS]
    report["passed"] = len(RESULTS) - n_fail
    report["failed"] = n_fail
    rp = args.report or f"{stem}_verify_report.json"
    with open(rp, "w") as f:
        json.dump(report, f, indent=1)
    print(f"\n{len(RESULTS) - n_fail}/{len(RESULTS)} checks passed; report: {rp}")
    return 1 if n_fail else 0


if __name__ == "__main__":
    sys.exit(main())
