"""Rendering with ffmpeg.

Strategy: **per-shot intermediates + concat demuxer**, then one final encode.

* Each shot is cut from its source with an accurate input seek (sync offset
  applied), normalised (fps -> scale/pad -> SAR 1 -> yuv420p; rotation handled by
  ffmpeg's autorotate) and encoded with an exact frame count (``-frames:v N``).
  Shots are independent, so they run in parallel, are cached by content hash
  (re-runs are fast) and memory stays flat for hundreds of shots.  A single
  ``filter_complex`` with hundreds of trim/split branches would decode every input
  once per branch and buffer frames, which scales badly.
* The final pass reads the segments through the concat demuxer, re-stamps video
  timestamps as ``N/fps`` (frame index -> exact timeline time, so per-segment
  rounding can never accumulate into A/V drift), burns captions (libass ``ass``
  filter, or a Pillow PNG overlay if ffmpeg lacks libass) and muxes ONE continuous
  master audio track, so cuts never touch the audio.
"""

from __future__ import annotations

import csv
import hashlib
import json
import logging
import math
import os
import shutil
from concurrent.futures import ThreadPoolExecutor, as_completed
from dataclasses import dataclass
from fractions import Fraction
from typing import Callable, Dict, List, Optional, Sequence

from . import CAM_A, CAM_B, WIDE
from .config import RenderConfig
from .datatypes import Shot
from .ffmpeg_utils import FFmpegError, FFmpegInfo, MediaInfo, run

log = logging.getLogger(__name__)


@dataclass
class SourceClip:
    path: str
    offset: float      # clip t=0 on the reference timeline
    info: MediaInfo

    def coverage(self) -> tuple:
        d = self.info.video_duration or self.info.duration
        return (self.offset, self.offset + d)


@dataclass
class AudioSource:
    path: str
    offset: float = 0.0


def fps_fraction(fps: float) -> Fraction:
    """Exact frame rate: integers stay integers; NTSC rates given as decimals
    (23.976, 29.97, 47.952, 59.94, 119.88) become N*1000/1001."""
    if abs(fps - round(fps)) < 1e-3:
        return Fraction(int(round(fps)))
    ntsc = round(fps * 1.001)
    if ntsc > 0 and abs(fps - ntsc * 1000 / 1001) < 0.002 * ntsc / 30:
        return Fraction(ntsc * 1000, 1001)
    return Fraction(fps).limit_denominator(1001)


def normalize_fps(fps: float) -> float:
    """``fps`` as the exact float of :func:`fps_fraction` (29.97 -> 30000/1001)."""
    return float(fps_fraction(fps))


def fps_str(fps: float) -> str:
    f = fps_fraction(fps)
    return str(f.numerator) if f.denominator == 1 else f"{f.numerator}/{f.denominator}"


def timecode(frame: int, fps: float) -> str:
    nf = int(round(fps))
    s, f = divmod(frame, nf)
    m, s = divmod(s, 60)
    h, m = divmod(m, 60)
    return f"{h:02d}:{m:02d}:{s:02d}:{f:02d}"


def _video_encoder(ff: FFmpegInfo, cfg: RenderConfig) -> str:
    if cfg.encoder != "auto":
        if cfg.encoder not in ff.encoders:
            raise FFmpegError(f"encoder {cfg.encoder} not available in {ff.ffmpeg}")
        return cfg.encoder
    if ff.has_libx264:
        return "libx264"
    if ff.has_videotoolbox:
        log.warning("libx264 not available; using h264_videotoolbox")
        return "h264_videotoolbox"
    raise FFmpegError("no H.264 encoder (libx264 / h264_videotoolbox) in this ffmpeg build")


def _enc_args(enc: str, crf: int, preset: str, bitrate: str, threads: int = 0) -> List[str]:
    if enc == "libx264":
        a = ["-c:v", "libx264", "-preset", preset, "-crf", str(crf), "-pix_fmt", "yuv420p"]
        if threads:
            a += ["-threads", str(threads)]
        return a
    return ["-c:v", "h264_videotoolbox", "-b:v", bitrate, "-pix_fmt", "yuv420p", "-allow_sw", "1"]


# Anamorphic sources (sample aspect ratio != 1) are first stretched to square pixels, so the
# letterbox maths and the single-clip crops (display pixels) see the picture as it is shown.
SQUARE_PIXELS = "scale=w='if(gt(sar,0),trunc(iw*sar/2)*2,iw)':h=ih,setsar=1"


def _norm_filter(cfg: RenderConfig) -> str:
    W, H = cfg.width, cfg.height
    return (f"fps={fps_str(cfg.fps)},{SQUARE_PIXELS},"
            f"scale={W}:{H}:force_original_aspect_ratio=decrease:flags=bicubic,"
            f"pad={W}:{H}:(ow-iw)/2:(oh-ih)/2:color=black,setsar=1,format=yuv420p")


def _crop_filter(cfg: RenderConfig, crop: Sequence[int]) -> str:
    """Virtual close-up: crop the source (display pixels: x, y, w, h) and scale it to the
    output with lanczos (it is always an upscale or a mild downscale of a small region)."""
    x, y, w, h = (int(v) for v in crop)
    return (f"fps={fps_str(cfg.fps)},{SQUARE_PIXELS},crop={w}:{h}:{x}:{y},"
            f"scale={cfg.width}:{cfg.height}:flags=lanczos,setsar=1,format=yuv420p")


def _segment_cmd(ff: FFmpegInfo, shot: Shot, src: Optional[SourceClip], cfg: RenderConfig,
                 enc: str, out: str, threads: int, crop: Optional[Sequence[int]] = None) -> List[str]:
    n = shot.n_frames
    fps = fps_str(cfg.fps)
    base = [ff.ffmpeg, "-hide_banner", "-nostdin", "-loglevel", "error", "-y"]
    enc_args = _enc_args(enc, cfg.intermediate_crf, cfg.intermediate_preset, "20M", threads)
    common_out = ["-frames:v", str(n), *enc_args, "-r", fps, "-an", "-sn", "-dn",
                  "-video_track_timescale", "90000", "-f", "mp4", out]
    src_start = None
    if src is not None:
        src_start = shot.start - src.offset
        clip_d = src.info.video_duration or src.info.duration
        if src_start >= clip_d - 0.5 / cfg.fps or src_start + shot.duration <= 0:
            src = None  # no footage at all for this shot
    if src is None:
        return base + ["-f", "lavfi", "-i", f"color=c=black:s={cfg.width}x{cfg.height}:r={fps}",
                       "-vf", "format=yuv420p,setsar=1"] + common_out
    assert src_start is not None
    vf = _crop_filter(cfg, crop) if crop is not None else _norm_filter(cfg)
    pre = []
    if src_start > 0:
        pre = ["-ss", f"{src_start:.6f}"]
    else:
        if src_start < -0.5 / cfg.fps:
            vf += f",tpad=start_duration={-src_start:.6f}:start_mode=add:color=black"
    vf += ",tpad=stop_mode=clone:stop=-1"
    return base + pre + ["-i", src.path, "-map", "0:v:0", "-vf", vf] + common_out


def _seg_key(shot: Shot, src: Optional[SourceClip], cfg: RenderConfig, enc: str,
             crop: Optional[Sequence[int]] = None) -> str:
    h = hashlib.sha1()
    if src is not None:
        st = os.stat(src.path)
        h.update(f"{os.path.abspath(src.path)}|{st.st_size}|{st.st_mtime_ns}|{src.offset:.6f}".encode())
    h.update(f"{shot.start:.6f}|{shot.n_frames}|{cfg.width}x{cfg.height}|{cfg.fps}|"
             f"{cfg.intermediate_crf}|{cfg.intermediate_preset}|{enc}|v3".encode())
    if crop is not None:
        h.update(("|crop=" + ",".join(str(int(v)) for v in crop)).encode())
    return h.hexdigest()[:16]


def render_segments(
    ff: FFmpegInfo,
    shots: Sequence[Shot],
    sources: Dict[str, Optional[SourceClip]],
    cfg: RenderConfig,
    seg_dir: str,
    progress: Optional[Callable[[float, str], None]] = None,
    crops: Optional[Sequence[Optional[Sequence[int]]]] = None,
) -> List[str]:
    """Encode one intermediate per shot.  ``crops`` (single-clip mode) optionally gives a
    source crop (x, y, w, h) per shot; None entries use the full frame."""
    os.makedirs(seg_dir, exist_ok=True)
    enc = _video_encoder(ff, cfg)
    cpu = os.cpu_count() or 4
    jobs = cfg.jobs or max(1, min(4, cpu // 2))
    threads = max(1, cpu // jobs)
    paths: List[str] = [""] * len(shots)
    todo = []
    for i, sh in enumerate(shots):
        src = sources.get(sh.camera)
        crop = crops[i] if crops is not None else None
        out = os.path.join(seg_dir, f"seg_{_seg_key(sh, src, cfg, enc, crop)}.mp4")
        paths[i] = out
        if os.path.isfile(out) and os.path.getsize(out) > 0:
            continue
        todo.append((i, sh, src, out, crop))
    log.info("Rendering %d segments (%d cached) with %d parallel jobs [%s]",
             len(shots), len(shots) - len(todo), jobs, enc)
    done = len(shots) - len(todo)

    def work(item) -> str:
        i, sh, src, out, crop = item
        tmp = out + ".part.mp4"
        cmd = _segment_cmd(ff, sh, src, cfg, enc, tmp, threads, crop)
        run(cmd, desc=f"segment {i} ({sh.camera} {sh.start:.2f}-{sh.end:.2f}s)")
        os.replace(tmp, out)
        return out

    with ThreadPoolExecutor(max_workers=jobs) as ex:
        futs = [ex.submit(work, it) for it in todo]
        for f in as_completed(futs):
            f.result()
            done += 1
            if progress:
                progress(done / max(1, len(shots)), f"segments {done}/{len(shots)}")
    return paths


def _concat_list(paths: Sequence[str], list_path: str) -> str:
    d = os.path.dirname(list_path)
    with open(list_path, "w") as f:
        f.write("ffconcat version 1.0\n")
        for p in paths:
            rel = os.path.relpath(p, d).replace("'", "'\\''")
            f.write(f"file '{rel}'\n")
    return list_path


def render_final(
    ff: FFmpegInfo,
    seg_paths: Sequence[str],
    audio: Optional[AudioSource],
    duration: float,
    cfg: RenderConfig,
    work_dir: str,
    out_path: str,
    ass_path: Optional[str] = None,
    overlay_list: Optional[str] = None,
    progress: Optional[Callable[[float, str], None]] = None,
) -> str:
    """Concat segments, burn captions, mux the master audio."""
    os.makedirs(os.path.dirname(os.path.abspath(out_path)) or ".", exist_ok=True)
    enc = _video_encoder(ff, cfg)
    list_path = _concat_list(seg_paths, os.path.join(work_dir, "segments.ffconcat"))
    fps = fps_str(cfg.fps)
    cmd = [ff.ffmpeg, "-hide_banner", "-nostdin", "-y", "-loglevel", "error",
           "-f", "concat", "-safe", "0", "-i", os.path.basename(list_path)]
    inputs = 1
    a_idx = c_idx = None
    if audio is not None:
        if audio.offset < 0:
            cmd += ["-ss", f"{-audio.offset:.6f}"]
        cmd += ["-i", os.path.abspath(audio.path)]
        a_idx = inputs
        inputs += 1
    if overlay_list:
        cmd += ["-f", "concat", "-safe", "0", "-i", os.path.relpath(overlay_list, work_dir)]
        c_idx = inputs
        inputs += 1
    v = f"[0:v]setpts=N/({fps}*TB),format=yuv420p"
    fc: List[str] = []
    if ass_path:
        rel = os.path.relpath(ass_path, work_dir)
        if any(ch in rel for ch in ":'\\,[];"):
            shutil.copyfile(ass_path, os.path.join(work_dir, "captions_burn.ass"))
            rel = "captions_burn.ass"
        fc.append(v + f",ass=filename={rel}[v]")
    elif c_idx is not None:
        fc.append(v + "[base]")
        fc.append(f"[{c_idx}:v]format=rgba[cap]")
        fc.append("[base][cap]overlay=x=0:y=main_h-overlay_h:eof_action=repeat:format=auto,"
                  "format=yuv420p[v]")
    else:
        fc.append(v + "[v]")
    if a_idx is not None:
        af = f"[{a_idx}:a:0]aresample=48000"
        if audio is not None and audio.offset > 0:
            ms = int(round(audio.offset * 1000))
            af += f",adelay=delays={ms}:all=1"
        af += ",apad[a]"
        fc.append(af)
    cmd += ["-filter_complex", ";".join(fc), "-map", "[v]"]
    if a_idx is not None:
        cmd += ["-map", "[a]", "-c:a", "aac", "-b:a", cfg.audio_bitrate, "-ac", "2"]
    cmd += _enc_args(enc, cfg.crf, cfg.preset, cfg.video_bitrate)
    cmd += ["-r", fps, "-t", f"{duration:.6f}", "-movflags", "+faststart",
            "-progress", "pipe:1", "-nostats", os.path.abspath(out_path)]
    log.info("Final encode -> %s", out_path)
    cb = (lambda f: progress(f, "final encode")) if progress else None
    run(cmd, desc="final render", cwd=work_dir, progress_total=duration, progress_cb=cb)
    return out_path


# --------------------------------------------------------------------------- EDL

def edl_rows(shots: Sequence[Shot], sources: Dict[str, Optional[SourceClip]], fps: float,
             crops: Optional[Sequence[Optional[Sequence[int]]]] = None) -> List[Dict]:
    """EDL rows; with ``crops`` (single-clip mode) each row also records the source crop
    rectangle used for the shot (empty for full-frame WIDE shots)."""
    rows = []
    for i, s in enumerate(shots):
        src = sources.get(s.camera)
        off = src.offset if src else 0.0
        rows.append({
            "index": i,
            "camera": s.camera,
            "reason": s.reason,
            "start": round(s.start, 6),
            "end": round(s.end, 6),
            "duration": round(s.duration, 6),
            "start_frame": s.start_frame,
            "end_frame": s.end_frame,
            "start_tc": timecode(s.start_frame, fps),
            "end_tc": timecode(s.end_frame, fps),
            "source": os.path.basename(src.path) if src else "",
            "source_in": round(s.start - off, 6),
            "source_out": round(s.end - off, 6),
        })
        if crops is not None:
            c = crops[i]
            for k, name in enumerate(("crop_x", "crop_y", "crop_w", "crop_h")):
                rows[-1][name] = int(c[k]) if c is not None else ""
    return rows


def write_edl(rows: List[Dict], json_path: str, csv_path: str, meta: Optional[Dict] = None) -> None:
    with open(json_path, "w") as f:
        json.dump({"meta": meta or {}, "shots": rows}, f, indent=1)
    with open(csv_path, "w", newline="") as f:
        if rows:
            w = csv.DictWriter(f, fieldnames=list(rows[0].keys()))
            w.writeheader()
            w.writerows(rows)
