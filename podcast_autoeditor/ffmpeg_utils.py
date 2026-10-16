"""ffmpeg / ffprobe discovery, probing and execution helpers."""

from __future__ import annotations

import json
import logging
import os
import shutil
import subprocess
from dataclasses import dataclass, field
from fractions import Fraction
from pathlib import Path
from typing import Callable, Dict, List, Optional, Sequence

log = logging.getLogger(__name__)

# Extra places to look (keg-only Homebrew formula "ffmpeg-full" ships libass).
_EXTRA_CANDIDATES = [
    "/opt/homebrew/opt/ffmpeg-full/bin/ffmpeg",
    "/usr/local/opt/ffmpeg-full/bin/ffmpeg",
    "/opt/homebrew/bin/ffmpeg",
    "/usr/local/bin/ffmpeg",
    "/usr/bin/ffmpeg",
]


class FFmpegError(RuntimeError):
    """Raised when an ffmpeg/ffprobe invocation fails."""


@dataclass
class FFmpegInfo:
    ffmpeg: str
    ffprobe: str
    filters: set = field(default_factory=set)
    encoders: set = field(default_factory=set)

    @property
    def has_libass(self) -> bool:
        return "ass" in self.filters and "subtitles" in self.filters

    @property
    def has_libx264(self) -> bool:
        return "libx264" in self.encoders

    @property
    def has_videotoolbox(self) -> bool:
        return "h264_videotoolbox" in self.encoders


def _list_tokens(ffmpeg: str, what: str) -> set:
    try:
        out = subprocess.run(
            [ffmpeg, "-hide_banner", f"-{what}"], capture_output=True, text=True, timeout=30
        ).stdout
    except Exception:  # pragma: no cover - broken binary
        return set()
    names = set()
    for line in out.splitlines():
        parts = line.split()
        if len(parts) >= 2 and not line.startswith(" ="):
            names.add(parts[1])
    return names


def _inspect(ffmpeg: str) -> Optional[FFmpegInfo]:
    if not ffmpeg or not os.path.isfile(ffmpeg) or not os.access(ffmpeg, os.X_OK):
        return None
    sibling = os.path.join(os.path.dirname(ffmpeg), "ffprobe")
    ffprobe = sibling if os.path.isfile(sibling) else (shutil.which("ffprobe") or "")
    if not ffprobe:
        return None
    return FFmpegInfo(
        ffmpeg=ffmpeg,
        ffprobe=ffprobe,
        filters=_list_tokens(ffmpeg, "filters"),
        encoders=_list_tokens(ffmpeg, "encoders"),
    )


_CACHE: Dict[str, FFmpegInfo] = {}


def find_ffmpeg(explicit: Optional[str] = None, prefer_libass: bool = True) -> FFmpegInfo:
    """Locate ffmpeg. Order: explicit arg, $PODCAST_AUTOEDITOR_FFMPEG, PATH, known
    Homebrew locations. If ``prefer_libass`` a build with the ``ass`` filter wins."""
    key = f"{explicit}|{prefer_libass}|{os.environ.get('PODCAST_AUTOEDITOR_FFMPEG')}"
    if key in _CACHE:
        return _CACHE[key]
    if explicit or os.environ.get("PODCAST_AUTOEDITOR_FFMPEG"):
        path = explicit or os.environ["PODCAST_AUTOEDITOR_FFMPEG"]
        info = _inspect(path)
        if info is None:
            raise FFmpegError(f"ffmpeg binary not usable: {path} (is ffprobe next to it?)")
        _CACHE[key] = info
        return info
    candidates: List[str] = []
    w = shutil.which("ffmpeg")
    if w:
        candidates.append(w)
    candidates += [c for c in _EXTRA_CANDIDATES if c not in candidates]
    found: List[FFmpegInfo] = []
    seen = set()
    for c in candidates:
        real = os.path.realpath(c) if os.path.exists(c) else c
        if real in seen:
            continue
        seen.add(real)
        info = _inspect(c)
        if info is not None:
            found.append(info)
            if not prefer_libass or info.has_libass:
                break
    if not found:
        raise FFmpegError(
            "ffmpeg/ffprobe not found. Install with `brew install ffmpeg` "
            "(or `brew install ffmpeg-full` for libass caption burning)."
        )
    best = next((f for f in found if f.has_libass), found[0]) if prefer_libass else found[0]
    log.info(
        "Using ffmpeg %s (libass=%s, libx264=%s, videotoolbox=%s)",
        best.ffmpeg, best.has_libass, best.has_libx264, best.has_videotoolbox,
    )
    _CACHE[key] = best
    return best


def run(
    cmd: Sequence[str],
    desc: str = "ffmpeg",
    cwd: Optional[str] = None,
    progress_total: Optional[float] = None,
    progress_cb: Optional[Callable[[float], None]] = None,
) -> subprocess.CompletedProcess:
    """Run a command; raise FFmpegError with the stderr tail on failure.

    If ``progress_cb`` is given the command must contain ``-progress pipe:1``; the
    callback receives a 0..1 fraction based on ``out_time_us``/``progress_total``.
    """
    cmd = [str(c) for c in cmd]
    log.debug("%s: %s", desc, " ".join(_q(c) for c in cmd))
    if progress_cb is None or not progress_total:
        proc = subprocess.run(cmd, capture_output=True, text=True, cwd=cwd)
        if proc.returncode != 0:
            raise FFmpegError(f"{desc} failed (exit {proc.returncode}):\n{_tail(proc.stderr)}")
        return proc
    proc = subprocess.Popen(
        cmd, stdout=subprocess.PIPE, stderr=subprocess.PIPE, text=True, cwd=cwd
    )
    stderr_chunks: List[str] = []
    import threading

    def _drain() -> None:
        assert proc.stderr is not None
        for line in proc.stderr:
            stderr_chunks.append(line)
            if len(stderr_chunks) > 400:
                del stderr_chunks[:200]

    t = threading.Thread(target=_drain, daemon=True)
    t.start()
    assert proc.stdout is not None
    for line in proc.stdout:
        line = line.strip()
        if line.startswith("out_time_us=") or line.startswith("out_time_ms="):
            try:
                us = float(line.split("=", 1)[1])
                progress_cb(max(0.0, min(1.0, us / 1e6 / progress_total)))
            except ValueError:
                pass
    proc.wait()
    t.join(timeout=5)
    if proc.returncode != 0:
        raise FFmpegError(f"{desc} failed (exit {proc.returncode}):\n{_tail(''.join(stderr_chunks))}")
    return subprocess.CompletedProcess(cmd, proc.returncode, "", "".join(stderr_chunks))


def _q(s: str) -> str:
    return f"'{s}'" if (" " in s or not s) else s


def _tail(s: str, n: int = 25) -> str:
    lines = (s or "").strip().splitlines()
    return "\n".join(lines[-n:])


@dataclass
class MediaInfo:
    path: str
    duration: float
    has_video: bool
    has_audio: bool
    width: int = 0
    height: int = 0
    fps: float = 0.0
    rotation: int = 0
    sample_rate: int = 0
    video_duration: float = 0.0
    audio_duration: float = 0.0

    def to_dict(self) -> Dict[str, object]:
        return dict(self.__dict__)


def _parse_rate(s: Optional[str]) -> float:
    if not s or s in ("0/0", "N/A"):
        return 0.0
    try:
        return float(Fraction(s))
    except (ValueError, ZeroDivisionError):
        return 0.0


def probe(ff: FFmpegInfo, path: str) -> MediaInfo:
    if not os.path.isfile(path):
        raise FileNotFoundError(f"Input not found: {path}")
    cmd = [
        ff.ffprobe, "-v", "error", "-print_format", "json",
        "-show_format", "-show_streams", path,
    ]
    proc = subprocess.run(cmd, capture_output=True, text=True)
    if proc.returncode != 0:
        raise FFmpegError(f"ffprobe failed on {path}:\n{_tail(proc.stderr)}")
    data = json.loads(proc.stdout or "{}")
    streams = data.get("streams", [])
    fmt = data.get("format", {})
    vs = next((s for s in streams if s.get("codec_type") == "video"
               and not s.get("disposition", {}).get("attached_pic")), None)
    as_ = next((s for s in streams if s.get("codec_type") == "audio"), None)
    dur = float(fmt.get("duration") or 0.0)
    info = MediaInfo(path=path, duration=dur, has_video=vs is not None, has_audio=as_ is not None)
    if vs is not None:
        info.width = int(vs.get("width") or 0)
        info.height = int(vs.get("height") or 0)
        info.fps = _parse_rate(vs.get("avg_frame_rate")) or _parse_rate(vs.get("r_frame_rate"))
        info.video_duration = float(vs.get("duration") or dur or 0.0)
        rot = 0
        for sd in vs.get("side_data_list", []) or []:
            if "rotation" in sd:
                try:
                    rot = int(round(float(sd["rotation"])))
                except (TypeError, ValueError):
                    pass
        if not rot and "rotate" in (vs.get("tags") or {}):
            try:
                rot = int(vs["tags"]["rotate"])
            except ValueError:
                pass
        info.rotation = rot % 360
    if as_ is not None:
        info.sample_rate = int(as_.get("sample_rate") or 0)
        info.audio_duration = float(as_.get("duration") or dur or 0.0)
    if not info.duration:
        info.duration = max(info.video_duration, info.audio_duration)
    return info


def extract_audio(
    ff: FFmpegInfo, src: str, dst: str, sr: int = 16000, mono: bool = True
) -> str:
    """Decode the first audio stream of ``src`` to a PCM wav."""
    Path(dst).parent.mkdir(parents=True, exist_ok=True)
    tmp = dst + ".tmp.wav"
    cmd = [ff.ffmpeg, "-hide_banner", "-nostdin", "-y", "-i", src, "-vn", "-sn", "-dn",
           "-map", "0:a:0", "-ac", "1" if mono else "2", "-ar", str(sr),
           "-c:a", "pcm_s16le", tmp]
    run(cmd, desc=f"extract audio from {os.path.basename(src)}")
    os.replace(tmp, dst)
    return dst
