"""Decide which diarized speaker (cluster "0"/"1") is the person on cam A.

Strategies (``auto`` tries them in this order):

1. **mic** - per-camera mic level.  For each cluster, compare the RMS level of the
   cam-A audio vs the cam-B audio during that cluster's speech.  The cluster for
   which cam A is relatively louder is the person wearing / sitting at cam A's mic.
   The score ``(dA0-dB0) - (dA1-dB1)`` cancels absolute mic gains.
2. **visual** - mouth activity.  Face detection (YuNet if its model is available,
   else a Haar cascade) on each close-up at ~5 fps; motion energy in the mouth
   region is correlated with each cluster's speaking activity.
3. **first** - last resort: the first speaker heard is assumed to be A (loud warning).

``speaker_map`` "AB"/"BA" overrides (AB = first-heard speaker is A) and ``swap``
flips whatever was decided.
"""

from __future__ import annotations

import logging
import os
import subprocess
import urllib.request
from dataclasses import asdict, dataclass, field
from typing import Any, Dict, List, Optional, Sequence, Tuple

import numpy as np

from .config import MappingConfig
from .datatypes import SpeakerSegment
from .sync import load_wav

log = logging.getLogger(__name__)

_CACHE_DIR = os.path.expanduser("~/.cache/podcast_autoeditor/opencv")
_YUNET_URLS = [
    "https://github.com/opencv/opencv_zoo/raw/main/models/face_detection_yunet/"
    "face_detection_yunet_2023mar.onnx",
    "https://huggingface.co/opencv/face_detection_yunet/resolve/main/face_detection_yunet_2023mar.onnx",
]
_HAAR_URL = ("https://raw.githubusercontent.com/opencv/opencv/4.x/data/haarcascades/"
             "haarcascade_frontalface_default.xml")


@dataclass
class MappingResult:
    mapping: Dict[str, str]          # cluster label -> "A" | "B" (both -> same letter if 1 speaker)
    method: str
    confident: bool
    evidence: Dict[str, Any] = field(default_factory=dict)
    num_speakers: int = 2
    count_reasons: List[str] = field(default_factory=list)

    def to_dict(self) -> Dict[str, Any]:
        return asdict(self)

    @staticmethod
    def from_dict(d: Dict[str, Any]) -> "MappingResult":
        return MappingResult(dict(d["mapping"]), d["method"], bool(d["confident"]),
                             dict(d.get("evidence", {})), int(d.get("num_speakers", 2)),
                             list(d.get("count_reasons", [])))


IDENTITY = {"0": "A", "1": "B"}
SWAPPED = {"0": "B", "1": "A"}


# --------------------------------------------------------------------------- mic level

def _power_in(x: np.ndarray, sr: int, s: float, e: float) -> Tuple[float, int]:
    i0, i1 = max(0, int(s * sr)), min(x.size, int(e * sr))
    if i1 <= i0:
        return 0.0, 0
    seg = x[i0:i1].astype(np.float64)
    return float(np.dot(seg, seg)), i1 - i0


def mic_level_evidence(
    segments: Sequence[SpeakerSegment],
    wav_a: str,
    wav_b: str,
    offset_a: float,
    offset_b: float,
    labels: Sequence[str] = ("0", "1"),
) -> Dict[str, Any]:
    """Return per-cluster dB levels on each close-up mic and the decision score
    (positive => cluster labels[0] is on cam A)."""
    xa, sra = load_wav(wav_a)
    xb, srb = load_wav(wav_b)
    per: Dict[str, Dict[str, float]] = {}
    for lab in labels:
        pa = na = pb = nb = 0
        for sg in segments:
            if sg.label != lab:
                continue
            p, n = _power_in(xa, sra, sg.start - offset_a, sg.end - offset_a)
            pa, na = pa + p, na + n
            p, n = _power_in(xb, srb, sg.start - offset_b, sg.end - offset_b)
            pb, nb = pb + p, nb + n
        if na == 0 or nb == 0:
            return {"available": False, "reason": f"no overlapping audio for cluster {lab}"}
        da = 10 * np.log10(pa / na + 1e-12)
        db = 10 * np.log10(pb / nb + 1e-12)
        per[lab] = {"cam_a_db": round(float(da), 2), "cam_b_db": round(float(db), 2),
                    "diff_db": round(float(da - db), 2)}
    score = per[labels[0]]["diff_db"] - per[labels[1]]["diff_db"]
    return {"available": True, "per_cluster": per, "score_db": round(float(score), 2)}


def decide_from_mic(ev: Dict[str, Any], margin_db: float) -> Optional[Dict[str, str]]:
    """Two-speaker mapping from the mic score; None unless |score| > margin_db."""
    if not ev.get("available"):
        return None
    s = ev["score_db"]
    if s > margin_db:
        return dict(IDENTITY)
    if s < -margin_db:
        return dict(SWAPPED)
    return None


def mic_says_single(ev: Dict[str, Any], same_db: float, dominance_db: float) -> bool:
    """Both diarized clusters are clearly louder on the SAME close-up mic and their
    level profiles match (|score| < same_db): one voice that was split in two.
    (Caveat: two cameras recording one shared mixed feed with different gains look the
    same - documented limitation.)"""
    if not ev.get("available"):
        return False
    d = [v["diff_db"] for v in ev["per_cluster"].values()]
    return (abs(ev["score_db"]) < same_db and min(abs(x) for x in d) >= dominance_db
            and (all(x > 0 for x in d) or all(x < 0 for x in d)))


def single_speaker_mic_evidence(
    segments: Sequence[SpeakerSegment], wav_a: str, wav_b: str, offset_a: float, offset_b: float,
) -> Dict[str, Any]:
    """For ONE speaker: speech level minus each mic's own noise floor (gain invariant).
    The near mic has the higher SNR.  Also returns the raw speech level difference."""
    xa, sra = load_wav(wav_a)
    xb, srb = load_wav(wav_b)
    out: Dict[str, Any] = {"available": False}
    res = {}
    for cam, x, sr, off in (("a", xa, sra, offset_a), ("b", xb, srb, offset_b)):
        hop = int(0.05 * sr)
        m = x.size // hop
        if m < 20:
            return {"available": False, "reason": "audio too short"}
        db = 10 * np.log10(np.mean(x[: m * hop].reshape(m, hop).astype(np.float64) ** 2, axis=1) + 1e-12)
        t = (np.arange(m) + 0.5) * 0.05 + off  # frame centres on the reference timeline
        sp = np.zeros(m, bool)
        for sg in segments:
            sp |= (t >= sg.start) & (t < sg.end)
        if sp.sum() < 10:
            return {"available": False, "reason": "no speech on this mic"}
        speech = 10 * np.log10(np.mean(10 ** (db[sp] / 10)))
        floor = float(np.percentile(db[~sp], 20)) if (~sp).sum() >= 10 else float(np.percentile(db, 5))
        res[cam] = {"speech_db": round(float(speech), 2), "floor_db": round(floor, 2),
                    "snr_db": round(float(speech - floor), 2)}
    out = {"available": True, "per_mic": res,
           "snr_diff_db": round(res["a"]["snr_db"] - res["b"]["snr_db"], 2),
           "level_diff_db": round(res["a"]["speech_db"] - res["b"]["speech_db"], 2)}
    return out


def decide_single_from_mic(ev: Dict[str, Any], snr_margin_db: float) -> Optional[str]:
    """Camera letter for a single speaker from mic SNRs (falls back to a large raw level
    difference when the SNRs are inconclusive)."""
    if not ev.get("available"):
        return None
    if ev["snr_diff_db"] > snr_margin_db:
        return "A"
    if ev["snr_diff_db"] < -snr_margin_db:
        return "B"
    if abs(ev["level_diff_db"]) >= 2 * snr_margin_db:
        return "A" if ev["level_diff_db"] > 0 else "B"
    return None


# --------------------------------------------------------------------------- visual

def _download(url: str, dst: str, timeout: float = 20.0) -> Optional[str]:
    try:
        os.makedirs(os.path.dirname(dst), exist_ok=True)
        tmp = dst + ".part"
        ctx = None
        try:  # python.org builds on macOS often lack a CA bundle; prefer certifi's
            import ssl

            import certifi

            ctx = ssl.create_default_context(cafile=certifi.where())
        except ImportError:
            pass
        with urllib.request.urlopen(url, timeout=timeout, context=ctx) as r, open(tmp, "wb") as f:
            f.write(r.read())
        os.replace(tmp, dst)
        return dst
    except Exception as e:  # noqa: BLE001
        log.warning("Could not download %s: %s", url, e)
        return None


class FaceDetector:
    """YuNet (preferred) or Haar-cascade face detector returning a mouth ROI."""

    def __init__(self) -> None:
        import cv2

        self.cv2 = cv2
        self.kind = None
        self.yunet = None
        self.haar = None
        yunet_path = os.path.join(_CACHE_DIR, "face_detection_yunet_2023mar.onnx")
        if hasattr(cv2, "FaceDetectorYN"):
            for url in _YUNET_URLS:
                if os.path.isfile(yunet_path):
                    break
                _download(url, yunet_path)
            if os.path.isfile(yunet_path):
                try:
                    self.yunet = cv2.FaceDetectorYN.create(yunet_path, "", (320, 320), 0.6, 0.3, 5)
                    self.kind = "yunet"
                except Exception as e:  # noqa: BLE001
                    log.warning("YuNet init failed: %s", e)
        if self.kind is None:
            cands = []
            data_dir = getattr(getattr(cv2, "data", None), "haarcascades", None)
            if data_dir:
                cands.append(os.path.join(data_dir, "haarcascade_frontalface_default.xml"))
            local = os.path.join(_CACHE_DIR, "haarcascade_frontalface_default.xml")
            cands.append(local)
            path = next((c for c in cands if os.path.isfile(c)), None) or _download(_HAAR_URL, local)
            if path:
                clf = cv2.CascadeClassifier(path)
                if not clf.empty():
                    self.haar = clf
                    self.kind = "haar"
        if self.kind is None:
            raise RuntimeError("no OpenCV face detector available (YuNet/Haar)")

    def faces(self, bgr: np.ndarray) -> List[Dict[str, Any]]:
        """ALL faces in the frame (single-clip reframing): ``{x, y, w, h, eye_y, score,
        mouth: (x0, y0, x1, y1)}`` in pixels of ``bgr``."""
        cv2 = self.cv2
        H, W = bgr.shape[:2]
        out: List[Dict[str, Any]] = []
        if self.kind == "yunet":
            self.yunet.setInputSize((W, H))
            _, found = self.yunet.detect(bgr)
            for f in (found if found is not None else []):
                x, y, fw, fh = (float(v) for v in f[:4])
                if fw < 8 or fh < 8:
                    continue
                mrx, mry, mlx, mly = f[10], f[11], f[12], f[13]
                cx, cy = 0.5 * (mrx + mlx), 0.5 * (mry + mly)
                d = max(4.0, abs(mlx - mrx))
                out.append({"x": x, "y": y, "w": fw, "h": fh, "eye_y": float(0.5 * (f[5] + f[7])),
                            "score": float(f[14]),
                            "mouth": (cx - 0.65 * d, cy - 0.35 * d, cx + 0.65 * d, cy + 0.5 * d)})
        else:
            gray = cv2.cvtColor(bgr, cv2.COLOR_BGR2GRAY)
            for (x, y, fw, fh) in self.haar.detectMultiScale(gray, 1.1, 5, minSize=(max(20, H // 14),) * 2):
                # Haar boxes are square and looser than YuNet's: shrink to a face-sized box
                out.append({"x": x + 0.12 * fw, "y": y + 0.1 * fh, "w": 0.76 * fw, "h": 0.9 * fh,
                            "eye_y": y + 0.4 * fh, "score": 0.5,
                            "mouth": (x + 0.25 * fw, y + 0.62 * fh, x + 0.75 * fw, y + 0.95 * fh)})
        return out

    def mouth_roi(self, bgr: np.ndarray) -> Optional[Tuple[int, int, int, int]]:
        cv2 = self.cv2
        h, w = bgr.shape[:2]
        if self.kind == "yunet":
            self.yunet.setInputSize((w, h))
            _, faces = self.yunet.detect(bgr)
            if faces is None or len(faces) == 0:
                return None
            f = max(faces, key=lambda r: r[2] * r[3])
            mrx, mry, mlx, mly = f[10], f[11], f[12], f[13]
            cx, cy = 0.5 * (mrx + mlx), 0.5 * (mry + mly)
            d = max(4.0, abs(mlx - mrx))
            x0, x1 = cx - 0.65 * d, cx + 0.65 * d
            y0, y1 = cy - 0.35 * d, cy + 0.5 * d
        else:
            gray = cv2.cvtColor(bgr, cv2.COLOR_BGR2GRAY)
            faces = self.haar.detectMultiScale(gray, 1.1, 5, minSize=(max(24, h // 10),) * 2)
            if len(faces) == 0:
                return None
            x, y, fw, fh = max(faces, key=lambda r: r[2] * r[3])
            x0, x1 = x + 0.25 * fw, x + 0.75 * fw
            y0, y1 = y + 0.62 * fh, y + 0.95 * fh
        x0, y0 = int(max(0, x0)), int(max(0, y0))
        x1, y1 = int(min(w, x1)), int(min(h, y1))
        if x1 - x0 < 4 or y1 - y0 < 4:
            return None
        return x0, y0, x1, y1


def _probe_display_size(ffmpeg: str, video: str) -> Tuple[int, int]:
    """Display size of the first video stream in SQUARE pixels: rotation metadata applied
    and anamorphic footage (sample aspect ratio != 1, e.g. HDV 1440x1080 SAR 4:3) widened,
    matching what render.SQUARE_PIXELS feeds to the crop / scale filters."""
    import json
    import shutil

    sib = os.path.join(os.path.dirname(ffmpeg), "ffprobe")
    ffprobe = sib if os.path.isfile(sib) else (shutil.which("ffprobe") or "ffprobe")
    r = subprocess.run([ffprobe, "-v", "error", "-select_streams", "v:0", "-show_entries",
                        "stream=width,height,sample_aspect_ratio:stream_side_data=rotation", "-of", "json",
                        video], capture_output=True, text=True, timeout=60)
    st = (json.loads(r.stdout or "{}").get("streams") or [{}])[0]
    w, h = int(st.get("width") or 0), int(st.get("height") or 0)
    if not (w and h):
        raise RuntimeError(f"cannot read video size of {video}")
    try:
        num, den = (float(v) for v in str(st.get("sample_aspect_ratio") or "1:1").split(":"))
        if num > 0 and den > 0 and num != den:
            w = int(w * num / den / 2) * 2               # == trunc(iw*sar/2)*2 in SQUARE_PIXELS
    except ValueError:
        pass
    rot = 0
    for sd in st.get("side_data_list", []) or []:
        if "rotation" in sd:
            rot = int(round(float(sd["rotation"])))
    if rot % 180 in (90, -90):
        w, h = h, w
    return w, h


def _iter_frames(ffmpeg: str, video: str, start: float, dur: float, fps: float, width: int = 640,
                 deadline: Optional[float] = None, max_frames: Optional[int] = None,
                 info: Optional[Dict[str, Any]] = None):
    """Stream frames at ``fps`` (longer side scaled to ``width``) from an ffmpeg pipe.

    Frames are yielded one at a time (constant memory).  Stops - and kills ffmpeg - after
    ``max_frames`` frames or when ``time.monotonic()`` passes ``deadline``; a hung ffmpeg
    cannot block past the deadline because reading happens on a helper thread.  Sets
    ``info["capped"]`` / ``info["timed_out"]`` accordingly."""
    import queue
    import threading
    import time

    info = info if info is not None else {}
    w, h = _probe_display_size(ffmpeg, video)
    sc = width / max(w, h)
    W, H = max(2, int(round(w * sc / 2)) * 2), max(2, int(round(h * sc / 2)) * 2)
    cmd = [ffmpeg, "-hide_banner", "-nostdin", "-loglevel", "error", "-ss", f"{max(0.0, start):.3f}",
           "-t", f"{dur:.3f}", "-i", video, "-an", "-sn", "-dn", "-vf", f"fps={fps},scale={W}:{H}",
           "-f", "rawvideo", "-pix_fmt", "bgr24", "-"]
    p = subprocess.Popen(cmd, stdout=subprocess.PIPE, stderr=subprocess.DEVNULL)
    nbytes = W * H * 3
    q: "queue.Queue[Optional[bytes]]" = queue.Queue(maxsize=8)
    stop = threading.Event()

    def _reader() -> None:
        assert p.stdout is not None
        while not stop.is_set():
            buf = p.stdout.read(nbytes)
            if len(buf) < nbytes:
                break
            while not stop.is_set():
                try:
                    q.put(buf, timeout=0.2)
                    break
                except queue.Full:
                    continue
        q.put(None) if not stop.is_set() else None

    th = threading.Thread(target=_reader, daemon=True)
    th.start()
    n = 0
    try:
        while True:
            if max_frames is not None and n >= max_frames:
                info["capped"] = True
                break
            remaining = None if deadline is None else deadline - time.monotonic()
            if remaining is not None and remaining <= 0:
                info["timed_out"] = True
                break
            try:
                buf = q.get(timeout=remaining)
            except queue.Empty:
                info["timed_out"] = True
                break
            if buf is None:
                break
            n += 1
            yield np.frombuffer(buf, dtype=np.uint8).reshape(H, W, 3)
    finally:
        stop.set()
        p.kill()
        p.wait()
        info["frames"] = n


def _read_frames(ffmpeg: str, video: str, start: float, dur: float, fps: float,
                 width: int = 640) -> Tuple[List[np.ndarray], float]:
    """All frames of :func:`_iter_frames` as a list (kept for tests / small clips)."""
    return list(_iter_frames(ffmpeg, video, start, dur, fps, width)), max(0.0, start)


def mouth_motion_series(
    ffmpeg: str, video: str, offset: float, t0: float, t1: float, fps: float,
    detector: FaceDetector,
) -> Tuple[np.ndarray, np.ndarray, float]:
    """Return (ref_times, motion (nan where no face), face_detection_rate)."""
    cv2 = detector.cv2
    clip_start = t0 - offset
    if clip_start < 0:
        t0 -= clip_start
        clip_start = 0.0
    info: Dict[str, Any] = {}
    frames = _iter_frames(ffmpeg, video, clip_start, t1 - t0, fps,
                          deadline=getattr(detector, "deadline", None),
                          max_frames=getattr(detector, "max_frames", None), info=info)
    motion_l: List[float] = []
    prev = None
    found = 0
    last_roi = None
    for i, fr in enumerate(frames):
        motion_l.append(np.nan)
        roi = detector.mouth_roi(fr)
        if roi is not None:
            found += 1
            last_roi = roi
        elif last_roi is not None:
            roi = last_roi  # tolerate brief detection drop-outs
        if roi is None:
            prev = None
            continue
        x0, y0, x1, y1 = roi
        crop = cv2.cvtColor(fr[y0:y1, x0:x1], cv2.COLOR_BGR2GRAY)
        crop = cv2.resize(crop, (48, 32), interpolation=cv2.INTER_AREA).astype(np.float32)
        crop = (crop - crop.mean()) / (crop.std() + 1e-3)
        if prev is not None:
            motion_l[i] = float(np.mean(np.abs(crop - prev)))
        prev = crop
    if info.get("timed_out") or info.get("capped"):
        log.warning("visual analysis of %s stopped after %d frames (%s) - mouth-motion evidence "
                    "uses only that part", os.path.basename(video), info.get("frames", 0),
                    "wall-clock budget exceeded" if info.get("timed_out") else "frame cap reached")
    motion = np.asarray(motion_l, dtype=float)
    times = t0 + np.arange(motion.size) / fps
    rate = found / max(1, motion.size)
    return times, motion, rate


def activity_series(times: np.ndarray, segments: Sequence[SpeakerSegment], label: str) -> np.ndarray:
    act = np.zeros(times.size)
    for sg in segments:
        if sg.label == label:
            act[(times >= sg.start) & (times < sg.end)] = 1.0
    return act


def _corr(a: np.ndarray, b: np.ndarray) -> float:
    m = ~(np.isnan(a) | np.isnan(b))
    if m.sum() < 10:
        return float("nan")
    a, b = a[m], b[m]
    if a.std() < 1e-9 or b.std() < 1e-9:
        return 0.0
    return float(np.corrcoef(a, b)[0, 1])


def decide_from_motion(
    motion_a: Optional[np.ndarray],
    motion_b: Optional[np.ndarray],
    act0: np.ndarray,
    act1: np.ndarray,
    margin: float,
    min_term: float = 0.05,
) -> Tuple[Optional[Dict[str, str]], Dict[str, Any]]:
    """Pure decision rule (unit tested).  Positive score => cluster "0" is on cam A.

    Confident only if every usable camera agrees in sign with a term of at least
    ``min_term`` (cam A: corr0-corr1, cam B: corr1-corr0) AND |score| exceeds both
    ``margin`` and the noise floor: n samples of pure noise correlate with each activity
    with sd ~1/sqrt(n), so a term has sd ~2/sqrt(n) and the score sd sqrt(sum 4/n_cam) -
    3 sd are required (static faces used to give a "confident" mapping in ~15 % of
    one-minute clips).  Near-zero or contradicting evidence returns None."""
    terms: Dict[str, Any] = {}
    parts: List[float] = []
    var = 0.0
    speech = np.clip(act0 + act1, 0, 1)
    for cam, mo, sign in (("cam_a", motion_a, 1.0), ("cam_b", motion_b, -1.0)):
        if mo is None:
            continue
        c0, c1 = _corr(mo, act0), _corr(mo, act1)
        n_ok = int((~np.isnan(np.asarray(mo, dtype=float))).sum())
        terms[cam] = {"corr_0": c0, "corr_1": c1, "corr_speech": _corr(mo, speech), "n": n_ok,
                      "motion_mean": float(np.nanmean(mo)) if n_ok else float("nan")}
        if not (np.isnan(c0) or np.isnan(c1)):
            parts.append(sign * (c0 - c1))
            var += 4.0 / max(n_ok, 1)
    score = float(sum(parts))
    agree = bool(parts) and (all(p >= min_term for p in parts) or all(p <= -min_term for p in parts))
    floor = max(margin, 3.0 * float(np.sqrt(var)))
    ev = {"terms": terms, "score": score, "cameras_used": len(parts), "cameras_agree": agree,
          "score_needed": round(floor, 3)}
    if not parts or not agree:
        return None, ev
    if score > floor:
        return dict(IDENTITY), ev
    if score < -floor:
        return dict(SWAPPED), ev
    return None, ev


def decide_single_from_motion(ev: Dict[str, Any], margin: float = 0.1) -> Optional[str]:
    """Camera letter for a single speaker: the close-up whose mouth motion follows the
    speech activity clearly better than the other one."""
    t = ev.get("terms", {})
    ca = t.get("cam_a", {}).get("corr_speech", float("nan"))
    cb = t.get("cam_b", {}).get("corr_speech", float("nan"))
    ca = -1.0 if ca is None or np.isnan(ca) else ca
    cb = -1.0 if cb is None or np.isnan(cb) else cb
    ns = [v.get("n") for v in t.values() if v.get("n")]
    if ns:  # a correlation of n noise samples has sd ~1/sqrt(n): 0.1 is noise below ~900 samples
        margin = max(margin, 3.0 / float(np.sqrt(min(ns))))
    if max(ca, cb) < margin:
        return None
    if ca - cb > margin:
        return "A"
    if cb - ca > margin:
        return "B"
    return None


def decide_single_from_energy(ev: Dict[str, Any], ratio: float = 3.0, floor: float = 0.05) -> Optional[str]:
    """Camera letter for an (already established) single speaker from how MUCH each mouth
    moves.  One person talking almost without pause makes the speech activity nearly
    constant, so its correlation with either mouth is ~0 - but the talker's mouth moves
    several times more than the listener's.  Only used once the count is known to be 1.
    ``floor``: mean motion (normalised-patch units; talking ~0.09-0.18) below which neither
    mouth really moves - two static faces differ only in their noise (0.0001 vs 0.03)."""
    t = ev.get("terms", {})
    ma = t.get("cam_a", {}).get("motion_mean")
    mb = t.get("cam_b", {}).get("motion_mean")
    if ma is None or mb is None or np.isnan(ma) or np.isnan(mb) or max(ma, mb) < floor:
        return None
    if ma > ratio * mb:
        return "A"
    if mb > ratio * ma:
        return "B"
    return None


def motion_says_single(ev: Dict[str, Any], margin: float = 0.1) -> bool:
    """Mouth motion follows the speech on one close-up only, and the per-cluster terms
    do not separate the clusters onto different cameras."""
    return (decide_single_from_motion(ev, margin) is not None and not ev.get("cameras_agree", False))


def motion_says_two(ev: Dict[str, Any], dec: Optional[Dict[str, str]], min_corr: float = 0.1) -> bool:
    """Stricter than a confident ``decide_from_motion``: on BOTH close-ups the mouth moves
    with that camera's cluster (corr >= min_corr) and stays still for the other one
    (corr <= -min_corr). A single voice split into two interleaved clusters correlates
    alike with both, so this can overrule an embedding 'one speaker' verdict."""
    t = ev.get("terms", {})
    if not dec or "cam_a" not in t or "cam_b" not in t:
        return False
    for cam, letter in (("cam_a", "A"), ("cam_b", "B")):
        own = next((k for k, v in dec.items() if v == letter), None)
        if own is None:
            return False
        other = "1" if own == "0" else "0"
        c_own, c_other = t[cam].get(f"corr_{own}"), t[cam].get(f"corr_{other}")
        if c_own is None or c_other is None or np.isnan(c_own) or np.isnan(c_other):
            return False
        if not (c_own >= min_corr and c_other <= -min_corr):
            return False
    return True


def visual_evidence(
    ffmpeg: str,
    video_a: str,
    video_b: str,
    offset_a: float,
    offset_b: float,
    segments: Sequence[SpeakerSegment],
    cfg: MappingConfig,
) -> Tuple[Optional[Dict[str, str]], Dict[str, Any]]:
    try:
        det = FaceDetector()
    except Exception as e:  # noqa: BLE001
        return None, {"available": False, "reason": str(e)}
    speech = [s for s in segments if s.end > s.start]
    if not speech:
        return None, {"available": False, "reason": "no speech segments"}
    t0 = max(0.0, speech[0].start - 1.0)
    t1 = min(speech[-1].end + 1.0, t0 + cfg.visual_max_s)
    if (t1 - t0) * cfg.visual_fps > cfg.visual_max_frames:
        t1 = t0 + cfg.visual_max_frames / cfg.visual_fps
        log.warning("visual analysis limited to %.0f s (%d frames per camera)", t1 - t0,
                    cfg.visual_max_frames)
    import time

    # one wall-clock budget for decoding + face detection of both cameras
    det.deadline = time.monotonic() + cfg.visual_timeout_s
    det.max_frames = cfg.visual_max_frames
    motions: Dict[str, Optional[np.ndarray]] = {}
    rates: Dict[str, float] = {}
    # shared time grid; each camera's series is placed on it by its own timestamps (a clip
    # that starts after t0 yields a series starting later - pairing by index would shift it)
    times = t0 + np.arange(int(round((t1 - t0) * cfg.visual_fps))) / cfg.visual_fps
    for cam, video, off in (("a", video_a, offset_a), ("b", video_b, offset_b)):
        try:
            tt, mo, rate = mouth_motion_series(ffmpeg, video, off, t0, t1, cfg.visual_fps, det)
        except Exception as e:  # noqa: BLE001
            log.warning("visual analysis of cam %s failed: %s", cam.upper(), e)
            motions[cam], rates[cam] = None, 0.0
            continue
        rates[cam] = rate
        if rate < 0.3:
            log.warning("cam %s: faces found in only %.0f%% of sampled frames", cam.upper(), 100 * rate)
            motions[cam] = None
        else:
            grid = np.full(times.size, np.nan)
            idx = np.round((np.asarray(tt) - t0) * cfg.visual_fps).astype(int)
            ok = (idx >= 0) & (idx < times.size)
            grid[idx[ok]] = np.asarray(mo)[ok]
            motions[cam] = grid
    if all(m is None for m in motions.values()):
        return None, {"available": False, "detector": det.kind, "face_rate": rates}
    a0 = activity_series(times, segments, "0")
    a1 = activity_series(times, segments, "1")
    ma = motions.get("a")
    mb = motions.get("b")
    if ma is not None and len(ma) != len(times):
        ma = None
    if mb is not None and len(mb) != len(times):
        mb = None
    decision, ev = decide_from_motion(ma, mb, a0, a1, cfg.visual_margin, cfg.visual_min_term)
    ev.update({"available": True, "detector": det.kind, "face_rate": rates,
               "window": [round(t0, 2), round(t1, 2)]})
    return decision, ev


def visual_evidence_from_series(
    times: np.ndarray,
    motion_a: Optional[np.ndarray],
    motion_b: Optional[np.ndarray],
    segments: Sequence[SpeakerSegment],
    cfg: MappingConfig,
    info: Optional[Dict[str, Any]] = None,
) -> Tuple[Optional[Dict[str, str]], Dict[str, Any]]:
    """Same decision as :func:`visual_evidence`, for mouth-motion series that were already
    measured on a shared reference-time grid (single-clip mode: the two face regions of the
    one wide clip play the role of the two close-ups)."""
    info = dict(info or {})
    if motion_a is None and motion_b is None:
        return None, dict(info, available=False)
    times = np.asarray(times, dtype=float)
    a0 = activity_series(times, segments, "0")
    a1 = activity_series(times, segments, "1")
    decision, ev = decide_from_motion(motion_a, motion_b, a0, a1, cfg.visual_margin, cfg.visual_min_term)
    ev.update(info)
    ev["available"] = True
    return decision, ev


def _visual(vi: Dict[str, Any], segments: Sequence[SpeakerSegment], cfg: MappingConfig
            ) -> Tuple[Optional[Dict[str, str]], Dict[str, Any]]:
    """Mouth-motion evidence from precomputed series if given, else by analysing the two
    close-up clips."""
    ser = vi.get("series")
    if ser is not None:
        return visual_evidence_from_series(ser["times"], ser.get("motion_a"), ser.get("motion_b"),
                                           segments, cfg, ser.get("info"))
    return visual_evidence(vi["ffmpeg"], vi["video_a"], vi["video_b"], vi["offset_a"], vi["offset_b"],
                           segments, cfg)


# --------------------------------------------------------------------------- main

def decide_mapping(
    segments: Sequence[SpeakerSegment],
    cfg: MappingConfig,
    enrolled: bool = False,
    mic_inputs: Optional[Dict[str, Any]] = None,
    visual_inputs: Optional[Dict[str, Any]] = None,
    diar_stats: Optional[Dict[str, Any]] = None,
    num_speakers: str = "auto",
) -> MappingResult:
    """Decide how many people talk (1 or 2) and which camera each diarized cluster is.

    ``mic_inputs``: {wav_a, wav_b, offset_a, offset_b} (wav may be None);
    ``visual_inputs``: {ffmpeg, video_a, video_b, offset_a, offset_b}, or
    {series: {times, motion_a, motion_b, info}} with precomputed mouth-motion series;
    ``diar_stats``: DiarizationResult.stats (embedding 1-vs-2 verdict);
    ``num_speakers``: "auto" | "1" | "2".

    Speaker count (auto): one speaker if the embeddings say so and the mics do not
    clearly separate two voices, or if both clusters dominate the same close-up mic, or
    (no mic evidence) if only one close-up's mouth motion follows the speech."""
    evidence: Dict[str, Any] = {}
    mode = str(num_speakers).strip().lower()
    if mode not in ("auto", "1", "2"):
        raise ValueError(f"num_speakers must be auto, 1 or 2 (got {num_speakers!r})")
    if enrolled and mode != "1":
        res = MappingResult({"A": "A", "B": "B"}, "enrollment", True)
        return _finish(res, cfg)
    mi = mic_inputs or {}
    vi = visual_inputs or {}
    has_mic = bool(mi.get("wav_a") and mi.get("wav_b"))
    has_vis = bool(vi.get("series") is not None
                   or (vi.get("video_a") and vi.get("video_b") and vi.get("ffmpeg")))
    mic_ev: Optional[Dict[str, Any]] = None
    if has_mic and cfg.strategy in ("auto", "mic"):
        mic_ev = mic_level_evidence(segments, mi["wav_a"], mi["wav_b"], mi["offset_a"], mi["offset_b"])
        evidence["mic"] = mic_ev
        log.info("Mapping evidence (mic levels): %s", mic_ev)
    mic_single = bool(mic_ev) and mic_says_single(mic_ev, cfg.mic_same_db, cfg.mic_dominance_db)
    if mic_single:
        # The same level profile also arises when both cameras record ONE shared feed with
        # different input gains (two talkers). Only a real near/far mic raises the speech
        # SNR on one side (gain invariant) - require that before calling it one voice.
        snr_ev = single_speaker_mic_evidence(segments, mi["wav_a"], mi["wav_b"], mi["offset_a"],
                                             mi["offset_b"])
        evidence["mic_single_snr"] = snr_ev
        if not (snr_ev.get("available") and abs(snr_ev["snr_diff_db"]) > cfg.single_snr_margin_db):
            log.info("both clusters are louder on the same camera, but the mic SNRs match (%s): "
                     "shared feed with different gains, not one voice", snr_ev.get("snr_diff_db"))
            mic_single = False
    mic_two = (bool(mic_ev) and mic_ev.get("available", False) and not mic_single
               and abs(mic_ev["score_db"]) >= cfg.mic_margin_db)

    # ---------------------------------------------------------------- how many speakers
    reasons: List[str] = []
    n = 2
    vis: Optional[Tuple[Optional[Dict[str, str]], Dict[str, Any]]] = None  # computed at most once
    if mode == "1":
        n, reasons = 1, ["--num-speakers 1"]
    elif mode == "auto":
        st = diar_stats or {}
        emb_single = st.get("embedding_verdict", st.get("verdict")) == 1
        if (emb_single and not mic_two and not mic_single and has_vis
                and cfg.strategy in ("auto", "visual")):
            # similar voices on a shared feed: the mics are silent on the count, but the two
            # close-ups' mouths following DIFFERENT clusters overrules the embeddings (like mic_two)
            vis = _visual(vi, segments, cfg)
            evidence["visual"] = vis[1]
            log.info("Mapping evidence (mouth motion): %s", vis[1])
        if emb_single and vis is not None and motion_says_two(vis[1], vis[0]):
            log.info("embeddings suggested one speaker, but each close-up's mouth motion follows a "
                     "different cluster (score %+.2f) -> 2 speakers", vis[1].get("score", 0.0))
        elif emb_single and not mic_two:
            n = 1
            reasons = ["embeddings: " + "; ".join(st.get("reasons", []))]
            if mic_single:
                reasons.append("mics agree: both clusters dominate the same close-up mic")
        elif mic_single:
            n = 1
            d = {k: v["diff_db"] for k, v in mic_ev["per_cluster"].items()}  # type: ignore[index]
            reasons = [f"both clusters dominate the same close-up mic (A-B dB per cluster {d}, "
                       f"score {mic_ev['score_db']:+.1f} dB)"]  # type: ignore[index]
        elif emb_single and mic_two:
            log.info("embeddings suggested one speaker, but the close-up mics clearly separate "
                     "two voices (score %+.1f dB) -> 2 speakers", mic_ev["score_db"])  # type: ignore[index]
    if n == 1:
        return _finish(_single_speaker(segments, cfg, evidence, reasons, mi, vi, has_mic, has_vis), cfg)

    # ---------------------------------------------------------------- two speakers
    result: Optional[MappingResult] = None
    if cfg.speaker_map.upper() in ("AB", "BA"):
        m = IDENTITY if cfg.speaker_map.upper() == "AB" else SWAPPED
        return _finish(MappingResult(dict(m), f"manual:{cfg.speaker_map.upper()}", True, evidence), cfg)
    strategies = ["mic", "visual", "first"] if cfg.strategy == "auto" else [cfg.strategy]
    for strat in strategies:
        if strat == "mic":
            if not has_mic:
                evidence["mic"] = {"available": False, "reason": "cam A or cam B has no audio"}
                continue
            if mic_ev is None:
                mic_ev = mic_level_evidence(segments, mi["wav_a"], mi["wav_b"], mi["offset_a"], mi["offset_b"])
                evidence["mic"] = mic_ev
            dec = decide_from_mic(mic_ev, cfg.mic_margin_db)
            if dec:
                result = MappingResult(dec, "mic", True, evidence)
                break
        elif strat == "visual":
            if not has_vis:
                evidence["visual"] = {"available": False}
                continue
            if vis is None:
                vis = _visual(vi, segments, cfg)
                log.info("Mapping evidence (mouth motion): %s", vis[1])
            dec, ev = vis
            evidence["visual"] = ev
            if dec:
                result = MappingResult(dec, "visual", True, evidence)
                break
            if mode == "auto" and not mic_two and ev.get("available") and motion_says_single(ev):
                reasons = ["mouth motion follows the speech on one close-up only"]
                return _finish(_single_speaker(segments, cfg, evidence, reasons, mi, vi, has_mic,
                                               has_vis), cfg)
        elif strat == "first":
            break
        else:
            raise ValueError(f"unknown mapping strategy {strat!r}")
    if result is None:
        log.warning(
            "!!! SPEAKER->CAMERA MAPPING IS A GUESS (evidence inconclusive): assuming the first "
            "speaker heard is Speaker A (cam A). If the edit cuts to the wrong person, re-run with "
            "--swap-speakers (or tick 'Swap speakers' in the UI). !!!")
        result = MappingResult(dict(IDENTITY), "first-speaker-guess", False, evidence)
    return _finish(result, cfg)


def _single_speaker(segments, cfg, evidence, reasons, mi, vi, has_mic, has_vis) -> MappingResult:
    """One person talks: pick that person's camera (A or B)."""
    log.info("Single speaker detected: %s", "; ".join(reasons))
    letter: Optional[str] = None
    how = ""
    if cfg.speaker_map.upper() in ("AB", "BA"):
        letter, how = ("A" if cfg.speaker_map.upper() == "AB" else "B"), f"manual:{cfg.speaker_map.upper()}"
    strategies = ["mic", "visual"] if cfg.strategy == "auto" else [cfg.strategy]
    for strat in strategies if letter is None else []:
        if strat == "mic" and has_mic:
            ev = single_speaker_mic_evidence(segments, mi["wav_a"], mi["wav_b"], mi["offset_a"], mi["offset_b"])
            evidence["mic_single"] = ev
            log.info("Single-speaker mic evidence: %s", ev)
            letter = decide_single_from_mic(ev, cfg.single_snr_margin_db)
            how = "mic"
        elif strat == "visual" and has_vis:
            ev = evidence.get("visual")
            if not ev or "terms" not in ev:
                _, ev = _visual(vi, segments, cfg)
                evidence["visual"] = ev
            letter = decide_single_from_motion(ev) if ev.get("available") else None
            if letter is None and ev.get("available"):
                letter = decide_single_from_energy(ev)
            how = "visual"
        if letter:
            break
    confident = letter is not None
    if letter is None:
        log.warning("!!! SINGLE SPEAKER, CAMERA IS A GUESS: showing cam A. If it should be cam B, "
                    "re-run with --swap-speakers (or tick 'Swap speakers' in the UI). !!!")
        letter, how = "A", "guess"
    return MappingResult({"0": letter, "1": letter}, f"single-speaker:{how}", confident, evidence,
                         num_speakers=1, count_reasons=list(reasons))


def _finish(result: MappingResult, cfg: MappingConfig) -> MappingResult:
    if cfg.swap:
        result.mapping = {k: ("B" if v == "A" else "A") for k, v in result.mapping.items()}
        result.method += "+swapped"
    log.info("Speaker mapping: %s via %s (speakers=%d, confident=%s)", result.mapping, result.method,
             result.num_speakers, result.confident)
    return result
