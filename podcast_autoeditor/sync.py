"""Audio-based multicam synchronisation.

Convention: ``offset`` of a clip = the time on the reference timeline at which the
clip's own t=0 occurs.  ``clip_time = ref_time - offset``.  A clip that started
recording 1.3 s *after* the reference has offset +1.3; one that started 0.5 s
*before* has offset -0.5.

Offsets are found with generalised cross-correlation (GCC with partial PHAT
whitening), which is robust to the different gains/EQ of a close-up mic vs. a
room mic, evaluated only over lags inside +/- ``max_lag_s``.
"""

from __future__ import annotations

import logging
from dataclasses import asdict, dataclass
from typing import Dict, Optional, Tuple

import numpy as np

log = logging.getLogger(__name__)


@dataclass
class SyncResult:
    offset: float
    confidence: float          # peak-to-sidelobe ratio (>= ~3 is very reliable)
    method: str                # gcc | manual | reference | none
    note: str = ""

    def to_dict(self) -> Dict[str, object]:
        return asdict(self)


def load_wav(path: str) -> Tuple[np.ndarray, int]:
    """Load a wav as mono float32 in [-1, 1]."""
    try:
        import soundfile as sf

        x, sr = sf.read(path, dtype="float32", always_2d=True)
        return x.mean(axis=1).astype(np.float32), int(sr)
    except ImportError:  # pragma: no cover
        from scipy.io import wavfile

        sr, x = wavfile.read(path)
        x = x.astype(np.float32)
        if x.ndim > 1:
            x = x.mean(axis=1)
        if np.abs(x).max() > 1.5:
            x /= 32768.0
        return x, int(sr)


def _next_pow2(n: int) -> int:
    return 1 << (int(n - 1).bit_length())


def gcc_lag(
    ref: np.ndarray, sig: np.ndarray, sr: int, max_lag_s: float, beta: float = 0.7
) -> Tuple[float, float]:
    """Return (lag_seconds, confidence) such that ``sig[n] ~= ref[n + lag*sr]``.

    ``confidence`` is the ratio of the correlation peak to the largest value
    outside a +/-20 ms neighbourhood of the peak (peak-to-sidelobe ratio).
    """
    ref = np.asarray(ref, dtype=np.float64)
    sig = np.asarray(sig, dtype=np.float64)
    if ref.size == 0 or sig.size == 0:
        return 0.0, 0.0
    ref = ref - ref.mean()
    sig = sig - sig.mean()
    if np.max(np.abs(sig)) < 1e-6 or np.max(np.abs(ref)) < 1e-6:
        return 0.0, 0.0
    max_lag = int(round(max_lag_s * sr))
    nfft = _next_pow2(ref.size + sig.size)
    R = np.fft.rfft(ref, nfft)
    S = np.fft.rfft(sig, nfft)
    X = R * np.conj(S)
    mag = np.abs(X)
    X = X / (np.power(mag, beta) + 1e-12 * (mag.max() + 1e-30))
    cc = np.fft.irfft(X, nfft)
    max_lag = min(max_lag, nfft // 2 - 1)
    # lags -max_lag..+max_lag
    window = np.concatenate([cc[nfft - max_lag:], cc[: max_lag + 1]])
    lags = np.arange(-max_lag, max_lag + 1)
    i = int(np.argmax(window))
    peak = float(window[i])
    if peak <= 0:
        return 0.0, 0.0
    guard = max(1, int(0.02 * sr))
    mask = np.ones_like(window, dtype=bool)
    mask[max(0, i - guard): i + guard + 1] = False
    side = float(np.max(np.abs(window[mask]))) if mask.any() else 0.0
    conf = peak / side if side > 0 else float("inf")
    # parabolic sub-sample refinement
    frac = 0.0
    if 0 < i < window.size - 1:
        a, b, c = window[i - 1], window[i], window[i + 1]
        den = a - 2 * b + c
        if abs(den) > 1e-12:
            frac = float(np.clip(0.5 * (a - c) / den, -0.5, 0.5))
    return (lags[i] + frac) / sr, float(conf)


def estimate_offset(
    ref: np.ndarray,
    sig: np.ndarray,
    sr: int,
    max_lag_s: float = 10.0,
    analysis_s: float = 180.0,
    min_confidence: float = 2.0,
) -> Tuple[float, float]:
    """Estimate the offset (seconds) of ``sig`` (a clip) relative to ``ref``.

    Tries a chunk at the start; if confidence is low, also tries chunks later in
    the recording and keeps the most confident result.
    """
    L = int(analysis_s * sr)
    pad = int(max_lag_s * sr)
    starts = [0]
    total = min(ref.size, sig.size)
    if total > 2 * L:
        starts += [total // 3, (2 * total) // 3]
    best = (0.0, 0.0)
    for k, st in enumerate(starts):
        r = ref[st: st + L + pad]
        s = sig[max(0, st - pad): st + L + pad]
        s_start = max(0, st - pad)
        lag, conf = gcc_lag(r, s, sr, max_lag_s + (st - s_start) / sr)
        # sig[n] ~ ref_chunk[n + lag] with chunks starting at st / s_start
        offset = lag + (st - s_start) / sr
        if abs(offset) > max_lag_s + 1e-6:
            conf = 0.0
        if conf > best[1]:
            best = (offset, conf)
        if k == 0 and conf >= min_confidence:
            break
    return best


OFFSET_FLAG = {"cam_a": "--offset-a", "cam_b": "--offset-b", "wide": "--offset-wide"}


def compute_offsets(
    ref_wav: str,
    clip_wavs: Dict[str, Optional[str]],
    max_lag_s: float = 10.0,
    analysis_s: float = 180.0,
    min_confidence: float = 2.0,
    manual: Optional[Dict[str, Optional[float]]] = None,
    enabled: bool = True,
    reference_name: Optional[str] = None,
    force: bool = False,
) -> Dict[str, SyncResult]:
    """Compute offsets for every clip.  ``clip_wavs`` maps clip name -> wav path
    (or None if the clip has no audio).

    An estimate with confidence below ``min_confidence`` is NOT applied (offset 0.0,
    method "low-confidence", the rejected estimate is kept in ``note``) unless
    ``force`` is set; manual offsets always win."""
    manual = manual or {}
    out: Dict[str, SyncResult] = {}
    ref = None
    sr = 16000
    for name, wav in clip_wavs.items():
        flag = OFFSET_FLAG.get(name, "--offset-*")
        if manual.get(name) is not None:
            out[name] = SyncResult(float(manual[name]), float("nan"), "manual")
            log.info("Sync %s: manual offset %+.3f s", name, out[name].offset)
            continue
        if name == reference_name:
            out[name] = SyncResult(0.0, float("inf"), "reference")
            continue
        if not enabled:
            out[name] = SyncResult(0.0, float("nan"), "none", "sync disabled")
            continue
        if wav is None:
            msg = (f"{name} has no audio, so it cannot be synced automatically -> assuming it "
                   f"started together with the reference (offset 0.0 s). If it did not, pass "
                   f"{flag} SECONDS (where the clip's first frame falls on the timeline).")
            log.warning("Sync %s", msg)
            out[name] = SyncResult(0.0, 0.0, "none", msg)
            continue
        if ref is None:
            ref, sr = load_wav(ref_wav)
        sig, sr2 = load_wav(wav)
        if sr2 != sr:
            raise ValueError(f"sample-rate mismatch {sr2} vs {sr} for {name}")
        off, conf = estimate_offset(ref, sig, sr, max_lag_s, analysis_s, min_confidence)
        if conf < min_confidence and not force:
            msg = (f"{name}: LOW sync confidence ({conf:.2f} < {min_confidence}) for estimated offset "
                   f"{off:+.3f} s -> NOT applied, using 0.0 s. Pass {flag} SECONDS, or --force-sync "
                   f"to apply the estimate anyway.")
            log.warning("Sync %s", msg)
            out[name] = SyncResult(0.0, float(conf), "low-confidence", msg)
            continue
        note = ""
        if conf < min_confidence:
            note = f"low confidence ({conf:.2f}) - applied because of --force-sync"
            log.warning("Sync %s: %s, offset %+.3f s", name, note, off)
        else:
            log.info("Sync %s: offset %+.3f s (confidence %.1f)", name, off, conf)
        out[name] = SyncResult(float(off), float(conf), "gcc", note)
    return out
