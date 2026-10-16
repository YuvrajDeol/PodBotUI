"""Speech recognition with faster-whisper (CPU int8 by default)."""

from __future__ import annotations

import logging
from typing import Any, Callable, Dict, List, Optional

from .config import AsrConfig
from .datatypes import Word
from .sync import load_wav

log = logging.getLogger(__name__)


class AsrUnavailable(RuntimeError):
    pass


def transcribe(
    wav_path: str,
    cfg: AsrConfig,
    progress: Optional[Callable[[float], None]] = None,
) -> Dict[str, Any]:
    """Transcribe ``wav_path``.

    Returns ``{"language", "duration", "segments": [{start,end,text,words:[...]}],
    "words": [{word,start,end,prob}]}``.
    """
    try:
        from faster_whisper import WhisperModel
    except ImportError as e:  # pragma: no cover
        raise AsrUnavailable(
            "faster-whisper is not installed: pip install faster-whisper"
        ) from e

    log.info(
        "Loading faster-whisper model '%s' (%s/%s)", cfg.model_size, cfg.device, cfg.compute_type
    )
    model = WhisperModel(cfg.model_size, device=cfg.device, compute_type=cfg.compute_type)
    # Pass decoded samples (16 kHz mono float32) rather than a path: avoids faster-whisper's
    # PyAV decoding path (incompatible with some PyAV releases) - we already have a wav.
    audio, sr = load_wav(wav_path)
    if sr != 16000:
        raise ValueError(f"ASR expects 16 kHz audio, got {sr} Hz")
    segments_iter, info = model.transcribe(
        audio,
        language=cfg.language or None,
        beam_size=cfg.beam_size,
        word_timestamps=True,
        vad_filter=cfg.vad_filter,
        vad_parameters={"min_silence_duration_ms": 300} if cfg.vad_filter else None,
        condition_on_previous_text=cfg.condition_on_previous_text,
        initial_prompt=cfg.initial_prompt or None,
    )
    total = float(getattr(info, "duration", 0.0) or 0.0)
    log.info("ASR language: %s (p=%.2f), audio %.1fs", info.language,
             getattr(info, "language_probability", 0.0), total)
    segments: List[Dict[str, Any]] = []
    words: List[Dict[str, Any]] = []
    for seg in segments_iter:
        seg_words = []
        for w in seg.words or []:
            if not w.word.strip():
                continue
            wd = Word(word=w.word.strip(), start=float(w.start), end=float(w.end),
                      prob=float(w.probability)).to_dict()
            wd.pop("speaker", None)
            if not seg_words:
                wd["_seg_first"] = True
            seg_words.append(wd)
        segments.append({"start": float(seg.start), "end": float(seg.end),
                         "text": seg.text.strip(), "words": seg_words})
        words.extend(seg_words)
        if progress and total > 0:
            progress(min(1.0, float(seg.end) / total))
    # enforce monotonic, non-degenerate word times
    for i, w in enumerate(words):
        if w["end"] <= w["start"]:
            w["end"] = w["start"] + 0.05
        if i > 0 and w["start"] < words[i - 1]["start"]:
            w["start"] = words[i - 1]["start"]
    if cfg.refine_word_times and words:
        n = refine_word_times(words, audio, sr)
        log.info("Refined %d word boundaries against the audio energy", n)
    for w in words:
        w.pop("_seg_first", None)
    log.info("ASR: %d segments, %d words", len(segments), len(words))
    return {
        "language": info.language,
        "duration": total,
        "model": cfg.model_size,
        "segments": segments,
        "words": words,
    }


def words_from_transcript(tr: Dict[str, Any]) -> List[Word]:
    return [Word.from_dict(w) for w in tr.get("words", [])]


def _runs(mask) -> List[tuple]:
    """(start, end) index runs where mask is True."""
    out = []
    i, n = 0, len(mask)
    while i < n:
        if mask[i]:
            j = i
            while j < n and mask[j]:
                j += 1
            out.append((i, j))
            i = j
        else:
            i += 1
    return out


def refine_word_times(words: List[Dict[str, Any]], audio, sr: int, hop: float = 0.01,
                      min_pause: float = 0.12) -> int:
    """Whisper often stretches a word over an adjacent pause: the first word of a
    segment (typically the first word after a speaker change) starts right where the
    previous word ended, and a segment's last word may run into the following pause.

    Only words much longer than their text suggests are touched.  If such a word
    contains a pause (>= ``min_pause`` below an adaptive energy threshold), the
    first word of a segment is moved to start after its last pause, any other word
    is cut at its first pause; otherwise only pure leading/trailing silence is
    trimmed.  Words are modified in place; returns the number changed."""
    import numpy as np

    n = int(hop * sr)
    m = audio.size // n
    if m < 10:
        return 0
    db = 10 * np.log10(np.mean(audio[: m * n].reshape(m, n).astype(np.float64) ** 2, axis=1) + 1e-12)
    floor, peak = np.percentile(db, 10), np.percentile(db, 97)
    if peak - floor < 12:  # too noisy / compressed to tell silence from speech
        return 0
    thr = floor + 0.3 * (peak - floor)
    min_run = max(1, int(round(min_pause / hop)))
    changed = 0
    for w in words:
        expected = 0.12 + 0.085 * len(w["word"].strip(" ,.?!\"'"))
        dur = w["end"] - w["start"]
        if dur <= max(0.35, 1.4 * expected):
            continue
        i0, i1 = int(round(w["start"] / hop)), min(m, int(round(w["end"] / hop)))
        if i1 - i0 < 3:
            continue
        loud = db[i0:i1] > thr
        if not loud.any():
            continue
        ns, ne = w["start"], w["end"]
        pauses = [r for r in _runs(~loud) if r[1] - r[0] >= min_run and 0 < r[0] and r[1] < len(loud)]
        if pauses and w.get("_seg_first"):
            a = pauses[-1][1]  # speech resumes after the last inner pause
            if loud[a:].sum() * hop >= 0.03:
                ns = w["start"] + max(0, a - 3) * hop
        elif pauses:
            b = pauses[0][0]
            if loud[:b].sum() * hop >= 0.08:
                ne = w["start"] + min(len(loud), b + 5) * hop
        idx = np.flatnonzero(loud)
        ns = max(ns, w["start"] + max(0, idx[0] - 3) * hop)
        ne = min(ne, w["start"] + min(len(loud), idx[-1] + 6) * hop)
        ns = min(ns, ne - 0.08)
        if ns - w["start"] > 0.05 or w["end"] - ne > 0.05:
            w["start"], w["end"] = round(max(w["start"], ns), 3), round(min(w["end"], ne), 3)
            changed += 1
    for w in words:
        w.pop("_seg_first", None)
    return changed
