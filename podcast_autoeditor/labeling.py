"""Assign ASR words to speakers, smooth isolated flips, build turns, find overlaps."""

from __future__ import annotations

import bisect
import logging
from typing import Any, Dict, List, Optional, Sequence, Tuple

import numpy as np

from .config import LabelConfig
from .datatypes import Interval, SpeakerSegment, Turn, Word

log = logging.getLogger(__name__)


def apply_mapping(segments: Sequence[SpeakerSegment], mapping: Dict[str, str]) -> List[SpeakerSegment]:
    return [SpeakerSegment(s.start, s.end, mapping.get(s.label, s.label)) for s in segments]


def assign_words(words: Sequence[Word], segments: Sequence[SpeakerSegment],
                 max_dist: float = 1.0) -> List[Word]:
    """Label each word with the speaker whose segments overlap it the most.

    Words with no overlapping segment take the nearest segment's speaker if it is
    within ``max_dist`` seconds, else the previous word's speaker."""
    segs = sorted(segments, key=lambda s: s.start)
    starts = [s.start for s in segs]
    out: List[Word] = []
    prev: Optional[str] = None
    for w in words:
        ov: Dict[str, float] = {}
        i = max(0, bisect.bisect_right(starts, w.end) - 1)
        # walk back while segments may still overlap
        j = i
        while j >= 0 and j < len(segs):
            s = segs[j]
            if s.end < w.start - 30:  # segments are short; stop scanning far back
                break
            o = min(w.end, s.end) - max(w.start, s.start)
            if o > 0:
                ov[s.label] = ov.get(s.label, 0.0) + o
            j -= 1
        if ov:
            spk = max(ov.items(), key=lambda kv: kv[1])[0]
        else:
            best, bd = None, max_dist
            for k in (i - 1, i, i + 1):
                if 0 <= k < len(segs):
                    s = segs[k]
                    d = max(s.start - w.end, w.start - s.end, 0.0)
                    if d <= bd:
                        best, bd = s.label, d
            spk = best if best is not None else prev
        nw = Word(w.word, w.start, w.end, w.prob, spk)
        out.append(nw)
        prev = spk if spk is not None else prev
    # words still unlabelled (very beginning) take the first known label
    first = next((w.speaker for w in out if w.speaker), "A")
    for w in out:
        if w.speaker is None:
            w.speaker = first
        else:
            break
    return out


def smooth_flips(words: List[Word], cfg: LabelConfig) -> List[Word]:
    """Relabel short isolated runs (<= flip_max_words words, <= flip_max_dur s)
    surrounded on both sides by the other speaker with no pause around them.
    Real one-word answers ("Yes.") are usually separated by pauses and survive."""
    if len(words) < 3:
        return words
    ws = [Word(w.word, w.start, w.end, w.prob, w.speaker) for w in words]
    changed = True
    passes = 0
    while changed and passes < 3:
        changed = False
        passes += 1
        i = 0
        while i < len(ws):
            j = i
            while j + 1 < len(ws) and ws[j + 1].speaker == ws[i].speaker:
                j += 1
            run_len = j - i + 1
            if 0 < i and j < len(ws) - 1:
                before, after = ws[i - 1], ws[j + 1]
                dur = ws[j].end - ws[i].start
                gap_b = ws[i].start - before.end
                gap_a = after.start - ws[j].end
                if (before.speaker == after.speaker != ws[i].speaker
                        and run_len <= cfg.flip_max_words and dur <= cfg.flip_max_dur
                        and gap_b <= cfg.flip_max_gap and gap_a <= cfg.flip_max_gap):
                    for k in range(i, j + 1):
                        ws[k].speaker = before.speaker
                    changed = True
            i = j + 1
    return ws


def build_turns(words: Sequence[Word], turn_gap: float = 1.0) -> List[Turn]:
    turns: List[Turn] = []
    for w in words:
        if (turns and turns[-1].speaker == w.speaker
                and w.start - turns[-1].end <= turn_gap):
            turns[-1].words.append(w)
        else:
            turns.append(Turn(speaker=w.speaker or "A", words=[w]))
    return turns


def label_words(words: Sequence[Word], segments_ab: Sequence[SpeakerSegment],
                cfg: LabelConfig) -> List[Word]:
    labelled = assign_words(words, segments_ab)
    return smooth_flips(labelled, cfg)


# --------------------------------------------------------------------------- overlaps

def overlaps_from_words(words: Sequence[Word], min_dur: float = 0.3) -> List[Interval]:
    """Intervals where A-words and B-words overlap in time."""
    a = [w for w in words if w.speaker == "A"]
    b = [w for w in words if w.speaker == "B"]
    out: List[Interval] = []
    j = 0
    for wa in a:
        while j < len(b) and b[j].end <= wa.start:
            j += 1
        k = j
        while k < len(b) and b[k].start < wa.end:
            s, e = max(wa.start, b[k].start), min(wa.end, b[k].end)
            if e > s:
                out.append(Interval(s, e))
            k += 1
    return _merge([i for i in _merge(out, 0.2) if i.end - i.start >= min_dur], 0.0)


def _merge(iv: List[Interval], gap: float) -> List[Interval]:
    iv = sorted(iv, key=lambda x: x.start)
    out: List[Interval] = []
    for x in iv:
        if out and x.start <= out[-1].end + gap:
            out[-1].end = max(out[-1].end, x.end)
        else:
            out.append(Interval(x.start, x.end))
    return out


def _frame_db(x: np.ndarray, sr: int, hop: float) -> np.ndarray:
    n = int(hop * sr)
    m = x.size // n
    if m == 0:
        return np.zeros(0)
    p = np.mean(x[: m * n].reshape(m, n).astype(np.float64) ** 2, axis=1)
    return 10 * np.log10(p + 1e-12)


def overlaps_from_mics(
    words: Sequence[Word],
    wav_a: str,
    wav_b: str,
    offset_a: float,
    offset_b: float,
    duration: float,
    min_dur: float = 0.4,
    min_separation_db: float = 6.0,
) -> List[Interval]:
    """Detect simultaneous speech from the two close-up mics.

    Only used when each mic is clearly dominated by its own speaker (median level
    while its speaker talks exceeds the level while the other talks by
    ``min_separation_db``).  A frame is 'active' for a mic when its level is above
    the midpoint of those two medians; both active for >= ``min_dur`` => overlap."""
    from .sync import load_wav

    hop = 0.1
    xa, sr = load_wav(wav_a)
    xb, _ = load_wav(wav_b)
    n = int(np.ceil(duration / hop))
    t = np.arange(n) * hop

    def on_ref(x: np.ndarray, off: float) -> np.ndarray:
        db = _frame_db(x, sr, hop)
        idx = np.round((t - off) / hop).astype(int)
        out = np.full(n, -120.0)
        ok = (idx >= 0) & (idx < db.size)
        out[ok] = db[idx[ok]]
        return out

    da, db_ = on_ref(xa, offset_a), on_ref(xb, offset_b)
    act = {"A": np.zeros(n, bool), "B": np.zeros(n, bool)}
    for w in words:
        if w.speaker in act:
            act[w.speaker][int(w.start / hop): int(np.ceil(w.end / hop))] = True
    only_a, only_b = act["A"] & ~act["B"], act["B"] & ~act["A"]
    if only_a.sum() < 20 or only_b.sum() < 20:
        return []
    a_own, a_bleed = np.median(da[only_a]), np.median(da[only_b])
    b_own, b_bleed = np.median(db_[only_b]), np.median(db_[only_a])
    if a_own - a_bleed < min_separation_db or b_own - b_bleed < min_separation_db:
        log.info("mic overlap detection skipped (separation %.1f / %.1f dB)",
                 a_own - a_bleed, b_own - b_bleed)
        return []
    ta, tb = 0.5 * (a_own + a_bleed), 0.5 * (b_own + b_bleed)
    both = (da > ta) & (db_ > tb)
    out: List[Interval] = []
    i = 0
    while i < n:
        if both[i]:
            j = i
            while j < n and both[j]:
                j += 1
            if (j - i) * hop >= min_dur:
                out.append(Interval(i * hop, j * hop))
            i = j
        else:
            i += 1
    return _merge(out, 0.3)


def _on_timeline(x: np.ndarray, sr: int, off: float, n: int, hop: float) -> np.ndarray:
    """Frame levels (dB) of a clip resampled onto the reference timeline (-120 = no audio)."""
    db = _frame_db(x, sr, hop)
    t = np.arange(n) * hop
    idx = np.round((t - off) / hop).astype(int)
    out = np.full(n, -120.0)
    ok = (idx >= 0) & (idx < db.size)
    out[ok] = db[idx[ok]]
    return out


def _runs_of(mask: np.ndarray, hop: float, min_dur: float) -> List[Interval]:
    out: List[Interval] = []
    i, n = 0, mask.size
    while i < n:
        if mask[i]:
            j = i
            while j < n and mask[j]:
                j += 1
            if (j - i) * hop >= min_dur - 1e-9:
                out.append(Interval(round(i * hop, 3), round(j * hop, 3)))
            i = j
        else:
            i += 1
    return out


def mic_coupling_overlaps(
    da: np.ndarray,
    db: np.ndarray,
    a_only: np.ndarray,
    b_only: np.ndarray,
    hop: float,
    min_dur: float = 0.3,
    min_sep_db: float = 3.0,
    excess_db: float = 3.0,
    excess_frac: float = 0.4,
    gate_db: float = 10.0,
    win_s: float = 0.6,
    present_frac: float = 0.4,
) -> Tuple[Optional[List[Interval]], Dict[str, Any]]:
    """Overlapping speech from two close-up mics (pure; unit tested).

    ``da``/``db``: frame levels (dB) of the cam-A / cam-B mic on the timeline;
    ``a_only``/``b_only``: frames where only A / only B is known to talk.

    The bleed of B into mic A is learned from B-only frames as a level offset
    ``g_ba = median(da - db)`` (gain invariant), likewise ``g_ab`` from A-only frames.
    Given mic B's level, mic A should then read ``db + g_ba`` if only B were talking; the
    *excess* ``da - (db + g_ba)`` is ~0 when only B talks and large when A talks too.
    A talker is active in a frame when its excess passes max(excess_db, excess_frac x the
    own-vs-bleed separation) and its mic carries speech energy (gate over the mic's noise
    floor); it is *present* when active in >= ``present_frac`` of a ``win_s`` window.
    Overlap = both present for >= ``min_dur``.  The excesses and the separation are
    invariant to the mics' gains.  Returns (intervals, info) or (None, info) if the mics can't be
    calibrated (too little single-speaker speech, or bleed too close to own level)."""
    info: Dict[str, Any] = {"a_only_s": round(float(a_only.sum() * hop), 2),
                            "b_only_s": round(float(b_only.sum() * hop), 2)}
    if a_only.sum() * hop < 1.0 or b_only.sum() * hop < 1.0:
        info["reason"] = "less than 1 s of single-speaker speech per talker"
        return None, info
    g_ba = float(np.median(da[b_only] - db[b_only]))   # mic A rel. to mic B while only B talks
    g_ab = float(np.median(db[a_only] - da[a_only]))   # mic B rel. to mic A while only A talks
    # Each coupling contains the (unknown) gain difference of the two mics with opposite
    # sign; their sum does not: own-vs-bleed separation per talker = -(g_ab + g_ba) / 2.
    sep = -(g_ab + g_ba) / 2.0
    info.update(g_ab_db=round(g_ab, 2), g_ba_db=round(g_ba, 2), separation_db=round(sep, 2))
    if sep < min_sep_db:
        info["reason"] = f"mics not separated (own vs bleed {sep:.1f} dB < {min_sep_db} dB)"
        return None, info
    floor_a, floor_b = np.percentile(da[da > -119], 10), np.percentile(db[db > -119], 10)
    exc_a = da - (db + g_ba)
    exc_b = db - (da + g_ab)
    thr = max(excess_db, excess_frac * sep)
    # per-talker activity: excess over the predicted bleed, on a mic carrying speech energy
    act_a = (exc_a > thr) & (da > floor_a + gate_db)
    act_b = (exc_b > thr) & (db > floor_b + gate_db)
    # Two people talking at once alternate syllables at 50 ms resolution, so "both in the
    # same frame" only catches fragments.  A talker is *present* when active in >= 40 % of
    # a 0.6 s window; a latched turn change (A stops, B starts) satisfies both for <= 3
    # frames, well under ``min_dur``.
    k = max(3, int(round(win_s / hop)))
    kern = np.ones(k) / k
    pres_a = np.convolve(act_a.astype(float), kern, mode="same") >= present_frac
    pres_b = np.convolve(act_b.astype(float), kern, mode="same") >= present_frac
    both = pres_a & pres_b
    info.update(thr_db=round(thr, 2), window_s=win_s, present_frac=present_frac)
    return _merge(_runs_of(both, hop, min_dur), 0.2), info


def overlaps_from_mic_coupling(
    words: Sequence[Word],
    wav_a: str,
    wav_b: str,
    offset_a: float,
    offset_b: float,
    duration: float,
    min_dur: float = 0.3,
    min_sep_db: float = 3.0,
    excess_db: float = 3.0,
    excess_frac: float = 0.4,
) -> Tuple[Optional[List[Interval]], Dict[str, Any]]:
    """File-level wrapper of :func:`mic_coupling_overlaps`; single-speaker frames come
    from the labelled words (kept 0.15 s away from the other talker's words)."""
    from .sync import load_wav

    hop = 0.05
    xa, sr = load_wav(wav_a)
    xb, srb = load_wav(wav_b)
    n = int(np.ceil(duration / hop))
    da = _on_timeline(xa, sr, offset_a, n, hop)
    db = _on_timeline(xb, srb, offset_b, n, hop)
    act = {"A": np.zeros(n, bool), "B": np.zeros(n, bool)}
    for w in words:
        if w.speaker in act:
            act[w.speaker][int(w.start / hop): int(np.ceil(w.end / hop))] = True
    pad = 3  # 0.15 s guard band
    near = {k: np.convolve(v.astype(int), np.ones(2 * pad + 1), mode="same") > 0 for k, v in act.items()}
    valid = (da > -119) & (db > -119)
    a_only = act["A"] & ~near["B"] & valid
    b_only = act["B"] & ~near["A"] & valid
    return mic_coupling_overlaps(da, db, a_only, b_only, hop, min_dur, min_sep_db, excess_db,
                                 excess_frac)
