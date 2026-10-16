"""Shot planner: turns/words -> camera shots.

Pure function, no I/O.  The timeline is discretised to output frames, painted in
priority order and run-length encoded, so the result always covers
[0, duration] with no gaps/overlaps and frame-accurate boundaries.

Painting order (later wins):
  1. everything WIDE ("no_speech": intro, long silences, outro)
  2. each turn's close-up (A -> CAM_A, B -> CAM_B) from ``start - lead`` until the
     next turn's ``start - lead`` (or ``end + silence_hold`` before a long silence /
     ``end + outro_hold`` after the last turn)
  3. WIDE over overlapping speech
  4. WIDE over rapid back-and-forth runs
  5. WIDE over question -> answer exchanges
Then: camera availability fix, min-shot merging (preferring WIDE), same-camera
merging.
"""

from __future__ import annotations

import math
import re
from dataclasses import dataclass
from typing import Dict, List, Optional, Sequence, Tuple

import numpy as np

from . import CAM_A, CAM_B, WIDE
from .config import PlannerConfig
from .datatypes import Interval, Shot, Turn, Word

_CAMS = [CAM_A, CAM_B, WIDE]
_CAM_IDX = {c: i for i, c in enumerate(_CAMS)}
SPEAKER_CAM = {"A": CAM_A, "B": CAM_B}

# reasons (index into list)
_REASONS = ["no_speech", "speaker_A", "speaker_B", "overlap", "rapid_exchange",
            "question_answer", "camera_unavailable", "merged_short"]
_R = {r: i for i, r in enumerate(_REASONS)}
_WIDE_PRIORITY = ["question_answer", "rapid_exchange", "overlap", "no_speech"]

_WH = {"what", "why", "how", "who", "whom", "whose", "where", "when", "which"}
_WH_CONTRACTED = {"what's", "how's", "who's", "where's", "when's", "why's", "what're", "who're",
                  "how're", "where're"}
_AUX = {"do", "does", "did", "is", "are", "was", "were", "am", "can", "could", "would", "will",
        "should", "shall", "have", "has", "had", "may", "might", "must", "isn't", "aren't", "don't",
        "doesn't", "didn't", "won't", "wouldn't", "can't", "couldn't", "shouldn't", "haven't",
        "hasn't", "wasn't", "weren't"}
# words that may sit between a wh-word and its auxiliary ("how long does", "which one is")
_WH_MODIFIERS = {"long", "much", "many", "often", "far", "old", "big", "else", "one", "kind",
                 "time", "exactly", "come"}
_SUBJECTS = {"i", "you", "we", "they", "he", "she", "it", "there", "this", "that", "these", "those",
             "your", "my", "our", "their", "his", "her", "its", "the", "a", "an", "anyone",
             "anybody", "someone", "somebody", "everyone", "people", "any"}
_LEADING_FILLERS = {"so", "and", "but", "well", "okay", "ok", "now", "oh", "hey", "then", "alright",
                    "um", "uh", "also", "anyway", "actually", "just", "yeah"}
_TAGS = ("right", "correct", "isn't it", "aren't you", "aren't they", "aren't we", "don't you",
         "don't they", "don't we", "doesn't it", "doesn't he", "doesn't she", "didn't you",
         "didn't it", "didn't they", "wasn't it", "weren't you", "weren't they", "won't you",
         "won't it", "can't you", "haven't you", "hasn't it", "wouldn't you", "wouldn't it",
         "shouldn't we", "isn't that", "is that right", "or not", "huh", "eh")
_STRIP = "\"'“”‘’()[],.;:!?…"
# interviewer prompts that ask for an answer without being interrogative ("Tell me how you
# started.") - Whisper writes them with a period
_PROMPTS = (("tell", "me"), ("tell", "us"), ("walk", "me", "through"), ("walk", "us", "through"),
            ("talk", "me", "through"), ("talk", "us", "through"))


@dataclass
class PlanEvent:
    start: float
    end: float
    kind: str
    detail: str = ""


def question_start(sentence: Sequence[Word]) -> Optional[int]:
    """Heuristic for sentences without a "?" (Whisper sometimes drops it).  Returns the
    index of the word where the question starts, or None.

    * interrogative opening: a wh-word followed by an auxiliary (directly or after one
      modifier: "how long does", "which one is"), or "what's/how's/..." - so statements
      like "What I mean is..." or "When I was young..." are *not* questions;
    * auxiliary inversion: an auxiliary followed by a subject ("do you", "is it",
      "can I", "did the") with at least one more word after it;
    * up to two leading fillers ("so", "and", "well", "okay"...) are skipped; in a run
      of words without final punctuation the onset is also searched in the last 12 words;
    * a trailing tag: "..., right", "..., correct", "isn't it", "don't you", "or not", "huh";
    * an interviewer prompt: "tell me/us ...", "walk/talk me/us through ...".
    """
    toks = [w.text.strip(_STRIP).lower() for w in sentence]
    n = len(toks)
    if n == 0:
        return None
    last_raw = sentence[-1].text.rstrip("\"'”’)]")
    if last_raw.endswith("!"):  # exclamations / imperatives ("Do it now!") are not questions
        return None
    if n >= 3:
        for tag in _TAGS:
            k = len(tag.split())
            if n > k and toks[-k:] == tag.split():
                # one-word tags must be set off by a comma ("..., right") to count
                if k == 1 and not sentence[-2].text.endswith(","):
                    continue
                return 0

    def opens_question(i: int) -> bool:
        t = toks[i]
        if t in _WH_CONTRACTED:
            return i + 1 < n
        if t in _WH:
            nxt = toks[i + 1: i + 3]
            if nxt and nxt[0] in _AUX:
                return True
            return len(nxt) == 2 and nxt[0] in _WH_MODIFIERS and nxt[1] in (_AUX | _SUBJECTS)
        if any(tuple(toks[i:i + len(p)]) == p and i + len(p) < n for p in _PROMPTS):
            return True
        return t in _AUX and i + 2 < n and toks[i + 1] in _SUBJECTS

    i = 0
    while i < min(2, n - 1) and toks[i] in _LEADING_FILLERS:
        i += 1
    if opens_question(i):
        return i
    if not last_raw.endswith((".", "!", "…")):  # unpunctuated run: onset may come late
        for j in range(max(i + 1, n - 12), n - 1):
            if opens_question(j):
                return j
    return None


def _is_question(sentence: Sequence[Word], heuristic: bool) -> Tuple[bool, int]:
    """(is_question, index of the first word of the question within ``sentence``)."""
    if not sentence:
        return False, 0
    if sentence[-1].text.rstrip("\"'”’)]").endswith("?"):
        return True, 0
    if heuristic:
        k = question_start(sentence)
        if k is not None:
            return True, k
    return False, 0


def find_qa_events(turns: Sequence[Turn], cfg: PlannerConfig) -> List[PlanEvent]:
    """Question -> answer: a turn whose final sentence is a question, followed by a
    turn of the *other* speaker starting within ``qa_gap`` seconds.  WIDE runs from
    the start of the question sentence (capped to ``qa_max_question`` s before the
    question ends) until ``qa_answer_hold`` s into the answer (or the answer's end)."""
    ev: List[PlanEvent] = []
    for q, a in zip(turns, turns[1:]):
        if q.speaker == a.speaker:
            continue
        if a.start - q.end > cfg.qa_gap:
            continue
        sents = q.sentences()
        if not sents:
            continue
        is_q, k = _is_question(sents[-1], cfg.question_heuristic)
        if not is_q:
            continue
        qs = max(sents[-1][k].start, q.end - cfg.qa_max_question)
        end = min(a.start + cfg.qa_answer_hold, a.end)
        how = "?" if sents[-1][-1].text.rstrip("\"'”’)]").endswith("?") else "heuristic"
        ev.append(PlanEvent(qs - cfg.lead, end, "question_answer",
                            f"{q.speaker}->{a.speaker} ({how}): "
                            f"{' '.join(w.text for w in sents[-1][k:])[:60]}"))
    return ev


def find_rapid_events(turns: Sequence[Turn], cfg: PlannerConfig) -> List[PlanEvent]:
    """>= ``rapid_turns`` speaker changes within ``rapid_window`` seconds."""
    changes: List[Tuple[float, int]] = []  # (time of change, index of new turn)
    for i in range(1, len(turns)):
        if turns[i].speaker != turns[i - 1].speaker:
            changes.append((turns[i].start, i))
    n = cfg.rapid_turns
    ev: List[PlanEvent] = []
    if n < 1 or len(changes) < n:
        return ev
    for k in range(len(changes) - n + 1):
        t_first, i_first = changes[k]
        t_last, i_last = changes[k + n - 1]
        if t_last - t_first <= cfg.rapid_window:
            prev_turn = turns[i_first - 1]
            last_turn = turns[i_last]
            s = max(prev_turn.start, t_first - cfg.rapid_pad) - cfg.lead
            e = min(last_turn.end, t_last + cfg.rapid_pad)
            ev.append(PlanEvent(s, e, "rapid_exchange", f"{n} changes in {t_last - t_first:.1f}s"))
    return _merge_events(ev)


def _merge_events(ev: List[PlanEvent]) -> List[PlanEvent]:
    ev = sorted(ev, key=lambda e: e.start)
    out: List[PlanEvent] = []
    for e in ev:
        if out and e.start <= out[-1].end and e.kind == out[-1].kind:
            out[-1].end = max(out[-1].end, e.end)
        else:
            out.append(PlanEvent(e.start, e.end, e.kind, e.detail))
    return out


def plan_shots(
    turns: Sequence[Turn],
    duration: float,
    fps: float,
    cfg: Optional[PlannerConfig] = None,
    overlaps: Optional[Sequence[Interval]] = None,
    coverage: Optional[Dict[str, Tuple[float, float]]] = None,
    events_out: Optional[List[PlanEvent]] = None,
) -> List[Shot]:
    """Plan the edit.

    Args:
        turns: speaker turns (speaker "A"/"B"), time-ordered.
        duration: timeline length in seconds (master audio length).
        fps: output frame rate; all boundaries snap to 1/fps.
        cfg: thresholds.
        overlaps: intervals of overlapping speech (-> WIDE).
        coverage: optional {camera: (start, end)} availability on the timeline;
            frames where the chosen camera has no footage fall back to WIDE (or any
            available camera).
        events_out: if given, detected WIDE events are appended (for logging/EDL).
    """
    cfg = cfg or PlannerConfig()
    if duration <= 0:
        raise ValueError("duration must be > 0")
    if fps <= 0:
        raise ValueError("fps must be > 0")
    n = int(math.ceil(duration * fps - 1e-6))
    cams = np.full(n, _CAM_IDX[WIDE], dtype=np.int8)
    why = np.full(n, _R["no_speech"], dtype=np.int8)

    def fr(t: float) -> int:
        return int(min(n, max(0, round(t * fps))))

    def paint(s: float, e: float, cam: str, reason: str) -> None:
        a, b = fr(s), fr(e)
        if b > a:
            cams[a:b] = _CAM_IDX[cam]
            why[a:b] = _R[reason]

    turns = sorted([t for t in turns if t.words], key=lambda t: t.start)
    # 2. close-ups
    for i, t in enumerate(turns):
        cam = SPEAKER_CAM.get(t.speaker)
        if cam is None:
            continue
        start = t.start - cfg.lead
        if i + 1 < len(turns):
            nxt = turns[i + 1]
            gap = nxt.start - t.end
            end = (nxt.start - cfg.lead) if gap <= cfg.silence_wide else (t.end + cfg.silence_hold)
        else:
            end = t.end + cfg.outro_hold
        paint(start, end, cam, f"speaker_{t.speaker}")
    events: List[PlanEvent] = []
    # 3. overlaps
    if cfg.overlap_wide and overlaps:
        for o in overlaps:
            if o.end - o.start >= cfg.min_overlap:
                events.append(PlanEvent(o.start - cfg.lead, o.end, "overlap"))
    # 4. rapid exchanges
    if cfg.rapid_enabled:
        events += find_rapid_events(turns, cfg)
    # 5. Q -> A
    if cfg.qa_enabled:
        events += find_qa_events(turns, cfg)
    for kind in ("overlap", "rapid_exchange", "question_answer"):
        for e in events:
            if e.kind == kind:
                paint(e.start, e.end, WIDE, kind)
    if events_out is not None:
        events_out.extend(events)

    # camera availability
    if coverage:
        avail = {c: np.zeros(n, dtype=bool) for c in _CAMS}
        for c in _CAMS:
            if c in coverage and coverage[c] is not None:
                s, e = coverage[c]
                # half-frame tolerance: a source ending mid-frame still provides that frame
                a, b = int(math.ceil(s * fps - 0.5)), int(math.floor(e * fps + 0.5))
                avail[c][max(0, a):max(0, min(n, b))] = True
            elif c not in coverage:
                avail[c][:] = True
        for c in (CAM_A, CAM_B, WIDE):
            bad = (cams == _CAM_IDX[c]) & ~avail[c]
            if not bad.any():
                continue
            for alt in (WIDE, CAM_A, CAM_B):
                if alt == c:
                    continue
                fix = bad & avail[alt]
                cams[fix] = _CAM_IDX[alt]
                why[fix] = _R["camera_unavailable"]
                bad &= ~fix
            # frames where nothing is available keep their camera (renderer pads black)

    shots = _rle(cams, why)
    shots = _enforce_min_shot(shots, int(round(cfg.min_shot * fps)))
    shots = _merge_same(shots)
    if coverage:
        shots = _fix_unavailable(shots, avail, max(2, int(round(0.1 * fps))))
    return _to_shots(shots, fps, duration)


# ----------------------------------------------------------------------------- helpers

# internal shot: [start_frame, end_frame, cam_idx, {reason: frames}]
_IShot = List  # type alias


def _rle(cams: np.ndarray, why: np.ndarray) -> List[_IShot]:
    out: List[_IShot] = []
    n = len(cams)
    i = 0
    while i < n:
        j = i
        reasons: Dict[int, int] = {}
        while j < n and cams[j] == cams[i]:
            reasons[int(why[j])] = reasons.get(int(why[j]), 0) + 1
            j += 1
        out.append([i, j, int(cams[i]), reasons])
        i = j
    return out


def _add_reasons(a: Dict[int, int], b: Dict[int, int]) -> Dict[int, int]:
    r = dict(a)
    for k, v in b.items():
        r[k] = r.get(k, 0) + v
    return r


def _merge_same(shots: List[_IShot]) -> List[_IShot]:
    out: List[_IShot] = []
    for s in shots:
        if out and out[-1][2] == s[2] and out[-1][1] == s[0]:
            out[-1][1] = s[1]
            out[-1][3] = _add_reasons(out[-1][3], s[3])
        else:
            out.append([s[0], s[1], s[2], dict(s[3])])
    return out


def _enforce_min_shot(shots: List[_IShot], min_frames: int) -> List[_IShot]:
    """Repeatedly fix the shortest too-short shot.

    * short WIDE shot: grow it by taking frames from the longer neighbour if that
      neighbour stays >= min; otherwise absorb into neighbour (prefer keeping WIDE:
      a close-up neighbour that would become too short is converted to WIDE).
    * short close-up next to a WIDE shot: absorbed by the WIDE shot.
    * short close-up between close-ups: absorbed by the longer neighbour (or both
      neighbours merge if they show the same camera).
    """
    wide = _CAM_IDX[WIDE]
    shots = _merge_same(shots)
    guard = 0
    while len(shots) > 1 and guard < 100000:
        guard += 1
        lens = [s[1] - s[0] for s in shots]
        k = int(np.argmin(lens))
        if lens[k] >= min_frames:
            break
        s = shots[k]
        left = shots[k - 1] if k > 0 else None
        right = shots[k + 1] if k + 1 < len(shots) else None
        need = min_frames - lens[k]
        if s[2] == wide:
            # try to grow into neighbours
            cands = [(nb, side) for nb, side in ((left, "L"), (right, "R")) if nb is not None]
            cands.sort(key=lambda x: -(x[0][1] - x[0][0]))
            grown = False
            for nb, side in cands:
                if (nb[1] - nb[0]) - need >= min_frames:
                    if side == "L":
                        nb[1] -= need
                        s[0] -= need
                    else:
                        nb[0] += need
                        s[1] += need
                    s[3] = _add_reasons(s[3], {_R["merged_short"]: need})
                    grown = True
                    break
            if grown:
                continue
            # absorb the shorter neighbour into WIDE
            nb, side = cands[-1] if len(cands) > 1 else cands[0]
            nb[2] = wide
            nb[3] = _add_reasons(nb[3], {_R["merged_short"]: nb[1] - nb[0]})
            shots = _merge_same(shots)
            continue
        # close-up that is too short
        if left is not None and right is not None and left[2] == right[2]:
            target = left
        elif left is not None and left[2] == wide:
            target = left
        elif right is not None and right[2] == wide:
            target = right
        elif left is None:
            target = right
        elif right is None:
            target = left
        else:
            target = left if (left[1] - left[0]) >= (right[1] - right[0]) else right
        assert target is not None
        s[2] = target[2]
        s[3] = _add_reasons(s[3], {_R["merged_short"]: lens[k]})
        shots = _merge_same(shots)
    return shots


def _fix_unavailable(shots: List[_IShot], avail: Dict[str, np.ndarray], tol: int) -> List[_IShot]:
    """Min-shot merging prefers WIDE, so it can hand frames back to the wide camera where
    it has no footage (e.g. a short fallback shot absorbed into the intro WIDE before the
    wide camera started -> black frames). Switch such a WIDE shot to a camera that covers
    all of its frames, preferring a neighbour's camera so it merges instead of adding a
    cut. Up to ``tol`` missing frames are ignored (rounding slivers at the very end; the
    renderer clones the last frame). Close-ups are left alone (a few padded frames beat
    showing the wrong person). Shots only grow, so min_shot still holds."""
    for k, s in enumerate(shots):
        a, b = s[0], s[1]
        if s[2] != _CAM_IDX[WIDE] or int((~avail[WIDE][a:b]).sum()) <= tol:
            continue
        nbs = [shots[j][2] for j in (k - 1, k + 1) if 0 <= j < len(shots)]
        for alt in nbs + [_CAM_IDX[c] for c in (CAM_A, CAM_B)]:
            if alt != s[2] and avail[_CAMS[alt]][a:b].all():
                s[2] = alt
                s[3] = _add_reasons(s[3], {_R["camera_unavailable"]: b - a})
                break
    return _merge_same(shots)


def _reason_label(cam_idx: int, reasons: Dict[int, int]) -> str:
    names = {_REASONS[k]: v for k, v in reasons.items()}
    if _CAMS[cam_idx] == WIDE:
        present = [r for r in _WIDE_PRIORITY if names.get(r)]
        extra = [r for r in ("camera_unavailable",) if names.get(r)]
        lab = "+".join(present[:2] + extra) or max(names, key=names.get)
    else:
        lab = max((r for r in names if r != "merged_short"), key=lambda r: names[r], default="merged_short")
    if names.get("merged_short"):
        lab += "+merged_short"
    return lab


def _to_shots(shots: List[_IShot], fps: float, duration: float) -> List[Shot]:
    out: List[Shot] = []
    for i, (a, b, c, reasons) in enumerate(shots):
        start = a / fps
        end = duration if i == len(shots) - 1 else b / fps
        out.append(Shot(round(start, 6), round(end, 6), _CAMS[c], _reason_label(c, reasons), a, b))
    return out


def validate_shots(shots: Sequence[Shot], duration: float, fps: float) -> None:
    """Raise AssertionError if shots violate the timeline invariants."""
    assert shots, "no shots"
    assert shots[0].start == 0 and shots[0].start_frame == 0, "timeline must start at 0"
    for a, b in zip(shots, shots[1:]):
        assert a.end_frame == b.start_frame, f"gap/overlap between {a} and {b}"
        assert abs(a.end - b.start) < 1e-6, f"time gap/overlap between {a} and {b}"
        assert a.camera != b.camera, f"adjacent shots share camera {a.camera}"
    for s in shots:
        assert s.end_frame > s.start_frame, f"empty shot {s}"
        assert abs(s.start - s.start_frame / fps) < 1e-6, f"start not on frame grid {s}"
    assert abs(shots[-1].end - duration) < 1e-6, "timeline must end at duration"
    assert shots[-1].end_frame == int(math.ceil(duration * fps - 1e-6))
