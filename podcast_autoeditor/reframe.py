"""Single-clip mode: virtual close-ups cropped out of one wide shot.

1. :func:`scan_faces` samples the wide clip (streamed, with a frame cap and a wall-clock
   budget), detects every face and measures mouth motion per face.
2. :func:`cluster_faces` / :func:`select_people` turn the detections into at most two
   person tracks, clustered by horizontal position (people sit still in a podcast; a
   position cluster survives missed detections and mic occlusion far better than
   frame-to-frame tracking).  The left track is Speaker A by default.
3. :func:`compute_crop` frames each person like a real close-up (head and shoulders, eyes
   on the upper third, output aspect ratio, even pixels, inside the frame, the other
   person kept out).  Crops are locked off; :func:`shot_crops` re-frames only at shot
   boundaries when a person has clearly moved.

All geometry is in *display* pixels of the source (after rotation metadata), which is what
ffmpeg's ``crop`` filter sees.
"""

from __future__ import annotations

import logging
import os
import subprocess
import time
from dataclasses import asdict, dataclass, field
from typing import Any, Dict, List, Optional, Sequence, Tuple

import numpy as np

from . import CAM_A, CAM_B
from .config import ReframeConfig
from .datatypes import Shot

log = logging.getLogger(__name__)

SIDES = ("left", "right")


# --------------------------------------------------------------------------- data

@dataclass
class CropRect:
    x: int
    y: int
    w: int
    h: int

    def as_list(self) -> List[int]:
        return [self.x, self.y, self.w, self.h]

    def filter(self) -> str:
        return f"crop={self.w}:{self.h}:{self.x}:{self.y}"

    def contains(self, x0: float, y0: float, x1: float, y1: float) -> bool:
        return x0 >= self.x and y0 >= self.y and x1 <= self.x + self.w and y1 <= self.y + self.h

    def overlaps_x(self, x0: float, x1: float) -> bool:
        return x0 < self.x + self.w and x1 > self.x


@dataclass
class PersonTrack:
    """One person: robust (median) face geometry + the position samples over time."""
    side: str                                   # left | right (of the frame)
    cx: float
    cy: float
    w: float
    h: float
    eye_y: float
    hits: int = 0
    persistence: float = 0.0
    times: List[float] = field(default_factory=list)    # clip time of each detection
    xs: List[float] = field(default_factory=list)
    ys: List[float] = field(default_factory=list)
    hs: List[float] = field(default_factory=list)
    eyes: List[float] = field(default_factory=list)
    motion_t: List[float] = field(default_factory=list)  # clip times of mouth-motion samples
    motion: List[float] = field(default_factory=list)

    def box(self) -> Tuple[float, float, float, float]:
        return (self.cx - self.w / 2, self.cy - self.h / 2, self.cx + self.w / 2, self.cy + self.h / 2)

    def extent_box(self) -> Tuple[float, float, float, float]:
        """Face box widened to everywhere the face has been (5th-95th percentile of its
        positions): what a locked-off crop of the OTHER person has to stay clear of."""
        if len(self.xs) < 5:
            return self.box()
        x0, x1 = np.percentile(self.xs, [5, 95])
        y0, y1 = np.percentile(self.ys, [5, 95])
        return (float(x0) - self.w / 2, float(y0) - self.h / 2, float(x1) + self.w / 2, float(y1) + self.h / 2)

    def box_between(self, t0: float, t1: float) -> Optional[Tuple[float, float, float, float]]:
        """Median face box over clip times [t0, t1), or None with fewer than 3 samples."""
        t = np.asarray(self.times)
        m = (t >= t0) & (t < t1)
        if m.sum() < 3:
            return None
        cx, cy = float(np.median(np.asarray(self.xs)[m])), float(np.median(np.asarray(self.ys)[m]))
        return (cx - self.w / 2, cy - self.h / 2, cx + self.w / 2, cy + self.h / 2)

    def to_dict(self) -> Dict[str, Any]:
        return asdict(self)

    @staticmethod
    def from_dict(d: Dict[str, Any]) -> "PersonTrack":
        return PersonTrack(**d)


def parse_crop(spec: Optional[str]) -> Optional[CropRect]:
    """'x,y,w,h' -> CropRect (ValueError on bad input)."""
    if not spec:
        return None
    try:
        x, y, w, h = (int(round(float(v))) for v in str(spec).replace("x", ",").replace(":", ",").split(","))
    except ValueError as e:
        raise ValueError(f"crop must be 'x,y,w,h' in source pixels, got {spec!r}") from e
    if w <= 0 or h <= 0 or x < 0 or y < 0:
        raise ValueError(f"crop must have x,y >= 0 and w,h > 0, got {spec!r}")
    return CropRect(x, y, w, h)


# --------------------------------------------------------------------------- tracks

def cluster_faces(samples: Sequence[Dict[str, Any]], gap: float = 0.8) -> List[List[Tuple[int, int]]]:
    """Group detections into people by horizontal position (pure; unit tested).

    ``samples``: [{"t": clip_time, "faces": [{"cx","cy","w","h",...}]}].  Detections are
    sorted by x and split wherever two neighbours are more than ``gap`` x (median face
    width) apart.  Within a group at most one face per sample is kept (the largest), so a
    smaller face *behind* a person never joins that person's track.
    Returns groups of (sample index, face index), left to right."""
    dets = [(f["cx"], i, j) for i, s in enumerate(samples) for j, f in enumerate(s["faces"])]
    if not dets:
        return []
    dets.sort()
    med_w = float(np.median([samples[i]["faces"][j]["w"] for _, i, j in dets]))
    thr = max(1.0, gap * med_w)
    groups: List[List[Tuple[int, int]]] = [[]]
    prev = dets[0][0]
    for cx, i, j in dets:
        if cx - prev > thr:
            groups.append([])
        groups[-1].append((i, j))
        prev = cx
    out = []
    for g in groups:
        best: Dict[int, int] = {}
        for i, j in g:
            f = samples[i]["faces"][j]
            if i not in best or f["w"] * f["h"] > samples[i]["faces"][best[i]]["w"] * samples[i]["faces"][best[i]]["h"]:
                best[i] = j
        out.append(sorted(best.items()))
    return _merge_moved(out, samples, med_w)


def _merge_moved(groups: List[List[Tuple[int, int]]], samples: Sequence[Dict[str, Any]], med_w: float,
                 max_dist: float = 3.0) -> List[List[Tuple[int, int]]]:
    """A person who shifts sideways by more than the clustering gap shows up as two
    neighbouring clusters.  Two real people are seen in the SAME frames; one person who
    moved never is.  Merge neighbours that (almost) never co-occur, are within ``max_dist``
    face widths of each other and have a similar face size."""
    def stats(g):
        fs = [samples[i]["faces"][j] for i, j in g]
        return float(np.median([f["cx"] for f in fs])), float(np.median([f["h"] for f in fs]))

    changed = True
    while changed and len(groups) > 1:
        changed = False
        for k in range(len(groups) - 1):
            a, b = groups[k], groups[k + 1]
            ia, ib = {i for i, _ in a}, {i for i, _ in b}
            both = len(ia & ib)
            (xa, ha), (xb, hb) = stats(a), stats(b)
            if (both <= 0.02 * min(len(ia), len(ib)) and abs(xb - xa) <= max_dist * med_w
                    and 0.7 <= ha / max(hb, 1e-6) <= 1.43):
                merged = dict(a)
                merged.update({i: j for i, j in b if i not in merged})
                groups = groups[:k] + [sorted(merged.items())] + groups[k + 2:]
                changed = True
                break
    return groups


def build_tracks(samples: Sequence[Dict[str, Any]], frame_w: int, gap: float = 0.8) -> List[PersonTrack]:
    """All position clusters as tracks (left to right), with mouth motion where the
    samples carry a ``mouth`` patch (see :func:`scan_faces`)."""
    n = max(1, len(samples))
    tracks: List[PersonTrack] = []
    for g in cluster_faces(samples, gap):
        fs = [samples[i]["faces"][j] for i, j in g]
        med = lambda k: float(np.median([f[k] for f in fs]))  # noqa: E731
        tr = PersonTrack(
            side="left" if med("cx") < frame_w / 2 else "right",
            cx=med("cx"), cy=med("cy"), w=med("w"), h=med("h"), eye_y=med("eye_y"),
            hits=len(g), persistence=len(g) / n,
            times=[round(float(samples[i]["t"]), 3) for i, _ in g],
            xs=[round(float(f["cx"]), 1) for f in fs], ys=[round(float(f["cy"]), 1) for f in fs],
            hs=[round(float(f["h"]), 1) for f in fs], eyes=[round(float(f["eye_y"]), 1) for f in fs])
        # mouth motion between consecutive *dense* samples of this person
        prev_i, prev_patch = None, None
        for (i, j), f in zip(g, fs):
            patch = f.get("mouth_patch")
            if patch is not None and samples[i].get("dense", True):
                if prev_patch is not None and prev_i is not None and i - prev_i == 1:
                    a, b = _norm_patch(prev_patch), _norm_patch(patch)
                    tr.motion_t.append(round(float(samples[i]["t"]), 3))
                    tr.motion.append(round(float(np.mean(np.abs(a - b))), 5))
                prev_i, prev_patch = i, patch
            else:
                prev_i, prev_patch = None, None
        tracks.append(tr)
    tracks.sort(key=lambda t: t.cx)
    return tracks


def _norm_patch(p: np.ndarray) -> np.ndarray:
    p = p.astype(np.float32)
    return (p - p.mean()) / (p.std() + 1e-3)


def select_people(tracks: Sequence[PersonTrack], frame_w: int, frame_h: int, cfg: ReframeConfig
                  ) -> Tuple[Dict[str, Optional[PersonTrack]], List[str]]:
    """Pick the (at most) two main people and name them A/B by side (pure; unit tested).

    Main = persistent (found in >= ``min_persistence`` of the scanned frames) and large;
    ranked by persistence x face height.  The left one is Speaker A unless
    ``cfg.a_side == "right"``.  With a single face its half of the frame decides its side.
    Returns ({"A": track|None, "B": track|None}, warnings)."""
    warns: List[str] = []
    if cfg.a_side not in SIDES:
        raise ValueError(f"a_side must be left or right, got {cfg.a_side!r}")
    ok = [t for t in tracks if t.persistence >= cfg.min_persistence and t.h >= 0.02 * frame_h]
    ok.sort(key=lambda t: -(t.persistence * t.h))
    if len(ok) > 2 and all(len(t.motion) >= 10 for t in ok):
        # A face whose mouth never moves (poster, framed photo, picture on a screen) is not a
        # speaker, however large and persistent: rank the live faces first.
        mm = [float(np.mean(t.motion)) for t in ok]
        top = max(mm)
        static = [m < 0.15 * top for m in mm] if top > 0 else [False] * len(ok)
        if any(static) and static.count(False) >= 2:     # only when two live faces remain
            for t, s in zip(ok, static):
                if s:
                    warns.append(f"the face at x={t.cx:.0f}, y={t.cy:.0f} looks static (its mouth never "
                                 "moves - a poster or photo?) and is not used as a speaker")
            ok = [t for t, s in zip(ok, static) if not s] + [t for t, s in zip(ok, static) if s]
    if len(ok) > 2:
        warns.append(f"{len(ok)} faces found in the wide shot; using the two most persistent/largest "
                     f"(at x={ok[0].cx:.0f} and x={ok[1].cx:.0f}) and ignoring the rest")
    main = sorted(ok[:2], key=lambda t: t.cx)
    people: Dict[str, Optional[PersonTrack]] = {"A": None, "B": None}
    a_left = cfg.a_side == "left"
    if len(main) == 2:
        main[0].side, main[1].side = "left", "right"
        people["A"], people["B"] = (main[0], main[1]) if a_left else (main[1], main[0])
    elif len(main) == 1:
        t = main[0]
        t.side = "left" if t.cx < frame_w / 2 else "right"
        letter = "A" if (t.side == "left") == a_left else "B"
        people[letter] = t
        other = "B" if letter == "A" else "A"
        warns.append(f"only ONE face found in the wide shot (on the {t.side}): Speaker {letter} gets a "
                     f"virtual close-up, Speaker {other}'s turns stay on the wide shot")
    else:
        warns.append("NO face found in the wide shot: the edit stays on the wide shot throughout "
                     "(captions are still colour-coded per speaker). Use --crop-a/--crop-b to set "
                     "the close-up crops by hand.")
    return people, warns


# --------------------------------------------------------------------------- crops

def _even(v: float) -> int:
    return int(round(v / 2.0)) * 2


def snap_size(cw: float, ch: float, frame_w: int, frame_h: int, out_w: int, out_h: int,
              min_h: float = 0.0) -> Tuple[int, int]:
    """Even integer (w, h) just below (cw, ch), inside the frame, with the output aspect
    ratio - exactly, when the ratio is a small fraction such as 16:9 or 9:16 (h is then a
    multiple of 2*q for aspect p/q).  Rounds DOWN (so a size that was checked to exclude
    the other person never grows), but not below ``min_h`` when the frame allows."""
    from fractions import Fraction

    fr = Fraction(int(out_w), int(out_h))
    p, q = fr.numerator, fr.denominator
    if 2 * q <= 64 and 2 * p <= frame_w and 2 * q <= frame_h:
        k_max = min(frame_h // (2 * q), frame_w // (2 * p))
        k = max(1, int((ch + 1e-6) // (2 * q)))
        if 2 * q * k < min_h:
            k += 1
        k = min(k, k_max)
        return 2 * p * k, 2 * q * k
    aspect = out_w / float(out_h)
    h = max(2, min(int(ch // 2) * 2, frame_h - frame_h % 2))
    if h < min_h:
        h = min(h + 2, frame_h - frame_h % 2)
    w = max(2, min(_even(h * aspect), frame_w - frame_w % 2))
    if w < _even(h * aspect):                            # width clipped by the frame
        h = max(2, min(_even(w / aspect), frame_h - frame_h % 2))
    return w, h


def compute_crop(
    face: Tuple[float, float, float, float, float],
    frame_w: int,
    frame_h: int,
    out_w: int,
    out_h: int,
    cfg: ReframeConfig,
    other: Optional[Tuple[float, float, float, float]] = None,
) -> Tuple[CropRect, Dict[str, Any]]:
    """Close-up crop for one person (pure; unit tested).

    ``face`` = (cx, cy, w, h, eye_y) of the robust face box; ``other`` = (x0, y0, x1, y1)
    of the other person's face box, which the crop tries to exclude.

    * height = ``crop_scale`` x face height (head and shoulders), width from the OUTPUT
      aspect ratio, so it works for 16:9 and 9:16 alike;
    * never upscaled more than ``max_zoom`` (crop height >= out_h / max_zoom), never larger
      than the frame;
    * eyes on ``eye_line`` (upper third), centred on the face, then shifted away from the
      other person - and tightened down to ``min_crop_scale`` if needed - while the own
      face keeps a margin inside the crop;
    * clamped to the frame, even x/y/w/h.
    Returns (crop, info) with info = {zoom, includes_other, tightened, ...}."""
    cx, cy, fw, fh, eye_y = face
    aspect = out_w / float(out_h)
    info: Dict[str, Any] = {}

    min_h = out_h / cfg.max_zoom if cfg.max_zoom and cfg.max_zoom > 0 else 0.0

    def size_for(scale: float) -> Tuple[float, float]:
        ch = max(scale * fh, min_h)                      # cap the upscaling
        ch = min(ch, float(frame_h))
        cw = ch * aspect
        if cw > frame_w:                                 # frame narrower than the crop wants
            cw = float(frame_w)
            ch = cw / aspect
        w_, h_ = snap_size(cw, ch, frame_w, frame_h, out_w, out_h, min(min_h, frame_h))
        return float(w_), float(h_)

    def place(cw: float, ch: float) -> Tuple[float, float, bool]:
        x = cx - cw / 2.0
        y = eye_y - cfg.eye_line * ch
        inc = False
        if other is not None:
            margin = 0.15 * fw
            keep = 0.65 * fw                             # own face stays this far inside the crop
            ox0, _, ox1, _ = other
            if ox0 >= cx:                                # other person is to the right
                limit = ox0 - margin                     # crop's right edge must stay left of this
                if x + cw > limit:
                    x = max(limit - cw, (cx + keep) - cw)
                inc = x + cw > ox0
            else:                                        # other person is to the left
                limit = ox1 + margin
                if x < limit:
                    x = min(limit, cx - keep)
                inc = x < ox1
        x = min(max(0.0, x), frame_w - cw)
        y = min(max(0.0, y), frame_h - ch)
        if other is not None:
            inc = (x < other[2]) and (x + cw > other[0])
        return x, y, inc

    scale = cfg.crop_scale
    cw, ch = size_for(scale)
    x, y, inc = place(cw, ch)
    tightened = False
    while inc and scale > cfg.min_crop_scale + 1e-6:     # zoom in a little to lose the other face
        scale = max(cfg.min_crop_scale, scale - 0.1)
        cw2, ch2 = size_for(scale)
        if ch2 >= ch - 1e-6:                             # max_zoom / frame limits stop us
            break
        cw, ch = cw2, ch2
        x, y, inc = place(cw, ch)
        tightened = True
    # sizes are already even / exact-aspect (size_for); round the position towards keeping
    # the other person out, stay inside the frame
    w, h = int(cw), int(ch)
    if other is not None and other[0] < cx:              # other on the left: round x up
        xi = int(np.ceil(x / 2.0)) * 2
    else:
        xi = int(np.floor(x / 2.0)) * 2
    xi = max(0, min(xi, frame_w - w))
    yi = max(0, min(_even(y), frame_h - h))
    xi -= xi % 2
    yi -= yi % 2
    crop = CropRect(xi, yi, w, h)
    info.update(zoom=round(out_h / float(h), 3), scale_used=round(h / fh, 2) if fh else None,
                tightened=tightened,
                includes_other=bool(other is not None and crop.overlaps_x(other[0], other[2])
                                    and other[1] < yi + h and other[3] > yi),
                face_inside=crop.contains(cx - fw / 2, cy - fh / 2, cx + fw / 2, cy + fh / 2))
    return crop, info


def fit_manual_crop(c: CropRect, frame_w: int, frame_h: int, out_w: int, out_h: int
                    ) -> Tuple[CropRect, bool]:
    """Clamp a user crop to the frame, make it even and give it the output aspect ratio
    (keeping its centre and height).  Returns (crop, changed)."""
    aspect = out_w / float(out_h)
    h = min(c.h, frame_h)
    w = h * aspect
    if w > frame_w:
        w = float(frame_w)
        h = w / aspect
    cx, cy = c.x + c.w / 2.0, c.y + c.h / 2.0
    wi, hi = snap_size(w, h, frame_w, frame_h, out_w, out_h)
    xi = max(0, min(_even(cx - wi / 2.0), frame_w - wi))
    yi = max(0, min(_even(cy - hi / 2.0), frame_h - hi))
    xi -= xi % 2
    yi -= yi % 2
    out = CropRect(xi, yi, wi, hi)
    return out, out != c


def base_crops(people: Dict[str, Optional[PersonTrack]], frame_w: int, frame_h: int, out_w: int,
               out_h: int, cfg: ReframeConfig) -> Tuple[Dict[str, Optional[CropRect]], Dict[str, Any], List[str]]:
    """Locked-off crop per speaker from the median face position (or the manual override)."""
    crops: Dict[str, Optional[CropRect]] = {"A": None, "B": None}
    info: Dict[str, Any] = {}
    warns: List[str] = []
    manual = {"A": parse_crop(cfg.crop_a), "B": parse_crop(cfg.crop_b)}
    for letter in ("A", "B"):
        other = people["B" if letter == "A" else "A"]
        if manual[letter] is not None:
            crop, changed = fit_manual_crop(manual[letter], frame_w, frame_h, out_w, out_h)
            if changed:
                warns.append(f"--crop-{letter.lower()} adjusted to {crop.as_list()} (inside the frame, even "
                             f"pixels, output aspect {out_w}:{out_h})")
            crops[letter] = crop
            info[letter] = {"manual": True, "zoom": round(out_h / float(crop.h), 3)}
        elif people[letter] is not None:
            p = people[letter]
            crop, ci = compute_crop((p.cx, p.cy, p.w, p.h, p.eye_y), frame_w, frame_h, out_w, out_h, cfg,
                                    other.extent_box() if other is not None else None)
            crops[letter] = crop
            info[letter] = ci
            if ci["includes_other"]:
                warns.append(f"Speaker {letter}'s close-up cannot fully exclude the other person "
                             "(they sit too close together for this framing)")
        if crops[letter] is not None and info[letter]["zoom"] > cfg.warn_zoom:
            warns.append(
                f"Speaker {letter}'s close-up is a {crops[letter].w}x{crops[letter].h} px crop upscaled "
                f"{info[letter]['zoom']:.1f}x to {out_w}x{out_h} - it will look soft. Use a higher-"
                "resolution wide shot (4K for 1080p output), a larger --crop-scale or a lower --max-zoom.")
    return crops, info, warns


def shot_crops(
    shots: Sequence[Shot],
    people: Dict[str, Optional[PersonTrack]],
    crops: Dict[str, Optional[CropRect]],
    frame_w: int,
    frame_h: int,
    out_w: int,
    out_h: int,
    cfg: ReframeConfig,
    offset: float = 0.0,
) -> Tuple[List[Optional[CropRect]], int]:
    """One crop per shot (pure; unit tested).  WIDE shots get None.

    Crops are locked off: a close-up uses the speaker's current crop for its WHOLE
    duration.  Only when, over a shot, the person's median face position has moved more
    than ``safe_frac`` of the crop size from the position the current crop was built for
    (or the face is no longer inside it, or the other person has moved into it) is a new
    crop computed from that shot's own medians - so re-framing happens at shot
    boundaries, never mid-shot.  Manual crops are never re-framed.  ``offset`` = clip t=0 on the
    timeline.  Returns (crops per shot, number of re-framed shots)."""
    manual = {"A": bool(cfg.crop_a), "B": bool(cfg.crop_b)}
    cur = dict(crops)
    out: List[Optional[CropRect]] = []
    reframed = 0
    # where each crop expects its person (face centre x, eye line y): the position it was built for
    base_centre = {k: (p.cx, p.eye_y) for k, p in people.items() if p is not None}
    exp_centre = dict(base_centre)
    arrays = {}
    for letter, p in people.items():
        if p is not None and p.times:
            arrays[letter] = (np.asarray(p.times), np.asarray(p.xs), np.asarray(p.ys),
                              np.asarray(p.hs), np.asarray(p.eyes))
    for sh in shots:
        letter = {CAM_A: "A", CAM_B: "B"}.get(sh.camera)
        if letter is None or crops.get(letter) is None:
            out.append(None)
            continue
        crop = cur[letter]
        p = people.get(letter)
        if cfg.per_shot and not manual[letter] and p is not None and letter in arrays:
            t, xs, ys, hs, eyes = arrays[letter]
            m = (t + offset >= sh.start) & (t + offset < sh.end)
            if m.sum() >= 3:
                cx, cy, fh, eye = (float(np.median(v[m])) for v in (xs, ys, hs, eyes))
                fw = p.w * fh / p.h if p.h else p.w
                other = people.get("B" if letter == "A" else "A")
                obox = None
                if other is not None:   # where the other person is DURING this shot
                    obox = other.box_between(sh.start - offset, sh.end - offset) or other.extent_box()

                def fits(c: CropRect) -> bool:
                    """Own face inside the safe area of crop c (it was built with the face
                    centred, or shifted to keep the other person out - so judge the face
                    by where c's own geometry puts it, with a tolerance)."""
                    return (c.contains(cx - fw / 2, cy - fh / 2, cx + fw / 2, cy + fh / 2)
                            and abs(eye - (c.y + cfg.eye_line * c.h)) <= cfg.safe_frac * c.h
                            and c.x + 0.12 * c.w <= cx <= c.x + 0.88 * c.w)

                def clear(c: CropRect) -> bool:
                    return obox is None or not (c.overlaps_x(obox[0], obox[2]) and obox[1] < c.y + c.h
                                                and obox[3] > c.y)

                exp = exp_centre.get(letter)
                moved = exp is not None and (abs(cx - exp[0]) > cfg.safe_frac * crop.w
                                             or abs(eye - exp[1]) > cfg.safe_frac * crop.h)
                if moved or not fits(crop) or not clear(crop):
                    # would the locked-off base crop do?  (person moved back) - else re-frame
                    base = crops[letter]
                    bexp = base_centre.get(letter)
                    back = bexp is not None and (abs(cx - bexp[0]) <= cfg.safe_frac * base.w
                                                 and abs(eye - bexp[1]) <= cfg.safe_frac * base.h)
                    if back and fits(base) and clear(base):
                        crop = base
                        exp_centre[letter] = bexp
                    else:
                        crop, _ = compute_crop((cx, cy, fw, fh, eye), frame_w, frame_h, out_w, out_h,
                                               cfg, obox)
                        exp_centre[letter] = (cx, eye)
                    if crop != cur[letter]:
                        reframed += 1
                    cur[letter] = crop
        out.append(crop)
    return out, reframed


# --------------------------------------------------------------------------- scanning

def wide_detector() -> Any:
    """Face detector for a frame with SEVERAL people.  The shared ``FaceDetector`` keeps
    only the 5 best candidate boxes before non-maximum suppression (fine for a one-person
    close-up); in a wide shot all five can belong to the same face and the second person
    is never reported - so raise top-k for this instance."""
    from .mapping import FaceDetector

    det = FaceDetector()
    if det.kind == "yunet" and hasattr(det.yunet, "setTopK"):
        det.yunet.setTopK(2000)
    return det


def scan_faces(
    ffmpeg: str,
    video: str,
    duration: float,
    cfg: ReframeConfig,
    dense_start: float = 0.0,
    detector: Any = None,
) -> Dict[str, Any]:
    """Detect faces over the clip and build position tracks with mouth motion.

    Dense pass (``dense_fps``, mouth patches kept) over ``dense_s`` seconds from
    ``dense_start`` (clip time) for the who-is-talking evidence; sparse pass
    (``sparse_fps``, positions only) over the rest so tracks cover the whole clip.
    Streamed; bounded by ``max_frames`` and ``timeout_s`` (a warning says so).
    Returns a JSON-serialisable dict: {frame_w, frame_h, detector, n_samples, tracks, ...}."""
    from .mapping import _iter_frames, _probe_display_size

    det = detector or wide_detector()
    cv2 = det.cv2
    frame_w, frame_h = _probe_display_size(ffmpeg, video)
    sc = cfg.analysis_width / max(frame_w, frame_h)
    aw, ah = max(2, int(round(frame_w * sc / 2)) * 2), max(2, int(round(frame_h * sc / 2)) * 2)
    fx, fy = frame_w / float(aw), frame_h / float(ah)
    deadline = time.monotonic() + cfg.timeout_s
    d0 = max(0.0, min(dense_start, max(0.0, duration - 1.0)))
    d1 = min(duration, d0 + cfg.dense_s)
    passes: List[Tuple[float, float, float, bool]] = [(d0, d1, cfg.dense_fps, True)]
    budget = cfg.max_frames - int((d1 - d0) * cfg.dense_fps)
    rest = (d0 - 0.0) + (duration - d1)
    if rest > 5.0 and budget > 20:
        sfps = min(cfg.sparse_fps, budget / rest)
        if d0 > 2.5:
            passes.insert(0, (0.0, d0, sfps, False))
        if duration - d1 > 2.5:
            passes.append((d1, duration, sfps, False))
    samples: List[Dict[str, Any]] = []
    stopped = ""
    for (t0, t1, fps, dense) in passes:
        left = cfg.max_frames - len(samples)
        if left <= 0 or time.monotonic() > deadline:
            stopped = stopped or "frame cap reached"
            break
        info: Dict[str, Any] = {}
        for k, fr in enumerate(_iter_frames(ffmpeg, video, t0, t1 - t0, fps, cfg.analysis_width,
                                            deadline=deadline, max_frames=left, info=info)):
            faces = []
            for f in det.faces(fr):
                rec = {"cx": (f["x"] + f["w"] / 2) * fx, "cy": (f["y"] + f["h"] / 2) * fy,
                       "w": f["w"] * fx, "h": f["h"] * fy, "eye_y": f["eye_y"] * fy,
                       "score": f["score"]}
                if dense:
                    x0, y0, x1, y1 = f["mouth"]
                    x0, y0 = int(max(0, x0)), int(max(0, y0))
                    x1, y1 = int(min(aw, x1)), int(min(ah, y1))
                    if x1 - x0 >= 4 and y1 - y0 >= 4:
                        g = cv2.cvtColor(fr[y0:y1, x0:x1], cv2.COLOR_BGR2GRAY)
                        rec["mouth_patch"] = cv2.resize(g, (48, 32), interpolation=cv2.INTER_AREA)
                faces.append(rec)
            samples.append({"t": t0 + k / fps, "dense": dense, "faces": faces})
        if info.get("timed_out"):
            stopped = "wall-clock budget exceeded"
        elif info.get("capped"):
            stopped = "frame cap reached"
    if stopped:
        log.warning("face scan of %s stopped after %d frames (%s) - tracks use only that part",
                    os.path.basename(video), len(samples), stopped)
    # sample order must be chronological for the consecutive-sample motion rule
    order = sorted(range(len(samples)), key=lambda i: samples[i]["t"])
    samples = [samples[i] for i in order]
    tracks = build_tracks(samples, frame_w, cfg.cluster_gap)
    n_with = sum(1 for s in samples if s["faces"])
    return {
        "frame_w": frame_w, "frame_h": frame_h, "detector": det.kind, "n_samples": len(samples),
        "n_dense": sum(1 for s in samples if s["dense"]), "frames_with_faces": n_with,
        "dense_window": [round(d0, 2), round(d1, 2)], "dense_fps": cfg.dense_fps,
        "stopped": stopped, "tracks": [t.to_dict() for t in tracks],
    }


def motion_on_grid(track: Optional[PersonTrack], times: np.ndarray, fps: float, offset: float
                   ) -> Optional[np.ndarray]:
    """A track's mouth motion on the reference-time grid ``times`` (NaN where unknown)."""
    if track is None or len(track.motion) < 10:
        return None
    grid = np.full(times.size, np.nan)
    idx = np.round((np.asarray(track.motion_t) + offset - times[0]) * fps).astype(int)
    ok = (idx >= 0) & (idx < times.size)
    grid[idx[ok]] = np.asarray(track.motion)[ok]
    return grid


def motion_series(people: Dict[str, Optional[PersonTrack]], scan: Dict[str, Any], offset: float
                  ) -> Optional[Dict[str, Any]]:
    """Mouth-motion series of face A / face B on one reference-time grid, in the form
    ``mapping.decide_mapping`` accepts as ``visual_inputs["series"]``."""
    fps = float(scan.get("dense_fps") or 5.0)
    d0, d1 = scan.get("dense_window", [0.0, 0.0])
    n = int(round((d1 - d0) * fps))
    if n < 10:
        return None
    times = d0 + offset + np.arange(n) / fps
    n_dense = max(1, int(scan.get("n_dense") or n))
    rates = {k.lower(): round(len(p.motion) / n_dense, 3) if p is not None else 0.0
             for k, p in people.items()}
    # same rule as mapping.visual_evidence: a face seen in < 30 % of the frames is not used
    ma = motion_on_grid(people.get("A"), times, fps, offset) if rates["a"] >= 0.3 else None
    mb = motion_on_grid(people.get("B"), times, fps, offset) if rates["b"] >= 0.3 else None
    return {"times": times, "motion_a": ma, "motion_b": mb,
            "info": {"detector": scan.get("detector"), "source": "face regions of the wide clip",
                     "face_rate": rates, "window": [round(float(times[0]), 2), round(float(times[-1]), 2)]}}


# --------------------------------------------------------------------------- preview

def write_preview(ffmpeg: str, video: str, t: float, crops: Dict[str, Optional[CropRect]],
                  people: Dict[str, Optional[PersonTrack]], path: str,
                  colors: Optional[Dict[str, Tuple[int, int, int]]] = None, max_width: int = 1280) -> str:
    """Write a PNG of the wide frame at clip time ``t`` with both crop rectangles drawn
    and labelled A / B (plus the tracked face boxes), to check the framing before rendering."""
    import cv2

    from .mapping import _probe_display_size

    w, h = _probe_display_size(ffmpeg, video)
    raw = subprocess.run([ffmpeg, "-hide_banner", "-nostdin", "-loglevel", "error", "-ss", f"{max(0.0, t):.3f}",
                          "-i", video, "-frames:v", "1", "-vf", f"scale={w}:{h}", "-f", "rawvideo",
                          "-pix_fmt", "bgr24", "-"],
                         capture_output=True, timeout=120).stdout
    if len(raw) < w * h * 3:
        raise RuntimeError(f"could not read a frame of {video} at {t:.2f}s for the crop preview")
    img = np.frombuffer(raw[: w * h * 3], np.uint8).reshape(h, w, 3).copy()
    colors = colors or {"A": (45, 156, 255), "B": (255, 255, 255)}
    th = max(2, int(round(h / 270)))
    for letter in ("A", "B"):
        c = crops.get(letter)
        bgr = tuple(int(v) for v in colors[letter][::-1])
        p = people.get(letter)
        if p is not None:
            x0, y0, x1, y1 = (int(v) for v in p.box())
            cv2.rectangle(img, (x0, y0), (x1, y1), bgr, max(1, th // 2))
        if c is None:
            continue
        cv2.rectangle(img, (c.x, c.y), (c.x + c.w - 1, c.y + c.h - 1), (0, 0, 0), th + 4)
        cv2.rectangle(img, (c.x, c.y), (c.x + c.w - 1, c.y + c.h - 1), bgr, th)
        fs = h / 540.0
        label = f"{letter}  {c.w}x{c.h}"
        (tw, tht), _ = cv2.getTextSize(label, cv2.FONT_HERSHEY_SIMPLEX, fs, th)
        lx, ly = c.x + th * 3, c.y + tht + th * 4
        cv2.rectangle(img, (lx - th * 2, ly - tht - th * 2), (lx + tw + th * 2, ly + th * 2), (0, 0, 0), -1)
        cv2.putText(img, label, (lx, ly), cv2.FONT_HERSHEY_SIMPLEX, fs, bgr, th, cv2.LINE_AA)
    if w > max_width:
        img = cv2.resize(img, (max_width, int(round(h * max_width / w))), interpolation=cv2.INTER_AREA)
    os.makedirs(os.path.dirname(os.path.abspath(path)) or ".", exist_ok=True)
    if not cv2.imwrite(path, img):
        raise RuntimeError(f"could not write crop preview {path}")
    return path
