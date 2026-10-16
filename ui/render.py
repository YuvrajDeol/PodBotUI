"""Pure HTML renderers for the PodBot UI.

No Gradio imports and no I/O: every function takes plain data and returns an HTML string, so
they are easy to test.  Every user- or pipeline-supplied string is passed through ``html.escape``.
Colours are only ever inserted after ``safe_hex`` has reduced them to ``#RRGGBB``.
"""

from __future__ import annotations

import html
import os
import re
from typing import Any, Dict, Iterable, List, Mapping, Optional, Sequence, Tuple

INK = "#1B1B1D"          # --pb-ink (light); used as the label colour on caption-colour fills
WHITE = "#FFFFFF"

# --------------------------------------------------------------------------- small helpers

_HEX = re.compile(r"^#[0-9A-Fa-f]{6}$")


def esc(s: Any) -> str:
    return html.escape(str(s), quote=True)


def safe_hex(c: Any, default: str) -> str:
    """``#RRGGBB`` or ``default`` - the only form of colour ever written into a style attribute."""
    c = str(c or "").strip()
    return c.upper() if _HEX.match(c) else default


def _lum(hex_color: str) -> float:
    def ch(v: int) -> float:
        x = v / 255.0
        return x / 12.92 if x <= 0.03928 else ((x + 0.055) / 1.055) ** 2.4

    r, g, b = (int(hex_color[i:i + 2], 16) for i in (1, 3, 5))
    return 0.2126 * ch(r) + 0.7152 * ch(g) + 0.0722 * ch(b)


def contrast(c1: str, c2: str) -> float:
    a, b = sorted((_lum(c1), _lum(c2)), reverse=True)
    return (a + 0.05) / (b + 0.05)


def ink_for(fill: str) -> str:
    """Label colour for text on a caption-colour fill: dark ink unless white reads better."""
    return INK if contrast(fill, INK) >= contrast(fill, WHITE) else WHITE


def clock(seconds: float) -> str:
    """0:07, 2:14, 1:02:03."""
    s = max(0, int(round(float(seconds))))
    h, rem = divmod(s, 3600)
    m, s = divmod(rem, 60)
    return f"{h}:{m:02d}:{s:02d}" if h else f"{m}:{s:02d}"


def clock2(seconds: float) -> str:
    """00:07, 02:14 (progress header)."""
    s = max(0, int(seconds))
    h, rem = divmod(s, 3600)
    m, s = divmod(rem, 60)
    return f"{h}:{m:02d}:{s:02d}" if h else f"{m:02d}:{s:02d}"


_TICK = ('<svg viewBox="0 0 16 16" width="16" height="16" aria-hidden="true" focusable="false">'
         '<path d="M3.5 8.5l3 3 6-7" fill="none" stroke="currentColor" stroke-width="2" '
         'stroke-linecap="square"/></svg>')


# --------------------------------------------------------------------------- page shell

def header_html() -> str:
    return (
        '<header class="pb-header">'
        '<span class="pb-wordmark">PodBot</span>'
        '<span class="pb-meta pb-tag">LOCAL · OFFLINE</span>'
        '</header>'
        '<div class="pb-hero">'
        '<h1 class="pb-h1"><span>Podcast in.</span> <span>Edited show out.</span></h1>'
        '<p class="pb-lede">Drop your recordings. PodBot syncs them, finds who&#x27;s talking and '
        'cuts between close-ups and the wide shot.</p>'
        '</div>')


def section_head(num: str, title: str, label: str) -> str:
    return (f'<div class="pb-section-head">'
            f'<div class="pb-rule" aria-hidden="true"><span class="pb-meta">'
            f'<span class="pb-step">{esc(num)}</span> / {esc(label)}</span></div>'
            f'<h2 class="pb-h2">{esc(title)}</h2></div>')


def group_label(text: str) -> str:
    return f'<p class="pb-meta pb-group">{esc(text)}</p>'


def helper_html(text: str, cls: str = "") -> str:
    return f'<p class="pb-help {cls}">{esc(text)}</p>' if text else ""


def chip(text: str, tone: str = "", title: str = "") -> str:
    t = f' title="{esc(title)}"' if title else ""
    cls = f"pb-chip pb-chip-{tone}" if tone else "pb-chip"
    return f'<span class="{cls}"{t}>{esc(text)}</span>'


def detect_html(n_files: int) -> str:
    if n_files == 1:
        return chip("DETECTED · 1 WIDE CLIP · CLOSE-UPS CROPPED AUTOMATICALLY")
    if n_files == 3:
        return chip("DETECTED · 3 CAMERAS")
    return ""


def inline_error(text: str) -> str:
    return (f'<p class="pb-inline-err"><span class="pb-err-dot" aria-hidden="true">!</span>'
            f'<span>{esc(text)}</span></p>')


def count_error_html(n_files: int) -> str:
    return inline_error(f"PodBot needs 1 wide clip or 3 clips (A, B, wide). You added {n_files}.")


ROLE_ERROR = "Each role needs a different file."
OFFSET_ERROR = "Enter a number of seconds, e.g. -1.25"


# --------------------------------------------------------------------------- roles

_WIDE_WORDS = ("wide", "master", "both", "full")


def _matches_side(stem: str, x: str) -> bool:
    return (any(k in stem for k in (f"cam_{x}", f"cam-{x}", f"cam{x}", f"speaker_{x}"))
            or stem.endswith((f"_{x}", f"-{x}")))


def guess_roles(paths: Sequence[str]) -> Tuple[Optional[str], Optional[str], Optional[str]]:
    """(Speaker A close-up, Speaker B close-up, wide) for three uploaded clips.

    Case-insensitive basename rules: wide = contains wide/master/both/full; A = contains
    cam_a/cam-a/cama/speaker_a or ends in _a/-a before the extension; B likewise.  Roles still
    unassigned take the remaining files in upload order (A, B, wide)."""
    stems = [os.path.splitext(os.path.basename(p))[0].lower() for p in paths]
    roles: Dict[str, Optional[int]] = {"wide": None, "a": None, "b": None}
    rules = {"wide": lambda s: any(w in s for w in _WIDE_WORDS),
             "a": lambda s: _matches_side(s, "a"),
             "b": lambda s: _matches_side(s, "b")}
    for role in ("wide", "a", "b"):
        for i, s in enumerate(stems):
            if i not in roles.values() and rules[role](s):
                roles[role] = i
                break
    rest = [i for i in range(len(paths)) if i not in roles.values()]
    for role in ("a", "b", "wide"):
        if roles[role] is None and rest:
            roles[role] = rest.pop(0)
    pick = lambda r: paths[roles[r]] if roles[r] is not None else None  # noqa: E731
    return pick("a"), pick("b"), pick("wide")


def role_choices(paths: Sequence[str]) -> List[Tuple[str, str]]:
    """Dropdown (label, value) pairs: label = basename (numbered if two share a name)."""
    names = [os.path.basename(p) for p in paths]
    dup = len(set(names)) < len(names)
    return [((f"{i + 1} · {n}" if dup else n), p) for i, (n, p) in enumerate(zip(names, paths))]


# --------------------------------------------------------------------------- caption preview

def caption_preview_html(color_a: str, color_b: str) -> str:
    a, b = safe_hex(color_a, "#2D9CFF"), safe_hex(color_b, "#FFFFFF")
    return (f'<div class="pb-cap" role="img" aria-label="Caption preview: Speaker A captions in {a}, '
            f'Speaker B captions in {b}">'
            f'<span class="pb-meta pb-cap-tag" aria-hidden="true">PREVIEW</span>'
            f'<div class="pb-cap-lines" aria-hidden="true">'
            f'<p class="pb-cap-line" style="color:{a}">A: So how did it start?</p>'
            f'<p class="pb-cap-line" style="color:{b}">B: Honestly, by accident.</p>'
            f'</div></div>')


# --------------------------------------------------------------------------- progress

STAGES = ("Prepare & sync", "Transcribe", "Identify speakers", "Plan cuts", "Render")
DONE = len(STAGES)
_PLAN, _RENDER = 3, 4
_STAGE_PREFIXES = (
    (("probing inputs", "extracting audio", "synchronising clips"), 0),
    (("transcribing",), 1),
    (("identifying speakers", "mapping speakers", "finding faces", "labelling words"), 2),
    (("planning shots",), _PLAN),
)


def stage_for(msg: str, current: int) -> int:
    """Stage index for a pipeline progress message (never moves backwards).

    Known prefixes map to their stage; anything after "planning shots" (rendering segments,
    final encode, segment counters) is Render; "done" finishes all; unknown text keeps the
    current stage."""
    m = (msg or "").strip().lower()
    if m == "done":
        return DONE
    for prefixes, idx in _STAGE_PREFIXES:
        if m.startswith(prefixes):
            return max(current, idx)
    if current >= _PLAN:
        return max(current, _RENDER)
    return current


def progress_html(stage: int, frac: float, msg: str, elapsed: float, plan_only: bool,
                  finished: bool = False) -> str:
    pct = int(round(100 * max(0.0, min(1.0, frac if not finished else 1.0))))
    head = ("FINISHED" if finished else "EDITING") + f" · {clock2(elapsed)}"
    items = []
    for i, name in enumerate(STAGES):
        skipped = plan_only and i == _RENDER
        if skipped:
            state, note = "skipped", "Skipped"
        elif finished or i < stage:
            state, note = "done", "done"
        elif i == stage:
            state, note = "active", "in progress"
        else:
            state, note = "pending", "pending"
        ico = _TICK if state == "done" else ""
        tag = '<span class="pb-stage-tag">Skipped</span>' if skipped else ""
        cur = ' aria-current="step"' if state == "active" else ""
        items.append(f'<li class="pb-stage is-{state}"{cur}><span class="pb-stage-ico">{ico}</span>'
                     f'<span class="pb-stage-name">{esc(name)}</span>{tag}'
                     f'<span class="pb-sr">{"" if skipped else ", " + note}</span></li>')
    active = STAGES[min(stage, DONE - 1)] if not finished else "All steps"
    bar = "" if finished else (
        f'<div class="pb-bar-row" aria-hidden="true">'
        f'<div class="pb-bar"><div class="pb-bar-fill" style="width:{pct}%"></div></div>'
        f'<span class="pb-pct">{pct}%</span></div>'
        f'<p class="pb-prog-msg" aria-hidden="true">{esc(msg or "starting")}</p>')
    status = f"{active} done" if finished else f"{active}, {pct} percent"
    return (f'<div class="pb-prog{" is-finished" if finished else ""}" aria-live="polite">'
            f'<div class="pb-prog-head"><span class="pb-meta">{head}</span></div>'
            f'<span class="pb-sr">{esc(status)}</span>'
            f'<ol class="pb-stages">{"".join(items)}</ol>{bar}</div>')


# --------------------------------------------------------------------------- error

def error_html(exc_type: str, message: str, is_pipeline_error: bool, log_lines: Sequence[str]) -> str:
    hint = ("Check the files and settings above, then try again." if is_pipeline_error
            else "See technical details below.")
    line = f"{exc_type}: {message}"[:300]
    tail = "\n".join(log_lines[-40:]) or "(no log output)"
    return (f'<div class="pb-err" role="alert">'
            f'<span class="pb-err-ico" aria-hidden="true">!</span>'
            f'<div class="pb-err-body">'
            f'<h3 class="pb-err-title">The edit didn&#x27;t finish</h3>'
            f'<p class="pb-err-msg">{esc(line)}</p>'
            f'<p class="pb-err-hint">{esc(hint)}</p>'
            f'<details class="pb-details"><summary>Technical details</summary>'
            f'<pre class="pb-pre">{esc(tail)}</pre></details>'
            f'</div></div>')


# --------------------------------------------------------------------------- result

def summary_chips_html(stats: Mapping[str, Any], mapping: Mapping[str, Any], elapsed: float) -> str:
    n = int(stats.get("n_shots", 0))
    n_spk = int(stats.get("num_speakers", 2))
    reasons = "; ".join(stats.get("speaker_count_reasons", []) or [])
    chips = [chip(f"DONE · {n} SHOT{'S' if n != 1 else ''}", "brand"),
             chip(f"{n_spk} SPEAKER{'S' if n_spk != 1 else ''}", "",
                  reasons if n_spk == 1 else ""),
             (chip("MAPPING CONFIDENT", "ok") if mapping.get("confident")
              else chip("CHECK SPEAKERS", "warn")),
             chip(f"ELAPSED {clock(int(elapsed))}")]  # floored like the progress header
    return f'<div class="pb-chips">{"".join(chips)}</div>'


CAMERA_INFO = {  # EDL camera -> (letter, long label, legend label, css class)
    "CAM_A": ("A", "Speaker A close-up", "A close-up", "a"),
    "CAM_B": ("B", "Speaker B close-up", "B close-up", "b"),
    "WIDE": ("W", "Wide shot", "Wide", "w"),
}


def _cam(camera: str) -> Tuple[str, str, str, str]:
    return CAMERA_INFO.get(str(camera), ("?", str(camera), str(camera), "w"))


def _cam_vars(color_a: str, color_b: str) -> str:
    a, b = safe_hex(color_a, "#2D9CFF"), safe_hex(color_b, "#FFFFFF")
    return f"--pb-cam-a:{a};--pb-cam-b:{b};--pb-cam-a-ink:{ink_for(a)};--pb-cam-b-ink:{ink_for(b)}"


def timeline_html(edl: Sequence[Mapping[str, Any]], camera_seconds: Mapping[str, float],
                  color_a: str, color_b: str) -> str:
    if not edl:
        return '<p class="pb-help">No shots were planned.</p>'
    total = max(float(r["end"]) for r in edl) or 1.0
    segs = []
    for r in edl:
        start, end = float(r["start"]), float(r["end"])
        letter, label, _, cls = _cam(r["camera"])
        aria = f"{label}, {clock(start)} to {clock(end)}"
        segs.append(f'<button type="button" class="pb-seg pb-seg-{cls}" style="flex-grow:{end - start:.3f}" '
                    f'data-t="{start:.3f}" aria-label="{esc(aria)}" title="{esc(aria)}">'
                    f'<span class="pb-seg-l" aria-hidden="true">{letter}</span></button>')
    legend = []
    for cam in ("CAM_A", "CAM_B", "WIDE"):
        if cam in camera_seconds:
            _, _, short, cls = _cam(cam)
            legend.append(f'<li><span class="pb-sw pb-seg-{cls}" aria-hidden="true"></span>'
                          f'{esc(short)} · {int(round(float(camera_seconds[cam])))} s</li>')
    return (f'<div class="pb-tl" style="{_cam_vars(color_a, color_b)}">'
            f'<div class="pb-tl-head"><span class="pb-meta">CUT TIMELINE · {len(edl)} '
            f'SHOT{"S" if len(edl) != 1 else ""}</span>'
            f'<span class="pb-meta">{clock(total)}</span></div>'
            f'<div class="pb-tl-scroll">'
            f'<div class="pb-tl-track" role="group" aria-label="Cut timeline: select a shot to play it" '
            f'data-total="{total:.3f}" style="min-width:max(100%,{2 * total:.0f}px)">'
            f'{"".join(segs)}<span class="pb-playhead" aria-hidden="true"></span></div></div>'
            f'<ul class="pb-legend">{"".join(legend)}</ul></div>')


_LINE = re.compile(r"^\[(\d+):(\d+(?:\.\d+)?)\] Speaker (\S+): (.*)$")


def transcript_html(text: str, color_a: str, color_b: str) -> str:
    lines = [ln for ln in (text or "").splitlines() if ln.strip()]
    if not lines:
        return '<p class="pb-help">No speech was found in the recording.</p>'
    parsed = [_LINE.match(ln) for ln in lines]
    if not all(parsed):
        return f'<pre class="pb-pre pb-transcript-raw">{esc(text)}</pre>'
    rows = []
    for m in parsed:
        t = int(m.group(1)) * 60 + float(m.group(2))
        spk = m.group(3)
        cls = {"A": "a", "B": "b"}.get(spk, "w")
        rows.append(f'<li><button type="button" class="pb-line" data-t="{t:.2f}">'
                    f'<span class="pb-line-t">{clock2(t)}</span>'
                    f'<span class="pb-spk pb-seg-{cls}"><span class="pb-sr">Speaker </span>{esc(spk)}</span>'
                    f'<span class="pb-line-x">{esc(m.group(4))}</span></button></li>')
    return (f'<ol class="pb-transcript" style="{_cam_vars(color_a, color_b)}" '
            f'aria-label="Transcript: select a line to play from there">{"".join(rows)}</ol>')


def warnings_html(warnings: Iterable[str], confident: bool) -> str:
    items = "".join(f"<li>{esc(w)}</li>" for w in list(dict.fromkeys(warnings))[:12])
    unsure = ("" if confident else
              '<p class="pb-warn-unsure">PodBot isn&#x27;t sure who is A and who is B.</p>')
    return (f'<div class="pb-warn-inner"><h3 class="pb-warn-title">'
            f'<span class="pb-warn-ico" aria-hidden="true">!</span>Things to check</h3>'
            f'{"<ul class=pb-warn-list>" + items + "</ul>" if items else ""}{unsure}</div>')


def plan_note_html() -> str:
    return ('<div class="pb-plan-note"><span class="pb-meta">PLAN ONLY</span>'
            '<p>Plan only: no video rendered.</p></div>')


def framing_caption_html(a_side: str) -> str:
    side = "right" if a_side == "right" else "left"
    return helper_html(f"Speaker A = person on the {side}. Close-up crops drawn on the wide frame.")


def _fmt_dict(d: Mapping[str, Any]) -> str:
    return " · ".join(f"{k} {v}" for k, v in d.items()) or "none"


def details_html(offsets: Mapping[str, Mapping[str, Any]], mapping: Mapping[str, Any],
                 diarization_backend: str, stats: Mapping[str, Any]) -> str:
    rows: List[Tuple[str, str]] = []
    rows.append(("Sync offsets", " · ".join(
        f"{k} {float(v.get('offset', 0)):+.3f} s ({v.get('method', '?')})" for k, v in offsets.items())
        or "none"))
    m = mapping.get("mapping", {}) or {}
    rows.append(("Speaker mapping",
                 f"{', '.join(f'{k} → {v}' for k, v in m.items())} via {mapping.get('method', '?')}"
                 f" ({'confident' if mapping.get('confident') else 'not confident'})"))
    n_spk = stats.get("num_speakers", 2)
    reasons = "; ".join(stats.get("speaker_count_reasons", []) or [])
    rows.append(("Speakers", f"{n_spk}" + (f" ({reasons})" if n_spk == 1 and reasons else "")))
    rows.append(("Diarization", str(diarization_backend)))
    rows.append(("Shots", f"{stats.get('n_shots', 0)} · seconds per camera "
                          f"{_fmt_dict(stats.get('camera_seconds', {}))}"))
    rows.append(("Events", _fmt_dict(stats.get("events", {}))))
    rf = stats.get("reframe")
    if rf:
        crops = rf.get("crops", {})
        other = "right" if rf.get("a_side") == "left" else "left"
        rows.append(("Framing", f"Speaker A = person on the {rf.get('a_side')}, Speaker B = person on "
                                f"the {other}; faces found {rf.get('faces_found')}"))
        rows.append(("Crops (x, y, w, h)", f"A {crops.get('A')} · B {crops.get('B')}"))
    body = "".join(f"<dt>{esc(k)}</dt><dd>{esc(v)}</dd>" for k, v in rows)
    return f'<dl class="pb-dl">{body}</dl>'
