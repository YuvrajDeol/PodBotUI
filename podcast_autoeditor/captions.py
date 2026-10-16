"""Caption generation: chunking, layout, ASS/SRT/JSON export.

ASS colours are ``&HAABBGGRR`` (alpha, blue, green, red; alpha 00 = opaque), i.e.
the byte order is *reversed* relative to HTML ``#RRGGBB``.
"""

from __future__ import annotations

import json
import math
import re
from dataclasses import asdict, dataclass, field
from typing import Any, Dict, List, Optional, Sequence, Tuple

from .config import CaptionConfig
from .datatypes import Turn, Word

_PUNCT_BREAK = (".", "?", "!", "…")
_SOFT_BREAK = (",", ";", ":", "—", "-")


# --------------------------------------------------------------------------- colours

def parse_color(color: str) -> Tuple[int, int, int]:
    """Parse '#RRGGBB', 'RRGGBB', '#RGB', 'rgb(r,g,b)' or 'rgba(r,g,b,a)' -> (r,g,b)."""
    if color is None:
        raise ValueError("colour is None")
    c = str(color).strip()
    m = re.match(r"^rgba?\(\s*([\d.]+)\s*,\s*([\d.]+)\s*,\s*([\d.]+)", c, re.I)
    if m:
        r, g, b = (int(round(float(v))) for v in m.groups())
        if not all(0 <= v <= 255 for v in (r, g, b)):
            raise ValueError(f"colour component out of range: {color!r}")
        return r, g, b
    c = c.lstrip("#")
    if c.lower().startswith("0x"):
        c = c[2:]
    if len(c) == 3:
        c = "".join(ch * 2 for ch in c)
    if len(c) == 8:  # RRGGBBAA -> ignore alpha
        c = c[:6]
    if len(c) != 6 or not re.fullmatch(r"[0-9a-fA-F]{6}", c):
        raise ValueError(f"invalid hex colour: {color!r}")
    return int(c[0:2], 16), int(c[2:4], 16), int(c[4:6], 16)


def hex_to_ass(color: str, alpha: int = 0) -> str:
    """'#2D9CFF' -> '&H00FF9C2D' (ASS &HAABBGGRR; alpha 0 = opaque, 255 = transparent)."""
    r, g, b = parse_color(color)
    if not 0 <= alpha <= 255:
        raise ValueError("alpha must be 0..255")
    return f"&H{alpha:02X}{b:02X}{g:02X}{r:02X}"


def ass_to_rgb(ass: str) -> Tuple[int, int, int, int]:
    """Inverse of :func:`hex_to_ass` -> (r, g, b, alpha)."""
    m = re.fullmatch(r"&H([0-9A-Fa-f]{2})?([0-9A-Fa-f]{2})([0-9A-Fa-f]{2})([0-9A-Fa-f]{2})&?", ass.strip())
    if not m:
        raise ValueError(f"bad ASS colour {ass!r}")
    a = int(m.group(1) or "00", 16)
    b, g, r = (int(m.group(i), 16) for i in (2, 3, 4))
    return r, g, b, a


# --------------------------------------------------------------------------- chunking

@dataclass
class CaptionLine:
    speaker: str
    start: float
    end: float
    text: str
    words: List[Dict[str, Any]] = field(default_factory=list)
    lane: int = 0          # 0 = bottom row, 1 = row above, ...
    margin_v: int = 0      # pixels from bottom (filled by layout)

    def to_dict(self) -> Dict[str, Any]:
        return asdict(self)


def chunk_words(words: Sequence[Word], cfg: CaptionConfig) -> List[CaptionLine]:
    """Group words into short caption lines; breaks on speaker change, sentence
    punctuation, pauses and the max words / chars / duration limits."""
    lines: List[CaptionLine] = []
    cur: List[Word] = []

    def flush() -> None:
        if cur:
            text = " ".join(w.text for w in cur)
            lines.append(CaptionLine(cur[0].speaker or "A", cur[0].start, cur[-1].end, text,
                                     [w.to_dict() for w in cur]))
            cur.clear()

    for w in sorted(words, key=lambda w: w.start):
        if not w.text:
            continue
        if cur:
            text_len = len(" ".join(x.text for x in cur)) + 1 + len(w.text)
            if (w.speaker != cur[-1].speaker
                    or len(cur) >= cfg.max_words
                    or text_len > cfg.max_chars
                    or w.end - cur[0].start > cfg.max_dur
                    or w.start - cur[-1].end > 0.7):
                flush()
        cur.append(w)
        t = w.text
        if t.endswith(_PUNCT_BREAK) or (t.endswith(_SOFT_BREAK) and len(cur) >= 3):
            flush()
    flush()
    _fix_timing(lines, cfg)
    return lines


def _fix_timing(lines: List[CaptionLine], cfg: CaptionConfig) -> None:
    """Add linger + minimum duration without overlapping the same speaker's next line."""
    by_spk: Dict[str, List[CaptionLine]] = {}
    for ln in lines:
        by_spk.setdefault(ln.speaker, []).append(ln)
    for spk_lines in by_spk.values():
        spk_lines.sort(key=lambda l: l.start)
        for i, ln in enumerate(spk_lines):
            nxt = spk_lines[i + 1].start if i + 1 < len(spk_lines) else math.inf
            end = max(ln.end + cfg.linger, ln.start + cfg.min_dur)
            ln.end = round(max(ln.end, min(end, nxt - 0.02)), 3)
            ln.start = round(ln.start, 3)


# --------------------------------------------------------------------------- layout

@dataclass
class CaptionStyle:
    width: int
    height: int
    font_size: int
    outline: float
    shadow: float
    margin_v: int
    margin_lr: int
    lane_height: int
    rows_per_line: int


def make_style(cfg: CaptionConfig, width: int, height: int,
               max_chars_seen: Optional[int] = None) -> CaptionStyle:
    fs = max(12, int(round(height * cfg.font_size_pct / 100.0)))
    scale = height / 1080.0
    margin_lr = int(round(width * 0.04))
    # rough bold-sans width estimate: 0.6 em per character
    chars = max_chars_seen or cfg.max_chars
    est_w = chars * 0.6 * fs
    rows = max(1, int(math.ceil(est_w / max(1.0, width - 2 * margin_lr))))
    line_h = int(round(fs * 1.2 + 2 * cfg.outline_px * scale))
    return CaptionStyle(
        width=width, height=height, font_size=fs,
        outline=round(cfg.outline_px * scale, 2), shadow=round(cfg.shadow_px * scale, 2),
        margin_v=int(round(height * cfg.margin_v_pct / 100.0)), margin_lr=margin_lr,
        lane_height=line_h * rows + int(round(0.15 * fs)), rows_per_line=rows,
    )


def layout_lines(lines: List[CaptionLine], style: CaptionStyle) -> List[CaptionLine]:
    """Greedy lane assignment: a line takes the lowest row not occupied by a line that
    is still on screen, so simultaneous A/B captions stack instead of colliding.
    Lines never move while visible."""
    active: List[CaptionLine] = []
    for ln in sorted(lines, key=lambda l: (l.start, l.speaker)):
        active = [a for a in active if a.end > ln.start + 1e-6]
        used = {a.lane for a in active}
        lane = 0
        while lane in used:
            lane += 1
        ln.lane = lane
        ln.margin_v = style.margin_v + lane * style.lane_height
        active.append(ln)
    return lines


def build_captions(words: Sequence[Word], cfg: CaptionConfig, width: int, height: int
                   ) -> Tuple[List[CaptionLine], CaptionStyle]:
    lines = chunk_words(words, cfg)
    style = make_style(cfg, width, height, max((len(l.text) for l in lines), default=None))
    layout_lines(lines, style)
    return lines, style


# --------------------------------------------------------------------------- writers

def _ass_time(t: float) -> str:
    cs = int(round(max(0.0, t) * 100))
    h, cs = divmod(cs, 360000)
    m, cs = divmod(cs, 6000)
    s, cs = divmod(cs, 100)
    return f"{h:d}:{m:02d}:{s:02d}.{cs:02d}"


def _srt_time(t: float) -> str:
    ms = int(round(max(0.0, t) * 1000))
    h, ms = divmod(ms, 3600000)
    m, ms = divmod(ms, 60000)
    s, ms = divmod(ms, 1000)
    return f"{h:02d}:{m:02d}:{s:02d},{ms:03d}"


def _ass_escape(text: str) -> str:
    return text.replace("\\", "/").replace("{", "(").replace("}", ")").replace("\n", " ")


def write_ass(path: str, lines: Sequence[CaptionLine], style: CaptionStyle, cfg: CaptionConfig) -> str:
    col = {"A": hex_to_ass(cfg.color_a), "B": hex_to_ass(cfg.color_b)}
    bold = -1 if cfg.bold else 0
    hdr = [
        "[Script Info]",
        "; generated by podcast_autoeditor",
        "ScriptType: v4.00+",
        f"PlayResX: {style.width}",
        f"PlayResY: {style.height}",
        "WrapStyle: 0",
        "ScaledBorderAndShadow: yes",
        "YCbCr Matrix: TV.709",
        "",
        "[V4+ Styles]",
        "Format: Name, Fontname, Fontsize, PrimaryColour, SecondaryColour, OutlineColour, "
        "BackColour, Bold, Italic, Underline, StrikeOut, ScaleX, ScaleY, Spacing, Angle, "
        "BorderStyle, Outline, Shadow, Alignment, MarginL, MarginR, MarginV, Encoding",
    ]
    for spk in ("A", "B"):
        hdr.append(
            f"Style: Speaker{spk},{cfg.font},{style.font_size},{col[spk]},&H000000FF,&H00000000,"
            f"&H96000000,{bold},0,0,0,100,100,0,0,1,{style.outline:g},{style.shadow:g},2,"
            f"{style.margin_lr},{style.margin_lr},{style.margin_v},1"
        )
    hdr += ["", "[Events]",
            "Format: Layer, Start, End, Style, Name, MarginL, MarginR, MarginV, Effect, Text"]
    ev = []
    for ln in sorted(lines, key=lambda l: l.start):
        spk = ln.speaker if ln.speaker in ("A", "B") else "A"
        ev.append(
            f"Dialogue: {ln.lane},{_ass_time(ln.start)},{_ass_time(ln.end)},Speaker{spk},"
            f"Speaker {spk},0,0,{ln.margin_v},,{_ass_escape(ln.text)}"
        )
    with open(path, "w", encoding="utf-8") as f:
        f.write("\n".join(hdr + ev) + "\n")
    return path


def write_srt(path: str, lines: Sequence[CaptionLine]) -> str:
    out = []
    for i, ln in enumerate(sorted(lines, key=lambda l: l.start), 1):
        out.append(f"{i}\n{_srt_time(ln.start)} --> {_srt_time(ln.end)}\nSpeaker {ln.speaker}: {ln.text}\n")
    with open(path, "w", encoding="utf-8") as f:
        f.write("\n".join(out))
    return path


def write_transcript_json(path: str, words: Sequence[Word], turns: Sequence[Turn],
                          lines: Sequence[CaptionLine], cfg: CaptionConfig,
                          extra: Optional[Dict[str, Any]] = None) -> str:
    data = {
        "speakers": {"A": {"camera": "CAM_A", "caption_color": cfg.color_a},
                     "B": {"camera": "CAM_B", "caption_color": cfg.color_b}},
        "turns": [t.to_dict() for t in turns],
        "captions": [l.to_dict() for l in lines],
        "words": [w.to_dict() for w in words],
    }
    if extra:
        data.update(extra)
    with open(path, "w", encoding="utf-8") as f:
        json.dump(data, f, indent=1, ensure_ascii=False)
    return path


def transcript_text(turns: Sequence[Turn]) -> str:
    out = []
    for t in turns:
        m, s = divmod(t.start, 60)
        out.append(f"[{int(m):02d}:{s:05.2f}] Speaker {t.speaker}: {t.text}")
    return "\n".join(out)
