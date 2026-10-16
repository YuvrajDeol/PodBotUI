"""Pillow-based caption burn-in (fallback when ffmpeg lacks libass).

Every distinct "caption state" (set of lines visible at once) is rendered to a
transparent PNG strip; an ffconcat slideshow of those PNGs (with exact durations)
is overlaid on the video in a single ``overlay`` filter.  Uses the same layout
(lanes / margins / colours) as the ASS file.
"""

from __future__ import annotations

import logging
import os
from typing import Dict, List, Optional, Sequence, Tuple

from .captions import CaptionLine, CaptionStyle, parse_color
from .config import CaptionConfig

log = logging.getLogger(__name__)

_FONT_CANDIDATES = [
    "/System/Library/Fonts/Supplemental/Arial Bold.ttf",
    "/Library/Fonts/Arial Bold.ttf",
    "/System/Library/Fonts/Supplemental/Verdana Bold.ttf",
    "/System/Library/Fonts/Helvetica.ttc",
    "/usr/share/fonts/truetype/dejavu/DejaVuSans-Bold.ttf",
    "/usr/share/fonts/dejavu/DejaVuSans-Bold.ttf",
    "C:/Windows/Fonts/arialbd.ttf",
]


def find_font(name: str, bold: bool = True) -> Optional[str]:
    cands = []
    if name and os.path.isfile(name):
        return name
    if name:
        for d in ("/System/Library/Fonts/Supplemental", "/Library/Fonts", "/System/Library/Fonts",
                  os.path.expanduser("~/Library/Fonts")):
            for suffix in ((" Bold.ttf", "-Bold.ttf", " Bold.ttc") if bold else ()) + (".ttf", ".ttc"):
                cands.append(os.path.join(d, name + suffix))
    cands += _FONT_CANDIDATES
    return next((c for c in cands if os.path.isfile(c)), None)


def _wrap(draw, text: str, font, max_w: int, stroke: int) -> List[str]:
    words = text.split()
    rows: List[str] = []
    cur = ""
    for w in words:
        cand = (cur + " " + w).strip()
        if draw.textlength(cand, font=font) + 2 * stroke <= max_w or not cur:
            cur = cand
        else:
            rows.append(cur)
            cur = w
    if cur:
        rows.append(cur)
    return rows


def render_overlay(
    lines: Sequence[CaptionLine],
    style: CaptionStyle,
    cfg: CaptionConfig,
    out_dir: str,
    duration: float,
) -> Tuple[str, int]:
    """Render caption PNG strips + an ffconcat list.  Returns (list_path, strip_height)."""
    from PIL import Image, ImageDraw, ImageFont

    os.makedirs(out_dir, exist_ok=True)
    font_path = find_font(cfg.font, cfg.bold)
    if font_path:
        font = ImageFont.truetype(font_path, style.font_size)
    else:  # pragma: no cover
        log.warning("No TrueType font found; using Pillow default font")
        font = ImageFont.load_default(size=style.font_size)
    W, H = style.width, style.height
    max_lane = max((l.lane for l in lines), default=0)
    strip_h = min(H, style.margin_v + (max_lane + 1) * style.lane_height + style.font_size)
    stroke = max(1, int(round(style.outline)))
    sh = max(1, int(round(style.shadow)))
    colors = {"A": parse_color(cfg.color_a), "B": parse_color(cfg.color_b)}

    lines = sorted(lines, key=lambda l: l.start)
    bounds = sorted({0.0, round(duration, 3)} | {round(l.start, 3) for l in lines}
                    | {round(min(l.end, duration), 3) for l in lines})
    bounds = [b for b in bounds if 0.0 <= b <= duration]
    cache: Dict[Tuple[int, ...], str] = {}
    blank = os.path.join(out_dir, "blank.png")
    Image.new("RGBA", (W, strip_h), (0, 0, 0, 0)).save(blank)
    entries: List[Tuple[str, float]] = []
    for a, b in zip(bounds, bounds[1:]):
        if b - a <= 1e-6:
            continue
        active = tuple(i for i, l in enumerate(lines) if l.start <= a + 1e-6 and l.end > a + 1e-6)
        if not active:
            entries.append(("blank.png", b - a))
            continue
        if active not in cache:
            img = Image.new("RGBA", (W, strip_h), (0, 0, 0, 0))
            dr = ImageDraw.Draw(img)
            for i in active:
                ln = lines[i]
                rows = _wrap(dr, ln.text, font, W - 2 * style.margin_lr, stroke)
                row_h = int(style.font_size * 1.2)
                y_bottom = strip_h - ln.margin_v
                col = colors.get(ln.speaker, (255, 255, 255))
                for r_i, row in enumerate(reversed(rows)):
                    y = y_bottom - r_i * row_h
                    x = W // 2
                    dr.text((x + sh, y + sh), row, font=font, anchor="md", fill=(0, 0, 0, 150),
                            stroke_width=stroke, stroke_fill=(0, 0, 0, 150))
                    dr.text((x, y), row, font=font, anchor="md", fill=col + (255,),
                            stroke_width=stroke, stroke_fill=(0, 0, 0, 255))
            name = f"cap_{len(cache):05d}.png"
            img.save(os.path.join(out_dir, name), optimize=False, compress_level=3)
            cache[active] = name
        entries.append((cache[active], b - a))
    list_path = os.path.join(out_dir, "captions_overlay.ffconcat")
    with open(list_path, "w") as f:
        f.write("ffconcat version 1.0\n")
        for name, d in entries:
            f.write(f"file '{name}'\nduration {d:.6f}\n")
        # the last entry must be repeated for its duration to be honoured
        f.write("file 'blank.png'\n")
    log.info("Rendered %d caption strips (%dx%d) for Pillow burn-in", len(cache), W, strip_h)
    return list_path, strip_h
