"""End-to-end render smoke test on tiny synthetic clips (needs ffmpeg; no ML models)."""

import json
import os
import subprocess

import numpy as np
import pytest

from podcast_autoeditor import CAM_A, CAM_B, WIDE
from podcast_autoeditor.captions import CaptionLine, layout_lines, make_style, write_ass
from podcast_autoeditor.config import CaptionConfig, RenderConfig
from podcast_autoeditor.datatypes import Shot
from podcast_autoeditor.ffmpeg_utils import FFmpegError, find_ffmpeg, probe
from podcast_autoeditor.render import AudioSource, SourceClip, render_final, render_segments

try:
    FF = find_ffmpeg()
except FFmpegError:  # pragma: no cover
    FF = None

pytestmark = pytest.mark.skipif(FF is None, reason="ffmpeg not installed")

W, H, FPS = 320, 180, 30.0
COLORS = {CAM_A: (200, 0, 0), CAM_B: (0, 160, 0), WIDE: (90, 90, 90)}


def _clip(path, rgb, dur, offset_pad=0.0, size="320x180", rate=25):
    col = "0x%02X%02X%02X" % rgb
    subprocess.run([FF.ffmpeg, "-hide_banner", "-loglevel", "error", "-y",
                    "-f", "lavfi", "-i", f"color=c={col}:s={size}:r={rate}:d={dur}",
                    "-f", "lavfi", "-i", f"sine=frequency=440:sample_rate=48000:duration={dur}",
                    "-c:v", "libx264" if FF.has_libx264 else "h264_videotoolbox", "-pix_fmt", "yuv420p",
                    "-c:a", "aac", "-shortest", str(path)], check=True)


def _frame(path, t):
    out = subprocess.run([FF.ffmpeg, "-hide_banner", "-loglevel", "error", "-ss", f"{t}", "-i", str(path),
                          "-frames:v", "1", "-f", "rawvideo", "-pix_fmt", "rgb24", "-"],
                         capture_output=True, check=True).stdout
    return np.frombuffer(out, np.uint8).reshape(H, W, 3)


def _shots():
    spec = [(CAM_A, 0, 45), (WIDE, 45, 81), (CAM_B, 81, 120)]
    return [Shot(a / FPS, b / FPS if i < 2 else 4.0, c, "test", a, b) for i, (c, a, b) in enumerate(spec)]


@pytest.mark.parametrize("burner", ["ass", "pillow"])
def test_render_pipeline(tmp_path, burner):
    if burner == "ass" and not FF.has_libass:
        pytest.skip("ffmpeg without libass")
    paths = {}
    for cam, size, rate in ((CAM_A, "640x360", 30), (CAM_B, "320x180", 25), (WIDE, "480x270", 30)):
        p = tmp_path / f"{cam}.mp4"
        _clip(p, COLORS[cam], 6.0, size=size, rate=rate)
        paths[cam] = str(p)
    # cam B starts 0.5 s after the reference -> offset +0.5
    sources = {c: SourceClip(p, 0.5 if c == CAM_B else 0.0, probe(FF, p)) for c, p in paths.items()}
    cfg = RenderConfig(width=W, height=H, fps=FPS, preset="ultrafast", jobs=2)
    shots = _shots()
    segs = render_segments(FF, shots, sources, cfg, str(tmp_path / "segs"))
    assert len(segs) == 3 and all(os.path.getsize(s) > 0 for s in segs)
    cc = CaptionConfig(font_size_pct=9.0)
    lines = [CaptionLine("A", 0.3, 2.6, "Hello there"), CaptionLine("B", 1.6, 3.6, "Hi back")]
    style = make_style(cc, W, H)
    layout_lines(lines, style)
    assert lines[0].margin_v != lines[1].margin_v
    ass = overlay = None
    if burner == "ass":
        ass = write_ass(str(tmp_path / "c.ass"), lines, style, cc)
    else:
        from podcast_autoeditor.caption_overlay import render_overlay

        overlay, _ = render_overlay(lines, style, cc, str(tmp_path / "png"), 4.0)
    out = tmp_path / "out.mp4"
    render_final(FF, segs, AudioSource(paths[WIDE], 0.0), 4.0, cfg, str(tmp_path), str(out), ass, overlay)
    pr = json.loads(subprocess.run([FF.ffprobe, "-v", "error", "-count_frames", "-print_format", "json",
                                    "-show_streams", str(out)], capture_output=True, text=True).stdout)
    v = next(s for s in pr["streams"] if s["codec_type"] == "video")
    a = next(s for s in pr["streams"] if s["codec_type"] == "audio")
    assert int(v["nb_read_frames"]) == 120
    assert abs(float(a["duration"]) - 4.0) < 0.05
    # camera identity from the background (top-left corner, away from captions)
    for t, cam in ((0.5, CAM_A), (2.0, WIDE), (3.5, CAM_B)):
        px = _frame(out, t)[10:30, 10:30].reshape(-1, 3).mean(0)
        assert np.linalg.norm(px - np.array(COLORS[cam])) < 40, (t, cam, px)
    # at t=2.0 both captions are visible: blue (A) above/below white (B), not on the same rows
    f = _frame(out, 2.0).astype(int)
    blue = (np.abs(f - np.array([45, 156, 255])).sum(-1) < 90)
    white = (f.min(-1) > 225)
    assert blue.sum() > 30 and white.sum() > 30
    rows_b = set(np.nonzero(blue.any(1))[0])
    rows_w = set(np.nonzero(white.any(1))[0])
    assert len(rows_b & rows_w) < 0.2 * min(len(rows_b), len(rows_w))


def test_rotation_metadata_is_honoured(tmp_path):
    """A landscape-encoded clip flagged as rotated 90deg is decoded portrait (ffmpeg
    autorotate) and pillarboxed into the landscape output instead of being squashed."""
    raw = tmp_path / "raw.mp4"
    rot = tmp_path / "rot.mp4"
    _clip(raw, (200, 0, 0), 2.0, size="320x180", rate=30)
    r = subprocess.run([FF.ffmpeg, "-hide_banner", "-loglevel", "error", "-y", "-display_rotation", "90",
                        "-i", str(raw), "-c", "copy", str(rot)], capture_output=True, text=True)
    if r.returncode != 0:
        pytest.skip(f"ffmpeg cannot write display rotation: {r.stderr[-200:]}")
    info = probe(FF, str(rot))
    assert info.rotation in (90, 270)
    src = SourceClip(str(rot), 0.0, info)
    cfg = RenderConfig(width=W, height=H, fps=FPS, preset="ultrafast", jobs=1)
    shot = Shot(0.0, 1.0, CAM_A, "test", 0, 30)
    segs = render_segments(FF, [shot], {CAM_A: src}, cfg, str(tmp_path / "segs"))
    out = tmp_path / "o.mp4"
    render_final(FF, segs, None, 1.0, cfg, str(tmp_path), str(out))
    f = _frame(out, 0.5).astype(int)
    assert f[:, :20].max() < 40 and f[:, -20:].max() < 40      # black pillars left/right
    assert np.linalg.norm(f[H // 2, W // 2] - np.array([200, 0, 0])) < 50  # content in the middle


def test_visual_frame_reader_bounds_portrait_frames(tmp_path):
    """Regression: the mouth-motion reader scaled frames to 640 px *wide*, so a portrait
    close-up was decoded at 640x1138 (3x the memory of a landscape clip)."""
    from podcast_autoeditor.mapping import _read_frames

    for size, want in (("1080x1920", (640, 360)), ("1920x1080", (360, 640))):
        p = tmp_path / f"{size}.mp4"
        _clip(p, (200, 0, 0), 1.0, size=size, rate=30)
        frames, _ = _read_frames(FF.ffmpeg, str(p), 0.0, 1.0, 5.0)
        assert len(frames) == 5 and frames[0].shape[:2] == want, frames[0].shape


def test_visual_reader_respects_frame_cap_and_deadline(tmp_path):
    """Regression for the unexplained 10-minute visual-analysis stall: decoding is
    streamed and bounded by a frame cap and a wall-clock deadline."""
    import time

    from podcast_autoeditor.mapping import _iter_frames

    p = tmp_path / "v.mp4"
    _clip(p, (0, 0, 200), 4.0, size="640x360", rate=30)
    info = {}
    got = list(_iter_frames(FF.ffmpeg, str(p), 0.0, 4.0, 5.0, max_frames=3, info=info))
    assert len(got) == 3 and info.get("capped") and info["frames"] == 3
    info = {}
    got = list(_iter_frames(FF.ffmpeg, str(p), 0.0, 4.0, 5.0, deadline=time.monotonic() - 1, info=info))
    assert got == [] and info.get("timed_out")
    info = {}
    got = list(_iter_frames(FF.ffmpeg, str(p), 0.0, 4.0, 5.0, info=info))
    assert len(got) == 20 and not info.get("capped") and not info.get("timed_out")


def test_anamorphic_source_is_rendered_in_square_pixels(tmp_path):
    """Regression (reviewer B, round 3): a non-square-pixel source (HDV 1440x1080 SAR 4:3,
    here 160x180 SAR 2:1 = 320x180 on screen) was treated as storage pixels: the WIDE shot
    came out squeezed with black pillars, and single-clip crops (display coordinates) cut
    the wrong region and stretched it."""
    from podcast_autoeditor.mapping import _iter_frames, _probe_display_size

    src = tmp_path / "anamorphic.mp4"
    # storage 160x180: left half red, right half green; flagged SAR 2:1
    subprocess.run([FF.ffmpeg, "-hide_banner", "-loglevel", "error", "-y",
                    "-f", "lavfi", "-i", "color=c=0xC80000:s=80x180:r=30:d=2",
                    "-f", "lavfi", "-i", "color=c=0x00A000:s=80x180:r=30:d=2",
                    "-filter_complex", "[0:v][1:v]hstack,setsar=2/1[v]", "-map", "[v]",
                    "-c:v", "libx264" if FF.has_libx264 else "h264_videotoolbox", "-pix_fmt", "yuv420p",
                    str(src)], check=True)
    assert _probe_display_size(FF.ffmpeg, str(src)) == (320, 180)
    fr = next(iter(_iter_frames(FF.ffmpeg, str(src), 0.0, 1.0, 5.0, width=320)))
    assert fr.shape[:2] == (180, 320)
    info = probe(FF, str(src))
    cfg = RenderConfig(width=W, height=H, fps=FPS, preset="ultrafast", jobs=1)
    shots = [Shot(0.0, 1.0, WIDE, "test", 0, 30), Shot(1.0, 2.0, CAM_A, "test", 30, 60)]
    sources = {WIDE: SourceClip(str(src), 0.0, info), CAM_A: SourceClip(str(src), 0.0, info)}
    # close-up = left-top quarter of the DISPLAYED frame -> red only
    segs = render_segments(FF, shots, sources, cfg, str(tmp_path / "segs"), crops=[None, [0, 0, 160, 90]])
    out = tmp_path / "o.mp4"
    render_final(FF, segs, None, 2.0, cfg, str(tmp_path), str(out))
    wide = _frame(out, 0.5).astype(int)
    red, green = np.array([200, 0, 0]), np.array([0, 160, 0])
    assert np.linalg.norm(wide[H // 2, 8] - red) < 60 and np.linalg.norm(wide[H // 2, W - 8] - green) < 60  # no pillars
    assert np.linalg.norm(wide[H // 2, W // 4] - red) < 60 and np.linalg.norm(wide[H // 2, 3 * W // 4] - green) < 60
    close = _frame(out, 1.5).astype(int)
    assert np.linalg.norm(close[H // 2, W - 8] - red) < 60, close[H // 2, W - 8]   # still red at the right edge
