#!/usr/bin/env python3
"""PodBot web UI for the podcast auto-editor (local only: binds to 127.0.0.1).

Run:  .venv/bin/python app.py   ->  http://127.0.0.1:7860

One dropzone (1 wide clip = single-clip mode, 3 clips = multicam), three numbered sections and
an "Advanced settings" drawer whose untouched defaults give exactly the PipelineConfig of the
previous UI (see build_config / DEFAULTS).  Styling lives in ui/podbot.css + ui/podbot.js, the
HTML widgets in ui/render.py.
"""

from __future__ import annotations

import argparse
import hashlib
import logging
import math
import os
import queue
import subprocess
import sys
import threading
import time
from types import SimpleNamespace
from typing import Any, Dict, List, Optional, Sequence, Tuple

ROOT = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, ROOT)

import gradio as gr  # noqa: E402

from podcast_autoeditor.captions import parse_color  # noqa: E402
from podcast_autoeditor.cli import setup_logging  # noqa: E402
from podcast_autoeditor.config import (AsrConfig, CaptionConfig, DiarizationConfig,  # noqa: E402
                                       MappingConfig, PipelineConfig, PlannerConfig, ReframeConfig,
                                       RenderConfig, SyncConfig)
from podcast_autoeditor.ffmpeg_utils import find_ffmpeg, probe  # noqa: E402
from podcast_autoeditor.pipeline import PipelineError, run_pipeline  # noqa: E402
from ui import render as R  # noqa: E402

RUNS_DIR = os.environ.get("PODCAST_AUTOEDITOR_RUNS", os.path.join(ROOT, "ui_runs"))
UI_DIR = os.path.join(ROOT, "ui")
EDL_COLS = ["index", "camera", "reason", "start", "end", "duration", "source", "source_in", "source_out",
            "crop"]
log = logging.getLogger("app")


class _ListHandler(logging.Handler):
    def __init__(self) -> None:
        super().__init__(logging.INFO)
        self.lines: List[str] = []
        self.warnings: List[str] = []
        self.setFormatter(logging.Formatter("%(asctime)s %(levelname)s %(name)s: %(message)s", "%H:%M:%S"))

    def emit(self, record: logging.LogRecord) -> None:
        self.lines.append(self.format(record))
        if record.levelno >= logging.WARNING:
            self.warnings.append(record.getMessage())


def _path(f: Any) -> Optional[str]:
    if f is None:
        return None
    if isinstance(f, str):
        return f
    return getattr(f, "name", None) or (f.get("path") if isinstance(f, dict) else None)


def _paths(files: Any) -> List[str]:
    if files is None:
        return []
    if not isinstance(files, (list, tuple)):
        files = [files]
    return [p for p in (_path(f) for f in files) if p]


def _hex(c: str, default: str) -> str:
    try:
        r, g, b = parse_color(c or default)
        return f"#{r:02X}{g:02X}{b:02X}"
    except ValueError:
        return default


def _opt_float(v) -> Optional[float]:
    try:
        return None if v is None or str(v).strip() == "" else float(v)
    except ValueError:
        raise ValueError(R.OFFSET_ERROR)


# --------------------------------------------------------------------------- controls (brief s.4)

ACCURACY = {"Fast": "base", "Balanced": "small", "Best": "large-v3"}  # -> AsrConfig.model_size
AUTO_LANG = "Auto-detect"
LANGUAGES = [(AUTO_LANG, AUTO_LANG), ("English (en)", "en"), ("Hindi (hi)", "hi"),
             ("Punjabi (pa)", "pa"), ("Spanish (es)", "es"), ("French (fr)", "fr"),
             ("German (de)", "de"), ("Japanese (ja)", "ja"), ("Chinese (zh)", "zh")]
SPEAKERS = [("Auto", "auto"), ("1", "1"), ("2", "2")]
A_SIDES = [("Left person", "left"), ("Right person", "right")]
CUT_KEYS = ("qa_gap", "qa_hold", "rapid_turns", "rapid_window", "min_shot", "silence_wide",
            "question_heuristic")
CUT_PRESETS: Dict[str, Dict[str, Any]] = {
    "Calm (fewer cuts)": dict(qa_gap=1.5, qa_hold=2.5, rapid_turns=4, rapid_window=6.0, min_shot=2.0,
                              silence_wide=3.5, question_heuristic=True),
    "Balanced": dict(qa_gap=1.5, qa_hold=2.0, rapid_turns=3, rapid_window=6.0, min_shot=1.2,
                     silence_wide=2.5, question_heuristic=True),
    "Snappy (more cuts)": dict(qa_gap=1.5, qa_hold=1.5, rapid_turns=3, rapid_window=6.0, min_shot=0.8,
                               silence_wide=2.0, question_heuristic=True),
}
CUSTOM = "Custom"
QUALITY: Dict[str, Tuple[int, int, int]] = {"1080p · 30 fps": (1920, 1080, 30),
                                            "720p · 30 fps": (1280, 720, 30),
                                            "4K · 30 fps": (3840, 2160, 30)}

# Untouched controls.  These give the same PipelineConfig as the previous UI's defaults.
DEFAULTS: Dict[str, Any] = {
    "master": None, "a_side": "left", "color_a": "#2D9CFF", "color_b": "#FFFFFF",
    "accuracy": "Balanced", "language": AUTO_LANG, "num_speakers": "auto", "swap": False,
    **CUT_PRESETS["Balanced"],
    "quality": "1080p · 30 fps", "width": 1920, "height": 1080, "fps": 30,
    "offset_a": "", "offset_b": "", "offset_wide": "",
    "crop_scale": 3.2, "max_zoom": 3.0,
}
SETTING_KEYS = tuple(DEFAULTS)
RUN_LABEL, PLAN_LABEL, BUSY_LABEL = "Edit my podcast", "Preview cut plan", "Editing…"


def build_config(files: Any, roles: Sequence[Any] = (None, None, None), plan_only: bool = False,
                 run_dir: Optional[str] = None, **settings: Any) -> PipelineConfig:
    """PipelineConfig for the uploaded clips and the form's settings (missing = DEFAULTS).

    1 file = single-clip mode, 3 files = multicam with ``roles`` = (Speaker A close-up,
    Speaker B close-up, wide) chosen from ``files`` (all None = guess from the file names).
    Pure: no Gradio calls; raises ValueError for input the form blocks (wrong file count,
    duplicate roles, bad offsets or numbers)."""
    unknown = set(settings) - set(DEFAULTS)
    if unknown:
        raise TypeError(f"unknown settings: {sorted(unknown)}")
    s = {**DEFAULTS, **settings}
    paths = _paths(files)
    single = len(paths) == 1
    if single:
        a = b = ""
        w = paths[0]
    elif len(paths) == 3:
        if not any(roles):  # API callers may leave the roles to the filename guess
            roles = R.guess_roles(paths)
        a, b, w = (_path(r) for r in roles)
        if not (a and b and w) or len({a, b, w}) < 3 or not {a, b, w} <= set(paths):
            raise ValueError(R.ROLE_ERROR)
    else:
        raise ValueError(f"PodBot needs 1 wide clip or 3 clips (A, B, wide). You added {len(paths)}.")
    if run_dir is None:
        run_dir = os.path.join(RUNS_DIR, time.strftime("%Y%m%d-%H%M%S"))
    width, height, fps = QUALITY.get(s["quality"]) or (s["width"], s["height"], s["fps"])
    if any(v is None or not math.isfinite(float(v)) or float(v) <= 0 for v in (width, height, fps)):
        raise ValueError("Width, height and FPS must be positive numbers.")
    lang = str(s["language"] or "").strip()
    key_clip = w if single else a
    return PipelineConfig(
        mode="single" if single else "multicam",
        reframe=ReframeConfig(a_side=str(s["a_side"] or "left"), crop_scale=float(s["crop_scale"]),
                              max_zoom=float(s["max_zoom"])),
        cam_a=a, cam_b=b, wide=w, audio=_path(s["master"]), out=os.path.join(run_dir, "edited.mp4"),
        work_dir=os.path.join(RUNS_DIR, "_cache",
                              f"{'single-' if single else ''}{os.path.basename(key_clip)}-"
                              f"{os.path.getsize(key_clip)}"),
        plan_only=bool(plan_only),
        asr=AsrConfig(model_size=ACCURACY.get(s["accuracy"], s["accuracy"]),
                      language=None if lang.lower() in ("", AUTO_LANG.lower()) else lang),
        # single-clip mode hides the cam A/B offset fields; they cannot apply there
        sync=SyncConfig(offset_a=None if single else _opt_float(s["offset_a"]),
                        offset_b=None if single else _opt_float(s["offset_b"]),
                        offset_wide=_opt_float(s["offset_wide"])),
        diar=DiarizationConfig(num_speakers=str(s["num_speakers"] or "auto")),
        mapping=MappingConfig(swap=bool(s["swap"])),
        planner=PlannerConfig(qa_gap=float(s["qa_gap"]), qa_answer_hold=float(s["qa_hold"]),
                              rapid_turns=int(s["rapid_turns"]), rapid_window=float(s["rapid_window"]),
                              min_shot=float(s["min_shot"]), silence_wide=float(s["silence_wide"]),
                              question_heuristic=bool(s["question_heuristic"])),
        captions=CaptionConfig(color_a=_hex(s["color_a"], "#2D9CFF"), color_b=_hex(s["color_b"], "#FFFFFF")),
        render=RenderConfig(width=int(width), height=int(height), fps=float(fps)),
    )


def _offset_ok(v: Any) -> bool:
    try:
        x = _opt_float(v)
    except ValueError:
        return False
    return x is None or math.isfinite(x)


def form_check(paths: Sequence[str], roles: Sequence[Optional[str]], offsets: Sequence[Any]
               ) -> Dict[str, Any]:
    """What the form may do right now: ``ready`` to run, plus which inline errors to show."""
    n = len(paths)
    bad_offsets = [(n != 1 or i == 2) and not _offset_ok(v) for i, v in enumerate(offsets)]
    picked = [r for r in roles if r]
    dup_roles = n == 3 and (len(picked) < 3 or len(set(picked)) < 3 or not set(picked) <= set(paths))
    ready = n in (1, 3) and not dup_roles and not any(bad_offsets)
    helper = "" if ready else ("Add your video to start." if n == 0 else "Fix the problem above to start.")
    return {"ready": ready, "count_error": n not in (0, 1, 3), "dup_roles": dup_roles,
            "bad_offsets": bad_offsets, "helper": helper}


def _thumbnail(path: str) -> Optional[str]:
    """JPEG frame at 10% of the clip for the "Who is Speaker A?" picker; None on any failure."""
    try:
        st = os.stat(path)
        key = hashlib.sha1(f"{os.path.abspath(path)}|{st.st_size}|{st.st_mtime_ns}".encode()).hexdigest()
        out = os.path.join(RUNS_DIR, "_thumbs", f"{key[:16]}.jpg")
        if not os.path.isfile(out):
            ff = find_ffmpeg()
            info = probe(ff, path)
            t = 0.10 * (info.video_duration or info.duration or 0.0)
            os.makedirs(os.path.dirname(out), exist_ok=True)
            subprocess.run([ff.ffmpeg, "-v", "error", "-y", "-ss", f"{t:.3f}", "-i", path,
                            "-frames:v", "1", "-vf", "scale=640:-2", out],
                           check=True, capture_output=True, timeout=60)
        return out if os.path.isfile(out) and os.path.getsize(out) > 0 else None
    except Exception as e:  # noqa: BLE001  - the thumbnail is optional
        log.info("no thumbnail for %s: %s", path, e)
        return None


# --------------------------------------------------------------------------- theme

_RED = gr.themes.Color(c50="#FDECEE", c100="#FAD3D8", c200="#F4A7B0", c300="#EC7A88", c400="#FF5468",
                       c500="#CF1830", c600="#B8142B", c700="#A8122A", c800="#870F22", c900="#6B0C1B",
                       c950="#3D0710", name="podbot_red")
_GREY = gr.themes.Color(c50="#F8F4F4", c100="#F1ECEC", c200="#E3DCDC", c300="#C9C1C2", c400="#8C8587",
                        c500="#6E686B", c600="#5E5A5B", c700="#3C393B", c800="#2C2A2E", c900="#18171A",
                        c950="#0F0E10", name="podbot_grey")


def make_theme() -> gr.themes.Base:
    L = dict(bg="#FFFFFF", surface="#F8F4F4", ink="#1B1B1D", muted="#5E5A5B", line="#E3DCDC",
             strong="#8C8587", brand="#CF1830", hover="#A8122A", on="#FFFFFF", soft="#FDECEE")
    D = dict(bg="#0F0E10", surface="#18171A", ink="#F0EDE6", muted="#A9A3A0", line="#2C2A2E",
             strong="#6E686B", brand="#FF5468", hover="#FF7383", on="#0F0E10", soft="#2A1418")
    pairs = {  # theme variable -> token (light value, and the same token in dark)
        "body_background_fill": "bg", "background_fill_primary": "bg",
        "background_fill_secondary": "surface", "body_text_color": "ink",
        "body_text_color_subdued": "muted", "block_title_text_color": "ink",
        "block_label_text_color": "muted", "block_info_text_color": "muted",
        "accordion_text_color": "ink", "border_color_primary": "line", "border_color_accent": "brand",
        "color_accent": "brand", "color_accent_soft": "soft", "link_text_color": "brand",
        "link_text_color_hover": "hover", "link_text_color_visited": "brand",
        "link_text_color_active": "brand", "loader_color": "brand", "slider_color": "brand",
        "block_background_fill": "bg", "block_border_color": "line", "panel_background_fill": "surface",
        "panel_border_color": "line", "input_background_fill": "bg", "input_background_fill_focus": "bg",
        "input_background_fill_hover": "bg", "input_border_color": "strong",
        "input_border_color_hover": "ink", "input_border_color_focus": "brand",
        "input_placeholder_color": "muted", "button_primary_background_fill": "brand",
        "button_primary_background_fill_hover": "hover", "button_primary_border_color": "brand",
        "button_primary_border_color_hover": "hover", "button_primary_text_color": "on",
        "button_primary_text_color_hover": "on", "button_secondary_background_fill": "bg",
        "button_secondary_background_fill_hover": "surface", "button_secondary_border_color": "strong",
        "button_secondary_border_color_hover": "ink", "button_secondary_text_color": "ink",
        "button_secondary_text_color_hover": "ink", "checkbox_background_color": "bg",
        "checkbox_background_color_selected": "brand", "checkbox_border_color": "strong",
        "checkbox_border_color_selected": "brand", "checkbox_border_color_focus": "brand",
        "checkbox_border_color_hover": "ink", "checkbox_label_background_fill": "bg",
        "checkbox_label_background_fill_hover": "surface",
        "checkbox_label_background_fill_selected": "soft", "checkbox_label_border_color": "strong",
        "checkbox_label_border_color_hover": "ink", "checkbox_label_border_color_selected": "brand",
        "checkbox_label_text_color": "ink", "checkbox_label_text_color_selected": "ink",
        "table_border_color": "line", "table_even_background_fill": "bg",
        "table_odd_background_fill": "surface", "code_background_fill": "surface",
        "block_label_background_fill": "bg", "block_label_border_color": "line",
    }
    base = gr.themes.Base(
        primary_hue=_RED, secondary_hue=_RED, neutral_hue=_GREY,
        radius_size=gr.themes.sizes.radius_sm,
        font=[gr.themes.LocalFont("IBM Plex Sans"), "system-ui", "-apple-system", "Segoe UI", "sans-serif"],
        font_mono=[gr.themes.LocalFont("IBM Plex Mono"), "ui-monospace", "Menlo", "Consolas", "monospace"],
    )
    values: Dict[str, str] = {}
    for var, tok in pairs.items():
        values[var] = L[tok]
        if hasattr(base, var + "_dark"):  # a few variables have no dark twin
            values[var + "_dark"] = D[tok]
    values.update(
        block_border_width="0px", block_padding="0px", block_shadow="none", block_radius="2px",
        container_radius="2px",
        block_label_radius="2px", block_label_shadow="none", input_radius="2px", input_shadow="none",
        input_shadow_focus="none", input_border_width="1px", button_border_width="1px",
        button_large_radius="2px", button_medium_radius="2px", button_small_radius="2px",
        button_primary_shadow="none", button_primary_shadow_hover="none", button_primary_shadow_active="none",
        button_secondary_shadow="none", button_secondary_shadow_hover="none",
        button_secondary_shadow_active="none", button_transform_hover="none", button_transform_active="none",
        button_transition="background-color 150ms ease-out, border-color 150ms ease-out, color 150ms ease-out",
        checkbox_shadow="none", checkbox_label_shadow="none", checkbox_label_border_width="1px",
        checkbox_border_radius="2px", shadow_drop="none", shadow_drop_lg="none", shadow_inset="none",
        shadow_spread="0px", embed_radius="2px", table_radius="2px",
        error_background_fill="#FDECEE", error_background_fill_dark="#2A1418",
        error_border_color="#A8122A", error_border_color_dark="#FF7383",
        error_text_color="#A8122A", error_text_color_dark="#FF7383",
        error_icon_color="#A8122A", error_icon_color_dark="#FF7383",
        body_text_size="16px", prose_text_size="16px",
    )
    return base.set(**values)


def launch_kwargs() -> Dict[str, Any]:
    """Theme/CSS/JS for Blocks.launch() (Gradio 6: these are launch() parameters)."""
    with open(os.path.join(UI_DIR, "podbot.js"), encoding="utf-8") as f:
        js = f.read()
    return dict(theme=make_theme(), css_paths=[os.path.join(UI_DIR, "podbot.css")], js=js,
                footer_links=[])


# --------------------------------------------------------------------------- UI

# client-side: freeze the form while a job runs (inert = no mouse, keyboard or AT focus)
_FREEZE = "(...args) => { document.getElementById('pb-form')?.setAttribute('inert', ''); return args; }"
_UNFREEZE = "() => { document.getElementById('pb-form')?.removeAttribute('inert'); }"

# One pipeline at a time.  Gradio frees the "pipeline" concurrency slot as soon as a run's page
# goes away (reload, closed tab), but the pipeline thread cannot be stopped and keeps running; a
# new run must wait for it instead of sharing the cache work_dir and the log handler with it.
_PIPELINE_LOCK = threading.Lock()
WAIT_MSG = "waiting for the previous edit to finish"


def _html(value: str = "", **kw: Any) -> gr.HTML:
    return gr.HTML(value, js_on_load=None, apply_default_css=False, **kw)


def build_ui() -> gr.Blocks:
    c = SimpleNamespace()  # every component, by name
    with gr.Blocks(title="PodBot", analytics_enabled=False) as demo:  # the page says LOCAL · OFFLINE
        with gr.Column(elem_id="pb-page"):
            _html(R.header_html(), elem_id="pb-head")
            with gr.Column(elem_id="pb-form"):
                # ---------------------------------------------------------- 01 upload
                _html(R.section_head("01", "Upload", "UPLOAD"), elem_classes=["pb-sec"])
                c.drop = gr.File(label="Your video", file_count="multiple", file_types=["video"],
                                 type="filepath", elem_id="pb-drop")
                _html(R.helper_html("1 wide clip with both people, or 3 clips: Speaker A close-up, "
                                    "Speaker B close-up and a wide shot."), elem_id="pb-drop-help")
                c.detect = _html(visible=False, elem_id="pb-detect")
                c.count_err = _html(visible=False, elem_id="pb-count-err")
                with gr.Row(visible=False, elem_id="pb-roles") as c.roles_row:
                    c.role_a = gr.Dropdown([], label="Speaker A close-up", elem_id="pb-role-a")
                    c.role_b = gr.Dropdown([], label="Speaker B close-up", elem_id="pb-role-b")
                    c.role_wide = gr.Dropdown([], label="Wide shot", elem_id="pb-role-wide")
                c.role_err = _html(R.inline_error(R.ROLE_ERROR), visible=False, elem_id="pb-role-err")

                # ---------------------------------------------------------- 02 speakers & captions
                _html(R.section_head("02", "Speakers & captions", "SPEAKERS"), elem_classes=["pb-sec"])
                with gr.Row(visible=False, elem_id="pb-single", equal_height=False) as c.single_row:
                    c.thumb = gr.Image(label="Frame from your clip", show_label=False, interactive=False,
                                       type="filepath", elem_id="pb-thumb", buttons=[], scale=1,
                                       alt_text="Frame from your clip: the person on the left and the person "
                                                "on the right")
                    with gr.Column(scale=1, elem_id="pb-aside-col"):
                        c.a_side = gr.Radio(A_SIDES, value=DEFAULTS["a_side"], label="Who is Speaker A?",
                                            info="Speaker A gets the Speaker A caption colour and close-up.",
                                            elem_id="pb-aside")
                with gr.Row(elem_id="pb-colors", equal_height=False):
                    with gr.Column(scale=1, min_width=0, elem_id="pb-pickers"):
                        c.color_a = gr.ColorPicker(DEFAULTS["color_a"], label="Speaker A captions",
                                                   elem_id="pb-color-a")
                        c.color_b = gr.ColorPicker(DEFAULTS["color_b"], label="Speaker B captions",
                                                   elem_id="pb-color-b")
                        _html(R.helper_html("Captions are burned into the edited video in these colours."))
                    c.cap_preview = _html(R.caption_preview_html(DEFAULTS["color_a"], DEFAULTS["color_b"]),
                                          elem_id="pb-cap-preview", scale=1, min_width=0)

                # ---------------------------------------------------------- 03 edit
                _html(R.section_head("03", "Edit", "EDIT"), elem_classes=["pb-sec"])
                with gr.Accordion("Advanced settings", open=False, elem_id="pb-adv") as c.adv:
                    _html(R.group_label("Speech"))
                    with gr.Row(elem_classes=["pb-grp"]):
                        c.accuracy = gr.Radio(list(ACCURACY), value=DEFAULTS["accuracy"], label="Accuracy",
                                              info="Best is a large download on first use. Other Whisper "
                                                   "models: CLI --whisper-model.", elem_id="pb-accuracy")
                        c.language = gr.Dropdown(LANGUAGES, value=DEFAULTS["language"], label="Language",
                                                 allow_custom_value=True, info="Or type any language code.",
                                                 elem_id="pb-language")
                    with gr.Row(elem_classes=["pb-grp"]):
                        c.num_speakers = gr.Radio(SPEAKERS, value=DEFAULTS["num_speakers"],
                                                  label="People talking",
                                                  info="Auto detects a one-person recording.",
                                                  elem_id="pb-speakers")
                        c.swap = gr.Checkbox(DEFAULTS["swap"], label="Swap Speaker A and B",
                                             info="Use when the close-ups show the wrong person.",
                                             elem_id="pb-swap")
                    _html(R.group_label("Cut style"))
                    c.cut_style = gr.Radio([*CUT_PRESETS, CUSTOM], value="Balanced", label="Cut style",
                                           show_label=False, elem_id="pb-cut-style")
                    with gr.Accordion("Fine-tune", open=False, elem_id="pb-finetune"):
                        with gr.Row(elem_classes=["pb-grp"]):
                            c.qa_gap = gr.Slider(0.2, 5.0, DEFAULTS["qa_gap"], step=0.1,
                                                 label="Question → answer gap (s)")
                            c.qa_hold = gr.Slider(0.5, 6.0, DEFAULTS["qa_hold"], step=0.1,
                                                  label="Stay wide into answer (s)")
                            c.min_shot = gr.Slider(0.5, 5.0, DEFAULTS["min_shot"], step=0.1,
                                                   label="Shortest shot (s)")
                        with gr.Row(elem_classes=["pb-grp"]):
                            c.rapid_turns = gr.Slider(2, 8, DEFAULTS["rapid_turns"], step=1,
                                                      label="Back-and-forth: speaker changes")
                            c.rapid_window = gr.Slider(2.0, 20.0, DEFAULTS["rapid_window"], step=0.5,
                                                       label="Back-and-forth: within (s)")
                            c.silence_wide = gr.Slider(1.0, 10.0, DEFAULTS["silence_wide"], step=0.5,
                                                       label="Go wide after silence (s)")
                        c.question_heuristic = gr.Checkbox(DEFAULTS["question_heuristic"],
                                                           label='Spot questions without "?"')
                    _html(R.group_label("Output"))
                    c.quality = gr.Radio([*QUALITY, CUSTOM], value=DEFAULTS["quality"], label="Quality",
                                         show_label=False, elem_id="pb-quality")
                    with gr.Row(visible=False, elem_classes=["pb-grp"]) as c.custom_row:
                        c.width = gr.Number(DEFAULTS["width"], label="Width", precision=0, minimum=16)
                        c.height = gr.Number(DEFAULTS["height"], label="Height", precision=0, minimum=16)
                        c.fps = gr.Number(DEFAULTS["fps"], label="FPS", minimum=1)
                    _html(R.group_label("Sync"))
                    c.master = gr.File(label="Master audio (optional)", file_types=["audio"],
                                       type="filepath", elem_id="pb-master")
                    with gr.Row(elem_classes=["pb-grp"]):
                        with gr.Column(min_width=160) as c.off_a_col:
                            c.offset_a = gr.Textbox("", label="Cam A offset (s)", placeholder="auto",
                                                    max_lines=1)
                            c.off_err_a = _html(R.inline_error(R.OFFSET_ERROR), visible=False)
                        with gr.Column(min_width=160) as c.off_b_col:
                            c.offset_b = gr.Textbox("", label="Cam B offset (s)", placeholder="auto",
                                                    max_lines=1)
                            c.off_err_b = _html(R.inline_error(R.OFFSET_ERROR), visible=False)
                        with gr.Column(min_width=160):
                            c.offset_wide = gr.Textbox("", label="Wide offset (s)", placeholder="auto",
                                                       max_lines=1)
                            c.off_err_w = _html(R.inline_error(R.OFFSET_ERROR), visible=False)
                    with gr.Column(visible=False) as c.framing_grp:
                        _html(R.group_label("Single-clip framing"))
                        with gr.Row(elem_classes=["pb-grp"]):
                            c.crop_scale = gr.Slider(2.4, 4.5, DEFAULTS["crop_scale"], step=0.1,
                                                     label="Close-up size (x face height)",
                                                     info="Smaller = tighter close-up.")
                            c.max_zoom = gr.Slider(1.0, 5.0, DEFAULTS["max_zoom"], step=0.1, label="Max zoom",
                                                   info="Upscaling limit; a 4K wide shot gives sharp "
                                                        "1080p close-ups.")
                with gr.Row(elem_id="pb-actions"):
                    c.run_btn = gr.Button(RUN_LABEL, variant="primary", interactive=False, elem_id="pb-run")
                    c.plan_btn = gr.Button(PLAN_LABEL, variant="secondary", interactive=False,
                                           elem_id="pb-plan")
                    c.run_help = _html(R.helper_html("Add your video to start."), elem_id="pb-run-help")
                _html(R.helper_html("Preview cut plan: plans cuts and captions without rendering "
                                    "video – fast.", "pb-small"), elem_id="pb-plan-help")

            # -------------------------------------------------------------- progress / error
            c.progress = _html(visible=False, elem_id="pb-progress")
            c.error = _html(visible=False, elem_id="pb-error")

            # -------------------------------------------------------------- result
            with gr.Column(visible=False, elem_id="pb-result") as c.result:
                with gr.Row(elem_id="pb-summary", equal_height=False):
                    c.chips = _html(elem_id="pb-chips", scale=1, min_width=0)
                    c.again_btn = gr.Button("Edit another", variant="secondary", size="md",
                                            elem_id="pb-again", scale=0, min_width=0)
                c.video = gr.Video(label="Edited video", show_label=False, interactive=False,
                                   elem_id="pb-video")
                c.plan_note = _html(R.plan_note_html(), visible=False, elem_id="pb-plan-note")
                c.timeline = _html(elem_id="pb-timeline")
                with gr.Row(elem_id="pb-downloads"):
                    c.dl_video = gr.DownloadButton("Video (.mp4)", visible=False, elem_classes=["pb-dl"])
                    c.dl_srt = gr.DownloadButton("Captions (.srt)", visible=False, elem_classes=["pb-dl"])
                    c.dl_csv = gr.DownloadButton("Cut list (.csv)", visible=False, elem_classes=["pb-dl"])
                    c.dl_json = gr.DownloadButton("Transcript (.json)", visible=False,
                                                  elem_classes=["pb-dl"])
                with gr.Column(visible=False, elem_id="pb-warn") as c.warn_col:
                    c.warn_html = _html()
                    c.swap_btn = gr.Button("Swap A and B, re-run", variant="secondary", visible=False,
                                           elem_id="pb-swap-rerun")
                with gr.Column(visible=False, elem_id="pb-framing") as c.framing_col:
                    _html(R.group_label("Framing"))
                    c.crop_img = gr.Image(label="Crop preview", show_label=False, interactive=False,
                                          type="filepath", elem_id="pb-crop", buttons=["fullscreen"],
                                          alt_text="Wide frame with the Speaker A and Speaker B close-up crops")
                    c.framing_caption = _html()
                with gr.Tabs(elem_id="pb-tabs") as c.tabs:
                    with gr.Tab("Transcript", id="transcript"):
                        c.transcript = _html(elem_id="pb-transcript")
                    with gr.Tab("Details", id="details"):
                        c.details = _html(elem_id="pb-details")
                        c.edl = gr.Dataframe(headers=EDL_COLS, label="Edit decision list", interactive=False,
                                             wrap=True, elem_id="pb-edl")
                        c.files_out = gr.File(label="All output files", file_count="multiple",
                                              interactive=False, elem_id="pb-files")
                        c.log_box = gr.Textbox(label="Log", lines=8, max_lines=8, interactive=False,
                                               elem_id="pb-log")
        c.last_plan = gr.State(False)

        settings = [getattr(c, k) for k in SETTING_KEYS]
        run_inputs = [c.drop, c.role_a, c.role_b, c.role_wide, *settings]
        offsets = [c.offset_a, c.offset_b, c.offset_wide]
        # small UI-only handlers: no spinner, no queue, not part of the documented API
        quiet: Dict[str, Any] = dict(show_progress="hidden", queue=False, api_visibility="undocumented")

        # ------------------------------------------------------------------ form gating
        def gate(paths: List[str], roles: Sequence[Optional[str]], offs: Sequence[Any]) -> Dict:
            chk = form_check(paths, roles, offs)
            return {c.run_btn: gr.update(interactive=chk["ready"]),
                    c.plan_btn: gr.update(interactive=chk["ready"]),
                    c.run_help: R.helper_html(chk["helper"]),
                    c.role_err: gr.update(visible=chk["dup_roles"]),
                    c.off_err_a: gr.update(visible=chk["bad_offsets"][0]),
                    c.off_err_b: gr.update(visible=chk["bad_offsets"][1]),
                    c.off_err_w: gr.update(visible=chk["bad_offsets"][2])}

        gate_outputs = [c.run_btn, c.plan_btn, c.run_help, c.role_err, c.off_err_a, c.off_err_b, c.off_err_w]

        def on_files(files, oa, ob, ow):
            paths = _paths(files)
            n = len(paths)
            roles: Tuple[Optional[str], ...] = (None, None, None)
            if n == 3:
                roles = R.guess_roles(paths)
            choices = R.role_choices(paths) if n == 3 else []
            thumb = _thumbnail(paths[0]) if n == 1 else None
            out = {
                c.detect: gr.update(value=R.detect_html(n), visible=n in (1, 3)),
                c.count_err: gr.update(value=R.count_error_html(n), visible=n not in (0, 1, 3)),
                c.roles_row: gr.update(visible=n == 3),
                c.role_a: gr.update(choices=choices, value=roles[0]),
                c.role_b: gr.update(choices=choices, value=roles[1]),
                c.role_wide: gr.update(choices=choices, value=roles[2]),
                c.single_row: gr.update(visible=n == 1),
                c.thumb: gr.update(value=thumb, visible=thumb is not None),
                c.framing_grp: gr.update(visible=n == 1),
                c.off_a_col: gr.update(visible=n != 1),
                c.off_b_col: gr.update(visible=n != 1),
                # a new set of files makes any shown result stale
                c.result: gr.update(visible=False),
                c.progress: gr.update(visible=False),
                c.error: gr.update(visible=False),
            }
            out.update(gate(paths, roles, (oa, ob, ow)))
            return out

        c.drop.change(on_files, [c.drop, *offsets],
                      [c.detect, c.count_err, c.roles_row, c.role_a, c.role_b, c.role_wide, c.single_row,
                       c.thumb, c.framing_grp, c.off_a_col, c.off_b_col, c.result, c.progress, c.error,
                       *gate_outputs], **quiet)

        # Gradio 6.29.1 drops a visible=False update to a shown component inside an accordion
        # that has never been opened (the cam A/B offsets stayed visible in single-clip mode),
        # so re-apply the mode-dependent groups whenever the drawer opens.
        def on_adv_open(files):
            single = len(_paths(files)) == 1
            return gr.update(visible=not single), gr.update(visible=not single), gr.update(visible=single)

        c.adv.expand(on_adv_open, c.drop, [c.off_a_col, c.off_b_col, c.framing_grp], **quiet)

        def on_form(files, ra, rb, rw, oa, ob, ow):
            return gate(_paths(files), (ra, rb, rw), (oa, ob, ow))

        form_inputs = [c.drop, c.role_a, c.role_b, c.role_wide, *offsets]
        for comp in (c.role_a, c.role_b, c.role_wide):  # .input: on_files sets them programmatically
            comp.input(on_form, form_inputs, gate_outputs, **quiet)
        for comp in offsets:  # only the user edits these (Textbox .input does not reach us in 6.29.1)
            comp.change(on_form, form_inputs, gate_outputs, **quiet)

        # ------------------------------------------------------------------ live widgets
        for comp in (c.color_a, c.color_b):
            comp.change(lambda a, b: R.caption_preview_html(_hex(a, "#2D9CFF"), _hex(b, "#FFFFFF")),
                        [c.color_a, c.color_b], c.cap_preview, **quiet)

        cut_sliders = [getattr(c, k) for k in CUT_KEYS]

        def on_cut_style(style):
            p = CUT_PRESETS.get(style)
            if p is None:  # "Custom": keep whatever the sliders say
                return [gr.skip()] * len(CUT_KEYS)
            return [p[k] for k in CUT_KEYS]

        c.cut_style.change(on_cut_style, c.cut_style, cut_sliders, **quiet)
        for comp in cut_sliders:  # .input = the user moved it (preset updates fire .change only)
            comp.input(lambda: CUSTOM, None, c.cut_style, **quiet)

        def on_quality(q):
            preset = QUALITY.get(q)
            if preset is None:
                return gr.update(visible=True), gr.skip(), gr.skip(), gr.skip()
            return gr.update(visible=False), *preset

        c.quality.change(on_quality, c.quality, [c.custom_row, c.width, c.height, c.fps], **quiet)

        # ------------------------------------------------------------------ running
        result_outputs = [c.last_plan, c.run_btn, c.plan_btn, c.progress, c.error, c.result, c.chips,
                          c.video, c.plan_note, c.timeline, c.dl_video, c.dl_srt, c.dl_csv, c.dl_json,
                          c.warn_col, c.warn_html, c.swap_btn, c.framing_col, c.crop_img,
                          c.framing_caption, c.tabs, c.transcript, c.details, c.edl, c.files_out,
                          c.log_box]

        def result_updates(res, cfg: PipelineConfig, handler: _ListHandler, elapsed: float,
                           plan_only: bool) -> Dict:
            ca, cb = cfg.captions.color_a, cfg.captions.color_b
            rows = [[(",".join(str(r[k]) for k in ("crop_x", "crop_y", "crop_w", "crop_h"))
                      if col == "crop" and r.get("crop_w") not in ("", None) else r.get(col, ""))
                     for col in EDL_COLS] for r in res.edl]
            files = [p for p in res.outputs.values() if p and os.path.isfile(p)]
            warnings = list(dict.fromkeys(handler.warnings))
            confident = bool(res.mapping.get("confident"))
            preview = res.outputs.get("crop_preview")
            has_preview = cfg.mode == "single" and bool(preview) and os.path.isfile(preview)

            def dl(key: str):
                p = res.outputs.get(key)
                ok = bool(p) and os.path.isfile(p) and os.path.getsize(p) > 0
                return gr.update(value=p if ok else None, visible=ok)

            return {
                c.progress: R.progress_html(R.DONE, 1.0, "done", elapsed, plan_only, finished=True),
                c.result: gr.update(visible=True),
                c.chips: R.summary_chips_html(res.stats, res.mapping, elapsed),
                c.video: gr.update(value=res.output_video, visible=not plan_only and bool(res.output_video)),
                c.plan_note: gr.update(visible=plan_only),
                c.timeline: R.timeline_html(res.edl, res.stats.get("camera_seconds", {}), ca, cb),
                c.dl_video: dl("video"), c.dl_srt: dl("srt"), c.dl_csv: dl("edl_csv"),
                c.dl_json: dl("transcript"),
                c.warn_col: gr.update(visible=bool(warnings) or not confident),
                c.warn_html: R.warnings_html(warnings, confident),
                c.swap_btn: gr.update(visible=not confident),
                c.framing_col: gr.update(visible=has_preview),
                c.crop_img: gr.update(value=preview if has_preview else None),
                c.framing_caption: R.framing_caption_html(cfg.reframe.a_side),
                c.tabs: gr.Tabs(selected="transcript"),
                c.transcript: R.transcript_html(res.transcript_text, ca, cb),
                c.details: R.details_html(res.offsets, res.mapping, res.diarization_backend, res.stats),
                c.edl: rows,
                c.files_out: files,
                c.log_box: "\n".join(handler.lines[-300:]),
            }

        def run(plan_only: bool, files, role_a, role_b, role_wide, *values):
            t0 = time.time()
            plan_only = bool(plan_only)
            yield {c.last_plan: plan_only,
                   c.run_btn: gr.update(value=BUSY_LABEL, interactive=False),
                   c.plan_btn: gr.update(interactive=False),
                   c.result: gr.update(visible=False),
                   c.error: gr.update(visible=False),
                   c.progress: gr.update(visible=True,
                                         value=R.progress_html(0, 0.0, "starting", 0.0, plan_only))}
            buttons = {c.run_btn: gr.update(value=RUN_LABEL, interactive=True),
                       c.plan_btn: gr.update(interactive=True)}
            handler = _ListHandler()

            def failed(e: BaseException, known: bool) -> Dict:
                return {**buttons, c.progress: gr.update(visible=False),
                        c.error: gr.update(visible=True, value=R.error_html(
                            type(e).__name__, str(e), known, handler.lines))}

            run_dir = os.path.join(RUNS_DIR, time.strftime("%Y%m%d-%H%M%S"))
            try:
                cfg = build_config(files, (role_a, role_b, role_wide), plan_only, run_dir,
                                   **dict(zip(SETTING_KEYS, values)))
                os.makedirs(run_dir, exist_ok=True)
            except (ValueError, OSError) as e:
                yield failed(e, True)
                return
            events: "queue.Queue[Tuple[float, str]]" = queue.Queue()
            box: Dict[str, Any] = {}
            root = logging.getLogger()

            def work() -> None:
                if not _PIPELINE_LOCK.acquire(blocking=False):
                    events.put((0.0, WAIT_MSG))
                    _PIPELINE_LOCK.acquire()
                # attached here, not in the generator: a page reload closes the generator while
                # this thread runs on, and the next run's handler must not collect our log lines
                root.addHandler(handler)
                try:
                    box["res"] = run_pipeline(cfg, lambda f, msg: events.put((f, msg)))
                except Exception as e:  # noqa: BLE001
                    log.exception("pipeline failed")
                    box["err"] = e
                finally:
                    root.removeHandler(handler)
                    _PIPELINE_LOCK.release()

            th = threading.Thread(target=work, name="podbot-pipeline", daemon=True)
            th.start()
            stage, frac, msg, shown = 0, 0.0, "starting", ""
            while True:
                alive = th.is_alive()
                batch = []
                try:
                    batch.append(events.get(timeout=0.4))
                    while True:
                        batch.append(events.get_nowait())
                except queue.Empty:
                    pass
                for frac, msg in batch:
                    stage = R.stage_for(msg, stage)
                page = R.progress_html(stage, frac, msg, time.time() - t0, plan_only)
                if page != shown:
                    shown = page
                    yield {c.progress: page}
                if not alive and events.empty():
                    break
            th.join()
            if "err" in box:
                yield failed(box["err"], isinstance(box["err"], PipelineError))
                return
            yield {**buttons, **result_updates(box["res"], cfg, handler, time.time() - t0, plan_only)}

        def run_edit(*v):
            yield from run(False, *v)

        def run_plan(*v):
            yield from run(True, *v)

        def run_again(plan_only, *v):
            yield from run(plan_only, *v)

        # one pipeline job at a time across all run buttons; we draw our own progress
        common: Dict[str, Any] = dict(outputs=result_outputs, show_progress="hidden",
                                      concurrency_id="pipeline", concurrency_limit=1)
        c.run_btn.click(run_edit, run_inputs, js=_FREEZE, api_name="edit", **common
                        ).then(None, js=_UNFREEZE)
        c.plan_btn.click(run_plan, run_inputs, js=_FREEZE, api_name="plan", **common
                         ).then(None, js=_UNFREEZE)
        # "Swap A and B, re-run": flip the Advanced checkbox, then repeat the last run type
        c.swap_btn.click(lambda s: not s, c.swap, c.swap, js=_FREEZE, **quiet
                         ).then(run_again, [c.last_plan, *run_inputs], api_visibility="undocumented",
                                **common
                                ).then(None, js=_UNFREEZE)
        c.again_btn.click(lambda: (None, gr.update(visible=False), gr.update(visible=False),
                                   gr.update(visible=False)),
                          None, [c.drop, c.result, c.progress, c.error], **quiet)
    return demo


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--port", type=int, default=7860)
    args = ap.parse_args()
    setup_logging(False)
    os.makedirs(RUNS_DIR, exist_ok=True)
    build_ui().queue().launch(server_name="127.0.0.1", server_port=args.port, share=False,
                              allowed_paths=[RUNS_DIR], **launch_kwargs())


if __name__ == "__main__":
    main()
