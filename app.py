#!/usr/bin/env python3
"""Gradio web UI for the podcast auto-editor (local only: binds to 127.0.0.1).

Run:  .venv/bin/python app.py   ->  http://127.0.0.1:7860
"""

from __future__ import annotations

import argparse
import logging
import os
import sys
import time
from typing import Any, List, Optional

ROOT = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, ROOT)

import gradio as gr  # noqa: E402

from podcast_autoeditor.captions import parse_color  # noqa: E402
from podcast_autoeditor.cli import setup_logging  # noqa: E402
from podcast_autoeditor.config import (AsrConfig, CaptionConfig, DiarizationConfig,  # noqa: E402
                                       MappingConfig, PipelineConfig, PlannerConfig, ReframeConfig,
                                       RenderConfig, SyncConfig)
from podcast_autoeditor.pipeline import run_pipeline  # noqa: E402

RUNS_DIR = os.environ.get("PODCAST_AUTOEDITOR_RUNS", os.path.join(ROOT, "ui_runs"))
WHISPER_MODELS = ["tiny", "base", "small", "medium", "large-v3", "distil-large-v3"]
EDL_COLS = ["index", "camera", "reason", "start", "end", "duration", "source", "source_in", "source_out",
            "crop"]


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
        raise gr.Error(f"not a number: {v!r}")


MODE_MULTI = "3 clips (multicam)"
MODE_SINGLE = "1 wide clip (auto-reframe)"


def run_edit(cam_a, cam_b, wide, master, color_a, color_b, model, language, swap, plan_only,
             qa_gap, qa_hold, rapid_turns, rapid_window, min_shot, silence_wide, width, height, fps,
             num_speakers="auto", question_heuristic=True, offset_a="", offset_b="", offset_wide="",
             mode=MODE_MULTI, a_side="left", crop_scale=3.2, max_zoom=3.0,
             progress=gr.Progress()):
    single = str(mode) == MODE_SINGLE or str(mode).lower() == "single"
    a, b, w = _path(cam_a), _path(cam_b), _path(wide)
    if single:
        if not w:
            raise gr.Error("Please upload the wide shot.")
        a = b = ""
    elif not (a and b and w):
        raise gr.Error("Please upload cam A, cam B and the wide shot (or switch to the "
                       "'1 wide clip' mode).")
    run_dir = os.path.join(RUNS_DIR, time.strftime("%Y%m%d-%H%M%S"))
    os.makedirs(run_dir, exist_ok=True)
    key_clip = w if single else a
    cfg = PipelineConfig(
        mode="single" if single else "multicam",
        reframe=ReframeConfig(a_side=str(a_side or "left"), crop_scale=float(crop_scale),
                              max_zoom=float(max_zoom)),
        cam_a=a, cam_b=b, wide=w, audio=_path(master), out=os.path.join(run_dir, "edited.mp4"),
        work_dir=os.path.join(RUNS_DIR, "_cache",
                              f"{'single-' if single else ''}{os.path.basename(key_clip)}-"
                              f"{os.path.getsize(key_clip)}"),
        plan_only=bool(plan_only),
        asr=AsrConfig(model_size=model, language=(language or "").strip() or None),
        sync=SyncConfig(offset_a=_opt_float(offset_a), offset_b=_opt_float(offset_b),
                        offset_wide=_opt_float(offset_wide)),
        diar=DiarizationConfig(num_speakers=str(num_speakers or "auto")),
        mapping=MappingConfig(swap=bool(swap)),
        planner=PlannerConfig(qa_gap=float(qa_gap), qa_answer_hold=float(qa_hold),
                              rapid_turns=int(rapid_turns), rapid_window=float(rapid_window),
                              min_shot=float(min_shot), silence_wide=float(silence_wide),
                              question_heuristic=bool(question_heuristic)),
        captions=CaptionConfig(color_a=_hex(color_a, "#2D9CFF"), color_b=_hex(color_b, "#FFFFFF")),
        render=RenderConfig(width=int(width), height=int(height), fps=float(fps)),
    )
    handler = _ListHandler()
    logging.getLogger().addHandler(handler)
    try:
        res = run_pipeline(cfg, lambda f, msg: progress(f, desc=msg))
    except Exception as e:  # noqa: BLE001
        logging.getLogger("app").exception("pipeline failed")
        raise gr.Error(f"{type(e).__name__}: {e}")
    finally:
        logging.getLogger().removeHandler(handler)
    rows = [[(",".join(str(r[k]) for k in ("crop_x", "crop_y", "crop_w", "crop_h"))
              if c == "crop" and r.get("crop_w") not in ("", None) else r.get(c, ""))
             for c in EDL_COLS] for r in res.edl]
    files = [p for k, p in res.outputs.items() if p and os.path.isfile(p)]
    m = res.mapping
    warnings = list(dict.fromkeys(handler.warnings))  # unique, in order
    warn_md = ("### ⚠️ Warnings\n" + "\n".join(f"- {w}" for w in warnings[:12]) + "\n\n") if warnings else ""
    n_spk = res.stats.get("num_speakers", 2)
    spk_md = (f"**Speakers:** {n_spk}"
              + (f" ({'; '.join(res.stats.get('speaker_count_reasons', []))})" if n_spk == 1 else "")
              + "\n\n")
    rf = res.stats.get("reframe")
    if rf:
        other = "right" if rf["a_side"] == "left" else "left"
        spk_md += (f"**Mode:** 1 wide clip — Speaker A = person on the **{rf['a_side']}**, Speaker B = "
                   f"person on the **{other}**; close-up crops (x,y,w,h): A {rf['crops'].get('A')}, "
                   f"B {rf['crops'].get('B')}; faces found: {rf['faces_found']}\n\n")
    summary = warn_md + spk_md + (
        f"**Shots:** {res.stats['n_shots']}  ·  **seconds per camera:** {res.stats['camera_seconds']}  ·  "
        f"**events:** {res.stats.get('events', {})}\n\n"
        f"**Speaker→camera mapping:** {m['mapping']} via `{m['method']}`"
        f"{'' if m['confident'] else '  ⚠️ not confident — tick *Swap speakers* if A/B are reversed'}\n\n"
        f"**Sync offsets (s):** " + ", ".join(f"{k} {v['offset']:+.3f}" for k, v in res.offsets.items())
        + f"\n\n**Diarization:** {res.diarization_backend}"
    )
    preview = res.outputs.get("crop_preview")
    return (res.output_video, files, rows, res.transcript_text, summary, "\n".join(handler.lines[-300:]),
            preview if preview and os.path.isfile(preview) else None)


def build_ui() -> gr.Blocks:
    with gr.Blocks(title="Podcast Auto-Editor") as demo:
        gr.Markdown("# 🎙️ Multicam Podcast Auto-Editor\nSpeaker A → cam A (blue captions), "
                    "Speaker B → cam B (white captions), question→answer exchanges and rapid "
                    "back-and-forth → wide shot. Upload the two close-ups and the wide shot — or "
                    "just ONE wide clip, and the close-ups are cropped out of it automatically.")
        mode = gr.Radio([MODE_MULTI, MODE_SINGLE], value=MODE_MULTI, label="Mode")
        with gr.Row():
            cam_a = gr.File(label="Cam A – close-up of Speaker A", file_types=["video"], type="filepath")
            cam_b = gr.File(label="Cam B – close-up of Speaker B", file_types=["video"], type="filepath")
            wide = gr.File(label="Wide – both speakers", file_types=["video"], type="filepath")
            master = gr.File(label="Master audio (optional)", file_types=["audio"], type="filepath")
        with gr.Row(visible=False) as single_opts:
            a_side = gr.Radio(["left", "right"], value="left", label="Speaker A is the person on the…",
                              info="Speaker A gets the blue captions")
            crop_scale = gr.Slider(2.4, 4.5, 3.2, step=0.1, label="Close-up size (× face height)",
                                   info="smaller = tighter close-up")
            max_zoom = gr.Slider(1.0, 5.0, 3.0, step=0.1, label="Max zoom (upscaling limit)",
                                 info="use a 4K wide shot for sharp 1080p close-ups")
        with gr.Row():
            color_a = gr.ColorPicker(value="#2D9CFF", label="Speaker A caption colour")
            color_b = gr.ColorPicker(value="#FFFFFF", label="Speaker B caption colour")
            model = gr.Dropdown(WHISPER_MODELS, value="small", label="Whisper model")
            language = gr.Textbox(value="", label="Language code (blank = auto)", max_lines=1)
            num_speakers = gr.Radio(["auto", "1", "2"], value="auto", label="Speakers",
                                    info="auto detects a one-person recording")
            swap = gr.Checkbox(value=False, label="Swap speakers (A↔B)")
            plan_only = gr.Checkbox(value=False, label="Plan only (no render)")
        with gr.Accordion("Editing rules", open=False):
            with gr.Row():
                qa_gap = gr.Slider(0.2, 5.0, 1.5, step=0.1, label="Q→A max gap (s)")
                qa_hold = gr.Slider(0.5, 6.0, 2.0, step=0.1, label="Wide hold into answer (s)")
                rapid_turns = gr.Slider(2, 8, 3, step=1, label="Rapid: speaker changes")
                rapid_window = gr.Slider(2.0, 20.0, 6.0, step=0.5, label="Rapid: window (s)")
                min_shot = gr.Slider(0.5, 5.0, 1.2, step=0.1, label="Min shot (s)")
                silence_wide = gr.Slider(1.0, 10.0, 2.5, step=0.5, label="Silence → wide (s)")
            with gr.Row():
                question_heuristic = gr.Checkbox(
                    value=True, label="Question heuristic (questions without '?')")
                offset_a = gr.Textbox("", label="Manual offset cam A (s, optional)", max_lines=1)
                offset_b = gr.Textbox("", label="Manual offset cam B (s, optional)", max_lines=1)
                offset_wide = gr.Textbox("", label="Manual offset wide (s, optional)", max_lines=1)
            with gr.Row():
                width = gr.Number(1920, label="Width", precision=0)
                height = gr.Number(1080, label="Height", precision=0)
                fps = gr.Number(30, label="FPS")
        run = gr.Button("Auto-edit", variant="primary")
        with gr.Row():
            video = gr.Video(label="Edited video")
            files = gr.File(label="Downloads (video, EDL, captions, transcript)", file_count="multiple")
        summary = gr.Markdown()
        preview = gr.Image(label="Crop preview – the wide frame with the Speaker A / B close-up crops "
                                 "(tick 'Plan only' to check the framing before a long render)",
                           visible=False, interactive=False)
        edl = gr.Dataframe(headers=EDL_COLS, label="Edit decision list", interactive=False, wrap=True)
        transcript = gr.Textbox(label="Transcript (speaker-labelled)", lines=12)
        log_box = gr.Textbox(label="Log", lines=8)
        run.click(run_edit,
                  inputs=[cam_a, cam_b, wide, master, color_a, color_b, model, language, swap, plan_only,
                          qa_gap, qa_hold, rapid_turns, rapid_window, min_shot, silence_wide, width,
                          height, fps, num_speakers, question_heuristic, offset_a, offset_b, offset_wide,
                          mode, a_side, crop_scale, max_zoom],
                  outputs=[video, files, edl, transcript, summary, log_box, preview])

        def _switch(m):
            one = m == MODE_SINGLE
            return (gr.update(visible=not one), gr.update(visible=not one), gr.update(visible=one),
                    gr.update(visible=one),
                    gr.update(label="The wide clip – both speakers in frame" if one else "Wide – both speakers"))

        mode.change(_switch, inputs=[mode], outputs=[cam_a, cam_b, single_opts, preview, wide])
    return demo


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--port", type=int, default=7860)
    args = ap.parse_args()
    setup_logging(False)
    os.makedirs(RUNS_DIR, exist_ok=True)
    build_ui().queue().launch(server_name="127.0.0.1", server_port=args.port, share=False,
                              allowed_paths=[RUNS_DIR])


if __name__ == "__main__":
    main()
