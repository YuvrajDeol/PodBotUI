"""Command-line interface.

Multicam:     ``python -m podcast_autoeditor --cam-a A.mp4 --cam-b B.mp4 --wide W.mp4 --out out.mp4``
Single clip:  ``python -m podcast_autoeditor --wide W.mp4 --out out.mp4`` (close-ups are cropped
              from the wide shot)."""

from __future__ import annotations

import argparse
import json
import logging
import sys
from typing import List, Optional

from . import __version__
from .config import (AsrConfig, CaptionConfig, DiarizationConfig, LabelConfig, MappingConfig,
                     PipelineConfig, PlannerConfig, ReframeConfig, RenderConfig, SyncConfig)


def build_parser() -> argparse.ArgumentParser:
    P, S, A, D, M, L, C, R = (PlannerConfig(), SyncConfig(), AsrConfig(), DiarizationConfig(),
                              MappingConfig(), LabelConfig(), CaptionConfig(), RenderConfig())
    p = argparse.ArgumentParser(
        prog="python -m podcast_autoeditor",
        description="AI multicam podcast auto-editor: transcribes, identifies speakers A/B and "
                    "cuts between cam A, cam B and the wide shot with colour-coded captions. "
                    "With only --wide it auto-reframes that single clip into A / B / wide.",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )
    p.add_argument("--version", action="version", version=__version__)
    g = p.add_argument_group("inputs / outputs")
    g.add_argument("--cam-a", default=None, help="close-up of Speaker A (multicam mode)")
    g.add_argument("--cam-b", default=None, help="close-up of Speaker B (multicam mode)")
    g.add_argument("--wide", required=True, help="wide / full shot of both (the only clip needed "
                                                 "in single-clip mode)")
    g.add_argument("--mode", default="auto", choices=["auto", "multicam", "single"],
                   help="auto: single-clip mode when --cam-a/--cam-b are not given")
    g.add_argument("--audio", default=None, help="optional external master audio (timeline reference)")
    g.add_argument("--out", required=True, help="output .mp4")
    g.add_argument("--work-dir", default=None, help="cache dir (default: <out>_work next to output)")
    g.add_argument("--plan-only", action="store_true", help="skip rendering (EDL/captions only)")
    g.add_argument("--force", action="store_true", help="ignore cached intermediate results")
    g.add_argument("--keep-intermediates", action="store_true", help="keep per-shot segment files")
    g.add_argument("--ffmpeg", default=None, help="path to ffmpeg (ffprobe must be next to it)")
    g.add_argument("-v", "--verbose", action="store_true")
    g.add_argument("--log-file", default=None)

    g = p.add_argument_group("sync")
    g.add_argument("--no-sync", action="store_true", help="assume all clips start together")
    g.add_argument("--sync-window", type=float, default=S.max_lag_s, help="max |offset| searched (s)")
    g.add_argument("--sync-analysis", type=float, default=S.analysis_s, help="seconds of audio correlated")
    g.add_argument("--offset-a", type=float, default=None, help="manual offset of cam A (s; clip t=0 on timeline)")
    g.add_argument("--offset-b", type=float, default=None, help="manual offset of cam B")
    g.add_argument("--offset-wide", type=float, default=None, help="manual offset of wide")
    g.add_argument("--sync-min-confidence", type=float, default=S.min_confidence,
                   help="offsets with a lower peak-to-sidelobe ratio are not applied (0.0 is used)")
    g.add_argument("--force-sync", action="store_true",
                   help="apply automatic offsets even when their confidence is low")

    g = p.add_argument_group("speech recognition")
    g.add_argument("--whisper-model", default=A.model_size,
                   help="faster-whisper size: tiny, base, small, medium, large-v3, ...")
    g.add_argument("--language", default=None, help="language code (default: auto-detect)")
    g.add_argument("--compute-type", default=A.compute_type)
    g.add_argument("--device", default=A.device)
    g.add_argument("--beam-size", type=int, default=A.beam_size)
    g.add_argument("--no-vad", action="store_true", help="disable whisper VAD filter")
    g.add_argument("--asr-prompt", default=A.initial_prompt,
                   help="punctuated primer prompt for Whisper ('' to disable)")
    g.add_argument("--no-condition-previous", action="store_true",
                   help="don't condition Whisper on previous text (less punctuation, fewer loops)")
    g.add_argument("--no-word-refine", action="store_true",
                   help="don't trim silence that Whisper absorbed into word timestamps")

    g = p.add_argument_group("speaker identification")
    g.add_argument("--diarization", default=D.backend, choices=["auto", "ecapa", "mfcc", "pyannote"])
    g.add_argument("--num-speakers", default=D.num_speakers, choices=["auto", "1", "2"],
                   help="auto: detect whether one or two people talk (1 = everything to one speaker)")
    g.add_argument("--diar-window", type=float, default=D.window_s)
    g.add_argument("--diar-hop", type=float, default=D.hop_s)
    g.add_argument("--diar-median", type=int, default=D.median_filter)
    g.add_argument("--no-phrase-refine", action="store_true",
                   help="skip phrase-level re-scoring of speakers (window clustering only)")
    g.add_argument("--ref-a", default=None, help="enrollment audio of Speaker A (optional)")
    g.add_argument("--ref-b", default=None, help="enrollment audio of Speaker B (optional)")
    g.add_argument("--mapping-strategy", default=M.strategy, choices=["auto", "mic", "visual", "first"])
    g.add_argument("--speaker-map", default=M.speaker_map, choices=["auto", "AB", "BA"],
                   help="AB: first speaker heard is A; BA: first speaker heard is B")
    g.add_argument("--swap-speakers", action="store_true", help="flip the A/B decision")
    g.add_argument("--mic-margin-db", type=float, default=M.mic_margin_db,
                   help="min per-camera mic score to decide the A/B mapping")
    g.add_argument("--visual-margin", type=float, default=M.visual_margin,
                   help="min mouth-motion score to decide the A/B mapping")
    g.add_argument("--visual-timeout", type=float, default=M.visual_timeout_s,
                   help="wall-clock budget (s) for the mouth-motion analysis")
    g.add_argument("--turn-gap", type=float, default=L.turn_gap)

    g = p.add_argument_group("editing rules")
    g.add_argument("--lead", type=float, default=P.lead, help="cut this early before speech")
    g.add_argument("--min-shot", type=float, default=P.min_shot)
    g.add_argument("--qa-gap", type=float, default=P.qa_gap)
    g.add_argument("--qa-answer-hold", type=float, default=P.qa_answer_hold)
    g.add_argument("--qa-max-question", type=float, default=P.qa_max_question)
    g.add_argument("--question-heuristic", default="on" if P.question_heuristic else "off",
                   choices=["on", "off"],
                   help="also treat sentences without '?' as questions when they open like one "
                        "('what do you', 'do you', 'is it') or end in a tag (', right', 'isn't it')")
    g.add_argument("--qa-question-words", action="store_true", help=argparse.SUPPRESS)  # old alias
    g.add_argument("--no-qa", action="store_true", help="disable question->answer wide shots")
    g.add_argument("--rapid-turns", type=int, default=P.rapid_turns)
    g.add_argument("--rapid-window", type=float, default=P.rapid_window)
    g.add_argument("--no-rapid", action="store_true")
    g.add_argument("--no-overlap-wide", action="store_true")
    g.add_argument("--silence-wide", type=float, default=P.silence_wide)

    g = p.add_argument_group("captions")
    g.add_argument("--color-a", default=C.color_a)
    g.add_argument("--color-b", default=C.color_b)
    g.add_argument("--font", default=C.font)
    g.add_argument("--font-size-pct", type=float, default=C.font_size_pct)
    g.add_argument("--max-words", type=int, default=C.max_words)
    g.add_argument("--max-chars", type=int, default=C.max_chars)
    g.add_argument("--max-line-dur", type=float, default=C.max_dur)
    g.add_argument("--no-captions", action="store_true")
    g.add_argument("--caption-burner", default=C.burner, choices=["auto", "ass", "pillow"])

    g = p.add_argument_group("render")
    g.add_argument("--width", type=int, default=R.width)
    g.add_argument("--height", type=int, default=R.height)
    g.add_argument("--fps", type=float, default=R.fps)
    g.add_argument("--encoder", default=R.encoder, choices=["auto", "libx264", "h264_videotoolbox"])
    g.add_argument("--crf", type=int, default=R.crf)
    g.add_argument("--preset", default=R.preset)
    g.add_argument("--audio-source", default=R.audio_source,
                   choices=["auto", "external", "wide", "cam-a", "cam-b"])
    g.add_argument("--jobs", type=int, default=R.jobs, help="parallel segment encodes (0=auto)")

    F = ReframeConfig()
    g = p.add_argument_group("single-clip mode (virtual close-ups cropped from --wide)")
    g.add_argument("--a-side", default=F.a_side, choices=["left", "right"],
                   help="which person in the wide frame is Speaker A")
    g.add_argument("--crop-scale", type=float, default=F.crop_scale,
                   help="close-up height as a multiple of the face height (3-3.5 = head and shoulders)")
    g.add_argument("--max-zoom", type=float, default=F.max_zoom,
                   help="never upscale a crop more than this (crop height >= output height / max-zoom)")
    g.add_argument("--crop-a", default=None, metavar="X,Y,W,H",
                   help="manual crop for Speaker A in source pixels (overrides face framing)")
    g.add_argument("--crop-b", default=None, metavar="X,Y,W,H", help="manual crop for Speaker B")
    g.add_argument("--static-crops", action="store_true",
                   help="never re-frame at shot boundaries, even if a person moves")
    g.add_argument("--preview-crops", default=None, metavar="PNG",
                   help="also write the crop preview (wide frame with A/B rectangles) to this path; "
                        "<out>_crop_preview.png is always written in single-clip mode")
    return p


def config_from_args(a: argparse.Namespace) -> PipelineConfig:
    return PipelineConfig(
        mode=a.mode,
        reframe=ReframeConfig(a_side=a.a_side, crop_scale=a.crop_scale, max_zoom=a.max_zoom,
                              crop_a=a.crop_a, crop_b=a.crop_b, per_shot=not a.static_crops,
                              preview=a.preview_crops),
        cam_a=a.cam_a or "", cam_b=a.cam_b or "", wide=a.wide, audio=a.audio, out=a.out,
        work_dir=a.work_dir, plan_only=a.plan_only, force=a.force, ffmpeg=a.ffmpeg,
        sync=SyncConfig(enabled=not a.no_sync, max_lag_s=a.sync_window, analysis_s=a.sync_analysis,
                        offset_a=a.offset_a, offset_b=a.offset_b, offset_wide=a.offset_wide,
                        min_confidence=a.sync_min_confidence, force=a.force_sync),
        asr=AsrConfig(model_size=a.whisper_model, language=a.language, device=a.device,
                      compute_type=a.compute_type, beam_size=a.beam_size, vad_filter=not a.no_vad,
                      initial_prompt=a.asr_prompt or None,
                      condition_on_previous_text=not a.no_condition_previous,
                      refine_word_times=not a.no_word_refine),
        diar=DiarizationConfig(backend=a.diarization, window_s=a.diar_window, hop_s=a.diar_hop,
                               median_filter=a.diar_median, ref_a=a.ref_a, ref_b=a.ref_b,
                               word_refine=not a.no_phrase_refine, num_speakers=a.num_speakers),
        mapping=MappingConfig(strategy=a.mapping_strategy, speaker_map=a.speaker_map,
                              swap=a.swap_speakers, mic_margin_db=a.mic_margin_db,
                              visual_margin=a.visual_margin, visual_timeout_s=a.visual_timeout),
        label=LabelConfig(turn_gap=a.turn_gap),
        planner=PlannerConfig(lead=a.lead, min_shot=a.min_shot, qa_enabled=not a.no_qa,
                              qa_gap=a.qa_gap, qa_answer_hold=a.qa_answer_hold,
                              qa_max_question=a.qa_max_question,
                              question_heuristic=a.question_heuristic == "on" or a.qa_question_words,
                              rapid_enabled=not a.no_rapid, rapid_turns=a.rapid_turns,
                              rapid_window=a.rapid_window, overlap_wide=not a.no_overlap_wide,
                              silence_wide=a.silence_wide),
        captions=CaptionConfig(enabled=not a.no_captions, color_a=a.color_a, color_b=a.color_b,
                               font=a.font, font_size_pct=a.font_size_pct, max_words=a.max_words,
                               max_chars=a.max_chars, max_dur=a.max_line_dur, burner=a.caption_burner),
        render=RenderConfig(width=a.width, height=a.height, fps=a.fps, encoder=a.encoder, crf=a.crf,
                            preset=a.preset, audio_source=a.audio_source, jobs=a.jobs,
                            keep_intermediates=a.keep_intermediates),
    )


def setup_logging(verbose: bool = False, log_file: Optional[str] = None) -> None:
    handlers: List[logging.Handler] = [logging.StreamHandler(sys.stderr)]
    if log_file:
        handlers.append(logging.FileHandler(log_file))
    logging.basicConfig(level=logging.DEBUG if verbose else logging.INFO,
                        format="%(asctime)s %(levelname)-7s %(name)s: %(message)s",
                        datefmt="%H:%M:%S", handlers=handlers, force=True)
    for noisy in ("speechbrain", "faster_whisper", "httpx", "urllib3", "huggingface_hub", "numba"):
        logging.getLogger(noisy).setLevel(logging.WARNING)


def main(argv: Optional[List[str]] = None) -> int:
    args = build_parser().parse_args(argv)
    setup_logging(args.verbose, args.log_file)
    log = logging.getLogger("podcast_autoeditor")
    from .captions import parse_color

    try:
        parse_color(args.color_a)
        parse_color(args.color_b)
    except ValueError as e:
        log.error("%s", e)
        return 2
    from .pipeline import PipelineError, run_pipeline
    from .ffmpeg_utils import FFmpegError

    cfg = config_from_args(args)
    last = [-1.0]

    def progress(frac: float, msg: str) -> None:
        if frac - last[0] >= 0.05 or frac >= 1.0:
            last[0] = frac
            log.info("[%3d%%] %s", int(frac * 100), msg)

    try:
        res = run_pipeline(cfg, progress)
    except (PipelineError, FFmpegError, FileNotFoundError, ValueError) as e:
        log.error("%s", e)
        return 1
    print(json.dumps({"outputs": res.outputs, "stats": res.stats, "mapping": res.mapping,
                      "offsets": {k: round(v["offset"], 3) for k, v in res.offsets.items()}},
                     indent=1))
    return 0


if __name__ == "__main__":  # pragma: no cover
    sys.exit(main())
