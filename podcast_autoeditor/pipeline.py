"""End-to-end orchestration with on-disk caching of intermediate results."""

from __future__ import annotations

import dataclasses
import hashlib
import json
import logging
import os
import time
from dataclasses import asdict, dataclass, field
from typing import Any, Callable, Dict, List, Optional

from . import CAM_A, CAM_B, WIDE, __version__
from .captions import (build_captions, transcript_text, write_ass, write_srt,
                       write_transcript_json)
from .config import PipelineConfig
from .datatypes import Interval, Shot, Turn, Word
from .diarize import DiarizationResult, diarize
from .ffmpeg_utils import FFmpegInfo, MediaInfo, extract_audio, find_ffmpeg, probe
from .labeling import (apply_mapping, build_turns, label_words, overlaps_from_mic_coupling,
                       overlaps_from_mics, overlaps_from_words, _merge)
from .mapping import MappingResult, decide_mapping
from .planner import PlanEvent, plan_shots, validate_shots
from .reframe import (CropRect, PersonTrack, base_crops, motion_series, parse_crop, scan_faces,
                      select_people, shot_crops, write_preview)
from .render import (AudioSource, SourceClip, edl_rows, normalize_fps, render_final,
                     render_segments, write_edl)
from .sync import SyncResult, compute_offsets, load_wav
from .transcribe import transcribe, words_from_transcript

log = logging.getLogger(__name__)

ProgressFn = Callable[[float, str], None]


class PipelineError(RuntimeError):
    pass


@dataclass
class PipelineResult:
    output_video: Optional[str]
    outputs: Dict[str, str]
    shots: List[Shot]
    turns: List[Turn]
    words: List[Word]
    offsets: Dict[str, Dict[str, Any]]
    mapping: Dict[str, Any]
    diarization_backend: str
    stats: Dict[str, Any] = field(default_factory=dict)
    edl: List[Dict[str, Any]] = field(default_factory=list)
    transcript_text: str = ""


def _fp(path: Optional[str]) -> str:
    if not path:
        return "none"
    st = os.stat(path)
    return f"{os.path.abspath(path)}|{st.st_size}|{st.st_mtime_ns}"


def _key(*parts: Any) -> str:
    return hashlib.sha1(json.dumps(parts, sort_keys=True, default=str).encode()).hexdigest()


def _cached(path: str, key: str, force: bool, compute: Callable[[], Any], what: str) -> Any:
    if not force and os.path.isfile(path):
        try:
            with open(path) as f:
                d = json.load(f)
            if d.get("_key") == key:
                log.info("Using cached %s (%s)", what, os.path.basename(path))
                return d["data"]
        except (json.JSONDecodeError, KeyError):
            pass
    data = compute()
    tmp = path + ".tmp"
    with open(tmp, "w") as f:
        json.dump({"_key": key, "_version": __version__, "data": data}, f, indent=1, default=str)
    os.replace(tmp, path)
    return data


def _cached_wav(ff: FFmpegInfo, src: str, dst: str, force: bool) -> str:
    stamp = dst + ".src"
    fp = _fp(src)
    if not force and os.path.isfile(dst) and os.path.isfile(stamp):
        with open(stamp) as f:
            if f.read() == fp:
                return dst
    extract_audio(ff, src, dst)
    with open(stamp, "w") as f:
        f.write(fp)
    return dst


def resolve_mode(cfg: PipelineConfig) -> str:
    """"multicam" (cam A + cam B + wide) or "single" (one wide clip, virtual close-ups).

    ``auto``: single when neither --cam-a nor --cam-b is given, multicam when both are."""
    mode = (cfg.mode or "auto").strip().lower()
    if mode not in ("auto", "multicam", "single"):
        raise PipelineError(f"--mode must be auto, multicam or single (got {cfg.mode!r})")
    if not cfg.wide:
        raise PipelineError("--wide is required")
    if mode == "auto":
        if cfg.cam_a and cfg.cam_b:
            return "multicam"
        if not cfg.cam_a and not cfg.cam_b:
            return "single"
        missing = "--cam-b" if cfg.cam_a else "--cam-a"
        raise PipelineError(f"{missing} is missing: give both --cam-a and --cam-b (multicam), or "
                            "neither to auto-reframe the single wide clip (--mode single)")
    if mode == "multicam":
        for name, p in (("--cam-a", cfg.cam_a), ("--cam-b", cfg.cam_b)):
            if not p:
                raise PipelineError(f"{name} is required in multicam mode")
    return mode


def run_pipeline(cfg: PipelineConfig, progress: Optional[ProgressFn] = None) -> PipelineResult:
    t_start = time.time()

    def prog(frac: float, msg: str) -> None:
        if progress:
            progress(max(0.0, min(1.0, frac)), msg)

    # ------------------------------------------------------------------ inputs
    single = resolve_mode(cfg) == "single"
    if single and (cfg.cam_a or cfg.cam_b):
        log.warning("--mode single: --cam-a/--cam-b are ignored (close-ups are cropped from --wide)")
        cfg = dataclasses.replace(cfg, cam_a="", cam_b="")
    clip_keys = ("wide",) if single else ("cam_a", "cam_b", "wide")
    for name, p in (("--cam-a", cfg.cam_a), ("--cam-b", cfg.cam_b), ("--wide", cfg.wide)):
        if single and name != "--wide":
            continue
        if not p:
            raise PipelineError(f"{name} is required")
        if not os.path.isfile(p):
            raise PipelineError(f"{name}: file not found: {p}")
    if single:
        try:
            parse_crop(cfg.reframe.crop_a)
            parse_crop(cfg.reframe.crop_b)
        except ValueError as e:
            raise PipelineError(str(e)) from e
        if cfg.reframe.a_side not in ("left", "right"):
            raise PipelineError(f"--a-side must be left or right (got {cfg.reframe.a_side!r})")
    if cfg.audio and not os.path.isfile(cfg.audio):
        raise PipelineError(f"--audio: file not found: {cfg.audio}")
    for nm, p in (("--ref-a", cfg.diar.ref_a), ("--ref-b", cfg.diar.ref_b)):
        if p and not os.path.isfile(p):
            raise PipelineError(f"{nm}: file not found: {p}")
    if bool(cfg.diar.ref_a) != bool(cfg.diar.ref_b):
        raise PipelineError("voice enrollment needs both --ref-a and --ref-b")
    # exact NTSC frame rates (29.97 -> 30000/1001) so the planner's frame grid == ffmpeg's
    cfg = dataclasses.replace(cfg, render=dataclasses.replace(cfg.render, fps=normalize_fps(cfg.render.fps)))
    # absolute paths everywhere (ffmpeg's final pass runs inside the work dir)
    ab = lambda p: os.path.abspath(p) if p else p  # noqa: E731
    cfg = dataclasses.replace(
        cfg, cam_a=ab(cfg.cam_a), cam_b=ab(cfg.cam_b), wide=ab(cfg.wide), audio=ab(cfg.audio),
        diar=dataclasses.replace(cfg.diar, ref_a=ab(cfg.diar.ref_a), ref_b=ab(cfg.diar.ref_b)))

    out_path = os.path.abspath(cfg.out)
    out_dir = os.path.dirname(out_path)
    stem = os.path.splitext(os.path.basename(out_path))[0]
    work = os.path.abspath(cfg.work_dir or os.path.join(out_dir, f"{stem}_work"))
    for d in (out_dir, work, os.path.join(work, "audio"), os.path.join(work, "cache")):
        os.makedirs(d, exist_ok=True)
    cache = os.path.join(work, "cache")

    ff = find_ffmpeg(cfg.ffmpeg, prefer_libass=cfg.captions.burner in ("auto", "ass"))
    prog(0.01, "probing inputs")
    clip_path = {"cam_a": cfg.cam_a, "cam_b": cfg.cam_b, "wide": cfg.wide}
    infos: Dict[str, MediaInfo] = {k: probe(ff, clip_path[k]) for k in clip_keys}
    if single:
        log.info("Single-clip mode: Speaker A / Speaker B close-ups are cropped from the wide shot")
    for k, inf in infos.items():
        if not inf.has_video:
            raise PipelineError(f"{k}: no video stream in {inf.path}")
        log.info("%s: %dx%d @ %.3f fps, %.2fs, audio=%s, rotation=%d", k, inf.width, inf.height,
                 inf.fps, inf.duration, inf.has_audio, inf.rotation)
    if cfg.audio:
        ainf = probe(ff, cfg.audio)
        if not ainf.has_audio:
            raise PipelineError(f"--audio has no audio stream: {cfg.audio}")

    # ------------------------------------------------------------------ audio
    prog(0.02, "extracting audio")
    adir = os.path.join(work, "audio")
    wavs: Dict[str, Optional[str]] = {"cam_a": None, "cam_b": None, "wide": None}
    for k in clip_keys:
        path = clip_path[k]
        if infos[k].has_audio:
            wavs[k] = _cached_wav(ff, path, os.path.join(adir, f"{k}.wav"), cfg.force)
        else:
            log.warning("%s has no audio track", k)
            wavs[k] = None
    if cfg.audio:
        ref_name, ref_src = "external", cfg.audio
        ref_wav = _cached_wav(ff, cfg.audio, os.path.join(adir, "external.wav"), cfg.force)
    else:
        ref_name = next((k for k in ("wide", "cam_a", "cam_b") if wavs.get(k)), None)
        if ref_name is None:
            raise PipelineError("no audio anywhere: provide --audio or clips with audio")
        if ref_name != "wide":
            log.warning("wide shot has no audio; using %s audio as the reference/master", ref_name)
        ref_src = clip_path[ref_name]
        ref_wav = wavs[ref_name]  # type: ignore[assignment]
    ref_x, sr = load_wav(ref_wav)
    duration = ref_x.size / sr
    del ref_x
    log.info("Reference/master audio: %s (%.2fs)", ref_name, duration)
    ref_a_wav = ref_b_wav = None
    if cfg.diar.ref_a and cfg.diar.ref_b:
        ref_a_wav = _cached_wav(ff, cfg.diar.ref_a, os.path.join(adir, "enroll_a.wav"), cfg.force)
        ref_b_wav = _cached_wav(ff, cfg.diar.ref_b, os.path.join(adir, "enroll_b.wav"), cfg.force)

    # ------------------------------------------------------------------ sync
    prog(0.06, "synchronising clips")
    sc = cfg.sync
    manual = {"cam_a": sc.offset_a, "cam_b": sc.offset_b, "wide": sc.offset_wide}
    sync_key = _key("sync", _fp(ref_wav), [_fp(wavs[k]) for k in ("cam_a", "cam_b", "wide")],
                    sc.max_lag_s, sc.analysis_s, sc.enabled, ref_name, manual, sc.min_confidence,
                    sc.force, *(["single"] if single else []))

    fresh_sync: List[bool] = []

    def _do_sync() -> Dict[str, Any]:
        fresh_sync.append(True)
        res = compute_offsets(ref_wav, {k: wavs[k] for k in clip_keys},
                              sc.max_lag_s, sc.analysis_s, sc.min_confidence, manual,
                              sc.enabled, reference_name=ref_name if ref_name != "external" else None,
                              force=sc.force)
        return {k: v.to_dict() for k, v in res.items()}

    offsets = _cached(os.path.join(cache, "sync.json"), sync_key, cfg.force, _do_sync, "sync offsets")
    sync_warnings: List[str] = []
    for k, v in offsets.items():
        log.info("offset %-6s %+8.3f s  (%s, confidence %s)", k, v["offset"], v["method"], v["confidence"])
        if v["method"] in ("none", "low-confidence") and v.get("note") and "disabled" not in v["note"]:
            sync_warnings.append(v["note"])
            if not fresh_sync:  # re-surface the warning when offsets come from the cache
                log.warning("Sync %s", v["note"])

    # ------------------------------------------------------------------ ASR
    prog(0.10, f"transcribing ({cfg.asr.model_size})")
    a = cfg.asr
    asr_key = _key("asr", _fp(ref_wav), a.model_size, a.language, a.beam_size, a.vad_filter,
                   a.compute_type, a.condition_on_previous_text, a.initial_prompt, a.refine_word_times)
    tr = _cached(os.path.join(cache, "transcript_raw.json"), asr_key, cfg.force,
                 lambda: transcribe(ref_wav, a, lambda f: prog(0.10 + 0.35 * f, "transcribing")),
                 "transcript")
    words = words_from_transcript(tr)
    if not words:
        log.warning("ASR found no words - the whole timeline will be WIDE")

    # ------------------------------------------------------------------ diarization
    prog(0.46, "identifying speakers")
    d = cfg.diar
    diar_key = _key("diar", _fp(ref_wav), asr_key, asdict(dataclasses.replace(d, ref_a=None, ref_b=None)),
                    _fp(ref_a_wav), _fp(ref_b_wav), bool(os.environ.get("HF_TOKEN")))
    diar = DiarizationResult.from_dict(_cached(
        os.path.join(cache, "diarization.json"), diar_key, cfg.force,
        lambda: diarize(ref_wav, words, d, ref_a_wav, ref_b_wav).to_dict(), "diarization"))

    # ------------------------------------------------------------------ mapping
    prog(0.60, "mapping speakers to cameras")
    off = {k: float(v["offset"]) for k, v in offsets.items()}
    rc = cfg.render
    rf = cfg.reframe
    reframe_meta: Dict[str, Any] = {}
    people: Dict[str, Optional[PersonTrack]] = {"A": None, "B": None}
    crops_ab: Dict[str, Optional[CropRect]] = {"A": None, "B": None}
    series = None
    scan_key = ""
    if single:
        off["cam_a"] = off["cam_b"] = off["wide"]  # the virtual cameras ARE the wide clip
        prog(0.60, "finding faces in the wide shot")
        speech0 = min((sg.start for sg in diar.segments), default=0.0)
        dense_start = round(max(0.0, speech0 - 1.0 - off["wide"]), 1)
        clip_dur = infos["wide"].video_duration or infos["wide"].duration
        scan_key = _key("faces", _fp(cfg.wide), rf.dense_fps, rf.dense_s, rf.sparse_fps,
                        rf.analysis_width, rf.max_frames, rf.timeout_s, rf.cluster_gap, dense_start)
        scan = _cached(os.path.join(cache, "faces.json"), scan_key, cfg.force,
                       lambda: scan_faces(ff.ffmpeg, cfg.wide, clip_dur, rf, dense_start), "face tracks")
        fw, fh = int(scan["frame_w"]), int(scan["frame_h"])
        tracks = [PersonTrack.from_dict(t) for t in scan["tracks"]]
        people, rf_warn = select_people(tracks, fw, fh, rf)
        crops_ab, crop_info, crop_warn = base_crops(people, fw, fh, rc.width, rc.height, rf)
        for w_ in rf_warn + crop_warn:
            log.warning("Reframe: %s", w_)
        for letter in ("A", "B"):
            p_, c_ = people[letter], crops_ab[letter]
            log.info("Speaker %s: %s; crop %s", letter,
                     (f"face on the {p_.side} at x={p_.cx:.0f}, y={p_.cy:.0f}, {p_.w:.0f}x{p_.h:.0f} px, "
                      f"seen in {100 * p_.persistence:.0f}% of scanned frames") if p_ else "no face",
                     (f"{c_.as_list()} (zoom {crop_info[letter]['zoom']:.2f}x)") if c_ else "none -> WIDE")
        series = motion_series(people, scan, off["wide"])
        reframe_meta = {
            "a_side": rf.a_side, "frame": [fw, fh], "detector": scan.get("detector"),
            "scanned_frames": scan.get("n_samples"), "frames_with_faces": scan.get("frames_with_faces"),
            "faces_found": len(tracks), "scan_stopped": scan.get("stopped", ""),
            "people": {k: ({"side": v.side, "cx": round(v.cx, 1), "cy": round(v.cy, 1), "w": round(v.w, 1),
                            "h": round(v.h, 1), "eye_y": round(v.eye_y, 1),
                            "persistence": round(v.persistence, 3)} if v else None)
                       for k, v in people.items()},
            "crops": {k: (v.as_list() if v else None) for k, v in crops_ab.items()},
            "crop_info": crop_info, "warnings": rf_warn + crop_warn,
        }
    m = dataclasses.replace(cfg.mapping, swap=False)  # swap is applied after the cache
    map_key = _key("map", diar_key, sync_key, asdict(m), str(d.num_speakers),
                   _fp(cfg.cam_a), _fp(cfg.cam_b),
                   *([scan_key, rf.a_side, rf.min_persistence] if single else []))
    if single:
        vis_in: Optional[Dict[str, Any]] = {"series": series} if series is not None else None
        mapping = MappingResult.from_dict(_cached(
            os.path.join(cache, "mapping.json"), map_key, cfg.force,
            lambda: decide_mapping(diar.segments, m, enrolled=diar.enrolled, mic_inputs=None,
                                   visual_inputs=vis_in, diar_stats=diar.stats,
                                   num_speakers=str(d.num_speakers)).to_dict(), "speaker mapping"))
    else:
        mapping = MappingResult.from_dict(_cached(
            os.path.join(cache, "mapping.json"), map_key, cfg.force,
            lambda: decide_mapping(
                diar.segments, m, enrolled=diar.enrolled,
                mic_inputs={"wav_a": wavs["cam_a"], "wav_b": wavs["cam_b"],
                            "offset_a": off["cam_a"], "offset_b": off["cam_b"]},
                visual_inputs={"ffmpeg": ff.ffmpeg, "video_a": cfg.cam_a, "video_b": cfg.cam_b,
                               "offset_a": off["cam_a"], "offset_b": off["cam_b"]},
                diar_stats=diar.stats, num_speakers=str(d.num_speakers),
            ).to_dict(), "speaker mapping"))
    if mapping.num_speakers == 1:
        who = next(iter(mapping.mapping.values()))
        log.warning("Single speaker detected (%s): every word goes to Speaker %s / cam %s. "
                    "Use --num-speakers 2 if this is wrong.", "; ".join(mapping.count_reasons), who, who)
    if cfg.mapping.swap:
        mapping.mapping = {k: ("B" if v == "A" else "A") for k, v in mapping.mapping.items()}
        mapping.method += "+swapped"
        log.info("Speakers swapped by user: %s", mapping.mapping)
    if not mapping.confident:
        log.warning("Speaker->camera mapping is NOT confident (%s). Check the output and use "
                    "--swap-speakers if A/B are reversed.", mapping.method)
        if single:
            log.warning("Single-clip mode: could not tell from mouth motion which VOICE belongs to "
                        "which FACE; assuming the first voice heard is Speaker A (the person on the "
                        "%s). If the close-ups show the wrong person, re-run with --speaker-map BA "
                        "(or --swap-speakers).", rf.a_side)

    # ------------------------------------------------------------------ labeling
    prog(0.64, "labelling words")
    segs_ab = apply_mapping(diar.segments, mapping.mapping)
    words = label_words(words, segs_ab, cfg.label)
    turns = build_turns(words, cfg.label.turn_gap)
    overlaps: List[Interval] = overlaps_from_words(words, cfg.planner.min_overlap)
    overlap_info: Dict[str, Any] = {"words": len(overlaps)}
    if wavs.get("cam_a") and wavs.get("cam_b") and mapping.num_speakers == 2:
        lc = cfg.label
        try:
            mic_ov, info = overlaps_from_mic_coupling(
                words, wavs["cam_a"], wavs["cam_b"], off["cam_a"], off["cam_b"], duration,
                min_dur=cfg.planner.min_overlap, min_sep_db=lc.overlap_mic_min_sep_db,
                excess_db=lc.overlap_excess_db, excess_frac=lc.overlap_excess_frac)
            overlap_info["mic_coupling"] = info
            if mic_ov is None:  # mics can't be calibrated -> previous (midpoint) method
                log.info("mic-coupling overlap detection unavailable (%s); using midpoint method",
                         info.get("reason"))
                mic_ov = overlaps_from_mics(words, wavs["cam_a"], wavs["cam_b"], off["cam_a"],
                                            off["cam_b"], duration,
                                            min_dur=max(0.4, cfg.planner.min_overlap))
                overlap_info["method"] = "midpoint"
            else:
                overlap_info["method"] = "mic_coupling"
                log.info("mic-coupling overlap detection: %d regions (%s)", len(mic_ov), info)
            overlap_info["mics"] = len(mic_ov)
            overlaps = _merge(overlaps + mic_ov, 0.2)
        except Exception as e:  # noqa: BLE001
            log.warning("mic-based overlap detection failed: %s", e)
    log.info("%d words, %d turns (A=%d, B=%d), %d overlap regions", len(words), len(turns),
             sum(t.speaker == "A" for t in turns), sum(t.speaker == "B" for t in turns), len(overlaps))

    # ------------------------------------------------------------------ plan
    prog(0.66, "planning shots")
    wide_src = SourceClip(cfg.wide, off["wide"], infos["wide"])
    shot_crop_list: Optional[List[Optional[List[int]]]] = None
    if single:
        # both virtual cameras exist whenever the wide clip does - unless that person has
        # no crop (no face found): None = never available -> those turns stay WIDE
        sources: Dict[str, Optional[SourceClip]] = {CAM_A: wide_src, CAM_B: wide_src, WIDE: wide_src}
        coverage: Dict[str, Any] = {WIDE: wide_src.coverage(),
                                    CAM_A: wide_src.coverage() if crops_ab["A"] else None,
                                    CAM_B: wide_src.coverage() if crops_ab["B"] else None}
    else:
        sources = {
            CAM_A: SourceClip(cfg.cam_a, off["cam_a"], infos["cam_a"]),
            CAM_B: SourceClip(cfg.cam_b, off["cam_b"], infos["cam_b"]),
            WIDE: wide_src,
        }
        coverage = {c: s.coverage() for c, s in sources.items() if s is not None}
    events: List[PlanEvent] = []
    shots = plan_shots(turns, duration, rc.fps, cfg.planner, overlaps, coverage, events)
    validate_shots(shots, duration, rc.fps)
    stats = _stats(shots, events, duration)
    stats["mode"] = "single" if single else "multicam"
    if single:
        per_shot, n_reframed = shot_crops(shots, people, crops_ab, int(reframe_meta["frame"][0]),
                                          int(reframe_meta["frame"][1]), rc.width, rc.height, rf,
                                          off["wide"])
        shot_crop_list = [c.as_list() if c is not None else None for c in per_shot]
        reframe_meta["reframed_shots"] = n_reframed
        if n_reframed:
            log.info("Reframe: %d close-up shots re-framed (a person moved); crops change only at "
                     "shot boundaries", n_reframed)
        stats["reframe"] = {k: reframe_meta[k] for k in ("a_side", "crops", "faces_found",
                                                         "reframed_shots", "warnings")}
    stats["num_speakers"] = mapping.num_speakers
    if sync_warnings:
        stats["sync_warnings"] = sync_warnings
    stats["overlap_detection"] = overlap_info
    if mapping.count_reasons:
        stats["speaker_count_reasons"] = mapping.count_reasons
    log.info("Plan: %d shots %s", len(shots), stats["camera_seconds"])

    # ------------------------------------------------------------------ outputs
    outputs: Dict[str, str] = {}
    lines, style = build_captions(words, cfg.captions, rc.width, rc.height)
    outputs["ass"] = write_ass(os.path.join(out_dir, f"{stem}_captions.ass"), lines, style, cfg.captions)
    outputs["srt"] = write_srt(os.path.join(out_dir, f"{stem}_captions.srt"), lines)
    outputs["transcript"] = write_transcript_json(
        os.path.join(out_dir, f"{stem}_transcript.json"), words, turns, lines, cfg.captions,
        {"language": tr.get("language"), "asr_model": tr.get("model"),
         "segments": tr.get("segments", []),
         "diarization": {"backend": diar.backend, "segments": [s.to_dict() for s in segs_ab]},
         "overlaps": [o.to_dict() for o in overlaps]})
    rows = edl_rows(shots, sources, rc.fps, shot_crop_list)
    if single:
        # crop preview: the wide frame with both crop rectangles, to check the framing
        both = sorted(set(people["A"].times if people["A"] else []) & set(people["B"].times if people["B"] else []))
        anyt = both or sorted((people["A"].times if people["A"] else []) + (people["B"].times if people["B"] else []))
        t_prev = anyt[len(anyt) // 2] if anyt else min(duration / 2.0, max(0.0, clip_dur - 0.5))
        from .captions import parse_color

        colors = {"A": parse_color(cfg.captions.color_a), "B": parse_color(cfg.captions.color_b)}
        outputs["crop_preview"] = os.path.join(out_dir, f"{stem}_crop_preview.png")
        try:
            write_preview(ff.ffmpeg, cfg.wide, t_prev, crops_ab, people, outputs["crop_preview"], colors)
            log.info("Crop preview: %s", outputs["crop_preview"])
            if rf.preview:
                import shutil as _sh

                os.makedirs(os.path.dirname(os.path.abspath(rf.preview)) or ".", exist_ok=True)
                _sh.copyfile(outputs["crop_preview"], rf.preview)
                outputs["crop_preview_copy"] = os.path.abspath(rf.preview)
        except Exception as e:  # noqa: BLE001
            log.warning("could not write the crop preview: %s", e)
            outputs.pop("crop_preview", None)
    meta = {
        "mode": "single" if single else "multicam", "reframe": reframe_meta,
        "version": __version__, "fps": rc.fps, "width": rc.width, "height": rc.height,
        "duration": duration, "reference_audio": ref_name, "offsets": offsets,
        "mapping": mapping.to_dict(), "diarization_backend": diar.backend,
        "events": [asdict(e) for e in events], "stats": stats,
        "inputs": {"cam_a": cfg.cam_a, "cam_b": cfg.cam_b, "wide": cfg.wide, "audio": cfg.audio},
    }
    outputs["edl_json"] = os.path.join(out_dir, f"{stem}_edit_decision_list.json")
    outputs["edl_csv"] = os.path.join(out_dir, f"{stem}_edit_decision_list.csv")
    write_edl(rows, outputs["edl_json"], outputs["edl_csv"], meta)

    # ------------------------------------------------------------------ render
    video: Optional[str] = None
    if not cfg.plan_only:
        prog(0.68, "rendering segments")
        seg_paths = render_segments(ff, shots, sources, rc, os.path.join(work, "segments"),
                                    lambda f, msg: prog(0.68 + 0.17 * f, msg), crops=shot_crop_list)
        audio = _audio_source(cfg, rc.audio_source, ref_src, ref_name, off, infos)
        ass_path = overlay = None
        if cfg.captions.enabled and lines:
            burner = cfg.captions.burner
            if burner == "auto":
                burner = "ass" if ff.has_libass else "pillow"
            if burner == "ass":
                if not ff.has_libass:
                    raise PipelineError(f"{ff.ffmpeg} has no libass; use --caption-burner pillow "
                                        "or install ffmpeg-full")
                ass_path = os.path.join(work, "captions.ass")
                write_ass(ass_path, lines, style, cfg.captions)
            else:
                from .caption_overlay import render_overlay

                overlay, _ = render_overlay(lines, style, cfg.captions,
                                            os.path.join(work, "caption_png"), duration)
            log.info("Caption burn-in: %s", burner)
            stats["caption_burner"] = burner
        prog(0.86, "final encode")
        video = render_final(ff, seg_paths, audio, duration, rc, work, out_path, ass_path, overlay,
                             lambda f, msg: prog(0.86 + 0.14 * f, msg))
        outputs["video"] = video
        if not rc.keep_intermediates:
            import shutil

            shutil.rmtree(os.path.join(work, "segments"), ignore_errors=True)
            shutil.rmtree(os.path.join(work, "caption_png"), ignore_errors=True)
    stats["elapsed_s"] = round(time.time() - t_start, 1)
    summary = {
        "outputs": outputs, "stats": stats, "offsets": offsets, "mapping": mapping.to_dict(),
        "diarization_backend": diar.backend, "diarization_notes": diar.notes,
        "ffmpeg": ff.ffmpeg, "config": cfg.to_dict(),
    }
    outputs["summary"] = os.path.join(out_dir, f"{stem}_run_summary.json")
    with open(outputs["summary"], "w") as f:
        json.dump(summary, f, indent=1, default=str)
    prog(1.0, "done")
    return PipelineResult(video, outputs, shots, turns, words, offsets, mapping.to_dict(),
                          diar.backend, stats, rows, transcript_text(turns))


def _audio_source(cfg: PipelineConfig, choice: str, ref_src: str, ref_name: str,
                  off: Dict[str, float], infos: Dict[str, MediaInfo]) -> Optional[AudioSource]:
    choice = (choice or "auto").lower()
    if choice == "auto":
        if cfg.audio:
            return AudioSource(cfg.audio, 0.0)
        if infos["wide"].has_audio:
            return AudioSource(cfg.wide, off["wide"] if ref_name != "wide" else 0.0)
        return AudioSource(ref_src, 0.0)
    if choice in ("external", "master"):
        if not cfg.audio:
            raise PipelineError("--audio-source external requires --audio")
        return AudioSource(cfg.audio, 0.0)
    key = {"wide": "wide", "cam-a": "cam_a", "cam_a": "cam_a", "cam-b": "cam_b", "cam_b": "cam_b"}.get(choice)
    if key is None:
        raise PipelineError(f"unknown --audio-source {choice!r}")
    if key not in infos:
        raise PipelineError(f"--audio-source {choice} is not available in single-clip mode")
    if not infos[key].has_audio:
        raise PipelineError(f"--audio-source {choice}: that clip has no audio")
    path = {"cam_a": cfg.cam_a, "cam_b": cfg.cam_b, "wide": cfg.wide}[key]
    return AudioSource(path, 0.0 if key == ref_name else off[key])


def _stats(shots: List[Shot], events: List[PlanEvent], duration: float) -> Dict[str, Any]:
    secs: Dict[str, float] = {}
    cnt: Dict[str, int] = {}
    for s in shots:
        secs[s.camera] = round(secs.get(s.camera, 0.0) + s.duration, 2)
        cnt[s.camera] = cnt.get(s.camera, 0) + 1
    ev: Dict[str, int] = {}
    for e in events:
        ev[e.kind] = ev.get(e.kind, 0) + 1
    return {"n_shots": len(shots), "camera_seconds": secs, "camera_shots": cnt,
            "events": ev, "duration": round(duration, 3),
            "min_shot": round(min((s.duration for s in shots), default=0), 3)}
