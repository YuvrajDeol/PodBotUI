"""UI layer: pure renderers, form rules, presets and the defaults-equivalence with the old UI."""

import importlib.util
import os
import sys

import pytest

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, ROOT)

gr = pytest.importorskip("gradio")
import app  # noqa: E402
from ui import render as R  # noqa: E402

BEFORE = os.path.join(ROOT, "design", "app_before.py")


# --------------------------------------------------------------------------- roles

@pytest.mark.parametrize("order", [(0, 1, 2), (2, 0, 1), (1, 2, 0), (2, 1, 0)])
def test_guess_roles_demo_names_any_order(order):
    names = ["/t/x/cam_a.mp4", "/t/y/cam_b.mp4", "/t/z/wide.mp4"]
    paths = [names[i] for i in order]
    assert R.guess_roles(paths) == tuple(names)


def test_guess_roles_rules_and_fallback():
    assert R.guess_roles(["Speaker_B.mov", "MASTER.mp4", "host-a.mp4"]) == (
        "host-a.mp4", "Speaker_B.mov", "MASTER.mp4")
    assert R.guess_roles(["CamB.mp4", "CamA.mp4", "both.mp4"]) == ("CamA.mp4", "CamB.mp4", "both.mp4")
    # nothing recognisable: upload order A, B, wide
    assert R.guess_roles(["1.mp4", "2.mp4", "3.mp4"]) == ("1.mp4", "2.mp4", "3.mp4")
    # only the wide shot recognisable: the rest in upload order
    assert R.guess_roles(["x.mp4", "wide.mp4", "y.mp4"]) == ("x.mp4", "y.mp4", "wide.mp4")


def test_role_choices_number_duplicate_names():
    assert R.role_choices(["/a/v.mp4", "/b/w.mp4"]) == [("v.mp4", "/a/v.mp4"), ("w.mp4", "/b/w.mp4")]
    labels = [lab for lab, _ in R.role_choices(["/a/v.mp4", "/b/v.mp4", "/c/w.mp4"])]
    assert labels == ["1 · v.mp4", "2 · v.mp4", "3 · w.mp4"]


# --------------------------------------------------------------------------- form rules

def test_form_check():
    three = ["/a.mp4", "/b.mp4", "/w.mp4"]
    ok = app.form_check(three, three, ("", "", ""))
    assert ok["ready"] and not ok["dup_roles"] and ok["helper"] == ""
    assert app.form_check([], (None,) * 3, ("",) * 3)["helper"] == "Add your video to start."
    two = app.form_check(three[:2], (None,) * 3, ("",) * 3)
    assert two["count_error"] and not two["ready"]
    dup = app.form_check(three, ["/a.mp4", "/a.mp4", "/w.mp4"], ("",) * 3)
    assert dup["dup_roles"] and not dup["ready"]
    bad = app.form_check(three, three, ("1.2x", "-1.25", " "))
    assert bad["bad_offsets"] == [True, False, False] and not bad["ready"]
    # single-clip mode ignores the (hidden) cam A/B offsets
    single = app.form_check(["/w.mp4"], (None,) * 3, ("junk", "junk", "0.5"))
    assert single["ready"] and single["bad_offsets"] == [False, False, False]


def test_build_config_errors(tmp_path):
    clips = []
    for n in ("cam_a.mp4", "cam_b.mp4", "wide.mp4"):
        p = tmp_path / n
        p.write_bytes(b"x")
        clips.append(str(p))
    with pytest.raises(ValueError, match="You added 2"):
        app.build_config(clips[:2], run_dir=str(tmp_path))
    with pytest.raises(ValueError, match=R.ROLE_ERROR):
        app.build_config(clips, (clips[0], clips[0], clips[2]), run_dir=str(tmp_path))
    with pytest.raises(ValueError, match="number of seconds"):
        app.build_config(clips, clips, run_dir=str(tmp_path), offset_a="abc")
    with pytest.raises(TypeError):
        app.build_config(clips, clips, run_dir=str(tmp_path), not_a_setting=1)
    # all roles empty (API use): roles come from the file names
    cfg = app.build_config(list(reversed(clips)), run_dir=str(tmp_path))
    assert (cfg.cam_a, cfg.cam_b, cfg.wide) == tuple(clips)


def test_presets_and_mappings(tmp_path):
    wide = tmp_path / "wide.mp4"
    wide.write_bytes(b"x")
    assert app.CUT_PRESETS["Calm (fewer cuts)"] == dict(qa_gap=1.5, qa_hold=2.5, rapid_turns=4, rapid_window=6.0,
                                                       min_shot=2.0, silence_wide=3.5, question_heuristic=True)
    assert app.CUT_PRESETS["Balanced"] == dict(qa_gap=1.5, qa_hold=2.0, rapid_turns=3, rapid_window=6.0,
                                              min_shot=1.2, silence_wide=2.5, question_heuristic=True)
    assert app.CUT_PRESETS["Snappy (more cuts)"] == dict(qa_gap=1.5, qa_hold=1.5, rapid_turns=3, rapid_window=6.0,
                                                        min_shot=0.8, silence_wide=2.0, question_heuristic=True)
    cfg = app.build_config([str(wide)], run_dir=str(tmp_path), accuracy="Best", language=" hi ",
                           quality="720p · 30 fps", width=1, height=1, fps=1, **app.CUT_PRESETS["Snappy (more cuts)"])
    assert cfg.mode == "single" and cfg.cam_a == cfg.cam_b == ""
    assert cfg.asr.model_size == "large-v3" and cfg.asr.language == "hi"
    assert (cfg.render.width, cfg.render.height, cfg.render.fps) == (1280, 720, 30.0)
    assert cfg.planner.min_shot == 0.8 and cfg.planner.qa_answer_hold == 1.5
    custom = app.build_config([str(wide)], run_dir=str(tmp_path), quality="Custom", width=1280, height=544, fps=25)
    assert (custom.render.width, custom.render.height, custom.render.fps) == (1280, 544, 25.0)
    assert app.build_config([str(wide)], run_dir=str(tmp_path), accuracy="Fast").asr.model_size == "base"
    assert app.build_config([str(wide)], run_dir=str(tmp_path), language="Auto-detect").asr.language is None


# --------------------------------------------------------------------------- defaults == previous UI

def _component_defaults(demo, fn_name):
    fn = next(f for f in demo.fns.values() if getattr(f, "name", None) == fn_name)
    return [getattr(comp, "value", None) for comp in fn.inputs]


@pytest.mark.skipif(not os.path.isfile(BEFORE), reason="design/app_before.py not present")
@pytest.mark.parametrize("mode", ["multicam", "single"])
def test_untouched_controls_match_previous_ui(mode, tmp_path, monkeypatch):
    """build_config(untouched new controls) == PipelineConfig of the old app's run_edit."""
    spec = importlib.util.spec_from_file_location("app_before", BEFORE)
    before = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(before)
    clips = []
    for n in ("cam_a.mp4", "cam_b.mp4", "wide.mp4"):
        p = tmp_path / n
        p.write_bytes(b"0" * (10 + len(clips)))
        clips.append(str(p))
    runs = str(tmp_path / "runs")
    monkeypatch.setattr(before, "RUNS_DIR", runs)
    monkeypatch.setattr(app, "RUNS_DIR", runs)

    captured = []

    def fake_pipeline(cfg, progress=None):
        captured.append(cfg)
        raise RuntimeError("captured")

    monkeypatch.setattr(before, "run_pipeline", fake_pipeline)
    old = _component_defaults(before.build_ui(), "run_edit")
    old[0:3] = (clips if mode == "multicam" else [None, None, clips[2]])
    if mode == "single":
        old[24] = before.MODE_SINGLE  # the old UI's mode radio is the one control a single-clip user changes
    with pytest.raises(gr.Error):
        before.run_edit(*old, progress=lambda *a, **k: None)
    old_cfg = captured[0]

    new_vals = _component_defaults(app.build_ui(), "run_edit")
    settings = dict(zip(app.SETTING_KEYS, new_vals[4:]))
    assert settings == app.DEFAULTS  # the UI really starts at DEFAULTS
    files = clips if mode == "multicam" else [clips[2]]
    roles = R.guess_roles(files) if mode == "multicam" else (None, None, None)
    new_cfg = app.build_config(files, roles, False, os.path.dirname(old_cfg.out), **settings)

    a, b = old_cfg.to_dict(), new_cfg.to_dict()
    assert a.pop("out") and b.pop("out")
    assert a == b
    assert new_cfg.work_dir == old_cfg.work_dir


def test_advanced_drawer_reapplies_mode_groups_when_opened():
    """Gradio drops visible=False for controls inside a never-opened accordion, so opening the
    drawer re-applies: cam A/B offsets hidden and framing shown in single-clip mode."""
    demo = app.build_ui()
    fn = next(f for f in demo.fns.values() if getattr(f, "name", None) == "on_adv_open")
    adv = next(b for b in demo.blocks.values() if getattr(b, "elem_id", None) == "pb-adv")
    assert any(t[0] == adv._id and t[1] == "expand" for t in fn.targets)
    vis = lambda ups: [u["visible"] for u in ups]  # noqa: E731
    assert vis(fn.fn(["/w.mp4"])) == [False, False, True]  # cam A offset, cam B offset, framing
    assert vis(fn.fn(["/a.mp4", "/b.mp4", "/w.mp4"])) == [True, True, False]
    assert vis(fn.fn(None)) == [True, True, False]


# --------------------------------------------------------------------------- run handler

def test_runs_never_overlap_after_a_page_reload(tmp_path, monkeypatch):
    """A reload closes the run generator (Gradio frees the concurrency slot) but cannot stop the
    pipeline thread.  The next run must wait for it, not share its work_dir and log handler."""
    import logging
    import threading

    wide = tmp_path / "wide.mp4"
    wide.write_bytes(b"x")
    monkeypatch.setattr(app, "RUNS_DIR", str(tmp_path / "runs"))
    release, guard = threading.Event(), threading.Lock()
    calls, active, peak = [], [0], [0]

    def fake_pipeline(cfg, progress):
        with guard:
            calls.append(cfg.out)
            n = len(calls)
            active[0] += 1
            peak[0] = max(peak[0], active[0])
        try:
            logging.getLogger("podcast_autoeditor.fake").warning("run %d started", n)
            progress(0.1, "transcribing")
            if n == 1:
                release.wait(10)
                logging.getLogger("podcast_autoeditor.fake").warning("run 1 tail")
            raise app.PipelineError(f"stop {n}")
        finally:
            with guard:
                active[0] -= 1

    monkeypatch.setattr(app, "run_pipeline", fake_pipeline)
    fn = next(f for f in app.build_ui().fns.values() if getattr(f, "name", None) == "run_plan")
    vals = [getattr(comp, "value", None) for comp in fn.inputs]
    vals[0] = [str(wide)]

    def progress_until(gen, text):
        for _ in range(20):
            upd = next(gen)
            page = next((v for k, v in upd.items() if getattr(k, "elem_id", None) == "pb-progress"), "")
            if text in str(page):
                return
        raise AssertionError(f"no progress update containing {text!r}")

    try:
        first = fn.fn(*vals)
        progress_until(first, "transcribing")
        first.close()  # the page was reloaded: Gradio closes the generator, the thread runs on
        second = fn.fn(*vals)
        progress_until(second, app.WAIT_MSG)
        assert len(calls) == 1  # run 2 has not started next to the orphaned run 1
    finally:
        release.set()
    last = list(second)[-1]
    err = next(v for k, v in last.items() if getattr(k, "elem_id", None) == "pb-error")
    assert len(calls) == 2 and peak[0] == 1
    assert "stop 2" in err["value"] and "run 2 started" in err["value"]
    assert "run 1" not in err["value"]  # run 2's log holds only its own lines


# --------------------------------------------------------------------------- renderers

def test_stage_mapping():
    s = 0
    seen = []
    for msg in ["probing inputs", "extracting audio", "synchronising clips", "transcribing (small)",
                "transcribing", "identifying speakers", "mapping speakers to cameras",
                "finding faces in the wide shot", "labelling words", "planning shots", "something new",
                "rendering segments", "segments 3/14", "final encode", "done"]:
        s = R.stage_for(msg, s)
        seen.append(s)
    assert seen == [0, 0, 0, 1, 1, 2, 2, 2, 2, 3, 4, 4, 4, 4, 5]
    assert R.stage_for("unknown text", 1) == 1  # unknown before planning: stay
    assert R.stage_for("probing inputs", 3) == 3  # never backwards


def test_progress_html_states():
    h = R.progress_html(1, 0.42, "transcribing (small)", 75, plan_only=True)
    assert 'aria-live="polite"' in h and "EDITING · 01:15" in h and "42%" in h
    assert h.count("is-done") == 1 and h.count("is-active") == 1 and "Skipped" in h
    done = R.progress_html(R.DONE, 1.0, "done", 75, plan_only=False, finished=True)
    assert "FINISHED" in done and done.count("is-done") == 5 and "pb-bar" not in done


def test_html_escaping():
    evil = '<img src=x onerror=alert(1)>'
    t = R.transcript_html(f"[00:01.50] Speaker A: {evil}\n[01:02.00] Speaker B: ok", "#2D9CFF", "#FFFFFF")
    assert evil not in t and "&lt;img" in t and 'data-t="1.50"' in t and 'data-t="62.00"' in t
    raw = R.transcript_html(f"not a transcript line {evil}", "#2D9CFF", "#FFFFFF")
    assert raw.startswith("<pre") and evil not in raw
    e = R.error_html("PipelineError", evil, True, [evil])
    assert evil not in e and 'role="alert"' in e and "Check the files and settings above" in e
    w = R.warnings_html([evil, evil], confident=False)
    assert evil not in w and w.count("<li>") == 1 and "isn&#x27;t sure" in w
    # colours only ever reach a style attribute as #RRGGBB
    p = R.caption_preview_html('red;background:url(x)', "#ffffff")
    assert "url(" not in p and "#2D9CFF" in p and "#FFFFFF" in p


def test_timeline_html():
    edl = [{"camera": "WIDE", "start": 0.0, "end": 3.0}, {"camera": "CAM_A", "start": 3.0, "end": 9.5},
           {"camera": "CAM_B", "start": 9.5, "end": 12.0}]
    h = R.timeline_html(edl, {"WIDE": 3.0, "CAM_A": 6.5, "CAM_B": 2.5}, "#2D9CFF", "#FFFFFF")
    assert h.count('class="pb-seg ') == 3
    assert 'aria-label="Speaker A close-up, 0:03 to 0:10"' in h and 'data-t="9.500"' in h
    assert "A close-up · 6 s" in h and "B close-up · 2 s" in h and "Wide · 3 s" in h
    assert "min-width:max(100%,24px)" in h  # >= 2 px per second on phones
    assert "CUT TIMELINE · 3 SHOTS" in h
    assert "CUT TIMELINE · 1 SHOT<" in R.timeline_html(edl[:1], {"WIDE": 3.0}, "#2D9CFF", "#FFFFFF")


def test_css_never_overrides_a_var_shorthand_with_its_longhand():
    """Gradio re-scopes our CSS by re-serializing the CSSOM, where e.g. "border: 1px solid var(--x);
    border-left-width: 4px" comes out as empty longhands and is dropped: the error panel and the
    warnings card lost their coloured border."""
    import re
    css = open(os.path.join(ROOT, "ui", "podbot.css"), encoding="utf-8").read()
    css = re.sub(r"/\*.*?\*/", "", css, flags=re.S)
    longhand = {"border": r"border-(?!radius)", "font": r"(font-|line-height$)", "background": r"background-",
                "transition": r"transition-", "animation": r"animation-", "outline": r"outline-(?!offset)"}
    for block in re.findall(r"\{([^{}]*)\}", css):
        decls = [(p.strip(), v) for p, v in (d.split(":", 1) for d in block.split(";") if ":" in d)]
        for i, (prop, val) in enumerate(decls):
            if prop in longhand and "var(" in val:
                later = [p for p, _ in decls[i + 1:] if re.match(longhand[prop], p)]
                assert not later, f"{prop}: {val.strip()} followed by {later} in {{{block.strip()[:80]}}}"


def test_ink_for_contrast():
    assert R.ink_for("#2D9CFF") == R.INK and R.ink_for("#FFFFFF") == R.INK
    assert R.ink_for("#101060") == R.WHITE
    assert round(R.contrast("#CF1830", "#FFFFFF"), 2) == 5.49
