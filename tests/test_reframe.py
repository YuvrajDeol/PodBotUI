"""Single-clip mode: person tracks, crop geometry, per-shot re-framing, fallbacks, mode."""

import numpy as np
import pytest

from podcast_autoeditor import CAM_A, CAM_B, WIDE
from podcast_autoeditor.config import PipelineConfig, PlannerConfig, ReframeConfig
from podcast_autoeditor.datatypes import Shot, SpeakerSegment
from podcast_autoeditor.planner import plan_shots, validate_shots
from podcast_autoeditor.reframe import (CropRect, PersonTrack, base_crops, build_tracks,
                                        cluster_faces, compute_crop, fit_manual_crop,
                                        motion_series, parse_crop, select_people, shot_crops,
                                        snap_size)

from helpers import cam_at, turns

FW, FH = 1920, 1080


def face(cx, cy=500.0, w=120.0, h=150.0, patch=None):
    f = {"cx": float(cx), "cy": float(cy), "w": float(w), "h": float(h), "eye_y": cy - 0.1 * h, "score": 0.9}
    if patch is not None:
        f["mouth_patch"] = patch
    return f


def samples_two(n=100, miss_left=(), miss_right=(), third=None, jitter=4.0, seed=0):
    """Two people at x~500 and x~1400 (+ optional third face), with missed detections."""
    rng = np.random.default_rng(seed)
    out = []
    for i in range(n):
        faces = []
        if i not in miss_left:
            faces.append(face(500 + rng.normal(0, jitter), 500 + rng.normal(0, jitter)))
        if i not in miss_right:
            faces.append(face(1400 + rng.normal(0, jitter), 520 + rng.normal(0, jitter)))
        if third is not None and i % third["every"] == 0:
            faces.append(face(third["cx"], third.get("cy", 300), third.get("w", 60), third.get("h", 75)))
        out.append({"t": i * 0.2, "dense": True, "faces": faces})
    return out


# ----------------------------------------------------------------------------- tracks

def test_two_tracks_survive_missed_detections():
    s = samples_two(100, miss_left=set(range(20, 45)), miss_right=set(range(0, 100, 3)))
    tracks = build_tracks(s, FW)
    assert len(tracks) == 2
    left, right = tracks
    assert abs(left.cx - 500) < 6 and abs(right.cx - 1400) < 6
    assert left.hits == 75 and left.persistence == pytest.approx(0.75)
    assert right.hits == 66
    people, warns = select_people(tracks, FW, FH, ReframeConfig())
    assert people["A"] is left and people["B"] is right and not warns
    assert left.side == "left" and right.side == "right"


def test_a_side_right_swaps_the_letters():
    tracks = build_tracks(samples_two(60), FW)
    people, _ = select_people(tracks, FW, FH, ReframeConfig(a_side="right"))
    assert people["A"].cx > 1000 and people["B"].cx < 1000
    with pytest.raises(ValueError):
        select_people(tracks, FW, FH, ReframeConfig(a_side="middle"))


def test_third_background_face_is_ignored_with_a_warning():
    # a smaller face on a poster between them, present in every frame
    s = samples_two(80, third={"cx": 950, "cy": 250, "w": 50, "h": 62, "every": 1})
    tracks = build_tracks(s, FW)
    assert len(tracks) == 3
    people, warns = select_people(tracks, FW, FH, ReframeConfig())
    assert abs(people["A"].cx - 500) < 6 and abs(people["B"].cx - 1400) < 6
    assert any("3 faces" in w for w in warns)


def test_sporadic_false_detection_is_dropped_silently():
    s = samples_two(100, third={"cx": 950, "every": 20})    # in 5 % of frames
    people, warns = select_people(build_tracks(s, FW), FW, FH, ReframeConfig())
    assert people["A"] is not None and people["B"] is not None and not warns


def test_smaller_face_behind_a_person_does_not_join_the_track():
    s = samples_two(50)
    for smp in s:                                           # someone standing behind the left person
        smp["faces"].append(face(520, 380, 40, 50))
    groups = cluster_faces(s, 0.8)
    assert len(groups) == 2 and all(len(g) == 50 for g in groups)
    left = build_tracks(s, FW)[0]
    assert left.h == pytest.approx(150, abs=1) and abs(left.cy - 500) < 6


def test_zero_and_one_face_fallbacks():
    none = [{"t": i * 0.2, "dense": True, "faces": []} for i in range(30)]
    people, warns = select_people(build_tracks(none, FW), FW, FH, ReframeConfig())
    assert people == {"A": None, "B": None} and any("NO face" in w for w in warns)
    only_right = samples_two(60, miss_left=set(range(60)))
    people, warns = select_people(build_tracks(only_right, FW), FW, FH, ReframeConfig())
    assert people["A"] is None and people["B"] is not None and any("only ONE face" in w for w in warns)
    people, _ = select_people(build_tracks(only_right, FW), FW, FH, ReframeConfig(a_side="right"))
    assert people["A"] is not None and people["B"] is None


def test_missing_virtual_camera_keeps_those_turns_wide():
    """1 face: the planner sees the other speaker's camera as never available -> WIDE.
    0 faces: everything WIDE."""
    ts = turns([("A", 1.0, 10.0, "I have been working on this project for a long time now."),
                ("B", 10.6, 20.0, "That sounds great and I really like the direction you took."),
                ("A", 20.6, 30.0, "Thanks a lot I appreciate it very much indeed my friend.")])
    cov = {WIDE: (0.0, 31.0), CAM_A: (0.0, 31.0), CAM_B: None}
    shots = plan_shots(ts, 31.0, 30.0, PlannerConfig(), coverage=cov)
    validate_shots(shots, 31.0, 30.0)
    assert cam_at(shots, 5.0) == CAM_A and cam_at(shots, 15.0) == WIDE and cam_at(shots, 25.0) == CAM_A
    assert CAM_B not in {s.camera for s in shots}
    shots = plan_shots(ts, 31.0, 30.0, PlannerConfig(), coverage={WIDE: (0.0, 31.0), CAM_A: None, CAM_B: None})
    assert [s.camera for s in shots] == [WIDE]


def test_mouth_motion_follows_the_right_track():
    rng = np.random.default_rng(3)
    still = rng.integers(0, 255, (32, 48)).astype(np.uint8)
    s = []
    for i in range(100):
        talking_left = (i // 20) % 2 == 0
        pl = rng.integers(0, 255, (32, 48)).astype(np.uint8) if talking_left else still
        pr = still if talking_left else rng.integers(0, 255, (32, 48)).astype(np.uint8)
        s.append({"t": i * 0.2, "dense": True, "faces": [face(500, patch=pl), face(1400, patch=pr)]})
    left, right = build_tracks(s, FW)
    assert len(left.motion) == 99
    m = np.array(left.motion)
    t = np.array(left.motion_t)
    talk = ((np.round(t / 0.2).astype(int) // 20) % 2 == 0)
    assert m[talk][1:].mean() > 5 * m[~talk][1:].mean()
    people, _ = select_people([left, right], FW, FH, ReframeConfig())
    ser = motion_series(people, {"dense_fps": 5.0, "dense_window": [0.0, 20.0], "n_dense": 100,
                                 "detector": "fake"}, offset=1.5)
    assert ser["times"][0] == pytest.approx(1.5) and ser["motion_a"] is not None
    # reference time = clip time + offset
    k = int(np.nanargmax(np.where(np.isnan(ser["motion_a"]), -1, ser["motion_a"])))
    assert ser["times"][k] - 1.5 == pytest.approx(left.motion_t[int(np.argmax(left.motion))], abs=0.11)


def test_series_drive_the_voice_to_face_mapping():
    """decide_mapping with precomputed face-region series: cluster 0 talks when the LEFT
    mouth moves -> cluster 0 is Speaker A; swapped seats -> swapped; noise -> a guess."""
    from podcast_autoeditor.config import MappingConfig
    from podcast_autoeditor.mapping import decide_mapping

    fps = 5.0
    times = np.arange(300) / fps
    segs = [SpeakerSegment(float(t), float(t + 4), "0" if (t // 4) % 2 == 0 else "1") for t in range(0, 60, 4)]
    act0 = np.array([1.0 if (int(x) // 4) % 2 == 0 else 0.0 for x in times])
    rng = np.random.default_rng(0)
    left = act0 + 0.3 * rng.random(300)
    right = (1 - act0) + 0.3 * rng.random(300)
    res = decide_mapping(segs, MappingConfig(), visual_inputs={"series": {"times": times, "motion_a": left,
                                                                           "motion_b": right, "info": {}}})
    assert res.method == "visual" and res.confident and res.mapping == {"0": "A", "1": "B"}
    res = decide_mapping(segs, MappingConfig(), visual_inputs={"series": {"times": times, "motion_a": right,
                                                                           "motion_b": left, "info": {}}})
    assert res.mapping == {"0": "B", "1": "A"}
    noise = {"times": times, "motion_a": rng.random(300), "motion_b": rng.random(300), "info": {}}
    res = decide_mapping(segs, MappingConfig(), visual_inputs={"series": noise})
    assert not res.confident and res.method == "first-speaker-guess"
    res = decide_mapping(segs, MappingConfig(speaker_map="BA"), visual_inputs={"series": noise})
    assert res.confident and res.mapping == {"0": "B", "1": "A"}
    # one face only: its mouth still identifies its own cluster
    res = decide_mapping(segs, MappingConfig(), visual_inputs={"series": {"times": times, "motion_a": None,
                                                                           "motion_b": right, "info": {}}})
    assert res.confident and res.mapping == {"0": "A", "1": "B"}


# ----------------------------------------------------------------------------- crop geometry

def _check(crop, fw=FW, fh=FH):
    assert crop.x % 2 == 0 and crop.y % 2 == 0 and crop.w % 2 == 0 and crop.h % 2 == 0
    assert crop.x >= 0 and crop.y >= 0 and crop.x + crop.w <= fw and crop.y + crop.h <= fh


def test_crop_is_head_and_shoulders_with_eyes_on_the_upper_third():
    cfg = ReframeConfig()
    crop, info = compute_crop((960, 500, 120, 150, 485), FW, FH, 1920, 1080, cfg)
    _check(crop)
    assert crop.w * 9 == crop.h * 16                       # exact 16:9
    assert 3.0 * 150 <= crop.h <= 3.5 * 150                # crop_scale 3.2 x face height
    assert abs((crop.x + crop.w / 2) - 960) <= 2           # centred on the face
    assert abs((485 - crop.y) / crop.h - 1 / 3) < 0.02     # eyes on the upper third
    assert info["face_inside"] and info["zoom"] == pytest.approx(1080 / crop.h, abs=0.01)
    headroom = (500 - 75) - crop.y                          # above the face box
    assert 0.2 * 150 < headroom < 1.0 * 150


def test_crop_clamps_at_the_frame_edges():
    cfg = ReframeConfig()
    for cx, cy in ((40, 60), (1900, 1040), (30, 1000), (1890, 80)):
        crop, _ = compute_crop((cx, cy, 120, 150, cy - 15), FW, FH, 1920, 1080, cfg)
        _check(crop)
        assert crop.w * 9 == crop.h * 16
    crop, _ = compute_crop((40, 60, 120, 150, 45), FW, FH, 1920, 1080, cfg)
    assert crop.x == 0 and crop.y == 0


def test_crop_excludes_the_other_face():
    cfg = ReframeConfig()
    # two people 620 px apart, faces 170 px tall: a centred 3.2x crop (967 px wide) would
    # reach the other face -> the crop shifts away / tightens instead
    a = (650, 500, 140, 170, 480)
    b = (1270, 500, 140, 170, 480)
    box = lambda f: (f[0] - f[2] / 2, f[1] - f[3] / 2, f[0] + f[2] / 2, f[1] + f[3] / 2)  # noqa: E731
    ca, ia = compute_crop(a, FW, FH, 1920, 1080, cfg, box(b))
    cb, ib = compute_crop(b, FW, FH, 1920, 1080, cfg, box(a))
    for c in (ca, cb):
        _check(c)
    assert ca.x + ca.w <= box(b)[0] and not ia["includes_other"]
    assert cb.x >= box(a)[2] and not ib["includes_other"]
    assert ia["face_inside"] and ib["face_inside"]
    assert ca.h >= cfg.min_crop_scale * 170 - 18           # tightened no further than allowed
    # impossibly close: reported, not hidden
    c2, i2 = compute_crop((900, 500, 140, 170, 480), FW, FH, 1920, 1080, cfg, (960, 415, 1100, 585))
    assert i2["includes_other"]


def test_max_zoom_caps_the_upscaling_and_small_faces_warn():
    tiny = (960, 500, 40, 50, 495)                          # 50 px face -> 160 px crop -> 6.75x
    crop, info = compute_crop(tiny, FW, FH, 1920, 1080, ReframeConfig(max_zoom=3.0))
    assert info["zoom"] <= 3.0 + 1e-6 and crop.h >= 360
    crop2, info2 = compute_crop(tiny, FW, FH, 1920, 1080, ReframeConfig(max_zoom=1.5))
    assert info2["zoom"] <= 1.5 + 1e-6 and crop2.h >= 720
    crop3, info3 = compute_crop(tiny, FW, FH, 1920, 1080, ReframeConfig(max_zoom=0))   # no cap
    assert info3["zoom"] > 5
    t = PersonTrack("left", 500, 500, 40, 50, 495, hits=10, persistence=1.0)
    crops, info, warns = base_crops({"A": t, "B": None}, FW, FH, 1920, 1080, ReframeConfig())
    assert crops["B"] is None and any("upscaled" in w and "4K" in w for w in warns)


def test_portrait_output_gets_a_portrait_crop():
    cfg = ReframeConfig()
    crop, info = compute_crop((600, 500, 120, 150, 485), FW, FH, 1080, 1920, cfg)
    _check(crop)
    assert crop.w * 16 == crop.h * 9                        # exact 9:16
    # a 480 px crop would be a 4x upscale to 1920: the default max_zoom=3 widens it to 640
    assert crop.h == 640 and info["zoom"] == pytest.approx(3.0)
    free, _ = compute_crop((600, 500, 120, 150, 485), FW, FH, 1080, 1920, ReframeConfig(max_zoom=0))
    assert free.w * 16 == free.h * 9 and free.h == pytest.approx(3.2 * 150, abs=32)
    crop4k, _ = compute_crop((1200, 1000, 240, 300, 970), 3840, 2160, 1080, 1920, cfg)
    _check(crop4k, 3840, 2160)
    assert crop4k.w * 16 == crop4k.h * 9
    # odd aspect ratio: approximate but even and inside
    w, h = snap_size(700.0, 393.0, FW, FH, 1000, 562)
    assert w % 2 == 0 and h % 2 == 0 and abs(w / h - 1000 / 562) < 0.01


def test_manual_crops_are_parsed_and_fitted():
    assert parse_crop("100,50,640,360") == CropRect(100, 50, 640, 360)
    assert parse_crop(None) is None
    for bad in ("1,2,3", "a,b,c,d", "0,0,-5,10"):
        with pytest.raises(ValueError):
            parse_crop(bad)
    c, changed = fit_manual_crop(CropRect(101, 51, 641, 361), FW, FH, 1920, 1080)
    _check(c)
    assert changed and c.w * 9 == c.h * 16
    c, changed = fit_manual_crop(CropRect(100, 50, 640, 360), FW, FH, 1920, 1080)
    assert c == CropRect(100, 50, 640, 360) and not changed
    crops, info, warns = base_crops({"A": None, "B": None}, FW, FH, 1920, 1080,
                                    ReframeConfig(crop_a="100,50,640,360"))
    assert crops["A"] == CropRect(100, 50, 640, 360) and info["A"]["manual"] and crops["B"] is None


# ----------------------------------------------------------------------------- per-shot crops

def _moving_track(move_at=50.0, dx=260.0):
    t = np.arange(0, 100, 0.5)
    xs = np.where(t < move_at, 1400.0, 1400.0 + dx)
    tr = PersonTrack("right", float(np.median(xs)), 500, 120, 150, 485, hits=len(t), persistence=1.0,
                     times=list(t), xs=list(xs), ys=[500.0] * len(t), hs=[150.0] * len(t),
                     eyes=[485.0] * len(t))
    return tr


def test_reframing_happens_only_at_shot_boundaries():
    cfg = ReframeConfig()
    b = _moving_track()
    a = PersonTrack("left", 500, 500, 120, 150, 485, hits=200, persistence=1.0,
                    times=list(np.arange(0, 100, 0.5)), xs=[500.0] * 200, ys=[500.0] * 200,
                    hs=[150.0] * 200, eyes=[485.0] * 200)
    people = {"A": a, "B": b}
    b.cx = 1400.0                                           # locked-off crop from the early position
    crops, _, _ = base_crops(people, FW, FH, 1920, 1080, cfg)
    shots = [Shot(0, 10, WIDE, "", 0, 300), Shot(10, 30, CAM_B, "", 300, 900),
             Shot(30, 40, CAM_A, "", 900, 1200), Shot(40, 60, CAM_B, "", 1200, 1800),   # B moves at 50 s
             Shot(60, 70, CAM_A, "", 1800, 2100), Shot(70, 90, CAM_B, "", 2100, 2700),
             Shot(90, 100, CAM_A, "", 2700, 3000)]
    per, n = shot_crops(shots, people, crops, FW, FH, 1920, 1080, cfg)
    assert per[0] is None                                   # WIDE: full frame
    assert per[1] == crops["B"]                             # before the move: locked-off crop
    assert per[3] == crops["B"]                             # move happens MID-shot: crop unchanged
    assert per[5] != crops["B"] and n == 1                  # next B shot: re-framed once
    assert abs((per[5].x + per[5].w / 2) - 1660) < 40 or per[5].x + per[5].w == FW
    assert per[2] == per[4] == per[6] == crops["A"]         # A never moved: one stable crop
    for c in per:
        if c is not None:
            _check(c)
    # --static-crops: never re-frame
    per2, n2 = shot_crops(shots, people, crops, FW, FH, 1920, 1080, ReframeConfig(per_shot=False))
    assert n2 == 0 and per2[5] == crops["B"]
    # manual crop for B: never re-framed either
    cfgm = ReframeConfig(crop_b="1000,200,640,360")
    cm, _, _ = base_crops(people, FW, FH, 1920, 1080, cfgm)
    per3, n3 = shot_crops(shots, people, cm, FW, FH, 1920, 1080, cfgm)
    assert n3 == 0 and per3[5] == CropRect(1000, 200, 640, 360)


def test_small_jitter_never_reframes():
    rng = np.random.default_rng(1)
    t = np.arange(0, 60, 0.5)
    tr = PersonTrack("left", 500, 500, 120, 150, 485, hits=len(t), persistence=1.0, times=list(t),
                     xs=list(500 + rng.normal(0, 8, len(t))), ys=list(500 + rng.normal(0, 6, len(t))),
                     hs=[150.0] * len(t), eyes=list(485 + rng.normal(0, 6, len(t))))
    crops, _, _ = base_crops({"A": tr, "B": None}, FW, FH, 1920, 1080, ReframeConfig())
    shots = [Shot(i * 5.0, i * 5.0 + 5, CAM_A, "", i * 150, i * 150 + 150) for i in range(12)]
    per, n = shot_crops(shots, {"A": tr, "B": None}, crops, FW, FH, 1920, 1080, ReframeConfig())
    assert n == 0 and all(c == crops["A"] for c in per)


# ----------------------------------------------------------------------------- mode selection

def test_mode_selection():
    from podcast_autoeditor.pipeline import PipelineError, resolve_mode

    assert resolve_mode(PipelineConfig(wide="w.mp4")) == "single"
    assert resolve_mode(PipelineConfig(cam_a="a.mp4", cam_b="b.mp4", wide="w.mp4")) == "multicam"
    assert resolve_mode(PipelineConfig(mode="single", cam_a="a.mp4", cam_b="b.mp4", wide="w.mp4")) == "single"
    with pytest.raises(PipelineError, match="--cam-b is missing"):
        resolve_mode(PipelineConfig(cam_a="a.mp4", wide="w.mp4"))
    with pytest.raises(PipelineError, match="required in multicam"):
        resolve_mode(PipelineConfig(mode="multicam", wide="w.mp4"))
    with pytest.raises(PipelineError, match="--wide is required"):
        resolve_mode(PipelineConfig(cam_a="a.mp4", cam_b="b.mp4"))
    with pytest.raises(PipelineError):
        resolve_mode(PipelineConfig(mode="triple", wide="w.mp4"))


def test_cli_selects_single_mode_when_only_wide_is_given():
    from podcast_autoeditor.cli import build_parser, config_from_args
    from podcast_autoeditor.pipeline import resolve_mode

    p = build_parser()
    cfg = config_from_args(p.parse_args(["--wide", "w.mp4", "--out", "o.mp4", "--a-side", "right",
                                         "--crop-scale", "3.4", "--max-zoom", "2", "--crop-b", "0,0,640,360",
                                         "--static-crops", "--preview-crops", "p.png"]))
    assert resolve_mode(cfg) == "single"
    r = cfg.reframe
    assert (r.a_side, r.crop_scale, r.max_zoom, r.crop_b, r.per_shot, r.preview) == \
        ("right", 3.4, 2.0, "0,0,640,360", False, "p.png")
    cfg = config_from_args(p.parse_args(["--cam-a", "a.mp4", "--cam-b", "b.mp4", "--wide", "w.mp4",
                                         "--out", "o.mp4"]))
    assert resolve_mode(cfg) == "multicam" and cfg.reframe.a_side == "left"
    cfg = config_from_args(p.parse_args(["--cam-a", "a.mp4", "--cam-b", "b.mp4", "--wide", "w.mp4",
                                         "--out", "o.mp4", "--mode", "single"]))
    assert resolve_mode(cfg) == "single"


# ----------------------------------------------------------------------------- a person who moves

def _two_with_move(n=120, move_at=60, dx=200.0, seed=2):
    """Left person fixed at 500; right person at 1400, then 1400+dx from sample ``move_at``."""
    rng = np.random.default_rng(seed)
    out = []
    for i in range(n):
        xr = 1400.0 + (dx if i >= move_at else 0.0)
        out.append({"t": i * 0.5, "dense": True,
                    "faces": [face(500 + rng.normal(0, 3)), face(xr + rng.normal(0, 3), 520)]})
    return out


def test_person_who_moves_stays_one_track():
    # the jump (200 px) exceeds the clustering gap (0.8 x 120 px): without the co-occurrence
    # merge this would look like a third person and half of the track would be dropped
    tracks = build_tracks(_two_with_move(), FW)
    assert len(tracks) == 2
    people, warns = select_people(tracks, FW, FH, ReframeConfig())
    assert not warns and people["B"].hits == 120 and people["B"].persistence == pytest.approx(1.0)
    x0, _, x1, _ = people["B"].extent_box()
    assert x0 < 1400 - 50 and x1 > 1600 + 50               # covers both seats


def test_two_people_sitting_close_are_not_merged():
    rng = np.random.default_rng(4)
    s = [{"t": i * 0.5, "dense": True,
          "faces": [face(800 + rng.normal(0, 3)), face(1050 + rng.normal(0, 3))]} for i in range(80)]
    tracks = build_tracks(s, FW)                            # 250 px apart, always seen together
    assert len(tracks) == 2 and all(t.hits == 80 for t in tracks)


def test_locked_off_crop_avoids_everywhere_the_other_person_sits():
    s = _two_with_move(dx=-330.0)                           # the right person moves TOWARDS the left one
    people, _ = select_people(build_tracks(s, FW), FW, FH, ReframeConfig())
    crops, info, _ = base_crops(people, FW, FH, 1920, 1080, ReframeConfig())
    b_min_x0 = min(1400, 1400 - 330) - 60                   # leftmost edge B's face ever has
    assert crops["A"].x + crops["A"].w <= b_min_x0 + 2 and not info["A"]["includes_other"]


def test_shot_crops_follow_a_moved_person_and_keep_the_other_out():
    s = _two_with_move(n=240, move_at=160, dx=360.0)        # B moves at t=80 s (during A's shot)
    people, _ = select_people(build_tracks(s, FW), FW, FH, ReframeConfig())
    cfg = ReframeConfig()
    crops, _, _ = base_crops(people, FW, FH, 1920, 1080, cfg)
    shots = [Shot(0, 20, CAM_B, "", 0, 600), Shot(20, 40, CAM_A, "", 600, 1200),
             Shot(40, 60, CAM_B, "", 1200, 1800), Shot(60, 90, CAM_A, "", 1800, 2700),
             Shot(90, 110, CAM_B, "", 2700, 3300), Shot(110, 120, CAM_A, "", 3300, 3600)]
    per, n = shot_crops(shots, people, crops, FW, FH, 1920, 1080, cfg)
    centre = lambda c: c.x + c.w / 2.0                      # noqa: E731
    early, late = per[0], per[4]
    assert early == per[2] == crops["B"]                    # seat 1: the locked-off crop, twice
    assert early.contains(1340, 445, 1460, 595)             # B's face at seat 1 is inside it
    assert late != early and n == 1                         # after the move: re-framed once
    assert late.contains(1700, 445, 1820, 595)              # ... around seat 2
    assert centre(late) >= centre(early) + 100              # (clamped at the right frame edge)
    assert late.x + late.w == FW
    assert per[1] == per[3] == per[5] == crops["A"]         # A's crop never changes
    assert crops["A"].x + crops["A"].w <= 1340              # and never shows B's face
    for c in per:
        _check(c)


# ----------------------------------------------------------------------------- reviewer B, round 3

def _track(cx, h, motion_mean, n=120, cy=460.0):
    rng = np.random.default_rng(int(cx))
    mo = list(np.abs(rng.normal(motion_mean, motion_mean / 4 + 1e-6, n)))
    return PersonTrack(side="left", cx=float(cx), cy=cy, w=0.8 * h, h=float(h), eye_y=cy - 0.1 * h, hits=n,
                       persistence=1.0, times=[i * 0.2 for i in range(n)], xs=[float(cx)] * n, ys=[cy] * n,
                       hs=[float(h)] * n, eyes=[cy - 0.1 * h] * n, motion_t=[i * 0.2 for i in range(n)], motion=mo)


def test_large_static_poster_face_is_not_picked_as_a_speaker():
    """Regression: a framed photo on the wall (bigger than the two people, in every frame,
    mouth never moving: mean motion 0.010 vs 0.089/0.095) was ranked first by
    persistence x size and became Speaker A - A's close-up showed the poster."""
    left, poster, right = _track(577, 142, 0.089), _track(968, 255, 0.0097, cy=229.0), _track(1343, 147, 0.095)
    people, warns = select_people([left, poster, right], FW, FH, ReframeConfig())
    assert people["A"] is left and people["B"] is right
    assert any("static" in w for w in warns)
    # one talker + a still listener + a poster: the talker must be kept (the others never get a close-up)
    talker, listener, poster = _track(1343, 147, 0.18), _track(577, 142, 0.023), _track(968, 255, 0.0097, cy=229.0)
    people, _ = select_people([listener, poster, talker], FW, FH, ReframeConfig())
    assert talker in (people["A"], people["B"])
    # without mouth-motion data the old size ranking still applies
    for t in (left, poster, right):
        t.motion, t.motion_t = [], []
    people, _ = select_people([left, poster, right], FW, FH, ReframeConfig())
    assert poster in (people["A"], people["B"])


def test_module_entry_point_keeps_exit_codes_and_output(tmp_path):
    """Regression (reviewer B): `python -m podcast_autoeditor` intermittently died with
    exit code 134 in native-library teardown after a successful run; __main__ now flushes
    and exits without teardown - exit codes and printed output must be unchanged."""
    import os
    import subprocess
    import sys

    root = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
    r = subprocess.run([sys.executable, "-m", "podcast_autoeditor", "--version"], cwd=root,
                       capture_output=True, text=True)
    assert r.returncode == 0 and r.stdout.strip()
    r = subprocess.run([sys.executable, "-m", "podcast_autoeditor", "--wide", str(tmp_path / "missing.mp4"),
                        "--out", str(tmp_path / "o.mp4")], cwd=root, capture_output=True, text=True)
    assert r.returncode == 1 and "file not found" in r.stderr
    r = subprocess.run([sys.executable, "-m", "podcast_autoeditor", "--mode", "multicam", "--wide",
                        str(tmp_path / "missing.mp4"), "--out", str(tmp_path / "o.mp4")], cwd=root,
                       capture_output=True, text=True)
    assert r.returncode == 1 and "--cam-a is required" in r.stderr
