import math

import pytest

from podcast_autoeditor import CAM_A, CAM_B, WIDE
from podcast_autoeditor.config import PlannerConfig
from podcast_autoeditor.datatypes import Interval
from podcast_autoeditor.planner import (find_qa_events, find_rapid_events, plan_shots,
                                        validate_shots)

from helpers import cam_at, turns

FPS = 30.0


def plan(ts, duration, **kw):
    cfg = PlannerConfig(**kw)
    shots = plan_shots(ts, duration, FPS, cfg)
    validate_shots(shots, duration, FPS)
    return shots


def test_monologue_stays_on_closeup():
    ts = turns([("A", 2.0, 30.0, " ".join(["word"] * 60) + ".")])
    shots = plan(ts, 33.0)
    # intro WIDE, then A for the whole monologue, then outro WIDE
    assert [s.camera for s in shots] == [WIDE, CAM_A, WIDE]
    assert cam_at(shots, 10.0) == CAM_A and cam_at(shots, 29.0) == CAM_A
    a = shots[1]
    assert abs(a.start - (2.0 - 0.15)) <= 1 / FPS + 1e-9  # cut lead before speech


def test_speaker_cuts_without_question():
    ts = turns([("A", 0.5, 10.0, "I have been working on this project for a long time now."),
                ("B", 10.6, 20.0, "That sounds great and I really like the direction you took."),
                ("A", 20.6, 30.0, "Thanks a lot I appreciate it very much indeed my friend.")])
    shots = plan(ts, 31.0)
    assert cam_at(shots, 5.0) == CAM_A
    assert cam_at(shots, 15.0) == CAM_B
    assert cam_at(shots, 25.0) == CAM_A
    assert WIDE not in {cam_at(shots, t) for t in (5.0, 15.0, 25.0)}


def test_question_answer_goes_wide():
    ts = turns([("A", 1.0, 8.0, "Welcome back to the show. So what do you think about electric cars?"),
                ("B", 8.8, 20.0, "I think they will dominate the market within a decade or so honestly.")])
    shots = plan(ts, 21.0)
    ev = find_qa_events(ts, PlannerConfig())
    assert len(ev) == 1
    # WIDE covers the question sentence end and the first 2 s of the answer
    assert cam_at(shots, 7.5) == WIDE
    assert cam_at(shots, 9.5) == WIDE
    assert cam_at(shots, 10.5) == WIDE  # 8.8 + 2.0 hold = 10.8
    assert cam_at(shots, 11.5) == CAM_B  # then cut to the answerer
    # the first statement of A (before the question sentence) stays on cam A
    assert cam_at(shots, 2.0) == CAM_A


def test_question_without_answer_gap_too_long_is_not_qa():
    ts = turns([("A", 1.0, 5.0, "What do you think about that?"),
                ("B", 8.0, 15.0, "Well I am not entirely sure about it to be honest.")])
    assert find_qa_events(ts, PlannerConfig(qa_gap=1.5)) == []


def test_question_followed_by_same_speaker_is_not_qa():
    ts = turns([("A", 1.0, 5.0, "What do you think about that?"),
                ("A", 5.2, 9.0, "Actually let me tell you what I think first.")])
    assert find_qa_events(ts, PlannerConfig()) == []


def test_rapid_back_and_forth_goes_wide():
    ts = turns([("A", 0.5, 8.0, "Let me tell you a story about my trip to the mountains."),
                ("B", 8.3, 9.3, "Really now"),
                ("A", 9.6, 10.6, "Yes really"),
                ("B", 10.9, 11.9, "No way"),
                ("A", 12.2, 13.2, "I swear"),
                ("B", 13.5, 25.0, "Okay then tell me everything about it from the very beginning please.")])
    ev = find_rapid_events(ts, PlannerConfig())
    assert ev, "rapid exchange not detected"
    shots = plan(ts, 26.0)
    for t in (9.0, 10.0, 11.0, 12.5):
        assert cam_at(shots, t) == WIDE, t
    assert cam_at(shots, 4.0) == CAM_A
    assert cam_at(shots, 20.0) == CAM_B


def test_slow_exchange_is_not_rapid():
    ts = turns([("A", 0.0, 5.0, "one two three four five six"),
                ("B", 5.5, 10.0, "one two three four five six"),
                ("A", 10.5, 15.0, "one two three four five six"),
                ("B", 15.5, 20.0, "one two three four five six")])
    assert find_rapid_events(ts, PlannerConfig()) == []


def test_min_shot_merging():
    # A short interjection by B (0.4 s) inside A's speech must not create a flash cut
    ts = turns([("A", 0.5, 10.0, "I was saying that the results look good overall"),
                ("B", 10.1, 10.5, "Mhm"),
                ("A", 10.7, 20.0, "and then we will publish the paper next month")])
    shots = plan(ts, 21.0, qa_enabled=False, rapid_enabled=False)
    assert min(s.duration for s in shots) >= 1.2 - 1e-9
    assert cam_at(shots, 10.3) == CAM_A


def test_short_wide_shot_is_grown_not_dropped():
    # Q->A with a very short answer hold still yields a >= min_shot WIDE shot
    ts = turns([("A", 0.0, 6.0, "So tell me. Why?"),
                ("B", 6.2, 15.0, "Because the data supports it and so on and so forth.")])
    shots = plan(ts, 16.0, qa_answer_hold=0.3, qa_max_question=0.5)
    wides = [s for s in shots if s.camera == WIDE and s.start < 10]
    assert wides and all(s.duration >= 1.2 - 1e-9 for s in shots)


@pytest.mark.parametrize("duration", [7.0, 13.37, 61.01])
def test_full_coverage_invariant(duration):
    ts = turns([("A", 1.0, 3.0, "hello there how are you?"),
                ("B", 3.2, 4.0, "fine thanks"),
                ("A", 4.3, 5.0, "great"),
                ("B", 5.2, min(duration - 0.5, 6.5), "okay then")])
    shots = plan(ts, duration)
    assert shots[0].start == 0.0
    assert shots[-1].end == pytest.approx(duration)
    for a, b in zip(shots, shots[1:]):
        assert a.end == pytest.approx(b.start)
        assert a.end_frame == b.start_frame
        assert a.camera != b.camera
        assert abs(a.end * FPS - round(a.end * FPS)) < 1e-6  # on frame grid
    assert sum(s.n_frames for s in shots) == math.ceil(duration * FPS - 1e-6)


def test_long_silence_goes_wide_and_intro_outro_wide():
    ts = turns([("A", 3.0, 8.0, "first part of the talk ends here."),
                ("A", 14.0, 20.0, "second part after a long pause.")])
    shots = plan(ts, 25.0)
    assert cam_at(shots, 1.0) == WIDE      # intro
    assert cam_at(shots, 11.0) == WIDE     # 6 s silence
    assert cam_at(shots, 23.0) == WIDE     # outro
    assert cam_at(shots, 5.0) == CAM_A and cam_at(shots, 17.0) == CAM_A


def test_overlap_goes_wide():
    ts = turns([("A", 0.5, 10.0, "I think we should go with the first option because it is cheaper"),
                ("B", 10.5, 20.0, "but the second one is more reliable in the long run you know")])
    shots = plan_shots(ts, 21.0, FPS, PlannerConfig(), overlaps=[Interval(14.0, 15.5)])
    validate_shots(shots, 21.0, FPS)
    assert cam_at(shots, 14.7) == WIDE
    assert cam_at(shots, 18.0) == CAM_B


def test_camera_unavailable_falls_back_to_wide():
    ts = turns([("A", 0.5, 10.0, "I have been working on this project for a long time now."),
                ("B", 10.6, 20.0, "That sounds great and I really like the direction you took.")])
    # cam B footage only starts at t=15
    shots = plan_shots(ts, 21.0, FPS, PlannerConfig(), coverage={CAM_B: (15.0, 100.0)})
    validate_shots(shots, 21.0, FPS)
    assert cam_at(shots, 12.0) == WIDE
    assert cam_at(shots, 17.0) == CAM_B


def test_empty_turns_all_wide():
    shots = plan([], 10.0)
    assert len(shots) == 1 and shots[0].camera == WIDE and shots[0].reason.startswith("no_speech")


def test_min_shot_merge_does_not_reintroduce_unavailable_camera():
    """Regression: the wide camera starts 0.8 s after the master audio. The availability
    fix put cam B on those frames, but min-shot merging folded that 0.8 s shot back into
    the intro WIDE shot -> 0.8 s of black in the render although cam B had footage."""
    ts = turns([("B", 2.0, 17.0, "Hi everyone and welcome to the show. My guest today installs solar panels."),
                ("A", 17.7, 30.0, "It was almost an accident and I never looked back since then honestly.")])
    cov = {WIDE: (0.8, 100.0), CAM_A: (2.1, 100.0), CAM_B: (-1.7, 100.0)}
    shots = plan_shots(ts, 31.0, FPS, PlannerConfig(), coverage=cov)
    validate_shots(shots, 31.0, FPS)
    assert min(s.duration for s in shots) >= 1.2 - 1e-9
    for s in shots:  # no shot may use a camera outside its footage
        lo, hi = cov[s.camera]
        assert s.start_frame >= math.ceil(lo * FPS - 0.5) and s.end_frame <= math.floor(hi * FPS + 0.5), s
    assert cam_at(shots, 0.3) == CAM_B


def test_rounding_sliver_at_end_keeps_outro_wide():
    """The master audio can be a fraction of a frame longer than the wide video; that
    last partial frame must not flip the whole outro WIDE to a close-up."""
    ts = turns([("B", 1.0, 10.0, "Some words from speaker B for a while here."),
                ("A", 10.5, 20.0, "And some words from speaker A to finish the show.")])
    d = 23.0003
    shots = plan_shots(ts, d, FPS, PlannerConfig(), coverage={WIDE: (0.0, 23.0), CAM_A: (-0.6, 24.6),
                                                               CAM_B: (1.3, 22.0)})
    validate_shots(shots, d, FPS)
    assert shots[-1].camera == WIDE and cam_at(shots, 22.0) == WIDE


# ----------------------------------------------------------------------------- question heuristic

from podcast_autoeditor.datatypes import Word  # noqa: E402
from podcast_autoeditor.planner import question_start  # noqa: E402


def _sent(text):
    return [Word(t, i * 0.3, i * 0.3 + 0.25, 1.0, "A") for i, t in enumerate(text.split())]


@pytest.mark.parametrize("text,start", [
    ("do you drive an electric car yourself", 0),
    ("So what do you think about that.", 1),
    ("And how long does a battery pack last", 1),
    ("Is it hard to install", 0),
    ("what's the biggest change", 0),
    ("It gets really cold there, right", 0),
    ("That is a lot of money isn't it", 0),
    ("You switched three years ago, correct", 0),
    ("I have been doing this for years and what do you think about it", 8),
])
def test_question_heuristic_detects(text, start):
    assert question_start(_sent(text)) == start


@pytest.mark.parametrize("text", [
    "What I mean is the battery lasts.",
    "When I was young we had no solar panels.",
    "I think so.",
    "Do it now!",
    "That is right.",
    "It is cold outside today.",
    "Right.",
])
def test_question_heuristic_rejects(text):
    assert question_start(_sent(text)) is None


def test_question_without_question_mark_goes_wide_when_answered():
    ts = turns([("A", 1.0, 5.0, "do you drive an electric car yourself"),
                ("B", 5.6, 16.0, "I do and I switched three years ago and never looked back since then")])
    shots = plan(ts, 17.0)
    ev = find_qa_events(ts, PlannerConfig())
    assert len(ev) == 1 and "heuristic" in ev[0].detail
    assert cam_at(shots, 4.5) == WIDE and cam_at(shots, 6.5) == WIDE
    assert cam_at(shots, 12.0) == CAM_B


def test_question_heuristic_can_be_disabled():
    ts = turns([("A", 1.0, 5.0, "do you drive an electric car yourself"),
                ("B", 5.6, 16.0, "I do and I switched three years ago and never looked back since then")])
    assert find_qa_events(ts, PlannerConfig(question_heuristic=False)) == []
    shots = plan(ts, 17.0, question_heuristic=False)
    assert cam_at(shots, 4.5) == CAM_A


def test_heuristic_question_needs_an_answer_from_the_other_speaker():
    ts = turns([("A", 1.0, 5.0, "do you drive an electric car yourself"),
                ("B", 8.0, 16.0, "I do and I switched three years ago and never looked back since then")])
    assert find_qa_events(ts, PlannerConfig()) == []  # gap 3 s > qa_gap


def test_late_question_onset_in_unpunctuated_turn_starts_wide_at_the_question():
    ts = turns([("A", 0.0, 9.0, "I have been doing this for years and what do you think about it"),
                ("B", 9.5, 20.0, "honestly I think it is great and everybody should try it at home")])
    ev = find_qa_events(ts, PlannerConfig())
    assert len(ev) == 1
    q_word_start = ts[0].words[8].start  # "what"
    assert abs(ev[0].start - (q_word_start - 0.15)) < 1e-6
    shots = plan(ts, 21.0)
    assert cam_at(shots, 2.0) == CAM_A and cam_at(shots, 8.5) == WIDE


@pytest.mark.parametrize("text,start", [
    ("Tell me how you started.", 0),
    ("So tell us about your first restaurant.", 1),
    ("Walk me through a typical morning in the kitchen.", 0),
])
def test_question_heuristic_detects_interview_prompts(text, start):
    """Regression (reviewer B, round 2): an interviewer's "Tell me how you started." followed
    by the guest's answer is a question->answer exchange, but Whisper writes it with a
    period and the interrogative/auxiliary heuristic missed it (no WIDE shot)."""
    assert question_start(_sent(text)) == start


@pytest.mark.parametrize("text", ["Let me tell you a story.", "Tell you what, I loved it.",
                                  "I will tell them later."])
def test_question_heuristic_prompt_rejects(text):
    assert question_start(_sent(text)) is None


def test_tell_me_prompt_answered_goes_wide():
    ts = turns([("A", 1.0, 9.0, "Welcome to the show my guest is a chef. Tell me how you started."),
                ("B", 9.7, 20.0, "I started in my grandmother's kitchen chopping onions every Sunday morning.")])
    ev = find_qa_events(ts, PlannerConfig())
    assert len(ev) == 1 and "Tell me" in ev[0].detail
    shots = plan(ts, 21.0)
    assert cam_at(shots, 3.0) == CAM_A and cam_at(shots, 8.8) == WIDE and cam_at(shots, 10.5) == WIDE
    assert cam_at(shots, 15.0) == CAM_B
