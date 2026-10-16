from podcast_autoeditor.config import LabelConfig
from podcast_autoeditor.datatypes import SpeakerSegment, Word
from podcast_autoeditor.labeling import (apply_mapping, assign_words, build_turns,
                                         overlaps_from_words, smooth_flips)


def W(t, s, e, spk=None):
    return Word(t, s, e, 1.0, spk)


def test_assign_by_max_overlap():
    segs = [SpeakerSegment(0.0, 2.0, "A"), SpeakerSegment(2.0, 4.0, "B")]
    words = [W("hello", 0.1, 0.5), W("straddle", 1.8, 2.5), W("world", 3.0, 3.5),
             W("late", 4.3, 4.6), W("far", 9.0, 9.2)]
    out = assign_words(words, segs)
    assert [w.speaker for w in out] == ["A", "B", "B", "B", "B"]
    # 'straddle' overlaps A by 0.2 s and B by 0.5 s -> B ; 'late' nearest seg -> B ;
    # 'far' has no segment within 1 s -> previous word's speaker


def test_assign_straddle_majority_a():
    segs = [SpeakerSegment(0.0, 2.0, "A"), SpeakerSegment(2.0, 4.0, "B")]
    out = assign_words([W("x", 1.5, 2.1)], segs)
    assert out[0].speaker == "A"


def test_apply_mapping():
    segs = [SpeakerSegment(0, 1, "0"), SpeakerSegment(1, 2, "1")]
    assert [s.label for s in apply_mapping(segs, {"0": "B", "1": "A"})] == ["B", "A"]


def test_isolated_flip_is_smoothed():
    ws = [W("a", 0.0, 0.3, "A"), W("b", 0.35, 0.6, "A"), W("c", 0.65, 0.9, "B"),
          W("d", 0.95, 1.2, "A"), W("e", 1.25, 1.5, "A")]
    out = smooth_flips(ws, LabelConfig())
    assert [w.speaker for w in out] == ["A"] * 5


def test_real_one_word_answer_with_pauses_survives():
    ws = [W("Ready?", 0.0, 0.5, "A"), W("Yes.", 1.2, 1.5, "B"), W("Go", 2.2, 2.5, "A"),
          W("now", 2.55, 2.8, "A")]
    out = smooth_flips(ws, LabelConfig())
    assert [w.speaker for w in out] == ["A", "B", "A", "A"]


def test_build_turns_splits_on_speaker_and_gap():
    ws = [W("a", 0.0, 0.3, "A"), W("b", 0.4, 0.7, "A"), W("c", 2.0, 2.3, "A"),
          W("d", 2.5, 2.8, "B"), W("e", 2.9, 3.2, "B")]
    turns = build_turns(ws, turn_gap=1.0)
    assert [(t.speaker, len(t.words)) for t in turns] == [("A", 2), ("A", 1), ("B", 2)]
    assert turns[0].text == "a b"


def test_overlaps_from_words():
    ws = [W("long", 0.0, 2.0, "A"), W("inter", 1.0, 1.6, "B"), W("x", 3.0, 3.1, "B")]
    ov = overlaps_from_words(ws, min_dur=0.3)
    assert len(ov) == 1 and abs(ov[0].start - 1.0) < 1e-9 and abs(ov[0].end - 1.6) < 1e-9


# ----------------------------------------------------------------------------- mic-coupling overlap

import numpy as np  # noqa: E402

from podcast_autoeditor.labeling import mic_coupling_overlaps, overlaps_from_mic_coupling  # noqa: E402


def _levels(a_talk, b_talk, bleed_db, gain_b_db=0.0, a_db=-20.0, b_db=-20.0, floor=-60.0):
    """Frame levels of two close mics from per-frame talk masks (power sum in dB)."""
    pa = np.where(a_talk, 10 ** (a_db / 10), 0) + np.where(b_talk, 10 ** ((b_db + bleed_db) / 10), 0)
    pb = np.where(b_talk, 10 ** (b_db / 10), 0) + np.where(a_talk, 10 ** ((a_db + bleed_db) / 10), 0)
    pa += 10 ** (floor / 10)
    pb += 10 ** (floor / 10)
    return 10 * np.log10(pa), 10 * np.log10(pb) + gain_b_db


def test_mic_coupling_finds_overlap_and_ignores_single_speech():
    hop = 0.05
    n = 400  # 20 s
    a = np.zeros(n, bool)
    b = np.zeros(n, bool)
    a[20:160] = True          # A 1-8 s
    b[180:340] = True         # B 9-17 s
    a[240:270] = True         # A interjects 12.0-13.5 s over B
    for bleed in (-15.0, -9.0):
        for gain in (0.0, -12.0):   # mic B recorded 12 dB quieter: must not matter
            da, db = _levels(a, b, bleed, gain)
            a_only = a & ~b
            b_only = b & ~a
            ov, info = mic_coupling_overlaps(da, db, a_only, b_only, hop)
            assert ov is not None, info
            assert len(ov) == 1, (bleed, gain, ov, info)
            assert abs(ov[0].start - 12.0) <= 0.1 and abs(ov[0].end - 13.5) <= 0.1


def test_mic_coupling_unavailable_without_separation_or_data():
    hop = 0.05
    n = 400
    a = np.zeros(n, bool)
    b = np.zeros(n, bool)
    a[20:160] = True
    b[180:340] = True
    da, db = _levels(a, b, -1.0)              # bleed almost as loud as own voice
    ov, info = mic_coupling_overlaps(da, db, a, b, hop)
    assert ov is None and "not separated" in info["reason"]
    ov, info = mic_coupling_overlaps(da, db, a & False, b, hop)
    assert ov is None and "single-speaker" in info["reason"]


def test_overlaps_from_mic_coupling_on_audio(tmp_path):
    import soundfile as sf

    from podcast_autoeditor.datatypes import Word

    SR = 16000
    rng = np.random.default_rng(1)
    T = 16
    sa = np.zeros(T * SR, np.float32)
    sb = np.zeros(T * SR, np.float32)
    sa[1 * SR:7 * SR] = rng.standard_normal(6 * SR) * 0.2
    sb[8 * SR:15 * SR] = rng.standard_normal(7 * SR) * 0.2
    sa[10 * SR:11 * SR] += rng.standard_normal(SR) * 0.2      # A talks over B at 10-11 s
    nz = lambda: rng.standard_normal(T * SR).astype(np.float32) * 0.001  # noqa: E731
    sf.write(tmp_path / "a.wav", sa + 0.3 * sb + nz(), SR)   # bleed ~ -10 dB
    sf.write(tmp_path / "b.wav", (sb + 0.3 * sa + nz())[SR // 2:], SR)  # cam B starts 0.5 s late
    words = [Word("a", t, t + 0.4, 1.0, "A") for t in np.arange(1.0, 6.6, 0.5)]
    words += [Word("b", t, t + 0.4, 1.0, "B") for t in np.arange(8.0, 14.6, 0.5)]
    ov, info = overlaps_from_mic_coupling(words, str(tmp_path / "a.wav"), str(tmp_path / "b.wav"),
                                          0.0, 0.5, float(T))
    assert ov is not None and len(ov) == 1, (ov, info)
    assert abs(ov[0].start - 10.0) <= 0.15 and abs(ov[0].end - 11.0) <= 0.15


def test_mic_coupling_alternating_syllables_and_latched_turns():
    """Real overlaps alternate syllables (each talker silent ~1/3 of the frames); a turn
    change with no pause (B starts the instant A stops) must not count as overlap."""
    hop = 0.05
    n = 600  # 30 s
    a = np.zeros(n, bool)
    b = np.zeros(n, bool)
    a[20:200] = True                  # A  1-10 s
    b[200:380] = True                 # B 10-19 s, latched onto A (no gap)
    a[380:560] = True                 # A 19-28 s, latched onto B
    syl = (np.arange(n) % 3) != 0     # each talker pauses every 3rd frame
    b[460:500] = syl[460:500]         # B talks over A at 23-25 s, with micro-pauses
    a_talk = a & (syl | (np.arange(n) < 460) | (np.arange(n) >= 500))
    da, db = _levels(a_talk, b, -9.0)
    a_only = a & ~b
    b_only = b & ~a
    a_only[455:505] = False
    b_only[455:505] = False
    ov, info = mic_coupling_overlaps(da, db, a_only, b_only, hop)
    assert ov is not None and len(ov) == 1, (ov, info)
    assert abs(ov[0].start - 23.0) <= 0.3 and abs(ov[0].end - 25.0) <= 0.3
