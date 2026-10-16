import numpy as np

from podcast_autoeditor.transcribe import refine_word_times

SR = 16000


def _audio(spans, total=6.0, seed=0):
    rng = np.random.default_rng(seed)
    x = rng.standard_normal(int(total * SR)).astype(np.float32) * 0.001  # room noise
    for s, e in spans:
        x[int(s * SR):int(e * SR)] += rng.standard_normal(int(e * SR) - int(s * SR)).astype(np.float32) * 0.2
    return x


def test_first_word_after_pause_is_moved_to_speech_onset():
    # previous speaker talks until 1.15; new speaker starts at 2.0; whisper says "That" = 1.0-2.3
    x = _audio([(0.2, 1.15), (2.0, 2.3), (2.35, 3.0)])
    words = [{"word": "hooked.", "start": 0.7, "end": 1.0},
             {"word": "That", "start": 1.0, "end": 2.3, "_seg_first": True},
             {"word": "is", "start": 2.35, "end": 2.6}]
    n = refine_word_times(words, x, SR)
    assert n == 1
    assert abs(words[1]["start"] - 2.0) < 0.06 and words[1]["end"] == 2.3
    assert words[0] == {"word": "hooked.", "start": 0.7, "end": 1.0}


def test_last_word_running_into_pause_is_cut():
    x = _audio([(0.5, 1.4)])
    words = [{"word": "fast.", "start": 1.0, "end": 2.5}]
    refine_word_times(words, x, SR)
    assert abs(words[0]["end"] - 1.4) < 0.08 and words[0]["start"] == 1.0


def test_normal_words_untouched():
    x = _audio([(0.5, 3.0)])
    words = [{"word": "hello", "start": 0.5, "end": 0.9}, {"word": "there", "start": 0.9, "end": 1.3}]
    before = [dict(w) for w in words]
    assert refine_word_times(words, x, SR) == 0 and words == before
