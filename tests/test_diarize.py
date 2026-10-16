"""Diarization plumbing tests with synthetic 'voices' (MFCC backend; no model downloads)."""

import logging
import sys

import numpy as np
import soundfile as sf

from podcast_autoeditor.config import DiarizationConfig
from podcast_autoeditor.datatypes import Word
from podcast_autoeditor.diarize import (_choose_backend, cluster_two, diarize, make_windows,
                                        phrase_chunks, speech_regions_from_words)
from podcast_autoeditor.datatypes import Interval

SR = 16000


def voice(f0: float, dur: float, bright: float, seed: int) -> np.ndarray:
    """Harmonic 'voice' with vibrato and a spectral tilt; distinct per speaker."""
    rng = np.random.default_rng(seed)
    t = np.arange(int(dur * SR)) / SR
    f = f0 * (1 + 0.03 * np.sin(2 * np.pi * 5 * t))
    ph = 2 * np.pi * np.cumsum(f) / SR
    x = sum((bright ** k) * np.sin(k * ph) for k in range(1, 15))
    env = 0.5 + 0.5 * np.sin(2 * np.pi * 3 * t + rng.uniform(0, 6)) ** 2
    return (0.1 * x * env + 0.002 * rng.standard_normal(t.size)).astype(np.float32)


def test_mfcc_diarization_separates_two_synthetic_speakers(tmp_path):
    # A: low pitch, dark; B: high pitch, bright. Alternating 2-3 s turns with 0.4 s pauses.
    plan = [("A", 2.5), ("B", 2.0), ("A", 3.0), ("B", 2.5), ("A", 2.0), ("B", 3.0)]
    x = np.zeros(int(1.0 * SR), np.float32)
    words, truth = [], []
    t = 1.0
    for i, (spk, d) in enumerate(plan):
        seg = voice(110 if spk == "A" else 240, d, 0.55 if spk == "A" else 0.85, i)
        x = np.concatenate([x, seg, np.zeros(int(0.4 * SR), np.float32)])
        # fake ASR words every 0.4 s
        k = t
        while k + 0.35 <= t + d:
            words.append(Word("w", k, k + 0.35))
            k += 0.4
        truth.append((spk, t, t + d))
        t += d + 0.4
    p = tmp_path / "x.wav"
    sf.write(p, x, SR)
    res = diarize(str(p), words, DiarizationConfig(backend="mfcc"))
    assert res.backend == "mfcc"
    # cluster "0" must be the first speaker heard (A)
    lab = {"0": "A", "1": "B"}
    correct = total = 0
    for s in res.segments:
        mid = 0.5 * (s.start + s.end)
        g = next((spk for spk, a, b in truth if a <= mid <= b), None)
        if g:
            total += 1
            correct += lab[s.label] == g
    assert total > 0 and correct / total >= 0.9, (correct, total)


def test_auto_backend_falls_back_to_mfcc_without_speechbrain(monkeypatch, caplog):
    import podcast_autoeditor.diarize as dz

    monkeypatch.setitem(sys.modules, "speechbrain", None)
    monkeypatch.setitem(sys.modules, "speechbrain.inference", None)
    monkeypatch.setitem(sys.modules, "speechbrain.inference.speaker", None)
    monkeypatch.setitem(sys.modules, "speechbrain.pretrained", None)
    monkeypatch.setattr(dz, "_ECAPA_CACHE", {})
    with caplog.at_level(logging.WARNING):
        emb, notes = _choose_backend(DiarizationConfig(backend="auto"))
    assert emb.name == "mfcc" and notes and "fallback" in notes[0]


def test_cluster_two_on_separable_embeddings():
    rng = np.random.default_rng(0)
    a = rng.normal([1, 0, 0], 0.1, (30, 3))
    b = rng.normal([0, 1, 0], 0.1, (20, 3))
    labels, C = cluster_two(np.vstack([a, b]))
    assert len(set(labels[:30])) == 1 and len(set(labels[30:])) == 1 and labels[0] != labels[-1]


def test_windows_and_phrases():
    ws = [Word("Hi.", 0.0, 0.3), Word("how", 0.35, 0.5), Word("are", 0.5, 0.7), Word("you?", 0.7, 1.0),
          Word("fine", 2.0, 2.4)]
    regs = speech_regions_from_words(ws, merge_gap=0.15)
    assert len(regs) == 2
    wins = make_windows([Interval(0, 5.0)], 1.5, 0.5, 0.6, 10)
    assert wins[0] == (0, 1.5) and wins[-1][1] == 5.0
    ph = phrase_chunks(ws, 0.15, 3.0)
    assert ph[0] == (0.0, 0.3) and ph[1] == (0.35, 1.0) and ph[2] == (2.0, 2.4)


def test_speech_running_to_end_of_audio_does_not_crash(tmp_path):
    """Regression: a word ending at the last sample (region padded past the end) made the
    frame-vote slice shorter than the window weights -> numpy broadcast ValueError."""
    x = np.concatenate([np.zeros(SR, np.float32), voice(110, 4.0, 0.55, 0),
                        np.zeros(int(0.4 * SR), np.float32), voice(240, 4.65, 0.85, 1)])
    assert x.size / SR == 10.05
    p = tmp_path / "end.wav"
    sf.write(p, x, SR)
    words = []
    for t0, t1 in ((1.0, 5.0), (5.4, 9.65)):
        t = t0
        while t + 0.35 <= t1:
            words.append(Word("w", t, t + 0.35))
            t += 0.4
    words.append(Word("end.", t, 10.05))  # speech continues until the very end of the file
    res = diarize(str(p), words, DiarizationConfig(backend="mfcc"))
    assert res.segments and res.segments[-1].end <= 10.05 + 1e-6


def test_embedding_verdict_rules():
    from podcast_autoeditor.diarize import embedding_verdict

    cfg = DiarizationConfig()
    two = {"centroid_cos": 0.25, "window_counts": [20, 18], "speech_s": {"0": 10.0, "1": 9.0}}
    assert embedding_verdict(two, "ecapa", cfg)[0] == 2
    one = dict(two, centroid_cos=0.80)
    v, why = embedding_verdict(one, "ecapa", cfg)
    assert v == 1 and "centroid cosine" in why[0]
    tiny = dict(two, speech_s={"0": 20.0, "1": 0.4})
    assert embedding_verdict(tiny, "ecapa", cfg)[0] == 1
    assert embedding_verdict(dict(two, window_counts=[30, 1]), "ecapa", cfg)[0] == 1
    # MFCC: centroid cosine is meaningless (standardised features) - silhouette decides
    m2 = {"centroid_cos": -0.99, "silhouette": 0.40, "window_counts": [9, 9], "speech_s": {"0": 5, "1": 5}}
    assert embedding_verdict(m2, "mfcc", cfg)[0] == 2
    assert embedding_verdict(dict(m2, silhouette=0.12), "mfcc", cfg)[0] == 1


def test_cluster_stats_one_voice_vs_two():
    from podcast_autoeditor.diarize import cluster_stats

    rng = np.random.default_rng(0)
    base = rng.normal(size=32)
    one = base + 0.6 * rng.normal(size=(40, 32))            # one "voice": one blob
    labels, C = cluster_two(one)
    s1 = cluster_stats(one, labels, C, "ecapa")
    other = rng.normal(size=32)
    two = np.vstack([base + 0.6 * rng.normal(size=(20, 32)), other + 0.6 * rng.normal(size=(20, 32))])
    labels, C = cluster_two(two)
    s2 = cluster_stats(two, labels, C, "ecapa")
    assert s1["centroid_cos"] > 0.6 > s2["centroid_cos"]


def test_num_speakers_one_puts_everything_on_one_label(tmp_path):
    plan = [("A", 2.0), ("B", 2.0), ("A", 2.0)]
    x = np.zeros(int(0.5 * SR), np.float32)
    words = []
    t = 0.5
    for i, (spk, d) in enumerate(plan):
        x = np.concatenate([x, voice(110 if spk == "A" else 240, d, 0.55 if spk == "A" else 0.85, i),
                            np.zeros(int(0.4 * SR), np.float32)])
        k = t
        while k + 0.35 <= t + d:
            words.append(Word("w", k, k + 0.35))
            k += 0.4
        t += d + 0.4
    p = tmp_path / "x.wav"
    sf.write(p, x, SR)
    res = diarize(str(p), words, DiarizationConfig(backend="mfcc", num_speakers="1"))
    assert {s.label for s in res.segments} == {"0"} and res.stats["verdict"] == 1
    res2 = diarize(str(p), words, DiarizationConfig(backend="mfcc", num_speakers="auto"))
    assert {s.label for s in res2.segments} == {"0", "1"} and "embedding_verdict" in res2.stats
