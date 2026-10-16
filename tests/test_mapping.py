import numpy as np
import soundfile as sf

from podcast_autoeditor.config import MappingConfig
from podcast_autoeditor.datatypes import SpeakerSegment
from podcast_autoeditor.mapping import (decide_from_motion, decide_mapping, mic_level_evidence)

SR = 16000


def _tracks(tmp_path, a_on_cam_a=True, offset_b=0.0):
    rng = np.random.default_rng(0)
    T = 20
    sa = np.zeros(T * SR, np.float32)
    sb = np.zeros(T * SR, np.float32)
    sa[0:5 * SR] = rng.standard_normal(5 * SR) * 0.3      # cluster 0 talks 0-5 and 10-15
    sa[10 * SR:15 * SR] = rng.standard_normal(5 * SR) * 0.3
    sb[5 * SR:10 * SR] = rng.standard_normal(5 * SR) * 0.3  # cluster 1 talks 5-10, 15-20
    sb[15 * SR:] = rng.standard_normal(5 * SR) * 0.3
    near, far = (sa, sb) if a_on_cam_a else (sb, sa)
    cam_a = near + 0.15 * far
    cam_b = 0.5 * (far + 0.15 * near)  # different overall gain on purpose
    k = int(offset_b * SR)
    cam_b = cam_b[k:]
    pa, pb = tmp_path / "a.wav", tmp_path / "b.wav"
    sf.write(pa, cam_a, SR)
    sf.write(pb, cam_b, SR)
    segs = [SpeakerSegment(0, 5, "0"), SpeakerSegment(5, 10, "1"),
            SpeakerSegment(10, 15, "0"), SpeakerSegment(15, 20, "1")]
    return str(pa), str(pb), segs


def test_mic_level_identity(tmp_path):
    pa, pb, segs = _tracks(tmp_path, True, offset_b=1.0)
    ev = mic_level_evidence(segs, pa, pb, 0.0, 1.0)
    assert ev["available"] and ev["score_db"] > 10
    res = decide_mapping(segs, MappingConfig(), mic_inputs={"wav_a": pa, "wav_b": pb,
                                                             "offset_a": 0.0, "offset_b": 1.0})
    assert res.method == "mic" and res.mapping == {"0": "A", "1": "B"} and res.confident


def test_mic_level_swapped(tmp_path):
    pa, pb, segs = _tracks(tmp_path, False)
    res = decide_mapping(segs, MappingConfig(), mic_inputs={"wav_a": pa, "wav_b": pb,
                                                             "offset_a": 0.0, "offset_b": 0.0})
    assert res.mapping == {"0": "B", "1": "A"}


def test_manual_and_swap_and_fallback(tmp_path):
    segs = [SpeakerSegment(0, 5, "0"), SpeakerSegment(5, 10, "1")]
    assert decide_mapping(segs, MappingConfig(speaker_map="BA")).mapping == {"0": "B", "1": "A"}
    r = decide_mapping(segs, MappingConfig(speaker_map="AB", swap=True))
    assert r.mapping == {"0": "B", "1": "A"} and "swapped" in r.method
    r = decide_mapping(segs, MappingConfig(strategy="auto"), mic_inputs={"wav_a": None, "wav_b": None})
    assert r.method == "first-speaker-guess" and not r.confident


def test_motion_decision_rule():
    rng = np.random.default_rng(1)
    t = np.arange(400)
    act0 = ((t // 40) % 2 == 0).astype(float)
    act1 = 1 - act0
    mot_a = act0 * 1.0 + 0.3 * rng.random(400)   # cam A mouth moves when cluster 0 talks
    mot_b = act1 * 1.0 + 0.3 * rng.random(400)
    dec, ev = decide_from_motion(mot_a, mot_b, act0, act1, 0.05)
    assert dec == {"0": "A", "1": "B"} and ev["score"] > 1.0
    dec, _ = decide_from_motion(mot_b, mot_a, act0, act1, 0.05)
    assert dec == {"0": "B", "1": "A"}
    dec, _ = decide_from_motion(None, None, act0, act1, 0.05)
    assert dec is None


def test_visual_series_aligned_by_time_not_index(monkeypatch):
    """Regression: when one close-up starts later than the analysis window (positive sync
    offset) its mouth-motion series starts later; it used to be paired with the other
    camera's time axis by index, i.e. shifted by the offset difference."""
    import podcast_autoeditor.mapping as mp

    fps = 5.0
    # 1 s alternating turns: a 1 s misalignment inverts the activity pattern
    segs = [SpeakerSegment(float(t), float(t + 1), "0" if t % 2 == 0 else "1") for t in range(1, 59)]

    class _Det:
        kind = "fake"
        cv2 = None

    def fake_series(ffmpeg, video, offset, t0, t1, fps_, det):
        clip_start = t0 - offset
        if clip_start < 0:  # same clamping as the real implementation
            t0 -= clip_start
        tt = t0 + np.arange(int((t1 - t0) * fps_)) / fps_
        own = "0" if video == "a" else "1"  # cluster 0 is on cam A
        return tt, mp.activity_series(tt, segs, own) + 0.01 * np.cos(tt), 1.0

    monkeypatch.setattr(mp, "FaceDetector", _Det)
    monkeypatch.setattr(mp, "mouth_motion_series", fake_series)
    dec, ev = mp.visual_evidence("ffmpeg", "a", "b", 1.0, 0.0, segs, MappingConfig(visual_fps=fps))
    assert dec == {"0": "A", "1": "B"}, ev
    assert ev["terms"]["cam_b"]["corr_1"] > 0.9


# ----------------------------------------------------------------------------- single speaker

from podcast_autoeditor.mapping import (decide_single_from_mic, mic_says_single,  # noqa: E402
                                        single_speaker_mic_evidence)


def _mic_ev(d0, d1):
    return {"available": True, "per_cluster": {"0": {"diff_db": d0}, "1": {"diff_db": d1}},
            "score_db": round(d0 - d1, 2)}


def test_mic_says_single_rules():
    # reviewer's 16 s monologue: both clusters +13.97 dB louder on cam A's mic
    assert mic_says_single(_mic_ev(13.97, 13.97), 3.0, 6.0)
    # reviewer's 5 s clip: both on cam B's mic, score -2.16 dB
    assert mic_says_single(_mic_ev(-13.75, -11.59), 3.0, 6.0)
    assert not mic_says_single(_mic_ev(14.0, -14.9), 3.0, 6.0)   # two separated talkers
    assert not mic_says_single(_mic_ev(0.5, -0.3), 3.0, 6.0)     # identical feeds: no info
    assert not mic_says_single(_mic_ev(12.0, 4.0), 3.0, 6.0)     # score 8 dB -> two talkers


def _one_voice(tmp_path, near="a", offset_b=0.0):
    rng = np.random.default_rng(5)
    T = 12
    v = np.zeros(T * SR, np.float32)
    v[1 * SR:5 * SR] = rng.standard_normal(4 * SR) * 0.3
    v[6 * SR:11 * SR] = rng.standard_normal(5 * SR) * 0.3
    noise = lambda: rng.standard_normal(T * SR).astype(np.float32) * 0.002  # noqa: E731
    close, far = v + noise(), 0.15 * v + noise()
    cam_a, cam_b = (close, far) if near == "a" else (far, close)
    pa, pb = tmp_path / "a.wav", tmp_path / "b.wav"
    sf.write(pa, cam_a, SR)
    sf.write(pb, cam_b[int(offset_b * SR):], SR)
    # one voice split into two clusters by a forced 2-way clustering
    segs = [SpeakerSegment(1, 3, "0"), SpeakerSegment(3, 5, "1"), SpeakerSegment(6, 9, "0"),
            SpeakerSegment(9, 11, "1")]
    return str(pa), str(pb), segs


def test_single_speaker_detected_from_mics_and_camera_chosen(tmp_path):
    for near, want in (("a", "A"), ("b", "B")):
        pa, pb, segs = _one_voice(tmp_path, near, offset_b=0.5)
        res = decide_mapping(segs, MappingConfig(), mic_inputs={"wav_a": pa, "wav_b": pb,
                                                                "offset_a": 0.0, "offset_b": 0.5},
                             diar_stats={"embedding_verdict": 2})
        assert res.num_speakers == 1, res
        assert res.mapping == {"0": want, "1": want} and res.confident
        assert res.method == "single-speaker:mic"


def test_single_speaker_from_embeddings_without_mics():
    segs = [SpeakerSegment(0, 5, "0"), SpeakerSegment(5, 10, "1")]
    res = decide_mapping(segs, MappingConfig(), mic_inputs={"wav_a": None, "wav_b": None},
                         diar_stats={"embedding_verdict": 1, "reasons": ["centroid cosine 0.80"]})
    assert res.num_speakers == 1 and res.mapping["0"] == res.mapping["1"] == "A"
    assert not res.confident and "guess" in res.method
    # manual map still chooses the camera of the single speaker
    res = decide_mapping(segs, MappingConfig(speaker_map="BA"),
                         diar_stats={"embedding_verdict": 1, "reasons": []})
    assert res.num_speakers == 1 and res.mapping == {"0": "B", "1": "B"} and res.confident


def test_embedding_single_verdict_overruled_by_separated_mics(tmp_path):
    pa, pb, segs = _tracks(tmp_path, True)
    res = decide_mapping(segs, MappingConfig(), mic_inputs={"wav_a": pa, "wav_b": pb,
                                                            "offset_a": 0.0, "offset_b": 0.0},
                         diar_stats={"embedding_verdict": 1, "reasons": ["similar voices"]})
    assert res.num_speakers == 2 and res.mapping == {"0": "A", "1": "B"} and res.method == "mic"


def test_forced_speaker_counts(tmp_path):
    pa, pb, segs = _tracks(tmp_path, True)
    mi = {"wav_a": pa, "wav_b": pb, "offset_a": 0.0, "offset_b": 0.0}
    r1 = decide_mapping(segs, MappingConfig(), mic_inputs=mi, num_speakers="1")
    assert r1.num_speakers == 1 and r1.mapping["0"] == r1.mapping["1"]
    pa, pb, segs1 = _one_voice(tmp_path, "a")
    r2 = decide_mapping(segs1, MappingConfig(), mic_inputs={"wav_a": pa, "wav_b": pb, "offset_a": 0.0,
                                                            "offset_b": 0.0}, num_speakers="2")
    assert r2.num_speakers == 2 and not r2.confident  # score ~0 -> inconclusive, not "confident"


def test_single_speaker_snr_is_gain_invariant(tmp_path):
    pa, pb, segs = _one_voice(tmp_path, "a")
    import soundfile as sf_

    x, sr = sf_.read(pb, dtype="float32")
    sf_.write(pb, x * 20.0, sr)  # far mic recorded 26 dB hotter (louder speech AND noise)
    ev = single_speaker_mic_evidence(segs, pa, pb, 0.0, 0.0)
    assert ev["level_diff_db"] < 0          # raw level would wrongly point at cam B
    assert decide_single_from_mic(ev, 3.0) == "A"


def test_near_zero_or_contradicting_motion_is_not_confident():
    # reviewer's single-speaker case: score -0.09 with the two cameras disagreeing
    t = np.arange(80)
    act0 = ((t // 8) % 2 == 0).astype(float)
    act1 = 1 - act0
    rng = np.random.default_rng(2)
    noise_a, noise_b = rng.random(80), rng.random(80)
    dec, ev = decide_from_motion(noise_a, noise_b, act0, act1, 0.2)
    assert dec is None
    # one camera weakly positive, the other weakly negative -> disagreement -> None
    weak_a = act0 * 0.15 + rng.random(80)
    weak_b = act0 * 0.15 + rng.random(80)
    dec, ev = decide_from_motion(weak_a, weak_b, act0, act1, 0.0)
    assert dec is None and not ev["cameras_agree"]


def test_shared_feed_with_camera_gain_difference_stays_two_speakers(tmp_path):
    """Regression (reviewer B, round 2): both close-up cameras record the SAME mixed feed
    but cam A's input gain is 8 dB hotter. Both clusters then look 'louder on cam A' with
    a ~0 dB score, which mic_says_single() read as one voice split in two - overruling a
    clear 2-speaker embedding verdict and putting the whole conversation on cam A. The
    gain-invariant SNR shows no near/far mic, so it must not count as a single speaker."""
    rng = np.random.default_rng(3)
    T = 20
    sa = np.zeros(T * SR, np.float32)
    sb = np.zeros(T * SR, np.float32)
    sa[0:5 * SR] = rng.standard_normal(5 * SR) * 0.3
    sa[10 * SR:15 * SR] = rng.standard_normal(5 * SR) * 0.3
    sb[5 * SR:10 * SR] = rng.standard_normal(5 * SR) * 0.3
    sb[15 * SR:] = rng.standard_normal(5 * SR) * 0.3
    feed = sa + sb + rng.standard_normal(T * SR).astype(np.float32) * 0.002
    pa, pb = tmp_path / "a.wav", tmp_path / "b.wav"
    sf.write(pa, feed * 10 ** (8 / 20), SR)
    sf.write(pb, feed, SR)
    segs = [SpeakerSegment(0, 5, "0"), SpeakerSegment(5, 10, "1"),
            SpeakerSegment(10, 15, "0"), SpeakerSegment(15, 20, "1")]
    res = decide_mapping(segs, MappingConfig(), mic_inputs={"wav_a": str(pa), "wav_b": str(pb),
                                                            "offset_a": 0.0, "offset_b": 0.0},
                         diar_stats={"embedding_verdict": 2, "centroid_cos": -0.07})
    assert res.num_speakers == 2, res
    assert not res.confident  # identical feeds carry no A/B evidence -> flagged guess


def test_embedding_single_verdict_overruled_by_confident_mouth_motion(tmp_path, monkeypatch):
    """Regression (reviewer B, round 2): two similar voices (Aman/Rishi, centroid cos 0.71)
    on a shared audio feed. Embeddings said 'one speaker', the mics carry no information,
    but each close-up's mouth motion followed a different cluster (score 1.68, cameras
    agree). That evidence was ignored and the whole conversation went to cam A."""
    import podcast_autoeditor.mapping as mp

    rng = np.random.default_rng(4)
    feed = rng.standard_normal(20 * SR).astype(np.float32) * 0.1
    pa, pb = tmp_path / "a.wav", tmp_path / "b.wav"
    sf.write(pa, feed, SR)
    sf.write(pb, feed, SR)
    segs = [SpeakerSegment(0, 5, "0"), SpeakerSegment(5, 10, "1"),
            SpeakerSegment(10, 15, "0"), SpeakerSegment(15, 20, "1")]
    calls = []

    def fake_visual(*a, **k):
        calls.append(1)
        return {"0": "B", "1": "A"}, {"available": True, "score": -1.68, "cameras_agree": True,
                                      "terms": {"cam_a": {"corr_0": -0.47, "corr_1": 0.45},
                                                "cam_b": {"corr_0": 0.33, "corr_1": -0.42}}}

    monkeypatch.setattr(mp, "visual_evidence", fake_visual)
    vi = {"ffmpeg": "ffmpeg", "video_a": "a.mp4", "video_b": "b.mp4", "offset_a": 0.0, "offset_b": 0.0}
    mi = {"wav_a": str(pa), "wav_b": str(pb), "offset_a": 0.0, "offset_b": 0.0}
    st = {"embedding_verdict": 1, "reasons": ["centroid cosine 0.71 >= 0.62"]}
    res = decide_mapping(segs, MappingConfig(), mic_inputs=mi, visual_inputs=vi, diar_stats=st)
    assert res.num_speakers == 2 and res.mapping == {"0": "B", "1": "A"} and res.method == "visual"
    assert len(calls) == 1  # the (expensive) visual analysis runs once
    # without a clear visual separation the single-speaker verdict stands - including a weakly
    # "confident" score where neither mouth falls silent for the other cluster (reviewer's
    # one-voice shared-feed case: score +0.31, own-cluster correlation negative)
    for dec, terms in ((None, {}),
                       ({"0": "A", "1": "B"}, {"cam_a": {"corr_0": -0.11, "corr_1": -0.28},
                                               "cam_b": {"corr_0": -0.05, "corr_1": 0.09}})):
        monkeypatch.setattr(mp, "visual_evidence", lambda *a, _d=dec, _t=terms, **k: (
            _d, {"available": True, "score": 0.31, "cameras_agree": True, "terms": _t}))
        res = decide_mapping(segs, MappingConfig(), mic_inputs=mi, visual_inputs=vi, diar_stats=st)
        assert res.num_speakers == 1, terms


def _alternating(n, fps=5.0, seed=0):
    """Irregular alternating turns: (times, segments, act0, act1)."""
    rng = np.random.default_rng(seed)
    times = np.arange(n) / fps
    segs, t, who = [], 0.0, 0
    while t < n / fps:
        d = float(rng.uniform(3, 12))
        segs.append(SpeakerSegment(round(t, 2), round(min(t + d, n / fps), 2), str(who)))
        t, who = t + d, 1 - who
    import podcast_autoeditor.mapping as mp
    return times, segs, mp.activity_series(times, segs, "0"), mp.activity_series(times, segs, "1")


def test_noise_only_mouth_motion_is_never_a_confident_mapping():
    """Regression (reviewer B, round 3): with static faces (no mouth motion at all) the
    correlations are pure noise, yet |score| > 0.2 with both terms >= 0.05 happened in
    ~15 % of 56 s clips (28 % of 20 s clips) and was reported as a CONFIDENT voice->face
    mapping. The score must now clear 3 sd of what that many noise samples give."""
    rng = np.random.default_rng(7)
    for n in (100, 280):
        confident = 0
        for k in range(400):
            _, _, a0, a1 = _alternating(n, seed=k)
            dec, ev = decide_from_motion(rng.random(n), rng.random(n), a0, a1, 0.2, 0.05)
            confident += dec is not None
        assert confident <= 4, (n, confident)        # was ~110 (n=100) / ~60 (n=280) of 400
    # the reviewer's static-face clip: terms +0.124 / +0.127 from 280 samples -> not confident
    _, _, a0, a1 = _alternating(280, seed=1)
    weak = lambda act: 0.035 * act + rng.random(280)  # noqa: E731  (corr ~0.06 per activity)
    dec, ev = decide_from_motion(weak(a0), weak(a1), a0, a1, 0.2, 0.05)
    assert dec is None
    # reviewer's 4K clip (38 s, 187 samples): cam A term -0.81, cam B only -0.34 (noisier
    # face) - same sign, score -1.15: confident, and NOT "one speaker" (corr_speech -0.08/+0.12)
    ev = {"terms": {"cam_a": {"corr_speech": -0.084, "n": 187}, "cam_b": {"corr_speech": 0.121, "n": 187}},
          "cameras_agree": False}
    from podcast_autoeditor.mapping import decide_single_from_motion, motion_says_single
    assert decide_single_from_motion(ev) is None and not motion_says_single(ev)
    _, _, a0, a1 = _alternating(187, seed=2)
    dec, ev = decide_from_motion(a1 + 0.6 * rng.random(187), 0.35 * a0 + rng.random(187), a0, a1, 0.2, 0.05)
    assert dec == {"0": "B", "1": "A"}, ev
    # real mouth motion still decides, also on a short clip
    for n in (100, 280, 1500):
        _, _, a0, a1 = _alternating(n, seed=3)
        dec, ev = decide_from_motion(a0 + 0.5 * rng.random(n), a1 + 0.5 * rng.random(n), a0, a1, 0.2, 0.05)
        assert dec == {"0": "A", "1": "B"}, (n, ev)


def test_single_talker_face_is_chosen_by_mouth_motion_energy():
    """Regression (reviewer B, round 3): one person talks almost without pause while the
    other sits still. Correlation with the (nearly constant) speech activity was ~0.06 for
    both faces -> 'CAMERA IS A GUESS: showing cam A' although the talker's mouth moved 8x
    more (0.181 vs 0.023) and sat on the other side."""
    import podcast_autoeditor.mapping as mp

    rng = np.random.default_rng(5)
    n = 110
    times = np.arange(n) / 5.0
    segs = [SpeakerSegment(0.6, 9.0, "0"), SpeakerSegment(9.2, 21.0, "1")]   # one voice, split in two
    talk = 0.18 + 0.15 * rng.random(n)          # mouth moving all the time
    still = 0.02 * rng.random(n)
    st = {"embedding_verdict": 1, "reasons": ["centroid cosine 0.87 >= 0.62"]}
    for ma, mb, want in ((still, talk, "B"), (talk, still, "A")):
        res = decide_mapping(segs, MappingConfig(), diar_stats=st,
                             visual_inputs={"series": {"times": times, "motion_a": ma, "motion_b": mb, "info": {}}})
        assert res.num_speakers == 1 and res.mapping == {"0": want, "1": want}, res
        assert res.confident and res.method == "single-speaker:visual"
    # similar motion on both faces -> still an honest guess
    res = decide_mapping(segs, MappingConfig(), diar_stats=st,
                         visual_inputs={"series": {"times": times, "motion_a": talk, "motion_b": talk[::-1].copy(),
                                                   "info": {}}})
    assert res.num_speakers == 1 and not res.confident
    assert mp.decide_single_from_energy({"terms": {}}) is None
    # two static faces (reviewer's no-mouth-motion clip): noise 0.032 vs 0.0001 is not evidence
    noisy = {"terms": {"cam_a": {"motion_mean": 0.032}, "cam_b": {"motion_mean": 0.0001}}}
    assert mp.decide_single_from_energy(noisy) is None
    res = decide_mapping(segs, MappingConfig(), diar_stats=st,
                         visual_inputs={"series": {"times": times, "motion_a": 0.064 * rng.random(n),
                                                   "motion_b": 0.0002 * rng.random(n), "info": {}}})
    assert res.num_speakers == 1 and not res.confident and res.method == "single-speaker:guess"
