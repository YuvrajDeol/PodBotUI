import numpy as np
import pytest

from podcast_autoeditor.sync import compute_offsets, estimate_offset, gcc_lag

SR = 16000
FRAME = 1 / 30


def speechlike(seconds: float, seed: int) -> np.ndarray:
    """Bursty band-limited noise resembling speech energy patterns."""
    rng = np.random.default_rng(seed)
    n = int(seconds * SR)
    x = rng.standard_normal(n)
    # crude band-limiting
    x = np.convolve(x, np.ones(8) / 8, mode="same")
    env = np.zeros(n)
    t = 0
    while t < n:
        L = int(rng.uniform(0.15, 0.6) * SR)
        if rng.random() < 0.65:
            env[t:t + L] = rng.uniform(0.3, 1.0)
        t += L
    env = np.convolve(env, np.ones(200) / 200, mode="same")
    return (x * env).astype(np.float32)


def make_clip(mix: np.ndarray, offset: float, noise: float, seed: int) -> np.ndarray:
    """A clip whose t=0 is at ``offset`` on the reference timeline."""
    rng = np.random.default_rng(seed)
    k = int(round(offset * SR))
    if k >= 0:
        clip = mix[k:]
    else:
        clip = np.concatenate([rng.standard_normal(-k).astype(np.float32) * 0.01, mix])
    return clip + noise * rng.standard_normal(clip.size).astype(np.float32)


@pytest.mark.parametrize("offset", [1.3, -0.6, 0.0, 7.25, -9.5])
def test_offset_recovered_within_one_frame(offset):
    a = speechlike(60, 1)
    b = speechlike(60, 2)
    ref = 0.7 * a + 0.7 * b                      # wide mic: balanced mix
    cam = make_clip(1.0 * a + 0.15 * b, offset, 0.01, 3)  # close mic: A loud, B bleed
    est, conf = estimate_offset(ref, cam, SR, max_lag_s=10.0, analysis_s=40.0)
    assert abs(est - offset) < FRAME, (est, offset)
    assert conf > 2.0


def test_gcc_lag_sign_convention():
    x = speechlike(20, 5)
    # sig[n] = ref[n + d] for d = 0.5 s  -> lag +0.5
    d = int(0.5 * SR)
    lag, _ = gcc_lag(x, x[d:], SR, 2.0)
    assert lag == pytest.approx(0.5, abs=1e-3)


def test_silent_clip_gives_zero_confidence():
    ref = speechlike(20, 7)
    lag, conf = gcc_lag(ref, np.zeros(SR * 10, np.float32), SR, 5.0)
    assert lag == 0.0 and conf == 0.0


def test_compute_offsets_with_files(tmp_path):
    import soundfile as sf

    a, b = speechlike(45, 11), speechlike(45, 12)
    ref = 0.7 * a + 0.7 * b
    files = {}
    sf.write(tmp_path / "ref.wav", ref, SR)
    for name, off, mix in (("cam_a", -0.5, a + 0.2 * b), ("cam_b", 1.3, 0.2 * a + b)):
        p = tmp_path / f"{name}.wav"
        sf.write(p, make_clip(mix, off, 0.005, 4), SR)
        files[name] = str(p)
    files["wide"] = str(tmp_path / "ref.wav")
    files["noaudio"] = None
    res = compute_offsets(str(tmp_path / "ref.wav"), files, 10.0, 30.0,
                          manual={"cam_b": None}, reference_name="wide")
    assert abs(res["cam_a"].offset + 0.5) < FRAME
    assert abs(res["cam_b"].offset - 1.3) < FRAME
    assert res["wide"].offset == 0.0 and res["wide"].method == "reference"
    assert res["noaudio"].offset == 0.0 and res["noaudio"].method == "none"
    res2 = compute_offsets(str(tmp_path / "ref.wav"), files, manual={"cam_a": 2.0})
    assert res2["cam_a"].offset == 2.0 and res2["cam_a"].method == "manual"


@pytest.mark.parametrize("offset", [25.0, -42.5])
def test_default_window_recovers_offsets_beyond_10s(offset):
    """Regression: the default +/-10 s search window silently produced a wrong offset
    (confidence ~1, warning only) for cameras started more than 10 s apart."""
    from podcast_autoeditor.config import SyncConfig

    a, b = speechlike(120, 21), speechlike(120, 22)
    ref = 0.7 * a + 0.7 * b
    cam = make_clip(1.0 * a + 0.15 * b, offset, 0.01, 23)
    cfg = SyncConfig()
    est, conf = estimate_offset(ref, cam, SR, cfg.max_lag_s, cfg.analysis_s, cfg.min_confidence)
    assert abs(est - offset) < FRAME and conf > cfg.min_confidence, (est, conf)


def test_low_confidence_offset_is_not_applied_unless_forced(tmp_path):
    import soundfile as sf

    ref = speechlike(30, 21)
    unrelated = speechlike(30, 99)          # a clip whose audio has nothing in common
    sf.write(tmp_path / "ref.wav", ref, SR)
    sf.write(tmp_path / "cam.wav", unrelated, SR)
    files = {"cam_b": str(tmp_path / "cam.wav")}
    res = compute_offsets(str(tmp_path / "ref.wav"), files, 5.0, 20.0, min_confidence=2.0)
    r = res["cam_b"]
    assert r.confidence < 2.0
    assert r.offset == 0.0 and r.method == "low-confidence" and "--offset-b" in r.note
    forced = compute_offsets(str(tmp_path / "ref.wav"), files, 5.0, 20.0, min_confidence=2.0,
                             force=True)["cam_b"]
    assert forced.method == "gcc" and "force-sync" in forced.note
    manual = compute_offsets(str(tmp_path / "ref.wav"), files, 5.0, 20.0, min_confidence=2.0,
                             manual={"cam_b": 1.25})["cam_b"]
    assert manual.offset == 1.25 and manual.method == "manual"


def test_no_audio_warning_names_the_offset_flag(tmp_path, caplog):
    import logging

    import soundfile as sf

    sf.write(tmp_path / "ref.wav", speechlike(10, 3), SR)
    with caplog.at_level(logging.WARNING):
        res = compute_offsets(str(tmp_path / "ref.wav"), {"cam_a": None, "wide": None})
    assert "--offset-a" in res["cam_a"].note and "--offset-wide" in res["wide"].note
    assert "--offset-a" in caplog.text


def test_ntsc_frame_rates_are_exact():
    from podcast_autoeditor.render import fps_str, normalize_fps

    assert fps_str(29.97) == "30000/1001"
    assert fps_str(23.976) == "24000/1001"
    assert fps_str(59.94) == "60000/1001"
    assert fps_str(30) == "30" and fps_str(25.0) == "25" and fps_str(12.5) == "25/2"
    assert normalize_fps(29.97) == 30000 / 1001
