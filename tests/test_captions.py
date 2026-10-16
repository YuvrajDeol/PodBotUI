import pytest

from podcast_autoeditor.captions import (ass_to_rgb, build_captions, chunk_words, hex_to_ass,
                                         parse_color, write_ass, write_srt)
from podcast_autoeditor.config import CaptionConfig
from podcast_autoeditor.datatypes import Word


@pytest.mark.parametrize("hexc,ass", [
    ("#2D9CFF", "&H00FF9C2D"),
    ("#FFFFFF", "&H00FFFFFF"),
    ("#FF0000", "&H000000FF"),   # red -> RR in the last byte
    ("#0000FF", "&H00FF0000"),   # blue -> BB in the first colour byte
    ("#00FF00", "&H0000FF00"),
    ("123456", "&H00563412"),
    ("#abc", "&H00CCBBAA"),
    ("rgb(45, 156, 255)", "&H00FF9C2D"),
    ("rgba(45,156,255,0.5)", "&H00FF9C2D"),
])
def test_hex_to_ass(hexc, ass):
    assert hex_to_ass(hexc) == ass


def test_hex_to_ass_alpha_and_roundtrip():
    assert hex_to_ass("#2D9CFF", alpha=0x80) == "&H80FF9C2D"
    assert ass_to_rgb(hex_to_ass("#2D9CFF")) == (0x2D, 0x9C, 0xFF, 0)


@pytest.mark.parametrize("bad", ["#12345", "zzzzzz", "", "#GGGGGG"])
def test_bad_colors(bad):
    with pytest.raises(ValueError):
        parse_color(bad)


def _words(spec):
    out = []
    for spk, t0, text in spec:
        t = t0
        for tok in text.split():
            out.append(Word(tok, t, t + 0.3, 1.0, spk))
            t += 0.35
    return out


def test_chunking_limits_and_speaker_breaks():
    cfg = CaptionConfig()
    ws = _words([("A", 0.0, "one two three four five six seven eight nine ten"),
                 ("B", 4.0, "yes")])
    lines = chunk_words(ws, cfg)
    assert [l.speaker for l in lines] == ["A", "A", "B"]
    for l in lines:
        assert len(l.text.split()) <= cfg.max_words
        assert len(l.text) <= cfg.max_chars
        assert l.end - l.start <= cfg.max_dur + cfg.linger + 1e-6
    assert lines[0].text == "one two three four five six"


def test_chunking_breaks_on_punctuation_and_chars():
    cfg = CaptionConfig(max_chars=20)
    ws = _words([("A", 0.0, "Hello. Extraordinarily complicated words appear")])
    lines = chunk_words(ws, cfg)
    assert lines[0].text == "Hello."
    assert all(len(l.text) <= 20 or len(l.text.split()) == 1 for l in lines)


def test_overlapping_speakers_get_separate_rows(tmp_path):
    cfg = CaptionConfig()
    ws = _words([("A", 0.0, "are you sure about that?"), ("B", 1.0, "absolutely yes I am")])
    lines, style = build_captions(ws, cfg, 1920, 1080)
    a = [l for l in lines if l.speaker == "A"][0]
    b = [l for l in lines if l.speaker == "B"][0]
    assert a.start < b.end and b.start < a.end  # they overlap in time
    assert a.margin_v != b.margin_v
    assert abs(a.margin_v - b.margin_v) >= style.font_size  # no vertical collision
    p = tmp_path / "c.ass"
    write_ass(str(p), lines, style, cfg)
    txt = p.read_text()
    assert "Style: SpeakerA,Arial," in txt and "&H00FF9C2D" in txt
    assert "Style: SpeakerB,Arial," in txt and "&H00FFFFFF" in txt
    assert "PlayResY: 1080" in txt
    assert f",{a.margin_v},," in txt and f",{b.margin_v},," in txt
    s = tmp_path / "c.srt"
    write_srt(str(s), lines)
    assert "Speaker A:" in s.read_text() and "Speaker B:" in s.read_text()


def test_non_overlapping_lines_share_bottom_row():
    cfg = CaptionConfig()
    ws = _words([("A", 0.0, "first line here."), ("B", 5.0, "second line here.")])
    lines, style = build_captions(ws, cfg, 1920, 1080)
    assert {l.margin_v for l in lines} == {style.margin_v}


def test_font_size_is_about_5_to_6_percent():
    _, style = build_captions(_words([("A", 0, "hi")]), CaptionConfig(), 1920, 1080)
    assert 0.05 * 1080 <= style.font_size <= 0.06 * 1080
