from typing import List, Sequence, Tuple

from podcast_autoeditor.datatypes import Turn, Word


def make_turn(speaker: str, start: float, end: float, text: str) -> Turn:
    """Spread the words of ``text`` evenly over [start, end]."""
    toks = text.split()
    n = len(toks)
    step = (end - start) / n
    words = [Word(t, start + i * step, start + (i + 1) * step - 0.02, 1.0, speaker)
             for i, t in enumerate(toks)]
    return Turn(speaker, words)


def turns(spec: Sequence[Tuple[str, float, float, str]]) -> List[Turn]:
    return [make_turn(*s) for s in spec]


def cam_at(shots, t: float) -> str:
    for s in shots:
        if s.start <= t < s.end:
            return s.camera
    return shots[-1].camera
