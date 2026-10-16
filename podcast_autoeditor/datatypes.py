"""Plain data types shared across the pipeline (JSON-serialisable)."""

from __future__ import annotations

from dataclasses import asdict, dataclass, field
from typing import Any, Dict, List, Optional


@dataclass
class Word:
    word: str
    start: float
    end: float
    prob: float = 1.0
    speaker: Optional[str] = None  # "A" | "B" | None

    @property
    def text(self) -> str:
        return self.word.strip()

    @property
    def mid(self) -> float:
        return 0.5 * (self.start + self.end)

    def to_dict(self) -> Dict[str, Any]:
        return asdict(self)

    @staticmethod
    def from_dict(d: Dict[str, Any]) -> "Word":
        return Word(
            word=str(d["word"]),
            start=float(d["start"]),
            end=float(d["end"]),
            prob=float(d.get("prob", 1.0)),
            speaker=d.get("speaker"),
        )


@dataclass
class SpeakerSegment:
    start: float
    end: float
    label: str  # cluster id ("0"/"1") before mapping, "A"/"B" after mapping

    def to_dict(self) -> Dict[str, Any]:
        return asdict(self)

    @staticmethod
    def from_dict(d: Dict[str, Any]) -> "SpeakerSegment":
        return SpeakerSegment(float(d["start"]), float(d["end"]), str(d["label"]))


@dataclass
class Turn:
    speaker: str
    words: List[Word] = field(default_factory=list)

    @property
    def start(self) -> float:
        return self.words[0].start

    @property
    def end(self) -> float:
        return self.words[-1].end

    @property
    def text(self) -> str:
        return " ".join(w.text for w in self.words).strip()

    def sentences(self) -> List[List[Word]]:
        """Split the turn's words into sentences on . ? ! terminators."""
        out: List[List[Word]] = []
        cur: List[Word] = []
        for w in self.words:
            cur.append(w)
            if w.text.endswith(("?", ".", "!", "...", "…")):
                out.append(cur)
                cur = []
        if cur:
            out.append(cur)
        return out

    def to_dict(self) -> Dict[str, Any]:
        return {
            "speaker": self.speaker,
            "start": self.start,
            "end": self.end,
            "text": self.text,
            "words": [w.to_dict() for w in self.words],
        }


@dataclass
class Shot:
    start: float
    end: float
    camera: str  # CAM_A | CAM_B | WIDE
    reason: str
    start_frame: int = 0
    end_frame: int = 0

    @property
    def duration(self) -> float:
        return self.end - self.start

    @property
    def n_frames(self) -> int:
        return self.end_frame - self.start_frame

    def to_dict(self) -> Dict[str, Any]:
        d = asdict(self)
        d["duration"] = round(self.duration, 6)
        return d

    @staticmethod
    def from_dict(d: Dict[str, Any]) -> "Shot":
        return Shot(
            start=float(d["start"]),
            end=float(d["end"]),
            camera=str(d["camera"]),
            reason=str(d.get("reason", "")),
            start_frame=int(d.get("start_frame", 0)),
            end_frame=int(d.get("end_frame", 0)),
        )


@dataclass
class Interval:
    start: float
    end: float

    def to_dict(self) -> Dict[str, Any]:
        return asdict(self)
