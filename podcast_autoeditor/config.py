"""Configuration dataclasses. Every threshold used by the pipeline lives here."""

from __future__ import annotations

from dataclasses import asdict, dataclass, field
from typing import Any, Dict, Optional


@dataclass
class SyncConfig:
    enabled: bool = True
    max_lag_s: float = 60.0          # search window (+/-) for offsets (cameras are rarely
                                     # started within 10 s of each other)
    analysis_s: float = 180.0        # seconds of audio used for correlation
    min_confidence: float = 2.0      # peak-to-sidelobe ratio below which the offset is NOT used
    force: bool = False              # apply low-confidence offsets anyway (--force-sync)
    # Manual overrides (seconds; position of the clip's t=0 on the reference timeline)
    offset_a: Optional[float] = None
    offset_b: Optional[float] = None
    offset_wide: Optional[float] = None


@dataclass
class AsrConfig:
    model_size: str = "small"
    language: Optional[str] = None   # None = auto-detect
    device: str = "cpu"
    compute_type: str = "int8"
    beam_size: int = 5
    vad_filter: bool = True
    # Conditioning on previous text + a punctuated primer keeps Whisper emitting
    # punctuation ("?" drives the question->answer rule). Set prompt to "" to disable.
    condition_on_previous_text: bool = True
    initial_prompt: Optional[str] = "Hello, and welcome to the show. How are you? I'm great, thanks!"
    refine_word_times: bool = True   # trim leading/trailing silence absorbed into words


@dataclass
class DiarizationConfig:
    backend: str = "auto"            # auto | ecapa | mfcc | pyannote
    num_speakers: str = "auto"       # auto | 1 | 2  (auto: 2 clusters, then a 1-vs-2 test)
    # 1-vs-2 speaker test on the 2-cluster split, calibrated with
    # scripts/calibrate_num_speakers.py (out/calibration_num_speakers.json): ECAPA centroid
    # cosine of 16 single voices x {5 s, 30 s} = 0.676-0.885; 7 distinct-voice pairs x
    # {10 s, 30 s} + 4 demo podcasts = 0.044-0.549.  (Reed/Eddy, two near-identical Apple
    # voices, scored 0.74/0.88 - indistinguishable to ECAPA; the mic test still separates them.)
    single_centroid_cos: float = 0.62  # ECAPA: cluster centroids this similar -> one voice
    # MFCC fallback: silhouette of single voices 0.10-0.18 (30 s) but 0.12-0.27 (5 s);
    # distinct pairs >= 0.27 -> short single-voice clips may still be read as 2 speakers.
    single_silhouette: float = 0.22    # MFCC: silhouette below this -> one voice
    min_speaker_s: float = 1.0         # 2nd cluster with less speech than this -> one voice
    min_speaker_share: float = 0.04    # ... or less than this share of all speech
    window_s: float = 1.5
    hop_s: float = 0.5
    min_window_s: float = 0.6        # shortest speech region we embed
    median_filter: int = 3           # median smoothing (in hops) of window labels
    region_merge_gap: float = 0.15   # words closer than this form one speech region
    word_refine: bool = True         # re-score ASR phrases against the speaker centroids
    phrase_gap: float = 0.15         # split phrases on pauses longer than this
    phrase_max: float = 3.0          # ... and into chunks of at most this many seconds
    refine_min_margin: float = 0.04  # min cosine margin to trust a phrase-level decision
    ref_a: Optional[str] = None      # optional enrollment audio for speaker A
    ref_b: Optional[str] = None      # optional enrollment audio for speaker B
    ecapa_source: str = "speechbrain/spkrec-ecapa-voxceleb"


@dataclass
class MappingConfig:
    strategy: str = "auto"           # auto | mic | visual | first
    speaker_map: str = "auto"        # auto | AB | BA  (AB: first diarized speaker -> A)
    swap: bool = False               # flip the final decision
    # Two speakers: per-camera mic score (dA0-dB0)-(dA1-dB1) must exceed this to decide.
    # Distinct voices on separate mics give >= ~2x the own-vs-bleed separation (demos:
    # 19-29 dB); a single voice split into 2 clusters gave 0.0-2.2 dB.
    mic_margin_db: float = 4.0
    # One voice split into 2 clusters: both clusters dominate the SAME mic (|diff| >= this
    # with the same sign) while the score stays below mic_same_db.
    mic_same_db: float = 3.0
    mic_dominance_db: float = 6.0
    single_snr_margin_db: float = 3.0  # single speaker: near-mic SNR advantage needed
    visual_fps: float = 5.0
    visual_max_s: float = 300.0
    # Mouth-motion score must exceed this AND every camera with faces must agree in sign
    # (each term >= visual_min_term), otherwise the visual evidence is not "confident".
    visual_margin: float = 0.2
    visual_min_term: float = 0.05
    visual_timeout_s: float = 120.0    # wall-clock budget for decoding + face detection
    visual_max_frames: int = 2000      # per camera


@dataclass
class LabelConfig:
    flip_max_words: int = 1          # isolated runs up to this many words may be flipped
    flip_max_dur: float = 0.5
    flip_max_gap: float = 0.25       # only flip if neighbours are this close (no pause)
    turn_gap: float = 1.0            # split same-speaker words into new turn on gap > this
    # Overlap from per-camera mics (labeling.overlaps_from_mic_coupling): each mic's bleed
    # from the other talker is learned from single-speaker regions; a frame is overlapped
    # when BOTH mics exceed their predicted bleed level by a margin.
    overlap_mic_min_sep_db: float = 3.0  # min own-vs-bleed separation to use the mics at all
    overlap_excess_db: float = 3.0       # min excess over predicted bleed (dB) ...
    overlap_excess_frac: float = 0.4     # ... or this fraction of the separation, if larger


@dataclass
class PlannerConfig:
    lead: float = 0.15               # cut this much before speech onset
    min_shot: float = 1.2
    qa_enabled: bool = True
    qa_gap: float = 1.5              # max gap between question end and answer start
    qa_answer_hold: float = 2.0      # stay WIDE this long into the answer
    qa_max_question: float = 6.0     # cap on how much of a long question sentence goes WIDE
    # A question = a sentence ending in "?"; with question_heuristic also one that opens like
    # a question ("what do you", "do you", "is it") or ends in a tag (", right", "isn't it") -
    # Whisper sometimes drops the "?" (see planner.question_start)
    question_heuristic: bool = True
    rapid_enabled: bool = True
    rapid_turns: int = 3             # >= this many speaker changes ...
    rapid_window: float = 6.0        # ... within this many seconds -> WIDE
    rapid_pad: float = 1.0           # WIDE padding around the first/last change of a rapid run
    overlap_wide: bool = True
    min_overlap: float = 0.3         # ignore overlaps shorter than this
    silence_wide: float = 2.5        # silences longer than this go WIDE
    silence_hold: float = 0.5        # keep speaker's close-up this long into a long silence
    outro_hold: float = 0.5          # after the last word, hold close-up then go WIDE


@dataclass
class CaptionConfig:
    enabled: bool = True
    color_a: str = "#2D9CFF"
    color_b: str = "#FFFFFF"
    font: str = "Arial"
    bold: bool = True
    font_size_pct: float = 5.5       # % of output height
    outline_px: float = 3.0          # at 1080p; scaled with height
    shadow_px: float = 1.5
    margin_v_pct: float = 6.0        # bottom margin, % of height
    max_words: int = 6
    max_chars: int = 32
    max_dur: float = 2.5
    linger: float = 0.25             # keep a line on screen after its last word
    min_dur: float = 0.6
    burner: str = "auto"             # auto | ass | pillow


@dataclass
class RenderConfig:
    width: int = 1920
    height: int = 1080
    fps: float = 30.0
    encoder: str = "auto"            # auto | libx264 | h264_videotoolbox
    crf: int = 20
    preset: str = "medium"
    video_bitrate: str = "8M"        # used by h264_videotoolbox
    intermediate_crf: int = 14
    intermediate_preset: str = "veryfast"
    audio_source: str = "auto"       # auto | external | wide | cam-a | cam-b
    audio_bitrate: str = "192k"
    jobs: int = 0                    # parallel segment encodes (0 = auto)
    keep_intermediates: bool = False


@dataclass
class ReframeConfig:
    """Single-clip mode: virtual close-ups cropped from the one wide shot."""
    a_side: str = "left"             # which person is Speaker A: left | right (of the frame)
    crop_scale: float = 3.2          # crop height = crop_scale x face height (3-3.5 = head+shoulders)
    min_crop_scale: float = 2.2      # may tighten to this to keep the other person out of frame
    eye_line: float = 1.0 / 3.0      # eyes at this fraction of the crop height from the top
    max_zoom: float = 3.0            # cap on upscaling: crop height >= output height / max_zoom
    warn_zoom: float = 2.0           # warn when a crop is upscaled more than this
    crop_a: Optional[str] = None     # manual crop "x,y,w,h" (source pixels) for Speaker A
    crop_b: Optional[str] = None     # ... and Speaker B
    per_shot: bool = True            # re-frame (only at shot boundaries) when a person moves
    safe_frac: float = 0.18          # face centre may drift this fraction of the crop size
    dense_fps: float = 5.0           # face scan rate inside the mouth-motion window
    dense_s: float = 300.0           # length of that window (from the start of speech)
    sparse_fps: float = 1.0          # scan rate for the rest of the clip (positions only)
    analysis_width: int = 960        # frames are scaled to this (long side) for detection
    max_frames: int = 2400           # hard cap on analysed frames
    timeout_s: float = 240.0         # wall-clock budget for the face scan
    min_persistence: float = 0.15    # a person must be found in this share of scanned frames
    cluster_gap: float = 0.8         # faces further apart (x, in face widths) are other people
    preview: Optional[str] = None    # extra path for the crop preview PNG


@dataclass
class PipelineConfig:
    mode: str = "auto"               # auto | multicam | single (auto: single if no cam A/B)
    cam_a: str = ""
    cam_b: str = ""
    wide: str = ""
    audio: Optional[str] = None
    out: str = "edited.mp4"
    work_dir: Optional[str] = None
    plan_only: bool = False
    force: bool = False
    ffmpeg: Optional[str] = None
    sync: SyncConfig = field(default_factory=SyncConfig)
    asr: AsrConfig = field(default_factory=AsrConfig)
    diar: DiarizationConfig = field(default_factory=DiarizationConfig)
    mapping: MappingConfig = field(default_factory=MappingConfig)
    label: LabelConfig = field(default_factory=LabelConfig)
    planner: PlannerConfig = field(default_factory=PlannerConfig)
    captions: CaptionConfig = field(default_factory=CaptionConfig)
    render: RenderConfig = field(default_factory=RenderConfig)
    reframe: ReframeConfig = field(default_factory=ReframeConfig)

    def to_dict(self) -> Dict[str, Any]:
        return asdict(self)
