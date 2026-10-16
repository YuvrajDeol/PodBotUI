"""AI multicam podcast auto-editor.

Pipeline: probe + sync -> ASR (faster-whisper) -> 2-speaker diarization ->
speaker->camera mapping -> word labeling / turns -> shot planning ->
captions (.ass/.srt/.json) -> render (ffmpeg).
"""

__version__ = "0.1.0"

CAM_A = "CAM_A"
CAM_B = "CAM_B"
WIDE = "WIDE"
CAMERAS = (CAM_A, CAM_B, WIDE)

__all__ = ["__version__", "CAM_A", "CAM_B", "WIDE", "CAMERAS"]
