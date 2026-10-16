"""Two-speaker diarization.

Backends
--------
* ``ecapa`` (primary): SpeechBrain ECAPA-TDNN speaker embeddings
  (``speechbrain/spkrec-ecapa-voxceleb``) on sliding windows inside speech regions,
  clustered into 2 groups (agglomerative / cosine, refined with spherical k-means),
  then median-smoothed.
* ``mfcc`` (fallback): MFCC mean/std + pitch statistics per window, same clustering.
* ``pyannote`` (optional): used only if requested / ``HF_TOKEN`` is set and
  ``pyannote.audio`` is installed.

If enrollment audio for A and B is given (``ref_a``/``ref_b``) windows are labelled
"A"/"B" by cosine similarity to the enrollment embeddings instead of clustering.
Otherwise labels are cluster ids "0"/"1" ("0" = the speaker heard first).
"""

from __future__ import annotations

import logging
import os
import warnings
from dataclasses import dataclass, field
from typing import Any, Dict, List, Optional, Sequence, Tuple

import numpy as np

from .config import DiarizationConfig
from .datatypes import Interval, SpeakerSegment, Word
from .sync import load_wav

log = logging.getLogger(__name__)

SR = 16000


@dataclass
class DiarizationResult:
    backend: str
    segments: List[SpeakerSegment]
    enrolled: bool = False
    windows: List[Dict[str, Any]] = field(default_factory=list)
    centroid_similarity: float = float("nan")
    notes: List[str] = field(default_factory=list)
    phrases: List[Dict[str, Any]] = field(default_factory=list)
    # cluster statistics + the embedding-only 1-vs-2 speaker verdict (see speaker_count_stats)
    stats: Dict[str, Any] = field(default_factory=dict)

    def to_dict(self) -> Dict[str, Any]:
        return {
            "backend": self.backend,
            "enrolled": self.enrolled,
            "centroid_similarity": self.centroid_similarity,
            "notes": self.notes,
            "segments": [s.to_dict() for s in self.segments],
            "windows": self.windows,
            "phrases": self.phrases,
            "stats": self.stats,
        }

    @staticmethod
    def from_dict(d: Dict[str, Any]) -> "DiarizationResult":
        return DiarizationResult(
            backend=d["backend"],
            segments=[SpeakerSegment.from_dict(s) for s in d["segments"]],
            enrolled=bool(d.get("enrolled", False)),
            windows=list(d.get("windows", [])),
            centroid_similarity=float(d.get("centroid_similarity", float("nan"))),
            notes=list(d.get("notes", [])),
            phrases=list(d.get("phrases", [])),
            stats=dict(d.get("stats", {})),
        )


# --------------------------------------------------------------------------- regions

def speech_regions_from_words(
    words: Sequence[Word], merge_gap: float = 0.3, pad: float = 0.05
) -> List[Interval]:
    regions: List[Interval] = []
    for w in sorted(words, key=lambda w: w.start):
        s, e = max(0.0, w.start - pad), w.end + pad
        if regions and s <= regions[-1].end + merge_gap:
            regions[-1].end = max(regions[-1].end, e)
        else:
            regions.append(Interval(s, e))
    return regions


def energy_vad(x: np.ndarray, sr: int = SR, frame: float = 0.03, thresh_db: float = -35.0,
               min_speech: float = 0.25, merge_gap: float = 0.3) -> List[Interval]:
    """Very simple energy VAD (used only when ASR produced no words)."""
    n = int(frame * sr)
    if x.size < n:
        return []
    m = x.size // n
    rms = np.sqrt(np.mean(x[: m * n].reshape(m, n) ** 2, axis=1) + 1e-12)
    db = 20 * np.log10(rms)
    ref = np.percentile(db, 95)
    active = db > max(thresh_db, ref - 30)
    out: List[Interval] = []
    i = 0
    while i < m:
        if active[i]:
            j = i
            while j < m and active[j]:
                j += 1
            s, e = i * frame, j * frame
            if out and s - out[-1].end < merge_gap:
                out[-1].end = e
            else:
                out.append(Interval(s, e))
            i = j
        else:
            i += 1
    return [r for r in out if r.end - r.start >= min_speech]


def make_windows(regions: Sequence[Interval], win: float, hop: float, min_win: float,
                 total: float) -> List[Tuple[float, float]]:
    out: List[Tuple[float, float]] = []
    for r in regions:
        L = r.end - r.start
        if L <= win:
            if L < min_win:
                c = 0.5 * (r.start + r.end)
                s, e = max(0.0, c - min_win / 2), min(total, c + min_win / 2)
            else:
                s, e = r.start, r.end
            out.append((s, e))
            continue
        t = r.start
        while t + win <= r.end + 1e-6:
            out.append((t, t + win))
            t += hop
        if out[-1][1] < r.end - 1e-3:
            out.append((r.end - win, r.end))
    return out


# --------------------------------------------------------------------------- embedders

_ECAPA_CACHE: Dict[str, Any] = {}


def _load_ecapa(source: str):
    if source in _ECAPA_CACHE:
        return _ECAPA_CACHE[source]
    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        logging.getLogger("speechbrain").setLevel(logging.WARNING)
        try:
            from speechbrain.inference.speaker import EncoderClassifier
        except ImportError:  # speechbrain < 1.0
            from speechbrain.pretrained import EncoderClassifier  # type: ignore
        savedir = os.path.join(
            os.path.expanduser("~/.cache/podcast_autoeditor"), source.replace("/", "__")
        )
        model = EncoderClassifier.from_hparams(
            source=source, savedir=savedir, run_opts={"device": "cpu"}
        )
    model.eval()
    _ECAPA_CACHE[source] = model
    return model


class EcapaEmbedder:
    name = "ecapa"

    batch_size = 8

    def __init__(self, source: str):
        self.model = _load_ecapa(source)

    def __call__(self, chunks: List[np.ndarray]) -> np.ndarray:
        import torch

        embs: List[Optional[np.ndarray]] = [None] * len(chunks)
        by_len: Dict[int, List[int]] = {}
        for i, c in enumerate(chunks):
            by_len.setdefault(len(c), []).append(i)
        with torch.no_grad():
            for _, idxs in by_len.items():
                # Small batches: on CPU torch, ECAPA batches >= 16 can be ~100x slower
                # per item than batches of 8 (measured with torch 2.14 on Apple silicon).
                for b in range(0, len(idxs), self.batch_size):
                    ids = idxs[b: b + self.batch_size]
                    batch = torch.from_numpy(np.stack([chunks[i] for i in ids]).astype(np.float32))
                    e = self.model.encode_batch(batch).squeeze(1).cpu().numpy()
                    for k, i in enumerate(ids):
                        embs[i] = e[k]
        return np.stack(embs)  # type: ignore[arg-type]


def _mel_filterbank(sr: int, n_fft: int, n_mels: int, fmin: float = 60.0,
                    fmax: Optional[float] = None) -> np.ndarray:
    fmax = fmax or sr / 2
    mel = lambda f: 2595.0 * np.log10(1.0 + f / 700.0)  # noqa: E731
    imel = lambda m: 700.0 * (10 ** (m / 2595.0) - 1.0)  # noqa: E731
    pts = imel(np.linspace(mel(fmin), mel(fmax), n_mels + 2))
    bins = np.floor((n_fft + 1) * pts / sr).astype(int)
    fb = np.zeros((n_mels, n_fft // 2 + 1))
    for m in range(1, n_mels + 1):
        l, c, r = bins[m - 1], bins[m], bins[m + 1]
        for k in range(l, c):
            if c > l:
                fb[m - 1, k] = (k - l) / (c - l)
        for k in range(c, r):
            if r > c:
                fb[m - 1, k] = (r - k) / (r - c)
    return fb


def mfcc(x: np.ndarray, sr: int = SR, n_mfcc: int = 20, n_mels: int = 40,
         win: int = 400, hop: int = 160, n_fft: int = 512) -> np.ndarray:
    """MFCCs (frames x n_mfcc), numpy only."""
    from scipy.fft import dct

    if x.size < win:
        x = np.pad(x, (0, win - x.size))
    n = 1 + (x.size - win) // hop
    idx = np.arange(win)[None, :] + hop * np.arange(n)[:, None]
    frames = x[idx] * np.hamming(win)[None, :]
    spec = np.abs(np.fft.rfft(frames, n_fft)) ** 2
    fb = _mel_filterbank(sr, n_fft, n_mels)
    mel = np.log(spec @ fb.T + 1e-10)
    return dct(mel, type=2, axis=1, norm="ortho")[:, 1: n_mfcc + 1]


def pitch_stats(x: np.ndarray, sr: int = SR, frame: int = 640, hop: int = 320) -> Tuple[float, float]:
    """Crude autocorrelation F0: (median log-F0 of voiced frames, voiced fraction)."""
    if x.size < frame:
        return 0.0, 0.0
    n = 1 + (x.size - frame) // hop
    lo, hi = int(sr / 400), int(sr / 60)
    f0s = []
    for i in range(n):
        f = x[i * hop: i * hop + frame]
        f = f - f.mean()
        e = float(np.dot(f, f))
        if e < 1e-6:
            continue
        spec = np.fft.rfft(f, 2 * frame)
        ac = np.fft.irfft(np.abs(spec) ** 2)[: frame]
        ac /= ac[0] + 1e-12
        seg = ac[lo:hi]
        k = int(np.argmax(seg))
        if seg[k] > 0.35:
            f0s.append(np.log(sr / (lo + k)))
    if not f0s:
        return 0.0, 0.0
    return float(np.median(f0s)), len(f0s) / n


class MfccEmbedder:
    name = "mfcc"

    def __init__(self) -> None:
        self.mu: Optional[np.ndarray] = None
        self.sd: Optional[np.ndarray] = None

    def __call__(self, chunks: List[np.ndarray]) -> np.ndarray:
        feats = []
        for c in chunks:
            m = mfcc(c)
            lf0, vf = pitch_stats(c)
            feats.append(np.concatenate([m.mean(0), m.std(0), [lf0 * 4.0, vf]]))
        F = np.asarray(feats)
        if self.mu is None:  # statistics of the main recording, reused for enrollment
            self.mu, self.sd = F.mean(0), F.std(0) + 1e-6
        return (F - self.mu) / self.sd


# --------------------------------------------------------------------------- clustering

def _unit(X: np.ndarray) -> np.ndarray:
    return X / (np.linalg.norm(X, axis=1, keepdims=True) + 1e-9)


def cluster_two(X: np.ndarray, iters: int = 20) -> Tuple[np.ndarray, np.ndarray]:
    """Cluster unit-normalised embeddings into 2 groups.

    Agglomerative (average linkage, cosine) initialisation, refined with spherical
    k-means; falls back to k-means++ if a cluster ends up degenerate.
    Returns (labels, centroids)."""
    from sklearn.cluster import AgglomerativeClustering, KMeans

    Xn = _unit(X)
    n = len(Xn)
    if n < 2:
        base = Xn[:1] if n else np.zeros((1, X.shape[1]))
        return np.zeros(n, dtype=int), np.vstack([base, base])
    labels = AgglomerativeClustering(n_clusters=2, metric="cosine", linkage="average").fit_predict(Xn)
    if min(np.bincount(labels, minlength=2)) < max(2, 0.05 * n):
        labels = KMeans(n_clusters=2, n_init=10, random_state=0).fit_predict(Xn)
    for _ in range(iters):
        C = _unit(np.stack([Xn[labels == k].mean(0) if np.any(labels == k) else Xn[0] for k in (0, 1)]))
        new = np.argmax(Xn @ C.T, axis=1)
        if min(np.bincount(new, minlength=2)) == 0 or np.array_equal(new, labels):
            break
        labels = new
    C = _unit(np.stack([Xn[labels == k].mean(0) for k in (0, 1)]))
    return labels.astype(int), C


def _median_filter_labels(labels: np.ndarray, k: int) -> np.ndarray:
    if k <= 1 or labels.size < k:
        return labels
    from scipy.signal import medfilt

    return medfilt(labels.astype(float), kernel_size=k | 1).round().astype(int)


# --------------------------------------------------------------------------- main API

def _embed_reference(path: str, embed, cfg: DiarizationConfig) -> np.ndarray:
    x, sr = load_wav(path)
    if sr != SR:
        raise ValueError(f"enrollment audio must be {SR} Hz wav (got {sr}) - pipeline converts it")
    regs = energy_vad(x, sr) or [Interval(0.0, x.size / sr)]
    wins = make_windows(regs, cfg.window_s, cfg.hop_s, cfg.min_window_s, x.size / sr)
    chunks = [x[int(s * SR): int(e * SR)] for s, e in wins]
    E = embed(chunks)
    return _unit(_unit(E).mean(0, keepdims=True))[0]


def _choose_backend(cfg: DiarizationConfig):
    b = cfg.backend.lower()
    if b == "mfcc":
        return MfccEmbedder(), []
    if b in ("auto", "ecapa"):
        try:
            return EcapaEmbedder(cfg.ecapa_source), []
        except Exception as e:  # noqa: BLE001
            if b == "ecapa":
                raise
            msg = f"SpeechBrain ECAPA unavailable ({type(e).__name__}: {e}); using MFCC fallback"
            log.warning(msg)
            return MfccEmbedder(), [msg]
    raise ValueError(f"unknown diarization backend {cfg.backend!r}")


def _pyannote(wav: str, cfg: DiarizationConfig) -> Optional[DiarizationResult]:
    token = os.environ.get("HF_TOKEN")
    try:
        from pyannote.audio import Pipeline  # type: ignore
    except ImportError:
        return None
    if not token:
        return None
    log.info("Running pyannote speaker-diarization-3.1")
    try:
        pipe = Pipeline.from_pretrained("pyannote/speaker-diarization-3.1", token=token)
    except TypeError:  # older pyannote
        pipe = Pipeline.from_pretrained("pyannote/speaker-diarization-3.1", use_auth_token=token)
    mode = str(cfg.num_speakers).lower()
    ann = (pipe(wav, min_speakers=1, max_speakers=2) if mode == "auto"
           else pipe(wav, num_speakers=int(mode)))
    ann = getattr(ann, "speaker_diarization", ann)
    order: Dict[str, str] = {}
    segs: List[SpeakerSegment] = []
    for turn, _, spk in ann.itertracks(yield_label=True):
        if spk not in order:
            order[spk] = str(len(order))
        segs.append(SpeakerSegment(float(turn.start), float(turn.end), order[spk]))
    return DiarizationResult("pyannote", segs)


def diarize(
    wav_path: str,
    words: Sequence[Word],
    cfg: DiarizationConfig,
    ref_a_wav: Optional[str] = None,
    ref_b_wav: Optional[str] = None,
) -> DiarizationResult:
    mode = str(cfg.num_speakers).strip().lower()
    if mode not in ("auto", "1", "2"):
        raise ValueError(f"num_speakers must be auto, 1 or 2 (got {cfg.num_speakers!r})")
    if cfg.backend == "pyannote":
        res = _pyannote(wav_path, cfg)
        if res is None:
            raise RuntimeError("pyannote backend requested but pyannote.audio/HF_TOKEN unavailable")
        return res
    if cfg.backend == "auto" and os.environ.get("HF_TOKEN") and not (ref_a_wav and ref_b_wav):
        try:
            res = _pyannote(wav_path, cfg)
            if res is not None:
                return res
        except Exception as e:  # noqa: BLE001
            log.warning("pyannote failed (%s); falling back to embeddings", e)

    x, sr = load_wav(wav_path)
    if sr != SR:
        raise ValueError(f"diarization expects {SR} Hz audio, got {sr}")
    total = x.size / SR
    regions = (speech_regions_from_words(words, merge_gap=cfg.region_merge_gap) if words
               else energy_vad(x))
    if not regions:
        log.warning("No speech found for diarization")
        return DiarizationResult("none", [], notes=["no speech"])
    wins = make_windows(regions, cfg.window_s, cfg.hop_s, cfg.min_window_s, total)
    chunks = [x[int(s * SR): max(int(s * SR) + 1, int(e * SR))] for s, e in wins]
    embed, notes = _choose_backend(cfg)
    log.info("Diarization: %d windows over %d speech regions (%s)", len(wins), len(regions), embed.name)
    E = embed(chunks)
    En = _unit(E)

    enrolled = bool(ref_a_wav and ref_b_wav)
    if enrolled:
        ca = _embed_reference(ref_a_wav, embed, cfg)  # type: ignore[arg-type]
        cb = _embed_reference(ref_b_wav, embed, cfg)  # type: ignore[arg-type]
        C = np.stack([ca, cb])
        labels = np.argmax(En @ C.T, axis=1)
        names = ["A", "B"]
    elif mode == "1":
        labels = np.zeros(len(En), dtype=int)
        c = _unit(En.mean(0, keepdims=True))[0]
        C = np.stack([c, c])
        names = ["0", "1"]
    else:
        labels, C = cluster_two(E)
        names = ["0", "1"]
    sims = En @ C.T
    margin = np.abs(sims[:, 0] - sims[:, 1])
    centroid_sim = float(C[0] @ C[1])
    stats: Dict[str, Any] = {"mode": mode, "n_windows": int(len(En))}
    if not enrolled and mode != "1":
        stats.update(cluster_stats(E, labels, C, embed.name))

    # median smoothing of window labels within each region (windows are time-ordered)
    order = np.argsort([w[0] for w in wins])
    wins = [wins[i] for i in order]
    labels, margin = labels[order], margin[order]
    smoothed = labels.copy()
    reg_idx = np.zeros(len(wins), dtype=int)
    for i, (s, e) in enumerate(wins):
        c = 0.5 * (s + e)
        reg_idx[i] = next((k for k, r in enumerate(regions) if r.start - 0.5 <= c <= r.end + 0.5), -1)
    for k in np.unique(reg_idx):
        m = reg_idx == k
        smoothed[m] = _median_filter_labels(labels[m], cfg.median_filter)

    # frame-level voting (50 ms) restricted to speech regions
    step = 0.05
    nfr = int(np.ceil(total / step)) + 1
    votes = np.zeros((nfr, 2))
    for (s, e), lab, mg in zip(wins, smoothed, margin):
        i0, i1 = int(s / step), max(int(s / step) + 1, int(np.ceil(e / step)))
        i1 = min(i1, nfr)  # padded speech regions may run past the end of the audio
        n = i1 - i0
        if n <= 0:
            continue
        tri = 1.0 - np.abs(np.linspace(-1, 1, n)) * 0.8
        votes[i0:i1, lab] += tri * (0.2 + mg)
    speech = np.zeros(nfr, dtype=bool)
    for r in regions:
        speech[int(r.start / step): int(np.ceil(r.end / step))] = True
    lab_fr = np.where(votes.sum(1) > 0, np.argmax(votes, 1), -1)
    lab_fr[~speech] = -1
    segs: List[SpeakerSegment] = []
    i = 0
    while i < nfr:
        if lab_fr[i] < 0:
            i += 1
            continue
        j = i
        while j < nfr and lab_fr[j] == lab_fr[i]:
            j += 1
        segs.append(SpeakerSegment(round(i * step, 3), round(min(total, j * step), 3), names[lab_fr[i]]))
        i = j

    phrases_out: List[Dict[str, Any]] = []
    if words and cfg.word_refine:
        segs, phrases_out = _refine_with_phrases(x, words, embed, C, lab_fr, step, names, cfg)

    if not enrolled:
        # make "0" the speaker heard first (deterministic ids for --speaker-map AB/BA)
        first = next((s.label for s in segs), "0")
        if first == "1":
            for s in segs:
                s.label = "0" if s.label == "1" else "1"
            smoothed = 1 - smoothed
            for p in phrases_out:
                p["label"] = "0" if p["label"] == "1" else "1"
    win_out = [
        {"start": round(s, 3), "end": round(e, 3), "label": names[int(l)], "margin": round(float(m), 4)}
        for (s, e), l, m in zip(wins, smoothed, margin)
    ]
    dur = {n: sum(s.end - s.start for s in segs if s.label == n) for n in names}
    log.info("Diarization result: %s, centroid cos=%.2f", {k: round(v, 1) for k, v in dur.items()}, centroid_sim)
    if not enrolled:
        stats["speech_s"] = {k: round(v, 2) for k, v in dur.items()}
        if mode == "1":
            stats.update(verdict=1, reasons=["--num-speakers 1"])
        else:
            v, why = embedding_verdict(stats, embed.name, cfg)
            stats.update(verdict=v if mode == "auto" else 2, embedding_verdict=v, reasons=why)
            if v == 1:
                msg = "embeddings suggest a single speaker: " + "; ".join(why)
                log.info(msg)
                notes.append(msg)
            elif centroid_sim > 0.85:
                msg = (f"the two speaker clusters are very similar (cos={centroid_sim:.2f}); "
                       "speaker separation may be unreliable")
                log.warning(msg)
                notes.append(msg)
    if phrases_out:
        n_ph = sum(p["source"] == "phrase" for p in phrases_out)
        notes.append(f"phrase-level refinement: {n_ph}/{len(phrases_out)} phrases decided by "
                     "embedding, rest by window votes")
    return DiarizationResult(embed.name, segs, enrolled, win_out, centroid_sim, notes, phrases_out,
                             stats)


def cluster_stats(E: np.ndarray, labels: np.ndarray, C: np.ndarray, backend: str) -> Dict[str, Any]:
    """Separation statistics of a 2-cluster split (used for the 1-vs-2 speaker test).

    * ``centroid_cos``: cosine between the two (unit) cluster centroids.  Forcing one
      voice into 2 clusters still leaves both centroids close (they average the same
      speaker); two different voices give clearly lower values.  Meaningless for the
      MFCC backend, whose features are standardised per recording.
    * ``silhouette``: mean silhouette (cosine for ECAPA, euclidean for MFCC)."""
    out: Dict[str, Any] = {"centroid_cos": round(float(C[0] @ C[1]), 4)}
    counts = np.bincount(labels, minlength=2)
    out["window_counts"] = [int(c) for c in counts]
    if min(counts) >= 2 and len(labels) >= 4:
        from sklearn.metrics import silhouette_score

        X = _unit(E) if backend != "mfcc" else E
        out["silhouette"] = round(float(silhouette_score(
            X, labels, metric="cosine" if backend != "mfcc" else "euclidean")), 4)
    else:
        out["silhouette"] = float("nan")
    return out


def embedding_verdict(stats: Dict[str, Any], backend: str, cfg: DiarizationConfig) -> Tuple[int, List[str]]:
    """1 or 2 speakers from the clustering alone (pure; unit tested).  The pipeline may
    still overrule it with per-camera mic / mouth-motion evidence."""
    why: List[str] = []
    sp = stats.get("speech_s", {})
    if sp:
        vals = sorted(sp.values())
        minor, total = vals[0], sum(vals)
        if minor < cfg.min_speaker_s:
            why.append(f"2nd cluster has only {minor:.1f}s of speech (< {cfg.min_speaker_s}s)")
        elif total > 0 and minor / total < cfg.min_speaker_share:
            why.append(f"2nd cluster is {100 * minor / total:.1f}% of speech (< {100 * cfg.min_speaker_share:.0f}%)")
    if min(stats.get("window_counts", [0, 0])) < 2:
        why.append("2nd cluster has fewer than 2 windows")
    if backend == "mfcc":
        sil = stats.get("silhouette", float("nan"))
        if not np.isnan(sil) and sil < cfg.single_silhouette:
            why.append(f"silhouette {sil:.2f} < {cfg.single_silhouette}")
    else:
        cc = stats.get("centroid_cos", float("nan"))
        if not np.isnan(cc) and cc >= cfg.single_centroid_cos:
            why.append(f"centroid cosine {cc:.2f} >= {cfg.single_centroid_cos}")
    return (1 if why else 2), why


def phrase_chunks(words: Sequence[Word], gap: float, max_len: float) -> List[Tuple[float, float]]:
    """Split words into phrases at pauses > ``gap`` or after sentence punctuation,
    then cut phrases longer than ``max_len`` into roughly equal word-aligned chunks."""
    phrases: List[List[Word]] = []
    for w in sorted(words, key=lambda w: w.start):
        if phrases and (w.start - phrases[-1][-1].end <= gap
                        and not phrases[-1][-1].text.endswith((".", "?", "!"))):
            phrases[-1].append(w)
        else:
            phrases.append([w])
    out: List[Tuple[float, float]] = []
    for ph in phrases:
        L = ph[-1].end - ph[0].start
        k = max(1, int(np.ceil(L / max_len)))
        per = int(np.ceil(len(ph) / k))
        for i in range(0, len(ph), per):
            sub = ph[i: i + per]
            out.append((sub[0].start, sub[-1].end))
    return out


def _refine_with_phrases(x, words, embed, C, lab_fr, step, names, cfg):
    """Re-label speech at ASR-phrase granularity: embed each phrase and compare it to
    the speaker centroids; fall back to the window votes when the margin is small."""
    chunks = phrase_chunks(words, cfg.phrase_gap, cfg.phrase_max)
    total = x.size / SR
    audio = []
    for s, e in chunks:
        if e - s < 0.4:  # very short: widen a little for a usable embedding
            c = 0.5 * (s + e)
            s2, e2 = max(0.0, c - 0.2), min(total, c + 0.2)
        else:
            s2, e2 = s, e
        audio.append(x[int(s2 * SR): max(int(s2 * SR) + 160, int(e2 * SR))])
    E = _unit(embed(audio))
    sims = E @ C.T
    segs: List[SpeakerSegment] = []
    out: List[Dict[str, Any]] = []
    for (s, e), sim in zip(chunks, sims):
        margin = float(sim[0] - sim[1])
        i0, i1 = int(s / step), max(int(s / step) + 1, int(np.ceil(e / step)))
        fr = lab_fr[i0:i1]
        fr = fr[fr >= 0]
        vote = int(np.bincount(fr, minlength=2).argmax()) if fr.size else int(margin < 0)
        if abs(margin) >= cfg.refine_min_margin:
            lab, src = (0 if margin > 0 else 1), "phrase"
        else:
            lab, src = vote, "windows"
        segs.append(SpeakerSegment(round(s, 3), round(e, 3), names[lab]))
        out.append({"start": round(s, 3), "end": round(e, 3), "label": names[lab],
                    "margin": round(margin, 4), "source": src, "window_vote": names[vote]})
    return segs, out
