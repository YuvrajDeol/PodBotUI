# Podcast Auto-Editor (AI multicam)

Give it three recordings of a two-person podcast: a close-up of **Speaker A**, a close-up of
**Speaker B** and a **wide shot** of both. It transcribes the audio, works out who is talking,
and cuts an edit:

* Speaker A talking → **cam A**, with **blue** captions (`#2D9CFF`)
* Speaker B talking → **cam B**, with **white** captions (`#FFFFFF`)
* question → answer exchanges, rapid back-and-forth, overlapping speech, the intro, long
  silences and the outro → **wide shot**, with each speaker's captions in their own colour.

**Only have one camera?** Give it just the wide shot. Single-clip mode finds the two faces and
crops a virtual close-up for each person out of the wide frame; the edit and captions then
work exactly the same way. See [Single-clip mode](#single-clip-mode-one-wide-shot).

Everything runs locally: faster-whisper for ASR, SpeechBrain ECAPA for speaker embeddings,
OpenCV, ffmpeg. You don't need API keys. Public models are downloaded the first time they are
used: whisper `small` (~480 MB), ECAPA (~80 MB) and the YuNet face model (230 KB, only if the
visual strategy runs).

---

## 1. Setup (macOS)

```bash
# ffmpeg. The regular formula has no libass, so captions are burned in with the built-in
# Pillow fallback. ffmpeg-full adds libass (the `ass` filter); it is keg-only and is
# auto-detected at /opt/homebrew/opt/ffmpeg-full/bin/ffmpeg.
brew install ffmpeg
brew install ffmpeg-full          # optional, recommended

cd "podcast_autoeditor"           # note: the parent folder name may contain spaces - quote paths
python3 -m venv --system-site-packages .venv      # reuses the global torch / numpy / opencv
.venv/bin/python -m pip install -r requirements.txt
# speechbrain + torchaudio WITHOUT deps (torchaudio wheels pin an exact torch version and a
# normal install would downgrade torch; only pure-python parts of torchaudio are used):
.venv/bin/python -m pip install --no-deps speechbrain==1.1.1 torchaudio==2.11.0
```

Check: `ffmpeg -hide_banner -filters | grep -E ' (ass|subtitles) '` (and the same with
`/opt/homebrew/opt/ffmpeg-full/bin/ffmpeg`). If neither binary has libass, nothing breaks:
captions are drawn with Pillow and overlaid. See [Captions](#captions).

## 2. Quick start with the synthetic demo

```bash
.venv/bin/python scripts/make_demo.py --out-dir demo                  # ~40 s, uses macOS `say`
.venv/bin/python -m podcast_autoeditor --cam-a demo/cam_a.mp4 --cam-b demo/cam_b.mp4 \
    --wide demo/wide.mp4 --out out/demo_edit.mp4                       # ~30 s on Apple silicon
.venv/bin/python scripts/verify_demo.py --demo-dir demo --out out/demo_edit.mp4   # 25 checks
```

## 3. CLI

```bash
# three clips (multicam)
python -m podcast_autoeditor --cam-a A.mp4 --cam-b B.mp4 --wide W.mp4 [--audio master.wav] \
       --out out.mp4 [options]
# one wide clip (auto-reframe): no --cam-a/--cam-b
python -m podcast_autoeditor --wide W.mp4 [--audio master.wav] --out out.mp4 [options]
```

`--mode auto|multicam|single` (default auto) picks single-clip mode when neither `--cam-a` nor
`--cam-b` is given. Giving only one of them is an error.

Run `python -m podcast_autoeditor --help` for every option. The main ones:

| group | options |
|---|---|
| I/O | `--work-dir DIR` (cache, default `<out>_work/`), `--plan-only` (EDL + captions, no render), `--force` (ignore cache), `--keep-intermediates`, `--ffmpeg PATH`, `-v`, `--log-file` |
| sync | `--no-sync`, `--sync-window 60`, `--offset-a/--offset-b/--offset-wide SECONDS`, `--sync-min-confidence 2`, `--force-sync` |
| ASR | `--whisper-model small` (tiny/base/small/medium/large-v3…), `--language en`, `--compute-type int8`, `--beam-size 5`, `--no-vad`, `--asr-prompt`, `--no-condition-previous`, `--no-word-refine` |
| speakers | `--diarization auto\|ecapa\|mfcc\|pyannote`, `--diar-window 1.5`, `--diar-hop 0.5`, `--ref-a a.wav --ref-b b.wav` (voice enrollment), `--num-speakers auto\|1\|2`, `--mapping-strategy auto\|mic\|visual\|first`, `--speaker-map auto\|AB\|BA`, `--swap-speakers`, `--mic-margin-db 4`, `--visual-margin 0.2`, `--visual-timeout 120` |
| editing | `--lead 0.15`, `--min-shot 1.2`, `--qa-gap 1.5`, `--qa-answer-hold 2.0`, `--qa-max-question 6`, `--question-heuristic on\|off`, `--no-qa`, `--rapid-turns 3`, `--rapid-window 6`, `--no-rapid`, `--no-overlap-wide`, `--silence-wide 2.5`, `--turn-gap 1.0` |
| captions | `--color-a '#2D9CFF'`, `--color-b '#FFFFFF'`, `--font Arial`, `--font-size-pct 5.5`, `--max-words 6`, `--max-chars 32`, `--max-line-dur 2.5`, `--no-captions`, `--caption-burner auto\|ass\|pillow` |
| single clip | `--a-side left\|right`, `--crop-scale 3.2`, `--max-zoom 3`, `--crop-a X,Y,W,H`, `--crop-b X,Y,W,H`, `--static-crops`, `--preview-crops out.png` |
| render | `--width 1920 --height 1080 --fps 30`, `--encoder auto\|libx264\|h264_videotoolbox`, `--crf 20`, `--preset medium`, `--audio-source auto\|external\|wide\|cam-a\|cam-b`, `--jobs N` |

**Offset convention:** an offset is where the clip's own t=0 falls on the reference
timeline. A camera that started recording 1.3 s late has `+1.3`; one that started 0.6 s
early has `-0.6`.

**Caching:** extracted audio, sync offsets, the raw transcript, diarization and the speaker
mapping are stored as JSON in the work dir. Each entry is keyed by input file fingerprints
(path, size, mtime) and the parameters that affect it. Per-shot video segments are cached by
content hash. Re-running with different editing or caption settings skips ASR and diarization
and only re-plans and re-renders.

## 4. Web UI

```bash
.venv/bin/python app.py            # http://127.0.0.1:7860  (bound to localhost only)
```

One page, three numbered steps. It follows the system light/dark setting and works offline
(fonts are served locally).

1. **Upload**: one dropzone. Drop **1 wide clip** (single-clip mode: close-ups are cropped from
   it) or **3 clips** (multicam). With 3 clips, the *Speaker A close-up*, *Speaker B close-up* and
   *Wide shot* dropdowns are pre-filled from the file names (`cam_a`, `cam_b`, `wide`, ...). 2 or
   4+ files, or the same file in two roles, show an inline message and keep the buttons disabled.
2. **Speakers & captions**: in single-clip mode a frame from the clip and two tiles, *Who is
   Speaker A? Left person / Right person*. Caption colour pickers with a live preview.
3. **Edit**: **Edit my podcast** renders the video; **Preview cut plan** plans cuts and captions
   without rendering (fast; use it to check the framing). **Advanced settings** (collapsed) holds
   everything else: *Accuracy* (Fast/Balanced/Best = whisper `base`/`small`/`large-v3`),
   language, people talking, swap A/B, *Cut style* presets (Calm / Balanced / Snappy, with the six
   rule sliders under *Fine-tune*), *Quality* (1080p / 720p / 4K at 30 fps, or Custom), master
   audio and manual sync offsets, and the single-clip close-up size and max zoom. Untouched, the
   settings are exactly the CLI defaults; other whisper models and the remaining options are
   CLI-only.

While it runs you see the five stages (prepare & sync, transcribe, identify speakers, plan cuts,
render) with a percentage and elapsed time. The result shows summary chips, the video, a **cut
timeline** (click a shot, or a transcript line, to jump the video there), download buttons for
the video, captions (.srt), cut list (.csv) and transcript (.json), a warnings card (with
**Swap A and B, re-run** when the speaker mapping is not confident), the crop preview in
single-clip mode, and a *Details* tab with sync offsets, mapping, the EDL table, every output
file and the log. Errors appear in a panel with technical details; the form stays filled in for
a retry. Each run is written to `ui_runs/<timestamp>/`. Styling lives in `ui/podbot.css` and
`ui/podbot.js`, the HTML widgets in `ui/render.py`.

## 5. How it works

```
clips ──ffmpeg──► 16 kHz mono wavs ──GCC-PHAT──► offsets
reference audio ──faster-whisper──► words (+ punctuation)  ──energy refine──► word times
reference audio + words ──ECAPA windows → 2 clusters → phrase re-scoring──► speaker 0/1 segments
cam-A/cam-B mics (or mouths) ──► speaker 0/1 → A/B
words + A/B segments ──► labelled words → turns ──planner──► shots ──► EDL, captions ──► render
```

**Reference timeline / master audio.** This is the external `--audio` if you give one,
otherwise the wide shot's audio. If the wide shot has no audio, cam A's audio is used, then
cam B's. ASR and diarization run on this track, and the output carries it as one continuous
audio track.

**Sync (`sync.py`).** Generalised cross-correlation with partial PHAT whitening. This is robust
to the different gain and EQ of a lapel mic versus a room mic. Only lags within ±`sync-window`
are searched. By default the first 180 s are used; if confidence is low, two later chunks are
tried too. Confidence is the peak-to-sidelobe ratio. An estimate below
`--sync-min-confidence` (2) is **not applied**: the clip gets offset 0 and a warning naming the
`--offset-*` flag to pass, unless you add `--force-sync`. A clip with no audio also gets offset
0, with a warning (also shown in the UI) to pass its `--offset-*` value. Manual `--offset-*`
values always win.

**ASR (`transcribe.py`).** faster-whisper, CPU int8, `word_timestamps=True`, VAD filter on.
It is conditioned on previous text and given a short punctuated primer prompt so that it keeps
emitting `?` and `.`; without that, Whisper sometimes drops all punctuation after a long pause.
Whisper also tends to stretch the first word after a pause back over the silence, which would
make cuts land early. Words that are much longer than their text suggests are trimmed against
the audio energy.

**Diarization (`diarize.py`).**
1. Speech regions come from the ASR words.
2. ECAPA embeddings (`speechbrain/spkrec-ecapa-voxceleb`) are computed on 1.5 s windows with a
   0.5 s hop.
3. The windows are clustered into 2 groups: agglomerative, cosine, average linkage, refined by
   spherical k-means.
4. Labels are median-smoothed and turned into frame votes.
5. Each ASR phrase (split on pauses > 0.15 s, sentence ends, and at most 3 s long) is embedded
   and assigned to the nearest centroid. If its margin is small, the window votes decide.

This phrase step is what makes short turns ("Easily.") come out right.

**One or two speakers? (`--num-speakers auto|1|2`, default auto).** Clustering always finds 2
groups, even in a monologue. `auto` therefore tests the split:
* **Embeddings:** the cosine between the two ECAPA cluster centroids. A single voice forced
  into 2 clusters keeps both centroids close.
  - Calibrated with `scripts/calibrate_num_speakers.py`: 16 macOS voices × (5 s, 30 s)
    monologues gave **0.68–0.89**. 8 two-voice pairs × (10 s, 30 s) plus the 4 demo podcasts
    gave **0.04–0.55**.
  - Threshold: `single_centroid_cos = 0.62`. The MFCC fallback uses a silhouette threshold of
    0.22 instead.
  - A second cluster with less than 1 s of speech also counts as one speaker.
  - Exception: Reed/Eddy, two near-identical Apple voices, scored 0.74/0.88. ECAPA cannot tell
    them apart.
* **Per-camera mics:** both clusters dominate the *same* close-up mic (|A−B| ≥ 6 dB with the
  same sign, score < 3 dB). This is how the reviewer's 16 s and 5 s single-speaker clips were
  caught. The second one has only 3 s of speech, too little for the embedding test.
* **Two clearly separated voices** on the mics (score ≥ 4 dB) override an embedding "single"
  verdict.
* **Visual:** without mic evidence, mouth motion that follows the speech on only one close-up
  also means one speaker.

With one speaker, every word goes to that person. Their camera is chosen by mic SNR (speech
level over each mic's own noise floor, so it does not depend on mic gain), then by mouth motion,
then `--speaker-map AB|BA`. If none of these decides, cam A is used with a loud warning.

Fallbacks and options:
* If SpeechBrain is missing or fails, an MFCC mean/std + pitch embedding is used, with a warning.
* `--diarization pyannote` uses pyannote 3.1 if `pyannote.audio` is installed and `HF_TOKEN` is
  set. This path is untested here.
* `--ref-a/--ref-b` enrollment clips label speech as A/B by cosine similarity. No clustering
  and no mapping step is needed.

**Speaker → camera mapping (`mapping.py`).**
1. **mic**: for each cluster, compare the RMS level of the cam-A audio with the cam-B audio
   during that cluster's speech (with sync offsets applied). The score
   `(A−B)₀ − (A−B)₁` cancels absolute gains, and a margin above **4 dB** decides. Distinct
   voices on separate mics score about twice the own-vs-bleed separation (19–29 dB in the
   demos); one voice split in two scored 0.0–2.2 dB.
2. **visual**: YuNet face detection (Haar cascade if YuNet can't be loaded) on each close-up at
   5 fps over up to 300 s. Mouth-region motion energy from the YuNet mouth landmarks (or the
   lower third of the face box) is correlated with each cluster's speaking activity.
   * The decision needs a score ≥ 0.2, and every camera with faces must agree in sign (each
     term ≥ 0.05). Near-zero or contradicting evidence (e.g. the reviewer's −0.09) is never
     "confident".
   * Decoding is streamed, with a wall-clock budget (`--visual-timeout`, 120 s) and a cap of
     2000 frames per camera.
3. **first**: assume the first speaker heard is A, with a loud warning.

`--speaker-map AB|BA` overrides this (AB means the first speaker heard is A), and
`--swap-speakers` or the UI checkbox flips the result. The decision and its evidence go to the
log and to `*_run_summary.json`.

**Word labelling (`labeling.py`).** Each word takes the speaker with the most time overlap.
An isolated run of at most 1 word (≤ 0.5 s) with no pause on either side, surrounded by the
other speaker, is flipped back. A real "Yes." answer has pauses around it and survives. Turns
are consecutive same-speaker words, split on gaps > 1.0 s. Overlapping speech is detected from
overlapping A/B words, and from the close-up mics (**mic coupling**):
1. From frames where only B talks, learn how loud B's voice is on mic A *relative to mic B*.
   Do the same for A on mic B.
2. In every frame, predict mic A's level from mic B as if only B were talking. The *excess*
   above that prediction is about 0 dB when only B talks and large when A talks too.
3. A frame is overlapped when both mics show an excess above max(3 dB, 0.4 × separation) and
   both carry speech energy, for at least 0.3 s.

This is invariant to mic gains and works down to 3 dB of own-vs-bleed separation. The old
midpoint method is the fallback when the mics cannot be calibrated.

### Editing rules (`planner.py`, a pure function)

The timeline is discretised to output frames. It starts all WIDE, and the rules below are
painted on in order, with later rules winning:

| # | rule | parameters |
|---|---|---|
| 1 | Intro before the first word, silences longer than `silence_wide`, and the outro stay **WIDE** | `silence_wide=2.5`, `silence_hold=0.5`, `outro_hold=0.5` |
| 2 | Each turn → its speaker's close-up, from `start − lead` until the next turn's `start − lead` | `lead=0.15` |
| 3 | Overlapping speech → WIDE | `min_overlap=0.3` |
| 4 | **Rapid back-and-forth**: ≥ `rapid_turns` speaker changes within `rapid_window` s → WIDE from 1 s before the first change to 1 s after the last (clipped to those turns) | `rapid_turns=3`, `rapid_window=6`, `rapid_pad=1` |
| 5 | **Question → answer** → WIDE (see below) | `qa_gap=1.5`, `qa_answer_hold=2.0`, `qa_max_question=6` |

After painting:
* Cameras without footage at that time (because of offsets or clip length) fall back to WIDE.
* Shots shorter than `min_shot` (1.2 s) are resolved. A short close-up merges into a WIDE
  neighbour, or into identical neighbours on both sides, or into the longer neighbour. A short
  WIDE grows by borrowing frames from a long neighbour instead of disappearing, so WIDE is
  preferred.
* Adjacent same-camera shots are merged.

The result covers `[0, duration]` with no gaps or overlaps, and every cut is on a frame
boundary. `validate_shots()` asserts this.

**Interpretation of "question/answer":** a *question* is a turn whose **final sentence** ends
with `?`. Whisper sometimes drops the `?`, so with `--question-heuristic on` (the default) the
final sentence also counts as a question if it:
* **opens like a question:** a wh-word followed by an auxiliary ("what do you…", "how long
  does…", "what's…"), or an auxiliary followed by a subject ("do you…", "is it…", "can I…").
  Up to two leading fillers such as "so", "and" or "well" are skipped. In an unpunctuated run of
  words, the question onset is also searched among the last 12 words.
* **or ends in a tag:** ", right", ", correct", "isn't it", "don't you", "or not", "huh".

Statements like "What I mean is…" or "When I was young…" and exclamations ("Do it now!") are
not questions. It is a *question→answer exchange* when the **other** speaker's turn starts within `qa_gap` (1.5 s)
after the question ends. The edit goes WIDE from the start of the question sentence (at most
`qa_max_question` s before its end) through the first `qa_answer_hold` (2 s) of the answer, or
to the end of the answer if it is shorter, then cuts to the answerer's close-up. A question
followed by the same speaker, or by a long pause, is not an exchange.

### Captions

`captions.py` breaks words into lines of at most 6 words, 32 characters or 2.5 s. Lines also
break on speaker change, on sentence punctuation (and on a comma after ≥ 3 words), and on
pauses > 0.7 s. Each line lingers 0.25 s after its last word and stays at least 0.6 s, without
overlapping the same speaker's next line.

Style: bold Arial, about 5.5 % of the frame height, 3 px black outline (scaled to the
resolution), a soft shadow, bottom-centre. A is `#2D9CFF` and B is `#FFFFFF`, both
configurable. **ASS colours are `&HAABBGGRR`**, so `#2D9CFF` becomes `&H00FF9C2D`; the converter
is unit-tested.

When A's and B's lines overlap in time, each new line takes the lowest free row, so the two
stack instead of colliding, and a line never moves while it is on screen. The `.ass` file sets
an explicit `MarginV` per event.

Burn-in uses libass (the `ass` filter) when the ffmpeg build has it. Otherwise each distinct
caption state is drawn to a transparent PNG strip with Pillow, using the same layout, and
overlaid through an ffconcat slideshow in a single `overlay` filter. Both paths are tested.

### Render (`render.py`)

**Per-shot intermediates + concat demuxer**, then one final encode. Each shot is cut with an
accurate input seek (sync offset applied) and normalised: `fps` → `scale` with letterbox
`pad` → `setsar=1` → yuv420p. Rotation metadata is honoured through ffmpeg autorotate. Each
shot is encoded with an exact frame count (`-frames:v N`); shots render in parallel and are
cached.

The final pass reads the segments through the concat demuxer and re-stamps video PTS as
`N/fps`. Frame index maps to timeline time exactly, so per-segment rounding cannot add up to
A/V drift. It then burns the captions and muxes **one continuous master audio track**, so a cut
never touches the audio.

Output is libx264 (or `h264_videotoolbox`) + AAC with `+faststart`. A single `filter_complex`
with hundreds of trim/split branches was rejected: it decodes every input once per branch and
buffers frames, it can't be parallelised or resumed, and one error kills the whole graph.

## Single-clip mode (one wide shot)

```bash
.venv/bin/python -m podcast_autoeditor --wide wide.mp4 --out out.mp4 --plan-only   # check the framing
open out_crop_preview.png
.venv/bin/python -m podcast_autoeditor --wide wide.mp4 --out out.mp4               # render
```

**Who is A and who is B.** Speaker A is the person on the **left** of the frame and Speaker B
the person on the right. `--a-side right` swaps that. A gets the blue captions and the "cam A"
close-up, as in multicam mode.

**Finding the people (`reframe.py`).**
1. The wide clip is scanned at 5 fps for the first 5 minutes of speech and at 1 fps for the
   rest. The scan is streamed, with a frame cap (2400) and a wall-clock budget (240 s).
2. Every face in every sampled frame is detected with YuNet (Haar cascade as fallback).
3. Detections are grouped into people by **horizontal position**. People sit still in a
   podcast, so a position cluster survives missed detections and microphones in front of a
   face better than frame-to-frame tracking does.
4. The two most persistent and largest clusters are kept. A third face (a poster, someone
   walking past) is ignored, with a warning if it is persistent.

**Which voice belongs to which face.** There are no per-camera microphones in this mode, so the
tool measures mouth motion inside each person's face region and correlates it with each
diarized voice. This uses the same strict rules as the multicam visual strategy: score ≥ 0.2
and both faces must agree. If the evidence is not confident, the log and UI warn loudly and the
first voice heard is assumed to be Speaker A. Override with `--speaker-map AB|BA` (AB = the
first voice heard is Speaker A) or `--swap-speakers`. One-speaker detection still works, from
the embeddings and mouth motion.

**Framing.** Each close-up is a crop with the output aspect ratio, so 9:16 output works too:
* height = `--crop-scale` × face height (default 3.2, head and shoulders);
* eyes on the upper third, centred on the face, clamped inside the frame, even pixel values;
* shifted away from the other person, and tightened down to 2.2× if needed, so the other face
  stays out of shot. If that is impossible, a warning says so;
* `--max-zoom` (default 3) caps the upscaling: the crop is never smaller than output height ÷
  max-zoom;
* `--crop-a X,Y,W,H` / `--crop-b X,Y,W,H` set a crop by hand, in source pixels.

Crops are **locked off**: one crop per person from their median position, not a jittery
follow. If a person clearly moves (their face leaves the safe area of the crop for a whole
shot), the next close-up of that person is re-framed. A crop only ever changes at a shot
boundary, never mid-shot. `--static-crops` turns re-framing off.

**Resolution.** A close-up is an enlargement of part of the frame. From a 1080p wide shot a
1080p close-up is typically upscaled 1.5–2.5×, which looks soft. **Use a 4K wide shot for 1080p
output.** The tool warns when a crop is upscaled more than 2×. Crops are scaled with lanczos.

**Fallbacks.**

| faces found | result |
|---|---|
| none | a warning, and the whole edit stays on the wide shot; captions are still colour-coded per speaker |
| one | that person gets a virtual close-up; the other speaker's turns stay wide, with a warning |
| three or more | the two main ones are used, with a warning |

**What is the same as multicam.** The shot plan and editing rules, captions, the single
continuous master audio and frame accuracy are unchanged. An external `--audio` master works
(the wide clip is synced to it). The EDL records the crop rectangle of every close-up
(`crop_x, crop_y, crop_w, crop_h`).

## 6. Outputs (next to `--out`)

| file | content |
|---|---|
| `<name>.mp4` | edited video |
| `<name>_edit_decision_list.json` / `.csv` | shots: camera, reason, start/end (s, frames, timecode), source file and source in/out, plus meta (offsets, mapping evidence, detected events) |
| `<name>_captions.ass` / `.srt` | styled captions; the SRT has `Speaker A:` / `Speaker B:` labels |
| `<name>_transcript.json` | words with speaker, turns, caption lines, ASR segments, diarization segments, overlaps |
| `<name>_run_summary.json` | outputs, stats, offsets, mapping decision with evidence, full config |
| `<name>_crop_preview.png` | single-clip mode only: the wide frame with the Speaker A / B crop rectangles and tracked face boxes |

## 7. Tests and verification

```bash
.venv/bin/python -m pytest -q                       # 141 tests (planner + question heuristic, captions,
                                                    #  sync, labelling + overlaps, mapping + speaker count,
                                                    #  diarization, refinement, render smoke, visual budget,
                                                    #  single-clip tracks / crops / re-framing / mode)
.venv/bin/python scripts/make_demo.py --out-dir demo            # synthetic podcast + ground truth
.venv/bin/python scripts/verify_demo.py --demo-dir demo --out out/demo_edit.mp4
.venv/bin/python scripts/stress_render.py --demo-dir demo --shots 300   # drift test, 300 cuts
.venv/bin/python scripts/make_demo.py --out-dir demo_mono --monologue A --monologue-len short  # 1 speaker
.venv/bin/python scripts/make_demo.py --out-dir demo_overlap --extra-overlaps   # 2 extra interjections
.venv/bin/python scripts/calibrate_num_speakers.py   # re-measure the 1-vs-2 speaker statistics
# single-clip mode: ONE wide clip with two real face photos, then edit and verify it
.venv/bin/python scripts/make_demo.py --out-dir sw/left_first --single-wide
.venv/bin/python -m podcast_autoeditor --wide sw/left_first/wide.mp4 --out out/sw.mp4
.venv/bin/python scripts/verify_demo.py --demo-dir sw/left_first --out out/sw.mp4   # add --a-side right if used
```

The demo uses two macOS voices: A is Samantha and B is Daniel. Cam A hears A loudly and B at
−15 dB; cam B is the reverse; the wide shot is a balanced mix. Cam A starts 0.6 s early, and cam
B starts 1.3 s late, at 25 fps and 960×540. The script has monologues, four question→answer
pairs, a rapid exchange, a 3.5 s silence and an overlapping interjection.

Every frame carries a barcode of its true reference time, so `verify_demo.py` checks rendered
frames for the right camera (from the background colour), frame-accurate sync (from the
barcode) and the right caption colours. Other variants:
* `--photo-faces` uses real face photos (the Olivetti set, via scikit-learn) with animated
  mouths, to exercise the visual mapping strategy.
* `--voice-a/--voice-b/--bleed` makes harder variants.
* `--monologue A|B --monologue-len short|long` makes a single-speaker recording (~4 s or ~28 s
  of speech).
* `--extra-overlaps` adds two interjections spoken over the other person.
* `--single-wide` writes only `wide.mp4`: two real face photos (left and right) whose mouths
  move with their own voice, mixed audio, and a time-code strip above each face so that
  cropped close-ups can be checked for sync. Variants: `--first-speaker-side right` (the
  person on the right speaks first), `--hide-face left|right` (one detectable face) and
  `--move-at SECONDS` (the right-hand person changes seat).

For single-clip runs `verify_demo.py` also checks the face tracks and crop rectangles against
the known face positions, and compares each sampled output frame with the expected render
(the source frame cropped by the EDL rectangle and scaled).

Results measured on this machine:

| demo | measured |
|---|---|
| default demo, full pipeline | all 26 checks pass (round 2, fresh work dir) |
| sync | offsets recovered exactly (0.0 ms error) |
| words | 188/188 attributed to the right speaker |
| shots | 30/30 sampled frames show the EDL camera; worst barcode error 20 ms (< 1 source frame) |
| captions | 25/25 frames have the right colour |
| other variants | Pillow burner, videotoolbox, MFCC backend, external master, enrollment, manual offsets: all pass |
| photo-face demo | visual mapping chose correctly (mouth-motion correlation +0.34 vs −0.48) and flips when cam A and cam B files are swapped |
| same-gender demo | Samantha vs Shelley with −9 dB bleed: 100 % word attribution; the overlap is now detected (26/26; it was missed in round 1) |
| overlap demos | 3 deliberate overlaps at −15 dB and at −9 dB bleed: all 3 WIDE, no false overlap regions (28/28 each) |
| single speaker | 4 monologues (Karen/Samantha as A, Daniel/Moira as B; ~4 s and ~28 s of speech): 1 speaker detected, every word and caption in the speaker's colour, the whole speech on their close-up (17/17 each) |
| reviewer scenarios | independent demo and Moira/Karen demo: 37/37 each with the reviewer's checker; 16 s and 5 s single-speaker clips: all words on the right speaker and camera (before: 19/48 words on the wrong speaker) |
| single-clip demos | one wide clip, two face photos: 188/188 words on the right person, both faces tracked in 100 % of frames, crops contain the own face and exclude the other; 28/28 sampled output frames match the expected source crop (mean abs diff ≤ 0.14/255), time code read through the crop within 3 ms; 29/29 checks each for: left person speaks first, right person speaks first, `--a-side right`, a person who changes seat (re-framed only at shot boundaries) |
| single-clip fallbacks | one detectable face: that speaker gets close-ups, the other stays wide (28/28); no detectable face: one WIDE shot with both caption colours and a loud warning; external master audio: wide synced at +1.700 s; 540×960 output: exact 9:16 crops |
| speaker-count calibration | ECAPA embedding verdict alone: 51/54 synthetic cases correct (the misses are Reed/Eddy and a clip with 3 s of speech; both are decided by the mic test in multicam use) |
| stress render | 300 random cuts: exact frame count, 60/60 sampled frames correct |

## 8. Limitations

* **Exactly two speakers.** Guests, a third voice, music beds and laughter tracks are not modelled.
* **Validated only on synthetic TTS audio.** Real rooms (reverb, crosstalk, compressed audio,
  similar voices, short backchannels like "mm-hm") will make diarization less accurate.
  Enrollment clips (`--ref-a/--ref-b`) help most.
* **Overlap detection needs separate close-up mics** (at least 3 dB own-vs-bleed
  separation). Without them only overlapping ASR words count, and Whisper usually transcribes
  just one of two overlapping voices. A quiet interjection under a loud talker (more than
  about 6 dB quieter at −9 dB bleed) can still be missed. Previously, in the −9 dB-bleed demo,
  the interjection overlap was missed and B's
  interjection got a short close-up instead.
* **Question detection is heuristic.** It uses `?` from Whisper plus the opening/tag
  heuristic above. Rhetorical questions and imperatives that start with an auxiliary followed by
  a subject ("Have you…") also trigger WIDE when the other person answers within 1.5 s.
* **Speaker count:** two near-identical voices recorded without separate mics (like Reed/Eddy)
  are treated as one speaker; force `--num-speakers 2`. The MFCC fallback often misjudges
  single-voice clips under ~5 s. With separate per-camera mics, both cases are decided by the
  mic test instead.
* **The visual mapping needs visible frontal faces.** It is skipped when faces are found in
  fewer than 30 % of the sampled frames. The cartoon faces in the default demo are not
  detected. The face models are downloaded on first use; if offline, the next strategy runs.
* **The mic strategy needs separate mics.** It requires the close-up cameras to record
  different audio. If both carry the same mixed feed it is inconclusive, and the visual or
  first-speaker strategy is used.
* **Sync searches ±60 s by default.** Clips that drift over time (clock skew) are not corrected:
  one constant offset per clip.
* **Single-clip mode:**
  - Close-ups are enlargements of the wide frame, so they are softer than a real second
    camera. Use a 4K wide shot for 1080p output.
  - Overlapping speech is only detected from overlapping ASR words. The mic-based detector
    needs separate close-up mics, so most short interjections are missed.
  - The voice→face mapping needs both mouths visible. Faces hidden by a microphone, in profile,
    or very small are found less often; with weak evidence the tool warns and guesses.
  - People are told apart by where they sit. Two people who swap seats mid-recording are not
    followed.
  - A locked-off crop excludes the other person's face, not their shoulders or hands.
  - There is no smoothed "follow" camera: crops only change at shot boundaries.
  - For portrait output the WIDE shots are letterboxed (the full frame is kept).
* **pyannote is untested.** The backend is optional and was not tested here (not installed, no
  token).
* **CPU only by default.** whisper `small` takes about 10–15 s for 80 s of audio on Apple
  silicon; `medium` and `large-v3` are much slower.

## 9. Troubleshooting

* **`ffmpeg/ffprobe not found`**: `brew install ffmpeg`, or pass `--ffmpeg /path/to/ffmpeg`
  (ffprobe must be next to it). You can also set `PODCAST_AUTOEDITOR_FFMPEG`.
* **`brew install` fails with "Cannot link X / Another version is already linked"** (seen on
  macOS 27): run `brew unlink X && brew link --overwrite X`, then re-run the install.
* **The cuts show the wrong person**: re-run with `--swap-speakers` (fast, everything is
  cached), or provide `--ref-a/--ref-b` voice samples. Check the `mapping` evidence in
  `*_run_summary.json`.
* **`LOW sync confidence … NOT applied`**: the clip's audio may be silent or unrelated, so the
  offset was set to 0. Set `--offset-a/--offset-b/--offset-wide` by hand (or in the UI's
  manual offset fields), use `--force-sync` to trust the estimate anyway, or use `--no-sync` if
  the cameras were genlocked.
* **"Single speaker detected" but two people talk** (or the reverse): force
  `--num-speakers 2` (or `1`). The evidence is in `*_run_summary.json` →
  `stats.speaker_count_reasons` and `mapping.evidence`.
* **No punctuation, so no Q→A wide shots**: try a bigger `--whisper-model` or set
  `--language`. The question heuristic is on by default; `--question-heuristic off` disables it.
* **Model downloads hang or fail**: the HF models download once into `~/.cache/huggingface`.
  On slow links, `HF_HUB_DISABLE_XET=1` can help. The face models are stored in
  `~/.cache/podcast_autoeditor/opencv/`, and you can copy them there by hand.
* **SpeechBrain import errors**: the tool falls back to MFCC diarization with a warning. To use
  ECAPA, reinstall with `pip install --no-deps speechbrain torchaudio==<version>`.
* **A run takes far longer than usual, with long pauses in the log**: on a laptop running on
  battery, macOS puts the machine into maintenance sleep and every process freezes (check with
  `pmset -g log | grep Sleep`). Keep it awake for the run: `caffeinate -i .venv/bin/python -m
  podcast_autoeditor ...`. The visual analysis has its own wall-clock budget
  (`--visual-timeout`).
* **The captions font looks wrong**: use `--font "Helvetica Neue"` (libass) or
  `--caption-burner pillow` (searches system TTF fonts).

## 10. Layout

```
podcast_autoeditor/
  app.py                    Gradio UI (127.0.0.1)
  ui/                       UI theme (podbot.css, podbot.js) + HTML widgets (render.py)
  requirements.txt
  podcast_autoeditor/
    __main__.py, cli.py     CLI
    pipeline.py             orchestration + caching
    config.py               all thresholds (dataclasses)
    datatypes.py            Word / Turn / Shot / SpeakerSegment
    ffmpeg_utils.py         discovery (prefers a libass build), probe, run, audio extraction
    sync.py                 GCC-PHAT offsets
    transcribe.py           faster-whisper + word-boundary refinement
    diarize.py              ECAPA / MFCC / pyannote, clustering, phrase re-scoring, enrollment
    mapping.py              mic-level / mouth-motion / fallback speaker->camera mapping
    labeling.py             word->speaker, flip smoothing, turns, overlaps
    reframe.py              single-clip mode: face tracks, close-up crops, crop preview
    planner.py              shot planner (pure)
    captions.py             chunking, layout, hex->ASS, .ass/.srt/.json
    caption_overlay.py      Pillow caption burn-in fallback
    render.py               per-shot segments, concat, final mux, EDL
  scripts/  make_demo.py  verify_demo.py  stress_render.py  calibrate_num_speakers.py
  tests/    pytest suite
```
