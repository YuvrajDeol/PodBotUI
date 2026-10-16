# Agent A design proposals - PodBot UI redesign

Inspiration: neutomni.com (studied live in the built-in browser, desktop + 375px phone; CSS tokens read from its `:root`). Inspiration only - no logo, copy or imagery reused.
Screenshots: the browser tool returns images inline only, so none were saved to `design/screens/` (folder created, empty).

## 1. Design language (neutomni -> PodBot)

Observed on neutomni: a single saturated red `#cf1830` used for ALL text and for full-bleed panels; white `#ffffff` page; off-white `#f8f4f4` secondary; sky blue accent `#6db1f0`; cream `#f0ede6` text on red. Fonts: "Editorial New" serif (headings, with italic "Italic Local" for the witty second half of a line), "Vazirmatn" sans for body (16px, line-height 120%), "Fira Code" mono for small labels (13px, uppercase, wide tracking: "LOCAL TIME", "OPEN FOR WORK", "CUT ALONG LINE"). Headings regular weight (400), huge (52px h1, 118px display). Zero border-radius, no shadows, 1px hairlines and dashed "tear here" dividers, bracketed text CTAs `[ Let's talk ]`, ticket/file metaphors (pricing as tear-off tickets, FAQ as `FAQ.md`), diamond/dot pager glyphs, cookie banner as a red slab with outlined buttons. Mood: editorial, tactile, flat, witty, retro-tech.

Mapping to PodBot (podcast = editorial + broadcast "cut" metaphor; the app literally cuts along a line):

| Token | Light | Dark |
|---|---|---|
| --bg | #FFFFFF | #0E0E10 |
| --surface (cards) | #F8F4F4 | #17171A |
| --ink (text) | #1B1B1D | #F0EDE6 |
| --brand (red, CTAs, headings accents) | #CF1830 | #FF4A5E (lighter for AA on dark) |
| --on-brand | #F0EDE6 | #0E0E10 |
| --line | #E3DCDC (dashed for dividers) | #2A2A2F |
| --speaker-a (default caption blue) | #2D9CFF (matches pipeline default) | same |
| --speaker-b | #FFFFFF with 1px ink outline | #FFFFFF |
| --wide-shot (timeline) | #9A9A9F grey | #6A6A70 |
| --ok / --warn | #1F8F5F / #B26A00 | #4CC38A / #F5A524 |

Fonts (self-host or Google Fonts via Gradio `head`): headings "Instrument Serif" (free stand-in for Editorial New; italic for emphasis), fallback `Georgia, serif`; body "Inter" or "Vazirmatn", fallback `system-ui, sans-serif`; labels "Fira Code" / "JetBrains Mono", fallback `ui-monospace, monospace`. Scale: display 56/40 (desktop/phone), h2 28, body 16, label 12-13 mono uppercase +0.08em tracking.
Radii: 0 on panels (neutomni signature), 2px on inputs for usability; pill only for the speaker colour chips. Shadows: none; use hairline borders and a 2px offset "paper" shadow only on the result card. Motion: 150-200ms ease on hover (invert button fill), progress "tape" stripe animation, result slides up; honour `prefers-reduced-motion`. Light default, dark via `prefers-color-scheme` + manual toggle (Gradio's `?__theme=dark`).

## 2. Current UI inventory and verdicts (app.py, Gradio 6.29.1)

Observed: one 1835px-tall dark page; four equal upload boxes in a row, then a wall of 7 mixed controls, then collapsed "Editing rules" with 13 fields, then result stack (video, files, markdown summary, preview, EDL table, transcript, log). Confusing: Mode is a radio above everything; single-mode options appear in a hidden row far from the upload; "Master audio" has same weight as required files; Whisper/language/speakers/swap/plan-only sit beside colours; emoji title; the log and EDL dominate the result.

| Control | Verdict | Reason / default |
|---|---|---|
| Mode radio (multicam / single) | Replace with smart default | Auto-detect from drop: 1 video = single, 3 = multicam; show a small "Detected: 1 wide clip" chip with a "change" link. |
| Cam A upload | Keep (main, multicam only) | Revealed when 2+ files dropped; one multi-file dropzone assigns by filename (cam_a/cam_b/wide) with a reorder UI. |
| Cam B upload | Keep (main, multicam only) | Same. |
| Wide upload | Keep (main) | The only file in single mode. |
| Master audio | Move to Advanced | Optional; rarely used. |
| A-side radio (left/right) | Keep in main flow, single mode only | The one real question: "Who is Speaker A?" Show as two big tiles (left person / right person) over a frame thumbnail. Default left. |
| Close-up size slider | Move to Advanced | Default 3.2. |
| Max zoom slider | Move to Advanced | Default 3.0. |
| Caption colour A | Keep (main, optional) | Round swatch chips with 5 presets + custom. Default #2D9CFF. |
| Caption colour B | Keep (main, optional) | Default #FFFFFF. |
| Whisper model | Move to Advanced, relabel "Accuracy vs speed" | Presets: Fast=base, Balanced=small (default), Best=large-v3. Hide raw model names. |
| Language | Move to Advanced | Default auto. |
| Speakers (auto/1/2) | Move to Advanced | Default auto. |
| Swap speakers | Replace with smart default + post-result action | Auto mapping already reports confidence; show "Swap A and B" button on the result page and re-run (cache makes it fast). |
| Plan only | Replace | Becomes a "Preview framing / cut plan" secondary button next to Edit. |
| Q->A max gap, wide hold, rapid turns, rapid window, min shot, silence->wide | Move to Advanced, grouped as one 3-way "Cut style" preset (Calm / Balanced / Snappy) with a "custom" disclosure holding the six sliders | Defaults 1.5 / 2.0 / 3 / 6.0 / 1.2 / 2.5 = Balanced. |
| Question heuristic | Move to Advanced | Default on. |
| Manual offsets A/B/wide | Move to Advanced ("Sync fix") | Default blank = auto. |
| Width / Height / FPS | Replace with a "Quality" select in Advanced (1080p30 default, 720p30, 4K30 if source allows) | Three raw numbers invite invalid combos. |
| Auto-edit button | Keep, restyle | Primary `[ Edit my podcast ]`, disabled until valid, reason in helper text. |
| Edited video | Keep (hero of result) | |
| Downloads file list | Replace | Labelled buttons: Video, Captions (.srt), EDL, Transcript, "All (zip)". |
| Summary markdown | Replace | Short status chips (speakers, shots, mapping confidence) + warnings card. |
| Crop preview image | Keep, relocate | Shown at upload step in single mode (thumbnail with two crop boxes) rather than hidden. |
| EDL dataframe | Move to "Details" tab | Power users only; the timeline strip replaces it for normal use. |
| Transcript | Keep in result tab | Speaker-coloured, click line to seek. |
| Log | Move to Details tab / collapsed | |

Primary flow target: 1) Drop video(s) 2) (optional) Who's Speaker A + caption colours 3) Edit. Everything else under one "Advanced" drawer.

## 3. Design ideas

1. **One smart dropzone, auto mode.** One large multi-file dropzone; 1 file -> single mode, 3 files -> multicam with filename-guessed roles shown as removable labelled cards (swap roles via dropdown). Why: removes the mode radio and 4 boxes. Effort M. Risk: wrong filename guess (mitigate with editable labels); 2 files = error state "need 1 or 3". Gradio: `gr.File(file_count="multiple")` + `.upload` handler returning `gr.update` for HTML role cards; feasible.
2. **Three-step vertical stepper** (01 Upload / 02 Speakers / 03 Edit) with mono step numbers and dashed dividers, steps unlock progressively. Why: clear primary flow. S-M. Low risk. Feasible with `gr.Group`/`gr.Column(visible=)` + CSS counters.
3. **Neutomni-style theme layer.** Custom `gr.themes.Base` subclass (radius 0, red primary, serif heading font) plus ~150 lines of CSS via `css=` and `elem_classes`, Google Fonts through `head=`. Why: most of the "better looking" for least work. S. Risk: Gradio class names change between versions - pin 6.29.1 and target our own `elem_classes`. Feasible.
4. **Speaker A picker on a frame thumbnail (single mode).** Extract one frame client-side or via ffmpeg after upload; show it with two overlaid halves "A" and "B" that flip on click, plus caption colour chips tinted live. Why: replaces left/right radio, makes it obvious. M. Risk: needs a tiny ffmpeg frame grab in app.py (not the pipeline). Feasible with `gr.Image` + `gr.HTML`; an HTML overlay with JS click is also possible via `head` script but Gradio-state sync is fiddly (use hidden `gr.Radio` toggled by JS).
5. **Advanced drawer with presets.** One `gr.Accordion("Advanced")` containing Accuracy (3 presets), Cut style (3 presets + custom sliders), Quality, Language, Speakers, Sync offsets, Master audio, Close-up size/zoom. Why: hides 20+ controls. S. Low risk; defaults identical to today's so output is unchanged. Feasible.
6. **Live progress "tape" with named stages.** pipeline already calls `progress(f, msg)`; render a custom `gr.HTML` list: Syncing -> Transcribing -> Who's talking -> Planning cuts -> Rendering, current stage highlighted with a striped bar and elapsed time; cancel button (`cancels=`). Why: the default Gradio spinner gives little. M. Risk: matching stage names to free-form messages - map by keyword with generic fallback. Feasible (yield HTML from a generator instead of `gr.Progress`).
7. **Result page with cut timeline strip.** Below the video, a horizontal bar built from the EDL (columns camera, start, end): blue = A, white/outlined = B, grey = wide, with hover tooltips and click-to-seek (JS sets `video.currentTime`); legend and per-camera seconds. Why: shows what the edit did at a glance, replaces the EDL table. M. Risk: seek JS needs a `gr.HTML` script hook/`js=`; fine. Feasible.
8. **Download bar and result summary chips.** Labelled buttons (Download video, Captions, EDL, Transcript, ZIP) as `gr.DownloadButton`s; chips for speakers, shots, "mapping confident / not confident". Not-confident chip carries a "Swap A and B" action. Why: finding things in a generic file list is slow. S. Low risk. Feasible.
9. **Transcript as speaker-coloured, click-to-seek lines** in a Transcript tab (A blue label, B ink label), with EDL table and log in a "Details" tab. Why: matches the editorial feel, hides noise. M. Risk: parse `transcript_text` format. Feasible via `gr.HTML` + JS seek.
10. **Empty, error and warning states.** Empty: dashed "tear here" dropzone with copy like "Drop your recording. We'll find the cuts." and sample-clip link. Validation: inline red helper under the dropzone ("Need 1 wide clip or 3 clips") and Edit button disabled. Errors: red slab (like the neutomni cookie banner) with plain-language message and "Show technical details" for the exception. Warnings: amber card listing de-duplicated pipeline warnings, each with a suggested fix (e.g. low-confidence mapping -> Swap). Why: current `gr.Error` toast vanishes. S-M. Feasible (`gr.HTML` status area; keep `gr.Error` as fallback).
11. **Phone layout.** Single column, 16px gutters, sticky bottom `[ Edit ]` bar, stepper collapses to "Step 2 of 3" header, timeline strip scrolls horizontally with min-height 44px targets, video full-bleed. Why: Gradio default rows already stack but the controls are tiny. S-M via CSS media queries on our classes. Feasible. Risk: file upload on phones is huge videos - label size limits.
12. **Accessibility pass.** Contrast: brand red #CF1830 on white is 5.9:1 (passes AA), cream on red 5.0:1; never use A-blue #2D9CFF as text on white (2.8:1) - use it as fill with dark label, or darken to #0B6FD0 for text. Visible 2px focus ring (`:focus-visible` outline offset 2px), keyboard-operable dropzone and speaker tiles (real buttons/radios, not div clicks), `aria-live="polite"` on the progress stage text, timeline segments exposed as a list with text labels (not colour alone: use letters A/B/W inside segments), `prefers-reduced-motion`. S-M. Feasible; custom HTML pieces need hand-added ARIA.
13. **Signature micro-details** from the inspiration: mono uppercase meta labels ("LOCAL RUN 14:50", "DEMO CLIP"), bracketed CTAs, dashed "CUT ALONG LINE" divider that animates a scissor-cut line left to right when Edit is pressed, dots/diamond pager for the 3 steps. Why: personality at near-zero cost. S. Risk: don't overdo it; keep the red cut line decorative (aria-hidden). Feasible with CSS.
14. **Re-run actions on result.** "Try again with: swap A/B, snappier cuts, plan only" chips that re-submit with one changed setting (work dir cache makes it fast). Why: replaces upfront guesswork. M. Risk: needs state of last config in `gr.State`. Feasible.

## 4. Architecture options

(a) **Restyled Gradio app (custom theme + CSS + a few `gr.HTML` widgets).** Pros: `run_edit` and queue/progress/file serving already work; app.py edit only, no new server; fastest; low regression risk; supports every idea above except pixel-perfect bespoke interactions. Cons: Gradio DOM/class names are brittle across versions; stepper/timeline/click-to-seek need JS hacks; mobile polish is limited; some neutomni layout (full-bleed red panels, huge type) fights Gradio's container.
(b) **Custom HTML/JS front end + FastAPI** calling `run_pipeline` in a background thread, polling/SSE for progress, serving `ui_runs` files. Pros: exact visual control, true timeline/seek, best mobile and a11y. Cons: reimplement uploads (multipart, large files), job queue, progress streaming, file serving, cancel, CORS/security (bind 127.0.0.1); about 2-3x the effort and a new test surface; Gradio dependency may still be needed by the existing tests.

**Recommendation: (a) for the college timeline.** Build a theme + CSS file loaded from app.py (`css=`, `head=` for fonts), the one-dropzone auto mode, the Advanced drawer, an HTML progress/timeline/summary renderer, and keep `run_edit`'s signature and the pipeline untouched (add presets by mapping to the same config values). Keep (b) as stretch only if time remains; if so, reuse (a)'s Python helpers (EDL -> timeline, warnings -> cards).

Suggested build order: 3 -> 5 -> 1 -> 2 -> 10 -> 6 -> 8 -> 7 -> 9 -> 11/12 -> 4 -> 13/14.

## 5. Wireframes

Desktop main (>=1024px):
```
 PODBOT                                   LOCAL / OFFLINE   [light|dark]
 ----------------------------------------------------------------------
  Your podcast,                     01 --- 02 --- 03
  cut along the line.               Upload  Speakers  Edit
  ----------------------------------------------------------------------
  01  UPLOAD
  +- - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - -+
  :   Drop 1 wide clip - or 3 clips (cam A, cam B, wide)           :
  :                      [ Browse files ]        try sample clip   :
  +- - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - -+
  Detected: 1 wide clip (virtual cameras)   change
  - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - -
  02  SPEAKERS  (optional)
  [ frame thumbnail with  | A  left person | B right person | ]   (flip)
  Caption colours:  A (o)(o)(o)(o)(o)+   B (o)(o)(o)(o)(o)+
  - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - -
  03  EDIT                      [ Edit my podcast ]  [ Preview cuts ]
  v Advanced  (accuracy, cut style, quality, language, sync, audio)
```
Phone (375px):
```
 PODBOT            [=]
 Step 1 of 3 .o.
 Cut along the line.
 +- - - - - - - - - -+
 : Drop video(s)     :
 : [ Browse ]        :
 +- - - - - - - - - -+
 Detected: 1 clip
 Who is Speaker A?
 [ Left ] [ Right ]
 A (o)(o)(o)+
 B (o)(o)(o)+
 > Advanced
 =====================
 [  Edit my podcast  ]   <- sticky
```
Progress (replaces step 3):
```
 EDITING ... 02:14
 [x] Syncing   [x] Transcribing   [>] Who's talking   [ ] Cuts   [ ] Render
 ////////////////------------------   (striped tape)       [ Cancel ]
```
Result:
```
 DONE - 14 shots - 2 speakers - mapping confident        [Edit another]
 +----------------------------------------------+ Downloads
 |                 VIDEO                        | [ Video  mp4 ]
 |                                              | [ Captions   ]
 +----------------------------------------------+ [ EDL ][ Transcript ]
 TIMELINE  |AAAA|wide|BBB|wide|AAAAAA|B|wide|...|   A blue  B white  W grey
 Tabs: [ Transcript ] [ Details: EDL / log ]
  A  00:03  Welcome to the show...        (click to seek)
  B  00:09  Thanks for having me...
 Warning card (if any) -> [ Swap A and B ] [ Snappier cuts ]
```
Not done: ran no single-clip job (inspected progress code and the idle UI only), and no screenshots were saved to disk.
