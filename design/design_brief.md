# PodBot UI - implementation brief (Agent B, design lead)

Audience: Agent C (build), Agent D (verify against section 6).
Inputs reviewed: `design/agent_a_proposals.md`, `app.py`, `podcast_autoeditor/{pipeline,config,cli,render,captions,ffmpeg_utils}.py`, neutomni.com (live, desktop).
Inspiration only: no neutomni logo, name, copy, images or its custom fonts.

---

## 1. Decision summary

**Architecture: restyled Gradio 6.29.1 (custom theme + one CSS file + a few `gr.HTML` renderers), all changes in `app.py` plus new `ui/` assets.**
Gradio already gives us the upload transport for multi-GB videos, the job queue, file serving from `ui_runs`, and the 127.0.0.1 binding; a FastAPI/HTML rewrite would re-implement all of that for a single-user local tool and add a new failure surface next to a pipeline that works. Everything the request needs (one calm page, fewer controls, a real progress view, a better result) is reachable with a theme, scoped CSS on our own `elem_id`/`elem_classes`, `gr.HTML` widgets, and one global JS file for click-to-seek. The pipeline package is not modified.

Accepted (A's numbers):
| # | Idea | Status / modification |
|---|---|---|
| 1 | One dropzone, auto mode | **Accepted, modified.** Mode is derived from file count (1 = single, 3 = multicam). Roles are assigned with **three dropdowns** (Speaker A close-up / Speaker B close-up / Wide shot) pre-filled by filename guess, then upload order. No drag-reorder UI, no role "cards". 2 or 4+ files = inline validation message. |
| 2 | Three steps | **Accepted, modified.** Static numbered sections `01 Upload`, `02 Speakers & captions`, `03 Edit`. No progressive locking, no pager dots, no "Step 2 of 3". |
| 3 | Theme layer | **Accepted, modified.** Near-square (2 px) radii, not 0; **sans headings, no serif** (see 2.2); fonts are Gradio's bundled local fonts, no Google Fonts. |
| 4 | Speaker A picker on a frame | **Accepted, modified.** Thumbnail (ffmpeg frame grab in `app.py`) + a `gr.Radio` styled as two large tiles "Left person" / "Right person". No JS click-on-image overlay. Thumbnail fails soft (hidden). |
| 5 | Advanced drawer with presets | **Accepted, modified.** Presets map to exact values (section 4). Cut-style sliders stay reachable ("Fine-tune"). Quality preset keeps a "Custom" option exposing width/height/fps. |
| 6 | Live staged progress | **Accepted without cancel.** Pipeline cannot be interrupted (no hook); a Cancel that leaves a render running is misleading. |
| 7 | Cut timeline strip | **Accepted.** Built from `res.edl`; segments are `<button>`s with click-to-seek. |
| 8 | Download bar + summary chips | **Accepted minus ZIP.** 4 `gr.DownloadButton`s; every other output file stays in Details. |
| 9 | Transcript coloured, click-to-seek | **Accepted.** Parsed from `transcript_text`; raw `<pre>` fallback. |
| 10 | Empty/error/warning states | **Accepted.** Error panel replaces the vanishing `gr.Error` toast for pipeline failures. |
| 11 | Phone layout | **Accepted minus sticky bottom bar** (fights Gradio's scroll container and covers content). |
| 12 | Accessibility pass | **Accepted.** Contrast numbers corrected in 2.1. |
| 13 | Micro-details | **Partly.** Keep: mono uppercase meta labels, dashed "cut line" dividers, "LOCAL · OFFLINE" tag. Drop: bracketed `[ CTA ]` text, animated scissors. |
| 14 | Re-run chips | **Partly.** Only **"Swap A and B, re-run"**. Drop "snappier cuts"/"plan only" chips. |

Rejected outright: no whole idea; the rejected **parts** are: #1 role cards + reorder (fragile, poor keyboard support); #2 progressive locking/pager (hides context, extra state); #3 zero radii, serif headings, Google Fonts (see below); #4 JS click overlay (Gradio state sync); #6 Cancel (cannot stop the pipeline); #8 ZIP (low value); #11 sticky bar (overlaps content); #13 bracketed CTAs (read aloud as "bracket"), scissor animation (noise); #14 extra re-run chips (scope). Also rejected from A section 1/2: manual light/dark toggle (follow system; Gradio `?__theme=` still works), swatch-chip colour pickers (JS-state sync; use `gr.ColorPicker` + live preview instead), crop preview at the upload step (it needs the pipeline's face scan; it lives in the result, and "Preview cut plan" is the fast way to get it).

Where B overrules A (main three):
1. **Fonts/headings:** no serif, no Google Fonts. neutomni's big headings are actually sans (serif is only the italic accent line), the bundled Lora has no italic, and the app must look right offline. Use IBM Plex Sans + IBM Plex Mono, which Gradio ships locally.
2. **Role assignment:** A's filename-guess cards + reorder UI become three plain dropdowns with a distinctness check. That's dependable, keyboard-friendly and testable.
3. **Red stays brand-only, never data or error-only:** red `#CF1830` is used for the primary button, focus ring, active progress stage and accents. Speaker/camera colours (blue A, white B, grey wide) are used only in caption-related visuals. Errors always carry an icon + heading + tinted panel, so they are never "just red".

---

## 2. Design tokens

Define in `ui/podbot.css` on `:root` (light) and `.dark` (Gradio sets `.dark` on `<body>`/`.gradio-container` from system preference). Also map the main ones into the theme via `gr.themes.Base(...).set(...)` (section 6).

### 2.1 Colour
| Token | Light | Dark | Use |
|---|---|---|---|
| `--pb-bg` | `#FFFFFF` | `#0F0E10` | page |
| `--pb-surface` | `#F8F4F4` | `#18171A` | cards, dropzone, accordion |
| `--pb-ink` | `#1B1B1D` | `#F0EDE6` | body text, headings |
| `--pb-muted` | `#5E5A5B` | `#A9A3A0` | helper text, meta labels |
| `--pb-line` | `#E3DCDC` | `#2C2A2E` | hairlines, dashed dividers (decorative) |
| `--pb-line-strong` | `#8C8587` | `#6E686B` | input borders, dropzone border (needs >= 3:1) |
| `--pb-brand` | `#CF1830` | `#FF5468` | primary button fill, focus ring, active stage, links |
| `--pb-brand-hover` | `#A8122A` | `#FF7383` | primary hover |
| `--pb-on-brand` | `#FFFFFF` | `#0F0E10` | text on brand fill |
| `--pb-ok` | `#1F7A4F` | `#4CC38A` | done ticks, "mapping confident" |
| `--pb-warn-ink` / `--pb-warn-bg` | `#8A5300` / `#FFF4E0` | `#F5A524` / `#2A2112` | warnings card |
| `--pb-err-ink` / `--pb-err-bg` | `#A8122A` / `#FDECEE` | `#FF7383` / `#2A1418` | error panel |
| `--pb-cam-a` | user's caption colour A (default `#2D9CFF`) | same | timeline/chips fill only, never text |
| `--pb-cam-b` | user's caption colour B (default `#FFFFFF`) + 1 px `--pb-line-strong` outline | same | timeline/chips fill only |
| `--pb-cam-wide` | `#9A9A9F` with 45deg hatch | `#6A6A70` hatch | timeline fill only |

WCAG AA (computed):
| Pair | Ratio | Result |
|---|---|---|
| ink on bg (L / D) | 17.2 / 16.5 | AA |
| ink on surface (L / D) | 15.8 / 15.3 | AA |
| muted on bg (L / D) | 6.8 / 7.7 | AA |
| muted on surface (L / D) | 6.2 / 7.2 | AA |
| brand text on bg (L / D) | 5.49 / 6.16 | AA |
| on-brand on brand button (L / D) | 5.49 / 6.16 | AA |
| on-brand on brand-hover (L) | 7.54 | AA |
| ok on bg (L / D) | 5.31 / 8.69 | AA |
| warn-ink on warn-bg (L) / warn on bg (D) | 5.81 / 9.44 | AA |
| err-ink on err-bg (L) | 6.61 | AA |
| line-strong vs bg (non-text, L / D) | 3.61 / 3.54 | AA 1.4.11 |
| ink on default cam-A blue (chip label) | 5.99 | AA |
| cam-A blue `#2D9CFF` as text on white | 2.87 | **forbidden**; use as fill with ink label |

(A quoted 5.9:1 for red on white; it is 5.49:1, still AA.)

### 2.2 Typography (all offline)
- Sans: `gr.themes.LocalFont("IBM Plex Sans", weights=(400, 600))` then `system-ui, -apple-system, "Segoe UI", sans-serif`.
- Mono: `gr.themes.LocalFont("IBM Plex Mono", weights=(400, 600))` then `ui-monospace, Menlo, Consolas, monospace`.
- No `GoogleFont`, no `<link>` to fonts.googleapis.com, no network font request at all.
- Scale (desktop / <=640px): display h1 44/32 px, weight 600, letter-spacing -0.02em, line-height 1.05; h2 (section titles) 22/20 px 600; body 16 px / 1.5; helper 14 px; meta label 12 px mono, uppercase, letter-spacing 0.08em; step number `01` 12 px mono in `--pb-brand`.

### 2.3 Spacing, radii, borders, shadows, motion
- Spacing scale 4 / 8 / 12 / 16 / 24 / 32 / 48 px. Page max-width 1080 px, centred; side gutter 32 px desktop, 16 px <=640px. Section gap 32 px; inside cards 16-24 px.
- Radii: 2 px on cards, inputs, buttons, video; 999 px only on status chips; 0 on dividers.
- Borders: 1 px `--pb-line` for cards; 1 px `--pb-line-strong` for inputs; dropzone 1.5 px dashed `--pb-line-strong`, turns `--pb-brand` on hover/drag-over. Section dividers: 1 px dashed `--pb-line` with a mono meta label at the right (e.g. `CUT ALONG LINE`-style, our wording: `01 / UPLOAD`).
- Shadows: none, except the result video card: `4px 4px 0 var(--pb-line)` (flat "paper" offset).
- Focus: `:focus-visible { outline: 2px solid var(--pb-brand); outline-offset: 2px; }` on every interactive element including timeline/transcript buttons.
- Motion: 150 ms ease-out for hover/fill changes; progress bar stripe animation 1 s linear; result fade-in 200 ms. Under `@media (prefers-reduced-motion: reduce)` all animations/transitions off (stripe becomes static).

---

## 3. Screen-by-screen spec

Page shell (all states): header row: wordmark text `PodBot` (sans 600, 20 px) left; mono meta `LOCAL · OFFLINE` right. Below: h1 `Podcast in. Edited show out.` + one muted line: `Drop your recordings. PodBot syncs them, finds who's talking and cuts between close-ups and the wide shot.` No emoji anywhere. Browser title `PodBot`.

### 3.1 Empty / upload state
- `01 / UPLOAD` section. One `gr.File(file_count="multiple", file_types=["video"], type="filepath", elem_id="pb-drop")`, label `Your video`, styled as a large dashed dropzone (min-height 200 px desktop, 160 px phone).
- Helper under it (muted): `1 wide clip with both people, or 3 clips: Speaker A close-up, Speaker B close-up and a wide shot.`
- `02` and `03` sections are rendered but `02` shows only the caption colours (no speaker tiles yet).
- Primary button `Edit my podcast` (`elem_id="pb-run"`) disabled; helper next to it: `Add your video to start.`
- Secondary button `Preview cut plan` (`elem_id="pb-plan"`) disabled, same rule.

### 3.2 Files-chosen state
On `pb-drop.change` (`_on_files`):
- **1 file -> single-clip mode.** Show chip `DETECTED · 1 WIDE CLIP · CLOSE-UPS CROPPED AUTOMATICALLY`. Section 02 shows: thumbnail `gr.Image` (`elem_id="pb-thumb"`, non-interactive, frame at 10% of duration grabbed with `find_ffmpeg()` + `ffmpeg -ss T -i wide -frames:v 1 -vf scale=640:-2`; hidden on any failure) and radio `Who is Speaker A?` with choices `Left person` / `Right person` (values map to `left` / `right`, default `Left person`), info `Speaker A gets the Speaker A caption colour and close-up.` Styled as two tiles >= 44 px high.
- **3 files -> multicam.** Chip `DETECTED · 3 CAMERAS`. Section 01 shows three dropdowns (`elem_id`s `pb-role-a`, `pb-role-b`, `pb-role-wide`), labels `Speaker A close-up`, `Speaker B close-up`, `Wide shot`, choices = uploaded basenames. Pre-fill rule (case-insensitive basename): wide = name containing `wide`, `master`, `both`, `full`; A = name containing `cam_a`, `cam-a`, `cama`, `speaker_a`, or ending `_a`/`-a` before the extension; B the same with `b`. Any role still unassigned gets the remaining files in upload order (A, B, wide). Demo files `cam_a.mp4 / cam_b.mp4 / wide.mp4` must map correctly. If two roles pick the same file: inline error `Each role needs a different file.` and buttons disabled.
- **2 or 4+ files:** inline error under dropzone (`--pb-err-ink`, icon): `PodBot needs 1 wide clip or 3 clips (A, B, wide). You added N.` Buttons disabled.
- **Removing files** returns to 3.1.
- Valid -> both buttons enabled, helper text cleared.
- `02` always also shows `Speaker A captions` / `Speaker B captions` (`gr.ColorPicker`, defaults `#2D9CFF` / `#FFFFFF`) and a live preview `gr.HTML` (`elem_id="pb-cap-preview"`): a 16:9 dark tile (`#111`) with two sample lines `A: So how did it start?` / `B: Honestly, by accident.` in bold sans, coloured with the picks, 2 px black text-stroke/shadow like the render. Updates on `.change`.
- `03 / EDIT`: the Advanced accordion (collapsed) sits **above** the button row so opening it never moves the button off-screen afterwards. Button row: `Edit my podcast` (primary) + `Preview cut plan` (secondary, info tooltip/helper `Plans cuts and captions without rendering video - fast.`).

### 3.3 Running / progress state
- Clicking either button hides section-level inputs' interactivity (buttons disabled, label `Editing...`), and shows `gr.HTML` `elem_id="pb-progress"` with `aria-live="polite"`:
  - Header mono `EDITING · mm:ss` (elapsed, updates each tick).
  - Five stages with state icons (done tick `--pb-ok`, active dot `--pb-brand`, pending hollow): `Prepare & sync`, `Transcribe`, `Identify speakers`, `Plan cuts`, `Render` (`Render` shows `Skipped` in plan-only).
  - Striped bar with width = pipeline fraction (0-100%), numeric `%` text beside it, current raw message in muted mono (e.g. `transcribing (small)`).
  - Stage mapping by message prefix: `probing inputs|extracting audio|synchronising clips` -> Prepare & sync; `transcribing` -> Transcribe; `identifying speakers|mapping speakers|finding faces|labelling words` -> Identify speakers; `planning shots` -> Plan cuts; anything after `planning shots` (`rendering segments`, `final encode`, render sub-messages) -> Render; `done` -> all done. Unknown message: keep current stage, show the text.
- Previous result panels are hidden at run start.

### 3.4 Result state
Top: summary row of chips: `DONE · 14 SHOTS`, `2 SPEAKERS` (1-speaker: `1 SPEAKER` with reasons as `title`), `MAPPING CONFIDENT` (`--pb-ok`) or `CHECK SPEAKERS` (`--pb-warn`), `ELAPSED 2:14`. Right: secondary `Edit another` (clears files and results, keeps settings).
1. **Video** `gr.Video(elem_id="pb-video")` full width of the content column (hidden in plan-only; replaced by note `Plan only: no video rendered.`).
2. **Timeline** `gr.HTML(elem_id="pb-timeline")`: horizontal strip, segment width proportional to duration, fills `--pb-cam-a` / `--pb-cam-b` / hatched wide; each segment is a `<button class="pb-seg" data-t="{start}" aria-label="Speaker A close-up, 0:03 to 0:09">` containing letter `A`/`B`/`W` when wide enough (>= 18 px). Legend below: `A close-up · 41 s`, `B close-up · 33 s`, `Wide · 20 s` (from `stats.camera_seconds`). Click/Enter seeks the video (6.3). Min height 44 px. On phone the strip scrolls horizontally (min 2 px per second, at least 100% width).
3. **Downloads** row of `gr.DownloadButton`s: `Video (.mp4)` <- `outputs["video"]`, `Captions (.srt)` <- `outputs["srt"]`, `Cut list (.csv)` <- `outputs["edl_csv"]`, `Transcript (.json)` <- `outputs["transcript"]`. Hide any whose file is missing.
4. **Warnings card** (only if warnings): amber card `Things to check`, up to 12 de-duplicated warning lines. If mapping not confident: line `PodBot isn't sure who is A and who is B.` + button `Swap A and B, re-run` (`elem_id="pb-swap-rerun"`): flips the Advanced `Swap Speaker A and B` checkbox, then re-runs with identical other inputs.
5. **Framing** (single mode only): crop preview `gr.Image` from `outputs["crop_preview"]`, caption `Speaker A = person on the left. Close-up crops drawn on the wide frame.`
6. **Tabs**: `Transcript` (default) and `Details`.
   - Transcript: `gr.HTML(elem_id="pb-transcript")`, one row per turn parsed by `^\[(\d+):(\d+(?:\.\d+)?)\] Speaker (\S+): (.*)$`; row = `<button class="pb-line" data-t=secs>` with mono time, speaker chip (fill = caption colour, ink label, outline), text. Unparseable -> escaped `<pre>`. Text HTML-escaped.
   - Details: sync offsets, mapping (`mapping` via `method`), diarization backend, events, crop coords (single), as a small definition list; `gr.Dataframe` EDL (same columns as today); `gr.File(file_count="multiple")` `All output files` (every existing path in `res.outputs`, incl. .ass, EDL json, summary json, crop preview); `gr.Textbox` `Log` (last 300 lines, 8 lines tall).

### 3.5 Error state
- Input validation (before running): inline under the relevant control (3.2), buttons disabled. No toast.
- Pipeline exception: `pb-progress` is replaced by error panel `gr.HTML(elem_id="pb-error", role="alert")`: `--pb-err-bg`, 1 px `--pb-err-ink` left border 4 px, icon `!` in a circle, heading `The edit didn't finish`, plain line `{ExceptionType}: {message}` (escaped, first 300 chars), hint line by type (`PipelineError` -> `Check the files and settings above, then try again.`; other -> `See technical details below.`), `<details><summary>Technical details</summary>` with last 40 log lines. Buttons re-enabled; inputs untouched so user can retry.
- Bad numeric input (custom offsets): inline message under the field `Enter a number of seconds, e.g. -1.25`, run blocked.

### 3.6 Phone layout (<=640px; check at 375 px)
- Single column, 16 px gutters, no horizontal page scroll (only the timeline strip scrolls inside its own box).
- h1 32 px; dropzone 160 px; role dropdowns stack; speaker tiles side by side (each 50%); colour pickers stack.
- Buttons full width, stacked: `Edit my podcast` then `Preview cut plan`; all targets >= 44 px tall.
- Result: video full content width; download buttons 2-per-row grid; chips wrap.

---

## 4. Controls table

Defaults below must give a `PipelineConfig` equal to today's untouched `app.py` (except run-dir timestamps).

| Today (app.py) | Default today | New home | New label / values | Maps to |
|---|---|---|---|---|
| Mode radio | multicam | **auto** | from file count (1 -> single, 3 -> multicam) | `PipelineConfig.mode` "single"/"multicam" (never "auto") |
| Cam A file | - | main (01) | dropdown `Speaker A close-up` | `cam_a` (`""` in single) |
| Cam B file | - | main (01) | dropdown `Speaker B close-up` | `cam_b` (`""` in single) |
| Wide file | - | main (01) | dropdown `Wide shot` / the 1 file | `wide` |
| Master audio | none | Advanced > Sync | `Master audio (optional)` gr.File audio | `audio` |
| A-side radio | left | main (02, single only) | `Who is Speaker A?` Left person / Right person | `reframe.a_side` left/right |
| Close-up size | 3.2 (2.4-4.5, 0.1) | Advanced > Single-clip framing (single only) | `Close-up size (x face height)` same range | `reframe.crop_scale` |
| Max zoom | 3.0 (1.0-5.0, 0.1) | Advanced > Single-clip framing (single only) | `Max zoom` same range | `reframe.max_zoom` |
| Caption colour A | #2D9CFF | main (02) | `Speaker A captions` | `captions.color_a` via `_hex` |
| Caption colour B | #FFFFFF | main (02) | `Speaker B captions` | `captions.color_b` via `_hex` |
| Whisper model | small | Advanced > Speech | `Accuracy`: Fast / **Balanced** / Best | `asr.model_size` = `base` / **`small`** / `large-v3`. `tiny`, `medium`, `distil-large-v3` -> CLI `--whisper-model` (info text says so; Best: `large download on first use`) |
| Language | "" (auto) | Advanced > Speech | `Language` dropdown, `allow_custom_value=True`: **Auto-detect**, en, hi, pa, es, fr, de, ja, zh (any typed code accepted) | Auto-detect / blank -> `None`; else stripped code |
| Speakers | auto | Advanced > Speech | `People talking`: **Auto** / 1 / 2 | `diar.num_speakers` "auto"/"1"/"2" |
| Swap speakers | False | Advanced > Speech + result button | `Swap Speaker A and B` | `mapping.swap` |
| Plan only | False | main (03) secondary button | `Preview cut plan` | `plan_only=True` for that click only; main button always `False` |
| Q->A max gap | 1.5 | Advanced > Cut style > Fine-tune | `Question -> answer gap (s)` | `planner.qa_gap` |
| Wide hold | 2.0 | same | `Stay wide into answer (s)` | `planner.qa_answer_hold` |
| Rapid turns | 3 | same | `Back-and-forth: speaker changes` | `planner.rapid_turns` |
| Rapid window | 6.0 | same | `Back-and-forth: within (s)` | `planner.rapid_window` |
| Min shot | 1.2 | same | `Shortest shot (s)` | `planner.min_shot` |
| Silence -> wide | 2.5 | same | `Go wide after silence (s)` | `planner.silence_wide` |
| Question heuristic | True | same | `Spot questions without "?"` | `planner.question_heuristic` |
| Offsets A / B / wide | "" | Advanced > Sync (A/B hidden in single) | `Cam A offset (s)` etc., placeholder `auto` | `sync.offset_*` via `_opt_float` |
| Width / Height / FPS | 1920 / 1080 / 30 | Advanced > Output | `Quality`: **1080p · 30 fps** / 720p · 30 fps / 4K · 30 fps / Custom (reveals the 3 numbers) | (1920,1080,30) / (1280,720,30) / (3840,2160,30) / numbers |
| Auto-edit button | - | main (03) | `Edit my podcast` | runs with `plan_only=False` |
| Outputs (video, files, summary, preview, EDL, transcript, log) | - | result | see 3.4 | nothing dropped: all files in Details > All output files |

**Cut style preset** (`Cut style` radio, Advanced): sets the six sliders + checkbox; editing any of them via `.input` switches the radio to `Custom`.
| Preset | qa_gap | qa_hold | rapid_turns | rapid_window | min_shot | silence_wide | question_heuristic |
|---|---|---|---|---|---|---|---|
| Calm (fewer cuts) | 1.5 | 2.5 | 4 | 6.0 | 2.0 | 3.5 | on |
| **Balanced (default = today)** | **1.5** | **2.0** | **3** | **6.0** | **1.2** | **2.5** | **on** |
| Snappy (more cuts) | 1.5 | 1.5 | 3 | 6.0 | 0.8 | 2.0 | on |
| Custom | slider values | | | | | | |

Advanced accordion label `Advanced settings`, collapsed; groups in order: `Speech`, `Cut style`, `Output`, `Sync`, `Single-clip framing` (only in single). Each group a mono meta label + controls. Everything else in `config.py`/CLI (diarization backend, captions font/size, encoder, crf, `--no-qa`, `--no-rapid`, crop_a/b, etc.) remains CLI-only as today.

---

## 5. Implementation notes for C

Files:
- `app.py`: rewrite `build_ui()`; keep `main()` binding/`allowed_paths`; keep `_ListHandler`, `_path`, `_hex`, `_opt_float`.
- New `ui/podbot.css`, `ui/podbot.js` (loaded by `app.py` at launch), new small helpers may live in `app.py` or `ui/render.py` (pure functions: `stage_html`, `timeline_html`, `transcript_html`, `chips_html`, `error_html`, `guess_roles`).
- **Before editing**, copy the untouched `app.py` to `design/app_before.py` (reference for D; not imported by the app). Do not commit.
- Pipeline package: **no changes.** Everything needed (`progress(frac, msg)`, `res.edl`, `res.outputs`, `res.stats`, `res.mapping`, `res.transcript_text`, `find_ffmpeg`, `probe`) already exists.

Structure:
- Split config building out: `build_config(files_roles..., all control values, plan_only) -> PipelineConfig` as a pure function (no Gradio calls), so D can compare it to the old construction. Keep the `work_dir` cache key formula identical (`single-` prefix, basename, size).
- Run handler is a **generator**: start `run_pipeline` in a `threading.Thread`, progress callback puts `(frac, msg)` in a `queue.Queue`; generator loops every 0.4 s yielding `gr.update`s for `pb-progress` (and elapsed time) until the thread ends; then yields result updates or error panel. Attach `_ListHandler` before the thread starts, remove in `finally`. Catch the exception in the thread and hand it back.
- Use `show_progress="hidden"` on the run events (we draw our own). Keep `demo.queue()` default concurrency (1 job at a time).
- Theme: `gr.themes.Base(primary_hue=..., font=[LocalFont("IBM Plex Sans"), "system-ui", "sans-serif"], font_mono=[LocalFont("IBM Plex Mono"), "ui-monospace", "monospace"], radius_size=gr.themes.sizes.radius_sm).set(...)` with `button_primary_background_fill="#CF1830"`, `_dark="#FF5468"`, text etc. In 6.29.1 **`theme`, `css`, `css_paths`, `js`, `head` are `launch()` parameters**, not `Blocks()` ones. Pass `css_paths=[ui/podbot.css]`, `js=<contents of ui/podbot.js>`.
- Click-to-seek: `gr.HTML` does not execute inline `<script>` on update. Use one delegated listener in `podbot.js`: on `click`/`keydown Enter|Space` for `.pb-seg, .pb-line`, find `#pb-video video` and set `currentTime = +dataset.t; play()`.
- Visibility: use `gr.update(visible=...)` / `gr.Column(visible=...)`; role dropdowns via `gr.update(choices=..., value=...)`.
- Preset radio <-> sliders: preset `.change` sets sliders; sliders use **`.input`** (user-only) to set radio `Custom`, otherwise the preset's own updates flip it to Custom.
- Swap re-run: `pb-swap-rerun.click(lambda s: not s, swap, swap).then(run_handler, same inputs, same outputs)`.
- Gradio's uploaded temp paths keep the original basename (use `os.path.basename` for display and role guessing).
- CSS: target only our `elem_id`/`elem_classes` plus documented theme variables; avoid Svelte hash classes. Hide the Gradio footer via `footer_links=[]` if supported, else leave it.
- Escape every user/pipeline string inserted into HTML (`html.escape`).

Pitfalls: 2 files is not a mode; master audio is not counted in the dropzone; ColorPicker may return `rgba(...)` (keep `_hex`); `res.output_video` is `None` in plan-only; `stats["reframe"]` only in single; EDL `start/end` are seconds floats, camera values are `CAM_A|CAM_B|WIDE`; the `ui_runs` folder must stay in `allowed_paths` for downloads.

---

## 6. Acceptance criteria (D's checklist)

Visual
1. Launching `.venv/bin/python app.py` serves on 127.0.0.1:7860; page title `PodBot`; no emoji in UI text.
2. Light mode uses the 2.1 light tokens (bg `#FFFFFF`, primary button `#CF1830` with white text); dark system mode uses the dark tokens (button `#FF5468` with `#0F0E10` text).
3. Computed `font-family` of body starts with `IBM Plex Sans`, meta labels with `IBM Plex Mono`; **zero requests to fonts.googleapis.com / gstatic** (network panel), and the page still renders with fonts when offline.
4. Radii: buttons/inputs/cards 2 px; chips fully rounded. No box shadows except the result video card.
5. Idle page at 1440x900 shows header, h1, 01 dropzone, 02 caption colours, 03 Advanced (collapsed) + both buttons, with no other visible controls.

Functional
6. Dropping the 3 multicam demo files shows `DETECTED · 3 CAMERAS` and the roles pre-filled A=cam_a.mp4, B=cam_b.mp4, Wide=wide.mp4.
7. Dropping `single_wide/wide.mp4` shows `DETECTED · 1 WIDE CLIP...`, a thumbnail, and the `Who is Speaker A?` tiles defaulting to Left person; framing sliders appear in Advanced only in this mode.
8. 2 files -> inline error naming the count, both buttons disabled; duplicate role selection -> `Each role needs a different file.` and disabled.
9. Running an edit shows the 5-stage progress with advancing percentage and elapsed time; no default Gradio progress overlay.
10. Multicam demo edit from the UI completes: video plays, timeline renders with A/B/W segments matching EDL row count, 4 download buttons download non-empty files, Transcript tab shows coloured lines.
11. Single-clip demo edit from the UI completes, including the Framing crop preview.
12. Clicking a timeline segment or transcript line seeks the video to that row's start (+-0.5 s).
13. `Preview cut plan` finishes without a video, shows timeline/transcript/details and `Render: Skipped`.
14. Cut style preset buttons set the slider values exactly per section 4; touching a slider switches preset to Custom; Quality Custom reveals width/height/fps.
15. `Swap A and B, re-run` appears only when `res.mapping["confident"]` is false; clicking it flips the Advanced `Swap Speaker A and B` checkbox and re-runs. Ticking that checkbox manually on the multicam demo gives an EDL whose speaker shots have CAM_A and CAM_B exchanged vs the default run.
16. Forcing a pipeline error (e.g. upload a 1-second non-podcast/corrupt video, or a text file renamed .mp4) shows the `pb-error` panel with `role="alert"` and Technical details; the app stays usable for another run.
17. Every option of today's UI is settable in the new UI or listed in section 4 as CLI-only; all output files of a run are downloadable from Details.

Defaults / regression
18. `build_config(...)` with untouched controls equals the `PipelineConfig` built by `design/app_before.py:run_edit` with untouched controls (capture by monkeypatching its `run_pipeline`), for both multicam (3 demo files) and single (wide demo), comparing `to_dict()` excluding `out`; `work_dir` must be equal.
19. Plan-only EDL CSV (`*_edit_decision_list.csv`) from the new UI's default multicam run is byte-identical in rows to one from `app_before.py` defaults (same inputs), and likewise for single mode.
20. `.venv/bin/python -m pytest -q` : all 146 existing tests pass (new tests, if any, also pass).
21. `git status` shows no changes under `podcast_autoeditor/`; nothing committed or pushed; `/Users/yuvraj/AI_Electronics /podcast_autoeditor` untouched.

Accessibility
22. All controls reachable and operable by keyboard in visual order (dropzone, role dropdowns, tiles, colours, Advanced, buttons, timeline segments, transcript lines, downloads); focus ring visible (2 px brand).
23. Text contrast meets the 2.1 table in both themes (spot-check with devtools); speaker identity in timeline/transcript is never colour-only (A/B/W letter or text label present).
24. `pb-progress` has `aria-live="polite"`; error panel `role="alert"`; timeline segments have `aria-label`s with camera and times.
25. With `prefers-reduced-motion: reduce`, no stripe animation or transitions.

Mobile (375x812)
26. No horizontal page scroll in idle, running and result states; gutters 16 px; buttons full width and >= 44 px tall; timeline scrolls inside its own box; download buttons 2 per row.

---

## 7. Out of scope / stretch
- Cancel button for a running job (needs a pipeline hook).
- Custom FastAPI/HTML frontend.
- Manual light/dark toggle, ZIP download, swatch presets, drag-to-reorder roles, sticky phone action bar.
- "Snappier cuts"/"Plan only" re-run chips; editing the cut list in the UI.
- Face-overlay click-to-pick on the thumbnail; crop preview before running.
- Upload size limits/warnings for phone uploads.
