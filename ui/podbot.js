() => {
  // PodBot page script (passed to Blocks.launch(js=...); Gradio runs it once on page load).
  // gr.HTML does not run inline <script>, so every widget behaviour is a delegated listener here.
  if (window.__podbot) return;
  window.__podbot = true;

  const player = () => document.querySelector("#pb-video video");

  // Click-to-seek: timeline segments (.pb-seg) and transcript lines (.pb-line) are <button>s
  // carrying data-t (seconds).  Native buttons turn Enter/Space into "click", so one handler
  // covers mouse and keyboard.
  document.addEventListener("click", (e) => {
    const el = e.target.closest && e.target.closest(".pb-seg[data-t], .pb-line[data-t]");
    if (!el) return;
    const v = player();
    const t = parseFloat(el.dataset.t);
    if (!v || !Number.isFinite(t)) return;
    v.currentTime = t;
    const p = v.play();
    if (p && p.catch) p.catch(() => {});
  });

  // Bring the run's status into view when it appears (on a phone it is below the buttons).
  // Gradio 6 adds/removes hidden components from the DOM, so watch for the panels' arrival.
  const calm = window.matchMedia("(prefers-reduced-motion: reduce)");
  const present = {};
  new MutationObserver(() => {
    for (const [id, block] of [["pb-progress", "nearest"], ["pb-error", "nearest"], ["pb-result", "start"]]) {
      const el = document.getElementById(id);
      if (el && !present[id]) {
        // the result scrolls so the finished progress line above it stays visible
        const target = id === "pb-result" ? (document.getElementById("pb-progress") || el) : el;
        // wait for Gradio to fill the panel, or it is scrolled to while still 0 px tall
        setTimeout(() => target.scrollIntoView({ block, behavior: calm.matches ? "auto" : "smooth" }), 250);
      }
      present[id] = !!el;
    }
  }).observe(document.documentElement, { childList: true, subtree: true });

  // Playhead on the cut timeline while the edited video plays.
  // (timeupdate does not bubble, so listen in the capture phase.)
  document.addEventListener("timeupdate", (e) => {
    const v = e.target;
    if (!(v instanceof HTMLVideoElement) || !v.closest("#pb-video")) return;
    document.querySelectorAll(".pb-tl-track").forEach((track) => {
      const total = parseFloat(track.dataset.total) || v.duration || 1;
      track.style.setProperty("--pb-ph", String(Math.min(1, Math.max(0, v.currentTime / total))));
      track.classList.add("has-playhead");
    });
  }, true);
}
