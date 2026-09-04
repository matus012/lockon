# lockon.demo — specification (lead-authored 2026-09-04; plan.md steps 8/10, gate G7)

Import rule: `lockon.core`, `lockon.harness` (plan §9 D1). Owns nothing that computes a number.

## render_all.py — `python -m lockon.demo.render_all [--policy best|scripted] [--out demo/shots]`
The three shots (110_plain.md), each a call into `lockon.harness.render` with a pinned window:
1. `occlusion.mp4` — scene `occlusion`, window around the gap (from `SCENES` metadata), hunter =
   the hero policy (`runs/ppo_local/best.zip` if present and `--policy best`, else scripted).
2. `lights_cut.mp4` — scene `lights_cut`, window t 45..110, static camera (the point is the
   channel hand-over, not movement).
3. `chase.mp4` — scene `chase`, full 200 steps, hero policy vs prey at mid difficulty.
Also writes `demo/shots/manifest.json`: shot → scene, seed, policy, window, retention in the
window, git HEAD. Idempotent; exit 1 if any mp4 is missing or < 10 KB afterwards.

## viewer.py — `python -m lockon.demo.viewer [--check] [--port 7860]`
Gradio Blocks: a tab per shot with `gr.Video` (local path, scrubber) and a one-line caption
(what to watch for, retention in the window from the manifest); a "Curves" tab showing
`reports/curves/summary.png` and the retention table from `reports/sweep/results.json` (mean ±
std at mid difficulty for static / scripted / ppo); a "Where it breaks" tab rendering
`reports/sweep/failure_report.md`. `--check`: build the Blocks object without launching, verify
every referenced file exists, print the manifest, exit 0/1 — that is gate G7's second command.
Fallback (project.md §4): if gradio import fails, `--check` still passes when the mp4s exist and
`demo/index.html` (static `<video controls>` page written by render_all) exists.

## README (root) — written at step 10, not here. Wording rules: project.md §1 framing, "thermal
proxy", retention % everywhere, honest "where it breaks" paragraph (ByteTrack coast ≈ 80 % id
hold across 2 s gaps at noise 0.3; darkness alone never breaks lock because thermal proxy is
dark-immune — say so).

## Tests (`tests/test_demo.py`)
1. `render_all` with `--policy scripted` on a tmp out dir and `--steps 12` override produces
   three mp4 files > 10 KB and a manifest with the three keys (mark `render`).
2. `viewer --check` exits 0 against that tmp dir (monkeypatch paths) and 1 when a shot is missing.
