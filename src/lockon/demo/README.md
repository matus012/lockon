# lockon.demo

Owns the three pre-rendered shots and the Gradio viewer — the shippable artifact (project.md §1:
"demo is the product"). Computes nothing: every number and frame comes from `lockon.harness`.
`render_all.py` writes `occlusion.mp4`, `lights_cut.mp4`, `chase.mp4` + `demo/shots/manifest.json`
(scene, seed, policy, window, retention in the window, git HEAD) + a static `demo/index.html`
(the mp4-only Gradio fallback) via `lockon.harness.render.render_frames`. `viewer.py` is a Gradio
Blocks app: a tab per shot, a Curves tab off `reports/curves/summary.png` +
`reports/sweep/results.json`, a "Where it breaks" tab off `reports/sweep/failure_report.md`;
`--check` builds the Blocks object (or falls back if `gradio` fails to import) and verifies every
referenced file exists without launching a server. Root README (drone safety / search & rescue /
filming / wildlife framing, "thermal proxy" wording, lock retention % throughout) is written at
step 10, not owned here.

Import rule (`tests/test_boundaries.py`): `demo` may import `core` and `harness` only
(`ALLOWED["demo"] = {"core", "harness", "demo"}`).

## Public API
- `render_all.render_all(policy, out, steps=None, fps=10) -> dict[str, dict]` — `main` CLI flags
  `--policy {best,scripted}` (best = `runs/ppo_local/best.zip` if present, else scripted),
  `--out demo/shots`, `--steps N` (shrinks every shot to a plain N-step run, for smoke tests),
  `--fps`. Only the `chase` shot follows `--policy`; `occlusion` and `lights_cut` always run their
  own scene's certified hunter (`ScriptedHunter` / `StaticCamera`) regardless of the flag — the
  occlusion showcase seed (track README, deviation row 7) is only certified under `ScriptedHunter`.
- `render_all.main(argv=None) -> int` — also exits 1 if any shot mp4 is missing or < 10 KB.
- `viewer.build_blocks() -> gr.Blocks`; `viewer.check() -> int`; `viewer.main(argv=None) -> int` —
  CLI flags `--check`, `--port 7860`.

## Invariants (tests/test_demo.py)
- Three real mp4s (> 10 KB) + a 3-key manifest from `render_all --policy scripted --steps N` on a
  tmp dir (marked `render`): `test_render_all_scripted_produces_three_shots`.
- `viewer --check` exits 0 when all three shots + `index.html` exist, 1 when one is missing:
  `test_viewer_check`.

## Gate (plan.md §3, G7)
```
uv run python -m lockon.demo.render_all
uv run python -m lockon.demo.viewer --check
```
