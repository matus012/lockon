# lockon.demo

Owns the three pre-rendered shots and the Gradio viewer — the shippable artifact (project.md §1:
"demo is the product"). Computes nothing: every number and frame comes from `lockon.harness`.
Planned modules per `SPEC.md`: `render_all.py` (writes `occlusion.mp4`, `lights_cut.mp4`,
`chase.mp4` + `demo/shots/manifest.json` via `lockon.harness.render`) and `viewer.py` (a Gradio
Blocks app: a tab per shot, a Curves tab off `reports/curves/summary.png` +
`reports/sweep/results.json`, a "Where it breaks" tab off `reports/sweep/failure_report.md`;
`--check` builds the Blocks object and verifies every referenced file exists without launching a
server — an mp4-only static-HTML fallback applies if `gradio` fails to import). Root README
(drone safety / search & rescue / filming / wildlife framing, "thermal proxy" wording, lock
retention % throughout) is written at step 10, not owned here.

**Current state (docs-from-code, 2026-09-04): only `SPEC.md` and `__init__.py` exist.**
`render_all.py`, `viewer.py`, and `tests/test_demo.py` are not yet implemented (context.md:
"in flight — demo package (render_all + viewer, impl agent)"). Everything below the import rule
is the SPEC's planned contract, not verified code — do not cite it as shipped.

Import rule (`tests/test_boundaries.py`): `demo` may import `core` and `harness` only (`ALLOWED["demo"] = {"core", "harness", "demo"}`); `test_all_packages_exist` already requires `src/lockon/demo/__init__.py`, which exists.

## Public API (planned, per SPEC.md — not yet in the code)
- `render_all.main(argv=None) -> int` — CLI flags `--policy {best,scripted}`, `--out demo/shots`.
- `viewer.main(argv=None) -> int` — CLI flags `--check`, `--port 7860`.

## Invariants (planned, per SPEC.md — `tests/test_demo.py` does not exist yet)
- Three real mp4s (> 10 KB) + a 3-key manifest from `render_all --policy scripted --steps 12` on a tmp dir (marked `render`).
- `viewer --check` exits 0 when all three shots exist, 1 when one is missing.

## Gate (plan.md §3, G7) — not yet passing, no `demo/` CLI to run
```
uv run python -m lockon.demo.render_all
uv run python -m lockon.demo.viewer --check
```
