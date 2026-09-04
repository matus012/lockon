# context.md — 110_lockon (CLI-maintained; sufficient to resume cold)

## Read order on resume
1. `status.txt` (owner-facing one-liner, project.md §9) → 2. this file → 3. `plan.md` (§5 task
graph, §3 gates) → 4. `project.md` only when a decision is questioned (it is LOCKED).
Execution law: `../000_infra/refactored_method.md`. Doctrines: `../000_infra/doctrines.md`.

## Position
- **Step 4 of 12 — G4 running** (2026-09-04). Commits: 8cbe538 step 1 · 2171ecf packages ·
  f455aa9 review pass 1 · harness phase A (episode/scenes/gym/eval). Gates PASS: G0 G1 G8.
  G4 preview n=5: scripted 0.628 vs static 0.447. Full G4 (n=20) running →
  `runs/g4_eval.log`, `reports/eval_scripted_vs_static.json`.
- In flight: phase B1 render.py (GIFs G2/G3, chase clip, mp4 shots) · phase B2 train.py +
  sweep.py (G5/G6). After B2 lands: launch the ≤5 h PPO run (`runs/ppo_local`), then G6 sweep
  on scripted/static while it trains (plan §5 step 8 non-PPO work).
- Fresh-context lead review (opus) of env/sensor/track/policy done; all findings fixed or
  logged (deviation-log rows 2–6). Known honest property: ByteTrack coast across a 2 s gap
  under jitter 4.4 px keeps the id in 16/20 seeds (flat in box speed and state model) — the
  hunter's job is to shorten gaps; this goes in the README "where it breaks".
- G4 verdict annotation owed (review finding 6): the prey's LOS-break fires on
  `person_visible`, so it is a harder examiner against a mover than against the static camera —
  bias is conservative (works against scripted/PPO), a pass is safe.
- Session clock: ~2.5 h in. Ritual (plan §8) every ~2 h: commit → update this file +
  status.txt → compaction.

## Architecture (plan §9 D1, D4)
```
src/lockon/
  core/      schemas only: SensorFrame Detection Track LockStatus AgentObs AgentAction
             WorldState Difficulty · metrics.py (retention, time-to-reacquire) · deps ⊆ numpy
  env/       MuJoCo arena builder (MJCF from Difficulty), mocap drone + humanoid figure,
             LOS + raycasts (mj_ray), GT box projection, Env.reset/step/state
  sensor/    channel registry {rgb, depth, thermal}; renders from env.model/env.data;
             lights-cut, dropout flags; emits SensorFrame with per-channel GT Detection
  track/     noise injector → ByteTrack (trackers, Apache-2.0) → Track + LockStatus
  policy/    scripted prey · scripted hunter · static cam · obs featurizer · reward · PPO config
  harness/   COMPOSITION ROOT: episode runner, Gymnasium wrapper, train/eval/sweep/render/bench
  demo/      render_all (three shots), viewer (Gradio, mp4-only fallback), root README assets
tests/       test_boundaries (import rule) · test_license_guard (P1 pattern) · per-package tests
scripts/     check_gates.py · probe_render.py · make_hpc_bundle.py (step 7, from ws/100)
reports/     c0_findings.md · blockers.md · deviation-log.md · gates.json · gifs/ · curves/
configs/     sweep_local.yaml · ppo_local.yaml
runs/        (gitignored) checkpoints, eval JSONs
```
Import rule (test-enforced): env/sensor/track/policy import only `lockon.core` + third-party;
harness imports anything; demo imports harness + core; core imports numpy only.

## Decisions (dated; readings of project.md, never re-decisions)
- D1–D5 2026-09-04: see plan.md §9 (package boundary + composition root, no WSL, GT boxes from
  state, single distribution, mid difficulty = 0.5).
- D6 2026-09-04: tracker library = Roboflow `trackers` (Apache-2.0). `supervision` ByteTrack is
  deprecated (removal at 0.31), `boxmot` AGPL. Fallback = vendor kadirnar MIT source.
- D7 2026-09-04: mp4 via PyAV (`av`, BSD-3, LGPL ffmpeg build); GIF via imageio + Pillow.
  No `imageio-ffmpeg` (unstated ffmpeg build flags).
- D8 2026-09-04: humanoid = hand-authored capsule MJCF in env; no dm_control dependency.
- D9 2026-09-04: state-PPO trains on CPU by design (plan §8). GPU = rendering only.
- D10 2026-09-04: channel physics is data in core (`CHANNEL_SPECS`, `channel_sees`); env iterates
  `CHANNELS` and never names a channel; sensor owns renderers. Adding a channel = one core entry
  + one sensor renderer (gate G2 wording "touches only sensor" read as sensor + registry line).
- D11 2026-09-04: eval-time perception frozen at `NoiseConfig.from_dial(0.3)` + LockTracker
  defaults (harness constant `EVAL_NOISE`); every owner-facing retention number uses it.
- D12 2026-09-04: arena half-size 12 m, drone altitude 3 m, pillars 3.5 m tall (LOS is
  horizontal geometry: the drone must go around, not over); 10 Hz, 200-step episodes.

## Environment
Native Win11, PowerShell; uv venv `.venv` Python 3.11; torch cu126; RTX 4060 8 GB. Render:
GLFW/WGL (MUJOCO_GL unset). HPC: PERUN, account perun26011488, gpu_short, bundle pattern from
`../100_occlusion_mot/scripts/{make_hpc_bundle.py,hpc_bootstrap.sh,hpc_env.sh,sweep_common.py}`.

## How to resume cold
```
cd C:\Users\matus\A_MAIN\ws\110_lockon
type status.txt ; git log --oneline -5 ; uv run python scripts/check_gates.py --all
```
Then open plan.md §5 at the step named in status.txt and continue from the first unmet gate.

## Open items / blockers
None. `reports/blockers.md` is empty. Deviation log has 6 rows (all gate-semantics **no**).
