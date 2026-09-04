# context.md — 110_lockon (CLI-maintained; sufficient to resume cold)

## Read order on resume
1. `status.txt` (owner-facing one-liner, project.md §9) → 2. this file → 3. `plan.md` (§5 task
graph, §3 gates) → 4. `project.md` only when a decision is questioned (it is LOCKED).
Execution law: `../000_infra/refactored_method.md`. Doctrines: `../000_infra/doctrines.md`.

## Position
- **Step 5 of 12 — PPO run 5 in flight** (2026-09-04, ~8 h in). Gates PASS at HEAD: G0 G1 G2 G3
  G4 (per-episode noise, seeds 1000–1019: scripted 56.2 % vs static 46.0 %) G8. G5 passed on
  run 4 (54.5 vs 44.2) BEFORE deviation rows 11–12; it is re-run on run 5's best.zip when the
  run ends (`check_gates.py G5`), then PPO vs scripted (`eval --policy runs/ppo_local/best.zip
  --vs scripted --n 20 --seed-base 1000 --json reports/eval_ppo_vs_scripted.json`).
- Then, in order (all one-command): `sweep --config configs/sweep_local.yaml --workers 6` (G6,
  ~20 min, memory-bound — never run with the render or training), `demo.render_all --policy
  scripted` + the three GIF renders (commands in runs/render_final.log), `check_gates.py G6 G7 G8
  G9`, `scripts/readme_table.py` → paste into README (PPO row + n + CI), C6 pass 2 on the
  package READMEs (mech), status.txt = NEEDS YOU "approve HPC launch" with
  reports/hpc_launch_plan.md, session report + strejc-nav.
- PPO history: runs 1–3 instrument-defective (fixed layouts; visibility reward; std-1
  exploration); run 4 = fork count 1 (tie with scripted, row 10); run 5 = pre-registered
  observation change (row 12, D15) — if it does not beat scripted it stays a tie, hero = scripted.
- Two fresh-context lead reviews done (rows 2–6, 7–12). Known honest properties for the README:
  ByteTrack coast 16/20 across a 2 s gap; darkness alone never breaks lock (thermal proxy);
  1-step detection latency handicaps movers; prey LOS-break bias +0.065 conservative
  (reports/los_break_bias.json); n = 20 local gates are direction-only (CI ±0.17).
- Ritual (plan §8) every ~2 h: commit → this file + status.txt → compaction.

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
- D15 2026-09-04: the policy observes the tracker's lock as its "seen" signal (rows 11–12); eval
  noise seeded per episode; loss causes from explicit in_fov/unoccluded flags; occlusion shot
  pinned to the certified scripted hunter; frame stack depth read from the model.
- D14 2026-09-04: drone commands pass a first-order filter (tau 0.3 s) — kinematic, no flight
  physics; PPO exploration std starts at 0.2 with no entropy bonus (deviation row 9).
  Rejected after measurement: yaw-compensated detections (flat), higher exploration std.
- D13 2026-09-04: reward = tracker lock held, not state visibility (deviation row 8); the frozen
  tracker runs inside LockonGym at state speed with the eval noise dial and a per-episode seed.
  Also fixed: LockonGym re-samples layout + prey seed every episode (SB3 resets with seed=None).
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
