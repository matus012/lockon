# context.md — 110_lockon (CLI-maintained; sufficient to resume cold)

## Read order on resume
1. `status.txt` (owner-facing one-liner, project.md §9) → 2. this file → 3. `plan.md` (§5 task
graph, §3 gates) → 4. `project.md` only when a decision is questioned (it is LOCKED).
Execution law: `../000_infra/refactored_method.md`. Doctrines: `../000_infra/doctrines.md`.

## Position
- **Step 7 of 12 — HPC running** (2026-09-05, owner approved the launch).
- PERUN facts (all verified this session): SSH `login02.perun.tuke.sk` (user in ~/.ssh/config, key in
  ~/.ssh); repo `/mnt/project/perun26011488/lockon/repo` (runs/ stay there — data gravity);
  venvs `.venv` (CPU torch) + `.venv_gpu` (cu126) built ONLINE on the login node
  (`scripts/hpc/hpc_bootstrap_online.sh`, row 15); partitions cpu_short / gpu_short (row 16);
  **slurmdbd is down — `sacct` never works; detect job completion by absence from `squeue`**;
  submit-time cap = 4 jobs per user (array tasks count), so submit GPU work while the array is
  throttled, then raise `scontrol update JobId=<arr> ArrayTaskThrottle=8`.
- Live jobs: **82268** CPU array (45 units, throttle 8, ~1.7 h/unit → ~10 h; results
  `runs/hpc/<unit>/result.json`) · **82273** learned-prey trial on an H200 (`device=cuda`
  verified; 18 h training timeout, then its own eval table). Probe 82271 passed (CUDA + EGL
  render + 3.4k steps/s). Local `runs/ppo_local/best.zip` uploaded so both tables can use it.
- GPU-h: probe ≈ 0.05 spent; prey ≤ 19.5; step-9 renders ≤ 2 → ≤ 22 of the owner's 70 cap.
- Step 9 is scripted end-to-end: `bash scripts/hpc/run_step9.sh` (waits for 45/45, pulls result
  JSONs, runs `scripts/hpc/aggregate.py` → `reports/hpc/best_unit.txt`, pulls that best.zip,
  submits `step9.sbatch` with BEST=<unit> for the n=80 fork verdict + curves + shots on a GPU
  node, pulls the small artifacts back). Step 10 = swap the hero if the best unit beats
  scripted, README H200 line, status READY.
- Local n=80 verdict on the held-out seeds running in parallel (`reports/eval_n80_*.json`) to
  replace the n=20 README numbers.

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
