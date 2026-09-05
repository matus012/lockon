# context.md — 110_lockon (CLI-maintained; sufficient to resume cold)

## Read order on resume
1. `status.txt` (owner-facing one-liner, project.md §9) → 2. this file → 3. `plan.md` (§5 task
graph, §3 gates) → 4. `project.md` only when a decision is questioned (it is LOCKED).
Execution law: `../000_infra/refactored_method.md`. Doctrines: `../000_infra/doctrines.md`.

## Position
- **Step 7 of 12 — HPC running** (2026-09-05). Owner approved the launch. PERUN facts: SSH
  `login02.perun.tuke.sk` (user mafike202, key in ~/.ssh); repo at
  `/mnt/project/perun26011488/lockon/repo` (data gravity: runs/ stay there); venvs `.venv`
  (CPU torch) and `.venv_gpu` (cu126) built ONLINE on the login node (`scripts/hpc/
  hpc_bootstrap_online.sh`, deviation row 15); partitions cpu_short / gpu_short (row 16).
  Site cap: 4 queued jobs per user (array tasks count) — the array is throttled to 2
  (`scontrol update JobId=82248 ArrayTaskThrottle=2`) so GPU jobs can be submitted.
- Jobs: smoke 82247 COMPLETED (SMOKE OK, result moved to `runs/hpc/_smoke_base_d0.3_s0`);
  CPU array 82248 (45 units, `runs/hpc/<unit>/result.json`; index 0 skipped because the
  smoke shadowed its path — **resubmit `sbatch --array=0 runs/hpc/submit.sbatch` when the
  queue has ≤ 3 of my jobs**; fixed for the future via `--results-root runs/hpc_smoke`).
  GPU: `scripts/hpc/launch_gpu_trial.sh` (dev-box driver, log `runs/hpc_gpu_launch.log`)
  syncs code, waits for a slot, runs `gpu_probe.sbatch` (≤0.25 GPU-h), then submits
  `prey_trial.sbatch` (19.5 h wall, 18 h training timeout, 1 GPU, eval table at the end).
  Monitor `hpc:` events every 5 min in this session.
- Budget: GPU-h spent 0 so far; planned ≤ 0.25 probe + ≤ 19.5 prey + ≤ 2 renders ≤ 22 of the
  70 cap the owner set. CPU-h ≈ 45.
- Next: step 8 nothing owed locally; step 9 when the array finishes → pull result JSONs
  (HPC_RUNBOOK step 8), pick the best unit by retention vs static on its own eval, pull its
  best.zip, fork verdict on seeds 1000–1079 (n=80) vs scripted; render curves ± std + three
  shots on a GPU node (EGL) → pull PNG/GIF; step 10 README H200 line, status READY.
- Learned-prey trial code landed (commit eca3c39): `lockon.policy.prey_learned`,
  `lockon.harness.{prey_gym,train_prey}`, `eval --prey`, `configs/prey_gpu.yaml`.

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
