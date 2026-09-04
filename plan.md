# Plan — 110_lockon

date: 2026-09-04
status: approved — expands project.md v1 §5 (LOCKED by owner); introduces no new decisions
owner: matus
tier: demo-grade

Master context = `project.md`. This file cites it, never restates it (doctrine 12). Prior art =
`reports/c0_findings.md`. Readings of project.md this plan had to fix → §9.

## 0. Prior art (C0) — `reports/c0_findings.md`
Adopt: `mujoco` 3.12 (Apache-2.0) · `gymnasium` 1.3 + `stable-baselines3` 2.9 (MIT, torch ≥2.8
cu126) · ByteTrack from Roboflow `trackers` (Apache-2.0; `supervision`'s ByteTrack is deprecated,
`boxmot` is AGPL → rejected) · `gradio` 6.x (Apache-2.0) · `av` (BSD-3, LGPL ffmpeg build) for
mp4, `imageio`+Pillow for GIF. Fallback tracker: vendor kadirnar's MIT ByteTrack source.
Humanoid: hand-authored capsule MJCF (no `dm_control` dep). Null result: no Apache/MIT
camera-drone pursuit-evasion MuJoCo env exists → env is built.

## 1. Goal
A public, capability-neutral repo where a PPO drone-camera (or, on fallback, the scripted hunter)
keeps lock on a scripted evader in MuJoCo under occlusion, darkness and sensor loss; three
pre-rendered shots + degradation curves (lock retention %, mean ± std, ≥20 eps/point) + Gradio
scrubber, each reproducible from one command. Local demo ships at every rung (project.md §2).

## 2. Non-goals
No sim2real · no flight physics · no learned detector · no pixels in RL · no live render in the
viewer · no external datasets · no AGPL deps · no cosmetic mesh work pre-ship · no learned prey
before step 7 · no README wording naming weapons/military (scope.md §2) · no WSL (§9 D2).

## 3. Acceptance gates — machine-checkable
Every gate = one command, exit 0/1, run through `uv run python scripts/check_gates.py G<n>`,
which records verdicts in `reports/gates.json`. Retry law in §5. Fallbacks = project.md §4.

| # | gate (project.md §4) | command | passing |
|---|---|---|---|
| G0 | core: mypy strict, deps ⊆ {numpy}; boundary guard | `mypy --strict src/lockon/core` + `pytest tests/test_boundaries.py` | no |
| G1 | env: scripted episode headless, ≥200 steps/s state-only | `python -m lockon.harness.bench_env --steps 2000` | no |
| G2 | sensor: 3-channel side-by-side GIF; add-channel touches only sensor | `python -m lockon.harness.render --scene sensor3 --gif reports/gifs/sensor_3ch.gif` + `pytest tests/test_sensor.py` | no |
| G3 | track: ID held through occlusion (GIF1) + through lights-cut / 1-channel dropout (GIF2) | `pytest tests/test_track.py` + both GIFs exist | no |
| G4 | scripted prey+hunter: chase clip renders; hunter > static on retention, mid difficulty, n=20 | `python -m lockon.harness.eval --policy scripted --vs static --n 20` | no |
| G5 | PPO > static on retention, one ≤5 h chained run, n=20 | `python -m lockon.harness.eval --policy runs/ppo_local/best.zip --vs static --n 20` | no |
| G6 | harness: full curve set (5 axes × 3 policies × ≥20 eps) + failure report, one command | `python -m lockon.harness.sweep --config configs/sweep_local.yaml` | no |
| G7 | demo: three shots from one command; viewer self-check | `python -m lockon.demo.render_all` + `python -m lockon.demo.viewer --check` | no |
| G8 | license/visual guard (P1 pattern): anchored .gitignore, per-file visual allowlist, all src tracked | `pytest tests/test_license_guard.py` | no |
| G9 | unit suite | `pytest -q` | no |

Retention (single source `lockon.core.metrics`): steps where the tracker outputs a track carrying
the target's first-assigned ID with IoU ≥ 0.5 to the GT box ÷ episode steps. Time-to-reacquire:
steps from lock loss to the next correct-ID step, mean over loss events. "Beats" (G4/G5): mean
retention difference > 0 over n=20 episodes, identical seeds, mid difficulty (§9 D5). Threshold
is 0 by project.md wording; std is reported beside it and never used to soften a verdict.

## 4. Not machine-testable
| item | who | when |
|---|---|---|
| shot legibility (<10 s non-expert read) | owner | D3 / D4 |
| README framing neutrality (scope.md §2) | lead cold-read, then owner | step 10 / D4 |
| "thermal proxy" honesty wording | owner | D4 |

## 5. Task graph (project.md §5, expanded)
The chain is sequential by construction (each package's gate feeds the next); parallelism is
subagents inside a step (cap 3 text, 1 GPU). Tiers: lead=opus (specs, gate authorship, review),
impl=sonnet (code + tests), mech=haiku (sweeps, tables). Every gate → commit.
**Retry law:** gate FAIL → check the instrument first → fix → rerun, max 3 attempts → fallback
(project.md §4) fires without asking → fallback also fails → `reports/blockers.md` row +
`status.txt` BLOCKED(owner), continue on everything independent.

| step | tasks | gate | fallback | est |
|---|---|---|---|---|
| 1 | plan.md · context.md · status.txt · git init · anchored .gitignore · THIRD_PARTY.md · uv venv 3.11 · deps · `scripts/probe_render.py` (rgb+depth+seg 640×480, GLFW) · c0_findings · wslconfig check **dropped** (§9 D2) | probe exits 0 | none on Windows (GLFW is the only backend) → blockers | 0.5 h |
| 2a | core: `SensorFrame, Detection, Track, LockStatus, AgentObs, AgentAction, WorldState, Difficulty` + `metrics.py` + boundary test | G0 | — | 0.5 h |
| 2b | env: arena MJCF builder (30×30 m, walls, pillar occluders by density, lights by level), mocap drone at fixed altitude (vx, vy, yaw-rate, clamped), mocap humanoid capsule figure, LOS via `mj_ray`, 16 horizontal raycasts, GT 3D box → projected 2D box (pure math), `Env.reset/step/state` | G1 | n/a (state path has no renderer) | 1.5 h |
| 2c | sensor: channel registry {rgb, depth, thermal}; thermal = second pass, emissive target material; lights-cut = rgb gain→0 + gaussian noise; per-channel dropout flags; `SensorFrame` carries per-channel GT `Detection` | G2 | rgb+depth only | 1 h |
| 3 | track: noise injector (miss p, jitter σ, occlusion dropout, latency k) → ByteTrack → `Track` + `LockStatus`; scripted scenes `occlusion`, `lights_cut`; GIF1, GIF2 | G3 | — (3 retries → blockers) | 1.5 h |
| 4 | policy: scripted prey (cover-seek, dark-seek, LOS-break, speed; dial 0–1 each) · visibility-greedy scripted hunter · static cam · obs featurizer + reward (`+1 visible − λ‖a‖² − 10 once per loss > K`, λ=0.01, K=20) · harness `episode.py`, `eval`, chase clip · **status → D1** | G4 | prey dial −1 notch, deviation-log row | 1.5 h |
| 5 | harness Gymnasium wrapper (§9 D1) · SB3 PPO MlpPolicy · `VecFrameStack(4)` · 8 SubprocVecEnv · ckpt every 100k steps · resume via `reset_num_timesteps=False` · wall cap 5 h · eval every 200k (n=20 vs static) · curriculum dial 0.3→0.5 at 40 % budget | G5 | scripted hunter = hero, PPO reported honestly | ≤5 h, background from ~h5 |
| 6 | **STOP** — status.txt NEEDS YOU: "approve HPC launch" + exact sbatch plan + GPU-h (§7) | owner | — | — |
| 7 | HPC: git push → `scripts/make_hpc_bundle.py` (100 pattern) · CPU array seeds×rewards×arenas · GPU learned-prey trial ≤20 GPU-h | array done, ≤ cap | local best policy | — |
| 8 | local ∥ 7: curves on local policy (G6) · per-pkg READMEs · viewer (G7 on local shots) · **status → D3** | G6, G7 | mp4-only viewer | 2 h |
| 9 | HPC: select best on held-out seeds · render curves ± std + three shots (EGL→OSMesa) · pull GIFs/PNGs/one ckpt <100 MB | files local | local renders stand | — |
| 10 | swap best policy · final README + H200 line · G8 · one C6 rotation pass · status=READY | G7–G9 | — | 1 h |
| 11–12 | owner: ship / fix / +300 h · approve public · optional 1 h push | owner | — | — |

Steps 1–5 ≈ 8–10 h unattended. Step 5 trains in the background while step 8's non-PPO work
(curves on the scripted hunter, READMEs, viewer) proceeds — nothing there depends on G5.

## 6. Kill criterion / demo-defining fork
The project cannot die by design — every rung ships a fallback. The one demo-defining fork
(project.md §8): **PPO never beats the scripted hunter after 3 × ≤5 h local runs** → ping with the
default recommendation (scripted = hero, PPO reported honestly); continue on the default after
12 h of silence. Counter `ppo_vs_scripted: 0 of 3` lives in `status.txt`; only G5-class design
rejections increment it, instrument findings never do. Local PPO stops once G5 passes — HPC
runs the sweep — so the counter exists for the failure path only.

## 7. HPC plan for the step-6 ping (priced at step 6 from measured local throughput)
Account perun26011488 · confirm allocation before first submit · 100's bundle pattern verbatim
(hash-manifest bundle, offline wheelhouse cp311, `hpc_bootstrap.sh`, smoke job gating the
array, resume-by-result-JSON, per-unit `timeout`, identical local/slurm entrypoint).
- CPU array (no GPU-h): seeds 5 × reward variants 3 (λ, K) × arenas 3 (density 0.3/0.5/0.7) =
  45 units × ~2 h CPU each, `--array`, skip-if-result-JSON.
- GPU (gpu_short): learned-prey trial (PPO prey vs frozen best hunter) ≤20 GPU-h hard `timeout`;
  curves + three shots on one GPU node, EGL, ≤2 GPU-h.
- Estimate ≤25 GPU-h of the 70–100 cap; exact numbers land in status.txt at step 6. Measured P1
  lesson: 4060-derived H200 estimates overstate cost ~4×, so the estimate is quoted as a cap.

## 8. Session ritual + stop-and-ping (project.md §7–8, refactored_method §7)
Every ~2 h or before context pressure: commit → update `context.md` (position, decisions, cold
resume) + `status.txt` (project.md §9 format) → compaction → reload `context.md` + `plan.md`.
Stop only for: approval list (public / money / delete / force-push / wipe) · a fallback that also
failed · the §6 fork · step 6. Everything else: decide, log in `context.md`, continue. Blockers
→ `reports/blockers.md` with a recommendation; work continues on independent tasks.
Compute rule: state-PPO is **CPU by design** (project.md §6: "CPU-h for all state-PPO sweeps";
SB3 MlpPolicy is CPU-optimal). The §5 CPU-fallback STOP applies only to render / torch-GPU work.

## 9. Readings of project.md this plan fixes (logged, not new scope)
- **D1 package boundary.** "Packages import core, never each other" is enforced for the four
  library packages (env, sensor, track, policy → core + third-party only). `harness` is the single
  composition root (episode runner, Gymnasium wrapper, train/eval/sweep/render CLIs) and may
  import any package; `demo` imports harness + core. Encoded in `tests/test_boundaries.py`.
  Without a composition root the rule is unsatisfiable — something must step env → sensor → track.
- **D2 no WSL.** project.md header/§5/§7 mention WSL2 + .wslconfig; execution law §6 and
  ws/CLAUDE.md say native Windows. Native wins; wslconfig check dropped; Windows render backend
  = GLFW (WGL) only; the EGL → OSMesa order applies on HPC. `reports/deviation-log.md` row 1.
- **D3 GT boxes are state math.** The 2D GT box is the projection of the person's 3D bounding
  box through camera intrinsics, computed from state without rendering, so track and harness
  run at state speed; rendering exists only for clips (project.md §2).
- **D4 layout.** One distribution `lockon`, subpackages `src/lockon/{core,env,sensor,track,
  policy,harness,demo}`, one `pyproject.toml`. Seven pyprojects would be doctrine-13 debt.
- **D5 mid difficulty = all dials 0.5** for every "beats" comparison; the sweep varies one axis
  at a time with the others held at 0.5.
