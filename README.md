# lockon — active perception under degradation

A simulated drone camera that **moves to keep lock on a person who is actively hiding** — behind
pillars, in the dark, when a sensor dies. Everything runs inside a MuJoCo arena the repo owns: no
real footage, no datasets, no learned detector. The demo is the product.

Use cases this is built for: drone safety, search and rescue, autonomous filming, wildlife
monitoring. Nothing here transfers to real hardware; it is a simulation study.

| Person ducks behind a pillar → the camera repositions → lock survives | Lights cut → the colour channel goes to noise → the thermal proxy takes over |
|---|---|
| ![occlusion](reports/gifs/occlusion_lock.gif) | ![lights cut](reports/gifs/lights_cut_lock.gif) |

Panels: colour · depth · thermal proxy, tracker box + id on every live channel, banner with lock
state and running **lock retention %**, top-down minimap (green line = line of sight).

## The one number: lock retention %

`retention = steps on which the tracker outputs a track carrying the target's FIRST id with
IoU ≥ 0.5 to the true box ÷ episode steps` (`lockon.core.metrics`). An id switch is a loss even
if some other track still overlaps — that is the whole point. Secondary: time-to-reacquire.

Mid difficulty (every dial at 0.5), 20 episodes, identical seeds, tracker noise dial 0.3:

| camera policy | lock retention % | ± std | time-to-reacquire (steps) |
|---|---|---|---|
| static camera (floor) | 37.5 | 25.4 | 1.8 |
| scripted hunter (visibility-greedy) | 56.6 | 35.3 | 2.2 |
| PPO hunter | pending (step 5) | | |

Degradation curves (retention vs occluder density · darkness · prey speed · prey aggressiveness ·
channel dropout, mean ± std over ≥ 20 episodes per point, one axis at a time with the others at
0.5) land in `reports/curves/` from a single command (`python -m lockon.harness.sweep`).

## How it works (7 packages, each standalone)

```
core     typed schemas + the retention metric (numpy only)
env      MuJoCo arena: pillars, point lights, a kinematic drone camera (velocity commands,
         fixed altitude, first-order response), a capsule humanoid; line of sight via mj_ray;
         the true 2D box is projected from state — no render in the loop
sensor   colour / depth / thermal-proxy render passes, channel registry, dead-channel flags
track    noisy-GT boxes (miss, jitter, range dropout, latency) → ByteTrack → lock status
policy   scripted prey (cover-seek, dark-seek, LOS-break, speed dials) · scripted hunter ·
         PPO hunter (Stable-Baselines3, state observations only, CPU)
harness  episode runner, Gymnasium wrapper, train / eval / sweep / render CLIs
demo     three shots + Gradio viewer (static HTML fallback)
```

Design law: **nothing outside our control in the critical path.** The tracker consumes the
simulator's own true boxes plus injected noise, so it can never fail for a reason we cannot fix.
RL trains on state (own pose, last-seen target pose, time since seen, 16 raycasts, light level,
channel-alive flags; 4-frame stack), never on pixels; rendering exists for clips only.
"Thermal" is a **thermal proxy**: a second render pass with an emissive target material,
unaffected by light level.

## Where it breaks (honest list)

- **Coasting through occlusion is a lottery.** ByteTrack's image-space Kalman keeps the id across
  a 2 s gap in 16 of 20 noise seeds at noise dial 0.3, flat in box speed and state model
  (`reports/deviation-log.md` row 6). The hunter's job is to keep gaps short.
- **Darkness alone never breaks lock**: the thermal proxy is dark-immune, so the darkness curve
  is flat unless the thermal channel also drops. That is the design, stated, not a result.
- **Reward ≠ visibility.** A policy rewarded for raw visibility learned to strafe and yaw so hard
  that boxes jumped ~8 px/step and the tracker switched ids 5.6× per episode — visibility up,
  retention down. Training now rewards *tracker lock* (row 8). Exploration noise above
  std ≈ 0.2 pushes the target out of the field of view (row 9).
- The prey's line-of-sight break fires on visibility, so it examines a moving camera harder than
  a static one: the comparisons above are conservative for the hunters.

## Run it

```bash
uv sync                                                # Python 3.11, per-project venv
uv run python scripts/check_gates.py --all             # every gate, verdicts -> reports/gates.json
uv run python -m lockon.harness.eval --policy scripted --vs static --n 20
uv run python -m lockon.harness.sweep --config configs/sweep_local.yaml
uv run python -m lockon.harness.train --config configs/ppo_local.yaml --out runs/ppo_local
uv run python -m lockon.demo.render_all && uv run python -m lockon.demo.viewer
```

Windows 11 + GLFW offscreen rendering is the dev target; Linux uses EGL or OSMesa
(`MUJOCO_GL`). State-only PPO trains on CPU by design (~2.9k env steps/s on a laptop).

## Licences

Apache-2.0. Dependencies are Apache/MIT/BSD only (`THIRD_PARTY.md`); no datasets, no weights.
A guard test (`tests/test_license_guard.py`) classifies every committed visual and rejects
untracked source files and anchored-ignore mistakes.
