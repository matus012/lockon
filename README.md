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

Every number below is regenerated from `reports/eval_*.json` by `scripts/readme_table.py`:

| camera policy | lock retention % | ± std | 95 % CI of mean | Δ vs static (95 % CI) | time-to-reacquire |
|---|---|---|---|---|---|
| static camera (floor) | 44.0 | 32.1 | [36.9, 51.0] | — | 1.66 |
| scripted hunter (visibility-greedy) | 52.0 | 33.8 | [44.6, 59.4] | +8.0 [+2.3, +13.7] | 2.09 |
| PPO hunter (best local checkpoint) | 51.3 | 35.1 | [43.6, 59.0] | +7.3 [+1.1, +13.5] | 1.70 |

n = 80 episodes per arm, seeds 1000–1079 (held out from model selection), mid difficulty (every
dial 0.5), tracker noise dial 0.3, per-episode noise seed. **Moving beats standing still:** both
hunters clear the static floor by about 8 points with confidence intervals that exclude zero.
**Learning does not beat the script:** scripted − PPO is +0.7 points, CI [−5.1, +6.5], so the two
are indistinguishable and neither is better by more than ~6 points. The hero policy is therefore
the **scripted hunter**; the PPO result is reported as it is.

Degradation curves — retention vs occluder density · darkness · prey speed · prey aggressiveness
· channel dropout, mean ± std over 20 episodes per point, one axis at a time with the others at
0.5 — from one command (`python -m lockon.harness.sweep --config configs/sweep_local.yaml`):

![curves](reports/curves/summary.png)

`reports/sweep/failure_report.md` names the worst point and the dominant loss cause per axis. Across
all 1,500 sweep episodes every lock loss is attributed to occlusion or field-of-view, never to
darkness or channel dropout: with a dark-immune thermal proxy and at most one dead channel at a
time, those two axes cannot break lock on their own (a design statement, now also a measurement).

## The evader that learned to hide (PERUN, one H200)

A second PPO agent was trained as the **prey** — privileged observations, rewarded for being
unseen — against the frozen scripted hunter, 20M steps on one H200. Then every camera policy was
scored against both prey on the same held-out seeds:

| hunter | vs scripted prey | vs learned prey | Δ (95 % CI) |
|---|---|---|---|
| static camera | 44.0 ± 32.1 | 27.6 ± 28.1 | −16.4 [−25.0, −7.7] |
| scripted hunter | 52.0 ± 33.8 | 50.8 ± 39.0 | −1.2 [−11.8, +9.5] |
| PPO hunter | 51.3 ± 35.1 | 35.9 ± 31.3 | −15.4 [−25.1, −5.6] |

n = 80 episodes per cell, seeds 1000–1079, mid difficulty. Full table: `reports/prey_trial.md`.

**The hand-written hunter is the only policy the learned evader cannot degrade.** It loses 1.2
points and the interval spans zero; the fixed camera loses 16.4 and the learned hunter 15.4, both
with intervals excluding zero. The evader trained *against* the scripted hunter for 20M steps and
still could not exploit it, while the evasions it found transfer devastatingly to a camera policy
it had never seen. Caveat: the prey's checkpoint was picked on 20 selection episodes; the table
above is the held-out read.

## How it works (7 packages, each standalone)

```
core     typed schemas + the retention metric (numpy only)
env      MuJoCo arena: pillars, point lights, a kinematic drone camera (velocity commands,
         fixed altitude, first-order response), a capsule humanoid; line of sight via mj_ray;
         the true 2D box is projected from state — no render in the loop
sensor   colour / depth / thermal-proxy render passes, channel registry, dead-channel flags
track    noisy-GT boxes (miss, jitter, range dropout, 1-step latency) → ByteTrack → lock status
policy   scripted prey (cover-seek, dark-seek, LOS-break, speed dials) · scripted hunter ·
         PPO hunter (Stable-Baselines3, state observations only, CPU)
harness  episode runner, Gymnasium wrapper, train / eval / sweep / render CLIs
demo     three shots + Gradio viewer (static HTML fallback)
```

Design law: **nothing outside our control in the critical path.** The tracker consumes the
simulator's own true boxes plus injected noise, so it can never fail for a reason we cannot fix.
RL trains on state (own pose, last-seen target pose, time since seen, 16 raycasts, light level
at the last sighting, channel-alive flags; 4-frame stack), never on pixels; rendering exists for
clips only. The reward is the frozen tracker's lock, run inside the training loop at state speed.
"Thermal" is a **thermal proxy**: a second render pass with an emissive target material,
unaffected by light level.

## Where it breaks (honest list — every line traces to `reports/deviation-log.md` or a JSON)

- **Coasting through occlusion is a lottery.** ByteTrack's image-space Kalman keeps the id across
  a 2 s gap in 16 of 20 noise seeds at noise dial 0.3, flat in box speed and state model
  (row 6). The hunter's job is to keep gaps short.
- **Moving cameras pay a latency tax.** The noise model delivers boxes one step late at dial
  0.3; retention scores against the current box, so the faster the box moves in the image, the
  lower the IoU. The static floor is exempt by construction.
- **Darkness alone never breaks lock**: the thermal proxy is dark-immune, so the darkness curve
  is flat-to-rising. It rises because the scripted prey blends dark-seeking with cover-seeking
  above darkness 0.3 and hides behind pillars less — a property of the examiner, stated here.
- **The prey examines movers harder than the floor.** Disabling its line-of-sight break widens the
  scripted-vs-static gap from +10.2 to +16.8 points (`reports/los_break_bias.json`): the shipped
  comparison is conservative for the hunters by ≈ 6.5 points.
- **Reward ≠ visibility.** A policy rewarded for raw visibility learned to strafe and yaw so hard
  that boxes jumped ~8 px/step and the tracker switched ids 5.6× per episode — visibility up,
  retention down (row 8). Exploration noise above std ≈ 0.2 pushes the target out of the field
  of view (row 9). Letting the policy observe the tracker's lock instead of geometry (row 12)
  made it worse on held-out seeds (row 13). Three local PPO designs; one passes the floor, none
  beats the scripted hunter.
- **The learned policy never beat the script, and breaks under pressure.** The prey trial above
  is the sharpest version: under an adversarially trained evader the PPO hunter loses 15 points
  while the scripted one loses 1. Five local training runs (three of them
  instrument-defective and diagnosed as such) and a 45-unit HPC sweep over seeds, reward
  variants and arena densities; the best PPO hunter ties the hand-written one. What PPO does
  buy is a faster reacquisition (1.70 vs 2.09 steps) at the same retention.

## Run it

```bash
uv sync                                                # Python 3.11, per-project venv
uv run python scripts/check_gates.py --all             # every gate, verdicts -> reports/gates.json
uv run python -m lockon.harness.eval --policy scripted --vs static --n 20 --seed-base 1000
uv run python -m lockon.harness.sweep --config configs/sweep_local.yaml --workers 6
uv run python -m lockon.harness.train --config configs/ppo_local.yaml --out runs/ppo_local
uv run python -m lockon.demo.render_all --policy scripted && uv run python -m lockon.demo.viewer
```

Windows 11 + GLFW offscreen rendering is the dev target; Linux uses EGL or OSMesa
(`MUJOCO_GL`). State-only PPO trains on CPU by design (~2.9k env steps/s on a laptop; 6M steps
≈ 35 min). The sweep is memory-bound: keep `--workers` ≤ 6 on 16 GB.

## Licences

Apache-2.0. Dependencies are Apache/MIT/BSD only (`THIRD_PARTY.md`); no datasets, no weights.
A guard test (`tests/test_license_guard.py`) classifies every committed visual and rejects
untracked source files and anchored-ignore mistakes.
