# demos/showcase — the reel

Twelve clips and one captioned cut, all produced by a single command:

```bash
uv run python -m lockon.demo.showcase                 # everything, ~25 min
uv run python -m lockon.demo.showcase --list          # the section names
uv run python -m lockon.demo.showcase --only sizzle   # just the cut
```

**Nothing in this directory is trained, and nothing here produces a number that appears in the
[root README](../../README.md)'s result tables.** Every clip is one episode of policies that
already exist (`models/*.zip`, the scripted baselines) played through the same frozen perception
stack every evaluation uses. The retention counter burned into a clip is that clip's own
within-window retention over a single episode; the headline figures are means over 80 held-out
seeds. Every clip says so on its face.

The `.mp4` files are **not committed** — `.gitignore` denies every `mp4`/`gif` by default and the
per-file allowlist in `tests/test_license_guard.py` negates only the six small README GIFs under
`reports/gifs/`. Rebuild the mp4s from the command above; `manifest.json` records what was last
built, with sizes, durations and the git HEAD they came from.

## What is in it

| section | what it shows |
|---|---|
| `scenario_open` | 4 pillars. Almost nothing to hide behind. |
| `scenario_dense` | 24 pillars. Occlusion is the dominant loss cause. |
| `scenario_corridor` | Two pillar rows, one lane — a fixed geometry that forces repeated brief occlusions. |
| `scenario_dark` | Lights out. The colour channel is blind and the thermal proxy carries the lock. |
| `split_*` | static camera \| scripted hunter \| PPO hunter, one per arena, on the same seed and clock. |
| `pair_hunter` | An untrained PPO hunter beside the trained one. |
| `pair_evader` | A random-walk evader beside the one trained to break the lock. |
| `evader_reel` | The trained evader alone, with its behaviour labelled every step. |
| `sensor_reel` | rgb \| depth \| thermal through a lights-cut, each flagged alive / blind / dropped out. |
| `sizzle` | A ~75 s captioned cut of all of the above. |

## How to read the overlays

Boxes are coloured by what the **metric** makes of each track, not by what looks good:

- **green** — a track carrying the target's first id, matched to the ground-truth box at IoU ≥ 0.5.
  This is the only state that counts as retained.
- **amber** — the target's id is still alive but unmatched this step: the tracker is coasting
  through an occlusion. Looks like a lock, does not count as one.
- **red** — some other id is on the target. An id switch *is* the failure the metric measures.

The minimap shows the arena from above: grey squares are pillars, yellow is the drone with its
heading, magenta is the person, and the line between them is green when the line of sight is
clear and red when it is not.

## Three things the clips do not let themselves imply

**1. The split-screen panels diverge, and that is physics.** All three hunters start from the
same seed, the same dials and the same clock, but the person reacts to whichever drone it is
facing. After step 0 the three worlds are genuinely different episodes. There is no way to hold
an evader's trajectory fixed across hunters in a closed loop, and pretending otherwise by
replaying one trajectory would make the comparison meaningless.

**2. The before/after pairs are one episode each, so they were measured separately.**
`scripts/showcase_pair_stats.py` runs both pairs over the 80 pre-registered held-out seeds
(1000–1079, the same block behind the headline) at mid difficulty, paired per seed:

| pair | before | after | difference [95 % CI] |
|---|---|---|---|
| PPO hunter: untrained → trained | 44.4 % | 51.3 % | **+6.9 pts [+0.9, +13.0]** |
| evader: random walk → trained | 52.9 % | 50.8 % | +2.1 pts [−6.7, +10.8] |

So the hunter pair is real: training the PPO camera policy is worth about seven points, and the
interval excludes zero. An **untrained** PPO hunter scores 44.4 %, which is a static camera
(44.0 %) — the policy starts out no better than not moving at all.

The evader pair is **not** a detectable difference. The published evader is attempt 1, the one
trained against geometric visibility before deviation-log row 20 corrected the objective; against
the scripted hunter it costs only two points more than a person wandering at random, and the
interval spans zero. The clip is still true — you can watch it move behind a pillar and the box
turn amber — but the aggregate effect it has on this hunter is not distinguishable from noise,
and this measurement independently reproduces why row 20 exists. The two corrected-objective
evaders that the root README's replicated ordering rests on were trained on the cluster and only
their eval JSONs came home, so they cannot be rendered here.

**3. The geometry scenarios run with the lights on.** `open`, `dense` and `corridor` set
`darkness = 0.0`; at the mid dial (0.5) the arena renders almost black and a viewer cannot see
the occlusion the clip is about. Every clip prints all five difficulty dials in its HUD, so the
condition is always on screen. The `dark` scenario runs at `darkness = 1.0` and the reported
results are all at mid difficulty.

## Selected, never tuned

Every showcase seed comes from `showcase.select_showcase_seed()`: the first candidate in
101–160 whose episode keeps the person geometrically visible on ≥ 45 % of steps **and** contains
at least one occlusion of ≥ 5 steps (plus, for the corridor, ≥ 60 % of steps inside the lane).
Selected seeds: open 101, dense 101, corridor 119, lights-out 103, evader reel 108.

This is a **staging** criterion — is the target in frame at all, and does anything happen —
computed from the simulator's `person_visible` geometry flag. It never reads retention, so it
cannot pick a seed that flatters a policy; it only decides where the camera is standing.
`scenes.select_occlusion_seed()` already works this way (deviation-log rows 7 and 14).

It is needed because a 300-step clip is half again a standard episode, and once the scripted
hunter loses the target it often never recovers — the metric's target id is dead for good after a
switch. The first cut of this reel had a lights-out clip whose last 170 steps were an entirely
black frame, and an evader clip whose last third pointed at a pillar. Both were accurate; neither
showed a viewer anything. The rule also rejects the opposite failure: dense seeds 104–106 keep the
person visible on 100 % of steps with no occlusion at all, which demonstrates nothing either.

The behaviour labels in `evader_reel` (`BREAKING LINE OF SIGHT`, `HUGGING COVER`, `RUNNING INTO
THE DARK`, `OPENING THE GAP`, `OUT OF FRAME`, `IN THE OPEN`) are read off the simulator's own
ground-truth state — the `in_fov` and `unoccluded` geometry flags, the light field at the person,
and the distance to the nearest pillar. None of them is inferred from the video.
