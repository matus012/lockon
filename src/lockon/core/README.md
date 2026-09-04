# lockon.core

Typed schemas and pure-numpy math shared by every other package: world/image frame conventions,
`Difficulty` dials, `Pose2D`/`ArenaLayout`/`WorldState` (ground truth), `Detection`/`Track`/
`LockStatus`/`SensorFrame` (perception types), `AgentObs`/`AgentAction` (RL types), the channel
registry (`CHANNELS`, `CHANNEL_SPECS`, `channel_sees`), and the headline metric in `metrics.py`
(`iou_xyxy`, `retention`, `RetentionResult` — steps tracked with the target's first-assigned id,
i.e. lock retention %, plus time-to-reacquire). No SPEC.md exists for this package; the code and
`tests/test_core.py` are authoritative.

Import rule (`tests/test_boundaries.py`): `core` imports numpy + stdlib only
(`test_core_has_no_third_party_beyond_numpy`); no package may import from `core` back into
itself and no other package's import target may be missing from `ALLOWED` — every package
imports `core`, `core` imports nothing under `lockon.*`.

## Public API
- `Difficulty(occluder_density, darkness, prey_speed, prey_aggressiveness, channel_dropout)` — 5 dials in [0,1], `.replace(**kw)`.
- `person_max_speed(difficulty) -> float` — 0.6 to 2.2 m/s.
- `Pose2D(x, y, yaw)` / `.as_array()`.
- `ArenaLayout(half_size, pillars, lights, drone_altitude)`.
- `WorldState(t, drone, person, person_visible, person_box, raycasts, illumination_at_person, channels_alive, channels_see)`.
- `Detection(box, score, channel, t)`, `Track(track_id, box, t)`, `LockStatus(t, locked, track_id, steps_since_lock)`.
- `SensorFrame(t, images, alive, gt, width, height)`.
- `AgentObs(own, last_seen_rel, time_since_seen, raycasts, light, channels_alive)` — `.vector()`, `.size()`.
- `AgentAction(vx, vy, yaw_rate)` — `.from_array()`, `.as_array()`, `.zero()`, `.size()`.
- `CHANNELS`, `CHANNEL_SPECS: dict[str, ChannelSpec]`, `channel_sees(channel, illumination, range_m) -> bool`.
- `iou_xyxy(a, b) -> float`; `retention(gt_boxes, tracks, iou_threshold=0.5, episode_steps=None) -> RetentionResult`.

## Invariants (tests/test_core.py)
- IoU identity/disjoint/partial: `test_iou_identity_and_disjoint`.
- Perfect tracking scores 1.0, no loss events: `test_retention_perfect_tracking`.
- An id switch is a loss even when the box still matches: `test_retention_id_switch_is_a_loss_even_if_box_matches`.
- Reacquire time is measured correctly: `test_retention_reacquire_time`.
- A missing GT box never counts as retained: `test_retention_gt_none_counts_as_not_retained`.
- Row-count mismatch (incl. vs `episode_steps`) raises: `test_retention_length_mismatch_raises`.
- Dial bounds enforced, `.replace` works: `test_difficulty_bounds_and_replace`.
- Obs/action vector shapes and clipping: `test_obs_and_action_vectors`.

## Gate (plan.md §3, G0)
```
uv run mypy --strict src/lockon/core
uv run pytest tests/test_boundaries.py
```
Unit suite: `uv run pytest tests/test_core.py -q`.
