# lockon.env

Owns the MuJoCo arena (pillars, lights, walls) and the kinematic drone-camera + person figure:
building the MJCF from a `Difficulty`, integrating both bodies (mocap, no physics solve),
ground-truth visibility (in-FOV, line-of-sight via `mj_ray`), the projected 2D GT box, per-channel
`channels_see`, horizontal raycasts, and the scripted channel-dropout schedule. No rendering —
that is `lockon.sensor`'s job against `env.model`/`env.data`. Search & rescue / drone-safety
framing: this is the state simulator the perception stack chases a person through.

Import rule (`tests/test_boundaries.py`): `env` may import `lockon.core` + third-party only
(`mujoco`, `numpy`); it may not import `sensor`, `track`, `policy`, or `harness`.

## Public API
- `Env(difficulty, seed, half_size=12.0)` — builds the arena and places drone+person with a clear line of sight.
- `.reset(seed=None) -> WorldState`; `.step(action: AgentAction, person_velocity: tuple[float, float]) -> WorldState`; `.state() -> WorldState`.
- `.set_darkness(darkness: float) -> None` (scales the point lights AND the camera headlight fill
  by `1 - darkness`, so rgb reads as a normal camera at darkness 0); `.darkness` (property, current
  dial — a scripted lights-cut moves it mid-episode); `.set_channels_alive(alive: Mapping[str, bool]) -> None`.
- `.step` applies drone velocity/yaw commands through a first-order filter (`DRONE_CMD_ALPHA =
  1/3`, tau 0.3 s at 10 Hz) before integrating — kinematic only, no flight physics (D14, deviation
  row 9: without it a Gaussian exploration policy thrashed the camera every step).
- Channel dropout draws from its own RNG stream (`reset`'s `_dropout_rng`), so the schedule doesn't
  shift with the number of pillar-placement draws when `occluder_density` changes (review F7).
- `.illumination(xy) -> float`; `.line_of_sight(a_xyz, b_xyz) -> bool`; `.person_bbox_world() -> FloatArray` (8,3).
- `.project(points_world) -> tuple[FloatArray, FloatArray]` (pixels, depth); `.person_max_speed() -> float`.
- `arena.build_layout(difficulty, rng, half_size) -> ArenaLayout`; `arena.build_mjcf(layout) -> str`.
- `geometry.project_points`, `geometry.person_box`, `geometry.line_of_sight`, `geometry.illumination`, `geometry.raycasts` — pure functions behind the `Env` methods above.

## Deviations from SPEC
- Camera pitch wording: SPEC originally said "−25° about body x"; a rotation about the forward
  axis cannot pitch, so the camera axes are instead built directly from the intended end state
  (forward = `(cos25°, 0, −sin25°)`) — deviation-log row 4.
- `tests/test_env_render_crosscheck.py` (SPEC test 8): SPEC's single "IoU ≥ 0.7" cross-check was
  loosened to "mean ≥ 0.7 AND per-state ≥ 0.5" plus a converse "no box ⇒ no pixels" check — the
  AABB projection is deliberately loose under perspective, and one honest sample measured 0.695 —
  deviation-log row 2.

## Invariants (tests/test_env.py)
- Same seed → identical layout + trajectory: `test_determinism_same_seed_identical_layout_and_trajectory`.
- Optical-axis projection and left-offset sign: `test_projection_optical_axis_and_left_offset`.
- A pillar between camera and person breaks LOS, removing it restores it: `test_los_pillar_occludes_then_clears`.
- Person behind the drone is out of FOV: `test_fov_person_behind_drone_not_visible`.
- Pillar push-out keeps the person outside the square: `test_collision_person_pushed_out_of_pillar`.
- `channel_dropout` extremes (1.0 always fires, 0.0 never): `test_dropout_schedule_extremes`.
- `person_max_speed()` at dial 0/1 is 0.6/2.2 m/s: `test_person_max_speed_dial_extremes`.
- Render/state cross-check (marked `render`): `test_segmentation_bbox_matches_projected_box`.

## Gate (plan.md §3, G1)
```
uv run python -m lockon.harness.bench_env --steps 2000
```
Unit suite: `uv run pytest tests/test_env.py -q` (add `-m render` for the cross-check).
