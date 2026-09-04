# lockon.env — specification (lead-authored 2026-09-04; plan.md step 2b, gate G1)

Import rule: `lockon.core` + `mujoco` + `numpy` only. No rendering here (sensor renders from
`env.model` / `env.data`). Everything below is state math; G1 = ≥200 steps/s headless.

## Public API (`lockon/env/env.py`)
```python
class Env:
    CAMERA = "drone_cam"                       # MJCF camera name, child of body "drone"
    PERSON_BODY = "person"; DRONE_BODY = "drone"
    def __init__(self, difficulty: Difficulty, seed: int, half_size: float = 12.0) -> None
    layout: ArenaLayout; model: mujoco.MjModel; data: mujoco.MjData; difficulty: Difficulty
    def reset(self, seed: int | None = None) -> WorldState
    def step(self, action: AgentAction, person_velocity: tuple[float, float]) -> WorldState
    def state(self) -> WorldState                       # recompute from data, no side effects
    def set_darkness(self, darkness: float) -> None     # scripted lights-cut; updates light field + MJCF light intensities
    def set_channels_alive(self, alive: dict[str, bool]) -> None   # scripted override; None-safe merge
    def illumination(self, xy: ArrayLike) -> float      # [0,1] light field, used by prey dark-seek
    def line_of_sight(self, a_xyz, b_xyz) -> bool       # mj_ray, pillars+walls occlude, person/drone geoms excluded
    def person_bbox_world(self) -> FloatArray           # (8,3) corners of the figure's AABB
    def project(self, points_world: FloatArray) -> tuple[FloatArray, FloatArray]   # (N,2) pixels, (N,) depth>0 mask
    def person_max_speed(self) -> float                 # 0.6 + 1.6*prey_speed m/s
```
`WorldState` fields are filled exactly as documented in `core/schemas.py`.

## Arena (`arena.py`)
- `build_layout(difficulty, rng, half_size) -> ArenaLayout`: M = round(4 + 20*occluder_density)
  square pillars, half-width U(0.5, 1.0) m, height 3.5 m, rejection-sampled non-overlapping,
  ≥1.5 m from walls, ≥2.5 m centre clearance. L = 6 point lights, positions uniform, intensity
  U(0.6, 1.0). `drone_altitude` = 3.0 m (pillars are taller: LOS is horizontal geometry).
- `build_mjcf(layout) -> str`: plane 2*half_size, four wall boxes height 3.5, pillar boxes
  (named `pillar_i`, group 1, textured), lights (`light_i`, diffuse scaled by (1-darkness)),
  `<visual><global offwidth=640 offheight=480/>`, headlight off. Materials: floor checker texture,
  pillars flat grey-brown. `<option gravity="0 0 0"/>`; nothing is dynamic.
- Drone: `<body name="drone" mocap="true">` small box geom (group 2, contype 0) + camera
  `drone_cam` fovy=60, pitched −25° about the body x axis so it looks forward-down along +x body.
- Person: `<body name="person" mocap="true">` capsule figure, all geoms named `person_*`, group 3,
  contype 0: torso capsule r0.15 half-len 0.30 at z 1.05; head sphere r0.12 at z 1.58; two leg
  capsules r0.07 from z 0.1 to 0.75 at x ±0.1; two arm capsules r0.05 at y ±0.25 from z 0.75 to
  1.30; rgba clothing colours (torso 0.8 0.2 0.2, legs 0.2 0.2 0.6, skin 0.9 0.75 0.6). Set
  `material` `person_mat` on every person geom so sensor can swap emission for the thermal pass.
- Light field (state, not render): `illumination(xy) = clip((1-darkness)*(0.25 + Σ_i I_i*max(0,
  1-d_i/8)), 0, 1)`.

## Kinematics (`env.py`)
- Control at CONTROL_HZ; dt = 0.1 s. Drone: v_max 2.5 m/s (body-frame vx, vy scaled from
  [-1,1]), yaw_rate max 1.5 rad/s, altitude fixed. Person: world-frame velocity clamped to
  `person_max_speed()`, yaw = heading of motion when speed > 0.05.
- Collisions (both bodies): circle (r_drone 0.35, r_person 0.3) vs square pillar push-out, then
  clamp to walls with 0.4 m margin. No MuJoCo physics; `mj_forward` after mocap update.
- Channel dropout schedule (per episode, from seed and `channel_dropout`): each channel dies
  with per-step hazard 0.01*dial for a burst of U(10, 40) steps; at most one channel dead at a
  time; overrides from `set_channels_alive` persist until the next `reset`.
- `reset`: person uniform in arena (clear of pillars); drone 6–10 m away, yaw facing person, with
  line of sight (rejection-sample ≤50 tries, else place along the clear ray). Episode starts
  visible.

## Visibility and boxes (`geometry.py`)
- `person_box`: project the 8 AABB corners; if the torso centre is behind the camera → None;
  box = min/max of projected corners clipped to the image; None if clipped area < 4 px².
- In-FOV: torso centre projects inside the image. LOS: `mj_ray` from camera position to torso
  centre AND to head centre with `geomgroup` mask excluding groups 2 and 3 (drone + person); a
  point is clear if the ray hits nothing before the point distance. Unoccluded if either clear.
- `channels_see[c] = alive[c] and in_fov and unoccluded and core.channel_sees(c, illumination_at_person, range_m)`
  iterating `core.CHANNELS`; range_m = camera position → torso centre. The literal channel names
  never appear in this package (sensor test 7 greps for them). `person_visible = any(channels_see)`.
- Raycasts: N_RAYCASTS horizontal rays from the drone position at its altitude, angles
  yaw + k*2π/N, `mj_ray` against groups 0/1 (walls, pillars), distance / (2*half_size) clipped
  to [0,1]; 1.0 if nothing hit.

## Tests (`tests/test_env.py`) — instrument proofs before any number is trusted
1. determinism: same seed → identical layout and identical state trajectory under a fixed
   action sequence.
2. projection: torso centre placed on the camera's optical axis projects to (320, 240) ±1 px;
   a point 1 m to the camera's left projects at u < 320.
3. LOS: a pillar placed between camera and person → `person_visible` False; removed → True.
4. FOV: person behind the drone → `person_box` None and not visible.
5. collision: commanding the person into a pillar leaves it outside the pillar's square.
6. dropout: with `channel_dropout=1.0` some step has a dead channel; with 0.0 none ever.
7. speed: `person_max_speed()` at dial 0 and 1 equals 0.6 and 2.2.
8. `tests/test_env_render_crosscheck.py` marked `render`: render the segmentation pass at the
   camera, take the pixel bbox of person geoms, assert IoU with `person_box` ≥ 0.7 over 20 random
   states (this is the transform proof from instrument-proof-every-transform).

## Bench (`lockon/harness/bench_env.py`, tiny — harness owns CLIs)
`python -m lockon.harness.bench_env --steps 2000 --min-sps 200`: random actions + random person
velocity, reports steps/s, exits 1 below `--min-sps`.
