# lockon.sensor

Owns rendering: turns a `WorldState` + a duck-typed `(model, data, camera)` handle into a
multi-channel `SensorFrame` (rgb, depth, "thermal proxy" — an emissive second render pass that
kills every light + the headlight and whitens the person geoms before rendering, then restores
model state byte-exact; never called "thermal" alone in public wording) and composes a
side-by-side strip for GIFs. Each channel
renderer is registered by name in `RENDERERS`; adding a channel means one `core.CHANNEL_SPECS`
entry plus one renderer here, nothing else. Framing: this is the sensing side of the drone-safety
/ search-and-rescue perception stack, degraded by darkness and channel loss (project.md §1).

Import rule (`tests/test_boundaries.py`): `sensor` may import `lockon.core` + third-party only
(`mujoco`, `numpy`); explicitly not `lockon.env` — it takes a duck-typed handle instead
(`env.model`, `env.data`, `Env.CAMERA` happen to satisfy it).

## Public API
- `Sensor(model, data, camera, width=IMAGE_WIDTH, height=IMAGE_HEIGHT, seed=0)`.
- `.capture(state: WorldState, darkness: float) -> SensorFrame`; `.close() -> None`.
- `side_by_side(frame: SensorFrame, labels: bool = True) -> npt.NDArray[np.uint8]` — (H, 3W, 3).
- `RENDERERS: dict[str, ChannelRenderer]`; `register(name) -> Callable[[ChannelRenderer], ChannelRenderer]`.
- `RenderContext(model, data, renderer, camera, state, darkness, rng, scene_option)` —
  `scene_option.geomgroup` renders every geom group except group 2 (the drone body, set 0), so the
  camera never sees its own airframe; the person figure (group 3) is visible by default.
- Renderers registered under `"rgb"`, `"depth"`, `"thermal"` in `channels.py`.

## Deviations from SPEC
- `rgb` renderer: SPEC said "apply gain `(1 − darkness)` and additive gaussian noise". The code
  applies only the additive noise (σ = 4 + 40·darkness) — scene brightness is already the env's
  job (`Env.set_darkness` scales the MJCF lights + headlight fill), so an extra gain here
  double-dimmed the image. Not found in `reports/deviation-log.md`; flagged for a log row.

## Invariants (tests/test_sensor.py)
- Registry keys match `CHANNELS` exactly: `test_registry_matches_channels`.
- Every renderer returns uint8 `(H,W)` or `(H,W,3)`: `test_each_renderer_returns_uint8_image`.
- rgb darkens ≥ 6.7x at darkness 1.0 vs 0.0; depth/thermal stay within 5%: `test_rgb_dims_with_darkness_thermal_depth_stable`.
- Thermal pass restores `light_active`/`mat_emission`/`mat_rgba` byte-exact: `test_thermal_pass_restores_model`.
- Dead channel → `images[c]`/`gt[c]` both None; alive+unseen → image present, `gt` None; seen → `gt` box: `test_capture_alive_and_gt_semantics`.
- `side_by_side` shape `(H, 3W, 3)`, dead panel has noise: `test_side_by_side_shape_and_no_signal_noise`.
- Add-channel locality — `"rgb"`/`"depth"`/`"thermal"` literals never appear in env/track/policy: `test_add_channel_locality`.
- Module is marked `render`: `test_module_is_marked_render`.

## Gate (plan.md §3, G2)
```
uv run python -m lockon.harness.render --scene sensor3 --gif reports/gifs/sensor_3ch.gif
uv run pytest tests/test_sensor.py
```
