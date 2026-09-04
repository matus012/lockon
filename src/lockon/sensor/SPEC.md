# lockon.sensor — specification (lead-authored 2026-09-04; plan.md step 2c, gate G2)

Import rule: `lockon.core`, `mujoco`, `numpy`. Not `lockon.env`: the sensor takes a duck-typed
handle `(model, data, camera: str)` — env exposes exactly those attributes (`env.model`,
`env.data`, `Env.CAMERA`). Channel names and state physics come from `core.CHANNEL_SPECS`;
this package owns rendering only.

## Public API
```python
ChannelRenderer = Callable[[RenderContext], npt.NDArray[np.uint8]]   # returns (H, W) or (H, W, 3)

@dataclass
class RenderContext:
    model: mujoco.MjModel; data: mujoco.MjData; renderer: mujoco.Renderer; camera: str
    state: WorldState; darkness: float; rng: np.random.Generator

RENDERERS: dict[str, ChannelRenderer]          # registry.py — the ONLY place renderers are listed
def register(name: str) -> Callable[[ChannelRenderer], ChannelRenderer]

class Sensor:
    def __init__(self, model, data, camera: str, width=IMAGE_WIDTH, height=IMAGE_HEIGHT, seed=0) -> None
    def capture(self, state: WorldState, darkness: float) -> SensorFrame
        # for c in CHANNELS: images[c] = RENDERERS[c](ctx) if state.channels_alive[c] else None
        # gt[c] = Detection(box=state.person_box, score=1.0, channel=c, t) if state.channels_see[c] else None
        # alive = dict(state.channels_alive)
    def close(self) -> None

def side_by_side(frame: SensorFrame, labels: bool = True) -> npt.NDArray[np.uint8]   # (H, 3W, 3)
    # dead channel → grey "NO SIGNAL" panel with static noise; GT box drawn as a thin outline
    # on channels where gt[c] is not None (pure numpy drawing; no cv2/PIL dependency needed).
```

## Renderers (`channels.py`) — one function each, registered by name
- `rgb`: normal scene render at `camera`. Lights are already dimmed by env (`set_darkness`
  scales light diffuse); apply gain `(1 - darkness)` and additive gaussian noise
  σ = 4 + 40·darkness, clip to uint8. At darkness 1.0 the image is noise around 0.
- `depth`: `renderer.enable_depth_rendering()` → metres; uint8 = 255·(1 − clip(d / 20, 0, 1))
  (near = bright); disable afterwards. Unaffected by darkness.
- `thermal`: second pass ("thermal proxy", README wording): temporarily set every light
  inactive (`model.light_active[:] = 0`), set the person material's `emission` to 1.0 and its
  rgba to (1,1,1,1) (material named `person_mat`, from env SPEC), render, convert to grayscale,
  then apply a mild blur-free tone: background ≤ 40, person ≈ 255; **restore** light_active,
  emission and rgba exactly (save/restore around the render; test asserts model unchanged).
  Unaffected by darkness by construction.
Renderer reuse: one `mujoco.Renderer(model, height, width)` per Sensor; toggle depth mode
around the depth pass; never leave a mode enabled.

## Tests (`tests/test_sensor.py`)
Build a tiny MJCF inline (plane, one pillar, a "person" body with material `person_mat`, one
light, camera "cam") — do NOT import lockon.env (boundary). Construct WorldState by hand.
1. registry == CHANNELS exactly (same names, same order).
2. each renderer returns uint8 of shape (H, W) or (H, W, 3).
3. rgb: mean intensity at darkness 1.0 < 0.15 × mean at darkness 0.0 (with light diffuse
   scaled by (1−darkness) in the test to mimic env); thermal and depth means change < 5 %.
4. thermal pass leaves `model.light_active`, `model.mat_emission`, `model.mat_rgba` byte-equal.
5. dead channel → images[c] is None and gt[c] is None; alive channel with channels_see False →
   image present, gt None; channels_see True → gt box equals state.person_box.
6. side_by_side shape (H, 3W, 3) and a dead panel is non-uniform (noise present).
7. add-channel locality: scan `src/lockon/{env,track,policy}` sources for the literal channel
   names `"rgb"`, `"depth"`, `"thermal"` — none may appear (they must iterate `core.CHANNELS` /
   `CHANNEL_SPECS`). This is the machine form of "add-channel touches only sensor (+ core entry)".
   Skip gracefully for packages that do not exist yet.
8. `@pytest.mark.render` (they all render, mark the module).

Also deliver `lockon/sensor/__init__.py` exporting Sensor, RENDERERS, register, side_by_side,
RenderContext; `uv run mypy --strict src/lockon/sensor` clean. GIF writing is harness work
(`lockon.harness.render`), not sensor.
