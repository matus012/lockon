"""lockon.sensor spec tests (src/lockon/sensor/SPEC.md). Do not import lockon.env (boundary)."""

from __future__ import annotations

import ast
from collections.abc import Iterator
from pathlib import Path

import mujoco
import numpy as np
import pytest

from lockon.core import CHANNELS, Pose2D, WorldState
from lockon.sensor import RENDERERS, RenderContext, Sensor, side_by_side

pytestmark = pytest.mark.render

_WIDTH = 320
_HEIGHT = 240

_XML = """
<mujoco>
  <asset>
    <material name="person_mat" rgba="0.8 0.2 0.2 1" emission="0"/>
  </asset>
  <worldbody>
    <light name="sun" pos="0 0 5" dir="0 0 -1" diffuse="6 6 6" specular="2 2 2"/>
    <geom type="plane" size="10 10 0.1" rgba="1 1 1 1"/>
    <geom name="pillar" type="box" pos="2 2 1" size="0.3 0.3 1" rgba="1 1 1 1"/>
    <body name="person" pos="0 0 0.9">
      <geom name="torso" type="capsule" size="0.18 0.35" material="person_mat"/>
    </body>
    <camera name="cam" pos="0 -5 2" xyaxes="1 0 0 0 0.4 0.9"/>
  </worldbody>
</mujoco>
"""


def _all_groups() -> mujoco.MjvOption:
    opt = mujoco.MjvOption()
    opt.geomgroup[:] = 1
    return opt


def _make_state(t: int = 0, box: np.ndarray | None = None, **overrides: object) -> WorldState:
    defaults: dict[str, object] = {
        "t": t,
        "drone": Pose2D(0.0, -5.0, 1.5708),
        "person": Pose2D(0.0, 0.0, 0.0),
        "person_visible": True,
        "person_box": box if box is not None else np.array([130.0, 90.0, 190.0, 200.0]),
        "raycasts": np.zeros(16, dtype=np.float64),
        "illumination_at_person": 1.0,
    }
    defaults.update(overrides)
    return WorldState(**defaults)  # type: ignore[arg-type]


@pytest.fixture()
def model_data() -> tuple[mujoco.MjModel, mujoco.MjData]:
    model = mujoco.MjModel.from_xml_string(_XML)
    data = mujoco.MjData(model)
    mujoco.mj_forward(model, data)
    return model, data


@pytest.fixture()
def sensor(model_data: tuple[mujoco.MjModel, mujoco.MjData]) -> Iterator[Sensor]:
    model, data = model_data
    s = Sensor(model, data, camera="cam", width=_WIDTH, height=_HEIGHT, seed=0)
    yield s
    s.close()


def test_registry_matches_channels() -> None:
    assert tuple(RENDERERS) == CHANNELS


def test_each_renderer_returns_uint8_image(
    model_data: tuple[mujoco.MjModel, mujoco.MjData], sensor: Sensor
) -> None:
    model, data = model_data
    state = _make_state()
    ctx = RenderContext(
        model=model,
        data=data,
        renderer=sensor.renderer,
        camera="cam",
        state=state,
        darkness=0.0,
        rng=np.random.default_rng(0),
        scene_option=_all_groups(),
    )
    for c in CHANNELS:
        img = RENDERERS[c](ctx)
        assert img.dtype == np.uint8
        assert img.shape in {(_HEIGHT, _WIDTH), (_HEIGHT, _WIDTH, 3)}


def test_rgb_dims_with_darkness_thermal_depth_stable(
    model_data: tuple[mujoco.MjModel, mujoco.MjData], sensor: Sensor
) -> None:
    model, data = model_data
    state = _make_state()
    light_diffuse_full = model.light_diffuse.copy()
    headlight_ambient_full = model.vis.headlight.ambient.copy()
    headlight_diffuse_full = model.vis.headlight.diffuse.copy()

    def render_all(darkness: float) -> dict[str, np.ndarray]:
        model.light_diffuse[:] = light_diffuse_full * (1.0 - darkness)
        # env also scales the camera headlight fill with darkness (Env.set_darkness)
        model.vis.headlight.ambient[:] = headlight_ambient_full * (1.0 - darkness)
        model.vis.headlight.diffuse[:] = headlight_diffuse_full * (1.0 - darkness)
        mujoco.mj_forward(model, data)
        ctx = RenderContext(
            model=model,
            data=data,
            renderer=sensor.renderer,
            camera="cam",
            state=state,
            darkness=darkness,
            rng=np.random.default_rng(0),
            scene_option=_all_groups(),
        )
        return {c: RENDERERS[c](ctx) for c in CHANNELS}

    bright = render_all(0.0)
    dark = render_all(1.0)
    model.light_diffuse[:] = light_diffuse_full
    model.vis.headlight.ambient[:] = headlight_ambient_full
    model.vis.headlight.diffuse[:] = headlight_diffuse_full
    mujoco.mj_forward(model, data)

    rgb_bright_mean = bright["rgb"].astype(np.float64).mean()
    rgb_dark_mean = dark["rgb"].astype(np.float64).mean()
    assert rgb_dark_mean < 0.15 * rgb_bright_mean

    for c in ("depth", "thermal"):
        m0 = bright[c].astype(np.float64).mean()
        m1 = dark[c].astype(np.float64).mean()
        denom = max(m0, 1e-6)
        assert abs(m1 - m0) / denom < 0.05


def test_thermal_pass_restores_model(
    model_data: tuple[mujoco.MjModel, mujoco.MjData], sensor: Sensor
) -> None:
    model, data = model_data
    state = _make_state()
    light_active_before = model.light_active.copy()
    mat_emission_before = model.mat_emission.copy()
    mat_rgba_before = model.mat_rgba.copy()

    ctx = RenderContext(
        model=model,
        data=data,
        renderer=sensor.renderer,
        camera="cam",
        state=state,
        darkness=0.0,
        rng=np.random.default_rng(0),
        scene_option=_all_groups(),
    )
    RENDERERS["thermal"](ctx)

    assert np.array_equal(model.light_active, light_active_before)
    assert np.array_equal(model.mat_emission, mat_emission_before)
    assert np.array_equal(model.mat_rgba, mat_rgba_before)


def test_capture_alive_and_gt_semantics(sensor: Sensor) -> None:
    box = np.array([130.0, 90.0, 190.0, 200.0])
    state = _make_state(
        box=box,
        channels_alive={"rgb": True, "depth": False, "thermal": True},
        channels_see={"rgb": False, "depth": False, "thermal": True},
    )
    frame = sensor.capture(state, darkness=0.0)

    assert frame.images["depth"] is None
    assert frame.gt["depth"] is None

    assert frame.images["rgb"] is not None
    assert frame.gt["rgb"] is None

    assert frame.images["thermal"] is not None
    assert frame.gt["thermal"] is not None
    assert np.array_equal(frame.gt["thermal"].box, box)


def test_side_by_side_shape_and_no_signal_noise(sensor: Sensor) -> None:
    box = np.array([130.0, 90.0, 190.0, 200.0])
    state = _make_state(
        box=box,
        channels_alive={"rgb": True, "depth": False, "thermal": True},
        channels_see={"rgb": True, "depth": False, "thermal": True},
    )
    frame = sensor.capture(state, darkness=0.0)
    strip = side_by_side(frame)

    assert strip.shape == (_HEIGHT, 3 * _WIDTH, 3)
    assert strip.dtype == np.uint8

    dead_panel = strip[:, _WIDTH : 2 * _WIDTH]
    assert dead_panel.std() > 1.0


def test_add_channel_locality() -> None:
    src = Path(__file__).resolve().parents[1] / "src" / "lockon"
    literal_names = {"rgb", "depth", "thermal"}
    for pkg in ("env", "track", "policy"):
        pkg_dir = src / pkg
        if not pkg_dir.exists():
            continue
        for f in pkg_dir.rglob("*.py"):
            tree = ast.parse(f.read_text(encoding="utf-8"))
            for node in ast.walk(tree):
                if isinstance(node, ast.Constant) and isinstance(node.value, str):
                    assert node.value not in literal_names, (
                        f"{f}: literal channel name {node.value!r} found; "
                        "iterate core.CHANNELS/CHANNEL_SPECS instead"
                    )


def test_module_is_marked_render() -> None:
    assert pytestmark.name == "render" or pytestmark.mark.name == "render"
