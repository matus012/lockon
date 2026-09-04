"""Sensor: renders a WorldState into a multi-channel SensorFrame, plus a side-by-side composer."""

from __future__ import annotations

import logging

import mujoco  # type: ignore[import-untyped]
import numpy as np
import numpy.typing as npt

from lockon.core import CHANNELS, IMAGE_HEIGHT, IMAGE_WIDTH, Detection, SensorFrame, WorldState
from lockon.sensor import channels as _channels  # noqa: F401  (registers rgb/depth/thermal)
from lockon.sensor.registry import RENDERERS, RenderContext

logger = logging.getLogger(__name__)

_NO_SIGNAL_GREY = 60
_NO_SIGNAL_NOISE_SIGMA = 25.0
_GT_BOX_COLOR = (255, 215, 0)
_GT_BOX_THICKNESS = 2

# 5x7 bitmap font, only the glyphs the two label strings ("NO SIGNAL" and CHANNELS) need.
_FONT: dict[str, tuple[str, ...]] = {
    "A": (".###.", "#...#", "#...#", "#####", "#...#", "#...#", "#...#"),
    "B": ("####.", "#...#", "#...#", "####.", "#...#", "#...#", "####."),
    "D": ("####.", "#...#", "#...#", "#...#", "#...#", "#...#", "####."),
    "E": ("#####", "#....", "#....", "###..", "#....", "#....", "#####"),
    "G": (".###.", "#....", "#....", "#.###", "#...#", "#...#", ".###."),
    "H": ("#...#", "#...#", "#...#", "#####", "#...#", "#...#", "#...#"),
    "I": ("#####", "..#..", "..#..", "..#..", "..#..", "..#..", "#####"),
    "L": ("#....", "#....", "#....", "#....", "#....", "#....", "#####"),
    "M": ("#...#", "##.##", "#.#.#", "#...#", "#...#", "#...#", "#...#"),
    "N": ("#...#", "##..#", "#.#.#", "#..##", "#...#", "#...#", "#...#"),
    "O": (".###.", "#...#", "#...#", "#...#", "#...#", "#...#", ".###."),
    "P": ("####.", "#...#", "#...#", "####.", "#....", "#....", "#...."),
    "R": ("####.", "#...#", "#...#", "####.", "#.#..", "#..#.", "#...#"),
    "S": (".####", "#....", "#....", ".###.", "....#", "....#", "####."),
    "T": ("#####", "..#..", "..#..", "..#..", "..#..", "..#..", "..#.."),
    " ": (".....", ".....", ".....", ".....", ".....", ".....", "....."),
}


def _draw_text(
    panel: npt.NDArray[np.uint8], text: str, x0: int, y0: int, scale: int = 2
) -> None:
    """Draw uppercase `text` onto `panel` (H, W, 3) at top-left (x0, y0) in white."""
    x = x0
    for ch in text.upper():
        glyph = _FONT.get(ch, _FONT[" "])
        for row, bits in enumerate(glyph):
            for col, bit in enumerate(bits):
                if bit != "#":
                    continue
                y = y0 + row * scale
                xx = x + col * scale
                panel[y : y + scale, xx : xx + scale] = 255
        x += (len(glyph[0]) + 1) * scale


def _draw_box_outline(
    panel: npt.NDArray[np.uint8], box: npt.NDArray[np.float64], color: tuple[int, int, int]
) -> None:
    h, w = panel.shape[:2]
    x0, y0, x1, y1 = box
    xi0, yi0 = int(np.clip(x0, 0, w - 1)), int(np.clip(y0, 0, h - 1))
    xi1, yi1 = int(np.clip(x1, 0, w - 1)), int(np.clip(y1, 0, h - 1))
    t = _GT_BOX_THICKNESS
    panel[yi0 : yi0 + t, xi0 : xi1 + 1] = color
    panel[max(yi1 - t + 1, 0) : yi1 + 1, xi0 : xi1 + 1] = color
    panel[yi0 : yi1 + 1, xi0 : xi0 + t] = color
    panel[yi0 : yi1 + 1, max(xi1 - t + 1, 0) : xi1 + 1] = color


def _no_signal_panel(
    width: int, height: int, rng: np.random.Generator
) -> npt.NDArray[np.uint8]:
    noise = rng.normal(_NO_SIGNAL_GREY, _NO_SIGNAL_NOISE_SIGMA, size=(height, width, 3))
    panel = np.clip(noise, 0, 255).astype(np.uint8)
    _draw_text(panel, "NO SIGNAL", x0=width // 2 - 90, y0=height // 2 - 7, scale=2)
    return panel


def _to_rgb(image: npt.NDArray[np.uint8]) -> npt.NDArray[np.uint8]:
    if image.ndim == 2:
        return np.stack([image] * 3, axis=-1)
    return image


class Sensor:
    """Owns one `mujoco.Renderer` and renders every alive channel into a `SensorFrame`."""

    def __init__(
        self,
        model: mujoco.MjModel,
        data: mujoco.MjData,
        camera: str,
        width: int = IMAGE_WIDTH,
        height: int = IMAGE_HEIGHT,
        seed: int = 0,
    ) -> None:
        self.model = model
        self.data = data
        self.camera = camera
        self.width = width
        self.height = height
        self.renderer = mujoco.Renderer(model, height=height, width=width)
        self.rng = np.random.default_rng(seed)
        logger.info(
            "Sensor initialised: camera=%s size=%dx%d channels=%s seed=%d",
            camera, width, height, CHANNELS, seed,
        )

    def capture(self, state: WorldState, darkness: float) -> SensorFrame:
        ctx = RenderContext(
            model=self.model,
            data=self.data,
            renderer=self.renderer,
            camera=self.camera,
            state=state,
            darkness=darkness,
            rng=self.rng,
        )
        images: dict[str, npt.NDArray[np.uint8] | None] = {}
        gt: dict[str, Detection | None] = {}
        for c in CHANNELS:
            images[c] = RENDERERS[c](ctx) if state.channels_alive[c] else None
            if state.channels_see[c] and state.person_box is not None:
                gt[c] = Detection(box=state.person_box, score=1.0, channel=c, t=state.t)
            else:
                gt[c] = None
        alive = dict(state.channels_alive)
        return SensorFrame(
            t=state.t, images=images, alive=alive, gt=gt, width=self.width, height=self.height
        )

    def close(self) -> None:
        self.renderer.close()


def side_by_side(frame: SensorFrame, labels: bool = True) -> npt.NDArray[np.uint8]:
    """Compose `frame.images` into one (H, 3W, 3) uint8 strip, in `CHANNELS` order."""
    rng = np.random.default_rng(frame.t)
    panels: list[npt.NDArray[np.uint8]] = []
    for c in CHANNELS:
        image = frame.images.get(c)
        if image is None:
            panel = _no_signal_panel(frame.width, frame.height, rng)
        else:
            panel = _to_rgb(image).copy()
            det = frame.gt.get(c)
            if det is not None:
                _draw_box_outline(panel, det.box, _GT_BOX_COLOR)
        if labels and image is not None:
            _draw_text(panel, c, x0=8, y0=8, scale=2)
        panels.append(panel)
    return np.concatenate(panels, axis=1)
