"""Renderer registry — the ONLY place channel renderers are listed (SPEC.md).

`RENDERERS` maps a channel name (from `core.CHANNELS`) to a `ChannelRenderer`. Renderers
register themselves via the `@register(name)` decorator in `channels.py`.
"""

from __future__ import annotations

from collections.abc import Callable
from dataclasses import dataclass

import mujoco  # type: ignore[import-untyped]
import numpy as np
import numpy.typing as npt

from lockon.core import WorldState

ChannelRenderer = Callable[["RenderContext"], "npt.NDArray[np.uint8]"]


@dataclass
class RenderContext:
    """Everything a renderer needs for one channel pass."""

    model: mujoco.MjModel
    data: mujoco.MjData
    renderer: mujoco.Renderer
    camera: str
    state: WorldState
    darkness: float
    rng: np.random.Generator
    scene_option: mujoco.MjvOption


RENDERERS: dict[str, ChannelRenderer] = {}


def register(name: str) -> Callable[[ChannelRenderer], ChannelRenderer]:
    """Register a channel renderer under `name` in `RENDERERS`."""

    def _wrap(fn: ChannelRenderer) -> ChannelRenderer:
        RENDERERS[name] = fn
        return fn

    return _wrap
