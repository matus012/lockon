"""Channel renderers (SPEC.md `## Renderers`). One function each, registered by name."""

from __future__ import annotations

import logging
from typing import cast

import mujoco  # type: ignore[import-untyped]
import numpy as np
import numpy.typing as npt

from lockon.core.schemas import PERSON_MATERIAL
from lockon.sensor.registry import RenderContext, register

logger = logging.getLogger(__name__)

_MAX_DEPTH_M = 20.0
_PERSON_MAT_NAME = PERSON_MATERIAL
_THERMAL_GLOW_THRESHOLD = 64  # emissive person renders ~255; an empty frame must stay dark (review finding 14)
_THERMAL_BACKGROUND = 40.0
_THERMAL_PERSON = 255.0


@register("rgb")
def render_rgb(ctx: RenderContext) -> npt.NDArray[np.uint8]:
    """Normal scene render plus additive sensor noise that grows with darkness.

    Scene brightness is the env's job (`Env.set_darkness` scales the lights and the headlight
    fill); applying a gain here as well double-dimmed the image (visual review 2026-09-04).
    """
    ctx.renderer.update_scene(ctx.data, camera=ctx.camera, scene_option=ctx.scene_option)
    rgb = ctx.renderer.render().astype(np.float64)
    sigma = 4.0 + 40.0 * ctx.darkness
    noisy = rgb + ctx.rng.normal(0.0, sigma, size=rgb.shape)
    return cast(npt.NDArray[np.uint8], np.clip(noisy, 0, 255).astype(np.uint8))


@register("depth")
def render_depth(ctx: RenderContext) -> npt.NDArray[np.uint8]:
    """Metric depth render mapped to uint8, near = bright (SPEC.md). Unaffected by darkness."""
    ctx.renderer.enable_depth_rendering()
    try:
        ctx.renderer.update_scene(ctx.data, camera=ctx.camera, scene_option=ctx.scene_option)
        depth_m = ctx.renderer.render()
        scaled = 255.0 * (1.0 - np.clip(depth_m / _MAX_DEPTH_M, 0.0, 1.0))
        return cast(npt.NDArray[np.uint8], scaled.astype(np.uint8))
    finally:
        ctx.renderer.disable_depth_rendering()


@register("thermal")
def render_thermal(ctx: RenderContext) -> npt.NDArray[np.uint8]:
    """Thermal proxy pass: lights off, person material glowing; state restored exactly."""
    model = ctx.model
    mat_id = mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_MATERIAL, _PERSON_MAT_NAME)
    if mat_id < 0:
        raise ValueError(f"material {_PERSON_MAT_NAME!r} not found in model")

    light_active = model.light_active.copy()
    mat_emission = model.mat_emission.copy()
    mat_rgba = model.mat_rgba.copy()
    geom_rgba = model.geom_rgba.copy()
    head_ambient = model.vis.headlight.ambient.copy()
    head_diffuse = model.vis.headlight.diffuse.copy()
    person_geoms = np.flatnonzero(model.geom_matid == mat_id)
    try:
        model.light_active[:] = 0
        model.vis.headlight.ambient[:] = 0.0
        model.vis.headlight.diffuse[:] = 0.0
        model.mat_emission[mat_id] = 1.0
        model.mat_rgba[mat_id] = (1.0, 1.0, 1.0, 1.0)
        # per-geom rgba overrides the material colour, so whiten the geoms themselves
        model.geom_rgba[person_geoms] = (1.0, 1.0, 1.0, 1.0)

        ctx.renderer.update_scene(ctx.data, camera=ctx.camera, scene_option=ctx.scene_option)
        rgb = ctx.renderer.render().astype(np.float64)
        gray = rgb.mean(axis=-1)

        glow_mask = gray >= _THERMAL_GLOW_THRESHOLD
        toned = np.where(
            glow_mask,
            _THERMAL_PERSON,
            np.clip(gray, 0.0, _THERMAL_BACKGROUND),
        )
        return cast(npt.NDArray[np.uint8], toned.astype(np.uint8))
    finally:
        model.light_active[:] = light_active
        model.mat_emission[:] = mat_emission
        model.mat_rgba[:] = mat_rgba
        model.geom_rgba[:] = geom_rgba
        model.vis.headlight.ambient[:] = head_ambient
        model.vis.headlight.diffuse[:] = head_diffuse
