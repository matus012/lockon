"""Deterministic scripted scenes (SPEC.md `scenes.py`) — fixed layouts via fixed seeds, used by
tests and (Phase B) `render.py` GIFs.
"""

from __future__ import annotations

import logging
from collections.abc import Callable
from dataclasses import dataclass

import numpy as np
import numpy.typing as npt

from lockon.core.schemas import EPISODE_STEPS, ArenaLayout, Difficulty, FloatArray, WorldState
from lockon.env.env import Env
from lockon.harness.episode import Scene
from lockon.policy.base import Hunter, Prey
from lockon.policy.hunters import ScriptedHunter, StaticCamera
from lockon.policy.prey import ScriptedPrey

logger = logging.getLogger(__name__)


class PathPrey:
    """Prey that walks a fixed straight-line path (SPEC.md `occlusion`). The path is a constant
    velocity, set once via `set_path` by a `Scene` callable that has read `Env.state()`; until
    set, it stands still (matches `Prey.reset` being the only thing called before the scene runs).
    """

    def __init__(self) -> None:
        self.name = "path_prey"
        self._velocity: FloatArray = np.zeros(2, dtype=np.float64)

    def reset(self, layout: ArenaLayout, difficulty: Difficulty, seed: int) -> None:
        self._velocity = np.zeros(2, dtype=np.float64)

    def set_path(self, velocity: FloatArray) -> None:
        self._velocity = velocity

    def act(
        self, state: WorldState, illumination: Callable[[npt.ArrayLike], float]
    ) -> tuple[float, float]:
        return float(self._velocity[0]), float(self._velocity[1])


def _pick_occluding_pillar(
    pillars: FloatArray, drone_xy: FloatArray, person_xy: FloatArray
) -> tuple[float, float, float] | None:
    """The pillar nearest the drone -> person segment (relative to the drone's start, SPEC.md) —
    the one a short walk from the person's own position is most likely to bring into occlusion.
    Candidates are restricted to `s` ahead of the drone (not behind it, `s_frac` unclamped so a
    pillar a bit beyond the person still counts, since the walk continues past its own start).
    """
    ray = person_xy - drone_xy
    dist = float(np.linalg.norm(ray))
    if dist < 1e-9 or pillars.shape[0] == 0:
        return None
    ray_dir = ray / dist
    best: tuple[float, float, float] | None = None
    best_perp = float("inf")
    for px, py, hw in pillars:
        rel = np.array([px, py], dtype=np.float64) - drone_xy
        s = float(np.dot(rel, ray_dir))
        if s <= 0.0:
            continue
        perp = float(abs(rel[0] * ray_dir[1] - rel[1] * ray_dir[0]))
        if perp < best_perp:
            best_perp = perp
            best = (float(px), float(py), float(hw))
    return best


_SHADOW_MARGIN_M: float = 1.0  # past the pillar's far edge, so the aim point sits inside its shadow


def _shadow_point(drone_xy: FloatArray, pillar: tuple[float, float, float]) -> FloatArray:
    """A point on the drone -> pillar ray, `_SHADOW_MARGIN_M` beyond the pillar's far edge — by
    construction inside the pillar's occlusion shadow as seen from `drone_xy` (the ray passes
    through the pillar's centre, and the pillar's footprint spans `+/- half_width` around it).
    """
    px, py, hw = pillar
    p_xy = np.array([px, py], dtype=np.float64)
    d = float(np.linalg.norm(p_xy - drone_xy))
    unit = (p_xy - drone_xy) / d if d > 1e-9 else np.array([1.0, 0.0], dtype=np.float64)
    return np.asarray(drone_xy + (d + hw + _SHADOW_MARGIN_M) * unit, dtype=np.float64)


def _make_occlusion_scene(prey: PathPrey) -> Scene:
    def scene(env: Env, t: int) -> None:
        if t != 0:
            return
        state = env.state()
        drone_xy = np.array([state.drone.x, state.drone.y], dtype=np.float64)
        person_xy = np.array([state.person.x, state.person.y], dtype=np.float64)

        # Pillar nearest the drone -> person line of sight at t=0 (SPEC.md: "compute the pillar
        # from the layout at reset" / "relative to the drone's start"). Aim at a point just past
        # its far edge, on the drone -> pillar ray (`_shadow_point`): the walk approaches through
        # the pillar's shadow, `Env.step`'s collision push-out slides the person along its edge
        # rather than through it, and the same straight heading then carries the person back out
        # the other side — visible, occluded, visible again (empirically tuned against
        # `ScriptedHunter`'s pursuit for this scene's fixed seed; see tests/test_harness.py #2).
        pillar = _pick_occluding_pillar(env.layout.pillars, drone_xy, person_xy)
        aim = _shadow_point(drone_xy, pillar) if pillar is not None else person_xy

        direction = aim - person_xy
        d_norm = float(np.linalg.norm(direction))
        unit = direction / d_norm if d_norm > 1e-9 else np.array([1.0, 0.0], dtype=np.float64)
        speed = 0.5 * env.person_max_speed()
        prey.set_path(unit * speed)

    return scene


def _lights_cut_scene(env: Env, t: int) -> None:
    if t == 60:
        env.set_darkness(1.0)


@dataclass(frozen=True)
class SceneSpec:
    difficulty: Difficulty
    seed: int
    steps: int
    hunter: Hunter
    prey: Prey
    scene: Scene | None = None


def _build_occlusion() -> SceneSpec:
    prey = PathPrey()
    return SceneSpec(
        difficulty=Difficulty(),  # mid (SPEC.md)
        seed=11,
        steps=EPISODE_STEPS,
        hunter=ScriptedHunter(),
        prey=prey,
        scene=_make_occlusion_scene(prey),
    )


def _build_lights_cut() -> SceneSpec:
    return SceneSpec(
        difficulty=Difficulty(darkness=0.0, prey_aggressiveness=0.0, prey_speed=0.2, channel_dropout=0.0),
        seed=12,
        steps=EPISODE_STEPS,
        hunter=StaticCamera(),
        prey=ScriptedPrey(),
        scene=_lights_cut_scene,
    )


def _build_sensor3() -> SceneSpec:
    return SceneSpec(
        difficulty=Difficulty(darkness=0.2, prey_aggressiveness=0.0, prey_speed=0.2, channel_dropout=0.0),
        seed=13,
        steps=60,
        hunter=StaticCamera(),
        prey=ScriptedPrey(),
        scene=None,
    )


def _build_chase() -> SceneSpec:
    return SceneSpec(
        difficulty=Difficulty(),  # mid
        seed=14,
        steps=EPISODE_STEPS,
        hunter=ScriptedHunter(),
        prey=ScriptedPrey(),
        scene=None,
    )


SCENES: dict[str, SceneSpec] = {
    "occlusion": _build_occlusion(),
    "lights_cut": _build_lights_cut(),
    "sensor3": _build_sensor3(),
    "chase": _build_chase(),
}
