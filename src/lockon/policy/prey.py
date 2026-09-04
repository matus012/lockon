"""Scripted prey / examiner (SPEC.md `Prey`). Privileged: reads full `WorldState`.

Speed limit comes from `lockon.core.person_max_speed(difficulty)` (single source; env clamps).

ASSUMPTION: `visibility_penalty` in the cover-seek score (SPEC.md, undefined) is taken as 0 when
the candidate pillar lies roughly between the person and the drone (so hiding behind it would
actually break line of sight) and 1 otherwise — i.e. it penalises pillars that would not help.
"""

from __future__ import annotations

import logging
import math
from collections.abc import Callable

import numpy as np
import numpy.typing as npt

from lockon.core.schemas import ArenaLayout, Difficulty, FloatArray, WorldState, person_max_speed

logger = logging.getLogger(__name__)

REPLAN_EVERY_STEPS = 10
WANDER_SPEED_FRACTION = 0.5
DARK_SAMPLE_COUNT = 12
DARK_SAMPLE_RADIUS_M = 6.0
LOS_BREAK_GAIN = 1.0


class ScriptedPrey:
    """Cover-seek + dark-seek + LOS-break, mixed by `Difficulty` (SPEC.md `Prey`)."""

    def __init__(self) -> None:
        self.name = "scripted_prey"
        self._layout: ArenaLayout | None = None
        self._difficulty: Difficulty | None = None
        self._rng: np.random.Generator | None = None
        self._target = np.zeros(2, dtype=np.float64)
        self._evasive = False
        self._t = 0

    def reset(self, layout: ArenaLayout, difficulty: Difficulty, seed: int) -> None:
        self._layout = layout
        self._difficulty = difficulty
        self._rng = np.random.default_rng(seed)
        self._target = np.zeros(2, dtype=np.float64)
        self._evasive = False
        self._t = 0

    def act(
        self, state: WorldState, illumination: Callable[[npt.ArrayLike], float]
    ) -> tuple[float, float]:
        assert self._layout is not None and self._difficulty is not None and self._rng is not None
        person_xy = np.array([state.person.x, state.person.y], dtype=np.float64)
        drone_xy = np.array([state.drone.x, state.drone.y], dtype=np.float64)

        reached = float(np.linalg.norm(self._target - person_xy)) < 0.3
        if self._t == 0 or self._t % REPLAN_EVERY_STEPS == 0 or reached:
            self._replan(person_xy, drone_xy, illumination)
        self._t += 1

        direction = self._target - person_xy
        aggressiveness = self._difficulty.prey_aggressiveness

        if self._evasive:
            speed = person_max_speed(self._difficulty)
            if state.person_visible and aggressiveness > 0.0:
                direction = direction + self._los_break(person_xy, drone_xy, aggressiveness)
        else:
            speed = WANDER_SPEED_FRACTION * person_max_speed(self._difficulty)

        norm_d = float(np.linalg.norm(direction))
        unit = direction / norm_d if norm_d > 1e-9 else np.zeros(2, dtype=np.float64)
        vel = unit * speed
        return float(vel[0]), float(vel[1])

    def _replan(
        self, person_xy: FloatArray, drone_xy: FloatArray, illumination: Callable[[npt.ArrayLike], float]
    ) -> None:
        assert self._layout is not None and self._difficulty is not None and self._rng is not None
        aggressiveness = self._difficulty.prey_aggressiveness
        has_pillars = self._layout.pillars.shape[0] > 0
        evasive = bool(self._rng.random() < aggressiveness) and has_pillars

        if evasive:
            cover_t = self._cover_target(person_xy, drone_xy)
            if self._difficulty.darkness > 0.3:
                dark_t = self._dark_target(person_xy, illumination)
                self._target = 0.5 * cover_t + 0.5 * dark_t
            else:
                self._target = cover_t
        else:
            self._target = self._wander_target()
        self._evasive = evasive

    def _cover_target(self, person_xy: FloatArray, drone_xy: FloatArray) -> FloatArray:
        assert self._layout is not None
        pillars = self._layout.pillars
        best_idx = 0
        best_score = math.inf
        to_drone = drone_xy - person_xy
        for i in range(pillars.shape[0]):
            p = pillars[i, :2]
            dist = float(np.linalg.norm(p - person_xy))
            to_pillar = p - person_xy
            occludes = float(np.dot(to_pillar, to_drone)) > 0.0
            visibility_penalty = 0.0 if occludes else 1.0
            score = dist + 2.0 * visibility_penalty
            if score < best_score:
                best_score = score
                best_idx = i
        p = pillars[best_idx, :2]
        half_width = float(pillars[best_idx, 2])
        away = p - drone_xy
        n = float(np.linalg.norm(away))
        unit = away / n if n > 1e-9 else np.array([1.0, 0.0], dtype=np.float64)
        result: FloatArray = p + unit * (half_width + 0.6)
        return result

    def _dark_target(
        self, person_xy: FloatArray, illumination: Callable[[npt.ArrayLike], float]
    ) -> FloatArray:
        assert self._rng is not None
        angles = self._rng.uniform(0.0, 2.0 * math.pi, size=DARK_SAMPLE_COUNT)
        radii = self._rng.uniform(0.0, DARK_SAMPLE_RADIUS_M, size=DARK_SAMPLE_COUNT)
        offsets = np.stack([radii * np.cos(angles), radii * np.sin(angles)], axis=-1)
        points = person_xy + offsets
        values = [illumination(points[i]) for i in range(points.shape[0])]
        best = int(np.argmin(values))
        result: FloatArray = points[best]
        return result

    def _wander_target(self) -> FloatArray:
        assert self._layout is not None and self._rng is not None
        bound = self._layout.half_size - 1.0
        return np.array(
            [self._rng.uniform(-bound, bound), self._rng.uniform(-bound, bound)], dtype=np.float64
        )

    def _los_break(self, person_xy: FloatArray, drone_xy: FloatArray, aggressiveness: float) -> FloatArray:
        """Lateral component perpendicular to drone->person, sign toward the nearest pillar."""
        assert self._layout is not None
        drone_to_person = person_xy - drone_xy
        perp = np.array([-drone_to_person[1], drone_to_person[0]], dtype=np.float64)
        norm_perp = float(np.linalg.norm(perp))
        if norm_perp < 1e-9:
            return np.zeros(2, dtype=np.float64)
        perp_unit = perp / norm_perp

        pillars = self._layout.pillars
        if pillars.shape[0] > 0:
            dists = np.linalg.norm(pillars[:, :2] - person_xy, axis=1)
            nearest = pillars[int(np.argmin(dists)), :2]
            if float(np.dot(perp_unit, nearest - person_xy)) < 0.0:
                perp_unit = -perp_unit
        return perp_unit * (aggressiveness * LOS_BREAK_GAIN)
