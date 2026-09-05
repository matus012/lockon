"""Learned-prey trial (SPEC_prey.md): `PreyObsBuilder`, `PreyRewardConfig`, `prey_reward`,
`LearnedPrey(Prey)`. Privileged observation (the prey is the examiner, project.md §3) — reads
full `WorldState`, unlike the hunter's `AgentObs`. core + numpy + SB3 only (boundary rule).
"""

from __future__ import annotations

import json
import logging
import math
from collections import deque
from collections.abc import Callable
from dataclasses import dataclass
from pathlib import Path

import numpy as np
import numpy.typing as npt

from lockon.core.schemas import (
    EPISODE_STEPS,
    ArenaLayout,
    Difficulty,
    FloatArray,
    WorldState,
    person_max_speed,
)
from lockon.policy.features import _rotate

logger = logging.getLogger(__name__)

_N_PILLAR_SLOTS = 4
_PILLAR_FEATS = 3  # dx, dy, half-width per pillar slot


def _wrap_angle(a: float) -> float:
    return float((a + math.pi) % (2.0 * math.pi) - math.pi)


class PreyObsBuilder:
    """Privileged prey observation (SPEC_prey.md `Prey observation`): own pose, drone relative
    in the person's own frame, visibility, illumination, the 4 nearest pillars, and a
    steps-visible-in-a-row counter (the mirror of the hunter's steps-since-seen)."""

    def __init__(self, layout: ArenaLayout) -> None:
        self._half_size = layout.half_size
        self._pillars = layout.pillars
        self._steps_visible = 0

    @staticmethod
    def size() -> int:
        return 3 + 3 + 1 + 1 + _N_PILLAR_SLOTS * _PILLAR_FEATS + 1

    def reset(self, state: WorldState) -> npt.NDArray[np.float32]:
        self._steps_visible = 1 if state.person_visible else 0
        return self._build(state)

    def step(self, state: WorldState) -> npt.NDArray[np.float32]:
        self._steps_visible = self._steps_visible + 1 if state.person_visible else 0
        return self._build(state)

    def _nearest_pillars(self, person_xy: FloatArray, person_yaw: float) -> FloatArray:
        hs = self._half_size
        feats: list[float] = []
        pillars = self._pillars
        if pillars.shape[0] > 0:
            dists = np.linalg.norm(pillars[:, :2] - person_xy, axis=1)
            order = np.argsort(dists)[:_N_PILLAR_SLOTS]
            for idx in order:
                p = pillars[idx]
                rel_body = _rotate(p[:2] - person_xy, -person_yaw)
                feats.extend([rel_body[0] / (2.0 * hs), rel_body[1] / (2.0 * hs), float(p[2]) / hs])
        while len(feats) < _N_PILLAR_SLOTS * _PILLAR_FEATS:
            feats.append(0.0)
        return np.array(feats, dtype=np.float64)

    def _build(self, state: WorldState) -> npt.NDArray[np.float32]:
        hs = self._half_size
        person_xy = np.array([state.person.x, state.person.y], dtype=np.float64)
        drone_xy = np.array([state.drone.x, state.drone.y], dtype=np.float64)
        own = np.array(
            [state.person.x / hs, state.person.y / hs, state.person.yaw / np.pi], dtype=np.float64
        )
        drone_rel_body = _rotate(drone_xy - person_xy, -state.person.yaw)
        drone_yaw_rel = _wrap_angle(state.drone.yaw - state.person.yaw) / np.pi
        drone_rel = np.array(
            [drone_rel_body[0] / (2.0 * hs), drone_rel_body[1] / (2.0 * hs), drone_yaw_rel],
            dtype=np.float64,
        )
        visible = np.array([1.0 if state.person_visible else 0.0], dtype=np.float64)
        illum = np.array([state.illumination_at_person], dtype=np.float64)
        pillars = self._nearest_pillars(person_xy, state.person.yaw)
        steps_visible = np.array(
            [min(self._steps_visible, EPISODE_STEPS) / EPISODE_STEPS], dtype=np.float64
        )
        vec = np.concatenate([own, drone_rel, visible, illum, pillars, steps_visible]).astype(np.float32)
        assert vec.shape == (self.size(),)
        return vec


@dataclass(frozen=True)
class PreyRewardConfig:
    """SPEC_prey.md `Prey action + reward`; YAML-loadable via `train_prey.py`'s `reward:` block."""

    visible_penalty: float = 1.0
    action_l2: float = 0.01
    transition_bonus: float = 0.5


def prey_reward(
    state: WorldState,
    action: FloatArray,
    was_visible: bool,
    cfg: PreyRewardConfig,
    seen: bool | None = None,
) -> float:
    """`-visible_penalty` if seen, minus an L2 action penalty, plus `transition_bonus` exactly
    once on the seen->unseen transition (SPEC_prey.md: the evader is paid for being unseen).

    `seen` defaults to geometric visibility; the training wrapper passes the frozen tracker's
    lock so the evader optimises the quantity the trial reports (deviation row 20). Paying on
    `state.person_visible` is the proxy divergence already measured for the hunter in row 8.
    """
    if seen is None:
        seen = state.person_visible
    r = -cfg.visible_penalty if seen else 0.0
    r -= cfg.action_l2 * float(np.dot(action, action))
    if was_visible and not seen:
        r += cfg.transition_bonus
    return r


class LearnedPrey:
    """Loads an SB3 PPO policy trained by `train_prey.py` (SPEC_prey.md `LearnedPrey`). Builds
    its own frame-stacked `PreyObsBuilder` vector, lazily initialised on the first `act()` call
    (the `Prey.reset()` protocol carries no `WorldState` to seed it with)."""

    def __init__(self, path: str) -> None:
        self.name = "learned_prey"
        self.path = path
        self._model: object | None = None
        self._obs_builder: PreyObsBuilder | None = None
        self._difficulty: Difficulty | None = None
        self._n_stack = 4
        self._stack: deque[npt.NDArray[np.float32]] | None = None

    def reset(self, layout: ArenaLayout, difficulty: Difficulty, seed: int) -> None:
        from stable_baselines3 import PPO

        self._model = PPO.load(self.path, device="cpu")
        sidecar = Path(str(self.path) + ".prey.json")
        obs_convention = "prey_v1"
        if sidecar.exists():
            obs_convention = str(json.loads(sidecar.read_text(encoding="utf-8")).get("obs", "prey_v1"))
        assert obs_convention == "prey_v1", f"unknown prey obs convention {obs_convention!r}"
        obs_space = self._model.observation_space
        assert obs_space is not None and obs_space.shape is not None
        dim = int(obs_space.shape[0])
        self._n_stack = max(1, dim // PreyObsBuilder.size())
        self._obs_builder = PreyObsBuilder(layout)
        self._difficulty = difficulty
        self._stack = None
        logger.info("LearnedPrey loaded %s on device=cpu (n_stack=%d)", self.path, self._n_stack)

    def act(
        self, state: WorldState, illumination: Callable[[npt.ArrayLike], float]
    ) -> tuple[float, float]:
        assert self._model is not None and self._obs_builder is not None and self._difficulty is not None
        if self._stack is None:
            obs_vec = self._obs_builder.reset(state)
            self._stack = deque(maxlen=self._n_stack)
            for _ in range(self._n_stack - 1):
                self._stack.append(np.zeros_like(obs_vec))
            self._stack.append(obs_vec)
        else:
            obs_vec = self._obs_builder.step(state)
            self._stack.append(obs_vec)

        vector = np.concatenate(list(self._stack)).astype(np.float64)
        action, _ = self._model.predict(vector, deterministic=True)  # type: ignore[attr-defined]
        speed = person_max_speed(self._difficulty)
        clipped = np.clip(np.asarray(action, dtype=np.float64).reshape(2), -1.0, 1.0)
        vx, vy = clipped[0] * speed, clipped[1] * speed
        return float(vx), float(vy)
