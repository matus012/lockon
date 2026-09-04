"""Observation builder + reward (SPEC.md `Observation + reward`). Single source, used by the
harness for both RL training and eval — must stay a pure function of `WorldState`.
"""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np

from lockon.core.schemas import (
    CHANNELS,
    EPISODE_STEPS,
    AgentAction,
    AgentObs,
    ArenaLayout,
    FloatArray,
    WorldState,
)


def _rotate(vec: FloatArray, angle: float) -> FloatArray:
    """Active rotation by `angle` radians, consistent with the world yaw convention (yaw=0 = +x,
    increasing yaw turns +x toward +y).
    """
    c, s = np.cos(angle), np.sin(angle)
    return np.array([vec[0] * c - vec[1] * s, vec[0] * s + vec[1] * c], dtype=np.float64)


class ObsBuilder:
    """Tracks the last sighting of the target and turns `WorldState` into `AgentObs`."""

    def __init__(self, layout: ArenaLayout) -> None:
        self._half_size = layout.half_size
        self._last_seen_world = np.zeros(2, dtype=np.float64)
        self._last_seen_illum = 0.0
        self._steps_since_seen = 0

    @property
    def steps_since_seen(self) -> int:
        """Consecutive steps without a sighting - the counter `reward()` consumes (single source)."""
        return self._steps_since_seen

    def reset(self, state: WorldState) -> AgentObs:
        # Episode starts visible: last_seen = current (SPEC.md).
        self._last_seen_world = np.array([state.person.x, state.person.y], dtype=np.float64)
        self._last_seen_illum = state.illumination_at_person
        self._steps_since_seen = 0
        return self._build(state)

    def step(self, state: WorldState) -> AgentObs:
        if state.person_visible:
            self._last_seen_world = np.array([state.person.x, state.person.y], dtype=np.float64)
            self._last_seen_illum = state.illumination_at_person
            self._steps_since_seen = 0
        else:
            self._steps_since_seen += 1
        return self._build(state)

    def _build(self, state: WorldState) -> AgentObs:
        hs = self._half_size
        own = np.array(
            [state.drone.x / hs, state.drone.y / hs, state.drone.yaw / np.pi], dtype=np.float64
        )
        drone_xy = np.array([state.drone.x, state.drone.y], dtype=np.float64)
        rel_world = self._last_seen_world - drone_xy
        rel_body = _rotate(rel_world, -state.drone.yaw)
        last_seen_rel = rel_body / (2.0 * hs)
        time_since_seen = min(self._steps_since_seen, EPISODE_STEPS) / EPISODE_STEPS
        channels_alive = np.array(
            [1.0 if state.channels_alive[c] else 0.0 for c in CHANNELS], dtype=np.float64
        )
        return AgentObs(
            own=own,
            last_seen_rel=last_seen_rel,
            time_since_seen=time_since_seen,
            raycasts=state.raycasts,
            light=self._last_seen_illum,
            channels_alive=channels_alive,
        )


@dataclass(frozen=True)
class RewardConfig:
    visible: float = 1.0
    action_l2: float = 0.01
    lost_penalty: float = 10.0
    lost_after: int = 20


def reward(
    state: WorldState,
    action: AgentAction,
    steps_since_seen: int,
    cfg: RewardConfig,
    visible: bool | None = None,
) -> float:
    """+visible if visible, minus an L2 action penalty, minus `lost_penalty` exactly once the
    step `steps_since_seen == lost_after` fires (project.md §3: "big on lock lost > K steps").

    `visible` defaults to state visibility; the training wrapper passes the frozen tracker's
    lock instead (context.md D13): a policy rewarded on raw visibility learned to strafe/yaw so
    hard that the person's image box jumped ~8 px/step and ByteTrack switched ids 5.6x per
    episode — reward up, retention down. Lock held IS the headline metric, so it is the reward.
    """
    if visible is None:
        visible = state.person_visible
    r = cfg.visible if visible else 0.0
    a = action.as_array()
    r -= cfg.action_l2 * float(np.dot(a, a))
    if steps_since_seen == cfg.lost_after:
        r -= cfg.lost_penalty
    return r
