"""Hunter / Prey protocols (SPEC.md `Interfaces`). Pure interfaces — no logic here."""

from __future__ import annotations

from collections.abc import Callable
from typing import Protocol

import numpy.typing as npt

from lockon.core.schemas import AgentAction, AgentObs, ArenaLayout, Difficulty, WorldState


class Hunter(Protocol):
    """Acts from a partial observation only (project.md §3: hunter is not privileged)."""

    name: str

    def reset(self, layout: ArenaLayout, seed: int) -> None: ...

    def act(self, obs: AgentObs) -> AgentAction: ...


class Prey(Protocol):
    """Scripted examiner. Privileged: reads full ground-truth `WorldState`, unlike the hunter."""

    name: str

    def reset(self, layout: ArenaLayout, difficulty: Difficulty, seed: int) -> None: ...

    def act(
        self, state: WorldState, illumination: Callable[[npt.ArrayLike], float]
    ) -> tuple[float, float]: ...
