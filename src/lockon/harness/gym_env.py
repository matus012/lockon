"""`LockonGym` — Gymnasium wrapper used for PPO training (SPEC.md `gym_env.py`). State only,
never touches sensor/render (project.md §2); the hunter is the RL agent driving `step(action)`.
"""

from __future__ import annotations

import logging
from typing import Any

import gymnasium as gym
import numpy as np
import numpy.typing as npt

from lockon.core.schemas import EPISODE_STEPS, AgentAction, AgentObs, Difficulty
from lockon.env.env import Env
from lockon.policy.base import Prey
from lockon.policy.features import ObsBuilder, RewardConfig, reward

logger = logging.getLogger(__name__)

ObsType = npt.NDArray[np.float32]
ActType = npt.NDArray[np.float32]


class LockonGym(gym.Env[ObsType, ActType]):
    """Hunter-only training env: `prey` is a fixed scripted opponent (SPEC.md)."""

    metadata: dict[str, Any] = {"render_modes": []}  # noqa: RUF012 (matches gym.Env's own declaration)

    def __init__(self, difficulty: Difficulty, seed: int, prey: Prey, reward: RewardConfig) -> None:
        super().__init__()
        self.difficulty = difficulty
        self._seed = seed
        self._prey = prey
        self._reward_cfg = reward
        self._env = Env(difficulty, seed)
        self._obs_builder = ObsBuilder(self._env.layout)
        self._steps_since_seen = 0
        self._t = 0
        self.observation_space = gym.spaces.Box(
            low=-np.inf, high=np.inf, shape=(AgentObs.size(),), dtype=np.float32
        )
        self.action_space = gym.spaces.Box(
            low=-1.0, high=1.0, shape=(AgentAction.size(),), dtype=np.float32
        )

    def reset(
        self, *, seed: int | None = None, options: dict[str, Any] | None = None
    ) -> tuple[ObsType, dict[str, Any]]:
        super().reset(seed=seed)
        self._env.difficulty = self.difficulty
        # `Env(difficulty, seed)` already reset deterministically in __init__; an explicit `seed`
        # here rebuilds the layout for a new episode, `None` continues the env's own rng stream
        # (gymnasium allows `reset(seed=None)`; SB3/VecEnv reseed explicitly at episode starts).
        state = self._env.reset(seed)
        self._prey.reset(self._env.layout, self.difficulty, seed if seed is not None else self._seed)
        self._obs_builder = ObsBuilder(self._env.layout)
        obs = self._obs_builder.reset(state)
        self._steps_since_seen = 0
        self._t = 0
        info: dict[str, Any] = {"visible": state.person_visible, "steps_since_seen": 0}
        return obs.vector(), info

    def step(self, action: ActType) -> tuple[ObsType, float, bool, bool, dict[str, Any]]:
        agent_action = AgentAction.from_array(action)
        pv = self._prey.act(self._env.state(), self._env.illumination)
        state = self._env.step(agent_action, pv)
        obs = self._obs_builder.step(state)
        self._steps_since_seen = 0 if state.person_visible else self._steps_since_seen + 1
        r = reward(state, agent_action, self._steps_since_seen, self._reward_cfg)
        self._t += 1
        terminated = self._t >= EPISODE_STEPS
        truncated = False
        info: dict[str, Any] = {
            "visible": state.person_visible,
            "steps_since_seen": self._steps_since_seen,
        }
        return obs.vector(), r, terminated, truncated, info

    def set_difficulty(self, difficulty: Difficulty) -> None:
        """Curriculum hook, called via `VecEnv.env_method` (SPEC.md); applies at the next reset."""
        self.difficulty = difficulty
