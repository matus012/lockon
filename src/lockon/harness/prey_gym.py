"""`PreyGym` — Gymnasium wrapper used to train the learned prey (SPEC_prey.md `PreyGym`).
Composition root: state only, mirrors `gym_env.LockonGym` but with the roles reversed — the
prey is the RL agent driving `step(action)` and the hunter is FROZEN, driven exactly as
`episode.run_episode` drives it (same `ObsBuilder` + tracker-lock convention, same frame-stack
helpers, so a checkpoint trained here and one evaluated in `run_episode` see identical hunters).
"""

from __future__ import annotations

import logging
from collections import deque
from typing import Any

import gymnasium as gym
import numpy as np
import numpy.typing as npt

from lockon.core.schemas import EPISODE_STEPS, AgentObs, Difficulty, person_max_speed
from lockon.env.env import Env
from lockon.harness.episode import (
    EVAL_NOISE,
    _gt_detections,
    hunter_action,
    hunter_obs_mode,
    init_hunter_stack,
)
from lockon.policy.base import Hunter
from lockon.policy.features import ObsBuilder
from lockon.policy.prey_learned import PreyObsBuilder, PreyRewardConfig, prey_reward
from lockon.track import LockTracker, NoiseConfig, NoiseInjector

logger = logging.getLogger(__name__)

ObsType = npt.NDArray[np.float32]
ActType = npt.NDArray[np.float32]


class PreyGym(gym.Env[ObsType, ActType]):
    """Prey-only training env: `hunter` is a fixed, frozen opponent (SPEC_prey.md)."""

    metadata: dict[str, Any] = {"render_modes": []}  # noqa: RUF012 (matches gym.Env's own declaration)

    def __init__(
        self,
        difficulty: Difficulty,
        seed: int,
        hunter: Hunter,
        hunter_frame_stack: int,
        reward: PreyRewardConfig,
    ) -> None:
        super().__init__()
        self.difficulty = difficulty
        self._seed = seed
        self._hunter = hunter
        self._hunter_n_stack = hunter_frame_stack
        self._reward_cfg = reward
        self._env = Env(difficulty, seed)
        self._noise: NoiseInjector | None = None
        self._tracker: LockTracker | None = None
        self._hunter_obs_builder = ObsBuilder(self._env.layout)
        self._prey_obs_builder = PreyObsBuilder(self._env.layout)
        self._obs_mode = "lock"
        self._hunter_obs: AgentObs | None = None
        self._stack: deque[npt.NDArray[np.float32]] | None = None
        self._prev_visible = False
        self._t = 0
        self.observation_space = gym.spaces.Box(
            low=-np.inf, high=np.inf, shape=(PreyObsBuilder.size(),), dtype=np.float32
        )
        self.action_space = gym.spaces.Box(low=-1.0, high=1.0, shape=(2,), dtype=np.float32)

    def reset(
        self, *, seed: int | None = None, options: dict[str, Any] | None = None
    ) -> tuple[ObsType, dict[str, Any]]:
        super().reset(seed=seed)
        self._env.difficulty = self.difficulty
        # Fresh layout + hunter seed every episode (same rule as `LockonGym.reset`, gym_env.py).
        episode_seed = seed if seed is not None else int(self.np_random.integers(0, 2**31 - 1))
        state = self._env.reset(episode_seed)
        self._hunter.reset(self._env.layout, episode_seed)
        # frozen perception in the loop (D13): same noise dial as eval, per-episode noise seed
        self._noise = NoiseInjector(NoiseConfig(**{**vars(EVAL_NOISE), "seed": episode_seed}))
        self._tracker = LockTracker()
        self._tracker.reset()
        _, lock0 = self._tracker.update(self._noise(_gt_detections(state), 0), 0)

        self._hunter_obs_builder = ObsBuilder(self._env.layout)
        hunter_obs = self._hunter_obs_builder.reset(state)
        self._obs_mode = hunter_obs_mode(self._hunter)
        self._stack = init_hunter_stack(hunter_obs.vector(), self._hunter_n_stack)
        self._hunter_obs = hunter_obs

        self._prey_obs_builder = PreyObsBuilder(self._env.layout)
        prey_obs = self._prey_obs_builder.reset(state)
        self._prev_visible = lock0.locked
        self._t = 0
        info: dict[str, Any] = {"visible": state.person_visible}
        return prey_obs, info

    def step(self, action: ActType) -> tuple[ObsType, float, bool, bool, dict[str, Any]]:
        assert self._tracker is not None and self._noise is not None
        assert self._hunter_obs is not None and self._stack is not None
        prey_action = np.clip(np.asarray(action, dtype=np.float64).reshape(2), -1.0, 1.0)
        speed = person_max_speed(self.difficulty)
        pvx, pvy = float(prey_action[0]) * speed, float(prey_action[1]) * speed

        # hunter acts from the observation built at the END of the previous step (same order as
        # `episode.run_episode`'s loop: prey/hunter act on the CURRENT obs, then env.step).
        agent_action = hunter_action(self._hunter, self._hunter_obs, self._stack)
        state = self._env.step(agent_action, (pvx, pvy))
        self._t += 1

        _, lock = self._tracker.update(self._noise(_gt_detections(state), self._t), self._t)
        seen = lock.locked if self._obs_mode == "lock" else state.person_visible
        hunter_obs = self._hunter_obs_builder.step(state, seen=seen)
        self._stack.append(hunter_obs.vector())
        self._hunter_obs = hunter_obs

        was_visible = self._prev_visible
        prey_obs = self._prey_obs_builder.step(state)
        # the evader is paid for breaking the LOCK, not geometric visibility (row 20)
        r = prey_reward(state, prey_action, was_visible, self._reward_cfg, seen=lock.locked)
        self._prev_visible = lock.locked

        terminated = self._t >= EPISODE_STEPS
        truncated = False
        info: dict[str, Any] = {"visible": state.person_visible, "locked": lock.locked}
        return prey_obs, r, terminated, truncated, info

    def set_difficulty(self, difficulty: Difficulty) -> None:
        """Curriculum hook, called via `VecEnv.env_method` (SPEC_prey.md); applies at next reset."""
        self.difficulty = difficulty
