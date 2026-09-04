"""Hunter implementations (SPEC.md `Hunters`)."""

from __future__ import annotations

import logging
import math

import numpy as np

from lockon.core.schemas import (
    EPISODE_STEPS,
    N_RAYCASTS,
    AgentAction,
    AgentObs,
    ArenaLayout,
    FloatArray,
)

logger = logging.getLogger(__name__)

_RAY_STEP = 2.0 * math.pi / N_RAYCASTS


def _ray_index(angle: float) -> int:
    """Nearest raycast index for a body-frame direction angle (raycasts assumed evenly spaced
    starting at angle 0 = forward, increasing with yaw convention).
    """
    return round(angle / _RAY_STEP) % N_RAYCASTS


class StaticCamera:
    """The floor baseline (project.md §3): never moves."""

    def __init__(self) -> None:
        self.name = "static_camera"

    def reset(self, layout: ArenaLayout, seed: int) -> None:
        return None

    def act(self, obs: AgentObs) -> AgentAction:
        return AgentAction.zero()


class ScriptedHunter:
    """Visibility-greedy baseline / fallback hero (SPEC.md `Hunters`). No randomness."""

    def __init__(self, standoff_m: float = 6.0, gain_yaw: float = 2.0, gain_v: float = 1.0) -> None:
        self.name = "scripted_hunter"
        self.standoff_m = standoff_m
        self.gain_yaw = gain_yaw
        self.gain_v = gain_v
        self._half_size = 1.0

    def reset(self, layout: ArenaLayout, seed: int) -> None:
        self._half_size = layout.half_size

    def act(self, obs: AgentObs) -> AgentAction:
        hs = self._half_size
        # Target estimate in the drone body frame (SPEC.md): last_seen_rel is already
        # (target - drone) at last sighting, rotated into the CURRENT body frame.
        target_body = obs.last_seen_rel * (2.0 * hs)
        range_m = float(np.linalg.norm(target_body))
        bearing = math.atan2(target_body[1], target_body[0]) if range_m > 1e-9 else 0.0

        lost_steps_val = self._lost_steps(obs)
        yaw_rate = float(np.clip(self.gain_yaw * bearing, -1.0, 1.0))

        if lost_steps_val == 0:
            vx, vy = self._standoff_action(target_body, range_m, bearing)
        else:
            vx, vy = self._lost_action(target_body, range_m, bearing, obs.raycasts, lost_steps_val)

        return AgentAction.from_array(np.array([vx, vy, yaw_rate], dtype=np.float64))

    @staticmethod
    def _lost_steps(obs: AgentObs) -> int:
        return round(obs.time_since_seen * EPISODE_STEPS)

    def _standoff_action(
        self, target_body: FloatArray, range_m: float, bearing: float
    ) -> tuple[float, float]:
        vx = float(np.clip(self.gain_v * (range_m - self.standoff_m) / self.standoff_m, -1.0, 1.0))
        vy = float(np.clip(self.gain_v * math.sin(bearing), -1.0, 1.0))
        return vx, vy

    def _lost_action(
        self,
        target_body: FloatArray,
        range_m: float,
        bearing: float,
        raycasts: FloatArray,
        lost_steps: int,
    ) -> tuple[float, float]:
        direction = target_body / range_m if range_m > 1e-9 else np.zeros(2, dtype=np.float64)

        if lost_steps > 30:
            # Orbit the last-seen point at radius 4 m.
            tangent = np.array([-direction[1], direction[0]], dtype=np.float64)
            radial_err = range_m - 4.0
            radial = np.clip(radial_err / 4.0, -1.0, 1.0) * direction
            vec = self.gain_v * (tangent + radial)
            return float(np.clip(vec[0], -1.0, 1.0)), float(np.clip(vec[1], -1.0, 1.0))

        idx = _ray_index(bearing)
        if raycasts[idx] < 0.08:
            left = raycasts[(idx - 1) % N_RAYCASTS]
            right = raycasts[(idx + 1) % N_RAYCASTS]
            idx2 = (idx - 1) % N_RAYCASTS if left > right else (idx + 1) % N_RAYCASTS
            angle2 = idx2 * _RAY_STEP
            vx = float(np.clip(self.gain_v * math.cos(angle2), -1.0, 1.0))
            vy = float(np.clip(self.gain_v * math.sin(angle2), -1.0, 1.0))
            return vx, vy

        vx = float(np.clip(self.gain_v * direction[0], -1.0, 1.0))
        vy = float(np.clip(self.gain_v * direction[1], -1.0, 1.0))
        return vx, vy


class PPOHunter:
    """Loads an SB3 PPO policy (`ppo.py`). Requires the frame-stacked vector observation built
    by the harness's Gymnasium wrapper — `act(obs)` is undefined for a single-step `AgentObs`.
    """

    def __init__(self, path: str) -> None:
        self.name = "ppo_hunter"
        self.path = path
        self._model: object | None = None

    def reset(self, layout: ArenaLayout, seed: int) -> None:
        from stable_baselines3 import PPO

        self._model = PPO.load(self.path, device="cpu")
        logger.info("PPOHunter loaded %s on device=cpu", self.path)

    def act(self, obs: AgentObs) -> AgentAction:
        raise NotImplementedError(
            "PPOHunter.act requires the harness's frame-stacked vector; use act_vector(vector)"
        )

    @property
    def n_stack(self) -> int:
        """Frame-stack depth implied by the loaded model's observation space (never a default)."""
        model = self._model
        if model is None:
            raise RuntimeError("PPOHunter.reset() must load the model first")
        dim = int(model.observation_space.shape[0])  # type: ignore[attr-defined]
        return max(1, dim // AgentObs.size())

    def act_vector(self, vector: FloatArray) -> AgentAction:
        assert self._model is not None, "PPOHunter.reset() must be called before act_vector()"
        action, _ = self._model.predict(vector, deterministic=True)  # type: ignore[attr-defined]
        return AgentAction.from_array(action)
