"""PPO config + builder (SPEC.md `PPO config`). State-PPO is CPU by design (plan.md §8) — always
asserted and logged, never GPU even when CUDA is available.
"""

from __future__ import annotations

import dataclasses
import logging
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

import gymnasium as gym
import yaml
from stable_baselines3 import PPO
from stable_baselines3.common.vec_env import VecEnv

logger = logging.getLogger(__name__)

DEVICE = "cpu"  # plan.md §8: state-PPO trains on CPU by design.


@dataclass(frozen=True)
class PPOConfig:
    """SB3 PPO hyperparameters (SPEC.md `PPO config`) plus the harness scheduling knobs.

    Extra YAML keys not modelled as fields (`name`, `seed`, `curriculum`, `reward`, `wall_hours`,
    `eval_episodes`, ...) are tolerated and exposed via `extra`.
    """

    seed: int = 0
    total_steps: int = 6_000_000
    n_envs: int = 8
    frame_stack: int = 4
    n_steps: int = 1024
    batch_size: int = 256
    n_epochs: int = 10
    gamma: float = 0.99
    gae_lambda: float = 0.95
    learning_rate: float = 3e-4
    ent_coef: float = 0.0
    log_std_init: float = -1.6
    clip_range: float = 0.2
    net_arch: tuple[int, ...] = (128, 128)
    checkpoint_every: int = 100_000
    eval_every: int = 200_000
    extra: dict[str, Any] = field(default_factory=dict)

    @classmethod
    def from_yaml(cls, path: str | Path) -> PPOConfig:
        with open(path, encoding="utf-8") as f:
            raw = yaml.safe_load(f)
        known = {f.name for f in dataclasses.fields(cls) if f.name != "extra"}
        kwargs: dict[str, Any] = {}
        extra: dict[str, Any] = {}
        for key, value in raw.items():
            if key == "net_arch":
                kwargs[key] = tuple(value)
            elif key in known:
                kwargs[key] = value
            else:
                extra[key] = value
        return cls(**kwargs, extra=extra)


def make_ppo(env: gym.Env[Any, Any] | VecEnv, seed: int, cfg: PPOConfig) -> PPO:
    """Build an SB3 PPO with MlpPolicy [128, 128] on CPU (SPEC.md `PPO config`)."""
    model = PPO(
        "MlpPolicy",
        env,
        learning_rate=cfg.learning_rate,
        n_steps=cfg.n_steps,
        batch_size=cfg.batch_size,
        n_epochs=cfg.n_epochs,
        gamma=cfg.gamma,
        gae_lambda=cfg.gae_lambda,
        ent_coef=cfg.ent_coef,
        clip_range=cfg.clip_range,
        policy_kwargs={"net_arch": list(cfg.net_arch), "log_std_init": cfg.log_std_init},
        seed=seed,
        device=DEVICE,
        verbose=0,
    )
    assert str(model.device) == DEVICE, f"PPO must run on cpu by design, got {model.device}"
    logger.info("PPO built: device=%s net_arch=%s", model.device, cfg.net_arch)
    return model
