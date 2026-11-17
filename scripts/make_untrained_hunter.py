"""Untrained PPO hunter checkpoint for the "untrained vs trained" showcase (render-only demo).

This is NOT training: it builds the exact same SB3 PPO architecture that
`lockon.harness.train` produces (same `build_vec_env`, same `PPOConfig` defaults/net_arch,
same `frame_stack`) via `configs/ppo_local.yaml`, seeds it, and saves the freshly
initialised weights at 0 timesteps -- no `model.learn()` call. The obs-mode sidecar is
written with the same convention as `train.py`'s `RetentionEvalCallback`
(`<path>.obs.json` -> `{"obs_seen": "lock"}`), since `PPOHunter` /
`lockon.harness.episode.hunter_obs_mode` read it.

    uv run python scripts/make_untrained_hunter.py [--out runs/showcase/untrained_hunter.zip] \\
        [--config configs/ppo_local.yaml] [--seed 0]
"""

from __future__ import annotations

import argparse
import logging
import sys
from pathlib import Path

from lockon.harness.train import build_vec_env
from lockon.policy.ppo import PPOConfig, make_ppo

logger = logging.getLogger(__name__)


def make_untrained_hunter(config_path: str, out: str, seed: int) -> Path:
    cfg = PPOConfig.from_yaml(config_path)
    curriculum = cfg.extra.get("curriculum", {})
    start_dial = float(curriculum.get("start_dial", 0.3))
    overrides = {k: float(v) for k, v in dict(cfg.extra.get("difficulty", {}) or {}).items()}

    env = build_vec_env(cfg, seed, start_dial, overrides)
    model = make_ppo(env, seed, cfg)
    assert model.num_timesteps == 0, f"expected 0 timesteps (no learn() call), got {model.num_timesteps}"

    out_path = Path(out)
    out_path.parent.mkdir(parents=True, exist_ok=True)
    model.save(str(out_path))
    Path(str(out_path) + ".obs.json").write_text('{"obs_seen": "lock"}', encoding="utf-8")
    env.close()

    logger.info(
        "untrained hunter saved: %s (obs_seen=lock, timesteps=%d, seed=%d, net_arch=%s, frame_stack=%d)",
        out_path, model.num_timesteps, seed, cfg.net_arch, cfg.frame_stack,
    )
    return out_path


def main(argv: list[str] | None = None) -> int:
    logging.basicConfig(level=logging.INFO, format="%(levelname)s %(message)s")
    parser = argparse.ArgumentParser(description="Save an untrained PPO hunter checkpoint (no learn() call)")
    parser.add_argument("--config", default="configs/ppo_local.yaml")
    parser.add_argument("--out", default="runs/showcase/untrained_hunter.zip")
    parser.add_argument("--seed", type=int, default=0)
    args = parser.parse_args(argv)

    path = make_untrained_hunter(args.config, args.out, args.seed)
    logger.info("wrote %s and %s.obs.json", path, path)
    return 0


if __name__ == "__main__":
    sys.exit(main())
