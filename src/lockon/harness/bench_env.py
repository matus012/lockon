"""Tiny CLI: headless step-rate benchmark for `lockon.env.Env` (SPEC.md `Bench`).

    uv run python -m lockon.harness.bench_env --steps 2000 --min-sps 200
"""

from __future__ import annotations

import argparse
import logging
import sys
import time

import numpy as np

from lockon.core.schemas import AgentAction, Difficulty
from lockon.env.env import Env

logger = logging.getLogger(__name__)


def run(steps: int, min_sps: float, seed: int = 0) -> float:
    rng = np.random.default_rng(seed)
    env = Env(Difficulty(), seed=seed)
    max_speed = env.person_max_speed()

    start = time.perf_counter()
    for _ in range(steps):
        action = AgentAction.from_array(rng.uniform(-1.0, 1.0, size=3))
        heading = rng.uniform(-np.pi, np.pi)
        speed = rng.uniform(0.0, max_speed)
        person_velocity = (float(speed * np.cos(heading)), float(speed * np.sin(heading)))
        env.step(action, person_velocity)
    elapsed = time.perf_counter() - start

    sps = steps / elapsed if elapsed > 0 else float("inf")
    logger.info("steps=%d elapsed=%.3fs steps/s=%.1f (min %.1f)", steps, elapsed, sps, min_sps)
    print(f"steps={steps} elapsed={elapsed:.3f}s steps/s={sps:.1f} (min {min_sps:.1f})")
    return sps


def main(argv: list[str] | None = None) -> int:
    logging.basicConfig(level=logging.INFO)
    parser = argparse.ArgumentParser(description="lockon.env headless step-rate benchmark")
    parser.add_argument("--steps", type=int, default=2000)
    parser.add_argument("--min-sps", type=float, default=200.0)
    parser.add_argument("--seed", type=int, default=0)
    args = parser.parse_args(argv)

    sps = run(args.steps, args.min_sps, args.seed)
    return 0 if sps >= args.min_sps else 1


if __name__ == "__main__":
    sys.exit(main())
