"""Measure the two before/after pairs the showcase reel shows, so their captions quote a number.

    uv run python scripts/showcase_pair_stats.py     # -> reports/evals/showcase_pairs.json

A side-by-side clip is one episode and one episode proves nothing. Both pairs are therefore run
here over the pre-registered HELD-OUT verdict seeds (1000+, plan §9 / the same block the n=80
headline uses), paired per seed, and reported as a mean difference with a 95 % CI:

  hunter pair  untrained PPO hunter  vs  trained PPO hunter   (evader: scripted)
  evader pair  random-walk evader    vs  trained evader       (hunter: scripted)

Whatever comes out is what the showcase README says. This script exists precisely so the reel
cannot imply a difference the seeds do not support.
"""

from __future__ import annotations

import argparse
import json
import logging
import sys
from pathlib import Path

import numpy as np

from lockon.core.schemas import Difficulty
from lockon.harness.episode import resolve_hunter, run_episode
from lockon.harness.eval import resolve_prey

logger = logging.getLogger(__name__)

ROOT = Path(__file__).resolve().parents[1]
SEED_BASE = 1000  # held-out verdict block (selection seeds are 0-19; never reuse them here)
TRAINED_HUNTER = "models/ppo_hunter.zip"
TRAINED_PREY = "models/learned_prey_seed0.zip"
UNTRAINED_HUNTER = "runs/showcase/untrained_hunter.zip"


def _arm(hunter: str, prey: str, seeds: list[int], difficulty: Difficulty) -> np.ndarray:
    out = []
    for s in seeds:
        res = run_episode(difficulty, s, resolve_hunter(hunter), resolve_prey(prey))
        out.append(res.retention.retention)
        logger.info("  seed %d  %s vs %s  %.3f", s, hunter, prey, res.retention.retention)
    return np.array(out, dtype=np.float64)


def _paired(a: np.ndarray, b: np.ndarray) -> dict[str, float]:
    """a - b, paired per seed, in percentage points."""
    d = (a - b) * 100.0
    se = d.std(ddof=1) / np.sqrt(len(d))
    return {
        "a_mean_pct": float(a.mean() * 100),
        "b_mean_pct": float(b.mean() * 100),
        "delta_pct": float(d.mean()),
        "ci_lo": float(d.mean() - 1.96 * se),
        "ci_hi": float(d.mean() + 1.96 * se),
        "n": len(d),
    }


def main(argv: list[str] | None = None) -> int:
    logging.basicConfig(level=logging.INFO, format="%(message)s")
    p = argparse.ArgumentParser(description="Paired stats for the showcase before/after pairs.")
    p.add_argument("--n", type=int, default=20)
    p.add_argument("--out", type=Path, default=ROOT / "reports/evals/showcase_pairs.json")
    args = p.parse_args(argv)

    seeds = list(range(SEED_BASE, SEED_BASE + args.n))
    difficulty = Difficulty()  # mid: every dial 0.5 (plan §9 D5)
    trained_prey = str(ROOT / TRAINED_PREY)

    if not (ROOT / UNTRAINED_HUNTER).exists():
        print(f"missing {UNTRAINED_HUNTER}; run scripts/make_untrained_hunter.py first", file=sys.stderr)
        return 1

    logger.info("hunter pair: untrained vs trained PPO (evader scripted), n=%d", args.n)
    untrained = _arm(str(ROOT / UNTRAINED_HUNTER), "scripted", seeds, difficulty)
    trained = _arm(str(ROOT / TRAINED_HUNTER), "scripted", seeds, difficulty)

    logger.info("evader pair: random-walk vs trained evader (hunter scripted), n=%d", args.n)
    vs_random = _arm("scripted", "random", seeds, difficulty)
    vs_trained = _arm("scripted", trained_prey, seeds, difficulty)

    payload = {
        "seeds": seeds,
        "difficulty": vars(difficulty),
        "hunter_pair": {
            "a": "untrained PPO hunter",
            "b": "trained PPO hunter",
            **_paired(untrained, trained),
        },
        "evader_pair": {
            "a": "scripted hunter vs random-walk evader",
            "b": "scripted hunter vs trained evader",
            **_paired(vs_random, vs_trained),
        },
    }
    args.out.parent.mkdir(parents=True, exist_ok=True)
    args.out.write_text(json.dumps(payload, indent=2), encoding="utf-8")
    print(json.dumps(payload, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
