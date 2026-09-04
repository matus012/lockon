"""`evaluate()` + CLI (SPEC.md `eval.py`) — a preview of gate G4: does policy P beat policy Q on
mean lock retention, on the same seeds, under the frozen eval-time perception (`episode.EVAL_NOISE`)?

    uv run python -m lockon.harness.eval --policy scripted --vs static --n 20
"""

from __future__ import annotations

import argparse
import json
import logging
import math
import sys
from collections import Counter
from collections.abc import Iterable, Sequence
from pathlib import Path
from typing import Any

import numpy as np

from lockon.core.schemas import Difficulty
from lockon.harness.episode import resolve_hunter, run_episode
from lockon.policy.prey import ScriptedPrey

logger = logging.getLogger(__name__)

_DIFFICULTY_FIELDS = (
    "occluder_density",
    "darkness",
    "prey_speed",
    "prey_aggressiveness",
    "channel_dropout",
)


def evaluate(policy_name_or_path: str, difficulty: Difficulty, seeds: Iterable[int]) -> dict[str, Any]:
    """Run `policy_name_or_path` (`static` / `scripted` / a PPO zip path) over `seeds`, returning
    retention mean/std, mean time-to-reacquire, the per-episode list, and a loss-cause histogram.
    """
    retentions: list[float] = []
    ttrs: list[float] = []
    per_episode: list[dict[str, Any]] = []
    causes: Counter[str] = Counter()

    for seed in seeds:
        hunter = resolve_hunter(policy_name_or_path)
        result = run_episode(difficulty, seed, hunter, ScriptedPrey())
        retentions.append(result.retention.retention)
        if not math.isnan(result.retention.time_to_reacquire):
            ttrs.append(result.retention.time_to_reacquire)
        causes.update(result.loss_causes)
        per_episode.append(
            {
                "seed": seed,
                "retention": result.retention.retention,
                "time_to_reacquire": result.retention.time_to_reacquire,
                "n_loss_events": result.retention.n_loss_events,
                "last_loss_censored": result.retention.last_loss_censored,
            }
        )

    return {
        "policy": policy_name_or_path,
        "retention_mean": float(np.mean(retentions)) if retentions else float("nan"),
        "retention_std": float(np.std(retentions)) if retentions else float("nan"),
        "ttr_mean": float(np.mean(ttrs)) if ttrs else float("nan"),
        "per_episode": per_episode,
        "loss_cause_histogram": dict(causes),
    }


def _parse_difficulty(tokens: Sequence[str]) -> Difficulty:
    if len(tokens) == 1 and tokens[0] == "mid":
        return Difficulty()
    if len(tokens) == 5:
        values = [float(x) for x in tokens]
        return Difficulty(**dict(zip(_DIFFICULTY_FIELDS, values, strict=True)))
    raise ValueError(f"--difficulty expects 'mid' or 5 floats, got {tokens!r}")


def _print_table(rows: list[dict[str, Any]]) -> None:
    header = f"{'policy':<16}{'retention_mean':>16}{'retention_std':>16}{'ttr_mean':>12}{'n':>6}"
    print(header)
    print("-" * len(header))
    for row in rows:
        print(
            f"{row['policy']:<16}{row['retention_mean']:>16.4f}{row['retention_std']:>16.4f}"
            f"{row['ttr_mean']:>12.2f}{len(row['per_episode']):>6}"
        )


def main(argv: list[str] | None = None) -> int:
    logging.basicConfig(level=logging.INFO)
    parser = argparse.ArgumentParser(description="lockon.harness eval: does P beat Q on retention?")
    parser.add_argument("--policy", required=True, help="P: static | scripted | path to a PPO zip")
    parser.add_argument("--vs", required=True, dest="vs_policy", help="Q: static | scripted | path")
    parser.add_argument("--n", type=int, default=20, help="episodes (seeds base..base+n-1)")
    parser.add_argument("--seed-base", type=int, default=0, help="first seed; gates use 1000 (held out from training-time model selection, which used 0..19)")
    parser.add_argument("--difficulty", nargs="+", default=["mid"])
    parser.add_argument("--json", type=str, default=None, help="write full results here")
    args = parser.parse_args(argv)

    difficulty = _parse_difficulty(args.difficulty)
    seeds = list(range(args.seed_base, args.seed_base + args.n))

    # Union of {P, Q, static, scripted} so the table always carries both baselines (SPEC.md: "3-row
    # table (static / scripted / policy where applicable)") alongside whatever P/Q were requested.
    names = list(dict.fromkeys([args.policy, args.vs_policy, "static", "scripted"]))
    results = {name: evaluate(name, difficulty, seeds) for name in names}

    _print_table([results[name] for name in names])

    if args.json is not None:
        out_path = Path(args.json)
        out_path.parent.mkdir(parents=True, exist_ok=True)
        out_path.write_text(json.dumps({"P": args.policy, "Q": args.vs_policy, "results": results}, indent=2))
        logger.info("wrote %s", out_path)

    p_mean = results[args.policy]["retention_mean"]
    q_mean = results[args.vs_policy]["retention_mean"]
    beats = p_mean > q_mean
    print(f"\n{args.policy} beats {args.vs_policy}: {beats} ({p_mean:.4f} vs {q_mean:.4f})")
    return 0 if beats else 1


if __name__ == "__main__":
    sys.exit(main())
