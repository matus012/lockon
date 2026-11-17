"""Regenerate the README headline table from a single eval JSON (one seed base, all arms).

Every number in the README traces here (review F1). Prints mean, std, the 95 % CI of the mean,
the paired difference vs the static floor with its CI, and time-to-reacquire.

    uv run python scripts/readme_table.py [--json reports/eval_n80_ppo_vs_scripted.json]
"""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path
from typing import Any

import numpy as np

ROOT = Path(__file__).resolve().parents[1]
LABELS = {
    "static": "static camera (floor)",
    "scripted": "scripted hunter (visibility-greedy)",
}


def _label(key: str) -> str:
    return LABELS.get(key, "PPO hunter (best local checkpoint)" if key.endswith(".zip") else key)


def _arms(path: Path) -> dict[str, np.ndarray]:
    results: dict[str, Any] = json.loads(path.read_text(encoding="utf-8"))["results"]
    return {k: np.array([e["retention"] for e in v["per_episode"]], dtype=np.float64) for k, v in results.items()}


def _ttr(path: Path) -> dict[str, float]:
    results: dict[str, Any] = json.loads(path.read_text(encoding="utf-8"))["results"]
    return {k: float(v.get("ttr_mean", float("nan"))) for k, v in results.items()}


def main(argv: list[str] | None = None) -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--json", type=Path, default=ROOT / "reports" / "eval_n80_ppo_vs_scripted.json")
    ap.add_argument("--seed-base", type=int, default=1000)
    args = ap.parse_args(argv)
    arms = _arms(args.json)
    ttr = _ttr(args.json)
    if "static" not in arms:
        raise SystemExit(f"{args.json}: no 'static' arm to difference against")
    static = arms["static"]
    n = len(static)
    order = ["static"] + [k for k in arms if k != "static" and not k.endswith(".zip")] + [k for k in arms if k.endswith(".zip")]

    print("| camera policy | lock retention % | ± std | 95 % CI of mean | Δ vs static (95 % CI) | time-to-reacquire |")
    print("|---|---|---|---|---|---|")
    for key in order:
        v = arms[key]
        if len(v) != n:
            raise SystemExit(f"arm {key} has {len(v)} episodes, static has {n}")
        se = v.std(ddof=1) / np.sqrt(n)
        d = v - static
        dse = d.std(ddof=1) / np.sqrt(n)
        delta = "—" if key == "static" else (
            f"{d.mean()*100:+.1f} [{(d.mean()-1.96*dse)*100:+.1f}, {(d.mean()+1.96*dse)*100:+.1f}]")
        print(f"| {_label(key)} | {v.mean()*100:.1f} | {v.std(ddof=1)*100:.1f} | "
              f"[{(v.mean()-1.96*se)*100:.1f}, {(v.mean()+1.96*se)*100:.1f}] | {delta} | {ttr[key]:.2f} |")

    movers = [k for k in order if k != "static"]
    if len(movers) == 2:
        a, b = arms[movers[0]], arms[movers[1]]
        d = a - b
        dse = d.std(ddof=1) / np.sqrt(n)
        print(f"\n{_label(movers[0])} − {_label(movers[1])}: {d.mean()*100:+.1f} "
              f"[{(d.mean()-1.96*dse)*100:+.1f}, {(d.mean()+1.96*dse)*100:+.1f}] points")
    print(f"\nn = {n} episodes per arm, seeds {args.seed_base}–{args.seed_base+n-1} (held out from model "
          "selection), mid difficulty, tracker noise dial 0.3, per-episode noise seed. Source: "
          f"{args.json.relative_to(ROOT).as_posix()}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
