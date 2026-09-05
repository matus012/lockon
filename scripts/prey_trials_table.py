"""Combine the three learned-prey trials (the README's prey table) from the published eval JSONs.

    uv run python scripts/prey_trials_table.py

attempt 1 = visibility reward (deviation row 20 corrected it); seeds 1-2 = lock reward. Every
input lives under reports/evals/, so the table can be recomputed from the repo alone.
"""

from __future__ import annotations

import json
import sys
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parents[1]
HUNTERS = {"static": "static camera", "scripted": "scripted hunter", "best": "PPO hunter"}
TRIALS = {
    "attempt 1 (visibility reward)": "reports/evals/eval_{h}_vs_best.json",
    "seed 1 (lock reward)": "reports/evals/seed1_{h}_vs_learned.json",
    "seed 2 (lock reward)": "reports/evals/seed2_{h}_vs_learned.json",
}


def _load(rel: str, hunter: str) -> np.ndarray | None:
    f = ROOT / rel
    if not f.exists():
        return None
    data = json.loads(f.read_text(encoding="utf-8"))["results"]
    key = next((k for k in data if k.endswith(".zip") or k == hunter), None)
    if key is None:
        return None
    return np.array([e["retention"] for e in data[key]["per_episode"]], dtype=np.float64)


def _ci(d: np.ndarray) -> tuple[float, float, float]:
    se = d.std(ddof=1) / np.sqrt(len(d)) * 100
    return d.mean() * 100, d.mean() * 100 - 1.96 * se, d.mean() * 100 + 1.96 * se


def main() -> int:
    maybe_base = {h: _load(f"reports/evals/eval_{h}_vs_scripted.json", h) for h in HUNTERS}
    if any(v is None for v in maybe_base.values()):
        print("missing baseline evals under reports/evals/", file=sys.stderr)
        return 1
    base: dict[str, np.ndarray] = {h: v for h, v in maybe_base.items() if v is not None}
    trials = {name: {h: _load(pat.format(h=h), h) for h in HUNTERS} for name, pat in TRIALS.items()}

    header = "  ".join(f"{label:>16s}" for label in HUNTERS.values())
    print(f"{'evader':32s} {header}")
    print(f"{'scripted prey (baseline)':32s} " + "  ".join(f"{base[h].mean() * 100:16.1f}" for h in HUNTERS))
    for name, arms in trials.items():
        cells = "  ".join(
            f"{a.mean() * 100:16.1f}" if (a := arms[h]) is not None else f"{'--':>16s}" for h in HUNTERS
        )
        print(f"{name:32s} {cells}")

    print("\ndegradation vs the scripted prey, percentage points [95 % CI]")
    for name, arms in trials.items():
        if any(v is None for v in arms.values()):
            continue
        deltas = {h: arms[h] - base[h] for h in HUNTERS}
        print(f"\n{name}")
        for h, label in HUNTERS.items():
            m, lo, hi = _ci(deltas[h])
            print(f"  {label:16s} {m:+6.1f} [{lo:+6.1f}, {hi:+6.1f}]")
        for other in ("best", "static"):
            m, lo, hi = _ci(deltas["scripted"] - deltas[other])
            print(f"  interaction scripted - {HUNTERS[other]:14s} {m:+6.1f} [{lo:+6.1f}, {hi:+6.1f}]")

    corrected = [n for n in TRIALS if "lock reward" in n and all(v is not None for v in trials[n].values())]
    if len(corrected) > 1:
        pooled: dict[str, np.ndarray] = {
            h: np.concatenate([a for n in corrected if (a := trials[n][h]) is not None]) for h in HUNTERS
        }
        rep = {h: np.concatenate([base[h]] * len(corrected)) for h in HUNTERS}
        deltas = {h: pooled[h] - rep[h] for h in HUNTERS}
        print(f"\npooled over the {len(corrected)} corrected seeds (n={len(pooled['static'])})")
        for h, label in HUNTERS.items():
            m, lo, hi = _ci(deltas[h])
            print(f"  {label:16s} {pooled[h].mean() * 100:5.1f}   delta {m:+6.1f} [{lo:+6.1f}, {hi:+6.1f}]")
        for other in ("best", "static"):
            m, lo, hi = _ci(deltas["scripted"] - deltas[other])
            print(f"  interaction scripted - {HUNTERS[other]:14s} {m:+6.1f} [{lo:+6.1f}, {hi:+6.1f}]")
    return 0


if __name__ == "__main__":
    sys.exit(main())
