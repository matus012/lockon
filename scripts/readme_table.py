"""Regenerate the README headline table from reports/eval_*.json (single source; review F1).

Prints a markdown table with mean, std, n, 95 % CI of the mean and the paired difference vs
static with its 95 % CI, so no number in the README exists without an artifact behind it.
"""

from __future__ import annotations

import json
import sys
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parents[1]
FILES = {
    "static": ("reports/eval_ppo_vs_static.json", "static"),
    "scripted hunter (visibility-greedy)": ("reports/eval_ppo_vs_scripted.json", "scripted"),
    "PPO hunter (best local checkpoint)": ("reports/eval_ppo_vs_static.json", "runs/ppo_local/best.zip"),
}


def _per_episode(path: str, key: str) -> np.ndarray:
    d = json.loads((ROOT / path).read_text(encoding="utf-8"))["results"][key]
    eps = d.get("episodes") or d.get("per_episode") or d.get("retentions")
    if eps is None:
        raise SystemExit(f"{path}: no per-episode list under {key}; keys={list(d)}")
    vals = [e["retention"] if isinstance(e, dict) else float(e) for e in eps]
    return np.array(vals, dtype=np.float64)


def main() -> int:
    arms = {name: _per_episode(*spec) for name, spec in FILES.items()}
    static = arms["static"]
    n = len(static)
    print(f"| camera policy | lock retention % (mean) | ± std | 95 % CI of mean | Δ vs static (95 % CI) | time-to-reacquire |")
    print("|---|---|---|---|---|---|")
    for name, v in arms.items():
        assert len(v) == n
        se = v.std(ddof=1) / np.sqrt(n)
        d = v - static
        dse = d.std(ddof=1) / np.sqrt(n)
        delta = "—" if name == "static" else f"{d.mean()*100:+.1f} [{(d.mean()-1.96*dse)*100:+.1f}, {(d.mean()+1.96*dse)*100:+.1f}]"
        print(f"| {name} | {v.mean()*100:.1f} | {v.std(ddof=1)*100:.1f} | [{(v.mean()-1.96*se)*100:.1f}, {(v.mean()+1.96*se)*100:.1f}] | {delta} | see JSON |")
    print(f"\nn = {n} episodes per arm, seeds 1000–{1000+n-1} (held out from model selection), mid difficulty, tracker noise dial 0.3, per-episode noise seed.")
    return 0


if __name__ == "__main__":
    sys.exit(main())
