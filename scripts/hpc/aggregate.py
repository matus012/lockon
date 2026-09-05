"""Aggregate the PERUN CPU array (plan.md §5 step 9): rank units, pick the best by the
pre-registered rule, write the ablation table.

Rule (reports/hpc_launch_plan.md, deviation-log row 10): best unit = highest `retention.policy`
on the unit's own 20 eval episodes (seeds 0-19, mid difficulty) — selection seeds, never the
held-out 1000+ seeds, which the step-9 fork verdict uses. Ties → lower unit name (deterministic).

    uv run python scripts/hpc/aggregate.py [--root runs/hpc] [--out reports/hpc]
"""

from __future__ import annotations

import argparse
import json
import sys
from collections import defaultdict
from pathlib import Path
from typing import Any

import numpy as np

ROOT = Path(__file__).resolve().parents[2]


def load_results(root: Path) -> list[dict[str, Any]]:
    rows: list[dict[str, Any]] = []
    for p in sorted(root.glob("*/result.json")):
        if p.parent.name.startswith("_"):
            continue  # smoke / scratch dirs
        rows.append(json.loads(p.read_text(encoding="utf-8")))
    return rows


def select_best(rows: list[dict[str, Any]]) -> dict[str, Any] | None:
    if not rows:
        return None
    return min(rows, key=lambda r: (-float(r["retention"]["policy"]), r["unit"]))


def ablation(rows: list[dict[str, Any]]) -> list[dict[str, Any]]:
    """Mean ± std of policy retention over seeds per (reward_variant, arena_density)."""
    groups: dict[tuple[str, float], list[float]] = defaultdict(list)
    for r in rows:
        groups[(r["reward_variant"], float(r["arena_density"]))].append(float(r["retention"]["policy"]))
    out = []
    for (rv, d), vals in sorted(groups.items()):
        v = np.array(vals)
        out.append({"reward_variant": rv, "arena_density": d, "n": len(v), "mean": float(v.mean()),
                    "std": float(v.std(ddof=1)) if len(v) > 1 else 0.0})
    return out


def render_md(rows: list[dict[str, Any]], best: dict[str, Any] | None, abl: list[dict[str, Any]]) -> str:
    intro = (f"{len(rows)} of 45 units reported. Selection rule: highest policy retention on the unit's own "
             "20 selection episodes (seeds 0–19, mid difficulty); the fork verdict uses seeds 1000–1079.")
    lines = ["# PERUN CPU array — results", "", intro, ""]
    if best:
        b = best["retention"]
        best_line = (f"**Best unit: `{best['unit']}`** — policy {b['policy']*100:.1f} % vs static "
                     f"{b['static']*100:.1f} % vs scripted {b['scripted']*100:.1f} % "
                     f"(selection seeds; std {best['retention_std']['policy']*100:.1f})")
        lines += [best_line, ""]
    lines += ["## Ablation — policy retention % (mean ± std over seeds)", "",
              "| reward variant | arena density | n | mean | std |", "|---|---|---|---|---|"]
    for a in abl:
        lines.append(f"| {a['reward_variant']} | {a['arena_density']:.1f} | {a['n']} | {a['mean']*100:.1f} | {a['std']*100:.1f} |")
    lines += ["", "## All units (sorted by policy retention on selection seeds)", "",
              "| unit | policy | static | scripted | beats static | beats scripted | train h |", "|---|---|---|---|---|---|---|"]
    for r in sorted(rows, key=lambda r: -float(r["retention"]["policy"])):
        lines.append(f"| {r['unit']} | {r['retention']['policy']*100:.1f} | {r['retention']['static']*100:.1f} | "
                     f"{r['retention']['scripted']*100:.1f} | {r['beats_static']} | {r['beats_scripted']} | "
                     f"{r['runtimes']['train_s']/3600:.2f} |")
    n_bs = sum(1 for r in rows if r["beats_static"])
    n_bsc = sum(1 for r in rows if r["beats_scripted"])
    lines += ["", f"Units beating static on selection seeds: {n_bs}/{len(rows)}; beating scripted: {n_bsc}/{len(rows)}."]
    return "\n".join(lines) + "\n"


def main(argv: list[str] | None = None) -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--root", type=Path, default=ROOT / "runs" / "hpc")
    ap.add_argument("--out", type=Path, default=ROOT / "reports" / "hpc")
    args = ap.parse_args(argv)
    rows = load_results(args.root)
    best = select_best(rows)
    abl = ablation(rows)
    args.out.mkdir(parents=True, exist_ok=True)
    (args.out / "array_results.json").write_text(
        json.dumps({"n_units": len(rows), "best_unit": best["unit"] if best else None, "ablation": abl, "units": rows}, indent=2),
        encoding="utf-8")
    (args.out / "array_results.md").write_text(render_md(rows, best, abl), encoding="utf-8")
    if best:
        (args.out / "best_unit.txt").write_text(best["unit"] + "\n", encoding="utf-8")
    print(f"{len(rows)} units; best = {best['unit'] if best else None}; wrote {args.out}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
