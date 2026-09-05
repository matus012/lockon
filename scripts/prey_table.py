"""Render the learned-prey trial table (plan.md §7 GPU line) from the trial's eval JSONs.

The trial asks one question: **is a learned evader a harder examiner than the scripted one?**
Rows = hunters, columns = prey. Lower retention against the same hunter = harder prey.

    uv run python scripts/prey_table.py [--dir runs/hpc_prey] [--out reports/prey_trial.md]
"""

from __future__ import annotations

import argparse
import json
import re
import sys
from pathlib import Path
from typing import Any

import numpy as np

ROOT = Path(__file__).resolve().parents[1]
HUNTER_LABEL = {"static": "static camera", "scripted": "scripted hunter"}
PREY_LABEL = {"scripted": "scripted prey", "best": "learned prey"}


def _hunter_label(key: str) -> str:
    return HUNTER_LABEL.get(key, "PPO hunter")


def _episodes(entry: dict[str, Any]) -> np.ndarray:
    return np.array([e["retention"] for e in entry["per_episode"]], dtype=np.float64)


def collect(d: Path) -> dict[tuple[str, str], np.ndarray]:
    """eval_<hunter>_vs_<prey>.json -> {(hunter, prey): per-episode retention of that hunter}."""
    out: dict[tuple[str, str], np.ndarray] = {}
    for p in sorted(d.glob("eval_*_vs_*.json")):
        m = re.match(r"eval_(.+)_vs_(.+)\.json$", p.name)
        if not m:
            continue
        hunter, prey = m.group(1), m.group(2)
        data = json.loads(p.read_text(encoding="utf-8"))["results"]
        key = next((k for k in data if k.endswith(".zip") or k == hunter), None)
        if key is None:
            continue
        out[(hunter, prey)] = _episodes(data[key])
    return out


def render(cells: dict[tuple[str, str], np.ndarray]) -> str:
    hunters = sorted({h for h, _ in cells}, key=lambda h: (h != "static", h != "scripted", h))
    preys = sorted({p for _, p in cells}, key=lambda p: p != "scripted")
    intro = ("Lock retention % of each hunter against each prey; **lower is a harder prey**. "
             "Held-out seeds, mid difficulty, per-episode tracker noise.")
    lines = ["# Learned-prey trial (PERUN, one H200)", "", intro, "",
             "| hunter | " + " | ".join(PREY_LABEL.get(p, p) for p in preys) + " | Δ (learned − scripted) |",
             "|---" * (len(preys) + 2) + "|"]
    for h in hunters:
        row = [_hunter_label(h)]
        vals = {}
        for p in preys:
            v = cells.get((h, p))
            vals[p] = v
            row.append("—" if v is None else f"{v.mean()*100:.1f} ± {v.std(ddof=1)*100:.1f}")
        a, b = vals.get(preys[-1]), vals.get(preys[0])
        if a is not None and b is not None and len(a) == len(b):
            d = a - b
            se = d.std(ddof=1) / np.sqrt(len(d))
            row.append(f"{d.mean()*100:+.1f} [{(d.mean()-1.96*se)*100:+.1f}, {(d.mean()+1.96*se)*100:+.1f}]")
        else:
            row.append("—")
        lines.append("| " + " | ".join(row) + " |")
    n = len(next(iter(cells.values()))) if cells else 0
    note = (f"n = {n} episodes per cell. A negative Δ means the learned prey evades better than the "
            "scripted one against that hunter; a CI spanning zero means the trial could not "
            "separate them.")
    lines += ["", note]
    return "\n".join(lines) + "\n"


def main(argv: list[str] | None = None) -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--dir", type=Path, default=ROOT / "runs" / "hpc_prey")
    ap.add_argument("--out", type=Path, default=ROOT / "reports" / "prey_trial.md")
    args = ap.parse_args(argv)
    cells = collect(args.dir)
    if not cells:
        print(f"no eval_*_vs_*.json under {args.dir}", file=sys.stderr)
        return 1
    args.out.parent.mkdir(parents=True, exist_ok=True)
    args.out.write_text(render(cells), encoding="utf-8")
    print(f"{len(cells)} cells -> {args.out}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
