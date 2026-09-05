"""Check every number quoted in README.md against the artifact it is supposed to come from.

Run it after editing the README, or to convince yourself the published claims are real:

    uv run python scripts/audit_readme.py     # exits 1 on any mismatch

Every source it reads is committed, so a reader can reproduce the whole table set.
"""

from __future__ import annotations

import json
import sys
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parents[1]
TOL = 0.06  # percentage points; the README quotes one decimal


def _arms(rel: str) -> dict[str, np.ndarray]:
    data = json.loads((ROOT / rel).read_text(encoding="utf-8"))["results"]
    return {k: np.array([e["retention"] for e in v["per_episode"]], dtype=np.float64) for k, v in data.items()}


def _ci(d: np.ndarray) -> tuple[float, float, float]:
    se = d.std(ddof=1) / np.sqrt(len(d))
    return d.mean() * 100, (d.mean() - 1.96 * se) * 100, (d.mean() + 1.96 * se) * 100


def main() -> int:
    ok: list[str] = []
    bad: list[str] = []

    def chk(label: str, got: float, want: float, tol: float = TOL) -> None:
        (ok if abs(got - want) <= tol else bad).append(f"{label}: README {want} vs artifact {got:.1f}")

    # 1. headline table
    a = _arms("reports/eval_n80_ppo_vs_scripted.json")
    zips = [k for k in a if k.endswith(".zip")]
    chk("headline static", a["static"].mean() * 100, 44.0)
    chk("headline scripted", a["scripted"].mean() * 100, 52.0)
    chk("headline ppo", a[zips[0]].mean() * 100, 51.3)
    m, lo, hi = _ci(a["scripted"] - a[zips[0]])
    chk("scripted-ppo", m, 0.7)
    chk("scripted-ppo CI lo", lo, -5.1)
    chk("scripted-ppo CI hi", hi, 6.5)

    # 2. prey trial (published evals)
    pv: dict[tuple[str, str], np.ndarray] = {}
    for h in ("static", "scripted", "best"):
        for prey in ("scripted", "best"):
            f = ROOT / f"reports/evals/eval_{h}_vs_{prey}.json"
            if not f.exists():
                continue
            d = json.loads(f.read_text(encoding="utf-8"))["results"]
            key = next((k for k in d if k.endswith(".zip") or k == h), None)
            if key:
                pv[(h, prey)] = np.array([e["retention"] for e in d[key]["per_episode"]], dtype=np.float64)
    chk("prey: static vs learned", pv[("static", "best")].mean() * 100, 27.6)
    chk("prey: scripted vs learned", pv[("scripted", "best")].mean() * 100, 50.8)
    chk("prey: ppo vs learned", pv[("best", "best")].mean() * 100, 35.9)
    d_scr = pv[("scripted", "best")] - pv[("scripted", "scripted")]
    d_ppo = pv[("best", "best")] - pv[("best", "scripted")]
    d_sta = pv[("static", "best")] - pv[("static", "scripted")]
    m, lo, hi = _ci(d_scr - d_ppo)
    chk("interaction scripted-ppo", m, 14.2)
    chk("interaction scripted-ppo CI lo", lo, 4.9)
    chk("interaction scripted-ppo CI hi", hi, 23.6)
    m2, lo2, hi2 = _ci(d_scr - d_sta)
    chk("interaction scripted-static", m2, 15.2)
    chk("interaction scripted-static CI lo", lo2, 5.8)
    chk("interaction scripted-static CI hi", hi2, 24.6)

    # 3. the 45-unit sweep
    arr = json.loads((ROOT / "reports/hpc/array_results.json").read_text(encoding="utf-8"))
    best = next(u for u in arr["units"] if u["unit"] == arr["best_unit"])
    chk("sweep units", float(arr["n_units"]), 45.0, tol=0)
    chk("sweep best on selection seeds", best["retention"]["policy"] * 100, 55.5)
    chk("sweep scripted on selection seeds", best["retention"]["scripted"] * 100, 54.3)
    chk("sweep static on selection seeds", best["retention"]["static"] * 100, 37.6)
    s9 = _arms("reports/evals/step9_hpcbest_vs_scripted.json")
    s9zip = [k for k in s9 if k.endswith(".zip")][0]
    chk("sweep best held out", s9[s9zip].mean() * 100, 49.4)
    chk("sweep scripted held out", s9["scripted"].mean() * 100, 52.0)
    chk("sweep static held out", s9["static"].mean() * 100, 44.0)

    means = [c["mean"] * 100 for c in arr["ablation"]]
    chk("ablation min", min(means), 50.3)
    chk("ablation max", max(means), 52.5)

    for line in ok:
        print("  ok   " + line)
    for line in bad:
        print("  BAD  " + line)
    print(f"{len(ok)} checks passed, {len(bad)} mismatched")
    return 1 if bad else 0


if __name__ == "__main__":
    sys.exit(main())
