from __future__ import annotations

import json
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "scripts" / "hpc"))

import aggregate


def _result(unit: str, rv: str, d: float, seed: int, pol: float) -> dict:
    return {
        "unit": unit, "reward_variant": rv, "arena_density": d, "seed": seed, "config_digest": "x",
        "retention": {"static": 0.4, "scripted": 0.55, "policy": pol},
        "retention_std": {"static": 0.1, "scripted": 0.1, "policy": 0.1},
        "beats_static": pol > 0.4, "beats_scripted": pol > 0.55,
        "runtimes": {"train_s": 3600.0, "eval_s": 10.0},
    }


def test_aggregate_selects_best_and_ignores_smoke(tmp_path: Path) -> None:
    root = tmp_path / "hpc"
    for unit, rv, d, s, pol in [("base_d0.5_s0", "base", 0.5, 0, 0.5), ("base_d0.5_s1", "base", 0.5, 1, 0.6),
                                ("lam_hi_d0.3_s0", "lam_hi", 0.3, 0, 0.6), ("_smoke_base_d0.3_s0", "base", 0.3, 0, 0.99)]:
        (root / unit).mkdir(parents=True)
        (root / unit / "result.json").write_text(json.dumps(_result(unit, rv, d, s, pol)), encoding="utf-8")
    out = tmp_path / "out"
    assert aggregate.main(["--root", str(root), "--out", str(out)]) == 0
    data = json.loads((out / "array_results.json").read_text(encoding="utf-8"))
    assert data["n_units"] == 3
    assert data["best_unit"] == "base_d0.5_s1"  # tie 0.6 with lam_hi -> lower unit name wins
    abl = {(a["reward_variant"], a["arena_density"]): a for a in data["ablation"]}
    assert abl[("base", 0.5)]["n"] == 2 and abs(abl[("base", 0.5)]["mean"] - 0.55) < 1e-9
    md = (out / "array_results.md").read_text(encoding="utf-8")
    assert "Best unit: `base_d0.5_s1`" in md and "_smoke" not in md
    assert (out / "best_unit.txt").read_text(encoding="utf-8").strip() == "base_d0.5_s1"


def test_aggregate_empty_root(tmp_path: Path) -> None:
    root = tmp_path / "empty"
    root.mkdir()
    out = tmp_path / "out"
    assert aggregate.main(["--root", str(root), "--out", str(out)]) == 0
    assert json.loads((out / "array_results.json").read_text(encoding="utf-8"))["best_unit"] is None
