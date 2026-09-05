from __future__ import annotations

import json
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "scripts"))

import prey_table


def _write(d: Path, hunter: str, prey: str, key: str, vals: list[float]) -> None:
    payload = {"results": {key: {"policy": key, "retention_mean": sum(vals) / len(vals),
                                 "per_episode": [{"seed": 1000 + i, "retention": v} for i, v in enumerate(vals)]}}}
    (d / f"eval_{hunter}_vs_{prey}.json").write_text(json.dumps(payload), encoding="utf-8")


def test_prey_table_reports_delta(tmp_path: Path) -> None:
    d = tmp_path / "prey"
    d.mkdir()
    _write(d, "scripted", "scripted", "scripted", [0.6, 0.6, 0.6, 0.6])
    _write(d, "scripted", "best", "scripted", [0.4, 0.4, 0.4, 0.4])
    _write(d, "static", "scripted", "static", [0.5, 0.5, 0.5, 0.5])
    _write(d, "static", "best", "static", [0.5, 0.5, 0.5, 0.5])
    out = tmp_path / "prey_trial.md"
    assert prey_table.main(["--dir", str(d), "--out", str(out)]) == 0
    md = out.read_text(encoding="utf-8")
    assert "scripted hunter" in md and "static camera" in md
    assert "60.0" in md and "40.0" in md
    assert "-20.0" in md  # learned prey is 20 points harder against the scripted hunter
    assert "n = 4 episodes per cell" in md


def test_prey_table_missing_dir(tmp_path: Path) -> None:
    assert prey_table.main(["--dir", str(tmp_path / "nope"), "--out", str(tmp_path / "x.md")]) == 1
