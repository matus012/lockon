"""Tests for `lockon.demo.render_all` and `lockon.demo.viewer` (SPEC.md `demo/SPEC.md`)."""

from __future__ import annotations

import json
from pathlib import Path

import pytest

from lockon.demo import render_all, viewer


@pytest.mark.render
def test_render_all_scripted_produces_three_shots(tmp_path: Path) -> None:
    out = tmp_path / "shots"
    manifest = render_all.render_all("scripted", out, steps=12, fps=10)

    assert set(manifest) == {"occlusion", "lights_cut", "chase"}
    for name in manifest:
        mp4 = out / f"{name}.mp4"
        assert mp4.exists()
        assert mp4.stat().st_size > 10_000

    written = json.loads((out / "manifest.json").read_text(encoding="utf-8"))
    assert set(written) == {"occlusion", "lights_cut", "chase"}
    assert (out.parent / "index.html").exists()


def test_viewer_check(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    shots_dir = tmp_path / "demo" / "shots"
    shots_dir.mkdir(parents=True)
    manifest = {
        "occlusion": {"file": "occlusion.mp4", "caption": "gap", "retention": 0.9},
        "lights_cut": {"file": "lights_cut.mp4", "caption": "cut", "retention": 0.8},
        "chase": {"file": "chase.mp4", "caption": "chase", "retention": 0.7},
    }
    for entry in manifest.values():
        (shots_dir / str(entry["file"])).write_bytes(b"0" * 20_000)
    manifest_path = shots_dir / "manifest.json"
    manifest_path.write_text(json.dumps(manifest), encoding="utf-8")
    index_html = tmp_path / "demo" / "index.html"
    index_html.write_text("<html></html>", encoding="utf-8")

    monkeypatch.setattr(viewer, "SHOTS_DIR", shots_dir)
    monkeypatch.setattr(viewer, "MANIFEST_PATH", manifest_path)
    monkeypatch.setattr(viewer, "INDEX_HTML", index_html)
    monkeypatch.setattr(viewer, "CURVES_PNG", tmp_path / "reports" / "curves" / "summary.png")
    monkeypatch.setattr(viewer, "SWEEP_RESULTS", tmp_path / "reports" / "sweep" / "results.json")
    monkeypatch.setattr(viewer, "FAILURE_REPORT", tmp_path / "reports" / "sweep" / "failure_report.md")

    assert viewer.main(["--check"]) == 0

    (shots_dir / "chase.mp4").unlink()
    assert viewer.main(["--check"]) == 1
