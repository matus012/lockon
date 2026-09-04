"""License / visual guard (project.md §10, P1 pattern). Gate G8.

Invariants:
1. Nothing under runs/ data/ videos/ checkpoints/ is tracked; no weight/blob extensions tracked
   outside tests/fixtures.
2. Every tracked visual is classified in ALLOWED_TRACKED_VISUALS as "plot" (pure matplotlib) or
   "sim-render" (MuJoCo output). No other class exists: this project has no dataset-derived
   pixels, so nothing else can ever be allowlisted.
3. .gitignore directory patterns are anchored (leading slash) unless they carry a wildcard —
   an unanchored `data/` once silently dropped ws/100's src/omot/data package (P1 D47).
4. Every .py under src/ tests/ scripts/ is git-tracked (a bundle = `git archive HEAD`).
5. Tracked files outside the visual allowlist are ≤ 2 MB.
6. No AGPL or otherwise disallowed package is pinned in requirements.txt.
"""

from __future__ import annotations

import subprocess
from pathlib import Path

import pytest

ROOT = Path(__file__).resolve().parents[1]
VISUAL_EXT = {".png", ".gif", ".mp4", ".avi", ".webm", ".svg", ".jpg", ".jpeg"}
BLOB_EXT = {".pt", ".pth", ".onnx", ".safetensors", ".ckpt", ".npy", ".npz", ".zip", ".tar", ".gz"}
FORBIDDEN_DIRS = ("runs/", "data/", "videos/", "checkpoints/", "weights/")
SIZE_CEILING = 2 * 1024 * 1024
DISALLOWED_PACKAGES = {"ultralytics", "boxmot", "yolov5", "mmtrack"}
ALLOWED_CLASSES = {"plot", "sim-render"}

# Every committed visual, classified by a human. Extend deliberately; never wildcard.
ALLOWED_TRACKED_VISUALS: dict[str, str] = {
    "reports/gifs/sensor_3ch.gif": "sim-render",
    "reports/gifs/occlusion_lock.gif": "sim-render",
    "reports/gifs/lights_cut_lock.gif": "sim-render",
    "reports/gifs/chase.gif": "sim-render",
}


def _tracked() -> list[str]:
    out = subprocess.run(["git", "ls-files", "-z"], cwd=ROOT, capture_output=True, check=True)
    return [p for p in out.stdout.decode("utf-8").split("\0") if p]


@pytest.fixture(scope="module")
def tracked() -> list[str]:
    return _tracked()


def test_no_tracked_files_in_forbidden_dirs(tracked: list[str]) -> None:
    bad = [p for p in tracked if p.startswith(FORBIDDEN_DIRS)]
    assert not bad, bad


def test_no_tracked_blobs_outside_fixtures(tracked: list[str]) -> None:
    bad = [p for p in tracked if Path(p).suffix.lower() in BLOB_EXT and not p.startswith("tests/fixtures/")]
    assert not bad, bad


def test_tracked_visuals_are_allowlisted(tracked: list[str]) -> None:
    visuals = [p for p in tracked if Path(p).suffix.lower() in VISUAL_EXT]
    bad = [p for p in visuals if p not in ALLOWED_TRACKED_VISUALS]
    assert not bad, f"unclassified tracked visuals: {bad}"


def test_allowlist_is_clean() -> None:
    for path, cls in ALLOWED_TRACKED_VISUALS.items():
        assert cls in ALLOWED_CLASSES, (path, cls)
        assert "crop" not in Path(path).name.lower(), path


def test_gitignore_directory_patterns_are_anchored() -> None:
    src_dirs = {p.name for p in (ROOT / "src").rglob("*") if p.is_dir() and p.name != "__pycache__"}
    offenders = []
    for line in (ROOT / ".gitignore").read_text(encoding="utf-8").splitlines():
        s = line.strip()
        if not s or s.startswith(("#", "!")):
            continue
        if s.endswith("/") and not s.startswith("/") and "*" not in s and s.rstrip("/") in src_dirs:
            offenders.append(s)
    assert not offenders, f"unanchored dir patterns shadowing src/: {offenders}"


def test_every_source_file_is_tracked(tracked: list[str]) -> None:
    tracked_set = set(tracked)
    missing = []
    for top in ("src", "tests", "scripts"):
        for f in (ROOT / top).rglob("*.py"):
            if "__pycache__" in f.parts:
                continue
            rel = f.relative_to(ROOT).as_posix()
            if rel not in tracked_set:
                missing.append(rel)
    assert not missing, f"untracked source files (commit them): {missing}"


def test_no_large_tracked_files_outside_visual_allowlist(tracked: list[str]) -> None:
    big = []
    for p in tracked:
        if p in ALLOWED_TRACKED_VISUALS:
            continue
        f = ROOT / p
        if f.exists() and f.stat().st_size > SIZE_CEILING:
            big.append((p, f.stat().st_size))
    assert not big, big


def test_no_disallowed_packages_pinned() -> None:
    names = set()
    for line in (ROOT / "requirements.txt").read_text(encoding="utf-8").splitlines():
        s = line.strip().lower()
        if s and not s.startswith(("#", "-")):
            names.add(s.split("==")[0].split("[")[0])
    assert not (names & DISALLOWED_PACKAGES), names & DISALLOWED_PACKAGES
