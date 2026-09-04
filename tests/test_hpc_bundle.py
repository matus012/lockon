"""plan.md §7 (HPC): the bundle must ship every input the CPU sweep needs, unit
enumeration must be exact, and emitted sbatch scripts must be LF-only and resume-safe.

Ported/trimmed from ws/100_occlusion_mot/tests/test_hpc_bundle.py: this repo ships no
external datasets, so there is no reid/detector/asset coverage to guard -- only the
repo + wheelhouse plan.
"""
from __future__ import annotations

import json
import subprocess
import sys
import tarfile
from pathlib import Path

import pytest

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "scripts" / "hpc"))

import make_hpc_bundle as mhb
import sweep_launcher as sl
from sweep_common import (
    ARENA_DENSITIES,
    REWARD_VARIANTS,
    SEEDS,
    budget_table,
    enumerate_units,
    generate_unit_config,
    hms,
    load_config,
    parse_unit,
    unit_name,
    wall_seconds,
)

CONFIG_PATH = ROOT / "configs" / "hpc_sweep.yaml"


@pytest.fixture(scope="module")
def cfg() -> dict:
    return load_config(CONFIG_PATH)


# ----------------------------------------------------------------- unit enumeration


def test_enumerate_units_yields_exactly_45_unique_names(cfg: dict) -> None:
    units = enumerate_units(cfg)
    assert len(units) == 45
    assert len(set(units)) == 45


def test_enumeration_matches_module_defaults() -> None:
    assert len(SEEDS) == 5
    assert len(REWARD_VARIANTS) == 3
    assert len(ARENA_DENSITIES) == 3
    assert 5 * 3 * 3 == 45


def test_unit_name_round_trips() -> None:
    for reward in REWARD_VARIANTS:
        for density in ARENA_DENSITIES:
            for seed in SEEDS:
                name = unit_name(reward, density, seed)
                r, d, s = parse_unit(name)
                assert (r, d, s) == (reward, density, seed)


def test_parse_unit_rejects_malformed_ids() -> None:
    with pytest.raises(AssertionError):
        parse_unit("not-a-unit")
    with pytest.raises(AssertionError):
        parse_unit("base_0.5_s0")  # missing 'd' prefix


def test_generate_unit_config_applies_overrides(cfg: dict) -> None:
    unit = unit_name("lam_hi", 0.7, 3)
    unit_cfg = generate_unit_config(cfg, unit)
    assert unit_cfg["seed"] == 3
    assert unit_cfg["reward"]["action_l2"] == 0.03
    assert unit_cfg["difficulty"] == {"occluder_density": 0.7}
    # untouched base fields survive
    assert unit_cfg["n_envs"] == 8


# ----------------------------------------------------------------- budget table


def test_budget_table_arithmetic(cfg: dict) -> None:
    table = budget_table(cfg)
    assert table["n_units"] == 45
    assert table["cpu_h"] == pytest.approx(45 * cfg["est_h_per_unit"])
    assert table["gpu_h"] == pytest.approx(sum(cfg["gpu_h"].values()))
    assert table["total_h"] == pytest.approx(table["cpu_h"] + table["gpu_h"])
    # the ceiling is a GPU-h allocation cap (plan.md §7), not a combined CPU+GPU budget
    assert table["fits_ceiling"] == (table["gpu_h"] <= table["ceiling_h"])
    assert table["gpu_h"] <= table["ceiling_h"], "GPU-h grid must fit its own cap"


def test_wall_seconds_and_hms_agree_on_scale() -> None:
    # hms() is HH:MM:SS rounded up to 5 min; wall_seconds() is the same wall in
    # seconds for coreutils `timeout` -- they must describe the same duration
    # (D49-style lesson from ws/100: the two formats are not interchangeable, but
    # the underlying minutes must match).
    h, m, s = (int(x) for x in hms(2.0).split(":"))
    assert s == 0
    assert wall_seconds(2.0) == (h * 3600 + m * 60)


# ----------------------------------------------------------------- bundle plan


def test_bundle_plan_covers_every_required_config(cfg: dict) -> None:
    gaps = mhb.coverage_gaps(ROOT)
    assert not gaps, f"bundle plan does not cover required input(s): {gaps}"


def test_bundle_plan_without_wheelhouse_still_covers_repo_inputs() -> None:
    gaps = mhb.coverage_gaps(ROOT, include_wheelhouse=False)
    assert not gaps


def test_required_config_paths_exist_on_the_dev_tree() -> None:
    missing = [r.path for r in mhb.required_inputs(ROOT, include_wheelhouse=False)
               if not (ROOT / r.path).exists()]
    assert not missing, f"required bundle input(s) absent from the dev tree: {missing}"


def test_plan_entries_have_no_duplicate_coverage() -> None:
    seen: dict[str, str] = {}
    for entry in mhb.plan_entries(ROOT):
        for key in entry.covers:
            assert key not in seen, f"{key} covered by both {seen[key]} and {entry.dest}"
            seen[key] = entry.dest


# ----------------------------------------------------------------- round trip (fixture repo)


def _fixture_repo(tmp_path: Path) -> Path:
    root = tmp_path / "repo"
    files = {
        "configs/ppo_local.yaml": "name: ppo_local\nseed: 0\ntotal_steps: 1000\n"
                                   "n_envs: 1\nframe_stack: 1\ncurriculum:\n"
                                   "  start_dial: 0.3\n  end_dial: 0.5\n"
                                   "  switch_fraction: 0.4\nreward:\n  visible: 1.0\n",
        "configs/hpc_sweep.yaml": "name: tiny\n",
        "requirements-hpc.txt": "numpy==2.4.6\n",
        "data/wheelhouse/numpy-2.4.6.whl": "wheel-bytes",
        "src/lockon/__init__.py": "",
    }
    for rel, content in files.items():
        dest = root / rel
        dest.parent.mkdir(parents=True, exist_ok=True)
        dest.write_text(content, encoding="utf-8")

    subprocess.run(["git", "init", "-q"], cwd=root, check=True)
    subprocess.run(["git", "config", "user.email", "t@t"], cwd=root, check=True)
    subprocess.run(["git", "config", "user.name", "t"], cwd=root, check=True)
    subprocess.run(["git", "add", "configs", "src"], cwd=root, check=True)
    subprocess.run(["git", "commit", "-qm", "fixture"], cwd=root, check=True)
    return root


def test_build_verify_unpack_round_trip(tmp_path: Path) -> None:
    root = _fixture_repo(tmp_path)
    bundle = mhb.build(root=root, out_dir=tmp_path / "dist", archive=False)
    manifest = json.loads((bundle / mhb.MANIFEST_NAME).read_text(encoding="utf-8"))

    assert manifest["git"]["dirty"] is False
    assert {e["path"] for e in manifest["entries"]} >= {"repo.tar", "wheelhouse.tar"}
    assert mhb.verify(bundle) == 0

    dest = tmp_path / "out"
    assert mhb.unpack(bundle, dest) == 0
    for rel in ("repo/configs/hpc_sweep.yaml", "repo/src/lockon/__init__.py",
                "wheelhouse/numpy-2.4.6.whl"):
        assert (dest / rel).exists(), f"unpack lost {rel}"


def test_build_no_wheelhouse_stages_repo_only(tmp_path: Path) -> None:
    root = _fixture_repo(tmp_path)
    bundle = mhb.build(root=root, out_dir=tmp_path / "dist", archive=False,
                        include_wheelhouse=False)
    manifest = json.loads((bundle / mhb.MANIFEST_NAME).read_text(encoding="utf-8"))
    assert {e["path"] for e in manifest["entries"]} == {"repo.tar"}
    assert mhb.verify(bundle) == 0


def test_verify_detects_corruption(tmp_path: Path) -> None:
    root = _fixture_repo(tmp_path)
    bundle = mhb.build(root=root, out_dir=tmp_path / "dist", archive=False)
    target = bundle / "repo.tar"
    data = bytearray(target.read_bytes())
    data[-1] ^= 0xFF
    target.write_bytes(bytes(data))
    assert mhb.verify(bundle) == 1


def test_build_refuses_a_dirty_worktree(tmp_path: Path) -> None:
    root = _fixture_repo(tmp_path)
    (root / "configs" / "hpc_sweep.yaml").write_text("name: dirty\n", encoding="utf-8")
    with pytest.raises(SystemExit, match="dirty worktree"):
        mhb.build(root=root, out_dir=tmp_path / "dist", archive=False)


def test_repo_tar_excludes_untracked_files(tmp_path: Path) -> None:
    root = _fixture_repo(tmp_path)
    (root / "runs_junk.txt").write_text("not tracked", encoding="utf-8")
    bundle = mhb.build(root=root, out_dir=tmp_path / "dist", archive=False,
                        allow_dirty=True)
    with tarfile.open(bundle / "repo.tar", "r:") as tar:
        names = tar.getnames()
    assert "runs_junk.txt" not in names
    assert "configs/ppo_local.yaml" in names


# ----------------------------------------------------------------- emitted scripts


def test_emitted_sbatch_is_lf_and_resume_safe(tmp_path: Path, cfg: dict) -> None:
    """CRLF in an sbatch script reaches the cluster as `#!/bin/bash\\r` and dies with
    "bad interpreter" (the exact ws/100 D47 failure this pattern guards against)."""
    text = sl.render_sbatch(cfg, CONFIG_PATH)
    raw = text.encode("utf-8")
    assert b"\r\n" not in raw
    assert raw.startswith(b"#!/bin/bash\n")
    assert 'if [ -f "$RESULT" ]; then' in text, "array is not resume-safe"
    assert "timeout --signal=TERM" in text, "per-unit wall limit not enforced"
    assert "runs/hpc/logs/" in text


def test_sbatch_array_covers_every_unit_once(cfg: dict) -> None:
    units = enumerate_units(cfg)
    text = sl.render_sbatch(cfg, CONFIG_PATH)
    assert f"#SBATCH --array=0-{len(units) - 1}%" in text
    for unit in units:
        assert f'"{unit}"' in text, f"{unit} missing from the emitted array"


def test_smoke_sbatch_uses_one_unit_and_small_step_count(cfg: dict) -> None:
    text = sl.render_smoke_sbatch(cfg, CONFIG_PATH)
    assert b"\r\n" not in text.encode("utf-8")
    assert f"--total-steps {cfg['smoke_total_steps']}" in text
    units = enumerate_units(cfg)
    assert units[0] in text


def test_gpu_sbatch_is_a_marked_placeholder(cfg: dict) -> None:
    text = sl.render_gpu_sbatch(cfg)
    assert b"\r\n" not in text.encode("utf-8")
    assert "torch.cuda.is_available()" in text, "GPU sbatch must probe the GPU"
    assert "train_prey" in text
    assert "NOT YET IMPLEMENTED" in text
    assert "--gres=gpu:1" in text


def test_sbatch_carries_no_windows_paths(cfg: dict) -> None:
    for text in (sl.render_sbatch(cfg, CONFIG_PATH), sl.render_smoke_sbatch(cfg, CONFIG_PATH),
                 sl.render_gpu_sbatch(cfg)):
        assert "\\" not in text.replace("\\\n", ""), "backslash path leaked into sbatch"
        assert ".venv\\Scripts" not in text
        assert "C:/" not in text and "C:\\" not in text
