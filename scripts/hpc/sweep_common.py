"""Shared plumbing for the PERUN CPU sweep (plan.md §7): unit enumeration, per-unit PPO
config generation, result paths, budget arithmetic, sbatch helpers. Single source of
truth for scripts/hpc/{unit,sweep_launcher,make_hpc_bundle}.py -- local and SLURM
execution paths must never drift apart (same pattern as ws/100_occlusion_mot's
sweep_common.py, ported and trimmed: lockon has no reid/detector/data-source axes).

45 units = 5 seeds x 3 reward variants x 3 arena occluder densities (task spec). Every
unit trains `lockon.harness.train` (harness/SPEC.md CLI contract, untouched here) on a
generated per-unit YAML derived from configs/ppo_local.yaml.

ASSUMPTION (impl agent, 2026-09-04, logged for the owner): train.py's config exposes
difficulty via `curriculum.{start_dial,end_dial}` (all axes) plus a per-axis `difficulty:`
override block; the arena axis pins `occluder_density` only (2026-09-04).
"""
from __future__ import annotations

import hashlib
import logging
import re
import subprocess
from pathlib import Path
from typing import Any

import yaml

logger = logging.getLogger(__name__)

ROOT = Path(__file__).resolve().parents[2]
BASE_PPO_CONFIG = ROOT / "configs" / "ppo_local.yaml"
RESULTS_ROOT = ROOT / "runs" / "hpc"

SEEDS: tuple[int, ...] = (0, 1, 2, 3, 4)

# reward variant name -> overrides merged into the base config's `reward` block
REWARD_VARIANTS: dict[str, dict[str, float]] = {
    "base": {},
    "lam_hi": {"action_l2": 0.03},
    "k_short": {"lost_after": 10},
}

ARENA_DENSITIES: tuple[float, ...] = (0.3, 0.5, 0.7)

_REQUIRED_KEYS = ("name", "seeds", "reward_variants", "arena_densities", "est_h_per_unit",
                   "total_steps", "gpu_h", "slurm")


def load_config(path: Path) -> dict[str, Any]:
    """Load + validate configs/hpc_sweep.yaml. Fails fast on schema violations."""
    cfg = yaml.safe_load(path.read_text(encoding="utf-8"))
    assert isinstance(cfg, dict), f"{path}: config must be a YAML mapping"
    for key in _REQUIRED_KEYS:
        assert key in cfg, f"{path}: missing required key '{key}'"
    assert cfg["seeds"], f"{path}: seeds must be non-empty"
    assert cfg["reward_variants"], f"{path}: reward_variants must be non-empty"
    assert cfg["arena_densities"], f"{path}: arena_densities must be non-empty"
    assert cfg["est_h_per_unit"] > 0, f"{path}: est_h_per_unit must be positive"
    return cfg


def unit_name(reward: str, density: float, seed: int) -> str:
    return f"{reward}_d{density:.1f}_s{seed}"


# reward variant names may themselves contain '_' (e.g. "lam_hi", "k_short"), so a
# plain split("_") is ambiguous -- anchor on the fixed '_d<density>_s<seed>' suffix
# instead and take everything before it as the reward name.
_UNIT_RE = re.compile(r"^(?P<reward>.+)_d(?P<density>[0-9.]+)_s(?P<seed>\d+)$")


def parse_unit(unit: str) -> tuple[str, float, int]:
    """'lam_hi_d0.7_s3' -> ('lam_hi', 0.7, 3). Fails fast on a malformed unit id."""
    m = _UNIT_RE.match(unit)
    assert m is not None, f"malformed unit '{unit}' (expected '<reward>_d<density>_s<seed>')"
    return m.group("reward"), float(m.group("density")), int(m.group("seed"))


def enumerate_units(cfg: dict[str, Any]) -> list[str]:
    """Deterministic full unit list -- the single source of truth both local and SLURM
    modes enumerate from (keeps the two execution paths in lockstep)."""
    units: list[str] = []
    for reward in cfg["reward_variants"]:
        for density in cfg["arena_densities"]:
            for seed in cfg["seeds"]:
                units.append(unit_name(reward, float(density), int(seed)))
    return units


def result_path(unit: str, results_root: Path = RESULTS_ROOT) -> Path:
    return results_root / unit / "result.json"


def generate_unit_config(
    sweep_cfg: dict[str, Any], unit: str, base_path: Path = BASE_PPO_CONFIG
) -> dict[str, Any]:
    """The per-unit YAML: base ppo_local.yaml with seed / reward / difficulty overrides
    (see module docstring for the difficulty-axis assumption)."""
    reward, density, seed = parse_unit(unit)
    assert reward in sweep_cfg["reward_variants"], (
        f"unit '{unit}': reward variant '{reward}' not in config"
    )
    base = yaml.safe_load(base_path.read_text(encoding="utf-8"))
    out = dict(base)
    out["name"] = unit
    out["seed"] = seed
    out["total_steps"] = sweep_cfg["total_steps"]
    reward_block = dict(base.get("reward", {}))
    reward_block.update(sweep_cfg["reward_variants"][reward])
    out["reward"] = reward_block
    # arena axis: pin occluder_density only; the curriculum still ramps the other four dials
    # (train.py config key `difficulty:` = per-axis overrides, added 2026-09-04)
    out["difficulty"] = {"occluder_density": density}
    return out


def budget_table(cfg: dict[str, Any]) -> dict[str, Any]:
    """CPU-h = units x est_h/unit; GPU-h = sum(cfg['gpu_h'].values()) (plan.md §7:
    learned-prey trial + render pass). The ceiling (plan.md §7: "70-100 cap") is a
    GPU-h allocation cap, not a combined wall-clock budget -- PERUN's CPU array has
    no such ceiling, only the GPU-h grant is metered. Never rescales estimates to make
    the ceiling check pass (CLAUDE.md: thresholds never move to make a gate pass)."""
    units = enumerate_units(cfg)
    est_h = float(cfg["est_h_per_unit"])
    cpu_h = round(len(units) * est_h, 2)
    gpu_h = round(float(sum(cfg["gpu_h"].values())), 2)
    ceiling = float(cfg["slurm"].get("budget_ceiling_h", gpu_h))
    total = round(cpu_h + gpu_h, 2)
    return {
        "n_units": len(units),
        "est_h_per_unit": est_h,
        "cpu_h": cpu_h,
        "gpu_h": gpu_h,
        "gpu_h_breakdown": dict(cfg["gpu_h"]),
        "total_h": total,
        "ceiling_h": ceiling,
        "fits_ceiling": gpu_h <= ceiling,
    }


def format_budget_table(table: dict[str, Any]) -> str:
    lines = [
        (f"budget: {table['n_units']} CPU units x {table['est_h_per_unit']} h/unit "
         f"= {table['cpu_h']} CPU-h"),
    ]
    for name, h in table["gpu_h_breakdown"].items():
        lines.append(f"  GPU: {name} = {h} h")
    lines.append(f"  GPU-h total = {table['gpu_h']} h")
    lines.append(
        f"  GPU-h {table['gpu_h']} h vs ceiling {table['ceiling_h']} h "
        f"(fits={table['fits_ceiling']}; total incl. CPU-h = {table['total_h']} h -- "
        f"the ceiling is a GPU-h allocation cap, plan.md §7)"
    )
    return "\n".join(lines)


def config_digest(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def write_text_lf(path: Path, text: str) -> None:
    """Write with LF endings on every platform.

    Path.write_text() opens in text mode with newline=None, so on Windows every '\\n'
    becomes '\\r\\n'. A CRLF sbatch script reaches the cluster with a '#!/bin/bash\\r'
    shebang and dies with "bad interpreter". Anything destined for Linux goes through
    this function (ported verbatim from ws/100_occlusion_mot/scripts/sweep_common.py).
    """
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8", newline="\n") as f:
        f.write(text)


def posix_relpath(path: Path | str, root: Path = ROOT) -> str:
    """POSIX-style path, relative to `root` when possible -- the SLURM (Linux) target
    must never see a Windows-separator or Windows-absolute path emitted from this dev
    box. Falls back to an as-posix() absolute path outside `root` (e.g. a tmp_path)."""
    p = Path(path)
    try:
        rel = p.resolve().relative_to(root.resolve())
        return rel.as_posix()
    except ValueError:
        return p.as_posix()


def entrypoint_cmd(config_path: Path | str, unit: str, total_steps: int | None = None) -> list[str]:
    """The IDENTICAL argument shape run locally (in-process) and on SLURM: only the
    --unit value (and, for the smoke job, --total-steps) varies."""
    cmd = ["scripts/hpc/unit.py", "--config", str(config_path), "--unit", unit]
    if total_steps is not None:
        cmd += ["--total-steps", str(total_steps)]
    return cmd


def run_subprocess(cmd: list[str], log_path: Path, cwd: Path = ROOT) -> None:
    """Run one subprocess, streaming output to `logger` and appending to `log_path`.
    Fail-fast: a non-zero return code raises immediately (no silent swallowing)."""
    logger.info("$ %s", " ".join(cmd))
    log_path.parent.mkdir(parents=True, exist_ok=True)
    with log_path.open("a", encoding="utf-8") as lf:
        lf.write(f"\n=== {' '.join(cmd)} ===\n")
        with subprocess.Popen(
            cmd, cwd=cwd, stdout=subprocess.PIPE, stderr=subprocess.STDOUT,
            text=True, bufsize=1, encoding="utf-8", errors="replace",
        ) as proc:
            assert proc.stdout is not None
            for line in proc.stdout:
                logger.info("  | %s", line.rstrip("\n"))
                lf.write(line if line.endswith("\n") else line + "\n")
            returncode = proc.wait()
    if returncode != 0:
        raise RuntimeError(f"subprocess failed (rc={returncode}): {' '.join(cmd)}")


def wall_seconds(est_h: float, margin: float = 1.5) -> int:
    """coreutils `timeout` DURATION seconds -- NOT SLURM's HH:MM:SS `--time` form (the
    two are not interchangeable; passing HH:MM:SS to `timeout` kills the task in under
    a second with "invalid time interval", D49 lesson from ws/100)."""
    return int(est_h * margin * 3600)


def hms(est_h: float, margin: float = 1.5) -> str:
    """SLURM `--time` form (HH:MM:SS), rounded up to the next 5 minutes."""
    total_min = int(-(-est_h * margin * 60 // 5) * 5)
    return f"{total_min // 60:02d}:{total_min % 60:02d}:00"
