"""scripts/hpc/unit.py -- the identical unit executor for local dry-runs and PERUN
SLURM array jobs alike (plan.md §7; hard design constraint mirrored from
ws/100_occlusion_mot/scripts/sweep_unit.py: same script, different config/env, no code
fork). One invocation = one unit: generate the per-unit PPO config, train via the
frozen `lockon.harness.train` CLI (harness/SPEC.md; never imported directly -- this
script only shells out to the documented contract), then eval n=20 vs static and vs
scripted, and write runs/hpc/<unit>/result.json.

    uv run python scripts/hpc/unit.py --config configs/hpc_sweep.yaml --unit base_d0.5_s0 \\
        [--total-steps 20000] [--wall-hours 2.5]
"""
from __future__ import annotations

import argparse
import json
import logging
import sys
import time
from pathlib import Path
from typing import Any

import yaml

SCRIPTS_HPC = Path(__file__).resolve().parent
ROOT = SCRIPTS_HPC.parents[1]
sys.path.insert(0, str(SCRIPTS_HPC))
sys.path.insert(0, str(ROOT / "src"))

from sweep_common import (
    RESULTS_ROOT,
    config_digest,
    generate_unit_config,
    load_config,
    parse_unit,
    result_path,
    run_subprocess,
    write_text_lf,
)

logging.basicConfig(level=logging.INFO, format="%(asctime)s %(message)s", datefmt="%H:%M:%S",
                     stream=sys.stdout)
logger = logging.getLogger("hpc_unit")

N_EVAL_EPISODES = 20
DEFAULT_WALL_HOURS = 2.5


def run_unit(
    sweep_cfg: dict[str, Any],
    unit: str,
    config_path: Path,
    total_steps: int | None,
    wall_hours: float | None,
    results_root: Path = RESULTS_ROOT,
) -> dict[str, Any]:
    dest = result_path(unit, results_root)
    if dest.exists():
        logger.info("skip %s (result exists: %s)", unit, dest)
        return dict(json.loads(dest.read_text(encoding="utf-8")))

    reward, density, seed = parse_unit(unit)
    out_dir = results_root / unit
    out_dir.mkdir(parents=True, exist_ok=True)

    unit_cfg = generate_unit_config(sweep_cfg, unit)
    wall = wall_hours if wall_hours is not None else float(sweep_cfg.get("est_h_per_unit",
                                                                          DEFAULT_WALL_HOURS))
    unit_cfg_path = out_dir / "config.yaml"
    write_text_lf(unit_cfg_path, yaml.safe_dump(unit_cfg, sort_keys=False))

    train_cmd = [
        sys.executable, "-m", "lockon.harness.train",
        "--config", str(unit_cfg_path), "--out", str(out_dir),
        "--wall-hours", str(wall),
    ]
    if total_steps is not None:
        train_cmd += ["--total-steps", str(total_steps)]
    t0 = time.time()
    run_subprocess(train_cmd, out_dir / "train.log", cwd=ROOT)
    train_s = time.time() - t0

    import lockon.harness.eval as harness_eval  # no py.typed marker (core package only)
    from lockon.core.schemas import Difficulty

    evaluate = harness_eval.evaluate

    policy_path = out_dir / "best.zip"
    assert policy_path.exists(), f"train.py did not produce {policy_path}"

    t0 = time.time()
    eval_results = {
        name: evaluate(name, Difficulty(), range(N_EVAL_EPISODES))
        for name in ("static", "scripted")
    }
    eval_results["policy"] = evaluate(str(policy_path), Difficulty(), range(N_EVAL_EPISODES))
    eval_s = time.time() - t0

    policy_mean = eval_results["policy"]["retention_mean"]
    result: dict[str, Any] = {
        "unit": unit,
        "reward_variant": reward,
        "arena_density": density,
        "seed": seed,
        "config_digest": config_digest(config_path),
        "retention": {name: r["retention_mean"] for name, r in eval_results.items()},
        "retention_std": {name: r["retention_std"] for name, r in eval_results.items()},
        "beats_static": bool(policy_mean > eval_results["static"]["retention_mean"]),
        "beats_scripted": bool(policy_mean > eval_results["scripted"]["retention_mean"]),
        "runtimes": {"train_s": train_s, "eval_s": eval_s},
    }
    write_text_lf(dest, json.dumps(result, indent=2))
    logger.info("unit %s -> %s (policy retention=%.4f)", unit, dest, policy_mean)
    return result


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--config", type=Path, required=True)
    ap.add_argument("--unit", required=True, help="'<reward>_d<density>_s<seed>'")
    ap.add_argument("--total-steps", type=int, default=None,
                     help="override total_steps (smoke jobs use a small value)")
    ap.add_argument("--wall-hours", type=float, default=None)
    ap.add_argument("--results-root", type=Path, default=None,
                    help="where <unit>/result.json lives (default runs/hpc); the smoke job uses "
                         "runs/hpc_smoke so it can never shadow an array unit (2026-09-05)")
    args = ap.parse_args()

    sweep_cfg = load_config(args.config)
    run_unit(sweep_cfg, args.unit, args.config, args.total_steps, args.wall_hours,
             results_root=args.results_root if args.results_root is not None else RESULTS_ROOT)
    return 0


if __name__ == "__main__":
    sys.exit(main())
