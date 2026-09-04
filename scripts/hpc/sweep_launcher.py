"""PERUN CPU sweep launcher (plan.md §7): local in-process driver + SLURM array/smoke
sbatch emitter, plus a separate GPU sbatch template for the not-yet-implemented
learned-prey trial. Ported/trimmed from ws/100_occlusion_mot/scripts/sweep_launcher.py.

--mode local runs the IDENTICAL entrypoint (scripts/hpc/unit.py::run_unit) in-process --
one --unit for parity-testing before a submission, or every pending unit sequentially
when --unit is omitted (slow: 45 x ~est_h_per_unit; the smoke/parity use is --unit).
--mode slurm only EMITS submit.sbatch + submit_smoke.sbatch (never submits) and prints
the CPU-h / GPU-h budget table.

Usage:
  uv run python scripts/hpc/sweep_launcher.py --mode local --unit base_d0.5_s0
  uv run python scripts/hpc/sweep_launcher.py --mode slurm --config configs/hpc_sweep.yaml
  uv run python scripts/hpc/sweep_launcher.py --mode slurm --config configs/hpc_sweep.yaml --gpu
"""
from __future__ import annotations

import argparse
import logging
import sys
from pathlib import Path
from typing import Any

SCRIPTS_HPC = Path(__file__).resolve().parent
sys.path.insert(0, str(SCRIPTS_HPC))

from sweep_common import (
    RESULTS_ROOT,
    ROOT,
    budget_table,
    enumerate_units,
    format_budget_table,
    hms,
    load_config,
    parse_unit,
    posix_relpath,
    result_path,
    wall_seconds,
    write_text_lf,
)

logging.basicConfig(level=logging.INFO, format="%(asctime)s %(message)s", datefmt="%H:%M:%S",
                     stream=sys.stdout)
logger = logging.getLogger("sweep_launcher")

SWEEP_OUT = RESULTS_ROOT  # runs/hpc/


def run_local(cfg: dict[str, Any], config_path: Path, unit: str | None) -> None:
    from unit import run_unit  # local import: keeps CLI startup light for --mode slurm

    if unit is not None:
        run_unit(cfg, unit, config_path, total_steps=None, wall_hours=None)
        return

    units = enumerate_units(cfg)
    pending = [u for u in units if not result_path(u).exists()]
    logger.info("local sweep '%s': %d/%d units pending (sequential, in-process)",
                cfg["name"], len(pending), len(units))
    for u in pending:
        run_unit(cfg, u, config_path, total_steps=None, wall_hours=None)


def _slurm_header(cfg: dict[str, Any], job_name: str, time_limit: str, array: str | None,
                   cpus: int, mem: str) -> str:
    slurm = cfg["slurm"]
    lines = [
        "#!/bin/bash",
        f"#SBATCH --job-name={job_name}",
        f"#SBATCH --partition={slurm.get('partition', '<FILL-PERUN-PARTITION>')}",
        f"#SBATCH --account={slurm.get('account', '<FILL-PERUN-ACCOUNT>')}",
        f"#SBATCH --time={time_limit}",
    ]
    if array:
        lines.append(f"#SBATCH --array={array}")
    stem = "%x_%A_%a" if array else "%x_%j"
    lines += [
        f"#SBATCH --cpus-per-task={cpus}",
        f"#SBATCH --mem={mem}",
        f"#SBATCH --output=runs/hpc/logs/{stem}.out",
        f"#SBATCH --error=runs/hpc/logs/{stem}.err",
    ]
    return "\n".join(lines)


_PREAMBLE = """
set -euo pipefail

# submit FROM the repo root: every path in this script (and the entrypoint) is
# repo-root-relative by design.
if [ ! -f scripts/hpc/unit.py ]; then
  echo "ERROR: submit this script from the repo root (scripts/hpc/unit.py not found)" >&2
  exit 2
fi

source scripts/hpc/hpc_env.sh
"""


def render_sbatch(cfg: dict[str, Any], config_path: Path) -> str:
    """Emit (never submit) the CPU array. Resume-safety: a task whose result JSON
    already exists exits 0 immediately (same skip rule as run_local's `pending`
    filter). Per-unit `timeout` bounds a hung unit to its own class budget."""
    units = enumerate_units(cfg)
    table = budget_table(cfg)
    slurm = cfg["slurm"]
    est_h = float(cfg["est_h_per_unit"])
    time_limit = hms(est_h)
    timeout_arg = f"{wall_seconds(est_h)}s"
    array_range = f"0-{len(units) - 1}%{slurm.get('max_concurrent', 8)}"

    unit_lines = "\n".join(f'  "{u}"' for u in units)
    result_lines = "\n".join(f'  "{posix_relpath(result_path(u))}"' for u in units)
    config_posix = posix_relpath(config_path)

    header = _slurm_header(cfg, f"lockon_sweep_{cfg['name']}", time_limit, array_range,
                            slurm.get("cpus_per_task", 8), slurm.get("mem", "16G"))
    budget_note = (
        f"# budget: {table['n_units']} CPU units x {est_h} h/unit = {table['cpu_h']} CPU-h "
        f"(GPU-h separate, see the GPU template) vs {table['ceiling_h']} h ceiling"
    )
    return f"""{header}
{budget_note}
{_PREAMBLE}
UNITS=(
{unit_lines}
)
RESULTS=(
{result_lines}
)

UNIT="${{UNITS[$SLURM_ARRAY_TASK_ID]}}"
RESULT="${{RESULTS[$SLURM_ARRAY_TASK_ID]}}"

if [ -f "$RESULT" ]; then
  echo "skip $UNIT (result exists: $RESULT)"
  exit 0
fi

echo "unit=$UNIT node=$(hostname)"
exec timeout --signal=TERM --kill-after=120 {timeout_arg} \\
  "$PYTHON" scripts/hpc/unit.py --config {config_posix} --unit "$UNIT"
"""


def render_smoke_sbatch(cfg: dict[str, Any], config_path: Path) -> str:
    """One unit, total_steps from cfg['smoke_total_steps'] -- proves the offline env,
    the train.py CLI contract, and the eval path before the 45-unit array commits
    CPU-h. Runs before the array (HPC_RUNBOOK.md step 6, --dependency=afterok)."""
    units = enumerate_units(cfg)
    smoke_unit = units[0]
    smoke_steps = cfg.get("smoke_total_steps", 20_000)
    slurm = cfg["slurm"]
    smoke_time = cfg.get("time_smoke", "00:30:00")
    config_posix = posix_relpath(config_path)
    header = _slurm_header(cfg, f"lockon_smoke_{cfg['name']}", smoke_time, None,
                            slurm.get("cpus_per_task", 8), slurm.get("mem", "16G"))
    return f"""{header}
{_PREAMBLE}
echo "=== smoke: unit={smoke_unit} total_steps={smoke_steps} ==="
"$PYTHON" scripts/hpc/unit.py --config {config_posix} --unit {smoke_unit} \\
  --total-steps {smoke_steps}
echo "SMOKE OK -- safe to submit runs/hpc/submit.sbatch"
"""


def render_gpu_sbatch(cfg: dict[str, Any]) -> str:
    """GPU template for the learned-prey trial (plan.md §7). `lockon.harness.train_prey`
    does NOT EXIST YET (harness/SPEC.md places it after step 7) -- this is a placeholder
    command marked clearly; do not submit until it lands. gpu_short partition, one GPU,
    a hard 20 h `timeout` (plan.md §7 GPU cap)."""
    gpu = cfg.get("gpu_slurm", {})
    header = "\n".join([
        "#!/bin/bash",
        f"#SBATCH --job-name=lockon_prey_{cfg['name']}",
        f"#SBATCH --partition={gpu.get('partition', '<FILL-PERUN-PARTITION>')}",
        f"#SBATCH --account={gpu.get('account', '<FILL-PERUN-ACCOUNT>')}",
        f"#SBATCH --time={hms(float(gpu.get('time_limit_h', 20)), margin=1.0)}",
        f"#SBATCH --gres={gpu.get('gres', 'gpu:1')}",
        f"#SBATCH --cpus-per-task={gpu.get('cpus_per_task', 8)}",
        f"#SBATCH --mem={gpu.get('mem', '32G')}",
        "#SBATCH --output=runs/hpc/logs/%x_%j.out",
        "#SBATCH --error=runs/hpc/logs/%x_%j.err",
    ])
    timeout_s = wall_seconds(float(gpu.get("time_limit_h", 20)), margin=1.0)
    return f"""{header}
{_PREAMBLE}
export MUJOCO_GL="${{MUJOCO_GL:-egl}}"

echo "=== GPU probe (CLAUDE.md: CPU fallback on GPU-intended work = STOP) ==="
"$PYTHON" - <<'PY'
import torch
assert torch.cuda.is_available(), "CUDA unavailable -- do NOT submit this job"
dev = torch.device("cuda")
x = torch.ones(1024, 1024, device=dev) @ torch.ones(1024, 1024, device=dev)
assert x.device.type == "cuda", x.device
print(f"OK  {{torch.cuda.get_device_name(0)}}  torch={{torch.__version__}}  "
      f"cuda={{torch.version.cuda}}")
PY

echo "=== NOT YET IMPLEMENTED: lockon.harness.train_prey does not exist ==="
echo "placeholder command -- do not submit until harness/SPEC.md's train_prey.py lands:"
exec timeout --signal=TERM --kill-after=120 {timeout_s}s \\
  "$PYTHON" -m lockon.harness.train_prey --config configs/prey_hpc.yaml --out runs/hpc_prey
"""


def run_slurm(cfg: dict[str, Any], config_path: Path) -> None:
    out_dir = ROOT / "runs" / "hpc"
    (out_dir / "logs").mkdir(parents=True, exist_ok=True)

    write_text_lf(out_dir / "submit.sbatch", render_sbatch(cfg, config_path))
    write_text_lf(out_dir / "submit_smoke.sbatch", render_smoke_sbatch(cfg, config_path))
    write_text_lf(out_dir / "submit_gpu_prey.sbatch", render_gpu_sbatch(cfg))
    logger.info("emitted (NOT submitted): %s", out_dir / "submit_smoke.sbatch")
    logger.info("emitted (NOT submitted): %s", out_dir / "submit.sbatch")
    logger.info("emitted (NOT submitted, placeholder -- train_prey.py does not exist yet): %s",
                 out_dir / "submit_gpu_prey.sbatch")

    table = budget_table(cfg)
    logger.info("\n%s", format_budget_table(table))
    for key, val in cfg["slurm"].items():
        if isinstance(val, str) and val.startswith("<FILL-"):
            logger.warning("slurm.%s is STILL a placeholder (%s) -- sbatch will be rejected",
                            key, val)


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--config", type=Path, default=ROOT / "configs" / "hpc_sweep.yaml")
    ap.add_argument("--mode", choices=["local", "slurm"], default="local")
    ap.add_argument("--unit", default=None, help="run exactly one unit in-process (local mode)")
    args = ap.parse_args()

    cfg = load_config(args.config)
    for u in enumerate_units(cfg):
        parse_unit(u)  # fail fast on a config that would produce unparsable units

    if args.mode == "local":
        run_local(cfg, args.config, args.unit)
    else:
        run_slurm(cfg, args.config)
    return 0


if __name__ == "__main__":
    sys.exit(main())
