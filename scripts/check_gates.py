"""Gate runner (plan.md §3). `uv run python scripts/check_gates.py G0 [G1 ...] | --all`.

Each gate is a list of commands; the gate passes iff every command exits 0. Verdicts are
appended to reports/gates.json with a timestamp and the git HEAD, so a gate is never claimed
from memory.
"""

from __future__ import annotations

import argparse
import json
import subprocess
import sys
import time
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
PY = [sys.executable]

GATES: dict[str, list[list[str]]] = {
    "G0": [PY + ["-m", "mypy", "--strict", "src/lockon/core"], PY + ["-m", "pytest", "-q", "tests/test_boundaries.py", "tests/test_core.py"]],
    "G1": [PY + ["-m", "lockon.harness.bench_env", "--steps", "2000", "--min-sps", "200"]],
    "G2": [PY + ["-m", "lockon.harness.render", "--scene", "sensor3", "--gif", "reports/gifs/sensor_3ch.gif", "--steps", "30", "--scale", "0.4", "--fps", "6"], PY + ["-m", "pytest", "-q", "tests/test_sensor.py"]],
    "G3": [PY + ["-m", "pytest", "-q", "tests/test_track.py"], PY + ["-c", "import pathlib,sys; sys.exit(0 if all(pathlib.Path(p).exists() for p in ['reports/gifs/occlusion_lock.gif','reports/gifs/lights_cut_lock.gif']) else 1)"]],
    "G4": [PY + ["-m", "lockon.harness.eval", "--policy", "scripted", "--vs", "static", "--n", "20", "--seed-base", "1000", "--json", "reports/eval_scripted_vs_static.json"]],
    "G5": [PY + ["-m", "lockon.harness.eval", "--policy", "runs/ppo_local/best.zip", "--vs", "static", "--n", "20", "--seed-base", "1000", "--json", "reports/eval_ppo_vs_static.json"]],
    "G6": [PY + ["-m", "lockon.harness.sweep", "--config", "configs/sweep_local.yaml"]],
    "G7": [PY + ["-m", "lockon.demo.render_all"], PY + ["-m", "lockon.demo.viewer", "--check"]],
    "G8": [PY + ["-m", "pytest", "-q", "tests/test_license_guard.py"]],
    "G9": [PY + ["-m", "pytest", "-q"]],
}


def _head() -> str:
    try:
        return subprocess.run(["git", "rev-parse", "--short", "HEAD"], cwd=ROOT, capture_output=True, text=True, check=True).stdout.strip()
    except Exception:  # noqa: BLE001 — no commits yet is a valid state
        return "no-head"


def run_gate(name: str) -> bool:
    ok = True
    for cmd in GATES[name]:
        print(f"[{name}] $ {' '.join(cmd)}", flush=True)
        rc = subprocess.run(cmd, cwd=ROOT, check=False).returncode
        print(f"[{name}] exit {rc}", flush=True)
        ok = ok and rc == 0
        if not ok:
            break
    return ok


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("gates", nargs="*")
    ap.add_argument("--all", action="store_true")
    args = ap.parse_args()
    names = list(GATES) if args.all else args.gates
    if not names:
        ap.error("name at least one gate or pass --all")
    out = ROOT / "reports" / "gates.json"
    log: dict[str, dict[str, object]] = json.loads(out.read_text()) if out.exists() else {}
    all_ok = True
    for name in names:
        ok = run_gate(name)
        all_ok = all_ok and ok
        log[name] = {"pass": ok, "at": time.strftime("%Y-%m-%dT%H:%M:%S"), "head": _head()}
        print(f"[{name}] {'PASS' if ok else 'FAIL'}", flush=True)
    out.write_text(json.dumps(log, indent=2))
    return 0 if all_ok else 1


if __name__ == "__main__":
    sys.exit(main())
