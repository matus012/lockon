"""Degradation-curve sweep CLI (SPEC.md `sweep.py`, plan.md step 8, gate G6).

    uv run python -m lockon.harness.sweep --config configs/sweep_local.yaml \\
        [--policies static,scripted] [--n N] [--values 0.0,0.25,...]

axes x values x policies x n_episodes, other dials at 0.5 (plan.md §9 D5). Resume-safe by
(axis, value, policy) rows in `reports/sweep/results.json`. Uses `multiprocessing.Pool` (spawn)
over cells so all CPU cores are used; each cell runs `evaluate()` for one policy.
"""

from __future__ import annotations

import argparse
import json
import logging
import multiprocessing as mp
import sys
from collections import Counter
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import yaml

from lockon.core.schemas import Difficulty
from lockon.harness.eval import evaluate

logger = logging.getLogger(__name__)

_AXIS_FIELDS = (
    "occluder_density",
    "darkness",
    "prey_speed",
    "prey_aggressiveness",
    "channel_dropout",
)


@dataclass(frozen=True)
class SweepConfig:
    name: str
    n_episodes: int
    seed_base: int
    values: list[float]
    axes: list[str]
    policies: dict[str, str]
    out_dir: str
    curves_dir: str

    @classmethod
    def from_yaml(cls, path: str | Path) -> SweepConfig:
        with open(path, encoding="utf-8") as f:
            raw = yaml.safe_load(f)
        return cls(
            name=raw["name"],
            n_episodes=int(raw["n_episodes"]),
            seed_base=int(raw["seed_base"]),
            values=[float(v) for v in raw["values"]],
            axes=list(raw["axes"]),
            policies=dict(raw["policies"]),
            out_dir=raw["out_dir"],
            curves_dir=raw["curves_dir"],
        )


def _cell_difficulty(axis: str, value: float) -> Difficulty:
    kwargs = dict.fromkeys(_AXIS_FIELDS, 0.5)
    kwargs[axis] = value
    return Difficulty(**kwargs)


def _run_cell(
    axis: str, value: float, policy_name: str, policy_ref: str, seeds: list[int]
) -> dict[str, Any]:
    difficulty = _cell_difficulty(axis, value)
    result = evaluate(policy_ref, difficulty, seeds)
    return {
        "axis": axis,
        "value": value,
        "policy": policy_name,
        "retention_mean": result["retention_mean"],
        "retention_std": result["retention_std"],
        "ttr_mean": result["ttr_mean"],
        "n": len(seeds),
        "loss_cause_histogram": result["loss_cause_histogram"],
    }


def _cell_key(row: dict[str, Any]) -> tuple[str, float, str]:
    return (row["axis"], row["value"], row["policy"])


def _load_results(path: Path) -> list[dict[str, Any]]:
    if path.exists():
        return list(json.loads(path.read_text(encoding="utf-8")))
    return []


def _save_results(path: Path, rows: list[dict[str, Any]]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(rows, indent=2), encoding="utf-8")


def run_sweep(
    cfg: SweepConfig,
    *,
    policy_filter: list[str] | None = None,
    values_override: list[float] | None = None,
) -> list[dict[str, Any]]:
    values = values_override if values_override is not None else cfg.values
    policy_names = policy_filter if policy_filter is not None else list(cfg.policies)

    active_policies: dict[str, str] = {}
    for name in policy_names:
        ref = cfg.policies[name]
        if name == "ppo" and not (ref in ("static", "scripted") or Path(ref).exists()):
            logger.warning("skipping policy %r: %s does not exist", name, ref)
            print(f"WARNING: skipping policy {name!r}: {ref} does not exist")
            continue
        active_policies[name] = ref

    results_path = Path(cfg.out_dir) / "results.json"
    rows = _load_results(results_path)
    done = {_cell_key(r) for r in rows}

    seeds = [cfg.seed_base + i for i in range(cfg.n_episodes)]
    pending = [
        (axis, value, name, ref, seeds)
        for axis in cfg.axes
        for value in values
        for name, ref in active_policies.items()
        if (axis, value, name) not in done
    ]
    logger.info("sweep: %d cells total, %d already done, %d pending", len(done) + len(pending), len(done), len(pending))

    if pending:
        ctx = mp.get_context("spawn")
        with ctx.Pool(processes=mp.cpu_count()) as pool:
            for row in pool.starmap(_run_cell, pending):
                rows.append(row)
                _save_results(results_path, rows)
                logger.info(
                    "cell done: axis=%s value=%s policy=%s retention_mean=%.4f",
                    row["axis"],
                    row["value"],
                    row["policy"],
                    row["retention_mean"],
                )

    return rows


def _plot_axis(ax: Any, rows: list[dict[str, Any]], axis: str, policies: list[str]) -> None:
    for policy in policies:
        pts = sorted(
            (r["value"], r["retention_mean"] * 100.0, r["retention_std"] * 100.0)
            for r in rows
            if r["axis"] == axis and r["policy"] == policy
        )
        if not pts:
            continue
        xs = [p[0] for p in pts]
        ys = [p[1] for p in pts]
        es = [p[2] for p in pts]
        ax.errorbar(xs, ys, yerr=es, marker="o", label=policy, capsize=3)
    ax.set_xlabel(axis)
    ax.set_ylabel("lock retention (%)")
    ax.set_title(axis)
    ax.legend()
    ax.grid(True, alpha=0.3)


def write_curves(rows: list[dict[str, Any]], cfg: SweepConfig) -> None:
    curves_dir = Path(cfg.curves_dir)
    curves_dir.mkdir(parents=True, exist_ok=True)
    axes = sorted({r["axis"] for r in rows})
    policies = sorted({r["policy"] for r in rows})

    for axis in axes:
        fig, ax = plt.subplots(figsize=(6, 4))
        _plot_axis(ax, rows, axis, policies)
        fig.tight_layout()
        fig.savefig(curves_dir / f"{axis}.png", dpi=120)
        plt.close(fig)

    n = len(axes)
    ncols = min(3, n) if n > 0 else 1
    nrows = (n + ncols - 1) // ncols if n > 0 else 1
    fig, axs = plt.subplots(nrows, ncols, figsize=(5 * ncols, 4 * nrows), squeeze=False)
    for i, axis in enumerate(axes):
        _plot_axis(axs[i // ncols][i % ncols], rows, axis, policies)
    for j in range(n, nrows * ncols):
        axs[j // ncols][j % ncols].axis("off")
    fig.tight_layout()
    fig.savefig(curves_dir / "summary.png", dpi=120)
    plt.close(fig)
    logger.info("wrote %d axis curves + summary.png to %s", len(axes), curves_dir)


def write_failure_report(rows: list[dict[str, Any]], cfg: SweepConfig) -> None:
    out_path = Path(cfg.out_dir) / "failure_report.md"
    axes = sorted({r["axis"] for r in rows})
    policies = sorted({r["policy"] for r in rows})

    lines = ["# Sweep failure report", ""]
    for axis in axes:
        lines.append(f"## {axis}")
        for policy in policies:
            axis_rows = [r for r in rows if r["axis"] == axis and r["policy"] == policy]
            if not axis_rows:
                continue
            worst = min(axis_rows, key=lambda r: r["retention_mean"])
            causes: Counter[str] = Counter()
            for r in axis_rows:
                causes.update(r["loss_cause_histogram"])
            dominant = causes.most_common(1)[0][0] if causes else "none"
            ttrs = [r["ttr_mean"] for r in axis_rows if r["ttr_mean"] == r["ttr_mean"]]  # drop nan
            mean_ttr = sum(ttrs) / len(ttrs) if ttrs else float("nan")
            lines.append(
                f"- **{policy}**: worst point value={worst['value']:.2f} "
                f"retention_mean={worst['retention_mean']:.4f} "
                f"(dominant loss cause: {dominant}, mean time-to-reacquire: {mean_ttr:.2f} steps)"
            )
        lines.append("")

    out_path.parent.mkdir(parents=True, exist_ok=True)
    out_path.write_text("\n".join(lines), encoding="utf-8")
    logger.info("wrote %s", out_path)


def main(argv: list[str] | None = None) -> int:
    logging.basicConfig(level=logging.INFO)
    parser = argparse.ArgumentParser(description="lockon.harness degradation-curve sweep")
    parser.add_argument("--config", required=True)
    parser.add_argument("--policies", type=str, default=None, help="comma-separated policy names")
    parser.add_argument("--n", type=int, default=None, help="episodes per cell (overrides config)")
    parser.add_argument("--values", type=str, default=None, help="comma-separated dial values")
    args = parser.parse_args(argv)

    cfg = SweepConfig.from_yaml(args.config)
    if args.n is not None:
        cfg = SweepConfig(**{**cfg.__dict__, "n_episodes": args.n})
    policy_filter = args.policies.split(",") if args.policies is not None else None
    values_override = [float(v) for v in args.values.split(",")] if args.values is not None else None

    rows = run_sweep(cfg, policy_filter=policy_filter, values_override=values_override)
    write_curves(rows, cfg)
    write_failure_report(rows, cfg)
    print(f"sweep complete: {len(rows)} cells; results -> {Path(cfg.out_dir) / 'results.json'}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
