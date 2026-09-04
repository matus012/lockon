"""Phase B2 tests (SPEC.md `train.py`/`sweep.py`): train smoke, frame-stack ordering, sweep smoke."""

from __future__ import annotations

import json
from pathlib import Path

import numpy as np
import pytest
import yaml
from stable_baselines3 import PPO

from lockon.core.schemas import AgentObs
from lockon.harness.sweep import SweepConfig, run_sweep, write_curves, write_failure_report
from lockon.harness.train import train
from lockon.policy.hunters import PPOHunter

_TRAIN_CFG: dict[str, object] = {
    "name": "ppo_smoke",
    "seed": 0,
    "total_steps": 4096,
    "n_envs": 2,
    "frame_stack": 4,
    "n_steps": 256,
    "batch_size": 64,
    "n_epochs": 4,
    "gamma": 0.99,
    "gae_lambda": 0.95,
    "learning_rate": 0.0003,
    "ent_coef": 0.01,
    "clip_range": 0.2,
    "net_arch": [32, 32],
    "checkpoint_every": 2048,
    "eval_every": 2048,
    "eval_episodes": 2,
    "curriculum": {"start_dial": 0.3, "end_dial": 0.5, "switch_fraction": 0.4},
    "reward": {"visible": 1.0, "action_l2": 0.01, "lost_penalty": 10.0, "lost_after": 20},
    "wall_hours": 0.05,
}


@pytest.fixture
def train_config(tmp_path: Path) -> Path:
    path = tmp_path / "ppo_smoke.yaml"
    path.write_text(yaml.dump(_TRAIN_CFG), encoding="utf-8")
    return path


# (a) train smoke -------------------------------------------------------------------------------


def test_train_smoke_writes_best_and_log(train_config: Path, tmp_path: Path) -> None:
    out_dir = tmp_path / "runs" / "ppo_smoke"
    result_dir = train(str(train_config), str(out_dir), wall_hours=0.05, total_steps=4096)

    assert result_dir == out_dir
    best = out_dir / "best.zip"
    log = out_dir / "train_log.jsonl"
    assert best.exists()
    assert log.exists()

    rows = [json.loads(line) for line in log.read_text(encoding="utf-8").splitlines()]
    assert len(rows) >= 1
    for row in rows:
        assert "policy_retention_mean" in row
        assert "static_retention_mean" in row

    hunter = PPOHunter(str(best))
    hunter.reset(layout=None, seed=0)  # type: ignore[arg-type]
    obs_size = 4 * AgentObs.size()
    vector = np.zeros(obs_size, dtype=np.float64)
    action = hunter.act_vector(vector)
    assert action.as_array().shape == (3,)
    assert np.all(np.isfinite(action.as_array()))


# (b) frame-stack ordering matches VecFrameStack --------------------------------------------


def test_frame_stack_ordering_matches_vecframestack(train_config: Path, tmp_path: Path) -> None:
    out_dir = tmp_path / "runs" / "ppo_smoke_fs"
    train(str(train_config), str(out_dir), wall_hours=0.05, total_steps=4096)
    best = out_dir / "best.zip"

    model = PPO.load(str(best), device="cpu")

    rng = np.random.default_rng(0)
    obs_size = AgentObs.size()
    frame_stack = 4
    frames = [rng.uniform(-1.0, 1.0, size=obs_size).astype(np.float32) for _ in range(frame_stack)]
    stacked_vector = np.concatenate(frames).astype(np.float64)

    hunter = PPOHunter(str(best))
    hunter.reset(layout=None, seed=0)  # type: ignore[arg-type]
    hunter_action = hunter.act_vector(stacked_vector)

    model_action, _ = model.predict(stacked_vector.astype(np.float32), deterministic=True)

    np.testing.assert_allclose(hunter_action.as_array(), model_action, rtol=1e-5, atol=1e-6)


# (c) sweep smoke ---------------------------------------------------------------------------


@pytest.fixture
def sweep_config(tmp_path: Path) -> Path:
    cfg = {
        "name": "sweep_smoke",
        "n_episodes": 2,
        "seed_base": 1000,
        "values": [0.0, 1.0],
        "axes": ["occluder_density", "darkness"],
        "policies": {"static": "static", "scripted": "scripted"},
        "out_dir": str(tmp_path / "reports" / "sweep"),
        "curves_dir": str(tmp_path / "reports" / "curves"),
    }
    path = tmp_path / "sweep_smoke.yaml"
    path.write_text(yaml.dump(cfg), encoding="utf-8")
    return path


def test_sweep_smoke_writes_outputs_and_resumes(sweep_config: Path, tmp_path: Path) -> None:
    cfg = SweepConfig.from_yaml(sweep_config)

    rows = run_sweep(cfg, policy_filter=["static"], values_override=[0.0, 1.0])
    write_curves(rows, cfg)
    write_failure_report(rows, cfg)

    results_path = Path(cfg.out_dir) / "results.json"
    assert results_path.exists()
    saved_rows = json.loads(results_path.read_text(encoding="utf-8"))
    assert len(saved_rows) == len(cfg.axes) * 2  # 2 values x 1 policy x 2 axes

    curves_dir = Path(cfg.curves_dir)
    assert (curves_dir / "occluder_density.png").exists()
    assert (curves_dir / "darkness.png").exists()
    assert (curves_dir / "summary.png").exists()

    failure_report = Path(cfg.out_dir) / "failure_report.md"
    assert failure_report.exists()
    assert "occluder_density" in failure_report.read_text(encoding="utf-8")

    # Re-run: resume-safe, no new cells to compute.
    rows_again = run_sweep(cfg, policy_filter=["static"], values_override=[0.0, 1.0])
    assert len(rows_again) == len(saved_rows)
    assert {(r["axis"], r["value"], r["policy"]) for r in rows_again} == {
        (r["axis"], r["value"], r["policy"]) for r in saved_rows
    }
