"""Phase B2 tests (SPEC.md `train.py`/`sweep.py`): train smoke, frame-stack ordering, sweep smoke."""

from __future__ import annotations

import json
from pathlib import Path

import numpy as np
import pytest
import yaml

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


def test_frame_stack_ordering_matches_vecframestack() -> None:
    """SB3's VecFrameStack = zero-filled history, newest last. The harness deque in
    `run_episode` replicates that; assert it against SB3 on a real env with identical seeds and
    actions (a same-vector predict comparison could not fail under a reversed stack)."""
    from stable_baselines3.common.vec_env import DummyVecEnv, VecFrameStack

    from lockon.core import AgentObs, Difficulty
    from lockon.harness.gym_env import LockonGym
    from lockon.policy.features import RewardConfig
    from lockon.policy.prey import ScriptedPrey

    def make() -> LockonGym:
        return LockonGym(Difficulty(), seed=0, prey=ScriptedPrey(), reward=RewardConfig())

    raw_env = DummyVecEnv([make])
    stk_env = VecFrameStack(DummyVecEnv([make]), 4)
    raw_env.env_method("reset", seed=77)
    stk_env.env_method("reset", seed=77)
    raw = [raw_env.reset()[0]]
    stacked = [stk_env.reset()[0]]
    rng = np.random.default_rng(3)
    for _ in range(6):
        a = rng.uniform(-1, 1, size=(1, 3)).astype(np.float32)
        raw.append(raw_env.step(a)[0][0])
        stacked.append(stk_env.step(a)[0][0])
    d = AgentObs.size()
    for k, s in enumerate(stacked):
        hist = [np.zeros(d, dtype=np.float32)] * max(0, 3 - k) + raw[max(0, k - 3) : k + 1]
        expected = np.concatenate(hist[-4:])
        assert np.allclose(s, expected), f"stack mismatch at step {k}"



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
