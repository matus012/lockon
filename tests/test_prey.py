"""SPEC_prey.md `Tests` 1-5. CPU, tiny: n_envs 2, <=2048 steps."""

from __future__ import annotations

import math

import numpy as np
import pytest
from gymnasium.utils.env_checker import check_env

from lockon.core.schemas import Difficulty, Pose2D, WorldState
from lockon.harness.episode import run_episode
from lockon.harness.eval import evaluate, resolve_prey
from lockon.harness.prey_gym import PreyGym
from lockon.harness.train_prey import train_prey
from lockon.policy.hunters import ScriptedHunter, StaticCamera
from lockon.policy.prey_learned import LearnedPrey, PreyObsBuilder, PreyRewardConfig, prey_reward


def _state(person_visible: bool) -> WorldState:
    return WorldState(
        t=0,
        drone=Pose2D(1.0, 0.0, 0.0),
        person=Pose2D(0.0, 0.0, 0.0),
        person_visible=person_visible,
        person_box=None,
        raycasts=np.ones(16, dtype=np.float64),
        illumination_at_person=0.5,
    )


# 1. PreyGym passes check_env; 10 random steps -------------------------------------------------


def test_prey_gym_check_env_and_random_steps() -> None:
    env = PreyGym(
        Difficulty(), seed=0, hunter=ScriptedHunter(), hunter_frame_stack=1, reward=PreyRewardConfig()
    )
    check_env(env, skip_render_check=True)

    obs, _info = env.reset(seed=1)
    assert obs.shape == env.observation_space.shape
    for _ in range(10):
        action = env.action_space.sample()
        obs, r, terminated, truncated, _info = env.step(action)
        assert math.isfinite(r)
        if terminated or truncated:
            obs, _info = env.reset()


# 2. reward: visible -1, hidden 0, transition bonus +0.5 exactly once --------------------------


def test_prey_reward_visible_hidden_transition() -> None:
    cfg = PreyRewardConfig()
    zero_action = np.zeros(2, dtype=np.float64)

    visible_state = _state(person_visible=True)
    r_visible = prey_reward(visible_state, zero_action, was_visible=True, cfg=cfg)
    assert r_visible == pytest.approx(-1.0)

    hidden_state = _state(person_visible=False)
    r_hidden = prey_reward(hidden_state, zero_action, was_visible=False, cfg=cfg)
    assert r_hidden == pytest.approx(0.0)

    r_transition = prey_reward(hidden_state, zero_action, was_visible=True, cfg=cfg)
    assert r_transition == pytest.approx(0.0 + 0.5)

    r_stay_visible = prey_reward(visible_state, zero_action, was_visible=True, cfg=cfg)
    assert r_stay_visible == pytest.approx(-1.0)  # no transition bonus while still visible


# 3. train_prey smoke: best.zip + sidecar; LearnedPrey loads it and runs in run_episode ---------


def test_train_prey_smoke_and_learned_prey_runs(tmp_path) -> None:  # type: ignore[no-untyped-def]
    cfg = {
        "seed": 0,
        "total_steps": 2048,
        "n_envs": 2,
        "frame_stack": 4,
        "n_steps": 128,
        "batch_size": 32,
        "n_epochs": 2,
        "checkpoint_every": 2048,
        "eval_every": 2048,
        "eval_episodes": 2,
        "net_arch": [16, 16],
        "reward": {"visible_penalty": 1.0, "action_l2": 0.01, "transition_bonus": 0.5},
        "wall_hours": 0.05,
    }
    cfg_path = tmp_path / "prey_smoke.yaml"
    import yaml

    cfg_path.write_text(yaml.dump(cfg), encoding="utf-8")

    out_dir = tmp_path / "runs" / "prey_smoke"
    result_dir = train_prey(
        str(cfg_path), str(out_dir), "scripted", device="cpu", wall_hours=0.05, total_steps=2048, n_envs=2
    )
    assert result_dir == out_dir

    best = out_dir / "best.zip"
    sidecar = out_dir / "best.zip.prey.json"
    assert best.exists()
    assert sidecar.exists()

    prey = LearnedPrey(str(best))
    result = run_episode(Difficulty(), seed=0, hunter=StaticCamera(), prey=prey, steps=20)
    assert len(result.states) == 20


# 4. eval --prey <zip> resolves and runs (n=2) --------------------------------------------------


def test_eval_prey_zip_resolves_and_runs(tmp_path) -> None:  # type: ignore[no-untyped-def]
    cfg = {
        "seed": 0,
        "total_steps": 2048,
        "n_envs": 2,
        "frame_stack": 4,
        "n_steps": 128,
        "batch_size": 32,
        "n_epochs": 2,
        "checkpoint_every": 2048,
        "eval_every": 2048,
        "eval_episodes": 2,
        "net_arch": [16, 16],
        "wall_hours": 0.05,
    }
    cfg_path = tmp_path / "prey_smoke.yaml"
    import yaml

    cfg_path.write_text(yaml.dump(cfg), encoding="utf-8")
    out_dir = tmp_path / "runs" / "prey_smoke2"
    train_prey(str(cfg_path), str(out_dir), "scripted", device="cpu", wall_hours=0.05, total_steps=2048, n_envs=2)
    best = out_dir / "best.zip"

    prey = resolve_prey(str(best))
    assert isinstance(prey, LearnedPrey)

    result = evaluate("scripted", Difficulty(), range(2), prey=str(best))
    assert len(result["per_episode"]) == 2
    assert math.isfinite(result["retention_mean"])


# 5. frame-stack ordering for the prey mirrors the hunter's -------------------------------------


def test_prey_frame_stack_ordering_matches_vecframestack() -> None:
    from stable_baselines3.common.vec_env import DummyVecEnv, VecFrameStack

    def make() -> PreyGym:
        return PreyGym(
            Difficulty(),
            seed=0,
            hunter=ScriptedHunter(),
            hunter_frame_stack=1,
            reward=PreyRewardConfig(),
        )

    raw_env = DummyVecEnv([make])
    stk_env = VecFrameStack(DummyVecEnv([make]), 4)
    raw_env.env_method("reset", seed=77)
    stk_env.env_method("reset", seed=77)
    raw = [raw_env.reset()[0]]
    stacked = [stk_env.reset()[0]]
    rng = np.random.default_rng(3)
    for _ in range(6):
        a = rng.uniform(-1, 1, size=(1, 2)).astype(np.float32)
        raw.append(raw_env.step(a)[0][0])
        stacked.append(stk_env.step(a)[0][0])
    d = PreyObsBuilder.size()
    for k, s in enumerate(stacked):
        hist = [np.zeros(d, dtype=np.float32)] * max(0, 3 - k) + raw[max(0, k - 3) : k + 1]
        expected = np.concatenate(hist[-4:])
        assert np.allclose(s, expected), f"stack mismatch at step {k}"


def test_prey_reward_seen_override_uses_tracker_lock() -> None:
    """Row 20: the evader is paid for breaking the LOCK, not geometric visibility."""
    import numpy as np

    from lockon.core import Pose2D, WorldState
    from lockon.policy.prey_learned import PreyRewardConfig, prey_reward

    st = WorldState(
        t=0, drone=Pose2D(0.0, 0.0, 0.0), person=Pose2D(5.0, 0.0, 0.0),
        person_visible=True, person_box=np.array([1.0, 1.0, 2.0, 2.0]),
        raycasts=np.ones(16), illumination_at_person=1.0, in_fov=True, unoccluded=True,
    )
    cfg = PreyRewardConfig()
    a = np.zeros(2)
    # visible AND locked -> penalised; visible but lock broken -> not penalised, and the
    # seen->unseen transition bonus fires off the lock, not the geometry
    assert prey_reward(st, a, was_visible=True, cfg=cfg) == pytest.approx(-cfg.visible_penalty)
    assert prey_reward(st, a, was_visible=True, cfg=cfg, seen=False) == pytest.approx(cfg.transition_bonus)

