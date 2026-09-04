"""SPEC.md `Tests` 1-6."""

from __future__ import annotations

import math
from pathlib import Path

import pytest
from gymnasium.utils.env_checker import check_env

from lockon.core.schemas import Difficulty
from lockon.harness.episode import run_episode
from lockon.harness.eval import evaluate
from lockon.harness.gym_env import LockonGym
from lockon.harness.render import _write_gif, render_frames
from lockon.harness.scenes import SCENES
from lockon.policy.features import RewardConfig
from lockon.policy.hunters import StaticCamera
from lockon.policy.prey import ScriptedPrey

# 1. run_episode basic consistency -----------------------------------------------------------


def test_run_episode_basic_consistency() -> None:
    n = 50
    result = run_episode(Difficulty(), seed=0, hunter=StaticCamera(), prey=ScriptedPrey(), steps=n)

    assert len(result.states) == n
    assert len(result.actions) == n
    assert len(result.gt_boxes) == n
    assert len(result.tracks) == n
    assert len(result.lock) == n
    assert 0.0 <= result.retention.retention <= 1.0

    for state, gt in zip(result.states, result.gt_boxes, strict=True):
        assert (gt is None) == (not state.person_visible)


# 2. occlusion scene actually occludes (instrument proof for GIF1) ----------------------------


def test_occlusion_scene_actually_occludes() -> None:
    spec = SCENES["occlusion"]
    result = run_episode(
        spec.difficulty, spec.seed, spec.hunter, spec.prey, steps=spec.steps, scene=spec.scene
    )
    visible = [s.person_visible for s in result.states]

    lost_at = next((i for i, v in enumerate(visible) if not v), None)
    assert lost_at is not None, "occlusion scene never occludes the target"
    assert any(visible[lost_at + 1 :]), "occlusion scene never re-acquires after occlusion"


# 3. lights_cut scene kills rgb, keeps thermal (instrument proof for GIF2) --------------------


def test_lights_cut_scene_kills_rgb_keeps_thermal() -> None:
    spec = SCENES["lights_cut"]
    result = run_episode(
        spec.difficulty, spec.seed, spec.hunter, spec.prey, steps=spec.steps, scene=spec.scene
    )
    tail = result.states[60:]
    assert tail, "scene too short to exercise the post-cut tail"

    assert all(not s.channels_see["rgb"] for s in tail)
    thermal_frac = sum(1 for s in tail if s.channels_see["thermal"]) / len(tail)
    assert thermal_frac >= 0.8


# 4. LockonGym is a valid Gymnasium env --------------------------------------------------------


def test_lockon_gym_check_env_and_random_steps() -> None:
    env = LockonGym(Difficulty(), seed=0, prey=ScriptedPrey(), reward=RewardConfig())
    check_env(env, skip_render_check=True)

    obs, _info = env.reset(seed=1)
    assert obs.shape == env.observation_space.shape
    for _ in range(20):
        action = env.action_space.sample()
        obs, r, terminated, truncated, _info = env.step(action)
        assert math.isfinite(r)
        if terminated or truncated:
            obs, _info = env.reset()


# 5. evaluate() returns finite numbers ---------------------------------------------------------


def test_evaluate_returns_finite_numbers() -> None:
    static_result = evaluate("static", Difficulty(), range(3))
    assert math.isfinite(static_result["retention_mean"])
    assert math.isfinite(static_result["retention_std"])
    assert len(static_result["per_episode"]) == 3

    scripted_result = evaluate("scripted", Difficulty(), range(3))
    assert math.isfinite(scripted_result["retention_mean"])
    assert math.isfinite(scripted_result["retention_std"])


# 6. render smoke -------------------------------------------------------------------------------


@pytest.mark.render
def test_render_smoke_produces_a_gif(tmp_path: Path) -> None:
    frames = render_frames("sensor3", steps=10)
    assert len(frames) == 10
    gif_path = tmp_path / "sensor3_smoke.gif"
    _write_gif(frames, gif_path, fps=10, scale=1.0)
    assert gif_path.exists()
    assert gif_path.stat().st_size > 1024
