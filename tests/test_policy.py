"""SPEC.md `Tests` — no MuJoCo, hand-built WorldState / ArenaLayout only."""

from __future__ import annotations

import math

import gymnasium as gym
import numpy as np
import numpy.typing as npt
import pytest
from gymnasium import spaces

from lockon.core import person_max_speed
from lockon.core.schemas import (
    CHANNELS,
    EPISODE_STEPS,
    N_RAYCASTS,
    AgentAction,
    AgentObs,
    ArenaLayout,
    Difficulty,
    Pose2D,
    WorldState,
)
from lockon.policy.features import ObsBuilder, RewardConfig, reward
from lockon.policy.hunters import ScriptedHunter, StaticCamera
from lockon.policy.ppo import PPOConfig, make_ppo
from lockon.policy.prey import ScriptedPrey

HALF_SIZE = 10.0


def _layout(pillars: npt.NDArray[np.float64] | None = None) -> ArenaLayout:
    return ArenaLayout(
        half_size=HALF_SIZE,
        pillars=pillars if pillars is not None else np.zeros((0, 3), dtype=np.float64),
        lights=np.zeros((0, 3), dtype=np.float64),
        drone_altitude=3.0,
    )


def _state(
    drone: Pose2D,
    person: Pose2D,
    person_visible: bool,
    illumination_at_person: float = 0.8,
    raycasts: npt.NDArray[np.float64] | None = None,
) -> WorldState:
    return WorldState(
        t=0,
        drone=drone,
        person=person,
        person_visible=person_visible,
        person_box=None,
        raycasts=raycasts if raycasts is not None else np.ones(N_RAYCASTS, dtype=np.float64),
        illumination_at_person=illumination_at_person,
        channels_alive=dict.fromkeys(CHANNELS, True),
        channels_see=dict.fromkeys(CHANNELS, person_visible),
    )


def _obs(
    last_seen_rel: npt.NDArray[np.float64],
    time_since_seen: float,
    raycasts: npt.NDArray[np.float64] | None = None,
) -> AgentObs:
    return AgentObs(
        own=np.array([0.0, 0.0, 0.0], dtype=np.float64),
        last_seen_rel=last_seen_rel,
        time_since_seen=time_since_seen,
        raycasts=raycasts if raycasts is not None else np.ones(N_RAYCASTS, dtype=np.float64),
        light=0.5,
        channels_alive=np.ones(len(CHANNELS), dtype=np.float64),
    )


# 1. ObsBuilder --------------------------------------------------------------------------------


def test_obs_builder_visible_then_lost_rotates_with_yaw() -> None:
    layout = _layout()
    builder = ObsBuilder(layout)

    person = Pose2D(x=5.0, y=0.0, yaw=0.0)
    drone0 = Pose2D(x=0.0, y=0.0, yaw=0.0)
    obs0 = builder.reset(_state(drone0, person, person_visible=True))

    assert obs0.time_since_seen == 0.0
    # Person is 5m along +x, drone faces +x at the origin -> last_seen_rel points at +x, y ~ 0.
    assert obs0.last_seen_rel[0] > 0.0
    assert math.isclose(obs0.last_seen_rel[1], 0.0, abs_tol=1e-9)

    drone_still = Pose2D(x=0.0, y=0.0, yaw=0.0)
    builder.step(_state(drone_still, person, person_visible=False))
    builder.step(_state(drone_still, person, person_visible=False))
    drone_turned = Pose2D(x=0.0, y=0.0, yaw=math.pi / 2.0)
    obs3 = builder.step(_state(drone_turned, person, person_visible=False))

    assert math.isclose(obs3.time_since_seen, 3.0 / EPISODE_STEPS)
    # last_seen point is still world (5, 0); drone now faces +y -> in body frame that is behind
    # to the left/right depending on rotation sign: rotate((5,0), -pi/2) = (0, -5).
    expected = np.array([0.0, -5.0]) / (2.0 * HALF_SIZE)
    assert np.allclose(obs3.last_seen_rel, expected, atol=1e-9)


# 2. reward --------------------------------------------------------------------------------------


def test_reward_visible_and_lost_penalty_fires_once_at_threshold() -> None:
    cfg = RewardConfig()
    drone = Pose2D(0.0, 0.0, 0.0)
    person = Pose2D(1.0, 0.0, 0.0)
    action = AgentAction(vx=0.1, vy=0.2, yaw_rate=0.0)

    visible_state = _state(drone, person, person_visible=True)
    r_visible = reward(visible_state, action, steps_since_seen=5, cfg=cfg)
    expected = cfg.visible - cfg.action_l2 * (0.1**2 + 0.2**2 + 0.0**2)
    assert math.isclose(r_visible, expected)

    lost_state = _state(drone, person, person_visible=False)
    r_before = reward(lost_state, action, steps_since_seen=19, cfg=cfg)
    r_at = reward(lost_state, action, steps_since_seen=20, cfg=cfg)
    r_after = reward(lost_state, action, steps_since_seen=21, cfg=cfg)

    assert math.isclose(r_before, r_after)
    assert math.isclose(r_before - r_at, cfg.lost_penalty)


# 3. StaticCamera / ScriptedHunter standoff -------------------------------------------------------


def test_static_camera_is_always_zero() -> None:
    layout = _layout()
    cam = StaticCamera()
    cam.reset(layout, seed=0)
    action = cam.act(_obs(np.array([0.3, 0.1]), time_since_seen=0.0))
    assert action.vx == 0.0
    assert action.vy == 0.0
    assert action.yaw_rate == 0.0


def test_scripted_hunter_standoff_turns_and_moves_toward_range() -> None:
    layout = _layout()
    hunter = ScriptedHunter()
    hunter.reset(layout, seed=0)

    # Target on the left (positive body-y) -> yaw_rate > 0.
    target_body_left = np.array([5.0, 3.0])
    obs_left = _obs(target_body_left / (2.0 * HALF_SIZE), time_since_seen=0.0)
    action_left = hunter.act(obs_left)
    assert action_left.yaw_rate > 0.0

    # Range (10 m) > standoff (6 m) -> move forward.
    obs_far = _obs(np.array([10.0, 0.0]) / (2.0 * HALF_SIZE), time_since_seen=0.0)
    assert hunter.act(obs_far).vx > 0.0

    # Range (3 m) < standoff (6 m) -> move backward.
    obs_near = _obs(np.array([3.0, 0.0]) / (2.0 * HALF_SIZE), time_since_seen=0.0)
    assert hunter.act(obs_near).vx < 0.0


# 4. ScriptedHunter lost mode ----------------------------------------------------------------------


def test_scripted_hunter_lost_sidesteps_blocked_forward_ray() -> None:
    layout = _layout()
    hunter = ScriptedHunter()
    hunter.reset(layout, seed=0)

    target_body = np.array([5.0, 0.0])  # straight ahead -> forward ray (index 0) is in the way.
    raycasts = np.full(N_RAYCASTS, 0.9)
    raycasts[0] = 0.05  # blocked forward
    raycasts[1] = 0.9  # freer to the (spec) left
    raycasts[-1] = 0.5

    steps_since_seen = 5
    obs = _obs(
        target_body / (2.0 * HALF_SIZE),
        time_since_seen=steps_since_seen / EPISODE_STEPS,
        raycasts=raycasts,
    )
    action = hunter.act(obs)
    assert abs(action.vy) > 1e-6


# 5. ScriptedPrey ------------------------------------------------------------------------------


def _flat_illumination(_: npt.ArrayLike) -> float:
    return 0.5


def test_scripted_prey_seeks_cover_when_aggressive() -> None:
    pillar = np.array([[5.0, 0.0, 1.0]])
    layout = _layout(pillar)
    difficulty = Difficulty(prey_aggressiveness=1.0, darkness=0.0)
    prey = ScriptedPrey()
    prey.reset(layout, difficulty, seed=0)

    drone = Pose2D(-5.0, 0.0, 0.0)
    person = Pose2D(0.0, 0.0, 0.0)
    state = _state(drone, person, person_visible=True)

    vx, vy = prey.act(state, _flat_illumination)
    pillar_dir = pillar[0, :2] - np.array([person.x, person.y])
    assert np.dot(np.array([vx, vy]), pillar_dir) > 0.0


def test_scripted_prey_wanders_below_half_speed_when_passive() -> None:
    pillar = np.array([[5.0, 0.0, 1.0]])
    layout = _layout(pillar)
    difficulty = Difficulty(prey_aggressiveness=0.0, darkness=0.0)
    prey = ScriptedPrey()
    prey.reset(layout, difficulty, seed=0)

    drone = Pose2D(-5.0, 0.0, 0.0)
    person = Pose2D(0.0, 0.0, 0.0)
    state = _state(drone, person, person_visible=True)

    for _ in range(15):
        vx, vy = prey.act(state, _flat_illumination)
        speed = math.hypot(vx, vy)
        assert speed <= 0.5 * person_max_speed(difficulty) + 1e-9


# 6. make_ppo -------------------------------------------------------------------------------------


class _DummyEnv(gym.Env[npt.NDArray[np.float32], npt.NDArray[np.float32]]):
    def __init__(self) -> None:
        super().__init__()
        self.observation_space = spaces.Box(low=-1.0, high=1.0, shape=(AgentObs.size(),), dtype=np.float32)
        self.action_space = spaces.Box(low=-1.0, high=1.0, shape=(3,), dtype=np.float32)

    def reset(
        self, *, seed: int | None = None, options: dict[str, object] | None = None
    ) -> tuple[npt.NDArray[np.float32], dict[str, object]]:
        super().reset(seed=seed)
        return self.observation_space.sample(), {}

    def step(
        self, action: npt.NDArray[np.float32]
    ) -> tuple[npt.NDArray[np.float32], float, bool, bool, dict[str, object]]:
        return self.observation_space.sample(), 0.0, False, False, {}


def test_make_ppo_builds_on_cpu_and_predicts() -> None:
    env = _DummyEnv()
    cfg = PPOConfig(n_steps=64, batch_size=32)
    model = make_ppo(env, seed=0, cfg=cfg)

    assert str(model.device) == "cpu"

    obs, _ = env.reset()
    action, _ = model.predict(obs, deterministic=True)
    assert action.shape == (3,)


def test_ppo_config_from_yaml_tolerates_extra_keys() -> None:
    cfg = PPOConfig.from_yaml("configs/ppo_local.yaml")
    assert cfg.total_steps == 6_000_000
    assert cfg.net_arch == (128, 128)
    assert cfg.extra["name"] == "ppo_local"
    assert "curriculum" in cfg.extra
    assert "reward" in cfg.extra
    assert cfg.extra["wall_hours"] == 5


def test_reward_visible_override_uses_tracker_lock() -> None:
    from lockon.policy.features import RewardConfig, reward

    drone = Pose2D(0.0, 0.0, 0.0)
    person = Pose2D(5.0, 0.0, 0.0)
    state = _state(drone, person, person_visible=True)
    cfg = RewardConfig()
    assert reward(state, AgentAction.zero(), 0, cfg) == pytest.approx(1.0)
    assert reward(state, AgentAction.zero(), 0, cfg, visible=False) == pytest.approx(0.0)

