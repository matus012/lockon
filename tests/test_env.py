"""env/SPEC.md `Tests` 1-7 — instrument proofs before any number is trusted."""

from __future__ import annotations

import dataclasses
import math

import mujoco
import numpy as np
import pytest

from lockon.core.schemas import AgentAction, Difficulty, Pose2D
from lockon.env.env import R_PERSON, Env


def _step_seq(n: int, seed: int) -> list[tuple[AgentAction, tuple[float, float]]]:
    rng = np.random.default_rng(seed)
    seq = []
    for _ in range(n):
        a = AgentAction.from_array(rng.uniform(-1.0, 1.0, size=3))
        v = (float(rng.uniform(-1.0, 1.0)), float(rng.uniform(-1.0, 1.0)))
        seq.append((a, v))
    return seq


# 1. determinism ---------------------------------------------------------------------------
def test_determinism_same_seed_identical_layout_and_trajectory() -> None:
    env_a = Env(Difficulty(), seed=7)
    env_b = Env(Difficulty(), seed=7)

    np.testing.assert_array_equal(env_a.layout.pillars, env_b.layout.pillars)
    np.testing.assert_array_equal(env_a.layout.lights, env_b.layout.lights)

    seq = _step_seq(30, seed=123)
    for action, pv in seq:
        sa = env_a.step(action, pv)
        sb = env_b.step(action, pv)
        assert sa.drone == sb.drone
        assert sa.person == sb.person
        assert sa.person_visible == sb.person_visible
        np.testing.assert_array_equal(sa.raycasts, sb.raycasts)
        assert sa.channels_alive == sb.channels_alive


# 2. projection -----------------------------------------------------------------------------
def test_projection_optical_axis_and_left_offset() -> None:
    env = Env(Difficulty(), seed=1)
    cam_pos = np.array(env.data.cam_xpos[env._cam_id])
    cam_mat = np.array(env.data.cam_xmat[env._cam_id]).reshape(3, 3)
    forward = -cam_mat[:, 2]
    right = cam_mat[:, 0]

    on_axis = cam_pos + forward * 5.0
    pixels, depth = env.project(on_axis.reshape(1, 3))
    assert depth[0] > 0
    assert pixels[0, 0] == pytest.approx(320.0, abs=1.0)
    assert pixels[0, 1] == pytest.approx(240.0, abs=1.0)

    left_point = on_axis - right * 1.0
    pixels2, _ = env.project(left_point.reshape(1, 3))
    assert pixels2[0, 0] < 320.0


# 3. line of sight ---------------------------------------------------------------------------
def test_los_pillar_occludes_then_clears() -> None:
    env = Env(Difficulty(), seed=3)
    drone_pos = np.array([0.0, 0.0, env.layout.drone_altitude])
    person_pos = np.array([5.0, 0.0, 1.05])
    env.layout = dataclasses.replace(env.layout, pillars=np.array([[2.5, 0.0, 0.6]], dtype=np.float64))
    env.model, env.data = _rebuild_model_for_pillars(env)

    occluded = env.line_of_sight(drone_pos, person_pos)
    assert occluded is False

    env.layout = dataclasses.replace(env.layout, pillars=np.zeros((0, 3), dtype=np.float64))
    env.model, env.data = _rebuild_model_for_pillars(env)
    clear = env.line_of_sight(drone_pos, person_pos)
    assert clear is True


def _rebuild_model_for_pillars(env: Env) -> tuple[object, object]:
    from lockon.env import arena

    mjcf = arena.build_mjcf(env.layout)
    model = mujoco.MjModel.from_xml_string(mjcf)
    data = mujoco.MjData(model)
    mujoco.mj_forward(model, data)
    return model, data


# 4. FOV ---------------------------------------------------------------------------------------
def test_fov_person_behind_drone_not_visible() -> None:
    env = Env(Difficulty(), seed=5)
    env.layout = dataclasses.replace(env.layout, pillars=np.zeros((0, 3), dtype=np.float64))
    env.model, env.data = _rebuild_model_for_pillars(env)
    env._cam_id = mujoco.mj_name2id(env.model, mujoco.mjtObj.mjOBJ_CAMERA, env.CAMERA)
    env._drone_body_id = mujoco.mj_name2id(env.model, mujoco.mjtObj.mjOBJ_BODY, env.DRONE_BODY)
    env._person_body_id = mujoco.mj_name2id(env.model, mujoco.mjtObj.mjOBJ_BODY, env.PERSON_BODY)
    env._drone_mocap_id = int(env.model.body_mocapid[env._drone_body_id])
    env._person_mocap_id = int(env.model.body_mocapid[env._person_body_id])

    env._drone_pose = Pose2D(0.0, 0.0, 0.0)  # facing +x
    env._person_pose = Pose2D(-5.0, 0.0, 0.0)  # directly behind the drone
    env._write_mocap()
    mujoco.mj_forward(env.model, env.data)

    st = env.state()
    assert st.person_box is None
    assert st.person_visible is False


# 5. collision -------------------------------------------------------------------------------
def test_collision_person_pushed_out_of_pillar() -> None:
    env = Env(Difficulty(), seed=9)
    px, py, hw = 0.0, 0.0, 0.8
    env.layout = dataclasses.replace(env.layout, pillars=np.array([[px, py, hw]], dtype=np.float64))
    env.model, env.data = _rebuild_model_for_pillars(env)

    env._person_pose = Pose2D(-2.0, 0.0, 0.0)
    env._drone_pose = Pose2D(-6.0, 0.0, 0.0)
    env._write_mocap()
    mujoco.mj_forward(env.model, env.data)

    # drive the person straight toward, and through, the pillar
    for _ in range(60):
        st = env.step(AgentAction.zero(), (2.2, 0.0))

    outside = abs(st.person.x - px) > hw + R_PERSON - 1e-6 or abs(st.person.y - py) > hw + R_PERSON - 1e-6
    assert outside


# 6. dropout -----------------------------------------------------------------------------------
def test_dropout_schedule_extremes() -> None:
    env_hot = Env(Difficulty(channel_dropout=1.0), seed=11)
    any_dead = False
    for action, pv in _step_seq(200, seed=99):
        st = env_hot.step(action, pv)
        if not all(st.channels_alive.values()):
            any_dead = True
    assert any_dead

    env_cold = Env(Difficulty(channel_dropout=0.0), seed=11)
    for action, pv in _step_seq(200, seed=99):
        st = env_cold.step(action, pv)
        assert all(st.channels_alive.values())


# 7. speed ---------------------------------------------------------------------------------
def test_person_max_speed_dial_extremes() -> None:
    env_lo = Env(Difficulty(prey_speed=0.0), seed=2)
    assert env_lo.person_max_speed() == pytest.approx(0.6)
    env_hi = Env(Difficulty(prey_speed=1.0), seed=2)
    assert env_hi.person_max_speed() == pytest.approx(2.2)


def test_drone_commands_are_low_pass_filtered() -> None:
    """A full-speed command from rest moves the drone ALPHA of the max in the first step and
    approaches the max geometrically (env.DRONE_CMD_ALPHA); the filter resets on reset()."""
    from lockon.env import env as env_mod

    e = Env(Difficulty(), seed=5)
    st0 = e.reset()
    st1 = e.step(AgentAction(1.0, 0.0, 0.0), (0.0, 0.0))
    d1 = math.hypot(st1.drone.x - st0.drone.x, st1.drone.y - st0.drone.y)
    assert d1 == pytest.approx(env_mod.DRONE_CMD_ALPHA * env_mod.DRONE_V_MAX * env_mod.DT, rel=0.05)
    for _ in range(30):
        e.step(AgentAction(1.0, 0.0, 0.0), (0.0, 0.0))
    # after 30 steps the command has converged (may be wall/pillar clamped, so only check the filter)
    assert float(e._drone_cmd[0]) == pytest.approx(1.0, abs=1e-3)
    e.reset()
    assert float(np.abs(e._drone_cmd).max()) == 0.0

