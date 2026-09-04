from __future__ import annotations

import math

import numpy as np
import pytest

from lockon.core import AgentAction, AgentObs, Difficulty, Track, iou_xyxy, retention


def _box(x: float) -> np.ndarray:
    return np.array([x, 0.0, x + 10.0, 10.0])


def test_iou_identity_and_disjoint() -> None:
    assert iou_xyxy(_box(0), _box(0)) == pytest.approx(1.0)
    assert iou_xyxy(_box(0), _box(20)) == 0.0
    assert iou_xyxy(_box(0), _box(5)) == pytest.approx(5 / 15)


def test_retention_perfect_tracking() -> None:
    gt = [_box(0) for _ in range(5)]
    tracks = [[Track(7, _box(0), t)] for t in range(5)]
    r = retention(gt, tracks)
    assert r.retention == 1.0 and r.target_id == 7 and r.n_loss_events == 0
    assert math.isnan(r.time_to_reacquire)


def test_retention_id_switch_is_a_loss_even_if_box_matches() -> None:
    gt = [_box(0) for _ in range(6)]
    tracks = [[Track(1, _box(0), t)] for t in range(3)] + [[Track(2, _box(0), t)] for t in range(3, 6)]
    r = retention(gt, tracks)
    assert r.target_id == 1 and r.retention == pytest.approx(0.5)
    assert r.n_loss_events == 1 and r.n_censored_losses == 1


def test_retention_reacquire_time() -> None:
    gt = [_box(0) for _ in range(6)]
    tracks = [[Track(1, _box(0), 0)], [Track(1, _box(0), 1)], [], [], [Track(1, _box(0), 4)], [Track(1, _box(0), 5)]]
    r = retention(gt, tracks)
    assert r.n_loss_events == 1 and r.time_to_reacquire == pytest.approx(2.0)
    assert r.retention == pytest.approx(4 / 6)


def test_retention_gt_none_counts_as_not_retained() -> None:
    gt = [_box(0), None, _box(0)]
    tracks = [[Track(1, _box(0), t)] for t in range(3)]
    assert retention(gt, tracks).retention == pytest.approx(2 / 3)


def test_retention_length_mismatch_raises() -> None:
    with pytest.raises(ValueError):
        retention([_box(0)], [])


def test_difficulty_bounds_and_replace() -> None:
    d = Difficulty()
    assert d.replace(prey_speed=1.0).prey_speed == 1.0
    with pytest.raises(ValueError):
        Difficulty(darkness=1.5)


def test_obs_and_action_vectors() -> None:
    obs = AgentObs(
        own=np.zeros(3), last_seen_rel=np.zeros(2), time_since_seen=0.0,
        raycasts=np.ones(16), light=1.0, channels_alive=np.ones(3),
    )
    assert obs.vector().shape == (AgentObs.size(),)
    a = AgentAction.from_array([2.0, -3.0, 0.5])
    assert (a.vx, a.vy, a.yaw_rate) == (1.0, -1.0, 0.5)
    assert AgentAction.zero().as_array().tolist() == [0.0, 0.0, 0.0]
