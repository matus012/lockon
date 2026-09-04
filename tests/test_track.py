from __future__ import annotations

import numpy as np
import pytest

from lockon.core import CHANNELS, Detection, retention
from lockon.track import LockTracker, NoiseConfig, NoiseInjector

BOX_W = 40.0
BOX_H = 90.0


def _box(x: float, y: float = 0.0) -> np.ndarray:
    return np.array([x, y, x + BOX_W, y + BOX_H], dtype=np.float64)


def walk_boxes(n: int, x0: float = 0.0, dx: float = 2.0) -> list[dict[str, Detection | None]]:
    """A 40x90 px box translating dx px/step; GT dict for rgb+depth+thermal."""
    scene: list[dict[str, Detection | None]] = []
    for t in range(n):
        box = _box(x0 + dx * t)
        scene.append({c: Detection(box=box.copy(), score=1.0, channel=c, t=t) for c in CHANNELS})
    return scene


def occlusion_scene() -> list[dict[str, Detection | None]]:
    """120 steps; box visible 0-39, GT None on ALL channels 40-59 (behind a pillar), visible
    again 60-119 continuing the same motion."""
    n = 120
    base = walk_boxes(n)
    scene: list[dict[str, Detection | None]] = []
    for t in range(n):
        if 40 <= t < 60:
            scene.append({c: None for c in CHANNELS})
        else:
            scene.append(base[t])
    return scene


def lights_cut_scene() -> list[dict[str, Detection | None]]:
    """120 steps; rgb channel GT None from step 50 on (rgb dead / dark), depth and thermal keep
    reporting."""
    n = 120
    base = walk_boxes(n)
    scene: list[dict[str, Detection | None]] = []
    for t in range(n):
        frame = dict(base[t])
        if t >= 50:
            frame["rgb"] = None
        scene.append(frame)
    return scene


def _run(
    scene: list[dict[str, Detection | None]], noise: NoiseInjector, tracker: LockTracker
) -> tuple[list[np.ndarray | None], list[list]]:
    gt_boxes: list[np.ndarray | None] = []
    all_tracks: list[list] = []
    for t, gt in enumerate(scene):
        detections = noise(gt, t)
        tracks, _status = tracker.update(detections, t)
        all_tracks.append(tracks)
        # ground truth box for retention: any channel that still sees the target (they all agree).
        seen = next((d.box for d in gt.values() if d is not None), None)
        gt_boxes.append(seen)
    return gt_boxes, all_tracks


def test_noise_injector_deterministic_and_rates() -> None:
    n = 2000
    scene = walk_boxes(n, dx=0.0)
    cfg = NoiseConfig(miss_rate=0.3, jitter_px=0.0, seed=0)

    injector_a = NoiseInjector(cfg)
    outputs_a = [injector_a(gt, t) for t, gt in enumerate(scene)]
    injector_b = NoiseInjector(cfg)
    outputs_b = [injector_b(gt, t) for t, gt in enumerate(scene)]

    for out_a, out_b in zip(outputs_a, outputs_b, strict=True):
        assert len(out_a) == len(out_b)
        for det_a, det_b in zip(out_a, out_b, strict=True):
            assert det_a.channel == det_b.channel
            assert det_a.score == det_b.score
            assert np.array_equal(det_a.box, det_b.box)

    total_possible = n * len(CHANNELS)
    total_delivered = sum(len(out) for out in outputs_a)
    drop_fraction = 1.0 - total_delivered / total_possible
    assert drop_fraction == pytest.approx(0.3, abs=0.05)


def test_latency_delays_delivery() -> None:
    cfg = NoiseConfig(miss_rate=0.0, jitter_px=0.0, latency_steps=2, seed=0)
    injector = NoiseInjector(cfg)
    scene = walk_boxes(5)

    out0 = injector(scene[0], 0)
    out1 = injector(scene[1], 1)
    out2 = injector(scene[2], 2)

    assert out0 == []
    assert out1 == []
    assert len(out2) == len(CHANNELS)
    assert all(det.t == 0 for det in out2)


def test_id_held_through_occlusion() -> None:
    scene = occlusion_scene()
    cfg = NoiseConfig.from_dial(0.3)
    noise = NoiseInjector(cfg)
    tracker = LockTracker(lost_track_buffer=30)

    gt_boxes, all_tracks = _run(scene, noise, tracker)
    result = retention(gt_boxes, all_tracks)

    assert result.retention >= 0.75
    tail_ids = {tr.track_id for tracks in all_tracks[60:120] for tr in tracks}
    assert tail_ids == {result.target_id}

    tail_gt = gt_boxes[60:120]
    tail_tracks = all_tracks[60:120]
    tail_result = retention(tail_gt, tail_tracks)
    assert tail_result.retention >= 0.95


def test_id_held_through_channel_dropout() -> None:
    scene = lights_cut_scene()
    cfg = NoiseConfig.from_dial(0.3)
    noise = NoiseInjector(cfg)
    tracker = LockTracker(lost_track_buffer=30)

    gt_boxes, all_tracks = _run(scene, noise, tracker)
    result = retention(gt_boxes, all_tracks)

    assert result.retention >= 0.9
    all_ids = {tr.track_id for tracks in all_tracks for tr in tracks}
    assert all_ids == {result.target_id}


def test_lock_status_counts() -> None:
    scene = occlusion_scene()
    cfg = NoiseConfig(miss_rate=0.0, jitter_px=0.0, seed=0)
    noise = NoiseInjector(cfg)
    tracker = LockTracker(lost_track_buffer=30)

    statuses = []
    for t, gt in enumerate(scene):
        detections = noise(gt, t)
        _tracks, status = tracker.update(detections, t)
        statuses.append(status)

    for s in statuses:
        if not s.locked:
            assert s.track_id is None

    # steps_since_lock increments while unlocked, resets to 0 on re-lock.
    run = 0
    for s in statuses:
        if s.locked:
            assert s.steps_since_lock == 0
            run = 0
        else:
            run += 1
            assert s.steps_since_lock == run

    # occlusion window (all channels dropped) eventually produces an unlocked step.
    assert any(not s.locked for s in statuses[40:60])


def test_fusion_yields_single_track() -> None:
    scene = walk_boxes(50)
    tracker = LockTracker(lost_track_buffer=30)
    seen_ids: set[int] = set()
    for t, gt in enumerate(scene):
        detections = [d for d in gt.values() if d is not None]
        assert len(detections) == len(CHANNELS)
        tracks, _status = tracker.update(detections, t)
        assert len(tracks) <= 1
        seen_ids.update(tr.track_id for tr in tracks)
    assert len(seen_ids) == 1


def test_zero_noise_perfect() -> None:
    scene = walk_boxes(100)
    cfg = NoiseConfig(miss_rate=0.0, jitter_px=0.0)
    noise = NoiseInjector(cfg)
    tracker = LockTracker(lost_track_buffer=30)

    gt_boxes, all_tracks = _run(scene, noise, tracker)
    result = retention(gt_boxes, all_tracks)
    assert result.retention == 1.0
