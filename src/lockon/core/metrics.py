"""Headline metric (project.md §3): lock retention %, plus time-to-reacquire. Single source."""

from __future__ import annotations

from collections.abc import Sequence
from dataclasses import dataclass

import numpy as np
import numpy.typing as npt

from lockon.core.schemas import Track

IOU_THRESHOLD = 0.5


def iou_xyxy(a: npt.ArrayLike, b: npt.ArrayLike) -> float:
    a_ = np.asarray(a, dtype=np.float64)
    b_ = np.asarray(b, dtype=np.float64)
    ix = max(0.0, min(a_[2], b_[2]) - max(a_[0], b_[0]))
    iy = max(0.0, min(a_[3], b_[3]) - max(a_[1], b_[1]))
    inter = ix * iy
    area_a = max(0.0, a_[2] - a_[0]) * max(0.0, a_[3] - a_[1])
    area_b = max(0.0, b_[2] - b_[0]) * max(0.0, b_[3] - b_[1])
    union = area_a + area_b - inter
    return float(inter / union) if union > 0 else 0.0


@dataclass(frozen=True)
class RetentionResult:
    retention: float  # retained steps / episode steps, in [0, 1]
    target_id: int | None  # first track id matched to the target; None if never matched
    retained: npt.NDArray[np.bool_]  # per-step flag
    n_loss_events: int
    time_to_reacquire: float  # mean steps from loss to reacquisition over reacquired events; nan if none
    last_loss_censored: bool  # the final loss was never reacquired before the episode ended


def retention(
    gt_boxes: Sequence[npt.NDArray[np.float64] | None],
    tracks: Sequence[Sequence[Track]],
    iou_threshold: float = IOU_THRESHOLD,
    episode_steps: int | None = None,
) -> RetentionResult:
    """Steps tracked with the CORRECT id (the target's first assigned id) / episode steps.

    A step is retained iff a track with `target_id` exists whose IoU to the GT box >= threshold.
    `target_id` is fixed at the first step any track overlaps the GT box; an id switch after that
    is a loss even if some other track still overlaps — that is the whole point of the metric.
    Steps where the GT box is None (target out of frame) count as not retained.
    """
    if len(gt_boxes) != len(tracks):
        raise ValueError(f"gt_boxes ({len(gt_boxes)}) and tracks ({len(tracks)}) differ in length")
    if episode_steps is not None and len(gt_boxes) != episode_steps:
        # one row per env.step, unconditionally - a harness that records only visible steps
        # would otherwise score ~1.0 (review 2026-09-04 finding 3)
        raise ValueError(f"{len(gt_boxes)} rows for a {episode_steps}-step episode")
    n = len(gt_boxes)
    retained = np.zeros(n, dtype=np.bool_)
    target_id: int | None = None
    for t in range(n):
        gt = gt_boxes[t]
        if gt is None:
            continue
        if target_id is None:
            best = max(tracks[t], key=lambda tr: iou_xyxy(tr.box, gt), default=None)
            if best is not None and iou_xyxy(best.box, gt) >= iou_threshold:
                target_id = best.track_id
                retained[t] = True
            continue
        retained[t] = any(
            tr.track_id == target_id and iou_xyxy(tr.box, gt) >= iou_threshold for tr in tracks[t]
        )

    losses: list[int] = []
    reacq: list[int] = []
    censored = False
    in_loss_since: int | None = None
    for t in range(n):
        if retained[t]:
            if in_loss_since is not None:
                reacq.append(t - in_loss_since)
                in_loss_since = None
        elif in_loss_since is None and t > 0 and retained[t - 1]:
            in_loss_since = t
            losses.append(t)
    if in_loss_since is not None:
        censored = True
    return RetentionResult(
        retention=float(retained.mean()) if n else 0.0,
        target_id=target_id,
        retained=retained,
        n_loss_events=len(losses),
        time_to_reacquire=float(np.mean(reacq)) if reacq else float("nan"),
        last_loss_censored=censored,
    )
