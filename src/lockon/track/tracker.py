"""ByteTrack over fused multi-channel detections (project.md §7: perception frozen during RL).

Fusion: ByteTrack treats each detection it is fed as an independent object. If several channels
(rgb/depth/thermal) all report the same physical person at the same step, feeding all of them in
would spawn one track per channel for a single target. Instead we fuse to at most one detection
per step: average the boxes (they all see the same person) and take the max score.
"""

from __future__ import annotations

import logging

import numpy as np
import numpy.typing as npt
import supervision as sv
from trackers import ByteTrackTracker

from lockon.core.schemas import CONTROL_HZ, Detection, LockStatus, Track

logger = logging.getLogger(__name__)


def _fuse(detections: list[Detection]) -> Detection | None:
    if not detections:
        return None
    boxes = np.stack([d.box for d in detections]).astype(np.float64)
    box = boxes.mean(axis=0)
    score = max(d.score for d in detections)
    return Detection(box=box, score=score, channel="fused", t=detections[0].t)


class LockTracker:
    """ByteTrack over fused multi-channel detections. Perception is frozen during RL (project.md §7)."""

    def __init__(
        self,
        lost_track_buffer: int = 30,
        activation: float = 0.5,
        high_conf: float = 0.6,
        min_iou: float = 0.1,
        min_consecutive: int = 1,
    ) -> None:
        self._lost_track_buffer = lost_track_buffer
        self._activation = activation
        self._high_conf = high_conf
        self._min_iou = min_iou
        self._min_consecutive = min_consecutive
        self._tracker = self._make_tracker()
        self._target_id: int | None = None
        self._steps_since_lock = 0
        self._primed = False

    def _make_tracker(self) -> ByteTrackTracker:
        # ByteTrackTracker expresses lost_track_buffer in "30 FPS frame" units and rescales it by
        # frame_rate/30 (trackers.core.base._compute_maximum_frames_without_update) whenever we
        # call update() without a timestamp (fixed-rate mode, which we always do). At
        # frame_rate=CONTROL_HZ=10 that would shrink our "30 steps of gap tolerance" down to 10
        # real steps, so we undo the library's scaling here: lost_track_buffer means real control
        # steps to LockTracker's caller.
        return ByteTrackTracker(
            lost_track_buffer=round(self._lost_track_buffer * 30.0 / CONTROL_HZ),
            frame_rate=CONTROL_HZ,
            track_activation_threshold=self._activation,
            minimum_consecutive_frames=self._min_consecutive,
            minimum_iou_threshold=self._min_iou,
            high_conf_det_threshold=self._high_conf,
        )

    def reset(self) -> None:
        self._tracker = self._make_tracker()
        self._target_id = None
        self._steps_since_lock = 0
        self._primed = False

    def update(self, detections: list[Detection], t: int) -> tuple[list[Track], LockStatus]:
        fused = _fuse(detections)
        if fused is None:
            xyxy: npt.NDArray[np.float64] = np.zeros((0, 4), dtype=np.float64)
            confidence: npt.NDArray[np.float64] = np.zeros((0,), dtype=np.float64)
        else:
            xyxy = fused.box.reshape(1, 4)
            confidence = np.array([fused.score], dtype=np.float64)
        sv_detections = sv.Detections(xyxy=xyxy, confidence=confidence)
        if not self._primed and fused is not None:
            # trackers.core.bytetrack always reports tracker_id=-1 on the exact frame a track is
            # spawned (SORT convention, unconditional on minimum_consecutive_frames). Warm the
            # tracker with one throwaway call on the same box so the first real step already
            # reports a confirmed id instead of losing lock for exactly one frame every episode.
            self._tracker.update(sv_detections)
            self._primed = True
        out = self._tracker.update(sv_detections)

        tracks: list[Track] = []
        if out.tracker_id is not None:
            for box, track_id in zip(out.xyxy, out.tracker_id, strict=True):
                # -1 = unconfirmed/untracked detection (trackers.core.bytetrack), not a real track.
                if track_id == -1:
                    continue
                tracks.append(Track(track_id=int(track_id), box=box.astype(np.float64), t=t))

        if self._target_id is None and tracks:
            self._target_id = tracks[0].track_id
        locked = self._target_id is not None and any(tr.track_id == self._target_id for tr in tracks)
        self._steps_since_lock = 0 if locked else self._steps_since_lock + 1
        status = LockStatus(
            t=t,
            locked=locked,
            track_id=self._target_id if locked else None,
            steps_since_lock=self._steps_since_lock,
        )
        return tracks, status

    @property
    def target_id(self) -> int | None:
        return self._target_id
