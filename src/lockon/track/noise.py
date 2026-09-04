"""Synthetic detector noise (project.md §2: the tracker consumes GT boxes + injected noise).

`NoiseInjector` turns per-channel GT `Detection`s into what a noisy detector would report: some
detections are dropped, boxes are jittered, scores are resampled, and the whole stream is delayed
by a fixed FIFO latency. Deterministic given a seed and the call order (one call per step, in
order) — no other source of randomness is used.
"""

from __future__ import annotations

import logging
from collections import deque
from dataclasses import dataclass

import numpy as np

from lockon.core.schemas import IMAGE_HEIGHT, Detection

logger = logging.getLogger(__name__)

# 'occlusion_dropout' is an extra miss probability for SMALL boxes (target far / mostly hidden):
# box height below a quarter of the image (120 px at 480) - at 6 m standoff boxes are ~110 px,
# so the term is live across the working range (review 2026-09-04 finding 2: 24 px never fired).
_SMALL_BOX_FRACTION = 0.25


@dataclass(frozen=True)
class NoiseConfig:
    miss_rate: float = 0.1  # P(drop a detection) per channel per step
    jitter_px: float = 4.0  # gaussian sigma on each box edge, pixels
    occlusion_dropout: float = 0.0  # extra P(drop) when the box is small (< 25 % of image height)
    latency_steps: int = 0  # detections are delivered latency_steps late (FIFO)
    score_mean: float = 0.85
    score_std: float = 0.1  # confidence noise, clipped to [0.05, 1]
    seed: int = 0

    @classmethod
    def from_dial(cls, level: float, seed: int = 0) -> NoiseConfig:
        """level in [0,1]: miss 0.02->0.3, jitter 2->10, occl 0->0.4, latency 0->2."""

        def lerp(lo: float, hi: float) -> float:
            return lo + level * (hi - lo)

        return cls(
            miss_rate=lerp(0.02, 0.3),
            jitter_px=lerp(2.0, 10.0),
            occlusion_dropout=lerp(0.0, 0.4),
            latency_steps=round(lerp(0.0, 2.0)),
            seed=seed,
        )


class NoiseInjector:
    def __init__(self, cfg: NoiseConfig) -> None:
        self._cfg = cfg
        self._rng = np.random.default_rng(cfg.seed)
        # FIFO of length latency_steps: what's popped now was produced latency_steps calls ago.
        self._queue: deque[list[Detection]] = deque([[] for _ in range(cfg.latency_steps)])

    def __call__(self, gt: dict[str, Detection | None], t: int) -> list[Detection]:
        cfg = self._cfg
        produced: list[Detection] = []
        for channel, det in gt.items():
            if det is None:
                continue
            small = (det.box[3] - det.box[1]) < _SMALL_BOX_FRACTION * IMAGE_HEIGHT
            drop_p = cfg.miss_rate + (cfg.occlusion_dropout if small else 0.0)
            if self._rng.random() < drop_p:
                continue
            jitter = self._rng.normal(0.0, cfg.jitter_px, size=4)
            box = (det.box.astype(np.float64) + jitter)
            score = float(np.clip(self._rng.normal(cfg.score_mean, cfg.score_std), 0.05, 1.0))
            produced.append(Detection(box=box, score=score, channel=channel, t=det.t))
        self._queue.append(produced)
        return self._queue.popleft()
