"""lockon.core — typed schemas and metrics only. Depends on numpy and nothing else.

Every other package imports this one; this one imports none of them (plan.md §9 D1).
"""

from lockon.core.metrics import RetentionResult, iou_xyxy, retention
from lockon.core.schemas import (
    CHANNELS,
    CONTROL_HZ,
    DEPTH_MAX_RANGE_M,
    EPISODE_STEPS,
    IMAGE_HEIGHT,
    IMAGE_WIDTH,
    N_RAYCASTS,
    RGB_MIN_ILLUMINATION,
    AgentAction,
    AgentObs,
    ArenaLayout,
    Detection,
    Difficulty,
    LockStatus,
    Pose2D,
    SensorFrame,
    Track,
    WorldState,
)

__all__ = [
    "CHANNELS",
    "CONTROL_HZ",
    "DEPTH_MAX_RANGE_M",
    "EPISODE_STEPS",
    "IMAGE_HEIGHT",
    "IMAGE_WIDTH",
    "N_RAYCASTS",
    "RGB_MIN_ILLUMINATION",
    "AgentAction",
    "AgentObs",
    "ArenaLayout",
    "Detection",
    "Difficulty",
    "LockStatus",
    "Pose2D",
    "RetentionResult",
    "SensorFrame",
    "Track",
    "WorldState",
    "iou_xyxy",
    "retention",
]
