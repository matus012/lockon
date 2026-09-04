"""lockon.core — typed schemas and metrics only. Depends on numpy and nothing else.

Every other package imports this one; this one imports none of them (plan.md §9 D1).
"""

from lockon.core.metrics import RetentionResult, iou_xyxy, retention
from lockon.core.schemas import (
    CHANNEL_SPECS,
    CHANNELS,
    CONTROL_HZ,
    EPISODE_STEPS,
    IMAGE_HEIGHT,
    IMAGE_WIDTH,
    N_RAYCASTS,
    PERSON_MATERIAL,
    AgentAction,
    AgentObs,
    ArenaLayout,
    ChannelSpec,
    Detection,
    Difficulty,
    LockStatus,
    Pose2D,
    SensorFrame,
    Track,
    WorldState,
    channel_sees,
    person_max_speed,
)

__all__ = [
    "CHANNELS",
    "CHANNEL_SPECS",
    "CONTROL_HZ",
    "EPISODE_STEPS",
    "IMAGE_HEIGHT",
    "IMAGE_WIDTH",
    "N_RAYCASTS",
    "PERSON_MATERIAL",
    "AgentAction",
    "AgentObs",
    "ArenaLayout",
    "ChannelSpec",
    "Detection",
    "Difficulty",
    "LockStatus",
    "Pose2D",
    "RetentionResult",
    "SensorFrame",
    "Track",
    "WorldState",
    "channel_sees",
    "iou_xyxy",
    "person_max_speed",
    "retention",
]
