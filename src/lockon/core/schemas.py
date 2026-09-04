"""Schemas shared by every package. Frozen where a value is a message, mutable where it is state.

Conventions (single source of truth — do not restate elsewhere):
- World frame: metres, z up, arena centred at the origin. Yaw in radians, 0 = +x.
- Image frame: pixels, origin top-left, boxes are xyxy float arrays of shape (4,).
- Time: `t` is the discrete control step (CONTROL_HZ Hz); an episode is EPISODE_STEPS steps.
- Dials (Difficulty) are floats in [0, 1]; 0 = easiest, 1 = hardest, 0.5 = "mid" (plan §9 D5).
- Channels are named by CHANNELS; adding a channel touches lockon.sensor only (project.md §4).
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Final

import numpy as np
import numpy.typing as npt

FloatArray = npt.NDArray[np.float64]

CHANNELS: Final[tuple[str, ...]] = ("rgb", "depth", "thermal")
PERSON_MATERIAL: Final[str] = "person_mat"  # MJCF material on every person geom (env writes, sensor swaps)


@dataclass(frozen=True)
class ChannelSpec:
    """State-level physics of a sensing channel (what env needs to decide `channels_see`).

    Rendering lives in lockon.sensor; this is the only other place a channel is defined.
    Adding a channel = one entry here + one renderer in sensor (project.md §4, gate G2).
    """

    min_illumination: float  # channel sees only if illumination at the target >= this
    max_range_m: float  # channel sees only if camera-target range <= this
    # dark immunity == min_illumination 0.0; no second flag (review 2026-09-04 finding 10)


CHANNEL_SPECS: Final[dict[str, ChannelSpec]] = {
    "rgb": ChannelSpec(min_illumination=0.15, max_range_m=60.0),
    "depth": ChannelSpec(min_illumination=0.0, max_range_m=20.0),
    "thermal": ChannelSpec(min_illumination=0.0, max_range_m=60.0),
}
assert tuple(CHANNEL_SPECS) == CHANNELS


def channel_sees(channel: str, illumination: float, range_m: float) -> bool:
    """Channel physics only — caller ANDs this with alive / in-FOV / line-of-sight."""
    spec = CHANNEL_SPECS[channel]
    return illumination >= spec.min_illumination and range_m <= spec.max_range_m
N_RAYCASTS: Final[int] = 16
CONTROL_HZ: Final[int] = 10
EPISODE_STEPS: Final[int] = 200
IMAGE_WIDTH: Final[int] = 640
IMAGE_HEIGHT: Final[int] = 480


@dataclass(frozen=True)
class Difficulty:
    """Harness axes (project.md §3). Each in [0, 1]."""

    occluder_density: float = 0.5
    darkness: float = 0.5  # 1.0 = lights out (hardest); 0.0 = full light
    prey_speed: float = 0.5
    prey_aggressiveness: float = 0.5
    channel_dropout: float = 0.5

    def __post_init__(self) -> None:
        for name, v in vars(self).items():
            if not 0.0 <= v <= 1.0:
                raise ValueError(f"Difficulty.{name}={v} outside [0, 1]")

    def replace(self, **kw: float) -> Difficulty:
        return Difficulty(**{**vars(self), **kw})


def person_max_speed(difficulty: Difficulty) -> float:
    """Prey speed limit in m/s from the dial: 0.6 (dial 0) to 2.2 (dial 1). Env clamps, prey plans."""
    return 0.6 + 1.6 * difficulty.prey_speed


@dataclass(frozen=True)
class Pose2D:
    x: float
    y: float
    yaw: float

    def as_array(self) -> FloatArray:
        return np.array([self.x, self.y, self.yaw], dtype=np.float64)


@dataclass(frozen=True)
class ArenaLayout:
    """Static per-episode geometry. Prey scripts and hunters read it; only env writes it."""

    half_size: float
    pillars: FloatArray  # (M, 3): x, y, half-width of a square pillar
    lights: FloatArray  # (L, 3): x, y, intensity in [0, 1]
    drone_altitude: float


@dataclass
class WorldState:
    """Ground-truth state at step t. Written by env; read by policy (scripted), harness, sensor."""

    t: int
    drone: Pose2D
    person: Pose2D
    person_visible: bool  # in-FOV AND unoccluded line of sight, from state (no render)
    person_box: FloatArray | None  # projected GT box xyxy or None if outside the image
    raycasts: FloatArray  # (N_RAYCASTS,) distances normalised to [0, 1] by 2*half_size
    illumination_at_person: float  # [0, 1], from the light field
    in_fov: bool = False  # torso centre projects inside the image (geometry only)
    unoccluded: bool = False  # line of sight to torso or head is clear (geometry only)
    channels_alive: dict[str, bool] = field(default_factory=lambda: dict.fromkeys(CHANNELS, True))
    channels_see: dict[str, bool] = field(default_factory=lambda: dict.fromkeys(CHANNELS, False))
    # channels_see[c] = alive[c] AND in-FOV AND unoccluded AND channel_sees(c, illum, range).
    # person_visible == any(channels_see.values()). Computed in lockon.env only.


@dataclass(frozen=True)
class Detection:
    box: FloatArray  # xyxy
    score: float
    channel: str
    t: int


@dataclass(frozen=True)
class Track:
    track_id: int
    box: FloatArray  # xyxy
    t: int


@dataclass(frozen=True)
class LockStatus:
    """Tracker verdict for step t. `locked` = a track carrying the target's first ID is alive."""

    t: int
    locked: bool
    track_id: int | None
    steps_since_lock: int


@dataclass
class SensorFrame:
    """One multi-channel observation. `images[c]` is None when channel c is dead or unrendered."""

    t: int
    images: dict[str, npt.NDArray[np.uint8] | None]
    alive: dict[str, bool]
    gt: dict[str, Detection | None]  # per-channel GT detection (None if dead or out of view)
    width: int = IMAGE_WIDTH
    height: int = IMAGE_HEIGHT


@dataclass(frozen=True)
class AgentObs:
    """Partial observation for the hunter (project.md §3). All fields normalised."""

    own: FloatArray  # (3,): x/half_size, y/half_size, yaw/pi
    last_seen_rel: FloatArray  # (2,): target position relative to drone at last sighting, /2*half_size
    time_since_seen: float  # steps since last sighting / EPISODE_STEPS, clipped to [0, 1]
    raycasts: FloatArray  # (N_RAYCASTS,)
    light: float  # illumination at last-seen position, [0, 1]
    channels_alive: FloatArray  # (len(CHANNELS),) 0/1

    @staticmethod
    def size() -> int:
        return 3 + 2 + 1 + N_RAYCASTS + 1 + len(CHANNELS)

    def vector(self) -> npt.NDArray[np.float32]:
        v = np.concatenate(
            [
                self.own,
                self.last_seen_rel,
                [self.time_since_seen],
                self.raycasts,
                [self.light],
                self.channels_alive,
            ]
        ).astype(np.float32)
        assert v.shape == (self.size(),)
        return v


@dataclass(frozen=True)
class AgentAction:
    """Kinematic velocity command, each in [-1, 1]; env scales to physical limits."""

    vx: float
    vy: float
    yaw_rate: float

    @staticmethod
    def size() -> int:
        return 3

    @classmethod
    def from_array(cls, a: npt.ArrayLike) -> AgentAction:
        arr = np.clip(np.asarray(a, dtype=np.float64).reshape(3), -1.0, 1.0)
        return cls(float(arr[0]), float(arr[1]), float(arr[2]))

    def as_array(self) -> FloatArray:
        return np.array([self.vx, self.vy, self.yaw_rate], dtype=np.float64)

    @classmethod
    def zero(cls) -> AgentAction:
        return cls(0.0, 0.0, 0.0)
