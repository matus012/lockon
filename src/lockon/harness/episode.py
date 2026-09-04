"""Episode runner — the harness's core loop (SPEC.md `episode.py`). Composes env + track + policy
for one episode and attributes lock-loss causes from `WorldState` (project.md §7: perception is
frozen for every number the owner reads).
"""

from __future__ import annotations

import logging
from collections import deque
from collections.abc import Callable
from dataclasses import dataclass

import numpy as np
import numpy.typing as npt

from lockon.core.metrics import RetentionResult, retention
from lockon.core.schemas import (
    CHANNEL_SPECS,
    CHANNELS,
    EPISODE_STEPS,
    AgentAction,
    Detection,
    Difficulty,
    FloatArray,
    LockStatus,
    SensorFrame,
    Track,
    WorldState,
)
from lockon.env.env import Env
from lockon.policy.base import Hunter, Prey
from lockon.policy.features import ObsBuilder
from lockon.policy.hunters import PPOHunter, ScriptedHunter, StaticCamera
from lockon.policy.ppo import PPOConfig
from lockon.sensor.sensor import Sensor
from lockon.track.noise import NoiseConfig, NoiseInjector
from lockon.track.tracker import LockTracker

logger = logging.getLogger(__name__)

Scene = Callable[[Env, int], None]

# Eval-time perception is frozen (project.md §7, SPEC.md): every retention number the owner reads
# uses this single NoiseConfig + LockTracker() defaults. seed=0 is `NoiseConfig.from_dial`'s own
# default and matches context.md D11's literal `NoiseConfig.from_dial(0.3)` invocation — chosen so
# EVAL_NOISE is a true single frozen constant, not re-seeded per episode/caller.
EVAL_NOISE: NoiseConfig = NoiseConfig.from_dial(0.3, seed=0)

_DEFAULT_FRAME_STACK: int = PPOConfig().frame_stack


@dataclass
class EpisodeResult:
    seed: int
    difficulty: Difficulty
    policy: str
    states: list[WorldState]
    actions: list[AgentAction]
    gt_boxes: list[FloatArray | None]
    tracks: list[list[Track]]
    lock: list[LockStatus]
    retention: RetentionResult
    loss_causes: list[str]
    frames: list[SensorFrame] | None


def resolve_hunter(name_or_path: str) -> Hunter:
    """`static` -> StaticCamera, `scripted` -> ScriptedHunter, anything else -> PPOHunter(path)."""
    if name_or_path == "static":
        return StaticCamera()
    if name_or_path == "scripted":
        return ScriptedHunter()
    return PPOHunter(name_or_path)


def _gt_detections(state: WorldState) -> dict[str, Detection | None]:
    """GT per channel where `channels_see` (SPEC.md), mirroring `sensor.sensor.Sensor.capture`."""
    gt: dict[str, Detection | None] = {}
    for c in CHANNELS:
        if state.channels_see[c] and state.person_box is not None:
            gt[c] = Detection(box=state.person_box, score=1.0, channel=c, t=state.t)
        else:
            gt[c] = None
    return gt


def _loss_cause(state: WorldState) -> str:
    """Cause of a lock-loss event, from `state` at the step visibility was lost (SPEC.md).

    `WorldState` does not carry raw in-FOV / line-of-sight flags (only their AND with channel
    physics, via `channels_see`); this reconstructs the cause from what is available:
    - `person_box is None` -> the target could not be projected into the image at all -> "fov".
    - Else a dark-immune alive channel (depth/thermal) exists that still does not see -> since
      those channels ignore illumination, only occlusion explains it -> "occlusion".
    - Else the only-alive candidate is rgb and illumination is below its threshold -> "darkness".
    - Else (rgb alive but bright enough, or no channel alive at all) -> "dropout" / "occlusion":
      no channel alive at all -> "dropout"; rgb alive+bright but still unseen -> "occlusion".
    """
    if state.person_box is None:
        return "fov"
    alive = state.channels_alive
    # dark immunity == min_illumination 0.0 (core.schemas.ChannelSpec — no separate flag).
    if any(alive[c] for c in CHANNELS if CHANNEL_SPECS[c].min_illumination == 0.0):
        return "occlusion"
    if not any(alive.values()):
        return "dropout"
    if alive["rgb"] and state.illumination_at_person < CHANNEL_SPECS["rgb"].min_illumination:
        return "darkness"
    return "occlusion"


def run_episode(
    difficulty: Difficulty,
    seed: int,
    hunter: Hunter,
    prey: Prey,
    *,
    steps: int = EPISODE_STEPS,
    scene: Scene | None = None,
    capture: bool = False,
    noise: NoiseConfig | None = None,
) -> EpisodeResult:
    env = Env(difficulty, seed)
    layout = env.layout
    hunter.reset(layout, seed)
    prey.reset(layout, difficulty, seed)
    obs_builder = ObsBuilder(layout)
    tracker = LockTracker()
    injector = NoiseInjector(noise if noise is not None else EVAL_NOISE)
    sensor = Sensor(env.model, env.data, env.CAMERA, seed=seed) if capture else None

    state = env.state()
    obs = obs_builder.reset(state)
    stack: deque[npt.NDArray[np.float32]] = deque(
        [obs.vector() for _ in range(_DEFAULT_FRAME_STACK)], maxlen=_DEFAULT_FRAME_STACK
    )

    states: list[WorldState] = []
    actions: list[AgentAction] = []
    gt_boxes: list[FloatArray | None] = []
    tracks_all: list[list[Track]] = []
    lock_all: list[LockStatus] = []
    loss_causes: list[str] = []
    frames: list[SensorFrame] | None = [] if capture else None

    prev_visible = state.person_visible
    for t in range(steps):
        if scene is not None:
            scene(env, t)

        pvx, pvy = prey.act(state, env.illumination)
        if isinstance(hunter, PPOHunter):
            action = hunter.act_vector(np.concatenate(list(stack)).astype(np.float64))
        else:
            action = hunter.act(obs)

        state = env.step(action, (pvx, pvy))
        obs = obs_builder.step(state)
        stack.append(obs.vector())

        gt = _gt_detections(state)
        detections = injector(gt, t)
        tracks, lock_status = tracker.update(detections, t)

        if sensor is not None and frames is not None:
            # darkness is read here
            # only for the render's cosmetic per-channel noise, never for a retained number.
            frames.append(sensor.capture(state, env.darkness))

        states.append(state)
        actions.append(action)
        gt_boxes.append(state.person_box if state.person_visible else None)
        tracks_all.append(tracks)
        lock_all.append(lock_status)

        if prev_visible and not state.person_visible:
            loss_causes.append(_loss_cause(state))
        prev_visible = state.person_visible

    if sensor is not None:
        sensor.close()

    result = retention(gt_boxes, tracks_all)
    return EpisodeResult(
        seed=seed,
        difficulty=difficulty,
        policy=hunter.name,
        states=states,
        actions=actions,
        gt_boxes=gt_boxes,
        tracks=tracks_all,
        lock=lock_all,
        retention=result,
        loss_causes=loss_causes,
        frames=frames,
    )
