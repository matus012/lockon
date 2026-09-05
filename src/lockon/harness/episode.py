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
    AgentObs,
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


def hunter_frame_stack(hunter: Hunter) -> int:
    """Frame-stack depth the hunter expects: `PPOHunter.n_stack`, else 1 (single-step `AgentObs`).
    Shared with `prey_gym.PreyGym`, which drives the same frozen hunter inside its own loop."""
    return hunter.n_stack if isinstance(hunter, PPOHunter) else 1


def hunter_obs_mode(hunter: Hunter) -> str:
    """`PPOHunter.obs_seen` sidecar convention, else `"lock"` for the rule-based hunters."""
    return hunter.obs_seen if isinstance(hunter, PPOHunter) else "lock"


def init_hunter_stack(
    obs_vector: npt.NDArray[np.float32], n_stack: int
) -> deque[npt.NDArray[np.float32]]:
    """SB3 `VecFrameStack` semantics: zero-filled history, newest observation last (review F7)."""
    stack: deque[npt.NDArray[np.float32]] = deque(maxlen=n_stack)
    for _ in range(n_stack - 1):
        stack.append(np.zeros_like(obs_vector))
    stack.append(obs_vector)
    return stack


def hunter_action(
    hunter: Hunter, obs: AgentObs, stack: deque[npt.NDArray[np.float32]]
) -> AgentAction:
    """Drives `hunter` exactly as the harness does: `PPOHunter` gets the stacked vector, the
    rule-based hunters get the single-step `AgentObs` (SPEC_prey.md: same rule for `PreyGym`)."""
    if isinstance(hunter, PPOHunter):
        return hunter.act_vector(np.concatenate(list(stack)).astype(np.float64))
    return hunter.act(obs)


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
    """Cause of a lock-loss event, from the state at the step visibility was lost.

    Uses the env's explicit geometry flags (review 2026-09-04 F2: inferring "fov" from
    `person_box is None` labelled every edge-of-frame loss "occlusion").
    - not in_fov      -> "fov"
    - occluded        -> "occlusion"
    - no alive channel -> "dropout"
    - alive channels exist but none sees: rgb-only alive and too dark -> "darkness", else "dropout"
    """
    if not state.in_fov:
        return "fov"
    if not state.unoccluded:
        return "occlusion"
    alive = [c for c, a in state.channels_alive.items() if a]
    if not alive:
        return "dropout"
    dark_immune_alive = [c for c in alive if CHANNEL_SPECS[c].min_illumination == 0.0]
    if not dark_immune_alive and all(
        state.illumination_at_person < CHANNEL_SPECS[c].min_illumination for c in alive
    ):
        return "darkness"
    return "dropout"


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
    # per-episode noise seed: one frozen seed for all 20 episodes hid the noise variance that
    # deviation row 6 measured (review 2026-09-04 F4, row 11). Distribution unchanged.
    injector = NoiseInjector(noise if noise is not None else NoiseConfig(**{**vars(EVAL_NOISE), "seed": seed}))
    sensor = Sensor(env.model, env.data, env.CAMERA, seed=seed) if capture else None

    state = env.state()
    obs = obs_builder.reset(state)
    n_stack = hunter_frame_stack(hunter)
    obs_mode = hunter_obs_mode(hunter)
    stack = init_hunter_stack(obs.vector(), n_stack)

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
        action = hunter_action(hunter, obs, stack)

        state = env.step(action, (pvx, pvy))

        gt = _gt_detections(state)
        detections = injector(gt, t)
        tracks, lock_status = tracker.update(detections, t)
        # the policy's "time since seen" is its own tracker's lock, not privileged geometry (D15)
        seen = lock_status.locked if obs_mode == "lock" else state.person_visible
        obs = obs_builder.step(state, seen=seen)
        stack.append(obs.vector())

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
