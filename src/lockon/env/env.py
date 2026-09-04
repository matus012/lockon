"""Env: MuJoCo arena + drone/person kinematics, state-only (SPEC.md `Kinematics`). No physics
solve — both bodies are mocap, positions are integrated in Python and written each step.
"""

from __future__ import annotations

import logging
import math
from collections.abc import Mapping
from typing import cast

import mujoco  # type: ignore[import-untyped]
import numpy as np
import numpy.typing as npt

from lockon.core.schemas import (
    CHANNELS,
    CONTROL_HZ,
    IMAGE_HEIGHT,
    IMAGE_WIDTH,
    N_RAYCASTS,
    AgentAction,
    ArenaLayout,
    Difficulty,
    FloatArray,
    Pose2D,
    WorldState,
    channel_sees,
    person_max_speed,
)
from lockon.env import arena, geometry

logger = logging.getLogger(__name__)

DT: float = 1.0 / CONTROL_HZ
DRONE_V_MAX: float = 2.5
DRONE_YAW_RATE_MAX: float = 1.5
PERSON_YAW_MIN_SPEED: float = 0.05
R_DRONE: float = 0.35
R_PERSON: float = 0.30
WALL_MARGIN: float = 0.4
DROPOUT_BURST_LO: int = 10
DROPOUT_BURST_HI: int = 40
RESET_PLACEMENT_TRIES: int = 200
RESET_DRONE_LOS_TRIES: int = 50
RESET_PAIR_TRIES = 8
RESET_DRONE_DIST_LO: float = 6.0
RESET_DRONE_DIST_HI: float = 10.0


def _quat_from_yaw(yaw: float) -> FloatArray:
    return np.array([math.cos(yaw / 2.0), 0.0, 0.0, math.sin(yaw / 2.0)], dtype=np.float64)


def _wrap_angle(a: float) -> float:
    return float((a + math.pi) % (2.0 * math.pi) - math.pi)


class Env:
    CAMERA: str = "drone_cam"
    PERSON_BODY: str = "person"
    DRONE_BODY: str = "drone"

    def __init__(self, difficulty: Difficulty, seed: int, half_size: float = 12.0) -> None:
        self.difficulty = difficulty
        self.half_size = half_size
        self.layout: ArenaLayout
        self.model: mujoco.MjModel
        self.data: mujoco.MjData
        self._rng: np.random.Generator
        self._dropout_rng: np.random.Generator
        self._t: int = 0
        self._drone_pose = Pose2D(0.0, 0.0, 0.0)
        self._person_pose = Pose2D(0.0, 0.0, 0.0)
        self._darkness: float = difficulty.darkness
        self._dead_channel: str | None = None
        self._dead_until: int = 0
        self._channel_overrides: dict[str, bool] = {}
        self._drone_body_id: int = -1
        self._person_body_id: int = -1
        self._drone_mocap_id: int = -1
        self._person_mocap_id: int = -1
        self._cam_id: int = -1
        self.reset(seed)

    # -- lifecycle ---------------------------------------------------------------------------
    def reset(self, seed: int | None = None) -> WorldState:
        if seed is not None:
            self._rng = np.random.default_rng(seed)
            # dropout gets its own stream so the schedule does not shift with the number of
            # pillar rejection draws when occluder_density changes (review finding 7)
            self._dropout_rng = np.random.default_rng([seed, 0x5D5D])
            self.layout = arena.build_layout(self.difficulty, self._rng, self.half_size)
            mjcf = arena.build_mjcf(self.layout)
            self.model = mujoco.MjModel.from_xml_string(mjcf)
            self.data = mujoco.MjData(self.model)
            self._drone_body_id = mujoco.mj_name2id(self.model, mujoco.mjtObj.mjOBJ_BODY, self.DRONE_BODY)
            self._person_body_id = mujoco.mj_name2id(self.model, mujoco.mjtObj.mjOBJ_BODY, self.PERSON_BODY)
            self._drone_mocap_id = int(self.model.body_mocapid[self._drone_body_id])
            self._person_mocap_id = int(self.model.body_mocapid[self._person_body_id])
            self._cam_id = mujoco.mj_name2id(self.model, mujoco.mjtObj.mjOBJ_CAMERA, self.CAMERA)
            # populate geom_xpos/cam_xpos for static geoms before any placement LOS check —
            # a fresh MjData has zero-initialized derived quantities until forward runs once.
            mujoco.mj_forward(self.model, self.data)

        self._t = 0
        self._dead_channel = None
        self._dead_until = 0
        self._channel_overrides = {}

        # episode starts visible (SPEC): re-sample the person if no clear drone pose exists
        for _attempt in range(RESET_PAIR_TRIES):
            self._person_pose = self._sample_person_pose()
            drone = self._sample_drone_pose(self._person_pose)
            if drone is not None:
                self._drone_pose = drone
                break
        else:
            raise RuntimeError(f"no visible start after {RESET_PAIR_TRIES} person placements")
        self._write_mocap()
        self.set_darkness(self.difficulty.darkness)
        mujoco.mj_forward(self.model, self.data)
        return self.state()

    def _sample_person_pose(self) -> Pose2D:
        bound = self.half_size - WALL_MARGIN - R_PERSON
        for _ in range(RESET_PLACEMENT_TRIES):
            x = float(self._rng.uniform(-bound, bound))
            y = float(self._rng.uniform(-bound, bound))
            if not any(
                abs(x - px) < (hw + R_PERSON) and abs(y - py) < (hw + R_PERSON)
                for px, py, hw in self.layout.pillars
            ):
                yaw = float(self._rng.uniform(-math.pi, math.pi))
                return Pose2D(x, y, yaw)
        logger.warning("person placement: exhausted %d tries, using last candidate", RESET_PLACEMENT_TRIES)
        return Pose2D(x, y, float(self._rng.uniform(-math.pi, math.pi)))

    def _sample_drone_pose(self, person: Pose2D) -> Pose2D | None:
        bound = self.half_size - WALL_MARGIN - R_DRONE
        last = Pose2D(0.0, 0.0, 0.0)
        for _ in range(RESET_DRONE_LOS_TRIES):
            dist = float(self._rng.uniform(RESET_DRONE_DIST_LO, RESET_DRONE_DIST_HI))
            angle = float(self._rng.uniform(-math.pi, math.pi))
            x = float(np.clip(person.x + dist * math.cos(angle), -bound, bound))
            y = float(np.clip(person.y + dist * math.sin(angle), -bound, bound))
            yaw = math.atan2(person.y - y, person.x - x)
            last = Pose2D(x, y, yaw)
            eye = np.array([x, y, self.layout.drone_altitude])
            target = np.array([person.x, person.y, arena.TORSO_CENTER[2]])
            if geometry.line_of_sight(self.model, self.data, eye, target):
                return last
        # fallback: scan a ring of angles at a fixed distance for a clear line of sight.
        dist = (RESET_DRONE_DIST_LO + RESET_DRONE_DIST_HI) / 2.0
        for k in range(36):
            angle = k * 2.0 * math.pi / 36
            x = float(np.clip(person.x + dist * math.cos(angle), -bound, bound))
            y = float(np.clip(person.y + dist * math.sin(angle), -bound, bound))
            yaw = math.atan2(person.y - y, person.x - x)
            candidate = Pose2D(x, y, yaw)
            eye = np.array([x, y, self.layout.drone_altitude])
            target = np.array([person.x, person.y, arena.TORSO_CENTER[2]])
            if geometry.line_of_sight(self.model, self.data, eye, target):
                return candidate
        logger.debug("drone placement: no clear line of sight for this person pose; re-sampling")
        return None

    def _write_mocap(self) -> None:
        self.data.mocap_pos[self._drone_mocap_id] = [
            self._drone_pose.x,
            self._drone_pose.y,
            self.layout.drone_altitude,
        ]
        self.data.mocap_quat[self._drone_mocap_id] = _quat_from_yaw(self._drone_pose.yaw)
        self.data.mocap_pos[self._person_mocap_id] = [self._person_pose.x, self._person_pose.y, 0.0]
        self.data.mocap_quat[self._person_mocap_id] = _quat_from_yaw(self._person_pose.yaw)

    # -- step ----------------------------------------------------------------------------------
    def step(self, action: AgentAction, person_velocity: tuple[float, float]) -> WorldState:
        a = np.clip(action.as_array(), -1.0, 1.0)
        vx_b, vy_b, yaw_rate_n = float(a[0]), float(a[1]), float(a[2])

        yaw = self._drone_pose.yaw
        c, s = math.cos(yaw), math.sin(yaw)
        vx_b *= DRONE_V_MAX
        vy_b *= DRONE_V_MAX
        world_dx = vx_b * c - vy_b * s
        world_dy = vx_b * s + vy_b * c
        new_yaw = _wrap_angle(yaw + yaw_rate_n * DRONE_YAW_RATE_MAX * DT)
        drone_xy = np.array([self._drone_pose.x + world_dx * DT, self._drone_pose.y + world_dy * DT])
        drone_xy = geometry_resolve(drone_xy, R_DRONE, self.layout.pillars)
        drone_xy = _clamp_to_walls(drone_xy, self.half_size, WALL_MARGIN)
        self._drone_pose = Pose2D(float(drone_xy[0]), float(drone_xy[1]), new_yaw)

        pvx, pvy = person_velocity
        speed = math.hypot(pvx, pvy)
        max_speed = self.person_max_speed()
        if speed > max_speed and speed > 0.0:
            scale = max_speed / speed
            pvx *= scale
            pvy *= scale
            speed = max_speed
        person_xy = np.array([self._person_pose.x + pvx * DT, self._person_pose.y + pvy * DT])
        person_xy = geometry_resolve(person_xy, R_PERSON, self.layout.pillars)
        person_xy = _clamp_to_walls(person_xy, self.half_size, WALL_MARGIN)
        person_yaw = math.atan2(pvy, pvx) if speed > PERSON_YAW_MIN_SPEED else self._person_pose.yaw
        self._person_pose = Pose2D(float(person_xy[0]), float(person_xy[1]), person_yaw)

        self._write_mocap()
        mujoco.mj_forward(self.model, self.data)

        self._t += 1
        self._advance_dropout()
        return self.state()

    def _advance_dropout(self) -> None:
        dial = self.difficulty.channel_dropout
        if self._dead_channel is not None:
            if self._t >= self._dead_until:
                self._dead_channel = None
            return
        hazard = 0.01 * dial
        if hazard <= 0.0:
            return
        for c in CHANNELS:
            if float(self._dropout_rng.random()) < hazard:
                self._dead_channel = c
                duration = int(self._dropout_rng.integers(DROPOUT_BURST_LO, DROPOUT_BURST_HI + 1))
                self._dead_until = self._t + duration
                break

    # -- state -----------------------------------------------------------------------------
    def state(self) -> WorldState:
        cam_pos = np.array(self.data.cam_xpos[self._cam_id], dtype=np.float64)
        cam_mat = np.array(self.data.cam_xmat[self._cam_id], dtype=np.float64)
        fovy = float(self.model.cam_fovy[self._cam_id])

        px, py, pyaw = self._person_pose.x, self._person_pose.y, self._person_pose.yaw
        box = geometry.person_box(cam_pos, cam_mat, fovy, IMAGE_WIDTH, IMAGE_HEIGHT, px, py, pyaw)

        torso_world = np.array(
            [
                px + arena.TORSO_CENTER[0] * math.cos(pyaw) - arena.TORSO_CENTER[1] * math.sin(pyaw),
                py + arena.TORSO_CENTER[0] * math.sin(pyaw) + arena.TORSO_CENTER[1] * math.cos(pyaw),
                arena.TORSO_CENTER[2],
            ]
        )
        head_world = np.array(
            [
                px + arena.HEAD_CENTER[0] * math.cos(pyaw) - arena.HEAD_CENTER[1] * math.sin(pyaw),
                py + arena.HEAD_CENTER[0] * math.sin(pyaw) + arena.HEAD_CENTER[1] * math.cos(pyaw),
                arena.HEAD_CENTER[2],
            ]
        )
        pixels, depth = geometry.project_points(
            torso_world.reshape(1, 3), cam_pos, cam_mat, fovy, IMAGE_WIDTH, IMAGE_HEIGHT
        )
        in_fov = bool(depth[0] > 0.0 and 0.0 <= pixels[0, 0] <= IMAGE_WIDTH and 0.0 <= pixels[0, 1] <= IMAGE_HEIGHT)
        torso_clear = geometry.line_of_sight(self.model, self.data, cam_pos, torso_world)
        head_clear = geometry.line_of_sight(self.model, self.data, cam_pos, head_world)
        unoccluded = torso_clear or head_clear

        illum = self.illumination(np.array([px, py]))
        range_to_person = float(np.linalg.norm(torso_world - cam_pos))

        alive = {c: True for c in CHANNELS}
        if self._dead_channel is not None:
            alive[self._dead_channel] = False
        alive.update(self._channel_overrides)

        visible_geom = in_fov and unoccluded
        channels_see: dict[str, bool] = {
            c: alive[c] and visible_geom and channel_sees(c, illum, range_to_person) for c in CHANNELS
        }

        rc = geometry.raycasts(
            self.model,
            self.data,
            self._drone_pose.x,
            self._drone_pose.y,
            self.layout.drone_altitude,
            self._drone_pose.yaw,
            N_RAYCASTS,
            self.half_size,
        )

        return WorldState(
            t=self._t,
            drone=self._drone_pose,
            person=self._person_pose,
            person_visible=any(channels_see.values()),
            person_box=box,
            raycasts=rc,
            illumination_at_person=illum,
            channels_alive=alive,
            channels_see=channels_see,
        )

    # -- public helpers ----------------------------------------------------------------------
    @property
    def darkness(self) -> float:
        """Current darkness dial (scripted lights-cut may move it mid-episode)."""
        return self._darkness

    def set_darkness(self, darkness: float) -> None:
        self._darkness = float(np.clip(darkness, 0.0, 1.0))
        for i in range(self.layout.lights.shape[0]):
            base = float(self.layout.lights[i, 2])
            v = base * (1.0 - self._darkness)
            self.model.light_diffuse[i] = [v, v, v]
        fill = 1.0 - self._darkness
        self.model.vis.headlight.ambient[:] = arena.HEADLIGHT_AMBIENT * fill
        self.model.vis.headlight.diffuse[:] = arena.HEADLIGHT_DIFFUSE * fill

    def set_channels_alive(self, alive: Mapping[str, bool]) -> None:
        self._channel_overrides.update(alive)

    def illumination(self, xy: npt.ArrayLike) -> float:
        return geometry.illumination(np.asarray(xy, dtype=np.float64), self.layout.lights, self._darkness)

    def line_of_sight(self, a_xyz: npt.ArrayLike, b_xyz: npt.ArrayLike) -> bool:
        a = np.asarray(a_xyz, dtype=np.float64)
        b = np.asarray(b_xyz, dtype=np.float64)
        return geometry.line_of_sight(self.model, self.data, a, b)

    def person_bbox_world(self) -> FloatArray:
        return geometry.person_bbox_world(self._person_pose.x, self._person_pose.y, self._person_pose.yaw)

    def project(self, points_world: FloatArray) -> tuple[FloatArray, FloatArray]:
        cam_pos = np.array(self.data.cam_xpos[self._cam_id], dtype=np.float64)
        cam_mat = np.array(self.data.cam_xmat[self._cam_id], dtype=np.float64)
        fovy = float(self.model.cam_fovy[self._cam_id])
        return geometry.project_points(points_world, cam_pos, cam_mat, fovy, IMAGE_WIDTH, IMAGE_HEIGHT)

    def person_max_speed(self) -> float:
        return person_max_speed(self.difficulty)


def geometry_resolve(pos: FloatArray, radius: float, pillars: FloatArray) -> FloatArray:
    """Circle-vs-square-pillar push-out, single pass (SPEC.md `Kinematics`, no MuJoCo physics)."""
    p = pos.copy()
    for px, py, hw in pillars:
        dx, dy = p[0] - px, p[1] - py
        if abs(dx) < hw and abs(dy) < hw:
            pen_x, pen_y = hw - abs(dx), hw - abs(dy)
            if pen_x < pen_y:
                p[0] = px + (hw + radius if dx >= 0 else -(hw + radius))
            else:
                p[1] = py + (hw + radius if dy >= 0 else -(hw + radius))
        else:
            cx = float(np.clip(dx, -hw, hw))
            cy = float(np.clip(dy, -hw, hw))
            closest = np.array([px + cx, py + cy])
            diff = p - closest
            dist = float(np.linalg.norm(diff))
            if 1e-9 < dist < radius:
                p = closest + diff / dist * radius
    return p


def _clamp_to_walls(pos: FloatArray, half_size: float, margin: float) -> FloatArray:
    lim = half_size - margin
    return cast(FloatArray, np.clip(pos, -lim, lim))
