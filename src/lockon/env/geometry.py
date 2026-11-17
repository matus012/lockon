"""Pure(ish) geometry: projection, LOS/raycasts (mj_ray), illumination (SPEC.md `Visibility and
boxes`). `line_of_sight` / `raycasts` take `model, data` because they call `mj_ray`; everything
else is plain numpy.
"""

from __future__ import annotations

import math

import mujoco  # type: ignore[import-untyped]
import numpy as np

from lockon.core.schemas import FloatArray
from lockon.env import arena

# geomgroup mask enabling groups 0 (floor/walls) and 1 (pillars) only — excludes drone (2) and
# person (3), used for both occlusion raycasts and the LOS check.
STATIC_GEOMGROUP: FloatArray = np.array([1, 1, 0, 0, 0, 0], dtype=np.uint8)

_PARTS: FloatArray = np.array(arena.PERSON_PARTS, dtype=np.float64)  # (P, 5): cx, cy, r, zlo, zhi


def project_points(
    points_world: FloatArray,
    cam_pos: FloatArray,
    cam_mat: FloatArray,
    fovy_deg: float,
    width: int,
    height: int,
) -> tuple[FloatArray, FloatArray]:
    """Project world points into the camera's pixel frame. Returns (pixels (N,2), depth (N,));
    depth > 0 means in front of the camera (see env/SPEC.md note on MuJoCo camera convention).
    """
    rot = np.asarray(cam_mat, dtype=np.float64).reshape(3, 3)
    rel = np.asarray(points_world, dtype=np.float64).reshape(-1, 3) - cam_pos
    local = rel @ rot  # row-vector form of R^T @ rel
    x_cam, y_cam, z_cam = local[:, 0], local[:, 1], local[:, 2]
    depth = -z_cam
    f = (height / 2.0) / math.tan(math.radians(fovy_deg) / 2.0)
    safe_depth = np.where(depth != 0.0, depth, np.nan)
    u = width / 2.0 + f * x_cam / safe_depth
    v = height / 2.0 - f * y_cam / safe_depth
    pixels = np.stack([u, v], axis=-1)
    return pixels, depth


def person_bbox_world(person_x: float, person_y: float, person_yaw: float) -> FloatArray:
    """(8,3) corners of the person's true world-frame AABB at this pose.

    Each capsule/sphere part has a circular (x, y) footprint, which is rotation-invariant about
    z — so the true world AABB is the union, over parts, of `R(yaw) @ (local_x, local_y) +/-
    radius` (see `arena.PERSON_PARTS`), not a single local box rotated as a rigid unit (which
    over-estimates the box at non-zero yaw).
    """
    c, s = math.cos(person_yaw), math.sin(person_yaw)
    cx = _PARTS[:, 0] * c - _PARTS[:, 1] * s + person_x
    cy = _PARTS[:, 0] * s + _PARTS[:, 1] * c + person_y
    r = _PARTS[:, 2]
    x0, x1 = float(np.min(cx - r)), float(np.max(cx + r))
    y0, y1 = float(np.min(cy - r)), float(np.max(cy + r))
    z0, z1 = arena.PERSON_AABB_Z_LO, arena.PERSON_AABB_Z_HI
    return np.array(
        [[x, y, z] for x in (x0, x1) for y in (y0, y1) for z in (z0, z1)],
        dtype=np.float64,
    )


def person_box(
    cam_pos: FloatArray,
    cam_mat: FloatArray,
    fovy_deg: float,
    width: int,
    height: int,
    person_x: float,
    person_y: float,
    person_yaw: float,
) -> FloatArray | None:
    """Project the person AABB corners to a clipped pixel box, or None (SPEC.md rule)."""
    torso_world = np.array(
        [
            person_x + arena.TORSO_CENTER[0] * math.cos(person_yaw) - arena.TORSO_CENTER[1] * math.sin(person_yaw),
            person_y + arena.TORSO_CENTER[0] * math.sin(person_yaw) + arena.TORSO_CENTER[1] * math.cos(person_yaw),
            arena.TORSO_CENTER[2],
        ]
    )
    _, torso_depth = project_points(torso_world.reshape(1, 3), cam_pos, cam_mat, fovy_deg, width, height)
    if float(torso_depth[0]) <= 0.0:
        return None

    corners = person_bbox_world(person_x, person_y, person_yaw)
    pixels, _ = project_points(corners, cam_pos, cam_mat, fovy_deg, width, height)
    x0 = float(np.clip(np.nanmin(pixels[:, 0]), 0.0, width))
    x1 = float(np.clip(np.nanmax(pixels[:, 0]), 0.0, width))
    y0 = float(np.clip(np.nanmin(pixels[:, 1]), 0.0, height))
    y1 = float(np.clip(np.nanmax(pixels[:, 1]), 0.0, height))
    area = max(0.0, x1 - x0) * max(0.0, y1 - y0)
    if area < 4.0:
        return None
    return np.array([x0, y0, x1, y1], dtype=np.float64)


def line_of_sight(model: mujoco.MjModel, data: mujoco.MjData, a_xyz: FloatArray, b_xyz: FloatArray) -> bool:
    """`mj_ray` from a to b, excluding drone/person geoms (groups 2, 3). Clear if nothing is hit
    strictly before b (walls/pillars only, groups 0/1).
    """
    a = np.asarray(a_xyz, dtype=np.float64).reshape(3)
    b = np.asarray(b_xyz, dtype=np.float64).reshape(3)
    vec = b - a
    dist = float(np.linalg.norm(vec))
    if dist < 1e-9:
        return True
    direction = (vec / dist).reshape(3, 1)
    geomid = np.zeros(1, dtype=np.int32)
    hit = mujoco.mj_ray(
        model, data, a.reshape(3, 1), direction, STATIC_GEOMGROUP.reshape(6, 1), 1, -1, geomid
    )
    return bool(hit < 0.0 or hit >= dist - 1e-3)


def illumination(xy: FloatArray, lights: FloatArray, darkness: float) -> float:
    """SPEC.md light field: clip((1-darkness)*(0.25 + sum_i I_i*max(0, 1-d_i/8)), 0, 1)."""
    p = np.asarray(xy, dtype=np.float64).reshape(2)
    if lights.shape[0] == 0:
        total = 0.25
    else:
        d = np.linalg.norm(lights[:, :2] - p, axis=1)
        total = 0.25 + float(np.sum(lights[:, 2] * np.maximum(0.0, 1.0 - d / 8.0)))
    return float(np.clip((1.0 - darkness) * total, 0.0, 1.0))


def raycasts(
    model: mujoco.MjModel,
    data: mujoco.MjData,
    drone_x: float,
    drone_y: float,
    drone_altitude: float,
    drone_yaw: float,
    n_rays: int,
    half_size: float,
) -> FloatArray:
    """N horizontal rays from the drone, angles yaw + k*2pi/N, against groups 0/1 (walls,
    pillars); distance normalised to [0, 1] by 2*half_size, 1.0 if nothing hit.
    """
    out = np.empty(n_rays, dtype=np.float64)
    pnt = np.array([drone_x, drone_y, drone_altitude], dtype=np.float64).reshape(3, 1)
    geomid = np.zeros(1, dtype=np.int32)
    for k in range(n_rays):
        angle = drone_yaw + k * 2.0 * math.pi / n_rays
        vec = np.array([math.cos(angle), math.sin(angle), 0.0], dtype=np.float64).reshape(3, 1)
        hit = mujoco.mj_ray(model, data, pnt, vec, STATIC_GEOMGROUP.reshape(6, 1), 1, -1, geomid)
        out[k] = 1.0 if hit < 0.0 else float(np.clip(hit / (2.0 * half_size), 0.0, 1.0))
    return out
