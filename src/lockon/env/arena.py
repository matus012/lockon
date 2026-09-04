"""Arena layout + MJCF builder (SPEC.md `Arena`). Pure geometry — no MuJoCo objects held here;
`env.py` compiles the returned MJCF string into an `MjModel`.
"""

from __future__ import annotations

import math

import numpy as np

from lockon.core.schemas import IMAGE_HEIGHT, IMAGE_WIDTH, PERSON_MATERIAL, ArenaLayout, Difficulty

WALL_MARGIN_M: float = 1.5
CENTER_CLEARANCE_M: float = 2.5
PILLAR_HEIGHT_M: float = 3.5
N_LIGHTS: int = 6
DRONE_ALTITUDE_M: float = 3.0
LIGHT_HEIGHT_M: float = 3.0
CAMERA_PITCH_DEG: float = 25.0
WALL_HEIGHT_M: float = 3.5
WALL_THICKNESS_M: float = 0.1

# Person figure geometry (single source, shared with `geometry.py`). All z are ground-relative
# (person mocap body sits at z=0); x, y are in the body's unrotated local frame.
TORSO_CENTER: tuple[float, float, float] = (0.0, 0.0, 1.05)
TORSO_HALF_LEN: float = 0.30
TORSO_RADIUS: float = 0.15
HEAD_CENTER: tuple[float, float, float] = (0.0, 0.0, 1.58)
HEAD_RADIUS: float = 0.12
LEG_X_OFFSET: float = 0.1
LEG_RADIUS: float = 0.07
LEG_Z_LO: float = 0.1
LEG_Z_HI: float = 0.75
ARM_Y_OFFSET: float = 0.25
ARM_RADIUS: float = 0.05
ARM_Z_LO: float = 0.75
ARM_Z_HI: float = 1.30

# Per-part (local x, local y, radius, z_lo, z_hi) for every capsule/sphere. Each part's local
# (x, y) footprint is a circle (capsules here are vertical, so their cross-section is rotation-
# invariant about z) — so the TRUE world-frame AABB after a yaw rotation is the union, over
# parts, of (R(yaw) @ (x, y)) +/- radius; rotating a single enclosing box instead (as opposed to
# each part) over-estimates the box at non-zero yaw. See `geometry.person_bbox_world`.
PERSON_PARTS: tuple[tuple[float, float, float, float, float], ...] = (
    (TORSO_CENTER[0], TORSO_CENTER[1], TORSO_RADIUS, TORSO_CENTER[2] - TORSO_HALF_LEN - TORSO_RADIUS,
     TORSO_CENTER[2] + TORSO_HALF_LEN + TORSO_RADIUS),
    (HEAD_CENTER[0], HEAD_CENTER[1], HEAD_RADIUS, HEAD_CENTER[2] - HEAD_RADIUS, HEAD_CENTER[2] + HEAD_RADIUS),
    (-LEG_X_OFFSET, 0.0, LEG_RADIUS, LEG_Z_LO - LEG_RADIUS, LEG_Z_HI + LEG_RADIUS),
    (LEG_X_OFFSET, 0.0, LEG_RADIUS, LEG_Z_LO - LEG_RADIUS, LEG_Z_HI + LEG_RADIUS),
    (0.0, ARM_Y_OFFSET, ARM_RADIUS, ARM_Z_LO - ARM_RADIUS, ARM_Z_HI + ARM_RADIUS),
    (0.0, -ARM_Y_OFFSET, ARM_RADIUS, ARM_Z_LO - ARM_RADIUS, ARM_Z_HI + ARM_RADIUS),
)
PERSON_AABB_Z_LO: float = min(p[3] for p in PERSON_PARTS)  # 0.03
PERSON_AABB_Z_HI: float = max(p[4] for p in PERSON_PARTS)  # 1.70


def build_layout(difficulty: Difficulty, rng: np.random.Generator, half_size: float) -> ArenaLayout:
    """Rejection-sampled pillars + uniform lights (SPEC.md `Arena`)."""
    m = round(4 + 20 * difficulty.occluder_density)
    pillars: list[tuple[float, float, float]] = []
    max_tries = 300
    for _ in range(m):
        for _try in range(max_tries):
            hw = float(rng.uniform(0.5, 1.0))
            bound = half_size - WALL_MARGIN_M - hw
            if bound <= 0:
                break
            x = float(rng.uniform(-bound, bound))
            y = float(rng.uniform(-bound, bound))
            if math.hypot(x, y) < CENTER_CLEARANCE_M + hw:
                continue
            if any(abs(x - px) < (hw + phw) and abs(y - py) < (hw + phw) for px, py, phw in pillars):
                continue
            pillars.append((x, y, hw))
            break
    pillars_arr = np.array(pillars, dtype=np.float64).reshape(-1, 3) if pillars else np.zeros((0, 3))

    lights = np.empty((N_LIGHTS, 3), dtype=np.float64)
    for i in range(N_LIGHTS):
        lights[i, 0] = rng.uniform(-half_size + 1.0, half_size - 1.0)
        lights[i, 1] = rng.uniform(-half_size + 1.0, half_size - 1.0)
        lights[i, 2] = rng.uniform(0.6, 1.0)

    return ArenaLayout(
        half_size=half_size,
        pillars=pillars_arr,
        lights=lights,
        drone_altitude=DRONE_ALTITUDE_M,
    )


def _mat_to_quat(rot: np.ndarray) -> tuple[float, float, float, float]:
    """3x3 rotation matrix -> (w, x, y, z) quaternion (MuJoCo convention). Standard trace method."""
    trace = rot[0, 0] + rot[1, 1] + rot[2, 2]
    if trace > 0:
        s = 0.5 / math.sqrt(trace + 1.0)
        w = 0.25 / s
        x = (rot[2, 1] - rot[1, 2]) * s
        y = (rot[0, 2] - rot[2, 0]) * s
        z = (rot[1, 0] - rot[0, 1]) * s
    elif rot[0, 0] > rot[1, 1] and rot[0, 0] > rot[2, 2]:
        s = 2.0 * math.sqrt(1.0 + rot[0, 0] - rot[1, 1] - rot[2, 2])
        w = (rot[2, 1] - rot[1, 2]) / s
        x = 0.25 * s
        y = (rot[0, 1] + rot[1, 0]) / s
        z = (rot[0, 2] + rot[2, 0]) / s
    elif rot[1, 1] > rot[2, 2]:
        s = 2.0 * math.sqrt(1.0 + rot[1, 1] - rot[0, 0] - rot[2, 2])
        w = (rot[0, 2] - rot[2, 0]) / s
        x = (rot[0, 1] + rot[1, 0]) / s
        y = 0.25 * s
        z = (rot[1, 2] + rot[2, 1]) / s
    else:
        s = 2.0 * math.sqrt(1.0 + rot[2, 2] - rot[0, 0] - rot[1, 1])
        w = (rot[1, 0] - rot[0, 1]) / s
        x = (rot[0, 2] + rot[2, 0]) / s
        y = (rot[1, 2] + rot[2, 1]) / s
        z = 0.25 * s
    return float(w), float(x), float(y), float(z)


def _drone_camera_quat() -> tuple[float, float, float, float]:
    """Camera axes (columns, in body frame) chosen so that, at zero pitch, forward (-z_cam) is
    body +x and up (+y_cam) is body +z; then tilted `CAMERA_PITCH_DEG` further down. See
    env/SPEC.md deviation note in the impl report: a literal rotation about the body x axis
    (colinear with the desired forward direction) cannot pitch the view, so the forward/up
    vectors are built directly from the described end state instead.
    """
    theta = math.radians(CAMERA_PITCH_DEG)
    x_cam = np.array([0.0, -1.0, 0.0])
    y_cam = np.array([math.sin(theta), 0.0, math.cos(theta)])
    z_cam = np.array([-math.cos(theta), 0.0, math.sin(theta)])
    rot = np.column_stack([x_cam, y_cam, z_cam])
    return _mat_to_quat(rot)


def build_mjcf(layout: ArenaLayout) -> str:
    """Static MJCF for the arena (SPEC.md `Arena`). Lights are built at raw intensity; darkness
    scaling is applied at runtime by `Env.set_darkness` via `model.light_diffuse`.
    """
    hs = layout.half_size
    qw, qx, qy, qz = _drone_camera_quat()

    parts: list[str] = []
    parts.append('<mujoco model="lockon_arena">')
    parts.append("  <option gravity=\"0 0 0\"/>")
    parts.append("  <visual>")
    parts.append(f'    <global offwidth="{IMAGE_WIDTH}" offheight="{IMAGE_HEIGHT}"/>')
    parts.append('    <headlight active="0" ambient="0 0 0" diffuse="0 0 0" specular="0 0 0"/>')
    parts.append("  </visual>")
    parts.append("  <asset>")
    parts.append(
        '    <texture name="floor_tex" type="2d" builtin="checker" rgb1="0.2 0.2 0.2" '
        'rgb2="0.3 0.3 0.3" width="300" height="300"/>'
    )
    parts.append(
        '    <material name="floor_mat" texture="floor_tex" texrepeat="10 10" reflectance="0.0"/>'
    )
    parts.append('    <material name="pillar_mat" rgba="0.45 0.35 0.28 1"/>')
    parts.append('    <material name="wall_mat" rgba="0.5 0.5 0.55 1"/>')
    parts.append(f'    <material name="{PERSON_MATERIAL}" rgba="1 1 1 1"/>')
    parts.append("  </asset>")
    parts.append("  <worldbody>")
    parts.append(
        f'    <geom name="floor" type="plane" size="{hs} {hs} 0.1" material="floor_mat" group="0"/>'
    )

    wt = WALL_THICKNESS_M
    wh = WALL_HEIGHT_M / 2.0
    walls = [
        (0.0, hs, hs + wt, wt),
        (0.0, -hs, hs + wt, wt),
        (hs, 0.0, wt, hs + wt),
        (-hs, 0.0, wt, hs + wt),
    ]
    for i, (x, y, sx, sy) in enumerate(walls):
        parts.append(
            f'    <geom name="wall_{i}" type="box" pos="{x} {y} {wh}" size="{sx} {sy} {wh}" '
            'material="wall_mat" group="0"/>'
        )

    ph = PILLAR_HEIGHT_M / 2.0
    for i, (x, y, hw) in enumerate(layout.pillars):
        parts.append(
            f'    <geom name="pillar_{i}" type="box" pos="{x} {y} {ph}" size="{hw} {hw} {ph}" '
            'material="pillar_mat" group="1"/>'
        )

    for i, (x, y, intensity) in enumerate(layout.lights):
        parts.append(
            f'    <light name="light_{i}" pos="{x} {y} {LIGHT_HEIGHT_M}" '
            f'diffuse="{intensity} {intensity} {intensity}" directional="false" castshadow="false"/>'
        )

    parts.append(f'    <body name="drone" mocap="true" pos="0 0 {layout.drone_altitude}">')
    parts.append(
        '      <geom name="drone_body" type="box" size="0.15 0.15 0.08" group="2" '
        'contype="0" conaffinity="0" rgba="0.1 0.1 0.1 1"/>'
    )
    parts.append(
        f'      <camera name="drone_cam" fovy="60" quat="{qw} {qx} {qy} {qz}"/>'
    )
    parts.append("    </body>")

    parts.append('    <body name="person" mocap="true" pos="0 0 0">')
    t0, t1 = TORSO_CENTER[2] - TORSO_HALF_LEN, TORSO_CENTER[2] + TORSO_HALF_LEN
    parts.append(
        f'      <geom name="person_torso" type="capsule" fromto="0 0 {t0} 0 0 {t1}" '
        f'size="{TORSO_RADIUS}" material="{PERSON_MATERIAL}" rgba="0.8 0.2 0.2 1" group="3" '
        'contype="0" conaffinity="0"/>'
    )
    parts.append(
        f'      <geom name="person_head" type="sphere" pos="0 0 {HEAD_CENTER[2]}" '
        f'size="{HEAD_RADIUS}" material="{PERSON_MATERIAL}" rgba="0.9 0.75 0.6 1" group="3" '
        'contype="0" conaffinity="0"/>'
    )
    for side, xoff in (("l", -LEG_X_OFFSET), ("r", LEG_X_OFFSET)):
        parts.append(
            f'      <geom name="person_leg_{side}" type="capsule" '
            f'fromto="{xoff} 0 {LEG_Z_LO} {xoff} 0 {LEG_Z_HI}" size="{LEG_RADIUS}" '
            f'material="{PERSON_MATERIAL}" rgba="0.2 0.2 0.6 1" group="3" contype="0" conaffinity="0"/>'
        )
    for side, yoff in (("l", ARM_Y_OFFSET), ("r", -ARM_Y_OFFSET)):
        parts.append(
            f'      <geom name="person_arm_{side}" type="capsule" '
            f'fromto="0 {yoff} {ARM_Z_LO} 0 {yoff} {ARM_Z_HI}" size="{ARM_RADIUS}" '
            f'material="{PERSON_MATERIAL}" rgba="0.9 0.75 0.6 1" group="3" contype="0" conaffinity="0"/>'
        )
    parts.append("    </body>")

    parts.append("  </worldbody>")
    parts.append("</mujoco>")
    return "\n".join(parts)
