"""env/SPEC.md `Tests` 8 — transform proof: render segmentation at the drone camera, take the
pixel bbox of person geoms, and check it against the analytic `person_box` (instrument-proof-
every-transform). Needs the MuJoCo offscreen renderer (GLFW here, MUJOCO_GL unset).
"""

from __future__ import annotations

import mujoco
import numpy as np
import pytest

from lockon.core.metrics import iou_xyxy
from lockon.core.schemas import AgentAction, Difficulty
from lockon.env.env import Env

pytestmark = pytest.mark.render

IOU_MIN = 0.7
N_STATES = 20


def _person_geom_ids(model: mujoco.MjModel) -> set[int]:
    ids = set()
    for i in range(model.ngeom):
        name = mujoco.mj_id2name(model, mujoco.mjtObj.mjOBJ_GEOM, i)
        if name is not None and name.startswith("person_"):
            ids.add(i)
    return ids


def _render_person_bbox(
    renderer: mujoco.Renderer, model: mujoco.MjModel, data: mujoco.MjData, cam: str, person_ids: set[int]
) -> tuple[float, float, float, float] | None:
    opt = mujoco.MjvOption()
    opt.geomgroup[:] = 1  # render every group, including person's group 3 (off by default)
    renderer.update_scene(data, camera=cam, scene_option=opt)
    renderer.enable_segmentation_rendering()
    seg = renderer.render()
    renderer.disable_segmentation_rendering()
    mask = np.isin(seg[:, :, 0], list(person_ids))
    ys, xs = np.nonzero(mask)
    if xs.size == 0:
        return None
    return float(xs.min()), float(ys.min()), float(xs.max() + 1), float(ys.max() + 1)


def test_segmentation_bbox_matches_projected_box() -> None:
    """`person_box` projects 8 AABB corners (a box superset of the true silhouette, looser under
    perspective and any partial occlusion the naive box ignores) — so the transform proof is the
    *mean* IoU over N_STATES random in-frame states, not a per-sample gate (a real transform bug
    collapses the mean toward 0, which this still catches).
    """
    env = Env(Difficulty(), seed=17)
    renderer = mujoco.Renderer(env.model, height=480, width=640)
    person_ids = _person_geom_ids(env.model)

    rng = np.random.default_rng(555)
    ious: list[float] = []
    tries = 0
    max_speed = env.person_max_speed()
    while len(ious) < N_STATES and tries < 500:
        tries += 1
        action = AgentAction.from_array(rng.uniform(-1.0, 1.0, size=3))
        heading = float(rng.uniform(-np.pi, np.pi))
        speed = float(rng.uniform(0.0, max_speed))
        st = env.step(action, (speed * np.cos(heading), speed * np.sin(heading)))
        if st.person_box is None:
            continue

        rendered = _render_person_bbox(renderer, env.model, env.data, env.CAMERA, person_ids)
        assert rendered is not None, "person_box is not None but no person pixels rendered"
        ious.append(iou_xyxy(np.array(rendered), st.person_box))

    assert len(ious) == N_STATES, f"only found {len(ious)}/{N_STATES} in-frame states in {tries} steps"
    mean_iou = float(np.mean(ious))
    assert mean_iou >= IOU_MIN, f"mean IoU {mean_iou:.3f} < {IOU_MIN}; per-state: {ious}"
