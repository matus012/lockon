"""Step-1 render probe: MuJoCo offscreen rgb + depth + segmentation at 640x480 on this box.

Exit 0 = renderer works headless (GLFW/WGL on Windows). Exit 1 = it does not; that is a
blockers.md row, not a retry (plan.md §5 step 1).
"""

from __future__ import annotations

import logging
import os
import sys
import time

import mujoco
import numpy as np

_XML = """
<mujoco>
  <visual><global offwidth="640" offheight="480"/></visual>
  <worldbody>
    <light pos="0 0 5" dir="0 0 -1" diffuse="1 1 1"/>
    <geom type="plane" size="10 10 0.1" rgba="0.3 0.3 0.3 1"/>
    <geom name="pillar" type="box" pos="1 0 1" size="0.3 0.3 1" rgba="0.6 0.5 0.4 1"/>
    <body name="person" pos="0 0 0.9">
      <geom name="torso" type="capsule" size="0.18 0.35" rgba="0.8 0.2 0.2 1"/>
    </body>
    <camera name="drone" pos="0 -5 3" xyaxes="1 0 0 0 0.6 0.8"/>
  </worldbody>
</mujoco>
"""


def main() -> int:
    logging.basicConfig(level=logging.INFO, format="%(levelname)s %(message)s")
    log = logging.getLogger("probe_render")
    log.info("MUJOCO_GL=%s mujoco=%s", os.environ.get("MUJOCO_GL", "<unset>"), mujoco.__version__)
    model = mujoco.MjModel.from_xml_string(_XML)
    data = mujoco.MjData(model)
    mujoco.mj_forward(model, data)
    renderer = mujoco.Renderer(model, height=480, width=640)

    t0 = time.perf_counter()
    renderer.update_scene(data, camera="drone")
    rgb = renderer.render()
    renderer.enable_depth_rendering()
    renderer.update_scene(data, camera="drone")
    depth = renderer.render()
    renderer.disable_depth_rendering()
    renderer.enable_segmentation_rendering()
    renderer.update_scene(data, camera="drone")
    seg = renderer.render()
    renderer.disable_segmentation_rendering()
    dt = time.perf_counter() - t0

    ok = (
        rgb.shape == (480, 640, 3)
        and rgb.max() > 0
        and depth.shape == (480, 640)
        and np.isfinite(depth).all()
        and seg.shape == (480, 640, 2)
        and (seg[..., 0] >= 0).any()
    )
    log.info(
        "rgb %s max=%d | depth %s min=%.2f max=%.2f | seg %s ids=%s | 3 passes %.3fs",
        rgb.shape, int(rgb.max()), depth.shape, float(depth.min()), float(depth.max()),
        seg.shape, sorted(set(np.unique(seg[..., 0]).tolist()))[:8], dt,
    )
    log.info("PROBE %s", "OK" if ok else "FAIL")
    renderer.close()
    return 0 if ok else 1


if __name__ == "__main__":
    sys.exit(main())
