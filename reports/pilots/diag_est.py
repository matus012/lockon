"""ID-hold rate across 20 seeds x 3 state estimators (occlusion scene, noise dial 0.3 and 0.5)."""
import sys

sys.path.insert(0, r"C:\Users\matus\A_MAIN\ws\110_lockon\tests")
import numpy as np
import test_track as tt  # noqa: E402
from trackers.utils.state_representations import (  # noqa: E402
    XCYCSRStateEstimator,
    XCYCWHStateEstimator,
    XYXYStateEstimator,
)

from lockon.core.metrics import retention  # noqa: E402
from lockon.track import LockTracker, NoiseConfig, NoiseInjector  # noqa: E402


def run(est, seed: int, dial: float, scene_fn) -> tuple[float, int]:
    cfg = NoiseConfig.from_dial(dial, seed=seed)
    inj, trk = NoiseInjector(cfg), LockTracker()
    trk._tracker.state_estimator_class = est
    trk.reset()
    trk._tracker.state_estimator_class = est
    gts, trs, ids = [], [], set()
    for t, gt in enumerate(scene_fn()):
        tr, _ = trk.update(inj(gt, t), t)
        trs.append(tr)
        ids |= {x.track_id for x in tr}
        gts.append(next((d.box for d in gt.values() if d is not None), None))
    return retention(gts, trs).retention, len(ids)


for name, est in (("XYXY(default)", XYXYStateEstimator), ("XCYCSR(official)", XCYCSRStateEstimator), ("XCYCWH", XCYCWHStateEstimator)):
    for dial in (0.3, 0.5):
        for sname, fn in (("occlusion", tt.occlusion_scene), ("lights_cut", tt.lights_cut_scene)):
            rs = [run(est, s, dial, fn) for s in range(20)]
            held = sum(1 for r in rs if r[1] == 1)
            print(f"{name:17s} dial={dial} {sname:10s}: held {held}/20, mean {np.mean([r[0] for r in rs]):.3f}, min {min(r[0] for r in rs):.3f}")
