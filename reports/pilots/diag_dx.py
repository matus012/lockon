"""ID-hold rate vs box speed dx across 20 seeds at noise dial 0.3 (occlusion scene)."""
import sys

sys.path.insert(0, r"C:\Users\matus\A_MAIN\ws\110_lockon\tests")
import numpy as np
import test_track as tt  # noqa: E402

from lockon.core.metrics import retention  # noqa: E402
from lockon.track import LockTracker, NoiseConfig, NoiseInjector  # noqa: E402


def scene(dx: float):
    base = tt.walk_boxes(120, dx=dx)
    out = []
    for t, gt in enumerate(base):
        out.append(dict.fromkeys(gt, None) if 40 <= t < 60 else gt)
    return out


def run(dx: float, seed: int, dial: float = 0.3) -> tuple[float, int]:
    cfg = NoiseConfig.from_dial(dial, seed=seed)
    inj, trk = NoiseInjector(cfg), LockTracker()
    trk.reset()
    gts, trs, ids = [], [], set()
    for t, gt in enumerate(scene(dx)):
        tr, _ = trk.update(inj(gt, t), t)
        trs.append(tr)
        ids |= {x.track_id for x in tr}
        gts.append(next((d.box for d in gt.values() if d is not None), None))
    return retention(gts, trs).retention, len(ids)


for dx in (2.0, 4.0, 6.0, 8.0):
    rs = [run(dx, s) for s in range(20)]
    held = sum(1 for r in rs if r[1] == 1)
    print(f"dx={dx}: held {held}/20, retention mean {np.mean([r[0] for r in rs]):.3f}, min {min(r[0] for r in rs):.3f}")
