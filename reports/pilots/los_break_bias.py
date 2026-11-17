"""Is the prey a harder examiner for a mover than for the static camera? (README claim, review F-prey)

Arms on the held-out gate seeds 1000-1019, mid difficulty, per-episode noise seeds:
  scripted vs static with the prey's LOS-break ON (the shipped prey) and OFF.
Bias = (scripted - static)_off - (scripted - static)_on. Positive = the LOS-break term costs the
mover more than the floor (the claimed conservative direction).
"""

import json
from pathlib import Path

import numpy as np

from lockon.core import Difficulty
from lockon.harness.episode import resolve_hunter, run_episode
from lockon.policy.prey import ScriptedPrey

SEEDS = range(1000, 1020)


def arm(policy: str, los_break: bool) -> list[float]:
    out = []
    for s in SEEDS:
        res = run_episode(Difficulty(), s, resolve_hunter(policy), ScriptedPrey(los_break=los_break))
        out.append(res.retention.retention)
    return out


rows = {}
for lb in (True, False):
    sc, st = np.array(arm("scripted", lb)), np.array(arm("static", lb))
    d = sc - st
    rows[f"los_break_{'on' if lb else 'off'}"] = {
        "scripted_mean": float(sc.mean()), "static_mean": float(st.mean()),
        "gap_mean": float(d.mean()), "gap_se": float(d.std(ddof=1) / np.sqrt(len(d))), "n": len(d),
    }
    print(f"LOS-break {'on ' if lb else 'off'}: scripted {sc.mean():.3f} static {st.mean():.3f} gap {d.mean():+.3f} ± {d.std(ddof=1)/np.sqrt(len(d)):.3f} (se)")
bias = rows["los_break_off"]["gap_mean"] - rows["los_break_on"]["gap_mean"]
rows["bias_off_minus_on"] = float(bias)
print(f"bias (gap_off - gap_on) = {bias:+.3f}  -> {'conservative for the mover' if bias > 0 else 'NOT conservative'}")
Path("reports/los_break_bias.json").write_text(json.dumps(rows, indent=2), encoding="utf-8")
