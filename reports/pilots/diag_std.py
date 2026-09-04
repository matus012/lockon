"""Noisy-static policy: retention and state visibility vs exploration std (filtered commands)."""
import numpy as np
from lockon.core import Difficulty, AgentAction
from lockon.core.metrics import retention
from lockon.env.env import Env
from lockon.harness.episode import EVAL_NOISE, _gt_detections
from lockon.policy.prey import ScriptedPrey
from lockon.track import LockTracker, NoiseConfig, NoiseInjector


def run(seed: int, std: float, d: float = 0.5):
    diff = Difficulty(d, d, d, d, d)
    env = Env(diff, seed=seed); st = env.reset()
    prey = ScriptedPrey(); prey.reset(env.layout, diff, seed)
    rng = np.random.default_rng(seed)
    inj = NoiseInjector(NoiseConfig(**{**vars(EVAL_NOISE), "seed": seed})); trk = LockTracker(); trk.reset()
    gts, trs, vis, occl, fov = [], [], [], 0, 0
    for t in range(200):
        a = AgentAction.from_array(rng.normal(0.0, std, size=3)) if std > 0 else AgentAction.zero()
        st = env.step(a, prey.act(env.state(), env.illumination))
        tracks, _ = trk.update(inj(_gt_detections(st), t), t)
        gts.append(st.person_box if st.person_visible else None); trs.append(tracks); vis.append(st.person_visible)
        if not st.person_visible:
            if st.person_box is None: fov += 1
            else: occl += 1
    return retention(gts, trs).retention, float(np.mean(vis)), fov / 200, occl / 200


for std in (0.0, 0.1, 0.2, 0.37):
    r = np.array([run(s, std) for s in range(12)])
    print(f"std {std:.2f}: retention {r[:,0].mean():.3f}  visible {r[:,1].mean():.3f}  out-of-fov {r[:,2].mean():.3f}  occluded {r[:,3].mean():.3f}")
