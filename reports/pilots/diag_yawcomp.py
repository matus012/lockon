"""Does yaw compensation of detections rescue lock under camera jitter? Static+noise policy."""
import math
import numpy as np
from lockon.core import Difficulty, IMAGE_HEIGHT, AgentAction, Detection
from lockon.core.metrics import retention
from lockon.env.env import Env
from lockon.harness.episode import EVAL_NOISE, _gt_detections
from lockon.policy.prey import ScriptedPrey
from lockon.track import LockTracker, NoiseConfig, NoiseInjector

FOVY = 60.0
F = (IMAGE_HEIGHT / 2) / math.tan(math.radians(FOVY / 2))


def run(seed: int, noise_std: float, comp: bool, d: float = 0.5) -> float:
    diff = Difficulty(d, d, d, d, d)
    env = Env(diff, seed=seed)
    st = env.reset()
    prey = ScriptedPrey(); prey.reset(env.layout, diff, seed)
    rng = np.random.default_rng(seed)
    inj = NoiseInjector(NoiseConfig(**{**vars(EVAL_NOISE), "seed": seed}))
    trk = LockTracker(); trk.reset()
    yaw0 = st.drone.yaw
    gts, trs = [], []
    for t in range(200):
        a = AgentAction.from_array(rng.normal(0.0, noise_std, size=3)) if noise_std > 0 else AgentAction.zero()
        st = env.step(a, prey.act(env.state(), env.illumination))
        dets = inj(_gt_detections(st), t)
        dyaw = (st.drone.yaw - yaw0 + math.pi) % (2 * math.pi) - math.pi
        shift = -F * dyaw if comp else 0.0  # yaw left (+) moves image content to +u; subtract it
        dets_c = [Detection(box=dd.box + np.array([shift, 0, shift, 0]), score=dd.score, channel=dd.channel, t=dd.t) for dd in dets]
        tracks, _ = trk.update(dets_c, t)
        gt = st.person_box + np.array([shift, 0, shift, 0]) if (st.person_visible and st.person_box is not None) else None
        gts.append(gt); trs.append(tracks)
    return retention(gts, trs).retention


for std in (0.0, 0.37):
    for comp in (False, True):
        r = [run(s, std, comp) for s in range(10)]
        print(f"noise std {std:.2f} yaw-comp {comp!s:5s}: retention mean {np.mean(r):.3f} min {min(r):.3f}")
