"""Reward vs retention for static / scripted / ppo ckpt at the training dial and the eval dial."""
import glob
import numpy as np
from lockon.core import Difficulty
from lockon.harness.episode import run_episode, resolve_hunter
from lockon.policy.features import RewardConfig, reward
from lockon.policy.prey import ScriptedPrey

cfg = RewardConfig()
ckpts = sorted(glob.glob(r"runs/ppo_local/ckpt_*.zip"), key=lambda p: int(p.split("_")[-1].split(".")[0]))
policies = ["static", "scripted"] + ckpts[-1:]


def dial(d: float) -> Difficulty:
    return Difficulty(d, d, d, d, d)


for d in (0.3, 0.5):
    print(f"--- dial {d}")
    for pol in policies:
        rews, vis, rets, acts = [], [], [], []
        for seed in range(8):
            hunter = resolve_hunter(pol)
            res = run_episode(dial(d), 100 + seed, hunter, ScriptedPrey())
            ssl, r_ep = 0, 0.0
            for st, a in zip(res.states, res.actions):
                ssl = 0 if st.person_visible else ssl + 1
                r_ep += reward(st, a, ssl, cfg)
            rews.append(r_ep)
            vis.append(np.mean([s.person_visible for s in res.states]))
            rets.append(res.retention.retention)
            acts.append(np.mean([np.abs(a.as_array()) for a in res.actions], axis=0))
        print(f"{pol[-22:]:24s} reward {np.mean(rews):7.1f}  visible {np.mean(vis):.3f}  retention {np.mean(rets):.3f}  |a| {np.round(np.mean(acts, axis=0), 2)}")
