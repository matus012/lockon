"""Same checkpoint through the SB3 training stack vs the harness eval path."""
import glob
import numpy as np
from stable_baselines3 import PPO
from stable_baselines3.common.vec_env import DummyVecEnv, VecFrameStack

from lockon.core import Difficulty
from lockon.harness.episode import run_episode, resolve_hunter
from lockon.harness.gym_env import LockonGym
from lockon.policy.features import RewardConfig
from lockon.policy.prey import ScriptedPrey

ck = sorted(glob.glob(r"runs/ppo_local/ckpt_*.zip"), key=lambda p: int(p.split("_")[-1].split(".")[0]))[-1]
print("ckpt", ck)
model = PPO.load(ck, device="cpu")
print("obs space", model.observation_space.shape, "n_stack expected", model.observation_space.shape[0] // 25)


def dial(d):
    return Difficulty(d, d, d, d, d)


for d in (0.3, 0.5):
    # training path
    venv = VecFrameStack(DummyVecEnv([lambda d=d: LockonGym(dial(d), seed=0, prey=ScriptedPrey(), reward=RewardConfig())]), 4)
    locked_fracs, rews, acts = [], [], []
    for ep in range(6):
        obs = venv.reset()
        venv.env_method("reset", seed=500 + ep)  # reseed episode deterministically
        obs = venv.reset()
        done = False
        lk, rw, aa = [], 0.0, []
        while not done:
            a, _ = model.predict(obs, deterministic=True)
            obs, r, dones, infos = venv.step(a)
            lk.append(infos[0]["locked"]); rw += float(r[0]); aa.append(np.abs(a[0]))
            done = bool(dones[0])
        locked_fracs.append(np.mean(lk)); rews.append(rw); acts.append(np.mean(aa, axis=0))
    # harness path
    rets, vis = [], []
    for ep in range(6):
        res = run_episode(dial(d), 500 + ep, resolve_hunter(ck), ScriptedPrey())
        rets.append(res.retention.retention); vis.append(np.mean([s.person_visible for s in res.states]))
    print(f"dial {d}: TRAIN-PATH locked {np.mean(locked_fracs):.3f} reward {np.mean(rews):6.1f} |a| {np.round(np.mean(acts,axis=0),2)} | HARNESS retention {np.mean(rets):.3f} visible {np.mean(vis):.3f}")

# static through the training path for reference
venv = VecFrameStack(DummyVecEnv([lambda: LockonGym(dial(0.5), seed=0, prey=ScriptedPrey(), reward=RewardConfig())]), 4)
lks = []
for ep in range(6):
    venv.env_method("reset", seed=500 + ep); obs = venv.reset(); done = False; lk = []
    while not done:
        obs, r, dones, infos = venv.step(np.zeros((1, 3), dtype=np.float32)); lk.append(infos[0]["locked"]); done = bool(dones[0])
    lks.append(np.mean(lk))
print(f"static TRAIN-PATH locked at 0.5: {np.mean(lks):.3f}")
