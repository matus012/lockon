"""Short verbose PPO run through the real training stack; print SB3 stats per rollout."""
import sys
import logging
import numpy as np
from stable_baselines3.common.callbacks import BaseCallback
from stable_baselines3.common.vec_env import DummyVecEnv, VecFrameStack, VecMonitor

from lockon.core import Difficulty
from lockon.harness.gym_env import LockonGym
from lockon.policy.features import RewardConfig
from lockon.policy.ppo import PPOConfig, make_ppo
from lockon.policy.prey import ScriptedPrey

logging.basicConfig(level=logging.WARNING)
mode = sys.argv[1] if len(sys.argv) > 1 else "lock"
cfg = PPOConfig.from_yaml("configs/ppo_local.yaml")
d = Difficulty(0.3, 0.3, 0.3, 0.3, 0.3)


class ActionOnlyGym(LockonGym):
    def step(self, action):
        obs, r, term, trunc, info = super().step(action)
        a = np.asarray(action, dtype=np.float64)
        return obs, -float(np.dot(a, a)), term, trunc, info


cls = ActionOnlyGym if mode == "action" else LockonGym
n_envs = 8
venv = VecFrameStack(VecMonitor(DummyVecEnv([lambda i=i: cls(d, seed=i, prey=ScriptedPrey(), reward=RewardConfig()) for i in range(n_envs)])), 4)
model = make_ppo(venv, seed=0, cfg=cfg)
model.verbose = 0


class Stats(BaseCallback):
    def _on_rollout_end(self) -> None:
        ep = self.model.ep_info_buffer
        rew = np.mean([e["r"] for e in ep]) if ep else float("nan")
        std = float(np.exp(self.model.policy.log_std.detach()).mean())
        obs = self.model.rollout_buffer.observations
        print(f"steps {self.model.num_timesteps:7d} ep_rew {rew:8.2f} policy_std {std:.3f} obs[min,max]=({obs.min():.2f},{obs.max():.2f}) nan={np.isnan(obs).any()}", flush=True)

    def _on_step(self) -> bool:
        return True


model.learn(total_timesteps=8192 * 8, callback=Stats())
for k in ("train/value_loss", "train/explained_variance", "train/approx_kl", "train/entropy_loss", "train/policy_gradient_loss"):
    print(k, model.logger.name_to_value.get(k))
