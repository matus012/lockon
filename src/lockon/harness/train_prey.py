"""Learned-prey training CLI (SPEC_prey.md `train_prey.py`, plan.md §5 step 7 GPU line).

    uv run python -m lockon.harness.train_prey --config configs/prey_gpu.yaml --out runs/prey \\
        --hunter scripted|<zip> [--device cuda|cpu] [--wall-hours 18] [--resume]

Same skeleton as `train.py`: SubprocVecEnv(n_envs) of `PreyGym` -> VecMonitor ->
VecFrameStack(frame_stack) -> `make_ppo`. The frozen hunter is fixed for the run (`--hunter`);
`best.zip` = the checkpoint giving the hunter the LOWEST retention (SPEC_prey.md: "lower =
better prey"). Environment probe (§5): on `--device cuda` assert `torch.cuda.is_available()`
and log the device before learning; a CPU fallback on the GPU job is a STOP (exit 3).
"""

from __future__ import annotations

import argparse
import json
import logging
import re
import shutil
import sys
import tempfile
import time
from pathlib import Path
from typing import Any

import torch
from stable_baselines3 import PPO
from stable_baselines3.common.callbacks import BaseCallback
from stable_baselines3.common.vec_env import SubprocVecEnv, VecFrameStack, VecMonitor

from lockon.core.schemas import Difficulty
from lockon.harness.episode import hunter_frame_stack, resolve_hunter
from lockon.harness.eval import evaluate
from lockon.harness.prey_gym import PreyGym
from lockon.policy.base import Hunter
from lockon.policy.hunters import PPOHunter
from lockon.policy.prey_learned import PreyRewardConfig

logger = logging.getLogger(__name__)

_CKPT_RE = re.compile(r"^ckpt_(\d+)\.zip$")

_DIFFICULTY_DIAL = 0.5  # SPEC_prey.md configs/prey_gpu.yaml: "curriculum none - fixed mid difficulty"


def _reward_config(raw: dict[str, Any]) -> PreyRewardConfig:
    known = {"visible_penalty", "action_l2", "transition_bonus"}
    return PreyRewardConfig(**{k: v for k, v in raw.items() if k in known})


def _make_env_fn(
    seed: int, hunter_name: str, hunter_n_stack: int, reward_cfg: PreyRewardConfig
) -> Any:
    def _init() -> PreyGym:
        hunter: Hunter = resolve_hunter(hunter_name)
        return PreyGym(
            Difficulty(
                occluder_density=_DIFFICULTY_DIAL,
                darkness=_DIFFICULTY_DIAL,
                prey_speed=_DIFFICULTY_DIAL,
                prey_aggressiveness=_DIFFICULTY_DIAL,
                channel_dropout=_DIFFICULTY_DIAL,
            ),
            seed=seed,
            hunter=hunter,
            hunter_frame_stack=hunter_n_stack,
            reward=reward_cfg,
        )

    return _init


def _hunter_n_stack(hunter_name: str) -> int:
    """`resolve_hunter` + a throwaway `reset()` so `PPOHunter.n_stack` is known before the
    SubprocVecEnv workers spawn (each worker builds its own frozen hunter instance). The
    rule-based hunters never touch the loaded model, so only `PPOHunter` needs the probe (its
    `reset()` ignores `layout`, unlike `ScriptedHunter`'s, so passing `None` is safe there only)."""
    probe = resolve_hunter(hunter_name)
    if isinstance(probe, PPOHunter):
        probe.reset(layout=None, seed=0)  # type: ignore[arg-type]
        return hunter_frame_stack(probe)
    return 1


def build_vec_env(
    n_envs: int, frame_stack: int, seed: int, hunter_name: str, reward_cfg: PreyRewardConfig
) -> VecFrameStack:
    hunter_n_stack = _hunter_n_stack(hunter_name)
    fns = [_make_env_fn(seed + i, hunter_name, hunter_n_stack, reward_cfg) for i in range(n_envs)]
    subproc = SubprocVecEnv(fns, start_method="spawn")
    monitored = VecMonitor(subproc)
    return VecFrameStack(monitored, n_stack=frame_stack)


class CheckpointCallback(BaseCallback):
    """Saves `runs/<name>/ckpt_<steps>.zip` every `checkpoint_every` env steps."""

    def __init__(self, out_dir: Path, checkpoint_every: int, verbose: int = 0) -> None:
        super().__init__(verbose)
        self._out_dir = out_dir
        self._checkpoint_every = checkpoint_every
        self._last_saved = -1

    def _on_step(self) -> bool:
        steps = self.model.num_timesteps
        if steps - self._last_saved >= self._checkpoint_every:
            path = self._out_dir / f"ckpt_{steps}.zip"
            self.model.save(str(path))
            logger.info("checkpoint saved: %s", path)
            self._last_saved = steps
        return True


class WallClockStopCallback(BaseCallback):
    """Stops training once `wall_hours` have elapsed."""

    def __init__(self, wall_hours: float, verbose: int = 0) -> None:
        super().__init__(verbose)
        self._deadline = time.perf_counter() + wall_hours * 3600.0

    def _on_step(self) -> bool:
        if time.perf_counter() >= self._deadline:
            logger.info("wall-clock cap reached at step %d; stopping", self.model.num_timesteps)
            return False
        return True


class PreyRetentionEvalCallback(BaseCallback):
    """Every `eval_every` steps: save a temp checkpoint + sidecar, run the FROZEN `hunter`
    against it on `eval_episodes` seeds, log the resulting retention, and promote to
    `best.zip` when retention is the LOWEST seen so far (SPEC_prey.md: lower = better prey)."""

    def __init__(
        self,
        out_dir: Path,
        hunter_name: str,
        eval_every: int,
        eval_episodes: int,
        verbose: int = 0,
    ) -> None:
        super().__init__(verbose)
        self._out_dir = out_dir
        self._hunter_name = hunter_name
        self._eval_every = eval_every
        self._eval_episodes = eval_episodes
        self._last_eval = 0
        self._best_retention = float("inf")
        self._log_path = out_dir / "train_log.jsonl"

    def _on_step(self) -> bool:
        steps = self.model.num_timesteps
        if steps - self._last_eval < self._eval_every:
            return True
        self._last_eval = steps

        with tempfile.TemporaryDirectory() as tmp:
            tmp_zip = Path(tmp) / "eval.zip"
            self.model.save(str(tmp_zip))
            Path(str(tmp_zip) + ".prey.json").write_text(
                json.dumps({"hunter": self._hunter_name, "obs": "prey_v1"}), encoding="utf-8"
            )
            result = evaluate(
                self._hunter_name, Difficulty(), range(self._eval_episodes), prey=str(tmp_zip)
            )

        hunter_retention = result["retention_mean"]
        row = {
            "step": steps,
            "wall_s": time.perf_counter(),
            "hunter_retention_mean": hunter_retention,
        }
        with open(self._log_path, "a", encoding="utf-8") as f:
            f.write(json.dumps(row) + "\n")
        print(f"[eval] step={steps} hunter_retention={hunter_retention:.4f}")

        if hunter_retention < self._best_retention:
            self._best_retention = hunter_retention
            best_path = self._out_dir / "best.zip"
            self.model.save(str(best_path))
            Path(str(best_path) + ".prey.json").write_text(
                json.dumps({"hunter": self._hunter_name, "obs": "prey_v1"}), encoding="utf-8"
            )
            logger.info("best.zip updated: hunter_retention=%.4f (step %d)", hunter_retention, steps)
        return True


class ThroughputLogCallback(BaseCallback):
    """Logs `steps/s` once, after the first rollout completes (environment probe, §5)."""

    def __init__(self, device: str, verbose: int = 0) -> None:
        super().__init__(verbose)
        self._device = device
        self._start = time.perf_counter()
        self._logged = False

    def _on_rollout_end(self) -> None:
        if self._logged:
            return
        elapsed = time.perf_counter() - self._start
        sps = self.model.num_timesteps / elapsed if elapsed > 0 else float("inf")
        msg = (
            f"steps/s={sps:.1f} (steps={self.model.num_timesteps} elapsed={elapsed:.1f}s "
            f"device={self._device})"
        )
        logger.info(msg)
        print(msg)
        self._logged = True

    def _on_step(self) -> bool:
        return True


def _newest_checkpoint(out_dir: Path) -> Path:
    candidates: list[tuple[int, Path]] = []
    for p in out_dir.glob("ckpt_*.zip"):
        m = _CKPT_RE.match(p.name)
        if m:
            candidates.append((int(m.group(1)), p))
    if not candidates:
        raise FileNotFoundError(f"--resume: no ckpt_*.zip found in {out_dir}")
    candidates.sort(key=lambda x: x[0])
    return candidates[-1][1]


def _resolve_device(requested: str | None) -> str:
    """Environment probe (§5): `--device cuda` STOPs (exit 3) on no CUDA; default logs the pick."""
    if requested == "cuda":
        if not torch.cuda.is_available():
            print(
                "STOP: --device cuda requested but torch.cuda.is_available() is False; "
                "GPU fallback to CPU is a blocker, not a grind (refactored_method.md §5).",
                file=sys.stderr,
            )
            raise SystemExit(3)
        logger.info("device=cuda (torch.cuda.is_available()=True)")
        return "cuda"
    if requested == "cpu":
        logger.info("device=cpu (explicit)")
        return "cpu"
    device = "cuda" if torch.cuda.is_available() else "cpu"
    logger.info("device=%s (auto: torch.cuda.is_available()=%s)", device, torch.cuda.is_available())
    return device


def train_prey(
    config_path: str,
    out: str,
    hunter_name: str,
    *,
    device: str | None = None,
    resume: bool = False,
    wall_hours: float | None = None,
    total_steps: int | None = None,
    n_envs: int | None = None,
) -> Path:
    import yaml

    with open(config_path, encoding="utf-8") as f:
        cfg: dict[str, Any] = yaml.safe_load(f)

    resolved_device = _resolve_device(device)
    if resolved_device == "cuda":
        assert torch.cuda.is_available(), "device probe passed but CUDA vanished before training"

    out_dir = Path(out)
    out_dir.mkdir(parents=True, exist_ok=True)

    seed = int(cfg.get("seed", 0))
    total = total_steps if total_steps is not None else int(cfg.get("total_steps", 20_000_000))
    envs = n_envs if n_envs is not None else int(cfg.get("n_envs", 32))
    frame_stack = int(cfg.get("frame_stack", 4))
    checkpoint_every = int(cfg.get("checkpoint_every", 250_000))
    eval_every = int(cfg.get("eval_every", 500_000))
    eval_episodes = int(cfg.get("eval_episodes", 20))
    wall = wall_hours if wall_hours is not None else float(cfg.get("wall_hours", 18.0))
    reward_cfg = _reward_config(cfg.get("reward", {}) or {})

    env = build_vec_env(envs, frame_stack, seed, hunter_name, reward_cfg)

    if resume:
        ckpt = _newest_checkpoint(out_dir)
        model = PPO.load(str(ckpt), env=env, device=resolved_device)
        logger.info("resumed from %s at step %d", ckpt, model.num_timesteps)
        remaining = max(total - model.num_timesteps, 0)
    else:
        model = PPO(
            "MlpPolicy",
            env,
            learning_rate=float(cfg.get("learning_rate", 3e-4)),
            n_steps=int(cfg.get("n_steps", 1024)),
            batch_size=int(cfg.get("batch_size", 256)),
            n_epochs=int(cfg.get("n_epochs", 10)),
            gamma=float(cfg.get("gamma", 0.99)),
            gae_lambda=float(cfg.get("gae_lambda", 0.95)),
            ent_coef=float(cfg.get("ent_coef", 0.0)),
            clip_range=float(cfg.get("clip_range", 0.2)),
            policy_kwargs={"net_arch": list(cfg.get("net_arch", [128, 128]))},
            seed=seed,
            device=resolved_device,
            verbose=0,
        )
        remaining = total

    logger.info("torch device=%s", model.device)

    callbacks = [
        ThroughputLogCallback(resolved_device),
        CheckpointCallback(out_dir, checkpoint_every),
        WallClockStopCallback(wall),
        PreyRetentionEvalCallback(out_dir, hunter_name, eval_every, eval_episodes),
    ]

    if remaining > 0:
        model.learn(total_timesteps=remaining, reset_num_timesteps=not resume, callback=callbacks)
    else:
        logger.info("resume: already at/past total_steps=%d, nothing to train", total)

    final_path = out_dir / f"ckpt_{model.num_timesteps}.zip"
    model.save(str(final_path))
    logger.info("final checkpoint: %s", final_path)

    if not (out_dir / "best.zip").exists():
        shutil.copy(final_path, out_dir / "best.zip")
        Path(str(out_dir / "best.zip") + ".prey.json").write_text(
            json.dumps({"hunter": hunter_name, "obs": "prey_v1"}), encoding="utf-8"
        )
        logger.info("best.zip absent after run; copied final checkpoint as fallback")

    env.close()
    return out_dir


def main(argv: list[str] | None = None) -> int:
    logging.basicConfig(level=logging.INFO)
    parser = argparse.ArgumentParser(description="lockon.harness learned-prey PPO training")
    parser.add_argument("--config", required=True)
    parser.add_argument("--out", required=True)
    parser.add_argument("--hunter", default="scripted", help="frozen hunter: static|scripted|<zip>")
    parser.add_argument("--device", choices=["cuda", "cpu"], default=None)
    parser.add_argument("--resume", action="store_true")
    parser.add_argument("--wall-hours", type=float, default=None)
    parser.add_argument("--total-steps", type=int, default=None)
    parser.add_argument("--n-envs", type=int, default=None)
    args = parser.parse_args(argv)

    train_prey(
        args.config,
        args.out,
        args.hunter,
        device=args.device,
        resume=args.resume,
        wall_hours=args.wall_hours,
        total_steps=args.total_steps,
        n_envs=args.n_envs,
    )
    return 0


if __name__ == "__main__":
    sys.exit(main())
