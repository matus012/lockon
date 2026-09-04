"""PPO training CLI (SPEC.md `train.py`, plan.md step 5, gate G5).

    uv run python -m lockon.harness.train --config configs/ppo_local.yaml --out runs/ppo_local \\
        [--resume] [--wall-hours 5] [--total-steps N]

SubprocVecEnv(n_envs) of `LockonGym` at the curriculum start dial -> VecMonitor ->
VecFrameStack(frame_stack) -> `make_ppo`. Callbacks: checkpointing, a wall-clock stop, a
curriculum switch, and a retention-eval-against-static callback that tracks `best.zip`.
State-PPO is CPU by design (plan.md §8) — the device is asserted and logged, never GPU.
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

from stable_baselines3.common.callbacks import BaseCallback
from stable_baselines3.common.vec_env import SubprocVecEnv, VecFrameStack, VecMonitor

from lockon.core.schemas import Difficulty
from lockon.harness.eval import evaluate
from lockon.harness.gym_env import LockonGym
from lockon.policy.features import RewardConfig
from lockon.policy.ppo import DEVICE, PPOConfig, make_ppo
from lockon.policy.prey import ScriptedPrey

logger = logging.getLogger(__name__)

_CKPT_RE = re.compile(r"^ckpt_(\d+)\.zip$")


def _difficulty_at(dial: float, overrides: dict[str, float] | None = None) -> Difficulty:
    """All five dials at `dial`, then per-axis `overrides` (config key `difficulty:`), so an HPC
    arena unit can pin occluder_density while the curriculum still ramps the other axes."""
    d = Difficulty(
        occluder_density=dial,
        darkness=dial,
        prey_speed=dial,
        prey_aggressiveness=dial,
        channel_dropout=dial,
    )
    return d.replace(**overrides) if overrides else d


def _reward_config(cfg: PPOConfig) -> RewardConfig:
    raw = cfg.extra.get("reward", {})
    known = {"visible", "action_l2", "lost_penalty", "lost_after"}
    return RewardConfig(**{k: v for k, v in raw.items() if k in known})


def _make_env_fn(seed: int, dial: float, reward_cfg: RewardConfig, overrides: dict[str, float] | None = None) -> Any:
    def _init() -> LockonGym:
        return LockonGym(_difficulty_at(dial, overrides), seed=seed, prey=ScriptedPrey(), reward=reward_cfg)

    return _init


def build_vec_env(cfg: PPOConfig, seed: int, start_dial: float, overrides: dict[str, float] | None = None) -> VecFrameStack:
    reward_cfg = _reward_config(cfg)
    fns = [_make_env_fn(seed + i, start_dial, reward_cfg, overrides) for i in range(cfg.n_envs)]
    subproc = SubprocVecEnv(fns, start_method="spawn")
    monitored = VecMonitor(subproc)
    return VecFrameStack(monitored, n_stack=cfg.frame_stack)


class CheckpointCallback(BaseCallback):
    """Saves `runs/<name>/ckpt_<steps>.zip` every `checkpoint_every` env steps (SPEC.md)."""

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
    """Stops training once `wall_hours` have elapsed (plan.md §5 "wall cap")."""

    def __init__(self, wall_hours: float, verbose: int = 0) -> None:
        super().__init__(verbose)
        self._deadline = time.perf_counter() + wall_hours * 3600.0

    def _on_step(self) -> bool:
        if time.perf_counter() >= self._deadline:
            logger.info("wall-clock cap reached at step %d; stopping", self.model.num_timesteps)
            return False
        return True


class CurriculumCallback(BaseCallback):
    """Switches every sub-env's difficulty to `end_dial` once, at `switch_fraction *
    total_steps` (SPEC.md; applied via `VecEnv.env_method`, takes effect at the next reset).
    """

    def __init__(self, switch_step: int, end_dial: float, verbose: int = 0, overrides: dict[str, float] | None = None) -> None:
        self._overrides = overrides
        super().__init__(verbose)
        self._switch_step = switch_step
        self._end_dial = end_dial
        self._switched = False

    def _on_step(self) -> bool:
        if not self._switched and self.model.num_timesteps >= self._switch_step:
            self.training_env.env_method("set_difficulty", _difficulty_at(self._end_dial, self._overrides))
            logger.info(
                "curriculum switch at step %d: dial -> %.2f", self.model.num_timesteps, self._end_dial
            )
            self._switched = True
        return True


class RetentionEvalCallback(BaseCallback):
    """Every `eval_every` steps: save a temp checkpoint, evaluate it and `static` (cached once)
    on `eval_episodes` seeds at mid difficulty, log both, and promote to `best.zip` on
    improvement (SPEC.md).
    """

    def __init__(
        self,
        out_dir: Path,
        eval_every: int,
        eval_episodes: int,
        verbose: int = 0,
    ) -> None:
        super().__init__(verbose)
        self._out_dir = out_dir
        self._eval_every = eval_every
        self._eval_episodes = eval_episodes
        self._last_eval = 0
        self._best_retention = float("-inf")
        self._static_retention: float | None = None
        self._log_path = out_dir / "train_log.jsonl"

    def _on_step(self) -> bool:
        steps = self.model.num_timesteps
        if steps - self._last_eval < self._eval_every:
            return True
        self._last_eval = steps

        if self._static_retention is None:
            static_result = evaluate("static", Difficulty(), range(self._eval_episodes))
            self._static_retention = static_result["retention_mean"]
            logger.info("static baseline retention_mean=%.4f", self._static_retention)

        with tempfile.TemporaryDirectory() as tmp:
            tmp_zip = Path(tmp) / "eval.zip"
            self.model.save(str(tmp_zip))
            policy_result = evaluate(str(tmp_zip), Difficulty(), range(self._eval_episodes))

        policy_retention = policy_result["retention_mean"]
        row = {
            "step": steps,
            "wall_s": time.perf_counter(),
            "policy_retention_mean": policy_retention,
            "static_retention_mean": self._static_retention,
        }
        with open(self._log_path, "a", encoding="utf-8") as f:
            f.write(json.dumps(row) + "\n")
        print(
            f"[eval] step={steps} policy_retention={policy_retention:.4f} "
            f"static_retention={self._static_retention:.4f}"
        )

        if policy_retention > self._best_retention:
            self._best_retention = policy_retention
            best_path = self._out_dir / "best.zip"
            self.model.save(str(best_path))
            logger.info("best.zip updated: retention=%.4f (step %d)", policy_retention, steps)
        return True


class ThroughputLogCallback(BaseCallback):
    """Logs `steps/s` once, after the first rollout completes (SPEC.md environment probe)."""

    def __init__(self, verbose: int = 0) -> None:
        super().__init__(verbose)
        self._start = time.perf_counter()
        self._logged = False

    def _on_rollout_end(self) -> None:
        if self._logged:
            return
        elapsed = time.perf_counter() - self._start
        sps = self.model.num_timesteps / elapsed if elapsed > 0 else float("inf")
        msg = f"steps/s={sps:.1f} (steps={self.model.num_timesteps} elapsed={elapsed:.1f}s device={DEVICE})"
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


def train(
    config_path: str,
    out: str,
    *,
    resume: bool = False,
    wall_hours: float | None = None,
    total_steps: int | None = None,
) -> Path:
    cfg = PPOConfig.from_yaml(config_path)
    out_dir = Path(out)
    out_dir.mkdir(parents=True, exist_ok=True)

    total = total_steps if total_steps is not None else cfg.total_steps
    wall = wall_hours if wall_hours is not None else float(cfg.extra.get("wall_hours", 5.0))
    eval_episodes = int(cfg.extra.get("eval_episodes", 20))
    curriculum = cfg.extra.get("curriculum", {})
    start_dial = float(curriculum.get("start_dial", 0.3))
    end_dial = float(curriculum.get("end_dial", 0.5))
    overrides = {k: float(v) for k, v in dict(cfg.extra.get("difficulty", {}) or {}).items()}
    if overrides:
        logger.info("difficulty overrides: %s", overrides)
    switch_fraction = float(curriculum.get("switch_fraction", 0.4))

    env = build_vec_env(cfg, cfg.seed, start_dial, overrides)

    if resume:
        from stable_baselines3 import PPO

        ckpt = _newest_checkpoint(out_dir)
        model = PPO.load(str(ckpt), env=env, device=DEVICE)
        logger.info("resumed from %s at step %d", ckpt, model.num_timesteps)
        remaining = max(total - model.num_timesteps, 0)
    else:
        model = make_ppo(env, cfg.seed, cfg)
        remaining = total

    logger.info("torch device=%s (must be cpu by design, plan.md §8)", DEVICE)
    assert DEVICE == "cpu", f"state-PPO must train on cpu, got {DEVICE}"

    callbacks = [
        ThroughputLogCallback(),
        CheckpointCallback(out_dir, cfg.checkpoint_every),
        WallClockStopCallback(wall),
        CurriculumCallback(int(switch_fraction * total), end_dial, overrides=overrides),
        RetentionEvalCallback(out_dir, cfg.eval_every, eval_episodes),
    ]

    if remaining > 0:
        model.learn(total_timesteps=remaining, reset_num_timesteps=not resume, callback=callbacks)
    else:
        logger.info("resume: already at/past total_steps=%d, nothing to train", total)

    final_path = out_dir / f"ckpt_{model.num_timesteps}.zip"
    model.save(str(final_path))
    logger.info("final checkpoint: %s", final_path)

    if not (out_dir / "best.zip").exists():
        # No eval callback fired (run shorter than eval_every) — fall back to the final model so
        # PPOHunter always has something to load (train smoke uses total_steps < eval_every*... ).
        shutil.copy(final_path, out_dir / "best.zip")
        logger.info("best.zip absent after run; copied final checkpoint as fallback")

    env.close()
    return out_dir


def main(argv: list[str] | None = None) -> int:
    logging.basicConfig(level=logging.INFO)
    parser = argparse.ArgumentParser(description="lockon.harness PPO training")
    parser.add_argument("--config", required=True)
    parser.add_argument("--out", required=True)
    parser.add_argument("--resume", action="store_true")
    parser.add_argument("--wall-hours", type=float, default=None)
    parser.add_argument("--total-steps", type=int, default=None)
    args = parser.parse_args(argv)

    train(
        args.config,
        args.out,
        resume=args.resume,
        wall_hours=args.wall_hours,
        total_steps=args.total_steps,
    )
    return 0


if __name__ == "__main__":
    sys.exit(main())
