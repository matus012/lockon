# lockon.harness

The composition root (plan.md §9 D1): the only package allowed to import env, sensor, track and
policy together, and the owner of every CLI (bench, render, train, train_prey, eval, sweep) plus
the two Gymnasium wrappers (hunter-training `LockonGym`, prey-training `PreyGym`). Runs one episode
end to end (`episode.py`), attributes lock-loss causes, renders every visual the owner sees,
trains PPO on state (CPU by design), evaluates policies on lock retention %, and sweeps the 5
difficulty axes into curves. Perception is frozen for every number the owner reads: `EVAL_NOISE =
NoiseConfig.from_dial(0.3, seed=0)` + `LockTracker()` defaults, one constant, one place —
`episode.py`'s hunter-driving helpers are that one place, shared by `run_episode` and `PreyGym`.

Import rule (`tests/test_boundaries.py`): `harness` may import `core`, `env`, `sensor`, `track`,
`policy`, `harness` — i.e. anything.

## Public API
- `episode.EpisodeResult` (seed, difficulty, policy, states, actions, gt_boxes, tracks, lock,
  retention, loss_causes, frames); `run_episode(difficulty, seed, hunter, prey, *, steps=EPISODE_STEPS,
  scene=None, capture=False, noise=None) -> EpisodeResult`; `resolve_hunter(name_or_path) -> Hunter`;
  `EVAL_NOISE: NoiseConfig` (frozen eval-time dial).
- Shared hunter-driving helpers (`episode.py`, single source of truth for `run_episode` and
  `prey_gym.PreyGym`): `hunter_frame_stack`, `hunter_obs_mode`, `init_hunter_stack` (SB3 zero-fill),
  `hunter_action` — so training and eval never diverge.
- `prey_gym.PreyGym(difficulty, seed, hunter, hunter_frame_stack, reward: PreyRewardConfig)` —
  `gymnasium.Env`, obs `Box((PreyObsBuilder.size(),))`, action `Box((2,))`; prey is the RL agent,
  `hunter` is FROZEN and driven by the helpers above; pays on the tracker's lock, not raw
  visibility (row 20); `set_difficulty(...)` curriculum hook (unused by `configs/prey_gpu.yaml`).
- `scenes.SCENES` — `"occlusion"`/`"lights_cut"`/`"sensor3"`/`"chase"`; `select_occlusion_seed()`.
  `gym_env.LockonGym(difficulty, seed, prey, reward)` — hunter-training `gymnasium.Env`, own frozen
  `LockTracker`, rewards its lock (D13/D15).
- `eval.evaluate(policy_name_or_path, difficulty, seeds, prey="scripted") -> dict[str, Any]` —
  retention mean/std, ttr, loss-cause histogram; CLI: `--seed-base` (gates use 1000), `--prey
  scripted|<zip>` via `resolve_prey` (`"scripted"`→`ScriptedPrey()`, else `LearnedPrey(path)`).
- `train.train(...)`, `train.build_vec_env(...)`. `train_prey.train_prey(config_path, out,
  hunter_name, *, device=None, resume=False, wall_hours=None, total_steps=None, n_envs=None) ->
  Path` — mirrors `train.py` against a frozen `hunter`; `best.zip` = LOWEST hunter retention,
  sidecar `{"hunter", "obs": "prey_v1"}`; `--device cuda` exits 3 on a CPU fallback (STOP, §5).
- `sweep.run_sweep(...) -> list[dict]`, `write_curves(...)`, `write_failure_report(...)`. CLIs:
  `bench_env.py`, `render.py`, `train.py`, `train_prey.py`, `eval.py`, `sweep.py` (prey trial changes no gate).

## Deviations from SPEC
- `scenes.py`'s occlusion showcase seed: SPEC's fixed seed 11 gives a 30-step gap equal to the
  tracker's coast buffer, id dies mid-shot. Replaced with `select_occlusion_seed()`, first seed in
  11..60 passing `occlusion_showcase_ok`, written once, never hand-tuned (row 7). `PreyGym.reset()`
  primes its tracker with an extra `t=0` update before stepping at `t=1` (`run_episode` doesn't) —
  a one-step offset (~1/200), direction unmeasured (row 21).

## Invariants (tests/test_harness.py, tests/test_train_sweep.py, tests/test_prey.py)
- `run_episode`: consistent lengths, retention in [0,1], `gt_boxes[t] is None` iff not visible;
  `occlusion` occludes then recovers, `lights_cut` kills rgb but not thermal; `evaluate(...)`
  returns finite numbers; render smoke produces a real GIF.
- `LockonGym`/`PreyGym` both pass `check_env` and reseed layout + prey every episode; training
  smoke (hunter and prey) writes `best.zip` + log/sidecar, frame-stack order matches `VecFrameStack`
  for both; sweep smoke writes outputs and resumes (skips completed cells).

## Gates (plan.md §3)
```
uv run python -m lockon.harness.bench_env --steps 2000                                    # G1
uv run python -m lockon.harness.render --scene sensor3 --gif reports/gifs/sensor_3ch.gif   # G2
uv run python -m lockon.harness.eval --policy scripted --vs static --n 20                  # G4
uv run python -m lockon.harness.eval --policy runs/ppo_local/best.zip --vs static --n 20   # G5
uv run python -m lockon.harness.sweep --config configs/sweep_local.yaml                    # G6
```
