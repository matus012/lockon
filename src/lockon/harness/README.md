# lockon.harness

The composition root (plan.md §9 D1): the only package allowed to import env, sensor, track and
policy together, and the owner of every CLI (bench, render, train, eval, sweep) plus the
Gymnasium wrapper. Runs one episode end to end (`episode.py`), attributes lock-loss causes,
renders every visual the owner sees, trains PPO on state (CPU by design), evaluates policies
head to head on lock retention %, and sweeps the 5 difficulty axes into curves. Perception is
frozen for every number the owner reads: `EVAL_NOISE = NoiseConfig.from_dial(0.3, seed=0)` +
`LockTracker()` defaults, one constant, one place.

Import rule (`tests/test_boundaries.py`): `harness` may import `core`, `env`, `sensor`, `track`,
`policy`, `harness` — i.e. anything.

## Public API
- `episode.EpisodeResult` (seed, difficulty, policy, states, actions, gt_boxes, tracks, lock, retention, loss_causes, frames); `episode.Scene = Callable[[Env, int], None]`.
- `episode.run_episode(difficulty, seed, hunter, prey, *, steps=EPISODE_STEPS, scene=None, capture=False, noise=None) -> EpisodeResult`.
- `episode.resolve_hunter(name_or_path) -> Hunter` — `"static"`/`"scripted"`/path→`PPOHunter`.
- `episode.EVAL_NOISE: NoiseConfig` — the frozen eval-time noise constant.
- `scenes.SCENES: dict[str, SceneSpec]` — `"occlusion"`, `"lights_cut"`, `"sensor3"`, `"chase"` (built lazily; `occlusion` runs a seed search).
- `scenes.select_occlusion_seed() -> int`; `scenes.occlusion_showcase_ok(states, retained) -> bool`.
- `gym_env.LockonGym(difficulty, seed, prey, reward)` — `gymnasium.Env`, obs `Box((AgentObs.size(),))`, action `Box((3,))`.
- `eval.evaluate(policy_name_or_path, difficulty, seeds) -> dict[str, Any]` — retention mean/std, ttr, loss-cause histogram.
- `train.train(...)`, `train.build_vec_env(cfg, seed, start_dial, overrides=None) -> VecFrameStack`.
- `sweep.run_sweep(...)`, `sweep.write_curves(rows, cfg)`, `sweep.write_failure_report(rows, cfg)`.
- CLIs: `bench_env.py`, `render.py`, `train.py`, `eval.py`, `sweep.py` (each `main(argv=None) -> int`, see gates below).

## Deviations from SPEC
- `scenes.py`'s occlusion showcase seed: SPEC's fixed seed 11 produces a 30-step gap equal to the
  tracker's coast buffer, so the id dies mid-shot. Replaced with `select_occlusion_seed()` — first
  seed in 11..60 passing `occlusion_showcase_ok` (gap ≥ 5 steps, post-gap retention ≥ 0.8, same
  id); the rule is written once and never hand-tuned — deviation-log row 7.

## Invariants (tests/test_harness.py, tests/test_train_sweep.py)
- `run_episode` produces consistent lengths, retention in [0,1], `gt_boxes[t] is None` iff not visible: `test_run_episode_basic_consistency`.
- `occlusion` scene genuinely occludes then recovers: `test_occlusion_scene_actually_occludes`.
- `lights_cut` scene kills rgb, thermal proxy keeps seeing: `test_lights_cut_scene_kills_rgb_keeps_thermal`.
- `LockonGym` passes `gymnasium`'s `check_env` plus 20 random steps: `test_lockon_gym_check_env_and_random_steps`.
- Layout + prey reseed every episode (SB3 calls `reset(seed=None)`): `test_gym_resamples_layout_every_episode`.
- `evaluate("static"/"scripted", ...)` returns finite numbers: `test_evaluate_returns_finite_numbers`.
- Render smoke produces a real GIF file: `test_render_smoke_produces_a_gif`.
- Training smoke writes `best.zip` + a log: `test_train_smoke_writes_best_and_log`.
- Frame-stack order matches SB3's `VecFrameStack`: `test_frame_stack_ordering_matches_vecframestack`.
- Sweep smoke writes outputs and resumes (skips completed cells): `test_sweep_smoke_writes_outputs_and_resumes`.

## Gates (plan.md §3)
```
uv run python -m lockon.harness.bench_env --steps 2000                                    # G1
uv run python -m lockon.harness.render --scene sensor3 --gif reports/gifs/sensor_3ch.gif   # G2
uv run python -m lockon.harness.eval --policy scripted --vs static --n 20                  # G4
uv run python -m lockon.harness.eval --policy runs/ppo_local/best.zip --vs static --n 20   # G5
uv run python -m lockon.harness.sweep --config configs/sweep_local.yaml                    # G6
```
