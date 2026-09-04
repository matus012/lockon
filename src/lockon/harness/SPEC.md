# lockon.harness — specification (lead-authored 2026-09-04; steps 3b/4/5/8; gates G1–G6)

The composition root (plan.md §9 D1): imports env, sensor, track, policy, core. Owns every CLI.
Eval-time perception is frozen: `EVAL_NOISE = NoiseConfig.from_dial(0.3, seed)` and
`LockTracker()` defaults, for every number the owner reads (project.md §7). Single constant.

## episode.py
```python
@dataclass
class EpisodeResult:
    seed: int; difficulty: Difficulty; policy: str
    states: list[WorldState]; actions: list[AgentAction]
    gt_boxes: list[FloatArray | None]        # state.person_box per step (any channel that sees; None otherwise)
    tracks: list[list[Track]]; lock: list[LockStatus]
    retention: RetentionResult                # core.metrics.retention(gt_boxes, tracks)
    loss_causes: list[str]                    # per loss event: "occlusion" | "darkness" | "dropout" | "fov"
    frames: list[SensorFrame] | None          # only when capture=True

Scene = Callable[[Env, int], None]           # scripted overrides applied before step t

def run_episode(difficulty, seed, hunter: Hunter, prey: Prey, *, steps=EPISODE_STEPS,
                scene: Scene | None = None, capture: bool = False,
                noise: NoiseConfig | None = None) -> EpisodeResult
```
Loop: env.reset → obs_builder.reset → for t: scene(env, t) → prey velocity → hunter action
(PPOHunter gets the frame-stacked vector via a small deque held here) → env.step → gt per
channel from state (Detection where channels_see) → NoiseInjector → LockTracker.update →
record. `gt_boxes[t] = state.person_box if state.person_visible else None`. Loss cause at a loss
event, from state at that step: not in FOV → "fov"; in FOV but occluded (no LOS) → "occlusion";
LOS but no channel sees because illumination fails → "darkness"; LOS, physics permit, but the
seeing channels are dead → "dropout".

## scenes.py — deterministic scripted scenes (fixed layouts via fixed seeds)
- `occlusion`: mid difficulty, seed 11; prey overridden to walk a straight line that passes
  behind a pillar relative to the drone's start (compute the pillar from the layout at reset);
  hunter = ScriptedHunter. Used by GIF1 and test_scenes.
- `lights_cut`: seed 12, darkness 0.0 → `env.set_darkness(1.0)` at t=60; prey wanders slowly;
  hunter static. rgb goes to noise; depth+thermal keep the lock. GIF2.
- `sensor3`: seed 13, 60 steps, prey wanders, hunter static, darkness 0.2. Sensor GIF (G2).
- `chase`: mid difficulty, seed 14, full 200 steps, ScriptedHunter vs ScriptedPrey. Chase clip.
Registry `SCENES: dict[str, SceneSpec]` (name → difficulty, seed, steps, hunter, prey/override,
scene callable).

## render.py — `python -m lockon.harness.render --scene S --gif P [--mp4 P] [--policy static|scripted|<zip>] [--fps 10]`
Runs the scene with `capture=True`, composes per step: `sensor.side_by_side(frame)` on top,
tracker boxes + ids drawn on every alive channel (numpy line drawing; PIL may be used for text),
banner line "t=… | lock: HELD/LOST (n steps) | retention so far xx %", plus a 160-px top-down
minimap (arena, pillars, drone, person, LOS line green/red). GIF via imageio (palette), mp4 via
`av` (h264, yuv420p, crf 23). Every visual the owner sees is produced here — one place.

## gym_env.py — `LockonGym(gymnasium.Env)`
`__init__(difficulty, seed, prey: Prey, reward: RewardConfig)`; observation Box(-inf, inf,
(AgentObs.size(),)); action Box(-1, 1, (3,)). `reset(seed)` reseeds env/prey; `step`: prey →
env.step → obs → reward (policy.features.reward) → terminated at EPISODE_STEPS; `info` carries
`visible`, `steps_since_seen`. `set_difficulty(d)` for curriculum via `VecEnv.env_method`.
State only — never touches sensor/render (project.md §2).

## train.py — `python -m lockon.harness.train --config configs/ppo_local.yaml --out runs/ppo_local [--resume] [--wall-hours 5]`
SubprocVecEnv(n_envs) → VecMonitor → VecFrameStack(frame_stack); `make_ppo`; callbacks:
checkpoint every `checkpoint_every`, wall-clock stop, curriculum (dial 0.3 → 0.5 at 40 % of
`total_steps`), and a retention-eval callback every `eval_every`: runs `evaluate()` (below) with
n=20 for the current policy AND static (cached once), logs both, saves `best.zip` by retention.
`--resume` loads the newest checkpoint and continues with `reset_num_timesteps=False`.
Environment probe: log torch device (must be cpu — plan §8) and steps/s after the first rollout.
Writes `runs/<name>/train_log.jsonl` (step, mean reward, eval retention, wall s).

## eval.py — `python -m lockon.harness.eval --policy P --vs Q --n 20 [--difficulty mid] [--json reports/eval_P_vs_Q.json]`
`evaluate(policy_name_or_path, difficulty, seeds) -> dict` with retention mean/std, ttr mean,
per-episode list, loss-cause histogram. Same seeds for both sides. Prints a 3-row table
(static / scripted / policy where applicable); exit 0 iff mean(P) > mean(Q) (plan §3 "beats").
Policies resolved by name: `static`, `scripted`, or a path → PPOHunter.

## sweep.py — `python -m lockon.harness.sweep --config configs/sweep_local.yaml`
Config: axes (occluder_density, darkness, prey_speed, prey_aggressiveness, channel_dropout) ×
values [0, 0.25, 0.5, 0.75, 1] with the others at 0.5 (plan §9 D5) × policies [static, scripted,
<ppo path if exists>] × n_episodes 20 × seeds fixed. Output: `reports/sweep/results.json`,
`reports/curves/<axis>.png` (retention mean ± std vs axis, one line per policy, matplotlib,
retention in %), `reports/curves/summary.png` (5 panels), `reports/sweep/failure_report.md`
(per axis: worst point per policy, dominant loss cause, mean time-to-reacquire). Resume-safe:
skip (axis, value, policy) cells whose result rows exist. Runs on CPU; ~1500 episodes.

## Tests (`tests/test_harness.py`)
1. `run_episode` mid difficulty, static hunter, 50 steps: lengths consistent, retention in [0,1],
   `gt_boxes[t] is None` iff not visible.
2. `occlusion` scene: at some step in the middle `person_visible` is False and later True again
   (the scene actually occludes) — instrument proof for GIF1.
3. `lights_cut` scene: after t=60 `channels_see["rgb"]` is False on every step while thermal
   sees at ≥ 80 % of steps — proof for GIF2. (Channel names allowed in tests only.)
4. `LockonGym` passes `gymnasium.utils.env_checker.check_env`; 20 random steps run.
5. `evaluate("static", mid, seeds=range(3))` returns finite numbers; `scripted` runs too.
6. render smoke (`@pytest.mark.render`): `sensor3` scene 10 steps → GIF file exists, > 1 KB.
