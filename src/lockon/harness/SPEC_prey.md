# Learned-prey trial — specification (lead-authored 2026-09-05; plan.md §5 step 7 GPU line, ≤20 GPU-h)

Question the trial answers: **does a learned evader beat the scripted prey as an examiner?**
Deliverable = one table: hunters {static, scripted, ppo-best} × prey {scripted, learned} →
lock retention % on held-out seeds, plus the learned prey's training curve. Nothing here
changes any gate; the scripted prey stays the examiner of record for G4–G6.

## Package placement (boundary rule unchanged)
- `lockon.policy.prey_learned`: `PreyObsBuilder`, `PreyRewardConfig`, `prey_reward`,
  `LearnedPrey(Prey)` — core + numpy + SB3 only.
- `lockon.harness.prey_gym`: `PreyGym(gymnasium.Env)` — composition root.
- `lockon.harness.train_prey`: CLI (mirrors `train.py`).
- `lockon.harness.eval`: gains `--prey scripted|<zip>` (default scripted) → `resolve_prey`.

## Prey observation (privileged — the prey is the examiner, project.md §3)
`PreyObsBuilder(layout).step(state) -> np.ndarray[float32]`, size 3+3+1+1+ 4·3 + 1 = 21:
own (x/hs, y/hs, yaw/π) · drone relative in the person's frame (dx, dy)/2hs and drone yaw
relative/π · `person_visible` (0/1) · illumination at person · the 4 nearest pillars: relative
(dx, dy)/2hs and half-width/hs · steps-visible-in-a-row / EPISODE_STEPS. Frame stack 4.

## Prey action + reward
Action Box(-1, 1, (2,)) → world-frame velocity × `person_max_speed(difficulty)` (env clamps).
`prey_reward = −1·[person_visible] − 0.01·‖a‖² + 0.5·[transition visible→hidden]`
(the evader is paid for being unseen, with a small bonus for breaking line of sight).
`PreyRewardConfig` frozen dataclass; YAML-loadable.

## PreyGym
`PreyGym(difficulty, seed, hunter: Hunter, hunter_frame_stack: int, reward)`: fresh layout +
episode seed on every reset (same rule as `LockonGym`); the hunter is FROZEN and runs inside
`step()` exactly as the harness runs it (ScriptedHunter on `ObsBuilder`, PPOHunter via its
stacked vector and `obs_seen` sidecar convention; the frozen tracker feeds the hunter's
"seen" when its `obs_seen == "lock"`). `set_difficulty` for a curriculum. State only.

## train_prey.py — `python -m lockon.harness.train_prey --config configs/prey_gpu.yaml --out runs/prey --hunter scripted|<zip> [--device cuda|cpu] [--wall-hours 18] [--resume]`
Same skeleton as `train.py` (SubprocVecEnv spawn, VecMonitor, VecFrameStack, checkpoints,
wall stop, `best.zip` by the eval below, `train_log.jsonl`). **Environment probe (§5): on
`--device cuda` assert `torch.cuda.is_available()` and log the device before learning; a CPU
fallback on the GPU job is a STOP (exit 3), not a grind.** n_envs 32 on a GPU node.
Eval every `eval_every`: retention of the frozen hunter against the current prey on 20 seeds
(lower = better prey); `best.zip` = lowest hunter retention. Sidecar `best.zip.prey.json`
= {"hunter": <name or path>, "obs": "prey_v1"}.

## LearnedPrey
`LearnedPrey(path)`: `reset(layout, difficulty, seed)` loads the zip + sidecar;
`act(state, illumination)` builds the obs, stacks (SB3 zero-fill, newest last), predicts
deterministically, returns (vx, vy) in m/s. `name = "learned_prey"`.

## Tests (`tests/test_prey.py`, CPU, tiny)
1. `PreyGym` passes `gymnasium.utils.env_checker.check_env`; 10 random steps.
2. reward: visible step −1, hidden step 0, transition bonus +0.5 exactly once.
3. `train_prey` smoke: 2 envs, 2048 steps, device cpu → best.zip + sidecar; `LearnedPrey`
   loads it and `run_episode(..., prey=LearnedPrey(...))` runs 20 steps.
4. `eval --prey <zip>` resolves and runs (n=2).
5. Frame-stack ordering for the prey mirrors the hunter's (same assertion pattern as
   `test_frame_stack_ordering_matches_vecframestack`).

## Budget line (plan §7, owner's cap)
One GPU job, `gpu_short`, 1 GPU, 32 CPUs, `timeout` 18 h hard + 1 h eval/render margin
≤ 19 GPU-h; the launcher's budget table must show GPU-h total ≤ 22 and ≤ 70 with everything.
