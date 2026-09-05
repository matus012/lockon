# lockon.policy

Owns every agent: the scripted evader (`ScriptedPrey` — cover-seek, dark-seek, LOS-break, speed,
dial-mixed) and its learned counterpart (`prey_learned.py`), the hunter baselines (`StaticCamera`
floor, `ScriptedHunter` visibility-greedy fallback hero, `PPOHunter` loading a Stable-Baselines3
policy), the shared observation builders + rewards, and the PPO config/builder. Pure functions of
`core` types — no MuJoCo, no privileged state for hunters (project.md §3: partial `AgentObs` only;
the prey is the privileged examiner, reading full `WorldState`). Framing: drone-safety /
search-and-rescue pursuit of a person actively evading detection (project.md §1).

Import rule (`tests/test_boundaries.py`): `policy` may import `lockon.core` + third-party only
(`numpy`, plus `stable_baselines3`/`torch`/`gymnasium` in `ppo.py`); no `lockon.env`.

## Public API
- `Hunter` / `Prey` protocols (`base.py`) — `reset(...)`, `act(...)`.
- `ObsBuilder(layout)` — `.reset(state) -> AgentObs`; `.step(state, seen: bool | None = None) -> AgentObs`
  (`seen` defaults to state visibility; the harness passes the frozen tracker's lock instead, D15,
  so the policy observes the same quantity the reward pays for); `.steps_since_seen` (property).
- `RewardConfig(visible=1.0, action_l2=0.01, lost_penalty=10.0, lost_after=20)`.
- `reward(state, action, steps_since_seen, cfg, visible: bool | None = None) -> float`.
- `StaticCamera()`, `ScriptedHunter(standoff_m=6.0, gain_yaw=2.0, gain_v=1.0)`, `PPOHunter(path)` — `.act_vector(vector)` (frame-stacked; `.act(obs)` raises `NotImplementedError`);
  `.n_stack` (property, frame-stack depth read back from the loaded model, never a default);
  `.obs_seen` ("lock"/"state", read from the sidecar `<path>.obs.json` when present — which
  observation convention the checkpoint trained with).
- `ScriptedPrey(los_break: bool = True)` — `.reset(layout, difficulty, seed)`, `.act(state, illumination) -> tuple[float, float]`;
  `los_break=False` disables the LOS-break lateral term (measurement arm only, not the default).
- `PPOConfig(...)` (SB3 hyperparams + scheduling knobs, `.from_yaml(path)`); `make_ppo(env, seed, cfg) -> PPO` (device pinned to `"cpu"`, asserted).
- `prey_learned.py` (learned-prey trial, SPEC_prey.md; privileged, reads `WorldState` not `AgentObs`):
  `PreyObsBuilder(layout)` — `.reset(state)`/`.step(state) -> ndarray`, `.size() -> int` (21);
  `PreyRewardConfig(visible_penalty=1.0, action_l2=0.01, transition_bonus=0.5)`;
  `prey_reward(state, action, was_visible, cfg, seen: bool | None = None) -> float` — `seen`
  defaults to `state.person_visible`, but the training wrapper passes the tracker's lock (row 20);
  `LearnedPrey(path)` — `.reset(layout, difficulty, seed)`, `.act(state, illumination) -> tuple[float, float]`,
  `.name = "learned_prey"` (loads the SB3 zip + `<path>.prey.json` sidecar).
- Published checkpoints: `models/README.md` (hunter/prey zips + `.obs.json`/`.prey.json` sidecars).

## Deviations from SPEC
- `reward()` gained a `visible: bool | None = None` override, absent from the SPEC signature: a
  hunter rewarded on raw visibility learned to strafe/yaw hard enough to make ByteTrack switch ids
  5.6x/episode — reward rose while retention fell. Lock retention % is the reward now (D13, row 8).
  Symmetrically, `prey_reward`'s published run paid on geometric visibility; corrected to pay on
  the frozen tracker's lock so the evader optimises the reported metric (row 20).
- `ScriptedPrey._cover_target`'s `visibility_penalty` term was left undefined by the SPEC; implemented
  as 0 when the candidate pillar actually breaks LOS, 1 otherwise (docstring, no row — a reading).
  `PreyGym` primes its tracker with an extra `t=0` update before stepping, a one-step offset (row 21).

## Invariants (tests/test_policy.py, tests/test_prey.py)
- Obs builder tracks last sighting, rotates with yaw; reward is `+1 − 0.01‖a‖²` visible, −10 fires
  once at `steps_since_seen == 20`; `visible` override switches source.
- `StaticCamera` never moves; `ScriptedHunter` turns/moves toward standoff range, sidesteps a
  blocked forward raycast when lost. Aggressive prey seeks cover; passive prey stays ≤ 0.5·max speed.
- `make_ppo` builds on cpu, `.predict` shape (3,); `PPOConfig.from_yaml` tolerates unknown keys.
  `prey_reward`: −1 visible / 0 hidden / +0.5 exactly once on visible→hidden; `PreyGym` passes
  `check_env`; `train_prey` smoke writes `best.zip` + sidecar and `LearnedPrey` loads it.

## Gate (plan.md §3, G4/G5)
```
uv run python -m lockon.harness.eval --policy scripted --vs static --n 20      # G4
uv run python -m lockon.harness.eval --policy runs/ppo_local/best.zip --vs static --n 20   # G5
```
