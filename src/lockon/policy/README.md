# lockon.policy

Owns every agent: the scripted evader (`ScriptedPrey` — cover-seek, dark-seek, LOS-break, speed,
dial-mixed), the hunter baselines (`StaticCamera` floor, `ScriptedHunter` visibility-greedy
fallback hero, `PPOHunter` loading a Stable-Baselines3 policy), the shared observation builder +
reward, and the PPO config/builder. Pure functions of `core` types — no MuJoCo, no privileged
state for hunters (project.md §3: hunters act from a partial `AgentObs` only; the prey is the
privileged examiner). Framing: drone-safety / search-and-rescue pursuit of a person actively
evading detection (project.md §1).

Import rule (`tests/test_boundaries.py`): `policy` may import `lockon.core` + third-party only
(`numpy`, plus `stable_baselines3`/`torch`/`gymnasium` in `ppo.py`); no `lockon.env`.

## Public API
- `Hunter` / `Prey` protocols (`base.py`) — `reset(...)`, `act(...)`.
- `ObsBuilder(layout)` — `.reset(state) -> AgentObs`; `.step(state) -> AgentObs`; `.steps_since_seen` (property).
- `RewardConfig(visible=1.0, action_l2=0.01, lost_penalty=10.0, lost_after=20)`.
- `reward(state, action, steps_since_seen, cfg, visible: bool | None = None) -> float`.
- `StaticCamera()`, `ScriptedHunter(standoff_m=6.0, gain_yaw=2.0, gain_v=1.0)`, `PPOHunter(path)` — `.act_vector(vector)` (frame-stacked; `.act(obs)` raises `NotImplementedError`).
- `ScriptedPrey()` — `.reset(layout, difficulty, seed)`, `.act(state, illumination) -> tuple[float, float]`.
- `PPOConfig(...)` (SB3 hyperparams + scheduling knobs, `.from_yaml(path)`); `make_ppo(env, seed, cfg) -> PPO` (device pinned to `"cpu"`, asserted).

## Deviations from SPEC
- `reward()` gained a `visible: bool | None = None` override, absent from the SPEC signature.
  Training passes the frozen tracker's lock status instead of raw state visibility: a policy
  rewarded on visibility learned to strafe/yaw hard enough that the person's box jumped ~8.5
  px/step and ByteTrack switched ids 5.6x/episode — retention fell while reward rose. Lock
  retention % is the reward now (context.md D13, deviation-log row 8).
- `ScriptedPrey._cover_target`'s `visibility_penalty` term was left undefined by the SPEC
  ("undefined" placeholder in the scoring formula); implemented as 0 when the candidate pillar
  actually sits between person and drone (would break LOS) and 1 otherwise — documented as an
  assumption in the module docstring, no deviation-log row (a reading, not a changed number).

## Invariants (tests/test_policy.py)
- Obs builder tracks last sighting, rotates with yaw: `test_obs_builder_visible_then_lost_rotates_with_yaw`.
- Reward is `+1 − 0.01‖a‖²` visible, −10 fires once exactly at `steps_since_seen == 20`: `test_reward_visible_and_lost_penalty_fires_once_at_threshold`.
- `StaticCamera` never moves: `test_static_camera_is_always_zero`.
- `ScriptedHunter` turns/moves toward standoff range: `test_scripted_hunter_standoff_turns_and_moves_toward_range`.
- Lost mode sidesteps a blocked forward raycast: `test_scripted_hunter_lost_sidesteps_blocked_forward_ray`.
- Aggressive prey heads toward a pillar's far side; passive prey stays ≤ 0.5·max speed: `test_scripted_prey_seeks_cover_when_aggressive`, `test_scripted_prey_wanders_below_half_speed_when_passive`.
- `make_ppo` builds on cpu, `.predict` returns shape (3,): `test_make_ppo_builds_on_cpu_and_predicts`.
- `PPOConfig.from_yaml` tolerates unknown keys via `.extra`: `test_ppo_config_from_yaml_tolerates_extra_keys`.
- Reward's `visible` override actually switches source: `test_reward_visible_override_uses_tracker_lock`.

## Gate (plan.md §3, G4/G5)
```
uv run python -m lockon.harness.eval --policy scripted --vs static --n 20      # G4
uv run python -m lockon.harness.eval --policy runs/ppo_local/best.zip --vs static --n 20   # G5
```
