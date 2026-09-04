# lockon.policy — specification (lead-authored 2026-09-04; plan.md steps 4–5, gates G4/G5)

Import rule: `lockon.core`, `numpy`, and (for `ppo.py` only) `stable_baselines3` / `torch` /
`gymnasium.spaces`. Never `lockon.env`: every policy is a pure function of core types. The
Gymnasium wrapper, training and evaluation CLIs live in `lockon.harness` (plan.md §9 D1).

## Interfaces (`base.py`)
```python
class Hunter(Protocol):
    name: str
    def reset(self, layout: ArenaLayout, seed: int) -> None
    def act(self, obs: AgentObs) -> AgentAction          # partial observation only (project.md §3)

class Prey(Protocol):
    name: str
    def reset(self, layout: ArenaLayout, difficulty: Difficulty, seed: int) -> None
    def act(self, state: WorldState, illumination: Callable[[ArrayLike], float]) -> tuple[float, float]
        # world-frame velocity (vx, vy) in m/s; env clamps to person_max_speed(). Prey is the
        # examiner, so it reads full state (privileged) — say so in the docstring.
```

## Observation + reward (`features.py`) — single source, used by harness for both RL and eval
```python
class ObsBuilder:
    def __init__(self, layout: ArenaLayout) -> None
    def reset(self, state: WorldState) -> AgentObs      # episode starts visible: last_seen = current
    def step(self, state: WorldState) -> AgentObs
        # own = (x/hs, y/hs, yaw/π); last_seen_rel = (target − drone) at last sighting in the DRONE
        # body frame at the current step (so the policy sees "where I last saw it relative to me
        # now"), / (2·hs); time_since_seen = min(steps, EPISODE_STEPS)/EPISODE_STEPS;
        # raycasts = state.raycasts; light = illumination at last-seen position (state value at
        # sighting time, held); channels_alive from state.
@dataclass(frozen=True)
class RewardConfig: visible: float = 1.0; action_l2: float = 0.01; lost_penalty: float = 10.0; lost_after: int = 20
def reward(state: WorldState, action: AgentAction, steps_since_seen: int, cfg: RewardConfig) -> float
    # +visible·[person_visible] − action_l2·‖a‖² − lost_penalty exactly once when steps_since_seen
    # == lost_after (project.md §3: "big on lock lost > K steps"; K = 20 = 2 s).
```

## Hunters (`hunters.py`)
- `StaticCamera`: never moves (action zero). The floor (project.md §3, curves baseline 1).
- `ScriptedHunter` (visibility-greedy; baseline 2 and the fallback hero). From obs only:
  - target estimate = own position + rotate(last_seen_rel·2·hs, yaw).
  - if time_since_seen == 0: hold a standoff — move so range → R* = 6 m and bearing → 0 (turn
    toward target with yaw_rate ∝ bearing, strafe to keep R*).
  - else (lost): move toward the last-seen point at full speed while turning toward it; if the
    raycast in the direction of travel is < 0.08 (normalised), sidestep along the freer of the
    two adjacent rays (this is what makes it "greedy on visibility": go where the occluder is
    not). After 30 lost steps, orbit the last-seen point at radius 4 m.
  - `ScriptedHunter(standoff_m=6.0, gain_yaw=2.0, gain_v=1.0)`; no randomness.
- `PPOHunter(path: str)`: loads an SB3 PPO zip (`ppo.py`), `act` = deterministic predict on
  `obs.vector()` with frame-stack handled by the harness wrapper (this class receives the
  stacked vector via `act_vector(np.ndarray)`; `act(obs)` raises NotImplementedError — document).

## Prey (`prey.py`) — scripted examiner, difficulty-dialled
`ScriptedPrey` with four behaviours mixed by `Difficulty`:
- **cover-seek** (aggressiveness): pick the pillar minimising `‖p_pillar − person‖ +
  2·visibility_penalty`, target the point on the far side of that pillar from the drone (pillar
  centre + (pillar − drone) unit · (half_width + 0.6)); re-plan every 10 steps or when the target
  is reached.
- **dark-seek** (aggressiveness · darkness): among 12 sampled points within 6 m, prefer the
  lowest `illumination`; blended 50/50 with cover-seek when darkness > 0.3.
- **LOS-break** (aggressiveness): when `state.person_visible`, add a lateral component
  perpendicular to the drone→person line, sign chosen toward the nearest pillar.
- **speed**: commanded speed = person_max_speed() (env clamps); at aggressiveness 0 the prey
  wanders (random waypoints, speed 0.5·max).
Mix: aggressiveness a ∈ [0,1] = P(evasive behaviour active this re-plan); otherwise wander.
Deterministic under seed. Wall/pillar avoidance is env's job (push-out); prey only steers.

## PPO config (`ppo.py`)
`make_ppo(env, seed, cfg: PPOConfig) -> stable_baselines3.PPO` with MlpPolicy [128, 128],
n_steps 1024, batch 256, n_epochs 10, γ 0.99, gae λ 0.95, lr 3e-4, ent 0.01, clip 0.2,
device "cpu" (plan §8: state-PPO is CPU by design — assert device is cpu and log it).
`PPOConfig` is a frozen dataclass with these fields + `total_steps`, `frame_stack=4`,
`n_envs=8`, `checkpoint_every=100_000`, `eval_every=200_000`. YAML-loadable by harness.

## Tests (`tests/test_policy.py`) — no MuJoCo; hand-built WorldState/ArenaLayout
1. ObsBuilder: visible step → time_since_seen 0 and last_seen_rel points at the person; after
   3 invisible steps → 3/EPISODE_STEPS and last_seen_rel rotates with the drone's yaw change.
2. reward: visible → +1 − 0.01‖a‖²; the −10 fires once at steps_since_seen == 20, not at 21.
3. StaticCamera.act is zero; ScriptedHunter turns toward a target on its left (yaw_rate > 0)
   and moves forward when range > standoff, backward when range < standoff.
4. ScriptedHunter lost mode: with a blocked forward ray, the lateral action is nonzero.
5. ScriptedPrey: with aggressiveness 1 and a drone in view, the commanded velocity points
   toward a pillar's far side (dot product with pillar direction > 0); with aggressiveness 0
   it never exceeds 0.5·max speed.
6. `make_ppo` on a dummy gymnasium Box env builds, device is cpu, `predict` returns shape (3,).
`uv run mypy --strict src/lockon/policy` clean (SB3 lacks full stubs → narrow ignores).
