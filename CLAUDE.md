# 110_lockon — project instructions

Active perception under degradation: a MuJoCo drone-camera (scripted or PPO) keeps lock on a
scripted evader under occlusion, darkness and sensor loss. Demo is the product.

## State files (read in this order)
- `status.txt` — owner-facing, project.md §9 format. Update at every gate and session end.
- `context.md` — position, decisions, architecture, cold-resume recipe. **Must always suffice to
  resume cold.**
- `plan.md` — gates (§3), task graph (§5), fork/kill (§6), ritual (§8), readings (§9). Locked.
- `project.md` — owner master context, all decisions LOCKED. Never edited by the CLI.

## Invariants
- Import rule (tests/test_boundaries.py): env/sensor/track/policy → core only; harness = the
  composition root; demo → harness + core; core → numpy only.
- Tracker consumes GT boxes + injected noise; RL trains on state; rendering = clips only.
- Every number the owner reads is lock retention % (steps with correct ID / episode steps).
- Gates run through `scripts/check_gates.py`; a gate never passes from memory.
- Anchored `.gitignore`; committed visuals are allowlisted per file in
  `tests/test_license_guard.py`. Apache/MIT/BSD deps only; `THIRD_PARTY.md` lists them.
- State-PPO trains on CPU by design; GPU is for rendering.
- Public wording: drone safety / search & rescue / filming / wildlife. Never weapons/military.

## Run and verify
```
uv run python scripts/check_gates.py --all      # every gate, verdicts → reports/gates.json
uv run pytest -q                                 # unit suite
uv run python -m lockon.demo.render_all          # three shots
```
