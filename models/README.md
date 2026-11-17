# models — published checkpoints

The two policies the README's tables are computed from, so the numbers can be reproduced.
Both were trained **entirely inside this repo's simulator**: no external data, no pretrained
weights. `tests/test_license_guard.py` allowlists them per file as class `sim-checkpoint`;
anything else with a blob extension still fails the guard.

| file | what it is | trained by |
|---|---|---|
| `ppo_hunter.zip` | the PPO camera policy in the headline table (local run 4; observes geometric visibility, hence the `.obs.json` sidecar) | `python -m lockon.harness.train --config configs/ppo_local.yaml` |
| `learned_prey_seed0.zip` | the learned evader of the prey trial, **attempt 1** (visibility reward, before deviation-log row 20 corrected the objective) | `python -m lockon.harness.train_prey --config configs/prey_gpu.yaml --hunter scripted` on one H200 |

Reproduce the tables:

```bash
uv run python -m lockon.harness.eval --policy models/ppo_hunter.zip --vs scripted --n 80 --seed-base 1000
uv run python -m lockon.harness.eval --policy scripted --vs static --n 80 --seed-base 1000 --prey models/learned_prey_seed0.zip
```

The sidecars are load-bearing: `.obs.json` records whether a hunter observes geometric visibility
or the tracker's lock, and `.prey.json` records which hunter a prey was trained against.
