# Learned-prey trial (PERUN, one H200)

Lock retention % of each hunter against each prey; **lower is a harder prey**. Held-out seeds, mid difficulty, per-episode tracker noise.

| hunter | scripted prey | learned prey | Δ (learned − scripted) |
|---|---|---|---|
| static camera | 44.0 ± 32.1 | 27.6 ± 28.1 | -16.4 [-25.0, -7.7] |
| scripted hunter | 52.0 ± 33.8 | 50.8 ± 39.0 | -1.2 [-11.8, +9.5] |
| PPO hunter | 51.3 ± 35.1 | 35.9 ± 31.3 | -15.4 [-25.1, -5.6] |

n = 80 episodes per cell. A negative Δ means the learned prey evades better than the scripted one against that hunter; a CI spanning zero means the trial could not separate them.
