# pilots — diagnostic scripts behind deviation-log rows 6, 8, 9

Kept verbatim for provenance (run from the repo root with `uv run python reports/pilots/<name>`).
None of them touched the held-out gate seeds 1000–1019.

| script | rows | seeds used | what it measured |
|---|---|---|---|
| diag_dx.py, diag_est.py | 6 | noise seeds 0–19 (synthetic scene) | id-hold rate vs box speed and Kalman state model: flat |
| diag_rl.py | 8 | episode seeds 100–107 | reward / visibility / retention per policy at dials 0.3 and 0.5 |
| diag_paths.py | 8, 9 | 500–505 | same checkpoint through the SB3 stack vs the harness path |
| diag_ppo_stats.py | 9 | SB3 seed 0, 8 envs | SB3 statistics of a 65k-step run (lock reward, action-only control) |
| diag_std.py | 9 | 0–11 | noisy-static retention / visibility / out-of-FOV vs exploration std |
| diag_yawcomp.py | 9 | 0–9 | yaw-compensated detections: flat (0.222 vs 0.223), rejected |
