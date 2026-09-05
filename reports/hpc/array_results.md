# PERUN CPU array — results

45 of 45 units reported. Selection rule: highest policy retention on the unit's own 20 selection episodes (seeds 0–19, mid difficulty); the fork verdict uses seeds 1000–1079.

**Best unit: `k_short_d0.3_s3`** — policy 55.5 % vs static 37.6 % vs scripted 54.3 % (selection seeds; std 36.8)

## Ablation — policy retention % (mean ± std over seeds)

| reward variant | arena density | n | mean | std |
|---|---|---|---|---|
| base | 0.3 | 5 | 52.2 | 1.4 |
| base | 0.5 | 5 | 51.1 | 2.8 |
| base | 0.7 | 5 | 50.3 | 3.2 |
| k_short | 0.3 | 5 | 51.9 | 3.7 |
| k_short | 0.5 | 5 | 51.2 | 3.3 |
| k_short | 0.7 | 5 | 52.5 | 2.6 |
| lam_hi | 0.3 | 5 | 50.5 | 2.0 |
| lam_hi | 0.5 | 5 | 50.9 | 3.4 |
| lam_hi | 0.7 | 5 | 52.0 | 1.9 |

## All units (sorted by policy retention on selection seeds)

| unit | policy | static | scripted | beats static | beats scripted | train h |
|---|---|---|---|---|---|---|
| k_short_d0.3_s3 | 55.5 | 37.6 | 54.3 | True | True | 1.63 |
| k_short_d0.7_s2 | 55.4 | 37.6 | 54.3 | True | True | 1.78 |
| lam_hi_d0.7_s3 | 55.2 | 37.6 | 54.3 | True | True | 1.70 |
| k_short_d0.7_s4 | 55.2 | 37.6 | 54.3 | True | True | 1.78 |
| base_d0.5_s4 | 54.9 | 37.6 | 54.3 | True | True | 1.52 |
| k_short_d0.3_s1 | 54.8 | 37.6 | 54.3 | True | True | 1.63 |
| base_d0.7_s3 | 54.8 | 37.6 | 54.3 | True | True | 1.66 |
| k_short_d0.5_s1 | 54.8 | 37.6 | 54.3 | True | True | 1.58 |
| lam_hi_d0.5_s3 | 54.6 | 37.6 | 54.3 | True | True | 1.60 |
| k_short_d0.5_s0 | 54.5 | 37.6 | 54.3 | True | True | 1.66 |
| base_d0.3_s3 | 54.2 | 37.6 | 54.3 | True | False | 1.66 |
| lam_hi_d0.3_s3 | 53.8 | 37.6 | 54.3 | True | False | 1.65 |
| base_d0.3_s0 | 53.3 | 37.6 | 54.3 | True | False | 1.65 |
| base_d0.5_s0 | 53.0 | 37.6 | 54.3 | True | False | 1.55 |
| lam_hi_d0.5_s4 | 52.6 | 37.6 | 54.3 | True | False | 1.61 |
| k_short_d0.3_s4 | 52.4 | 37.6 | 54.3 | True | False | 1.69 |
| base_d0.7_s4 | 52.0 | 37.6 | 54.3 | True | False | 1.68 |
| lam_hi_d0.5_s1 | 51.8 | 37.6 | 54.3 | True | False | 1.60 |
| lam_hi_d0.7_s1 | 51.6 | 37.6 | 54.3 | True | False | 1.73 |
| base_d0.3_s4 | 51.6 | 37.6 | 54.3 | True | False | 1.62 |
| lam_hi_d0.7_s2 | 51.4 | 37.6 | 54.3 | True | False | 1.71 |
| lam_hi_d0.7_s4 | 51.4 | 37.6 | 54.3 | True | False | 1.71 |
| base_d0.3_s2 | 51.1 | 37.6 | 54.3 | True | False | 1.69 |
| lam_hi_d0.3_s4 | 51.0 | 37.6 | 54.3 | True | False | 1.79 |
| base_d0.3_s1 | 51.0 | 37.6 | 54.3 | True | False | 1.70 |
| k_short_d0.7_s1 | 50.9 | 37.6 | 54.3 | True | False | 1.68 |
| k_short_d0.7_s0 | 50.9 | 37.6 | 54.3 | True | False | 1.75 |
| k_short_d0.5_s3 | 50.7 | 37.6 | 54.3 | True | False | 1.61 |
| k_short_d0.3_s0 | 50.5 | 37.6 | 54.3 | True | False | 1.68 |
| lam_hi_d0.7_s0 | 50.3 | 37.6 | 54.3 | True | False | 1.76 |
| base_d0.7_s0 | 50.1 | 37.6 | 54.3 | True | False | 1.62 |
| k_short_d0.7_s3 | 50.0 | 37.6 | 54.3 | True | False | 1.11 |
| lam_hi_d0.5_s0 | 49.8 | 37.6 | 54.3 | True | False | 1.61 |
| base_d0.5_s1 | 49.7 | 37.6 | 54.3 | True | False | 1.57 |
| lam_hi_d0.3_s1 | 49.6 | 37.6 | 54.3 | True | False | 1.61 |
| lam_hi_d0.3_s0 | 49.4 | 37.6 | 54.3 | True | False | 1.60 |
| base_d0.5_s3 | 49.2 | 37.6 | 54.3 | True | False | 1.47 |
| k_short_d0.5_s4 | 48.9 | 37.6 | 54.3 | True | False | 1.61 |
| lam_hi_d0.3_s2 | 48.8 | 37.6 | 54.3 | True | False | 1.62 |
| base_d0.5_s2 | 48.4 | 37.6 | 54.3 | True | False | 1.56 |
| base_d0.7_s1 | 47.7 | 37.6 | 54.3 | True | False | 1.58 |
| k_short_d0.5_s2 | 47.3 | 37.6 | 54.3 | True | False | 1.59 |
| base_d0.7_s2 | 47.1 | 37.6 | 54.3 | True | False | 1.62 |
| k_short_d0.3_s2 | 46.3 | 37.6 | 54.3 | True | False | 1.69 |
| lam_hi_d0.5_s2 | 45.6 | 37.6 | 54.3 | True | False | 1.60 |

Units beating static on selection seeds: 45/45; beating scripted: 10/45.
