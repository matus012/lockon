# HPC launch plan — step 6 approval request (plan.md §5 step 6, §7)

Account `perun26011488` · confirm allocation before the first `sbatch` · bundle pattern from ws/100.
Emitted by `uv run python scripts/hpc/sweep_launcher.py --mode slurm --config configs/hpc_sweep.yaml`
(placeholders `<FILL-PERUN-PARTITION>` / `<FILL-PERUN-ACCOUNT>` filled by hand on the cluster).

## What runs

| job | partition | shape | units | wall / unit | budget |
|---|---|---|---|---|---|
| smoke | cpu | 1 unit, 20k steps, GPU probe skipped | 1 | ≤ 10 min | ~0.2 CPU-h |
| CPU array `submit.sbatch` | cpu | seeds {0..4} × reward {base, lam_hi, k_short} × arena occluder_density {0.3, 0.5, 0.7}; 8 cpus, 16 GB, `%8`, `--dependency=afterok:<smoke>` | 45 | 1.0 h (measured 0.58 h on 8 laptop cores for 6M steps) | ≤ 45 CPU-h, **0 GPU-h** |
| learned-prey trial `submit_gpu_prey.sbatch` | gpu_short | 1 × H200, `timeout` 20 h hard; entrypoint `lockon.harness.train_prey` **not implemented yet** — submitted only if written before day 3 | 1 | ≤ 20 h | ≤ 20 GPU-h |
| renders (curves ± std + three shots, EGL) | gpu_short | 1 × H200 | 1 | ≤ 2 h | ≤ 2 GPU-h |

**GPU-h ask: ≤ 22 of the 70–100 cap** (CPU-h ≤ 46, not metered against the GPU cap).

## Decision rule at step 9 (pre-registered, deviation-log row 10)
Best array unit = highest retention vs static on the unit's own 20 eval episodes (seeds 0–19).
Fork verdict (PPO vs scripted) on **n ≥ 80 held-out episodes, seeds 1000–1079**, mid difficulty,
per-episode noise seeds: PPO beats scripted iff the paired mean difference > 0 (plan §3 wording);
the 95 % CI is reported beside it. At n = 20 the CI half-width is ±0.17; at n = 80 ≈ ±0.08.
If PPO does not beat scripted: hero stays scripted, PPO reported honestly, counter 2 of 3.

## Data gravity
`runs/`, checkpoints and videos stay on PERUN. Pulled home: `results/*.json`, `reports/curves/*.png`,
`reports/gifs/*.gif`, one `best.zip` (< 100 MB). Never rsync `runs/`.

## Owner action
One line: "approve HPC launch" (optionally with partition/account names). Everything else is scripted
in `HPC_RUNBOOK.md`.
