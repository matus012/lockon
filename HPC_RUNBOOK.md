# PERUN runbook — login day (plan.md §7 step 7)

Everything below assumes the bundle is already built on the dev box
(`scripts/hpc/make_hpc_bundle.py build`). **Exactly two values get filled in by hand**:
`slurm.partition` and `slurm.account` in `configs/hpc_sweep.yaml`, step 4. Nothing else
is edited on the cluster. Account `perun26011488` (plan.md §7) — confirm the allocation
before the first submit.

### 1 — transfer

```bash
rsync -avP --partial dist/lockon_hpc_<sha>.tar.gz dist/lockon_hpc_<sha>.tar.gz.sha256 <user>@perun.tuke.sk:~/
```

On a flaky link, build with `--no-archive` and send the staged directory instead —
same contents, resumable per member, and step 2 drops the `sha256sum`/`tar` line:
`rsync -avP dist/lockon_hpc/ <user>@perun.tuke.sk:~/lockon_hpc/`

### 2 — unpack + verify

```bash
cd ~ && sha256sum -c lockon_hpc_<sha>.tar.gz.sha256 && tar xzf lockon_hpc_<sha>.tar.gz
cd lockon_hpc && python3 make_hpc_bundle.py verify && python3 make_hpc_bundle.py unpack --dest .
```

`verify` must print `VERIFY OK`. Unpack creates `lockon_hpc/repo/` and
`lockon_hpc/wheelhouse/` — a few hundred files, no inode-quota concern (unlike
ws/100_occlusion_mot's 509k-file reid tree; this repo ships no external datasets).

### 3 — environment (offline, from the bundled wheelhouse)

```bash
cd ~/lockon_hpc/repo && module load Python/3.11 2>/dev/null; bash scripts/hpc/hpc_bootstrap.sh
```

Must end with `BOOTSTRAP OK`. It hard-fails on glibc < 2.28 or a non-3.11 interpreter.

### 4 — fill the two unknowns

```bash
sed -i 's|<FILL-PERUN-PARTITION>|YOUR_PARTITION|; s|<FILL-PERUN-ACCOUNT>|YOUR_ACCOUNT|' configs/hpc_sweep.yaml
grep -E 'partition|account' configs/hpc_sweep.yaml
```

### 5 — generate the job scripts

```bash
.venv/bin/python scripts/hpc/sweep_launcher.py --mode slurm --config configs/hpc_sweep.yaml
```

Emits `runs/hpc/{submit_smoke.sbatch,submit.sbatch,submit_gpu_prey.sbatch}` and prints
the CPU-h / GPU-h budget table. `submit_gpu_prey.sbatch` is a **placeholder** —
`lockon.harness.train_prey` does not exist yet (harness/SPEC.md places it after step
7); do not submit it until that lands. Any `STILL a placeholder` warning about
partition/account means step 4 did not take — stop and fix it.

### 6 — submit (smoke first, array chained behind it)

```bash
cd ~/lockon_hpc/repo
SMOKE=$(sbatch --parsable runs/hpc/submit_smoke.sbatch)
sbatch --dependency=afterok:$SMOKE runs/hpc/submit.sbatch
```

Submit **from the repo root** — the scripts check for this and refuse otherwise. The
smoke job runs one unit at `total_steps=20_000` end-to-end (train + eval), proving the
offline env and the `lockon.harness.train` CLI contract before the 45-unit array
commits CPU-h.

### 7 — monitor

```bash
squeue -u $USER -o '%.10i %.20j %.9P %.8T %.10M %.6D %R'
tail -f runs/hpc/logs/*.out
find runs/hpc -name result.json | wc -l    # units finished (target: 45)
```

**Done** = 45 `result.json` files under `runs/hpc/<unit>/` — 5 seeds × 3 reward
variants × 3 arena densities (`scripts/hpc/sweep_common.py::enumerate_units`).

### 8 — pull results back

Pull **only small artifacts** — result JSONs, curve PNGs, one `best.zip` per unit of
interest — never the raw checkpoints or logs directory wholesale:

```bash
rsync -avP --include='*/' --include='result.json' --include='config.yaml' \
  --exclude='*' <user>@perun.tuke.sk:~/lockon_hpc/repo/runs/hpc/ runs/hpc/
```

If a `best.zip` is wanted locally (< 100 MB — SB3 MlpPolicy [128,128] checkpoints are
a few MB), pull it explicitly per unit:
`rsync -avP <user>@perun.tuke.sk:~/lockon_hpc/repo/runs/hpc/<unit>/best.zip runs/hpc/<unit>/`

---

**If the array dies partway**, just re-submit it: every task exits immediately when
its `result.json` already exists. No flags, no bookkeeping.

**Budget** (plan.md §5 step 5, §7): CPU-h = 45 units × `est_h_per_unit`
(`configs/hpc_sweep.yaml`, default 2 h/unit until measured from the smoke job's wall
time — never guessed past that point). GPU-h = 20 h (learned-prey trial, hard `timeout`
cap) + 2 h (curves/three-shots render, EGL) = 22 GPU-h, well inside the 70–100 GPU-h
cap (plan.md §7). Update `est_h_per_unit` from the smoke job before submitting the
array; it is an input to `scripts/hpc/sweep_common.py::budget_table`, never a threshold
that moves to make the ceiling check pass (CLAUDE.md).

**GPU units.** The CPU wheelhouse (`requirements-hpc.in`, CPU-only torch) does not
cover the learned-prey trial or the render pass — both need a CUDA torch and MuJoCo's
EGL backend (`MUJOCO_GL=egl`, `osmesa` fallback via `LOCKON_FORCE_OSMESA=1` — see
`scripts/hpc/hpc_env.sh`). Build a second wheelhouse from a `--extra-index-url
.../whl/cu126` copy of `requirements-hpc.in` once `train_prey.py` exists; until then
`runs/hpc/submit_gpu_prey.sbatch` is emitted as a documented placeholder only.
