#!/usr/bin/env bash
# Dev-box driver for HPC_RUNBOOK.md steps 1-6 (CPU array). Idempotent-ish: re-run after a failure.
#   bash scripts/hpc/launch_cpu_array.sh   (from the repo root, Git Bash on Windows)
set -euo pipefail
HOST="${HOST:-login02.perun.tuke.sk}"
REMOTE_BASE="${REMOTE_BASE:-/mnt/project/perun26011488/lockon}"
SSH="ssh -o BatchMode=yes -o ConnectTimeout=30 ${HOST}"

echo "=== 0 build bundle (repo tar from HEAD, no wheelhouse: deviation row 15)"
uv run python scripts/hpc/make_hpc_bundle.py build --no-wheelhouse
TAR="$(ls -t dist/lockon_hpc_*.tar.gz | head -1)"
SHA="${TAR}.sha256"
NAME="$(basename "${TAR}")"
echo "bundle: ${TAR}"

echo "=== 1 transfer"
scp -o BatchMode=yes "${TAR}" "${SHA}" "${HOST}:${REMOTE_BASE}/"

echo "=== 2 unpack + verify"
${SSH} "cd ${REMOTE_BASE} && sha256sum -c ${NAME}.sha256 && rm -rf lockon_hpc_new && mkdir lockon_hpc_new && tar xzf ${NAME} -C lockon_hpc_new --strip-components=1 && cd lockon_hpc_new && python3 make_hpc_bundle.py verify && python3 make_hpc_bundle.py unpack --dest . && ls repo | head -5"

echo "=== 2b promote: repo/ -> ${REMOTE_BASE}/repo (runs/ preserved)"
${SSH} "cd ${REMOTE_BASE} && mkdir -p repo && cp -r lockon_hpc_new/repo/. repo/ && rm -rf lockon_hpc_new && cd repo && git init -q 2>/dev/null || true; ls"

echo "=== 3 bootstrap (online, CPU venv)"
${SSH} "cd ${REMOTE_BASE}/repo && bash scripts/hpc/hpc_bootstrap_online.sh cpu 2>&1 | tail -5"

echo "=== 5 emit job scripts"
${SSH} "cd ${REMOTE_BASE}/repo && .venv/bin/python scripts/hpc/sweep_launcher.py --mode slurm --config configs/hpc_sweep.yaml 2>&1 | tail -8 && grep -n 'partition\|account' runs/hpc/submit.sbatch | head -4"

echo "=== 6 submit smoke + array"
${SSH} "cd ${REMOTE_BASE}/repo && SMOKE=\$(sbatch --parsable runs/hpc/submit_smoke.sbatch) && echo SMOKE=\$SMOKE && ARR=\$(sbatch --parsable --dependency=afterok:\$SMOKE runs/hpc/submit.sbatch) && echo ARRAY=\$ARR && squeue -u \$USER -o '%.10i %.22j %.9P %.8T %.10M %.6D %R'"
echo "LAUNCH_DONE"
