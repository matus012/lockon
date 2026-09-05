#!/usr/bin/env bash
# Dev-box driver: sync code -> wait for a job slot (site cap: 4 queued jobs/user) -> GPU probe ->
# prey trial. Idempotent; re-run after a failure. bash scripts/hpc/launch_gpu_trial.sh
set -uo pipefail
HOST="${HOST:-login02.perun.tuke.sk}"
R="${REMOTE_BASE:-/mnt/project/perun26011488/lockon}/repo"
SSH="ssh -o BatchMode=yes -o ConnectTimeout=30 ${HOST}"
q() { ${SSH} "$@" 2>&1 | grep -v "post-quantum\|store now\|openssh.com\|WARNING: connection"; }

echo "=== sync src/configs/tests/scripts (HEAD $(git rev-parse --short HEAD))"
for d in src configs tests scripts pyproject.toml requirements-hpc.txt README.md; do
  scp -o BatchMode=yes -q -r "$d" "${HOST}:${R}/" || { echo "scp $d failed"; exit 1; }
done
q "cd ${R} && .venv_gpu/bin/python -c 'import lockon.harness.train_prey, lockon.harness.prey_gym; print(\"cluster import OK\")' && echo $(git rev-parse --short HEAD) > .synced_head"

wait_slot() {  # $1 = how many of my jobs may be queued before submitting one more
  while true; do
    n=$(q "squeue -h -u \$USER -o %i | wc -l" | tail -1 | tr -dc '0-9')
    [ -n "$n" ] && [ "$n" -le "$1" ] && return 0
    echo "$(date +%H:%M) queued=$n > $1, waiting"; sleep 300
  done
}

echo "=== GPU probe (waits for a slot)"
wait_slot 3
PROBE=$(q "cd ${R} && sbatch --parsable scripts/hpc/gpu_probe.sbatch" | tail -1)
echo "probe job $PROBE"
# slurmdbd is unreachable on this cluster (sacct: "Connection refused"), so job completion is
# detected by absence from squeue, never by sacct state (2026-09-05).
while q "squeue -h -j ${PROBE} -o %i" | grep -q "${PROBE}"; do sleep 60; done
echo "probe finished"
q "cd ${R} && tail -20 runs/hpc/logs/lockon_gpu_probe_${PROBE}.out; tail -5 runs/hpc/logs/lockon_gpu_probe_${PROBE}.err"
q "cd ${R} && grep -q 'GPU PROBE OK' runs/hpc/logs/lockon_gpu_probe_${PROBE}.out" || { echo "PROBE FAILED -- not submitting the trial"; exit 3; }

echo "=== prey trial (waits for a slot)"
wait_slot 3
TRIAL=$(q "cd ${R} && sbatch --parsable scripts/hpc/prey_trial.sbatch" | tail -1)
echo "prey trial job $TRIAL submitted $(date)"
echo "GPU_LAUNCH_DONE"
