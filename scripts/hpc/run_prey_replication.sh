#!/usr/bin/env bash
# Waits until the CPU array is finished and step 9 has taken its slot, then runs the learned-prey
# trial at two more training seeds, pulls their evals and rebuilds the combined table.
# Yields to step 9: never submits unless the queue has room to spare.
set -uo pipefail
HOST="${HOST:-login02.perun.tuke.sk}"
R="${REMOTE_BASE:-/mnt/project/perun26011488/lockon}/repo"
SSH="ssh -o BatchMode=yes -o ConnectTimeout=30 ${HOST}"
q() { ${SSH} "$@" 2>&1 | grep -v "post-quantum\|store now\|openssh.com\|WARNING: connection"; }
queued() { q "squeue -h -u \$USER -o %i | wc -l" | tail -1 | tr -dc '0-9'; }
array_jobs() { q "squeue -h -u \$USER -o %j | grep -c lockon_sweep" | tail -1 | tr -dc '0-9'; }

for SEED in 1 2; do
  echo "=== seed ${SEED}: waiting for the array to finish and a spare slot"
  while true; do
    a=$(array_jobs); n=$(queued)
    [ "${a:-1}" -eq 0 ] && [ "${n:-9}" -le 2 ] && break
    echo "$(date +%H:%M) array_jobs=$a queued=$n"; sleep 600
  done
  # see deviation row 23: --export leaves the job HELD on this cluster
  q "cd ${R} && sed 's|^set -euo pipefail\$|set -euo pipefail
SEED=${SEED}|' scripts/hpc/prey_replicate.sbatch > runs/prey_rep_s${SEED}.sbatch"
  JOB=$(q "cd ${R} && sbatch --parsable runs/prey_rep_s${SEED}.sbatch" | tail -1)
  echo "seed ${SEED} -> job ${JOB}"
  while q "squeue -h -j ${JOB} -o %i" | grep -q "${JOB}"; do sleep 300; done
  echo "seed ${SEED} finished"
  q "cd ${R} && tail -6 runs/hpc/logs/lockon_prey_rep_${JOB}.out"
done

echo "=== pull replication evals"
q "cd ${R} && tar czf /tmp/prey_rep.tgz runs/hpc_prey_s*/eval_*.json runs/hpc_prey_s*/train_log.jsonl"
scp -o BatchMode=yes -q "${HOST}:/tmp/prey_rep.tgz" runs/ && tar xzf runs/prey_rep.tgz -C . && rm runs/prey_rep.tgz
ls runs/hpc_prey_s1 runs/hpc_prey_s2 2>/dev/null | head -12
echo "=== per-seed tables"
for s in "" _s1 _s2; do
  d="runs/hpc_prey${s}"
  [ -d "$d" ] && uv run python scripts/prey_table.py --dir "$d" --out "reports/prey_trial${s:-_s0}.md" && echo "--- ${d}" && cat "reports/prey_trial${s:-_s0}.md" | sed -n '5,10p'
done
echo "PREY_REPLICATION_DONE"
