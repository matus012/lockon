#!/usr/bin/env bash
# Dev-box driver for plan.md step 9. Run once the CPU array is (almost) done; safe to re-run.
#   bash scripts/hpc/run_step9.sh
set -uo pipefail
HOST="${HOST:-login02.perun.tuke.sk}"
R="${REMOTE_BASE:-/mnt/project/perun26011488/lockon}/repo"
SSH="ssh -o BatchMode=yes -o ConnectTimeout=30 ${HOST}"
q() { ${SSH} "$@" 2>&1 | grep -v "post-quantum\|store now\|openssh.com\|WARNING: connection"; }
count() { q "find ${R}/runs/hpc -maxdepth 2 -name result.json -not -path '*/_smoke*' | wc -l" | tail -1 | tr -dc '0-9'; }
queued() { q "squeue -h -u \$USER -o %i | wc -l" | tail -1 | tr -dc '0-9'; }

echo "=== 0 array completeness"
while true; do
  n=$(count); qd=$(queued)
  if [ "${n:-0}" -ge 45 ]; then break; fi
  if ! q "test -f ${R}/runs/hpc/base_d0.3_s0/result.json" >/dev/null && ! q "squeue -h -u \$USER -o %i | grep -q '_0\$'" >/dev/null && [ "${qd:-9}" -le 3 ]; then
    echo "resubmitting array index 0: $(q "cd ${R} && sbatch --parsable --array=0 runs/hpc/submit.sbatch" | tail -1)"
  fi
  echo "$(date +%H:%M) results=$n/45 queued=$qd"; sleep 600
done
echo "45/45 results"

echo "=== 1 pull result JSONs + configs"
mkdir -p runs/hpc
q "cd ${R} && tar czf /tmp/lockon_results.tgz \$(find runs/hpc -maxdepth 2 \( -name result.json -o -name config.yaml -o -name train_log.jsonl \) -not -path '*/_smoke*')"
scp -o BatchMode=yes -q "${HOST}:/tmp/lockon_results.tgz" runs/ && tar xzf runs/lockon_results.tgz -C . && rm runs/lockon_results.tgz

echo "=== 2 aggregate (local, same rule as on the cluster)"
uv run python scripts/hpc/aggregate.py --root runs/hpc --out reports/hpc
BEST=$(cat reports/hpc/best_unit.txt)
echo "best unit: $BEST"
scp -o BatchMode=yes -q "${HOST}:${R}/runs/hpc/${BEST}/best.zip" "runs/hpc/${BEST}/best.zip" && printf '{"obs_seen": "lock"}' > "runs/hpc/${BEST}/best.zip.obs.json"

echo "=== 3 step 9 GPU job (fork verdict n=80, curves, shots), waits for a slot"
scp -o BatchMode=yes -q scripts/hpc/step9.sbatch "${HOST}:${R}/scripts/hpc/step9.sbatch"
while [ "$(queued)" -gt 3 ]; do echo "$(date +%H:%M) waiting for a slot"; sleep 300; done
JOB=$(q "cd ${R} && sbatch --parsable --export=ALL,BEST=${BEST} scripts/hpc/step9.sbatch" | tail -1)
echo "step9 job $JOB"
# completion by absence from squeue (slurmdbd unreachable: sacct "Connection refused")
while q "squeue -h -j ${JOB} -o %i" | grep -q "${JOB}"; do sleep 120; done
echo "step9 finished"; q "cd ${R} && tail -15 runs/hpc/logs/lockon_step9_${JOB}.out"

echo "=== 4 pull step-9 artifacts (small only)"
mkdir -p runs/hpc_step9
q "cd ${R} && tar czf /tmp/lockon_step9.tgz --exclude='*.zip' --exclude='ckpt_*' runs/hpc_step9"
scp -o BatchMode=yes -q "${HOST}:/tmp/lockon_step9.tgz" runs/ && tar xzf runs/lockon_step9.tgz -C . && rm runs/lockon_step9.tgz
ls -la runs/hpc_step9 runs/hpc_step9/curves 2>/dev/null | head -20
echo "STEP9_DONE"
