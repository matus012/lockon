#!/bin/bash
#SBATCH --job-name=lockon_prey_rep
#SBATCH --partition=gpu_short
#SBATCH --account=perun26011488
#SBATCH --time=04:00:00
#SBATCH --gres=gpu:1
#SBATCH --cpus-per-task=32
#SBATCH --mem=48G
#SBATCH --output=runs/hpc/logs/%x_%j.out
#SBATCH --error=runs/hpc/logs/%x_%j.err
# Replication of the learned-prey trial at a second/third training seed (deviation row 19).
# The headline claim — "the scripted hunter is the only policy the learned evader cannot
# degrade" — rested on one prey training run; this tests whether it replicates.
# Requires SEED=<n> (sbatch --export=ALL,SEED=1). Measured cost of seed 0: ~2.3 GPU-h.
set -euo pipefail
[ -f scripts/hpc/unit.py ] || { echo "submit from the repo root" >&2; exit 2; }
: "${SEED:?SEED=<n> required}"
export PYTHON="$PWD/.venv_gpu/bin/python"
source scripts/hpc/hpc_env.sh
export MUJOCO_GL=egl
OUT="runs/hpc_prey_s${SEED}"; mkdir -p "$OUT"
CFG="$OUT/prey_gpu_s${SEED}.yaml"
sed -E "s/^seed: [0-9]+/seed: ${SEED}/" configs/prey_gpu.yaml > "$CFG"
grep -E "^seed:" "$CFG"
echo "=== node $(hostname) seed=${SEED}"
"$PYTHON" - <<'PY'
import torch
assert torch.cuda.is_available(), "CUDA unavailable -- STOP (refactored_method §5)"
print("CUDA OK", torch.cuda.get_device_name(0))
PY
timeout --signal=TERM --kill-after=300 12600s \
  "$PYTHON" -m lockon.harness.train_prey --config "$CFG" --out "$OUT" \
    --hunter scripted --device cuda --wall-hours 3.4 --n-envs "${SLURM_CPUS_PER_TASK}" \
  || echo "train_prey exited $? (124 = planned wall stop)"
echo "=== same three hunters, same held-out seeds, this prey seed"
for H in static scripted runs/ppo_local/best.zip; do
  tag="$(basename "$H" .zip)_vs_best"
  timeout 1800s "$PYTHON" -m lockon.harness.eval --policy "$H" --vs static --n 80 --seed-base 1000 \
    --prey "$OUT/best.zip" --json "$OUT/eval_${tag}.json" | tail -2 || echo "eval ${tag} exit $?"
done
echo "PREY REPLICATE DONE seed=${SEED}"
