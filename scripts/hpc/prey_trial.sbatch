#!/bin/bash
#SBATCH --job-name=lockon_prey_trial
#SBATCH --partition=gpu_short
#SBATCH --account=perun26011488
#SBATCH --time=19:30:00
#SBATCH --gres=gpu:1
#SBATCH --cpus-per-task=32
#SBATCH --mem=48G
#SBATCH --output=runs/hpc/logs/%x_%j.out
#SBATCH --error=runs/hpc/logs/%x_%j.err
# Learned-prey trial (plan.md §7, owner cap ≤ 20 GPU-h): PPO prey vs the frozen scripted hunter.
# Hard wall: `timeout` 18 h on training + ≤ 1 h eval/table inside the same allocation. One GPU.
set -euo pipefail
[ -f scripts/hpc/unit.py ] || { echo "submit from the repo root" >&2; exit 2; }
export PYTHON="$PWD/.venv_gpu/bin/python"
source scripts/hpc/hpc_env.sh
export MUJOCO_GL=egl
echo "=== node $(hostname)"
"$PYTHON" - <<'PY'
import torch
assert torch.cuda.is_available(), "CUDA unavailable -- STOP (refactored_method §5)"
print("CUDA OK", torch.cuda.get_device_name(0))
PY
mkdir -p runs/hpc_prey
timeout --signal=TERM --kill-after=300 64800s \
  "$PYTHON" -m lockon.harness.train_prey --config configs/prey_gpu.yaml --out runs/hpc_prey \
    --hunter scripted --device cuda --wall-hours 17.5 --n-envs "${SLURM_CPUS_PER_TASK}" \
    ${PREY_RESUME:+--resume} || echo "train_prey exited $? (timeout=124 is the planned wall stop)"
echo "=== trial table: hunters x {scripted prey, learned prey}, held-out seeds 1000-1079"
for H in static scripted runs/ppo_local/best.zip; do
  for P in scripted runs/hpc_prey/best.zip; do
    tag="$(basename "$H" .zip)_vs_$(basename "$P" .zip)"
    timeout 1800s "$PYTHON" -m lockon.harness.eval --policy "$H" --vs static --n 80 --seed-base 1000 \
      --prey "$P" --json "runs/hpc_prey/eval_${tag}.json" | tail -3 || echo "eval ${tag} failed"
  done
done
echo "PREY TRIAL DONE"
