#!/bin/bash
#SBATCH --job-name=lockon_gpu_probe
#SBATCH --partition=gpu_short
#SBATCH --account=perun26011488
#SBATCH --time=00:15:00
#SBATCH --gres=gpu:1
#SBATCH --cpus-per-task=8
#SBATCH --mem=16G
#SBATCH --output=runs/hpc/logs/%x_%j.out
#SBATCH --error=runs/hpc/logs/%x_%j.err
# GPU-node probe before any GPU-hour is committed (refactored_method §5): CUDA visible to torch,
# MuJoCo EGL offscreen render works, train_prey CLI imports. Budget: ≤ 0.25 GPU-h.
set -euo pipefail
[ -f scripts/hpc/unit.py ] || { echo "submit from the repo root" >&2; exit 2; }
export PYTHON="$PWD/.venv_gpu/bin/python"
source scripts/hpc/hpc_env.sh
export MUJOCO_GL=egl
echo "=== node $(hostname) $(nvidia-smi --query-gpu=name,driver_version --format=csv,noheader | head -1)"
"$PYTHON" - <<'PY'
import torch
assert torch.cuda.is_available(), "CUDA unavailable on the GPU node -- STOP"
x = torch.ones(2048, 2048, device="cuda") @ torch.ones(2048, 2048, device="cuda")
print("CUDA OK", torch.cuda.get_device_name(0), "torch", torch.__version__, "cuda", torch.version.cuda, float(x[0, 0]))
PY
echo "=== EGL render probe"
"$PYTHON" scripts/probe_render.py || { echo "EGL failed, trying OSMesa"; MUJOCO_GL=osmesa "$PYTHON" scripts/probe_render.py; }
echo "=== state-speed env bench"
"$PYTHON" -m lockon.harness.bench_env --steps 2000 --min-sps 200
echo "=== train_prey import"
"$PYTHON" -c "import lockon.harness.train_prey as m; print('train_prey OK', m.__name__)" || echo "train_prey not present yet (expected before the prey code lands)"
echo "GPU PROBE OK"
