#!/bin/bash
# Offline runtime environment for the PERUN lockon CPU sweep (plan.md §7).
#
# SOURCED, never executed: by scripts/hpc/hpc_bootstrap.sh and by every emitted
# submit*.sbatch. One place decides where the interpreter and cache dirs live, so no
# sbatch template has to repeat it (ported from ws/100_occlusion_mot/scripts/hpc_env.sh).
#
# Compute nodes are assumed to have NO internet.
#   MPLCONFIGDIR  matplotlib font cache; unset, matplotlib writes into $HOME and warns
#                 (or stalls) when $HOME is a slow parallel filesystem
#   MUJOCO_GL     render backend. Default `egl` (GPU render/prey nodes). The CPU
#                 state-PPO array never renders (project.md §2: RL trains on state,
#                 not pixels) so this is a no-op for it; it matters only for the GPU
#                 template (scripts/hpc/sweep_launcher.py::render_gpu_sbatch) and any
#                 future render pass. `osmesa` fallback: export LOCKON_FORCE_OSMESA=1
#                 before sourcing this file on a node with no EGL context (no GPU).

_HPC_ENV_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
export REPO_ROOT="$(cd "${_HPC_ENV_DIR}/../.." && pwd)"

# The bootstrap venv. Override by exporting PYTHON before submitting.
export PYTHON="${PYTHON:-${REPO_ROOT}/.venv/bin/python}"

export MPLCONFIGDIR="${MPLCONFIGDIR:-${REPO_ROOT}/.cache/matplotlib}"

export MUJOCO_GL="${MUJOCO_GL:-egl}"
if [ -n "${LOCKON_FORCE_OSMESA:-}" ]; then
  export MUJOCO_GL=osmesa
fi

# One CPU node per task: keep BLAS/OMP from oversubscribing the cores SLURM granted us.
export OMP_NUM_THREADS="${SLURM_CPUS_PER_TASK:-8}"
export MKL_NUM_THREADS="${OMP_NUM_THREADS}"

mkdir -p "${MPLCONFIGDIR}"
