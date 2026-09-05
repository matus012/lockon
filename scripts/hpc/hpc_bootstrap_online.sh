#!/usr/bin/env bash
# Online bootstrap for PERUN (login node has PyPI + uv; measured 2026-09-05, deviation row 15).
# Builds two venvs in the repo: .venv (CPU torch, the 45-unit array) and .venv_gpu (cu126 torch,
# the learned-prey trial + render pass). Compute nodes are offline, so this runs on the login node.
#   bash scripts/hpc/hpc_bootstrap_online.sh [cpu|gpu|both]
set -euo pipefail
WHICH="${1:-both}"
REPO_ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/../.." && pwd)"
cd "${REPO_ROOT}"
PY311="${PY311:-/usr/bin/python3.11}"
[ -x "${PY311}" ] || { echo "ERROR: ${PY311} missing" >&2; exit 1; }
command -v uv >/dev/null || { echo "ERROR: uv not on PATH" >&2; exit 1; }
export UV_CACHE_DIR="${UV_CACHE_DIR:-${REPO_ROOT}/.cache/uv}"

make_venv() {  # $1 = venv dir, $2 = torch index url
  local venv="$1" idx="$2"
  [ -x "${venv}/bin/python" ] || uv venv --python "${PY311}" "${venv}"
  uv pip install --python "${venv}/bin/python" --index-strategy unsafe-best-match \
    --extra-index-url "${idx}" -r requirements-hpc.txt
  uv pip install --python "${venv}/bin/python" --no-deps -e .
  "${venv}/bin/python" - <<'PYEOF'
import mujoco, stable_baselines3, trackers, torch, lockon
from lockon.core import EPISODE_STEPS
print("self-test OK: torch", torch.__version__, "cuda_build", torch.version.cuda, "mujoco", mujoco.__version__, "steps", EPISODE_STEPS)
PYEOF
}

if [ "${WHICH}" = "cpu" ] || [ "${WHICH}" = "both" ]; then
  echo "=== CPU venv (.venv) ==="
  make_venv "${REPO_ROOT}/.venv" "https://download.pytorch.org/whl/cpu"
fi
if [ "${WHICH}" = "gpu" ] || [ "${WHICH}" = "both" ]; then
  echo "=== GPU venv (.venv_gpu, cu126) ==="
  make_venv "${REPO_ROOT}/.venv_gpu" "https://download.pytorch.org/whl/cu126"
fi
echo "BOOTSTRAP OK"
