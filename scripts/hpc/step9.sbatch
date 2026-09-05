#!/bin/bash
#SBATCH --job-name=lockon_step9
#SBATCH --partition=gpu_short
#SBATCH --account=perun26011488
#SBATCH --time=03:00:00
#SBATCH --gres=gpu:1
#SBATCH --cpus-per-task=32
#SBATCH --mem=64G
#SBATCH --output=runs/hpc/logs/%x_%j.out
#SBATCH --error=runs/hpc/logs/%x_%j.err
# Step 9 (plan.md §5): fork verdict at n=80 held-out episodes, degradation curves ± std with the
# HPC-best policy, three shots + GIFs rendered with EGL. ≤ 2 GPU-h. Needs BEST=<unit name>
# (from scripts/hpc/aggregate.py -> reports/hpc/best_unit.txt), passed as the first argument.
set -euo pipefail
[ -f scripts/hpc/unit.py ] || { echo "submit from the repo root" >&2; exit 2; }
BEST="${1:-${BEST:-}}"   # `sbatch step9.sbatch <unit>`; NEVER --export (row 23: job held)
[ -n "$BEST" ] || { echo "usage: sbatch step9.sbatch <best-unit>" >&2; exit 2; }
export PYTHON="$PWD/.venv_gpu/bin/python"
source scripts/hpc/hpc_env.sh
export MUJOCO_GL=egl
export PYTHONIOENCODING=utf-8
OUT=runs/hpc_step9; mkdir -p "$OUT"
BEST_ZIP="runs/hpc/${BEST}/best.zip"
[ -f "$BEST_ZIP" ] || { echo "missing $BEST_ZIP" >&2; exit 4; }
echo "=== node $(hostname) best=$BEST"

echo "=== fork verdict, n=80, seeds 1000-1079 (pre-registered, deviation row 10)"
"$PYTHON" -m lockon.harness.eval --policy "$BEST_ZIP" --vs scripted --n 80 --seed-base 1000 --json "$OUT/eval_best_vs_scripted.json" | tail -4 || true
"$PYTHON" -m lockon.harness.eval --policy "$BEST_ZIP" --vs static --n 80 --seed-base 1000 --json "$OUT/eval_best_vs_static.json" | tail -4 || true
"$PYTHON" -m lockon.harness.eval --policy runs/ppo_local/best.zip --vs scripted --n 80 --seed-base 1000 --json "$OUT/eval_local_vs_scripted.json" | tail -4 || true

echo "=== curves with the HPC-best policy (5 axes x 5 values x 3 policies x 20 episodes)"
sed -e "s|runs/ppo_local/best.zip|${BEST_ZIP}|" -e "s|out_dir: reports/sweep|out_dir: ${OUT}/sweep|" \
    -e "s|curves_dir: reports/curves|curves_dir: ${OUT}/curves|" configs/sweep_local.yaml > "$OUT/sweep_hpc.yaml"
"$PYTHON" -m lockon.harness.sweep --config "$OUT/sweep_hpc.yaml" --workers 24 | tail -3

echo "=== shots (hero = scripted per row 10; learned hunter chase as a 4th clip if best beats scripted)"
"$PYTHON" -m lockon.demo.render_all --policy scripted --out "$OUT/shots" | tail -3
for S in occlusion lights_cut sensor3; do
  case $S in occlusion) A="--start auto --scale 0.35 --fps 8";; lights_cut) A="--start 46 --steps 54 --scale 0.35 --fps 8";; sensor3) A="--steps 30 --scale 0.4 --fps 6";; esac
  "$PYTHON" -m lockon.harness.render --scene $S --gif "$OUT/gifs/${S}.gif" $A | tail -1
done
"$PYTHON" -m lockon.harness.render --scene chase --policy "$BEST_ZIP" --mp4 "$OUT/shots/chase_hpc_best.mp4" --gif "$OUT/gifs/chase_hpc_best.gif" --scale 0.3 --fps 6 | tail -1 || true
echo "STEP9 DONE"
