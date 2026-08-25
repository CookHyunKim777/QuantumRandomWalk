#!/usr/bin/env bash
#SBATCH --job-name=qrw_a_sim
#SBATCH --output=logs/qrw_a_%A_%a.out
#SBATCH --error=logs/qrw_a_%A_%a.err
#SBATCH --time=04:00:00
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=2

# ============================================================
# QRW_A (Convex Combination Quantum-Random Walk) SLURM Script
# M ← α·(QW step) + (1-α)·(RW step)
# ============================================================

set -euo pipefail

PYTHON="${PYTHON:-python3}"
PROJECT_DIR="${PROJECT_DIR:-${SLURM_SUBMIT_DIR:-$(pwd)}}"

cd "$PROJECT_DIR"
mkdir -p logs results/qrw_a

echo "=========================================="
echo "QRW_A Simulation"
echo "Job ID: $SLURM_JOB_ID"
echo "Array Task ID: $SLURM_ARRAY_TASK_ID"
echo "Start Time: $(date)"
echo "=========================================="

PARAM_FILE="params/qrw_a/params_$(printf '%03d' "${SLURM_ARRAY_TASK_ID}").json"

if [ -f "$PARAM_FILE" ]; then
    "$PYTHON" run_qrw_c.py --param_file "$PARAM_FILE" --job_id "qrw_a_${SLURM_ARRAY_JOB_ID}_${SLURM_ARRAY_TASK_ID}"
else
    echo "ERROR: Parameter file not found: $PARAM_FILE"
    exit 1
fi

echo "End Time: $(date)"
