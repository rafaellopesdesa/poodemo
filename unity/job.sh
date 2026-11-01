#!/bin/bash
set -euo pipefail
python_bin=$1
stage=$2
run_root=$3
cd "$run_root"
export OMP_NUM_THREADS="${SLURM_CPUS_PER_TASK:-4}"
export OPENBLAS_NUM_THREADS="$OMP_NUM_THREADS"
export MKL_NUM_THREADS="$OMP_NUM_THREADS"
export PYTHONUNBUFFERED=1
export CUBLAS_WORKSPACE_CONFIG=:4096:8
if [[ "$stage" == normalize ]]; then
    srun "$python_bin" -m unity.workflow normalize --root "$run_root" --index "$SLURM_ARRAY_TASK_ID" --device cuda
else
    srun "$python_bin" -m unity.workflow "$stage" --root "$run_root" --index "$SLURM_ARRAY_TASK_ID"
fi
