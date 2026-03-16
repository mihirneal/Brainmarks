#!/bin/bash

log_path="logs/make_aabc_task5.log"

mkdir -p "$(dirname "${log_path}")"

export MPLCONFIGDIR="${MPLCONFIGDIR:-/tmp}"

uv run python datasets/AABC/scripts/make_aabc_task5_arrow.py \
    --space flat \
    --train-batch-ids 0 1 2 3 7 10 11 12 13 14 \
    --validation-batch-ids 15 16 \
    --test-batch-ids 17 18 19 \
    --train-per-class 3800 \
    --validation-per-class 806 \
    --test-per-class 1008 \
    --overwrite \
    --num_proc 1 \
    --candidate-workers 8 \
    --sample-workers 4 \
    2>&1 | tee -a "${log_path}"
