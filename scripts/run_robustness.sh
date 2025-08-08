#!/bin/bash
# Run Robustness test evaluation on all four datasets

DATASETS=("bioasq" "pubmedqa" "medqa" "mmlu")

for DATASET in "${DATASETS[@]}"; do
    echo "Running Robustness test on ${DATASET}..."
    python scripts/run_evaluate.py \
        --config configs/robustness.yaml \
        --dataset "${DATASET}" \
        --qa_file "data/${DATASET}_test.jsonl"
done
