#!/bin/bash
# Run Integration test evaluation on all four datasets

DATASETS=("bioasq" "pubmedqa" "medqa" "mmlu")

for DATASET in "${DATASETS[@]}"; do
    echo "Running Integration test on ${DATASET}..."
    python scripts/run_evaluate.py \
        --config configs/integration.yaml \
        --dataset "${DATASET}" \
        --qa_file "data/${DATASET}_test.jsonl"
done
