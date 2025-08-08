#!/bin/bash
# Run Sufficiency test evaluation on all four datasets

DATASETS=("bioasq" "pubmedqa" "medqa" "mmlu")

for DATASET in "${DATASETS[@]}"; do
    echo "Running Sufficiency test on ${DATASET}..."
    python scripts/run_evaluate.py \
        --config configs/sufficiency.yaml \
        --dataset "${DATASET}" \
        --qa_file "data/${DATASET}_test.jsonl"
done
