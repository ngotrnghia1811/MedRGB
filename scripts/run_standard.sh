#!/bin/bash
# Run Standard-RAG evaluation on all four datasets

DATASETS=("bioasq" "pubmedqa" "medqa" "mmlu")

for DATASET in "${DATASETS[@]}"; do
    echo "Running Standard-RAG on ${DATASET}..."
    python scripts/run_evaluate.py \
        --config configs/standard_rag.yaml \
        --dataset "${DATASET}" \
        --qa_file "data/${DATASET}_test.jsonl"
done
