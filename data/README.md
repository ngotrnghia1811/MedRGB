# Data

MedRGB evaluates on four medical QA datasets from the [MIRAGE](https://github.com/Teddy-XiongGZ/MIRAGE) benchmark:

| Dataset | Split | # Questions | Answer Type | Source |
|---------|-------|:-----------:|-------------|--------|
| BioASQ-Y/N | Test (2019–2023) | 618 | Yes/No | Biomedical literature |
| PubMedQA* | Test | 500 | Yes/No/Maybe | PubMed abstracts |
| MedQA-US | Test | 1,273 | 4-choice MCQ | USMLE board exams |
| MMLU-Med | Test | 1,089 | 4-choice MCQ | MMLU medical subset |

## Obtaining the QA Datasets

Download via HuggingFace:

```bash
pip install datasets
python scripts/download_datasets.py
```

Or load directly in Python:

```python
from datasets import load_dataset
bioasq   = load_dataset("bigbio/bioasq", "bioasq_10b_source")
pubmedqa = load_dataset("pubmed_qa", "pqa_labeled")
medqa    = load_dataset("bigbio/med_qa", "med_qa_en_source")
mmlu     = load_dataset("cais/mmlu", "all")
```

## Corpus for Offline Retrieval

MedRGB uses [MedCorp](https://huggingface.co/MedRAG) from the MedRAG toolkit, consisting of four corpora:

| Corpus | #Snippets | Domain |
|--------|:---------:|--------|
| PubMed | 23.9M | Biomedical |
| StatPearls | 301.2k | Clinical |
| Textbooks | 125.8k | Medical |
| Wikipedia | 29.9M | General |

### Download and Index MedCorp

```bash
git lfs install

# Clone individual corpora (choose as needed)
git clone https://huggingface.co/datasets/MedRAG/pubmed      corpus/pubmed
git clone https://huggingface.co/datasets/MedRAG/textbooks   corpus/textbooks
git clone https://huggingface.co/datasets/MedRAG/statpearls  corpus/statpearls
git clone https://huggingface.co/datasets/MedRAG/wikipedia   corpus/wikipedia

# Chunk StatPearls (requires the raw NCBI dump)
python data/statpearls.py
```

Embedding and FAISS index are built automatically on first use of `RetrievalSystem`.

## Online Retrieval

Online retrieval requires:
- A Google Custom Search JSON API key (set `GOOGLE_API_KEY` env var)
- A Custom Search Engine ID (set `GOOGLE_CX` env var)

See `scripts/build_online_corpus.py` for the online retrieval pipeline.

## MedRGB Benchmark Files

After running the benchmark creation pipeline (`scripts/build_benchmark.py`), the following files are created under `data/benchmark/`:

```
data/benchmark/
├── signal_docs.json        # Per-question signal document sets
├── subqa_pairs.json        # Sub-question/answer pairs (integration)
└── counterfactual_docs.json  # Adversarially edited documents (robustness)
```
