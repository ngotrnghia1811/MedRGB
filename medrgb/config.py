from dataclasses import dataclass, field
from typing import List, Optional


CORPUS_NAMES = {
    "PubMed": ["pubmed"],
    "Textbooks": ["textbooks"],
    "StatPearls": ["statpearls"],
    "Wikipedia": ["wikipedia"],
    "MedCorp": ["pubmed", "textbooks", "statpearls", "wikipedia"],
}

RETRIEVER_NAMES = {
    "BM25": ["bm25"],
    "Contriever": ["facebook/contriever"],
    "SPECTER": ["allenai/specter"],
    "MedCPT": ["ncbi/MedCPT-Query-Encoder"],
    "RRF-2": ["bm25", "ncbi/MedCPT-Query-Encoder"],
    "RRF-4": ["bm25", "facebook/contriever", "allenai/specter", "ncbi/MedCPT-Query-Encoder"],
}

SUPPORTED_LLMS = [
    "OpenAI/gpt-3.5-turbo",
    "OpenAI/gpt-4o",
    "OpenAI/gpt-4o-mini",
    "Google/gemini-1.0-pro",
    "meta-llama/Meta-Llama-3-70B-Instruct",
    "meta-llama/Llama-2-70b-chat-hf",
    "mistralai/Mixtral-8x7B-Instruct-v0.1",
    "epfl-llm/meditron-70b",
    "axiong/PMC_LLaMA_13B",
    "google/gemma-2-27b-it",
]

SIGNAL_RATIOS = [0, 20, 40, 60, 80, 100]

BENCHMARK_SCENARIOS = ["standard", "sufficiency", "integration", "robustness"]


@dataclass
class RetrieverConfig:
    retriever_name: str = "MedCPT"
    corpus_name: str = "Textbooks"
    db_dir: str = "./corpus"
    k: int = 32
    rrf_k: int = 100


@dataclass
class LLMConfig:
    llm_name: str = "OpenAI/gpt-3.5-turbo"
    temperature: float = 0.0
    max_length: int = 16384
    context_length: int = 15000
    cache_dir: Optional[str] = None
    device_map: str = "auto"
    attn_implementation: str = "eager"
    openai_api_key: Optional[str] = None


@dataclass
class BenchmarkConfig:
    scenario: str = "standard"
    signal_ratios: List[int] = field(default_factory=lambda: [0, 20, 40, 60, 80, 100])
    n_docs: int = 10
    n_signal_docs: int = 5
    use_online_retrieval: bool = False
    google_api_key: Optional[str] = None
    google_cx: Optional[str] = None


@dataclass
class MedRGBConfig:
    retriever: RetrieverConfig = field(default_factory=RetrieverConfig)
    llm: LLMConfig = field(default_factory=LLMConfig)
    benchmark: BenchmarkConfig = field(default_factory=BenchmarkConfig)
    output_dir: str = "./results"
    datasets: List[str] = field(default_factory=lambda: ["bioasq", "pubmedqa", "medqa", "mmlu"])
