"""Build the MedRGB benchmark files: topic generation, offline retrieval, sub-QA creation, counterfactual editing."""
import os
import json
import argparse
from tqdm import tqdm

from medrgb.retrieval import RetrievalSystem
from medrgb.inference import MedRAGInference
from medrgb.benchmark import TopicGenerator, SubQAGenerator, CounterfactualGenerator
from medrgb.config import RetrieverConfig, LLMConfig


def parse_args():
    parser = argparse.ArgumentParser(description="Build MedRGB benchmark files")
    parser.add_argument("--qa_file", required=True, help="Input QA dataset JSONL (question_id, question, options, answer)")
    parser.add_argument("--output_dir", default="data/benchmark", help="Output directory")
    parser.add_argument("--retriever", default="MedCPT", help="Retriever name")
    parser.add_argument("--corpus", default="Textbooks", help="Corpus name")
    parser.add_argument("--db_dir", default="./corpus", help="Corpus directory")
    parser.add_argument("--llm", default="OpenAI/gpt-4o", help="LLM for data generation")
    parser.add_argument("--k", type=int, default=20, help="Docs per topic retrieved")
    parser.add_argument("--n_topics", type=int, default=5, help="Number of retrieval topics per question")
    return parser.parse_args()


def main():
    args = parse_args()
    os.makedirs(args.output_dir, exist_ok=True)

    retriever_cfg = RetrieverConfig(retriever_name=args.retriever, corpus_name=args.corpus, db_dir=args.db_dir, k=args.k)
    llm_cfg = LLMConfig(llm_name=args.llm)

    retrieval_system = RetrievalSystem(args.retriever, args.corpus, args.db_dir)
    llm_inference = MedRAGInference(llm_cfg, retriever_cfg, rag=False)

    topic_gen = TopicGenerator(llm_inference, n_topics=args.n_topics)
    subqa_gen = SubQAGenerator(llm_inference)
    cf_gen = CounterfactualGenerator(llm_inference)

    qa_items = [json.loads(line) for line in open(args.qa_file)]

    signal_docs = {}
    subqa_pairs = {}
    counterfactual_docs = {}

    for item in tqdm(qa_items, desc="Building benchmark"):
        qid = item["question_id"]
        question = item["question"]
        options_str = "\n".join(f"{k}. {v}" for k, v in item.get("options", {}).items())
        answer = item.get("answer", "")

        topics = topic_gen.generate(question, options_str)
        all_docs = []
        for topic in topics:
            docs, _ = retrieval_system.retrieve(topic, k=3)
            all_docs.extend(docs)

        seen_ids = set()
        unique_docs = []
        for d in all_docs:
            if d["id"] not in seen_ids:
                seen_ids.add(d["id"])
                unique_docs.append(d)

        signal_docs[qid] = unique_docs

        subqas = subqa_gen.generate(question, unique_docs)
        subqa_pairs[qid] = subqas

        cf_items = []
        for doc in unique_docs[:5]:
            cf = cf_gen.generate(question, answer, doc)
            if cf:
                cf_items.append(cf)
        counterfactual_docs[qid] = cf_items

    with open(os.path.join(args.output_dir, "signal_docs.json"), "w") as f:
        json.dump(signal_docs, f, indent=2)
    with open(os.path.join(args.output_dir, "subqa_pairs.json"), "w") as f:
        json.dump(subqa_pairs, f, indent=2)
    with open(os.path.join(args.output_dir, "counterfactual_docs.json"), "w") as f:
        json.dump(counterfactual_docs, f, indent=2)

    print(f"Benchmark files saved to {args.output_dir}/")


if __name__ == "__main__":
    main()
