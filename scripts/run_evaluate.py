"""Run MedRGB evaluation for a given scenario and dataset."""
import os
import json
import argparse
import yaml
from tqdm import tqdm

from medrgb.config import MedRGBConfig, RetrieverConfig, LLMConfig, BenchmarkConfig
from medrgb.inference import MedRAGInference
from medrgb.benchmark import MedRGBBenchmark
from medrgb.evaluate import MedRGBEvaluator


def parse_args():
    parser = argparse.ArgumentParser(description="Run MedRGB evaluation")
    parser.add_argument("--config", required=True, help="Path to YAML config file")
    parser.add_argument("--dataset", default="bioasq", choices=["bioasq", "pubmedqa", "medqa", "mmlu"])
    parser.add_argument("--qa_file", required=True, help="Path to QA JSONL file")
    parser.add_argument("--p_sig", type=int, default=None, help="Signal ratio (0-100); if None, run all ratios")
    return parser.parse_args()


def main():
    args = parse_args()
    with open(args.config) as f:
        cfg_dict = yaml.safe_load(f)

    retriever_cfg = RetrieverConfig(**cfg_dict.get("retriever", {}))
    llm_cfg = LLMConfig(**cfg_dict.get("llm", {}))
    benchmark_cfg = BenchmarkConfig(**cfg_dict.get("benchmark", {}))
    output_dir = os.path.join(cfg_dict.get("output_dir", "./results"), args.dataset)
    os.makedirs(output_dir, exist_ok=True)

    qa_items = [json.loads(line) for line in open(args.qa_file)]
    scenario = benchmark_cfg.scenario
    signal_ratios = [args.p_sig] if args.p_sig is not None else benchmark_cfg.signal_ratios

    data_cfg = cfg_dict.get("data", {})
    if scenario in ("integration", "robustness"):
        bench = MedRGBBenchmark(
            signal_docs_path=data_cfg.get("signal_docs_path", "data/benchmark/signal_docs.json"),
            subqa_path=data_cfg.get("subqa_path"),
            counterfactual_path=data_cfg.get("counterfactual_path"),
            n_docs=benchmark_cfg.n_docs,
        )
    elif scenario == "sufficiency":
        bench = MedRGBBenchmark(
            signal_docs_path=data_cfg.get("signal_docs_path", "data/benchmark/signal_docs.json"),
            n_docs=benchmark_cfg.n_docs,
        )

    model = MedRAGInference(llm_cfg, retriever_cfg, rag=(scenario != "cot"))
    evaluator = MedRGBEvaluator()

    all_results = {}
    for p_sig in signal_ratios:
        predictions = []
        for item in tqdm(qa_items, desc=f"p_sig={p_sig}"):
            qid = item["question_id"]

            if scenario == "standard":
                instance = bench.build_standard(qid)
            elif scenario == "sufficiency":
                instance = bench.build_sufficiency(qid, p_sig)
            elif scenario == "integration":
                if p_sig < 20:
                    continue
                instance = bench.build_integration(qid, p_sig)
            elif scenario == "robustness":
                instance = bench.build_robustness(qid, p_sig)

            sub_questions_str = instance.get("sub_questions_str")
            response, _, _ = model.answer(
                question=item["question"],
                options=item.get("options"),
                scenario=scenario,
                ctx_str=instance["context"],
                sub_questions=sub_questions_str,
            )

            pred_entry = {
                "question_id": qid,
                "p_sig": p_sig,
                "response": response,
                "noise_doc_ids": [d["id"] for d in instance.get("documents", []) if d not in instance.get("signal_docs", [])],
                "adversarial_doc_ids": [d.get("original_id", "") for d in instance.get("documents", []) if "original_id" in d],
            }
            predictions.append(pred_entry)

        pred_path = os.path.join(output_dir, f"{scenario}_p{p_sig}_predictions.json")
        with open(pred_path, "w") as f:
            json.dump(predictions, f, indent=2)

        if scenario == "standard":
            result = evaluator.evaluate_standard(predictions, qa_items)
        elif scenario == "sufficiency":
            result = evaluator.evaluate_sufficiency(predictions, qa_items, {})
        elif scenario == "integration":
            result = evaluator.evaluate_integration(predictions, qa_items)
        elif scenario == "robustness":
            result = evaluator.evaluate_robustness(predictions, qa_items)

        all_results[f"p_sig={p_sig}"] = result
        print(f"[{scenario}] p_sig={p_sig}: {result}")

    result_path = os.path.join(output_dir, f"{scenario}_results.json")
    with open(result_path, "w") as f:
        json.dump(all_results, f, indent=2)
    print(evaluator.summarize(all_results))


if __name__ == "__main__":
    main()
