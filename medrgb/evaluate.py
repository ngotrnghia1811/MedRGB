import re
import json
import string
from typing import List, Dict, Optional, Tuple


# ============================================================
# Answer extraction helpers
# ============================================================

def extract_answer_choice(response: str) -> Optional[str]:
    """Extract the answer letter (A/B/C/D/...) from a JSON response string."""
    try:
        data = json.loads(response)
        if isinstance(data, dict):
            for key in ("answer_choice", "answer"):
                val = data.get(key, "")
                if val:
                    return _normalize_choice(str(val))
    except (json.JSONDecodeError, AttributeError):
        pass

    match = re.search(r'"answer_choice"\s*:\s*"([^"]+)"', response)
    if match:
        return _normalize_choice(match.group(1))

    match = re.search(r'\b([A-E])\b', response)
    if match:
        return match.group(1).upper()

    return None


def _normalize_choice(text: str) -> str:
    text = text.strip().upper()
    if text in {"A", "B", "C", "D", "E"}:
        return text
    if "insufficient" in text.lower():
        return "insufficient information"
    return text


def extract_main_and_sub_answers(response: str) -> Tuple[Optional[str], dict]:
    """Parse an integration or robustness response into (main_answer, sub_answers).

    sub_answers is a dict: sub_key → {"answer": ..., "doc_id": ..., "factual_correctness": ...}
    """
    main_answer = None
    sub_answers = {}

    try:
        data = json.loads(response)
        if not isinstance(data, dict):
            return main_answer, sub_answers

        if "main" in data:
            main_answer = _normalize_choice(str(data["main"].get("answer_choice", "")))

        for key, val in data.items():
            if key.startswith("sub_") and isinstance(val, dict):
                sub_answers[key] = {
                    "answer": val.get("answer", ""),
                    "doc_id": val.get("doc_id", val.get("relevant_doc_id", "")),
                    "factual_correctness": val.get("factual_correctness", ""),
                }
    except (json.JSONDecodeError, AttributeError):
        main_answer = extract_answer_choice(response)

    return main_answer, sub_answers


def extract_noise_classification(response: str) -> Tuple[List[str], List[str]]:
    """Extract relevant and irrelevant doc IDs from a sufficiency response."""
    relevant, irrelevant = [], []
    try:
        data = json.loads(response)
        if isinstance(data, dict):
            relevant = data.get("relevant_doc_id", [])
            irrelevant = data.get("irrelevant_doc_id", [])
    except (json.JSONDecodeError, AttributeError):
        pass
    return relevant, irrelevant


# ============================================================
# Normalization for exact-match scoring
# ============================================================

def normalize_answer(text: str) -> str:
    """Lowercase, remove punctuation, articles, and extra whitespace."""
    text = text.lower()
    text = text.translate(str.maketrans("", "", string.punctuation))
    text = re.sub(r"\b(a|an|the)\b", " ", text)
    text = re.sub(r"\s+", " ", text).strip()
    return text


def exact_match(prediction: str, ground_truth: str) -> float:
    return float(normalize_answer(prediction) == normalize_answer(ground_truth))


def token_f1(prediction: str, ground_truth: str) -> float:
    pred_tokens = normalize_answer(prediction).split()
    gt_tokens = normalize_answer(ground_truth).split()
    if not pred_tokens or not gt_tokens:
        return float(pred_tokens == gt_tokens)
    common = set(pred_tokens) & set(gt_tokens)
    if not common:
        return 0.0
    precision = len(common) / len(pred_tokens)
    recall = len(common) / len(gt_tokens)
    return 2 * precision * recall / (precision + recall)


# ============================================================
# Scenario-specific metrics
# ============================================================

class MedRGBEvaluator:
    """Computes all MedRGB metrics for each test scenario."""

    def evaluate_standard(self, predictions: List[dict], references: List[dict]) -> dict:
        """Accuracy for Standard-RAG test.

        Args:
            predictions: list of {"question_id": ..., "response": ...}
            references:  list of {"question_id": ..., "answer": "A"/"B"/...}

        Returns:
            {"accuracy": float, "n_correct": int, "n_total": int}
        """
        ref_map = {r["question_id"]: r["answer"].upper() for r in references}
        correct = 0
        for pred in predictions:
            qid = pred["question_id"]
            pred_answer = extract_answer_choice(pred["response"])
            if pred_answer == ref_map.get(qid, ""):
                correct += 1
        n = len(predictions)
        return {"accuracy": correct / n if n else 0.0, "n_correct": correct, "n_total": n}

    def evaluate_sufficiency(self, predictions: List[dict], references: List[dict], signal_doc_ids: Dict[str, List[str]]) -> dict:
        """Three metrics for Sufficiency test:

        - main_accuracy: % correct answers
        - insuf_rate: % responses flagged as insufficient information
        - noise_detection_rate: % noise documents correctly identified as irrelevant

        Args:
            predictions:     list of {"question_id": ..., "p_sig": ..., "response": ..., "noise_doc_ids": [...]}
            references:      list of {"question_id": ..., "answer": ...}
            signal_doc_ids:  dict mapping question_id → list of signal doc IDs in the context
        """
        ref_map = {r["question_id"]: r["answer"].upper() for r in references}

        total = len(predictions)
        n_correct = 0
        n_insuf = 0
        noise_det_correct = 0
        noise_det_total = 0

        for pred in predictions:
            qid = pred["question_id"]
            pred_answer = extract_answer_choice(pred["response"])

            if pred_answer == ref_map.get(qid, ""):
                n_correct += 1
            if pred_answer == "insufficient information":
                n_insuf += 1

            noise_ids = set(pred.get("noise_doc_ids", []))
            if noise_ids:
                _, classified_irrelevant = extract_noise_classification(pred["response"])
                classified_irrelevant_set = set(classified_irrelevant)
                noise_det_correct += len(noise_ids & classified_irrelevant_set)
                noise_det_total += len(noise_ids)

        return {
            "main_accuracy": n_correct / total if total else 0.0,
            "insuf_rate": n_insuf / total if total else 0.0,
            "noise_detection_rate": noise_det_correct / noise_det_total if noise_det_total else 0.0,
            "n_total": total,
        }

    def evaluate_integration(self, predictions: List[dict], references: List[dict]) -> dict:
        """Metrics for Integration test:

        - main_accuracy: % correct main answers
        - sub_exact_match: avg exact-match score of sub-answers
        - sub_token_f1: avg token-F1 of sub-answers (proxy for GPT-based score)
        """
        ref_map = {r["question_id"]: r for r in references}

        main_correct = 0
        sub_em_scores = []
        sub_f1_scores = []
        total = len(predictions)

        for pred in predictions:
            qid = pred["question_id"]
            ref = ref_map.get(qid, {})
            main_answer, sub_answers = extract_main_and_sub_answers(pred["response"])

            if main_answer == str(ref.get("answer", "")).upper():
                main_correct += 1

            ref_subqas = {sq["question"]: sq["answer"] for sq in ref.get("sub_questions", [])}
            for sub_key, sub_val in sub_answers.items():
                pred_sub = sub_val.get("answer", "")
                matched_gt = None
                for gt_q, gt_a in ref_subqas.items():
                    if gt_q in pred.get("response", ""):
                        matched_gt = gt_a
                        break
                if matched_gt:
                    sub_em_scores.append(exact_match(pred_sub, matched_gt))
                    sub_f1_scores.append(token_f1(pred_sub, matched_gt))

        return {
            "main_accuracy": main_correct / total if total else 0.0,
            "sub_exact_match": sum(sub_em_scores) / len(sub_em_scores) if sub_em_scores else 0.0,
            "sub_token_f1": sum(sub_f1_scores) / len(sub_f1_scores) if sub_f1_scores else 0.0,
            "n_total": total,
        }

    def evaluate_robustness(self, predictions: List[dict], references: List[dict]) -> dict:
        """Metrics for Robustness test:

        - main_accuracy: % correct main answers
        - factual_detection_rate: % adversarial documents correctly identified as factually incorrect
        - sub_exact_match: sub-answer exact-match (corrected answers)
        - sub_token_f1: sub-answer token-F1
        """
        ref_map = {r["question_id"]: r for r in references}

        main_correct = 0
        fact_det_correct = 0
        fact_det_total = 0
        sub_em_scores = []
        sub_f1_scores = []
        total = len(predictions)

        for pred in predictions:
            qid = pred["question_id"]
            ref = ref_map.get(qid, {})
            main_answer, sub_answers = extract_main_and_sub_answers(pred["response"])

            if main_answer == str(ref.get("answer", "")).upper():
                main_correct += 1

            adversarial_ids = set(pred.get("adversarial_doc_ids", []))
            fact_det_total += len(adversarial_ids)
            for sub_key, sub_val in sub_answers.items():
                doc_id = sub_val.get("doc_id", "")
                if doc_id in adversarial_ids:
                    is_detected = str(sub_val.get("factual_correctness", "true")).lower() == "false"
                    if is_detected:
                        fact_det_correct += 1

            ref_subqas = {sq["question"]: sq.get("correct_answer", sq.get("answer", "")) for sq in ref.get("sub_questions", [])}
            for sub_key, sub_val in sub_answers.items():
                pred_sub = sub_val.get("answer", "")
                for gt_q, gt_a in ref_subqas.items():
                    if gt_q in pred.get("response", ""):
                        sub_em_scores.append(exact_match(pred_sub, gt_a))
                        sub_f1_scores.append(token_f1(pred_sub, gt_a))
                        break

        return {
            "main_accuracy": main_correct / total if total else 0.0,
            "factual_detection_rate": fact_det_correct / fact_det_total if fact_det_total else 0.0,
            "sub_exact_match": sum(sub_em_scores) / len(sub_em_scores) if sub_em_scores else 0.0,
            "sub_token_f1": sum(sub_f1_scores) / len(sub_f1_scores) if sub_f1_scores else 0.0,
            "n_total": total,
        }

    def summarize(self, all_results: Dict[str, dict]) -> str:
        """Format a summary table of results across scenarios and datasets."""
        lines = ["=" * 60, "MedRGB Evaluation Summary", "=" * 60]
        for key, metrics in all_results.items():
            lines.append(f"\n[{key}]")
            for metric_name, value in metrics.items():
                if isinstance(value, float):
                    lines.append(f"  {metric_name}: {value * 100:.1f}%")
                else:
                    lines.append(f"  {metric_name}: {value}")
        return "\n".join(lines)
