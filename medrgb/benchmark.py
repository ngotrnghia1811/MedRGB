import os
import json
import random
from typing import List, Dict, Optional


class MedRGBBenchmark:
    """Constructs MedRGB benchmark instances from base QA datasets and signal documents.

    The four test scenarios are:
    - standard:    Signal-only context; step-by-step RAG QA.
    - sufficiency: Mixed signal + noise context; tests noise rejection and insufficient-info detection.
    - integration: Sub-questions generated per signal document; tests multi-hop integration.
    - robustness:  Adversarially edited signal documents; tests factual error detection.

    Each non-standard scenario is evaluated across p_sig ∈ {0, 20, 40, 60, 80, 100},
    where p_sig is the percentage of signal documents in the retrieved context.
    """

    SIGNAL_RATIOS = [0, 20, 40, 60, 80, 100]

    def __init__(
        self,
        signal_docs_path: str,
        subqa_path: Optional[str] = None,
        counterfactual_path: Optional[str] = None,
        n_docs: int = 10,
        seed: int = 42,
    ):
        """
        Args:
            signal_docs_path:       JSON file mapping question_id → list of signal documents.
            subqa_path:             JSON file mapping question_id → list of sub-QA pairs (integration/robustness).
            counterfactual_path:    JSON file mapping question_id → list of counterfactual documents (robustness).
            n_docs:                 Total number of documents in the retrieved context.
            seed:                   Random seed for reproducibility.
        """
        self.n_docs = n_docs
        self.seed = seed
        random.seed(seed)

        with open(signal_docs_path) as f:
            self.signal_docs: Dict[str, List[dict]] = json.load(f)

        self.subqa: Dict[str, List[dict]] = {}
        if subqa_path and os.path.exists(subqa_path):
            with open(subqa_path) as f:
                self.subqa = json.load(f)

        self.counterfactual: Dict[str, List[dict]] = {}
        if counterfactual_path and os.path.exists(counterfactual_path):
            with open(counterfactual_path) as f:
                self.counterfactual = json.load(f)

        all_signal_docs = [doc for docs in self.signal_docs.values() for doc in docs]
        self._noise_pool = all_signal_docs

    def build_standard(self, question_id: str, n_signal: int = 5) -> dict:
        """Build a Standard-RAG instance with n_signal signal documents."""
        signals = self.signal_docs.get(question_id, [])
        sampled = random.sample(signals, min(n_signal, len(signals)))
        return {
            "question_id": question_id,
            "scenario": "standard",
            "n_signal": len(sampled),
            "documents": self._format_documents(sampled),
            "context": self._build_context(sampled),
        }

    def build_sufficiency(self, question_id: str, p_sig: int) -> dict:
        """Build a Sufficiency instance with p_sig% signal documents.

        The remaining (100 - p_sig)% are noise documents sampled from other questions' signals.
        """
        assert p_sig in self.SIGNAL_RATIOS, f"p_sig must be in {self.SIGNAL_RATIOS}"

        signals = self.signal_docs.get(question_id, [])
        n_signal = round(self.n_docs * p_sig / 100)
        n_noise = self.n_docs - n_signal

        sampled_signal = random.sample(signals, min(n_signal, len(signals)))
        noise_pool = [d for d in self._noise_pool if d not in signals]
        sampled_noise = random.sample(noise_pool, min(n_noise, len(noise_pool)))

        docs = sampled_signal + sampled_noise
        random.shuffle(docs)

        return {
            "question_id": question_id,
            "scenario": "sufficiency",
            "p_sig": p_sig,
            "n_signal": len(sampled_signal),
            "n_noise": len(sampled_noise),
            "documents": self._format_documents(docs),
            "context": self._build_context(docs),
        }

    def build_integration(self, question_id: str, p_sig: int) -> dict:
        """Build an Integration instance with p_sig% signal documents and sub-questions.

        p_sig starts from 20 so that there is at least one signal document.
        """
        assert p_sig in self.SIGNAL_RATIOS and p_sig >= 20, "Integration requires p_sig >= 20"

        signals = self.signal_docs.get(question_id, [])
        subqas = self.subqa.get(question_id, [])

        n_signal = round(self.n_docs * p_sig / 100)
        n_noise = self.n_docs - n_signal

        sampled_signal = random.sample(signals, min(n_signal, len(signals)))
        noise_pool = [d for d in self._noise_pool if d not in signals]
        sampled_noise = random.sample(noise_pool, min(n_noise, len(noise_pool)))

        signal_ids = {d.get("id") for d in sampled_signal}
        relevant_subqas = [sq for sq in subqas if sq.get("doc_id") in signal_ids]

        sub_questions_str = self._format_sub_questions(relevant_subqas)

        docs = sampled_signal + sampled_noise
        random.shuffle(docs)

        return {
            "question_id": question_id,
            "scenario": "integration",
            "p_sig": p_sig,
            "n_signal": len(sampled_signal),
            "n_noise": len(sampled_noise),
            "sub_questions": relevant_subqas,
            "sub_questions_str": sub_questions_str,
            "documents": self._format_documents(docs),
            "context": self._build_context(docs),
        }

    def build_robustness(self, question_id: str, p_sig: int) -> dict:
        """Build a Robustness instance with p_sig% factually correct documents.

        All documents are relevant; (100 - p_sig)% have been adversarially edited.
        Sub-questions are the same as in the Integration test.
        """
        assert p_sig in self.SIGNAL_RATIOS, f"p_sig must be in {self.SIGNAL_RATIOS}"

        signals = self.signal_docs.get(question_id, [])
        counterfactuals = self.counterfactual.get(question_id, [])
        subqas = self.subqa.get(question_id, [])

        n_correct = round(self.n_docs * p_sig / 100)
        n_adversarial = self.n_docs - n_correct

        sampled_correct = random.sample(signals, min(n_correct, len(signals)))
        sampled_adversarial = random.sample(counterfactuals, min(n_adversarial, len(counterfactuals)))

        all_doc_ids = {d.get("id") for d in sampled_correct} | {d.get("original_id") for d in sampled_adversarial}
        relevant_subqas = [sq for sq in subqas if sq.get("doc_id") in all_doc_ids]
        sub_questions_str = self._format_sub_questions(relevant_subqas)

        docs = sampled_correct + sampled_adversarial
        random.shuffle(docs)

        return {
            "question_id": question_id,
            "scenario": "robustness",
            "p_sig": p_sig,
            "n_correct": len(sampled_correct),
            "n_adversarial": len(sampled_adversarial),
            "sub_questions": relevant_subqas,
            "sub_questions_str": sub_questions_str,
            "documents": self._format_documents(docs),
            "context": self._build_context(docs),
        }

    @staticmethod
    def _format_documents(docs: List[dict]) -> List[dict]:
        return [
            {
                "id": d.get("id", f"doc_{i}"),
                "title": d.get("title", ""),
                "content": d.get("content", d.get("text", "")),
            }
            for i, d in enumerate(docs)
        ]

    @staticmethod
    def _build_context(docs: List[dict]) -> str:
        lines = []
        for i, d in enumerate(docs):
            doc_id = d.get("id", f"doc_{i}")
            title = d.get("title", "")
            content = d.get("content", d.get("text", ""))
            lines.append(
                f"--- Start of DOC_{i+1} ---\n"
                f"ID: {doc_id}\n"
                f"Title: {title}\n"
                f"Content: {content}\n"
                f"--- END of DOC_{i+1} ---"
            )
        return "\n\n".join(lines)

    @staticmethod
    def _format_sub_questions(subqas: List[dict]) -> str:
        return "\n".join(
            f"{i+1}. {sq.get('question', '')}"
            for i, sq in enumerate(subqas)
        )


class TopicGenerator:
    """Generates diverse retrieval sub-topics for a question using an LLM.

    Used in the benchmark creation pipeline to improve retrieval diversity.
    """

    SYSTEM_PROMPT = (
        "You are a medical expert. Generate ranked search topics to help answer a medical question. "
        "Follow these guidelines:\n"
        "1) Rank topics by importance to the question.\n"
        "2) Ensure relevance to the question and answer options.\n"
        "3) The topics should be differentiable and efficient for information retrieval.\n"
        "Output as a JSON list of topic strings."
    )

    def __init__(self, llm_client, n_topics: int = 5):
        self.client = llm_client
        self.n_topics = n_topics

    def generate(self, question: str, options: str = "") -> List[str]:
        user_content = f"Question: {question}"
        if options:
            user_content += f"\nOptions: {options}"
        user_content += f"\n\nGenerate {self.n_topics} ranked search topics as a JSON list."

        messages = [
            {"role": "system", "content": self.SYSTEM_PROMPT},
            {"role": "user", "content": user_content},
        ]
        response = self.client.generate(messages)
        try:
            topics = json.loads(response)
            return topics if isinstance(topics, list) else []
        except json.JSONDecodeError:
            return [question]


class SubQAGenerator:
    """Generates sub-question/answer pairs for signal documents (Integration/Robustness).

    Each sub-question is tied to one signal document and its answer is a short
    extractive span from that document.
    """

    SYSTEM_PROMPT = (
        "You are a medical research expert. Follow these instructions:\n"
        "1) Explore different aspects related to the main question.\n"
        "2) Sub-question must be specific to a document.\n"
        "3) Sub-answer must be a short string extracted from the corresponding document.\n"
        "Output in JSON: a list of objects with keys 'doc_id', 'question', 'answer'."
    )

    def __init__(self, llm_client):
        self.client = llm_client

    def generate(self, question: str, documents: List[dict]) -> List[dict]:
        docs_str = "\n\n".join(
            f"Document ID: {d['id']}\nContent: {d['content']}"
            for d in documents
        )
        messages = [
            {"role": "system", "content": self.SYSTEM_PROMPT},
            {"role": "user", "content": f"Main question: {question}\n\nDocuments:\n{docs_str}"},
        ]
        response = self.client.generate(messages)
        try:
            subqas = json.loads(response)
            return subqas if isinstance(subqas, list) else []
        except json.JSONDecodeError:
            return []


class CounterfactualGenerator:
    """Creates adversarially edited (counterfactual) document-answer pairs for the Robustness test."""

    SYSTEM_PROMPT = (
        "You are a medical expert. Follow these instructions:\n"
        "1) Analyze the provided question, original answer, and document.\n"
        "2) Generate a deliberately incorrect new answer.\n"
        "3) Minimally edit the original document to create a persuasive but factually incorrect "
        "new document supporting the incorrect answer.\n"
        "Output JSON with keys 'new_answer', 'edited_document'."
    )

    def __init__(self, llm_client):
        self.client = llm_client

    def generate(self, question: str, original_answer: str, document: dict) -> dict:
        user_content = (
            f"Question: {question}\n"
            f"Original answer: {original_answer}\n"
            f"Document ID: {document['id']}\n"
            f"Document content: {document['content']}"
        )
        messages = [
            {"role": "system", "content": self.SYSTEM_PROMPT},
            {"role": "user", "content": user_content},
        ]
        response = self.client.generate(messages)
        try:
            result = json.loads(response)
            result["original_id"] = document["id"]
            return result
        except json.JSONDecodeError:
            return {}
