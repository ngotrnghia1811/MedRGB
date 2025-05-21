import os
import re
import json
import torch
import transformers
from transformers import AutoTokenizer, StoppingCriteria, StoppingCriteriaList

from medrgb.config import LLMConfig, RetrieverConfig
from medrgb.retrieval import RetrievalSystem
from medrgb.prompts import (
    cot_system, cot_prompt,
    standard_rag_system, standard_rag_prompt,
    sufficiency_system, sufficiency_prompt,
    integration_system, integration_prompt,
    robustness_system, robustness_prompt,
    lfqa_system, lfqa_system_with_prob, lfqa_system_with_score, lfqa_system_with_score_and_prob,
    lfqa_input_prompt,
    meditron_cot, meditron_rag,
)


class CustomStoppingCriteria(StoppingCriteria):
    """Stop generation when any of the stop_words appear in generated tokens."""

    def __init__(self, stop_words, tokenizer, input_len: int = 0):
        super().__init__()
        self.tokenizer = tokenizer
        self.stop_words = stop_words
        self.input_len = input_len

    def __call__(self, input_ids: torch.LongTensor, scores: torch.FloatTensor):
        tokens = self.tokenizer.decode(input_ids[0][self.input_len:])
        return any(word in tokens for word in self.stop_words)


class MedRAGInference:
    """Unified inference wrapper for MedRAG across all four MedRGB scenarios.

    Supports:
    - CoT (no retrieval)
    - Standard-RAG
    - Sufficiency test
    - Integration test
    - Robustness test
    - LFQA case study variants
    """

    SCENARIO_SYSTEMS = {
        "standard": standard_rag_system,
        "cot": cot_system,
        "sufficiency": sufficiency_system,
        "integration": integration_system,
        "robustness": robustness_system,
        "lfqa": lfqa_system,
        "lfqa_prob": lfqa_system_with_prob,
        "lfqa_score": lfqa_system_with_score,
        "lfqa_score_prob": lfqa_system_with_score_and_prob,
    }

    def __init__(
        self,
        llm_config: LLMConfig,
        retriever_config: RetrieverConfig,
        rag: bool = True,
    ):
        self.llm_config = llm_config
        self.rag = rag
        self.llm_name = llm_config.llm_name

        if rag:
            self.retrieval_system = RetrievalSystem(
                retriever_config.retriever_name,
                retriever_config.corpus_name,
                retriever_config.db_dir,
            )
        else:
            self.retrieval_system = None

        self._init_model()

    def _init_model(self):
        llm = self.llm_name
        cfg = self.llm_config

        if llm.lower().startswith("openai"):
            import tiktoken
            import openai
            self.model = llm.split("/")[-1]
            self._openai_client = openai.OpenAI(api_key=cfg.openai_api_key or os.environ.get("OPENAI_API_KEY"))
            if "gpt-3.5" in self.model or "gpt-35" in self.model:
                self.max_length = 16384
                self.context_length = 15000
            elif "gpt-4" in self.model:
                self.max_length = 128000
                self.context_length = 120000
            self.tokenizer = tiktoken.get_encoding("cl100k_base")

        elif "gemini" in llm.lower():
            import tiktoken
            import google.generativeai as genai
            genai.configure(api_key=os.environ["GOOGLE_API_KEY"])
            self.model = genai.GenerativeModel(
                model_name=llm.split("/")[-1],
                generation_config={"temperature": 0, "max_output_tokens": 2048},
            )
            self.max_length = 30720
            self.context_length = 28672
            self.tokenizer = tiktoken.get_encoding("cl100k_base")

        else:
            self.tokenizer = AutoTokenizer.from_pretrained(llm, cache_dir=cfg.cache_dir)
            self.max_length = cfg.max_length
            self.context_length = cfg.context_length

            if "mixtral" in llm.lower():
                self.tokenizer.chat_template = open("templates/mistral-instruct.jinja").read().replace("    ", "").replace("\n", "")
                self.max_length = 32768
                self.context_length = 30000
            elif "llama-2" in llm.lower():
                self.max_length = 4096
                self.context_length = 3072
            elif "llama-3" in llm.lower():
                self.max_length = 8192
                self.context_length = 7168
            elif "meditron-70b" in llm.lower():
                self.tokenizer.chat_template = open("templates/meditron.jinja").read().replace("    ", "").replace("\n", "")
                self.max_length = 4096
                self.context_length = 3072
            elif "pmc_llama" in llm.lower():
                self.tokenizer.chat_template = open("templates/pmc_llama.jinja").read().replace("    ", "").replace("\n", "")
                self.max_length = 2048
                self.context_length = 1024
            elif "gemma" in llm.lower():
                self.max_length = 4096
                self.context_length = 3072

            self.model = transformers.pipeline(
                "text-generation",
                model=llm,
                torch_dtype=torch.bfloat16,
                device_map=cfg.device_map,
                model_kwargs={
                    "cache_dir": cfg.cache_dir,
                    "attn_implementation": cfg.attn_implementation,
                },
            )

    def answer(
        self,
        question: str,
        options: dict = None,
        k: int = 32,
        rrf_k: int = 100,
        scenario: str = "standard",
        ctx_str: str = None,
        sub_questions: str = None,
        save_dir: str = None,
        return_messages: bool = False,
    ):
        """Run inference for a single question.

        Args:
            question:       The medical question.
            options:        Dict of answer options, e.g. {"A": "...", "B": "..."}.
            k:              Number of documents to retrieve.
            rrf_k:          RRF constant for score fusion.
            scenario:       One of "cot", "standard", "sufficiency", "integration", "robustness", "lfqa*".
            ctx_str:        Pre-built context string (skips retrieval when provided).
            sub_questions:  Formatted sub-questions string (required for integration/robustness).
            save_dir:       If set, saves snippets.json and response.json here.
            return_messages: If True, also returns the message list.

        Returns:
            (answer_str, retrieved_snippets, scores) or with messages if requested.
        """
        options_str = (
            "\n".join(f"{k}. {v}" for k, v in sorted(options.items())) if options else ""
        )

        retrieved_snippets, scores, context = self._retrieve(question, k, rrf_k, ctx_str)

        messages = self._build_messages(scenario, context, question, options_str, sub_questions)
        answer_text = re.sub(r"\s+", " ", self.generate(messages))

        if save_dir:
            os.makedirs(save_dir, exist_ok=True)
            with open(os.path.join(save_dir, "snippets.json"), "w") as f:
                json.dump(retrieved_snippets, f, indent=4)
            with open(os.path.join(save_dir, "response.json"), "w") as f:
                json.dump([answer_text], f, indent=4)

        if return_messages:
            return answer_text, retrieved_snippets, scores, messages
        return answer_text, retrieved_snippets, scores

    def _retrieve(self, question: str, k: int, rrf_k: int, ctx_str: str = None):
        if not self.rag:
            return [], [], []

        if ctx_str is not None:
            return [], [], ctx_str

        snippets, scores = self.retrieval_system.retrieve(question, k=k, rrf_k=rrf_k)
        for i, s in enumerate(snippets):
            s["score"] = scores[i]

        contexts = [
            f"Document [{i}] (Title: {snippets[i]['title']}) {snippets[i]['content']}"
            for i in range(len(snippets))
        ] or [""]
        ctx_raw = "\n".join(contexts)

        if "openai" in self.llm_name.lower() or "gemini" in self.llm_name.lower():
            import tiktoken
            enc = self.tokenizer if hasattr(self, "tokenizer") else tiktoken.get_encoding("cl100k_base")
            ctx_truncated = enc.decode(enc.encode(ctx_raw)[: self.context_length])
        else:
            ctx_truncated = self.tokenizer.decode(
                self.tokenizer.encode(ctx_raw, add_special_tokens=False)[: self.context_length]
            )

        return snippets, scores, ctx_truncated

    def _build_messages(self, scenario: str, context, question: str, options_str: str, sub_questions: str):
        system_text = self.SCENARIO_SYSTEMS.get(scenario, standard_rag_system)

        is_gemma = "gemma" in self.llm_name.lower()
        sys_role = "user" if is_gemma else "system"
        user_role = "model" if is_gemma else "user"

        if not self.rag:
            prompt_text = cot_prompt.render(question=question, options=options_str)
            if "meditron" in self.llm_name.lower():
                prompt_text = meditron_cot.render(question=question, options=options_str)
        elif scenario in ("lfqa", "lfqa_prob", "lfqa_score", "lfqa_score_prob"):
            prompt_text = lfqa_input_prompt.render(
                context=context, question=question, sub_questions=sub_questions or ""
            )
        elif scenario in ("integration", "robustness"):
            prompt_text = (integration_prompt if scenario == "integration" else robustness_prompt).render(
                context=context, question=question, options=options_str, sub_questions=sub_questions or ""
            )
        elif scenario == "sufficiency":
            prompt_text = sufficiency_prompt.render(context=context, question=question, options=options_str)
        else:
            prompt_text = standard_rag_prompt.render(context=context, question=question, options=options_str)
            if "meditron" in self.llm_name.lower():
                prompt_text = meditron_rag.render(context=context, question=question, options=options_str)

        messages = [{"role": sys_role, "content": system_text}]
        if is_gemma:
            messages += [{"role": user_role, "content": "\n"}]
            messages += [{"role": sys_role, "content": prompt_text}]
        else:
            messages += [{"role": user_role, "content": prompt_text}]

        return messages

    def generate(self, messages: list) -> str:
        """Run generation for the given message list and return the response string."""
        llm = self.llm_name.lower()

        if "openai" in llm:
            response = self._openai_client.chat.completions.create(
                model=self.model,
                messages=messages,
                temperature=0.0,
            )
            return response.choices[0].message.content

        elif "gemini" in llm:
            combined = messages[0]["content"] + "\n\n" + messages[1]["content"]
            response = self.model.generate_content(combined)
            return response.candidates[0].content.parts[0].text

        else:
            prompt = self.tokenizer.apply_chat_template(messages, tokenize=False, add_generation_prompt=True)

            stopping_criteria = None
            if "meditron" in llm:
                stopping_criteria = StoppingCriteriaList([
                    CustomStoppingCriteria(
                        ["###", "User:", "\n\n\n"],
                        self.tokenizer,
                        input_len=len(self.tokenizer.encode(prompt, add_special_tokens=True)),
                    )
                ])

            eos_ids = [self.tokenizer.eos_token_id]
            if "llama-3" in llm:
                eos_ids.append(self.tokenizer.convert_tokens_to_ids("<|eot_id|>"))

            response = self.model(
                prompt,
                do_sample=False,
                eos_token_id=eos_ids if len(eos_ids) > 1 else eos_ids[0],
                pad_token_id=self.tokenizer.eos_token_id,
                max_length=self.max_length,
                truncation=True,
                stopping_criteria=stopping_criteria,
            )
            return response[0]["generated_text"]
