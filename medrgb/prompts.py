from liquid import Template


# ============================================================
# Standard-RAG prompts
# ============================================================

standard_rag_system = (
    "You are a helpful medical expert, and your task is to answer a multi-choice medical question "
    "using the relevant documents. Please first think step-by-step and then choose the answer from "
    "the provided options. Organize your output in a json formatted as "
    'Dict{"step_by_step_thinking": Str(explanation), "answer_choice": Str{A/B/C/...}}. '
    "Your responses will be used for research purposes only, so please have a definite answer."
)

cot_system = (
    "You are a helpful medical expert, and your task is to answer a multi-choice medical question. "
    "Please first think step-by-step and then choose the answer from the provided options. Organize "
    "your output in a json formatted as "
    'Dict{"step_by_step_thinking": Str(explanation), "answer_choice": Str{A/B/C/...}}. '
    "Your responses will be used for research purposes only, so please have a definite answer."
)

cot_prompt = Template("""
Here is the question:
{{question}}

Here are the potential choices:
{{options}}

Please think step-by-step and generate your output in json:
""")

standard_rag_prompt = Template("""
Here are the relevant documents:
{{context}}

Here is the question:
{{question}}

Here are the potential choices:
{{options}}

Please think step-by-step and generate your output in json:
""")


# ============================================================
# Sufficiency test prompts
# ============================================================

sufficiency_system = """You are a knowledgeable medical expert. Your task is to answer a multiple-choice medical question using the provided external documents. Note that some of these documents may be noisy or irrelevant to the question.

Instructions:
1. Carefully analyze all provided documents.
2. Identify which documents are relevant and which are irrelevant to the question.
3. Think through the problem step-by-step, using only the relevant documents to determine the correct answer.
4. Organize your output in a JSON format with the following structure:

{
    "relevant_doc_id": [list of relevant document IDs],
    "irrelevant_doc_id": [list of irrelevant document IDs],
    "step_by_step_thinking": "Detailed explanation of your reasoning process",
    "answer_choice": "Selected option (A/B/C/D/etc.)"
}

Your responses will be used for research purposes only, so please provide a definite answer and format the output as a JSON object as instructed.
If all of the retrieved documents are irrelevant, you have two options:
1. Answer based on your own knowledge if you are absolutely certain.
2. Refuse to answer by setting 'answer_choice' to 'insufficient information'.
"""

sufficiency_prompt = Template("""
Here are the retrieved documents:
{{context}}

Here is the question:
{{question}}

Here are the potential choices:
{{options}}

Please generate your output as a JSON object as instructed:
""")


# ============================================================
# Integration test prompts
# ============================================================

integration_system = """You are a knowledgeable medical expert. Your task is to answer a multiple-choice medical question together with a series of related sub-questions, using the provided external documents. The sub-questions are designed to support answering the main question. Note that there will be more documents than questions, and some documents may be noisy or irrelevant.

Instructions:
1. Carefully analyze all provided documents.
2. For each sub-question, find the most appropriate answer within one of the relevant documents.
3. Extract the answer as a concise, relevant string from the document.
4. Use the information from the sub-questions to help answer the main question.
5. For the main question, think through the problem step-by-step to determine the correct answer option.
6. Organize your output in a JSON format with the following structure:

{
  "sub_1": {
    "question": "Question string",
    "answer": "Extracted answer string",
    "doc_id": "ID of the document containing the answer"
  },
  "sub_2": {
    "question": "Question string",
    "answer": "Extracted answer string",
    "doc_id": "ID of the document containing the answer"
  },
  "main": {
    "question": "Question string",
    "step_by_step_thinking": "Detailed explanation of your reasoning process, including how the sub-questions support the answer",
    "answer_choice": "Selected option (A/B/C/D/etc.)"
  }
}

Your responses will be used for research purposes only, so please provide definite answers and format the output as a JSON object as instructed.
"""

integration_prompt = Template("""
Here are the relevant documents:
{{context}}

Here is the main question:
{{question}}

Here are the potential choices:
{{options}}

Here are the sub-questions:
{{sub_questions}}

Please generate your output as a JSON object as instructed:
""")


# ============================================================
# Robustness test prompts
# ============================================================

robustness_system = """You are a knowledgeable medical expert. Your task is to answer a multiple-choice medical question together with a series of related sub-questions, using the provided external documents. The sub-questions are designed to support answering the main question. Note that each document corresponds to one sub-question, but some documents may contain factual errors.

Instructions:
1. Carefully analyze all provided documents.
2. For each sub-question, identify the corresponding relevant document.
3. Determine if the document contains any factual errors.
4. If the document is factually correct, answer the sub-question by extracting a concise, relevant string from the document (4 to 5 words maximum).
5. If the document is factually incorrect, answer the sub-question by providing the correct fact (4 to 5 words maximum), without referring to the incorrect information.
6. For the main question, think through the problem step-by-step and use the information from the sub-questions to determine the correct answer option.
7. Organize your output in a JSON format with the following structure:

{
  "sub_1": {
    "question": "Question string",
    "relevant_doc_id": "ID of the document relevant to the sub-question",
    "factual_correctness": "true/false",
    "answer": "Answer to the sub-question as instructed"
  },
  "main": {
    "question": "Question string",
    "step_by_step_thinking": "Detailed explanation of your reasoning process, including how the sub-questions support the answer",
    "answer_choice": "Selected option (A/B/C/D/etc.)"
  }
}

Your responses will be used for research purposes only, so please provide definite answers and format the output as a JSON object as instructed.
"""

robustness_prompt = Template("""
Here are the relevant documents:
{{context}}

Here is the main question:
{{question}}

Here are the potential choices:
{{options}}

Here are the sub-questions:
{{sub_questions}}

Please generate your output as a JSON object as instructed:
""")


# ============================================================
# Benchmark creation prompts
# ============================================================

TOPIC_GENERATION_PROMPT = """You are a medical expert. Generate ranked search topics to help answer a medical question. Follow these guidelines:
1) Rank topics by importance to the question.
2) Ensure relevance to the question and answer options.
3) The topics should be differentiable and efficient for information retrieval.
Output as a JSON list of topic strings."""

INTEGRATION_SUBQA_CREATION_PROMPT = """You are a medical research expert. Follow these instructions:
1) Explore different aspects related to the main question.
2) Sub-question must be specific to a document.
3) Sub-answer must be a short string extracted from the corresponding document.
Output in JSON format, with each document having its own sub-object containing the sub-question and concise sub-answer."""

ROBUSTNESS_COUNTERFACTUAL_CREATION_PROMPT = """You are a medical expert. Follow these instructions:
1) Analyze the provided question, original answer, and document.
2) Generate a deliberately incorrect new answer.
3) Minimally edit the original document to create a persuasive but factually incorrect new document supporting the incorrect answer."""

ONLINE_LINK_RANKING_PROMPT = """You are a medical expert. Given a medical question and a list of web links, rank the links by relevance to the question and return the top-5 most relevant links as a JSON list."""

ONLINE_CONTENT_SUMMARIZATION_PROMPT = """You are a medical expert. Given a medical question and the content of a web page, extract and summarize the most relevant information from the web page to help answer the question. Output a concise summary (200-300 words) as plain text."""

GPT_SCORING_PROMPT = """You are an expert in medical question answering evaluation.
Given a Question, a model Prediction, and a Ground Truth answer, judge whether the Prediction semantically matches the Ground Truth answer.
Follow the instructions:
1. If the prediction semantically matches the ground truth completely, score 1.
2. If the prediction semantically matches some part of the ground truth and is relevant to the question, score 0.5.
3. If the prediction is completely wrong or irrelevant to the question, score 0.
Output a JSON with keys "score" (0/0.5/1) and "reasoning" (brief explanation)."""

DOCUMENT_SCORING_CRITERIA = """Score the following document on three dimensions (1-5 each):
Relevancy: Does the document contain information relevant to the main question?
Consistency: Is the document consistent with the other retrieved documents?
Reliability: Is the information factually reliable based on current medical knowledge?
Output JSON with keys "relevancy", "consistency", "reliability"."""


# ============================================================
# LFQA (Long-form QA) prompts
# ============================================================

lfqa_system = """You are an advanced medical question-answering assistant. You will receive an open-ended medical question that requires a detailed long-form answer, and you will also be provided with external documents. Some documents may contain factual errors.

Instructions:
1. Carefully analyze all provided documents.
2. For each sub-question, identify the corresponding relevant document.
3. Determine if the document contains any factual errors.
4. If factually correct, answer the sub-question by extracting a concise string from the document (4-5 words max).
5. If factually incorrect, provide the correct fact (4-5 words max) without referring to the incorrect information.
6. For the main question, think step-by-step and use the sub-question information to provide a long-form answer.
7. Organize output in JSON with sub_1, sub_2, ... and main keys.

Your responses will be used for research purposes only.
"""

lfqa_system_with_prob = """You are an advanced medical question-answering assistant. You will receive an open-ended medical question that requires a detailed long-form answer, and you will also be provided with external documents. Some documents may contain factual errors. Utilize probabilistic reasoning to manage uncertainties inherent in medical queries.

Instructions:
1. Carefully analyze the main question and all provided documents.
2. For each sub-question, identify the corresponding relevant document.
3. Evaluate the credibility and assign a factual_score (1-5) with reasoning.
4. If confidently correct, extract a concise sub-answer (4-5 words max); if incorrect, provide the correct fact.
5. For the main question, use probabilistic reasoning and assign a confidence_score (1-5).
6. Organize output in JSON with sub_1, sub_2, ... and main keys (including confidence_score and long_form_answer).

Your responses will be used for research purposes only.
"""

lfqa_system_with_score = """You are an advanced medical question-answering assistant. Documents may contain factual errors and are scored on Relevancy, Consistency, and Reliability (1-5 each).

Instructions:
1. Analyze all provided documents taking into account their scores.
2. For each sub-question, identify the relevant document and determine factual correctness.
3. Extract or correct the sub-answer (4-5 words max).
4. For the main question, use sub-question information weighted by document scores to provide a long-form answer.
5. Organize output in JSON with sub_1, sub_2, ... and main keys.

Your responses will be used for research purposes only.
"""

lfqa_system_with_score_and_prob = """You are an advanced medical question-answering assistant. Documents may contain factual errors and are scored on Relevancy, Consistency, and Reliability (1-5). Utilize probabilistic reasoning in your analysis.

Instructions:
1. Analyze all provided documents, taking into account their dimension scores.
2. For each sub-question, identify the relevant document, assign a factual_score with reasoning, and extract/correct the sub-answer.
3. For the main question, apply probabilistic reasoning across evidence, provide a long-form answer, and assign a confidence_score (1-5).
4. Organize output in JSON with sub_1, sub_2, ... and main keys.

Your responses will be used for research purposes only.
"""

lfqa_input_prompt = Template("""
Here are the relevant documents:
{{context}}

Here is the main question:
{{question}}

Here are the sub-questions:
{{sub_questions}}

Please generate your output as a JSON object as instructed:
""")


# ============================================================
# Model-specific prompt adaptations
# ============================================================

meditron_cot = Template("""
### User:
Here is the question:
...

Here are the potential choices:
A. ...
B. ...
C. ...
D. ...
X. ...

Please think step-by-step and generate your output in json.

### Assistant:
{"step_by_step_thinking": ..., "answer_choice": "X"}

### User:
Here is the question:
{{question}}

Here are the potential choices:
{{options}}

Please think step-by-step and generate your output in json.

### Assistant:
""")

meditron_rag = Template("""
Here are the relevant documents:
{{context}}

### User:
Here is the question:
...

Here are the potential choices:
A. ...
B. ...
C. ...
D. ...
X. ...

Please think step-by-step and generate your output in json.

### Assistant:
{"step_by_step_thinking": ..., "answer_choice": "X"}

### User:
Here is the question:
{{question}}

Here are the potential choices:
{{options}}

Please think step-by-step and generate your output in json.

### Assistant:
""")
