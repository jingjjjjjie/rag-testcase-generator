from typing import List

from deepeval.metrics import GEval
from deepeval.metrics.g_eval.utils import Rubric
from deepeval.test_case import LLMTestCaseParams
from geval.custom_qwen_model import CustomQwenModel

custom_qwen_model = CustomQwenModel()


# Calibrated acceptance thresholds – chosen from manual spot-checking so only
# high-quality answers pass after the instructions became deterministic.
# These lenient thresholds are optimized for local Ollama models.
# ANSWER_RELEVANCY_THRESHOLD = 0.55
# FAITHFULNESS_THRESHOLD = 0.50
# CONTEXT_RELEVANCE_THRESHOLD = 0.30
# ANSWER_CORRECTNESS_THRESHOLD = 0.40

# NEGATIVE_REASON_KEYWORDS: List[str] = [
#     "irrelevant",
#     "not relevant",
#     "contradiction",
#     "contradict",
#     "missing",
#     "does not",
#     "fails",
#     "hallucination",
#     "incorrect",
#     "wrong",
# ]


def get_answer_relevancy_metric_f(model_type: str = None) -> GEval:
    """
    GEval-based answer relevancy metric.

    Ensures the model judges whether actual_output fully addresses input while
    using domain-appropriate information from retrieval context.
    """
    return GEval(
        name="answer_relevancy",
        criteria=(
            "Determine if `actual_output` directly answers `input` using information "
            "from `retrieval_context` that is in-scope and domain-appropriate for the question. "
            "Penalize answers that use generic or out-of-domain information when "
            "specific company/domain procedures should be followed."
        ),
        evaluation_steps=[
            "Read `input` and restate its core intent in one sentence.",
            "Review `retrieval_context` to identify the domain and scope (e.g., company-specific procedures vs generic information).",
            "Review `actual_output` and note each claim or instruction.",
            "Mark a claim relevant only if it answers the question using domain-appropriate information aligned with the retrieval context scope.",
            "Downgrade claims that use generic or out-of-domain information when specific procedures exist in the retrieval context.",
            "Downgrade claims that introduce unrelated topics, avoid answering, or contradict the question's scope.",
            "Score higher when every part of `actual_output` is useful, on-topic, and uses appropriate domain-specific information; "
            "score very low when the response fails to answer, rambles, or provides generic answers when specific ones are needed."
        ],
        evaluation_params=[
            LLMTestCaseParams.INPUT,
            LLMTestCaseParams.ACTUAL_OUTPUT,
        ],
        rubric=[
            Rubric(
                score_range=(9, 10),
                expected_outcome="Direct, complete answer that stays on-topic and utilises retrieval context appropriately."
            ),
            Rubric(
                score_range=(7, 8),
                expected_outcome="Answers the question but includes minor digressions or omits a small detail."
            ),
            Rubric(
                score_range=(5, 6),
                expected_outcome="Partially answers the question or mixes relevant and irrelevant content."
            ),
            Rubric(
                score_range=(0, 4),
                expected_outcome="Fails to answer, contradicts the request, or is largely irrelevant."
            ),
        ],
        model=custom_qwen_model,
        #threshold=ANSWER_RELEVANCY_THRESHOLD,
    )


def get_faithfulness_metric_f(model_type: str = None) -> GEval:
    """
    GEval-based faithfulness metric.

    Judges whether every claim in actual_output is grounded in the retrieval context.
    """
    return GEval(
        name="faithfulness",
        criteria=(
            "Evaluate if every factual statement in `actual_output` is supported "
            "by `retrieval_context`. Penalise hallucinated or contradictory claims."
        ),
        evaluation_steps=[
            "List each factual statement present in `actual_output`.",
            "For every statement, search `retrieval_context` for explicit support.",
            "Flag a statement as unsupported when no context snippet confirms it.",
            "Flag a statement as conflicting when context states the opposite.",
            "Score 1.0 only when all statements are supported and none contradict context; "
            "score 0 when the response invents facts or opposes the context."
        ],
        evaluation_params=[
            LLMTestCaseParams.ACTUAL_OUTPUT,
            LLMTestCaseParams.RETRIEVAL_CONTEXT,
        ],
        rubric=[
            Rubric(
                score_range=(9, 10),
                expected_outcome="Every claim is explicitly supported by the retrieval context."
            ),
            Rubric(
                score_range=(7, 8),
                expected_outcome="Most claims are grounded; minor paraphrases lack exact wording but align with context."
            ),
            Rubric(
                score_range=(4, 6),
                expected_outcome="Mix of supported and unsupported claims or ambiguous grounding."
            ),
            Rubric(
                score_range=(0, 3),
                expected_outcome="Predominantly hallucinated, unsupported, or contradictory content."
            ),
        ],
        model=custom_qwen_model,
        #threshold=FAITHFULNESS_THRESHOLD,
    )


def get_context_relevance_metric_f(model_type: str = None) -> GEval:
    """
    GEval-based context relevance metric.

    Checks whether retrieval_context items are genuinely useful for answering the input.
    """
    return GEval(
        name="contextual_relevancy",
        criteria=(
            "Assess if each entry in `retrieval_context` helps answer `input`. "
            "Reward focused, on-topic retrieval and penalise noisy or irrelevant context."
        ),
        evaluation_steps=[
            "Summarise the information need expressed in `input`.",
            "Inspect each element of `retrieval_context` for overlap with that need.",
            "Count how many elements provide facts, instructions, or terminology directly useful for answering.",
            "Penalise context that distracts, contradicts, or omits key facts the question requires.",
            "Score based on the proportion and quality of relevant context elements."
        ],
        evaluation_params=[
            LLMTestCaseParams.INPUT,
            LLMTestCaseParams.RETRIEVAL_CONTEXT,
        ],
        rubric=[
            Rubric(
                score_range=(9, 10),
                expected_outcome="Nearly all context elements directly support the question."
            ),
            Rubric(
                score_range=(7, 8),
                expected_outcome="More than half the context is relevant with only minor noise."
            ),
            Rubric(
                score_range=(4, 6),
                expected_outcome="Mixed relevance; useful details exist but are buried among distractors."
            ),
            Rubric(
                score_range=(0, 3),
                expected_outcome="Context is mostly irrelevant, contradictory, or missing key facts."
            ),
        ],
        model=custom_qwen_model,
        #threshold=CONTEXT_RELEVANCE_THRESHOLD,
    )


def get_answer_correctness_metric_f(model_type: str = None) -> GEval:
    """
    GEval-based answer correctness metric for ground-truth evaluations.

    Evaluates semantic equivalence rather than verbatim matching. Rewards answers that
    convey the same essential information and actionable guidance, regardless of exact wording.
    """
    return GEval(
        name="answer_correctness",
        criteria=(
            "Compare `actual_output` with `expected_output` for semantic equivalence. "
            "Judge whether the core meaning and actionable information are preserved. "
            "Accept paraphrasing, different word order, and alternative phrasing as FULLY CORRECT "
            "as long as the essential facts and instructions are present and unambiguous. "
            "Only penalize for missing critical information or contradictions, NOT for different wording."
        ),
        evaluation_steps=[
            "Extract the essential facts and actionable instructions from `expected_output`.",
            "Identify which facts are critical (must have) vs nice-to-have (optional details).",
            "Check if `actual_output` conveys the same essential information with equivalent meaning.",
            "Accept ANY phrasing that preserves the core meaning - examples of EQUIVALENT phrasings:",
            "  • 'Do not make payment' = 'You don't need to pay' = 'No payment required' = SAME MEANING",
            "  • 'Visit nearest clinic' = 'Go to the closest clinic' = 'Seek care at nearby clinic' = SAME MEANING",
            "  • Different word order with same facts = ACCEPTABLE",
            "  • Additional helpful context that doesn't contradict = ACCEPTABLE",
            "Count how many essential facts are present versus missing.",
            "Score generously based on completeness of essential information, NOT wording similarity:",
            "  • 90-100% of essential facts present with correct meaning → score 0.8-1.0",
            "  • 70-89% of essential facts present → score 0.6-0.8",
            "  • 50-69% of essential facts present → score 0.4-0.6",
            "  • Less than 50% of essential facts → score 0.0-0.4",
            "Penalize ONLY for: (1) missing critical information, (2) factual contradictions, (3) misleading guidance."
        ],
        evaluation_params=[
            LLMTestCaseParams.ACTUAL_OUTPUT,
            LLMTestCaseParams.EXPECTED_OUTPUT,
        ],
        rubric=[
            Rubric(
                score_range=(8, 10),
                expected_outcome="All essential facts present with semantically equivalent meaning. Answer is complete and actionable, regardless of exact wording or phrasing style."
            ),
            Rubric(
                score_range=(6, 7),
                expected_outcome="Most essential facts present (70-90% complete). Minor details may be missing but answer conveys core guidance and is actionable."
            ),
            Rubric(
                score_range=(4, 5),
                expected_outcome="Core idea conveyed (50-70% of essential facts present). Multiple important specifics are missing but main concept is recognizable."
            ),
            Rubric(
                score_range=(0, 3),
                expected_outcome="Essential facts missing or contradicted (less than 50% present). Answer is not actionable or provides wrong guidance."
            ),
        ],
        model=custom_qwen_model,
        #threshold=ANSWER_CORRECTNESS_THRESHOLD,
    )
