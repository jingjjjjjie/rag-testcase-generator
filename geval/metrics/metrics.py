from typing import List
from deepeval.metrics import GEval
from deepeval.metrics.g_eval.utils import Rubric
from deepeval.test_case import LLMTestCaseParams
from geval.custom_qwen_model import CustomQwenModel

# criteria referencing the original definition by RAGAS: https://arxiv.org/pdf/2309.15217

custom_qwen_model = CustomQwenModel()

def get_answer_relevancy_metric(model_type: str = None) -> GEval:
    return GEval(
        name="Answer Relevancy (1-5)",
        criteria=(
            "To penalize cases where the answer (`actual_output`) is incomplete" 
            "or contains redundant information that does not help answer the question (`input`)."
        ),
        evaluation_steps=[
            "Carefully analyze the question (`input`) and identify all information needs that must be addressed.",
            "Analyze the full answer (`actual_output`) and determine which information needs are addressed, missing, or incorrectly handled.",
            "Identify and penalize any Noise: irrelevant, redundant, or distracting content.",
            "Score the answer based on completeness and the absence of Noise."
        ],
        rubric=[
            Rubric(score_range=(1, 1), expected_outcome="Answer is completely irrelevant and does not address the information needs."),
            Rubric(score_range=(2, 2), expected_outcome="Answer addresses very few information needs and includes substantial Noise."),
            Rubric(score_range=(3, 3), expected_outcome="Answer covers some information needs but contains gaps or noticeable Noise."),
            Rubric(score_range=(4, 4), expected_outcome="Answer covers most information needs with minimal Noise; minor omissions are allowed."),
            Rubric(score_range=(5, 5), expected_outcome="Answer fully covers all information needs with no Noise."),
        ],
        evaluation_params=[
            LLMTestCaseParams.INPUT,
            LLMTestCaseParams.ACTUAL_OUTPUT,
        ],
        model=custom_qwen_model if model_type is None else model_type,
        verbose_mode=False,
    )

def get_context_relevance_metric(model_type: str = None) -> GEval:
    return GEval(
        name="Context Relevance (1-5)",
        criteria=(
            "Evaluate the retrieved context (`retrieval_context`) for coverage of all information needs expressed in the question (`input`). "
            "Penalize inclusion of Noise: irrelevant, redundant, distracting, or contradictory content."
        ),
        evaluation_steps=[
            "Identify all information needs in the question (`input`).",
            "Analyze `retrieval_context` to check which information needs are addressed, missing, or contradicted.",
            "Identify any Noise in the context: irrelevant, redundant, distracting, or contradictory content.",
            "Score the context based on the proportion of information needs covered and the amount of Noise."
        ],
        evaluation_params=[
            LLMTestCaseParams.INPUT,
            LLMTestCaseParams.RETRIEVAL_CONTEXT,
        ],
        rubric=[
            Rubric(score_range=(1, 1), expected_outcome="Context addresses very few information needs; heavily dominated by Noise."),
            Rubric(score_range=(2, 2), expected_outcome="Context addresses some information needs but contains substantial Noise."),
            Rubric(score_range=(3, 3), expected_outcome="Context addresses roughly half of the information needs; moderate Noise present."),
            Rubric(score_range=(4, 4), expected_outcome="Context addresses most information needs; minor Noise or small missing details."),
            Rubric(score_range=(5, 5), expected_outcome="Context fully addresses all information needs with no Noise or contradictions.")
        ],
        model=custom_qwen_model
    )

def get_faithfulness_metric(model_type: str = None) -> GEval:
    return GEval(
        name="Faithfulness (1-5)",
        criteria=(
            "Determine if the answer (`actual_output`) is strictly grounded in the given context (`retrieval_context`). "
            "Penalize answers that contain unsupported claims or claims that contradict the context."
        ),
        evaluation_steps = [
            "Identify all factual claims in `actual_output`. Ignore stylistic, explanatory, or summarizing sentences.",
            "For each factual claim, check against `retrieval_context`: mark as Supported (clearly backed by context), Contradicted (conflicts with context), or Unsupported (no evidence in context).",
            "Calculate the percentage of Supported, Unsupported, and Contradicted claims among all factual claims.",
            "Assign score: penalize heavily for Contradicted claims, moderately for Unsupported claims. Full score only if all claims are Supported."
        ],
        evaluation_params=[
            LLMTestCaseParams.ACTUAL_OUTPUT,
            LLMTestCaseParams.RETRIEVAL_CONTEXT,
        ],
        rubric = [
            Rubric(score_range=(1, 1), expected_outcome="Majority of claims are Contradicted or Unsupported; answer is largely unfaithful to context."),
            Rubric(score_range=(2, 2), expected_outcome="Multiple claims are Contradicted or a large proportion is Unsupported; few claims are Supported."),
            Rubric(score_range=(3, 3), expected_outcome="Roughly half of claims are Supported; remaining claims are Unsupported with possible minor Contradictions."),
            Rubric(score_range=(4, 4), expected_outcome="Most claims are Supported; minor Unsupported claims may exist; no Contradictions."),
            Rubric(score_range=(5, 5), expected_outcome="All factual claims are fully Supported by the context; no Unsupported or Contradicted claims.")
        ],
        model=custom_qwen_model,
    )

def get_answer_correctness_metric(model_type: str = None) -> GEval:
    return GEval(
        name="Answer Correctness (1-5)",
        criteria=(
            "Check if the answer (`actual_output`) includes all key facts and essential information from the expected output (`expected_output`). "
            "Partial answers lose points; extra information is fine if it does not contradict or obscure key facts."
        ),
        evaluation_steps=[
            "Compare `actual_output` with `expected_output` to identify all key facts.",
            "Determine which facts are present, partially present, or missing.",
            "Check if the answer conveys the correct meaning, even if phrasing differs.",
            "Score the answer based on completeness, correctness, and clarity of key facts."
        ],
        evaluation_params=[
            LLMTestCaseParams.ACTUAL_OUTPUT,
            LLMTestCaseParams.EXPECTED_OUTPUT,
        ],
        rubric=[
            Rubric(score_range=(1, 1), expected_outcome="Most key facts missing or incorrect; answer is largely wrong."),
            Rubric(score_range=(2, 2), expected_outcome="Few key facts present; major gaps and partially correct."),
            Rubric(score_range=(3, 3), expected_outcome="Some key facts present; answer partially correct but gaps exist."),
            Rubric(score_range=(4, 4), expected_outcome="Most key facts present; answer mostly correct with minor gaps."),
            Rubric(score_range=(5, 5), expected_outcome="All key facts present; answer fully correct and semantically equivalent to expected output.")
        ],
        model=custom_qwen_model if model_type is None else model_type,
        verbose_mode=False,
    )


