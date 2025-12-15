import json
from datetime import datetime
from pathlib import Path
from typing import List, Dict, Tuple
from concurrent.futures import ThreadPoolExecutor
from deepeval.metrics import GEval
from deepeval.test_case import LLMTestCase

from src.utils.file_utils import load_json
from geval.metrics.metrics import *
from geval.metrics.metrics_faithlin import *

LOGS_DIR = Path(__file__).parent / "logs"
LOGS_DIR.mkdir(exist_ok=True)


def load_test_cases_from_json(json_path: str) -> List[LLMTestCase]:
    """Load LLMTestCase objects from JSON file."""
    data = load_json(json_path)
    return [LLMTestCase(**item) for item in data]

def extract_metric_config(metric: GEval) -> Dict:
    """Extract configuration from GEval metric."""
    config = {
        "name": metric.name,
        "criteria": metric.criteria,
        "evaluation_steps": metric.evaluation_steps,
        "rubric": []
    }

    if hasattr(metric, 'rubric') and metric.rubric:
        for rubric_item in metric.rubric:
            config["rubric"].append({
                "score_range": rubric_item.score_range,
                "expected_outcome": rubric_item.expected_outcome
            })

    return config

def run_single_evaluation(metric: GEval, test_case: LLMTestCase) -> Dict:
    """Run a single metric evaluation."""
    metric.measure(test_case)
    return {"score": metric.score, "reason": metric.reason}

def consistency_check(test_case: LLMTestCase, metric: GEval,
                     num_checks: int = 20, max_workers: int = 10) -> Dict:
    """Run consistency check on a test case with a metric."""
    with ThreadPoolExecutor(max_workers=max_workers) as executor:
        futures = [executor.submit(run_single_evaluation, metric, test_case)
                   for _ in range(num_checks)]
        results = [future.result() for future in futures]

    scores = [r["score"] for r in results]

    return {
        "scores": scores,
        "score_distribution": {score: scores.count(score) for score in set(scores)},
        "evaluations": results
    }

def save_all_logs(all_results: Dict, dataset_name: str) -> Tuple[Path, Path]:
    """Save all metrics evaluation logs to files."""
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")

    # Save configs and score distributions for all metrics
    log1_path = LOGS_DIR / f"{timestamp}_{dataset_name}_config.json"
    with open(log1_path, 'w') as f:
        json.dump(all_results["configs"], f, indent=2)

    # Save evaluations for all metrics
    log2_path = LOGS_DIR / f"{timestamp}_{dataset_name}_evaluations.json"
    with open(log2_path, 'w') as f:
        json.dump(all_results["evaluations"], f, indent=2)

    return log1_path, log2_path


def run_metrics_evaluation(metrics: List[Tuple[str, GEval]],
                          test_cases: List[LLMTestCase],
                          dataset_name: str = "results",
                          num_checks: int = 20,
                          max_workers: int = 10):
    """Run evaluation for multiple metrics on test cases."""
    all_results = {
        "configs": {},
        "evaluations": {}
    }

    for metric_name, metric in metrics:
        print(f"\n{'#'*80}")
        print(f"# Running metric: {metric_name}")
        print(f"{'#'*80}")

        metric_config = extract_metric_config(metric)
        all_score_distributions = {}
        all_evaluations = {}

        for idx, testcase in enumerate(test_cases):
            test_name = f"test_case_{idx:04d}"
            print(f"\n{'='*80}")
            print(f"Testing: {test_name}")
            print(f"{'='*80}")

            result = consistency_check(testcase, metric, num_checks, max_workers)

            all_score_distributions[test_name] = result["score_distribution"]
            all_evaluations[test_name] = result["evaluations"]

        # Store results for this metric
        all_results["configs"][metric_name] = {
            "metric_config": metric_config,
            "score_distributions": all_score_distributions
        }
        all_results["evaluations"][metric_name] = all_evaluations

    # Save all results to single files
    log1_path, log2_path = save_all_logs(all_results, dataset_name)

    print(f"\n{'='*80}")
    print(f"All logs saved:")
    print(f"  Config: {log1_path}")
    print(f"  Evaluations: {log2_path}")
    print(f"{'='*80}")


if __name__ == "__main__":
    metrics = [
        ("answer_relevancy", get_answer_relevancy_metric()),
        ("faithfulness", get_faithfulness_metric()),
        ("context_relevance", get_context_relevance_metric())
    ]

    test_cases = load_test_cases_from_json('geval/data/[ForMetricsTuning]amnesty_qa_test_cases.json')
    run_metrics_evaluation(metrics, test_cases, dataset_name="amnesty_qa", num_checks=20, max_workers=10)