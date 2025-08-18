"""
Evaluation Module for ChatGPT Factuality Assessment
=================================================

Comprehensive evaluation framework that provides statistical analysis,
performance evaluation, and comparison capabilities for factuality
evaluation research.

This module coordinates evaluation across all three factuality tasks
with academic-quality metrics and statistical validation.

Author: Michael Ogunjimi
Institution: University of Manchester
Course: MSc AI
"""

# Core evaluation components
from .evaluator import (
    # Evaluation classes
    BaseEvaluator,
    EntailmentEvaluator,
    RankingEvaluator,
    RatingEvaluator,
    EvaluatorFactory,
    EvaluationResult,
)

from .metrics import (
    # Core classes
    MetricResult,
    StatisticalAnalyzer,
    TaskSpecificMetrics,
    # Utility functions
    compute_comprehensive_metrics,
)

__all__ = [
    # Evaluator classes
    "BaseEvaluator",
    "EntailmentEvaluator", 
    "RankingEvaluator",
    "RatingEvaluator",
    "EvaluatorFactory",
    "EvaluationResult",
    # Metrics classes
    "MetricResult",
    "StatisticalAnalyzer",
    "TaskSpecificMetrics",
    # Utility functions
    "compute_comprehensive_metrics",
]

# Module metadata
__version__ = "1.0.0"
__author__ = "Michael Ogunjimi"

# Evaluation capabilities
EVALUATION_CAPABILITIES = {
    "statistical_tests": [
        "pearson_correlation",
        "spearman_correlation",
        "kendall_tau",
        "mcnemar_test",
        "wilcoxon_signed_rank",
        "mann_whitney_u",
    ],
    "performance_metrics": {
        "entailment_inference": [
            "accuracy",
            "precision",
            "recall",
            "f1_score",
            "confusion_matrix",
            "roc_auc",
        ],
        "summary_ranking": [
            "kendall_tau",
            "spearman_rho",
            "ndcg",
            "ranking_accuracy",
            "pairwise_accuracy",
        ],
        "consistency_rating": [
            "pearson_correlation",
            "spearman_correlation",
            "mae",
            "rmse",
            "agreement_within_threshold",
        ],
    },
    "comparison_methods": [
        "baseline_comparison",
        "human_correlation",
        "cross_task_analysis",
        "statistical_significance",
    ],
}


def get_evaluation_info():
    """Get information about evaluation capabilities."""
    return {
        "module_version": __version__,
        "capabilities": EVALUATION_CAPABILITIES,
        "supported_tasks": list(EVALUATION_CAPABILITIES["performance_metrics"].keys()),
        "description": "Comprehensive evaluation framework for factuality assessment",
    }


def quick_evaluate(task_results, task_type="entailment_inference", include_stats=True):
    """
    Quick evaluation function for immediate analysis.

    Args:
        task_results: List of task results to evaluate
        task_type: Type of factuality task
        include_stats: Whether to include statistical analysis

    Returns:
        Dictionary with evaluation metrics
    """
    try:
        # Create evaluator using factory
        evaluator = EvaluatorFactory.create_evaluator(task_type)

        # Compute basic metrics
        if task_type == "entailment_inference":
            from ..tasks.entailment_inference import BinaryClassificationMetrics

            # Extract predictions and true labels
            predictions = [
                r.prediction for r in task_results if hasattr(r, "prediction")
            ]
            true_labels = [
                r.human_label
                for r in task_results
                if hasattr(r, "human_label") and r.human_label is not None
            ]

            if len(predictions) == len(true_labels):
                metrics = BinaryClassificationMetrics.compute_basic_metrics(
                    true_labels, predictions
                )
            else:
                metrics = {"error": "Mismatch between predictions and labels"}

        elif task_type == "summary_ranking":
            from ..tasks.summary_ranking import RankingMetrics

            # Extract rankings
            predictions = [r.ranking for r in task_results if hasattr(r, "ranking")]
            true_rankings = [
                r.human_label
                for r in task_results
                if hasattr(r, "human_label") and r.human_label is not None
            ]

            if len(predictions) == len(true_rankings) and predictions:
                # Compute average metrics across all examples
                all_metrics = []
                for pred, true in zip(predictions, true_rankings):
                    if pred and true:
                        ex_metrics = RankingMetrics.compute_comprehensive_metrics(
                            true, pred
                        )
                        all_metrics.append(ex_metrics)

                if all_metrics:
                    # Average the metrics
                    metrics = {}
                    for key in all_metrics[0].keys():
                        values = [m[key] for m in all_metrics if key in m]
                        metrics[f"avg_{key}"] = (
                            sum(values) / len(values) if values else 0
                        )
                else:
                    metrics = {"error": "No valid ranking pairs"}
            else:
                metrics = {"error": "No rankings available for evaluation"}

        elif task_type == "consistency_rating":
            from ..tasks.consistency_rating import RatingMetrics

            # Extract ratings
            predictions = [r.rating for r in task_results if hasattr(r, "rating")]
            true_ratings = [
                float(r.human_label)
                for r in task_results
                if hasattr(r, "human_label") and r.human_label is not None
            ]

            if len(predictions) == len(true_ratings):
                metrics = RatingMetrics.compute_comprehensive_metrics(
                    true_ratings, predictions
                )
            else:
                metrics = {"error": "Mismatch between predictions and labels"}
        else:
            metrics = {"error": f"Unsupported task type: {task_type}"}

        # Add basic statistics
        basic_stats = {
            "total_results": len(task_results),
            "successful_results": len(
                [r for r in task_results if getattr(r, "success", True)]
            ),
            "task_type": task_type,
        }

        return {
            "basic_statistics": basic_stats,
            "performance_metrics": metrics,
            "evaluation_timestamp": str(time.time()),
        }

    except Exception as e:
        return {
            "error": f"Quick evaluation failed: {e}",
            "task_type": task_type,
            "num_results": len(task_results) if task_results else 0,
        }


def validate_evaluation_setup():
    """Validate that evaluation components are properly set up."""
    validation_results = {
        "status": "success",
        "components": {},
        "warnings": [],
        "errors": [],
    }

    # Test statistical analyzer
    try:
        analyzer = StatisticalAnalyzer()
        validation_results["components"]["statistical_analyzer"] = "success"
    except Exception as e:
        validation_results["components"]["statistical_analyzer"] = f"failed: {e}"
        validation_results["status"] = "error"

    # Test evaluator
    try:
        evaluator = EvaluatorFactory.create_evaluator("entailment_inference")
        validation_results["components"]["factuality_evaluator"] = "success"
    except Exception as e:
        validation_results["components"]["factuality_evaluator"] = f"failed: {e}"
        validation_results["status"] = "error"

    try:
        import scipy.stats
        import sklearn.metrics

        validation_results["components"]["statistical_dependencies"] = "success"
    except ImportError as e:
        validation_results["errors"].append(f"Missing statistical dependencies: {e}")
        validation_results["status"] = "error"

    return validation_results


# Import time to use in quick_evaluate
import time
