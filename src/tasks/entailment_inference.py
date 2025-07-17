"""
Binary Entailment Inference Task for ChatGPT Factuality Evaluation
================================================================

Implementation of binary entailment inference task that determines whether
a summary is factually consistent (ENTAILMENT) or inconsistent (CONTRADICTION)
with the source document.

This is a core task for factuality evaluation, providing binary classification
of summary consistency with comprehensive evaluation metrics.

Author: Michael Ogunjimi
Institution: University of Manchester
Course: MSc AI
"""

from dataclasses import dataclass
from typing import Dict, List, Any

import numpy as np
from sklearn.metrics import (
    accuracy_score,
    precision_score,
    recall_score,
    f1_score,
    confusion_matrix,
)

from .base_task import BaseFactualityTask, TaskExample, TaskResult, TaskConfig
from ..llm_clients.openai_client import APICallResult
from ..utils.logging import get_logger

logger = get_logger(__name__)


@dataclass
class EntailmentResult(TaskResult):
    """
    Specialized result for entailment inference task.

    Extends TaskResult with binary classification specific fields
    and convenience methods for analysis.
    """

    binary_prediction: int = 0  # 0 = CONTRADICTION, 1 = ENTAILMENT
    prediction_label: str = "CONTRADICTION"  # "ENTAILMENT" or "CONTRADICTION"

    def __post_init__(self):
        super().__post_init__()
        # Ensure prediction is in correct format
        self.prediction = self.binary_prediction

    def is_entailment(self) -> bool:
        """Check if prediction is ENTAILMENT."""
        return self.binary_prediction == 1

    def is_contradiction(self) -> bool:
        """Check if prediction is CONTRADICTION."""
        return self.binary_prediction == 0

    def matches_binary_label(self, human_label: int) -> bool:
        """Check if prediction matches human binary label."""
        return self.binary_prediction == human_label

    def get_prediction_confidence(self) -> float:
        """Get confidence in the prediction."""
        return self.confidence if self.confidence is not None else 0.5

    def __sub__(self, other):
        """Handle subtraction operations."""
        if isinstance(other, (int, float)):
            return self.binary_prediction - other
        elif hasattr(other, 'binary_prediction'):
            return self.binary_prediction - other.binary_prediction
        else:
            return NotImplemented

    def __rsub__(self, other):
        """Handle reverse subtraction operations."""
        if isinstance(other, (int, float)):
            return other - self.binary_prediction
        else:
            return NotImplemented

    def __add__(self, other):
        """Handle addition operations."""
        if isinstance(other, (int, float)):
            return self.binary_prediction + other
        elif hasattr(other, 'binary_prediction'):
            return self.binary_prediction + other.binary_prediction
        else:
            return NotImplemented

    def __radd__(self, other):
        """Handle reverse addition operations."""
        if isinstance(other, (int, float)):
            return other + self.binary_prediction
        else:
            return NotImplemented


class BinaryClassificationMetrics:
    """
    Comprehensive metrics for binary classification evaluation.

    Computes standard classification metrics with confidence intervals
    and detailed analysis for academic research.
    """

    @staticmethod
    def compute_basic_metrics(y_true: List[int], y_pred: List[int]) -> Dict[str, float]:
        """
        Compute basic classification metrics.

        Args:
            y_true: True binary labels (0/1)
            y_pred: Predicted binary labels (0/1)

        Returns:
            Dictionary with basic metrics
        """
        if len(y_true) != len(y_pred):
            raise ValueError("y_true and y_pred must have same length")

        if len(y_true) == 0:
            return {"error": "No predictions to evaluate"}

        return {
            "accuracy": accuracy_score(y_true, y_pred),
            "precision": precision_score(y_true, y_pred, zero_division=0),
            "recall": recall_score(y_true, y_pred, zero_division=0),
            "f1_score": f1_score(y_true, y_pred, zero_division=0),
            "precision_0": precision_score(
                y_true, y_pred, pos_label=0, zero_division=0
            ),
            "recall_0": recall_score(y_true, y_pred, pos_label=0, zero_division=0),
            "f1_score_0": f1_score(y_true, y_pred, pos_label=0, zero_division=0),
        }

    @staticmethod
    def compute_confusion_matrix(
        y_true: List[int], y_pred: List[int]
    ) -> Dict[str, Any]:
        """
        Compute confusion matrix with detailed breakdown.

        Args:
            y_true: True binary labels
            y_pred: Predicted binary labels

        Returns:
            Dictionary with confusion matrix analysis
        """
        cm = confusion_matrix(y_true, y_pred, labels=[0, 1])

        # Extract components
        tn, fp, fn, tp = cm.ravel() if cm.size == 4 else (0, 0, 0, 0)

        total = len(y_true)

        return {
            "confusion_matrix": cm.tolist(),
            "true_negatives": int(tn),
            "false_positives": int(fp),
            "false_negatives": int(fn),
            "true_positives": int(tp),
            "total_samples": total,
            "support": {
                "contradiction": int(tn + fn),  # True label 0
                "entailment": int(tp + fp),  # True label 1
            },
        }

    @staticmethod
    def compute_confidence_metrics(
        y_true: List[int], y_pred: List[int], confidences: List[float]
    ) -> Dict[str, float]:
        """
        Compute confidence-based metrics.

        Args:
            y_true: True binary labels
            y_pred: Predicted binary labels
            confidences: Confidence scores for predictions

        Returns:
            Dictionary with confidence metrics
        """
        if len(confidences) != len(y_pred):
            return {"error": "Confidence scores length mismatch"}

        correct_predictions = [1 if yt == yp else 0 for yt, yp in zip(y_true, y_pred)]

        # Correlation between confidence and correctness
        if len(set(correct_predictions)) > 1 and len(set(confidences)) > 1:
            correlation = np.corrcoef(confidences, correct_predictions)[0, 1]
        else:
            correlation = 0.0

        # Average confidence for correct vs incorrect predictions
        correct_confidences = [
            conf
            for conf, correct in zip(confidences, correct_predictions)
            if correct == 1
        ]
        incorrect_confidences = [
            conf
            for conf, correct in zip(confidences, correct_predictions)
            if correct == 0
        ]

        return {
            "average_confidence": np.mean(confidences),
            "confidence_std": np.std(confidences),
            "confidence_correctness_correlation": correlation,
            "avg_confidence_correct": (
                np.mean(correct_confidences) if correct_confidences else 0.0
            ),
            "avg_confidence_incorrect": (
                np.mean(incorrect_confidences) if incorrect_confidences else 0.0
            ),
            "confidence_gap": (
                (np.mean(correct_confidences) - np.mean(incorrect_confidences))
                if correct_confidences and incorrect_confidences
                else 0.0
            ),
        }

    @staticmethod
    def compute_bootstrap_confidence_intervals(
        y_true: List[int],
        y_pred: List[int],
        n_bootstrap: int = 1000,
        confidence_level: float = 0.95,
    ) -> Dict[str, Dict[str, float]]:
        """
        Compute bootstrap confidence intervals for metrics.

        Args:
            y_true: True binary labels
            y_pred: Predicted binary labels
            n_bootstrap: Number of bootstrap samples
            confidence_level: Confidence level for intervals

        Returns:
            Dictionary with confidence intervals for each metric
        """
        if len(y_true) == 0:
            return {}

        n_samples = len(y_true)
        alpha = 1 - confidence_level
        lower_percentile = (alpha / 2) * 100
        upper_percentile = (1 - alpha / 2) * 100

        bootstrap_metrics = {
            "accuracy": [],
            "precision": [],
            "recall": [],
            "f1_score": [],
        }

        for _ in range(n_bootstrap):
            # Bootstrap sample
            indices = np.random.choice(n_samples, size=n_samples, replace=True)
            y_true_boot = [y_true[i] for i in indices]
            y_pred_boot = [y_pred[i] for i in indices]

            # Compute metrics for bootstrap sample
            try:
                metrics = BinaryClassificationMetrics.compute_basic_metrics(
                    y_true_boot, y_pred_boot
                )
                for metric in bootstrap_metrics:
                    if metric in metrics:
                        bootstrap_metrics[metric].append(metrics[metric])
            except:
                continue

        # Compute confidence intervals
        confidence_intervals = {}
        for metric, values in bootstrap_metrics.items():
            if values:
                confidence_intervals[metric] = {
                    "mean": np.mean(values),
                    "std": np.std(values),
                    "ci_lower": np.percentile(values, lower_percentile),
                    "ci_upper": np.percentile(values, upper_percentile),
                    "confidence_level": confidence_level,
                }

        return confidence_intervals


class EntailmentInferenceTask(BaseFactualityTask):
    """
    Binary entailment inference task for factuality evaluation.

    Determines whether a summary is factually consistent (ENTAILMENT)
    or inconsistent (CONTRADICTION) with the source document using ChatGPT.
    """

    def _create_task_config(self) -> TaskConfig:
        """Create configuration specific to entailment inference."""
        task_config_dict = self.config.get("tasks", {}).get("entailment_inference", {})

        return TaskConfig(
            task_type="entailment_inference",
            prompt_type=task_config_dict.get("prompt_type", "zero_shot"),
            model_name=task_config_dict.get("model_name", "gpt-4.1-mini"),
            temperature=task_config_dict.get("temperature", 0.0),
            max_tokens=task_config_dict.get("max_tokens", 50),  # Short response needed
            batch_size=task_config_dict.get("batch_size", 10),
            max_examples=task_config_dict.get("max_examples"),
            include_human_eval=task_config_dict.get("include_human_eval", False),
            save_intermediate=task_config_dict.get("save_intermediate", True),
            cache_responses=task_config_dict.get("cache_responses", True),
            retry_failed=task_config_dict.get("retry_failed", True),
        )

    def _validate_example(self, example: TaskExample) -> bool:
        """
        Validate example for entailment inference task.

        Args:
            example: TaskExample to validate

        Returns:
            True if example is valid for this task
        """
        try:
            # Must have source and at least one summary
            if not example.source or not example.source.strip():
                logger.warning(f"Example {example.example_id}: Empty source")
                return False

            # Get summary for binary task
            summary = example.get_summary_for_binary_task()
            if not summary or not summary.strip():
                logger.warning(f"Example {example.example_id}: Empty summary")
                return False

            # Check reasonable length constraints
            if len(example.source) < 10:
                logger.warning(f"Example {example.example_id}: Source too short")
                return False

            if len(summary) < 5:
                logger.warning(f"Example {example.example_id}: Summary too short")
                return False

            # Validate human label if present
            if example.human_label is not None:
                if not isinstance(
                    example.human_label, int
                ) or example.human_label not in [0, 1]:
                    logger.warning(f"Example {example.example_id}: Invalid human label")
                    return False

            return True

        except Exception as e:
            logger.error(f"Example validation failed for {example.example_id}: {e}")
            return False

    def _process_api_result(
        self, api_result: APICallResult, example: TaskExample
    ) -> TaskResult:
        """
        Process API result into EntailmentResult.

        Args:
            api_result: APICallResult from OpenAI client
            example: Original TaskExample

        Returns:
            EntailmentResult object
        """
        parsed_content = api_result.parsed_content
        raw_response = api_result.raw_response

        # Extract prediction
        binary_prediction = parsed_content.get("prediction", 0)
        prediction_label = parsed_content.get("answer", "CONTRADICTION")

        # Determine confidence (if available in response)
        confidence = api_result.confidence_score or 0.5

        return EntailmentResult(
            example_id=example.example_id,
            task_type=self.task_config.task_type,
            prompt_type=self.task_config.prompt_type,
            prediction=binary_prediction,
            binary_prediction=binary_prediction,
            prediction_label=prediction_label,
            confidence=confidence,
            raw_response=raw_response.content,
            processing_time=raw_response.response_time,
            cost=raw_response.cost,
            tokens_used=raw_response.total_tokens,
            timestamp=raw_response.timestamp,
            success=True,
            human_label=example.human_label,
            metadata={
                "source_length": len(example.source),
                "summary_length": len(example.get_summary_for_binary_task()),
                "dataset_name": example.dataset_name,
                "finish_reason": raw_response.finish_reason,
            },
        )

    def evaluate_predictions(self, results: List[TaskResult]) -> Dict[str, float]:
        """
        Compute comprehensive evaluation metrics for entailment inference.

        Args:
            results: List of EntailmentResult objects

        Returns:
            Dictionary with evaluation metrics
        """
        if not results:
            return {"error": "No results to evaluate"}

        # Filter to successful results
        successful_results = [
            r for r in results if r.success and isinstance(r, EntailmentResult)
        ]

        if not successful_results:
            return {"error": "No successful results to evaluate"}

        # Extract predictions
        predictions = [r.binary_prediction for r in successful_results]

        # Check if we have human labels for evaluation
        human_labels = [
            r.human_label for r in successful_results if r.human_label is not None
        ]
        has_human_labels = len(human_labels) == len(successful_results)

        evaluation_metrics = {
            "total_examples": len(successful_results),
            "has_human_labels": has_human_labels,
        }

        if has_human_labels:
            # Compute classification metrics against human labels
            y_true = [int(r.human_label) for r in successful_results]
            y_pred = predictions

            # Basic metrics
            basic_metrics = BinaryClassificationMetrics.compute_basic_metrics(
                y_true, y_pred
            )
            evaluation_metrics.update(basic_metrics)
            
            # Set primary metric for experiment reporting
            evaluation_metrics["primary_metric"] = basic_metrics.get("accuracy", 0.0)

            # Confusion matrix
            cm_metrics = BinaryClassificationMetrics.compute_confusion_matrix(
                y_true, y_pred
            )
            evaluation_metrics["confusion_matrix_analysis"] = cm_metrics

            # Confidence metrics
            confidences = [r.get_prediction_confidence() for r in successful_results]
            confidence_metrics = BinaryClassificationMetrics.compute_confidence_metrics(
                y_true, y_pred, confidences
            )
            evaluation_metrics["confidence_analysis"] = confidence_metrics

            # Bootstrap confidence intervals
            ci_metrics = (
                BinaryClassificationMetrics.compute_bootstrap_confidence_intervals(
                    y_true, y_pred, n_bootstrap=1000
                )
            )
            evaluation_metrics["confidence_intervals"] = ci_metrics

        else:
            # Without human labels, provide descriptive statistics
            evaluation_metrics.update(
                {
                    "entailment_rate": np.mean(predictions),
                    "contradiction_rate": 1 - np.mean(predictions),
                    "prediction_distribution": {
                        "entailment_count": sum(predictions),
                        "contradiction_count": len(predictions) - sum(predictions),
                    },
                }
            )
            
            # Set primary metric as entailment rate when no human labels
            evaluation_metrics["primary_metric"] = np.mean(predictions)

        # Performance metrics
        processing_times = [r.processing_time for r in successful_results]
        costs = [r.cost for r in successful_results]
        token_usage = [r.tokens_used for r in successful_results]

        evaluation_metrics["performance"] = {
            "avg_processing_time": np.mean(processing_times),
            "std_processing_time": np.std(processing_times),
            "total_cost": sum(costs),
            "avg_cost_per_example": np.mean(costs),
            "total_tokens": sum(token_usage),
            "avg_tokens_per_example": np.mean(token_usage),
        }

        return evaluation_metrics

    def analyze_errors(self, results: List[TaskResult]) -> Dict[str, Any]:
        """
        Analyze prediction errors for insights.

        Args:
            results: List of EntailmentResult objects

        Returns:
            Dictionary with error analysis
        """
        if not results:
            return {"error": "No results to analyze"}

        # Filter to results with human labels
        labeled_results = [
            r for r in results if r.success and r.human_label is not None
        ]

        if not labeled_results:
            return {"error": "No labeled results for error analysis"}

        # Identify correct and incorrect predictions
        correct_predictions = []
        incorrect_predictions = []

        for result in labeled_results:
            if isinstance(result, EntailmentResult):
                if result.matches_binary_label(int(result.human_label)):
                    correct_predictions.append(result)
                else:
                    incorrect_predictions.append(result)

        analysis = {
            "total_with_labels": len(labeled_results),
            "correct_predictions": len(correct_predictions),
            "incorrect_predictions": len(incorrect_predictions),
            "accuracy": len(correct_predictions) / len(labeled_results),
        }

        # Analyze error patterns
        if incorrect_predictions:
            # False positive analysis (predicted ENTAILMENT, actually CONTRADICTION)
            false_positives = [
                r
                for r in incorrect_predictions
                if r.binary_prediction == 1 and r.human_label == 0
            ]

            # False negative analysis (predicted CONTRADICTION, actually ENTAILMENT)
            false_negatives = [
                r
                for r in incorrect_predictions
                if r.binary_prediction == 0 and r.human_label == 1
            ]

            analysis["error_breakdown"] = {
                "false_positives": len(false_positives),
                "false_negatives": len(false_negatives),
                "false_positive_rate": len(false_positives)
                / len(incorrect_predictions),
                "false_negative_rate": len(false_negatives)
                / len(incorrect_predictions),
            }

            # Length analysis for errors
            if incorrect_predictions:
                incorrect_source_lengths = [
                    r.metadata.get("source_length", 0) for r in incorrect_predictions
                ]
                incorrect_summary_lengths = [
                    r.metadata.get("summary_length", 0) for r in incorrect_predictions
                ]

                analysis["error_characteristics"] = {
                    "avg_source_length_errors": np.mean(incorrect_source_lengths),
                    "avg_summary_length_errors": np.mean(incorrect_summary_lengths),
                    "source_length_std_errors": np.std(incorrect_source_lengths),
                    "summary_length_std_errors": np.std(incorrect_summary_lengths),
                }

        return analysis

    def get_task_specific_insights(self, results: List[TaskResult]) -> Dict[str, Any]:
        """
        Get insights specific to entailment inference task.

        Args:
            results: List of EntailmentResult objects

        Returns:
            Dictionary with task-specific insights
        """
        successful_results = [
            r for r in results if r.success and isinstance(r, EntailmentResult)
        ]

        if not successful_results:
            return {"error": "No successful results for analysis"}

        insights = {
            "prediction_patterns": {
                "entailment_predictions": sum(
                    1 for r in successful_results if r.is_entailment()
                ),
                "contradiction_predictions": sum(
                    1 for r in successful_results if r.is_contradiction()
                ),
                "entailment_rate": np.mean(
                    [r.binary_prediction for r in successful_results]
                ),
            },
            "confidence_patterns": {
                "avg_confidence": np.mean(
                    [r.get_prediction_confidence() for r in successful_results]
                ),
                "confidence_std": np.std(
                    [r.get_prediction_confidence() for r in successful_results]
                ),
                "high_confidence_predictions": sum(
                    1 for r in successful_results if r.get_prediction_confidence() > 0.8
                ),
                "low_confidence_predictions": sum(
                    1 for r in successful_results if r.get_prediction_confidence() < 0.3
                ),
            },
        }

        # Text length impact analysis
        source_lengths = [
            r.metadata.get("source_length", 0) for r in successful_results
        ]
        summary_lengths = [
            r.metadata.get("summary_length", 0) for r in successful_results
        ]
        predictions = [r.binary_prediction for r in successful_results]

        # Correlation between length and predictions
        if len(set(source_lengths)) > 1 and len(set(predictions)) > 1:
            source_pred_corr = np.corrcoef(source_lengths, predictions)[0, 1]
        else:
            source_pred_corr = 0.0

        if len(set(summary_lengths)) > 1 and len(set(predictions)) > 1:
            summary_pred_corr = np.corrcoef(summary_lengths, predictions)[0, 1]
        else:
            summary_pred_corr = 0.0

        insights["length_analysis"] = {
            "avg_source_length": np.mean(source_lengths),
            "avg_summary_length": np.mean(summary_lengths),
            "source_length_prediction_correlation": source_pred_corr,
            "summary_length_prediction_correlation": summary_pred_corr,
        }

        return insights
