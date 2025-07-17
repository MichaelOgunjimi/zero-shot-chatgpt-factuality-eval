"""
Main Evaluation Engine for ChatGPT Factuality Assessment
========================================================

This module provides the core evaluation framework for assessing ChatGPT's
performance on factuality evaluation tasks. It integrates with the data module
and provides comprehensive evaluation capabilities suitable for academic research.

The evaluator supports all three core tasks: entailment inference, summary ranking,
and consistency rating, with statistical analysis and human correlation assessment.

Author: Michael Ogunjimi
Institution: University of Manchester
Course: MSc AI
"""

import json
import logging
import time
from abc import ABC, abstractmethod
from dataclasses import dataclass, asdict
from datetime import datetime
from pathlib import Path
from typing import Dict, List, Optional, Union, Any, Tuple

import numpy as np
from scipy import stats
from scipy.stats import pearsonr, spearmanr

from ..data.loaders import FactualityExample, quick_load_dataset
from ..data.preprocessors import ProcessedExample, preprocess_for_task

# Configure logging
logger = logging.getLogger(__name__)


@dataclass
class EvaluationResult:
    """
    Container for evaluation results with comprehensive metadata.

    This class standardizes evaluation outputs across all tasks and provides
    rich information suitable for academic analysis and reporting.
    """

    task_type: str
    dataset_name: str
    model_name: str
    prompt_type: str
    evaluation_id: str

    # Core metrics
    performance_metrics: Dict[str, float]
    correlation_metrics: Optional[Dict[str, float]] = None

    # Detailed results
    example_results: Optional[List[Dict[str, Any]]] = None
    failed_examples: Optional[List[Dict[str, Any]]] = None

    # Metadata
    evaluation_metadata: Optional[Dict[str, Any]] = None
    evaluation_timestamp: Optional[str] = None
    processing_time: Optional[float] = None

    # Statistical analysis
    statistical_tests: Optional[Dict[str, Any]] = None
    confidence_intervals: Optional[Dict[str, Tuple[float, float]]] = None

    def __post_init__(self):
        """Post-initialization setup."""
        if self.evaluation_timestamp is None:
            self.evaluation_timestamp = datetime.now().isoformat()

        if self.evaluation_metadata is None:
            self.evaluation_metadata = {}

    def to_dict(self) -> Dict[str, Any]:
        """Convert to dictionary for serialization."""
        return asdict(self)

    def save_to_file(self, file_path: Union[str, Path]) -> None:
        """
        Save evaluation results to JSON file.

        Args:
            file_path: Path to save results
        """
        file_path = Path(file_path)
        file_path.parent.mkdir(parents=True, exist_ok=True)

        with open(file_path, "w", encoding="utf-8") as f:
            json.dump(self.to_dict(), f, indent=2, ensure_ascii=False)

        logger.info(f"Evaluation results saved to {file_path}")

    def get_summary_report(self) -> str:
        """
        Generate a human-readable summary report.

        Returns:
            Formatted summary string
        """
        report_lines = [
            f"Evaluation Summary Report",
            f"=" * 50,
            f"Task: {self.task_type}",
            f"Dataset: {self.dataset_name}",
            f"Model: {self.model_name}",
            f"Prompt Type: {self.prompt_type}",
            f"Evaluation ID: {self.evaluation_id}",
            f"Timestamp: {self.evaluation_timestamp}",
            f"",
            f"Performance Metrics:",
            f"-" * 20,
        ]

        for metric, value in self.performance_metrics.items():
            if isinstance(value, float):
                report_lines.append(f"{metric}: {value:.4f}")
            else:
                report_lines.append(f"{metric}: {value}")

        if self.correlation_metrics:
            report_lines.extend([f"", f"Correlation Metrics:", f"-" * 20])
            for metric, value in self.correlation_metrics.items():
                if isinstance(value, float):
                    report_lines.append(f"{metric}: {value:.4f}")
                else:
                    report_lines.append(f"{metric}: {value}")

        if self.processing_time:
            report_lines.extend(
                [f"", f"Processing Time: {self.processing_time:.2f} seconds"]
            )

        return "\n".join(report_lines)


class BaseEvaluator(ABC):
    """
    Abstract base class for task-specific evaluators.

    Provides common functionality for evaluation including data loading,
    preprocessing, metric calculation, and statistical analysis.
    """

    def __init__(
        self,
        data_dir: Union[str, Path] = "data",
        results_dir: Union[str, Path] = "results",
        cache_enabled: bool = True,
        validate_inputs: bool = True,
    ):
        """
        Initialize evaluator.

        Args:
            data_dir: Directory containing datasets
            results_dir: Directory for saving results
            cache_enabled: Whether to enable caching
            validate_inputs: Whether to validate input data
        """
        self.data_dir = Path(data_dir)
        self.results_dir = Path(results_dir)
        self.cache_enabled = cache_enabled
        self.validate_inputs = validate_inputs

        # Create results directory
        self.results_dir.mkdir(exist_ok=True)

        self.logger = logging.getLogger(f"{__name__}.{self.__class__.__name__}")

    @abstractmethod
    def task_type(self) -> str:
        """Return the task type this evaluator handles."""
        pass

    @abstractmethod
    def compute_performance_metrics(
        self, predictions: List[Any], ground_truth: List[Any]
    ) -> Dict[str, float]:
        """Compute task-specific performance metrics."""
        pass

    @abstractmethod
    def validate_predictions(self, predictions: List[Any]) -> Tuple[bool, List[str]]:
        """Validate prediction format for the task."""
        pass

    def load_dataset(
        self, dataset_name: str, split: str = "test", max_examples: Optional[int] = None
    ) -> List[FactualityExample]:
        """
        Load dataset for evaluation.

        Args:
            dataset_name: Name of dataset to load
            split: Dataset split to use
            max_examples: Maximum number of examples to load

        Returns:
            List of FactualityExample objects
        """
        self.logger.info(f"Loading {dataset_name} dataset (split: {split})")

        try:
            examples = quick_load_dataset(
                dataset_name=dataset_name,
                split=split,
                max_examples=max_examples,
                data_dir=self.data_dir,
                use_cache=self.cache_enabled,
            )

            self.logger.info(f"Loaded {len(examples)} examples")
            return examples

        except Exception as e:
            self.logger.error(f"Failed to load dataset {dataset_name}: {e}")
            raise

    def preprocess_examples(
        self, examples: List[FactualityExample], **preprocessor_kwargs
    ) -> List[ProcessedExample]:
        """
        Preprocess examples for the specific task.

        Args:
            examples: List of FactualityExample objects
            **preprocessor_kwargs: Arguments passed to preprocessor

        Returns:
            List of ProcessedExample objects
        """
        self.logger.info(
            f"Preprocessing {len(examples)} examples for {self.task_type()}"
        )

        try:
            processed = preprocess_for_task(
                examples=examples, task_type=self.task_type(), **preprocessor_kwargs
            )

            self.logger.info(f"Preprocessed {len(processed)} examples")
            return processed

        except Exception as e:
            self.logger.error(f"Failed to preprocess examples: {e}")
            raise

    def compute_correlation_metrics(
        self,
        predictions: List[Union[float, int]],
        ground_truth: List[Union[float, int]],
    ) -> Dict[str, float]:
        """
        Compute correlation metrics between predictions and ground truth.

        Args:
            predictions: Model predictions
            ground_truth: Human annotations

        Returns:
            Dictionary containing correlation metrics
        """
        if len(predictions) != len(ground_truth):
            raise ValueError("Predictions and ground truth must have same length")

        # Filter out None values
        valid_pairs = [
            (p, g)
            for p, g in zip(predictions, ground_truth)
            if p is not None and g is not None
        ]

        if len(valid_pairs) < 2:
            self.logger.warning("Not enough valid pairs for correlation analysis")
            return {"error": "insufficient_data"}

        preds, truth = zip(*valid_pairs)

        # Convert to numeric arrays
        try:
            preds = np.array(preds, dtype=float)
            truth = np.array(truth, dtype=float)
        except ValueError as e:
            self.logger.error(f"Failed to convert to numeric arrays: {e}")
            return {"error": "conversion_failed"}

        metrics = {}

        try:
            # Pearson correlation
            pearson_r, pearson_p = pearsonr(preds, truth)
            metrics["pearson_r"] = pearson_r
            metrics["pearson_p_value"] = pearson_p

            # Spearman correlation
            spearman_r, spearman_p = spearmanr(preds, truth)
            metrics["spearman_r"] = spearman_r
            metrics["spearman_p_value"] = spearman_p

            # Additional metrics
            metrics["n_valid_pairs"] = len(valid_pairs)
            metrics["n_total_pairs"] = len(predictions)
            metrics["coverage"] = len(valid_pairs) / len(predictions)

        except Exception as e:
            self.logger.error(f"Failed to compute correlations: {e}")
            metrics["error"] = str(e)

        return metrics

    def compute_statistical_tests(
        self,
        predictions: List[Any],
        ground_truth: List[Any],
        baseline_predictions: Optional[List[Any]] = None,
    ) -> Dict[str, Any]:
        """
        Compute statistical significance tests.

        Args:
            predictions: Model predictions
            ground_truth: Ground truth labels
            baseline_predictions: Optional baseline predictions for comparison

        Returns:
            Dictionary containing statistical test results
        """
        tests = {}

        # Basic descriptive statistics
        if all(isinstance(p, (int, float)) for p in predictions if p is not None):
            valid_preds = [p for p in predictions if p is not None]
            tests["descriptive"] = {
                "mean": np.mean(valid_preds),
                "std": np.std(valid_preds),
                "min": np.min(valid_preds),
                "max": np.max(valid_preds),
                "median": np.median(valid_preds),
                "n_valid": len(valid_preds),
            }

        # Normality test (for continuous predictions)
        if len(predictions) >= 8 and all(
            isinstance(p, (int, float)) for p in predictions if p is not None
        ):
            try:
                valid_preds = [p for p in predictions if p is not None]
                shapiro_stat, shapiro_p = stats.shapiro(valid_preds)
                tests["normality"] = {
                    "shapiro_stat": shapiro_stat,
                    "shapiro_p_value": shapiro_p,
                    "is_normal": shapiro_p > 0.05,
                }
            except Exception as e:
                tests["normality"] = {"error": str(e)}

        # Comparison with baseline if provided
        if baseline_predictions is not None:
            try:
                tests["baseline_comparison"] = self._compare_with_baseline(
                    predictions, baseline_predictions, ground_truth
                )
            except Exception as e:
                tests["baseline_comparison"] = {"error": str(e)}

        return tests

    def _compare_with_baseline(
        self,
        predictions: List[Any],
        baseline_predictions: List[Any],
        ground_truth: List[Any],
    ) -> Dict[str, Any]:
        """Compare model predictions with baseline."""
        # This is a placeholder for baseline comparison
        # In practice, this would implement specific statistical tests
        # like McNemar's test for classification or paired t-test for regression

        comparison = {
            "n_predictions": len(predictions),
            "n_baseline": len(baseline_predictions),
            "n_ground_truth": len(ground_truth),
            "comparison_type": "placeholder",
        }

        return comparison

    def evaluate(
        self,
        predictions: List[Any],
        examples: List[ProcessedExample],
        model_name: str = "chatgpt",
        prompt_type: str = "zero_shot",
        evaluation_id: Optional[str] = None,
        baseline_predictions: Optional[List[Any]] = None,
    ) -> EvaluationResult:
        """
        Perform comprehensive evaluation.

        Args:
            predictions: Model predictions
            examples: Processed examples
            model_name: Name of the model being evaluated
            prompt_type: Type of prompt used
            evaluation_id: Unique identifier for this evaluation
            baseline_predictions: Optional baseline predictions for comparison

        Returns:
            EvaluationResult object
        """
        start_time = time.time()

        if evaluation_id is None:
            evaluation_id = f"{self.task_type()}_{model_name}_{int(time.time())}"

        self.logger.info(f"Starting evaluation: {evaluation_id}")

        # Validate inputs
        if self.validate_inputs:
            is_valid, errors = self.validate_predictions(predictions)
            if not is_valid:
                raise ValueError(f"Invalid predictions: {errors}")

        # Extract ground truth labels
        ground_truth = []
        dataset_name = "unknown"

        for example in examples:
            if hasattr(example, "target_label"):
                ground_truth.append(example.target_label)
            else:
                ground_truth.append(None)

            if hasattr(example, "original_example"):
                dataset_name = example.original_example.dataset_name

        # Compute performance metrics
        try:
            performance_metrics = self.compute_performance_metrics(
                predictions, ground_truth
            )
        except Exception as e:
            self.logger.error(f"Failed to compute performance metrics: {e}")
            performance_metrics = {"error": str(e)}

        # Compute correlation metrics if applicable
        correlation_metrics = None
        if ground_truth and any(gt is not None for gt in ground_truth):
            try:
                # Only compute correlations for numeric data
                if all(
                    isinstance(p, (int, float)) for p in predictions if p is not None
                ) and all(
                    isinstance(gt, (int, float))
                    for gt in ground_truth
                    if gt is not None
                ):
                    correlation_metrics = self.compute_correlation_metrics(
                        predictions, ground_truth
                    )
            except Exception as e:
                self.logger.warning(f"Failed to compute correlation metrics: {e}")
                correlation_metrics = {"error": str(e)}

        # Compute statistical tests
        try:
            statistical_tests = self.compute_statistical_tests(
                predictions, ground_truth, baseline_predictions
            )
        except Exception as e:
            self.logger.warning(f"Failed to compute statistical tests: {e}")
            statistical_tests = {"error": str(e)}

        processing_time = time.time() - start_time

        # Create evaluation result
        result = EvaluationResult(
            task_type=self.task_type(),
            dataset_name=dataset_name,
            model_name=model_name,
            prompt_type=prompt_type,
            evaluation_id=evaluation_id,
            performance_metrics=performance_metrics,
            correlation_metrics=correlation_metrics,
            statistical_tests=statistical_tests,
            processing_time=processing_time,
            evaluation_metadata={
                "n_predictions": len(predictions),
                "n_examples": len(examples),
                "n_ground_truth": len([gt for gt in ground_truth if gt is not None]),
                "data_dir": str(self.data_dir),
                "results_dir": str(self.results_dir),
                "cache_enabled": self.cache_enabled,
                "validate_inputs": self.validate_inputs,
            },
        )

        self.logger.info(
            f"Evaluation completed: {evaluation_id} "
            f"(processing time: {processing_time:.2f}s)"
        )

        return result


class EntailmentEvaluator(BaseEvaluator):
    """Evaluator for entailment inference task."""

    def task_type(self) -> str:
        return "entailment_inference"

    def compute_performance_metrics(
        self, predictions: List[str], ground_truth: List[str]
    ) -> Dict[str, float]:
        """Compute classification metrics for entailment task."""
        # Filter valid pairs
        valid_pairs = [
            (p, g)
            for p, g in zip(predictions, ground_truth)
            if p is not None and g is not None
        ]

        if not valid_pairs:
            return {"error": "no_valid_pairs"}

        preds, truth = zip(*valid_pairs)

        # Convert to binary labels for computation
        pred_binary = [1 if p == "ENTAILMENT" else 0 for p in preds]
        truth_binary = [1 if g == "ENTAILMENT" else 0 for g in truth]

        # Compute confusion matrix components
        tp = sum(1 for p, t in zip(pred_binary, truth_binary) if p == 1 and t == 1)
        tn = sum(1 for p, t in zip(pred_binary, truth_binary) if p == 0 and t == 0)
        fp = sum(1 for p, t in zip(pred_binary, truth_binary) if p == 1 and t == 0)
        fn = sum(1 for p, t in zip(pred_binary, truth_binary) if p == 0 and t == 1)

        # Compute metrics
        accuracy = (tp + tn) / len(valid_pairs) if valid_pairs else 0
        precision = tp / (tp + fp) if (tp + fp) > 0 else 0
        recall = tp / (tp + fn) if (tp + fn) > 0 else 0
        f1 = (
            2 * (precision * recall) / (precision + recall)
            if (precision + recall) > 0
            else 0
        )

        return {
            "accuracy": accuracy,
            "precision": precision,
            "recall": recall,
            "f1_score": f1,
            "true_positives": tp,
            "true_negatives": tn,
            "false_positives": fp,
            "false_negatives": fn,
            "n_valid_pairs": len(valid_pairs),
            "entailment_predictions": sum(pred_binary),
            "entailment_ground_truth": sum(truth_binary),
        }

    def validate_predictions(self, predictions: List[str]) -> Tuple[bool, List[str]]:
        """Validate entailment predictions."""
        errors = []

        valid_labels = {"ENTAILMENT", "CONTRADICTION"}

        for i, pred in enumerate(predictions):
            if pred is not None and pred not in valid_labels:
                errors.append(f"Invalid prediction at index {i}: {pred}")

        return len(errors) == 0, errors

    def evaluate_predictions(self, predictions: List[Any], examples: List[Any]) -> Dict[str, float]:
        """Evaluate predictions using predictions and examples."""
        # Extract ground truth from examples
        ground_truth = []
        for example in examples:
            if hasattr(example, 'human_label') and example.human_label is not None:
                ground_truth.append(example.human_label)
            elif hasattr(example, 'target_label') and example.target_label is not None:
                ground_truth.append(example.target_label)
            else:
                ground_truth.append(None)
        
        # Filter to cases where we have both predictions and ground truth
        valid_pairs = [(p, g) for p, g in zip(predictions, ground_truth) if p is not None and g is not None]
        
        if not valid_pairs:
            return {"error": "No valid prediction-ground_truth pairs found"}
        
        filtered_predictions, filtered_ground_truth = zip(*valid_pairs)
        return self.compute_performance_metrics(list(filtered_predictions), list(filtered_ground_truth))


class RankingEvaluator(BaseEvaluator):
    """Evaluator for summary ranking task."""

    def task_type(self) -> str:
        return "summary_ranking"

    def compute_performance_metrics(
        self, predictions: List[List[int]], ground_truth: List[List[int]]
    ) -> Dict[str, float]:
        """Compute ranking metrics."""
        valid_pairs = [
            (p, g)
            for p, g in zip(predictions, ground_truth)
            if p is not None and g is not None
        ]

        if not valid_pairs:
            return {"error": "no_valid_pairs"}

        # Compute ranking metrics
        kendall_taus = []
        spearman_rs = []

        for pred_rank, true_rank in valid_pairs:
            if len(pred_rank) == len(true_rank):
                try:
                    # Kendall's tau
                    tau, _ = stats.kendalltau(pred_rank, true_rank)
                    kendall_taus.append(tau)

                    # Spearman correlation
                    rho, _ = spearmanr(pred_rank, true_rank)
                    spearman_rs.append(rho)
                except:
                    continue

        metrics = {
            "n_valid_pairs": len(valid_pairs),
            "n_kendall_computed": len(kendall_taus),
            "n_spearman_computed": len(spearman_rs),
        }

        if kendall_taus:
            metrics.update(
                {
                    "kendall_tau_mean": np.mean(kendall_taus),
                    "kendall_tau_std": np.std(kendall_taus),
                    "kendall_tau_median": np.median(kendall_taus),
                }
            )

        if spearman_rs:
            metrics.update(
                {
                    "spearman_r_mean": np.mean(spearman_rs),
                    "spearman_r_std": np.std(spearman_rs),
                    "spearman_r_median": np.median(spearman_rs),
                }
            )

        return metrics

    def validate_predictions(
        self, predictions: List[List[int]]
    ) -> Tuple[bool, List[str]]:
        """Validate ranking predictions."""
        errors = []

        for i, pred in enumerate(predictions):
            if pred is None:
                continue

            if not isinstance(pred, list):
                errors.append(f"Prediction at index {i} must be a list")
                continue

            if len(pred) < 2:
                errors.append(f"Ranking at index {i} must have at least 2 items")
                continue

            # Check if it's a valid ranking (consecutive integers starting from 1)
            expected = list(range(1, len(pred) + 1))
            if sorted(pred) != expected:
                errors.append(f"Invalid ranking at index {i}: {pred}")

        return len(errors) == 0, errors

    def evaluate_predictions(self, predictions: List[Any], examples: List[Any]) -> Dict[str, float]:
        """Evaluate predictions using predictions and examples."""
        # Extract ranking values from RankingResult objects or use raw predictions
        ranking_values = []
        for prediction in predictions:
            if hasattr(prediction, 'ranking'):
                # Handle RankingResult objects
                ranking_values.append(prediction.ranking)
            elif hasattr(prediction, 'prediction') and isinstance(prediction.prediction, list):
                # Handle TaskResult objects with list predictions
                ranking_values.append(prediction.prediction)
            elif isinstance(prediction, list):
                # Handle raw list predictions (original functionality)
                ranking_values.append(prediction)
            else:
                ranking_values.append(None)
        
        # Extract ground truth from examples
        ground_truth = []
        for example in examples:
            if hasattr(example, 'human_label') and example.human_label is not None:
                ground_truth.append(example.human_label)
            elif hasattr(example, 'target_label') and example.target_label is not None:
                ground_truth.append(example.target_label)
            else:
                ground_truth.append(None)
        
        # Filter to cases where we have both predictions and ground truth
        valid_pairs = [(p, g) for p, g in zip(ranking_values, ground_truth) if p is not None and g is not None and isinstance(p, list) and isinstance(g, list)]
        
        if not valid_pairs:
            return {"error": "No valid prediction-ground_truth pairs found"}
        
        filtered_predictions, filtered_ground_truth = zip(*valid_pairs)
        return self.compute_performance_metrics(list(filtered_predictions), list(filtered_ground_truth))


class RatingEvaluator(BaseEvaluator):
    """Evaluator for consistency rating task."""

    def task_type(self) -> str:
        return "consistency_rating"

    def compute_performance_metrics(
        self, predictions: List[float], ground_truth: List[float]
    ) -> Dict[str, float]:
        """Compute regression metrics for rating task."""
        valid_pairs = [
            (p, g)
            for p, g in zip(predictions, ground_truth)
            if p is not None and g is not None
        ]

        if not valid_pairs:
            return {"error": "no_valid_pairs"}

        preds, truth = zip(*valid_pairs)
        preds = np.array(preds)
        truth = np.array(truth)

        # Compute regression metrics
        mse = np.mean((preds - truth) ** 2)
        rmse = np.sqrt(mse)
        mae = np.mean(np.abs(preds - truth))

        # R-squared
        ss_res = np.sum((truth - preds) ** 2)
        ss_tot = np.sum((truth - np.mean(truth)) ** 2)
        r2 = 1 - (ss_res / ss_tot) if ss_tot > 0 else 0

        return {
            "mse": mse,
            "rmse": rmse,
            "mae": mae,
            "r_squared": r2,
            "n_valid_pairs": len(valid_pairs),
            "pred_mean": np.mean(preds),
            "pred_std": np.std(preds),
            "truth_mean": np.mean(truth),
            "truth_std": np.std(truth),
        }

    def validate_predictions(self, predictions: List[float]) -> Tuple[bool, List[str]]:
        """Validate rating predictions."""
        errors = []

        for i, pred in enumerate(predictions):
            if pred is None:
                continue

            if not isinstance(pred, (int, float)):
                errors.append(f"Prediction at index {i} must be numeric")
                continue

            if not (0 <= pred <= 100):
                errors.append(f"Rating at index {i} must be between 0 and 100: {pred}")

        return len(errors) == 0, errors

    def evaluate_predictions(self, predictions: List[Any], examples: List[Any]) -> Dict[str, float]:
        """Evaluate predictions using predictions and examples."""
        # Extract rating values from RatingResult objects or use raw predictions
        rating_values = []
        for prediction in predictions:
            if hasattr(prediction, 'rating'):
                # Handle RatingResult objects
                rating_values.append(prediction.rating)
            elif hasattr(prediction, 'prediction') and isinstance(prediction.prediction, (int, float)):
                # Handle TaskResult objects with numeric predictions
                rating_values.append(prediction.prediction)
            elif isinstance(prediction, (int, float)):
                # Handle raw numeric predictions (original functionality)
                rating_values.append(prediction)
            else:
                rating_values.append(None)
        
        # Extract ground truth from examples
        ground_truth = []
        for example in examples:
            if hasattr(example, 'human_label') and example.human_label is not None:
                ground_truth.append(example.human_label)
            elif hasattr(example, 'target_label') and example.target_label is not None:
                ground_truth.append(example.target_label)
            else:
                ground_truth.append(None)
        
        # Filter to cases where we have both predictions and ground truth
        valid_pairs = [(p, g) for p, g in zip(rating_values, ground_truth) if p is not None and g is not None and isinstance(p, (int, float)) and isinstance(g, (int, float))]
        
        if not valid_pairs:
            return {"error": "No valid prediction-ground_truth pairs found"}
        
        filtered_predictions, filtered_ground_truth = zip(*valid_pairs)
        return self.compute_performance_metrics(list(filtered_predictions), list(filtered_ground_truth))


# Evaluator factory
class EvaluatorFactory:
    """Factory for creating task-specific evaluators."""

    _evaluators = {
        "entailment_inference": EntailmentEvaluator,
        "summary_ranking": RankingEvaluator,
        "consistency_rating": RatingEvaluator,
    }

    @classmethod
    def create_evaluator(cls, task_type: str, **kwargs) -> BaseEvaluator:
        """
        Create evaluator for specific task.

        Args:
            task_type: Type of task
            **kwargs: Arguments passed to evaluator constructor

        Returns:
            Task-specific evaluator instance
        """
        if task_type not in cls._evaluators:
            available = ", ".join(cls._evaluators.keys())
            raise ValueError(
                f"Unsupported task type: {task_type}. " f"Available types: {available}"
            )

        evaluator_class = cls._evaluators[task_type]
        return evaluator_class(**kwargs)

    @classmethod
    def get_supported_tasks(cls) -> List[str]:
        """Get list of supported task types."""
        return list(cls._evaluators.keys())


def quick_evaluate(
    task_type: str, predictions: List[Any], examples: List[ProcessedExample], **kwargs
) -> EvaluationResult:
    """
    Quick evaluation function for convenience.

    Args:
        task_type: Type of task to evaluate
        predictions: Model predictions
        examples: Processed examples
        **kwargs: Additional arguments passed to evaluator

    Returns:
        EvaluationResult object
    """
    evaluator = EvaluatorFactory.create_evaluator(task_type, **kwargs)
    return evaluator.evaluate(predictions, examples, **kwargs)


if __name__ == "__main__":
    """Test the evaluators."""
    logging.basicConfig(level=logging.INFO)

    # Test evaluator creation
    for task_type in EvaluatorFactory.get_supported_tasks():
        print(f"Testing {task_type} evaluator:")
        evaluator = EvaluatorFactory.create_evaluator(task_type)
        print(f"  Created: {evaluator.__class__.__name__}")
        print(f"  Task type: {evaluator.task_type()}")
