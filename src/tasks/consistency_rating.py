"""
Consistency Rating Task for ChatGPT Factuality Evaluation
========================================================

Implementation of consistency rating task that provides fine-grained
0-100 scale ratings of factual consistency between summaries and
source documents.

This task evaluates ChatGPT's ability to provide nuanced, continuous
assessments of factual consistency suitable for correlation analysis
with human judgments.

Author: Michael Ogunjimi
Institution: University of Manchester
Course: MSc AI
"""

from dataclasses import dataclass
from typing import Dict, List, Any, Optional, Tuple

import numpy as np
from scipy.stats import pearsonr, spearmanr
from sklearn.metrics import mean_absolute_error, mean_squared_error

from .base_task import BaseFactualityTask, TaskExample, TaskResult, TaskConfig
from ..llm_clients.openai_client import APICallResult
from ..utils.logging import get_logger

logger = get_logger(__name__)


@dataclass
class RatingResult(TaskResult):
    """
    Specialized result for consistency rating task.

    Extends TaskResult with rating-specific fields and
    convenience methods for continuous score analysis.
    """

    rating: float = 50.0  # 0-100 scale rating
    normalized_rating: Optional[float] = None  # 0-1 scale rating
    rating_category: Optional[str] = None  # High/Medium/Low category

    def __post_init__(self):
        super().__post_init__()
        # Ensure prediction is in correct format
        self.prediction = self.rating

        # Compute normalized rating
        self.normalized_rating = self.rating / 100.0

        # Assign rating category
        self.rating_category = self._categorize_rating(self.rating)

        # Validate rating range
        if not (0 <= self.rating <= 100):
            logger.warning(f"Rating {self.rating} outside valid range [0, 100]")

    def _categorize_rating(self, rating: float) -> str:
        """Categorize rating into qualitative groups."""
        if rating >= 80:
            return "High"
        elif rating >= 50:
            return "Medium"
        elif rating >= 20:
            return "Low"
        else:
            return "Very Low"

    def is_high_consistency(self, threshold: float = 70.0) -> bool:
        """Check if rating indicates high consistency."""
        return self.rating >= threshold

    def is_low_consistency(self, threshold: float = 30.0) -> bool:
        """Check if rating indicates low consistency."""
        return self.rating <= threshold

    def get_rating_error(self, human_rating: float) -> float:
        """Calculate absolute error against human rating."""
        # Handle case where human_rating might be a RatingResult object
        if hasattr(human_rating, 'rating'):
            human_rating = human_rating.rating
        return abs(self.rating - human_rating)

    def get_rating_squared_error(self, human_rating: float) -> float:
        """Calculate squared error against human rating."""
        # Handle case where human_rating might be a RatingResult object
        if hasattr(human_rating, 'rating'):
            human_rating = human_rating.rating
        return (self.rating - human_rating) ** 2

    def matches_rating_category(self, human_rating: float) -> bool:
        """Check if rating category matches human rating category."""
        human_category = self._categorize_rating(human_rating)
        return self.rating_category == human_category

    def __sub__(self, other):
        """Handle subtraction operations."""
        if isinstance(other, (int, float)):
            return self.rating - other
        elif hasattr(other, 'rating'):
            return self.rating - other.rating
        else:
            return NotImplemented

    def __rsub__(self, other):
        """Handle reverse subtraction operations."""
        if isinstance(other, (int, float)):
            return other - self.rating
        elif hasattr(other, 'rating'):
            return other.rating - self.rating
        else:
            return NotImplemented


class RatingMetrics:
    """
    Comprehensive metrics for consistency rating evaluation.

    Implements regression and correlation metrics specifically
    designed for continuous rating assessment and human correlation.
    """

    @staticmethod
    def pearson_correlation(
        y_true: List[float], y_pred: List[float]
    ) -> Tuple[float, float]:
        """
        Compute Pearson correlation coefficient.

        Args:
            y_true: True ratings
            y_pred: Predicted ratings

        Returns:
            Tuple of (correlation, p_value)
        """
        if len(y_true) != len(y_pred):
            raise ValueError("Rating lists must have same length")

        if len(y_true) < 2:
            return 0.0, 1.0

        if len(set(y_true)) <= 1 or len(set(y_pred)) <= 1:
            return 0.0, 1.0

        try:
            corr, p_value = pearsonr(y_true, y_pred)
            return float(corr), float(p_value)
        except Exception as e:
            logger.warning(f"Pearson correlation calculation failed: {e}")
            return 0.0, 1.0

    @staticmethod
    def spearman_correlation(
        y_true: List[float], y_pred: List[float]
    ) -> Tuple[float, float]:
        """
        Compute Spearman rank correlation.

        Args:
            y_true: True ratings
            y_pred: Predicted ratings

        Returns:
            Tuple of (correlation, p_value)
        """
        if len(y_true) != len(y_pred):
            raise ValueError("Rating lists must have same length")

        if len(y_true) < 2:
            return 0.0, 1.0

        if len(set(y_true)) <= 1 or len(set(y_pred)) <= 1:
            return 0.0, 1.0

        try:
            corr, p_value = spearmanr(y_true, y_pred)
            return float(corr), float(p_value)
        except Exception as e:
            logger.warning(f"Spearman correlation calculation failed: {e}")
            return 0.0, 1.0

    @staticmethod
    def mean_absolute_error(y_true: List[float], y_pred: List[float]) -> float:
        """
        Compute Mean Absolute Error (MAE).

        Args:
            y_true: True ratings
            y_pred: Predicted ratings

        Returns:
            MAE score
        """
        if len(y_true) != len(y_pred):
            raise ValueError("Rating lists must have same length")

        try:
            return float(mean_absolute_error(y_true, y_pred))
        except Exception as e:
            logger.warning(f"MAE calculation failed: {e}")
            return float("inf")

    @staticmethod
    def root_mean_squared_error(y_true: List[float], y_pred: List[float]) -> float:
        """
        Compute Root Mean Squared Error (RMSE).

        Args:
            y_true: True ratings
            y_pred: Predicted ratings

        Returns:
            RMSE score
        """
        if len(y_true) != len(y_pred):
            raise ValueError("Rating lists must have same length")

        try:
            mse = mean_squared_error(y_true, y_pred)
            return float(np.sqrt(mse))
        except Exception as e:
            logger.warning(f"RMSE calculation failed: {e}")
            return float("inf")

    @staticmethod
    def mean_bias_error(y_true: List[float], y_pred: List[float]) -> float:
        """
        Compute Mean Bias Error (systematic over/under-estimation).

        Args:
            y_true: True ratings
            y_pred: Predicted ratings

        Returns:
            MBE score (positive = overestimation, negative = underestimation)
        """
        if len(y_true) != len(y_pred):
            raise ValueError("Rating lists must have same length")

        return float(np.mean(np.array(y_pred) - np.array(y_true)))

    @staticmethod
    def rating_agreement_within_threshold(
        y_true: List[float], y_pred: List[float], threshold: float = 10.0
    ) -> float:
        """
        Compute agreement rate within threshold.

        Args:
            y_true: True ratings
            y_pred: Predicted ratings
            threshold: Agreement threshold

        Returns:
            Agreement rate (0-1)
        """
        if len(y_true) != len(y_pred):
            raise ValueError("Rating lists must have same length")

        agreements = [
            abs(true - pred) <= threshold for true, pred in zip(y_true, y_pred)
        ]
        return np.mean(agreements)

    @staticmethod
    def categorical_agreement(
        y_true: List[float], y_pred: List[float]
    ) -> Dict[str, float]:
        """
        Compute agreement on categorical rating levels.

        Args:
            y_true: True ratings
            y_pred: Predicted ratings

        Returns:
            Dictionary with categorical agreement metrics
        """

        def categorize(rating):
            if rating >= 80:
                return "High"
            elif rating >= 50:
                return "Medium"
            elif rating >= 20:
                return "Low"
            else:
                return "Very Low"

        true_categories = [categorize(r) for r in y_true]
        pred_categories = [categorize(r) for r in y_pred]

        # Exact agreement
        exact_agreement = np.mean(
            [tc == pc for tc, pc in zip(true_categories, pred_categories)]
        )

        # Adjacent agreement (within one category)
        category_order = ["Very Low", "Low", "Medium", "High"]
        category_to_idx = {cat: i for i, cat in enumerate(category_order)}

        adjacent_agreement = 0
        for tc, pc in zip(true_categories, pred_categories):
            true_idx = category_to_idx[tc]
            pred_idx = category_to_idx[pc]
            if abs(true_idx - pred_idx) <= 1:
                adjacent_agreement += 1

        adjacent_agreement /= len(true_categories)

        return {
            "exact_categorical_agreement": exact_agreement,
            "adjacent_categorical_agreement": adjacent_agreement,
            "category_distribution_true": {
                cat: true_categories.count(cat) / len(true_categories)
                for cat in category_order
            },
            "category_distribution_pred": {
                cat: pred_categories.count(cat) / len(pred_categories)
                for cat in category_order
            },
        }

    @staticmethod
    def compute_comprehensive_metrics(
        y_true: List[float], y_pred: List[float]
    ) -> Dict[str, Any]:
        """
        Compute all rating metrics together.

        Args:
            y_true: True ratings
            y_pred: Predicted ratings

        Returns:
            Dictionary with all rating metrics
        """
        metrics = {}

        # Correlation metrics
        pearson_r, pearson_p = RatingMetrics.pearson_correlation(y_true, y_pred)
        spearman_r, spearman_p = RatingMetrics.spearman_correlation(y_true, y_pred)

        metrics.update(
            {
                "pearson_correlation": pearson_r,
                "pearson_p_value": pearson_p,
                "spearman_correlation": spearman_r,
                "spearman_p_value": spearman_p,
            }
        )

        # Error metrics
        metrics.update(
            {
                "mean_absolute_error": RatingMetrics.mean_absolute_error(
                    y_true, y_pred
                ),
                "root_mean_squared_error": RatingMetrics.root_mean_squared_error(
                    y_true, y_pred
                ),
                "mean_bias_error": RatingMetrics.mean_bias_error(y_true, y_pred),
            }
        )

        # Agreement metrics
        metrics.update(
            {
                "agreement_within_5": RatingMetrics.rating_agreement_within_threshold(
                    y_true, y_pred, 5.0
                ),
                "agreement_within_10": RatingMetrics.rating_agreement_within_threshold(
                    y_true, y_pred, 10.0
                ),
                "agreement_within_15": RatingMetrics.rating_agreement_within_threshold(
                    y_true, y_pred, 15.0
                ),
            }
        )

        # Categorical agreement
        categorical_metrics = RatingMetrics.categorical_agreement(y_true, y_pred)
        metrics["categorical_analysis"] = categorical_metrics

        return metrics


class ConsistencyRatingTask(BaseFactualityTask):
    """
    Consistency rating task for factuality evaluation.

    Provides fine-grained 0-100 scale ratings of factual consistency
    between summaries and source documents using ChatGPT.
    """

    def _create_task_config(self) -> TaskConfig:
        """Create configuration specific to consistency rating."""
        task_config_dict = self.config.get("tasks", {}).get("consistency_rating", {})

        return TaskConfig(
            task_type="consistency_rating",
            prompt_type=task_config_dict.get("prompt_type", "zero_shot"),
            model_name=task_config_dict.get("model_name", "gpt-4.1-mini"),
            temperature=task_config_dict.get("temperature", 0.0),
            max_tokens=task_config_dict.get(
                "max_tokens", None
            ),  # Use adaptive tokens from OpenAI client
            batch_size=task_config_dict.get("batch_size", 10),
            max_examples=task_config_dict.get("max_examples"),
            include_human_eval=task_config_dict.get("include_human_eval", False),
            save_intermediate=task_config_dict.get("save_intermediate", True),
            cache_responses=task_config_dict.get("cache_responses", True),
            retry_failed=task_config_dict.get("retry_failed", True),
        )

    def _validate_example(self, example: TaskExample) -> bool:
        """
        Validate example for consistency rating task.

        Args:
            example: TaskExample to validate

        Returns:
            True if example is valid for rating task
        """
        try:
            # Must have source and summary
            if not example.source or not example.source.strip():
                logger.warning(f"Example {example.example_id}: Empty source")
                return False

            # Get summary for rating task
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

            # Validate human rating if present
            if example.human_label is not None:
                if not isinstance(example.human_label, (int, float)):
                    logger.warning(
                        f"Example {example.example_id}: Human label must be numeric for rating"
                    )
                    return False

                human_rating = float(example.human_label)
                if not (0 <= human_rating <= 100):
                    logger.warning(
                        f"Example {example.example_id}: Human rating outside valid range [0, 100]"
                    )
                    return False

            return True

        except Exception as e:
            logger.error(f"Example validation failed for {example.example_id}: {e}")
            return False

    def _process_api_result(
        self, api_result: APICallResult, example: TaskExample
    ) -> TaskResult:
        """
        Process API result into RatingResult.

        Args:
            api_result: APICallResult from OpenAI client
            example: Original TaskExample

        Returns:
            RatingResult object
        """
        parsed_content = api_result.parsed_content
        raw_response = api_result.raw_response

        # Extract rating
        rating = parsed_content.get(
            "rating", 50.0
        )  # Default to middle rating if parsing fails

        # Ensure rating is in valid range
        rating = max(0.0, min(100.0, float(rating)))

        return RatingResult(
            example_id=example.example_id,
            task_type=self.task_config.task_type,
            prompt_type=self.task_config.prompt_type,
            prediction=rating,
            rating=rating,
            confidence=api_result.confidence_score,
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
                "rating_category": RatingResult(
                    example_id="temp",
                    task_type=self.task_config.task_type,
                    prompt_type=self.task_config.prompt_type,
                    prediction=rating,
                    rating=rating,
                    confidence=0.5,
                    raw_response="",
                    processing_time=0.0,
                    cost=0.0,
                    tokens_used=0,
                    timestamp="",
                    success=True,
                ).rating_category,
            },
        )

    def evaluate_predictions(self, results: List[TaskResult]) -> Dict[str, float]:
        """
        Compute comprehensive evaluation metrics for consistency rating.

        Args:
            results: List of RatingResult objects

        Returns:
            Dictionary with evaluation metrics
        """
        if not results:
            return {"error": "No results to evaluate"}

        # Filter to successful results
        successful_results = [
            r for r in results if r.success and isinstance(r, RatingResult)
        ]

        if not successful_results:
            return {"error": "No successful results to evaluate"}

        # Extract ratings
        ratings = [r.rating for r in successful_results]

        # Check if we have human labels for evaluation
        human_ratings = [
            float(r.human_label)
            for r in successful_results
            if r.human_label is not None
        ]
        has_human_labels = len(human_ratings) == len(successful_results)

        evaluation_metrics = {
            "total_examples": len(successful_results),
            "has_human_labels": has_human_labels,
        }

        if has_human_labels:
            # Compute rating metrics against human labels
            y_true = human_ratings
            y_pred = ratings

            comprehensive_metrics = RatingMetrics.compute_comprehensive_metrics(
                y_true, y_pred
            )
            evaluation_metrics.update(comprehensive_metrics)
            
            # Set primary metric for experiment reporting (using correlation)
            evaluation_metrics["primary_metric"] = comprehensive_metrics.get("pearson_correlation", 0.0)

        else:
            # Without human labels, provide descriptive statistics
            mean_rating = np.mean(ratings)
            evaluation_metrics.update(
                {
                    "rating_statistics": {
                        "mean_rating": mean_rating,
                        "std_rating": np.std(ratings),
                        "min_rating": np.min(ratings),
                        "max_rating": np.max(ratings),
                        "median_rating": np.median(ratings),
                    }
                }
            )
            
            # Set primary metric for experiment reporting (using mean rating normalized to 0-1)
            evaluation_metrics["primary_metric"] = mean_rating / 100.0

        # Rating distribution analysis
        rating_categories = [r.rating_category for r in successful_results]
        category_counts = {}
        for category in rating_categories:
            category_counts[category] = category_counts.get(category, 0) + 1

        evaluation_metrics["rating_distribution"] = {
            "category_counts": category_counts,
            "category_proportions": {
                cat: count / len(successful_results)
                for cat, count in category_counts.items()
            },
        }

        # Rating range analysis
        high_ratings = sum(1 for r in ratings if r >= 70)
        medium_ratings = sum(1 for r in ratings if 30 <= r < 70)
        low_ratings = sum(1 for r in ratings if r < 30)

        evaluation_metrics["rating_range_analysis"] = {
            "high_consistency": high_ratings,
            "medium_consistency": medium_ratings,
            "low_consistency": low_ratings,
            "high_consistency_rate": high_ratings / len(successful_results),
            "medium_consistency_rate": medium_ratings / len(successful_results),
            "low_consistency_rate": low_ratings / len(successful_results),
        }

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

    def analyze_rating_patterns(self, results: List[TaskResult]) -> Dict[str, Any]:
        """
        Analyze rating patterns and biases.

        Args:
            results: List of RatingResult objects

        Returns:
            Dictionary with rating pattern analysis
        """
        successful_results = [
            r for r in results if r.success and isinstance(r, RatingResult)
        ]

        if not successful_results:
            return {"error": "No successful results for analysis"}

        ratings = [r.rating for r in successful_results]

        analysis = {
            "total_ratings": len(successful_results),
            "rating_summary": {
                "mean": np.mean(ratings),
                "median": np.median(ratings),
                "std": np.std(ratings),
                "range": np.max(ratings) - np.min(ratings),
                "iqr": np.percentile(ratings, 75) - np.percentile(ratings, 25),
            },
        }

        # Bias analysis
        # Check for central tendency bias (clustering around 50)
        central_ratings = sum(1 for r in ratings if 40 <= r <= 60)
        analysis["bias_analysis"] = {
            "central_tendency_bias": central_ratings / len(ratings),
            "extreme_rating_avoidance": {
                "very_low_ratings": sum(1 for r in ratings if r <= 10) / len(ratings),
                "very_high_ratings": sum(1 for r in ratings if r >= 90) / len(ratings),
            },
        }

        # Distribution shape analysis
        # Check for skewness
        from scipy import stats as scipy_stats

        skewness = scipy_stats.skew(ratings)
        kurtosis = scipy_stats.kurtosis(ratings)

        analysis["distribution_shape"] = {
            "skewness": float(skewness),
            "kurtosis": float(kurtosis),
            "distribution_interpretation": {
                "skew_direction": (
                    "right"
                    if skewness > 0.5
                    else "left" if skewness < -0.5 else "approximately_symmetric"
                ),
                "tail_heaviness": (
                    "heavy" if kurtosis > 1 else "light" if kurtosis < -1 else "normal"
                ),
            },
        }

        # Text length impact analysis
        length_impact = {"source_length": {}, "summary_length": {}}

        for result in successful_results:
            source_length = result.metadata.get("source_length", 0)
            summary_length = result.metadata.get("summary_length", 0)

            # Bin lengths
            source_bin = f"{source_length//200 * 200}-{(source_length//200 + 1) * 200}"
            summary_bin = f"{summary_length//50 * 50}-{(summary_length//50 + 1) * 50}"

            if source_bin not in length_impact["source_length"]:
                length_impact["source_length"][source_bin] = []
            if summary_bin not in length_impact["summary_length"]:
                length_impact["summary_length"][summary_bin] = []

            length_impact["source_length"][source_bin].append(result.rating)
            length_impact["summary_length"][summary_bin].append(result.rating)

        # Calculate average ratings by length
        for length_type in ["source_length", "summary_length"]:
            for length_bin, bin_ratings in length_impact[length_type].items():
                if len(bin_ratings) >= 3:  # Only include bins with sufficient data
                    length_impact[length_type][length_bin] = {
                        "average_rating": np.mean(bin_ratings),
                        "count": len(bin_ratings),
                        "std": np.std(bin_ratings),
                    }
                else:
                    del length_impact[length_type][length_bin]

        analysis["length_impact"] = length_impact

        return analysis

    def get_task_specific_insights(self, results: List[TaskResult]) -> Dict[str, Any]:
        """
        Get insights specific to consistency rating task.

        Args:
            results: List of RatingResult objects

        Returns:
            Dictionary with task-specific insights
        """
        successful_results = [
            r for r in results if r.success and isinstance(r, RatingResult)
        ]

        if not successful_results:
            return {"error": "No successful results for analysis"}

        ratings = [r.rating for r in successful_results]

        insights = {
            "rating_granularity": {
                "unique_ratings": len(set(ratings)),
                "granularity_score": len(set(ratings)) / len(ratings),
                "most_common_ratings": {
                    str(rating): ratings.count(rating)
                    for rating in sorted(set(ratings), key=ratings.count, reverse=True)[
                        :5
                    ]
                },
            }
        }

        # Confidence vs rating analysis
        confidences = [
            r.confidence for r in successful_results if r.confidence is not None
        ]
        if len(confidences) == len(ratings):
            # Correlation between confidence and rating extremeness
            rating_extremeness = [abs(r - 50) for r in ratings]  # Distance from middle
            if len(set(confidences)) > 1 and len(set(rating_extremeness)) > 1:
                conf_extreme_corr = np.corrcoef(confidences, rating_extremeness)[0, 1]
            else:
                conf_extreme_corr = 0.0

            insights["confidence_analysis"] = {
                "avg_confidence": np.mean(confidences),
                "confidence_std": np.std(confidences),
                "confidence_extremeness_correlation": conf_extreme_corr,
                "high_confidence_ratings": sum(1 for c in confidences if c > 0.8),
                "low_confidence_ratings": sum(1 for c in confidences if c < 0.3),
            }

        # Rating consistency analysis
        # Check for patterns in rating categories
        categories = [r.rating_category for r in successful_results]
        category_consistency = {}

        for category in set(categories):
            category_ratings = [
                r.rating for r in successful_results if r.rating_category == category
            ]
            if len(category_ratings) > 1:
                category_consistency[category] = {
                    "count": len(category_ratings),
                    "mean": np.mean(category_ratings),
                    "std": np.std(category_ratings),
                    "range": np.max(category_ratings) - np.min(category_ratings),
                }

        insights["category_consistency"] = category_consistency

        return insights
