"""
Statistical Analysis and Metrics for ChatGPT Factuality Evaluation
==================================================================

This module provides comprehensive statistical analysis capabilities for
evaluating ChatGPT's factuality assessment performance. It includes metrics
computation, correlation analysis, significance testing, and visualization
support suitable for academic research.

The metrics align with the three core tasks and support comparison with
state-of-the-art baselines and human annotations.

Author: Michael Ogunjimi
Institution: University of Manchester
Course: MSc AI
"""

import logging
import warnings
from dataclasses import dataclass
from typing import Dict, List, Optional, Union, Any, Tuple

import numpy as np
from scipy import stats
from scipy.stats import (
    pearsonr,
    spearmanr,
    kendalltau,
    ttest_rel,
    wilcoxon,
    shapiro,
    normaltest,
    kstest,
)

# Suppress warnings for cleaner output
warnings.filterwarnings("ignore", category=RuntimeWarning)

logger = logging.getLogger(__name__)


@dataclass
class MetricResult:
    """
    Container for individual metric computation results.

    Provides standardized format for metric outputs with metadata
    for academic analysis and reporting.
    """

    metric_name: str
    value: Union[float, int, str, Dict[str, Any]]
    confidence_interval: Optional[Tuple[float, float]] = None
    p_value: Optional[float] = None
    n_samples: Optional[int] = None
    metadata: Optional[Dict[str, Any]] = None

    def __post_init__(self):
        """Post-initialization setup."""
        if self.metadata is None:
            self.metadata = {}

    def is_significant(self, alpha: float = 0.05) -> Optional[bool]:
        """Check if result is statistically significant."""
        if self.p_value is None:
            return None
        return self.p_value < alpha

    def to_dict(self) -> Dict[str, Any]:
        """Convert to dictionary for serialization."""
        return {
            "metric_name": self.metric_name,
            "value": self.value,
            "confidence_interval": self.confidence_interval,
            "p_value": self.p_value,
            "n_samples": self.n_samples,
            "metadata": self.metadata,
            "is_significant": self.is_significant() if self.p_value else None,
        }


class StatisticalAnalyzer:
    """
    Comprehensive statistical analysis toolkit for factuality evaluation.

    Provides methods for correlation analysis, significance testing,
    effect size computation, and distribution analysis suitable for
    academic research standards.
    """

    def __init__(
        self,
        alpha: float = 0.05,
        bootstrap_samples: int = 1000,
        random_state: Optional[int] = 42,
    ):
        """
        Initialize statistical analyzer.

        Args:
            alpha: Significance level for hypothesis testing
            bootstrap_samples: Number of bootstrap samples for CI estimation
            random_state: Random state for reproducibility
        """
        self.alpha = alpha
        self.bootstrap_samples = bootstrap_samples
        self.random_state = random_state

        if random_state is not None:
            np.random.seed(random_state)

        self.logger = logging.getLogger(f"{__name__}.{self.__class__.__name__}")

    def compute_correlation_analysis(
        self,
        predictions: List[Union[float, int]],
        ground_truth: List[Union[float, int]],
        method: str = "all",
    ) -> Dict[str, MetricResult]:
        """
        Compute comprehensive correlation analysis.

        Args:
            predictions: Model predictions
            ground_truth: Ground truth values
            method: Correlation method ("pearson", "spearman", "kendall", or "all")

        Returns:
            Dictionary of MetricResult objects for different correlation measures
        """
        # Clean and validate data
        valid_pairs = self._clean_numeric_pairs(predictions, ground_truth)
        if len(valid_pairs) < 3:
            self.logger.warning("Insufficient valid pairs for correlation analysis")
            return {"error": MetricResult("error", "insufficient_data")}

        preds, truth = zip(*valid_pairs)
        preds = np.array(preds)
        truth = np.array(truth)

        results = {}

        # Pearson correlation
        if method in ["pearson", "all"]:
            try:
                r, p = pearsonr(preds, truth)
                ci = self._bootstrap_correlation_ci(preds, truth, "pearson")

                results["pearson_r"] = MetricResult(
                    metric_name="pearson_r",
                    value=r,
                    confidence_interval=ci,
                    p_value=p,
                    n_samples=len(valid_pairs),
                    metadata={"method": "pearson", "type": "correlation"},
                )
            except Exception as e:
                self.logger.warning(f"Failed to compute Pearson correlation: {e}")
                results["pearson_r"] = MetricResult(
                    "pearson_r", None, metadata={"error": str(e)}
                )

        # Spearman correlation
        if method in ["spearman", "all"]:
            try:
                rho, p = spearmanr(preds, truth)
                ci = self._bootstrap_correlation_ci(preds, truth, "spearman")

                results["spearman_rho"] = MetricResult(
                    metric_name="spearman_rho",
                    value=rho,
                    confidence_interval=ci,
                    p_value=p,
                    n_samples=len(valid_pairs),
                    metadata={"method": "spearman", "type": "correlation"},
                )
            except Exception as e:
                self.logger.warning(f"Failed to compute Spearman correlation: {e}")
                results["spearman_rho"] = MetricResult(
                    "spearman_rho", None, metadata={"error": str(e)}
                )

        # Kendall's tau
        if method in ["kendall", "all"]:
            try:
                tau, p = kendalltau(preds, truth)
                ci = self._bootstrap_correlation_ci(preds, truth, "kendall")

                results["kendall_tau"] = MetricResult(
                    metric_name="kendall_tau",
                    value=tau,
                    confidence_interval=ci,
                    p_value=p,
                    n_samples=len(valid_pairs),
                    metadata={"method": "kendall", "type": "correlation"},
                )
            except Exception as e:
                self.logger.warning(f"Failed to compute Kendall's tau: {e}")
                results["kendall_tau"] = MetricResult(
                    "kendall_tau", None, metadata={"error": str(e)}
                )

        return results

    def compute_agreement_metrics(
        self,
        predictions: List[Any],
        ground_truth: List[Any],
        task_type: str = "classification",
    ) -> Dict[str, MetricResult]:
        """
        Compute inter-rater agreement metrics.

        Args:
            predictions: Model predictions
            ground_truth: Human annotations
            task_type: Type of task ("classification", "ranking", "rating")

        Returns:
            Dictionary of agreement metrics
        """
        valid_pairs = self._clean_valid_pairs(predictions, ground_truth)
        if len(valid_pairs) < 2:
            return {"error": MetricResult("error", "insufficient_data")}

        results = {}

        if task_type == "classification":
            results.update(self._compute_classification_agreement(valid_pairs))
        elif task_type == "ranking":
            results.update(self._compute_ranking_agreement(valid_pairs))
        elif task_type == "rating":
            results.update(self._compute_rating_agreement(valid_pairs))

        return results

    def compute_significance_tests(
        self,
        predictions_a: List[Union[float, int]],
        predictions_b: List[Union[float, int]],
        ground_truth: List[Union[float, int]],
        test_type: str = "auto",
    ) -> Dict[str, MetricResult]:
        """
        Compute significance tests for comparing two sets of predictions.

        Args:
            predictions_a: First set of predictions (e.g., ChatGPT)
            predictions_b: Second set of predictions (e.g., baseline)
            ground_truth: Ground truth values
            test_type: Type of test ("auto", "paired_t", "wilcoxon", "mcnemar")

        Returns:
            Dictionary of significance test results
        """
        # Compute errors for both prediction sets
        valid_triplets = []
        for a, b, gt in zip(predictions_a, predictions_b, ground_truth):
            if all(x is not None for x in [a, b, gt]):
                valid_triplets.append((a, b, gt))

        if len(valid_triplets) < 5:
            return {"error": MetricResult("error", "insufficient_data")}

        preds_a, preds_b, truth = zip(*valid_triplets)

        results = {}

        # Compute absolute errors
        errors_a = np.abs(np.array(preds_a) - np.array(truth))
        errors_b = np.abs(np.array(preds_b) - np.array(truth))

        # Choose appropriate test
        if test_type == "auto":
            # Use normality to decide between parametric and non-parametric tests
            _, p_norm_a = normaltest(errors_a)
            _, p_norm_b = normaltest(errors_b)

            if p_norm_a > 0.05 and p_norm_b > 0.05:
                test_type = "paired_t"
            else:
                test_type = "wilcoxon"

        # Paired t-test
        if test_type == "paired_t":
            try:
                t_stat, p_value = ttest_rel(errors_a, errors_b)

                results["paired_t_test"] = MetricResult(
                    metric_name="paired_t_test",
                    value=t_stat,
                    p_value=p_value,
                    n_samples=len(valid_triplets),
                    metadata={
                        "test_type": "paired_t",
                        "alternative": "two-sided",
                        "mean_error_a": np.mean(errors_a),
                        "mean_error_b": np.mean(errors_b),
                    },
                )
            except Exception as e:
                results["paired_t_test"] = MetricResult(
                    "paired_t_test", None, metadata={"error": str(e)}
                )

        # Wilcoxon signed-rank test
        if test_type == "wilcoxon":
            try:
                w_stat, p_value = wilcoxon(errors_a, errors_b)

                results["wilcoxon_test"] = MetricResult(
                    metric_name="wilcoxon_test",
                    value=w_stat,
                    p_value=p_value,
                    n_samples=len(valid_triplets),
                    metadata={
                        "test_type": "wilcoxon",
                        "alternative": "two-sided",
                        "median_error_a": np.median(errors_a),
                        "median_error_b": np.median(errors_b),
                    },
                )
            except Exception as e:
                results["wilcoxon_test"] = MetricResult(
                    "wilcoxon_test", None, metadata={"error": str(e)}
                )

        return results

    def compute_effect_sizes(
        self,
        predictions_a: List[Union[float, int]],
        predictions_b: List[Union[float, int]],
        ground_truth: List[Union[float, int]],
    ) -> Dict[str, MetricResult]:
        """
        Compute effect sizes for comparing prediction sets.

        Args:
            predictions_a: First set of predictions
            predictions_b: Second set of predictions
            ground_truth: Ground truth values

        Returns:
            Dictionary of effect size metrics
        """
        valid_triplets = []
        for a, b, gt in zip(predictions_a, predictions_b, ground_truth):
            if all(x is not None for x in [a, b, gt]):
                valid_triplets.append((a, b, gt))

        if len(valid_triplets) < 3:
            return {"error": MetricResult("error", "insufficient_data")}

        preds_a, preds_b, truth = zip(*valid_triplets)

        # Compute errors
        errors_a = np.abs(np.array(preds_a) - np.array(truth))
        errors_b = np.abs(np.array(preds_b) - np.array(truth))

        results = {}

        # Cohen's d for paired differences
        differences = errors_a - errors_b
        d = np.mean(differences) / np.std(differences) if np.std(differences) > 0 else 0

        results["cohens_d"] = MetricResult(
            metric_name="cohens_d",
            value=d,
            n_samples=len(valid_triplets),
            metadata={
                "effect_size": (
                    "small" if abs(d) < 0.5 else "medium" if abs(d) < 0.8 else "large"
                ),
                "interpretation": "Difference in prediction errors between models",
            },
        )

        # Improvement percentage
        improvement = (np.mean(errors_b) - np.mean(errors_a)) / np.mean(errors_b) * 100

        results["improvement_percentage"] = MetricResult(
            metric_name="improvement_percentage",
            value=improvement,
            n_samples=len(valid_triplets),
            metadata={
                "interpretation": "Percentage improvement of model A over model B"
            },
        )

        return results

    def compute_distribution_analysis(
        self,
        data: List[Union[float, int]],
        reference_distribution: Optional[str] = None,
    ) -> Dict[str, MetricResult]:
        """
        Analyze data distribution properties.

        Args:
            data: Data to analyze
            reference_distribution: Reference distribution for testing ("normal", "uniform")

        Returns:
            Dictionary of distribution analysis results
        """
        clean_data = [x for x in data if x is not None]
        if len(clean_data) < 3:
            return {"error": MetricResult("error", "insufficient_data")}

        data_array = np.array(clean_data)
        results = {}

        # Descriptive statistics
        results["descriptive_stats"] = MetricResult(
            metric_name="descriptive_stats",
            value={
                "mean": np.mean(data_array),
                "std": np.std(data_array, ddof=1),
                "median": np.median(data_array),
                "min": np.min(data_array),
                "max": np.max(data_array),
                "q25": np.percentile(data_array, 25),
                "q75": np.percentile(data_array, 75),
                "skewness": stats.skew(data_array),
                "kurtosis": stats.kurtosis(data_array),
            },
            n_samples=len(clean_data),
        )

        # Normality tests
        if len(clean_data) >= 8:
            # Shapiro-Wilk test
            try:
                w_stat, p_value = shapiro(data_array)
                results["shapiro_test"] = MetricResult(
                    metric_name="shapiro_test",
                    value=w_stat,
                    p_value=p_value,
                    n_samples=len(clean_data),
                    metadata={
                        "test_type": "normality",
                        "null_hypothesis": "data is normally distributed",
                    },
                )
            except Exception as e:
                results["shapiro_test"] = MetricResult(
                    "shapiro_test", None, metadata={"error": str(e)}
                )

        # Kolmogorov-Smirnov test against reference distribution
        if reference_distribution == "normal":
            try:
                # Standardize data
                standardized = (data_array - np.mean(data_array)) / np.std(data_array)
                ks_stat, p_value = kstest(standardized, "norm")

                results["ks_test_normal"] = MetricResult(
                    metric_name="ks_test_normal",
                    value=ks_stat,
                    p_value=p_value,
                    n_samples=len(clean_data),
                    metadata={"test_type": "goodness_of_fit", "reference": "normal"},
                )
            except Exception as e:
                results["ks_test_normal"] = MetricResult(
                    "ks_test_normal", None, metadata={"error": str(e)}
                )

        return results

    def _clean_numeric_pairs(
        self, list_a: List[Union[float, int]], list_b: List[Union[float, int]]
    ) -> List[Tuple[float, float]]:
        """Clean and validate numeric pairs."""
        valid_pairs = []
        for a, b in zip(list_a, list_b):
            try:
                if a is not None and b is not None:
                    a_float = float(a)
                    b_float = float(b)
                    if not (
                        np.isnan(a_float)
                        or np.isnan(b_float)
                        or np.isinf(a_float)
                        or np.isinf(b_float)
                    ):
                        valid_pairs.append((a_float, b_float))
            except (ValueError, TypeError):
                continue
        return valid_pairs

    def _clean_valid_pairs(
        self, list_a: List[Any], list_b: List[Any]
    ) -> List[Tuple[Any, Any]]:
        """Clean and validate general pairs."""
        return [
            (a, b) for a, b in zip(list_a, list_b) if a is not None and b is not None
        ]

    def _bootstrap_correlation_ci(
        self,
        x: np.ndarray,
        y: np.ndarray,
        method: str = "pearson",
        confidence: float = 0.95,
    ) -> Tuple[float, float]:
        """Compute bootstrap confidence interval for correlation."""
        correlations = []
        n = len(x)

        for _ in range(self.bootstrap_samples):
            indices = np.random.choice(n, size=n, replace=True)
            x_boot = x[indices]
            y_boot = y[indices]

            try:
                if method == "pearson":
                    corr, _ = pearsonr(x_boot, y_boot)
                elif method == "spearman":
                    corr, _ = spearmanr(x_boot, y_boot)
                elif method == "kendall":
                    corr, _ = kendalltau(x_boot, y_boot)
                else:
                    continue

                if not np.isnan(corr):
                    correlations.append(corr)
            except:
                continue

        if len(correlations) < 10:
            return None, None

        alpha = 1 - confidence
        lower = np.percentile(correlations, 100 * alpha / 2)
        upper = np.percentile(correlations, 100 * (1 - alpha / 2))

        return lower, upper

    def _compute_classification_agreement(
        self, valid_pairs: List[Tuple[Any, Any]]
    ) -> Dict[str, MetricResult]:
        """Compute agreement metrics for classification tasks."""
        results = {}

        pred_labels, true_labels = zip(*valid_pairs)

        # Simple accuracy
        accuracy = sum(1 for p, t in valid_pairs if p == t) / len(valid_pairs)
        results["accuracy"] = MetricResult(
            metric_name="accuracy", value=accuracy, n_samples=len(valid_pairs)
        )

        # Cohen's kappa (if applicable)
        try:
            unique_labels = list(set(pred_labels + true_labels))
            if len(unique_labels) <= 10:  # Reasonable number of categories
                # Create confusion matrix
                label_to_idx = {label: i for i, label in enumerate(unique_labels)}
                n_labels = len(unique_labels)
                confusion_matrix = np.zeros((n_labels, n_labels))

                for p, t in valid_pairs:
                    confusion_matrix[label_to_idx[t], label_to_idx[p]] += 1

                # Compute kappa
                n = len(valid_pairs)
                observed_accuracy = np.trace(confusion_matrix) / n

                marginal_pred = np.sum(confusion_matrix, axis=0) / n
                marginal_true = np.sum(confusion_matrix, axis=1) / n
                expected_accuracy = np.sum(marginal_pred * marginal_true)

                kappa = (observed_accuracy - expected_accuracy) / (
                    1 - expected_accuracy
                )

                results["cohens_kappa"] = MetricResult(
                    metric_name="cohens_kappa",
                    value=kappa,
                    n_samples=len(valid_pairs),
                    metadata={"interpretation": self._interpret_kappa(kappa)},
                )
        except Exception as e:
            self.logger.warning(f"Failed to compute Cohen's kappa: {e}")

        return results

    def _compute_ranking_agreement(
        self, valid_pairs: List[Tuple[List[int], List[int]]]
    ) -> Dict[str, MetricResult]:
        """Compute agreement metrics for ranking tasks."""
        results = {}

        kendall_taus = []
        spearman_rhos = []

        for pred_rank, true_rank in valid_pairs:
            if len(pred_rank) == len(true_rank):
                try:
                    tau, _ = kendalltau(pred_rank, true_rank)
                    if not np.isnan(tau):
                        kendall_taus.append(tau)

                    rho, _ = spearmanr(pred_rank, true_rank)
                    if not np.isnan(rho):
                        spearman_rhos.append(rho)
                except:
                    continue

        if kendall_taus:
            results["kendall_tau_agreement"] = MetricResult(
                metric_name="kendall_tau_agreement",
                value=np.mean(kendall_taus),
                n_samples=len(kendall_taus),
                metadata={"std": np.std(kendall_taus)},
            )

        if spearman_rhos:
            results["spearman_rho_agreement"] = MetricResult(
                metric_name="spearman_rho_agreement",
                value=np.mean(spearman_rhos),
                n_samples=len(spearman_rhos),
                metadata={"std": np.std(spearman_rhos)},
            )

        return results

    def _compute_rating_agreement(
        self, valid_pairs: List[Tuple[float, float]]
    ) -> Dict[str, MetricResult]:
        """Compute agreement metrics for rating tasks."""
        results = {}

        pred_ratings, true_ratings = zip(*valid_pairs)
        pred_array = np.array(pred_ratings)
        true_array = np.array(true_ratings)

        # Intraclass correlation coefficient (ICC)
        try:
            icc = self._compute_icc(pred_array, true_array)
            results["icc"] = MetricResult(
                metric_name="icc",
                value=icc,
                n_samples=len(valid_pairs),
                metadata={"interpretation": self._interpret_icc(icc)},
            )
        except Exception as e:
            self.logger.warning(f"Failed to compute ICC: {e}")

        # Mean absolute error
        mae = np.mean(np.abs(pred_array - true_array))
        results["mae"] = MetricResult(
            metric_name="mae", value=mae, n_samples=len(valid_pairs)
        )

        return results

    def _compute_icc(self, ratings_a: np.ndarray, ratings_b: np.ndarray) -> float:
        """Compute intraclass correlation coefficient."""
        mean_a = np.mean(ratings_a)
        mean_b = np.mean(ratings_b)
        mean_total = (mean_a + mean_b) / 2

        ss_between = len(ratings_a) * (
            (mean_a - mean_total) ** 2 + (mean_b - mean_total) ** 2
        )
        ss_within = np.sum((ratings_a - mean_a) ** 2) + np.sum(
            (ratings_b - mean_b) ** 2
        )

        ms_between = ss_between / 1
        ms_within = ss_within / (2 * len(ratings_a) - 2)

        if ms_within == 0:
            return 1.0

        icc = (ms_between - ms_within) / (ms_between + ms_within)
        return max(0, icc)  # ICC should be non-negative

    def _interpret_kappa(self, kappa: float) -> str:
        """Interpret Cohen's kappa value."""
        if kappa < 0:
            return "poor"
        elif kappa < 0.20:
            return "slight"
        elif kappa < 0.40:
            return "fair"
        elif kappa < 0.60:
            return "moderate"
        elif kappa < 0.80:
            return "substantial"
        else:
            return "almost_perfect"

    def _interpret_icc(self, icc: float) -> str:
        """Interpret ICC value."""
        if icc < 0.5:
            return "poor"
        elif icc < 0.75:
            return "moderate"
        elif icc < 0.9:
            return "good"
        else:
            return "excellent"


class TaskSpecificMetrics:
    """
    Task-specific metric computations for the three core factuality tasks.

    Provides specialized metrics aligned with academic evaluation standards
    for entailment inference, summary ranking, and consistency rating.
    """

    @staticmethod
    def compute_entailment_metrics(
        predictions: List[str], ground_truth: List[str]
    ) -> Dict[str, MetricResult]:
        """
        Compute comprehensive metrics for entailment inference task.

        Args:
            predictions: Model predictions ("ENTAILMENT" or "CONTRADICTION")
            ground_truth: True labels

        Returns:
            Dictionary of computed metrics
        """
        # Clean data
        valid_pairs = [
            (p, g)
            for p, g in zip(predictions, ground_truth)
            if p is not None and g is not None
        ]

        if not valid_pairs:
            return {"error": MetricResult("error", "no_valid_pairs")}

        # Convert to binary
        pred_binary = [1 if p == "ENTAILMENT" else 0 for p, _ in valid_pairs]
        true_binary = [1 if g == "ENTAILMENT" else 0 for _, g in valid_pairs]

        # Confusion matrix
        tp = sum(1 for p, t in zip(pred_binary, true_binary) if p == 1 and t == 1)
        tn = sum(1 for p, t in zip(pred_binary, true_binary) if p == 0 and t == 0)
        fp = sum(1 for p, t in zip(pred_binary, true_binary) if p == 1 and t == 0)
        fn = sum(1 for p, t in zip(pred_binary, true_binary) if p == 0 and t == 1)

        n = len(valid_pairs)

        results = {}

        accuracy = (tp + tn) / n if n > 0 else 0
        precision = tp / (tp + fp) if (tp + fp) > 0 else 0
        recall = tp / (tp + fn) if (tp + fn) > 0 else 0
        f1 = (
            2 * (precision * recall) / (precision + recall)
            if (precision + recall) > 0
            else 0
        )

        results["accuracy"] = MetricResult("accuracy", accuracy, n_samples=n)
        results["precision"] = MetricResult("precision", precision, n_samples=n)
        results["recall"] = MetricResult("recall", recall, n_samples=n)
        results["f1_score"] = MetricResult("f1_score", f1, n_samples=n)

        # Confusion matrix
        results["confusion_matrix"] = MetricResult(
            "confusion_matrix", {"tp": tp, "tn": tn, "fp": fp, "fn": fn}, n_samples=n
        )

        return results

    @staticmethod
    def compute_ranking_metrics(
        predictions: List[List[int]], ground_truth: List[List[int]]
    ) -> Dict[str, MetricResult]:
        """
        Compute comprehensive metrics for summary ranking task.

        Args:
            predictions: Model ranking predictions
            ground_truth: True rankings

        Returns:
            Dictionary of computed metrics
        """
        valid_pairs = [
            (p, g)
            for p, g in zip(predictions, ground_truth)
            if p is not None and g is not None and len(p) == len(g)
        ]

        if not valid_pairs:
            return {"error": MetricResult("error", "no_valid_pairs")}

        results = {}

        # Rank correlation metrics
        kendall_taus = []
        spearman_rhos = []

        for pred_rank, true_rank in valid_pairs:
            try:
                tau, _ = kendalltau(pred_rank, true_rank)
                if not np.isnan(tau):
                    kendall_taus.append(tau)

                rho, _ = spearmanr(pred_rank, true_rank)
                if not np.isnan(rho):
                    spearman_rhos.append(rho)
            except:
                continue

        if kendall_taus:
            results["kendall_tau"] = MetricResult(
                "kendall_tau",
                np.mean(kendall_taus),
                n_samples=len(kendall_taus),
                metadata={"std": np.std(kendall_taus), "values": kendall_taus},
            )

        if spearman_rhos:
            results["spearman_rho"] = MetricResult(
                "spearman_rho",
                np.mean(spearman_rhos),
                n_samples=len(spearman_rhos),
                metadata={"std": np.std(spearman_rhos), "values": spearman_rhos},
            )

        # Perfect ranking accuracy
        perfect_matches = sum(1 for p, g in valid_pairs if p == g)
        perfect_accuracy = perfect_matches / len(valid_pairs) if valid_pairs else 0

        results["perfect_ranking_accuracy"] = MetricResult(
            "perfect_ranking_accuracy", perfect_accuracy, n_samples=len(valid_pairs)
        )

        return results

    @staticmethod
    def compute_rating_metrics(
        predictions: List[float], ground_truth: List[float]
    ) -> Dict[str, MetricResult]:
        """
        Compute comprehensive metrics for consistency rating task.

        Args:
            predictions: Model rating predictions (0-100 scale)
            ground_truth: True ratings

        Returns:
            Dictionary of computed metrics
        """
        valid_pairs = [
            (p, g)
            for p, g in zip(predictions, ground_truth)
            if p is not None and g is not None
        ]

        if not valid_pairs:
            return {"error": MetricResult("error", "no_valid_pairs")}

        preds, truth = zip(*valid_pairs)
        pred_array = np.array(preds)
        true_array = np.array(truth)

        results = {}

        # Regression metrics
        mse = np.mean((pred_array - true_array) ** 2)
        rmse = np.sqrt(mse)
        mae = np.mean(np.abs(pred_array - true_array))

        # R-squared
        ss_res = np.sum((true_array - pred_array) ** 2)
        ss_tot = np.sum((true_array - np.mean(true_array)) ** 2)
        r2 = 1 - (ss_res / ss_tot) if ss_tot > 0 else 0

        results["mse"] = MetricResult("mse", mse, n_samples=len(valid_pairs))
        results["rmse"] = MetricResult("rmse", rmse, n_samples=len(valid_pairs))
        results["mae"] = MetricResult("mae", mae, n_samples=len(valid_pairs))
        results["r_squared"] = MetricResult("r_squared", r2, n_samples=len(valid_pairs))

        # Rating distribution analysis
        results["rating_distribution"] = MetricResult(
            "rating_distribution",
            {
                "pred_mean": np.mean(pred_array),
                "pred_std": np.std(pred_array),
                "true_mean": np.mean(true_array),
                "true_std": np.std(true_array),
                "pred_range": (np.min(pred_array), np.max(pred_array)),
                "true_range": (np.min(true_array), np.max(true_array)),
            },
            n_samples=len(valid_pairs),
        )

        return results


def compute_comprehensive_metrics(
    task_type: str,
    predictions: List[Any],
    ground_truth: List[Any],
    baseline_predictions: Optional[List[Any]] = None,
    analyzer: Optional[StatisticalAnalyzer] = None,
) -> Dict[str, MetricResult]:
    """
    Compute comprehensive metrics for any factuality evaluation task.

    Args:
        task_type: Type of task ("entailment_inference", "summary_ranking", "consistency_rating")
        predictions: Model predictions
        ground_truth: Ground truth labels
        baseline_predictions: Optional baseline predictions for comparison
        analyzer: Optional StatisticalAnalyzer instance

    Returns:
        Dictionary of comprehensive metrics

    Raises:
        ValueError: If task_type is not supported
    """
    if analyzer is None:
        analyzer = StatisticalAnalyzer()

    results = {}

    # Task-specific metrics
    if task_type == "entailment_inference":
        results.update(
            TaskSpecificMetrics.compute_entailment_metrics(predictions, ground_truth)
        )
    elif task_type == "summary_ranking":
        results.update(
            TaskSpecificMetrics.compute_ranking_metrics(predictions, ground_truth)
        )
    elif task_type == "consistency_rating":
        results.update(
            TaskSpecificMetrics.compute_rating_metrics(predictions, ground_truth)
        )
    else:
        raise ValueError(f"Unsupported task type: {task_type}")

    # Correlation analysis (for numeric tasks)
    if task_type in ["consistency_rating"]:
        try:
            correlation_results = analyzer.compute_correlation_analysis(
                predictions, ground_truth
            )
            results.update(
                {f"correlation_{k}": v for k, v in correlation_results.items()}
            )
        except Exception as e:
            logger.warning(f"Failed to compute correlation analysis: {e}")

    # Significance tests (if baseline provided)
    if baseline_predictions is not None:
        try:
            if task_type == "consistency_rating":
                significance_results = analyzer.compute_significance_tests(
                    predictions, baseline_predictions, ground_truth
                )
                results.update(
                    {f"significance_{k}": v for k, v in significance_results.items()}
                )

                effect_size_results = analyzer.compute_effect_sizes(
                    predictions, baseline_predictions, ground_truth
                )
                results.update(
                    {f"effect_size_{k}": v for k, v in effect_size_results.items()}
                )
        except Exception as e:
            logger.warning(f"Failed to compute significance tests: {e}")

    # Distribution analysis
    try:
        if isinstance(predictions[0], (int, float)):
            dist_results = analyzer.compute_distribution_analysis(predictions)
            results.update({f"distribution_{k}": v for k, v in dist_results.items()})
    except Exception as e:
        logger.warning(f"Failed to compute distribution analysis: {e}")

    return results


if __name__ == "__main__":
    """Test the metrics module."""
    logging.basicConfig(level=logging.INFO)

    # Test data
    predictions_ent = ["ENTAILMENT", "CONTRADICTION", "ENTAILMENT", "ENTAILMENT"]
    ground_truth_ent = ["ENTAILMENT", "ENTAILMENT", "ENTAILMENT", "CONTRADICTION"]

    predictions_rating = [85.0, 72.5, 90.0, 65.0]
    ground_truth_rating = [80.0, 75.0, 95.0, 60.0]

    # Test entailment metrics
    print("Testing entailment metrics:")
    ent_metrics = compute_comprehensive_metrics(
        "entailment_inference", predictions_ent, ground_truth_ent
    )
    for name, metric in ent_metrics.items():
        print(f"  {name}: {metric.value}")

    # Test rating metrics
    print("\nTesting rating metrics:")
    rating_metrics = compute_comprehensive_metrics(
        "consistency_rating", predictions_rating, ground_truth_rating
    )
    for name, metric in rating_metrics.items():
        print(f"  {name}: {metric.value}")
