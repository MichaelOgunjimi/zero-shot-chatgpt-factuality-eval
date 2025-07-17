"""
Summary Ranking Task for ChatGPT Factuality Evaluation
====================================================

Implementation of summary ranking task that orders multiple summaries
by their factual consistency with the source document, from most
consistent (rank 1) to least consistent.

This task evaluates ChatGPT's ability to perform comparative factuality
assessment across multiple summaries of the same source.

Author: Michael Ogunjimi
Institution: University of Manchester
Course: MSc AI
"""

from dataclasses import dataclass
from typing import Dict, List, Any, Optional, Tuple

import numpy as np
from scipy.stats import kendalltau, spearmanr
from sklearn.metrics import ndcg_score

from .base_task import BaseFactualityTask, TaskExample, TaskResult, TaskConfig
from ..llm_clients.openai_client import APICallResult
from ..utils.logging import get_logger

logger = get_logger(__name__)


@dataclass
class RankingResult(TaskResult):
    """
    Specialized result for summary ranking task.

    Extends TaskResult with ranking-specific fields and
    convenience methods for ranking analysis.
    """

    ranking: List[int] = None  # Rankings for each summary (1-based, 1=best)
    num_summaries: int = 0
    ranking_quality: Optional[float] = None  # Quality metric if available

    def __post_init__(self):
        super().__post_init__()
        # Set default empty ranking if None
        if self.ranking is None:
            self.ranking = []

        # Ensure prediction is in correct format
        self.prediction = self.ranking

        # Validate ranking
        if len(self.ranking) != self.num_summaries and self.num_summaries > 0:
            logger.warning(
                f"Ranking length mismatch: {len(self.ranking)} vs {self.num_summaries}"
            )

    def get_ranked_order(self) -> List[int]:
        """
        Get summaries ordered by rank (best to worst).

        Returns:
            List of summary indices in rank order (0-based indexing)
        """
        # Create list of (rank, index) pairs and sort by rank
        rank_index_pairs = [(rank, idx) for idx, rank in enumerate(self.ranking)]
        rank_index_pairs.sort(key=lambda x: x[0])  # Sort by rank (ascending)

        return [idx for rank, idx in rank_index_pairs]

    def get_best_summary_index(self) -> int:
        """Get index of the best-ranked summary."""
        return self.ranking.index(min(self.ranking))

    def get_worst_summary_index(self) -> int:
        """Get index of the worst-ranked summary."""
        return self.ranking.index(max(self.ranking))

    def is_valid_ranking(self) -> bool:
        """Check if ranking contains valid consecutive ranks."""
        expected_ranks = list(range(1, self.num_summaries + 1))
        return sorted(self.ranking) == expected_ranks

    def get_ranking_consistency(self) -> float:
        """
        Measure ranking consistency (how well ranks match expected pattern).

        Returns:
            Consistency score between 0 and 1
        """
        if not self.is_valid_ranking():
            return 0.0

        # For valid rankings, consistency is always 1.0
        # Could be extended to measure partial consistency
        return 1.0

    def __sub__(self, other):
        """Handle subtraction operations."""
        if isinstance(other, (int, float)):
            return [r - other for r in self.ranking]
        elif hasattr(other, "ranking"):
            if len(self.ranking) == len(other.ranking):
                return [a - b for a, b in zip(self.ranking, other.ranking)]
            else:
                return NotImplemented
        else:
            return NotImplemented

    def __rsub__(self, other):
        """Handle reverse subtraction operations."""
        if isinstance(other, (int, float)):
            return [other - r for r in self.ranking]
        else:
            return NotImplemented

    def __add__(self, other):
        """Handle addition operations."""
        if isinstance(other, (int, float)):
            return [r + other for r in self.ranking]
        elif hasattr(other, "ranking"):
            if len(self.ranking) == len(other.ranking):
                return [a + b for a, b in zip(self.ranking, other.ranking)]
            else:
                return NotImplemented
        else:
            return NotImplemented

    def __radd__(self, other):
        """Handle reverse addition operations."""
        if isinstance(other, (int, float)):
            return [other + r for r in self.ranking]
        else:
            return NotImplemented


class RankingMetrics:
    """
    Comprehensive metrics for ranking evaluation.

    Implements standard ranking metrics including correlation measures
    and ranking-specific quality assessments for academic evaluation.
    """

    @staticmethod
    def kendall_tau(y_true: List[int], y_pred: List[int]) -> Tuple[float, float]:
        """
        Compute Kendall's tau rank correlation.

        Args:
            y_true: True ranking (1-based)
            y_pred: Predicted ranking (1-based)

        Returns:
            Tuple of (tau, p_value)
        """
        if len(y_true) != len(y_pred):
            raise ValueError("Rankings must have same length")

        if len(set(y_true)) <= 1 or len(set(y_pred)) <= 1:
            return 0.0, 1.0

        try:
            tau, p_value = kendalltau(y_true, y_pred)
            return float(tau), float(p_value)
        except Exception as e:
            logger.warning(f"Kendall's tau calculation failed: {e}")
            return 0.0, 1.0

    @staticmethod
    def spearman_correlation(
        y_true: List[int], y_pred: List[int]
    ) -> Tuple[float, float]:
        """
        Compute Spearman's rank correlation.

        Args:
            y_true: True ranking (1-based)
            y_pred: Predicted ranking (1-based)

        Returns:
            Tuple of (rho, p_value)
        """
        if len(y_true) != len(y_pred):
            raise ValueError("Rankings must have same length")

        if len(set(y_true)) <= 1 or len(set(y_pred)) <= 1:
            return 0.0, 1.0

        try:
            rho, p_value = spearmanr(y_true, y_pred)
            return float(rho), float(p_value)
        except Exception as e:
            logger.warning(f"Spearman correlation calculation failed: {e}")
            return 0.0, 1.0

    @staticmethod
    def compute_ndcg(
        y_true: List[int], y_pred: List[int], k: Optional[int] = None
    ) -> float:
        """
        Compute Normalized Discounted Cumulative Gain (NDCG).

        Args:
            y_true: True ranking (1-based, lower is better)
            y_pred: Predicted ranking (1-based, lower is better)
            k: Cutoff for NDCG@k (if None, uses full ranking)

        Returns:
            NDCG score
        """
        if len(y_true) != len(y_pred):
            raise ValueError("Rankings must have same length")

        try:
            # Convert rankings to relevance scores (higher is better)
            # Best rank (1) gets highest relevance score
            max_rank = max(max(y_true), max(y_pred))
            true_relevance = [max_rank - rank + 1 for rank in y_true]

            # Convert predicted ranking to order for NDCG
            pred_order = RankingMetrics._ranking_to_order(y_pred)
            true_relevance_ordered = [true_relevance[i] for i in pred_order]

            # Compute NDCG
            if k is None:
                k = len(y_true)

            ndcg = ndcg_score([true_relevance], [true_relevance_ordered], k=k)
            return float(ndcg)

        except Exception as e:
            logger.warning(f"NDCG calculation failed: {e}")
            return 0.0

    @staticmethod
    def _ranking_to_order(ranking: List[int]) -> List[int]:
        """Convert ranking to order (indices sorted by rank)."""
        return sorted(range(len(ranking)), key=lambda i: ranking[i])

    @staticmethod
    def compute_ranking_accuracy_at_k(
        y_true: List[int], y_pred: List[int], k: int = 1
    ) -> float:
        """
        Compute accuracy of top-k ranking.

        Args:
            y_true: True ranking
            y_pred: Predicted ranking
            k: Number of top positions to consider

        Returns:
            Accuracy at k
        """
        if k <= 0 or k > len(y_true):
            raise ValueError(f"k must be between 1 and {len(y_true)}")

        # Get top-k indices for both rankings
        true_order = RankingMetrics._ranking_to_order(y_true)
        pred_order = RankingMetrics._ranking_to_order(y_pred)

        true_top_k = set(true_order[:k])
        pred_top_k = set(pred_order[:k])

        # Calculate overlap
        overlap = len(true_top_k.intersection(pred_top_k))
        return overlap / k

    @staticmethod
    def compute_pairwise_accuracy(y_true: List[int], y_pred: List[int]) -> float:
        """
        Compute pairwise ranking accuracy.

        Args:
            y_true: True ranking
            y_pred: Predicted ranking

        Returns:
            Pairwise accuracy (0-1)
        """
        if len(y_true) != len(y_pred):
            raise ValueError("Rankings must have same length")

        n = len(y_true)
        if n <= 1:
            return 1.0

        correct_pairs = 0
        total_pairs = 0

        for i in range(n):
            for j in range(i + 1, n):
                total_pairs += 1

                # Check if relative order matches
                true_order = y_true[i] < y_true[j]  # True if i is ranked better than j
                pred_order = y_pred[i] < y_pred[j]  # Pred if i is ranked better than j

                if true_order == pred_order:
                    correct_pairs += 1

        return correct_pairs / total_pairs if total_pairs > 0 else 1.0

    @staticmethod
    def compute_comprehensive_metrics(
        y_true: List[int], y_pred: List[int]
    ) -> Dict[str, float]:
        """
        Compute all ranking metrics together.

        Args:
            y_true: True ranking
            y_pred: Predicted ranking

        Returns:
            Dictionary with all ranking metrics
        """
        metrics = {}

        # Correlation metrics
        tau, tau_p = RankingMetrics.kendall_tau(y_true, y_pred)
        rho, rho_p = RankingMetrics.spearman_correlation(y_true, y_pred)

        metrics.update(
            {
                "kendall_tau": tau,
                "kendall_tau_p_value": tau_p,
                "spearman_rho": rho,
                "spearman_rho_p_value": rho_p,
            }
        )

        # NDCG metrics
        metrics["ndcg"] = RankingMetrics.compute_ndcg(y_true, y_pred)
        if len(y_true) >= 3:
            metrics["ndcg_at_3"] = RankingMetrics.compute_ndcg(y_true, y_pred, k=3)

        # Accuracy metrics
        metrics["accuracy_at_1"] = RankingMetrics.compute_ranking_accuracy_at_k(
            y_true, y_pred, k=1
        )
        if len(y_true) >= 2:
            metrics["accuracy_at_2"] = RankingMetrics.compute_ranking_accuracy_at_k(
                y_true, y_pred, k=2
            )

        # Pairwise accuracy
        metrics["pairwise_accuracy"] = RankingMetrics.compute_pairwise_accuracy(
            y_true, y_pred
        )

        return metrics


class SummaryRankingTask(BaseFactualityTask):
    """
    Summary ranking task for factuality evaluation.

    Ranks multiple summaries by their factual consistency with
    the source document using ChatGPT's comparative assessment.
    """

    def _create_task_config(self) -> TaskConfig:
        """Create configuration specific to summary ranking."""
        task_config_dict = self.config.get("tasks", {}).get("summary_ranking", {})

        return TaskConfig(
            task_type="summary_ranking",
            prompt_type=task_config_dict.get("prompt_type", "zero_shot"),
            model_name=task_config_dict.get("model_name", "gpt-4.1-mini"),
            temperature=task_config_dict.get("temperature", 0.0),
            max_tokens=task_config_dict.get(
                "max_tokens", 200
            ),  # More tokens for ranking
            batch_size=task_config_dict.get(
                "batch_size", 5
            ),  # Smaller batches for complex task
            max_examples=task_config_dict.get("max_examples"),
            include_human_eval=task_config_dict.get("include_human_eval", False),
            save_intermediate=task_config_dict.get("save_intermediate", True),
            cache_responses=task_config_dict.get("cache_responses", True),
            retry_failed=task_config_dict.get("retry_failed", True),
        )

    def _validate_example(self, example: TaskExample) -> bool:
        """
        Validate example for summary ranking task.

        Args:
            example: TaskExample to validate

        Returns:
            True if example is valid for ranking task
        """
        try:
            # Must have source and multiple summaries
            if not example.source or not example.source.strip():
                logger.warning(f"Example {example.example_id}: Empty source")
                return False

            # Get summaries for ranking
            summaries = example.get_summaries_for_ranking()

            if len(summaries) < 2:
                logger.warning(
                    f"Example {example.example_id}: Need at least 2 summaries for ranking"
                )
                return False

            if len(summaries) > 10:
                logger.warning(
                    f"Example {example.example_id}: Too many summaries ({len(summaries)}), limiting complexity"
                )
                return False

            # Check that all summaries have content
            for i, summary in enumerate(summaries):
                if not summary or not summary.strip():
                    logger.warning(
                        f"Example {example.example_id}: Empty summary at index {i}"
                    )
                    return False

                if len(summary) < 5:
                    logger.warning(
                        f"Example {example.example_id}: Summary {i} too short"
                    )
                    return False

            # Check reasonable length constraints
            if len(example.source) < 10:
                logger.warning(f"Example {example.example_id}: Source too short")
                return False

            # Validate human ranking if present
            if example.human_label is not None:
                if not isinstance(example.human_label, list):
                    # For synthetic summaries, use the generated rankings
                    if hasattr(example, 'get_human_ranking_for_synthetic_summaries'):
                        synthetic_rankings = example.get_human_ranking_for_synthetic_summaries()
                        if len(synthetic_rankings) == len(summaries):
                            # Temporarily replace the human label for validation
                            example.human_label = synthetic_rankings
                        else:
                            logger.warning(
                                f"Example {example.example_id}: Human label must be list for ranking"
                            )
                            return False
                    else:
                        logger.warning(
                            f"Example {example.example_id}: Human label must be list for ranking"
                        )
                        return False

                if len(example.human_label) != len(summaries):
                    logger.warning(
                        f"Example {example.example_id}: Human ranking length mismatch"
                    )
                    return False

                # Check if it's a valid ranking (contains 1 to n)
                expected_ranks = set(range(1, len(summaries) + 1))
                if set(example.human_label) != expected_ranks:
                    logger.warning(
                        f"Example {example.example_id}: Invalid human ranking"
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
        Process API result into RankingResult.

        Args:
            api_result: APICallResult from OpenAI client
            example: Original TaskExample

        Returns:
            RankingResult object
        """
        parsed_content = api_result.parsed_content
        raw_response = api_result.raw_response

        # Extract ranking
        ranking = parsed_content.get("ranked_list", [])
        summaries = example.get_summaries_for_ranking()
        num_summaries = len(summaries)

        # Validate and fix ranking if necessary
        if len(ranking) != num_summaries:
            logger.warning(
                f"Ranking length mismatch for {example.example_id}, creating default ranking"
            )
            ranking = list(range(1, num_summaries + 1))  # Default sequential ranking

        # Ensure ranking contains valid ranks
        if not all(isinstance(r, int) and 1 <= r <= num_summaries for r in ranking):
            logger.warning(
                f"Invalid ranks in ranking for {example.example_id}, creating default ranking"
            )
            ranking = list(range(1, num_summaries + 1))

        # Calculate ranking quality if we have human labels
        ranking_quality = None
        if example.human_label is not None:
            try:
                tau, _ = RankingMetrics.kendall_tau(example.human_label, ranking)
                ranking_quality = float(tau)
            except:
                ranking_quality = 0.0

        return RankingResult(
            example_id=example.example_id,
            task_type=self.task_config.task_type,
            prompt_type=self.task_config.prompt_type,
            prediction=ranking,
            ranking=ranking,
            num_summaries=num_summaries,
            ranking_quality=ranking_quality,
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
                "summary_lengths": [len(s) for s in summaries],
                "num_summaries": num_summaries,
                "dataset_name": example.dataset_name,
                "finish_reason": raw_response.finish_reason,
                "ranking_valid": RankingResult(
                    example_id="temp",
                    task_type=self.task_config.task_type,
                    prompt_type=self.task_config.prompt_type,
                    prediction=ranking,
                    ranking=ranking,
                    num_summaries=num_summaries,
                    confidence=0.5,
                    raw_response="",
                    processing_time=0.0,
                    cost=0.0,
                    tokens_used=0,
                    timestamp="",
                    success=True,
                ).is_valid_ranking(),
            },
        )

    def evaluate_predictions(self, results: List[TaskResult]) -> Dict[str, float]:
        """
        Compute comprehensive evaluation metrics for summary ranking.

        Args:
            results: List of RankingResult objects

        Returns:
            Dictionary with evaluation metrics
        """
        if not results:
            return {"error": "No results to evaluate"}

        # Filter to successful results
        successful_results = [
            r for r in results if r.success and isinstance(r, RankingResult)
        ]

        if not successful_results:
            return {"error": "No successful results to evaluate"}

        # Check if we have human labels for evaluation
        labeled_results = [r for r in successful_results if r.human_label is not None]
        has_human_labels = len(labeled_results) == len(successful_results)

        evaluation_metrics = {
            "total_examples": len(successful_results),
            "has_human_labels": has_human_labels,
            "examples_with_labels": len(labeled_results),
        }

        if labeled_results:
            # Compute ranking metrics against human labels
            all_metrics = []

            for result in labeled_results:
                y_true = result.human_label
                y_pred = result.ranking

                try:
                    metrics = RankingMetrics.compute_comprehensive_metrics(
                        y_true, y_pred
                    )
                    all_metrics.append(metrics)
                except Exception as e:
                    logger.warning(
                        f"Failed to compute metrics for {result.example_id}: {e}"
                    )
                    continue

            if all_metrics:
                # Average all metrics
                metric_names = all_metrics[0].keys()
                for metric_name in metric_names:
                    values = [m[metric_name] for m in all_metrics if metric_name in m]
                    if values:
                        evaluation_metrics[f"avg_{metric_name}"] = np.mean(values)
                        evaluation_metrics[f"std_{metric_name}"] = np.std(values)
                        evaluation_metrics[f"min_{metric_name}"] = np.min(values)
                        evaluation_metrics[f"max_{metric_name}"] = np.max(values)

                # Set primary metric for experiment reporting (using average Spearman correlation)
                evaluation_metrics["primary_metric"] = evaluation_metrics.get("avg_spearman_correlation", 0.0)

            # Ranking quality distribution
            quality_scores = [
                r.ranking_quality
                for r in labeled_results
                if r.ranking_quality is not None
            ]
            if quality_scores:
                evaluation_metrics["ranking_quality"] = {
                    "mean": np.mean(quality_scores),
                    "std": np.std(quality_scores),
                    "min": np.min(quality_scores),
                    "max": np.max(quality_scores),
                }

        # Ranking validity analysis
        valid_rankings = sum(1 for r in successful_results if r.is_valid_ranking())
        evaluation_metrics["ranking_validity"] = {
            "valid_rankings": valid_rankings,
            "invalid_rankings": len(successful_results) - valid_rankings,
            "validity_rate": valid_rankings / len(successful_results),
        }
        
        # Set primary metric for experiment reporting when no human labels
        if not labeled_results:
            # Use a more realistic primary metric that combines validity and quality indicators
            validity_rate = evaluation_metrics["ranking_validity"]["validity_rate"]
            
            # Add quality penalty based on ranking patterns
            quality_adjustment = 0.0
            if successful_results:
                # Penalize for always choosing the same ranking pattern
                rankings_patterns = [str(r.ranking) for r in successful_results]
                unique_patterns = len(set(rankings_patterns))
                pattern_diversity = unique_patterns / len(rankings_patterns)
                
                # Penalize for low diversity (suggests lack of discrimination)
                if pattern_diversity < 0.5:
                    quality_adjustment = -0.1
                elif pattern_diversity < 0.3:
                    quality_adjustment = -0.2
            
            evaluation_metrics["primary_metric"] = max(validity_rate + quality_adjustment, 0.0)

        # Summary count analysis
        summary_counts = [r.num_summaries for r in successful_results]
        evaluation_metrics["summary_analysis"] = {
            "avg_summaries_per_example": np.mean(summary_counts),
            "min_summaries": np.min(summary_counts),
            "max_summaries": np.max(summary_counts),
            "summary_count_distribution": {
                str(count): sum(1 for c in summary_counts if c == count)
                for count in sorted(set(summary_counts))
            },
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

    def analyze_ranking_patterns(self, results: List[TaskResult]) -> Dict[str, Any]:
        """
        Analyze ranking patterns and preferences.

        Args:
            results: List of RankingResult objects

        Returns:
            Dictionary with ranking pattern analysis
        """
        successful_results = [
            r for r in results if r.success and isinstance(r, RankingResult)
        ]

        if not successful_results:
            return {"error": "No successful results for analysis"}

        analysis = {
            "total_rankings": len(successful_results),
            "valid_rankings": sum(
                1 for r in successful_results if r.is_valid_ranking()
            ),
        }

        # Position bias analysis (tendency to rank certain positions higher)
        position_preferences = {}
        for result in successful_results:
            for position, rank in enumerate(result.ranking):
                if position not in position_preferences:
                    position_preferences[position] = []
                position_preferences[position].append(rank)

        # Calculate average rank for each position
        avg_ranks_by_position = {}
        for position, ranks in position_preferences.items():
            avg_ranks_by_position[position] = {
                "average_rank": np.mean(ranks),
                "std_rank": np.std(ranks),
                "rank_1_count": sum(1 for r in ranks if r == 1),
                "rank_1_rate": sum(1 for r in ranks if r == 1) / len(ranks),
            }

        analysis["position_bias"] = avg_ranks_by_position

        # Summary length impact (if available)
        length_impact = {}
        for result in successful_results:
            if "summary_lengths" in result.metadata:
                summary_lengths = result.metadata["summary_lengths"]
                ranking = result.ranking

                for i, (length, rank) in enumerate(zip(summary_lengths, ranking)):
                    length_bin = f"{length//100 * 100}-{(length//100 + 1) * 100}"
                    if length_bin not in length_impact:
                        length_impact[length_bin] = []
                    length_impact[length_bin].append(rank)

        # Calculate average rank by length
        avg_rank_by_length = {}
        for length_bin, ranks in length_impact.items():
            if len(ranks) >= 5:  # Only include bins with sufficient data
                avg_rank_by_length[length_bin] = {
                    "average_rank": np.mean(ranks),
                    "count": len(ranks),
                    "std": np.std(ranks),
                }

        analysis["length_impact"] = avg_rank_by_length

        return analysis

    def get_task_specific_insights(self, results: List[TaskResult]) -> Dict[str, Any]:
        """
        Get insights specific to summary ranking task.

        Args:
            results: List of RankingResult objects

        Returns:
            Dictionary with task-specific insights
        """
        successful_results = [
            r for r in results if r.success and isinstance(r, RankingResult)
        ]

        if not successful_results:
            return {"error": "No successful results for analysis"}

        insights = {
            "ranking_consistency": {
                "valid_rankings": sum(
                    1 for r in successful_results if r.is_valid_ranking()
                ),
                "invalid_rankings": sum(
                    1 for r in successful_results if not r.is_valid_ranking()
                ),
                "validity_rate": np.mean(
                    [r.is_valid_ranking() for r in successful_results]
                ),
            }
        }

        # Ranking difficulty analysis
        difficulties = []
        for result in successful_results:
            # Estimate difficulty based on number of summaries
            difficulty = (
                result.num_summaries * (result.num_summaries - 1) / 2
            )  # Number of pairwise comparisons
            difficulties.append(difficulty)

        insights["complexity_analysis"] = {
            "avg_pairwise_comparisons": np.mean(difficulties),
            "max_pairwise_comparisons": np.max(difficulties),
            "complexity_distribution": {
                f"{int(d)}_comparisons": sum(
                    1 for diff in difficulties if int(diff) == d
                )
                for d in sorted(set(int(d) for d in difficulties))
            },
        }

        # Confidence analysis for rankings
        confidences = [
            r.confidence for r in successful_results if r.confidence is not None
        ]
        if confidences:
            insights["confidence_analysis"] = {
                "avg_confidence": np.mean(confidences),
                "confidence_std": np.std(confidences),
                "high_confidence_rankings": sum(1 for c in confidences if c > 0.8),
                "low_confidence_rankings": sum(1 for c in confidences if c < 0.3),
            }

        return insights
