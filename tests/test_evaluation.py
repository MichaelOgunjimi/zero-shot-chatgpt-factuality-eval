"""
Essential Evaluation Framework Tests
===================================

Tests critical evaluation metrics and comparison logic for thesis research.
Focuses on correlation analysis, performance metrics, and statistical validation.

Author: Michael Ogunjimi
Institution: University of Manchester
"""

import pytest
import numpy as np
from scipy.stats import pearsonr, spearmanr, kendalltau
from typing import List, Dict, Any


class EvaluationMetrics:
    """Essential evaluation metrics for factuality research"""
    
    @staticmethod
    def compute_accuracy(predictions: List[int], ground_truth: List[int]) -> float:
        """Compute accuracy for binary classification"""
        if len(predictions) != len(ground_truth):
            raise ValueError("Predictions and ground truth must have same length")
        
        correct = sum(p == g for p, g in zip(predictions, ground_truth))
        return correct / len(predictions)
    
    @staticmethod
    def compute_correlation(predictions: List[float], ground_truth: List[float]) -> Dict[str, float]:
        """Compute correlation metrics for rating tasks"""
        if len(predictions) != len(ground_truth):
            raise ValueError("Predictions and ground truth must have same length")
        
        # Remove None values
        valid_pairs = [(p, g) for p, g in zip(predictions, ground_truth) 
                      if p is not None and g is not None]
        
        if len(valid_pairs) < 2:
            return {"pearson": 0.0, "spearman": 0.0, "n_valid": len(valid_pairs)}
        
        preds, truth = zip(*valid_pairs)
        
        pearson_r, _ = pearsonr(preds, truth)
        spearman_r, _ = spearmanr(preds, truth)
        
        return {
            "pearson": pearson_r,
            "spearman": spearman_r, 
            "n_valid": len(valid_pairs)
        }
    
    @staticmethod
    def compute_ranking_correlation(predictions: List[List[int]], ground_truth: List[List[int]]) -> Dict[str, float]:
        """Compute ranking correlation metrics"""
        if len(predictions) != len(ground_truth):
            raise ValueError("Predictions and ground truth must have same length")
        
        kendall_scores = []
        spearman_scores = []
        
        for pred_rank, true_rank in zip(predictions, ground_truth):
            if len(pred_rank) == len(true_rank):
                # Kendall's tau
                kendall_tau, _ = kendalltau(pred_rank, true_rank)
                kendall_scores.append(kendall_tau)
                
                # Spearman's rho
                spearman_rho, _ = spearmanr(pred_rank, true_rank)
                spearman_scores.append(spearman_rho)
        
        return {
            "kendall_tau": np.mean(kendall_scores) if kendall_scores else 0.0,
            "spearman_rho": np.mean(spearman_scores) if spearman_scores else 0.0,
            "n_valid": len(kendall_scores)
        }
    
    @staticmethod
    def compute_mae(predictions: List[float], ground_truth: List[float]) -> float:
        """Compute Mean Absolute Error"""
        valid_pairs = [(p, g) for p, g in zip(predictions, ground_truth) 
                      if p is not None and g is not None]
        
        if not valid_pairs:
            return float('inf')
        
        preds, truth = zip(*valid_pairs)
        return np.mean(np.abs(np.array(preds) - np.array(truth)))


class BaselineComparator:
    """Essential baseline comparison functionality"""
    
    def __init__(self):
        self.metrics = EvaluationMetrics()
    
    def compare_binary_predictions(self, chatgpt_preds: List[int], baseline_preds: List[int], 
                                 ground_truth: List[int]) -> Dict[str, Any]:
        """Compare binary predictions between ChatGPT and baseline"""
        chatgpt_acc = self.metrics.compute_accuracy(chatgpt_preds, ground_truth)
        baseline_acc = self.metrics.compute_accuracy(baseline_preds, ground_truth)
        
        # Agreement between methods
        agreement = self.metrics.compute_accuracy(chatgpt_preds, baseline_preds)
        
        return {
            "chatgpt_accuracy": chatgpt_acc,
            "baseline_accuracy": baseline_acc,
            "agreement": agreement,
            "chatgpt_better": chatgpt_acc > baseline_acc
        }
    
    def compare_rating_predictions(self, chatgpt_preds: List[float], baseline_preds: List[float],
                                 ground_truth: List[float]) -> Dict[str, Any]:
        """Compare rating predictions between ChatGPT and baseline"""
        chatgpt_corr = self.metrics.compute_correlation(chatgpt_preds, ground_truth)
        baseline_corr = self.metrics.compute_correlation(baseline_preds, ground_truth)
        
        chatgpt_mae = self.metrics.compute_mae(chatgpt_preds, ground_truth)
        baseline_mae = self.metrics.compute_mae(baseline_preds, ground_truth)
        
        return {
            "chatgpt_correlation": chatgpt_corr,
            "baseline_correlation": baseline_corr,
            "chatgpt_mae": chatgpt_mae,
            "baseline_mae": baseline_mae,
            "chatgpt_better_correlation": chatgpt_corr["pearson"] > baseline_corr["pearson"],
            "chatgpt_better_mae": chatgpt_mae < baseline_mae
        }


class TestEvaluationMetrics:
    """Test essential evaluation metrics"""
    
    def test_accuracy_computation(self):
        """Test accuracy computation for binary tasks"""
        predictions = [1, 0, 1, 1, 0]
        ground_truth = [1, 0, 0, 1, 0]  # 4/5 correct: positions 0,1,3,4 match
        
        accuracy = EvaluationMetrics.compute_accuracy(predictions, ground_truth)
        
        assert accuracy == 0.8
    
    def test_accuracy_perfect_score(self):
        """Test perfect accuracy"""
        predictions = [1, 0, 1, 0]
        ground_truth = [1, 0, 1, 0]
        
        accuracy = EvaluationMetrics.compute_accuracy(predictions, ground_truth)
        
        assert accuracy == 1.0
    
    def test_accuracy_length_mismatch(self):
        """Test error handling for mismatched lengths"""
        with pytest.raises(ValueError, match="must have same length"):
            EvaluationMetrics.compute_accuracy([1, 0], [1, 0, 1])
    
    def test_correlation_computation(self):
        """Test correlation computation for rating tasks"""
        predictions = [85.0, 70.0, 95.0, 60.0]
        ground_truth = [90.0, 75.0, 90.0, 65.0]
        
        correlations = EvaluationMetrics.compute_correlation(predictions, ground_truth)
        
        assert "pearson" in correlations
        assert "spearman" in correlations
        assert correlations["n_valid"] == 4
        assert -1 <= correlations["pearson"] <= 1
        assert -1 <= correlations["spearman"] <= 1
    
    def test_correlation_with_none_values(self):
        """Test correlation handling with None values"""
        predictions = [85.0, None, 95.0, 60.0]
        ground_truth = [90.0, 75.0, 90.0, None]
        
        correlations = EvaluationMetrics.compute_correlation(predictions, ground_truth)
        
        assert correlations["n_valid"] == 2  # Only 2 valid pairs
    
    def test_ranking_correlation(self):
        """Test ranking correlation metrics"""
        predictions = [[1, 2, 3], [2, 1, 3]]
        ground_truth = [[1, 3, 2], [1, 2, 3]]
        
        correlations = EvaluationMetrics.compute_ranking_correlation(predictions, ground_truth)
        
        assert "kendall_tau" in correlations
        assert "spearman_rho" in correlations
        assert correlations["n_valid"] == 2
    
    def test_mae_computation(self):
        """Test Mean Absolute Error computation"""
        predictions = [85.0, 70.0, 95.0]
        ground_truth = [90.0, 75.0, 90.0]  # Errors: 5, 5, 5
        
        mae = EvaluationMetrics.compute_mae(predictions, ground_truth)
        
        assert mae == 5.0
    
    def test_mae_with_none_values(self):
        """Test MAE handling with None values"""
        predictions = [85.0, None, 95.0]
        ground_truth = [90.0, 75.0, 90.0]
        
        mae = EvaluationMetrics.compute_mae(predictions, ground_truth)
        
        assert mae == 5.0  # Based on 2 valid pairs


class TestBaselineComparator:
    """Test baseline comparison functionality"""
    
    @pytest.fixture
    def comparator(self):
        return BaselineComparator()
    
    def test_binary_comparison(self, comparator):
        """Test comparison of binary predictions"""
        chatgpt_preds = [1, 0, 1, 1, 0]
        baseline_preds = [1, 1, 0, 1, 0]
        ground_truth = [1, 0, 1, 1, 0]
        
        comparison = comparator.compare_binary_predictions(
            chatgpt_preds, baseline_preds, ground_truth
        )
        
        assert "chatgpt_accuracy" in comparison
        assert "baseline_accuracy" in comparison
        assert "agreement" in comparison
        assert "chatgpt_better" in comparison
        
        # ChatGPT should have perfect accuracy, baseline should have 0.6
        assert comparison["chatgpt_accuracy"] == 1.0
        assert comparison["baseline_accuracy"] == 0.6
        assert comparison["chatgpt_better"] is True
    
    def test_rating_comparison(self, comparator):
        """Test comparison of rating predictions"""
        chatgpt_preds = [85.0, 70.0, 95.0, 60.0]
        baseline_preds = [80.0, 75.0, 90.0, 70.0]
        ground_truth = [90.0, 75.0, 90.0, 65.0]
        
        comparison = comparator.compare_rating_predictions(
            chatgpt_preds, baseline_preds, ground_truth
        )
        
        assert "chatgpt_correlation" in comparison
        assert "baseline_correlation" in comparison
        assert "chatgpt_mae" in comparison
        assert "baseline_mae" in comparison
        assert "chatgpt_better_correlation" in comparison
        assert "chatgpt_better_mae" in comparison
    
    def test_equal_performance(self, comparator):
        """Test when ChatGPT and baseline have equal performance"""
        predictions = [1, 0, 1, 0]
        ground_truth = [1, 0, 1, 0]
        
        comparison = comparator.compare_binary_predictions(
            predictions, predictions, ground_truth
        )
        
        assert comparison["chatgpt_accuracy"] == comparison["baseline_accuracy"]
        assert comparison["agreement"] == 1.0  # Perfect agreement
        assert comparison["chatgpt_better"] is False  # Not better, equal


class TestStatisticalValidation:
    """Test statistical validation functionality"""
    
    def test_significant_difference_detection(self):
        """Test detection of statistically significant differences"""
        # Create clearly different performance
        high_performance = [1, 1, 1, 1, 1, 0, 1, 1, 1, 1]  # 90% accuracy
        low_performance = [1, 0, 0, 0, 0, 0, 1, 0, 0, 0]   # 30% accuracy
        ground_truth = [1, 1, 1, 1, 1, 1, 1, 1, 1, 1]     # All positive
        
        metrics = EvaluationMetrics()
        
        high_acc = metrics.compute_accuracy(high_performance, ground_truth)
        low_acc = metrics.compute_accuracy(low_performance, ground_truth)
        
        # Should detect significant difference
        assert high_acc > low_acc
        assert (high_acc - low_acc) > 0.5  # Large difference
    
    def test_correlation_strength_interpretation(self):
        """Test interpretation of correlation strength"""
        # Strong positive correlation
        strong_preds = [10, 20, 30, 40, 50]
        strong_truth = [12, 22, 32, 42, 52]
        
        # Weak correlation
        weak_preds = [10, 50, 20, 40, 30]
        weak_truth = [12, 22, 32, 42, 52]
        
        metrics = EvaluationMetrics()
        
        strong_corr = metrics.compute_correlation(strong_preds, strong_truth)
        weak_corr = metrics.compute_correlation(weak_preds, weak_truth)
        
        assert strong_corr["pearson"] > weak_corr["pearson"]
        assert strong_corr["pearson"] > 0.8  # Should be strong correlation