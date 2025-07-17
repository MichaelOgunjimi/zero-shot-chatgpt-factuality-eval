#!/usr/bin/env python3
"""
SOTA Baseline Comparison Experiment
==================================

This script compares ChatGPT's factuality evaluation performance with
state-of-the-art baseline methods including:
- FactCC (BERT-based factual consistency classifier)
- BERTScore (Contextual embedding similarity)
- ROUGE (N-gram overlap metrics)

The experiment provides comprehensive correlation analysis and statistical
comparison between ChatGPT and traditional evaluation methods.

Usage:
    # As script
    python experiments/sota_comparison.py --config config/default.yaml
    python experiments/sota_comparison.py --baseline factcc --task entailment_inference
    python experiments/sota_comparison.py --quick-test
    
    # As module
    python -m experiments.sota_comparison --config config/default.yaml

Author: Michael Ogunjimi
Institution: University of Manchester
Course: MSc AI
"""

import argparse
import asyncio
import json
import logging
import sys
import time
from datetime import datetime
from pathlib import Path
from typing import Dict, List, Any, Optional, Tuple

import numpy as np
from scipy.stats import pearsonr, spearmanr
import plotly.graph_objects as go
from plotly.subplots import make_subplots

# Add project root to path for both script and module execution
if __name__ == "__main__":
    # Script execution - add parent directory
    sys.path.insert(0, str(Path(__file__).parent.parent))
else:
    # Module execution - add project root if needed
    current_dir = Path(__file__).resolve().parent.parent
    if str(current_dir) not in sys.path:
        sys.path.insert(0, str(current_dir))

try:
    # Import project modules with error handling
    from src.utils import (
        setup_experiment_logger,
        create_visualization_engine,
        load_config,
        get_config,
        setup_reproducibility,
        create_output_directories,
        validate_api_keys
    )
    from src.tasks import create_task, get_supported_tasks
    from src.data import load_processed_dataset, get_available_datasets
    from src.evaluation import EvaluatorFactory
    from src.baselines import (
        create_baseline,
        get_available_baselines,
        compare_with_chatgpt,
        BaselineComparator,
        create_all_baselines
    )
except ImportError as e:
    print(f"Error importing required modules: {e}")
    print("Please ensure you're running from the project root directory.")
    print("Try: cd /path/to/factuality-evaluation && python experiments/sota_comparison.py")
    sys.exit(1)


class SOTAComparisonExperiment:
    """
    Experimental framework for comparing ChatGPT with SOTA baseline methods.
    
    This class orchestrates comprehensive comparisons between ChatGPT's zero-shot
    factuality evaluation and established baseline metrics, providing correlation
    analysis and performance benchmarking.
    """
    
    def __init__(self, config_path: str, experiment_name: str = None, log_dir: str = None, output_dir: str = None):
        """Initialize the SOTA comparison experiment."""
        # Load configuration
        self.config = load_config(config_path)
        
        # Set up experiment tracking
        self.experiment_name = experiment_name or f"sota_comparison_{datetime.now().strftime('%Y%m%d_%H%M%S')}"
        
        # Use provided output_dir or create default
        if output_dir:
            self.output_dir = Path(output_dir)
        else:
            self.output_dir = Path(f"results/experiments/{self.experiment_name}")
        
        # Create output directories only if not using custom output_dir
        if not output_dir:
            create_output_directories(self.config)
        
        # Set up logging - reduced verbosity with custom log_dir if provided
        self.experiment_logger = setup_experiment_logger(
            self.experiment_name,
            self.config,
            log_dir
        )
        self.logger = self.experiment_logger.logger
        
        # Reduce external library logging
        self._configure_logging_levels()
        
        # Set up reproducibility
        setup_reproducibility(self.config)
        
        # Validate API keys
        validate_api_keys(self.config)
        
        # Initialize visualization engine
        self.visualization_engine = create_visualization_engine(self.config)
        
        # Initialize baseline comparator
        self.baseline_comparator = BaselineComparator(self.config)
        
        self.logger.info(f"Initialized SOTA comparison experiment: {self.experiment_name}")
        
    def _configure_logging_levels(self):
        """Configure logging levels to reduce verbosity."""
        # Reduce external library logging
        logging.getLogger("httpx").setLevel(logging.WARNING)
        logging.getLogger("openai").setLevel(logging.WARNING)
        logging.getLogger("choreographer").setLevel(logging.WARNING)
        logging.getLogger("kaleido").setLevel(logging.WARNING)
        logging.getLogger("progress").setLevel(logging.WARNING)
        logging.getLogger("cost_tracker").setLevel(logging.WARNING)
        logging.getLogger("PromptManager").setLevel(logging.WARNING)
        logging.getLogger("OpenAIClient").setLevel(logging.WARNING)
        logging.getLogger("transformers").setLevel(logging.ERROR)
        logging.getLogger("absl").setLevel(logging.ERROR)
        logging.getLogger("torch").setLevel(logging.ERROR)
        
        # Results storage
        self.results = {
            'experiment_metadata': {
                'name': self.experiment_name,
                'timestamp': datetime.now().isoformat(),
                'config': self.config.to_dict(),
                'experiment_type': 'sota_comparison'
            },
            'chatgpt_results': {},
            'baseline_results': {},
            'correlation_analysis': {},
            'performance_comparison': {},
            'statistical_analysis': {},
            'cost_analysis': {}
        }
        
        self.logger.info(f"Initialized SOTA comparison experiment: {self.experiment_name}")
    
    async def run_sota_comparison(
        self,
        tasks: List[str] = None,
        datasets: List[str] = None,
        baselines: List[str] = None,
        sample_size: int = None,
        prompt_type: str = "zero_shot"
    ) -> Dict[str, Any]:
        """
        Run comprehensive SOTA baseline comparison.
        
        Args:
            tasks: List of tasks to evaluate (default: entailment_inference, consistency_rating)
            datasets: List of datasets to use (default: cnn_dailymail, xsum)
            baselines: List of baseline methods (default: all available)
            sample_size: Number of examples per dataset (default: from config)
            prompt_type: ChatGPT prompt type to use
            
        Returns:
            Complete comparison results with correlation analysis
        """
        self.logger.info("Starting SOTA baseline comparison experiment")
        
        # Set defaults (note: summary_ranking excluded as most baselines don't support it)
        if tasks is None:
            tasks = ['entailment_inference', 'consistency_rating']
        if datasets is None:
            datasets = ['cnn_dailymail', 'xsum']
        if baselines is None:
            baselines = get_available_baselines()
        if sample_size is None:
            sample_size = self.config.get('experiments.main_experiments.sota_comparison.sample_size', 300)
        
        try:
            # Phase 1: Run ChatGPT evaluations
            await self._run_chatgpt_evaluations(tasks, datasets, sample_size, prompt_type)
            
            # Phase 2: Run baseline evaluations
            await self._run_baseline_evaluations(tasks, datasets, baselines, sample_size)
            
            # Phase 3: Compute correlations
            await self._compute_correlation_analysis()
            
            # Phase 4: Performance comparison analysis
            await self._analyze_performance_comparison()
            
            # Phase 5: Statistical significance testing
            await self._perform_statistical_analysis()
            
            # Phase 6: Generate visualizations
            await self._generate_comparison_visualizations()
            
            # Phase 7: Save results and generate report
            await self._save_results()
            
            self.logger.info("SOTA comparison experiment completed successfully")
            return self.results
            
        except Exception as e:
            self.logger.error(f"SOTA comparison experiment failed: {e}")
            raise
    
    async def _run_chatgpt_evaluations(
        self,
        tasks: List[str],
        datasets: List[str],
        sample_size: int,
        prompt_type: str
    ):
        """Run ChatGPT evaluations for comparison with baselines."""
        print("\n🤖 Running ChatGPT Evaluations")
        print("=" * 40)
        
        total_evaluations = len(tasks) * len(datasets)
        current_eval = 0
        
        total_cost = 0.0
        
        for task_name in tasks:
            print(f"\n⚡ Evaluating ChatGPT on task: {task_name}")
            self.results['chatgpt_results'][task_name] = {}
            
            # Create task instance
            task_config = self.config.to_dict()
            if "tasks" not in task_config:
                task_config["tasks"] = {}
            if task_name not in task_config["tasks"]:
                task_config["tasks"][task_name] = {}
            task_config["tasks"][task_name]["prompt_type"] = prompt_type
            
            task = create_task(task_name, task_config)
            
            for dataset_name in datasets:
                current_eval += 1
                print(f"📊 [{current_eval}/{total_evaluations}] {task_name} on {dataset_name}")
                
                try:
                    # Load processed dataset with synthetic labels
                    examples = load_processed_dataset(
                        dataset_name,
                        task_name,
                        max_examples=sample_size
                    )
                    
                    # Run ChatGPT evaluation
                    start_time = time.time()
                    predictions = await task.process_examples(examples)
                    processing_time = time.time() - start_time
                    
                    # Calculate cost
                    task_cost = getattr(task, 'total_cost', 0.0)
                    total_cost += task_cost
                    
                    # Evaluate performance using task's built-in evaluation
                    performance_metrics = task.evaluate_predictions(predictions)
                    
                    # Store results for baseline comparison
                    self.results['chatgpt_results'][task_name][dataset_name] = {
                        'predictions': predictions,
                        'examples': examples,  # Store for baseline comparison
                        'performance_metrics': performance_metrics,
                        'dataset_size': len(examples),
                        'processing_time': processing_time,
                        'cost': task_cost,
                        'prompt_type': prompt_type
                    }
                    
                    # Log progress
                    primary_metric = performance_metrics.get('primary_metric', 'N/A')
                    self.logger.info(
                        f"ChatGPT - {task_name} on {dataset_name}: "
                        f"Performance = {primary_metric}, "
                        f"Time = {processing_time:.2f}s, "
                        f"Cost = ${task_cost:.4f}"
                    )
                    
                except Exception as e:
                    self.logger.error(f"Failed to process ChatGPT {task_name} on {dataset_name}: {e}")
                    self.results['chatgpt_results'][task_name][dataset_name] = {
                        'error': str(e),
                        'status': 'failed'
                    }
        
        # Store ChatGPT cost
        self.results['cost_analysis']['chatgpt_cost'] = total_cost
        self.logger.info(f"ChatGPT evaluations completed. Total cost: ${total_cost:.4f}")
    
    async def _run_baseline_evaluations(
        self,
        tasks: List[str],
        datasets: List[str],
        baselines: List[str],
        sample_size: int
    ):
        """Run baseline method evaluations."""
        print("\n🔧 Running Baseline Evaluations")
        print("=" * 40)
        
        # Create baseline instances
        baseline_instances = create_all_baselines(self.config)
        
        for baseline_name in baselines:
            if baseline_name not in baseline_instances:
                self.logger.warning(f"Baseline {baseline_name} not available, skipping")
                continue
                
            print(f"\n🎯 Evaluating baseline: {baseline_name}")
            baseline = baseline_instances[baseline_name]
            self.results['baseline_results'][baseline_name] = {}
            
            for task_name in tasks:
                # Check if baseline supports this task
                if not baseline.supports_task(task_name):
                    print(f"   ⚠️  Baseline {baseline_name} does not support task {task_name}, skipping")
                    continue
                
                print(f"   📊 Running {baseline_name} on task: {task_name}")
                self.results['baseline_results'][baseline_name][task_name] = {}
                
                for dataset_name in datasets:
                    # Check if we have ChatGPT results for this combination
                    if (task_name not in self.results['chatgpt_results'] or 
                        dataset_name not in self.results['chatgpt_results'][task_name] or
                        'examples' not in self.results['chatgpt_results'][task_name][dataset_name]):
                        self.logger.warning(f"No ChatGPT results for {task_name}-{dataset_name}, skipping baseline")
                        continue
                    
                    try:
                        # Get examples from ChatGPT results to ensure consistency
                        chatgpt_data = self.results['chatgpt_results'][task_name][dataset_name]
                        examples = chatgpt_data['examples']
                        chatgpt_predictions = chatgpt_data['predictions']
                        
                        self.logger.info(f"Processing {baseline_name} on {dataset_name}")
                        
                        # Run baseline evaluation
                        start_time = time.time()
                        baseline_results = await self._evaluate_baseline_on_examples(
                            baseline, baseline_name, task_name, examples
                        )
                        processing_time = time.time() - start_time
                        
                        # Compare with ChatGPT
                        # Create proper format for comparison
                        baseline_results_dict = {baseline_name: baseline_results}
                        
                        # Convert TaskResult objects to dictionaries for comparison
                        chatgpt_predictions_dict = [
                            {
                                'example_id': result.example_id,
                                'prediction': result.prediction,
                                'confidence': result.confidence,
                                'task_type': result.task_type,
                                'prompt_type': result.prompt_type,
                                'raw_response': result.raw_response,
                                'processing_time': result.processing_time,
                                'cost': result.cost,
                                'tokens_used': result.tokens_used,
                                'timestamp': result.timestamp,
                                'success': result.success,
                                'error_message': result.error_message,
                                'human_label': result.human_label,
                                'metadata': result.metadata
                            }
                            for result in chatgpt_predictions
                        ]
                        
                        comparison_results = compare_with_chatgpt(
                            chatgpt_predictions_dict,
                            baseline_results_dict,
                            task_name,
                            self.config.to_dict()
                        )
                        
                        # Store results
                        self.results['baseline_results'][baseline_name][task_name][dataset_name] = {
                            'baseline_predictions': baseline_results,
                            'comparison_with_chatgpt': comparison_results,
                            'processing_time': processing_time,
                            'dataset_size': len(examples)
                        }
                        
                        # Log progress
                        # Extract correlation from the baseline's comparison result
                        baseline_comparison = comparison_results.get(baseline_name, {})
                        if task_name == "entailment_inference":
                            correlation = baseline_comparison.get('cohens_kappa', 'N/A')
                        elif task_name == "consistency_rating":
                            correlation = baseline_comparison.get('pearson_correlation', 'N/A')
                        elif task_name == "summary_ranking":
                            correlation = baseline_comparison.get('avg_spearman_rho', 'N/A')
                        else:
                            correlation = 'N/A'
                        
                        # Handle NaN values
                        if isinstance(correlation, float) and np.isnan(correlation):
                            correlation = 'N/A (insufficient data)'
                        
                        self.logger.info(
                            f"{baseline_name} - {task_name} on {dataset_name}: "
                            f"Correlation = {correlation}, "
                            f"Time = {processing_time:.2f}s"
                        )
                        
                    except Exception as e:
                        self.logger.error(f"Failed to process {baseline_name} on {task_name}-{dataset_name}: {e}")
                        self.results['baseline_results'][baseline_name][task_name][dataset_name] = {
                            'error': str(e),
                            'status': 'failed'
                        }
        
        self.logger.info("Baseline evaluations completed")
    
    async def _evaluate_baseline_on_examples(
        self,
        baseline,
        baseline_name: str,
        task_name: str,
        examples: List[Any]
    ) -> List[Any]:
        """Evaluate a baseline method on a list of examples."""
        predictions = []
        
        for i, example in enumerate(examples):
            try:
                if task_name == 'entailment_inference':
                    result = baseline.evaluate_entailment_inference(
                        example.source,
                        example.summary,
                        example_id=example.example_id
                    )
                elif task_name == 'consistency_rating':
                    result = baseline.evaluate_consistency_rating(
                        example.source,
                        example.summary,
                        example_id=example.example_id
                    )
                else:
                    continue  # Skip unsupported tasks
                
                # Store the actual BaselineResult object
                predictions.append(result)
                
            except Exception as e:
                self.logger.warning(f"Baseline {baseline_name} failed on example {i}: {e}")
                # Create a default BaselineResult for failed examples
                from src.baselines.sota_metrics import BaselineResult
                failed_result = BaselineResult(
                    baseline_name=baseline_name,
                    task_name=task_name,
                    example_id=example.example_id,
                    prediction=0,  # Default prediction
                    confidence=0.0,
                    raw_scores={'error': str(e)},
                    processing_time=0.0,
                    metadata={'failed': True}
                )
                predictions.append(failed_result)
        
        return predictions
    
    def _convert_to_binary(self, scores: List[float], task_name: str, baseline_name: str = None) -> List[int]:
        """Convert continuous scores to binary decisions using appropriate thresholds."""
        if task_name == 'consistency_rating':
            # Use baseline-specific thresholds for consistency rating
            if baseline_name == 'factcc':
                # FactCC typically produces very low scores (1-5), so use median-based threshold
                # Use the median of the actual scores as threshold
                if len(scores) > 0:
                    threshold = np.median(scores)
                else:
                    threshold = 3.0  # Default fallback
            else:
                # For ChatGPT and other baselines, use 50 as threshold
                threshold = 50.0
            
            return [1 if score > threshold else 0 for score in scores]
        elif task_name == 'entailment_inference':
            # For entailment inference: > 0.5 = ENTAILMENT (1), <= 0.5 = CONTRADICTION (0)
            return [1 if score > 0.5 else 0 for score in scores]
        else:
            # Default threshold of 0.5
            return [1 if score > 0.5 else 0 for score in scores]
    
    def _calculate_agreement_metrics(self, pred1: List[int], pred2: List[int], n_samples: int) -> Dict[str, Any]:
        """Calculate agreement metrics for binary predictions."""
        pred1 = np.array(pred1)
        pred2 = np.array(pred2)
        
        # Basic agreement
        agreement = np.sum(pred1 == pred2) / len(pred1)
        
        # Cohen's kappa - use manual calculation to avoid NaN issues
        p0 = agreement  # observed agreement
        p1_pred1 = np.mean(pred1)
        p1_pred2 = np.mean(pred2)
        p0_pred1 = 1 - p1_pred1
        p0_pred2 = 1 - p1_pred2
        pe = p1_pred1 * p1_pred2 + p0_pred1 * p0_pred2  # expected agreement
        
        if pe == 1.0:
            kappa = 0.0  # No chance-corrected agreement when expected agreement is perfect
        else:
            kappa = (p0 - pe) / (1 - pe)
        
        # Handle NaN values
        if np.isnan(kappa):
            kappa = 0.0
        
        # Confusion matrix-based metrics
        tp = np.sum((pred1 == 1) & (pred2 == 1))
        tn = np.sum((pred1 == 0) & (pred2 == 0))
        fp = np.sum((pred1 == 0) & (pred2 == 1))
        fn = np.sum((pred1 == 1) & (pred2 == 0))
        
        # Precision, recall, F1 treating pred1 as ground truth
        precision = tp / (tp + fp) if (tp + fp) > 0 else 0
        recall = tp / (tp + fn) if (tp + fn) > 0 else 0
        f1 = 2 * precision * recall / (precision + recall) if (precision + recall) > 0 else 0
        
        return {
            'agreement': agreement,
            'cohens_kappa': kappa,
            'precision': precision,
            'recall': recall,
            'f1_score': f1,
            'confusion_matrix': {
                'tp': int(tp), 'tn': int(tn), 'fp': int(fp), 'fn': int(fn)
            },
            'n_samples': n_samples
        }
    
    async def _compute_correlation_analysis(self):
        """Compute comprehensive correlation analysis between ChatGPT and baselines."""
        self.logger.info("Computing correlation analysis")
        
        correlation_analysis = {
            'pearson_correlations': {},
            'spearman_correlations': {},
            'agreement_metrics': {},  # New: for imbalanced datasets
            'correlation_summary': {},
            'method_rankings': {}
        }
        
        # Compute correlations for each task-dataset-baseline combination
        for baseline_name, baseline_results in self.results['baseline_results'].items():
            correlation_analysis['pearson_correlations'][baseline_name] = {}
            correlation_analysis['spearman_correlations'][baseline_name] = {}
            correlation_analysis['agreement_metrics'][baseline_name] = {}
            
            for task_name, task_results in baseline_results.items():
                correlation_analysis['pearson_correlations'][baseline_name][task_name] = {}
                correlation_analysis['spearman_correlations'][baseline_name][task_name] = {}
                correlation_analysis['agreement_metrics'][baseline_name][task_name] = {}
                
                for dataset_name, dataset_results in task_results.items():
                    if 'error' in dataset_results:
                        continue
                    
                    # Get predictions from both methods
                    chatgpt_data = self.results['chatgpt_results'][task_name][dataset_name]
                    
                    if 'predictions' not in chatgpt_data or 'baseline_predictions' not in dataset_results:
                        self.logger.warning(f"Missing predictions for {baseline_name}-{task_name}-{dataset_name}")
                        continue
                    
                    chatgpt_predictions = chatgpt_data['predictions']
                    baseline_predictions = dataset_results['baseline_predictions']
                    
                    # Extract numerical predictions for correlation
                    chatgpt_scores = self._extract_numerical_predictions(chatgpt_predictions, task_name)
                    baseline_scores = self._extract_numerical_predictions(baseline_predictions, task_name)
                    
                    self.logger.info(f"Extracted scores for {baseline_name}-{task_name}-{dataset_name}: ChatGPT={len(chatgpt_scores)}, Baseline={len(baseline_scores)}")
                    
                    # Ensure same length
                    min_length = min(len(chatgpt_scores), len(baseline_scores))
                    if min_length < 2:
                        self.logger.warning(f"Insufficient data for {baseline_name}-{task_name}-{dataset_name}: {min_length} samples")
                        continue
                    
                    chatgpt_scores = chatgpt_scores[:min_length]
                    baseline_scores = baseline_scores[:min_length]
                    
                    self.logger.info(f"Computing correlation for {baseline_name}-{task_name}-{dataset_name}: {min_length} samples")
                    
                    # Special handling for BERTScore entailment inference
                    if baseline_name == 'bertscore' and task_name == 'entailment_inference':
                        # Apply median threshold to create better variance
                        baseline_var = np.var(baseline_scores)
                        if baseline_var > 0:  # Only if there's some variance
                            threshold = np.median(baseline_scores)
                            self.logger.info(f"Applying median threshold {threshold:.4f} to BERTScore")
                            baseline_scores = [1.0 if score > threshold else 0.0 for score in baseline_scores]
                    
                    # Calculate correlations
                    try:
                        pearson_corr, pearson_p = pearsonr(chatgpt_scores, baseline_scores)
                        spearman_corr, spearman_p = spearmanr(chatgpt_scores, baseline_scores)
                        
                        # Check if correlation is valid
                        if np.isnan(pearson_corr) or np.isnan(spearman_corr):
                            raise ValueError("Correlation is NaN - likely due to zero variance")
                        
                        self.logger.info(f"Correlation computed: {baseline_name}-{task_name}-{dataset_name}: Pearson={pearson_corr:.4f}")
                        
                        correlation_analysis['pearson_correlations'][baseline_name][task_name][dataset_name] = {
                            'correlation': pearson_corr,
                            'p_value': pearson_p,
                            'n_samples': min_length
                        }
                        
                        correlation_analysis['spearman_correlations'][baseline_name][task_name][dataset_name] = {
                            'correlation': spearman_corr,
                            'p_value': spearman_p,
                            'n_samples': min_length
                        }
                        
                    except Exception as e:
                        self.logger.warning(f"Correlation calculation failed for {baseline_name}-{task_name}-{dataset_name}: {e}")
                        
                        # For imbalanced datasets, use agreement metrics instead
                        correlation_analysis['pearson_correlations'][baseline_name][task_name][dataset_name] = {
                            'correlation': float('nan'),
                            'p_value': float('nan'),
                            'n_samples': min_length,
                            'error': str(e)
                        }
                        
                        correlation_analysis['spearman_correlations'][baseline_name][task_name][dataset_name] = {
                            'correlation': float('nan'),
                            'p_value': float('nan'),
                            'n_samples': min_length,
                            'error': str(e)
                        }
                    
                    # Always calculate agreement metrics for all datasets
                    try:
                        # Convert to binary decisions based on thresholds
                        chatgpt_binary = self._convert_to_binary(chatgpt_scores, task_name)
                        baseline_binary = self._convert_to_binary(baseline_scores, task_name, baseline_name)
                        
                        # Calculate agreement metrics
                        agreement_metrics = self._calculate_agreement_metrics(
                            chatgpt_binary, baseline_binary, min_length
                        )
                        
                        self.logger.info(f"Agreement metrics: {agreement_metrics}")
                        
                        # Store agreement metrics without verbose logging
                        correlation_analysis['agreement_metrics'][baseline_name][task_name][dataset_name] = agreement_metrics
                        
                    except Exception as e:
                        self.logger.warning(f"Agreement calculation failed for {baseline_name}-{task_name}-{dataset_name}: {e}")
                        
                        correlation_analysis['agreement_metrics'][baseline_name][task_name][dataset_name] = {
                            'agreement': 0,
                            'cohens_kappa': 0,
                            'precision': 0,
                            'recall': 0,
                            'f1_score': 0,
                            'confusion_matrix': {'tp': 0, 'tn': 0, 'fp': 0, 'fn': 0},
                            'n_samples': min_length,
                            'error': str(e)
                        }
        
        # Compute summary statistics
        all_pearson_correlations = []
        all_spearman_correlations = []
        baseline_avg_correlations = {}
        
        # Agreement metrics summary
        all_agreement_scores = []
        all_kappa_scores = []
        baseline_avg_agreement = {}
        
        for baseline_name, baseline_corr in correlation_analysis['pearson_correlations'].items():
            baseline_correlations = []
            
            for task_corr in baseline_corr.values():
                for dataset_corr in task_corr.values():
                    if isinstance(dataset_corr, dict) and 'correlation' in dataset_corr:
                        corr_val = dataset_corr['correlation']
                        if not np.isnan(corr_val):
                            all_pearson_correlations.append(corr_val)
                            baseline_correlations.append(corr_val)
            
            if baseline_correlations:
                baseline_avg_correlations[baseline_name] = sum(baseline_correlations) / len(baseline_correlations)
        
        # Compute agreement metrics summary
        for baseline_name, baseline_agreement in correlation_analysis['agreement_metrics'].items():
            baseline_agreements = []
            baseline_kappas = []
            
            for task_agreement in baseline_agreement.values():
                for dataset_agreement in task_agreement.values():
                    if isinstance(dataset_agreement, dict):
                        agreement_val = dataset_agreement.get('agreement', 0)
                        kappa_val = dataset_agreement.get('cohens_kappa', 0)
                        
                        all_agreement_scores.append(agreement_val)
                        all_kappa_scores.append(kappa_val)
                        baseline_agreements.append(agreement_val)
                        baseline_kappas.append(kappa_val)
            
            if baseline_agreements:
                baseline_avg_agreement[baseline_name] = sum(baseline_agreements) / len(baseline_agreements)
        
        # Generate correlation summary
        correlation_analysis['correlation_summary'] = {
            'correlations': {
                'overall_mean_pearson': sum(all_pearson_correlations) / len(all_pearson_correlations) if all_pearson_correlations else 0,
                'overall_max_pearson': max(all_pearson_correlations) if all_pearson_correlations else 0,
                'overall_min_pearson': min(all_pearson_correlations) if all_pearson_correlations else 0,
                'baseline_average_correlations': baseline_avg_correlations,
                'best_correlating_baseline': max(baseline_avg_correlations.items(), key=lambda x: x[1])[0] if baseline_avg_correlations else None,
                'valid_correlations': len(all_pearson_correlations)
            },
            'agreement_metrics': {
                'overall_mean_agreement': sum(all_agreement_scores) / len(all_agreement_scores) if all_agreement_scores else 0,
                'overall_mean_kappa': sum(all_kappa_scores) / len(all_kappa_scores) if all_kappa_scores else 0,
                'baseline_average_agreement': baseline_avg_agreement,
                'best_agreeing_baseline': max(baseline_avg_agreement.items(), key=lambda x: x[1])[0] if baseline_avg_agreement else None,
                'total_comparisons': len(all_agreement_scores)
            }
        }
        
        # Rank baselines by correlation strength and agreement
        if baseline_avg_correlations or baseline_avg_agreement:
            correlation_analysis['method_rankings'] = {}
            
            if baseline_avg_correlations:
                sorted_baselines = sorted(baseline_avg_correlations.items(), key=lambda x: abs(x[1]), reverse=True)
                correlation_analysis['method_rankings']['by_correlation_strength'] = [{'baseline': name, 'avg_correlation': corr} for name, corr in sorted_baselines]
            
            if baseline_avg_agreement:
                sorted_agreement = sorted(baseline_avg_agreement.items(), key=lambda x: x[1], reverse=True)
                correlation_analysis['method_rankings']['by_agreement'] = [{'baseline': name, 'avg_agreement': agree} for name, agree in sorted_agreement]
        
        self.results['correlation_analysis'] = correlation_analysis
    
    def _extract_numerical_predictions(self, predictions: List[Dict], task_name: str) -> List[float]:
        """Extract numerical values from predictions for correlation analysis."""
        numerical_predictions = []
        
        self.logger.debug(f"Extracting {len(predictions)} predictions for task {task_name}")
        
        for i, pred in enumerate(predictions):
            # Handle both dict and object formats
            if hasattr(pred, 'prediction'):
                # Object format (RatingResult, BaselineResult)
                prediction = pred.prediction
                if i < 3:  # Debug first few predictions
                    self.logger.debug(f"Pred {i}: object with prediction={prediction}")
                numerical_predictions.append(float(prediction))
            elif isinstance(pred, dict):
                # Dictionary format
                if task_name == 'entailment_inference':
                    # Binary predictions: convert to 0/1
                    prediction = pred.get('prediction', 0)
                    if isinstance(prediction, (int, float)):
                        numerical_predictions.append(float(prediction))
                    elif isinstance(prediction, str):
                        # Handle string predictions like "ENTAILMENT"/"CONTRADICTION"
                        if prediction.upper() in ['ENTAILMENT', '1', 'TRUE']:
                            numerical_predictions.append(1.0)
                        else:
                            numerical_predictions.append(0.0)
                    else:
                        numerical_predictions.append(0.0)
                
                elif task_name == 'consistency_rating':
                    # Rating predictions: use direct numerical value
                    prediction = pred.get('prediction', 0)
                    if i < 3:  # Debug first few predictions
                        self.logger.debug(f"Pred {i}: {pred} -> prediction={prediction}")
                    if isinstance(prediction, (int, float)):
                        numerical_predictions.append(float(prediction))
                    else:
                        numerical_predictions.append(0.0)
                
                else:
                    # Default: try to extract numerical value
                    prediction = pred.get('prediction', 0)
                    try:
                        numerical_predictions.append(float(prediction))
                    except (ValueError, TypeError):
                        numerical_predictions.append(0.0)
            else:
                # Unknown format - default to 0
                numerical_predictions.append(0.0)
        
        self.logger.debug(f"Extracted {len(numerical_predictions)} numerical predictions")
        return numerical_predictions
    
    async def _analyze_performance_comparison(self):
        """Analyze performance comparison between ChatGPT and baselines."""
        self.logger.info("Analyzing performance comparison")
        
        performance_comparison = {
            'task_performance': {},
            'baseline_performance': {},
            'relative_performance': {},
            'performance_insights': {}
        }
        
        # Analyze performance for each task
        for task_name in self.results['chatgpt_results'].keys():
            performance_comparison['task_performance'][task_name] = {}
            
            # Get ChatGPT performance
            chatgpt_performances = []
            for dataset_name, dataset_results in self.results['chatgpt_results'][task_name].items():
                if 'performance_metrics' in dataset_results:
                    primary_metric = dataset_results['performance_metrics'].get('primary_metric', 0)
                    chatgpt_performances.append(primary_metric)
            
            if chatgpt_performances:
                performance_comparison['task_performance'][task_name]['chatgpt'] = {
                    'mean_performance': sum(chatgpt_performances) / len(chatgpt_performances),
                    'performances': chatgpt_performances
                }
            
            # Compare with each baseline
            for baseline_name in self.results['baseline_results'].keys():
                if task_name in self.results['baseline_results'][baseline_name]:
                    baseline_correlations = []
                    
                    for dataset_name, dataset_results in self.results['baseline_results'][baseline_name][task_name].items():
                        if 'comparison_with_chatgpt' in dataset_results:
                            comparison = dataset_results['comparison_with_chatgpt']
                            correlation = comparison.get('correlation_with_chatgpt', 0)
                            if not np.isnan(correlation):
                                baseline_correlations.append(correlation)
                    
                    if baseline_correlations:
                        performance_comparison['task_performance'][task_name][baseline_name] = {
                            'mean_correlation': sum(baseline_correlations) / len(baseline_correlations),
                            'correlations': baseline_correlations
                        }
        
        # Generate performance insights
        insights = {
            'best_correlating_baselines': {},
            'task_difficulty_ranking': [],
            'correlation_strength_summary': {}
        }
        
        # Find best correlating baseline for each task
        for task_name, task_perf in performance_comparison['task_performance'].items():
            best_baseline = None
            best_correlation = -1
            
            for method_name, method_data in task_perf.items():
                if method_name != 'chatgpt' and 'mean_correlation' in method_data:
                    correlation = abs(method_data['mean_correlation'])
                    if correlation > best_correlation:
                        best_correlation = correlation
                        best_baseline = method_name
            
            if best_baseline:
                insights['best_correlating_baselines'][task_name] = {
                    'baseline': best_baseline,
                    'correlation': best_correlation
                }
        
        # Rank tasks by correlation difficulty (lower correlation = more difficult)
        task_avg_correlations = {}
        for task_name, task_perf in performance_comparison['task_performance'].items():
            correlations = []
            for method_name, method_data in task_perf.items():
                if method_name != 'chatgpt' and 'mean_correlation' in method_data:
                    correlations.append(abs(method_data['mean_correlation']))
            
            if correlations:
                task_avg_correlations[task_name] = sum(correlations) / len(correlations)
        
        if task_avg_correlations:
            insights['task_difficulty_ranking'] = sorted(
                task_avg_correlations.items(),
                key=lambda x: x[1]
            )
        
        performance_comparison['performance_insights'] = insights
        self.results['performance_comparison'] = performance_comparison
    
    async def _perform_statistical_analysis(self):
        """Perform statistical significance testing."""
        self.logger.info("Performing statistical analysis")
        
        statistical_analysis = {
            'correlation_significance': {},
            'effect_size_analysis': {},
            'confidence_intervals': {},
            'significance_summary': {}
        }
        
        # Analyze correlation significance
        significant_correlations = 0
        total_correlations = 0
        
        for baseline_name, baseline_corr in self.results['correlation_analysis']['pearson_correlations'].items():
            statistical_analysis['correlation_significance'][baseline_name] = {}
            
            for task_name, task_corr in baseline_corr.items():
                statistical_analysis['correlation_significance'][baseline_name][task_name] = {}
                
                for dataset_name, dataset_corr in task_corr.items():
                    if isinstance(dataset_corr, dict) and 'p_value' in dataset_corr:
                        p_value = dataset_corr['p_value']
                        correlation = dataset_corr['correlation']
                        n_samples = dataset_corr['n_samples']
                        
                        is_significant = p_value < 0.05 and not np.isnan(correlation)
                        
                        statistical_analysis['correlation_significance'][baseline_name][task_name][dataset_name] = {
                            'correlation': correlation,
                            'p_value': p_value,
                            'is_significant': is_significant,
                            'n_samples': n_samples,
                            'effect_size': self._calculate_correlation_effect_size(correlation)
                        }
                        
                        total_correlations += 1
                        if is_significant:
                            significant_correlations += 1
        
        # Generate significance summary
        statistical_analysis['significance_summary'] = {
            'significant_correlations': significant_correlations,
            'total_correlations': total_correlations,
            'significance_rate': significant_correlations / total_correlations if total_correlations > 0 else 0,
            'interpretation': self._interpret_significance_results(significant_correlations, total_correlations)
        }
        
        self.results['statistical_analysis'] = statistical_analysis
    
    def _calculate_correlation_effect_size(self, correlation: float) -> str:
        """Calculate effect size interpretation for correlation."""
        abs_corr = abs(correlation) if not np.isnan(correlation) else 0
        
        if abs_corr >= 0.7:
            return 'large'
        elif abs_corr >= 0.5:
            return 'medium'
        elif abs_corr >= 0.3:
            return 'small'
        else:
            return 'negligible'
    
    def _interpret_significance_results(self, significant: int, total: int) -> str:
        """Interpret statistical significance results."""
        if total == 0:
            return "No correlations computed"
        
        rate = significant / total
        
        if rate >= 0.8:
            return "Strong evidence of correlation with ChatGPT across methods"
        elif rate >= 0.6:
            return "Moderate evidence of correlation with ChatGPT"
        elif rate >= 0.4:
            return "Limited evidence of correlation with ChatGPT"
        else:
            return "Weak evidence of correlation with ChatGPT"
    
    async def _generate_comparison_visualizations(self):
        """Generate comprehensive comparison visualizations."""
        self.logger.info("Generating comparison visualizations")
        
        viz_dir = self.output_dir / "figures"
        viz_dir.mkdir(exist_ok=True)
        
        try:
            # 1. Correlation heatmap
            self._create_correlation_heatmap(viz_dir)
            
            # 2. Baseline performance comparison
            self._create_baseline_performance_chart(viz_dir)
            
            # 3. Correlation scatter plots
            self._create_correlation_scatter_plots(viz_dir)
            
            # 4. Method ranking visualization
            self._create_method_ranking_chart(viz_dir)
            
            # Specialized visualizations for entailment inference task
            if 'entailment_inference' in self.results.get('chatgpt_results', {}):
                self._create_entailment_inference_analysis(viz_dir)
            
        except Exception as e:
            self.logger.warning(f"Visualization generation failed: {e}")
            self.results['visualizations'] = {'error': str(e)}
    
    def _create_correlation_heatmap(self, viz_dir: Path):
        """Create correlation heatmap visualization."""
        # Extract correlation data
        correlation_data = []
        
        pearson_correlations = self.results['correlation_analysis']['pearson_correlations']
        
        for baseline_name, baseline_corr in pearson_correlations.items():
            for task_name, task_corr in baseline_corr.items():
                for dataset_name, dataset_corr in task_corr.items():
                    if isinstance(dataset_corr, dict) and 'correlation' in dataset_corr:
                        correlation = dataset_corr['correlation']
                        if not np.isnan(correlation):
                            correlation_data.append({
                                'baseline': baseline_name,
                                'task': task_name,
                                'dataset': dataset_name,
                                'correlation': correlation,
                                'task_dataset': f"{task_name}_{dataset_name}"
                            })
        
        if not correlation_data:
            return
        
        # Create heatmap data structure
        baselines = sorted(list(set(item['baseline'] for item in correlation_data)))
        task_datasets = sorted(list(set(item['task_dataset'] for item in correlation_data)))
        
        # Create correlation matrix
        correlation_matrix = np.zeros((len(baselines), len(task_datasets)))
        
        for i, baseline in enumerate(baselines):
            for j, task_dataset in enumerate(task_datasets):
                matching_items = [item for item in correlation_data 
                                if item['baseline'] == baseline and item['task_dataset'] == task_dataset]
                if matching_items:
                    correlation_matrix[i][j] = matching_items[0]['correlation']
        
        # Create heatmap
        fig = go.Figure(data=go.Heatmap(
            z=correlation_matrix,
            x=task_datasets,
            y=baselines,
            colorscale='RdBu',
            zmid=0,
            zmin=-1,
            zmax=1,
            colorbar=dict(
                title="Correlation with ChatGPT",
                title_font=dict(family='Times New Roman', size=12)
            ),
            text=correlation_matrix,
            texttemplate="%{text:.3f}",
            textfont=dict(family='Times New Roman', size=10),
            hoverongaps=False
        ))
        
        fig.update_layout(
            title='ChatGPT-Baseline Correlation Matrix',
            xaxis_title='Task-Dataset',
            yaxis_title='Baseline Method',
            font=dict(family='Times New Roman', size=12),
            paper_bgcolor='white'
        )
        
        fig_path = viz_dir / "correlation_heatmap.png"
        fig.write_image(str(fig_path), width=1200, height=600, scale=2)
        
        if 'visualizations' not in self.results:
            self.results['visualizations'] = {}
        self.results['visualizations']['correlation_heatmap'] = str(fig_path)
    
    def _create_baseline_performance_chart(self, viz_dir: Path):
        """Create baseline performance comparison chart."""
        # Extract correlation summary data
        correlation_summary = self.results['correlation_analysis'].get('correlation_summary', {})
        baseline_avg_correlations = correlation_summary.get('baseline_average_correlations', {})
        
        if not baseline_avg_correlations:
            return
        
        # Create bar chart
        baselines = list(baseline_avg_correlations.keys())
        correlations = list(baseline_avg_correlations.values())
        
        # Sort by correlation strength
        sorted_data = sorted(zip(baselines, correlations), key=lambda x: abs(x[1]), reverse=True)
        baselines, correlations = zip(*sorted_data)
        
        fig = go.Figure()
        
        # Color bars based on correlation strength
        colors = ['green' if abs(corr) > 0.7 else 'orange' if abs(corr) > 0.5 else 'red' 
                 for corr in correlations]
        
        fig.add_trace(go.Bar(
            x=baselines,
            y=correlations,
            marker_color=colors,
            text=[f'{corr:.3f}' for corr in correlations],
            textposition='auto'
        ))
        
        # Add correlation strength reference lines
        fig.add_hline(y=0.7, line_dash="dash", line_color="green",
                     annotation_text="Strong correlation", annotation_position="right")
        fig.add_hline(y=0.5, line_dash="dash", line_color="orange",
                     annotation_text="Moderate correlation", annotation_position="right")
        fig.add_hline(y=0.3, line_dash="dash", line_color="red",
                     annotation_text="Weak correlation", annotation_position="right")
        fig.add_hline(y=0, line_dash="solid", line_color="black")
        
        fig.update_layout(
            title='Average Correlation Strength: ChatGPT vs SOTA Baselines',
            xaxis_title='Baseline Method',
            yaxis_title='Average Correlation Coefficient',
            font=dict(family='Times New Roman', size=12),
            plot_bgcolor='white',
            paper_bgcolor='white'
        )
        
        fig_path = viz_dir / "baseline_performance_comparison.png"
        fig.write_image(str(fig_path), width=1000, height=600, scale=2)
        
        self.results['visualizations']['baseline_performance_comparison'] = str(fig_path)
    
    def _create_correlation_scatter_plots(self, viz_dir: Path):
        """Create correlation scatter plots for best performing baselines."""
        # Find the best correlating baseline overall
        correlation_summary = self.results['correlation_analysis'].get('correlation_summary', {})
        best_baseline = correlation_summary.get('best_correlating_baseline')
        
        if not best_baseline or best_baseline not in self.results['baseline_results']:
            return
        
        # Create scatter plots for the best baseline
        fig = make_subplots(
            rows=2, cols=2,
            subplot_titles=[f'{best_baseline} vs ChatGPT - Task {i+1}' for i in range(4)],
            specs=[[{"type": "scatter"}, {"type": "scatter"}],
                   [{"type": "scatter"}, {"type": "scatter"}]]
        )
        
        plot_count = 0
        
        for task_name in self.results['baseline_results'][best_baseline].keys():
            for dataset_name in self.results['baseline_results'][best_baseline][task_name].keys():
                if plot_count >= 4:
                    break
                
                dataset_results = self.results['baseline_results'][best_baseline][task_name][dataset_name]
                
                if 'baseline_predictions' not in dataset_results:
                    continue
                
                # Get predictions
                chatgpt_data = self.results['chatgpt_results'][task_name][dataset_name]
                chatgpt_predictions = chatgpt_data['predictions']
                baseline_predictions = dataset_results['baseline_predictions']
                
                # Extract numerical values
                chatgpt_scores = self._extract_numerical_predictions(chatgpt_predictions, task_name)
                baseline_scores = self._extract_numerical_predictions(baseline_predictions, task_name)
                
                # Ensure same length
                min_length = min(len(chatgpt_scores), len(baseline_scores))
                if min_length < 2:
                    continue
                
                chatgpt_scores = chatgpt_scores[:min_length]
                baseline_scores = baseline_scores[:min_length]
                
                # Add scatter plot
                row = (plot_count // 2) + 1
                col = (plot_count % 2) + 1
                
                fig.add_trace(
                    go.Scatter(
                        x=baseline_scores,
                        y=chatgpt_scores,
                        mode='markers',
                        name=f'{task_name}_{dataset_name}',
                        showlegend=False
                    ),
                    row=row, col=col
                )
                
                # Add trend line
                if len(chatgpt_scores) > 1 and len(baseline_scores) > 1:
                    z = np.polyfit(baseline_scores, chatgpt_scores, 1)
                    p = np.poly1d(z)
                    x_line = np.linspace(min(baseline_scores), max(baseline_scores), 100)
                    
                    fig.add_trace(
                        go.Scatter(
                            x=x_line,
                            y=p(x_line),
                            mode='lines',
                            line=dict(color='red', dash='dash'),
                            showlegend=False
                        ),
                        row=row, col=col
                    )
                
                plot_count += 1
        
        fig.update_layout(
            title=f'Correlation Analysis: {best_baseline} vs ChatGPT',
            font=dict(family='Times New Roman', size=12),
            paper_bgcolor='white'
        )
        
        fig_path = viz_dir / f"correlation_scatter_{best_baseline}.png"
        fig.write_image(str(fig_path), width=1200, height=900, scale=2)
        
        self.results['visualizations']['correlation_scatter_plots'] = str(fig_path)
    
    def _create_method_ranking_chart(self, viz_dir: Path):
        """Create method ranking visualization."""
        method_rankings = self.results['correlation_analysis'].get('method_rankings', {})
        
        if 'by_correlation_strength' not in method_rankings:
            return
        
        rankings = method_rankings['by_correlation_strength']
        
        # Create ranking chart
        baselines = [item['baseline'] for item in rankings]
        correlations = [abs(item['avg_correlation']) for item in rankings]
        
        fig = go.Figure()
        
        # Create horizontal bar chart
        fig.add_trace(go.Bar(
            y=baselines,
            x=correlations,
            orientation='h',
            marker_color='#2E86AB',
            text=[f'{corr:.3f}' for corr in correlations],
            textposition='auto'
        ))
        
        fig.update_layout(
            title='Baseline Method Ranking by Correlation Strength',
            xaxis_title='Average Absolute Correlation',
            yaxis_title='Baseline Method',
            font=dict(family='Times New Roman', size=12),
            plot_bgcolor='white',
            paper_bgcolor='white'
        )
        
        fig_path = viz_dir / "method_ranking.png"
        fig.write_image(str(fig_path), width=1000, height=600, scale=2)
        
        self.results['visualizations']['method_ranking'] = str(fig_path)
    
    def _create_entailment_inference_analysis(self, viz_dir: Path):
        """Create specialized analysis for entailment inference task."""
        # Create agreement matrix for entailment inference
        fig = make_subplots(
            rows=1, cols=2,
            subplot_titles=['CNN/DailyMail Agreement', 'XSum Agreement'],
            specs=[[{"type": "heatmap"}, {"type": "heatmap"}]]
        )
        
        datasets = ['cnn_dailymail', 'xsum']
        
        for i, dataset in enumerate(datasets):
            agreement_matrix = []
            labels = []
            
            # Get ChatGPT results
            chatgpt_data = self.results['chatgpt_results']['entailment_inference'][dataset]
            chatgpt_predictions = chatgpt_data['predictions']
            
            # Convert to binary for analysis
            chatgpt_binary = []
            for pred in chatgpt_predictions:
                if isinstance(pred, str):
                    chatgpt_binary.append(1 if pred.upper() == 'ENTAILMENT' else 0)
                elif hasattr(pred, 'prediction'):
                    # Handle object with prediction attribute
                    prediction = pred.prediction
                    if isinstance(prediction, str):
                        chatgpt_binary.append(1 if prediction.upper() == 'ENTAILMENT' else 0)
                    else:
                        chatgpt_binary.append(1 if prediction == 1 else 0)
                else:
                    # Handle other cases
                    chatgpt_binary.append(1 if str(pred).upper() == 'ENTAILMENT' else 0)
            
            # Check each baseline
            for baseline in ['factcc', 'bertscore', 'rouge']:
                if baseline in self.results['baseline_results']:
                    baseline_data = self.results['baseline_results'][baseline].get('entailment_inference', {}).get(dataset, {})
                    
                    if 'baseline_predictions' in baseline_data:
                        baseline_predictions = baseline_data['baseline_predictions']
                        
                        # Convert baseline predictions to binary
                        if baseline == 'factcc':
                            baseline_binary = [1 if pred == 'CONSISTENT' else 0 for pred in baseline_predictions]
                        else:
                            # For BERTScore and ROUGE, use threshold
                            baseline_scores = [float(pred) if isinstance(pred, (int, float)) else 0.5 for pred in baseline_predictions]
                            threshold = np.median(baseline_scores)
                            baseline_binary = [1 if score >= threshold else 0 for score in baseline_scores]
                        
                        # Calculate agreement
                        min_length = min(len(chatgpt_binary), len(baseline_binary))
                        if min_length > 0:
                            agreement = np.mean([c == b for c, b in zip(chatgpt_binary[:min_length], baseline_binary[:min_length])])
                            agreement_matrix.append(agreement)
                            labels.append(baseline.upper())
            
            if agreement_matrix:
                # Create heatmap
                fig.add_trace(
                    go.Heatmap(
                        z=[agreement_matrix],
                        x=labels,
                        y=['Agreement'],
                        colorscale='RdYlBu',
                        zmin=0,
                        zmax=1,
                        text=[[f'{val:.3f}' for val in agreement_matrix]],
                        texttemplate='%{text}',
                        textfont={"size": 12},
                        hovertemplate='%{x}<br>Agreement: %{z:.3f}<extra></extra>'
                    ),
                    row=1, col=i+1
                )
        
        fig.update_layout(
            title='Entailment Inference: ChatGPT vs Baselines Agreement Analysis',
            height=400,
            showlegend=False
        )
        
        # Save the figure
        fig.write_image(viz_dir / 'entailment_inference_agreement.png', width=800, height=400, scale=2)
    
    def _create_task_specific_performance_charts(self, viz_dir: Path):
        """Create task-specific performance comparison charts."""
        # Only create charts for tasks that were actually executed
        executed_tasks = set()
        if 'chatgpt_results' in self.results:
            executed_tasks.update(self.results['chatgpt_results'].keys())
        
        # 1. Entailment Inference Performance
        if 'entailment_inference' in executed_tasks:
            self._create_entailment_performance_chart(viz_dir)
        
        # 2. Consistency Rating Performance  
        if 'consistency_rating' in executed_tasks:
            self._create_consistency_performance_chart(viz_dir)
        
    def _create_entailment_performance_chart(self, viz_dir: Path):
        """Create entailment inference specific performance chart."""
        fig = go.Figure()
        
        datasets = ['cnn_dailymail', 'xsum']
        baselines = ['factcc', 'bertscore', 'rouge']
        
        # Calculate ChatGPT performance (% ENTAILMENT predictions)
        chatgpt_performance = {}
        for dataset in datasets:
            if 'entailment_inference' in self.results['chatgpt_results']:
                chatgpt_data = self.results['chatgpt_results']['entailment_inference'][dataset]
                predictions = chatgpt_data['predictions']
                
                # Handle different prediction formats
                entailment_count = 0
                for pred in predictions:
                    if isinstance(pred, str):
                        if pred.upper() == 'ENTAILMENT':
                            entailment_count += 1
                    elif hasattr(pred, 'prediction'):
                        # Handle object with prediction attribute
                        prediction = pred.prediction
                        if isinstance(prediction, str) and prediction.upper() == 'ENTAILMENT':
                            entailment_count += 1
                        elif prediction == 1:
                            entailment_count += 1
                    else:
                        # Handle other cases
                        if str(pred).upper() == 'ENTAILMENT':
                            entailment_count += 1
                
                entailment_rate = entailment_count / len(predictions)
                chatgpt_performance[dataset] = entailment_rate
        
        # Add ChatGPT performance
        fig.add_trace(go.Bar(
            name='ChatGPT',
            x=datasets,
            y=[chatgpt_performance.get(d, 0) for d in datasets],
            marker_color='#FF6B6B',
            text=[f'{chatgpt_performance.get(d, 0):.1%}' for d in datasets],
            textposition='auto',
        ))
        
        # Add baseline performance
        colors = {'factcc': '#4ECDC4', 'bertscore': '#45B7D1', 'rouge': '#96CEB4'}
        
        for baseline in baselines:
            if baseline in self.results['baseline_results']:
                baseline_performance = {}
                for dataset in datasets:
                    baseline_data = self.results['baseline_results'][baseline].get('entailment_inference', {}).get(dataset, {})
                    
                    if 'baseline_predictions' in baseline_data:
                        predictions = baseline_data['baseline_predictions']
                        
                        if baseline == 'factcc':
                            # FactCC uses CONSISTENT/INCONSISTENT
                            consistent_rate = sum(1 for pred in predictions if pred == 'CONSISTENT') / len(predictions)
                            baseline_performance[dataset] = consistent_rate
                        else:
                            # BERTScore and ROUGE use numerical scores
                            scores = [float(pred) if isinstance(pred, (int, float)) else 0.5 for pred in predictions]
                            threshold = np.median(scores)
                            above_threshold_rate = sum(1 for score in scores if score >= threshold) / len(scores)
                            baseline_performance[dataset] = above_threshold_rate
                
                fig.add_trace(go.Bar(
                    name=baseline.upper(),
                    x=datasets,
                    y=[baseline_performance.get(d, 0) for d in datasets],
                    marker_color=colors[baseline],
                    text=[f'{baseline_performance.get(d, 0):.1%}' for d in datasets],
                    textposition='auto',
                ))
        
        fig.update_layout(
            title='Entailment Inference: Positive Prediction Rates',
            xaxis_title='Dataset',
            yaxis_title='Positive Prediction Rate',
            yaxis=dict(tickformat='.0%'),
            barmode='group',
            height=500,
            showlegend=True,
            legend=dict(orientation="h", yanchor="bottom", y=1.02, xanchor="right", x=1)
        )
        
        fig.write_image(viz_dir / 'entailment_inference_performance.png', width=800, height=500, scale=2)
        
    def _create_consistency_performance_chart(self, viz_dir: Path):
        """Create consistency rating specific performance chart."""
        fig = go.Figure()
        
        datasets = ['cnn_dailymail', 'xsum']
        baselines = ['factcc', 'bertscore', 'rouge']
        
        # Calculate ChatGPT average consistency scores
        chatgpt_performance = {}
        for dataset in datasets:
            if 'consistency_rating' in self.results['chatgpt_results']:
                chatgpt_data = self.results['chatgpt_results']['consistency_rating'][dataset]
                predictions = chatgpt_data['predictions']
                # Extract numerical scores
                scores = []
                for pred in predictions:
                    if isinstance(pred, (int, float)):
                        scores.append(float(pred))
                    elif isinstance(pred, str) and pred.replace('.', '').isdigit():
                        scores.append(float(pred))
                
                if scores:
                    chatgpt_performance[dataset] = np.mean(scores)
        
        # Add ChatGPT performance
        fig.add_trace(go.Bar(
            name='ChatGPT',
            x=datasets,
            y=[chatgpt_performance.get(d, 0) for d in datasets],
            marker_color='#FF6B6B',
            text=[f'{chatgpt_performance.get(d, 0):.1f}' for d in datasets],
            textposition='auto',
        ))
        
        # Add baseline performance
        colors = {'factcc': '#4ECDC4', 'bertscore': '#45B7D1', 'rouge': '#96CEB4'}
        
        for baseline in baselines:
            if baseline in self.results['baseline_results']:
                baseline_performance = {}
                for dataset in datasets:
                    baseline_data = self.results['baseline_results'][baseline].get('consistency_rating', {}).get(dataset, {})
                    
                    if 'baseline_predictions' in baseline_data:
                        predictions = baseline_data['baseline_predictions']
                        scores = [float(pred) if isinstance(pred, (int, float)) else 0 for pred in predictions]
                        
                        if scores:
                            # Normalize scores to 0-100 scale for comparison
                            if baseline == 'bertscore':
                                # BERTScore is typically 0-1, scale to 0-100
                                baseline_performance[dataset] = np.mean(scores) * 100
                            elif baseline == 'rouge':
                                # ROUGE is typically 0-1, scale to 0-100  
                                baseline_performance[dataset] = np.mean(scores) * 100
                            else:
                                baseline_performance[dataset] = np.mean(scores)
                
                fig.add_trace(go.Bar(
                    name=baseline.upper(),
                    x=datasets,
                    y=[baseline_performance.get(d, 0) for d in datasets],
                    marker_color=colors[baseline],
                    text=[f'{baseline_performance.get(d, 0):.1f}' for d in datasets],
                    textposition='auto',
                ))
        
        fig.update_layout(
            title='Consistency Rating: Average Scores by Method',
            xaxis_title='Dataset',
            yaxis_title='Average Score',
            barmode='group',
            height=500,
            showlegend=True,
            legend=dict(orientation="h", yanchor="bottom", y=1.02, xanchor="right", x=1)
        )
        
        fig.write_image(viz_dir / 'consistency_rating_performance.png', width=800, height=500, scale=2)

    async def _save_results(self):
        """Save comprehensive results and generate report."""
        self.logger.info("Saving results and generating report")
        
        # Save as JSON
        json_path = self.output_dir / "sota_comparison_results.json"
        with open(json_path, 'w') as f:
            json.dump(self.results, f, indent=2, default=str)
        
        # Generate markdown report
        report_path = self.output_dir / "sota_comparison_report.md"
        with open(report_path, 'w') as f:
            f.write(self._generate_comparison_report())
        
        self.logger.info(f"Results saved to: {self.output_dir}")
    
    def _generate_comparison_report(self) -> str:
        """Generate comprehensive comparison report."""
        report = f"""# SOTA Baseline Comparison Report

**Experiment**: {self.experiment_name}  
**Generated**: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}  
**Author**: Michael Ogunjimi  
**Institution**: University of Manchester, MSc AI  

## Executive Summary

This report compares ChatGPT's zero-shot factuality evaluation performance with
state-of-the-art baseline methods across multiple tasks and datasets.

## Experimental Setup

- **ChatGPT Cost**: ${self.results.get('cost_analysis', {}).get('chatgpt_cost', 0):.4f}
- **Baselines Evaluated**: {', '.join(self.results.get('baseline_results', {}).keys())}
- **Tasks**: {', '.join(self.results.get('chatgpt_results', {}).keys())}

## Correlation Analysis Results

"""
        
        # Add correlation summary
        correlation_summary = self.results.get('correlation_analysis', {}).get('correlation_summary', {})
        
        if correlation_summary:
            report += f"### Overall Correlation Statistics\n\n"
            report += f"- **Overall Mean Correlation**: {correlation_summary.get('overall_mean_pearson', 0):.4f}\n"
            report += f"- **Maximum Correlation**: {correlation_summary.get('overall_max_pearson', 0):.4f}\n"
            report += f"- **Minimum Correlation**: {correlation_summary.get('overall_min_pearson', 0):.4f}\n"
            
            best_baseline = correlation_summary.get('best_correlating_baseline')
            if best_baseline:
                best_corr = correlation_summary.get('baseline_average_correlations', {}).get(best_baseline, 0)
                report += f"- **Best Correlating Baseline**: {best_baseline} ({best_corr:.4f})\n\n"
        
        # Add detailed baseline performance
        baseline_avg_correlations = correlation_summary.get('baseline_average_correlations', {})
        if baseline_avg_correlations:
            report += "### Baseline Performance Summary\n\n"
            
            for baseline, correlation in sorted(baseline_avg_correlations.items(), 
                                               key=lambda x: abs(x[1]), reverse=True):
                # Handle case where baseline might be an object instead of string
                try:
                    baseline_name = baseline.upper() if isinstance(baseline, str) else str(baseline).upper()
                except:
                    baseline_name = str(baseline).upper()
                
                strength = self._calculate_correlation_effect_size(correlation)
                report += f"- **{baseline_name}**: {correlation:.4f} ({strength} correlation)\n"
            report += "\n"
        
        # Add task-specific analysis
        performance_comparison = self.results.get('performance_comparison', {})
        task_performance = performance_comparison.get('task_performance', {})
        
        if task_performance:
            report += "## Task-Specific Analysis\n\n"
            
            for task_name, task_data in task_performance.items():
                report += f"### {task_name.replace('_', ' ').title()}\n\n"
                
                for method_name, method_data in task_data.items():
                    if method_name == 'chatgpt':
                        performance = method_data.get('mean_performance', 0)
                        report += f"- **ChatGPT Performance**: {performance:.4f}\n"
                    else:
                        correlation = method_data.get('mean_correlation', 0)
                        report += f"- **{method_name.upper()} Correlation**: {correlation:.4f}\n"
                report += "\n"
        
        # Add statistical analysis
        statistical_analysis = self.results.get('statistical_analysis', {})
        significance_summary = statistical_analysis.get('significance_summary', {})
        
        if significance_summary:
            report += "## Statistical Significance Analysis\n\n"
            report += f"- **Significant Correlations**: {significance_summary.get('significant_correlations', 0)}/{significance_summary.get('total_correlations', 0)}\n"
            report += f"- **Significance Rate**: {significance_summary.get('significance_rate', 0):.2f}\n"
            report += f"- **Interpretation**: {significance_summary.get('interpretation', 'No interpretation available')}\n\n"
        
        # Add performance insights
        performance_insights = performance_comparison.get('performance_insights', {})
        
        if performance_insights:
            report += "## Key Insights\n\n"
            
            # Best correlating baselines
            best_correlating = performance_insights.get('best_correlating_baselines', {})
            if best_correlating:
                report += "### Best Correlating Baselines by Task\n\n"
                for task, baseline_info in best_correlating.items():
                    baseline = baseline_info['baseline']
                    correlation = baseline_info['correlation']
                    report += f"- **{task.replace('_', ' ').title()}**: {baseline.upper()} ({correlation:.4f})\n"
                report += "\n"
            
            # Task difficulty ranking
            task_difficulty = performance_insights.get('task_difficulty_ranking', [])
            if task_difficulty:
                report += "### Task Difficulty Ranking (by correlation strength)\n\n"
                for i, (task, avg_corr) in enumerate(task_difficulty, 1):
                    difficulty = "Easy" if avg_corr > 0.7 else "Moderate" if avg_corr > 0.5 else "Difficult"
                    report += f"{i}. **{task.replace('_', ' ').title()}**: {avg_corr:.4f} ({difficulty})\n"
                report += "\n"
        
        # Add recommendations
        report += "## Conclusions and Recommendations\n\n"
        
        # Generate conclusions based on data
        if baseline_avg_correlations:
            best_baseline = max(baseline_avg_correlations.items(), key=lambda x: abs(x[1]))
            best_correlation = best_baseline[1]
            
            if abs(best_correlation) > 0.7:
                report += "1. **Strong Baseline Agreement**: ChatGPT shows strong correlation with traditional metrics, suggesting reliability.\n"
            elif abs(best_correlation) > 0.5:
                report += "1. **Moderate Baseline Agreement**: ChatGPT shows moderate correlation with traditional metrics.\n"
            else:
                report += "1. **Limited Baseline Agreement**: ChatGPT shows novel evaluation patterns compared to traditional metrics.\n"
        
        # Significance interpretation
        if significance_summary:
            significance_rate = significance_summary.get('significance_rate', 0)
            if significance_rate > 0.6:
                report += "2. **Statistical Reliability**: Results show strong statistical significance across methods.\n"
            else:
                report += "2. **Statistical Caution**: Limited statistical significance suggests need for larger studies.\n"
        
        report += "3. **Future Work**: Consider ensemble methods combining ChatGPT with best-correlating baselines.\n"
        
        report += "\n### Methodological Notes\n\n"
        report += "- Correlations computed using Pearson correlation coefficient\n"
        report += "- Statistical significance tested at α = 0.05 level\n"
        report += "- Effect sizes interpreted using standard correlation strength guidelines\n"
        
        report += f"\n---\n*Report generated by SOTA Comparison Experiment*"
        
        return report


def main():
    """Main entry point for SOTA comparison experiment."""
    parser = argparse.ArgumentParser(
        description="Compare ChatGPT with SOTA baseline methods for factuality evaluation"
    )
    parser.add_argument(
        "--config",
        type=str,
        default="config/default.yaml",
        help="Path to configuration file"
    )
    parser.add_argument(
        "--experiment-name",
        type=str,
        help="Name for this experiment"
    )
    parser.add_argument(
        "--task",
        type=str,
        choices=['entailment_inference', 'consistency_rating'],
        help="Run single task only"
    )
    parser.add_argument(
        "--dataset",
        type=str,
        choices=['cnn_dailymail', 'xsum'],
        help="Use single dataset only"
    )
    parser.add_argument(
        "--baseline",
        type=str,
        choices=['factcc', 'bertscore', 'rouge'],
        help="Use single baseline only"
    )
    parser.add_argument(
        "--sample-size",
        type=int,
        help="Number of examples per dataset"
    )
    parser.add_argument(
        "--prompt-type",
        type=str,
        choices=['zero_shot', 'chain_of_thought'],
        default='zero_shot',
        help="ChatGPT prompt type to use"
    )
    parser.add_argument(
        "--quick-test",
        action="store_true",
        help="Run quick test with minimal data"
    )
    
    args = parser.parse_args()
    
    # Set parameters
    tasks = [args.task] if args.task else None
    datasets = [args.dataset] if args.dataset else None
    baselines = [args.baseline] if args.baseline else None
    sample_size = args.sample_size
    
    if args.quick_test:
        sample_size = 20
        print("Running quick test with 20 examples per dataset")
    
    # Initialize and run experiment
    experiment = SOTAComparisonExperiment(
        config_path=args.config,
        experiment_name=args.experiment_name
    )
    
    results = asyncio.run(experiment.run_sota_comparison(
        tasks=tasks,
        datasets=datasets,
        baselines=baselines,
        sample_size=sample_size,
        prompt_type=args.prompt_type
    ))
     # Print summary
    print(f"\n{'='*60}")
    print(f"SOTA COMPARISON EXPERIMENT COMPLETED")
    print(f"{'='*60}")
    print(f"Experiment: {experiment.experiment_name}")
    print(f"Output directory: {experiment.output_dir}")
    print(f"ChatGPT cost: ${results.get('cost_analysis', {}).get('chatgpt_cost', 0):.4f}")
    
    # Print correlation summary
    correlation_summary = results.get('correlation_analysis', {}).get('correlation_summary', {})
    if correlation_summary:
        corr_metrics = correlation_summary.get('correlations', {})
        agreement_metrics = correlation_summary.get('agreement_metrics', {})
        
        print(f"\n{'='*40}")
        print(f"CORRELATION ANALYSIS")
        print(f"{'='*40}")
        print(f"Valid correlations: {corr_metrics.get('valid_correlations', 0)}")
        
        # Show both Pearson and Spearman correlations
        mean_pearson = corr_metrics.get('overall_mean_pearson', 0)
        spearman_correlations = results.get('correlation_analysis', {}).get('spearman_correlations', {})
        mean_spearman = 0
        
        if spearman_correlations:
            # Calculate Spearman mean
            all_spearman = []
            for baseline_name, baseline_corr in spearman_correlations.items():
                for task_corr in baseline_corr.values():
                    for dataset_corr in task_corr.values():
                        if isinstance(dataset_corr, dict) and 'correlation' in dataset_corr:
                            corr_val = dataset_corr['correlation']
                            if not np.isnan(corr_val):
                                all_spearman.append(corr_val)
            
            if all_spearman:
                mean_spearman = sum(all_spearman) / len(all_spearman)
        
        print(f"Overall mean Pearson: {mean_pearson:.4f}, Overall mean Spearman: {mean_spearman:.4f}")
        
        best_baseline = corr_metrics.get('best_correlating_baseline')
        if best_baseline:
            best_corr = corr_metrics.get('baseline_average_correlations', {}).get(best_baseline, 0)
            print(f"Best correlating baseline: {best_baseline} ({best_corr:.4f})")
        
        print(f"\n{'='*40}")
        print(f"AGREEMENT ANALYSIS")
        print(f"{'='*40}")
        print(f"Total comparisons: {agreement_metrics.get('total_comparisons', 0)}")
        print(f"Overall mean agreement: {agreement_metrics.get('overall_mean_agreement', 0):.4f}")
        print(f"Overall mean Cohen's κ: {agreement_metrics.get('overall_mean_kappa', 0):.4f}")
        
        best_agreeing = agreement_metrics.get('best_agreeing_baseline')
        if best_agreeing:
            best_agree = agreement_metrics.get('baseline_average_agreement', {}).get(best_agreeing, 0)
            print(f"Best agreeing baseline: {best_agreeing} ({best_agree:.4f})")
        
        # Show detailed results by baseline
        if corr_metrics.get('baseline_average_correlations') or agreement_metrics.get('baseline_average_agreement'):
            print(f"\n{'='*40}")
            print(f"BASELINE PERFORMANCE")
            print(f"{'='*40}")
            
            # Get Spearman correlations
            spearman_correlations = results.get('correlation_analysis', {}).get('spearman_correlations', {})
            spearman_baseline_avg = {}
            
            for baseline_name, baseline_corr in spearman_correlations.items():
                baseline_correlations = []
                for task_corr in baseline_corr.values():
                    for dataset_corr in task_corr.values():
                        if isinstance(dataset_corr, dict) and 'correlation' in dataset_corr:
                            corr_val = dataset_corr['correlation']
                            if not np.isnan(corr_val):
                                baseline_correlations.append(corr_val)
                
                if baseline_correlations:
                    spearman_baseline_avg[baseline_name] = sum(baseline_correlations) / len(baseline_correlations)
            
            all_baselines = set(corr_metrics.get('baseline_average_correlations', {}).keys()) | set(agreement_metrics.get('baseline_average_agreement', {}).keys())
            for baseline in sorted(all_baselines):
                pearson_val = corr_metrics.get('baseline_average_correlations', {}).get(baseline, 0)
                spearman_val = spearman_baseline_avg.get(baseline, 0)
                agree_val = agreement_metrics.get('baseline_average_agreement', {}).get(baseline, 0)
                kappa_val = 0  # We'll calculate this separately
                
                # Get Cohen's kappa from the agreement metrics
                agreement_details = results.get('correlation_analysis', {}).get('agreement_metrics', {}).get(baseline, {})
                if agreement_details:
                    for task_agreement in agreement_details.values():
                        for dataset_agreement in task_agreement.values():
                            if isinstance(dataset_agreement, dict) and 'cohens_kappa' in dataset_agreement:
                                kappa_val = dataset_agreement['cohens_kappa']
                                break
                
                print(f"{baseline:20} | Pearson: {pearson_val:7.4f} | Spearman: {spearman_val:7.4f} | Agreement: {agree_val:6.4f} | κ: {kappa_val:6.4f}")
    
    # Print statistical summary
    stats = results.get('statistical_analysis', {}).get('significance_summary', {})
    if stats:
        print(f"\n{'='*40}")
        print(f"STATISTICAL SUMMARY")
        print(f"{'='*40}")
        print(f"Significant correlations: {stats.get('significant_correlations', 0)}/{stats.get('total_correlations', 0)}")
        print(f"Interpretation: {stats.get('interpretation', 'No interpretation')}")
    
    # Final interpretation
    print(f"\n{'='*60}")
    print(f"INTERPRETATION")
    print(f"{'='*60}")
    if corr_metrics.get('valid_correlations', 0) == 0:
        print("⚠️  No valid correlations found - dataset may be imbalanced")
        print("📊 Using agreement metrics instead for evaluation")
        if agreement_metrics.get('total_comparisons', 0) > 0:
            print(f"💡 Mean agreement: {agreement_metrics.get('overall_mean_agreement', 0):.1%}")
    else:
        print(f"✅ Found {corr_metrics.get('valid_correlations', 0)} valid correlations")
        print(f"📊 Mean correlation: {corr_metrics.get('overall_mean_pearson', 0):.4f}")
    print(f"{'='*60}")


if __name__ == "__main__":
    main()