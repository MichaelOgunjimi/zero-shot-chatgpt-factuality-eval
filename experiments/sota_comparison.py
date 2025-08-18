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
import re
import sys
import time
from datetime import datetime
from pathlib import Path
from typing import Dict, List, Any, Optional, Tuple

import numpy as np
from scipy.stats import pearsonr, spearmanr
import plotly.graph_objects as go
from plotly.subplots import make_subplots

if __name__ == "__main__":
    sys.path.insert(0, str(Path(__file__).parent.parent))
else:
    current_dir = Path(__file__).resolve().parent.parent
    if str(current_dir) not in sys.path:
        sys.path.insert(0, str(current_dir))

try:
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
    
    def __init__(self, model: str, tier: str, experiment_name: str = None, log_dir: str = None, output_dir: str = None):
        """Initialize the SOTA comparison experiment."""
        # Load configuration
        self.config = get_config(model=model, tier=tier)
        
        # Set up experiment tracking
        self.experiment_name = experiment_name or f"sota_comparison_{datetime.now().strftime('%Y%m%d_%H%M%S')}"
        
        # Use provided output_dir or create default
        if output_dir:
            self.output_dir = Path(output_dir)
        else:
            self.output_dir = Path(f"results/experiments/{self.experiment_name}")
        
        # Create the desired folder structure
        self.output_dir.mkdir(parents=True, exist_ok=True)
        (self.output_dir / "figures").mkdir(exist_ok=True)
        (self.output_dir / "logs").mkdir(exist_ok=True)
        (self.output_dir / "baseline_results").mkdir(exist_ok=True)
        
        # Create output directories only if not using custom output_dir
        if not output_dir:
            create_output_directories(self.config)
        
        # Set up logging - reduced verbosity with custom log_dir if provided
        self.experiment_logger = setup_experiment_logger(
            self.experiment_name,
            self.config,
            str(self.output_dir / "logs")  # Use our logs folder
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
        
        self.logger.info(f"SOTA comparison experiment initialized: {self.experiment_name}")
    
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
        print(f"\n🤖 Running ChatGPT Evaluations")
        print("=" * 50)
        
        total_evaluations = len(tasks) * len(datasets)
        current_eval = 0
        total_cost = 0.0
        
        for task_name in tasks:
            print(f"\n⚡ Task: {task_name}")
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
                print(f"   📊 [{current_eval}/{total_evaluations}] Processing {dataset_name} ({sample_size} examples)")
                
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
                    
                    # Log concise progress
                    primary_metric = performance_metrics.get('primary_metric', 'N/A')
                    print(f"   ✅ Completed: Performance={primary_metric:.3f}, Time={processing_time:.1f}s, Cost=${task_cost:.4f}")
                    
                except Exception as e:
                    print(f"   ❌ Failed: {str(e)}")
                    self.logger.error(f"Failed to process ChatGPT {task_name} on {dataset_name}: {e}")
                    self.results['chatgpt_results'][task_name][dataset_name] = {
                        'error': str(e),
                        'status': 'failed'
                    }
        
        # Store ChatGPT cost
        self.results['cost_analysis']['chatgpt_cost'] = total_cost
        print(f"\n✅ ChatGPT evaluation completed. Total cost: ${total_cost:.4f}")
        self.logger.info(f"ChatGPT evaluations completed. Total cost: ${total_cost:.4f}")
    
    async def _run_baseline_evaluations(
        self,
        tasks: List[str],
        datasets: List[str],
        baselines: List[str],
        sample_size: int
    ):
        """Run baseline method evaluations."""
        print(f"\n🔧 Running Baseline Evaluations")
        print("=" * 50)
        
        baseline_instances = create_all_baselines(self.config)
        
        for baseline_name in baselines:
            if baseline_name not in baseline_instances:
                print(f"⚠️  Baseline {baseline_name} not available, skipping")
                continue
                
            print(f"\n🎯 Baseline: {baseline_name.upper()}")
            baseline = baseline_instances[baseline_name]
            self.results['baseline_results'][baseline_name] = {}
            
            for task_name in tasks:
                if not baseline.supports_task(task_name):
                    print(f"   ⚠️  Does not support {task_name}, skipping")
                    continue
                
                print(f"   📊 Processing task: {task_name}")
                self.results['baseline_results'][baseline_name][task_name] = {}
                
                for dataset_name in datasets:
                    if (task_name not in self.results['chatgpt_results'] or 
                        dataset_name not in self.results['chatgpt_results'][task_name] or
                        'examples' not in self.results['chatgpt_results'][task_name][dataset_name]):
                        print(f"      ⚠️  No ChatGPT data for {dataset_name}, skipping")
                        continue
                    
                    try:
                        chatgpt_data = self.results['chatgpt_results'][task_name][dataset_name]
                        examples = chatgpt_data['examples']
                        chatgpt_predictions = chatgpt_data['predictions']
                        
                        print(f"      🔄 Processing {dataset_name} ({len(examples)} examples)")
                        
                        # Run baseline evaluation
                        start_time = time.time()
                        baseline_results = await self._evaluate_baseline_on_examples(
                            baseline, baseline_name, task_name, examples
                        )
                        processing_time = time.time() - start_time
                        
                        # Compare with ChatGPT
                        baseline_results_dict = {baseline_name: baseline_results}
                        
                        # Convert TaskResult objects to dictionaries for comparison
                        chatgpt_predictions_dict = [
                            {
                                'example_id': result.example_id,
                                'prediction': result.prediction,
                                'binary_prediction': getattr(result, 'binary_prediction', result.prediction),
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
                        
                        self.results['baseline_results'][baseline_name][task_name][dataset_name] = {
                            'baseline_predictions': baseline_results,
                            'comparison_with_chatgpt': comparison_results,
                            'processing_time': processing_time,
                            'dataset_size': len(examples)
                        }
                        
                        baseline_comparison = comparison_results.get(baseline_name, {})
                        if task_name == "entailment_inference":
                            correlation = baseline_comparison.get('cohens_kappa', 'N/A')
                            metric_name = "κ"
                        elif task_name == "consistency_rating":
                            correlation = baseline_comparison.get('pearson_correlation', 'N/A')
                            metric_name = "r"
                        else:
                            correlation = baseline_comparison.get('avg_spearman_rho', 'N/A')
                            metric_name = "ρ"
                        
                        if isinstance(correlation, float) and np.isnan(correlation):
                            correlation = 'N/A'
                        elif isinstance(correlation, float):
                            correlation = f"{correlation:.3f}"
                        
                        print(f"      ✅ {dataset_name}: {metric_name}={correlation}, Time={processing_time:.1f}s")
                        
                    except Exception as e:
                        print(f"      ❌ Failed on {dataset_name}: {str(e)}")
                        self.logger.error(f"Failed to process {baseline_name} on {task_name}-{dataset_name}: {e}")
                        self.results['baseline_results'][baseline_name][task_name][dataset_name] = {
                            'error': str(e),
                            'status': 'failed'
                        }
        
        print(f"\n✅ Baseline evaluations completed")
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
                if len(scores) > 0:
                    threshold = np.median(scores)
                else:
                    threshold = 3.0  # Default fallback
            else:
                threshold = 50.0
            
            return [1 if score > threshold else 0 for score in scores]
        elif task_name == 'entailment_inference':
            return [1 if score > 0.5 else 0 for score in scores]
        else:
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
                    
                    chatgpt_data = self.results['chatgpt_results'][task_name][dataset_name]
                    
                    if 'predictions' not in chatgpt_data or 'baseline_predictions' not in dataset_results:
                        self.logger.warning(f"Missing predictions for {baseline_name}-{task_name}-{dataset_name}")
                        continue
                    
                    chatgpt_predictions = chatgpt_data['predictions']
                    baseline_predictions = dataset_results['baseline_predictions']
                    
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
        
        self.output_dir.mkdir(parents=True, exist_ok=True)
        viz_dir = self.output_dir / "figures"
        viz_dir.mkdir(parents=True, exist_ok=True)
        
        try:
            self._create_correlation_heatmap(viz_dir)
            self._create_baseline_performance_chart(viz_dir)
            self._create_correlation_scatter_plots(viz_dir)
            self._create_method_ranking_chart(viz_dir)
            
            self._create_cost_analysis_chart(viz_dir)
            self._create_processing_time_comparison(viz_dir)
            self._create_agreement_analysis_charts(viz_dir)
            self._create_statistical_significance_chart(viz_dir)
            self._create_dataset_comparison_charts(viz_dir)
            self._create_correlation_matrix_3d(viz_dir)
            self._create_performance_radar_chart(viz_dir)
            
            self._create_correlation_stability_analysis(viz_dir)
            self._create_performance_evolution_timeline(viz_dir)
            self._create_baseline_robustness_analysis(viz_dir)
            self._create_task_difficulty_heatmap(viz_dir)
            
            # Task-specific visualizations
            self._create_task_specific_performance_charts(viz_dir)
            
            # Specialized visualizations for entailment inference task
            if 'entailment_inference' in self.results.get('chatgpt_results', {}):
                self._create_entailment_inference_analysis(viz_dir)
            
            print(f"✅ Generated {len(self.results.get('visualizations', {}))} visualizations in: {viz_dir}")
            
        except Exception as e:
            self.logger.warning(f"Visualization generation failed: {e}")
            print(f"⚠️  Visualization generation failed: {e}")
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
        correlation_summary = self.results['correlation_analysis'].get('correlation_summary', {})
        baseline_avg_correlations = correlation_summary.get('baseline_average_correlations', {})
        
        if not baseline_avg_correlations:
            return
        
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
        
        baselines = [item['baseline'] for item in rankings]
        correlations = [abs(item['avg_correlation']) for item in rankings]
        
        fig = go.Figure()
        
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
        # Load baseline results from separate files if not in main results
        self._load_baseline_results_for_visualization()
        
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
                            # FactCC uses binary predictions (0=CORRECT, 1=INCORRECT)
                            # Parse predictions from string representations of BaselineResult objects
                            parsed_predictions = []
                            for pred in predictions:
                                try:
                                    if isinstance(pred, str) and 'prediction=' in pred:
                                        # Extract prediction value from string representation
                                        match = re.search(r'prediction=([01])', pred)
                                        if match:
                                            parsed_predictions.append(int(match.group(1)))
                                    elif hasattr(pred, 'prediction'):
                                        parsed_predictions.append(int(pred.prediction))
                                    elif isinstance(pred, int):
                                        parsed_predictions.append(pred)
                                except (ValueError, AttributeError):
                                    continue
                            
                            if parsed_predictions:
                                # For FactCC: 1=INCORRECT (like "INCONSISTENT"), 0=CORRECT (like "CONSISTENT")  
                                # We want the rate of "positive" predictions (1=INCONSISTENT/ENTAILMENT)
                                inconsistent_rate = sum(1 for pred in parsed_predictions if pred == 1) / len(parsed_predictions)
                                baseline_performance[dataset] = inconsistent_rate
                        else:
                            # BERTScore and ROUGE use numerical scores
                            # Parse scores from string representations of BaselineResult objects
                            scores = []
                            for pred in predictions:
                                try:
                                    if isinstance(pred, str) and 'prediction=' in pred:
                                        # Extract prediction value from string representation
                                        match = re.search(r'prediction=([0-9.]+)', pred)
                                        if match:
                                            scores.append(float(match.group(1)))
                                    elif hasattr(pred, 'prediction'):
                                        scores.append(float(pred.prediction))
                                    elif isinstance(pred, (int, float)):
                                        scores.append(float(pred))
                                except (ValueError, AttributeError):
                                    continue
                            
                            if scores:
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
        self._load_baseline_results_for_visualization()
        
        # Check if we have consistency_rating data - improved detection logic
        has_consistency_data = (
            'consistency_rating' in self.results.get('chatgpt_results', {}) and
            bool(self.results['chatgpt_results']['consistency_rating'])
        ) or any(
            'consistency_rating' in baseline_data and bool(baseline_data.get('consistency_rating', {}))
            for baseline_data in self.results.get('baseline_results', {}).values()
        )
        
        # Debug logging to understand data structure
        self.logger.debug(f"Checking consistency data availability:")
        self.logger.debug(f"ChatGPT results keys: {list(self.results.get('chatgpt_results', {}).keys())}")
        if 'consistency_rating' in self.results.get('chatgpt_results', {}):
            consistency_data = self.results['chatgpt_results']['consistency_rating']
            self.logger.debug(f"ChatGPT consistency_rating datasets: {list(consistency_data.keys())}")
            for dataset, data in consistency_data.items():
                prediction_count = len(data.get('predictions', []))
                self.logger.debug(f"  {dataset}: {prediction_count} predictions")
        
        if not has_consistency_data:
            fig = go.Figure()
            fig.add_annotation(
                text="No consistency rating data available.<br>Run with --task consistency_rating to generate this chart.",
                xref="paper", yref="paper",
                x=0.5, y=0.5, xanchor='center', yanchor='middle',
                showarrow=False,
                font=dict(size=16, color="gray")
            )
            fig.update_layout(
                title='Consistency Rating: Average Scores by Method',
                xaxis_title='Dataset',
                yaxis_title='Average Score',
                height=500,
                showlegend=False,
                xaxis=dict(showgrid=False, showticklabels=False),
                yaxis=dict(showgrid=False, showticklabels=False)
            )
            fig.write_image(viz_dir / 'consistency_rating_performance.png', width=800, height=500, scale=2)
            return
        
        fig = go.Figure()
        
        datasets = ['cnn_dailymail', 'xsum']
        baselines = ['factcc', 'bertscore', 'rouge']
        
        # Calculate ChatGPT average consistency scores
        chatgpt_performance = {}
        for dataset in datasets:
            if 'consistency_rating' in self.results.get('chatgpt_results', {}):
                chatgpt_data = self.results['chatgpt_results']['consistency_rating'].get(dataset, {})
                predictions = chatgpt_data.get('predictions', [])
                scores = []
                for pred in predictions:
                    try:
                        if isinstance(pred, str) and 'prediction=' in pred:
                            match = re.search(r'prediction=([0-9.]+)', pred)
                            if match:
                                scores.append(float(match.group(1)))
                        elif hasattr(pred, 'prediction') and isinstance(pred.prediction, (int, float)):
                            scores.append(float(pred.prediction))
                        elif hasattr(pred, 'rating') and isinstance(pred.rating, (int, float)):
                            scores.append(float(pred.rating))
                        elif isinstance(pred, dict):
                            if 'prediction' in pred and isinstance(pred['prediction'], (int, float)):
                                scores.append(float(pred['prediction']))
                            elif 'rating' in pred and isinstance(pred['rating'], (int, float)):
                                scores.append(float(pred['rating']))
                        elif isinstance(pred, (int, float)):
                            scores.append(float(pred))
                        elif isinstance(pred, str) and pred.replace('.', '').isdigit():
                            scores.append(float(pred))
                    except (ValueError, AttributeError) as e:
                        self.logger.debug(f"Could not parse ChatGPT prediction: {str(pred)[:100]}... Error: {e}")
                        continue
                
                if scores:
                    chatgpt_performance[dataset] = np.mean(scores)
                    self.logger.debug(f"ChatGPT {dataset} consistency: {np.mean(scores):.2f} (from {len(scores)} scores)")
        
        if chatgpt_performance:
            fig.add_trace(go.Bar(
                name='ChatGPT',
                x=list(chatgpt_performance.keys()),
                y=list(chatgpt_performance.values()),
                marker_color='#FF6B6B',
                text=[f'{score:.1f}' for score in chatgpt_performance.values()],
                textposition='auto',
            ))
        
        colors = {'factcc': '#4ECDC4', 'bertscore': '#45B7D1', 'rouge': '#96CEB4'}
        
        for baseline in baselines:
            if baseline in self.results.get('baseline_results', {}):
                baseline_performance = {}
                for dataset in datasets:
                    baseline_data = self.results['baseline_results'][baseline].get('consistency_rating', {}).get(dataset, {})
                    
                    if 'baseline_predictions' in baseline_data:
                        predictions = baseline_data['baseline_predictions']
                        scores = []
                        
                        for pred in predictions:
                            try:
                                if isinstance(pred, str) and 'prediction=' in pred:
                                    # Handle both regular floats and numpy.float64
                                    match = re.search(r'prediction=(?:np\.float64\()?([0-9.]+)(?:\))?', pred)
                                    if match:
                                        scores.append(float(match.group(1)))
                                elif isinstance(pred, (int, float)):
                                    scores.append(float(pred))
                                elif hasattr(pred, 'prediction'):
                                    scores.append(float(pred.prediction))
                            except (ValueError, AttributeError) as e:
                                self.logger.debug(f"Could not parse prediction: {pred[:100]}... Error: {e}")
                                continue
                        
                        if scores:
                            # Baseline scores are already in appropriate scales for consistency rating
                            baseline_performance[dataset] = np.mean(scores)
                            
                            self.logger.debug(f"{baseline.upper()} {dataset} consistency: {baseline_performance[dataset]:.2f} (from {len(scores)} scores)")
                
                if baseline_performance:  # Only add trace if we have data
                    fig.add_trace(go.Bar(
                        name=baseline.upper(),
                        x=list(baseline_performance.keys()),
                        y=list(baseline_performance.values()),
                        marker_color=colors[baseline],
                        text=[f'{score:.1f}' for score in baseline_performance.values()],
                        textposition='auto',
                    ))
        
        # If still no data to plot, show informative message
        if not fig.data:
            fig.add_annotation(
                text="Consistency rating data detected but no valid scores found.<br>Check data format and prediction extraction logic.",
                xref="paper", yref="paper",
                x=0.5, y=0.5, xanchor='center', yanchor='middle',
                showarrow=False,
                font=dict(size=16, color="orange")
            )
            fig.update_layout(
                title='Consistency Rating: Average Scores by Method',
                xaxis_title='Dataset',
                yaxis_title='Average Score',
                height=500,
                showlegend=False,
                xaxis=dict(showgrid=False, showticklabels=False),
                yaxis=dict(showgrid=False, showticklabels=False)
            )
        else:
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

    def _load_baseline_results_for_visualization(self):
        """Load baseline results from separate JSON files for visualization."""
        baseline_results_dir = self.output_dir / "baseline_results"
        if not baseline_results_dir.exists():
            return
            
        for baseline_file in baseline_results_dir.glob("*_results.json"):
            baseline_name = baseline_file.stem.replace('_results', '')
            try:
                with open(baseline_file, 'r') as f:
                    baseline_data = json.load(f)
                    
                # Ensure baseline_results exists in self.results
                if 'baseline_results' not in self.results:
                    self.results['baseline_results'] = {}
                    
                self.results['baseline_results'][baseline_name] = baseline_data
                self.logger.debug(f"Loaded baseline data for {baseline_name} from {baseline_file.name}")
            except Exception as e:
                self.logger.warning(f"Could not load baseline data from {baseline_file.name}: {e}")

    def _create_cost_analysis_chart(self, viz_dir: Path):
        """Create cost analysis visualization."""
        if 'cost_analysis' not in self.results or 'chatgpt_cost' not in self.results['cost_analysis']:
            return
            
        chatgpt_cost = self.results['cost_analysis']['chatgpt_cost']
        
        # Calculate cost per task and dataset
        cost_breakdown = {}
        
        for task_name, task_data in self.results['chatgpt_results'].items():
            cost_breakdown[task_name] = {}
            for dataset_name, dataset_data in task_data.items():
                if 'cost' in dataset_data:
                    cost_breakdown[task_name][dataset_name] = dataset_data['cost']
        
        fig = go.Figure()
        
        datasets = []
        tasks = []
        costs = []
        
        for task_name, task_costs in cost_breakdown.items():
            for dataset_name, cost in task_costs.items():
                tasks.append(task_name)
                datasets.append(dataset_name)
                costs.append(cost)
        
        fig.add_trace(go.Bar(
            x=[f"{task}_{dataset}" for task, dataset in zip(tasks, datasets)],
            y=costs,
            marker_color=['#FF6B6B' if 'entailment' in task else '#4ECDC4' for task in tasks],
            text=[f'${cost:.4f}' for cost in costs],
            textposition='auto'
        ))
        
        fig.update_layout(
            title=f'ChatGPT Cost Analysis (Total: ${chatgpt_cost:.4f})',
            xaxis_title='Task-Dataset Combination',
            yaxis_title='Cost ($)',
            font=dict(family='Times New Roman', size=12),
            plot_bgcolor='white',
            paper_bgcolor='white'
        )
        
        fig.write_image(viz_dir / 'cost_analysis.png', width=1000, height=600, scale=2)
        
        if 'visualizations' not in self.results:
            self.results['visualizations'] = {}
        self.results['visualizations']['cost_analysis'] = str(viz_dir / 'cost_analysis.png')

    def _create_processing_time_comparison(self, viz_dir: Path):
        """Create processing time comparison between ChatGPT and baselines."""
        fig = make_subplots(
            rows=1, cols=2,
            subplot_titles=['ChatGPT Processing Times', 'Baseline Processing Times'],
            specs=[[{"type": "bar"}, {"type": "bar"}]]
        )
        
        # ChatGPT processing times
        chatgpt_times = []
        chatgpt_labels = []
        
        for task_name, task_data in self.results['chatgpt_results'].items():
            for dataset_name, dataset_data in task_data.items():
                if 'processing_time' in dataset_data:
                    chatgpt_times.append(dataset_data['processing_time'])
                    chatgpt_labels.append(f"{task_name}_{dataset_name}")
        
        fig.add_trace(
            go.Bar(
                x=chatgpt_labels,
                y=chatgpt_times,
                name='ChatGPT',
                marker_color='#FF6B6B',
                text=[f'{t:.1f}s' for t in chatgpt_times],
                textposition='auto'
            ),
            row=1, col=1
        )
        
        # Baseline processing times
        baseline_times = []
        baseline_labels = []
        baseline_names = []
        
        for baseline_name, baseline_data in self.results['baseline_results'].items():
            for task_name, task_data in baseline_data.items():
                for dataset_name, dataset_data in task_data.items():
                    if 'processing_time' in dataset_data:
                        baseline_times.append(dataset_data['processing_time'])
                        baseline_labels.append(f"{baseline_name}_{task_name}_{dataset_name}")
                        baseline_names.append(baseline_name)
        
        colors = {'factcc': '#4ECDC4', 'bertscore': '#45B7D1', 'rouge': '#96CEB4'}
        baseline_colors = [colors.get(name, '#888888') for name in baseline_names]
        
        fig.add_trace(
            go.Bar(
                x=baseline_labels,
                y=baseline_times,
                name='Baselines',
                marker_color=baseline_colors,
                text=[f'{t:.1f}s' for t in baseline_times],
                textposition='auto'
            ),
            row=1, col=2
        )
        
        fig.update_layout(
            title='Processing Time Comparison',
            font=dict(family='Times New Roman', size=12),
            showlegend=False
        )
        
        fig.write_image(viz_dir / 'processing_time_comparison.png', width=1200, height=600, scale=2)
        self.results['visualizations']['processing_time_comparison'] = str(viz_dir / 'processing_time_comparison.png')

    def _create_agreement_analysis_charts(self, viz_dir: Path):
        """Create comprehensive agreement analysis charts."""
        if 'correlation_analysis' not in self.results or 'agreement_metrics' not in self.results['correlation_analysis']:
            return
            
        agreement_metrics = self.results['correlation_analysis']['agreement_metrics']
        
        fig = make_subplots(
            rows=2, cols=2,
            subplot_titles=['Agreement Rates', "Cohen's Kappa", 'Precision', 'Recall'],
            specs=[[{"type": "bar"}, {"type": "bar"}],
                   [{"type": "bar"}, {"type": "bar"}]]
        )
        
        # Collect data
        baselines = []
        agreements = []
        kappas = []
        precisions = []
        recalls = []
        
        for baseline_name, baseline_agreement in agreement_metrics.items():
            for task_agreement in baseline_agreement.values():
                for dataset_agreement in task_agreement.values():
                    if isinstance(dataset_agreement, dict):
                        baselines.append(baseline_name)
                        agreements.append(dataset_agreement.get('agreement', 0))
                        kappas.append(dataset_agreement.get('cohens_kappa', 0))
                        precisions.append(dataset_agreement.get('precision', 0))
                        recalls.append(dataset_agreement.get('recall', 0))
        
        colors = {'factcc': '#4ECDC4', 'bertscore': '#45B7D1', 'rouge': '#96CEB4'}
        bar_colors = [colors.get(baseline, '#888888') for baseline in baselines]
        
        fig.add_trace(go.Bar(x=baselines, y=agreements, marker_color=bar_colors, showlegend=False), row=1, col=1)
        fig.add_trace(go.Bar(x=baselines, y=kappas, marker_color=bar_colors, showlegend=False), row=1, col=2)
        fig.add_trace(go.Bar(x=baselines, y=precisions, marker_color=bar_colors, showlegend=False), row=2, col=1)
        fig.add_trace(go.Bar(x=baselines, y=recalls, marker_color=bar_colors, showlegend=False), row=2, col=2)
        
        fig.update_layout(
            title='Agreement Metrics Analysis',
            font=dict(family='Times New Roman', size=12),
            height=800
        )
        
        fig.write_image(viz_dir / 'agreement_analysis.png', width=1200, height=800, scale=2)
        self.results['visualizations']['agreement_analysis'] = str(viz_dir / 'agreement_analysis.png')

    def _create_statistical_significance_chart(self, viz_dir: Path):
        """Create statistical significance visualization."""
        if 'statistical_analysis' not in self.results:
            return
            
        statistical_analysis = self.results['statistical_analysis']
        correlation_significance = statistical_analysis.get('correlation_significance', {})
        
        # Collect significance data
        labels = []
        p_values = []
        correlations = []
        significant = []
        
        for baseline_name, baseline_sig in correlation_significance.items():
            for task_name, task_sig in baseline_sig.items():
                for dataset_name, dataset_sig in task_sig.items():
                    if isinstance(dataset_sig, dict) and 'p_value' in dataset_sig:
                        label = f"{baseline_name}-{task_name}-{dataset_name}"
                        labels.append(label)
                        p_values.append(dataset_sig['p_value'])
                        correlations.append(abs(dataset_sig['correlation']) if not np.isnan(dataset_sig['correlation']) else 0)
                        significant.append(dataset_sig['is_significant'])
        
        if not labels:
            # Create placeholder plot if no data
            fig = go.Figure()
            fig.add_annotation(
                text="No correlation data available for significance testing.<br>This may occur when baselines have zero variance.",
                xref="paper", yref="paper",
                x=0.5, y=0.5, xanchor='center', yanchor='middle',
                showarrow=False,
                font=dict(size=14, color="gray")
            )
            fig.update_layout(
                title='Statistical Significance of Correlations',
                xaxis_title='Absolute Correlation Coefficient',
                yaxis_title='P-value',
                height=600,
                showlegend=False
            )
            fig.write_image(viz_dir / 'statistical_significance.png', width=1000, height=600, scale=2)
            return
            
        # Create significance scatter plot
        fig = go.Figure()
        
        # Significant correlations
        sig_indices = [i for i, sig in enumerate(significant) if sig]
        non_sig_indices = [i for i, sig in enumerate(significant) if not sig]
        
        if sig_indices:
            fig.add_trace(go.Scatter(
                x=[correlations[i] for i in sig_indices],
                y=[p_values[i] for i in sig_indices],
                mode='markers',
                marker=dict(color='green', size=12, symbol='circle'),
                name='Significant (p < 0.05)',
                text=[labels[i] for i in sig_indices],
                hovertemplate='%{text}<br>|r|: %{x:.3f}<br>p-value: %{y:.4f}<extra></extra>'
            ))
        
        if non_sig_indices:
            fig.add_trace(go.Scatter(
                x=[correlations[i] for i in non_sig_indices],
                y=[p_values[i] for i in non_sig_indices],
                mode='markers',
                marker=dict(color='red', size=12, symbol='x'),
                name='Not Significant',
                text=[labels[i] for i in non_sig_indices],
                hovertemplate='%{text}<br>|r|: %{x:.3f}<br>p-value: %{y:.4f}<extra></extra>'
            ))
        
        # Add significance threshold line
        fig.add_hline(y=0.05, line_dash="dash", line_color="black",
                     annotation_text="p = 0.05 (significance threshold)", 
                     annotation_position="top right")
        
        # Add interpretation text
        sig_count = len(sig_indices)
        total_count = len(labels)
        fig.add_annotation(
            text=f"Significant correlations: {sig_count}/{total_count}<br>" +
                 f"Most correlations are not significant due to<br>" +
                 f"baseline models having limited variance.",
            xref="paper", yref="paper",
            x=0.02, y=0.98, xanchor='left', yanchor='top',
            showarrow=False,
            font=dict(size=10, color="darkblue"),
            bgcolor="rgba(255,255,255,0.8)",
            bordercolor="darkblue",
            borderwidth=1
        )
        
        fig.update_layout(
            title='Statistical Significance of Correlations',
            xaxis_title='Absolute Correlation Coefficient',
            yaxis_title='P-value',
            yaxis_type='log',
            font=dict(family='Times New Roman', size=12),
            plot_bgcolor='white',
            paper_bgcolor='white',
            height=600
        )
        
        fig.write_image(viz_dir / 'statistical_significance.png', width=1000, height=600, scale=2)
        self.results['visualizations']['statistical_significance'] = str(viz_dir / 'statistical_significance.png')

    def _create_dataset_comparison_charts(self, viz_dir: Path):
        """Create dataset-specific comparison charts."""
        datasets = set()
        for task_data in self.results['chatgpt_results'].values():
            datasets.update(task_data.keys())
        
        datasets = sorted(list(datasets))
        
        if len(datasets) < 2:
            return
            
        fig = make_subplots(
            rows=1, cols=len(datasets),
            subplot_titles=[f'{dataset.upper()}' for dataset in datasets]
        )
        
        for col, dataset in enumerate(datasets, 1):
            baseline_correlations = {}
            
            # Collect correlations for this dataset
            pearson_correlations = self.results['correlation_analysis']['pearson_correlations']
            for baseline_name, baseline_corr in pearson_correlations.items():
                correlations = []
                for task_corr in baseline_corr.values():
                    if dataset in task_corr and 'correlation' in task_corr[dataset]:
                        corr_val = task_corr[dataset]['correlation']
                        if not np.isnan(corr_val):
                            correlations.append(abs(corr_val))
                
                if correlations:
                    baseline_correlations[baseline_name] = np.mean(correlations)
            
            if baseline_correlations:
                baselines = list(baseline_correlations.keys())
                correlations = list(baseline_correlations.values())
                
                colors = {'factcc': '#4ECDC4', 'bertscore': '#45B7D1', 'rouge': '#96CEB4'}
                bar_colors = [colors.get(baseline, '#888888') for baseline in baselines]
                
                fig.add_trace(
                    go.Bar(
                        x=baselines,
                        y=correlations,
                        marker_color=bar_colors,
                        showlegend=False,
                        text=[f'{corr:.3f}' for corr in correlations],
                        textposition='auto'
                    ),
                    row=1, col=col
                )
        
        fig.update_layout(
            title='Baseline Performance Comparison Across Datasets',
            font=dict(family='Times New Roman', size=12),
            height=500
        )
        
        fig.write_image(viz_dir / 'dataset_comparison.png', width=1200, height=500, scale=2)
        self.results['visualizations']['dataset_comparison'] = str(viz_dir / 'dataset_comparison.png')

    def _create_correlation_matrix_3d(self, viz_dir: Path):
        """Create 3D correlation matrix visualization."""
        pearson_correlations = self.results['correlation_analysis']['pearson_correlations']
        
        # Prepare data for 3D visualization
        x_data = []  # Baselines
        y_data = []  # Task-Dataset combinations
        z_data = []  # Correlations
        
        for baseline_name, baseline_corr in pearson_correlations.items():
            for task_name, task_corr in baseline_corr.items():
                for dataset_name, dataset_corr in task_corr.items():
                    if isinstance(dataset_corr, dict) and 'correlation' in dataset_corr:
                        correlation = dataset_corr['correlation']
                        if not np.isnan(correlation):
                            x_data.append(baseline_name)
                            y_data.append(f"{task_name}_{dataset_name}")
                            z_data.append(correlation)
        
        if not x_data:
            return
            
        # Create 3D surface plot
        fig = go.Figure(data=[go.Scatter3d(
            x=x_data,
            y=y_data,
            z=z_data,
            mode='markers',
            marker=dict(
                size=8,
                color=z_data,
                colorscale='RdBu',
                colorbar=dict(title="Correlation"),
                cmin=-1,
                cmax=1
            ),
            text=[f'{baseline}<br>{task_dataset}<br>r={corr:.3f}' 
                  for baseline, task_dataset, corr in zip(x_data, y_data, z_data)],
            hovertemplate='%{text}<extra></extra>'
        )])
        
        fig.update_layout(
            title='3D Correlation Visualization',
            scene=dict(
                xaxis_title='Baseline',
                yaxis_title='Task-Dataset',
                zaxis_title='Correlation'
            ),
            font=dict(family='Times New Roman', size=12)
        )
        
        fig.write_image(viz_dir / 'correlation_3d.png', width=1000, height=800, scale=2)
        self.results['visualizations']['correlation_3d'] = str(viz_dir / 'correlation_3d.png')

    def _create_performance_radar_chart(self, viz_dir: Path):
        """Create radar chart comparing baseline performance across different metrics."""
        if 'correlation_analysis' not in self.results:
            return
            
        correlation_summary = self.results['correlation_analysis'].get('correlation_summary', {})
        baseline_avg_correlations = correlation_summary.get('correlations', {}).get('baseline_average_correlations', {})
        baseline_avg_agreement = correlation_summary.get('agreement_metrics', {}).get('baseline_average_agreement', {})
        
        if not baseline_avg_correlations and not baseline_avg_agreement:
            return
            
        # Prepare radar chart data
        categories = ['Correlation Strength', 'Agreement Rate', 'Processing Speed', 'Task Coverage']
        
        fig = go.Figure()
        
        # Get processing speed data (inverse of processing time)
        baseline_speeds = {}
        for baseline_name, baseline_data in self.results['baseline_results'].items():
            times = []
            for task_data in baseline_data.values():
                for dataset_data in task_data.values():
                    if 'processing_time' in dataset_data:
                        times.append(dataset_data['processing_time'])
            if times:
                # Normalize speed (higher is better, so use inverse)
                avg_time = np.mean(times)
                baseline_speeds[baseline_name] = 1 / (avg_time + 0.1)  # Add small value to avoid division by zero
        
        # Get task coverage (how many tasks each baseline supports)
        baseline_coverage = {}
        total_tasks = len(self.results['chatgpt_results'])
        for baseline_name, baseline_data in self.results['baseline_results'].items():
            coverage = len(baseline_data) / total_tasks if total_tasks > 0 else 0
            baseline_coverage[baseline_name] = coverage
        
        # Normalize all metrics to 0-1 scale
        def normalize_dict(d):
            if not d:
                return {}
            max_val = max(d.values())
            min_val = min(d.values())
            if max_val == min_val:
                return {k: 0.5 for k in d.keys()}
            return {k: (v - min_val) / (max_val - min_val) for k, v in d.items()}
        
        norm_correlations = normalize_dict({k: abs(v) for k, v in baseline_avg_correlations.items()})
        norm_agreement = normalize_dict(baseline_avg_agreement)
        norm_speeds = normalize_dict(baseline_speeds)
        norm_coverage = normalize_dict(baseline_coverage)
        
        colors = {'factcc': '#4ECDC4', 'bertscore': '#45B7D1', 'rouge': '#96CEB4'}
        
        for baseline_name in set(list(norm_correlations.keys()) + list(norm_agreement.keys())):
            values = [
                norm_correlations.get(baseline_name, 0),
                norm_agreement.get(baseline_name, 0),
                norm_speeds.get(baseline_name, 0),
                norm_coverage.get(baseline_name, 0)
            ]
            
            # Close the radar chart
            values.append(values[0])
            categories_closed = categories + [categories[0]]
            
            fig.add_trace(go.Scatterpolar(
                r=values,
                theta=categories_closed,
                fill='toself',
                name=baseline_name.upper(),
                line_color=colors.get(baseline_name, '#888888'),
                fillcolor=colors.get(baseline_name, '#888888'),
                opacity=0.3
            ))
        
        fig.update_layout(
            polar=dict(
                radialaxis=dict(
                    visible=True,
                    range=[0, 1]
                )
            ),
            title='Baseline Performance Radar Chart (Normalized)',
            font=dict(family='Times New Roman', size=12)
        )
        
        fig.write_image(viz_dir / 'performance_radar.png', width=800, height=800, scale=2)
        self.results['visualizations']['performance_radar'] = str(viz_dir / 'performance_radar.png')

    def _create_correlation_stability_analysis(self, viz_dir: Path):
        """Create correlation stability analysis comparing different correlation measures."""
        if 'correlation_analysis' not in self.results:
            return
            
        pearson_correlations = self.results['correlation_analysis'].get('pearson_correlations', {})
        spearman_correlations = self.results['correlation_analysis'].get('spearman_correlations', {})
        
        if not pearson_correlations or not spearman_correlations:
            return
            
        # Collect paired correlation data
        pearson_values = []
        spearman_values = []
        labels = []
        
        for baseline_name in pearson_correlations.keys():
            if baseline_name not in spearman_correlations:
                continue
                
            for task_name in pearson_correlations[baseline_name].keys():
                if task_name not in spearman_correlations[baseline_name]:
                    continue
                    
                for dataset_name in pearson_correlations[baseline_name][task_name].keys():
                    if dataset_name not in spearman_correlations[baseline_name][task_name]:
                        continue
                        
                    pearson_data = pearson_correlations[baseline_name][task_name][dataset_name]
                    spearman_data = spearman_correlations[baseline_name][task_name][dataset_name]
                    
                    if isinstance(pearson_data, dict) and isinstance(spearman_data, dict):
                        pearson_corr = pearson_data.get('correlation', 0)
                        spearman_corr = spearman_data.get('correlation', 0)
                        
                        if not np.isnan(pearson_corr) and not np.isnan(spearman_corr):
                            pearson_values.append(pearson_corr)
                            spearman_values.append(spearman_corr)
                            labels.append(f"{baseline_name}_{task_name}_{dataset_name}")
        
        if not pearson_values:
            return
            
        # Create stability comparison plot
        fig = go.Figure()
        
        # Scatter plot
        fig.add_trace(go.Scatter(
            x=pearson_values,
            y=spearman_values,
            mode='markers+text',
            text=labels,
            textposition='top center',
            marker=dict(
                size=10,
                color=pearson_values,
                colorscale='RdBu',
                colorbar=dict(title="Pearson Correlation"),
                line=dict(width=1, color='black')
            ),
            name='Correlations'
        ))
        
        # Add perfect correlation line
        min_val = min(min(pearson_values), min(spearman_values))
        max_val = max(max(pearson_values), max(spearman_values))
        fig.add_trace(go.Scatter(
            x=[min_val, max_val],
            y=[min_val, max_val],
            mode='lines',
            line=dict(color='red', dash='dash'),
            name='Perfect Agreement',
            showlegend=False
        ))
        
        fig.update_layout(
            title='Correlation Stability: Pearson vs Spearman',
            xaxis_title='Pearson Correlation',
            yaxis_title='Spearman Correlation',
            font=dict(family='Times New Roman', size=12),
            plot_bgcolor='white',
            paper_bgcolor='white'
        )
        
        fig.write_image(viz_dir / 'correlation_stability_analysis.png', width=1000, height=800, scale=2)
        self.results['visualizations']['correlation_stability_analysis'] = str(viz_dir / 'correlation_stability_analysis.png')

    def _create_performance_evolution_timeline(self, viz_dir: Path):
        """Create timeline showing performance evolution across different metrics."""
        if 'chatgpt_results' not in self.results:
            return
            
        timeline_data = []
        
        for task_name, task_data in self.results['chatgpt_results'].items():
            for dataset_name, dataset_data in task_data.items():
                if 'performance_metrics' in dataset_data:
                    metrics = dataset_data['performance_metrics']
                    timeline_data.append({
                        'task_dataset': f"{task_name}_{dataset_name}",
                        'performance': metrics.get('primary_metric', 0),
                        'cost': dataset_data.get('cost', 0),
                        'processing_time': dataset_data.get('processing_time', 0)
                    })
        
        if not timeline_data:
            return
            
        # Sort by performance
        timeline_data.sort(key=lambda x: x['performance'])
        
        fig = make_subplots(
            rows=3, cols=1,
            subplot_titles=['Performance Scores', 'Processing Costs', 'Processing Times'],
            vertical_spacing=0.08
        )
        
        x_labels = [item['task_dataset'] for item in timeline_data]
        
        # Performance evolution
        fig.add_trace(
            go.Scatter(
                x=x_labels,
                y=[item['performance'] for item in timeline_data],
                mode='lines+markers',
                name='Performance',
                line=dict(color='#FF6B6B', width=3),
                marker=dict(size=8)
            ),
            row=1, col=1
        )
        
        # Cost evolution
        fig.add_trace(
            go.Scatter(
                x=x_labels,
                y=[item['cost'] for item in timeline_data],
                mode='lines+markers',
                name='Cost',
                line=dict(color='#4ECDC4', width=3),
                marker=dict(size=8)
            ),
            row=2, col=1
        )
        
        # Processing time evolution
        fig.add_trace(
            go.Scatter(
                x=x_labels,
                y=[item['processing_time'] for item in timeline_data],
                mode='lines+markers',
                name='Time',
                line=dict(color='#45B7D1', width=3),
                marker=dict(size=8)
            ),
            row=3, col=1
        )
        
        fig.update_layout(
            title='Performance Evolution Timeline',
            font=dict(family='Times New Roman', size=12),
            height=900,
            showlegend=False
        )
        
        fig.write_image(viz_dir / 'performance_evolution_timeline.png', width=1200, height=900, scale=2)
        self.results['visualizations']['performance_evolution_timeline'] = str(viz_dir / 'performance_evolution_timeline.png')

    def _create_baseline_robustness_analysis(self, viz_dir: Path):
        """Create robustness analysis showing baseline performance consistency."""
        if 'correlation_analysis' not in self.results:
            return
            
        pearson_correlations = self.results['correlation_analysis'].get('pearson_correlations', {})
        
        # Calculate robustness metrics for each baseline
        robustness_data = {}
        
        for baseline_name, baseline_data in pearson_correlations.items():
            correlations = []
            
            for task_name, task_data in baseline_data.items():
                for dataset_name, dataset_result in task_data.items():
                    if isinstance(dataset_result, dict):
                        corr = dataset_result.get('correlation', 0)
                        if not np.isnan(corr):
                            correlations.append(abs(corr))
            
            if correlations:
                robustness_data[baseline_name] = {
                    'mean': np.mean(correlations),
                    'std': np.std(correlations),
                    'min': np.min(correlations),
                    'max': np.max(correlations),
                    'consistency': 1 - (np.std(correlations) / np.mean(correlations)) if np.mean(correlations) > 0 else 0
                }
        
        if not robustness_data:
            return
            
        # Create robustness visualization
        fig = make_subplots(
            rows=2, cols=2,
            subplot_titles=['Mean Performance', 'Performance Variance', 'Performance Range', 'Consistency Score'],
            specs=[[{"type": "bar"}, {"type": "bar"}],
                   [{"type": "bar"}, {"type": "bar"}]]
        )
        
        baselines = list(robustness_data.keys())
        colors = {'factcc': '#4ECDC4', 'bertscore': '#45B7D1', 'rouge': '#96CEB4'}
        bar_colors = [colors.get(baseline, '#888888') for baseline in baselines]
        
        # Mean performance
        fig.add_trace(
            go.Bar(x=baselines, y=[robustness_data[b]['mean'] for b in baselines], 
                   marker_color=bar_colors, showlegend=False),
            row=1, col=1
        )
        
        # Performance variance
        fig.add_trace(
            go.Bar(x=baselines, y=[robustness_data[b]['std'] for b in baselines], 
                   marker_color=bar_colors, showlegend=False),
            row=1, col=2
        )
        
        # Performance range
        fig.add_trace(
            go.Bar(x=baselines, y=[robustness_data[b]['max'] - robustness_data[b]['min'] for b in baselines], 
                   marker_color=bar_colors, showlegend=False),
            row=2, col=1
        )
        
        # Consistency score
        fig.add_trace(
            go.Bar(x=baselines, y=[robustness_data[b]['consistency'] for b in baselines], 
                   marker_color=bar_colors, showlegend=False),
            row=2, col=2
        )
        
        fig.update_layout(
            title='Baseline Robustness Analysis',
            font=dict(family='Times New Roman', size=12),
            height=800
        )
        
        fig.write_image(viz_dir / 'baseline_robustness_analysis.png', width=1200, height=800, scale=2)
        self.results['visualizations']['baseline_robustness_analysis'] = str(viz_dir / 'baseline_robustness_analysis.png')

    def _create_task_difficulty_heatmap(self, viz_dir: Path):
        """Create heatmap showing task difficulty based on multiple metrics."""
        if 'chatgpt_results' not in self.results or 'baseline_results' not in self.results:
            return
            
        # Collect difficulty metrics
        task_difficulty = {}
        
        for task_name, task_data in self.results['chatgpt_results'].items():
            difficulty_metrics = {}
            
            # Average performance (lower = more difficult)
            performances = []
            costs = []
            times = []
            
            for dataset_name, dataset_data in task_data.items():
                if 'performance_metrics' in dataset_data:
                    performances.append(dataset_data['performance_metrics'].get('primary_metric', 0))
                costs.append(dataset_data.get('cost', 0))
                times.append(dataset_data.get('processing_time', 0))
            
            if performances:
                difficulty_metrics['low_performance'] = 1 - np.mean(performances)  # Invert so higher = more difficult
                difficulty_metrics['high_cost'] = np.mean(costs) * 1000  # Scale for visibility
                difficulty_metrics['long_processing'] = np.mean(times)
                
                # Baseline disagreement (higher = more difficult)
                correlations = []
                if task_name in self.results.get('correlation_analysis', {}).get('pearson_correlations', {}):
                    for baseline_data in self.results['correlation_analysis']['pearson_correlations'].values():
                        if task_name in baseline_data:
                            for dataset_data in baseline_data[task_name].values():
                                if isinstance(dataset_data, dict):
                                    corr = dataset_data.get('correlation', 0)
                                    if not np.isnan(corr):
                                        correlations.append(abs(corr))
                
                difficulty_metrics['baseline_disagreement'] = 1 - np.mean(correlations) if correlations else 0.5
                
                task_difficulty[task_name] = difficulty_metrics
        
        if not task_difficulty:
            return
            
        tasks = list(task_difficulty.keys())
        metrics = ['low_performance', 'high_cost', 'long_processing', 'baseline_disagreement']
        metric_labels = ['Low Performance', 'High Cost', 'Long Processing', 'Baseline Disagreement']
        
        # Normalize data for heatmap
        heatmap_data = []
        for metric in metrics:
            metric_values = [task_difficulty[task][metric] for task in tasks]
            max_val = max(metric_values) if metric_values else 1
            min_val = min(metric_values) if metric_values else 0
            
            if max_val == min_val:
                normalized = [0.5] * len(metric_values)
            else:
                normalized = [(val - min_val) / (max_val - min_val) for val in metric_values]
            
            heatmap_data.append(normalized)
        
        fig = go.Figure(data=go.Heatmap(
            z=heatmap_data,
            x=[task.replace('_', ' ').title() for task in tasks],
            y=metric_labels,
            colorscale='Reds',
            colorbar=dict(title="Difficulty Level"),
            text=[[f'{task_difficulty[tasks[j]][metrics[i]]:.3f}' for j in range(len(tasks))] for i in range(len(metrics))],
            texttemplate='%{text}',
            textfont={"size": 10}
        ))
        
        fig.update_layout(
            title='Task Difficulty Analysis Heatmap',
            font=dict(family='Times New Roman', size=12),
            height=500
        )
        
        fig.write_image(viz_dir / 'task_difficulty_heatmap.png', width=800, height=500, scale=2)
        self.results['visualizations']['task_difficulty_heatmap'] = str(viz_dir / 'task_difficulty_heatmap.png')

    async def _save_results(self):
        """Save comprehensive results and generate report."""
        self.logger.info("Saving results and generating report")
        
        # Create intermediate_results subfolder for better organization
        intermediate_results_dir = self.output_dir / "intermediate_results"
        intermediate_results_dir.mkdir(exist_ok=True)
        
        # Move intermediate result files from general results folder to experiment folder
        # Use experiment timestamp to avoid moving files from other experiments
        general_results_dir = Path("results")
        if general_results_dir.exists():
            # Extract timestamp from experiment name for precise file matching
            experiment_timestamp = self.experiment_name.split('_')[-2:]  # Get last two parts (date_time)
            if len(experiment_timestamp) == 2:
                date_part, time_part = experiment_timestamp
                # Start with specific hour matching (first 2 digits of time)
                hour_pattern = f"*_intermediate_{date_part}_{time_part[0:2]}*.json"
                
                moved_files = []
                for intermediate_file in general_results_dir.glob(hour_pattern):
                    try:
                        destination = intermediate_results_dir / intermediate_file.name
                        if not destination.exists():  # Avoid duplicate moves
                            intermediate_file.rename(destination)
                            moved_files.append(intermediate_file.name)
                    except Exception as e:
                        self.logger.warning(f"Could not move intermediate file {intermediate_file.name}: {e}")
                
                # If no files found with hour matching, try broader time window
                # This accounts for intermediate files generated during the experiment run
                if not moved_files:
                    # Try within ±1 hour window to catch files generated during execution
                    current_hour = int(time_part[0:2])
                    for hour_offset in [-1, 0, 1]:
                        search_hour = (current_hour + hour_offset) % 24
                        broader_pattern = f"*_intermediate_{date_part}_{search_hour:02d}*.json"
                        
                        for intermediate_file in general_results_dir.glob(broader_pattern):
                            try:
                                destination = intermediate_results_dir / intermediate_file.name
                                if not destination.exists():  # Avoid duplicate moves
                                    intermediate_file.rename(destination)
                                    moved_files.append(intermediate_file.name)
                            except Exception as e:
                                self.logger.warning(f"Could not move intermediate file {intermediate_file.name}: {e}")
                
                # Final fallback: all files from the same date
                if not moved_files:
                    self.logger.warning(f"No intermediate files found with hour matching. Trying date-only pattern.")
                    fallback_pattern = f"*_intermediate_{date_part}*.json"
                    for intermediate_file in general_results_dir.glob(fallback_pattern):
                        try:
                            destination = intermediate_results_dir / intermediate_file.name
                            if not destination.exists():  # Avoid duplicate moves
                                intermediate_file.rename(destination)
                                moved_files.append(intermediate_file.name)
                        except Exception as e:
                            self.logger.warning(f"Could not move intermediate file {intermediate_file.name}: {e}")
                
                if moved_files:
                    print(f"📁 Organized {len(moved_files)} intermediate files into {intermediate_results_dir}")
                else:
                    self.logger.warning("No intermediate files found to move. This might indicate the tasks didn't generate intermediate results.")
            else:
                self.logger.warning(f"Could not parse experiment timestamp from name: {self.experiment_name}")
        
        # Save main results as JSON
        json_path = self.output_dir / "sota_comparison_results.json"
        with open(json_path, 'w') as f:
            json.dump(self.results, f, indent=2, default=str)
        
        # Save individual baseline results in baseline_results folder
        baseline_results_dir = self.output_dir / "baseline_results"
        for baseline_name, baseline_data in self.results.get('baseline_results', {}).items():
            baseline_file = baseline_results_dir / f"{baseline_name}_results.json"
            with open(baseline_file, 'w') as f:
                json.dump(baseline_data, f, indent=2, default=str)
        
        # Generate markdown report
        report_path = self.output_dir / "sota_comparison_report.md"
        with open(report_path, 'w') as f:
            f.write(self._generate_comparison_report())
        
        self.logger.info(f"Results saved to: {self.output_dir}")
        self.logger.info(f"Baseline results saved to: {baseline_results_dir}")
        self.logger.info(f"Intermediate results saved to: {intermediate_results_dir}")
        self.logger.info(f"Figures saved to: {self.output_dir / 'figures'}")
        self.logger.info(f"Logs saved to: {self.output_dir / 'logs'}")
    
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
        
        baseline_avg_correlations = correlation_summary.get('baseline_average_correlations', {})
        if baseline_avg_correlations:
            report += "### Baseline Performance Summary\n\n"
            
            for baseline, correlation in sorted(baseline_avg_correlations.items(), 
                                               key=lambda x: abs(x[1]), reverse=True):
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
        
        statistical_analysis = self.results.get('statistical_analysis', {})
        significance_summary = statistical_analysis.get('significance_summary', {})
        
        if significance_summary:
            report += "## Statistical Significance Analysis\n\n"
            report += f"- **Significant Correlations**: {significance_summary.get('significant_correlations', 0)}/{significance_summary.get('total_correlations', 0)}\n"
            report += f"- **Significance Rate**: {significance_summary.get('significance_rate', 0):.2f}\n"
            report += f"- **Interpretation**: {significance_summary.get('interpretation', 'No interpretation available')}\n\n"
        
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
        "--model",
        type=str,
        default="gpt-4.1-mini",
        choices=["gpt-4.1-mini", "gpt-4o-mini", "o1-mini", "gpt-4o"],
        help="OpenAI model to use"
    )
    parser.add_argument(
        "--tier",
        type=str,
        default="tier2",
        choices=["tier1", "tier2", "tier3", "tier4", "tier5"],
        help="OpenAI API tier"
    )
    parser.add_argument(
        "--config",
        type=str,
        help="DEPRECATED: Use --model and --tier instead"
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
    
    # Check for deprecated config argument
    if args.config:
        print("⚠️  Warning: --config is deprecated. Please use --model and --tier instead.")
        print(f"   Using defaults: --model {args.model} --tier {args.tier}")
    
    # Set parameters
    tasks = [args.task] if args.task else None
    datasets = [args.dataset] if args.dataset else None
    baselines = [args.baseline] if args.baseline else None
    sample_size = args.sample_size
    
    if args.quick_test:
        sample_size = 50
        print("Running quick test with 50 examples per dataset")
    
    # Initialize and run experiment
    experiment = SOTAComparisonExperiment(
        model=args.model,
        tier=args.tier,
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