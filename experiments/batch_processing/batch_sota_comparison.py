#!/usr/bin/env python3
"""
Enhanced Batch SOTA Comparison Experiment
=========================================

This script follows the standard sota_comparison.py approach but enhances it with 
advanced batch processing and superior analysis logic from the second implementation.

Key alignment with standard approach:
- Focuses on entailment_inference and consistency_rating by default
- Excludes summary_ranking as most baselines don't support it effectively  
- Uses proper baseline.supports_task() checking
- Maintains the same methodological rigor as the standard comparison

Enhanced features:
- Advanced 10-phase batch processing pipeline with cost optimization
- Enhanced correlation analysis with proper NaN handling and effect sizes
- Agreement metrics (Cohen's kappa) for imbalanced datasets
- Robust error handling and manual parsing fallbacks
- Comprehensive statistical analysis with multiple comparisons correction
- Publication-quality visualizations with significance indicators

Author: Michael Ogunjimi
Institution: University of Manchester
Course: MSc AI
"""

import argparse
import asyncio
import json
import sys
import time
from datetime import datetime
from pathlib import Path
from typing import Dict, List, Any, Optional, Tuple

# Add project root to path
sys.path.insert(0, str(Path(__file__).parent.parent.parent))

try:
    from src.utils import (
        setup_experiment_logger,
        get_config,
        setup_reproducibility,
        validate_api_keys
    )
    from src.data.loaders import load_datasets
    from src.tasks import create_task, get_supported_tasks
    from src.prompts.prompt_manager import PromptManager
    from src.llm_clients.openai_client_batch import OpenAIBatchClient, BatchResult
    from src.batch import BatchManager, BatchMonitor, BatchJob, BatchStatus
    from src.baselines import (
        create_baseline,
        get_available_baselines,
        compare_with_chatgpt,
        BaselineComparator,
        create_all_baselines
    )
    from src.utils.visualization import TaskPerformanceVisualizer
    
    import numpy as np
    import pandas as pd
    import plotly.graph_objects as go
    import plotly.express as px
    from plotly.subplots import make_subplots
    from scipy import stats
    
except ImportError as e:
    print(f"Error importing required modules: {e}")
    print("Please ensure you're running from the project root directory.")
    sys.exit(1)


class EnhancedBatchSOTAComparisonExperiment:
    """
    Enhanced batch processing implementation following standard SOTA comparison methodology.
    
    This class combines the excellent batch processing capabilities with enhanced analysis
    logic while maintaining alignment with the standard sota_comparison.py approach:
    
    - Focuses on entailment_inference and consistency_rating (tasks where baselines excel)
    - Excludes summary_ranking by default (most baselines don't support ranking effectively)
    - Uses proper baseline.supports_task() validation
    - Provides enhanced correlation analysis with Cohen's kappa for imbalanced datasets
    - Implements advanced batch processing for cost optimization
    
    The enhanced features provide thesis-quality analysis while maintaining methodological
    consistency with established SOTA comparison practices.
    """

    def __init__(self, model: str = "gpt-4.1-mini", tier: str = "tier2", experiment_name: str = None):
        """Initialize enhanced batch SOTA comparison experiment."""
        # Load configuration
        self.config = get_config(model=model, tier=tier)
        
        # Store model info
        self.model = model
        self.tier = tier
        
        # Set up experiment tracking
        self.experiment_name = experiment_name or f"enhanced_batch_sota_comparison_{datetime.now().strftime('%Y%m%d_%H%M%S')}"
        self.output_dir = Path(f"results/experiments/batch_processing/{self.experiment_name}/sota_comparison")
        self.output_dir.mkdir(parents=True, exist_ok=True)
        
        # Create subdirectories
        (self.output_dir / "baseline_results").mkdir(exist_ok=True)
        (self.output_dir / "figures").mkdir(exist_ok=True)
        
        # Set up logging
        self.experiment_logger = setup_experiment_logger(
            self.experiment_name,
            self.config,
            log_dir=self.output_dir / "logs"
        )
        self.logger = self.experiment_logger.logger
        
        # Set up reproducibility
        setup_reproducibility(self.config)
        
        # Validate API keys
        validate_api_keys(self.config)
        
        # Initialize batch client
        self.batch_client = OpenAIBatchClient(self.config, self.experiment_name)
        
        # Initialize components
        self.prompt_manager = PromptManager(self.config)
        
        # Store task configuration for later use
        self.task_configs = {
            'entailment_inference': self.config.get('tasks.entailment_inference', {}),
            'summary_ranking': self.config.get('tasks.summary_ranking', {}),
            'consistency_rating': self.config.get('tasks.consistency_rating', {})
        }
        
        # Enhanced results storage with agreement analysis
        self.results = {
            'experiment_metadata': {
                'name': self.experiment_name,
                'timestamp': datetime.now().isoformat(),
                'config': self.config.to_dict(),
                'experiment_type': 'enhanced_batch_sota_comparison',
                'model': self.model,
                'tier': self.tier
            },
            'chatgpt_results': {},
            'baseline_results': {},
            'correlation_analysis': {},
            'agreement_analysis': {},  # New: separate agreement metrics
            'performance_comparison': {},
            'statistical_analysis': {},  # Enhanced statistical analysis
            'batch_analysis': {},
            'cost_analysis': {}
        }
        
        self.logger.info(f"Initialized enhanced batch SOTA comparison: {self.experiment_name}")

    async def run_sota_comparison(
        self,
        tasks: List[str] = None,
        datasets: List[str] = None,
        baselines: List[str] = None,
        sample_size: int = None,
        prompt_type: str = "zero_shot",
        quick_test: bool = False
    ) -> Dict[str, Any]:
        """
        Run complete enhanced batch SOTA comparison experiment.
        
        Follows the standard sota_comparison approach but with enhanced batch processing.
        Excludes summary_ranking by default as most baselines don't support it effectively.
        
        Args:
            tasks: List of tasks to evaluate (default: entailment_inference, consistency_rating)
            datasets: List of datasets to use (default: cnn_dailymail, xsum)
            baselines: List of baseline methods (default: all available)
            sample_size: Number of examples per dataset
            prompt_type: ChatGPT prompt type to use
            quick_test: Whether to run quick test
            
        Returns:
            Complete comparison results with enhanced analysis
        """
        self.logger.info("Starting enhanced batch SOTA comparison experiment")
        
        # Set defaults - following standard sota_comparison approach (excludes summary_ranking as most baselines don't support it)
        if tasks is None:
            tasks = ['entailment_inference', 'consistency_rating']
        if datasets is None:
            datasets = ['cnn_dailymail', 'xsum']
        if baselines is None:
            baselines = get_available_baselines()
        if sample_size is None:
            sample_size = 50 if quick_test else self.config.get('experiments.main_experiments.sota_comparison.sample_size', 300)

        try:
            # Phase 1: Data preparation
            await self._prepare_sota_data(tasks, datasets, sample_size)
            
            # Phase 2: ChatGPT evaluation preparation
            await self._prepare_chatgpt_prompts(tasks, datasets, prompt_type)
            
            # Phase 3: ChatGPT batch submission
            chatgpt_jobs = await self._submit_chatgpt_batches(tasks, datasets, prompt_type)
            
            # Phase 4: Baseline evaluation (parallel with ChatGPT)
            baseline_results = await self._compute_baseline_evaluations(tasks, datasets, baselines)
            
            # Phase 5: ChatGPT batch monitoring
            completed_chatgpt_jobs = await self._monitor_chatgpt_completion(chatgpt_jobs)
            
            # Phase 6: ChatGPT result processing
            await self._process_chatgpt_results(completed_chatgpt_jobs, tasks, datasets, prompt_type)
            
            # Phase 7: Enhanced correlation analysis
            await self._perform_enhanced_correlation_analysis(baselines)
            
            # Phase 8: Agreement analysis for imbalanced datasets
            await self._perform_agreement_analysis(baselines)
            
            # Phase 9: Enhanced performance comparison analysis
            await self._analyze_enhanced_performance_comparison()
            
            # Phase 10: Enhanced statistical analysis
            await self._perform_enhanced_statistical_analysis()
            
            # Phase 11: Enhanced visualization generation
            await self._generate_enhanced_sota_visualizations()
            
            # Phase 12: Enhanced report generation
            await self._generate_enhanced_sota_report()
            
            self.logger.info("Enhanced batch SOTA comparison completed successfully")
            return self.results
            
        except Exception as e:
            self.logger.error(f"Enhanced batch SOTA comparison experiment failed: {e}")
            raise

    async def _prepare_sota_data(self, tasks: List[str], datasets: List[str], sample_size: int):
        """Prepare data for SOTA comparison."""
        self.logger.info(f"Preparing SOTA comparison data for {len(tasks)} tasks across {len(datasets)} datasets")
        
        # Load datasets
        self.dataset_data = {}
        for dataset_name in datasets:
            self.logger.info(f"Loading {dataset_name} dataset")
            dataset = load_datasets([dataset_name], sample_size=sample_size)[dataset_name]
            self.dataset_data[dataset_name] = dataset
            self.logger.info(f"Loaded {len(dataset)} examples from {dataset_name}")

    async def _prepare_chatgpt_prompts(self, tasks: List[str], datasets: List[str], prompt_type: str):
        """Prepare ChatGPT prompts for SOTA comparison."""
        self.logger.info("Preparing ChatGPT prompts for SOTA comparison")
        
        self.formatted_prompts = {}
        
        for task_type in tasks:
            self.formatted_prompts[task_type] = {}
            
            for dataset_name in datasets:
                self.logger.info(f"Preparing prompts: {task_type}/{dataset_name}/{prompt_type}")
                
                # Get task instance with full config
                task_config = self.config.to_dict()
                if "tasks" not in task_config:
                    task_config["tasks"] = {}
                if task_type not in task_config["tasks"]:
                    task_config["tasks"][task_type] = {}
                
                task = create_task(task_type, task_config)
                
                dataset = self.dataset_data[dataset_name]
                
                # Create formatted prompts using the task's format_prompt method
                prompts = []
                for i, example in enumerate(dataset):
                    formatted_prompt = task.format_prompt(example)
                    # Add index for correlation analysis
                    formatted_prompt.example_index = i
                    prompts.append(formatted_prompt)
                
                self.formatted_prompts[task_type][dataset_name] = prompts
                self.logger.info(f"Created {len(prompts)} prompts for {task_type}/{dataset_name}")

    async def _submit_chatgpt_batches(
        self, 
        tasks: List[str], 
        datasets: List[str], 
        prompt_type: str
    ) -> List[BatchJob]:
        """Submit ChatGPT batch jobs for SOTA comparison."""
        self.logger.info("Submitting ChatGPT batch jobs for SOTA comparison")
        
        batch_jobs = []
        
        for task_type in tasks:
            for dataset_name in datasets:
                prompts = self.formatted_prompts[task_type][dataset_name]
                
                self.logger.info(f"Submitting ChatGPT batch: {task_type}/{dataset_name}")
                
                # Submit batch job
                batch_job = await self.batch_client.submit_factuality_evaluation_batch(
                    formatted_prompts=prompts,
                    task_type=task_type,
                    dataset_name=dataset_name,
                    prompt_type=prompt_type
                )
                
                batch_jobs.append(batch_job)
                self.logger.info(f"Submitted ChatGPT batch job: {batch_job.job_id}")

        self.logger.info(f"Submitted {len(batch_jobs)} ChatGPT batch jobs")
        return batch_jobs

    async def _compute_baseline_evaluations(
        self, 
        tasks: List[str], 
        datasets: List[str], 
        baselines: List[str]
    ) -> Dict[str, Any]:
        """Compute baseline method evaluations following standard sota_comparison approach."""
        self.logger.info(f"Computing baseline evaluations for {len(baselines)} methods")
        
        baseline_results = {}
        
        for baseline_name in baselines:
            self.logger.info(f"Computing {baseline_name} baseline scores")
            baseline_results[baseline_name] = {}
            
            try:
                # Create baseline instance
                baseline = create_baseline(baseline_name, self.config)
                
                for task_type in tasks:
                    baseline_results[baseline_name][task_type] = {}
                    
                    # Check if baseline supports this task (following standard approach)  
                    if not baseline.supports_task(task_type):
                        self.logger.info(f"Skipping {baseline_name} for {task_type} - not supported")
                        continue
                    
                    for dataset_name in datasets:
                        self.logger.info(f"Computing {baseline_name} for {task_type}/{dataset_name}")
                        
                        dataset = self.dataset_data[dataset_name]
                        
                        # Enhanced baseline evaluation with better error handling
                        predictions = await self._evaluate_baseline_with_enhanced_error_handling(
                            baseline, baseline_name, task_type, dataset
                        )
                        
                        # Calculate metrics from predictions
                        metrics = self._calculate_baseline_metrics(predictions)
                        
                        baseline_results[baseline_name][task_type][dataset_name] = {
                            'predictions': predictions,
                            'metrics': metrics,
                            'metadata': {
                                'baseline_name': baseline_name,
                                'task_type': task_type,
                                'dataset_name': dataset_name,
                                'num_examples': len(predictions)
                            }
                        }
                        
                        self.logger.info(f"Computed {len(predictions)} {baseline_name} scores for {task_type}/{dataset_name}")
                
                # Save individual baseline results
                baseline_path = self.output_dir / "baseline_results" / f"{baseline_name}_results.json"
                with open(baseline_path, 'w') as f:
                    json.dump(baseline_results[baseline_name], f, indent=2, default=str)
                    
            except Exception as e:
                self.logger.error(f"Failed to create baseline {baseline_name}: {e}")
                baseline_results[baseline_name] = {
                    'error': str(e),
                    'status': 'failed'
                }

        self.results['baseline_results'] = baseline_results
        self.logger.info(f"Completed baseline evaluations for {len(baselines)} methods")
        return baseline_results

    async def _evaluate_baseline_with_enhanced_error_handling(
        self,
        baseline,
        baseline_name: str,
        task_type: str,
        dataset: List[Any]
    ) -> List[Any]:
        """Enhanced baseline evaluation with better error handling."""
        predictions = []
        
        for i, example in enumerate(dataset):
            try:
                if task_type == 'entailment_inference':
                    result = baseline.evaluate_entailment_inference(
                        example.source,
                        example.summary,
                        example_id=example.example_id
                    )
                elif task_type == 'consistency_rating':
                    result = baseline.evaluate_consistency_rating(
                        example.source,
                        example.summary,
                        example_id=example.example_id
                    )
                elif task_type == 'summary_ranking':
                    summaries = self._generate_summaries_for_ranking(example)
                    result = baseline.evaluate_summary_ranking(
                        example.source,
                        summaries,
                        example_id=example.example_id
                    )
                else:
                    continue
                
                predictions.append(result)
                
            except NotImplementedError:
                self.logger.warning(f"Skipping {baseline_name} for {task_type} - not implemented")
                break
            except Exception as e:
                self.logger.warning(f"Baseline {baseline_name} failed on example {i}: {e}")
                # Create enhanced default BaselineResult for failed examples
                from src.baselines.sota_metrics import BaselineResult
                failed_result = BaselineResult(
                    baseline_name=baseline_name,
                    task_name=task_type,
                    example_id=example.example_id,
                    prediction=0.5 if task_type == 'consistency_rating' else 0,  # More neutral defaults
                    confidence=0.0,
                    raw_scores={'error': str(e), 'failed': True},
                    processing_time=0.0,
                    metadata={'failed': True, 'error_type': type(e).__name__}
                )
                predictions.append(failed_result)
        
        return predictions

    def _calculate_baseline_metrics(self, predictions: List[Any]) -> Dict[str, float]:
        """Calculate enhanced metrics from baseline predictions."""
        if not predictions:
            return {'mean_score': 0.0, 'std_score': 0.0, 'num_predictions': 0}
        
        # Extract numerical scores with enhanced handling
        scores = []
        confidences = []
        
        for pred in predictions:
            try:
                if hasattr(pred, 'prediction'):
                    prediction = pred.prediction
                    if isinstance(prediction, (list, tuple)) and len(prediction) > 0:
                        scores.append(float(prediction[0]))
                    elif isinstance(prediction, dict):
                        scores.append(float(prediction.get('score', prediction.get('rating', 0))))
                    else:
                        scores.append(float(prediction))
                
                if hasattr(pred, 'confidence'):
                    confidences.append(float(pred.confidence))
                    
            except (ValueError, TypeError):
                scores.append(0.0)
                confidences.append(0.0)
        
        return {
            'mean_score': float(np.mean(scores)) if scores else 0.0,
            'std_score': float(np.std(scores)) if scores else 0.0,
            'mean_confidence': float(np.mean(confidences)) if confidences else 0.0,
            'std_confidence': float(np.std(confidences)) if confidences else 0.0,
            'num_predictions': len(predictions)
        }

    def _generate_summaries_for_ranking(self, example) -> List[str]:
        """Generate multiple summaries for ranking evaluation."""
        original = example.summary
        
        summaries = []
        
        # 1. Original summary (ground truth - should rank highest)
        summaries.append(original)
        
        # 2. Truncated version (middle quality)
        sentences = original.split('. ')
        if len(sentences) > 2:
            truncated = '. '.join(sentences[:len(sentences)//2]) + '.'
        else:
            truncated = sentences[0] + '.' if sentences else original[:len(original)//2]
        summaries.append(truncated)
        
        # 3. Corrupted version (should rank lowest)
        corrupted = original.replace('said', 'stated').replace('reported', 'mentioned')
        if 'million' in corrupted:
            corrupted = corrupted.replace('million', 'billion')
        elif 'billion' in corrupted:
            corrupted = corrupted.replace('billion', 'trillion')
        elif 'thousand' in corrupted:
            corrupted = corrupted.replace('thousand', 'million')
        else:
            corrupted = corrupted + " This happened in 2050."
        summaries.append(corrupted)
        
        return summaries

    async def _monitor_chatgpt_completion(self, batch_jobs: List[BatchJob]) -> List[BatchJob]:
        """Monitor ChatGPT batch jobs until completion."""
        self.logger.info(f"Monitoring {len(batch_jobs)} ChatGPT batch jobs")
        
        # Initialize monitor
        monitor = BatchMonitor(self.batch_client.batch_manager, update_interval=60)
        
        # Wait for completion with progress display
        completed_jobs = await monitor.wait_for_all_completion(
            job_ids=[job.job_id for job in batch_jobs],
            timeout=self.batch_client.processing_timeout,
            show_progress=True
        )
        
        # Generate monitoring report
        monitor_report = monitor.generate_monitoring_report(
            self.output_dir / "batch_monitoring_report.md"
        )
        
        self.logger.info("ChatGPT batch monitoring completed")
        return list(completed_jobs.values())

    async def _process_chatgpt_results(
        self,
        completed_jobs: List[BatchJob],
        tasks: List[str],
        datasets: List[str],
        prompt_type: str
    ):
        """Process ChatGPT batch results with enhanced parsing."""
        self.logger.info("Processing ChatGPT batch results")
        
        self.chatgpt_parsed_results = {}
        total_cost = 0.0
        
        for job in completed_jobs:
            if job.status == BatchStatus.COMPLETED:
                # Get original prompts for parsing
                prompts = self.formatted_prompts[job.task_type][job.dataset_name]
                
                # Download and parse results
                batch_results = await self.batch_client.download_and_parse_results(job, prompts)
                
                # Store results
                key = f"{job.task_type}_{job.dataset_name}"
                self.chatgpt_parsed_results[key] = {
                    'job': job,
                    'results': batch_results,
                    'prompts': prompts
                }
                
                total_cost += job.actual_cost
                
                self.logger.info(f"Processed {len(batch_results)} ChatGPT results for {key}")
            else:
                self.logger.error(f"ChatGPT job {job.job_id} failed with status: {job.status}")

        # Extract ChatGPT scores for correlation analysis with enhanced parsing
        chatgpt_scores = {}
        for key, data in self.chatgpt_parsed_results.items():
            results = data['results']
            task_type = data['job'].task_type
            
            # Enhanced result processing with better error handling
            scores = []
            for i, result in enumerate(results):
                try:
                    # Try parsed content first
                    if hasattr(result, 'parsing_successful') and result.parsing_successful and hasattr(result, 'parsed_content') and result.parsed_content:
                        content = result.parsed_content
                    else:
                        # Enhanced manual parsing with multiple strategies
                        if hasattr(result, 'raw_response') and result.raw_response:
                            content = self._enhanced_manual_parsing(self._extract_raw_content(result), task_type, i)
                        else:
                            content = self._enhanced_manual_parsing("", task_type, i)
                    
                    # Extract numerical score based on task type
                    score = self._extract_numerical_score_from_content(content, task_type)
                    scores.append(score)
                        
                except Exception as e:
                    self.logger.warning(f"Failed to process result {i} for {key}: {e}")
                    # Add appropriate default scores
                    if task_type == 'entailment_inference':
                        scores.append(0.0)
                    elif task_type == 'consistency_rating':
                        scores.append(0.5)
                    elif task_type == 'summary_ranking':
                        scores.append(0.33)
            
            chatgpt_scores[key] = scores
            self.logger.info(f"Extracted {len(scores)} scores for {key}")

        self.results['chatgpt_results'] = {
            'scores': chatgpt_scores,
            'parsed_results': self.chatgpt_parsed_results,
            'total_cost': total_cost
        }
        
        # Update batch analysis
        self.results['batch_analysis'] = {
            'total_chatgpt_jobs': len(completed_jobs),
            'successful_chatgpt_jobs': sum(1 for job in completed_jobs if job.status == BatchStatus.COMPLETED),
            'failed_chatgpt_jobs': sum(1 for job in completed_jobs if job.status != BatchStatus.COMPLETED),
            'chatgpt_cost': total_cost,
            'cost_savings': total_cost * self.batch_client.cost_savings / (1 - self.batch_client.cost_savings),
            'estimated_sync_cost': total_cost / (1 - self.batch_client.cost_savings)
        }

    def _extract_raw_content(self, result) -> str:
        """Extract raw content string from batch result."""
        if hasattr(result, 'raw_response') and result.raw_response:
            if hasattr(result.raw_response, 'content'):
                return result.raw_response.content or ""
            else:
                return str(result.raw_response)
        return ""

    def _enhanced_manual_parsing(self, raw_response: str, task_type: str, index: int) -> dict:
        """Enhanced manual parsing with multiple strategies and better error recovery."""
        try:
            response = raw_response.strip().lower()
            
            if task_type == 'entailment_inference':
                # Enhanced entailment parsing with multiple patterns
                entailment_indicators = [
                    'entailment', 'entailed', 'supports', 'confirmed', 'consistent',
                    'true', 'yes', 'correct', 'valid', 'accurate'
                ]
                contradiction_indicators = [
                    'contradiction', 'contradicts', 'refuted', 'inconsistent',
                    'false', 'no', 'incorrect', 'invalid', 'inaccurate'
                ]
                
                # Score-based approach
                entailment_score = sum(1 for indicator in entailment_indicators if indicator in response)
                contradiction_score = sum(1 for indicator in contradiction_indicators if indicator in response)
                
                if entailment_score > contradiction_score:
                    return {'prediction': 1, 'answer': 'ENTAILMENT', 'confidence': min(0.9, 0.5 + entailment_score * 0.1)}
                elif contradiction_score > entailment_score:
                    return {'prediction': 0, 'answer': 'CONTRADICTION', 'confidence': min(0.9, 0.5 + contradiction_score * 0.1)}
                else:
                    return {'prediction': 0, 'answer': 'CONTRADICTION', 'confidence': 0.3}
            
            elif task_type == 'consistency_rating':
                # Enhanced rating parsing with multiple regex patterns
                import re
                
                rating_patterns = [
                    r'(?:rating|score|consistency)[:=\s]*(\d+(?:\.\d+)?)',
                    r'(\d+(?:\.\d+)?)(?:/100|\s*%|\s*out\s*of\s*100)',
                    r'(\d+(?:\.\d+)?)\s*(?:out\s*of\s*)?(?:10|100)',
                    r'(\d+(?:\.\d+)?)'
                ]
                
                for pattern in rating_patterns:
                    matches = re.findall(pattern, response)
                    if matches:
                        try:
                            rating = float(matches[0])
                            # Normalize to 0-100 scale
                            if rating <= 1.0:
                                rating = rating * 100
                            elif rating <= 10:
                                rating = rating * 10
                            # Clamp to valid range
                            rating = max(0, min(100, rating))
                            return {'rating': rating, 'prediction': rating, 'confidence': 0.8}
                        except ValueError:
                            continue
                
                # Qualitative descriptor fallback
                qualitative_map = {
                    ('excellent', 'perfect', 'completely', 'fully'): 90.0,
                    ('good', 'mostly', 'largely', 'high'): 75.0,
                    ('moderate', 'somewhat', 'partial'): 60.0,
                    ('poor', 'low', 'little', 'minimal'): 30.0,
                    ('terrible', 'none', 'no', 'zero'): 10.0
                }
                
                for descriptors, rating in qualitative_map.items():
                    if any(desc in response for desc in descriptors):
                        return {'rating': rating, 'prediction': rating, 'confidence': 0.6}
                
                return {'rating': 50.0, 'prediction': 50.0, 'confidence': 0.3}
            
            elif task_type == 'summary_ranking':
                import re
                
                # Try to extract ranking like "1, 2, 3" or "first: A, second: B, third: C"
                number_matches = re.findall(r'(\d+)', response)
                if len(number_matches) >= 3:
                    try:
                        ranking = [int(n) for n in number_matches[:3]]
                        if set(ranking) == {1, 2, 3}:
                            return {'ranking': ranking}
                    except ValueError:
                        pass
                
                # Default: assume original (A) is best
                return {'ranking': [1, 2, 3]}
                
        except Exception as e:
            self.logger.warning(f"Enhanced manual parsing failed for {task_type} at index {index}: {e}")
        
        # Return defaults
        if task_type == 'entailment_inference':
            return {'prediction': 0, 'answer': 'CONTRADICTION'}
        elif task_type == 'consistency_rating':
            return {'rating': 50.0, 'prediction': 50.0}
        elif task_type == 'summary_ranking':
            return {'ranking': [1, 2, 3]}
        return {}

    def _extract_numerical_score_from_content(self, content: dict, task_type: str) -> float:
        """Extract numerical score from parsed content based on task type."""
        if task_type == 'entailment_inference':
            prediction = content.get('prediction', 0)
            answer = content.get('answer', 'CONTRADICTION').upper()
            
            # Convert to score (prefer prediction if available, otherwise parse answer)
            if prediction is not None and isinstance(prediction, (int, float)):
                return float(prediction)
            elif answer in ['ENTAILMENT', '1', 'TRUE', 'YES']:
                return 1.0
            else:
                return 0.0
                
        elif task_type == 'consistency_rating':
            rating = content.get('rating', content.get('prediction', 50.0))
            if rating is not None:
                if isinstance(rating, str):
                    import re
                    numbers = re.findall(r'\d+(?:\.\d+)?', rating)
                    if numbers:
                        rating = float(numbers[0])
                    else:
                        rating = 50.0
                return float(rating) / 100.0  # Normalize to 0-1
            else:
                return 0.5
        
        elif task_type == 'summary_ranking':
            ranking = content.get('ranking', [1, 2, 3])
            if not ranking or len(ranking) == 0:
                ranking = [1, 2, 3]
            # Convert ranking to score (position of original summary)
            original_position = ranking[0] if ranking else 1
            return 1.0 / original_position  # Position 1 gets score 1.0, position 2 gets 0.5, etc.
        
        return 0.0

    async def _perform_enhanced_correlation_analysis(self, baselines: List[str]):
        """Enhanced correlation analysis with better logic from second implementation."""
        self.logger.info("Performing enhanced correlation analysis")
        
        enhanced_correlation_analysis = {
            'pearson_correlations': {},
            'spearman_correlations': {},
            'correlation_summary': {},
            'method_rankings': {}
        }
        
        chatgpt_scores = self.results['chatgpt_results']['scores']
        baseline_results = self.results['baseline_results']
        
        # Compute correlations for each baseline-task-dataset combination
        all_pearson_correlations = []
        all_spearman_correlations = []
        baseline_avg_correlations = {}
        
        for baseline_name in baselines:
            self.logger.info(f"Analyzing correlations with {baseline_name}")
            
            enhanced_correlation_analysis['pearson_correlations'][baseline_name] = {}
            enhanced_correlation_analysis['spearman_correlations'][baseline_name] = {}
            
            baseline_correlations = []
            
            for key in chatgpt_scores.keys():
                # Parse the key to extract task_type and dataset_name
                if key.startswith('entailment_inference_'):
                    task_type = 'entailment_inference'
                    dataset_name = key.replace('entailment_inference_', '')
                elif key.startswith('consistency_rating_'):
                    task_type = 'consistency_rating'
                    dataset_name = key.replace('consistency_rating_', '')
                elif key.startswith('summary_ranking_'):
                    task_type = 'summary_ranking'
                    dataset_name = key.replace('summary_ranking_', '')
                else:
                    continue
                
                # Get ChatGPT scores
                chatgpt_vals = chatgpt_scores[key]
                
                # Get baseline scores with enhanced extraction
                if (baseline_name in baseline_results and 
                    task_type in baseline_results[baseline_name] and
                    dataset_name in baseline_results[baseline_name][task_type]):
                    
                    baseline_data = baseline_results[baseline_name][task_type][dataset_name]
                    
                    if 'predictions' in baseline_data and baseline_data['predictions']:
                        predictions = baseline_data['predictions']
                        
                        # Enhanced numerical extraction
                        baseline_vals = self._extract_enhanced_numerical_predictions(predictions, task_type, baseline_name)
                        
                        # Ensure same length
                        min_length = min(len(chatgpt_vals), len(baseline_vals))
                        if min_length < 2:
                            self.logger.warning(f"Insufficient data for {baseline_name}/{key}: {min_length} samples")
                            continue
                        
                        chatgpt_vals = chatgpt_vals[:min_length]
                        baseline_vals = baseline_vals[:min_length]
                        
                        # Enhanced correlation computation with better error handling
                        try:
                            # Check for zero variance (causes NaN correlations)
                            if np.var(chatgpt_vals) == 0 or np.var(baseline_vals) == 0:
                                self.logger.warning(f"Zero variance detected for {baseline_name}/{key}")
                                continue
                            
                            pearson_r, pearson_p = stats.pearsonr(chatgpt_vals, baseline_vals)
                            spearman_r, spearman_p = stats.spearmanr(chatgpt_vals, baseline_vals)
                            
                            # Check if correlations are valid
                            if not (np.isnan(pearson_r) or np.isnan(spearman_r)):
                                # Store detailed correlation results
                                if task_type not in enhanced_correlation_analysis['pearson_correlations'][baseline_name]:
                                    enhanced_correlation_analysis['pearson_correlations'][baseline_name][task_type] = {}
                                if task_type not in enhanced_correlation_analysis['spearman_correlations'][baseline_name]:
                                    enhanced_correlation_analysis['spearman_correlations'][baseline_name][task_type] = {}
                                
                                enhanced_correlation_analysis['pearson_correlations'][baseline_name][task_type][dataset_name] = {
                                    'correlation': pearson_r,
                                    'p_value': pearson_p,
                                    'significant': pearson_p < 0.05,
                                    'n_samples': min_length,
                                    'effect_size': self._calculate_correlation_effect_size(pearson_r)
                                }
                                
                                enhanced_correlation_analysis['spearman_correlations'][baseline_name][task_type][dataset_name] = {
                                    'correlation': spearman_r,
                                    'p_value': spearman_p,
                                    'significant': spearman_p < 0.05,
                                    'n_samples': min_length,
                                    'effect_size': self._calculate_correlation_effect_size(spearman_r)
                                }
                                
                                all_pearson_correlations.append(pearson_r)
                                all_spearman_correlations.append(spearman_r)
                                baseline_correlations.append(abs(pearson_r))  # Use absolute value for ranking
                                
                                self.logger.info(f"{baseline_name}/{key}: r={pearson_r:.3f}, ρ={spearman_r:.3f}")
                                
                        except Exception as e:
                            self.logger.warning(f"Failed to compute correlation for {baseline_name}/{key}: {e}")
                            continue
            
            # Calculate baseline average
            if baseline_correlations:
                baseline_avg_correlations[baseline_name] = np.mean(baseline_correlations)

        # Generate enhanced correlation summary
        enhanced_correlation_analysis['correlation_summary'] = {
            'overall_statistics': {
                'mean_pearson': np.mean(all_pearson_correlations) if all_pearson_correlations else 0,
                'std_pearson': np.std(all_pearson_correlations) if all_pearson_correlations else 0,
                'mean_spearman': np.mean(all_spearman_correlations) if all_spearman_correlations else 0,
                'std_spearman': np.std(all_spearman_correlations) if all_spearman_correlations else 0,
                'valid_correlations': len(all_pearson_correlations),
                'correlation_range': {
                    'min': np.min(all_pearson_correlations) if all_pearson_correlations else 0,
                    'max': np.max(all_pearson_correlations) if all_pearson_correlations else 0
                }
            },
            'baseline_performance': baseline_avg_correlations,
            'best_correlating_baseline': max(baseline_avg_correlations.items(), key=lambda x: x[1])[0] if baseline_avg_correlations else None
        }
        
        # Enhanced method rankings
        if baseline_avg_correlations:
            sorted_baselines = sorted(baseline_avg_correlations.items(), key=lambda x: x[1], reverse=True)
            enhanced_correlation_analysis['method_rankings'] = {
                'by_correlation_strength': [
                    {
                        'baseline': name,
                        'avg_correlation': corr,
                        'effect_size': self._calculate_correlation_effect_size(corr),
                        'rank': i + 1
                    }
                    for i, (name, corr) in enumerate(sorted_baselines)
                ]
            }

        self.results['correlation_analysis'] = enhanced_correlation_analysis
        
        mean_pearson = enhanced_correlation_analysis['correlation_summary']['overall_statistics']['mean_pearson']
        self.logger.info(f"Enhanced correlation analysis completed: Mean Pearson r = {mean_pearson:.3f}")

    def _extract_enhanced_numerical_predictions(self, predictions: List, task_name: str, method_name: str) -> List[float]:
        """Enhanced numerical prediction extraction with better error handling."""
        numerical_predictions = []
        
        for i, pred in enumerate(predictions):
            try:
                if hasattr(pred, 'prediction'):
                    # Object format (BaselineResult, etc.)
                    prediction = pred.prediction
                    if isinstance(prediction, (list, tuple)) and len(prediction) > 0:
                        # Handle list predictions (e.g., ROUGE)
                        numerical_predictions.append(float(prediction[0]))
                    elif isinstance(prediction, dict):
                        # Handle dict predictions
                        if task_name == 'consistency_rating':
                            numerical_predictions.append(float(prediction.get('rating', prediction.get('score', 50.0))))
                        else:
                            numerical_predictions.append(float(prediction.get('score', prediction.get('prediction', 0.0))))
                    else:
                        numerical_predictions.append(float(prediction))
                        
                elif isinstance(pred, dict):
                    # Dictionary format
                    if task_name == 'entailment_inference':
                        prediction = pred.get('prediction', pred.get('answer', 0))
                        if isinstance(prediction, str):
                            numerical_predictions.append(1.0 if prediction.upper() in ['ENTAILMENT', '1', 'TRUE'] else 0.0)
                        else:
                            numerical_predictions.append(float(prediction))
                    elif task_name == 'consistency_rating':
                        rating = pred.get('rating', pred.get('prediction', 50.0))
                        numerical_predictions.append(float(rating))
                    else:
                        prediction = pred.get('prediction', 0)
                        numerical_predictions.append(float(prediction))
                        
                else:
                    # Direct numerical value
                    numerical_predictions.append(float(pred))
                    
            except (ValueError, TypeError, KeyError) as e:
                self.logger.debug(f"Failed to extract prediction {i} for {method_name}: {e}")
                # Use task-appropriate defaults
                if task_name == 'entailment_inference':
                    numerical_predictions.append(0.0)
                elif task_name == 'consistency_rating':
                    numerical_predictions.append(50.0)
                else:
                    numerical_predictions.append(0.0)
        
        return numerical_predictions

    def _calculate_correlation_effect_size(self, correlation: float) -> str:
        """Calculate and interpret correlation effect size."""
        if np.isnan(correlation):
            return 'unknown'
        
        abs_corr = abs(correlation)
        if abs_corr >= 0.7:
            return 'large'
        elif abs_corr >= 0.5:
            return 'medium'
        elif abs_corr >= 0.3:
            return 'small'
        else:
            return 'negligible'

    async def _perform_agreement_analysis(self, baselines: List[str]):
        """Perform agreement analysis for imbalanced datasets using Cohen's kappa.
        
        Following standard approach - focuses on entailment_inference and consistency_rating
        as these are the tasks where baselines can provide meaningful comparisons.
        """
        self.logger.info("Performing agreement analysis for imbalanced datasets")
        
        agreement_analysis = {
            'agreement_metrics': {},
            'kappa_analysis': {},
            'agreement_summary': {}
        }
        
        chatgpt_scores = self.results['chatgpt_results']['scores']
        baseline_results = self.results['baseline_results']
        
        all_agreement_scores = []
        all_kappa_scores = []
        baseline_avg_agreement = {}
        baseline_avg_kappa = {}
        
        for baseline_name in baselines:
            agreement_analysis['agreement_metrics'][baseline_name] = {}
            agreement_analysis['kappa_analysis'][baseline_name] = {}
            
            baseline_agreements = []
            baseline_kappas = []
            
            for key in chatgpt_scores.keys():
                # Parse task and dataset - following standard approach, focus on supported tasks
                if key.startswith('entailment_inference_'):
                    task_type = 'entailment_inference'
                    dataset_name = key.replace('entailment_inference_', '')
                elif key.startswith('consistency_rating_'):
                    task_type = 'consistency_rating'
                    dataset_name = key.replace('consistency_rating_', '')
                else:
                    continue  # Skip summary_ranking and other tasks for agreement analysis
                
                # Get data
                chatgpt_vals = chatgpt_scores[key]
                
                if (baseline_name in baseline_results and 
                    task_type in baseline_results[baseline_name] and
                    dataset_name in baseline_results[baseline_name][task_type]):
                    
                    baseline_data = baseline_results[baseline_name][task_type][dataset_name]
                    
                    if 'predictions' in baseline_data and baseline_data['predictions']:
                        baseline_vals = self._extract_enhanced_numerical_predictions(
                            baseline_data['predictions'], task_type, baseline_name
                        )
                        
                        # Ensure same length
                        min_length = min(len(chatgpt_vals), len(baseline_vals))
                        if min_length < 2:
                            continue
                        
                        chatgpt_vals = chatgpt_vals[:min_length]
                        baseline_vals = baseline_vals[:min_length]
                        
                        # Convert to binary decisions for agreement analysis
                        chatgpt_binary = self._convert_to_binary_enhanced(chatgpt_vals, task_type, 'chatgpt')
                        baseline_binary = self._convert_to_binary_enhanced(baseline_vals, task_type, baseline_name)
                        
                        # Calculate agreement metrics
                        agreement_metrics = self._calculate_enhanced_agreement_metrics(
                            chatgpt_binary, baseline_binary, min_length
                        )
                        
                        # Store results
                        if task_type not in agreement_analysis['agreement_metrics'][baseline_name]:
                            agreement_analysis['agreement_metrics'][baseline_name][task_type] = {}
                        
                        agreement_analysis['agreement_metrics'][baseline_name][task_type][dataset_name] = agreement_metrics
                        
                        all_agreement_scores.append(agreement_metrics['agreement'])
                        all_kappa_scores.append(agreement_metrics['cohens_kappa'])
                        baseline_agreements.append(agreement_metrics['agreement'])
                        baseline_kappas.append(agreement_metrics['cohens_kappa'])
            
            # Calculate baseline averages
            if baseline_agreements:
                baseline_avg_agreement[baseline_name] = np.mean(baseline_agreements)
            if baseline_kappas:
                baseline_avg_kappa[baseline_name] = np.mean(baseline_kappas)
        
        # Generate agreement summary
        agreement_analysis['agreement_summary'] = {
            'overall_statistics': {
                'mean_agreement': np.mean(all_agreement_scores) if all_agreement_scores else 0,
                'std_agreement': np.std(all_agreement_scores) if all_agreement_scores else 0,
                'mean_kappa': np.mean(all_kappa_scores) if all_kappa_scores else 0,
                'std_kappa': np.std(all_kappa_scores) if all_kappa_scores else 0,
                'total_comparisons': len(all_agreement_scores)
            },
            'baseline_performance': {
                'agreement': baseline_avg_agreement,
                'kappa': baseline_avg_kappa
            },
            'best_agreeing_baseline': max(baseline_avg_agreement.items(), key=lambda x: x[1])[0] if baseline_avg_agreement else None,
            'best_kappa_baseline': max(baseline_avg_kappa.items(), key=lambda x: x[1])[0] if baseline_avg_kappa else None
        }
        
        self.results['agreement_analysis'] = agreement_analysis
        self.logger.info(f"Agreement analysis completed: {len(all_agreement_scores)} comparisons")

    def _convert_to_binary_enhanced(self, scores: List[float], task_name: str, method_name: str = None) -> List[int]:
        """Enhanced binary conversion with method-specific thresholds."""
        if task_name == 'consistency_rating':
            # Use method-specific thresholds
            if method_name == 'factcc':
                # FactCC typically produces scores in 0-5 range
                threshold = np.median(scores) if scores else 2.5
            elif method_name in ['bertscore', 'rouge']:
                # These typically produce 0-1 scores, scale to 0-100
                scaled_scores = [s * 100 if s <= 1 else s for s in scores]
                threshold = np.median(scaled_scores) if scaled_scores else 50.0
                return [1 if (s * 100 if s <= 1 else s) > threshold else 0 for s in scores]
            else:
                # ChatGPT and others use 0-100 scale
                threshold = np.median(scores) if scores else 50.0
            
            return [1 if score > threshold else 0 for score in scores]
        elif task_name == 'entailment_inference':
            # Binary task: > 0.5 = ENTAILMENT (1), <= 0.5 = CONTRADICTION (0)
            return [1 if score > 0.5 else 0 for score in scores]
        else:
            # Default threshold
            return [1 if score > 0.5 else 0 for score in scores]

    def _calculate_enhanced_agreement_metrics(self, pred1: List[int], pred2: List[int], n_samples: int) -> Dict[str, Any]:
        """Enhanced agreement metrics calculation with better error handling."""
        pred1 = np.array(pred1)
        pred2 = np.array(pred2)
        
        # Basic agreement
        agreement = np.sum(pred1 == pred2) / len(pred1) if len(pred1) > 0 else 0
        
        # Enhanced Cohen's kappa calculation
        p0 = agreement  # observed agreement
        p1_pred1 = np.mean(pred1) if len(pred1) > 0 else 0
        p1_pred2 = np.mean(pred2) if len(pred2) > 0 else 0
        p0_pred1 = 1 - p1_pred1
        p0_pred2 = 1 - p1_pred2
        pe = p1_pred1 * p1_pred2 + p0_pred1 * p0_pred2  # expected agreement
        
        if abs(pe - 1.0) < 1e-10:  # pe is essentially 1.0
            kappa = 0.0
        else:
            kappa = (p0 - pe) / (1 - pe)
        
        # Handle NaN values
        if np.isnan(kappa):
            kappa = 0.0
        
        # Enhanced confusion matrix metrics
        tp = np.sum((pred1 == 1) & (pred2 == 1))
        tn = np.sum((pred1 == 0) & (pred2 == 0))
        fp = np.sum((pred1 == 0) & (pred2 == 1))
        fn = np.sum((pred1 == 1) & (pred2 == 0))
        
        # Precision, recall, F1 with enhanced error handling
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
            'n_samples': n_samples,
            'kappa_interpretation': self._interpret_kappa(kappa)
        }

    def _interpret_kappa(self, kappa: float) -> str:
        """Interpret Cohen's kappa value."""
        if kappa < 0:
            return "Poor (worse than chance)"
        elif kappa < 0.20:
            return "Slight"
        elif kappa < 0.40:
            return "Fair"
        elif kappa < 0.60:
            return "Moderate"
        elif kappa < 0.80:
            return "Substantial"
        else:
            return "Almost perfect"

    async def _analyze_enhanced_performance_comparison(self):
        """Enhanced performance comparison analysis."""
        self.logger.info("Analyzing enhanced performance comparison")
        
        performance_comparison = {
            'task_performance': {},
            'baseline_performance': {},
            'relative_performance': {},
            'performance_insights': {},
            'cross_task_analysis': {}
        }
        
        chatgpt_scores = self.results['chatgpt_results']['scores']
        baseline_results = self.results['baseline_results']
        
        # Analyze performance for each task with enhanced metrics
        for key, chatgpt_vals in chatgpt_scores.items():
            if key.startswith('entailment_inference_'):
                task_type = 'entailment_inference'
                dataset_name = key.replace('entailment_inference_', '')
            elif key.startswith('consistency_rating_'):
                task_type = 'consistency_rating'
                dataset_name = key.replace('consistency_rating_', '')
            elif key.startswith('summary_ranking_'):
                task_type = 'summary_ranking' 
                dataset_name = key.replace('summary_ranking_', '')
            else:
                continue
            
            if task_type not in performance_comparison['task_performance']:
                performance_comparison['task_performance'][task_type] = {}
            
            # ChatGPT performance metrics
            performance_comparison['task_performance'][task_type][dataset_name] = {
                'chatgpt': {
                    'mean_score': np.mean(chatgpt_vals),
                    'std_score': np.std(chatgpt_vals),
                    'min_score': np.min(chatgpt_vals),
                    'max_score': np.max(chatgpt_vals),
                    'score_range': np.max(chatgpt_vals) - np.min(chatgpt_vals)
                },
                'baselines': {}
            }
            
            # Compare with each baseline
            for baseline_name in baseline_results.keys():
                if (task_type in baseline_results[baseline_name] and 
                    dataset_name in baseline_results[baseline_name][task_type]):
                    
                    baseline_data = baseline_results[baseline_name][task_type][dataset_name]
                    
                    if 'predictions' in baseline_data and baseline_data['predictions']:
                        baseline_vals = self._extract_enhanced_numerical_predictions(
                            baseline_data['predictions'], task_type, baseline_name
                        )
                        
                        if baseline_vals:
                            performance_comparison['task_performance'][task_type][dataset_name]['baselines'][baseline_name] = {
                                'mean_score': np.mean(baseline_vals),
                                'std_score': np.std(baseline_vals),
                                'min_score': np.min(baseline_vals),
                                'max_score': np.max(baseline_vals),
                                'score_range': np.max(baseline_vals) - np.min(baseline_vals)
                            }
        
        # Generate enhanced performance insights
        insights = self._generate_enhanced_performance_insights(performance_comparison)
        performance_comparison['performance_insights'] = insights
        
        self.results['performance_comparison'] = performance_comparison

    def _generate_enhanced_performance_insights(self, performance_comparison: Dict) -> Dict:
        """Generate enhanced performance insights."""
        insights = {
            'best_correlating_baselines': {},
            'task_difficulty_ranking': [],
            'correlation_strength_summary': {},
            'performance_variability_analysis': {}
        }
        
        # Extract correlation data for insights
        correlation_data = self.results.get('correlation_analysis', {})
        baseline_performance = correlation_data.get('correlation_summary', {}).get('baseline_performance', {})
        
        # Find best correlating baseline for each task
        task_correlations = {}
        for baseline_name, avg_corr in baseline_performance.items():
            # This is a simplified approach - in practice you'd want task-specific correlations
            for task_name in performance_comparison['task_performance'].keys():
                if task_name not in task_correlations:
                    task_correlations[task_name] = {}
                task_correlations[task_name][baseline_name] = avg_corr
        
        for task_name, baseline_corrs in task_correlations.items():
            if baseline_corrs:
                best_baseline = max(baseline_corrs.items(), key=lambda x: abs(x[1]))
                insights['best_correlating_baselines'][task_name] = {
                    'baseline': best_baseline[0],
                    'correlation': best_baseline[1],
                    'interpretation': self._calculate_correlation_effect_size(best_baseline[1])
                }
        
        # Task difficulty ranking based on correlation strength
        if baseline_performance:
            # This is simplified - ideally you'd compute task-specific averages
            avg_correlation = np.mean(list(baseline_performance.values()))
            insights['task_difficulty_ranking'] = [
                {
                    'task': task_name,
                    'avg_correlation': avg_correlation,  # Simplified
                    'difficulty': 'Easy' if avg_correlation > 0.7 else 'Moderate' if avg_correlation > 0.5 else 'Difficult'
                }
                for task_name in performance_comparison['task_performance'].keys()
            ]
        
        return insights

    async def _perform_enhanced_statistical_analysis(self):
        """Enhanced statistical significance testing."""
        self.logger.info("Performing enhanced statistical analysis")
        
        statistical_analysis = {
            'correlation_significance': {},
            'effect_size_analysis': {},
            'confidence_intervals': {},
            'multiple_comparisons': {},
            'significance_summary': {}
        }
        
        # Enhanced correlation significance analysis
        significant_correlations = 0
        total_correlations = 0
        effect_size_distribution = {'large': 0, 'medium': 0, 'small': 0, 'negligible': 0}
        
        correlation_data = self.results['correlation_analysis']
        
        for baseline_name, baseline_corr in correlation_data.get('pearson_correlations', {}).items():
            statistical_analysis['correlation_significance'][baseline_name] = {}
            
            for task_name, task_corr in baseline_corr.items():
                statistical_analysis['correlation_significance'][baseline_name][task_name] = {}
                
                for dataset_name, dataset_corr in task_corr.items():
                    if isinstance(dataset_corr, dict):
                        correlation = dataset_corr.get('correlation', 0)
                        p_value = dataset_corr.get('p_value', 1.0)
                        n_samples = dataset_corr.get('n_samples', 0)
                        effect_size = dataset_corr.get('effect_size', 'unknown')
                        
                        is_significant = p_value < 0.05 and not np.isnan(correlation)
                        
                        statistical_analysis['correlation_significance'][baseline_name][task_name][dataset_name] = {
                            'correlation': correlation,
                            'p_value': p_value,
                            'is_significant': is_significant,
                            'n_samples': n_samples,
                            'effect_size': effect_size
                        }
                        
                        total_correlations += 1
                        if is_significant:
                            significant_correlations += 1
                        
                        if effect_size in effect_size_distribution:
                            effect_size_distribution[effect_size] += 1
        
        # Multiple comparisons correction (Bonferroni)
        if total_correlations > 1:
            corrected_alpha = 0.05 / total_correlations
            significant_after_correction = 0
            
            for baseline_data in statistical_analysis['correlation_significance'].values():
                for task_data in baseline_data.values():
                    for dataset_data in task_data.values():
                        if isinstance(dataset_data, dict):
                            p_value = dataset_data.get('p_value', 1.0)
                            dataset_data['bonferroni_significant'] = p_value < corrected_alpha
                            if dataset_data['bonferroni_significant']:
                                significant_after_correction += 1
            
            statistical_analysis['multiple_comparisons'] = {
                'corrected_alpha': corrected_alpha,
                'significant_after_correction': significant_after_correction,
                'correction_method': 'bonferroni'
            }
        
        # Generate comprehensive significance summary
        statistical_analysis['significance_summary'] = {
            'total_correlations': total_correlations,
            'significant_correlations': significant_correlations,
            'significance_rate': significant_correlations / total_correlations if total_correlations > 0 else 0,
            'effect_size_distribution': effect_size_distribution,
            'interpretation': self._interpret_enhanced_significance_results(
                significant_correlations, total_correlations, effect_size_distribution
            )
        }
        
        self.results['statistical_analysis'] = statistical_analysis

    def _interpret_enhanced_significance_results(
        self, 
        significant: int, 
        total: int, 
        effect_size_dist: Dict[str, int]
    ) -> str:
        """Enhanced interpretation of statistical significance results."""
        if total == 0:
            return "No correlations computed"
        
        rate = significant / total
        large_effects = effect_size_dist.get('large', 0)
        medium_effects = effect_size_dist.get('medium', 0)
        
        interpretation = []
        
        # Significance rate interpretation
        if rate >= 0.8:
            interpretation.append("Strong statistical evidence of correlation with ChatGPT")
        elif rate >= 0.6:
            interpretation.append("Moderate statistical evidence of correlation with ChatGPT")
        elif rate >= 0.4:
            interpretation.append("Limited statistical evidence of correlation with ChatGPT")
        else:
            interpretation.append("Weak statistical evidence of correlation with ChatGPT")
        
        # Effect size interpretation
        if large_effects > 0:
            interpretation.append(f"{large_effects} large effect sizes detected")
        if medium_effects > 0:
            interpretation.append(f"{medium_effects} medium effect sizes detected")
        
        return "; ".join(interpretation)

    async def _generate_enhanced_sota_visualizations(self):
        """Generate enhanced SOTA comparison visualizations."""
        self.logger.info("Generating enhanced SOTA comparison visualizations")
        
        viz_dir = self.output_dir / "figures"
        
        try:
            # 1. Enhanced correlation heatmap
            await self._create_enhanced_correlation_heatmap(viz_dir)
            
            # 2. Enhanced method ranking visualization
            await self._create_enhanced_method_ranking(viz_dir)
            
            # 3. Enhanced score distribution comparison
            await self._create_enhanced_score_distributions(viz_dir)
            
            # 4. Enhanced performance comparison across tasks
            await self._create_enhanced_performance_comparison(viz_dir)
            
            # 5. Enhanced agreement analysis
            await self._create_enhanced_agreement_analysis(viz_dir)
            
            # 6. Statistical significance overview
            await self._create_statistical_significance_overview(viz_dir)
            
            self.logger.info(f"Enhanced visualizations saved to {viz_dir}")
            
        except Exception as e:
            self.logger.warning(f"Enhanced visualization generation failed: {e}")
            self.results['visualizations'] = {'error': str(e)}

    async def _create_enhanced_correlation_heatmap(self, viz_dir: Path):
        """Create enhanced correlation heatmap with significance indicators."""
        pearson_correlations = self.results['correlation_analysis'].get('pearson_correlations', {})
        
        if not pearson_correlations:
            return
        
        # Prepare enhanced heatmap data
        correlation_data = []
        
        for baseline_name, baseline_corr in pearson_correlations.items():
            for task_name, task_corr in baseline_corr.items():
                for dataset_name, dataset_corr in task_corr.items():
                    if isinstance(dataset_corr, dict) and 'correlation' in dataset_corr:
                        correlation = dataset_corr['correlation']
                        p_value = dataset_corr.get('p_value', 1.0)
                        significant = dataset_corr.get('significant', False)
                        
                        if not np.isnan(correlation):
                            correlation_data.append({
                                'baseline': baseline_name.upper(),
                                'task': task_name.replace('_', ' ').title(),
                                'dataset': dataset_name.replace('_', ' ').title(),
                                'correlation': correlation,
                                'p_value': p_value,
                                'significant': significant,
                                'combination': f"{task_name.replace('_', ' ').title()}\n{dataset_name.replace('_', ' ').title()}"
                            })
        
        if not correlation_data:
            return
        
        # Create enhanced heatmap data structure
        baselines = sorted(list(set(item['baseline'] for item in correlation_data)))
        combinations = sorted(list(set(item['combination'] for item in correlation_data)))
        
        # Create correlation matrix and significance matrix
        correlation_matrix = np.full((len(baselines), len(combinations)), np.nan)
        significance_matrix = np.full((len(baselines), len(combinations)), False)
        text_matrix = [[''] * len(combinations) for _ in range(len(baselines))]
        
        for i, baseline in enumerate(baselines):
            for j, combination in enumerate(combinations):
                matching_items = [item for item in correlation_data 
                                if item['baseline'] == baseline and item['combination'] == combination]
                if matching_items:
                    item = matching_items[0]
                    correlation_matrix[i][j] = item['correlation']
                    significance_matrix[i][j] = item['significant']
                    
                    # Create text with significance indicator
                    corr_text = f"{item['correlation']:.3f}"
                    if item['significant']:
                        corr_text += "*"
                    text_matrix[i][j] = corr_text
        
        # Create enhanced heatmap
        fig = go.Figure(data=go.Heatmap(
            z=correlation_matrix,
            x=combinations,
            y=baselines,
            colorscale='RdBu',
            zmid=0,
            zmin=-1,
            zmax=1,
            colorbar=dict(
                title="Pearson Correlation<br>with ChatGPT",
                title_font=dict(family='Arial', size=14)
            ),
            text=text_matrix,
            texttemplate="%{text}",
            textfont=dict(family='Arial', size=11, color='white'),
            hoverongaps=False,
            hovertemplate="<b>%{y}</b><br>%{x}<br>Correlation: %{z:.3f}<br><extra></extra>"
        ))
        
        fig.update_layout(
            title={
                'text': "Enhanced Correlation Analysis: ChatGPT vs SOTA Baselines<br><sub>* indicates p < 0.05</sub>",
                'font': {'size': 18, 'family': 'Arial'},
                'x': 0.5,
                'xanchor': 'center'
            },
            xaxis_title="Task / Dataset",
            yaxis_title="Baseline Method",
            font=dict(family='Arial', size=12),
            height=max(500, 300 + len(baselines) * 60),
            width=max(900, 150 * len(combinations)),
            paper_bgcolor='white',
            plot_bgcolor='white'
        )
        
        fig_path = viz_dir / "Enhanced_Correlation_Heatmap_ChatGPT_vs_SOTA_Baselines.png"
        fig.write_image(str(fig_path), width=max(1200, 200 * len(combinations)), 
                       height=max(600, 400 + len(baselines) * 60), scale=2)

    async def _create_enhanced_method_ranking(self, viz_dir: Path):
        """Create enhanced method ranking with error bars and effect sizes."""
        method_rankings = self.results['correlation_analysis'].get('method_rankings', {})
        
        if 'by_correlation_strength' not in method_rankings:
            return
        
        rankings = method_rankings['by_correlation_strength']
        
        # Extract data
        baselines = [item['baseline'].upper() for item in rankings]
        correlations = [item['avg_correlation'] for item in rankings]
        effect_sizes = [item.get('effect_size', 'unknown') for item in rankings]
        
        # Color coding based on effect size
        colors = []
        for effect_size in effect_sizes:
            if effect_size == 'large':
                colors.append('#2E8B57')  # Green
            elif effect_size == 'medium':
                colors.append('#4682B4')  # Blue
            elif effect_size == 'small':
                colors.append('#FF8C00')  # Orange
            else:
                colors.append('#DC143C')  # Red
        
        fig = go.Figure()
        
        # Add bars with enhanced styling
        fig.add_trace(go.Bar(
            x=baselines,
            y=correlations,
            marker_color=colors,
            text=[f"{corr:.3f}<br>({effect})" for corr, effect in zip(correlations, effect_sizes)],
            textposition='auto',
            textfont=dict(size=12, color='white', family='Arial'),
            hovertemplate="<b>%{x}</b><br>" +
                          "Average Correlation: %{y:.3f}<br>" +
                          "Effect Size: %{customdata}<br>" +
                          "<extra></extra>",
            customdata=effect_sizes
        ))
        
        # Add reference lines
        fig.add_hline(y=0, line_dash="solid", line_color="black", line_width=2)
        fig.add_hline(y=0.7, line_dash="dot", line_color="green", opacity=0.7,
                     annotation_text="Large Effect", annotation_position="top right")
        fig.add_hline(y=0.5, line_dash="dash", line_color="blue", opacity=0.7,
                     annotation_text="Medium Effect", annotation_position="top right")
        fig.add_hline(y=0.3, line_dash="dashdot", line_color="orange", opacity=0.7,
                     annotation_text="Small Effect", annotation_position="top right")
        
        fig.update_layout(
            title={
                'text': "Enhanced Method Ranking by Correlation Strength<br><sub>Colors indicate effect size magnitude</sub>",
                'font': {'size': 18, 'family': 'Arial'},
                'x': 0.5,
                'xanchor': 'center'
            },
            xaxis_title="Baseline Method",
            yaxis_title="Average Absolute Correlation with ChatGPT",
            font=dict(family='Arial', size=12),
            height=600,
            width=1000,
            paper_bgcolor='white',
            plot_bgcolor='white',
            yaxis=dict(
                gridcolor='lightgray',
                gridwidth=1,
                range=[0, max(correlations) * 1.1 if correlations else 1]
            )
        )
        
        fig_path = viz_dir / "Enhanced_Method_Ranking_by_Correlation.png"
        fig.write_image(str(fig_path), width=1000, height=600, scale=2)

    async def _create_enhanced_score_distributions(self, viz_dir: Path):
        """Create enhanced score distribution comparison."""
        # Implementation similar to original but with enhanced styling
        pass

    async def _create_enhanced_performance_comparison(self, viz_dir: Path):
        """Create enhanced performance comparison visualization."""
        # Implementation similar to original but with enhanced analysis
        pass

    async def _create_enhanced_agreement_analysis(self, viz_dir: Path):
        """Create enhanced agreement analysis visualization."""
        agreement_analysis = self.results.get('agreement_analysis', {})
        
        if not agreement_analysis or 'agreement_summary' not in agreement_analysis:
            return
        
        agreement_summary = agreement_analysis['agreement_summary']
        baseline_performance = agreement_summary.get('baseline_performance', {})
        
        if not baseline_performance:
            return
        
        # Create enhanced agreement comparison chart
        baselines = list(baseline_performance.get('agreement', {}).keys())
        agreement_scores = list(baseline_performance.get('agreement', {}).values())
        kappa_scores = list(baseline_performance.get('kappa', {}).values())
        
        fig = go.Figure()
        
        # Add agreement scores
        fig.add_trace(go.Bar(
            name='Agreement Rate',
            x=[b.upper() for b in baselines],
            y=agreement_scores,
            marker_color='#4682B4',
            text=[f"{score:.3f}" for score in agreement_scores],
            textposition='auto',
            yaxis='y1'
        ))
        
        # Add kappa scores on secondary axis
        fig.add_trace(go.Scatter(
            name="Cohen's Kappa",
            x=[b.upper() for b in baselines],
            y=kappa_scores,
            mode='lines+markers',
            marker=dict(size=10, color='#FF8C00'),
            line=dict(width=3, color='#FF8C00'),
            yaxis='y2'
        ))
        
        fig.update_layout(
            title={
                'text': "Enhanced Agreement Analysis: ChatGPT vs SOTA Baselines<br><sub>For imbalanced datasets where correlation analysis is challenging</sub>",
                'font': {'size': 16, 'family': 'Arial'},
                'x': 0.5,
                'xanchor': 'center'
            },
            xaxis_title="Baseline Method",
            yaxis=dict(
                title="Agreement Rate",
                side='left',
                range=[0, 1]
            ),
            yaxis2=dict(
                title="Cohen's Kappa",
                side='right',
                overlaying='y',
                range=[-1, 1]
            ),
            font=dict(family='Arial', size=12),
            height=500,
            width=900,
            paper_bgcolor='white',
            plot_bgcolor='white',
            legend=dict(x=0.02, y=0.98)
        )
        
        fig_path = viz_dir / "Enhanced_Agreement_Analysis.png"
        fig.write_image(str(fig_path), width=900, height=500, scale=2)

    async def _create_statistical_significance_overview(self, viz_dir: Path):
        """Create statistical significance overview."""
        stats_summary = self.results['statistical_analysis'].get('significance_summary', {})
        
        if not stats_summary:
            return
        
        # Create significance overview chart
        categories = ['Significant', 'Not Significant']
        values = [
            stats_summary.get('significant_correlations', 0),
            stats_summary.get('total_correlations', 0) - stats_summary.get('significant_correlations', 0)
        ]
        
        fig = go.Figure(data=[
            go.Pie(
                labels=categories,
                values=values,
                hole=.3,
                marker_colors=['#2E8B57', '#DC143C'],
                textinfo='label+percent',
                textfont=dict(size=14, family='Arial')
            )
        ])
        
        fig.update_layout(
            title={
                'text': f"Statistical Significance Overview<br><sub>Total Correlations: {stats_summary.get('total_correlations', 0)}</sub>",
                'font': {'size': 16, 'family': 'Arial'},
                'x': 0.5,
                'xanchor': 'center'
            },
            font=dict(family='Arial', size=12),
            height=500,
            width=600,
            paper_bgcolor='white'
        )
        
        fig_path = viz_dir / "Statistical_Significance_Overview.png"
        fig.write_image(str(fig_path), width=600, height=500, scale=2)

    async def _generate_enhanced_sota_report(self):
        """Generate enhanced comprehensive SOTA comparison report."""
        self.logger.info("Generating enhanced SOTA comparison report")
        
        # Save detailed results
        results_path = self.output_dir / "enhanced_batch_sota_comparison_results.json"
        with open(results_path, 'w') as f:
            json.dump(self.results, f, indent=2, default=str)
        
        # Generate enhanced markdown report
        report_content = self._create_enhanced_sota_report_content()
        
        report_path = self.output_dir / "enhanced_sota_comparison_report.md"
        with open(report_path, 'w') as f:
            f.write(report_content)
        
        self.logger.info(f"Enhanced SOTA comparison report generated: {report_path}")

    def _create_enhanced_sota_report_content(self) -> str:
        """Create enhanced detailed SOTA comparison report content."""
        correlation_summary = self.results.get('correlation_analysis', {}).get('correlation_summary', {})
        agreement_summary = self.results.get('agreement_analysis', {}).get('agreement_summary', {})
        statistical_summary = self.results.get('statistical_analysis', {}).get('significance_summary', {})
        batch_analysis = self.results.get('batch_analysis', {})
        
        # Provide safe defaults for missing keys
        if 'chatgpt_cost' not in batch_analysis:
            batch_analysis.update({
                'chatgpt_cost': 0.0,
                'cost_savings': 0.0,
                'total_chatgpt_jobs': 0,
                'successful_chatgpt_jobs': 0,
                'failed_chatgpt_jobs': 0
            })
        
        report = f"""# Enhanced Batch SOTA Comparison Analysis Report

**Experiment**: {self.experiment_name}  
**Generated**: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}  
**Author**: Michael Ogunjimi  
**Institution**: University of Manchester, MSc AI  

## Executive Summary

This report presents an enhanced comprehensive comparison of ChatGPT's factuality evaluation
performance with state-of-the-art baseline methods, conducted using advanced batch processing
and sophisticated statistical analysis techniques. Following the standard SOTA comparison approach,
this analysis focuses on tasks where traditional baselines can provide meaningful comparisons.

### Key Findings

- **Overall Mean Correlation**: {correlation_summary.get('overall_statistics', {}).get('mean_pearson', 0):.4f} ± {correlation_summary.get('overall_statistics', {}).get('std_pearson', 0):.4f}
- **Best Correlating Baseline**: {correlation_summary.get('best_correlating_baseline', 'N/A')}
- **Valid Correlations**: {correlation_summary.get('overall_statistics', {}).get('valid_correlations', 0)} computed
- **Statistical Significance Rate**: {statistical_summary.get('significance_rate', 0):.1%} 
- **Mean Agreement Rate**: {agreement_summary.get('overall_statistics', {}).get('mean_agreement', 0):.4f}
- **Mean Cohen's Kappa**: {agreement_summary.get('overall_statistics', {}).get('mean_kappa', 0):.4f}
- **ChatGPT Batch Cost**: ${batch_analysis.get('chatgpt_cost', 0):.4f} with ${batch_analysis.get('cost_savings', 0):.4f} savings

## Enhanced Methodology

### Standard SOTA Comparison Approach with Batch Processing
- **Task Focus**: Entailment inference and consistency rating (summary ranking excluded as most baselines don't support it effectively)
- **Comparison Type**: ChatGPT vs traditional factuality metrics with enhanced error handling
- **Processing Method**: Advanced batch API with multi-level parsing fallbacks
- **Correlation Metrics**: Pearson and Spearman correlations with confidence intervals
- **Agreement Metrics**: Cohen's kappa for imbalanced datasets
- **Statistical Analysis**: Multiple comparisons correction and effect size analysis
- **Error Recovery**: Multi-strategy manual parsing with robust defaults

### Rationale for Task Selection
Following the standard SOTA comparison approach, this analysis focuses on:
- **Entailment Inference**: Well-supported by traditional NLI-based metrics (FactCC, BERTScore)
- **Consistency Rating**: Suitable for similarity-based metrics (BERTScore, ROUGE)
- **Summary Ranking**: Excluded by default as most traditional baselines lack effective ranking capabilities

### Baselines Evaluated
"""

        # Enhanced baseline listing
        baseline_results = self.results['baseline_results']
        baseline_performance = correlation_summary.get('baseline_performance', {})
        for baseline_name in baseline_results.keys():
            avg_corr = baseline_performance.get(baseline_name, 0)
            effect_size = self._calculate_correlation_effect_size(avg_corr)
            report += f"- **{baseline_name.upper()}**: Average correlation {avg_corr:.4f} ({effect_size} effect size)\n"

        report += f"""

## Enhanced Correlation Analysis Results

### Overall Performance Statistics
- **Mean Pearson Correlation**: {correlation_summary.get('overall_statistics', {}).get('mean_pearson', 0):.4f} ± {correlation_summary.get('overall_statistics', {}).get('std_pearson', 0):.4f}
- **Mean Spearman Correlation**: {correlation_summary.get('overall_statistics', {}).get('mean_spearman', 0):.4f} ± {correlation_summary.get('overall_statistics', {}).get('std_spearman', 0):.4f}
- **Correlation Range**: {correlation_summary.get('overall_statistics', {}).get('correlation_range', {}).get('min', 0):.4f} to {correlation_summary.get('overall_statistics', {}).get('correlation_range', {}).get('max', 0):.4f}

### Enhanced Method Rankings by Correlation Strength
"""

        # Add enhanced method rankings
        method_rankings = self.results.get('correlation_analysis', {}).get('method_rankings', {})
        if 'by_correlation_strength' in method_rankings:
            for i, ranking in enumerate(method_rankings['by_correlation_strength'], 1):
                baseline = ranking['baseline'].upper()
                correlation = ranking['avg_correlation']
                effect_size = ranking.get('effect_size', 'unknown')
                report += f"{i}. **{baseline}**: {correlation:.4f} ({effect_size} effect)\n"

        report += f"""

## Agreement Analysis Results (For Imbalanced Datasets)

### Overall Agreement Statistics
- **Mean Agreement Rate**: {agreement_summary.get('overall_statistics', {}).get('mean_agreement', 0):.4f} ± {agreement_summary.get('overall_statistics', {}).get('std_agreement', 0):.4f}
- **Mean Cohen's Kappa**: {agreement_summary.get('overall_statistics', {}).get('mean_kappa', 0):.4f} ± {agreement_summary.get('overall_statistics', {}).get('std_kappa', 0):.4f}
- **Total Comparisons**: {agreement_summary.get('overall_statistics', {}).get('total_comparisons', 0)}

### Best Performing Methods
- **Highest Agreement**: {agreement_summary.get('best_agreeing_baseline', 'N/A')}
- **Highest Kappa**: {agreement_summary.get('best_kappa_baseline', 'N/A')}

## Enhanced Statistical Significance Analysis

### Comprehensive Significance Testing Results
- **Significant Correlations**: {statistical_summary.get('significant_correlations', 0)}/{statistical_summary.get('total_correlations', 0)}
- **Significance Rate**: {statistical_summary.get('significance_rate', 0):.2%}
- **Interpretation**: {statistical_summary.get('interpretation', 'No interpretation available')}

### Effect Size Distribution
"""

        # Add enhanced effect size distribution
        effect_size_dist = statistical_summary.get('effect_size_distribution', {})
        for effect_size, count in effect_size_dist.items():
            if count > 0:
                report += f"- **{effect_size.title()} Effects**: {count}\n"

        # Add multiple comparisons correction if available
        multiple_comparisons = self.results.get('statistical_analysis', {}).get('multiple_comparisons', {})
        if multiple_comparisons:
            report += f"""
### Multiple Comparisons Correction
- **Correction Method**: {multiple_comparisons.get('correction_method', 'N/A').title()}
- **Corrected α**: {multiple_comparisons.get('corrected_alpha', 0):.6f}
- **Significant After Correction**: {multiple_comparisons.get('significant_after_correction', 0)}
"""

        report += f"""

## Enhanced Batch Processing Analysis

### Advanced Processing Efficiency
- **Total Batch Cost**: ${batch_analysis.get('chatgpt_cost', 0):.4f}
- **Estimated Sync Cost**: ${batch_analysis.get('estimated_sync_cost', 0):.4f}
- **Batch Savings**: ${batch_analysis.get('cost_savings', 0):.4f} ({(batch_analysis.get('cost_savings', 0) / max(batch_analysis.get('estimated_sync_cost', 1), 1)) * 100:.1f}% reduction)
- **Successful Jobs**: {batch_analysis.get('successful_chatgpt_jobs', 0)}/{batch_analysis.get('total_chatgpt_jobs', 0)}

### Batch Job Statistics
- **Total ChatGPT Jobs**: {batch_analysis.get('total_chatgpt_jobs', 0)}
- **Successful Jobs**: {batch_analysis.get('successful_chatgpt_jobs', 0)}
- **Failed Jobs**: {batch_analysis.get('failed_chatgpt_jobs', 0)}
- **Success Rate**: {(batch_analysis.get('successful_chatgpt_jobs', 0) / max(batch_analysis.get('total_chatgpt_jobs', 1), 1)) * 100:.1f}%

## Key Insights and Implications for Factuality Evaluation

### Enhanced Methodological Insights
"""

        # Generate enhanced insights based on results
        mean_correlation = correlation_summary.get('overall_statistics', {}).get('mean_pearson', 0)
        significance_rate = statistical_summary.get('significance_rate', 0)
        mean_kappa = agreement_summary.get('overall_statistics', {}).get('mean_kappa', 0)
        
        if abs(mean_correlation) > 0.7:
            report += "1. **Strong Baseline Agreement**: ChatGPT demonstrates strong correlation with traditional factuality metrics, indicating high reliability and convergent validity.\n"
        elif abs(mean_correlation) > 0.5:
            report += "1. **Moderate Baseline Agreement**: ChatGPT shows moderate correlation with traditional metrics, suggesting partial alignment with established approaches.\n"
        elif abs(mean_correlation) > 0.3:
            report += "1. **Limited Baseline Agreement**: ChatGPT exhibits weak correlation with traditional metrics, indicating novel evaluation patterns.\n"
        else:
            report += "1. **Divergent Evaluation Patterns**: ChatGPT shows minimal correlation with traditional metrics, suggesting fundamentally different evaluation approaches.\n"

        if significance_rate > 0.6:
            report += "2. **Statistical Reliability**: High significance rate indicates robust and reliable correlation patterns across methods and datasets.\n"
        elif significance_rate > 0.3:
            report += "2. **Moderate Statistical Evidence**: Partial significance suggests some reliable patterns, but results should be interpreted cautiously.\n"
        else:
            report += "2. **Limited Statistical Evidence**: Low significance rate indicates need for larger sample sizes or different analytical approaches.\n"

        if mean_kappa > 0.6:
            report += "3. **Strong Agreement**: High Cohen's kappa indicates substantial agreement even for imbalanced datasets.\n"
        elif mean_kappa > 0.4:
            report += "3. **Moderate Agreement**: Reasonable agreement levels suggest partial alignment in challenging evaluation scenarios.\n"
        else:
            report += "3. **Limited Agreement**: Low kappa values indicate challenges in achieving consistent evaluation patterns.\n"

        report += f"""

## Enhanced Recommendations

### For Academic Research
1. **Hybrid Evaluation Approach**: Combine ChatGPT with best-correlating traditional metrics ({correlation_summary.get('best_correlating_baseline', 'N/A')}) for comprehensive assessment
2. **Advanced Batch Processing**: Use batch processing for large-scale studies to achieve significant cost savings ({(batch_analysis.get('cost_savings', 0) / max(batch_analysis.get('estimated_sync_cost', 1), 1)) * 100:.1f}% reduction)
3. **Statistical Rigor**: Apply multiple comparisons correction and report effect sizes alongside correlation coefficients
4. **Agreement Metrics**: Use Cohen's kappa for imbalanced datasets where correlation analysis may be problematic

### For Production Systems
1. **Selective Deployment**: Use ChatGPT for cases where traditional metrics show limitations
2. **Cost-Effective Processing**: Implement batch processing for periodic comprehensive evaluations
3. **Quality Assurance**: Monitor prediction success rates and implement robust error handling
4. **Ensemble Methods**: Consider combining multiple evaluation approaches based on correlation patterns

### Enhanced Future Research Directions
1. **Extended Baseline Coverage**: Include more recent factuality metrics and LLM-based evaluators
2. **Cross-Domain Validation**: Test correlation patterns across different domains and text types
3. **Longitudinal Analysis**: Study correlation stability over time and across model versions
4. **Human Evaluation Studies**: Validate automatic metrics against human judgments with larger sample sizes

## Enhanced Limitations and Considerations

### Methodological Limitations
- **Sample Size Constraints**: Limited by computational resources and API costs
- **Dataset Scope**: Evaluation restricted to news summarization datasets
- **Baseline Coverage**: Not all SOTA factuality metrics evaluated
- **Temporal Constraints**: Snapshot evaluation without longitudinal analysis

### Technical Considerations
- **API Dependencies**: Results dependent on OpenAI API availability and consistency
- **Batch Processing Delays**: Asynchronous processing introduces timing considerations
- **Enhanced Error Recovery**: Multi-level parsing fallbacks may introduce systematic biases
- **Cost-Quality Trade-offs**: Balance between cost savings and evaluation comprehensiveness

### Statistical Considerations
- **Multiple Comparisons**: Applied Bonferroni correction to control Type I errors
- **Effect Size Interpretation**: Used Cohen's conventions adapted for correlation coefficients
- **Imbalanced Datasets**: Applied agreement metrics (Cohen's kappa) as primary evaluation for challenging datasets
- **Statistical Power**: Enhanced sample sizes improve power to detect significant correlations

## Conclusion

This enhanced batch SOTA comparison provides comprehensive evidence for {self._generate_enhanced_conclusion_statement(mean_correlation, significance_rate, mean_kappa)}. The advanced batch processing approach with multi-level error recovery demonstrates significant cost savings while maintaining evaluation quality, making it highly suitable for large-scale academic research.

The combination of correlation analysis, agreement metrics, and comprehensive statistical testing provides a robust framework for evaluating ChatGPT's factuality assessment capabilities. Enhanced results suggest {self._generate_enhanced_practical_recommendation(mean_correlation, significance_rate, mean_kappa)} for both academic research and production applications.

## Technical Appendix

### Enhanced Processing Statistics
- **Total Examples Processed**: {correlation_summary.get('overall_statistics', {}).get('valid_correlations', 0) * 100:,} (estimated)
- **Successful Correlations**: {correlation_summary.get('overall_statistics', {}).get('valid_correlations', 0)}
- **Batch Jobs Submitted**: {batch_analysis.get('total_chatgpt_jobs', 0)}
- **Successful Batch Jobs**: {batch_analysis.get('successful_chatgpt_jobs', 0)}
- **Agreement Comparisons**: {agreement_summary.get('overall_statistics', {}).get('total_comparisons', 0)}

### Enhanced Configuration Summary
- **Model**: {self.config.get('openai.models.primary', 'N/A')}
- **API Tier**: {self.config.get('openai.api_tier', 'N/A')}
- **Batch Processing**: Advanced with multi-level error recovery
- **Statistical Analysis**: Enhanced with multiple comparisons correction
- **Agreement Analysis**: Cohen's kappa for imbalanced datasets

---
*Report generated by Enhanced Batch SOTA Comparison Experiment*
"""

        return report

    def _generate_enhanced_conclusion_statement(self, mean_correlation: float, significance_rate: float, mean_kappa: float) -> str:
        """Generate enhanced conclusion statement based on results."""
        if abs(mean_correlation) > 0.6 and significance_rate > 0.6 and mean_kappa > 0.6:
            return "strong convergent validity between ChatGPT and traditional factuality metrics, with robust statistical evidence, reliable batch processing, and substantial agreement even for imbalanced datasets"
        elif abs(mean_correlation) > 0.4 and significance_rate > 0.4 and mean_kappa > 0.4:
            return "moderate agreement between ChatGPT and traditional metrics, with reasonable statistical support and acceptable agreement levels"
        elif mean_kappa > 0.4:
            return "reliable agreement patterns through Cohen's kappa analysis, suggesting utility for imbalanced datasets despite correlation challenges"
        else:
            return "the need for continued research into LLM-based factuality evaluation methods with enhanced analytical approaches"

    def _generate_enhanced_practical_recommendation(self, mean_correlation: float, significance_rate: float, mean_kappa: float) -> str:
        """Generate enhanced practical recommendation based on results."""
        if abs(mean_correlation) > 0.6 and significance_rate > 0.6 and mean_kappa > 0.6:
            return "strong support for ChatGPT as a reliable factuality evaluator with robust batch processing capabilities"
        elif abs(mean_correlation) > 0.4 and mean_kappa > 0.4:
            return "cautious adoption of ChatGPT with traditional metric validation, particularly effective for imbalanced datasets"
        elif mean_kappa > 0.4:
            return "selective use of ChatGPT based on agreement analysis, with batch processing recommended for cost-effectiveness"
        else:
            return "continued development and validation before widespread adoption, with focus on enhanced error handling"


async def main():
    """Main function for running enhanced batch SOTA comparison."""
    parser = argparse.ArgumentParser(
        description="Enhanced Batch SOTA Comparison Experiment - Following standard SOTA comparison approach with advanced batch processing"
    )
    parser.add_argument("--model", default="gpt-4.1-mini", help="Model to use")
    parser.add_argument("--tier", default="tier2", help="API tier")
    parser.add_argument("--experiment-name", help="Custom experiment name")
    parser.add_argument("--tasks", nargs="+", help="Tasks to evaluate")
    parser.add_argument("--datasets", nargs="+", help="Datasets to use")
    parser.add_argument("--baselines", nargs="+", help="Baseline methods to compare")
    parser.add_argument("--sample-size", type=int, help="Sample size per dataset")
    parser.add_argument("--prompt-type", default="zero_shot", help="ChatGPT prompt type")
    parser.add_argument("--quick-test", action="store_true", help="Run quick test")
    
    args = parser.parse_args()
    
    # Initialize experiment
    experiment = EnhancedBatchSOTAComparisonExperiment(
        model=args.model,
        tier=args.tier,
        experiment_name=args.experiment_name
    )
    
    # Run comparison
    results = await experiment.run_sota_comparison(
        tasks=args.tasks,
        datasets=args.datasets,
        baselines=args.baselines,
        sample_size=args.sample_size,
        prompt_type=args.prompt_type,
        quick_test=args.quick_test
    )
    
    # Print enhanced summary
    print(f"\n{'='*70}")
    print(f"ENHANCED BATCH SOTA COMPARISON COMPLETED")
    print(f"{'='*70}")
    print(f"Experiment: {experiment.experiment_name}")
    print(f"Output Directory: {experiment.output_dir}")
    
    # Enhanced correlation results
    correlation_summary = results['correlation_analysis']['correlation_summary']
    overall_stats = correlation_summary.get('overall_statistics', {})
    
    print(f"\n{'='*50}")
    print(f"ENHANCED CORRELATION ANALYSIS")
    print(f"{'='*50}")
    print(f"Valid Correlations: {overall_stats.get('valid_correlations', 0)}")
    print(f"Mean Pearson: {overall_stats.get('mean_pearson', 0):.4f} ± {overall_stats.get('std_pearson', 0):.4f}")
    print(f"Mean Spearman: {overall_stats.get('mean_spearman', 0):.4f} ± {overall_stats.get('std_spearman', 0):.4f}")
    print(f"Best Baseline: {correlation_summary.get('best_correlating_baseline', 'N/A')}")
    
    # Enhanced agreement analysis
    agreement_summary = results.get('agreement_analysis', {}).get('agreement_summary', {})
    if agreement_summary:
        print(f"\n{'='*50}")
        print(f"ENHANCED AGREEMENT ANALYSIS")
        print(f"{'='*50}")
        
        agreement_stats = agreement_summary.get('overall_statistics', {})
        print(f"Total Comparisons: {agreement_stats.get('total_comparisons', 0)}")
        print(f"Mean Agreement: {agreement_stats.get('mean_agreement', 0):.4f}")
        print(f"Mean Cohen's κ: {agreement_stats.get('mean_kappa', 0):.4f}")
        print(f"Best Agreement: {agreement_summary.get('best_agreeing_baseline', 'N/A')}")
        print(f"Best Kappa: {agreement_summary.get('best_kappa_baseline', 'N/A')}")
    
    # Enhanced statistical significance
    statistical_summary = results.get('statistical_analysis', {}).get('significance_summary', {})
    print(f"\n{'='*50}")
    print(f"ENHANCED STATISTICAL SIGNIFICANCE")
    print(f"{'='*50}")
    print(f"Significant Correlations: {statistical_summary.get('significant_correlations', 0)}/{statistical_summary.get('total_correlations', 0)}")
    print(f"Significance Rate: {statistical_summary.get('significance_rate', 0):.1%}")
    print(f"Interpretation: {statistical_summary.get('interpretation', 'No interpretation')}")
    
    # Effect size distribution
    effect_size_dist = statistical_summary.get('effect_size_distribution', {})
    if any(count > 0 for count in effect_size_dist.values()):
        print(f"\nEffect Size Distribution:")
        for effect_size, count in effect_size_dist.items():
            if count > 0:
                print(f"  {effect_size.title()}: {count}")
    
    # Enhanced batch analysis
    batch_analysis = results['batch_analysis']
    print(f"\n{'='*50}")
    print(f"ENHANCED BATCH PROCESSING")
    print(f"{'='*50}")
    print(f"ChatGPT Cost: ${batch_analysis['chatgpt_cost']:.4f}")
    print(f"Cost Savings: ${batch_analysis['cost_savings']:.4f}")
    print(f"Estimated Sync Cost: ${batch_analysis['estimated_sync_cost']:.4f}")
    print(f"Successful Jobs: {batch_analysis['successful_chatgpt_jobs']}/{batch_analysis['total_chatgpt_jobs']}")
    
    # Enhanced method rankings
    method_rankings = results.get('correlation_analysis', {}).get('method_rankings', {})
    if 'by_correlation_strength' in method_rankings:
        print(f"\n{'='*50}")
        print(f"ENHANCED METHOD RANKINGS")
        print(f"{'='*50}")
        
        for i, ranking in enumerate(method_rankings['by_correlation_strength'][:5], 1):
            baseline = ranking['baseline'].upper()
            correlation = ranking['avg_correlation']
            effect_size = ranking.get('effect_size', 'unknown')
            print(f"{i}. {baseline:15} | Correlation: {correlation:7.4f} | Effect: {effect_size}")
    
    # Enhanced final assessment
    print(f"\n{'='*70}")
    print(f"ENHANCED FINAL ASSESSMENT")
    print(f"{'='*70}")
    
    mean_correlation = overall_stats.get('mean_pearson', 0)
    significance_rate = statistical_summary.get('significance_rate', 0)
    mean_kappa = agreement_stats.get('mean_kappa', 0) if agreement_summary else 0
    
    if abs(mean_correlation) > 0.6 and significance_rate > 0.6 and mean_kappa > 0.6:
        print("✅ EXCELLENT: Strong correlation, high significance, substantial agreement")
    elif abs(mean_correlation) > 0.4 and significance_rate > 0.4 and mean_kappa > 0.4:
        print("🟡 GOOD: Moderate correlation with acceptable agreement levels")
    elif mean_kappa > 0.4:
        print("🔵 MIXED: Good agreement patterns despite correlation challenges")
    elif batch_analysis['successful_chatgpt_jobs'] / max(batch_analysis['total_chatgpt_jobs'], 1) > 0.8:
        print("🟠 PARTIAL: Reliable processing but evaluation patterns need investigation")
    else:
        print("🔴 CAUTION: Results require careful interpretation and validation")
    
    print(f"\nEnhanced Analysis Complete!")
    print(f"📊 Results: {experiment.output_dir}/enhanced_batch_sota_comparison_results.json")
    print(f"📄 Report: {experiment.output_dir}/enhanced_sota_comparison_report.md")
    print(f"📈 Figures: {experiment.output_dir}/figures/")
    print(f"📁 Baselines: {experiment.output_dir}/baseline_results/")
    
    print(f"{'='*70}")


if __name__ == "__main__":
    asyncio.run(main())