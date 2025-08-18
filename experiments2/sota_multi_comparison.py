#!/usr/bin/env python3
"""
Multi-LLM SOTA Baseline Comparison Experiment
============================================

Enhanced implementation for comparing Multiple LLMs' factuality evaluation 
performance with state-of-the-art baseline methods.

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
import warnings

import numpy as np
from scipy.stats import pearsonr, spearmanr, kendalltau
from scipy import stats
import plotly.graph_objects as go
from plotly.subplots import make_subplots
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score, cohen_kappa_score

# Suppress warnings for cleaner output
warnings.filterwarnings('ignore')

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
        get_config,
        setup_reproducibility,
        validate_api_keys
    )
    from src.data import quick_load_dataset
    from src.baselines import (
        get_available_baselines,
        BaselineComparator,
        create_all_baselines
    )
    from experiments2.run_llm_evaluation import MultiLLMEvaluationExperiment
except ImportError as e:
    print(f"Error importing required modules: {e}")
    print("Please ensure you're running from the project root directory.")
    sys.exit(1)


class SOTAMultiLLMComparisonExperiment:
    """
    Enhanced experimental framework for comparing multiple LLMs with SOTA baseline methods.
    
    This class orchestrates comprehensive comparisons between multiple LLMs' zero-shot
    factuality evaluation and established baseline metrics, providing correlation
    analysis, performance benchmarking, and statistical testing.
    """
    
    # Comprehensive color palette for all visualizations
    COLOR_SCHEMES = {
        'baselines': {
            'factcc': '#E74C3C',      # Red - strong and distinctive
            'bertscore': '#3498DB',   # Blue - professional 
            'rouge': '#27AE60'        # Green - natural
        },
        'models': {
            'gpt-4.1-mini': '#9B59B6',     # Purple - premium
            'gpt-4o': '#8E44AD',           # Dark purple
            'gpt-4o-mini': '#BB8FCE',      # Light purple
            'llama3.1:8b': '#E67E22',      # Orange - distinctive
            'qwen2.5:7b': '#F39C12',       # Amber - warm
            'claude-3.5-sonnet': '#1ABC9C', # Teal - sophisticated
            'gemini-pro': '#E91E63'        # Pink - vibrant
        },
        'tasks': {
            'entailment_inference': '#FF6B6B',    # Coral red
            'summary_ranking': '#4ECDC4',         # Turquoise
            'consistency_rating': '#FFE66D'       # Yellow
        },
        'datasets': {
            'frank': '#FF8A65',        # Orange red
            'summeval': '#81C784'      # Light green
        },
        'task_dataset_combinations': {
            'entailment_inference_frank': '#FF6B6B',
            'entailment_inference_summeval': '#FF8A80', 
            'summary_ranking_frank': '#4ECDC4',
            'summary_ranking_summeval': '#80CBC4',
            'consistency_rating_frank': '#FFE66D',
            'consistency_rating_summeval': '#FFF176'
        },
        'performance_levels': {
            'excellent': '#27AE60',    # Green
            'good': '#F39C12',         # Orange  
            'fair': '#E67E22',         # Dark orange
            'poor': '#E74C3C'          # Red
        },
        'agreement_metrics': {
            'high_agreement': '#2ECC71',      # Bright green
            'medium_agreement': '#F1C40F',    # Yellow
            'low_agreement': '#E74C3C',       # Red
            'binary_agreement': '#3498DB'     # Blue
        }
    }
    
    def __init__(self, model: str, tier: str, experiment_name: str = None, output_dir: str = None):
        """Initialize the Multi-LLM SOTA comparison experiment."""
        # Store tier for subprocess calls
        self.tier = tier
        
        # Load configuration
        self.config = get_config(model=model, tier=tier)
        
        # Set up experiment tracking
        self.experiment_name = experiment_name or f"multi_llm_sota_comparison_{datetime.now().strftime('%Y%m%d_%H%M%S')}"
        
        # Set up output directory structure
        if output_dir:
            self.base_output_dir = Path(output_dir)
        else:
            self.base_output_dir = Path("results/experiments") / self.experiment_name
        
        # Create SOTA-specific subdirectory
        self.output_dir = self.base_output_dir / "sota_multi_comparison"
        
        # Create complete directory structure
        self._create_directory_structure()
        
        # Set up logging with reduced verbosity
        self.experiment_logger = setup_experiment_logger(
            self.experiment_name,
            self.config,
            str(self.output_dir / "logs")
        )
        self.logger = self.experiment_logger.logger
        self._configure_logging_levels()
        
        # Set up reproducibility
        setup_reproducibility(self.config)
        validate_api_keys(self.config)
        
        # Initialize components
        self.visualization_engine = create_visualization_engine(self.config)
        self.baseline_comparator = BaselineComparator(self.config)
        
        # Initialize results storage with complete structure
        self.results = {
            'experiment_metadata': {
                'name': self.experiment_name,
                'timestamp': datetime.now().isoformat(),
                'config': self.config.to_dict(),
                'experiment_type': 'multi_llm_sota_comparison'
            },
            'llm_results': {},
            'baseline_results': {},
            'correlation_analysis': {},
            'performance_comparison': {},
            'statistical_analysis': {},
            'cost_analysis': {},
            'visualizations': {},
            'summary_statistics': {}
        }
        
        # Store models list (will be set by main)
        self.models = []
        
        # Log initialization with metadata
        self.logger.info(
            "Multi-LLM SOTA comparison experiment initialized",
            extra={
                'experiment_name': self.experiment_name,
                'task_name': 'initialization',
                'metadata': {
                    'config_model': self.config.get('model', {}).get('name', 'unknown'),
                    'tier': tier,
                    'output_dir': str(self.output_dir)
                }
            }
        )
        
        print(f"🤖 Multi-LLM SOTA comparison experiment initialized: {self.experiment_name}")
    
    def get_color(self, category: str, item: str, default: str = '#888888') -> str:
        """Get color for specific item from color schemes."""
        return self.COLOR_SCHEMES.get(category, {}).get(item, default)
    
    def get_baseline_color(self, baseline_name: str) -> str:
        """Get color for baseline method."""
        return self.get_color('baselines', baseline_name, '#888888')
    
    def get_model_color(self, model_name: str) -> str:
        """Get color for model."""
        return self.get_color('models', model_name, '#888888')
    
    def get_task_color(self, task_name: str) -> str:
        """Get color for task."""
        return self.get_color('tasks', task_name, '#888888')
    
    def get_task_dataset_color(self, task_name: str, dataset_name: str) -> str:
        """Get color for task-dataset combination."""
        combination = f"{task_name}_{dataset_name}"
        return self.get_color('task_dataset_combinations', combination, '#888888')
    
    def _create_directory_structure(self):
        """Create complete directory structure for the experiment."""
        directories = [
            self.base_output_dir,
            self.output_dir,
            self.output_dir / "figures",
            self.output_dir / "logs",
            self.output_dir / "baseline_results",
            self.output_dir / "llm_results",
            self.output_dir / "correlations",
            self.output_dir / "statistics",
            self.output_dir / "tables"
        ]
        
        for directory in directories:
            directory.mkdir(parents=True, exist_ok=True)
    
    def _configure_logging_levels(self):
        """Configure logging levels to reduce verbosity."""
        # Suppress verbose logging from external libraries
        suppressed_loggers = [
            "httpx", "openai", "choreographer", "kaleido", "progress",
            "cost_tracker", "PromptManager", "OpenAIClient", "transformers",
            "absl", "torch", "urllib3", "requests", "matplotlib", "PIL"
        ]
        
        for logger_name in suppressed_loggers:
            logging.getLogger(logger_name).setLevel(logging.ERROR)
        
        experiment_logger = logging.getLogger(f"experiment.{self.experiment_name}")
        
        # Remove only console handlers, keep file handlers
        handlers_to_remove = []
        for handler in experiment_logger.handlers:
            if isinstance(handler, logging.StreamHandler) and not isinstance(handler, logging.FileHandler):
                handlers_to_remove.append(handler)
        
        for handler in handlers_to_remove:
            experiment_logger.removeHandler(handler)
    
    async def run_sota_comparison(
        self,
        tasks: List[str] = None,
        datasets: List[str] = None,
        baselines: List[str] = None,
        sample_size: int = None,
        prompt_type: str = "zero_shot"
    ) -> Dict[str, Any]:
        """
        Run comprehensive SOTA baseline comparison with multiple LLMs.
        
        Args:
            tasks: List of tasks to evaluate
            datasets: List of datasets to use
            baselines: List of baseline methods
            sample_size: Number of examples per dataset
            prompt_type: LLM prompt type to use
            
        Returns:
            Complete comparison results with correlation analysis
        """
        print(f"\n🔬 Starting Multi-LLM SOTA Comparison Experiment")
        print("=" * 60)
        
        if tasks is None:
            tasks = ['entailment_inference', 'consistency_rating']
        if datasets is None:
            datasets = ['frank', 'summeval']
        if baselines is None:
            baselines = get_available_baselines()
        if sample_size is None:
            sample_size = self.config.get('experiments.sota_comparison.sample_size', 300)
        
        self.experiment_params = {
            'tasks': tasks,
            'datasets': datasets,
            'baselines': baselines,
            'sample_size': sample_size,
            'prompt_type': prompt_type
        }
        
        try:
            # Log experiment start with parameters
            self.experiment_logger.log_task_start(
                'sota_comparison_experiment',
                metadata={
                    'tasks': tasks,
                    'datasets': datasets,
                    'baselines': baselines,
                    'sample_size': sample_size,
                    'prompt_type': prompt_type,
                    'models': self.models
                }
            )
            
            # Phase 1: Run Multi-LLM evaluations
            await self._run_llm_evaluations(tasks, datasets, sample_size, prompt_type)
            
            # Phase 2: Run baseline evaluations
            await self._run_baseline_evaluations(tasks, datasets, baselines, sample_size)
            
            # Phase 3: Compute correlations
            await self._compute_correlation_analysis()
            
            # Phase 4: Performance comparison analysis
            await self._analyze_performance_comparison()
            
            # Phase 5: Statistical significance testing
            await self._perform_statistical_analysis()
            
            # Phase 6: Generate summary statistics
            await self._generate_summary_statistics()
            
            # Phase 7: Generate visualizations
            await self._generate_comparison_visualizations()
            
            # Phase 8: Save comprehensive results
            await self._save_results()
            
            # Log experiment completion
            duration = self.experiment_logger.log_task_end(
                'sota_comparison_experiment',
                metadata={
                    'models_evaluated': len(self.models),
                    'visualizations_created': len(self.results.get('visualizations', {}))
                }
            )
            
            print(f"\n✅ Multi-LLM SOTA comparison completed successfully")
            return self.results
            
        except Exception as e:
            print(f"❌ SOTA comparison experiment failed: {e}")
            self.logger.error(
                f"SOTA comparison experiment failed: {e}",
                extra={
                    'experiment_name': self.experiment_name,
                    'task_name': 'sota_comparison_experiment',
                    'metadata': {'error': str(e)}
                },
                exc_info=True
            )
            raise
    
    async def _run_llm_evaluations(
        self,
        tasks: List[str],
        datasets: List[str],
        sample_size: int,
        prompt_type: str
    ):
        """Run Multi-LLM evaluations via subprocess call."""
        print(f"\n🤖 Running Multi-LLM Evaluations")
        print("-" * 40)
        
        # Log LLM evaluation start
        self.experiment_logger.log_task_start(
            'llm_evaluations',
            metadata={
                'models': self.models,
                'tasks': tasks,
                'datasets': datasets,
                'sample_size': sample_size,
                'prompt_type': prompt_type
            }
        )
        
        import subprocess
        import sys
        
        # Build command arguments
        cmd_args = [
            sys.executable, "experiments2/run_llm_evaluation.py",
            "--models"] + self.models + [
            "--prompt-type", prompt_type,
            "--sota-follows",
            "--output-dir", str(self.base_output_dir / "llm_multi_evaluation")
        ]
        
        print(f"Models: {self.models}")

        # Add task and dataset specifications
        if tasks and len(tasks) == 1:
            cmd_args.extend(["--task", tasks[0]])
        
        if datasets and len(datasets) == 1:
            cmd_args.extend(["--dataset", datasets[0]])
        
        if sample_size:
            cmd_args.extend(["--sample-size", str(sample_size)])
        
        print(f"🔧 Command: {' '.join(cmd_args)}")

        # Run subprocess with real-time output filtering
        import sys
        import re

        process = subprocess.Popen(
            cmd_args,
            stdout=subprocess.PIPE,
            stderr=subprocess.STDOUT,
            text=True,
            cwd=Path.cwd()
        )

        # Stream output with smart filtering
        important_keywords = [
            "🎯", "⚡", "📊", "✅", "💰", "🤖", "❌", "ERROR", "Failed",
            "Models:", "Tasks:", "Datasets:", "completed", "cost:", 
            "Score:", "Visualizations", "Results saved for SOTA"
        ]

        current_task = None
        progress_active = False

        def clear_line():
            """Clear the current line"""
            sys.stdout.write('\r' + ' ' * 80 + '\r')
            sys.stdout.flush()

        def print_progress(task_name, percentage, current=None, total=None):
            """Print progress bar that updates in place"""
            bar_length = 30
            filled = int(bar_length * percentage / 100)
            bar = '█' * filled + '░' * (bar_length - filled)
            
            if current and total:
                status = f"   📊 {task_name}: [{bar}] {percentage:3d}% ({current}/{total})"
            else:
                status = f"   📊 {task_name}: [{bar}] {percentage:3d}%"
            
            clear_line()
            sys.stdout.write(status)
            sys.stdout.flush()

        while True:
            output = process.stdout.readline()
            if output == '' and process.poll() is not None:
                break
            if output:
                line = output.strip()
                
                # Check if this is a progress bar line
                if '|' in line and '%' in line and any(c in line for c in ['█', '▌', '▏', '▎', '▍', '▋', '▊', '▉']):
                    # Extract task name
                    task_match = re.match(r'^([^:]+):', line)
                    task_name = task_match.group(1) if task_match else "Processing"
                    
                    # Extract percentage and items
                    percent_match = re.search(r'(\d+)%', line)
                    items_match = re.search(r'(\d+)/(\d+)', line)
                    
                    if percent_match:
                        percentage = int(percent_match.group(1))
                        
                        # Check if this is a new task
                        if task_name != current_task:
                            if progress_active:
                                print()  # New line after previous task
                            current_task = task_name
                            progress_active = True
                        
                        # Update progress
                        if items_match:
                            current = items_match.group(1)
                            total = items_match.group(2)
                            print_progress(task_name, percentage, current, total)
                        else:
                            print_progress(task_name, percentage)
                        
                        # If complete, print final status and move to new line
                        if percentage == 100:
                            clear_line()
                            if items_match:
                                print(f"   ✅ {task_name}: Complete [{items_match.group(2)} items]")
                            else:
                                print(f"   ✅ {task_name}: Complete")
                            progress_active = False
                            current_task = None
                
                # For non-progress bar lines
                elif any(keyword in line for keyword in important_keywords):
                    if progress_active:
                        print()  # New line if progress was active
                        progress_active = False
                    print(line)

        # Ensure we end cleanly
        if progress_active:
            print()

        process.wait()
        
        if process.returncode != 0:
            raise RuntimeError(f"LLM evaluation failed with return code: {process.returncode}")
        
        # Load and process results
        results_file = Path("results/experiments/llm_sota_results.json")
        if not results_file.exists():
            raise FileNotFoundError(f"LLM evaluation results not found at {results_file}")
        
        with open(results_file, 'r') as f:
            llm_data = json.load(f)
        
        # Store LLM results
        self.results['llm_results'] = llm_data.get('llm_results', {}).get('model_results', {})
        
        # Calculate total cost
        total_cost = 0.0
        for model_results in self.results['llm_results'].values():
            for task_data in model_results.values():
                for dataset_data in task_data.values():
                    cost = dataset_data.get('cost', 0.0)
                    total_cost += cost
                    
                    # Log each cost component
                    if cost > 0:
                        self.experiment_logger.log_cost(
                            cost=cost,
                            model=dataset_data.get('model_name', 'unknown'),
                            task_name=dataset_data.get('task_name', 'unknown'),
                            metadata={
                                'dataset': dataset_data.get('dataset_name', 'unknown'),
                                'examples_processed': dataset_data.get('total_examples', 0)
                            }
                        )
        
        self.results['cost_analysis']['llm_cost'] = total_cost
        
        # Save LLM results separately
        llm_results_file = self.output_dir / "llm_results" / "llm_evaluation_results.json"
        with open(llm_results_file, 'w') as f:
            json.dump(self.results['llm_results'], f, indent=2)
        
        # Log LLM evaluation completion
        self.experiment_logger.log_task_end(
            'llm_evaluations',
            metadata={
                'models_evaluated': len(self.results['llm_results']),
                'results_file': str(llm_results_file)
            }
        )
        
        print(f"✅ Multi-LLM evaluation completed. Total cost: ${total_cost:.4f}")
    
    async def _run_baseline_evaluations(
        self,
        tasks: List[str],
        datasets: List[str],
        baselines: List[str],
        sample_size: int
    ):
        """Run baseline method evaluations on SOTA-supported tasks."""
        print(f"\n🔧 Running Baseline Evaluations")
        print("-" * 40)
        
        # Log baseline evaluation start
        self.experiment_logger.log_task_start(
            'baseline_evaluations',
            metadata={
                'baselines': baselines,
                'tasks': tasks,
                'datasets': datasets,
                'sample_size': sample_size
            }
        )
        
        baseline_instances = create_all_baselines(self.config)
        
        for baseline_name in baselines:
            if baseline_name not in baseline_instances:
                print(f"⚠️  Baseline {baseline_name} not available, skipping")
                continue
            
            print(f"🎯 Processing: {baseline_name.upper()}")
            baseline = baseline_instances[baseline_name]
            self.results['baseline_results'][baseline_name] = {}
            
            for task_name in tasks:
                if not baseline.supports_task(task_name):
                    print(f"   ⚠️  {task_name}: Not supported by {baseline_name}, skipping")
                    continue
                
                self.results['baseline_results'][baseline_name][task_name] = {}
                
                for dataset_name in datasets:
                    if not self._has_llm_results_for(task_name, dataset_name):
                        continue
                    
                    try:
                        print(f"   📊 {dataset_name}: Loading examples for baseline evaluation...")
                        examples = await self._load_examples_for_baseline(task_name, dataset_name, sample_size)
                        
                        if not examples:
                            print(f"   ⚠️  No examples found for {task_name} on {dataset_name}")
                            continue
                        
                        # Evaluate baseline
                        start_time = time.time()
                        predictions = await self._evaluate_baseline_on_examples(
                            baseline, baseline_name, task_name, examples
                        )
                        processing_time = time.time() - start_time
                        
                        # Calculate performance metrics
                        performance_metrics = self._calculate_baseline_performance(
                            predictions, examples, task_name, baseline_name
                        )
                        
                        self.results['baseline_results'][baseline_name][task_name][dataset_name] = {
                            'predictions': self._serialize_predictions(predictions),
                            'performance_metrics': performance_metrics,
                            'dataset_size': len(examples),
                            'processing_time': processing_time,
                            'baseline_method': baseline_name,
                            'prediction_distribution': self._analyze_prediction_distribution(predictions, task_name)
                        }
                        
                        # Display results
                        primary_metric = performance_metrics.get('primary_metric', 0.0)
                        print(f"   ✅ {dataset_name}: Performance={primary_metric:.3f}, Time={processing_time:.1f}s")
                        
                    except Exception as e:
                        print(f"   ❌ {dataset_name}: Failed - {str(e)[:50]}")
                        self.logger.error(f"Baseline {baseline_name} failed on {task_name}-{dataset_name}: {e}")
                        self.results['baseline_results'][baseline_name][task_name][dataset_name] = {
                            'error': str(e),
                            'status': 'failed'
                        }
        
        for baseline_name, baseline_data in self.results['baseline_results'].items():
            baseline_file = self.output_dir / "baseline_results" / f"{baseline_name}_results.json"
            with open(baseline_file, 'w') as f:
                json.dump(baseline_data, f, indent=2)
            print(f"   ✅ Saved {baseline_name} results to baseline_results/")
        
        # Log baseline evaluation completion
        self.experiment_logger.log_task_end(
            'baseline_evaluations',
            metadata={
                'baselines_evaluated': list(self.results['baseline_results'].keys()),
                'total_task_dataset_combinations': sum(
                    len(baseline_data) for baseline_data in self.results['baseline_results'].values()
                )
            }
        )
        
        print(f"✅ Baseline evaluations completed")
    
    def _has_llm_results_for(self, task_name: str, dataset_name: str) -> bool:
        """Check if we have LLM results for the given task and dataset combination."""
        for model_results in self.results['llm_results'].values():
            if task_name in model_results and dataset_name in model_results[task_name]:
                return True
        return False
    
    async def _load_examples_for_baseline(self, task_name: str, dataset_name: str, sample_size: int) -> List[Any]:
        """Load examples for baseline evaluation."""
        try:
            examples = quick_load_dataset(dataset_name, max_examples=sample_size)
            
            if not examples:
                self.logger.warning(f"No examples loaded for {dataset_name}")
                return []
            
            print(f"   ✅ Loaded {len(examples)} examples for baseline evaluation")
            return examples
            
        except Exception as e:
            self.logger.error(f"Error loading examples for {dataset_name}: {e}")
            return []
    
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
                # Extract data from example
                if hasattr(example, 'source'):
                    source = example.source
                    summary = example.summary
                    summaries = getattr(example, 'summaries', [])
                    example_id = example.example_id
                else:
                    source = example.get('source', example.get('document', ''))
                    summary = example.get('summary', '')
                    summaries = example.get('summaries', [])
                    example_id = example.get('example_id', str(i))
                
                # Call appropriate baseline method
                if task_name == 'entailment_inference':
                    result = baseline.evaluate_entailment_inference(
                        source=source,
                        summary=summary,
                        example_id=example_id
                    )
                elif task_name == 'consistency_rating':
                    result = baseline.evaluate_consistency_rating(
                        source=source,
                        summary=summary,
                        example_id=example_id
                    )
                elif task_name == 'summary_ranking':
                    result = baseline.evaluate_summary_ranking(
                        source=source,
                        summaries=summaries,
                        example_id=example_id
                    )
                else:
                    result = None
                
                # Extract prediction value
                if hasattr(result, 'prediction'):
                    prediction = result.prediction
                else:
                    prediction = result
                
                predictions.append(prediction)
                
            except Exception as e:
                self.logger.warning(f"Baseline {baseline_name} failed on example {i}: {e}")
                predictions.append(None)
        
        return predictions
    
    def _serialize_predictions(self, predictions: List[Any]) -> List[Any]:
        """Convert predictions to JSON-serializable format."""
        serialized = []
        for pred in predictions:
            if hasattr(pred, '__dict__'):
                serialized.append(pred.__dict__)
            elif isinstance(pred, (np.integer, np.floating)):
                serialized.append(float(pred))
            elif isinstance(pred, np.ndarray):
                serialized.append(pred.tolist())
            else:
                serialized.append(pred)
        return serialized
    
    def _analyze_prediction_distribution(self, predictions: List[Any], task_name: str) -> Dict[str, Any]:
        """Analyze the distribution of predictions for better insights."""
        valid_preds = [p for p in predictions if p is not None]
        
        if not valid_preds:
            return {'error': 'No valid predictions'}
        
        # Convert to numerical values
        numerical_preds = []
        for pred in valid_preds:
            if isinstance(pred, (int, float)):
                numerical_preds.append(float(pred))
            elif hasattr(pred, 'prediction'):
                numerical_preds.append(float(pred.prediction))
            else:
                try:
                    numerical_preds.append(float(pred))
                except:
                    pass
        
        if not numerical_preds:
            return {'error': 'Could not convert predictions to numerical values'}
        
        distribution = {
            'mean': np.mean(numerical_preds),
            'std': np.std(numerical_preds),
            'min': np.min(numerical_preds),
            'max': np.max(numerical_preds),
            'median': np.median(numerical_preds),
            'unique_values': len(np.unique(numerical_preds)),
            'total_predictions': len(numerical_preds)
        }
        
        # Task-specific analysis
        if task_name == 'entailment_inference':
            binary_preds = [1 if p > 0.5 else 0 for p in numerical_preds]
            distribution['positive_rate'] = sum(binary_preds) / len(binary_preds)
            distribution['negative_rate'] = 1 - distribution['positive_rate']
        
        return distribution
    
    def _calculate_baseline_performance(
        self,
        predictions: List[Any],
        examples: List[Any],
        task_name: str,
        baseline_name: str
    ) -> Dict[str, Any]:
        """Calculate comprehensive performance metrics for baseline predictions."""
        try:
            if task_name == 'entailment_inference':
                return self._calculate_entailment_performance_enhanced(predictions, examples, baseline_name)
            elif task_name == 'consistency_rating':
                return self._calculate_consistency_performance_enhanced(predictions, examples, baseline_name)
            elif task_name == 'summary_ranking':
                return self._calculate_ranking_performance_enhanced(predictions, examples, baseline_name)
            else:
                return {'primary_metric': 0.0, 'error': f'Unknown task: {task_name}'}
        except Exception as e:
            self.logger.error(f"Error calculating performance for {baseline_name} on {task_name}: {e}")
            return {'primary_metric': 0.0, 'error': str(e)}
    
    def _calculate_entailment_performance_enhanced(self, predictions: List[Any], examples: List[Any], baseline_name: str) -> Dict[str, Any]:
        """Calculate enhanced entailment inference performance metrics."""
        binary_predictions = []
        ground_truth = []
        
        for pred, example in zip(predictions, examples):
            if pred is None:
                continue
            
            # Get human label
            if hasattr(example, 'human_label'):
                human_label = example.human_label
            elif isinstance(example, dict):
                human_label = example.get('human_label', 0)
            else:
                continue
            
            # Convert prediction to binary based on baseline type
            if baseline_name == 'factcc':
                # FactCC: 1 = inconsistent (contradiction), 0 = consistent (entailment)
                binary_pred = int(pred) if isinstance(pred, (int, float)) else 0
            elif baseline_name == 'bertscore':
                # BERTScore: Higher score = more similar = entailment
                # Use adaptive threshold based on score distribution
                threshold = 0.85  # Typical threshold for BERTScore
                binary_pred = 1 if float(pred) > threshold else 0
            elif baseline_name == 'rouge':
                # ROUGE: Higher score = more overlap = entailment
                threshold = 0.5
                binary_pred = 1 if float(pred) > threshold else 0
            else:
                binary_pred = 1 if float(pred) > 0.5 else 0
            
            # Convert human label to binary
            gt = 1 if human_label > 0 else 0
            
            binary_predictions.append(binary_pred)
            ground_truth.append(gt)
        
        if not binary_predictions:
            return {'primary_metric': 0.0, 'accuracy': 0.0, 'error': 'No valid predictions'}
        
        # Calculate comprehensive metrics
        accuracy = accuracy_score(ground_truth, binary_predictions)
        
        # Handle edge cases for precision/recall
        try:
            precision = precision_score(ground_truth, binary_predictions, zero_division=0)
            recall = recall_score(ground_truth, binary_predictions, zero_division=0)
            f1 = f1_score(ground_truth, binary_predictions, zero_division=0)
            kappa = cohen_kappa_score(ground_truth, binary_predictions)
        except:
            precision = recall = f1 = kappa = 0.0
        
        # Calculate confusion matrix elements
        tp = sum(p == 1 and gt == 1 for p, gt in zip(binary_predictions, ground_truth))
        tn = sum(p == 0 and gt == 0 for p, gt in zip(binary_predictions, ground_truth))
        fp = sum(p == 1 and gt == 0 for p, gt in zip(binary_predictions, ground_truth))
        fn = sum(p == 0 and gt == 1 for p, gt in zip(binary_predictions, ground_truth))
        
        return {
            'primary_metric': accuracy,
            'accuracy': accuracy,
            'precision': precision,
            'recall': recall,
            'f1_score': f1,
            'cohen_kappa': kappa,
            'confusion_matrix': {
                'tp': tp, 'tn': tn, 'fp': fp, 'fn': fn
            },
            'total_examples': len(binary_predictions),
            'predictions_sample': binary_predictions[:5],
            'ground_truth_sample': ground_truth[:5]
        }
    
    def _calculate_consistency_performance_enhanced(self, predictions: List[Any], examples: List[Any], baseline_name: str) -> Dict[str, Any]:
        """Calculate enhanced consistency rating performance metrics."""
        numerical_predictions = []
        ground_truth = []
        
        for pred, example in zip(predictions, examples):
            if pred is None:
                continue
            
            if hasattr(example, 'human_label'):
                human_rating = example.human_label
            elif isinstance(example, dict):
                human_rating = example.get('human_rating', example.get('human_label', 0))
            else:
                continue
            
            # Convert to numerical and normalize to 0-1 scale
            if baseline_name == 'bertscore':
                # BERTScore is already in 0-1 range
                num_pred = float(pred)
            elif baseline_name == 'rouge':
                # ROUGE scores are typically 0-1
                num_pred = float(pred)
            else:
                # Normalize other scores to 0-1
                num_pred = min(1.0, max(0.0, float(pred)))
            
            numerical_predictions.append(num_pred)
            ground_truth.append(float(human_rating))
        
        if len(numerical_predictions) < 2:
            return {
                'primary_metric': 0.0,
                'insufficient_data': True,
                'n_samples': len(numerical_predictions),
                'error': 'Need at least 2 samples for correlation'
            }
        
        # Calculate metrics with small sample handling
        try:
            if np.std(numerical_predictions) == 0 or np.std(ground_truth) == 0:
                # If no variance, use accuracy-based metric instead
                threshold = 0.5
                binary_preds = [1 if p > threshold else 0 for p in numerical_predictions]
                binary_gt = [1 if g > threshold else 0 for g in ground_truth]
                accuracy = sum(p == g for p, g in zip(binary_preds, binary_gt)) / len(binary_preds)
                
                return {
                    'primary_metric': accuracy,
                    'accuracy_based': True,
                    'pearson_correlation': 0.0,
                    'spearman_correlation': 0.0,
                    'mae': np.mean(np.abs(np.array(numerical_predictions) - np.array(ground_truth))),
                    'rmse': np.sqrt(np.mean((np.array(numerical_predictions) - np.array(ground_truth)) ** 2)),
                    'total_examples': len(numerical_predictions),
                    'zero_variance': True,
                    'prediction_mean': np.mean(numerical_predictions),
                    'ground_truth_mean': np.mean(ground_truth)
                }
            
            # Calculate correlations
            pearson_corr, p_value = pearsonr(numerical_predictions, ground_truth)
            spearman_corr, _ = spearmanr(numerical_predictions, ground_truth)
            
            if np.isnan(pearson_corr):
                pearson_corr = 0.0
            if np.isnan(spearman_corr):
                spearman_corr = 0.0
            
            mae = np.mean(np.abs(np.array(numerical_predictions) - np.array(ground_truth)))
            rmse = np.sqrt(np.mean((np.array(numerical_predictions) - np.array(ground_truth)) ** 2))
            
        except Exception as e:
            self.logger.warning(f"Error calculating correlations: {e}")
            pearson_corr = spearman_corr = 0.0
            mae = np.mean(np.abs(np.array(numerical_predictions) - np.array(ground_truth)))
            rmse = np.sqrt(np.mean((np.array(numerical_predictions) - np.array(ground_truth)) ** 2))
            p_value = 1.0
        
        # Use absolute correlation as primary metric, but fall back to 1-MAE if correlation is 0
        primary_metric = abs(pearson_corr) if pearson_corr != 0 else max(0, 1 - mae)
        
        return {
            'primary_metric': primary_metric,
            'pearson_correlation': pearson_corr,
            'spearman_correlation': spearman_corr,
            'mae': mae,
            'rmse': rmse,
            'p_value': p_value,
            'total_examples': len(numerical_predictions),
            'prediction_std': np.std(numerical_predictions),
            'ground_truth_std': np.std(ground_truth)
        }
    
    def _calculate_ranking_performance_enhanced(self, predictions: List[Any], examples: List[Any], baseline_name: str) -> Dict[str, Any]:
        """Calculate enhanced summary ranking performance metrics."""
        correlations = []
        
        for pred, example in zip(predictions, examples):
            if pred is None:
                continue
            
            # Get ground truth ranking
            if hasattr(example, 'ground_truth_ranking'):
                gt_ranking = example.ground_truth_ranking
            else:
                # Generate default ranking
                gt_ranking = list(range(len(pred))) if isinstance(pred, list) else [0]
            
            # Calculate correlation
            if len(pred) > 1 and len(gt_ranking) > 1:
                try:
                    tau, _ = kendalltau(pred, gt_ranking)
                    if not np.isnan(tau):
                        correlations.append(tau)
                    
                    rho, _ = spearmanr(pred, gt_ranking)
                    if not np.isnan(rho):
                        correlations.append(rho)
                except:
                    pass
        
        if not correlations:
            return {
                'primary_metric': 0.0,
                'avg_correlation': 0.0,
                'total_examples': 0,
                'error': 'No valid ranking pairs'
            }
        
        return {
            'primary_metric': np.mean(correlations),
            'avg_correlation': np.mean(correlations),
            'kendall_tau': np.mean([c for c in correlations[::2]]) if len(correlations) > 0 else 0,
            'spearman_rho': np.mean([c for c in correlations[1::2]]) if len(correlations) > 1 else 0,
            'total_examples': len(correlations) // 2
        }
    
    async def _compute_correlation_analysis(self):
        """Compute comprehensive correlation analysis between multiple LLMs and baselines."""
        print(f"\n📊 Computing Correlation Analysis")
        print("-" * 40)
        
        
        correlation_analysis = {
            'pearson_correlations': {},
            'spearman_correlations': {},
            'agreement_metrics': {},
            'correlation_summary': {},
            'method_rankings': {}
        }
        
        # Compute correlations for each model-baseline-task-dataset combination
        for baseline_name, baseline_results in self.results['baseline_results'].items():
            correlation_analysis['pearson_correlations'][baseline_name] = {}
            correlation_analysis['spearman_correlations'][baseline_name] = {}
            correlation_analysis['agreement_metrics'][baseline_name] = {}
            
            for task_name in baseline_results.keys():
                correlation_analysis['pearson_correlations'][baseline_name][task_name] = {}
                correlation_analysis['spearman_correlations'][baseline_name][task_name] = {}
                correlation_analysis['agreement_metrics'][baseline_name][task_name] = {}
                
                for dataset_name in baseline_results[task_name].keys():
                    if 'error' in baseline_results[task_name][dataset_name]:
                        continue
                    
                    for model_name, model_results in self.results['llm_results'].items():
                        if (task_name not in model_results or 
                            dataset_name not in model_results[task_name]):
                            continue
                        
                        model_key = f"{model_name}_{dataset_name}"
                        
                        model_preds = model_results[task_name][dataset_name].get('predictions', [])
                        baseline_preds = baseline_results[task_name][dataset_name].get('predictions', [])
                        
                        if not model_preds or not baseline_preds:
                            continue
                        
                        model_scores = self._extract_numerical_predictions(model_preds, task_name)
                        baseline_scores = self._extract_numerical_predictions(baseline_preds, task_name)
                        
                        # Ensure same length
                        min_length = min(len(model_scores), len(baseline_scores))
                        if min_length < 2:
                            # With very small samples, use agreement metrics instead
                            correlation_analysis['pearson_correlations'][baseline_name][task_name][model_key] = {
                                'correlation': 0.0,
                                'p_value': 1.0,
                                'n_samples': min_length,
                                'error': 'Insufficient samples for correlation',
                                'model_name': model_name
                            }
                            continue
                        
                        model_scores = model_scores[:min_length]
                        baseline_scores = baseline_scores[:min_length]
                        
                        # Calculate correlations with small sample handling
                        try:
                            if np.std(model_scores) < 1e-10 or np.std(baseline_scores) < 1e-10:
                                # Use agreement-based metric instead
                                agreement = sum(abs(m - b) < 0.1 for m, b in zip(model_scores, baseline_scores)) / min_length
                                
                                correlation_analysis['pearson_correlations'][baseline_name][task_name][model_key] = {
                                    'correlation': agreement,
                                    'p_value': 0.05 if agreement > 0.5 else 0.5,
                                    'n_samples': min_length,
                                    'model_name': model_name,
                                    'agreement_based': True
                                }
                                
                                # Compute agreement metrics for low-variance case
                                agreement_metrics = self._compute_agreement_metrics(model_scores, baseline_scores, task_name)
                                correlation_analysis['agreement_metrics'][baseline_name][task_name][model_key] = agreement_metrics
                            else:
                                pearson_corr, pearson_p = pearsonr(model_scores, baseline_scores)
                                spearman_corr, spearman_p = spearmanr(model_scores, baseline_scores)
                                
                                if np.isnan(pearson_corr):
                                    pearson_corr = 0.0
                                if np.isnan(spearman_corr):
                                    spearman_corr = 0.0
                                
                                correlation_analysis['pearson_correlations'][baseline_name][task_name][model_key] = {
                                    'correlation': pearson_corr,
                                    'p_value': pearson_p,
                                    'n_samples': min_length,
                                    'model_name': model_name
                                }
                                
                                correlation_analysis['spearman_correlations'][baseline_name][task_name][model_key] = {
                                    'correlation': spearman_corr,
                                    'p_value': spearman_p,
                                    'n_samples': min_length,
                                    'model_name': model_name
                                }
                                
                                # Compute agreement metrics
                                agreement_metrics = self._compute_agreement_metrics(model_scores, baseline_scores, task_name)
                                correlation_analysis['agreement_metrics'][baseline_name][task_name][model_key] = agreement_metrics
                            
                        except Exception as e:
                            correlation_analysis['pearson_correlations'][baseline_name][task_name][model_key] = {
                                'correlation': 0.0,
                                'p_value': 1.0,
                                'n_samples': min_length,
                                'error': str(e),
                                'model_name': model_name
                            }
        
        # Generate correlation summary
        correlation_analysis['correlation_summary'] = self._generate_correlation_summary(correlation_analysis)
        
        # Generate method rankings
        correlation_analysis['method_rankings'] = self._generate_method_rankings(correlation_analysis)
        
        self.results['correlation_analysis'] = correlation_analysis
        
        corr_file = self.output_dir / "correlations" / "correlation_analysis.json"
        with open(corr_file, 'w') as f:
            json.dump(correlation_analysis, f, indent=2, default=str)
        
        print(f"✅ Correlation analysis completed")
    
    def _extract_numerical_predictions(self, predictions: List[Any], task_name: str) -> List[float]:
        """Extract numerical values from predictions for correlation analysis."""
        numerical_predictions = []
        
        for pred in predictions:
            try:
                value = 0.0
                
                if isinstance(pred, dict):
                    # Extract from dictionary
                    if 'prediction' in pred:
                        value = pred['prediction']
                    elif 'score' in pred:
                        value = pred['score']
                    elif 'confidence' in pred:
                        value = pred['confidence']
                elif hasattr(pred, 'prediction'):
                    value = pred.prediction
                elif isinstance(pred, (int, float)):
                    value = float(pred)
                else:
                    # Try to convert directly
                    try:
                        value = float(pred)
                    except:
                        value = 0.0
                
                # Normalize based on task
                if task_name == 'entailment_inference':
                    # Binary task: convert to 0 or 1
                    if isinstance(value, str):
                        value = 1.0 if value.upper() in ['ENTAILMENT', 'TRUE', '1'] else 0.0
                    else:
                        value = 1.0 if float(value) > 0.5 else 0.0
                elif task_name == 'consistency_rating':
                    # Rating task: ensure in valid range
                    value = float(value)
                    # Normalize to 0-1 if needed
                    if value > 1.0:
                        value = value / 100.0  # Assume percentage
                else:
                    value = float(value)
                
                numerical_predictions.append(value)
                
            except Exception as e:
                self.logger.debug(f"Error extracting prediction: {e}")
                numerical_predictions.append(0.0)
        
        return numerical_predictions
    
    def _compute_agreement_metrics(self, model_scores: List[float], baseline_scores: List[float], task_name: str) -> Dict[str, Any]:
        """Compute agreement metrics between model and baseline predictions."""
        if len(model_scores) != len(baseline_scores) or len(model_scores) == 0:
            return {
                'percentage_agreement': 0.0,
                'mean_absolute_error': 1.0,
                'exact_matches': 0,
                'near_matches': 0,
                'n_samples': 0,
                'error': 'Mismatched or empty arrays'
            }
        
        n_samples = len(model_scores)
        
        # Percentage agreement (within threshold)
        threshold = 0.1 if task_name in ['consistency_rating'] else 0.01
        exact_matches = sum(1 for m, b in zip(model_scores, baseline_scores) if abs(m - b) < threshold)
        
        # Near matches (within larger threshold)
        near_threshold = 0.2 if task_name in ['consistency_rating'] else 0.1
        near_matches = sum(1 for m, b in zip(model_scores, baseline_scores) if abs(m - b) < near_threshold)
        
        percentage_agreement = exact_matches / n_samples
        near_agreement = near_matches / n_samples
        
        # Mean absolute error
        mae = np.mean([abs(m - b) for m, b in zip(model_scores, baseline_scores)])
        
        # Binary agreement for classification tasks
        binary_agreement = None
        if task_name == 'entailment_inference':
            # Convert to binary predictions
            model_binary = [1 if score > 0.5 else 0 for score in model_scores]
            baseline_binary = [1 if score > 0.5 else 0 for score in baseline_scores]
            binary_matches = sum(1 for m, b in zip(model_binary, baseline_binary) if m == b)
            binary_agreement = binary_matches / n_samples
        
        return {
            'percentage_agreement': percentage_agreement,
            'near_agreement': near_agreement,
            'mean_absolute_error': mae,
            'exact_matches': exact_matches,
            'near_matches': near_matches,
            'binary_agreement': binary_agreement,
            'n_samples': n_samples,
            'task_name': task_name
        }
    
    def _generate_correlation_summary(self, correlation_analysis: Dict[str, Any]) -> Dict[str, Any]:
        """Generate summary statistics for correlation analysis."""
        summary = {
            'baseline_average_correlations': {},
            'model_average_correlations': {},
            'best_correlating_baseline': None,
            'best_correlation': 0.0,
            'overall_statistics': {}
        }
        
        # Calculate average correlations by baseline
        for baseline_name, baseline_corr in correlation_analysis['pearson_correlations'].items():
            correlations = []
            agreement_scores = []
            
            for task_corr in baseline_corr.values():
                for dataset_corr in task_corr.values():
                    if isinstance(dataset_corr, dict):
                        if dataset_corr.get('agreement_based'):
                            # Use agreement score
                            agreement_scores.append(dataset_corr['correlation'])
                        elif 'correlation' in dataset_corr and 'error' not in dataset_corr:
                            corr_val = dataset_corr['correlation']
                            if not np.isnan(corr_val):
                                correlations.append(abs(corr_val))
            
            # Combine correlations and agreement scores
            all_scores = correlations + agreement_scores
            if all_scores:
                summary['baseline_average_correlations'][baseline_name] = np.mean(all_scores)
        
        # Find best correlating baseline
        if summary['baseline_average_correlations']:
            best_baseline = max(summary['baseline_average_correlations'].items(), key=lambda x: x[1])
            summary['best_correlating_baseline'] = best_baseline[0]
            summary['best_correlation'] = best_baseline[1]
        
        # Overall statistics
        all_correlations = []
        for baseline_corr in correlation_analysis['pearson_correlations'].values():
            for task_corr in baseline_corr.values():
                for dataset_corr in task_corr.values():
                    if isinstance(dataset_corr, dict) and 'correlation' in dataset_corr:
                        corr_val = dataset_corr['correlation']
                        if not np.isnan(corr_val) and 'error' not in dataset_corr:
                            all_correlations.append(corr_val)
        
        if all_correlations:
            summary['overall_statistics'] = {
                'mean': np.mean(all_correlations),
                'std': np.std(all_correlations),
                'median': np.median(all_correlations),
                'min': np.min(all_correlations),
                'max': np.max(all_correlations),
                'valid_correlations': len(all_correlations)
            }
        else:
            summary['overall_statistics'] = {
                'mean': 0.0,
                'std': 0.0,
                'median': 0.0,
                'min': 0.0,
                'max': 0.0,
                'valid_correlations': 0
            }
        
        return summary
    
    def _generate_method_rankings(self, correlation_analysis: Dict[str, Any]) -> Dict[str, Any]:
        """Generate rankings of methods based on various criteria."""
        rankings = {
            'by_correlation_strength': [],
            'by_model_performance': [],
            'by_task_performance': {}
        }
        
        # Rank baselines by correlation strength
        baseline_avg = correlation_analysis['correlation_summary'].get('baseline_average_correlations', {})
        if baseline_avg:
            rankings['by_correlation_strength'] = sorted(
                [{'baseline': k, 'avg_correlation': v} for k, v in baseline_avg.items()],
                key=lambda x: x['avg_correlation'],
                reverse=True
            )
        
        return rankings
    
    async def _analyze_performance_comparison(self):
        """Analyze performance comparison between multiple LLMs and baselines."""
        print(f"\n📈 Analyzing Performance Comparison")
        print("-" * 40)
        
        performance_comparison = {
            'task_performance': {},
            'baseline_performance': {},
            'model_performance': {},
            'relative_performance': {},
            'performance_insights': {}
        }
        
        # Analyze task performance for each model
        for model_name, model_results in self.results['llm_results'].items():
            if model_name not in performance_comparison['model_performance']:
                performance_comparison['model_performance'][model_name] = {}
            
            for task_name, task_results in model_results.items():
                if task_name not in performance_comparison['task_performance']:
                    performance_comparison['task_performance'][task_name] = {}
                
                performances = []
                for dataset_name, dataset_results in task_results.items():
                    # Extract performance metric
                    if 'performance_metrics' in dataset_results:
                        metric = dataset_results['performance_metrics'].get('primary_metric', 0)
                    elif 'comprehensive_metrics' in dataset_results:
                        metric = dataset_results['comprehensive_metrics'].get('primary_metric', 0)
                    else:
                        metric = 0
                    
                    performances.append(metric)
                
                if performances:
                    performance_comparison['task_performance'][task_name][model_name] = {
                        'mean_performance': np.mean(performances),
                        'std_performance': np.std(performances) if len(performances) > 1 else 0,
                        'performances': performances
                    }
                    
                    performance_comparison['model_performance'][model_name][task_name] = {
                        'mean_performance': np.mean(performances),
                        'std_performance': np.std(performances) if len(performances) > 1 else 0
                    }
        
        # Analyze baseline performance
        for baseline_name, baseline_results in self.results['baseline_results'].items():
            if baseline_name not in performance_comparison['baseline_performance']:
                performance_comparison['baseline_performance'][baseline_name] = {}
            
            for task_name, task_results in baseline_results.items():
                performances = []
                for dataset_name, dataset_results in task_results.items():
                    if 'performance_metrics' in dataset_results:
                        metric = dataset_results['performance_metrics'].get('primary_metric', 0)
                        performances.append(metric)
                
                if performances:
                    performance_comparison['baseline_performance'][baseline_name][task_name] = {
                        'mean_performance': np.mean(performances),
                        'std_performance': np.std(performances) if len(performances) > 1 else 0,
                        'performances': performances
                    }
        
        # Generate performance insights
        insights = {
            'best_model_per_task': {},
            'best_baseline_per_task': {},
            'model_baseline_alignment': {}
        }
        
        # Find best performers
        for task_name in performance_comparison['task_performance'].keys():
            # Best model
            if performance_comparison['task_performance'][task_name]:
                best_model = max(
                    performance_comparison['task_performance'][task_name].items(),
                    key=lambda x: x[1]['mean_performance']
                )
                insights['best_model_per_task'][task_name] = {
                    'model': best_model[0],
                    'performance': best_model[1]['mean_performance']
                }
            
            # Best baseline
            task_baselines = {
                k: v[task_name] for k, v in performance_comparison['baseline_performance'].items()
                if task_name in v
            }
            if task_baselines:
                best_baseline = max(
                    task_baselines.items(),
                    key=lambda x: x[1]['mean_performance']
                )
                insights['best_baseline_per_task'][task_name] = {
                    'baseline': best_baseline[0],
                    'performance': best_baseline[1]['mean_performance']
                }
        
        performance_comparison['performance_insights'] = insights
        self.results['performance_comparison'] = performance_comparison
        
        print(f"✅ Performance comparison completed")
    
    async def _perform_statistical_analysis(self):
        """Perform comprehensive statistical significance testing."""
        print(f"\n📊 Performing Statistical Analysis")
        print("-" * 40)
        
        statistical_analysis = {
            'correlation_significance': {},
            'performance_significance': {},
            'effect_sizes': {},
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
                    if isinstance(dataset_corr, dict):
                        p_value = dataset_corr.get('p_value', 1.0)
                        correlation = dataset_corr.get('correlation', 0)
                        n_samples = dataset_corr.get('n_samples', 0)
                        
                        is_significant = p_value < 0.05 and not np.isnan(correlation) and 'error' not in dataset_corr
                        
                        # Calculate effect size
                        effect_size = self._calculate_correlation_effect_size(correlation)
                        
                        statistical_analysis['correlation_significance'][baseline_name][task_name][dataset_name] = {
                            'correlation': correlation,
                            'p_value': p_value,
                            'is_significant': is_significant,
                            'n_samples': n_samples,
                            'effect_size': effect_size,
                            'model_name': dataset_corr.get('model_name', 'unknown'),
                            'agreement_based': dataset_corr.get('agreement_based', False)
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
        
        print(f"✅ Statistical analysis completed")
    
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
            return "No correlations computed - insufficient data"
        
        rate = significant / total
        
        # Adjust interpretation for small samples
        sample_size = self.experiment_params.get('sample_size', 0)
        if sample_size < 10:
            return f"Limited analysis with {sample_size} samples - results are preliminary"
        elif rate >= 0.8:
            return "Strong evidence of correlation between models and baselines"
        elif rate >= 0.6:
            return "Moderate evidence of correlation between models and baselines"
        elif rate >= 0.4:
            return "Limited evidence of correlation between models and baselines"
        else:
            return "Weak evidence of correlation - models show novel patterns"
    
    async def _generate_summary_statistics(self):
        """Generate comprehensive summary statistics."""
        print(f"\n📊 Generating Summary Statistics")
        print("-" * 40)
        
        summary = {
            'models_evaluated': list(self.results['llm_results'].keys()),
            'baselines_evaluated': list(self.results['baseline_results'].keys()),
            'tasks_evaluated': set(),
            'datasets_evaluated': set(),
            'total_experiments': 0,
            'total_cost': self.results['cost_analysis'].get('llm_cost', 0),
            'sample_size': self.experiment_params.get('sample_size', 0),
            'key_findings': []
        }
        
        # Collect tasks and datasets
        for model_results in self.results['llm_results'].values():
            summary['tasks_evaluated'].update(model_results.keys())
            for task_results in model_results.values():
                summary['datasets_evaluated'].update(task_results.keys())
                summary['total_experiments'] += len(task_results)
        
        summary['tasks_evaluated'] = list(summary['tasks_evaluated'])
        summary['datasets_evaluated'] = list(summary['datasets_evaluated'])
        
        # Generate key findings
        corr_summary = self.results['correlation_analysis'].get('correlation_summary', {})
        if corr_summary.get('best_correlating_baseline'):
            summary['key_findings'].append(
                f"Best correlating baseline: {corr_summary['best_correlating_baseline']} "
                f"(score={corr_summary['best_correlation']:.3f})"
            )
        
        perf_insights = self.results['performance_comparison'].get('performance_insights', {})
        if perf_insights.get('best_model_per_task'):
            for task, info in perf_insights['best_model_per_task'].items():
                summary['key_findings'].append(
                    f"Best model for {task}: {info['model']} "
                    f"(performance={info['performance']:.3f})"
                )
        
        self.results['summary_statistics'] = summary
        
        print(f"✅ Summary statistics generated")
    
    async def _generate_comparison_visualizations(self):
        """Generate comprehensive comparison visualizations matching original SOTA comparison."""
        print(f"\n📊 Generating Visualizations")
        print("-" * 40)
        
        viz_dir = self.output_dir / "figures"
        viz_dir.mkdir(parents=True, exist_ok=True)
        
        try:
            

            self._create_correlation_heatmap(viz_dir)
            self._create_baseline_performance_chart(viz_dir)
            self._create_correlation_scatter_plots(viz_dir)
            self._create_method_ranking_chart(viz_dir)
            
            self._create_processing_time_comparison(viz_dir)
            self._create_agreement_analysis_charts(viz_dir)
            self._create_statistical_significance_chart(viz_dir)
            self._create_model_performance_comparison(viz_dir)
            self._create_dataset_comparison_charts(viz_dir)
            self._create_correlation_matrix_3d(viz_dir)
            self._create_performance_radar_chart(viz_dir)
            
            self._create_correlation_stability_analysis(viz_dir)
            self._create_performance_evolution_timeline(viz_dir)
            self._create_baseline_robustness_analysis(viz_dir)
            self._create_task_difficulty_heatmap(viz_dir)
            
            # Task-specific visualizations
            self._create_task_specific_performance_charts(viz_dir)
            
            # Multi-model specific visualizations
            self._create_model_comparison_radar(viz_dir)
            self._create_model_performance_heatmap(viz_dir)
            
            print(f"✅ Generated {len(self.results.get('visualizations', {}))} visualizations in: {viz_dir}")
            
        except Exception as e:
            print(f"⚠️  Visualization generation failed: {e}")
            self.logger.warning(f"Visualization generation failed: {e}")

    def _create_correlation_scatter_plots(self, viz_dir: Path):
        """Create correlation scatter plots for best performing baselines."""
        correlation_summary = self.results['correlation_analysis'].get('correlation_summary', {})
        best_baseline = correlation_summary.get('best_correlating_baseline')
        
        if not best_baseline or best_baseline not in self.results['baseline_results']:
            print("⚠️ No best baseline found for correlation scatter plots")
            return
        
        # Create scatter plots for each model
        models = list(self.results['llm_results'].keys())
        n_models = len(models)
        
        if n_models == 0:
            print("⚠️ No models found for correlation scatter plots")
            return
        
        # Calculate grid dimensions based on number of models
        if n_models == 1:
            rows, cols = 1, 1
        elif n_models == 2:
            rows, cols = 1, 2
        elif n_models <= 4:
            rows, cols = 2, 2
        else:
            # For more than 4 models, use a larger grid
            rows = (n_models + 1) // 2
            cols = 2
        
        fig = make_subplots(
            rows=rows, 
            cols=cols,
            subplot_titles=[f'{model} vs {best_baseline}' for model in models]
        )
        
        # Color mapping for different task-dataset combinations
        task_dataset_colors = self.COLOR_SCHEMES['task_dataset_combinations']
        
        # Create one scatter plot per model
        for idx, model_name in enumerate(models):
            if model_name not in self.results['llm_results']:
                continue
            
            # Calculate subplot position
            row = (idx // cols) + 1
            col = (idx % cols) + 1
            
            # Collect data points for this model by task-dataset combination
            for task_name in self.results['baseline_results'][best_baseline].keys():
                for dataset_name in self.results['baseline_results'][best_baseline][task_name].keys():
                    # Get model predictions
                    if (task_name in self.results['llm_results'][model_name] and 
                        dataset_name in self.results['llm_results'][model_name][task_name]):
                        
                        model_preds = self.results['llm_results'][model_name][task_name][dataset_name].get('predictions', [])
                        baseline_preds = self.results['baseline_results'][best_baseline][task_name][dataset_name].get('predictions', [])
                        
                        if model_preds and baseline_preds:
                            # Extract numerical values
                            model_scores = self._extract_numerical_predictions(model_preds, task_name)
                            baseline_scores = self._extract_numerical_predictions(baseline_preds, task_name)
                            
                            min_length = min(len(model_scores), len(baseline_scores))
                            if min_length > 0:
                                model_scores = model_scores[:min_length]
                                baseline_scores = baseline_scores[:min_length]
                                
                                # Get color for this task-dataset combination
                                task_dataset_key = f"{task_name}_{dataset_name}"
                                color = task_dataset_colors.get(task_dataset_key, '#888888')
                                
                                # Add scatter trace for this specific task-dataset
                                fig.add_trace(
                                    go.Scatter(
                                        x=baseline_scores,
                                        y=model_scores,
                                        mode='markers',
                                        name=f'{task_name}_{dataset_name}',
                                        marker=dict(
                                            color=color,
                                            size=8,
                                            opacity=0.7,
                                            line=dict(width=1, color='white')
                                        ),
                                        text=[f'{task_name}_{dataset_name}'] * len(model_scores),
                                        hovertemplate='%{text}<br>Baseline: %{x:.3f}<br>Model: %{y:.3f}<extra></extra>',
                                        showlegend=(idx == 0)  # Only show legend for first model
                                    ),
                                    row=row, col=col
                                )
        
        # Add perfect correlation line to each subplot
        for i in range(1, rows + 1):
            for j in range(1, cols + 1):
                if (i - 1) * cols + j <= n_models:
                    fig.add_trace(
                        go.Scatter(
                            x=[0, 1],
                            y=[0, 1],
                            mode='lines',
                            line=dict(color='red', dash='dash', width=2),
                            name='Perfect Correlation',
                            showlegend=(i == 1 and j == 1)  # Only show legend once
                        ),
                        row=i, col=j
                    )
        
        fig.update_layout(
            title=f'Correlation Analysis: Models vs {best_baseline}',
            font=dict(size=12),
            paper_bgcolor='white',
            height=800,
            legend=dict(
                orientation="h",
                yanchor="bottom",
                y=1.02,
                xanchor="right",
                x=1
            )
        )
        
        # Update axis labels
        for i in range(1, rows + 1):
            for j in range(1, cols + 1):
                if (i - 1) * cols + j <= n_models:
                    fig.update_xaxes(title_text=f"{best_baseline} Score", row=i, col=j)
                    fig.update_yaxes(title_text="Model Score", row=i, col=j)
        
        fig_path = viz_dir / f"correlation_scatter_{best_baseline}.png"
        fig.write_image(str(fig_path), width=1200, height=800, scale=2)
        self.results['visualizations']['correlation_scatter_plots'] = str(fig_path)
        print(f"✅ Correlation scatter plot saved: {fig_path}")

    def _create_method_ranking_chart(self, viz_dir: Path):
        """Create method ranking visualization."""
        method_rankings = self.results['correlation_analysis'].get('method_rankings', {})
        
        if 'by_correlation_strength' not in method_rankings:
            return
        
        rankings = method_rankings['by_correlation_strength']
        if not rankings:
            return
        
        baselines = [item['baseline'] for item in rankings]
        correlations = [item['avg_correlation'] for item in rankings]
        baseline_colors = [self.get_baseline_color(baseline) for baseline in baselines]
        
        fig = go.Figure()
        
        fig.add_trace(go.Bar(
            y=baselines,
            x=correlations,
            orientation='h',
            marker_color=baseline_colors,
            text=[f'{corr:.3f}' for corr in correlations],
            textposition='auto'
        ))
        
        fig.update_layout(
            title='Baseline Method Ranking by Correlation Strength',
            xaxis_title='Average Absolute Correlation',
            yaxis_title='Baseline Method',
            font=dict(size=12),
            plot_bgcolor='white',
            paper_bgcolor='white'
        )
        
        fig_path = viz_dir / "method_ranking.png"
        fig.write_image(str(fig_path), width=1000, height=600, scale=2)
        self.results['visualizations']['method_ranking'] = str(fig_path)

    def _create_processing_time_comparison(self, viz_dir: Path):
        """Create processing time comparison between models and baselines."""
        fig = make_subplots(
            rows=1, cols=2,
            subplot_titles=['Model Processing Times', 'Baseline Processing Times']
        )
        
        # Model processing times
        model_times = []
        model_labels = []
        model_colors = []
        
        for model_name, model_results in self.results['llm_results'].items():
            for task_name, task_data in model_results.items():
                for dataset_name, dataset_data in task_data.items():
                    if 'processing_time' in dataset_data:
                        model_times.append(dataset_data['processing_time'])
                        model_labels.append(f"{model_name[:10]}_{task_name[:3]}_{dataset_name[:5]}")
                        model_colors.append(self.get_model_color(model_name))
        
        if model_times:
            fig.add_trace(
                go.Bar(
                    x=model_labels,
                    y=model_times,
                    marker_color=model_colors,
                    text=[f'{t:.1f}s' for t in model_times],
                    textposition='auto'
                ),
                row=1, col=1
            )
        
        # Baseline processing times
        baseline_times = []
        baseline_labels = []
        baseline_colors = []
        
        for baseline_name, baseline_data in self.results['baseline_results'].items():
            for task_name, task_data in baseline_data.items():
                for dataset_name, dataset_data in task_data.items():
                    if 'processing_time' in dataset_data:
                        baseline_times.append(dataset_data['processing_time'])
                        baseline_labels.append(f"{baseline_name}_{task_name[:3]}_{dataset_name[:5]}")
                        baseline_colors.append(self.get_baseline_color(baseline_name))
        
        if baseline_times:
            fig.add_trace(
                go.Bar(
                    x=baseline_labels,
                    y=baseline_times,
                    marker_color=baseline_colors,
                    text=[f'{t:.1f}s' for t in baseline_times],
                    textposition='auto'
                ),
                row=1, col=2
            )
        
        fig.update_layout(
            title='Processing Time Comparison',
            font=dict(size=12),
            showlegend=False,
            height=500
        )
        
        fig_path = viz_dir / 'processing_time_comparison.png'
        fig.write_image(str(fig_path), width=1200, height=500, scale=2)
        self.results['visualizations']['processing_time_comparison'] = str(fig_path)

    def _create_agreement_analysis_charts(self, viz_dir: Path):
        """Create comprehensive agreement analysis charts using agreement_metrics."""
        try:
            agreement_metrics = self.results['correlation_analysis'].get('agreement_metrics', {})
            
            if not agreement_metrics:
                print("❌ No agreement_metrics data available for agreement analysis")
                # Fall back to correlation-based analysis
                self._create_agreement_analysis_fallback(viz_dir)
                return
            
            print(f"📊 Agreement analysis data: {len(agreement_metrics)} baselines with agreement metrics")
            
            fig = make_subplots(
                rows=2, cols=2,
                subplot_titles=[
                    'Percentage Agreement by Baseline',
                    'Mean Absolute Error by Baseline', 
                    'Binary Agreement by Task',
                    'Agreement Distribution'
                ],
                specs=[[{"type": "bar"}, {"type": "bar"}],
                       [{"type": "box"}, {"type": "histogram"}]]
            )
            
            baseline_data = {}
            task_data = {}
            all_agreements = []
            
            for baseline_name, baseline_metrics in agreement_metrics.items():
                baseline_agreements = []
                baseline_maes = []
                baseline_binary_agreements = []
                
                for task_name, task_metrics in baseline_metrics.items():
                    if not task_metrics:
                        continue
                        
                    for model_key, metrics in task_metrics.items():
                        if isinstance(metrics, dict) and 'percentage_agreement' in metrics:
                            agreement = metrics['percentage_agreement']
                            mae = metrics.get('mean_absolute_error', 0.0)
                            binary_agreement = metrics.get('binary_agreement')
                            
                            baseline_agreements.append(agreement)
                            baseline_maes.append(mae)
                            all_agreements.append(agreement)
                            
                            if binary_agreement is not None:
                                baseline_binary_agreements.append(binary_agreement)
                                
                                # Store task-specific data
                                if task_name not in task_data:
                                    task_data[task_name] = []
                                task_data[task_name].append(binary_agreement)
                
                if baseline_agreements:
                    baseline_data[baseline_name] = {
                        'mean_agreement': np.mean(baseline_agreements),
                        'mean_mae': np.mean(baseline_maes),
                        'mean_binary_agreement': np.mean(baseline_binary_agreements) if baseline_binary_agreements else 0.0,
                        'agreements': baseline_agreements,
                        'maes': baseline_maes
                    }
            
            if not baseline_data:
                print("❌ No valid agreement data found")
                return
            
            baseline_names = list(baseline_data.keys())
            baseline_colors = [self.get_baseline_color(name) for name in baseline_names]
            
            # Task colors for box plots
            task_colors = {
                'entailment_inference': self.get_task_color('entailment_inference'),
                'summary_ranking': self.get_task_color('summary_ranking'), 
                'consistency_rating': self.get_task_color('consistency_rating')
            }
            
            # 1. Percentage Agreement by Baseline
            baseline_names = list(baseline_data.keys())
            mean_agreements = [baseline_data[name]['mean_agreement'] for name in baseline_names]
            
            fig.add_trace(
                go.Bar(
                    x=baseline_names,
                    y=mean_agreements,
                    name="Agreement %",
                    marker=dict(color=baseline_colors),
                    text=[f"{ag:.1%}" for ag in mean_agreements],
                    textposition='auto',
                    showlegend=False
                ),
                row=1, col=1
            )
            
            # 2. Mean Absolute Error by Baseline
            mean_maes = [baseline_data[name]['mean_mae'] for name in baseline_names]
            
            fig.add_trace(
                go.Bar(
                    x=baseline_names,
                    y=mean_maes,
                    name="MAE",
                    marker=dict(color=baseline_colors),
                    text=[f"{mae:.3f}" for mae in mean_maes],
                    textposition='auto',
                    showlegend=False
                ),
                row=1, col=2
            )
            
            # 3. Binary Agreement by Task
            if task_data:
                for i, (task_name, agreements) in enumerate(task_data.items()):
                    task_color = task_colors.get(task_name, '#888888')
                    fig.add_trace(
                        go.Box(
                            y=agreements,
                            name=task_name.replace('_', ' ').title(),
                            boxpoints='all',
                            jitter=0.3,
                            pointpos=-1.8,
                            marker=dict(color=task_color),
                            line=dict(color=task_color)
                        ),
                        row=2, col=1
                    )
            
            # 4. Agreement Distribution
            if all_agreements:
                fig.add_trace(
                    go.Histogram(
                        x=all_agreements,
                        name="Agreement Distribution",
                        nbinsx=20,
                        marker=dict(color=self.COLOR_SCHEMES['agreement_metrics']['medium_agreement'], opacity=0.7),
                        showlegend=False
                    ),
                    row=2, col=2
                )
            
            fig.update_layout(
                title="Model-Baseline Agreement Analysis",
                height=800,
                font=dict(size=12)
            )
            
            fig.update_xaxes(title_text="Baseline Method", row=1, col=1)
            fig.update_yaxes(title_text="Agreement Percentage", row=1, col=1)
            
            fig.update_xaxes(title_text="Baseline Method", row=1, col=2)
            fig.update_yaxes(title_text="Mean Absolute Error", row=1, col=2)
            
            fig.update_xaxes(title_text="Task", row=2, col=1)
            fig.update_yaxes(title_text="Binary Agreement", row=2, col=1)
            
            fig.update_xaxes(title_text="Agreement Percentage", row=2, col=2)
            fig.update_yaxes(title_text="Frequency", row=2, col=2)
            
            fig_path = viz_dir / "agreement_analysis.png"
            fig.write_image(str(fig_path), width=1200, height=800, scale=2)
            self.results['visualizations']['agreement_analysis'] = str(fig_path)
            
            print(f"✅ Agreement analysis saved: {fig_path}")
            
        except Exception as e:
            print(f"❌ Error creating agreement analysis: {e}")
            self.logger.error(f"Agreement analysis error: {e}")
            import traceback
            traceback.print_exc()
    
    def _create_agreement_analysis_fallback(self, viz_dir: Path):
        """Fallback agreement analysis using correlation data when agreement_metrics is empty."""
        # Get data from correlation analysis results instead of missing agreement_metrics
        pearson_correlations = self.results['correlation_analysis'].get('pearson_correlations', {})
        
        if not pearson_correlations:
            print("⚠️ No correlation data available for agreement analysis")
            return
        
        fig = make_subplots(
            rows=2, cols=2,
            subplot_titles=['Correlation Strength', 'Statistical Significance', 'Task Coverage', 'Dataset Coverage']
        )
        
        # Collect data from actual correlation results
        baselines = []
        correlations = []
        p_values = []
        task_coverage = []
        dataset_coverage = []
        
        for baseline_name, baseline_corr in pearson_correlations.items():
            baseline_correlations = []
            baseline_p_values = []
            tasks_covered = set()
            datasets_covered = set()
            
            for task_name, task_corr in baseline_corr.items():
                tasks_covered.add(task_name)
                for dataset_key, dataset_corr in task_corr.items():
                    if isinstance(dataset_corr, dict):
                        # Extract dataset name from key
                        dataset_name = dataset_key.split('_')[-1] if '_' in dataset_key else dataset_key
                        datasets_covered.add(dataset_name)
                        
                        corr_val = dataset_corr.get('correlation', 0)
                        p_val = dataset_corr.get('p_value', 1.0)
                        
                        if not np.isnan(corr_val):
                            baseline_correlations.append(abs(corr_val))
                        if not np.isnan(p_val):
                            baseline_p_values.append(p_val)
            
            if baseline_correlations:
                baselines.append(baseline_name)
                correlations.append(np.mean(baseline_correlations))
                p_values.append(np.mean(baseline_p_values) if baseline_p_values else 1.0)
                task_coverage.append(len(tasks_covered))
                dataset_coverage.append(len(datasets_covered))
        
        if baselines:
            bar_colors = [self.get_baseline_color(baseline) for baseline in baselines]
            
            # Correlation strength
            fig.add_trace(go.Bar(
                x=baselines, 
                y=correlations, 
                marker_color=bar_colors, 
                showlegend=False,
                text=[f'{c:.3f}' for c in correlations],
                textposition='auto'
            ), row=1, col=1)
            
            # Statistical significance (inverse of p-value)
            significance_scores = [1 - p for p in p_values]
            fig.add_trace(go.Bar(
                x=baselines, 
                y=significance_scores, 
                marker_color=bar_colors, 
                showlegend=False,
                text=[f'{s:.3f}' for s in significance_scores],
                textposition='auto'
            ), row=1, col=2)
            
            # Task coverage
            fig.add_trace(go.Bar(
                x=baselines, 
                y=task_coverage, 
                marker_color=bar_colors, 
                showlegend=False,
                text=[str(t) for t in task_coverage],
                textposition='auto'
            ), row=2, col=1)
            
            # Dataset coverage
            fig.add_trace(go.Bar(
                x=baselines, 
                y=dataset_coverage, 
                marker_color=bar_colors, 
                showlegend=False,
                text=[str(d) for d in dataset_coverage],
                textposition='auto'
            ), row=2, col=2)
            
            print(f"📊 Agreement analysis data: {len(baselines)} baselines with correlations: {correlations}")
        else:
            print("⚠️ No valid baseline data found for agreement analysis")
        
        fig.update_layout(
            title='Baseline Performance Analysis (Fallback)',
            font=dict(size=12),
            height=800
        )
        
        fig_path = viz_dir / 'agreement_analysis.png'
        fig.write_image(str(fig_path), width=1200, height=800, scale=2)
        self.results['visualizations']['agreement_analysis'] = str(fig_path)

    def _create_dataset_comparison_charts(self, viz_dir: Path):
        """Create dataset-specific comparison charts."""
        datasets = set()
        for model_results in self.results['llm_results'].values():
            for task_data in model_results.values():
                datasets.update(task_data.keys())
        
        datasets = sorted(list(datasets))
        
        if len(datasets) < 1:
            return
        
        fig = make_subplots(
            rows=1, cols=len(datasets),
            subplot_titles=[f'{dataset.upper()}' for dataset in datasets]
        )
        
        for col, dataset in enumerate(datasets, 1):
            baseline_correlations = {}
            
            pearson_correlations = self.results['correlation_analysis']['pearson_correlations']
            for baseline_name, baseline_corr in pearson_correlations.items():
                correlations = []
                for task_corr in baseline_corr.values():
                    for key, dataset_corr in task_corr.items():
                        if dataset in key and isinstance(dataset_corr, dict):
                            corr_val = dataset_corr.get('correlation', 0)
                            if not np.isnan(corr_val):
                                correlations.append(abs(corr_val))
                
                if correlations:
                    baseline_correlations[baseline_name] = np.mean(correlations)
            
            if baseline_correlations:
                baselines = list(baseline_correlations.keys())
                correlations = list(baseline_correlations.values())
                bar_colors = [self.get_baseline_color(baseline) for baseline in baselines]
                
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
            font=dict(size=12),
            height=500
        )
        
        fig_path = viz_dir / 'dataset_comparison.png'
        fig.write_image(str(fig_path), width=1200, height=500, scale=2)
        self.results['visualizations']['dataset_comparison'] = str(fig_path)

    def _create_correlation_matrix_3d(self, viz_dir: Path):
        """Create 3D correlation matrix visualization."""
        pearson_correlations = self.results['correlation_analysis']['pearson_correlations']
        
        x_data = []  # Baselines
        y_data = []  # Model-Task-Dataset combinations
        z_data = []  # Correlations
        
        for baseline_name, baseline_corr in pearson_correlations.items():
            for task_name, task_corr in baseline_corr.items():
                for key, dataset_corr in task_corr.items():
                    if isinstance(dataset_corr, dict) and 'correlation' in dataset_corr:
                        correlation = dataset_corr['correlation']
                        if not np.isnan(correlation):
                            x_data.append(baseline_name)
                            y_data.append(f"{dataset_corr.get('model_name', 'unknown')}_{task_name}_{key.split('_')[-1]}")
                            z_data.append(correlation)
        
        if not x_data:
            return
        
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
            text=[f'{baseline}<br>{task}<br>r={corr:.3f}' 
                for baseline, task, corr in zip(x_data, y_data, z_data)],
            hovertemplate='%{text}<extra></extra>'
        )])
        
        fig.update_layout(
            title='3D Correlation Visualization',
            scene=dict(
                xaxis_title='Baseline',
                yaxis_title='Model-Task-Dataset',
                zaxis_title='Correlation'
            ),
            font=dict(size=12)
        )
        
        fig_path = viz_dir / 'correlation_3d.png'
        fig.write_image(str(fig_path), width=1000, height=800, scale=2)
        self.results['visualizations']['correlation_3d'] = str(fig_path)

    def _create_performance_radar_chart(self, viz_dir: Path):
        """Create radar chart comparing baseline performance across metrics."""
        correlation_summary = self.results['correlation_analysis'].get('correlation_summary', {})
        baseline_avg_correlations = correlation_summary.get('baseline_average_correlations', {})
        
        if not baseline_avg_correlations:
            return
        
        categories = ['Correlation', 'Speed', 'Coverage', 'Consistency']
        
        fig = go.Figure()
        
        colors = {'factcc': "#032E2C", 'bertscore': "#A0D8E5", 'rouge': "#09653A"}
        
        for baseline_name in baseline_avg_correlations.keys():
            # Calculate metrics
            correlation = baseline_avg_correlations.get(baseline_name, 0)
            
            # Speed (inverse of processing time)
            speed = 0.5  # Default
            if baseline_name in self.results['baseline_results']:
                times = []
                for task_data in self.results['baseline_results'][baseline_name].values():
                    for dataset_data in task_data.values():
                        if 'processing_time' in dataset_data:
                            times.append(dataset_data['processing_time'])
                if times:
                    avg_time = np.mean(times)
                    speed = 1 / (1 + avg_time)  # Normalize
            
            # Coverage (how many tasks supported)
            coverage = len(self.results['baseline_results'].get(baseline_name, {})) / 3  # Normalize by max tasks
            
            # Consistency (1 - std of correlations)
            consistency = 0.5  # Default
            
            values = [correlation, speed, coverage, consistency]
            values.append(values[0])  # Close the radar
            categories_closed = categories + [categories[0]]
            
            fig.add_trace(go.Scatterpolar(
                r=values,
                theta=categories_closed,
                fill='toself',
                name=baseline_name.upper(),
                line_color=self.get_baseline_color(baseline_name),
                fillcolor=self.get_baseline_color(baseline_name),
                opacity=0.3
            ))
        
        fig.update_layout(
            polar=dict(
                radialaxis=dict(
                    visible=True,
                    range=[0, 1]
                )
            ),
            title='Baseline Performance Radar Chart',
            font=dict(size=12)
        )
        
        fig_path = viz_dir / 'performance_radar.png'
        fig.write_image(str(fig_path), width=800, height=800, scale=2)
        self.results['visualizations']['performance_radar'] = str(fig_path)

    def _create_correlation_stability_analysis(self, viz_dir: Path):
        """Create correlation stability analysis."""
        pearson_correlations = self.results['correlation_analysis'].get('pearson_correlations', {})
        spearman_correlations = self.results['correlation_analysis'].get('spearman_correlations', {})
        
        if not pearson_correlations or not spearman_correlations:
            return
        
        pearson_values = []
        spearman_values = []
        labels = []
        
        for baseline_name in pearson_correlations.keys():
            if baseline_name not in spearman_correlations:
                continue
            
            for task_name in pearson_correlations[baseline_name].keys():
                if task_name not in spearman_correlations[baseline_name]:
                    continue
                
                for key in pearson_correlations[baseline_name][task_name].keys():
                    if key not in spearman_correlations[baseline_name][task_name]:
                        continue
                    
                    pearson_data = pearson_correlations[baseline_name][task_name][key]
                    spearman_data = spearman_correlations[baseline_name][task_name][key]
                    
                    if isinstance(pearson_data, dict) and isinstance(spearman_data, dict):
                        p_corr = pearson_data.get('correlation', 0)
                        s_corr = spearman_data.get('correlation', 0)
                        
                        if not np.isnan(p_corr) and not np.isnan(s_corr):
                            pearson_values.append(p_corr)
                            spearman_values.append(s_corr)
                            labels.append(f"{baseline_name}_{task_name}")
        
        if not pearson_values:
            return
        
        fig = go.Figure()
        
        fig.add_trace(go.Scatter(
            x=pearson_values,
            y=spearman_values,
            mode='markers',
            marker=dict(
                size=10,
                color=pearson_values,
                colorscale='RdBu',
                colorbar=dict(title="Pearson"),
                line=dict(width=1, color='black')
            ),
            text=labels,
            hovertemplate='%{text}<br>Pearson: %{x:.3f}<br>Spearman: %{y:.3f}<extra></extra>'
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
            font=dict(size=12),
            plot_bgcolor='white'
        )
        
        fig_path = viz_dir / 'correlation_stability_analysis.png'
        fig.write_image(str(fig_path), width=1000, height=800, scale=2)
        self.results['visualizations']['correlation_stability_analysis'] = str(fig_path)

    def _create_performance_evolution_timeline(self, viz_dir: Path):
        """Create timeline showing performance evolution."""
        timeline_data = []
        
        for model_name, model_results in self.results['llm_results'].items():
            for task_name, task_data in model_results.items():
                for dataset_name, dataset_data in task_data.items():
                    if 'performance_metrics' in dataset_data:
                        metrics = dataset_data['performance_metrics']
                        timeline_data.append({
                            'model': model_name,
                            'task_dataset': f"{task_name}_{dataset_name}",
                            'performance': metrics.get('primary_metric', 0),
                            'cost': dataset_data.get('cost', 0),
                            'time': dataset_data.get('processing_time', 0)
                        })
        
        if not timeline_data:
            return
        
        # Sort by performance
        timeline_data.sort(key=lambda x: x['performance'])
        
        fig = make_subplots(
            rows=3, cols=1,
            subplot_titles=['Performance Scores', 'Processing Costs', 'Processing Times'],
            vertical_spacing=0.1
        )
        
        x_labels = [f"{item['model'][:10]}_{item['task_dataset']}" for item in timeline_data]
        
        # Performance
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
        
        # Cost
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
        
        # Time
        fig.add_trace(
            go.Scatter(
                x=x_labels,
                y=[item['time'] for item in timeline_data],
                mode='lines+markers',
                name='Time',
                line=dict(color="#197133", width=3),
                marker=dict(size=8)
            ),
            row=3, col=1
        )
        
        fig.update_layout(
            title='Performance Evolution Timeline',
            font=dict(size=12),
            height=900,
            showlegend=False
        )
        
        fig_path = viz_dir / 'performance_evolution_timeline.png'
        fig.write_image(str(fig_path), width=1200, height=900, scale=2)
        self.results['visualizations']['performance_evolution_timeline'] = str(fig_path)

    def _create_baseline_robustness_analysis(self, viz_dir: Path):
        """Create robustness analysis for baselines."""
        pearson_correlations = self.results['correlation_analysis'].get('pearson_correlations', {})
        
        robustness_data = {}
        
        for baseline_name, baseline_data in pearson_correlations.items():
            correlations = []
            
            for task_name, task_data in baseline_data.items():
                for key, dataset_result in task_data.items():
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
                    'consistency': 1 - (np.std(correlations) / (np.mean(correlations) + 0.001))
                }
        
        if not robustness_data:
            return
        
        fig = make_subplots(
            rows=2, cols=2,
            subplot_titles=['Mean Performance', 'Performance Variance', 'Performance Range', 'Consistency Score']
        )
        
        baselines = list(robustness_data.keys())
        bar_colors = [self.get_baseline_color(baseline) for baseline in baselines]
        
        fig.add_trace(
            go.Bar(x=baselines, y=[robustness_data[b]['mean'] for b in baselines], 
                marker_color=bar_colors, showlegend=False),
            row=1, col=1
        )
        
        fig.add_trace(
            go.Bar(x=baselines, y=[robustness_data[b]['std'] for b in baselines], 
                marker_color=bar_colors, showlegend=False),
            row=1, col=2
        )
        
        fig.add_trace(
            go.Bar(x=baselines, y=[robustness_data[b]['max'] - robustness_data[b]['min'] for b in baselines], 
                marker_color=bar_colors, showlegend=False),
            row=2, col=1
        )
        
        fig.add_trace(
            go.Bar(x=baselines, y=[robustness_data[b]['consistency'] for b in baselines], 
                marker_color=bar_colors, showlegend=False),
            row=2, col=2
        )
        
        fig.update_layout(
            title='Baseline Robustness Analysis',
            font=dict(size=12),
            height=800
        )
        
        fig_path = viz_dir / 'baseline_robustness_analysis.png'
        fig.write_image(str(fig_path), width=1200, height=800, scale=2)
        self.results['visualizations']['baseline_robustness_analysis'] = str(fig_path)

    def _create_task_difficulty_heatmap(self, viz_dir: Path):
        """Create heatmap showing task difficulty."""
        task_difficulty = {}
        
        for model_name, model_results in self.results['llm_results'].items():
            for task_name, task_data in model_results.items():
                if task_name not in task_difficulty:
                    task_difficulty[task_name] = {
                        'performances': [],
                        'costs': [],
                        'times': []
                    }
                
                for dataset_name, dataset_data in task_data.items():
                    if 'performance_metrics' in dataset_data:
                        task_difficulty[task_name]['performances'].append(
                            dataset_data['performance_metrics'].get('primary_metric', 0)
                        )
                    task_difficulty[task_name]['costs'].append(dataset_data.get('cost', 0))
                    task_difficulty[task_name]['times'].append(dataset_data.get('processing_time', 0))
        
        if not task_difficulty:
            return
        
        tasks = list(task_difficulty.keys())
        metrics = ['Avg Performance', 'Avg Time', 'Performance Variance']
        
        heatmap_data = []
        for metric in metrics:
            row = []
            for task in tasks:
                if metric == 'Avg Performance':
                    val = np.mean(task_difficulty[task]['performances']) if task_difficulty[task]['performances'] else 0
                elif metric == 'Avg Time':
                    val = np.mean(task_difficulty[task]['times']) if task_difficulty[task]['times'] else 0
                else:  # Performance Variance
                    val = np.std(task_difficulty[task]['performances']) if len(task_difficulty[task]['performances']) > 1 else 0
                row.append(val)
            heatmap_data.append(row)
        
        # Normalize each row
        for i in range(len(heatmap_data)):
            row_max = max(heatmap_data[i]) if max(heatmap_data[i]) > 0 else 1
            heatmap_data[i] = [val / row_max for val in heatmap_data[i]]
        
        fig = go.Figure(data=go.Heatmap(
            z=heatmap_data,
            x=[task.replace('_', ' ').title() for task in tasks],
            y=metrics,
            colorscale='Reds',
            colorbar=dict(title="Normalized Score"),
            text=heatmap_data,
            texttemplate='%{text:.2f}',
            textfont={"size": 10}
        ))
        
        fig.update_layout(
            title='Task Difficulty Analysis Heatmap',
            font=dict(size=12),
            height=500
        )
        
        fig_path = viz_dir / 'task_difficulty_heatmap.png'
        fig.write_image(str(fig_path), width=800, height=500, scale=2)
        self.results['visualizations']['task_difficulty_heatmap'] = str(fig_path)

    def _create_task_specific_performance_charts(self, viz_dir: Path):
        """Create task-specific performance charts."""
        if 'entailment_inference' in self.experiment_params.get('tasks', []):
            self._create_entailment_performance_chart(viz_dir)
        
        if 'consistency_rating' in self.experiment_params.get('tasks', []):
            self._create_consistency_performance_chart(viz_dir)

    def _create_entailment_performance_chart(self, viz_dir: Path):
        """Create entailment inference specific performance chart."""
        fig = go.Figure()
        
        datasets = self.experiment_params.get('datasets', [])
        models = list(self.results['llm_results'].keys())
        baselines = list(self.results['baseline_results'].keys())
        
        # Model performance
        for i, model in enumerate(models):
            performances = []
            for dataset in datasets:
                if ('entailment_inference' in self.results['llm_results'][model] and 
                    dataset in self.results['llm_results'][model]['entailment_inference']):
                    perf_data = self.results['llm_results'][model]['entailment_inference'][dataset]
                    if 'performance_metrics' in perf_data:
                        performances.append(perf_data['performance_metrics'].get('accuracy', 0))
                    else:
                        performances.append(0)
                else:
                    performances.append(0)
            
            if performances:
                fig.add_trace(go.Bar(
                    name=model,
                    x=datasets,
                    y=performances,
                    text=[f'{p:.1%}' for p in performances],
                    textposition='auto'
                ))
        
        # Baseline performance
        for baseline in baselines:
            if 'entailment_inference' in self.results['baseline_results'].get(baseline, {}):
                performances = []
                for dataset in datasets:
                    if dataset in self.results['baseline_results'][baseline]['entailment_inference']:
                        perf_data = self.results['baseline_results'][baseline]['entailment_inference'][dataset]
                        if 'performance_metrics' in perf_data:
                            performances.append(perf_data['performance_metrics'].get('accuracy', 0))
                        else:
                            performances.append(0)
                    else:
                        performances.append(0)
                
                if performances:
                    fig.add_trace(go.Bar(
                        name=baseline.upper(),
                        x=datasets,
                        y=performances,
                        marker_color=self.get_baseline_color(baseline),
                        text=[f'{p:.1%}' for p in performances],
                        textposition='auto'
                    ))
        
        fig.update_layout(
            title='Entailment Inference: Accuracy Comparison',
            xaxis_title='Dataset',
            yaxis_title='Accuracy',
            yaxis=dict(tickformat='.0%'),
            barmode='group',
            height=500,
            legend=dict(orientation="h", yanchor="bottom", y=1.02, xanchor="right", x=1)
        )
        
        fig_path = viz_dir / 'entailment_inference_performance.png'
        fig.write_image(str(fig_path), width=1000, height=500, scale=2)
        self.results['visualizations']['entailment_performance'] = str(fig_path)

    def _create_consistency_performance_chart(self, viz_dir: Path):
        """Create consistency rating specific performance chart."""
        fig = go.Figure()
        
        datasets = self.experiment_params.get('datasets', [])
        models = list(self.results['llm_results'].keys())
        baselines = list(self.results['baseline_results'].keys())
        
        # Model performance
        for i, model in enumerate(models):
            correlations = []
            for dataset in datasets:
                if ('consistency_rating' in self.results['llm_results'][model] and 
                    dataset in self.results['llm_results'][model]['consistency_rating']):
                    perf_data = self.results['llm_results'][model]['consistency_rating'][dataset]
                    if 'performance_metrics' in perf_data:
                        correlations.append(perf_data['performance_metrics'].get('pearson_correlation', 0))
                    else:
                        correlations.append(0)
                else:
                    correlations.append(0)
            
            if correlations:
                fig.add_trace(go.Bar(
                    name=model,
                    x=datasets,
                    y=correlations,
                    text=[f'{c:.3f}' for c in correlations],
                    textposition='auto'
                ))
        
        # Baseline performance
        colors = {'factcc': "#023D1C", 'bertscore': "#A0DDEB", 'rouge': "#08673B"}
        for baseline in baselines:
            if 'consistency_rating' in self.results['baseline_results'].get(baseline, {}):
                correlations = []
                for dataset in datasets:
                    if dataset in self.results['baseline_results'][baseline]['consistency_rating']:
                        perf_data = self.results['baseline_results'][baseline]['consistency_rating'][dataset]
                        if 'performance_metrics' in perf_data:
                            correlations.append(perf_data['performance_metrics'].get('pearson_correlation', 0))
                        else:
                            correlations.append(0)
                    else:
                        correlations.append(0)
                
                if correlations:
                    fig.add_trace(go.Bar(
                        name=baseline.upper(),
                        x=datasets,
                        y=correlations,
                        marker_color=colors.get(baseline, '#888888'),
                        text=[f'{c:.3f}' for c in correlations],
                        textposition='auto'
                    ))
        
        fig.update_layout(
            title='Consistency Rating: Correlation Comparison',
            xaxis_title='Dataset',
            yaxis_title='Pearson Correlation',
            barmode='group',
            height=500,
            legend=dict(orientation="h", yanchor="bottom", y=1.02, xanchor="right", x=1)
        )
        
        fig_path = viz_dir / 'consistency_rating_performance.png'
        fig.write_image(str(fig_path), width=1000, height=500, scale=2)
        self.results['visualizations']['consistency_performance'] = str(fig_path)

    def _create_model_comparison_radar(self, viz_dir: Path):
        """Create radar chart comparing multiple models across dimensions."""
        models = list(self.results['llm_results'].keys())
        
        if not models:
            return
        
        categories = ['Accuracy', 'Correlation', 'Cost Efficiency', 'Speed', 'Consistency']
        
        fig = go.Figure()
        
        for model_name in models:
            # Calculate metrics for this model
            accuracies = []
            correlations = []
            costs = []
            times = []
            
            for task_name, task_data in self.results['llm_results'][model_name].items():
                for dataset_name, dataset_data in task_data.items():
                    if 'performance_metrics' in dataset_data:
                        metrics = dataset_data['performance_metrics']
                        if task_name == 'entailment_inference':
                            accuracies.append(metrics.get('accuracy', 0))
                        elif task_name == 'consistency_rating':
                            corr = metrics.get('pearson_correlation', 0)
                            correlations.append(abs(corr) if not np.isnan(corr) else 0)
                    
                    costs.append(dataset_data.get('cost', 0))
                    times.append(dataset_data.get('processing_time', 0))
            
            # Normalize metrics
            accuracy_score = np.mean(accuracies) if accuracies else 0.5
            correlation_score = np.mean(correlations) if correlations else 0.5
            cost_efficiency = 1 / (1 + np.mean(costs)) if costs else 0.5
            speed = 1 / (1 + np.mean(times)) if times else 0.5
            consistency = 1 - (np.std(accuracies + correlations) if (accuracies + correlations) else 0.5)
            
            values = [accuracy_score, correlation_score, cost_efficiency, speed, consistency]
            values.append(values[0])  # Close the radar
            categories_closed = categories + [categories[0]]
            
            fig.add_trace(go.Scatterpolar(
                r=values,
                theta=categories_closed,
                fill='toself',
                name=model_name,
                opacity=0.4
            ))
        
        fig.update_layout(
            polar=dict(
                radialaxis=dict(
                    visible=True,
                    range=[0, 1]
                )
            ),
            title='Multi-Model Performance Radar Chart',
            font=dict(size=12)
        )
        
        fig_path = viz_dir / 'model_comparison_radar.png'
        fig.write_image(str(fig_path), width=800, height=800, scale=2)
        self.results['visualizations']['model_comparison_radar'] = str(fig_path)

    def _create_model_performance_heatmap(self, viz_dir: Path):
        """Create heatmap showing model performance across tasks and datasets."""
        models = list(self.results['llm_results'].keys())
        
        if not models:
            return
        
        # Collect task-dataset combinations
        task_datasets = set()
        for model_results in self.results['llm_results'].values():
            for task_name, task_data in model_results.items():
                for dataset_name in task_data.keys():
                    task_datasets.add(f"{task_name}_{dataset_name}")
        
        task_datasets = sorted(list(task_datasets))
        
        if not task_datasets:
            return
        
        # Create performance matrix
        performance_matrix = []
        for model in models:
            row = []
            for td in task_datasets:
                task, dataset = td.rsplit('_', 1)
                if (task in self.results['llm_results'][model] and 
                    dataset in self.results['llm_results'][model][task]):
                    data = self.results['llm_results'][model][task][dataset]
                    if 'performance_metrics' in data:
                        row.append(data['performance_metrics'].get('primary_metric', 0))
                    elif 'comprehensive_metrics' in data:
                        row.append(data['comprehensive_metrics'].get('primary_metric', 0))
                    else:
                        row.append(0)
                else:
                    row.append(0)
            performance_matrix.append(row)
        
        fig = go.Figure(data=go.Heatmap(
            z=performance_matrix,
            x=[td.replace('_', ' ').title() for td in task_datasets],
            y=models,
            colorscale='Viridis',
            colorbar=dict(title="Performance"),
            text=performance_matrix,
            texttemplate='%{text:.3f}',
            textfont={"size": 10}
        ))
        
        fig.update_layout(
            title='Model Performance Heatmap',
            xaxis_title='Task-Dataset',
            yaxis_title='Model',
            font=dict(size=12),
            height=600
        )
        
        fig_path = viz_dir / 'model_performance_heatmap.png'
        fig.write_image(str(fig_path), width=1200, height=600, scale=2)
        self.results['visualizations']['model_performance_heatmap'] = str(fig_path)
    
    def _create_correlation_heatmap(self, viz_dir: Path):
        """Create correlation heatmap visualization."""
        correlation_data = []
        pearson_correlations = self.results['correlation_analysis']['pearson_correlations']
        
        for baseline_name, baseline_corr in pearson_correlations.items():
            for task_name, task_corr in baseline_corr.items():
                for dataset_key, dataset_corr in task_corr.items():
                    if isinstance(dataset_corr, dict):
                        # Extract dataset name from key like 'gpt-4.1-mini_frank'
                        dataset_name = dataset_key.split('_')[-1]
                        correlation_data.append({
                            'baseline': baseline_name,
                            'task_dataset': f"{task_name}_{dataset_name}",
                            'correlation': dataset_corr.get('correlation', 0),
                            'model': dataset_corr.get('model_name', 'unknown')
                        })
        
        if not correlation_data:
            return
        
        baselines = sorted(list(set(item['baseline'] for item in correlation_data)))
        task_datasets = sorted(list(set(item['task_dataset'] for item in correlation_data)))
        
        correlation_matrix = np.zeros((len(baselines), len(task_datasets)))
        
        for i, baseline in enumerate(baselines):
            for j, task_dataset in enumerate(task_datasets):
                matching = [item for item in correlation_data 
                          if item['baseline'] == baseline and item['task_dataset'] == task_dataset]
                if matching:
                    correlation_matrix[i, j] = matching[0]['correlation']
        
        fig = go.Figure(data=go.Heatmap(
            z=correlation_matrix,
            x=task_datasets,
            y=baselines,
            colorscale='RdBu',
            zmid=0,
            zmin=-1,
            zmax=1,
            colorbar=dict(title="Correlation"),
            text=correlation_matrix,
            texttemplate="%{text:.3f}",
            textfont=dict(size=10),
            hoverongaps=False
        ))
        
        fig.update_layout(
            title='Multi-LLM vs Baseline Correlation Matrix',
            xaxis_title='Task-Dataset',
            yaxis_title='Baseline Method',
            font=dict(size=12),
            paper_bgcolor='white'
        )
        
        fig_path = viz_dir / "correlation_heatmap.png"
        fig.write_image(str(fig_path), width=1200, height=600, scale=2)
        self.results['visualizations']['correlation_heatmap'] = str(fig_path)
    
    def _create_baseline_performance_chart(self, viz_dir: Path):
        """Create baseline performance comparison chart."""
        correlation_summary = self.results['correlation_analysis'].get('correlation_summary', {})
        baseline_avg = correlation_summary.get('baseline_average_correlations', {})
        
        if not baseline_avg:
            return
        
        # Sort by correlation strength
        sorted_data = sorted(baseline_avg.items(), key=lambda x: x[1], reverse=True)
        baselines, correlations = zip(*sorted_data) if sorted_data else ([], [])
        
        fig = go.Figure()
        
        # Color based on correlation strength
        colors = ['green' if c > 0.7 else 'orange' if c > 0.5 else 'red' for c in correlations]
        
        fig.add_trace(go.Bar(
            x=baselines,
            y=correlations,
            marker_color=colors,
            text=[f'{c:.3f}' for c in correlations],
            textposition='auto'
        ))
        
        # Add reference lines
        fig.add_hline(y=0.7, line_dash="dash", line_color="green", annotation_text="Strong")
        fig.add_hline(y=0.5, line_dash="dash", line_color="orange", annotation_text="Moderate")
        fig.add_hline(y=0.3, line_dash="dash", line_color="red", annotation_text="Weak")
        
        fig.update_layout(
            title='Average Correlation: Multi-LLM vs SOTA Baselines',
            xaxis_title='Baseline Method',
            yaxis_title='Average Correlation',
            font=dict(size=12),
            plot_bgcolor='white'
        )
        
        fig_path = viz_dir / "baseline_performance_comparison.png"
        fig.write_image(str(fig_path), width=1000, height=600, scale=2)
        self.results['visualizations']['baseline_performance_comparison'] = str(fig_path)
    
    def _create_model_performance_comparison(self, viz_dir: Path):
        """Create model performance comparison visualization."""
        models = list(self.results['llm_results'].keys())
        tasks = list(self.experiment_params.get('tasks', []))
        datasets = list(self.experiment_params.get('datasets', []))
        
        if not models or not tasks:
            print("⚠️ No models or tasks data available for model performance comparison")
            return
        
        n_tasks = len(tasks)
        if n_tasks == 1:
            rows, cols = 1, 1
        elif n_tasks == 2:
            rows, cols = 1, 2
        else:
            rows = (n_tasks + 1) // 2
            cols = 2
        
        fig = make_subplots(
            rows=rows, cols=cols,
            subplot_titles=[f'{task.replace("_", " ").title()} Performance' for task in tasks],
            specs=[[{"secondary_y": False} for _ in range(cols)] for _ in range(rows)]
        )
        
        # Colors for different models
        model_colors = ['#1f77b4', '#ff7f0e', '#2ca02c', '#d62728', '#9467bd', '#8c564b']
        
        for task_idx, task_name in enumerate(tasks):
            row = (task_idx // cols) + 1
            col = (task_idx % cols) + 1
            
            # Collect performance data for this task
            model_performances = {}
            for model_name in models:
                if model_name in self.results['llm_results'] and task_name in self.results['llm_results'][model_name]:
                    task_data = self.results['llm_results'][model_name][task_name]
                    
                    # Calculate average performance across datasets for this task
                    performances = []
                    for dataset_name in datasets:
                        if dataset_name in task_data:
                            dataset_data = task_data[dataset_name]
                            if 'performance_metrics' in dataset_data:
                                perf = dataset_data['performance_metrics'].get('primary_metric', 0)
                            elif 'comprehensive_metrics' in dataset_data:
                                perf = dataset_data['comprehensive_metrics'].get('primary_metric', 0)
                            else:
                                # Fallback to looking for common metric names
                                perf = dataset_data.get('score', dataset_data.get('accuracy', 0))
                            performances.append(perf)
                    
                    if performances:
                        model_performances[model_name] = np.mean(performances)
            
            if model_performances:
                model_names = list(model_performances.keys())
                performance_values = list(model_performances.values())
                
                fig.add_trace(
                    go.Bar(
                        name=f'{task_name}',
                        x=model_names,
                        y=performance_values,
                        marker_color=[model_colors[i % len(model_colors)] for i in range(len(model_names))],
                        text=[f'{v:.3f}' for v in performance_values],
                        textposition='auto',
                        showlegend=False
                    ),
                    row=row, col=col
                )
                print(f"📊 Added data for {task_name}: {dict(zip(model_names, performance_values))}")
            else:
                print(f"⚠️ No performance data found for task: {task_name}")
        
        fig.update_layout(
            title='Multi-Model Performance Comparison Across Tasks',
            font=dict(size=12),
            height=400 * rows,
            showlegend=False
        )
        
        # Update y-axis titles
        for i in range(1, rows + 1):
            for j in range(1, cols + 1):
                if (i - 1) * cols + j <= len(tasks):
                    fig.update_yaxes(title_text="Performance Score", row=i, col=j)
                    fig.update_xaxes(title_text="Models", row=i, col=j)
        
        fig_path = viz_dir / "model_performance_comparison.png"
        fig.write_image(str(fig_path), width=1200, height=400 * rows, scale=2)
        self.results['visualizations']['model_performance_comparison'] = str(fig_path)
        print(f"✅ Model performance comparison saved to {fig_path}")
    
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
                title='Statistical Significance of Multi-LLM vs Baseline Correlations',
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
                 f"Most correlations may not be significant due to<br>" +
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
            title='Statistical Significance of Multi-LLM vs Baseline Correlations',
            xaxis_title='Absolute Correlation Coefficient',
            yaxis_title='P-value',
            yaxis_type='log',
            font=dict(size=12),
            plot_bgcolor='white',
            paper_bgcolor='white',
            height=600
        )
        
        fig.write_image(viz_dir / 'statistical_significance.png', width=1000, height=600, scale=2)
        self.results['visualizations']['statistical_significance'] = str(viz_dir / 'statistical_significance.png')
    
    async def _save_results(self):
        """Save comprehensive results and generate reports."""
        print(f"\n💾 Saving Results")
        print("-" * 40)
        
        results_file = self.output_dir / "sota_multi_comparison_results.json"
        with open(results_file, 'w') as f:
            json.dump(self._make_json_serializable(self.results), f, indent=2)
        
        for baseline_name in self.results['baseline_results'].keys():
            print(f"   ✅ Saved {baseline_name} results to baseline_results/")
        
        # Generate and save report
        report_file = self.output_dir / "sota_multi_comparison_report.md"
        with open(report_file, 'w') as f:
            f.write(self._generate_report())
        
        # Generate summary file
        summary_file = self.output_dir / "experiment_summary.txt"
        with open(summary_file, 'w') as f:
            f.write(self._generate_summary())
        
        print(f"✅ Results saved to {self.output_dir}")
    
    def _make_json_serializable(self, obj):
        """Convert numpy and other non-serializable objects to JSON-compatible format."""
        if isinstance(obj, dict):
            return {key: self._make_json_serializable(value) for key, value in obj.items()}
        elif isinstance(obj, list):
            return [self._make_json_serializable(item) for item in obj]
        elif isinstance(obj, (np.ndarray, np.generic)):
            return obj.tolist()
        elif isinstance(obj, (np.integer, np.floating)):
            return float(obj)
        elif isinstance(obj, (np.bool_, bool)):
            return bool(obj)
        elif isinstance(obj, Path):
            return str(obj)
        elif isinstance(obj, set):
            return list(obj)
        elif hasattr(obj, '__dict__'):
            return self._make_json_serializable(obj.__dict__)
        else:
            return obj
    
    def _generate_summary(self):
        """Generate experiment summary."""
        summary = f"# Multi-LLM SOTA Comparison Experiment Summary\n\n"
        summary += f"**Experiment**: {self.experiment_name}\n"
        summary += f"**Timestamp**: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}\n\n"
        
        # Model and baseline info
        summary += f"**Models Evaluated**: {', '.join(self.results.get('summary_statistics', {}).get('models_evaluated', []))}\n"
        summary += f"**Baselines**: {', '.join(self.results.get('summary_statistics', {}).get('baselines_evaluated', []))}\n"
        summary += f"**Sample Size**: {self.results.get('summary_statistics', {}).get('sample_size', 'N/A')}\n\n"
        
        # Cost and experiments
        summary += f"**Total Cost**: ${self.results.get('cost_analysis', {}).get('llm_cost', 0):.4f}\n\n"
        
        # Key metrics
        correlation_summary = self.results.get('correlation_analysis', {}).get('correlation_summary', {})
        if correlation_summary.get('best_correlating_baseline'):
            summary += f"**Best Baseline**: {correlation_summary['best_correlating_baseline']}\n"
            summary += f"**Best Score**: {correlation_summary['best_correlation']:.4f}\n"
        
        overall_stats = correlation_summary.get('overall_statistics', {})
        if overall_stats:
            summary += f"**Mean Correlation**: {overall_stats.get('mean', 0):.4f}\n"
            summary += f"**Valid Comparisons**: {overall_stats.get('valid_correlations', 0)}\n"
        
        return summary
    
    def _generate_report(self):
        """Generate comprehensive experiment report."""
        report = f"# Multi-LLM SOTA Comparison Experiment Report\n\n"
        report += f"**Experiment Name**: {self.experiment_name}\n"
        report += f"**Generated**: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}\n\n"
        
        # Executive Summary
        report += "## Executive Summary\n\n"
        report += "This experiment compares multiple Large Language Models (LLMs) against "
        report += "state-of-the-art baseline methods for factuality evaluation tasks.\n\n"
        
        # Add sample size warning if small
        sample_size = self.results.get('summary_statistics', {}).get('sample_size', 0)
        if sample_size < 10:
            report += f"**Note**: This analysis used only {sample_size} samples. "
            report += "Results should be considered preliminary and may not generalize to larger datasets.\n\n"
        
        # Results Summary
        report += "## Results Summary\n\n"
        
        # Correlation Analysis
        correlation_summary = self.results.get('correlation_analysis', {}).get('correlation_summary', {})
        if correlation_summary:
            report += "### Correlation Analysis\n\n"
            
            baseline_avg = correlation_summary.get('baseline_average_correlations', {})
            if baseline_avg:
                report += "#### Baseline Performance\n\n"
                for baseline, corr in sorted(baseline_avg.items(), key=lambda x: x[1], reverse=True):
                    interpretation = "Strong" if corr > 0.7 else "Moderate" if corr > 0.5 else "Weak"
                    report += f"- **{baseline.upper()}**: {corr:.4f} ({interpretation})\n"
                report += "\n"
        
        # Key Findings
        report += "## Key Findings\n\n"
        for i, finding in enumerate(self.results.get('summary_statistics', {}).get('key_findings', []), 1):
            report += f"{i}. {finding}\n"
        
        # Conclusions
        report += "\n## Conclusions and Recommendations\n\n"
        
        if sample_size < 10:
            report += "1. **Limited Sample Size**: Increase sample size for more reliable results.\n"
        
        if correlation_summary.get('best_correlating_baseline'):
            best_corr = correlation_summary.get('best_correlation', 0)
            if best_corr > 0.7:
                report += "2. **Strong Baseline Agreement**: Multi-LLM shows strong correlation with traditional metrics.\n"
            elif best_corr > 0.5:
                report += "2. **Moderate Baseline Agreement**: Multi-LLM shows moderate correlation with traditional metrics.\n"
            else:
                report += "2. **Limited Baseline Agreement**: Multi-LLM shows novel evaluation patterns.\n"
        
        report += "3. **Future Work**: Consider larger sample sizes and additional baseline methods.\n"
        
        return report


def main():
    """Main entry point for Multi-LLM SOTA comparison experiment."""
    parser = argparse.ArgumentParser(
        description="Compare Multiple LLMs with SOTA baseline methods for factuality evaluation"
    )
    parser.add_argument(
        "--models",
        nargs="+",
        default="gpt-4.1-mini qwen2.5:7b llama3.1:8b".split(),
        help="List of models to evaluate (e.g., gpt-4.1-mini qwen2.5:7b llama3.1:8b)"
    )
    parser.add_argument(
        "--tier",
        type=str,
        default="tier2",
        choices=["tier1", "tier2", "tier3", "tier4", "tier5"],
        help="OpenAI API tier"
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
        choices=['frank', 'summeval'],
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
        help="LLM prompt type to use"
    )
    parser.add_argument(
        "--quick-test",
        action="store_true",
        help="Run quick test with minimal data"
    )
    parser.add_argument(
        "--comprehensive",
        action="store_true",
        help="Run comprehensive experiment with 800 samples per dataset"
    )
    
    args = parser.parse_args()
    
    tasks = [args.task] if args.task else None
    datasets = [args.dataset] if args.dataset else None
    baselines = [args.baseline] if args.baseline else None
    sample_size = args.sample_size
    
    if args.comprehensive:
        if sample_size is None:
            sample_size = 800
        print(f"🚀 Running comprehensive experiment with {sample_size} examples per dataset")
    elif args.quick_test:
        if sample_size is None:
            sample_size = 50
        print(f"⚡ Running quick test with {sample_size} examples per dataset")
  
    experiment = SOTAMultiLLMComparisonExperiment(
        model=args.models[0],
        tier=args.tier,
        experiment_name=args.experiment_name
    )
    
    experiment.models = args.models
    
    # Run experiment
    results = asyncio.run(experiment.run_sota_comparison(
        tasks=tasks,
        datasets=datasets,
        baselines=baselines,
        sample_size=sample_size,
        prompt_type=args.prompt_type
    ))
    
    # Print final summary
    print(f"\n{'='*60}")
    print(f"MULTI-LLM SOTA COMPARISON EXPERIMENT COMPLETED")
    print(f"{'='*60}")
    print(f"Experiment: {experiment.experiment_name}")
    print(f"Output directory: {experiment.output_dir}")
    
    # Display key metrics
    summary_stats = results.get('summary_statistics', {})
    print(f"Models evaluated: {', '.join(summary_stats.get('models_evaluated', []))}")
    print(f"Total cost: ${results.get('cost_analysis', {}).get('llm_cost', 0):.4f}")
    
    # Display correlation summary
    correlation_summary = results.get('correlation_analysis', {}).get('correlation_summary', {})
    if correlation_summary.get('overall_statistics'):
        stats = correlation_summary['overall_statistics']
        print(f"\nCorrelation Statistics:")
        print(f"  Valid correlations: {stats.get('valid_correlations', 0)}")
        print(f"  Mean correlation: {stats.get('mean', 0):.4f}")
        
        if correlation_summary.get('best_correlating_baseline'):
            print(f"  Best baseline: {correlation_summary['best_correlating_baseline']} "
                  f"({correlation_summary['best_correlation']:.4f})")
    
    print(f"{'='*60}")


if __name__ == "__main__":
    main()