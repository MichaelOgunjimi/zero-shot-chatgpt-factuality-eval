#!/usr/bin/env python3
"""
Cost Estimation Script for ChatGPT Factuality Evaluation
========================================================

This script estimates the costs for running different experiment configurations
to help with budget planning and API usage monitoring.

Usage:
    python scripts/estimate_costs.py
    python scripts/estimate_costs.py --experiment quick-test
    python scripts/estimate_costs.py --experiment comprehensive
    python scripts/estimate_costs.py --custom-config config/my_config.yaml

Author: Michael Ogunjimi
Institution: University of Manchester
Course: MSc AI
"""

import argparse
import logging
import sys
from pathlib import Path
from typing import Dict, List, Optional, Tuple

# Add project root to path
sys.path.insert(0, str(Path(__file__).parent.parent))

# Setup logging
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)

class CostEstimator:
    """Cost estimation for factuality evaluation experiments."""
    
    def __init__(self, config_path: str = "config/default.yaml"):
        """Initialize cost estimator."""
        try:
            from src.utils import load_config
            self.config = load_config(config_path)
        except Exception as e:
            logger.error(f"Failed to load config: {e}")
            sys.exit(1)
        
        # OpenAI pricing (as of July 2025)
        self.openai_pricing = {
            # Primary and fallback models from config
            'gpt-4.1-mini': {
                'input': 0.40 / 1000000,    # $0.40 per 1M tokens
                'output': 1.60 / 1000000,   # $1.60 per 1M tokens
            },
            'o1-mini': {
                'input': 3.00 / 1000000,    # $3.00 per 1M tokens
                'output': 12.00 / 1000000,  # $12.00 per 1M tokens
            },
            'gpt-4o': {
                'input': 2.50 / 1000000,    # $2.50 per 1M tokens
                'output': 10.00 / 1000000,  # $10.00 per 1M tokens
            },
            # Additional models
            'o4-mini': {
                'input': 1.10 / 1000000,    # $1.10 per 1M tokens
                'output': 4.40 / 1000000,   # $4.40 per 1M tokens
            },
            # Legacy models (keeping for backward compatibility)
            'gpt-4-turbo': {
                'input': 0.01 / 1000,    # $0.01 per 1K tokens
                'output': 0.03 / 1000,   # $0.03 per 1K tokens
            },
            'gpt-4': {
                'input': 0.03 / 1000,    # $0.03 per 1K tokens
                'output': 0.06 / 1000,   # $0.06 per 1K tokens
            },
            'gpt-3.5-turbo': {
                'input': 0.0005 / 1000,  # $0.0005 per 1K tokens
                'output': 0.0015 / 1000, # $0.0015 per 1K tokens
            },
            'gpt-4-1106-preview': {
                'input': 0.01 / 1000,    # $0.01 per 1K tokens
                'output': 0.03 / 1000,   # $0.03 per 1K tokens
            }
        }
        
        # Default model from config (fallback to cost-effective gpt-4.1-mini)
        self.model_name = self.config.get('llm_clients', {}).get('openai', {}).get('model', 'gpt-4.1-mini')
        
        # Estimated token counts for different tasks
        self.token_estimates = {
            'entailment_inference': {
                'input_tokens': 800,   # prompt + source + summary
                'output_tokens': 50,   # short response
            },
            'summary_ranking': {
                'input_tokens': 1200,  # prompt + source + multiple summaries
                'output_tokens': 100,  # ranking explanation
            },
            'consistency_rating': {
                'input_tokens': 900,   # prompt + source + summary
                'output_tokens': 80,   # rating + explanation
            }
        }
    
    def estimate_single_call_cost(self, task_name: str, model: str = None) -> float:
        """Estimate cost for a single API call."""
        model = model or self.model_name
        
        if model not in self.openai_pricing:
            logger.warning(f"Unknown model {model}, using gpt-4-turbo pricing")
            model = 'gpt-4-turbo'
        
        if task_name not in self.token_estimates:
            logger.warning(f"Unknown task {task_name}, using average estimates")
            task_name = 'consistency_rating'
        
        pricing = self.openai_pricing[model]
        tokens = self.token_estimates[task_name]
        
        input_cost = tokens['input_tokens'] * pricing['input']
        output_cost = tokens['output_tokens'] * pricing['output']
        
        return input_cost + output_cost
    
    def estimate_experiment_cost(self, experiment_config: Dict) -> Dict:
        """Estimate cost for a specific experiment configuration."""
        total_cost = 0
        breakdown = {}
        
        tasks = experiment_config.get('tasks', ['consistency_rating'])
        datasets = experiment_config.get('datasets', ['cnn_dailymail'])
        sample_size = experiment_config.get('sample_size', 100)
        prompt_types = experiment_config.get('prompt_types', ['zero_shot'])
        
        for task in tasks:
            task_cost = 0
            task_breakdown = {}
            
            for dataset in datasets:
                dataset_cost = 0
                
                for prompt_type in prompt_types:
                    # Single call cost
                    call_cost = self.estimate_single_call_cost(task)
                    
                    # Multiply by sample size
                    prompt_cost = call_cost * sample_size
                    dataset_cost += prompt_cost
                    
                    task_breakdown[f"{dataset}_{prompt_type}"] = {
                        'call_cost': call_cost,
                        'sample_size': sample_size,
                        'total_cost': prompt_cost
                    }
                
                task_cost += dataset_cost
            
            breakdown[task] = {
                'total_cost': task_cost,
                'breakdown': task_breakdown
            }
            total_cost += task_cost
        
        return {
            'total_cost': total_cost,
            'model': self.model_name,
            'breakdown': breakdown,
            'configuration': experiment_config
        }
    
    def estimate_predefined_experiments(self) -> Dict:
        """Estimate costs for predefined experiment configurations based on actual project setup."""
        experiments = {
            'quick_test': {
                'tasks': ['consistency_rating'],
                'datasets': ['cnn_dailymail'],
                'sample_size': 20,
                'prompt_types': ['zero_shot'],
                'description': 'Quick test with minimal data'
            },
            'development': {
                'tasks': ['entailment_inference', 'consistency_rating'],
                'datasets': ['cnn_dailymail', 'xsum'],
                'sample_size': 100,
                'prompt_types': ['zero_shot'],
                'description': 'Development testing'
            },
            'prompt_comparison': {
                'tasks': ['entailment_inference', 'summary_ranking', 'consistency_rating'],
                'datasets': ['cnn_dailymail', 'xsum'],
                'sample_size': 200,  # From config: experiments.main_experiments.prompt_comparison.sample_size
                'prompt_types': ['zero_shot', 'chain_of_thought'],
                'description': 'Prompt comparison experiment (config-based)'
            },
            'chatgpt_evaluation': {
                'tasks': ['entailment_inference', 'summary_ranking', 'consistency_rating'],
                'datasets': ['cnn_dailymail', 'xsum'],
                'sample_size': 1000,
                'prompt_types': ['zero_shot'],
                'description': 'Main ChatGPT evaluation (development scale)'
            },
            'sota_comparison': {
                'tasks': ['entailment_inference', 'consistency_rating'],
                'datasets': ['cnn_dailymail', 'xsum'],
                'sample_size': 300,  # From config: experiments.main_experiments.sota_comparison.sample_size
                'prompt_types': ['zero_shot'],
                'description': 'SOTA baseline comparison (config-based)'
            },
            'comprehensive_thesis': {
                'tasks': ['entailment_inference', 'summary_ranking', 'consistency_rating'],
                'datasets': ['cnn_dailymail', 'xsum'],
                'sample_size': 2000,
                'prompt_types': ['zero_shot', 'chain_of_thought'],
                'description': 'Complete thesis experimental suite'
            },
            # New experiments based on run_all_experiments.py
            'thesis_chatgpt_evaluation': {
                'tasks': ['entailment_inference', 'summary_ranking', 'consistency_rating'],
                'datasets': ['cnn_dailymail', 'xsum'],
                'sample_size': 10000,  # From run_all_experiments.py: 10k samples
                'prompt_types': ['zero_shot'],
                'description': 'Full-scale ChatGPT evaluation (thesis comprehensive)'
            },
            'thesis_prompt_comparison': {
                'tasks': ['entailment_inference', 'summary_ranking', 'consistency_rating'],
                'datasets': ['cnn_dailymail', 'xsum'],
                'sample_size': 1500,  # From run_all_experiments.py: 1.5k samples
                'prompt_types': ['zero_shot', 'chain_of_thought'],
                'description': 'Full-scale prompt comparison (thesis comprehensive)'
            },
            'thesis_sota_comparison': {
                'tasks': ['entailment_inference', 'consistency_rating'],
                'datasets': ['cnn_dailymail', 'xsum'],
                'sample_size': 2000,  # From run_all_experiments.py: 2k samples
                'prompt_types': ['zero_shot'],
                'description': 'Full-scale SOTA comparison (thesis comprehensive)'
            }
        }
        
        estimates = {}
        for exp_name, config in experiments.items():
            estimates[exp_name] = self.estimate_experiment_cost(config)
        
        return estimates
    
    def estimate_monthly_budget(self, experiments: List[str], runs_per_month: int = 1) -> Dict:
        """Estimate monthly budget for running experiments."""
        predefined = self.estimate_predefined_experiments()
        
        monthly_cost = 0
        breakdown = {}
        
        for exp_name in experiments:
            if exp_name in predefined:
                exp_cost = predefined[exp_name]['total_cost']
                monthly_exp_cost = exp_cost * runs_per_month
                monthly_cost += monthly_exp_cost
                
                breakdown[exp_name] = {
                    'cost_per_run': exp_cost,
                    'runs_per_month': runs_per_month,
                    'monthly_cost': monthly_exp_cost
                }
        
        return {
            'total_monthly_cost': monthly_cost,
            'breakdown': breakdown,
            'daily_average': monthly_cost / 30,
            'weekly_average': monthly_cost / 4
        }
    
    def get_model_comparison(self, experiment_config: Dict) -> Dict:
        """Compare costs across different models including primary and fallback models."""
        # Include primary and fallback models from config
        models = ['gpt-4.1-mini', 'o1-mini', 'gpt-4o', 'o4-mini', 'gpt-3.5-turbo', 'gpt-4-turbo', 'gpt-4']
        comparisons = {}
        
        original_model = self.model_name
        
        for model in models:
            self.model_name = model
            estimate = self.estimate_experiment_cost(experiment_config)
            comparisons[model] = {
                'total_cost': estimate['total_cost'],
                'cost_per_call': estimate['total_cost'] / (
                    len(experiment_config.get('tasks', [1])) * 
                    len(experiment_config.get('datasets', [1])) * 
                    len(experiment_config.get('prompt_types', [1])) * 
                    experiment_config.get('sample_size', 1)
                )
            }
        
        # Restore original model
        self.model_name = original_model
        
        return comparisons
    
    def print_single_experiment_estimate(self, experiment_name: str, estimate: Dict):
        """Print detailed estimate for a single experiment."""
        config = estimate['configuration']
        
        logger.info(f"\n{'='*60}")
        logger.info(f"COST ESTIMATE: {experiment_name.upper()}")
        logger.info(f"{'='*60}")
        logger.info(f"Description: {config.get('description', 'N/A')}")
        logger.info(f"Model: {estimate['model']}")
        logger.info(f"Total Estimated Cost: ${estimate['total_cost']:.4f}")
        logger.info(f"")
        
        # Configuration details
        logger.info(f"Configuration:")
        logger.info(f"  Tasks: {', '.join(config['tasks'])}")
        logger.info(f"  Datasets: {', '.join(config['datasets'])}")
        logger.info(f"  Sample Size: {config['sample_size']:,}")
        logger.info(f"  Prompt Types: {', '.join(config['prompt_types'])}")
        logger.info(f"")
        
        # Breakdown by task
        logger.info(f"Cost Breakdown by Task:")
        for task, task_data in estimate['breakdown'].items():
            logger.info(f"  {task}: ${task_data['total_cost']:.4f}")
            for dataset_prompt, details in task_data['breakdown'].items():
                logger.info(f"    {dataset_prompt}: ${details['total_cost']:.4f} ({details['sample_size']} calls @ ${details['call_cost']:.4f} each)")
        
        logger.info(f"{'='*60}")
    
    def print_comparison_table(self, estimates: Dict):
        """Print comprehensive comparison table of all experiments."""
        logger.info(f"\n{'='*120}")
        logger.info(f"COMPREHENSIVE EXPERIMENT COST COMPARISON (Primary Model: {self.model_name})")
        logger.info(f"{'='*120}")
        
        # Enhanced header with more fields
        header = f"{'Experiment':<25} | {'Sample Size':<12} | {'Tasks':<8} | {'Datasets':<8} | {'Prompts':<8} | {'Cost ($)':<12} | {'Cost/Sample':<12} | {'Category':<15}"
        separator = f"{'-'*25} | {'-'*12} | {'-'*8} | {'-'*8} | {'-'*8} | {'-'*12} | {'-'*12} | {'-'*15}"
        
        logger.info(header)
        logger.info(separator)
        
        # Sort by category and then by cost
        sorted_experiments = sorted(estimates.items(), key=lambda x: (
            self._get_experiment_category(x[0]),
            x[1]['total_cost']
        ))
        
        for exp_name, estimate in sorted_experiments:
            config = estimate['configuration']
            sample_size = config['sample_size']
            num_tasks = len(config['tasks'])
            num_datasets = len(config['datasets'])
            num_prompts = len(config['prompt_types'])
            cost = estimate['total_cost']
            cost_per_sample = cost / sample_size if sample_size > 0 else 0
            category = self._get_experiment_category(exp_name)
            
            logger.info(f"{exp_name:<25} | {sample_size:<12,} | {num_tasks:<8} | {num_datasets:<8} | {num_prompts:<8} | ${cost:<11.4f} | ${cost_per_sample:<11.6f} | {category:<15}")
        
        logger.info(f"{'='*120}")
        
        # Add model comparison summary
        self._print_model_comparison_summary(estimates)
        
        # Add cost breakdown by category
        self._print_cost_breakdown_by_category(estimates)
    
    def _get_experiment_category(self, exp_name: str) -> str:
        """Categorize experiments for better organization."""
        if 'quick' in exp_name or 'development' in exp_name:
            return 'Development'
        elif 'thesis_' in exp_name:
            return 'Thesis Scale'
        elif 'comprehensive' in exp_name:
            return 'Comprehensive'
        else:
            return 'Standard'
    
    def _print_model_comparison_summary(self, estimates: Dict):
        """Print model comparison summary for key experiments."""
        logger.info(f"\n{'='*80}")
        logger.info(f"MODEL COST COMPARISON FOR KEY EXPERIMENTS")
        logger.info(f"{'='*80}")
        
        # Compare models for thesis-scale experiments
        key_experiments = ['thesis_chatgpt_evaluation', 'thesis_prompt_comparison', 'thesis_sota_comparison']
        models = ['gpt-4.1-mini', 'o1-mini', 'gpt-4o', 'o4-mini']
        
        for exp_name in key_experiments:
            if exp_name in estimates:
                logger.info(f"\n{exp_name.replace('_', ' ').title()}:")
                config = estimates[exp_name]['configuration']
                
                header = f"{'Model':<15} | {'Total Cost':<12} | {'Cost/Sample':<12} | {'Savings vs GPT-4':<18}"
                logger.info(header)
                logger.info(f"{'-'*15} | {'-'*12} | {'-'*12} | {'-'*18}")
                
                model_costs = {}
                original_model = self.model_name
                
                for model in models:
                    self.model_name = model
                    estimate = self.estimate_experiment_cost(config)
                    model_costs[model] = estimate['total_cost']
                
                # Sort by cost
                sorted_models = sorted(model_costs.items(), key=lambda x: x[1])
                
                # Calculate savings vs most expensive model
                max_cost = max(model_costs.values())
                
                for model, cost in sorted_models:
                    cost_per_sample = cost / config['sample_size']
                    savings = ((max_cost - cost) / max_cost) * 100
                    logger.info(f"{model:<15} | ${cost:<11.4f} | ${cost_per_sample:<11.6f} | {savings:<17.1f}%")
                
                # Restore original model
                self.model_name = original_model
    
    def _print_cost_breakdown_by_category(self, estimates: Dict):
        """Print cost breakdown by experiment category."""
        logger.info(f"\n{'='*60}")
        logger.info(f"COST BREAKDOWN BY CATEGORY")
        logger.info(f"{'='*60}")
        
        categories = {}
        for exp_name, estimate in estimates.items():
            category = self._get_experiment_category(exp_name)
            if category not in categories:
                categories[category] = []
            categories[category].append((exp_name, estimate['total_cost']))
        
        total_all_categories = 0
        for category, experiments in categories.items():
            category_total = sum(cost for _, cost in experiments)
            total_all_categories += category_total
            
            logger.info(f"\n{category}:")
            logger.info(f"  Total Cost: ${category_total:.4f}")
            logger.info(f"  Experiments: {len(experiments)}")
            
            for exp_name, cost in sorted(experiments, key=lambda x: x[1], reverse=True):
                logger.info(f"    {exp_name}: ${cost:.4f}")
        
        logger.info(f"\nGRAND TOTAL (All Categories): ${total_all_categories:.4f}")
        logger.info(f"{'='*60}")
    
    def print_budget_recommendations(self, estimates: Dict):
        """Print budget recommendations."""
        costs = [est['total_cost'] for est in estimates.values()]
        
        logger.info(f"\n{'='*60}")
        logger.info(f"BUDGET RECOMMENDATIONS")
        logger.info(f"{'='*60}")
        
        # Basic recommendations
        min_cost = min(costs)
        max_cost = max(costs)
        avg_cost = sum(costs) / len(costs)
        
        logger.info(f"Development Budget (quick tests): ${min_cost * 10:.2f}")
        logger.info(f"Research Budget (regular experiments): ${avg_cost * 5:.2f}")
        logger.info(f"Thesis Budget (comprehensive): ${max_cost * 2:.2f}")
        logger.info(f"")
        
        # Monthly budget scenarios
        logger.info(f"Monthly Budget Scenarios:")
        scenarios = [
            ("Light Development", ['quick_test', 'development'], 5),
            ("Regular Research", ['development', 'prompt_comparison'], 3),
            ("Intensive Research", ['chatgpt_evaluation', 'sota_comparison'], 2),
            ("Thesis Writing", ['comprehensive_thesis'], 1)
        ]
        
        for scenario_name, experiments, runs in scenarios:
            monthly = self.estimate_monthly_budget(experiments, runs)
            logger.info(f"  {scenario_name}: ${monthly['total_monthly_cost']:.2f}/month")
        
        logger.info(f"")
        logger.info(f"Safety Buffer: Add 20-30% to estimates for:")
        logger.info(f"  - API rate limiting retries")
        logger.info(f"  - Experimental iterations")
        logger.info(f"  - Unexpected token usage")
        
        logger.info(f"{'='*60}")

def main():
    """Main entry point."""
    parser = argparse.ArgumentParser(description="Estimate costs for factuality evaluation experiments")
    parser.add_argument("--experiment", type=str, help="Estimate specific experiment")
    parser.add_argument("--custom-config", type=str, help="Path to custom configuration file")
    parser.add_argument("--model", type=str, help="OpenAI model to use for estimation")
    parser.add_argument("--compare-models", action="store_true", help="Compare costs across models")
    parser.add_argument("--monthly-budget", nargs="+", help="Estimate monthly budget for experiments")
    
    args = parser.parse_args()
    
    # Initialize estimator
    config_path = args.custom_config or "config/default.yaml"
    estimator = CostEstimator(config_path)
    
    # Override model if specified
    if args.model:
        estimator.model_name = args.model
    
    if args.experiment:
        # Estimate single experiment
        predefined = estimator.estimate_predefined_experiments()
        if args.experiment in predefined:
            estimate = predefined[args.experiment]
            estimator.print_single_experiment_estimate(args.experiment, estimate)
            
            if args.compare_models:
                logger.info(f"\nModel Cost Comparison:")
                comparisons = estimator.get_model_comparison(estimate['configuration'])
                for model, comp in comparisons.items():
                    logger.info(f"  {model}: ${comp['total_cost']:.4f} (${comp['cost_per_call']:.4f} per call)")
        else:
            logger.error(f"Unknown experiment: {args.experiment}")
            logger.info(f"Available experiments: {', '.join(predefined.keys())}")
    
    elif args.monthly_budget:
        # Estimate monthly budget
        monthly = estimator.estimate_monthly_budget(args.monthly_budget)
        logger.info(f"\nMonthly Budget Estimate:")
        logger.info(f"Total: ${monthly['total_monthly_cost']:.2f}/month")
        logger.info(f"Daily Average: ${monthly['daily_average']:.2f}/day")
        logger.info(f"Weekly Average: ${monthly['weekly_average']:.2f}/week")
        
        logger.info(f"\nBreakdown:")
        for exp, details in monthly['breakdown'].items():
            logger.info(f"  {exp}: ${details['monthly_cost']:.2f} ({details['runs_per_month']} runs @ ${details['cost_per_run']:.2f} each)")
    
    else:
        # Full comparison
        estimates = estimator.estimate_predefined_experiments()
        estimator.print_comparison_table(estimates)
        estimator.print_budget_recommendations(estimates)
        
        logger.info(f"\nFor detailed estimates, use:")
        logger.info(f"  python scripts/estimate_costs.py --experiment <experiment_name>")
        logger.info(f"  python scripts/estimate_costs.py --compare-models --experiment <experiment_name>")
        logger.info(f"  python scripts/estimate_costs.py --monthly-budget <exp1> <exp2> ...")

if __name__ == "__main__":
    main()
