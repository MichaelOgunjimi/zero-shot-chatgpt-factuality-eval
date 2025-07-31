"""
Batch Processing Experiments Package
===================================

Complete batch processing experimental suite for ChatGPT factuality evaluation.
Provides cost-effective large-scale evaluation using OpenAI's Batch API with
comprehensive monitoring, analysis, and reporting capabilities.

This package contains:
- BatchChatGPTEvaluationExperiment: Main evaluation experiment
- BatchPromptComparisonExperiment: Zero-shot vs Chain-of-Thought comparison
- BatchSOTAComparisonExperiment: Comparison with traditional metrics
- BatchMasterExperimentRunner: Master orchestrator for all experiments

Author: Michael Ogunjimi
Institution: University of Manchester
Course: MSc AI
"""

from .batch_run_chatgpt_evaluation import BatchChatGPTEvaluationExperiment
from .batch_sota_comparison import BatchSOTAComparisonExperiment
from .batch_prompt_comparison import BatchPromptComparisonExperiment
from .batch_sota_comparison import BatchSOTAComparisonExperiment
from .batch_run_all_experiments import BatchMasterExperimentRunner

__all__ = [
    "BatchChatGPTEvaluationExperiment",
    "BatchPromptComparisonExperiment", 
    "BatchSOTAComparisonExperiment",
    "BatchMasterExperimentRunner"
]

__version__ = "1.0.0"
__author__ = "Michael Ogunjimi"
__email__ = "michael.ogunjimi@postgrad.manchester.ac.uk"
__institution__ = "University of Manchester"
__course__ = "MSc AI"

# Package-level configuration for batch experiments
BATCH_EXPERIMENT_CONFIG = {
    "default_model": "gpt-4.1-mini",
    "default_tier": "tier2",
    "cost_savings_target": 0.5,  # 50% savings vs sync
    "processing_timeout": 86400,  # 24 hours
    "monitoring_interval": 60,    # 1 minute
    "quick_test_samples": 30,
    "full_test_samples": 300
}

# Experiment descriptions for documentation
EXPERIMENT_DESCRIPTIONS = {
    "batch_chatgpt_evaluation": {
        "name": "Batch ChatGPT Factuality Evaluation",
        "description": "Core performance assessment across factuality tasks using batch processing",
        "estimated_duration": "1-2 hours (including batch processing)",
        "estimated_cost": "$2-8 depending on sample sizes",
        "key_outputs": [
            "Task performance metrics",
            "Success rate analysis", 
            "Cost optimization results",
            "Comprehensive evaluation report"
        ]
    },
    "batch_prompt_comparison": {
        "name": "Batch Prompt Strategy Comparison",
        "description": "Statistical comparison of zero-shot vs chain-of-thought prompting",
        "estimated_duration": "2-3 hours (including batch processing)",
        "estimated_cost": "$4-12 depending on sample sizes",
        "key_outputs": [
            "Performance improvement analysis",
            "Statistical significance testing",
            "Effect size calculations",
            "Cost-benefit analysis"
        ]
    },
    "batch_sota_comparison": {
        "name": "Batch SOTA Baseline Comparison", 
        "description": "Correlation analysis with traditional factuality metrics",
        "estimated_duration": "1-2 hours (including batch processing)",
        "estimated_cost": "$2-6 depending on sample sizes",
        "key_outputs": [
            "Correlation analysis with baselines",
            "Method ranking and agreement",
            "Performance comparison",
            "Validation against traditional metrics"
        ]
    },
    "batch_master_suite": {
        "name": "Complete Batch Experimental Suite",
        "description": "All three experiments with consolidated analysis and reporting",
        "estimated_duration": "3-6 hours (including batch processing)",
        "estimated_cost": "$8-25 depending on sample sizes", 
        "key_outputs": [
            "Comprehensive cross-experiment analysis",
            "Master performance dashboard",
            "Consolidated cost analysis",
            "Thesis-ready reports and visualizations"
        ]
    }
}

def get_experiment_info(experiment_type: str = None) -> dict:
    """
    Get information about available batch experiments.
    
    Args:
        experiment_type: Specific experiment to get info for (None for all)
        
    Returns:
        Experiment information dictionary
    """
    if experiment_type and experiment_type in EXPERIMENT_DESCRIPTIONS:
        return EXPERIMENT_DESCRIPTIONS[experiment_type]
    
    return {
        "available_experiments": list(EXPERIMENT_DESCRIPTIONS.keys()),
        "descriptions": EXPERIMENT_DESCRIPTIONS,
        "package_info": {
            "name": "Batch Processing Experiments",
            "version": __version__,
            "author": __author__,
            "institution": __institution__,
            "purpose": "Cost-effective large-scale factuality evaluation using batch processing"
        },
        "configuration": BATCH_EXPERIMENT_CONFIG
    }

def estimate_experiment_cost(
    experiment_type: str,
    sample_size: int = 300,
    model: str = "gpt-4.1-mini"
) -> dict:
    """
    Estimate cost for running a batch experiment.
    
    Args:
        experiment_type: Type of experiment to estimate
        sample_size: Number of examples per dataset
        model: Model to use for estimation
        
    Returns:
        Cost estimation dictionary
    """
    # Model pricing (cost per 1k tokens)
    model_pricing = {
        "gpt-4o-mini": {"input": 0.00015, "output": 0.0006},
        "gpt-4.1-mini": {"input": 0.0004, "output": 0.0016},
        "gpt-4o": {"input": 0.0025, "output": 0.01},
        "o1-mini": {"input": 0.003, "output": 0.012}
    }
    
    pricing = model_pricing.get(model, model_pricing["gpt-4.1-mini"])
    batch_discount = 0.5  # 50% savings
    
    # Estimate tokens per example
    avg_prompt_tokens = 800  # Estimated average
    avg_completion_tokens = 150  # Estimated average
    
    # Calculate base cost per example
    prompt_cost = (avg_prompt_tokens / 1000) * pricing["input"]
    completion_cost = (avg_completion_tokens / 1000) * pricing["output"]
    cost_per_example = (prompt_cost + completion_cost) * (1 - batch_discount)
    
    # Experiment-specific multipliers
    multipliers = {
        "batch_chatgpt_evaluation": 3 * 2,  # 3 tasks × 2 datasets
        "batch_prompt_comparison": 3 * 2 * 2,  # 3 tasks × 2 datasets × 2 prompt types
        "batch_sota_comparison": 2 * 2,  # 2 tasks × 2 datasets (summary_ranking excluded)
        "batch_master_suite": 3 * 2 + 3 * 2 * 2 + 2 * 2  # Sum of all experiments
    }
    
    multiplier = multipliers.get(experiment_type, 1)
    estimated_cost = cost_per_example * sample_size * multiplier
    
    return {
        "experiment_type": experiment_type,
        "sample_size": sample_size,
        "model": model,
        "estimated_cost": estimated_cost,
        "cost_per_example": cost_per_example,
        "batch_discount": batch_discount,
        "sync_cost_estimate": estimated_cost / (1 - batch_discount),
        "estimated_savings": estimated_cost * batch_discount / (1 - batch_discount),
        "breakdown": {
            "prompt_cost_per_example": prompt_cost * (1 - batch_discount),
            "completion_cost_per_example": completion_cost * (1 - batch_discount),
            "total_examples": sample_size * multiplier
        }
    }

def print_experiment_guide():
    """Print comprehensive experiment guide."""
    print("🚀 BATCH PROCESSING EXPERIMENTS GUIDE")
    print("=" * 50)
    print()
    
    for exp_type, desc in EXPERIMENT_DESCRIPTIONS.items():
        print(f"📊 {desc['name']}")
        print(f"   Description: {desc['description']}")
        print(f"   Duration: {desc['estimated_duration']}")
        print(f"   Cost: {desc['estimated_cost']}")
        print(f"   Key Outputs:")
        for output in desc['key_outputs']:
            print(f"     • {output}")
        print()
    
    print("💡 USAGE EXAMPLES:")
    print("   # Run individual experiments")
    print("   python experiments/batch_processing/batch_run_chatgpt_evaluation.py --quick-test")
    print("   python experiments/batch_processing/batch_run_prompt_comparison.py --model gpt-4.1-mini")
    print("   python experiments/batch_processing/batch_run_sota_comparison.py --sample-size 500")
    print()
    print("   # Run complete suite")
    print("   python experiments/batch_processing/batch_run_all_experiments.py --full-suite")
    print("   python experiments/batch_processing/batch_run_all_experiments.py --quick-test --experiments chatgpt_evaluation prompt_comparison")
    print()
    print("📁 OUTPUT STRUCTURE:")
    print("   results/experiments/batch_processing/[experiment_name]/")
    print("   ├── [individual_experiment_folders]/")
    print("   ├── master_analysis/")
    print("   ├── batch_master_results.json")
    print("   └── README.md")

if __name__ == "__main__":
    print_experiment_guide()