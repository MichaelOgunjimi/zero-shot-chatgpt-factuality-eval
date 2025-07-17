"""
Experiments Package for ChatGPT Factuality Evaluation
=====================================================

This package contains all experimental scripts for evaluating ChatGPT's
performance on factuality assessment tasks. Each experiment can be run
both as a standalone script and as a module.

Available Experiments:
- quick_experiment_setup: Environment validation and setup testing
- prompt_comparison: Comparison of different prompting strategies
- comprehensive_prompt_comparison: Extended prompt analysis with statistical tests
- sota_comparison: Comparison with state-of-the-art baseline methods
- run_chatgpt_evaluation: Main ChatGPT evaluation experiment
- run_all_experiments: Master runner for all experiments

Usage Examples:
    # As scripts
    python experiments/quick_experiment_setup.py --complete-validation
    python experiments/prompt_comparison.py --config config/default.yaml
    
    # As modules
    python -m experiments.quick_experiment_setup --complete-validation
    python -m experiments.prompt_comparison --config config/default.yaml

Author: Michael Ogunjimi
Institution: University of Manchester
Course: MSc AI
"""

__version__ = "1.0.0"
__author__ = "Michael Ogunjimi"
__email__ = "michael.ogunjimi@postgrad.manchester.ac.uk"
__institution__ = "University of Manchester"
__course__ = "MSc AI"

# Available experiment modules
AVAILABLE_EXPERIMENTS = [
    "quick_experiment_setup",
    "prompt_comparison", 
    "comprehensive_prompt_comparison",
    "sota_comparison",
    "run_chatgpt_evaluation",
    "run_all_experiments"
]

# Experiment descriptions
EXPERIMENT_DESCRIPTIONS = {
    "quick_experiment_setup": {
        "name": "Quick Experiment Setup and Validation",
        "description": "Validates experimental framework and runs quick tests",
        "runtime": "~5-10 minutes",
        "purpose": "Environment setup validation and component testing"
    },
    "prompt_comparison": {
        "name": "Prompt Design Comparison", 
        "description": "Compares zero-shot vs chain-of-thought prompting",
        "runtime": "~30-60 minutes",
        "purpose": "Evaluate impact of prompt engineering on performance"
    },
    "comprehensive_prompt_comparison": {
        "name": "Comprehensive Prompt Analysis",
        "description": "Extended prompt comparison with statistical analysis",
        "runtime": "~2-4 hours", 
        "purpose": "Detailed prompt strategy evaluation with thesis-quality analysis"
    },
    "sota_comparison": {
        "name": "SOTA Baseline Comparison",
        "description": "Compares ChatGPT with traditional factuality metrics",
        "runtime": "~1-2 hours",
        "purpose": "Benchmark against established evaluation methods"
    },
    "run_chatgpt_evaluation": {
        "name": "Main ChatGPT Evaluation",
        "description": "Core ChatGPT factuality evaluation experiment",
        "runtime": "~1-3 hours",
        "purpose": "Primary evaluation of ChatGPT factuality assessment capabilities"
    },
    "run_all_experiments": {
        "name": "Master Experiment Runner",
        "description": "Orchestrates all experiments in sequence",
        "runtime": "~4-8 hours",
        "purpose": "Complete experimental suite for thesis"
    }
}


def get_experiment_info(experiment_name: str = None) -> dict:
    """
    Get information about available experiments.
    
    Args:
        experiment_name: Specific experiment to get info for, or None for all
        
    Returns:
        Dictionary with experiment information
    """
    if experiment_name is None:
        return {
            "available_experiments": AVAILABLE_EXPERIMENTS,
            "experiment_descriptions": EXPERIMENT_DESCRIPTIONS,
            "total_experiments": len(AVAILABLE_EXPERIMENTS),
            "package_version": __version__
        }
    
    if experiment_name not in AVAILABLE_EXPERIMENTS:
        raise ValueError(f"Unknown experiment: {experiment_name}")
        
    return EXPERIMENT_DESCRIPTIONS[experiment_name]


def get_recommended_experiment_order() -> list:
    """
    Get recommended order for running experiments.
    
    Returns:
        List of experiment names in recommended execution order
    """
    return [
        "quick_experiment_setup",  # Always run first for validation
        "run_chatgpt_evaluation",  # Core evaluation
        "prompt_comparison",       # Prompt strategy analysis
        "sota_comparison",         # Baseline comparison
        "comprehensive_prompt_comparison"  # Extended analysis (optional)
    ]


def estimate_total_runtime(experiments: list = None) -> str:
    """
    Estimate total runtime for a set of experiments.
    
    Args:
        experiments: List of experiment names, or None for all
        
    Returns:
        Estimated runtime string
    """
    if experiments is None:
        experiments = get_recommended_experiment_order()
    
    # Runtime estimates in minutes
    runtime_estimates = {
        "quick_experiment_setup": 10,
        "prompt_comparison": 45,
        "comprehensive_prompt_comparison": 180,
        "sota_comparison": 90,
        "run_chatgpt_evaluation": 120,
        "run_all_experiments": 360  # This runs others, so don't double count
    }
    
    if "run_all_experiments" in experiments:
        total_minutes = runtime_estimates["run_all_experiments"]
    else:
        total_minutes = sum(runtime_estimates.get(exp, 60) for exp in experiments)
    
    hours = total_minutes // 60
    minutes = total_minutes % 60
    
    if hours > 0:
        return f"~{hours}h {minutes}m"
    else:
        return f"~{minutes}m"
