"""
Tasks Module for ChatGPT Factuality Evaluation
============================================

Complete implementation of the three core factuality evaluation tasks:
1. Binary Entailment Inference
2. Summary Ranking
3. Consistency Rating (0-100 scale)

This module provides the core task implementations that combine prompts,
LLM clients, and evaluation logic for comprehensive factuality assessment.

Author: Michael Ogunjimi
Institution: University of Manchester
Course: MSc AI
"""

from .base_task import (
    # Abstract base class
    BaseFactualityTask,
    # Data structures
    TaskResult,
    TaskExample,
    TaskConfig,
    # Utility functions
    validate_task_config,
    create_task_example,
)
from .consistency_rating import (
    ConsistencyRatingTask,
    RatingResult,
    RatingMetrics,
)
from .entailment_inference import (
    EntailmentInferenceTask,
    EntailmentResult,
    BinaryClassificationMetrics,
)
from .summary_ranking import (
    SummaryRankingTask,
    RankingResult,
    RankingMetrics,
)

__all__ = [
    # Base classes
    "BaseFactualityTask",
    "TaskResult",
    "TaskExample",
    "TaskConfig",
    # Task implementations
    "EntailmentInferenceTask",
    "SummaryRankingTask",
    "ConsistencyRatingTask",
    # Result types
    "EntailmentResult",
    "RankingResult",
    "RatingResult",
    # Metrics
    "BinaryClassificationMetrics",
    "RankingMetrics",
    "RatingMetrics",
    # Utility functions
    "validate_task_config",
    "create_task_example",
    "create_task",
    "get_supported_tasks",
    "get_task_info",
]

# Module metadata
__version__ = "1.0.0"
__author__ = "Michael Ogunjimi"
__email__ = "michael.ogunjimi@postgrad.manchester.ac.uk"
__institution__ = "University of Manchester"
__course__ = "MSc AI"

# Supported factuality evaluation tasks
SUPPORTED_TASKS = ["entailment_inference", "summary_ranking", "consistency_rating"]

# Task descriptions for academic documentation
TASK_DESCRIPTIONS = {
    "entailment_inference": {
        "name": "Binary Entailment Inference",
        "description": "Determine whether a summary is factually consistent (ENTAILMENT) or inconsistent (CONTRADICTION) with the source document",
        "output_format": "Binary classification (0/1)",
        "evaluation_metrics": ["Accuracy", "Precision", "Recall", "F1-Score"],
        "use_cases": ["Factual consistency detection", "Hallucination identification"],
    },
    "summary_ranking": {
        "name": "Summary Ranking by Factual Consistency",
        "description": "Rank multiple summaries of the same source document by their factual consistency",
        "output_format": "Ranked list (1=most consistent)",
        "evaluation_metrics": ["Kendall's τ", "Spearman's ρ", "NDCG"],
        "use_cases": ["Summary quality assessment", "Model comparison"],
    },
    "consistency_rating": {
        "name": "Factual Consistency Rating",
        "description": "Rate the factual consistency of a summary on a 0-100 scale",
        "output_format": "Numerical rating (0-100)",
        "evaluation_metrics": ["Pearson correlation", "MAE", "RMSE"],
        "use_cases": ["Fine-grained assessment", "Human correlation analysis"],
    },
}


def create_task(task_type: str, config: dict = None, **kwargs) -> BaseFactualityTask:
    """
    Factory function to create task instances.

    Args:
        task_type: Type of task to create
        config: Configuration dictionary
        **kwargs: Additional task-specific parameters

    Returns:
        Initialized task instance

    Raises:
        ValueError: If task type is not supported
    """
    if task_type not in SUPPORTED_TASKS:
        raise ValueError(
            f"Unsupported task type: {task_type}. Supported: {SUPPORTED_TASKS}"
        )

    if task_type == "entailment_inference":
        return EntailmentInferenceTask(config=config, **kwargs)
    elif task_type == "summary_ranking":
        return SummaryRankingTask(config=config, **kwargs)
    elif task_type == "consistency_rating":
        return ConsistencyRatingTask(config=config, **kwargs)
    else:
        raise ValueError(f"Unknown task type: {task_type}")


def get_supported_tasks() -> list:
    """
    Get list of supported task types.

    Returns:
        List of supported task type strings
    """
    return SUPPORTED_TASKS.copy()


def get_task_info(task_type: str = None) -> dict:
    """
    Get information about tasks.

    Args:
        task_type: Specific task to get info for, or None for all tasks

    Returns:
        Dictionary with task information

    Raises:
        ValueError: If task_type is specified but not supported
    """
    if task_type is None:
        return {
            "supported_tasks": SUPPORTED_TASKS,
            "task_descriptions": TASK_DESCRIPTIONS,
            "total_tasks": len(SUPPORTED_TASKS),
            "module_version": __version__,
        }

    if task_type not in SUPPORTED_TASKS:
        raise ValueError(f"Unsupported task type: {task_type}")

    return TASK_DESCRIPTIONS[task_type]


def validate_task_setup() -> dict:
    """
    Validate that all tasks can be properly instantiated.

    Returns:
        Dictionary with validation results
    """
    validation_results = {
        "overall_status": "success",
        "task_status": {},
        "errors": [],
        "warnings": [],
    }

    for task_type in SUPPORTED_TASKS:
        try:
            # Try to create each task
            task = create_task(task_type)
            validation_results["task_status"][task_type] = "success"
        except Exception as e:
            validation_results["task_status"][task_type] = "failed"
            validation_results["errors"].append(f"{task_type}: {str(e)}")
            validation_results["overall_status"] = "error"

    return validation_results


def get_task_requirements() -> dict:
    """
    Get requirements for running factuality evaluation tasks.

    Returns:
        Dictionary with requirements information
    """
    return {
        "required_components": [
            "PromptManager (src.prompts)",
            "OpenAIClient (src.llm_clients)",
            "Configuration system (src.utils.config)",
            "Logging system (src.utils.logging)",
        ],
        "required_environment_variables": ["OPENAI_API_KEY"],
        "required_datasets": [
            "CNN/DailyMail (for news domain evaluation)",
            "XSum (for abstractive summarization)",
        ],
        "computational_requirements": {
            "memory": "4GB+ RAM recommended",
            "storage": "1GB+ for datasets and results",
            "network": "Stable internet for OpenAI API calls",
        },
        "budget_considerations": {
            "api_costs": "Varies by model and usage",
            "recommended_daily_budget": "$10-50 for research",
            "cost_per_example": "$0.001-0.01 depending on model",
        },
    }


# Quick setup verification
def quick_setup_check() -> bool:
    """
    Quick check if the tasks module is properly set up.

    Returns:
        True if basic setup is working
    """
    try:
        # Import required dependencies
        from ..prompts import create_prompt_manager
        from ..llm_clients import create_openai_client
        from ..utils.config import get_config
        from ..utils.logging import get_logger

        # Check if basic task creation works
        task = create_task("entailment_inference")

        return True
    except Exception:
        return False


# Performance optimization hints
PERFORMANCE_TIPS = {
    "batch_processing": "Process examples in batches to optimize API usage",
    "caching": "Cache formatted prompts and responses when possible",
    "rate_limiting": "Use appropriate rate limits to avoid API errors",
    "cost_optimization": "Consider using GPT-4.1 Mini for best cost-performance balance",
    "parallelization": "Use async processing for multiple examples",
    "memory_management": "Clear caches periodically for long-running experiments",
}


def get_performance_tips() -> dict:
    """Get performance optimization tips for task execution."""
    return PERFORMANCE_TIPS.copy()
