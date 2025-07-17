"""
SOTA Baselines Package for ChatGPT Factuality Evaluation
=======================================================

This package provides state-of-the-art baseline implementations for factuality
evaluation, specifically designed for comparison with ChatGPT's performance
across the three core factuality tasks.

Available Baselines:
- FactCC: BERT-based factual consistency classifier
- BERTScore: Contextual embedding-based semantic similarity
- ROUGE: N-gram overlap metrics

Author: Michael Ogunjimi
Institution: University of Manchester
Course: MSc AI
"""

# Core baseline classes
from .sota_metrics import (
    SOTABaseline,
    BaselineResult,
    FactCCBaseline,
    BERTScoreBaseline,
    ROUGEBaseline,
    BaselineComparator,
)

# Factory functions and utilities
from .sota_metrics import (
    create_baseline,
    get_available_baselines,
    adapt_data_for_baseline,
    compare_with_chatgpt,
    generate_baseline_report,
)

# Convenience imports for common usage patterns
__all__ = [
    # Core Classes
    "SOTABaseline",
    "BaselineResult",
    "FactCCBaseline",
    "BERTScoreBaseline",
    "ROUGEBaseline",
    "BaselineComparator",

    # Factory Functions
    "create_baseline",
    "get_available_baselines",

    # Evaluation Functions
    "adapt_data_for_baseline",
    "compare_with_chatgpt",
    "generate_baseline_report",

    # Convenience Functions
    "create_all_baselines",
    "evaluate_baseline_on_examples",
    "quick_baseline_comparison",
]

# Module metadata
__version__ = "1.0.0"
__supported_baselines__ = ["factcc", "bertscore", "rouge"]
__supported_tasks__ = ["entailment_inference", "summary_ranking", "consistency_rating"]


def create_all_baselines(config=None):
    """
    Create instances of all available baselines.

    Args:
        config: Optional configuration dictionary

    Returns:
        Dictionary mapping baseline names to initialized instances
    """
    baselines = {}

    for baseline_name in get_available_baselines():
        try:
            baselines[baseline_name] = create_baseline(baseline_name, config)
        except Exception as e:
            from ..utils.logging import get_logger
            logger = get_logger(__name__)
            logger.warning(f"Failed to create baseline {baseline_name}: {e}")

    return baselines


def evaluate_baseline_on_examples(baseline_name, examples, task_name, config=None):
    """
    Evaluate a specific baseline on a list of examples.

    Args:
        baseline_name: Name of the baseline to use
        examples: List of DataExample objects
        task_name: Name of the factuality task
        config: Optional configuration dictionary

    Returns:
        List of BaselineResult objects
    """
    from ..utils.logging import get_logger
    logger = get_logger(__name__)

    # Create baseline instance
    baseline = create_baseline(baseline_name, config)

    # Adapt data if needed
    adapted_examples = adapt_data_for_baseline(examples, task_name)

    # Evaluate examples
    results = []
    logger.info(f"Evaluating {len(adapted_examples)} examples with {baseline_name}")

    for example in adapted_examples:
        try:
            result = baseline.evaluate_example(example, task_name)
            results.append(result)
        except Exception as e:
            logger.error(f"Failed to evaluate example {example.example_id}: {e}")

    logger.info(f"Completed evaluation: {len(results)}/{len(adapted_examples)} successful")
    return results


def quick_baseline_comparison(chatgpt_results, examples, task_name, config=None):
    """
    Quick comparison of ChatGPT results against all available baselines.

    Args:
        chatgpt_results: List of ChatGPT results
        examples: List of DataExample objects
        task_name: Name of the factuality task
        config: Optional configuration dictionary

    Returns:
        Dictionary with comparison results for each baseline
    """
    from ..utils.logging import get_logger
    logger = get_logger(__name__)

    logger.info(f"Running quick comparison for task: {task_name}")

    # Evaluate all baselines
    baseline_results = {}

    for baseline_name in get_available_baselines():
        try:
            results = evaluate_baseline_on_examples(
                baseline_name, examples, task_name, config
            )
            baseline_results[baseline_name] = results
            logger.info(f"Completed {baseline_name}: {len(results)} results")
        except Exception as e:
            logger.error(f"Failed to evaluate {baseline_name}: {e}")

    # Compare with ChatGPT
    if baseline_results:
        comparison = compare_with_chatgpt(
            chatgpt_results, baseline_results, task_name, config
        )
        logger.info("Baseline comparison completed")
        return comparison
    else:
        logger.warning("No baseline results available for comparison")
        return {}


def get_baseline_info():
    """
    Get comprehensive information about available baselines.

    Returns:
        Dictionary with baseline information
    """
    return {
        "supported_baselines": __supported_baselines__,
        "supported_tasks": __supported_tasks__,
        "version": __version__,
        "baseline_descriptions": {
            "factcc": "BERT-based factual consistency classifier",
            "bertscore": "Contextual embedding-based semantic similarity",
            "rouge": "N-gram overlap metrics (ROUGE-1, ROUGE-2, ROUGE-L)"
        },
        "task_support": {
            "factcc": ["entailment_inference", "consistency_rating"],
            "bertscore": ["entailment_inference", "summary_ranking", "consistency_rating"],
            "rouge": ["summary_ranking", "consistency_rating"]
        }
    }


# Validation function
def validate_baseline_setup():
    """
    Validate that baseline dependencies are properly installed.

    Returns:
        Dictionary with validation results
    """
    from ..utils.logging import get_logger
    logger = get_logger(__name__)

    validation_results = {
        "status": "success",
        "available_baselines": [],
        "missing_dependencies": [],
        "errors": []
    }

    # Test each baseline
    for baseline_name in __supported_baselines__:
        try:
            baseline = create_baseline(baseline_name)
            validation_results["available_baselines"].append(baseline_name)
            logger.info(f"Baseline {baseline_name} available")
        except ImportError as e:
            validation_results["missing_dependencies"].append({
                "baseline": baseline_name,
                "error": str(e)
            })
            validation_results["status"] = "partial"
        except Exception as e:
            validation_results["errors"].append({
                "baseline": baseline_name,
                "error": str(e)
            })
            validation_results["status"] = "error"

    return validation_results


# Package initialization logging
import logging
logger = logging.getLogger(__name__)
logger.info(f"SOTA Baselines package initialized: {len(__supported_baselines__)} baselines available")