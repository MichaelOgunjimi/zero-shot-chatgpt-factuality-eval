"""
ChatGPT Factuality Evaluation Package
===================================

Comprehensive system for evaluating ChatGPT's factuality assessment
capabilities across three core tasks: entailment inference, summary
ranking, and consistency rating.

This package provides academic-quality implementations suitable for
MSc thesis research and publication.

Author: Michael Ogunjimi
Institution: University of Manchester
Course: MSc AI
"""

# Package metadata
__version__ = "1.0.0"
__author__ = "Michael Ogunjimi"
__email__ = "michael.ogunjimi@postgrad.manchester.ac.uk"
__institution__ = "University of Manchester"
__course__ = "MSc AI"
__description__ = "ChatGPT Factuality Evaluation for Text Summarization"

from pathlib import Path

from .data import quick_load_dataset
from .tasks import create_task, get_supported_tasks
# Core imports for convenience
from .utils.config import load_config, get_config, setup_reproducibility
from .utils.logging import get_logger, setup_experiment_logger

# Make common functionality easily accessible
__all__ = [
    # Configuration
    "load_config",
    "get_config",
    "setup_reproducibility",
    # Logging
    "get_logger",
    "setup_experiment_logger",
    # Tasks
    "create_task",
    "get_supported_tasks",
    # Data loading
    "quick_load_dataset",
    # Package info
    "__version__",
    "__author__",
    "__description__",
]


def get_package_info():
    """Get comprehensive package information."""
    return {
        "name": "chatgpt-factuality-eval",
        "version": __version__,
        "author": __author__,
        "email": __email__,
        "institution": __institution__,
        "course": __course__,
        "description": __description__,
        "supported_tasks": [
            "entailment_inference",
            "summary_ranking",
            "consistency_rating",
        ],
        "supported_datasets": ["cnn_dailymail", "xsum"],
        "prompt_types": ["zero_shot", "chain_of_thought"],
    }


def quick_setup():
    """Quick setup function for getting started."""
    print(f"ChatGPT Factuality Evaluation v{__version__}")
    print(f"Author: {__author__}")
    print(f"Institution: {__institution__}")
    print()
    print("Quick Start:")
    print("1. Set OPENAI_API_KEY environment variable")
    print("2. Load configuration: config = load_config()")
    print("3. Create task: task = create_task('entailment_inference')")
    print("4. Load data: examples = quick_load_dataset('cnn_dailymail')")
    print("5. Run evaluation: results = await task.process_examples(examples)")
    print()
    print("For detailed documentation, see docs/ folder")


# Package validation
def validate_package_setup():
    """Validate that the package is properly set up."""
    import os

    validation_results = {"status": "success", "warnings": [], "errors": []}

    # Check environment variables
    if not os.getenv("OPENAI_API_KEY"):
        validation_results["errors"].append(
            "OPENAI_API_KEY environment variable not set"
        )
        validation_results["status"] = "error"

    # Check if required directories exist
    required_dirs = ["data", "results", "logs", "config"]
    for dir_name in required_dirs:
        if not Path(dir_name).exists():
            validation_results["warnings"].append(
                f"Directory '{dir_name}' does not exist"
            )

    # Try importing key modules
    try:
        from . import tasks, utils, data, prompts, llm_clients, baselines

        validation_results["modules_imported"] = True
    except ImportError as e:
        validation_results["errors"].append(f"Failed to import modules: {e}")
        validation_results["status"] = "error"

    return validation_results


# Show info when package is imported
if __name__ != "__main__":
    import logging

    logger = logging.getLogger(__name__)
    logger.info(f"ChatGPT Factuality Evaluation package v{__version__} loaded")
