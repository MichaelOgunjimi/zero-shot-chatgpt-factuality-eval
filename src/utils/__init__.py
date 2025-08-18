"""
Utilities Package for ChatGPT Factuality Evaluation
==================================================

Comprehensive utility functions for configuration management, logging,
and visualization specifically designed for academic research and thesis requirements.

Core Modules:
- config: Configuration management and reproducibility
- logging: Experiment tracking and cost monitoring
- visualization: Publication-quality figures and analysis

Author: Michael Ogunjimi
Institution: University of Manchester
Course: MSc AI
"""

# Configuration utilities - based on actual implementation
from .config import (
    ConfigManager,
    ConfigWrapper,
    load_config,
    get_config,
    set_global_seed,
    get_device,
    create_output_directories,
    validate_api_keys,
    setup_reproducibility,
)

# Logging utilities - based on actual implementation
from .logging import (
    ExperimentLogger,
    ProgressTracker,
    CostTracker,
    LogEntry,
    setup_experiment_logger,
    setup_basic_logging,
    get_logger,
    log_execution_time,
    log_model_loading,
    log_evaluation_results,
    log_task_context,
)

# Visualization utilities - based on actual implementation
from .visualization import (
    VisualizationEngine,
    TaskPerformanceVisualizer,
    BaselineComparisonVisualizer,
    StatisticalAnalysisVisualizer,
    CostAnalysisVisualizer,
    create_visualization_engine,
    create_dashboard_figure,
    ACADEMIC_COLORS,
    TASK_COLORS,
    BASELINE_COLORS,
)

__all__ = [
    # Configuration Management
    "ConfigManager",
    "ConfigWrapper",
    "load_config",
    "get_config",
    "set_global_seed",
    "get_device",
    "create_output_directories",
    "validate_api_keys",
    "setup_reproducibility",

    # Logging System
    "ExperimentLogger",
    "ProgressTracker",
    "CostTracker",
    "LogEntry",
    "setup_experiment_logger",
    "setup_basic_logging",
    "get_logger",
    "log_execution_time",
    "log_model_loading",
    "log_evaluation_results",
    "log_task_context",

    # Visualization System
    "VisualizationEngine",
    "TaskPerformanceVisualizer",
    "BaselineComparisonVisualizer",
    "StatisticalAnalysisVisualizer",
    "CostAnalysisVisualizer",
    "create_visualization_engine",
    "create_dashboard_figure",
    "ACADEMIC_COLORS",
    "TASK_COLORS",
    "BASELINE_COLORS",

    # Convenience Functions
    "get_utility_info",
    "validate_environment",
]

# Module metadata - matching the actual implementation
__version__ = "1.0.0"
__author__ = "Michael Ogunjimi"
__email__ = "michael.ogunjimi@postgrad.manchester.ac.uk"
__institution__ = "University of Manchester"
__course__ = "MSc AI"
__description__ = "Utilities for ChatGPT factuality evaluation research"

# Utility categories based on actual implementation
UTILITY_CATEGORIES = {
    "configuration": [
        "ConfigManager", "ConfigWrapper", "load_config", "get_config",
        "set_global_seed", "get_device", "create_output_directories",
        "validate_api_keys", "setup_reproducibility"
    ],
    "logging": [
        "ExperimentLogger", "ProgressTracker", "CostTracker", "LogEntry",
        "setup_experiment_logger", "setup_basic_logging", "get_logger"
    ],
    "visualization": [
        "VisualizationEngine", "TaskPerformanceVisualizer",
        "BaselineComparisonVisualizer", "StatisticalAnalysisVisualizer",
        "CostAnalysisVisualizer", "create_visualization_engine"
    ],
    "reproducibility": [
        "set_global_seed", "create_output_directories", "get_device",
        "setup_reproducibility"
    ]
}


def get_utility_info() -> dict:
    """
    Get comprehensive information about available utilities.

    Returns:
        Dictionary containing utility information and categories
    """
    return {
        "categories": UTILITY_CATEGORIES,
        "version": __version__,
        "author": __author__,
        "institution": __institution__,
        "description": __description__,
        "available_modules": ["config", "logging", "visualization"],
        "supported_tasks": ["entailment_inference", "summary_ranking", "consistency_rating"]
    }


def validate_environment() -> dict:
    """
    Validate the environment for running factuality evaluation experiments.

    Returns:
        Dictionary with environment validation results
    """
    import sys
    import os
    from pathlib import Path

    validation_results = {
        "python_version": sys.version,
        "platform": sys.platform,
        "working_directory": str(Path.cwd()),
        "environment_variables": {},
        "warnings": [],
        "errors": []
    }

    critical_env_vars = ["OPENAI_API_KEY"]
    for var in critical_env_vars:
        value = os.getenv(var)
        validation_results["environment_variables"][var] = {
            "set": value is not None,
            "length": len(value) if value else 0
        }
        if not value:
            validation_results["errors"].append(f"Missing required environment variable: {var}")

    optional_env_vars = ["TORCH_DEVICE", "PROJECT_NAME", "LOG_LEVEL"]
    for var in optional_env_vars:
        value = os.getenv(var)
        validation_results["environment_variables"][var] = {
            "set": value is not None,
            "value": value if value else "not_set"
        }
        if not value:
            validation_results["warnings"].append(f"Optional environment variable not set: {var}")

    if sys.version_info < (3, 8):
        validation_results["errors"].append(f"Python 3.8+ required, found {sys.version}")

    required_dirs = ["data", "results", "logs", "config"]
    for dir_name in required_dirs:
        dir_path = Path(dir_name)
        if not dir_path.exists():
            validation_results["warnings"].append(f"Directory does not exist: {dir_name}")

    # Overall status - only consider critical missing directories and env vars as warnings
    critical_warnings = [w for w in validation_results["warnings"] if "Directory does not exist" in w]
    validation_results["status"] = "error" if validation_results["errors"] else (
        "warning" if critical_warnings else "success"
    )

    return validation_results


# Package initialization
import logging

logger = logging.getLogger(__name__)
logger.debug(f"Utils package initialized: version {__version__}")

# Validate setup on import
try:
    validation = validate_environment()
    if validation["status"] == "error":
        logger.error("Critical requirements missing - check validation results")
    elif validation["status"] == "warning":
        logger.warning("Some optional requirements missing")
    else:
        logger.info("All requirements validated successfully")
except Exception as e:
    logger.warning(f"Could not validate requirements: {e}")

try:
    setup_basic_logging({"logging": {"level": "INFO"}})
except Exception:
    pass  # Ignore errors in basic logging setup