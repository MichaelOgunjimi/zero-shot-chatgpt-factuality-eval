"""
Factuality Evaluation for Automatic Text Summarisation
MSc Computer Science Thesis Project
University of Manchester 2025

A comprehensive system for evaluating factual consistency in abstractive summarisation.
"""

__version__ = "1.0.0"
__author__ = "MSc Computer Science Student"
__email__ = "student@manchester.ac.uk"
__description__ = "Factuality evaluation system for text summarisation"

# Core imports for easy access
from src.utils.config import load_config, get_device
from src.utils.logging import setup_logging, get_logger
from src.data.loaders import CNNDailyMailLoader, XSumLoader
from src.metrics.factcc import FactCCMetric
from src.metrics.rouge import ROUGEMetric
from src.evaluation.pipeline import EvaluationPipeline

# Version info
def get_version_info():
    """Get detailed version information."""
    import sys
    import torch
    import transformers

    return {
        "factuality_eval": __version__,
        "python": sys.version.split()[0],
        "torch": torch.__version__,
        "transformers": transformers.__version__,
    }

# Package-level configuration
DEFAULT_CONFIG_PATH = "config/default.yaml"
DEFAULT_LOG_LEVEL = "INFO"

__all__ = [
    "__version__",
    "__author__",
    "__email__",
    "__description__",
    "load_config",
    "get_device",
    "setup_logging",
    "get_logger",
    "CNNDailyMailLoader",
    "XSumLoader",
    "FactCCMetric",
    "ROUGEMetric",
    "EvaluationPipeline",
    "get_version_info",
]