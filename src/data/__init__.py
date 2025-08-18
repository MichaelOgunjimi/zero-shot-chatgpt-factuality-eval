"""
Data Module for Factuality Evaluation
====================================

This module provides data loading and preprocessing capabilities for the
Frank and SummEval factuality evaluation datasets.

Author: Michael Ogunjimi
Institution: University of Manchester
"""

from pathlib import Path
from typing import Union

# Core data structures
from .loaders import (
    FactualityExample,
    DatasetLoader,
    FrankLoader,
    SummEvalLoader,
    quick_load_dataset,
    quick_load_balanced_dataset,
    get_available_datasets,
    get_dataset_info,
)

# Data processing
from .processor import DataProcessor

# Preprocessing functionality
from .preprocessors import (
    ProcessedExample,
    BasePreprocessor,
    EntailmentPreprocessor,
    RankingPreprocessor,
    RatingPreprocessor,
    TaskPreprocessorFactory,
    preprocess_for_task,
    validate_example_format,
    create_dataset_specific_preprocessor,
)

def process_all_datasets(base_data_dir: str = "data"):
    """Convenience function to process all datasets.
    
    Args:
        base_data_dir: Base directory for data storage
        
    Returns:
        Dictionary with processing results
    """
    processor = DataProcessor(base_data_dir)
    return processor.process_all_datasets()

# Package metadata
__version__ = "1.0.0"
__author__ = "Michael Ogunjimi"

# Export public API
__all__ = [
    # Core data structures
    "FactualityExample",
    "ProcessedExample",
    
    # Dataset loaders
    "DatasetLoader",
    "FrankLoader", 
    "SummEvalLoader",
    
    # Data processing
    "DataProcessor",
    "process_all_datasets",
    
    # Preprocessors
    "BasePreprocessor",
    "EntailmentPreprocessor",
    "RankingPreprocessor", 
    "RatingPreprocessor",
    "TaskPreprocessorFactory",
    
    "quick_load_dataset",
    "preprocess_for_task",
    "get_available_datasets",
    "get_dataset_info",
    "validate_example_format",
    "create_dataset_specific_preprocessor",
]
