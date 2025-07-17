"""
Data Module for ChatGPT Factuality Evaluation
=============================================

This module provides comprehensive data loading, downloading, and preprocessing
capabilities for the two core factuality evaluation datasets:
CNN/DailyMail and XSum.

Designed for academic research with robust error handling, caching,
and validation suitable for MSc thesis requirements.

Author: Michael Ogunjimi
Institution: University of Manchester
Course: MSc AI
"""

from pathlib import Path
from typing import Union

# Core data structures
from .loaders import (
    FactualityExample,
    DatasetLoader,
    CNNDailyMailLoader,
    XSumLoader,
    quick_load_dataset,
    get_available_datasets,
    validate_dataset_path,
    get_dataset_info,
    load_processed_dataset,
)

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

# Download functionality
from .downloader import (
    DatasetDownloader,
    download_datasets,
)

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
    "CNNDailyMailLoader", 
    "XSumLoader",
    
    # Preprocessors
    "BasePreprocessor",
    "EntailmentPreprocessor",
    "RankingPreprocessor", 
    "RatingPreprocessor",
    "TaskPreprocessorFactory",
    
    # Download functionality
    "DatasetDownloader",
    "download_datasets",
    
    # Main utility functions
    "quick_load_dataset",
    "preprocess_for_task",
    "get_available_datasets",
    "validate_dataset_path",
    "get_dataset_info",
    "load_processed_dataset",
    "validate_example_format",
    "create_dataset_specific_preprocessor",
]


def get_module_info():
    """Get comprehensive module information."""
    return {
        "name": "src.data",
        "version": __version__,
        "author": __author__,
        "description": "Data loading and preprocessing for factuality evaluation",
        "supported_datasets": get_available_datasets(),
        "supported_tasks": TaskPreprocessorFactory.get_supported_tasks(),
        "components": {
            "loaders": "Dataset loading with caching and validation",
            "preprocessors": "Task-specific data preprocessing",
            "downloader": "Automatic dataset downloading from HuggingFace"
        }
    }


def quick_setup_check():
    """
    Quick check if the data module is properly set up.

    Returns:
        Dictionary with setup status and recommendations
    """
    import logging
    from pathlib import Path
    
    logger = logging.getLogger(__name__)
    status = {
        "module_loaded": True,
        "datasets_available": False,
        "datasets_downloaded": {},
        "recommendations": [],
        "errors": []
    }
    
    try:
        # Check if datasets are downloaded
        for dataset_name in get_available_datasets():
            validation = validate_dataset_path(dataset_name)
            status["datasets_downloaded"][dataset_name] = validation["status"] == "success"
            
        # Check if any datasets are available
        status["datasets_available"] = any(status["datasets_downloaded"].values())
        
        # Provide recommendations
        if not status["datasets_available"]:
            status["recommendations"].append(
                "No datasets found. Run: python -m src.data.downloader --download-all"
            )
        
        missing_datasets = [
            name for name, available in status["datasets_downloaded"].items() 
            if not available
        ]
        if missing_datasets:
            status["recommendations"].append(
                f"Missing datasets: {missing_datasets}. "
                f"Run: python -m src.data.downloader --download {' '.join(missing_datasets)}"
            )
            
    except Exception as e:
        status["errors"].append(f"Setup check failed: {e}")
        logger.error(f"Data module setup check failed: {e}")
    
    return status


def download_all_datasets(development_mode: bool = False, force: bool = False):
    """
    Convenience function to download all datasets.
    
    Args:
        development_mode: Use small sample sizes for testing
        force: Force redownload even if files exist
        
    Returns:
        Dictionary with download results
    """
    return download_datasets(
        datasets=None,  # Download all
        development_mode=development_mode,
        force_redownload=force
    )


# Quick access functions for common operations
def load_dataset_for_task(
    dataset_name: str,
    task_type: str,
    split: str = "test",
    max_examples: int = None,
    use_processed: bool = True,
    data_dir: Union[str, Path] = "data",
    **preprocessor_kwargs
):
    """
    Load and preprocess dataset for specific task in one step.
    
    This function first tries to load from preprocessed data if available,
    otherwise falls back to loading raw data and preprocessing on-the-fly.
    
    Args:
        dataset_name: Name of dataset ("cnn_dailymail", "xsum")
        task_type: Task type ("entailment_inference", "summary_ranking", "consistency_rating")
        split: Dataset split to load
        max_examples: Maximum examples to load
        use_processed: Whether to try loading preprocessed data first
        data_dir: Directory containing dataset files
        **preprocessor_kwargs: Additional preprocessing arguments
        
    Returns:
        List of ProcessedExample objects ready for evaluation
        
    Example:
        >>> examples = load_dataset_for_task("cnn_dailymail", "entailment_inference", max_examples=100)
        >>> print(f"Loaded {len(examples)} processed examples")
    """
    # Try loading preprocessed data first
    if use_processed:
        try:
            from .loaders import load_processed_dataset
            return load_processed_dataset(
                dataset_name=dataset_name,
                task_type=task_type,
                split=split,
                max_examples=max_examples,
                data_dir=data_dir
            )
        except (FileNotFoundError, ImportError):
            # Fall back to on-the-fly processing
            pass
    
    # Load raw examples and process on-the-fly
    raw_examples = quick_load_dataset(
        dataset_name=dataset_name,
        split=split,
        max_examples=max_examples,
        data_dir=data_dir
    )
    
    # Create dataset-specific preprocessor
    preprocessor = create_dataset_specific_preprocessor(
        dataset_name=dataset_name,
        task_type=task_type,
        **preprocessor_kwargs
    )
    
    # Process examples
    processed_examples = []
    for example in raw_examples:
        try:
            processed = preprocessor.process_example(example)
            is_valid, _ = processed.validate_for_task()
            if is_valid:
                processed_examples.append(processed)
        except Exception:
            continue
    
    return processed_examples


def get_dataset_statistics(dataset_name: str = None):
    """
    Get comprehensive statistics about datasets.
    
    Args:
        dataset_name: Specific dataset name, or None for all datasets
        
    Returns:
        Dictionary with dataset statistics
    """
    import json
    from pathlib import Path
    
    stats = {}
    datasets_to_check = [dataset_name] if dataset_name else get_available_datasets()
    
    for name in datasets_to_check:
        dataset_stats = {
            "name": name,
            "splits": {},
            "total_examples": 0,
            "avg_source_length": 0,
            "avg_summary_length": 0,
            "status": "not_found"
        }
        
        try:
            # Check if dataset exists
            validation = validate_dataset_path(name)
            if validation["status"] != "success":
                dataset_stats["status"] = "not_downloaded"
                stats[name] = dataset_stats
                continue
            
            # Load and analyze examples
            examples = quick_load_dataset(name, max_examples=1000)  # Sample for stats
            if examples:
                dataset_stats["status"] = "available"
                dataset_stats["total_examples"] = len(examples)
                
                source_lengths = [len(ex.source) for ex in examples]
                summary_lengths = [len(ex.summary) for ex in examples]
                
                dataset_stats["avg_source_length"] = sum(source_lengths) / len(source_lengths)
                dataset_stats["avg_summary_length"] = sum(summary_lengths) / len(summary_lengths)
                dataset_stats["min_source_length"] = min(source_lengths)
                dataset_stats["max_source_length"] = max(source_lengths)
                dataset_stats["min_summary_length"] = min(summary_lengths)
                dataset_stats["max_summary_length"] = max(summary_lengths)
                
        except Exception as e:
            dataset_stats["status"] = "error"
            dataset_stats["error"] = str(e)
        
        stats[name] = dataset_stats
    
    return stats if dataset_name is None else stats.get(dataset_name, {})


# Configuration for easy access
DATASET_CONFIGS = {
    "cnn_dailymail": {
        "description": "News summarization with extractive summaries",
        "domain": "news",
        "abstractiveness": "moderate",
        "recommended_tasks": ["entailment_inference", "summary_ranking", "consistency_rating"],
        "preprocessing_defaults": {
            "max_source_length": 1024,
            "max_summary_length": 256,
        }
    },
    "xsum": {
        "description": "Abstractive summarization of BBC articles", 
        "domain": "news",
        "abstractiveness": "high",
        "recommended_tasks": ["entailment_inference", "summary_ranking", "consistency_rating"],
        "preprocessing_defaults": {
            "max_source_length": 1024,
            "max_summary_length": 128,
        }
    }
}


def get_dataset_config(dataset_name: str):
    """Get configuration information for a specific dataset."""
    return DATASET_CONFIGS.get(dataset_name, {})


def print_module_status():
    """Print a comprehensive status report of the data module."""
    print("=" * 60)
    print("ChatGPT Factuality Evaluation - Data Module Status")
    print("=" * 60)
    
    # Module info
    info = get_module_info()
    print(f"Version: {info['version']}")
    print(f"Author: {info['author']}")
    print()
    
    # Setup check
    status = quick_setup_check()
    print("Setup Status:")
    print(f"  Module loaded: {'✓' if status['module_loaded'] else '✗'}")
    print(f"  Datasets available: {'✓' if status['datasets_available'] else '✗'}")
    print()
    
    # Dataset status
    print("Dataset Status:")
    for dataset, available in status["datasets_downloaded"].items():
        status_symbol = "✓" if available else "✗"
        print(f"  {dataset}: {status_symbol}")
    print()
    
    # Recommendations
    if status["recommendations"]:
        print("Recommendations:")
        for rec in status["recommendations"]:
            print(f"  • {rec}")
        print()
    
    # Errors
    if status["errors"]:
        print("Errors:")
        for error in status["errors"]:
            print(f"  ✗ {error}")
        print()
    
    # Available functionality
    print("Available Functionality:")
    print(f"  • Datasets: {', '.join(get_available_datasets())}")
    print(f"  • Tasks: {', '.join(TaskPreprocessorFactory.get_supported_tasks())}")
    print(f"  • Components: {', '.join(info['components'].keys())}")


if __name__ == "__main__":
    """Run module status check when called directly."""
    print_module_status()