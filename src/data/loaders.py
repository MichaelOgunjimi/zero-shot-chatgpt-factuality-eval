"""
Dataset Loaders for ChatGPT Factuality Evaluation
=================================================

This module provides robust dataset loading capabilities for the three core
factuality evaluation datasets: CNN/DailyMail and XSum.

Designed for academic research with comprehensive error handling,
caching, and validation suitable for MSc thesis requirements.

Author: Michael Ogunjimi
Institution: University of Manchester
Course: MSc AI
"""

import json
import logging
import pickle
import time
from abc import ABC, abstractmethod
from dataclasses import dataclass
from datetime import datetime
from pathlib import Path
from typing import Dict, List, Optional, Union, Any, Tuple
import argparse
import sys

# Add datasets library import
try:
    from datasets import load_dataset
    HF_DATASETS_AVAILABLE = True
except ImportError:
    print("Warning: datasets library not available. Install with: pip install datasets")
    load_dataset = None
    HF_DATASETS_AVAILABLE = False

# Configure logging
logger = logging.getLogger(__name__)


@dataclass
class FactualityExample:
    """
    Standard example format for factuality evaluation tasks.

    This class provides a unified interface for all factuality evaluation
    examples across different datasets and tasks.
    """

    example_id: str
    source: str
    summary: str
    dataset_name: str
    task_type: Optional[str] = None
    human_label: Optional[Union[str, int, float, List]] = None
    metadata: Optional[Dict[str, Any]] = None

    def __post_init__(self):
        """Post-initialization validation and setup."""
        if self.metadata is None:
            self.metadata = {}

        # Add creation timestamp
        self.metadata["created_at"] = datetime.now().isoformat()

        # Validate required fields
        if not self.example_id:
            raise ValueError("example_id cannot be empty")
        if not self.source.strip():
            raise ValueError("source cannot be empty")
        if not self.summary.strip():
            raise ValueError("summary cannot be empty")

    def to_dict(self) -> Dict[str, Any]:
        """Convert to dictionary for serialization."""
        return {
            "example_id": self.example_id,
            "source": self.source,
            "summary": self.summary,
            "dataset_name": self.dataset_name,
            "task_type": self.task_type,
            "human_label": self.human_label,
            "metadata": self.metadata,
        }

    @classmethod
    def from_dict(cls, data: Dict[str, Any]) -> "FactualityExample":
        """Create instance from dictionary."""
        return cls(**data)

    def get_summary_for_binary_task(self) -> str:
        """Get summary for binary tasks (entailment, rating)."""
        return self.summary

    def get_summaries_for_ranking(self) -> List[str]:
        """Get summaries for ranking task."""
        # For ranking tasks, we need at least 2 summaries
        # Generate synthetic variants if only one summary is available
        if not self.summary:
            return []
        
        summaries = [self.summary]
        
        # Generate a second summary by creating a truncated version
        # This provides a simple comparison for ranking tasks
        words = self.summary.split()
        if len(words) > 5:
            # Create a truncated version (removes information, likely less factual)
            truncated = ' '.join(words[:max(len(words)//2, 3)])
            if not truncated.endswith('.'):
                truncated += '.'
            summaries.append(truncated)
        
        # Generate a third summary by creating a slightly modified version
        if len(words) > 10:
            # Create a version with some repetition (likely less coherent)
            modified = ' '.join(words[:len(words)//2]) + ' ' + ' '.join(words[:min(5, len(words)//3)])
            summaries.append(modified)
        
        return summaries

    def get_human_ranking_for_synthetic_summaries(self) -> List[int]:
        """Get human rankings for synthetic summaries."""
        summaries = self.get_summaries_for_ranking()
        num_summaries = len(summaries)
        
        if num_summaries < 2:
            return []
        
        # For synthetic summaries, we assume:
        # - Original summary (index 0) is best (rank 1)
        # - Truncated summary (index 1) is second (rank 2)
        # - Modified/repetitive summary (index 2) is worst (rank 3)
        rankings = list(range(1, num_summaries + 1))
        return rankings

    def validate(self) -> Tuple[bool, List[str]]:
        """
        Validate example data quality.

        Returns:
            Tuple of (is_valid, error_messages)
        """
        errors = []

        # Check text lengths
        if len(self.source) < 50:
            errors.append("Source text is too short (< 50 characters)")
        if len(self.summary) < 10:
            errors.append("Summary text is too short (< 10 characters)")

        # Check for reasonable length ratios
        if len(self.summary) > len(self.source):
            errors.append("Summary is longer than source text")

        # Check for valid characters
        if not self.source.strip() or not self.summary.strip():
            errors.append("Source or summary contains only whitespace")

        return len(errors) == 0, errors


class DatasetLoader(ABC):
    """
    Abstract base class for dataset loaders.

    Provides common functionality for loading, caching, and validating
    factuality evaluation datasets with academic research standards.
    """

    def __init__(
        self,
        data_dir: Union[str, Path] = "data",
        cache_enabled: bool = True,
        validate_examples: bool = True,
    ):
        """
        Initialize dataset loader.

        Args:
            data_dir: Base directory for data storage
            cache_enabled: Whether to enable result caching
            validate_examples: Whether to validate loaded examples
        """
        self.data_dir = Path(data_dir)
        self.cache_enabled = cache_enabled
        self.validate_examples = validate_examples

        # Create necessary directories
        self.data_dir.mkdir(exist_ok=True)
        (self.data_dir / "raw").mkdir(exist_ok=True)
        (self.data_dir / "processed").mkdir(exist_ok=True)
        (self.data_dir / "cache").mkdir(exist_ok=True)

        self.logger = logging.getLogger(f"{__name__}.{self.__class__.__name__}")

    @abstractmethod
    def dataset_name(self) -> str:
        """Return the name of the dataset."""
        pass

    @abstractmethod
    def load_raw_data(self, split: str = "test") -> List[Dict[str, Any]]:
        """Load raw dataset from files."""
        pass

    @abstractmethod
    def process_example(self, raw_example: Dict[str, Any]) -> FactualityExample:
        """Convert raw example to FactualityExample format."""
        pass

    def load_dataset(
        self,
        split: str = "test",
        max_examples: Optional[int] = None,
        use_cache: bool = True,
    ) -> List[FactualityExample]:
        """
        Load and process dataset examples.

        Args:
            split: Dataset split to load ("train", "test", "validation")
            max_examples: Maximum number of examples to load
            use_cache: Whether to use cached results

        Returns:
            List of processed FactualityExample objects

        Raises:
            FileNotFoundError: If dataset files are not found
            ValueError: If dataset processing fails
        """
        cache_key = f"{self.dataset_name()}_{split}_{max_examples or 'all'}"
        cache_path = self.data_dir / "cache" / f"{cache_key}.pkl"

        # Try loading from cache
        if use_cache and self.cache_enabled and cache_path.exists():
            try:
                with open(cache_path, "rb") as f:
                    cached_data = pickle.load(f)

                self.logger.info(
                    f"Loaded {len(cached_data)} examples from cache: {cache_path}"
                )
                return cached_data

            except Exception as e:
                self.logger.warning(f"Failed to load cache {cache_path}: {e}")

        # Load and process data
        start_time = time.time()
        self.logger.info(f"Loading {self.dataset_name()} dataset (split: {split})")

        try:
            # Load raw data
            raw_examples = self.load_raw_data(split)

            if max_examples:
                raw_examples = raw_examples[:max_examples]

            # Process examples
            processed_examples = []
            failed_examples = 0

            for i, raw_example in enumerate(raw_examples):
                try:
                    example = self.process_example(raw_example)

                    # Validate if enabled
                    if self.validate_examples:
                        is_valid, errors = example.validate()
                        if not is_valid:
                            self.logger.warning(
                                f"Example {i} validation failed: {errors}"
                            )
                            failed_examples += 1
                            continue

                    processed_examples.append(example)

                except Exception as e:
                    self.logger.warning(f"Failed to process example {i}: {e}")
                    failed_examples += 1
                    continue

            load_time = time.time() - start_time

            self.logger.info(
                f"Loaded {len(processed_examples)} examples "
                f"({failed_examples} failed) in {load_time:.2f}s"
            )

            # Cache results
            if self.cache_enabled and processed_examples:
                try:
                    cache_path.parent.mkdir(exist_ok=True)
                    with open(cache_path, "wb") as f:
                        pickle.dump(processed_examples, f)
                    self.logger.debug(f"Cached results to {cache_path}")
                except Exception as e:
                    self.logger.warning(f"Failed to cache results: {e}")

            return processed_examples

        except Exception as e:
            self.logger.error(f"Failed to load {self.dataset_name()}: {e}")
            raise


class CNNDailyMailLoader(DatasetLoader):
    """Loader for CNN/DailyMail summarization dataset."""

    def dataset_name(self) -> str:
        return "cnn_dailymail"

    def load_raw_data(self, split: str = "test") -> List[Dict[str, Any]]:
        """Load CNN/DailyMail data from files."""
        data_path = self.data_dir / "raw" / "cnn_dailymail" / f"{split}.json"

        if not data_path.exists():
            raise FileNotFoundError(
                f"CNN/DailyMail {split} data not found at {data_path}. "
                "Please download the dataset first using the download_datasets() function."
            )

        with open(data_path, "r", encoding="utf-8") as f:
            return json.load(f)

    def process_example(self, raw_example: Dict[str, Any]) -> FactualityExample:
        """Process CNN/DailyMail example to standard format."""
        return FactualityExample(
            example_id=raw_example.get("id", str(hash(raw_example["article"]))),
            source=raw_example["article"],
            summary=raw_example["highlights"],
            dataset_name=self.dataset_name(),
            human_label=None,  # No human factual consistency labels - will use synthetic generation
            metadata={
                "source_length": len(raw_example["article"]),
                "summary_length": len(raw_example["highlights"]),
                "original_format": "cnn_dailymail",
                "domain": "news",
                "ground_truth_type": "synthetic_labels",
            },
        )


class XSumLoader(DatasetLoader):
    """Loader for XSum abstractive summarization dataset."""

    def dataset_name(self) -> str:
        return "xsum"

    def load_raw_data(self, split: str = "test") -> List[Dict[str, Any]]:
        """Load XSum data from files."""
        data_path = self.data_dir / "raw" / "xsum" / f"{split}.json"

        if not data_path.exists():
            raise FileNotFoundError(
                f"XSum {split} data not found at {data_path}. "
                "Please download the dataset first using the download_datasets() function."
            )

        with open(data_path, "r", encoding="utf-8") as f:
            return json.load(f)

    def process_example(self, raw_example: Dict[str, Any]) -> FactualityExample:
        """Process XSum example to standard format."""
        return FactualityExample(
            example_id=raw_example.get("id", raw_example.get("bbcid", str(hash(raw_example["document"])))),
            source=raw_example["document"],
            summary=raw_example["summary"],
            dataset_name=self.dataset_name(),
            human_label=None,  # No human factual consistency labels - will use synthetic generation
            metadata={
                "source_length": len(raw_example["document"]),
                "summary_length": len(raw_example["summary"]),
                "original_format": "xsum",
                "domain": "news",
                "abstractiveness": "high",
                "ground_truth_type": "synthetic_labels",
            },
        )


# Dataset registry - CNN/DailyMail and XSum only
DATASET_LOADERS = {
    "cnn_dailymail": CNNDailyMailLoader,
    "xsum": XSumLoader,
}


def quick_load_dataset(
    dataset_name: str,
    split: str = "test",
    max_examples: Optional[int] = None,
    data_dir: Union[str, Path] = "data",
    use_cache: bool = True,
    validate_examples: bool = True,
) -> List[FactualityExample]:
    """
    Quick dataset loading function for convenience.

    This is the main function referenced in src/__init__.py that provides
    a simple interface for loading any supported dataset.

    Args:
        dataset_name: Name of dataset to load ("cnn_dailymail", "xsum")
        split: Dataset split ("train", "test", "validation")
        max_examples: Maximum number of examples to load
        data_dir: Directory containing dataset files
        use_cache: Whether to use cached results
        validate_examples: Whether to validate loaded examples

    Returns:
        List of FactualityExample objects

    Raises:
        ValueError: If dataset_name is not supported
        FileNotFoundError: If dataset files are not found

    Example:
        >>> examples = quick_load_dataset("cnn_dailymail", max_examples=100)
        >>> print(f"Loaded {len(examples)} examples")
    """
    if dataset_name not in DATASET_LOADERS:
        available = ", ".join(DATASET_LOADERS.keys())
        raise ValueError(
            f"Unsupported dataset: {dataset_name}. " f"Available datasets: {available}"
        )

    loader_class = DATASET_LOADERS[dataset_name]
    loader = loader_class(
        data_dir=data_dir, cache_enabled=use_cache, validate_examples=validate_examples
    )

    return loader.load_dataset(
        split=split, max_examples=max_examples, use_cache=use_cache
    )


def load_datasets(
    dataset_names: List[str],
    split: str = "test",
    sample_size: Optional[int] = None,
    data_dir: Union[str, Path] = "data",
    use_cache: bool = True,
    validate_examples: bool = True,
) -> Dict[str, List[FactualityExample]]:
    """
    Load multiple datasets and return as a dictionary.
    
    This function provides the interface expected by the batch experiments,
    loading multiple datasets and returning them in a dictionary format.
    
    Args:
        dataset_names: List of dataset names to load
        split: Dataset split ("train", "test", "validation")
        sample_size: Maximum number of examples per dataset (alias for max_examples)
        data_dir: Directory containing dataset files
        use_cache: Whether to use cached results
        validate_examples: Whether to validate loaded examples
        
    Returns:
        Dictionary mapping dataset names to lists of FactualityExample objects
        
    Example:
        >>> datasets = load_datasets(["cnn_dailymail", "xsum"], sample_size=100)
        >>> cnn_data = datasets["cnn_dailymail"]
        >>> print(f"Loaded {len(cnn_data)} CNN examples")
    """
    results = {}
    
    for dataset_name in dataset_names:
        logger.info(f"Loading dataset: {dataset_name}")
        try:
            dataset = quick_load_dataset(
                dataset_name=dataset_name,
                split=split,
                max_examples=sample_size,
                data_dir=data_dir,
                use_cache=use_cache,
                validate_examples=validate_examples
            )
            results[dataset_name] = dataset
            logger.info(f"Successfully loaded {len(dataset)} examples from {dataset_name}")
        except Exception as e:
            logger.error(f"Failed to load dataset {dataset_name}: {e}")
            raise
            
    return results


def load_processed_dataset(
    dataset_name: str,
    task_type: str,
    split: str = "test",
    max_examples: Optional[int] = None,
    data_dir: Union[str, Path] = "data",
):
    """
    Load preprocessed dataset from the processed folder.
    
    Args:
        dataset_name: Name of dataset ("cnn_dailymail", "xsum")
        task_type: Task type ("entailment_inference", "summary_ranking", "consistency_rating")
        split: Dataset split (not used for processed data, kept for compatibility)
        max_examples: Maximum number of examples to load
        data_dir: Directory containing dataset files
        
    Returns:
        List of ProcessedExample objects
        
    Raises:
        FileNotFoundError: If processed data file is not found
    """
    processed_path = Path(data_dir) / "processed" / dataset_name / f"{task_type}.json"
    
    if not processed_path.exists():
        raise FileNotFoundError(
            f"Processed data not found at {processed_path}. "
            f"Run the download script with preprocessing enabled first."
        )
    
    with open(processed_path, "r", encoding="utf-8") as f:
        data = json.load(f)
    
    if max_examples:
        data = data[:max_examples]
    
    # Import ProcessedExample here to avoid circular imports
    from .preprocessors import ProcessedExample
    examples = [ProcessedExample.from_dict(example_dict) for example_dict in data]
    
    logger.info(f"Loaded {len(examples)} preprocessed examples from {processed_path}")
    return examples


def get_available_datasets() -> List[str]:
    """
    Get list of available dataset names.

    Returns:
        List of supported dataset names: ["cnn_dailymail", "xsum"]
    """
    return list(DATASET_LOADERS.keys())


def validate_dataset_path(
    dataset_name: str, data_dir: Union[str, Path] = "data"
) -> Dict[str, Any]:
    """
    Validate that dataset files exist and are accessible.

    Args:
        dataset_name: Name of dataset to validate
        data_dir: Directory containing dataset files

    Returns:
        Dictionary containing validation results
    """
    validation_results = {
        "dataset_name": dataset_name,
        "status": "success",
        "files_found": [],
        "files_missing": [],
        "warnings": [],
        "errors": [],
    }

    if dataset_name not in DATASET_LOADERS:
        validation_results["status"] = "error"
        validation_results["errors"].append(f"Unsupported dataset: {dataset_name}")
        return validation_results

    data_path = Path(data_dir) / "raw" / dataset_name

    if not data_path.exists():
        validation_results["status"] = "error"
        validation_results["errors"].append(f"Dataset directory not found: {data_path}")
        return validation_results

    # Check for common split files
    splits = ["train", "test", "validation", "dev"]
    for split in splits:
        split_file = data_path / f"{split}.json"
        if split_file.exists():
            validation_results["files_found"].append(str(split_file))
        else:
            validation_results["files_missing"].append(str(split_file))

    if not validation_results["files_found"]:
        validation_results["status"] = "error"
        validation_results["errors"].append("No dataset files found")
    elif len(validation_results["files_missing"]) > 0:
        validation_results["status"] = "warning"
        validation_results["warnings"].append("Some dataset splits are missing")

    return validation_results


def get_dataset_info() -> Dict[str, Dict[str, Any]]:
    """
    Get comprehensive information about all supported datasets.

    Returns:
        Dictionary with dataset information including configuration details
    """
    return {
        "cnn_dailymail": {
            "name": "CNN/DailyMail",
            "description": "News summarization dataset with extractive summaries",
            "huggingface_name": "ccdv/cnn_dailymail",
            "version": "3.0.0",
            "domain": "news",
            "abstractiveness": "moderate",
            "total_examples": 287113,
            "splits": ["train", "validation", "test"],
            "avg_source_length": 781,
            "avg_summary_length": 56,
            "use_cases": ["entailment_inference", "summary_ranking", "consistency_rating"],
        },
        "xsum": {
            "name": "XSum",
            "description": "Abstractive summarization of BBC articles",
            "huggingface_name": "EdinburghNLP/xsum",
            "domain": "news",
            "abstractiveness": "high",
            "total_examples": 204045,
            "splits": ["train", "validation", "test"],
            "avg_source_length": 431,
            "avg_summary_length": 23,
            "use_cases": ["entailment_inference", "summary_ranking", "consistency_rating"],
        },
    }


if __name__ == "__main__":
    """Command line interface for dataset operations."""
    parser = argparse.ArgumentParser(description="Dataset operations for factuality evaluation")
    parser.add_argument("--info", action="store_true",
                        help="Show dataset information")
    parser.add_argument("--validate", action="store_true",
                        help="Validate existing datasets")
    parser.add_argument("--test-loaders", action="store_true",
                        help="Test dataset loaders")
    parser.add_argument("--data-dir", default="data",
                        help="Directory containing datasets")
    
    args = parser.parse_args()
    
    # Configure logging
    logging.basicConfig(
        level=logging.INFO,
        format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
    )
    
    if args.info:
        dataset_info = get_dataset_info()
        print("\n=== Dataset Information ===")
        for dataset_name, info in dataset_info.items():
            print(f"\n{info['name']} ({dataset_name}):")
            print(f"  Description: {info['description']}")
            print(f"  Domain: {info['domain']}")
            print(f"  Total examples: {info.get('total_examples', 'Unknown')}")
            print(f"  Splits: {info.get('splits', ['test'])}")
            print(f"  Use cases: {info['use_cases']}")
            
    elif args.validate:
        print("Validating existing datasets...")
        for dataset in get_available_datasets():
            result = validate_dataset_path(dataset, args.data_dir)
            print(f"\n{dataset}: {result['status']}")
            if result["errors"]:
                print(f"  Errors: {result['errors']}")
            if result["warnings"]:
                print(f"  Warnings: {result['warnings']}")
            if result["files_found"]:
                print(f"  Files found: {len(result['files_found'])}")
                
    elif args.test_loaders:
        print("Testing dataset loaders...")
        for dataset_name in get_available_datasets():
            try:
                examples = quick_load_dataset(dataset_name, max_examples=5, data_dir=args.data_dir)
                print(f"✓ {dataset_name}: Successfully loaded {len(examples)} examples")
                if examples:
                    example = examples[0]
                    print(f"  Sample: {example.source[:100]}...")
            except Exception as e:
                print(f"✗ {dataset_name}: Failed to load - {e}")
                
    else:
        # Default behavior - show available datasets
        print("Available datasets:", get_available_datasets())
        print("Dataset info:", get_dataset_info())
        print("\nUse --help for more options")