"""
Dataset Download Pipeline for ChatGPT Factuality Evaluation
==========================================================

Comprehensive dataset downloader for the two core datasets:
CNN/DailyMail and XSum. Handles automatic downloading from
HuggingFace, format conversion, and validation.

Author: Michael Ogunjimi
Institution: University of Manchester
Course: MSc AI
"""

import json
import logging
import time
from datetime import datetime
from pathlib import Path
from typing import Dict, List, Optional, Union, Any
import argparse

try:
    from datasets import load_dataset
    HF_DATASETS_AVAILABLE = True
except ImportError:
    print("Error: datasets library not available. Install with: pip install datasets")
    load_dataset = None
    HF_DATASETS_AVAILABLE = False

# Configure logging
logger = logging.getLogger(__name__)


class DatasetDownloader:
    """
    Downloads and processes the three core datasets for factuality evaluation.
    
    Handles downloading from HuggingFace, format conversion to match the
    loader expectations, and proper validation.
    """
    
    def __init__(self, data_dir: Union[str, Path] = "data"):
        """Initialize the dataset downloader.
        
        Args:
            data_dir: Directory to store downloaded datasets
        """
        self.data_dir = Path(data_dir)
        self.raw_dir = self.data_dir / "raw"
        self.raw_dir.mkdir(parents=True, exist_ok=True)
        
        # Dataset configurations for CNN/DailyMail and XSum
        self.dataset_configs = {
            "cnn_dailymail": {
                "hf_name": "ccdv/cnn_dailymail",
                "version": "3.0.0",
                "splits": ["train", "validation", "test"],
                "key_mappings": {
                    "source": "article",
                    "summary": "highlights",
                    "id": "id"
                },
                "preprocessing": {
                    "max_source_length": 1024,
                    "max_summary_length": 256,
                    "min_summary_length": 10,
                    "remove_duplicates": True,
                    "clean_text": True
                }
            },
            "xsum": {
                "hf_name": "EdinburghNLP/xsum",
                "splits": ["train", "validation", "test"],
                "key_mappings": {
                    "source": "document",
                    "summary": "summary", 
                    "id": "id"
                },
                "preprocessing": {
                    "max_source_length": 1024,
                    "max_summary_length": 128,
                    "min_summary_length": 8,
                    "remove_duplicates": True,
                    "clean_text": True
                }
            }
        }
        
        self.logger = logging.getLogger(f"{__name__}.{self.__class__.__name__}")
    
    def download_dataset(
        self, 
        dataset_name: str, 
        max_samples: Optional[Dict[str, int]] = None,
        force_redownload: bool = False
    ) -> bool:
        """Download a specific dataset.
        
        Args:
            dataset_name: Name of the dataset to download
            max_samples: Maximum samples per split for development
            force_redownload: Whether to redownload even if files exist
            
        Returns:
            True if successful, False otherwise
        """
        if not HF_DATASETS_AVAILABLE:
            self.logger.error("datasets library not available. Install with: pip install datasets")
            return False
            
        if dataset_name not in self.dataset_configs:
            self.logger.error(f"Unknown dataset: {dataset_name}")
            return False
            
        config = self.dataset_configs[dataset_name]
        hf_name = config["hf_name"]
        version = config.get("version")
        
        # Check if already exists and not forcing redownload
        output_dir = self.raw_dir / dataset_name
        if output_dir.exists() and not force_redownload:
            existing_files = list(output_dir.glob("*.json"))
            if existing_files:
                self.logger.info(f"Dataset {dataset_name} already exists. Use force_redownload=True to redownload.")
                return True
        
        self.logger.info(f"Downloading {dataset_name} from {hf_name}")
        
        try:
            # Download dataset
            if version:
                dataset = load_dataset(hf_name, version)
            else:
                dataset = load_dataset(hf_name)
                
            # Create output directory
            output_dir.mkdir(exist_ok=True)
            
            # Process each split
            for split in config["splits"]:
                if split not in dataset:
                    self.logger.warning(f"Split '{split}' not found in {dataset_name}")
                    continue
                    
                self.logger.info(f"Processing {dataset_name} {split} split...")
                
                # Get data for this split
                split_data = dataset[split]
                
                # Apply sampling if specified
                if max_samples and split in max_samples:
                    max_for_split = max_samples[split]
                    if len(split_data) > max_for_split:
                        split_data = split_data.select(range(max_for_split))
                        self.logger.info(f"Sampling {max_for_split} examples from {split}")
                
                # Convert to expected format
                processed_examples = []
                key_mappings = config["key_mappings"]
                preprocessing_config = config.get("preprocessing", {})
                
                for i, example in enumerate(split_data):
                    try:
                        # Extract basic fields
                        source_text = example[key_mappings["source"]]
                        summary_text = example[key_mappings["summary"]]
                        example_id = example.get(key_mappings["id"], str(i))
                        
                        # Apply preprocessing filters
                        if not self._validate_example_text(source_text, summary_text, preprocessing_config):
                            continue
                        
                        # Clean text if enabled
                        if preprocessing_config.get("clean_text", False):
                            source_text = self._clean_text(source_text)
                            summary_text = self._clean_text(summary_text)
                        
                        # Create processed example in loader-compatible format
                        processed_example = {
                            "id": example_id,
                            "article": source_text,
                            "highlights": summary_text  # CNN/DM format
                        }
                        
                        # Handle dataset-specific formats
                        if dataset_name == "xsum":
                            processed_example = {
                                "id": example_id,
                                "document": source_text,
                                "summary": summary_text
                            }
                            
                        processed_examples.append(processed_example)
                        
                    except Exception as e:
                        self.logger.warning(f"Error processing example {i}: {e}")
                        continue
                
                # Save to JSON file
                output_file = output_dir / f"{split}.json"
                with open(output_file, 'w', encoding='utf-8') as f:
                    json.dump(processed_examples, f, indent=2, ensure_ascii=False)
                    
                self.logger.info(f"Saved {len(processed_examples)} examples to {output_file}")
                
            return True
            
        except Exception as e:
            self.logger.error(f"Error downloading {dataset_name}: {e}")
            return False
    
    def _validate_example_text(self, source: str, summary: str, config: Dict) -> bool:
        """Validate example text against preprocessing requirements."""
        min_summary_length = config.get("min_summary_length", 5)
        max_source_length = config.get("max_source_length", 10000)
        max_summary_length = config.get("max_summary_length", 1000)
        
        # Check minimum lengths
        if len(summary.strip()) < min_summary_length:
            return False
        if len(source.strip()) < 50:  # Minimum source length
            return False
            
        # Check maximum lengths (character count)
        if len(source) > max_source_length * 4:  # Rough char to token ratio
            return False
        if len(summary) > max_summary_length * 4:
            return False
            
        # Check for valid content
        if not source.strip() or not summary.strip():
            return False
            
        return True
    
    def _clean_text(self, text: str) -> str:
        """Clean text content."""
        import re
        
        # Remove control characters
        text = re.sub(r'[\x00-\x08\x0b\x0c\x0e-\x1f\x7f-\x84\x86-\x9f]', ' ', text)
        
        # Normalize unicode
        text = text.encode('utf-8', errors='ignore').decode('utf-8')
        
        # Normalize whitespace
        text = re.sub(r'\s+', ' ', text)
        text = text.strip()
        
        return text
    
    def download_all(
        self, 
        max_samples_per_split: Optional[int] = None,
        development_mode: bool = False,
        force_redownload: bool = False
    ) -> Dict[str, bool]:
        """Download all configured datasets.
        
        Args:
            max_samples_per_split: Maximum samples per split for development
            development_mode: If True, use smaller sample sizes
            force_redownload: Whether to redownload existing datasets
            
        Returns:
            Dictionary mapping dataset names to success status
        """
        results = {}
        
        # Set sampling limits for development
        max_samples = None
        if max_samples_per_split or development_mode:
            if development_mode:
                # Small samples for development
                max_samples = {
                    "train": 100,
                    "validation": 50,
                    "test": 50
                }
            else:
                max_samples = {
                    "train": max_samples_per_split,
                    "validation": max_samples_per_split // 2,
                    "test": max_samples_per_split // 2
                }
        
        for dataset_name in self.dataset_configs:
            self.logger.info(f"Starting download of {dataset_name}...")
            start_time = time.time()
            
            success = self.download_dataset(
                dataset_name, 
                max_samples, 
                force_redownload=force_redownload
            )
            
            download_time = time.time() - start_time
            results[dataset_name] = success
            
            if success:
                self.logger.info(f"✓ {dataset_name} completed in {download_time:.1f}s")
            else:
                self.logger.error(f"✗ {dataset_name} failed")
            
        return results
    
    def get_download_info(self) -> Dict[str, Dict]:
        """Get information about downloaded datasets."""
        info = {}
        
        for dataset_name, config in self.dataset_configs.items():
            dataset_info = {
                "name": dataset_name,
                "huggingface_name": config["hf_name"],
                "version": config.get("version", "latest"),
                "splits": {},
                "total_examples": 0,
                "download_status": "not_downloaded"
            }
            
            dataset_dir = self.raw_dir / dataset_name
            
            if dataset_dir.exists():
                dataset_info["download_status"] = "partial"
                
                for split in config["splits"]:
                    split_file = dataset_dir / f"{split}.json"
                    if split_file.exists():
                        try:
                            with open(split_file, 'r', encoding='utf-8') as f:
                                data = json.load(f)
                                dataset_info["splits"][split] = {
                                    "examples": len(data),
                                    "file_size": split_file.stat().st_size,
                                    "last_modified": datetime.fromtimestamp(
                                        split_file.stat().st_mtime
                                    ).isoformat()
                                }
                                dataset_info["total_examples"] += len(data)
                        except Exception as e:
                            self.logger.warning(f"Error reading {split_file}: {e}")
                            dataset_info["splits"][split] = {"examples": 0, "error": str(e)}
                    else:
                        dataset_info["splits"][split] = {"examples": 0, "status": "missing"}
                
                # Check if all expected splits are present
                expected_splits = set(config["splits"])
                downloaded_splits = set(split for split, info in dataset_info["splits"].items() 
                                     if info.get("examples", 0) > 0)
                
                if downloaded_splits == expected_splits:
                    dataset_info["download_status"] = "complete"
                    
            info[dataset_name] = dataset_info
            
        return info
    
    def save_download_metadata(self, download_results: Dict[str, bool]) -> None:
        """Save download metadata."""
        metadata = {
            "download_timestamp": datetime.now().isoformat(),
            "download_results": download_results,
            "dataset_info": self.get_download_info(),
            "data_directory": str(self.data_dir),
            "raw_directory": str(self.raw_dir),
            "datasets_library_version": self._get_datasets_version(),
            "configurations": self.dataset_configs
        }
        
        metadata_file = self.data_dir / "download_metadata.json"
        with open(metadata_file, 'w', encoding='utf-8') as f:
            json.dump(metadata, f, indent=2, ensure_ascii=False)
            
        self.logger.info(f"Saved download metadata to {metadata_file}")
    
    def _get_datasets_version(self) -> str:
        """Get version of datasets library."""
        try:
            import datasets
            return getattr(datasets, '__version__', 'unknown')
        except:
            return 'not_available'
    
    def validate_downloads(self) -> Dict[str, Dict]:
        """Validate all downloaded datasets."""
        validation_results = {}
        
        for dataset_name in self.dataset_configs:
            result = {
                "dataset": dataset_name,
                "status": "not_found",
                "issues": [],
                "splits_validated": {}
            }
            
            dataset_dir = self.raw_dir / dataset_name
            if not dataset_dir.exists():
                result["issues"].append("Dataset directory not found")
                validation_results[dataset_name] = result
                continue
            
            config = self.dataset_configs[dataset_name]
            all_splits_valid = True
            
            for split in config["splits"]:
                split_file = dataset_dir / f"{split}.json"
                split_result = {"status": "missing", "examples": 0, "issues": []}
                
                if split_file.exists():
                    try:
                        with open(split_file, 'r', encoding='utf-8') as f:
                            data = json.load(f)
                            
                        split_result["examples"] = len(data)
                        split_result["status"] = "valid"
                        
                        # Validate a few examples
                        if data:
                            sample_example = data[0]
                            required_fields = ["id"]
                            
                            if dataset_name == "xsum":
                                required_fields.extend(["document", "summary"])
                            else:
                                required_fields.extend(["article"])
                                if dataset_name == "cnn_dailymail":
                                    required_fields.append("highlights")
                            
                            for field in required_fields:
                                if field not in sample_example:
                                    split_result["issues"].append(f"Missing required field: {field}")
                                    split_result["status"] = "invalid"
                                    all_splits_valid = False
                        
                    except Exception as e:
                        split_result["status"] = "error"
                        split_result["issues"].append(f"Error reading file: {e}")
                        all_splits_valid = False
                else:
                    split_result["issues"].append("File not found")
                    all_splits_valid = False
                
                result["splits_validated"][split] = split_result
            
            result["status"] = "valid" if all_splits_valid else "invalid"
            validation_results[dataset_name] = result
        
        return validation_results


def download_datasets(
    datasets: Optional[List[str]] = None,
    data_dir: Union[str, Path] = "data",
    max_samples: Optional[int] = None,
    development_mode: bool = False,
    force_redownload: bool = False
) -> Dict[str, bool]:
    """Convenience function to download datasets.
    
    Args:
        datasets: List of dataset names to download (None for all)
        data_dir: Directory to store datasets
        max_samples: Maximum samples per split for development
        development_mode: Use small sample sizes for testing
        force_redownload: Redownload even if files exist
        
    Returns:
        Dictionary mapping dataset names to success status
    """
    downloader = DatasetDownloader(data_dir)
    
    if datasets is None:
        return downloader.download_all(
            max_samples_per_split=max_samples,
            development_mode=development_mode,
            force_redownload=force_redownload
        )
    else:
        results = {}
        max_samples_dict = None
        if max_samples:
            max_samples_dict = {
                "train": max_samples,
                "validation": max_samples // 2,
                "test": max_samples // 2
            }
        
        for dataset_name in datasets:
            if dataset_name not in downloader.dataset_configs:
                results[dataset_name] = False
                logger.error(f"Unknown dataset: {dataset_name}")
                continue
                
            results[dataset_name] = downloader.download_dataset(
                dataset_name, 
                max_samples_dict,
                force_redownload=force_redownload
            )
            
        return results


def main():
    """Command line interface for dataset downloading."""
    parser = argparse.ArgumentParser(description="Download datasets for factuality evaluation")
    parser.add_argument("--download-all", action="store_true", 
                        help="Download all configured datasets")
    parser.add_argument("--download", nargs="+", 
                        choices=["cnn_dailymail", "xsum"],
                        help="Download specific datasets")
    parser.add_argument("--development", action="store_true",
                        help="Use small sample sizes for development")
    parser.add_argument("--max-samples", type=int,
                        help="Maximum samples per split")
    parser.add_argument("--force", action="store_true",
                        help="Force redownload even if files exist")
    parser.add_argument("--validate", action="store_true",
                        help="Validate existing downloads")
    parser.add_argument("--info", action="store_true",
                        help="Show dataset information")
    parser.add_argument("--data-dir", default="data",
                        help="Directory containing datasets")
    parser.add_argument("--verbose", "-v", action="store_true",
                        help="Enable verbose logging")
    
    args = parser.parse_args()
    
    # Configure logging
    log_level = logging.DEBUG if args.verbose else logging.INFO
    logging.basicConfig(
        level=log_level,
        format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
    )
    
    downloader = DatasetDownloader(args.data_dir)
    
    if args.download_all:
        print("Downloading all datasets...")
        results = downloader.download_all(
            max_samples_per_split=args.max_samples,
            development_mode=args.development,
            force_redownload=args.force
        )
        downloader.save_download_metadata(results)
        
        print("\n=== Download Results ===")
        for dataset_name, success in results.items():
            status = "✓" if success else "✗"
            print(f"{dataset_name}: {status}")
            
    elif args.download:
        print(f"Downloading datasets: {args.download}")
        results = download_datasets(
            datasets=args.download, 
            data_dir=args.data_dir, 
            max_samples=args.max_samples,
            development_mode=args.development,
            force_redownload=args.force
        )
        
        print("\n=== Download Results ===")
        for dataset_name, success in results.items():
            status = "✓" if success else "✗"
            print(f"{dataset_name}: {status}")
            
    elif args.validate:
        print("Validating downloaded datasets...")
        validation_results = downloader.validate_downloads()
        
        print("\n=== Validation Results ===")
        for dataset_name, result in validation_results.items():
            print(f"\n{dataset_name}: {result['status']}")
            if result["issues"]:
                print(f"  Issues: {result['issues']}")
            for split, split_result in result["splits_validated"].items():
                print(f"  {split}: {split_result['examples']} examples ({split_result['status']})")
                if split_result["issues"]:
                    print(f"    Issues: {split_result['issues']}")
                    
    elif args.info:
        info = downloader.get_download_info()
        
        print("\n=== Dataset Information ===")
        for dataset_name, dataset_info in info.items():
            print(f"\n{dataset_info['name']} ({dataset_name}):")
            print(f"  HuggingFace: {dataset_info['huggingface_name']}")
            print(f"  Version: {dataset_info['version']}")
            print(f"  Status: {dataset_info['download_status']}")
            print(f"  Total examples: {dataset_info['total_examples']}")
            print("  Splits:")
            for split, split_info in dataset_info["splits"].items():
                if isinstance(split_info, dict) and "examples" in split_info:
                    print(f"    {split}: {split_info['examples']} examples")
                else:
                    print(f"    {split}: not downloaded")
    else:
        parser.print_help()


if __name__ == "__main__":
    main()