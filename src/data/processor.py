#!/usr/bin/env python3
"""
Main data processing module for the factuality evaluation framework.

This module handles all data processing tasks including:
- Loading raw JSONL datasets
- Processing and converting to standard format
- Combining and randomizing splits
- Saving processed datasets
"""

import json
import random
import logging
from pathlib import Path
from typing import Dict, List, Any, Tuple
from collections import Counter

from .loaders import FrankLoader, SummEvalLoader

logger = logging.getLogger(__name__)


class DataProcessor:
    """Main data processor for factuality evaluation datasets."""
    
    def __init__(self, base_data_dir: str = "data"):
        """Initialize the data processor.
        
        Args:
            base_data_dir: Base directory for data storage
        """
        self.base_data_dir = Path(base_data_dir)
        self.raw_dir = self.base_data_dir / "raw"
        self.processed_dir = self.base_data_dir / "processed"
        self.cache_dir = self.base_data_dir / "cache"
        
        # Ensure directories exist
        self.processed_dir.mkdir(parents=True, exist_ok=True)
        self.cache_dir.mkdir(parents=True, exist_ok=True)
    
    def clear_cache(self):
        """Clear the cache directory to force fresh processing."""
        import shutil
        if self.cache_dir.exists():
            shutil.rmtree(self.cache_dir)
        self.cache_dir.mkdir(parents=True, exist_ok=True)
        logger.info("Cache cleared for fresh processing")
    
    def process_dataset_splits(self, dataset_name: str, loader_class, splits: List[str]) -> Dict[str, int]:
        """Process individual splits for a dataset.
        
        Args:
            dataset_name: Name of the dataset (e.g., 'frank', 'summeval')
            loader_class: Loader class to use
            splits: List of split names to process
            
        Returns:
            Dictionary mapping split names to number of examples
        """
        logger.info(f"Processing {dataset_name} dataset splits...")
        loader = loader_class(data_dir=self.base_data_dir)
        results = {}
        
        for split in splits:
            logger.info(f"  Processing {split} split...")
            
            raw_path = self.raw_dir / dataset_name / f"{dataset_name}_{split}.jsonl"
            examples = []
            
            with open(raw_path, 'r', encoding='utf-8') as f:
                for line in f:
                    if line.strip():
                        examples.append(json.loads(line))
            
            processed_examples = [loader.process_example(ex) for ex in examples]
            
            # Randomize the order
            logger.info(f"    Randomizing {len(processed_examples)} {split} examples...")
            random.seed(42)  # Set seed for reproducibility
            random.shuffle(processed_examples)
            
            output_dir = self.processed_dir / dataset_name
            output_dir.mkdir(parents=True, exist_ok=True)
            output_path = output_dir / f"{dataset_name}_{split}_processed.json"
            
            with open(output_path, 'w', encoding='utf-8') as f:
                json.dump([ex.to_dict() for ex in processed_examples], f, indent=2)
            
            logger.info(f"    Saved {len(processed_examples)} {split} examples to {output_path}")
            results[split] = len(processed_examples)
        
        return results
    
    def combine_splits(self, dataset_name: str, splits: List[str]) -> int:
        """Combine processed splits into a single randomized dataset.
        
        Args:
            dataset_name: Name of the dataset
            splits: List of split names to combine
            
        Returns:
            Total number of examples in combined dataset
        """
        logger.info(f"Combining {dataset_name} splits: {splits}")
        
        all_data = []
        split_counts = {}
        
        for split in splits:
            split_file = self.processed_dir / dataset_name / f"{dataset_name}_{split}_processed.json"
            
            with open(split_file, 'r', encoding='utf-8') as f:
                split_data = json.load(f)
            
            all_data.extend(split_data)
            split_counts[split] = len(split_data)
            logger.info(f"  Loaded {len(split_data)} examples from {split} split")
        
        # Randomize the combined data
        logger.info(f"  Randomizing {len(all_data)} combined examples...")
        random.seed(42)
        random.shuffle(all_data)
        
        # Analyze label distribution
        human_labels = [item['human_label'] for item in all_data]
        label_counts = Counter(human_labels)
        
        logger.info(f"  Combined label distribution:")
        for label, count in sorted(label_counts.items()):
            percentage = (count / len(all_data)) * 100
            logger.info(f"    {label}: {count} ({percentage:.1f}%)")
        
        combined_path = self.processed_dir / dataset_name / f"{dataset_name}_processed.json"
        
        with open(combined_path, 'w', encoding='utf-8') as f:
            json.dump(all_data, f, indent=2, ensure_ascii=False)
        
        logger.info(f"  Saved {len(all_data)} combined examples to {combined_path}")
        
        # Show first few labels to verify randomization
        first_10_labels = human_labels[:10]
        logger.info(f"  First 10 labels after randomization: {first_10_labels}")
        
        return len(all_data)
    
    def process_all_datasets(self) -> Dict[str, Dict[str, Any]]:
        """Process all datasets with splits and combinations.
        
        Returns:
            Dictionary with processing results for each dataset
        """
        logger.info("Starting comprehensive data processing...")
        
        # Clear cache for fresh processing
        self.clear_cache()
        
        results = {}
        
        frank_splits = self.process_dataset_splits("frank", FrankLoader, ["test", "val"])
        frank_combined = self.combine_splits("frank", ["test", "val"])
        results["frank"] = {
            "splits": frank_splits,
            "combined": frank_combined
        }
        
        summeval_splits = self.process_dataset_splits("summeval", SummEvalLoader, ["test", "val"])
        summeval_combined = self.combine_splits("summeval", ["test", "val"])
        results["summeval"] = {
            "splits": summeval_splits,
            "combined": summeval_combined
        }
        
        total_examples = frank_combined + summeval_combined
        logger.info("="*60)
        logger.info("DATA PROCESSING COMPLETED!")
        logger.info(f"Frank: {frank_combined} examples ({frank_splits})")
        logger.info(f"SummEval: {summeval_combined} examples ({summeval_splits})")
        logger.info(f"Total processed examples: {total_examples}")
        logger.info("="*60)
        
        return results
    
    def analyze_dataset_statistics(self) -> None:
        """Analyze and display statistics for processed datasets."""
        logger.info("Analyzing dataset statistics...")
        
        datasets = ["frank", "summeval"]
        
        for dataset_name in datasets:
            combined_file = self.processed_dir / dataset_name / f"{dataset_name}_processed.json"
            
            if combined_file.exists():
                with open(combined_file, 'r', encoding='utf-8') as f:
                    data = json.load(f)
                
                human_labels = [item['human_label'] for item in data]
                label_counts = Counter(human_labels)
                
                logger.info(f"{dataset_name.upper()} Dataset Statistics:")
                logger.info(f"  Total examples: {len(data)}")
                for label, count in sorted(label_counts.items()):
                    percentage = (count / len(data)) * 100
                    logger.info(f"  {label}: {count} ({percentage:.1f}%)")
                logger.info("")


def main():
    """Main function to process all datasets."""
    logging.basicConfig(
        level=logging.INFO,
        format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
    )
    
    processor = DataProcessor()
    results = processor.process_all_datasets()
    
    # Show final statistics
    processor.analyze_dataset_statistics()
    
    return results


if __name__ == "__main__":
    main()
