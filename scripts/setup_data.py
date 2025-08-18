#!/usr/bin/env python3
"""
Data Setup Script for ChatGPT Factuality Evaluation
==================================================

This script downloads and prepares the datasets needed for factuality evaluation
experiments. It handles the complete data pipeline from downloading to preprocessing.

Usage:
    python scripts/setup_data.py
    python scripts/setup_data.py --quick-setup
    python scripts/setup_data.py --validate-only
    python scripts/setup_data.py --clean-and-reload

Author: Michael Ogunjimi
Institution: University of Manchester
Course: MSc AI
"""

import argparse
import logging
import sys
import time
from pathlib import Path
from typing import Dict, List, Optional

sys.path.insert(0, str(Path(__file__).parent.parent))

# Setup logging
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)

class DataSetup:
    """Complete data setup for factuality evaluation."""
    
    def __init__(self, project_root: Path = None):
        """Initialize data setup."""
        self.project_root = project_root or Path(__file__).parent.parent
        self.data_dir = self.project_root / "data"
        
    def check_data_availability(self) -> Dict[str, bool]:
        """Check if datasets are already available."""
        logger.info("Checking data availability...")
        
        results = {}
        
        # Check for raw data
        raw_datasets = ['frank', 'summeval']
        for dataset in raw_datasets:
            raw_path = self.data_dir / "raw" / dataset
            if raw_path.exists() and any(raw_path.glob("*.json")):
                logger.info(f"✓ Raw data found: {dataset}")
                results[f"raw_{dataset}"] = True
            else:
                logger.warning(f"⚠️  Raw data missing: {dataset}")
                results[f"raw_{dataset}"] = False
        
        # Check for processed data
        processed_datasets = ['frank', 'summeval']
        tasks = ['entailment_inference', 'summary_ranking', 'consistency_rating']
        
        for dataset in processed_datasets:
            processed_path = self.data_dir / "processed" / dataset
            if processed_path.exists():
                task_files = [processed_path / f"{task}.json" for task in tasks]
                if all(f.exists() for f in task_files):
                    logger.info(f"✓ Processed data found: {dataset}")
                    results[f"processed_{dataset}"] = True
                else:
                    logger.warning(f"⚠️  Processed data incomplete: {dataset}")
                    results[f"processed_{dataset}"] = False
            else:
                logger.warning(f"⚠️  Processed data missing: {dataset}")
                results[f"processed_{dataset}"] = False
        
        return results
    
    def download_raw_datasets(self, quick_setup: bool = False) -> bool:
        """Download raw datasets."""
        logger.info("Downloading raw datasets...")
        
        try:
            from src.data import download_datasets
            
            if quick_setup:
                logger.info("Quick setup: downloading small samples")
                max_samples = 50
            else:
                logger.info("Full setup: downloading complete datasets")
                max_samples = None
            
            # Download both datasets
            results = download_datasets(
                datasets=['frank', 'summeval'],
                data_dir=self.data_dir,
                max_samples=max_samples,
                development_mode=quick_setup,
                force_redownload=False
            )
            
            success = all(results.values())
            
            if success:
                logger.info("✓ Raw datasets downloaded successfully")
            else:
                logger.error("✗ Some datasets failed to download")
                for dataset, result in results.items():
                    status = "✓" if result else "✗"
                    logger.info(f"  {status} {dataset}")
            
            return success
            
        except Exception as e:
            logger.error(f"✗ Error downloading datasets: {e}")
            return False
    
    def preprocess_datasets(self, quick_setup: bool = False) -> bool:
        """Preprocess datasets for all tasks."""
        logger.info("Preprocessing datasets...")
        
        try:
            from src.data import load_dataset_for_task
            
            datasets = ['frank', 'summeval']
            tasks = ['entailment_inference', 'summary_ranking', 'consistency_rating']
            
            # Create processed directory
            processed_dir = self.data_dir / "processed"
            processed_dir.mkdir(exist_ok=True)
            
            success = True
            
            for dataset in datasets:
                logger.info(f"Processing {dataset}...")
                
                # Create dataset directory
                dataset_dir = processed_dir / dataset
                dataset_dir.mkdir(exist_ok=True)
                
                for task in tasks:
                    try:
                        logger.info(f"  Processing {task}...")
                        
                        # Set sample size
                        max_examples = 100 if quick_setup else None
                        
                        # Load and preprocess data
                        examples = load_dataset_for_task(
                            dataset_name=dataset,
                            task_type=task,
                            max_examples=max_examples,
                            data_dir=self.data_dir
                        )
                        
                        if examples:
                            # Save processed data
                            import json
                            output_file = dataset_dir / f"{task}.json"
                            processed_data = [example.to_dict() for example in examples]
                            
                            with open(output_file, 'w') as f:
                                json.dump(processed_data, f, indent=2)
                            
                            logger.info(f"  ✓ {task}: {len(examples)} examples saved")
                        else:
                            logger.warning(f"  ⚠️  {task}: No examples processed")
                            
                    except Exception as e:
                        logger.error(f"  ✗ {task}: {e}")
                        success = False
            
            if success:
                logger.info("✓ Dataset preprocessing completed successfully")
            else:
                logger.error("✗ Some preprocessing tasks failed")
            
            return success
            
        except Exception as e:
            logger.error(f"✗ Error preprocessing datasets: {e}")
            return False
    
    def validate_data_integrity(self) -> bool:
        """Validate data integrity."""
        logger.info("Validating data integrity...")
        
        try:
            from src.data import validate_dataset_path, quick_load_dataset
            
            # Validate raw datasets
            datasets = ['frank', 'summeval']
            all_valid = True
            
            for dataset in datasets:
                logger.info(f"Validating {dataset}...")
                
                # Validate path
                result = validate_dataset_path(dataset, self.data_dir)
                if result['status'] != 'valid':
                    logger.error(f"  ✗ Path validation failed: {result['errors']}")
                    all_valid = False
                    continue
                
                # Test loading
                try:
                    examples = quick_load_dataset(dataset, max_examples=5, data_dir=self.data_dir)
                    if examples:
                        logger.info(f"  ✓ Loaded {len(examples)} test examples")
                    else:
                        logger.error(f"  ✗ No examples loaded")
                        all_valid = False
                except Exception as e:
                    logger.error(f"  ✗ Loading failed: {e}")
                    all_valid = False
            
            # Validate processed data
            processed_dir = self.data_dir / "processed"
            if processed_dir.exists():
                for dataset in datasets:
                    dataset_dir = processed_dir / dataset
                    if dataset_dir.exists():
                        tasks = ['entailment_inference', 'summary_ranking', 'consistency_rating']
                        for task in tasks:
                            task_file = dataset_dir / f"{task}.json"
                            if task_file.exists():
                                try:
                                    import json
                                    with open(task_file, 'r') as f:
                                        data = json.load(f)
                                    logger.info(f"  ✓ {dataset}/{task}: {len(data)} processed examples")
                                except Exception as e:
                                    logger.error(f"  ✗ {dataset}/{task}: {e}")
                                    all_valid = False
                            else:
                                logger.warning(f"  ⚠️  {dataset}/{task}: file not found")
            
            if all_valid:
                logger.info("✓ Data integrity validation passed")
            else:
                logger.error("✗ Data integrity validation failed")
            
            return all_valid
            
        except Exception as e:
            logger.error(f"✗ Error validating data: {e}")
            return False
    
    def clean_data_directories(self) -> bool:
        """Clean existing data directories."""
        logger.info("Cleaning data directories...")
        
        try:
            import shutil
            
            # Clean raw data
            raw_dir = self.data_dir / "raw"
            if raw_dir.exists():
                shutil.rmtree(raw_dir)
                logger.info("✓ Cleaned raw data directory")
            
            # Clean processed data
            processed_dir = self.data_dir / "processed"
            if processed_dir.exists():
                shutil.rmtree(processed_dir)
                logger.info("✓ Cleaned processed data directory")
            
            # Clean cache
            cache_dir = self.data_dir / "cache"
            if cache_dir.exists():
                shutil.rmtree(cache_dir)
                logger.info("✓ Cleaned cache directory")
            
            return True
            
        except Exception as e:
            logger.error(f"✗ Error cleaning directories: {e}")
            return False
    
    def get_data_statistics(self) -> Dict[str, any]:
        """Get statistics about available data."""
        logger.info("Gathering data statistics...")
        
        stats = {
            'raw_datasets': {},
            'processed_datasets': {},
            'cache_size': 0,
            'total_examples': 0
        }
        
        try:
            # Raw data statistics
            raw_dir = self.data_dir / "raw"
            if raw_dir.exists():
                for dataset_dir in raw_dir.iterdir():
                    if dataset_dir.is_dir():
                        dataset_stats = {}
                        for split_file in dataset_dir.glob("*.json"):
                            try:
                                import json
                                with open(split_file, 'r') as f:
                                    data = json.load(f)
                                dataset_stats[split_file.stem] = len(data)
                            except:
                                dataset_stats[split_file.stem] = 0
                        stats['raw_datasets'][dataset_dir.name] = dataset_stats
            
            # Processed data statistics
            processed_dir = self.data_dir / "processed"
            if processed_dir.exists():
                for dataset_dir in processed_dir.iterdir():
                    if dataset_dir.is_dir():
                        dataset_stats = {}
                        for task_file in dataset_dir.glob("*.json"):
                            try:
                                import json
                                with open(task_file, 'r') as f:
                                    data = json.load(f)
                                dataset_stats[task_file.stem] = len(data)
                                stats['total_examples'] += len(data)
                            except:
                                dataset_stats[task_file.stem] = 0
                        stats['processed_datasets'][dataset_dir.name] = dataset_stats
            
            # Cache size
            cache_dir = self.data_dir / "cache"
            if cache_dir.exists():
                cache_size = sum(f.stat().st_size for f in cache_dir.rglob('*') if f.is_file())
                stats['cache_size'] = cache_size / (1024 * 1024)  # MB
            
            return stats
            
        except Exception as e:
            logger.error(f"Error gathering statistics: {e}")
            return stats
    
    def run_complete_setup(self, quick_setup: bool = False) -> bool:
        """Run complete data setup."""
        logger.info("🚀 Starting complete data setup...")
        
        success = True
        
        # Check current data availability
        availability = self.check_data_availability()
        
        # Download raw datasets if needed
        if not all(availability.get(f"raw_{dataset}", False) for dataset in ['frank', 'summeval']):
            if not self.download_raw_datasets(quick_setup):
                success = False
        else:
            logger.info("✓ Raw datasets already available")
        
        # Preprocess datasets if needed
        if not all(availability.get(f"processed_{dataset}", False) for dataset in ['frank', 'summeval']):
            if not self.preprocess_datasets(quick_setup):
                success = False
        else:
            logger.info("✓ Processed datasets already available")
        
        # Validate data integrity
        if not self.validate_data_integrity():
            success = False
        
        # Show statistics
        stats = self.get_data_statistics()
        self._print_statistics(stats)
        
        if success:
            logger.info("🎉 Data setup completed successfully!")
            self._print_next_steps()
        else:
            logger.error("❌ Data setup failed - please check errors above")
        
        return success
    
    def _print_statistics(self, stats: Dict):
        """Print data statistics."""
        logger.info("\n" + "="*50)
        logger.info("DATA STATISTICS")
        logger.info("="*50)
        
        if stats['raw_datasets']:
            logger.info("Raw Datasets:")
            for dataset, splits in stats['raw_datasets'].items():
                logger.info(f"  {dataset}:")
                for split, count in splits.items():
                    logger.info(f"    {split}: {count:,} examples")
        
        if stats['processed_datasets']:
            logger.info("Processed Datasets:")
            for dataset, tasks in stats['processed_datasets'].items():
                logger.info(f"  {dataset}:")
                for task, count in tasks.items():
                    logger.info(f"    {task}: {count:,} examples")
        
        logger.info(f"Total processed examples: {stats['total_examples']:,}")
        logger.info(f"Cache size: {stats['cache_size']:.1f} MB")
        logger.info("="*50)
    
    def _print_next_steps(self):
        """Print next steps after setup."""
        logger.info("\n" + "="*50)
        logger.info("NEXT STEPS")
        logger.info("="*50)
        logger.info("1. Run quick test to verify setup:")
        logger.info("   python scripts/quick_test.py")
        logger.info("")
        logger.info("2. Check cost estimation:")
        logger.info("   python scripts/estimate_costs.py")
        logger.info("")
        logger.info("3. Run experiments:")
        logger.info("   python experiments/run_all_experiments.py --quick-test")
        logger.info("="*50)

def main():
    """Main entry point."""
    parser = argparse.ArgumentParser(description="Setup data for factuality evaluation")
    parser.add_argument("--quick-setup", action="store_true", help="Quick setup with small samples")
    parser.add_argument("--validate-only", action="store_true", help="Only validate existing data")
    parser.add_argument("--clean-and-reload", action="store_true", help="Clean existing data and reload")
    parser.add_argument("--show-stats", action="store_true", help="Show data statistics")
    
    args = parser.parse_args()
    
    setup = DataSetup()
    
    if args.validate_only:
        success = setup.validate_data_integrity()
    elif args.clean_and_reload:
        logger.info("Cleaning and reloading data...")
        setup.clean_data_directories()
        success = setup.run_complete_setup(args.quick_setup)
    elif args.show_stats:
        stats = setup.get_data_statistics()
        setup._print_statistics(stats)
        success = True
    else:
        success = setup.run_complete_setup(args.quick_setup)
    
    sys.exit(0 if success else 1)

if __name__ == "__main__":
    main()
