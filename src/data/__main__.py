#!/usr/bin/env python3
"""
Command line interface for data processing.

Usage:
    python -m src.data.processor          # Process all datasets
    python -m src.data.processor --stats  # Show dataset statistics only
"""

import argparse
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent.parent.parent))

from src.data.processor import DataProcessor

def main():
    """Main CLI function."""
    parser = argparse.ArgumentParser(description="Factuality Evaluation Data Processor")
    parser.add_argument(
        "--stats", 
        action="store_true",
        help="Show dataset statistics only (no processing)"
    )
    parser.add_argument(
        "--data-dir",
        default="data",
        help="Base data directory (default: data)"
    )
    
    args = parser.parse_args()
    
    processor = DataProcessor(args.data_dir)
    
    if args.stats:
        print("📊 Analyzing dataset statistics...")
        processor.analyze_dataset_statistics()
    else:
        print("🚀 Processing all datasets...")
        results = processor.process_all_datasets()
        print("\n✅ Processing completed!")
        
        # Show brief summary
        total = sum(results[ds]["combined"] for ds in results)
        print(f"📈 Total examples processed: {total}")

if __name__ == "__main__":
    main()
