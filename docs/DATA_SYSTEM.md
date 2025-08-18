# Data Processing and Loading System

This document explains the consolidated data processing and loading system in `src/data/`.

## 🚀 Quick Start

### Process All Datasets

```python
# From Python
from src.data import process_all_datasets
results = process_all_datasets()

# From command line
python -m src.data.processor
```

### Load Datasets for Evaluation

```python
from src.data import FrankLoader, SummEvalLoader

# Load Frank dataset (uses combined processed file by default)
frank_loader = FrankLoader()
frank_examples = frank_loader.load_dataset(max_examples=100)

# Load SummEval dataset
summeval_loader = SummEvalLoader()
summeval_examples = summeval_loader.load_dataset(max_examples=100)
```

## 📁 File Structure

```text
src/data/
├── __init__.py          # Main exports and convenience functions
├── loaders.py           # Dataset loaders (FrankLoader, SummEvalLoader)
├── preprocessors.py     # Task-specific preprocessors
├── processor.py         # Main data processing engine
└── __main__.py          # CLI interface
```

## 🔧 Key Components

### DataProcessor (`processor.py`)

- **Main processing engine** that handles:
  - Loading raw JSONL files
  - Processing to standard format
  - Randomizing examples
  - Combining splits
  - Saving processed files

### Loaders (`loaders.py`)

- **FrankLoader**: Loads Frank factuality dataset
- **SummEvalLoader**: Loads SummEval factuality dataset
- **Priority order**: Combined processed → Split processed → Raw JSONL

### CLI Interface (`__main__.py`)

- `python -m src.data.processor` - Process all datasets
- `python -m src.data.processor --stats` - Show statistics only

## 📊 Processed Files

The system creates these files in `data/processed/`:

```text
data/processed/
├── frank/
│   ├── frank_test_processed.json    # Test split (1575 examples)
│   ├── frank_val_processed.json     # Val split (671 examples) 
│   └── frank_processed.json         # Combined randomized (2246 examples)
└── summeval/
    ├── summeval_test_processed.json # Test split (850 examples)
    ├── summeval_val_processed.json  # Val split (850 examples)
    └── summeval_processed.json      # Combined randomized (1700 examples)
```

## ✨ Features

- ✅ **Automatic prioritization**: Combined files used by default for evaluation
- ✅ **Randomization**: All datasets properly shuffled with seed=42
- ✅ **Real diverse data**: No more repeated fake samples  
- ✅ **Fallback loading**: Raw → Split processed → Combined processed
- ✅ **Integrated system**: All functionality in `src/data/`
- ✅ **Clean interface**: Simple imports and function calls

## 🎯 Usage Examples

### For Model Evaluation

```python
from src.data import FrankLoader, SummEvalLoader

# Load full datasets for evaluation
frank_data = FrankLoader().load_dataset()  # 2246 examples
summeval_data = SummEvalLoader().load_dataset()  # 1700 examples

# Both datasets are randomized and contain real diverse examples
```

### For Dataset Analysis

```python
from src.data import DataProcessor

processor = DataProcessor()
processor.analyze_dataset_statistics()
```

### For Reprocessing

```python
from src.data import process_all_datasets

# Reprocess everything from raw JSONL files
results = process_all_datasets()
```

## 📈 Dataset Statistics

- **Frank**: 2246 examples (66.5% False, 33.5% True)
- **SummEval**: 1700 examples (10.7% False, 89.3% True)  
- **Total**: 3946 examples across both datasets
- **All randomized** with reproducible seed=42
