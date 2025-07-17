"""
Data Loaders Tests
==================

Tests for dataset loading functionality including caching, validation,
and processing of CNN/DailyMail and XSum datasets.

Author: Michael Ogunjimi
Institution: University of Manchester
"""

import pytest
import json
import pickle
import tempfile
from pathlib import Path
from unittest.mock import Mock, patch, MagicMock


class FactualityExample:
    """Simplified FactualityExample for testing"""
    
    def __init__(self, example_id, source_text, summary_text, dataset_name="test"):
        self.example_id = example_id
        self.source_text = source_text
        self.summary_text = summary_text
        self.dataset_name = dataset_name
        
    def validate(self):
        """Simple validation logic"""
        errors = []
        
        if not self.source_text or len(self.source_text.strip()) < 10:
            errors.append("Source text too short")
        
        if not self.summary_text or len(self.summary_text.strip()) < 5:
            errors.append("Summary text too short")
            
        return len(errors) == 0, errors


class DatasetLoader:
    """Base class for dataset loaders"""
    
    def __init__(self, data_dir="data", cache_enabled=True, validate_examples=True):
        self.data_dir = Path(data_dir)
        self.cache_enabled = cache_enabled
        self.validate_examples = validate_examples
        self.cache_dir = self.data_dir / "cache"
        
    def dataset_name(self):
        raise NotImplementedError
        
    def load_raw_data(self, split="test"):
        raise NotImplementedError
        
    def process_example(self, raw_example):
        raise NotImplementedError
    
    def load_dataset(self, split="test", max_examples=None, use_cache=True):
        """Load dataset with caching and validation"""
        cache_key = f"{self.dataset_name()}_{split}_{max_examples or 'all'}"
        cache_path = self.cache_dir / f"{cache_key}.pkl"
        
        # Try to load from cache
        if use_cache and self.cache_enabled and cache_path.exists():
            try:
                with open(cache_path, "rb") as f:
                    cached_data = pickle.load(f)
                return cached_data
            except Exception:
                pass  # Fall back to loading fresh data
        
        # Load and process data
        raw_examples = self.load_raw_data(split)
        
        if max_examples:
            raw_examples = raw_examples[:max_examples]
        
        processed_examples = []
        failed_examples = 0
        
        for i, raw_example in enumerate(raw_examples):
            try:
                example = self.process_example(raw_example)
                
                # Validate if enabled
                if self.validate_examples:
                    is_valid, errors = example.validate()
                    if not is_valid:
                        failed_examples += 1
                        continue
                
                processed_examples.append(example)
                
            except Exception:
                failed_examples += 1
                continue
        
        # Cache results
        if self.cache_enabled and processed_examples:
            try:
                cache_path.parent.mkdir(exist_ok=True)
                with open(cache_path, "wb") as f:
                    pickle.dump(processed_examples, f)
            except Exception:
                pass  # Cache failure shouldn't break loading
        
        return processed_examples


class CNNDailyMailLoader(DatasetLoader):
    """Loader for CNN/DailyMail dataset"""
    
    def dataset_name(self):
        return "cnn_dailymail"
    
    def load_raw_data(self, split="test"):
        """Load CNN/DailyMail data from files"""
        dataset_dir = self.data_dir / "raw" / "cnn_dailymail"
        split_file = dataset_dir / f"{split}.json"
        
        if not split_file.exists():
            raise FileNotFoundError(f"CNN/DailyMail {split} file not found: {split_file}")
        
        with open(split_file, "r", encoding="utf-8") as f:
            data = json.load(f)
        
        return data
    
    def process_example(self, raw_example):
        """Process CNN/DailyMail example to standard format"""
        return FactualityExample(
            example_id=raw_example["id"],
            source_text=raw_example["article"],
            summary_text=raw_example["highlights"],
            dataset_name="cnn_dailymail"
        )


class XSumLoader(DatasetLoader):
    """Loader for XSum dataset"""
    
    def dataset_name(self):
        return "xsum"
    
    def load_raw_data(self, split="test"):
        """Load XSum data from files"""
        dataset_dir = self.data_dir / "raw" / "xsum"
        split_file = dataset_dir / f"{split}.json"
        
        if not split_file.exists():
            raise FileNotFoundError(f"XSum {split} file not found: {split_file}")
        
        with open(split_file, "r", encoding="utf-8") as f:
            data = json.load(f)
        
        return data
    
    def process_example(self, raw_example):
        """Process XSum example to standard format"""
        return FactualityExample(
            example_id=raw_example["id"],
            source_text=raw_example["document"],
            summary_text=raw_example["summary"],
            dataset_name="xsum"
        )


# Dataset registry
DATASET_LOADERS = {
    "cnn_dailymail": CNNDailyMailLoader,
    "xsum": XSumLoader,
}


def quick_load_dataset(dataset_name, split="test", max_examples=None, 
                      data_dir="data", use_cache=True, validate_examples=True):
    """Quick dataset loading utility"""
    if dataset_name not in DATASET_LOADERS:
        available = ", ".join(DATASET_LOADERS.keys())
        raise ValueError(f"Unknown dataset: {dataset_name}. Available: {available}")
    
    loader_class = DATASET_LOADERS[dataset_name]
    loader = loader_class(
        data_dir=data_dir,
        cache_enabled=use_cache,
        validate_examples=validate_examples
    )
    
    return loader.load_dataset(
        split=split,
        max_examples=max_examples,
        use_cache=use_cache
    )


def get_available_datasets():
    """Get list of available datasets"""
    return list(DATASET_LOADERS.keys())


def validate_dataset_path(dataset_name, data_dir="data"):
    """Validate dataset path and files"""
    validation_results = {
        "status": "unknown",
        "errors": [],
        "warnings": [],
        "files_found": []
    }
    
    if dataset_name not in DATASET_LOADERS:
        validation_results["status"] = "error"
        validation_results["errors"].append(f"Unknown dataset: {dataset_name}")
        return validation_results
    
    data_path = Path(data_dir)
    dataset_dir = data_path / "raw" / dataset_name
    
    if not dataset_dir.exists():
        validation_results["status"] = "error"
        validation_results["errors"].append(f"Dataset directory not found: {dataset_dir}")
        return validation_results
    
    # Check for split files
    expected_splits = ["train", "validation", "test"]
    missing_splits = []
    
    for split in expected_splits:
        split_file = dataset_dir / f"{split}.json"
        if split_file.exists():
            validation_results["files_found"].append(str(split_file))
        else:
            missing_splits.append(split)
    
    if missing_splits:
        validation_results["warnings"].append(f"Missing splits: {missing_splits}")
    
    if validation_results["files_found"]:
        validation_results["status"] = "valid"
    else:
        validation_results["status"] = "error"
        validation_results["errors"].append("No dataset files found")
    
    return validation_results


class TestDatasetLoader:
    """Test base dataset loader functionality"""
    
    @pytest.fixture
    def temp_dir(self):
        with tempfile.TemporaryDirectory() as tmpdir:
            yield Path(tmpdir)
    
    def test_loader_initialization(self, temp_dir):
        """Test loader initialization"""
        loader = DatasetLoader(data_dir=temp_dir)
        
        assert loader.data_dir == temp_dir
        assert loader.cache_enabled is True
        assert loader.validate_examples is True
        assert loader.cache_dir == temp_dir / "cache"
    
    def test_loader_with_cache_disabled(self, temp_dir):
        """Test loader with caching disabled"""
        loader = DatasetLoader(data_dir=temp_dir, cache_enabled=False)
        
        assert loader.cache_enabled is False


class TestCNNDailyMailLoader:
    """Test CNN/DailyMail dataset loader"""
    
    @pytest.fixture
    def temp_dir(self):
        with tempfile.TemporaryDirectory() as tmpdir:
            yield Path(tmpdir)
    
    @pytest.fixture
    def sample_cnn_data(self, temp_dir):
        """Create sample CNN/DailyMail data"""
        dataset_dir = temp_dir / "raw" / "cnn_dailymail"
        dataset_dir.mkdir(parents=True)
        
        sample_data = [
            {
                "id": "cnn_test_1",
                "article": "This is a sample news article about factuality evaluation. " * 10,
                "highlights": "Sample summary for factuality testing."
            },
            {
                "id": "cnn_test_2", 
                "article": "Another news article for testing purposes. " * 8,
                "highlights": "Another summary for testing."
            }
        ]
        
        # Create test split
        test_file = dataset_dir / "test.json"
        with open(test_file, "w", encoding="utf-8") as f:
            json.dump(sample_data, f, ensure_ascii=False, indent=2)
        
        return temp_dir
    
    def test_cnn_loader_initialization(self, temp_dir):
        """Test CNN/DailyMail loader initialization"""
        loader = CNNDailyMailLoader(data_dir=temp_dir)
        
        assert loader.dataset_name() == "cnn_dailymail"
    
    def test_cnn_load_raw_data(self, sample_cnn_data):
        """Test loading raw CNN/DailyMail data"""
        loader = CNNDailyMailLoader(data_dir=sample_cnn_data)
        
        raw_data = loader.load_raw_data("test")
        
        assert len(raw_data) == 2
        assert raw_data[0]["id"] == "cnn_test_1"
        assert "article" in raw_data[0]
        assert "highlights" in raw_data[0]
    
    def test_cnn_load_missing_file(self, temp_dir):
        """Test loading missing CNN/DailyMail file"""
        loader = CNNDailyMailLoader(data_dir=temp_dir)
        
        with pytest.raises(FileNotFoundError):
            loader.load_raw_data("test")
    
    def test_cnn_process_example(self, temp_dir):
        """Test processing CNN/DailyMail examples"""
        loader = CNNDailyMailLoader(data_dir=temp_dir)
        
        raw_example = {
            "id": "test_id",
            "article": "Test article content",
            "highlights": "Test summary"
        }
        
        processed = loader.process_example(raw_example)
        
        assert processed.example_id == "test_id"
        assert processed.source_text == "Test article content"
        assert processed.summary_text == "Test summary"
        assert processed.dataset_name == "cnn_dailymail"
    
    def test_cnn_load_dataset(self, sample_cnn_data):
        """Test loading complete CNN/DailyMail dataset"""
        loader = CNNDailyMailLoader(data_dir=sample_cnn_data, cache_enabled=False)
        
        examples = loader.load_dataset("test")
        
        assert len(examples) == 2
        assert all(isinstance(ex, FactualityExample) for ex in examples)
        assert examples[0].dataset_name == "cnn_dailymail"
    
    def test_cnn_load_with_max_examples(self, sample_cnn_data):
        """Test loading CNN/DailyMail with example limit"""
        loader = CNNDailyMailLoader(data_dir=sample_cnn_data, cache_enabled=False)
        
        examples = loader.load_dataset("test", max_examples=1)
        
        assert len(examples) == 1
        assert examples[0].example_id == "cnn_test_1"


class TestXSumLoader:
    """Test XSum dataset loader"""
    
    @pytest.fixture
    def temp_dir(self):
        with tempfile.TemporaryDirectory() as tmpdir:
            yield Path(tmpdir)
    
    @pytest.fixture
    def sample_xsum_data(self, temp_dir):
        """Create sample XSum data"""
        dataset_dir = temp_dir / "raw" / "xsum"
        dataset_dir.mkdir(parents=True)
        
        sample_data = [
            {
                "id": "xsum_test_1",
                "document": "This is a sample BBC article for testing XSum loader functionality. " * 15,
                "summary": "Sample XSum summary for testing."
            },
            {
                "id": "xsum_test_2",
                "document": "Another BBC article for XSum testing purposes. " * 12,
                "summary": "Another XSum summary."
            }
        ]
        
        # Create test split
        test_file = dataset_dir / "test.json"
        with open(test_file, "w", encoding="utf-8") as f:
            json.dump(sample_data, f, ensure_ascii=False, indent=2)
        
        return temp_dir
    
    def test_xsum_loader_initialization(self, temp_dir):
        """Test XSum loader initialization"""
        loader = XSumLoader(data_dir=temp_dir)
        
        assert loader.dataset_name() == "xsum"
    
    def test_xsum_load_raw_data(self, sample_xsum_data):
        """Test loading raw XSum data"""
        loader = XSumLoader(data_dir=sample_xsum_data)
        
        raw_data = loader.load_raw_data("test")
        
        assert len(raw_data) == 2
        assert raw_data[0]["id"] == "xsum_test_1"
        assert "document" in raw_data[0]
        assert "summary" in raw_data[0]
    
    def test_xsum_process_example(self, temp_dir):
        """Test processing XSum examples"""
        loader = XSumLoader(data_dir=temp_dir)
        
        raw_example = {
            "id": "test_id",
            "document": "Test document content",
            "summary": "Test summary"
        }
        
        processed = loader.process_example(raw_example)
        
        assert processed.example_id == "test_id"
        assert processed.source_text == "Test document content"
        assert processed.summary_text == "Test summary"
        assert processed.dataset_name == "xsum"
    
    def test_xsum_load_dataset(self, sample_xsum_data):
        """Test loading complete XSum dataset"""
        loader = XSumLoader(data_dir=sample_xsum_data, cache_enabled=False)
        
        examples = loader.load_dataset("test")
        
        assert len(examples) == 2
        assert all(isinstance(ex, FactualityExample) for ex in examples)
        assert examples[0].dataset_name == "xsum"


class TestCaching:
    """Test dataset caching functionality"""
    
    @pytest.fixture
    def temp_dir(self):
        with tempfile.TemporaryDirectory() as tmpdir:
            yield Path(tmpdir)
    
    @pytest.fixture
    def sample_data_with_cache(self, temp_dir):
        """Create sample data and cache setup"""
        dataset_dir = temp_dir / "raw" / "cnn_dailymail"
        dataset_dir.mkdir(parents=True)
        
        sample_data = [
            {
                "id": "cache_test_1",
                "article": "Article for cache testing. " * 10,
                "highlights": "Summary for cache testing."
            }
        ]
        
        test_file = dataset_dir / "test.json"
        with open(test_file, "w", encoding="utf-8") as f:
            json.dump(sample_data, f)
        
        return temp_dir
    
    def test_cache_creation(self, sample_data_with_cache):
        """Test that cache files are created"""
        loader = CNNDailyMailLoader(data_dir=sample_data_with_cache, cache_enabled=True)
        
        # Load dataset (should create cache)
        examples = loader.load_dataset("test")
        
        assert len(examples) == 1
        
        # Check cache file exists
        cache_dir = sample_data_with_cache / "cache"
        cache_files = list(cache_dir.glob("*.pkl"))
        assert len(cache_files) > 0
    
    def test_cache_loading(self, sample_data_with_cache):
        """Test loading from cache"""
        loader = CNNDailyMailLoader(data_dir=sample_data_with_cache, cache_enabled=True)
        
        # First load (creates cache)
        examples1 = loader.load_dataset("test")
        
        # Second load (should use cache)
        examples2 = loader.load_dataset("test", use_cache=True)
        
        assert len(examples1) == len(examples2)
        assert examples1[0].example_id == examples2[0].example_id
    
    def test_cache_disabled(self, sample_data_with_cache):
        """Test with caching disabled"""
        loader = CNNDailyMailLoader(data_dir=sample_data_with_cache, cache_enabled=False)
        
        examples = loader.load_dataset("test")
        
        assert len(examples) == 1
        
        # No cache should be created
        cache_dir = sample_data_with_cache / "cache"
        if cache_dir.exists():
            cache_files = list(cache_dir.glob("*.pkl"))
            assert len(cache_files) == 0


class TestValidation:
    """Test example validation functionality"""
    
    @pytest.fixture
    def temp_dir(self):
        with tempfile.TemporaryDirectory() as tmpdir:
            yield Path(tmpdir)
    
    def test_example_validation_pass(self):
        """Test valid example passes validation"""
        example = FactualityExample(
            example_id="test_1",
            source_text="This is a valid source text for testing validation functionality.",
            summary_text="Valid summary text."
        )
        
        is_valid, errors = example.validate()
        
        assert is_valid is True
        assert len(errors) == 0
    
    def test_example_validation_fail_short_source(self):
        """Test example fails validation with short source"""
        example = FactualityExample(
            example_id="test_2",
            source_text="Short",  # Too short
            summary_text="Valid summary text."
        )
        
        is_valid, errors = example.validate()
        
        assert is_valid is False
        assert "Source text too short" in errors
    
    def test_example_validation_fail_short_summary(self):
        """Test example fails validation with short summary"""
        example = FactualityExample(
            example_id="test_3",
            source_text="This is a valid source text for testing.",
            summary_text="Hi"  # Too short
        )
        
        is_valid, errors = example.validate()
        
        assert is_valid is False
        assert "Summary text too short" in errors
    
    def test_validation_disabled(self, temp_dir):
        """Test loading with validation disabled"""
        # Create data with invalid examples
        dataset_dir = temp_dir / "raw" / "cnn_dailymail"
        dataset_dir.mkdir(parents=True)
        
        sample_data = [
            {
                "id": "invalid_1",
                "article": "Short",  # Too short, would fail validation
                "highlights": "Also short"  # Too short
            },
            {
                "id": "valid_1",
                "article": "This is a longer article that should pass validation. " * 5,
                "highlights": "Valid summary text."
            }
        ]
        
        test_file = dataset_dir / "test.json"
        with open(test_file, "w") as f:
            json.dump(sample_data, f)
        
        # Load with validation disabled
        loader = CNNDailyMailLoader(
            data_dir=temp_dir,
            cache_enabled=False,
            validate_examples=False
        )
        
        examples = loader.load_dataset("test")
        
        # Should load both examples (validation disabled)
        assert len(examples) == 2


class TestUtilityFunctions:
    """Test utility functions"""
    
    @pytest.fixture
    def temp_dir(self):
        with tempfile.TemporaryDirectory() as tmpdir:
            yield Path(tmpdir)
    
    @pytest.fixture
    def setup_datasets(self, temp_dir):
        """Setup sample datasets"""
        # Create CNN/DailyMail data
        cnn_dir = temp_dir / "raw" / "cnn_dailymail"
        cnn_dir.mkdir(parents=True)
        
        cnn_data = [{"id": "cnn_1", "article": "CNN article", "highlights": "CNN summary"}]
        with open(cnn_dir / "test.json", "w") as f:
            json.dump(cnn_data, f)
        
        # Create XSum data
        xsum_dir = temp_dir / "raw" / "xsum"
        xsum_dir.mkdir(parents=True)
        
        xsum_data = [{"id": "xsum_1", "document": "XSum document", "summary": "XSum summary"}]
        with open(xsum_dir / "test.json", "w") as f:
            json.dump(xsum_data, f)
        
        return temp_dir
    
    def test_get_available_datasets(self):
        """Test getting available datasets"""
        datasets = get_available_datasets()
        
        assert "cnn_dailymail" in datasets
        assert "xsum" in datasets
        assert len(datasets) == 2
    
    def test_quick_load_dataset_cnn(self, setup_datasets):
        """Test quick loading CNN/DailyMail"""
        examples = quick_load_dataset(
            "cnn_dailymail",
            split="test",
            data_dir=setup_datasets,
            use_cache=False
        )
        
        assert len(examples) == 1
        assert examples[0].dataset_name == "cnn_dailymail"
    
    def test_quick_load_dataset_xsum(self, setup_datasets):
        """Test quick loading XSum"""
        examples = quick_load_dataset(
            "xsum",
            split="test", 
            data_dir=setup_datasets,
            use_cache=False
        )
        
        assert len(examples) == 1
        assert examples[0].dataset_name == "xsum"
    
    def test_quick_load_invalid_dataset(self, setup_datasets):
        """Test quick loading invalid dataset"""
        with pytest.raises(ValueError, match="Unknown dataset"):
            quick_load_dataset("invalid_dataset", data_dir=setup_datasets)
    
    def test_validate_dataset_path_valid(self, setup_datasets):
        """Test validating valid dataset path"""
        result = validate_dataset_path("cnn_dailymail", data_dir=setup_datasets)
        
        assert result["status"] == "valid"
        assert len(result["errors"]) == 0
        assert len(result["files_found"]) > 0
    
    def test_validate_dataset_path_missing(self, temp_dir):
        """Test validating missing dataset"""
        result = validate_dataset_path("cnn_dailymail", data_dir=temp_dir)
        
        assert result["status"] == "error"
        assert any("not found" in error for error in result["errors"])
    
    def test_validate_dataset_path_invalid_name(self, temp_dir):
        """Test validating invalid dataset name"""
        result = validate_dataset_path("invalid_dataset", data_dir=temp_dir)
        
        assert result["status"] == "error"
        assert any("Unknown dataset" in error for error in result["errors"])


class TestLoaderIntegration:
    """Test integration scenarios"""
    
    @pytest.fixture
    def temp_dir(self):
        with tempfile.TemporaryDirectory() as tmpdir:
            yield Path(tmpdir)
    
    def test_multi_dataset_loading(self, temp_dir):
        """Test loading multiple datasets"""
        # Setup both datasets
        for dataset_name, fields in [
            ("cnn_dailymail", {"article": "CNN content", "highlights": "CNN summary"}),
            ("xsum", {"document": "XSum content", "summary": "XSum summary"})
        ]:
            dataset_dir = temp_dir / "raw" / dataset_name
            dataset_dir.mkdir(parents=True)
            
            data = [{"id": f"{dataset_name}_1", **fields}]
            with open(dataset_dir / "test.json", "w") as f:
                json.dump(data, f)
        
        # Load both datasets
        cnn_examples = quick_load_dataset("cnn_dailymail", data_dir=temp_dir, use_cache=False)
        xsum_examples = quick_load_dataset("xsum", data_dir=temp_dir, use_cache=False)
        
        assert len(cnn_examples) == 1
        assert len(xsum_examples) == 1
        assert cnn_examples[0].dataset_name == "cnn_dailymail"
        assert xsum_examples[0].dataset_name == "xsum"
    
    def test_loader_error_handling(self, temp_dir):
        """Test loader error handling"""
        # Create corrupted data file
        dataset_dir = temp_dir / "raw" / "cnn_dailymail"
        dataset_dir.mkdir(parents=True)
        
        # Write invalid JSON
        with open(dataset_dir / "test.json", "w") as f:
            f.write("invalid json content")
        
        loader = CNNDailyMailLoader(data_dir=temp_dir)
        
        with pytest.raises(json.JSONDecodeError):
            loader.load_raw_data("test")
