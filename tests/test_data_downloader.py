"""
Data Downloader Tests
====================

Tests for dataset downloading functionality including HuggingFace integration,
data validation, and preprocessing during download.

Author: Michael Ogunjimi
Institution: University of Manchester
"""

import pytest
import json
import tempfile
from pathlib import Path
from unittest.mock import Mock, patch, MagicMock


class MockHFDataset:
    """Mock HuggingFace dataset for testing"""
    
    def __init__(self, data, splits=None):
        self.data = data or {}
        self.splits = splits or ["train", "validation", "test"]
        
    def __getitem__(self, split):
        return MockSplit(self.data.get(split, []))
    
    def __contains__(self, split):
        return split in self.data


class MockSplit:
    """Mock dataset split"""
    
    def __init__(self, examples):
        self.examples = examples or []
    
    def __len__(self):
        return len(self.examples)
    
    def __iter__(self):
        return iter(self.examples)
    
    def select(self, indices):
        selected = [self.examples[i] for i in indices if i < len(self.examples)]
        return MockSplit(selected)


class DatasetDownloader:
    """Simplified downloader for testing"""
    
    def __init__(self, data_dir="data"):
        self.data_dir = Path(data_dir)
        self.raw_dir = self.data_dir / "raw"
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
                    "clean_text": True
                }
            }
        }
    
    def download_dataset(self, dataset_name, max_samples=None, force_redownload=False):
        """Download and process a single dataset"""
        if dataset_name not in self.dataset_configs:
            return False
            
        config = self.dataset_configs[dataset_name]
        output_dir = self.raw_dir / dataset_name
        
        # Check existing files
        if not force_redownload and output_dir.exists():
            existing_files = list(output_dir.glob("*.json"))
            if existing_files:
                return True
        
        # Create mock dataset
        mock_data = self._create_mock_data(dataset_name)
        
        # Create output directory
        output_dir.mkdir(parents=True, exist_ok=True)
        
        # Process each split
        for split in config["splits"]:
            if split not in mock_data:
                continue
                
            split_data = mock_data[split]
            
            # Apply sampling if specified
            if max_samples and split in max_samples:
                max_for_split = max_samples[split]
                split_data = split_data[:max_for_split]
            
            # Save processed data
            output_file = output_dir / f"{split}.json"
            with open(output_file, "w", encoding="utf-8") as f:
                json.dump(split_data, f, ensure_ascii=False, indent=2)
        
        return True
    
    def _create_mock_data(self, dataset_name):
        """Create mock data for testing"""
        if dataset_name == "cnn_dailymail":
            return {
                "train": [
                    {
                        "id": "cnn_train_1",
                        "article": "This is a sample news article for training. " * 20,
                        "highlights": "Sample summary for training."
                    }
                ] * 10,
                "validation": [
                    {
                        "id": "cnn_val_1", 
                        "article": "This is a sample validation article. " * 15,
                        "highlights": "Validation summary."
                    }
                ] * 5,
                "test": [
                    {
                        "id": "cnn_test_1",
                        "article": "This is a sample test article. " * 10,
                        "highlights": "Test summary."
                    }
                ] * 3
            }
        elif dataset_name == "xsum":
            return {
                "train": [
                    {
                        "id": "xsum_train_1",
                        "document": "This is a sample BBC article for training. " * 25,
                        "summary": "Training summary for XSum."
                    }
                ] * 8,
                "validation": [
                    {
                        "id": "xsum_val_1",
                        "document": "This is a validation BBC article. " * 20,
                        "summary": "Validation summary for XSum."
                    }
                ] * 4,
                "test": [
                    {
                        "id": "xsum_test_1", 
                        "document": "This is a test BBC article. " * 15,
                        "summary": "Test summary for XSum."
                    }
                ] * 2
            }
        return {}
    
    def download_all(self, max_samples_per_split=None, development_mode=False, force_redownload=False):
        """Download all configured datasets"""
        results = {}
        
        max_samples = None
        if max_samples_per_split:
            max_samples = {
                "train": max_samples_per_split,
                "validation": max_samples_per_split // 2,
                "test": max_samples_per_split // 2
            }
        
        for dataset_name in self.dataset_configs:
            results[dataset_name] = self.download_dataset(
                dataset_name, max_samples, force_redownload
            )
            
        return results
    
    def validate_downloads(self):
        """Validate downloaded datasets"""
        validation_results = {}
        
        for dataset_name in self.dataset_configs:
            dataset_dir = self.raw_dir / dataset_name
            
            result = {
                "status": "unknown",
                "splits_validated": {},
                "files_found": [],
                "total_examples": 0
            }
            
            if not dataset_dir.exists():
                result["status"] = "not_found"
                validation_results[dataset_name] = result
                continue
            
            all_splits_valid = True
            config = self.dataset_configs[dataset_name]
            
            for split in config["splits"]:
                split_file = dataset_dir / f"{split}.json"
                split_result = {
                    "status": "unknown",
                    "file_exists": split_file.exists(),
                    "examples_count": 0,
                    "issues": []
                }
                
                if split_file.exists():
                    result["files_found"].append(str(split_file))
                    try:
                        with open(split_file, "r", encoding="utf-8") as f:
                            data = json.load(f)
                        
                        split_result["examples_count"] = len(data)
                        result["total_examples"] += len(data)
                        split_result["status"] = "valid"
                        
                        # Validate a sample
                        if data:
                            sample = data[0]
                            required_fields = ["id"]
                            
                            if dataset_name == "xsum":
                                required_fields.extend(["document", "summary"])
                            else:
                                required_fields.extend(["article", "highlights"])
                            
                            for field in required_fields:
                                if field not in sample:
                                    split_result["issues"].append(f"Missing field: {field}")
                                    split_result["status"] = "invalid"
                                    all_splits_valid = False
                    
                    except Exception as e:
                        split_result["status"] = "error"
                        split_result["issues"].append(f"Error reading: {e}")
                        all_splits_valid = False
                else:
                    split_result["issues"].append("File not found")
                    all_splits_valid = False
                
                result["splits_validated"][split] = split_result
            
            result["status"] = "valid" if all_splits_valid else "invalid"
            validation_results[dataset_name] = result
        
        return validation_results
    
    def get_download_info(self):
        """Get information about configured datasets"""
        info = {}
        for dataset_name, config in self.dataset_configs.items():
            info[dataset_name] = {
                "hf_name": config["hf_name"],
                "splits": config["splits"],
                "preprocessing": config["preprocessing"]
            }
        return info


class TestDatasetDownloader:
    """Test dataset downloading functionality"""
    
    @pytest.fixture
    def temp_dir(self):
        with tempfile.TemporaryDirectory() as tmpdir:
            yield Path(tmpdir)
    
    @pytest.fixture  
    def downloader(self, temp_dir):
        return DatasetDownloader(data_dir=temp_dir)
    
    def test_downloader_initialization(self, temp_dir):
        """Test downloader initialization"""
        downloader = DatasetDownloader(data_dir=temp_dir)
        
        assert downloader.data_dir == temp_dir
        assert downloader.raw_dir == temp_dir / "raw"
        assert "cnn_dailymail" in downloader.dataset_configs
        assert "xsum" in downloader.dataset_configs
    
    def test_download_single_dataset(self, downloader, temp_dir):
        """Test downloading a single dataset"""
        result = downloader.download_dataset("cnn_dailymail")
        
        assert result is True
        
        # Check files were created
        cnn_dir = temp_dir / "raw" / "cnn_dailymail"
        assert cnn_dir.exists()
        assert (cnn_dir / "train.json").exists()
        assert (cnn_dir / "validation.json").exists()
        assert (cnn_dir / "test.json").exists()
    
    def test_download_with_sampling(self, downloader, temp_dir):
        """Test downloading with sample limits"""
        max_samples = {"train": 5, "validation": 2, "test": 1}
        result = downloader.download_dataset("cnn_dailymail", max_samples)
        
        assert result is True
        
        # Check sample counts
        cnn_dir = temp_dir / "raw" / "cnn_dailymail"
        
        with open(cnn_dir / "train.json", "r") as f:
            train_data = json.load(f)
        assert len(train_data) == 5
        
        with open(cnn_dir / "test.json", "r") as f:
            test_data = json.load(f)
        assert len(test_data) == 1
    
    def test_download_invalid_dataset(self, downloader):
        """Test downloading non-existent dataset"""
        result = downloader.download_dataset("invalid_dataset")
        assert result is False
    
    def test_download_all_datasets(self, downloader, temp_dir):
        """Test downloading all configured datasets"""
        results = downloader.download_all(max_samples_per_split=3)
        
        assert "cnn_dailymail" in results
        assert "xsum" in results
        assert results["cnn_dailymail"] is True
        assert results["xsum"] is True
        
        # Check both datasets were created
        assert (temp_dir / "raw" / "cnn_dailymail").exists()
        assert (temp_dir / "raw" / "xsum").exists()
    
    def test_skip_existing_downloads(self, downloader, temp_dir):
        """Test that existing downloads are skipped"""
        # First download
        result1 = downloader.download_dataset("cnn_dailymail")
        assert result1 is True
        
        # Second download should skip
        result2 = downloader.download_dataset("cnn_dailymail")
        assert result2 is True
    
    def test_force_redownload(self, downloader, temp_dir):
        """Test force redownload functionality"""
        # First download
        downloader.download_dataset("cnn_dailymail")
        
        # Force redownload
        result = downloader.download_dataset("cnn_dailymail", force_redownload=True)
        assert result is True
    
    def test_validate_downloads(self, downloader, temp_dir):
        """Test download validation"""
        # Download datasets first
        downloader.download_all(max_samples_per_split=2)
        
        # Validate
        validation_results = downloader.validate_downloads()
        
        assert "cnn_dailymail" in validation_results
        assert "xsum" in validation_results
        
        cnn_result = validation_results["cnn_dailymail"]
        assert cnn_result["status"] == "valid"
        assert len(cnn_result["splits_validated"]) == 3
        assert cnn_result["total_examples"] > 0
    
    def test_validate_missing_datasets(self, downloader):
        """Test validation of missing datasets"""
        validation_results = downloader.validate_downloads()
        
        for dataset_name in ["cnn_dailymail", "xsum"]:
            result = validation_results[dataset_name]
            assert result["status"] == "not_found"
            assert result["total_examples"] == 0
    
    def test_validate_corrupted_file(self, downloader, temp_dir):
        """Test validation of corrupted dataset files"""
        # Create a corrupted file
        dataset_dir = temp_dir / "raw" / "cnn_dailymail"
        dataset_dir.mkdir(parents=True)
        
        corrupted_file = dataset_dir / "train.json"
        with open(corrupted_file, "w") as f:
            f.write("invalid json content")
        
        validation_results = downloader.validate_downloads()
        cnn_result = validation_results["cnn_dailymail"]
        
        assert cnn_result["status"] == "invalid"
        train_result = cnn_result["splits_validated"]["train"]
        assert train_result["status"] == "error"
        assert any("Error reading" in issue for issue in train_result["issues"])
    
    def test_get_download_info(self, downloader):
        """Test getting download information"""
        info = downloader.get_download_info()
        
        assert "cnn_dailymail" in info
        assert "xsum" in info
        
        cnn_info = info["cnn_dailymail"]
        assert "hf_name" in cnn_info
        assert "splits" in cnn_info
        assert "preprocessing" in cnn_info
        assert cnn_info["hf_name"] == "ccdv/cnn_dailymail"
    
    def test_dataset_configs(self, downloader):
        """Test dataset configuration structure"""
        configs = downloader.dataset_configs
        
        for dataset_name in ["cnn_dailymail", "xsum"]:
            config = configs[dataset_name]
            
            # Required fields
            assert "hf_name" in config
            assert "splits" in config
            assert "key_mappings" in config
            assert "preprocessing" in config
            
            # Key mappings
            mappings = config["key_mappings"]
            assert "source" in mappings
            assert "summary" in mappings
            assert "id" in mappings
            
            # Preprocessing config
            preprocessing = config["preprocessing"]
            assert "max_source_length" in preprocessing
            assert "max_summary_length" in preprocessing
            assert "min_summary_length" in preprocessing


class TestDownloadUtilities:
    """Test download utility functions"""
    
    @pytest.fixture
    def temp_dir(self):
        """Create temporary directory for tests"""
        with tempfile.TemporaryDirectory() as tmp:
            yield tmp
    
    def test_download_datasets_function(self, temp_dir):
        """Test the standalone download function"""
        # This would test a standalone function like download_datasets()
        # For now, just test that basic functionality works
        downloader = DatasetDownloader(data_dir=temp_dir)
        
        results = downloader.download_all(
            max_samples_per_split=2,
            development_mode=True
        )
        
        assert isinstance(results, dict)
        assert all(isinstance(v, bool) for v in results.values())


class TestDownloadIntegration:
    """Test integration scenarios"""
    
    @pytest.fixture
    def temp_dir(self):
        with tempfile.TemporaryDirectory() as tmpdir:
            yield Path(tmpdir)
    
    def test_full_download_workflow(self, temp_dir):
        """Test complete download workflow"""
        downloader = DatasetDownloader(data_dir=temp_dir)
        
        # 1. Download datasets
        download_results = downloader.download_all(max_samples_per_split=3)
        assert all(download_results.values())
        
        # 2. Validate downloads
        validation_results = downloader.validate_downloads()
        for dataset_name in ["cnn_dailymail", "xsum"]:
            assert validation_results[dataset_name]["status"] == "valid"
        
        # 3. Check file structure
        for dataset_name in ["cnn_dailymail", "xsum"]:
            dataset_dir = temp_dir / "raw" / dataset_name
            assert dataset_dir.exists()
            
            for split in ["train", "validation", "test"]:
                split_file = dataset_dir / f"{split}.json"
                assert split_file.exists()
                
                # Verify content
                with open(split_file, "r") as f:
                    data = json.load(f)
                assert isinstance(data, list)
                assert len(data) > 0
    
    def test_development_mode_workflow(self, temp_dir):
        """Test development mode with limited samples"""
        downloader = DatasetDownloader(data_dir=temp_dir)
        
        # Download in development mode
        results = downloader.download_all(
            max_samples_per_split=1,
            development_mode=True
        )
        
        assert all(results.values())
        
        # Verify sample counts are limited
        for dataset_name in ["cnn_dailymail", "xsum"]:
            dataset_dir = temp_dir / "raw" / dataset_name
            
            with open(dataset_dir / "train.json", "r") as f:
                train_data = json.load(f)
            assert len(train_data) == 1  # Limited by max_samples_per_split
