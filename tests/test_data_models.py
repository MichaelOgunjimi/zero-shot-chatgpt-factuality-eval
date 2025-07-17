"""
Essential Data Model Tests
=========================

Tests only critical validation for FactualityExample and TaskResult.
Focuses on data integrity essential for thesis research.

Author: Michael Ogunjimi
Institution: University of Manchester
"""

import pytest
from dataclasses import dataclass
from typing import Union, List, Optional, Dict, Any


@dataclass
class FactualityExample:
    """Core data model for testing"""
    example_id: str
    source: str
    summary: str = None
    summaries: List[str] = None
    human_label: Union[int, float, List[int]] = None

    def __post_init__(self):
        if not self.source.strip():
            raise ValueError("Source text cannot be empty")
        if not self.summary and not self.summaries:
            raise ValueError("Must provide either summary or summaries")


@dataclass 
class TaskResult:
    """Core result model for testing"""
    example_id: str
    task_type: str
    prediction: Union[int, float, List[int]]
    success: bool
    human_label: Union[int, float, List[int]] = None
    error_message: str = None

    def matches_human_label(self) -> Optional[bool]:
        if self.human_label is None:
            return None
        if isinstance(self.prediction, (int, float)) and isinstance(self.human_label, (int, float)):
            return abs(self.prediction - self.human_label) < 0.5
        return self.prediction == self.human_label


class TestFactualityExample:
    """Test essential FactualityExample functionality"""
    
    def test_valid_creation(self):
        """Test basic valid example creation"""
        example = FactualityExample(
            example_id="test_001", 
            source="Source text", 
            summary="Summary text"
        )
        assert example.example_id == "test_001"
        assert example.source == "Source text"
    
    def test_empty_source_fails(self):
        """Test empty source validation"""
        with pytest.raises(ValueError, match="Source text cannot be empty"):
            FactualityExample(example_id="test", source="", summary="Summary")
    
    def test_no_summary_fails(self):
        """Test missing summary validation"""
        with pytest.raises(ValueError, match="Must provide either summary or summaries"):
            FactualityExample(example_id="test", source="Source")


class TestTaskResult:
    """Test essential TaskResult functionality"""
    
    def test_successful_result(self):
        """Test successful task result"""
        result = TaskResult(
            example_id="test_001",
            task_type="entailment_inference", 
            prediction=1,
            success=True
        )
        assert result.success is True
        assert result.prediction == 1
    
    def test_failed_result(self):
        """Test failed task result"""
        result = TaskResult(
            example_id="test_002",
            task_type="consistency_rating",
            prediction=None,
            success=False,
            error_message="API timeout"
        )
        assert result.success is False
        assert result.error_message == "API timeout"
    
    def test_human_label_matching(self):
        """Test human label comparison logic"""
        # Binary task exact match
        result1 = TaskResult(
            example_id="test", 
            task_type="entailment_inference",
            prediction=1, 
            success=True, 
            human_label=1
        )
        assert result1.matches_human_label() is True
        
        # Binary task mismatch
        result2 = TaskResult(
            example_id="test", 
            task_type="entailment_inference",
            prediction=0, 
            success=True, 
            human_label=1
        )
        assert result2.matches_human_label() is False
        
        # No human label
        result3 = TaskResult(
            example_id="test", 
            task_type="entailment_inference",
            prediction=1, 
            success=True
        )
        assert result3.matches_human_label() is None