"""
Task-Specific Preprocessors for Factuality Evaluation
====================================================

Basic preprocessors for factuality evaluation tasks.
This is a simplified version that only includes the essential functionality.

Author: Michael Ogunjimi
Institution: University of Manchester
"""

import logging
from dataclasses import dataclass
from datetime import datetime
from typing import Dict, List, Optional, Union, Any, Tuple

from .loaders import FactualityExample

logger = logging.getLogger(__name__)


@dataclass
class ProcessedExample:
    """Container for task-specific processed examples."""

    example_id: str
    source: str
    summary: str
    task_type: str
    original_example: FactualityExample
    processed_at: str
    preprocessing_metadata: Dict[str, Any]

    # Task-specific fields (populated based on task_type)
    summaries: Optional[List[str]] = None  # For ranking tasks
    target_label: Optional[Union[str, int, float, List]] = None
    confidence_score: Optional[float] = None

    def __post_init__(self):
        """Post-initialization validation and setup."""
        if not self.processed_at:
            self.processed_at = datetime.now().isoformat()

        if self.preprocessing_metadata is None:
            self.preprocessing_metadata = {}

    def to_dict(self) -> Dict[str, Any]:
        """Convert to dictionary for serialization."""
        return {
            "example_id": self.example_id,
            "source": self.source,
            "summary": self.summary,
            "task_type": self.task_type,
            "summaries": self.summaries,
            "target_label": self.target_label,
            "confidence_score": self.confidence_score,
            "processed_at": self.processed_at,
            "preprocessing_metadata": self.preprocessing_metadata,
            "original_example": self.original_example.to_dict(),
        }

    @classmethod
    def from_dict(cls, data: Dict[str, Any]) -> "ProcessedExample":
        """Create instance from dictionary."""
        original_example = FactualityExample.from_dict(data.pop("original_example"))
        return cls(original_example=original_example, **data)


class BasePreprocessor:
    """Base class for task-specific preprocessors."""

    def __init__(self, task_type: str):
        """Initialize preprocessor."""
        self.task_type = task_type
        self.logger = logging.getLogger(f"{__name__}.{self.__class__.__name__}")

    def preprocess(self, example: FactualityExample) -> ProcessedExample:
        """Preprocess an example for the specific task."""
        raise NotImplementedError("Subclasses must implement preprocess()")


class EntailmentPreprocessor(BasePreprocessor):
    """Preprocessor for entailment inference task."""

    def __init__(self):
        """Initialize entailment preprocessor."""
        super().__init__(task_type="entailment_inference")

    def preprocess(self, example: FactualityExample) -> ProcessedExample:
        """Preprocess example for entailment inference."""
        return ProcessedExample(
            example_id=example.example_id,
            source=example.source,
            summary=example.summary,
            task_type=self.task_type,
            original_example=example,
            processed_at=datetime.now().isoformat(),
            preprocessing_metadata={
                "dataset": example.dataset_name,
                "task": self.task_type,
            },
            target_label=example.human_label,
        )


class RankingPreprocessor(BasePreprocessor):
    """Preprocessor for summary ranking task."""

    def __init__(self):
        """Initialize ranking preprocessor."""
        super().__init__(task_type="summary_ranking")

    def preprocess(self, example: FactualityExample) -> ProcessedExample:
        """Preprocess example for summary ranking task."""
        return ProcessedExample(
            example_id=example.example_id,
            source=example.source,
            summary=example.summary,
            summaries=[example.summary],  # In real scenario, would include multiple summaries
            task_type=self.task_type,
            original_example=example,
            processed_at=datetime.now().isoformat(),
            preprocessing_metadata={
                "dataset": example.dataset_name,
                "task": self.task_type,
            },
            target_label=getattr(example, "ranking", None),  # Use actual ranking if available
        )


class RatingPreprocessor(BasePreprocessor):
    """Preprocessor for consistency rating task."""

    def __init__(self):
        """Initialize rating preprocessor."""
        super().__init__(task_type="consistency_rating")

    def preprocess(self, example: FactualityExample) -> ProcessedExample:
        """Preprocess example for consistency rating task."""
        return ProcessedExample(
            example_id=example.example_id,
            source=example.source,
            summary=example.summary,
            task_type=self.task_type,
            original_example=example,
            processed_at=datetime.now().isoformat(),
            preprocessing_metadata={
                "dataset": example.dataset_name,
                "task": self.task_type,
            },
            target_label=example.human_label,
        )


class TaskPreprocessorFactory:
    """Factory for creating task-specific preprocessors."""

    TASK_PREPROCESSORS = {
        "entailment_inference": EntailmentPreprocessor,
        "summary_ranking": RankingPreprocessor,
        "consistency_rating": RatingPreprocessor,
    }

    @classmethod
    def create_preprocessor(cls, task_type: str) -> BasePreprocessor:
        """Create preprocessor for specified task type."""
        if task_type not in cls.TASK_PREPROCESSORS:
            supported = ", ".join(cls.TASK_PREPROCESSORS.keys())
            raise ValueError(
                f"Unsupported task type: {task_type}. " f"Supported types: {supported}"
            )

        preprocessor_class = cls.TASK_PREPROCESSORS[task_type]
        return preprocessor_class()

    @classmethod
    def get_supported_tasks(cls) -> List[str]:
        """Get list of supported task types."""
        return list(cls.TASK_PREPROCESSORS.keys())


def preprocess_for_task(
    examples: List[FactualityExample], task_type: str
) -> List[ProcessedExample]:
    """
    Preprocess examples for a specific task.

    Args:
        examples: List of FactualityExample objects
        task_type: Task type for preprocessing

    Returns:
        List of ProcessedExample objects ready for the specified task
    """
    preprocessor = TaskPreprocessorFactory.create_preprocessor(task_type)
    processed_examples = []

    for example in examples:
        try:
            processed_example = preprocessor.preprocess(example)
            processed_examples.append(processed_example)
        except Exception as e:
            logger.warning(f"Failed to preprocess example {example.example_id}: {e}")

    return processed_examples


def validate_example_format(example: Union[FactualityExample, ProcessedExample]) -> bool:
    """Validate that an example has the correct format."""
    if isinstance(example, FactualityExample):
        return True
    elif isinstance(example, ProcessedExample):
        return True
    return False


def create_dataset_specific_preprocessor(
    dataset_name: str, task_type: str
) -> BasePreprocessor:
    """Create a dataset-specific preprocessor if needed."""
    # Currently, we don't have dataset-specific preprocessing logic
    # This is a placeholder for potential future customizations
    return TaskPreprocessorFactory.create_preprocessor(task_type)
