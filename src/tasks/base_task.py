"""
Abstract Base Task for ChatGPT Factuality Evaluation
==================================================

Defines the common interface and shared functionality for all
factuality evaluation tasks. Provides standardized data structures,
evaluation metrics, and task execution patterns.

Author: Michael Ogunjimi
Institution: University of Manchester
Course: MSc AI
"""

import time
from abc import ABC, abstractmethod
from dataclasses import dataclass, asdict
from datetime import datetime
from pathlib import Path
from typing import Dict, List, Any, Optional, Union, Tuple

from ..llm_clients.openai_client import OpenAIClient, APICallResult
from ..prompts.prompt_manager import PromptManager, FormattedPrompt
from ..utils.config import get_config
from ..utils.logging import get_logger, ProgressTracker


@dataclass
class TaskExample:
    """
    Standardized data structure for factuality evaluation examples.

    Provides a common interface for all task types while allowing
    task-specific extensions through the metadata field.
    """

    example_id: str
    source: str
    summary: Optional[str] = None
    summaries: Optional[List[str]] = None  # For ranking tasks
    reference_summaries: Optional[List[str]] = None  # Reference summaries if available
    human_label: Optional[Union[int, float, List[int]]] = None  # Human annotations
    dataset_name: Optional[str] = None
    metadata: Optional[Dict[str, Any]] = None

    def __post_init__(self):
        """Validate example data on creation."""
        if not self.source.strip():
            raise ValueError("Source text cannot be empty")

        # Ensure we have at least one summary format
        if not self.summary and not self.summaries:
            raise ValueError("Must provide either summary or summaries")

        # Initialize metadata if not provided
        if self.metadata is None:
            self.metadata = {}

    def to_dict(self) -> Dict[str, Any]:
        """Convert to dictionary for serialization."""
        return asdict(self)

    def get_summary_for_binary_task(self) -> str:
        """Get summary for binary tasks (entailment, rating)."""
        if self.summary:
            return self.summary
        elif self.summaries and len(self.summaries) > 0:
            return self.summaries[0]  # Use first summary
        else:
            raise ValueError("No summary available for binary task")

    def get_summaries_for_ranking(self) -> List[str]:
        """Get summaries for ranking task."""
        if self.summaries:
            return self.summaries
        elif self.summary:
            return [self.summary]  # Single summary as list
        else:
            raise ValueError("No summaries available for ranking task")


@dataclass
class TaskConfig:
    """
    Configuration for factuality evaluation tasks.

    Centralizes task-specific settings and parameters
    for reproducible experiments.
    """

    task_type: str
    prompt_type: str  # "zero_shot" or "chain_of_thought"
    model_name: str = "gpt-4.1-mini"
    temperature: float = 0.0  # Deterministic for factuality evaluation
    max_tokens: int = 150
    batch_size: int = 10
    max_examples: Optional[int] = None
    include_human_eval: bool = False
    save_intermediate: bool = True
    cache_responses: bool = True
    retry_failed: bool = True

    def __post_init__(self):
        """Validate configuration."""
        valid_task_types = [
            "entailment_inference",
            "summary_ranking",
            "consistency_rating",
        ]
        if self.task_type not in valid_task_types:
            raise ValueError(f"Invalid task_type: {self.task_type}")

        valid_prompt_types = ["zero_shot", "chain_of_thought"]
        if self.prompt_type not in valid_prompt_types:
            raise ValueError(f"Invalid prompt_type: {self.prompt_type}")

        if self.temperature < 0 or self.temperature > 2:
            raise ValueError("Temperature must be between 0 and 2")

        if self.max_tokens <= 0:
            raise ValueError("max_tokens must be positive")

    def to_dict(self) -> Dict[str, Any]:
        """Convert to dictionary."""
        return asdict(self)


@dataclass
class TaskResult:
    """
    Standardized result structure for factuality evaluation.

    Contains the prediction, metadata, and evaluation metrics
    for a single example processed by a factuality task.
    """

    example_id: str
    task_type: str
    prompt_type: str
    prediction: Union[int, float, List[int]]  # Task-specific format
    confidence: Optional[float]
    raw_response: str
    processing_time: float
    cost: float
    tokens_used: int
    timestamp: str
    success: bool
    error_message: Optional[str] = None
    human_label: Optional[Union[int, float, List[int]]] = None
    metadata: Optional[Dict[str, Any]] = None

    def __post_init__(self):
        if self.timestamp is None:
            self.timestamp = datetime.now().isoformat()

        if self.metadata is None:
            self.metadata = {}

    def to_dict(self) -> Dict[str, Any]:
        """Convert to dictionary for analysis and storage."""
        return asdict(self)

    def matches_human_label(self) -> Optional[bool]:
        """Check if prediction matches human label (if available)."""
        if self.human_label is None:
            return None

        # Handle different prediction types
        prediction_value = self.prediction
        
        # Extract numeric value from result objects
        if hasattr(self.prediction, 'rating'):
            prediction_value = self.prediction.rating
        elif hasattr(self.prediction, 'prediction'):
            prediction_value = self.prediction.prediction
        elif hasattr(self.prediction, 'entailment_score'):
            prediction_value = self.prediction.entailment_score
        
        if isinstance(prediction_value, (int, float)) and isinstance(
            self.human_label, (int, float)
        ):
            # For binary or rating tasks
            return abs(prediction_value - self.human_label) < 0.5
        elif isinstance(prediction_value, list) and isinstance(self.human_label, list):
            # For ranking tasks
            return prediction_value == self.human_label
        else:
            return False


class BaseFactualityTask(ABC):
    """
    Abstract base class for all factuality evaluation tasks.

    Defines the common interface and provides shared functionality
    for ChatGPT-based factuality evaluation across different task types.
    """

    def __init__(
        self,
        config: Optional[Dict[str, Any]] = None,
        prompt_manager: Optional[PromptManager] = None,
        llm_client: Optional[OpenAIClient] = None,
    ):
        """
        Initialize base task.

        Args:
            config: Configuration dictionary
            prompt_manager: PromptManager instance
            llm_client: OpenAIClient instance
        """
        self.config = config or get_config()
        self.logger = get_logger(f"{self.__class__.__name__}")

        # Initialize task-specific configuration
        self.task_config = self._create_task_config()

        # Initialize components
        self.prompt_manager = prompt_manager or PromptManager(self.config)
        self.llm_client = llm_client or OpenAIClient(self.config)

        # Result tracking
        self.results: List[TaskResult] = []
        self.failed_examples: List[Tuple[TaskExample, str]] = []

        # Performance tracking
        self.total_examples_processed = 0
        self.total_cost = 0.0
        self.total_time = 0.0

        self.logger.info(
            f"Initialized {self.task_config.task_type} task with {self.task_config.prompt_type} prompts"
        )

    @abstractmethod
    def _create_task_config(self) -> TaskConfig:
        """Create task-specific configuration. Must be implemented by subclasses."""
        pass

    @abstractmethod
    def _validate_example(self, example: TaskExample) -> bool:
        """Validate that example is suitable for this task type."""
        pass

    @abstractmethod
    def _process_api_result(
        self, api_result: APICallResult, example: TaskExample
    ) -> TaskResult:
        """Process API result into task-specific result format."""
        pass

    @abstractmethod
    def evaluate_predictions(self, results: List[TaskResult]) -> Dict[str, float]:
        """Compute task-specific evaluation metrics."""
        pass

    def format_prompt(self, example: TaskExample) -> FormattedPrompt:
        """
        Format prompt for the given example.

        Args:
            example: TaskExample to format prompt for

        Returns:
            FormattedPrompt ready for API call
        """
        # Prepare variables based on task type
        variables = {
            "source": example.source,
            "source_document": example.source  # For compatibility with different templates
        }

        # Add task-specific variables
        if self.task_config.task_type == "summary_ranking":
            summaries = example.get_summaries_for_ranking()
            variables["summaries"] = "\n".join(
                f"Summary {i+1}: {s}" for i, s in enumerate(summaries)
            )
            variables["num_summaries"] = len(summaries)
        else:
            variables["summary"] = example.get_summary_for_binary_task()

        # Format the prompt
        formatted_prompt = self.prompt_manager.format_prompt(
            task_type=self.task_config.task_type,
            prompt_type=self.task_config.prompt_type,
            **variables,
        )

        if not formatted_prompt.validation_passed:
            raise ValueError(
                f"Prompt formatting failed: {formatted_prompt.validation_errors}"
            )

        return formatted_prompt

    async def process_single_example(
        self, example: TaskExample
    ) -> Optional[TaskResult]:
        """
        Process a single example through the complete pipeline.

        Args:
            example: TaskExample to process

        Returns:
            TaskResult or None if processing failed
        """
        start_time = time.time()

        try:
            # Validate example
            if not self._validate_example(example):
                raise ValueError("Example validation failed")

            # Format prompt
            formatted_prompt = self.format_prompt(example)

            # Generate response using LLM client
            api_result = await self.llm_client.evaluate_factuality(
                formatted_prompt=formatted_prompt,
                temperature=self.task_config.temperature,
                max_tokens=self.task_config.max_tokens,
                model=self.task_config.model_name,
            )

            # Process result
            if api_result.parsing_successful:
                task_result = self._process_api_result(api_result, example)
                self.results.append(task_result)

                # Update performance tracking
                self.total_examples_processed += 1
                self.total_cost += api_result.raw_response.cost
                self.total_time += time.time() - start_time

                return task_result
            else:
                error_msg = f"Response parsing failed: {api_result.parsing_errors}"
                self.failed_examples.append((example, error_msg))
                self.logger.error(
                    f"Failed to process example {example.example_id}: {error_msg}"
                )
                return None

        except Exception as e:
            error_msg = f"Processing error: {str(e)}"
            self.failed_examples.append((example, error_msg))
            self.logger.error(
                f"Failed to process example {example.example_id}: {error_msg}"
            )
            return None

    async def process_examples(
        self, examples: List[TaskExample], progress_callback: Optional[callable] = None
    ) -> List[TaskResult]:
        """
        Process multiple examples with progress tracking.

        Args:
            examples: List of TaskExample objects to process
            progress_callback: Optional callback for progress updates

        Returns:
            List of TaskResult objects
        """
        if self.task_config.max_examples:
            examples = examples[: self.task_config.max_examples]

        self.logger.info(f"Processing {len(examples)} examples")

        # Create progress tracker
        progress_tracker = ProgressTracker(
            total=len(examples),
            description=f"{self.task_config.task_type} evaluation",
            experiment_name=self.config.get("experiment_name"),
            show_cost=True,
        )

        results = []
        batch_size = self.task_config.batch_size

        # Process in batches
        for i in range(0, len(examples), batch_size):
            batch = examples[i : i + batch_size]

            # Process batch (could be parallelized further if needed)
            batch_results = []
            for example in batch:
                result = await self.process_single_example(example)
                if result:
                    batch_results.append(result)

                # Update progress
                cost = result.cost if result else 0.0
                progress_tracker.update(cost=cost)

                # Call progress callback if provided
                if progress_callback:
                    progress_callback(len(results) + len(batch_results), len(examples))

            results.extend(batch_results)

            # Save intermediate results if configured
            if (
                self.task_config.save_intermediate
                and len(results) % max(batch_size * 2, 10) == 0  # Save every 2 batches or at least every 10 results
            ):
                await self._save_intermediate_results(results)

        # Finalize progress tracking
        summary = progress_tracker.finish()

        # Save final intermediate results to ensure we always have them
        if self.task_config.save_intermediate and results:
            await self._save_intermediate_results(results)

        self.logger.info(
            f"Completed processing: {len(results)} successful, "
            f"{len(self.failed_examples)} failed, "
            f"Total cost: ${summary['total_cost']:.4f}"
        )

        return results

    async def _save_intermediate_results(self, results: List[TaskResult]) -> None:
        """Save intermediate results to prevent data loss."""
        if not results:
            return

        output_dir = Path(self.config.get("paths.results_dir", "./results"))
        output_dir.mkdir(parents=True, exist_ok=True)

        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        filename = f"{self.task_config.task_type}_{self.task_config.prompt_type}_intermediate_{timestamp}.json"
        filepath = output_dir / filename

        try:
            import json

            with open(filepath, "w") as f:
                json.dump(
                    [result.to_dict() for result in results], f, indent=2, default=str
                )

            self.logger.debug(f"Saved intermediate results to {filepath}")
        except Exception as e:
            self.logger.warning(f"Failed to save intermediate results: {e}")

    def get_performance_summary(self) -> Dict[str, Any]:
        """Get comprehensive performance summary."""
        successful_results = [r for r in self.results if r.success]

        if successful_results:
            avg_processing_time = sum(
                r.processing_time for r in successful_results
            ) / len(successful_results)
            avg_cost_per_example = sum(r.cost for r in successful_results) / len(
                successful_results
            )
            avg_tokens_per_example = sum(
                r.tokens_used for r in successful_results
            ) / len(successful_results)
        else:
            avg_processing_time = 0.0
            avg_cost_per_example = 0.0
            avg_tokens_per_example = 0.0

        return {
            "task_info": {
                "task_type": self.task_config.task_type,
                "prompt_type": self.task_config.prompt_type,
                "model_name": self.task_config.model_name,
            },
            "processing_stats": {
                "total_examples_attempted": self.total_examples_processed
                + len(self.failed_examples),
                "successful_examples": len(successful_results),
                "failed_examples": len(self.failed_examples),
                "success_rate": len(successful_results)
                / max(self.total_examples_processed + len(self.failed_examples), 1),
            },
            "performance_metrics": {
                "total_cost": self.total_cost,
                "total_time": self.total_time,
                "avg_cost_per_example": avg_cost_per_example,
                "avg_processing_time": avg_processing_time,
                "avg_tokens_per_example": avg_tokens_per_example,
            },
            "quality_metrics": (
                self.evaluate_predictions(successful_results)
                if successful_results
                else {}
            ),
        }

    def export_results(
        self,
        output_path: Path,
        include_metadata: bool = True,
        include_failed: bool = True,
    ) -> None:
        """
        Export results to file for analysis.

        Args:
            output_path: Path to save results
            include_metadata: Whether to include task metadata
            include_failed: Whether to include failed examples
        """
        export_data = {
            "task_config": self.task_config.to_dict(),
            "performance_summary": self.get_performance_summary(),
            "results": [result.to_dict() for result in self.results],
            "export_timestamp": datetime.now().isoformat(),
        }

        if include_failed:
            export_data["failed_examples"] = [
                {"example": example.to_dict(), "error": error}
                for example, error in self.failed_examples
            ]

        if include_metadata:
            export_data["metadata"] = {
                "total_examples": len(self.results) + len(self.failed_examples),
                "success_rate": len(self.results)
                / max(len(self.results) + len(self.failed_examples), 1),
                "total_cost": self.total_cost,
                "avg_cost_per_example": self.total_cost / max(len(self.results), 1),
            }

        # Ensure output directory exists
        output_path.parent.mkdir(parents=True, exist_ok=True)

        # Save results
        import json

        with open(output_path, "w") as f:
            json.dump(export_data, f, indent=2, default=str)

        self.logger.info(f"Exported results to {output_path}")


# Utility functions


def validate_task_config(config: Dict[str, Any]) -> bool:
    """
    Validate task configuration.

    Args:
        config: Configuration dictionary

    Returns:
        True if configuration is valid

    Raises:
        ValueError: If configuration is invalid
    """
    required_sections = ["tasks", "prompts", "openai"]

    for section in required_sections:
        if section not in config:
            raise ValueError(f"Missing required configuration section: {section}")

    # Validate task-specific configurations
    tasks_config = config["tasks"]
    required_tasks = ["entailment_inference", "summary_ranking", "consistency_rating"]

    for task in required_tasks:
        if task not in tasks_config:
            raise ValueError(f"Missing task configuration: {task}")

        task_config = tasks_config[task]
        if not task_config.get("enabled", False):
            continue

        # Check required task fields
        required_fields = ["prompt_types", "evaluation_metrics"]
        for field in required_fields:
            if field not in task_config:
                raise ValueError(f"Missing required field in {task} config: {field}")

    return True


def create_task_example(
    example_id: str,
    source: str,
    summary: Optional[str] = None,
    summaries: Optional[List[str]] = None,
    **kwargs,
) -> TaskExample:
    """
    Convenience function to create TaskExample objects.

    Args:
        example_id: Unique identifier
        source: Source document text
        summary: Single summary (for binary tasks)
        summaries: Multiple summaries (for ranking tasks)
        **kwargs: Additional metadata

    Returns:
        TaskExample object
    """
    return TaskExample(
        example_id=example_id,
        source=source,
        summary=summary,
        summaries=summaries,
        **kwargs,
    )
