"""
Comprehensive Prompt Management System for ChatGPT Factuality Evaluation
========================================================================

Unified prompt management system that handles loading, formatting, validation,
and optimization of prompts for all three factuality evaluation tasks with
both zero-shot and chain-of-thought approaches.

This module provides academic-quality prompt engineering specifically designed
for factuality evaluation research, with emphasis on reproducibility and
systematic prompt design.

Author: Michael Ogunjimi
Institution: University of Manchester
Course: MSc AI
"""

import hashlib
import json
import logging
import re
from abc import ABC, abstractmethod
from dataclasses import dataclass, asdict
from datetime import datetime
from pathlib import Path
from typing import Dict, List, Any, Optional, Tuple

from ..utils.config import get_config
from ..utils.logging import get_logger

logger = get_logger(__name__)


@dataclass
class PromptTemplate:
    """
    Template for prompt with comprehensive metadata.

    Stores prompt templates with version control, validation,
    and academic research metadata.
    """

    name: str
    task_type: str  # "entailment_inference", "summary_ranking", "consistency_rating"
    prompt_type: str  # "zero_shot", "chain_of_thought"
    template: str
    variables: List[str]
    description: str
    version: str = "1.0"
    created_date: Optional[str] = None
    author: str = "Michael Ogunjimi"
    validation_examples: Optional[List[Dict]] = None

    def __post_init__(self):
        if self.created_date is None:
            self.created_date = datetime.now().isoformat()

        # Extract variables from template if not provided
        if not self.variables:
            self.variables = self._extract_variables()

    def _extract_variables(self) -> List[str]:
        """Extract variable names from template string."""
        # Find all variables in {variable} format
        variables = set(re.findall(r"\{(\w+)\}", self.template))
        return sorted(list(variables))

    def get_hash(self) -> str:
        """Get hash of template for version tracking."""
        content = f"{self.template}{self.variables}{self.version}"
        return hashlib.md5(content.encode()).hexdigest()[:8]

    def validate_template(self) -> Tuple[bool, List[str]]:
        """
        Validate template structure and content.

        Returns:
            Tuple of (is_valid, error_messages)
        """
        errors = []

        # Check for basic structure
        if not self.template.strip():
            errors.append("Template cannot be empty")

        # Check for required task-specific elements
        if self.task_type == "entailment_inference":
            if (
                "ENTAILMENT" not in self.template
                or "CONTRADICTION" not in self.template
            ):
                errors.append(
                    "Entailment inference template must include ENTAILMENT and CONTRADICTION options"
                )

        elif self.task_type == "summary_ranking":
            if (
                "rank" not in self.template.lower()
                and "order" not in self.template.lower()
            ):
                errors.append(
                    "Summary ranking template should include ranking instructions"
                )

        elif self.task_type == "consistency_rating":
            if not any(scale in self.template for scale in ["0-100", "0–100", "0-10", "0–10", "1-5", "1–5"]):
                errors.append(
                    "Consistency rating template should include a rating scale"
                )

        # Check for chain-of-thought requirements
        if self.prompt_type == "chain_of_thought":
            cot_indicators = ["step", "analyze", "think", "reason", "first", "then"]
            if not any(
                indicator in self.template.lower() for indicator in cot_indicators
            ):
                errors.append(
                    "Chain-of-thought template should include reasoning indicators"
                )

        # Check variable consistency
        template_vars = set(self._extract_variables())
        declared_vars = set(self.variables)

        if template_vars != declared_vars:
            missing = template_vars - declared_vars
            extra = declared_vars - template_vars
            if missing:
                errors.append(f"Template contains undeclared variables: {missing}")
            if extra:
                errors.append(f"Declared variables not used in template: {extra}")

        return len(errors) == 0, errors

    def to_dict(self) -> Dict[str, Any]:
        """Convert to dictionary for serialization."""
        return asdict(self)


@dataclass
class FormattedPrompt:
    """
    Formatted prompt ready for API call with metadata.

    Contains the final prompt text along with metadata needed
    for tracking, evaluation, and reproducibility.
    """

    prompt_text: str
    template_name: str
    task_type: str
    prompt_type: str
    variables_used: Dict[str, Any]
    formatted_at: str
    token_count: Optional[int] = None
    validation_passed: bool = True
    validation_errors: Optional[List[str]] = None

    def __post_init__(self):
        if self.formatted_at is None:
            self.formatted_at = datetime.now().isoformat()

    def estimate_tokens(self) -> int:
        """Rough token estimation for cost calculation."""
        # Simple estimation: ~4 characters per token
        estimated = len(self.prompt_text) // 4
        self.token_count = estimated
        return estimated

    def to_dict(self) -> Dict[str, Any]:
        """Convert to dictionary for logging and analysis."""
        return asdict(self)


class BasePromptFormatter(ABC):
    """Abstract base class for prompt formatters."""

    @abstractmethod
    def format_prompt(self, template: str, **kwargs) -> str:
        """Format template with provided variables."""
        pass

    @abstractmethod
    def validate_variables(
        self, template: str, variables: Dict[str, Any]
    ) -> Tuple[bool, List[str]]:
        """Validate that all required variables are provided."""
        pass


class StandardPromptFormatter(BasePromptFormatter):
    """
    Standard prompt formatter for basic variable substitution.

    Handles simple {variable} substitution with validation
    and error handling for academic research requirements.
    """

    def format_prompt(self, template: str, **kwargs) -> str:
        """
        Format prompt with standard variable substitution.

        Args:
            template: Template string with {variable} placeholders
            **kwargs: Variables to substitute

        Returns:
            Formatted prompt string

        Raises:
            ValueError: If required variables are missing
        """
        try:
            # Clean and normalize variables
            cleaned_kwargs = self._clean_variables(kwargs)

            # Format the template
            formatted = template.format(**cleaned_kwargs)

            # Post-process for consistency
            formatted = self._post_process(formatted)

            return formatted

        except KeyError as e:
            missing_var = str(e).strip("'\"")
            raise ValueError(f"Missing required variable: {missing_var}")

    def _clean_variables(self, variables: Dict[str, Any]) -> Dict[str, Any]:
        """Clean and normalize variables for formatting."""
        cleaned = {}

        for key, value in variables.items():
            if isinstance(value, str):
                # Remove extra whitespace and normalize
                cleaned[key] = " ".join(value.split())
            elif isinstance(value, list):
                # Handle list formatting for rankings
                if all(isinstance(item, str) for item in value):
                    cleaned[key] = "\n".join(
                        f"{i+1}. {item}" for i, item in enumerate(value)
                    )
                else:
                    cleaned[key] = str(value)
            else:
                cleaned[key] = str(value)

        return cleaned

    def _post_process(self, formatted_prompt: str) -> str:
        """Post-process formatted prompt for consistency."""
        # Remove excessive whitespace
        formatted_prompt = re.sub(r"\n\s*\n\s*\n", "\n\n", formatted_prompt)
        formatted_prompt = re.sub(r"[ \t]+", " ", formatted_prompt)

        # Ensure proper spacing around punctuation
        formatted_prompt = re.sub(r"\s+([.!?])", r"\1", formatted_prompt)

        return formatted_prompt.strip()

    def validate_variables(
        self, template: str, variables: Dict[str, Any]
    ) -> Tuple[bool, List[str]]:
        """
        Validate that all required variables are provided.

        Args:
            template: Template string
            variables: Dictionary of variables

        Returns:
            Tuple of (is_valid, error_messages)
        """
        errors = []

        # Extract required variables from template
        required_vars = set(re.findall(r"\{(\w+)\}", template))
        provided_vars = set(variables.keys())

        # Check for missing variables
        missing_vars = required_vars - provided_vars
        if missing_vars:
            errors.append(
                f"Missing required variables: {', '.join(sorted(missing_vars))}"
            )

        # Check for empty variables
        empty_vars = [
            var
            for var, value in variables.items()
            if var in required_vars and (not value or str(value).strip() == "")
        ]
        if empty_vars:
            errors.append(
                f"Empty variables not allowed: {', '.join(sorted(empty_vars))}"
            )

        return len(errors) == 0, errors


class ChainOfThoughtFormatter(BasePromptFormatter):
    """
    Specialized formatter for chain-of-thought prompts.

    Handles step-by-step reasoning prompts with structured
    thinking sections and proper formatting for academic analysis.
    """

    def format_prompt(self, template: str, **kwargs) -> str:
        """Format chain-of-thought prompt with reasoning structure."""
        # Add reasoning structure if not present
        if not self._has_reasoning_structure(template):
            template = self._add_reasoning_structure(template)

        # Standard formatting
        formatted = template.format(**kwargs)

        # Enhance reasoning sections
        formatted = self._enhance_reasoning_sections(formatted)

        return formatted

    def _has_reasoning_structure(self, template: str) -> bool:
        """Check if template already has reasoning structure."""
        reasoning_indicators = [
            "step",
            "first",
            "then",
            "next",
            "finally",
            "analyze",
            "consider",
            "examine",
            "think",
        ]
        return any(indicator in template.lower() for indicator in reasoning_indicators)

    def _add_reasoning_structure(self, template: str) -> str:
        """Add reasoning structure to template if missing."""
        reasoning_prefix = """Let me analyze this step by step:

1. First, I'll examine the key information
2. Then, I'll consider the relevant factors  
3. Finally, I'll provide my conclusion

"""
        return reasoning_prefix + template

    def _enhance_reasoning_sections(self, formatted_prompt: str) -> str:
        """Enhance reasoning sections for clarity."""
        # Add clear section separators
        enhanced = formatted_prompt

        # Ensure numbered steps are well-formatted
        enhanced = re.sub(r"(\d+)\.\s*([A-Z])", r"\1. \2", enhanced)

        # Add emphasis to conclusion section
        enhanced = re.sub(
            r"(conclusion|answer|final|decision):\s*",
            r"\1: ",
            enhanced,
            flags=re.IGNORECASE,
        )

        return enhanced

    def validate_variables(
        self, template: str, variables: Dict[str, Any]
    ) -> Tuple[bool, List[str]]:
        """Validate variables for chain-of-thought templates."""
        # Use standard validation plus CoT-specific checks
        is_valid, errors = StandardPromptFormatter().validate_variables(
            template, variables
        )

        # Additional CoT validation
        if "source" in variables and "summary" in variables:
            source_length = len(str(variables["source"]))
            summary_length = len(str(variables["summary"]))

            # Check for reasonable length ratios
            if source_length > 0 and summary_length / source_length > 1.5:
                errors.append(
                    "Summary appears longer than source - may indicate data issues"
                )

        return len(errors) == 0, errors


class PromptManager:
    """
    Comprehensive prompt management system for factuality evaluation.

    Manages loading, formatting, validation, and optimization of prompts
    for all three factuality evaluation tasks with academic research standards.
    """

    def __init__(self, config: Optional[Dict[str, Any]] = None):
        """
        Initialize prompt manager.

        Args:
            config: Configuration dictionary
        """
        self.config = config or get_config()
        self.logger = get_logger(self.__class__.__name__)

        # Setup paths
        prompts_config = self.config.get("prompts", {})
        self.templates_dir = Path(prompts_config.get("templates", {}).get("directory", "./prompts"))
        self.templates_dir.mkdir(parents=True, exist_ok=True)

        # Formatting settings
        self.max_prompt_length = prompts_config.get("templates", {}).get("max_length", 8000)
        self.validate_format = prompts_config.get("templates", {}).get("validate_format", True)

        # Initialize formatters
        self.formatters = {
            "standard": StandardPromptFormatter(),
            "chain_of_thought": ChainOfThoughtFormatter(),
        }

        # Template storage
        self.templates: Dict[str, PromptTemplate] = {}
        self.prompt_cache: Dict[str, FormattedPrompt] = {}

        # Load templates
        self._load_all_templates()
        self._create_default_templates_if_missing()

        self.logger.info(
            f"Initialized prompt manager with {len(self.templates)} templates"
        )

    def _load_all_templates(self) -> None:
        """Load all prompt templates from disk."""
        tasks = ["entailment_inference", "summary_ranking", "consistency_rating"]
        prompt_types = ["zero_shot", "chain_of_thought"]

        for task in tasks:
            task_dir = self.templates_dir / task
            if not task_dir.exists():
                self.logger.warning(f"Task directory not found: {task_dir}")
                continue

            for prompt_type in prompt_types:
                template_file = task_dir / f"{prompt_type}.txt"
                if template_file.exists():
                    self._load_template_file(template_file, task, prompt_type)
                else:
                    self.logger.debug(f"Template file not found: {template_file}")

    def _load_template_file(self, file_path: Path, task: str, prompt_type: str) -> None:
        """Load a single template file."""
        try:
            with open(file_path, "r", encoding="utf-8") as f:
                template_content = f.read().strip()

            # Create template object
            template_name = f"{task}_{prompt_type}"
            template = PromptTemplate(
                name=template_name,
                task_type=task,
                prompt_type=prompt_type,
                template=template_content,
                variables=[],  # Will be extracted automatically
                description=f"{prompt_type.replace('_', ' ').title()} prompt for {task.replace('_', ' ')}",
                version="1.0",
            )

            # Validate template
            is_valid, errors = template.validate_template()
            if not is_valid:
                self.logger.warning(
                    f"Template validation failed for {template_name}: {errors}"
                )

            self.templates[template_name] = template
            self.logger.debug(f"Loaded template: {template_name}")

        except Exception as e:
            self.logger.error(f"Failed to load template from {file_path}: {e}")

    def _create_default_templates_if_missing(self) -> None:
        """Create default templates if not found on disk."""
        default_templates = self._get_default_templates()

        for template_name, template_data in default_templates.items():
            if template_name not in self.templates:
                self.logger.info(f"Creating default template: {template_name}")

                # Create template object
                template = PromptTemplate(**template_data)
                self.templates[template_name] = template

                # Save to disk
                self._save_template_to_disk(template)

    def _get_default_templates(self) -> Dict[str, Dict[str, Any]]:
        """Get default template definitions for academic factuality evaluation."""
        return {
            # Entailment Inference Templates
            "entailment_inference_zero_shot": {
                "name": "entailment_inference_zero_shot",
                "task_type": "entailment_inference",
                "prompt_type": "zero_shot",
                "template": """Determine whether the summary is factually consistent with the source document.

Source Document:
{source}

Summary:
{summary}

Question: Is the summary factually consistent with the source document?

Instructions:
- Answer ENTAILMENT if the summary is factually consistent with the source
- Answer CONTRADICTION if the summary contains information that contradicts the source
- Base your judgment only on factual accuracy, not writing quality

Answer: """,
                "variables": ["source", "summary"],
                "description": "Zero-shot prompt for binary entailment inference task",
            },
            "entailment_inference_chain_of_thought": {
                "name": "entailment_inference_chain_of_thought",
                "task_type": "entailment_inference",
                "prompt_type": "chain_of_thought",
                "template": """Determine whether the summary is factually consistent with the source document.

Source Document:
{source}

Summary:
{summary}

Let me analyze this step-by-step:

1. Key facts in the source document:
   - I'll identify the main factual claims and information

2. Claims made in the summary:
   - I'll extract all factual assertions from the summary

3. Fact-checking analysis:
   - I'll compare each summary claim against the source information
   - I'll look for any contradictions or unsupported claims

4. Final judgment:
   - Based on my analysis, I'll determine factual consistency

Question: Is the summary factually consistent with the source document?

Answer: """,
                "variables": ["source", "summary"],
                "description": "Chain-of-thought prompt for entailment inference with step-by-step reasoning",
            },
            # Summary Ranking Templates
            "summary_ranking_zero_shot": {
                "name": "summary_ranking_zero_shot",
                "task_type": "summary_ranking",
                "prompt_type": "zero_shot",
                "template": """Rank the following summaries by their factual consistency with the source document.

Source Document:
{source}

Summaries to rank:
{summaries}

Instructions:
- Rank from 1 (most factually consistent) to {num_summaries} (least factually consistent)
- Focus only on factual accuracy, not writing style or fluency
- Ensure each summary gets a unique rank

Ranking (format: Summary X: Rank Y):""",
                "variables": ["source", "summaries", "num_summaries"],
                "description": "Zero-shot prompt for ranking summaries by factual consistency",
            },
            "summary_ranking_chain_of_thought": {
                "name": "summary_ranking_chain_of_thought",
                "task_type": "summary_ranking",
                "prompt_type": "chain_of_thought",
                "template": """Rank the following summaries by their factual consistency with the source document.

Source Document:
{source}

Summaries to rank:
{summaries}

Let me analyze this systematically:

1. Source document analysis:
   - I'll identify the key facts and information in the source

2. Individual summary evaluation:
   - For each summary, I'll check factual accuracy against the source
   - I'll note any contradictions or unsupported claims

3. Comparative ranking:
   - I'll compare summaries based on factual consistency
   - I'll assign ranks from most to least consistent

4. Final ranking:
   - I'll provide the final ranking with justification

Ranking (format: Summary X: Rank Y):""",
                "variables": ["source", "summaries", "num_summaries"],
                "description": "Chain-of-thought prompt for summary ranking with detailed analysis",
            },
            # Consistency Rating Templates
            "consistency_rating_zero_shot": {
                "name": "consistency_rating_zero_shot",
                "task_type": "consistency_rating",
                "prompt_type": "zero_shot",
                "template": """Rate the factual consistency of the summary with respect to the source document.

Source Document:
{source}

Summary:
{summary}

Instructions:
- Provide a rating from 0 to 100
- 0 = Completely inconsistent (major contradictions)
- 50 = Partially consistent (some errors or unsupported claims)
- 100 = Fully consistent (all information is accurate)
- Focus only on factual accuracy

Rating (0-100): """,
                "variables": ["source", "summary"],
                "description": "Zero-shot prompt for 0-100 consistency rating",
            },
            "consistency_rating_chain_of_thought": {
                "name": "consistency_rating_chain_of_thought",
                "task_type": "consistency_rating",
                "prompt_type": "chain_of_thought",
                "template": """Rate the factual consistency of the summary with respect to the source document.

Source Document:
{source}

Summary:
{summary}

Let me evaluate this carefully:

1. Source information analysis:
   - I'll identify all factual claims in the source document

2. Summary content review:
   - I'll extract all factual assertions from the summary

3. Consistency assessment:
   - I'll check each summary claim against the source
   - I'll identify any contradictions or unsupported information
   - I'll assess the overall accuracy

4. Rating determination:
   - Based on my analysis, I'll assign a score from 0-100
   - 0 = Completely inconsistent, 50 = Partially consistent, 100 = Fully consistent

Rating (0-100): """,
                "variables": ["source", "summary"],
                "description": "Chain-of-thought prompt for consistency rating with detailed reasoning",
            },
        }

    def _save_template_to_disk(self, template: PromptTemplate) -> None:
        """Save template to disk for persistence."""
        task_dir = self.templates_dir / template.task_type
        task_dir.mkdir(parents=True, exist_ok=True)

        template_file = task_dir / f"{template.prompt_type}.txt"

        try:
            with open(template_file, "w", encoding="utf-8") as f:
                f.write(template.template)
            self.logger.debug(f"Saved template to disk: {template_file}")
        except Exception as e:
            self.logger.error(f"Failed to save template {template.name}: {e}")

    def format_prompt(
        self, task_type: str, prompt_type: str, **variables
    ) -> FormattedPrompt:
        """
        Format a prompt for the specified task and type.

        Args:
            task_type: Type of task ("entailment_inference", "summary_ranking", "consistency_rating")
            prompt_type: Type of prompt ("zero_shot", "chain_of_thought")
            **variables: Variables to substitute in the template

        Returns:
            FormattedPrompt object ready for API call

        Raises:
            ValueError: If template not found or variables invalid
        """
        template_name = f"{task_type}_{prompt_type}"

        if template_name not in self.templates:
            raise ValueError(f"Template not found: {template_name}")

        template = self.templates[template_name]

        # Get appropriate formatter
        formatter = self.formatters.get(prompt_type, self.formatters["standard"])

        # Validate variables
        if self.validate_format:
            is_valid, errors = formatter.validate_variables(
                template.template, variables
            )
            if not is_valid:
                formatted_prompt = FormattedPrompt(
                    prompt_text="",
                    template_name=template_name,
                    task_type=task_type,
                    prompt_type=prompt_type,
                    variables_used=variables,
                    formatted_at=datetime.now().isoformat(),
                    validation_passed=False,
                    validation_errors=errors,
                )
                return formatted_prompt

        # Format the prompt
        try:
            formatted_text = formatter.format_prompt(template.template, **variables)

            # Check length limits
            if len(formatted_text) > self.max_prompt_length:
                self.logger.warning(
                    f"Prompt exceeds max length: {len(formatted_text)} > {self.max_prompt_length}"
                )

            # Create formatted prompt object
            formatted_prompt = FormattedPrompt(
                prompt_text=formatted_text,
                template_name=template_name,
                task_type=task_type,
                prompt_type=prompt_type,
                variables_used=variables,
                formatted_at=datetime.now().isoformat(),
                validation_passed=True,
                validation_errors=None,
            )

            # Estimate tokens
            formatted_prompt.estimate_tokens()

            # Cache the result
            cache_key = f"{template_name}_{hash(str(sorted(variables.items())))}"
            self.prompt_cache[cache_key] = formatted_prompt

            return formatted_prompt

        except Exception as e:
            self.logger.error(f"Failed to format prompt {template_name}: {e}")
            raise ValueError(f"Failed to format prompt: {e}")

    def get_template(self, template_name: str) -> Optional[PromptTemplate]:
        """Get template by name."""
        return self.templates.get(template_name)

    def list_templates(
        self, task_type: Optional[str] = None, prompt_type: Optional[str] = None
    ) -> List[str]:
        """
        List available templates with optional filtering.

        Args:
            task_type: Filter by task type
            prompt_type: Filter by prompt type

        Returns:
            List of template names
        """
        templates = []

        for name, template in self.templates.items():
            if task_type and template.task_type != task_type:
                continue
            if prompt_type and template.prompt_type != prompt_type:
                continue
            templates.append(name)

        return sorted(templates)

    def validate_template(self, template_name: str) -> Tuple[bool, List[str]]:
        """
        Validate a specific template.

        Args:
            template_name: Name of template to validate

        Returns:
            Tuple of (is_valid, error_messages)
        """
        if template_name not in self.templates:
            return False, [f"Template not found: {template_name}"]

        template = self.templates[template_name]
        return template.validate_template()

    def add_template(self, template: PromptTemplate, save_to_disk: bool = True) -> None:
        """
        Add a new template to the manager.

        Args:
            template: PromptTemplate object to add
            save_to_disk: Whether to save template to disk
        """
        # Validate template
        is_valid, errors = template.validate_template()
        if not is_valid:
            raise ValueError(f"Invalid template: {errors}")

        self.templates[template.name] = template

        if save_to_disk:
            self._save_template_to_disk(template)

        self.logger.info(f"Added template: {template.name}")

    def clear_cache(self) -> None:
        """Clear the prompt cache."""
        self.prompt_cache.clear()
        self.logger.info("Prompt cache cleared")

    def get_cache_stats(self) -> Dict[str, Any]:
        """Get cache statistics."""
        return {
            "cache_size": len(self.prompt_cache),
            "templates_loaded": len(self.templates),
            "memory_usage_estimate": sum(
                len(fp.prompt_text) for fp in self.prompt_cache.values()
            ),
        }

    def export_templates(self, output_file: Path) -> None:
        """
        Export all templates to JSON file for backup and analysis.

        Args:
            output_file: Path to output file
        """
        export_data = {
            "metadata": {
                "export_timestamp": datetime.now().isoformat(),
                "total_templates": len(self.templates),
                "config": self.config.get("prompts", {}),
                "version": "1.0",
            },
            "templates": {},
        }

        for name, template in self.templates.items():
            export_data["templates"][name] = template.to_dict()

        output_file.parent.mkdir(parents=True, exist_ok=True)

        with open(output_file, "w", encoding="utf-8") as f:
            json.dump(export_data, f, indent=2, ensure_ascii=False)

        self.logger.info(f"Exported {len(self.templates)} templates to {output_file}")


# Convenience functions


def create_prompt_manager(config: Optional[Dict[str, Any]] = None) -> PromptManager:
    """
    Create prompt manager instance.

    Args:
        config: Configuration dictionary

    Returns:
        PromptManager instance
    """
    return PromptManager(config)


def validate_prompt_config(config: Dict[str, Any]) -> bool:
    """
    Validate prompt configuration.

    Args:
        config: Configuration dictionary

    Returns:
        True if configuration is valid

    Raises:
        ValueError: If configuration is invalid
    """
    if "prompts" not in config:
        raise ValueError("Missing 'prompts' section in configuration")

    prompts_config = config["prompts"]

    # Check for required fields
    required_fields = ["templates_dir", "max_length", "validate_format"]
    for field in required_fields:
        if field not in prompts_config:
            logging.warning(f"Missing optional prompt config field: {field}")

    return True


def quick_format_prompt(
    task_type: str,
    prompt_type: str,
    source: str,
    summary: str,
    summaries: Optional[List[str]] = None,
    config: Optional[Dict[str, Any]] = None,
) -> str:
    """
    Quick utility for formatting common prompts.

    Args:
        task_type: Type of task
        prompt_type: Type of prompt
        source: Source document text
        summary: Summary text (for single summary tasks)
        summaries: List of summaries (for ranking task)
        config: Optional configuration

    Returns:
        Formatted prompt text
    """
    manager = create_prompt_manager(config)

    variables = {"source": source}

    if task_type == "summary_ranking":
        if summaries is None:
            raise ValueError("summaries parameter required for ranking task")
        variables["summaries"] = "\n".join(
            f"Summary {i+1}: {s}" for i, s in enumerate(summaries)
        )
        variables["num_summaries"] = len(summaries)
    else:
        variables["summary"] = summary

    formatted_prompt = manager.format_prompt(task_type, prompt_type, **variables)

    if not formatted_prompt.validation_passed:
        raise ValueError(
            f"Prompt validation failed: {formatted_prompt.validation_errors}"
        )

    return formatted_prompt.prompt_text


def get_supported_tasks() -> List[str]:
    """Get list of supported task types."""
    return ["entailment_inference", "summary_ranking", "consistency_rating"]


def get_supported_prompt_types() -> List[str]:
    """Get list of supported prompt types."""
    return ["zero_shot", "chain_of_thought"]


def create_default_templates() -> Dict[str, str]:
    """
    Create default template content for setup.

    Returns:
        Dictionary mapping template names to content
    """
    manager = PromptManager()
    defaults = manager._get_default_templates()

    return {name: data["template"] for name, data in defaults.items()}
