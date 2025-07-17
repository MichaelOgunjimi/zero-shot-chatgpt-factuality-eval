"""
Prompts Module for ChatGPT Factuality Evaluation
==============================================

Comprehensive prompt management system for zero-shot and chain-of-thought
prompts across all three factuality evaluation tasks.

Author: Michael Ogunjimi
Institution: University of Manchester
Course: MSc AI
"""

# Core prompt management
from .prompt_manager import (
    PromptTemplate,
    FormattedPrompt,
    BasePromptFormatter,
    StandardPromptFormatter,
    ChainOfThoughtFormatter,
    PromptManager,
    create_prompt_manager,
    validate_prompt_config,
)

# Version information
__version__ = "1.0.0"
__author__ = "Michael Ogunjimi"

# Public API
__all__ = [
    # Data structures
    "PromptTemplate",
    "FormattedPrompt",
    # Formatters
    "BasePromptFormatter",
    "StandardPromptFormatter",
    "ChainOfThoughtFormatter",
    # Main manager
    "PromptManager",
    # Utility functions
    "create_prompt_manager",
    "validate_prompt_config",
    "quick_format_prompt",
    "get_supported_prompt_types",
    "create_default_templates",
]


def get_supported_prompt_types() -> list:
    """
    Get list of supported prompt types.

    Returns:
        List of prompt type names
    """
    return ["zero_shot", "chain_of_thought"]


def get_supported_tasks() -> list:
    """
    Get list of supported task types for prompts.

    Returns:
        List of task names
    """
    return ["entailment_inference", "summary_ranking", "consistency_rating"]


def quick_format_prompt(task: str, prompt_type: str, config: dict, **variables) -> str:
    """
    Quick function to format a prompt.

    Args:
        task: Task name
        prompt_type: Prompt type
        config: Configuration dictionary
        **variables: Variables for template

    Returns:
        Formatted prompt string
    """
    manager = create_prompt_manager(config)
    formatted_prompt = manager.format_prompt(task, prompt_type, **variables)
    return formatted_prompt.content


def create_default_templates() -> dict:
    """
    Create default prompt templates for all tasks.

    Returns:
        Dictionary mapping template names to template content
    """
    templates = {}

    # Entailment Inference Templates
    templates[
        "entailment_inference_zero_shot"
    ] = """Evaluate whether the summary is factually consistent with the source document.

Source Document:
{source}

Summary:
{summary}

Task: Determine if the summary is factually consistent with the source document.

Response Format:
- ENTAILMENT: The summary is factually consistent with the source
- CONTRADICTION: The summary contains factual errors or contradictions

Answer: ENTAILMENT or CONTRADICTION

Answer: """

    templates[
        "entailment_inference_chain_of_thought"
    ] = """Evaluate whether the summary is factually consistent with the source document using step-by-step analysis.

Source Document:
{source}

Summary:
{summary}

Please analyze this step-by-step:

1. Key Facts in Source: [Extract the main factual claims from the source document]
2. Claims in Summary: [Identify all factual claims made in the summary]
3. Fact Verification: [Check each summary claim against the source facts]
4. Final Judgment: [ENTAILMENT or CONTRADICTION based on your analysis]

Analysis:
1. Key Facts in Source: """

    # Summary Ranking Templates
    templates[
        "summary_ranking_zero_shot"
    ] = """Rank the following summaries by their factual consistency with the source document.

Source Document:
{source}

Summaries to Rank:
{numbered_summaries}

Task: Rank these summaries from 1 (most factually consistent) to {num_summaries} (least factually consistent).

Consider:
- Factual accuracy of claims
- Presence of contradictions or errors
- Overall consistency with source content

Response Format: Provide rankings as numbers separated by commas.
Example: If summary 3 is best, summary 1 is second, summary 2 is worst, respond: "3, 1, 2"

RANKING: [Your ranking from best to worst]

RANKING: """

    templates[
        "summary_ranking_chain_of_thought"
    ] = """Rank the following summaries by their factual consistency with the source document using step-by-step analysis.

Source Document:
{source}

Summaries to Rank:
{numbered_summaries}

Please analyze this step-by-step:

1. Key Facts in Source: [Extract the main factual claims from source]
2. Evaluate Each Summary: [Assess factual accuracy of each summary]
3. Final Ranking: [Rank from 1 (best) to {num_summaries} (worst) with rationale]

Analysis:
1. Key Facts in Source: """

    # Consistency Rating Templates
    templates[
        "consistency_rating_zero_shot"
    ] = """Rate the factual consistency between the source document and summary on a scale of 0.0 to 1.0.

Source Document:
{source}

Summary:
{summary}

Rating Scale:
1.0 = Perfect factual consistency
0.8 = Minor factual issues
0.6 = Some factual errors
0.4 = Multiple factual errors
0.2 = Major factual errors
0.0 = Completely inconsistent

Task: Evaluate how factually consistent the summary is with the source document. Consider:
- Accuracy of factual claims
- Presence of contradictions or errors
- Completeness of important facts
- Overall reliability of information

Provide a numerical rating between 0.0 and 1.0.

RATING: [Your numerical rating]

RATING: """

    templates[
        "consistency_rating_chain_of_thought"
    ] = """Rate the factual consistency between the source document and summary using step-by-step analysis.

Source Document:
{source}

Summary:
{summary}

Rating Scale:
1.0 = Perfect factual consistency
0.8 = Minor factual issues
0.6 = Some factual errors
0.4 = Multiple factual errors
0.2 = Major factual errors
0.0 = Completely inconsistent

Please analyze this step-by-step:

1. Key Facts in Source: [Extract main factual claims from source]
2. Claims in Summary: [Identify factual claims in summary]
3. Fact Verification: [Check each summary claim against source]
4. Rating Calculation: [Assign 0.0-1.0 rating with rationale]

Analysis:
1. Key Facts in Source: """

    return templates


def validate_template_variables(template: str, variables: dict) -> tuple:
    """
    Validate that template has all required variables.

    Args:
        template: Template string
        variables: Dictionary of variables

    Returns:
        Tuple of (is_valid, missing_variables)
    """
    import re

    # Extract required variables from template
    required_vars = set(re.findall(r"\{(\w+)\}", template))
    provided_vars = set(variables.keys())

    missing_vars = required_vars - provided_vars

    return len(missing_vars) == 0, list(missing_vars)


def estimate_prompt_tokens(prompt_content: str) -> int:
    """
    Estimate token count for prompt content.

    Args:
        prompt_content: Formatted prompt content

    Returns:
        Estimated token count
    """
    # Rough approximation: 1.3 tokens per word
    word_count = len(prompt_content.split())
    return int(word_count * 1.3)


def optimize_prompt_length(prompt_content: str, max_tokens: int = 3000) -> str:
    """
    Optimize prompt length to stay within token limits.

    Args:
        prompt_content: Original prompt content
        max_tokens: Maximum allowed tokens

    Returns:
        Optimized prompt content
    """
    estimated_tokens = estimate_prompt_tokens(prompt_content)

    if estimated_tokens <= max_tokens:
        return prompt_content

    # Simple optimization: remove extra whitespace and truncate if needed
    optimized = prompt_content.strip()

    # Remove multiple newlines
    import re

    optimized = re.sub(r"\n\s*\n\s*\n", "\n\n", optimized)

    # Remove extra spaces
    optimized = re.sub(r" +", " ", optimized)

    # If still too long, truncate source/summary sections
    if estimate_prompt_tokens(optimized) > max_tokens:
        lines = optimized.split("\n")

        # Find and truncate long sections
        for i, line in enumerate(lines):
            if len(line) > 500:  # Long line
                lines[i] = line[:500] + "..."

        optimized = "\n".join(lines)

    return optimized


def create_prompt_variations(base_template: str, variations: list = None) -> dict:
    """
    Create variations of a base prompt template.

    Args:
        base_template: Base template string
        variations: List of variation types to create

    Returns:
        Dictionary mapping variation names to templates
    """
    if variations is None:
        variations = ["concise", "detailed", "formal", "conversational"]

    templates = {"base": base_template}

    for variation in variations:
        if variation == "concise":
            # Remove explanatory text
            concise = base_template.replace(
                "Please analyze this step-by-step:", "Analyze:"
            )
            concise = concise.replace("Consider:", "")
            templates["concise"] = concise

        elif variation == "detailed":
            # Add more specific instructions
            detailed = base_template.replace("Task:", "Task: Carefully and thoroughly")
            templates["detailed"] = detailed

        elif variation == "formal":
            # Make language more formal
            formal = base_template.replace("you", "one")
            formal = formal.replace("Your", "The")
            templates["formal"] = formal

        elif variation == "conversational":
            # Make language more conversational
            conversational = base_template.replace("Evaluate", "Please evaluate")
            conversational = conversational.replace("Determine", "Can you determine")
            templates["conversational"] = conversational

    return templates


# Module-level convenience functions
def setup_default_prompts(config: dict, templates_dir: str = "./prompts") -> bool:
    """
    Set up default prompt templates in the specified directory.

    Args:
        config: Configuration dictionary
        templates_dir: Directory to create templates in

    Returns:
        True if setup successful
    """
    from pathlib import Path

    templates_path = Path(templates_dir)

    # Create task directories
    tasks = get_supported_tasks()
    for task in tasks:
        task_dir = templates_path / task
        task_dir.mkdir(parents=True, exist_ok=True)

    # Get default templates
    default_templates = create_default_templates()

    # Write template files
    for template_name, content in default_templates.items():
        parts = template_name.split("_")
        task = "_".join(parts[:-1])  # Everything except last part
        prompt_type = parts[-1]  # Last part

        file_path = templates_path / task / f"{prompt_type}.txt"

        with open(file_path, "w", encoding="utf-8") as f:
            f.write(content)

    return True


# Module initialization
import logging

logger = logging.getLogger(__name__)

# Validate module requirements
try:
    import re

    logger.debug("Regular expressions module available")
except ImportError:
    logger.error("Regular expressions module not available")

logger.info("Prompts module initialized")
