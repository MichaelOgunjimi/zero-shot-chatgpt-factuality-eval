"""
Prompt Manager Tests
====================

Tests for prompt management functionality including prompt templates,
formatting, and task-specific prompt generation.

Author: Michael Ogunjimi
Institution: University of Manchester
"""

import pytest
import tempfile
from pathlib import Path
from unittest.mock import Mock, patch


class FormattedPrompt:
    """Formatted prompt for factuality tasks"""
    
    def __init__(self, prompt_text, task_type, prompt_type="zero_shot", metadata=None):
        self.prompt_text = prompt_text
        self.task_type = task_type
        self.prompt_type = prompt_type
        self.metadata = metadata or {}
        self.word_count = len(prompt_text.split())
        self.char_count = len(prompt_text)


class PromptTemplate:
    """Template for generating prompts"""
    
    def __init__(self, template_text, required_variables=None):
        self.template_text = template_text
        self.required_variables = required_variables or []
    
    def format(self, **kwargs):
        """Format template with provided variables"""
        # Check required variables
        missing = [var for var in self.required_variables if var not in kwargs]
        if missing:
            raise ValueError(f"Missing required variables: {missing}")
        
        try:
            return self.template_text.format(**kwargs)
        except KeyError as e:
            raise ValueError(f"Missing variable in template: {e}")


class PromptManager:
    """Manages prompts for factuality evaluation tasks"""
    
    def __init__(self, prompts_dir="prompts"):
        self.prompts_dir = Path(prompts_dir)
        self.templates = {}
        self.system_prompts = {}
        self._load_templates()
    
    def _load_templates(self):
        """Load prompt templates from files"""
        # Simulate loading templates
        self.templates = {
            "entailment_inference": {
                "zero_shot": PromptTemplate(
                    "Determine if the summary is factually consistent with the source.\n\n"
                    "Source: {source_text}\n\n"
                    "Summary: {summary_text}\n\n"
                    "Is the summary factually consistent? Answer ENTAILMENT or CONTRADICTION.",
                    required_variables=["source_text", "summary_text"]
                ),
                "chain_of_thought": PromptTemplate(
                    "Determine if the summary is factually consistent with the source. "
                    "Think step by step.\n\n"
                    "Source: {source_text}\n\n"
                    "Summary: {summary_text}\n\n"
                    "Let's analyze this step by step:\n"
                    "1. What are the key facts in the source?\n"
                    "2. What are the key facts in the summary?\n"
                    "3. Are the summary facts supported by the source?\n\n"
                    "Conclusion: ENTAILMENT or CONTRADICTION",
                    required_variables=["source_text", "summary_text"]
                )
            },
            "summary_ranking": {
                "zero_shot": PromptTemplate(
                    "Rank the following summaries by their factual consistency with the source. "
                    "Provide ranking as numbers separated by commas.\n\n"
                    "Source: {source_text}\n\n"
                    "Summaries:\n{summaries}\n\n"
                    "Ranking (most consistent to least consistent):",
                    required_variables=["source_text", "summaries"]
                ),
                "chain_of_thought": PromptTemplate(
                    "Rank the following summaries by their factual consistency with the source. "
                    "Think step by step.\n\n"
                    "Source: {source_text}\n\n"
                    "Summaries:\n{summaries}\n\n"
                    "Let's analyze each summary:\n"
                    "1. Check factual accuracy\n"
                    "2. Check completeness\n"
                    "3. Check consistency\n\n"
                    "Ranking (most consistent to least consistent):",
                    required_variables=["source_text", "summaries"]
                )
            },
            "consistency_rating": {
                "zero_shot": PromptTemplate(
                    "Rate the factual consistency of the summary with the source on a scale of 0-100.\n\n"
                    "Source: {source_text}\n\n"
                    "Summary: {summary_text}\n\n"
                    "Consistency rating (0-100):",
                    required_variables=["source_text", "summary_text"]
                ),
                "chain_of_thought": PromptTemplate(
                    "Rate the factual consistency of the summary with the source on a scale of 0-100. "
                    "Think step by step.\n\n"
                    "Source: {source_text}\n\n"
                    "Summary: {summary_text}\n\n"
                    "Let's evaluate:\n"
                    "1. Factual accuracy (0-40 points)\n"
                    "2. Completeness (0-30 points)\n"
                    "3. No contradictions (0-30 points)\n\n"
                    "Total consistency rating (0-100):",
                    required_variables=["source_text", "summary_text"]
                )
            }
        }
        
        # System prompts
        self.system_prompts = {
            "factuality_expert": (
                "You are an expert in factual consistency evaluation for text summarization. "
                "Your task is to carefully analyze the relationship between source documents "
                "and their summaries to determine factual accuracy."
            ),
            "default": "You are a helpful AI assistant."
        }
    
    def get_prompt(self, task_type, prompt_type="zero_shot", system_prompt="factuality_expert"):
        """Get a prompt template"""
        if task_type not in self.templates:
            raise ValueError(f"Unknown task type: {task_type}")
        
        if prompt_type not in self.templates[task_type]:
            raise ValueError(f"Unknown prompt type: {prompt_type} for task: {task_type}")
        
        template = self.templates[task_type][prompt_type]
        system = self.system_prompts.get(system_prompt, self.system_prompts["default"])
        
        return template, system
    
    def format_prompt(self, task_type, example_data, prompt_type="zero_shot", 
                     system_prompt="factuality_expert", **extra_kwargs):
        """Format a prompt for a specific example"""
        template, system = self.get_prompt(task_type, prompt_type, system_prompt)
        
        # Prepare formatting variables
        format_vars = {}
        
        if task_type == "entailment_inference":
            format_vars.update({
                "source_text": example_data.get("source_text", ""),
                "summary_text": example_data.get("summary_text", "")
            })
        elif task_type == "summary_ranking":
            summaries = example_data.get("summaries", [])
            formatted_summaries = "\n".join(
                f"{i+1}. {summary}" for i, summary in enumerate(summaries)
            )
            format_vars.update({
                "source_text": example_data.get("source_text", ""),
                "summaries": formatted_summaries
            })
        elif task_type == "consistency_rating":
            format_vars.update({
                "source_text": example_data.get("source_text", ""),
                "summary_text": example_data.get("summary_text", "")
            })
        
        # Add any extra variables
        format_vars.update(extra_kwargs)
        
        # Format the prompt
        prompt_text = template.format(**format_vars)
        
        return FormattedPrompt(
            prompt_text=prompt_text,
            task_type=task_type,
            prompt_type=prompt_type,
            metadata={
                "system_prompt": system,
                "template_variables": format_vars
            }
        )
    
    def get_available_tasks(self):
        """Get list of available task types"""
        return list(self.templates.keys())
    
    def get_available_prompt_types(self, task_type):
        """Get available prompt types for a task"""
        if task_type not in self.templates:
            return []
        return list(self.templates[task_type].keys())
    
    def get_system_prompts(self):
        """Get available system prompts"""
        return list(self.system_prompts.keys())
    
    def validate_prompt_config(self, task_type, prompt_type):
        """Validate prompt configuration"""
        errors = []
        
        if task_type not in self.templates:
            errors.append(f"Unknown task type: {task_type}")
        elif prompt_type not in self.templates[task_type]:
            errors.append(f"Unknown prompt type: {prompt_type} for task: {task_type}")
        
        return len(errors) == 0, errors


class TestPromptTemplate:
    """Test prompt template functionality"""
    
    def test_template_creation(self):
        """Test creating a prompt template"""
        template = PromptTemplate(
            "Hello {name}, you are {age} years old.",
            required_variables=["name", "age"]
        )
        
        assert template.template_text == "Hello {name}, you are {age} years old."
        assert template.required_variables == ["name", "age"]
    
    def test_template_formatting(self):
        """Test template formatting with variables"""
        template = PromptTemplate(
            "Source: {source}\nSummary: {summary}",
            required_variables=["source", "summary"]
        )
        
        formatted = template.format(
            source="Test source text",
            summary="Test summary text"
        )
        
        expected = "Source: Test source text\nSummary: Test summary text"
        assert formatted == expected
    
    def test_template_missing_required_variable(self):
        """Test template with missing required variable"""
        template = PromptTemplate(
            "Hello {name}",
            required_variables=["name", "age"]
        )
        
        with pytest.raises(ValueError, match="Missing required variables"):
            template.format(name="Alice")  # Missing 'age'
    
    def test_template_missing_variable_in_text(self):
        """Test template with variable in text but not provided"""
        template = PromptTemplate("Hello {name} and {friend}")
        
        with pytest.raises(ValueError, match="Missing variable in template"):
            template.format(name="Alice")  # Missing 'friend'
    
    def test_template_no_required_variables(self):
        """Test template with no required variables"""
        template = PromptTemplate("Static prompt text")
        
        formatted = template.format()
        assert formatted == "Static prompt text"


class TestFormattedPrompt:
    """Test formatted prompt functionality"""
    
    def test_formatted_prompt_creation(self):
        """Test creating a formatted prompt"""
        prompt = FormattedPrompt(
            prompt_text="Test prompt text",
            task_type="entailment_inference",
            prompt_type="zero_shot"
        )
        
        assert prompt.prompt_text == "Test prompt text"
        assert prompt.task_type == "entailment_inference"
        assert prompt.prompt_type == "zero_shot"
        assert prompt.word_count == 3
        assert prompt.char_count == 16
    
    def test_formatted_prompt_with_metadata(self):
        """Test formatted prompt with metadata"""
        metadata = {"system_prompt": "Test system", "variables": {"key": "value"}}
        
        prompt = FormattedPrompt(
            prompt_text="Test prompt",
            task_type="consistency_rating",
            metadata=metadata
        )
        
        assert prompt.metadata == metadata
        assert prompt.metadata["system_prompt"] == "Test system"


class TestPromptManager:
    """Test prompt manager functionality"""
    
    @pytest.fixture
    def prompt_manager(self):
        return PromptManager()
    
    def test_prompt_manager_initialization(self, prompt_manager):
        """Test prompt manager initialization"""
        assert isinstance(prompt_manager.templates, dict)
        assert isinstance(prompt_manager.system_prompts, dict)
        assert len(prompt_manager.templates) > 0
        assert len(prompt_manager.system_prompts) > 0
    
    def test_get_available_tasks(self, prompt_manager):
        """Test getting available task types"""
        tasks = prompt_manager.get_available_tasks()
        
        assert "entailment_inference" in tasks
        assert "summary_ranking" in tasks
        assert "consistency_rating" in tasks
        assert len(tasks) == 3
    
    def test_get_available_prompt_types(self, prompt_manager):
        """Test getting available prompt types"""
        prompt_types = prompt_manager.get_available_prompt_types("entailment_inference")
        
        assert "zero_shot" in prompt_types
        assert "chain_of_thought" in prompt_types
        assert len(prompt_types) == 2
    
    def test_get_available_prompt_types_invalid_task(self, prompt_manager):
        """Test getting prompt types for invalid task"""
        prompt_types = prompt_manager.get_available_prompt_types("invalid_task")
        assert prompt_types == []
    
    def test_get_system_prompts(self, prompt_manager):
        """Test getting available system prompts"""
        system_prompts = prompt_manager.get_system_prompts()
        
        assert "factuality_expert" in system_prompts
        assert "default" in system_prompts
    
    def test_get_prompt_valid(self, prompt_manager):
        """Test getting a valid prompt template"""
        template, system = prompt_manager.get_prompt("entailment_inference", "zero_shot")
        
        assert isinstance(template, PromptTemplate)
        assert isinstance(system, str)
        assert "source_text" in template.required_variables
        assert "summary_text" in template.required_variables
    
    def test_get_prompt_invalid_task(self, prompt_manager):
        """Test getting prompt with invalid task"""
        with pytest.raises(ValueError, match="Unknown task type"):
            prompt_manager.get_prompt("invalid_task")
    
    def test_get_prompt_invalid_prompt_type(self, prompt_manager):
        """Test getting prompt with invalid prompt type"""
        with pytest.raises(ValueError, match="Unknown prompt type"):
            prompt_manager.get_prompt("entailment_inference", "invalid_type")


class TestPromptFormatting:
    """Test prompt formatting functionality"""
    
    @pytest.fixture
    def prompt_manager(self):
        return PromptManager()
    
    def test_format_entailment_prompt(self, prompt_manager):
        """Test formatting entailment inference prompt"""
        example_data = {
            "source_text": "The cat sat on the mat.",
            "summary_text": "A cat was on the mat."
        }
        
        formatted = prompt_manager.format_prompt(
            "entailment_inference",
            example_data,
            prompt_type="zero_shot"
        )
        
        assert isinstance(formatted, FormattedPrompt)
        assert formatted.task_type == "entailment_inference"
        assert formatted.prompt_type == "zero_shot"
        assert "The cat sat on the mat." in formatted.prompt_text
        assert "A cat was on the mat." in formatted.prompt_text
        assert "ENTAILMENT or CONTRADICTION" in formatted.prompt_text
    
    def test_format_ranking_prompt(self, prompt_manager):
        """Test formatting summary ranking prompt"""
        example_data = {
            "source_text": "The cat sat on the mat.",
            "summaries": [
                "A cat was on the mat.",
                "The dog ran in the park.",
                "A feline sat on a rug."
            ]
        }
        
        formatted = prompt_manager.format_prompt(
            "summary_ranking",
            example_data,
            prompt_type="zero_shot"
        )
        
        assert isinstance(formatted, FormattedPrompt)
        assert formatted.task_type == "summary_ranking"
        assert "The cat sat on the mat." in formatted.prompt_text
        assert "1. A cat was on the mat." in formatted.prompt_text
        assert "2. The dog ran in the park." in formatted.prompt_text
        assert "3. A feline sat on a rug." in formatted.prompt_text
    
    def test_format_rating_prompt(self, prompt_manager):
        """Test formatting consistency rating prompt"""
        example_data = {
            "source_text": "The cat sat on the mat.",
            "summary_text": "A cat was on the mat."
        }
        
        formatted = prompt_manager.format_prompt(
            "consistency_rating",
            example_data,
            prompt_type="zero_shot"
        )
        
        assert isinstance(formatted, FormattedPrompt)
        assert formatted.task_type == "consistency_rating"
        assert "0-100" in formatted.prompt_text
        assert "The cat sat on the mat." in formatted.prompt_text
        assert "A cat was on the mat." in formatted.prompt_text
    
    def test_format_chain_of_thought_prompt(self, prompt_manager):
        """Test formatting chain of thought prompt"""
        example_data = {
            "source_text": "The cat sat on the mat.",
            "summary_text": "A cat was on the mat."
        }
        
        formatted = prompt_manager.format_prompt(
            "entailment_inference",
            example_data,
            prompt_type="chain_of_thought"
        )
        
        assert "Think step by step" in formatted.prompt_text
        assert "Let's analyze this step by step" in formatted.prompt_text
        assert "key facts" in formatted.prompt_text
    
    def test_format_prompt_with_extra_kwargs(self, prompt_manager):
        """Test formatting prompt with extra keyword arguments"""
        example_data = {
            "source_text": "Test source",
            "summary_text": "Test summary"
        }
        
        # This should work even with extra kwargs (they're ignored if not needed)
        formatted = prompt_manager.format_prompt(
            "entailment_inference",
            example_data,
            extra_var="extra_value"
        )
        
        assert isinstance(formatted, FormattedPrompt)
    
    def test_format_prompt_metadata(self, prompt_manager):
        """Test that formatted prompt includes correct metadata"""
        example_data = {
            "source_text": "Test source",
            "summary_text": "Test summary"
        }
        
        formatted = prompt_manager.format_prompt(
            "entailment_inference",
            example_data,
            prompt_type="zero_shot",
            system_prompt="factuality_expert"
        )
        
        assert "system_prompt" in formatted.metadata
        assert "template_variables" in formatted.metadata
        assert formatted.metadata["template_variables"]["source_text"] == "Test source"
        assert formatted.metadata["template_variables"]["summary_text"] == "Test summary"


class TestPromptValidation:
    """Test prompt validation functionality"""
    
    @pytest.fixture
    def prompt_manager(self):
        return PromptManager()
    
    def test_validate_valid_config(self, prompt_manager):
        """Test validating valid prompt configuration"""
        is_valid, errors = prompt_manager.validate_prompt_config(
            "entailment_inference", "zero_shot"
        )
        
        assert is_valid is True
        assert len(errors) == 0
    
    def test_validate_invalid_task(self, prompt_manager):
        """Test validating invalid task type"""
        is_valid, errors = prompt_manager.validate_prompt_config(
            "invalid_task", "zero_shot"
        )
        
        assert is_valid is False
        assert len(errors) == 1
        assert "Unknown task type" in errors[0]
    
    def test_validate_invalid_prompt_type(self, prompt_manager):
        """Test validating invalid prompt type"""
        is_valid, errors = prompt_manager.validate_prompt_config(
            "entailment_inference", "invalid_type"
        )
        
        assert is_valid is False
        assert len(errors) == 1
        assert "Unknown prompt type" in errors[0]
    
    def test_validate_both_invalid(self, prompt_manager):
        """Test validating with both task and prompt type invalid"""
        is_valid, errors = prompt_manager.validate_prompt_config(
            "invalid_task", "invalid_type"
        )
        
        assert is_valid is False
        assert len(errors) == 1  # Only task type error since that's checked first


class TestPromptIntegration:
    """Test prompt integration scenarios"""
    
    @pytest.fixture
    def prompt_manager(self):
        return PromptManager()
    
    def test_multiple_task_workflow(self, prompt_manager):
        """Test working with multiple task types"""
        example_data = {
            "source_text": "The research shows positive results.",
            "summary_text": "Research was positive.",
            "summaries": ["Research was positive.", "Study failed.", "Results unclear."]
        }
        
        # Test all task types
        for task_type in ["entailment_inference", "summary_ranking", "consistency_rating"]:
            formatted = prompt_manager.format_prompt(task_type, example_data)
            
            assert isinstance(formatted, FormattedPrompt)
            assert formatted.task_type == task_type
            assert len(formatted.prompt_text) > 0
    
    def test_all_prompt_types_workflow(self, prompt_manager):
        """Test all prompt types for a task"""
        example_data = {
            "source_text": "Test source text.",
            "summary_text": "Test summary."
        }
        
        for prompt_type in ["zero_shot", "chain_of_thought"]:
            formatted = prompt_manager.format_prompt(
                "entailment_inference",
                example_data,
                prompt_type=prompt_type
            )
            
            assert formatted.prompt_type == prompt_type
            if prompt_type == "chain_of_thought":
                assert "step by step" in formatted.prompt_text.lower()
    
    def test_system_prompt_variations(self, prompt_manager):
        """Test different system prompts"""
        example_data = {
            "source_text": "Test source",
            "summary_text": "Test summary"
        }
        
        for system_prompt in ["factuality_expert", "default"]:
            formatted = prompt_manager.format_prompt(
                "entailment_inference",
                example_data,
                system_prompt=system_prompt
            )
            
            assert system_prompt in formatted.metadata["system_prompt"] or \
                   formatted.metadata["system_prompt"] == prompt_manager.system_prompts[system_prompt]


class TestEdgeCases:
    """Test edge cases and error conditions"""
    
    @pytest.fixture
    def prompt_manager(self):
        return PromptManager()
    
    def test_empty_source_text(self, prompt_manager):
        """Test with empty source text"""
        example_data = {
            "source_text": "",
            "summary_text": "Test summary"
        }
        
        formatted = prompt_manager.format_prompt("entailment_inference", example_data)
        
        # Should work but with empty source
        assert formatted.prompt_text is not None
        assert "Summary: Test summary" in formatted.prompt_text
    
    def test_empty_summaries_list(self, prompt_manager):
        """Test with empty summaries list"""
        example_data = {
            "source_text": "Test source",
            "summaries": []
        }
        
        formatted = prompt_manager.format_prompt("summary_ranking", example_data)
        
        # Should work but with empty summaries section
        assert formatted.prompt_text is not None
        assert "Source: Test source" in formatted.prompt_text
    
    def test_missing_example_data_fields(self, prompt_manager):
        """Test with missing fields in example data"""
        example_data = {}  # Missing required fields
        
        formatted = prompt_manager.format_prompt("entailment_inference", example_data)
        
        # Should work with empty strings for missing fields
        assert formatted.prompt_text is not None
        assert "Source:" in formatted.prompt_text
        assert "Summary:" in formatted.prompt_text
    
    def test_very_long_text(self, prompt_manager):
        """Test with very long text"""
        long_text = "Very long text. " * 1000
        
        example_data = {
            "source_text": long_text,
            "summary_text": "Short summary"
        }
        
        formatted = prompt_manager.format_prompt("entailment_inference", example_data)
        
        assert formatted.prompt_text is not None
        assert formatted.char_count > 15000  # Should be quite long
        assert formatted.word_count > 2000
