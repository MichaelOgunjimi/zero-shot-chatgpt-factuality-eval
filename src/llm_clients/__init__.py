"""
LLM Clients Module for ChatGPT Factuality Evaluation
==================================================

Comprehensive LLM client implementations for OpenAI API integration
with rate limiting, cost tracking, batch processing, and academic research features.

This module provides robust API integration specifically designed for
factuality evaluation research with proper error handling, logging,
budget management, and cost-effective batch processing capabilities.

Author: Michael Ogunjimi
Institution: University of Manchester
Course: MSc AI
"""

from typing import Optional

from .openai_client import (
    # Main client class
    OpenAIClient,
    # Response handling
    ChatGPTResponse,
    APICallResult,
    # Rate limiting and cost tracking
    RateLimiter,
    CostCalculator,
    # Utility functions
    create_openai_client,
    validate_openai_config,
    estimate_token_cost,
    parse_factuality_response,
)

from .openai_client_batch import (
    # Batch client class
    OpenAIBatchClient,
    # Batch result handling
    BatchResult,
)

__all__ = [
    # Core clients
    "OpenAIClient",
    "OpenAIBatchClient",
    # Response types
    "ChatGPTResponse",
    "APICallResult",
    "BatchResult",
    # Supporting classes
    "RateLimiter",
    "CostCalculator",
    # Utility functions
    "create_openai_client",
    "create_batch_client",
    "setup_chatgpt_client",
    "setup_batch_client",
    "validate_openai_config",
    "estimate_token_cost",
    "parse_factuality_response",
]

# Module metadata
__version__ = "1.0.0"
__author__ = "Michael Ogunjimi"
__email__ = "michael.ogunjimi@postgrad.manchester.ac.uk"
__institution__ = "University of Manchester"
__course__ = "MSc AI"

# Supported models for factuality evaluation
SUPPORTED_MODELS = [
    "gpt-4.1-mini",
    "gpt-4o-mini", 
    "o1-mini",
    "gpt-4o",
]

# Model pricing (tokens per dollar - approximate)
MODEL_PRICING = {
    "gpt-4.1-mini": {"input_cost_per_1k": 0.0004, "output_cost_per_1k": 0.0016},
    "gpt-4o-mini": {"input_cost_per_1k": 0.00015, "output_cost_per_1k": 0.0006},
    "o1-mini": {"input_cost_per_1k": 0.003, "output_cost_per_1k": 0.012},
    "gpt-4o": {"input_cost_per_1k": 0.0025, "output_cost_per_1k": 0.01},
}


def get_client_info() -> dict:
    """
    Get comprehensive information about LLM client capabilities.

    Returns:
        Dictionary containing client information and supported features
    """
    return {
        "supported_models": SUPPORTED_MODELS,
        "model_pricing": MODEL_PRICING,
        "version": __version__,
        "features": [
            "Rate limiting",
            "Cost tracking", 
            "Error handling",
            "Response parsing",
            "Academic logging",
            "Budget management",
            "Batch processing",
            "Cost optimization",
        ],
        "description": "OpenAI API clients for ChatGPT factuality evaluation research with batch processing support",
    }


def create_batch_client(
    model: str = "gpt-4.1-mini", 
    tier: str = "tier2",
    experiment_name: Optional[str] = None
) -> 'OpenAIBatchClient':
    """
    Create an OpenAI batch client with proper configuration.
    
    Args:
        model: Model name to use (default: gpt-4.1-mini)
        tier: API tier configuration (default: tier2) 
        experiment_name: Optional experiment name for tracking
        
    Returns:
        Configured OpenAIBatchClient instance
        
    Example:
        >>> client = create_batch_client("gpt-4.1-mini", "tier2", "my_experiment")
        >>> # Use client for batch processing
    """
    from ..utils.config import get_config
    
    config = get_config(model=model, tier=tier)
    return OpenAIBatchClient(config=config, experiment_name=experiment_name)


def validate_model_name(model_name: str) -> bool:
    """
    Validate if model name is supported.

    Args:
        model_name: Name of the model to validate

    Returns:
        True if model is supported
    """
    return model_name in SUPPORTED_MODELS


def get_model_cost_info(model_name: str) -> dict:
    """
    Get cost information for a specific model.

    Args:
        model_name: Name of the model

    Returns:
        Dictionary with cost information

    Raises:
        ValueError: If model is not supported
    """
    if model_name not in MODEL_PRICING:
        raise ValueError(f"Unsupported model: {model_name}")

    return MODEL_PRICING[model_name]


def estimate_experiment_cost(
    num_examples: int,
    avg_prompt_tokens: int,
    avg_response_tokens: int,
    model_name: Optional[str] = None,
) -> dict:
    """
    Estimate cost for an experiment.

    Args:
        num_examples: Number of examples to process
        avg_prompt_tokens: Average tokens per prompt
        avg_response_tokens: Average tokens per response
        model_name: Model to use for estimation (if None, uses first available model)

    Returns:
        Dictionary with cost estimation
    """
    if model_name is None:
        model_name = SUPPORTED_MODELS[0]  # Use first available model
    
    if model_name not in MODEL_PRICING:
        raise ValueError(f"Unsupported model: {model_name}")

    pricing = MODEL_PRICING[model_name]

    total_input_tokens = num_examples * avg_prompt_tokens
    total_output_tokens = num_examples * avg_response_tokens

    input_cost = (total_input_tokens / 1000) * pricing["input_cost_per_1k"]
    output_cost = (total_output_tokens / 1000) * pricing["output_cost_per_1k"]
    total_cost = input_cost + output_cost

    return {
        "model": model_name,
        "num_examples": num_examples,
        "total_input_tokens": total_input_tokens,
        "total_output_tokens": total_output_tokens,
        "input_cost": round(input_cost, 4),
        "output_cost": round(output_cost, 4),
        "total_cost": round(total_cost, 4),
        "cost_per_example": round(total_cost / num_examples, 4),
    }


# Quick setup function for common use cases
def setup_chatgpt_client(
    model_name: Optional[str] = None,
    max_requests_per_minute: int = 10,
    daily_budget: float = 50.0,
    config: dict = None,
) -> OpenAIClient:
    """
    Quick setup function for ChatGPT client with reasonable defaults.

    Args:
        model_name: Model to use (if None, uses first available model)
        max_requests_per_minute: Rate limit for API calls
        daily_budget: Daily spending limit
        config: Optional configuration dictionary

    Returns:
        Configured OpenAI client
    """
    if model_name is None:
        model_name = SUPPORTED_MODELS[0]  # Use first available model
    
    if config is None:
        config = {
            "openai": {
                "models": {"primary": model_name},
                "api": {"timeout": 30, "max_retries": 3},
                "rate_limits": {
                    "requests_per_minute": max_requests_per_minute,
                    "tokens_per_minute": 150000,
                },
                "cost_control": {
                    "daily_budget": daily_budget,
                    "total_budget": daily_budget * 7,
                    "warning_threshold": 0.8,
                },
            }
        }

    return create_openai_client(config)


def setup_batch_client(
    model_name: Optional[str] = None,
    experiment_name: Optional[str] = None,
    tier: str = "tier2"
) -> OpenAIBatchClient:
    """
    Quick setup function for batch processing client with reasonable defaults.
    
    Args:
        model_name: Model to use (default: gpt-4.1-mini)
        experiment_name: Name for the experiment (auto-generated if None)
        tier: API tier configuration (default: tier2)
        
    Returns:
        Configured OpenAI batch client
        
    Example:
        >>> batch_client = setup_batch_client("gpt-4.1-mini", "my_experiment")
        >>> # Use for cost-effective batch processing
    """
    if model_name is None:
        model_name = "gpt-4.1-mini"  # Default to most cost-effective model
        
    return create_batch_client(
        model=model_name,
        tier=tier,
        experiment_name=experiment_name
    )
