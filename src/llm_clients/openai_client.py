"""
Enhanced Multi-Provider LLM Client for Factuality Evaluation
==========================================================

Unified client supporting both OpenAI API and Ollama models with consistent
interface for factuality evaluation experiments. Supports GPT-4.1-mini,
Qwen2.5:7b, and Llama3.1:8b models.

Features:
- Unified API for OpenAI and Ollama models
- Automatic provider detection based on model configuration
- Cost tracking for OpenAI models
- Rate limiting and error handling
- Response validation and parsing

Author: Michael Ogunjimi
Institution: University of Manchester
Course: MSc AI
Date: August 4, 2025
"""

import os
import re
import time
from collections import deque
from dataclasses import dataclass, asdict
from datetime import datetime, timedelta
from typing import Dict, List, Any, Optional

import openai
import tiktoken
from tenacity import (
    retry,
    stop_after_attempt,
    wait_exponential,
    retry_if_exception_type,
)

from ..prompts.prompt_manager import FormattedPrompt
from ..utils.config import get_config
from ..utils.logging import get_logger, CostTracker

logger = get_logger(__name__)


@dataclass
class LLMResponse:
    """
    Unified response structure for all LLM providers.

    Contains the response content along with metadata needed
    for analysis, cost tracking, and reproducibility across
    both OpenAI and Ollama models.
    """

    content: str
    model: str
    provider: str  # 'openai' or 'ollama'
    prompt_tokens: int
    completion_tokens: int
    total_tokens: int
    cost: float
    response_time: float
    timestamp: str
    finish_reason: str
    temperature: float
    max_tokens: Optional[int]
    request_id: Optional[str] = None

    def to_dict(self) -> Dict[str, Any]:
        """Convert to dictionary for logging and analysis."""
        return asdict(self)

    def get_token_efficiency(self) -> float:
        """Calculate tokens per second for performance analysis."""
        if self.response_time > 0:
            return self.total_tokens / self.response_time
        return 0.0


@dataclass
class APICallResult:
    """
    Result from API call with parsed factuality evaluation response.

    Provides structured access to the response content along with
    metadata and parsed task-specific results. Works with both
    OpenAI and Ollama providers.
    """

    raw_response: LLMResponse  # Updated to use LLMResponse
    parsed_content: Dict[str, Any]
    task_type: str
    parsing_successful: bool
    parsing_errors: Optional[List[str]] = None
    confidence_score: Optional[float] = None

    def to_dict(self) -> Dict[str, Any]:
        """Convert to dictionary for logging and analysis."""
        return {
            "raw_response": self.raw_response.to_dict(),
            "parsed_content": self.parsed_content,
            "task_type": self.task_type,
            "parsing_successful": self.parsing_successful,
            "parsing_errors": self.parsing_errors,
            "confidence_score": self.confidence_score,
        }


class RateLimiter:
    """
    Token bucket rate limiter for OpenAI API calls.

    Implements both request-per-minute and token-per-minute limiting
    to comply with OpenAI rate limits and ensure stable performance.
    """

    def __init__(
        self,
        requests_per_minute: int = 20,
        tokens_per_minute: int = 150000,
        requests_per_day: int = 1000,
    ):
        """
        Initialize rate limiter.

        Args:
            requests_per_minute: Maximum requests per minute
            tokens_per_minute: Maximum tokens per minute
            requests_per_day: Maximum requests per day
        """
        self.requests_per_minute = requests_per_minute
        self.tokens_per_minute = tokens_per_minute
        self.requests_per_day = requests_per_day

        # Request tracking
        self.request_times = deque()
        self.daily_requests = 0
        self.last_reset_date = datetime.now().date()

        # Token tracking
        self.token_usage = deque()

        self.logger = get_logger(f"{self.__class__.__name__}")

    def wait_if_needed(self, estimated_tokens: int = 0) -> float:
        """
        Wait if necessary to respect rate limits.

        Args:
            estimated_tokens: Estimated tokens for the request

        Returns:
            Wait time in seconds
        """
        current_time = time.time()
        wait_time = 0.0

        # Reset daily counter if new day
        current_date = datetime.now().date()
        if current_date > self.last_reset_date:
            self.daily_requests = 0
            self.last_reset_date = current_date
            self.logger.info("Daily request counter reset")

        # Check daily limit
        if self.daily_requests >= self.requests_per_day:
            # Wait until next day
            tomorrow = datetime.combine(
                current_date + timedelta(days=1), datetime.min.time()
            )
            wait_until_tomorrow = (tomorrow - datetime.now()).total_seconds()
            self.logger.warning(
                f"Daily request limit reached. Waiting {wait_until_tomorrow:.0f} seconds until tomorrow."
            )
            time.sleep(wait_until_tomorrow)
            self.daily_requests = 0
            return wait_until_tomorrow

        # Remove old request times (older than 1 minute)
        cutoff_time = current_time - 60
        while self.request_times and self.request_times[0] < cutoff_time:
            self.request_times.popleft()

        # Remove old token usage (older than 1 minute)
        while self.token_usage and self.token_usage[0][0] < cutoff_time:
            self.token_usage.popleft()

        # Check request rate limit
        if len(self.request_times) >= self.requests_per_minute:
            wait_time = max(wait_time, 60 - (current_time - self.request_times[0]))

        # Check token rate limit
        current_token_usage = sum(tokens for _, tokens in self.token_usage)
        if current_token_usage + estimated_tokens > self.tokens_per_minute:
            # Calculate wait time based on oldest token usage
            if self.token_usage:
                wait_time = max(wait_time, 60 - (current_time - self.token_usage[0][0]))

        if wait_time > 0:
            # Only log if wait time is significant (> 1 second)
            if wait_time > 1.0:
                self.logger.info(f"Rate limit reached. Waiting {wait_time:.1f}s.")
            time.sleep(wait_time)

        # Record this request
        self.request_times.append(current_time + wait_time)
        self.token_usage.append((current_time + wait_time, estimated_tokens))
        self.daily_requests += 1

        return wait_time

    def get_current_usage(self) -> Dict[str, Any]:
        """Get current rate limit usage statistics."""
        current_time = time.time()
        cutoff_time = current_time - 60

        # Count recent requests and tokens
        recent_requests = sum(1 for t in self.request_times if t > cutoff_time)
        recent_tokens = sum(
            tokens for timestamp, tokens in self.token_usage if timestamp > cutoff_time
        )

        return {
            "requests_last_minute": recent_requests,
            "requests_per_minute_limit": self.requests_per_minute,
            "tokens_last_minute": recent_tokens,
            "tokens_per_minute_limit": self.tokens_per_minute,
            "daily_requests": self.daily_requests,
            "daily_limit": self.requests_per_day,
            "requests_remaining_today": max(
                0, self.requests_per_day - self.daily_requests
            ),
        }


class CostCalculator:
    """
    Accurate cost calculation for OpenAI API usage.

    Calculates costs based on actual token usage and current
    OpenAI pricing with support for different models.
    """

    PRICING = {
        "gpt-4.1-mini": {
            "input": 0.0004,    # $0.00040 per 1K tokens
            "output": 0.0016,   # $0.00160 per 1K tokens
        },
        "o1-mini": {
            "input": 0.003,     # $0.00300 per 1K tokens
            "output": 0.012,    # $0.01200 per 1K tokens
        },
        "gpt-4o": {
            "input": 0.0025,    # $0.00250 per 1K tokens
            "output": 0.01,     # $0.01000 per 1K tokens
        },
    }

    @classmethod
    def calculate_cost(
        cls, model: str, prompt_tokens: int, completion_tokens: int
    ) -> float:
        """
        Calculate cost for API call.

        Args:
            model: Model name used
            prompt_tokens: Number of input tokens
            completion_tokens: Number of output tokens

        Returns:
            Cost in USD
        """
        if model not in cls.PRICING:
            fallback_model = next(iter(cls.PRICING.keys()))
            logger.warning(f"Unknown model for pricing: {model}. Using {fallback_model} pricing.")
            model = fallback_model

        pricing = cls.PRICING[model]

        input_cost = (prompt_tokens / 1000) * pricing["input"]
        output_cost = (completion_tokens / 1000) * pricing["output"]

        return round(input_cost + output_cost, 6)

    @classmethod
    def estimate_cost(
        cls, model: str, estimated_prompt_tokens: int, estimated_completion_tokens: int
    ) -> float:
        """Estimate cost before making API call."""
        return cls.calculate_cost(
            model, estimated_prompt_tokens, estimated_completion_tokens
        )


class OpenAIClient:
    """
    Enhanced multi-provider LLM client for factuality evaluation.

    Supports both OpenAI API and Ollama models with unified interface.
    Provides robust API integration with rate limiting, cost tracking,
    response parsing, and academic research features.
    """

    def __init__(self, config: Optional[Dict[str, Any]] = None):
        """
        Initialize multi-provider LLM client.

        Args:
            config: Configuration dictionary
        """
        self.config = config or get_config()
        self.logger = get_logger(self.__class__.__name__)

        # Setup API configurations
        self._setup_api_clients()

        # Load model configurations from experiments2 if available, otherwise use legacy config
        self._load_model_configurations()

        # Initialize rate limiter and cost tracker
        self._initialize_rate_limiting()
        self._initialize_cost_tracking()

        # Token encoder for accurate token counting
        self._setup_token_encoder()

        # Statistics tracking
        self.total_requests = 0
        self.successful_requests = 0
        self.failed_requests = 0
        self.total_tokens_used = 0

        self.logger.info(f"Multi-provider LLM client initialized with primary model: {self.primary_model}")

    def _load_model_configurations(self):
        """Load model configurations from config (experiments2 or legacy format)."""
        experiments2_config = self.config.get('experiments2', {})
        if experiments2_config and 'models' in experiments2_config:
            self._load_experiments2_models()
        else:
            self._load_legacy_models()

    def _load_experiments2_models(self):
        """Load models from experiments2 configuration."""
        models_config = self.config.get('experiments2', {}).get('models', {})
        
        # Find enabled models
        self.available_models = {}
        enabled_models = []
        
        for model_key, model_data in models_config.items():
            if model_data.get('enabled', False):
                # Expand environment variables in configuration
                api_config = model_data.get('api_config', {})
                base_url = os.path.expandvars(api_config.get('base_url', ''))
                api_key = os.path.expandvars(api_config.get('api_key', ''))
                
                model_config = {
                    'name': model_data.get('name', model_key),
                    'provider': model_data.get('provider'),
                    'model_id': model_data.get('model_id'),
                    'base_url': base_url,
                    'api_key': api_key,
                    'timeout': api_config.get('timeout', 120),
                    'max_retries': api_config.get('max_retries', 3),
                    'retry_delay': api_config.get('retry_delay', 2.0),
                    'generation_params': model_data.get('generation_params', {}),
                    'cost_tracking': model_data.get('cost_tracking', False)
                }
                
                self.available_models[model_key] = model_config
                enabled_models.append(model_key)
        
        if enabled_models:
            # Use first enabled model as primary
            self.primary_model = enabled_models[0]
            self.primary_model_config = self.available_models[self.primary_model]
            self.fallback_models = enabled_models[1:]
            
            # Set generation parameters from primary model
            gen_params = self.primary_model_config.get('generation_params', {})
            self.default_temperature = gen_params.get('temperature', 0.0)
            self.default_max_tokens = gen_params.get('max_tokens', 2048)
            self.default_top_p = gen_params.get('top_p', 1.0)
            
            self.logger.info(f"Loaded {len(enabled_models)} models from experiments2 config")
        else:
            raise ValueError("No enabled models found in experiments2 configuration")

    def _load_legacy_models(self):
        """Load models from legacy OpenAI configuration."""
        openai_config = self.config.get("openai", {})
        models_config = openai_config.get("models", {})
        
        self.primary_model = models_config.get("primary")
        self.fallback_models = models_config.get("fallbacks", [])
        
        if not self.primary_model:
            raise ValueError("No primary model specified in configuration. Use get_config(model='model-name') to load model-specific config.")

        self.available_models = {
            self.primary_model: {
                'name': self.primary_model,
                'provider': 'openai',
                'model_id': self.primary_model,
                'base_url': 'https://api.openai.com/v1',
                'api_key': os.getenv('OPENAI_API_KEY'),
                'timeout': 120,
                'max_retries': 3,
                'retry_delay': 2.0,
                'generation_params': openai_config.get('generation', {}),
                'cost_tracking': True
            }
        }
        self.primary_model_config = self.available_models[self.primary_model]

        # Generation parameters (model-agnostic)
        generation_config = openai_config.get("generation", {})
        self.default_temperature = generation_config.get("temperature", 0.0)
        self.default_max_tokens = generation_config.get("max_tokens", 2048)
        self.default_top_p = generation_config.get("top_p", 1.0)

    def _setup_api_clients(self):
        """Setup API clients for different providers."""
        self.openai_clients = {}  # Will store provider-specific clients

    def _get_or_create_client(self, model_key: str):
        """Get or create API client for specific model."""
        if model_key not in self.available_models:
            raise ValueError(f"Model {model_key} not available. Available models: {list(self.available_models.keys())}")
        
        if model_key not in self.openai_clients:
            model_config = self.available_models[model_key]
            
            # Create OpenAI client (works for both OpenAI and Ollama via OpenAI-compatible API)
            try:
                if model_config['provider'] == 'ollama':
                    # Validate Ollama endpoint
                    if not model_config['base_url'].startswith('http://') and not model_config['base_url'].startswith('https://'):
                        raise ValueError(f"Ollama base_url must start with http:// or https://, got: {model_config['base_url']}")
                    if 'ollama' not in model_config['base_url']:
                        self.logger.warning(f"Ollama provider base_url does not contain 'ollama': {model_config['base_url']}")
                self.openai_clients[model_key] = openai.AsyncOpenAI(
                    base_url=model_config['base_url'],
                    api_key=model_config['api_key'],
                    timeout=model_config['timeout']
                )
                self.logger.info(f"Created client for {model_config['name']} ({model_config['provider']})")
            except Exception as e:
                self.logger.error(f"Failed to create API client for {model_config['name']} ({model_config['provider']}): {e}")
                raise
        
        return self.openai_clients[model_key]

    def _initialize_rate_limiting(self):
        """Initialize rate limiting based on configuration."""
        if hasattr(self, 'primary_model_config'):
            # Use model-specific rate limits from experiments2 config
            api_config = self.primary_model_config.get('api_config', {})
            rate_limits = api_config.get('rate_limits', {})
            
            # If no rate limits in api_config, check global config
            if not rate_limits:
                global_config = self.config.get("experiments2", {}).get("models", {})
                model_id = self.primary_model_config.get('model_id', self.primary_model)
                model_config = global_config.get(model_id, {})
                rate_limits = model_config.get('api_config', {}).get('rate_limits', {})
            
            # Initialize with actual rate limits or reasonable defaults
            self.rate_limiter = RateLimiter(
                requests_per_minute=rate_limits.get("requests_per_minute", 4500),  # GPT-4.1-mini Tier 2
                tokens_per_minute=rate_limits.get("tokens_per_minute", 1800000),   # GPT-4.1-mini Tier 2
                requests_per_day=rate_limits.get("requests_per_day", 648000),      # Conservative daily estimate
            )
        else:
            # Legacy configuration
            openai_config = self.config.get("openai", {})
            rate_limits = openai_config.get("rate_limits", {})
            self.rate_limiter = RateLimiter(
                requests_per_minute=rate_limits.get("requests_per_minute", 4500),
                tokens_per_minute=rate_limits.get("tokens_per_minute", 1800000), 
                requests_per_day=rate_limits.get("requests_per_day", 648000),
            )

    def _initialize_cost_tracking(self):
        """Initialize cost tracking."""
        if hasattr(self, 'primary_model_config'):
            self.cost_tracker = CostTracker(
                daily_budget=50.0,
                total_budget=200.0,
                warning_threshold=0.8,
            )
        else:
            # Legacy configuration
            openai_config = self.config.get("openai", {})
            cost_config = openai_config.get("cost_control", {})
            self.cost_tracker = CostTracker(
                daily_budget=cost_config.get("daily_budget", 50.0),
                total_budget=cost_config.get("total_budget", 200.0),
                warning_threshold=cost_config.get("warning_threshold", 0.8),
            )

    def _setup_token_encoder(self):
        """Setup token encoder for token counting."""
        try:
            # Try to use encoder for primary model
            if hasattr(self, 'primary_model_config'):
                model_id = self.primary_model_config['model_id']
                if 'gpt' in model_id.lower():
                    self.token_encoder = tiktoken.encoding_for_model(model_id)
                else:
                    # For non-OpenAI models, use default encoding
                    self.token_encoder = tiktoken.get_encoding("cl100k_base")
            else:
                self.token_encoder = tiktoken.encoding_for_model(self.primary_model)
        except KeyError:
            self.token_encoder = tiktoken.get_encoding("cl100k_base")
            self.logger.warning(f"Using default encoding for token counting")

    def _setup_openai_api(self) -> None:
        """Setup OpenAI API configuration."""
        api_key = os.getenv("OPENAI_API_KEY")
        if not api_key:
            raise ValueError("OPENAI_API_KEY environment variable not set")

        # Initialize OpenAI client (v1.0+ style) - AsyncOpenAI for async operations
        org_id = os.getenv("OPENAI_ORG_ID")
        self.openai_client = openai.AsyncOpenAI(
            api_key=api_key,
            organization=org_id if org_id else None
        )

        self.logger.info("OpenAI API configured successfully")

    def count_tokens(self, text: str) -> int:
        """
        Count tokens in text using the model's tokenizer.

        Args:
            text: Text to count tokens for

        Returns:
            Number of tokens
        """
        try:
            return len(self.token_encoder.encode(text))
        except Exception as e:
            self.logger.warning(f"Token counting failed: {e}. Using estimation.")
            # Fallback estimation: ~4 characters per token
            return len(text) // 4

    def estimate_cost_for_prompt(
        self, formatted_prompt: FormattedPrompt, estimated_response_tokens: int = 50
    ) -> float:
        """
        Estimate cost for a formatted prompt.

        Args:
            formatted_prompt: FormattedPrompt object
            estimated_response_tokens: Estimated response length

        Returns:
            Estimated cost in USD
        """
        prompt_tokens = self.count_tokens(formatted_prompt.prompt_text)
        return CostCalculator.estimate_cost(
            self.primary_model, prompt_tokens, estimated_response_tokens
        )

    @retry(
        stop=stop_after_attempt(5),
        wait=wait_exponential(multiplier=2, min=8, max=30),
        retry=retry_if_exception_type(
            (openai.RateLimitError, openai.APIConnectionError)
        ),
    )
    async def _make_api_call(
        self,
        messages: List[Dict[str, str]],
        model_key: str,
        temperature: float,
        max_tokens: int,
        **kwargs,
    ) -> Dict[str, Any]:
        """
        Make the actual API call with retry logic for any provider.

        Args:
            messages: Messages for the API call
            model_key: Model key to identify which client/model to use
            temperature: Temperature parameter
            max_tokens: Maximum tokens to generate
            **kwargs: Additional parameters

        Returns:
            API response
        """
        try:
            model_config = self.available_models[model_key]
            client = self._get_or_create_client(model_key)
            
            response = await client.chat.completions.create(
                model=model_config['model_id'],
                messages=messages,
                temperature=temperature,
                max_tokens=max_tokens,
                timeout=model_config['timeout'],
                **kwargs,
            )
            return response
        except Exception as e:
            self.logger.error(f"API call failed for {model_key}: {e}")
            raise

    def _get_adaptive_max_tokens(self, formatted_prompt: FormattedPrompt) -> int:
        """Calculate adaptive max_tokens based on prompt type and input length."""
        base_tokens = self.default_max_tokens
        
        # Check if chain-of-thought prompt (needs significantly more space)
        is_cot = "chain_of_thought" in formatted_prompt.prompt_type.lower() or \
                 "step by step" in formatted_prompt.prompt_text.lower() or \
                 "reasoning" in formatted_prompt.prompt_text.lower() or \
                 "analyze" in formatted_prompt.prompt_text.lower()
        
        if is_cot:
            # Chain-of-thought needs much more tokens for proper reasoning
            adaptive_tokens = 8192  # Give CoT plenty of space
        else:
            # Zero-shot can use fewer tokens
            adaptive_tokens = min(2048, max(1024, base_tokens))
        
        # For very long inputs, ensure we have enough space regardless
        input_tokens = self.count_tokens(formatted_prompt.prompt_text)
        if input_tokens > 10000:  # Very long inputs
            adaptive_tokens = min(adaptive_tokens + 2048, 12288)
        elif input_tokens > 6000:  # Long inputs  
            adaptive_tokens = min(adaptive_tokens + 1024, 10240)
        
        self.logger.debug(f"Adaptive tokens: {adaptive_tokens} (input: {input_tokens}, CoT: {is_cot})")
        return adaptive_tokens

    async def generate_response(
        self,
        formatted_prompt: FormattedPrompt,
        temperature: Optional[float] = None,
        max_tokens: Optional[int] = None,
        model: Optional[str] = None,
        **kwargs,
    ) -> LLMResponse:
        """
        Generate response from any configured LLM for factuality evaluation.

        Args:
            formatted_prompt: FormattedPrompt object
            temperature: Temperature parameter (0.0 for deterministic)
            max_tokens: Maximum tokens to generate
            model: Model key to use (defaults to primary model)
            **kwargs: Additional generation parameters

        Returns:
            LLMResponse object

        Raises:
            ValueError: If budget exceeded or other validation errors
        """
        start_time = time.time()

        # Use defaults if not specified
        temperature = (
            temperature if temperature is not None else self.default_temperature
        )
        max_tokens = max_tokens if max_tokens is not None else self._get_adaptive_max_tokens(formatted_prompt)
        model_key = model or self.primary_model

        # Get model configuration
        if model_key not in self.available_models:
            raise ValueError(f"Model {model_key} not available. Available models: {list(self.available_models.keys())}")
        
        model_config = self.available_models[model_key]

        # Count prompt tokens
        prompt_tokens = self.count_tokens(formatted_prompt.prompt_text)

        # Check budget before making call (only for OpenAI models)
        if model_config.get('cost_tracking', False):
            estimated_cost = CostCalculator.estimate_cost(model_config['model_id'], prompt_tokens, max_tokens)
            if not self.cost_tracker.can_afford(estimated_cost):
                raise ValueError("Estimated cost exceeds budget limits")
        else:
            estimated_cost = 0.0

        # Apply rate limiting
        wait_time = self.rate_limiter.wait_if_needed(prompt_tokens + max_tokens)

        # Prepare messages
        messages = [{"role": "user", "content": formatted_prompt.prompt_text}]

        # Track request
        self.total_requests += 1

        try:
            # Make API call
            response = await self._make_api_call(
                messages=messages,
                model_key=model_key,
                temperature=temperature,
                max_tokens=max_tokens,
                **kwargs,
            )

            # Extract response data
            choice = response.choices[0]
            content = choice.message.content.strip()
            finish_reason = choice.finish_reason

            # Log what model was actually used
            returned_model = response.model
            self.logger.info(f"API returned model: {returned_model}, requested: {model_config['model_id']}")

            usage = response.usage
            completion_tokens = usage.completion_tokens
            total_tokens = usage.total_tokens
            actual_prompt_tokens = usage.prompt_tokens

            # Calculate actual cost (only for OpenAI models)
            actual_cost = 0.0
            if model_config.get('cost_tracking', False):
                actual_cost = CostCalculator.calculate_cost(
                    model_config['model_id'], actual_prompt_tokens, completion_tokens
                )

                # Track cost
                self.cost_tracker.add_cost(
                    cost=actual_cost,
                    model=model_config['model_id'],
                    experiment_name=self.config.get("experiment_name"),
                    task_name=formatted_prompt.task_type,
                )

            # Calculate response time
            response_time = time.time() - start_time - wait_time

            # Create response object
            llm_response = LLMResponse(
                content=content,
                model=model_config['model_id'],
                provider=model_config['provider'],
                prompt_tokens=actual_prompt_tokens,
                completion_tokens=completion_tokens,
                total_tokens=total_tokens,
                cost=actual_cost,
                response_time=response_time,
                timestamp=datetime.now().isoformat(),
                finish_reason=finish_reason,
                temperature=temperature,
                max_tokens=max_tokens,
                request_id=response.id,
            )

            # Update statistics
            self.successful_requests += 1
            self.total_tokens_used += total_tokens

            # Log detailed request/response for debugging - only for errors or significant costs
            if actual_cost > 0.001 or response_time > 5.0:  # Only log expensive or slow requests
                self.logger.debug(
                    f"{model_config['name']} response: {completion_tokens} tokens, "
                    f"${actual_cost:.4f}, {response_time:.2f}s"
                )

            return llm_response

        except Exception as e:
            self.failed_requests += 1
            self.logger.error(f"Failed to generate response with {model_key}: {e}")
            raise

    def get_available_models(self) -> List[str]:
        """Get list of available model keys."""
        return list(self.available_models.keys())

    def get_model_info(self, model_key: str) -> Dict[str, Any]:
        """Get information about a specific model."""
        if model_key not in self.available_models:
            raise ValueError(f"Model {model_key} not available")
        
        model_config = self.available_models[model_key]
        return {
            'name': model_config['name'],
            'provider': model_config['provider'],
            'model_id': model_config['model_id'],
            'cost_tracking': model_config['cost_tracking']
        }

    def switch_primary_model(self, model_key: str):
        """Switch the primary model to a different available model."""
        if model_key not in self.available_models:
            raise ValueError(f"Model {model_key} not available. Available models: {list(self.available_models.keys())}")
        
        old_primary = self.primary_model
        self.primary_model = model_key
        self.primary_model_config = self.available_models[model_key]
        
        self.logger.info(f"Switched primary model from {old_primary} to {model_key}")

    def get_provider_info(self) -> Dict[str, Any]:
        """Get information about the current provider and available models."""
        return {
            "primary_model": self.primary_model,
            "primary_provider": self.primary_model_config.get('provider', 'unknown'),
            "available_models": {
                model_key: {
                    "name": config.get('name', model_key),
                    "provider": config.get('provider', 'unknown'),
                    "model_id": config.get('model_id', model_key)
                }
                for model_key, config in self.available_models.items()
            },
            "total_models": len(self.available_models)
        }

    def get_cost_summary(self) -> Dict[str, Any]:
        """Get cost tracking summary for OpenAI models."""
        cost_analysis = self.cost_tracker.get_analysis()
        
        # Add model-specific cost breakdown
        cost_analysis["model_costs"] = {}
        openai_models = [
            model_key for model_key, config in self.available_models.items()
            if config.get('cost_tracking', False)
        ]
        
        cost_analysis["tracking_enabled_models"] = openai_models
        cost_analysis["tracking_disabled_models"] = [
            model_key for model_key, config in self.available_models.items()
            if not config.get('cost_tracking', False)
        ]
        
        return cost_analysis

    def parse_factuality_response(
        self, response: LLMResponse, task_type: str
    ) -> APICallResult:
        """
        Parse LLM response for factuality evaluation tasks.

        Args:
            response: LLMResponse object
            task_type: Type of factuality task

        Returns:
            APICallResult with parsed content
        """
        content = response.content.strip()
        parsing_errors = []
        confidence_score = None

        try:
            if task_type == "entailment_inference":
                parsed_content = self._parse_entailment_response(content)
            elif task_type == "summary_ranking":
                parsed_content = self._parse_ranking_response(content, response.finish_reason)
            elif task_type == "consistency_rating":
                parsed_content = self._parse_rating_response(content)
            else:
                raise ValueError(f"Unknown task type: {task_type}")

            parsing_successful = True

        except Exception as e:
            parsing_errors.append(str(e))
            parsed_content = {"raw_content": content, "parsing_error": str(e)}
            parsing_successful = False

        return APICallResult(
            raw_response=response,
            parsed_content=parsed_content,
            task_type=task_type,
            parsing_successful=parsing_successful,
            parsing_errors=parsing_errors,
            confidence_score=confidence_score,
        )

    def _parse_entailment_response(self, content: str) -> Dict[str, Any]:
        """Parse entailment inference response."""
        # Focus on final classification section for chain-of-thought responses
        search_content = content
        if "FINAL CLASSIFICATION:" in content.upper():
            final_classification_idx = content.upper().find("FINAL CLASSIFICATION:")
            search_content = content[final_classification_idx:]
        
        content_upper = search_content.upper()
        
        # Count occurrences to find the most likely answer
        entailment_count = content_upper.count("ENTAILMENT")
        contradiction_count = content_upper.count("CONTRADICTION")
        
        # Determine prediction based on counts and positions
        if entailment_count > contradiction_count:
            prediction = 1
            answer = "ENTAILMENT"
        elif contradiction_count > entailment_count:
            prediction = 0
            answer = "CONTRADICTION"
        elif entailment_count == contradiction_count and entailment_count > 0:
            # If tied, use the last occurrence (likely the final decision)
            last_entailment = content_upper.rfind("ENTAILMENT")
            last_contradiction = content_upper.rfind("CONTRADICTION")
            
            if last_entailment > last_contradiction:
                prediction = 1
                answer = "ENTAILMENT"
            else:
                prediction = 0
                answer = "CONTRADICTION"
        else:
            # No clear answer found, default to CONTRADICTION
            prediction = 0
            answer = "CONTRADICTION"

        return {"prediction": prediction, "answer": answer, "raw_content": content}

    def _parse_ranking_response(self, content: str, finish_reason: str = None) -> Dict[str, Any]:
        """
        Parse summary ranking response with multiple fallback strategies.
        
        Strategies (in order of priority):
        1. Summary sequence extraction (works for most formats)
        2. Numbered list parsing (position → summary mapping)
        3. Explicit ranking patterns (Summary X: Rank Y)
        4. Generic ranking patterns (RANKING: 1, 2, 3)
        5. Aggressive fallback parsing
        """
        # Simplified parsing: focus on the most reliable patterns
        # 1. Try explicit ranking patterns first (RANKING: X, Y, Z)
        ranking = self._try_generic_pattern_parsing(content)
        if ranking is not None:
            return self._finalize_ranking(ranking, content)
            
        # 2. Try simple number sequences (1, 2, 3 or 1 2 3)
        ranking = self._try_simple_sequence_parsing(content)
        if ranking is not None:
            return self._finalize_ranking(ranking, content)
            
        # 3. Basic fallback for edge cases
        ranking = self._try_basic_fallback_parsing(content, finish_reason)
        if ranking is not None:
            return self._finalize_ranking(ranking, content)
            
        raise ValueError("No ranking information found")

    def _try_simple_sequence_parsing(self, content: str) -> Optional[Dict[int, int]]:
        """Parse simple number sequences like '1, 2, 3' or '1 2 3'."""
        # Look for sequences of numbers (not preceded by explicit ranking keywords)
        if any(keyword in content.upper() for keyword in ["RANKING:", "FINAL RANKING:"]):
            return None  # Let generic pattern parsing handle these
            
        # Find sequences of 2+ numbers
        number_sequences = re.findall(r"(?:^|\s)(\d+(?:[,\s]+\d+)+)(?:\s|$)", content, re.MULTILINE)
        
        for sequence in number_sequences:
            numbers = re.findall(r"\d+", sequence)
            if len(numbers) >= 2:
                valid_numbers = [int(n) for n in numbers if 1 <= int(n) <= 10]
                if len(valid_numbers) >= 2:
                    # Interpret as preference order
                    ranking = {}
                    for rank_position, summary_number in enumerate(valid_numbers):
                        summary_idx = summary_number - 1  # Convert to 0-based
                        actual_rank = rank_position + 1   # 1-based rank (1=best)
                        ranking[summary_idx] = actual_rank
                    return ranking
        
        return None

    def _try_basic_fallback_parsing(self, content: str, finish_reason: str) -> Optional[Dict[int, int]]:
        """Simple fallback parsing for edge cases."""
        # Check for truncation first
        if finish_reason == 'length':
            raise ValueError(f"Response was truncated due to token limit. Content: {content[:100]}...")
            
        # Extract all numbers and try to form a ranking
        all_numbers = re.findall(r"\d+", content)
        valid_numbers = [int(n) for n in all_numbers if 1 <= int(n) <= 10]
        
        if len(valid_numbers) >= 2:
            # Take the last few numbers as potential ranking
            last_numbers = valid_numbers[-3:] if len(valid_numbers) >= 3 else valid_numbers
            
            # Remove duplicates while preserving order
            seen = set()
            unique_numbers = []
            for num in reversed(last_numbers):
                if num not in seen:
                    unique_numbers.append(num)
                    seen.add(num)
            unique_numbers = list(reversed(unique_numbers))
            
            if len(unique_numbers) >= 2:
                # Interpret as preference order
                ranking = {}
                for rank_position, summary_number in enumerate(unique_numbers):
                    summary_idx = summary_number - 1  # Convert to 0-based
                    actual_rank = rank_position + 1   # 1-based rank (1=best)
                    ranking[summary_idx] = actual_rank
                return ranking
        
        raise ValueError(f"Could not find sufficient ranking numbers. Content: {content[:200]}...")

    def _try_generic_pattern_parsing(self, content: str) -> Optional[Dict[int, int]]:
        """Strategy 4: Parse generic ranking patterns like 'RANKING: 1, 2, 3'."""
        content_clean = content.strip()
        
        patterns = [
            r"(?:FINAL RANKING|RANKING|ranking)\s*(?:is)?\s*:\s*([0-9,\s\[\]]+)",
            r"(?:FINAL RANKING|RANKING|ranking)\s*(?:is)?\s*[\s:]*([0-9,\s\[\]]+)",
            r"FINAL RANKING[\s:]*([0-9,\s\[\]]+)",
            r"\[([0-9,\s]+)\]",
            r"([0-9]+(?:\s*,\s*[0-9]+)+)",
            r"([0-9,\s]+)$",
        ]
        
        for pattern in patterns:
            matches = re.findall(pattern, content_clean, re.IGNORECASE)
            if matches:
                ranking_numbers = matches[-1]  # Take the last match
                numbers = re.findall(r"\d+", ranking_numbers)
                
                if numbers:
                    valid_numbers = [int(n) for n in numbers if 1 <= int(n) <= 10]
                    if len(valid_numbers) >= 2:
                        # Interpret as order preference: "2, 1, 3" means Summary 2 first, Summary 1 second, Summary 3 third
                        # Convert to ranking: Summary at position i gets rank based on its position in the list
                        ranking = {}
                        for rank_position, summary_number in enumerate(valid_numbers):
                            summary_idx = summary_number - 1  # Convert to 0-based
                            actual_rank = rank_position + 1   # 1-based rank (1=best)
                            ranking[summary_idx] = actual_rank
                        return ranking
        
        return None

    def _finalize_ranking(self, ranking: Dict[int, int], content: str) -> Dict[str, Any]:
        """Apply smart extension logic and convert to final format."""
        # Smart ranking extension logic for partial rankings
        expected_count = 3  # Assume 3 summaries by default
        current_count = len(ranking)
        
        if current_count < expected_count:
            # Find missing indices and fill with missing ranks
            provided_ranks = set(ranking.values())
            all_possible_ranks = set(range(1, expected_count + 1))
            missing_ranks = sorted(all_possible_ranks - provided_ranks)
            
            for i in range(current_count, expected_count):
                if missing_ranks:
                    ranking[i] = missing_ranks.pop(0)
                else:
                    ranking[i] = max(provided_ranks) + 1 + (i - current_count)
                    
        elif current_count > expected_count:
            # Truncate to expected size
            ranking = {i: ranking[i] for i in range(expected_count) if i in ranking}

        # Convert to ranked list format
        max_summary_idx = max(ranking.keys())
        ranked_list = [0] * (max_summary_idx + 1)
        for summary_idx, rank in ranking.items():
            ranked_list[summary_idx] = rank

        return {"ranking": ranking, "ranked_list": ranked_list, "raw_content": content}

    def _parse_rating_response(self, content: str) -> Dict[str, Any]:
        """Parse consistency rating response."""
        # Focus on final rating section for chain-of-thought responses
        search_content = content
        if "FINAL RATING:" in content.upper():
            final_rating_idx = content.upper().find("FINAL RATING:")
            search_content = content[final_rating_idx:]
        
        # Try common rating patterns
        rating_patterns = [
            r"[Rr]ating:?\s*(\d+(?:\.\d+)?)",
            r"[Ss]core:?\s*(\d+(?:\.\d+)?)", 
            r"(\d+(?:\.\d+)?)\s*\/\s*100",
            r"(\d+(?:\.\d+)?)\s*out\s*of\s*100",
            r"(\d+(?:\.\d+)?)\s*points?"
        ]
        
        rating = None
        for pattern in rating_patterns:
            matches = re.findall(pattern, search_content, re.IGNORECASE)
            if matches:
                candidate = float(matches[-1])  # Take the last match
                if 0 <= candidate <= 100:
                    rating = candidate
                    break
        
        # Fallback: look for any valid numbers
        if rating is None:
            numbers = re.findall(r"\b(\d+(?:\.\d+)?)\b", search_content)
            for num_str in reversed(numbers):  # Start from the end
                candidate = float(num_str)
                if 0 <= candidate <= 100:
                    rating = candidate
                    break
        
        if rating is None:
            finish_reason = getattr(self, '_last_finish_reason', None)
            if finish_reason == 'length':
                raise ValueError(f"Response truncated due to token limit. Content: {content[:100]}...")
            else:
                raise ValueError(f"No numeric rating found in response: {content[:200]}...")

        return {"rating": rating, "raw_content": content}

    async def evaluate_factuality(
        self, formatted_prompt: FormattedPrompt, **generation_kwargs
    ) -> APICallResult:
        """
        Complete factuality evaluation pipeline.

        Args:
            formatted_prompt: FormattedPrompt object
            **generation_kwargs: Additional generation parameters

        Returns:
            APICallResult with parsed factuality evaluation
        """
        # Generate response
        response = await self.generate_response(formatted_prompt, **generation_kwargs)

        # Parse response for factuality task
        result = self.parse_factuality_response(response, formatted_prompt.task_type)

        return result

    def get_usage_statistics(self) -> Dict[str, Any]:
        """Get comprehensive usage statistics."""
        cost_analysis = self.cost_tracker.get_analysis()
        rate_limit_status = self.rate_limiter.get_current_usage()

        return {
            "requests": {
                "total": self.total_requests,
                "successful": self.successful_requests,
                "failed": self.failed_requests,
                "success_rate": self.successful_requests / max(self.total_requests, 1),
            },
            "tokens": {
                "total_used": self.total_tokens_used,
                "average_per_request": self.total_tokens_used
                / max(self.successful_requests, 1),
            },
            "cost": cost_analysis,
            "rate_limits": rate_limit_status,
            "models": {
                "primary": self.primary_model,
                "fallbacks": self.fallback_models,
            },
        }


# Convenience functions


def create_openai_client(config: Optional[Dict[str, Any]] = None) -> OpenAIClient:
    """
    Create OpenAI client instance.

    Args:
        config: Configuration dictionary

    Returns:
        OpenAIClient instance
    """
    return OpenAIClient(config)


def validate_openai_config(config: Dict[str, Any]) -> bool:
    """
    Validate OpenAI configuration.

    Args:
        config: Configuration dictionary

    Returns:
        True if configuration is valid

    Raises:
        ValueError: If configuration is invalid
    """
    if "openai" not in config:
        raise ValueError("Missing 'openai' section in configuration")

    openai_config = config["openai"]

    # Check required sections
    required_sections = ["models", "api", "rate_limits", "generation", "cost_control"]
    for section in required_sections:
        if section not in openai_config:
            raise ValueError(f"Missing required OpenAI config section: {section}")

    # Validate models
    models = openai_config["models"]
    if "primary" not in models:
        raise ValueError("Missing primary model in OpenAI configuration")

    # Validate cost control
    cost_control = openai_config["cost_control"]
    if cost_control.get("total_budget", 0) <= 0:
        raise ValueError("Total budget must be positive")

    # Check API key
    if not os.getenv("OPENAI_API_KEY"):
        raise ValueError("OPENAI_API_KEY environment variable not set")

    return True


def estimate_token_cost(
    text: str, model: Optional[str] = None, estimated_response_tokens: int = 50
) -> float:
    """
    Estimate cost for text processing.

    Args:
        text: Input text
        model: Model name (if None, uses first available model for estimation)
        estimated_response_tokens: Estimated response length

    Returns:
        Estimated cost in USD
    """
    if model is None:
        # Use first available model for estimation
        model = next(iter(CostCalculator.PRICING.keys()))
    
    try:
        encoder = tiktoken.encoding_for_model(model)
        prompt_tokens = len(encoder.encode(text))
    except:
        prompt_tokens = len(text) // 4  # Rough estimation

    return CostCalculator.estimate_cost(model, prompt_tokens, estimated_response_tokens)


def parse_factuality_response(content: str, task_type: str) -> Dict[str, Any]:
    """
    Standalone function to parse factuality responses.

    Args:
        content: Response content
        task_type: Type of factuality task

    Returns:
        Parsed response dictionary
    """
    # Create temporary client for parsing
    client = OpenAIClient()

    # Create dummy response for parsing
    dummy_response = LLMResponse(
        content=content,
        model="gpt-4o-mini",
        provider="openai",
        prompt_tokens=0,
        completion_tokens=0,
        total_tokens=0,
        cost=0.0,
        response_time=0.0,
        timestamp=datetime.now().isoformat(),
        finish_reason="stop",
        temperature=0.0,
        max_tokens=0,
    )

    result = client.parse_factuality_response(dummy_response, task_type)
    return result.parsed_content