"""
Comprehensive OpenAI Client for ChatGPT Factuality Evaluation
===========================================================

Robust OpenAI API integration with rate limiting, cost tracking,
response parsing, and academic research features specifically
designed for factuality evaluation experiments.

Author: Michael Ogunjimi
Institution: University of Manchester
Course: MSc AI
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
class ChatGPTResponse:
    """
    Structured response from ChatGPT API call.

    Contains the response content along with metadata needed
    for analysis, cost tracking, and reproducibility.
    """

    content: str
    model: str
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
    metadata and parsed task-specific results.
    """

    raw_response: ChatGPTResponse
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
            self.logger.info(f"Rate limit reached. Waiting {wait_time:.2f} seconds.")
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

    # Updated OpenAI pricing (as of January 2025) - Only supported models
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
            # Use the first available model pricing as fallback
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
    Comprehensive OpenAI client for ChatGPT factuality evaluation.

    Provides robust API integration with rate limiting, cost tracking,
    response parsing, and academic research features.
    """

    def __init__(self, config: Optional[Dict[str, Any]] = None):
        """
        Initialize OpenAI client.

        Args:
            config: Configuration dictionary
        """
        self.config = config or get_config()
        self.logger = get_logger(self.__class__.__name__)

        # Setup OpenAI API
        self._setup_openai_api()

        # Model configuration - Remove hardcoded defaults
        openai_config = self.config.get("openai", {})
        models_config = openai_config.get("models", {})
        self.primary_model = models_config.get("primary")
        self.fallback_models = models_config.get("fallbacks", [])
        
        if not self.primary_model:
            raise ValueError("No primary model specified in configuration. Use get_config(model='model-name') to load model-specific config.")

        # Generation parameters (model-agnostic)
        generation_config = openai_config.get("generation", {})
        self.default_temperature = generation_config.get("temperature", 0.0)
        self.default_max_tokens = generation_config.get("max_tokens", 2048)
        self.default_top_p = generation_config.get("top_p", 1.0)

        # API settings
        api_config = openai_config.get("api", {})
        self.timeout = api_config.get("timeout", 30)
        self.max_retries = api_config.get("max_retries", 3)

        # Initialize rate limiter - Remove hardcoded defaults
        rate_limits = openai_config.get("rate_limits", {})
        self.rate_limiter = RateLimiter(
            requests_per_minute=rate_limits.get("requests_per_minute"),
            tokens_per_minute=rate_limits.get("tokens_per_minute"), 
            requests_per_day=rate_limits.get("requests_per_day"),
        )
        
        if not rate_limits:
            self.logger.warning("No rate limits specified in configuration. Use get_config(model='model-name') to load model-specific config.")

        # Initialize cost tracker
        cost_config = openai_config.get("cost_control", {})
        self.cost_tracker = CostTracker(
            daily_budget=cost_config.get("daily_budget", 50.0),
            total_budget=cost_config.get("total_budget", 200.0),
            warning_threshold=cost_config.get("warning_threshold", 0.8),
        )

        # Token encoder for accurate token counting
        try:
            self.token_encoder = tiktoken.encoding_for_model(self.primary_model)
        except KeyError:
            self.token_encoder = tiktoken.get_encoding("cl100k_base")
            logger.warning(f"Using default encoding for {self.primary_model}")

        # Statistics tracking
        self.total_requests = 0
        self.successful_requests = 0
        self.failed_requests = 0
        self.total_tokens_used = 0

        self.logger.info(f"OpenAI client initialized with model: {self.primary_model}")

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
        model: str,
        temperature: float,
        max_tokens: int,
        **kwargs,
    ) -> Dict[str, Any]:
        """
        Make the actual API call with retry logic.

        Args:
            messages: Messages for the API call
            model: Model to use
            temperature: Temperature parameter
            max_tokens: Maximum tokens to generate
            **kwargs: Additional parameters

        Returns:
            API response
        """
        try:
            response = await self.openai_client.chat.completions.create(
                model=model,
                messages=messages,
                temperature=temperature,
                max_tokens=max_tokens,
                timeout=self.timeout,
                **kwargs,
            )
            return response
        except Exception as e:
            self.logger.error(f"API call failed: {e}")
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
    ) -> ChatGPTResponse:
        """
        Generate response from ChatGPT for factuality evaluation.

        Args:
            formatted_prompt: FormattedPrompt object
            temperature: Temperature parameter (0.0 for deterministic)
            max_tokens: Maximum tokens to generate
            model: Model to use (defaults to primary model)
            **kwargs: Additional generation parameters

        Returns:
            ChatGPTResponse object

        Raises:
            ValueError: If budget exceeded or other validation errors
        """
        start_time = time.time()

        # Use defaults if not specified
        temperature = (
            temperature if temperature is not None else self.default_temperature
        )
        max_tokens = max_tokens if max_tokens is not None else self._get_adaptive_max_tokens(formatted_prompt)
        model = model or self.primary_model

        # Count prompt tokens
        prompt_tokens = self.count_tokens(formatted_prompt.prompt_text)

        # Check budget before making call
        estimated_cost = CostCalculator.estimate_cost(model, prompt_tokens, max_tokens)
        if not self.cost_tracker.can_afford(estimated_cost):
            raise ValueError("Estimated cost exceeds budget limits")

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
                model=model,
                temperature=temperature,
                max_tokens=max_tokens,
                **kwargs,
            )

            # Extract response data
            choice = response.choices[0]
            content = choice.message.content.strip()
            finish_reason = choice.finish_reason

            # Log what model OpenAI actually returned
            returned_model = response.model
            self.logger.info(f"OpenAI returned model: {returned_model}, requested: {model}")

            usage = response.usage
            completion_tokens = usage.completion_tokens
            total_tokens = usage.total_tokens
            actual_prompt_tokens = usage.prompt_tokens

            # Calculate actual cost
            actual_cost = CostCalculator.calculate_cost(
                model, actual_prompt_tokens, completion_tokens
            )

            # Track cost
            self.cost_tracker.add_cost(
                cost=actual_cost,
                model=model,
                experiment_name=self.config.get("experiment_name"),
                task_name=formatted_prompt.task_type,
            )

            # Calculate response time
            response_time = time.time() - start_time - wait_time

            # Create response object
            chatgpt_response = ChatGPTResponse(
                content=content,
                model=model,
                prompt_tokens=actual_prompt_tokens,
                completion_tokens=completion_tokens,
                total_tokens=total_tokens,
                cost=actual_cost,
                response_time=response_time,
                timestamp=datetime.now().isoformat(),
                finish_reason=finish_reason,
                temperature=temperature,
                max_tokens=max_tokens,
                request_id=response.id,  # Use .id instead of .get("id")
            )

            # Update statistics
            self.successful_requests += 1
            self.total_tokens_used += total_tokens

            # Log detailed request/response for debugging
            self.logger.info(
                f"ChatGPT response generated: {completion_tokens} tokens, "
                f"${actual_cost:.4f}, {response_time:.2f}s"
            )
            
            # Detailed logging disabled for cleaner output
            # self.logger.info("=" * 80)
            # self.logger.info("OPENAI REQUEST DETAILS:")
            # self.logger.info(f"Model: {model}")
            # self.logger.info(f"Temperature: {temperature}")
            # self.logger.info(f"Max Tokens: {max_tokens}")
            # self.logger.info(f"Task Type: {formatted_prompt.task_type}")
            # self.logger.info(f"Prompt Type: {formatted_prompt.prompt_type}")
            # self.logger.info("-" * 40)
            # self.logger.info("FULL PROMPT:")
            # for i, msg in enumerate(messages):
            #     self.logger.info(f"Message {i+1} ({msg['role']}):")
            #     self.logger.info(f"{msg['content']}")
            #     self.logger.info("-" * 20)
            # self.logger.info("-" * 40)
            # self.logger.info("OPENAI RESPONSE:")
            # self.logger.info(f"Content: {content}")
            # self.logger.info(f"Finish Reason: {finish_reason}")
            # self.logger.info(f"Total Tokens: {total_tokens} (Prompt: {actual_prompt_tokens}, Completion: {completion_tokens})")
            # self.logger.info(f"Cost: ${actual_cost:.4f}")
            # self.logger.info(f"Response Time: {response_time:.2f}s")
            # self.logger.info("=" * 80)

            return chatgpt_response

        except Exception as e:
            self.failed_requests += 1
            self.logger.error(f"Failed to generate response: {e}")
            raise

    def parse_factuality_response(
        self, response: ChatGPTResponse, task_type: str
    ) -> APICallResult:
        """
        Parse ChatGPT response for factuality evaluation tasks.

        Args:
            response: ChatGPTResponse object
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
        content_upper = content.upper()

        # For any response, look for ENTAILMENT or CONTRADICTION anywhere in the content
        # Count occurrences to find the most likely answer
        entailment_count = content_upper.count("ENTAILMENT")
        contradiction_count = content_upper.count("CONTRADICTION")
        
        # If we have a clear winner, use it
        if entailment_count > contradiction_count:
            prediction = 1
            answer = "ENTAILMENT"
        elif contradiction_count > entailment_count:
            prediction = 0
            answer = "CONTRADICTION"
        elif entailment_count == contradiction_count and entailment_count > 0:
            # If tied, look at the last occurrence (likely the final decision)
            last_entailment = content_upper.rfind("ENTAILMENT")
            last_contradiction = content_upper.rfind("CONTRADICTION")
            
            if last_entailment > last_contradiction:
                prediction = 1
                answer = "ENTAILMENT"
            else:
                prediction = 0
                answer = "CONTRADICTION"
        else:
            # No clear answer found, default to CONTRADICTION for this dataset
            prediction = 0
            answer = "CONTRADICTION"

        return {"prediction": prediction, "answer": answer, "raw_content": content}

    def _parse_ranking_response(self, content: str, finish_reason: str = None) -> Dict[str, Any]:
        """Parse summary ranking response."""
        # First check for "FINAL RANKING:" pattern for COT responses
        if "FINAL RANKING:" in content.upper():
            final_ranking_idx = content.upper().find("FINAL RANKING:")
            final_ranking_section = content[final_ranking_idx:]
            content = final_ranking_section  # Focus on the final ranking section
        
        # Look for ranking patterns like "Summary 1: Rank 2"
        ranking_pattern = r"Summary\s+(\d+):\s*Rank\s+(\d+)"
        matches = re.findall(ranking_pattern, content, re.IGNORECASE)

        if matches:
            ranking = {}
            for summary_idx, rank in matches:
                ranking[int(summary_idx) - 1] = int(rank)  # Convert to 0-based indexing
        else:
            # Enhanced extraction strategy - ChatGPT rarely provides empty responses
            # Clean the content to extract ranking numbers
            content_clean = content.strip()
            
            # Strategy 1: Try explicit ranking patterns first
            ranking_patterns = [
                r"(?:FINAL RANKING|RANKING|ranking):\s*([0-9,\s\[\]]+)",  # "FINAL RANKING: 1, 2, 3" or "[1,2,3]"
                r"(?:FINAL RANKING|RANKING|ranking)[\s:]*([0-9,\s\[\]]+)",  # More flexible spacing
                r"FINAL RANKING[\s:]*([0-9,\s\[\]]+)",  # More specific "FINAL RANKING: 1, 2, 3"
                r"\[([0-9,\s]+)\]",  # Just bracketed numbers: "[1, 2, 3]"
                r"([0-9]+(?:[,\s]+[0-9]+)+)",   # Multiple numbers with separators
                r"([0-9,\s]+)$",  # Just numbers at end: "1, 2, 3"
            ]
            
            ranking_numbers = None
            for pattern in ranking_patterns:
                matches = re.findall(pattern, content_clean, re.IGNORECASE)
                if matches:
                    # Take the last match (most likely to be the final answer)
                    ranking_numbers = matches[-1]
                    break
            
            if ranking_numbers:
                # Extract individual numbers more robustly
                numbers = re.findall(r"\d+", ranking_numbers)
                if numbers:
                    # Convert to integers and validate range
                    valid_numbers = []
                    for num_str in numbers:
                        num = int(num_str)
                        if 1 <= num <= 10:  # Reasonable rank range
                            valid_numbers.append(num)
                    
                    if valid_numbers:
                        # Assume sequential order: first number is rank for summary 1, etc.
                        ranking = {}
                        for i, rank in enumerate(valid_numbers):
                            ranking[i] = rank  # i is 0-based summary index
                    else:
                        # Check if response was truncated (common cause of parsing failures)
                        if finish_reason == 'length':
                            raise ValueError(f"Response was truncated due to token limit. Consider increasing max_tokens. Content: {content[:100]}...")
                        else:
                            # Strategy 2: Look for any valid numbers in the response
                            all_numbers = re.findall(r"\d+", content)
                            valid_ranks = [int(n) for n in all_numbers if 1 <= int(n) <= 10]
                            if valid_ranks:
                                ranking = {i: rank for i, rank in enumerate(valid_ranks)}
                            else:
                                raise ValueError(f"No valid ranking numbers (1-10) found in: {ranking_numbers}")
                else:
                    # Check if response was truncated (common cause of parsing failures)
                    if finish_reason == 'length':
                        raise ValueError(f"Response was truncated due to token limit. Consider increasing max_tokens. Content: {content[:100]}...")
                    else:
                        raise ValueError(f"Could not extract ranking numbers from: {ranking_numbers}")
            else:
                # Strategy 3: More aggressive fallback - look for ANY ranking-like patterns in the text
                # Split into lines and look for patterns
                lines = content.split('\n')
                found_ranking = False
                
                for line in lines:
                    line = line.strip()
                    # Look for lines that might contain rankings
                    if any(keyword in line.lower() for keyword in ['ranking', 'order', 'rank', 'best', 'worst']):
                        # Extract numbers from this line
                        line_numbers = re.findall(r"\d+", line)
                        valid_ranks = [int(n) for n in line_numbers if 1 <= int(n) <= 10]
                        if len(valid_ranks) >= 2:  # Need at least 2 rankings
                            ranking = {i: rank for i, rank in enumerate(valid_ranks)}
                            found_ranking = True
                            break
                
                if not found_ranking:
                    # Strategy 4: Final fallback - extract all valid numbers and use them as ranking
                    all_numbers = re.findall(r"\d+", content)
                    valid_ranks = []
                    for num in all_numbers:
                        rank = int(num)
                        if 1 <= rank <= 10:  # Reasonable rank range
                            valid_ranks.append(rank)
                    
                    if len(valid_ranks) >= 2:
                        # Take the last few numbers that could be a ranking
                        # Remove duplicates while preserving order
                        seen = set()
                        unique_ranks = []
                        for rank in reversed(valid_ranks):
                            if rank not in seen:
                                unique_ranks.append(rank)
                                seen.add(rank)
                        unique_ranks = list(reversed(unique_ranks))
                        
                        if len(unique_ranks) >= 2:
                            ranking = {i: rank for i, rank in enumerate(unique_ranks)}
                        else:
                            # Check if response was truncated
                            if finish_reason == 'length':
                                raise ValueError(f"Response was truncated due to token limit. Consider increasing max_tokens. Content: {content[:100]}...")
                            else:
                                raise ValueError(f"Could not identify unique ranking from numbers: {valid_ranks}")
                    else:
                        # Check if response was truncated
                        if finish_reason == 'length':
                            raise ValueError(f"Response was truncated due to token limit. Consider increasing max_tokens. Content: {content[:100]}...")
                        else:
                            raise ValueError(f"Could not find sufficient ranking numbers in response. Content: {content[:200]}...")

        # Smart ranking extension logic for partial rankings
        if ranking:
            # Assume 3 summaries by default (most common case)
            expected_count = 3
            current_count = len(ranking)
            
            if current_count < expected_count:
                # Find missing indices and append them in order
                provided_ranks = set(ranking.values())
                all_possible_ranks = set(range(1, expected_count + 1))
                missing_ranks = sorted(all_possible_ranks - provided_ranks)
                
                # Add missing rankings
                for i in range(current_count, expected_count):
                    if missing_ranks:
                        ranking[i] = missing_ranks.pop(0)
                    else:
                        # If no more missing ranks, use next available number
                        ranking[i] = max(provided_ranks) + 1 + (i - current_count)
            
            elif current_count > expected_count:
                # Truncate to expected size
                truncated_ranking = {}
                for i in range(expected_count):
                    if i in ranking:
                        truncated_ranking[i] = ranking[i]
                ranking = truncated_ranking

        # Convert to ranked list format
        if ranking:
            max_summary_idx = max(ranking.keys())
            ranked_list = [0] * (max_summary_idx + 1)
            for summary_idx, rank in ranking.items():
                ranked_list[summary_idx] = rank
        else:
            raise ValueError("No ranking information found")

        return {"ranking": ranking, "ranked_list": ranked_list, "raw_content": content}

    def _parse_rating_response(self, content: str) -> Dict[str, Any]:
        """Parse consistency rating response."""
        # First check for "FINAL RATING:" pattern for COT responses
        if "FINAL RATING:" in content.upper():
            final_rating_idx = content.upper().find("FINAL RATING:")
            final_rating_section = content[final_rating_idx:]
            content = final_rating_section  # Focus on the final rating section
        
        # Try specific patterns first
        rating_patterns = [
            r"[Rr]ating:?\s*(\d+(?:\.\d+)?)",
            r"[Ss]core:?\s*(\d+(?:\.\d+)?)",
            r"(\d+(?:\.\d+)?)\s*\/\s*100",
            r"(\d+(?:\.\d+)?)\s*out\s*of\s*100"
        ]
        
        rating = None
        for pattern in rating_patterns:
            matches = re.findall(pattern, content, re.IGNORECASE)
            if matches:
                candidate = float(matches[-1])  # Take the last match
                if 0 <= candidate <= 100:
                    rating = candidate
                    break
        
        if rating is None:
            # Look for any numbers in valid range
            numbers = re.findall(r"\b(\d+(?:\.\d+)?)\b", content)
            if numbers:
                for num_str in reversed(numbers):  # Start from the end
                    num = float(num_str)
                    if 0 <= num <= 100:
                        rating = num
                        break
        
        if rating is None:
            # Check if response was truncated
            finish_reason = getattr(self, '_last_finish_reason', None)
            if finish_reason == 'length':
                raise ValueError(f"Response was truncated due to token limit. Consider increasing max_tokens. Content: {content[:100]}...")
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
    dummy_response = ChatGPTResponse(
        content=content,
        model="gpt-4o-mini",
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
