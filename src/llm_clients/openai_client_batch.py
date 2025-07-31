"""
OpenAI Batch Client for Factuality Evaluation
=============================================

Specialized OpenAI client for batch processing operations, extending the
standard client with batch-specific functionality for cost-effective
large-scale factuality evaluation experiments.

Author: Michael Ogunjimi
Institution: University of Manchester
Course: MSc AI
"""

import asyncio
import json
import os
import time
from datetime import datetime
from pathlib import Path
from typing import Dict, List, Any, Optional, Tuple
from dataclasses import dataclass

import openai
import tiktoken
from tenacity import retry, stop_after_attempt, wait_exponential

from .openai_client import OpenAIClient, ChatGPTResponse, APICallResult, CostCalculator
from ..batch.batch_manager import BatchManager, BatchJob, BatchJobRequest, BatchStatus
from ..prompts.prompt_manager import FormattedPrompt
from ..utils.config import get_config
from ..utils.logging import get_logger, CostTracker


@dataclass
class BatchResult:
    """Result from batch processing operation."""
    job_id: str
    custom_id: str
    response: Optional[ChatGPTResponse]
    error: Optional[str]
    parsing_successful: bool
    parsed_content: Optional[Dict[str, Any]]

    def to_dict(self) -> Dict[str, Any]:
        """Convert to dictionary for analysis."""
        return {
            "job_id": self.job_id,
            "custom_id": self.custom_id,
            "response": self.response.to_dict() if self.response else None,
            "error": self.error,
            "parsing_successful": self.parsing_successful,
            "parsed_content": self.parsed_content
        }


class OpenAIBatchClient:
    """∫
    Specialized OpenAI client for batch processing operations.
    
    Extends the standard OpenAI client with batch-specific functionality
    including job management, cost optimization, and result processing
    tailored for academic factuality evaluation research.
    """

    def __init__(self, config=None, experiment_name: str = None):
        """
        Initialize batch client.

        Args:
            config: Configuration object
            experiment_name: Name of the current experiment
        """
        self.config = config or get_config()
        self.experiment_name = experiment_name or f"batch_client_{datetime.now().strftime('%Y%m%d_%H%M%S')}"
        
        # Initialize standard client for parsing utilities
        self.standard_client = OpenAIClient(self.config)
        
        # Initialize batch manager
        self.batch_manager = BatchManager(self.config, self.experiment_name)
        
        # Setup logging
        self.logger = get_logger(f"{self.__class__.__name__}")
        
        # Batch-specific configuration
        batch_config = self.config.get("openai.batch", {})
        self.enabled = batch_config.get("enabled", True)
        self.cost_savings = batch_config.get("cost_savings", 0.5)
        self.processing_timeout = batch_config.get("processing_timeout", 86400)
        
        # Model configuration
        self.model = self.config.get("openai.models.primary", "gpt-4.1-mini")
        
        # Token encoder for cost calculation
        try:
            self.token_encoder = tiktoken.encoding_for_model(self.model)
        except KeyError:
            self.token_encoder = tiktoken.get_encoding("cl100k_base")
            self.logger.warning(f"Using default encoding for {self.model}")
        
        # Initialize cost tracker for batch operations
        cost_config = self.config.get("openai.cost_control", {})
        self.cost_tracker = CostTracker(
            daily_budget=cost_config.get("daily_budget", 50.0),
            total_budget=cost_config.get("total_budget", 200.0),
            warning_threshold=cost_config.get("warning_threshold", 0.8)
        )
        
        self.logger.info(f"OpenAI Batch Client initialized for experiment: {self.experiment_name}")

    def is_batch_processing_available(self) -> bool:
        """
        Check if batch processing is available and enabled.

        Returns:
            True if batch processing can be used
        """
        return self.enabled and self.batch_manager.max_queue_size > 0

    async def submit_factuality_evaluation_batch(
        self,
        formatted_prompts: List[FormattedPrompt],
        task_type: str,
        dataset_name: str,
        prompt_type: str,
        generation_kwargs: Dict[str, Any] = None
    ) -> BatchJob:
        """
        Submit a batch of factuality evaluation prompts.

        Args:
            formatted_prompts: List of formatted prompts to evaluate
            task_type: Type of evaluation task
            dataset_name: Name of the dataset
            prompt_type: Type of prompt (zero_shot/chain_of_thought)
            generation_kwargs: Additional generation parameters

        Returns:
            BatchJob object for tracking
        """
        if not self.is_batch_processing_available():
            raise ValueError("Batch processing is not available or enabled")

        self.logger.info(f"Submitting batch: {task_type}/{dataset_name}/{prompt_type} with {len(formatted_prompts)} prompts")

        # Prepare generation parameters
        generation_kwargs = generation_kwargs or {}
        default_params = {
            "temperature": self.config.get("openai.generation.temperature", 0.0),
            "max_tokens": self.config.get("openai.generation.max_tokens", 2048),
            "top_p": self.config.get("openai.generation.top_p", 1.0)
        }
        default_params.update(generation_kwargs)

        # Create batch requests
        requests = []
        for i, prompt in enumerate(formatted_prompts):
            custom_id = f"{task_type}_{dataset_name}_{prompt_type}_{i}_{int(time.time())}"
            
            request = BatchJobRequest(
                custom_id=custom_id,
                method="POST",
                url="/v1/chat/completions",
                body={
                    "model": self.model,
                    "messages": [{"role": "user", "content": prompt.prompt_text}],
                    **default_params
                }
            )
            requests.append(request)

        # Submit batch job
        batch_job = await self.batch_manager.submit_batch_job(
            requests=requests,
            task_type=task_type,
            dataset_name=dataset_name,
            prompt_type=prompt_type,
            description=f"{self.experiment_name}_{task_type}_{dataset_name}_{prompt_type}"
        )

        self.logger.info(f"Batch job submitted: {batch_job.job_id}")
        return batch_job

    async def wait_for_batch_completion(
        self,
        batch_jobs: List[BatchJob],
        show_progress: bool = True
    ) -> List[BatchJob]:
        """
        Wait for multiple batch jobs to complete.

        Args:
            batch_jobs: List of batch jobs to wait for
            show_progress: Whether to show progress updates

        Returns:
            List of completed batch jobs
        """
        job_ids = [job.job_id for job in batch_jobs]
        self.logger.info(f"Waiting for {len(job_ids)} batch jobs to complete")

        completed_jobs_dict = await self.batch_manager.wait_for_completion(
            job_ids=job_ids,
            timeout=self.processing_timeout
        )

        # Return completed jobs in original order
        completed_jobs = []
        for job in batch_jobs:
            if job.job_id in completed_jobs_dict:
                completed_jobs.append(completed_jobs_dict[job.job_id])
            else:
                # Job not completed - update status
                updated_job = await self.batch_manager.get_batch_status(job.job_id)
                completed_jobs.append(updated_job or job)

        return completed_jobs

    async def download_and_parse_results(
        self,
        batch_job: BatchJob,
        formatted_prompts: List[FormattedPrompt]
    ) -> List[BatchResult]:
        """
        Download and parse results from a completed batch job.

        Args:
            batch_job: Completed batch job
            formatted_prompts: Original prompts for parsing context

        Returns:
            List of parsed batch results
        """
        if batch_job.status != BatchStatus.COMPLETED:
            raise ValueError(f"Batch job {batch_job.job_id} is not completed (status: {batch_job.status})")

        self.logger.info(f"Downloading results for batch job: {batch_job.job_id}")

        # Download raw results
        raw_results = await self.batch_manager.download_batch_results(batch_job)
        
        # Download errors if any
        errors = await self.batch_manager.download_batch_errors(batch_job)
        error_dict = {error.get("custom_id"): error for error in errors}

        # Create prompt lookup for parsing context
        prompt_lookup = {}
        for i, prompt in enumerate(formatted_prompts):
            # Match the custom_id pattern from batch submission
            custom_id_pattern = f"{batch_job.task_type}_{batch_job.dataset_name}_{batch_job.prompt_type}_{i}_"
            prompt_lookup[custom_id_pattern] = prompt

        # Parse results
        batch_results = []
        total_cost = 0.0

        for raw_result in raw_results:
            custom_id = raw_result.get("custom_id")
            
            # Check for errors first
            if custom_id in error_dict:
                error_info = error_dict[custom_id]
                batch_result = BatchResult(
                    job_id=batch_job.job_id,
                    custom_id=custom_id,
                    response=None,
                    error=str(error_info.get("error", "Unknown error")),
                    parsing_successful=False,
                    parsed_content=None
                )
                batch_results.append(batch_result)
                continue

            # Process successful response
            try:
                response_data = raw_result.get("response", {})
                
                if not response_data:
                    batch_result = BatchResult(
                        job_id=batch_job.job_id,
                        custom_id=custom_id,
                        response=None,
                        error="No response data",
                        parsing_successful=False,
                        parsed_content=None
                    )
                    batch_results.append(batch_result)
                    continue

                # Extract response content
                body = response_data.get("body", {})
                choices = body.get("choices", [])
                usage = body.get("usage", {})
                
                if not choices:
                    batch_result = BatchResult(
                        job_id=batch_job.job_id,
                        custom_id=custom_id,
                        response=None,
                        error="No choices in response",
                        parsing_successful=False,
                        parsed_content=None
                    )
                    batch_results.append(batch_result)
                    continue

                content = choices[0].get("message", {}).get("content", "")
                finish_reason = choices[0].get("finish_reason", "unknown")

                # Calculate cost
                prompt_tokens = usage.get("prompt_tokens", 0)
                completion_tokens = usage.get("completion_tokens", 0)
                total_tokens = usage.get("total_tokens", prompt_tokens + completion_tokens)
                
                cost = CostCalculator.calculate_cost(
                    self.model, prompt_tokens, completion_tokens
                ) * (1 - self.cost_savings)  # Apply batch discount
                
                total_cost += cost

                # Create ChatGPTResponse object
                chatgpt_response = ChatGPTResponse(
                    content=content,
                    model=self.model,
                    prompt_tokens=prompt_tokens,
                    completion_tokens=completion_tokens,
                    total_tokens=total_tokens,
                    cost=cost,
                    response_time=0.0,  # Not available for batch
                    timestamp=datetime.now().isoformat(),
                    finish_reason=finish_reason,
                    temperature=0.0,  # From batch request
                    max_tokens=2048,  # From batch request
                    request_id=batch_job.job_id
                )

                # Find matching prompt for parsing context
                matching_prompt = None
                for pattern, prompt in prompt_lookup.items():
                    if custom_id.startswith(pattern):
                        matching_prompt = prompt
                        break

                # Parse factuality response
                parsed_content = None
                parsing_successful = False
                
                if matching_prompt:
                    try:
                        api_result = self.standard_client.parse_factuality_response(
                            chatgpt_response, matching_prompt.task_type
                        )
                        parsed_content = api_result.parsed_content
                        parsing_successful = api_result.parsing_successful
                    except Exception as e:
                        self.logger.warning(f"Failed to parse response for {custom_id}: {e}")

                batch_result = BatchResult(
                    job_id=batch_job.job_id,
                    custom_id=custom_id,
                    response=chatgpt_response,
                    error=None,
                    parsing_successful=parsing_successful,
                    parsed_content=parsed_content
                )
                batch_results.append(batch_result)

            except Exception as e:
                self.logger.error(f"Failed to process result for {custom_id}: {e}")
                batch_result = BatchResult(
                    job_id=batch_job.job_id,
                    custom_id=custom_id,
                    response=None,
                    error=str(e),
                    parsing_successful=False,
                    parsed_content=None
                )
                batch_results.append(batch_result)

        # Update actual cost
        batch_job.actual_cost = total_cost
        self.cost_tracker.add_cost(
            cost=total_cost,
            model=self.model,
            experiment_name=self.experiment_name,
        )

        self.logger.info(f"Parsed {len(batch_results)} results, total cost: ${total_cost:.4f}")
        return batch_results

    async def process_factuality_evaluation_batch(
        self,
        formatted_prompts: List[FormattedPrompt],
        task_type: str,
        dataset_name: str,
        prompt_type: str,
        wait_for_completion: bool = True,
        generation_kwargs: Dict[str, Any] = None
    ) -> Tuple[BatchJob, Optional[List[BatchResult]]]:
        """
        High-level method to process factuality evaluation as a batch.

        Args:
            formatted_prompts: List of formatted prompts
            task_type: Type of evaluation task
            dataset_name: Name of the dataset
            prompt_type: Type of prompt
            wait_for_completion: Whether to wait for completion
            generation_kwargs: Additional generation parameters

        Returns:
            Tuple of (BatchJob, parsed results if completed)
        """
        self.logger.info(f"Starting batch factuality evaluation: {task_type}/{dataset_name}/{prompt_type}")

        # Submit batch job
        batch_job = await self.submit_factuality_evaluation_batch(
            formatted_prompts=formatted_prompts,
            task_type=task_type,
            dataset_name=dataset_name,
            prompt_type=prompt_type,
            generation_kwargs=generation_kwargs
        )

        # Wait for completion if requested
        if wait_for_completion:
            self.logger.info(f"Waiting for batch job completion: {batch_job.job_id}")
            
            completed_jobs = await self.wait_for_batch_completion([batch_job])
            completed_job = completed_jobs[0]

            if completed_job.status == BatchStatus.COMPLETED:
                # Download and parse results
                results = await self.download_and_parse_results(completed_job, formatted_prompts)
                return completed_job, results
            else:
                self.logger.error(f"Batch job failed with status: {completed_job.status}")
                return completed_job, None

        return batch_job, None

    def estimate_batch_cost(
        self,
        formatted_prompts: List[FormattedPrompt],
        generation_kwargs: Dict[str, Any] = None
    ) -> Dict[str, float]:
        """
        Estimate cost for batch processing.

        Args:
            formatted_prompts: List of prompts to estimate
            generation_kwargs: Generation parameters

        Returns:
            Dictionary with cost estimates
        """
        generation_kwargs = generation_kwargs or {}
        max_tokens = generation_kwargs.get("max_tokens", self.config.get("openai.generation.max_tokens", 2048))

        total_prompt_tokens = 0
        total_completion_tokens = len(formatted_prompts) * max_tokens

        # Count tokens in prompts
        for prompt in formatted_prompts:
            tokens = len(self.token_encoder.encode(prompt.prompt_text))
            total_prompt_tokens += tokens

        # Calculate costs
        sync_cost = CostCalculator.calculate_cost(
            self.model, total_prompt_tokens, total_completion_tokens
        )
        batch_cost = sync_cost * (1 - self.cost_savings)
        savings = sync_cost - batch_cost

        return {
            "sync_cost": sync_cost,
            "batch_cost": batch_cost,
            "savings": savings,
            "savings_percentage": self.cost_savings * 100,
            "total_tokens": total_prompt_tokens + total_completion_tokens,
            "prompt_tokens": total_prompt_tokens,
            "completion_tokens": total_completion_tokens
        }

    async def get_batch_job_status(self, job_id: str) -> Optional[BatchJob]:
        """
        Get status of a specific batch job.

        Args:
            job_id: Batch job ID

        Returns:
            Updated BatchJob object
        """
        return await self.batch_manager.get_batch_status(job_id)

    async def cancel_batch_job(self, job_id: str) -> bool:
        """
        Cancel a batch job.

        Args:
            job_id: Job ID to cancel

        Returns:
            True if cancellation was successful
        """
        return await self.batch_manager.cancel_batch_job(job_id)

    def get_all_batch_jobs(self) -> Dict[str, List[BatchJob]]:
        """
        Get all tracked batch jobs.

        Returns:
            Dictionary with job lists by status
        """
        return self.batch_manager.get_all_jobs()

    def get_batch_cost_summary(self) -> Dict[str, Any]:
        """
        Get comprehensive cost analysis for batch operations.

        Returns:
            Cost summary dictionary
        """
        return self.batch_manager.get_cost_summary()

    async def process_experiment_in_batches(
        self,
        all_prompts: List[FormattedPrompt],
        task_type: str,
        datasets: List[str],
        prompt_types: List[str],
        batch_size: int = None,
        generation_kwargs: Dict[str, Any] = None
    ) -> Dict[str, Dict[str, Tuple[BatchJob, List[BatchResult]]]]:
        """
        Process an entire experiment across multiple datasets and prompt types.

        Args:
            all_prompts: All formatted prompts for the experiment
            task_type: Type of evaluation task
            datasets: List of dataset names
            prompt_types: List of prompt types
            batch_size: Maximum prompts per batch (None for no limit)
            generation_kwargs: Generation parameters

        Returns:
            Nested dictionary: {dataset: {prompt_type: (job, results)}}
        """
        batch_size = batch_size or len(all_prompts)
        results = {}

        # Group prompts by dataset and prompt type
        prompt_groups = {}
        for prompt in all_prompts:
            key = (prompt.dataset_name, prompt.prompt_type)
            if key not in prompt_groups:
                prompt_groups[key] = []
            prompt_groups[key].append(prompt)

        # Submit all batch jobs first
        submitted_jobs = {}
        for (dataset_name, prompt_type), prompts in prompt_groups.items():
            if dataset_name in datasets and prompt_type in prompt_types:
                self.logger.info(f"Submitting batch for {dataset_name}/{prompt_type}")
                
                # Split into batches if needed
                prompt_batches = [prompts[i:i+batch_size] for i in range(0, len(prompts), batch_size)]
                
                batch_jobs = []
                for batch_idx, prompt_batch in enumerate(prompt_batches):
                    suffix = f"_batch{batch_idx}" if len(prompt_batches) > 1 else ""
                    
                    batch_job = await self.submit_factuality_evaluation_batch(
                        formatted_prompts=prompt_batch,
                        task_type=task_type,
                        dataset_name=f"{dataset_name}{suffix}",
                        prompt_type=prompt_type,
                        generation_kwargs=generation_kwargs
                    )
                    batch_jobs.append((batch_job, prompt_batch))

                submitted_jobs[(dataset_name, prompt_type)] = batch_jobs

        # Wait for all jobs to complete
        all_batch_jobs = []
        job_to_info = {}
        
        for (dataset_name, prompt_type), batch_jobs in submitted_jobs.items():
            for batch_job, prompt_batch in batch_jobs:
                all_batch_jobs.append(batch_job)
                job_to_info[batch_job.job_id] = (dataset_name, prompt_type, prompt_batch)

        completed_jobs = await self.wait_for_batch_completion(all_batch_jobs, show_progress=True)

        # Process results
        for completed_job in completed_jobs:
            if completed_job.job_id in job_to_info:
                dataset_name, prompt_type, prompt_batch = job_to_info[completed_job.job_id]
                
                if dataset_name not in results:
                    results[dataset_name] = {}

                if completed_job.status == BatchStatus.COMPLETED:
                    # Download and parse results
                    batch_results = await self.download_and_parse_results(completed_job, prompt_batch)
                    
                    # Merge results if multiple batches for same dataset/prompt_type
                    if prompt_type in results[dataset_name]:
                        existing_job, existing_results = results[dataset_name][prompt_type]
                        # Combine results
                        combined_results = existing_results + batch_results
                        results[dataset_name][prompt_type] = (completed_job, combined_results)
                    else:
                        results[dataset_name][prompt_type] = (completed_job, batch_results)
                else:
                    self.logger.error(f"Batch job failed: {completed_job.job_id} (status: {completed_job.status})")
                    results[dataset_name][prompt_type] = (completed_job, [])

        return results

    def convert_batch_results_to_api_results(
        self,
        batch_results: List[BatchResult],
        formatted_prompts: List[FormattedPrompt]
    ) -> List[APICallResult]:
        """
        Convert batch results to standard API call results format.

        Args:
            batch_results: List of batch results
            formatted_prompts: Original formatted prompts

        Returns:
            List of APICallResult objects compatible with standard evaluation
        """
        api_results = []

        for batch_result in batch_results:
            if batch_result.response and batch_result.parsing_successful:
                # Create APICallResult
                api_result = APICallResult(
                    raw_response=batch_result.response,
                    parsed_content=batch_result.parsed_content,
                    task_type=batch_result.response.model,  # Use model as task type identifier
                    parsing_successful=batch_result.parsing_successful,
                    parsing_errors=None,
                    confidence_score=batch_result.parsed_content.get("confidence") if batch_result.parsed_content else None
                )
                api_results.append(api_result)
            else:
                # Create failed result
                error_response = ChatGPTResponse(
                    content=batch_result.error or "Batch processing failed",
                    model=self.model,
                    prompt_tokens=0,
                    completion_tokens=0,
                    total_tokens=0,
                    cost=0.0,
                    response_time=0.0,
                    timestamp=datetime.now().isoformat(),
                    finish_reason="error",
                    temperature=0.0,
                    max_tokens=None,
                    request_id=batch_result.custom_id
                )
                
                api_result = APICallResult(
                    raw_response=error_response,
                    parsed_content={},
                    task_type="error",
                    parsing_successful=False,
                    parsing_errors=[batch_result.error] if batch_result.error else ["Unknown error"]
                )
                api_results.append(api_result)

        return api_results

    def get_usage_statistics(self) -> Dict[str, Any]:
        """
        Get comprehensive usage statistics for batch operations.

        Returns:
            Usage statistics dictionary
        """
        all_jobs = self.batch_manager.get_all_jobs()
        cost_summary = self.batch_manager.get_cost_summary()
        
        # Calculate success metrics
        total_jobs = len(all_jobs["active"]) + len(all_jobs["completed"]) + len(all_jobs["failed"])
        success_rate = len(all_jobs["completed"]) / max(total_jobs, 1)
        
        # Calculate average processing time
        processing_times = []
        for job in all_jobs["completed"]:
            if job.started_at and job.completed_at:
                duration = (job.completed_at - job.started_at).total_seconds()
                processing_times.append(duration)
        
        avg_processing_time = sum(processing_times) / len(processing_times) if processing_times else 0

        return {
            "batch_operations": {
                "total_jobs": total_jobs,
                "completed_jobs": len(all_jobs["completed"]),
                "failed_jobs": len(all_jobs["failed"]),
                "active_jobs": len(all_jobs["active"]),
                "success_rate": success_rate,
                "average_processing_time": avg_processing_time
            },
            "cost_analysis": cost_summary,
            "standard_client_stats": self.standard_client.get_usage_statistics(),
            "batch_settings": {
                "enabled": self.enabled,
                "cost_savings": self.cost_savings,
                "processing_timeout": self.processing_timeout,
                "model": self.model
            }
        }

    async def cleanup_old_files(self, older_than_days: int = 7) -> None:
        """
        Clean up old batch files to save storage.

        Args:
            older_than_days: Remove files older than this many days
        """
        await self.batch_manager.cleanup_completed_files(older_than_days)

    def save_batch_results(
        self,
        batch_results: List[BatchResult],
        output_path: Path,
        include_metadata: bool = True
    ) -> None:
        """
        Save batch results to file.

        Args:
            batch_results: Results to save
            output_path: Path to save results
            include_metadata: Whether to include metadata
        """
        output_data = {
            "results": [result.to_dict() for result in batch_results]
        }

        if include_metadata:
            output_data["metadata"] = {
                "experiment_name": self.experiment_name,
                "total_results": len(batch_results),
                "successful_results": sum(1 for r in batch_results if r.parsing_successful),
                "failed_results": sum(1 for r in batch_results if not r.parsing_successful),
                "total_cost": sum(r.response.cost for r in batch_results if r.response),
                "timestamp": datetime.now().isoformat(),
                "model": self.model,
                "cost_savings": self.cost_savings
            }

        output_path.parent.mkdir(parents=True, exist_ok=True)
        with open(output_path, 'w') as f:
            json.dump(output_data, f, indent=2)

        self.logger.info(f"Batch results saved to: {output_path}")

    def save_detailed_experiment_data(
        self,
        batch_results: Dict[str, Dict[str, Any]],
        formatted_prompts: Dict[str, Dict[str, Dict[str, List[FormattedPrompt]]]],
        output_dir: Path,
        experiment_name: str
    ) -> None:
        """
        Save detailed experiment data including requests and responses.

        Args:
            batch_results: Nested dict of batch results by task/dataset/prompt_type
            formatted_prompts: Nested dict of original prompts
            output_dir: Directory to save detailed data
            experiment_name: Name of the experiment
        """
        self.logger.info("Saving detailed experiment data with requests and responses")
        
        # Create detailed data directory
        detailed_dir = output_dir / "detailed_data"
        detailed_dir.mkdir(parents=True, exist_ok=True)
        
        # Save comprehensive request-response data
        comprehensive_data = {
            "experiment_metadata": {
                "experiment_name": experiment_name,
                "timestamp": datetime.now().isoformat(),
                "model": self.model,
                "total_combinations": 0,
                "successful_combinations": 0,
                "total_cost": 0.0
            },
            "task_combinations": {}
        }
        
        total_cost = 0.0
        total_combinations = 0
        successful_combinations = 0
        
        # Process each combination in the flat batch_results structure
        for combination_key, result_data in batch_results.items():
            total_combinations += 1
            
            # Get job and results
            job = result_data.get('job')
            results = result_data.get('results', [])
            original_prompts = result_data.get('prompts', [])
            
            if job and results:
                successful_combinations += 1
                total_cost += job.actual_cost
                
                # Parse task/dataset/prompt_type from combination key or job
                task_type = job.task_type
                dataset_name = job.dataset_name
                prompt_type = job.prompt_type
                
                # Initialize nested structure
                if task_type not in comprehensive_data["task_combinations"]:
                    comprehensive_data["task_combinations"][task_type] = {}
                if dataset_name not in comprehensive_data["task_combinations"][task_type]:
                    comprehensive_data["task_combinations"][task_type][dataset_name] = {}
                
                # Create detailed request-response pairs
                request_response_pairs = []
                for i, (result, prompt) in enumerate(zip(results, original_prompts)):
                    pair = {
                        "request_id": f"{combination_key}_{i}",
                        "original_prompt": {
                            "task_type": prompt.task_type,
                            "dataset_name": dataset_name,  # Get from job metadata
                            "prompt_type": prompt.prompt_type,
                            "example_id": getattr(prompt, 'example_id', i),
                            "formatted_prompt": prompt.prompt_text,
                            "template_name": prompt.template_name,
                            "variables_used": prompt.variables_used
                        },
                        "batch_response": result.to_dict() if result else None,
                        "parsing_success": result.parsing_successful if result else False,
                        "parsed_content": result.parsed_content if result else None,
                        "error": result.error if result else None
                    }
                    request_response_pairs.append(pair)
                
                # Store combination data
                combination_data = {
                    "job_metadata": {
                        "job_id": job.job_id,
                        "status": job.status.value,
                        "estimated_cost": job.estimated_cost,
                        "actual_cost": job.actual_cost,
                        "created_at": job.created_at.isoformat() if job.created_at else None,
                        "completed_at": job.completed_at.isoformat() if job.completed_at else None
                    },
                    "statistics": {
                        "total_requests": len(results),
                        "successful_responses": sum(1 for r in results if r and r.parsing_successful),
                        "failed_responses": sum(1 for r in results if r and not r.parsing_successful),
                        "total_cost": job.actual_cost
                    },
                    "request_response_pairs": request_response_pairs
                }
                
                comprehensive_data["task_combinations"][task_type][dataset_name][prompt_type] = combination_data
                
                # Save individual combination file for easy access
                combo_file = detailed_dir / f"{combination_key}_detailed.json"
                with open(combo_file, 'w') as f:
                    json.dump(combination_data, f, indent=2, default=str)
        
        # Update metadata
        comprehensive_data["experiment_metadata"]["total_combinations"] = total_combinations
        comprehensive_data["experiment_metadata"]["successful_combinations"] = successful_combinations
        comprehensive_data["experiment_metadata"]["total_cost"] = total_cost
        
        # Save comprehensive data file
        comprehensive_file = detailed_dir / f"{experiment_name}_comprehensive_data.json"
        with open(comprehensive_file, 'w') as f:
            json.dump(comprehensive_data, f, indent=2, default=str)
            
        # Save summary file
        summary_file = detailed_dir / f"{experiment_name}_data_summary.json"
        summary_data = {
            "experiment_name": experiment_name,
            "timestamp": datetime.now().isoformat(),
            "total_combinations": total_combinations,
            "successful_combinations": successful_combinations,
            "success_rate": successful_combinations / max(total_combinations, 1),
            "total_cost": total_cost,
            "model": self.model,
            "detailed_files_created": {
                "comprehensive_data": str(comprehensive_file.name),
                "individual_combinations": [f"{task}_{dataset}_{prompt}_detailed.json" 
                                          for task in comprehensive_data["task_combinations"]
                                          for dataset in comprehensive_data["task_combinations"][task]
                                          for prompt in comprehensive_data["task_combinations"][task][dataset]]
            }
        }
        
        with open(summary_file, 'w') as f:
            json.dump(summary_data, f, indent=2)
        
        self.logger.info(f"Detailed experiment data saved to {detailed_dir}")
        self.logger.info(f"Created {len(summary_data['detailed_files_created']['individual_combinations'])} individual combination files")
        self.logger.info(f"Comprehensive data: {comprehensive_file}")
        self.logger.info(f"Summary: {summary_file}")