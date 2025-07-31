"""
Batch Manager for OpenAI Batch API Processing
=============================================

Comprehensive batch job management system for factuality evaluation experiments.
Handles batch submission, monitoring, cost optimization, and result processing.

Author: Michael Ogunjimi
Institution: University of Manchester
Course: MSc AI
"""

import asyncio
import json
import os
import time
from datetime import datetime, timedelta
from pathlib import Path
from typing import Dict, List, Any, Optional, Tuple
from dataclasses import dataclass, asdict
from enum import Enum

import openai
from tenacity import retry, stop_after_attempt, wait_exponential, retry_if_exception_type

from ..utils.config import get_config
from ..utils.logging import get_logger
from ..prompts.prompt_manager import FormattedPrompt


class BatchStatus(Enum):
    """Batch job status enumeration."""
    VALIDATING = "validating"
    IN_PROGRESS = "in_progress"
    FINALIZING = "finalizing"
    COMPLETED = "completed"
    FAILED = "failed"
    EXPIRED = "expired"
    CANCELLING = "cancelling"
    CANCELLED = "cancelled"


@dataclass
class BatchJobRequest:
    """Individual batch request structure."""
    custom_id: str
    method: str
    url: str
    body: Dict[str, Any]

    def to_dict(self) -> Dict[str, Any]:
        """Convert to OpenAI batch format."""
        return {
            "custom_id": self.custom_id,
            "method": self.method,
            "url": self.url,
            "body": self.body
        }


@dataclass
class BatchJob:
    """Batch job metadata and tracking."""
    job_id: str
    experiment_name: str
    task_type: str
    dataset_name: str
    prompt_type: str
    total_requests: int
    status: BatchStatus
    created_at: datetime
    started_at: Optional[datetime] = None
    completed_at: Optional[datetime] = None
    input_file_id: Optional[str] = None
    output_file_id: Optional[str] = None
    error_file_id: Optional[str] = None
    estimated_cost: float = 0.0
    actual_cost: float = 0.0
    metadata: Optional[Dict[str, Any]] = None

    def to_dict(self) -> Dict[str, Any]:
        """Convert to dictionary for storage."""
        return {
            "job_id": self.job_id,
            "experiment_name": self.experiment_name,
            "task_type": self.task_type,
            "dataset_name": self.dataset_name,
            "prompt_type": self.prompt_type,
            "total_requests": self.total_requests,
            "status": self.status.value,
            "created_at": self.created_at.isoformat(),
            "started_at": self.started_at.isoformat() if self.started_at else None,
            "completed_at": self.completed_at.isoformat() if self.completed_at else None,
            "input_file_id": self.input_file_id,
            "output_file_id": self.output_file_id,
            "error_file_id": self.error_file_id,
            "estimated_cost": self.estimated_cost,
            "actual_cost": self.actual_cost,
            "metadata": self.metadata or {}
        }

    @classmethod
    def from_dict(cls, data: Dict[str, Any]) -> 'BatchJob':
        """Create BatchJob from dictionary."""
        return cls(
            job_id=data["job_id"],
            experiment_name=data["experiment_name"],
            task_type=data["task_type"],
            dataset_name=data["dataset_name"],
            prompt_type=data["prompt_type"],
            total_requests=data["total_requests"],
            status=BatchStatus(data["status"]),
            created_at=datetime.fromisoformat(data["created_at"]),
            started_at=datetime.fromisoformat(data["started_at"]) if data.get("started_at") else None,
            completed_at=datetime.fromisoformat(data["completed_at"]) if data.get("completed_at") else None,
            input_file_id=data.get("input_file_id"),
            output_file_id=data.get("output_file_id"),
            error_file_id=data.get("error_file_id"),
            estimated_cost=data.get("estimated_cost", 0.0),
            actual_cost=data.get("actual_cost", 0.0),
            metadata=data.get("metadata", {})
        )


class BatchManager:
    """
    Manages OpenAI batch API operations for factuality evaluation experiments.
    
    Provides comprehensive batch job management including submission, monitoring,
    cost optimization, and result processing specifically designed for academic
    research requirements.
    """

    def __init__(self, config=None, experiment_name: str = None):
        """
        Initialize batch manager.

        Args:
            config: Configuration object
            experiment_name: Name of the current experiment
        """
        self.config = config or get_config()
        self.experiment_name = experiment_name or f"batch_experiment_{datetime.now().strftime('%Y%m%d_%H%M%S')}"
        
        # Setup OpenAI client
        api_key = os.getenv("OPENAI_API_KEY")
        if not api_key:
            raise ValueError("OPENAI_API_KEY environment variable not set")
        
        self.openai_client = openai.AsyncOpenAI(api_key=api_key)
        
        # Setup logging
        self.logger = get_logger(f"{self.__class__.__name__}")
        
        # Batch configuration
        batch_config = self.config.get("openai.batch", {})
        self.max_queue_size = batch_config.get("max_queue_size", 1000000)
        self.processing_timeout = batch_config.get("processing_timeout", 86400)  # 24 hours
        self.cost_savings = batch_config.get("cost_savings", 0.5)  # 50% savings
        
        # Model and pricing info
        self.model = self.config.get("openai.models.primary", "gpt-4.1-mini")
        cost_config = self.config.get("openai.cost_control.cost_per_1k_tokens", {})
        model_costs = cost_config.get(self.model, {"input": 0.00015, "output": 0.0006})
        self.input_cost_per_token = model_costs["input"] / 1000
        self.output_cost_per_token = model_costs["output"] / 1000
        
        # Storage
        self.storage_dir = Path(f"results/experiments/batch_processing")
        self.storage_dir.mkdir(parents=True, exist_ok=True)
        
        # Job tracking
        self.active_jobs: Dict[str, BatchJob] = {}
        self.completed_jobs: Dict[str, BatchJob] = {}
        self.failed_jobs: Dict[str, BatchJob] = {}
        
        # Load existing jobs
        self._load_job_states()
        
        self.logger.info(f"BatchManager initialized for experiment: {self.experiment_name}")

    def _load_job_states(self) -> None:
        """Load existing job states from storage."""
        try:
            # Load active jobs
            active_file = self.storage_dir / "batch_monitor_status" / "active_jobs.json"
            if active_file.exists():
                with open(active_file, 'r') as f:
                    active_data = json.load(f)
                    for job_data in active_data:
                        job = BatchJob.from_dict(job_data)
                        self.active_jobs[job.job_id] = job

            # Load completed jobs
            completed_file = self.storage_dir / "batch_monitor_status" / "completed_jobs.json"
            if completed_file.exists():
                with open(completed_file, 'r') as f:
                    completed_data = json.load(f)
                    for job_data in completed_data:
                        job = BatchJob.from_dict(job_data)
                        self.completed_jobs[job.job_id] = job

            # Load failed jobs
            failed_file = self.storage_dir / "batch_monitor_status" / "failed_jobs.json"
            if failed_file.exists():
                with open(failed_file, 'r') as f:
                    failed_data = json.load(f)
                    for job_data in failed_data:
                        job = BatchJob.from_dict(job_data)
                        self.failed_jobs[job.job_id] = job

            self.logger.info(f"Loaded {len(self.active_jobs)} active, {len(self.completed_jobs)} completed, {len(self.failed_jobs)} failed jobs")

        except Exception as e:
            self.logger.warning(f"Could not load existing job states: {e}")

    def _save_job_states(self) -> None:
        """Save current job states to storage."""
        try:
            monitor_dir = self.storage_dir / "batch_monitor_status"
            monitor_dir.mkdir(parents=True, exist_ok=True)

            # Save active jobs
            with open(monitor_dir / "active_jobs.json", 'w') as f:
                json.dump([job.to_dict() for job in self.active_jobs.values()], f, indent=2)

            # Save completed jobs
            with open(monitor_dir / "completed_jobs.json", 'w') as f:
                json.dump([job.to_dict() for job in self.completed_jobs.values()], f, indent=2)

            # Save failed jobs
            with open(monitor_dir / "failed_jobs.json", 'w') as f:
                json.dump([job.to_dict() for job in self.failed_jobs.values()], f, indent=2)

            # Save cost tracking
            total_cost = sum(job.actual_cost for job in self.completed_jobs.values())
            estimated_cost = sum(job.estimated_cost for job in self.active_jobs.values())
            
            cost_data = {
                "total_completed_cost": total_cost,
                "estimated_pending_cost": estimated_cost,
                "total_estimated_cost": total_cost + estimated_cost,
                "cost_savings_vs_sync": total_cost * self.cost_savings,
                "last_updated": datetime.now().isoformat()
            }
            
            with open(monitor_dir / "cost_tracking.json", 'w') as f:
                json.dump(cost_data, f, indent=2)

        except Exception as e:
            self.logger.error(f"Failed to save job states: {e}")

    def estimate_cost(self, requests: List[BatchJobRequest]) -> float:
        """
        Estimate cost for batch requests.

        Args:
            requests: List of batch requests

        Returns:
            Estimated cost in dollars
        """
        total_cost = 0.0
        
        for request in requests:
            # Estimate tokens for the request
            prompt_text = json.dumps(request.body.get("messages", []))
            estimated_prompt_tokens = len(prompt_text) // 4  # Rough estimate
            estimated_completion_tokens = request.body.get("max_tokens", 2048)
            
            # Calculate cost
            prompt_cost = estimated_prompt_tokens * self.input_cost_per_token
            completion_cost = estimated_completion_tokens * self.output_cost_per_token
            total_cost += (prompt_cost + completion_cost) * (1 - self.cost_savings)
        
        return total_cost

    async def create_batch_input_file(self, requests: List[BatchJobRequest]) -> str:
        """
        Create batch input file and upload to OpenAI.

        Args:
            requests: List of batch requests

        Returns:
            File ID for the uploaded batch input
        """
        # Create JSONL content
        jsonl_content = "\n".join(json.dumps(req.to_dict()) for req in requests)
        
        # Upload file
        response = await self.openai_client.files.create(
            file=jsonl_content.encode('utf-8'),
            purpose="batch"
        )
        
        self.logger.info(f"Created batch input file: {response.id}")
        return response.id

    @retry(
        stop=stop_after_attempt(3),
        wait=wait_exponential(multiplier=1, min=4, max=10),
        retry=retry_if_exception_type(openai.APIError)
    )
    async def submit_batch_job(
        self,
        requests: List[BatchJobRequest],
        task_type: str,
        dataset_name: str,
        prompt_type: str,
        description: str = None
    ) -> BatchJob:
        """
        Submit a batch job to OpenAI.

        Args:
            requests: List of batch requests
            task_type: Type of evaluation task
            dataset_name: Name of the dataset
            prompt_type: Type of prompt (zero_shot/chain_of_thought)
            description: Optional job description

        Returns:
            BatchJob object with job metadata
        """
        self.logger.info(f"Submitting batch job with {len(requests)} requests")
        
        # Check queue limits
        if len(requests) > self.max_queue_size:
            raise ValueError(f"Batch size {len(requests)} exceeds maximum queue size {self.max_queue_size}")
        
        # Estimate cost
        estimated_cost = self.estimate_cost(requests)
        
        # Create input file
        input_file_id = await self.create_batch_input_file(requests)
        
        # Submit batch job
        batch_response = await self.openai_client.batches.create(
            input_file_id=input_file_id,
            endpoint="/v1/chat/completions",
            completion_window="24h",
            metadata={
                "experiment_name": self.experiment_name,
                "task_type": task_type,
                "dataset_name": dataset_name,
                "prompt_type": prompt_type,
                "description": description or f"{task_type}_{dataset_name}_{prompt_type}"
            }
        )
        
        # Create BatchJob object
        batch_job = BatchJob(
            job_id=batch_response.id,
            experiment_name=self.experiment_name,
            task_type=task_type,
            dataset_name=dataset_name,
            prompt_type=prompt_type,
            total_requests=len(requests),
            status=BatchStatus(batch_response.status),
            created_at=datetime.now(),
            input_file_id=input_file_id,
            estimated_cost=estimated_cost,
            metadata={
                "description": description,
                "model": self.model,
                "endpoint": "/v1/chat/completions",
                "completion_window": "24h"
            }
        )
        
        # Add to active jobs
        self.active_jobs[batch_job.job_id] = batch_job
        self._save_job_states()
        
        self.logger.info(f"Batch job submitted: {batch_job.job_id}")
        return batch_job

    async def get_batch_status(self, job_id: str) -> BatchJob:
        """
        Get current status of a batch job.

        Args:
            job_id: Batch job ID

        Returns:
            Updated BatchJob object
        """
        try:
            # Check if job is already completed/failed first
            if job_id in self.completed_jobs:
                return self.completed_jobs[job_id]
            if job_id in self.failed_jobs:
                return self.failed_jobs[job_id]
            
            batch_response = await self.openai_client.batches.retrieve(job_id)
            
            # Update job status
            if job_id in self.active_jobs:
                job = self.active_jobs[job_id]
                job.status = BatchStatus(batch_response.status)
                
                # Update timestamps
                if batch_response.in_progress_at and not job.started_at:
                    job.started_at = datetime.fromtimestamp(batch_response.in_progress_at)
                
                if batch_response.completed_at and not job.completed_at:
                    job.completed_at = datetime.fromtimestamp(batch_response.completed_at)
                
                # Update file IDs
                if batch_response.output_file_id:
                    job.output_file_id = batch_response.output_file_id
                if batch_response.error_file_id:
                    job.error_file_id = batch_response.error_file_id
                
                # Move completed/failed jobs
                if job.status in [BatchStatus.COMPLETED, BatchStatus.FAILED, BatchStatus.EXPIRED, BatchStatus.CANCELLED]:
                    if job.status == BatchStatus.COMPLETED:
                        self.completed_jobs[job_id] = job
                    else:
                        self.failed_jobs[job_id] = job
                    del self.active_jobs[job_id]
                    self._save_job_states()
                
                return job
            else:
                self.logger.warning(f"Job {job_id} not found in any job collection")
                return None

        except Exception as e:
            self.logger.error(f"Failed to get batch status for {job_id}: {e}")
            return None

    async def download_batch_results(self, job: BatchJob) -> List[Dict[str, Any]]:
        """
        Download and parse batch job results.

        Args:
            job: Completed BatchJob

        Returns:
            List of parsed results
        """
        if not job.output_file_id:
            raise ValueError(f"No output file available for job {job.job_id}")
        
        try:
            # Download output file
            file_response = await self.openai_client.files.content(job.output_file_id)
            content = file_response.content.decode('utf-8')
            
            # Parse JSONL results
            results = []
            for line in content.strip().split('\n'):
                if line.strip():
                    result = json.loads(line)
                    results.append(result)
            
            self.logger.info(f"Downloaded {len(results)} results for job {job.job_id}")
            return results
            
        except Exception as e:
            self.logger.error(f"Failed to download results for job {job.job_id}: {e}")
            raise

    async def download_batch_errors(self, job: BatchJob) -> List[Dict[str, Any]]:
        """
        Download and parse batch job errors.

        Args:
            job: BatchJob with errors

        Returns:
            List of error records
        """
        if not job.error_file_id:
            return []
        
        try:
            file_response = await self.openai_client.files.content(job.error_file_id)
            content = file_response.content.decode('utf-8')
            
            errors = []
            for line in content.strip().split('\n'):
                if line.strip():
                    error = json.loads(line)
                    errors.append(error)
            
            self.logger.info(f"Downloaded {len(errors)} errors for job {job.job_id}")
            return errors
            
        except Exception as e:
            self.logger.warning(f"Failed to download errors for job {job.job_id}: {e}")
            return []

    async def wait_for_completion(
        self,
        job_ids: List[str],
        check_interval: int = 60,
        timeout: int = None
    ) -> Dict[str, BatchJob]:
        """
        Wait for multiple batch jobs to complete.

        Args:
            job_ids: List of job IDs to wait for
            check_interval: Seconds between status checks
            timeout: Maximum wait time in seconds

        Returns:
            Dictionary mapping job IDs to final BatchJob objects
        """
        timeout = timeout or self.processing_timeout
        start_time = time.time()
        
        self.logger.info(f"Waiting for {len(job_ids)} batch jobs to complete")
        
        while True:
            # Check all jobs
            all_completed = True
            current_jobs = {}
            
            for job_id in job_ids:
                job = await self.get_batch_status(job_id)
                if job:
                    current_jobs[job_id] = job
                    if job.status not in [BatchStatus.COMPLETED, BatchStatus.FAILED, BatchStatus.EXPIRED, BatchStatus.CANCELLED]:
                        all_completed = False
                else:
                    all_completed = False
            
            # Check if all completed
            if all_completed:
                self.logger.info("All batch jobs completed")
                break
            
            # Check timeout
            elapsed = time.time() - start_time
            if elapsed > timeout:
                self.logger.warning(f"Batch jobs timed out after {elapsed:.1f} seconds")
                break
            
            # Log progress
            completed_count = sum(1 for job in current_jobs.values() 
                                if job.status in [BatchStatus.COMPLETED, BatchStatus.FAILED, BatchStatus.EXPIRED, BatchStatus.CANCELLED])
            self.logger.info(f"Batch progress: {completed_count}/{len(job_ids)} completed ({elapsed:.1f}s elapsed)")
            
            # Wait before next check
            await asyncio.sleep(check_interval)
        
        return current_jobs

    async def cancel_batch_job(self, job_id: str) -> bool:
        """
        Cancel a batch job.

        Args:
            job_id: Job ID to cancel

        Returns:
            True if cancellation was successful
        """
        try:
            await self.openai_client.batches.cancel(job_id)
            
            # Update job status
            if job_id in self.active_jobs:
                job = self.active_jobs[job_id]
                job.status = BatchStatus.CANCELLING
                self._save_job_states()
            
            self.logger.info(f"Cancellation requested for job {job_id}")
            return True
            
        except Exception as e:
            self.logger.error(f"Failed to cancel job {job_id}: {e}")
            return False

    def get_all_jobs(self) -> Dict[str, List[BatchJob]]:
        """
        Get all tracked jobs by status.

        Returns:
            Dictionary with job lists by status
        """
        return {
            "active": list(self.active_jobs.values()),
            "completed": list(self.completed_jobs.values()),
            "failed": list(self.failed_jobs.values())
        }

    def get_cost_summary(self) -> Dict[str, Any]:
        """
        Get comprehensive cost analysis.

        Returns:
            Cost summary dictionary
        """
        total_actual_cost = sum(job.actual_cost for job in self.completed_jobs.values())
        total_estimated_cost = sum(job.estimated_cost for job in self.active_jobs.values())
        
        # Calculate what sync cost would have been
        total_sync_cost = total_actual_cost / (1 - self.cost_savings)
        savings = total_sync_cost - total_actual_cost
        
        return {
            "total_actual_cost": total_actual_cost,
            "total_estimated_pending_cost": total_estimated_cost,
            "total_estimated_cost": total_actual_cost + total_estimated_cost,
            "estimated_sync_cost": total_sync_cost,
            "total_savings": savings,
            "savings_percentage": self.cost_savings * 100,
            "completed_jobs": len(self.completed_jobs),
            "active_jobs": len(self.active_jobs),
            "failed_jobs": len(self.failed_jobs),
            "last_updated": datetime.now().isoformat()
        }

    async def cleanup_completed_files(self, older_than_days: int = 7) -> None:
        """
        Clean up old batch files to save storage.

        Args:
            older_than_days: Remove files older than this many days
        """
        cutoff_date = datetime.now() - timedelta(days=older_than_days)
        deleted_count = 0
        
        for job in self.completed_jobs.values():
            if job.completed_at and job.completed_at < cutoff_date:
                try:
                    # Delete input file
                    if job.input_file_id:
                        await self.openai_client.files.delete(job.input_file_id)
                    
                    # Delete output file
                    if job.output_file_id:
                        await self.openai_client.files.delete(job.output_file_id)
                    
                    # Delete error file
                    if job.error_file_id:
                        await self.openai_client.files.delete(job.error_file_id)
                    
                    deleted_count += 1
                    
                except Exception as e:
                    self.logger.warning(f"Failed to delete files for job {job.job_id}: {e}")
        
        self.logger.info(f"Cleaned up files for {deleted_count} completed jobs")

    def create_job_requests_from_prompts(
        self,
        formatted_prompts: List[FormattedPrompt],
        generation_kwargs: Dict[str, Any] = None
    ) -> List[BatchJobRequest]:
        """
        Convert formatted prompts to batch job requests.

        Args:
            formatted_prompts: List of formatted prompts
            generation_kwargs: Generation parameters

        Returns:
            List of batch job requests
        """
        generation_kwargs = generation_kwargs or {}
        
        # Default generation parameters
        default_params = {
            "model": self.model,
            "temperature": self.config.get("openai.generation.temperature", 0.0),
            "max_tokens": self.config.get("openai.generation.max_tokens", 2048),
            "top_p": self.config.get("openai.generation.top_p", 1.0)
        }
        default_params.update(generation_kwargs)
        
        requests = []
        for i, prompt in enumerate(formatted_prompts):
            request = BatchJobRequest(
                custom_id=f"{prompt.task_type}_{prompt.dataset_name}_{i}_{int(time.time())}",
                method="POST",
                url="/v1/chat/completions",
                body={
                    "messages": [{"role": "user", "content": prompt.formatted_text}],
                    **default_params
                }
            )
            requests.append(request)
        
        return requests

    async def process_experiment_batch(
        self,
        formatted_prompts: List[FormattedPrompt],
        task_type: str,
        dataset_name: str,
        prompt_type: str,
        description: str = None,
        generation_kwargs: Dict[str, Any] = None
    ) -> BatchJob:
        """
        High-level method to process an entire experiment as a batch.

        Args:
            formatted_prompts: List of formatted prompts
            task_type: Type of evaluation task
            dataset_name: Name of the dataset
            prompt_type: Type of prompt
            description: Optional description
            generation_kwargs: Generation parameters

        Returns:
            Submitted BatchJob
        """
        self.logger.info(f"Processing experiment batch: {task_type}/{dataset_name}/{prompt_type}")
        
        # Create batch requests
        requests = self.create_job_requests_from_prompts(formatted_prompts, generation_kwargs)
        
        # Submit batch job
        batch_job = await self.submit_batch_job(
            requests=requests,
            task_type=task_type,
            dataset_name=dataset_name,
            prompt_type=prompt_type,
            description=description
        )
        
        self.logger.info(f"Experiment batch submitted: {batch_job.job_id}")
        return batch_job