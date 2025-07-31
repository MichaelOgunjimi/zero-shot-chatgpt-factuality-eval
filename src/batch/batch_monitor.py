"""
Batch Monitor for OpenAI Batch API Status Tracking
=================================================

Real-time monitoring and status tracking for batch jobs with detailed
progress reporting, cost analysis, and health monitoring.

Author: Michael Ogunjimi
Institution: University of Manchester
Course: MSc AI
"""

import asyncio
import json
import time
from datetime import datetime, timedelta
from pathlib import Path
from typing import Dict, List, Any, Optional, Callable
from dataclasses import dataclass

import openai
from rich.console import Console
from rich.table import Table
from rich.progress import Progress, TaskID, SpinnerColumn, TextColumn, BarColumn, TimeElapsedColumn
from rich.live import Live
from rich.panel import Panel
from rich.layout import Layout

from .batch_manager import BatchManager, BatchJob, BatchStatus
from ..utils.logging import get_logger


@dataclass
class MonitoringStats:
    """Batch monitoring statistics."""
    total_jobs: int
    active_jobs: int
    completed_jobs: int
    failed_jobs: int
    total_requests: int
    completed_requests: int
    failed_requests: int
    total_estimated_cost: float
    total_actual_cost: float
    estimated_savings: float
    average_processing_time: float
    last_updated: datetime

    def to_dict(self) -> Dict[str, Any]:
        """Convert to dictionary."""
        return {
            "total_jobs": self.total_jobs,
            "active_jobs": self.active_jobs,
            "completed_jobs": self.completed_jobs,
            "failed_jobs": self.failed_jobs,
            "total_requests": self.total_requests,
            "completed_requests": self.completed_requests,
            "failed_requests": self.failed_requests,
            "total_estimated_cost": self.total_estimated_cost,
            "total_actual_cost": self.total_actual_cost,
            "estimated_savings": self.estimated_savings,
            "average_processing_time": self.average_processing_time,
            "last_updated": self.last_updated.isoformat()
        }


class BatchMonitor:
    """
    Comprehensive batch job monitoring system.
    
    Provides real-time status tracking, progress reporting, cost analysis,
    and health monitoring for OpenAI batch jobs in academic research context.
    """

    def __init__(self, batch_manager: BatchManager, update_interval: int = 30):
        """
        Initialize batch monitor.

        Args:
            batch_manager: BatchManager instance to monitor
            update_interval: Seconds between status updates
        """
        self.batch_manager = batch_manager
        self.update_interval = update_interval
        
        # Setup logging
        self.logger = get_logger(f"{self.__class__.__name__}")
        
        # Rich console for pretty output
        self.console = Console()
        
        # Monitoring state
        self.monitoring_active = False
        self.monitor_task = None
        self.stats_history: List[MonitoringStats] = []
        
        # Callbacks
        self.status_callbacks: List[Callable] = []
        self.completion_callbacks: List[Callable] = []
        
        self.logger.info("BatchMonitor initialized")

    def add_status_callback(self, callback: Callable[[MonitoringStats], None]) -> None:
        """Add callback for status updates."""
        self.status_callbacks.append(callback)

    def add_completion_callback(self, callback: Callable[[BatchJob], None]) -> None:
        """Add callback for job completion."""
        self.completion_callbacks.append(callback)

    async def start_monitoring(self, job_ids: List[str] = None) -> None:
        """
        Start monitoring batch jobs.

        Args:
            job_ids: Specific job IDs to monitor (None for all active jobs)
        """
        if self.monitoring_active:
            self.logger.warning("Monitoring already active")
            return

        self.monitoring_active = True
        self.monitor_task = asyncio.create_task(self._monitoring_loop(job_ids))
        self.logger.info("Batch monitoring started")

    async def stop_monitoring(self) -> None:
        """Stop monitoring batch jobs."""
        if not self.monitoring_active:
            return

        self.monitoring_active = False
        if self.monitor_task:
            self.monitor_task.cancel()
            try:
                await self.monitor_task
            except asyncio.CancelledError:
                pass

        self.logger.info("Batch monitoring stopped")

    async def _monitoring_loop(self, job_ids: List[str] = None) -> None:
        """Main monitoring loop."""
        try:
            while self.monitoring_active:
                # Update job statuses
                if job_ids:
                    for job_id in job_ids:
                        await self.batch_manager.get_batch_status(job_id)
                else:
                    # Update all active jobs
                    for job_id in list(self.batch_manager.active_jobs.keys()):
                        await self.batch_manager.get_batch_status(job_id)

                # Calculate statistics
                stats = self._calculate_stats()
                self.stats_history.append(stats)
                
                # Trigger callbacks
                for callback in self.status_callbacks:
                    try:
                        callback(stats)
                    except Exception as e:
                        self.logger.warning(f"Status callback failed: {e}")

                # Check for newly completed jobs
                await self._check_completions()

                # Wait for next update
                await asyncio.sleep(self.update_interval)

        except asyncio.CancelledError:
            self.logger.info("Monitoring loop cancelled")
        except Exception as e:
            self.logger.error(f"Monitoring loop failed: {e}")

    def _calculate_stats(self) -> MonitoringStats:
        """Calculate current monitoring statistics."""
        all_jobs = self.batch_manager.get_all_jobs()
        
        total_jobs = len(all_jobs["active"]) + len(all_jobs["completed"]) + len(all_jobs["failed"])
        total_requests = sum(job.total_requests for jobs in all_jobs.values() for job in jobs)
        completed_requests = sum(job.total_requests for job in all_jobs["completed"])
        failed_requests = sum(job.total_requests for job in all_jobs["failed"])
        
        total_estimated_cost = sum(job.estimated_cost for job in all_jobs["active"])
        total_actual_cost = sum(job.actual_cost for job in all_jobs["completed"])
        
        # Calculate average processing time for completed jobs
        processing_times = []
        for job in all_jobs["completed"]:
            if job.started_at and job.completed_at:
                duration = (job.completed_at - job.started_at).total_seconds()
                processing_times.append(duration)
        
        avg_processing_time = sum(processing_times) / len(processing_times) if processing_times else 0
        
        # Calculate estimated savings
        estimated_sync_cost = total_actual_cost / (1 - self.batch_manager.cost_savings)
        estimated_savings = estimated_sync_cost - total_actual_cost

        return MonitoringStats(
            total_jobs=total_jobs,
            active_jobs=len(all_jobs["active"]),
            completed_jobs=len(all_jobs["completed"]),
            failed_jobs=len(all_jobs["failed"]),
            total_requests=total_requests,
            completed_requests=completed_requests,
            failed_requests=failed_requests,
            total_estimated_cost=total_estimated_cost,
            total_actual_cost=total_actual_cost,
            estimated_savings=estimated_savings,
            average_processing_time=avg_processing_time,
            last_updated=datetime.now()
        )

    async def _check_completions(self) -> None:
        """Check for newly completed jobs and trigger callbacks."""
        for job in self.batch_manager.completed_jobs.values():
            for callback in self.completion_callbacks:
                try:
                    callback(job)
                except Exception as e:
                    self.logger.warning(f"Completion callback failed: {e}")

    def display_status_table(self) -> None:
        """Display current batch status in a formatted table."""
        all_jobs = self.batch_manager.get_all_jobs()
        
        table = Table(title="Batch Job Status")
        table.add_column("Job ID", style="cyan")
        table.add_column("Experiment", style="green")
        table.add_column("Task", style="yellow")
        table.add_column("Dataset", style="blue")
        table.add_column("Status", style="magenta")
        table.add_column("Requests", justify="right")
        table.add_column("Est. Cost", justify="right")
        table.add_column("Created", style="dim")

        # Add active jobs
        for job in all_jobs["active"]:
            table.add_row(
                job.job_id[:8] + "...",
                job.experiment_name[:15] + "...",
                job.task_type,
                job.dataset_name,
                f"🟡 {job.status.value}",
                str(job.total_requests),
                f"${job.estimated_cost:.4f}",
                job.created_at.strftime("%H:%M")
            )

        # Add completed jobs
        for job in all_jobs["completed"]:
            table.add_row(
                job.job_id[:8] + "...",
                job.experiment_name[:15] + "...",
                job.task_type,
                job.dataset_name,
                f"🟢 {job.status.value}",
                str(job.total_requests),
                f"${job.actual_cost:.4f}",
                job.created_at.strftime("%H:%M")
            )

        # Add failed jobs
        for job in all_jobs["failed"]:
            table.add_row(
                job.job_id[:8] + "...",
                job.experiment_name[:15] + "...",
                job.task_type,
                job.dataset_name,
                f"🔴 {job.status.value}",
                str(job.total_requests),
                f"${job.estimated_cost:.4f}",
                job.created_at.strftime("%H:%M")
            )

        self.console.print(table)

    def display_cost_summary(self) -> None:
        """Display cost analysis summary."""
        cost_summary = self.batch_manager.get_cost_summary()
        
        cost_table = Table(title="Cost Analysis")
        cost_table.add_column("Metric", style="cyan")
        cost_table.add_column("Value", justify="right", style="green")

        cost_table.add_row("Completed Cost", f"${cost_summary['total_actual_cost']:.4f}")
        cost_table.add_row("Pending Est. Cost", f"${cost_summary['total_estimated_pending_cost']:.4f}")
        cost_table.add_row("Total Est. Cost", f"${cost_summary['total_estimated_cost']:.4f}")
        cost_table.add_row("Est. Sync Cost", f"${cost_summary['estimated_sync_cost']:.4f}")
        cost_table.add_row("Total Savings", f"${cost_summary['total_savings']:.4f}")
        cost_table.add_row("Savings %", f"{cost_summary['savings_percentage']:.1f}%")

        self.console.print(cost_table)

    async def monitor_with_live_display(
        self,
        job_ids: List[str] = None,
        refresh_interval: int = 30
    ) -> None:
        """
        Monitor batch jobs with live updating display.

        Args:
            job_ids: Specific job IDs to monitor
            refresh_interval: Display refresh interval in seconds
        """
        self.logger.info("Starting live monitoring display")
        
        layout = Layout()
        layout.split_column(
            Layout(name="status", size=10),
            Layout(name="cost", size=8),
            Layout(name="progress", size=5)
        )

        async def update_display():
            # Update job statuses
            if job_ids:
                for job_id in job_ids:
                    await self.batch_manager.get_batch_status(job_id)
            else:
                for job_id in list(self.batch_manager.active_jobs.keys()):
                    await self.batch_manager.get_batch_status(job_id)

            # Create status table
            all_jobs = self.batch_manager.get_all_jobs()
            status_table = Table(title="Batch Job Status")
            status_table.add_column("Job ID", style="cyan")
            status_table.add_column("Task", style="yellow")
            status_table.add_column("Status", style="magenta")
            status_table.add_column("Requests", justify="right")
            status_table.add_column("Duration", justify="right")

            for job in all_jobs["active"]:
                duration = (datetime.now() - job.created_at).total_seconds() / 60
                status_table.add_row(
                    job.job_id[:12] + "...",
                    f"{job.task_type}/{job.dataset_name}",
                    f"🟡 {job.status.value}",
                    str(job.total_requests),
                    f"{duration:.1f}m"
                )

            layout["status"].update(Panel(status_table, title="Active Jobs"))

            # Create cost summary
            cost_summary = self.batch_manager.get_cost_summary()
            cost_text = f"""
Total Actual Cost: ${cost_summary['total_actual_cost']:.4f}
Pending Est. Cost: ${cost_summary['total_estimated_pending_cost']:.4f}
Total Savings: ${cost_summary['total_savings']:.4f} ({cost_summary['savings_percentage']:.1f}%)
            """.strip()
            layout["cost"].update(Panel(cost_text, title="Cost Analysis"))

            # Create progress summary
            stats = self._calculate_stats()
            progress_text = f"""
Jobs: {stats.completed_jobs}/{stats.total_jobs} completed
Requests: {stats.completed_requests}/{stats.total_requests} processed
Success Rate: {(stats.completed_requests / max(stats.total_requests, 1)) * 100:.1f}%
            """.strip()
            layout["progress"].update(Panel(progress_text, title="Progress"))

            return layout

        # Live monitoring loop
        try:
            with Live(await update_display(), refresh_per_second=1/refresh_interval) as live:
                while any(job.status not in [BatchStatus.COMPLETED, BatchStatus.FAILED, BatchStatus.EXPIRED, BatchStatus.CANCELLED] 
                         for job in self.batch_manager.active_jobs.values()):
                    
                    await asyncio.sleep(refresh_interval)
                    live.update(await update_display())

            self.logger.info("Live monitoring completed")

        except KeyboardInterrupt:
            self.logger.info("Live monitoring interrupted by user")
        except Exception as e:
            self.logger.error(f"Live monitoring failed: {e}")

    async def wait_for_all_completion(
        self,
        job_ids: List[str] = None,
        timeout: int = 86400,
        show_progress: bool = True
    ) -> Dict[str, BatchJob]:
        """
        Wait for all specified jobs to complete with progress display.

        Args:
            job_ids: Job IDs to wait for (None for all active)
            timeout: Maximum wait time in seconds
            show_progress: Whether to show progress bar

        Returns:
            Dictionary of completed jobs
        """
        if job_ids is None:
            job_ids = list(self.batch_manager.active_jobs.keys())

        if not job_ids:
            self.logger.info("No jobs to monitor")
            return {}

        self.logger.info(f"Waiting for {len(job_ids)} batch jobs to complete")
        
        start_time = time.time()
        completed_jobs = {}

        if show_progress:
            with Progress(
                SpinnerColumn(),
                TextColumn("[progress.description]{task.description}"),
                BarColumn(),
                TextColumn("[progress.percentage]{task.percentage:>3.0f}%"),
                TimeElapsedColumn(),
                console=self.console
            ) as progress:
                
                task = progress.add_task("Waiting for batch jobs...", total=len(job_ids))
                
                while True:
                    # Update job statuses
                    current_completed = 0
                    for job_id in job_ids:
                        job = await self.batch_manager.get_batch_status(job_id)
                        if job and job.status in [BatchStatus.COMPLETED, BatchStatus.FAILED, BatchStatus.EXPIRED, BatchStatus.CANCELLED]:
                            completed_jobs[job_id] = job
                            current_completed += 1

                    # Update progress
                    progress.update(task, completed=current_completed)
                    
                    # Check if all completed
                    if current_completed >= len(job_ids):
                        break
                    
                    # Check timeout
                    elapsed = time.time() - start_time
                    if elapsed > timeout:
                        self.logger.warning(f"Timeout reached after {elapsed:.1f} seconds")
                        break
                    
                    # Update description with current status
                    remaining = len(job_ids) - current_completed
                    progress.update(task, description=f"Waiting for {remaining} batch jobs...")
                    
                    await asyncio.sleep(self.update_interval)

        else:
            # Simple wait without progress bar
            while True:
                current_completed = 0
                for job_id in job_ids:
                    job = await self.batch_manager.get_batch_status(job_id)
                    if job and job.status in [BatchStatus.COMPLETED, BatchStatus.FAILED, BatchStatus.EXPIRED, BatchStatus.CANCELLED]:
                        completed_jobs[job_id] = job
                        current_completed += 1

                if current_completed >= len(job_ids):
                    break
                
                elapsed = time.time() - start_time
                if elapsed > timeout:
                    self.logger.warning(f"Timeout reached after {elapsed:.1f} seconds")
                    break
                
                if elapsed % 300 == 0:  # Log every 5 minutes
                    remaining = len(job_ids) - current_completed
                    self.logger.info(f"Still waiting for {remaining} jobs ({elapsed:.1f}s elapsed)")
                
                await asyncio.sleep(self.update_interval)

        # Trigger completion callbacks
        for job in completed_jobs.values():
            for callback in self.completion_callbacks:
                try:
                    callback(job)
                except Exception as e:
                    self.logger.warning(f"Completion callback failed: {e}")

        self.logger.info(f"Completed monitoring {len(completed_jobs)} jobs")
        return completed_jobs

    def generate_monitoring_report(self, output_path: Path = None) -> str:
        """
        Generate comprehensive monitoring report.

        Args:
            output_path: Optional path to save report

        Returns:
            Report content as string
        """
        all_jobs = self.batch_manager.get_all_jobs()
        cost_summary = self.batch_manager.get_cost_summary()
        current_stats = self._calculate_stats()

        report = f"""# Batch Processing Monitoring Report

**Generated**: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}  
**Experiment**: {self.batch_manager.experiment_name}  
**Author**: Michael Ogunjimi  
**Institution**: University of Manchester, MSc AI  

## Summary Statistics

- **Total Jobs**: {current_stats.total_jobs}
- **Active Jobs**: {current_stats.active_jobs}
- **Completed Jobs**: {current_stats.completed_jobs}
- **Failed Jobs**: {current_stats.failed_jobs}
- **Total Requests**: {current_stats.total_requests}
- **Completed Requests**: {current_stats.completed_requests}
- **Success Rate**: {(current_stats.completed_requests / max(current_stats.total_requests, 1)) * 100:.2f}%

## Cost Analysis

- **Total Actual Cost**: ${current_stats.total_actual_cost:.4f}
- **Estimated Pending Cost**: ${current_stats.total_estimated_cost:.4f}
- **Estimated Savings**: ${current_stats.estimated_savings:.4f}
- **Savings Percentage**: {self.batch_manager.cost_savings * 100:.1f}%

## Job Details

### Active Jobs
"""

        if all_jobs["active"]:
            for job in all_jobs["active"]:
                duration = (datetime.now() - job.created_at).total_seconds() / 60
                report += f"""
- **{job.job_id}**
  - Task: {job.task_type}/{job.dataset_name}
  - Status: {job.status.value}
  - Requests: {job.total_requests}
  - Duration: {duration:.1f} minutes
  - Est. Cost: ${job.estimated_cost:.4f}
"""
        else:
            report += "\nNo active jobs.\n"

        report += "\n### Completed Jobs\n"
        
        if all_jobs["completed"]:
            for job in all_jobs["completed"]:
                processing_time = 0
                if job.started_at and job.completed_at:
                    processing_time = (job.completed_at - job.started_at).total_seconds() / 60
                
                report += f"""
- **{job.job_id}**
  - Task: {job.task_type}/{job.dataset_name}
  - Requests: {job.total_requests}
  - Processing Time: {processing_time:.1f} minutes
  - Actual Cost: ${job.actual_cost:.4f}
"""
        else:
            report += "\nNo completed jobs.\n"

        report += "\n### Failed Jobs\n"
        
        if all_jobs["failed"]:
            for job in all_jobs["failed"]:
                report += f"""
- **{job.job_id}**
  - Task: {job.task_type}/{job.dataset_name}
  - Status: {job.status.value}
  - Requests: {job.total_requests}
  - Est. Cost: ${job.estimated_cost:.4f}
"""
        else:
            report += "\nNo failed jobs.\n"

        # Performance trends
        if len(self.stats_history) > 1:
            report += "\n## Performance Trends\n"
            
            first_stat = self.stats_history[0]
            last_stat = self.stats_history[-1]
            
            completed_rate = (last_stat.completed_jobs - first_stat.completed_jobs) / max(len(self.stats_history) * self.update_interval / 3600, 1)
            
            report += f"""
- **Job Completion Rate**: {completed_rate:.2f} jobs/hour
- **Average Processing Time**: {current_stats.average_processing_time / 60:.1f} minutes
- **Monitoring Duration**: {len(self.stats_history) * self.update_interval / 60:.1f} minutes
"""

        report += f"""
## Technical Details

- **Model**: {self.batch_manager.model}
- **Cost Savings**: {self.batch_manager.cost_savings * 100:.1f}%
- **Max Queue Size**: {self.batch_manager.max_queue_size:,}
- **Processing Timeout**: {self.batch_manager.processing_timeout / 3600:.1f} hours
- **Update Interval**: {self.update_interval} seconds

---
*Report generated by BatchMonitor*
"""

        # Save report if path specified
        if output_path:
            output_path.parent.mkdir(parents=True, exist_ok=True)
            with open(output_path, 'w') as f:
                f.write(report)
            self.logger.info(f"Monitoring report saved to: {output_path}")

        return report

    async def health_check(self) -> Dict[str, Any]:
        """
        Perform health check on batch processing system.

        Returns:
            Health status dictionary
        """
        health_status = {
            "timestamp": datetime.now().isoformat(),
            "batch_manager_healthy": True,
            "openai_api_accessible": False,
            "job_states_loadable": True,
            "active_jobs_count": len(self.batch_manager.active_jobs),
            "issues": []
        }

        try:
            # Test OpenAI API access
            await self.batch_manager.openai_client.models.list()
            health_status["openai_api_accessible"] = True
        except Exception as e:
            health_status["openai_api_accessible"] = False
            health_status["issues"].append(f"OpenAI API not accessible: {e}")

        # Check for stuck jobs
        stuck_jobs = []
        cutoff_time = datetime.now() - timedelta(hours=25)  # 1 hour past 24h window
        
        for job in self.batch_manager.active_jobs.values():
            if job.created_at < cutoff_time:
                stuck_jobs.append(job.job_id)

        if stuck_jobs:
            health_status["issues"].append(f"Potentially stuck jobs: {stuck_jobs}")

        # Check queue usage
        total_pending_requests = sum(job.total_requests for job in self.batch_manager.active_jobs.values())
        queue_usage = total_pending_requests / self.batch_manager.max_queue_size
        
        if queue_usage > 0.9:
            health_status["issues"].append(f"Queue usage high: {queue_usage * 100:.1f}%")

        health_status["overall_healthy"] = len(health_status["issues"]) == 0

        return health_status

    def export_job_data(self, output_path: Path) -> None:
        """
        Export all job data for analysis.

        Args:
            output_path: Path to save job data
        """
        all_jobs = self.batch_manager.get_all_jobs()
        cost_summary = self.batch_manager.get_cost_summary()
        
        export_data = {
            "export_metadata": {
                "timestamp": datetime.now().isoformat(),
                "experiment_name": self.batch_manager.experiment_name,
                "total_jobs": len(all_jobs["active"]) + len(all_jobs["completed"]) + len(all_jobs["failed"])
            },
            "jobs": {
                "active": [job.to_dict() for job in all_jobs["active"]],
                "completed": [job.to_dict() for job in all_jobs["completed"]],
                "failed": [job.to_dict() for job in all_jobs["failed"]]
            },
            "cost_summary": cost_summary,
            "monitoring_stats": [stats.to_dict() for stats in self.stats_history]
        }

        output_path.parent.mkdir(parents=True, exist_ok=True)
        with open(output_path, 'w') as f:
            json.dump(export_data, f, indent=2)

        self.logger.info(f"Job data exported to: {output_path}")