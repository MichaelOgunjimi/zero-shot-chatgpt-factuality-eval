"""
Comprehensive Logging System for ChatGPT Factuality Evaluation
============================================================

Advanced logging system with experiment tracking, API monitoring,
cost analysis, and thesis-ready output formatting.

This module provides academic-quality logging infrastructure including
structured logging, experiment tracking, progress monitoring, and
cost analysis for OpenAI API usage.

Author: Michael Ogunjimi
Institution: University of Manchester
Course: MSc AI
"""

import functools
import json
import logging
import logging.handlers
import time
from collections import defaultdict
from contextlib import contextmanager
from dataclasses import dataclass, asdict
from datetime import datetime
from pathlib import Path
from typing import Dict, Any, Optional, Union, Callable, List

import numpy as np
from tqdm import tqdm

logger = logging.getLogger(__name__)


@dataclass
class LogEntry:
    """Structured log entry for experiment tracking."""

    timestamp: str
    level: str
    message: str
    experiment_name: Optional[str] = None
    task_name: Optional[str] = None
    example_id: Optional[str] = None
    metric_name: Optional[str] = None
    cost: Optional[float] = None
    duration: Optional[float] = None
    metadata: Optional[Dict[str, Any]] = None

    def to_dict(self) -> Dict[str, Any]:
        """Convert to dictionary for JSON serialization."""
        return asdict(self)


class StructuredFormatter(logging.Formatter):
    """JSON formatter for structured logging."""

    def format(self, record) -> str:
        """Format log record as JSON."""
        log_entry = LogEntry(
            timestamp=datetime.fromtimestamp(record.created).isoformat(),
            level=record.levelname,
            message=record.getMessage(),
            experiment_name=getattr(record, 'experiment_name', None),
            task_name=getattr(record, 'task_name', None),
            example_id=getattr(record, 'example_id', None),
            metric_name=getattr(record, 'metric_name', None),
            cost=getattr(record, 'cost', None),
            duration=getattr(record, 'duration', None),
            metadata=getattr(record, 'metadata', None)
        )

        return json.dumps(log_entry.to_dict(), default=str)


class AcademicFormatter(logging.Formatter):
    """Academic-style formatter for console output."""

    def __init__(self):
        super().__init__(
            fmt='%(asctime)s | %(name)-20s | %(levelname)-8s | %(message)s',
            datefmt='%Y-%m-%d %H:%M:%S'
        )

    def format(self, record) -> str:
        """Format with academic styling."""
        # Add experiment context if available
        if hasattr(record, 'experiment_name') and record.experiment_name:
            record.name = f"{record.name}[{record.experiment_name}]"

        formatted = super().format(record)

        # Add cost information if available
        if hasattr(record, 'cost') and record.cost:
            formatted += f" | Cost: ${record.cost:.4f}"

        # Add duration if available
        if hasattr(record, 'duration') and record.duration:
            formatted += f" | Duration: {record.duration:.2f}s"

        return formatted


class ProgressTracker:
    """
    Advanced progress tracker for long-running experiments.

    Provides detailed progress monitoring with cost tracking,
    ETA estimation, and experiment-specific logging.
    """

    def __init__(
        self,
        total: int,
        description: str = "Processing",
        experiment_name: Optional[str] = None,
        show_cost: bool = True,
        update_frequency: int = 10
    ):
        """
        Initialize progress tracker.

        Args:
            total: Total number of items to process
            description: Description of the task
            experiment_name: Name of the experiment
            show_cost: Whether to show cost information
            update_frequency: How often to update progress (every N items)
        """
        self.total = total
        self.description = description
        self.experiment_name = experiment_name
        self.show_cost = show_cost
        self.update_frequency = update_frequency

        # Progress tracking
        self.current = 0
        self.start_time = time.time()
        self.last_update_time = self.start_time
        self.costs = []
        self.durations = []

        # Setup progress bar
        self.pbar = tqdm(
            total=total,
            desc=description,
            unit="items",
            bar_format='{l_bar}{bar}| {n_fmt}/{total_fmt} [{elapsed}<{remaining}, {rate_fmt}]'
        )

        # Setup logger
        self.logger = get_logger(f"progress.{experiment_name or 'default'}")
        self.logger.info(f"Started progress tracking: {description} ({total} items)")

    def update(self, cost: Optional[float] = None, metadata: Optional[Dict] = None) -> None:
        """
        Update progress with optional cost and metadata.

        Args:
            cost: Cost for this iteration
            metadata: Additional metadata
        """
        self.current += 1
        current_time = time.time()
        duration = current_time - self.last_update_time

        # Track costs and durations
        if cost is not None:
            self.costs.append(cost)
        self.durations.append(duration)

        # Update progress bar
        self.pbar.update(1)

        # Add cost to progress bar if available
        if self.show_cost and self.costs:
            total_cost = sum(self.costs)
            avg_cost = total_cost / len(self.costs)
            self.pbar.set_postfix({
                'Total Cost': f"${total_cost:.4f}",
                'Avg Cost': f"${avg_cost:.4f}"
            })

        # Log progress at intervals
        if self.current % self.update_frequency == 0 or self.current == self.total:
            self._log_progress(cost, metadata, duration)

        self.last_update_time = current_time

    def _log_progress(self, cost: Optional[float], metadata: Optional[Dict], duration: float) -> None:
        """Log detailed progress information."""
        progress_pct = (self.current / self.total) * 100
        elapsed = time.time() - self.start_time

        # Calculate statistics
        avg_duration = np.mean(self.durations) if self.durations else 0
        total_cost = sum(self.costs) if self.costs else 0
        avg_cost = total_cost / len(self.costs) if self.costs else 0

        # Estimate remaining time
        if self.current > 0:
            estimated_total_time = elapsed * (self.total / self.current)
            eta = estimated_total_time - elapsed
        else:
            eta = 0

        progress_message = (
            f"Progress: {self.current}/{self.total} ({progress_pct:.1f}%) | "
            f"Elapsed: {elapsed:.1f}s | ETA: {eta:.1f}s"
        )

        if self.costs:
            progress_message += f" | Total Cost: ${total_cost:.4f} | Avg Cost: ${avg_cost:.4f}"

        # Log with structured data
        extra_data = {
            'experiment_name': self.experiment_name,
            'progress_current': self.current,
            'progress_total': self.total,
            'progress_percentage': progress_pct,
            'elapsed_time': elapsed,
            'eta_seconds': eta,
            'average_duration': avg_duration,
            'cost': cost,
            'total_cost': total_cost,
            'average_cost': avg_cost,
            'metadata': metadata
        }

        self.logger.info(progress_message, extra=extra_data)

    def finish(self) -> Dict[str, Any]:
        """
        Finish progress tracking and return summary.

        Returns:
            Summary statistics
        """
        self.pbar.close()
        total_time = time.time() - self.start_time

        summary = {
            'description': self.description,
            'experiment_name': self.experiment_name,
            'total_items': self.total,
            'total_time': total_time,
            'average_time_per_item': total_time / self.total if self.total > 0 else 0,
            'total_cost': sum(self.costs) if self.costs else 0,
            'average_cost_per_item': sum(self.costs) / len(self.costs) if self.costs else 0,
            'completed_at': datetime.now().isoformat()
        }

        self.logger.info(f"Progress tracking completed: {self.description}", extra=summary)
        return summary


class CostTracker:
    """
    Comprehensive cost tracking for API usage.

    Tracks costs across different models, tasks, and experiments
    with budget monitoring and alerts.
    """

    def __init__(
        self,
        daily_budget: float = 50.0,
        total_budget: float = 200.0,
        warning_threshold: float = 0.8
    ):
        """
        Initialize cost tracker.

        Args:
            daily_budget: Maximum daily spending
            total_budget: Maximum total spending
            warning_threshold: Fraction of budget to trigger warnings
        """
        self.daily_budget = daily_budget
        self.total_budget = total_budget
        self.warning_threshold = warning_threshold

        # Cost tracking
        self.total_spent = 0.0
        self.daily_spent = 0.0
        self.last_reset_date = datetime.now().date()

        # Detailed tracking
        self.cost_history = []
        self.costs_by_model = defaultdict(float)
        self.costs_by_experiment = defaultdict(float)
        self.costs_by_task = defaultdict(float)

        # Setup logger
        self.logger = get_logger("cost_tracker")
        self.logger.info(f"Cost tracker initialized: Daily=${daily_budget}, Total=${total_budget}")

    def add_cost(
        self,
        cost: float,
        model: str,
        experiment_name: Optional[str] = None,
        task_name: Optional[str] = None,
        metadata: Optional[Dict] = None
    ) -> None:
        """
        Add a cost entry.

        Args:
            cost: Cost amount
            model: Model name
            experiment_name: Experiment name
            task_name: Task name
            metadata: Additional metadata
        """
        # Reset daily costs if new day
        current_date = datetime.now().date()
        if current_date > self.last_reset_date:
            self.daily_spent = 0.0
            self.last_reset_date = current_date
            self.logger.info("Daily cost counter reset")

        # Add to totals
        self.total_spent += cost
        self.daily_spent += cost

        # Track by categories
        self.costs_by_model[model] += cost
        if experiment_name:
            self.costs_by_experiment[experiment_name] += cost
        if task_name:
            self.costs_by_task[task_name] += cost

        # Record in history
        cost_entry = {
            'timestamp': datetime.now().isoformat(),
            'cost': cost,
            'model': model,
            'experiment_name': experiment_name,
            'task_name': task_name,
            'metadata': metadata,
            'running_total': self.total_spent,
            'daily_total': self.daily_spent
        }
        self.cost_history.append(cost_entry)

        # Check budgets and log
        self._check_budgets()

        # Log cost entry
        self.logger.info(
            f"Cost added: ${cost:.4f} ({model})",
            extra={
                'cost': cost,
                'model': model,
                'experiment_name': experiment_name,
                'task_name': task_name,
                'running_total': self.total_spent,
                'daily_total': self.daily_spent,
                'metadata': metadata
            }
        )

    def _check_budgets(self) -> None:
        """Check budget limits and issue warnings."""
        # Total budget warning
        if self.total_spent >= self.total_budget * self.warning_threshold:
            self.logger.warning(
                f"Budget warning: ${self.total_spent:.2f} / ${self.total_budget:.2f} spent "
                f"({(self.total_spent/self.total_budget)*100:.1f}%)"
            )

        # Daily budget warning
        if self.daily_spent >= self.daily_budget * self.warning_threshold:
            self.logger.warning(
                f"Daily budget warning: ${self.daily_spent:.2f} / ${self.daily_budget:.2f} spent "
                f"({(self.daily_spent/self.daily_budget)*100:.1f}%)"
            )

        # Budget exceeded errors
        if self.total_spent >= self.total_budget:
            self.logger.error(f"Total budget exceeded: ${self.total_spent:.2f} / ${self.total_budget:.2f}")

        if self.daily_spent >= self.daily_budget:
            self.logger.error(f"Daily budget exceeded: ${self.daily_spent:.2f} / ${self.daily_budget:.2f}")

    def get_analysis(self) -> Dict[str, Any]:
        """Get comprehensive cost analysis."""
        return {
            'summary': {
                'total_spent': self.total_spent,
                'daily_spent': self.daily_spent,
                'total_budget': self.total_budget,
                'daily_budget': self.daily_budget,
                'budget_utilization': self.total_spent / self.total_budget,
                'daily_utilization': self.daily_spent / self.daily_budget,
                'remaining_budget': max(0, self.total_budget - self.total_spent),
                'remaining_daily_budget': max(0, self.daily_budget - self.daily_spent)
            },
            'breakdown': {
                'by_model': dict(self.costs_by_model),
                'by_experiment': dict(self.costs_by_experiment),
                'by_task': dict(self.costs_by_task)
            },
            'history': self.cost_history[-50:]  # Last 50 entries
        }

    def can_afford(self, estimated_cost: float) -> bool:
        """
        Check if an estimated cost is within budget.

        Args:
            estimated_cost: Estimated cost for operation

        Returns:
            True if operation is within budget
        """
        within_total = (self.total_spent + estimated_cost) <= self.total_budget
        within_daily = (self.daily_spent + estimated_cost) <= self.daily_budget

        return within_total and within_daily


class ExperimentLogger:
    """
    Comprehensive experiment logger with structured output.

    Provides experiment-specific logging with automatic cost tracking,
    progress monitoring, and structured output for thesis analysis.
    """

    def __init__(
        self,
        experiment_name: str,
        config: Optional[Dict[str, Any]] = None,
        log_dir: Optional[str] = None
    ):
        """
        Initialize experiment logger.

        Args:
            experiment_name: Name of the experiment
            config: Configuration dictionary
            log_dir: Directory for log files (overrides default path)
        """
        self.experiment_name = experiment_name
        self.config = config or {}
        
        # Use provided log_dir or create experiment-specific log directory
        if log_dir:
            self.log_dir = Path(log_dir)
        else:
            experiment_base_dir = Path(f"results/experiments/{experiment_name}")
            self.log_dir = experiment_base_dir / "logs"
        
        self.log_dir.mkdir(parents=True, exist_ok=True)

        # Initialize cost tracker
        cost_config = self.config.get('cost_control', {})
        self.cost_tracker = CostTracker(
            daily_budget=cost_config.get('daily_budget', 50.0),
            total_budget=cost_config.get('total_budget', 200.0),
            warning_threshold=cost_config.get('warning_threshold', 0.8)
        )

        # Setup loggers
        self._setup_loggers()

        # Experiment tracking
        self.start_time = datetime.now()
        self.task_timings = {}
        self.task_costs = {}
        self.experiment_metadata = {
            'experiment_name': experiment_name,
            'start_time': self.start_time.isoformat(),
            'config': config
        }

        self.logger.info(f"Experiment logger initialized: {experiment_name}")

    def _setup_loggers(self) -> None:
        """Setup structured and console loggers."""
        # Main experiment logger
        self.logger = logging.getLogger(f"experiment.{self.experiment_name}")
        self.logger.setLevel(logging.INFO)

        # Clear existing handlers
        self.logger.handlers.clear()

        # Console handler with academic formatting
        console_handler = logging.StreamHandler()
        console_handler.setFormatter(AcademicFormatter())
        self.logger.addHandler(console_handler)

        # Structured JSON log file
        json_log_file = self.log_dir / f"{self.experiment_name}_structured.jsonl"
        json_handler = logging.FileHandler(json_log_file)
        json_handler.setFormatter(StructuredFormatter())
        self.logger.addHandler(json_handler)

        # Human-readable log file
        text_log_file = self.log_dir / f"{self.experiment_name}.log"
        text_handler = logging.FileHandler(text_log_file)
        text_handler.setFormatter(AcademicFormatter())
        self.logger.addHandler(text_handler)

        # Rotating handler for large experiments
        rotating_file = self.log_dir / f"{self.experiment_name}_rotating.log"
        rotating_handler = logging.handlers.RotatingFileHandler(
            rotating_file, maxBytes=50*1024*1024, backupCount=5
        )
        rotating_handler.setFormatter(AcademicFormatter())
        self.logger.addHandler(rotating_handler)

    def log_task_start(self, task_name: str, metadata: Optional[Dict] = None) -> None:
        """Log the start of a task."""
        start_time = datetime.now()
        self.task_timings[task_name] = {'start': start_time, 'end': None}

        self.logger.info(
            f"Task started: {task_name}",
            extra={
                'experiment_name': self.experiment_name,
                'task_name': task_name,
                'metadata': metadata
            }
        )

    def log_task_end(self, task_name: str, metadata: Optional[Dict] = None) -> float:
        """
        Log the end of a task and return duration.

        Args:
            task_name: Name of the task
            metadata: Additional metadata

        Returns:
            Task duration in seconds
        """
        end_time = datetime.now()

        if task_name in self.task_timings:
            self.task_timings[task_name]['end'] = end_time
            duration = (end_time - self.task_timings[task_name]['start']).total_seconds()
        else:
            duration = 0.0

        # Get task cost if available
        task_cost = self.task_costs.get(task_name, 0.0)

        self.logger.info(
            f"Task completed: {task_name} (Duration: {duration:.2f}s, Cost: ${task_cost:.4f})",
            extra={
                'experiment_name': self.experiment_name,
                'task_name': task_name,
                'duration': duration,
                'cost': task_cost,
                'metadata': metadata
            }
        )

        return duration

    def log_cost(
        self,
        cost: float,
        model: str,
        task_name: Optional[str] = None,
        metadata: Optional[Dict] = None
    ) -> None:
        """Log cost information."""
        self.cost_tracker.add_cost(
            cost=cost,
            model=model,
            experiment_name=self.experiment_name,
            task_name=task_name,
            metadata=metadata
        )

        # Add to task costs
        if task_name:
            self.task_costs[task_name] = self.task_costs.get(task_name, 0.0) + cost

    def create_progress_tracker(
        self,
        total: int,
        description: str,
        show_cost: bool = True
    ) -> ProgressTracker:
        """Create a progress tracker for this experiment."""
        return ProgressTracker(
            total=total,
            description=description,
            experiment_name=self.experiment_name,
            show_cost=show_cost
        )

    def finalize_experiment(self) -> Dict[str, Any]:
        """
        Finalize experiment and return summary.

        Returns:
            Experiment summary
        """
        end_time = datetime.now()
        total_duration = (end_time - self.start_time).total_seconds()

        # Calculate task statistics
        task_durations = {}
        for task_name, timing in self.task_timings.items():
            if timing['end']:
                duration = (timing['end'] - timing['start']).total_seconds()
                task_durations[task_name] = duration

        # Get cost analysis
        cost_analysis = self.cost_tracker.get_analysis()

        experiment_summary = {
            'experiment_name': self.experiment_name,
            'start_time': self.start_time.isoformat(),
            'end_time': end_time.isoformat(),
            'total_duration': total_duration,
            'task_durations': task_durations,
            'task_costs': dict(self.task_costs),
            'cost_analysis': cost_analysis,
            'metadata': self.experiment_metadata
        }

        # Save summary
        summary_file = self.log_dir / f"{self.experiment_name}_summary.json"
        with open(summary_file, 'w') as f:
            json.dump(experiment_summary, f, indent=2, default=str)

        self.logger.info(
            f"Experiment finalized: {self.experiment_name} "
            f"(Duration: {total_duration:.2f}s, Total Cost: ${cost_analysis['summary']['total_spent']:.4f})",
            extra=experiment_summary
        )

        return experiment_summary


# Convenience functions

def setup_experiment_logger(experiment_name: str, config: Dict[str, Any], log_dir: Optional[str] = None) -> ExperimentLogger:
    """
    Setup experiment logger with configuration.

    Args:
        experiment_name: Name of the experiment
        config: Configuration dictionary
        log_dir: Custom log directory path (overrides config default)

    Returns:
        Configured ExperimentLogger instance
    """
    if log_dir is None:
        log_dir = config.get('paths', {}).get('logs_dir', './logs')
    return ExperimentLogger(experiment_name, config, log_dir)


def setup_basic_logging(config: Dict[str, Any]) -> logging.Logger:
    """
    Setup basic logging configuration.

    Args:
        config: Configuration dictionary

    Returns:
        Configured logger
    """
    logging_config = config.get("logging", {})
    log_level = logging_config.get("level", "INFO")

    logging.basicConfig(
        level=getattr(logging, log_level.upper()),
        format='%(asctime)s - %(name)s - %(levelname)s - %(message)s',
        handlers=[logging.StreamHandler()]
    )

    return logging.getLogger("chatgpt_evaluation")


def get_logger(name: str) -> logging.Logger:
    """Get a logger with the specified name."""
    return logging.getLogger(name)


def log_execution_time(func: Callable) -> Callable:
    """
    Decorator to log function execution time.

    Args:
        func: Function to decorate

    Returns:
        Decorated function
    """
    @functools.wraps(func)
    def wrapper(*args, **kwargs):
        start_time = time.time()
        logger = get_logger(func.__module__)

        try:
            result = func(*args, **kwargs)
            duration = time.time() - start_time

            logger.info(
                f"Function {func.__name__} completed in {duration:.2f}s",
                extra={'duration': duration, 'function_name': func.__name__}
            )

            return result
        except Exception as e:
            duration = time.time() - start_time
            logger.error(
                f"Function {func.__name__} failed after {duration:.2f}s: {e}",
                extra={'duration': duration, 'function_name': func.__name__, 'error': str(e)}
            )
            raise

    return wrapper


def log_model_loading(model_name: str, device: str, model_size: Optional[str] = None) -> None:
    """
    Log model loading information.

    Args:
        model_name: Name of the model
        device: Device the model is on
        model_size: Optional model size description
    """
    logger = get_logger(__name__)
    message = f"Model loaded: {model_name} on {device}"
    if model_size:
        message += f" ({model_size})"

    logger.info(
        message,
        extra={
            'model_name': model_name,
            'model_device': device,
            'model_size': model_size
        }
    )


def log_evaluation_results(metric_name: str, results: Dict[str, float]) -> None:
    """
    Log evaluation results.

    Args:
        metric_name: Name of the metric
        results: Dictionary of result names to values
    """
    logger = get_logger(__name__)

    for result_name, value in results.items():
        logger.info(
            f"{metric_name} - {result_name}: {value:.4f}",
            extra={
                'metric_name': metric_name,
                'result_name': result_name,
                'result_value': value
            }
        )


@contextmanager
def log_task_context(task_name: str, experiment_logger: Optional[ExperimentLogger] = None):
    """
    Context manager for logging task execution.

    Args:
        task_name: Name of the task
        experiment_logger: Optional experiment logger instance
    """
    if experiment_logger:
        experiment_logger.log_task_start(task_name)

    start_time = time.time()

    try:
        yield
        duration = time.time() - start_time

        if experiment_logger:
            experiment_logger.log_task_end(task_name)
        else:
            logger = get_logger(__name__)
            logger.info(f"Task {task_name} completed in {duration:.2f}s")

    except Exception as e:
        duration = time.time() - start_time

        if experiment_logger:
            experiment_logger.log_task_end(task_name, metadata={'error': str(e)})
        else:
            logger = get_logger(__name__)
            logger.error(f"Task {task_name} failed after {duration:.2f}s: {e}")

        raise