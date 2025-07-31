"""
Batch Processing Package for ChatGPT Factuality Evaluation
=========================================================

Comprehensive batch processing system for cost-effective large-scale
factuality evaluation experiments using OpenAI's Batch API.

This package provides:
- BatchManager: Core batch job orchestration
- BatchMonitor: Real-time job monitoring and progress tracking
- OpenAIBatchClient: Specialized client for batch operations

Author: Michael Ogunjimi
Institution: University of Manchester
Course: MSc AI
"""

from .batch_manager import (
    BatchManager,
    BatchJob,
    BatchJobRequest,
    BatchStatus
)

from .batch_monitor import (
    BatchMonitor,
    MonitoringStats
)

__all__ = [
    "BatchManager",
    "BatchJob", 
    "BatchJobRequest",
    "BatchStatus",
    "BatchMonitor",
    "MonitoringStats"
]

__version__ = "1.0.0"
__author__ = "Michael Ogunjimi"
__email__ = "michael.ogunjimi@postgrad.manchester.ac.uk"
__institution__ = "University of Manchester"
__course__ = "MSc AI"

# Package-level configuration
DEFAULT_BATCH_CONFIG = {
    "max_queue_size": 1000000,
    "processing_timeout": 86400,  # 24 hours
    "cost_savings": 0.5,  # 50% savings vs sync
    "update_interval": 60,  # Status check interval
    "cleanup_after_days": 7
}

def get_batch_package_info() -> dict:
    """Get batch package information."""
    return {
        "name": "ChatGPT Factuality Evaluation - Batch Processing",
        "version": __version__,
        "author": __author__,
        "institution": __institution__,
        "course": __course__,
        "description": "Batch processing system for cost-effective factuality evaluation",
        "components": [
            "BatchManager - Core batch job orchestration",
            "BatchMonitor - Real-time monitoring and progress tracking", 
            "Integration with OpenAI Batch API for cost optimization"
        ]
    }