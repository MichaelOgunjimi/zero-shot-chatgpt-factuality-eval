#!/usr/bin/env python3
"""
Batch Processing Monitor
=======================

Monitor running OpenAI batch jobs for factuality evaluation experiments.
This script helps you track the status of your batch processing jobs in real-time.

Usage:
    # Monitor all active batches
    python scripts/monitor_batches.py
    
    # Monitor specific batch by ID
    python scripts/monitor_batches.py --batch-id batch_abc123
    
    # Continuous monitoring with auto-refresh
    python scripts/monitor_batches.py --watch --interval 60

Author: Michael Ogunjimi
Institution: University of Manchester
Course: MSc AI
"""

import argparse
import asyncio
import json
import sys
import time
from datetime import datetime, timedelta
from pathlib import Path
from typing import Dict, List, Any, Optional

# Add project root to path
sys.path.insert(0, str(Path(__file__).parent.parent))

try:
    from src.utils.config import get_config
    from src.llm_clients.openai_client_batch import OpenAIBatchClient
    from src.batch.batch_monitor import BatchMonitor
except ImportError as e:
    print(f"Error importing required modules: {e}")
    print("Please ensure you're running from the project root directory.")
    sys.exit(1)


class BatchStatusMonitor:
    """Monitor and display batch processing status."""
    
    def __init__(self, config_path: Optional[str] = None):
        """Initialize the batch monitor."""
        self.config = get_config(config_path)
        self.batch_client = OpenAIBatchClient(self.config)
        self.batch_monitor = BatchMonitor(self.config)
        
    async def list_all_batches(self) -> List[Dict[str, Any]]:
        """List all batch jobs."""
        try:
            # Get batches from OpenAI API
            batches = await self.batch_client.list_batches()
            return batches
        except Exception as e:
            print(f"Error listing batches: {e}")
            return []
    
    async def get_batch_details(self, batch_id: str) -> Optional[Dict[str, Any]]:
        """Get detailed information about a specific batch."""
        try:
            batch_info = await self.batch_client.get_batch_status(batch_id)
            return batch_info
        except Exception as e:
            print(f"Error getting batch details for {batch_id}: {e}")
            return None
    
    def display_batch_summary(self, batches: List[Dict[str, Any]]) -> None:
        """Display a summary of all batches."""
        if not batches:
            print("No batch jobs found.")
            return
        
        print(f"\n{'='*80}")
        print(f"BATCH PROCESSING MONITOR - {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
        print(f"{'='*80}")
        
        # Group batches by status
        status_groups = {}
        for batch in batches:
            status = batch.get('status', 'unknown')
            if status not in status_groups:
                status_groups[status] = []
            status_groups[status].append(batch)
        
        # Display summary by status
        print(f"\nBatch Summary:")
        for status, batch_list in status_groups.items():
            print(f"  {status.upper()}: {len(batch_list)} batches")
        
        print(f"\nDetailed Batch Information:")
        print(f"{'Batch ID':<20} {'Status':<12} {'Created':<20} {'Progress':<10} {'Cost':<8}")
        print(f"{'-'*80}")
        
        for batch in batches:
            batch_id = batch.get('id', 'unknown')[:18]
            status = batch.get('status', 'unknown')
            created_at = batch.get('created_at', 0)
            created_str = datetime.fromtimestamp(created_at).strftime('%Y-%m-%d %H:%M') if created_at else 'unknown'
            
            # Calculate progress
            total_requests = batch.get('request_counts', {}).get('total', 0)
            completed_requests = batch.get('request_counts', {}).get('completed', 0)
            progress = f"{completed_requests}/{total_requests}" if total_requests > 0 else "N/A"
            
            # Estimate cost (rough approximation)
            estimated_cost = self._estimate_batch_cost(batch)
            cost_str = f"${estimated_cost:.2f}" if estimated_cost > 0 else "N/A"
            
            print(f"{batch_id:<20} {status:<12} {created_str:<20} {progress:<10} {cost_str:<8}")
        
        # Display experiment-specific information
        self._display_experiment_batches(batches)
    
    def _estimate_batch_cost(self, batch: Dict[str, Any]) -> float:
        """Estimate batch cost based on request counts and model."""
        request_counts = batch.get('request_counts', {})
        completed = request_counts.get('completed', 0)
        
        # Rough cost estimation (this would need to be more sophisticated in practice)
        # Assuming gpt-4o-mini at ~$0.0001 per request (very rough estimate)
        estimated_cost_per_request = 0.0001
        return completed * estimated_cost_per_request
    
    def _display_experiment_batches(self, batches: List[Dict[str, Any]]) -> None:
        """Display batches grouped by experiment type."""
        experiment_batches = {
            'chatgpt_evaluation': [],
            'sota_comparison': [],
            'prompt_comparison': [],
            'other': []
        }
        
        for batch in batches:
            metadata = batch.get('metadata', {})
            experiment_type = metadata.get('experiment_type', 'other')
            
            if experiment_type in experiment_batches:
                experiment_batches[experiment_type].append(batch)
            else:
                experiment_batches['other'].append(batch)
        
        print(f"\n{'='*80}")
        print(f"BATCHES BY EXPERIMENT TYPE")
        print(f"{'='*80}")
        
        for exp_type, batch_list in experiment_batches.items():
            if batch_list:
                print(f"\n{exp_type.replace('_', ' ').title()} ({len(batch_list)} batches):")
                for batch in batch_list:
                    batch_id = batch.get('id', 'unknown')[:18]
                    status = batch.get('status', 'unknown')
                    metadata = batch.get('metadata', {})
                    
                    tasks = metadata.get('tasks', [])
                    datasets = metadata.get('datasets', [])
                    prompt_type = metadata.get('prompt_type', 'unknown')
                    
                    print(f"  • {batch_id} [{status}]")
                    print(f"    Tasks: {', '.join(tasks) if tasks else 'N/A'}")
                    print(f"    Datasets: {', '.join(datasets) if datasets else 'N/A'}")
                    print(f"    Prompt Type: {prompt_type}")
    
    async def display_batch_details(self, batch_id: str) -> None:
        """Display detailed information about a specific batch."""
        batch_details = await self.get_batch_details(batch_id)
        
        if not batch_details:
            print(f"Batch {batch_id} not found or error retrieving details.")
            return
        
        print(f"\n{'='*80}")
        print(f"DETAILED BATCH INFORMATION: {batch_id}")
        print(f"{'='*80}")
        
        # Basic information
        print(f"ID: {batch_details.get('id', 'N/A')}")
        print(f"Status: {batch_details.get('status', 'N/A')}")
        print(f"Created: {datetime.fromtimestamp(batch_details.get('created_at', 0)).strftime('%Y-%m-%d %H:%M:%S')}")
        
        if batch_details.get('completed_at'):
            print(f"Completed: {datetime.fromtimestamp(batch_details.get('completed_at')).strftime('%Y-%m-%d %H:%M:%S')}")
        
        if batch_details.get('expires_at'):
            print(f"Expires: {datetime.fromtimestamp(batch_details.get('expires_at')).strftime('%Y-%m-%d %H:%M:%S')}")
        
        # Request counts
        request_counts = batch_details.get('request_counts', {})
        print(f"\nRequest Counts:")
        print(f"  Total: {request_counts.get('total', 0)}")
        print(f"  Completed: {request_counts.get('completed', 0)}")
        print(f"  Failed: {request_counts.get('failed', 0)}")
        
        # Progress
        total = request_counts.get('total', 0)
        completed = request_counts.get('completed', 0)
        if total > 0:
            progress_pct = (completed / total) * 100
            print(f"  Progress: {progress_pct:.1f}%")
        
        # Metadata (experiment information)
        metadata = batch_details.get('metadata', {})
        if metadata:
            print(f"\nExperiment Metadata:")
            for key, value in metadata.items():
                print(f"  {key}: {value}")
        
        # Error information if available
        errors = batch_details.get('errors')
        if errors:
            print(f"\nErrors:")
            for error in errors:
                print(f"  • {error}")
    
    async def watch_batches(self, interval: int = 60) -> None:
        """Continuously monitor batches with auto-refresh."""
        print(f"Starting batch monitoring (refresh every {interval} seconds)")
        print("Press Ctrl+C to stop...")
        
        try:
            while True:
                # Clear screen (simple approach)
                print("\033[2J\033[H")  # ANSI escape codes to clear screen and move cursor to top
                
                batches = await self.list_all_batches()
                self.display_batch_summary(batches)
                
                print(f"\nNext refresh in {interval} seconds... (Press Ctrl+C to stop)")
                
                await asyncio.sleep(interval)
                
        except KeyboardInterrupt:
            print("\nMonitoring stopped.")
    
    async def get_monitoring_recommendations(self, batches: List[Dict[str, Any]]) -> List[str]:
        """Generate recommendations based on batch status."""
        recommendations = []
        
        # Check for failed batches
        failed_batches = [b for b in batches if b.get('status') == 'failed']
        if failed_batches:
            recommendations.append(f"⚠️  {len(failed_batches)} batch(es) have failed. Check error logs and consider resubmitting.")
        
        # Check for long-running batches
        current_time = time.time()
        long_running_batches = []
        for batch in batches:
            if batch.get('status') == 'in_progress':
                created_at = batch.get('created_at', current_time)
                if current_time - created_at > 24 * 3600:  # More than 24 hours
                    long_running_batches.append(batch)
        
        if long_running_batches:
            recommendations.append(f"⏰ {len(long_running_batches)} batch(es) have been running for more than 24 hours. Consider checking for issues.")
        
        # Check for completed batches
        completed_batches = [b for b in batches if b.get('status') == 'completed']
        if completed_batches:
            recommendations.append(f"✅ {len(completed_batches)} batch(es) completed successfully. You can download results.")
        
        # Check for pending batches
        pending_batches = [b for b in batches if b.get('status') in ['validating', 'in_progress']]
        if pending_batches:
            recommendations.append(f"🔄 {len(pending_batches)} batch(es) are currently processing. Monitor progress regularly.")
        
        return recommendations


async def main():
    """Main entry point for batch monitoring."""
    parser = argparse.ArgumentParser(description="Monitor OpenAI batch processing jobs")
    parser.add_argument("--config", type=str, help="Path to configuration file")
    parser.add_argument("--batch-id", type=str, help="Monitor specific batch by ID")
    parser.add_argument("--watch", action="store_true", help="Continuous monitoring mode")
    parser.add_argument("--interval", type=int, default=60, help="Refresh interval for watch mode (seconds)")
    parser.add_argument("--recommendations", action="store_true", help="Show monitoring recommendations")
    
    args = parser.parse_args()
    
    # Initialize monitor
    monitor = BatchStatusMonitor(args.config)
    
    try:
        if args.batch_id:
            # Monitor specific batch
            await monitor.display_batch_details(args.batch_id)
        elif args.watch:
            # Continuous monitoring
            await monitor.watch_batches(args.interval)
        else:
            # Single status check
            batches = await monitor.list_all_batches()
            monitor.display_batch_summary(batches)
            
            if args.recommendations:
                recommendations = await monitor.get_monitoring_recommendations(batches)
                if recommendations:
                    print(f"\n{'='*80}")
                    print(f"MONITORING RECOMMENDATIONS")
                    print(f"{'='*80}")
                    for rec in recommendations:
                        print(f"{rec}")
                else:
                    print(f"\n✅ All batches are running normally. No recommendations at this time.")
    
    except Exception as e:
        print(f"Error during monitoring: {e}")
        sys.exit(1)


if __name__ == "__main__":
    asyncio.run(main())
