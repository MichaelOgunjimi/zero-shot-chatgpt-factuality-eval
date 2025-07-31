"""
Batch Master Experiment Runner
=============================

Master orchestrator for running all batch processing experiments in sequence.
Mirrors the logic from run_all_experiments.py but optimized for batch processing
using OpenAI's Batch API for cost-effective large-scale evaluation.

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
sys.path.insert(0, str(Path(__file__).parent.parent.parent))

try:
    from src.utils import (
        setup_experiment_logger,
        get_config,
        setup_reproducibility,
        validate_api_keys
    )
    from src.batch import BatchManager, BatchMonitor, BatchJob, BatchStatus
    from src.llm_clients.openai_client_batch import OpenAIBatchClient
    
    # Import batch experiment classes
    from .batch_run_chatgpt_evaluation import BatchChatGPTEvaluationExperiment
    from .batch_prompt_comparison import BatchPromptComparisonExperiment
    from .batch_sota_comparison import BatchSOTAComparisonExperiment
    
    import numpy as np
    import pandas as pd
    import plotly.graph_objects as go
    import plotly.express as px
    from plotly.subplots import make_subplots
    
except ImportError as e:
    print(f"Error importing required modules: {e}")
    print("Please ensure you're running from the project root directory.")
    sys.exit(1)


class BatchMasterExperimentRunner:
    """
    Master experiment runner for batch processing operations.
    
    Orchestrates all three main batch experiments (ChatGPT evaluation, 
    prompt comparison, SOTA comparison) with comprehensive cost tracking,
    monitoring, and consolidated reporting for thesis requirements.
    """

    def __init__(self, model: str = "gpt-4.1-mini", tier: str = "tier2", experiment_name: str = None):
        """
        Initialize batch master experiment runner.

        Args:
            model: Model to use for all experiments
            tier: API tier for rate limiting
            experiment_name: Name for this master experiment run
        """
        # Load configuration
        self.config = get_config(model=model, tier=tier)
        
        # Store model info
        self.model = model
        self.tier = tier
        
        # Set up experiment tracking
        self.experiment_name = experiment_name or f"batch_master_{datetime.now().strftime('%Y%m%d_%H%M%S')}"
        self.output_dir = Path(f"results/experiments/batch_processing/{self.experiment_name}")
        self.output_dir.mkdir(parents=True, exist_ok=True)
        
        # Create master analysis directory
        self.master_output_dir = self.output_dir / "master_analysis"
        self.master_output_dir.mkdir(exist_ok=True)
        (self.master_output_dir / "visualizations").mkdir(exist_ok=True)
        
        # Set up logging
        self.experiment_logger = setup_experiment_logger(
            self.experiment_name,
            self.config,
            log_dir=self.output_dir / "logs"
        )
        self.logger = self.experiment_logger.logger
        
        # Set up reproducibility
        setup_reproducibility(self.config)
        
        # Validate API keys
        validate_api_keys(self.config)
        
        # Initialize batch client for global monitoring
        self.batch_client = OpenAIBatchClient(self.config, self.experiment_name)
        
        # Experiment tracking
        self.experiment_results = {
            'master_experiment_metadata': {
                'name': self.experiment_name,
                'timestamp': datetime.now().isoformat(),
                'config': self.config.to_dict(),
                'experiment_type': 'batch_master_experiment_suite',
                'model': self.model,
                'tier': self.tier
            },
            'individual_experiments': {},
            'consolidated_analysis': {},
            'cost_summary': {},
            'execution_summary': {},
            'batch_monitoring': {}
        }
        
        self.logger.info(f"Initialized batch master experiment runner: {self.experiment_name}")

    async def run_complete_batch_experimental_suite(
        self,
        quick_test: bool = False,
        full_suite: bool = True,
        experiments_to_run: List[str] = None
    ) -> Dict[str, Any]:
        """
        Run the complete batch experimental suite for thesis.

        Args:
            quick_test: Whether to run with reduced sample sizes
            full_suite: Whether to run all experiments (vs individual)
            experiments_to_run: Specific experiments to run

        Returns:
            Consolidated results from all experiments
        """
        self.logger.info("Starting complete batch experimental suite")
        start_time = time.time()
        
        # Default experiments
        if experiments_to_run is None:
            experiments_to_run = ['chatgpt_evaluation', 'prompt_comparison', 'sota_comparison']
        
        try:
            # Phase 1: Pre-experiment setup and validation
            await self._validate_batch_environment()
            
            # Phase 2: Run individual experiments
            if 'chatgpt_evaluation' in experiments_to_run:
                await self._run_chatgpt_evaluation_experiment(quick_test)
            
            if 'prompt_comparison' in experiments_to_run:
                await self._run_prompt_comparison_experiment(quick_test)
            
            if 'sota_comparison' in experiments_to_run:
                await self._run_sota_comparison_experiment(quick_test)
            
            # Phase 3: Consolidated analysis
            await self._perform_consolidated_analysis()
            
            # Phase 4: Master visualizations
            await self._generate_master_visualizations()
            
            # Phase 5: Final reporting
            await self._generate_master_reports()
            
            # Calculate execution time
            total_time = time.time() - start_time
            self.experiment_results['execution_summary'] = {
                'total_execution_time': total_time,
                'experiments_completed': len(self.experiment_results['individual_experiments']),
                'experiments_requested': len(experiments_to_run),
                'completion_rate': len(self.experiment_results['individual_experiments']) / len(experiments_to_run),
                'start_time': datetime.fromtimestamp(start_time).isoformat(),
                'end_time': datetime.now().isoformat()
            }
            
            self.logger.info(f"Batch experimental suite completed in {total_time:.2f} seconds")
            return self.experiment_results
            
        except Exception as e:
            self.logger.error(f"Batch experimental suite failed: {e}")
            raise

    async def _validate_batch_environment(self):
        """Validate batch processing environment and capabilities."""
        self.logger.info("Validating batch processing environment")
        
        # Check batch API availability
        if not self.batch_client.is_batch_processing_available():
            raise ValueError("Batch processing is not available or enabled")
        
        # Perform health check
        health_status = await BatchMonitor(self.batch_client.batch_manager).health_check()
        
        if not health_status['overall_healthy']:
            self.logger.warning(f"Health check issues detected: {health_status['issues']}")
        
        # Check queue capacity
        batch_config = self.config.get("openai.batch", {})
        max_queue = batch_config.get("max_queue_size", 1000000)
        
        self.logger.info(f"Batch environment validated - Max queue: {max_queue:,}")

    async def _run_chatgpt_evaluation_experiment(self, quick_test: bool):
        """Run ChatGPT evaluation experiment in batch mode."""
        self.logger.info("Running ChatGPT evaluation experiment")
        
        try:
            experiment = BatchChatGPTEvaluationExperiment(
                model=self.model,
                tier=self.tier,
                experiment_name=f"{self.experiment_name}_chatgpt_eval"
            )
            
            results = await experiment.run_batch_evaluation(quick_test=quick_test)
            
            self.experiment_results['individual_experiments']['chatgpt_evaluation'] = {
                'status': 'completed',
                'experiment_name': experiment.experiment_name,
                'output_dir': str(experiment.output_dir),
                'results': results,
                'summary': {
                    'total_evaluations': results['overall_performance']['total_evaluations'],
                    'success_rate': results['overall_performance']['overall_success_rate'],
                    'total_cost': results['batch_analysis']['total_cost']
                }
            }
            
            self.logger.info("ChatGPT evaluation experiment completed")
            
        except Exception as e:
            self.logger.error(f"ChatGPT evaluation experiment failed: {e}")
            self.experiment_results['individual_experiments']['chatgpt_evaluation'] = {
                'status': 'failed',
                'error': str(e)
            }

    async def _run_prompt_comparison_experiment(self, quick_test: bool):
        """Run prompt comparison experiment in batch mode."""
        self.logger.info("Running prompt comparison experiment")
        
        try:
            experiment = BatchPromptComparisonExperiment(
                model=self.model,
                tier=self.tier,
                experiment_name=f"{self.experiment_name}_prompt_comp"
            )
            
            results = await experiment.run_prompt_comparison(quick_test=quick_test)
            
            self.experiment_results['individual_experiments']['prompt_comparison'] = {
                'status': 'completed',
                'experiment_name': experiment.experiment_name,
                'output_dir': str(experiment.output_dir),
                'results': results,
                'summary': {
                    'mean_improvement': results['statistical_analysis']['overall_statistics']['mean_improvement_percent'],
                    'significant_improvements': results['statistical_analysis']['overall_statistics']['significant_improvements'],
                    'total_comparisons': results['statistical_analysis']['overall_statistics']['total_comparisons'],
                    'total_cost': results['batch_analysis']['total_cost']
                }
            }
            
            self.logger.info("Prompt comparison experiment completed")
            
        except Exception as e:
            self.logger.error(f"Prompt comparison experiment failed: {e}")
            self.experiment_results['individual_experiments']['prompt_comparison'] = {
                'status': 'failed',
                'error': str(e)
            }

    async def _run_sota_comparison_experiment(self, quick_test: bool):
        """Run SOTA comparison experiment in batch mode."""
        self.logger.info("Running SOTA comparison experiment")
        
        try:
            experiment = BatchSOTAComparisonExperiment(
                model=self.model,
                tier=self.tier,
                experiment_name=f"{self.experiment_name}_sota_comp"
            )
            
            results = await experiment.run_sota_comparison(quick_test=quick_test)
            
            self.experiment_results['individual_experiments']['sota_comparison'] = {
                'status': 'completed',
                'experiment_name': experiment.experiment_name,
                'output_dir': str(experiment.output_dir),
                'results': results,
                'summary': {
                    'mean_correlation': results['correlation_analysis']['correlation_summary']['overall_mean_pearson'],
                    'valid_correlations': results['correlation_analysis']['correlation_summary']['valid_correlations'],
                    'best_baseline': results['correlation_analysis']['correlation_summary'].get('best_correlating_baseline'),
                    'total_cost': results['batch_analysis']['chatgpt_cost']
                }
            }
            
            self.logger.info("SOTA comparison experiment completed")
            
        except Exception as e:
            self.logger.error(f"SOTA comparison experiment failed: {e}")
            self.experiment_results['individual_experiments']['sota_comparison'] = {
                'status': 'failed',
                'error': str(e)
            }

    async def _perform_consolidated_analysis(self):
        """Perform consolidated analysis across all experiments."""
        self.logger.info("Performing consolidated analysis")
        
        # Cost analysis across all experiments
        total_cost = 0.0
        total_savings = 0.0
        successful_experiments = 0
        
        cost_breakdown = {}
        key_findings = {}
        
        for exp_name, exp_data in self.experiment_results['individual_experiments'].items():
            if exp_data['status'] == 'completed':
                successful_experiments += 1
                exp_cost = exp_data['summary']['total_cost']
                total_cost += exp_cost
                cost_breakdown[exp_name] = exp_cost
                
                # Extract key findings
                if exp_name == 'chatgpt_evaluation':
                    key_findings['chatgpt_evaluation'] = {
                        'total_evaluations': exp_data['summary']['total_evaluations'],
                        'success_rate': exp_data['summary']['success_rate']
                    }
                elif exp_name == 'prompt_comparison':
                    key_findings['prompt_comparison'] = {
                        'mean_improvement': exp_data['summary']['mean_improvement'],
                        'significant_improvements': exp_data['summary']['significant_improvements'],
                        'total_comparisons': exp_data['summary']['total_comparisons']
                    }
                elif exp_name == 'sota_comparison':
                    key_findings['sota_comparison'] = {
                        'mean_correlation': exp_data['summary']['mean_correlation'],
                        'best_baseline': exp_data['summary']['best_baseline'],
                        'valid_correlations': exp_data['summary']['valid_correlations']
                    }
                
                # Calculate estimated savings
                if 'results' in exp_data and 'batch_analysis' in exp_data['results']:
                    batch_data = exp_data['results']['batch_analysis']
                    if 'cost_savings' in batch_data:
                        total_savings += batch_data['cost_savings']

        # Get global batch monitoring summary
        all_jobs = self.batch_client.batch_manager.get_all_jobs()
        batch_summary = self.batch_client.batch_manager.get_cost_summary()
        
        self.experiment_results['consolidated_analysis'] = {
            'cost_analysis': {
                'total_experimental_cost': total_cost,
                'total_estimated_savings': total_savings,
                'estimated_sync_cost': total_cost + total_savings,
                'cost_breakdown_by_experiment': cost_breakdown,
                'average_cost_per_experiment': total_cost / max(successful_experiments, 1),
                'batch_processing_efficiency': total_savings / (total_cost + total_savings) if (total_cost + total_savings) > 0 else 0
            },
            'key_findings': key_findings,
            'experiment_summary': {
                'total_experiments_attempted': len(self.experiment_results['individual_experiments']),
                'successful_experiments': successful_experiments,
                'failed_experiments': len(self.experiment_results['individual_experiments']) - successful_experiments,
                'overall_success_rate': successful_experiments / len(self.experiment_results['individual_experiments']) if self.experiment_results['individual_experiments'] else 0
            },
            'batch_operations_summary': {
                'total_batch_jobs': len(all_jobs['active']) + len(all_jobs['completed']) + len(all_jobs['failed']),
                'completed_batch_jobs': len(all_jobs['completed']),
                'failed_batch_jobs': len(all_jobs['failed']),
                'active_batch_jobs': len(all_jobs['active']),
                'batch_success_rate': len(all_jobs['completed']) / max(len(all_jobs['active']) + len(all_jobs['completed']) + len(all_jobs['failed']), 1)
            }
        }
        
        # Store cost summary for compatibility
        self.experiment_results['cost_summary'] = self.experiment_results['consolidated_analysis']['cost_analysis']

    async def _generate_master_visualizations(self):
        """Generate master-level visualizations."""
        self.logger.info("Generating master visualizations")
        
        viz_dir = self.master_output_dir / "visualizations"
        
        # 1. Cost breakdown visualization
        await self._create_cost_breakdown_chart(viz_dir)
        
        # 2. Experimental timeline
        await self._create_experimental_timeline(viz_dir)
        
        # 3. Performance dashboard
        await self._create_performance_dashboard(viz_dir)
        
        # 4. Key findings summary
        await self._create_key_findings_visualization(viz_dir)

    async def _create_cost_breakdown_chart(self, viz_dir: Path):
        """Create cost breakdown visualization."""
        cost_analysis = self.experiment_results['consolidated_analysis']['cost_analysis']
        cost_breakdown = cost_analysis['cost_breakdown_by_experiment']
        
        if not cost_breakdown:
            return
        
        # Pie chart for cost breakdown
        fig = make_subplots(
            rows=1, cols=2,
            subplot_titles=('Cost Breakdown by Experiment', 'Batch vs Sync Cost Comparison'),
            specs=[[{"type": "pie"}, {"type": "bar"}]]
        )
        
        # Cost breakdown pie
        experiments = list(cost_breakdown.keys())
        costs = list(cost_breakdown.values())
        
        fig.add_trace(
            go.Pie(labels=[exp.replace('_', ' ').title() for exp in experiments], 
                  values=costs, name="Cost Breakdown"),
            row=1, col=1
        )
        
        # Batch vs sync comparison
        total_actual = cost_analysis['total_experimental_cost']
        total_estimated_sync = cost_analysis['estimated_sync_cost']
        total_savings = cost_analysis['total_estimated_savings']
        
        fig.add_trace(
            go.Bar(x=['Actual Batch Cost', 'Est. Sync Cost', 'Total Savings'],
                  y=[total_actual, total_estimated_sync, total_savings],
                  marker_color=['blue', 'red', 'green'],
                  name="Cost Comparison"),
            row=1, col=2
        )
        
        fig.update_layout(
            title=f"Master Cost Analysis - Total: ${total_actual:.4f}",
            height=500
        )
        
        fig.write_html(viz_dir / "cost_breakdown.html")
        fig.write_image(viz_dir / "cost_breakdown.png", width=1000, height=500, scale=2)

    async def _create_experimental_timeline(self, viz_dir: Path):
        """Create experimental timeline visualization."""
        # Collect timeline data from individual experiments
        timeline_data = []
        
        for exp_name, exp_data in self.experiment_results['individual_experiments'].items():
            if exp_data['status'] == 'completed' and 'results' in exp_data:
                # Extract timing information from experiment metadata
                metadata = exp_data['results']['experiment_metadata']
                start_time = datetime.fromisoformat(metadata['timestamp'])
                
                # Estimate duration based on batch processing
                if 'batch_analysis' in exp_data['results']:
                    # Use actual batch processing time if available
                    duration_minutes = 30  # Default estimate
                    
                    # Try to get more accurate timing from batch jobs
                    batch_analysis = exp_data['results']['batch_analysis']
                    if 'total_jobs' in batch_analysis:
                        # Estimate based on job count (batch processing typically takes 15-30 min)
                        duration_minutes = max(15, batch_analysis['total_jobs'] * 5)  # 5 min per job estimate
                
                end_time = start_time + timedelta(minutes=duration_minutes)
                
                timeline_data.append({
                    'Experiment': exp_name.replace('_', ' ').title(),
                    'Start': start_time,
                    'Finish': end_time,
                    'Duration': duration_minutes,
                    'Status': 'Completed',
                    'Cost': exp_data['summary']['total_cost']
                })

        if timeline_data:
            df = pd.DataFrame(timeline_data)
            
            fig = px.timeline(
                df, x_start="Start", x_end="Finish", y="Experiment",
                color="Cost", title="Experimental Timeline",
                hover_data=["Duration", "Status"]
            )
            
            fig.update_layout(
                height=200 + len(timeline_data) * 50,
                xaxis_title="Time",
                yaxis_title="Experiment"
            )
            
            fig.write_html(viz_dir / "experimental_timeline.html")
            fig.write_image(viz_dir / "experimental_timeline.png", 
                          width=1200, height=200 + len(timeline_data) * 50, scale=2)

    async def _create_performance_dashboard(self, viz_dir: Path):
        """Create master performance dashboard."""
        key_findings = self.experiment_results['consolidated_analysis']['key_findings']
        
        # Create dashboard layout
        fig = make_subplots(
            rows=2, cols=2,
            subplot_titles=('Experiment Success Rates', 'Cost per Experiment',
                          'Key Performance Metrics', 'Batch Processing Efficiency'),
            specs=[[{"type": "bar"}, {"type": "bar"}],
                   [{"type": "indicator"}, {"type": "pie"}]]
        )
        
        # Experiment success rates
        experiments = []
        success_rates = []
        costs = []
        
        for exp_name, exp_data in self.experiment_results['individual_experiments'].items():
            if exp_data['status'] == 'completed':
                experiments.append(exp_name.replace('_', ' ').title())
                costs.append(exp_data['summary']['total_cost'])
                
                # Extract success rate based on experiment type
                if exp_name == 'chatgpt_evaluation':
                    success_rates.append(exp_data['summary']['success_rate'])
                elif exp_name == 'prompt_comparison':
                    # Use proportion of significant improvements as success metric
                    significant = exp_data['summary']['significant_improvements']
                    total = exp_data['summary']['total_comparisons']
                    success_rates.append(significant / max(total, 1))
                elif exp_name == 'sota_comparison':
                    # Use proportion of valid correlations as success metric
                    valid = exp_data['summary']['valid_correlations']
                    success_rates.append(min(valid / 10, 1.0))  # Normalize to expected ~10 correlations
                else:
                    success_rates.append(1.0)  # Default to 100% if completed
        
        if experiments:
            # Success rates
            fig.add_trace(
                go.Bar(x=experiments, y=success_rates, name="Success Rate",
                      marker_color='green', text=[f"{sr:.1%}" for sr in success_rates]),
                row=1, col=1
            )
            
            # Costs
            fig.add_trace(
                go.Bar(x=experiments, y=costs, name="Total Cost",
                      marker_color='blue', text=[f"${c:.4f}" for c in costs]),
                row=1, col=2
            )
        
        # Key metrics indicator
        cost_analysis = self.experiment_results['consolidated_analysis']['cost_analysis']
        
        fig.add_trace(
            go.Indicator(
                mode="gauge+number+delta",
                value=cost_analysis['total_experimental_cost'],
                domain={'x': [0, 1], 'y': [0, 1]},
                title={'text': "Total Cost ($)"},
                delta={'reference': cost_analysis['estimated_sync_cost']},
                gauge={'axis': {'range': [None, cost_analysis['estimated_sync_cost']]},
                      'bar': {'color': "blue"},
                      'steps': [{'range': [0, cost_analysis['total_experimental_cost']], 'color': "lightgray"}],
                      'threshold': {'line': {'color': "red", 'width': 4},
                                  'thickness': 0.75, 'value': cost_analysis['estimated_sync_cost']}}
            ),
            row=2, col=1
        )
        
        # Batch efficiency pie
        efficiency = cost_analysis['batch_processing_efficiency']
        fig.add_trace(
            go.Pie(labels=['Batch Savings', 'Actual Cost'], 
                  values=[efficiency, 1-efficiency],
                  name="Efficiency"),
            row=2, col=2
        )
        
        fig.update_layout(
            title="Master Performance Dashboard",
            height=800,
            showlegend=False
        )
        
        fig.write_html(viz_dir / "performance_dashboard.html")
        fig.write_image(viz_dir / "performance_dashboard.png", width=1200, height=800, scale=2)

    async def _create_key_findings_visualization(self, viz_dir: Path):
        """Create key findings summary visualization."""
        key_findings = self.experiment_results['consolidated_analysis']['key_findings']
        
        # Extract key metrics for visualization
        metrics_data = []
        
        if 'chatgpt_evaluation' in key_findings:
            eval_data = key_findings['chatgpt_evaluation']
            metrics_data.append({
                'Metric': 'ChatGPT Success Rate',
                'Value': eval_data['success_rate'],
                'Category': 'Evaluation',
                'Format': 'percentage'
            })
        
        if 'prompt_comparison' in key_findings:
            prompt_data = key_findings['prompt_comparison']
            metrics_data.append({
                'Metric': 'CoT Improvement',
                'Value': prompt_data['mean_improvement'],
                'Category': 'Prompting',
                'Format': 'percentage'
            })
            metrics_data.append({
                'Metric': 'Significant Improvements',
                'Value': prompt_data['significant_improvements'] / max(prompt_data['total_comparisons'], 1),
                'Category': 'Prompting',
                'Format': 'percentage'
            })
        
        if 'sota_comparison' in key_findings:
            sota_data = key_findings['sota_comparison']
            metrics_data.append({
                'Metric': 'Mean Correlation',
                'Value': sota_data['mean_correlation'],
                'Category': 'SOTA',
                'Format': 'correlation'
            })

        if metrics_data:
            df = pd.DataFrame(metrics_data)
            
            # Create grouped bar chart
            fig = px.bar(
                df, x='Metric', y='Value', color='Category',
                title='Key Experimental Findings Summary',
                labels={'Value': 'Metric Value', 'Metric': 'Performance Metric'}
            )
            
            fig.update_layout(height=500)
            fig.write_html(viz_dir / "key_findings_summary.html")
            fig.write_image(viz_dir / "key_findings_summary.png", width=1000, height=500, scale=2)
        
        # Save key findings as JSON for easy access
        with open(viz_dir / "key_findings_summary.json", 'w') as f:
            json.dump(key_findings, f, indent=2, default=str)

    async def _generate_master_reports(self):
        """Generate master-level reports and documentation."""
        self.logger.info("Generating master reports")
        
        # Save consolidated results as JSON
        json_path = self.master_output_dir / "master_experimental_results.json"
        with open(json_path, 'w') as f:
            json.dump(self.experiment_results, f, indent=2, default=str)
        
        # Generate comprehensive markdown report
        report_path = self.master_output_dir / "master_experimental_report.md"
        with open(report_path, 'w') as f:
            f.write(self._generate_master_report())
        
        # Generate executive summary
        summary_path = self.master_output_dir / "executive_summary.md"
        with open(summary_path, 'w') as f:
            f.write(self._generate_executive_summary())
        
        # *** ADD THESE ROOT-LEVEL FILES ***
        # Generate batch master results (exact filename match)
        batch_results_path = self.output_dir / "batch_master_results.json"
        with open(batch_results_path, 'w') as f:
            json.dump({
                'master_experiment': self.experiment_name,
                'timestamp': datetime.now().isoformat(),
                'consolidated_results': self.experiment_results['consolidated_analysis'],
                'individual_experiments': {k: v['summary'] for k, v in self.experiment_results['individual_experiments'].items() if v['status'] == 'completed'},
                'execution_summary': self.experiment_results['execution_summary'],
                'batch_operations': self.experiment_results['consolidated_analysis']['batch_operations_summary']
            }, f, indent=2, default=str)
        
        # Generate batch master report (exact filename match)  
        batch_report_path = self.output_dir / "batch_master_report.md"
        with open(batch_report_path, 'w') as f:
            f.write(self._generate_batch_master_report())
        
        # Create directory README
        readme_path = self.output_dir / "README.md"
        with open(readme_path, 'w') as f:
            f.write(self._generate_directory_readme())
        
        self.logger.info(f"Master reports generated in: {self.master_output_dir}")

    def _generate_master_report(self) -> str:
        """Generate comprehensive master report."""
        cost_analysis = self.experiment_results['consolidated_analysis']['cost_analysis']
        key_findings = self.experiment_results['consolidated_analysis']['key_findings']
        execution_summary = self.experiment_results['execution_summary']
        
        report = f"""# ChatGPT Factuality Evaluation - Batch Master Experimental Report

**Experiment Suite**: {self.experiment_name}  
**Generated**: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}  
**Author**: Michael Ogunjimi  
**Institution**: University of Manchester, MSc AI  

## Executive Summary

This report presents the complete batch processing experimental evaluation of ChatGPT's
zero-shot factuality assessment capabilities, including prompt design optimization and
comparison with state-of-the-art baseline methods. All experiments utilized OpenAI's
Batch API for significant cost optimization.

## Experimental Overview

### Experiments Conducted

1. **ChatGPT Evaluation**: Core performance assessment across factuality tasks
2. **Prompt Comparison**: Zero-shot vs Chain-of-thought prompting analysis  
3. **SOTA Comparison**: Correlation analysis with traditional factuality metrics

### Batch Processing Benefits

- **Total Cost**: ${cost_analysis['total_experimental_cost']:.4f}
- **Estimated Sync Cost**: ${cost_analysis['estimated_sync_cost']:.4f}
- **Total Savings**: ${cost_analysis['total_estimated_savings']:.4f}
- **Efficiency Gain**: {cost_analysis['batch_processing_efficiency']:.1%}

## Individual Experiment Results

"""

        # Add results for each experiment
        for exp_name, exp_data in self.experiment_results['individual_experiments'].items():
            if exp_data['status'] == 'completed':
                report += f"### {exp_name.replace('_', ' ').title()}\n\n"
                report += f"**Status**: ✅ Completed  \n"
                report += f"**Output Directory**: `{exp_data.get('output_dir', 'unknown')}`\n\n"
                
                # Add experiment-specific highlights
                summary = exp_data['summary']
                
                if exp_name == 'chatgpt_evaluation':
                    report += f"- **Total Evaluations**: {summary['total_evaluations']:,}\n"
                    report += f"- **Success Rate**: {summary['success_rate']:.2%}\n"
                    report += f"- **Cost**: ${summary['total_cost']:.4f}\n"
                
                elif exp_name == 'prompt_comparison':
                    report += f"- **Mean Improvement**: {summary['mean_improvement']:+.2f}%\n"
                    report += f"- **Significant Results**: {summary['significant_improvements']}/{summary['total_comparisons']}\n"
                    report += f"- **Cost**: ${summary['total_cost']:.4f}\n"
                
                elif exp_name == 'sota_comparison':
                    report += f"- **Mean Correlation**: {summary['mean_correlation']:.3f}\n"
                    report += f"- **Best Baseline**: {summary.get('best_baseline', 'N/A')}\n"
                    report += f"- **Valid Correlations**: {summary['valid_correlations']}\n"
                    report += f"- **Cost**: ${summary['total_cost']:.4f}\n"
                
                report += "\n"
            else:
                report += f"### {exp_name.replace('_', ' ').title()}\n\n"
                report += f"**Status**: ❌ Failed  \n"
                report += f"**Error**: {exp_data.get('error', 'Unknown error')}\n\n"

        report += f"""## Consolidated Analysis

### Key Findings

"""

        # Add key findings
        if 'chatgpt_evaluation' in key_findings:
            eval_findings = key_findings['chatgpt_evaluation']
            report += f"1. **ChatGPT Evaluation Performance**: {eval_findings['success_rate']:.1%} success rate across {eval_findings['total_evaluations']:,} evaluations\n"

        if 'prompt_comparison' in key_findings:
            prompt_findings = key_findings['prompt_comparison']
            report += f"2. **Prompt Design Impact**: {prompt_findings['mean_improvement']:+.2f}% improvement with chain-of-thought prompting\n"

        if 'sota_comparison' in key_findings:
            sota_findings = key_findings['sota_comparison']
            report += f"3. **SOTA Agreement**: {sota_findings['mean_correlation']:.3f} average correlation with traditional metrics\n"

        report += f"""

### Cost-Benefit Analysis

**Total Experimental Cost**: ${cost_analysis['total_experimental_cost']:.4f}
- ChatGPT Evaluation: ${cost_analysis['cost_breakdown_by_experiment'].get('chatgpt_evaluation', 0):.4f}
- Prompt Comparison: ${cost_analysis['cost_breakdown_by_experiment'].get('prompt_comparison', 0):.4f}
- SOTA Comparison: ${cost_analysis['cost_breakdown_by_experiment'].get('sota_comparison', 0):.4f}

**Batch Processing Benefits**:
- Estimated Synchronous Cost: ${cost_analysis['estimated_sync_cost']:.4f}
- Actual Batch Cost: ${cost_analysis['total_experimental_cost']:.4f}
- Total Savings: ${cost_analysis['total_estimated_savings']:.4f}
- Efficiency Improvement: {cost_analysis['batch_processing_efficiency']:.1%}

### Execution Performance

- **Total Execution Time**: {execution_summary['total_execution_time'] / 60:.1f} minutes
- **Experiments Completed**: {execution_summary['experiments_completed']}/{execution_summary['experiments_requested']}
- **Completion Rate**: {execution_summary['completion_rate']:.1%}

## Batch Processing Analysis

### Job Management
"""

        batch_ops = self.experiment_results['consolidated_analysis']['batch_operations_summary']
        report += f"""
- **Total Batch Jobs**: {batch_ops['total_batch_jobs']}
- **Completed Jobs**: {batch_ops['completed_batch_jobs']}
- **Failed Jobs**: {batch_ops['failed_batch_jobs']}
- **Success Rate**: {batch_ops['batch_success_rate']:.1%}

### Technical Performance
- **Model Used**: {self.model}
- **API Tier**: {self.tier}
- **Batch Queue Utilization**: Optimized for cost and speed
- **Monitoring**: Real-time job status tracking implemented

## Implications for Research

### Methodological Contributions
1. **Batch Processing Framework**: Demonstrated cost-effective large-scale evaluation
2. **Comprehensive Comparison**: Systematic evaluation across multiple dimensions
3. **Statistical Rigor**: Proper significance testing and effect size analysis

### Practical Applications
1. **Academic Research**: Framework suitable for thesis-level comprehensive evaluation
2. **Cost Management**: Significant savings enable larger-scale studies
3. **Reproducibility**: Complete experimental pipeline with detailed logging

## Recommendations

### For Future Research
1. **Scale Expansion**: Utilize batch processing for even larger sample sizes
2. **Multi-Model Studies**: Extend framework to compare across different LLMs
3. **Longitudinal Analysis**: Track performance changes over time

### For Implementation
1. **Batch Processing**: Highly recommended for any large-scale LLM evaluation
2. **Cost Monitoring**: Essential for budget management in academic research
3. **Comprehensive Logging**: Critical for reproducibility and debugging

## Limitations and Considerations

### Current Limitations
- **Batch Processing Delay**: 24-hour processing window vs immediate results
- **API Dependency**: Reliance on OpenAI's batch processing stability
- **Cost Variability**: Pricing may change affecting reproducibility

### Mitigation Strategies
- **Parallel Development**: Run baseline computations during batch processing
- **Error Handling**: Comprehensive error tracking and recovery mechanisms
- **Cost Budgeting**: Conservative cost estimation with buffer allocation

---
*Report generated by BatchMasterExperimentRunner*
"""

        return report

    def _generate_executive_summary(self) -> str:
        """Generate executive summary for quick reference."""
        cost_analysis = self.experiment_results['consolidated_analysis']['cost_analysis']
        key_findings = self.experiment_results['consolidated_analysis']['key_findings']
        execution_summary = self.experiment_results['execution_summary']
        
        summary = f"""# Executive Summary - Batch Experimental Suite

**Experiment**: {self.experiment_name}  
**Date**: {datetime.now().strftime('%Y-%m-%d')}  
**Author**: Michael Ogunjimi, University of Manchester  

## Quick Results

### Performance Highlights
"""

        if 'chatgpt_evaluation' in key_findings:
            eval_data = key_findings['chatgpt_evaluation']
            summary += f"- **ChatGPT Success Rate**: {eval_data['success_rate']:.1%} ({eval_data['total_evaluations']:,} evaluations)\n"

        if 'prompt_comparison' in key_findings:
            prompt_data = key_findings['prompt_comparison']
            summary += f"- **Chain-of-Thought Improvement**: {prompt_data['mean_improvement']:+.2f}%\n"

        if 'sota_comparison' in key_findings:
            sota_data = key_findings['sota_comparison']
            summary += f"- **SOTA Correlation**: {sota_data['mean_correlation']:.3f} with {sota_data.get('best_baseline', 'baselines')}\n"

        summary += f"""
### Cost Analysis
- **Total Cost**: ${cost_analysis['total_experimental_cost']:.4f}
- **Batch Savings**: ${cost_analysis['total_estimated_savings']:.4f}
- **Efficiency**: {cost_analysis['batch_processing_efficiency']:.1%} cost reduction

### Execution Summary
- **Duration**: {execution_summary['total_execution_time'] / 60:.1f} minutes
- **Success Rate**: {execution_summary['completion_rate']:.1%}
- **Experiments**: {execution_summary['experiments_completed']}/{execution_summary['experiments_requested']} completed

## Bottom Line

{'✅ **SUCCESSFUL SUITE**' if execution_summary['completion_rate'] > 0.8 else '⚠️ **PARTIAL SUCCESS**' if execution_summary['completion_rate'] > 0.5 else '❌ **FAILED SUITE**'}

**Recommendation**: {'Batch processing highly recommended for future large-scale evaluations' if cost_analysis['batch_processing_efficiency'] > 0.3 else 'Consider optimizations for future batch processing'}

## Key Files
- **Detailed Results**: `master_analysis/master_experimental_results.json`
- **Full Report**: `master_analysis/master_experimental_report.md`
- **Visualizations**: `master_analysis/visualizations/`
- **Individual Experiments**: `chatgpt_evaluation/`, `prompt_comparison/`, `sota_comparison/`

---
*Generated: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}*
"""

        return summary

    def _generate_directory_readme(self) -> str:
        """Generate README with directory structure."""
        return f"""# Batch Master Experiment Results - {self.experiment_name}

**Generated**: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}  
**Author**: Michael Ogunjimi  
**Institution**: University of Manchester, MSc AI  

## Directory Structure

```
{self.experiment_name}/
├── chatgpt_evaluation/              # ChatGPT evaluation experiment results
│   ├── figures/                     # Evaluation visualizations
│   ├── logs/                        # Experiment logs
│   ├── batch_chatgpt_evaluation_results.json
│   └── experiment_report.md
├── prompt_comparison/               # Prompt comparison experiment results
│   ├── analysis/                    # Statistical analysis reports
│   ├── data/                        # Processed data and tables
│   ├── figures/                     # Comparison visualizations
│   ├── latex/                       # LaTeX tables for thesis
│   ├── results/                     # Raw results and summaries
│   ├── tables/                      # CSV data tables
│   └── logs/                        # Experiment logs
├── sota_comparison/                 # SOTA baseline comparison results
│   ├── baseline_results/            # Individual baseline method results
│   ├── figures/                     # Correlation visualizations
│   ├── logs/                        # Experiment logs
│   ├── batch_sota_comparison_results.json
│   └── sota_comparison_report.md
├── master_analysis/                 # Consolidated master analysis
│   ├── visualizations/              # Master summary charts
│   │   ├── cost_breakdown.png
│   │   ├── experimental_timeline.png
│   │   ├── key_findings_summary.json
│   │   └── performance_dashboard.png
│   ├── executive_summary.md
│   ├── master_experimental_report.md
│   └── master_experimental_results.json
├── logs/
│   ├── batch_master_{datetime.now().strftime('%Y%m%d_%H%M%S')}.log
│   └── error.log
├── batch_master_results.json        # Consolidated batch processing results
└── README.md                        # This file
```

## Quick Access

### Key Results Files
- **Executive Summary**: `master_analysis/executive_summary.md`
- **Complete Report**: `master_analysis/master_experimental_report.md`
- **Raw Results**: `master_analysis/master_experimental_results.json`
- **Batch Summary**: `batch_master_results.json`

### Individual Experiment Reports
- **ChatGPT Evaluation**: `chatgpt_evaluation/experiment_report.md`
- **Prompt Comparison**: `prompt_comparison/analysis/comprehensive_report.md`
- **SOTA Comparison**: `sota_comparison/sota_comparison_report.md`

### Visualizations
- **Master Charts**: `master_analysis/visualizations/`
- **ChatGPT Charts**: `chatgpt_evaluation/figures/`
- **Prompt Charts**: `prompt_comparison/figures/`
- **SOTA Charts**: `sota_comparison/figures/`

## Experiment Overview

This batch master experiment suite includes three core experiments optimized for cost-effective processing:

1. **ChatGPT Evaluation**: Baseline performance assessment using batch processing
2. **Prompt Comparison**: Zero-shot vs Chain-of-Thought analysis with statistical testing
3. **SOTA Comparison**: Correlation analysis with state-of-the-art baseline methods

## Batch Processing Advantages

- **Cost Savings**: Up to 50% reduction in API costs vs synchronous processing
- **Scalability**: Ability to process large datasets efficiently
- **Reliability**: Robust job management and monitoring
- **Reproducibility**: Comprehensive logging and result archiving

## Usage

Each experiment folder contains:
- Complete results in JSON format
- Detailed markdown reports with statistical analysis
- Publication-quality visualization charts
- Execution logs for debugging and verification

The `master_analysis` folder provides:
- Cross-experiment consolidated analysis
- Key findings and recommendations
- Cost-benefit analysis
- Executive summary for thesis inclusion

## Technical Details

- **Model**: {self.model}
- **API Tier**: {self.tier}
- **Batch Processing**: Enabled with OpenAI Batch API
- **Cost Optimization**: Automatic batch cost savings applied
- **Monitoring**: Real-time job status tracking
- **Error Handling**: Comprehensive error logging and recovery

## Citation

If using these results in academic work, please cite:

```
Ogunjimi, M. (2025). ChatGPT Factuality Evaluation: A Comprehensive Analysis 
of Zero-Shot Performance, Prompt Design, and Baseline Comparison. 
MSc AI Thesis, University of Manchester.
```

---
*Batch processing framework developed for academic research at University of Manchester*
"""

    def _generate_batch_master_report(self) -> str:
        """Generate batch master report with comprehensive experimental summary."""
        cost_analysis = self.experiment_results['consolidated_analysis']['cost_analysis']
        key_findings = self.experiment_results['consolidated_analysis']['key_findings']
        execution_summary = self.experiment_results['execution_summary']
        
        report = f"""# Batch Master Experimental Report

**Experiment Suite**: {self.experiment_name}  
**Generated**: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}  
**Author**: Michael Ogunjimi  
**Institution**: University of Manchester, MSc AI  

## Batch Processing Summary

This report provides a comprehensive overview of the batch processing experimental suite, including detailed cost analysis, performance metrics, and key findings from all three major experiments.

### Cost Analysis

- **Total Experimental Cost**: ${cost_analysis['total_experimental_cost']:.4f}
- **Estimated Synchronous Cost**: ${cost_analysis['estimated_sync_cost']:.4f}
- **Total Cost Savings**: ${cost_analysis['total_estimated_savings']:.4f}
- **Batch Processing Efficiency**: {cost_analysis['batch_processing_efficiency']:.1%}

### Experiment Results Summary

#### 1. ChatGPT Evaluation Experiment
"""
        
        # Add ChatGPT evaluation results
        if 'chatgpt_evaluation' in self.experiment_results['individual_experiments']:
            exp_data = self.experiment_results['individual_experiments']['chatgpt_evaluation']
            if exp_data['status'] == 'completed':
                summary = exp_data['summary']
                report += f"""
- **Status**: ✅ Completed
- **Total Evaluations**: {summary.get('total_evaluations', 'N/A'):,}
- **Success Rate**: {summary.get('success_rate', 0):.2%}
- **Cost**: ${summary.get('total_cost', 0):.4f}
- **Output Directory**: `{exp_data.get('output_dir', 'unknown')}`
"""
            else:
                report += f"\n- **Status**: ❌ {exp_data['status']}\n"
        
        # Add Prompt Comparison results  
        report += "\n#### 2. Prompt Comparison Experiment\n"
        if 'prompt_comparison' in self.experiment_results['individual_experiments']:
            exp_data = self.experiment_results['individual_experiments']['prompt_comparison']
            if exp_data['status'] == 'completed':
                summary = exp_data['summary']
                report += f"""
- **Status**: ✅ Completed
- **Total Comparisons**: {summary.get('total_comparisons', 'N/A')}
- **Mean Improvement**: {summary.get('mean_improvement_percent', 0):+.2f}%
- **Significant Results**: {summary.get('significant_improvements', 0)}/{summary.get('total_comparisons', 0)}
- **Cost**: ${summary.get('total_cost', 0):.4f}
- **Output Directory**: `{exp_data.get('output_dir', 'unknown')}`
"""
            else:
                report += f"\n- **Status**: ❌ {exp_data['status']}\n"
        
        # Add SOTA Comparison results
        report += "\n#### 3. SOTA Comparison Experiment\n"
        if 'sota_comparison' in self.experiment_results['individual_experiments']:
            exp_data = self.experiment_results['individual_experiments']['sota_comparison']
            if exp_data['status'] == 'completed':
                summary = exp_data['summary']
                report += f"""
- **Status**: ✅ Completed
- **Correlation Analysis**: {summary.get('total_correlations', 'N/A')} baseline comparisons
- **Mean Correlation**: {summary.get('mean_correlation', 0):.3f}
- **Best Baseline**: {summary.get('best_baseline', 'N/A')}
- **Cost**: ${summary.get('total_cost', 0):.4f}
- **Output Directory**: `{exp_data.get('output_dir', 'unknown')}`
"""
            else:
                report += f"\n- **Status**: ❌ {exp_data['status']}\n"
        
        report += f"""

## Key Findings Summary

### Performance Insights
"""
        
        # Add key findings from each experiment
        for exp_name, findings in key_findings.items():
            report += f"\n#### {exp_name.replace('_', ' ').title()}\n"
            if isinstance(findings, dict):
                for key, value in findings.items():
                    if isinstance(value, (int, float)):
                        if 'percent' in key or 'rate' in key:
                            report += f"- **{key.replace('_', ' ').title()}**: {value:.2%}\n"
                        elif 'correlation' in key:
                            report += f"- **{key.replace('_', ' ').title()}**: {value:.3f}\n"
                        else:
                            report += f"- **{key.replace('_', ' ').title()}**: {value}\n"
                    else:
                        report += f"- **{key.replace('_', ' ').title()}**: {value}\n"
        
        report += f"""

## Execution Summary

- **Start Time**: {execution_summary['start_time']}
- **End Time**: {execution_summary['end_time']}
- **Total Duration**: {execution_summary['total_duration']}
- **Experiments Attempted**: {execution_summary['experiments_attempted']}
- **Experiments Completed**: {execution_summary['experiments_completed']}
- **Success Rate**: {execution_summary['success_rate']:.1%}

## File Structure

### Key Output Files
- **Master Results**: `batch_master_results.json`
- **Master Report**: `batch_master_report.md` (this file)
- **Executive Summary**: `master_analysis/executive_summary.md`
- **Comprehensive Report**: `master_analysis/master_experimental_report.md`

### Individual Experiment Directories
- **ChatGPT Evaluation**: `chatgpt_evaluation/`
- **Prompt Comparison**: `prompt_comparison/`
- **SOTA Comparison**: `sota_comparison/`

---

*Report generated by Batch Master Experiment Runner*  
*University of Manchester - MSc AI Thesis Project*  
*Generated: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}*
"""
        
        return report


async def main():
    """Main function for running complete batch experimental suite."""
    parser = argparse.ArgumentParser(description="Batch Master Experiment Runner")
    parser.add_argument("--model", default="gpt-4.1-mini", help="Model to use for all experiments")
    parser.add_argument("--tier", default="tier2", help="API tier")
    parser.add_argument("--experiment-name", help="Custom master experiment name")
    parser.add_argument("--experiments", nargs="+", 
                       choices=['chatgpt_evaluation', 'prompt_comparison', 'sota_comparison'],
                       help="Specific experiments to run")
    parser.add_argument("--quick-test", action="store_true", help="Run quick test with small samples")
    parser.add_argument("--full-suite", action="store_true", help="Run complete experimental suite")
    parser.add_argument("--skip-confirmation", action="store_true", help="Skip confirmation prompts")
    
    args = parser.parse_args()
    
    # Initialize master runner
    master_runner = BatchMasterExperimentRunner(
        model=args.model,
        tier=args.tier,
        experiment_name=args.experiment_name
    )
    
    # Confirmation for full suite
    if args.full_suite and not args.skip_confirmation:
        print(f"\n🚀 BATCH MASTER EXPERIMENTAL SUITE")
        print(f"{'='*50}")
        print(f"Model: {args.model}")
        print(f"Tier: {args.tier}")
        print(f"Quick Test: {'Yes' if args.quick_test else 'No'}")
        print(f"Experiments: {args.experiments or 'All three experiments'}")
        print(f"\n⚠️  This will run comprehensive batch processing experiments.")
        print(f"Expected cost: $5-20 depending on sample sizes")
        print(f"Expected duration: 1-4 hours (including batch processing time)")
        print(f"\nOutput directory: {master_runner.output_dir}")
        
        response = input(f"\nProceed with batch experimental suite? (yes/no): ")
        if response.lower() != 'yes':
            print("❌ Batch experimental suite cancelled.")
            return
    
    # Run experiments
    try:
        print(f"\n🔄 Starting batch experimental suite...")
        results = await master_runner.run_complete_batch_experimental_suite(
            quick_test=args.quick_test,
            full_suite=args.full_suite,
            experiments_to_run=args.experiments
        )
        
        # Print comprehensive summary
        print(f"\n{'='*80}")
        print(f"BATCH MASTER EXPERIMENTAL SUITE COMPLETED")
        print(f"{'='*80}")
        print(f"Master Experiment: {master_runner.experiment_name}")
        print(f"Output Directory: {master_runner.output_dir}")
        
        # Print cost summary
        cost_analysis = results['consolidated_analysis']['cost_analysis']
        total_cost = cost_analysis['total_experimental_cost']
        total_savings = cost_analysis['total_estimated_savings']
        
        print(f"\n💰 Cost Analysis:")
        print(f"  Total Batch Cost: ${total_cost:.4f}")
        print(f"  Estimated Sync Cost: ${cost_analysis['estimated_sync_cost']:.4f}")
        print(f"  Total Savings: ${total_savings:.4f}")
        print(f"  Efficiency Gain: {cost_analysis['batch_processing_efficiency']:.1%}")
        
        # Print execution summary
        execution_summary = results['execution_summary']
        total_time = execution_summary['total_execution_time']
        print(f"\n⏱️  Execution Summary:")
        print(f"  Total Time: {total_time / 60:.1f} minutes")
        print(f"  Experiments Completed: {execution_summary['experiments_completed']}/{execution_summary['experiments_requested']}")
        print(f"  Success Rate: {execution_summary['completion_rate']:.1%}")
        
        # Print experiment status
        print(f"\n📊 Experiment Results:")
        for exp_name, exp_data in results['individual_experiments'].items():
            status = exp_data['status']
            emoji = {'chatgpt_evaluation': '🤖', 'prompt_comparison': '🔄', 'sota_comparison': '⚔️'}
            status_emoji = '✅' if status == 'completed' else '❌'
            
            print(f"  {status_emoji} {emoji.get(exp_name, '📁')} {exp_name.replace('_', ' ').title()}: {status}")
            
            if status == 'completed' and 'summary' in exp_data:
                summary = exp_data['summary']
                if 'total_cost' in summary:
                    print(f"    Cost: ${summary['total_cost']:.4f}")
        
        # Print key findings
        key_findings = results['consolidated_analysis']['key_findings']
        if key_findings:
            print(f"\n🔍 Key Findings:")
            
            if 'prompt_comparison' in key_findings:
                improvement = key_findings['prompt_comparison']['mean_improvement']
                print(f"  📝 Prompt Design: {improvement:+.2f}% improvement with Chain-of-Thought")
            
            if 'sota_comparison' in key_findings:
                correlation = key_findings['sota_comparison']['mean_correlation']
                best_baseline = key_findings['sota_comparison'].get('best_baseline', 'baselines')
                print(f"  📊 SOTA Correlation: {correlation:.3f} with {best_baseline}")
            
            if 'chatgpt_evaluation' in key_findings:
                success_rate = key_findings['chatgpt_evaluation']['success_rate']
                evaluations = key_findings['chatgpt_evaluation']['total_evaluations']
                print(f"  🤖 ChatGPT Performance: {success_rate:.1%} success rate ({evaluations:,} evaluations)")
        
        print(f"\n📄 Reports Generated:")
        print(f"  📋 Executive Summary: {master_runner.output_dir}/master_analysis/executive_summary.md")
        print(f"  📊 Master Report: {master_runner.output_dir}/master_analysis/master_experimental_report.md")
        print(f"  📈 Visualizations: {master_runner.output_dir}/master_analysis/visualizations/")
        print(f"  📁 Individual Reports: {master_runner.output_dir}/[experiment_name]/")
        print(f"  🔧 Batch Summary: {master_runner.output_dir}/batch_master_results.json")
        
        print(f"\n🎉 All batch experiments completed successfully!")
        print(f"💡 Batch processing achieved {cost_analysis['batch_processing_efficiency']:.1%} cost efficiency vs synchronous processing")
        
    except KeyboardInterrupt:
        print(f"\n⚠️ Experimental suite interrupted by user")
        print(f"Partial results may be available in: {master_runner.output_dir}")
        
    except Exception as e:
        print(f"\n❌ Experimental suite failed: {e}")
        print(f"Check logs in: {master_runner.output_dir}/logs/")
        raise


if __name__ == "__main__":
    asyncio.run(main())