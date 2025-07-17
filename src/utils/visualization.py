"""
Publication-Quality Visualization System for ChatGPT Factuality Evaluation
========================================================================

Comprehensive visualization toolkit for thesis-ready figures, including
performance comparisons, statistical analysis, cost tracking, and correlation plots.

This module provides academic-quality visualizations specifically designed
for factuality evaluation research, with emphasis on clarity, reproducibility,
and publication standards.

Author: Michael Ogunjimi
Institution: University of Manchester
Course: MSc AI
"""

import warnings
from pathlib import Path
from typing import Dict, List, Optional, Tuple, Union, Any

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import matplotlib.patches as patches
import matplotlib.colors as mcolors
import seaborn as sns
from scipy import stats
import plotly.graph_objects as go
import plotly.express as px
from plotly.subplots import make_subplots
import plotly.figure_factory as ff

from .config import get_config
from .logging import get_logger

# Suppress warnings for cleaner output
warnings.filterwarnings("ignore", category=UserWarning)
warnings.filterwarnings("ignore", category=FutureWarning)

logger = get_logger(__name__)

# Academic color palette
ACADEMIC_COLORS = {
    'primary': '#2E86AB',      # Professional blue
    'secondary': '#A23B72',    # Deep pink
    'accent': '#F18F01',       # Orange
    'success': '#C73E1D',      # Red
    'neutral': '#6B7280',      # Gray
    'light': '#F3F4F6',        # Light gray
    'background': '#FFFFFF'    # White
}

# Task-specific colors
TASK_COLORS = {
    'entailment_inference': '#2E86AB',
    'summary_ranking': '#A23B72',
    'consistency_rating': '#F18F01'
}

# Baseline colors
BASELINE_COLORS = {
    'chatgpt': '#2E86AB',
    'factcc': '#A23B72',
    'bertscore': '#F18F01',
    'rouge': '#C73E1D',
    'qags': '#6B7280'
}


class VisualizationEngine:
    """
    Main visualization engine for academic-quality figures.

    Provides comprehensive visualization capabilities specifically
    designed for factuality evaluation research and thesis publication.
    """

    def __init__(self, config: Optional[Dict] = None):
        """
        Initialize visualization engine.

        Args:
            config: Configuration dictionary
        """
        self.config = config or get_config()
        self.viz_config = self.config.get("visualization", {})

        # Output configuration
        self.output_dir = Path(self.config.get("paths.figures_dir", "./results/figures"))
        self.output_dir.mkdir(parents=True, exist_ok=True)

        self.dpi = self.viz_config.get("dpi", 300)
        self.figure_size = self.viz_config.get("figure_size", (10, 6))
        self.style = self.viz_config.get("style", "academic")

        # Setup plotting style
        self._setup_style()

        logger.info(f"Visualization engine initialized: {self.style} style, DPI={self.dpi}")

    def _setup_style(self) -> None:
        """Setup academic plotting style."""
        # Matplotlib style
        plt.rcParams.update({
            'figure.figsize': self.figure_size,
            'figure.dpi': self.dpi,
            'savefig.dpi': self.dpi,
            'savefig.bbox': 'tight',
            'savefig.pad_inches': 0.1,
            'font.size': 12,
            'axes.titlesize': 14,
            'axes.labelsize': 12,
            'xtick.labelsize': 10,
            'ytick.labelsize': 10,
            'legend.fontsize': 10,
            'figure.titlesize': 16,
            'axes.grid': True,
            'grid.alpha': 0.3,
            'axes.spines.top': False,
            'axes.spines.right': False,
            'font.family': 'serif'
        })

        # Seaborn style
        sns.set_palette("husl")
        sns.set_style("whitegrid")

    def save_figure(
        self,
        fig,
        filename: str,
        formats: List[str] = None,
        tight_layout: bool = True
    ) -> Dict[str, Path]:
        """
        Save figure in multiple formats.

        Args:
            fig: Matplotlib or Plotly figure
            filename: Base filename (without extension)
            formats: List of formats to save ('png', 'pdf', 'svg', 'html')
            tight_layout: Apply tight layout for matplotlib figures

        Returns:
            Dictionary mapping formats to saved file paths
        """
        formats = formats or self.viz_config.get("formats", ["png", "pdf"])
        saved_files = {}

        # Handle matplotlib figures
        if hasattr(fig, 'savefig'):
            if tight_layout:
                try:
                    fig.tight_layout()
                except:
                    pass  # Ignore tight_layout errors

            for fmt in formats:
                if fmt in ['png', 'pdf', 'svg', 'eps']:
                    filepath = self.output_dir / f"{filename}.{fmt}"
                    fig.savefig(filepath, format=fmt, dpi=self.dpi, bbox_inches='tight')
                    saved_files[fmt] = filepath
                    logger.debug(f"Saved figure: {filepath}")

        # Handle plotly figures
        elif hasattr(fig, 'write_html'):
            for fmt in formats:
                if fmt == 'html':
                    filepath = self.output_dir / f"{filename}.html"
                    fig.write_html(filepath)
                    saved_files[fmt] = filepath
                elif fmt == 'png':
                    filepath = self.output_dir / f"{filename}.png"
                    fig.write_image(filepath, width=800, height=600)
                    saved_files[fmt] = filepath
                elif fmt == 'pdf':
                    filepath = self.output_dir / f"{filename}.pdf"
                    fig.write_image(filepath, width=800, height=600)
                    saved_files[fmt] = filepath

        return saved_files


class TaskPerformanceVisualizer:
    """Specialized visualizer for task performance analysis."""

    def __init__(self, viz_engine: VisualizationEngine):
        self.viz_engine = viz_engine

    def plot_task_comparison(
        self,
        results: Dict[str, Dict[str, Any]],
        title: str = "ChatGPT Task Performance Comparison",
        save_name: Optional[str] = None
    ) -> plt.Figure:
        """
        Create task performance comparison plot.

        Args:
            results: Dictionary mapping task names to performance metrics
            title: Plot title
            save_name: Optional filename for saving

        Returns:
            Matplotlib figure
        """
        fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(15, 7))
        fig.suptitle(title, fontsize=16, fontweight='bold', y=0.98)

        tasks = list(results.keys())
        if not tasks:
            # Create empty plot with message
            ax1.text(0.5, 0.5, 'No task results available', ha='center', va='center', 
                    transform=ax1.transAxes, fontsize=12)
            ax1.set_title('Performance Metrics')
            ax2.text(0.5, 0.5, 'No cost data available', ha='center', va='center', 
                    transform=ax2.transAxes, fontsize=12)
            ax2.set_title('Cost Analysis')
            return fig

        # Extract primary metrics (the main performance score)
        primary_metrics = []
        costs = []
        times = []
        
        for task in tasks:
            task_data = results[task]
            # Get primary metric
            primary_metric = task_data.get('primary_metric', 0)
            primary_metrics.append(primary_metric)
            
            # Get cost and time data
            costs.append(task_data.get('cost', 0))
            times.append(task_data.get('processing_time', 0))

        # Performance comparison
        colors = [TASK_COLORS.get(task, ACADEMIC_COLORS['primary']) for task in tasks]
        bars = ax1.bar(tasks, primary_metrics, color=colors, alpha=0.7)
        ax1.set_ylabel('Performance Score')
        ax1.set_title('Task Performance Comparison')
        ax1.set_ylim(0, 1.1)
        ax1.grid(axis='y', alpha=0.3)
        
        # Add value labels on bars
        for bar, value in zip(bars, primary_metrics):
            height = bar.get_height()
            ax1.text(bar.get_x() + bar.get_width()/2., height + 0.01,
                    f'{value:.3f}', ha='center', va='bottom', fontweight='bold')

        # Clean up task names for display
        clean_names = [task.replace('_', ' ').title() for task in tasks]
        ax1.set_xticklabels(clean_names, rotation=0, ha='center')

        # Cost vs Time analysis
        if any(costs) or any(times):
            ax2_twin = ax2.twinx()
            
            bars1 = ax2.bar(np.arange(len(tasks)) - 0.2, costs, 0.4, 
                           label='Cost ($)', color=ACADEMIC_COLORS['accent'], alpha=0.7)
            bars2 = ax2_twin.bar(np.arange(len(tasks)) + 0.2, times, 0.4, 
                                label='Time (s)', color=ACADEMIC_COLORS['secondary'], alpha=0.7)
            
            ax2.set_ylabel('Cost ($)', color=ACADEMIC_COLORS['accent'])
            ax2_twin.set_ylabel('Time (s)', color=ACADEMIC_COLORS['secondary'])
            ax2.set_title('Cost and Time Analysis')
            ax2.set_xticks(np.arange(len(tasks)))
            ax2.set_xticklabels(clean_names, rotation=0, ha='center')
            ax2.grid(axis='y', alpha=0.3)
            
            # Add legends
            ax2.legend(loc='upper left')
            ax2_twin.legend(loc='upper right')
        else:
            ax2.text(0.5, 0.5, 'No cost/time data available', ha='center', va='center', 
                    transform=ax2.transAxes, fontsize=12)
            ax2.set_title('Cost and Time Analysis')

        # Better spacing for labels and title
        plt.tight_layout(rect=[0, 0.03, 1, 0.95])

        if save_name:
            self.viz_engine.save_figure(fig, save_name)

        return fig

    def plot_prompt_comparison(
        self,
        zero_shot_results: Dict[str, float],
        cot_results: Dict[str, float],
        title: str = "Zero-Shot vs Chain-of-Thought Comparison",
        save_name: Optional[str] = None
    ) -> plt.Figure:
        """
        Compare zero-shot vs chain-of-thought prompt performance.

        Args:
            zero_shot_results: Zero-shot performance results
            cot_results: Chain-of-thought performance results
            title: Plot title
            save_name: Optional filename for saving

        Returns:
            Matplotlib figure
        """
        fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(15, 7))
        fig.suptitle(title, fontsize=16, fontweight='bold', y=0.98)

        tasks = list(zero_shot_results.keys())

        # Side-by-side comparison
        x = np.arange(len(tasks))
        width = 0.35

        zero_shot_scores = [zero_shot_results[task] for task in tasks]
        cot_scores = [cot_results[task] for task in tasks]

        bars1 = ax1.bar(x - width/2, zero_shot_scores, width, label='Zero-Shot',
                       color=ACADEMIC_COLORS['primary'], alpha=0.8)
        bars2 = ax1.bar(x + width/2, cot_scores, width, label='Chain-of-Thought',
                       color=ACADEMIC_COLORS['secondary'], alpha=0.8)

        ax1.set_xlabel('Tasks')
        ax1.set_ylabel('Performance Score')
        ax1.set_title('Performance Comparison')
        ax1.set_xticks(x)
        ax1.set_xticklabels([task.replace('_', ' ').title() for task in tasks], rotation=0)
        ax1.legend()
        ax1.set_ylim(0, 1)

        # Add value labels on bars
        for bars in [bars1, bars2]:
            for bar in bars:
                height = bar.get_height()
                ax1.annotate(f'{height:.3f}',
                           xy=(bar.get_x() + bar.get_width() / 2, height),
                           xytext=(0, 3), textcoords="offset points",
                           ha='center', va='bottom', fontsize=9)

        # Improvement analysis
        improvements = [(cot_scores[i] - zero_shot_scores[i]) for i in range(len(tasks))]
        improvement_pct = [(improvements[i] / zero_shot_scores[i]) * 100
                          if zero_shot_scores[i] > 0 else 0 for i in range(len(tasks))]

        colors = [ACADEMIC_COLORS['success'] if imp > 0 else ACADEMIC_COLORS['accent']
                 for imp in improvements]

        bars = ax2.bar(tasks, improvement_pct, color=colors, alpha=0.7)
        ax2.axhline(y=0, color='black', linestyle='-', alpha=0.3)
        ax2.set_ylabel('Improvement (%)', rotation=0)
        ax2.set_title('Chain-of-Thought Improvement over Zero-Shot')
        ax2.set_xticklabels([task.replace('_', ' ').title() for task in tasks], rotation=0, ha='center')

        # Add value labels
        for bar, pct in zip(bars, improvement_pct):
            height = bar.get_height()
            ax2.annotate(f'{pct:.1f}%',
                        xy=(bar.get_x() + bar.get_width() / 2, height),
                        xytext=(0, 3 if height >= 0 else -15),
                        textcoords="offset points",
                        ha='center', va='bottom' if height >= 0 else 'top',
                        fontsize=9)

        # Better spacing for labels and title
        plt.tight_layout(rect=[0, 0.03, 1, 0.95])

        if save_name:
            self.viz_engine.save_figure(fig, save_name)

        return fig


class BaselineComparisonVisualizer:
    """Specialized visualizer for baseline comparison analysis."""

    def __init__(self, viz_engine: VisualizationEngine):
        self.viz_engine = viz_engine

    def plot_correlation_analysis(
        self,
        chatgpt_scores: List[float],
        baseline_scores: Dict[str, List[float]],
        task_name: str,
        title: Optional[str] = None,
        save_name: Optional[str] = None
    ) -> plt.Figure:
        """
        Create correlation analysis plots between ChatGPT and baselines.

        Args:
            chatgpt_scores: ChatGPT scores
            baseline_scores: Dictionary mapping baseline names to scores
            task_name: Name of the task
            title: Plot title
            save_name: Optional filename for saving

        Returns:
            Matplotlib figure
        """
        n_baselines = len(baseline_scores)
        cols = min(3, n_baselines)
        rows = (n_baselines + cols - 1) // cols

        fig, axes = plt.subplots(rows, cols, figsize=(5*cols, 4*rows + 1))
        if n_baselines == 1:
            axes = [axes]
        elif rows == 1:
            axes = [axes]
        else:
            axes = axes.flatten()

        title = title or f"ChatGPT vs Baseline Correlations: {task_name.replace('_', ' ').title()}"
        fig.suptitle(title, fontsize=16, fontweight='bold', y=0.98)

        correlation_results = {}

        for i, (baseline_name, baseline_vals) in enumerate(baseline_scores.items()):
            ax = axes[i]

            # Calculate correlations
            pearson_r, pearson_p = stats.pearsonr(chatgpt_scores, baseline_vals)
            spearman_r, spearman_p = stats.spearmanr(chatgpt_scores, baseline_vals)

            correlation_results[baseline_name] = {
                'pearson_r': pearson_r,
                'pearson_p': pearson_p,
                'spearman_r': spearman_r,
                'spearman_p': spearman_p
            }

            # Scatter plot
            color = BASELINE_COLORS.get(baseline_name, ACADEMIC_COLORS['neutral'])
            ax.scatter(chatgpt_scores, baseline_vals, alpha=0.6, color=color, s=30)

            # Fit line
            z = np.polyfit(chatgpt_scores, baseline_vals, 1)
            p = np.poly1d(z)
            ax.plot(chatgpt_scores, p(chatgpt_scores), "r--", alpha=0.8, linewidth=1)

            # Labels and title
            ax.set_xlabel('ChatGPT Scores')
            ax.set_ylabel(f'{baseline_name.upper()} Scores')
            ax.set_title(f'{baseline_name.upper()}\nPearson r={pearson_r:.3f} (p={pearson_p:.3f})')
            ax.grid(True, alpha=0.3)

            # Add correlation text
            textstr = f'Spearman ρ={spearman_r:.3f}\np={spearman_p:.3f}'
            props = dict(boxstyle='round', facecolor='wheat', alpha=0.5)
            ax.text(0.05, 0.95, textstr, transform=ax.transAxes, fontsize=8,
                   verticalalignment='top', bbox=props)

        # Hide empty subplots
        for i in range(n_baselines, len(axes)):
            axes[i].set_visible(False)

        # Better spacing for labels and title
        plt.tight_layout(rect=[0, 0.03, 1, 0.95])

        if save_name:
            self.viz_engine.save_figure(fig, save_name)

        return fig, correlation_results

    def plot_agreement_matrix(
        self,
        results_dict: Dict[str, List[int]],
        title: str = "Inter-Method Agreement Matrix",
        save_name: Optional[str] = None
    ) -> plt.Figure:
        """
        Create agreement matrix heatmap between methods.

        Args:
            results_dict: Dictionary mapping method names to binary predictions
            title: Plot title
            save_name: Optional filename for saving

        Returns:
            Matplotlib figure
        """
        methods = list(results_dict.keys())
        n_methods = len(methods)

        # Calculate agreement matrix
        agreement_matrix = np.zeros((n_methods, n_methods))

        for i, method1 in enumerate(methods):
            for j, method2 in enumerate(methods):
                if i == j:
                    agreement_matrix[i, j] = 1.0
                else:
                    pred1 = np.array(results_dict[method1])
                    pred2 = np.array(results_dict[method2])
                    agreement = np.mean(pred1 == pred2)
                    agreement_matrix[i, j] = agreement

        # Create heatmap
        fig, ax = plt.subplots(figsize=(8, 6))
        im = ax.imshow(agreement_matrix, cmap='RdYlBu_r', vmin=0, vmax=1)

        # Set ticks and labels
        ax.set_xticks(np.arange(n_methods))
        ax.set_yticks(np.arange(n_methods))
        ax.set_xticklabels([m.upper() for m in methods])
        ax.set_yticklabels([m.upper() for m in methods])

        # Rotate the tick labels and set their alignment
        plt.setp(ax.get_xticklabels(), rotation=0, ha="center", rotation_mode="anchor")

        # Add text annotations
        for i in range(n_methods):
            for j in range(n_methods):
                text = ax.text(j, i, f'{agreement_matrix[i, j]:.3f}',
                             ha="center", va="center", color="black", fontweight='bold')

        ax.set_title(title, fontsize=14, fontweight='bold')

        # Add colorbar
        cbar = plt.colorbar(im, ax=ax, shrink=0.8)
        cbar.set_label('Agreement Rate', rotation=270, labelpad=15)

        plt.tight_layout()

        if save_name:
            self.viz_engine.save_figure(fig, save_name)

        return fig


class StatisticalAnalysisVisualizer:
    """Specialized visualizer for statistical analysis plots."""

    def __init__(self, viz_engine: VisualizationEngine):
        self.viz_engine = viz_engine

    def plot_significance_analysis(
        self,
        comparison_results: Dict[str, Dict[str, Any]],
        title: str = "Statistical Significance Analysis",
        save_name: Optional[str] = None
    ) -> plt.Figure:
        """
        Create statistical significance analysis visualization.

        Args:
            comparison_results: Results from statistical comparisons
            title: Plot title
            save_name: Optional filename for saving

        Returns:
            Matplotlib figure
        """
        fig, ((ax1, ax2), (ax3, ax4)) = plt.subplots(2, 2, figsize=(15, 11))
        fig.suptitle(title, fontsize=16, fontweight='bold', y=0.98)

        methods = list(comparison_results.keys())

        # P-values plot
        p_values = [comparison_results[method].get('p_value', 1.0) for method in methods]
        significance_threshold = 0.05

        colors = [ACADEMIC_COLORS['success'] if p < significance_threshold
                 else ACADEMIC_COLORS['neutral'] for p in p_values]

        bars = ax1.bar(methods, p_values, color=colors, alpha=0.7)
        ax1.axhline(y=significance_threshold, color='red', linestyle='--',
                   label=f'Significance threshold (p={significance_threshold})')
        ax1.set_ylabel('P-value')
        ax1.set_title('Statistical Significance (P-values)')
        ax1.set_yscale('log')
        ax1.legend()
        ax1.tick_params(axis='x', rotation=0)

        # Effect sizes plot
        effect_sizes = [comparison_results[method].get('effect_size', 0.0) for method in methods]

        ax2.bar(methods, effect_sizes, color=ACADEMIC_COLORS['primary'], alpha=0.7)
        ax2.set_ylabel('Effect Size (Cohen\'s d)')
        ax2.set_title('Effect Sizes')
        ax2.tick_params(axis='x', rotation=0)

        # Add effect size interpretation lines
        ax2.axhline(y=0.2, color='green', linestyle=':', alpha=0.7, label='Small (0.2)')
        ax2.axhline(y=0.5, color='orange', linestyle=':', alpha=0.7, label='Medium (0.5)')
        ax2.axhline(y=0.8, color='red', linestyle=':', alpha=0.7, label='Large (0.8)')
        ax2.legend()

        # Confidence intervals plot
        correlations = [comparison_results[method].get('correlation', 0.0) for method in methods]
        ci_lower = [comparison_results[method].get('ci_lower', 0.0) for method in methods]
        ci_upper = [comparison_results[method].get('ci_upper', 0.0) for method in methods]

        x_pos = np.arange(len(methods))
        ax3.errorbar(x_pos, correlations, yerr=[np.array(correlations) - np.array(ci_lower),
                                               np.array(ci_upper) - np.array(correlations)],
                    fmt='o', capsize=5, color=ACADEMIC_COLORS['primary'])
        ax3.set_xticks(x_pos)
        ax3.set_xticklabels(methods, rotation=0, ha='center')
        ax3.set_ylabel('Correlation')
        ax3.set_title('Correlations with Confidence Intervals')
        ax3.grid(True, alpha=0.3)

        # Power analysis plot (if available)
        power_values = [comparison_results[method].get('statistical_power', 0.8) for method in methods]

        ax4.bar(methods, power_values, color=ACADEMIC_COLORS['accent'], alpha=0.7)
        ax4.axhline(y=0.8, color='red', linestyle='--', label='Recommended power (0.8)')
        ax4.set_ylabel('Statistical Power')
        ax4.set_title('Statistical Power Analysis')
        ax4.tick_params(axis='x', rotation=0)
        ax4.legend()
        ax4.set_ylim(0, 1)

        # Better spacing for labels and title
        plt.tight_layout(rect=[0, 0.03, 1, 0.95])

        if save_name:
            self.viz_engine.save_figure(fig, save_name)

        return fig


class CostAnalysisVisualizer:
    """Specialized visualizer for cost and resource analysis."""

    def __init__(self, viz_engine: VisualizationEngine):
        self.viz_engine = viz_engine

    def plot_cost_breakdown(
        self,
        cost_data: Dict[str, Any],
        title: str = "API Cost Analysis",
        save_name: Optional[str] = None
    ) -> plt.Figure:
        """
        Create comprehensive cost breakdown visualization.

        Args:
            cost_data: Cost analysis data
            title: Plot title
            save_name: Optional filename for saving

        Returns:
            Matplotlib figure
        """
        fig, ((ax1, ax2), (ax3, ax4)) = plt.subplots(2, 2, figsize=(15, 11))
        fig.suptitle(title, fontsize=16, fontweight='bold', y=0.98)

        # Cost by model pie chart
        model_costs = cost_data.get('by_model', {})
        if model_costs:
            ax1.pie(model_costs.values(), labels=model_costs.keys(), autopct='%1.1f%%',
                   colors=list(ACADEMIC_COLORS.values())[:len(model_costs)])
            ax1.set_title('Cost Distribution by Model')

        # Cost by task bar chart
        task_costs = cost_data.get('by_task', {})
        if task_costs:
            ax2.bar(task_costs.keys(), task_costs.values(),
                   color=list(TASK_COLORS.values())[:len(task_costs)], alpha=0.7)
            ax2.set_ylabel('Cost ($)')
            ax2.set_title('Cost by Task')
            ax2.tick_params(axis='x', rotation=0)

        # Cost over time
        cost_history = cost_data.get('history', [])
        if cost_history:
            timestamps = [entry['timestamp'] for entry in cost_history[-50:]]  # Last 50
            running_totals = [entry['running_total'] for entry in cost_history[-50:]]

            ax3.plot(range(len(running_totals)), running_totals,
                    color=ACADEMIC_COLORS['primary'], linewidth=2)
            ax3.set_xlabel('API Calls')
            ax3.set_ylabel('Cumulative Cost ($)')
            ax3.set_title('Cost Accumulation Over Time')
            ax3.grid(True, alpha=0.3)

        # Budget utilization
        summary = cost_data.get('summary', {})
        total_budget = summary.get('total_budget', 100)
        total_spent = summary.get('total_spent', 0)
        daily_budget = summary.get('daily_budget', 20)
        daily_spent = summary.get('daily_spent', 0)

        categories = ['Total Budget', 'Daily Budget']
        spent = [total_spent, daily_spent]
        budgets = [total_budget, daily_budget]
        remaining = [budgets[i] - spent[i] for i in range(len(budgets))]

        x = np.arange(len(categories))
        width = 0.35

        ax4.bar(x, spent, width, label='Spent', color=ACADEMIC_COLORS['accent'], alpha=0.8)
        ax4.bar(x, remaining, width, bottom=spent, label='Remaining',
               color=ACADEMIC_COLORS['neutral'], alpha=0.5)

        ax4.set_ylabel('Amount ($)')
        ax4.set_title('Budget Utilization')
        ax4.set_xticks(x)
        ax4.set_xticklabels(categories)
        ax4.legend()

        # Add percentage labels
        for i, (s, b) in enumerate(zip(spent, budgets)):
            percentage = (s / b) * 100 if b > 0 else 0
            ax4.text(i, s/2, f'{percentage:.1f}%', ha='center', va='center',
                    fontweight='bold', color='white')

        # Better spacing for labels and title
        plt.tight_layout(rect=[0, 0.03, 1, 0.95])

        if save_name:
            self.viz_engine.save_figure(fig, save_name)

        return fig


def create_visualization_engine(config: Optional[Dict] = None) -> VisualizationEngine:
    """
    Factory function to create visualization engine.

    Args:
        config: Configuration dictionary

    Returns:
        Visualization engine instance
    """
    return VisualizationEngine(config)


def create_dashboard_figure(
    task_results: Dict[str, Any],
    baseline_results: Dict[str, Any],
    cost_data: Optional[Dict[str, Any]] = None,
    title: str = "ChatGPT Factuality Evaluation Dashboard"
) -> go.Figure:
    """
    Create an interactive Plotly dashboard figure.

    Args:
        task_results: Task performance results
        baseline_results: Baseline comparison results
        cost_data: Optional cost analysis data
        title: Dashboard title

    Returns:
        Plotly figure with interactive dashboard
    """
    # Create subplots
    fig = make_subplots(
        rows=2, cols=2,
        subplot_titles=('Task Performance', 'Baseline Correlations',
                       'Cost Analysis', 'Statistical Significance'),
        specs=[[{"type": "bar"}, {"type": "scatter"}],
               [{"type": "pie"}, {"type": "bar"}]]
    )

    # Task performance bar chart
    tasks = list(task_results.keys())
    performance_scores = [task_results[task].get('f1_score', 0) for task in tasks]

    fig.add_trace(
        go.Bar(x=tasks, y=performance_scores, name='F1 Score',
               marker_color=list(TASK_COLORS.values())[:len(tasks)]),
        row=1, col=1
    )

    # Baseline correlations scatter plot
    if baseline_results:
        for baseline_name, results in baseline_results.items():
            correlation = results.get('correlation', 0)
            p_value = results.get('p_value', 1)

            fig.add_trace(
                go.Scatter(x=[baseline_name], y=[correlation], mode='markers',
                          marker=dict(size=max(10, -np.log10(p_value) * 5),
                                    color=BASELINE_COLORS.get(baseline_name, '#666666')),
                          name=f'{baseline_name} (r={correlation:.3f})'),
                row=1, col=2
            )

    # Cost analysis pie chart
    if cost_data and 'by_model' in cost_data:
        model_costs = cost_data['by_model']
        fig.add_trace(
            go.Pie(labels=list(model_costs.keys()), values=list(model_costs.values()),
                  name="Model Costs"),
            row=2, col=1
        )

    # Statistical significance
    p_values = [baseline_results[baseline].get('p_value', 1.0)
               for baseline in baseline_results.keys()]
    baselines = list(baseline_results.keys())

    colors = ['green' if p < 0.05 else 'red' for p in p_values]

    fig.add_trace(
        go.Bar(x=baselines, y=p_values, name='P-values',
               marker_color=colors),
        row=2, col=2
    )

    # Update layout
    fig.update_layout(
        title_text=title,
        showlegend=True,
        height=800,
        template='plotly_white'
    )

    return fig


# Convenience function for quick plotting
def quick_plot_comparison(
    data: Dict[str, List[float]],
    title: str = "Performance Comparison",
    save_name: Optional[str] = None
) -> plt.Figure:
    """
    Quick comparison plot for development and testing.

    Args:
        data: Dictionary mapping method names to score lists
        title: Plot title
        save_name: Optional filename for saving

    Returns:
        Matplotlib figure
    """
    fig, ax = plt.subplots(figsize=(10, 6))

    methods = list(data.keys())
    positions = np.arange(len(methods))

    # Create box plot
    box_data = [data[method] for method in methods]
    bp = ax.boxplot(box_data, positions=positions, patch_artist=True)

    # Color the boxes
    colors = list(ACADEMIC_COLORS.values())[:len(methods)]
    for patch, color in zip(bp['boxes'], colors):
        patch.set_facecolor(color)
        patch.set_alpha(0.7)

    ax.set_xticklabels([method.replace('_', ' ').title() for method in methods])
    ax.set_ylabel('Performance Score', rotation=0)
    ax.set_title(title)
    ax.grid(True, alpha=0.3)

    if save_name:
        viz_engine = create_visualization_engine()
        viz_engine.save_figure(fig, save_name)

    return fig


class EssentialRatingVisualizer:
    """Specialized visualizer for essential rating distributions."""

    def __init__(self, viz_engine: VisualizationEngine):
        self.viz_engine = viz_engine

    def plot_rating_distribution(
        self,
        ratings: List[float],
        task_name: str,
        title: Optional[str] = None,
        save_name: Optional[str] = None
    ) -> plt.Figure:
        """
        Create rating distribution plot for consistency rating task.

        Args:
            ratings: List of consistency ratings (0-100)
            task_name: Name of the task
            title: Plot title
            save_name: Optional filename for saving

        Returns:
            Matplotlib figure
        """
        fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(12, 6))
        
        title = title or f"Rating Distribution: {task_name.replace('_', ' ').title()}"
        fig.suptitle(title, fontsize=14, fontweight='bold', y=0.98)

        # Histogram
        ax1.hist(ratings, bins=20, color=ACADEMIC_COLORS['primary'], alpha=0.7, edgecolor='black')
        ax1.set_xlabel('Consistency Rating')
        ax1.set_ylabel('Frequency', rotation=0)
        ax1.set_title('Rating Distribution')
        ax1.axvline(np.mean(ratings), color='red', linestyle='--', linewidth=2, label=f'Mean: {np.mean(ratings):.1f}')
        ax1.legend()
        ax1.grid(True, alpha=0.3)

        # Box plot with statistics
        bp = ax2.boxplot(ratings, patch_artist=True, labels=[task_name.replace('_', ' ').title()])
        bp['boxes'][0].set_facecolor(ACADEMIC_COLORS['primary'])
        bp['boxes'][0].set_alpha(0.7)
        
        ax2.set_ylabel('Consistency Rating', rotation=0)
        ax2.set_title('Rating Statistics')
        ax2.grid(True, alpha=0.3)
        
        # Add statistics text
        stats_text = f'Mean: {np.mean(ratings):.1f}\nStd: {np.std(ratings):.1f}\nMin: {np.min(ratings):.1f}\nMax: {np.max(ratings):.1f}'
        ax2.text(0.02, 0.98, stats_text, transform=ax2.transAxes, fontsize=10,
                verticalalignment='top', bbox=dict(boxstyle='round', facecolor='wheat', alpha=0.8))

        # Better spacing for labels and title
        plt.tight_layout(rect=[0, 0.03, 1, 0.95])

        if save_name:
            self.viz_engine.save_figure(fig, save_name)

        return fig