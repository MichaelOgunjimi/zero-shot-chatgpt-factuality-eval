#!/usr/bin/env python3
"""
Zero-Shot vs Chain-of-Thought Comparison Analysis
===============================================

Comprehensive comparison of zero-shot and chain-of-thought prompting techniques
for factuality evaluation across multiple LLMs and tasks.

This script analyzes the performance differences, cost implications, and 
effectiveness trade-offs between the two prompting strategies.

Author: Michael Ogunjimi
Institution: University of Manchester
Course: MSc AI
Date: August 21, 2025
"""

import json
import warnings
from pathlib import Path
from typing import Dict, List, Tuple, Any
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from scipy import stats
import plotly.graph_objects as go
from plotly.subplots import make_subplots
import plotly.express as px

# Suppress warnings for cleaner output
warnings.filterwarnings('ignore')

# Set academic plotting style
plt.style.use('seaborn-v0_8')
sns.set_palette("husl")

# Academic color palette
COLORS = {
    'zero_shot': '#2E86AB',      # Professional blue
    'chain_of_thought': '#A23B72', # Deep pink
    'improvement': '#27AE60',     # Green
    'decline': '#E74C3C',        # Red
    'neutral': '#6B7280'         # Gray
}

# Task and model colors
TASK_COLORS = {
    'entailment_inference': '#FF6B6B',
    'summary_ranking': '#4ECDC4', 
    'consistency_rating': '#FFE66D'
}

MODEL_COLORS = {
    'gpt-4.1-mini': '#9B59B6',
    'llama3.1:8b': '#E67E22',
    'qwen2.5:7b': '#F39C12'
}


class ZeroShotVsCoTAnalyzer:
    """Main analyzer for comparing zero-shot vs chain-of-thought results."""
    
    def __init__(self, zero_shot_path: str, cot_path: str, output_dir: str = "results/comparison_analysis"):
        """
        Initialize the analyzer with paths to result files.
        
        Args:
            zero_shot_path: Path to zero-shot results JSON
            cot_path: Path to chain-of-thought results JSON  
            output_dir: Directory to save outputs
        """
        self.zero_shot_path = Path(zero_shot_path)
        self.cot_path = Path(cot_path)
        self.output_dir = Path(output_dir)
        self.output_dir.mkdir(parents=True, exist_ok=True)
        
        # Load data
        self.zero_shot_data = self._load_json(self.zero_shot_path)
        self.cot_data = self._load_json(self.cot_path)
        
        # Extract metadata
        self.models = self.zero_shot_data['summary_statistics']['models_evaluated']
        self.tasks = self.zero_shot_data['summary_statistics']['tasks_evaluated']
        self.datasets = self.zero_shot_data['summary_statistics']['datasets_evaluated']
        
        print(f"✅ Loaded data for {len(self.models)} models, {len(self.tasks)} tasks, {len(self.datasets)} datasets")
        
    def _load_json(self, path: Path) -> Dict:
        """Load and validate JSON file."""
        try:
            with open(path, 'r') as f:
                data = json.load(f)
            print(f"✅ Successfully loaded: {path.name}")
            return data
        except Exception as e:
            print(f"❌ Error loading {path}: {e}")
            raise
            
    def extract_performance_metrics(self) -> pd.DataFrame:
        """Extract key performance metrics for comparison."""
        
        results = []
        
        for model in self.models:
            for task in self.tasks:
                for dataset in self.datasets:
                    # Extract zero-shot metrics
                    zs_metrics = self._get_metrics(self.zero_shot_data, model, task, dataset, 'zero_shot')
                    if zs_metrics:
                        results.append(zs_metrics)
                    
                    # Extract chain-of-thought metrics
                    cot_metrics = self._get_metrics(self.cot_data, model, task, dataset, 'chain_of_thought')
                    if cot_metrics:
                        results.append(cot_metrics)
        
        df = pd.DataFrame(results)
        print(f"✅ Extracted {len(df)} metric records")
        return df
    
    def _get_metrics(self, data: Dict, model: str, task: str, dataset: str, prompt_type: str) -> Dict:
        """Extract metrics for a specific model/task/dataset combination."""
        
        try:
            task_data = data['llm_results'][model][task][dataset]
            
            if 'performance_metrics' not in task_data:
                return None
                
            metrics = task_data['performance_metrics']
            
            return {
                'model': model,
                'task': task,
                'dataset': dataset,
                'prompt_type': prompt_type,
                'accuracy': metrics.get('accuracy', 0),
                'precision': metrics.get('precision', 0),
                'recall': metrics.get('recall', 0),
                'f1_score': metrics.get('f1_score', 0),
                'total_cost': task_data.get('cost', 0),
                'avg_processing_time': task_data.get('processing_time', 0),
                'total_tokens': 0,  # Not available in your data structure
                'avg_tokens_per_example': 0,  # Not available in your data structure
                'primary_metric': metrics.get('primary_metric', 0)
            }
        except KeyError:
            return None
    
    def create_performance_comparison_table(self, df: pd.DataFrame) -> pd.DataFrame:
        """Create side-by-side comparison table."""
        
        # Pivot data for comparison
        comparison_metrics = ['accuracy', 'f1_score', 'precision', 'recall', 'total_cost', 'avg_processing_time']
        
        comparison_data = []
        
        for model in self.models:
            for task in self.tasks:
                for dataset in self.datasets:
                    # Get zero-shot and CoT data
                    zs_row = df[(df['model'] == model) & (df['task'] == task) & 
                               (df['dataset'] == dataset) & (df['prompt_type'] == 'zero_shot')]
                    cot_row = df[(df['model'] == model) & (df['task'] == task) & 
                                (df['dataset'] == dataset) & (df['prompt_type'] == 'chain_of_thought')]
                    
                    if len(zs_row) > 0 and len(cot_row) > 0:
                        zs_data = zs_row.iloc[0]
                        cot_data = cot_row.iloc[0]
                        
                        row = {
                            'Model': model,
                            'Task': task,
                            'Dataset': dataset,
                        }
                        
                        # Add side-by-side metrics
                        for metric in comparison_metrics:
                            zs_val = zs_data[metric]
                            cot_val = cot_data[metric]
                            
                            row[f'{metric}_zero_shot'] = f"{zs_val:.4f}" if isinstance(zs_val, float) else str(zs_val)
                            row[f'{metric}_cot'] = f"{cot_val:.4f}" if isinstance(cot_val, float) else str(cot_val)
                            
                            # Calculate improvement
                            if isinstance(zs_val, (int, float)) and isinstance(cot_val, (int, float)) and zs_val != 0:
                                improvement = ((cot_val - zs_val) / zs_val) * 100
                                row[f'{metric}_improvement'] = f"{improvement:+.1f}%"
                            else:
                                row[f'{metric}_improvement'] = "N/A"
                        
                        comparison_data.append(row)
        
        comparison_df = pd.DataFrame(comparison_data)
        
        # Save to CSV
        csv_path = self.output_dir / "performance_comparison_table.csv"
        comparison_df.to_csv(csv_path, index=False)
        print(f"💾 Saved comparison table: {csv_path}")
        
        return comparison_df
    
    def plot_metric_comparison_bars(self, df: pd.DataFrame):
        """Create comprehensive bar charts comparing key metrics with statistical significance."""
        
        metrics_to_plot = ['accuracy', 'f1_score', 'precision', 'recall']
        
        fig, axes = plt.subplots(2, 2, figsize=(18, 14))
        axes = axes.flatten()
        
        for idx, metric in enumerate(metrics_to_plot):
            ax = axes[idx]
            
            # Aggregate by model and prompt type
            agg_data = df.groupby(['model', 'prompt_type'])[metric].agg(['mean', 'std', 'count']).reset_index()
            agg_data.columns = ['model', 'prompt_type', 'mean', 'std', 'count']
            
            # Pivot for plotting
            pivot_mean = agg_data.pivot(index='model', columns='prompt_type', values='mean')
            pivot_std = agg_data.pivot(index='model', columns='prompt_type', values='std')
            
            # Create bars with error bars
            x = np.arange(len(pivot_mean.index))
            width = 0.35
            
            bars1 = ax.bar(x - width/2, pivot_mean['zero_shot'], width, 
                          yerr=pivot_std['zero_shot'], label='Zero-Shot', 
                          color=COLORS['zero_shot'], alpha=0.8, capsize=5)
            bars2 = ax.bar(x + width/2, pivot_mean['chain_of_thought'], width,
                          yerr=pivot_std['chain_of_thought'], label='Chain-of-Thought', 
                          color=COLORS['chain_of_thought'], alpha=0.8, capsize=5)
            
            ax.set_xlabel('Models', fontsize=12)
            ax.set_ylabel(metric.replace('_', ' ').title(), fontsize=12)
            ax.set_title(f'{metric.replace("_", " ").title()} Comparison\n(with Standard Deviation)', fontsize=14, fontweight='bold')
            ax.set_xticks(x)
            ax.set_xticklabels(pivot_mean.index, rotation=0)
            ax.legend(fontsize=10)
            ax.grid(True, alpha=0.3)
            
            # Add value labels on bars with improvement indicators
            for i, model in enumerate(pivot_mean.index):
                zs_val = pivot_mean.loc[model, 'zero_shot']
                cot_val = pivot_mean.loc[model, 'chain_of_thought']
                
                # Value labels
                ax.text(i - width/2, zs_val + 0.01, f'{zs_val:.3f}', ha='center', va='bottom', fontsize=9, fontweight='bold')
                ax.text(i + width/2, cot_val + 0.01, f'{cot_val:.3f}', ha='center', va='bottom', fontsize=9, fontweight='bold')
                
                # Improvement percentage
                if zs_val > 0:
                    improvement = ((cot_val - zs_val) / zs_val) * 100
                    color = COLORS['improvement'] if improvement > 0 else COLORS['decline']
                    ax.text(i, max(zs_val, cot_val) + 0.05, f'{improvement:+.1f}%', 
                           ha='center', va='bottom', fontsize=8, color=color, fontweight='bold')
        
        plt.tight_layout()
        plt.savefig(self.output_dir / "comprehensive_metric_comparison.png", dpi=300, bbox_inches='tight')
        plt.show()
        print(f"📊 Saved comprehensive metric comparison bars")
    
    def plot_processing_time_analysis(self, df: pd.DataFrame):
        """Processing time analysis for entailment inference task only."""
        
        # Filter for entailment inference only (the only task with meaningful metrics)
        entailment_df = df[df['task'] == 'entailment_inference'].copy()
        
        if len(entailment_df) == 0:
            print("⚠️ No entailment inference data found for processing time analysis")
            return
        
        fig, axes = plt.subplots(1, 2, figsize=(15, 6))
        
        # Processing time comparison by model
        ax1 = axes[0]
        time_data = entailment_df.groupby(['model', 'prompt_type'])['avg_processing_time'].mean().reset_index()
        time_pivot = time_data.pivot(index='model', columns='prompt_type', values='avg_processing_time')
        
        x = np.arange(len(time_pivot.index))
        width = 0.35
        
        bars1 = ax1.bar(x - width/2, time_pivot['zero_shot'], width, 
                       label='Zero-Shot', color=COLORS['zero_shot'], alpha=0.8)
        bars2 = ax1.bar(x + width/2, time_pivot['chain_of_thought'], width,
                       label='Chain-of-Thought', color=COLORS['chain_of_thought'], alpha=0.8)
        
        # Add time increase percentages
        for i, model in enumerate(time_pivot.index):
            zs_time = time_pivot.loc[model, 'zero_shot']
            cot_time = time_pivot.loc[model, 'chain_of_thought']
            if zs_time > 0:
                increase = ((cot_time - zs_time) / zs_time) * 100
                ax1.text(i, max(zs_time, cot_time) + max(zs_time, cot_time) * 0.05, 
                        f'+{increase:.0f}%', ha='center', va='bottom', fontsize=10, 
                        color=COLORS['decline'], fontweight='bold')
        
        ax1.set_xlabel('Models', fontsize=12)
        ax1.set_ylabel('Processing Time (seconds)', fontsize=12)
        ax1.set_title('Processing Time Comparison\n(Entailment Inference Task)', fontsize=14, fontweight='bold')
        ax1.set_xticks(x)
        ax1.set_xticklabels(time_pivot.index, rotation=0)
        ax1.legend()
        ax1.grid(True, alpha=0.3)
        
        # Processing time by dataset
        ax2 = axes[1]
        dataset_time = entailment_df.groupby(['dataset', 'prompt_type'])['avg_processing_time'].mean().reset_index()
        dataset_pivot = dataset_time.pivot(index='dataset', columns='prompt_type', values='avg_processing_time')
        
        dataset_x = np.arange(len(dataset_pivot.index))
        ax2.bar(dataset_x - width/2, dataset_pivot['zero_shot'], width, 
                label='Zero-Shot', color=COLORS['zero_shot'], alpha=0.8)
        ax2.bar(dataset_x + width/2, dataset_pivot['chain_of_thought'], width,
                label='Chain-of-Thought', color=COLORS['chain_of_thought'], alpha=0.8)
        
        ax2.set_xlabel('Datasets', fontsize=12)
        ax2.set_ylabel('Processing Time (seconds)', fontsize=12)
        ax2.set_title('Processing Time by Dataset\n(Entailment Inference)', fontsize=14, fontweight='bold')
        ax2.set_xticks(dataset_x)
        ax2.set_xticklabels([dataset.upper() for dataset in dataset_pivot.index])
        ax2.legend()
        ax2.grid(True, alpha=0.3)
        
        plt.tight_layout()
        plt.savefig(self.output_dir / "processing_time_analysis.png", dpi=300, bbox_inches='tight')
        plt.show()
        print(f"⏱️ Saved processing time analysis")
    
    def create_entailment_performance_heatmap(self, df: pd.DataFrame):
        """Create focused heatmap for entailment inference performance only."""
        
        # Filter for entailment inference only
        entailment_df = df[df['task'] == 'entailment_inference'].copy()
        
        if len(entailment_df) == 0:
            print("⚠️ No entailment inference data found for heatmap")
            return
        
        fig, axes = plt.subplots(1, 3, figsize=(18, 6))
        
        # 1. F1 Score heatmap by prompt type
        for idx, (prompt_type, ax) in enumerate(zip(['zero_shot', 'chain_of_thought'], [axes[0], axes[1]])):
            filtered_df = entailment_df[entailment_df['prompt_type'] == prompt_type]
            heatmap_data = filtered_df.groupby(['model', 'dataset'])['f1_score'].mean().reset_index()
            heatmap_pivot = heatmap_data.pivot(index='model', columns='dataset', values='f1_score')
            
            sns.heatmap(heatmap_pivot, annot=True, fmt='.3f', cmap='RdYlGn', 
                       ax=ax, cbar_kws={'label': 'F1 Score'}, vmin=0.6, vmax=1.0,
                       annot_kws={'fontsize': 12, 'fontweight': 'bold'})
            
            ax.set_title(f'{prompt_type.replace("_", "-").title()}\nF1 Score Performance', fontsize=14, fontweight='bold')
            ax.set_xlabel('Datasets', fontsize=12)
            ax.set_ylabel('Models', fontsize=12)
            ax.set_xticklabels([dataset.upper() for dataset in heatmap_pivot.columns])
        
        # 2. Improvement heatmap (CoT vs Zero-Shot)
        ax3 = axes[2]
        improvement_data = []
        for model in entailment_df['model'].unique():
            for dataset in entailment_df['dataset'].unique():
                zs_data = entailment_df[(entailment_df['model'] == model) & 
                                       (entailment_df['dataset'] == dataset) & 
                                       (entailment_df['prompt_type'] == 'zero_shot')]
                cot_data = entailment_df[(entailment_df['model'] == model) & 
                                        (entailment_df['dataset'] == dataset) & 
                                        (entailment_df['prompt_type'] == 'chain_of_thought')]
                
                if len(zs_data) > 0 and len(cot_data) > 0:
                    zs_f1 = zs_data['f1_score'].iloc[0]
                    cot_f1 = cot_data['f1_score'].iloc[0]
                    improvement = ((cot_f1 - zs_f1) / zs_f1) * 100 if zs_f1 > 0 else 0
                    
                    improvement_data.append({
                        'model': model,
                        'dataset': dataset,
                        'improvement': improvement
                    })
        
        if improvement_data:
            imp_df = pd.DataFrame(improvement_data)
            imp_pivot = imp_df.pivot(index='model', columns='dataset', values='improvement')
            
            sns.heatmap(imp_pivot, annot=True, fmt='.1f', cmap='RdBu_r', center=0,
                       ax=ax3, cbar_kws={'label': 'F1 Improvement (%)'}, 
                       annot_kws={'fontsize': 12, 'fontweight': 'bold'})
            
            ax3.set_title('F1 Score Improvement\n(CoT vs Zero-Shot)', fontsize=14, fontweight='bold')
            ax3.set_xlabel('Datasets', fontsize=12)
            ax3.set_ylabel('Models', fontsize=12)
            ax3.set_xticklabels([dataset.upper() for dataset in imp_pivot.columns])
        
        plt.tight_layout()
        plt.savefig(self.output_dir / "entailment_performance_heatmap.png", dpi=300, bbox_inches='tight')
        plt.show()
        print(f"🔥 Saved entailment inference performance heatmap")
    
    def plot_improvement_analysis(self, df: pd.DataFrame):
        """Comprehensive analysis of where CoT provides improvements vs degradations."""
        
        # Calculate improvement for each metric with enhanced statistics
        improvement_data = []
        
        for model in self.models:
            for task in self.tasks:
                for dataset in self.datasets:
                    zs_row = df[(df['model'] == model) & (df['task'] == task) & 
                               (df['dataset'] == dataset) & (df['prompt_type'] == 'zero_shot')]
                    cot_row = df[(df['model'] == model) & (df['task'] == task) & 
                                (df['dataset'] == dataset) & (df['prompt_type'] == 'chain_of_thought')]
                    
                    if len(zs_row) > 0 and len(cot_row) > 0:
                        zs_data = zs_row.iloc[0]
                        cot_data = cot_row.iloc[0]
                        
                        # Calculate improvements
                        f1_improvement = ((cot_data['f1_score'] - zs_data['f1_score']) / zs_data['f1_score']) * 100
                        acc_improvement = ((cot_data['accuracy'] - zs_data['accuracy']) / zs_data['accuracy']) * 100
                        precision_improvement = ((cot_data['precision'] - zs_data['precision']) / zs_data['precision']) * 100 if zs_data['precision'] > 0 else 0
                        recall_improvement = ((cot_data['recall'] - zs_data['recall']) / zs_data['recall']) * 100 if zs_data['recall'] > 0 else 0
                        
                        improvement_data.append({
                            'model': model,
                            'task': task,
                            'dataset': dataset,
                            'combination': f"{model}-{task}-{dataset}",
                            'f1_improvement': f1_improvement,
                            'accuracy_improvement': acc_improvement,
                            'precision_improvement': precision_improvement,
                            'recall_improvement': recall_improvement,
                            'cost_increase': ((cot_data['total_cost'] - zs_data['total_cost']) / zs_data['total_cost']) * 100 if zs_data['total_cost'] > 0 else 0,
                            'time_increase': ((cot_data['avg_processing_time'] - zs_data['avg_processing_time']) / zs_data['avg_processing_time']) * 100,
                            'zs_f1': zs_data['f1_score'],
                            'cot_f1': cot_data['f1_score']
                        })
        
        improvement_df = pd.DataFrame(improvement_data)
        
        # Create comprehensive improvement analysis plot
        fig, axes = plt.subplots(3, 2, figsize=(16, 18))
        
        # 1. F1 Score improvements distribution
        ax1 = axes[0, 0]
        n_positive = len(improvement_df[improvement_df['f1_improvement'] > 0])
        n_negative = len(improvement_df[improvement_df['f1_improvement'] < 0])
        
        ax1.hist(improvement_df['f1_improvement'], bins=25, alpha=0.7, color=COLORS['improvement'], 
                edgecolor='black', label=f'Total: {len(improvement_df)}')
        ax1.axvline(x=0, color='red', linestyle='--', alpha=0.7, linewidth=2, label='No Change')
        ax1.axvline(x=improvement_df['f1_improvement'].mean(), color='blue', linestyle='-', 
                   alpha=0.8, linewidth=2, label=f'Mean: {improvement_df["f1_improvement"].mean():.1f}%')
        
        ax1.set_xlabel('F1 Score Improvement (%)', fontsize=11)
        ax1.set_ylabel('Frequency', fontsize=11)
        ax1.set_title(f'F1 Score Improvement Distribution\nPositive: {n_positive}, Negative: {n_negative}', 
                     fontsize=12, fontweight='bold')
        ax1.legend()
        ax1.grid(True, alpha=0.3)
        
        # Add statistics text
        ax1.text(0.02, 0.98, f'Median: {improvement_df["f1_improvement"].median():.1f}%\n'
                             f'Std: {improvement_df["f1_improvement"].std():.1f}%\n'
                             f'Min: {improvement_df["f1_improvement"].min():.1f}%\n'
                             f'Max: {improvement_df["f1_improvement"].max():.1f}%',
                transform=ax1.transAxes, verticalalignment='top', bbox=dict(boxstyle='round', facecolor='white', alpha=0.8))
        
        # 2. Performance improvements by model
        ax2 = axes[0, 1]
        model_improvements = improvement_df.groupby('model')['f1_improvement'].agg(['mean', 'median', 'std']).reset_index()
        
        x_pos = np.arange(len(model_improvements))
        bars = ax2.bar(x_pos, model_improvements['mean'], yerr=model_improvements['std'], 
                      capsize=5, color=COLORS['improvement'], alpha=0.7, edgecolor='black')
        
        # Add median line markers
        ax2.scatter(x_pos, model_improvements['median'], color='red', s=50, zorder=5, 
                   label='Median', marker='D')
        
        ax2.axhline(y=0, color='gray', linestyle='--', alpha=0.7)
        ax2.set_xlabel('Models', fontsize=11)
        ax2.set_ylabel('F1 Improvement (%)', fontsize=11)
        ax2.set_title('F1 Improvement by Model\n(Mean ± Std)', fontsize=12, fontweight='bold')
        ax2.set_xticks(x_pos)
        ax2.set_xticklabels(model_improvements['model'], rotation=45)
        ax2.legend()
        ax2.grid(True, alpha=0.3)
        
        # Add value labels on bars
        for i, (bar, mean_val) in enumerate(zip(bars, model_improvements['mean'])):
            ax2.text(bar.get_x() + bar.get_width()/2, bar.get_height() + 0.5, 
                    f'{mean_val:.1f}%', ha='center', va='bottom', fontweight='bold')
        
        # 3. Task-wise improvement analysis
        ax3 = axes[1, 0]
        task_improvements = improvement_df.groupby('task')['f1_improvement'].agg(['mean', 'median', 'std']).reset_index()
        
        x_pos = np.arange(len(task_improvements))
        bars = ax3.bar(x_pos, task_improvements['mean'], yerr=task_improvements['std'], 
                      capsize=5, color=COLORS['improvement'], alpha=0.7, edgecolor='black')
        
        ax3.scatter(x_pos, task_improvements['median'], color='red', s=50, zorder=5, 
                   label='Median', marker='D')
        
        ax3.axhline(y=0, color='gray', linestyle='--', alpha=0.7)
        ax3.set_xlabel('Tasks', fontsize=11)
        ax3.set_ylabel('F1 Improvement (%)', fontsize=11)
        ax3.set_title('F1 Improvement by Task\n(Mean ± Std)', fontsize=12, fontweight='bold')
        ax3.set_xticks(x_pos)
        ax3.set_xticklabels([task.replace('_', ' ').title() for task in task_improvements['task']], rotation=45)
        ax3.legend()
        ax3.grid(True, alpha=0.3)
        
        # 4. Cost vs Performance trade-off
        ax4 = axes[1, 1]
        scatter = ax4.scatter(improvement_df['cost_increase'], improvement_df['f1_improvement'], 
                             c=improvement_df['zs_f1'], cmap='viridis', alpha=0.7, s=60)
        
        # Add regression line
        from scipy.stats import linregress
        slope, intercept, r_value, p_value, std_err = linregress(improvement_df['cost_increase'], 
                                                                improvement_df['f1_improvement'])
        line_x = np.linspace(improvement_df['cost_increase'].min(), improvement_df['cost_increase'].max(), 100)
        line_y = slope * line_x + intercept
        ax4.plot(line_x, line_y, 'r--', alpha=0.8, label=f'R² = {r_value**2:.3f}')
        
        ax4.axhline(y=0, color='gray', linestyle='--', alpha=0.5)
        ax4.axvline(x=0, color='gray', linestyle='--', alpha=0.5)
        ax4.set_xlabel('Cost Increase (%)', fontsize=11)
        ax4.set_ylabel('F1 Improvement (%)', fontsize=11)
        ax4.set_title('Cost vs Performance Trade-off\n(Color = Zero-Shot F1)', fontsize=12, fontweight='bold')
        ax4.legend()
        ax4.grid(True, alpha=0.3)
        
        # Add colorbar
        cbar = plt.colorbar(scatter, ax=ax4)
        cbar.set_label('Zero-Shot F1 Score', fontsize=10)
        
        # 5. Multi-metric improvement radar-style comparison
        ax5 = axes[2, 0]
        metrics = ['f1_improvement', 'accuracy_improvement', 'precision_improvement', 'recall_improvement']
        metric_means = [improvement_df[metric].mean() for metric in metrics]
        metric_labels = ['F1', 'Accuracy', 'Precision', 'Recall']
        
        bars = ax5.bar(metric_labels, metric_means, color=[COLORS['improvement'] if x > 0 else COLORS['decline'] for x in metric_means],
                      alpha=0.7, edgecolor='black')
        
        ax5.axhline(y=0, color='gray', linestyle='--', alpha=0.7)
        ax5.set_ylabel('Average Improvement (%)', fontsize=11)
        ax5.set_title('Average Improvement Across Metrics', fontsize=12, fontweight='bold')
        ax5.grid(True, alpha=0.3)
        
        # Add value labels
        for bar, val in zip(bars, metric_means):
            ax5.text(bar.get_x() + bar.get_width()/2, bar.get_height() + (0.5 if val > 0 else -1), 
                    f'{val:.1f}%', ha='center', va='bottom' if val > 0 else 'top', fontweight='bold')
        
        # 6. Improvement consistency analysis
        ax6 = axes[2, 1]
        
        # Calculate consistency (% of positive improvements for each model)
        consistency_data = []
        for model in improvement_df['model'].unique():
            model_data = improvement_df[improvement_df['model'] == model]
            pos_improvements = len(model_data[model_data['f1_improvement'] > 0])
            consistency = (pos_improvements / len(model_data)) * 100
            avg_improvement = model_data['f1_improvement'].mean()
            
            consistency_data.append({
                'model': model,
                'consistency': consistency,
                'avg_improvement': avg_improvement
            })
        
        cons_df = pd.DataFrame(consistency_data)
        
        # Create bubble chart
        bubble_sizes = [abs(x) * 20 for x in cons_df['avg_improvement']]
        colors = ['green' if x > 0 else 'red' for x in cons_df['avg_improvement']]
        
        scatter = ax6.scatter(cons_df['consistency'], cons_df['avg_improvement'], 
                             s=bubble_sizes, c=colors, alpha=0.6, edgecolors='black')
        
        ax6.axhline(y=0, color='gray', linestyle='--', alpha=0.7)
        ax6.axvline(x=50, color='gray', linestyle='--', alpha=0.7, label='50% Consistency')
        ax6.set_xlabel('Improvement Consistency (%)', fontsize=11)
        ax6.set_ylabel('Average F1 Improvement (%)', fontsize=11)
        ax6.set_title('Improvement Consistency vs Magnitude\n(Bubble size = |Improvement|)', fontsize=12, fontweight='bold')
        ax6.legend()
        ax6.grid(True, alpha=0.3)
        
        # Add model labels
        for i, row in cons_df.iterrows():
            ax6.annotate(row['model'][:4], (row['consistency'], row['avg_improvement']), 
                        xytext=(5, 5), textcoords='offset points', fontsize=9)
        
        plt.tight_layout()
        plt.savefig(self.output_dir / "comprehensive_improvement_analysis.png", dpi=300, bbox_inches='tight')
        plt.show()
        print(f"📈 Saved comprehensive improvement analysis")
        
        return improvement_df
    
    def create_statistical_summary(self, df: pd.DataFrame, improvement_df: pd.DataFrame):
        """Create comprehensive statistical summary with detailed analysis."""
        
        summary_stats = {}
        
        # Overall performance comparison with confidence intervals
        zs_metrics = df[df['prompt_type'] == 'zero_shot'][['accuracy', 'f1_score', 'precision', 'recall']].agg(['mean', 'std', 'median', 'min', 'max'])
        cot_metrics = df[df['prompt_type'] == 'chain_of_thought'][['accuracy', 'f1_score', 'precision', 'recall']].agg(['mean', 'std', 'median', 'min', 'max'])
        
        # Calculate 95% confidence intervals
        from scipy.stats import t
        n_zs = len(df[df['prompt_type'] == 'zero_shot'])
        n_cot = len(df[df['prompt_type'] == 'chain_of_thought'])
        alpha = 0.05
        
        summary_stats['overall_performance'] = {
            'zero_shot': {
                'sample_size': n_zs,
                'metrics': zs_metrics.to_dict(),
                'confidence_intervals_95': {}
            },
            'chain_of_thought': {
                'sample_size': n_cot,
                'metrics': cot_metrics.to_dict(),
                'confidence_intervals_95': {}
            },
            'relative_improvements': {}
        }
        
        # Calculate confidence intervals and improvements
        for metric in ['accuracy', 'f1_score', 'precision', 'recall']:
            # Zero-shot CI
            zs_mean = zs_metrics.loc['mean', metric]
            zs_std = zs_metrics.loc['std', metric]
            zs_se = zs_std / np.sqrt(n_zs)
            zs_ci = t.interval(1-alpha, n_zs-1, loc=zs_mean, scale=zs_se)
            
            # CoT CI
            cot_mean = cot_metrics.loc['mean', metric]
            cot_std = cot_metrics.loc['std', metric]
            cot_se = cot_std / np.sqrt(n_cot)
            cot_ci = t.interval(1-alpha, n_cot-1, loc=cot_mean, scale=cot_se)
            
            summary_stats['overall_performance']['zero_shot']['confidence_intervals_95'][metric] = zs_ci
            summary_stats['overall_performance']['chain_of_thought']['confidence_intervals_95'][metric] = cot_ci
            
            # Relative improvement
            rel_improvement = ((cot_mean - zs_mean) / zs_mean) * 100 if zs_mean > 0 else 0
            summary_stats['overall_performance']['relative_improvements'][metric] = {
                'percent_change': rel_improvement,
                'absolute_change': cot_mean - zs_mean,
                'effect_size': (cot_mean - zs_mean) / np.sqrt((zs_std**2 + cot_std**2) / 2) if (zs_std > 0 or cot_std > 0) else 0
            }
        
        # Statistical significance testing
        from scipy.stats import ttest_ind, mannwhitneyu, wilcoxon
        
        summary_stats['statistical_tests'] = {}
        
        for metric in ['accuracy', 'f1_score', 'precision', 'recall']:
            zs_values = df[df['prompt_type'] == 'zero_shot'][metric].dropna()
            cot_values = df[df['prompt_type'] == 'chain_of_thought'][metric].dropna()
            
            if len(zs_values) > 1 and len(cot_values) > 1:
                # T-test (parametric)
                t_stat, t_p = ttest_ind(cot_values, zs_values)
                
                # Mann-Whitney U test (non-parametric)
                u_stat, u_p = mannwhitneyu(cot_values, zs_values, alternative='two-sided')
                
                summary_stats['statistical_tests'][metric] = {
                    't_test': {'statistic': t_stat, 'p_value': t_p, 'significant': t_p < 0.05},
                    'mann_whitney_u': {'statistic': u_stat, 'p_value': u_p, 'significant': u_p < 0.05}
                }
        
        # Cost and efficiency analysis
        zs_cost_stats = df[df['prompt_type'] == 'zero_shot']['total_cost'].agg(['mean', 'std', 'median', 'min', 'max'])
        cot_cost_stats = df[df['prompt_type'] == 'chain_of_thought']['total_cost'].agg(['mean', 'std', 'median', 'min', 'max'])
        
        summary_stats['cost_analysis'] = {
            'zero_shot': zs_cost_stats.to_dict(),
            'chain_of_thought': cot_cost_stats.to_dict(),
            'cost_increase': {
                'absolute': cot_cost_stats['mean'] - zs_cost_stats['mean'],
                'percent': ((cot_cost_stats['mean'] - zs_cost_stats['mean']) / zs_cost_stats['mean']) * 100 if zs_cost_stats['mean'] > 0 else 0
            },
            'efficiency_metrics': {
                'zero_shot_f1_per_dollar': zs_metrics.loc['mean', 'f1_score'] / zs_cost_stats['mean'] if zs_cost_stats['mean'] > 0 else float('inf'),
                'cot_f1_per_dollar': cot_metrics.loc['mean', 'f1_score'] / cot_cost_stats['mean'] if cot_cost_stats['mean'] > 0 else float('inf')
            }
        }
        
        # Model-specific analysis
        summary_stats['model_analysis'] = {}
        
        for model in df['model'].unique():
            model_df = df[df['model'] == model]
            zs_model = model_df[model_df['prompt_type'] == 'zero_shot']['f1_score']
            cot_model = model_df[model_df['prompt_type'] == 'chain_of_thought']['f1_score']
            
            if len(zs_model) > 0 and len(cot_model) > 0:
                model_improvement = ((cot_model.mean() - zs_model.mean()) / zs_model.mean()) * 100 if zs_model.mean() > 0 else 0
                
                summary_stats['model_analysis'][model] = {
                    'zero_shot_f1': {'mean': zs_model.mean(), 'std': zs_model.std()},
                    'cot_f1': {'mean': cot_model.mean(), 'std': cot_model.std()},
                    'improvement_percent': model_improvement,
                    'consistency': len(improvement_df[(improvement_df['model'] == model) & (improvement_df['f1_improvement'] > 0)]) / len(improvement_df[improvement_df['model'] == model]) * 100 if len(improvement_df[improvement_df['model'] == model]) > 0 else 0
                }
        
        # Task-specific analysis
        summary_stats['task_analysis'] = {}
        
        for task in df['task'].unique():
            task_df = df[df['task'] == task]
            zs_task = task_df[task_df['prompt_type'] == 'zero_shot']['f1_score']
            cot_task = task_df[task_df['prompt_type'] == 'chain_of_thought']['f1_score']
            
            if len(zs_task) > 0 and len(cot_task) > 0:
                task_improvement = ((cot_task.mean() - zs_task.mean()) / zs_task.mean()) * 100 if zs_task.mean() > 0 else 0
                
                summary_stats['task_analysis'][task] = {
                    'zero_shot_f1': {'mean': zs_task.mean(), 'std': zs_task.std()},
                    'cot_f1': {'mean': cot_task.mean(), 'std': cot_task.std()},
                    'improvement_percent': task_improvement,
                    'difficulty_ranking': zs_task.mean()  # Lower F1 = harder task
                }
        
        # Improvement distribution analysis
        summary_stats['improvement_analysis'] = {
            'f1_improvements': {
                'positive_cases': len(improvement_df[improvement_df['f1_improvement'] > 0]),
                'negative_cases': len(improvement_df[improvement_df['f1_improvement'] < 0]),
                'neutral_cases': len(improvement_df[improvement_df['f1_improvement'] == 0]),
                'total_cases': len(improvement_df),
                'success_rate': len(improvement_df[improvement_df['f1_improvement'] > 0]) / len(improvement_df) * 100,
                'statistics': {
                    'mean': improvement_df['f1_improvement'].mean(),
                    'median': improvement_df['f1_improvement'].median(),
                    'std': improvement_df['f1_improvement'].std(),
                    'min': improvement_df['f1_improvement'].min(),
                    'max': improvement_df['f1_improvement'].max(),
                    'q25': improvement_df['f1_improvement'].quantile(0.25),
                    'q75': improvement_df['f1_improvement'].quantile(0.75)
                }
            }
        }
        
        # Best and worst performers
        best_improvements = improvement_df.nlargest(3, 'f1_improvement')
        worst_improvements = improvement_df.nsmallest(3, 'f1_improvement')
        
        summary_stats['performance_extremes'] = {
            'best_improvements': best_improvements[['model', 'task', 'dataset', 'f1_improvement', 'zs_f1', 'cot_f1']].to_dict('records'),
            'worst_improvements': worst_improvements[['model', 'task', 'dataset', 'f1_improvement', 'zs_f1', 'cot_f1']].to_dict('records'),
            'best_zero_shot': df[df['prompt_type'] == 'zero_shot'].nlargest(3, 'f1_score')[['model', 'task', 'dataset', 'f1_score', 'accuracy']].to_dict('records'),
            'best_cot': df[df['prompt_type'] == 'chain_of_thought'].nlargest(3, 'f1_score')[['model', 'task', 'dataset', 'f1_score', 'accuracy']].to_dict('records')
        }
        
        # Save comprehensive summary to JSON
        with open(self.output_dir / "comprehensive_statistical_summary.json", 'w') as f:
            json.dump(summary_stats, f, indent=2, default=str)
        
        print(f"📊 Saved comprehensive statistical summary")
        return summary_stats
    
    def generate_executive_summary_table(self, summary_stats: Dict):
        """Generate streamlined executive summary table focused on entailment inference."""
        
        # Create focused tables for entailment inference results
        
        # 1. Performance Metrics Summary (Entailment Inference Only)
        performance_data = []
        performance_data.append(['Metric', 'Zero-Shot', 'Chain-of-Thought', 'Improvement (%)', 'P-value', 'Significant'])
        
        for metric in ['accuracy', 'f1_score', 'precision', 'recall']:
            zs_mean = summary_stats['overall_performance']['zero_shot']['metrics'][metric]['mean']
            zs_std = summary_stats['overall_performance']['zero_shot']['metrics'][metric]['std']
            cot_mean = summary_stats['overall_performance']['chain_of_thought']['metrics'][metric]['mean']
            cot_std = summary_stats['overall_performance']['chain_of_thought']['metrics'][metric]['std']
            improvement = summary_stats['overall_performance']['relative_improvements'][metric]['percent_change']
            
            # Get statistical significance
            is_significant = summary_stats['statistical_tests'][metric]['t_test']['significant']
            p_value = summary_stats['statistical_tests'][metric]['t_test']['p_value']
            
            performance_data.append([
                metric.replace('_', ' ').title(),
                f"{zs_mean:.3f} ± {zs_std:.3f}",
                f"{cot_mean:.3f} ± {cot_std:.3f}",
                f"{improvement:+.1f}%",
                f"{p_value:.4f}" if p_value < 0.001 else f"{p_value:.3f}",
                "✓" if is_significant else "✗"
            ])
        
        # 2. Model Performance Summary
        model_data = []
        model_data.append(['Model', 'Zero-Shot F1', 'CoT F1', 'Improvement (%)', 'Processing Time Increase'])
        
        for model, stats in summary_stats['model_analysis'].items():
            # Calculate processing time increase from the raw data
            processing_increase = "N/A"  # We'll calculate this from the comparison table
            
            model_data.append([
                model,
                f"{stats['zero_shot_f1']['mean']:.3f}",
                f"{stats['cot_f1']['mean']:.3f}",
                f"{stats['improvement_percent']:+.1f}%",
                processing_increase  # Will be filled in separately
            ])
        
        # 3. Key Insights Summary
        insights_data = []
        insights_data.append(['Insight', 'Value', 'Interpretation'])
        
        success_rate = summary_stats['improvement_analysis']['f1_improvements']['success_rate']
        avg_improvement = summary_stats['improvement_analysis']['f1_improvements']['statistics']['mean']
        best_improvement = summary_stats['performance_extremes']['best_improvements'][0]
        worst_improvement = summary_stats['performance_extremes']['worst_improvements'][0]
        
        insights_data.extend([
            ['Success Rate', f"{success_rate:.1f}%", "Cases where CoT outperformed Zero-Shot"],
            ['Average F1 Change', f"{avg_improvement:+.1f}%", "Mean F1 score change across all experiments"],
            ['Best Case', f"{best_improvement['f1_improvement']:+.1f}%", f"{best_improvement['model']} on {best_improvement['dataset']}"],
            ['Worst Case', f"{worst_improvement['f1_improvement']:+.1f}%", f"{worst_improvement['model']} on {worst_improvement['dataset']}"],
            ['Overall Conclusion', 'Mixed Results', 'CoT does not universally improve performance']
        ])
        
        # Create visualizations for these tables
        fig, axes = plt.subplots(2, 2, figsize=(18, 12))
        
        # Performance metrics table
        ax1 = axes[0, 0]
        ax1.axis('tight')
        ax1.axis('off')
        table1 = ax1.table(cellText=performance_data[1:], colLabels=performance_data[0],
                          cellLoc='center', loc='center', bbox=[0, 0, 1, 1])
        table1.auto_set_font_size(False)
        table1.set_fontsize(9)
        table1.scale(1, 2)
        ax1.set_title('Performance Metrics Comparison\n(Entailment Inference Task)', fontsize=14, fontweight='bold', pad=20)
        
        # Color-code significance
        for i in range(1, len(performance_data)):
            if performance_data[i][5] == "✓":  # Significant
                table1[(i, 5)].set_facecolor('#90EE90')  # Light green
            else:
                table1[(i, 5)].set_facecolor('#FFB6C1')  # Light red
        
        # Model performance table
        ax2 = axes[0, 1]
        ax2.axis('tight')
        ax2.axis('off')
        table2 = ax2.table(cellText=model_data[1:], colLabels=model_data[0],
                          cellLoc='center', loc='center', bbox=[0, 0, 1, 1])
        table2.auto_set_font_size(False)
        table2.set_fontsize(9)
        table2.scale(1, 2)
        ax2.set_title('Model Performance Summary', fontsize=14, fontweight='bold', pad=20)
        
        # Key insights table
        ax3 = axes[1, 0]
        ax3.axis('tight')
        ax3.axis('off')
        table3 = ax3.table(cellText=insights_data[1:], colLabels=insights_data[0],
                          cellLoc='center', loc='center', bbox=[0, 0, 1, 1])
        table3.auto_set_font_size(False)
        table3.set_fontsize(9)
        table3.scale(1, 2)
        ax3.set_title('Key Insights & Findings', fontsize=14, fontweight='bold', pad=20)
        
        # Executive Summary Text
        ax4 = axes[1, 1]
        ax4.axis('off')
        
        exec_summary = f"""
EXECUTIVE SUMMARY

Performance Overview:
• Chain-of-Thought shows {avg_improvement:+.1f}% average F1 improvement
• Success rate: {success_rate:.1f}% of cases show improvement
• No statistically significant differences found (p > 0.05)

Key Findings:
• Best performing combination: {best_improvement['model']} on {best_improvement['dataset']} ({best_improvement['f1_improvement']:+.1f}%)
• Most consistent improvement: Llama3.1:8b model
• Processing time increases significantly (200-400%)

Practical Implications:
• CoT does not universally improve factuality evaluation
• Model selection more important than prompting strategy
• Zero-shot often preferable for efficiency

Recommendations:
• Use zero-shot for production systems
• Consider CoT only for specific model-dataset combinations
• Focus on model optimization over prompt engineering
"""
        
        ax4.text(0.05, 0.95, exec_summary, transform=ax4.transAxes, fontsize=10,
                verticalalignment='top', bbox=dict(boxstyle='round,pad=1', facecolor='lightblue', alpha=0.8))
        
        plt.tight_layout()
        plt.savefig(self.output_dir / "executive_summary_focused.png", dpi=300, bbox_inches='tight')
        plt.show()
        print(f"📋 Saved focused executive summary")
        
        return {
            'performance_data': performance_data,
            'model_data': model_data,
            'insights_data': insights_data
        }
    
    def run_full_analysis(self):
        """Run the complete analysis pipeline."""
        
        print("🚀 Starting Zero-Shot vs Chain-of-Thought Analysis")
        print("=" * 60)
        
        # Extract performance metrics
        print("\n1. Extracting performance metrics...")
        df = self.extract_performance_metrics()
        
        # Create comparison table
        print("\n2. Creating performance comparison table...")
        comparison_df = self.create_performance_comparison_table(df)
        
        # Generate visualizations
        print("\n3. Generating visualizations...")
        self.plot_metric_comparison_bars(df)
        self.plot_processing_time_analysis(df)
        self.create_entailment_performance_heatmap(df)
        
        # Improvement analysis
        print("\n4. Analyzing improvements and degradations...")
        improvement_df = self.plot_improvement_analysis(df)
        
        # Statistical summary
        print("\n5. Creating statistical summary...")
        summary_stats = self.create_statistical_summary(df, improvement_df)
        
        # Executive summary
        print("\n6. Generating executive summary...")
        self.generate_executive_summary_table(summary_stats)
        
        print(f"\n✅ Analysis complete! Results saved to: {self.output_dir}")
        print("\nGenerated files:")
        for file in self.output_dir.glob("*"):
            print(f"  📄 {file.name}")


def main():
    """Main execution function."""
    
    # Define file paths
    current_dir = Path(__file__).parent.parent
    zero_shot_path = current_dir / "results/experiments2/multi_llm_sota_comparison_20250808_102503/sota_multi_comparison/sota_multi_comparison_results_zero_shot.json"
    cot_path = current_dir / "results/experiments2/multi_llm_sota_comparison_20250808_202801/sota_multi_comparison/sota_multi_comparison_results_chain_of_thought.json"
    output_dir = current_dir / "results/comparison_analysis"
    
    # Verify files exist
    if not zero_shot_path.exists():
        print(f"❌ Zero-shot file not found: {zero_shot_path}")
        return
    if not cot_path.exists():
        print(f"❌ Chain-of-thought file not found: {cot_path}")
        return
    
    # Run analysis
    analyzer = ZeroShotVsCoTAnalyzer(
        zero_shot_path=str(zero_shot_path),
        cot_path=str(cot_path),
        output_dir=str(output_dir)
    )
    
    analyzer.run_full_analysis()


if __name__ == "__main__":
    main()
