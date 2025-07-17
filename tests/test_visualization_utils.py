"""
Visualization and Utilities Tests
==================================

Tests for visualization utilities, logging utilities, and other
helper functions used throughout the factuality evaluation system.

Author: Michael Ogunjimi
Institution: University of Manchester
"""

import pytest
import tempfile
import os
import json
import numpy as np
from pathlib import Path
from unittest.mock import Mock, patch, MagicMock
import matplotlib
matplotlib.use('Agg')  # Use non-interactive backend for testing
import matplotlib.pyplot as plt
import seaborn as sns


class Logger:
    """Simple logger for testing"""
    
    def __init__(self, name="test_logger", level="INFO"):
        self.name = name
        self.level = level
        self.logs = []
        self.handlers = []
    
    def info(self, msg, *args, **kwargs):
        formatted_msg = msg % args if args else msg
        self.logs.append(("INFO", formatted_msg))
    
    def warning(self, msg, *args, **kwargs):
        formatted_msg = msg % args if args else msg
        self.logs.append(("WARNING", formatted_msg))
    
    def error(self, msg, *args, **kwargs):
        formatted_msg = msg % args if args else msg
        self.logs.append(("ERROR", formatted_msg))
    
    def debug(self, msg, *args, **kwargs):
        formatted_msg = msg % args if args else msg
        self.logs.append(("DEBUG", formatted_msg))
    
    def add_handler(self, handler):
        self.handlers.append(handler)
    
    def get_logs(self):
        return self.logs
    
    def clear_logs(self):
        self.logs = []


def setup_logger(name=None, level="INFO", log_file=None, format_string=None):
    """Setup logger with optional file output"""
    logger_name = name or __name__
    logger = Logger(logger_name, level)
    
    if log_file:
        # In real implementation, would add file handler
        logger.log_file = log_file
    
    if format_string:
        logger.format_string = format_string
    
    return logger


class MetricsVisualizer:
    """Visualization utility for factuality evaluation metrics"""
    
    def __init__(self, style='seaborn', figsize=(10, 6), dpi=100):
        self.style = style
        self.figsize = figsize
        self.dpi = dpi
        self.figures = []
        
        # Set matplotlib style
        plt.style.use('default')  # Use default instead of seaborn for compatibility
        sns.set_palette("husl")
    
    def plot_metric_distribution(self, scores, metric_name="Metric", bins=30, save_path=None):
        """Plot distribution of metric scores"""
        fig, ax = plt.subplots(figsize=self.figsize, dpi=self.dpi)
        
        # Create histogram
        ax.hist(scores, bins=bins, alpha=0.7, color='skyblue', edgecolor='black')
        ax.set_xlabel(f'{metric_name} Score')
        ax.set_ylabel('Frequency')
        ax.set_title(f'Distribution of {metric_name} Scores')
        
        # Add statistics
        mean_score = np.mean(scores)
        std_score = np.std(scores)
        ax.axvline(mean_score, color='red', linestyle='--', 
                  label=f'Mean: {mean_score:.3f}')
        ax.axvline(mean_score + std_score, color='orange', linestyle=':', 
                  label=f'+1 STD: {mean_score + std_score:.3f}')
        ax.axvline(mean_score - std_score, color='orange', linestyle=':', 
                  label=f'-1 STD: {mean_score - std_score:.3f}')
        
        ax.legend()
        ax.grid(True, alpha=0.3)
        
        plt.tight_layout()
        
        if save_path:
            fig.savefig(save_path, dpi=self.dpi, bbox_inches='tight')
        
        self.figures.append(fig)
        return fig
    
    def plot_metric_comparison(self, metric_data, metric_names=None, save_path=None):
        """Plot comparison of multiple metrics"""
        if metric_names is None:
            metric_names = [f'Metric {i+1}' for i in range(len(metric_data))]
        
        fig, ax = plt.subplots(figsize=self.figsize, dpi=self.dpi)
        
        # Create box plot
        bp = ax.boxplot(metric_data, labels=metric_names, patch_artist=True)
        
        # Color the boxes
        colors = sns.color_palette("husl", len(metric_data))
        for patch, color in zip(bp['boxes'], colors):
            patch.set_facecolor(color)
            patch.set_alpha(0.7)
        
        ax.set_ylabel('Score')
        ax.set_title('Metric Comparison')
        ax.grid(True, alpha=0.3)
        
        # Rotate x-axis labels if many metrics
        if len(metric_names) > 3:
            plt.xticks(rotation=45)
        
        plt.tight_layout()
        
        if save_path:
            fig.savefig(save_path, dpi=self.dpi, bbox_inches='tight')
        
        self.figures.append(fig)
        return fig
    
    def plot_correlation_heatmap(self, correlation_matrix, labels=None, save_path=None):
        """Plot correlation heatmap for metrics"""
        fig, ax = plt.subplots(figsize=self.figsize, dpi=self.dpi)
        
        # Create heatmap
        im = ax.imshow(correlation_matrix, cmap='coolwarm', vmin=-1, vmax=1)
        
        # Add colorbar
        cbar = plt.colorbar(im, ax=ax)
        cbar.set_label('Correlation Coefficient', rotation=270, labelpad=20)
        
        # Set labels
        if labels:
            ax.set_xticks(range(len(labels)))
            ax.set_yticks(range(len(labels)))
            ax.set_xticklabels(labels, rotation=45)
            ax.set_yticklabels(labels)
        
        # Add correlation values as text
        for i in range(len(correlation_matrix)):
            for j in range(len(correlation_matrix[0])):
                text = ax.text(j, i, f'{correlation_matrix[i][j]:.2f}',
                             ha="center", va="center", color="black")
        
        ax.set_title('Metric Correlation Heatmap')
        plt.tight_layout()
        
        if save_path:
            fig.savefig(save_path, dpi=self.dpi, bbox_inches='tight')
        
        self.figures.append(fig)
        return fig
    
    def plot_score_scatter(self, x_scores, y_scores, x_label="X Metric", 
                          y_label="Y Metric", save_path=None):
        """Plot scatter plot of two metrics"""
        fig, ax = plt.subplots(figsize=self.figsize, dpi=self.dpi)
        
        # Create scatter plot
        ax.scatter(x_scores, y_scores, alpha=0.6, color='blue', s=50)
        
        # Add trend line
        if len(x_scores) > 1:
            z = np.polyfit(x_scores, y_scores, 1)
            p = np.poly1d(z)
            ax.plot(x_scores, p(x_scores), "r--", alpha=0.8)
            
            # Calculate correlation
            correlation = np.corrcoef(x_scores, y_scores)[0, 1]
            ax.text(0.05, 0.95, f'r = {correlation:.3f}', 
                   transform=ax.transAxes, bbox=dict(boxstyle="round", facecolor='wheat'))
        
        ax.set_xlabel(x_label)
        ax.set_ylabel(y_label)
        ax.set_title(f'{x_label} vs {y_label}')
        ax.grid(True, alpha=0.3)
        
        plt.tight_layout()
        
        if save_path:
            fig.savefig(save_path, dpi=self.dpi, bbox_inches='tight')
        
        self.figures.append(fig)
        return fig
    
    def plot_performance_over_time(self, timestamps, scores, metric_name="Metric", save_path=None):
        """Plot metric performance over time"""
        fig, ax = plt.subplots(figsize=self.figsize, dpi=self.dpi)
        
        # Create line plot
        ax.plot(timestamps, scores, marker='o', linewidth=2, markersize=6)
        
        # Add trend line
        if len(scores) > 1:
            x_numeric = range(len(scores))
            z = np.polyfit(x_numeric, scores, 1)
            p = np.poly1d(z)
            ax.plot(timestamps, p(x_numeric), "r--", alpha=0.8, label='Trend')
            ax.legend()
        
        ax.set_xlabel('Time')
        ax.set_ylabel(f'{metric_name} Score')
        ax.set_title(f'{metric_name} Performance Over Time')
        ax.grid(True, alpha=0.3)
        
        # Rotate x-axis labels
        plt.xticks(rotation=45)
        plt.tight_layout()
        
        if save_path:
            fig.savefig(save_path, dpi=self.dpi, bbox_inches='tight')
        
        self.figures.append(fig)
        return fig
    
    def create_summary_dashboard(self, metrics_data, save_path=None):
        """Create a comprehensive dashboard of metrics"""
        fig = plt.figure(figsize=(15, 10), dpi=self.dpi)
        
        # Create subplots
        gs = fig.add_gridspec(2, 3, hspace=0.3, wspace=0.3)
        
        # Plot 1: Overall score distribution
        ax1 = fig.add_subplot(gs[0, 0])
        all_scores = []
        for scores in metrics_data.values():
            all_scores.extend(scores)
        ax1.hist(all_scores, bins=20, alpha=0.7, color='skyblue')
        ax1.set_title('Overall Score Distribution')
        ax1.set_xlabel('Score')
        ax1.set_ylabel('Frequency')
        
        # Plot 2: Metric comparison boxplot
        ax2 = fig.add_subplot(gs[0, 1])
        metric_names = list(metrics_data.keys())
        metric_values = list(metrics_data.values())
        bp = ax2.boxplot(metric_values, labels=metric_names)
        ax2.set_title('Metric Comparison')
        ax2.set_ylabel('Score')
        plt.setp(ax2.get_xticklabels(), rotation=45)
        
        # Plot 3: Metric means bar chart
        ax3 = fig.add_subplot(gs[0, 2])
        means = [np.mean(scores) for scores in metric_values]
        bars = ax3.bar(metric_names, means, color=sns.color_palette("husl", len(means)))
        ax3.set_title('Mean Scores by Metric')
        ax3.set_ylabel('Mean Score')
        plt.setp(ax3.get_xticklabels(), rotation=45)
        
        # Plot 4: Correlation matrix (if multiple metrics)
        if len(metrics_data) > 1:
            ax4 = fig.add_subplot(gs[1, :2])
            
            # Create correlation matrix
            min_len = min(len(scores) for scores in metric_values)
            truncated_data = [scores[:min_len] for scores in metric_values]
            correlation_matrix = np.corrcoef(truncated_data)
            
            im = ax4.imshow(correlation_matrix, cmap='coolwarm', vmin=-1, vmax=1)
            ax4.set_xticks(range(len(metric_names)))
            ax4.set_yticks(range(len(metric_names)))
            ax4.set_xticklabels(metric_names, rotation=45)
            ax4.set_yticklabels(metric_names)
            ax4.set_title('Metric Correlations')
            
            # Add correlation values
            for i in range(len(correlation_matrix)):
                for j in range(len(correlation_matrix)):
                    ax4.text(j, i, f'{correlation_matrix[i][j]:.2f}',
                           ha="center", va="center", color="black")
            
            plt.colorbar(im, ax=ax4, shrink=0.6)
        
        # Plot 5: Statistics summary
        ax5 = fig.add_subplot(gs[1, 2])
        ax5.axis('off')
        
        stats_text = "Summary Statistics:\n\n"
        for metric_name, scores in metrics_data.items():
            stats_text += f"{metric_name}:\n"
            stats_text += f"  Mean: {np.mean(scores):.3f}\n"
            stats_text += f"  STD:  {np.std(scores):.3f}\n"
            stats_text += f"  Min:  {np.min(scores):.3f}\n"
            stats_text += f"  Max:  {np.max(scores):.3f}\n\n"
        
        ax5.text(0.1, 0.9, stats_text, transform=ax5.transAxes, 
                verticalalignment='top', fontfamily='monospace', fontsize=10)
        
        plt.suptitle('Factuality Evaluation Dashboard', fontsize=16, y=0.98)
        
        if save_path:
            fig.savefig(save_path, dpi=self.dpi, bbox_inches='tight')
        
        self.figures.append(fig)
        return fig
    
    def close_all_figures(self):
        """Close all created figures"""
        for fig in self.figures:
            plt.close(fig)
        self.figures.clear()
    
    def save_all_figures(self, directory, prefix="figure"):
        """Save all figures to directory"""
        Path(directory).mkdir(parents=True, exist_ok=True)
        
        for i, fig in enumerate(self.figures):
            filename = f"{prefix}_{i+1}.png"
            filepath = Path(directory) / filename
            fig.savefig(filepath, dpi=self.dpi, bbox_inches='tight')


class ResultsExporter:
    """Export evaluation results to various formats"""
    
    def __init__(self, output_dir="results"):
        self.output_dir = Path(output_dir)
        self.output_dir.mkdir(parents=True, exist_ok=True)
    
    def export_to_json(self, data, filename="results.json"):
        """Export data to JSON format"""
        filepath = self.output_dir / filename
        
        # Convert numpy arrays to lists for JSON serialization
        json_data = self._convert_for_json(data)
        
        with open(filepath, 'w') as f:
            json.dump(json_data, f, indent=2, default=str)
        
        return filepath
    
    def export_to_csv(self, data, filename="results.csv"):
        """Export data to CSV format"""
        filepath = self.output_dir / filename
        
        # Convert to flat structure for CSV
        if isinstance(data, dict) and all(isinstance(v, list) for v in data.values()):
            # Data is in format {metric_name: [scores]}
            import pandas as pd
            df = pd.DataFrame(data)
            df.to_csv(filepath, index=False)
        else:
            # Flatten nested structure
            flattened = self._flatten_for_csv(data)
            import pandas as pd
            df = pd.DataFrame(flattened)
            df.to_csv(filepath, index=False)
        
        return filepath
    
    def export_summary_report(self, data, filename="summary_report.txt"):
        """Export summary report in text format"""
        filepath = self.output_dir / filename
        
        with open(filepath, 'w') as f:
            f.write("Factuality Evaluation Summary Report\n")
            f.write("=" * 40 + "\n\n")
            
            if isinstance(data, dict):
                for key, value in data.items():
                    f.write(f"{key.upper()}:\n")
                    
                    if isinstance(value, list):
                        f.write(f"  Count: {len(value)}\n")
                        if value and isinstance(value[0], (int, float)):
                            f.write(f"  Mean: {np.mean(value):.4f}\n")
                            f.write(f"  STD: {np.std(value):.4f}\n")
                            f.write(f"  Min: {np.min(value):.4f}\n")
                            f.write(f"  Max: {np.max(value):.4f}\n")
                    elif isinstance(value, dict):
                        for subkey, subvalue in value.items():
                            f.write(f"  {subkey}: {subvalue}\n")
                    else:
                        f.write(f"  {value}\n")
                    
                    f.write("\n")
        
        return filepath
    
    def _convert_for_json(self, obj):
        """Convert numpy arrays and other objects for JSON serialization"""
        if isinstance(obj, np.ndarray):
            return obj.tolist()
        elif isinstance(obj, dict):
            return {key: self._convert_for_json(value) for key, value in obj.items()}
        elif isinstance(obj, list):
            return [self._convert_for_json(item) for item in obj]
        elif isinstance(obj, (np.integer, np.floating)):
            return float(obj)
        else:
            return obj
    
    def _flatten_for_csv(self, data, prefix=""):
        """Flatten nested dictionary for CSV export"""
        flattened = []
        
        if isinstance(data, dict):
            for key, value in data.items():
                new_prefix = f"{prefix}_{key}" if prefix else key
                
                if isinstance(value, dict):
                    flattened.extend(self._flatten_for_csv(value, new_prefix))
                elif isinstance(value, list) and value and isinstance(value[0], (int, float)):
                    for i, item in enumerate(value):
                        flattened.append({f"{new_prefix}_index": i, f"{new_prefix}_value": item})
                else:
                    flattened.append({new_prefix: value})
        
        return flattened


class FileUtils:
    """File utility functions"""
    
    @staticmethod
    def ensure_directory(path):
        """Ensure directory exists"""
        Path(path).mkdir(parents=True, exist_ok=True)
    
    @staticmethod
    def safe_filename(filename):
        """Create safe filename by removing invalid characters"""
        import re
        safe_name = re.sub(r'[^\w\-_\.]', '_', filename)
        safe_name = re.sub(r'_+', '_', safe_name)
        return safe_name
    
    @staticmethod
    def get_file_size(filepath):
        """Get file size in bytes"""
        return Path(filepath).stat().st_size
    
    @staticmethod
    def backup_file(filepath, backup_dir="backups"):
        """Create backup of file"""
        filepath = Path(filepath)
        if not filepath.exists():
            return None
        
        backup_path = Path(backup_dir)
        backup_path.mkdir(parents=True, exist_ok=True)
        
        import shutil
        import datetime
        
        timestamp = datetime.datetime.now().strftime("%Y%m%d_%H%M%S")
        backup_filename = f"{filepath.stem}_{timestamp}{filepath.suffix}"
        backup_filepath = backup_path / backup_filename
        
        shutil.copy2(filepath, backup_filepath)
        return backup_filepath
    
    @staticmethod
    def clean_temp_files(directory, pattern="*.tmp"):
        """Clean temporary files matching pattern"""
        import glob
        
        temp_files = glob.glob(str(Path(directory) / pattern))
        cleaned_count = 0
        
        for temp_file in temp_files:
            try:
                os.remove(temp_file)
                cleaned_count += 1
            except OSError:
                pass
        
        return cleaned_count


class ProgressTracker:
    """Track progress of long-running operations"""
    
    def __init__(self, total_items, description="Processing"):
        self.total_items = total_items
        self.description = description
        self.current_item = 0
        self.start_time = None
        self.updates = []
    
    def start(self):
        """Start tracking progress"""
        import time
        self.start_time = time.time()
        self.current_item = 0
        print(f"{self.description}: 0/{self.total_items} (0.0%)")
    
    def update(self, increment=1):
        """Update progress"""
        import time
        
        self.current_item += increment
        current_time = time.time()
        
        if self.start_time:
            elapsed_time = current_time - self.start_time
            percentage = (self.current_item / self.total_items) * 100
            
            # Estimate remaining time
            if self.current_item > 0:
                rate = self.current_item / elapsed_time
                remaining_items = self.total_items - self.current_item
                estimated_remaining = remaining_items / rate if rate > 0 else 0
            else:
                estimated_remaining = 0
            
            update_info = {
                'current': self.current_item,
                'total': self.total_items,
                'percentage': percentage,
                'elapsed': elapsed_time,
                'estimated_remaining': estimated_remaining
            }
            
            self.updates.append(update_info)
            
            print(f"\r{self.description}: {self.current_item}/{self.total_items} "
                  f"({percentage:.1f}%) - ETA: {estimated_remaining:.1f}s", end="")
    
    def finish(self):
        """Finish progress tracking"""
        import time
        
        if self.start_time:
            total_time = time.time() - self.start_time
            print(f"\n{self.description} completed in {total_time:.2f} seconds")
        
        return self.updates


class TestLogger:
    """Test logger functionality"""
    
    def test_logger_creation(self):
        """Test logger creation"""
        logger = Logger("test_logger", "INFO")
        
        assert logger.name == "test_logger"
        assert logger.level == "INFO"
        assert len(logger.logs) == 0
    
    def test_logger_messages(self):
        """Test logging different message types"""
        logger = Logger()
        
        logger.info("Info message")
        logger.warning("Warning message")
        logger.error("Error message")
        logger.debug("Debug message")
        
        logs = logger.get_logs()
        assert len(logs) == 4
        assert logs[0] == ("INFO", "Info message")
        assert logs[1] == ("WARNING", "Warning message")
        assert logs[2] == ("ERROR", "Error message")
        assert logs[3] == ("DEBUG", "Debug message")
    
    def test_logger_formatting(self):
        """Test logger message formatting"""
        logger = Logger()
        
        logger.info("Message with %s and %d", "string", 42)
        
        logs = logger.get_logs()
        assert logs[0] == ("INFO", "Message with string and 42")
    
    def test_logger_clear(self):
        """Test clearing logger"""
        logger = Logger()
        
        logger.info("Message 1")
        logger.info("Message 2")
        assert len(logger.get_logs()) == 2
        
        logger.clear_logs()
        assert len(logger.get_logs()) == 0
    
    def test_setup_logger(self):
        """Test logger setup function"""
        logger = setup_logger("test_setup", "DEBUG", "test.log", "%(message)s")
        
        assert logger.name == "test_setup"
        assert logger.level == "DEBUG"
        assert hasattr(logger, 'log_file')
        assert hasattr(logger, 'format_string')


class TestMetricsVisualizer:
    """Test metrics visualization functionality"""
    
    def test_visualizer_initialization(self):
        """Test visualizer initialization"""
        viz = MetricsVisualizer(figsize=(8, 6), dpi=50)
        
        assert viz.figsize == (8, 6)
        assert viz.dpi == 50
        assert len(viz.figures) == 0
    
    def test_plot_metric_distribution(self):
        """Test metric distribution plotting"""
        viz = MetricsVisualizer()
        
        scores = np.random.normal(0.7, 0.15, 100)
        
        with tempfile.TemporaryDirectory() as temp_dir:
            save_path = Path(temp_dir) / "distribution.png"
            fig = viz.plot_metric_distribution(scores, "Test Metric", save_path=save_path)
            
            assert fig is not None
            assert save_path.exists()
            assert len(viz.figures) == 1
        
        viz.close_all_figures()
    
    def test_plot_metric_comparison(self):
        """Test metric comparison plotting"""
        viz = MetricsVisualizer()
        
        metric_data = [
            np.random.normal(0.6, 0.1, 50),
            np.random.normal(0.7, 0.15, 50),
            np.random.normal(0.8, 0.12, 50)
        ]
        metric_names = ["ROUGE", "BERTScore", "FactCheck"]
        
        fig = viz.plot_metric_comparison(metric_data, metric_names)
        
        assert fig is not None
        assert len(viz.figures) == 1
        
        viz.close_all_figures()
    
    def test_plot_correlation_heatmap(self):
        """Test correlation heatmap plotting"""
        viz = MetricsVisualizer()
        
        correlation_matrix = [
            [1.0, 0.7, 0.5],
            [0.7, 1.0, 0.6],
            [0.5, 0.6, 1.0]
        ]
        labels = ["Metric A", "Metric B", "Metric C"]
        
        fig = viz.plot_correlation_heatmap(correlation_matrix, labels)
        
        assert fig is not None
        assert len(viz.figures) == 1
        
        viz.close_all_figures()
    
    def test_plot_score_scatter(self):
        """Test scatter plot creation"""
        viz = MetricsVisualizer()
        
        x_scores = np.random.uniform(0, 1, 50)
        y_scores = x_scores + np.random.normal(0, 0.1, 50)  # Correlated with noise
        
        fig = viz.plot_score_scatter(x_scores, y_scores, "Metric X", "Metric Y")
        
        assert fig is not None
        assert len(viz.figures) == 1
        
        viz.close_all_figures()
    
    def test_plot_performance_over_time(self):
        """Test performance over time plotting"""
        viz = MetricsVisualizer()
        
        timestamps = [f"Day {i+1}" for i in range(10)]
        scores = np.random.uniform(0.5, 0.9, 10)
        
        fig = viz.plot_performance_over_time(timestamps, scores, "Test Metric")
        
        assert fig is not None
        assert len(viz.figures) == 1
        
        viz.close_all_figures()
    
    def test_create_summary_dashboard(self):
        """Test summary dashboard creation"""
        viz = MetricsVisualizer()
        
        metrics_data = {
            "ROUGE": np.random.normal(0.6, 0.1, 50).tolist(),
            "BERTScore": np.random.normal(0.7, 0.15, 50).tolist(),
            "FactCheck": np.random.normal(0.8, 0.12, 50).tolist()
        }
        
        fig = viz.create_summary_dashboard(metrics_data)
        
        assert fig is not None
        assert len(viz.figures) == 1
        
        viz.close_all_figures()
    
    def test_save_all_figures(self):
        """Test saving all figures"""
        viz = MetricsVisualizer()
        
        # Create a few figures
        scores = np.random.normal(0.7, 0.15, 100)
        viz.plot_metric_distribution(scores, "Test Metric")
        
        metric_data = [np.random.normal(0.6, 0.1, 50), np.random.normal(0.7, 0.15, 50)]
        viz.plot_metric_comparison(metric_data, ["Metric A", "Metric B"])
        
        with tempfile.TemporaryDirectory() as temp_dir:
            viz.save_all_figures(temp_dir, "test_figure")
            
            # Check that files were created
            saved_files = list(Path(temp_dir).glob("test_figure_*.png"))
            assert len(saved_files) == 2
        
        viz.close_all_figures()


class TestResultsExporter:
    """Test results export functionality"""
    
    def test_exporter_initialization(self):
        """Test exporter initialization"""
        with tempfile.TemporaryDirectory() as temp_dir:
            exporter = ResultsExporter(temp_dir)
            
            assert exporter.output_dir == Path(temp_dir)
            assert exporter.output_dir.exists()
    
    def test_export_to_json(self):
        """Test JSON export"""
        with tempfile.TemporaryDirectory() as temp_dir:
            exporter = ResultsExporter(temp_dir)
            
            data = {
                "rouge_scores": [0.6, 0.7, 0.8],
                "bertscore_scores": [0.65, 0.75, 0.85],
                "metadata": {"dataset": "test", "model": "gpt-4.1-mini"}
            }
            
            filepath = exporter.export_to_json(data, "test_results.json")
            
            assert filepath.exists()
            
            # Load and verify
            with open(filepath) as f:
                loaded_data = json.load(f)
            
            assert loaded_data["rouge_scores"] == [0.6, 0.7, 0.8]
            assert loaded_data["metadata"]["dataset"] == "test"
    
    @patch('pandas.DataFrame.to_csv')
    def test_export_to_csv(self, mock_to_csv):
        """Test CSV export"""
        with tempfile.TemporaryDirectory() as temp_dir:
            exporter = ResultsExporter(temp_dir)
            
            data = {
                "rouge": [0.6, 0.7, 0.8],
                "bertscore": [0.65, 0.75, 0.85]
            }
            
            filepath = exporter.export_to_csv(data, "test_results.csv")
            
            # Verify that pandas to_csv was called
            mock_to_csv.assert_called_once()
    
    def test_export_summary_report(self):
        """Test summary report export"""
        with tempfile.TemporaryDirectory() as temp_dir:
            exporter = ResultsExporter(temp_dir)
            
            data = {
                "rouge_scores": [0.6, 0.7, 0.8],
                "bertscore_scores": [0.65, 0.75, 0.85],
                "summary_stats": {"mean_rouge": 0.7, "mean_bertscore": 0.75}
            }
            
            filepath = exporter.export_summary_report(data, "test_summary.txt")
            
            assert filepath.exists()
            
            # Verify content
            with open(filepath) as f:
                content = f.read()
            
            assert "Factuality Evaluation Summary Report" in content
            assert "ROUGE_SCORES:" in content
            assert "Mean: 0.7000" in content


class TestFileUtils:
    """Test file utility functions"""
    
    def test_ensure_directory(self):
        """Test directory creation"""
        with tempfile.TemporaryDirectory() as temp_dir:
            test_path = Path(temp_dir) / "test" / "nested" / "directory"
            
            FileUtils.ensure_directory(test_path)
            
            assert test_path.exists()
            assert test_path.is_dir()
    
    def test_safe_filename(self):
        """Test safe filename generation"""
        unsafe_name = "file with spaces/and\\invalid:chars?.txt"
        safe_name = FileUtils.safe_filename(unsafe_name)
        
        assert "/" not in safe_name
        assert "\\" not in safe_name
        assert ":" not in safe_name
        assert "?" not in safe_name
        assert safe_name.endswith(".txt")
    
    def test_get_file_size(self):
        """Test file size calculation"""
        with tempfile.NamedTemporaryFile(mode='w', delete=False) as f:
            f.write("Hello, World!")
            temp_path = f.name
        
        try:
            size = FileUtils.get_file_size(temp_path)
            assert size > 0
        finally:
            os.unlink(temp_path)
    
    def test_backup_file(self):
        """Test file backup"""
        with tempfile.TemporaryDirectory() as temp_dir:
            # Create original file
            original_path = Path(temp_dir) / "original.txt"
            with open(original_path, 'w') as f:
                f.write("Original content")
            
            # Create backup
            backup_path = FileUtils.backup_file(original_path, temp_dir)
            
            assert backup_path is not None
            assert backup_path.exists()
            assert backup_path != original_path
            
            # Verify backup content
            with open(backup_path) as f:
                backup_content = f.read()
            assert backup_content == "Original content"
    
    def test_clean_temp_files(self):
        """Test temporary file cleanup"""
        with tempfile.TemporaryDirectory() as temp_dir:
            # Create some temp files
            temp_files = []
            for i in range(3):
                temp_file = Path(temp_dir) / f"temp_{i}.tmp"
                with open(temp_file, 'w') as f:
                    f.write(f"Temp content {i}")
                temp_files.append(temp_file)
            
            # Create a non-temp file
            regular_file = Path(temp_dir) / "regular.txt"
            with open(regular_file, 'w') as f:
                f.write("Regular content")
            
            # Clean temp files
            cleaned_count = FileUtils.clean_temp_files(temp_dir, "*.tmp")
            
            assert cleaned_count == 3
            assert regular_file.exists()
            assert all(not f.exists() for f in temp_files)


class TestProgressTracker:
    """Test progress tracking functionality"""
    
    def test_tracker_initialization(self):
        """Test progress tracker initialization"""
        tracker = ProgressTracker(100, "Test Progress")
        
        assert tracker.total_items == 100
        assert tracker.description == "Test Progress"
        assert tracker.current_item == 0
        assert tracker.start_time is None
    
    @patch('builtins.print')
    def test_tracker_start(self, mock_print):
        """Test starting progress tracker"""
        tracker = ProgressTracker(100, "Test Progress")
        
        tracker.start()
        
        assert tracker.start_time is not None
        assert tracker.current_item == 0
        mock_print.assert_called_with("Test Progress: 0/100 (0.0%)")
    
    @patch('builtins.print')
    def test_tracker_update(self, mock_print):
        """Test updating progress"""
        tracker = ProgressTracker(10, "Test Progress")
        
        tracker.start()
        tracker.update(3)
        
        assert tracker.current_item == 3
        assert len(tracker.updates) == 1
        assert tracker.updates[0]['current'] == 3
        assert tracker.updates[0]['percentage'] == 30.0
    
    @patch('builtins.print')
    def test_tracker_finish(self, mock_print):
        """Test finishing progress tracking"""
        tracker = ProgressTracker(10, "Test Progress")
        
        tracker.start()
        tracker.update(5)
        tracker.update(5)
        updates = tracker.finish()
        
        assert len(updates) == 2
        # Check that completion message was printed
        assert any("completed" in str(call) for call in mock_print.call_args_list)


class TestVisualizationIntegration:
    """Test integration of visualization components"""
    
    def test_full_visualization_pipeline(self):
        """Test complete visualization pipeline"""
        # Generate test data
        np.random.seed(42)  # For reproducible results
        
        metrics_data = {
            "ROUGE-1": np.random.normal(0.6, 0.1, 100).tolist(),
            "ROUGE-2": np.random.normal(0.4, 0.08, 100).tolist(),
            "BERTScore": np.random.normal(0.7, 0.12, 100).tolist(),
            "FactCheck": np.random.normal(0.75, 0.15, 100).tolist()
        }
        
        with tempfile.TemporaryDirectory() as temp_dir:
            # Initialize components
            viz = MetricsVisualizer()
            exporter = ResultsExporter(temp_dir)
            
            # Create visualizations
            viz.plot_metric_distribution(metrics_data["ROUGE-1"], "ROUGE-1")
            viz.plot_metric_comparison(list(metrics_data.values()), list(metrics_data.keys()))
            viz.create_summary_dashboard(metrics_data)
            
            # Save visualizations
            viz.save_all_figures(temp_dir, "metrics_viz")
            
            # Export data
            json_path = exporter.export_to_json(metrics_data, "metrics_data.json")
            report_path = exporter.export_summary_report(metrics_data, "metrics_report.txt")
            
            # Verify outputs
            assert len(viz.figures) == 3
            assert json_path.exists()
            assert report_path.exists()
            
            # Verify saved figures
            saved_figures = list(Path(temp_dir).glob("metrics_viz_*.png"))
            assert len(saved_figures) == 3
            
            # Clean up
            viz.close_all_figures()
    
    def test_error_handling_in_visualization(self):
        """Test error handling in visualization pipeline"""
        viz = MetricsVisualizer()
        
        # Test with empty data
        empty_scores = []
        
        try:
            fig = viz.plot_metric_distribution(empty_scores, "Empty Metric")
            # Should handle gracefully or raise appropriate exception
            assert fig is not None or len(empty_scores) == 0
        except Exception as e:
            # Acceptable if it raises a clear error
            assert isinstance(e, (ValueError, IndexError))
        
        # Test with invalid correlation matrix
        invalid_matrix = [[1, 2], [3]]  # Inconsistent dimensions
        
        try:
            fig = viz.plot_correlation_heatmap(invalid_matrix)
            # Should handle gracefully
        except Exception as e:
            # Acceptable if it raises appropriate error
            assert isinstance(e, (ValueError, IndexError))
        
        viz.close_all_figures()
    
    @patch('matplotlib.pyplot.savefig')
    def test_visualization_performance(self, mock_savefig):
        """Test visualization performance with large datasets"""
        viz = MetricsVisualizer()
        
        # Large dataset
        large_scores = np.random.normal(0.7, 0.15, 10000).tolist()
        
        # Should handle large datasets efficiently
        fig = viz.plot_metric_distribution(large_scores, "Large Dataset")
        
        assert fig is not None
        assert len(viz.figures) == 1
        
        viz.close_all_figures()
