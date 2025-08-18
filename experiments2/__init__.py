"""
Experiments2 Package for Multi-LLM Factuality Evaluation
======================================================

This package contains experimental scripts for evaluating multiple Large Language Models
(GPT-4.1-mini, Qwen2.5:7b, Llama3.1:8b) on factuality assessment tasks compared against
SOTA baselines (FactCC, BERTScore, ROUGE).

Available Experiments:
- run_llm_evaluation: Multi-LLM factuality and capability evaluation across all tasks
- sota_multi_comparison: Enhanced SOTA baseline comparison with statistical analysis

Usage Examples:
    # As scripts
    python experiments2/run_llm_evaluation.py --config config/default.yaml
    python experiments2/run_llm_evaluation.py --quick-test
    python experiments2/sota_multi_comparison.py --dataset frank --task entailment_inference
    
    # As modules
    python -m experiments2.run_llm_evaluation --config config/default.yaml
    python -m experiments2.sota_multi_comparison --quick-test

Author: Michael Ogunjimi
Institution: University of Manchester
Course: MSc AI
Date: August 7, 2025
"""

__version__ = "2.0.0"
__author__ = "Michael Ogunjimi"
__email__ = "michael.ogunjimi@postgrad.manchester.ac.uk"
__institution__ = "University of Manchester"
__course__ = "MSc AI"

# Available experiment modules
AVAILABLE_EXPERIMENTS = [
    "run_llm_evaluation",
    "sota_multi_comparison",
]

# Experiment descriptions
EXPERIMENT_DESCRIPTIONS = {
    "run_llm_evaluation": {
        "name": "Multi-LLM Factuality and Capability Evaluation",
        "description": "Comprehensive evaluation of GPT-4.1-mini, Qwen2.5:7b, and Llama3.1:8b across all factuality tasks",
        "tasks": ["entailment_inference", "summary_ranking", "consistency_rating"],
        "models": ["gpt-4.1-mini", "qwen2.5:7b", "llama3.1:8b"],
        "datasets": ["frank", "summeval"],
        "sample_size": 1000,
        "features": [
            "Multi-model evaluation with statistical analysis",
            "Comprehensive visualizations (bar charts, radar charts, heatmaps)",
            "Performance tables with confidence intervals",
            "Failure mode analysis and error categorization"
        ],
        "output": "Performance metrics, visualizations, and comparative analysis"
    },
    "sota_multi_comparison": {
        "name": "Enhanced SOTA Baseline Comparison",
        "description": "Compare all three LLMs against FactCC, BERTScore, and ROUGE baselines with statistical testing",
        "tasks": ["entailment_inference", "consistency_rating"],
        "models": ["gpt-4.1-mini", "qwen2.5:7b", "llama3.1:8b"],
        "baselines": ["factcc", "bertscore", "rouge"],
        "datasets": ["frank", "summeval"],
        "sample_size": 500,
        "features": [
            "Statistical significance testing",
            "Interactive Plotly visualizations",
            "Correlation analysis between models and baselines",
            "Performance comparison matrices"
        ],
        "output": "Correlation analysis, statistical comparison, and interactive plots"
    },
}

# Supported models
SUPPORTED_MODELS = {
    "gpt-4.1-mini": {
        "provider": "openai",
        "api_base": "https://api.openai.com/v1",
        "requires_api_key": True,
        "cost_tracking": True,
        "description": "OpenAI's fast and efficient model for high-quality outputs"
    },
    "qwen2.5:7b": {
        "provider": "ollama",
        "api_base": "${OLLAMA_HOST}/v1",
        "requires_api_key": False,
        "cost_tracking": False,
        "description": "Alibaba's Qwen 2.5 7B parameter model via Ollama"
    },
    "llama3.1:8b": {
        "provider": "ollama", 
        "api_base": "${OLLAMA_HOST}/v1",
        "requires_api_key": False,
        "cost_tracking": False,
        "description": "Meta's Llama 3.1 8B parameter model via Ollama"
    }
}

# Supported datasets
SUPPORTED_DATASETS = {
    "frank": {
        "name": "FRANK",
        "description": "Factuality Ranking for Abstractive News Summarization",
        "tasks": ["entailment_inference", "summary_ranking", "consistency_rating"],
        "size": 2250,
        "domain": "news_summarization"
    },
    "summeval": {
        "name": "SummEval",
        "description": "Summarization evaluation benchmark with human annotations",
        "tasks": ["entailment_inference", "summary_ranking", "consistency_rating"],
        "size": 1600,
        "domain": "news_summarization"
    }
}

# Task configurations
TASK_CONFIGURATIONS = {
    "entailment_inference": {
        "description": "Binary classification of factual consistency (ENTAILMENT vs CONTRADICTION)",
        "output_format": "binary",
        "metrics": ["accuracy", "f1_score", "precision", "recall", "bertscore"],
        "sample_size": 1000
    },
    "summary_ranking": {
        "description": "Rank multiple summaries by factual consistency",
        "output_format": "ranked_list", 
        "metrics": ["kendall_tau", "spearman_rho", "ndcg", "bertscore"],
        "sample_size": 1000
    },
    "consistency_rating": {
        "description": "Quantitative rating of factual consistency (0-100 scale)",
        "output_format": "float",
        "metrics": ["pearson_correlation", "spearman_correlation", "mae", "rmse", "bertscore"],
        "sample_size": 1000
    }
}

# Baseline configurations
BASELINE_CONFIGURATIONS = {
    "factcc": {
        "name": "FactCC",
        "description": "BERT-based factual consistency classifier",
        "type": "neural",
        "tasks": ["entailment_inference", "consistency_rating"],
        "metrics": ["accuracy", "f1_score", "pearson_correlation"],
        "paper": "Kryscinski et al. (2020)"
    },
    "bertscore": {
        "name": "BERTScore",
        "description": "BERT-based semantic similarity metric",
        "type": "embedding",
        "tasks": ["entailment_inference", "consistency_rating"],
        "metrics": ["bertscore", "pearson_correlation"],
        "paper": "Zhang et al. (2020)"
    },
    "rouge": {
        "name": "ROUGE",
        "description": "Recall-oriented n-gram overlap metric",
        "type": "lexical",
        "tasks": ["entailment_inference", "consistency_rating"],
        "metrics": ["rouge_l", "pearson_correlation"],
        "paper": "Lin (2004)"
    }
}

# Visualization configurations
VISUALIZATION_CONFIG = {
    "style": "seaborn-v0_8",
    "color_palette": "Set2",
    "figure_size": (12, 8),
    "dpi": 300,
    "save_formats": ["png", "pdf"],
    "interactive": True
}

# Metric configurations
METRIC_CONFIGURATIONS = {
    "accuracy": {"higher_is_better": True, "range": [0, 1]},
    "f1_score": {"higher_is_better": True, "range": [0, 1]},
    "precision": {"higher_is_better": True, "range": [0, 1]},
    "recall": {"higher_is_better": True, "range": [0, 1]},
    "pearson_correlation": {"higher_is_better": True, "range": [-1, 1]},
    "spearman_correlation": {"higher_is_better": True, "range": [-1, 1]},
    "kendall_tau": {"higher_is_better": True, "range": [-1, 1]},
    "mae": {"higher_is_better": False, "range": [0, float('inf')]},
    "rmse": {"higher_is_better": False, "range": [0, float('inf')]},
    "bertscore": {"higher_is_better": True, "range": [0, 1]},
    "rouge_l": {"higher_is_better": True, "range": [0, 1]},
    "ndcg": {"higher_is_better": True, "range": [0, 1]}
}
