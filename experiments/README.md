# ChatGPT Factuality Evaluation - Experiment Guide

**Author**: Michael Ogunjimi  
**Institution**: University of Manchester, MSc AI  
**Project**: ChatGPT Zero-Shot Factuality Evaluation for Text Summarization  

## Table of Contents

1. [Quick Start](#quick-start)
2. [Experimental Framework Overview](#experimental-framework-overview)
3. [Individual Experiments](#individual-experiments)
4. [Master Experiment Suite](#master-experiment-suite)
5. [Configuration Management](#configuration-management)
6. [Results Analysis](#results-analysis)
7. [Troubleshooting](#troubleshooting)
8. [Reproducibility Guidelines](#reproducibility-guidelines)

---

## Quick Start

### Prerequisites

1. **Environment Setup**

   ```bash
   # Clone repository and set up environment
   cd chatgpt-factuality-eval
   chmod +x setup.sh && ./setup.sh
   source venv/bin/activate
   ```

2. **API Configuration**

   ```bash
   # Add your OpenAI API key
   cp .env.template .env
   echo "OPENAI_API_KEY=sk-your-key-here" >> .env
   ```

3. **Quick Validation**

   ```bash
   # Validate setup (recommended first step)
   python experiments/quick_experiment_setup.py --complete-validation
   ```

### 30-Second Test Run

```bash
# Run complete experimental suite with minimal data
python experiments/run_all_experiments.py --quick-test
```

This will execute all three core experiments with small datasets (~10 examples each) to verify everything works correctly before running full experiments.

---

## Experimental Framework Overview

### Core Experiments

The thesis includes three main experimental components:

| Experiment | Script | Purpose | Outputs |
|------------|--------|---------|---------|
| **ChatGPT Evaluation** | `run_chatgpt_evaluation.py` | Core performance assessment | Task performance metrics |
| **Prompt Comparison** | `prompt_comparison.py` | Zero-shot vs CoT analysis | Prompt effectiveness analysis |
| **SOTA Comparison** | `sota_comparison.py` | Baseline correlation study | Correlation with traditional metrics |

### Master Runner

- **Script**: `run_all_experiments.py`
- **Purpose**: Orchestrates all experiments and generates consolidated thesis-ready reports
- **Recommended**: Use this for final thesis experiments

---

## Individual Experiments

### 1. ChatGPT Evaluation Experiment

**Purpose**: Assess ChatGPT's performance across three factuality evaluation tasks.

#### Running ChatGPT Evaluation

```bash
# Run complete ChatGPT evaluation
python experiments/run_chatgpt_evaluation.py --config config/default.yaml

# Quick test with minimal data
python experiments/run_chatgpt_evaluation.py --quick-test

# Single task evaluation
python experiments/run_chatgpt_evaluation.py --task entailment_inference --dataset cnn_dailymail

# Chain-of-thought prompting
python experiments/run_chatgpt_evaluation.py --prompt-type chain_of_thought
```

#### Advanced Options

```bash
# Custom sample size
python experiments/run_chatgpt_evaluation.py --sample-size 200

# Specific configuration
python experiments/run_chatgpt_evaluation.py --config config/custom_config.yaml --experiment-name my_experiment
```

#### Expected Outputs

```text
results/experiments/chatgpt_eval_YYYYMMDD_HHMMSS/
├── results.json                    # Raw experimental results
├── experiment_report.md            # Human-readable report
├── figures/
│   └── task_performance_comparison.png
└── logs/
    └── experiment.log              # Detailed execution logs
```

#### Key Metrics Generated

- **Entailment Inference**: Accuracy, Precision, Recall, F1-score
- **Summary Ranking**: Kendall's τ, Spearman's ρ, NDCG
- **Consistency Rating**: Pearson correlation, MAE, RMSE

---

### 2. Prompt Comparison Experiment

**Purpose**: Compare zero-shot versus chain-of-thought prompting effectiveness.

#### Basic Usage

```bash
# Run complete prompt comparison
python experiments/prompt_comparison.py --config config/default.yaml

# Quick test
python experiments/prompt_comparison.py --quick-test

# Single task comparison
python experiments/prompt_comparison.py --task entailment_inference --dataset cnn_dailymail
```

#### Expected Outputs

```text
results/experiments/prompt_comparison_YYYYMMDD_HHMMSS/
├── prompt_comparison_results.json
├── prompt_comparison_report.md
├── figures/
│   ├── prompt_performance_comparison.png
│   ├── improvement_analysis.png
│   ├── cost_benefit_analysis.png
│   └── statistical_significance.png
└── logs/
```

#### Key Analyses

- **Performance Comparison**: Direct comparison between prompt types
- **Improvement Analysis**: Relative and absolute improvements
- **Cost-Benefit Analysis**: Performance vs cost trade-offs
- **Statistical Significance**: Paired t-tests and effect sizes

---

### 3. SOTA Comparison Experiment

**Purpose**: Correlate ChatGPT evaluations with established baseline methods.

#### Basic Usage

```bash
# Run complete SOTA comparison
python experiments/sota_comparison.py --config config/default.yaml

# Quick test
python experiments/sota_comparison.py --quick-test

# Single baseline comparison
python experiments/sota_comparison.py --baseline factcc --task entailment_inference
```

#### Supported Baselines

- **FactCC**: BERT-based factual consistency classifier
- **BERTScore**: Contextual embedding similarity
- **ROUGE**: N-gram overlap metrics

#### Expected Outputs

```text
results/experiments/sota_comparison_YYYYMMDD_HHMMSS/
├── sota_comparison_results.json
├── sota_comparison_report.md
├── figures/
│   ├── correlation_heatmap.png
│   ├── baseline_performance_comparison.png
│   ├── correlation_scatter_plots.png
│   └── method_ranking.png
└── logs/
```

#### Key Analyses

- **Correlation Analysis**: Pearson and Spearman correlations
- **Statistical Significance**: P-values and effect sizes
- **Method Ranking**: Baseline performance comparison
- **Scatter Plot Analysis**: Detailed correlation visualization

---

## Master Experiment Suite

### Purpose

The master experiment runner (`run_all_experiments.py`) orchestrates all three experiments in sequence and generates consolidated thesis-ready reports.

### Recommended Usage

#### For Thesis Final Results

```bash
# Complete experimental suite (recommended for thesis)
python experiments/run_all_experiments.py --config config/default.yaml --experiment-name thesis_final
```

#### For Development and Testing

```bash
# Quick validation run
python experiments/run_all_experiments.py --quick-test
```

### Master Outputs

```text
results/experiments/master_experiment_YYYYMMDD_HHMMSS/
├── master_experimental_results.json       # Consolidated results
├── master_experimental_report.md          # Comprehensive thesis report
├── executive_summary.md                   # Executive summary
├── master_visualizations/
│   ├── cost_breakdown.png
│   ├── performance_dashboard.png
│   ├── key_findings_summary.png
│   └── experimental_timeline.png
├── individual_experiments/                # Links to individual results
└── logs/
```

### Resource Planning

#### Estimated Costs (OpenAI API)

| Configuration | Cost Estimate | Duration | Samples |
|---------------|---------------|----------|---------|
| Quick Test | $2-5 | 5-10 min | ~50 total examples |
| Development | $10-20 | 20-30 min | ~200 total examples |
| Thesis Final | $50-100 | 1-2 hours | ~1000+ total examples |

#### Cost Optimization Tips

1. **Start with Quick Test**: Always validate with `--quick-test` first
2. **Use Sample Size Limits**: Specify `--sample-size` for development
3. **Monitor Costs**: Check logs for running cost totals
4. **Incremental Approach**: Run individual experiments first

---

## Configuration Management

### Default Configuration

The `config/default.yaml` file contains all experimental parameters:

```yaml
# Key configuration sections
llm_clients:
  openai:
    api_key: ${OPENAI_API_KEY}
    model: "gpt-4.1-mini"
    rate_limit: 50  # requests per minute

datasets:
  cnn_dailymail:
    sample_sizes:
      development: 50
      evaluation: 500
      full: null

experiments:
  main_experiments:
    prompt_comparison:
      sample_size: 200
    sota_comparison:
      sample_size: 300
```

### Custom Configurations

Create custom config files for specific experimental setups:

```bash
# Copy default configuration
cp config/default.yaml config/my_experiment.yaml

# Edit configuration
nano config/my_experiment.yaml

# Run with custom config
python experiments/run_all_experiments.py --config config/my_experiment.yaml
```

### Environment Variables

Set up environment variables in `.env`:

```bash
# Required
OPENAI_API_KEY=sk-your-key-here

# Optional
EXPERIMENT_NAME=my_thesis_experiment
OUTPUT_DIR=custom_results
```

---

## Results Analysis

### Understanding Output Structure

Each experiment generates standardized outputs:

#### JSON Results (`*_results.json`)

```json
{
  "experiment_metadata": {
    "name": "experiment_name",
    "timestamp": "2024-07-10T14:30:22",
    "config": {...}
  },
  "task_results": {
    "entailment_inference": {
      "cnn_dailymail": {
        "performance_metrics": {
          "primary_metric": 0.85,
          "accuracy": 0.85,
          "precision": 0.83,
          "recall": 0.87,
          "f1_score": 0.85
        }
      }
    }
  }
}
```

#### Markdown Reports (`*_report.md`)

Human-readable reports include:

- Executive summary
- Experimental setup
- Detailed results
- Statistical analysis
- Key findings and recommendations

### Key Metrics Interpretation

#### Performance Metrics

- **Accuracy**: Overall correctness (higher = better)
- **Correlation**: Agreement with baselines (-1 to 1, closer to ±1 = better)
- **F1-Score**: Balanced precision/recall (0 to 1, higher = better)

#### Statistical Significance

- **p-value < 0.05**: Statistically significant difference
- **Effect size**: Practical significance (small/medium/large)
- **Confidence intervals**: Uncertainty bounds

#### Cost Metrics

- **Total Cost**: API usage in USD
- **Cost per Example**: Efficiency metric
- **Cost Ratio**: Relative cost comparison

---

## Troubleshooting

### Common Issues and Solutions

#### 1. API Key Issues

**Problem**: `OpenAI API key not found` or authentication errors

**Solutions**:

```bash
# Check API key setup
cat .env | grep OPENAI_API_KEY

# Validate API key
python -c "import openai; print('API key valid')"

# Test API access
python experiments/quick_experiment_setup.py --validate-environment
```

#### 2. Memory Issues

**Problem**: Out of memory errors with large datasets

**Solutions**:

```bash
# Reduce sample size
python experiments/run_chatgpt_evaluation.py --sample-size 100

# Use quick test mode
python experiments/run_all_experiments.py --quick-test

# Monitor memory usage
python experiments/quick_experiment_setup.py --benchmark-performance
```

#### 3. Dataset Loading Errors

**Problem**: Cannot load datasets or missing data files

**Solutions**:

```bash
# Download datasets manually
python scripts/download_all_datasets.py

# Validate data loading
python -c "from src.data import quick_load_dataset; print(len(quick_load_dataset('cnn_dailymail', 5)))"

# Check data directory
ls -la data/
```

#### 4. Baseline Model Issues

**Problem**: FactCC or BERTScore initialization fails

**Solutions**:

```bash
# Install missing dependencies
pip install bert-score factcc transformers

# Test baseline creation
python -c "from src.baselines import create_baseline; create_baseline('bertscore')"

# Skip problematic baselines
python experiments/sota_comparison.py --baseline bertscore  # single baseline
```

#### 5. Visualization Issues

**Problem**: Cannot generate plots or missing visualization libraries

**Solutions**:

```bash
# Install visualization dependencies
pip install plotly kaleido matplotlib seaborn

# Test visualization creation
python -c "import plotly.graph_objects as go; print('Plotly working')"

# Skip visualization generation
# Edit config to disable: visualization.generate_plots: false
```

### Performance Optimization

#### Speed Optimization

```bash
# Use smaller sample sizes for development
python experiments/run_chatgpt_evaluation.py --sample-size 50

# Run single tasks
python experiments/run_chatgpt_evaluation.py --task entailment_inference

# Parallel processing (if supported)
# Edit config: experiments.general.parallel_processing: true
```

#### Cost Optimization

```bash
# Use GPT-4.1 Mini for optimal cost-performance balance
# Edit config: llm_clients.openai.model: "gpt-4.1-mini"

# Reduce rate limits to control costs
# Edit config: llm_clients.openai.rate_limit: 20

# Set cost limits
# Edit config: llm_clients.openai.daily_cost_limit: 25.0
```

---

## Reproducibility Guidelines

### For Thesis Submission

1. **Document Configuration**

   ```bash
   # Save final configuration
   cp config/default.yaml thesis_config.yaml
   
   # Document environment
   pip freeze > requirements_final.txt
   
   # Save system info
   python -c "import sys; print(sys.version)" > system_info.txt
   ```

2. **Set Reproducible Seeds**

   ```yaml
   # In config file
   evaluation:
     design:
       random_seed: 42
   ```

3. **Version Control**

   ```bash
   # Tag final experiment version
   git tag -a thesis-final -m "Final experimental version for thesis"
   
   # Document code version
   git rev-parse HEAD > code_version.txt
   ```

### For Publication

1. **Complete Data Documentation**
   - Dataset versions and access dates
   - Preprocessing parameters
   - Sampling procedures

2. **Experimental Parameters**
   - All hyperparameters
   - API model versions
   - Evaluation metrics

3. **Statistical Methods**
   - Significance tests used
   - Multiple comparison corrections
   - Effect size calculations

### Sharing Results

1. **Anonymize API Keys**

   ```bash
   # Clean config for sharing
   sed 's/sk-[a-zA-Z0-9]*/REDACTED_API_KEY/g' config/default.yaml > config/shared.yaml
   ```

2. **Package Results**

   ```bash
   # Create results package
   tar -czf thesis_results.tar.gz results/experiments/master_experiment_*/
   ```

3. **Documentation Package**

   ```bash
   # Include all documentation
   zip -r thesis_package.zip \
     experiments/ \
     config/ \
     docs/ \
     results/ \
     requirements.txt \
     README.md
   ```

---

## Advanced Usage

### Custom Experiment Design

#### Adding New Tasks

1. Create task class in `src/tasks/`
2. Add configuration in `config/default.yaml`
3. Update experiment scripts to include new task

#### Custom Prompts

1. Add prompt templates in `prompts/`
2. Update `src/prompts/prompt_manager.py`
3. Test with single examples before full experiments

#### New Baseline Methods

1. Implement baseline in `src/baselines/`
2. Add to baseline factory
3. Update SOTA comparison experiment

### Batch Experiment Execution

```bash
# Run multiple configurations
for config in config/experiment_*.yaml; do
    python experiments/run_all_experiments.py --config $config
done

# Parallel execution (use with caution for API limits)
parallel python experiments/run_chatgpt_evaluation.py --config {} ::: config/batch_*.yaml
```

### Results Aggregation

```bash
# Combine results from multiple experiments
python scripts/aggregate_results.py --input-dir results/experiments/ --output combined_analysis.json
```

---

## Support and Resources

### Getting Help

1. **Check Logs**: Always check `logs/` directory for detailed error messages
2. **Validate Setup**: Run `quick_experiment_setup.py --complete-validation`
3. **Test Components**: Use individual experiment scripts to isolate issues
4. **Monitor Resources**: Check API costs and rate limits

### Additional Resources

- **OpenAI API Documentation**: <https://docs.openai.com/>
- **Project Repository**: [Your GitHub repository]
- **Configuration Reference**: `config/default.yaml` with inline comments
- **Error Logs**: `results/experiments/*/logs/` for detailed troubleshooting

### Best Practices

1. **Always Start Small**: Use `--quick-test` for validation
2. **Monitor Costs**: Check logs regularly for API usage
3. **Save Configurations**: Document all experimental parameters
4. **Version Control**: Commit code before major experiments
5. **Backup Results**: Save important experimental outputs

---

## Conclusion

This experimental framework provides a comprehensive system for evaluating ChatGPT's factuality assessment capabilities. The modular design allows for flexible experimentation while maintaining reproducibility standards required for academic research.

For thesis work, we recommend:

1. **Start with validation**: `quick_experiment_setup.py --complete-validation`
2. **Test with minimal data**: `run_all_experiments.py --quick-test`
3. **Run full experiments**: `run_all_experiments.py --experiment-name thesis_final`
4. **Generate final reports**: All outputs will be thesis-ready

The framework generates publication-quality visualizations and comprehensive reports suitable for direct inclusion in academic writing.

---

*This guide is maintained as part of the ChatGPT Factuality Evaluation project. For updates and additional documentation, please refer to the project repository.*
