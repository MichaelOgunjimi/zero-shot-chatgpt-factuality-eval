#!/usr/bin/env python3
"""
Script Index and Usage Guide
============================

This script provides an overview of all available scripts and their usage.

Usage:
    python scripts/index.py
    python scripts/index.py --script <script_name>

Author: Michael Ogunjimi
Institution: University of Manchester
Course: MSc AI
"""

import argparse
import sys
from pathlib import Path

def show_all_scripts():
    """Show all available scripts with descriptions."""
    scripts = {
        "setup_environment.py": {
            "description": "Complete environment setup including dependencies and configuration",
            "usage": [
                "python scripts/setup_environment.py",
                "python scripts/setup_environment.py --check-only",
                "python scripts/setup_environment.py --install-deps"
            ],
            "purpose": "First-time setup of the development environment"
        },
        "setup_data.py": {
            "description": "Download and prepare datasets for experiments",
            "usage": [
                "python scripts/setup_data.py",
                "python scripts/setup_data.py --quick-setup",
                "python scripts/setup_data.py --validate-only"
            ],
            "purpose": "Data preparation and validation"
        },
        "quick_test.py": {
            "description": "Quick tests to validate environment setup",
            "usage": [
                "python scripts/quick_test.py",
                "python scripts/quick_test.py --minimal",
                "python scripts/quick_test.py --full-validation"
            ],
            "purpose": "Environment validation and testing"
        },
        "estimate_costs.py": {
            "description": "Comprehensive API cost estimation with model comparison and budget planning",
            "usage": [
                "python scripts/estimate_costs.py",
                "python scripts/estimate_costs.py --experiment thesis_chatgpt_evaluation",
                "python scripts/estimate_costs.py --experiment quick_test --compare-models",
                "python scripts/estimate_costs.py --monthly-budget thesis_chatgpt_evaluation thesis_prompt_comparison",
                "python scripts/estimate_costs.py --model gpt-4.1-mini --experiment comprehensive_thesis"
            ],
            "purpose": "Advanced budget planning with thesis-scale cost estimation and model comparison"
        },
        "check_environment.py": {
            "description": "Comprehensive environment health check",
            "usage": [
                "python scripts/check_environment.py",
                "python scripts/check_environment.py --detailed",
                "python scripts/check_environment.py --fix-issues"
            ],
            "purpose": "Detailed environment diagnostics"
        }
    }
    
    print("=" * 80)
    print("FACTUALITY EVALUATION SCRIPTS")
    print("=" * 80)
    print()
    
    for script_name, info in scripts.items():
        print(f"📄 {script_name}")
        print(f"   Description: {info['description']}")
        print(f"   Purpose: {info['purpose']}")
        print(f"   Usage:")
        for usage in info['usage']:
            print(f"     {usage}")
        print()
    
    print("=" * 80)
    print("RECOMMENDED SETUP SEQUENCE")
    print("=" * 80)
    print("1. python scripts/setup_environment.py")
    print("2. python scripts/setup_data.py")
    print("3. python scripts/quick_test.py")
    print("4. python scripts/estimate_costs.py")
    print("5. python scripts/estimate_costs.py --experiment thesis_chatgpt_evaluation --compare-models")
    print("6. python experiments/run_all_experiments.py --quick-test")
    print()
    print("For thesis-scale experiments:")
    print("  python scripts/estimate_costs.py --monthly-budget thesis_chatgpt_evaluation thesis_prompt_comparison thesis_sota_comparison")
    print()
    print("For detailed help on any script:")
    print("  python scripts/<script_name> --help")
    print("=" * 80)

def show_specific_script(script_name):
    """Show detailed information about a specific script."""
    script_descriptions = {
        "setup_environment": {
            "name": "Environment Setup",
            "description": "Sets up the complete development environment",
            "features": [
                "Checks Python version compatibility",
                "Installs dependencies from requirements.txt",
                "Creates necessary directories",
                "Sets up .env file from template",
                "Validates project structure",
                "Tests basic imports"
            ],
            "options": [
                "--check-only: Only check environment without changes",
                "--install-deps: Only install dependencies"
            ]
        },
        "setup_data": {
            "name": "Data Setup",
            "description": "Downloads and prepares datasets for experiments",
            "features": [
                "Downloads CNN/DailyMail and XSum datasets",
                "Preprocesses data for all tasks",
                "Validates data integrity",
                "Creates processed data files",
                "Provides data statistics"
            ],
            "options": [
                "--quick-setup: Download small samples for testing",
                "--validate-only: Only validate existing data",
                "--clean-and-reload: Clean and reload all data"
            ]
        },
        "quick_test": {
            "name": "Quick Test",
            "description": "Validates environment setup with quick tests",
            "features": [
                "Tests all module imports",
                "Validates configuration loading",
                "Tests data loading functionality",
                "Checks task creation",
                "Tests LLM client setup",
                "Validates evaluation pipeline"
            ],
            "options": [
                "--minimal: Run minimal test suite",
                "--full-validation: Run full validation including experiment"
            ]
        },
        "estimate_costs": {
            "name": "Advanced Cost Estimation",
            "description": "Comprehensive API cost estimation with model comparison and budget planning",
            "features": [
                "Estimates costs for 9 different experiment configurations",
                "Compares costs across 7 OpenAI models (gpt-4.1-mini, o1-mini, gpt-4o, o4-mini, etc.)",
                "Provides detailed cost breakdowns by task, dataset, and prompt type",
                "Calculates thesis-scale experiment costs (10k/2k/1.5k samples)",
                "Shows cost per sample efficiency metrics",
                "Provides monthly budget scenarios and recommendations",
                "Includes model comparison with savings analysis",
                "Supports custom configuration files",
                "Categorizes experiments by development/standard/comprehensive/thesis scale"
            ],
            "options": [
                "--experiment <name>: Estimate specific experiment (quick_test, development, thesis_chatgpt_evaluation, etc.)",
                "--compare-models: Compare costs across all available models",
                "--monthly-budget <exp1> <exp2>: Estimate monthly budget for selected experiments",
                "--model <model_name>: Override model for estimation",
                "--custom-config <path>: Use custom configuration file"
            ]
        },
        "check_environment": {
            "name": "Environment Check",
            "description": "Comprehensive environment health check",
            "features": [
                "Checks Python environment and dependencies",
                "Validates API configuration",
                "Tests project structure",
                "Checks data availability",
                "Validates configuration files",
                "Tests project imports",
                "Checks GPU availability"
            ],
            "options": [
                "--detailed: Show detailed output",
                "--fix-issues: Attempt to fix common issues"
            ]
        }
    }
    
    if script_name in script_descriptions:
        info = script_descriptions[script_name]
        print(f"=" * 60)
        print(f"{info['name'].upper()}")
        print(f"=" * 60)
        print(f"Description: {info['description']}")
        print()
        print("Features:")
        for feature in info['features']:
            print(f"  • {feature}")
        print()
        print("Options:")
        for option in info['options']:
            print(f"  • {option}")
        print()
        print(f"Usage: python scripts/{script_name}.py --help")
        print("=" * 60)
    else:
        print(f"Unknown script: {script_name}")
        print("Available scripts: setup_environment, setup_data, quick_test, estimate_costs, check_environment")

def main():
    """Main entry point."""
    parser = argparse.ArgumentParser(description="Script index and usage guide")
    parser.add_argument("--script", type=str, help="Show detailed info about specific script")
    
    args = parser.parse_args()
    
    if args.script:
        show_specific_script(args.script)
    else:
        show_all_scripts()

if __name__ == "__main__":
    main()
