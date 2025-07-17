#!/usr/bin/env python3
"""
Environment Check Script for ChatGPT Factuality Evaluation
=========================================================

This script performs comprehensive environment checks to ensure all components
are properly configured and working correctly.

Usage:
    python scripts/check_environment.py
    python scripts/check_environment.py --fix-issues
    python scripts/check_environment.py --detailed

Author: Michael Ogunjimi
Institution: University of Manchester
Course: MSc AI
"""

import argparse
import logging
import os
import sys
import subprocess
from pathlib import Path
from typing import Dict, List, Optional, Tuple

# Add project root to path
sys.path.insert(0, str(Path(__file__).parent.parent))

# Setup logging
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)

class EnvironmentChecker:
    """Comprehensive environment checker for factuality evaluation."""
    
    def __init__(self, project_root: Path = None):
        """Initialize environment checker."""
        self.project_root = project_root or Path(__file__).parent.parent
        self.check_results = {}
        
    def check_python_environment(self) -> Dict[str, bool]:
        """Check Python environment and version."""
        logger.info("Checking Python environment...")
        
        results = {}
        
        # Python version
        version_info = sys.version_info
        python_ok = version_info.major >= 3 and version_info.minor >= 8
        results['python_version'] = python_ok
        
        if python_ok:
            logger.info(f"✓ Python {version_info.major}.{version_info.minor}.{version_info.micro} is compatible")
        else:
            logger.error(f"✗ Python {version_info.major}.{version_info.minor}.{version_info.micro} is too old (need 3.8+)")
        
        # Virtual environment
        in_venv = hasattr(sys, 'real_prefix') or (hasattr(sys, 'base_prefix') and sys.base_prefix != sys.prefix)
        results['virtual_environment'] = in_venv
        
        if in_venv:
            logger.info("✓ Running in virtual environment")
        else:
            logger.warning("⚠️  Not running in virtual environment (recommended)")
        
        # Pip version
        try:
            import pip
            logger.info(f"✓ Pip available: {pip.__version__}")
            results['pip_available'] = True
        except ImportError:
            logger.error("✗ Pip not available")
            results['pip_available'] = False
        
        return results
    
    def check_dependencies(self) -> Dict[str, bool]:
        """Check all required dependencies."""
        logger.info("Checking dependencies...")
        
        # Core dependencies with version requirements
        dependencies = {
            'torch': {'min_version': '1.9.0', 'required': True},
            'transformers': {'min_version': '4.20.0', 'required': True},
            'datasets': {'min_version': '2.0.0', 'required': True},
            'openai': {'min_version': '1.0.0', 'required': True},
            'pandas': {'min_version': '1.3.0', 'required': True},
            'numpy': {'min_version': '1.20.0', 'required': True},
            'scipy': {'min_version': '1.7.0', 'required': True},
            'scikit-learn': {'min_version': '1.0.0', 'required': True},
            'tqdm': {'min_version': '4.60.0', 'required': True},
            'pyyaml': {'min_version': '5.4.0', 'required': True},
            'plotly': {'min_version': '5.0.0', 'required': True},
            'matplotlib': {'min_version': '3.3.0', 'required': False},
            'seaborn': {'min_version': '0.11.0', 'required': False},
            'jupyter': {'min_version': '1.0.0', 'required': False},
        }
        
        results = {}
        
        for package, info in dependencies.items():
            try:
                module = __import__(package)
                version = getattr(module, '__version__', 'unknown')
                
                logger.info(f"✓ {package} {version} installed")
                results[package] = True
                
            except ImportError:
                if info['required']:
                    logger.error(f"✗ {package} not installed (required)")
                    results[package] = False
                else:
                    logger.warning(f"⚠️  {package} not installed (optional)")
                    results[package] = False
        
        return results
    
    def check_api_configuration(self) -> Dict[str, bool]:
        """Check API key configuration."""
        logger.info("Checking API configuration...")
        
        results = {}
        
        # OpenAI API key
        openai_key = os.getenv('OPENAI_API_KEY')
        if openai_key and openai_key != 'your-openai-api-key-here' and openai_key.startswith('sk-'):
            logger.info("✓ OpenAI API key configured")
            results['openai_key'] = True
            
            # Test API connection
            try:
                import openai
                client = openai.OpenAI(api_key=openai_key)
                # Simple test call
                response = client.chat.completions.create(
                    model="gpt-3.5-turbo",
                    messages=[{"role": "user", "content": "Hello"}],
                    max_tokens=5
                )
                logger.info("✓ OpenAI API connection successful")
                results['openai_connection'] = True
            except Exception as e:
                logger.error(f"✗ OpenAI API connection failed: {e}")
                results['openai_connection'] = False
        else:
            logger.error("✗ OpenAI API key not configured or invalid")
            results['openai_key'] = False
            results['openai_connection'] = False
        
        return results
    
    def check_project_structure(self) -> Dict[str, bool]:
        """Check project directory structure."""
        logger.info("Checking project structure...")
        
        required_dirs = [
            'src',
            'src/utils',
            'src/data',
            'src/tasks',
            'src/evaluation',
            'src/llm_clients',
            'src/prompts',
            'src/baselines',
            'experiments',
            'config',
            'prompts',
            'data',
            'results',
            'tests'
        ]
        
        required_files = [
            'requirements.txt',
            'config/default.yaml',
            '.env.template',
            'src/__init__.py',
            'experiments/__init__.py',
            'experiments/run_all_experiments.py',
            'experiments/run_chatgpt_evaluation.py'
        ]
        
        results = {}
        
        # Check directories
        for directory in required_dirs:
            dir_path = self.project_root / directory
            if dir_path.exists() and dir_path.is_dir():
                logger.info(f"✓ Directory exists: {directory}")
                results[f"dir_{directory}"] = True
            else:
                logger.error(f"✗ Directory missing: {directory}")
                results[f"dir_{directory}"] = False
        
        # Check files
        for file_path in required_files:
            full_path = self.project_root / file_path
            if full_path.exists() and full_path.is_file():
                logger.info(f"✓ File exists: {file_path}")
                results[f"file_{file_path}"] = True
            else:
                logger.error(f"✗ File missing: {file_path}")
                results[f"file_{file_path}"] = False
        
        return results
    
    def check_data_availability(self) -> Dict[str, bool]:
        """Check data availability."""
        logger.info("Checking data availability...")
        
        results = {}
        
        # Check raw data
        raw_data_dir = self.project_root / "data" / "raw"
        datasets = ['cnn_dailymail', 'xsum']
        
        for dataset in datasets:
            dataset_dir = raw_data_dir / dataset
            if dataset_dir.exists():
                json_files = list(dataset_dir.glob("*.json"))
                if json_files:
                    logger.info(f"✓ Raw data available: {dataset} ({len(json_files)} files)")
                    results[f"raw_{dataset}"] = True
                else:
                    logger.warning(f"⚠️  Raw data directory exists but no JSON files: {dataset}")
                    results[f"raw_{dataset}"] = False
            else:
                logger.warning(f"⚠️  Raw data not available: {dataset}")
                results[f"raw_{dataset}"] = False
        
        # Check processed data
        processed_data_dir = self.project_root / "data" / "processed"
        tasks = ['entailment_inference', 'summary_ranking', 'consistency_rating']
        
        for dataset in datasets:
            dataset_dir = processed_data_dir / dataset
            if dataset_dir.exists():
                task_files = [dataset_dir / f"{task}.json" for task in tasks]
                available_tasks = [f for f in task_files if f.exists()]
                
                if len(available_tasks) == len(tasks):
                    logger.info(f"✓ Processed data complete: {dataset} ({len(available_tasks)}/{len(tasks)} tasks)")
                    results[f"processed_{dataset}"] = True
                else:
                    logger.warning(f"⚠️  Processed data incomplete: {dataset} ({len(available_tasks)}/{len(tasks)} tasks)")
                    results[f"processed_{dataset}"] = False
            else:
                logger.warning(f"⚠️  Processed data not available: {dataset}")
                results[f"processed_{dataset}"] = False
        
        return results
    
    def check_configuration_files(self) -> Dict[str, bool]:
        """Check configuration files."""
        logger.info("Checking configuration files...")
        
        results = {}
        
        # Check main config
        config_file = self.project_root / "config" / "default.yaml"
        if config_file.exists():
            try:
                import yaml
                with open(config_file, 'r') as f:
                    config = yaml.safe_load(f)
                
                required_sections = ['datasets', 'tasks', 'llm_clients', 'evaluation', 'experiments']
                missing_sections = [s for s in required_sections if s not in config]
                
                if not missing_sections:
                    logger.info("✓ Configuration file valid")
                    results['config_valid'] = True
                else:
                    logger.error(f"✗ Configuration missing sections: {missing_sections}")
                    results['config_valid'] = False
            except Exception as e:
                logger.error(f"✗ Configuration file error: {e}")
                results['config_valid'] = False
        else:
            logger.error("✗ Configuration file missing")
            results['config_valid'] = False
        
        # Check environment file
        env_file = self.project_root / ".env"
        env_template = self.project_root / ".env.template"
        
        if env_file.exists():
            logger.info("✓ Environment file exists")
            results['env_file'] = True
        elif env_template.exists():
            logger.warning("⚠️  Environment template exists but .env file missing")
            results['env_file'] = False
        else:
            logger.error("✗ No environment file or template")
            results['env_file'] = False
        
        return results
    
    def check_imports(self) -> Dict[str, bool]:
        """Check that all project modules can be imported."""
        logger.info("Checking project imports...")
        
        modules_to_test = [
            'src.utils',
            'src.data',
            'src.tasks',
            'src.evaluation',
            'src.llm_clients',
            'src.prompts',
            'src.baselines'
        ]
        
        results = {}
        
        for module in modules_to_test:
            try:
                __import__(module)
                logger.info(f"✓ Import successful: {module}")
                results[f"import_{module}"] = True
            except ImportError as e:
                logger.error(f"✗ Import failed: {module} - {e}")
                results[f"import_{module}"] = False
        
        return results
    
    def check_gpu_availability(self) -> Dict[str, bool]:
        """Check GPU availability for PyTorch."""
        logger.info("Checking GPU availability...")
        
        results = {}
        
        try:
            import torch
            
            # CUDA availability
            cuda_available = torch.cuda.is_available()
            results['cuda_available'] = cuda_available
            
            if cuda_available:
                gpu_count = torch.cuda.device_count()
                gpu_name = torch.cuda.get_device_name(0)
                logger.info(f"✓ CUDA available: {gpu_count} GPU(s) - {gpu_name}")
                results['gpu_count'] = gpu_count
            else:
                logger.info("ℹ️  CUDA not available - will use CPU")
                results['gpu_count'] = 0
            
            # MPS availability (Apple Silicon)
            if hasattr(torch.backends, 'mps') and torch.backends.mps.is_available():
                logger.info("✓ MPS (Apple Silicon) available")
                results['mps_available'] = True
            else:
                results['mps_available'] = False
        
        except ImportError:
            logger.error("✗ PyTorch not available for GPU check")
            results['cuda_available'] = False
            results['mps_available'] = False
            results['gpu_count'] = 0
        
        return results
    
    def run_comprehensive_check(self) -> Dict[str, Dict[str, bool]]:
        """Run comprehensive environment check."""
        logger.info("🔍 Running comprehensive environment check...")
        
        checks = [
            ("Python Environment", self.check_python_environment),
            ("Dependencies", self.check_dependencies),
            ("API Configuration", self.check_api_configuration),
            ("Project Structure", self.check_project_structure),
            ("Data Availability", self.check_data_availability),
            ("Configuration Files", self.check_configuration_files),
            ("Project Imports", self.check_imports),
            ("GPU Availability", self.check_gpu_availability)
        ]
        
        all_results = {}
        overall_success = True
        
        for check_name, check_func in checks:
            logger.info(f"\n--- {check_name} ---")
            try:
                results = check_func()
                all_results[check_name] = results
                
                # Check if any critical checks failed
                if check_name in ["Python Environment", "Dependencies", "API Configuration"]:
                    if not all(results.values()):
                        overall_success = False
                        
            except Exception as e:
                logger.error(f"✗ {check_name} check failed: {e}")
                all_results[check_name] = {"error": False}
                overall_success = False
        
        self.check_results = all_results
        return all_results
    
    def print_summary(self):
        """Print comprehensive summary."""
        logger.info("\n" + "="*80)
        logger.info("ENVIRONMENT CHECK SUMMARY")
        logger.info("="*80)
        
        total_checks = 0
        passed_checks = 0
        failed_checks = 0
        warnings = 0
        
        for category, checks in self.check_results.items():
            logger.info(f"\n{category}:")
            
            category_passed = 0
            category_failed = 0
            
            for check_name, result in checks.items():
                total_checks += 1
                
                if result is True:
                    logger.info(f"  ✓ {check_name}")
                    passed_checks += 1
                    category_passed += 1
                elif result is False:
                    logger.info(f"  ✗ {check_name}")
                    failed_checks += 1
                    category_failed += 1
                else:
                    logger.info(f"  ⚠️  {check_name}")
                    warnings += 1
            
            # Category summary
            if category_failed == 0:
                logger.info(f"  → {category}: ALL PASSED ({category_passed}/{category_passed + category_failed})")
            else:
                logger.info(f"  → {category}: {category_failed} FAILED ({category_passed}/{category_passed + category_failed})")
        
        logger.info(f"\n" + "="*80)
        logger.info(f"OVERALL SUMMARY")
        logger.info(f"="*80)
        logger.info(f"Total checks: {total_checks}")
        logger.info(f"Passed: {passed_checks}")
        logger.info(f"Failed: {failed_checks}")
        logger.info(f"Warnings: {warnings}")
        
        if failed_checks == 0:
            logger.info("🎉 All critical checks passed! Environment is ready.")
        else:
            logger.error(f"❌ {failed_checks} checks failed. Please fix issues before running experiments.")
        
        logger.info("="*80)
    
    def suggest_fixes(self):
        """Suggest fixes for common issues."""
        logger.info("\n" + "="*60)
        logger.info("SUGGESTED FIXES")
        logger.info("="*60)
        
        fixes = []
        
        # Check for common issues
        if 'Dependencies' in self.check_results:
            failed_deps = [k for k, v in self.check_results['Dependencies'].items() if not v]
            if failed_deps:
                fixes.append(f"Install missing dependencies: pip install {' '.join(failed_deps)}")
        
        if 'API Configuration' in self.check_results:
            if not self.check_results['API Configuration'].get('openai_key', True):
                fixes.append("Add OpenAI API key to .env file: OPENAI_API_KEY=sk-your-key-here")
        
        if 'Project Structure' in self.check_results:
            if not self.check_results['Project Structure'].get('file_.env.template', True):
                fixes.append("Create .env file from template: cp .env.template .env")
        
        if 'Data Availability' in self.check_results:
            no_data = all(not v for k, v in self.check_results['Data Availability'].items())
            if no_data:
                fixes.append("Download data: python scripts/setup_data.py")
        
        if fixes:
            for i, fix in enumerate(fixes, 1):
                logger.info(f"{i}. {fix}")
        else:
            logger.info("No specific fixes needed!")
        
        logger.info("\nGeneral setup commands:")
        logger.info("1. python scripts/setup_environment.py")
        logger.info("2. python scripts/setup_data.py")
        logger.info("3. python scripts/quick_test.py")
        
        logger.info("="*60)

def main():
    """Main entry point."""
    parser = argparse.ArgumentParser(description="Check environment for factuality evaluation")
    parser.add_argument("--fix-issues", action="store_true", help="Attempt to fix common issues")
    parser.add_argument("--detailed", action="store_true", help="Show detailed output")
    
    args = parser.parse_args()
    
    # Set logging level
    if args.detailed:
        logging.getLogger().setLevel(logging.DEBUG)
    
    checker = EnvironmentChecker()
    
    # Run comprehensive check
    results = checker.run_comprehensive_check()
    
    # Print summary
    checker.print_summary()
    
    # Suggest fixes
    checker.suggest_fixes()
    
    # Determine exit code
    failed_checks = sum(sum(1 for v in checks.values() if v is False) for checks in results.values())
    sys.exit(0 if failed_checks == 0 else 1)

if __name__ == "__main__":
    main()
