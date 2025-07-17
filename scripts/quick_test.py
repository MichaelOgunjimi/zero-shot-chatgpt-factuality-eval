#!/usr/bin/env python3
"""
Quick Test Script for ChatGPT Factuality Evaluation
==================================================

This script runs quick tests to validate that the environment is set up
correctly and all components are working properly.

Usage:
    python scripts/quick_test.py
    python scripts/quick_test.py --minimal
    python scripts/quick_test.py --full-validation

Author: Michael Ogunjimi
Institution: University of Manchester
Course: MSc AI
"""

import argparse
import logging
import sys
import time
from pathlib import Path
from typing import Dict, List, Optional

# Add project root to path
sys.path.insert(0, str(Path(__file__).parent.parent))

# Setup logging
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)

class QuickTest:
    """Quick test suite for factuality evaluation setup."""
    
    def __init__(self, project_root: Path = None):
        """Initialize quick test."""
        self.project_root = project_root or Path(__file__).parent.parent
        self.results = {}
        
    def test_imports(self) -> bool:
        """Test that all required modules can be imported."""
        logger.info("Testing imports...")
        
        try:
            # Core imports
            from src.utils import load_config, get_config
            from src.data import get_available_datasets, quick_load_dataset
            from src.tasks import get_supported_tasks, create_task
            from src.evaluation import EvaluatorFactory
            from src.llm_clients import OpenAIClient
            
            logger.info("✓ All core modules imported successfully")
            return True
            
        except Exception as e:
            logger.error(f"✗ Import test failed: {e}")
            return False
    
    def test_configuration(self) -> bool:
        """Test configuration loading."""
        logger.info("Testing configuration...")
        
        try:
            from src.utils import load_config
            
            # Load default configuration
            config = load_config("config/default.yaml")
            logger.info("✓ Configuration loaded successfully")
            
            # Check key configuration sections
            required_sections = ['datasets', 'tasks', 'llm_clients', 'evaluation']
            for section in required_sections:
                if section not in config:
                    logger.error(f"✗ Missing configuration section: {section}")
                    return False
                logger.info(f"✓ Configuration section found: {section}")
            
            return True
            
        except Exception as e:
            logger.error(f"✗ Configuration test failed: {e}")
            return False
    
    def test_data_loading(self) -> bool:
        """Test data loading functionality."""
        logger.info("Testing data loading...")
        
        try:
            from src.data import get_available_datasets, quick_load_dataset
            
            # Get available datasets
            datasets = get_available_datasets()
            logger.info(f"✓ Found {len(datasets)} datasets: {', '.join(datasets)}")
            
            # Test loading each dataset
            for dataset in datasets:
                try:
                    examples = quick_load_dataset(dataset, max_examples=2, use_cache=False)
                    if examples:
                        logger.info(f"✓ {dataset}: Loaded {len(examples)} examples")
                    else:
                        logger.warning(f"⚠️  {dataset}: No examples loaded")
                except Exception as e:
                    logger.error(f"✗ {dataset}: Loading failed - {e}")
                    return False
            
            return True
            
        except Exception as e:
            logger.error(f"✗ Data loading test failed: {e}")
            return False
    
    def test_tasks(self) -> bool:
        """Test task creation and functionality."""
        logger.info("Testing tasks...")
        
        try:
            from src.tasks import get_supported_tasks, create_task
            from src.data import quick_load_dataset
            
            # Get supported tasks
            tasks = get_supported_tasks()
            logger.info(f"✓ Found {len(tasks)} tasks: {', '.join(tasks)}")
            
            # Test each task
            for task_name in tasks:
                try:
                    task = create_task(task_name)
                    logger.info(f"✓ {task_name}: Task created successfully")
                    
                    # Test with a sample example
                    examples = quick_load_dataset('cnn_dailymail', max_examples=1, use_cache=False)
                    if examples:
                        example = examples[0]
                        processed = task.preprocess_example(example)
                        logger.info(f"✓ {task_name}: Example preprocessing successful")
                    
                except Exception as e:
                    logger.error(f"✗ {task_name}: Task test failed - {e}")
                    return False
            
            return True
            
        except Exception as e:
            logger.error(f"✗ Task test failed: {e}")
            return False
    
    def test_llm_client(self) -> bool:
        """Test LLM client functionality."""
        logger.info("Testing LLM client...")
        
        try:
            from src.llm_clients import OpenAIClient
            from src.utils import load_config
            
            # Load configuration
            config = load_config("config/default.yaml")
            
            # Create client
            client = OpenAIClient(config)
            logger.info("✓ OpenAI client created successfully")
            
            # Test simple call (if API key is available)
            import os
            if os.getenv('OPENAI_API_KEY') and os.getenv('OPENAI_API_KEY') != 'your-openai-api-key-here':
                try:
                    response = client.generate_response("Say 'test' if you can hear me.", max_tokens=10)
                    if response:
                        logger.info("✓ OpenAI API call successful")
                    else:
                        logger.warning("⚠️  OpenAI API call returned empty response")
                except Exception as e:
                    logger.warning(f"⚠️  OpenAI API call failed: {e}")
            else:
                logger.info("ℹ️  OpenAI API key not configured - skipping API test")
            
            return True
            
        except Exception as e:
            logger.error(f"✗ LLM client test failed: {e}")
            return False
    
    def test_evaluation(self) -> bool:
        """Test evaluation functionality."""
        logger.info("Testing evaluation...")
        
        try:
            from src.evaluation import EvaluatorFactory
            from src.utils import load_config
            
            # Load configuration
            config = load_config("config/default.yaml")
            
            # Test evaluator creation
            evaluator = EvaluatorFactory.create_evaluator('consistency_rating', config)
            logger.info("✓ Evaluator created successfully")
            
            # Test with sample data
            sample_predictions = ['CONSISTENT', 'INCONSISTENT', 'CONSISTENT']
            sample_targets = ['CONSISTENT', 'CONSISTENT', 'INCONSISTENT']
            
            metrics = evaluator.evaluate_predictions(sample_predictions, sample_targets)
            logger.info(f"✓ Evaluation completed: {len(metrics)} metrics computed")
            
            return True
            
        except Exception as e:
            logger.error(f"✗ Evaluation test failed: {e}")
            return False
    
    def test_full_pipeline(self) -> bool:
        """Test a complete mini pipeline."""
        logger.info("Testing full pipeline...")
        
        try:
            from src.utils import load_config
            from src.data import quick_load_dataset
            from src.tasks import create_task
            from src.evaluation import EvaluatorFactory
            
            # Load configuration
            config = load_config("config/default.yaml")
            
            # Load data
            examples = quick_load_dataset('cnn_dailymail', max_examples=2, use_cache=False)
            if not examples:
                logger.error("✗ No examples loaded for pipeline test")
                return False
            
            # Create task
            task = create_task('consistency_rating')
            
            # Process examples
            processed_examples = []
            for example in examples:
                processed = task.preprocess_example(example)
                processed_examples.append(processed)
            
            logger.info(f"✓ Processed {len(processed_examples)} examples")
            
            # Create evaluator
            evaluator = EvaluatorFactory.create_evaluator('consistency_rating', config)
            
            # Simulate predictions
            predictions = ['CONSISTENT'] * len(processed_examples)
            targets = ['CONSISTENT'] * len(processed_examples)
            
            # Evaluate
            metrics = evaluator.evaluate_predictions(predictions, targets)
            logger.info(f"✓ Pipeline test completed with {len(metrics)} metrics")
            
            return True
            
        except Exception as e:
            logger.error(f"✗ Full pipeline test failed: {e}")
            return False
    
    def test_environment_variables(self) -> bool:
        """Test environment variable setup."""
        logger.info("Testing environment variables...")
        
        import os
        
        # Check required environment variables
        required_vars = ['OPENAI_API_KEY']
        optional_vars = ['CACHE_DIR', 'BATCH_SIZE', 'DEVICE']
        
        success = True
        
        for var in required_vars:
            value = os.getenv(var)
            if value and value != f'your-{var.lower().replace("_", "-")}-here':
                logger.info(f"✓ {var} is configured")
            else:
                logger.warning(f"⚠️  {var} is not configured")
                success = False
        
        for var in optional_vars:
            value = os.getenv(var)
            if value:
                logger.info(f"✓ {var} is configured: {value}")
            else:
                logger.info(f"ℹ️  {var} is not configured (optional)")
        
        return success
    
    def run_minimal_test(self) -> bool:
        """Run minimal test suite."""
        logger.info("🚀 Running minimal test suite...")
        
        tests = [
            ("Imports", self.test_imports),
            ("Configuration", self.test_configuration),
            ("Environment Variables", self.test_environment_variables)
        ]
        
        success = True
        for test_name, test_func in tests:
            try:
                result = test_func()
                self.results[test_name] = result
                if not result:
                    success = False
            except Exception as e:
                logger.error(f"✗ {test_name} test crashed: {e}")
                self.results[test_name] = False
                success = False
        
        return success
    
    def run_full_test(self) -> bool:
        """Run complete test suite."""
        logger.info("🚀 Running full test suite...")
        
        tests = [
            ("Imports", self.test_imports),
            ("Configuration", self.test_configuration),
            ("Environment Variables", self.test_environment_variables),
            ("Data Loading", self.test_data_loading),
            ("Tasks", self.test_tasks),
            ("LLM Client", self.test_llm_client),
            ("Evaluation", self.test_evaluation),
            ("Full Pipeline", self.test_full_pipeline)
        ]
        
        success = True
        for test_name, test_func in tests:
            try:
                result = test_func()
                self.results[test_name] = result
                if not result:
                    success = False
            except Exception as e:
                logger.error(f"✗ {test_name} test crashed: {e}")
                self.results[test_name] = False
                success = False
        
        return success
    
    def print_results(self):
        """Print test results summary."""
        logger.info("\n" + "="*50)
        logger.info("TEST RESULTS SUMMARY")
        logger.info("="*50)
        
        passed = 0
        failed = 0
        
        for test_name, result in self.results.items():
            if result:
                logger.info(f"✓ {test_name}: PASSED")
                passed += 1
            else:
                logger.error(f"✗ {test_name}: FAILED")
                failed += 1
        
        logger.info("="*50)
        logger.info(f"Total: {passed + failed} tests")
        logger.info(f"Passed: {passed}")
        logger.info(f"Failed: {failed}")
        
        if failed == 0:
            logger.info("🎉 All tests passed!")
            logger.info("Your environment is ready for experiments!")
        else:
            logger.error(f"❌ {failed} tests failed")
            logger.error("Please fix the issues above before running experiments")
        
        logger.info("="*50)
    
    def run_quick_experiment(self) -> bool:
        """Run a very quick experiment to test everything end-to-end."""
        logger.info("🚀 Running quick experiment test...")
        
        try:
            from experiments.run_chatgpt_evaluation import ChatGPTEvaluationExperiment
            
            # Create experiment
            experiment = ChatGPTEvaluationExperiment(
                config_path="config/default.yaml",
                experiment_name="quick_test_experiment"
            )
            
            # Run very small experiment
            import asyncio
            results = asyncio.run(experiment.run_full_evaluation(
                tasks=['consistency_rating'],
                datasets=['cnn_dailymail'],
                sample_size=2,
                prompt_type='zero_shot'
            ))
            
            if results:
                logger.info("✓ Quick experiment completed successfully")
                
                # Clean up test experiment
                import shutil
                test_dir = Path("results/experiments/quick_test_experiment")
                if test_dir.exists():
                    shutil.rmtree(test_dir)
                
                return True
            else:
                logger.error("✗ Quick experiment failed")
                return False
                
        except Exception as e:
            logger.error(f"✗ Quick experiment failed: {e}")
            return False

def main():
    """Main entry point."""
    parser = argparse.ArgumentParser(description="Run quick tests for factuality evaluation")
    parser.add_argument("--minimal", action="store_true", help="Run minimal test suite")
    parser.add_argument("--full-validation", action="store_true", help="Run full validation including experiment")
    
    args = parser.parse_args()
    
    tester = QuickTest()
    
    start_time = time.time()
    
    if args.minimal:
        success = tester.run_minimal_test()
    elif args.full_validation:
        success = tester.run_full_test()
        if success:
            logger.info("Running quick experiment test...")
            experiment_success = tester.run_quick_experiment()
            tester.results["Quick Experiment"] = experiment_success
            success = success and experiment_success
    else:
        success = tester.run_full_test()
    
    execution_time = time.time() - start_time
    
    tester.print_results()
    
    logger.info(f"\n⏱️  Total execution time: {execution_time:.2f} seconds")
    
    if success:
        logger.info("\n🎉 Ready to run experiments!")
        logger.info("Next steps:")
        logger.info("  1. Run cost estimation: python scripts/estimate_costs.py")
        logger.info("  2. Run experiments: python experiments/run_all_experiments.py --quick-test")
    else:
        logger.error("\n❌ Environment not ready. Please fix the issues above.")
    
    sys.exit(0 if success else 1)

if __name__ == "__main__":
    main()
