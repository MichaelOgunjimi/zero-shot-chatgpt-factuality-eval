#!/usr/bin/env python3
"""
Environment Setup Script for ChatGPT Factuality Evaluation
=========================================================

This script sets up the complete environment for running factuality evaluation
experiments, including dependency installation, environment configuration,
and validation.

Usage:
    python scripts/setup_environment.py
    python scripts/setup_environment.py --check-only
    python scripts/setup_environment.py --install-deps

Author: Michael Ogunjimi
Institution: University of Manchester
Course: MSc AI
"""

import argparse
import logging
import os
import subprocess
import sys
from pathlib import Path
from typing import Dict, List, Tuple

# Setup logging
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)

class EnvironmentSetup:
    """Complete environment setup for factuality evaluation."""
    
    def __init__(self, project_root: Path = None):
        """Initialize setup with project root."""
        self.project_root = project_root or Path(__file__).parent.parent
        self.requirements_file = self.project_root / "requirements.txt"
        self.env_template = self.project_root / ".env.template"
        self.env_file = self.project_root / ".env"
        
    def check_python_version(self) -> bool:
        """Check if Python version is compatible."""
        logger.info("Checking Python version...")
        
        version_info = sys.version_info
        if version_info.major < 3 or (version_info.major == 3 and version_info.minor < 8):
            logger.error(f"Python 3.8+ required, found {version_info.major}.{version_info.minor}")
            return False
        
        logger.info(f"✓ Python {version_info.major}.{version_info.minor}.{version_info.micro} is compatible")
        return True
    
    def check_dependencies(self) -> Dict[str, bool]:
        """Check if all required dependencies are installed."""
        logger.info("Checking dependencies...")
        
        # Core dependencies
        required_packages = {
            'torch': 'PyTorch',
            'transformers': 'Transformers',
            'datasets': 'HuggingFace Datasets',
            'openai': 'OpenAI API',
            'pandas': 'Pandas',
            'numpy': 'NumPy',
            'tqdm': 'Progress bars',
            'pyyaml': 'YAML support',
            'plotly': 'Visualization',
            'scipy': 'Scientific computing',
            'scikit-learn': 'Machine learning'
        }
        
        results = {}
        for package, description in required_packages.items():
            try:
                __import__(package)
                logger.info(f"✓ {description} ({package}) is installed")
                results[package] = True
            except ImportError:
                logger.error(f"✗ {description} ({package}) is NOT installed")
                results[package] = False
        
        return results
    
    def install_dependencies(self) -> bool:
        """Install dependencies from requirements.txt."""
        logger.info("Installing dependencies...")
        
        if not self.requirements_file.exists():
            logger.error(f"Requirements file not found: {self.requirements_file}")
            return False
        
        try:
            cmd = [sys.executable, "-m", "pip", "install", "-r", str(self.requirements_file)]
            result = subprocess.run(cmd, capture_output=True, text=True)
            
            if result.returncode == 0:
                logger.info("✓ Dependencies installed successfully")
                return True
            else:
                logger.error(f"✗ Failed to install dependencies: {result.stderr}")
                return False
                
        except Exception as e:
            logger.error(f"✗ Error installing dependencies: {e}")
            return False
    
    def setup_environment_file(self) -> bool:
        """Set up .env file from template."""
        logger.info("Setting up environment file...")
        
        if not self.env_template.exists():
            logger.error(f"Environment template not found: {self.env_template}")
            return False
        
        if self.env_file.exists():
            logger.info("✓ .env file already exists")
            return True
        
        try:
            # Copy template to .env
            import shutil
            shutil.copy(self.env_template, self.env_file)
            
            logger.info("✓ Created .env file from template")
            logger.warning("⚠️  Please edit .env file to add your OpenAI API key")
            
            return True
            
        except Exception as e:
            logger.error(f"✗ Error creating .env file: {e}")
            return False
    
    def check_api_keys(self) -> Dict[str, bool]:
        """Check if API keys are configured."""
        logger.info("Checking API key configuration...")
        
        results = {}
        
        # Check OpenAI API key
        openai_key = os.getenv('OPENAI_API_KEY')
        if openai_key and openai_key != 'your-openai-api-key-here':
            logger.info("✓ OpenAI API key is configured")
            results['openai'] = True
        else:
            logger.warning("⚠️  OpenAI API key not configured")
            results['openai'] = False
        
        return results
    
    def create_directories(self) -> bool:
        """Create necessary directories."""
        logger.info("Creating project directories...")
        
        directories = [
            'data/raw',
            'data/processed',
            'data/cache',
            'results/experiments',
            'results/figures',
            'results/tables',
            'results/logs',
            'logs'
        ]
        
        try:
            for directory in directories:
                dir_path = self.project_root / directory
                dir_path.mkdir(parents=True, exist_ok=True)
                logger.info(f"✓ Created/verified directory: {directory}")
            
            return True
            
        except Exception as e:
            logger.error(f"✗ Error creating directories: {e}")
            return False
    
    def validate_project_structure(self) -> bool:
        """Validate project structure."""
        logger.info("Validating project structure...")
        
        required_paths = [
            'src/__init__.py',
            'src/utils/__init__.py',
            'src/data/__init__.py',
            'src/tasks/__init__.py',
            'src/evaluation/__init__.py',
            'experiments/__init__.py',
            'config/default.yaml',
            'requirements.txt'
        ]
        
        all_valid = True
        for path in required_paths:
            full_path = self.project_root / path
            if full_path.exists():
                logger.info(f"✓ Found: {path}")
            else:
                logger.error(f"✗ Missing: {path}")
                all_valid = False
        
        return all_valid
    
    def run_basic_import_test(self) -> bool:
        """Test basic imports to ensure everything works."""
        logger.info("Testing basic imports...")
        
        # Add project root to path
        sys.path.insert(0, str(self.project_root))
        
        try:
            # Test core imports
            from src.utils import load_config
            from src.data import get_available_datasets
            from src.tasks import get_supported_tasks
            
            logger.info("✓ Core imports successful")
            
            # Test configuration loading
            config = load_config("config/default.yaml")
            logger.info("✓ Configuration loading successful")
            
            # Test dataset discovery
            datasets = get_available_datasets()
            logger.info(f"✓ Found {len(datasets)} datasets: {', '.join(datasets)}")
            
            # Test task discovery
            tasks = get_supported_tasks()
            logger.info(f"✓ Found {len(tasks)} tasks: {', '.join(tasks)}")
            
            return True
            
        except Exception as e:
            logger.error(f"✗ Import test failed: {e}")
            return False
    
    def run_complete_setup(self) -> bool:
        """Run complete environment setup."""
        logger.info("🚀 Starting complete environment setup...")
        
        success = True
        
        # Check Python version
        if not self.check_python_version():
            success = False
        
        # Install dependencies
        if not self.install_dependencies():
            success = False
        
        # Check dependencies
        dep_results = self.check_dependencies()
        if not all(dep_results.values()):
            success = False
        
        # Setup environment file
        if not self.setup_environment_file():
            success = False
        
        # Create directories
        if not self.create_directories():
            success = False
        
        # Validate project structure
        if not self.validate_project_structure():
            success = False
        
        # Run import test
        if not self.run_basic_import_test():
            success = False
        
        # Check API keys
        api_results = self.check_api_keys()
        if not api_results.get('openai', False):
            logger.warning("⚠️  OpenAI API key not configured - experiments will not work")
        
        if success:
            logger.info("🎉 Environment setup completed successfully!")
            self._print_next_steps()
        else:
            logger.error("❌ Environment setup failed - please check errors above")
        
        return success
    
    def run_check_only(self) -> bool:
        """Run environment check without making changes."""
        logger.info("🔍 Running environment check...")
        
        checks = [
            ("Python version", self.check_python_version()),
            ("Dependencies", all(self.check_dependencies().values())),
            ("Project structure", self.validate_project_structure()),
            ("API keys", all(self.check_api_keys().values())),
            ("Basic imports", self.run_basic_import_test())
        ]
        
        all_passed = True
        for check_name, result in checks:
            if result:
                logger.info(f"✓ {check_name}: PASSED")
            else:
                logger.error(f"✗ {check_name}: FAILED")
                all_passed = False
        
        if all_passed:
            logger.info("🎉 All environment checks passed!")
        else:
            logger.error("❌ Some environment checks failed")
        
        return all_passed
    
    def _print_next_steps(self):
        """Print next steps after setup."""
        logger.info("\n" + "="*60)
        logger.info("NEXT STEPS")
        logger.info("="*60)
        logger.info("1. Edit .env file to add your OpenAI API key:")
        logger.info("   nano .env")
        logger.info("   # Add: OPENAI_API_KEY=sk-your-key-here")
        logger.info("")
        logger.info("2. Download datasets:")
        logger.info("   python scripts/setup_data.py")
        logger.info("")
        logger.info("3. Run quick test:")
        logger.info("   python scripts/quick_test.py")
        logger.info("")
        logger.info("4. Run experiments:")
        logger.info("   python experiments/run_all_experiments.py --quick-test")
        logger.info("="*60)

def main():
    """Main entry point."""
    parser = argparse.ArgumentParser(description="Setup environment for factuality evaluation")
    parser.add_argument("--check-only", action="store_true", help="Only check environment without making changes")
    parser.add_argument("--install-deps", action="store_true", help="Only install dependencies")
    
    args = parser.parse_args()
    
    setup = EnvironmentSetup()
    
    if args.check_only:
        success = setup.run_check_only()
    elif args.install_deps:
        success = setup.install_dependencies()
    else:
        success = setup.run_complete_setup()
    
    sys.exit(0 if success else 1)

if __name__ == "__main__":
    main()
