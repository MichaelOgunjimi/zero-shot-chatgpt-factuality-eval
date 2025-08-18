"""
Configuration Management System for ChatGPT Factuality Evaluation
===============================================================

Streamlined configuration management with YAML support, environment
variable substitution, validation, and reproducibility features.

Author: Michael Ogunjimi
Institution: University of Manchester
Course: MSc AI
"""

import logging
import os
import random
import sys
from datetime import datetime
from pathlib import Path
from typing import Dict, Any, Optional

import numpy as np
import yaml
from dotenv import load_dotenv

# Load environment variables from .env file
load_dotenv()

logger = logging.getLogger(__name__)

# Import model configuration manager
try:
    from .model_config import ModelConfigManager, merge_model_config
except ImportError:
    ModelConfigManager = None
    merge_model_config = None
    logger.warning("Model configuration manager not available")

try:
    import torch
except ImportError:
    torch = None
    logger.warning("PyTorch not available")


class ConfigWrapper:
    """
    Configuration wrapper providing dot notation access and additional utilities.

    Wraps configuration dictionaries to provide consistent access patterns
    and academic research features like experiment tracking.
    """

    def __init__(self, config: Dict[str, Any]):
        """
        Initialize configuration wrapper.

        Args:
            config: Configuration dictionary
        """
        self._config = config

    def get(self, key: str, default: Any = None) -> Any:
        """
        Get configuration value with dot notation support.

        Args:
            key: Configuration key (supports dot notation like "tasks.entailment_inference.enabled")
            default: Default value if key not found

        Returns:
            Configuration value or default
        """
        if '.' not in key:
            return self._config.get(key, default)

        keys = key.split('.')
        value = self._config

        try:
            for k in keys:
                if isinstance(value, dict) and k in value:
                    value = value[k]
                else:
                    return default
            return value
        except (KeyError, TypeError):
            return default

    def set(self, key: str, value: Any) -> None:
        """
        Set configuration value with dot notation support.

        Args:
            key: Configuration key (supports dot notation)
            value: Value to set
        """
        if '.' not in key:
            self._config[key] = value
            return

        keys = key.split('.')
        config = self._config

        # Navigate to parent dictionary
        for k in keys[:-1]:
            if k not in config:
                config[k] = {}
            config = config[k]

        config[keys[-1]] = value

    def update(self, updates: Dict[str, Any]) -> None:
        """
        Update configuration with new values.

        Args:
            updates: Dictionary of updates (supports dot notation keys)
        """
        for key, value in updates.items():
            self.set(key, value)

    def to_dict(self) -> Dict[str, Any]:
        """Convert to plain dictionary."""
        return dict(self._config)

    def keys(self):
        """Get configuration keys."""
        return self._config.keys()

    def items(self):
        """Get configuration items."""
        return self._config.items()

    def values(self):
        """Get configuration values."""
        return self._config.values()

    def __contains__(self, key: str) -> bool:
        """Check if key exists in configuration."""
        return self.get(key) is not None

    def __getitem__(self, key: str) -> Any:
        """Dictionary-style access."""
        result = self.get(key)
        if result is None:
            raise KeyError(f"Configuration key not found: {key}")
        return result

    def __setitem__(self, key: str, value: Any) -> None:
        """Dictionary-style assignment."""
        self.set(key, value)


class ConfigManager:
    """
    Advanced configuration manager for academic research projects.

    Provides comprehensive configuration management with validation,
    environment support, and reproducibility features.
    """

    def __init__(self, config_path: Optional[str] = None):
        """
        Initialize configuration manager.

        Args:
            config_path: Path to main configuration file
        """
        self.config_path = Path(config_path or "config/default.yaml")
        self.config = None
        self.environment = os.getenv("FACTUALITY_ENV", "development")

        # Setup basic logging for config manager
        logging.basicConfig(level=logging.INFO)
        self.logger = logging.getLogger(__name__)

    def load_config(
        self,
        config_path: Optional[str] = None,
        environment: Optional[str] = None,
        model: Optional[str] = None,
        tier: Optional[str] = None
    ) -> ConfigWrapper:
        """
        Load configuration from YAML file with environment variable substitution and model-specific configuration.

        Args:
            config_path: Path to configuration file
            environment: Environment name (development, testing, production)
            model: Model name (e.g., "gpt-4.1-mini", "gpt-4o-mini")
            tier: API tier (e.g., "tier1", "tier2")

        Returns:
            Loaded configuration wrapped for enhanced access

        Raises:
            FileNotFoundError: If configuration file doesn't exist
            yaml.YAMLError: If YAML parsing fails
        """
        config_path = Path(config_path or self.config_path)
        environment = environment or self.environment

        if not config_path.exists():
            raise FileNotFoundError(f"Configuration file not found: {config_path}")

        try:
            # Load main configuration
            with open(config_path, 'r', encoding='utf-8') as file:
                raw_config = yaml.safe_load(file)

            # Apply environment-specific overrides
            if "environments" in raw_config and environment in raw_config["environments"]:
                env_overrides = raw_config["environments"][environment]
                raw_config = self._merge_configs(raw_config, env_overrides)
                self.logger.info(f"Applied {environment} environment overrides")

            # Apply model-specific configuration if specified
            if model and ModelConfigManager and merge_model_config:
                try:
                    # Use tier2 as default if not specified (since user is on tier 2)
                    tier = tier or "tier2"
                    raw_config = merge_model_config(raw_config, model, tier)
                    self.logger.info(f"Applied {model} configuration for {tier}")
                except Exception as e:
                    self.logger.warning(f"Failed to apply model-specific config: {e}")

            # Substitute environment variables
            raw_config = self._substitute_env_vars(raw_config)

            # Wrap configuration
            self.config = ConfigWrapper(raw_config)

            # Validate configuration
            self._validate_config()

            # Setup environment
            self._setup_environment()

            self.logger.info(f"Configuration loaded from {config_path} (env: {environment})")
            return self.config

        except yaml.YAMLError as e:
            self.logger.error(f"Failed to parse YAML configuration: {e}")
            raise
        except Exception as e:
            self.logger.error(f"Failed to load configuration: {e}")
            raise

    def _merge_configs(self, base: Dict[str, Any], override: Dict[str, Any]) -> Dict[str, Any]:
        """
        Deep merge configuration dictionaries.

        Args:
            base: Base configuration
            override: Override configuration

        Returns:
            Merged configuration
        """
        result = base.copy()

        for key, value in override.items():
            if key in result and isinstance(result[key], dict) and isinstance(value, dict):
                result[key] = self._merge_configs(result[key], value)
            else:
                result[key] = value

        return result

    def _substitute_env_vars(self, config: Any) -> Any:
        """
        Recursively substitute environment variables in configuration.

        Args:
            config: Configuration value (dict, list, or string)

        Returns:
            Configuration with environment variables substituted
        """
        if isinstance(config, dict):
            return {key: self._substitute_env_vars(value) for key, value in config.items()}
        elif isinstance(config, list):
            return [self._substitute_env_vars(item) for item in config]
        elif isinstance(config, str):
            # Replace ${VAR_NAME} patterns with environment variables
            import re
            pattern = r'\$\{([^}]+)\}'

            def replace_env_var(match):
                var_name = match.group(1)
                return os.getenv(var_name, match.group(0))  # Keep original if not found

            return re.sub(pattern, replace_env_var, config)
        else:
            return config

    def _validate_config(self) -> None:
        """
        Validate configuration structure and required fields.

        Raises:
            ValueError: If required sections or fields are missing
        """
        required_sections = [
            "project", "global", "paths", "tasks", "openai",
            "prompts", "datasets", "evaluation"
        ]

        for section in required_sections:
            if section not in self.config:
                raise ValueError(f"Missing required configuration section: {section}")

        # Validate project metadata
        project_config = self.config.get("project", {})
        required_project_fields = ["name", "version", "author"]
        for field in required_project_fields:
            if field not in project_config:
                raise ValueError(f"Missing required project field: {field}")

        # Validate OpenAI configuration
        openai_config = self.config.get("openai", {})
        required_openai_fields = ["models", "api", "rate_limits", "generation"]
        for field in required_openai_fields:
            if field not in openai_config:
                raise ValueError(f"Missing required OpenAI field: {field}")

        # Validate datasets configuration
        datasets_config = self.config.get("datasets", {})
        if not any(datasets_config.get(ds, {}).get("enabled", False) for ds in datasets_config):
            self.logger.warning("No datasets are enabled")

        self.logger.info("Configuration validation completed successfully")

    def _setup_environment(self) -> None:
        """Setup environment variables."""
        if "project" in self.config:
            os.environ["PROJECT_NAME"] = self.config.get("project.name", "chatgpt_factuality_eval")

        if "global" in self.config:
            os.environ["LOG_LEVEL"] = self.config.get("global.log_level", "INFO")
            os.environ["TORCH_DEVICE"] = self._determine_device()

        self.logger.info("Environment setup completed")

    def _determine_device(self) -> str:
        """
        Determine the appropriate device for computation.

        Returns:
            Device string (cuda, mps, or cpu)
        """
        device_config = self.config.get("global.device", "auto")

        if device_config == "auto":
            if torch and torch.cuda.is_available():
                return "cuda"
            elif torch and hasattr(torch.backends, 'mps') and torch.backends.mps.is_available():
                return "mps"
            else:
                return "cpu"
        else:
            return device_config

    def save_config(self, output_path: Optional[str] = None) -> None:
        """
        Save current configuration to file.

        Args:
            output_path: Path to save configuration. If None, saves to original path.
        """
        if self.config is None:
            raise ValueError("No configuration loaded")

        output_path = Path(output_path or self.config_path)
        output_path.parent.mkdir(parents=True, exist_ok=True)

        with open(output_path, 'w', encoding='utf-8') as file:
            yaml.dump(self.config.to_dict(), file, default_flow_style=False, indent=2)

        self.logger.info(f"Configuration saved to {output_path}")

    def create_experiment_snapshot(self, experiment_name: str) -> Dict[str, Any]:
        """
        Create a complete snapshot of configuration for an experiment.

        Args:
            experiment_name: Name of the experiment

        Returns:
            Dictionary containing full configuration snapshot
        """
        if self.config is None:
            raise ValueError("No configuration loaded")

        snapshot = {
            "timestamp": datetime.now().isoformat(),
            "experiment_name": experiment_name,
            "config": self.config.to_dict(),
            "environment": {
                "python_version": ".".join(map(str, sys.version_info[:3])),
                "torch_version": torch.__version__ if torch else "not available",
                "device": os.environ.get("TORCH_DEVICE", "unknown"),
                "working_directory": str(Path.cwd()),
                "environment_name": self.environment
            }
        }

        return snapshot


# Global configuration manager instance
_config_manager = ConfigManager()


def load_config(
    config_path: Optional[str] = None,
    environment: Optional[str] = None
) -> ConfigWrapper:
    """
    Load configuration file with environment support.

    Args:
        config_path: Path to configuration file
        environment: Environment name (development, testing, production)

    Returns:
        Loaded configuration with enhanced access
    """
    return _config_manager.load_config(config_path, environment)


def get_config(model: Optional[str] = None, tier: str = "tier2") -> ConfigWrapper:
    """
    Get current configuration.

    Args:
        model: Optional model name for model-specific configuration
        tier: API tier (defaults to "tier2")

    Returns:
        Current configuration (loads default if not already loaded)
    """
    if model:
        # Load with model-specific configuration
        return get_config_with_model(model=model, tier=tier)
    
    if _config_manager.config is None:
        return load_config()
    return _config_manager.config


def get_device(device_spec: Optional[str] = None) -> str:
    """
    Get appropriate device for computation.

    Args:
        device_spec: Device specification ("auto", "cpu", "cuda", "mps")

    Returns:
        Device string
    """
    if device_spec is None:
        try:
            config = get_config()
            device_spec = config.get("global.device", "auto")
        except:
            device_spec = "auto"

    if device_spec == "auto":
        if torch and torch.cuda.is_available():
            device = "cuda"
            logger.info(f"Using CUDA device: {torch.cuda.get_device_name()}")
        elif torch and hasattr(torch.backends, 'mps') and torch.backends.mps.is_available():
            device = "mps"
            logger.info("Using Metal Performance Shaders (MPS)")
        else:
            device = "cpu"
            logger.info("Using CPU")
    else:
        device = device_spec
        logger.info(f"Using specified device: {device}")

    return device


def set_global_seed(seed: Optional[int] = None) -> int:
    """
    Set global random seed for reproducibility.

    Args:
        seed: Random seed (loads from config if not provided)

    Returns:
        The seed that was set
    """
    if seed is None:
        try:
            config = get_config()
            seed = config.get("global.seed", 42)
        except:
            seed = 42

    random.seed(seed)
    np.random.seed(seed)
    
    if torch:
        torch.manual_seed(seed)
        if torch.cuda.is_available():
            torch.cuda.manual_seed_all(seed)
        torch.backends.cudnn.deterministic = True
        torch.backends.cudnn.benchmark = False

    os.environ["TOKENIZERS_PARALLELISM"] = "false"

    logger.info(f"Global seed set to: {seed}")
    return seed


def create_output_directories(config: Optional[ConfigWrapper] = None) -> Dict[str, Path]:
    """
    Create output directories for experiments.

    Args:
        config: Configuration object

    Returns:
        Dictionary mapping directory names to paths
    """
    if config is None:
        config = get_config()

    # Base directories
    base_dirs = {
        "data": Path(config.get("paths.data_dir", "./data")),
        "cache": Path(config.get("paths.cache_dir", "./cache"))
    }

    for name, path in base_dirs.items():
        path.mkdir(parents=True, exist_ok=True)
        logger.debug(f"Created directory {name}: {path}")

    return base_dirs


def validate_api_keys(config: Optional[ConfigWrapper] = None) -> Dict[str, bool]:
    """
    Validate that required API keys are available.

    Args:
        config: Configuration object

    Returns:
        Dictionary indicating which API keys are available
    """
    if config is None:
        config = get_config()

    api_status = {}

    openai_key = os.getenv("OPENAI_API_KEY")
    api_status["openai"] = bool(openai_key and openai_key.startswith("sk-"))

    # Log status
    available_apis = [api for api, available in api_status.items() if available]
    if available_apis:
        logger.info(f"API keys available: {available_apis}")
    else:
        logger.warning("No API keys detected")

    return api_status


def setup_reproducibility(config: Optional[ConfigWrapper] = None) -> Dict[str, Any]:
    """
    Setup reproducibility settings for experiments.

    Args:
        config: Configuration object

    Returns:
        Reproducibility settings that were applied
    """
    if config is None:
        config = get_config()

    seed = set_global_seed(config.get("global.seed"))

    output_dirs = create_output_directories(config)

    # Validate API keys
    api_status = validate_api_keys(config)

    device = get_device(config.get("global.device"))

    settings = {
        "seed": seed,
        "device": device,
        "output_directories": output_dirs,
        "api_status": api_status,
        "timestamp": datetime.now().isoformat()
    }

    logger.info("Reproducibility setup completed")
    return settings


def get_config_with_model(
    config_path: Optional[str] = None,
    model: Optional[str] = None,
    tier: str = "tier2"
) -> ConfigWrapper:
    """
    Load configuration with model-specific settings.
    
    Args:
        config_path: Path to configuration file
        model: Model name (e.g., "gpt-4.1-mini", "gpt-4o-mini", "o1-mini", "gpt-4o")
        tier: API tier (defaults to "tier2" since user is on tier 2)
        
    Returns:
        Configuration with model-specific rate limits and settings
        
    Examples:
        # Load config with gpt-4.1-mini for tier 2
        config = get_config_with_model(model="gpt-4.1-mini")
        
        # Load config with gpt-4o-mini for tier 2
        config = get_config_with_model(model="gpt-4o-mini")
        
        # Load config with o1-mini for tier 2
        config = get_config_with_model(model="o1-mini")
    """
    manager = ConfigManager(config_path)
    return manager.load_config(model=model, tier=tier)


def get_model_rate_limits(model: str, tier: str = "tier2") -> Dict[str, int]:
    """
    Get rate limits for a specific model and tier.
    
    Args:
        model: Model name
        tier: API tier (defaults to "tier2")
        
    Returns:
        Dictionary with rate limit values
    """
    if ModelConfigManager:
        manager = ModelConfigManager()
        return manager.get_rate_limits(model, tier)
    else:
        logger.warning("Model configuration manager not available")
        return {}