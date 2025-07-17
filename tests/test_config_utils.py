"""
Configuration and Utilities Tests
=================================

Tests for configuration management, logging utilities, and other
utility functions used throughout the factuality evaluation system.

Author: Michael Ogunjimi
Institution: University of Manchester
"""

import pytest
import tempfile
import os
import logging
from pathlib import Path
from unittest.mock import Mock, patch, MagicMock


class Config:
    """Simple configuration class for testing"""
    
    def __init__(self, config_dict=None):
        self.config = config_dict or {}
    
    def get(self, key, default=None):
        """Get configuration value"""
        keys = key.split('.')
        value = self.config
        
        try:
            for k in keys:
                value = value[k]
            return value
        except (KeyError, TypeError):
            return default
    
    def set(self, key, value):
        """Set configuration value"""
        keys = key.split('.')
        target = self.config
        
        for k in keys[:-1]:
            if k not in target:
                target[k] = {}
            target = target[k]
        
        target[keys[-1]] = value
    
    def update(self, other_config):
        """Update configuration with another config"""
        if isinstance(other_config, dict):
            self._deep_update(self.config, other_config)
        elif isinstance(other_config, Config):
            self._deep_update(self.config, other_config.config)
    
    def _deep_update(self, target, source):
        """Deep update dictionary"""
        for key, value in source.items():
            if key in target and isinstance(target[key], dict) and isinstance(value, dict):
                self._deep_update(target[key], value)
            else:
                target[key] = value
    
    def to_dict(self):
        """Convert to dictionary"""
        return self.config.copy()


def get_config(config_path=None):
    """Get configuration object"""
    default_config = {
        "openai": {
            "models": {
                "primary": "gpt-4.1-mini",
                "fallbacks": ["o1-mini", "gpt-4o"]
            },
            "api": {
                "timeout": 30,
                "max_retries": 3
            },
            "rate_limits": {
                "requests_per_minute": 60,
                "tokens_per_minute": 150000
            }
        },
        "data": {
            "datasets": ["cnn_dailymail", "xsum"],
            "cache_enabled": True,
            "validation_enabled": True
        },
        "evaluation": {
            "batch_size": 32,
            "max_examples": None,
            "save_results": True
        },
        "logging": {
            "level": "INFO",
            "format": "%(asctime)s - %(name)s - %(levelname)s - %(message)s",
            "file_enabled": True
        }
    }
    
    config = Config(default_config)
    
    # Load from file if specified
    if config_path and Path(config_path).exists():
        import yaml
        with open(config_path, 'r') as f:
            file_config = yaml.safe_load(f)
        config.update(file_config)
    
    return config


class Logger:
    """Simple logger for testing"""
    
    def __init__(self, name="test_logger", level=logging.INFO):
        self.name = name
        self.level = level
        self.logs = []
    
    def info(self, msg, *args, **kwargs):
        self.logs.append(("INFO", msg % args if args else msg))
    
    def warning(self, msg, *args, **kwargs):
        self.logs.append(("WARNING", msg % args if args else msg))
    
    def error(self, msg, *args, **kwargs):
        self.logs.append(("ERROR", msg % args if args else msg))
    
    def debug(self, msg, *args, **kwargs):
        self.logs.append(("DEBUG", msg % args if args else msg))
    
    def get_logs(self):
        return self.logs
    
    def clear_logs(self):
        self.logs = []


def get_logger(name=__name__):
    """Get logger instance"""
    return Logger(name)


class CostTracker:
    """Cost tracking utility for testing"""
    
    def __init__(self, daily_budget=50.0, total_budget=200.0, warning_threshold=0.8):
        self.daily_budget = daily_budget
        self.total_budget = total_budget
        self.warning_threshold = warning_threshold
        self.total_spent = 0.0
        self.daily_spent = 0.0
        self.cost_history = []
        self.costs_by_model = {}
        self.costs_by_task = {}
    
    def add_cost(self, cost, model="gpt-4.1-mini", experiment_name=None, task_name=None, metadata=None):
        """Add cost entry"""
        self.total_spent += cost
        self.daily_spent += cost
        
        # Track by model
        if model not in self.costs_by_model:
            self.costs_by_model[model] = 0.0
        self.costs_by_model[model] += cost
        
        # Track by task
        if task_name:
            if task_name not in self.costs_by_task:
                self.costs_by_task[task_name] = 0.0
            self.costs_by_task[task_name] += cost
        
        # Add to history
        entry = {
            "cost": cost,
            "model": model,
            "experiment_name": experiment_name,
            "task_name": task_name,
            "metadata": metadata,
            "running_total": self.total_spent,
            "daily_total": self.daily_spent
        }
        self.cost_history.append(entry)
        
        # Check budgets
        self._check_budgets()
    
    def _check_budgets(self):
        """Check budget limits"""
        daily_usage = self.daily_spent / self.daily_budget
        total_usage = self.total_spent / self.total_budget
        
        if daily_usage >= 1.0:
            raise ValueError("Daily budget exceeded")
        elif daily_usage >= self.warning_threshold:
            pass  # Would normally log warning
        
        if total_usage >= 1.0:
            raise ValueError("Total budget exceeded")
    
    def can_afford(self, estimated_cost):
        """Check if we can afford an estimated cost"""
        return (self.daily_spent + estimated_cost <= self.daily_budget and
                self.total_spent + estimated_cost <= self.total_budget)
    
    def get_analysis(self):
        """Get cost analysis"""
        return {
            "total_spent": self.total_spent,
            "daily_spent": self.daily_spent,
            "daily_budget": self.daily_budget,
            "total_budget": self.total_budget,
            "daily_usage_percent": (self.daily_spent / self.daily_budget) * 100,
            "total_usage_percent": (self.total_spent / self.total_budget) * 100,
            "costs_by_model": self.costs_by_model.copy(),
            "costs_by_task": self.costs_by_task.copy(),
            "total_transactions": len(self.cost_history)
        }
    
    def reset_daily(self):
        """Reset daily spending"""
        self.daily_spent = 0.0


def validate_config(config):
    """Validate configuration"""
    errors = []
    warnings = []
    
    # Check required sections
    required_sections = ["openai", "data", "evaluation"]
    for section in required_sections:
        if not config.get(section):
            errors.append(f"Missing required section: {section}")
    
    # Check OpenAI config
    if config.get("openai"):
        if not config.get("openai.models.primary"):
            errors.append("Missing primary OpenAI model")
        
        timeout = config.get("openai.api.timeout")
        if timeout and timeout < 10:
            warnings.append("API timeout may be too low")
    
    # Check data config
    datasets = config.get("data.datasets", [])
    if not datasets:
        warnings.append("No datasets configured")
    
    return len(errors) == 0, errors, warnings


class TestConfig:
    """Test configuration functionality"""
    
    def test_config_creation(self):
        """Test creating configuration"""
        config_dict = {"key": "value", "nested": {"inner": "data"}}
        config = Config(config_dict)
        
        assert config.get("key") == "value"
        assert config.get("nested.inner") == "data"
    
    def test_config_get_default(self):
        """Test getting config value with default"""
        config = Config({"existing": "value"})
        
        assert config.get("existing") == "value"
        assert config.get("missing", "default") == "default"
        assert config.get("missing") is None
    
    def test_config_set(self):
        """Test setting config values"""
        config = Config()
        
        config.set("simple", "value")
        config.set("nested.deep.key", "deep_value")
        
        assert config.get("simple") == "value"
        assert config.get("nested.deep.key") == "deep_value"
    
    def test_config_update_dict(self):
        """Test updating config with dictionary"""
        config = Config({"existing": "old"})
        
        update_dict = {
            "existing": "new",
            "additional": "value",
            "nested": {"key": "nested_value"}
        }
        
        config.update(update_dict)
        
        assert config.get("existing") == "new"
        assert config.get("additional") == "value"
        assert config.get("nested.key") == "nested_value"
    
    def test_config_update_config(self):
        """Test updating config with another config"""
        config1 = Config({"key1": "value1"})
        config2 = Config({"key2": "value2", "key1": "updated"})
        
        config1.update(config2)
        
        assert config1.get("key1") == "updated"
        assert config1.get("key2") == "value2"
    
    def test_config_deep_update(self):
        """Test deep update of nested dictionaries"""
        config = Config({
            "section": {
                "keep": "original",
                "update": "old"
            }
        })
        
        config.update({
            "section": {
                "update": "new",
                "add": "additional"
            }
        })
        
        assert config.get("section.keep") == "original"
        assert config.get("section.update") == "new"
        assert config.get("section.add") == "additional"
    
    def test_config_to_dict(self):
        """Test converting config to dictionary"""
        original_dict = {"key": "value", "nested": {"inner": "data"}}
        config = Config(original_dict)
        
        result_dict = config.to_dict()
        
        assert result_dict == original_dict
        assert result_dict is not config.config  # Should be a copy


class TestGetConfig:
    """Test get_config function"""
    
    def test_get_default_config(self):
        """Test getting default configuration"""
        config = get_config()
        
        assert config.get("openai.models.primary") == "gpt-4.1-mini"
        assert config.get("data.cache_enabled") is True
        assert "cnn_dailymail" in config.get("data.datasets")
        assert "xsum" in config.get("data.datasets")
    
    def test_get_config_sections(self):
        """Test that all expected sections are present"""
        config = get_config()
        
        required_sections = ["openai", "data", "evaluation", "logging"]
        for section in required_sections:
            assert config.get(section) is not None
    
    def test_get_config_openai_settings(self):
        """Test OpenAI configuration settings"""
        config = get_config()
        
        assert config.get("openai.models.primary") is not None
        assert isinstance(config.get("openai.models.fallbacks"), list)
        assert config.get("openai.api.timeout") > 0
        assert config.get("openai.api.max_retries") > 0
        assert config.get("openai.rate_limits.requests_per_minute") > 0
    
    @pytest.fixture
    def temp_config_file(self):
        """Create temporary config file"""
        config_content = """
openai:
  models:
    primary: "gpt-4.1-mini"
    fallbacks: ["o1-mini", "gpt-4o"]
  api:
    timeout: 45
data:
  datasets: ["cnn_dailymail"]
  custom_setting: true
"""
        with tempfile.NamedTemporaryFile(mode='w', suffix='.yaml', delete=False) as f:
            f.write(config_content)
            temp_path = f.name
        
        yield temp_path
        
        # Cleanup
        os.unlink(temp_path)
    
    @patch('yaml.safe_load')
    @patch('builtins.open', create=True)
    def test_get_config_from_file(self, mock_open, mock_yaml):
        """Test loading config from file"""
        # Mock file content
        mock_yaml.return_value = {
            "openai": {"models": {"primary": "gpt-4.1-mini"}},
            "custom": {"setting": "value"}
        }
        mock_open.return_value.__enter__.return_value = Mock()
        
        # Create a real file for Path.exists() check
        with tempfile.NamedTemporaryFile(mode='w', delete=False) as f:
            config_path = f.name
        
        try:
            config = get_config(config_path)
            
            # Should have default + file settings
            assert config.get("custom.setting") == "value"
            # Primary model should be overridden
            assert config.get("openai.models.primary") == "gpt-4.1-mini"
            # Other defaults should remain
            assert config.get("data.cache_enabled") is True
            
        finally:
            os.unlink(config_path)


class TestLogger:
    """Test logger functionality"""
    
    def test_logger_creation(self):
        """Test creating logger"""
        logger = Logger("test", logging.INFO)
        
        assert logger.name == "test"
        assert logger.level == logging.INFO
        assert len(logger.logs) == 0
    
    def test_logger_info(self):
        """Test info logging"""
        logger = Logger()
        
        logger.info("Test message")
        logger.info("Message with %s", "argument")
        
        logs = logger.get_logs()
        assert len(logs) == 2
        assert logs[0] == ("INFO", "Test message")
        assert logs[1] == ("INFO", "Message with argument")
    
    def test_logger_warning(self):
        """Test warning logging"""
        logger = Logger()
        
        logger.warning("Warning message")
        
        logs = logger.get_logs()
        assert len(logs) == 1
        assert logs[0] == ("WARNING", "Warning message")
    
    def test_logger_error(self):
        """Test error logging"""
        logger = Logger()
        
        logger.error("Error message")
        
        logs = logger.get_logs()
        assert len(logs) == 1
        assert logs[0] == ("ERROR", "Error message")
    
    def test_logger_debug(self):
        """Test debug logging"""
        logger = Logger()
        
        logger.debug("Debug message")
        
        logs = logger.get_logs()
        assert len(logs) == 1
        assert logs[0] == ("DEBUG", "Debug message")
    
    def test_logger_clear_logs(self):
        """Test clearing logs"""
        logger = Logger()
        
        logger.info("Message 1")
        logger.info("Message 2")
        assert len(logger.get_logs()) == 2
        
        logger.clear_logs()
        assert len(logger.get_logs()) == 0


class TestGetLogger:
    """Test get_logger function"""
    
    def test_get_logger_default(self):
        """Test getting logger with default name"""
        logger = get_logger()
        
        assert isinstance(logger, Logger)
        assert logger.name.endswith("test_config_utils")  # Should use module name
    
    def test_get_logger_custom_name(self):
        """Test getting logger with custom name"""
        logger = get_logger("custom_logger")
        
        assert isinstance(logger, Logger)
        assert logger.name == "custom_logger"


class TestCostTracker:
    """Test cost tracking functionality"""
    
    def test_cost_tracker_initialization(self):
        """Test cost tracker initialization"""
        tracker = CostTracker(daily_budget=100.0, total_budget=500.0)
        
        assert tracker.daily_budget == 100.0
        assert tracker.total_budget == 500.0
        assert tracker.total_spent == 0.0
        assert tracker.daily_spent == 0.0
        assert len(tracker.cost_history) == 0
    
    def test_add_cost_basic(self):
        """Test adding basic cost"""
        tracker = CostTracker()
        
        tracker.add_cost(10.0, model="gpt-4.1-mini", task_name="test_task")
        
        assert tracker.total_spent == 10.0
        assert tracker.daily_spent == 10.0
        assert tracker.costs_by_model["gpt-4.1-mini"] == 10.0
        assert tracker.costs_by_task["test_task"] == 10.0
        assert len(tracker.cost_history) == 1
    
    def test_add_multiple_costs(self):
        """Test adding multiple costs"""
        tracker = CostTracker()
        
        tracker.add_cost(5.0, model="gpt-4.1-mini", task_name="task1")
        tracker.add_cost(3.0, model="o1-mini", task_name="task2")
        tracker.add_cost(2.0, model="gpt-4.1-mini", task_name="task1")
        
        assert tracker.total_spent == 10.0
        assert tracker.costs_by_model["gpt-4.1-mini"] == 7.0
        assert tracker.costs_by_model["o1-mini"] == 3.0
        assert tracker.costs_by_task["task1"] == 7.0
        assert tracker.costs_by_task["task2"] == 3.0
    
    def test_cost_history(self):
        """Test cost history tracking"""
        tracker = CostTracker()
        
        tracker.add_cost(5.0, model="gpt-4.1-mini", experiment_name="exp1", task_name="task1")
        
        history = tracker.cost_history
        assert len(history) == 1
        
        entry = history[0]
        assert entry["cost"] == 5.0
        assert entry["model"] == "gpt-4.1-mini"
        assert entry["experiment_name"] == "exp1"
        assert entry["task_name"] == "task1"
        assert entry["running_total"] == 5.0
        assert entry["daily_total"] == 5.0
    
    def test_can_afford(self):
        """Test budget checking"""
        tracker = CostTracker(daily_budget=20.0, total_budget=100.0)
        
        # Initially should be able to afford
        assert tracker.can_afford(10.0) is True
        
        # Add some cost
        tracker.add_cost(15.0)
        
        # Should still afford small amount
        assert tracker.can_afford(4.0) is True
        
        # Should not afford amount that exceeds daily budget
        assert tracker.can_afford(10.0) is False
    
    def test_budget_warning_threshold(self):
        """Test budget warning threshold"""
        tracker = CostTracker(daily_budget=10.0, warning_threshold=0.8)
        
        # Should not raise exception below warning threshold
        tracker.add_cost(7.0)  # 70% of budget
        
        # Should not raise exception at warning threshold
        tracker.add_cost(1.0)  # 80% of budget
    
    def test_daily_budget_exceeded(self):
        """Test daily budget exceeded"""
        tracker = CostTracker(daily_budget=10.0, total_budget=100.0)
        
        with pytest.raises(ValueError, match="Daily budget exceeded"):
            tracker.add_cost(15.0)
    
    def test_total_budget_exceeded(self):
        """Test total budget exceeded"""
        tracker = CostTracker(daily_budget=50.0, total_budget=20.0)
        
        with pytest.raises(ValueError, match="Total budget exceeded"):
            tracker.add_cost(25.0)
    
    def test_get_analysis(self):
        """Test getting cost analysis"""
        tracker = CostTracker(daily_budget=50.0, total_budget=200.0)
        
        tracker.add_cost(10.0, model="gpt-4.1-mini", task_name="task1")
        tracker.add_cost(5.0, model="o1-mini", task_name="task2")
        
        analysis = tracker.get_analysis()
        
        assert analysis["total_spent"] == 15.0
        assert analysis["daily_spent"] == 15.0
        assert analysis["daily_usage_percent"] == 30.0
        assert analysis["total_usage_percent"] == 7.5
        assert analysis["costs_by_model"]["gpt-4.1-mini"] == 10.0
        assert analysis["costs_by_task"]["task1"] == 10.0
        assert analysis["total_transactions"] == 2
    
    def test_reset_daily(self):
        """Test resetting daily spending"""
        tracker = CostTracker()
        
        tracker.add_cost(10.0)
        assert tracker.daily_spent == 10.0
        assert tracker.total_spent == 10.0
        
        tracker.reset_daily()
        assert tracker.daily_spent == 0.0
        assert tracker.total_spent == 10.0  # Total should remain


class TestValidateConfig:
    """Test configuration validation"""
    
    def test_validate_valid_config(self):
        """Test validating valid configuration"""
        config = get_config()
        
        is_valid, errors, warnings = validate_config(config)
        
        assert is_valid is True
        assert len(errors) == 0
    
    def test_validate_missing_sections(self):
        """Test validating config with missing sections"""
        config = Config({"logging": {"level": "INFO"}})
        
        is_valid, errors, warnings = validate_config(config)
        
        assert is_valid is False
        assert len(errors) >= 2  # Missing openai, data, evaluation
        assert any("openai" in error for error in errors)
        assert any("data" in error for error in errors)
    
    def test_validate_missing_primary_model(self):
        """Test validating config with missing primary model"""
        config = Config({
            "openai": {"api": {"timeout": 30}},
            "data": {"datasets": ["cnn_dailymail"]},
            "evaluation": {"batch_size": 32}
        })
        
        is_valid, errors, warnings = validate_config(config)
        
        assert is_valid is False
        assert any("primary" in error and "model" in error for error in errors)
    
    def test_validate_warnings(self):
        """Test configuration validation warnings"""
        config = Config({
            "openai": {
                "models": {"primary": "gpt-4.1-mini"},
                "api": {"timeout": 5}  # Low timeout should trigger warning
            },
            "data": {"datasets": []},  # Empty datasets should trigger warning
            "evaluation": {"batch_size": 32}
        })
        
        is_valid, errors, warnings = validate_config(config)
        
        assert is_valid is True  # No errors, just warnings
        assert len(warnings) >= 1
        assert any("timeout" in warning or "datasets" in warning for warning in warnings)


class TestUtilityIntegration:
    """Test integration of utility components"""
    
    def test_config_and_logger_integration(self):
        """Test using config with logger"""
        config = get_config()
        logger = get_logger("integration_test")
        
        # Use config values
        log_level = config.get("logging.level", "INFO")
        log_format = config.get("logging.format")
        
        # Log with configuration
        logger.info("Logging with level: %s", log_level)
        logger.info("Format: %s", log_format)
        
        logs = logger.get_logs()
        assert len(logs) == 2
        assert "INFO" in logs[0][1]
        assert "Format:" in logs[1][1]
    
    def test_config_and_cost_tracker_integration(self):
        """Test using config with cost tracker"""
        config = get_config()
        
        # Use config for cost tracker settings
        daily_budget = config.get("evaluation.daily_budget", 50.0)
        total_budget = config.get("evaluation.total_budget", 200.0)
        
        tracker = CostTracker(daily_budget=daily_budget, total_budget=total_budget)
        
        # Add some costs
        primary_model = config.get("openai.models.primary")
        tracker.add_cost(10.0, model=primary_model, task_name="test_integration")
        
        analysis = tracker.get_analysis()
        assert analysis["total_spent"] == 10.0
        assert primary_model in analysis["costs_by_model"]
    
    def test_full_utility_workflow(self):
        """Test complete utility workflow"""
        # 1. Load configuration
        config = get_config()
        
        # 2. Validate configuration
        is_valid, errors, warnings = validate_config(config)
        assert is_valid is True
        
        # 3. Setup logger
        logger = get_logger("workflow_test")
        logger.info("Starting workflow with config validation")
        
        # 4. Setup cost tracking
        tracker = CostTracker(
            daily_budget=config.get("evaluation.daily_budget", 50.0),
            total_budget=config.get("evaluation.total_budget", 200.0)
        )
        
        # 5. Simulate some operations
        for i in range(3):
            model = config.get("openai.models.primary")
            cost = 2.0 + i
            tracker.add_cost(cost, model=model, task_name=f"task_{i}")
            logger.info("Added cost: %.2f for task_%d", cost, i)
        
        # 6. Get final analysis
        analysis = tracker.get_analysis()
        logger.info("Total spent: %.2f", analysis["total_spent"])
        
        # Verify workflow completed successfully
        assert len(logger.get_logs()) == 5  # 1 start + 3 cost logs + 1 final
        assert analysis["total_spent"] == 9.0  # 2.0 + 3.0 + 4.0
        assert analysis["total_transactions"] == 3
