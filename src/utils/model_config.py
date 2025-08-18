"""
Model Configuration Manager

Dynamically loads model-specific configurations based on model name and tier.
Removes hardcoded model dependencies and supports flexible model switching.
"""

import os
import yaml
from typing import Dict, Any, Optional
from pathlib import Path

# Model tier mappings based on OpenAI documentation
MODEL_TIER_LIMITS = {
    "gpt-4.1-mini": {
        "tier1": {"rpm": 500, "tpm": 200000, "batch_limit": 2000000},
        "tier2": {"rpm": 5000, "tpm": 2000000, "batch_limit": 20000000}
    },
    "gpt-4o-mini": {
        "tier1": {"rpm": 1000, "tpm": 100000, "batch_limit": 1000000},
        "tier2": {"rpm": 2000, "tpm": 2000000, "batch_limit": 2000000}
    },
    "o1-mini": {
        "tier1": {"rpm": 1000, "tpm": 100000, "batch_limit": 1000000},
        "tier2": {"rpm": 2000, "tpm": 2000000, "batch_limit": 2000000}
    },
    "gpt-4o": {
        "tier1": {"rpm": 500, "tpm": 30000, "batch_limit": 90000},
        "tier2": {"rpm": 5000, "tpm": 450000, "batch_limit": 1350000},
        "tier3": {"rpm": 5000, "tpm": 800000, "batch_limit": 50000000},
        "tier4": {"rpm": 10000, "tpm": 2000000, "batch_limit": 200000000},
        "tier5": {"rpm": 10000, "tpm": 30000000, "batch_limit": 5000000000}
    }
}

# Model pricing (cost per 1k tokens)
MODEL_PRICING = {
    "gpt-4.1-mini": {"input": 0.0004, "output": 0.0016},
    "gpt-4o-mini": {"input": 0.00015, "output": 0.0006},
    "o1-mini": {"input": 0.003, "output": 0.012},
    "gpt-4o": {"input": 0.0025, "output": 0.01}
}

DEFAULT_FALLBACKS = {
    "gpt-4.1-mini": ["gpt-4o-mini", "o1-mini"],
    "gpt-4o-mini": ["gpt-4.1-mini", "o1-mini"],
    "o1-mini": ["gpt-4.1-mini", "gpt-4o-mini"],
    "gpt-4o": ["gpt-4.1-mini", "gpt-4o-mini"]
}


class ModelConfigManager:
    """Manages model-specific configurations and rate limits."""
    
    def __init__(self, config_dir: str = "./config"):
        self.config_dir = Path(config_dir)
        self.models_dir = self.config_dir / "models"
        
    def get_model_config(self, model: str, tier: str = "tier2") -> Dict[str, Any]:
        """
        Get configuration for a specific model and tier.
        
        Args:
            model: Model name (e.g., "gpt-4.1-mini", "gpt-4o-mini")
            tier: API tier (e.g., "tier1", "tier2")
            
        Returns:
            Model-specific configuration dictionary
        """
        # Try to load from file first
        config_file = self.models_dir / f"{model}_{tier}.yaml"
        if config_file.exists():
            with open(config_file, 'r') as f:
                return yaml.safe_load(f)
        
        # Generate configuration dynamically if file doesn't exist
        return self._generate_model_config(model, tier)
    
    def _generate_model_config(self, model: str, tier: str) -> Dict[str, Any]:
        """Generate model configuration dynamically based on tier limits."""
        if model not in MODEL_TIER_LIMITS:
            raise ValueError(f"Unsupported model: {model}")
        
        tier_limits = MODEL_TIER_LIMITS[model].get(tier)
        if not tier_limits:
            # Fall back to highest available tier for the model
            available_tiers = list(MODEL_TIER_LIMITS[model].keys())
            tier = available_tiers[-1]  # Use highest tier
            tier_limits = MODEL_TIER_LIMITS[model][tier]
        
        # Apply 90% safety margin
        safe_rpm = int(tier_limits["rpm"] * 0.9)
        safe_tpm = int(tier_limits["tpm"] * 0.9)
        safe_batch = int(tier_limits["batch_limit"] * 0.9)
        
        # Estimate daily requests (conservative: 10% of theoretical maximum)
        daily_requests = int(safe_rpm * 60 * 24 * 0.1)
        
        # Determine concurrent requests based on tier
        tier_num = int(tier.replace("tier", ""))
        concurrent_requests = min(10 + (tier_num - 1) * 5, 30)
        
        config = {
            "openai": {
                "models": {
                    "primary": model,
                    "fallbacks": DEFAULT_FALLBACKS.get(model, ["gpt-4.1-mini"])
                },
                "rate_limits": {
                    "requests_per_minute": safe_rpm,
                    "tokens_per_minute": safe_tpm,
                    "requests_per_day": daily_requests,
                    "concurrent_requests": concurrent_requests,
                    "batch_queue_limit": safe_batch
                },
                "generation": {
                    "temperature": 0.0,
                    "max_tokens": 2048,
                    "top_p": 1.0,
                    "frequency_penalty": 0.0,
                    "presence_penalty": 0.0,
                    "stop_sequences": []
                },
                "cost_control": {
                    "cost_per_1k_tokens": {
                        model: MODEL_PRICING.get(model, MODEL_PRICING["gpt-4.1-mini"])
                    }
                },
                "batch": {
                    "enabled": True,
                    "max_queue_size": safe_batch,
                    "processing_timeout": 86400,
                    "cost_savings": 0.5
                }
            }
        }
        
        return config
    
    def list_supported_models(self) -> list:
        """List all supported models."""
        return list(MODEL_TIER_LIMITS.keys())
    
    def list_available_tiers(self, model: str) -> list:
        """List available tiers for a specific model."""
        if model not in MODEL_TIER_LIMITS:
            return []
        return list(MODEL_TIER_LIMITS[model].keys())
    
    def get_rate_limits(self, model: str, tier: str = "tier2") -> Dict[str, int]:
        """Get rate limits for a specific model and tier."""
        config = self.get_model_config(model, tier)
        return config["openai"]["rate_limits"]
    
    def get_pricing(self, model: str) -> Dict[str, float]:
        """Get pricing information for a model."""
        return MODEL_PRICING.get(model, MODEL_PRICING["gpt-4.1-mini"])
    
    def save_model_config(self, model: str, tier: str, config: Dict[str, Any]):
        """Save a model configuration to file."""
        self.models_dir.mkdir(parents=True, exist_ok=True)
        config_file = self.models_dir / f"{model}_{tier}.yaml"
        
        with open(config_file, 'w') as f:
            yaml.dump(config, f, default_flow_style=False, sort_keys=False)


def get_model_config(model: str, tier: str = "tier2", config_dir: str = "./config") -> Dict[str, Any]:
    """
    Convenience function to get model configuration.
    
    Args:
        model: Model name
        tier: API tier
        config_dir: Configuration directory path
        
    Returns:
        Model configuration dictionary
    """
    manager = ModelConfigManager(config_dir)
    return manager.get_model_config(model, tier)


def merge_model_config(base_config: Dict[str, Any], model: str, tier: str = "tier2") -> Dict[str, Any]:
    """
    Merge model-specific configuration with base configuration.
    
    Args:
        base_config: Base configuration dictionary
        model: Model name
        tier: API tier
        
    Returns:
        Merged configuration dictionary
    """
    model_config = get_model_config(model, tier)
    
    # Deep merge the configurations
    merged = base_config.copy()
    
    if "openai" in model_config:
        if "openai" not in merged:
            merged["openai"] = {}
        
        for key, value in model_config["openai"].items():
            if isinstance(value, dict) and key in merged["openai"]:
                merged["openai"][key].update(value)
            else:
                merged["openai"][key] = value
    
    return merged
