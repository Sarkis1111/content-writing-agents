"""Configuration loader with environment-specific overrides."""

import os
import yaml
from typing import Any, Dict, Optional
from pathlib import Path

from .base import BaseConfig, MCPConfig, FrameworkConfig, ToolConfig


class ConfigLoader:
    """Loads configuration from YAML files with environment overrides."""
    
    def __init__(self, config_dir: Optional[str] = None):
        self.config_dir = Path(config_dir or "config")
        self.environment = os.getenv("ENVIRONMENT", "development")
    
    def load_yaml(self, filename: str) -> Dict[str, Any]:
        """Load a YAML configuration file."""
        filepath = self.config_dir / filename
        if not filepath.exists():
            return {}
        
        with open(filepath, 'r') as f:
            return yaml.safe_load(f) or {}
    
    def load_config(self) -> Dict[str, Any]:
        """Load base configuration with environment overrides."""
        # Load base settings
        config = self.load_yaml("settings.yaml")
        
        # Load environment-specific overrides
        env_config = self.load_yaml(f"{self.environment}.yaml")
        
        # Merge configurations (environment overrides base)
        return self._deep_merge(config, env_config)
    
    def _deep_merge(self, base: Dict[str, Any], override: Dict[str, Any]) -> Dict[str, Any]:
        """Deep merge two dictionaries."""
        result = base.copy()
        
        for key, value in override.items():
            if key in result and isinstance(result[key], dict) and isinstance(value, dict):
                result[key] = self._deep_merge(result[key], value)
            else:
                result[key] = value
        
        return result
    
    def get_framework_configs(self) -> Dict[str, Any]:
        """Get framework-specific configurations."""
        config = self.load_config()
        return config.get("frameworks", {})
    
    def get_tool_config(self) -> Dict[str, Any]:
        """Get tool configuration."""
        config = self.load_config()
        return config.get("tools", {})
    
    def get_monitoring_config(self) -> Dict[str, Any]:
        """Get monitoring configuration."""
        config = self.load_config()
        return config.get("monitoring", {})


def load_config(config_dir: Optional[str] = None) -> Dict[str, Any]:
    """Load complete system configuration."""
    loader = ConfigLoader(config_dir)
    return loader.load_config()