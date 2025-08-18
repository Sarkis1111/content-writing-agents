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


def get_settings() -> Dict[str, Any]:
    """Get system settings with environment variable overrides."""
    import os
    from dotenv import load_dotenv
    
    # Load .env file if it exists
    load_dotenv()
    
    # Basic settings with environment variable overrides
    settings = {
        'openai_api_key': os.getenv('OPENAI_API_KEY'),
        'serpapi_key': os.getenv('SERPAPI_KEY'),
        'google_api_key': os.getenv('GOOGLE_API_KEY'),
        'google_cse_id': os.getenv('GOOGLE_CSE_ID'),
        'news_api_key': os.getenv('NEWS_API_KEY'),
        'reddit_client_id': os.getenv('REDDIT_CLIENT_ID'),
        'reddit_client_secret': os.getenv('REDDIT_CLIENT_SECRET'),
        'reddit_user_agent': os.getenv('REDDIT_USER_AGENT', 'ContentWritingAgents/1.0'),
        
        'environment': os.getenv('ENVIRONMENT', 'development'),
        'log_level': os.getenv('LOG_LEVEL', 'INFO'),
        'cache_ttl': int(os.getenv('CACHE_TTL', '3600')),
        'request_timeout': int(os.getenv('REQUEST_TIMEOUT', '30')),
        'max_concurrent_requests': int(os.getenv('MAX_CONCURRENT_REQUESTS', '10')),
        
        'default_content_model': os.getenv('DEFAULT_CONTENT_MODEL', 'gpt-4'),
        'default_content_temperature': float(os.getenv('DEFAULT_CONTENT_TEMPERATURE', '0.7')),
        'max_content_tokens': int(os.getenv('MAX_CONTENT_TOKENS', '4000')),
        
        'default_image_size': os.getenv('DEFAULT_IMAGE_SIZE', '1024x1024'),
        'default_image_style': os.getenv('DEFAULT_IMAGE_STYLE', 'natural'),
        'default_image_quality': os.getenv('DEFAULT_IMAGE_QUALITY', 'standard'),
    }
    
    return settings