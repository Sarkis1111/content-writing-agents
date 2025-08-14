"""Base configuration management for the agentic system."""

import os
from typing import Any, Dict, Optional
from pydantic import Field
from pydantic_settings import BaseSettings


class BaseConfig(BaseSettings):
    """Base configuration class with common settings."""
    
    environment: str = Field(default="development", env="ENVIRONMENT")
    debug: bool = Field(default=False, env="DEBUG")
    log_level: str = Field(default="INFO", env="LOG_LEVEL")
    
    # API Keys
    openai_api_key: Optional[str] = Field(default=None, env="OPENAI_API_KEY")
    anthropic_api_key: Optional[str] = Field(default=None, env="ANTHROPIC_API_KEY")
    serp_api_key: Optional[str] = Field(default=None, env="SERP_API_KEY")
    
    # Database and Storage
    redis_url: str = Field(default="redis://localhost:6379", env="REDIS_URL")
    
    class Config:
        env_file = ".env"
        env_file_encoding = "utf-8"
        case_sensitive = True


class MCPConfig(BaseConfig):
    """MCP Server specific configuration."""
    
    mcp_server_host: str = Field(default="localhost", env="MCP_SERVER_HOST")
    mcp_server_port: int = Field(default=8000, env="MCP_SERVER_PORT")
    mcp_max_connections: int = Field(default=100, env="MCP_MAX_CONNECTIONS")


class FrameworkConfig(BaseConfig):
    """Configuration for agentic frameworks."""
    
    # CrewAI Configuration
    crewai_memory: bool = Field(default=True, env="CREWAI_MEMORY")
    crewai_verbose: bool = Field(default=True, env="CREWAI_VERBOSE")
    crewai_embedder_model: str = Field(default="text-embedding-3-small", env="CREWAI_EMBEDDER_MODEL")
    
    # LangGraph Configuration
    langgraph_checkpointer: str = Field(default="memory", env="LANGGRAPH_CHECKPOINTER")
    langgraph_interrupt_before: list = Field(default=["human_review"], env="LANGGRAPH_INTERRUPT_BEFORE")
    langgraph_interrupt_after: list = Field(default=["quality_gate"], env="LANGGRAPH_INTERRUPT_AFTER")
    
    # AutoGen Configuration
    autogen_cache_seed: int = Field(default=42, env="AUTOGEN_CACHE_SEED")
    autogen_temperature: float = Field(default=0.7, env="AUTOGEN_TEMPERATURE")
    autogen_max_tokens: int = Field(default=2000, env="AUTOGEN_MAX_TOKENS")
    
    @property
    def crewai_config(self) -> Dict[str, Any]:
        """Get CrewAI configuration dictionary."""
        return {
            "memory": self.crewai_memory,
            "verbose": self.crewai_verbose,
            "embedder": {
                "provider": "openai",
                "config": {"model": self.crewai_embedder_model}
            }
        }
    
    @property
    def langgraph_config(self) -> Dict[str, Any]:
        """Get LangGraph configuration dictionary."""
        return {
            "checkpointer": self.langgraph_checkpointer,
            "interrupt_before": self.langgraph_interrupt_before,
            "interrupt_after": self.langgraph_interrupt_after
        }
    
    @property
    def autogen_config(self) -> Dict[str, Any]:
        """Get AutoGen configuration dictionary."""
        return {
            "cache_seed": self.autogen_cache_seed,
            "temperature": self.autogen_temperature,
            "max_tokens": self.autogen_max_tokens
        }


class ToolConfig(BaseConfig):
    """Configuration for tools and external APIs."""
    
    # Rate limiting
    rate_limit_per_minute: int = Field(default=60, env="RATE_LIMIT_PER_MINUTE")
    rate_limit_burst: int = Field(default=10, env="RATE_LIMIT_BURST")
    
    # Retry configuration
    max_retries: int = Field(default=3, env="MAX_RETRIES")
    retry_delay: float = Field(default=1.0, env="RETRY_DELAY")
    retry_backoff: float = Field(default=2.0, env="RETRY_BACKOFF")
    
    # Content processing
    max_content_length: int = Field(default=50000, env="MAX_CONTENT_LENGTH")
    request_timeout: int = Field(default=30, env="REQUEST_TIMEOUT")


def get_config() -> Dict[str, Any]:
    """Get all configuration objects."""
    return {
        "base": BaseConfig(),
        "mcp": MCPConfig(),
        "frameworks": FrameworkConfig(),
        "tools": ToolConfig()
    }