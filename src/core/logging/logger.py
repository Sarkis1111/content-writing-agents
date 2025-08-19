"""Unified logging infrastructure for all agentic frameworks."""

import os
import logging
import logging.handlers
from typing import Dict, Optional
from pathlib import Path
from enum import Enum

from ..config.loader import load_config


class LogLevel(Enum):
    """Log level enumeration."""
    DEBUG = "DEBUG"
    INFO = "INFO"
    WARNING = "WARNING"
    ERROR = "ERROR"
    CRITICAL = "CRITICAL"


class FrameworkLogger:
    """Framework-specific logger wrapper."""
    
    def __init__(self, framework_name: str, logger: logging.Logger):
        self.framework_name = framework_name
        self.logger = logger
    
    def debug(self, message: str, **kwargs):
        """Log debug message with framework context."""
        self.logger.debug(f"[{self.framework_name}] {message}", extra=kwargs)
    
    def info(self, message: str, **kwargs):
        """Log info message with framework context."""
        self.logger.info(f"[{self.framework_name}] {message}", extra=kwargs)
    
    def warning(self, message: str, **kwargs):
        """Log warning message with framework context."""
        self.logger.warning(f"[{self.framework_name}] {message}", extra=kwargs)
    
    def error(self, message: str, **kwargs):
        """Log error message with framework context."""
        self.logger.error(f"[{self.framework_name}] {message}", extra=kwargs)
    
    def critical(self, message: str, **kwargs):
        """Log critical message with framework context."""
        self.logger.critical(f"[{self.framework_name}] {message}", extra=kwargs)
    
    def exception(self, message: str, **kwargs):
        """Log exception with framework context."""
        self.logger.exception(f"[{self.framework_name}] {message}", extra=kwargs)


class LoggingManager:
    """Centralized logging manager for all frameworks."""
    
    def __init__(self, config_dir: Optional[str] = None):
        self.config = load_config(config_dir)
        self.loggers: Dict[str, FrameworkLogger] = {}
        self._setup_logging()
    
    def _setup_logging(self):
        """Initialize logging configuration."""
        logging_config = self.config.get("logging", {})
        
        # Create logs directory if it doesn't exist
        if logging_config.get("file", {}).get("enabled", False):
            log_path = Path(logging_config["file"]["path"])
            log_path.parent.mkdir(parents=True, exist_ok=True)
        
        # Configure root logger
        root_logger = logging.getLogger()
        root_logger.setLevel(getattr(logging, self.config.get("log_level", "INFO")))
        
        # Clear existing handlers
        root_logger.handlers.clear()
        
        # Set up formatters
        formatter = logging.Formatter(
            logging_config.get("format", "%(asctime)s - %(name)s - %(levelname)s - %(message)s")
        )
        
        # Console handler
        console_config = logging_config.get("console", {})
        if console_config.get("enabled", True):
            console_handler = logging.StreamHandler()
            console_handler.setFormatter(formatter)
            console_handler.setLevel(getattr(logging, console_config.get("level", "INFO")))
            root_logger.addHandler(console_handler)
        
        # File handler
        file_config = logging_config.get("file", {})
        if file_config.get("enabled", False):
            file_handler = logging.handlers.RotatingFileHandler(
                filename=file_config["path"],
                maxBytes=self._parse_size(file_config.get("max_size", "10MB")),
                backupCount=file_config.get("backup_count", 5)
            )
            file_handler.setFormatter(formatter)
            file_handler.setLevel(getattr(logging, file_config.get("level", "INFO")))
            root_logger.addHandler(file_handler)
    
    def _parse_size(self, size_str: str) -> int:
        """Parse size string like '10MB' to bytes."""
        size_str = size_str.upper()
        if size_str.endswith("KB"):
            return int(size_str[:-2]) * 1024
        elif size_str.endswith("MB"):
            return int(size_str[:-2]) * 1024 * 1024
        elif size_str.endswith("GB"):
            return int(size_str[:-2]) * 1024 * 1024 * 1024
        else:
            return int(size_str)
    
    def get_framework_logger(self, framework_name: str) -> FrameworkLogger:
        """Get a framework-specific logger."""
        if framework_name not in self.loggers:
            # Get framework-specific log level
            framework_config = self.config.get("logging", {}).get("framework_specific", {})
            framework_settings = framework_config.get(framework_name.lower(), {})
            
            # Handle both string level and dict with level key
            if isinstance(framework_settings, dict):
                framework_level = framework_settings.get("level", "INFO")
            else:
                framework_level = framework_settings or "INFO"
            
            # Create logger
            logger = logging.getLogger(f"frameworks.{framework_name}")
            logger.setLevel(getattr(logging, framework_level))
            
            # Create framework logger wrapper
            self.loggers[framework_name] = FrameworkLogger(framework_name, logger)
        
        return self.loggers[framework_name]
    
    def get_component_logger(self, component_name: str) -> logging.Logger:
        """Get a component-specific logger."""
        return logging.getLogger(f"components.{component_name}")
    
    def get_tool_logger(self, tool_name: str) -> logging.Logger:
        """Get a tool-specific logger."""
        return logging.getLogger(f"tools.{tool_name}")
    
    def get_agent_logger(self, agent_name: str) -> logging.Logger:
        """Get an agent-specific logger."""
        return logging.getLogger(f"agents.{agent_name}")


# Global logging manager instance
_logging_manager: Optional[LoggingManager] = None


def get_logging_manager(config_dir: Optional[str] = None) -> LoggingManager:
    """Get the global logging manager instance."""
    global _logging_manager
    if _logging_manager is None:
        _logging_manager = LoggingManager(config_dir)
    return _logging_manager


def get_framework_logger(framework_name: str) -> FrameworkLogger:
    """Get a framework-specific logger."""
    return get_logging_manager().get_framework_logger(framework_name)


def get_component_logger(component_name: str) -> logging.Logger:
    """Get a component-specific logger."""
    return get_logging_manager().get_component_logger(component_name)


def get_tool_logger(tool_name: str) -> logging.Logger:
    """Get a tool-specific logger."""
    return get_logging_manager().get_tool_logger(tool_name)


def get_agent_logger(agent_name: str) -> logging.Logger:
    """Get an agent-specific logger."""
    return get_logging_manager().get_agent_logger(agent_name)


def get_logger(name: str) -> logging.Logger:
    """Get a logger by name."""
    return logging.getLogger(name)


def setup_logging(config_dir: Optional[str] = None) -> LoggingManager:
    """Set up logging configuration."""
    manager = get_logging_manager(config_dir)
    return manager