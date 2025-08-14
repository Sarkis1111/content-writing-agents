"""Error handling framework module."""

from .exceptions import (
    AgenticSystemError,
    FrameworkError,
    CrewAIError,
    LangGraphError,
    AutoGenError,
    AgentError,
    ToolError,
    ConfigurationError,
    APIError,
    RateLimitError,
    AuthenticationError,
    ValidationError,
    WorkflowError,
    ResourceError,
    TimeoutError,
    ConcurrencyError
)
from .handlers import (
    ErrorHandler,
    get_error_handler,
    handle_errors,
    handle_async_errors,
    ErrorContext
)

__all__ = [
    # Exceptions
    "AgenticSystemError",
    "FrameworkError", 
    "CrewAIError",
    "LangGraphError",
    "AutoGenError",
    "AgentError",
    "ToolError",
    "ConfigurationError",
    "APIError",
    "RateLimitError",
    "AuthenticationError",
    "ValidationError",
    "WorkflowError",
    "ResourceError",
    "TimeoutError",
    "ConcurrencyError",
    # Handlers
    "ErrorHandler",
    "get_error_handler",
    "handle_errors",
    "handle_async_errors",
    "ErrorContext"
]