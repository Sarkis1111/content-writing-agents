"""Error handling framework with framework-specific handling."""

import traceback
from typing import Type, Callable, Dict, Any, Optional, List
from functools import wraps
from datetime import datetime

from .exceptions import AgenticSystemError, FrameworkError
from ..logging import get_component_logger
from ..monitoring import get_metrics_collector


class ErrorHandler:
    """Central error handling system."""
    
    def __init__(self):
        self.handlers: Dict[Type[Exception], List[Callable]] = {}
        self.global_handlers: List[Callable] = []
        self.logger = get_component_logger("error_handler")
        self.metrics = get_metrics_collector()
    
    def register_handler(self, exception_type: Type[Exception], handler: Callable):
        """Register an exception handler for a specific exception type."""
        if exception_type not in self.handlers:
            self.handlers[exception_type] = []
        self.handlers[exception_type].append(handler)
        self.logger.info(f"Registered handler for {exception_type.__name__}")
    
    def register_global_handler(self, handler: Callable):
        """Register a global exception handler."""
        self.global_handlers.append(handler)
        self.logger.info("Registered global exception handler")
    
    def handle_exception(self, exception: Exception, context: Optional[Dict[str, Any]] = None):
        """Handle an exception using registered handlers."""
        context = context or {}
        
        # Record metrics
        self._record_error_metrics(exception, context)
        
        # Log the exception
        self._log_exception(exception, context)
        
        # Find and execute appropriate handlers
        handlers_executed = False
        
        # Execute specific handlers
        for exception_type, handler_list in self.handlers.items():
            if isinstance(exception, exception_type):
                for handler in handler_list:
                    try:
                        handler(exception, context)
                        handlers_executed = True
                    except Exception as handler_error:
                        self.logger.error(
                            f"Error in exception handler: {handler_error}",
                            exc_info=True
                        )
        
        # Execute global handlers if no specific handlers were found
        if not handlers_executed:
            for handler in self.global_handlers:
                try:
                    handler(exception, context)
                except Exception as handler_error:
                    self.logger.error(
                        f"Error in global exception handler: {handler_error}",
                        exc_info=True
                    )
    
    def _record_error_metrics(self, exception: Exception, context: Dict[str, Any]):
        """Record error metrics."""
        tags = {
            "exception_type": exception.__class__.__name__,
            "module": exception.__class__.__module__
        }
        
        # Add context-specific tags
        if hasattr(exception, 'framework') and exception.framework:
            tags["framework"] = exception.framework
        
        if hasattr(exception, 'agent') and exception.agent:
            tags["agent"] = exception.agent
        
        if hasattr(exception, 'tool') and exception.tool:
            tags["tool"] = exception.tool
        
        # Add workflow context if available
        workflow_id = context.get("workflow_id")
        if workflow_id:
            tags["workflow_id"] = workflow_id
        
        self.metrics.record_counter("error_occurred", **tags)
        
        # Record framework-specific error if applicable
        if isinstance(exception, FrameworkError):
            self.metrics.record_counter(
                "framework_error",
                framework=exception.framework,
                error_type=exception.__class__.__name__
            )
    
    def _log_exception(self, exception: Exception, context: Dict[str, Any]):
        """Log exception details."""
        error_info = {
            "exception_type": exception.__class__.__name__,
            "message": str(exception),
            "traceback": traceback.format_exc(),
            "context": context
        }
        
        # Add structured logging for AgenticSystemError
        if isinstance(exception, AgenticSystemError):
            error_info.update(exception.to_dict())
        
        self.logger.error(f"Exception occurred: {exception}", extra=error_info)


# Global error handler instance
_error_handler: Optional[ErrorHandler] = None


def get_error_handler() -> ErrorHandler:
    """Get the global error handler instance."""
    global _error_handler
    if _error_handler is None:
        _error_handler = ErrorHandler()
        _setup_default_handlers(_error_handler)
    return _error_handler


def _setup_default_handlers(handler: ErrorHandler):
    """Set up default error handlers."""
    
    # CrewAI error handler
    def handle_crewai_error(exception: Exception, context: Dict[str, Any]):
        from .exceptions import CrewAIError
        if isinstance(exception, CrewAIError):
            logger = get_component_logger("crewai_error_handler")
            logger.error(f"CrewAI error in crew '{exception.crew_name}': {exception.message}")
    
    # LangGraph error handler  
    def handle_langgraph_error(exception: Exception, context: Dict[str, Any]):
        from .exceptions import LangGraphError
        if isinstance(exception, LangGraphError):
            logger = get_component_logger("langgraph_error_handler")
            logger.error(f"LangGraph error in workflow '{exception.workflow_name}': {exception.message}")
    
    # AutoGen error handler
    def handle_autogen_error(exception: Exception, context: Dict[str, Any]):
        from .exceptions import AutoGenError
        if isinstance(exception, AutoGenError):
            logger = get_component_logger("autogen_error_handler")
            logger.error(f"AutoGen error in group chat '{exception.group_chat_name}': {exception.message}")
    
    # API error handler
    def handle_api_error(exception: Exception, context: Dict[str, Any]):
        from .exceptions import APIError, RateLimitError
        if isinstance(exception, RateLimitError):
            logger = get_component_logger("api_error_handler")
            logger.warning(f"Rate limit hit for {exception.api_name}: {exception.message}")
        elif isinstance(exception, APIError):
            logger = get_component_logger("api_error_handler")
            logger.error(f"API error for {exception.api_name}: {exception.message}")
    
    # Register handlers
    from .exceptions import (
        CrewAIError, LangGraphError, AutoGenError, APIError, RateLimitError
    )
    
    handler.register_handler(CrewAIError, handle_crewai_error)
    handler.register_handler(LangGraphError, handle_langgraph_error)
    handler.register_handler(AutoGenError, handle_autogen_error)
    handler.register_handler(APIError, handle_api_error)
    handler.register_handler(RateLimitError, handle_api_error)


def handle_errors(
    reraise: bool = False,
    default_return: Any = None,
    context: Optional[Dict[str, Any]] = None
):
    """Decorator for automatic error handling."""
    
    def decorator(func):
        @wraps(func)
        def wrapper(*args, **kwargs):
            try:
                return func(*args, **kwargs)
            except Exception as e:
                # Merge context
                merged_context = context.copy() if context else {}
                merged_context.update({
                    "function_name": func.__name__,
                    "function_module": func.__module__,
                    "args": str(args)[:200],  # Limit size
                    "kwargs": str(kwargs)[:200]  # Limit size
                })
                
                # Handle the exception
                get_error_handler().handle_exception(e, merged_context)
                
                if reraise:
                    raise
                else:
                    return default_return
        
        return wrapper
    return decorator


def handle_async_errors(
    reraise: bool = False,
    default_return: Any = None,
    context: Optional[Dict[str, Any]] = None
):
    """Decorator for automatic async error handling."""
    
    def decorator(func):
        @wraps(func)
        async def wrapper(*args, **kwargs):
            try:
                return await func(*args, **kwargs)
            except Exception as e:
                # Merge context
                merged_context = context.copy() if context else {}
                merged_context.update({
                    "function_name": func.__name__,
                    "function_module": func.__module__,
                    "args": str(args)[:200],  # Limit size
                    "kwargs": str(kwargs)[:200]  # Limit size
                })
                
                # Handle the exception
                get_error_handler().handle_exception(e, merged_context)
                
                if reraise:
                    raise
                else:
                    return default_return
        
        return wrapper
    return decorator


class ErrorContext:
    """Context manager for error handling with additional context."""
    
    def __init__(self, context: Optional[Dict[str, Any]] = None, reraise: bool = True):
        self.context = context or {}
        self.reraise = reraise
        self.error_handler = get_error_handler()
    
    def __enter__(self):
        return self
    
    def __exit__(self, exc_type, exc_val, exc_tb):
        if exc_type is not None:
            self.error_handler.handle_exception(exc_val, self.context)
            return not self.reraise  # Suppress exception if not reraising
        return False