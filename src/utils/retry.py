"""Retry mechanisms for external API calls with exponential backoff."""

import asyncio
import time
import random
from typing import Callable, Any, Optional, Type, Union, List
from functools import wraps
from datetime import datetime, timedelta

from ..core.logging import get_component_logger
from ..core.errors import (
    APIError,
    RateLimitError,
    TimeoutError,
    AgenticSystemError,
    handle_errors
)
from ..core.monitoring import get_metrics_collector


class RetryConfig:
    """Configuration for retry mechanisms."""
    
    def __init__(
        self,
        max_attempts: int = 3,
        initial_delay: float = 1.0,
        max_delay: float = 60.0,
        backoff_factor: float = 2.0,
        jitter: bool = True,
        retry_on: Optional[List[Type[Exception]]] = None
    ):
        self.max_attempts = max_attempts
        self.initial_delay = initial_delay
        self.max_delay = max_delay
        self.backoff_factor = backoff_factor
        self.jitter = jitter
        self.retry_on = retry_on or [APIError, TimeoutError, ConnectionError]


class RetryState:
    """State tracking for retry attempts."""
    
    def __init__(self, operation_name: str, config: RetryConfig):
        self.operation_name = operation_name
        self.config = config
        self.attempt = 0
        self.start_time = time.time()
        self.last_exception: Optional[Exception] = None
        self.total_delay = 0.0


class RetryManager:
    """Central retry management system."""
    
    def __init__(self, default_config: Optional[RetryConfig] = None):
        self.default_config = default_config or RetryConfig()
        self.logger = get_component_logger("retry_manager")
        self.metrics = get_metrics_collector()
    
    def should_retry(self, exception: Exception, state: RetryState) -> bool:
        """Determine if an operation should be retried."""
        
        # Check if we've exceeded max attempts
        if state.attempt >= state.config.max_attempts:
            return False
        
        # Check if the exception type is retryable
        if not any(isinstance(exception, exc_type) for exc_type in state.config.retry_on):
            return False
        
        # Special handling for rate limit errors
        if isinstance(exception, RateLimitError):
            # Always retry rate limit errors (within attempt limits)
            return True
        
        # For API errors, check status codes if available
        if isinstance(exception, APIError):
            if hasattr(exception, 'status_code') and exception.status_code:
                # Don't retry client errors (4xx except 429)
                if 400 <= exception.status_code < 500 and exception.status_code != 429:
                    return False
            return True
        
        return True
    
    def calculate_delay(self, state: RetryState, exception: Exception) -> float:
        """Calculate delay before next retry attempt."""
        
        # Special handling for rate limit errors
        if isinstance(exception, RateLimitError) and hasattr(exception, 'retry_after'):
            if exception.retry_after:
                delay = float(exception.retry_after)
                self.logger.info(f"Rate limited, waiting {delay}s as requested")
                return min(delay, state.config.max_delay)
        
        # Exponential backoff
        delay = state.config.initial_delay * (state.config.backoff_factor ** (state.attempt - 1))
        
        # Add jitter to prevent thundering herd
        if state.config.jitter:
            jitter_range = delay * 0.1  # 10% jitter
            delay += random.uniform(-jitter_range, jitter_range)
        
        # Cap at max delay
        delay = min(delay, state.config.max_delay)
        
        return max(0, delay)  # Ensure non-negative
    
    async def execute_with_retry(
        self,
        operation: Callable,
        operation_name: str,
        config: Optional[RetryConfig] = None,
        context: Optional[dict] = None
    ) -> Any:
        """Execute an async operation with retry logic."""
        
        config = config or self.default_config
        state = RetryState(operation_name, config)
        context = context or {}
        
        self.logger.debug(f"Starting retry operation: {operation_name}")
        
        while True:
            state.attempt += 1
            
            try:
                # Record attempt metrics
                self.metrics.record_counter(
                    "retry_attempt",
                    operation=operation_name,
                    attempt=str(state.attempt)
                )
                
                # Execute the operation
                result = await operation()
                
                # Success - record metrics and return
                total_time = time.time() - state.start_time
                self.metrics.record_timing(
                    "retry_success",
                    total_time,
                    operation=operation_name,
                    attempts=str(state.attempt)
                )
                
                if state.attempt > 1:
                    self.logger.info(
                        f"Operation {operation_name} succeeded after {state.attempt} attempts "
                        f"(total time: {total_time:.2f}s)"
                    )
                
                return result
                
            except Exception as e:
                state.last_exception = e
                
                # Check if we should retry
                if not self.should_retry(e, state):
                    # No more retries - record failure and re-raise
                    total_time = time.time() - state.start_time
                    self.metrics.record_counter(
                        "retry_failed",
                        operation=operation_name,
                        attempts=str(state.attempt),
                        exception_type=e.__class__.__name__
                    )
                    
                    self.logger.error(
                        f"Operation {operation_name} failed after {state.attempt} attempts "
                        f"(total time: {total_time:.2f}s): {e}"
                    )
                    raise
                
                # Calculate delay for next attempt
                delay = self.calculate_delay(state, e)
                state.total_delay += delay
                
                self.logger.warning(
                    f"Operation {operation_name} failed (attempt {state.attempt}), "
                    f"retrying in {delay:.2f}s: {e}"
                )
                
                # Record retry metrics
                self.metrics.record_counter(
                    "retry_delay",
                    operation=operation_name,
                    attempt=str(state.attempt),
                    exception_type=e.__class__.__name__
                )
                
                # Wait before retry
                await asyncio.sleep(delay)
    
    def execute_with_retry_sync(
        self,
        operation: Callable,
        operation_name: str,
        config: Optional[RetryConfig] = None,
        context: Optional[dict] = None
    ) -> Any:
        """Execute a sync operation with retry logic."""
        
        config = config or self.default_config
        state = RetryState(operation_name, config)
        context = context or {}
        
        self.logger.debug(f"Starting retry operation: {operation_name}")
        
        while True:
            state.attempt += 1
            
            try:
                # Record attempt metrics
                self.metrics.record_counter(
                    "retry_attempt",
                    operation=operation_name,
                    attempt=str(state.attempt)
                )
                
                # Execute the operation
                result = operation()
                
                # Success - record metrics and return
                total_time = time.time() - state.start_time
                self.metrics.record_timing(
                    "retry_success",
                    total_time,
                    operation=operation_name,
                    attempts=str(state.attempt)
                )
                
                if state.attempt > 1:
                    self.logger.info(
                        f"Operation {operation_name} succeeded after {state.attempt} attempts "
                        f"(total time: {total_time:.2f}s)"
                    )
                
                return result
                
            except Exception as e:
                state.last_exception = e
                
                # Check if we should retry
                if not self.should_retry(e, state):
                    # No more retries - record failure and re-raise
                    total_time = time.time() - state.start_time
                    self.metrics.record_counter(
                        "retry_failed",
                        operation=operation_name,
                        attempts=str(state.attempt),
                        exception_type=e.__class__.__name__
                    )
                    
                    self.logger.error(
                        f"Operation {operation_name} failed after {state.attempt} attempts "
                        f"(total time: {total_time:.2f}s): {e}"
                    )
                    raise
                
                # Calculate delay for next attempt
                delay = self.calculate_delay(state, e)
                state.total_delay += delay
                
                self.logger.warning(
                    f"Operation {operation_name} failed (attempt {state.attempt}), "
                    f"retrying in {delay:.2f}s: {e}"
                )
                
                # Record retry metrics
                self.metrics.record_counter(
                    "retry_delay",
                    operation=operation_name,
                    attempt=str(state.attempt),
                    exception_type=e.__class__.__name__
                )
                
                # Wait before retry
                time.sleep(delay)


# Global retry manager instance
_retry_manager: Optional[RetryManager] = None


def get_retry_manager(config: Optional[RetryConfig] = None) -> RetryManager:
    """Get the global retry manager instance."""
    global _retry_manager
    if _retry_manager is None:
        _retry_manager = RetryManager(config)
    return _retry_manager


def retry_async(
    operation_name: Optional[str] = None,
    max_attempts: int = 3,
    initial_delay: float = 1.0,
    max_delay: float = 60.0,
    backoff_factor: float = 2.0,
    jitter: bool = True,
    retry_on: Optional[List[Type[Exception]]] = None
):
    """Decorator for async functions with retry logic."""
    
    def decorator(func):
        @wraps(func)
        async def wrapper(*args, **kwargs):
            config = RetryConfig(
                max_attempts=max_attempts,
                initial_delay=initial_delay,
                max_delay=max_delay,
                backoff_factor=backoff_factor,
                jitter=jitter,
                retry_on=retry_on
            )
            
            name = operation_name or f"{func.__module__}.{func.__name__}"
            retry_manager = get_retry_manager()
            
            async def operation():
                return await func(*args, **kwargs)
            
            return await retry_manager.execute_with_retry(operation, name, config)
        
        return wrapper
    return decorator


def retry_sync(
    operation_name: Optional[str] = None,
    max_attempts: int = 3,
    initial_delay: float = 1.0,
    max_delay: float = 60.0,
    backoff_factor: float = 2.0,
    jitter: bool = True,
    retry_on: Optional[List[Type[Exception]]] = None
):
    """Decorator for sync functions with retry logic."""
    
    def decorator(func):
        @wraps(func)
        def wrapper(*args, **kwargs):
            config = RetryConfig(
                max_attempts=max_attempts,
                initial_delay=initial_delay,
                max_delay=max_delay,
                backoff_factor=backoff_factor,
                jitter=jitter,
                retry_on=retry_on
            )
            
            name = operation_name or f"{func.__module__}.{func.__name__}"
            retry_manager = get_retry_manager()
            
            def operation():
                return func(*args, **kwargs)
            
            return retry_manager.execute_with_retry_sync(operation, name, config)
        
        return wrapper
    return decorator


class CircuitBreaker:
    """Circuit breaker pattern for preventing cascading failures."""
    
    def __init__(
        self,
        failure_threshold: int = 5,
        recovery_timeout: float = 60.0,
        expected_exception: Type[Exception] = Exception
    ):
        self.failure_threshold = failure_threshold
        self.recovery_timeout = recovery_timeout
        self.expected_exception = expected_exception
        
        self.failure_count = 0
        self.last_failure_time: Optional[datetime] = None
        self.state = "CLOSED"  # CLOSED, OPEN, HALF_OPEN
        
        self.logger = get_component_logger("circuit_breaker")
        self.metrics = get_metrics_collector()
    
    def __call__(self, func):
        @wraps(func)
        async def wrapper(*args, **kwargs):
            return await self._call_async(func, *args, **kwargs)
        return wrapper
    
    async def _call_async(self, func, *args, **kwargs):
        # Check circuit state
        if self.state == "OPEN":
            if self._should_attempt_reset():
                self.state = "HALF_OPEN"
                self.logger.info("Circuit breaker state: HALF_OPEN")
            else:
                self.metrics.record_counter("circuit_breaker_blocked")
                raise AgenticSystemError("Circuit breaker is OPEN")
        
        try:
            # Execute function
            result = await func(*args, **kwargs) if asyncio.iscoroutinefunction(func) else func(*args, **kwargs)
            
            # Success - reset failure count if we were half open
            if self.state == "HALF_OPEN":
                self._reset()
            
            return result
            
        except self.expected_exception as e:
            self._record_failure()
            raise
    
    def _should_attempt_reset(self) -> bool:
        return (
            self.last_failure_time is not None and
            datetime.now() - self.last_failure_time >= timedelta(seconds=self.recovery_timeout)
        )
    
    def _record_failure(self):
        self.failure_count += 1
        self.last_failure_time = datetime.now()
        
        if self.failure_count >= self.failure_threshold:
            self.state = "OPEN"
            self.logger.warning(f"Circuit breaker opened after {self.failure_count} failures")
            self.metrics.record_counter("circuit_breaker_opened")
    
    def _reset(self):
        self.failure_count = 0
        self.state = "CLOSED"
        self.last_failure_time = None
        self.logger.info("Circuit breaker reset to CLOSED")
        self.metrics.record_counter("circuit_breaker_reset")