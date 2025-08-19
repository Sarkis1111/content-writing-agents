"""Simple retry decorator for tools without complex dependencies."""

import asyncio
import time
import random
from typing import Callable, Any, Optional, Type, Union, List
from functools import wraps


def with_retry(
    max_attempts: int = 3,
    delay: float = 1.0,
    backoff_factor: float = 2.0,
    exceptions: tuple = (Exception,)
):
    """Simple retry decorator for both sync and async functions."""
    
    def decorator(func: Callable) -> Callable:
        @wraps(func)
        async def async_wrapper(*args, **kwargs):
            last_exception = None
            current_delay = delay
            
            for attempt in range(max_attempts):
                try:
                    if asyncio.iscoroutinefunction(func):
                        return await func(*args, **kwargs)
                    else:
                        return func(*args, **kwargs)
                except exceptions as e:
                    last_exception = e
                    if attempt == max_attempts - 1:
                        # Last attempt failed
                        break
                    
                    # Wait before retry with exponential backoff
                    sleep_time = current_delay + random.uniform(0, current_delay * 0.1)  # Add jitter
                    await asyncio.sleep(sleep_time)
                    current_delay *= backoff_factor
            
            # All attempts failed
            raise last_exception
        
        @wraps(func)
        def sync_wrapper(*args, **kwargs):
            last_exception = None
            current_delay = delay
            
            for attempt in range(max_attempts):
                try:
                    return func(*args, **kwargs)
                except exceptions as e:
                    last_exception = e
                    if attempt == max_attempts - 1:
                        # Last attempt failed
                        break
                    
                    # Wait before retry with exponential backoff
                    sleep_time = current_delay + random.uniform(0, current_delay * 0.1)  # Add jitter
                    time.sleep(sleep_time)
                    current_delay *= backoff_factor
            
            # All attempts failed
            raise last_exception
        
        # Return appropriate wrapper based on whether function is async
        if asyncio.iscoroutinefunction(func):
            return async_wrapper
        else:
            return sync_wrapper
    
    return decorator