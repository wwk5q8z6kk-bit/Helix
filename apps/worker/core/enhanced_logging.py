"""Helix enhanced logging — thin compatibility shim.

Provides get_logger, track_performance, and log_security_event used by
several modules that originally imported from the Helix enhanced_logging
module. Delegates to Python's standard logging library.
"""

import logging
import functools
import time
from typing import Any, Callable, Optional


def get_logger(name: str) -> logging.Logger:
    return logging.getLogger(name)


def track_performance(func: Optional[Callable] = None, *, operation: str = ""):
    """Decorator that logs execution time of a function."""
    def decorator(fn: Callable) -> Callable:
        op = operation or fn.__qualname__

        @functools.wraps(fn)
        def sync_wrapper(*args: Any, **kwargs: Any) -> Any:
            start = time.perf_counter()
            try:
                return fn(*args, **kwargs)
            finally:
                elapsed = time.perf_counter() - start
                logging.getLogger(fn.__module__).debug(
                    "%s completed in %.3fs", op, elapsed
                )

        @functools.wraps(fn)
        async def async_wrapper(*args: Any, **kwargs: Any) -> Any:
            start = time.perf_counter()
            try:
                return await fn(*args, **kwargs)
            finally:
                elapsed = time.perf_counter() - start
                logging.getLogger(fn.__module__).debug(
                    "%s completed in %.3fs", op, elapsed
                )

        import asyncio
        if asyncio.iscoroutinefunction(fn):
            return async_wrapper
        return sync_wrapper

    if func is not None:
        return decorator(func)
    return decorator


def log_security_event(event_type: str, **kwargs: Any) -> None:
    """Log a security-relevant event."""
    logger = logging.getLogger("helix.security")
    logger.info("Security event: %s %s", event_type, kwargs)
