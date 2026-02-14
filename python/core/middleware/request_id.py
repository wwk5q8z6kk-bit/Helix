"""
Request ID Middleware for Correlation Tracking
Adds unique request IDs to every API call for distributed tracing
"""
import uuid
import logging
import time
from contextvars import ContextVar
from typing import Callable
from fastapi import Request, Response
from starlette.middleware.base import BaseHTTPMiddleware
from starlette.types import ASGIApp
from core.exceptions_unified import NetworkError

logger = logging.getLogger(__name__)

# Context variable to store request ID across async contexts
request_id_context: ContextVar[str] = ContextVar("request_id", default=None)


class RequestIDMiddleware(BaseHTTPMiddleware):
    """
    Middleware to add unique request IDs for correlation tracking.

    Features:
    - Generates unique UUID for each request
    - Accepts existing X-Request-ID header from clients
    - Adds request_id to response headers
    - Stores request_id in context for logging
    - Tracks request duration
    """

    def __init__(self, app: ASGIApp, header_name: str = "X-Request-ID"):
        super().__init__(app)
        self.header_name = header_name

    async def dispatch(self, request: Request, call_next: Callable) -> Response:
        # Get or generate request ID
        request_id = request.headers.get(self.header_name)
        if not request_id:
            request_id = str(uuid.uuid4())

        # Store in context for access by other code
        request_id_context.set(request_id)

        # Add to request state
        request.state.request_id = request_id

        # Track request start time
        start_time = time.time()

        # Log request with correlation ID
        logger.info(
            f"Request started",
            extra={
                "request_id": request_id,
                "method": request.method,
                "path": request.url.path,
                "client": request.client.host if request.client else None
            }
        )

        # Process request
        try:
            response = await call_next(request)

            # Calculate duration
            duration = time.time() - start_time

            # Add request ID to response headers
            response.headers[self.header_name] = request_id

            # Log response with correlation ID
            logger.info(
                f"Request completed",
                extra={
                    "request_id": request_id,
                    "method": request.method,
                    "path": request.url.path,
                    "status_code": response.status_code,
                    "duration_ms": round(duration * 1000, 2)
                }
            )

            return response

        except NetworkError as e:
            # Log error with correlation ID
            duration = time.time() - start_time
            logger.error(
                f"Request failed: {str(e)}",
                extra={
                    "request_id": request_id,
                    "method": request.method,
                    "path": request.url.path,
                    "duration_ms": round(duration * 1000, 2),
                    "error": str(e)
                },
                exc_info=True
            )
            raise


def get_request_id() -> str:
    """
    Get current request ID from context.

    Returns:
        Request ID string or None if not in request context
    """
    return request_id_context.get()


def log_with_request_id(level: str, message: str, **kwargs):
    """
    Helper function to log with request ID context.

    Args:
        level: Log level (info, warning, error, debug)
        message: Log message
        **kwargs: Additional context to include in log
    """
    request_id = get_request_id()
    extra = {"request_id": request_id, **kwargs}

    log_func = getattr(logger, level.lower())
    log_func(message, extra=extra)
