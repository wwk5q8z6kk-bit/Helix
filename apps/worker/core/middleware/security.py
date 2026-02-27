#!/usr/bin/env python3
"""
Security Middleware for Helix
Implements rate limiting and security headers
"""

import time
import logging
from typing import Dict, Callable
from datetime import datetime, timedelta
from collections import defaultdict
from fastapi import Request, HTTPException, status
from fastapi.responses import JSONResponse
from starlette.middleware.base import BaseHTTPMiddleware
from starlette.middleware.cors import CORSMiddleware

logger = logging.getLogger(__name__)


class RateLimiter:
    """Simple in-memory rate limiter"""

    def __init__(self, requests_per_minute: int = 60):
        self.requests_per_minute = requests_per_minute
        self.requests: Dict[str, list] = defaultdict(list)

    def is_rate_limited(self, identifier: str) -> tuple[bool, int]:
        """
        Check if identifier is rate limited.

        Returns:
            (is_limited, retry_after_seconds)
        """
        now = time.time()
        minute_ago = now - 60

        # Clean old requests
        self.requests[identifier] = [
            req_time for req_time in self.requests[identifier]
            if req_time > minute_ago
        ]

        # Check limit
        if len(self.requests[identifier]) >= self.requests_per_minute:
            oldest_request = min(self.requests[identifier])
            retry_after = int(60 - (now - oldest_request))
            return True, retry_after

        # Record new request
        self.requests[identifier].append(now)
        return False, 0

    def get_remaining(self, identifier: str) -> int:
        """Get remaining requests for identifier"""
        now = time.time()
        minute_ago = now - 60

        recent_requests = [
            req_time for req_time in self.requests[identifier]
            if req_time > minute_ago
        ]

        return max(0, self.requests_per_minute - len(recent_requests))


# Global rate limiter instances
rate_limiters = {
    "default": RateLimiter(requests_per_minute=60),
    "ai": RateLimiter(requests_per_minute=20),  # AI endpoints more restricted
    "docs": RateLimiter(requests_per_minute=100),  # Docs can be accessed more
}


class RateLimitMiddleware(BaseHTTPMiddleware):
    """Rate limiting middleware"""

    async def dispatch(self, request: Request, call_next: Callable):
        # Skip rate limiting for health checks and docs
        if request.url.path in ["/health", "/docs", "/redoc", "/openapi.json"]:
            return await call_next(request)

        # Get client identifier (IP address)
        client_ip = request.client.host

        # Determine rate limiter based on endpoint
        if request.url.path.startswith("/ai/") or request.url.path.startswith("/chat/"):
            limiter = rate_limiters["ai"]
            limit_type = "AI"
        elif request.url.path.startswith("/api/docs/"):
            limiter = rate_limiters["docs"]
            limit_type = "Docs"
        else:
            limiter = rate_limiters["default"]
            limit_type = "Default"

        # Check rate limit
        is_limited, retry_after = limiter.is_rate_limited(client_ip)

        if is_limited:
            logger.warning(f"Rate limit exceeded for {client_ip} on {request.url.path}")
            return JSONResponse(
                status_code=status.HTTP_429_TOO_MANY_REQUESTS,
                content={
                    "error": "Rate limit exceeded",
                    "detail": "Rate limit exceeded",
                    "message": f"Too many requests. Please try again in {retry_after} seconds.",
                    "retry_after": retry_after,
                    "limit_type": limit_type
                },
                headers={
                    "X-RateLimit-Limit": str(limiter.requests_per_minute),
                    "X-RateLimit-Remaining": "0",
                    "X-RateLimit-Reset": str(int(time.time() + retry_after)),
                    "Retry-After": str(retry_after)
                }
            )

        # Process request
        response = await call_next(request)

        # Add rate limit headers
        remaining = limiter.get_remaining(client_ip)
        response.headers["X-RateLimit-Limit"] = str(limiter.requests_per_minute)
        response.headers["X-RateLimit-Remaining"] = str(remaining)
        response.headers["X-RateLimit-Reset"] = str(int(time.time() + 60))

        return response


class SecurityHeadersMiddleware(BaseHTTPMiddleware):
    """Security headers middleware"""

    async def dispatch(self, request: Request, call_next: Callable):
        response = await call_next(request)

        # Security headers
        response.headers["X-Content-Type-Options"] = "nosniff"
        response.headers["X-Frame-Options"] = "DENY"
        response.headers["X-XSS-Protection"] = "1; mode=block"
        response.headers["Strict-Transport-Security"] = "max-age=31536000; includeSubDomains"
        response.headers["Content-Security-Policy"] = (
            "default-src 'self'; "
            "script-src 'self' 'unsafe-inline'; "
            "style-src 'self' 'unsafe-inline'; "
            "img-src 'self' data: https:; "
            "font-src 'self' data:; "
            "connect-src 'self' http://localhost:*; "  # Allow local API calls
            "frame-ancestors 'none';"
        )
        response.headers["Referrer-Policy"] = "strict-origin-when-cross-origin"
        response.headers["Permissions-Policy"] = (
            "geolocation=(), microphone=(), camera=(), "
            "payment=(), usb=(), magnetometer=()"
        )

        # Server identity
        response.headers["Server"] = "Helix"
        response.headers["X-Powered-By"] = "Helix AI Platform"

        return response


def setup_security_middleware(app):
    """Setup all security middleware"""

    # CORS middleware
    app.add_middleware(
        CORSMiddleware,
        allow_origins=[
            "http://localhost:3000",
            "http://localhost:8000",
            "http://localhost:8001",
            "http://127.0.0.1:3000",
            "http://127.0.0.1:8000",
            "http://127.0.0.1:8001",
        ],
        allow_credentials=True,
        allow_methods=["*"],
        allow_headers=["*"],
        expose_headers=[
            "X-RateLimit-Limit",
            "X-RateLimit-Remaining",
            "X-RateLimit-Reset"
        ]
    )

    # Security headers
    app.add_middleware(SecurityHeadersMiddleware)

    # Rate limiting
    app.add_middleware(RateLimitMiddleware)

    logger.info("Security middleware configured: CORS, Security Headers, Rate Limiting")


def get_rate_limit_status() -> Dict[str, any]:
    """Get current rate limit status"""
    status = {}

    for name, limiter in rate_limiters.items():
        total_clients = len(limiter.requests)
        total_requests = sum(len(requests) for requests in limiter.requests.values())

        status[name] = {
            "limit_per_minute": limiter.requests_per_minute,
            "active_clients": total_clients,
            "total_requests_last_minute": total_requests
        }

    return status
