"""
Rate Limiting Middleware
Token bucket algorithm with Redis backend and in-memory fallback
"""

import time
import hashlib
import logging
from typing import Optional, Dict, Tuple
from collections import defaultdict
from datetime import datetime, timedelta
from fastapi import Request, HTTPException, status
from starlette.middleware.base import BaseHTTPMiddleware
import asyncio

logger = logging.getLogger(__name__)


class TokenBucket:
    """
    Token bucket algorithm for rate limiting

    Attributes:
        capacity: Maximum tokens in bucket
        refill_rate: Tokens added per second
        tokens: Current token count
        last_refill: Last refill timestamp
    """

    def __init__(self, capacity: int, refill_rate: float):
        self.capacity = capacity
        self.refill_rate = refill_rate
        self.tokens = capacity
        self.last_refill = time.time()

    def consume(self, tokens: int = 1) -> bool:
        """
        Try to consume tokens from bucket

        Args:
            tokens: Number of tokens to consume

        Returns:
            True if tokens consumed, False if insufficient tokens
        """
        # Refill bucket based on time elapsed
        now = time.time()
        elapsed = now - self.last_refill
        refill_amount = elapsed * self.refill_rate

        self.tokens = min(self.capacity, self.tokens + refill_amount)
        self.last_refill = now

        # Try to consume tokens
        if self.tokens >= tokens:
            self.tokens -= tokens
            return True

        return False

    def get_wait_time(self, tokens: int = 1) -> float:
        """
        Calculate wait time until tokens available

        Args:
            tokens: Number of tokens needed

        Returns:
            Wait time in seconds
        """
        # Refill bucket first
        now = time.time()
        elapsed = now - self.last_refill
        refill_amount = elapsed * self.refill_rate
        current_tokens = min(self.capacity, self.tokens + refill_amount)

        if current_tokens >= tokens:
            return 0.0

        tokens_needed = tokens - current_tokens
        return tokens_needed / self.refill_rate


class RateLimiter:
    """
    Rate limiter with Redis backend and in-memory fallback
    """

    def __init__(self, redis_client=None):
        """
        Initialize rate limiter

        Args:
            redis_client: Optional Redis client for distributed rate limiting
        """
        self.redis_client = redis_client
        self._buckets: Dict[str, TokenBucket] = {}  # In-memory fallback
        self._lock = asyncio.Lock()

        # Rate limit configurations (endpoint -> (capacity, refill_rate))
        self.limits = {
            "/auth/login": (5, 5/60),  # 5 attempts per minute
            "/auth/register": (3, 3/3600),  # 3 registrations per hour
            "/auth/refresh": (10, 10/60),  # 10 refreshes per minute
            "default": (100, 100/60),  # 100 requests per minute for other endpoints
        }

        logger.info(
            f"Rate limiter initialized with Redis: {self.redis_client is not None}"
        )

    def _get_client_identifier(self, request: Request) -> str:
        """
        Get unique identifier for client (IP + User-Agent)

        Args:
            request: FastAPI request

        Returns:
            Client identifier hash
        """
        # Try to get real IP from X-Forwarded-For header (if behind proxy)
        forwarded_for = request.headers.get("X-Forwarded-For")
        if forwarded_for:
            client_ip = forwarded_for.split(",")[0].strip()
        else:
            client_ip = request.client.host if request.client else "unknown"

        # Include user agent for better uniqueness
        user_agent = request.headers.get("User-Agent", "")

        # Hash the identifier
        identifier = f"{client_ip}:{user_agent}"
        return hashlib.sha256(identifier.encode()).hexdigest()[:16]

    def _get_rate_limit_config(self, path: str) -> Tuple[int, float]:
        """
        Get rate limit configuration for path

        Args:
            path: Request path

        Returns:
            Tuple of (capacity, refill_rate)
        """
        for limit_path, config in self.limits.items():
            if limit_path != "default" and path.startswith(limit_path):
                return config
        return self.limits["default"]

    async def _check_rate_limit_redis(
        self,
        key: str,
        capacity: int,
        refill_rate: float
    ) -> Tuple[bool, Optional[float]]:
        """
        Check rate limit using Redis

        Args:
            key: Rate limit key
            capacity: Bucket capacity
            refill_rate: Refill rate (tokens/second)

        Returns:
            Tuple of (allowed, wait_time)
        """
        try:
            # Use Redis sorted set with sliding window
            now = time.time()
            window_start = now - 60  # 1 minute window

            # Remove old entries
            await asyncio.get_event_loop().run_in_executor(
                None,
                self.redis_client.zremrangebyscore,
                key,
                0,
                window_start
            )

            # Count requests in window
            count = await asyncio.get_event_loop().run_in_executor(
                None,
                self.redis_client.zcard,
                key
            )

            # Check if under limit
            if count < capacity:
                # Add current request
                await asyncio.get_event_loop().run_in_executor(
                    None,
                    self.redis_client.zadd,
                    key,
                    {str(now): now}
                )
                # Set expiry
                await asyncio.get_event_loop().run_in_executor(
                    None,
                    self.redis_client.expire,
                    key,
                    60
                )
                return True, None

            # Calculate wait time
            oldest = await asyncio.get_event_loop().run_in_executor(
                None,
                lambda: self.redis_client.zrange(key, 0, 0, withscores=True)
            )
            if oldest:
                oldest_time = oldest[0][1]
                wait_time = max(0, (oldest_time + 60) - now)
                return False, wait_time

            return False, 60.0  # Default wait time

        except Exception as e:
            logger.error(f"Redis rate limit error: {e}, falling back to memory")
            return None, None  # Signal fallback to memory

    async def _check_rate_limit_memory(
        self,
        key: str,
        capacity: int,
        refill_rate: float
    ) -> Tuple[bool, Optional[float]]:
        """
        Check rate limit using in-memory token bucket

        Args:
            key: Rate limit key
            capacity: Bucket capacity
            refill_rate: Refill rate (tokens/second)

        Returns:
            Tuple of (allowed, wait_time)
        """
        async with self._lock:
            # Get or create bucket
            if key not in self._buckets:
                self._buckets[key] = TokenBucket(capacity, refill_rate)

            bucket = self._buckets[key]

            # Try to consume token
            if bucket.consume(1):
                return True, None

            # Calculate wait time
            wait_time = bucket.get_wait_time(1)
            return False, wait_time

    async def check_rate_limit(
        self,
        request: Request
    ) -> Tuple[bool, Optional[float], Dict[str, str]]:
        """
        Check if request is within rate limit

        Args:
            request: FastAPI request

        Returns:
            Tuple of (allowed, wait_time, headers)
        """
        # Get client identifier and config
        client_id = self._get_client_identifier(request)
        path = request.url.path
        capacity, refill_rate = self._get_rate_limit_config(path)

        # Create rate limit key
        key = f"rate_limit:{path}:{client_id}"

        # Try Redis first, fallback to memory
        if self.redis_client:
            allowed, wait_time = await self._check_rate_limit_redis(
                key, capacity, refill_rate
            )

            # If Redis failed, use memory fallback
            if allowed is None:
                allowed, wait_time = await self._check_rate_limit_memory(
                    key, capacity, refill_rate
                )
        else:
            # Use memory directly
            allowed, wait_time = await self._check_rate_limit_memory(
                key, capacity, refill_rate
            )

        # Calculate remaining requests
        if allowed:
            remaining = capacity - 1
        else:
            remaining = 0

        # Create rate limit headers
        headers = {
            "X-RateLimit-Limit": str(capacity),
            "X-RateLimit-Remaining": str(remaining),
            "X-RateLimit-Reset": str(int(time.time() + (60 if wait_time is None else wait_time))),
        }

        if not allowed and wait_time is not None:
            headers["Retry-After"] = str(int(wait_time))

        return allowed, wait_time, headers


class RateLimitMiddleware(BaseHTTPMiddleware):
    """
    Middleware for rate limiting requests
    """

    def __init__(
        self,
        app,
        redis_client=None,
        excluded_paths: Optional[list] = None
    ):
        """
        Initialize rate limit middleware

        Args:
            app: FastAPI app
            redis_client: Optional Redis client
            excluded_paths: Paths to exclude from rate limiting
        """
        super().__init__(app)
        self.rate_limiter = RateLimiter(redis_client)
        self.excluded_paths = excluded_paths or [
            "/health",
            "/docs",
            "/redoc",
            "/openapi.json",
            "/metrics",
        ]

    async def dispatch(self, request: Request, call_next):
        """Process request with rate limiting"""

        # Skip rate limiting for excluded paths
        if any(request.url.path.startswith(path) for path in self.excluded_paths):
            return await call_next(request)

        # Check rate limit
        allowed, wait_time, headers = await self.rate_limiter.check_rate_limit(request)

        if not allowed:
            # Log rate limit exceeded
            client_id = self.rate_limiter._get_client_identifier(request)
            logger.warning(
                f"Rate limit exceeded for {request.url.path} "
                f"from {request.client.host if request.client else 'unknown'} "
                f"(client_id: {client_id[:8]}...)"
            )

            # Return 429 Too Many Requests
            raise HTTPException(
                status_code=status.HTTP_429_TOO_MANY_REQUESTS,
                detail=f"Rate limit exceeded. Try again in {int(wait_time)} seconds.",
                headers=headers
            )

        # Add rate limit headers to response
        response = await call_next(request)
        for key, value in headers.items():
            response.headers[key] = value

        return response


# Global rate limiter instance
_rate_limiter: Optional[RateLimiter] = None


async def get_rate_limiter(redis_client=None) -> RateLimiter:
    """
    Get or create global rate limiter instance

    Args:
        redis_client: Optional Redis client

    Returns:
        RateLimiter instance
    """
    global _rate_limiter
    if _rate_limiter is None:
        _rate_limiter = RateLimiter(redis_client)
    return _rate_limiter


async def check_rate_limit(
    request: Request,
    rate_limiter: Optional[RateLimiter] = None
) -> None:
    """
    Dependency for checking rate limit

    Args:
        request: FastAPI request
        rate_limiter: Optional rate limiter instance

    Raises:
        HTTPException: If rate limit exceeded
    """
    if rate_limiter is None:
        rate_limiter = await get_rate_limiter()

    allowed, wait_time, headers = await rate_limiter.check_rate_limit(request)

    if not allowed:
        raise HTTPException(
            status_code=status.HTTP_429_TOO_MANY_REQUESTS,
            detail=f"Rate limit exceeded. Try again in {int(wait_time)} seconds.",
            headers=headers
        )
