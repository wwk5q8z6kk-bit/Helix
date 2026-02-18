"""Tests for middleware components (rate_limiter, security, request_id)."""

import time
import pytest
from unittest.mock import MagicMock, AsyncMock, patch

from core.middleware.rate_limiter import TokenBucket
from core.middleware.security import RateLimiter as SecurityRateLimiter


# ===== TokenBucket (rate_limiter.py) =====


def test_token_bucket_initial_full():
    bucket = TokenBucket(capacity=10, refill_rate=1.0)
    assert bucket.tokens == 10


def test_token_bucket_consume():
    bucket = TokenBucket(capacity=10, refill_rate=1.0)
    assert bucket.consume(1) is True
    # tokens should be less than initial (roughly 9, accounting for tiny refill)
    assert bucket.tokens < 10


def test_token_bucket_exhaustion():
    bucket = TokenBucket(capacity=3, refill_rate=0.0)  # no refill
    assert bucket.consume(1) is True
    assert bucket.consume(1) is True
    assert bucket.consume(1) is True
    assert bucket.consume(1) is False


def test_token_bucket_refill():
    bucket = TokenBucket(capacity=5, refill_rate=1000.0)  # fast refill
    bucket.tokens = 0
    bucket.last_refill = time.time() - 1  # 1 second ago
    # After 1 second at 1000/s rate, should be back to capacity
    assert bucket.consume(1) is True


def test_token_bucket_wait_time():
    bucket = TokenBucket(capacity=5, refill_rate=1.0)
    bucket.tokens = 0
    bucket.last_refill = time.time()
    wait = bucket.get_wait_time(1)
    assert wait > 0


# ===== Security RateLimiter (security.py) =====


def test_security_rate_limiter_not_limited_initially():
    limiter = SecurityRateLimiter(requests_per_minute=5)
    is_limited, _ = limiter.is_rate_limited("client1")
    assert is_limited is False


def test_security_rate_limiter_exceeds_limit():
    limiter = SecurityRateLimiter(requests_per_minute=3)
    for _ in range(3):
        limiter.is_rate_limited("client1")
    is_limited, retry_after = limiter.is_rate_limited("client1")
    assert is_limited is True
    assert retry_after > 0


def test_security_rate_limiter_remaining():
    limiter = SecurityRateLimiter(requests_per_minute=10)
    assert limiter.get_remaining("client1") == 10
    limiter.is_rate_limited("client1")
    assert limiter.get_remaining("client1") == 9
    limiter.is_rate_limited("client1")
    assert limiter.get_remaining("client1") == 8


def test_security_rate_limiter_separate_clients():
    limiter = SecurityRateLimiter(requests_per_minute=2)
    limiter.is_rate_limited("client1")
    limiter.is_rate_limited("client1")
    # client1 exhausted
    is_limited, _ = limiter.is_rate_limited("client1")
    assert is_limited is True
    # client2 still fresh
    is_limited, _ = limiter.is_rate_limited("client2")
    assert is_limited is False


# ===== RequestIDMiddleware (request_id.py) =====


@pytest.mark.asyncio
async def test_request_id_generation():
    """Middleware generates a UUID when no X-Request-ID header is present."""
    from core.middleware.request_id import RequestIDMiddleware

    captured_response_headers = {}

    async def fake_app(scope, receive, send):
        pass

    async def fake_call_next(request):
        resp = MagicMock()
        resp.status_code = 200
        resp.headers = captured_response_headers
        return resp

    middleware = RequestIDMiddleware(app=fake_app)

    request = MagicMock()
    request.headers = {}  # no X-Request-ID
    request.method = "GET"
    request.url.path = "/test"
    request.client.host = "127.0.0.1"
    request.state = MagicMock()

    response = await middleware.dispatch(request, fake_call_next)
    # Should have set X-Request-ID on the response
    assert "X-Request-ID" in response.headers
    # Should be a UUID-style string
    assert len(response.headers["X-Request-ID"]) == 36


@pytest.mark.asyncio
async def test_request_id_passthrough():
    """Middleware honours an existing X-Request-ID header."""
    from core.middleware.request_id import RequestIDMiddleware

    captured_response_headers = {}
    existing_id = "my-custom-request-id-12345"

    async def fake_app(scope, receive, send):
        pass

    async def fake_call_next(request):
        resp = MagicMock()
        resp.status_code = 200
        resp.headers = captured_response_headers
        return resp

    middleware = RequestIDMiddleware(app=fake_app)

    request = MagicMock()
    request.headers = {"X-Request-ID": existing_id}
    request.method = "GET"
    request.url.path = "/test"
    request.client.host = "127.0.0.1"
    request.state = MagicMock()

    response = await middleware.dispatch(request, fake_call_next)
    assert response.headers["X-Request-ID"] == existing_id
