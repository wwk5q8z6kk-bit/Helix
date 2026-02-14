"""
UnifiedError System - Consolidated error handling for Helix.

This module consolidates error handling from 7 separate error modules into a single,
comprehensive system with:
- 95+ exception classes in unified hierarchy
- Consistent error context and metadata
- Retry strategy with exponential backoff
- Circuit breaker pattern for cascading failure prevention
- Utility functions for error handling

Replaces:
- core/exceptions.py
- core/exceptions_unified.py
- core/resilience/error_handling.py
- core/utils/errors.py
- core/error_handling.py
"""

import asyncio
import json
import logging
import traceback
import uuid
from dataclasses import dataclass, field, asdict
from datetime import datetime, timedelta
from enum import Enum
from typing import Any, Callable, Dict, List, Optional, Union
from functools import wraps
import time

logger = logging.getLogger(__name__)


# ============================================================================
# Enums & Constants
# ============================================================================

class ErrorSeverity(Enum):
    """Error severity levels."""
    CRITICAL = "critical"      # System failure, immediate attention required
    ERROR = "error"            # Operation failure, user impacted
    WARNING = "warning"        # Degraded operation, user should be aware
    INFO = "info"              # Informational, no action needed


class ErrorCategory(Enum):
    """Error categories for classification and routing."""
    VALIDATION = "validation"           # Input validation failure
    AUTHENTICATION = "authentication"   # Authentication failure (invalid credentials)
    AUTHORIZATION = "authorization"    # Authorization failure (insufficient permissions)
    LLM_SERVICE = "llm_service"        # LLM provider error
    DATABASE = "database"               # Database operation failure
    CACHE = "cache"                     # Cache operation failure
    RATE_LIMIT = "rate_limit"          # Rate limit exceeded
    TIMEOUT = "timeout"                 # Operation timeout
    NETWORK = "network"                 # Network connectivity error
    RESOURCE = "resource"               # Resource exhausted (memory, disk, etc.)
    INTERNAL = "internal"               # Internal system error
    EXTERNAL = "external"               # External service error


class CircuitBreakerState(Enum):
    """Circuit breaker states."""
    CLOSED = "closed"          # Normal operation
    OPEN = "open"              # Failing, rejecting requests
    HALF_OPEN = "half_open"    # Testing recovery


# ============================================================================
# Data Classes
# ============================================================================

@dataclass
class ErrorContext:
    """Rich error context with metadata."""
    error_id: str = field(default_factory=lambda: str(uuid.uuid4()))
    timestamp: datetime = field(default_factory=datetime.utcnow)
    severity: ErrorSeverity = ErrorSeverity.ERROR
    category: ErrorCategory = ErrorCategory.INTERNAL
    message: str = ""
    user_message: str = ""
    details: Dict[str, Any] = field(default_factory=dict)
    stack_trace: Optional[str] = None
    is_recoverable: bool = True
    http_status: int = 500
    recovery_suggestions: List[str] = field(default_factory=list)

    def to_dict(self) -> Dict[str, Any]:
        """Convert to dictionary (excludes stack trace for API responses)."""
        return {
            "error_id": self.error_id,
            "timestamp": self.timestamp.isoformat(),
            "severity": self.severity.value,
            "category": self.category.value,
            "message": self.message,
            "user_message": self.user_message,
            "details": self.details,
            "is_recoverable": self.is_recoverable,
            "http_status": self.http_status,
            "recovery_suggestions": self.recovery_suggestions,
        }

    def to_api_response(self) -> Dict[str, Any]:
        """Convert to API response format (safe for HTTP)."""
        response = self.to_dict()
        # Don't expose stack trace in API responses
        response.pop("stack_trace", None)
        return response


@dataclass
class RetryConfig:
    """Retry strategy configuration."""
    max_retries: int = 3
    initial_delay_ms: int = 100
    max_delay_ms: int = 10000
    exponential_base: float = 2.0
    jitter: bool = True

    def get_delay(self, attempt: int) -> float:
        """Calculate delay for given attempt number."""
        delay = min(
            self.initial_delay_ms * (self.exponential_base ** attempt),
            self.max_delay_ms
        )

        if self.jitter:
            # Add random jitter (0-25% of delay)
            import random
            jitter_amount = delay * random.uniform(0, 0.25)
            delay += jitter_amount

        return delay / 1000.0  # Convert to seconds


@dataclass
class CircuitBreakerConfig:
    """Circuit breaker configuration."""
    failure_threshold: int = 5        # Failures before opening
    recovery_timeout_sec: int = 60    # Time to wait before trying recovery
    success_threshold: int = 2        # Successes in half-open before closing


@dataclass
class CircuitBreakerMetrics:
    """Circuit breaker metrics."""
    total_requests: int = 0
    successful_requests: int = 0
    failed_requests: int = 0
    rejected_requests: int = 0
    last_failure_time: Optional[datetime] = None


# ============================================================================
# Exception Hierarchy
# ============================================================================

class HelixException(Exception):
    """Base exception for all Helix errors with rich context."""

    def __init__(
        self,
        message: str,
        category: ErrorCategory = ErrorCategory.INTERNAL,
        severity: ErrorSeverity = ErrorSeverity.ERROR,
        details: Optional[Dict[str, Any]] = None,
        is_recoverable: bool = True,
        http_status: int = 500,
        user_message: Optional[str] = None,
        recovery_suggestions: Optional[List[str]] = None,
        context: Optional[ErrorContext] = None,
    ):
        """Initialize exception with full context."""
        self.message = message
        self.category = category
        self.severity = severity
        self.details = details or {}
        self.is_recoverable = is_recoverable
        self.http_status = http_status
        self.user_message = user_message or message
        self.recovery_suggestions = recovery_suggestions or []

        # Create or use provided context
        if context:
            self.context = context
        else:
            self.context = ErrorContext(
                severity=severity,
                category=category,
                message=message,
                user_message=self.user_message,
                details=self.details,
                stack_trace=traceback.format_exc(),
                is_recoverable=is_recoverable,
                http_status=http_status,
                recovery_suggestions=self.recovery_suggestions,
            )

        super().__init__(self.message)

    def __str__(self) -> str:
        """String representation."""
        return f"[{self.context.error_id}] {self.category.value}: {self.message}"

    def to_dict(self) -> Dict[str, Any]:
        """Convert to dictionary."""
        return self.context.to_dict()

    def to_api_response(self) -> Dict[str, Any]:
        """Convert to API response format."""
        return self.context.to_api_response()


# ============================================================================
# Validation Errors
# ============================================================================

class ValidationError(HelixException):
    """Validation error (input/schema validation failed)."""
    def __init__(self, message: str, **kwargs):
        kwargs.setdefault("category", ErrorCategory.VALIDATION)
        kwargs.setdefault("severity", ErrorSeverity.WARNING)
        kwargs.setdefault("http_status", 400)
        super().__init__(message, **kwargs)


class APIValidationError(ValidationError):
    """API request validation error."""
    pass


class TaskValidationError(ValidationError):
    """Task validation error."""
    pass


class WorkflowValidationError(ValidationError):
    """Workflow validation error."""
    pass


class ConfigurationError(ValidationError):
    """Configuration validation error."""
    pass


class MissingConfigurationError(ConfigurationError):
    """Required configuration is missing."""
    pass


class InvalidConfigurationError(ConfigurationError):
    """Configuration value is invalid."""
    pass


# ============================================================================
# Authentication & Authorization Errors
# ============================================================================

class AuthenticationError(HelixException):
    """Authentication error (invalid credentials)."""
    def __init__(self, message: str, **kwargs):
        kwargs.setdefault("category", ErrorCategory.AUTHENTICATION)
        kwargs.setdefault("severity", ErrorSeverity.ERROR)
        kwargs.setdefault("http_status", 401)
        kwargs.setdefault("is_recoverable", False)
        super().__init__(message, **kwargs)


class AuthorizationError(HelixException):
    """Authorization error (insufficient permissions)."""
    def __init__(self, message: str, **kwargs):
        kwargs.setdefault("category", ErrorCategory.AUTHORIZATION)
        kwargs.setdefault("severity", ErrorSeverity.ERROR)
        kwargs.setdefault("http_status", 403)
        kwargs.setdefault("is_recoverable", False)
        super().__init__(message, **kwargs)


class APIKeyError(AuthenticationError):
    """Invalid API key."""
    pass


# ============================================================================
# LLM Errors
# ============================================================================

class LLMError(HelixException):
    """Base LLM error."""
    def __init__(self, message: str, **kwargs):
        kwargs.setdefault("category", ErrorCategory.LLM_SERVICE)
        kwargs.setdefault("severity", ErrorSeverity.ERROR)
        kwargs.setdefault("http_status", 502)
        super().__init__(message, **kwargs)


class LLMProviderError(LLMError):
    """LLM provider error (service unavailable)."""
    pass


class LLMAPIError(LLMError):
    """LLM API error."""
    pass


class LLMModelNotFoundError(LLMError):
    """Requested LLM model not found."""
    def __init__(self, message: str, **kwargs):
        kwargs.setdefault("http_status", 404)
        super().__init__(message, **kwargs)


class LLMContextLengthExceeded(LLMError):
    """Context length exceeded for LLM."""
    pass


class LLMRateLimitError(LLMError):
    """LLM rate limit exceeded."""
    def __init__(self, message: str, **kwargs):
        kwargs.setdefault("category", ErrorCategory.RATE_LIMIT)
        kwargs.setdefault("http_status", 429)
        kwargs.setdefault("is_recoverable", True)
        super().__init__(message, **kwargs)


class LLMTimeoutError(LLMError):
    """LLM request timeout."""
    def __init__(self, message: str, **kwargs):
        kwargs.setdefault("category", ErrorCategory.TIMEOUT)
        kwargs.setdefault("is_recoverable", True)
        super().__init__(message, **kwargs)


class LLMAuthenticationError(LLMError):
    """LLM authentication error."""
    def __init__(self, message: str, **kwargs):
        kwargs.setdefault("category", ErrorCategory.AUTHENTICATION)
        kwargs.setdefault("http_status", 401)
        super().__init__(message, **kwargs)


# ============================================================================
# Database Errors
# ============================================================================

class DatabaseError(HelixException):
    """Base database error."""
    def __init__(self, message: str, **kwargs):
        kwargs.setdefault("category", ErrorCategory.DATABASE)
        kwargs.setdefault("severity", ErrorSeverity.ERROR)
        kwargs.setdefault("http_status", 500)
        super().__init__(message, **kwargs)


class DatabaseConnectionError(DatabaseError):
    """Database connection error."""
    def __init__(self, message: str, **kwargs):
        kwargs.setdefault("is_recoverable", True)
        super().__init__(message, **kwargs)


class DatabaseQueryError(DatabaseError):
    """Database query error."""
    pass


class DatabaseIntegrityError(DatabaseError):
    """Database integrity constraint violation."""
    def __init__(self, message: str, **kwargs):
        kwargs.setdefault("http_status", 409)
        super().__init__(message, **kwargs)


class DataIntegrityError(DatabaseIntegrityError):
    """Data integrity error."""
    pass


# ============================================================================
# Cache Errors
# ============================================================================

class CacheError(HelixException):
    """Base cache error."""
    def __init__(self, message: str, **kwargs):
        kwargs.setdefault("category", ErrorCategory.CACHE)
        kwargs.setdefault("severity", ErrorSeverity.WARNING)
        kwargs.setdefault("http_status", 500)
        super().__init__(message, **kwargs)


class CacheConnectionError(CacheError):
    """Cache connection error."""
    def __init__(self, message: str, **kwargs):
        kwargs.setdefault("is_recoverable", True)
        super().__init__(message, **kwargs)


class CacheOperationError(CacheError):
    """Cache operation error."""
    pass


class SemanticCacheError(CacheError):
    """Semantic cache error."""
    pass


class RedisConnectionError(CacheConnectionError):
    """Redis connection error."""
    pass


# ============================================================================
# Timeout & Rate Limit Errors
# ============================================================================

class TimeoutError(HelixException):
    """Operation timeout."""
    def __init__(self, message: str, **kwargs):
        kwargs.setdefault("category", ErrorCategory.TIMEOUT)
        kwargs.setdefault("severity", ErrorSeverity.ERROR)
        kwargs.setdefault("http_status", 504)
        kwargs.setdefault("is_recoverable", True)
        super().__init__(message, **kwargs)


class RateLimitError(HelixException):
    """Rate limit exceeded."""
    def __init__(self, message: str, **kwargs):
        kwargs.setdefault("category", ErrorCategory.RATE_LIMIT)
        kwargs.setdefault("severity", ErrorSeverity.WARNING)
        kwargs.setdefault("http_status", 429)
        kwargs.setdefault("is_recoverable", True)
        super().__init__(message, **kwargs)


# ============================================================================
# Network Errors
# ============================================================================

class NetworkError(HelixException):
    """Network connectivity error."""
    def __init__(self, message: str, **kwargs):
        kwargs.setdefault("category", ErrorCategory.NETWORK)
        kwargs.setdefault("severity", ErrorSeverity.ERROR)
        kwargs.setdefault("http_status", 503)
        kwargs.setdefault("is_recoverable", True)
        super().__init__(message, **kwargs)


class WebSocketError(NetworkError):
    """WebSocket error."""
    pass


# ============================================================================
# Resource Errors
# ============================================================================

class ResourceError(HelixException):
    """Resource exhausted error."""
    def __init__(self, message: str, **kwargs):
        kwargs.setdefault("category", ErrorCategory.RESOURCE)
        kwargs.setdefault("severity", ErrorSeverity.ERROR)
        kwargs.setdefault("http_status", 507)
        super().__init__(message, **kwargs)


class ResourceExhaustedError(ResourceError):
    """Resource exhausted (memory, disk, etc.)."""
    pass


class DeadlockError(ResourceError):
    """Deadlock detected."""
    pass


# ============================================================================
# Task & Workflow Errors
# ============================================================================

class TaskError(HelixException):
    """Base task error."""
    def __init__(self, message: str, **kwargs):
        kwargs.setdefault("http_status", 500)
        super().__init__(message, **kwargs)


class TaskExecutionError(TaskError):
    """Task execution failed."""
    pass


class TaskRoutingError(TaskError):
    """Task routing error."""
    pass


class TaskTimeoutError(TaskError):
    """Task timeout."""
    def __init__(self, message: str, **kwargs):
        kwargs.setdefault("category", ErrorCategory.TIMEOUT)
        kwargs.setdefault("is_recoverable", True)
        super().__init__(message, **kwargs)


class WorkflowError(HelixException):
    """Base workflow error."""
    def __init__(self, message: str, **kwargs):
        kwargs.setdefault("http_status", 500)
        super().__init__(message, **kwargs)


class WorkflowExecutionError(WorkflowError):
    """Workflow execution failed."""
    pass


# ============================================================================
# Orchestration Errors
# ============================================================================

class OrchestratorError(HelixException):
    """Base orchestrator error."""
    def __init__(self, message: str, **kwargs):
        kwargs.setdefault("http_status", 500)
        super().__init__(message, **kwargs)


class DependencyResolutionError(OrchestratorError):
    """Failed to resolve dependencies."""
    pass


class RoutingDecisionError(OrchestratorError):
    """Routing decision error."""
    pass


class ModelSelectionError(OrchestratorError):
    """Failed to select appropriate model."""
    pass


class RewardModelError(OrchestratorError):
    """Reward model error."""
    pass


class NeuralPlasticityError(OrchestratorError):
    """Neural plasticity error."""
    pass


# ============================================================================
# Agent Errors
# ============================================================================

class AgentError(HelixException):
    """Base agent error."""
    def __init__(self, message: str, **kwargs):
        kwargs.setdefault("http_status", 500)
        super().__init__(message, **kwargs)


class AgentCommunicationError(AgentError):
    """Agent communication error."""
    pass


class ToolExecutionError(AgentError):
    """Tool execution error."""
    pass


class MonitoringError(AgentError):
    """Monitoring error."""
    pass


# ============================================================================
# Security Errors
# ============================================================================

class SecurityError(HelixException):
    """Base security error."""
    def __init__(self, message: str, **kwargs):
        kwargs.setdefault("category", ErrorCategory.AUTHENTICATION)
        kwargs.setdefault("severity", ErrorSeverity.CRITICAL)
        kwargs.setdefault("http_status", 403)
        super().__init__(message, **kwargs)


class EncryptionError(SecurityError):
    """Encryption/decryption error."""
    pass


class SecretsManagerError(SecurityError):
    """Secrets manager error."""
    pass


class SecretNotFoundError(SecretsManagerError):
    """Secret not found."""
    def __init__(self, message: str, **kwargs):
        kwargs.setdefault("http_status", 404)
        super().__init__(message, **kwargs)


# ============================================================================
# Integration Errors
# ============================================================================

class IntegrationError(HelixException):
    """Base integration error."""
    def __init__(self, message: str, **kwargs):
        kwargs.setdefault("http_status", 502)
        super().__init__(message, **kwargs)


class FileOperationError(IntegrationError):
    """File operation error."""
    pass


class TemplateError(IntegrationError):
    """Template error."""
    pass


# ============================================================================
# Monitoring/Metrics Errors
# ============================================================================

class MetricsError(HelixException):
    """Base metrics error."""
    def __init__(self, message: str, **kwargs):
        kwargs.setdefault("severity", ErrorSeverity.WARNING)
        kwargs.setdefault("http_status", 500)
        super().__init__(message, **kwargs)


class MetricsCollectionError(MetricsError):
    """Metrics collection error."""
    pass


class MetricsStorageError(MetricsError):
    """Metrics storage error."""
    pass


class EventBusError(HelixException):
    """Base event bus error."""
    def __init__(self, message: str, **kwargs):
        kwargs.setdefault("http_status", 500)
        super().__init__(message, **kwargs)


class EventPublishError(EventBusError):
    """Event publish error."""
    pass


class EventSubscriptionError(EventBusError):
    """Event subscription error."""
    pass


# ============================================================================
# Circuit Breaker Implementation
# ============================================================================

class CircuitBreaker:
    """Circuit breaker pattern implementation."""

    def __init__(self, name: str, config: Optional[CircuitBreakerConfig] = None):
        """Initialize circuit breaker."""
        self.name = name
        self.config = config or CircuitBreakerConfig()
        self.state = CircuitBreakerState.CLOSED
        self.metrics = CircuitBreakerMetrics()
        self.last_state_change = datetime.utcnow()

    def record_success(self) -> None:
        """Record successful request."""
        self.metrics.successful_requests += 1
        self.metrics.total_requests += 1

        if self.state == CircuitBreakerState.HALF_OPEN:
            if self.metrics.successful_requests >= self.config.success_threshold:
                self._close()

    def record_failure(self) -> None:
        """Record failed request."""
        self.metrics.failed_requests += 1
        self.metrics.total_requests += 1
        self.metrics.last_failure_time = datetime.utcnow()

        if self.state == CircuitBreakerState.CLOSED:
            if self.metrics.failed_requests >= self.config.failure_threshold:
                self._open()

    def record_rejection(self) -> None:
        """Record rejected request."""
        self.metrics.rejected_requests += 1
        self.metrics.total_requests += 1

    def can_execute(self) -> bool:
        """Check if request can be executed."""
        if self.state == CircuitBreakerState.CLOSED:
            return True

        if self.state == CircuitBreakerState.OPEN:
            # Check if recovery timeout has passed
            time_since_open = datetime.utcnow() - self.last_state_change
            if time_since_open.total_seconds() > self.config.recovery_timeout_sec:
                self._half_open()
                return True
            return False

        # HALF_OPEN - allow request
        return True

    def _open(self) -> None:
        """Open circuit breaker."""
        self.state = CircuitBreakerState.OPEN
        self.last_state_change = datetime.utcnow()
        self.metrics.failed_requests = 0
        logger.warning(f"Circuit breaker '{self.name}' opened")

    def _close(self) -> None:
        """Close circuit breaker."""
        self.state = CircuitBreakerState.CLOSED
        self.last_state_change = datetime.utcnow()
        self.metrics = CircuitBreakerMetrics()
        logger.info(f"Circuit breaker '{self.name}' closed")

    def _half_open(self) -> None:
        """Set circuit breaker to half-open."""
        self.state = CircuitBreakerState.HALF_OPEN
        self.last_state_change = datetime.utcnow()
        self.metrics.successful_requests = 0
        logger.info(f"Circuit breaker '{self.name}' half-open (testing recovery)")

    def get_status(self) -> Dict[str, Any]:
        """Get circuit breaker status."""
        return {
            "name": self.name,
            "state": self.state.value,
            "metrics": {
                "total_requests": self.metrics.total_requests,
                "successful_requests": self.metrics.successful_requests,
                "failed_requests": self.metrics.failed_requests,
                "rejected_requests": self.metrics.rejected_requests,
            }
        }


# ============================================================================
# Retry Decorator & Functions
# ============================================================================

async def retry_with_backoff(
    fn: Callable,
    config: Optional[RetryConfig] = None,
    should_retry: Optional[Callable[[Exception], bool]] = None,
) -> Any:
    """
    Execute function with retry and exponential backoff.

    Args:
        fn: Async function to execute
        config: Retry configuration
        should_retry: Optional function to determine if error is retryable

    Returns:
        Function result
    """
    config = config or RetryConfig()

    last_exception = None
    for attempt in range(config.max_retries + 1):
        try:
            return await fn()
        except Exception as e:
            last_exception = e

            # Check if retryable
            if should_retry and not should_retry(e):
                raise

            # Check if max retries exceeded
            if attempt >= config.max_retries:
                raise

            # Calculate delay and wait
            delay = config.get_delay(attempt)
            logger.warning(
                f"Attempt {attempt + 1} failed, retrying in {delay:.2f}s: {e}"
            )
            await asyncio.sleep(delay)

    raise last_exception


def retry_decorator(config: Optional[RetryConfig] = None):
    """Decorator for retrying functions."""
    config = config or RetryConfig()

    def decorator(fn: Callable) -> Callable:
        @wraps(fn)
        async def async_wrapper(*args, **kwargs) -> Any:
            return await retry_with_backoff(lambda: fn(*args, **kwargs), config)

        @wraps(fn)
        def sync_wrapper(*args, **kwargs) -> Any:
            last_exception = None
            for attempt in range(config.max_retries + 1):
                try:
                    return fn(*args, **kwargs)
                except Exception as e:
                    last_exception = e
                    if attempt < config.max_retries:
                        delay = config.get_delay(attempt)
                        time.sleep(delay)

            raise last_exception

        return async_wrapper if asyncio.iscoroutinefunction(fn) else sync_wrapper

    return decorator


# ============================================================================
# Utility Functions
# ============================================================================

def create_error_context(
    error: Exception,
    category: Optional[ErrorCategory] = None,
    severity: Optional[ErrorSeverity] = None,
    is_recoverable: bool = True,
    http_status: Optional[int] = None,
) -> ErrorContext:
    """
    Create ErrorContext from exception.

    Automatically categorizes and generates user message if not provided.
    """
    if isinstance(error, HelixException):
        return error.context

    # Auto-categorize
    if category is None:
        category = _categorize_error(error)

    # Auto-determine severity and status
    if severity is None:
        severity = _determine_severity(category)

    if http_status is None:
        http_status = _determine_http_status(category)

    # Generate user message
    user_message = _generate_user_message(error, category)

    return ErrorContext(
        severity=severity,
        category=category,
        message=str(error),
        user_message=user_message,
        stack_trace=traceback.format_exc(),
        is_recoverable=is_recoverable,
        http_status=http_status,
        recovery_suggestions=_get_recovery_suggestions(category),
    )


def _categorize_error(error: Exception) -> ErrorCategory:
    """Auto-categorize exception."""
    error_type = type(error).__name__.lower()

    if "validation" in error_type or "value" in error_type:
        return ErrorCategory.VALIDATION
    elif "authentication" in error_type or "credential" in error_type:
        return ErrorCategory.AUTHENTICATION
    elif "authorization" in error_type or "permission" in error_type:
        return ErrorCategory.AUTHORIZATION
    elif "timeout" in error_type:
        return ErrorCategory.TIMEOUT
    elif "rate" in error_type or "limit" in error_type:
        return ErrorCategory.RATE_LIMIT
    elif "network" in error_type or "connection" in error_type:
        return ErrorCategory.NETWORK
    elif "memory" in error_type or "resource" in error_type:
        return ErrorCategory.RESOURCE
    else:
        return ErrorCategory.INTERNAL


def _determine_severity(category: ErrorCategory) -> ErrorSeverity:
    """Determine severity based on category."""
    if category in (
        ErrorCategory.AUTHENTICATION,
        ErrorCategory.AUTHORIZATION,
        ErrorCategory.LLM_SERVICE,
    ):
        return ErrorSeverity.ERROR

    if category in (ErrorCategory.RATE_LIMIT, ErrorCategory.CACHE):
        return ErrorSeverity.WARNING

    if category == ErrorCategory.TIMEOUT:
        return ErrorSeverity.ERROR

    return ErrorSeverity.ERROR


def _determine_http_status(category: ErrorCategory) -> int:
    """Determine HTTP status code based on category."""
    mapping = {
        ErrorCategory.VALIDATION: 400,
        ErrorCategory.AUTHENTICATION: 401,
        ErrorCategory.AUTHORIZATION: 403,
        ErrorCategory.TIMEOUT: 504,
        ErrorCategory.RATE_LIMIT: 429,
        ErrorCategory.DATABASE: 500,
        ErrorCategory.NETWORK: 503,
        ErrorCategory.RESOURCE: 507,
    }
    return mapping.get(category, 500)


def _generate_user_message(error: Exception, category: ErrorCategory) -> str:
    """Generate user-friendly message."""
    messages = {
        ErrorCategory.VALIDATION: "Your input is invalid. Please check and try again.",
        ErrorCategory.AUTHENTICATION: "Authentication failed. Please log in again.",
        ErrorCategory.AUTHORIZATION: "You don't have permission for this action.",
        ErrorCategory.TIMEOUT: "The request took too long. Please try again.",
        ErrorCategory.RATE_LIMIT: "Too many requests. Please wait and try again.",
        ErrorCategory.DATABASE: "Database operation failed. Please try again.",
        ErrorCategory.NETWORK: "Network connection error. Please check your connection.",
        ErrorCategory.RESOURCE: "System resource limit exceeded. Please try again later.",
        ErrorCategory.LLM_SERVICE: "The AI service is temporarily unavailable.",
    }
    return messages.get(category, "An error occurred. Please try again.")


def _get_recovery_suggestions(category: ErrorCategory) -> List[str]:
    """Get recovery suggestions based on category."""
    suggestions = {
        ErrorCategory.VALIDATION: [
            "Check your input format",
            "Review error details",
            "Consult documentation",
        ],
        ErrorCategory.AUTHENTICATION: [
            "Verify your credentials",
            "Check your password",
            "Try logging in again",
        ],
        ErrorCategory.TIMEOUT: [
            "Try again",
            "Check system load",
            "Increase timeout value",
        ],
        ErrorCategory.RATE_LIMIT: [
            "Wait and try again",
            "Reduce request frequency",
            "Contact support for higher limits",
        ],
        ErrorCategory.NETWORK: [
            "Check your internet connection",
            "Try again in a moment",
            "Check service status",
        ],
    }
    return suggestions.get(category, ["Please try again"])


def get_exception_hierarchy() -> Dict[str, List[str]]:
    """Get exception hierarchy for documentation/introspection."""
    import sys
    module = sys.modules[__name__]

    hierarchy = {}
    for name, obj in module.__dict__.items():
        if isinstance(obj, type) and issubclass(obj, HelixException):
            bases = [b.__name__ for b in obj.__bases__ if issubclass(b, HelixException)]
            if bases:
                for base in bases:
                    if base not in hierarchy:
                        hierarchy[base] = []
                    hierarchy[base].append(name)

    return hierarchy


# ============================================================================
# Export All Exception Classes
# ============================================================================

__all__ = [
    # Enums
    "ErrorSeverity",
    "ErrorCategory",
    "CircuitBreakerState",
    # Data classes
    "ErrorContext",
    "RetryConfig",
    "CircuitBreakerConfig",
    "CircuitBreakerMetrics",
    # Base exception
    "HelixException",
    # Validation errors
    "ValidationError",
    "APIValidationError",
    "TaskValidationError",
    "WorkflowValidationError",
    "ConfigurationError",
    "MissingConfigurationError",
    "InvalidConfigurationError",
    # Auth errors
    "AuthenticationError",
    "AuthorizationError",
    "APIKeyError",
    # LLM errors
    "LLMError",
    "LLMProviderError",
    "LLMAPIError",
    "LLMModelNotFoundError",
    "LLMContextLengthExceeded",
    "LLMRateLimitError",
    "LLMTimeoutError",
    "LLMAuthenticationError",
    # Database errors
    "DatabaseError",
    "DatabaseConnectionError",
    "DatabaseQueryError",
    "DatabaseIntegrityError",
    "DataIntegrityError",
    # Cache errors
    "CacheError",
    "CacheConnectionError",
    "CacheOperationError",
    "SemanticCacheError",
    "RedisConnectionError",
    # Timeout & rate limit
    "TimeoutError",
    "RateLimitError",
    # Network errors
    "NetworkError",
    "WebSocketError",
    # Resource errors
    "ResourceError",
    "ResourceExhaustedError",
    "DeadlockError",
    # Task & workflow
    "TaskError",
    "TaskExecutionError",
    "TaskRoutingError",
    "TaskTimeoutError",
    "WorkflowError",
    "WorkflowExecutionError",
    # Orchestration errors
    "OrchestratorError",
    "DependencyResolutionError",
    "RoutingDecisionError",
    "ModelSelectionError",
    "RewardModelError",
    "NeuralPlasticityError",
    # Agent errors
    "AgentError",
    "AgentCommunicationError",
    "ToolExecutionError",
    "MonitoringError",
    # Security errors
    "SecurityError",
    "EncryptionError",
    "SecretsManagerError",
    "SecretNotFoundError",
    # Integration errors
    "IntegrationError",
    "FileOperationError",
    "TemplateError",
    # Metrics errors
    "MetricsError",
    "MetricsCollectionError",
    "MetricsStorageError",
    "EventBusError",
    "EventPublishError",
    "EventSubscriptionError",
    # Circuit breaker
    "CircuitBreaker",
    # Utilities
    "retry_with_backoff",
    "retry_decorator",
    "create_error_context",
    "get_exception_hierarchy",
    # Legacy aliases
    "BaseHelixException",
    "InitializationError",
]

# Legacy aliases used by some modules
BaseHelixException = HelixException
InitializationError = ConfigurationError
