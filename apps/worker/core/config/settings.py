"""
Configuration management using Pydantic Settings.
Follows 12-factor app methodology with environment-based configuration.
"""
import logging
import os
from functools import lru_cache
from pathlib import Path
from typing import Optional, List

from pydantic import Field, field_validator

try:
    from pydantic_settings import BaseSettings, SettingsConfigDict
except ImportError:  # pragma: no cover - fallback for minimal installations
    from pydantic import BaseModel as _BaseModel

    class SettingsConfigDict(dict):
        """Fallback stub when pydantic-settings is unavailable."""
        pass

    class BaseSettings(_BaseModel):  # type: ignore
        """Lightweight fallback BaseSettings without env loading."""
        model_config = {}

    logging.getLogger(__name__).warning(
        "pydantic-settings not installed; using simplified BaseSettings fallback."
    )


class Settings(BaseSettings):
    """Application settings with validation and environment variable support."""
    
    model_config = SettingsConfigDict(
        env_file=".env",
        env_file_encoding="utf-8",
        case_sensitive=False,
        extra="ignore"
    )
    
    # Application
    app_name: str = Field(default="Helix", description="Application name")
    app_version: str = Field(default="1.0.0", description="Application version")
    environment: str = Field(default="development", description="Environment: development, staging, production")
    debug: bool = Field(default=False, description="Debug mode")
    
    # API
    api_host: str = Field(default="127.0.0.1", description="API host")
    api_port: int = Field(default=8000, ge=1024, le=65535, description="API port")
    api_workers: int = Field(default=4, ge=1, le=32, description="Number of worker processes")
    api_reload: bool = Field(default=False, description="Auto-reload on code changes")
    
    # CORS
    cors_origins: List[str] = Field(
        default=["http://localhost:8000", "http://127.0.0.1:8000"],
        description="Allowed CORS origins"
    )
    cors_credentials: bool = Field(default=True, description="Allow credentials")
    cors_methods: List[str] = Field(default=["*"], description="Allowed HTTP methods")
    cors_headers: List[str] = Field(default=["*"], description="Allowed HTTP headers")
    
    # Security
    secret_key: str = Field(default="", description="Secret key for JWT and encryption")
    access_token_expire_minutes: int = Field(default=30, ge=1, description="JWT expiration time")

    # Rate Limiting
    rate_limit_enabled: bool = Field(default=True, description="Enable rate limiting")
    rate_limit_storage_uri: str = Field(default="memory://", description="Rate limit storage (memory://, redis://)")
    rate_limit_default: str = Field(default="100/minute", description="Default rate limit")
    rate_limit_health: str = Field(default="1000/minute", description="Health endpoint rate limit")
    rate_limit_ai: str = Field(default="20/minute", description="AI endpoint rate limit")
    rate_limit_chat: str = Field(default="30/minute", description="Chat endpoint rate limit")
    rate_limit_code_gen: str = Field(default="20/minute", description="Code generation rate limit")
    rate_limit_code_review: str = Field(default="30/minute", description="Code review rate limit")
    rate_limit_file_ops: str = Field(default="50/minute", description="File operations rate limit")
    
    # Ollama
    ollama_url: str = Field(default="http://localhost:11434", description="Ollama API URL")
    ollama_model: str = Field(default="helix-general", description="Default Ollama model")
    ollama_timeout: int = Field(default=60, ge=1, description="Ollama request timeout")
    ollama_max_tokens: int = Field(default=512, ge=1, description="Max tokens for generation")
    ollama_temperature: float = Field(default=0.7, ge=0.0, le=2.0, description="Temperature for generation")
    
    # OpenAI (fallback)
    openai_api_key: Optional[str] = Field(default=None, description="OpenAI API key")
    openai_model: str = Field(default="gpt-4o-mini", description="OpenAI model")
    openai_org_id: Optional[str] = Field(default=None, description="OpenAI organization ID")
    openai_timeout: int = Field(default=60, ge=1, description="OpenAI request timeout")
    
    # Database
    db_path: str = Field(default=".helix/helix.db", description="SQLite database path")
    db_pool_size: int = Field(default=5, ge=1, description="Database connection pool size")
    db_max_overflow: int = Field(default=10, ge=0, description="Max overflow connections")
    db_echo: bool = Field(default=False, description="Echo SQL queries")
    
    # Memory Store
    memory_db_path: str = Field(default=".helix/project_memory.db", description="Project memory database path")
    memory_max_size_mb: int = Field(default=100, ge=1, description="Max memory store size in MB")
    
    # Logging
    log_level: str = Field(default="INFO", description="Logging level")
    log_format: str = Field(default="json", description="Log format: json or text")
    log_file: Optional[str] = Field(default=None, description="Log file path")
    log_rotation: str = Field(default="100 MB", description="Log rotation size")
    log_retention: str = Field(default="30 days", description="Log retention period")
    
    # Resource Limits
    max_memory_mb: int = Field(default=1024, ge=128, description="Max memory usage in MB")
    max_cpu_percent: float = Field(default=80.0, ge=1.0, le=100.0, description="Max CPU usage percentage")
    max_execution_time: int = Field(default=300, ge=1, description="Max execution time in seconds")
    
    # Circuit Breaker
    circuit_breaker_enabled: bool = Field(default=True, description="Enable circuit breakers")
    circuit_breaker_failure_threshold: int = Field(default=5, ge=1, description="Failures before opening")
    circuit_breaker_timeout: int = Field(default=60, ge=1, description="Timeout before retry")
    
    # Feature Flags
    enable_streaming: bool = Field(default=True, description="Enable streaming responses")
    enable_training: bool = Field(default=False, description="Enable model training")
    enable_caching: bool = Field(default=True, description="Enable response caching")

    # Performance Monitoring
    metrics_retention_minutes: int = Field(default=60, ge=1, description="Metrics retention in minutes")
    health_check_interval: int = Field(default=30, ge=1, description="Health check interval in seconds")

    # Security
    jwt_secret: str = Field(default="change-me", description="JWT secret key")
    jwt_algorithm: str = Field(default="HS256", description="JWT algorithm")
    jwt_expiration: int = Field(default=3600, ge=1, description="JWT expiration in seconds")
    encryption_key: Optional[str] = Field(default=None, description="Encryption key (base64)")
    allowed_hosts: List[str] = Field(default=["localhost", "127.0.0.1"], description="Allowed hosts")

    # File Storage
    data_dir: str = Field(default="./data", description="Data directory")
    logs_dir: str = Field(default="./logs", description="Logs directory")
    cache_dir: str = Field(default="./.cache", description="Cache directory")
    models_dir: str = Field(default="./models", description="Models directory")

    # macOS Integration
    macos_enabled: bool = Field(default=True, description="Enable macOS integration")
    macos_notifications: bool = Field(default=True, description="Enable macOS notifications")
    macos_keychain: bool = Field(default=True, description="Enable macOS Keychain")

    # Cloud Routing
    cloud_routing_enabled: bool = Field(default=False, description="Enable cloud routing")
    cloud_routing_percentage: int = Field(default=30, ge=0, le=100, description="% of requests to route to cloud")
    cloud_provider: str = Field(default="openai", description="Cloud provider: openai, anthropic, deepseek")

    # Workers
    worker_execution_replicas: int = Field(default=3, ge=1, description="Number of execution workers")
    max_concurrent_requests: int = Field(default=100, ge=1, description="Max concurrent requests")
    request_timeout: int = Field(default=300, ge=1, description="Request timeout in seconds")
    graceful_shutdown_timeout: int = Field(default=30, ge=1, description="Graceful shutdown timeout")

    @field_validator("environment")
    @classmethod
    def validate_environment(cls, v: str) -> str:
        """Validate environment is one of allowed values."""
        allowed = ["development", "staging", "production"]
        if v not in allowed:
            raise ValueError(f"Environment must be one of {allowed}")
        return v
    
    @field_validator("log_level")
    @classmethod
    def validate_log_level(cls, v: str) -> str:
        """Validate log level is valid."""
        allowed = ["DEBUG", "INFO", "WARNING", "ERROR", "CRITICAL"]
        v_upper = v.upper()
        if v_upper not in allowed:
            raise ValueError(f"Log level must be one of {allowed}")
        return v_upper
    
    @field_validator("log_format")
    @classmethod
    def validate_log_format(cls, v: str) -> str:
        """Validate log format is valid."""
        allowed = ["json", "text"]
        if v not in allowed:
            raise ValueError(f"Log format must be one of {allowed}")
        return v

    @field_validator("cloud_provider")
    @classmethod
    def validate_cloud_provider(cls, v: str) -> str:
        """Validate cloud provider."""
        allowed = ["openai", "anthropic", "deepseek"]
        if v not in allowed:
            raise ValueError(f"Cloud provider must be one of {allowed}")
        return v

    @field_validator("secret_key", "jwt_secret")
    @classmethod
    def validate_secrets_in_production(cls, v: str, info) -> str:
        """Ensure secrets are changed in production."""
        # Get environment from values if available
        environment = info.data.get("environment", "development")

        if environment == "production":
            if not v or "change" in v.lower() or "dev" in v.lower():
                raise ValueError(
                    f"Secret must be changed in production! Use: openssl rand -hex 32"
                )
        return v

    @property
    def is_production(self) -> bool:
        """Check if running in production environment."""
        return self.environment == "production"
    
    @property
    def is_development(self) -> bool:
        """Check if running in development environment."""
        return self.environment == "development"

    def get_log_level(self) -> int:
        """Get logging level as integer."""
        return getattr(logging, self.log_level)

    def ensure_directories(self) -> None:
        """Create required directories if they don't exist."""
        for dir_path in [self.data_dir, self.logs_dir, self.cache_dir, self.models_dir]:
            Path(dir_path).mkdir(parents=True, exist_ok=True)


@lru_cache(maxsize=1)
def get_settings() -> Settings:
    """
    Get cached settings instance.
    
    Returns:
        Settings: Application settings
    """
    return Settings()


# Export for convenience
settings = get_settings()
