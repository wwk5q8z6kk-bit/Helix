#!/usr/bin/env python3
"""
Base Provider Abstraction for Multi-Provider LLM Support
Defines the interface that all LLM providers must implement
"""

from abc import ABC, abstractmethod
from dataclasses import dataclass, field
from typing import Dict, List, Optional, Any, Tuple
from datetime import datetime
from enum import Enum


class ProviderType(Enum):
    """Supported provider types"""
    ANTHROPIC = "anthropic"
    OPENAI = "openai"
    GOOGLE = "google"
    COHERE = "cohere"
    AZURE = "azure"
    OLLAMA = "ollama"
    DEEPSEEK = "deepseek"
    XAI = "xai"
    PERPLEXITY = "perplexity"
    MISTRAL = "mistral"
    MINIMAX = "minimax"


@dataclass
class LLMRequest:
    """Standardized LLM request"""
    prompt: str
    model: Optional[str] = None
    temperature: float = 0.7
    max_tokens: int = 2048
    top_p: float = 1.0
    frequency_penalty: float = 0.0
    presence_penalty: float = 0.0
    stop_sequences: List[str] = field(default_factory=list)
    metadata: Dict[str, Any] = field(default_factory=dict)


@dataclass
class LLMResponse:
    """Standardized LLM response"""
    content: str
    model: str
    provider: str
    input_tokens: int
    output_tokens: int
    total_tokens: int
    cost_usd: float
    latency_ms: float
    finish_reason: str
    timestamp: datetime = field(default_factory=datetime.now)
    metadata: Dict[str, Any] = field(default_factory=dict)

    def to_dict(self) -> Dict[str, Any]:
        """Convert to dictionary"""
        return {
            "content": self.content,
            "model": self.model,
            "provider": self.provider,
            "input_tokens": self.input_tokens,
            "output_tokens": self.output_tokens,
            "total_tokens": self.total_tokens,
            "cost_usd": self.cost_usd,
            "latency_ms": self.latency_ms,
            "finish_reason": self.finish_reason,
            "timestamp": self.timestamp.isoformat(),
            "metadata": self.metadata
        }


class LLMProvider(ABC):
    """
    Abstract base class for LLM providers
    All providers must implement this interface
    """

    def __init__(self, api_key: Optional[str] = None, **config):
        """
        Initialize provider

        Args:
            api_key: API key for the provider (if required)
            **config: Additional provider-specific configuration
        """
        self.api_key = api_key
        self.config = config
        self.provider_type = self._get_provider_type()

    @abstractmethod
    def _get_provider_type(self) -> ProviderType:
        """Return the provider type"""
        pass

    @abstractmethod
    async def generate(self, request: LLMRequest) -> LLMResponse:
        """
        Generate a response from the LLM

        Args:
            request: Standardized LLM request

        Returns:
            Standardized LLM response

        Raises:
            ProviderError: If generation fails
        """
        pass

    @abstractmethod
    def get_available_models(self) -> List[str]:
        """
        Get list of available models for this provider

        Returns:
            List of model identifiers
        """
        pass

    @abstractmethod
    def get_cost_per_token(self, model: str) -> Tuple[float, float]:
        """
        Get cost per token for a specific model

        Args:
            model: Model identifier

        Returns:
            Tuple of (input_cost_per_million, output_cost_per_million)
        """
        pass

    def calculate_cost(
        self,
        model: str,
        input_tokens: int,
        output_tokens: int
    ) -> float:
        """
        Calculate cost for a request

        Args:
            model: Model identifier
            input_tokens: Number of input tokens
            output_tokens: Number of output tokens

        Returns:
            Total cost in USD
        """
        input_cost, output_cost = self.get_cost_per_token(model)
        total_cost = (
            (input_tokens * input_cost / 1_000_000) +
            (output_tokens * output_cost / 1_000_000)
        )
        return round(total_cost, 6)

    def validate_config(self) -> bool:
        """
        Validate provider configuration

        Returns:
            True if configuration is valid

        Raises:
            ConfigurationError: If configuration is invalid
        """
        if self.requires_api_key() and not self.api_key:
            raise ConfigurationError(
                f"{self.provider_type.value} requires an API key"
            )
        return True

    def requires_api_key(self) -> bool:
        """
        Check if provider requires API key

        Returns:
            True if API key is required
        """
        # Most providers require API keys, override for local providers
        return True

    def get_provider_info(self) -> Dict[str, Any]:
        """
        Get provider information

        Returns:
            Dictionary with provider metadata
        """
        return {
            "type": self.provider_type.value,
            "name": self.__class__.__name__,
            "requires_api_key": self.requires_api_key(),
            "available_models": self.get_available_models(),
            "configuration": {
                k: v for k, v in self.config.items()
                if k not in ['api_key', 'password', 'secret']
            }
        }


class ProviderError(Exception):
    """Base exception for provider errors"""
    pass


class ConfigurationError(ProviderError):
    """Configuration error"""
    pass


class APIError(ProviderError):
    """API-related error"""
    pass


class RateLimitError(ProviderError):
    """Rate limit exceeded"""
    pass


class ModelNotFoundError(ProviderError):
    """Model not available"""
    pass
