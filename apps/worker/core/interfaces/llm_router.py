"""Interface for LLM routing and generation.

This interface defines the contract that all LLM routing implementations must follow.
Implementations provide intelligent LLM provider selection and response generation.
"""

from typing import Protocol, Dict, List, Any, Optional, Tuple
from dataclasses import dataclass


@dataclass
class LLMRouterResult:
    """Result of an LLM generation call."""
    provider: str
    response: str
    tokens_used: int
    cost: float
    model: str


class ILLMRouter(Protocol):
    """Interface for LLM routing and generation.

    Matches the concrete ``IntelligentLLMRouter`` API so that the
    orchestrator can call ``select_provider`` / ``call_llm`` through
    the protocol without Pyright complaints.
    """

    async def select_provider(
        self,
        task_type: Any,
        prompt_tokens: int,
        required_quality: float = 8.0,
        speed_critical: bool = False,
        cost_critical: bool = False,
    ) -> Any:
        """Select optimal provider for a task."""
        ...

    async def call_llm(
        self,
        provider: Any,
        messages: List[Dict[str, str]],
        task_type: Any,
        max_tokens: int = 4096,
        temperature: float = 0.7,
    ) -> Tuple[str, Dict[str, Any]]:
        """Call LLM with selected provider.

        Returns:
            Tuple of (response_text, metadata_dict)
        """
        ...

    async def get_router_stats(self) -> Dict[str, Any]:
        """Get router statistics."""
        ...
