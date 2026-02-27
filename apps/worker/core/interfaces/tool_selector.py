"""Interface for intelligent tool and model selection.

Learns which tools/models work best for different tasks and continuously improves.
"""

from typing import Protocol, Dict, Any, Optional
from dataclasses import dataclass


@dataclass
class ToolSelectionResult:
    """Result of tool selection."""
    tool_name: str
    confidence: float
    reasoning: str


class IToolSelector(Protocol):
    """Interface for intelligent tool/model selection.

    Learns which tools/models work best for different tasks
    and continuously improves selection accuracy.
    """

    def select_tool(
        self,
        task_requirements: Dict[str, Any]
    ) -> ToolSelectionResult:
        """Select best tool for task requirements.

        Args:
            task_requirements: Dict describing what's needed

        Returns:
            Selection result with confidence score
        """
        ...

    def register_tool(
        self,
        tool_name: str,
        capabilities: Dict[str, Any],
        metadata: Optional[Dict[str, Any]] = None
    ) -> None:
        """Register a tool available for selection.

        Args:
            tool_name: Unique tool identifier
            capabilities: What this tool can do
            metadata: Optional additional metadata
        """
        ...

    async def provide_feedback(
        self,
        tool_name: str,
        task_result: Dict[str, Any],
        success: bool
    ) -> None:
        """Provide feedback on tool selection.

        Used to train selector to make better choices.

        Args:
            tool_name: Tool that was selected
            task_result: Result of using this tool
            success: Whether selection was good
        """
        ...
