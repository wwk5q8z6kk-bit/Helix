"""Interface for task orchestration and agent coordination.

Defines the contract for orchestrating development workflows and coordinating multiple agents.
"""

from typing import Protocol, Dict, Any, List, Optional
from dataclasses import dataclass


@dataclass
class Task:
    """Represents a development task."""
    task_id: str
    task_type: str
    description: str
    metadata: Dict[str, Any]


@dataclass
class Result:
    """Result of task execution."""
    task_id: str
    status: str  # 'success', 'failed', 'in_progress'
    output: Any
    error: Optional[str] = None
    metadata: Optional[Dict[str, Any]] = None


class IOrchestrator(Protocol):
    """Interface for task orchestration and agent coordination.

    Responsibilities:
    - Route tasks to appropriate agents
    - Manage task state and lifecycle
    - Coordinate multi-agent workflows
    - Track execution and results
    """

    async def execute_task(
        self,
        task: Task,
        agent_id: Optional[str] = None
    ) -> Result:
        """Execute a task through orchestration system.

        Args:
            task: Task to execute
            agent_id: Optional specific agent to use

        Returns:
            Result with output and status
        """
        ...

    async def execute_parallel(
        self,
        tasks: List[Task]
    ) -> List[Result]:
        """Execute multiple tasks in parallel.

        Args:
            tasks: List of tasks to execute

        Returns:
            List of results in same order as input
        """
        ...

    def register_agent(
        self,
        agent_id: str,
        agent_capabilities: Dict[str, Any]
    ) -> None:
        """Register an agent with orchestrator.

        Args:
            agent_id: Unique agent identifier
            agent_capabilities: What this agent can do
        """
        ...

    async def get_system_status(self) -> Dict[str, Any]:
        """Get overall system status and metrics.

        Returns:
            Status dict with agent states, task queue, etc.
        """
        ...
