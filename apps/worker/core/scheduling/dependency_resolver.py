"""DAG-based dependency resolver for Helix task scheduling.

Standalone module — no dependencies on orchestrator internals.  Pure Python.

Provides:
- Cycle detection (DFS on add_task)
- Topological ordering via Kahn's algorithm (execution waves)
- Failure propagation (BFS cancel of downstream)
- Serialisation for persistence (to_dict / from_dict)
"""

from __future__ import annotations

import logging
from collections import deque
from dataclasses import dataclass
from enum import Enum
from typing import Any, Dict, List, Optional, Set

logger = logging.getLogger(__name__)


# ── Enums / value objects ────────────────────────────────────────────


class TaskState(str, Enum):
    """Lifecycle states tracked by the resolver."""

    PENDING = "pending"  # waiting on unsatisfied deps
    READY = "ready"  # all deps met, eligible for dispatch
    RUNNING = "running"  # currently executing
    COMPLETED = "completed"  # finished successfully (in history)
    FAILED = "failed"  # finished with error (in history)
    CANCELLED = "cancelled"  # cancelled due to upstream failure (in history)


@dataclass(frozen=True)
class TaskNode:
    """Immutable snapshot of a task's position in the graph."""

    task_id: str
    state: TaskState
    dependencies: frozenset[str]
    dependents: frozenset[str]
    priority: int


# ── Exceptions ───────────────────────────────────────────────────────


class CycleDetectedError(Exception):
    """Raised when adding a task would create a dependency cycle."""

    def __init__(self, cycle: List[str]) -> None:
        self.cycle = cycle
        path = " -> ".join(cycle)
        super().__init__(f"Dependency cycle detected: {path}")


# ── Resolver ─────────────────────────────────────────────────────────


class DependencyResolver:
    """DAG-based dependency resolver.

    Tracks forward edges (task → deps it needs), reverse edges (task → tasks
    that need it), states, and a history of completed/failed/cancelled tasks.

    Not thread-safe — designed for a single asyncio event loop where dict
    operations are atomic in CPython.
    """

    def __init__(self) -> None:
        # Forward edges: task_id → set of task_ids it depends ON
        self._dependencies: Dict[str, Set[str]] = {}
        # Reverse edges: task_id → set of task_ids that depend on IT
        self._dependents: Dict[str, Set[str]] = {}
        # Active task states (PENDING / READY / RUNNING only)
        self._states: Dict[str, TaskState] = {}
        # Numeric priority per task (0 = CRITICAL … 3 = LOW)
        self._priorities: Dict[str, int] = {}
        # Completed / failed / cancelled tasks (retained for dep lookups)
        self._history: Dict[str, TaskState] = {}
        # Insertion order for same-priority FIFO tie-breaking
        self._insertion_order: List[str] = []

    # ── Graph mutation ───────────────────────────────────────────────

    def add_task(
        self,
        task_id: str,
        dependencies: Optional[List[str]] = None,
        priority: int = 2,
    ) -> TaskState:
        """Add a task to the dependency graph.

        Returns the initial state assigned (READY or PENDING).

        Raises:
            ValueError: if *task_id* is already in the active graph.
            CycleDetectedError: if adding this task would create a cycle.
        """
        if task_id in self._states:
            raise ValueError(f"Task {task_id!r} already exists in the graph")

        deps = set(dependencies) if dependencies else set()

        # Self-dependency is always a cycle
        if task_id in deps:
            raise CycleDetectedError([task_id, task_id])

        # Auto-resolve deps that are already in history
        resolved = set()
        for dep in deps:
            if dep in self._history:
                hist_state = self._history[dep]
                if hist_state == TaskState.COMPLETED:
                    resolved.add(dep)
                    logger.debug("Dep %s already completed — auto-resolved for %s", dep, task_id)
                elif hist_state in (TaskState.FAILED, TaskState.CANCELLED):
                    # Upstream failed → this task is immediately cancelled
                    self._history[task_id] = TaskState.CANCELLED
                    logger.info("Dep %s failed/cancelled — auto-cancelling %s", dep, task_id)
                    return TaskState.CANCELLED

        # Warn & auto-resolve deps not in the graph or history at all
        unknown = set()
        for dep in deps:
            if dep not in self._states and dep not in self._history:
                unknown.add(dep)
                logger.warning("Dep %s not in graph or history — auto-resolving for %s", dep, task_id)
        resolved |= unknown

        remaining_deps = deps - resolved

        # Cycle detection (before mutating state)
        if remaining_deps:
            self._check_cycle(task_id, remaining_deps)

        # Commit to graph
        self._dependencies[task_id] = remaining_deps
        self._dependents.setdefault(task_id, set())
        self._priorities[task_id] = priority
        self._insertion_order.append(task_id)

        for dep in remaining_deps:
            self._dependents.setdefault(dep, set()).add(task_id)

        if remaining_deps:
            self._states[task_id] = TaskState.PENDING
            return TaskState.PENDING
        else:
            self._states[task_id] = TaskState.READY
            return TaskState.READY

    def remove_task(self, task_id: str) -> Set[str]:
        """Remove a task from the graph.

        Returns the set of task_ids that became READY as a result.
        """
        if task_id not in self._states:
            return set()

        newly_ready: Set[str] = set()

        # Remove as a dependency from its dependents
        for dependent in list(self._dependents.get(task_id, [])):
            self._dependencies[dependent].discard(task_id)
            if not self._dependencies[dependent] and self._states.get(dependent) == TaskState.PENDING:
                self._states[dependent] = TaskState.READY
                newly_ready.add(dependent)

        # Remove from its own dependencies' reverse edges
        for dep in self._dependencies.get(task_id, set()):
            self._dependents.get(dep, set()).discard(task_id)

        # Purge from all maps
        self._dependencies.pop(task_id, None)
        self._dependents.pop(task_id, None)
        self._states.pop(task_id, None)
        self._priorities.pop(task_id, None)
        if task_id in self._insertion_order:
            self._insertion_order.remove(task_id)

        return newly_ready

    # ── State transitions ────────────────────────────────────────────

    def mark_running(self, task_id: str) -> None:
        """Transition READY → RUNNING.

        Raises ValueError if not in READY state.
        """
        state = self._states.get(task_id)
        if state != TaskState.READY:
            raise ValueError(
                f"Cannot mark {task_id!r} as running: current state is {state!r} (expected READY)"
            )
        self._states[task_id] = TaskState.RUNNING

    def mark_completed(self, task_id: str) -> List[str]:
        """Transition RUNNING → COMPLETED.  Move to history.

        Returns list of task_ids that became READY as a result.
        Raises ValueError if not in RUNNING state.
        """
        state = self._states.get(task_id)
        if state != TaskState.RUNNING:
            raise ValueError(
                f"Cannot mark {task_id!r} as completed: current state is {state!r} (expected RUNNING)"
            )

        # Move to history
        self._states.pop(task_id)
        self._history[task_id] = TaskState.COMPLETED

        # Unblock dependents
        newly_ready: List[str] = []
        for dependent in self._dependents.get(task_id, set()):
            self._dependencies[dependent].discard(task_id)
            if not self._dependencies[dependent] and self._states.get(dependent) == TaskState.PENDING:
                self._states[dependent] = TaskState.READY
                newly_ready.append(dependent)

        # Sort by priority then insertion order for determinism
        newly_ready.sort(key=lambda tid: (self._priorities.get(tid, 2), self._insertion_order.index(tid)))

        # Cleanup edges
        self._dependencies.pop(task_id, None)

        return newly_ready

    def mark_failed(self, task_id: str) -> List[str]:
        """Transition RUNNING → FAILED.  BFS-cancel all transitive dependents.

        Returns list of cancelled task_ids.
        """
        state = self._states.get(task_id)
        if state != TaskState.RUNNING:
            raise ValueError(
                f"Cannot mark {task_id!r} as failed: current state is {state!r} (expected RUNNING)"
            )

        # Move to history
        self._states.pop(task_id)
        self._history[task_id] = TaskState.FAILED

        # BFS-propagate cancellation to all transitive dependents
        cancelled: List[str] = []
        queue: deque[str] = deque(self._dependents.get(task_id, set()))
        visited: Set[str] = set()

        while queue:
            dep_id = queue.popleft()
            if dep_id in visited or dep_id not in self._states:
                continue
            visited.add(dep_id)

            current = self._states[dep_id]
            if current in (TaskState.PENDING, TaskState.READY):
                self._states.pop(dep_id)
                self._history[dep_id] = TaskState.CANCELLED
                cancelled.append(dep_id)
                # Continue propagating through this node's dependents
                queue.extend(self._dependents.get(dep_id, set()))

        # Cleanup edges for failed task
        self._dependencies.pop(task_id, None)
        # Cleanup edges for cancelled tasks
        for cid in cancelled:
            self._dependencies.pop(cid, None)

        return cancelled

    # ── Queries ──────────────────────────────────────────────────────

    def get_ready_tasks(self) -> List[str]:
        """Return all READY tasks, sorted by priority (asc) then insertion order."""
        ready = [tid for tid, s in self._states.items() if s == TaskState.READY]
        ready.sort(key=lambda tid: (self._priorities.get(tid, 2), self._insertion_order.index(tid)))
        return ready

    def get_execution_waves(self) -> List[List[str]]:
        """Kahn's algorithm producing parallel execution waves.

        Each wave contains tasks whose dependencies are fully satisfied by
        prior waves.  Within a wave, tasks are sorted by priority.
        """
        # Build in-degree map (only active tasks)
        in_degree: Dict[str, int] = {}
        adj: Dict[str, Set[str]] = {}
        for tid in self._states:
            in_degree[tid] = len(self._dependencies.get(tid, set()))
            adj[tid] = {d for d in self._dependents.get(tid, set()) if d in self._states}

        # Seed with zero-in-degree
        current_wave = [tid for tid, deg in in_degree.items() if deg == 0]
        waves: List[List[str]] = []

        while current_wave:
            current_wave.sort(key=lambda tid: (self._priorities.get(tid, 2), self._insertion_order.index(tid)))
            waves.append(current_wave)

            next_wave: List[str] = []
            for tid in current_wave:
                for dep_id in adj.get(tid, set()):
                    in_degree[dep_id] -= 1
                    if in_degree[dep_id] == 0:
                        next_wave.append(dep_id)
            current_wave = next_wave

        return waves

    def get_blocked_tasks(self) -> Dict[str, Set[str]]:
        """Return PENDING tasks mapped to their unsatisfied dependencies."""
        return {
            tid: set(self._dependencies.get(tid, set()))
            for tid, s in self._states.items()
            if s == TaskState.PENDING and self._dependencies.get(tid)
        }

    def get_downstream(self, task_id: str) -> Set[str]:
        """BFS to find all transitive dependents of *task_id*."""
        result: Set[str] = set()
        queue: deque[str] = deque(self._dependents.get(task_id, set()))
        while queue:
            nid = queue.popleft()
            if nid in result:
                continue
            result.add(nid)
            queue.extend(self._dependents.get(nid, set()))
        return result

    def get_task_state(self, task_id: str) -> Optional[TaskState]:
        """Return the current state, checking active then history."""
        return self._states.get(task_id) or self._history.get(task_id)

    def get_node(self, task_id: str) -> Optional[TaskNode]:
        """Return an immutable snapshot of the task's graph position."""
        state = self._states.get(task_id)
        if state is None:
            return None
        return TaskNode(
            task_id=task_id,
            state=state,
            dependencies=frozenset(self._dependencies.get(task_id, set())),
            dependents=frozenset(self._dependents.get(task_id, set())),
            priority=self._priorities.get(task_id, 2),
        )

    def validate_graph(self) -> Optional[List[str]]:
        """Full Kahn's validation.  Returns a cycle (list of task_ids) if
        the graph contains one, else ``None``."""
        in_degree: Dict[str, int] = {}
        adj: Dict[str, Set[str]] = {}
        for tid in self._states:
            in_degree[tid] = len(self._dependencies.get(tid, set()))
            adj[tid] = {d for d in self._dependents.get(tid, set()) if d in self._states}

        queue: deque[str] = deque(tid for tid, deg in in_degree.items() if deg == 0)
        visited = 0

        while queue:
            tid = queue.popleft()
            visited += 1
            for dep_id in adj.get(tid, set()):
                in_degree[dep_id] -= 1
                if in_degree[dep_id] == 0:
                    queue.append(dep_id)

        if visited < len(in_degree):
            # There's a cycle — find the participants
            return [tid for tid, deg in in_degree.items() if deg > 0]
        return None

    @property
    def stats(self) -> Dict[str, int]:
        """Counts by state (active + history)."""
        counts: Dict[str, int] = {s.value: 0 for s in TaskState}
        for s in self._states.values():
            counts[s.value] += 1
        for s in self._history.values():
            counts[s.value] += 1
        return counts

    # ── Serialisation ────────────────────────────────────────────────

    def to_dict(self) -> Dict[str, Any]:
        """Serialise the entire resolver state for persistence."""
        return {
            "dependencies": {k: sorted(v) for k, v in self._dependencies.items()},
            "dependents": {k: sorted(v) for k, v in self._dependents.items()},
            "states": {k: v.value for k, v in self._states.items()},
            "priorities": dict(self._priorities),
            "history": {k: v.value for k, v in self._history.items()},
            "insertion_order": list(self._insertion_order),
        }

    @classmethod
    def from_dict(cls, data: Dict[str, Any]) -> "DependencyResolver":
        """Reconstruct a resolver from persisted state."""
        resolver = cls()
        resolver._dependencies = {k: set(v) for k, v in data.get("dependencies", {}).items()}
        resolver._dependents = {k: set(v) for k, v in data.get("dependents", {}).items()}
        resolver._states = {k: TaskState(v) for k, v in data.get("states", {}).items()}
        resolver._priorities = dict(data.get("priorities", {}))
        resolver._history = {k: TaskState(v) for k, v in data.get("history", {}).items()}
        resolver._insertion_order = list(data.get("insertion_order", []))
        return resolver

    # ── Internal helpers ─────────────────────────────────────────────

    def _check_cycle(self, new_task: str, deps: Set[str]) -> None:
        """DFS from each dep backward through ``_dependents`` to see if
        *new_task* is reachable.  If so, raise ``CycleDetectedError`` with
        the full cycle path."""
        for dep in deps:
            path = self._dfs_find_path(dep, new_task)
            if path is not None:
                # path goes from dep back to new_task; complete the cycle
                cycle = [new_task] + path + [new_task]
                raise CycleDetectedError(cycle)

    def _dfs_find_path(self, start: str, target: str) -> Optional[List[str]]:
        """Return a path from *start* to *target* following ``_dependencies``
        edges, or ``None`` if no path exists."""
        visited: Set[str] = set()
        stack: List[tuple[str, List[str]]] = [(start, [start])]
        while stack:
            node, path = stack.pop()
            if node == target:
                return path
            if node in visited:
                continue
            visited.add(node)
            for dep in self._dependencies.get(node, set()):
                stack.append((dep, path + [dep]))
        return None
