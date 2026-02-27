"""Quality-gated dependency resolution for Helix.

Dependencies don't just wait for *completion* — they wait for *quality
completion*.  If a task completes with quality below its gate threshold,
it's treated as a soft failure: the system retries (with model escalation)
before unblocking downstream tasks.

This is what makes Helix fundamentally different from every other
orchestrator.  Airflow/Prefect/Dagster treat completion as binary.  Helix
treats completion as a *quality spectrum* and only propagates results that
meet a configurable standard.
"""

from __future__ import annotations

import logging
from dataclasses import dataclass, field
from enum import Enum
from typing import Any, Dict, List, Optional, Tuple

logger = logging.getLogger(__name__)


# ── Configuration ────────────────────────────────────────────────────


class GateAction(str, Enum):
    """What to do when a quality gate fails."""

    RETRY = "retry"  # Retry with model escalation
    WARN_AND_PASS = "warn_and_pass"  # Log warning, unblock anyway
    BLOCK = "block"  # Keep downstream blocked until manual override
    FAIL = "fail"  # Treat as hard failure, cancel downstream


@dataclass(frozen=True)
class QualityGate:
    """Quality threshold for a dependency edge or task output.

    When a task completes, its quality_score and hrm_score are checked
    against the gate.  If either is below threshold, the gate_action
    determines what happens next.
    """

    min_quality: float = 0.5  # Minimum quality_score (0.0 – 1.0)
    min_hrm: float = 0.0  # Minimum hrm_score (0.0 – 1.0); 0 = disabled
    max_retries: int = 2  # Max retries before gate_action fallback
    gate_action: GateAction = GateAction.RETRY  # What to do on failure
    escalation_models: Tuple[str, ...] = ()  # Models to try in order


# Sensible defaults for different contexts
GATE_STRICT = QualityGate(min_quality=0.7, min_hrm=0.6, max_retries=3, gate_action=GateAction.RETRY)
GATE_STANDARD = QualityGate(min_quality=0.5, min_hrm=0.0, max_retries=2, gate_action=GateAction.RETRY)
GATE_LENIENT = QualityGate(min_quality=0.3, min_hrm=0.0, max_retries=1, gate_action=GateAction.WARN_AND_PASS)
GATE_DISABLED = QualityGate(min_quality=0.0, min_hrm=0.0, max_retries=0, gate_action=GateAction.WARN_AND_PASS)


# ── Gate evaluation result ───────────────────────────────────────────


class GateVerdict(str, Enum):
    """Result of evaluating a quality gate."""

    PASSED = "passed"  # Quality meets threshold
    RETRY = "retry"  # Below threshold, should retry
    WARN_PASS = "warn_pass"  # Below threshold, but passing with warning
    BLOCKED = "blocked"  # Below threshold, downstream stays blocked
    FAILED = "failed"  # Below threshold, treat as hard failure


@dataclass
class GateResult:
    """Detailed result of a quality gate evaluation."""

    task_id: str
    verdict: GateVerdict
    gate: QualityGate
    quality_score: float
    hrm_score: float
    attempt: int
    reason: str = ""
    suggested_model: Optional[str] = None


# ── Quality Gate Policy ──────────────────────────────────────────────


class QualityGatePolicy:
    """Manages quality gates for tasks and dependency edges.

    Supports three levels of gate configuration:
    1. **Default gate**: Applied to all tasks unless overridden.
    2. **Task-level gates**: Override for specific task_ids.
    3. **Edge-level gates**: Override for specific (upstream, downstream) pairs.

    Edge gates take precedence over task gates, which take precedence
    over the default.
    """

    def __init__(self, default_gate: Optional[QualityGate] = None) -> None:
        self._default_gate: QualityGate = default_gate or GATE_STANDARD
        # Task-level gates: task_id → gate
        self._task_gates: Dict[str, QualityGate] = {}
        # Edge-level gates: (upstream_id, downstream_id) → gate
        self._edge_gates: Dict[Tuple[str, str], QualityGate] = {}
        # Retry attempt counters: task_id → current attempt for quality gate retries
        self._gate_attempts: Dict[str, int] = {}

    def set_default_gate(self, gate: QualityGate) -> None:
        """Set the default quality gate for all tasks."""
        self._default_gate = gate

    def set_task_gate(self, task_id: str, gate: QualityGate) -> None:
        """Set a quality gate for a specific task's output."""
        self._task_gates[task_id] = gate

    def set_edge_gate(
        self, upstream_id: str, downstream_id: str, gate: QualityGate
    ) -> None:
        """Set a quality gate for a specific dependency edge."""
        self._edge_gates[(upstream_id, downstream_id)] = gate

    def get_gate(
        self, task_id: str, downstream_id: Optional[str] = None
    ) -> QualityGate:
        """Get the applicable gate, respecting precedence."""
        if downstream_id and (task_id, downstream_id) in self._edge_gates:
            return self._edge_gates[(task_id, downstream_id)]
        if task_id in self._task_gates:
            return self._task_gates[task_id]
        return self._default_gate

    def evaluate(
        self,
        task_id: str,
        quality_score: float,
        hrm_score: float = 0.0,
        downstream_ids: Optional[List[str]] = None,
    ) -> GateResult:
        """Evaluate whether a task's output passes its quality gate.

        If ``downstream_ids`` is provided, the strictest applicable gate
        across all downstream edges is used.
        """
        # Find the strictest applicable gate
        gates: List[QualityGate] = []
        if downstream_ids:
            for did in downstream_ids:
                gates.append(self.get_gate(task_id, did))
        else:
            gates.append(self.get_gate(task_id))

        # Use the strictest gate (highest min_quality)
        gate = max(gates, key=lambda g: (g.min_quality, g.min_hrm))

        attempt = self._gate_attempts.get(task_id, 0) + 1
        self._gate_attempts[task_id] = attempt

        # Check quality threshold
        quality_ok = quality_score >= gate.min_quality
        hrm_ok = gate.min_hrm == 0.0 or hrm_score >= gate.min_hrm

        if quality_ok and hrm_ok:
            return GateResult(
                task_id=task_id,
                verdict=GateVerdict.PASSED,
                gate=gate,
                quality_score=quality_score,
                hrm_score=hrm_score,
                attempt=attempt,
            )

        # Gate failed — determine action
        reasons = []
        if not quality_ok:
            reasons.append(f"quality {quality_score:.2f} < {gate.min_quality:.2f}")
        if not hrm_ok:
            reasons.append(f"hrm {hrm_score:.2f} < {gate.min_hrm:.2f}")
        reason = "; ".join(reasons)

        # Determine escalation model
        suggested_model = None
        if gate.escalation_models and attempt <= len(gate.escalation_models):
            suggested_model = gate.escalation_models[attempt - 1]

        # Retry up to max_retries regardless of gate_action; gate_action is
        # the fallback AFTER retries are exhausted.
        if attempt <= gate.max_retries:
            logger.info(
                "Quality gate RETRY for %s (attempt %d/%d): %s",
                task_id, attempt, gate.max_retries, reason,
            )
            return GateResult(
                task_id=task_id,
                verdict=GateVerdict.RETRY,
                gate=gate,
                quality_score=quality_score,
                hrm_score=hrm_score,
                attempt=attempt,
                reason=reason,
                suggested_model=suggested_model,
            )

        # Exhausted retries or non-retry action
        if gate.gate_action == GateAction.WARN_AND_PASS:
            logger.warning(
                "Quality gate WARN_PASS for %s: %s (passing anyway)",
                task_id, reason,
            )
            return GateResult(
                task_id=task_id,
                verdict=GateVerdict.WARN_PASS,
                gate=gate,
                quality_score=quality_score,
                hrm_score=hrm_score,
                attempt=attempt,
                reason=reason,
            )

        if gate.gate_action == GateAction.BLOCK:
            logger.warning(
                "Quality gate BLOCKED for %s: %s",
                task_id, reason,
            )
            return GateResult(
                task_id=task_id,
                verdict=GateVerdict.BLOCKED,
                gate=gate,
                quality_score=quality_score,
                hrm_score=hrm_score,
                attempt=attempt,
                reason=reason,
            )

        # FAIL or exhausted RETRY
        logger.error(
            "Quality gate FAILED for %s: %s (after %d attempts)",
            task_id, reason, attempt,
        )
        return GateResult(
            task_id=task_id,
            verdict=GateVerdict.FAILED,
            gate=gate,
            quality_score=quality_score,
            hrm_score=hrm_score,
            attempt=attempt,
            reason=reason,
        )

    def reset_attempts(self, task_id: str) -> None:
        """Reset the gate attempt counter (e.g. on manual override)."""
        self._gate_attempts.pop(task_id, None)

    def remove_task(self, task_id: str) -> None:
        """Clean up all gates and counters for a task."""
        self._task_gates.pop(task_id, None)
        self._gate_attempts.pop(task_id, None)
        edges_to_remove = [k for k in self._edge_gates if task_id in k]
        for k in edges_to_remove:
            del self._edge_gates[k]

    # ── Serialisation ────────────────────────────────────────────────

    def to_dict(self) -> Dict[str, Any]:
        return {
            "default_gate": _gate_to_dict(self._default_gate),
            "task_gates": {tid: _gate_to_dict(g) for tid, g in self._task_gates.items()},
            "edge_gates": {
                f"{u}:{d}": _gate_to_dict(g)
                for (u, d), g in self._edge_gates.items()
            },
            "gate_attempts": dict(self._gate_attempts),
        }

    @classmethod
    def from_dict(cls, data: Dict[str, Any]) -> "QualityGatePolicy":
        policy = cls(default_gate=_gate_from_dict(data.get("default_gate", {})))
        for tid, gd in data.get("task_gates", {}).items():
            policy._task_gates[tid] = _gate_from_dict(gd)
        for key, gd in data.get("edge_gates", {}).items():
            parts = key.split(":", 1)
            if len(parts) == 2:
                policy._edge_gates[(parts[0], parts[1])] = _gate_from_dict(gd)
        policy._gate_attempts = dict(data.get("gate_attempts", {}))
        return policy


# ── Helpers ──────────────────────────────────────────────────────────


def _gate_to_dict(gate: QualityGate) -> Dict[str, Any]:
    return {
        "min_quality": gate.min_quality,
        "min_hrm": gate.min_hrm,
        "max_retries": gate.max_retries,
        "gate_action": gate.gate_action.value,
        "escalation_models": list(gate.escalation_models),
    }


def _gate_from_dict(data: Dict[str, Any]) -> QualityGate:
    return QualityGate(
        min_quality=data.get("min_quality", 0.5),
        min_hrm=data.get("min_hrm", 0.0),
        max_retries=data.get("max_retries", 2),
        gate_action=GateAction(data.get("gate_action", "retry")),
        escalation_models=tuple(data.get("escalation_models", [])),
    )
