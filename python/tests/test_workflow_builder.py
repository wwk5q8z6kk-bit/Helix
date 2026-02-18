"""Tests for the Workflow Builder — declarative DAGs with predictive preview."""

import time
import pytest

from core.scheduling.dependency_resolver import DependencyResolver
from core.scheduling.execution_tracker import ExecutionTracker
from core.scheduling.quality_gates import (
    QualityGate,
    QualityGatePolicy,
    GATE_LENIENT,
    GATE_STANDARD,
    GATE_STRICT,
)
from core.scheduling.retry_strategies import (
    RetryManager,
    strategy_fast_to_premium,
    strategy_no_retry,
)
from core.scheduling.scheduler_coordinator import SchedulerCoordinator
from core.scheduling.workflow_builder import (
    WorkflowBuilder,
    WorkflowDefinition,
    WorkflowPreview,
    WorkflowTask,
    code_review_pipeline,
    parallel_analysis_pipeline,
    research_pipeline,
)


@pytest.fixture(autouse=True)
def _reset_container():
    from core import di_container
    di_container._container = None
    yield
    di_container._container = None


def _make_coordinator(tracker=None):
    """Helper to build a coordinator for testing."""
    return SchedulerCoordinator(
        DependencyResolver(),
        tracker or ExecutionTracker(),
        QualityGatePolicy(),
        RetryManager(),
    )


def _seed(tracker, task_type, model, n=5, quality=0.8, cost=0.05):
    for i in range(n):
        tracker.record_start(f"{task_type}_{model}_{i}")
        tracker.record_completion(
            f"{task_type}_{model}_{i}",
            task_type=task_type, model=model,
            quality_score=quality, cost=cost,
        )


# ========================================================================
# BUILDER API
# ========================================================================


class TestBuilderAPI:
    def test_add_single_task(self):
        wf = WorkflowBuilder("test")
        wf.task("t1", task_type="coding")
        defn = wf.build()
        assert defn.task_count == 1
        assert "t1" in defn.tasks

    def test_fluent_chaining(self):
        defn = (
            WorkflowBuilder("test")
            .task("a", task_type="planning")
            .task("b", task_type="coding", depends_on=["a"])
            .task("c", task_type="testing", depends_on=["b"])
            .build()
        )
        assert defn.task_count == 3
        assert defn.name == "test"

    def test_duplicate_task_raises(self):
        wf = WorkflowBuilder("test").task("t1", task_type="x")
        with pytest.raises(ValueError, match="Duplicate"):
            wf.task("t1", task_type="y")

    def test_task_metadata(self):
        defn = (
            WorkflowBuilder("test")
            .task("t1", task_type="coding", prompt="Write a function")
            .build()
        )
        assert defn.tasks["t1"].metadata["prompt"] == "Write a function"

    def test_with_deadline(self):
        deadline = time.time() + 3600
        defn = (
            WorkflowBuilder("test")
            .task("t1", task_type="x")
            .with_deadline(deadline)
            .build()
        )
        assert defn.deadline == deadline

    def test_with_deadline_in(self):
        wf = (
            WorkflowBuilder("test")
            .task("t1", task_type="x")
            .with_deadline_in(3600)
        )
        defn = wf.build()
        assert defn.deadline is not None
        assert defn.deadline > time.time()

    def test_with_budget(self):
        defn = (
            WorkflowBuilder("test")
            .task("t1", task_type="x")
            .with_budget(10.0)
            .build()
        )
        assert defn.budget == 10.0

    def test_with_default_gate(self):
        defn = (
            WorkflowBuilder("test")
            .task("t1", task_type="x")
            .with_default_gate(GATE_STRICT)
            .build()
        )
        assert defn.default_gate == GATE_STRICT

    def test_with_metadata(self):
        defn = (
            WorkflowBuilder("test")
            .task("t1", task_type="x")
            .with_metadata(author="alice", version=2)
            .build()
        )
        assert defn.metadata["author"] == "alice"
        assert defn.metadata["version"] == 2


# ========================================================================
# VALIDATION
# ========================================================================


class TestValidation:
    def test_empty_workflow_invalid(self):
        errors = WorkflowBuilder("test").validate()
        assert len(errors) == 1
        assert "no tasks" in errors[0]

    def test_missing_dependency_invalid(self):
        errors = (
            WorkflowBuilder("test")
            .task("b", task_type="x", depends_on=["a"])
            .validate()
        )
        assert any("unknown task 'a'" in e for e in errors)

    def test_self_dependency_invalid(self):
        errors = (
            WorkflowBuilder("test")
            .task("a", task_type="x", depends_on=["a"])
            .validate()
        )
        assert any("Cycle" in e for e in errors)

    def test_negative_budget_invalid(self):
        errors = (
            WorkflowBuilder("test")
            .task("t1", task_type="x")
            .with_budget(-5.0)
            .validate()
        )
        assert any("positive" in e for e in errors)

    def test_past_deadline_invalid(self):
        errors = (
            WorkflowBuilder("test")
            .task("t1", task_type="x")
            .with_deadline(time.time() - 100)
            .validate()
        )
        assert any("past" in e for e in errors)

    def test_valid_workflow_no_errors(self):
        errors = (
            WorkflowBuilder("test")
            .task("a", task_type="x")
            .task("b", task_type="y", depends_on=["a"])
            .validate()
        )
        assert errors == []

    def test_build_raises_on_invalid(self):
        with pytest.raises(ValueError, match="no tasks"):
            WorkflowBuilder("test").build()


# ========================================================================
# WORKFLOW DEFINITION
# ========================================================================


class TestWorkflowDefinition:
    def test_topological_order(self):
        defn = (
            WorkflowBuilder("test")
            .task("c", task_type="x", depends_on=["b"])
            .task("a", task_type="x")
            .task("b", task_type="x", depends_on=["a"])
            .build()
        )
        # a must come before b, b before c
        assert defn.task_order.index("a") < defn.task_order.index("b")
        assert defn.task_order.index("b") < defn.task_order.index("c")

    def test_waves_parallel(self):
        defn = (
            WorkflowBuilder("test")
            .task("a", task_type="x")
            .task("b", task_type="x")
            .task("c", task_type="x")
            .build()
        )
        # All independent → one wave
        assert defn.wave_count == 1
        assert defn.max_parallelism == 3

    def test_waves_sequential(self):
        defn = (
            WorkflowBuilder("test")
            .task("a", task_type="x")
            .task("b", task_type="x", depends_on=["a"])
            .task("c", task_type="x", depends_on=["b"])
            .build()
        )
        assert defn.wave_count == 3
        assert defn.max_parallelism == 1

    def test_waves_diamond(self):
        defn = (
            WorkflowBuilder("test")
            .task("a", task_type="x")
            .task("b", task_type="x", depends_on=["a"])
            .task("c", task_type="x", depends_on=["a"])
            .task("d", task_type="x", depends_on=["b", "c"])
            .build()
        )
        assert defn.wave_count == 3
        assert defn.max_parallelism == 2

    def test_model_assignments_explicit(self):
        defn = (
            WorkflowBuilder("test")
            .task("t1", task_type="x", model="opus")
            .task("t2", task_type="x", model="flash")
            .build()
        )
        assert defn.model_assignments["t1"] == "opus"
        assert defn.model_assignments["t2"] == "flash"

    def test_model_auto_assignment_with_coordinator(self):
        tracker = ExecutionTracker()
        _seed(tracker, "coding", "opus", quality=0.95, cost=0.10)
        _seed(tracker, "coding", "flash", quality=0.6, cost=0.001)
        coord = _make_coordinator(tracker)

        defn = (
            WorkflowBuilder("test", coord)
            .task("t1", task_type="coding")  # No model specified
            .build()
        )
        # Should auto-assign based on Pareto
        assert defn.model_assignments["t1"] in ("opus", "flash")


# ========================================================================
# PREDICTIVE PREVIEW
# ========================================================================


class TestPreview:
    def test_preview_basic(self):
        preview = (
            WorkflowBuilder("test")
            .task("a", task_type="x")
            .task("b", task_type="y", depends_on=["a"])
            .preview()
        )
        assert isinstance(preview, WorkflowPreview)
        assert len(preview.task_predictions) == 2
        assert preview.total_predicted_duration > 0
        assert preview.total_predicted_cost > 0

    def test_preview_invalid_raises(self):
        with pytest.raises(ValueError, match="no tasks"):
            WorkflowBuilder("test").preview()

    def test_preview_meets_deadline(self):
        preview = (
            WorkflowBuilder("test")
            .task("t1", task_type="x")
            .with_deadline(time.time() + 86400)  # 1 day away
            .preview()
        )
        assert preview.meets_deadline is True
        assert preview.deadline_margin > 0

    def test_preview_misses_deadline(self):
        preview = (
            WorkflowBuilder("test")
            .task("t1", task_type="x")
            .with_deadline(time.time() + 0.001)  # ~immediate
            .preview()
        )
        # Default predicted duration is 10s, which exceeds 0.001s
        assert preview.meets_deadline is False
        assert preview.deadline_margin < 0

    def test_preview_budget_ok(self):
        preview = (
            WorkflowBuilder("test")
            .task("t1", task_type="x")
            .with_budget(100.0)
            .preview()
        )
        assert preview.meets_budget is True
        assert preview.budget_margin > 0

    def test_preview_budget_exceeded(self):
        preview = (
            WorkflowBuilder("test")
            .task("t1", task_type="x")
            .task("t2", task_type="x")
            .task("t3", task_type="x")
            .with_budget(0.001)  # Extremely tight
            .preview()
        )
        assert preview.meets_budget is False

    def test_preview_feasible(self):
        preview = (
            WorkflowBuilder("test")
            .task("t1", task_type="x")
            .with_deadline(time.time() + 86400)
            .with_budget(100.0)
            .preview()
        )
        assert preview.feasible is True

    def test_preview_risk_level(self):
        # No constraints → low risk
        preview = (
            WorkflowBuilder("test")
            .task("t1", task_type="x")
            .preview()
        )
        assert preview.risk_level == "low"

    def test_preview_with_coordinator(self):
        tracker = ExecutionTracker()
        _seed(tracker, "coding", "opus", quality=0.9, cost=0.10)
        coord = _make_coordinator(tracker)

        preview = (
            WorkflowBuilder("test", coord)
            .task("t1", task_type="coding", model="opus")
            .preview()
        )
        pred = preview.task_predictions[0]
        assert pred.model == "opus"
        assert pred.predicted_quality > 0.5  # Has history

    def test_preview_at_risk_tasks(self):
        """Tasks predicted below quality gate are flagged."""
        tracker = ExecutionTracker()
        _seed(tracker, "coding", "flash", n=5, quality=0.3, cost=0.001)
        coord = _make_coordinator(tracker)

        preview = (
            WorkflowBuilder("test", coord)
            .task("t1", task_type="coding", model="flash",
                  gate=QualityGate(min_quality=0.8))
            .preview()
        )
        assert "t1" in preview.at_risk_tasks

    def test_preview_critical_path(self):
        coord = _make_coordinator()
        preview = (
            WorkflowBuilder("test", coord)
            .task("a", task_type="x")
            .task("b", task_type="x", depends_on=["a"])
            .task("c", task_type="x", depends_on=["b"])
            .preview()
        )
        assert preview.critical_path == ["a", "b", "c"]

    def test_preview_wave_durations(self):
        preview = (
            WorkflowBuilder("test")
            .task("a", task_type="x")
            .task("b", task_type="x", depends_on=["a"])
            .preview()
        )
        assert len(preview.wave_durations) == 2
        assert all(d > 0 for d in preview.wave_durations)


# ========================================================================
# PRE-BUILT TEMPLATES
# ========================================================================


class TestTemplates:
    def test_code_review_pipeline(self):
        wf = code_review_pipeline()
        defn = wf.build()
        assert defn.name == "code_review"
        assert defn.task_count == 4
        assert defn.wave_count == 4  # All sequential
        assert "plan" in defn.tasks
        assert "implement" in defn.tasks
        assert "test" in defn.tasks
        assert "review" in defn.tasks

    def test_code_review_ordering(self):
        defn = code_review_pipeline().build()
        order = defn.task_order
        assert order.index("plan") < order.index("implement")
        assert order.index("implement") < order.index("test")
        assert order.index("test") < order.index("review")

    def test_code_review_gates(self):
        defn = code_review_pipeline().build()
        assert defn.tasks["implement"].gate == GATE_STRICT
        assert defn.tasks["test"].gate == GATE_LENIENT
        assert defn.tasks["review"].gate == GATE_STRICT

    def test_research_pipeline(self):
        defn = research_pipeline().build()
        assert defn.name == "research"
        assert defn.task_count == 4
        assert defn.wave_count == 4

    def test_parallel_analysis_pipeline(self):
        defn = parallel_analysis_pipeline().build()
        assert defn.name == "parallel_analysis"
        assert defn.task_count == 5
        assert defn.max_parallelism == 3  # Three analyses in parallel
        assert defn.wave_count == 3  # prepare → [3 analyses] → merge

    def test_template_with_coordinator(self):
        tracker = ExecutionTracker()
        _seed(tracker, "coding", "opus", quality=0.9, cost=0.10)
        _seed(tracker, "planning", "flash", quality=0.7, cost=0.001)
        coord = _make_coordinator(tracker)

        defn = code_review_pipeline(coord).build()
        # Should auto-assign models for tasks without explicit model
        assert all(m != "" for m in defn.model_assignments.values())

    def test_template_preview(self):
        preview = code_review_pipeline().preview()
        assert preview.name == "code_review"
        assert len(preview.task_predictions) == 4
        assert preview.feasible  # No constraints set

    def test_template_customization(self):
        """Templates are customizable via additional fluent calls."""
        defn = (
            code_review_pipeline()
            .with_budget(5.0)
            .with_deadline_in(7200)
            .build()
        )
        assert defn.budget == 5.0
        assert defn.deadline is not None

    def test_edge_gate_on_template(self):
        """Can add edge-level gates to pre-built templates."""
        strict = QualityGate(min_quality=0.95)
        wf = code_review_pipeline().edge_gate("test", "review", strict)
        defn = wf.build()
        # The edge gate is stored on the builder, not the definition
        assert ("test", "review") in wf._edge_gates
