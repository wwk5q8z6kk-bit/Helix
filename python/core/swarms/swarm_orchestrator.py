"""
Swarm Orchestrator - Routes tasks to appropriate specialized swarms
Integrates swarms into Helix's main task execution flow
"""

from typing import Dict, List, Optional, Any, Tuple
from dataclasses import dataclass, field
from enum import Enum
import logging
import re

from core.swarms.base_swarm import Task, SwarmResult
from core.swarms.shared_state import get_shared_state_manager
from core.swarms.a2a_protocol import get_a2a_protocol, MessageType
from core.swarms.knowledge_pool import get_knowledge_pool
from core.exceptions_unified import AgentError, TaskExecutionError
from core.reasoning import AgenticReasoner, get_agentic_reasoner
from core.reasoning.trajectory_tracker import get_trajectory_tracker
from core.reasoning.process_reward_model import get_process_reward_model
from core.reasoning.historical_learning import get_strategy_recommender
from core.learning import get_feedback_collector

logger = logging.getLogger(__name__)


class TaskCategory(Enum):
    """Categories of tasks that can be routed to swarms"""
    REQUIREMENTS = "requirements"
    ARCHITECTURE = "architecture"
    IMPLEMENTATION = "implementation"
    TESTING = "testing"
    CODE_REVIEW = "code_review"
    DEBUGGING = "debugging"
    DEPLOYMENT = "deployment"
    GENERAL = "general"
    UNKNOWN = "unknown"


@dataclass
class RoutingDecision:
    """Decision about which swarm should handle a task"""
    task_category: TaskCategory
    swarm_name: str
    confidence: float  # 0-1
    reasoning: str
    alternative_swarms: List[str] = field(default_factory=list)
    metadata: Dict[str, Any] = field(default_factory=dict)


@dataclass
class OrchestrationResult:
    """Result from orchestrated task execution"""
    task_id: str
    task_category: TaskCategory
    swarm_used: str
    swarm_result: SwarmResult
    routing_confidence: float
    execution_time: float
    quality_score: float
    learned_patterns: int
    metadata: Dict[str, Any] = field(default_factory=dict)


class SwarmOrchestrator:
    """
    Orchestrates task routing to specialized swarms

    Features:
    - Intelligent task classification using AgenticReasoner
    - Dynamic routing based on task characteristics
    - Learning from routing decisions (Enhancement 12)
    - Cross-swarm coordination via A2A protocol
    - Integration with Helix's main execution flow

    Architecture:
    This is the CRITICAL integration layer that makes swarms useful:
    - execute_task.py calls orchestrator.route_and_execute()
    - Orchestrator analyzes task and routes to appropriate swarm
    - Swarm executes with brain integration (E7-E12)
    - Results flow back through orchestrator to user
    """

    def __init__(self):
        """Initialize orchestrator with brain integration"""
        # Brain components (singletons)
        self.tracker = get_trajectory_tracker()
        self.prm = get_process_reward_model()
        self.reasoner = get_agentic_reasoner()

        # Swarm infrastructure
        self.state_manager = get_shared_state_manager()
        self.a2a = get_a2a_protocol()
        self.knowledge_pool = get_knowledge_pool()

        # Learning components (Phase 1 Enhancement)
        self.strategy_recommender = get_strategy_recommender()
        self.feedback_collector = get_feedback_collector()

        # Lazy-load swarms (only import when needed)
        self._swarms_cache: Dict[str, Any] = {}

        # Routing patterns learned from history
        self._routing_patterns: Dict[str, TaskCategory] = {}

        # Keyword weights (PRIORITY-BASED WEIGHTING)
        # Higher weight = stronger signal for that category
        self._keyword_weights = {
            # Strong signals (1.5x weight)
            "debug": 1.5, "debugging": 1.5, "troubleshoot": 1.5,
            "review": 1.5, "audit": 1.5, "inspect": 1.5,
            "test": 1.5, "testing": 1.5, "validate": 1.5,
            "deploy": 1.5, "deployment": 1.5, "release": 1.5,
            "requirements": 1.5, "stakeholder": 1.5, "elicit": 1.5,
            "architecture": 1.5, "design": 1.5, "schema": 1.5,
            # Medium signals (1.0x weight - default)
            "fix": 1.0, "bug": 1.0, "error": 1.0,
            "implement": 1.0, "build": 1.0, "develop": 1.0,
            "check": 1.0, "analyze": 1.0, "examine": 1.0,
            # Weak/ambiguous signals (0.7x weight)
            "create": 0.7, "make": 0.7, "write": 0.7,
            "api": 0.7, "code": 0.7, "function": 0.7,
        }

        # Multi-word phrase patterns (checked BEFORE individual keywords)
        # These are strong category indicators
        self._phrase_patterns = {
            # Requirements phrases
            "user stor": TaskCategory.REQUIREMENTS,  # "user story/stories"
            "acceptance criteria": TaskCategory.REQUIREMENTS,
            "business goal": TaskCategory.REQUIREMENTS,
            "functional requirement": TaskCategory.REQUIREMENTS,
            "gather requirement": TaskCategory.REQUIREMENTS,
            "analyze requirement": TaskCategory.REQUIREMENTS,

            # Architecture phrases
            "database schema": TaskCategory.ARCHITECTURE,
            "system design": TaskCategory.ARCHITECTURE,
            "system architecture": TaskCategory.ARCHITECTURE,
            "data model": TaskCategory.ARCHITECTURE,
            "high-level design": TaskCategory.ARCHITECTURE,

            # Testing phrases
            "unit test": TaskCategory.TESTING,
            "integration test": TaskCategory.TESTING,
            "e2e test": TaskCategory.TESTING,
            "test case": TaskCategory.TESTING,
            "test suite": TaskCategory.TESTING,
            "test coverage": TaskCategory.TESTING,

            # Code review phrases
            "code review": TaskCategory.CODE_REVIEW,
            "security review": TaskCategory.CODE_REVIEW,
            "performance review": TaskCategory.CODE_REVIEW,
            "code quality": TaskCategory.CODE_REVIEW,
            "best practice": TaskCategory.CODE_REVIEW,

            # Debugging phrases (HIGH PRIORITY)
            "debug": TaskCategory.DEBUGGING,  # Strong debugging signal
            "fix bug": TaskCategory.DEBUGGING,
            "fix error": TaskCategory.DEBUGGING,
            "troubleshoot": TaskCategory.DEBUGGING,

            # Implementation phrases
            "rest api": TaskCategory.IMPLEMENTATION,
            "api endpoint": TaskCategory.IMPLEMENTATION,

            # Deployment phrases
            "ci/cd": TaskCategory.DEPLOYMENT,
        }

        # Task classification keywords (ENHANCED with better coverage)
        self._category_keywords = {
            TaskCategory.REQUIREMENTS: [
                "requirements", "gather", "elicit", "stakeholder", "business goal",
                "what should", "needs to", "user story", "acceptance criteria",
                "analyze requirements", "requirement analysis", "specifications",
                "functional requirements", "non-functional", "use case"
            ],
            TaskCategory.ARCHITECTURE: [
                "architecture", "design", "system", "component", "pattern",
                "microservices", "scalability", "infrastructure", "deployment strategy",
                "design system", "system design", "database schema", "data model",
                "high-level design", "architectural", "schema", "structure"
            ],
            TaskCategory.IMPLEMENTATION: [
                "implement", "code", "build", "develop", "create", "generate",
                "write code", "backend", "frontend", "api", "function", "class",
                "write", "program", "coding", "development", "software",
                "application", "service", "endpoint", "logic"
            ],
            TaskCategory.TESTING: [
                "test", "unit test", "integration test", "e2e", "coverage",
                "test case", "assert", "verify", "validate", "testing",
                "test suite", "test plan", "qa", "quality assurance"
            ],
            TaskCategory.CODE_REVIEW: [
                "review", "audit", "check", "analyze code", "security review",
                "performance review", "code quality", "best practices",
                "code review", "inspect", "examine", "evaluate"
            ],
            TaskCategory.DEBUGGING: [
                "debug", "fix", "error", "bug", "issue", "problem",
                "not working", "failing", "crash", "exception",
                "troubleshoot", "resolve", "repair", "broken"
            ],
            TaskCategory.DEPLOYMENT: [
                "deploy", "release", "production", "build", "ci/cd",
                "docker", "kubernetes", "container", "monitor",
                "deployment", "rollout", "publish", "ship"
            ]
        }

        logger.info("SwarmOrchestrator initialized with brain integration")

    async def route_and_execute(
        self,
        task_description: str,
        context: Optional[Dict[str, Any]] = None
    ) -> OrchestrationResult:
        """
        Route task to appropriate swarm and execute

        This is the MAIN ENTRY POINT from execute_task.py

        Args:
            task_description: User's task description
            context: Additional context about the task

        Returns:
            OrchestrationResult with swarm execution results
        """
        logger.info(f"Orchestrating task: {task_description[:100]}...")

        # Step 1: Classify task and decide routing
        routing = await self._classify_and_route(task_description, context or {})

        logger.info(
            f"Routing decision: {routing.swarm_name} "
            f"(category: {routing.task_category.value}, "
            f"confidence: {routing.confidence:.2f})"
        )

        # Step 2: Get appropriate swarm
        swarm = await self._get_swarm(routing.swarm_name)

        # Step 3: Execute with swarm
        try:
            swarm_result = await self._execute_with_swarm(
                swarm,
                routing,
                task_description,
                context or {}
            )

            # Step 4: Learn from execution
            if swarm_result.quality.final_score > 0.8:
                self._learn_routing_pattern(task_description, routing.task_category)

            # Step 4.5: Collect feedback for continuous learning (Phase 1 Enhancement)
            asyncio.create_task(
                self._collect_execution_feedback(
                    task_description=task_description,
                    swarm_name=routing.swarm_name,
                    swarm_result=swarm_result
                )
            )

            # Step 5: Return orchestration result
            return OrchestrationResult(
                task_id=swarm_result.task_id,
                task_category=routing.task_category,
                swarm_used=routing.swarm_name,
                swarm_result=swarm_result,
                routing_confidence=routing.confidence,
                execution_time=swarm_result.execution_time,
                quality_score=swarm_result.quality.final_score,
                learned_patterns=len(swarm_result.patterns_used),
                metadata={
                    'routing_reasoning': routing.reasoning,
                    'alternatives': routing.alternative_swarms
                }
            )

        except AgentError as e:
            logger.error(f"Swarm execution failed: {e}")
            # Fallback to general execution
            return await self._fallback_execution(task_description, context or {}, str(e))

    async def _classify_and_route(
        self,
        task_description: str,
        context: Dict[str, Any]
    ) -> RoutingDecision:
        """
        Classify task and determine routing

        Uses historical learning + keyword matching + learned patterns
        Phase 1 Enhancement: Integrated historical learning for strategy recommendation
        """
        # PHASE 0: Check historical learning recommendations (HIGHEST PRIORITY)
        # This uses data from past successful executions to guide routing
        try:
            strategy, confidence, metadata = self.strategy_recommender.recommend_strategy(task_description)

            # Only use historical recommendation if confidence is high
            if confidence >= 0.7:
                # Map strategy to task category (heuristic mapping)
                problem_type = metadata.get('problem_type', 'general')
                task_category = self._map_problem_type_to_category(problem_type)

                logger.info(
                    f"Historical learning: {problem_type} → {task_category.value} "
                    f"(confidence={confidence:.2f}, {metadata.get('sample_count', 0)} samples)"
                )

                return RoutingDecision(
                    task_category=task_category,
                    swarm_name=self._get_swarm_name_for_category(task_category),
                    confidence=confidence,
                    reasoning=f"Historical learning: {metadata.get('reason', 'Based on past successes')}",
                    alternative_swarms=[],
                    metadata=metadata
                )
        except AgentError as e:
            logger.debug(f"Historical learning not available: {e}")

        # Check learned patterns first
        for pattern, category in self._routing_patterns.items():
            if pattern.lower() in task_description.lower():
                return RoutingDecision(
                    task_category=category,
                    swarm_name=self._get_swarm_name_for_category(category),
                    confidence=0.95,  # High confidence from learned pattern
                    reasoning=f"Matched learned pattern: {pattern}",
                    alternative_swarms=[]
                )

        task_lower = task_description.lower()

        # PHASE 1: Check phrase patterns FIRST (highest priority)
        # Phrases are checked before individual keywords to avoid ambiguity
        phrase_matches = {}
        for phrase, category in self._phrase_patterns.items():
            if phrase in task_lower:
                phrase_matches[category] = phrase_matches.get(category, 0) + 1

        # If strong phrase match found, use it with high confidence
        if phrase_matches:
            best_phrase_category = max(phrase_matches, key=phrase_matches.get)
            phrase_count = phrase_matches[best_phrase_category]

            # High confidence for phrase matches
            confidence = min(1.0, 0.7 + (phrase_count * 0.15))

            return RoutingDecision(
                task_category=best_phrase_category,
                swarm_name=self._get_swarm_name_for_category(best_phrase_category),
                confidence=confidence,
                reasoning=f"Phrase pattern match: {phrase_count} phrase(s) detected",
                alternative_swarms=[]
            )

        # PHASE 2: Keyword matching with WEIGHTING (ENHANCED)
        category_scores = {}
        for category, keywords in self._category_keywords.items():
            # Find matching keywords
            matches = [
                keyword for keyword in keywords
                if keyword.lower() in task_lower
            ]

            if matches:
                # WEIGHTED SCORING: Apply keyword weights
                weighted_sum = sum(
                    self._keyword_weights.get(keyword.lower(), 1.0)
                    for keyword in matches
                )

                # Normalize: divide by 3 (target for high score)
                # With weighting: 2 strong keywords (1.5 each) = 3.0 → score of 1.0
                base_score = min(1.0, weighted_sum / 3.0)

                category_scores[category] = base_score

        if category_scores:
            # Get best category
            best_category = max(category_scores, key=category_scores.get)
            confidence = category_scores[best_category]

            # BONUS: Add confidence boost if one category significantly better than others
            sorted_scores = sorted(category_scores.values(), reverse=True)
            if len(sorted_scores) > 1 and sorted_scores[0] > sorted_scores[1] * 1.5:
                # Clear winner: boost confidence by 0.2
                confidence = min(1.0, confidence + 0.2)

            # Get alternatives
            alternatives = [
                self._get_swarm_name_for_category(cat)
                for cat, score in sorted(
                    category_scores.items(),
                    key=lambda x: x[1],
                    reverse=True
                )[1:3]  # Top 2 alternatives
            ]

            # Scale confidence (ENHANCED: Better scaling for moderate scores)
            # Old: scaled_confidence * 2 (0.2 -> 0.4, 0.3 -> 0.6)
            # New: Use better curve that rewards even modest matches
            if confidence >= 0.5:
                scaled_confidence = min(1.0, confidence * 1.5)  # Strong matches: multiply by 1.5
            else:
                scaled_confidence = min(1.0, confidence + 0.3)   # Modest matches: add 0.3 boost

            # CONFIDENCE THRESHOLD ENFORCEMENT
            # Reject low-confidence routes (< 0.4) and fall back
            if scaled_confidence < 0.4:
                logger.warning(
                    f"Low routing confidence ({scaled_confidence:.2f}) for task: "
                    f"{task_description[:60]}... - falling back to implementation"
                )
                return RoutingDecision(
                    task_category=TaskCategory.IMPLEMENTATION,
                    swarm_name="implementation",
                    confidence=0.4,  # Minimum threshold
                    reasoning=(
                        f"Low confidence ({scaled_confidence:.2f}) for {best_category.value} "
                        f"- falling back to general implementation swarm"
                    ),
                    alternative_swarms=[
                        self._get_swarm_name_for_category(best_category),
                        *alternatives[:1]  # Include originally suggested swarm
                    ]
                )

            return RoutingDecision(
                task_category=best_category,
                swarm_name=self._get_swarm_name_for_category(best_category),
                confidence=scaled_confidence,
                reasoning=f"Keyword matching: {len(category_scores)} categories matched",
                alternative_swarms=alternatives
            )

        # Default to general/implementation
        return RoutingDecision(
            task_category=TaskCategory.IMPLEMENTATION,
            swarm_name="implementation",
            confidence=0.5,
            reasoning="No specific category matched - defaulting to implementation",
            alternative_swarms=["testing", "code_review"]
        )

    def _get_swarm_name_for_category(self, category: TaskCategory) -> str:
        """Map category to swarm name"""
        mapping = {
            TaskCategory.REQUIREMENTS: "requirements",
            TaskCategory.ARCHITECTURE: "architecture",
            TaskCategory.IMPLEMENTATION: "implementation",
            TaskCategory.TESTING: "testing",
            TaskCategory.CODE_REVIEW: "code_review",
            TaskCategory.DEBUGGING: "debugging",
            TaskCategory.DEPLOYMENT: "deployment",
            TaskCategory.GENERAL: "implementation",
            TaskCategory.UNKNOWN: "implementation"
        }
        return mapping.get(category, "implementation")

    def _map_problem_type_to_category(self, problem_type: str) -> TaskCategory:
        """
        Map historical learning problem type to TaskCategory

        Args:
            problem_type: Problem type from historical learning (e.g., 'api_design', 'debugging')

        Returns:
            TaskCategory enum value
        """
        mapping = {
            'api_design': TaskCategory.IMPLEMENTATION,
            'debugging': TaskCategory.DEBUGGING,
            'architecture': TaskCategory.ARCHITECTURE,
            'data': TaskCategory.IMPLEMENTATION,
            'authentication': TaskCategory.IMPLEMENTATION,
            'optimization': TaskCategory.IMPLEMENTATION,
            'testing': TaskCategory.TESTING,
            'general': TaskCategory.IMPLEMENTATION
        }
        return mapping.get(problem_type, TaskCategory.IMPLEMENTATION)

    async def _get_swarm(self, swarm_name: str) -> Any:
        """
        Get swarm instance (lazy loading)

        Args:
            swarm_name: Name of swarm to load

        Returns:
            Swarm instance
        """
        # Check cache first
        if swarm_name in self._swarms_cache:
            return self._swarms_cache[swarm_name]

        # Lazy import and instantiate
        try:
            if swarm_name == "requirements":
                from core.swarms.requirements_swarm import get_requirements_swarm
                swarm = get_requirements_swarm()
            elif swarm_name == "architecture":
                from core.swarms.architecture_swarm import get_architecture_swarm
                swarm = get_architecture_swarm()
            elif swarm_name == "implementation":
                from core.swarms.implementation_swarm import get_implementation_swarm
                swarm = get_implementation_swarm()
            elif swarm_name == "testing":
                from core.swarms.testing_swarm import get_testing_swarm
                swarm = get_testing_swarm()
            elif swarm_name == "code_review":
                from core.swarms.code_review_swarm import get_code_review_swarm
                swarm = get_code_review_swarm()
            elif swarm_name == "debugging":
                from core.swarms.debugging_swarm import get_debugging_swarm
                swarm = get_debugging_swarm()
            elif swarm_name == "deployment":
                from core.swarms.deployment_swarm import get_deployment_swarm
                swarm = get_deployment_swarm()
            else:
                # Default fallback
                from core.swarms.implementation_swarm import get_implementation_swarm
                swarm = get_implementation_swarm()

            # Cache for reuse
            self._swarms_cache[swarm_name] = swarm
            logger.info(f"Loaded and cached swarm: {swarm_name}")

            return swarm

        except ImportError as e:
            logger.error(f"Failed to import swarm {swarm_name}: {e}")
            # Fallback to implementation swarm
            from core.swarms.implementation_swarm import get_implementation_swarm
            return get_implementation_swarm()

    async def _execute_with_swarm(
        self,
        swarm: Any,
        routing: RoutingDecision,
        task_description: str,
        context: Dict[str, Any]
    ) -> SwarmResult:
        """
        Execute task with appropriate swarm

        Args:
            swarm: Swarm instance
            routing: Routing decision
            task_description: Task description
            context: Task context

        Returns:
            SwarmResult from swarm execution
        """
        # Create task object
        task = Task(
            task_id=f"{routing.task_category.value}_{hash(task_description) % 10000}",
            description=task_description,
            task_type=routing.task_category.value,
            context=context,
            requirements=[]
        )

        # Route to appropriate swarm method based on category
        if routing.task_category == TaskCategory.REQUIREMENTS:
            from core.swarms.requirements_swarm import RequirementsInput
            input_data = RequirementsInput(
                project_description=task_description,
                stakeholders=context.get('stakeholders', []),
                constraints=context.get('constraints', [])
            )
            result = await swarm.analyze_requirements(input_data)
            # Convert to SwarmResult
            return self._convert_to_swarm_result(task, result, routing.task_category)

        elif routing.task_category == TaskCategory.ARCHITECTURE:
            from core.swarms.architecture_swarm import ArchitectureInput
            input_data = ArchitectureInput(
                project_description=task_description,
                requirements=context.get('requirements', []),
                constraints=context.get('constraints', [])
            )
            result = await swarm.design_architecture(input_data)
            return self._convert_to_swarm_result(task, result, routing.task_category)

        elif routing.task_category == TaskCategory.IMPLEMENTATION:
            from core.swarms.implementation_swarm import ImplementationInput
            input_data = ImplementationInput(
                project_description=task_description,
                architecture_components=context.get('architecture_components', []),
                requirements=context.get('requirements', [])
            )
            result = await swarm.generate_implementation(input_data)
            return self._convert_to_swarm_result(task, result, routing.task_category)

        elif routing.task_category == TaskCategory.TESTING:
            from core.swarms.testing_swarm import TestRequirements
            reqs = TestRequirements(
                code=context.get('code', task_description),
                requirements=task_description,
                test_types=["unit", "integration"]
            )
            result = await swarm.generate_tests(reqs)
            return self._convert_to_swarm_result(task, result, routing.task_category)

        elif routing.task_category == TaskCategory.CODE_REVIEW:
            from core.swarms.code_review_swarm import CodeReviewRequirements
            reqs = CodeReviewRequirements(
                code=context.get('code', task_description),
                review_types=["security", "performance", "quality"]
            )
            result = await swarm.review_code(reqs)
            return self._convert_to_swarm_result(task, result, routing.task_category)

        elif routing.task_category == TaskCategory.DEBUGGING:
            from core.swarms.debugging_swarm import DebuggingInput
            input_data = DebuggingInput(
                problem_description=task_description,
                code=context.get('code'),
                error_messages=context.get('error_messages', []),
                stack_traces=context.get('stack_traces', [])
            )
            result = await swarm.debug_issue(input_data)
            return self._convert_to_swarm_result(task, result, routing.task_category)

        elif routing.task_category == TaskCategory.DEPLOYMENT:
            from core.swarms.deployment_swarm import DeploymentInput, DeploymentEnvironment, DeploymentStrategy
            input_data = DeploymentInput(
                project_name=context.get('project_name', 'project'),
                version=context.get('version', '1.0.0'),
                environment=DeploymentEnvironment(context.get('environment', 'staging')),
                strategy=DeploymentStrategy(context.get('strategy', 'rolling'))
            )
            result = await swarm.deploy(input_data)
            return self._convert_to_swarm_result(task, result, routing.task_category)

        else:
            # Fallback: use base swarm execute
            from core.reasoning import CollaborationStrategy
            return await swarm.execute(task, CollaborationStrategy.DIVIDE_AND_CONQUER)

    def _convert_to_swarm_result(
        self,
        task: Task,
        swarm_output: Any,
        category: TaskCategory
    ) -> SwarmResult:
        """Convert swarm-specific output to standard SwarmResult"""
        from core.reasoning import TrajectoryReward

        # Extract quality score
        quality_score = getattr(swarm_output, 'quality_score', 0.85)

        # Create TrajectoryReward (matching SwarmResult.quality type)
        quality = TrajectoryReward(
            trajectory_id=task.task_id,
            overall_score=quality_score,
            step_scores=[],
            temporal_coherence=quality_score * 0.95,
            efficiency_bonus=quality_score * 0.1,
            penalty=0.0,
            final_score=quality_score,
            metadata={'swarm_category': category.value}
        )

        # Create dummy trajectory for SwarmResult
        from core.reasoning import ReasoningTrajectory, ReasoningStep, StepType
        trajectory = ReasoningTrajectory(
            trajectory_id=task.task_id,
            problem=task.description,
            steps=[],
            final_answer=str(swarm_output),
            success=quality_score > 0.5,
            metadata={'swarm_category': category.value}
        )

        return SwarmResult(
            task_id=task.task_id,
            success=quality_score > 0.5,
            output=swarm_output,
            trajectory=trajectory,
            quality=quality,
            agents_involved=3,  # Most swarms have 3-5 agents
            execution_time=1.5,
            patterns_used=[],
            metadata={
                'swarm_category': category.value,
                'output_type': type(swarm_output).__name__
            }
        )

    def _learn_routing_pattern(self, task_description: str, category: TaskCategory):
        """Learn routing pattern from successful execution"""
        # Extract key phrases (simple version)
        words = task_description.lower().split()
        if len(words) >= 3:
            # Store 3-word phrases as patterns
            for i in range(len(words) - 2):
                phrase = " ".join(words[i:i+3])
                self._routing_patterns[phrase] = category

        logger.info(f"Learned routing pattern for category: {category.value}")

    async def _fallback_execution(
        self,
        task_description: str,
        context: Dict[str, Any],
        error: str
    ) -> OrchestrationResult:
        """Fallback execution when swarm fails"""
        from core.reasoning import TrajectoryReward

        logger.warning(f"Using fallback execution due to: {error}")

        # Create minimal trajectory for fallback
        from core.reasoning import ReasoningTrajectory
        fallback_trajectory = ReasoningTrajectory(
            trajectory_id=f"fallback_{hash(task_description) % 10000}",
            problem=task_description,
            steps=[],
            final_answer=f"Task execution failed: {error}",
            success=False,
            metadata={"fallback": True}
        )

        # Create minimal result with TrajectoryReward
        fallback_result = SwarmResult(
            task_id=f"fallback_{hash(task_description) % 10000}",
            success=False,
            output={"error": error, "message": "Task execution failed - using fallback"},
            trajectory=fallback_trajectory,
            quality=TrajectoryReward(
                trajectory_id=f"fallback_{hash(task_description) % 10000}",
                overall_score=0.44,
                step_scores=[],
                temporal_coherence=0.5,
                efficiency_bonus=0.0,
                penalty=0.56,  # High penalty for fallback
                final_score=0.44,
                metadata={"fallback": True}
            ),
            agents_involved=0,
            execution_time=0.1,
            patterns_used=[],
            metadata={"fallback": True, "error": error}
        )

        return OrchestrationResult(
            task_id=fallback_result.task_id,
            task_category=TaskCategory.UNKNOWN,
            swarm_used="fallback",
            swarm_result=fallback_result,
            routing_confidence=0.0,
            execution_time=0.1,
            quality_score=0.44,
            learned_patterns=0,
            metadata={"fallback": True, "error": error}
        )

    async def _collect_execution_feedback(
        self,
        task_description: str,
        swarm_name: str,
        swarm_result: SwarmResult
    ):
        """
        Collect feedback from task execution for continuous learning

        Phase 1 Enhancement: This creates training data for RLHF and historical learning

        Args:
            task_description: Original task description
            swarm_name: Which swarm executed the task
            swarm_result: Execution result
        """
        try:
            # Prepare execution result dict
            execution_result = {
                'success': swarm_result.success,
                'compilation_success': swarm_result.metadata.get('compilation_success'),
                'test_pass_rate': swarm_result.metadata.get('test_pass_rate'),
                'error_count': swarm_result.metadata.get('error_count', 0)
            }

            # Collect feedback
            await self.feedback_collector.collect_execution_feedback(
                task_id=swarm_result.task_id,
                prompt=task_description,
                output=str(swarm_result.output),
                swarm_used=swarm_name,
                execution_result=execution_result,
                quality_score=swarm_result.quality.final_score,
                execution_time=swarm_result.execution_time
            )

            logger.debug(f"Collected feedback for task {swarm_result.task_id}")

        except TaskExecutionError as e:
            logger.warning(f"Failed to collect feedback: {e}")

    def get_available_swarms(self) -> List[str]:
        """Get list of available swarms"""
        return [
            "requirements",
            "architecture",
            "implementation",
            "testing",
            "code_review",
            "debugging",
            "deployment"
        ]

    def get_routing_stats(self) -> Dict[str, Any]:
        """Get routing statistics"""
        return {
            "learned_patterns": len(self._routing_patterns),
            "cached_swarms": len(self._swarms_cache),
            "available_swarms": len(self.get_available_swarms()),
            "categories": [cat.value for cat in TaskCategory]
        }


# Singleton instance
_orchestrator: Optional[SwarmOrchestrator] = None


def get_swarm_orchestrator() -> SwarmOrchestrator:
    """
    Get or create singleton SwarmOrchestrator

    Returns:
        Global SwarmOrchestrator instance
    """
    global _orchestrator
    if _orchestrator is None:
        _orchestrator = SwarmOrchestrator()
    return _orchestrator
