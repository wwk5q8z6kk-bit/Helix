"""
RequirementsSwarm - Core swarm for requirements analysis and validation
Specializes in gathering, analyzing, and validating software requirements
"""

from typing import Dict, List, Optional, Any
from dataclasses import dataclass, field
from enum import Enum
import logging

from core.swarms.base_swarm import BaseSwarm, Task, SwarmResult
from core.reasoning import (
    AgenticReasoner,
    CollaborationStrategy,
    get_agentic_reasoner
)

logger = logging.getLogger(__name__)


class RequirementType(Enum):
    """Types of requirements"""
    FUNCTIONAL = "functional"
    NON_FUNCTIONAL = "non_functional"
    BUSINESS = "business"
    TECHNICAL = "technical"
    USER = "user"
    SYSTEM = "system"


class RequirementPriority(Enum):
    """Priority levels for requirements"""
    CRITICAL = "critical"
    HIGH = "high"
    MEDIUM = "medium"
    LOW = "low"


@dataclass
class Requirement:
    """Single software requirement"""
    req_id: str
    title: str
    description: str
    req_type: RequirementType
    priority: RequirementPriority
    acceptance_criteria: List[str]
    dependencies: List[str] = field(default_factory=list)
    stakeholders: List[str] = field(default_factory=list)
    validation_status: str = "pending"  # pending, validated, rejected
    validation_notes: Optional[str] = None


@dataclass
class RequirementsInput:
    """Input for requirements analysis"""
    project_description: str
    stakeholders: List[str] = field(default_factory=list)
    constraints: List[str] = field(default_factory=list)
    existing_requirements: List[str] = field(default_factory=list)
    business_goals: List[str] = field(default_factory=list)
    metadata: Dict[str, Any] = field(default_factory=dict)


@dataclass
class RequirementsOutput:
    """Output from requirements analysis"""
    functional_requirements: List[Requirement]
    non_functional_requirements: List[Requirement]
    business_requirements: List[Requirement]
    technical_requirements: List[Requirement]
    total_requirements: int
    critical_count: int
    high_count: int
    medium_count: int
    low_count: int
    completeness_score: float  # 0-1
    clarity_score: float  # 0-1
    consistency_score: float  # 0-1
    overall_quality: float  # 0-1
    recommendations: List[str]
    missing_areas: List[str]
    quality_score: float
    metadata: Dict[str, Any] = field(default_factory=dict)


class RequirementsSwarm(BaseSwarm):
    """
    Requirements analysis swarm with 3 specialized agents

    Agents:
    1. Requirements Gatherer - Elicits and documents requirements
    2. Requirements Analyzer - Analyzes completeness and consistency
    3. Requirements Validator - Validates against best practices

    Features:
    - Full brain integration (Enhancements 7-12)
    - SEQUENTIAL_REFINEMENT collaboration (iterative improvement)
    - Historical learning from past requirements analysis
    - Cross-domain pattern application from other projects
    """

    def __init__(self):
        """Initialize requirements swarm"""
        # Initialize base swarm with 3 agents
        super().__init__(
            domain='requirements',
            num_agents=3
        )

        # Specialized agents (each uses Enhancement 10 - AgenticReasoner)
        self.gatherer = AgenticReasoner(
            tracker=self.tracker,
            prm=self.prm,
            max_steps=30  # Requirements gathering is thorough
        )

        self.analyzer = AgenticReasoner(
            tracker=self.tracker,
            prm=self.prm,
            max_steps=25  # Analysis of gathered requirements
        )

        self.validator = AgenticReasoner(
            tracker=self.tracker,
            prm=self.prm,
            max_steps=20  # Validation against criteria
        )

        logger.info(
            f"Initialized RequirementsSwarm with 3 specialized agents "
            f"(Gatherer, Analyzer, Validator)"
        )

    async def analyze_requirements(
        self,
        input_data: RequirementsInput
    ) -> RequirementsOutput:
        """
        Perform comprehensive requirements analysis

        Workflow:
        1. Create task from input
        2. Execute with SEQUENTIAL_REFINEMENT strategy
        3. Agents work iteratively:
           - Agent 0 (Gatherer): Elicit and document requirements
           - Agent 1 (Analyzer): Analyze for completeness/consistency
           - Agent 2 (Validator): Validate against best practices
        4. Generate structured requirements document

        Args:
            input_data: Requirements input with project description

        Returns:
            RequirementsOutput with analyzed requirements
        """
        logger.info(
            f"Analyzing requirements for: {input_data.project_description[:100]}..."
        )

        # Create task
        task = Task(
            task_id=f"req_analysis_{hash(input_data.project_description) % 10000}",
            description=f"Analyze requirements for: {input_data.project_description}",
            task_type="requirements_analysis",
            context={
                'project_description': input_data.project_description,
                'stakeholders': input_data.stakeholders,
                'constraints': input_data.constraints,
                'existing_requirements': input_data.existing_requirements,
                'business_goals': input_data.business_goals
            },
            requirements=[
                "Gather functional requirements",
                "Gather non-functional requirements",
                "Analyze completeness and consistency",
                "Validate against best practices"
            ]
        )

        # Execute with sequential refinement (agents refine each other's work)
        swarm_result = await self.execute(
            task=task,
            collaboration_strategy=CollaborationStrategy.SEQUENTIAL_REFINEMENT
        )

        # Extract requirements from collaborative output
        requirements_output = self._extract_requirements(
            swarm_result,
            input_data
        )

        logger.info(
            f"Requirements analysis complete: {requirements_output.total_requirements} requirements "
            f"(quality: {requirements_output.overall_quality:.3f})"
        )

        return requirements_output

    def _extract_requirements(
        self,
        swarm_result: SwarmResult,
        input_data: RequirementsInput
    ) -> RequirementsOutput:
        """
        Extract structured requirements from swarm execution

        Args:
            swarm_result: Result from swarm execution
            input_data: Original input

        Returns:
            RequirementsOutput with structured requirements
        """
        # Parse output from collaborative solution
        # In practice, this would parse actual requirements
        # For now, generate structured examples

        output = swarm_result.output if isinstance(swarm_result.output, dict) else {}

        # Generate sample requirements based on quality
        functional_reqs = self._generate_sample_requirements(
            'functional',
            input_data,
            swarm_result.quality.final_score
        )
        non_functional_reqs = self._generate_sample_requirements(
            'non_functional',
            input_data,
            swarm_result.quality.final_score
        )
        business_reqs = self._generate_sample_requirements(
            'business',
            input_data,
            swarm_result.quality.final_score
        )
        technical_reqs = self._generate_sample_requirements(
            'technical',
            input_data,
            swarm_result.quality.final_score
        )

        # Combine all requirements
        all_requirements = (
            functional_reqs + non_functional_reqs +
            business_reqs + technical_reqs
        )

        # Count by priority
        critical_count = sum(1 for r in all_requirements if r.priority == RequirementPriority.CRITICAL)
        high_count = sum(1 for r in all_requirements if r.priority == RequirementPriority.HIGH)
        medium_count = sum(1 for r in all_requirements if r.priority == RequirementPriority.MEDIUM)
        low_count = sum(1 for r in all_requirements if r.priority == RequirementPriority.LOW)

        # Calculate quality metrics
        completeness_score = min(1.0, len(all_requirements) / 20)  # 20+ requirements = complete
        clarity_score = swarm_result.quality.clarity if hasattr(swarm_result.quality, 'clarity') else swarm_result.quality.final_score
        consistency_score = swarm_result.quality.coherence if hasattr(swarm_result.quality, 'coherence') else swarm_result.quality.final_score
        overall_quality = (completeness_score + clarity_score + consistency_score) / 3

        # Generate recommendations
        recommendations = self._generate_recommendations(
            all_requirements,
            completeness_score,
            swarm_result
        )

        # Identify missing areas
        missing_areas = self._identify_missing_areas(
            all_requirements,
            input_data
        )

        return RequirementsOutput(
            functional_requirements=functional_reqs,
            non_functional_requirements=non_functional_reqs,
            business_requirements=business_reqs,
            technical_requirements=technical_reqs,
            total_requirements=len(all_requirements),
            critical_count=critical_count,
            high_count=high_count,
            medium_count=medium_count,
            low_count=low_count,
            completeness_score=completeness_score,
            clarity_score=clarity_score,
            consistency_score=consistency_score,
            overall_quality=overall_quality,
            recommendations=recommendations,
            missing_areas=missing_areas,
            quality_score=swarm_result.quality.final_score,
            metadata={
                'agents_involved': swarm_result.agents_involved,
                'execution_time': swarm_result.execution_time,
                'strategy_used': swarm_result.metadata.get('strategy'),
                'patterns_used': len(swarm_result.patterns_used)
            }
        )

    def _generate_sample_requirements(
        self,
        req_type: str,
        input_data: RequirementsInput,
        quality: float
    ) -> List[Requirement]:
        """
        Generate sample requirements (production would use LLM)

        Args:
            req_type: Type of requirement
            input_data: Input data
            quality: Quality score

        Returns:
            List of requirements
        """
        requirements = []

        # Higher quality = more requirements
        num_reqs = max(2, int(5 * quality))

        if req_type == 'functional':
            templates = [
                ("User Authentication", "System shall provide secure user authentication", RequirementPriority.CRITICAL),
                ("Data Storage", "System shall store user data securely", RequirementPriority.HIGH),
                ("Search Functionality", "Users shall be able to search records", RequirementPriority.MEDIUM),
                ("Export Data", "Users shall be able to export data to CSV", RequirementPriority.LOW),
                ("Notifications", "System shall send email notifications", RequirementPriority.MEDIUM),
            ]
        elif req_type == 'non_functional':
            templates = [
                ("Performance", "System shall respond within 2 seconds", RequirementPriority.HIGH),
                ("Scalability", "System shall support 10,000 concurrent users", RequirementPriority.HIGH),
                ("Availability", "System shall have 99.9% uptime", RequirementPriority.CRITICAL),
                ("Security", "System shall encrypt data at rest and in transit", RequirementPriority.CRITICAL),
                ("Usability", "Interface shall be accessible (WCAG 2.1)", RequirementPriority.MEDIUM),
            ]
        elif req_type == 'business':
            templates = [
                ("ROI", "Project shall achieve positive ROI within 12 months", RequirementPriority.CRITICAL),
                ("User Adoption", "System shall have 80% user adoption rate", RequirementPriority.HIGH),
                ("Cost Reduction", "System shall reduce operational costs by 20%", RequirementPriority.HIGH),
                ("Market Share", "Product shall capture 5% market share", RequirementPriority.MEDIUM),
                ("Customer Satisfaction", "Achieve 4.5/5 customer satisfaction", RequirementPriority.MEDIUM),
            ]
        else:  # technical
            templates = [
                ("Architecture", "Use microservices architecture", RequirementPriority.HIGH),
                ("Database", "PostgreSQL for relational data", RequirementPriority.HIGH),
                ("API", "RESTful API with OpenAPI specification", RequirementPriority.MEDIUM),
                ("Deployment", "Containerized with Docker/Kubernetes", RequirementPriority.MEDIUM),
                ("Monitoring", "Implement comprehensive logging and monitoring", RequirementPriority.HIGH),
            ]

        for i, (title, desc, priority) in enumerate(templates[:num_reqs]):
            req = Requirement(
                req_id=f"{req_type.upper()}-{i+1:03d}",
                title=title,
                description=desc,
                req_type=RequirementType(req_type),
                priority=priority,
                acceptance_criteria=[
                    f"Criterion 1 for {title}",
                    f"Criterion 2 for {title}"
                ],
                stakeholders=input_data.stakeholders[:2] if input_data.stakeholders else ["Product Owner"],
                validation_status="validated" if quality > 0.7 else "pending"
            )
            requirements.append(req)

        return requirements

    def _generate_recommendations(
        self,
        requirements: List[Requirement],
        completeness_score: float,
        swarm_result: SwarmResult
    ) -> List[str]:
        """Generate recommendations for requirements improvement"""
        recommendations = []

        # Completeness recommendations
        if completeness_score < 0.5:
            recommendations.append(
                f"⚠️ Requirements coverage is low ({completeness_score:.0%}) - gather more detailed requirements"
            )

        # Priority recommendations
        critical_count = sum(1 for r in requirements if r.priority == RequirementPriority.CRITICAL)
        if critical_count == 0:
            recommendations.append(
                "⚠️ No critical requirements identified - ensure mission-critical needs are captured"
            )

        # Type coverage
        req_types = set(r.req_type for r in requirements)
        if RequirementType.NON_FUNCTIONAL not in req_types:
            recommendations.append(
                "⚠️ Missing non-functional requirements (performance, security, etc.)"
            )

        # Quality recommendations
        if swarm_result.quality.final_score < 0.7:
            recommendations.append(
                "⚠️ Overall quality below threshold - review and refine requirements"
            )

        # Success case
        if not recommendations:
            recommendations.append(
                f"✅ Excellent requirements! {len(requirements)} well-defined requirements captured"
            )

        return recommendations

    def _identify_missing_areas(
        self,
        requirements: List[Requirement],
        input_data: RequirementsInput
    ) -> List[str]:
        """Identify missing requirement areas"""
        missing = []

        # Check for standard areas
        req_types = set(r.req_type for r in requirements)

        if RequirementType.FUNCTIONAL not in req_types:
            missing.append("Functional requirements")
        if RequirementType.NON_FUNCTIONAL not in req_types:
            missing.append("Non-functional requirements (performance, security)")
        if RequirementType.BUSINESS not in req_types:
            missing.append("Business requirements and KPIs")
        if RequirementType.TECHNICAL not in req_types:
            missing.append("Technical constraints and architecture requirements")

        # Check for common categories
        titles = [r.title.lower() for r in requirements]

        if not any('auth' in t or 'login' in t for t in titles):
            missing.append("Authentication and authorization requirements")
        if not any('secur' in t for t in titles):
            missing.append("Security requirements")
        if not any('perform' in t or 'speed' in t for t in titles):
            missing.append("Performance requirements")

        return missing


# Singleton instance
_requirements_swarm: Optional[RequirementsSwarm] = None


def get_requirements_swarm() -> RequirementsSwarm:
    """
    Get or create singleton RequirementsSwarm

    Returns:
        Global RequirementsSwarm instance
    """
    global _requirements_swarm
    if _requirements_swarm is None:
        _requirements_swarm = RequirementsSwarm()
    return _requirements_swarm
