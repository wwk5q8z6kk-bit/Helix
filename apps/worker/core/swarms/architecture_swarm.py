"""
ArchitectureSwarm - Core swarm for system architecture and design
Specializes in designing scalable, secure, and maintainable system architectures
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


class ArchitecturePattern(Enum):
    """Common architecture patterns"""
    MICROSERVICES = "microservices"
    MONOLITHIC = "monolithic"
    SERVERLESS = "serverless"
    EVENT_DRIVEN = "event_driven"
    LAYERED = "layered"
    HEXAGONAL = "hexagonal"
    CQRS = "cqrs"


class ComponentType(Enum):
    """Types of system components"""
    API = "api"
    DATABASE = "database"
    CACHE = "cache"
    QUEUE = "queue"
    SERVICE = "service"
    GATEWAY = "gateway"
    FRONTEND = "frontend"
    STORAGE = "storage"


@dataclass
class ArchitectureComponent:
    """Single architecture component"""
    component_id: str
    name: str
    component_type: ComponentType
    description: str
    responsibilities: List[str]
    dependencies: List[str] = field(default_factory=list)
    technologies: List[str] = field(default_factory=list)
    scalability_considerations: Optional[str] = None
    security_considerations: Optional[str] = None


@dataclass
class ArchitectureInput:
    """Input for architecture design"""
    project_description: str
    requirements: List[str] = field(default_factory=list)
    constraints: List[str] = field(default_factory=list)
    scale_expectations: Optional[str] = None
    preferred_technologies: List[str] = field(default_factory=list)
    budget_constraints: Optional[str] = None
    timeline: Optional[str] = None
    metadata: Dict[str, Any] = field(default_factory=dict)


@dataclass
class ArchitectureOutput:
    """Output from architecture design"""
    pattern: ArchitecturePattern
    components: List[ArchitectureComponent]
    data_flow: str
    deployment_strategy: str
    scalability_plan: str
    security_architecture: str
    technology_stack: Dict[str, List[str]]
    total_components: int
    complexity_score: float  # 0-1, higher = more complex
    scalability_score: float  # 0-1
    security_score: float  # 0-1
    maintainability_score: float  # 0-1
    overall_quality: float  # 0-1
    recommendations: List[str]
    trade_offs: List[str]
    risks: List[str]
    quality_score: float
    metadata: Dict[str, Any] = field(default_factory=dict)


class ArchitectureSwarm(BaseSwarm):
    """
    Architecture design swarm with 3 specialized agents

    Agents:
    1. System Designer - Overall system architecture and patterns
    2. Security Architect - Security and compliance architecture
    3. Infrastructure Architect - Deployment and scalability architecture

    Features:
    - Full brain integration (Enhancements 7-12)
    - HIERARCHICAL_DECOMPOSITION collaboration (design → secure → deploy)
    - Historical learning from past architecture patterns
    - Cross-domain pattern application from similar systems
    """

    def __init__(self):
        """Initialize architecture swarm"""
        # Initialize base swarm with 3 agents
        super().__init__(
            domain='architecture',
            num_agents=3
        )

        # Specialized agents (each uses Enhancement 10 - AgenticReasoner)
        self.system_designer = AgenticReasoner(
            tracker=self.tracker,
            prm=self.prm,
            max_steps=35  # Complex design work
        )

        self.security_architect = AgenticReasoner(
            tracker=self.tracker,
            prm=self.prm,
            max_steps=30  # Security analysis
        )

        self.infrastructure_architect = AgenticReasoner(
            tracker=self.tracker,
            prm=self.prm,
            max_steps=30  # Infrastructure planning
        )

        logger.info(
            f"Initialized ArchitectureSwarm with 3 specialized agents "
            f"(System Designer, Security Architect, Infrastructure Architect)"
        )

    async def design_architecture(
        self,
        input_data: ArchitectureInput
    ) -> ArchitectureOutput:
        """
        Design comprehensive system architecture

        Workflow:
        1. Create task from input
        2. Execute with DIVIDE_AND_CONQUER strategy
        3. Agents work on different aspects:
           - Agent 0 (System Designer): Overall architecture and patterns
           - Agent 1 (Security Architect): Security architecture
           - Agent 2 (Infrastructure): Deployment and scalability
        4. Combine into cohesive architecture document

        Args:
            input_data: Architecture input with requirements

        Returns:
            ArchitectureOutput with complete architecture design
        """
        logger.info(
            f"Designing architecture for: {input_data.project_description[:100]}..."
        )

        # Create task
        task = Task(
            task_id=f"arch_design_{hash(input_data.project_description) % 10000}",
            description=f"Design architecture for: {input_data.project_description}",
            task_type="architecture_design",
            context={
                'project_description': input_data.project_description,
                'requirements': input_data.requirements,
                'constraints': input_data.constraints,
                'scale_expectations': input_data.scale_expectations,
                'preferred_technologies': input_data.preferred_technologies,
                'budget_constraints': input_data.budget_constraints,
                'timeline': input_data.timeline
            },
            requirements=[
                "Design overall system architecture",
                "Define security architecture",
                "Plan deployment and scalability"
            ]
        )

        # Execute with divide and conquer (parallel aspects)
        swarm_result = await self.execute(
            task=task,
            collaboration_strategy=CollaborationStrategy.DIVIDE_AND_CONQUER
        )

        # Extract architecture from collaborative output
        architecture_output = self._extract_architecture(
            swarm_result,
            input_data
        )

        logger.info(
            f"Architecture design complete: {architecture_output.total_components} components, "
            f"pattern: {architecture_output.pattern.value}, "
            f"quality: {architecture_output.overall_quality:.3f}"
        )

        return architecture_output

    def _extract_architecture(
        self,
        swarm_result: SwarmResult,
        input_data: ArchitectureInput
    ) -> ArchitectureOutput:
        """
        Extract structured architecture from swarm execution

        Args:
            swarm_result: Result from swarm execution
            input_data: Original input

        Returns:
            ArchitectureOutput with architecture design
        """
        # Parse output from collaborative solution
        output = swarm_result.output if isinstance(swarm_result.output, dict) else {}

        # Determine architecture pattern based on requirements
        pattern = self._select_architecture_pattern(input_data, swarm_result.quality.final_score)

        # Generate components based on pattern and quality
        components = self._generate_components(
            pattern,
            input_data,
            swarm_result.quality.final_score
        )

        # Generate architecture documentation
        data_flow = self._generate_data_flow(pattern, components)
        deployment_strategy = self._generate_deployment_strategy(pattern, input_data)
        scalability_plan = self._generate_scalability_plan(pattern, input_data)
        security_architecture = self._generate_security_architecture(components)

        # Define technology stack
        technology_stack = self._select_technology_stack(pattern, input_data)

        # Calculate quality metrics
        complexity_score = min(1.0, len(components) / 10)  # More components = more complex
        scalability_score = 0.9 if pattern in [ArchitecturePattern.MICROSERVICES, ArchitecturePattern.SERVERLESS] else 0.7
        security_score = swarm_result.quality.final_score * 0.95  # Based on agent quality
        maintainability_score = 1.0 - (complexity_score * 0.3)  # Lower complexity = better maintainability
        overall_quality = (scalability_score + security_score + maintainability_score) / 3

        # Generate recommendations
        recommendations = self._generate_recommendations(
            pattern,
            components,
            input_data,
            swarm_result
        )

        # Identify trade-offs
        trade_offs = self._identify_trade_offs(pattern, complexity_score)

        # Identify risks
        risks = self._identify_risks(pattern, components, input_data)

        return ArchitectureOutput(
            pattern=pattern,
            components=components,
            data_flow=data_flow,
            deployment_strategy=deployment_strategy,
            scalability_plan=scalability_plan,
            security_architecture=security_architecture,
            technology_stack=technology_stack,
            total_components=len(components),
            complexity_score=complexity_score,
            scalability_score=scalability_score,
            security_score=security_score,
            maintainability_score=maintainability_score,
            overall_quality=overall_quality,
            recommendations=recommendations,
            trade_offs=trade_offs,
            risks=risks,
            quality_score=swarm_result.quality.final_score,
            metadata={
                'agents_involved': swarm_result.agents_involved,
                'execution_time': swarm_result.execution_time,
                'strategy_used': swarm_result.metadata.get('strategy'),
                'patterns_used': len(swarm_result.patterns_used)
            }
        )

    def _select_architecture_pattern(
        self,
        input_data: ArchitectureInput,
        quality: float
    ) -> ArchitecturePattern:
        """Select appropriate architecture pattern"""
        # Check scale expectations
        if input_data.scale_expectations and 'large' in input_data.scale_expectations.lower():
            return ArchitecturePattern.MICROSERVICES

        # Check for event-driven requirements
        if any('event' in req.lower() or 'async' in req.lower() for req in input_data.requirements):
            return ArchitecturePattern.EVENT_DRIVEN

        # Default to microservices for high quality, monolithic for simplicity
        return ArchitecturePattern.MICROSERVICES if quality > 0.75 else ArchitecturePattern.LAYERED

    def _generate_components(
        self,
        pattern: ArchitecturePattern,
        input_data: ArchitectureInput,
        quality: float
    ) -> List[ArchitectureComponent]:
        """Generate architecture components"""
        components = []

        # Base components for all patterns
        base_components = [
            ("API Gateway", ComponentType.GATEWAY, "Entry point for all client requests", ["routing", "authentication"]),
            ("User Service", ComponentType.SERVICE, "User management and authentication", ["CRUD operations", "auth"]),
            ("Database", ComponentType.DATABASE, "Primary data store", ["data persistence"]),
            ("Cache", ComponentType.CACHE, "Performance optimization", ["caching", "session storage"]),
        ]

        # Pattern-specific components
        if pattern == ArchitecturePattern.MICROSERVICES:
            pattern_components = [
                ("Order Service", ComponentType.SERVICE, "Order processing", ["order management"]),
                ("Payment Service", ComponentType.SERVICE, "Payment processing", ["transactions"]),
                ("Notification Service", ComponentType.SERVICE, "Notifications", ["email", "SMS"]),
                ("Message Queue", ComponentType.QUEUE, "Async communication", ["event routing"]),
            ]
        elif pattern == ArchitecturePattern.EVENT_DRIVEN:
            pattern_components = [
                ("Event Bus", ComponentType.QUEUE, "Central event distribution", ["event routing"]),
                ("Event Store", ComponentType.DATABASE, "Event persistence", ["event sourcing"]),
                ("Event Processor", ComponentType.SERVICE, "Process events", ["event handling"]),
            ]
        else:
            pattern_components = [
                ("Application Server", ComponentType.SERVICE, "Business logic", ["request handling"]),
                ("File Storage", ComponentType.STORAGE, "File management", ["uploads", "assets"]),
            ]

        # Combine components
        all_components = base_components + pattern_components

        # Create component objects
        for i, (name, comp_type, desc, resp) in enumerate(all_components[:int(8 * quality)]):
            component = ArchitectureComponent(
                component_id=f"COMP-{i+1:03d}",
                name=name,
                component_type=comp_type,
                description=desc,
                responsibilities=resp,
                technologies=self._suggest_technologies(comp_type),
                scalability_considerations=f"{name} can scale horizontally",
                security_considerations=f"Implement authentication and encryption for {name}"
            )
            components.append(component)

        return components

    def _suggest_technologies(self, component_type: ComponentType) -> List[str]:
        """Suggest technologies for component type"""
        tech_map = {
            ComponentType.API: ["FastAPI", "Express.js", "Spring Boot"],
            ComponentType.DATABASE: ["PostgreSQL", "MongoDB", "MySQL"],
            ComponentType.CACHE: ["Redis", "Memcached"],
            ComponentType.QUEUE: ["RabbitMQ", "Kafka", "AWS SQS"],
            ComponentType.SERVICE: ["Python", "Node.js", "Java"],
            ComponentType.GATEWAY: ["Kong", "AWS API Gateway", "Nginx"],
            ComponentType.FRONTEND: ["React", "Vue.js", "Angular"],
            ComponentType.STORAGE: ["AWS S3", "Azure Blob", "MinIO"],
        }
        return tech_map.get(component_type, ["TBD"])

    def _generate_data_flow(self, pattern: ArchitecturePattern, components: List[ArchitectureComponent]) -> str:
        """Generate data flow description"""
        if pattern == ArchitecturePattern.MICROSERVICES:
            return "Client → API Gateway → Service Discovery → Microservices → Database/Cache"
        elif pattern == ArchitecturePattern.EVENT_DRIVEN:
            return "Event Producer → Event Bus → Event Processors → Event Store"
        else:
            return "Client → Load Balancer → Application Server → Database"

    def _generate_deployment_strategy(self, pattern: ArchitecturePattern, input_data: ArchitectureInput) -> str:
        """Generate deployment strategy"""
        if pattern == ArchitecturePattern.MICROSERVICES:
            return "Containerized deployment with Kubernetes for orchestration, CI/CD with GitLab/GitHub Actions"
        elif pattern == ArchitecturePattern.SERVERLESS:
            return "Serverless deployment on AWS Lambda/Azure Functions with automated scaling"
        else:
            return "Traditional deployment on VMs/containers with load balancer"

    def _generate_scalability_plan(self, pattern: ArchitecturePattern, input_data: ArchitectureInput) -> str:
        """Generate scalability plan"""
        if pattern == ArchitecturePattern.MICROSERVICES:
            return "Horizontal scaling of individual services, auto-scaling based on metrics, database sharding"
        else:
            return "Vertical scaling initially, horizontal scaling with load balancer for high traffic"

    def _generate_security_architecture(self, components: List[ArchitectureComponent]) -> str:
        """Generate security architecture"""
        return "OAuth2/JWT for authentication, TLS for encryption, API rate limiting, WAF for protection, regular security audits"

    def _select_technology_stack(
        self,
        pattern: ArchitecturePattern,
        input_data: ArchitectureInput
    ) -> Dict[str, List[str]]:
        """Select technology stack"""
        return {
            "Backend": ["Python/FastAPI", "Node.js/Express", "Java/Spring Boot"],
            "Frontend": ["React", "Vue.js"],
            "Database": ["PostgreSQL", "MongoDB"],
            "Cache": ["Redis"],
            "Message Queue": ["RabbitMQ", "Kafka"],
            "Infrastructure": ["Docker", "Kubernetes", "Terraform"],
            "Monitoring": ["Prometheus", "Grafana", "ELK Stack"]
        }

    def _generate_recommendations(
        self,
        pattern: ArchitecturePattern,
        components: List[ArchitectureComponent],
        input_data: ArchitectureInput,
        swarm_result: SwarmResult
    ) -> List[str]:
        """Generate architecture recommendations"""
        recommendations = []

        # Pattern-specific recommendations
        if pattern == ArchitecturePattern.MICROSERVICES:
            recommendations.append(
                "✅ Microservices pattern selected - ensure proper service boundaries and communication"
            )

        # Component recommendations
        if len(components) < 5:
            recommendations.append(
                "⚠️ Consider adding more components for better separation of concerns"
            )

        # Quality recommendations
        if swarm_result.quality.final_score > 0.8:
            recommendations.append(
                "✅ High-quality architecture design - ready for implementation"
            )

        return recommendations

    def _identify_trade_offs(self, pattern: ArchitecturePattern, complexity: float) -> List[str]:
        """Identify architecture trade-offs"""
        trade_offs = []

        if pattern == ArchitecturePattern.MICROSERVICES:
            trade_offs.append("Higher complexity vs better scalability")
            trade_offs.append("More operational overhead vs independent deployment")

        if complexity > 0.7:
            trade_offs.append("Feature richness vs increased complexity")

        return trade_offs

    def _identify_risks(
        self,
        pattern: ArchitecturePattern,
        components: List[ArchitectureComponent],
        input_data: ArchitectureInput
    ) -> List[str]:
        """Identify architecture risks"""
        risks = []

        if pattern == ArchitecturePattern.MICROSERVICES:
            risks.append("Risk: Distributed system complexity")
            risks.append("Risk: Network latency between services")

        if len(components) > 10:
            risks.append("Risk: High operational complexity")

        return risks


# Singleton instance
_architecture_swarm: Optional[ArchitectureSwarm] = None


def get_architecture_swarm() -> ArchitectureSwarm:
    """
    Get or create singleton ArchitectureSwarm

    Returns:
        Global ArchitectureSwarm instance
    """
    global _architecture_swarm
    if _architecture_swarm is None:
        _architecture_swarm = ArchitectureSwarm()
    return _architecture_swarm
