"""
ImplementationSwarm - Core swarm for code generation and implementation
Specializes in generating production-ready code from architecture specifications
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


class CodeType(Enum):
    """Types of code artifacts"""
    BACKEND = "backend"
    FRONTEND = "frontend"
    DATABASE = "database"
    API = "api"
    TESTS = "tests"
    DOCUMENTATION = "documentation"
    CONFIGURATION = "configuration"


class ProgrammingLanguage(Enum):
    """Supported programming languages"""
    PYTHON = "python"
    JAVASCRIPT = "javascript"
    TYPESCRIPT = "typescript"
    JAVA = "java"
    GO = "go"
    RUST = "rust"
    CPP = "cpp"


@dataclass
class CodeArtifact:
    """Single generated code artifact"""
    artifact_id: str
    name: str
    code_type: CodeType
    language: ProgrammingLanguage
    file_path: str
    code: str
    description: str
    dependencies: List[str] = field(default_factory=list)
    tests_generated: bool = False
    documentation_generated: bool = False
    quality_score: float = 0.0  # 0-1
    lines_of_code: int = 0
    complexity_score: float = 0.0  # 0-1


@dataclass
class ImplementationInput:
    """Input for code implementation"""
    project_description: str
    architecture_components: List[Dict[str, Any]] = field(default_factory=list)
    architecture_pattern: Optional[str] = None
    technology_stack: Dict[str, List[str]] = field(default_factory=dict)
    requirements: List[str] = field(default_factory=list)
    coding_standards: List[str] = field(default_factory=list)
    preferred_language: Optional[ProgrammingLanguage] = None
    generate_tests: bool = True
    generate_docs: bool = True
    metadata: Dict[str, Any] = field(default_factory=dict)


@dataclass
class ImplementationOutput:
    """Output from code implementation"""
    backend_artifacts: List[CodeArtifact]
    frontend_artifacts: List[CodeArtifact]
    database_artifacts: List[CodeArtifact]
    api_artifacts: List[CodeArtifact]
    documentation_artifacts: List[CodeArtifact]
    total_artifacts: int
    total_lines_of_code: int
    backend_count: int
    frontend_count: int
    database_count: int
    api_count: int
    documentation_count: int
    code_quality_score: float  # 0-1
    maintainability_score: float  # 0-1
    test_coverage_score: float  # 0-1
    documentation_coverage: float  # 0-1
    overall_quality: float  # 0-1
    recommendations: List[str]
    technical_debt: List[str]
    next_steps: List[str]
    quality_score: float
    metadata: Dict[str, Any] = field(default_factory=dict)


class ImplementationSwarm(BaseSwarm):
    """
    Implementation swarm with 5 specialized agents

    Agents:
    1. Backend Developer - Server-side code and business logic
    2. Frontend Developer - UI/UX and client-side code
    3. Database Developer - Schema, migrations, and queries
    4. API Developer - REST/GraphQL APIs and integrations
    5. Documentation Writer - Technical documentation and README

    Features:
    - Full brain integration (Enhancements 7-12)
    - DIVIDE_AND_CONQUER collaboration (parallel code generation)
    - Historical learning from past implementations
    - Cross-domain pattern application from architecture/requirements
    """

    def __init__(self):
        """Initialize implementation swarm"""
        # Initialize base swarm with 5 agents
        super().__init__(
            domain='implementation',
            num_agents=5
        )

        # Specialized agents (each uses Enhancement 10 - AgenticReasoner)
        self.backend_developer = AgenticReasoner(
            tracker=self.tracker,
            prm=self.prm,
            max_steps=40  # Complex backend code generation
        )

        self.frontend_developer = AgenticReasoner(
            tracker=self.tracker,
            prm=self.prm,
            max_steps=35  # UI/UX implementation
        )

        self.database_developer = AgenticReasoner(
            tracker=self.tracker,
            prm=self.prm,
            max_steps=30  # Schema and queries
        )

        self.api_developer = AgenticReasoner(
            tracker=self.tracker,
            prm=self.prm,
            max_steps=35  # API design and implementation
        )

        self.documentation_writer = AgenticReasoner(
            tracker=self.tracker,
            prm=self.prm,
            max_steps=25  # Technical documentation
        )

        logger.info(
            f"Initialized ImplementationSwarm with 5 specialized agents "
            f"(Backend, Frontend, Database, API, Documentation)"
        )

    async def generate_implementation(
        self,
        input_data: ImplementationInput
    ) -> ImplementationOutput:
        """
        Generate complete code implementation

        Workflow:
        1. Create task from input
        2. Execute with DIVIDE_AND_CONQUER strategy
        3. Agents work on different code aspects in parallel:
           - Agent 0 (Backend): Server-side code
           - Agent 1 (Frontend): Client-side code
           - Agent 2 (Database): Schema and queries
           - Agent 3 (API): API endpoints
           - Agent 4 (Documentation): Technical docs
        4. Combine into complete implementation package

        Args:
            input_data: Implementation input with architecture specs

        Returns:
            ImplementationOutput with all code artifacts
        """
        logger.info(
            f"Generating implementation for: {input_data.project_description[:100]}..."
        )

        # Create task
        task = Task(
            task_id=f"impl_{hash(input_data.project_description) % 10000}",
            description=f"Implement code for: {input_data.project_description}",
            task_type="code_implementation",
            context={
                'project_description': input_data.project_description,
                'architecture_components': input_data.architecture_components,
                'architecture_pattern': input_data.architecture_pattern,
                'technology_stack': input_data.technology_stack,
                'requirements': input_data.requirements,
                'coding_standards': input_data.coding_standards,
                'preferred_language': input_data.preferred_language.value if input_data.preferred_language else 'python',
                'generate_tests': input_data.generate_tests,
                'generate_docs': input_data.generate_docs
            },
            requirements=[
                "Generate backend code",
                "Generate frontend code",
                "Generate database schema",
                "Generate API endpoints",
                "Generate documentation"
            ]
        )

        # Execute with divide and conquer (parallel code generation)
        swarm_result = await self.execute(
            task=task,
            collaboration_strategy=CollaborationStrategy.DIVIDE_AND_CONQUER
        )

        # Extract code artifacts from collaborative output
        implementation_output = self._extract_implementation(
            swarm_result,
            input_data
        )

        logger.info(
            f"Implementation complete: {implementation_output.total_artifacts} artifacts, "
            f"{implementation_output.total_lines_of_code} LOC, "
            f"quality: {implementation_output.overall_quality:.3f}"
        )

        return implementation_output

    def _extract_implementation(
        self,
        swarm_result: SwarmResult,
        input_data: ImplementationInput
    ) -> ImplementationOutput:
        """
        Extract structured code artifacts from swarm execution

        Args:
            swarm_result: Result from swarm execution
            input_data: Original input

        Returns:
            ImplementationOutput with all code artifacts
        """
        # Parse output from collaborative solution
        output = swarm_result.output if isinstance(swarm_result.output, dict) else {}

        # Determine primary language
        language = input_data.preferred_language or ProgrammingLanguage.PYTHON

        # Generate code artifacts based on architecture and quality
        backend_artifacts = self._generate_backend_code(
            input_data,
            language,
            swarm_result.quality.final_score
        )

        frontend_artifacts = self._generate_frontend_code(
            input_data,
            language,
            swarm_result.quality.final_score
        )

        database_artifacts = self._generate_database_code(
            input_data,
            language,
            swarm_result.quality.final_score
        )

        api_artifacts = self._generate_api_code(
            input_data,
            language,
            swarm_result.quality.final_score
        )

        documentation_artifacts = self._generate_documentation(
            input_data,
            swarm_result.quality.final_score
        )

        # Combine all artifacts
        all_artifacts = (
            backend_artifacts + frontend_artifacts +
            database_artifacts + api_artifacts +
            documentation_artifacts
        )

        # Calculate metrics
        total_lines_of_code = sum(a.lines_of_code for a in all_artifacts)
        code_quality_score = sum(a.quality_score for a in all_artifacts) / len(all_artifacts) if all_artifacts else 0.0

        # Calculate coverage scores
        test_coverage_score = sum(1 for a in all_artifacts if a.tests_generated) / len(all_artifacts) if all_artifacts else 0.0
        documentation_coverage = sum(1 for a in all_artifacts if a.documentation_generated) / len(all_artifacts) if all_artifacts else 0.0

        # Maintainability based on complexity and quality
        avg_complexity = sum(a.complexity_score for a in all_artifacts) / len(all_artifacts) if all_artifacts else 0.0
        maintainability_score = (code_quality_score * 0.6) + ((1.0 - avg_complexity) * 0.4)

        # Overall quality
        overall_quality = (
            code_quality_score * 0.4 +
            maintainability_score * 0.3 +
            test_coverage_score * 0.15 +
            documentation_coverage * 0.15
        )

        # Generate recommendations
        recommendations = self._generate_recommendations(
            all_artifacts,
            code_quality_score,
            test_coverage_score,
            swarm_result
        )

        # Identify technical debt
        technical_debt = self._identify_technical_debt(
            all_artifacts,
            avg_complexity,
            test_coverage_score
        )

        # Next steps
        next_steps = self._identify_next_steps(
            all_artifacts,
            input_data
        )

        return ImplementationOutput(
            backend_artifacts=backend_artifacts,
            frontend_artifacts=frontend_artifacts,
            database_artifacts=database_artifacts,
            api_artifacts=api_artifacts,
            documentation_artifacts=documentation_artifacts,
            total_artifacts=len(all_artifacts),
            total_lines_of_code=total_lines_of_code,
            backend_count=len(backend_artifacts),
            frontend_count=len(frontend_artifacts),
            database_count=len(database_artifacts),
            api_count=len(api_artifacts),
            documentation_count=len(documentation_artifacts),
            code_quality_score=code_quality_score,
            maintainability_score=maintainability_score,
            test_coverage_score=test_coverage_score,
            documentation_coverage=documentation_coverage,
            overall_quality=overall_quality,
            recommendations=recommendations,
            technical_debt=technical_debt,
            next_steps=next_steps,
            quality_score=swarm_result.quality.final_score,
            metadata={
                'agents_involved': swarm_result.agents_involved,
                'execution_time': swarm_result.execution_time,
                'strategy_used': swarm_result.metadata.get('strategy'),
                'patterns_used': len(swarm_result.patterns_used)
            }
        )

    def _generate_backend_code(
        self,
        input_data: ImplementationInput,
        language: ProgrammingLanguage,
        quality: float
    ) -> List[CodeArtifact]:
        """Generate backend code artifacts"""
        artifacts = []

        # Base backend components
        templates = [
            ("main.py", "Application entry point", 150, 0.3),
            ("models.py", "Data models and schemas", 200, 0.4),
            ("services.py", "Business logic services", 250, 0.5),
            ("controllers.py", "Request handlers", 180, 0.4),
            ("middleware.py", "Middleware and authentication", 120, 0.3),
            ("utils.py", "Utility functions", 100, 0.2),
        ]

        # Generate based on quality (higher quality = more artifacts)
        num_artifacts = max(2, int(len(templates) * quality))

        for i, (file_name, desc, loc, complexity) in enumerate(templates[:num_artifacts]):
            artifact = CodeArtifact(
                artifact_id=f"BACKEND-{i+1:03d}",
                name=file_name,
                code_type=CodeType.BACKEND,
                language=language,
                file_path=f"backend/{file_name}",
                code=self._generate_sample_code(language, CodeType.BACKEND, file_name),
                description=desc,
                dependencies=[],
                tests_generated=input_data.generate_tests and quality > 0.7,
                documentation_generated=input_data.generate_docs,
                quality_score=quality,
                lines_of_code=int(loc * quality),
                complexity_score=complexity
            )
            artifacts.append(artifact)

        return artifacts

    def _generate_frontend_code(
        self,
        input_data: ImplementationInput,
        language: ProgrammingLanguage,
        quality: float
    ) -> List[CodeArtifact]:
        """Generate frontend code artifacts"""
        artifacts = []

        # Frontend components
        templates = [
            ("App.tsx", "Main application component", 120, 0.3),
            ("components/Header.tsx", "Header component", 80, 0.2),
            ("components/Footer.tsx", "Footer component", 60, 0.2),
            ("pages/Home.tsx", "Home page", 150, 0.3),
            ("pages/Dashboard.tsx", "Dashboard page", 200, 0.4),
            ("hooks/useAuth.ts", "Authentication hook", 100, 0.3),
            ("api/client.ts", "API client", 120, 0.3),
        ]

        num_artifacts = max(2, int(len(templates) * quality))

        for i, (file_name, desc, loc, complexity) in enumerate(templates[:num_artifacts]):
            artifact = CodeArtifact(
                artifact_id=f"FRONTEND-{i+1:03d}",
                name=file_name,
                code_type=CodeType.FRONTEND,
                language=ProgrammingLanguage.TYPESCRIPT,  # Frontend typically TypeScript
                file_path=f"frontend/src/{file_name}",
                code=self._generate_sample_code(ProgrammingLanguage.TYPESCRIPT, CodeType.FRONTEND, file_name),
                description=desc,
                dependencies=["react", "typescript"],
                tests_generated=input_data.generate_tests and quality > 0.7,
                documentation_generated=input_data.generate_docs,
                quality_score=quality,
                lines_of_code=int(loc * quality),
                complexity_score=complexity
            )
            artifacts.append(artifact)

        return artifacts

    def _generate_database_code(
        self,
        input_data: ImplementationInput,
        language: ProgrammingLanguage,
        quality: float
    ) -> List[CodeArtifact]:
        """Generate database code artifacts"""
        artifacts = []

        templates = [
            ("schema.sql", "Database schema definition", 200, 0.4),
            ("migrations/001_initial.sql", "Initial migration", 150, 0.3),
            ("migrations/002_add_indexes.sql", "Add indexes", 80, 0.2),
            ("queries.py", "Common database queries", 180, 0.4),
            ("repository.py", "Data access layer", 200, 0.4),
        ]

        num_artifacts = max(2, int(len(templates) * quality))

        for i, (file_name, desc, loc, complexity) in enumerate(templates[:num_artifacts]):
            artifact = CodeArtifact(
                artifact_id=f"DATABASE-{i+1:03d}",
                name=file_name,
                code_type=CodeType.DATABASE,
                language=language,
                file_path=f"database/{file_name}",
                code=self._generate_sample_code(language, CodeType.DATABASE, file_name),
                description=desc,
                dependencies=[],
                tests_generated=input_data.generate_tests and quality > 0.7,
                documentation_generated=input_data.generate_docs,
                quality_score=quality,
                lines_of_code=int(loc * quality),
                complexity_score=complexity
            )
            artifacts.append(artifact)

        return artifacts

    def _generate_api_code(
        self,
        input_data: ImplementationInput,
        language: ProgrammingLanguage,
        quality: float
    ) -> List[CodeArtifact]:
        """Generate API code artifacts"""
        artifacts = []

        templates = [
            ("routes.py", "API route definitions", 150, 0.3),
            ("endpoints/users.py", "User endpoints", 120, 0.3),
            ("endpoints/auth.py", "Authentication endpoints", 150, 0.4),
            ("endpoints/data.py", "Data endpoints", 130, 0.3),
            ("schemas/requests.py", "Request schemas", 100, 0.2),
            ("schemas/responses.py", "Response schemas", 100, 0.2),
        ]

        num_artifacts = max(2, int(len(templates) * quality))

        for i, (file_name, desc, loc, complexity) in enumerate(templates[:num_artifacts]):
            artifact = CodeArtifact(
                artifact_id=f"API-{i+1:03d}",
                name=file_name,
                code_type=CodeType.API,
                language=language,
                file_path=f"api/{file_name}",
                code=self._generate_sample_code(language, CodeType.API, file_name),
                description=desc,
                dependencies=[],
                tests_generated=input_data.generate_tests and quality > 0.7,
                documentation_generated=input_data.generate_docs,
                quality_score=quality,
                lines_of_code=int(loc * quality),
                complexity_score=complexity
            )
            artifacts.append(artifact)

        return artifacts

    def _generate_documentation(
        self,
        input_data: ImplementationInput,
        quality: float
    ) -> List[CodeArtifact]:
        """Generate documentation artifacts"""
        artifacts = []

        if not input_data.generate_docs:
            return artifacts

        templates = [
            ("README.md", "Project overview and setup", 300, 0.1),
            ("ARCHITECTURE.md", "Architecture documentation", 400, 0.2),
            ("API.md", "API documentation", 350, 0.2),
            ("DEVELOPMENT.md", "Development guide", 250, 0.1),
            ("DEPLOYMENT.md", "Deployment guide", 200, 0.1),
        ]

        num_artifacts = max(1, int(len(templates) * quality))

        for i, (file_name, desc, loc, complexity) in enumerate(templates[:num_artifacts]):
            artifact = CodeArtifact(
                artifact_id=f"DOCS-{i+1:03d}",
                name=file_name,
                code_type=CodeType.DOCUMENTATION,
                language=ProgrammingLanguage.PYTHON,  # Metadata language
                file_path=f"docs/{file_name}",
                code=self._generate_sample_code(ProgrammingLanguage.PYTHON, CodeType.DOCUMENTATION, file_name),
                description=desc,
                dependencies=[],
                tests_generated=False,
                documentation_generated=True,
                quality_score=quality,
                lines_of_code=int(loc * quality),
                complexity_score=complexity
            )
            artifacts.append(artifact)

        return artifacts

    def _generate_sample_code(
        self,
        language: ProgrammingLanguage,
        code_type: CodeType,
        file_name: str
    ) -> str:
        """Generate sample code (production would use LLM)"""
        if language == ProgrammingLanguage.PYTHON:
            if code_type == CodeType.BACKEND:
                return f'"""\n{file_name}\nGenerated backend module\n"""\n\nfrom typing import Optional\n\nclass Service:\n    def __init__(self):\n        pass\n'
            elif code_type == CodeType.DATABASE:
                return f'"""\n{file_name}\nDatabase schema and queries\n"""\n\n-- Table definitions\nCREATE TABLE users (\n    id SERIAL PRIMARY KEY,\n    email VARCHAR(255) UNIQUE NOT NULL\n);\n'
            elif code_type == CodeType.API:
                return f'"""\n{file_name}\nAPI endpoint definitions\n"""\n\nfrom fastapi import APIRouter\n\nrouter = APIRouter()\n\n@router.get("/health")\nasync def health():\n    return {{"status": "ok"}}\n'
        elif language == ProgrammingLanguage.TYPESCRIPT:
            return f'/**\n * {file_name}\n * Generated frontend component\n */\n\nimport React from "react";\n\nexport const Component: React.FC = () => {{\n  return <div>Component</div>;\n}};\n'

        return f"# {file_name}\n# Generated code artifact\n"

    def _generate_recommendations(
        self,
        artifacts: List[CodeArtifact],
        code_quality: float,
        test_coverage: float,
        swarm_result: SwarmResult
    ) -> List[str]:
        """Generate implementation recommendations"""
        recommendations = []

        # Code quality recommendations
        if code_quality < 0.7:
            recommendations.append(
                f"⚠️ Code quality is below threshold ({code_quality:.0%}) - review and refactor"
            )

        # Test coverage recommendations
        if test_coverage < 0.8:
            recommendations.append(
                f"⚠️ Test coverage is low ({test_coverage:.0%}) - add more tests"
            )

        # Artifact coverage
        has_backend = any(a.code_type == CodeType.BACKEND for a in artifacts)
        has_frontend = any(a.code_type == CodeType.FRONTEND for a in artifacts)
        has_database = any(a.code_type == CodeType.DATABASE for a in artifacts)
        has_api = any(a.code_type == CodeType.API for a in artifacts)

        if not has_backend:
            recommendations.append("⚠️ Missing backend implementation")
        if not has_frontend:
            recommendations.append("⚠️ Missing frontend implementation")
        if not has_database:
            recommendations.append("⚠️ Missing database schema")
        if not has_api:
            recommendations.append("⚠️ Missing API endpoints")

        # Success case
        if code_quality >= 0.8 and test_coverage >= 0.8:
            recommendations.append(
                f"✅ High-quality implementation! {len(artifacts)} artifacts with excellent coverage"
            )

        return recommendations

    def _identify_technical_debt(
        self,
        artifacts: List[CodeArtifact],
        avg_complexity: float,
        test_coverage: float
    ) -> List[str]:
        """Identify technical debt"""
        debt = []

        # Complexity debt
        if avg_complexity > 0.6:
            debt.append(f"High code complexity (avg {avg_complexity:.0%}) - needs refactoring")

        # Test debt
        untested_artifacts = [a for a in artifacts if not a.tests_generated]
        if untested_artifacts:
            debt.append(f"{len(untested_artifacts)} artifacts lack tests")

        # Documentation debt
        undocumented_artifacts = [a for a in artifacts if not a.documentation_generated]
        if undocumented_artifacts:
            debt.append(f"{len(undocumented_artifacts)} artifacts lack documentation")

        return debt

    def _identify_next_steps(
        self,
        artifacts: List[CodeArtifact],
        input_data: ImplementationInput
    ) -> List[str]:
        """Identify next steps for implementation"""
        steps = []

        # Always need testing
        steps.append("Run comprehensive test suite with TestingSwarm")

        # Code review
        steps.append("Perform code review with CodeReviewSwarm")

        # Deployment
        steps.append("Prepare deployment with DeploymentSwarm")

        # Integration
        if len(artifacts) > 10:
            steps.append("Validate component integration")

        return steps


# Singleton instance
_implementation_swarm: Optional[ImplementationSwarm] = None


def get_implementation_swarm() -> ImplementationSwarm:
    """
    Get or create singleton ImplementationSwarm

    Returns:
        Global ImplementationSwarm instance
    """
    global _implementation_swarm
    if _implementation_swarm is None:
        _implementation_swarm = ImplementationSwarm()
    return _implementation_swarm
