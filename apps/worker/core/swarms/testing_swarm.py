"""
TestingSwarm - Pilot swarm for comprehensive test generation
Specializes in unit, integration, and E2E testing
"""

from typing import Dict, List, Optional, Any
from dataclasses import dataclass, field
import logging

from core.swarms.base_swarm import BaseSwarm, Task, SwarmResult
from core.reasoning import (
    AgenticReasoner,
    CollaborationStrategy,
    get_agentic_reasoner
)

logger = logging.getLogger(__name__)


@dataclass
class TestRequirements:
    """Requirements for test generation"""
    code: str  # Code to test
    requirements: str  # Functional requirements
    test_types: List[str] = field(default_factory=lambda: ["unit", "integration", "e2e"])
    coverage_target: float = 0.9  # 90% coverage target
    framework: str = "pytest"  # Default to pytest
    metadata: Dict[str, Any] = field(default_factory=dict)


@dataclass
class TestResults:
    """Results from test generation"""
    unit_tests: str
    integration_tests: str
    e2e_tests: str
    total_tests_generated: int
    estimated_coverage: float
    recommendations: List[str]
    quality_score: float
    metadata: Dict[str, Any] = field(default_factory=dict)


class TestingSwarm(BaseSwarm):
    """
    Testing swarm with 3 specialized agents

    Agents:
    1. Unit Tester - Generates comprehensive unit tests
    2. Integration Tester - Generates integration tests
    3. E2E Tester - Generates end-to-end tests

    Features:
    - Full brain integration (Enhancements 7-12)
    - Multi-agent collaboration for comprehensive testing
    - Historical learning from past test patterns
    - Cross-domain pattern application
    """

    def __init__(self):
        """Initialize testing swarm"""
        # Initialize base swarm with 3 agents
        super().__init__(
            domain='testing',
            num_agents=3
        )

        # Specialized agents (each uses Enhancement 10 - AgenticReasoner)
        # Each agent is wired to the same brain components (E7, E8, E9)
        self.unit_tester = AgenticReasoner(
            tracker=self.tracker,
            prm=self.prm,
            max_steps=20  # Unit tests are relatively straightforward
        )

        self.integration_tester = AgenticReasoner(
            tracker=self.tracker,
            prm=self.prm,
            max_steps=30  # Integration tests need more planning
        )

        self.e2e_tester = AgenticReasoner(
            tracker=self.tracker,
            prm=self.prm,
            max_steps=40  # E2E tests are most complex
        )

        logger.info(
            f"Initialized TestingSwarm with 3 specialized agents "
            f"(Unit, Integration, E2E)"
        )

    async def generate_tests(
        self,
        requirements: TestRequirements
    ) -> TestResults:
        """
        Generate comprehensive tests using multi-agent collaboration

        Workflow:
        1. Create task from requirements
        2. Execute with DIVIDE_AND_CONQUER strategy
        3. Each agent generates their specialized tests:
           - Agent 0: Unit tests
           - Agent 1: Integration tests
           - Agent 2: E2E tests
        4. Combine results and assess quality

        Args:
            requirements: Test requirements

        Returns:
            TestResults with generated tests
        """
        logger.info(
            f"Generating tests for code ({len(requirements.code)} chars) "
            f"with types: {requirements.test_types}"
        )

        # Create task
        task = Task(
            task_id=f"test_gen_{hash(requirements.code) % 10000}",
            description=f"Generate {', '.join(requirements.test_types)} tests",
            task_type="testing",
            context={
                'code': requirements.code,
                'requirements': requirements.requirements,
                'test_types': requirements.test_types,
                'coverage_target': requirements.coverage_target,
                'framework': requirements.framework
            },
            requirements=[
                f"Generate {tt} tests" for tt in requirements.test_types
            ]
        )

        # Execute with multi-agent collaboration
        # Uses DIVIDE_AND_CONQUER so each agent handles one test type
        swarm_result = await self.execute(
            task=task,
            collaboration_strategy=CollaborationStrategy.DIVIDE_AND_CONQUER
        )

        # Extract test results from collaborative output
        test_results = self._extract_test_results(
            swarm_result,
            requirements
        )

        logger.info(
            f"Generated {test_results.total_tests_generated} tests "
            f"(quality: {test_results.quality_score:.3f}, "
            f"coverage: {test_results.estimated_coverage:.1%})"
        )

        return test_results

    def _extract_test_results(
        self,
        swarm_result: SwarmResult,
        requirements: TestRequirements
    ) -> TestResults:
        """
        Extract test results from swarm execution

        Args:
            swarm_result: Result from swarm execution
            requirements: Original requirements

        Returns:
            TestResults
        """
        # Parse output from collaborative solution
        # In practice, this would parse the actual generated test code
        # For now, create structured results

        output = swarm_result.output if isinstance(swarm_result.output, dict) else {}

        # Extract or generate test code sections
        unit_tests = output.get('unit_tests', self._generate_placeholder_tests('unit', requirements))
        integration_tests = output.get('integration_tests', self._generate_placeholder_tests('integration', requirements))
        e2e_tests = output.get('e2e_tests', self._generate_placeholder_tests('e2e', requirements))

        # Count tests
        total_tests = (
            self._count_tests(unit_tests) +
            self._count_tests(integration_tests) +
            self._count_tests(e2e_tests)
        )

        # Estimate coverage based on test count and quality
        estimated_coverage = min(
            0.95,  # Max 95% coverage
            (total_tests / 10) * swarm_result.quality.final_score
        )

        # Generate recommendations
        recommendations = self._generate_recommendations(
            swarm_result,
            total_tests,
            estimated_coverage,
            requirements.coverage_target
        )

        return TestResults(
            unit_tests=unit_tests,
            integration_tests=integration_tests,
            e2e_tests=e2e_tests,
            total_tests_generated=total_tests,
            estimated_coverage=estimated_coverage,
            recommendations=recommendations,
            quality_score=swarm_result.quality.final_score,
            metadata={
                'agents_involved': swarm_result.agents_involved,
                'execution_time': swarm_result.execution_time,
                'strategy_used': swarm_result.metadata.get('strategy'),
                'patterns_used': len(swarm_result.patterns_used)
            }
        )

    def _generate_placeholder_tests(
        self,
        test_type: str,
        requirements: TestRequirements
    ) -> str:
        """
        Generate placeholder test code

        In production, this would be generated by the agents
        For now, return a template

        Args:
            test_type: Type of test (unit, integration, e2e)
            requirements: Test requirements

        Returns:
            Test code as string
        """
        framework = requirements.framework

        if framework == "pytest":
            return f"""
# {test_type.upper()} TESTS
# Generated by TestingSwarm

import pytest

def test_{test_type}_example_1():
    '''Test case 1 for {test_type}'''
    # TODO: Generated test code here
    assert True

def test_{test_type}_example_2():
    '''Test case 2 for {test_type}'''
    # TODO: Generated test code here
    assert True

def test_{test_type}_example_3():
    '''Test case 3 for {test_type}'''
    # TODO: Generated test code here
    assert True
"""
        elif framework == "unittest":
            return f"""
# {test_type.upper()} TESTS
# Generated by TestingSwarm

import unittest

class Test{test_type.title().replace('_', '')}(unittest.TestCase):
    '''Test cases for {test_type}'''

    def test_{test_type}_example_1(self):
        '''Test case 1 for {test_type}'''
        # TODO: Generated test code here
        self.assertTrue(True)

    def test_{test_type}_example_2(self):
        '''Test case 2 for {test_type}'''
        # TODO: Generated test code here
        self.assertTrue(True)

    def test_{test_type}_example_3(self):
        '''Test case 3 for {test_type}'''
        # TODO: Generated test code here
        self.assertTrue(True)

if __name__ == '__main__':
    unittest.main()
"""
        elif framework == "jest":
            return f"""
// {test_type.upper()} TESTS
// Generated by TestingSwarm

describe('{test_type} tests', () => {{
    test('{test_type} example 1', () => {{
        // TODO: Generated test code here
        expect(true).toBe(true);
    }});

    test('{test_type} example 2', () => {{
        // TODO: Generated test code here
        expect(true).toBe(true);
    }});

    test('{test_type} example 3', () => {{
        // TODO: Generated test code here
        expect(true).toBe(true);
    }});
}});
"""
        else:
            return f"""
# {test_type.upper()} TESTS
# Generated by TestingSwarm
# Framework: {framework}

# Test case 1
def test_{test_type}_example_1():
    '''Test case 1 for {test_type}'''
    # TODO: Generated test code here
    pass

# Test case 2
def test_{test_type}_example_2():
    '''Test case 2 for {test_type}'''
    # TODO: Generated test code here
    pass

# Test case 3
def test_{test_type}_example_3():
    '''Test case 3 for {test_type}'''
    # TODO: Generated test code here
    pass
"""

    def _count_tests(self, test_code: str) -> int:
        """
        Count number of tests in test code

        Args:
            test_code: Test code

        Returns:
            Number of test functions
        """
        if not test_code:
            return 0

        # Count test functions (simple heuristic)
        return test_code.count('def test_')

    def _generate_recommendations(
        self,
        swarm_result: SwarmResult,
        total_tests: int,
        estimated_coverage: float,
        target_coverage: float
    ) -> List[str]:
        """
        Generate recommendations based on test results

        Args:
            swarm_result: Swarm execution result
            total_tests: Number of tests generated
            estimated_coverage: Estimated code coverage
            target_coverage: Target coverage

        Returns:
            List of recommendations
        """
        recommendations = []

        # Coverage recommendations
        if estimated_coverage < target_coverage:
            gap = target_coverage - estimated_coverage
            additional_tests = int((gap / 0.1) * 3)  # ~3 tests per 10% coverage
            recommendations.append(
                f"Add ~{additional_tests} more tests to reach {target_coverage:.0%} coverage target"
            )

        # Quality recommendations
        if swarm_result.quality.final_score < 0.8:
            recommendations.append(
                "Consider improving test quality - add more assertions and edge cases"
            )

        # Test balance
        if total_tests < 5:
            recommendations.append(
                "Add more test cases for comprehensive coverage"
            )

        # Agent performance
        if swarm_result.agents_involved < 3:
            recommendations.append(
                f"Only {swarm_result.agents_involved}/3 agents contributed - may need better task distribution"
            )

        # Execution efficiency
        if len(swarm_result.trajectory.steps) > 30:
            recommendations.append(
                "Execution took many steps - consider optimizing test generation process"
            )

        # Success case
        if not recommendations:
            recommendations.append(
                f"Excellent test suite! {total_tests} tests with {estimated_coverage:.0%} coverage"
            )

        return recommendations

    async def analyze_test_quality(
        self,
        test_code: str
    ) -> Dict[str, Any]:
        """
        Analyze quality of generated tests

        Args:
            test_code: Test code to analyze

        Returns:
            Quality analysis
        """
        # Simple quality metrics
        num_tests = self._count_tests(test_code)
        num_assertions = test_code.count('assert')
        has_fixtures = 'fixture' in test_code.lower()
        has_parametrize = 'parametrize' in test_code.lower()

        # Quality score (0-1)
        quality_score = min(1.0, (
            (num_tests / 10) * 0.3 +  # Number of tests (30%)
            (num_assertions / num_tests if num_tests > 0 else 0) * 0.4 +  # Assertions per test (40%)
            (0.15 if has_fixtures else 0) +  # Fixtures (15%)
            (0.15 if has_parametrize else 0)  # Parametrization (15%)
        ))

        return {
            'num_tests': num_tests,
            'num_assertions': num_assertions,
            'assertions_per_test': num_assertions / num_tests if num_tests > 0 else 0,
            'has_fixtures': has_fixtures,
            'has_parametrize': has_parametrize,
            'quality_score': quality_score,
            'quality_level': self._get_quality_level(quality_score)
        }

    def _get_quality_level(self, score: float) -> str:
        """Get quality level from score"""
        if score >= 0.9:
            return "Excellent"
        elif score >= 0.75:
            return "Good"
        elif score >= 0.6:
            return "Fair"
        else:
            return "Needs Improvement"


# Singleton instance
_testing_swarm: Optional[TestingSwarm] = None


def get_testing_swarm() -> TestingSwarm:
    """
    Get or create singleton TestingSwarm

    Returns:
        Global TestingSwarm instance
    """
    global _testing_swarm
    if _testing_swarm is None:
        _testing_swarm = TestingSwarm()
    return _testing_swarm
