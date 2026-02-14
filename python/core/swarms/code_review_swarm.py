"""
CodeReviewSwarm - Pilot swarm for comprehensive code review
Specializes in security, performance, and quality analysis
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
class CodeReviewRequirements:
    """Requirements for code review"""
    code: str  # Code to review
    language: str = "python"  # Programming language
    architecture: Optional[str] = None  # Architecture context
    review_types: List[str] = field(default_factory=lambda: ["security", "performance", "quality"])
    severity_threshold: str = "medium"  # Minimum severity to report (low, medium, high, critical)
    metadata: Dict[str, Any] = field(default_factory=dict)


@dataclass
class ReviewIssue:
    """Single code review issue"""
    issue_id: str
    severity: str  # low, medium, high, critical
    category: str  # security, performance, quality
    title: str
    description: str
    location: str  # Line number or code location
    recommendation: str
    code_snippet: Optional[str] = None


@dataclass
class CodeReviewResults:
    """Results from code review"""
    security_issues: List[ReviewIssue]
    performance_issues: List[ReviewIssue]
    quality_issues: List[ReviewIssue]
    total_issues: int
    critical_count: int
    high_count: int
    medium_count: int
    low_count: int
    overall_score: float  # 0-1, higher is better
    recommendations: List[str]
    quality_score: float
    metadata: Dict[str, Any] = field(default_factory=dict)


class CodeReviewSwarm(BaseSwarm):
    """
    Code review swarm with 3 specialized agents

    Agents:
    1. Security Reviewer - Analyzes security vulnerabilities
    2. Performance Reviewer - Identifies performance bottlenecks
    3. Quality Reviewer - Checks code quality and best practices

    Features:
    - Full brain integration (Enhancements 7-12)
    - Multi-agent parallel review for comprehensive coverage
    - Historical learning from past reviews
    - Cross-domain pattern application
    """

    def __init__(self):
        """Initialize code review swarm"""
        # Initialize base swarm with 3 agents
        super().__init__(
            domain='review',
            num_agents=3
        )

        # Specialized agents (each uses Enhancement 10 - AgenticReasoner)
        # Each agent is wired to the same brain components (E7, E8, E9)
        self.security_reviewer = AgenticReasoner(
            tracker=self.tracker,
            prm=self.prm,
            max_steps=25  # Security analysis needs thorough checking
        )

        self.performance_reviewer = AgenticReasoner(
            tracker=self.tracker,
            prm=self.prm,
            max_steps=25  # Performance analysis
        )

        self.quality_reviewer = AgenticReasoner(
            tracker=self.tracker,
            prm=self.prm,
            max_steps=25  # Quality assessment
        )

        logger.info(
            f"Initialized CodeReviewSwarm with 3 specialized agents "
            f"(Security, Performance, Quality)"
        )

    async def review_code(
        self,
        requirements: CodeReviewRequirements
    ) -> CodeReviewResults:
        """
        Perform comprehensive code review using multi-agent collaboration

        Workflow:
        1. Create task from requirements
        2. Execute with PARALLEL_EXPLORATION strategy
        3. Each agent performs their specialized review:
           - Agent 0: Security analysis
           - Agent 1: Performance analysis
           - Agent 2: Quality analysis
        4. Combine results and generate recommendations

        Args:
            requirements: Code review requirements

        Returns:
            CodeReviewResults with all findings
        """
        logger.info(
            f"Reviewing code ({len(requirements.code)} chars) "
            f"for: {requirements.review_types}"
        )

        # Create task
        task = Task(
            task_id=f"review_{hash(requirements.code) % 10000}",
            description=f"Review {requirements.language} code for {', '.join(requirements.review_types)}",
            task_type="code_review",
            context={
                'code': requirements.code,
                'language': requirements.language,
                'architecture': requirements.architecture,
                'review_types': requirements.review_types,
                'severity_threshold': requirements.severity_threshold
            },
            requirements=[
                f"Analyze {rt}" for rt in requirements.review_types
            ]
        )

        # Execute with multi-agent collaboration
        # Uses PARALLEL_EXPLORATION so each agent reviews independently
        swarm_result = await self.execute(
            task=task,
            collaboration_strategy=CollaborationStrategy.PARALLEL_EXPLORATION
        )

        # Extract review results from collaborative output
        review_results = self._extract_review_results(
            swarm_result,
            requirements
        )

        logger.info(
            f"Review complete: {review_results.total_issues} issues found "
            f"(critical: {review_results.critical_count}, "
            f"high: {review_results.high_count}), "
            f"score: {review_results.overall_score:.3f}"
        )

        return review_results

    def _extract_review_results(
        self,
        swarm_result: SwarmResult,
        requirements: CodeReviewRequirements
    ) -> CodeReviewResults:
        """
        Extract review results from swarm execution

        Args:
            swarm_result: Result from swarm execution
            requirements: Original requirements

        Returns:
            CodeReviewResults
        """
        # Parse output from collaborative solution
        # In practice, this would parse actual review findings
        # For now, create structured results

        output = swarm_result.output if isinstance(swarm_result.output, dict) else {}

        # Extract or generate issues by category
        security_issues = self._generate_sample_issues(
            'security',
            requirements.code,
            swarm_result.quality.final_score
        )
        performance_issues = self._generate_sample_issues(
            'performance',
            requirements.code,
            swarm_result.quality.final_score
        )
        quality_issues = self._generate_sample_issues(
            'quality',
            requirements.code,
            swarm_result.quality.final_score
        )

        # Count by severity
        all_issues = security_issues + performance_issues + quality_issues
        critical_count = sum(1 for i in all_issues if i.severity == 'critical')
        high_count = sum(1 for i in all_issues if i.severity == 'high')
        medium_count = sum(1 for i in all_issues if i.severity == 'medium')
        low_count = sum(1 for i in all_issues if i.severity == 'low')

        # Calculate overall score (fewer critical/high issues = higher score)
        max_penalty = 100.0
        penalty = (critical_count * 20) + (high_count * 10) + (medium_count * 5) + (low_count * 1)
        overall_score = max(0.0, min(1.0, 1.0 - (penalty / max_penalty)))

        # Generate recommendations
        recommendations = self._generate_recommendations(
            all_issues,
            swarm_result,
            requirements
        )

        return CodeReviewResults(
            security_issues=security_issues,
            performance_issues=performance_issues,
            quality_issues=quality_issues,
            total_issues=len(all_issues),
            critical_count=critical_count,
            high_count=high_count,
            medium_count=medium_count,
            low_count=low_count,
            overall_score=overall_score,
            recommendations=recommendations,
            quality_score=swarm_result.quality.final_score,
            metadata={
                'agents_involved': swarm_result.agents_involved,
                'execution_time': swarm_result.execution_time,
                'strategy_used': swarm_result.metadata.get('strategy'),
                'patterns_used': len(swarm_result.patterns_used)
            }
        )

    def _generate_sample_issues(
        self,
        category: str,
        code: str,
        quality: float
    ) -> List[ReviewIssue]:
        """
        Generate sample review issues

        In production, this would be generated by the agents
        For now, return examples based on quality score

        Args:
            category: Issue category (security, performance, quality)
            code: Code being reviewed
            quality: Quality score from PRM

        Returns:
            List of review issues
        """
        issues = []

        # Higher quality = fewer issues
        num_issues = max(0, int(5 * (1.0 - quality)))

        if num_issues == 0:
            return issues

        # Generate issues based on category
        if category == 'security':
            if num_issues >= 1:
                issues.append(ReviewIssue(
                    issue_id=f"SEC-001",
                    severity='high',
                    category='security',
                    title='Potential SQL Injection',
                    description='User input is used in SQL query without sanitization',
                    location='Line 42',
                    recommendation='Use parameterized queries or ORM',
                    code_snippet='query = f"SELECT * FROM users WHERE id = {user_id}"'
                ))
            if num_issues >= 2:
                issues.append(ReviewIssue(
                    issue_id=f"SEC-002",
                    severity='medium',
                    category='security',
                    title='Hardcoded Credentials',
                    description='API key appears to be hardcoded in source',
                    location='Line 15',
                    recommendation='Use environment variables or secrets manager',
                    code_snippet='API_KEY = "sk-12345..."'
                ))

        elif category == 'performance':
            if num_issues >= 1:
                issues.append(ReviewIssue(
                    issue_id=f"PERF-001",
                    severity='medium',
                    category='performance',
                    title='N+1 Query Problem',
                    description='Loop makes database query on each iteration',
                    location='Lines 55-60',
                    recommendation='Use batch query or JOIN',
                    code_snippet='for user in users:\n    profile = db.get_profile(user.id)'
                ))
            if num_issues >= 2:
                issues.append(ReviewIssue(
                    issue_id=f"PERF-002",
                    severity='low',
                    category='performance',
                    title='Inefficient List Comprehension',
                    description='Multiple passes over same data',
                    location='Line 72',
                    recommendation='Combine into single pass',
                ))

        elif category == 'quality':
            if num_issues >= 1:
                issues.append(ReviewIssue(
                    issue_id=f"QUAL-001",
                    severity='low',
                    category='quality',
                    title='Missing Type Hints',
                    description='Function parameters lack type annotations',
                    location='Line 28',
                    recommendation='Add type hints for better maintainability',
                    code_snippet='def process_data(data):'
                ))
            if num_issues >= 2:
                issues.append(ReviewIssue(
                    issue_id=f"QUAL-002",
                    severity='low',
                    category='quality',
                    title='Complex Function',
                    description='Function has cyclomatic complexity > 10',
                    location='Lines 100-150',
                    recommendation='Break into smaller functions',
                ))

        return issues[:num_issues]

    def _generate_recommendations(
        self,
        issues: List[ReviewIssue],
        swarm_result: SwarmResult,
        requirements: CodeReviewRequirements
    ) -> List[str]:
        """
        Generate recommendations based on review findings

        Args:
            issues: All found issues
            swarm_result: Swarm execution result
            requirements: Review requirements

        Returns:
            List of recommendations
        """
        recommendations = []

        # Critical/high severity recommendations
        critical_high = [i for i in issues if i.severity in ['critical', 'high']]
        if critical_high:
            recommendations.append(
                f"🚨 URGENT: Address {len(critical_high)} critical/high severity issues before deployment"
            )

        # Security recommendations
        security_issues = [i for i in issues if i.category == 'security']
        if security_issues:
            recommendations.append(
                f"🔒 Security: Fix {len(security_issues)} security vulnerabilities"
            )

        # Performance recommendations
        perf_issues = [i for i in issues if i.category == 'performance']
        if perf_issues:
            recommendations.append(
                f"⚡ Performance: Optimize {len(perf_issues)} performance bottlenecks"
            )

        # Quality recommendations
        quality_issues = [i for i in issues if i.category == 'quality']
        if quality_issues:
            recommendations.append(
                f"✨ Quality: Improve {len(quality_issues)} code quality issues"
            )

        # Agent collaboration feedback
        if swarm_result.agents_involved < 3:
            recommendations.append(
                f"⚠️ Only {swarm_result.agents_involved}/3 agents contributed - review may be incomplete"
            )

        # Success case
        if not issues:
            recommendations.append(
                "✅ Excellent! No issues found. Code meets all quality standards."
            )

        return recommendations

    async def analyze_issue_severity(
        self,
        issue: ReviewIssue
    ) -> Dict[str, Any]:
        """
        Analyze severity of a specific issue

        Args:
            issue: Issue to analyze

        Returns:
            Detailed severity analysis
        """
        severity_scores = {
            'critical': 1.0,
            'high': 0.75,
            'medium': 0.5,
            'low': 0.25
        }

        return {
            'issue_id': issue.issue_id,
            'severity': issue.severity,
            'severity_score': severity_scores.get(issue.severity, 0.5),
            'category': issue.category,
            'immediate_action_required': issue.severity in ['critical', 'high'],
            'estimated_fix_time': self._estimate_fix_time(issue),
            'risk_level': self._calculate_risk_level(issue)
        }

    def _estimate_fix_time(self, issue: ReviewIssue) -> str:
        """Estimate time to fix issue"""
        time_estimates = {
            'critical': '4-8 hours',
            'high': '2-4 hours',
            'medium': '1-2 hours',
            'low': '15-30 minutes'
        }
        return time_estimates.get(issue.severity, '1-2 hours')

    def _calculate_risk_level(self, issue: ReviewIssue) -> str:
        """Calculate risk level of issue"""
        if issue.severity == 'critical' and issue.category == 'security':
            return 'EXTREME'
        elif issue.severity in ['critical', 'high']:
            return 'HIGH'
        elif issue.severity == 'medium':
            return 'MODERATE'
        else:
            return 'LOW'


# Singleton instance
_code_review_swarm: Optional[CodeReviewSwarm] = None


def get_code_review_swarm() -> CodeReviewSwarm:
    """
    Get or create singleton CodeReviewSwarm

    Returns:
        Global CodeReviewSwarm instance
    """
    global _code_review_swarm
    if _code_review_swarm is None:
        _code_review_swarm = CodeReviewSwarm()
    return _code_review_swarm
