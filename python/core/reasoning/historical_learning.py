"""
Historical Learning System for Helix Reasoning
Analyzes past trajectories to improve future reasoning quality

Phase 1: Historical Learning (Enhancement 12)
- Trajectory analyzer: Extract patterns from historical data
- Pattern library: Store and categorize successful patterns
- Strategy recommender: Suggest best strategy for new problems

Expected Impact: 15-25% quality improvement
"""

import sqlite3
from typing import Dict, List, Optional, Tuple, Any
from dataclasses import dataclass, field
from datetime import datetime
from collections import defaultdict
import json
import re
from pathlib import Path

from .agentic_reasoner import PlanningStrategy
from .trajectory_tracker import TrajectoryTracker
from core.exceptions_unified import InitializationError, NetworkError


@dataclass
class TrajectoryPattern:
    """A learned pattern from successful trajectories"""
    pattern_id: str
    problem_type: str  # e.g., "API design", "debugging", "architecture"
    successful_strategy: PlanningStrategy
    avg_quality: float
    avg_steps: int
    common_steps: List[str]  # Common step types/actions
    success_rate: float
    sample_count: int
    metadata: Dict[str, Any] = field(default_factory=dict)

    def to_dict(self) -> Dict[str, Any]:
        return {
            'pattern_id': self.pattern_id,
            'problem_type': self.problem_type,
            'successful_strategy': self.successful_strategy.value,
            'avg_quality': self.avg_quality,
            'avg_steps': self.avg_steps,
            'common_steps': self.common_steps,
            'success_rate': self.success_rate,
            'sample_count': self.sample_count,
            'metadata': self.metadata
        }

    @staticmethod
    def from_dict(data: Dict[str, Any]) -> 'TrajectoryPattern':
        return TrajectoryPattern(
            pattern_id=data['pattern_id'],
            problem_type=data['problem_type'],
            successful_strategy=PlanningStrategy(data['successful_strategy']),
            avg_quality=data['avg_quality'],
            avg_steps=data['avg_steps'],
            common_steps=data['common_steps'],
            success_rate=data['success_rate'],
            sample_count=data['sample_count'],
            metadata=data.get('metadata', {})
        )


@dataclass
class ProblemClassification:
    """Classification of a problem for strategy recommendation"""
    problem_type: str
    confidence: float
    keywords: List[str]
    complexity: str  # 'simple', 'medium', 'complex'
    domain: str  # e.g., 'software', 'data', 'system design'


class TrajectoryAnalyzer:
    """
    Analyzes historical trajectories to extract patterns and insights

    This component learns from past reasoning to improve future performance:
    - Identifies successful reasoning patterns
    - Correlates strategies with problem types
    - Extracts common step sequences
    """

    def __init__(self, db_path: str | None = None):
        import os
        from pathlib import Path
        if db_path is None:
            db_path = str(Path(os.environ.get("HELIX_HOME", Path.home() / ".helix")) / "data" / "dashboard.db")
        self.db_path = db_path
        self.patterns: Dict[str, TrajectoryPattern] = {}

    def analyze_all_trajectories(self) -> Dict[str, Any]:
        """
        Analyze all trajectories in the database

        Returns:
            Statistics and patterns discovered
        """
        conn = sqlite3.connect(self.db_path)
        cursor = conn.cursor()

        # Get all successful trajectories
        cursor.execute("""
            SELECT trajectory_id, problem, success, efficiency_score, metadata
            FROM trajectories
            WHERE success = 1
            ORDER BY created_at DESC
        """)

        trajectories = cursor.fetchall()

        # Get steps for each trajectory
        trajectory_data = []
        for traj in trajectories:
            traj_id, problem, success, efficiency, metadata_str = traj

            # Get steps
            cursor.execute("""
                SELECT step_number, step_type, confidence, content, metadata
                FROM steps
                WHERE trajectory_id = ?
                ORDER BY step_number
            """, (traj_id,))

            steps = cursor.fetchall()

            # Parse metadata
            metadata = {}
            if metadata_str:
                try:
                    metadata = json.loads(metadata_str)
                except (json.JSONDecodeError, ValueError, TypeError) as e:
                    # Expected: Invalid JSON format or wrong type
                    logger.debug(f"Could not parse trajectory metadata: {e}")
                except NetworkError as e:
                    # Unexpected errors parsing metadata
                    logger.warning(f"Unexpected error parsing trajectory metadata: {e}")

            trajectory_data.append({
                'trajectory_id': traj_id,
                'problem': problem,
                'success': success,
                'efficiency': efficiency,
                'steps': steps,
                'metadata': metadata
            })

        conn.close()

        # Analyze patterns
        return self._extract_patterns(trajectory_data)

    def _extract_patterns(self, trajectory_data: List[Dict]) -> Dict[str, Any]:
        """Extract patterns from trajectory data"""

        # Group by problem type
        problem_types = defaultdict(list)

        for traj in trajectory_data:
            problem = traj['problem']
            problem_type = self._classify_problem_type(problem)

            # Extract strategy from metadata
            strategy = traj['metadata'].get('strategy', 'hierarchical')

            problem_types[problem_type].append({
                'strategy': strategy,
                'efficiency': traj['efficiency'] or 0.0,
                'steps': len(traj['steps']),
                'step_types': [s[1] for s in traj['steps']],  # step_type
                'success': traj['success']
            })

        # Create patterns
        patterns = {}
        for problem_type, trajs in problem_types.items():
            if len(trajs) < 2:  # Need at least 2 samples
                continue

            # Calculate statistics
            strategies = defaultdict(list)
            for t in trajs:
                strategies[t['strategy']].append(t)

            # Find best strategy for this problem type
            best_strategy = None
            best_quality = 0.0

            for strategy, strategy_trajs in strategies.items():
                avg_efficiency = sum(t['efficiency'] for t in strategy_trajs) / len(strategy_trajs)
                if avg_efficiency > best_quality:
                    best_quality = avg_efficiency
                    best_strategy = strategy

            # Create pattern
            if best_strategy:
                strategy_trajs = strategies[best_strategy]
                avg_steps = sum(t['steps'] for t in strategy_trajs) / len(strategy_trajs)

                # Find common step types
                step_type_counts = defaultdict(int)
                for t in strategy_trajs:
                    for step_type in set(t['step_types']):
                        step_type_counts[step_type] += 1

                common_steps = [
                    step_type for step_type, count in step_type_counts.items()
                    if count >= len(strategy_trajs) * 0.5  # Present in at least 50% of trajectories
                ]

                pattern = TrajectoryPattern(
                    pattern_id=f"pattern_{problem_type}_{best_strategy}",
                    problem_type=problem_type,
                    successful_strategy=PlanningStrategy(best_strategy) if isinstance(best_strategy, str) else best_strategy,
                    avg_quality=best_quality,
                    avg_steps=int(avg_steps),
                    common_steps=common_steps,
                    success_rate=1.0,  # All are successful
                    sample_count=len(strategy_trajs)
                )

                patterns[pattern.pattern_id] = pattern

        self.patterns = patterns

        return {
            'total_trajectories': len(trajectory_data),
            'problem_types': len(problem_types),
            'patterns_discovered': len(patterns),
            'patterns': {pid: p.to_dict() for pid, p in patterns.items()}
        }

    def _classify_problem_type(self, problem: str) -> str:
        """
        Classify problem into a type for pattern matching

        Categories:
        - api_design: REST API, GraphQL, endpoints
        - debugging: fix, error, bug, debug
        - architecture: system, microservices, design, architecture
        - data: database, query, data processing
        - authentication: auth, login, security, authentication
        - optimization: performance, optimize, speed
        - testing: test, unit test, integration
        """
        problem_lower = problem.lower()

        # Keywords for each category
        categories = {
            'api_design': ['api', 'rest', 'graphql', 'endpoint', 'route'],
            'debugging': ['fix', 'error', 'bug', 'debug', 'issue', 'problem'],
            'architecture': ['system', 'microservices', 'architecture', 'design', 'component'],
            'data': ['database', 'query', 'data', 'sql', 'storage'],
            'authentication': ['auth', 'login', 'security', 'authentication', 'jwt', 'token'],
            'optimization': ['performance', 'optimize', 'speed', 'efficient', 'fast'],
            'testing': ['test', 'unit', 'integration', 'e2e', 'testing']
        }

        # Score each category
        scores = {}
        for category, keywords in categories.items():
            score = sum(1 for keyword in keywords if keyword in problem_lower)
            if score > 0:
                scores[category] = score

        # Return highest scoring category
        if scores:
            return max(scores.items(), key=lambda x: x[1])[0]
        else:
            return 'general'

    def get_pattern_for_problem(self, problem: str) -> Optional[TrajectoryPattern]:
        """Get the best pattern for a given problem"""
        problem_type = self._classify_problem_type(problem)

        # Find patterns matching this problem type
        matching_patterns = [
            p for p in self.patterns.values()
            if p.problem_type == problem_type
        ]

        if not matching_patterns:
            return None

        # Return highest quality pattern
        return max(matching_patterns, key=lambda p: p.avg_quality)


class PatternLibrary:
    """
    Stores and manages learned patterns

    Provides efficient storage and retrieval of reasoning patterns:
    - Persistent storage of patterns
    - Pattern versioning and updates
    - Fast pattern lookup by problem type
    """

    def __init__(self, storage_path: str | None = None):
        import os
        from pathlib import Path
        if storage_path is None:
            storage_path = str(Path(os.environ.get("HELIX_HOME", Path.home() / ".helix")) / "data" / "pattern_library.json")
        self.storage_path = storage_path
        self.patterns: Dict[str, TrajectoryPattern] = {}
        self._load_patterns()

    def _load_patterns(self):
        """Load patterns from disk"""
        path = Path(self.storage_path)
        if path.exists():
            try:
                with open(path, 'r') as f:
                    data = json.load(f)
                    self.patterns = {
                        pid: TrajectoryPattern.from_dict(p)
                        for pid, p in data.items()
                    }
            except InitializationError as e:
                print(f"Warning: Could not load patterns: {e}")

    def _save_patterns(self):
        """Save patterns to disk"""
        path = Path(self.storage_path)
        path.parent.mkdir(parents=True, exist_ok=True)

        with open(path, 'w') as f:
            json.dump(
                {pid: p.to_dict() for pid, p in self.patterns.items()},
                f,
                indent=2
            )

    def add_pattern(self, pattern: TrajectoryPattern):
        """Add a pattern to the library"""
        self.patterns[pattern.pattern_id] = pattern
        self._save_patterns()

    def get_pattern(self, pattern_id: str) -> Optional[TrajectoryPattern]:
        """Get a specific pattern"""
        return self.patterns.get(pattern_id)

    def get_patterns_by_type(self, problem_type: str) -> List[TrajectoryPattern]:
        """Get all patterns for a problem type"""
        return [
            p for p in self.patterns.values()
            if p.problem_type == problem_type
        ]

    def get_all_patterns(self) -> Dict[str, TrajectoryPattern]:
        """Get all patterns"""
        return self.patterns.copy()

    def update_pattern(self, pattern_id: str, **updates):
        """Update an existing pattern"""
        if pattern_id in self.patterns:
            pattern = self.patterns[pattern_id]
            for key, value in updates.items():
                if hasattr(pattern, key):
                    setattr(pattern, key, value)
            self._save_patterns()


class StrategyRecommender:
    """
    Recommends the best reasoning strategy for a given problem

    Uses historical patterns to suggest:
    - Which planning strategy to use
    - Expected quality and step count
    - Confidence in the recommendation
    """

    def __init__(self, analyzer: TrajectoryAnalyzer, library: PatternLibrary):
        self.analyzer = analyzer
        self.library = library

    def recommend_strategy(self, problem: str) -> Tuple[PlanningStrategy, float, Dict[str, Any]]:
        """
        Recommend best strategy for a problem

        Args:
            problem: Problem statement

        Returns:
            Tuple of (strategy, confidence, metadata)
            - strategy: Recommended PlanningStrategy
            - confidence: Confidence in recommendation (0-1)
            - metadata: Additional info (expected_steps, expected_quality, etc.)
        """
        # Classify problem
        problem_type = self.analyzer._classify_problem_type(problem)

        # Get matching patterns
        patterns = self.library.get_patterns_by_type(problem_type)

        if not patterns:
            # No patterns found, use default
            return (
                PlanningStrategy.HIERARCHICAL,
                0.5,  # Medium confidence (no historical data)
                {
                    'reason': 'No historical patterns found, using default (hierarchical)',
                    'problem_type': problem_type
                }
            )

        # Find best pattern
        best_pattern = max(patterns, key=lambda p: p.avg_quality * p.sample_count)

        # Calculate confidence based on sample count and quality
        confidence = min(
            0.9,  # Max confidence
            0.5 + (best_pattern.sample_count / 20) * 0.3 + best_pattern.success_rate * 0.2
        )

        metadata = {
            'problem_type': problem_type,
            'pattern_id': best_pattern.pattern_id,
            'expected_quality': best_pattern.avg_quality,
            'expected_steps': best_pattern.avg_steps,
            'sample_count': best_pattern.sample_count,
            'common_steps': best_pattern.common_steps,
            'reason': f'Based on {best_pattern.sample_count} successful trajectories'
        }

        return (best_pattern.successful_strategy, confidence, metadata)

    def get_recommendation_explanation(self, problem: str) -> str:
        """Get a human-readable explanation of the recommendation"""
        strategy, confidence, metadata = self.recommend_strategy(problem)

        explanation = f"""
Strategy Recommendation for: "{problem[:80]}..."

Problem Type: {metadata['problem_type']}
Recommended Strategy: {strategy.value}
Confidence: {confidence:.1%}

Reasoning:
{metadata['reason']}

Expected Performance:
- Quality Score: {metadata.get('expected_quality', 'N/A')}
- Number of Steps: {metadata.get('expected_steps', 'N/A')}
- Sample Size: {metadata.get('sample_count', 0)} trajectories

Common Steps in Successful Solutions:
{', '.join(metadata.get('common_steps', []))}
"""
        return explanation.strip()


# Singleton instances
_trajectory_analyzer: Optional[TrajectoryAnalyzer] = None
_pattern_library: Optional[PatternLibrary] = None
_strategy_recommender: Optional[StrategyRecommender] = None


def get_trajectory_analyzer() -> TrajectoryAnalyzer:
    """Get the singleton trajectory analyzer instance"""
    global _trajectory_analyzer
    if _trajectory_analyzer is None:
        _trajectory_analyzer = TrajectoryAnalyzer()
    return _trajectory_analyzer


def get_pattern_library() -> PatternLibrary:
    """Get the singleton pattern library instance"""
    global _pattern_library
    if _pattern_library is None:
        _pattern_library = PatternLibrary()
    return _pattern_library


def get_strategy_recommender() -> StrategyRecommender:
    """Get the singleton strategy recommender instance"""
    global _strategy_recommender
    if _strategy_recommender is None:
        analyzer = get_trajectory_analyzer()
        library = get_pattern_library()
        _strategy_recommender = StrategyRecommender(analyzer, library)
    return _strategy_recommender


# Convenience function for quick recommendations
def recommend_strategy_for_problem(problem: str) -> Tuple[PlanningStrategy, float, Dict[str, Any]]:
    """
    Quick function to get strategy recommendation

    Usage:
        strategy, confidence, info = recommend_strategy_for_problem("Design a REST API")
        print(f"Use {strategy.value} with {confidence:.0%} confidence")
    """
    recommender = get_strategy_recommender()
    return recommender.recommend_strategy(problem)
