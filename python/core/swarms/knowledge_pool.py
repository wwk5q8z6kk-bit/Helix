"""
CentralizedKnowledgePool - Cross-swarm knowledge sharing and meta-learning
Integrates with Enhancement 12 (Historical Learning) to enable knowledge transfer
"""

from typing import Dict, List, Optional, Any, Set
from dataclasses import dataclass, field
from datetime import datetime
from collections import defaultdict
import logging

from core.reasoning import (
    TrajectoryPattern,
    TrajectoryAnalyzer,
    PatternLibrary,
    PlanningStrategy,
    get_trajectory_analyzer,
    get_pattern_library
)

logger = logging.getLogger(__name__)


@dataclass
class MetaPattern:
    """
    Meta-pattern extracted from multiple domain patterns
    Represents knowledge that applies across domains
    """
    meta_pattern_id: str
    name: str
    description: str
    applicable_domains: List[str]  # Domains where this pattern works
    base_patterns: List[str]  # Pattern IDs this is derived from
    avg_quality: float
    avg_steps: int
    recommended_strategy: PlanningStrategy
    success_rate: float
    sample_count: int
    common_characteristics: List[str]
    created_at: datetime = field(default_factory=datetime.now)
    metadata: Dict[str, Any] = field(default_factory=dict)

    def to_dict(self) -> Dict[str, Any]:
        """Serialize to dictionary"""
        return {
            'meta_pattern_id': self.meta_pattern_id,
            'name': self.name,
            'description': self.description,
            'applicable_domains': self.applicable_domains,
            'base_patterns': self.base_patterns,
            'avg_quality': self.avg_quality,
            'avg_steps': self.avg_steps,
            'recommended_strategy': self.recommended_strategy.value,
            'success_rate': self.success_rate,
            'sample_count': self.sample_count,
            'common_characteristics': self.common_characteristics,
            'created_at': self.created_at.isoformat(),
            'metadata': self.metadata
        }


@dataclass
class KnowledgeTransfer:
    """
    Record of knowledge transferred from one domain to another
    """
    transfer_id: str
    source_domain: str
    target_domain: str
    pattern_id: str
    transferred_at: datetime
    success: bool
    quality_improvement: Optional[float] = None  # Quality delta after transfer
    metadata: Dict[str, Any] = field(default_factory=dict)


class CentralizedKnowledgePool:
    """
    Centralized knowledge pool for cross-swarm learning

    Features:
    - Collects patterns from all swarms (via Enhancement 12)
    - Identifies meta-patterns across domains
    - Enables knowledge transfer between swarms
    - Tracks cross-domain success rates
    - Provides global recommendations

    This enables swarms to learn from each other, creating compound
    improvements across the entire system.
    """

    def __init__(
        self,
        historical_learning: Optional[TrajectoryAnalyzer] = None,
        pattern_library: Optional[PatternLibrary] = None
    ):
        """
        Initialize centralized knowledge pool

        Args:
            historical_learning: TrajectoryAnalyzer (Enhancement 12)
            pattern_library: Global PatternLibrary (Enhancement 12)
        """
        # Wire to Enhancement 12 components
        self.historical_learning = historical_learning or get_trajectory_analyzer()
        self.global_library = pattern_library or get_pattern_library()

        # Domain-specific pattern libraries
        self._domain_libraries: Dict[str, PatternLibrary] = {}

        # Meta-patterns extracted from multiple domains
        self._meta_patterns: Dict[str, MetaPattern] = {}

        # Knowledge transfer history
        self._transfers: List[KnowledgeTransfer] = []

        # Cross-domain pattern mapping
        # Maps: (domain_A, domain_B) → List[pattern_id]
        self._cross_domain_patterns: Dict[tuple, List[str]] = defaultdict(list)

        logger.info("Initialized CentralizedKnowledgePool with Enhancement 12 integration")

    def register_domain_library(
        self,
        domain: str,
        library: PatternLibrary
    ):
        """
        Register a domain-specific pattern library

        Args:
            domain: Domain name (e.g., "testing", "review")
            library: PatternLibrary for this domain
        """
        self._domain_libraries[domain] = library
        logger.info(f"Registered pattern library for domain: {domain}")

    def collect_patterns_from_domain(self, domain: str) -> List[TrajectoryPattern]:
        """
        Collect all patterns from a domain

        Args:
            domain: Domain to collect from

        Returns:
            List of patterns from this domain
        """
        if domain not in self._domain_libraries:
            logger.warning(f"Domain {domain} not registered")
            return []

        library = self._domain_libraries[domain]
        patterns = list(library.patterns.values())

        logger.info(f"Collected {len(patterns)} patterns from {domain}")

        return patterns

    def analyze_cross_domain_patterns(self) -> Dict[str, Any]:
        """
        Analyze patterns across all domains to find meta-patterns

        Returns:
            Analysis report with discovered meta-patterns
        """
        all_patterns: Dict[str, List[TrajectoryPattern]] = {}

        # Collect patterns from all domains
        for domain, library in self._domain_libraries.items():
            all_patterns[domain] = list(library.patterns.values())

        total_patterns = sum(len(patterns) for patterns in all_patterns.values())
        logger.info(f"Analyzing {total_patterns} patterns across {len(all_patterns)} domains")

        # Find patterns that work across multiple domains
        meta_patterns_found = self._extract_meta_patterns(all_patterns)

        # Find cross-domain similarities
        cross_domain_links = self._find_cross_domain_links(all_patterns)

        analysis = {
            'total_patterns': total_patterns,
            'domains_analyzed': len(all_patterns),
            'meta_patterns_found': len(meta_patterns_found),
            'cross_domain_links': len(cross_domain_links),
            'patterns_by_domain': {
                domain: len(patterns)
                for domain, patterns in all_patterns.items()
            },
            'meta_patterns': [mp.to_dict() for mp in meta_patterns_found]
        }

        logger.info(
            f"Analysis complete: {len(meta_patterns_found)} meta-patterns, "
            f"{len(cross_domain_links)} cross-domain links"
        )

        return analysis

    def _extract_meta_patterns(
        self,
        all_patterns: Dict[str, List[TrajectoryPattern]]
    ) -> List[MetaPattern]:
        """
        Extract meta-patterns from domain patterns

        A meta-pattern is a pattern that appears successful across multiple domains

        Args:
            all_patterns: Patterns grouped by domain

        Returns:
            List of discovered meta-patterns
        """
        meta_patterns = []

        # Group patterns by strategy
        by_strategy: Dict[PlanningStrategy, List[tuple]] = defaultdict(list)

        for domain, patterns in all_patterns.items():
            for pattern in patterns:
                # Only consider high-quality patterns
                if pattern.avg_quality > 0.75:
                    by_strategy[pattern.successful_strategy].append((domain, pattern))

        # For each strategy, check if it works across domains
        for strategy, domain_patterns in by_strategy.items():
            domains_with_strategy = set(domain for domain, _ in domain_patterns)

            # If strategy works in 2+ domains, it's a meta-pattern
            if len(domains_with_strategy) >= 2:
                patterns = [p for _, p in domain_patterns]

                meta_pattern = MetaPattern(
                    meta_pattern_id=f"meta_{strategy.value}_{datetime.now().strftime('%Y%m%d_%H%M%S')}",
                    name=f"{strategy.value.title()} Strategy Success",
                    description=f"The {strategy.value} strategy works well across {len(domains_with_strategy)} domains",
                    applicable_domains=list(domains_with_strategy),
                    base_patterns=[p.pattern_id for p in patterns],
                    avg_quality=sum(p.avg_quality for p in patterns) / len(patterns),
                    avg_steps=int(sum(p.avg_steps for p in patterns) / len(patterns)),
                    recommended_strategy=strategy,
                    success_rate=sum(p.success_rate for p in patterns) / len(patterns),
                    sample_count=sum(p.sample_count for p in patterns),
                    common_characteristics=[
                        f"strategy={strategy.value}",
                        f"domains={','.join(domains_with_strategy)}",
                        f"avg_quality={sum(p.avg_quality for p in patterns) / len(patterns):.3f}"
                    ]
                )

                meta_patterns.append(meta_pattern)
                self._meta_patterns[meta_pattern.meta_pattern_id] = meta_pattern

                logger.info(
                    f"Discovered meta-pattern: {meta_pattern.name} "
                    f"(quality: {meta_pattern.avg_quality:.3f}, "
                    f"domains: {len(domains_with_strategy)})"
                )

        return meta_patterns

    def _find_cross_domain_links(
        self,
        all_patterns: Dict[str, List[TrajectoryPattern]]
    ) -> List[tuple]:
        """
        Find patterns that could transfer between domains

        Args:
            all_patterns: Patterns grouped by domain

        Returns:
            List of (domain_A, domain_B, pattern_id) links
        """
        links = []

        domains = list(all_patterns.keys())

        # Compare each pair of domains
        for i, domain_a in enumerate(domains):
            for domain_b in domains[i+1:]:
                patterns_a = all_patterns[domain_a]
                patterns_b = all_patterns[domain_b]

                # Find similar patterns (same strategy, similar quality)
                for pattern_a in patterns_a:
                    for pattern_b in patterns_b:
                        if self._patterns_similar(pattern_a, pattern_b):
                            link = (domain_a, domain_b, pattern_a.pattern_id)
                            links.append(link)
                            self._cross_domain_patterns[(domain_a, domain_b)].append(
                                pattern_a.pattern_id
                            )

        logger.info(f"Found {len(links)} cross-domain pattern links")

        return links

    def _patterns_similar(
        self,
        pattern_a: TrajectoryPattern,
        pattern_b: TrajectoryPattern
    ) -> bool:
        """
        Check if two patterns are similar enough for cross-domain transfer

        Args:
            pattern_a: First pattern
            pattern_b: Second pattern

        Returns:
            True if patterns are similar
        """
        # Same strategy
        if pattern_a.successful_strategy != pattern_b.successful_strategy:
            return False

        # Similar quality (within 10%)
        quality_diff = abs(pattern_a.avg_quality - pattern_b.avg_quality)
        if quality_diff > 0.1:
            return False

        # Similar number of steps (within 50%)
        if pattern_a.avg_steps > 0 and pattern_b.avg_steps > 0:
            steps_ratio = max(pattern_a.avg_steps, pattern_b.avg_steps) / min(pattern_a.avg_steps, pattern_b.avg_steps)
            if steps_ratio > 1.5:
                return False

        return True

    def recommend_pattern_for_domain(
        self,
        domain: str,
        problem_type: str
    ) -> Optional[TrajectoryPattern]:
        """
        Recommend best pattern for a domain, considering cross-domain knowledge

        Args:
            domain: Target domain
            problem_type: Type of problem

        Returns:
            Recommended pattern (may be from another domain)
        """
        # First, try domain-specific pattern
        if domain in self._domain_libraries:
            library = self._domain_libraries[domain]
            patterns = library.get_patterns_by_type(problem_type)

            if patterns:
                # Return highest quality pattern
                best = max(patterns, key=lambda p: p.avg_quality)
                logger.info(
                    f"Recommended domain-specific pattern for {domain}: "
                    f"{best.pattern_id} (quality: {best.avg_quality:.3f})"
                )
                return best

        # If no domain-specific pattern, try cross-domain transfer
        logger.info(f"No domain-specific pattern for {domain}, checking cross-domain")

        # Check meta-patterns
        applicable_meta = [
            mp for mp in self._meta_patterns.values()
            if domain in mp.applicable_domains or not mp.applicable_domains
        ]

        if applicable_meta:
            # Return highest quality meta-pattern
            best_meta = max(applicable_meta, key=lambda mp: mp.avg_quality)

            # Convert meta-pattern to trajectory pattern
            pattern = TrajectoryPattern(
                pattern_id=f"transfer_{best_meta.meta_pattern_id}",
                problem_type=problem_type,
                successful_strategy=best_meta.recommended_strategy,
                avg_quality=best_meta.avg_quality,
                avg_steps=best_meta.avg_steps,
                common_steps=best_meta.common_characteristics,
                success_rate=best_meta.success_rate,
                sample_count=best_meta.sample_count
            )

            logger.info(
                f"Recommended cross-domain pattern for {domain}: "
                f"{best_meta.name} (quality: {best_meta.avg_quality:.3f})"
            )

            return pattern

        logger.info(f"No pattern found for {domain}")
        return None

    def transfer_knowledge(
        self,
        source_domain: str,
        target_domain: str,
        pattern_id: str
    ) -> KnowledgeTransfer:
        """
        Transfer a pattern from one domain to another

        Args:
            source_domain: Source domain
            target_domain: Target domain
            pattern_id: Pattern to transfer

        Returns:
            KnowledgeTransfer record
        """
        # Get source pattern
        if source_domain not in self._domain_libraries:
            raise ValueError(f"Source domain {source_domain} not registered")

        source_library = self._domain_libraries[source_domain]
        pattern = source_library.patterns.get(pattern_id)

        if not pattern:
            raise ValueError(f"Pattern {pattern_id} not found in {source_domain}")

        # Transfer to target domain
        if target_domain not in self._domain_libraries:
            raise ValueError(f"Target domain {target_domain} not registered")

        target_library = self._domain_libraries[target_domain]

        # Create new pattern ID for target domain
        new_pattern_id = f"{target_domain}_transfer_{pattern_id}"

        # Create transferred pattern
        transferred_pattern = TrajectoryPattern(
            pattern_id=new_pattern_id,
            problem_type=pattern.problem_type,
            successful_strategy=pattern.successful_strategy,
            avg_quality=pattern.avg_quality,
            avg_steps=pattern.avg_steps,
            common_steps=pattern.common_steps,
            success_rate=pattern.success_rate * 0.9,  # Slight penalty for cross-domain
            sample_count=pattern.sample_count
        )

        # Add to target library
        target_library.add_pattern(transferred_pattern)

        # Record transfer
        transfer = KnowledgeTransfer(
            transfer_id=str(datetime.now().timestamp()),
            source_domain=source_domain,
            target_domain=target_domain,
            pattern_id=pattern_id,
            transferred_at=datetime.now(),
            success=True,
            metadata={
                'new_pattern_id': new_pattern_id,
                'source_quality': pattern.avg_quality
            }
        )

        self._transfers.append(transfer)

        logger.info(
            f"Transferred pattern {pattern_id} from {source_domain} to {target_domain} "
            f"(quality: {pattern.avg_quality:.3f})"
        )

        return transfer

    def get_global_stats(self) -> Dict[str, Any]:
        """
        Get global knowledge pool statistics

        Returns:
            Dictionary with stats
        """
        total_patterns = sum(
            len(library.patterns)
            for library in self._domain_libraries.values()
        )

        return {
            'total_domains': len(self._domain_libraries),
            'total_patterns': total_patterns,
            'meta_patterns': len(self._meta_patterns),
            'knowledge_transfers': len(self._transfers),
            'cross_domain_links': sum(
                len(patterns)
                for patterns in self._cross_domain_patterns.values()
            ),
            'patterns_by_domain': {
                domain: len(library.patterns)
                for domain, library in self._domain_libraries.items()
            }
        }


# Singleton instance
_knowledge_pool: Optional[CentralizedKnowledgePool] = None


def get_knowledge_pool() -> CentralizedKnowledgePool:
    """
    Get or create singleton CentralizedKnowledgePool

    Returns:
        Global knowledge pool instance
    """
    global _knowledge_pool
    if _knowledge_pool is None:
        _knowledge_pool = CentralizedKnowledgePool()
    return _knowledge_pool
