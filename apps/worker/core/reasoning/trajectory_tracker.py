"""
Helix Reasoning Trajectory Tracker
Tracks, persists, and analyzes step-by-step reasoning chains
Based on Landscape of Thoughts and Visualization-of-Thought research (2025)
"""

import json
import sqlite3
import hashlib
from datetime import datetime
from typing import Optional, List, Dict, Any, Tuple
from dataclasses import dataclass, field, asdict
from enum import Enum
from pathlib import Path


class StepType(str, Enum):
    """Type of reasoning step"""
    THINK = "think"           # Pure reasoning/analysis
    RETRIEVE = "retrieve"     # Knowledge retrieval
    TOOL = "tool"            # Tool execution
    VERIFY = "verify"        # Verification/checking
    REFLECT = "reflect"      # Meta-reasoning/reflection
    BACKTRACK = "backtrack"  # Undo/backtrack


@dataclass
class ReasoningStep:
    """Single step in reasoning chain"""
    step_id: str = ""
    step_number: int = 0
    timestamp: datetime = field(default_factory=datetime.now)
    content: str = ""
    confidence: float = 1.0  # 0.0 to 1.0
    step_type: StepType = StepType.THINK
    parent_step: Optional[str] = None
    tool_used: Optional[str] = None
    tool_result: Optional[str] = None  # JSON string
    verification_result: Optional[bool] = None
    alternatives_considered: List[str] = field(default_factory=list)
    backtrack_from: Optional[str] = None
    metadata: Dict[str, Any] = field(default_factory=dict)

    def to_dict(self) -> Dict[str, Any]:
        """Convert to dictionary for JSON serialization"""
        d = asdict(self)
        d['timestamp'] = self.timestamp.isoformat()
        d['step_type'] = self.step_type.value
        return d

    @staticmethod
    def from_dict(data: Dict[str, Any]) -> 'ReasoningStep':
        """Create from dictionary"""
        data = data.copy()
        data['timestamp'] = datetime.fromisoformat(data['timestamp'])
        data['step_type'] = StepType(data['step_type'])
        return ReasoningStep(**data)


@dataclass
class ReasoningTrajectory:
    """Complete reasoning path from problem to solution"""
    trajectory_id: str = ""
    problem: str = ""
    start_time: datetime = field(default_factory=datetime.now)
    end_time: Optional[datetime] = None
    steps: List[ReasoningStep] = field(default_factory=list)
    final_answer: str = ""
    success: bool = False
    total_backtracks: int = 0
    tools_used: List[str] = field(default_factory=list)
    efficiency_score: float = 0.0  # 0.0 to 1.0
    metadata: Dict[str, Any] = field(default_factory=dict)

    @property
    def duration_seconds(self) -> float:
        """Duration of reasoning process"""
        if self.end_time:
            return (self.end_time - self.start_time).total_seconds()
        return (datetime.now() - self.start_time).total_seconds()

    @property
    def total_steps(self) -> int:
        """Total number of steps"""
        return len(self.steps)

    @property
    def avg_confidence(self) -> float:
        """Average confidence across steps"""
        if not self.steps:
            return 0.0
        return sum(s.confidence for s in self.steps) / len(self.steps)

    def add_step(
        self,
        content: str,
        step_type: StepType,
        confidence: float = 1.0,
        parent_step: Optional[str] = None,
        tool_used: Optional[str] = None,
        tool_result: Optional[Any] = None,
        verification_result: Optional[bool] = None,
        alternatives: Optional[List[str]] = None,
        backtrack_from: Optional[str] = None,
        metadata: Optional[Dict[str, Any]] = None
    ) -> ReasoningStep:
        """Add a new step to the trajectory"""
        step_id = self._generate_step_id(len(self.steps))

        step = ReasoningStep(
            step_id=step_id,
            step_number=len(self.steps),
            timestamp=datetime.now(),
            content=content,
            confidence=confidence,
            step_type=step_type,
            parent_step=parent_step,
            tool_used=tool_used,
            tool_result=json.dumps(tool_result) if tool_result else None,
            verification_result=verification_result,
            alternatives_considered=alternatives or [],
            backtrack_from=backtrack_from,
            metadata=metadata or {}
        )

        self.steps.append(step)

        # Update trajectory metadata
        if tool_used and tool_used not in self.tools_used:
            self.tools_used.append(tool_used)

        if step_type == StepType.BACKTRACK:
            self.total_backtracks += 1

        return step

    def complete(self, final_answer: str, success: bool):
        """Mark trajectory as complete"""
        self.final_answer = final_answer
        self.success = success
        self.end_time = datetime.now()
        self.efficiency_score = self._calculate_efficiency()

    def _calculate_efficiency(self) -> float:
        """
        Calculate efficiency score based on:
        - Success (50%)
        - Steps taken vs optimal (25%)
        - Backtracks (25%)
        """
        if not self.steps:
            return 0.0

        # Success component (0 or 0.5)
        success_score = 0.5 if self.success else 0.0

        # Steps efficiency (fewer steps = better, assuming optimal is 5)
        optimal_steps = 5
        steps_ratio = min(optimal_steps / len(self.steps), 1.0)
        steps_score = steps_ratio * 0.25

        # Backtrack penalty (fewer backtracks = better)
        max_backtracks = 3
        backtrack_ratio = max(1.0 - (self.total_backtracks / max_backtracks), 0.0)
        backtrack_score = backtrack_ratio * 0.25

        return success_score + steps_score + backtrack_score

    def _generate_step_id(self, step_num: int) -> str:
        """Generate unique step ID"""
        content = f"{self.trajectory_id}:step:{step_num}:{datetime.now().isoformat()}"
        return hashlib.sha256(content.encode()).hexdigest()[:12]

    def to_dict(self) -> Dict[str, Any]:
        """Convert to dictionary for JSON serialization"""
        return {
            'trajectory_id': self.trajectory_id,
            'problem': self.problem,
            'start_time': self.start_time.isoformat(),
            'end_time': self.end_time.isoformat() if self.end_time else None,
            'steps': [s.to_dict() for s in self.steps],
            'final_answer': self.final_answer,
            'success': self.success,
            'total_backtracks': self.total_backtracks,
            'tools_used': self.tools_used,
            'efficiency_score': self.efficiency_score,
            'metadata': self.metadata
        }


class TrajectoryTracker:
    """
    Tracks and persists reasoning trajectories

    Features:
    - Record step-by-step reasoning chains
    - SQLite persistence
    - Trajectory retrieval and analysis
    - Efficiency metrics
    - Pattern detection
    """

    def __init__(self, db_path: Optional[str] = None):
        """Initialize trajectory tracker"""
        if db_path is None:
            db_path = str(Path.home() / ".helix" / "trajectories.db")

        self.db_path = db_path
        Path(self.db_path).parent.mkdir(parents=True, exist_ok=True)

        self._init_database()

    def _init_database(self):
        """Initialize SQLite database"""
        conn = sqlite3.connect(self.db_path)
        cursor = conn.cursor()

        # Trajectories table
        cursor.execute("""
            CREATE TABLE IF NOT EXISTS trajectories (
                trajectory_id TEXT PRIMARY KEY,
                problem TEXT NOT NULL,
                start_time DATETIME NOT NULL,
                end_time DATETIME,
                final_answer TEXT,
                success BOOLEAN,
                total_backtracks INTEGER DEFAULT 0,
                tools_used TEXT,
                efficiency_score REAL DEFAULT 0.0,
                metadata TEXT,
                created_at DATETIME DEFAULT CURRENT_TIMESTAMP
            )
        """)

        # Steps table
        cursor.execute("""
            CREATE TABLE IF NOT EXISTS steps (
                step_id TEXT PRIMARY KEY,
                trajectory_id TEXT NOT NULL,
                step_number INTEGER NOT NULL,
                timestamp DATETIME NOT NULL,
                content TEXT NOT NULL,
                confidence REAL DEFAULT 1.0,
                step_type TEXT NOT NULL,
                parent_step TEXT,
                tool_used TEXT,
                tool_result TEXT,
                verification_result BOOLEAN,
                alternatives_considered TEXT,
                backtrack_from TEXT,
                metadata TEXT,
                FOREIGN KEY (trajectory_id) REFERENCES trajectories (trajectory_id)
            )
        """)

        # Indexes
        cursor.execute("""
            CREATE INDEX IF NOT EXISTS idx_trajectory_success
            ON trajectories (success)
        """)
        cursor.execute("""
            CREATE INDEX IF NOT EXISTS idx_step_type
            ON steps (step_type)
        """)
        cursor.execute("""
            CREATE INDEX IF NOT EXISTS idx_trajectory_steps
            ON steps (trajectory_id, step_number)
        """)

        conn.commit()
        conn.close()

    def create_trajectory(self, problem: str, metadata: Optional[Dict[str, Any]] = None) -> ReasoningTrajectory:
        """Create a new trajectory"""
        trajectory_id = self._generate_trajectory_id(problem)

        trajectory = ReasoningTrajectory(
            trajectory_id=trajectory_id,
            problem=problem,
            start_time=datetime.now(),
            metadata=metadata or {}
        )

        # Persist to database
        self._save_trajectory(trajectory)

        return trajectory

    def save_step(self, trajectory_id: str, step: ReasoningStep):
        """Save a step to database"""
        conn = sqlite3.connect(self.db_path)
        cursor = conn.cursor()

        cursor.execute("""
            INSERT OR REPLACE INTO steps (
                step_id, trajectory_id, step_number, timestamp, content,
                confidence, step_type, parent_step, tool_used, tool_result,
                verification_result, alternatives_considered, backtrack_from, metadata
            ) VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
        """, (
            step.step_id,
            trajectory_id,
            step.step_number,
            step.timestamp.isoformat(),
            step.content,
            step.confidence,
            step.step_type.value,
            step.parent_step,
            step.tool_used,
            json.dumps(step.tool_result) if step.tool_result is not None and not isinstance(step.tool_result, str) else step.tool_result,
            step.verification_result,
            json.dumps(step.alternatives_considered),
            step.backtrack_from,
            json.dumps(step.metadata)
        ))

        conn.commit()
        conn.close()

    def update_trajectory(self, trajectory: ReasoningTrajectory):
        """Update trajectory in database"""
        conn = sqlite3.connect(self.db_path)
        cursor = conn.cursor()

        # Serialize metadata, converting enums to strings
        serialized_metadata = {}
        for key, value in trajectory.metadata.items():
            if hasattr(value, 'value'):  # Enum
                serialized_metadata[key] = str(value.value)
            else:
                serialized_metadata[key] = value

        cursor.execute("""
            UPDATE trajectories
            SET end_time = ?,
                final_answer = ?,
                success = ?,
                total_backtracks = ?,
                tools_used = ?,
                efficiency_score = ?,
                metadata = ?
            WHERE trajectory_id = ?
        """, (
            trajectory.end_time.isoformat() if trajectory.end_time else None,
            trajectory.final_answer,
            int(trajectory.success) if trajectory.success is not None else None,
            trajectory.total_backtracks,
            json.dumps(trajectory.tools_used),
            trajectory.efficiency_score,
            json.dumps(serialized_metadata),
            trajectory.trajectory_id
        ))

        conn.commit()
        conn.close()

    def get_trajectory(self, trajectory_id: str) -> Optional[ReasoningTrajectory]:
        """Retrieve a trajectory by ID"""
        conn = sqlite3.connect(self.db_path)
        cursor = conn.cursor()

        # Get trajectory
        cursor.execute("""
            SELECT trajectory_id, problem, start_time, end_time, final_answer,
                   success, total_backtracks, tools_used, efficiency_score, metadata
            FROM trajectories
            WHERE trajectory_id = ?
        """, (trajectory_id,))

        row = cursor.fetchone()
        if not row:
            conn.close()
            return None

        trajectory = ReasoningTrajectory(
            trajectory_id=row[0],
            problem=row[1],
            start_time=datetime.fromisoformat(row[2]),
            end_time=datetime.fromisoformat(row[3]) if row[3] else None,
            final_answer=row[4] or "",
            success=bool(row[5]) if row[5] is not None else False,
            total_backtracks=row[6] or 0,
            tools_used=json.loads(row[7]) if row[7] else [],
            efficiency_score=row[8] or 0.0,
            metadata=json.loads(row[9]) if row[9] else {}
        )

        # Get steps
        cursor.execute("""
            SELECT step_id, step_number, timestamp, content, confidence,
                   step_type, parent_step, tool_used, tool_result,
                   verification_result, alternatives_considered, backtrack_from, metadata
            FROM steps
            WHERE trajectory_id = ?
            ORDER BY step_number
        """, (trajectory_id,))

        for step_row in cursor.fetchall():
            step = ReasoningStep(
                step_id=step_row[0],
                step_number=step_row[1],
                timestamp=datetime.fromisoformat(step_row[2]),
                content=step_row[3],
                confidence=step_row[4],
                step_type=StepType(step_row[5]),
                parent_step=step_row[6],
                tool_used=step_row[7],
                tool_result=json.loads(step_row[8]) if step_row[8] and step_row[8].startswith('{') else step_row[8],
                verification_result=bool(step_row[9]) if step_row[9] is not None else None,
                alternatives_considered=json.loads(step_row[10]) if step_row[10] else [],
                backtrack_from=step_row[11],
                metadata=json.loads(step_row[12]) if step_row[12] else {}
            )
            trajectory.steps.append(step)

        conn.close()
        return trajectory

    def get_trajectories(
        self,
        success: Optional[bool] = None,
        min_efficiency: float = 0.0,
        limit: int = 100
    ) -> List[ReasoningTrajectory]:
        """Get trajectories with filters"""
        conn = sqlite3.connect(self.db_path)
        cursor = conn.cursor()

        query = """
            SELECT trajectory_id, problem, start_time, end_time, final_answer,
                   success, total_backtracks, tools_used, efficiency_score, metadata
            FROM trajectories
            WHERE efficiency_score >= ?
        """
        params = [min_efficiency]

        if success is not None:
            query += " AND success = ?"
            params.append(success)

        query += " ORDER BY start_time DESC LIMIT ?"
        params.append(limit)

        cursor.execute(query, params)

        trajectories = []
        for row in cursor.fetchall():
            trajectory = ReasoningTrajectory(
                trajectory_id=row[0],
                problem=row[1],
                start_time=datetime.fromisoformat(row[2]),
                end_time=datetime.fromisoformat(row[3]) if row[3] else None,
                final_answer=row[4] or "",
                success=bool(row[5]) if row[5] is not None else False,
                total_backtracks=row[6] or 0,
                tools_used=json.loads(row[7]) if row[7] else [],
                efficiency_score=row[8] or 0.0,
                metadata=json.loads(row[9]) if row[9] else {}
            )
            trajectories.append(trajectory)

        conn.close()
        return trajectories

    def get_trajectory_statistics(self) -> Dict[str, Any]:
        """Get overall trajectory statistics"""
        conn = sqlite3.connect(self.db_path)
        cursor = conn.cursor()

        # Total trajectories
        cursor.execute("SELECT COUNT(*) FROM trajectories")
        total = cursor.fetchone()[0]

        # Success rate
        cursor.execute("SELECT COUNT(*) FROM trajectories WHERE success = 1")
        successes = cursor.fetchone()[0]
        success_rate = successes / total if total > 0 else 0.0

        # Average efficiency
        cursor.execute("SELECT AVG(efficiency_score) FROM trajectories")
        avg_efficiency = cursor.fetchone()[0] or 0.0

        # Average steps
        cursor.execute("""
            SELECT AVG(step_count)
            FROM (
                SELECT trajectory_id, COUNT(*) as step_count
                FROM steps
                GROUP BY trajectory_id
            )
        """)
        avg_steps = cursor.fetchone()[0] or 0.0

        # Most used tools
        cursor.execute("""
            SELECT tool_used, COUNT(*) as count
            FROM steps
            WHERE tool_used IS NOT NULL
            GROUP BY tool_used
            ORDER BY count DESC
            LIMIT 5
        """)
        top_tools = {row[0]: row[1] for row in cursor.fetchall()}

        conn.close()

        return {
            'total_trajectories': total,
            'success_rate': success_rate,
            'avg_efficiency': avg_efficiency,
            'avg_steps': avg_steps,
            'top_tools': top_tools
        }

    def _save_trajectory(self, trajectory: ReasoningTrajectory):
        """Save new trajectory to database"""
        conn = sqlite3.connect(self.db_path)
        cursor = conn.cursor()

        cursor.execute("""
            INSERT OR REPLACE INTO trajectories (
                trajectory_id, problem, start_time, end_time, final_answer,
                success, total_backtracks, tools_used, efficiency_score, metadata
            ) VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
        """, (
            trajectory.trajectory_id,
            trajectory.problem,
            trajectory.start_time.isoformat(),
            trajectory.end_time.isoformat() if trajectory.end_time else None,
            trajectory.final_answer,
            trajectory.success,
            trajectory.total_backtracks,
            json.dumps(trajectory.tools_used),
            trajectory.efficiency_score,
            json.dumps(trajectory.metadata)
        ))

        conn.commit()
        conn.close()

    def _generate_trajectory_id(self, problem: str) -> str:
        """Generate unique trajectory ID"""
        content = f"{problem}:{datetime.now().isoformat()}"
        return f"traj_{hashlib.sha256(content.encode()).hexdigest()[:12]}"


# Singleton instance
_trajectory_tracker_instance = None


def get_trajectory_tracker(db_path: Optional[str] = None) -> TrajectoryTracker:
    """Get singleton trajectory tracker instance"""
    global _trajectory_tracker_instance
    if _trajectory_tracker_instance is None:
        _trajectory_tracker_instance = TrajectoryTracker(db_path)
    return _trajectory_tracker_instance
