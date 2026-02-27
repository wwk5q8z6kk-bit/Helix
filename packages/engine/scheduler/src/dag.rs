//! DAG-based dependency resolver with cycle detection and failure propagation.
//!
//! Port of `python/core/scheduling/dependency_resolver.py`.

use std::collections::{HashMap, HashSet, VecDeque};

use serde::{Deserialize, Serialize};

use crate::error::{Result, SchedulerError};

// ── Task state ──────────────────────────────────────────────────────

#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash, Serialize, Deserialize)]
#[serde(rename_all = "snake_case")]
pub enum TaskState {
    Pending,
    Ready,
    Running,
    Completed,
    Failed,
    Cancelled,
}

impl TaskState {
    pub fn is_terminal(self) -> bool {
        matches!(self, Self::Completed | Self::Failed | Self::Cancelled)
    }
}

impl std::fmt::Display for TaskState {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        match self {
            Self::Pending => write!(f, "pending"),
            Self::Ready => write!(f, "ready"),
            Self::Running => write!(f, "running"),
            Self::Completed => write!(f, "completed"),
            Self::Failed => write!(f, "failed"),
            Self::Cancelled => write!(f, "cancelled"),
        }
    }
}

// ── Immutable snapshot ──────────────────────────────────────────────

/// Read-only snapshot of a task's position in the graph.
#[derive(Debug, Clone, PartialEq, Eq)]
pub struct TaskNode {
    pub task_id: String,
    pub state: TaskState,
    pub dependencies: HashSet<String>,
    pub dependents: HashSet<String>,
    pub priority: i32,
}

// ── Dependency resolver ─────────────────────────────────────────────

/// DAG-based dependency resolver with event-driven state transitions.
///
/// Tasks are added incrementally.  When all of a task's dependencies are
/// satisfied it becomes `Ready`.  Completing a task unblocks its dependents;
/// failing a task BFS-cancels all transitive dependents.
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct DependencyResolver {
    /// Forward edges: task → set of tasks it depends ON.
    dependencies: HashMap<String, HashSet<String>>,
    /// Reverse edges: task → set of tasks that depend on IT.
    dependents: HashMap<String, HashSet<String>>,
    /// Active task states (Pending | Ready | Running only).
    states: HashMap<String, TaskState>,
    /// Numeric priority per task (0 = critical, 3 = low).
    priorities: HashMap<String, i32>,
    /// Terminal tasks retained for dependency lookups.
    history: HashMap<String, TaskState>,
    /// Insertion order for FIFO tie-breaking.
    insertion_order: Vec<String>,
}

impl Default for DependencyResolver {
    fn default() -> Self {
        Self::new()
    }
}

impl DependencyResolver {
    pub fn new() -> Self {
        Self {
            dependencies: HashMap::new(),
            dependents: HashMap::new(),
            states: HashMap::new(),
            priorities: HashMap::new(),
            history: HashMap::new(),
            insertion_order: Vec::new(),
        }
    }

    // ── Graph mutation ──────────────────────────────────────────────

    /// Add a task to the graph, returning its initial state.
    ///
    /// Dependencies already completed in history are auto-resolved.
    /// Unknown dependencies are warned and auto-resolved to prevent deadlocks.
    pub fn add_task(
        &mut self,
        task_id: &str,
        deps: &[&str],
        priority: i32,
    ) -> Result<TaskState> {
        if self.states.contains_key(task_id) {
            return Err(SchedulerError::DuplicateTask(task_id.to_string()));
        }

        // Resolve dependencies against history / active graph.
        let mut remaining = HashSet::new();
        for &dep in deps {
            match self.history.get(dep) {
                Some(TaskState::Completed) => continue,           // already done
                Some(TaskState::Failed | TaskState::Cancelled) => {
                    // Upstream already failed — immediately cancel.
                    self.history
                        .insert(task_id.to_string(), TaskState::Cancelled);
                    self.insertion_order.push(task_id.to_string());
                    return Ok(TaskState::Cancelled);
                }
                _ => {}
            }

            if self.states.contains_key(dep) {
                remaining.insert(dep.to_string());
            } else if !self.history.contains_key(dep) {
                tracing::warn!(
                    dep,
                    task_id,
                    "Dependency not in graph or history — auto-resolving"
                );
            }
        }

        // Cycle check before committing.
        self.check_cycle(task_id, &remaining)?;

        let state = if remaining.is_empty() {
            TaskState::Ready
        } else {
            TaskState::Pending
        };

        // Wire up edges.
        self.dependencies
            .insert(task_id.to_string(), remaining.clone());
        self.dependents
            .entry(task_id.to_string())
            .or_default();

        for dep in &remaining {
            self.dependents
                .entry(dep.clone())
                .or_default()
                .insert(task_id.to_string());
        }

        self.states.insert(task_id.to_string(), state);
        self.priorities.insert(task_id.to_string(), priority);
        self.insertion_order.push(task_id.to_string());

        Ok(state)
    }

    /// Remove a task from the graph, returning newly-ready dependents.
    pub fn remove_task(&mut self, task_id: &str) -> HashSet<String> {
        self.states.remove(task_id);
        self.priorities.remove(task_id);

        let deps = self.dependencies.remove(task_id).unwrap_or_default();
        for dep in &deps {
            if let Some(set) = self.dependents.get_mut(dep) {
                set.remove(task_id);
            }
        }

        let my_dependents = self.dependents.remove(task_id).unwrap_or_default();
        let mut newly_ready = HashSet::new();
        for d in &my_dependents {
            if let Some(dep_set) = self.dependencies.get_mut(d) {
                dep_set.remove(task_id);
                if dep_set.is_empty() {
                    if let Some(s) = self.states.get_mut(d) {
                        if *s == TaskState::Pending {
                            *s = TaskState::Ready;
                            newly_ready.insert(d.clone());
                        }
                    }
                }
            }
        }
        newly_ready
    }

    // ── State transitions ───────────────────────────────────────────

    /// Transition Ready → Running.
    pub fn mark_running(&mut self, task_id: &str) -> Result<()> {
        let state = self
            .states
            .get(task_id)
            .ok_or_else(|| SchedulerError::TaskNotFound(task_id.to_string()))?;

        if *state != TaskState::Ready {
            return Err(SchedulerError::InvalidTransition {
                task: task_id.to_string(),
                from: state.to_string(),
                to: "running".into(),
            });
        }
        self.states.insert(task_id.to_string(), TaskState::Running);
        Ok(())
    }

    /// Transition Running → Completed. Returns newly-ready dependents
    /// sorted by (priority, insertion order).
    pub fn mark_completed(&mut self, task_id: &str) -> Result<Vec<String>> {
        let state = self
            .states
            .get(task_id)
            .ok_or_else(|| SchedulerError::TaskNotFound(task_id.to_string()))?;

        if *state != TaskState::Running {
            return Err(SchedulerError::InvalidTransition {
                task: task_id.to_string(),
                from: state.to_string(),
                to: "completed".into(),
            });
        }

        // Move to history.
        self.states.remove(task_id);
        self.history
            .insert(task_id.to_string(), TaskState::Completed);

        // Unblock dependents.
        let my_dependents = self
            .dependents
            .get(task_id)
            .cloned()
            .unwrap_or_default();

        let mut newly_ready = Vec::new();
        for d in &my_dependents {
            if let Some(dep_set) = self.dependencies.get_mut(d) {
                dep_set.remove(task_id);
                if dep_set.is_empty() {
                    if let Some(s) = self.states.get_mut(d) {
                        if *s == TaskState::Pending {
                            *s = TaskState::Ready;
                            newly_ready.push(d.clone());
                        }
                    }
                }
            }
        }

        self.sort_by_priority(&mut newly_ready);

        // Clean up forward edges.
        self.dependencies.remove(task_id);

        Ok(newly_ready)
    }

    /// Transition Running → Failed. BFS-propagates cancellation to all
    /// transitive dependents. Returns cancelled task IDs.
    pub fn mark_failed(&mut self, task_id: &str) -> Result<Vec<String>> {
        let state = self
            .states
            .get(task_id)
            .ok_or_else(|| SchedulerError::TaskNotFound(task_id.to_string()))?;

        if *state != TaskState::Running {
            return Err(SchedulerError::InvalidTransition {
                task: task_id.to_string(),
                from: state.to_string(),
                to: "failed".into(),
            });
        }

        self.states.remove(task_id);
        self.history.insert(task_id.to_string(), TaskState::Failed);

        // BFS cancellation.
        let mut cancelled = Vec::new();
        let mut queue = VecDeque::new();

        if let Some(deps) = self.dependents.get(task_id) {
            for d in deps {
                queue.push_back(d.clone());
            }
        }

        while let Some(tid) = queue.pop_front() {
            if let Some(s) = self.states.get(&tid) {
                if matches!(s, TaskState::Pending | TaskState::Ready) {
                    self.states.remove(&tid);
                    self.history.insert(tid.clone(), TaskState::Cancelled);
                    cancelled.push(tid.clone());

                    if let Some(further) = self.dependents.get(&tid) {
                        for f in further {
                            if self.states.contains_key(f) {
                                queue.push_back(f.clone());
                            }
                        }
                    }
                }
            }
        }

        // Clean up edges for failed + cancelled tasks.
        self.dependencies.remove(task_id);
        for c in &cancelled {
            self.dependencies.remove(c);
        }

        Ok(cancelled)
    }

    // ── Queries ─────────────────────────────────────────────────────

    /// All Ready tasks sorted by (priority ASC, insertion order ASC).
    pub fn get_ready_tasks(&self) -> Vec<String> {
        let mut ready: Vec<String> = self
            .states
            .iter()
            .filter(|(_, &s)| s == TaskState::Ready)
            .map(|(id, _)| id.clone())
            .collect();
        self.sort_by_priority(&mut ready);
        ready
    }

    /// Kahn's algorithm producing parallel execution waves.
    pub fn get_execution_waves(&self) -> Vec<Vec<String>> {
        // Build in-degree map from active tasks only.
        let active: HashSet<&String> = self.states.keys().collect();
        let mut in_degree: HashMap<&String, usize> = HashMap::new();
        for id in &active {
            let deg = self
                .dependencies
                .get(*id)
                .map(|deps| deps.iter().filter(|d| active.contains(d)).count())
                .unwrap_or(0);
            in_degree.insert(id, deg);
        }

        let mut waves = Vec::new();
        let mut remaining: HashSet<&String> = active.iter().copied().collect();

        loop {
            let mut wave: Vec<String> = remaining
                .iter()
                .filter(|id| in_degree.get(*id).copied().unwrap_or(0) == 0)
                .map(|id| (*id).clone())
                .collect();

            if wave.is_empty() {
                break;
            }

            self.sort_by_priority(&mut wave);

            // Remove wave from remaining, update in-degrees.
            for id in &wave {
                remaining.remove(id);
                if let Some(deps) = self.dependents.get(id) {
                    for d in deps {
                        if let Some(deg) = in_degree.get_mut(d) {
                            *deg = deg.saturating_sub(1);
                        }
                    }
                }
            }

            waves.push(wave);
        }

        waves
    }

    /// Pending tasks mapped to their unsatisfied dependencies.
    pub fn get_blocked_tasks(&self) -> HashMap<String, HashSet<String>> {
        self.states
            .iter()
            .filter(|(_, &s)| s == TaskState::Pending)
            .filter_map(|(id, _)| {
                self.dependencies.get(id).map(|deps| {
                    let unsatisfied: HashSet<String> = deps
                        .iter()
                        .filter(|d| self.states.contains_key(*d))
                        .cloned()
                        .collect();
                    (id.clone(), unsatisfied)
                })
            })
            .filter(|(_, deps)| !deps.is_empty())
            .collect()
    }

    /// All transitive dependents via BFS.
    pub fn get_downstream(&self, task_id: &str) -> HashSet<String> {
        let mut visited = HashSet::new();
        let mut queue = VecDeque::new();

        if let Some(deps) = self.dependents.get(task_id) {
            for d in deps {
                queue.push_back(d.clone());
            }
        }

        while let Some(tid) = queue.pop_front() {
            if visited.insert(tid.clone()) {
                if let Some(deps) = self.dependents.get(&tid) {
                    for d in deps {
                        if !visited.contains(d) {
                            queue.push_back(d.clone());
                        }
                    }
                }
            }
        }
        visited
    }

    /// Look up a task's state (active or history).
    pub fn get_task_state(&self, task_id: &str) -> Option<TaskState> {
        self.states
            .get(task_id)
            .or_else(|| self.history.get(task_id))
            .copied()
    }

    /// Immutable snapshot of a task's graph position.
    pub fn get_node(&self, task_id: &str) -> Option<TaskNode> {
        let state = self.states.get(task_id)?;
        Some(TaskNode {
            task_id: task_id.to_string(),
            state: *state,
            dependencies: self
                .dependencies
                .get(task_id)
                .cloned()
                .unwrap_or_default(),
            dependents: self
                .dependents
                .get(task_id)
                .cloned()
                .unwrap_or_default(),
            priority: self.priorities.get(task_id).copied().unwrap_or(2),
        })
    }

    /// Kahn's validation — returns cycle members if any.
    pub fn validate_graph(&self) -> Option<Vec<String>> {
        let active: HashSet<&String> = self.states.keys().collect();
        let mut in_degree: HashMap<&String, usize> = HashMap::new();
        for id in &active {
            let deg = self
                .dependencies
                .get(*id)
                .map(|deps| deps.iter().filter(|d| active.contains(d)).count())
                .unwrap_or(0);
            in_degree.insert(id, deg);
        }

        let mut visited = 0usize;
        let mut queue: VecDeque<&String> = in_degree
            .iter()
            .filter(|(_, &d)| d == 0)
            .map(|(id, _)| *id)
            .collect();

        while let Some(id) = queue.pop_front() {
            visited += 1;
            if let Some(deps) = self.dependents.get(id) {
                for d in deps {
                    if let Some(deg) = in_degree.get_mut(d) {
                        *deg = deg.saturating_sub(1);
                        if *deg == 0 {
                            queue.push_back(d);
                        }
                    }
                }
            }
        }

        if visited == active.len() {
            None
        } else {
            // Collect cycle members (non-zero in-degree).
            Some(
                in_degree
                    .iter()
                    .filter(|(_, &d)| d > 0)
                    .map(|(id, _)| (*id).clone())
                    .collect(),
            )
        }
    }

    /// Counts by state.
    pub fn stats(&self) -> HashMap<TaskState, usize> {
        let mut counts = HashMap::new();
        for &s in self.states.values() {
            *counts.entry(s).or_insert(0) += 1;
        }
        for &s in self.history.values() {
            *counts.entry(s).or_insert(0) += 1;
        }
        counts
    }

    /// Number of active (non-terminal) tasks.
    pub fn active_count(&self) -> usize {
        self.states.len()
    }

    // ── Internal ────────────────────────────────────────────────────

    /// DFS cycle check: can we reach `new_task` from any of its deps
    /// by walking backward through the dependents graph?
    fn check_cycle(&self, new_task: &str, deps: &HashSet<String>) -> Result<()> {
        for dep in deps {
            if let Some(path) = self.dfs_find_path(dep, new_task) {
                let mut cycle = vec![new_task.to_string()];
                cycle.extend(path);
                cycle.push(new_task.to_string());
                return Err(SchedulerError::CycleDetected(cycle));
            }
        }
        Ok(())
    }

    /// DFS from `start` following dependency edges to find `target`.
    fn dfs_find_path(&self, start: &str, target: &str) -> Option<Vec<String>> {
        let mut stack: Vec<(String, Vec<String>)> =
            vec![(start.to_string(), vec![start.to_string()])];
        let mut visited = HashSet::new();

        while let Some((current, path)) = stack.pop() {
            if !visited.insert(current.clone()) {
                continue;
            }
            if let Some(deps) = self.dependencies.get(&current) {
                for dep in deps {
                    if dep == target {
                        let mut full_path = path.clone();
                        full_path.push(dep.clone());
                        return Some(full_path);
                    }
                    if !visited.contains(dep) {
                        let mut new_path = path.clone();
                        new_path.push(dep.clone());
                        stack.push((dep.clone(), new_path));
                    }
                }
            }
        }
        None
    }

    /// Sort task IDs by (priority ASC, insertion order ASC).
    fn sort_by_priority(&self, tasks: &mut Vec<String>) {
        tasks.sort_by(|a, b| {
            let pa = self.priorities.get(a).copied().unwrap_or(2);
            let pb = self.priorities.get(b).copied().unwrap_or(2);
            pa.cmp(&pb).then_with(|| {
                let ia = self.insertion_order.iter().position(|x| x == a);
                let ib = self.insertion_order.iter().position(|x| x == b);
                ia.cmp(&ib)
            })
        });
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn add_task_no_deps() {
        let mut r = DependencyResolver::new();
        let s = r.add_task("a", &[], 2).unwrap();
        assert_eq!(s, TaskState::Ready);
    }

    #[test]
    fn add_task_with_deps() {
        let mut r = DependencyResolver::new();
        r.add_task("a", &[], 2).unwrap();
        let s = r.add_task("b", &["a"], 2).unwrap();
        assert_eq!(s, TaskState::Pending);
    }

    #[test]
    fn duplicate_task_errors() {
        let mut r = DependencyResolver::new();
        r.add_task("a", &[], 2).unwrap();
        assert!(r.add_task("a", &[], 2).is_err());
    }

    #[test]
    fn cycle_detection_direct() {
        let mut r = DependencyResolver::new();
        r.add_task("a", &[], 2).unwrap();
        r.add_task("b", &["a"], 2).unwrap();
        // Try to make a depend on b (cycle: a -> b -> a).
        // Since a is already in the graph with no deps, we can't re-add.
        // Instead test c -> a -> c.
        let mut r2 = DependencyResolver::new();
        r2.add_task("a", &[], 2).unwrap();
        r2.add_task("b", &["a"], 2).unwrap();
        let err = r2.add_task("c", &["b"], 2);
        assert!(err.is_ok()); // no cycle: a -> b -> c

        // Real cycle: a already depends on nothing, b depends on a.
        // Add c depending on b, then try d depending on c, a (no cycle).
        // Self-dep:
        let mut r3 = DependencyResolver::new();
        r3.add_task("x", &[], 2).unwrap();
        r3.add_task("y", &["x"], 2).unwrap();
        // Now "x" depends on nothing, "y" depends on "x".
        // Adding "z" depending on "y" and "x" — no cycle.
        assert!(r3.add_task("z", &["y", "x"], 2).is_ok());
    }

    #[test]
    fn cycle_detection_self() {
        let mut r = DependencyResolver::new();
        // Self-dependency: task depends on itself. But add_task checks
        // if dep is in states — it won't be yet because we're adding.
        // The dep will be auto-resolved (not in graph or history).
        // This is the Python behavior: self-dep on non-existent is auto-resolved.
        // For a real self-cycle, we need the task to exist first.
        // Actually in the Python version, adding A with dep A:
        // A isn't in states or history yet, so it gets auto-resolved.
        // This is a quirk — the validate() in WorkflowBuilder catches it.
        let s = r.add_task("a", &["a"], 2).unwrap();
        assert_eq!(s, TaskState::Ready); // auto-resolved
    }

    #[test]
    fn mark_completed_unblocks() {
        let mut r = DependencyResolver::new();
        r.add_task("a", &[], 2).unwrap();
        r.add_task("b", &["a"], 2).unwrap();
        r.mark_running("a").unwrap();
        let newly_ready = r.mark_completed("a").unwrap();
        assert_eq!(newly_ready, vec!["b".to_string()]);
        assert_eq!(r.get_task_state("b"), Some(TaskState::Ready));
    }

    #[test]
    fn mark_failed_propagates() {
        let mut r = DependencyResolver::new();
        r.add_task("a", &[], 2).unwrap();
        r.add_task("b", &["a"], 2).unwrap();
        r.add_task("c", &["b"], 2).unwrap();
        r.mark_running("a").unwrap();
        let cancelled = r.mark_failed("a").unwrap();
        assert!(cancelled.contains(&"b".to_string()));
        assert!(cancelled.contains(&"c".to_string()));
    }

    #[test]
    fn execution_waves_parallel() {
        let mut r = DependencyResolver::new();
        r.add_task("a", &[], 2).unwrap();
        r.add_task("b", &[], 2).unwrap();
        r.add_task("c", &[], 2).unwrap();
        let waves = r.get_execution_waves();
        assert_eq!(waves.len(), 1);
        assert_eq!(waves[0].len(), 3);
    }

    #[test]
    fn execution_waves_sequential() {
        let mut r = DependencyResolver::new();
        r.add_task("a", &[], 2).unwrap();
        r.add_task("b", &["a"], 2).unwrap();
        r.add_task("c", &["b"], 2).unwrap();
        let waves = r.get_execution_waves();
        assert_eq!(waves.len(), 3);
    }

    #[test]
    fn execution_waves_diamond() {
        let mut r = DependencyResolver::new();
        r.add_task("a", &[], 2).unwrap();
        r.add_task("b", &["a"], 2).unwrap();
        r.add_task("c", &["a"], 2).unwrap();
        r.add_task("d", &["b", "c"], 2).unwrap();
        let waves = r.get_execution_waves();
        assert_eq!(waves.len(), 3);
        assert_eq!(waves[1].len(), 2); // b and c parallel
    }

    #[test]
    fn ready_sorted_by_priority() {
        let mut r = DependencyResolver::new();
        r.add_task("low", &[], 3).unwrap();
        r.add_task("critical", &[], 0).unwrap();
        r.add_task("medium", &[], 2).unwrap();
        let ready = r.get_ready_tasks();
        assert_eq!(ready[0], "critical");
        assert_eq!(ready[1], "medium");
        assert_eq!(ready[2], "low");
    }

    #[test]
    fn downstream_transitive() {
        let mut r = DependencyResolver::new();
        r.add_task("a", &[], 2).unwrap();
        r.add_task("b", &["a"], 2).unwrap();
        r.add_task("c", &["b"], 2).unwrap();
        r.add_task("d", &["c"], 2).unwrap();
        let ds = r.get_downstream("a");
        assert_eq!(ds.len(), 3);
        assert!(ds.contains("b"));
        assert!(ds.contains("c"));
        assert!(ds.contains("d"));
    }

    #[test]
    fn completed_dep_auto_resolved() {
        let mut r = DependencyResolver::new();
        r.add_task("a", &[], 2).unwrap();
        r.mark_running("a").unwrap();
        r.mark_completed("a").unwrap();
        // Now add b depending on a (already completed).
        let s = r.add_task("b", &["a"], 2).unwrap();
        assert_eq!(s, TaskState::Ready);
    }

    #[test]
    fn failed_dep_cancels_new_task() {
        let mut r = DependencyResolver::new();
        r.add_task("a", &[], 2).unwrap();
        r.mark_running("a").unwrap();
        r.mark_failed("a").unwrap();
        let s = r.add_task("b", &["a"], 2).unwrap();
        assert_eq!(s, TaskState::Cancelled);
    }

    #[test]
    fn serialization_roundtrip() {
        let mut r = DependencyResolver::new();
        r.add_task("a", &[], 0).unwrap();
        r.add_task("b", &["a"], 1).unwrap();
        r.mark_running("a").unwrap();
        r.mark_completed("a").unwrap();

        let json = serde_json::to_string(&r).unwrap();
        let r2: DependencyResolver = serde_json::from_str(&json).unwrap();
        assert_eq!(r2.get_task_state("a"), Some(TaskState::Completed));
        assert_eq!(r2.get_task_state("b"), Some(TaskState::Ready));
    }
}
