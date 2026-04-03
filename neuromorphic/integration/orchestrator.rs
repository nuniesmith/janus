//! System Orchestrator — Workflow management and scheduling
//!
//! The `SystemOrchestrator` is the top-level control plane for the JANUS
//! neuromorphic architecture. It sits above the [`BrainCoordinator`] and
//! manages high-level workflows such as:
//!
//! - Scheduling and executing multi-step workflows (e.g. "ingest → analyse → decide → execute")
//! - Managing named workflow definitions with step ordering
//! - Tracking workflow execution history and statistics
//! - Providing a unified start/stop/pause interface for the entire system
//! - Health aggregation and system-wide status reporting
//!
//! # Architecture
//!
//! ```text
//! ┌─────────────────────────────────────────────────────────────────────┐
//! │                      SystemOrchestrator                              │
//! ├─────────────────────────────────────────────────────────────────────┤
//! │                                                                     │
//! │  ┌──────────────────────────────────────────────────────────────┐   │
//! │  │                  Workflow Registry                            │   │
//! │  │  ┌──────────┐ ┌──────────┐ ┌──────────┐ ┌──────────────┐   │   │
//! │  │  │ Ingest   │ │ Analyse  │ │ Decide   │ │   Execute    │   │   │
//! │  │  │ Workflow │ │ Workflow │ │ Workflow │ │   Workflow   │   │   │
//! │  │  └──────────┘ └──────────┘ └──────────┘ └──────────────┘   │   │
//! │  └──────────────────────────────────────────────────────────────┘   │
//! │                              │                                      │
//! │                    ┌─────────▼─────────┐                           │
//! │                    │ BrainCoordinator  │                           │
//! │                    │  (tick engine)    │                           │
//! │                    └──────────────────┘                           │
//! │                                                                     │
//! │  ┌──────────────────────────────────────────────────────────────┐   │
//! │  │                 Execution History                             │   │
//! │  │  Run #1: [Ingest ✓, Analyse ✓, Decide ✓, Execute ✓]         │   │
//! │  │  Run #2: [Ingest ✓, Analyse ✗ — timeout]                    │   │
//! │  └──────────────────────────────────────────────────────────────┘   │
//! │                                                                     │
//! └─────────────────────────────────────────────────────────────────────┘
//! ```
//!
//! # Integration Points
//!
//! - **BrainCoordinator**: The orchestrator delegates tick-level processing
//!   to the coordinator but manages higher-level workflow sequencing.
//! - **Service Bridges**: Workflows may trigger service bridge operations
//!   (e.g. sending training batches to the backward service).
//! - **CNS Health**: The orchestrator aggregates health from the coordinator
//!   and surfaces system-wide health status.

use std::collections::{HashMap, VecDeque};

// ---------------------------------------------------------------------------
// Configuration
// ---------------------------------------------------------------------------

/// Configuration for the SystemOrchestrator.
#[derive(Debug, Clone)]
pub struct OrchestratorConfig {
    /// Maximum number of workflow definitions.
    pub max_workflows: usize,

    /// Maximum execution history entries to retain.
    pub max_history: usize,

    /// Maximum steps per workflow.
    pub max_steps_per_workflow: usize,

    /// Default timeout (in ticks) for a workflow step.
    pub default_step_timeout: u64,

    /// EMA decay factor for orchestrator-level smoothing.
    pub ema_decay: f64,

    /// Rolling window size for windowed statistics.
    pub window_size: usize,

    /// Whether to automatically schedule the "default" workflow on start.
    pub auto_schedule_default: bool,
}

impl Default for OrchestratorConfig {
    fn default() -> Self {
        Self {
            max_workflows: 32,
            max_history: 256,
            max_steps_per_workflow: 16,
            default_step_timeout: 100,
            ema_decay: 0.1,
            window_size: 64,
            auto_schedule_default: false,
        }
    }
}

// ---------------------------------------------------------------------------
// Workflow definition
// ---------------------------------------------------------------------------

/// A single step in a workflow.
#[derive(Debug, Clone)]
pub struct WorkflowStep {
    /// Step name (e.g. "ingest", "analyse", "decide").
    pub name: String,

    /// Which brain region (or pseudo-region) should execute this step.
    pub region: String,

    /// Optional timeout in ticks (overrides default).
    pub timeout: Option<u64>,

    /// Whether this step is optional (failure won't abort the workflow).
    pub optional: bool,

    /// Descriptive label for logging / monitoring.
    pub description: String,
}

impl WorkflowStep {
    /// Create a new required workflow step.
    pub fn new(name: impl Into<String>, region: impl Into<String>) -> Self {
        Self {
            name: name.into(),
            region: region.into(),
            timeout: None,
            optional: false,
            description: String::new(),
        }
    }

    /// Set the step as optional.
    pub fn optional(mut self) -> Self {
        self.optional = true;
        self
    }

    /// Set a custom timeout.
    pub fn with_timeout(mut self, ticks: u64) -> Self {
        self.timeout = Some(ticks);
        self
    }

    /// Set a description.
    pub fn with_description(mut self, desc: impl Into<String>) -> Self {
        self.description = desc.into();
        self
    }
}

/// A named workflow comprising an ordered sequence of steps.
#[derive(Debug, Clone)]
pub struct WorkflowDefinition {
    /// Unique workflow name.
    pub name: String,

    /// Human-readable description.
    pub description: String,

    /// Ordered list of steps.
    pub steps: Vec<WorkflowStep>,

    /// Whether this workflow is enabled.
    pub enabled: bool,

    /// Priority (lower = higher priority when multiple are scheduled).
    pub priority: u32,
}

impl WorkflowDefinition {
    /// Create a new workflow definition.
    pub fn new(name: impl Into<String>) -> Self {
        Self {
            name: name.into(),
            description: String::new(),
            steps: Vec::new(),
            enabled: true,
            priority: 100,
        }
    }

    /// Set description.
    pub fn with_description(mut self, desc: impl Into<String>) -> Self {
        self.description = desc.into();
        self
    }

    /// Add a step.
    pub fn with_step(mut self, step: WorkflowStep) -> Self {
        self.steps.push(step);
        self
    }

    /// Set priority.
    pub fn with_priority(mut self, priority: u32) -> Self {
        self.priority = priority;
        self
    }

    /// Number of steps.
    pub fn step_count(&self) -> usize {
        self.steps.len()
    }
}

// ---------------------------------------------------------------------------
// Execution tracking
// ---------------------------------------------------------------------------

/// Outcome of a single step execution.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum StepOutcome {
    /// Step completed successfully.
    Success,
    /// Step failed.
    Failed,
    /// Step was skipped (e.g. optional + precondition not met).
    Skipped,
    /// Step timed out.
    TimedOut,
    /// Step is still running.
    Running,
    /// Step has not started yet.
    Pending,
}

impl std::fmt::Display for StepOutcome {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        match self {
            StepOutcome::Success => write!(f, "Success"),
            StepOutcome::Failed => write!(f, "Failed"),
            StepOutcome::Skipped => write!(f, "Skipped"),
            StepOutcome::TimedOut => write!(f, "TimedOut"),
            StepOutcome::Running => write!(f, "Running"),
            StepOutcome::Pending => write!(f, "Pending"),
        }
    }
}

/// Record of a single step's execution.
#[derive(Debug, Clone)]
pub struct StepRecord {
    /// Step name.
    pub step_name: String,

    /// Outcome of this step.
    pub outcome: StepOutcome,

    /// Tick at which this step started.
    pub started_tick: u64,

    /// Tick at which this step completed (0 if still running/pending).
    pub completed_tick: u64,

    /// Duration in ticks.
    pub duration_ticks: u64,
}

/// Outcome of a complete workflow execution.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum WorkflowOutcome {
    /// All required steps succeeded.
    Completed,
    /// At least one required step failed or timed out.
    Failed,
    /// Workflow was cancelled before completion.
    Cancelled,
    /// Workflow is still in progress.
    InProgress,
}

impl std::fmt::Display for WorkflowOutcome {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        match self {
            WorkflowOutcome::Completed => write!(f, "Completed"),
            WorkflowOutcome::Failed => write!(f, "Failed"),
            WorkflowOutcome::Cancelled => write!(f, "Cancelled"),
            WorkflowOutcome::InProgress => write!(f, "InProgress"),
        }
    }
}

/// Record of a complete workflow execution.
#[derive(Debug, Clone)]
pub struct WorkflowExecution {
    /// Execution run ID.
    pub run_id: u64,

    /// Name of the workflow that was executed.
    pub workflow_name: String,

    /// Overall outcome.
    pub outcome: WorkflowOutcome,

    /// Per-step execution records.
    pub steps: Vec<StepRecord>,

    /// Tick at which this execution started.
    pub started_tick: u64,

    /// Tick at which this execution completed.
    pub completed_tick: u64,

    /// Index of the currently-executing step (if InProgress).
    pub current_step: usize,
}

impl WorkflowExecution {
    /// Total duration in ticks.
    pub fn duration_ticks(&self) -> u64 {
        self.completed_tick.saturating_sub(self.started_tick)
    }

    /// Number of successful steps.
    pub fn successful_steps(&self) -> usize {
        self.steps
            .iter()
            .filter(|s| s.outcome == StepOutcome::Success)
            .count()
    }

    /// Number of failed steps.
    pub fn failed_steps(&self) -> usize {
        self.steps
            .iter()
            .filter(|s| s.outcome == StepOutcome::Failed || s.outcome == StepOutcome::TimedOut)
            .count()
    }
}

// ---------------------------------------------------------------------------
// Orchestrator lifecycle
// ---------------------------------------------------------------------------

/// Current lifecycle phase of the orchestrator.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum OrchestratorPhase {
    /// Created but not started.
    Idle,
    /// Actively processing workflows.
    Active,
    /// Paused — no new workflows will be started.
    Paused,
    /// Shutting down.
    Draining,
    /// Fully stopped.
    Stopped,
}

impl std::fmt::Display for OrchestratorPhase {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        match self {
            OrchestratorPhase::Idle => write!(f, "Idle"),
            OrchestratorPhase::Active => write!(f, "Active"),
            OrchestratorPhase::Paused => write!(f, "Paused"),
            OrchestratorPhase::Draining => write!(f, "Draining"),
            OrchestratorPhase::Stopped => write!(f, "Stopped"),
        }
    }
}

// ---------------------------------------------------------------------------
// Windowed tick snapshot
// ---------------------------------------------------------------------------

#[derive(Debug, Clone)]
#[allow(dead_code)]
struct TickSnapshot {
    workflows_active: usize,
    steps_completed: u64,
    steps_failed: u64,
}

// ---------------------------------------------------------------------------
// Orchestrator statistics
// ---------------------------------------------------------------------------

/// Operational statistics for the orchestrator.
#[derive(Debug, Clone, Default)]
pub struct OrchestratorStats {
    /// Total ticks processed.
    pub total_ticks: u64,

    /// Total workflows registered.
    pub total_workflows_registered: u64,

    /// Total workflow executions started.
    pub total_executions_started: u64,

    /// Total workflow executions completed (success).
    pub total_executions_completed: u64,

    /// Total workflow executions failed.
    pub total_executions_failed: u64,

    /// Total workflow executions cancelled.
    pub total_executions_cancelled: u64,

    /// Total individual steps completed.
    pub total_steps_completed: u64,

    /// Total individual steps failed.
    pub total_steps_failed: u64,

    /// Total individual steps skipped.
    pub total_steps_skipped: u64,

    /// Total individual steps timed out.
    pub total_steps_timed_out: u64,

    /// EMA of active workflows per tick.
    pub ema_active_workflows: f64,

    /// EMA of step completion rate.
    pub ema_step_completion_rate: f64,
}

impl OrchestratorStats {
    /// Workflow success rate.
    pub fn workflow_success_rate(&self) -> f64 {
        let total = self.total_executions_completed + self.total_executions_failed;
        if total == 0 {
            return 1.0;
        }
        self.total_executions_completed as f64 / total as f64
    }

    /// Step success rate.
    pub fn step_success_rate(&self) -> f64 {
        let total =
            self.total_steps_completed + self.total_steps_failed + self.total_steps_timed_out;
        if total == 0 {
            return 1.0;
        }
        self.total_steps_completed as f64 / total as f64
    }
}

// ---------------------------------------------------------------------------
// SystemOrchestrator
// ---------------------------------------------------------------------------

/// Top-level workflow orchestrator for the JANUS neuromorphic system.
///
/// See [module documentation](self) for architecture details.
pub struct SystemOrchestrator {
    config: OrchestratorConfig,

    /// Current lifecycle phase.
    phase: OrchestratorPhase,

    /// Registered workflow definitions keyed by name.
    workflows: HashMap<String, WorkflowDefinition>,

    /// Currently running workflow executions.
    active_executions: Vec<WorkflowExecution>,

    /// Queue of workflow names waiting to be started.
    pending_queue: VecDeque<String>,

    /// Execution history (most recent at the back).
    history: VecDeque<WorkflowExecution>,

    /// Next run ID counter.
    next_run_id: u64,

    /// Current tick counter.
    tick: u64,

    /// Whether EMA values have been initialized.
    ema_initialized: bool,

    /// Rolling window of recent tick snapshots.
    recent: VecDeque<TickSnapshot>,

    /// Accumulated statistics.
    stats: OrchestratorStats,
}

impl Default for SystemOrchestrator {
    fn default() -> Self {
        Self::new()
    }
}

impl SystemOrchestrator {
    // -------------------------------------------------------------------
    // Construction
    // -------------------------------------------------------------------

    /// Create a new orchestrator with default configuration.
    pub fn new() -> Self {
        Self::with_config(OrchestratorConfig::default())
    }

    /// Create a new orchestrator with custom configuration.
    pub fn with_config(config: OrchestratorConfig) -> Self {
        let window_size = config.window_size;
        Self {
            config,
            phase: OrchestratorPhase::Idle,
            workflows: HashMap::new(),
            active_executions: Vec::new(),
            pending_queue: VecDeque::new(),
            history: VecDeque::new(),
            next_run_id: 1,
            tick: 0,
            ema_initialized: false,
            recent: VecDeque::with_capacity(window_size + 1),
            stats: OrchestratorStats::default(),
        }
    }

    // -------------------------------------------------------------------
    // Workflow registration
    // -------------------------------------------------------------------

    /// Register a workflow definition.
    ///
    /// Returns `true` if the workflow was newly registered, `false` if a
    /// workflow with that name already exists or the registry is at capacity.
    pub fn register_workflow(&mut self, workflow: WorkflowDefinition) -> bool {
        if self.workflows.len() >= self.config.max_workflows {
            return false;
        }
        if self.workflows.contains_key(&workflow.name) {
            return false;
        }
        if workflow.steps.len() > self.config.max_steps_per_workflow {
            return false;
        }

        self.workflows.insert(workflow.name.clone(), workflow);
        self.stats.total_workflows_registered += 1;
        true
    }

    /// Remove a workflow definition.
    ///
    /// Returns `true` if the workflow existed and was removed.
    pub fn remove_workflow(&mut self, name: &str) -> bool {
        self.workflows.remove(name).is_some()
    }

    /// Get a reference to a workflow definition.
    pub fn workflow(&self, name: &str) -> Option<&WorkflowDefinition> {
        self.workflows.get(name)
    }

    /// Number of registered workflows.
    pub fn workflow_count(&self) -> usize {
        self.workflows.len()
    }

    /// Check if a workflow is registered.
    pub fn has_workflow(&self, name: &str) -> bool {
        self.workflows.contains_key(name)
    }

    /// List all registered workflow names.
    pub fn workflow_names(&self) -> Vec<String> {
        let mut names: Vec<String> = self.workflows.keys().cloned().collect();
        names.sort();
        names
    }

    // -------------------------------------------------------------------
    // Lifecycle management
    // -------------------------------------------------------------------

    /// Activate the orchestrator (start accepting and executing workflows).
    pub fn activate(&mut self) -> bool {
        if self.phase != OrchestratorPhase::Idle && self.phase != OrchestratorPhase::Paused {
            return false;
        }
        self.phase = OrchestratorPhase::Active;
        true
    }

    /// Pause the orchestrator (stop starting new workflows, let active ones finish).
    pub fn pause(&mut self) -> bool {
        if self.phase != OrchestratorPhase::Active {
            return false;
        }
        self.phase = OrchestratorPhase::Paused;
        true
    }

    /// Resume from paused state.
    pub fn resume(&mut self) -> bool {
        if self.phase != OrchestratorPhase::Paused {
            return false;
        }
        self.phase = OrchestratorPhase::Active;
        true
    }

    /// Begin draining: cancel pending workflows and wait for active ones to finish.
    pub fn drain(&mut self) -> bool {
        if self.phase == OrchestratorPhase::Stopped || self.phase == OrchestratorPhase::Draining {
            return false;
        }

        self.pending_queue.clear();
        self.phase = OrchestratorPhase::Draining;

        // If no active executions, go straight to stopped
        if self.active_executions.is_empty() {
            self.phase = OrchestratorPhase::Stopped;
        }
        true
    }

    /// Force-stop the orchestrator, cancelling everything.
    pub fn stop(&mut self) -> bool {
        if self.phase == OrchestratorPhase::Stopped {
            return false;
        }

        // Cancel all active executions
        for exec in &mut self.active_executions {
            exec.outcome = WorkflowOutcome::Cancelled;
            exec.completed_tick = self.tick;
            self.stats.total_executions_cancelled += 1;
        }

        // Move to history
        let cancelled: Vec<WorkflowExecution> = self.active_executions.drain(..).collect();
        for exec in cancelled {
            self.push_history(exec);
        }

        self.pending_queue.clear();
        self.phase = OrchestratorPhase::Stopped;
        true
    }

    /// Get the current lifecycle phase.
    pub fn phase(&self) -> OrchestratorPhase {
        self.phase
    }

    /// Whether the orchestrator is actively executing workflows.
    pub fn active(&self) -> bool {
        self.phase == OrchestratorPhase::Active
    }

    // -------------------------------------------------------------------
    // Workflow scheduling
    // -------------------------------------------------------------------

    /// Schedule a workflow for execution.
    ///
    /// The workflow is added to the pending queue and will be started on
    /// the next tick (if the orchestrator is active).
    ///
    /// Returns `true` if the workflow was successfully queued.
    pub fn schedule(&mut self, workflow_name: &str) -> bool {
        if !self.workflows.contains_key(workflow_name) {
            return false;
        }
        if let Some(wf) = self.workflows.get(workflow_name) {
            if !wf.enabled {
                return false;
            }
        }
        self.pending_queue.push_back(workflow_name.to_string());
        true
    }

    /// Schedule a workflow with higher priority (push to front of queue).
    pub fn schedule_priority(&mut self, workflow_name: &str) -> bool {
        if !self.workflows.contains_key(workflow_name) {
            return false;
        }
        if let Some(wf) = self.workflows.get(workflow_name) {
            if !wf.enabled {
                return false;
            }
        }
        self.pending_queue.push_front(workflow_name.to_string());
        true
    }

    /// Number of workflows in the pending queue.
    pub fn pending_count(&self) -> usize {
        self.pending_queue.len()
    }

    /// Number of actively running workflow executions.
    pub fn active_execution_count(&self) -> usize {
        self.active_executions.len()
    }

    /// Cancel a specific active execution by run ID.
    ///
    /// Returns `true` if the execution was found and cancelled.
    pub fn cancel_execution(&mut self, run_id: u64) -> bool {
        if let Some(idx) = self
            .active_executions
            .iter()
            .position(|e| e.run_id == run_id)
        {
            let mut exec = self.active_executions.remove(idx);
            exec.outcome = WorkflowOutcome::Cancelled;
            exec.completed_tick = self.tick;
            self.stats.total_executions_cancelled += 1;
            self.push_history(exec);
            true
        } else {
            false
        }
    }

    // -------------------------------------------------------------------
    // Step completion (external code reports step outcomes)
    // -------------------------------------------------------------------

    /// Mark the current step of an active execution as completed.
    ///
    /// Returns `true` if the step was found and marked.
    pub fn complete_step(&mut self, run_id: u64, outcome: StepOutcome) -> bool {
        if let Some(exec) = self
            .active_executions
            .iter_mut()
            .find(|e| e.run_id == run_id)
        {
            let step_idx = exec.current_step;
            if step_idx >= exec.steps.len() {
                return false;
            }

            exec.steps[step_idx].outcome = outcome;
            exec.steps[step_idx].completed_tick = self.tick;
            exec.steps[step_idx].duration_ticks =
                self.tick.saturating_sub(exec.steps[step_idx].started_tick);

            match outcome {
                StepOutcome::Success => {
                    self.stats.total_steps_completed += 1;
                }
                StepOutcome::Failed => {
                    self.stats.total_steps_failed += 1;
                }
                StepOutcome::Skipped => {
                    self.stats.total_steps_skipped += 1;
                }
                StepOutcome::TimedOut => {
                    self.stats.total_steps_timed_out += 1;
                }
                _ => {}
            }

            // Check if we should advance or complete the workflow
            let is_required_failure = (outcome == StepOutcome::Failed
                || outcome == StepOutcome::TimedOut)
                && step_idx < exec.steps.len();

            // Look up whether this step is optional from the workflow definition
            let step_is_optional = self
                .workflows
                .get(&exec.workflow_name)
                .and_then(|wf| wf.steps.get(step_idx))
                .map(|s| s.optional)
                .unwrap_or(false);

            if is_required_failure && !step_is_optional {
                // Required step failed — workflow fails
                exec.outcome = WorkflowOutcome::Failed;
                exec.completed_tick = self.tick;
                self.stats.total_executions_failed += 1;
            } else if step_idx + 1 >= exec.steps.len() {
                // Last step — workflow completed
                exec.outcome = WorkflowOutcome::Completed;
                exec.completed_tick = self.tick;
                self.stats.total_executions_completed += 1;
            } else {
                // Advance to next step
                exec.current_step = step_idx + 1;
                exec.steps[step_idx + 1].started_tick = self.tick;
                exec.steps[step_idx + 1].outcome = StepOutcome::Running;
            }

            true
        } else {
            false
        }
    }

    /// Get the current step name for an active execution.
    pub fn current_step_name(&self, run_id: u64) -> Option<String> {
        self.active_executions
            .iter()
            .find(|e| e.run_id == run_id)
            .and_then(|exec| exec.steps.get(exec.current_step))
            .map(|s| s.step_name.clone())
    }

    // -------------------------------------------------------------------
    // Tick processing
    // -------------------------------------------------------------------

    /// Process a single orchestrator tick.
    ///
    /// This starts pending workflows, checks for timeouts, and archives
    /// completed executions.
    ///
    /// Returns the number of workflow state changes that occurred.
    pub fn tick(&mut self) -> usize {
        self.tick += 1;
        let mut changes = 0usize;

        // Start pending workflows if we're active
        if self.phase == OrchestratorPhase::Active {
            while let Some(wf_name) = self.pending_queue.pop_front() {
                if let Some(wf) = self.workflows.get(&wf_name) {
                    if !wf.enabled || wf.steps.is_empty() {
                        continue;
                    }

                    let steps: Vec<StepRecord> = wf
                        .steps
                        .iter()
                        .enumerate()
                        .map(|(i, s)| StepRecord {
                            step_name: s.name.clone(),
                            outcome: if i == 0 {
                                StepOutcome::Running
                            } else {
                                StepOutcome::Pending
                            },
                            started_tick: if i == 0 { self.tick } else { 0 },
                            completed_tick: 0,
                            duration_ticks: 0,
                        })
                        .collect();

                    let exec = WorkflowExecution {
                        run_id: self.next_run_id,
                        workflow_name: wf_name.clone(),
                        outcome: WorkflowOutcome::InProgress,
                        steps,
                        started_tick: self.tick,
                        completed_tick: 0,
                        current_step: 0,
                    };

                    self.next_run_id += 1;
                    self.active_executions.push(exec);
                    self.stats.total_executions_started += 1;
                    changes += 1;
                }
            }
        }

        // Check for timeouts on active executions
        let default_timeout = self.config.default_step_timeout;
        let current_tick = self.tick;
        let workflows = &self.workflows;

        for exec in &mut self.active_executions {
            if exec.outcome != WorkflowOutcome::InProgress {
                continue;
            }

            let step_idx = exec.current_step;
            if step_idx >= exec.steps.len() {
                continue;
            }

            let step_started = exec.steps[step_idx].started_tick;
            if step_started == 0 {
                continue;
            }

            let timeout = workflows
                .get(&exec.workflow_name)
                .and_then(|wf| wf.steps.get(step_idx))
                .and_then(|s| s.timeout)
                .unwrap_or(default_timeout);

            if current_tick.saturating_sub(step_started) >= timeout {
                exec.steps[step_idx].outcome = StepOutcome::TimedOut;
                exec.steps[step_idx].completed_tick = current_tick;
                exec.steps[step_idx].duration_ticks = current_tick.saturating_sub(step_started);

                let step_optional = workflows
                    .get(&exec.workflow_name)
                    .and_then(|wf| wf.steps.get(step_idx))
                    .map(|s| s.optional)
                    .unwrap_or(false);

                if step_optional {
                    // Skip to next step
                    if step_idx + 1 < exec.steps.len() {
                        exec.current_step = step_idx + 1;
                        exec.steps[step_idx + 1].started_tick = current_tick;
                        exec.steps[step_idx + 1].outcome = StepOutcome::Running;
                    } else {
                        exec.outcome = WorkflowOutcome::Completed;
                        exec.completed_tick = current_tick;
                    }
                } else {
                    exec.outcome = WorkflowOutcome::Failed;
                    exec.completed_tick = current_tick;
                }

                changes += 1;
            }
        }

        // Archive completed / failed / cancelled executions
        let mut archived = Vec::new();
        self.active_executions.retain(|exec| {
            if exec.outcome != WorkflowOutcome::InProgress {
                archived.push(exec.clone());
                false
            } else {
                true
            }
        });

        for exec in archived {
            match exec.outcome {
                WorkflowOutcome::Completed => { /* already counted in complete_step */ }
                WorkflowOutcome::Failed => { /* already counted */ }
                WorkflowOutcome::Cancelled => { /* already counted */ }
                _ => {}
            }
            self.push_history(exec);
            changes += 1;
        }

        // Record tick snapshot
        let snapshot = TickSnapshot {
            workflows_active: self.active_executions.len(),
            steps_completed: self.stats.total_steps_completed,
            steps_failed: self.stats.total_steps_failed,
        };
        self.recent.push_back(snapshot);
        if self.recent.len() > self.config.window_size {
            self.recent.pop_front();
        }

        // Update EMAs
        let active_count = self.active_executions.len() as f64;
        let step_rate = self.stats.step_success_rate();

        if !self.ema_initialized {
            self.stats.ema_active_workflows = active_count;
            self.stats.ema_step_completion_rate = step_rate;
            self.ema_initialized = true;
        } else {
            let alpha = self.config.ema_decay;
            self.stats.ema_active_workflows =
                alpha * active_count + (1.0 - alpha) * self.stats.ema_active_workflows;
            self.stats.ema_step_completion_rate =
                alpha * step_rate + (1.0 - alpha) * self.stats.ema_step_completion_rate;
        }

        self.stats.total_ticks += 1;

        // If draining and no more active executions, transition to stopped
        if self.phase == OrchestratorPhase::Draining && self.active_executions.is_empty() {
            self.phase = OrchestratorPhase::Stopped;
        }

        changes
    }

    /// Process a tick (alias for compatibility).
    pub fn process(&mut self) -> usize {
        self.tick()
    }

    /// Get the current tick count.
    pub fn current_tick(&self) -> u64 {
        self.tick
    }

    // -------------------------------------------------------------------
    // History & introspection
    // -------------------------------------------------------------------

    /// Get the execution history (oldest first).
    pub fn history(&self) -> &VecDeque<WorkflowExecution> {
        &self.history
    }

    /// Get the most recent execution for a given workflow name.
    pub fn last_execution(&self, workflow_name: &str) -> Option<&WorkflowExecution> {
        self.history
            .iter()
            .rev()
            .find(|e| e.workflow_name == workflow_name)
    }

    /// Get an active execution by run ID.
    pub fn active_execution(&self, run_id: u64) -> Option<&WorkflowExecution> {
        self.active_executions.iter().find(|e| e.run_id == run_id)
    }

    /// Get all active executions.
    pub fn active_executions(&self) -> &[WorkflowExecution] {
        &self.active_executions
    }

    /// Get the execution history length.
    pub fn history_len(&self) -> usize {
        self.history.len()
    }

    // -------------------------------------------------------------------
    // Statistics & analytics
    // -------------------------------------------------------------------

    /// Get operational statistics.
    pub fn stats(&self) -> &OrchestratorStats {
        &self.stats
    }

    /// Get the orchestrator configuration.
    pub fn config(&self) -> &OrchestratorConfig {
        &self.config
    }

    /// Average active workflows over the recent window.
    pub fn windowed_active_workflows(&self) -> f64 {
        if self.recent.is_empty() {
            return 0.0;
        }
        let sum: f64 = self.recent.iter().map(|s| s.workflows_active as f64).sum();
        sum / self.recent.len() as f64
    }

    /// Smoothed (EMA) active workflow count.
    pub fn smoothed_active_workflows(&self) -> f64 {
        self.stats.ema_active_workflows
    }

    /// Smoothed (EMA) step completion rate.
    pub fn smoothed_step_completion_rate(&self) -> f64 {
        self.stats.ema_step_completion_rate
    }

    // -------------------------------------------------------------------
    // Reset
    // -------------------------------------------------------------------

    /// Reset the orchestrator to initial state.
    pub fn reset(&mut self) {
        self.workflows.clear();
        self.active_executions.clear();
        self.pending_queue.clear();
        self.history.clear();
        self.next_run_id = 1;
        self.tick = 0;
        self.ema_initialized = false;
        self.recent.clear();
        self.stats = OrchestratorStats::default();
        self.phase = OrchestratorPhase::Idle;
    }

    // -------------------------------------------------------------------
    // Private helpers
    // -------------------------------------------------------------------

    /// Push an execution record into history, evicting old entries if needed.
    fn push_history(&mut self, exec: WorkflowExecution) {
        self.history.push_back(exec);
        while self.history.len() > self.config.max_history {
            self.history.pop_front();
        }
    }
}

// ---------------------------------------------------------------------------
// Tests
// ---------------------------------------------------------------------------

#[cfg(test)]
mod tests {
    use super::*;

    fn small_config() -> OrchestratorConfig {
        OrchestratorConfig {
            max_workflows: 4,
            max_history: 8,
            max_steps_per_workflow: 6,
            default_step_timeout: 5,
            window_size: 8,
            ..Default::default()
        }
    }

    fn ingest_workflow() -> WorkflowDefinition {
        WorkflowDefinition::new("ingest")
            .with_description("Market data ingestion pipeline")
            .with_step(WorkflowStep::new("fetch", "thalamus").with_description("Fetch market data"))
            .with_step(
                WorkflowStep::new("validate", "thalamus").with_description("Validate data quality"),
            )
            .with_step(
                WorkflowStep::new("store", "hippocampus")
                    .with_description("Store in episodic memory"),
            )
    }

    fn analyse_workflow() -> WorkflowDefinition {
        WorkflowDefinition::new("analyse")
            .with_description("Market analysis pipeline")
            .with_step(
                WorkflowStep::new("regime_detect", "cortex")
                    .with_description("Detect market regime"),
            )
            .with_step(
                WorkflowStep::new("sentiment", "thalamus")
                    .with_description("Analyse sentiment")
                    .optional(),
            )
    }

    // ---------------------------------------------------------------
    // Construction
    // ---------------------------------------------------------------

    #[test]
    fn test_new_default() {
        let orch = SystemOrchestrator::new();
        assert_eq!(orch.phase(), OrchestratorPhase::Idle);
        assert_eq!(orch.workflow_count(), 0);
        assert_eq!(orch.current_tick(), 0);
    }

    #[test]
    fn test_with_config() {
        let orch = SystemOrchestrator::with_config(small_config());
        assert_eq!(orch.config().max_workflows, 4);
        assert_eq!(orch.config().default_step_timeout, 5);
    }

    #[test]
    fn test_default_trait() {
        let orch = SystemOrchestrator::default();
        assert_eq!(orch.phase(), OrchestratorPhase::Idle);
    }

    // ---------------------------------------------------------------
    // Workflow registration
    // ---------------------------------------------------------------

    #[test]
    fn test_register_workflow() {
        let mut orch = SystemOrchestrator::new();
        assert!(orch.register_workflow(ingest_workflow()));
        assert_eq!(orch.workflow_count(), 1);
        assert!(orch.has_workflow("ingest"));
        assert_eq!(orch.stats().total_workflows_registered, 1);
    }

    #[test]
    fn test_register_duplicate() {
        let mut orch = SystemOrchestrator::new();
        assert!(orch.register_workflow(ingest_workflow()));
        assert!(!orch.register_workflow(ingest_workflow()));
        assert_eq!(orch.workflow_count(), 1);
    }

    #[test]
    fn test_register_at_capacity() {
        let mut orch = SystemOrchestrator::with_config(small_config());
        for i in 0..4 {
            let wf = WorkflowDefinition::new(format!("wf_{}", i))
                .with_step(WorkflowStep::new("step", "region"));
            assert!(orch.register_workflow(wf));
        }
        let overflow =
            WorkflowDefinition::new("overflow").with_step(WorkflowStep::new("step", "region"));
        assert!(!orch.register_workflow(overflow));
    }

    #[test]
    fn test_register_too_many_steps() {
        let mut orch = SystemOrchestrator::with_config(small_config());
        let mut wf = WorkflowDefinition::new("big");
        for i in 0..7 {
            // max_steps = 6
            wf = wf.with_step(WorkflowStep::new(format!("step_{}", i), "region"));
        }
        assert!(!orch.register_workflow(wf));
    }

    #[test]
    fn test_remove_workflow() {
        let mut orch = SystemOrchestrator::new();
        orch.register_workflow(ingest_workflow());
        assert!(orch.remove_workflow("ingest"));
        assert!(!orch.has_workflow("ingest"));
    }

    #[test]
    fn test_remove_nonexistent() {
        let mut orch = SystemOrchestrator::new();
        assert!(!orch.remove_workflow("ghost"));
    }

    #[test]
    fn test_workflow_names_sorted() {
        let mut orch = SystemOrchestrator::new();
        orch.register_workflow(ingest_workflow());
        orch.register_workflow(analyse_workflow());
        assert_eq!(orch.workflow_names(), vec!["analyse", "ingest"]);
    }

    #[test]
    fn test_workflow_definition_step_count() {
        let wf = ingest_workflow();
        assert_eq!(wf.step_count(), 3);
    }

    // ---------------------------------------------------------------
    // Lifecycle
    // ---------------------------------------------------------------

    #[test]
    fn test_lifecycle_happy_path() {
        let mut orch = SystemOrchestrator::new();
        assert_eq!(orch.phase(), OrchestratorPhase::Idle);

        assert!(orch.activate());
        assert_eq!(orch.phase(), OrchestratorPhase::Active);
        assert!(orch.active());

        assert!(orch.pause());
        assert_eq!(orch.phase(), OrchestratorPhase::Paused);

        assert!(orch.resume());
        assert_eq!(orch.phase(), OrchestratorPhase::Active);

        assert!(orch.stop());
        assert_eq!(orch.phase(), OrchestratorPhase::Stopped);
    }

    #[test]
    fn test_cannot_activate_when_stopped() {
        let mut orch = SystemOrchestrator::new();
        orch.activate();
        orch.stop();
        assert!(!orch.activate());
    }

    #[test]
    fn test_cannot_pause_when_idle() {
        let mut orch = SystemOrchestrator::new();
        assert!(!orch.pause());
    }

    #[test]
    fn test_cannot_stop_twice() {
        let mut orch = SystemOrchestrator::new();
        orch.activate();
        assert!(orch.stop());
        assert!(!orch.stop());
    }

    #[test]
    fn test_drain_with_no_active() {
        let mut orch = SystemOrchestrator::new();
        orch.activate();
        assert!(orch.drain());
        assert_eq!(orch.phase(), OrchestratorPhase::Stopped);
    }

    // ---------------------------------------------------------------
    // Scheduling
    // ---------------------------------------------------------------

    #[test]
    fn test_schedule_unknown_workflow() {
        let mut orch = SystemOrchestrator::new();
        assert!(!orch.schedule("nonexistent"));
    }

    #[test]
    fn test_schedule_disabled_workflow() {
        let mut orch = SystemOrchestrator::new();
        let mut wf = ingest_workflow();
        wf.enabled = false;
        orch.register_workflow(wf);
        assert!(!orch.schedule("ingest"));
    }

    #[test]
    fn test_schedule_and_start() {
        let mut orch = SystemOrchestrator::with_config(small_config());
        orch.register_workflow(ingest_workflow());
        orch.activate();

        assert!(orch.schedule("ingest"));
        assert_eq!(orch.pending_count(), 1);

        let changes = orch.tick();
        assert!(changes > 0);
        assert_eq!(orch.pending_count(), 0);
        assert_eq!(orch.active_execution_count(), 1);
        assert_eq!(orch.stats().total_executions_started, 1);
    }

    #[test]
    fn test_schedule_priority() {
        let mut orch = SystemOrchestrator::new();
        orch.register_workflow(ingest_workflow());
        orch.register_workflow(analyse_workflow());

        orch.schedule("ingest");
        orch.schedule_priority("analyse");

        // "analyse" should be at the front
        assert_eq!(orch.pending_queue[0], "analyse");
        assert_eq!(orch.pending_queue[1], "ingest");
    }

    // ---------------------------------------------------------------
    // Step completion & workflow lifecycle
    // ---------------------------------------------------------------

    #[test]
    fn test_complete_workflow() {
        let mut orch = SystemOrchestrator::with_config(small_config());
        orch.register_workflow(ingest_workflow());
        orch.activate();
        orch.schedule("ingest");

        // Tick to start the workflow
        orch.tick();
        assert_eq!(orch.active_execution_count(), 1);

        let run_id = orch.active_executions()[0].run_id;
        assert_eq!(orch.current_step_name(run_id).as_deref(), Some("fetch"));

        // Complete step 1: fetch
        assert!(orch.complete_step(run_id, StepOutcome::Success));
        assert_eq!(orch.current_step_name(run_id).as_deref(), Some("validate"));

        // Complete step 2: validate
        assert!(orch.complete_step(run_id, StepOutcome::Success));
        assert_eq!(orch.current_step_name(run_id).as_deref(), Some("store"));

        // Complete step 3: store (last step)
        assert!(orch.complete_step(run_id, StepOutcome::Success));

        // Workflow should be completed
        let exec = orch.active_executions().iter().find(|e| e.run_id == run_id);
        assert!(exec.is_some());
        assert_eq!(exec.unwrap().outcome, WorkflowOutcome::Completed);

        // Tick to archive
        orch.tick();
        assert_eq!(orch.active_execution_count(), 0);
        assert_eq!(orch.history_len(), 1);
        assert_eq!(
            orch.last_execution("ingest").unwrap().outcome,
            WorkflowOutcome::Completed
        );
        assert_eq!(orch.stats().total_executions_completed, 1);
        assert_eq!(orch.stats().total_steps_completed, 3);
    }

    #[test]
    fn test_required_step_failure() {
        let mut orch = SystemOrchestrator::with_config(small_config());
        orch.register_workflow(ingest_workflow());
        orch.activate();
        orch.schedule("ingest");
        orch.tick();

        let run_id = orch.active_executions()[0].run_id;

        // Fail step 1 (required)
        orch.complete_step(run_id, StepOutcome::Failed);

        let exec = orch.active_executions().iter().find(|e| e.run_id == run_id);
        assert_eq!(exec.unwrap().outcome, WorkflowOutcome::Failed);

        // Archive
        orch.tick();
        assert_eq!(orch.stats().total_executions_failed, 1);
    }

    #[test]
    fn test_optional_step_failure_continues() {
        let mut orch = SystemOrchestrator::with_config(small_config());
        orch.register_workflow(analyse_workflow());
        orch.activate();
        orch.schedule("analyse");
        orch.tick();

        let run_id = orch.active_executions()[0].run_id;

        // Complete step 1: regime_detect
        orch.complete_step(run_id, StepOutcome::Success);

        // Fail step 2: sentiment (optional) — should complete the workflow
        orch.complete_step(run_id, StepOutcome::Failed);

        // Since it's optional and it's the last step, workflow should still complete
        // Actually the logic: optional failure on the last step → the step fails but
        // since it's optional and there's no next step, outcome becomes Completed
        // Wait, let me re-check: the complete_step logic says:
        //   if is_required_failure && !step_is_optional → Failed
        //   else if step_idx + 1 >= steps.len() → Completed
        //   else → advance
        // Since the step IS optional, `is_required_failure && !step_is_optional` is false,
        // and since step_idx + 1 >= steps.len(), it goes to Completed. ✓

        let exec = orch.active_executions().iter().find(|e| e.run_id == run_id);
        assert_eq!(exec.unwrap().outcome, WorkflowOutcome::Completed);
    }

    #[test]
    fn test_cancel_execution() {
        let mut orch = SystemOrchestrator::with_config(small_config());
        orch.register_workflow(ingest_workflow());
        orch.activate();
        orch.schedule("ingest");
        orch.tick();

        let run_id = orch.active_executions()[0].run_id;
        assert!(orch.cancel_execution(run_id));
        assert_eq!(orch.active_execution_count(), 0);
        assert_eq!(orch.history_len(), 1);
        assert_eq!(orch.stats().total_executions_cancelled, 1);
    }

    #[test]
    fn test_cancel_nonexistent() {
        let mut orch = SystemOrchestrator::new();
        assert!(!orch.cancel_execution(999));
    }

    // ---------------------------------------------------------------
    // Timeout
    // ---------------------------------------------------------------

    #[test]
    fn test_step_timeout() {
        let mut orch = SystemOrchestrator::with_config(small_config());
        orch.register_workflow(ingest_workflow()); // default timeout = 5 ticks
        orch.activate();
        orch.schedule("ingest");
        orch.tick(); // tick 1: starts workflow, step "fetch" begins

        // Don't complete the step — let it time out
        for _ in 0..5 {
            orch.tick();
        }

        // After 5 more ticks, the step should have timed out and the workflow failed
        // (since "fetch" is a required step)
        assert_eq!(orch.active_execution_count(), 0);
        assert_eq!(orch.history_len(), 1);
        let last = orch.last_execution("ingest").unwrap();
        assert_eq!(last.outcome, WorkflowOutcome::Failed);
        assert_eq!(last.steps[0].outcome, StepOutcome::TimedOut);
    }

    #[test]
    fn test_optional_step_timeout_continues() {
        let mut orch = SystemOrchestrator::with_config(small_config());
        orch.register_workflow(analyse_workflow());
        orch.activate();
        orch.schedule("analyse");
        orch.tick(); // starts workflow

        let run_id = orch.active_executions()[0].run_id;

        // Complete step 1 immediately
        orch.complete_step(run_id, StepOutcome::Success);

        // Now step 2 (optional "sentiment") is running. Let it time out.
        for _ in 0..6 {
            orch.tick();
        }

        // The optional step should time out, and since it's the last step,
        // the workflow should complete
        assert_eq!(orch.active_execution_count(), 0);
        let last = orch.last_execution("analyse").unwrap();
        assert_eq!(last.outcome, WorkflowOutcome::Completed);
    }

    // ---------------------------------------------------------------
    // History management
    // ---------------------------------------------------------------

    #[test]
    fn test_history_eviction() {
        let mut orch = SystemOrchestrator::with_config(small_config()); // max_history = 8
        let wf =
            WorkflowDefinition::new("quick").with_step(WorkflowStep::new("only_step", "region"));
        orch.register_workflow(wf);
        orch.activate();

        for _ in 0..12 {
            orch.schedule("quick");
            orch.tick(); // start
            let run_id = orch.active_executions()[0].run_id;
            orch.complete_step(run_id, StepOutcome::Success);
            orch.tick(); // archive
        }

        assert!(orch.history_len() <= 8);
    }

    // ---------------------------------------------------------------
    // Statistics
    // ---------------------------------------------------------------

    #[test]
    fn test_workflow_success_rate() {
        let stats = OrchestratorStats {
            total_executions_completed: 8,
            total_executions_failed: 2,
            ..Default::default()
        };
        assert!((stats.workflow_success_rate() - 0.8).abs() < 1e-10);
    }

    #[test]
    fn test_workflow_success_rate_none() {
        let stats = OrchestratorStats::default();
        assert!((stats.workflow_success_rate() - 1.0).abs() < 1e-10);
    }

    #[test]
    fn test_step_success_rate() {
        let stats = OrchestratorStats {
            total_steps_completed: 9,
            total_steps_failed: 1,
            total_steps_timed_out: 0,
            ..Default::default()
        };
        assert!((stats.step_success_rate() - 0.9).abs() < 1e-10);
    }

    #[test]
    fn test_workflow_execution_helpers() {
        let exec = WorkflowExecution {
            run_id: 1,
            workflow_name: "test".to_string(),
            outcome: WorkflowOutcome::Completed,
            steps: vec![
                StepRecord {
                    step_name: "a".to_string(),
                    outcome: StepOutcome::Success,
                    started_tick: 1,
                    completed_tick: 3,
                    duration_ticks: 2,
                },
                StepRecord {
                    step_name: "b".to_string(),
                    outcome: StepOutcome::Failed,
                    started_tick: 3,
                    completed_tick: 5,
                    duration_ticks: 2,
                },
            ],
            started_tick: 1,
            completed_tick: 5,
            current_step: 1,
        };

        assert_eq!(exec.duration_ticks(), 4);
        assert_eq!(exec.successful_steps(), 1);
        assert_eq!(exec.failed_steps(), 1);
    }

    // ---------------------------------------------------------------
    // Windowed analytics
    // ---------------------------------------------------------------

    #[test]
    fn test_windowed_active_empty() {
        let orch = SystemOrchestrator::new();
        assert!((orch.windowed_active_workflows() - 0.0).abs() < 1e-10);
    }

    #[test]
    fn test_windowed_analytics_with_ticks() {
        let mut orch = SystemOrchestrator::with_config(small_config());
        let wf =
            WorkflowDefinition::new("quick").with_step(WorkflowStep::new("only_step", "region"));
        orch.register_workflow(wf);
        orch.activate();
        orch.schedule("quick");

        for _ in 0..5 {
            orch.tick();
        }

        assert!(orch.stats().total_ticks >= 5);
    }

    // ---------------------------------------------------------------
    // Reset
    // ---------------------------------------------------------------

    #[test]
    fn test_reset() {
        let mut orch = SystemOrchestrator::with_config(small_config());
        orch.register_workflow(ingest_workflow());
        orch.activate();
        orch.schedule("ingest");
        orch.tick();
        orch.tick();

        orch.reset();

        assert_eq!(orch.phase(), OrchestratorPhase::Idle);
        assert_eq!(orch.workflow_count(), 0);
        assert_eq!(orch.current_tick(), 0);
        assert_eq!(orch.pending_count(), 0);
        assert_eq!(orch.active_execution_count(), 0);
        assert_eq!(orch.history_len(), 0);
        assert_eq!(orch.stats().total_ticks, 0);
    }

    // ---------------------------------------------------------------
    // Display impls
    // ---------------------------------------------------------------

    #[test]
    fn test_phase_display() {
        assert_eq!(format!("{}", OrchestratorPhase::Idle), "Idle");
        assert_eq!(format!("{}", OrchestratorPhase::Active), "Active");
        assert_eq!(format!("{}", OrchestratorPhase::Stopped), "Stopped");
    }

    #[test]
    fn test_step_outcome_display() {
        assert_eq!(format!("{}", StepOutcome::Success), "Success");
        assert_eq!(format!("{}", StepOutcome::TimedOut), "TimedOut");
    }

    #[test]
    fn test_workflow_outcome_display() {
        assert_eq!(format!("{}", WorkflowOutcome::Completed), "Completed");
        assert_eq!(format!("{}", WorkflowOutcome::InProgress), "InProgress");
    }

    // ---------------------------------------------------------------
    // Full lifecycle integration test
    // ---------------------------------------------------------------

    #[test]
    fn test_full_lifecycle() {
        let mut orch = SystemOrchestrator::with_config(small_config());

        // Register workflows
        orch.register_workflow(ingest_workflow());
        orch.register_workflow(analyse_workflow());

        // Activate and schedule
        orch.activate();
        orch.schedule("ingest");
        orch.schedule("analyse");

        // Tick to start both workflows
        orch.tick();
        assert_eq!(orch.active_execution_count(), 2);

        // Complete ingest: 3 steps
        let ingest_id = orch
            .active_executions()
            .iter()
            .find(|e| e.workflow_name == "ingest")
            .unwrap()
            .run_id;
        orch.complete_step(ingest_id, StepOutcome::Success);
        orch.complete_step(ingest_id, StepOutcome::Success);
        orch.complete_step(ingest_id, StepOutcome::Success);

        // Complete analyse: 2 steps
        let analyse_id = orch
            .active_executions()
            .iter()
            .find(|e| e.workflow_name == "analyse")
            .unwrap()
            .run_id;
        orch.complete_step(analyse_id, StepOutcome::Success);
        orch.complete_step(analyse_id, StepOutcome::Success);

        // Tick to archive completed workflows
        orch.tick();

        assert_eq!(orch.active_execution_count(), 0);
        assert_eq!(orch.history_len(), 2);
        assert_eq!(orch.stats().total_executions_completed, 2);
        assert_eq!(orch.stats().total_steps_completed, 5);
        assert!((orch.stats().workflow_success_rate() - 1.0).abs() < 1e-10);

        // Pause, resume, stop
        orch.pause();
        assert_eq!(orch.phase(), OrchestratorPhase::Paused);

        orch.resume();
        assert_eq!(orch.phase(), OrchestratorPhase::Active);

        orch.stop();
        assert_eq!(orch.phase(), OrchestratorPhase::Stopped);
    }

    // ---------------------------------------------------------------
    // Draining with active workflows
    // ---------------------------------------------------------------

    #[test]
    fn test_drain_waits_for_active() {
        let mut orch = SystemOrchestrator::with_config(small_config());
        orch.register_workflow(ingest_workflow());
        orch.activate();
        orch.schedule("ingest");
        orch.tick(); // start workflow
        assert_eq!(orch.active_execution_count(), 1);

        orch.drain();
        assert_eq!(orch.phase(), OrchestratorPhase::Draining);
        assert_eq!(orch.pending_count(), 0); // pending cleared

        // Complete the active workflow
        let run_id = orch.active_executions()[0].run_id;
        orch.complete_step(run_id, StepOutcome::Success);
        orch.complete_step(run_id, StepOutcome::Success);
        orch.complete_step(run_id, StepOutcome::Success);

        // Tick to archive and transition to Stopped
        orch.tick();
        assert_eq!(orch.phase(), OrchestratorPhase::Stopped);
    }

    // ---------------------------------------------------------------
    // Stop cancels active workflows
    // ---------------------------------------------------------------

    #[test]
    fn test_stop_cancels_active() {
        let mut orch = SystemOrchestrator::with_config(small_config());
        orch.register_workflow(ingest_workflow());
        orch.activate();
        orch.schedule("ingest");
        orch.tick();
        assert_eq!(orch.active_execution_count(), 1);

        orch.stop();
        assert_eq!(orch.active_execution_count(), 0);
        assert_eq!(orch.history_len(), 1);
        assert_eq!(orch.stats().total_executions_cancelled, 1);
        assert_eq!(
            orch.last_execution("ingest").unwrap().outcome,
            WorkflowOutcome::Cancelled
        );
    }

    // ---------------------------------------------------------------
    // WorkflowStep builder
    // ---------------------------------------------------------------

    #[test]
    fn test_workflow_step_builder() {
        let step = WorkflowStep::new("fetch", "thalamus")
            .optional()
            .with_timeout(50)
            .with_description("Fetch live data");

        assert_eq!(step.name, "fetch");
        assert_eq!(step.region, "thalamus");
        assert!(step.optional);
        assert_eq!(step.timeout, Some(50));
        assert_eq!(step.description, "Fetch live data");
    }

    // ---------------------------------------------------------------
    // WorkflowDefinition builder
    // ---------------------------------------------------------------

    #[test]
    fn test_workflow_definition_builder() {
        let wf = WorkflowDefinition::new("test")
            .with_description("Test workflow")
            .with_priority(50)
            .with_step(WorkflowStep::new("s1", "r1"))
            .with_step(WorkflowStep::new("s2", "r2"));

        assert_eq!(wf.name, "test");
        assert_eq!(wf.description, "Test workflow");
        assert_eq!(wf.priority, 50);
        assert_eq!(wf.step_count(), 2);
        assert!(wf.enabled);
    }

    // ---------------------------------------------------------------
    // Edge: scheduling without activation doesn't start
    // ---------------------------------------------------------------

    #[test]
    fn test_schedule_without_activation() {
        let mut orch = SystemOrchestrator::with_config(small_config());
        orch.register_workflow(ingest_workflow());

        // Schedule while idle
        orch.schedule("ingest");
        assert_eq!(orch.pending_count(), 1);

        // Tick while idle — workflows don't start
        orch.tick();
        assert_eq!(orch.active_execution_count(), 0);
        assert_eq!(orch.pending_count(), 1); // still pending
    }

    // ---------------------------------------------------------------
    // Edge: empty-step workflow
    // ---------------------------------------------------------------

    #[test]
    fn test_empty_workflow_not_started() {
        let mut orch = SystemOrchestrator::with_config(small_config());
        let wf = WorkflowDefinition::new("empty"); // no steps
        orch.register_workflow(wf);
        orch.activate();
        orch.schedule("empty");
        orch.tick();

        // Empty workflow should not start
        assert_eq!(orch.active_execution_count(), 0);
    }
}
