//! Service lifecycle state machine for the Janus Supervisor.
//!
//! Each service managed by the [`JanusSupervisor`](super::JanusSupervisor)
//! progresses through a well-defined set of states:
//!
//! ```text
//!   ┌──────────┐
//!   │ Starting │──────────────────────────┐
//!   └────┬─────┘                          │
//!        │ run() entered                  │ init error
//!        ▼                                ▼
//!   ┌──────────┐                    ┌────────────┐
//!   │ Running  │───── error ──────▶│ BackingOff │
//!   └────┬─────┘                    └──────┬─────┘
//!        │                                 │
//!        │ cancel / Ok(())                 │ retry
//!        │                                 │
//!        │    ┌────────────────────────────┘
//!        ▼    ▼
//!   ┌──────────┐         ┌────────────┐
//!   │ Stopping │────────▶│ Terminated │
//!   └──────────┘         └────────────┘
//! ```
//!
//! The state machine enforces deterministic behaviour:
//!
//! - **Starting**: The service is initializing resources (connections, channels).
//! - **Running**: The service's `run()` loop is active.
//! - **BackingOff**: The service failed and is waiting for the exponential
//!   backoff timer before the supervisor retries.
//! - **Stopping**: A cancellation signal was received; the service is
//!   finalizing (flushing WAL, closing connections).
//! - **Terminated**: The service has exited cleanly (or the circuit breaker
//!   tripped and the supervisor gave up). Terminal state.
//!
//! The `BackingOff` state prevents the supervisor from tight-looping on a
//! persistent failure, which would burn CPU and flood logs.

use std::fmt;
use std::time::{Duration, Instant};

// ---------------------------------------------------------------------------
// ServicePhase — the raw enum
// ---------------------------------------------------------------------------

/// Lifecycle phase of a supervised service.
///
/// This is a plain enum without associated data; the richer context (timing,
/// error info, attempt counts) lives in [`ServiceLifecycle`].
#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash, serde::Serialize, serde::Deserialize)]
#[serde(rename_all = "snake_case")]
pub enum ServicePhase {
    /// The service is initializing (connecting to databases, setting up
    /// channels, etc.).
    Starting,

    /// The service's main `run()` loop is executing.
    Running,

    /// The service failed and is waiting for the backoff timer to expire
    /// before the supervisor attempts a restart.
    BackingOff,

    /// A shutdown signal was received; the service is performing cleanup.
    Stopping,

    /// Terminal state — the service has exited. No further transitions are
    /// valid.
    Terminated,
}

impl fmt::Display for ServicePhase {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        match self {
            Self::Starting => write!(f, "starting"),
            Self::Running => write!(f, "running"),
            Self::BackingOff => write!(f, "backing_off"),
            Self::Stopping => write!(f, "stopping"),
            Self::Terminated => write!(f, "terminated"),
        }
    }
}

impl ServicePhase {
    /// Returns `true` if the service is in a terminal state and will not
    /// transition again.
    pub fn is_terminal(&self) -> bool {
        matches!(self, Self::Terminated)
    }

    /// Returns `true` if the service is considered "alive" (starting,
    /// running, or backing off for a retry).
    pub fn is_alive(&self) -> bool {
        matches!(self, Self::Starting | Self::Running | Self::BackingOff)
    }
}

// ---------------------------------------------------------------------------
// TerminationReason
// ---------------------------------------------------------------------------

/// Why a service reached the `Terminated` phase.
#[derive(Debug, Clone, PartialEq, Eq)]
pub enum TerminationReason {
    /// The service's `run()` returned `Ok(())` — clean completion.
    Completed,

    /// The supervisor's cancellation token was triggered (graceful shutdown).
    Cancelled,

    /// The circuit breaker tripped after too many consecutive failures.
    CircuitBreakerOpen {
        /// Number of failures observed within the circuit-breaker window.
        failures: u32,
        /// The configured maximum before tripping.
        max_retries: u32,
    },

    /// The service encountered an unrecoverable error and its restart
    /// policy is [`RestartPolicy::Never`](super::RestartPolicy::Never).
    Unrecoverable(String),
}

impl fmt::Display for TerminationReason {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        match self {
            Self::Completed => write!(f, "completed"),
            Self::Cancelled => write!(f, "cancelled"),
            Self::CircuitBreakerOpen {
                failures,
                max_retries,
            } => write!(
                f,
                "circuit breaker open ({failures}/{max_retries} failures)"
            ),
            Self::Unrecoverable(msg) => write!(f, "unrecoverable: {msg}"),
        }
    }
}

// ---------------------------------------------------------------------------
// TransitionError
// ---------------------------------------------------------------------------

/// Error returned when an invalid state transition is attempted.
#[derive(Debug, Clone, thiserror::Error)]
#[error("invalid lifecycle transition: {from} → {to}")]
pub struct TransitionError {
    pub from: ServicePhase,
    pub to: ServicePhase,
}

// ---------------------------------------------------------------------------
// ServiceLifecycle — the state machine
// ---------------------------------------------------------------------------

/// Full lifecycle tracker for a single supervised service.
///
/// Wraps the [`ServicePhase`] enum with timing data, counters, and
/// transition validation logic. The supervisor holds one of these per
/// managed service.
#[derive(Debug, Clone)]
pub struct ServiceLifecycle {
    /// Current phase.
    phase: ServicePhase,

    /// Name of the service (for logging/metrics).
    service_name: String,

    /// When the service first entered `Starting`.
    created_at: Instant,

    /// When the current phase was entered.
    phase_entered_at: Instant,

    /// Total number of times the service has been (re)started.
    start_count: u32,

    /// Total number of failures observed over the service's lifetime.
    total_failures: u32,

    /// The last error message if the service failed.
    last_error: Option<String>,

    /// Why the service terminated (only set in `Terminated` phase).
    termination_reason: Option<TerminationReason>,

    /// Cumulative time spent in the `Running` phase.
    cumulative_running: Duration,

    /// Snapshot of `phase_entered_at` when we last entered `Running`,
    /// used to accumulate `cumulative_running` on exit.
    running_since: Option<Instant>,
}

impl ServiceLifecycle {
    /// Create a new lifecycle tracker in the `Starting` phase.
    pub fn new(service_name: impl Into<String>) -> Self {
        let now = Instant::now();
        Self {
            phase: ServicePhase::Starting,
            service_name: service_name.into(),
            created_at: now,
            phase_entered_at: now,
            start_count: 1,
            total_failures: 0,
            last_error: None,
            termination_reason: None,
            cumulative_running: Duration::ZERO,
            running_since: None,
        }
    }

    // ── Accessors ─────────────────────────────────────────────────────

    /// Current lifecycle phase.
    pub fn phase(&self) -> ServicePhase {
        self.phase
    }

    /// Service name.
    pub fn service_name(&self) -> &str {
        &self.service_name
    }

    /// How long the service has existed (since first `Starting`).
    pub fn age(&self) -> Duration {
        self.created_at.elapsed()
    }

    /// How long the service has been in its current phase.
    pub fn time_in_current_phase(&self) -> Duration {
        self.phase_entered_at.elapsed()
    }

    /// Total number of times the service has been started.
    pub fn start_count(&self) -> u32 {
        self.start_count
    }

    /// Total failures over the service's lifetime.
    pub fn total_failures(&self) -> u32 {
        self.total_failures
    }

    /// The last error message, if any.
    pub fn last_error(&self) -> Option<&str> {
        self.last_error.as_deref()
    }

    /// Why the service terminated (only `Some` when phase is `Terminated`).
    pub fn termination_reason(&self) -> Option<&TerminationReason> {
        self.termination_reason.as_ref()
    }

    /// Cumulative wall-clock time spent in the `Running` phase.
    ///
    /// If the service is currently running, includes time up to *now*.
    pub fn cumulative_running_time(&self) -> Duration {
        let extra = self
            .running_since
            .map(|since| since.elapsed())
            .unwrap_or(Duration::ZERO);
        self.cumulative_running + extra
    }

    // ── Transitions ───────────────────────────────────────────────────

    /// Transition from `Starting` → `Running`.
    ///
    /// Called when the service's `run()` method is entered.
    pub fn transition_to_running(&mut self) -> Result<(), TransitionError> {
        self.validate_transition(ServicePhase::Running)?;
        self.set_phase(ServicePhase::Running);
        self.running_since = Some(Instant::now());
        tracing::info!(
            service = %self.service_name,
            start_count = self.start_count,
            "service entered Running phase"
        );
        Ok(())
    }

    /// Transition from `Running` → `BackingOff`.
    ///
    /// Called when the service's `run()` returns an `Err`.
    pub fn transition_to_backing_off(
        &mut self,
        error: &str,
        backoff_duration: Duration,
    ) -> Result<(), TransitionError> {
        self.validate_transition(ServicePhase::BackingOff)?;
        self.accumulate_running_time();
        self.total_failures += 1;
        self.last_error = Some(error.to_string());
        self.set_phase(ServicePhase::BackingOff);
        tracing::warn!(
            service = %self.service_name,
            error = %error,
            attempt = self.total_failures,
            backoff_ms = backoff_duration.as_millis() as u64,
            "service failed, entering BackingOff phase"
        );
        Ok(())
    }

    /// Transition from `BackingOff` → `Starting` (retry).
    ///
    /// Called when the backoff timer has expired and the supervisor is
    /// about to restart the service.
    pub fn transition_to_restarting(&mut self) -> Result<(), TransitionError> {
        // BackingOff → Starting is a restart
        self.validate_transition(ServicePhase::Starting)?;
        self.start_count += 1;
        self.set_phase(ServicePhase::Starting);
        tracing::info!(
            service = %self.service_name,
            start_count = self.start_count,
            "service restarting (entering Starting phase)"
        );
        Ok(())
    }

    /// Transition to `Stopping` from any alive phase.
    ///
    /// Called when the cancellation token is triggered.
    pub fn transition_to_stopping(&mut self) -> Result<(), TransitionError> {
        self.validate_transition(ServicePhase::Stopping)?;
        self.accumulate_running_time();
        self.set_phase(ServicePhase::Stopping);
        tracing::info!(
            service = %self.service_name,
            "service entering Stopping phase"
        );
        Ok(())
    }

    /// Transition to `Terminated` from `Stopping`, `Running`, `Starting`,
    /// or `BackingOff`.
    ///
    /// This is the terminal state; no further transitions are allowed.
    pub fn transition_to_terminated(
        &mut self,
        reason: TerminationReason,
    ) -> Result<(), TransitionError> {
        self.validate_transition(ServicePhase::Terminated)?;
        self.accumulate_running_time();
        self.termination_reason = Some(reason.clone());
        self.set_phase(ServicePhase::Terminated);
        tracing::info!(
            service = %self.service_name,
            reason = %reason,
            total_starts = self.start_count,
            total_failures = self.total_failures,
            cumulative_running_secs = self.cumulative_running.as_secs_f64(),
            "service terminated"
        );
        Ok(())
    }

    // ── Internal helpers ──────────────────────────────────────────────

    /// Validate that transitioning from the current phase to `target` is
    /// legal.
    fn validate_transition(&self, target: ServicePhase) -> Result<(), TransitionError> {
        let valid = match (self.phase, target) {
            // Starting can go to Running or Terminated (init failure) or Stopping
            (ServicePhase::Starting, ServicePhase::Running) => true,
            (ServicePhase::Starting, ServicePhase::Terminated) => true,
            (ServicePhase::Starting, ServicePhase::Stopping) => true,
            // Also allow Starting → BackingOff for init errors that are retryable
            (ServicePhase::Starting, ServicePhase::BackingOff) => true,

            // Running can go to BackingOff (failure), Stopping, or Terminated
            (ServicePhase::Running, ServicePhase::BackingOff) => true,
            (ServicePhase::Running, ServicePhase::Stopping) => true,
            (ServicePhase::Running, ServicePhase::Terminated) => true,

            // BackingOff can go to Starting (retry), Stopping, or Terminated
            (ServicePhase::BackingOff, ServicePhase::Starting) => true,
            (ServicePhase::BackingOff, ServicePhase::Stopping) => true,
            (ServicePhase::BackingOff, ServicePhase::Terminated) => true,

            // Stopping can only go to Terminated
            (ServicePhase::Stopping, ServicePhase::Terminated) => true,

            // Terminated is terminal — nothing is valid
            (ServicePhase::Terminated, _) => false,

            // Everything else is invalid
            _ => false,
        };

        if valid {
            Ok(())
        } else {
            Err(TransitionError {
                from: self.phase,
                to: target,
            })
        }
    }

    /// Set the phase and update `phase_entered_at`.
    fn set_phase(&mut self, phase: ServicePhase) {
        self.phase = phase;
        self.phase_entered_at = Instant::now();
    }

    /// If we were in Running, accumulate the elapsed running time and
    /// clear the `running_since` marker.
    fn accumulate_running_time(&mut self) {
        if let Some(since) = self.running_since.take() {
            self.cumulative_running += since.elapsed();
        }
    }
}

impl fmt::Display for ServiceLifecycle {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        write!(
            f,
            "{}[{}] starts={} failures={} running={:.1}s",
            self.service_name,
            self.phase,
            self.start_count,
            self.total_failures,
            self.cumulative_running_time().as_secs_f64(),
        )
    }
}

// ---------------------------------------------------------------------------
// Serializable snapshot for API / metrics
// ---------------------------------------------------------------------------

/// A point-in-time snapshot of a service's lifecycle, suitable for
/// serialization into JSON (e.g., for the `/api/health` endpoint).
#[derive(Debug, Clone, serde::Serialize, serde::Deserialize)]
pub struct ServiceLifecycleSnapshot {
    pub service_name: String,
    pub phase: ServicePhase,
    pub start_count: u32,
    pub total_failures: u32,
    pub last_error: Option<String>,
    pub cumulative_running_secs: f64,
    pub age_secs: f64,
    pub time_in_phase_secs: f64,
    pub termination_reason: Option<String>,
}

impl From<&ServiceLifecycle> for ServiceLifecycleSnapshot {
    fn from(lc: &ServiceLifecycle) -> Self {
        Self {
            service_name: lc.service_name.clone(),
            phase: lc.phase,
            start_count: lc.start_count,
            total_failures: lc.total_failures,
            last_error: lc.last_error.clone(),
            cumulative_running_secs: lc.cumulative_running_time().as_secs_f64(),
            age_secs: lc.age().as_secs_f64(),
            time_in_phase_secs: lc.time_in_current_phase().as_secs_f64(),
            termination_reason: lc.termination_reason.as_ref().map(|r| r.to_string()),
        }
    }
}

// ===========================================================================
// Tests
// ===========================================================================

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_new_lifecycle_starts_in_starting() {
        let lc = ServiceLifecycle::new("test-svc");
        assert_eq!(lc.phase(), ServicePhase::Starting);
        assert_eq!(lc.start_count(), 1);
        assert_eq!(lc.total_failures(), 0);
        assert!(lc.last_error().is_none());
        assert!(lc.termination_reason().is_none());
    }

    #[test]
    fn test_service_name() {
        let lc = ServiceLifecycle::new("data-service");
        assert_eq!(lc.service_name(), "data-service");
    }

    #[test]
    fn test_happy_path_starting_to_running_to_stopping_to_terminated() {
        let mut lc = ServiceLifecycle::new("happy");

        lc.transition_to_running().unwrap();
        assert_eq!(lc.phase(), ServicePhase::Running);

        lc.transition_to_stopping().unwrap();
        assert_eq!(lc.phase(), ServicePhase::Stopping);

        lc.transition_to_terminated(TerminationReason::Cancelled)
            .unwrap();
        assert_eq!(lc.phase(), ServicePhase::Terminated);
        assert_eq!(lc.termination_reason(), Some(&TerminationReason::Cancelled));
    }

    #[test]
    fn test_failure_and_restart_cycle() {
        let mut lc = ServiceLifecycle::new("flaky");

        // Start → Run → Fail (BackingOff) → Restart (Starting) → Run
        lc.transition_to_running().unwrap();
        assert_eq!(lc.start_count(), 1);

        lc.transition_to_backing_off("connection refused", Duration::from_millis(200))
            .unwrap();
        assert_eq!(lc.phase(), ServicePhase::BackingOff);
        assert_eq!(lc.total_failures(), 1);
        assert_eq!(lc.last_error(), Some("connection refused"));

        lc.transition_to_restarting().unwrap();
        assert_eq!(lc.phase(), ServicePhase::Starting);
        assert_eq!(lc.start_count(), 2);

        lc.transition_to_running().unwrap();
        assert_eq!(lc.phase(), ServicePhase::Running);
    }

    #[test]
    fn test_circuit_breaker_termination() {
        let mut lc = ServiceLifecycle::new("breaker");

        lc.transition_to_running().unwrap();
        lc.transition_to_backing_off("error 1", Duration::from_millis(100))
            .unwrap();

        lc.transition_to_terminated(TerminationReason::CircuitBreakerOpen {
            failures: 10,
            max_retries: 10,
        })
        .unwrap();

        assert_eq!(lc.phase(), ServicePhase::Terminated);
        assert!(matches!(
            lc.termination_reason(),
            Some(TerminationReason::CircuitBreakerOpen { .. })
        ));
    }

    #[test]
    fn test_completed_termination_from_running() {
        let mut lc = ServiceLifecycle::new("one-shot");

        lc.transition_to_running().unwrap();
        lc.transition_to_terminated(TerminationReason::Completed)
            .unwrap();

        assert_eq!(lc.phase(), ServicePhase::Terminated);
        assert_eq!(lc.termination_reason(), Some(&TerminationReason::Completed));
    }

    #[test]
    fn test_invalid_transition_terminated_to_anything() {
        let mut lc = ServiceLifecycle::new("dead");

        lc.transition_to_running().unwrap();
        lc.transition_to_terminated(TerminationReason::Completed)
            .unwrap();

        // Any further transition should fail
        assert!(lc.transition_to_running().is_err());
        assert!(lc.transition_to_stopping().is_err());
        assert!(
            lc.transition_to_terminated(TerminationReason::Cancelled)
                .is_err()
        );
        assert!(lc.transition_to_restarting().is_err());
    }

    #[test]
    fn test_invalid_transition_running_to_starting() {
        let mut lc = ServiceLifecycle::new("bad");

        lc.transition_to_running().unwrap();

        // Running → Starting is not valid (must go through BackingOff first)
        let err = lc.transition_to_restarting().unwrap_err();
        assert_eq!(err.from, ServicePhase::Running);
        assert_eq!(err.to, ServicePhase::Starting);
    }

    #[test]
    fn test_stopping_from_backing_off() {
        let mut lc = ServiceLifecycle::new("interrupted");

        lc.transition_to_running().unwrap();
        lc.transition_to_backing_off("timeout", Duration::from_secs(5))
            .unwrap();

        // Shutdown arrives while backing off
        lc.transition_to_stopping().unwrap();
        assert_eq!(lc.phase(), ServicePhase::Stopping);

        lc.transition_to_terminated(TerminationReason::Cancelled)
            .unwrap();
        assert_eq!(lc.phase(), ServicePhase::Terminated);
    }

    #[test]
    fn test_starting_directly_to_terminated() {
        let mut lc = ServiceLifecycle::new("init-fail");

        // If init fails catastrophically, we can go straight to Terminated
        lc.transition_to_terminated(TerminationReason::Unrecoverable(
            "missing config".to_string(),
        ))
        .unwrap();
        assert_eq!(lc.phase(), ServicePhase::Terminated);
    }

    #[test]
    fn test_starting_to_backing_off() {
        let mut lc = ServiceLifecycle::new("init-retry");

        // Init fails but is retryable
        lc.transition_to_backing_off("db connect timeout", Duration::from_millis(500))
            .unwrap();
        assert_eq!(lc.phase(), ServicePhase::BackingOff);
        assert_eq!(lc.total_failures(), 1);
    }

    #[test]
    fn test_phase_display() {
        assert_eq!(ServicePhase::Starting.to_string(), "starting");
        assert_eq!(ServicePhase::Running.to_string(), "running");
        assert_eq!(ServicePhase::BackingOff.to_string(), "backing_off");
        assert_eq!(ServicePhase::Stopping.to_string(), "stopping");
        assert_eq!(ServicePhase::Terminated.to_string(), "terminated");
    }

    #[test]
    fn test_phase_is_terminal() {
        assert!(!ServicePhase::Starting.is_terminal());
        assert!(!ServicePhase::Running.is_terminal());
        assert!(!ServicePhase::BackingOff.is_terminal());
        assert!(!ServicePhase::Stopping.is_terminal());
        assert!(ServicePhase::Terminated.is_terminal());
    }

    #[test]
    fn test_phase_is_alive() {
        assert!(ServicePhase::Starting.is_alive());
        assert!(ServicePhase::Running.is_alive());
        assert!(ServicePhase::BackingOff.is_alive());
        assert!(!ServicePhase::Stopping.is_alive());
        assert!(!ServicePhase::Terminated.is_alive());
    }

    #[test]
    fn test_lifecycle_display() {
        let lc = ServiceLifecycle::new("display-test");
        let display = format!("{lc}");
        assert!(display.contains("display-test"));
        assert!(display.contains("starting"));
        assert!(display.contains("starts=1"));
        assert!(display.contains("failures=0"));
    }

    #[test]
    fn test_snapshot_from_lifecycle() {
        let mut lc = ServiceLifecycle::new("snapshot-svc");
        lc.transition_to_running().unwrap();
        lc.transition_to_backing_off("oops", Duration::from_millis(100))
            .unwrap();

        let snap = ServiceLifecycleSnapshot::from(&lc);
        assert_eq!(snap.service_name, "snapshot-svc");
        assert_eq!(snap.phase, ServicePhase::BackingOff);
        assert_eq!(snap.start_count, 1);
        assert_eq!(snap.total_failures, 1);
        assert_eq!(snap.last_error.as_deref(), Some("oops"));
        assert!(snap.termination_reason.is_none());
        assert!(snap.age_secs >= 0.0);
    }

    #[test]
    fn test_termination_reason_display() {
        assert_eq!(TerminationReason::Completed.to_string(), "completed");
        assert_eq!(TerminationReason::Cancelled.to_string(), "cancelled");
        assert_eq!(
            TerminationReason::CircuitBreakerOpen {
                failures: 5,
                max_retries: 5
            }
            .to_string(),
            "circuit breaker open (5/5 failures)"
        );
        assert_eq!(
            TerminationReason::Unrecoverable("bad config".into()).to_string(),
            "unrecoverable: bad config"
        );
    }

    #[test]
    fn test_transition_error_display() {
        let err = TransitionError {
            from: ServicePhase::Terminated,
            to: ServicePhase::Running,
        };
        assert_eq!(
            err.to_string(),
            "invalid lifecycle transition: terminated → running"
        );
    }

    #[test]
    fn test_multiple_failure_cycles_accumulate() {
        let mut lc = ServiceLifecycle::new("multi-fail");

        for i in 1..=5 {
            if lc.phase() == ServicePhase::Starting && i > 1 {
                // After restarting
            }
            lc.transition_to_running().unwrap();
            lc.transition_to_backing_off(
                &format!("error {i}"),
                Duration::from_millis(100 * i as u64),
            )
            .unwrap();
            if i < 5 {
                lc.transition_to_restarting().unwrap();
            }
        }

        assert_eq!(lc.total_failures(), 5);
        assert_eq!(lc.start_count(), 5);
        assert_eq!(lc.last_error(), Some("error 5"));
    }

    #[test]
    fn test_stopping_from_starting() {
        let mut lc = ServiceLifecycle::new("early-stop");

        // Shutdown arrives before the service even starts running
        lc.transition_to_stopping().unwrap();
        assert_eq!(lc.phase(), ServicePhase::Stopping);

        lc.transition_to_terminated(TerminationReason::Cancelled)
            .unwrap();
    }
}
