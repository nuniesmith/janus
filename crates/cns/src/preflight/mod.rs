//! # Pre-Flight Check Framework
//!
//! Implements a structured boot-time verification system for the JANUS trading platform.
//! Inspired by aircraft pre-flight checklists, this module ensures all critical subsystems
//! are operational before the system begins trading.
//!
//! ## Architecture
//!
//! - **BootPhase**: Ordered phases that checks are grouped into (Infrastructure → Executive)
//! - **Criticality**: How important a check is (Critical = abort on fail, Optional = warn only)
//! - **PreFlightCheck**: Trait that each concrete check implements
//! - **PreFlightRunner**: Orchestrates checks in phase order, collects results
//! - **BootReport**: Summary of all check results with pass/fail/skip counts
//!
//! ## Usage
//!
//! ```rust,no_run
//! use janus_cns::preflight::{PreFlightRunner, BootPhase, Criticality};
//! use janus_cns::preflight::infra::RedisCheck;
//!
//! # async fn example() -> anyhow::Result<()> {
//! let mut runner = PreFlightRunner::new();
//! runner.add_check(Box::new(RedisCheck::new("redis://localhost:6379")));
//! let report = runner.run().await;
//! if !report.is_boot_safe() {
//!     panic!("Pre-flight failed: {}", report.summary());
//! }
//! # Ok(())
//! # }
//! ```

pub mod executive;
pub mod infra;
pub mod regulatory;
pub mod sensory;
pub mod strategy;

use async_trait::async_trait;
use chrono::{DateTime, Utc};
use serde::{Deserialize, Serialize};
use std::fmt;
use std::time::{Duration, Instant};
use tracing::{debug, error, info, warn};

// ============================================================================
// Boot Phase
// ============================================================================

/// Ordered boot phases. Checks run in phase order — if a critical check fails
/// in an earlier phase, later phases are skipped.
#[derive(Debug, Clone, Copy, PartialEq, Eq, PartialOrd, Ord, Hash, Serialize, Deserialize)]
pub enum BootPhase {
    /// Phase 0: Core infrastructure (Redis, Postgres, QuestDB, Prometheus)
    Infrastructure = 0,

    /// Phase 1: Data feeds and sensory input (WebSocket, data latency, ViViT model)
    Sensory = 1,

    /// Phase 2: Risk & regulatory guards (kill switch, circuit breakers, hypothalamus)
    Regulatory = 2,

    /// Phase 3: Strategy subsystems (regime detector, strategy init, correlation, affinity)
    Strategy = 3,

    /// Phase 4: Execution path (gRPC, exchange REST, order path integrity)
    Executive = 4,
}

impl fmt::Display for BootPhase {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        match self {
            BootPhase::Infrastructure => write!(f, "Infrastructure"),
            BootPhase::Sensory => write!(f, "Sensory"),
            BootPhase::Regulatory => write!(f, "Regulatory"),
            BootPhase::Strategy => write!(f, "Strategy"),
            BootPhase::Executive => write!(f, "Executive"),
        }
    }
}

impl BootPhase {
    /// All phases in execution order.
    pub fn all() -> &'static [BootPhase] {
        &[
            BootPhase::Infrastructure,
            BootPhase::Sensory,
            BootPhase::Regulatory,
            BootPhase::Strategy,
            BootPhase::Executive,
        ]
    }

    /// Emoji for display formatting.
    pub fn emoji(&self) -> &'static str {
        match self {
            BootPhase::Infrastructure => "🏗️",
            BootPhase::Sensory => "👁️",
            BootPhase::Regulatory => "⚖️",
            BootPhase::Strategy => "🧠",
            BootPhase::Executive => "⚡",
        }
    }
}

// ============================================================================
// Criticality
// ============================================================================

/// How important a check is to system boot.
#[derive(Debug, Clone, Copy, PartialEq, Eq, PartialOrd, Ord, Hash, Serialize, Deserialize)]
pub enum Criticality {
    /// System MUST NOT boot if this check fails. Abort immediately.
    Critical,

    /// System SHOULD NOT boot, but operator can override.
    Required,

    /// Failure is logged as a warning; boot continues.
    Optional,
}

impl fmt::Display for Criticality {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        match self {
            Criticality::Critical => write!(f, "CRITICAL"),
            Criticality::Required => write!(f, "REQUIRED"),
            Criticality::Optional => write!(f, "OPTIONAL"),
        }
    }
}

impl Criticality {
    /// Emoji for display formatting.
    pub fn emoji(&self) -> &'static str {
        match self {
            Criticality::Critical => "🔴",
            Criticality::Required => "🟡",
            Criticality::Optional => "🟢",
        }
    }
}

// ============================================================================
// Check Outcome
// ============================================================================

/// Outcome of an individual pre-flight check.
#[derive(Debug, Clone, PartialEq, Eq, Serialize, Deserialize)]
pub enum CheckOutcome {
    /// Check passed.
    Pass,
    /// Check failed with a reason.
    Fail(String),
    /// Check was skipped (e.g. because a prior critical check in the same phase failed).
    Skipped(String),
    /// Check timed out.
    Timeout,
}

impl CheckOutcome {
    pub fn is_pass(&self) -> bool {
        matches!(self, CheckOutcome::Pass)
    }

    pub fn is_fail(&self) -> bool {
        matches!(self, CheckOutcome::Fail(_) | CheckOutcome::Timeout)
    }

    pub fn is_skipped(&self) -> bool {
        matches!(self, CheckOutcome::Skipped(_))
    }
}

impl fmt::Display for CheckOutcome {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        match self {
            CheckOutcome::Pass => write!(f, "✅ PASS"),
            CheckOutcome::Fail(reason) => write!(f, "❌ FAIL: {}", reason),
            CheckOutcome::Skipped(reason) => write!(f, "⏭️ SKIP: {}", reason),
            CheckOutcome::Timeout => write!(f, "⏰ TIMEOUT"),
        }
    }
}

// ============================================================================
// Check Result
// ============================================================================

/// Full result of a single pre-flight check execution.
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct CheckResult {
    /// Name of the check.
    pub name: String,

    /// Which boot phase this check belongs to.
    pub phase: BootPhase,

    /// How critical this check is.
    pub criticality: Criticality,

    /// The outcome.
    pub outcome: CheckOutcome,

    /// How long the check took.
    pub duration_ms: u64,

    /// Timestamp when the check completed.
    pub completed_at: DateTime<Utc>,

    /// Optional detail message (even on success).
    pub detail: Option<String>,
}

impl CheckResult {
    /// Create a passing result.
    pub fn pass(
        name: impl Into<String>,
        phase: BootPhase,
        criticality: Criticality,
        duration: Duration,
    ) -> Self {
        Self {
            name: name.into(),
            phase,
            criticality,
            outcome: CheckOutcome::Pass,
            duration_ms: duration.as_millis() as u64,
            completed_at: Utc::now(),
            detail: None,
        }
    }

    /// Create a failing result.
    pub fn fail(
        name: impl Into<String>,
        phase: BootPhase,
        criticality: Criticality,
        duration: Duration,
        reason: impl Into<String>,
    ) -> Self {
        let reason = reason.into();
        Self {
            name: name.into(),
            phase,
            criticality,
            outcome: CheckOutcome::Fail(reason),
            duration_ms: duration.as_millis() as u64,
            completed_at: Utc::now(),
            detail: None,
        }
    }

    /// Create a skipped result.
    pub fn skipped(
        name: impl Into<String>,
        phase: BootPhase,
        criticality: Criticality,
        reason: impl Into<String>,
    ) -> Self {
        Self {
            name: name.into(),
            phase,
            criticality,
            outcome: CheckOutcome::Skipped(reason.into()),
            duration_ms: 0,
            completed_at: Utc::now(),
            detail: None,
        }
    }

    /// Create a timeout result.
    pub fn timeout(
        name: impl Into<String>,
        phase: BootPhase,
        criticality: Criticality,
        duration: Duration,
    ) -> Self {
        Self {
            name: name.into(),
            phase,
            criticality,
            outcome: CheckOutcome::Timeout,
            duration_ms: duration.as_millis() as u64,
            completed_at: Utc::now(),
            detail: None,
        }
    }

    /// Attach a detail message.
    pub fn with_detail(mut self, detail: impl Into<String>) -> Self {
        self.detail = Some(detail.into());
        self
    }

    /// Whether this failure should abort the boot.
    pub fn is_abort_worthy(&self) -> bool {
        self.criticality == Criticality::Critical && self.outcome.is_fail()
    }
}

impl fmt::Display for CheckResult {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        write!(
            f,
            "[{} {}] {} {} ({}ms)",
            self.phase.emoji(),
            self.criticality.emoji(),
            self.name,
            self.outcome,
            self.duration_ms,
        )?;
        if let Some(ref detail) = self.detail {
            write!(f, " — {}", detail)?;
        }
        Ok(())
    }
}

// ============================================================================
// PreFlightCheck Trait
// ============================================================================

/// Trait that each concrete pre-flight check implements.
#[async_trait]
pub trait PreFlightCheck: Send + Sync {
    /// Human-readable name for this check.
    fn name(&self) -> &str;

    /// Which boot phase this check belongs to.
    fn phase(&self) -> BootPhase;

    /// How critical is this check.
    fn criticality(&self) -> Criticality;

    /// Maximum time this check is allowed to run before timing out.
    fn timeout(&self) -> Duration {
        Duration::from_secs(10)
    }

    /// Execute the check. Returns `Ok(())` on success, `Err(reason)` on failure.
    async fn execute(&self) -> std::result::Result<(), String>;

    /// Optional detail to include in the result even on success.
    fn detail_on_success(&self) -> Option<String> {
        None
    }

    /// Run the check with timeout handling. Do not override.
    async fn run(&self) -> CheckResult {
        let start = Instant::now();
        let timeout_dur = self.timeout();

        let result = tokio::time::timeout(timeout_dur, self.execute()).await;
        let elapsed = start.elapsed();

        match result {
            Ok(Ok(())) => {
                let mut r =
                    CheckResult::pass(self.name(), self.phase(), self.criticality(), elapsed);
                if let Some(detail) = self.detail_on_success() {
                    r = r.with_detail(detail);
                }
                r
            }
            Ok(Err(reason)) => CheckResult::fail(
                self.name(),
                self.phase(),
                self.criticality(),
                elapsed,
                reason,
            ),
            Err(_elapsed) => {
                CheckResult::timeout(self.name(), self.phase(), self.criticality(), elapsed)
            }
        }
    }
}

// ============================================================================
// Boot Report
// ============================================================================

/// Aggregated report of all pre-flight check results.
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct BootReport {
    /// All individual check results, in execution order.
    pub results: Vec<CheckResult>,

    /// Total time for the entire pre-flight sequence.
    pub total_duration_ms: u64,

    /// Timestamp when the report was generated.
    pub generated_at: DateTime<Utc>,

    /// Whether the boot was aborted due to a critical failure.
    pub aborted: bool,

    /// The phase that caused the abort (if any).
    pub abort_phase: Option<BootPhase>,

    /// The check that caused the abort (if any).
    pub abort_check: Option<String>,
}

impl BootReport {
    fn new() -> Self {
        Self {
            results: Vec::new(),
            total_duration_ms: 0,
            generated_at: Utc::now(),
            aborted: false,
            abort_phase: None,
            abort_check: None,
        }
    }

    /// Is it safe to boot? True only if no critical or required checks failed.
    pub fn is_boot_safe(&self) -> bool {
        !self.aborted
            && !self.results.iter().any(|r| {
                r.outcome.is_fail()
                    && matches!(r.criticality, Criticality::Critical | Criticality::Required)
            })
    }

    /// Count of passed checks.
    pub fn pass_count(&self) -> usize {
        self.results.iter().filter(|r| r.outcome.is_pass()).count()
    }

    /// Count of failed checks (including timeouts).
    pub fn fail_count(&self) -> usize {
        self.results.iter().filter(|r| r.outcome.is_fail()).count()
    }

    /// Count of skipped checks.
    pub fn skip_count(&self) -> usize {
        self.results
            .iter()
            .filter(|r| r.outcome.is_skipped())
            .count()
    }

    /// Total number of checks.
    pub fn total_count(&self) -> usize {
        self.results.len()
    }

    /// Get results for a specific phase.
    pub fn phase_results(&self, phase: BootPhase) -> Vec<&CheckResult> {
        self.results.iter().filter(|r| r.phase == phase).collect()
    }

    /// Get all failures.
    pub fn failures(&self) -> Vec<&CheckResult> {
        self.results
            .iter()
            .filter(|r| r.outcome.is_fail())
            .collect()
    }

    /// Get critical failures only.
    pub fn critical_failures(&self) -> Vec<&CheckResult> {
        self.results
            .iter()
            .filter(|r| r.outcome.is_fail() && r.criticality == Criticality::Critical)
            .collect()
    }

    /// One-line summary string.
    pub fn summary(&self) -> String {
        let status = if self.is_boot_safe() {
            "✅ BOOT SAFE"
        } else {
            "❌ BOOT BLOCKED"
        };

        format!(
            "{} — {}/{} passed, {} failed, {} skipped ({}ms)",
            status,
            self.pass_count(),
            self.total_count(),
            self.fail_count(),
            self.skip_count(),
            self.total_duration_ms,
        )
    }

    /// Multi-line formatted report for console/Discord output.
    pub fn full_report(&self) -> String {
        let mut lines = Vec::new();

        lines.push("╔══════════════════════════════════════════════════════════╗".to_string());
        lines.push("║            JANUS PRE-FLIGHT BOOT REPORT                ║".to_string());
        lines.push("╠══════════════════════════════════════════════════════════╣".to_string());

        for phase in BootPhase::all() {
            let phase_results = self.phase_results(*phase);
            if phase_results.is_empty() {
                continue;
            }

            lines.push(format!("║ {} {:<52} ║", phase.emoji(), phase));
            lines.push("║──────────────────────────────────────────────────────────║".to_string());

            for result in &phase_results {
                let status_icon = match &result.outcome {
                    CheckOutcome::Pass => "✅",
                    CheckOutcome::Fail(_) => "❌",
                    CheckOutcome::Skipped(_) => "⏭️",
                    CheckOutcome::Timeout => "⏰",
                };

                lines.push(format!(
                    "║  {} {:<40} {:>6}ms ║",
                    status_icon, result.name, result.duration_ms,
                ));

                // Show failure reasons
                if let CheckOutcome::Fail(ref reason) = result.outcome {
                    // Truncate long reasons
                    let truncated = if reason.len() > 50 {
                        format!("{}…", &reason[..49])
                    } else {
                        reason.clone()
                    };
                    lines.push(format!("║     └─ {:<49} ║", truncated));
                }
            }

            lines.push("║                                                          ║".to_string());
        }

        lines.push("╠══════════════════════════════════════════════════════════╣".to_string());
        lines.push(format!("║  {:<56} ║", self.summary()));
        lines.push(format!(
            "║  Generated: {:<44} ║",
            self.generated_at.format("%Y-%m-%d %H:%M:%S UTC")
        ));
        lines.push("╚══════════════════════════════════════════════════════════╝".to_string());

        if self.aborted
            && let Some(ref check_name) = self.abort_check
        {
            lines.push(format!(
                "\n🚨 BOOT ABORTED: Critical failure in \"{}\" during {:?} phase",
                check_name,
                self.abort_phase.unwrap_or(BootPhase::Infrastructure),
            ));
        }

        lines.join("\n")
    }

    /// Format as a Discord-friendly message.
    pub fn discord_message(&self) -> String {
        let mut msg = String::new();

        if self.is_boot_safe() {
            msg.push_str("## ✅ JANUS Pre-Flight: BOOT SAFE\n\n");
        } else {
            msg.push_str("## ❌ JANUS Pre-Flight: BOOT BLOCKED\n\n");
        }

        msg.push_str(&format!(
            "**Results:** {}/{} passed, {} failed, {} skipped\n",
            self.pass_count(),
            self.total_count(),
            self.fail_count(),
            self.skip_count(),
        ));
        msg.push_str(&format!("**Duration:** {}ms\n\n", self.total_duration_ms));

        let failures = self.failures();
        if !failures.is_empty() {
            msg.push_str("### Failures\n");
            for f in &failures {
                let reason = match &f.outcome {
                    CheckOutcome::Fail(r) => r.as_str(),
                    CheckOutcome::Timeout => "Timed out",
                    _ => "Unknown",
                };
                msg.push_str(&format!(
                    "- {} **{}** [{}]: {}\n",
                    f.criticality.emoji(),
                    f.name,
                    f.phase,
                    reason,
                ));
            }
        }

        msg
    }
}

impl fmt::Display for BootReport {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        write!(f, "{}", self.summary())
    }
}

// ============================================================================
// Pre-Flight Runner
// ============================================================================

/// Configuration for the pre-flight runner.
#[derive(Debug, Clone)]
pub struct PreFlightConfig {
    /// Whether to abort on the first critical failure (true) or run all checks anyway (false).
    pub abort_on_critical: bool,

    /// Whether to run checks within a phase concurrently.
    pub parallel_within_phase: bool,

    /// Global timeout for the entire pre-flight sequence.
    pub global_timeout: Duration,
}

impl Default for PreFlightConfig {
    fn default() -> Self {
        Self {
            abort_on_critical: true,
            parallel_within_phase: false,
            global_timeout: Duration::from_secs(120),
        }
    }
}

/// Orchestrates pre-flight checks in phase order and produces a `BootReport`.
pub struct PreFlightRunner {
    checks: Vec<Box<dyn PreFlightCheck>>,
    config: PreFlightConfig,
}

impl PreFlightRunner {
    /// Create a new runner with default configuration.
    pub fn new() -> Self {
        Self {
            checks: Vec::new(),
            config: PreFlightConfig::default(),
        }
    }

    /// Create a new runner with custom configuration.
    pub fn with_config(config: PreFlightConfig) -> Self {
        Self {
            checks: Vec::new(),
            config,
        }
    }

    /// Add a single check.
    pub fn add_check(&mut self, check: Box<dyn PreFlightCheck>) {
        self.checks.push(check);
    }

    /// Add multiple checks at once.
    pub fn add_checks(&mut self, checks: Vec<Box<dyn PreFlightCheck>>) {
        self.checks.extend(checks);
    }

    /// Number of registered checks.
    pub fn check_count(&self) -> usize {
        self.checks.len()
    }

    /// Run all registered checks in phase order.
    ///
    /// If `abort_on_critical` is true, encountering a critical failure will skip
    /// all subsequent phases (but finish the current phase's remaining checks as skipped).
    pub async fn run(&self) -> BootReport {
        let global_start = Instant::now();
        let mut report = BootReport::new();

        info!(
            "🚀 JANUS Pre-Flight starting with {} checks across {} phases",
            self.checks.len(),
            BootPhase::all().len(),
        );

        let mut abort_triggered = false;

        for phase in BootPhase::all() {
            let phase_checks: Vec<&Box<dyn PreFlightCheck>> =
                self.checks.iter().filter(|c| c.phase() == *phase).collect();

            if phase_checks.is_empty() {
                debug!("Phase {} has no checks, skipping", phase);
                continue;
            }

            info!(
                "{} Phase {}: Running {} checks",
                phase.emoji(),
                phase,
                phase_checks.len(),
            );

            if abort_triggered {
                // Skip all checks in this phase
                for check in &phase_checks {
                    let result = CheckResult::skipped(
                        check.name(),
                        check.phase(),
                        check.criticality(),
                        "Skipped due to prior critical failure",
                    );
                    warn!("  {}", result);
                    report.results.push(result);
                }
                continue;
            }

            // Run checks in this phase
            if self.config.parallel_within_phase {
                // Parallel execution within phase
                let futures: Vec<_> = phase_checks.iter().map(|check| check.run()).collect();
                let results = futures::future::join_all(futures).await;

                for result in results {
                    if result.outcome.is_pass() {
                        info!("  {}", result);
                    } else if result.outcome.is_fail() {
                        error!("  {}", result);
                    } else {
                        warn!("  {}", result);
                    }

                    if result.is_abort_worthy() && self.config.abort_on_critical {
                        abort_triggered = true;
                        report.aborted = true;
                        report.abort_phase = Some(*phase);
                        report.abort_check = Some(result.name.clone());
                        error!(
                            "🚨 Critical failure in \"{}\": aborting remaining phases",
                            result.name
                        );
                    }

                    report.results.push(result);
                }
            } else {
                // Sequential execution within phase
                for check in &phase_checks {
                    if abort_triggered && self.config.abort_on_critical {
                        // We already triggered abort mid-phase — skip remaining in this phase too
                        let result = CheckResult::skipped(
                            check.name(),
                            check.phase(),
                            check.criticality(),
                            "Skipped due to prior critical failure in this phase",
                        );
                        warn!("  {}", result);
                        report.results.push(result);
                        continue;
                    }

                    let result = check.run().await;

                    if result.outcome.is_pass() {
                        info!("  {}", result);
                    } else if result.outcome.is_fail() {
                        error!("  {}", result);
                    } else {
                        warn!("  {}", result);
                    }

                    if result.is_abort_worthy() && self.config.abort_on_critical {
                        abort_triggered = true;
                        report.aborted = true;
                        report.abort_phase = Some(*phase);
                        report.abort_check = Some(result.name.clone());
                        error!(
                            "🚨 Critical failure in \"{}\": aborting remaining checks",
                            result.name
                        );
                    }

                    report.results.push(result);
                }
            }

            // Check global timeout
            if global_start.elapsed() >= self.config.global_timeout {
                error!("⏰ Global pre-flight timeout exceeded, aborting");
                report.aborted = true;
                break;
            }
        }

        report.total_duration_ms = global_start.elapsed().as_millis() as u64;
        report.generated_at = Utc::now();

        if report.is_boot_safe() {
            info!("✅ Pre-flight complete: {}", report.summary());
        } else {
            error!("❌ Pre-flight FAILED: {}", report.summary());
        }

        report
    }
}

impl Default for PreFlightRunner {
    fn default() -> Self {
        Self::new()
    }
}

// ============================================================================
// Tests
// ============================================================================

#[cfg(test)]
mod tests {
    use super::*;

    // --- Mock checks for testing ---

    struct PassingCheck {
        name: String,
        phase: BootPhase,
        criticality: Criticality,
    }

    impl PassingCheck {
        fn new(name: &str, phase: BootPhase, criticality: Criticality) -> Self {
            Self {
                name: name.to_string(),
                phase,
                criticality,
            }
        }
    }

    #[async_trait]
    impl PreFlightCheck for PassingCheck {
        fn name(&self) -> &str {
            &self.name
        }
        fn phase(&self) -> BootPhase {
            self.phase
        }
        fn criticality(&self) -> Criticality {
            self.criticality
        }
        async fn execute(&self) -> Result<(), String> {
            Ok(())
        }
    }

    struct FailingCheck {
        name: String,
        phase: BootPhase,
        criticality: Criticality,
        reason: String,
    }

    impl FailingCheck {
        fn new(name: &str, phase: BootPhase, criticality: Criticality, reason: &str) -> Self {
            Self {
                name: name.to_string(),
                phase,
                criticality,
                reason: reason.to_string(),
            }
        }
    }

    #[async_trait]
    impl PreFlightCheck for FailingCheck {
        fn name(&self) -> &str {
            &self.name
        }
        fn phase(&self) -> BootPhase {
            self.phase
        }
        fn criticality(&self) -> Criticality {
            self.criticality
        }
        async fn execute(&self) -> Result<(), String> {
            Err(self.reason.clone())
        }
    }

    struct SlowCheck {
        name: String,
        phase: BootPhase,
        criticality: Criticality,
        delay: Duration,
    }

    impl SlowCheck {
        fn new(name: &str, phase: BootPhase, criticality: Criticality, delay: Duration) -> Self {
            Self {
                name: name.to_string(),
                phase,
                criticality,
                delay,
            }
        }
    }

    #[async_trait]
    impl PreFlightCheck for SlowCheck {
        fn name(&self) -> &str {
            &self.name
        }
        fn phase(&self) -> BootPhase {
            self.phase
        }
        fn criticality(&self) -> Criticality {
            self.criticality
        }
        fn timeout(&self) -> Duration {
            Duration::from_millis(50)
        }
        async fn execute(&self) -> Result<(), String> {
            tokio::time::sleep(self.delay).await;
            Ok(())
        }
    }

    // --- BootPhase tests ---

    #[test]
    fn test_boot_phase_ordering() {
        assert!(BootPhase::Infrastructure < BootPhase::Sensory);
        assert!(BootPhase::Sensory < BootPhase::Regulatory);
        assert!(BootPhase::Regulatory < BootPhase::Strategy);
        assert!(BootPhase::Strategy < BootPhase::Executive);
    }

    #[test]
    fn test_boot_phase_all() {
        let phases = BootPhase::all();
        assert_eq!(phases.len(), 5);
        assert_eq!(phases[0], BootPhase::Infrastructure);
        assert_eq!(phases[4], BootPhase::Executive);
    }

    #[test]
    fn test_boot_phase_display() {
        assert_eq!(format!("{}", BootPhase::Infrastructure), "Infrastructure");
        assert_eq!(format!("{}", BootPhase::Executive), "Executive");
    }

    // --- Criticality tests ---

    #[test]
    fn test_criticality_ordering() {
        assert!(Criticality::Critical < Criticality::Required);
        assert!(Criticality::Required < Criticality::Optional);
    }

    #[test]
    fn test_criticality_display() {
        assert_eq!(format!("{}", Criticality::Critical), "CRITICAL");
        assert_eq!(format!("{}", Criticality::Required), "REQUIRED");
        assert_eq!(format!("{}", Criticality::Optional), "OPTIONAL");
    }

    // --- CheckOutcome tests ---

    #[test]
    fn test_check_outcome_pass() {
        let outcome = CheckOutcome::Pass;
        assert!(outcome.is_pass());
        assert!(!outcome.is_fail());
        assert!(!outcome.is_skipped());
    }

    #[test]
    fn test_check_outcome_fail() {
        let outcome = CheckOutcome::Fail("connection refused".to_string());
        assert!(!outcome.is_pass());
        assert!(outcome.is_fail());
    }

    #[test]
    fn test_check_outcome_timeout() {
        let outcome = CheckOutcome::Timeout;
        assert!(outcome.is_fail()); // Timeout counts as a failure
    }

    #[test]
    fn test_check_outcome_skipped() {
        let outcome = CheckOutcome::Skipped("prior failure".to_string());
        assert!(outcome.is_skipped());
        assert!(!outcome.is_fail());
    }

    // --- CheckResult tests ---

    #[test]
    fn test_check_result_pass() {
        let r = CheckResult::pass(
            "Redis",
            BootPhase::Infrastructure,
            Criticality::Critical,
            Duration::from_millis(5),
        );
        assert!(r.outcome.is_pass());
        assert!(!r.is_abort_worthy());
        assert_eq!(r.duration_ms, 5);
    }

    #[test]
    fn test_check_result_critical_failure_is_abort_worthy() {
        let r = CheckResult::fail(
            "Redis",
            BootPhase::Infrastructure,
            Criticality::Critical,
            Duration::from_millis(100),
            "connection refused",
        );
        assert!(r.is_abort_worthy());
    }

    #[test]
    fn test_check_result_optional_failure_not_abort_worthy() {
        let r = CheckResult::fail(
            "ViViT Model",
            BootPhase::Sensory,
            Criticality::Optional,
            Duration::from_millis(5),
            "model file not found",
        );
        assert!(!r.is_abort_worthy());
    }

    #[test]
    fn test_check_result_with_detail() {
        let r = CheckResult::pass(
            "Test",
            BootPhase::Infrastructure,
            Criticality::Optional,
            Duration::ZERO,
        )
        .with_detail("version 7.2.0");
        assert_eq!(r.detail.as_deref(), Some("version 7.2.0"));
    }

    // --- BootReport tests ---

    #[test]
    fn test_boot_report_empty_is_safe() {
        let report = BootReport::new();
        assert!(report.is_boot_safe());
        assert_eq!(report.total_count(), 0);
    }

    #[test]
    fn test_boot_report_all_pass() {
        let mut report = BootReport::new();
        report.results.push(CheckResult::pass(
            "A",
            BootPhase::Infrastructure,
            Criticality::Critical,
            Duration::ZERO,
        ));
        report.results.push(CheckResult::pass(
            "B",
            BootPhase::Sensory,
            Criticality::Required,
            Duration::ZERO,
        ));
        assert!(report.is_boot_safe());
        assert_eq!(report.pass_count(), 2);
        assert_eq!(report.fail_count(), 0);
    }

    #[test]
    fn test_boot_report_critical_failure_blocks_boot() {
        let mut report = BootReport::new();
        report.results.push(CheckResult::pass(
            "A",
            BootPhase::Infrastructure,
            Criticality::Critical,
            Duration::ZERO,
        ));
        report.results.push(CheckResult::fail(
            "B",
            BootPhase::Infrastructure,
            Criticality::Critical,
            Duration::ZERO,
            "down",
        ));
        assert!(!report.is_boot_safe());
        assert_eq!(report.critical_failures().len(), 1);
    }

    #[test]
    fn test_boot_report_required_failure_blocks_boot() {
        let mut report = BootReport::new();
        report.results.push(CheckResult::fail(
            "B",
            BootPhase::Regulatory,
            Criticality::Required,
            Duration::ZERO,
            "not armed",
        ));
        assert!(!report.is_boot_safe());
    }

    #[test]
    fn test_boot_report_optional_failure_allows_boot() {
        let mut report = BootReport::new();
        report.results.push(CheckResult::pass(
            "A",
            BootPhase::Infrastructure,
            Criticality::Critical,
            Duration::ZERO,
        ));
        report.results.push(CheckResult::fail(
            "ViViT",
            BootPhase::Sensory,
            Criticality::Optional,
            Duration::ZERO,
            "not found",
        ));
        assert!(report.is_boot_safe());
    }

    #[test]
    fn test_boot_report_aborted_blocks_boot() {
        let mut report = BootReport::new();
        report.aborted = true;
        assert!(!report.is_boot_safe());
    }

    #[test]
    fn test_boot_report_summary() {
        let mut report = BootReport::new();
        report.results.push(CheckResult::pass(
            "A",
            BootPhase::Infrastructure,
            Criticality::Critical,
            Duration::ZERO,
        ));
        let summary = report.summary();
        assert!(summary.contains("BOOT SAFE"));
        assert!(summary.contains("1/1 passed"));
    }

    #[test]
    fn test_boot_report_full_report_not_empty() {
        let mut report = BootReport::new();
        report.results.push(CheckResult::pass(
            "Redis",
            BootPhase::Infrastructure,
            Criticality::Critical,
            Duration::from_millis(3),
        ));
        report.results.push(CheckResult::fail(
            "ViViT",
            BootPhase::Sensory,
            Criticality::Optional,
            Duration::from_millis(1),
            "model not found",
        ));
        let text = report.full_report();
        assert!(text.contains("JANUS PRE-FLIGHT BOOT REPORT"));
        assert!(text.contains("Redis"));
        assert!(text.contains("ViViT"));
    }

    #[test]
    fn test_boot_report_discord_message() {
        let mut report = BootReport::new();
        report.results.push(CheckResult::fail(
            "Redis",
            BootPhase::Infrastructure,
            Criticality::Critical,
            Duration::ZERO,
            "connection refused",
        ));
        let msg = report.discord_message();
        assert!(msg.contains("BOOT BLOCKED"));
        assert!(msg.contains("Redis"));
        assert!(msg.contains("connection refused"));
    }

    #[test]
    fn test_boot_report_phase_results() {
        let mut report = BootReport::new();
        report.results.push(CheckResult::pass(
            "A",
            BootPhase::Infrastructure,
            Criticality::Critical,
            Duration::ZERO,
        ));
        report.results.push(CheckResult::pass(
            "B",
            BootPhase::Sensory,
            Criticality::Required,
            Duration::ZERO,
        ));
        report.results.push(CheckResult::pass(
            "C",
            BootPhase::Infrastructure,
            Criticality::Required,
            Duration::ZERO,
        ));

        let infra = report.phase_results(BootPhase::Infrastructure);
        assert_eq!(infra.len(), 2);

        let sensory = report.phase_results(BootPhase::Sensory);
        assert_eq!(sensory.len(), 1);

        let regulatory = report.phase_results(BootPhase::Regulatory);
        assert_eq!(regulatory.len(), 0);
    }

    // --- PreFlightRunner tests ---

    #[tokio::test]
    async fn test_runner_all_pass() {
        let mut runner = PreFlightRunner::new();
        runner.add_check(Box::new(PassingCheck::new(
            "Redis",
            BootPhase::Infrastructure,
            Criticality::Critical,
        )));
        runner.add_check(Box::new(PassingCheck::new(
            "Postgres",
            BootPhase::Infrastructure,
            Criticality::Critical,
        )));
        runner.add_check(Box::new(PassingCheck::new(
            "WS Feed",
            BootPhase::Sensory,
            Criticality::Required,
        )));

        let report = runner.run().await;
        assert!(report.is_boot_safe());
        assert_eq!(report.pass_count(), 3);
        assert_eq!(report.fail_count(), 0);
        assert_eq!(report.skip_count(), 0);
    }

    #[tokio::test]
    async fn test_runner_critical_failure_aborts_later_phases() {
        let mut runner = PreFlightRunner::new();
        runner.add_check(Box::new(FailingCheck::new(
            "Redis",
            BootPhase::Infrastructure,
            Criticality::Critical,
            "connection refused",
        )));
        runner.add_check(Box::new(PassingCheck::new(
            "WS Feed",
            BootPhase::Sensory,
            Criticality::Required,
        )));
        runner.add_check(Box::new(PassingCheck::new(
            "Kill Switch",
            BootPhase::Regulatory,
            Criticality::Critical,
        )));

        let report = runner.run().await;
        assert!(!report.is_boot_safe());
        assert!(report.aborted);
        assert_eq!(report.abort_phase, Some(BootPhase::Infrastructure));
        assert_eq!(report.abort_check.as_deref(), Some("Redis"));

        // Sensory and Regulatory checks should be skipped
        assert_eq!(report.fail_count(), 1); // Only Redis
        assert_eq!(report.skip_count(), 2); // WS Feed + Kill Switch
    }

    #[tokio::test]
    async fn test_runner_optional_failure_does_not_abort() {
        let mut runner = PreFlightRunner::new();
        runner.add_check(Box::new(PassingCheck::new(
            "Redis",
            BootPhase::Infrastructure,
            Criticality::Critical,
        )));
        runner.add_check(Box::new(FailingCheck::new(
            "ViViT",
            BootPhase::Sensory,
            Criticality::Optional,
            "model not found",
        )));
        runner.add_check(Box::new(PassingCheck::new(
            "Kill Switch",
            BootPhase::Regulatory,
            Criticality::Critical,
        )));

        let report = runner.run().await;
        assert!(report.is_boot_safe());
        assert!(!report.aborted);
        assert_eq!(report.pass_count(), 2);
        assert_eq!(report.fail_count(), 1);
        assert_eq!(report.skip_count(), 0);
    }

    #[tokio::test]
    async fn test_runner_timeout_check() {
        let mut runner = PreFlightRunner::new();
        runner.add_check(Box::new(SlowCheck::new(
            "Slow DB",
            BootPhase::Infrastructure,
            Criticality::Critical,
            Duration::from_secs(5), // Way longer than the 50ms timeout
        )));

        let report = runner.run().await;
        assert!(!report.is_boot_safe());
        assert_eq!(report.fail_count(), 1);

        let result = &report.results[0];
        assert_eq!(result.outcome, CheckOutcome::Timeout);
    }

    #[tokio::test]
    async fn test_runner_no_abort_mode() {
        let config = PreFlightConfig {
            abort_on_critical: false,
            ..Default::default()
        };
        let mut runner = PreFlightRunner::with_config(config);

        runner.add_check(Box::new(FailingCheck::new(
            "Redis",
            BootPhase::Infrastructure,
            Criticality::Critical,
            "down",
        )));
        runner.add_check(Box::new(PassingCheck::new(
            "WS Feed",
            BootPhase::Sensory,
            Criticality::Required,
        )));

        let report = runner.run().await;
        // Boot is still not safe because of the critical failure
        assert!(!report.is_boot_safe());
        // But nothing was skipped — all checks ran
        assert!(!report.aborted);
        assert_eq!(report.skip_count(), 0);
        assert_eq!(report.fail_count(), 1);
        assert_eq!(report.pass_count(), 1);
    }

    #[tokio::test]
    async fn test_runner_empty() {
        let runner = PreFlightRunner::new();
        let report = runner.run().await;
        assert!(report.is_boot_safe());
        assert_eq!(report.total_count(), 0);
    }

    #[tokio::test]
    async fn test_runner_check_count() {
        let mut runner = PreFlightRunner::new();
        assert_eq!(runner.check_count(), 0);
        runner.add_check(Box::new(PassingCheck::new(
            "A",
            BootPhase::Infrastructure,
            Criticality::Critical,
        )));
        assert_eq!(runner.check_count(), 1);
        runner.add_checks(vec![
            Box::new(PassingCheck::new(
                "B",
                BootPhase::Sensory,
                Criticality::Required,
            )),
            Box::new(PassingCheck::new(
                "C",
                BootPhase::Strategy,
                Criticality::Optional,
            )),
        ]);
        assert_eq!(runner.check_count(), 3);
    }

    #[tokio::test]
    async fn test_runner_multiple_phases_correct_order() {
        let mut runner = PreFlightRunner::new();

        // Add in reverse order — runner should still execute in phase order
        runner.add_check(Box::new(PassingCheck::new(
            "Exec",
            BootPhase::Executive,
            Criticality::Required,
        )));
        runner.add_check(Box::new(PassingCheck::new(
            "Strat",
            BootPhase::Strategy,
            Criticality::Required,
        )));
        runner.add_check(Box::new(PassingCheck::new(
            "Infra",
            BootPhase::Infrastructure,
            Criticality::Critical,
        )));

        let report = runner.run().await;
        assert!(report.is_boot_safe());

        // Results should be in phase order
        assert_eq!(report.results[0].name, "Infra");
        assert_eq!(report.results[1].name, "Strat");
        assert_eq!(report.results[2].name, "Exec");
    }

    #[tokio::test]
    async fn test_runner_critical_failure_mid_phase_skips_rest_of_phase() {
        let mut runner = PreFlightRunner::new();

        runner.add_check(Box::new(PassingCheck::new(
            "Infra-A",
            BootPhase::Infrastructure,
            Criticality::Required,
        )));
        runner.add_check(Box::new(FailingCheck::new(
            "Infra-B",
            BootPhase::Infrastructure,
            Criticality::Critical,
            "down",
        )));
        runner.add_check(Box::new(PassingCheck::new(
            "Infra-C",
            BootPhase::Infrastructure,
            Criticality::Required,
        )));

        let report = runner.run().await;
        assert!(!report.is_boot_safe());

        // Infra-A passed, Infra-B failed (critical), Infra-C should be skipped
        assert_eq!(report.results[0].outcome, CheckOutcome::Pass);
        assert!(report.results[1].outcome.is_fail());
        assert!(report.results[2].outcome.is_skipped());
    }

    #[tokio::test]
    async fn test_runner_parallel_within_phase() {
        let config = PreFlightConfig {
            parallel_within_phase: true,
            ..Default::default()
        };
        let mut runner = PreFlightRunner::with_config(config);

        runner.add_check(Box::new(PassingCheck::new(
            "A",
            BootPhase::Infrastructure,
            Criticality::Critical,
        )));
        runner.add_check(Box::new(PassingCheck::new(
            "B",
            BootPhase::Infrastructure,
            Criticality::Required,
        )));
        runner.add_check(Box::new(PassingCheck::new(
            "C",
            BootPhase::Sensory,
            Criticality::Required,
        )));

        let report = runner.run().await;
        assert!(report.is_boot_safe());
        assert_eq!(report.pass_count(), 3);
    }
}
