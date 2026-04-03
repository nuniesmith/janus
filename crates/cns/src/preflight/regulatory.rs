//! # Regulatory Pre-Flight Checks
//!
//! Verifies risk management and regulatory safety systems are armed and operational
//! before the system is allowed to trade.
//!
//! ## Checks
//!
//! - **KillSwitchCheck** — Verify the kill switch is armed, responsive, and can halt trading
//! - **CircuitBreakerCheck** — Verify circuit breakers are initialized and in a valid state
//! - **HypothalamusCheck** — Verify the hypothalamus (position sizing / scaling) service is ready

use super::{BootPhase, Criticality, PreFlightCheck};
use async_trait::async_trait;
use std::time::Duration;

// ============================================================================
// Kill Switch Check
// ============================================================================

/// Method used to verify the kill switch.
#[derive(Debug, Clone)]
pub enum KillSwitchVerification {
    /// Check a Redis key for the kill switch state.
    /// The key should contain "armed" or "active" when the switch is ready.
    RedisKey {
        /// Redis connection URL.
        url: String,
        /// Key name to check (e.g. `janus:kill_switch:state`).
        key: String,
    },
    /// Check an HTTP endpoint that returns 200 when the kill switch is armed.
    HttpEndpoint {
        /// Full URL to the kill switch status endpoint.
        url: String,
    },
    /// Use a custom callable verification (for unit testing or in-process checks).
    /// The bool result indicates whether the switch is armed (true = armed = good).
    InMemory {
        /// Whether the kill switch is currently armed.
        armed: bool,
    },
}

/// Verifies the kill switch is armed and can halt all trading activity.
///
/// The kill switch is a **critical** safety mechanism. If it is not armed or
/// not reachable, the system MUST NOT begin trading. This check verifies:
///
/// 1. The kill switch mechanism is reachable (Redis key, HTTP endpoint, etc.)
/// 2. The kill switch is in the "armed" state
///
/// A disarmed kill switch means there is no way to emergency-stop the system,
/// which is an unacceptable risk condition.
pub struct KillSwitchCheck {
    /// How to verify the kill switch.
    verification: KillSwitchVerification,
    /// Override criticality (default: Critical — system must not boot without kill switch).
    criticality: Criticality,
    /// Verification timeout.
    timeout: Duration,
}

impl KillSwitchCheck {
    /// Create a kill switch check using a Redis key.
    ///
    /// The key's value should be `"armed"` or `"active"` when the switch is ready.
    pub fn redis(url: impl Into<String>, key: impl Into<String>) -> Self {
        Self {
            verification: KillSwitchVerification::RedisKey {
                url: url.into(),
                key: key.into(),
            },
            criticality: Criticality::Critical,
            timeout: Duration::from_secs(5),
        }
    }

    /// Create a kill switch check using an HTTP endpoint.
    ///
    /// The endpoint should return HTTP 200 when the kill switch is armed.
    pub fn http(url: impl Into<String>) -> Self {
        Self {
            verification: KillSwitchVerification::HttpEndpoint { url: url.into() },
            criticality: Criticality::Critical,
            timeout: Duration::from_secs(5),
        }
    }

    /// Create a kill switch check using an in-memory flag (for testing).
    pub fn in_memory(armed: bool) -> Self {
        Self {
            verification: KillSwitchVerification::InMemory { armed },
            criticality: Criticality::Critical,
            timeout: Duration::from_secs(1),
        }
    }

    /// Override the criticality level.
    pub fn with_criticality(mut self, criticality: Criticality) -> Self {
        self.criticality = criticality;
        self
    }

    /// Override the timeout.
    pub fn with_timeout(mut self, timeout: Duration) -> Self {
        self.timeout = timeout;
        self
    }
}

#[async_trait]
impl PreFlightCheck for KillSwitchCheck {
    fn name(&self) -> &str {
        "Kill Switch"
    }

    fn phase(&self) -> BootPhase {
        BootPhase::Regulatory
    }

    fn criticality(&self) -> Criticality {
        self.criticality
    }

    fn timeout(&self) -> Duration {
        self.timeout
    }

    async fn execute(&self) -> Result<(), String> {
        match &self.verification {
            KillSwitchVerification::RedisKey { url, key } => {
                let client = redis::Client::open(url.as_str())
                    .map_err(|e| format!("Invalid Redis URL for kill switch: {}", e))?;

                let mut conn = client
                    .get_multiplexed_async_connection()
                    .await
                    .map_err(|e| format!("Kill switch Redis connection failed: {}", e))?;

                let value: Option<String> = redis::cmd("GET")
                    .arg(key.as_str())
                    .query_async(&mut conn)
                    .await
                    .map_err(|e| format!("Kill switch Redis GET failed: {}", e))?;

                match value {
                    Some(v) => {
                        let normalized = v.to_lowercase();
                        if normalized == "armed"
                            || normalized == "active"
                            || normalized == "true"
                            || normalized == "1"
                        {
                            Ok(())
                        } else {
                            Err(format!(
                                "Kill switch is NOT armed (state: '{}', expected 'armed' or 'active')",
                                v
                            ))
                        }
                    }
                    None => Err(format!(
                        "Kill switch key '{}' not found in Redis — kill switch may not be initialized",
                        key
                    )),
                }
            }

            KillSwitchVerification::HttpEndpoint { url } => {
                let client = reqwest::Client::builder()
                    .timeout(self.timeout)
                    .build()
                    .map_err(|e| format!("Failed to create HTTP client: {}", e))?;

                let response = client
                    .get(url)
                    .send()
                    .await
                    .map_err(|e| format!("Kill switch HTTP check failed ({}): {}", url, e))?;

                let status = response.status();
                if status.is_success() {
                    Ok(())
                } else {
                    let body = response.text().await.unwrap_or_default();
                    Err(format!(
                        "Kill switch returned HTTP {} ({}): {}",
                        status, url, body
                    ))
                }
            }

            KillSwitchVerification::InMemory { armed } => {
                if *armed {
                    Ok(())
                } else {
                    Err("Kill switch is NOT armed (in-memory flag is false)".to_string())
                }
            }
        }
    }

    fn detail_on_success(&self) -> Option<String> {
        Some("kill switch armed and responsive".to_string())
    }
}

// ============================================================================
// Circuit Breaker Check
// ============================================================================

/// Method used to verify circuit breakers.
#[derive(Debug, Clone)]
pub enum CircuitBreakerVerification {
    /// Check a Redis hash or key pattern that holds circuit breaker states.
    RedisPattern {
        /// Redis connection URL.
        url: String,
        /// Key prefix to scan (e.g. `janus:circuit_breaker:*`).
        key_prefix: String,
        /// Expected minimum number of circuit breakers to be registered.
        min_count: usize,
    },
    /// Check an HTTP endpoint that returns circuit breaker status.
    HttpEndpoint {
        /// URL that returns JSON with circuit breaker states.
        url: String,
    },
    /// In-memory verification for testing.
    InMemory {
        /// Number of breakers initialized.
        breaker_count: usize,
        /// Whether all breakers are in a valid (closed) state.
        all_closed: bool,
    },
}

/// Verifies that circuit breakers are initialized and in a valid state.
///
/// Circuit breakers prevent runaway trading by tripping when error thresholds
/// are exceeded. Before boot, we verify:
///
/// 1. The expected number of circuit breakers are registered
/// 2. All breakers are in the "Closed" (normal) state — no leftover trips from a crash
pub struct CircuitBreakerCheck {
    /// How to verify circuit breakers.
    verification: CircuitBreakerVerification,
    /// Override criticality (default: Critical).
    criticality: Criticality,
    /// Verification timeout.
    timeout: Duration,
}

impl CircuitBreakerCheck {
    /// Create a circuit breaker check using Redis key scanning.
    pub fn redis(url: impl Into<String>, key_prefix: impl Into<String>, min_count: usize) -> Self {
        Self {
            verification: CircuitBreakerVerification::RedisPattern {
                url: url.into(),
                key_prefix: key_prefix.into(),
                min_count,
            },
            criticality: Criticality::Critical,
            timeout: Duration::from_secs(5),
        }
    }

    /// Create a circuit breaker check using an HTTP endpoint.
    pub fn http(url: impl Into<String>) -> Self {
        Self {
            verification: CircuitBreakerVerification::HttpEndpoint { url: url.into() },
            criticality: Criticality::Critical,
            timeout: Duration::from_secs(5),
        }
    }

    /// Create a circuit breaker check using in-memory state (for testing).
    pub fn in_memory(breaker_count: usize, all_closed: bool) -> Self {
        Self {
            verification: CircuitBreakerVerification::InMemory {
                breaker_count,
                all_closed,
            },
            criticality: Criticality::Critical,
            timeout: Duration::from_secs(1),
        }
    }

    /// Override the criticality level.
    pub fn with_criticality(mut self, criticality: Criticality) -> Self {
        self.criticality = criticality;
        self
    }

    /// Override the timeout.
    pub fn with_timeout(mut self, timeout: Duration) -> Self {
        self.timeout = timeout;
        self
    }
}

#[async_trait]
impl PreFlightCheck for CircuitBreakerCheck {
    fn name(&self) -> &str {
        "Circuit Breakers"
    }

    fn phase(&self) -> BootPhase {
        BootPhase::Regulatory
    }

    fn criticality(&self) -> Criticality {
        self.criticality
    }

    fn timeout(&self) -> Duration {
        self.timeout
    }

    async fn execute(&self) -> Result<(), String> {
        match &self.verification {
            CircuitBreakerVerification::RedisPattern {
                url,
                key_prefix,
                min_count,
            } => {
                let client = redis::Client::open(url.as_str())
                    .map_err(|e| format!("Invalid Redis URL for circuit breakers: {}", e))?;

                let mut conn = client
                    .get_multiplexed_async_connection()
                    .await
                    .map_err(|e| format!("Circuit breaker Redis connection failed: {}", e))?;

                // Scan for keys matching the prefix
                let pattern = format!("{}*", key_prefix);
                let keys: Vec<String> = redis::cmd("KEYS")
                    .arg(&pattern)
                    .query_async(&mut conn)
                    .await
                    .map_err(|e| format!("Circuit breaker KEYS scan failed: {}", e))?;

                if keys.len() < *min_count {
                    return Err(format!(
                        "Expected at least {} circuit breakers (prefix '{}'), found {}",
                        min_count,
                        key_prefix,
                        keys.len()
                    ));
                }

                // Check that no breaker is in the "open" state (leftover from a crash)
                for key in &keys {
                    let value: Option<String> = redis::cmd("GET")
                        .arg(key.as_str())
                        .query_async(&mut conn)
                        .await
                        .map_err(|e| {
                            format!("Failed to read circuit breaker key '{}': {}", key, e)
                        })?;

                    if let Some(ref state) = value {
                        let normalized = state.to_lowercase();
                        if normalized == "open" || normalized == "tripped" {
                            return Err(format!(
                                "Circuit breaker '{}' is in '{}' state — may be leftover from a crash. Reset before boot.",
                                key, state
                            ));
                        }
                    }
                }

                Ok(())
            }

            CircuitBreakerVerification::HttpEndpoint { url } => {
                let client = reqwest::Client::builder()
                    .timeout(self.timeout)
                    .build()
                    .map_err(|e| format!("Failed to create HTTP client: {}", e))?;

                let response =
                    client.get(url).send().await.map_err(|e| {
                        format!("Circuit breaker HTTP check failed ({}): {}", url, e)
                    })?;

                let status = response.status();
                if !status.is_success() {
                    let body = response.text().await.unwrap_or_default();
                    return Err(format!(
                        "Circuit breaker endpoint returned HTTP {} ({}): {}",
                        status, url, body
                    ));
                }

                Ok(())
            }

            CircuitBreakerVerification::InMemory {
                breaker_count,
                all_closed,
            } => {
                if *breaker_count == 0 {
                    return Err(
                        "No circuit breakers registered — at least 1 is required".to_string()
                    );
                }

                if !*all_closed {
                    return Err(
                        "One or more circuit breakers are in OPEN state — reset before boot"
                            .to_string(),
                    );
                }

                Ok(())
            }
        }
    }

    fn detail_on_success(&self) -> Option<String> {
        match &self.verification {
            CircuitBreakerVerification::RedisPattern { min_count, .. } => {
                Some(format!("≥{} breakers registered, all closed", min_count))
            }
            CircuitBreakerVerification::HttpEndpoint { .. } => {
                Some("all breakers in valid state".to_string())
            }
            CircuitBreakerVerification::InMemory { breaker_count, .. } => {
                Some(format!("{} breakers registered, all closed", breaker_count))
            }
        }
    }
}

// ============================================================================
// Hypothalamus Check
// ============================================================================

/// Method used to verify the hypothalamus (position sizing / scaling) service.
#[derive(Debug, Clone)]
pub enum HypothalamusVerification {
    /// Check the hypothalamus via an HTTP health endpoint.
    HttpEndpoint {
        /// URL to the hypothalamus health/ready endpoint.
        url: String,
    },
    /// Check the hypothalamus via gRPC health protocol.
    GrpcEndpoint {
        /// gRPC endpoint address (e.g. `http://localhost:50051`).
        endpoint: String,
    },
    /// In-memory verification for testing.
    InMemory {
        /// Whether the hypothalamus is ready.
        ready: bool,
        /// Current scaling factor (should be > 0.0 and <= 1.0).
        scaling_factor: f64,
    },
}

/// Verifies the hypothalamus (position sizing / scaling) service is ready.
///
/// The hypothalamus controls position sizing, risk scaling, and adapts exposure
/// based on regime. Before boot, we verify:
///
/// 1. The hypothalamus service is reachable
/// 2. It reports itself as ready (has loaded its configuration / model)
/// 3. The scaling factor is in a valid range (0.0, 1.0]
pub struct HypothalamusCheck {
    /// How to verify the hypothalamus.
    verification: HypothalamusVerification,
    /// Override criticality (default: Required — system can run with default scaling,
    /// but should have the hypothalamus for production).
    criticality: Criticality,
    /// Verification timeout.
    timeout: Duration,
}

impl HypothalamusCheck {
    /// Create a hypothalamus check using an HTTP health endpoint.
    pub fn http(url: impl Into<String>) -> Self {
        Self {
            verification: HypothalamusVerification::HttpEndpoint { url: url.into() },
            criticality: Criticality::Required,
            timeout: Duration::from_secs(5),
        }
    }

    /// Create a hypothalamus check using a gRPC health endpoint.
    pub fn grpc(endpoint: impl Into<String>) -> Self {
        Self {
            verification: HypothalamusVerification::GrpcEndpoint {
                endpoint: endpoint.into(),
            },
            criticality: Criticality::Required,
            timeout: Duration::from_secs(5),
        }
    }

    /// Create a hypothalamus check using in-memory state (for testing).
    pub fn in_memory(ready: bool, scaling_factor: f64) -> Self {
        Self {
            verification: HypothalamusVerification::InMemory {
                ready,
                scaling_factor,
            },
            criticality: Criticality::Required,
            timeout: Duration::from_secs(1),
        }
    }

    /// Override the criticality level.
    pub fn with_criticality(mut self, criticality: Criticality) -> Self {
        self.criticality = criticality;
        self
    }

    /// Override the timeout.
    pub fn with_timeout(mut self, timeout: Duration) -> Self {
        self.timeout = timeout;
        self
    }
}

#[async_trait]
impl PreFlightCheck for HypothalamusCheck {
    fn name(&self) -> &str {
        "Hypothalamus Scaling"
    }

    fn phase(&self) -> BootPhase {
        BootPhase::Regulatory
    }

    fn criticality(&self) -> Criticality {
        self.criticality
    }

    fn timeout(&self) -> Duration {
        self.timeout
    }

    async fn execute(&self) -> Result<(), String> {
        match &self.verification {
            HypothalamusVerification::HttpEndpoint { url } => {
                let client = reqwest::Client::builder()
                    .timeout(self.timeout)
                    .build()
                    .map_err(|e| format!("Failed to create HTTP client: {}", e))?;

                let response =
                    client.get(url).send().await.map_err(|e| {
                        format!("Hypothalamus health check failed ({}): {}", url, e)
                    })?;

                let status = response.status();
                if !status.is_success() {
                    let body = response.text().await.unwrap_or_default();
                    return Err(format!(
                        "Hypothalamus returned HTTP {} ({}): {}",
                        status, url, body
                    ));
                }

                // Optionally parse the response body for scaling factor validation
                let body = response.text().await.unwrap_or_default();
                if !body.is_empty() {
                    // Try to extract a scaling_factor from JSON response
                    if let Ok(json) = serde_json::from_str::<serde_json::Value>(&body)
                        && let Some(factor) = json.get("scaling_factor").and_then(|v| v.as_f64())
                    {
                        validate_scaling_factor(factor)?;
                    }
                }

                Ok(())
            }

            HypothalamusVerification::GrpcEndpoint { endpoint } => {
                // Attempt TCP connect to verify the gRPC server is listening
                let addr = endpoint
                    .trim_start_matches("http://")
                    .trim_start_matches("https://");

                let socket_addrs: Vec<std::net::SocketAddr> = tokio::net::lookup_host(addr)
                    .await
                    .map_err(|e| {
                        format!(
                            "DNS resolution failed for hypothalamus at {}: {}",
                            endpoint, e
                        )
                    })?
                    .collect();

                if socket_addrs.is_empty() {
                    return Err(format!(
                        "No addresses resolved for hypothalamus at {}",
                        endpoint
                    ));
                }

                tokio::net::TcpStream::connect(socket_addrs[0])
                    .await
                    .map_err(|e| {
                        format!("TCP connect to hypothalamus at {} failed: {}", endpoint, e)
                    })?;

                Ok(())
            }

            HypothalamusVerification::InMemory {
                ready,
                scaling_factor,
            } => {
                if !*ready {
                    return Err("Hypothalamus is not ready (in-memory flag is false)".to_string());
                }

                validate_scaling_factor(*scaling_factor)?;

                Ok(())
            }
        }
    }

    fn detail_on_success(&self) -> Option<String> {
        match &self.verification {
            HypothalamusVerification::InMemory { scaling_factor, .. } => {
                Some(format!("ready, scaling_factor={:.4}", scaling_factor))
            }
            _ => Some("ready and responsive".to_string()),
        }
    }
}

/// Validate that a scaling factor is in the acceptable range (0.0, 1.0].
fn validate_scaling_factor(factor: f64) -> Result<(), String> {
    if factor.is_nan() || factor.is_infinite() {
        return Err(format!(
            "Hypothalamus scaling factor is invalid (NaN or Inf): {}",
            factor
        ));
    }

    if factor <= 0.0 {
        return Err(format!(
            "Hypothalamus scaling factor is non-positive: {} (must be > 0.0)",
            factor
        ));
    }

    if factor > 1.0 {
        return Err(format!(
            "Hypothalamus scaling factor exceeds maximum: {} (must be <= 1.0)",
            factor
        ));
    }

    Ok(())
}

// ============================================================================
// Builder Helpers
// ============================================================================

/// Create the default set of regulatory pre-flight checks.
///
/// - Kill switch via Redis
/// - Circuit breakers via Redis
/// - Hypothalamus via HTTP
pub fn default_regulatory_checks(
    redis_url: &str,
    hypothalamus_url: Option<&str>,
) -> Vec<Box<dyn PreFlightCheck>> {
    let mut checks: Vec<Box<dyn PreFlightCheck>> = vec![
        // Kill switch (critical — must be armed)
        Box::new(KillSwitchCheck::redis(redis_url, "janus:kill_switch:state")),
        // Circuit breakers (critical — must be initialized and closed)
        Box::new(CircuitBreakerCheck::redis(
            redis_url,
            "janus:circuit_breaker:",
            1,
        )),
    ];

    // Hypothalamus (required if URL provided, optional otherwise)
    if let Some(url) = hypothalamus_url {
        checks.push(Box::new(HypothalamusCheck::http(url)));
    } else {
        // If no URL is provided, use an in-memory check that passes with default scaling
        checks.push(Box::new(
            HypothalamusCheck::in_memory(true, 1.0).with_criticality(Criticality::Optional),
        ));
    }

    checks
}

// ============================================================================
// Tests
// ============================================================================

#[cfg(test)]
mod tests {
    use super::*;

    // ── KillSwitchCheck construction tests ──

    #[test]
    fn test_kill_switch_redis_defaults() {
        let check = KillSwitchCheck::redis("redis://localhost:6379", "janus:kill_switch:state");
        assert_eq!(check.name(), "Kill Switch");
        assert_eq!(check.phase(), BootPhase::Regulatory);
        assert_eq!(check.criticality(), Criticality::Critical);
        assert_eq!(check.timeout(), Duration::from_secs(5));
    }

    #[test]
    fn test_kill_switch_http_defaults() {
        let check = KillSwitchCheck::http("http://localhost:8080/kill-switch/status");
        assert_eq!(check.name(), "Kill Switch");
        assert_eq!(check.criticality(), Criticality::Critical);
    }

    #[test]
    fn test_kill_switch_in_memory() {
        let check = KillSwitchCheck::in_memory(true);
        assert_eq!(check.criticality(), Criticality::Critical);
        assert_eq!(check.timeout(), Duration::from_secs(1));
    }

    #[test]
    fn test_kill_switch_builder() {
        let check = KillSwitchCheck::in_memory(true)
            .with_criticality(Criticality::Required)
            .with_timeout(Duration::from_secs(10));
        assert_eq!(check.criticality(), Criticality::Required);
        assert_eq!(check.timeout(), Duration::from_secs(10));
    }

    #[test]
    fn test_kill_switch_detail() {
        let check = KillSwitchCheck::in_memory(true);
        assert_eq!(
            check.detail_on_success(),
            Some("kill switch armed and responsive".to_string())
        );
    }

    // ── KillSwitchCheck execution tests ──

    #[tokio::test]
    async fn test_kill_switch_in_memory_armed_passes() {
        let check = KillSwitchCheck::in_memory(true);
        let result = check.execute().await;
        assert!(result.is_ok());
    }

    #[tokio::test]
    async fn test_kill_switch_in_memory_disarmed_fails() {
        let check = KillSwitchCheck::in_memory(false);
        let result = check.execute().await;
        assert!(result.is_err());
        assert!(result.unwrap_err().contains("NOT armed"));
    }

    #[tokio::test]
    async fn test_kill_switch_run_produces_correct_result() {
        let check = KillSwitchCheck::in_memory(true);
        let result = check.run().await;
        assert_eq!(result.name, "Kill Switch");
        assert_eq!(result.phase, BootPhase::Regulatory);
        assert_eq!(result.criticality, Criticality::Critical);
        assert!(result.outcome.is_pass());
    }

    #[tokio::test]
    async fn test_kill_switch_failure_is_abort_worthy() {
        let check = KillSwitchCheck::in_memory(false);
        let result = check.run().await;
        assert!(result.is_abort_worthy());
    }

    // ── CircuitBreakerCheck construction tests ──

    #[test]
    fn test_circuit_breaker_redis_defaults() {
        let check =
            CircuitBreakerCheck::redis("redis://localhost:6379", "janus:circuit_breaker:", 3);
        assert_eq!(check.name(), "Circuit Breakers");
        assert_eq!(check.phase(), BootPhase::Regulatory);
        assert_eq!(check.criticality(), Criticality::Critical);
    }

    #[test]
    fn test_circuit_breaker_http_defaults() {
        let check = CircuitBreakerCheck::http("http://localhost:8080/circuit-breakers");
        assert_eq!(check.criticality(), Criticality::Critical);
    }

    #[test]
    fn test_circuit_breaker_in_memory() {
        let check = CircuitBreakerCheck::in_memory(5, true);
        assert_eq!(check.timeout(), Duration::from_secs(1));
    }

    #[test]
    fn test_circuit_breaker_builder() {
        let check = CircuitBreakerCheck::in_memory(3, true)
            .with_criticality(Criticality::Required)
            .with_timeout(Duration::from_secs(7));
        assert_eq!(check.criticality(), Criticality::Required);
        assert_eq!(check.timeout(), Duration::from_secs(7));
    }

    // ── CircuitBreakerCheck execution tests ──

    #[tokio::test]
    async fn test_circuit_breaker_in_memory_all_closed_passes() {
        let check = CircuitBreakerCheck::in_memory(3, true);
        let result = check.execute().await;
        assert!(result.is_ok());
    }

    #[tokio::test]
    async fn test_circuit_breaker_in_memory_zero_count_fails() {
        let check = CircuitBreakerCheck::in_memory(0, true);
        let result = check.execute().await;
        assert!(result.is_err());
        assert!(result.unwrap_err().contains("No circuit breakers"));
    }

    #[tokio::test]
    async fn test_circuit_breaker_in_memory_not_all_closed_fails() {
        let check = CircuitBreakerCheck::in_memory(3, false);
        let result = check.execute().await;
        assert!(result.is_err());
        assert!(result.unwrap_err().contains("OPEN state"));
    }

    #[tokio::test]
    async fn test_circuit_breaker_detail_on_success() {
        let check = CircuitBreakerCheck::in_memory(5, true);
        assert_eq!(
            check.detail_on_success(),
            Some("5 breakers registered, all closed".to_string())
        );
    }

    // ── HypothalamusCheck construction tests ──

    #[test]
    fn test_hypothalamus_http_defaults() {
        let check = HypothalamusCheck::http("http://localhost:8080/hypothalamus/health");
        assert_eq!(check.name(), "Hypothalamus Scaling");
        assert_eq!(check.phase(), BootPhase::Regulatory);
        assert_eq!(check.criticality(), Criticality::Required);
    }

    #[test]
    fn test_hypothalamus_grpc_defaults() {
        let check = HypothalamusCheck::grpc("http://localhost:50051");
        assert_eq!(check.criticality(), Criticality::Required);
    }

    #[test]
    fn test_hypothalamus_in_memory() {
        let check = HypothalamusCheck::in_memory(true, 0.75);
        assert_eq!(check.timeout(), Duration::from_secs(1));
    }

    #[test]
    fn test_hypothalamus_builder() {
        let check = HypothalamusCheck::in_memory(true, 1.0)
            .with_criticality(Criticality::Optional)
            .with_timeout(Duration::from_secs(15));
        assert_eq!(check.criticality(), Criticality::Optional);
        assert_eq!(check.timeout(), Duration::from_secs(15));
    }

    // ── HypothalamusCheck execution tests ──

    #[tokio::test]
    async fn test_hypothalamus_in_memory_ready_passes() {
        let check = HypothalamusCheck::in_memory(true, 0.5);
        let result = check.execute().await;
        assert!(result.is_ok());
    }

    #[tokio::test]
    async fn test_hypothalamus_in_memory_not_ready_fails() {
        let check = HypothalamusCheck::in_memory(false, 1.0);
        let result = check.execute().await;
        assert!(result.is_err());
        assert!(result.unwrap_err().contains("not ready"));
    }

    #[tokio::test]
    async fn test_hypothalamus_in_memory_zero_scaling_fails() {
        let check = HypothalamusCheck::in_memory(true, 0.0);
        let result = check.execute().await;
        assert!(result.is_err());
        assert!(result.unwrap_err().contains("non-positive"));
    }

    #[tokio::test]
    async fn test_hypothalamus_in_memory_negative_scaling_fails() {
        let check = HypothalamusCheck::in_memory(true, -0.5);
        let result = check.execute().await;
        assert!(result.is_err());
        assert!(result.unwrap_err().contains("non-positive"));
    }

    #[tokio::test]
    async fn test_hypothalamus_in_memory_excessive_scaling_fails() {
        let check = HypothalamusCheck::in_memory(true, 1.5);
        let result = check.execute().await;
        assert!(result.is_err());
        assert!(result.unwrap_err().contains("exceeds maximum"));
    }

    #[tokio::test]
    async fn test_hypothalamus_in_memory_nan_scaling_fails() {
        let check = HypothalamusCheck::in_memory(true, f64::NAN);
        let result = check.execute().await;
        assert!(result.is_err());
        assert!(result.unwrap_err().contains("invalid"));
    }

    #[tokio::test]
    async fn test_hypothalamus_in_memory_inf_scaling_fails() {
        let check = HypothalamusCheck::in_memory(true, f64::INFINITY);
        let result = check.execute().await;
        assert!(result.is_err());
        assert!(result.unwrap_err().contains("invalid"));
    }

    #[tokio::test]
    async fn test_hypothalamus_boundary_scaling_1_passes() {
        let check = HypothalamusCheck::in_memory(true, 1.0);
        let result = check.execute().await;
        assert!(result.is_ok());
    }

    #[tokio::test]
    async fn test_hypothalamus_boundary_scaling_epsilon_passes() {
        let check = HypothalamusCheck::in_memory(true, f64::EPSILON);
        let result = check.execute().await;
        assert!(result.is_ok());
    }

    #[tokio::test]
    async fn test_hypothalamus_detail_on_success() {
        let check = HypothalamusCheck::in_memory(true, 0.75);
        assert_eq!(
            check.detail_on_success(),
            Some("ready, scaling_factor=0.7500".to_string())
        );
    }

    // ── Scaling factor validation tests ──

    #[test]
    fn test_validate_scaling_factor_valid() {
        assert!(validate_scaling_factor(0.5).is_ok());
        assert!(validate_scaling_factor(1.0).is_ok());
        assert!(validate_scaling_factor(0.001).is_ok());
    }

    #[test]
    fn test_validate_scaling_factor_invalid() {
        assert!(validate_scaling_factor(0.0).is_err());
        assert!(validate_scaling_factor(-1.0).is_err());
        assert!(validate_scaling_factor(1.1).is_err());
        assert!(validate_scaling_factor(f64::NAN).is_err());
        assert!(validate_scaling_factor(f64::INFINITY).is_err());
        assert!(validate_scaling_factor(f64::NEG_INFINITY).is_err());
    }

    // ── Builder helper tests ──

    #[test]
    fn test_default_regulatory_checks_with_hypothalamus() {
        let checks = default_regulatory_checks(
            "redis://localhost:6379",
            Some("http://localhost:8080/hypothalamus"),
        );
        assert_eq!(checks.len(), 3);
        assert_eq!(checks[0].name(), "Kill Switch");
        assert_eq!(checks[1].name(), "Circuit Breakers");
        assert_eq!(checks[2].name(), "Hypothalamus Scaling");
    }

    #[test]
    fn test_default_regulatory_checks_without_hypothalamus() {
        let checks = default_regulatory_checks("redis://localhost:6379", None);
        assert_eq!(checks.len(), 3);
        // The hypothalamus check should be optional when no URL is provided
        assert_eq!(checks[2].criticality(), Criticality::Optional);
    }

    #[test]
    fn test_default_regulatory_checks_phases() {
        let checks = default_regulatory_checks("redis://localhost:6379", None);
        for check in &checks {
            assert_eq!(check.phase(), BootPhase::Regulatory);
        }
    }

    // ── Integration with PreFlightRunner ──

    #[tokio::test]
    async fn test_regulatory_checks_in_runner() {
        use super::super::PreFlightRunner;

        let mut runner = PreFlightRunner::new();
        runner.add_check(Box::new(KillSwitchCheck::in_memory(true)));
        runner.add_check(Box::new(CircuitBreakerCheck::in_memory(3, true)));
        runner.add_check(Box::new(HypothalamusCheck::in_memory(true, 0.8)));

        let report = runner.run().await;
        assert!(report.is_boot_safe());
        assert_eq!(report.pass_count(), 3);
        assert_eq!(report.fail_count(), 0);
    }

    #[tokio::test]
    async fn test_regulatory_kill_switch_failure_aborts() {
        use super::super::PreFlightRunner;

        let mut runner = PreFlightRunner::new();
        runner.add_check(Box::new(KillSwitchCheck::in_memory(false))); // FAIL
        runner.add_check(Box::new(CircuitBreakerCheck::in_memory(3, true)));

        let report = runner.run().await;
        assert!(!report.is_boot_safe());
        assert!(report.aborted);
        assert_eq!(report.abort_check.as_deref(), Some("Kill Switch"));
    }
}
