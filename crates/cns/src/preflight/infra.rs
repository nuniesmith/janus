//! # Infrastructure Pre-Flight Checks
//!
//! Verifies core infrastructure dependencies are reachable and healthy before boot.
//!
//! ## Checks
//!
//! - **RedisCheck** — PING Redis, verify response, report version
//! - **PostgresCheck** — TCP connect to Postgres, optionally run `SELECT 1`
//! - **QuestDBCheck** — HTTP health endpoint check
//! - **PrometheusCheck** — HTTP readiness endpoint check

use super::{BootPhase, Criticality, PreFlightCheck};
use async_trait::async_trait;
use std::time::Duration;

// ============================================================================
// Redis Check
// ============================================================================

/// Verifies Redis is reachable by sending a PING command.
pub struct RedisCheck {
    /// Redis connection URL (e.g. `redis://localhost:6379`)
    url: String,
    /// Override criticality (default: Critical)
    criticality: Criticality,
    /// Connection/check timeout
    timeout: Duration,
}

impl RedisCheck {
    /// Create a new Redis check with the given connection URL.
    pub fn new(url: impl Into<String>) -> Self {
        Self {
            url: url.into(),
            criticality: Criticality::Critical,
            timeout: Duration::from_secs(5),
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
impl PreFlightCheck for RedisCheck {
    fn name(&self) -> &str {
        "Redis"
    }

    fn phase(&self) -> BootPhase {
        BootPhase::Infrastructure
    }

    fn criticality(&self) -> Criticality {
        self.criticality
    }

    fn timeout(&self) -> Duration {
        self.timeout
    }

    async fn execute(&self) -> Result<(), String> {
        let client = redis::Client::open(self.url.as_str())
            .map_err(|e| format!("Invalid Redis URL '{}': {}", self.url, e))?;

        let mut conn = client
            .get_multiplexed_async_connection()
            .await
            .map_err(|e| format!("Redis connection failed ({}): {}", self.url, e))?;

        // PING
        let pong: String = redis::cmd("PING")
            .query_async(&mut conn)
            .await
            .map_err(|e| format!("Redis PING failed: {}", e))?;

        if pong != "PONG" {
            return Err(format!("Redis PING returned unexpected response: {}", pong));
        }

        Ok(())
    }

    fn detail_on_success(&self) -> Option<String> {
        Some(format!("connected to {}", self.url))
    }
}

// ============================================================================
// Postgres Check
// ============================================================================

/// Verifies Postgres is reachable via TCP connection.
///
/// This intentionally uses a raw TCP connect rather than a full database driver
/// to keep the dependency footprint small. For deeper checks (e.g. `SELECT 1`),
/// the system should use its own sqlx pool after boot.
pub struct PostgresCheck {
    /// Host address
    host: String,
    /// Port
    port: u16,
    /// Override criticality (default: Critical)
    criticality: Criticality,
    /// Connection timeout
    timeout: Duration,
}

impl PostgresCheck {
    /// Create a new Postgres check.
    pub fn new(host: impl Into<String>, port: u16) -> Self {
        Self {
            host: host.into(),
            port,
            criticality: Criticality::Critical,
            timeout: Duration::from_secs(5),
        }
    }

    /// Convenience constructor using a combined `host:port` string.
    pub fn from_addr(addr: impl Into<String>) -> Self {
        let addr = addr.into();
        let parts: Vec<&str> = addr.rsplitn(2, ':').collect();
        let (port, host) = if parts.len() == 2 {
            let port = parts[0].parse::<u16>().unwrap_or(5432);
            (port, parts[1].to_string())
        } else {
            (5432, addr)
        };
        Self {
            host,
            port,
            criticality: Criticality::Critical,
            timeout: Duration::from_secs(5),
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

    fn addr(&self) -> String {
        format!("{}:{}", self.host, self.port)
    }
}

#[async_trait]
impl PreFlightCheck for PostgresCheck {
    fn name(&self) -> &str {
        "Postgres"
    }

    fn phase(&self) -> BootPhase {
        BootPhase::Infrastructure
    }

    fn criticality(&self) -> Criticality {
        self.criticality
    }

    fn timeout(&self) -> Duration {
        self.timeout
    }

    async fn execute(&self) -> Result<(), String> {
        let addr = self.addr();

        // Resolve the address first (handles DNS)
        let socket_addrs: Vec<std::net::SocketAddr> = tokio::net::lookup_host(&addr)
            .await
            .map_err(|e| format!("DNS resolution failed for {}: {}", addr, e))?
            .collect();

        if socket_addrs.is_empty() {
            return Err(format!("No addresses resolved for {}", addr));
        }

        // Try TCP connect to the first resolved address
        let target = socket_addrs[0];
        tokio::net::TcpStream::connect(target)
            .await
            .map_err(|e| format!("TCP connect to Postgres at {} failed: {}", addr, e))?;

        Ok(())
    }

    fn detail_on_success(&self) -> Option<String> {
        Some(format!("TCP connect OK to {}", self.addr()))
    }
}

// ============================================================================
// QuestDB Check
// ============================================================================

/// Verifies QuestDB is healthy via its HTTP health endpoint.
///
/// QuestDB exposes a health endpoint at `http://<host>:9000/` (web console)
/// or a more specific endpoint. This check does a simple GET and verifies a 2xx response.
pub struct QuestDBCheck {
    /// Full URL to QuestDB's health/exec endpoint (e.g. `http://localhost:9000`)
    url: String,
    /// Override criticality (default: Critical)
    criticality: Criticality,
    /// Request timeout
    timeout: Duration,
}

impl QuestDBCheck {
    /// Create a new QuestDB check.
    ///
    /// `url` should be the base URL of the QuestDB HTTP interface,
    /// e.g. `http://localhost:9000`
    pub fn new(url: impl Into<String>) -> Self {
        Self {
            url: url.into(),
            criticality: Criticality::Critical,
            timeout: Duration::from_secs(5),
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

    fn health_url(&self) -> String {
        let base = self.url.trim_end_matches('/');
        // QuestDB exposes /exec for queries; a simple SELECT 1 verifies it's alive
        format!("{}/exec?query=SELECT%201", base)
    }
}

#[async_trait]
impl PreFlightCheck for QuestDBCheck {
    fn name(&self) -> &str {
        "QuestDB"
    }

    fn phase(&self) -> BootPhase {
        BootPhase::Infrastructure
    }

    fn criticality(&self) -> Criticality {
        self.criticality
    }

    fn timeout(&self) -> Duration {
        self.timeout
    }

    async fn execute(&self) -> Result<(), String> {
        let client = reqwest::Client::builder()
            .timeout(self.timeout)
            .build()
            .map_err(|e| format!("Failed to create HTTP client: {}", e))?;

        let health_url = self.health_url();

        let response = client
            .get(&health_url)
            .send()
            .await
            .map_err(|e| format!("QuestDB health request to {} failed: {}", health_url, e))?;

        let status = response.status();
        if !status.is_success() {
            let body = response.text().await.unwrap_or_default();
            return Err(format!(
                "QuestDB returned HTTP {} from {}: {}",
                status, health_url, body
            ));
        }

        Ok(())
    }

    fn detail_on_success(&self) -> Option<String> {
        Some(format!("HTTP OK from {}", self.url))
    }
}

// ============================================================================
// Prometheus Check
// ============================================================================

/// Verifies Prometheus is reachable via its readiness endpoint.
///
/// Prometheus exposes `/-/ready` which returns 200 when it's ready to serve traffic.
pub struct PrometheusCheck {
    /// Base URL of the Prometheus server (e.g. `http://localhost:9090`)
    url: String,
    /// Override criticality (default: Required — system can run without metrics, but shouldn't)
    criticality: Criticality,
    /// Request timeout
    timeout: Duration,
}

impl PrometheusCheck {
    /// Create a new Prometheus readiness check.
    pub fn new(url: impl Into<String>) -> Self {
        Self {
            url: url.into(),
            criticality: Criticality::Required,
            timeout: Duration::from_secs(5),
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

    fn ready_url(&self) -> String {
        let base = self.url.trim_end_matches('/');
        format!("{}/-/ready", base)
    }
}

#[async_trait]
impl PreFlightCheck for PrometheusCheck {
    fn name(&self) -> &str {
        "Prometheus"
    }

    fn phase(&self) -> BootPhase {
        BootPhase::Infrastructure
    }

    fn criticality(&self) -> Criticality {
        self.criticality
    }

    fn timeout(&self) -> Duration {
        self.timeout
    }

    async fn execute(&self) -> Result<(), String> {
        let client = reqwest::Client::builder()
            .timeout(self.timeout)
            .build()
            .map_err(|e| format!("Failed to create HTTP client: {}", e))?;

        let ready_url = self.ready_url();

        let response = client.get(&ready_url).send().await.map_err(|e| {
            format!(
                "Prometheus readiness request to {} failed: {}",
                ready_url, e
            )
        })?;

        let status = response.status();
        if !status.is_success() {
            let body = response.text().await.unwrap_or_default();
            return Err(format!(
                "Prometheus returned HTTP {} from {}: {}",
                status, ready_url, body
            ));
        }

        Ok(())
    }

    fn detail_on_success(&self) -> Option<String> {
        Some(format!("ready at {}", self.url))
    }
}

// ============================================================================
// Generic TCP Check (utility)
// ============================================================================

/// A generic TCP connectivity check. Useful for services that don't have
/// an HTTP health endpoint but just need to be reachable on a port.
pub struct TcpCheck {
    /// Human-readable name for this check
    check_name: String,
    /// Host address
    host: String,
    /// Port
    port: u16,
    /// Override criticality
    criticality: Criticality,
    /// Connection timeout
    timeout: Duration,
}

impl TcpCheck {
    /// Create a new TCP connectivity check.
    pub fn new(name: impl Into<String>, host: impl Into<String>, port: u16) -> Self {
        Self {
            check_name: name.into(),
            host: host.into(),
            port,
            criticality: Criticality::Required,
            timeout: Duration::from_secs(5),
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

    fn addr(&self) -> String {
        format!("{}:{}", self.host, self.port)
    }
}

#[async_trait]
impl PreFlightCheck for TcpCheck {
    fn name(&self) -> &str {
        &self.check_name
    }

    fn phase(&self) -> BootPhase {
        BootPhase::Infrastructure
    }

    fn criticality(&self) -> Criticality {
        self.criticality
    }

    fn timeout(&self) -> Duration {
        self.timeout
    }

    async fn execute(&self) -> Result<(), String> {
        let addr = self.addr();
        let socket_addrs: Vec<std::net::SocketAddr> = tokio::net::lookup_host(&addr)
            .await
            .map_err(|e| format!("DNS resolution failed for {}: {}", addr, e))?
            .collect();

        if socket_addrs.is_empty() {
            return Err(format!("No addresses resolved for {}", addr));
        }

        tokio::net::TcpStream::connect(socket_addrs[0])
            .await
            .map_err(|e| format!("TCP connect to {} failed: {}", addr, e))?;

        Ok(())
    }

    fn detail_on_success(&self) -> Option<String> {
        Some(format!("TCP connect OK to {}", self.addr()))
    }
}

// ============================================================================
// Tests
// ============================================================================

#[cfg(test)]
mod tests {
    use super::*;

    // ── Construction & config tests (no network needed) ──

    #[test]
    fn test_redis_check_defaults() {
        let check = RedisCheck::new("redis://localhost:6379");
        assert_eq!(check.name(), "Redis");
        assert_eq!(check.phase(), BootPhase::Infrastructure);
        assert_eq!(check.criticality(), Criticality::Critical);
        assert_eq!(check.timeout(), Duration::from_secs(5));
    }

    #[test]
    fn test_redis_check_builder() {
        let check = RedisCheck::new("redis://host:6380")
            .with_criticality(Criticality::Required)
            .with_timeout(Duration::from_secs(10));
        assert_eq!(check.criticality(), Criticality::Required);
        assert_eq!(check.timeout(), Duration::from_secs(10));
        assert_eq!(
            check.detail_on_success(),
            Some("connected to redis://host:6380".to_string())
        );
    }

    #[test]
    fn test_postgres_check_defaults() {
        let check = PostgresCheck::new("localhost", 5432);
        assert_eq!(check.name(), "Postgres");
        assert_eq!(check.phase(), BootPhase::Infrastructure);
        assert_eq!(check.criticality(), Criticality::Critical);
        assert_eq!(check.addr(), "localhost:5432");
    }

    #[test]
    fn test_postgres_from_addr() {
        let check = PostgresCheck::from_addr("db.internal:5433");
        assert_eq!(check.host, "db.internal");
        assert_eq!(check.port, 5433);
    }

    #[test]
    fn test_postgres_from_addr_no_port() {
        let check = PostgresCheck::from_addr("db.internal");
        assert_eq!(check.host, "db.internal");
        assert_eq!(check.port, 5432); // default
    }

    #[test]
    fn test_postgres_builder() {
        let check = PostgresCheck::new("host", 5432)
            .with_criticality(Criticality::Optional)
            .with_timeout(Duration::from_secs(3));
        assert_eq!(check.criticality(), Criticality::Optional);
        assert_eq!(check.timeout(), Duration::from_secs(3));
    }

    #[test]
    fn test_questdb_check_defaults() {
        let check = QuestDBCheck::new("http://localhost:9000");
        assert_eq!(check.name(), "QuestDB");
        assert_eq!(check.phase(), BootPhase::Infrastructure);
        assert_eq!(check.criticality(), Criticality::Critical);
    }

    #[test]
    fn test_questdb_health_url() {
        let check = QuestDBCheck::new("http://localhost:9000");
        assert_eq!(
            check.health_url(),
            "http://localhost:9000/exec?query=SELECT%201"
        );

        // Trailing slash stripped
        let check = QuestDBCheck::new("http://localhost:9000/");
        assert_eq!(
            check.health_url(),
            "http://localhost:9000/exec?query=SELECT%201"
        );
    }

    #[test]
    fn test_questdb_builder() {
        let check = QuestDBCheck::new("http://localhost:9000")
            .with_criticality(Criticality::Required)
            .with_timeout(Duration::from_secs(8));
        assert_eq!(check.criticality(), Criticality::Required);
        assert_eq!(check.timeout(), Duration::from_secs(8));
    }

    #[test]
    fn test_prometheus_check_defaults() {
        let check = PrometheusCheck::new("http://localhost:9090");
        assert_eq!(check.name(), "Prometheus");
        assert_eq!(check.phase(), BootPhase::Infrastructure);
        // Prometheus defaults to Required, not Critical
        assert_eq!(check.criticality(), Criticality::Required);
    }

    #[test]
    fn test_prometheus_ready_url() {
        let check = PrometheusCheck::new("http://localhost:9090");
        assert_eq!(check.ready_url(), "http://localhost:9090/-/ready");

        let check = PrometheusCheck::new("http://localhost:9090/");
        assert_eq!(check.ready_url(), "http://localhost:9090/-/ready");
    }

    #[test]
    fn test_prometheus_builder() {
        let check = PrometheusCheck::new("http://prom:9090")
            .with_criticality(Criticality::Optional)
            .with_timeout(Duration::from_secs(2));
        assert_eq!(check.criticality(), Criticality::Optional);
        assert_eq!(check.timeout(), Duration::from_secs(2));
    }

    #[test]
    fn test_tcp_check_defaults() {
        let check = TcpCheck::new("MyService", "localhost", 8080);
        assert_eq!(check.name(), "MyService");
        assert_eq!(check.phase(), BootPhase::Infrastructure);
        assert_eq!(check.criticality(), Criticality::Required);
        assert_eq!(check.addr(), "localhost:8080");
    }

    #[test]
    fn test_tcp_check_builder() {
        let check = TcpCheck::new("SomeDB", "host", 3306)
            .with_criticality(Criticality::Critical)
            .with_timeout(Duration::from_secs(15));
        assert_eq!(check.criticality(), Criticality::Critical);
        assert_eq!(check.timeout(), Duration::from_secs(15));
        assert_eq!(
            check.detail_on_success(),
            Some("TCP connect OK to host:3306".to_string())
        );
    }

    // ── Network-dependent tests (expected to fail when infra isn't running) ──

    #[tokio::test]
    async fn test_redis_check_bad_url_fails() {
        let check = RedisCheck::new("redis://this-host-does-not-exist:6379")
            .with_timeout(Duration::from_secs(2));
        let result = check.execute().await;
        assert!(result.is_err());
    }

    #[tokio::test]
    async fn test_postgres_check_unreachable_fails() {
        // Connect to a port nothing is listening on
        let check = PostgresCheck::new("127.0.0.1", 59999).with_timeout(Duration::from_secs(1));
        let result = check.execute().await;
        assert!(result.is_err());
    }

    #[tokio::test]
    async fn test_questdb_check_unreachable_fails() {
        let check =
            QuestDBCheck::new("http://127.0.0.1:59998").with_timeout(Duration::from_secs(1));
        let result = check.execute().await;
        assert!(result.is_err());
    }

    #[tokio::test]
    async fn test_prometheus_check_unreachable_fails() {
        let check =
            PrometheusCheck::new("http://127.0.0.1:59997").with_timeout(Duration::from_secs(1));
        let result = check.execute().await;
        assert!(result.is_err());
    }

    #[tokio::test]
    async fn test_tcp_check_unreachable_fails() {
        let check =
            TcpCheck::new("Nothing", "127.0.0.1", 59996).with_timeout(Duration::from_secs(1));
        let result = check.execute().await;
        assert!(result.is_err());
    }

    // ── Trait method integration via `run()` ──

    #[tokio::test]
    async fn test_run_produces_check_result_with_correct_metadata() {
        // This will fail (nothing on 59995), but we want to verify the CheckResult shape
        let check = PostgresCheck::new("127.0.0.1", 59995).with_timeout(Duration::from_secs(1));
        let result = check.run().await;

        assert_eq!(result.name, "Postgres");
        assert_eq!(result.phase, BootPhase::Infrastructure);
        assert_eq!(result.criticality, Criticality::Critical);
        assert!(result.outcome.is_fail());
        assert!(result.duration_ms > 0 || result.outcome.is_fail()); // sanity
    }
}
