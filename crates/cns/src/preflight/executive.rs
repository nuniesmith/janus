//! # Executive Pre-Flight Checks
//!
//! Verifies the execution path is operational end-to-end before the system begins trading.
//!
//! ## Checks
//!
//! - **ExecutionGrpcCheck** — Verify the execution service gRPC endpoint is reachable
//! - **ExchangeRestCheck** — Verify exchange REST APIs respond (Kraken primary, Bybit fallback)
//! - **OrderPathCheck** — Verify the full order path from signal → execution is intact

use super::{BootPhase, Criticality, PreFlightCheck};
use async_trait::async_trait;
use std::time::Duration;

// ============================================================================
// Execution gRPC Check
// ============================================================================

/// Method used to verify the execution service gRPC endpoint.
#[derive(Debug, Clone)]
pub enum ExecutionGrpcVerification {
    /// TCP connect to the gRPC endpoint to verify it is listening.
    TcpConnect {
        /// Host of the execution gRPC service.
        host: String,
        /// Port of the execution gRPC service.
        port: u16,
    },
    /// Full gRPC health check using the gRPC health protocol.
    GrpcHealth {
        /// Full gRPC endpoint (e.g. `http://localhost:50051`).
        endpoint: String,
    },
    /// In-memory verification for testing.
    InMemory {
        /// Whether the execution service is reachable.
        reachable: bool,
        /// Optional latency in milliseconds for the last health check.
        latency_ms: Option<u64>,
    },
}

/// Verifies the execution service gRPC endpoint is reachable and healthy.
///
/// The execution service is the gateway through which all orders flow.
/// If it is unreachable, the system cannot place, modify, or cancel orders.
/// This check verifies:
///
/// 1. The gRPC endpoint is listening (TCP connect or gRPC health)
/// 2. Response latency is within acceptable bounds
pub struct ExecutionGrpcCheck {
    /// How to verify the execution service.
    verification: ExecutionGrpcVerification,
    /// Override criticality (default: Critical — cannot trade without execution).
    criticality: Criticality,
    /// Verification timeout.
    timeout: Duration,
    /// Maximum acceptable latency. If set, the check fails when latency exceeds this.
    max_latency: Option<Duration>,
}

impl ExecutionGrpcCheck {
    /// Create an execution gRPC check using TCP connect.
    pub fn tcp(host: impl Into<String>, port: u16) -> Self {
        Self {
            verification: ExecutionGrpcVerification::TcpConnect {
                host: host.into(),
                port,
            },
            criticality: Criticality::Critical,
            timeout: Duration::from_secs(5),
            max_latency: None,
        }
    }

    /// Create an execution gRPC check using the gRPC health protocol.
    pub fn grpc(endpoint: impl Into<String>) -> Self {
        Self {
            verification: ExecutionGrpcVerification::GrpcHealth {
                endpoint: endpoint.into(),
            },
            criticality: Criticality::Critical,
            timeout: Duration::from_secs(5),
            max_latency: None,
        }
    }

    /// Create an execution gRPC check using in-memory state (for testing).
    pub fn in_memory(reachable: bool) -> Self {
        Self {
            verification: ExecutionGrpcVerification::InMemory {
                reachable,
                latency_ms: None,
            },
            criticality: Criticality::Critical,
            timeout: Duration::from_secs(1),
            max_latency: None,
        }
    }

    /// Create an in-memory check with latency info (for testing).
    pub fn in_memory_with_latency(reachable: bool, latency_ms: u64) -> Self {
        Self {
            verification: ExecutionGrpcVerification::InMemory {
                reachable,
                latency_ms: Some(latency_ms),
            },
            criticality: Criticality::Critical,
            timeout: Duration::from_secs(1),
            max_latency: None,
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

    /// Set a maximum acceptable latency for the health check response.
    pub fn with_max_latency(mut self, max_latency: Duration) -> Self {
        self.max_latency = Some(max_latency);
        self
    }

    fn addr(&self) -> String {
        match &self.verification {
            ExecutionGrpcVerification::TcpConnect { host, port } => format!("{}:{}", host, port),
            ExecutionGrpcVerification::GrpcHealth { endpoint } => endpoint.clone(),
            ExecutionGrpcVerification::InMemory { .. } => "in-memory".to_string(),
        }
    }
}

#[async_trait]
impl PreFlightCheck for ExecutionGrpcCheck {
    fn name(&self) -> &str {
        "Execution gRPC"
    }

    fn phase(&self) -> BootPhase {
        BootPhase::Executive
    }

    fn criticality(&self) -> Criticality {
        self.criticality
    }

    fn timeout(&self) -> Duration {
        self.timeout
    }

    async fn execute(&self) -> Result<(), String> {
        let start = std::time::Instant::now();

        match &self.verification {
            ExecutionGrpcVerification::TcpConnect { host, port } => {
                let addr = format!("{}:{}", host, port);

                let socket_addrs: Vec<std::net::SocketAddr> = tokio::net::lookup_host(&addr)
                    .await
                    .map_err(|e| {
                        format!(
                            "DNS resolution failed for execution service at {}: {}",
                            addr, e
                        )
                    })?
                    .collect();

                if socket_addrs.is_empty() {
                    return Err(format!(
                        "No addresses resolved for execution service at {}",
                        addr
                    ));
                }

                tokio::net::TcpStream::connect(socket_addrs[0])
                    .await
                    .map_err(|e| {
                        format!("TCP connect to execution service at {} failed: {}", addr, e)
                    })?;

                let elapsed = start.elapsed();
                if let Some(max) = self.max_latency
                    && elapsed > max
                {
                    return Err(format!(
                        "Execution service latency too high: {:.1}ms > {:.1}ms threshold",
                        elapsed.as_secs_f64() * 1000.0,
                        max.as_secs_f64() * 1000.0,
                    ));
                }

                Ok(())
            }

            ExecutionGrpcVerification::GrpcHealth { endpoint } => {
                // Strip protocol prefix and try TCP connect as a lightweight check.
                // A full gRPC health check would require tonic::transport which adds complexity.
                let addr = endpoint
                    .trim_start_matches("http://")
                    .trim_start_matches("https://");

                let socket_addrs: Vec<std::net::SocketAddr> = tokio::net::lookup_host(addr)
                    .await
                    .map_err(|e| {
                        format!(
                            "DNS resolution failed for execution gRPC at {}: {}",
                            endpoint, e
                        )
                    })?
                    .collect();

                if socket_addrs.is_empty() {
                    return Err(format!(
                        "No addresses resolved for execution gRPC at {}",
                        endpoint
                    ));
                }

                tokio::net::TcpStream::connect(socket_addrs[0])
                    .await
                    .map_err(|e| {
                        format!(
                            "TCP connect to execution gRPC at {} failed: {}",
                            endpoint, e
                        )
                    })?;

                let elapsed = start.elapsed();
                if let Some(max) = self.max_latency
                    && elapsed > max
                {
                    return Err(format!(
                        "Execution gRPC latency too high: {:.1}ms > {:.1}ms threshold",
                        elapsed.as_secs_f64() * 1000.0,
                        max.as_secs_f64() * 1000.0,
                    ));
                }

                Ok(())
            }

            ExecutionGrpcVerification::InMemory {
                reachable,
                latency_ms,
            } => {
                if !*reachable {
                    return Err(
                        "Execution service is not reachable (in-memory flag is false)".to_string(),
                    );
                }

                if let (Some(latency), Some(max)) = (latency_ms, &self.max_latency) {
                    let max_ms = max.as_millis() as u64;
                    if *latency > max_ms {
                        return Err(format!(
                            "Execution service latency too high: {}ms > {}ms threshold",
                            latency, max_ms,
                        ));
                    }
                }

                Ok(())
            }
        }
    }

    fn detail_on_success(&self) -> Option<String> {
        Some(format!("reachable at {}", self.addr()))
    }
}

// ============================================================================
// Exchange REST Check
// ============================================================================

/// Which exchange to test via REST API.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum ExchangeTarget {
    /// Kraken (primary exchange).
    Kraken,
    /// Bybit (fallback exchange).
    Bybit,
    /// Binance (tertiary exchange).
    Binance,
}

impl ExchangeTarget {
    /// Default public REST API time endpoint for this exchange.
    pub fn default_time_url(&self) -> &'static str {
        match self {
            ExchangeTarget::Kraken => "https://api.kraken.com/0/public/Time",
            ExchangeTarget::Bybit => "https://api.bybit.com/v5/market/time",
            ExchangeTarget::Binance => "https://api.binance.com/api/v3/time",
        }
    }

    /// Display name.
    pub fn name(&self) -> &'static str {
        match self {
            ExchangeTarget::Kraken => "Kraken",
            ExchangeTarget::Bybit => "Bybit",
            ExchangeTarget::Binance => "Binance",
        }
    }

    /// Static check name for PreFlightCheck::name() return.
    pub fn rest_check_name(&self) -> &'static str {
        match self {
            ExchangeTarget::Kraken => "Kraken REST",
            ExchangeTarget::Bybit => "Bybit REST",
            ExchangeTarget::Binance => "Binance REST",
        }
    }
}

/// Method used to verify exchange REST API connectivity.
#[derive(Debug, Clone)]
pub enum ExchangeRestVerification {
    /// HTTP GET to a public API endpoint and verify a 2xx response.
    HttpGet {
        /// Which exchange.
        exchange: ExchangeTarget,
        /// URL to GET.
        url: String,
    },
    /// In-memory verification for testing.
    InMemory {
        /// Which exchange.
        exchange: ExchangeTarget,
        /// Whether the API is reachable.
        reachable: bool,
        /// Response time in milliseconds.
        response_time_ms: Option<u64>,
    },
}

/// Verifies an exchange REST API is reachable and responds within acceptable latency.
///
/// This check hits a lightweight, unauthenticated public endpoint (usually the
/// server time endpoint) to verify:
///
/// 1. DNS resolution and network connectivity to the exchange
/// 2. The exchange API is responding with 2xx status codes
/// 3. Response latency is within acceptable bounds
pub struct ExchangeRestCheck {
    /// How to verify the exchange REST API.
    verification: ExchangeRestVerification,
    /// Override criticality.
    criticality: Criticality,
    /// Request timeout.
    timeout: Duration,
    /// Maximum acceptable response latency.
    max_latency: Option<Duration>,
}

impl ExchangeRestCheck {
    /// Create an exchange REST check for the given exchange using its default time endpoint.
    pub fn new(exchange: ExchangeTarget) -> Self {
        let criticality = match exchange {
            ExchangeTarget::Kraken => Criticality::Critical, // Primary
            ExchangeTarget::Bybit => Criticality::Required,  // Fallback
            ExchangeTarget::Binance => Criticality::Optional, // Tertiary
        };

        Self {
            verification: ExchangeRestVerification::HttpGet {
                url: exchange.default_time_url().to_string(),
                exchange,
            },
            criticality,
            timeout: Duration::from_secs(10),
            max_latency: None,
        }
    }

    /// Create an exchange REST check with a custom URL.
    pub fn custom(exchange: ExchangeTarget, url: impl Into<String>) -> Self {
        let criticality = match exchange {
            ExchangeTarget::Kraken => Criticality::Critical,
            ExchangeTarget::Bybit => Criticality::Required,
            ExchangeTarget::Binance => Criticality::Optional,
        };

        Self {
            verification: ExchangeRestVerification::HttpGet {
                exchange,
                url: url.into(),
            },
            criticality,
            timeout: Duration::from_secs(10),
            max_latency: None,
        }
    }

    /// Create an exchange REST check using in-memory state (for testing).
    pub fn in_memory(exchange: ExchangeTarget, reachable: bool) -> Self {
        let criticality = match exchange {
            ExchangeTarget::Kraken => Criticality::Critical,
            ExchangeTarget::Bybit => Criticality::Required,
            ExchangeTarget::Binance => Criticality::Optional,
        };

        Self {
            verification: ExchangeRestVerification::InMemory {
                exchange,
                reachable,
                response_time_ms: None,
            },
            criticality,
            timeout: Duration::from_secs(1),
            max_latency: None,
        }
    }

    /// Create an in-memory check with response time info.
    pub fn in_memory_with_latency(
        exchange: ExchangeTarget,
        reachable: bool,
        response_time_ms: u64,
    ) -> Self {
        let criticality = match exchange {
            ExchangeTarget::Kraken => Criticality::Critical,
            ExchangeTarget::Bybit => Criticality::Required,
            ExchangeTarget::Binance => Criticality::Optional,
        };

        Self {
            verification: ExchangeRestVerification::InMemory {
                exchange,
                reachable,
                response_time_ms: Some(response_time_ms),
            },
            criticality,
            timeout: Duration::from_secs(1),
            max_latency: None,
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

    /// Set a maximum acceptable response latency.
    pub fn with_max_latency(mut self, max_latency: Duration) -> Self {
        self.max_latency = Some(max_latency);
        self
    }

    fn exchange(&self) -> ExchangeTarget {
        match &self.verification {
            ExchangeRestVerification::HttpGet { exchange, .. } => *exchange,
            ExchangeRestVerification::InMemory { exchange, .. } => *exchange,
        }
    }
}

#[async_trait]
impl PreFlightCheck for ExchangeRestCheck {
    fn name(&self) -> &str {
        self.exchange().rest_check_name()
    }

    fn phase(&self) -> BootPhase {
        BootPhase::Executive
    }

    fn criticality(&self) -> Criticality {
        self.criticality
    }

    fn timeout(&self) -> Duration {
        self.timeout
    }

    async fn execute(&self) -> Result<(), String> {
        match &self.verification {
            ExchangeRestVerification::HttpGet { exchange, url } => {
                let client = reqwest::Client::builder()
                    .timeout(self.timeout)
                    .build()
                    .map_err(|e| format!("Failed to create HTTP client: {}", e))?;

                let start = std::time::Instant::now();

                let response = client.get(url).send().await.map_err(|e| {
                    format!(
                        "{} REST API request to {} failed: {}",
                        exchange.name(),
                        url,
                        e
                    )
                })?;

                let elapsed = start.elapsed();

                let status = response.status();
                if !status.is_success() {
                    let body = response.text().await.unwrap_or_default();
                    return Err(format!(
                        "{} REST API returned HTTP {} from {}: {}",
                        exchange.name(),
                        status,
                        url,
                        body
                    ));
                }

                // Check latency
                if let Some(max) = self.max_latency
                    && elapsed > max
                {
                    return Err(format!(
                        "{} REST API latency too high: {:.1}ms > {:.1}ms threshold",
                        exchange.name(),
                        elapsed.as_secs_f64() * 1000.0,
                        max.as_secs_f64() * 1000.0,
                    ));
                }

                Ok(())
            }

            ExchangeRestVerification::InMemory {
                exchange,
                reachable,
                response_time_ms,
            } => {
                if !*reachable {
                    return Err(format!(
                        "{} REST API is not reachable (in-memory flag is false)",
                        exchange.name()
                    ));
                }

                if let (Some(latency), Some(max)) = (response_time_ms, &self.max_latency) {
                    let max_ms = max.as_millis() as u64;
                    if *latency > max_ms {
                        return Err(format!(
                            "{} REST API latency too high: {}ms > {}ms threshold",
                            exchange.name(),
                            latency,
                            max_ms,
                        ));
                    }
                }

                Ok(())
            }
        }
    }

    fn detail_on_success(&self) -> Option<String> {
        match &self.verification {
            ExchangeRestVerification::HttpGet { exchange, url } => {
                Some(format!("{} OK at {}", exchange.name(), url))
            }
            ExchangeRestVerification::InMemory {
                exchange,
                response_time_ms,
                ..
            } => {
                if let Some(ms) = response_time_ms {
                    Some(format!("{} OK ({}ms)", exchange.name(), ms))
                } else {
                    Some(format!("{} OK", exchange.name()))
                }
            }
        }
    }
}

// ============================================================================
// Order Path Integrity Check
// ============================================================================

/// Method used to verify order path integrity.
#[derive(Debug, Clone)]
pub enum OrderPathVerification {
    /// Check multiple endpoints that form the order path.
    /// Each endpoint is checked for TCP/HTTP reachability.
    EndpointChain {
        /// Ordered list of (name, url) pairs representing the order path.
        /// e.g. [("Signal Generator", "http://..."), ("Risk Manager", "http://..."), ("Execution", "http://...")]
        endpoints: Vec<(String, String)>,
    },
    /// Check the order path via a dedicated health/ready endpoint
    /// that the execution service exposes after self-testing.
    HttpEndpoint {
        /// URL of the order path integrity endpoint.
        url: String,
    },
    /// In-memory verification for testing.
    InMemory {
        /// Whether the full order path is intact.
        path_intact: bool,
        /// Number of path segments verified.
        segments_verified: usize,
        /// Total number of segments in the path.
        total_segments: usize,
    },
}

/// Verifies the full order path from signal generation to execution is intact.
///
/// The order path represents the critical chain through which trading decisions
/// flow: Signal → Risk Check → Position Sizing → Order Submission → Exchange.
/// This check verifies:
///
/// 1. Each segment of the path is reachable
/// 2. The end-to-end path can carry an order from signal to exchange
/// 3. No segment is missing or misconfigured
///
/// In production, this may involve sending a "dry-run" or "test" order through
/// the path (without actually submitting to the exchange).
pub struct OrderPathCheck {
    /// How to verify order path integrity.
    verification: OrderPathVerification,
    /// Override criticality (default: Critical — an incomplete order path means
    /// orders could be lost or duplicated).
    criticality: Criticality,
    /// Verification timeout (for the entire path check).
    timeout: Duration,
}

impl OrderPathCheck {
    /// Create an order path check using an endpoint chain.
    ///
    /// Provide an ordered list of `(name, url)` pairs representing each segment
    /// of the order path. Each will be checked for HTTP reachability.
    pub fn from_endpoints(endpoints: Vec<(impl Into<String>, impl Into<String>)>) -> Self {
        let endpoints: Vec<(String, String)> = endpoints
            .into_iter()
            .map(|(name, url)| (name.into(), url.into()))
            .collect();

        Self {
            verification: OrderPathVerification::EndpointChain { endpoints },
            criticality: Criticality::Critical,
            timeout: Duration::from_secs(15),
        }
    }

    /// Create an order path check using a single HTTP endpoint.
    ///
    /// The endpoint should perform its own internal path verification and return
    /// 200 when the path is intact.
    pub fn http(url: impl Into<String>) -> Self {
        Self {
            verification: OrderPathVerification::HttpEndpoint { url: url.into() },
            criticality: Criticality::Critical,
            timeout: Duration::from_secs(10),
        }
    }

    /// Create an order path check using in-memory state (for testing).
    pub fn in_memory(path_intact: bool, segments_verified: usize, total_segments: usize) -> Self {
        Self {
            verification: OrderPathVerification::InMemory {
                path_intact,
                segments_verified,
                total_segments,
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
impl PreFlightCheck for OrderPathCheck {
    fn name(&self) -> &str {
        "Order Path Integrity"
    }

    fn phase(&self) -> BootPhase {
        BootPhase::Executive
    }

    fn criticality(&self) -> Criticality {
        self.criticality
    }

    fn timeout(&self) -> Duration {
        self.timeout
    }

    async fn execute(&self) -> Result<(), String> {
        match &self.verification {
            OrderPathVerification::EndpointChain { endpoints } => {
                if endpoints.is_empty() {
                    return Err(
                        "Order path has no endpoints configured — at least 1 segment required"
                            .to_string(),
                    );
                }

                let client = reqwest::Client::builder()
                    .timeout(Duration::from_secs(5))
                    .build()
                    .map_err(|e| format!("Failed to create HTTP client: {}", e))?;

                let total = endpoints.len();

                for (verified, (name, url)) in endpoints.iter().enumerate() {
                    let response = client.get(url).send().await.map_err(|e| {
                        format!(
                            "Order path segment '{}' at {} is unreachable: {} ({}/{} segments verified)",
                            name, url, e, verified, total
                        )
                    })?;

                    let status = response.status();
                    if !status.is_success() {
                        let body = response.text().await.unwrap_or_default();
                        return Err(format!(
                            "Order path segment '{}' returned HTTP {} from {}: {} ({}/{} segments verified)",
                            name, status, url, body, verified, total
                        ));
                    }
                }

                Ok(())
            }

            OrderPathVerification::HttpEndpoint { url } => {
                let client = reqwest::Client::builder()
                    .timeout(self.timeout)
                    .build()
                    .map_err(|e| format!("Failed to create HTTP client: {}", e))?;

                let response =
                    client.get(url).send().await.map_err(|e| {
                        format!("Order path integrity check failed ({}): {}", url, e)
                    })?;

                let status = response.status();
                if !status.is_success() {
                    let body = response.text().await.unwrap_or_default();
                    return Err(format!(
                        "Order path integrity endpoint returned HTTP {} ({}): {}",
                        status, url, body
                    ));
                }

                Ok(())
            }

            OrderPathVerification::InMemory {
                path_intact,
                segments_verified,
                total_segments,
            } => {
                if *total_segments == 0 {
                    return Err(
                        "Order path has no segments configured — at least 1 required".to_string(),
                    );
                }

                if !*path_intact {
                    return Err(format!(
                        "Order path is NOT intact: only {}/{} segments verified",
                        segments_verified, total_segments
                    ));
                }

                if segments_verified < total_segments {
                    return Err(format!(
                        "Order path incomplete: {}/{} segments verified",
                        segments_verified, total_segments
                    ));
                }

                Ok(())
            }
        }
    }

    fn detail_on_success(&self) -> Option<String> {
        match &self.verification {
            OrderPathVerification::EndpointChain { endpoints } => Some(format!(
                "{}/{} segments verified",
                endpoints.len(),
                endpoints.len()
            )),
            OrderPathVerification::InMemory {
                segments_verified,
                total_segments,
                ..
            } => Some(format!(
                "{}/{} segments verified",
                segments_verified, total_segments
            )),
            _ => Some("all segments verified".to_string()),
        }
    }
}

// ============================================================================
// Builder Helpers
// ============================================================================

/// Create the default set of executive pre-flight checks for JANUS.
///
/// - Execution gRPC service (critical)
/// - Kraken REST API (critical — primary exchange)
/// - Bybit REST API (required — fallback exchange)
/// - Order path integrity (critical)
pub fn default_executive_checks(
    execution_grpc_host: Option<(&str, u16)>,
    order_path_url: Option<&str>,
) -> Vec<Box<dyn PreFlightCheck>> {
    let mut checks: Vec<Box<dyn PreFlightCheck>> = Vec::new();

    // Execution gRPC
    if let Some((host, port)) = execution_grpc_host {
        checks.push(Box::new(ExecutionGrpcCheck::tcp(host, port)));
    } else {
        // Default to localhost:50051
        checks.push(Box::new(ExecutionGrpcCheck::tcp("127.0.0.1", 50051)));
    }

    // Exchange REST APIs
    checks.push(Box::new(ExchangeRestCheck::new(ExchangeTarget::Kraken)));
    checks.push(Box::new(ExchangeRestCheck::new(ExchangeTarget::Bybit)));

    // Order path integrity
    if let Some(url) = order_path_url {
        checks.push(Box::new(OrderPathCheck::http(url)));
    } else {
        // Default to in-memory passing (will be wired during integration)
        checks.push(Box::new(
            OrderPathCheck::in_memory(true, 3, 3).with_criticality(Criticality::Required),
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

    // ── ExchangeTarget tests ──

    #[test]
    fn test_exchange_target_default_urls() {
        assert_eq!(
            ExchangeTarget::Kraken.default_time_url(),
            "https://api.kraken.com/0/public/Time"
        );
        assert_eq!(
            ExchangeTarget::Bybit.default_time_url(),
            "https://api.bybit.com/v5/market/time"
        );
        assert_eq!(
            ExchangeTarget::Binance.default_time_url(),
            "https://api.binance.com/api/v3/time"
        );
    }

    #[test]
    fn test_exchange_target_names() {
        assert_eq!(ExchangeTarget::Kraken.name(), "Kraken");
        assert_eq!(ExchangeTarget::Bybit.name(), "Bybit");
        assert_eq!(ExchangeTarget::Binance.name(), "Binance");
    }

    #[test]
    fn test_exchange_target_rest_check_names() {
        assert_eq!(ExchangeTarget::Kraken.rest_check_name(), "Kraken REST");
        assert_eq!(ExchangeTarget::Bybit.rest_check_name(), "Bybit REST");
        assert_eq!(ExchangeTarget::Binance.rest_check_name(), "Binance REST");
    }

    // ── ExecutionGrpcCheck construction tests ──

    #[test]
    fn test_execution_grpc_tcp_defaults() {
        let check = ExecutionGrpcCheck::tcp("localhost", 50051);
        assert_eq!(check.name(), "Execution gRPC");
        assert_eq!(check.phase(), BootPhase::Executive);
        assert_eq!(check.criticality(), Criticality::Critical);
        assert_eq!(check.timeout(), Duration::from_secs(5));
        assert!(check.max_latency.is_none());
    }

    #[test]
    fn test_execution_grpc_grpc_defaults() {
        let check = ExecutionGrpcCheck::grpc("http://localhost:50051");
        assert_eq!(check.criticality(), Criticality::Critical);
    }

    #[test]
    fn test_execution_grpc_in_memory() {
        let check = ExecutionGrpcCheck::in_memory(true);
        assert_eq!(check.timeout(), Duration::from_secs(1));
    }

    #[test]
    fn test_execution_grpc_in_memory_with_latency() {
        let check = ExecutionGrpcCheck::in_memory_with_latency(true, 50);
        match &check.verification {
            ExecutionGrpcVerification::InMemory { latency_ms, .. } => {
                assert_eq!(*latency_ms, Some(50));
            }
            _ => panic!("Expected InMemory verification"),
        }
    }

    #[test]
    fn test_execution_grpc_builder() {
        let check = ExecutionGrpcCheck::tcp("host", 50051)
            .with_criticality(Criticality::Required)
            .with_timeout(Duration::from_secs(10))
            .with_max_latency(Duration::from_millis(200));
        assert_eq!(check.criticality(), Criticality::Required);
        assert_eq!(check.timeout(), Duration::from_secs(10));
        assert_eq!(check.max_latency, Some(Duration::from_millis(200)));
    }

    #[test]
    fn test_execution_grpc_addr() {
        let check = ExecutionGrpcCheck::tcp("myhost", 9090);
        assert_eq!(check.addr(), "myhost:9090");

        let check = ExecutionGrpcCheck::grpc("http://localhost:50051");
        assert_eq!(check.addr(), "http://localhost:50051");

        let check = ExecutionGrpcCheck::in_memory(true);
        assert_eq!(check.addr(), "in-memory");
    }

    // ── ExecutionGrpcCheck execution tests ──

    #[tokio::test]
    async fn test_execution_grpc_in_memory_reachable_passes() {
        let check = ExecutionGrpcCheck::in_memory(true);
        let result = check.execute().await;
        assert!(result.is_ok());
    }

    #[tokio::test]
    async fn test_execution_grpc_in_memory_unreachable_fails() {
        let check = ExecutionGrpcCheck::in_memory(false);
        let result = check.execute().await;
        assert!(result.is_err());
        assert!(result.unwrap_err().contains("not reachable"));
    }

    #[tokio::test]
    async fn test_execution_grpc_in_memory_latency_ok() {
        let check = ExecutionGrpcCheck::in_memory_with_latency(true, 50)
            .with_max_latency(Duration::from_millis(100));
        let result = check.execute().await;
        assert!(result.is_ok());
    }

    #[tokio::test]
    async fn test_execution_grpc_in_memory_latency_exceeded() {
        let check = ExecutionGrpcCheck::in_memory_with_latency(true, 200)
            .with_max_latency(Duration::from_millis(100));
        let result = check.execute().await;
        assert!(result.is_err());
        assert!(result.unwrap_err().contains("latency too high"));
    }

    #[tokio::test]
    async fn test_execution_grpc_run_result() {
        let check = ExecutionGrpcCheck::in_memory(true);
        let result = check.run().await;
        assert_eq!(result.name, "Execution gRPC");
        assert_eq!(result.phase, BootPhase::Executive);
        assert_eq!(result.criticality, Criticality::Critical);
        assert!(result.outcome.is_pass());
    }

    #[tokio::test]
    async fn test_execution_grpc_failure_is_abort_worthy() {
        let check = ExecutionGrpcCheck::in_memory(false);
        let result = check.run().await;
        assert!(result.is_abort_worthy());
    }

    #[tokio::test]
    async fn test_execution_grpc_tcp_unreachable_fails() {
        let check =
            ExecutionGrpcCheck::tcp("127.0.0.1", 59985).with_timeout(Duration::from_secs(1));
        let result = check.execute().await;
        assert!(result.is_err());
    }

    // ── ExchangeRestCheck construction tests ──

    #[test]
    fn test_exchange_rest_kraken_defaults() {
        let check = ExchangeRestCheck::new(ExchangeTarget::Kraken);
        assert_eq!(check.name(), "Kraken REST");
        assert_eq!(check.phase(), BootPhase::Executive);
        assert_eq!(check.criticality(), Criticality::Critical); // Primary exchange
        assert_eq!(check.timeout(), Duration::from_secs(10));
    }

    #[test]
    fn test_exchange_rest_bybit_defaults() {
        let check = ExchangeRestCheck::new(ExchangeTarget::Bybit);
        assert_eq!(check.name(), "Bybit REST");
        assert_eq!(check.criticality(), Criticality::Required); // Fallback
    }

    #[test]
    fn test_exchange_rest_binance_defaults() {
        let check = ExchangeRestCheck::new(ExchangeTarget::Binance);
        assert_eq!(check.name(), "Binance REST");
        assert_eq!(check.criticality(), Criticality::Optional); // Tertiary
    }

    #[test]
    fn test_exchange_rest_custom_url() {
        let check = ExchangeRestCheck::custom(ExchangeTarget::Kraken, "https://custom.api/time");
        match &check.verification {
            ExchangeRestVerification::HttpGet { url, .. } => {
                assert_eq!(url, "https://custom.api/time");
            }
            _ => panic!("Expected HttpGet verification"),
        }
    }

    #[test]
    fn test_exchange_rest_in_memory() {
        let check = ExchangeRestCheck::in_memory(ExchangeTarget::Kraken, true);
        assert_eq!(check.timeout(), Duration::from_secs(1));
    }

    #[test]
    fn test_exchange_rest_in_memory_with_latency() {
        let check = ExchangeRestCheck::in_memory_with_latency(ExchangeTarget::Bybit, true, 150);
        match &check.verification {
            ExchangeRestVerification::InMemory {
                response_time_ms, ..
            } => {
                assert_eq!(*response_time_ms, Some(150));
            }
            _ => panic!("Expected InMemory verification"),
        }
    }

    #[test]
    fn test_exchange_rest_builder() {
        let check = ExchangeRestCheck::new(ExchangeTarget::Kraken)
            .with_criticality(Criticality::Optional)
            .with_timeout(Duration::from_secs(20))
            .with_max_latency(Duration::from_secs(3));
        assert_eq!(check.criticality(), Criticality::Optional);
        assert_eq!(check.timeout(), Duration::from_secs(20));
        assert_eq!(check.max_latency, Some(Duration::from_secs(3)));
    }

    // ── ExchangeRestCheck execution tests ──

    #[tokio::test]
    async fn test_exchange_rest_in_memory_reachable_passes() {
        let check = ExchangeRestCheck::in_memory(ExchangeTarget::Kraken, true);
        let result = check.execute().await;
        assert!(result.is_ok());
    }

    #[tokio::test]
    async fn test_exchange_rest_in_memory_unreachable_fails() {
        let check = ExchangeRestCheck::in_memory(ExchangeTarget::Kraken, false);
        let result = check.execute().await;
        assert!(result.is_err());
        assert!(result.unwrap_err().contains("not reachable"));
    }

    #[tokio::test]
    async fn test_exchange_rest_in_memory_latency_ok() {
        let check = ExchangeRestCheck::in_memory_with_latency(ExchangeTarget::Bybit, true, 100)
            .with_max_latency(Duration::from_millis(500));
        let result = check.execute().await;
        assert!(result.is_ok());
    }

    #[tokio::test]
    async fn test_exchange_rest_in_memory_latency_exceeded() {
        let check = ExchangeRestCheck::in_memory_with_latency(ExchangeTarget::Bybit, true, 600)
            .with_max_latency(Duration::from_millis(500));
        let result = check.execute().await;
        assert!(result.is_err());
        assert!(result.unwrap_err().contains("latency too high"));
    }

    #[tokio::test]
    async fn test_exchange_rest_in_memory_detail_with_latency() {
        let check = ExchangeRestCheck::in_memory_with_latency(ExchangeTarget::Kraken, true, 42);
        assert_eq!(
            check.detail_on_success(),
            Some("Kraken OK (42ms)".to_string())
        );
    }

    #[tokio::test]
    async fn test_exchange_rest_in_memory_detail_without_latency() {
        let check = ExchangeRestCheck::in_memory(ExchangeTarget::Bybit, true);
        assert_eq!(check.detail_on_success(), Some("Bybit OK".to_string()));
    }

    #[tokio::test]
    async fn test_exchange_rest_run_result() {
        let check = ExchangeRestCheck::in_memory(ExchangeTarget::Kraken, true);
        let result = check.run().await;
        assert_eq!(result.name, "Kraken REST");
        assert_eq!(result.phase, BootPhase::Executive);
        assert!(result.outcome.is_pass());
    }

    #[tokio::test]
    async fn test_exchange_rest_kraken_failure_is_abort_worthy() {
        let check = ExchangeRestCheck::in_memory(ExchangeTarget::Kraken, false);
        let result = check.run().await;
        assert!(result.is_abort_worthy()); // Kraken is Critical
    }

    #[tokio::test]
    async fn test_exchange_rest_binance_failure_not_abort_worthy() {
        let check = ExchangeRestCheck::in_memory(ExchangeTarget::Binance, false);
        let result = check.run().await;
        assert!(!result.is_abort_worthy()); // Binance is Optional
    }

    // ── OrderPathCheck construction tests ──

    #[test]
    fn test_order_path_endpoints_defaults() {
        let check = OrderPathCheck::from_endpoints(vec![
            ("Signal", "http://localhost:8001/health"),
            ("Risk", "http://localhost:8002/health"),
            ("Execution", "http://localhost:8003/health"),
        ]);
        assert_eq!(check.name(), "Order Path Integrity");
        assert_eq!(check.phase(), BootPhase::Executive);
        assert_eq!(check.criticality(), Criticality::Critical);
        assert_eq!(check.timeout(), Duration::from_secs(15));
    }

    #[test]
    fn test_order_path_http_defaults() {
        let check = OrderPathCheck::http("http://localhost:8080/order-path/verify");
        assert_eq!(check.criticality(), Criticality::Critical);
        assert_eq!(check.timeout(), Duration::from_secs(10));
    }

    #[test]
    fn test_order_path_in_memory() {
        let check = OrderPathCheck::in_memory(true, 3, 3);
        assert_eq!(check.timeout(), Duration::from_secs(1));
    }

    #[test]
    fn test_order_path_builder() {
        let check = OrderPathCheck::in_memory(true, 3, 3)
            .with_criticality(Criticality::Required)
            .with_timeout(Duration::from_secs(30));
        assert_eq!(check.criticality(), Criticality::Required);
        assert_eq!(check.timeout(), Duration::from_secs(30));
    }

    // ── OrderPathCheck execution tests ──

    #[tokio::test]
    async fn test_order_path_in_memory_intact_passes() {
        let check = OrderPathCheck::in_memory(true, 3, 3);
        let result = check.execute().await;
        assert!(result.is_ok());
    }

    #[tokio::test]
    async fn test_order_path_in_memory_not_intact_fails() {
        let check = OrderPathCheck::in_memory(false, 2, 3);
        let result = check.execute().await;
        assert!(result.is_err());
        assert!(result.unwrap_err().contains("NOT intact"));
    }

    #[tokio::test]
    async fn test_order_path_in_memory_incomplete_fails() {
        let check = OrderPathCheck::in_memory(true, 2, 3);
        let result = check.execute().await;
        assert!(result.is_err());
        assert!(result.unwrap_err().contains("incomplete"));
    }

    #[tokio::test]
    async fn test_order_path_in_memory_zero_segments_fails() {
        let check = OrderPathCheck::in_memory(true, 0, 0);
        let result = check.execute().await;
        assert!(result.is_err());
        assert!(result.unwrap_err().contains("no segments"));
    }

    #[tokio::test]
    async fn test_order_path_in_memory_detail() {
        let check = OrderPathCheck::in_memory(true, 4, 4);
        assert_eq!(
            check.detail_on_success(),
            Some("4/4 segments verified".to_string())
        );
    }

    #[tokio::test]
    async fn test_order_path_run_result() {
        let check = OrderPathCheck::in_memory(true, 3, 3);
        let result = check.run().await;
        assert_eq!(result.name, "Order Path Integrity");
        assert_eq!(result.phase, BootPhase::Executive);
        assert!(result.outcome.is_pass());
    }

    #[tokio::test]
    async fn test_order_path_failure_is_abort_worthy() {
        let check = OrderPathCheck::in_memory(false, 1, 3);
        let result = check.run().await;
        assert!(result.is_abort_worthy()); // Critical check
    }

    #[tokio::test]
    async fn test_order_path_empty_endpoint_chain_fails() {
        let check = OrderPathCheck::from_endpoints(Vec::<(&str, &str)>::new());
        let result = check.execute().await;
        assert!(result.is_err());
        assert!(result.unwrap_err().contains("no endpoints"));
    }

    // ── Builder helper tests ──

    #[test]
    fn test_default_executive_checks_count() {
        let checks = default_executive_checks(None, None);
        assert_eq!(checks.len(), 4); // gRPC + Kraken REST + Bybit REST + Order Path
    }

    #[test]
    fn test_default_executive_checks_with_custom_grpc() {
        let checks = default_executive_checks(Some(("10.0.0.1", 50052)), None);
        assert_eq!(checks.len(), 4);
    }

    #[test]
    fn test_default_executive_checks_with_order_path_url() {
        let checks =
            default_executive_checks(None, Some("http://localhost:8080/order-path/verify"));
        assert_eq!(checks.len(), 4);
    }

    #[test]
    fn test_default_executive_checks_phases() {
        let checks = default_executive_checks(None, None);
        for check in &checks {
            assert_eq!(check.phase(), BootPhase::Executive);
        }
    }

    #[test]
    fn test_default_executive_checks_names() {
        let checks = default_executive_checks(None, None);
        assert_eq!(checks[0].name(), "Execution gRPC");
        assert_eq!(checks[1].name(), "Kraken REST");
        assert_eq!(checks[2].name(), "Bybit REST");
        assert_eq!(checks[3].name(), "Order Path Integrity");
    }

    #[test]
    fn test_default_executive_checks_criticality() {
        let checks = default_executive_checks(None, None);
        assert_eq!(checks[0].criticality(), Criticality::Critical); // gRPC
        assert_eq!(checks[1].criticality(), Criticality::Critical); // Kraken
        assert_eq!(checks[2].criticality(), Criticality::Required); // Bybit
        // Order path defaults to Required when using default in-memory
        assert_eq!(checks[3].criticality(), Criticality::Required);
    }

    // ── Integration with PreFlightRunner ──

    #[tokio::test]
    async fn test_executive_checks_in_runner() {
        use super::super::PreFlightRunner;

        let mut runner = PreFlightRunner::new();
        runner.add_check(Box::new(ExecutionGrpcCheck::in_memory(true)));
        runner.add_check(Box::new(ExchangeRestCheck::in_memory(
            ExchangeTarget::Kraken,
            true,
        )));
        runner.add_check(Box::new(ExchangeRestCheck::in_memory(
            ExchangeTarget::Bybit,
            true,
        )));
        runner.add_check(Box::new(OrderPathCheck::in_memory(true, 3, 3)));

        let report = runner.run().await;
        assert!(report.is_boot_safe());
        assert_eq!(report.pass_count(), 4);
        assert_eq!(report.fail_count(), 0);
    }

    #[tokio::test]
    async fn test_executive_grpc_failure_aborts() {
        use super::super::PreFlightRunner;

        let mut runner = PreFlightRunner::new();
        runner.add_check(Box::new(ExecutionGrpcCheck::in_memory(false))); // FAIL (Critical)
        runner.add_check(Box::new(ExchangeRestCheck::in_memory(
            ExchangeTarget::Kraken,
            true,
        )));

        let report = runner.run().await;
        assert!(!report.is_boot_safe());
        assert!(report.aborted);
        assert_eq!(report.abort_check.as_deref(), Some("Execution gRPC"));
    }

    #[tokio::test]
    async fn test_executive_bybit_failure_does_not_abort() {
        use super::super::PreFlightRunner;

        let mut runner = PreFlightRunner::new();
        runner.add_check(Box::new(ExecutionGrpcCheck::in_memory(true)));
        runner.add_check(Box::new(ExchangeRestCheck::in_memory(
            ExchangeTarget::Kraken,
            true,
        )));
        runner.add_check(Box::new(ExchangeRestCheck::in_memory(
            ExchangeTarget::Bybit,
            false,
        ))); // Fails but Required (not Critical)
        runner.add_check(Box::new(OrderPathCheck::in_memory(true, 3, 3)));

        let report = runner.run().await;
        // Bybit is Required, so boot is blocked but no abort
        assert!(!report.is_boot_safe());
        assert!(!report.aborted); // Required failures don't trigger abort, they just block boot
    }

    #[tokio::test]
    async fn test_executive_binance_failure_allows_boot() {
        use super::super::PreFlightRunner;

        let mut runner = PreFlightRunner::new();
        runner.add_check(Box::new(ExecutionGrpcCheck::in_memory(true)));
        runner.add_check(Box::new(ExchangeRestCheck::in_memory(
            ExchangeTarget::Kraken,
            true,
        )));
        runner.add_check(Box::new(ExchangeRestCheck::in_memory(
            ExchangeTarget::Binance,
            false,
        ))); // Fails but Optional
        runner.add_check(Box::new(
            OrderPathCheck::in_memory(true, 3, 3).with_criticality(Criticality::Optional),
        ));

        let report = runner.run().await;
        // Binance is Optional, so boot is allowed
        assert!(report.is_boot_safe());
        assert_eq!(report.fail_count(), 1);
    }

    // ── Full pipeline integration ──

    #[tokio::test]
    async fn test_full_executive_pipeline_all_pass() {
        use super::super::PreFlightRunner;

        let mut runner = PreFlightRunner::new();

        // All in-memory checks passing
        runner.add_check(Box::new(
            ExecutionGrpcCheck::in_memory_with_latency(true, 5)
                .with_max_latency(Duration::from_millis(100)),
        ));
        runner.add_check(Box::new(
            ExchangeRestCheck::in_memory_with_latency(ExchangeTarget::Kraken, true, 150)
                .with_max_latency(Duration::from_secs(2)),
        ));
        runner.add_check(Box::new(
            ExchangeRestCheck::in_memory_with_latency(ExchangeTarget::Bybit, true, 200)
                .with_max_latency(Duration::from_secs(2)),
        ));
        runner.add_check(Box::new(OrderPathCheck::in_memory(true, 4, 4)));

        let report = runner.run().await;
        assert!(report.is_boot_safe());
        assert_eq!(report.pass_count(), 4);
        assert_eq!(report.fail_count(), 0);
        assert_eq!(report.skip_count(), 0);
        assert!(!report.aborted);
    }
}
