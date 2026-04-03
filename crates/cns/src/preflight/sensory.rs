//! # Sensory Pre-Flight Checks
//!
//! Verifies data feeds and sensory input systems are operational before boot.
//!
//! ## Checks
//!
//! - **ExchangeWsCheck** — TCP connectivity to exchange WebSocket endpoints (Kraken, Bybit, Binance)
//! - **DataLatencyCheck** — Verify data feed freshness via HTTP or timestamp comparison
//! - **ViViTModelCheck** — Verify ViViT model checkpoint file exists on disk (optional/non-blocking)

use super::{BootPhase, Criticality, PreFlightCheck};
use async_trait::async_trait;
use std::path::{Path, PathBuf};
use std::time::Duration;

// ============================================================================
// Exchange WebSocket Check
// ============================================================================

/// Known exchange WebSocket endpoints.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum Exchange {
    Kraken,
    Bybit,
    Binance,
}

impl Exchange {
    /// Default public WebSocket host for this exchange.
    pub fn default_ws_host(&self) -> &'static str {
        match self {
            Exchange::Kraken => "ws.kraken.com",
            Exchange::Bybit => "stream.bybit.com",
            Exchange::Binance => "stream.binance.com",
        }
    }

    /// Default public WebSocket port (443 for wss).
    pub fn default_ws_port(&self) -> u16 {
        443
    }

    /// Display name.
    pub fn name(&self) -> &'static str {
        match self {
            Exchange::Kraken => "Kraken",
            Exchange::Bybit => "Bybit",
            Exchange::Binance => "Binance",
        }
    }
}

/// Verifies that an exchange WebSocket endpoint is reachable via TCP connect.
///
/// This performs a raw TCP connect to the exchange's WebSocket host:port to verify
/// network-level reachability. It does NOT perform the WebSocket handshake or
/// subscribe to any channels — that happens later during the data service startup.
pub struct ExchangeWsCheck {
    /// Which exchange to check.
    exchange: Exchange,
    /// Override host (if using a proxy or testnet).
    host: String,
    /// Override port.
    port: u16,
    /// Override criticality.
    criticality: Criticality,
    /// Connection timeout.
    timeout: Duration,
}

impl ExchangeWsCheck {
    /// Create a new WebSocket check for the given exchange using default endpoints.
    pub fn new(exchange: Exchange) -> Self {
        Self {
            host: exchange.default_ws_host().to_string(),
            port: exchange.default_ws_port(),
            exchange,
            criticality: Criticality::Required,
            timeout: Duration::from_secs(10),
        }
    }

    /// Create with a custom host and port (e.g. for testnet or proxy).
    pub fn custom(exchange: Exchange, host: impl Into<String>, port: u16) -> Self {
        Self {
            exchange,
            host: host.into(),
            port,
            criticality: Criticality::Required,
            timeout: Duration::from_secs(10),
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
impl PreFlightCheck for ExchangeWsCheck {
    fn name(&self) -> &str {
        // We can't return a reference to a computed String from &self,
        // so we use a static-ish approach per exchange variant.
        match self.exchange {
            Exchange::Kraken => "Kraken WebSocket",
            Exchange::Bybit => "Bybit WebSocket",
            Exchange::Binance => "Binance WebSocket",
        }
    }

    fn phase(&self) -> BootPhase {
        BootPhase::Sensory
    }

    fn criticality(&self) -> Criticality {
        self.criticality
    }

    fn timeout(&self) -> Duration {
        self.timeout
    }

    async fn execute(&self) -> Result<(), String> {
        let addr = self.addr();

        // Resolve DNS
        let socket_addrs: Vec<std::net::SocketAddr> = tokio::net::lookup_host(&addr)
            .await
            .map_err(|e| {
                format!(
                    "DNS resolution failed for {} ({}): {}",
                    self.exchange.name(),
                    addr,
                    e
                )
            })?
            .collect();

        if socket_addrs.is_empty() {
            return Err(format!(
                "No addresses resolved for {} ({})",
                self.exchange.name(),
                addr
            ));
        }

        // Attempt TCP connect to the first resolved address
        tokio::net::TcpStream::connect(socket_addrs[0])
            .await
            .map_err(|e| {
                format!(
                    "TCP connect to {} ({}) failed: {}",
                    self.exchange.name(),
                    addr,
                    e
                )
            })?;

        Ok(())
    }

    fn detail_on_success(&self) -> Option<String> {
        Some(format!("TCP connect OK to {}", self.addr()))
    }
}

// ============================================================================
// Data Latency Check
// ============================================================================

/// Verifies that a data feed endpoint responds within acceptable latency bounds.
///
/// This can check any HTTP endpoint that returns quickly — for example, a local
/// data service healthcheck, or a public exchange REST API endpoint. The check
/// verifies both reachability and that the response arrives within `max_latency`.
pub struct DataLatencyCheck {
    /// Human-readable label for which data feed this checks.
    label: String,
    /// URL to probe (HTTP GET).
    url: String,
    /// Maximum acceptable response latency. If the response takes longer, the check fails.
    max_latency: Duration,
    /// Override criticality (default: Required).
    criticality: Criticality,
    /// Request timeout (should be >= max_latency).
    timeout: Duration,
}

impl DataLatencyCheck {
    /// Create a new data latency check.
    ///
    /// - `label`: Human-readable name (e.g. "Kraken REST", "Local Data Service")
    /// - `url`: HTTP endpoint to probe
    /// - `max_latency`: Maximum acceptable response time
    pub fn new(label: impl Into<String>, url: impl Into<String>, max_latency: Duration) -> Self {
        let timeout = max_latency * 2; // Give some headroom for the timeout
        Self {
            label: label.into(),
            url: url.into(),
            max_latency,
            criticality: Criticality::Required,
            timeout,
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

    /// Convenience: create a latency check for Kraken's public REST API.
    pub fn kraken_rest(max_latency: Duration) -> Self {
        Self::new(
            "Kraken REST Latency",
            "https://api.kraken.com/0/public/Time",
            max_latency,
        )
    }

    /// Convenience: create a latency check for Bybit's public REST API.
    pub fn bybit_rest(max_latency: Duration) -> Self {
        Self::new(
            "Bybit REST Latency",
            "https://api.bybit.com/v5/market/time",
            max_latency,
        )
    }

    /// Convenience: create a latency check for a local data service.
    pub fn local_data_service(port: u16, max_latency: Duration) -> Self {
        Self::new(
            "Local Data Service",
            format!("http://127.0.0.1:{}/health", port),
            max_latency,
        )
    }
}

#[async_trait]
impl PreFlightCheck for DataLatencyCheck {
    fn name(&self) -> &str {
        &self.label
    }

    fn phase(&self) -> BootPhase {
        BootPhase::Sensory
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

        let start = std::time::Instant::now();

        let response = client.get(&self.url).send().await.map_err(|e| {
            format!(
                "Data latency check '{}' request to {} failed: {}",
                self.label, self.url, e
            )
        })?;

        let elapsed = start.elapsed();

        let status = response.status();
        if !status.is_success() {
            let body = response.text().await.unwrap_or_default();
            return Err(format!(
                "Data latency check '{}' returned HTTP {} from {}: {}",
                self.label, status, self.url, body
            ));
        }

        // Check latency
        if elapsed > self.max_latency {
            return Err(format!(
                "Data latency check '{}' exceeded max latency: {:.1}ms > {:.1}ms threshold",
                self.label,
                elapsed.as_secs_f64() * 1000.0,
                self.max_latency.as_secs_f64() * 1000.0,
            ));
        }

        Ok(())
    }

    fn detail_on_success(&self) -> Option<String> {
        Some(format!(
            "latency OK (threshold: {}ms)",
            self.max_latency.as_millis()
        ))
    }
}

// ============================================================================
// ViViT Model Check
// ============================================================================

/// Verifies that the ViViT (Video Vision Transformer) model checkpoint exists on disk.
///
/// This is an **optional** check — the system can run without ViViT. If the model
/// is not present, the visual cortex will simply be disabled and the check will
/// produce a warning rather than blocking boot.
pub struct ViViTModelCheck {
    /// Path to the model checkpoint file or directory.
    checkpoint_path: PathBuf,
    /// Minimum file size in bytes to consider the checkpoint valid.
    /// This catches truncated or empty files.
    min_size_bytes: u64,
    /// Override criticality (default: Optional — system boots without ViViT).
    criticality: Criticality,
}

impl ViViTModelCheck {
    /// Create a new ViViT model check.
    ///
    /// `checkpoint_path` should point to the model file (e.g. `checkpoints/vivit_model.onnx`)
    /// or a directory containing the model artifacts.
    pub fn new(checkpoint_path: impl Into<PathBuf>) -> Self {
        Self {
            checkpoint_path: checkpoint_path.into(),
            min_size_bytes: 1024, // At least 1 KB for a valid model
            criticality: Criticality::Optional,
        }
    }

    /// Override the minimum size check.
    pub fn with_min_size(mut self, min_size_bytes: u64) -> Self {
        self.min_size_bytes = min_size_bytes;
        self
    }

    /// Override the criticality level.
    pub fn with_criticality(mut self, criticality: Criticality) -> Self {
        self.criticality = criticality;
        self
    }

    /// Default checkpoint path used by JANUS.
    pub fn default_path() -> Self {
        Self::new("checkpoints/vivit_model.onnx")
    }

    /// Check a directory for any model files (`.onnx`, `.pt`, `.safetensors`).
    pub fn from_directory(dir: impl Into<PathBuf>) -> Self {
        Self {
            checkpoint_path: dir.into(),
            min_size_bytes: 1024,
            criticality: Criticality::Optional,
        }
    }
}

#[async_trait]
impl PreFlightCheck for ViViTModelCheck {
    fn name(&self) -> &str {
        "ViViT Model"
    }

    fn phase(&self) -> BootPhase {
        BootPhase::Sensory
    }

    fn criticality(&self) -> Criticality {
        self.criticality
    }

    fn timeout(&self) -> Duration {
        Duration::from_secs(5)
    }

    async fn execute(&self) -> Result<(), String> {
        let path = &self.checkpoint_path;

        // Use spawn_blocking for filesystem operations to avoid blocking the async runtime
        let path_clone = path.clone();
        let min_size = self.min_size_bytes;

        tokio::task::spawn_blocking(move || validate_model_path(&path_clone, min_size))
            .await
            .map_err(|e| format!("ViViT model check task panicked: {}", e))?
    }

    fn detail_on_success(&self) -> Option<String> {
        Some(format!("model found at {}", self.checkpoint_path.display()))
    }
}

/// Validate a model file or directory on disk.
fn validate_model_path(path: &Path, min_size_bytes: u64) -> Result<(), String> {
    if !path.exists() {
        return Err(format!(
            "Model checkpoint not found at '{}'",
            path.display()
        ));
    }

    if path.is_file() {
        // Single file check
        let metadata = std::fs::metadata(path)
            .map_err(|e| format!("Cannot read metadata for '{}': {}", path.display(), e))?;

        if metadata.len() < min_size_bytes {
            return Err(format!(
                "Model file '{}' is too small ({} bytes, minimum {} bytes) — possibly truncated",
                path.display(),
                metadata.len(),
                min_size_bytes,
            ));
        }

        // Verify it has a recognized extension
        let ext = path.extension().and_then(|e| e.to_str()).unwrap_or("");
        let valid_extensions = ["onnx", "pt", "pth", "safetensors", "bin", "model"];
        if !valid_extensions.contains(&ext) {
            return Err(format!(
                "Model file '{}' has unrecognized extension '.{}' (expected one of: {})",
                path.display(),
                ext,
                valid_extensions.join(", "),
            ));
        }

        Ok(())
    } else if path.is_dir() {
        // Directory check: look for any model files
        let model_extensions = ["onnx", "pt", "pth", "safetensors", "bin", "model"];
        let entries = std::fs::read_dir(path)
            .map_err(|e| format!("Cannot read directory '{}': {}", path.display(), e))?;

        let mut found_model = false;
        for entry in entries {
            let entry = entry.map_err(|e| format!("Error reading directory entry: {}", e))?;

            if let Some(ext) = entry.path().extension().and_then(|e| e.to_str())
                && model_extensions.contains(&ext)
            {
                let metadata = entry.metadata().map_err(|e| {
                    format!(
                        "Cannot read metadata for '{}': {}",
                        entry.path().display(),
                        e
                    )
                })?;

                if metadata.len() >= min_size_bytes {
                    found_model = true;
                    break;
                }
            }
        }

        if !found_model {
            return Err(format!(
                "No valid model files found in directory '{}' (looked for: {})",
                path.display(),
                model_extensions.join(", "),
            ));
        }

        Ok(())
    } else {
        Err(format!(
            "Path '{}' is neither a file nor a directory",
            path.display()
        ))
    }
}

// ============================================================================
// Builder Helpers
// ============================================================================

/// Create the default set of sensory pre-flight checks for JANUS.
///
/// This creates checks for:
/// - Kraken WebSocket (primary exchange)
/// - Bybit WebSocket (fallback exchange)
/// - Kraken REST API latency
/// - ViViT model checkpoint (optional)
pub fn default_sensory_checks(vivit_checkpoint: Option<PathBuf>) -> Vec<Box<dyn PreFlightCheck>> {
    let mut checks: Vec<Box<dyn PreFlightCheck>> = vec![
        // Primary exchange WS
        Box::new(ExchangeWsCheck::new(Exchange::Kraken).with_criticality(Criticality::Critical)),
        // Fallback exchange WS
        Box::new(ExchangeWsCheck::new(Exchange::Bybit).with_criticality(Criticality::Required)),
        // REST API latency — 2 second threshold
        Box::new(DataLatencyCheck::kraken_rest(Duration::from_secs(2))),
    ];

    // ViViT model (optional)
    if let Some(path) = vivit_checkpoint {
        checks.push(Box::new(ViViTModelCheck::new(path)));
    } else {
        checks.push(Box::new(ViViTModelCheck::default_path()));
    }

    checks
}

// ============================================================================
// Tests
// ============================================================================

#[cfg(test)]
mod tests {
    use super::*;
    use std::io::Write;
    use tempfile::TempDir;

    // ── Exchange enum tests ──

    #[test]
    fn test_exchange_default_hosts() {
        assert_eq!(Exchange::Kraken.default_ws_host(), "ws.kraken.com");
        assert_eq!(Exchange::Bybit.default_ws_host(), "stream.bybit.com");
        assert_eq!(Exchange::Binance.default_ws_host(), "stream.binance.com");
    }

    #[test]
    fn test_exchange_default_port() {
        assert_eq!(Exchange::Kraken.default_ws_port(), 443);
        assert_eq!(Exchange::Bybit.default_ws_port(), 443);
        assert_eq!(Exchange::Binance.default_ws_port(), 443);
    }

    #[test]
    fn test_exchange_names() {
        assert_eq!(Exchange::Kraken.name(), "Kraken");
        assert_eq!(Exchange::Bybit.name(), "Bybit");
        assert_eq!(Exchange::Binance.name(), "Binance");
    }

    // ── ExchangeWsCheck construction tests ──

    #[test]
    fn test_exchange_ws_check_defaults() {
        let check = ExchangeWsCheck::new(Exchange::Kraken);
        assert_eq!(check.name(), "Kraken WebSocket");
        assert_eq!(check.phase(), BootPhase::Sensory);
        assert_eq!(check.criticality(), Criticality::Required);
        assert_eq!(check.host, "ws.kraken.com");
        assert_eq!(check.port, 443);
    }

    #[test]
    fn test_exchange_ws_check_custom() {
        let check = ExchangeWsCheck::custom(Exchange::Bybit, "testnet.bybit.com", 8443);
        assert_eq!(check.host, "testnet.bybit.com");
        assert_eq!(check.port, 8443);
        assert_eq!(check.name(), "Bybit WebSocket");
    }

    #[test]
    fn test_exchange_ws_check_builder() {
        let check = ExchangeWsCheck::new(Exchange::Binance)
            .with_criticality(Criticality::Optional)
            .with_timeout(Duration::from_secs(20));
        assert_eq!(check.criticality(), Criticality::Optional);
        assert_eq!(check.timeout(), Duration::from_secs(20));
    }

    #[test]
    fn test_exchange_ws_check_addr() {
        let check = ExchangeWsCheck::new(Exchange::Kraken);
        assert_eq!(check.addr(), "ws.kraken.com:443");
    }

    #[test]
    fn test_exchange_ws_check_name_variants() {
        assert_eq!(
            ExchangeWsCheck::new(Exchange::Kraken).name(),
            "Kraken WebSocket"
        );
        assert_eq!(
            ExchangeWsCheck::new(Exchange::Bybit).name(),
            "Bybit WebSocket"
        );
        assert_eq!(
            ExchangeWsCheck::new(Exchange::Binance).name(),
            "Binance WebSocket"
        );
    }

    // ── DataLatencyCheck construction tests ──

    #[test]
    fn test_data_latency_check_defaults() {
        let check = DataLatencyCheck::new(
            "Test Feed",
            "http://localhost:8080/health",
            Duration::from_millis(500),
        );
        assert_eq!(check.name(), "Test Feed");
        assert_eq!(check.phase(), BootPhase::Sensory);
        assert_eq!(check.criticality(), Criticality::Required);
        assert_eq!(check.max_latency, Duration::from_millis(500));
        // Timeout should be 2x max_latency
        assert_eq!(check.timeout(), Duration::from_secs(1));
    }

    #[test]
    fn test_data_latency_check_builder() {
        let check = DataLatencyCheck::new("Feed", "http://example.com", Duration::from_millis(200))
            .with_criticality(Criticality::Critical)
            .with_timeout(Duration::from_secs(5));
        assert_eq!(check.criticality(), Criticality::Critical);
        assert_eq!(check.timeout(), Duration::from_secs(5));
    }

    #[test]
    fn test_data_latency_kraken_rest() {
        let check = DataLatencyCheck::kraken_rest(Duration::from_secs(2));
        assert_eq!(check.name(), "Kraken REST Latency");
        assert_eq!(check.url, "https://api.kraken.com/0/public/Time");
    }

    #[test]
    fn test_data_latency_bybit_rest() {
        let check = DataLatencyCheck::bybit_rest(Duration::from_secs(2));
        assert_eq!(check.name(), "Bybit REST Latency");
        assert_eq!(check.url, "https://api.bybit.com/v5/market/time");
    }

    #[test]
    fn test_data_latency_local_service() {
        let check = DataLatencyCheck::local_data_service(8080, Duration::from_millis(100));
        assert_eq!(check.name(), "Local Data Service");
        assert_eq!(check.url, "http://127.0.0.1:8080/health");
    }

    // ── ViViTModelCheck construction tests ──

    #[test]
    fn test_vivit_check_defaults() {
        let check = ViViTModelCheck::new("models/vivit.onnx");
        assert_eq!(check.name(), "ViViT Model");
        assert_eq!(check.phase(), BootPhase::Sensory);
        assert_eq!(check.criticality(), Criticality::Optional);
        assert_eq!(check.min_size_bytes, 1024);
        assert_eq!(check.checkpoint_path, PathBuf::from("models/vivit.onnx"));
    }

    #[test]
    fn test_vivit_check_builder() {
        let check = ViViTModelCheck::new("models/vivit.onnx")
            .with_min_size(1_000_000)
            .with_criticality(Criticality::Required);
        assert_eq!(check.criticality(), Criticality::Required);
        assert_eq!(check.min_size_bytes, 1_000_000);
    }

    #[test]
    fn test_vivit_default_path() {
        let check = ViViTModelCheck::default_path();
        assert_eq!(
            check.checkpoint_path,
            PathBuf::from("checkpoints/vivit_model.onnx")
        );
    }

    #[test]
    fn test_vivit_from_directory() {
        let check = ViViTModelCheck::from_directory("/opt/models/vivit");
        assert_eq!(check.checkpoint_path, PathBuf::from("/opt/models/vivit"));
        assert_eq!(check.criticality(), Criticality::Optional);
    }

    // ── Model validation logic tests ──

    #[test]
    fn test_validate_model_path_nonexistent() {
        let result = validate_model_path(Path::new("/nonexistent/model.onnx"), 1024);
        assert!(result.is_err());
        assert!(result.unwrap_err().contains("not found"));
    }

    #[test]
    fn test_validate_model_path_valid_file() {
        let dir = TempDir::new().unwrap();
        let model_path = dir.path().join("model.onnx");
        // Write enough data to pass the minimum size check
        let mut file = std::fs::File::create(&model_path).unwrap();
        file.write_all(&vec![0u8; 2048]).unwrap();
        drop(file);

        let result = validate_model_path(&model_path, 1024);
        assert!(result.is_ok());
    }

    #[test]
    fn test_validate_model_path_too_small() {
        let dir = TempDir::new().unwrap();
        let model_path = dir.path().join("model.onnx");
        let mut file = std::fs::File::create(&model_path).unwrap();
        file.write_all(&[0u8; 10]).unwrap(); // Way too small
        drop(file);

        let result = validate_model_path(&model_path, 1024);
        assert!(result.is_err());
        assert!(result.unwrap_err().contains("too small"));
    }

    #[test]
    fn test_validate_model_path_bad_extension() {
        let dir = TempDir::new().unwrap();
        let model_path = dir.path().join("model.txt");
        let mut file = std::fs::File::create(&model_path).unwrap();
        file.write_all(&vec![0u8; 2048]).unwrap();
        drop(file);

        let result = validate_model_path(&model_path, 1024);
        assert!(result.is_err());
        assert!(result.unwrap_err().contains("unrecognized extension"));
    }

    #[test]
    fn test_validate_model_directory_with_model() {
        let dir = TempDir::new().unwrap();
        let model_path = dir.path().join("weights.safetensors");
        let mut file = std::fs::File::create(&model_path).unwrap();
        file.write_all(&vec![0u8; 4096]).unwrap();
        drop(file);

        let result = validate_model_path(dir.path(), 1024);
        assert!(result.is_ok());
    }

    #[test]
    fn test_validate_model_directory_empty() {
        let dir = TempDir::new().unwrap();
        let result = validate_model_path(dir.path(), 1024);
        assert!(result.is_err());
        assert!(result.unwrap_err().contains("No valid model files"));
    }

    #[test]
    fn test_validate_model_directory_with_only_small_files() {
        let dir = TempDir::new().unwrap();
        let model_path = dir.path().join("model.onnx");
        let mut file = std::fs::File::create(&model_path).unwrap();
        file.write_all(&[0u8; 10]).unwrap(); // Too small
        drop(file);

        let result = validate_model_path(dir.path(), 1024);
        assert!(result.is_err());
        assert!(result.unwrap_err().contains("No valid model files"));
    }

    #[test]
    fn test_validate_model_multiple_extensions() {
        // Verify all valid extensions work
        let valid_exts = ["onnx", "pt", "pth", "safetensors", "bin", "model"];
        for ext in &valid_exts {
            let dir = TempDir::new().unwrap();
            let model_path = dir.path().join(format!("weights.{}", ext));
            let mut file = std::fs::File::create(&model_path).unwrap();
            file.write_all(&vec![0u8; 2048]).unwrap();
            drop(file);

            let result = validate_model_path(&model_path, 1024);
            assert!(result.is_ok(), "Extension '{}' should be valid", ext);
        }
    }

    // ── Async execution tests ──

    #[tokio::test]
    async fn test_vivit_check_nonexistent_file() {
        let check = ViViTModelCheck::new("/nonexistent/path/model.onnx");
        let result = check.execute().await;
        assert!(result.is_err());
    }

    #[tokio::test]
    async fn test_vivit_check_valid_file() {
        let dir = TempDir::new().unwrap();
        let model_path = dir.path().join("model.onnx");
        let mut file = std::fs::File::create(&model_path).unwrap();
        file.write_all(&vec![0u8; 2048]).unwrap();
        drop(file);

        let check = ViViTModelCheck::new(&model_path);
        let result = check.execute().await;
        assert!(result.is_ok());
    }

    #[tokio::test]
    async fn test_vivit_run_produces_correct_result() {
        let check = ViViTModelCheck::new("/nonexistent/model.onnx");
        let result = check.run().await;

        assert_eq!(result.name, "ViViT Model");
        assert_eq!(result.phase, BootPhase::Sensory);
        assert_eq!(result.criticality, Criticality::Optional);
        assert!(result.outcome.is_fail());
        // Optional failure should NOT be abort-worthy
        assert!(!result.is_abort_worthy());
    }

    // ── Network-dependent tests (expected to fail without internet) ──

    #[tokio::test]
    async fn test_exchange_ws_unreachable_fails() {
        let check = ExchangeWsCheck::custom(Exchange::Kraken, "127.0.0.1", 59990)
            .with_timeout(Duration::from_secs(1));
        let result = check.execute().await;
        assert!(result.is_err());
    }

    #[tokio::test]
    async fn test_data_latency_unreachable_fails() {
        let check = DataLatencyCheck::new(
            "Unreachable",
            "http://127.0.0.1:59989/health",
            Duration::from_millis(100),
        )
        .with_timeout(Duration::from_secs(1));
        let result = check.execute().await;
        assert!(result.is_err());
    }

    // ── Builder helper tests ──

    #[test]
    fn test_default_sensory_checks_count() {
        let checks = default_sensory_checks(None);
        assert_eq!(checks.len(), 4); // Kraken WS, Bybit WS, Kraken REST, ViViT
    }

    #[test]
    fn test_default_sensory_checks_with_custom_vivit() {
        let checks = default_sensory_checks(Some(PathBuf::from("/custom/model.onnx")));
        assert_eq!(checks.len(), 4);
    }

    #[test]
    fn test_default_sensory_checks_phases() {
        let checks = default_sensory_checks(None);
        for check in &checks {
            assert_eq!(check.phase(), BootPhase::Sensory);
        }
    }
}
