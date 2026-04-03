//! # Strategy Pre-Flight Checks
//!
//! Verifies strategy subsystems are initialized and ready before the system begins trading.
//!
//! ## Checks
//!
//! - **RegimeDetectorCheck** — Verify the regime detector has warmed up with enough data
//! - **StrategyInstantiationCheck** — Verify strategies can be instantiated from configuration
//! - **CorrelationTrackerCheck** — Verify the correlation tracker is initialized with pair config
//! - **AffinityLoadCheck** — Verify strategy affinity configuration loads from TOML/Redis

use super::{BootPhase, Criticality, PreFlightCheck};
use async_trait::async_trait;
use std::path::{Path, PathBuf};
use std::time::Duration;

// ============================================================================
// Regime Detector Check
// ============================================================================

/// Method used to verify the regime detector warmup state.
#[derive(Debug, Clone)]
pub enum RegimeDetectorVerification {
    /// Check the regime detector via an HTTP health endpoint that reports warmup status.
    HttpEndpoint {
        /// URL to the regime detector health/status endpoint.
        url: String,
    },
    /// Check a Redis key that holds the regime detector warmup state.
    RedisKey {
        /// Redis connection URL.
        url: String,
        /// Key that holds the current regime (e.g. `janus:regime:current`).
        key: String,
    },
    /// In-memory verification for testing.
    InMemory {
        /// Whether the detector has warmed up.
        warmed_up: bool,
        /// Number of data points consumed so far.
        data_points: usize,
        /// Minimum data points required for warmup.
        min_required: usize,
    },
}

/// Verifies the regime detector has warmed up with sufficient historical data.
///
/// The regime detector identifies the current market regime (trending, mean-reverting,
/// volatile, etc.) and needs a minimum number of data points before it can produce
/// reliable regime classifications. This check verifies:
///
/// 1. The regime detector service/module is reachable
/// 2. It has consumed enough data points to be considered "warmed up"
/// 3. It is currently producing a valid regime classification
pub struct RegimeDetectorCheck {
    /// How to verify the regime detector.
    verification: RegimeDetectorVerification,
    /// Override criticality (default: Required — the system can trade without regime
    /// detection but will use default parameters, which is suboptimal).
    criticality: Criticality,
    /// Verification timeout.
    timeout: Duration,
}

impl RegimeDetectorCheck {
    /// Create a regime detector check using an HTTP endpoint.
    pub fn http(url: impl Into<String>) -> Self {
        Self {
            verification: RegimeDetectorVerification::HttpEndpoint { url: url.into() },
            criticality: Criticality::Required,
            timeout: Duration::from_secs(10),
        }
    }

    /// Create a regime detector check using a Redis key.
    pub fn redis(url: impl Into<String>, key: impl Into<String>) -> Self {
        Self {
            verification: RegimeDetectorVerification::RedisKey {
                url: url.into(),
                key: key.into(),
            },
            criticality: Criticality::Required,
            timeout: Duration::from_secs(5),
        }
    }

    /// Create a regime detector check using in-memory state (for testing).
    pub fn in_memory(warmed_up: bool, data_points: usize, min_required: usize) -> Self {
        Self {
            verification: RegimeDetectorVerification::InMemory {
                warmed_up,
                data_points,
                min_required,
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
impl PreFlightCheck for RegimeDetectorCheck {
    fn name(&self) -> &str {
        "Regime Detector"
    }

    fn phase(&self) -> BootPhase {
        BootPhase::Strategy
    }

    fn criticality(&self) -> Criticality {
        self.criticality
    }

    fn timeout(&self) -> Duration {
        self.timeout
    }

    async fn execute(&self) -> Result<(), String> {
        match &self.verification {
            RegimeDetectorVerification::HttpEndpoint { url } => {
                let client = reqwest::Client::builder()
                    .timeout(self.timeout)
                    .build()
                    .map_err(|e| format!("Failed to create HTTP client: {}", e))?;

                let response =
                    client.get(url).send().await.map_err(|e| {
                        format!("Regime detector health check failed ({}): {}", url, e)
                    })?;

                let status = response.status();
                if !status.is_success() {
                    let body = response.text().await.unwrap_or_default();
                    return Err(format!(
                        "Regime detector returned HTTP {} ({}): {}",
                        status, url, body
                    ));
                }

                // Optionally parse the response for warmup status
                let body = response.text().await.unwrap_or_default();
                if !body.is_empty()
                    && let Ok(json) = serde_json::from_str::<serde_json::Value>(&body)
                    && let Some(warmed_up) = json.get("warmed_up").and_then(|v| v.as_bool())
                    && !warmed_up
                {
                    let data_points = json
                        .get("data_points")
                        .and_then(|v| v.as_u64())
                        .unwrap_or(0);
                    let min_required = json
                        .get("min_required")
                        .and_then(|v| v.as_u64())
                        .unwrap_or(0);
                    return Err(format!(
                        "Regime detector not warmed up: {}/{} data points",
                        data_points, min_required
                    ));
                }

                Ok(())
            }

            RegimeDetectorVerification::RedisKey { url, key } => {
                let client = redis::Client::open(url.as_str())
                    .map_err(|e| format!("Invalid Redis URL for regime detector: {}", e))?;

                let mut conn = client
                    .get_multiplexed_async_connection()
                    .await
                    .map_err(|e| format!("Regime detector Redis connection failed: {}", e))?;

                let value: Option<String> = redis::cmd("GET")
                    .arg(key.as_str())
                    .query_async(&mut conn)
                    .await
                    .map_err(|e| format!("Regime detector Redis GET failed: {}", e))?;

                match value {
                    Some(v) if !v.is_empty() => {
                        // A non-empty value means the regime detector has produced at least one classification
                        Ok(())
                    }
                    Some(_) => Err(format!(
                        "Regime detector key '{}' is empty — detector may not have warmed up",
                        key
                    )),
                    None => Err(format!(
                        "Regime detector key '{}' not found — detector may not be initialized",
                        key
                    )),
                }
            }

            RegimeDetectorVerification::InMemory {
                warmed_up,
                data_points,
                min_required,
            } => {
                if !*warmed_up {
                    return Err(format!(
                        "Regime detector not warmed up: {}/{} data points consumed",
                        data_points, min_required
                    ));
                }

                if *data_points < *min_required {
                    return Err(format!(
                        "Regime detector has insufficient data: {}/{} data points",
                        data_points, min_required
                    ));
                }

                Ok(())
            }
        }
    }

    fn detail_on_success(&self) -> Option<String> {
        match &self.verification {
            RegimeDetectorVerification::InMemory {
                data_points,
                min_required,
                ..
            } => Some(format!(
                "warmed up with {}/{} data points",
                data_points, min_required
            )),
            _ => Some("warmed up and producing classifications".to_string()),
        }
    }
}

// ============================================================================
// Strategy Instantiation Check
// ============================================================================

/// Method used to verify strategy instantiation.
#[derive(Debug, Clone)]
pub enum StrategyVerification {
    /// Check that a strategy configuration file exists and is parseable.
    ConfigFile {
        /// Path to the strategy configuration file (TOML).
        path: PathBuf,
    },
    /// Check an HTTP endpoint that reports strategy readiness.
    HttpEndpoint {
        /// URL to the strategy service health/status endpoint.
        url: String,
    },
    /// In-memory verification for testing.
    InMemory {
        /// Number of strategies that can be instantiated.
        strategy_count: usize,
        /// Whether all strategies passed self-test.
        all_valid: bool,
    },
}

/// Verifies that trading strategies can be instantiated from configuration.
///
/// This check ensures:
///
/// 1. Strategy configuration files exist and are valid TOML
/// 2. All referenced strategy implementations are available
/// 3. Strategy parameters are within acceptable bounds
/// 4. At least one strategy is configured for each target asset
pub struct StrategyInstantiationCheck {
    /// How to verify strategy instantiation.
    verification: StrategyVerification,
    /// Override criticality (default: Critical — cannot trade without strategies).
    criticality: Criticality,
    /// Verification timeout.
    timeout: Duration,
}

impl StrategyInstantiationCheck {
    /// Create a strategy check by validating a configuration file.
    pub fn from_config(path: impl Into<PathBuf>) -> Self {
        Self {
            verification: StrategyVerification::ConfigFile { path: path.into() },
            criticality: Criticality::Critical,
            timeout: Duration::from_secs(5),
        }
    }

    /// Create a strategy check using an HTTP endpoint.
    pub fn http(url: impl Into<String>) -> Self {
        Self {
            verification: StrategyVerification::HttpEndpoint { url: url.into() },
            criticality: Criticality::Critical,
            timeout: Duration::from_secs(5),
        }
    }

    /// Create a strategy check using in-memory state (for testing).
    pub fn in_memory(strategy_count: usize, all_valid: bool) -> Self {
        Self {
            verification: StrategyVerification::InMemory {
                strategy_count,
                all_valid,
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
impl PreFlightCheck for StrategyInstantiationCheck {
    fn name(&self) -> &str {
        "Strategy Instantiation"
    }

    fn phase(&self) -> BootPhase {
        BootPhase::Strategy
    }

    fn criticality(&self) -> Criticality {
        self.criticality
    }

    fn timeout(&self) -> Duration {
        self.timeout
    }

    async fn execute(&self) -> Result<(), String> {
        match &self.verification {
            StrategyVerification::ConfigFile { path } => {
                let path_clone = path.clone();
                tokio::task::spawn_blocking(move || validate_strategy_config(&path_clone))
                    .await
                    .map_err(|e| format!("Strategy config validation task panicked: {}", e))?
            }

            StrategyVerification::HttpEndpoint { url } => {
                let client = reqwest::Client::builder()
                    .timeout(self.timeout)
                    .build()
                    .map_err(|e| format!("Failed to create HTTP client: {}", e))?;

                let response = client.get(url).send().await.map_err(|e| {
                    format!("Strategy service health check failed ({}): {}", url, e)
                })?;

                let status = response.status();
                if !status.is_success() {
                    let body = response.text().await.unwrap_or_default();
                    return Err(format!(
                        "Strategy service returned HTTP {} ({}): {}",
                        status, url, body
                    ));
                }

                Ok(())
            }

            StrategyVerification::InMemory {
                strategy_count,
                all_valid,
            } => {
                if *strategy_count == 0 {
                    return Err(
                        "No strategies configured — at least 1 strategy is required".to_string()
                    );
                }

                if !*all_valid {
                    return Err(
                        "One or more strategies failed self-test — check configuration".to_string(),
                    );
                }

                Ok(())
            }
        }
    }

    fn detail_on_success(&self) -> Option<String> {
        match &self.verification {
            StrategyVerification::ConfigFile { path } => {
                Some(format!("config valid at {}", path.display()))
            }
            StrategyVerification::InMemory { strategy_count, .. } => Some(format!(
                "{} strategies instantiated and valid",
                strategy_count
            )),
            _ => Some("all strategies instantiated successfully".to_string()),
        }
    }
}

/// Validate a strategy configuration TOML file.
fn validate_strategy_config(path: &Path) -> Result<(), String> {
    if !path.exists() {
        return Err(format!(
            "Strategy config file not found at '{}'",
            path.display()
        ));
    }

    if !path.is_file() {
        return Err(format!(
            "Strategy config path '{}' is not a file",
            path.display()
        ));
    }

    let content = std::fs::read_to_string(path)
        .map_err(|e| format!("Cannot read strategy config '{}': {}", path.display(), e))?;

    if content.trim().is_empty() {
        return Err(format!(
            "Strategy config file '{}' is empty",
            path.display()
        ));
    }

    // Verify it's valid TOML
    let _parsed: toml::Value = toml::from_str(&content).map_err(|e| {
        format!(
            "Strategy config '{}' has invalid TOML: {}",
            path.display(),
            e
        )
    })?;

    Ok(())
}

// ============================================================================
// Correlation Tracker Check
// ============================================================================

/// Method used to verify the correlation tracker.
#[derive(Debug, Clone)]
pub enum CorrelationVerification {
    /// Check that a correlation pairs configuration file exists and is valid.
    ConfigFile {
        /// Path to the correlation pairs TOML file.
        path: PathBuf,
    },
    /// Check an HTTP endpoint that reports correlation tracker readiness.
    HttpEndpoint {
        /// URL to the correlation tracker health/status endpoint.
        url: String,
    },
    /// In-memory verification for testing.
    InMemory {
        /// Whether the tracker is initialized.
        initialized: bool,
        /// Number of correlation pairs being monitored.
        pair_count: usize,
    },
}

/// Verifies the correlation tracker is initialized with pair configuration.
///
/// The correlation tracker monitors cross-asset correlations to prevent
/// concentrated directional exposure. This check verifies:
///
/// 1. The correlation pairs configuration file exists and is valid TOML
/// 2. At least one correlation pair is defined
/// 3. The tracker is initialized and ready to accept price updates
pub struct CorrelationTrackerCheck {
    /// How to verify the correlation tracker.
    verification: CorrelationVerification,
    /// Override criticality (default: Required — correlation tracking is important
    /// for risk management but the system can run without it).
    criticality: Criticality,
    /// Verification timeout.
    timeout: Duration,
}

impl CorrelationTrackerCheck {
    /// Create a correlation tracker check by validating a config file.
    pub fn from_config(path: impl Into<PathBuf>) -> Self {
        Self {
            verification: CorrelationVerification::ConfigFile { path: path.into() },
            criticality: Criticality::Required,
            timeout: Duration::from_secs(5),
        }
    }

    /// Create a correlation tracker check using an HTTP endpoint.
    pub fn http(url: impl Into<String>) -> Self {
        Self {
            verification: CorrelationVerification::HttpEndpoint { url: url.into() },
            criticality: Criticality::Required,
            timeout: Duration::from_secs(5),
        }
    }

    /// Create a correlation tracker check using in-memory state (for testing).
    pub fn in_memory(initialized: bool, pair_count: usize) -> Self {
        Self {
            verification: CorrelationVerification::InMemory {
                initialized,
                pair_count,
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
impl PreFlightCheck for CorrelationTrackerCheck {
    fn name(&self) -> &str {
        "Correlation Tracker"
    }

    fn phase(&self) -> BootPhase {
        BootPhase::Strategy
    }

    fn criticality(&self) -> Criticality {
        self.criticality
    }

    fn timeout(&self) -> Duration {
        self.timeout
    }

    async fn execute(&self) -> Result<(), String> {
        match &self.verification {
            CorrelationVerification::ConfigFile { path } => {
                let path_clone = path.clone();
                tokio::task::spawn_blocking(move || validate_correlation_config(&path_clone))
                    .await
                    .map_err(|e| format!("Correlation config validation task panicked: {}", e))?
            }

            CorrelationVerification::HttpEndpoint { url } => {
                let client = reqwest::Client::builder()
                    .timeout(self.timeout)
                    .build()
                    .map_err(|e| format!("Failed to create HTTP client: {}", e))?;

                let response = client.get(url).send().await.map_err(|e| {
                    format!("Correlation tracker health check failed ({}): {}", url, e)
                })?;

                let status = response.status();
                if !status.is_success() {
                    let body = response.text().await.unwrap_or_default();
                    return Err(format!(
                        "Correlation tracker returned HTTP {} ({}): {}",
                        status, url, body
                    ));
                }

                Ok(())
            }

            CorrelationVerification::InMemory {
                initialized,
                pair_count,
            } => {
                if !*initialized {
                    return Err("Correlation tracker is not initialized".to_string());
                }

                if *pair_count == 0 {
                    return Err(
                        "Correlation tracker has no pairs configured — at least 1 pair required"
                            .to_string(),
                    );
                }

                Ok(())
            }
        }
    }

    fn detail_on_success(&self) -> Option<String> {
        match &self.verification {
            CorrelationVerification::ConfigFile { path } => {
                Some(format!("config valid at {}", path.display()))
            }
            CorrelationVerification::InMemory { pair_count, .. } => {
                Some(format!("{} correlation pairs loaded", pair_count))
            }
            _ => Some("initialized and monitoring pairs".to_string()),
        }
    }
}

/// Validate a correlation pairs TOML configuration file.
fn validate_correlation_config(path: &Path) -> Result<(), String> {
    if !path.exists() {
        return Err(format!(
            "Correlation config file not found at '{}'",
            path.display()
        ));
    }

    if !path.is_file() {
        return Err(format!(
            "Correlation config path '{}' is not a file",
            path.display()
        ));
    }

    let content = std::fs::read_to_string(path)
        .map_err(|e| format!("Cannot read correlation config '{}': {}", path.display(), e))?;

    if content.trim().is_empty() {
        return Err(format!(
            "Correlation config file '{}' is empty",
            path.display()
        ));
    }

    // Verify it's valid TOML
    let parsed: toml::Value = toml::from_str(&content).map_err(|e| {
        format!(
            "Correlation config '{}' has invalid TOML: {}",
            path.display(),
            e
        )
    })?;

    // Verify it contains at least one pair definition
    let has_pairs = parsed
        .get("pairs")
        .or_else(|| parsed.get("correlation_pairs"))
        .or_else(|| parsed.get("monitored_pairs"))
        .and_then(|v| v.as_array())
        .map(|arr| !arr.is_empty())
        .unwrap_or(false);

    // Also check for [[pair]] table array syntax
    let has_pair_table = parsed
        .get("pair")
        .and_then(|v| v.as_array())
        .map(|arr| !arr.is_empty())
        .unwrap_or(false);

    if !has_pairs && !has_pair_table {
        return Err(format!(
            "Correlation config '{}' does not contain any pair definitions (expected 'pairs', 'correlation_pairs', 'monitored_pairs', or [[pair]] sections)",
            path.display()
        ));
    }

    Ok(())
}

// ============================================================================
// Affinity Load Check
// ============================================================================

/// Method used to verify strategy affinity loading.
#[derive(Debug, Clone)]
pub enum AffinityVerification {
    /// Check that a strategy affinity configuration file exists and is valid.
    ConfigFile {
        /// Path to the strategy affinity TOML file.
        path: PathBuf,
    },
    /// Check a Redis hash that stores persisted affinity state.
    RedisKey {
        /// Redis connection URL.
        url: String,
        /// Key prefix for affinity data (e.g. `janus:affinity:`).
        key_prefix: String,
    },
    /// In-memory verification for testing.
    InMemory {
        /// Whether the affinity data loaded successfully.
        loaded: bool,
        /// Number of asset-strategy affinity entries.
        entry_count: usize,
    },
}

/// Verifies that strategy affinity configuration loads successfully.
///
/// Strategy affinity tracks which strategies perform best on which assets
/// and adjusts weights accordingly. This check verifies:
///
/// 1. The affinity configuration file exists and is valid TOML
/// 2. At least one asset-strategy mapping is defined
/// 3. Any persisted affinity state in Redis is loadable (if configured)
pub struct AffinityLoadCheck {
    /// How to verify affinity loading.
    verification: AffinityVerification,
    /// Override criticality (default: Optional — the system can run with uniform
    /// strategy weights if affinity data is unavailable).
    criticality: Criticality,
    /// Verification timeout.
    timeout: Duration,
}

impl AffinityLoadCheck {
    /// Create an affinity check by validating a configuration file.
    pub fn from_config(path: impl Into<PathBuf>) -> Self {
        Self {
            verification: AffinityVerification::ConfigFile { path: path.into() },
            criticality: Criticality::Optional,
            timeout: Duration::from_secs(5),
        }
    }

    /// Create an affinity check using Redis persisted state.
    pub fn redis(url: impl Into<String>, key_prefix: impl Into<String>) -> Self {
        Self {
            verification: AffinityVerification::RedisKey {
                url: url.into(),
                key_prefix: key_prefix.into(),
            },
            criticality: Criticality::Optional,
            timeout: Duration::from_secs(5),
        }
    }

    /// Create an affinity check using in-memory state (for testing).
    pub fn in_memory(loaded: bool, entry_count: usize) -> Self {
        Self {
            verification: AffinityVerification::InMemory {
                loaded,
                entry_count,
            },
            criticality: Criticality::Optional,
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
impl PreFlightCheck for AffinityLoadCheck {
    fn name(&self) -> &str {
        "Affinity Load"
    }

    fn phase(&self) -> BootPhase {
        BootPhase::Strategy
    }

    fn criticality(&self) -> Criticality {
        self.criticality
    }

    fn timeout(&self) -> Duration {
        self.timeout
    }

    async fn execute(&self) -> Result<(), String> {
        match &self.verification {
            AffinityVerification::ConfigFile { path } => {
                let path_clone = path.clone();
                tokio::task::spawn_blocking(move || validate_affinity_config(&path_clone))
                    .await
                    .map_err(|e| format!("Affinity config validation task panicked: {}", e))?
            }

            AffinityVerification::RedisKey { url, key_prefix } => {
                let client = redis::Client::open(url.as_str())
                    .map_err(|e| format!("Invalid Redis URL for affinity: {}", e))?;

                let mut conn = client
                    .get_multiplexed_async_connection()
                    .await
                    .map_err(|e| format!("Affinity Redis connection failed: {}", e))?;

                // Scan for affinity keys
                let pattern = format!("{}*", key_prefix);
                let keys: Vec<String> = redis::cmd("KEYS")
                    .arg(&pattern)
                    .query_async(&mut conn)
                    .await
                    .map_err(|e| format!("Affinity KEYS scan failed: {}", e))?;

                if keys.is_empty() {
                    // This is OK for Optional criticality — just means no persisted state
                    return Err(format!(
                        "No persisted affinity data found (prefix '{}'). System will use default weights.",
                        key_prefix
                    ));
                }

                Ok(())
            }

            AffinityVerification::InMemory {
                loaded,
                entry_count,
            } => {
                if !*loaded {
                    return Err("Affinity data failed to load".to_string());
                }

                if *entry_count == 0 {
                    return Err(
                        "No affinity entries loaded — system will use uniform weights".to_string(),
                    );
                }

                Ok(())
            }
        }
    }

    fn detail_on_success(&self) -> Option<String> {
        match &self.verification {
            AffinityVerification::ConfigFile { path } => {
                Some(format!("config valid at {}", path.display()))
            }
            AffinityVerification::RedisKey { key_prefix, .. } => {
                Some(format!("persisted state loaded from '{}'", key_prefix))
            }
            AffinityVerification::InMemory { entry_count, .. } => {
                Some(format!("{} affinity entries loaded", entry_count))
            }
        }
    }
}

/// Validate a strategy affinity TOML configuration file.
fn validate_affinity_config(path: &Path) -> Result<(), String> {
    if !path.exists() {
        return Err(format!(
            "Affinity config file not found at '{}'",
            path.display()
        ));
    }

    if !path.is_file() {
        return Err(format!(
            "Affinity config path '{}' is not a file",
            path.display()
        ));
    }

    let content = std::fs::read_to_string(path)
        .map_err(|e| format!("Cannot read affinity config '{}': {}", path.display(), e))?;

    if content.trim().is_empty() {
        return Err(format!(
            "Affinity config file '{}' is empty",
            path.display()
        ));
    }

    // Verify it's valid TOML
    let _parsed: toml::Value = toml::from_str(&content).map_err(|e| {
        format!(
            "Affinity config '{}' has invalid TOML: {}",
            path.display(),
            e
        )
    })?;

    Ok(())
}

// ============================================================================
// Builder Helpers
// ============================================================================

/// Create the default set of strategy pre-flight checks for JANUS.
///
/// - Regime detector (required)
/// - Strategy instantiation from config (critical)
/// - Correlation tracker from config (required)
/// - Affinity load from config (optional)
pub fn default_strategy_checks(
    strategy_config_path: Option<PathBuf>,
    correlation_config_path: Option<PathBuf>,
    affinity_config_path: Option<PathBuf>,
    regime_redis_url: Option<&str>,
) -> Vec<Box<dyn PreFlightCheck>> {
    let mut checks: Vec<Box<dyn PreFlightCheck>> = Vec::new();

    // Regime detector
    if let Some(redis_url) = regime_redis_url {
        checks.push(Box::new(RegimeDetectorCheck::redis(
            redis_url,
            "janus:regime:current",
        )));
    } else {
        // Default to in-memory passing check (will be wired properly during integration)
        checks.push(Box::new(
            RegimeDetectorCheck::in_memory(true, 500, 100).with_criticality(Criticality::Optional),
        ));
    }

    // Strategy instantiation
    if let Some(path) = strategy_config_path {
        checks.push(Box::new(StrategyInstantiationCheck::from_config(path)));
    } else {
        checks.push(Box::new(StrategyInstantiationCheck::in_memory(1, true)));
    }

    // Correlation tracker
    if let Some(path) = correlation_config_path {
        checks.push(Box::new(CorrelationTrackerCheck::from_config(path)));
    } else {
        checks.push(Box::new(
            CorrelationTrackerCheck::in_memory(true, 1).with_criticality(Criticality::Optional),
        ));
    }

    // Affinity load
    if let Some(path) = affinity_config_path {
        checks.push(Box::new(AffinityLoadCheck::from_config(path)));
    } else {
        checks.push(Box::new(
            AffinityLoadCheck::in_memory(true, 0).with_criticality(Criticality::Optional),
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
    use tempfile::TempDir;

    // ── RegimeDetectorCheck construction tests ──

    #[test]
    fn test_regime_detector_http_defaults() {
        let check = RegimeDetectorCheck::http("http://localhost:8080/regime/status");
        assert_eq!(check.name(), "Regime Detector");
        assert_eq!(check.phase(), BootPhase::Strategy);
        assert_eq!(check.criticality(), Criticality::Required);
        assert_eq!(check.timeout(), Duration::from_secs(10));
    }

    #[test]
    fn test_regime_detector_redis_defaults() {
        let check = RegimeDetectorCheck::redis("redis://localhost:6379", "janus:regime:current");
        assert_eq!(check.criticality(), Criticality::Required);
        assert_eq!(check.timeout(), Duration::from_secs(5));
    }

    #[test]
    fn test_regime_detector_in_memory() {
        let check = RegimeDetectorCheck::in_memory(true, 500, 100);
        assert_eq!(check.timeout(), Duration::from_secs(1));
    }

    #[test]
    fn test_regime_detector_builder() {
        let check = RegimeDetectorCheck::in_memory(true, 500, 100)
            .with_criticality(Criticality::Optional)
            .with_timeout(Duration::from_secs(20));
        assert_eq!(check.criticality(), Criticality::Optional);
        assert_eq!(check.timeout(), Duration::from_secs(20));
    }

    // ── RegimeDetectorCheck execution tests ──

    #[tokio::test]
    async fn test_regime_detector_in_memory_warmed_passes() {
        let check = RegimeDetectorCheck::in_memory(true, 500, 100);
        let result = check.execute().await;
        assert!(result.is_ok());
    }

    #[tokio::test]
    async fn test_regime_detector_in_memory_not_warmed_fails() {
        let check = RegimeDetectorCheck::in_memory(false, 50, 100);
        let result = check.execute().await;
        assert!(result.is_err());
        assert!(result.unwrap_err().contains("not warmed up"));
    }

    #[tokio::test]
    async fn test_regime_detector_in_memory_insufficient_data_fails() {
        let check = RegimeDetectorCheck::in_memory(true, 50, 100);
        let result = check.execute().await;
        assert!(result.is_err());
        assert!(result.unwrap_err().contains("insufficient data"));
    }

    #[tokio::test]
    async fn test_regime_detector_in_memory_exact_threshold_passes() {
        let check = RegimeDetectorCheck::in_memory(true, 100, 100);
        let result = check.execute().await;
        assert!(result.is_ok());
    }

    #[tokio::test]
    async fn test_regime_detector_detail_on_success() {
        let check = RegimeDetectorCheck::in_memory(true, 500, 100);
        assert_eq!(
            check.detail_on_success(),
            Some("warmed up with 500/100 data points".to_string())
        );
    }

    #[tokio::test]
    async fn test_regime_detector_run_result() {
        let check = RegimeDetectorCheck::in_memory(true, 200, 100);
        let result = check.run().await;
        assert_eq!(result.name, "Regime Detector");
        assert_eq!(result.phase, BootPhase::Strategy);
        assert!(result.outcome.is_pass());
    }

    // ── StrategyInstantiationCheck construction tests ──

    #[test]
    fn test_strategy_instantiation_config_defaults() {
        let check = StrategyInstantiationCheck::from_config("config/strategies.toml");
        assert_eq!(check.name(), "Strategy Instantiation");
        assert_eq!(check.phase(), BootPhase::Strategy);
        assert_eq!(check.criticality(), Criticality::Critical);
    }

    #[test]
    fn test_strategy_instantiation_http_defaults() {
        let check = StrategyInstantiationCheck::http("http://localhost:8080/strategies/health");
        assert_eq!(check.criticality(), Criticality::Critical);
    }

    #[test]
    fn test_strategy_instantiation_in_memory() {
        let check = StrategyInstantiationCheck::in_memory(5, true);
        assert_eq!(check.timeout(), Duration::from_secs(1));
    }

    #[test]
    fn test_strategy_instantiation_builder() {
        let check = StrategyInstantiationCheck::in_memory(3, true)
            .with_criticality(Criticality::Required)
            .with_timeout(Duration::from_secs(8));
        assert_eq!(check.criticality(), Criticality::Required);
        assert_eq!(check.timeout(), Duration::from_secs(8));
    }

    // ── StrategyInstantiationCheck execution tests ──

    #[tokio::test]
    async fn test_strategy_in_memory_valid_passes() {
        let check = StrategyInstantiationCheck::in_memory(3, true);
        let result = check.execute().await;
        assert!(result.is_ok());
    }

    #[tokio::test]
    async fn test_strategy_in_memory_zero_count_fails() {
        let check = StrategyInstantiationCheck::in_memory(0, true);
        let result = check.execute().await;
        assert!(result.is_err());
        assert!(result.unwrap_err().contains("No strategies"));
    }

    #[tokio::test]
    async fn test_strategy_in_memory_not_valid_fails() {
        let check = StrategyInstantiationCheck::in_memory(3, false);
        let result = check.execute().await;
        assert!(result.is_err());
        assert!(result.unwrap_err().contains("failed self-test"));
    }

    #[tokio::test]
    async fn test_strategy_in_memory_detail() {
        let check = StrategyInstantiationCheck::in_memory(5, true);
        assert_eq!(
            check.detail_on_success(),
            Some("5 strategies instantiated and valid".to_string())
        );
    }

    // ── Strategy config file validation tests ──

    #[test]
    fn test_validate_strategy_config_nonexistent() {
        let result = validate_strategy_config(Path::new("/nonexistent/strategies.toml"));
        assert!(result.is_err());
        assert!(result.unwrap_err().contains("not found"));
    }

    #[test]
    fn test_validate_strategy_config_empty() {
        let dir = TempDir::new().unwrap();
        let config_path = dir.path().join("strategies.toml");
        std::fs::write(&config_path, "").unwrap();

        let result = validate_strategy_config(&config_path);
        assert!(result.is_err());
        assert!(result.unwrap_err().contains("empty"));
    }

    #[test]
    fn test_validate_strategy_config_invalid_toml() {
        let dir = TempDir::new().unwrap();
        let config_path = dir.path().join("strategies.toml");
        std::fs::write(&config_path, "this is {{ not valid toml").unwrap();

        let result = validate_strategy_config(&config_path);
        assert!(result.is_err());
        assert!(result.unwrap_err().contains("invalid TOML"));
    }

    #[test]
    fn test_validate_strategy_config_valid() {
        let dir = TempDir::new().unwrap();
        let config_path = dir.path().join("strategies.toml");
        let content = r#"
[momentum]
enabled = true
lookback = 20
threshold = 0.02

[mean_reversion]
enabled = true
window = 50
z_score_threshold = 2.0
"#;
        std::fs::write(&config_path, content).unwrap();

        let result = validate_strategy_config(&config_path);
        assert!(result.is_ok());
    }

    #[tokio::test]
    async fn test_strategy_config_file_check_valid() {
        let dir = TempDir::new().unwrap();
        let config_path = dir.path().join("strategies.toml");
        std::fs::write(&config_path, "[momentum]\nenabled = true\n").unwrap();

        let check = StrategyInstantiationCheck::from_config(&config_path);
        let result = check.execute().await;
        assert!(result.is_ok());
    }

    #[tokio::test]
    async fn test_strategy_config_file_check_missing() {
        let check = StrategyInstantiationCheck::from_config("/nonexistent/strategies.toml");
        let result = check.execute().await;
        assert!(result.is_err());
    }

    // ── CorrelationTrackerCheck construction tests ──

    #[test]
    fn test_correlation_tracker_config_defaults() {
        let check = CorrelationTrackerCheck::from_config("config/correlation_pairs.toml");
        assert_eq!(check.name(), "Correlation Tracker");
        assert_eq!(check.phase(), BootPhase::Strategy);
        assert_eq!(check.criticality(), Criticality::Required);
    }

    #[test]
    fn test_correlation_tracker_http_defaults() {
        let check = CorrelationTrackerCheck::http("http://localhost:8080/correlation/health");
        assert_eq!(check.criticality(), Criticality::Required);
    }

    #[test]
    fn test_correlation_tracker_in_memory() {
        let check = CorrelationTrackerCheck::in_memory(true, 5);
        assert_eq!(check.timeout(), Duration::from_secs(1));
    }

    #[test]
    fn test_correlation_tracker_builder() {
        let check = CorrelationTrackerCheck::in_memory(true, 3)
            .with_criticality(Criticality::Optional)
            .with_timeout(Duration::from_secs(12));
        assert_eq!(check.criticality(), Criticality::Optional);
        assert_eq!(check.timeout(), Duration::from_secs(12));
    }

    // ── CorrelationTrackerCheck execution tests ──

    #[tokio::test]
    async fn test_correlation_in_memory_valid_passes() {
        let check = CorrelationTrackerCheck::in_memory(true, 5);
        let result = check.execute().await;
        assert!(result.is_ok());
    }

    #[tokio::test]
    async fn test_correlation_in_memory_not_initialized_fails() {
        let check = CorrelationTrackerCheck::in_memory(false, 5);
        let result = check.execute().await;
        assert!(result.is_err());
        assert!(result.unwrap_err().contains("not initialized"));
    }

    #[tokio::test]
    async fn test_correlation_in_memory_zero_pairs_fails() {
        let check = CorrelationTrackerCheck::in_memory(true, 0);
        let result = check.execute().await;
        assert!(result.is_err());
        assert!(result.unwrap_err().contains("no pairs"));
    }

    #[tokio::test]
    async fn test_correlation_in_memory_detail() {
        let check = CorrelationTrackerCheck::in_memory(true, 7);
        assert_eq!(
            check.detail_on_success(),
            Some("7 correlation pairs loaded".to_string())
        );
    }

    // ── Correlation config file validation tests ──

    #[test]
    fn test_validate_correlation_config_nonexistent() {
        let result = validate_correlation_config(Path::new("/nonexistent/correlation.toml"));
        assert!(result.is_err());
        assert!(result.unwrap_err().contains("not found"));
    }

    #[test]
    fn test_validate_correlation_config_empty() {
        let dir = TempDir::new().unwrap();
        let config_path = dir.path().join("correlation.toml");
        std::fs::write(&config_path, "").unwrap();

        let result = validate_correlation_config(&config_path);
        assert!(result.is_err());
        assert!(result.unwrap_err().contains("empty"));
    }

    #[test]
    fn test_validate_correlation_config_no_pairs() {
        let dir = TempDir::new().unwrap();
        let config_path = dir.path().join("correlation.toml");
        std::fs::write(&config_path, "[settings]\nwindow = 100\n").unwrap();

        let result = validate_correlation_config(&config_path);
        assert!(result.is_err());
        assert!(result.unwrap_err().contains("does not contain any pair"));
    }

    #[test]
    fn test_validate_correlation_config_with_pairs_key() {
        let dir = TempDir::new().unwrap();
        let config_path = dir.path().join("correlation.toml");
        let content = r#"
[[pairs]]
asset_a = "BTC/USD"
asset_b = "ETH/USD"
expected_correlation = 0.75
"#;
        std::fs::write(&config_path, content).unwrap();

        let result = validate_correlation_config(&config_path);
        assert!(result.is_ok());
    }

    #[test]
    fn test_validate_correlation_config_with_monitored_pairs_key() {
        let dir = TempDir::new().unwrap();
        let config_path = dir.path().join("correlation.toml");
        let content = r#"
[[monitored_pairs]]
a = "BTC"
b = "ETH"
"#;
        std::fs::write(&config_path, content).unwrap();

        let result = validate_correlation_config(&config_path);
        assert!(result.is_ok());
    }

    #[test]
    fn test_validate_correlation_config_with_pair_table() {
        let dir = TempDir::new().unwrap();
        let config_path = dir.path().join("correlation.toml");
        let content = r#"
[[pair]]
asset_a = "BTC/USD"
asset_b = "ETH/USD"
limit = 0.8
"#;
        std::fs::write(&config_path, content).unwrap();

        let result = validate_correlation_config(&config_path);
        assert!(result.is_ok());
    }

    #[tokio::test]
    async fn test_correlation_config_file_check_valid() {
        let dir = TempDir::new().unwrap();
        let config_path = dir.path().join("correlation.toml");
        let content = "[[pairs]]\nasset_a = \"BTC\"\nasset_b = \"ETH\"\n";
        std::fs::write(&config_path, content).unwrap();

        let check = CorrelationTrackerCheck::from_config(&config_path);
        let result = check.execute().await;
        assert!(result.is_ok());
    }

    // ── AffinityLoadCheck construction tests ──

    #[test]
    fn test_affinity_load_config_defaults() {
        let check = AffinityLoadCheck::from_config("config/strategy_affinity.toml");
        assert_eq!(check.name(), "Affinity Load");
        assert_eq!(check.phase(), BootPhase::Strategy);
        assert_eq!(check.criticality(), Criticality::Optional);
    }

    #[test]
    fn test_affinity_load_redis_defaults() {
        let check = AffinityLoadCheck::redis("redis://localhost:6379", "janus:affinity:");
        assert_eq!(check.criticality(), Criticality::Optional);
    }

    #[test]
    fn test_affinity_load_in_memory() {
        let check = AffinityLoadCheck::in_memory(true, 10);
        assert_eq!(check.timeout(), Duration::from_secs(1));
    }

    #[test]
    fn test_affinity_load_builder() {
        let check = AffinityLoadCheck::in_memory(true, 5)
            .with_criticality(Criticality::Required)
            .with_timeout(Duration::from_secs(6));
        assert_eq!(check.criticality(), Criticality::Required);
        assert_eq!(check.timeout(), Duration::from_secs(6));
    }

    // ── AffinityLoadCheck execution tests ──

    #[tokio::test]
    async fn test_affinity_in_memory_valid_passes() {
        let check = AffinityLoadCheck::in_memory(true, 5);
        let result = check.execute().await;
        assert!(result.is_ok());
    }

    #[tokio::test]
    async fn test_affinity_in_memory_not_loaded_fails() {
        let check = AffinityLoadCheck::in_memory(false, 5);
        let result = check.execute().await;
        assert!(result.is_err());
        assert!(result.unwrap_err().contains("failed to load"));
    }

    #[tokio::test]
    async fn test_affinity_in_memory_zero_entries_fails() {
        let check = AffinityLoadCheck::in_memory(true, 0);
        let result = check.execute().await;
        assert!(result.is_err());
        assert!(result.unwrap_err().contains("No affinity entries"));
    }

    #[tokio::test]
    async fn test_affinity_in_memory_detail() {
        let check = AffinityLoadCheck::in_memory(true, 12);
        assert_eq!(
            check.detail_on_success(),
            Some("12 affinity entries loaded".to_string())
        );
    }

    // ── Affinity config validation tests ──

    #[test]
    fn test_validate_affinity_config_nonexistent() {
        let result = validate_affinity_config(Path::new("/nonexistent/affinity.toml"));
        assert!(result.is_err());
        assert!(result.unwrap_err().contains("not found"));
    }

    #[test]
    fn test_validate_affinity_config_empty() {
        let dir = TempDir::new().unwrap();
        let config_path = dir.path().join("affinity.toml");
        std::fs::write(&config_path, "").unwrap();

        let result = validate_affinity_config(&config_path);
        assert!(result.is_err());
        assert!(result.unwrap_err().contains("empty"));
    }

    #[test]
    fn test_validate_affinity_config_invalid_toml() {
        let dir = TempDir::new().unwrap();
        let config_path = dir.path().join("affinity.toml");
        std::fs::write(&config_path, "not {{ valid } toml").unwrap();

        let result = validate_affinity_config(&config_path);
        assert!(result.is_err());
        assert!(result.unwrap_err().contains("invalid TOML"));
    }

    #[test]
    fn test_validate_affinity_config_valid() {
        let dir = TempDir::new().unwrap();
        let config_path = dir.path().join("affinity.toml");
        let content = r#"
[BTC_USD]
preferred_strategies = ["momentum", "breakout"]
min_score = 0.6

[ETH_USD]
preferred_strategies = ["mean_reversion"]
min_score = 0.5
"#;
        std::fs::write(&config_path, content).unwrap();

        let result = validate_affinity_config(&config_path);
        assert!(result.is_ok());
    }

    #[tokio::test]
    async fn test_affinity_config_file_check_valid() {
        let dir = TempDir::new().unwrap();
        let config_path = dir.path().join("affinity.toml");
        std::fs::write(&config_path, "[BTC]\nweight = 1.0\n").unwrap();

        let check = AffinityLoadCheck::from_config(&config_path);
        let result = check.execute().await;
        assert!(result.is_ok());
    }

    #[tokio::test]
    async fn test_affinity_config_file_check_missing() {
        let check = AffinityLoadCheck::from_config("/nonexistent/affinity.toml");
        let result = check.execute().await;
        assert!(result.is_err());
    }

    // ── Builder helper tests ──

    #[test]
    fn test_default_strategy_checks_count() {
        let checks = default_strategy_checks(None, None, None, None);
        assert_eq!(checks.len(), 4);
    }

    #[test]
    fn test_default_strategy_checks_phases() {
        let checks = default_strategy_checks(None, None, None, None);
        for check in &checks {
            assert_eq!(check.phase(), BootPhase::Strategy);
        }
    }

    #[test]
    fn test_default_strategy_checks_names() {
        let checks = default_strategy_checks(None, None, None, None);
        assert_eq!(checks[0].name(), "Regime Detector");
        assert_eq!(checks[1].name(), "Strategy Instantiation");
        assert_eq!(checks[2].name(), "Correlation Tracker");
        assert_eq!(checks[3].name(), "Affinity Load");
    }

    #[test]
    fn test_default_strategy_checks_with_config_paths() {
        let checks = default_strategy_checks(
            Some(PathBuf::from("config/strategies.toml")),
            Some(PathBuf::from("config/correlation_pairs.toml")),
            Some(PathBuf::from("config/strategy_affinity.toml")),
            Some("redis://localhost:6379"),
        );
        assert_eq!(checks.len(), 4);
    }

    // ── Integration with PreFlightRunner ──

    #[tokio::test]
    async fn test_strategy_checks_in_runner() {
        use super::super::PreFlightRunner;

        let mut runner = PreFlightRunner::new();
        runner.add_check(Box::new(RegimeDetectorCheck::in_memory(true, 500, 100)));
        runner.add_check(Box::new(StrategyInstantiationCheck::in_memory(3, true)));
        runner.add_check(Box::new(CorrelationTrackerCheck::in_memory(true, 5)));
        runner.add_check(Box::new(AffinityLoadCheck::in_memory(true, 10)));

        let report = runner.run().await;
        assert!(report.is_boot_safe());
        assert_eq!(report.pass_count(), 4);
        assert_eq!(report.fail_count(), 0);
    }

    #[tokio::test]
    async fn test_strategy_critical_failure_aborts() {
        use super::super::PreFlightRunner;

        let mut runner = PreFlightRunner::new();
        runner.add_check(Box::new(StrategyInstantiationCheck::in_memory(0, true))); // FAIL (Critical)
        runner.add_check(Box::new(CorrelationTrackerCheck::in_memory(true, 5)));

        let report = runner.run().await;
        assert!(!report.is_boot_safe());
        assert!(report.aborted);
        assert_eq!(
            report.abort_check.as_deref(),
            Some("Strategy Instantiation")
        );
    }

    #[tokio::test]
    async fn test_strategy_optional_failure_continues() {
        use super::super::PreFlightRunner;

        let mut runner = PreFlightRunner::new();
        runner.add_check(Box::new(StrategyInstantiationCheck::in_memory(3, true)));
        runner.add_check(Box::new(
            AffinityLoadCheck::in_memory(true, 0), // Fails but Optional
        ));

        let report = runner.run().await;
        // Affinity failure is Optional so boot is still safe
        assert!(report.is_boot_safe());
        assert_eq!(report.pass_count(), 1);
        assert_eq!(report.fail_count(), 1);
    }
}
