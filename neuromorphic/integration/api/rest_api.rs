//! REST API endpoint registry
//!
//! Part of the Integration region — API component.
//!
//! `RestApi` provides a registry of REST endpoints with route definitions,
//! request tracking, per-endpoint rate limiting and latency monitoring,
//! EMA-smoothed throughput metrics, and windowed diagnostics.

use crate::common::{Error, Result};
use std::collections::{HashMap, VecDeque};

// ---------------------------------------------------------------------------
// Configuration
// ---------------------------------------------------------------------------

/// Configuration for the REST API manager.
#[derive(Debug, Clone)]
pub struct RestApiConfig {
    /// Maximum number of endpoints that can be registered.
    pub max_endpoints: usize,
    /// Default rate limit (requests per window) for new endpoints.
    /// 0 means unlimited.
    pub default_rate_limit: u64,
    /// Rate-limit window size in ticks.
    pub rate_limit_window_ticks: u64,
    /// Latency threshold (µs) above which a request is considered slow.
    pub slow_request_threshold_us: f64,
    /// Maximum number of request records retained per endpoint.
    pub max_history_per_endpoint: usize,
    /// EMA decay factor (0 < α < 1). Higher → faster adaptation.
    pub ema_decay: f64,
    /// Window size for rolling diagnostics.
    pub window_size: usize,
}

impl Default for RestApiConfig {
    fn default() -> Self {
        Self {
            max_endpoints: 128,
            default_rate_limit: 0,
            rate_limit_window_ticks: 60,
            slow_request_threshold_us: 10000.0,
            max_history_per_endpoint: 50,
            ema_decay: 0.1,
            window_size: 50,
        }
    }
}

// ---------------------------------------------------------------------------
// HTTP method
// ---------------------------------------------------------------------------

/// Supported HTTP methods.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash)]
pub enum HttpMethod {
    Get,
    Post,
    Put,
    Patch,
    Delete,
    Options,
    Head,
}

impl std::fmt::Display for HttpMethod {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        match self {
            HttpMethod::Get => write!(f, "GET"),
            HttpMethod::Post => write!(f, "POST"),
            HttpMethod::Put => write!(f, "PUT"),
            HttpMethod::Patch => write!(f, "PATCH"),
            HttpMethod::Delete => write!(f, "DELETE"),
            HttpMethod::Options => write!(f, "OPTIONS"),
            HttpMethod::Head => write!(f, "HEAD"),
        }
    }
}

// ---------------------------------------------------------------------------
// Status code category
// ---------------------------------------------------------------------------

/// Coarse HTTP status-code category.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash)]
pub enum StatusCategory {
    /// 2xx success.
    Success,
    /// 3xx redirection.
    Redirect,
    /// 4xx client error.
    ClientError,
    /// 5xx server error.
    ServerError,
    /// Rate-limited (429).
    RateLimited,
}

impl StatusCategory {
    /// Map a numeric HTTP status code to a category.
    pub fn from_code(code: u16) -> Self {
        match code {
            200..=299 => StatusCategory::Success,
            300..=399 => StatusCategory::Redirect,
            429 => StatusCategory::RateLimited,
            400..=499 => StatusCategory::ClientError,
            500..=599 => StatusCategory::ServerError,
            _ => StatusCategory::ClientError,
        }
    }
}

// ---------------------------------------------------------------------------
// Request record
// ---------------------------------------------------------------------------

/// Record of a single API request.
#[derive(Debug, Clone)]
pub struct RequestRecord {
    /// Tick at which the request was received.
    pub tick: u64,
    /// HTTP method.
    pub method: HttpMethod,
    /// Endpoint path.
    pub path: String,
    /// Response status category.
    pub status: StatusCategory,
    /// Request-to-response latency in µs.
    pub latency_us: f64,
    /// Whether this request was rate-limited.
    pub rate_limited: bool,
}

// ---------------------------------------------------------------------------
// Endpoint definition
// ---------------------------------------------------------------------------

/// A registered REST endpoint.
#[derive(Debug, Clone)]
pub struct Endpoint {
    /// Route path pattern (e.g. `/api/v1/signals`).
    pub path: String,
    /// Allowed HTTP methods.
    pub methods: Vec<HttpMethod>,
    /// Human-readable description.
    pub description: String,
    /// Whether the endpoint is currently enabled.
    pub enabled: bool,
    /// Rate limit (requests per window). 0 = unlimited.
    pub rate_limit: u64,
    /// Requests received in the current rate-limit window.
    pub window_request_count: u64,
    /// Tick at which the current rate-limit window started.
    pub window_start_tick: u64,
    /// Total requests served.
    pub total_requests: u64,
    /// Total successful (2xx) responses.
    pub total_success: u64,
    /// Total client errors (4xx).
    pub total_client_errors: u64,
    /// Total server errors (5xx).
    pub total_server_errors: u64,
    /// Total rate-limited responses.
    pub total_rate_limited: u64,
    /// Last recorded latency in µs.
    pub last_latency_us: f64,
    /// EMA-smoothed latency.
    pub ema_latency_us: f64,
    /// Recent request history.
    history: VecDeque<RequestRecord>,
    /// Maximum history length.
    max_history: usize,
}

impl Endpoint {
    fn new(
        path: impl Into<String>,
        methods: Vec<HttpMethod>,
        description: impl Into<String>,
        rate_limit: u64,
        max_history: usize,
        current_tick: u64,
    ) -> Self {
        Self {
            path: path.into(),
            methods,
            description: description.into(),
            enabled: true,
            rate_limit,
            window_request_count: 0,
            window_start_tick: current_tick,
            total_requests: 0,
            total_success: 0,
            total_client_errors: 0,
            total_server_errors: 0,
            total_rate_limited: 0,
            last_latency_us: 0.0,
            ema_latency_us: 0.0,
            history: VecDeque::with_capacity(max_history.min(128)),
            max_history,
        }
    }

    /// Success rate across all requests.
    pub fn success_rate(&self) -> f64 {
        if self.total_requests == 0 {
            return 0.0;
        }
        self.total_success as f64 / self.total_requests as f64
    }

    /// Returns the recent request history.
    pub fn history(&self) -> &VecDeque<RequestRecord> {
        &self.history
    }
}

// ---------------------------------------------------------------------------
// Tick snapshot (windowed diagnostics)
// ---------------------------------------------------------------------------

#[derive(Debug, Clone)]
struct TickSnapshot {
    total_requests: u64,
    requests_this_tick: u64,
    success_ratio: f64,
    avg_latency_us: f64,
    slow_request_count: u64,
    active_endpoints: usize,
}

// ---------------------------------------------------------------------------
// Statistics
// ---------------------------------------------------------------------------

/// Cumulative statistics for the REST API.
#[derive(Debug, Clone)]
pub struct RestApiStats {
    /// Total requests across all endpoints.
    pub total_requests: u64,
    /// Total successful responses.
    pub total_success: u64,
    /// Total client errors.
    pub total_client_errors: u64,
    /// Total server errors.
    pub total_server_errors: u64,
    /// Total rate-limited responses.
    pub total_rate_limited: u64,
    /// Total slow requests (latency > threshold).
    pub total_slow_requests: u64,
    /// EMA-smoothed request throughput (requests per tick).
    pub ema_throughput: f64,
    /// EMA-smoothed success ratio.
    pub ema_success_ratio: f64,
    /// EMA-smoothed average latency.
    pub ema_avg_latency_us: f64,
}

impl Default for RestApiStats {
    fn default() -> Self {
        Self {
            total_requests: 0,
            total_success: 0,
            total_client_errors: 0,
            total_server_errors: 0,
            total_rate_limited: 0,
            total_slow_requests: 0,
            ema_throughput: 0.0,
            ema_success_ratio: 1.0,
            ema_avg_latency_us: 0.0,
        }
    }
}

// ---------------------------------------------------------------------------
// RestApi
// ---------------------------------------------------------------------------

/// REST API endpoint registry and request tracker.
///
/// Manages a set of named REST endpoints, tracks per-endpoint request
/// counts, latencies, rate limits, and provides EMA + windowed diagnostics.
pub struct RestApi {
    config: RestApiConfig,
    /// Registered endpoints keyed by path.
    endpoints: HashMap<String, Endpoint>,
    /// Insertion-order tracking.
    endpoint_order: Vec<String>,
    /// Current tick counter.
    tick: u64,
    /// Requests processed in the current tick.
    requests_this_tick: u64,
    /// Sum of latencies in the current tick (for per-tick average).
    latency_sum_this_tick: f64,
    /// Successes in the current tick.
    success_this_tick: u64,
    /// Slow requests in the current tick.
    slow_this_tick: u64,
    /// Whether EMA has been initialised.
    ema_initialized: bool,
    /// Rolling window of tick snapshots.
    recent: VecDeque<TickSnapshot>,
    /// Cumulative statistics.
    stats: RestApiStats,
}

impl Default for RestApi {
    fn default() -> Self {
        Self::new()
    }
}

impl RestApi {
    // -------------------------------------------------------------------
    // Construction
    // -------------------------------------------------------------------

    /// Create a new REST API manager with default configuration.
    pub fn new() -> Self {
        Self::with_config(RestApiConfig::default())
    }

    /// Create a new REST API manager with the given configuration.
    pub fn with_config(config: RestApiConfig) -> Self {
        Self {
            endpoints: HashMap::new(),
            endpoint_order: Vec::new(),
            tick: 0,
            requests_this_tick: 0,
            latency_sum_this_tick: 0.0,
            success_this_tick: 0,
            slow_this_tick: 0,
            ema_initialized: false,
            recent: VecDeque::with_capacity(config.window_size.min(256)),
            stats: RestApiStats::default(),
            config,
        }
    }

    // -------------------------------------------------------------------
    // Endpoint management
    // -------------------------------------------------------------------

    /// Register a new endpoint. Returns an error if the path already exists
    /// or the maximum endpoint count is reached.
    pub fn register_endpoint(
        &mut self,
        path: impl Into<String>,
        methods: Vec<HttpMethod>,
        description: impl Into<String>,
    ) -> Result<()> {
        let path = path.into();
        if self.endpoints.contains_key(&path) {
            return Err(Error::Configuration(format!(
                "Endpoint '{}' already registered",
                path
            )));
        }
        if self.endpoints.len() >= self.config.max_endpoints {
            return Err(Error::Configuration(format!(
                "Maximum endpoint count ({}) reached",
                self.config.max_endpoints
            )));
        }
        let rate_limit = self.config.default_rate_limit;
        let max_hist = self.config.max_history_per_endpoint;
        let tick = self.tick;
        self.endpoints.insert(
            path.clone(),
            Endpoint::new(&path, methods, description, rate_limit, max_hist, tick),
        );
        self.endpoint_order.push(path);
        Ok(())
    }

    /// Register an endpoint with a custom rate limit.
    pub fn register_endpoint_with_rate_limit(
        &mut self,
        path: impl Into<String>,
        methods: Vec<HttpMethod>,
        description: impl Into<String>,
        rate_limit: u64,
    ) -> Result<()> {
        let path = path.into();
        if self.endpoints.contains_key(&path) {
            return Err(Error::Configuration(format!(
                "Endpoint '{}' already registered",
                path
            )));
        }
        if self.endpoints.len() >= self.config.max_endpoints {
            return Err(Error::Configuration(format!(
                "Maximum endpoint count ({}) reached",
                self.config.max_endpoints
            )));
        }
        let max_hist = self.config.max_history_per_endpoint;
        let tick = self.tick;
        self.endpoints.insert(
            path.clone(),
            Endpoint::new(&path, methods, description, rate_limit, max_hist, tick),
        );
        self.endpoint_order.push(path);
        Ok(())
    }

    /// Enable an endpoint.
    pub fn enable_endpoint(&mut self, path: &str) -> Result<()> {
        let ep = self
            .endpoints
            .get_mut(path)
            .ok_or_else(|| Error::Configuration(format!("Unknown endpoint '{}'", path)))?;
        ep.enabled = true;
        Ok(())
    }

    /// Disable an endpoint.
    pub fn disable_endpoint(&mut self, path: &str) -> Result<()> {
        let ep = self
            .endpoints
            .get_mut(path)
            .ok_or_else(|| Error::Configuration(format!("Unknown endpoint '{}'", path)))?;
        ep.enabled = false;
        Ok(())
    }

    /// Look up an endpoint by path.
    pub fn endpoint(&self, path: &str) -> Option<&Endpoint> {
        self.endpoints.get(path)
    }

    /// Number of registered endpoints.
    pub fn endpoint_count(&self) -> usize {
        self.endpoints.len()
    }

    /// Paths of all registered endpoints in insertion order.
    pub fn endpoint_paths(&self) -> Vec<&str> {
        self.endpoint_order.iter().map(|s| s.as_str()).collect()
    }

    /// Number of currently enabled endpoints.
    pub fn active_endpoint_count(&self) -> usize {
        self.endpoints.values().filter(|e| e.enabled).count()
    }

    // -------------------------------------------------------------------
    // Request handling
    // -------------------------------------------------------------------

    /// Check whether a request to the given endpoint would be rate-limited.
    pub fn is_rate_limited(&self, path: &str) -> bool {
        if let Some(ep) = self.endpoints.get(path) {
            if ep.rate_limit == 0 {
                return false;
            }
            // Check if window has expired.
            let window_elapsed = self.tick.saturating_sub(ep.window_start_tick);
            if window_elapsed >= self.config.rate_limit_window_ticks {
                return false; // Window reset.
            }
            ep.window_request_count >= ep.rate_limit
        } else {
            false
        }
    }

    /// Record a request to an endpoint.
    ///
    /// If the endpoint is disabled, returns an error. If rate-limited, records
    /// the request as rate-limited (status 429). Otherwise records normally.
    pub fn handle_request(
        &mut self,
        path: &str,
        method: HttpMethod,
        latency_us: f64,
        status_code: u16,
    ) -> Result<RequestRecord> {
        if !self.endpoints.contains_key(path) {
            return Err(Error::Configuration(format!(
                "Unknown endpoint '{}'",
                path
            )));
        }

        // Check enabled.
        let enabled = self.endpoints[path].enabled;
        if !enabled {
            return Err(Error::Configuration(format!(
                "Endpoint '{}' is disabled",
                path
            )));
        }

        // Check method allowed.
        let method_allowed = self.endpoints[path].methods.contains(&method);
        if !method_allowed {
            return Err(Error::Configuration(format!(
                "Method {} not allowed for endpoint '{}'",
                method, path
            )));
        }

        let window_ticks = self.config.rate_limit_window_ticks;
        let threshold = self.config.slow_request_threshold_us;
        let decay = self.config.ema_decay;

        let ep = self.endpoints.get_mut(path).unwrap();

        // Rate-limit window management.
        let window_elapsed = self.tick.saturating_sub(ep.window_start_tick);
        if window_elapsed >= window_ticks {
            ep.window_request_count = 0;
            ep.window_start_tick = self.tick;
        }

        // Check rate limit.
        let rate_limited = ep.rate_limit > 0 && ep.window_request_count >= ep.rate_limit;

        let actual_status = if rate_limited {
            StatusCategory::RateLimited
        } else {
            StatusCategory::from_code(status_code)
        };

        // Update endpoint stats.
        ep.total_requests += 1;
        ep.window_request_count += 1;
        ep.last_latency_us = latency_us;

        if ep.total_requests == 1 {
            ep.ema_latency_us = latency_us;
        } else {
            ep.ema_latency_us = decay * latency_us + (1.0 - decay) * ep.ema_latency_us;
        }

        match actual_status {
            StatusCategory::Success => ep.total_success += 1,
            StatusCategory::ClientError => ep.total_client_errors += 1,
            StatusCategory::ServerError => ep.total_server_errors += 1,
            StatusCategory::RateLimited => ep.total_rate_limited += 1,
            _ => {}
        }

        let record = RequestRecord {
            tick: self.tick,
            method,
            path: path.to_string(),
            status: actual_status,
            latency_us,
            rate_limited,
        };

        if ep.history.len() >= ep.max_history {
            ep.history.pop_front();
        }
        ep.history.push_back(record.clone());

        // Update global stats.
        self.stats.total_requests += 1;
        self.requests_this_tick += 1;
        self.latency_sum_this_tick += latency_us;

        match actual_status {
            StatusCategory::Success => {
                self.stats.total_success += 1;
                self.success_this_tick += 1;
            }
            StatusCategory::ClientError => {
                self.stats.total_client_errors += 1;
            }
            StatusCategory::ServerError => {
                self.stats.total_server_errors += 1;
            }
            StatusCategory::RateLimited => {
                self.stats.total_rate_limited += 1;
            }
            _ => {}
        }

        if latency_us > threshold {
            self.stats.total_slow_requests += 1;
            self.slow_this_tick += 1;
        }

        Ok(record)
    }

    // -------------------------------------------------------------------
    // Tick
    // -------------------------------------------------------------------

    /// Advance the API manager by one tick, updating EMA and windowed
    /// diagnostics.
    pub fn tick(&mut self) {
        self.tick += 1;

        let throughput = self.requests_this_tick as f64;
        let success_ratio = if self.requests_this_tick > 0 {
            self.success_this_tick as f64 / self.requests_this_tick as f64
        } else {
            1.0
        };
        let avg_latency = if self.requests_this_tick > 0 {
            self.latency_sum_this_tick / self.requests_this_tick as f64
        } else {
            0.0
        };

        // EMA update.
        let alpha = self.config.ema_decay;
        if !self.ema_initialized && self.requests_this_tick > 0 {
            self.stats.ema_throughput = throughput;
            self.stats.ema_success_ratio = success_ratio;
            self.stats.ema_avg_latency_us = avg_latency;
            self.ema_initialized = true;
        } else if self.ema_initialized {
            self.stats.ema_throughput =
                alpha * throughput + (1.0 - alpha) * self.stats.ema_throughput;
            self.stats.ema_success_ratio =
                alpha * success_ratio + (1.0 - alpha) * self.stats.ema_success_ratio;
            self.stats.ema_avg_latency_us =
                alpha * avg_latency + (1.0 - alpha) * self.stats.ema_avg_latency_us;
        }

        // Windowed snapshot.
        let snapshot = TickSnapshot {
            total_requests: self.stats.total_requests,
            requests_this_tick: self.requests_this_tick,
            success_ratio,
            avg_latency_us: avg_latency,
            slow_request_count: self.slow_this_tick,
            active_endpoints: self.active_endpoint_count(),
        };
        if self.recent.len() >= self.config.window_size {
            self.recent.pop_front();
        }
        self.recent.push_back(snapshot);

        // Reset per-tick counters.
        self.requests_this_tick = 0;
        self.latency_sum_this_tick = 0.0;
        self.success_this_tick = 0;
        self.slow_this_tick = 0;
    }

    /// Current tick counter.
    pub fn current_tick(&self) -> u64 {
        self.tick
    }

    /// Alias for `tick()`.
    pub fn process(&mut self) {
        self.tick();
    }

    // -------------------------------------------------------------------
    // Diagnostics
    // -------------------------------------------------------------------

    /// Returns a reference to cumulative statistics.
    pub fn stats(&self) -> &RestApiStats {
        &self.stats
    }

    /// Returns a reference to the configuration.
    pub fn config(&self) -> &RestApiConfig {
        &self.config
    }

    /// EMA-smoothed request throughput (requests per tick).
    pub fn smoothed_throughput(&self) -> f64 {
        self.stats.ema_throughput
    }

    /// EMA-smoothed success ratio.
    pub fn smoothed_success_ratio(&self) -> f64 {
        self.stats.ema_success_ratio
    }

    /// EMA-smoothed average latency.
    pub fn smoothed_avg_latency(&self) -> f64 {
        self.stats.ema_avg_latency_us
    }

    /// Overall success rate (total_success / total_requests).
    pub fn overall_success_rate(&self) -> f64 {
        if self.stats.total_requests == 0 {
            return 0.0;
        }
        self.stats.total_success as f64 / self.stats.total_requests as f64
    }

    /// Windowed average throughput.
    pub fn windowed_throughput(&self) -> Option<f64> {
        if self.recent.is_empty() {
            return None;
        }
        let sum: f64 = self.recent.iter().map(|s| s.requests_this_tick as f64).sum();
        Some(sum / self.recent.len() as f64)
    }

    /// Windowed average success ratio.
    pub fn windowed_success_ratio(&self) -> Option<f64> {
        if self.recent.is_empty() {
            return None;
        }
        let sum: f64 = self.recent.iter().map(|s| s.success_ratio).sum();
        Some(sum / self.recent.len() as f64)
    }

    /// Windowed average latency.
    pub fn windowed_avg_latency(&self) -> Option<f64> {
        if self.recent.is_empty() {
            return None;
        }
        let sum: f64 = self.recent.iter().map(|s| s.avg_latency_us).sum();
        Some(sum / self.recent.len() as f64)
    }

    /// Windowed average slow-request count per tick.
    pub fn windowed_slow_request_count(&self) -> Option<f64> {
        if self.recent.is_empty() {
            return None;
        }
        let sum: f64 = self
            .recent
            .iter()
            .map(|s| s.slow_request_count as f64)
            .sum();
        Some(sum / self.recent.len() as f64)
    }

    /// Whether latency appears to be increasing over the window.
    pub fn is_latency_increasing(&self) -> bool {
        if self.recent.len() < 4 {
            return false;
        }
        let mid = self.recent.len() / 2;
        let first_half: f64 =
            self.recent.iter().take(mid).map(|s| s.avg_latency_us).sum::<f64>() / mid as f64;
        let second_half: f64 = self
            .recent
            .iter()
            .skip(mid)
            .map(|s| s.avg_latency_us)
            .sum::<f64>()
            / (self.recent.len() - mid) as f64;
        second_half > first_half * 1.05
    }

    /// Whether success ratio appears to be declining over the window.
    pub fn is_success_declining(&self) -> bool {
        if self.recent.len() < 4 {
            return false;
        }
        let mid = self.recent.len() / 2;
        let first_half: f64 =
            self.recent.iter().take(mid).map(|s| s.success_ratio).sum::<f64>() / mid as f64;
        let second_half: f64 = self
            .recent
            .iter()
            .skip(mid)
            .map(|s| s.success_ratio)
            .sum::<f64>()
            / (self.recent.len() - mid) as f64;
        second_half < first_half - 0.05
    }

    // -------------------------------------------------------------------
    // Reset
    // -------------------------------------------------------------------

    /// Reset all state, preserving configuration and registered endpoints
    /// (which are also reset).
    pub fn reset(&mut self) {
        self.tick = 0;
        self.requests_this_tick = 0;
        self.latency_sum_this_tick = 0.0;
        self.success_this_tick = 0;
        self.slow_this_tick = 0;
        self.ema_initialized = false;
        self.recent.clear();
        self.stats = RestApiStats::default();
        for ep in self.endpoints.values_mut() {
            ep.total_requests = 0;
            ep.total_success = 0;
            ep.total_client_errors = 0;
            ep.total_server_errors = 0;
            ep.total_rate_limited = 0;
            ep.window_request_count = 0;
            ep.window_start_tick = 0;
            ep.last_latency_us = 0.0;
            ep.ema_latency_us = 0.0;
            ep.history.clear();
        }
    }
}

// ===========================================================================
// Tests
// ===========================================================================

#[cfg(test)]
mod tests {
    use super::*;

    fn small_config() -> RestApiConfig {
        RestApiConfig {
            max_endpoints: 8,
            default_rate_limit: 0,
            rate_limit_window_ticks: 5,
            slow_request_threshold_us: 100.0,
            max_history_per_endpoint: 5,
            ema_decay: 0.5,
            window_size: 5,
        }
    }

    // -------------------------------------------------------------------
    // Construction
    // -------------------------------------------------------------------

    #[test]
    fn test_new_default() {
        let api = RestApi::new();
        assert_eq!(api.endpoint_count(), 0);
        assert_eq!(api.current_tick(), 0);
    }

    #[test]
    fn test_with_config() {
        let api = RestApi::with_config(small_config());
        assert_eq!(api.config().max_endpoints, 8);
    }

    // -------------------------------------------------------------------
    // Endpoint registration
    // -------------------------------------------------------------------

    #[test]
    fn test_register_endpoint() {
        let mut api = RestApi::with_config(small_config());
        api.register_endpoint("/api/v1/signals", vec![HttpMethod::Get], "Get signals")
            .unwrap();
        assert_eq!(api.endpoint_count(), 1);
        assert!(api.endpoint("/api/v1/signals").is_some());
        let ep = api.endpoint("/api/v1/signals").unwrap();
        assert!(ep.enabled);
        assert_eq!(ep.methods, vec![HttpMethod::Get]);
    }

    #[test]
    fn test_register_duplicate_fails() {
        let mut api = RestApi::with_config(small_config());
        api.register_endpoint("/a", vec![HttpMethod::Get], "").unwrap();
        assert!(api.register_endpoint("/a", vec![HttpMethod::Get], "").is_err());
    }

    #[test]
    fn test_register_at_capacity() {
        let mut api = RestApi::with_config(small_config());
        for i in 0..8 {
            api.register_endpoint(format!("/e{}", i), vec![HttpMethod::Get], "")
                .unwrap();
        }
        assert!(api
            .register_endpoint("/overflow", vec![HttpMethod::Get], "")
            .is_err());
    }

    #[test]
    fn test_register_with_rate_limit() {
        let mut api = RestApi::with_config(small_config());
        api.register_endpoint_with_rate_limit(
            "/limited",
            vec![HttpMethod::Get],
            "Rate limited",
            10,
        )
        .unwrap();
        assert_eq!(api.endpoint("/limited").unwrap().rate_limit, 10);
    }

    // -------------------------------------------------------------------
    // Enable / disable
    // -------------------------------------------------------------------

    #[test]
    fn test_enable_disable() {
        let mut api = RestApi::with_config(small_config());
        api.register_endpoint("/a", vec![HttpMethod::Get], "").unwrap();
        api.disable_endpoint("/a").unwrap();
        assert!(!api.endpoint("/a").unwrap().enabled);
        api.enable_endpoint("/a").unwrap();
        assert!(api.endpoint("/a").unwrap().enabled);
    }

    #[test]
    fn test_disable_unknown_fails() {
        let mut api = RestApi::with_config(small_config());
        assert!(api.disable_endpoint("/nope").is_err());
    }

    #[test]
    fn test_endpoint_paths_order() {
        let mut api = RestApi::with_config(small_config());
        api.register_endpoint("/b", vec![HttpMethod::Get], "").unwrap();
        api.register_endpoint("/a", vec![HttpMethod::Get], "").unwrap();
        assert_eq!(api.endpoint_paths(), vec!["/b", "/a"]);
    }

    #[test]
    fn test_active_endpoint_count() {
        let mut api = RestApi::with_config(small_config());
        api.register_endpoint("/a", vec![HttpMethod::Get], "").unwrap();
        api.register_endpoint("/b", vec![HttpMethod::Get], "").unwrap();
        assert_eq!(api.active_endpoint_count(), 2);
        api.disable_endpoint("/b").unwrap();
        assert_eq!(api.active_endpoint_count(), 1);
    }

    // -------------------------------------------------------------------
    // Request handling
    // -------------------------------------------------------------------

    #[test]
    fn test_handle_request_success() {
        let mut api = RestApi::with_config(small_config());
        api.register_endpoint("/a", vec![HttpMethod::Get], "").unwrap();
        let record = api.handle_request("/a", HttpMethod::Get, 50.0, 200).unwrap();
        assert_eq!(record.status, StatusCategory::Success);
        assert!(!record.rate_limited);
        assert!((record.latency_us - 50.0).abs() < 1e-9);
        assert_eq!(api.endpoint("/a").unwrap().total_requests, 1);
        assert_eq!(api.endpoint("/a").unwrap().total_success, 1);
        assert_eq!(api.stats().total_requests, 1);
        assert_eq!(api.stats().total_success, 1);
    }

    #[test]
    fn test_handle_request_client_error() {
        let mut api = RestApi::with_config(small_config());
        api.register_endpoint("/a", vec![HttpMethod::Get], "").unwrap();
        let record = api.handle_request("/a", HttpMethod::Get, 10.0, 404).unwrap();
        assert_eq!(record.status, StatusCategory::ClientError);
        assert_eq!(api.endpoint("/a").unwrap().total_client_errors, 1);
        assert_eq!(api.stats().total_client_errors, 1);
    }

    #[test]
    fn test_handle_request_server_error() {
        let mut api = RestApi::with_config(small_config());
        api.register_endpoint("/a", vec![HttpMethod::Get], "").unwrap();
        let record = api.handle_request("/a", HttpMethod::Get, 10.0, 500).unwrap();
        assert_eq!(record.status, StatusCategory::ServerError);
        assert_eq!(api.stats().total_server_errors, 1);
    }

    #[test]
    fn test_handle_request_unknown_endpoint() {
        let mut api = RestApi::with_config(small_config());
        assert!(api.handle_request("/nope", HttpMethod::Get, 10.0, 200).is_err());
    }

    #[test]
    fn test_handle_request_disabled_endpoint() {
        let mut api = RestApi::with_config(small_config());
        api.register_endpoint("/a", vec![HttpMethod::Get], "").unwrap();
        api.disable_endpoint("/a").unwrap();
        assert!(api.handle_request("/a", HttpMethod::Get, 10.0, 200).is_err());
    }

    #[test]
    fn test_handle_request_method_not_allowed() {
        let mut api = RestApi::with_config(small_config());
        api.register_endpoint("/a", vec![HttpMethod::Get], "").unwrap();
        assert!(api.handle_request("/a", HttpMethod::Post, 10.0, 200).is_err());
    }

    #[test]
    fn test_handle_request_slow() {
        let mut api = RestApi::with_config(small_config()); // threshold=100
        api.register_endpoint("/a", vec![HttpMethod::Get], "").unwrap();
        api.handle_request("/a", HttpMethod::Get, 200.0, 200).unwrap();
        assert_eq!(api.stats().total_slow_requests, 1);
    }

    // -------------------------------------------------------------------
    // Rate limiting
    // -------------------------------------------------------------------

    #[test]
    fn test_rate_limiting() {
        let mut api = RestApi::with_config(small_config());
        api.register_endpoint_with_rate_limit("/rl", vec![HttpMethod::Get], "", 2)
            .unwrap();

        // First two requests: OK.
        api.handle_request("/rl", HttpMethod::Get, 10.0, 200).unwrap();
        api.handle_request("/rl", HttpMethod::Get, 10.0, 200).unwrap();

        // Third request: rate limited.
        assert!(api.is_rate_limited("/rl"));
        let record = api.handle_request("/rl", HttpMethod::Get, 10.0, 200).unwrap();
        assert!(record.rate_limited);
        assert_eq!(record.status, StatusCategory::RateLimited);
        assert_eq!(api.endpoint("/rl").unwrap().total_rate_limited, 1);
        assert_eq!(api.stats().total_rate_limited, 1);
    }

    #[test]
    fn test_rate_limit_window_reset() {
        let mut api = RestApi::with_config(small_config()); // rate_limit_window_ticks=5
        api.register_endpoint_with_rate_limit("/rl", vec![HttpMethod::Get], "", 1)
            .unwrap();

        api.handle_request("/rl", HttpMethod::Get, 10.0, 200).unwrap();
        assert!(api.is_rate_limited("/rl"));

        // Advance past the window.
        for _ in 0..5 {
            api.tick();
        }
        assert!(!api.is_rate_limited("/rl"));

        // Should succeed again.
        let record = api.handle_request("/rl", HttpMethod::Get, 10.0, 200).unwrap();
        assert!(!record.rate_limited);
    }

    #[test]
    fn test_is_rate_limited_no_limit() {
        let mut api = RestApi::with_config(small_config());
        api.register_endpoint("/a", vec![HttpMethod::Get], "").unwrap();
        // rate_limit = 0 (default) means unlimited.
        for _ in 0..100 {
            api.handle_request("/a", HttpMethod::Get, 10.0, 200).unwrap();
        }
        assert!(!api.is_rate_limited("/a"));
    }

    #[test]
    fn test_is_rate_limited_unknown() {
        let api = RestApi::with_config(small_config());
        assert!(!api.is_rate_limited("/nope"));
    }

    // -------------------------------------------------------------------
    // Per-endpoint EMA latency
    // -------------------------------------------------------------------

    #[test]
    fn test_endpoint_ema_latency_init() {
        let mut api = RestApi::with_config(small_config());
        api.register_endpoint("/a", vec![HttpMethod::Get], "").unwrap();
        api.handle_request("/a", HttpMethod::Get, 100.0, 200).unwrap();
        assert!((api.endpoint("/a").unwrap().ema_latency_us - 100.0).abs() < 1e-9);
    }

    #[test]
    fn test_endpoint_ema_latency_blends() {
        let mut api = RestApi::with_config(small_config()); // ema_decay=0.5
        api.register_endpoint("/a", vec![HttpMethod::Get], "").unwrap();
        api.handle_request("/a", HttpMethod::Get, 100.0, 200).unwrap();
        api.handle_request("/a", HttpMethod::Get, 200.0, 200).unwrap();
        // EMA = 0.5*200 + 0.5*100 = 150
        assert!((api.endpoint("/a").unwrap().ema_latency_us - 150.0).abs() < 1e-9);
    }

    // -------------------------------------------------------------------
    // Endpoint success rate
    // -------------------------------------------------------------------

    #[test]
    fn test_endpoint_success_rate() {
        let mut api = RestApi::with_config(small_config());
        api.register_endpoint("/a", vec![HttpMethod::Get], "").unwrap();
        api.handle_request("/a", HttpMethod::Get, 10.0, 200).unwrap();
        api.handle_request("/a", HttpMethod::Get, 10.0, 500).unwrap();
        assert!((api.endpoint("/a").unwrap().success_rate() - 0.5).abs() < 1e-9);
    }

    #[test]
    fn test_endpoint_success_rate_empty() {
        let mut api = RestApi::with_config(small_config());
        api.register_endpoint("/a", vec![HttpMethod::Get], "").unwrap();
        assert!((api.endpoint("/a").unwrap().success_rate() - 0.0).abs() < 1e-9);
    }

    // -------------------------------------------------------------------
    // Per-endpoint history
    // -------------------------------------------------------------------

    #[test]
    fn test_endpoint_history() {
        let mut api = RestApi::with_config(small_config());
        api.register_endpoint("/a", vec![HttpMethod::Get], "").unwrap();
        api.handle_request("/a", HttpMethod::Get, 10.0, 200).unwrap();
        api.handle_request("/a", HttpMethod::Get, 20.0, 404).unwrap();
        let history = api.endpoint("/a").unwrap().history();
        assert_eq!(history.len(), 2);
        assert_eq!(history[0].status, StatusCategory::Success);
        assert_eq!(history[1].status, StatusCategory::ClientError);
    }

    #[test]
    fn test_endpoint_history_bounded() {
        let mut api = RestApi::with_config(small_config()); // max_history_per_endpoint=5
        api.register_endpoint("/a", vec![HttpMethod::Get], "").unwrap();
        for _ in 0..10 {
            api.handle_request("/a", HttpMethod::Get, 10.0, 200).unwrap();
        }
        assert_eq!(api.endpoint("/a").unwrap().history().len(), 5);
    }

    // -------------------------------------------------------------------
    // Overall success rate
    // -------------------------------------------------------------------

    #[test]
    fn test_overall_success_rate() {
        let mut api = RestApi::with_config(small_config());
        api.register_endpoint("/a", vec![HttpMethod::Get], "").unwrap();
        api.handle_request("/a", HttpMethod::Get, 10.0, 200).unwrap();
        api.handle_request("/a", HttpMethod::Get, 10.0, 500).unwrap();
        assert!((api.overall_success_rate() - 0.5).abs() < 1e-9);
    }

    #[test]
    fn test_overall_success_rate_empty() {
        let api = RestApi::with_config(small_config());
        assert!((api.overall_success_rate() - 0.0).abs() < 1e-9);
    }

    // -------------------------------------------------------------------
    // Tick & EMA
    // -------------------------------------------------------------------

    #[test]
    fn test_tick_increments() {
        let mut api = RestApi::with_config(small_config());
        api.tick();
        api.tick();
        assert_eq!(api.current_tick(), 2);
    }

    #[test]
    fn test_process_alias() {
        let mut api = RestApi::with_config(small_config());
        api.process();
        assert_eq!(api.current_tick(), 1);
    }

    #[test]
    fn test_ema_initialises_on_first_active_tick() {
        let mut api = RestApi::with_config(small_config());
        api.register_endpoint("/a", vec![HttpMethod::Get], "").unwrap();
        api.handle_request("/a", HttpMethod::Get, 50.0, 200).unwrap();
        api.tick();
        assert!((api.smoothed_throughput() - 1.0).abs() < 1e-9);
        assert!((api.smoothed_success_ratio() - 1.0).abs() < 1e-9);
        assert!((api.smoothed_avg_latency() - 50.0).abs() < 1e-9);
    }

    #[test]
    fn test_ema_blends() {
        let mut api = RestApi::with_config(small_config()); // decay=0.5
        api.register_endpoint("/a", vec![HttpMethod::Get], "").unwrap();

        // Tick 1: 1 request, latency 100.
        api.handle_request("/a", HttpMethod::Get, 100.0, 200).unwrap();
        api.tick(); // EMA: throughput=1, latency=100

        // Tick 2: 1 request, latency 200.
        api.handle_request("/a", HttpMethod::Get, 200.0, 200).unwrap();
        api.tick(); // EMA: throughput=0.5*1+0.5*1=1, latency=0.5*200+0.5*100=150

        assert!((api.smoothed_avg_latency() - 150.0).abs() < 1e-9);
    }

    #[test]
    fn test_ema_no_init_without_requests() {
        let mut api = RestApi::with_config(small_config());
        api.tick();
        // EMA should still be at defaults.
        assert!((api.smoothed_throughput() - 0.0).abs() < 1e-9);
    }

    // -------------------------------------------------------------------
    // Windowed diagnostics
    // -------------------------------------------------------------------

    #[test]
    fn test_windowed_empty() {
        let api = RestApi::with_config(small_config());
        assert!(api.windowed_throughput().is_none());
        assert!(api.windowed_success_ratio().is_none());
        assert!(api.windowed_avg_latency().is_none());
        assert!(api.windowed_slow_request_count().is_none());
    }

    #[test]
    fn test_windowed_throughput() {
        let mut api = RestApi::with_config(small_config());
        api.register_endpoint("/a", vec![HttpMethod::Get], "").unwrap();
        api.handle_request("/a", HttpMethod::Get, 10.0, 200).unwrap();
        api.handle_request("/a", HttpMethod::Get, 10.0, 200).unwrap();
        api.tick(); // 2 requests this tick
        api.tick(); // 0 requests this tick
        // Windowed avg = (2 + 0) / 2 = 1.0
        assert!((api.windowed_throughput().unwrap() - 1.0).abs() < 1e-9);
    }

    #[test]
    fn test_windowed_success_ratio() {
        let mut api = RestApi::with_config(small_config());
        api.register_endpoint("/a", vec![HttpMethod::Get], "").unwrap();
        api.handle_request("/a", HttpMethod::Get, 10.0, 200).unwrap();
        api.tick();
        api.handle_request("/a", HttpMethod::Get, 10.0, 500).unwrap();
        api.tick();
        // Tick 1: ratio=1.0, Tick 2: ratio=0.0 → avg=0.5
        let ratio = api.windowed_success_ratio().unwrap();
        assert!((ratio - 0.5).abs() < 1e-9);
    }

    #[test]
    fn test_windowed_slow_request_count() {
        let mut api = RestApi::with_config(small_config()); // threshold=100
        api.register_endpoint("/a", vec![HttpMethod::Get], "").unwrap();
        api.handle_request("/a", HttpMethod::Get, 200.0, 200).unwrap(); // slow
        api.handle_request("/a", HttpMethod::Get, 10.0, 200).unwrap(); // fast
        api.tick();
        let count = api.windowed_slow_request_count().unwrap();
        assert!((count - 1.0).abs() < 1e-9);
    }

    #[test]
    fn test_is_latency_increasing() {
        let mut api = RestApi::with_config(small_config());
        api.register_endpoint("/a", vec![HttpMethod::Get], "").unwrap();
        // First two ticks: low latency.
        for _ in 0..2 {
            api.handle_request("/a", HttpMethod::Get, 10.0, 200).unwrap();
            api.tick();
        }
        // Next two ticks: high latency.
        for _ in 0..2 {
            api.handle_request("/a", HttpMethod::Get, 200.0, 200).unwrap();
            api.tick();
        }
        assert!(api.is_latency_increasing());
    }

    #[test]
    fn test_is_latency_increasing_insufficient() {
        let mut api = RestApi::with_config(small_config());
        api.tick();
        assert!(!api.is_latency_increasing());
    }

    #[test]
    fn test_is_success_declining() {
        let mut api = RestApi::with_config(small_config());
        api.register_endpoint("/a", vec![HttpMethod::Get], "").unwrap();
        // First two ticks: all success.
        for _ in 0..2 {
            api.handle_request("/a", HttpMethod::Get, 10.0, 200).unwrap();
            api.tick();
        }
        // Next two ticks: all errors.
        for _ in 0..2 {
            api.handle_request("/a", HttpMethod::Get, 10.0, 500).unwrap();
            api.tick();
        }
        assert!(api.is_success_declining());
    }

    #[test]
    fn test_is_success_declining_insufficient() {
        let mut api = RestApi::with_config(small_config());
        api.tick();
        assert!(!api.is_success_declining());
    }

    #[test]
    fn test_window_rolls() {
        let mut api = RestApi::with_config(small_config()); // window_size=5
        api.register_endpoint("/a", vec![HttpMethod::Get], "").unwrap();
        for _ in 0..20 {
            api.handle_request("/a", HttpMethod::Get, 10.0, 200).unwrap();
            api.tick();
        }
        assert!(api.recent.len() <= 5);
    }

    // -------------------------------------------------------------------
    // Status code mapping
    // -------------------------------------------------------------------

    #[test]
    fn test_status_category_from_code() {
        assert_eq!(StatusCategory::from_code(200), StatusCategory::Success);
        assert_eq!(StatusCategory::from_code(201), StatusCategory::Success);
        assert_eq!(StatusCategory::from_code(301), StatusCategory::Redirect);
        assert_eq!(StatusCategory::from_code(400), StatusCategory::ClientError);
        assert_eq!(StatusCategory::from_code(429), StatusCategory::RateLimited);
        assert_eq!(StatusCategory::from_code(500), StatusCategory::ServerError);
    }

    // -------------------------------------------------------------------
    // HTTP method display
    // -------------------------------------------------------------------

    #[test]
    fn test_http_method_display() {
        assert_eq!(format!("{}", HttpMethod::Get), "GET");
        assert_eq!(format!("{}", HttpMethod::Post), "POST");
        assert_eq!(format!("{}", HttpMethod::Delete), "DELETE");
    }

    // -------------------------------------------------------------------
    // Reset
    // -------------------------------------------------------------------

    #[test]
    fn test_reset() {
        let mut api = RestApi::with_config(small_config());
        api.register_endpoint("/a", vec![HttpMethod::Get], "").unwrap();
        api.handle_request("/a", HttpMethod::Get, 50.0, 200).unwrap();
        api.tick();

        api.reset();

        assert_eq!(api.current_tick(), 0);
        assert_eq!(api.stats().total_requests, 0);
        assert_eq!(api.stats().total_success, 0);
        assert_eq!(api.endpoint("/a").unwrap().total_requests, 0);
        assert!(api.endpoint("/a").unwrap().history().is_empty());
        assert!(api.windowed_throughput().is_none());
        // Endpoint still registered.
        assert_eq!(api.endpoint_count(), 1);
    }

    // -------------------------------------------------------------------
    // Full lifecycle
    // -------------------------------------------------------------------

    #[test]
    fn test_full_lifecycle() {
        let mut api = RestApi::with_config(small_config());

        // Register endpoints.
        api.register_endpoint(
            "/api/v1/signals",
            vec![HttpMethod::Get, HttpMethod::Post],
            "Trading signals",
        )
        .unwrap();
        api.register_endpoint_with_rate_limit(
            "/api/v1/orders",
            vec![HttpMethod::Post],
            "Order submission",
            3,
        )
        .unwrap();

        // Tick 1: normal requests.
        api.handle_request("/api/v1/signals", HttpMethod::Get, 30.0, 200)
            .unwrap();
        api.handle_request("/api/v1/signals", HttpMethod::Post, 50.0, 201)
            .unwrap();
        api.handle_request("/api/v1/orders", HttpMethod::Post, 80.0, 200)
            .unwrap();
        api.tick();

        // Tick 2: some errors and rate limiting.
        api.handle_request("/api/v1/signals", HttpMethod::Get, 10.0, 500)
            .unwrap();
        api.handle_request("/api/v1/orders", HttpMethod::Post, 20.0, 200)
            .unwrap();
        api.handle_request("/api/v1/orders", HttpMethod::Post, 20.0, 200)
            .unwrap();
        // This should be rate-limited (limit=3, 3rd request in window).
        let rl = api
            .handle_request("/api/v1/orders", HttpMethod::Post, 5.0, 200)
            .unwrap();
        assert!(rl.rate_limited);
        api.tick();

        // Stats.
        assert_eq!(api.stats().total_requests, 7);
        assert!(api.stats().total_success > 0);
        assert!(api.stats().total_server_errors > 0);
        assert_eq!(api.stats().total_rate_limited, 1);

        // Diagnostics.
        assert!(api.smoothed_throughput() > 0.0);
        assert!(api.windowed_throughput().is_some());
        assert!(api.windowed_success_ratio().is_some());

        // Per-endpoint stats.
        let signals_ep = api.endpoint("/api/v1/signals").unwrap();
        assert_eq!(signals_ep.total_requests, 3);
        assert_eq!(signals_ep.total_success, 2);
        assert_eq!(signals_ep.total_server_errors, 1);

        let orders_ep = api.endpoint("/api/v1/orders").unwrap();
        assert_eq!(orders_ep.total_requests, 4);
        assert_eq!(orders_ep.total_rate_limited, 1);
    }
}
