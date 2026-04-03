//! gRPC server management
//!
//! Part of the Integration region — API component.
//!
//! `GrpcServer` provides a simulated gRPC service registry that tracks
//! registered services and methods, records per-method call latencies,
//! manages streaming state, and exposes EMA-smoothed and windowed diagnostics.

use crate::common::{Error, Result};
use std::collections::{HashMap, VecDeque};

// ---------------------------------------------------------------------------
// Configuration
// ---------------------------------------------------------------------------

/// Configuration for the gRPC server.
#[derive(Debug, Clone)]
pub struct GrpcServerConfig {
    /// Maximum number of services that can be registered.
    pub max_services: usize,
    /// Maximum number of methods per service.
    pub max_methods_per_service: usize,
    /// Maximum number of concurrent streams allowed.
    pub max_concurrent_streams: usize,
    /// Call latency threshold (µs) above which a method is considered slow.
    pub slow_method_threshold_us: f64,
    /// Maximum call history retained per method.
    pub max_call_history: usize,
    /// EMA decay factor (0 < α < 1). Higher → faster adaptation.
    pub ema_decay: f64,
    /// Window size for rolling diagnostics.
    pub window_size: usize,
}

impl Default for GrpcServerConfig {
    fn default() -> Self {
        Self {
            max_services: 32,
            max_methods_per_service: 64,
            max_concurrent_streams: 256,
            slow_method_threshold_us: 5000.0,
            max_call_history: 100,
            ema_decay: 0.1,
            window_size: 50,
        }
    }
}

// ---------------------------------------------------------------------------
// Method type
// ---------------------------------------------------------------------------

/// Type of gRPC method.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash)]
pub enum MethodType {
    /// Unary RPC — single request, single response.
    Unary,
    /// Server streaming — single request, stream of responses.
    ServerStreaming,
    /// Client streaming — stream of requests, single response.
    ClientStreaming,
    /// Bidirectional streaming — streams in both directions.
    BidirectionalStreaming,
}

impl std::fmt::Display for MethodType {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        match self {
            MethodType::Unary => write!(f, "Unary"),
            MethodType::ServerStreaming => write!(f, "ServerStreaming"),
            MethodType::ClientStreaming => write!(f, "ClientStreaming"),
            MethodType::BidirectionalStreaming => write!(f, "BidirectionalStreaming"),
        }
    }
}

// ---------------------------------------------------------------------------
// Call status
// ---------------------------------------------------------------------------

/// Status of a gRPC call.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash)]
pub enum CallStatus {
    /// Call completed successfully.
    Ok,
    /// Call completed with an error.
    Error,
    /// Call was cancelled by the client.
    Cancelled,
    /// Call exceeded the deadline.
    DeadlineExceeded,
    /// Internal server error.
    Internal,
}

impl std::fmt::Display for CallStatus {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        match self {
            CallStatus::Ok => write!(f, "OK"),
            CallStatus::Error => write!(f, "Error"),
            CallStatus::Cancelled => write!(f, "Cancelled"),
            CallStatus::DeadlineExceeded => write!(f, "DeadlineExceeded"),
            CallStatus::Internal => write!(f, "Internal"),
        }
    }
}

// ---------------------------------------------------------------------------
// Call record
// ---------------------------------------------------------------------------

/// Record of a single gRPC call.
#[derive(Debug, Clone)]
pub struct CallRecord {
    /// Fully-qualified method name (e.g. "service.Method").
    pub method: String,
    /// Call status.
    pub status: CallStatus,
    /// Latency in µs.
    pub latency_us: f64,
    /// Tick at which the call occurred.
    pub tick: u64,
}

// ---------------------------------------------------------------------------
// Method descriptor
// ---------------------------------------------------------------------------

/// Descriptor for a registered gRPC method.
#[derive(Debug, Clone)]
pub struct MethodDescriptor {
    /// Method name (relative to its service).
    pub name: String,
    /// Fully-qualified name (`service/method`).
    pub full_name: String,
    /// Method type (unary, streaming, etc.).
    pub method_type: MethodType,
    /// Total calls.
    pub total_calls: u64,
    /// Total successful calls.
    pub success_count: u64,
    /// Total error calls.
    pub error_count: u64,
    /// Last recorded latency in µs.
    pub last_latency_us: f64,
    /// EMA-smoothed latency.
    pub ema_latency_us: f64,
    /// Recent call history.
    call_history: VecDeque<CallRecord>,
    /// Maximum history length.
    max_history: usize,
}

impl MethodDescriptor {
    fn new(
        name: impl Into<String>,
        full_name: impl Into<String>,
        method_type: MethodType,
        max_history: usize,
    ) -> Self {
        Self {
            name: name.into(),
            full_name: full_name.into(),
            method_type,
            total_calls: 0,
            success_count: 0,
            error_count: 0,
            last_latency_us: 0.0,
            ema_latency_us: 0.0,
            call_history: VecDeque::with_capacity(max_history.min(128)),
            max_history,
        }
    }

    /// Success rate over all calls.
    pub fn success_rate(&self) -> f64 {
        if self.total_calls == 0 {
            return 0.0;
        }
        self.success_count as f64 / self.total_calls as f64
    }

    /// Returns the recent call history.
    pub fn call_history(&self) -> &VecDeque<CallRecord> {
        &self.call_history
    }
}

// ---------------------------------------------------------------------------
// Service descriptor
// ---------------------------------------------------------------------------

/// Descriptor for a registered gRPC service.
#[derive(Debug, Clone)]
pub struct ServiceDescriptor {
    /// Service name.
    pub name: String,
    /// Methods registered under this service, keyed by method name.
    pub methods: HashMap<String, MethodDescriptor>,
    /// Insertion order for methods.
    method_order: Vec<String>,
    /// Whether the service is enabled.
    pub enabled: bool,
    /// Total calls across all methods in this service.
    pub total_calls: u64,
}

impl ServiceDescriptor {
    fn new(name: impl Into<String>) -> Self {
        Self {
            name: name.into(),
            methods: HashMap::new(),
            method_order: Vec::new(),
            enabled: true,
            total_calls: 0,
        }
    }

    /// Return the method names registered under this service.
    pub fn method_names(&self) -> &[String] {
        &self.method_order
    }

    /// Return the number of methods in this service.
    pub fn method_count(&self) -> usize {
        self.methods.len()
    }
}

// ---------------------------------------------------------------------------
// Stream info
// ---------------------------------------------------------------------------

/// Information about an active stream.
#[derive(Debug, Clone)]
pub struct StreamInfo {
    /// Stream identifier.
    pub id: u64,
    /// Fully-qualified method name.
    pub method: String,
    /// Messages sent on this stream.
    pub messages_sent: u64,
    /// Messages received on this stream.
    pub messages_received: u64,
    /// Tick at which the stream was opened.
    pub opened_tick: u64,
}

// ---------------------------------------------------------------------------
// Tick snapshot (windowed diagnostics)
// ---------------------------------------------------------------------------

#[derive(Debug, Clone)]
struct TickSnapshot {
    total_calls: u64,
    success_rate: f64,
    active_streams: usize,
    avg_latency_us: f64,
}

// ---------------------------------------------------------------------------
// Statistics
// ---------------------------------------------------------------------------

/// Cumulative statistics for the gRPC server.
#[derive(Debug, Clone)]
pub struct GrpcServerStats {
    /// Total calls across all methods.
    pub total_calls: u64,
    /// Total successful calls.
    pub total_successes: u64,
    /// Total error calls.
    pub total_errors: u64,
    /// Total streams opened.
    pub total_streams_opened: u64,
    /// Total streams closed.
    pub total_streams_closed: u64,
    /// Calls in the current tick.
    pub calls_this_tick: u64,
    /// EMA-smoothed call rate (calls per tick).
    pub ema_call_rate: f64,
    /// EMA-smoothed success rate.
    pub ema_success_rate: f64,
    /// EMA-smoothed average latency.
    pub ema_avg_latency_us: f64,
}

impl Default for GrpcServerStats {
    fn default() -> Self {
        Self {
            total_calls: 0,
            total_successes: 0,
            total_errors: 0,
            total_streams_opened: 0,
            total_streams_closed: 0,
            calls_this_tick: 0,
            ema_call_rate: 0.0,
            ema_success_rate: 0.0,
            ema_avg_latency_us: 0.0,
        }
    }
}

// ---------------------------------------------------------------------------
// GrpcServer
// ---------------------------------------------------------------------------

/// Simulated gRPC server with service/method registry, call tracking,
/// streaming support, and EMA + windowed diagnostics.
pub struct GrpcServer {
    config: GrpcServerConfig,
    /// Registered services keyed by name.
    services: HashMap<String, ServiceDescriptor>,
    /// Insertion order for services.
    service_order: Vec<String>,
    /// Active streams keyed by stream ID.
    active_streams: HashMap<u64, StreamInfo>,
    /// Next stream ID.
    next_stream_id: u64,
    /// Current tick.
    tick: u64,
    /// Latency accumulator for the current tick (for averaging).
    latency_sum_this_tick: f64,
    /// Whether EMA has been initialised.
    ema_initialized: bool,
    /// Rolling window of tick snapshots.
    recent: VecDeque<TickSnapshot>,
    /// Cumulative statistics.
    stats: GrpcServerStats,
}

impl Default for GrpcServer {
    fn default() -> Self {
        Self::new()
    }
}

impl GrpcServer {
    // -------------------------------------------------------------------
    // Construction
    // -------------------------------------------------------------------

    /// Create a new gRPC server with default configuration.
    pub fn new() -> Self {
        Self::with_config(GrpcServerConfig::default())
    }

    /// Create a new gRPC server with the given configuration.
    pub fn with_config(config: GrpcServerConfig) -> Self {
        Self {
            services: HashMap::new(),
            service_order: Vec::new(),
            active_streams: HashMap::new(),
            next_stream_id: 1,
            tick: 0,
            latency_sum_this_tick: 0.0,
            ema_initialized: false,
            recent: VecDeque::with_capacity(config.window_size.min(256)),
            stats: GrpcServerStats::default(),
            config,
        }
    }

    // -------------------------------------------------------------------
    // Service management
    // -------------------------------------------------------------------

    /// Register a new service. Returns an error if the service already exists
    /// or the maximum number of services has been reached.
    pub fn register_service(&mut self, name: impl Into<String>) -> Result<()> {
        let name = name.into();
        if self.services.contains_key(&name) {
            return Err(Error::Configuration(format!(
                "Service '{}' is already registered",
                name
            )));
        }
        if self.services.len() >= self.config.max_services {
            return Err(Error::Configuration(format!(
                "Maximum service count ({}) reached",
                self.config.max_services
            )));
        }
        self.services
            .insert(name.clone(), ServiceDescriptor::new(&name));
        self.service_order.push(name);
        Ok(())
    }

    /// Enable a service.
    pub fn enable_service(&mut self, name: &str) -> Result<()> {
        let svc = self
            .services
            .get_mut(name)
            .ok_or_else(|| Error::Configuration(format!("Unknown service '{}'", name)))?;
        svc.enabled = true;
        Ok(())
    }

    /// Disable a service.
    pub fn disable_service(&mut self, name: &str) -> Result<()> {
        let svc = self
            .services
            .get_mut(name)
            .ok_or_else(|| Error::Configuration(format!("Unknown service '{}'", name)))?;
        svc.enabled = false;
        Ok(())
    }

    /// Look up a service descriptor.
    pub fn service(&self, name: &str) -> Option<&ServiceDescriptor> {
        self.services.get(name)
    }

    /// Return the number of registered services.
    pub fn service_count(&self) -> usize {
        self.services.len()
    }

    /// Return the names of all registered services.
    pub fn service_names(&self) -> Vec<&str> {
        self.service_order.iter().map(|s| s.as_str()).collect()
    }

    // -------------------------------------------------------------------
    // Method management
    // -------------------------------------------------------------------

    /// Register a method under a service. Returns an error if the service
    /// does not exist, the method already exists, or the per-service method
    /// limit has been reached.
    pub fn register_method(
        &mut self,
        service: &str,
        method_name: impl Into<String>,
        method_type: MethodType,
    ) -> Result<()> {
        let method_name = method_name.into();
        let max_history = self.config.max_call_history;
        let max_methods = self.config.max_methods_per_service;

        let svc = self
            .services
            .get_mut(service)
            .ok_or_else(|| Error::Configuration(format!("Unknown service '{}'", service)))?;

        if svc.methods.contains_key(&method_name) {
            return Err(Error::Configuration(format!(
                "Method '{}/{}' already registered",
                service, method_name
            )));
        }
        if svc.methods.len() >= max_methods {
            return Err(Error::Configuration(format!(
                "Service '{}' has reached the method limit ({})",
                service, max_methods
            )));
        }

        let full_name = format!("{}/{}", service, method_name);
        svc.methods.insert(
            method_name.clone(),
            MethodDescriptor::new(&method_name, full_name, method_type, max_history),
        );
        svc.method_order.push(method_name);
        Ok(())
    }

    /// Look up a method descriptor by service and method name.
    pub fn method(&self, service: &str, method_name: &str) -> Option<&MethodDescriptor> {
        self.services
            .get(service)
            .and_then(|svc| svc.methods.get(method_name))
    }

    // -------------------------------------------------------------------
    // Call recording
    // -------------------------------------------------------------------

    /// Record a call to a method.
    ///
    /// Returns an error if the service or method is not registered or the
    /// service is disabled.
    pub fn record_call(
        &mut self,
        service: &str,
        method_name: &str,
        status: CallStatus,
        latency_us: f64,
    ) -> Result<()> {
        let tick = self.tick;
        let decay = self.config.ema_decay;

        let svc = self
            .services
            .get_mut(service)
            .ok_or_else(|| Error::Configuration(format!("Unknown service '{}'", service)))?;

        if !svc.enabled {
            return Err(Error::Configuration(format!(
                "Service '{}' is disabled",
                service
            )));
        }

        let method = svc
            .methods
            .get_mut(method_name)
            .ok_or_else(|| {
                Error::Configuration(format!(
                    "Unknown method '{}/{}'",
                    service, method_name
                ))
            })?;

        method.total_calls += 1;
        method.last_latency_us = latency_us;

        if method.total_calls == 1 {
            method.ema_latency_us = latency_us;
        } else {
            method.ema_latency_us =
                decay * latency_us + (1.0 - decay) * method.ema_latency_us;
        }

        match status {
            CallStatus::Ok => {
                method.success_count += 1;
                self.stats.total_successes += 1;
            }
            _ => {
                method.error_count += 1;
                self.stats.total_errors += 1;
            }
        }

        let record = CallRecord {
            method: method.full_name.clone(),
            status,
            latency_us,
            tick,
        };
        if method.call_history.len() >= method.max_history {
            method.call_history.pop_front();
        }
        method.call_history.push_back(record);

        svc.total_calls += 1;
        self.stats.total_calls += 1;
        self.stats.calls_this_tick += 1;
        self.latency_sum_this_tick += latency_us;

        Ok(())
    }

    // -------------------------------------------------------------------
    // Streaming
    // -------------------------------------------------------------------

    /// Open a new stream for a method. Returns the stream ID.
    pub fn open_stream(&mut self, service: &str, method_name: &str) -> Result<u64> {
        // Validate method exists.
        let svc = self
            .services
            .get(service)
            .ok_or_else(|| Error::Configuration(format!("Unknown service '{}'", service)))?;

        if !svc.enabled {
            return Err(Error::Configuration(format!(
                "Service '{}' is disabled",
                service
            )));
        }

        if !svc.methods.contains_key(method_name) {
            return Err(Error::Configuration(format!(
                "Unknown method '{}/{}'",
                service, method_name
            )));
        }

        if self.active_streams.len() >= self.config.max_concurrent_streams {
            return Err(Error::Configuration(format!(
                "Maximum concurrent streams ({}) reached",
                self.config.max_concurrent_streams
            )));
        }

        let id = self.next_stream_id;
        self.next_stream_id += 1;

        let full_name = format!("{}/{}", service, method_name);
        self.active_streams.insert(
            id,
            StreamInfo {
                id,
                method: full_name,
                messages_sent: 0,
                messages_received: 0,
                opened_tick: self.tick,
            },
        );

        self.stats.total_streams_opened += 1;
        Ok(id)
    }

    /// Record a message sent on a stream.
    pub fn stream_send(&mut self, stream_id: u64) -> Result<()> {
        let stream = self
            .active_streams
            .get_mut(&stream_id)
            .ok_or_else(|| {
                Error::Configuration(format!("Unknown stream {}", stream_id))
            })?;
        stream.messages_sent += 1;
        Ok(())
    }

    /// Record a message received on a stream.
    pub fn stream_receive(&mut self, stream_id: u64) -> Result<()> {
        let stream = self
            .active_streams
            .get_mut(&stream_id)
            .ok_or_else(|| {
                Error::Configuration(format!("Unknown stream {}", stream_id))
            })?;
        stream.messages_received += 1;
        Ok(())
    }

    /// Close a stream. Returns the stream info on success.
    pub fn close_stream(&mut self, stream_id: u64) -> Result<StreamInfo> {
        let info = self
            .active_streams
            .remove(&stream_id)
            .ok_or_else(|| {
                Error::Configuration(format!("Unknown stream {}", stream_id))
            })?;
        self.stats.total_streams_closed += 1;
        Ok(info)
    }

    /// Return the number of active streams.
    pub fn active_stream_count(&self) -> usize {
        self.active_streams.len()
    }

    /// Look up a stream by ID.
    pub fn stream(&self, stream_id: u64) -> Option<&StreamInfo> {
        self.active_streams.get(&stream_id)
    }

    // -------------------------------------------------------------------
    // Tick
    // -------------------------------------------------------------------

    /// Advance the server by one tick, updating EMA and windowed diagnostics.
    pub fn tick(&mut self) {
        self.tick += 1;

        let calls = self.stats.calls_this_tick;
        let success_rate = if self.stats.total_calls > 0 {
            self.stats.total_successes as f64 / self.stats.total_calls as f64
        } else {
            1.0
        };
        let avg_latency = if calls > 0 {
            self.latency_sum_this_tick / calls as f64
        } else {
            0.0
        };

        let snapshot = TickSnapshot {
            total_calls: self.stats.total_calls,
            success_rate,
            active_streams: self.active_streams.len(),
            avg_latency_us: avg_latency,
        };

        let alpha = self.config.ema_decay;
        if !self.ema_initialized && calls > 0 {
            self.stats.ema_call_rate = calls as f64;
            self.stats.ema_success_rate = success_rate;
            self.stats.ema_avg_latency_us = avg_latency;
            self.ema_initialized = true;
        } else if self.ema_initialized {
            self.stats.ema_call_rate =
                alpha * calls as f64 + (1.0 - alpha) * self.stats.ema_call_rate;
            self.stats.ema_success_rate =
                alpha * success_rate + (1.0 - alpha) * self.stats.ema_success_rate;
            self.stats.ema_avg_latency_us =
                alpha * avg_latency + (1.0 - alpha) * self.stats.ema_avg_latency_us;
        }

        if self.recent.len() >= self.config.window_size {
            self.recent.pop_front();
        }
        self.recent.push_back(snapshot);

        // Reset per-tick counters.
        self.stats.calls_this_tick = 0;
        self.latency_sum_this_tick = 0.0;
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
    // Statistics & diagnostics
    // -------------------------------------------------------------------

    /// Returns a reference to the cumulative statistics.
    pub fn stats(&self) -> &GrpcServerStats {
        &self.stats
    }

    /// Returns a reference to the configuration.
    pub fn config(&self) -> &GrpcServerConfig {
        &self.config
    }

    /// Overall success rate across all calls.
    pub fn success_rate(&self) -> f64 {
        if self.stats.total_calls == 0 {
            return 1.0;
        }
        self.stats.total_successes as f64 / self.stats.total_calls as f64
    }

    /// EMA-smoothed call rate (calls per tick).
    pub fn smoothed_call_rate(&self) -> f64 {
        self.stats.ema_call_rate
    }

    /// EMA-smoothed success rate.
    pub fn smoothed_success_rate(&self) -> f64 {
        self.stats.ema_success_rate
    }

    /// EMA-smoothed average latency.
    pub fn smoothed_avg_latency(&self) -> f64 {
        self.stats.ema_avg_latency_us
    }

    /// Windowed average success rate.
    pub fn windowed_success_rate(&self) -> Option<f64> {
        if self.recent.is_empty() {
            return None;
        }
        let sum: f64 = self.recent.iter().map(|s| s.success_rate).sum();
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

    /// Windowed average active stream count.
    pub fn windowed_active_streams(&self) -> Option<f64> {
        if self.recent.is_empty() {
            return None;
        }
        let sum: f64 = self
            .recent
            .iter()
            .map(|s| s.active_streams as f64)
            .sum();
        Some(sum / self.recent.len() as f64)
    }

    /// Whether success rate appears to be declining over the window.
    pub fn is_success_rate_declining(&self) -> bool {
        if self.recent.len() < 4 {
            return false;
        }
        let mid = self.recent.len() / 2;
        let first_half: f64 =
            self.recent.iter().take(mid).map(|s| s.success_rate).sum::<f64>() / mid as f64;
        let second_half: f64 = self
            .recent
            .iter()
            .skip(mid)
            .map(|s| s.success_rate)
            .sum::<f64>()
            / (self.recent.len() - mid) as f64;
        second_half < first_half - 0.05
    }

    /// Return the names of methods whose EMA latency exceeds the slow
    /// threshold.
    pub fn slow_methods(&self) -> Vec<String> {
        let threshold = self.config.slow_method_threshold_us;
        let mut result = Vec::new();
        for svc in self.services.values() {
            for method in svc.methods.values() {
                if method.ema_latency_us > threshold && method.total_calls > 0 {
                    result.push(method.full_name.clone());
                }
            }
        }
        result.sort();
        result
    }

    // -------------------------------------------------------------------
    // Reset
    // -------------------------------------------------------------------

    /// Reset all state, preserving configuration and registered
    /// services/methods (whose counters are also reset).
    pub fn reset(&mut self) {
        self.tick = 0;
        self.latency_sum_this_tick = 0.0;
        self.ema_initialized = false;
        self.recent.clear();
        self.active_streams.clear();
        self.next_stream_id = 1;
        self.stats = GrpcServerStats::default();

        for svc in self.services.values_mut() {
            svc.total_calls = 0;
            for method in svc.methods.values_mut() {
                method.total_calls = 0;
                method.success_count = 0;
                method.error_count = 0;
                method.last_latency_us = 0.0;
                method.ema_latency_us = 0.0;
                method.call_history.clear();
            }
        }
    }
}

// ===========================================================================
// Tests
// ===========================================================================

#[cfg(test)]
mod tests {
    use super::*;

    fn small_config() -> GrpcServerConfig {
        GrpcServerConfig {
            max_services: 4,
            max_methods_per_service: 4,
            max_concurrent_streams: 4,
            slow_method_threshold_us: 100.0,
            max_call_history: 5,
            ema_decay: 0.5,
            window_size: 5,
        }
    }

    // -------------------------------------------------------------------
    // Construction
    // -------------------------------------------------------------------

    #[test]
    fn test_new_default() {
        let srv = GrpcServer::new();
        assert_eq!(srv.service_count(), 0);
        assert_eq!(srv.current_tick(), 0);
        assert_eq!(srv.active_stream_count(), 0);
    }

    #[test]
    fn test_with_config() {
        let srv = GrpcServer::with_config(small_config());
        assert_eq!(srv.config().max_services, 4);
    }

    // -------------------------------------------------------------------
    // Service management
    // -------------------------------------------------------------------

    #[test]
    fn test_register_service() {
        let mut srv = GrpcServer::with_config(small_config());
        srv.register_service("NeuromorphicService").unwrap();
        assert_eq!(srv.service_count(), 1);
        assert!(srv.service("NeuromorphicService").is_some());
    }

    #[test]
    fn test_register_service_duplicate() {
        let mut srv = GrpcServer::with_config(small_config());
        srv.register_service("svc").unwrap();
        assert!(srv.register_service("svc").is_err());
    }

    #[test]
    fn test_register_service_at_capacity() {
        let mut srv = GrpcServer::with_config(small_config());
        for i in 0..4 {
            srv.register_service(format!("svc{}", i)).unwrap();
        }
        assert!(srv.register_service("overflow").is_err());
    }

    #[test]
    fn test_enable_disable_service() {
        let mut srv = GrpcServer::with_config(small_config());
        srv.register_service("svc").unwrap();
        srv.disable_service("svc").unwrap();
        assert!(!srv.service("svc").unwrap().enabled);
        srv.enable_service("svc").unwrap();
        assert!(srv.service("svc").unwrap().enabled);
    }

    #[test]
    fn test_enable_unknown_service() {
        let mut srv = GrpcServer::with_config(small_config());
        assert!(srv.enable_service("nope").is_err());
        assert!(srv.disable_service("nope").is_err());
    }

    #[test]
    fn test_service_names() {
        let mut srv = GrpcServer::with_config(small_config());
        srv.register_service("b").unwrap();
        srv.register_service("a").unwrap();
        assert_eq!(srv.service_names(), vec!["b", "a"]);
    }

    // -------------------------------------------------------------------
    // Method management
    // -------------------------------------------------------------------

    #[test]
    fn test_register_method() {
        let mut srv = GrpcServer::with_config(small_config());
        srv.register_service("svc").unwrap();
        srv.register_method("svc", "GetData", MethodType::Unary)
            .unwrap();
        let method = srv.method("svc", "GetData").unwrap();
        assert_eq!(method.method_type, MethodType::Unary);
        assert_eq!(method.full_name, "svc/GetData");
    }

    #[test]
    fn test_register_method_unknown_service() {
        let mut srv = GrpcServer::with_config(small_config());
        assert!(srv
            .register_method("nope", "Method", MethodType::Unary)
            .is_err());
    }

    #[test]
    fn test_register_method_duplicate() {
        let mut srv = GrpcServer::with_config(small_config());
        srv.register_service("svc").unwrap();
        srv.register_method("svc", "M", MethodType::Unary).unwrap();
        assert!(srv.register_method("svc", "M", MethodType::Unary).is_err());
    }

    #[test]
    fn test_register_method_at_capacity() {
        let mut srv = GrpcServer::with_config(small_config());
        srv.register_service("svc").unwrap();
        for i in 0..4 {
            srv.register_method("svc", format!("m{}", i), MethodType::Unary)
                .unwrap();
        }
        assert!(srv
            .register_method("svc", "overflow", MethodType::Unary)
            .is_err());
    }

    #[test]
    fn test_method_names_in_service() {
        let mut srv = GrpcServer::with_config(small_config());
        srv.register_service("svc").unwrap();
        srv.register_method("svc", "B", MethodType::Unary).unwrap();
        srv.register_method("svc", "A", MethodType::ServerStreaming)
            .unwrap();
        let svc = srv.service("svc").unwrap();
        assert_eq!(svc.method_names(), &["B", "A"]);
        assert_eq!(svc.method_count(), 2);
    }

    #[test]
    fn test_method_lookup_nonexistent() {
        let srv = GrpcServer::with_config(small_config());
        assert!(srv.method("nope", "nope").is_none());
    }

    // -------------------------------------------------------------------
    // Call recording
    // -------------------------------------------------------------------

    #[test]
    fn test_record_call_success() {
        let mut srv = GrpcServer::with_config(small_config());
        srv.register_service("svc").unwrap();
        srv.register_method("svc", "Get", MethodType::Unary).unwrap();
        srv.record_call("svc", "Get", CallStatus::Ok, 50.0).unwrap();

        let method = srv.method("svc", "Get").unwrap();
        assert_eq!(method.total_calls, 1);
        assert_eq!(method.success_count, 1);
        assert_eq!(method.error_count, 0);
        assert!((method.last_latency_us - 50.0).abs() < 1e-9);
        assert!((method.ema_latency_us - 50.0).abs() < 1e-9);
        assert!((method.success_rate() - 1.0).abs() < 1e-9);

        assert_eq!(srv.stats().total_calls, 1);
        assert_eq!(srv.stats().total_successes, 1);
    }

    #[test]
    fn test_record_call_error() {
        let mut srv = GrpcServer::with_config(small_config());
        srv.register_service("svc").unwrap();
        srv.register_method("svc", "Get", MethodType::Unary).unwrap();
        srv.record_call("svc", "Get", CallStatus::Error, 100.0)
            .unwrap();

        let method = srv.method("svc", "Get").unwrap();
        assert_eq!(method.error_count, 1);
        assert_eq!(srv.stats().total_errors, 1);
    }

    #[test]
    fn test_record_call_ema_latency() {
        let mut srv = GrpcServer::with_config(small_config()); // ema_decay = 0.5
        srv.register_service("svc").unwrap();
        srv.register_method("svc", "Get", MethodType::Unary).unwrap();
        srv.record_call("svc", "Get", CallStatus::Ok, 100.0).unwrap();
        srv.record_call("svc", "Get", CallStatus::Ok, 200.0).unwrap();
        // EMA: 0.5*200 + 0.5*100 = 150
        let ema = srv.method("svc", "Get").unwrap().ema_latency_us;
        assert!((ema - 150.0).abs() < 1e-9);
    }

    #[test]
    fn test_record_call_unknown_service() {
        let mut srv = GrpcServer::with_config(small_config());
        assert!(srv
            .record_call("nope", "Get", CallStatus::Ok, 10.0)
            .is_err());
    }

    #[test]
    fn test_record_call_unknown_method() {
        let mut srv = GrpcServer::with_config(small_config());
        srv.register_service("svc").unwrap();
        assert!(srv
            .record_call("svc", "nope", CallStatus::Ok, 10.0)
            .is_err());
    }

    #[test]
    fn test_record_call_disabled_service() {
        let mut srv = GrpcServer::with_config(small_config());
        srv.register_service("svc").unwrap();
        srv.register_method("svc", "Get", MethodType::Unary).unwrap();
        srv.disable_service("svc").unwrap();
        assert!(srv
            .record_call("svc", "Get", CallStatus::Ok, 10.0)
            .is_err());
    }

    #[test]
    fn test_call_history_bounded() {
        let mut srv = GrpcServer::with_config(small_config()); // max_call_history = 5
        srv.register_service("svc").unwrap();
        srv.register_method("svc", "Get", MethodType::Unary).unwrap();
        for _ in 0..10 {
            srv.record_call("svc", "Get", CallStatus::Ok, 10.0).unwrap();
        }
        assert_eq!(srv.method("svc", "Get").unwrap().call_history().len(), 5);
    }

    #[test]
    fn test_service_total_calls() {
        let mut srv = GrpcServer::with_config(small_config());
        srv.register_service("svc").unwrap();
        srv.register_method("svc", "A", MethodType::Unary).unwrap();
        srv.register_method("svc", "B", MethodType::Unary).unwrap();
        srv.record_call("svc", "A", CallStatus::Ok, 10.0).unwrap();
        srv.record_call("svc", "B", CallStatus::Ok, 20.0).unwrap();
        assert_eq!(srv.service("svc").unwrap().total_calls, 2);
    }

    // -------------------------------------------------------------------
    // Streaming
    // -------------------------------------------------------------------

    #[test]
    fn test_open_close_stream() {
        let mut srv = GrpcServer::with_config(small_config());
        srv.register_service("svc").unwrap();
        srv.register_method("svc", "Stream", MethodType::ServerStreaming)
            .unwrap();

        let id = srv.open_stream("svc", "Stream").unwrap();
        assert_eq!(srv.active_stream_count(), 1);
        assert!(srv.stream(id).is_some());
        assert_eq!(srv.stream(id).unwrap().method, "svc/Stream");

        let info = srv.close_stream(id).unwrap();
        assert_eq!(info.id, id);
        assert_eq!(srv.active_stream_count(), 0);
        assert_eq!(srv.stats().total_streams_opened, 1);
        assert_eq!(srv.stats().total_streams_closed, 1);
    }

    #[test]
    fn test_stream_send_receive() {
        let mut srv = GrpcServer::with_config(small_config());
        srv.register_service("svc").unwrap();
        srv.register_method("svc", "Stream", MethodType::BidirectionalStreaming)
            .unwrap();

        let id = srv.open_stream("svc", "Stream").unwrap();
        srv.stream_send(id).unwrap();
        srv.stream_send(id).unwrap();
        srv.stream_receive(id).unwrap();

        let info = srv.stream(id).unwrap();
        assert_eq!(info.messages_sent, 2);
        assert_eq!(info.messages_received, 1);
    }

    #[test]
    fn test_stream_send_unknown() {
        let mut srv = GrpcServer::with_config(small_config());
        assert!(srv.stream_send(999).is_err());
        assert!(srv.stream_receive(999).is_err());
    }

    #[test]
    fn test_close_stream_unknown() {
        let mut srv = GrpcServer::with_config(small_config());
        assert!(srv.close_stream(999).is_err());
    }

    #[test]
    fn test_open_stream_unknown_service() {
        let mut srv = GrpcServer::with_config(small_config());
        assert!(srv.open_stream("nope", "Method").is_err());
    }

    #[test]
    fn test_open_stream_unknown_method() {
        let mut srv = GrpcServer::with_config(small_config());
        srv.register_service("svc").unwrap();
        assert!(srv.open_stream("svc", "nope").is_err());
    }

    #[test]
    fn test_open_stream_disabled_service() {
        let mut srv = GrpcServer::with_config(small_config());
        srv.register_service("svc").unwrap();
        srv.register_method("svc", "M", MethodType::ServerStreaming)
            .unwrap();
        srv.disable_service("svc").unwrap();
        assert!(srv.open_stream("svc", "M").is_err());
    }

    #[test]
    fn test_open_stream_at_capacity() {
        let mut srv = GrpcServer::with_config(small_config()); // max_concurrent_streams = 4
        srv.register_service("svc").unwrap();
        srv.register_method("svc", "M", MethodType::ServerStreaming)
            .unwrap();
        for _ in 0..4 {
            srv.open_stream("svc", "M").unwrap();
        }
        assert!(srv.open_stream("svc", "M").is_err());
    }

    // -------------------------------------------------------------------
    // Tick & EMA
    // -------------------------------------------------------------------

    #[test]
    fn test_tick_increments() {
        let mut srv = GrpcServer::with_config(small_config());
        srv.tick();
        srv.tick();
        assert_eq!(srv.current_tick(), 2);
    }

    #[test]
    fn test_process_alias() {
        let mut srv = GrpcServer::with_config(small_config());
        srv.process();
        assert_eq!(srv.current_tick(), 1);
    }

    #[test]
    fn test_ema_initialises() {
        let mut srv = GrpcServer::with_config(small_config());
        srv.register_service("svc").unwrap();
        srv.register_method("svc", "Get", MethodType::Unary).unwrap();
        srv.record_call("svc", "Get", CallStatus::Ok, 100.0).unwrap();
        srv.tick();
        assert!((srv.smoothed_call_rate() - 1.0).abs() < 1e-9);
        assert!((srv.smoothed_avg_latency() - 100.0).abs() < 1e-9);
    }

    #[test]
    fn test_ema_blends() {
        let mut srv = GrpcServer::with_config(small_config()); // ema_decay = 0.5
        srv.register_service("svc").unwrap();
        srv.register_method("svc", "Get", MethodType::Unary).unwrap();

        // Tick 1: 1 call at 100µs
        srv.record_call("svc", "Get", CallStatus::Ok, 100.0).unwrap();
        srv.tick(); // ema_call_rate = 1, ema_avg_latency = 100

        // Tick 2: 3 calls at 200µs
        for _ in 0..3 {
            srv.record_call("svc", "Get", CallStatus::Ok, 200.0)
                .unwrap();
        }
        srv.tick(); // ema_call_rate = 0.5*3 + 0.5*1 = 2.0
        assert!((srv.smoothed_call_rate() - 2.0).abs() < 1e-9);
        // ema_avg_latency = 0.5*200 + 0.5*100 = 150
        assert!((srv.smoothed_avg_latency() - 150.0).abs() < 1e-9);
    }

    #[test]
    fn test_calls_this_tick_resets() {
        let mut srv = GrpcServer::with_config(small_config());
        srv.register_service("svc").unwrap();
        srv.register_method("svc", "Get", MethodType::Unary).unwrap();
        srv.record_call("svc", "Get", CallStatus::Ok, 10.0).unwrap();
        assert_eq!(srv.stats().calls_this_tick, 1);
        srv.tick();
        assert_eq!(srv.stats().calls_this_tick, 0);
    }

    // -------------------------------------------------------------------
    // Overall success rate
    // -------------------------------------------------------------------

    #[test]
    fn test_success_rate_no_calls() {
        let srv = GrpcServer::with_config(small_config());
        assert!((srv.success_rate() - 1.0).abs() < 1e-9);
    }

    #[test]
    fn test_success_rate_mixed() {
        let mut srv = GrpcServer::with_config(small_config());
        srv.register_service("svc").unwrap();
        srv.register_method("svc", "Get", MethodType::Unary).unwrap();
        srv.record_call("svc", "Get", CallStatus::Ok, 10.0).unwrap();
        srv.record_call("svc", "Get", CallStatus::Error, 10.0)
            .unwrap();
        assert!((srv.success_rate() - 0.5).abs() < 1e-9);
    }

    // -------------------------------------------------------------------
    // Slow methods
    // -------------------------------------------------------------------

    #[test]
    fn test_slow_methods() {
        let mut srv = GrpcServer::with_config(small_config()); // threshold = 100
        srv.register_service("svc").unwrap();
        srv.register_method("svc", "Fast", MethodType::Unary).unwrap();
        srv.register_method("svc", "Slow", MethodType::Unary).unwrap();
        srv.record_call("svc", "Fast", CallStatus::Ok, 10.0).unwrap();
        srv.record_call("svc", "Slow", CallStatus::Ok, 200.0).unwrap();
        let slow = srv.slow_methods();
        assert!(slow.contains(&"svc/Slow".to_string()));
        assert!(!slow.contains(&"svc/Fast".to_string()));
    }

    #[test]
    fn test_slow_methods_empty() {
        let srv = GrpcServer::with_config(small_config());
        assert!(srv.slow_methods().is_empty());
    }

    // -------------------------------------------------------------------
    // Windowed diagnostics
    // -------------------------------------------------------------------

    #[test]
    fn test_windowed_empty() {
        let srv = GrpcServer::with_config(small_config());
        assert!(srv.windowed_success_rate().is_none());
        assert!(srv.windowed_avg_latency().is_none());
        assert!(srv.windowed_active_streams().is_none());
    }

    #[test]
    fn test_windowed_success_rate() {
        let mut srv = GrpcServer::with_config(small_config());
        srv.register_service("svc").unwrap();
        srv.register_method("svc", "Get", MethodType::Unary).unwrap();

        srv.record_call("svc", "Get", CallStatus::Ok, 10.0).unwrap();
        srv.tick();
        srv.record_call("svc", "Get", CallStatus::Error, 10.0)
            .unwrap();
        srv.tick();

        let rate = srv.windowed_success_rate().unwrap();
        // First tick: 1/1 = 1.0; second tick: 1/2 = 0.5; avg = 0.75
        assert!((rate - 0.75).abs() < 1e-9);
    }

    #[test]
    fn test_windowed_avg_latency() {
        let mut srv = GrpcServer::with_config(small_config());
        srv.register_service("svc").unwrap();
        srv.register_method("svc", "Get", MethodType::Unary).unwrap();

        srv.record_call("svc", "Get", CallStatus::Ok, 100.0).unwrap();
        srv.tick();
        srv.record_call("svc", "Get", CallStatus::Ok, 200.0).unwrap();
        srv.tick();

        let avg = srv.windowed_avg_latency().unwrap();
        // tick1 avg = 100, tick2 avg = 200 → window avg = 150
        assert!((avg - 150.0).abs() < 1e-9);
    }

    #[test]
    fn test_windowed_active_streams() {
        let mut srv = GrpcServer::with_config(small_config());
        srv.register_service("svc").unwrap();
        srv.register_method("svc", "S", MethodType::ServerStreaming)
            .unwrap();

        let id = srv.open_stream("svc", "S").unwrap();
        srv.tick(); // 1 active stream
        srv.close_stream(id).unwrap();
        srv.tick(); // 0 active streams

        let avg = srv.windowed_active_streams().unwrap();
        assert!((avg - 0.5).abs() < 1e-9);
    }

    #[test]
    fn test_is_success_rate_declining() {
        let mut srv = GrpcServer::with_config(small_config());
        srv.register_service("svc").unwrap();
        srv.register_method("svc", "Get", MethodType::Unary).unwrap();

        // First two ticks: all OK
        for _ in 0..2 {
            srv.record_call("svc", "Get", CallStatus::Ok, 10.0).unwrap();
            srv.tick();
        }
        // Next two ticks: add errors
        for _ in 0..2 {
            srv.record_call("svc", "Get", CallStatus::Error, 10.0)
                .unwrap();
            srv.tick();
        }
        assert!(srv.is_success_rate_declining());
    }

    #[test]
    fn test_is_success_rate_declining_insufficient() {
        let mut srv = GrpcServer::with_config(small_config());
        srv.tick();
        assert!(!srv.is_success_rate_declining());
    }

    #[test]
    fn test_window_rolls() {
        let mut srv = GrpcServer::with_config(small_config()); // window_size = 5
        for _ in 0..20 {
            srv.tick();
        }
        assert!(srv.recent.len() <= 5);
    }

    // -------------------------------------------------------------------
    // Reset
    // -------------------------------------------------------------------

    #[test]
    fn test_reset() {
        let mut srv = GrpcServer::with_config(small_config());
        srv.register_service("svc").unwrap();
        srv.register_method("svc", "Get", MethodType::Unary).unwrap();
        srv.record_call("svc", "Get", CallStatus::Ok, 50.0).unwrap();
        let _id = srv.open_stream("svc", "Get").unwrap();
        srv.tick();

        srv.reset();

        assert_eq!(srv.current_tick(), 0);
        assert_eq!(srv.stats().total_calls, 0);
        assert_eq!(srv.stats().total_successes, 0);
        assert_eq!(srv.active_stream_count(), 0);
        assert!(srv.windowed_success_rate().is_none());

        // Services and methods still registered but reset.
        assert_eq!(srv.service_count(), 1);
        let method = srv.method("svc", "Get").unwrap();
        assert_eq!(method.total_calls, 0);
        assert!(method.call_history().is_empty());
    }

    // -------------------------------------------------------------------
    // Full lifecycle
    // -------------------------------------------------------------------

    #[test]
    fn test_full_lifecycle() {
        let mut srv = GrpcServer::with_config(small_config());

        // Register services and methods.
        srv.register_service("MarketData").unwrap();
        srv.register_method("MarketData", "GetQuote", MethodType::Unary)
            .unwrap();
        srv.register_method("MarketData", "StreamPrices", MethodType::ServerStreaming)
            .unwrap();

        srv.register_service("Signals").unwrap();
        srv.register_method("Signals", "Submit", MethodType::Unary)
            .unwrap();

        // Record some calls.
        srv.record_call("MarketData", "GetQuote", CallStatus::Ok, 30.0)
            .unwrap();
        srv.record_call("Signals", "Submit", CallStatus::Ok, 20.0)
            .unwrap();
        srv.record_call("MarketData", "GetQuote", CallStatus::Error, 80.0)
            .unwrap();
        srv.tick();

        // Open a stream.
        let stream_id = srv.open_stream("MarketData", "StreamPrices").unwrap();
        srv.stream_send(stream_id).unwrap();
        srv.stream_send(stream_id).unwrap();
        srv.tick();

        // Close stream.
        let info = srv.close_stream(stream_id).unwrap();
        assert_eq!(info.messages_sent, 2);
        srv.tick();

        // Verify stats.
        assert_eq!(srv.stats().total_calls, 3);
        assert_eq!(srv.stats().total_successes, 2);
        assert_eq!(srv.stats().total_errors, 1);
        assert_eq!(srv.stats().total_streams_opened, 1);
        assert_eq!(srv.stats().total_streams_closed, 1);
        assert!(srv.smoothed_call_rate() > 0.0);
        assert!(srv.windowed_success_rate().is_some());
    }

    // -------------------------------------------------------------------
    // Display coverage
    // -------------------------------------------------------------------

    #[test]
    fn test_method_type_display() {
        assert_eq!(format!("{}", MethodType::Unary), "Unary");
        assert_eq!(format!("{}", MethodType::ServerStreaming), "ServerStreaming");
        assert_eq!(format!("{}", MethodType::ClientStreaming), "ClientStreaming");
        assert_eq!(
            format!("{}", MethodType::BidirectionalStreaming),
            "BidirectionalStreaming"
        );
    }

    #[test]
    fn test_call_status_display() {
        assert_eq!(format!("{}", CallStatus::Ok), "OK");
        assert_eq!(format!("{}", CallStatus::Error), "Error");
        assert_eq!(format!("{}", CallStatus::Cancelled), "Cancelled");
        assert_eq!(
            format!("{}", CallStatus::DeadlineExceeded),
            "DeadlineExceeded"
        );
        assert_eq!(format!("{}", CallStatus::Internal), "Internal");
    }
}
