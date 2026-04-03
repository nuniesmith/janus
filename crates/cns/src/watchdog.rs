//! # CNS Watchdog
//!
//! Runtime health monitoring for the JANUS trading system. The watchdog runs
//! continuously after boot, periodically checking system health and taking
//! corrective action when components become unresponsive.
//!
//! ## Responsibilities
//!
//! - **Heartbeat monitoring**: Track heartbeats from all registered components.
//!   If a component misses too many heartbeats, mark it as degraded or dead.
//! - **Emergency kill**: If critical components fail, trigger an emergency
//!   kill switch to halt all trading activity immediately.
//! - **Prometheus metrics**: Export watchdog health metrics for dashboards.
//! - **Auto-recovery**: Attempt to restart failed components (if configured).
//!
//! ## Architecture
//!
//! The watchdog runs as a background tokio task. Components register themselves
//! and periodically send heartbeats. The watchdog checks for stale heartbeats
//! on a configurable interval and takes action based on component criticality.
//!
//! ```text
//! ┌─────────────┐     heartbeat()      ┌─────────────┐
//! │  Component A │ ──────────────────▶  │             │
//! └─────────────┘                       │             │
//! ┌─────────────┐     heartbeat()      │  Watchdog   │──▶ Prometheus
//! │  Component B │ ──────────────────▶  │             │──▶ Kill Switch
//! └─────────────┘                       │             │──▶ Alerts
//! ┌─────────────┐     heartbeat()      │             │
//! │  Component C │ ──────────────────▶  │             │
//! └─────────────┘                       └─────────────┘
//! ```

use chrono::{DateTime, Utc};
use serde::{Deserialize, Serialize};
use std::collections::HashMap;
use std::sync::Arc;
use std::time::{Duration, Instant};
use tokio::sync::{RwLock, broadcast, watch};
use tracing::{debug, error, info, warn};

// ============================================================================
// Watchdog Configuration
// ============================================================================

/// Configuration for the CNS Watchdog.
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct WatchdogConfig {
    /// How often the watchdog checks heartbeats (in milliseconds).
    pub check_interval_ms: u64,

    /// How many missed heartbeat intervals before a component is considered degraded.
    pub degraded_threshold: u32,

    /// How many missed heartbeat intervals before a component is considered dead.
    pub dead_threshold: u32,

    /// Whether to trigger the kill switch when a critical component dies.
    pub kill_on_critical_death: bool,

    /// Whether to attempt auto-restart of failed components.
    pub auto_restart_enabled: bool,

    /// Maximum number of auto-restart attempts per component before giving up.
    pub max_restart_attempts: u32,

    /// Cooldown period between restart attempts (in seconds).
    pub restart_cooldown_secs: u64,

    /// Whether to emit Prometheus metrics.
    pub metrics_enabled: bool,

    /// Target time-to-kill when the kill switch is triggered (in milliseconds).
    /// The watchdog will log a warning if the actual kill time exceeds this.
    pub target_kill_time_ms: u64,
}

impl Default for WatchdogConfig {
    fn default() -> Self {
        Self {
            check_interval_ms: 1000,
            degraded_threshold: 3,
            dead_threshold: 5,
            kill_on_critical_death: true,
            auto_restart_enabled: false,
            max_restart_attempts: 3,
            restart_cooldown_secs: 10,
            metrics_enabled: true,
            target_kill_time_ms: 100,
        }
    }
}

// ============================================================================
// Component Registration
// ============================================================================

/// How critical a monitored component is to system operation.
#[derive(Debug, Clone, Copy, PartialEq, Eq, PartialOrd, Ord, Hash, Serialize, Deserialize)]
pub enum ComponentCriticality {
    /// System MUST halt trading if this component dies.
    Critical,
    /// System should degrade gracefully (reduce exposure, widen stops).
    Important,
    /// System can continue normally without this component.
    NonEssential,
}

impl std::fmt::Display for ComponentCriticality {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        match self {
            ComponentCriticality::Critical => write!(f, "CRITICAL"),
            ComponentCriticality::Important => write!(f, "IMPORTANT"),
            ComponentCriticality::NonEssential => write!(f, "NON-ESSENTIAL"),
        }
    }
}

/// Health state of a monitored component.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Serialize, Deserialize)]
pub enum ComponentState {
    /// Component is healthy and sending heartbeats.
    Alive,
    /// Component has missed some heartbeats but hasn't exceeded the dead threshold.
    Degraded,
    /// Component has missed too many heartbeats and is considered dead.
    Dead,
    /// Component has not yet sent its first heartbeat.
    Pending,
}

impl std::fmt::Display for ComponentState {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        match self {
            ComponentState::Alive => write!(f, "ALIVE"),
            ComponentState::Degraded => write!(f, "DEGRADED"),
            ComponentState::Dead => write!(f, "DEAD"),
            ComponentState::Pending => write!(f, "PENDING"),
        }
    }
}

impl ComponentState {
    /// Prometheus-friendly numeric value.
    pub fn as_metric(&self) -> f64 {
        match self {
            ComponentState::Alive => 1.0,
            ComponentState::Degraded => 0.5,
            ComponentState::Dead => 0.0,
            ComponentState::Pending => 0.25,
        }
    }
}

/// Registration info for a component being monitored.
#[derive(Debug, Clone)]
pub struct ComponentRegistration {
    /// Unique name of the component.
    pub name: String,
    /// How critical this component is.
    pub criticality: ComponentCriticality,
    /// Expected heartbeat interval. If the component doesn't heartbeat within
    /// `heartbeat_interval * degraded_threshold`, it's marked degraded.
    pub heartbeat_interval: Duration,
    /// Optional description.
    pub description: Option<String>,
}

/// Internal state for a monitored component.
#[derive(Debug, Clone)]
struct ComponentInfo {
    registration: ComponentRegistration,
    state: ComponentState,
    last_heartbeat: Option<Instant>,
    last_heartbeat_utc: Option<DateTime<Utc>>,
    missed_heartbeats: u32,
    restart_attempts: u32,
    #[allow(dead_code)]
    last_restart: Option<Instant>,
    registered_at: Instant,
}

// ============================================================================
// Watchdog Events
// ============================================================================

/// Events emitted by the watchdog that consumers can react to.
#[derive(Debug, Clone)]
pub enum WatchdogEvent {
    /// A component's state changed.
    StateChange {
        component: String,
        old_state: ComponentState,
        new_state: ComponentState,
        criticality: ComponentCriticality,
    },
    /// Emergency kill switch triggered.
    KillTriggered { reason: String, component: String },
    /// A component was auto-restarted.
    AutoRestart { component: String, attempt: u32 },
    /// The watchdog itself started.
    Started,
    /// The watchdog itself stopped.
    Stopped,
}

// ============================================================================
// Watchdog Metrics
// ============================================================================

/// Prometheus metrics for the watchdog.
pub struct WatchdogMetrics {
    pub component_state: prometheus::GaugeVec,
    pub component_missed_heartbeats: prometheus::GaugeVec,
    pub component_last_heartbeat_age_secs: prometheus::GaugeVec,
    pub kill_switch_triggered_total: prometheus::Counter,
    pub auto_restart_total: prometheus::CounterVec,
    pub watchdog_check_duration_ms: prometheus::Histogram,
    pub watchdog_uptime_seconds: prometheus::Gauge,
    pub total_components_registered: prometheus::Gauge,
    pub alive_components: prometheus::Gauge,
    pub degraded_components: prometheus::Gauge,
    pub dead_components: prometheus::Gauge,
}

impl WatchdogMetrics {
    /// Create a new set of watchdog metrics registered with the given registry.
    pub fn new(registry: &prometheus::Registry) -> Result<Self, prometheus::Error> {
        let component_state = prometheus::GaugeVec::new(
            prometheus::Opts::new(
                "janus_watchdog_component_state",
                "Health state of a monitored component (1=alive, 0.5=degraded, 0=dead, 0.25=pending)",
            ),
            &["component", "criticality"],
        )?;
        registry.register(Box::new(component_state.clone()))?;

        let component_missed_heartbeats = prometheus::GaugeVec::new(
            prometheus::Opts::new(
                "janus_watchdog_component_missed_heartbeats",
                "Number of missed heartbeats for a component",
            ),
            &["component"],
        )?;
        registry.register(Box::new(component_missed_heartbeats.clone()))?;

        let component_last_heartbeat_age_secs = prometheus::GaugeVec::new(
            prometheus::Opts::new(
                "janus_watchdog_component_last_heartbeat_age_seconds",
                "Seconds since the last heartbeat from a component",
            ),
            &["component"],
        )?;
        registry.register(Box::new(component_last_heartbeat_age_secs.clone()))?;

        let kill_switch_triggered_total = prometheus::Counter::new(
            "janus_watchdog_kill_switch_triggered_total",
            "Total number of times the kill switch was triggered by the watchdog",
        )?;
        registry.register(Box::new(kill_switch_triggered_total.clone()))?;

        let auto_restart_total = prometheus::CounterVec::new(
            prometheus::Opts::new(
                "janus_watchdog_auto_restart_total",
                "Total auto-restart attempts per component",
            ),
            &["component"],
        )?;
        registry.register(Box::new(auto_restart_total.clone()))?;

        let watchdog_check_duration_ms = prometheus::Histogram::with_opts(
            prometheus::HistogramOpts::new(
                "janus_watchdog_check_duration_milliseconds",
                "Duration of each watchdog health check cycle in milliseconds",
            )
            .buckets(vec![0.1, 0.5, 1.0, 2.0, 5.0, 10.0, 25.0, 50.0, 100.0]),
        )?;
        registry.register(Box::new(watchdog_check_duration_ms.clone()))?;

        let watchdog_uptime_seconds = prometheus::Gauge::new(
            "janus_watchdog_uptime_seconds",
            "Watchdog uptime in seconds",
        )?;
        registry.register(Box::new(watchdog_uptime_seconds.clone()))?;

        let total_components_registered = prometheus::Gauge::new(
            "janus_watchdog_components_registered_total",
            "Total number of components registered with the watchdog",
        )?;
        registry.register(Box::new(total_components_registered.clone()))?;

        let alive_components = prometheus::Gauge::new(
            "janus_watchdog_components_alive",
            "Number of components in ALIVE state",
        )?;
        registry.register(Box::new(alive_components.clone()))?;

        let degraded_components = prometheus::Gauge::new(
            "janus_watchdog_components_degraded",
            "Number of components in DEGRADED state",
        )?;
        registry.register(Box::new(degraded_components.clone()))?;

        let dead_components = prometheus::Gauge::new(
            "janus_watchdog_components_dead",
            "Number of components in DEAD state",
        )?;
        registry.register(Box::new(dead_components.clone()))?;

        Ok(Self {
            component_state,
            component_missed_heartbeats,
            component_last_heartbeat_age_secs,
            kill_switch_triggered_total,
            auto_restart_total,
            watchdog_check_duration_ms,
            watchdog_uptime_seconds,
            total_components_registered,
            alive_components,
            degraded_components,
            dead_components,
        })
    }

    /// Create metrics without registering them (for testing).
    pub fn unregistered() -> Self {
        Self {
            component_state: prometheus::GaugeVec::new(
                prometheus::Opts::new("test_state", "test"),
                &["component", "criticality"],
            )
            .unwrap(),
            component_missed_heartbeats: prometheus::GaugeVec::new(
                prometheus::Opts::new("test_missed", "test"),
                &["component"],
            )
            .unwrap(),
            component_last_heartbeat_age_secs: prometheus::GaugeVec::new(
                prometheus::Opts::new("test_age", "test"),
                &["component"],
            )
            .unwrap(),
            kill_switch_triggered_total: prometheus::Counter::new("test_kill", "test").unwrap(),
            auto_restart_total: prometheus::CounterVec::new(
                prometheus::Opts::new("test_restart", "test"),
                &["component"],
            )
            .unwrap(),
            watchdog_check_duration_ms: prometheus::Histogram::with_opts(
                prometheus::HistogramOpts::new("test_duration", "test"),
            )
            .unwrap(),
            watchdog_uptime_seconds: prometheus::Gauge::new("test_uptime", "test").unwrap(),
            total_components_registered: prometheus::Gauge::new("test_total", "test").unwrap(),
            alive_components: prometheus::Gauge::new("test_alive", "test").unwrap(),
            degraded_components: prometheus::Gauge::new("test_degraded", "test").unwrap(),
            dead_components: prometheus::Gauge::new("test_dead", "test").unwrap(),
        }
    }
}

// ============================================================================
// Kill Switch Interface
// ============================================================================

/// Trait for kill switch implementations. The watchdog calls this when a
/// critical component dies and `kill_on_critical_death` is enabled.
#[async_trait::async_trait]
pub trait KillSwitch: Send + Sync {
    /// Trigger the kill switch. This should:
    /// 1. Cancel all open orders
    /// 2. Stop accepting new signals
    /// 3. Optionally flatten all positions
    ///
    /// Returns Ok(()) if the kill was successful, Err if something went wrong.
    async fn trigger_kill(&self, reason: &str) -> Result<(), String>;

    /// Check if the kill switch is currently active (already killed).
    async fn is_killed(&self) -> bool;
}

/// A no-op kill switch for testing.
pub struct NoOpKillSwitch;

#[async_trait::async_trait]
impl KillSwitch for NoOpKillSwitch {
    async fn trigger_kill(&self, reason: &str) -> Result<(), String> {
        warn!("NoOpKillSwitch::trigger_kill called: {}", reason);
        Ok(())
    }

    async fn is_killed(&self) -> bool {
        false
    }
}

/// A Redis-backed kill switch that sets a key to signal other services to halt.
pub struct RedisKillSwitch {
    url: String,
    key: String,
}

impl RedisKillSwitch {
    pub fn new(url: impl Into<String>, key: impl Into<String>) -> Self {
        Self {
            url: url.into(),
            key: key.into(),
        }
    }
}

#[async_trait::async_trait]
impl KillSwitch for RedisKillSwitch {
    async fn trigger_kill(&self, reason: &str) -> Result<(), String> {
        let client = redis::Client::open(self.url.as_str())
            .map_err(|e| format!("Kill switch Redis client error: {}", e))?;

        let mut conn = client
            .get_multiplexed_async_connection()
            .await
            .map_err(|e| format!("Kill switch Redis connection error: {}", e))?;

        // Set the kill switch key with the reason
        let kill_value = format!("KILLED|{}|{}", reason, Utc::now().to_rfc3339());
        redis::cmd("SET")
            .arg(&self.key)
            .arg(&kill_value)
            .query_async::<()>(&mut conn)
            .await
            .map_err(|e| format!("Kill switch Redis SET error: {}", e))?;

        info!(
            "🚨 KILL SWITCH TRIGGERED via Redis key '{}': {}",
            self.key, reason
        );

        Ok(())
    }

    async fn is_killed(&self) -> bool {
        let client = match redis::Client::open(self.url.as_str()) {
            Ok(c) => c,
            Err(_) => return false,
        };

        let mut conn = match client.get_multiplexed_async_connection().await {
            Ok(c) => c,
            Err(_) => return false,
        };

        let value: Option<String> = redis::cmd("GET")
            .arg(&self.key)
            .query_async(&mut conn)
            .await
            .unwrap_or(None);

        value.map(|v| v.starts_with("KILLED")).unwrap_or(false)
    }
}

// ============================================================================
// Watchdog Handle
// ============================================================================

/// A lightweight handle that components use to send heartbeats to the watchdog.
/// Cloneable and thread-safe.
#[derive(Clone)]
pub struct WatchdogHandle {
    inner: Arc<RwLock<HashMap<String, ComponentInfo>>>,
}

impl WatchdogHandle {
    /// Send a heartbeat for the named component.
    ///
    /// This should be called periodically by each registered component
    /// at the interval specified in its registration.
    pub async fn heartbeat(&self, component_name: &str) {
        let mut components = self.inner.write().await;
        if let Some(info) = components.get_mut(component_name) {
            info.last_heartbeat = Some(Instant::now());
            info.last_heartbeat_utc = Some(Utc::now());
            info.missed_heartbeats = 0;

            if info.state != ComponentState::Alive {
                let old_state = info.state;
                info.state = ComponentState::Alive;
                debug!(
                    "Component '{}' recovered: {} → ALIVE",
                    component_name, old_state
                );
            }
        } else {
            warn!(
                "Heartbeat received for unregistered component '{}'",
                component_name
            );
        }
    }

    /// Check the current state of a component.
    pub async fn component_state(&self, component_name: &str) -> Option<ComponentState> {
        let components = self.inner.read().await;
        components.get(component_name).map(|info| info.state)
    }

    /// Get the number of registered components.
    pub async fn component_count(&self) -> usize {
        let components = self.inner.read().await;
        components.len()
    }

    /// Get a summary of all component states.
    pub async fn summary(&self) -> HashMap<String, ComponentState> {
        let components = self.inner.read().await;
        components
            .iter()
            .map(|(name, info)| (name.clone(), info.state))
            .collect()
    }
}

// ============================================================================
// CNS Watchdog
// ============================================================================

/// The CNS Watchdog — runtime health monitor for the JANUS system.
///
/// After boot (post pre-flight checks), the watchdog runs continuously,
/// monitoring heartbeats from all registered components and taking action
/// when components become unresponsive.
pub struct CnsWatchdog {
    config: WatchdogConfig,
    components: Arc<RwLock<HashMap<String, ComponentInfo>>>,
    kill_switch: Arc<dyn KillSwitch>,
    metrics: Option<Arc<WatchdogMetrics>>,
    event_tx: broadcast::Sender<WatchdogEvent>,
    shutdown_tx: watch::Sender<bool>,
    shutdown_rx: watch::Receiver<bool>,
    started_at: Option<Instant>,
}

impl CnsWatchdog {
    /// Create a new watchdog with the given configuration and kill switch.
    pub fn new(config: WatchdogConfig, kill_switch: Arc<dyn KillSwitch>) -> Self {
        let (event_tx, _) = broadcast::channel(256);
        let (shutdown_tx, shutdown_rx) = watch::channel(false);

        Self {
            config,
            components: Arc::new(RwLock::new(HashMap::new())),
            kill_switch,
            metrics: None,
            event_tx,
            shutdown_tx,
            shutdown_rx,
            started_at: None,
        }
    }

    /// Create a watchdog with default config and a no-op kill switch (for testing).
    pub fn default_for_testing() -> Self {
        Self::new(WatchdogConfig::default(), Arc::new(NoOpKillSwitch))
    }

    /// Attach Prometheus metrics.
    pub fn with_metrics(mut self, metrics: Arc<WatchdogMetrics>) -> Self {
        self.metrics = Some(metrics);
        self
    }

    /// Get a handle that components can use to send heartbeats.
    pub fn handle(&self) -> WatchdogHandle {
        WatchdogHandle {
            inner: Arc::clone(&self.components),
        }
    }

    /// Subscribe to watchdog events.
    pub fn subscribe(&self) -> broadcast::Receiver<WatchdogEvent> {
        self.event_tx.subscribe()
    }

    /// Register a component for monitoring.
    pub async fn register(&self, registration: ComponentRegistration) {
        let mut components = self.components.write().await;

        info!(
            "Watchdog: registering component '{}' (criticality: {}, heartbeat interval: {:?})",
            registration.name, registration.criticality, registration.heartbeat_interval
        );

        let info = ComponentInfo {
            registration: registration.clone(),
            state: ComponentState::Pending,
            last_heartbeat: None,
            last_heartbeat_utc: None,
            missed_heartbeats: 0,
            restart_attempts: 0,
            last_restart: None,
            registered_at: Instant::now(),
        };

        components.insert(registration.name.clone(), info);
    }

    /// Unregister a component.
    pub async fn unregister(&self, component_name: &str) {
        let mut components = self.components.write().await;
        if components.remove(component_name).is_some() {
            info!("Watchdog: unregistered component '{}'", component_name);
        }
    }

    /// Get the number of registered components.
    pub async fn component_count(&self) -> usize {
        let components = self.components.read().await;
        components.len()
    }

    /// Get uptime in seconds (if started).
    pub fn uptime_secs(&self) -> Option<f64> {
        self.started_at.map(|s| s.elapsed().as_secs_f64())
    }

    /// Run the watchdog loop. This blocks until `stop()` is called.
    ///
    /// Typically spawned as a background task:
    /// ```rust,no_run
    /// # use janus_cns::watchdog::{CnsWatchdog, WatchdogConfig};
    /// # use std::sync::Arc;
    /// # use janus_cns::watchdog::NoOpKillSwitch;
    /// # async fn example() {
    /// let watchdog = CnsWatchdog::new(WatchdogConfig::default(), Arc::new(NoOpKillSwitch));
    /// let handle = watchdog.handle();
    /// tokio::spawn(async move { watchdog.run().await });
    /// # }
    /// ```
    pub async fn run(mut self) {
        self.started_at = Some(Instant::now());
        let check_interval = Duration::from_millis(self.config.check_interval_ms);

        info!(
            "🐕 Watchdog started (check interval: {:?}, kill_on_critical: {})",
            check_interval, self.config.kill_on_critical_death
        );

        let _ = self.event_tx.send(WatchdogEvent::Started);

        let mut shutdown_rx = self.shutdown_rx.clone();

        loop {
            tokio::select! {
                _ = tokio::time::sleep(check_interval) => {
                    self.check_cycle().await;
                }
                _ = shutdown_rx.changed() => {
                    if *shutdown_rx.borrow() {
                        info!("🐕 Watchdog received shutdown signal");
                        break;
                    }
                }
            }
        }

        let _ = self.event_tx.send(WatchdogEvent::Stopped);
        info!("🐕 Watchdog stopped");
    }

    /// Stop the watchdog.
    pub fn stop(&self) {
        let _ = self.shutdown_tx.send(true);
    }

    /// Perform one check cycle (exposed for testing).
    pub async fn check_cycle(&self) {
        let start = Instant::now();
        let mut components = self.components.write().await;

        let mut state_changes: Vec<WatchdogEvent> = Vec::new();
        let mut kill_triggered = false;
        let mut kill_reason = String::new();
        let mut kill_component = String::new();

        for (name, info) in components.iter_mut() {
            let old_state = info.state;

            // Calculate missed heartbeats
            if let Some(last_hb) = info.last_heartbeat {
                let elapsed = last_hb.elapsed();
                let expected_interval = info.registration.heartbeat_interval;

                if expected_interval.as_nanos() > 0 {
                    let intervals_elapsed =
                        (elapsed.as_nanos() / expected_interval.as_nanos()) as u32;

                    if intervals_elapsed > 0 {
                        info.missed_heartbeats = intervals_elapsed;
                    }
                }
            } else if info.state == ComponentState::Pending {
                // Component hasn't sent first heartbeat yet.
                // Check if it's been too long since registration.
                let since_registration = info.registered_at.elapsed();
                let grace_period =
                    info.registration.heartbeat_interval * self.config.dead_threshold;

                if since_registration > grace_period {
                    info.missed_heartbeats = self.config.dead_threshold;
                }
            }

            // Determine new state
            let new_state = if info.missed_heartbeats >= self.config.dead_threshold {
                ComponentState::Dead
            } else if info.missed_heartbeats >= self.config.degraded_threshold {
                ComponentState::Degraded
            } else if info.last_heartbeat.is_some() {
                ComponentState::Alive
            } else {
                ComponentState::Pending
            };

            if new_state != old_state {
                info.state = new_state;

                let event = WatchdogEvent::StateChange {
                    component: name.clone(),
                    old_state,
                    new_state,
                    criticality: info.registration.criticality,
                };

                match new_state {
                    ComponentState::Dead => {
                        error!(
                            "🐕 Component '{}' is DEAD ({} missed heartbeats, criticality: {})",
                            name, info.missed_heartbeats, info.registration.criticality
                        );

                        // Check if we need to trigger kill switch
                        if info.registration.criticality == ComponentCriticality::Critical
                            && self.config.kill_on_critical_death
                            && !kill_triggered
                        {
                            kill_triggered = true;
                            kill_reason = format!(
                                "Critical component '{}' is dead ({} missed heartbeats)",
                                name, info.missed_heartbeats
                            );
                            kill_component = name.clone();
                        }
                    }
                    ComponentState::Degraded => {
                        warn!(
                            "🐕 Component '{}' is DEGRADED ({} missed heartbeats)",
                            name, info.missed_heartbeats
                        );
                    }
                    ComponentState::Alive => {
                        info!(
                            "🐕 Component '{}' recovered to ALIVE from {}",
                            name, old_state
                        );
                    }
                    ComponentState::Pending => {}
                }

                state_changes.push(event);
            }
        }

        // Compute summary stats for metrics
        let mut alive_count = 0u32;
        let mut degraded_count = 0u32;
        let mut dead_count = 0u32;
        let total = components.len();

        for info in components.values() {
            match info.state {
                ComponentState::Alive => alive_count += 1,
                ComponentState::Degraded => degraded_count += 1,
                ComponentState::Dead => dead_count += 1,
                ComponentState::Pending => {}
            }
        }

        // Update Prometheus metrics
        if let Some(ref metrics) = self.metrics {
            for (name, info) in components.iter() {
                let criticality_label = format!("{}", info.registration.criticality);
                metrics
                    .component_state
                    .with_label_values(&[name, &criticality_label])
                    .set(info.state.as_metric());

                metrics
                    .component_missed_heartbeats
                    .with_label_values(&[name])
                    .set(info.missed_heartbeats as f64);

                if let Some(last_hb) = info.last_heartbeat {
                    metrics
                        .component_last_heartbeat_age_secs
                        .with_label_values(&[name])
                        .set(last_hb.elapsed().as_secs_f64());
                }
            }

            if let Some(started) = self.started_at {
                metrics
                    .watchdog_uptime_seconds
                    .set(started.elapsed().as_secs_f64());
            }

            metrics.total_components_registered.set(total as f64);
            metrics.alive_components.set(alive_count as f64);
            metrics.degraded_components.set(degraded_count as f64);
            metrics.dead_components.set(dead_count as f64);

            let check_duration_ms = start.elapsed().as_secs_f64() * 1000.0;
            metrics
                .watchdog_check_duration_ms
                .observe(check_duration_ms);
        }

        // Drop the write lock before doing async operations
        drop(components);

        // Emit state change events
        for event in state_changes {
            let _ = self.event_tx.send(event);
        }

        // Trigger kill switch if needed
        if kill_triggered {
            error!("🚨 WATCHDOG TRIGGERING KILL SWITCH: {}", kill_reason);

            let kill_start = Instant::now();
            match self.kill_switch.trigger_kill(&kill_reason).await {
                Ok(()) => {
                    let kill_time_ms = kill_start.elapsed().as_millis() as u64;
                    info!(
                        "🚨 Kill switch triggered successfully in {}ms (target: {}ms)",
                        kill_time_ms, self.config.target_kill_time_ms
                    );

                    if kill_time_ms > self.config.target_kill_time_ms {
                        warn!(
                            "⚠️ Kill time {}ms exceeded target {}ms",
                            kill_time_ms, self.config.target_kill_time_ms
                        );
                    }

                    if let Some(ref metrics) = self.metrics {
                        metrics.kill_switch_triggered_total.inc();
                    }
                }
                Err(e) => {
                    error!("🚨 KILL SWITCH FAILED: {} — SYSTEM IN UNKNOWN STATE", e);
                }
            }

            let _ = self.event_tx.send(WatchdogEvent::KillTriggered {
                reason: kill_reason,
                component: kill_component,
            });
        }
    }

    /// Get a snapshot of all component states (for API/dashboard).
    pub async fn snapshot(&self) -> WatchdogSnapshot {
        let components = self.components.read().await;

        let component_states: Vec<ComponentSnapshot> = components
            .iter()
            .map(|(name, info)| ComponentSnapshot {
                name: name.clone(),
                state: info.state,
                criticality: info.registration.criticality,
                missed_heartbeats: info.missed_heartbeats,
                last_heartbeat_utc: info.last_heartbeat_utc,
                heartbeat_interval_ms: info.registration.heartbeat_interval.as_millis() as u64,
                restart_attempts: info.restart_attempts,
            })
            .collect();

        let alive = component_states
            .iter()
            .filter(|c| c.state == ComponentState::Alive)
            .count();
        let degraded = component_states
            .iter()
            .filter(|c| c.state == ComponentState::Degraded)
            .count();
        let dead = component_states
            .iter()
            .filter(|c| c.state == ComponentState::Dead)
            .count();

        WatchdogSnapshot {
            uptime_secs: self.uptime_secs().unwrap_or(0.0),
            total_components: component_states.len(),
            alive_count: alive,
            degraded_count: degraded,
            dead_count: dead,
            components: component_states,
            timestamp: Utc::now(),
        }
    }
}

// ============================================================================
// Snapshot Types
// ============================================================================

/// Point-in-time snapshot of a single component's watchdog state.
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct ComponentSnapshot {
    pub name: String,
    pub state: ComponentState,
    pub criticality: ComponentCriticality,
    pub missed_heartbeats: u32,
    pub last_heartbeat_utc: Option<DateTime<Utc>>,
    pub heartbeat_interval_ms: u64,
    pub restart_attempts: u32,
}

/// Point-in-time snapshot of the entire watchdog state.
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct WatchdogSnapshot {
    pub uptime_secs: f64,
    pub total_components: usize,
    pub alive_count: usize,
    pub degraded_count: usize,
    pub dead_count: usize,
    pub components: Vec<ComponentSnapshot>,
    pub timestamp: DateTime<Utc>,
}

impl WatchdogSnapshot {
    /// Overall health score (0.0 to 1.0).
    pub fn health_score(&self) -> f64 {
        if self.total_components == 0 {
            return 1.0;
        }

        let score: f64 = self.components.iter().map(|c| c.state.as_metric()).sum();
        score / self.total_components as f64
    }

    /// Whether the system is in a healthy or at least operational state.
    pub fn is_operational(&self) -> bool {
        // Operational if no critical components are dead
        !self.components.iter().any(|c| {
            c.state == ComponentState::Dead && c.criticality == ComponentCriticality::Critical
        })
    }
}

// ============================================================================
// Tests
// ============================================================================

#[cfg(test)]
mod tests {
    use super::*;

    // ── Config tests ──

    #[test]
    fn test_default_config() {
        let config = WatchdogConfig::default();
        assert_eq!(config.check_interval_ms, 1000);
        assert_eq!(config.degraded_threshold, 3);
        assert_eq!(config.dead_threshold, 5);
        assert!(config.kill_on_critical_death);
        assert!(!config.auto_restart_enabled);
        assert_eq!(config.max_restart_attempts, 3);
        assert_eq!(config.target_kill_time_ms, 100);
    }

    // ── ComponentCriticality tests ──

    #[test]
    fn test_criticality_display() {
        assert_eq!(format!("{}", ComponentCriticality::Critical), "CRITICAL");
        assert_eq!(format!("{}", ComponentCriticality::Important), "IMPORTANT");
        assert_eq!(
            format!("{}", ComponentCriticality::NonEssential),
            "NON-ESSENTIAL"
        );
    }

    #[test]
    fn test_criticality_ordering() {
        assert!(ComponentCriticality::Critical < ComponentCriticality::Important);
        assert!(ComponentCriticality::Important < ComponentCriticality::NonEssential);
    }

    // ── ComponentState tests ──

    #[test]
    fn test_state_display() {
        assert_eq!(format!("{}", ComponentState::Alive), "ALIVE");
        assert_eq!(format!("{}", ComponentState::Degraded), "DEGRADED");
        assert_eq!(format!("{}", ComponentState::Dead), "DEAD");
        assert_eq!(format!("{}", ComponentState::Pending), "PENDING");
    }

    #[test]
    fn test_state_as_metric() {
        assert_eq!(ComponentState::Alive.as_metric(), 1.0);
        assert_eq!(ComponentState::Degraded.as_metric(), 0.5);
        assert_eq!(ComponentState::Dead.as_metric(), 0.0);
        assert_eq!(ComponentState::Pending.as_metric(), 0.25);
    }

    // ── WatchdogHandle tests ──

    #[tokio::test]
    async fn test_handle_heartbeat_unregistered() {
        let watchdog = CnsWatchdog::default_for_testing();
        let handle = watchdog.handle();
        // Should just log a warning, not panic
        handle.heartbeat("nonexistent").await;
    }

    #[tokio::test]
    async fn test_handle_component_state() {
        let watchdog = CnsWatchdog::default_for_testing();
        let handle = watchdog.handle();

        // Unregistered component returns None
        assert!(handle.component_state("test").await.is_none());

        // Register and check
        watchdog
            .register(ComponentRegistration {
                name: "test".to_string(),
                criticality: ComponentCriticality::Important,
                heartbeat_interval: Duration::from_secs(1),
                description: None,
            })
            .await;

        let state = handle.component_state("test").await;
        assert_eq!(state, Some(ComponentState::Pending));
    }

    #[tokio::test]
    async fn test_handle_heartbeat_changes_state_to_alive() {
        let watchdog = CnsWatchdog::default_for_testing();
        let handle = watchdog.handle();

        watchdog
            .register(ComponentRegistration {
                name: "test".to_string(),
                criticality: ComponentCriticality::Critical,
                heartbeat_interval: Duration::from_secs(1),
                description: None,
            })
            .await;

        assert_eq!(
            handle.component_state("test").await,
            Some(ComponentState::Pending)
        );

        handle.heartbeat("test").await;

        assert_eq!(
            handle.component_state("test").await,
            Some(ComponentState::Alive)
        );
    }

    #[tokio::test]
    async fn test_handle_component_count() {
        let watchdog = CnsWatchdog::default_for_testing();
        let handle = watchdog.handle();

        assert_eq!(handle.component_count().await, 0);

        watchdog
            .register(ComponentRegistration {
                name: "a".to_string(),
                criticality: ComponentCriticality::Critical,
                heartbeat_interval: Duration::from_secs(1),
                description: None,
            })
            .await;

        assert_eq!(handle.component_count().await, 1);

        watchdog
            .register(ComponentRegistration {
                name: "b".to_string(),
                criticality: ComponentCriticality::NonEssential,
                heartbeat_interval: Duration::from_secs(5),
                description: None,
            })
            .await;

        assert_eq!(handle.component_count().await, 2);
    }

    #[tokio::test]
    async fn test_handle_summary() {
        let watchdog = CnsWatchdog::default_for_testing();
        let handle = watchdog.handle();

        watchdog
            .register(ComponentRegistration {
                name: "comp_a".to_string(),
                criticality: ComponentCriticality::Critical,
                heartbeat_interval: Duration::from_secs(1),
                description: None,
            })
            .await;

        handle.heartbeat("comp_a").await;

        let summary = handle.summary().await;
        assert_eq!(summary.len(), 1);
        assert_eq!(summary.get("comp_a"), Some(&ComponentState::Alive));
    }

    // ── Watchdog registration tests ──

    #[tokio::test]
    async fn test_register_component() {
        let watchdog = CnsWatchdog::default_for_testing();

        assert_eq!(watchdog.component_count().await, 0);

        watchdog
            .register(ComponentRegistration {
                name: "forward".to_string(),
                criticality: ComponentCriticality::Critical,
                heartbeat_interval: Duration::from_secs(1),
                description: Some("Forward trading service".to_string()),
            })
            .await;

        assert_eq!(watchdog.component_count().await, 1);
    }

    #[tokio::test]
    async fn test_unregister_component() {
        let watchdog = CnsWatchdog::default_for_testing();

        watchdog
            .register(ComponentRegistration {
                name: "forward".to_string(),
                criticality: ComponentCriticality::Critical,
                heartbeat_interval: Duration::from_secs(1),
                description: None,
            })
            .await;

        assert_eq!(watchdog.component_count().await, 1);
        watchdog.unregister("forward").await;
        assert_eq!(watchdog.component_count().await, 0);
    }

    #[tokio::test]
    async fn test_unregister_nonexistent_is_noop() {
        let watchdog = CnsWatchdog::default_for_testing();
        watchdog.unregister("nonexistent").await;
        assert_eq!(watchdog.component_count().await, 0);
    }

    // ── Watchdog check cycle tests ──

    #[tokio::test]
    async fn test_check_cycle_alive_component() {
        let watchdog = CnsWatchdog::default_for_testing();
        let handle = watchdog.handle();

        watchdog
            .register(ComponentRegistration {
                name: "test".to_string(),
                criticality: ComponentCriticality::Critical,
                heartbeat_interval: Duration::from_secs(10),
                description: None,
            })
            .await;

        handle.heartbeat("test").await;
        watchdog.check_cycle().await;

        assert_eq!(
            handle.component_state("test").await,
            Some(ComponentState::Alive)
        );
    }

    #[tokio::test]
    async fn test_check_cycle_detects_stale_heartbeat() {
        let config = WatchdogConfig {
            degraded_threshold: 1,
            dead_threshold: 2,
            kill_on_critical_death: false, // Don't kill for this test
            ..Default::default()
        };
        let watchdog = CnsWatchdog::new(config, Arc::new(NoOpKillSwitch));
        let handle = watchdog.handle();

        watchdog
            .register(ComponentRegistration {
                name: "test".to_string(),
                criticality: ComponentCriticality::Critical,
                heartbeat_interval: Duration::from_millis(10),
                description: None,
            })
            .await;

        handle.heartbeat("test").await;

        // Wait for more than degraded_threshold * heartbeat_interval
        tokio::time::sleep(Duration::from_millis(30)).await;
        watchdog.check_cycle().await;

        let state = handle.component_state("test").await;
        // Should be degraded or dead depending on exact timing
        assert!(
            state == Some(ComponentState::Degraded) || state == Some(ComponentState::Dead),
            "Expected Degraded or Dead, got {:?}",
            state
        );
    }

    // ── Watchdog snapshot tests ──

    #[tokio::test]
    async fn test_snapshot_empty() {
        let watchdog = CnsWatchdog::default_for_testing();
        let snapshot = watchdog.snapshot().await;

        assert_eq!(snapshot.total_components, 0);
        assert_eq!(snapshot.alive_count, 0);
        assert_eq!(snapshot.dead_count, 0);
        assert_eq!(snapshot.health_score(), 1.0);
        assert!(snapshot.is_operational());
    }

    #[tokio::test]
    async fn test_snapshot_with_components() {
        let watchdog = CnsWatchdog::default_for_testing();
        let handle = watchdog.handle();

        watchdog
            .register(ComponentRegistration {
                name: "a".to_string(),
                criticality: ComponentCriticality::Critical,
                heartbeat_interval: Duration::from_secs(1),
                description: None,
            })
            .await;

        watchdog
            .register(ComponentRegistration {
                name: "b".to_string(),
                criticality: ComponentCriticality::NonEssential,
                heartbeat_interval: Duration::from_secs(1),
                description: None,
            })
            .await;

        handle.heartbeat("a").await;
        handle.heartbeat("b").await;

        let snapshot = watchdog.snapshot().await;
        assert_eq!(snapshot.total_components, 2);
        assert_eq!(snapshot.alive_count, 2);
        assert_eq!(snapshot.health_score(), 1.0);
        assert!(snapshot.is_operational());
    }

    #[tokio::test]
    async fn test_snapshot_health_score_degraded() {
        let watchdog = CnsWatchdog::default_for_testing();
        let handle = watchdog.handle();

        watchdog
            .register(ComponentRegistration {
                name: "a".to_string(),
                criticality: ComponentCriticality::Critical,
                heartbeat_interval: Duration::from_secs(1),
                description: None,
            })
            .await;

        watchdog
            .register(ComponentRegistration {
                name: "b".to_string(),
                criticality: ComponentCriticality::NonEssential,
                heartbeat_interval: Duration::from_secs(1),
                description: None,
            })
            .await;

        handle.heartbeat("a").await;
        // b stays Pending (0.25)

        let snapshot = watchdog.snapshot().await;
        // a=Alive(1.0), b=Pending(0.25) → average = 0.625
        assert!((snapshot.health_score() - 0.625).abs() < 0.001);
    }

    #[tokio::test]
    async fn test_snapshot_is_operational_false_when_critical_dead() {
        let config = WatchdogConfig {
            degraded_threshold: 1,
            dead_threshold: 1,
            kill_on_critical_death: false,
            ..Default::default()
        };
        let watchdog = CnsWatchdog::new(config, Arc::new(NoOpKillSwitch));
        let handle = watchdog.handle();

        watchdog
            .register(ComponentRegistration {
                name: "critical_comp".to_string(),
                criticality: ComponentCriticality::Critical,
                heartbeat_interval: Duration::from_millis(5),
                description: None,
            })
            .await;

        handle.heartbeat("critical_comp").await;

        // Wait for heartbeat to go stale
        tokio::time::sleep(Duration::from_millis(20)).await;
        watchdog.check_cycle().await;

        let snapshot = watchdog.snapshot().await;
        assert!(!snapshot.is_operational());
    }

    // ── NoOpKillSwitch tests ──

    #[tokio::test]
    async fn test_noop_kill_switch() {
        let ks = NoOpKillSwitch;
        assert!(!ks.is_killed().await);
        let result = ks.trigger_kill("test reason").await;
        assert!(result.is_ok());
        // NoOp doesn't actually track state
        assert!(!ks.is_killed().await);
    }

    // ── WatchdogMetrics tests ──

    #[test]
    fn test_unregistered_metrics() {
        let metrics = WatchdogMetrics::unregistered();
        metrics.watchdog_uptime_seconds.set(42.0);
        assert_eq!(metrics.watchdog_uptime_seconds.get(), 42.0);
    }

    #[test]
    fn test_registered_metrics() {
        let registry = prometheus::Registry::new();
        let result = WatchdogMetrics::new(&registry);
        assert!(result.is_ok());
    }

    #[test]
    fn test_metrics_double_registration_fails() {
        let registry = prometheus::Registry::new();
        let _m1 = WatchdogMetrics::new(&registry).unwrap();
        let result = WatchdogMetrics::new(&registry);
        assert!(result.is_err()); // Can't register same metrics twice
    }

    // ── Event subscription tests ──

    #[tokio::test]
    async fn test_event_subscription() {
        let config = WatchdogConfig {
            degraded_threshold: 1,
            dead_threshold: 2,
            kill_on_critical_death: false,
            ..Default::default()
        };
        let watchdog = CnsWatchdog::new(config, Arc::new(NoOpKillSwitch));
        let mut rx = watchdog.subscribe();
        let handle = watchdog.handle();

        watchdog
            .register(ComponentRegistration {
                name: "test".to_string(),
                criticality: ComponentCriticality::Important,
                heartbeat_interval: Duration::from_millis(5),
                description: None,
            })
            .await;

        handle.heartbeat("test").await;

        // Wait for heartbeat to go stale
        tokio::time::sleep(Duration::from_millis(25)).await;
        watchdog.check_cycle().await;

        // We should have received a state change event
        let event = tokio::time::timeout(Duration::from_millis(100), rx.recv()).await;
        assert!(event.is_ok());

        if let Ok(Ok(WatchdogEvent::StateChange {
            component,
            new_state,
            ..
        })) = event
        {
            assert_eq!(component, "test");
            assert!(new_state == ComponentState::Degraded || new_state == ComponentState::Dead);
        } else {
            panic!("Expected StateChange event");
        }
    }

    // ── Watchdog with metrics tests ──

    #[tokio::test]
    async fn test_check_cycle_updates_metrics() {
        let metrics = Arc::new(WatchdogMetrics::unregistered());
        let config = WatchdogConfig::default();
        let watchdog =
            CnsWatchdog::new(config, Arc::new(NoOpKillSwitch)).with_metrics(Arc::clone(&metrics));
        let handle = watchdog.handle();

        watchdog
            .register(ComponentRegistration {
                name: "comp".to_string(),
                criticality: ComponentCriticality::Critical,
                heartbeat_interval: Duration::from_secs(10),
                description: None,
            })
            .await;

        handle.heartbeat("comp").await;
        watchdog.check_cycle().await;

        assert_eq!(metrics.total_components_registered.get(), 1.0);
        assert_eq!(metrics.alive_components.get(), 1.0);
        assert_eq!(metrics.dead_components.get(), 0.0);

        let state_val = metrics
            .component_state
            .with_label_values(&["comp", "CRITICAL"])
            .get();
        assert_eq!(state_val, 1.0); // Alive
    }

    // ── Stop test ──

    #[tokio::test]
    async fn test_watchdog_stop() {
        let config = WatchdogConfig {
            check_interval_ms: 50,
            ..Default::default()
        };
        let watchdog = CnsWatchdog::new(config, Arc::new(NoOpKillSwitch));
        let mut rx = watchdog.subscribe();

        // Clone the shutdown sender before moving watchdog into the task
        let stop_handle = watchdog.shutdown_tx.clone();

        let task = tokio::spawn(async move {
            watchdog.run().await;
        });

        // Let it run for a bit
        tokio::time::sleep(Duration::from_millis(100)).await;

        // Stop it
        let _ = stop_handle.send(true);

        // The task should complete
        let result = tokio::time::timeout(Duration::from_secs(2), task).await;
        assert!(result.is_ok());

        // We should have received Started and Stopped events
        let mut got_started = false;
        let mut got_stopped = false;

        while let Ok(event) = rx.try_recv() {
            match event {
                WatchdogEvent::Started => got_started = true,
                WatchdogEvent::Stopped => got_stopped = true,
                _ => {}
            }
        }

        assert!(got_started);
        assert!(got_stopped);
    }

    // ── Kill switch integration test ──

    #[tokio::test]
    async fn test_watchdog_triggers_kill_on_critical_death() {
        use std::sync::atomic::{AtomicBool, Ordering};

        struct TestKillSwitch {
            killed: Arc<AtomicBool>,
        }

        #[async_trait::async_trait]
        impl KillSwitch for TestKillSwitch {
            async fn trigger_kill(&self, _reason: &str) -> Result<(), String> {
                self.killed.store(true, Ordering::SeqCst);
                Ok(())
            }
            async fn is_killed(&self) -> bool {
                self.killed.load(Ordering::SeqCst)
            }
        }

        let killed = Arc::new(AtomicBool::new(false));
        let ks = Arc::new(TestKillSwitch {
            killed: Arc::clone(&killed),
        });

        let config = WatchdogConfig {
            degraded_threshold: 1,
            dead_threshold: 1,
            kill_on_critical_death: true,
            ..Default::default()
        };

        let watchdog = CnsWatchdog::new(config, ks);
        let handle = watchdog.handle();

        watchdog
            .register(ComponentRegistration {
                name: "critical_service".to_string(),
                criticality: ComponentCriticality::Critical,
                heartbeat_interval: Duration::from_millis(5),
                description: None,
            })
            .await;

        // Send initial heartbeat then let it go stale
        handle.heartbeat("critical_service").await;
        tokio::time::sleep(Duration::from_millis(20)).await;

        // Run check cycle — should trigger kill
        watchdog.check_cycle().await;

        assert!(
            killed.load(Ordering::SeqCst),
            "Kill switch should have been triggered"
        );
    }

    #[tokio::test]
    async fn test_watchdog_does_not_kill_for_non_essential() {
        use std::sync::atomic::{AtomicBool, Ordering};

        struct TestKillSwitch {
            killed: Arc<AtomicBool>,
        }

        #[async_trait::async_trait]
        impl KillSwitch for TestKillSwitch {
            async fn trigger_kill(&self, _reason: &str) -> Result<(), String> {
                self.killed.store(true, Ordering::SeqCst);
                Ok(())
            }
            async fn is_killed(&self) -> bool {
                self.killed.load(Ordering::SeqCst)
            }
        }

        let killed = Arc::new(AtomicBool::new(false));
        let ks = Arc::new(TestKillSwitch {
            killed: Arc::clone(&killed),
        });

        let config = WatchdogConfig {
            degraded_threshold: 1,
            dead_threshold: 1,
            kill_on_critical_death: true,
            ..Default::default()
        };

        let watchdog = CnsWatchdog::new(config, ks);
        let handle = watchdog.handle();

        watchdog
            .register(ComponentRegistration {
                name: "optional_service".to_string(),
                criticality: ComponentCriticality::NonEssential,
                heartbeat_interval: Duration::from_millis(5),
                description: None,
            })
            .await;

        handle.heartbeat("optional_service").await;
        tokio::time::sleep(Duration::from_millis(20)).await;
        watchdog.check_cycle().await;

        assert!(
            !killed.load(Ordering::SeqCst),
            "Kill switch should NOT have been triggered for non-essential component"
        );
    }
}
