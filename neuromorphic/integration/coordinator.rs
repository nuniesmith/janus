//! Brain Coordinator — Central orchestration hub for neuromorphic regions
//!
//! The `BrainCoordinator` is the brainstem of the JANUS neuromorphic architecture.
//! It manages the lifecycle of all registered brain regions, coordinates per-tick
//! processing, routes signals between regions via the [`MessageBus`], tracks
//! region health, and provides a unified interface for the integration layer.
//!
//! # Architecture
//!
//! ```text
//! ┌─────────────────────────────────────────────────────────────────┐
//! │                      BrainCoordinator                           │
//! ├─────────────────────────────────────────────────────────────────┤
//! │                                                                 │
//! │  ┌────────────────────────────────────────────────────────────┐ │
//! │  │                  Region Registry                           │ │
//! │  │  ┌────────┐ ┌──────────┐ ┌─────────┐ ┌──────────────┐    │ │
//! │  │  │ Cortex │ │Hippocampus│ │Thalamus │ │Basal Ganglia │    │ │
//! │  │  └────┬───┘ └─────┬────┘ └────┬────┘ └──────┬───────┘    │ │
//! │  │       │           │           │             │             │ │
//! │  │  ┌────┴───┐ ┌─────┴────┐ ┌────┴────┐ ┌─────┴────────┐   │ │
//! │  │  │Amygdala│ │Hypothal. │ │Cerebell.│ │Visual Cortex │   │ │
//! │  │  └────┬───┘ └─────┬────┘ └────┬────┘ └──────┬───────┘   │ │
//! │  │       └───────────┼───────────┼─────────────┘            │ │
//! │  └───────────────────┼───────────┼──────────────────────────┘ │
//! │                      ▼           ▼                            │
//! │            ┌─────────────────────────────┐                    │
//! │            │       Message Bus           │                    │
//! │            │  (topic-based routing)      │                    │
//! │            └─────────────┬───────────────┘                    │
//! │                          │                                    │
//! │            ┌─────────────▼───────────────┐                    │
//! │            │       Global State          │                    │
//! │            │  (shared key-value store)   │                    │
//! │            └─────────────────────────────┘                    │
//! │                                                                │
//! └─────────────────────────────────────────────────────────────────┘
//! ```
//!
//! # Integration Points
//!
//! - **MessageBus**: All inter-region communication flows through the message bus.
//!   The coordinator creates standard channels during initialization.
//! - **GlobalState**: Shared state (market regime, risk level, etc.) is stored in
//!   the global state and accessible to all regions.
//! - **Service Bridges**: The coordinator interfaces with Forward/Backward/CNS
//!   services via the `BridgeManager` from `service_bridges`.
//!
//! # Lifecycle
//!
//! 1. **Construction**: `BrainCoordinator::new()` or `::with_config()`
//! 2. **Registration**: `register_region()` adds regions to the registry
//! 3. **Initialization**: `initialize()` sets up message bus channels and global state
//! 4. **Running**: `tick()` drives per-tick processing across all regions
//! 5. **Shutdown**: `shutdown()` gracefully stops all regions

use super::heterogeneity::{HeterogeneityConfig, HeterogeneityProfile};
use super::state::event_dispatcher::{EventDispatcher, EventDispatcherConfig};
use super::state::global_state::{GlobalState, GlobalStateConfig, RegionHealth};
use super::state::message_bus::{MessageBus, MessageBusConfig};
use super::state::state_sync::{StateSync, StateSyncConfig};
use std::collections::{HashMap, VecDeque};

// ---------------------------------------------------------------------------
// Configuration
// ---------------------------------------------------------------------------

/// Configuration for the BrainCoordinator.
#[derive(Debug, Clone)]
pub struct BrainCoordinatorConfig {
    /// Maximum number of brain regions that can be registered.
    pub max_regions: usize,

    /// Number of ticks between automatic health checks.
    pub health_check_interval: u64,

    /// Number of consecutive unhealthy ticks before a region is marked degraded.
    pub degraded_threshold: u64,

    /// Number of consecutive unhealthy ticks before a region is marked unhealthy.
    pub unhealthy_threshold: u64,

    /// EMA decay factor for coordinator-level smoothing.
    pub ema_decay: f64,

    /// Rolling window size for windowed statistics.
    pub window_size: usize,

    /// Whether to auto-initialize default message bus channels.
    pub auto_create_channels: bool,

    /// Message bus configuration.
    pub message_bus_config: MessageBusConfig,

    /// Global state configuration.
    pub global_state_config: GlobalStateConfig,

    /// Event dispatcher configuration.
    pub event_dispatcher_config: EventDispatcherConfig,

    /// State sync configuration.
    pub state_sync_config: StateSyncConfig,

    /// Heterogeneity configuration for crowding resistance.
    /// Controls the strength and bounds of per-instance parameter perturbation.
    pub heterogeneity_config: HeterogeneityConfig,

    /// Instance identifier used to generate the heterogeneity profile.
    /// Different instance IDs produce different parameter profiles, enabling
    /// idiosyncratic order flow across deployments.
    pub instance_id: String,
}

impl Default for BrainCoordinatorConfig {
    fn default() -> Self {
        Self {
            max_regions: 16,
            health_check_interval: 10,
            degraded_threshold: 3,
            unhealthy_threshold: 10,
            ema_decay: 0.1,
            window_size: 64,
            auto_create_channels: true,
            message_bus_config: MessageBusConfig::default(),
            global_state_config: GlobalStateConfig::default(),
            event_dispatcher_config: EventDispatcherConfig::default(),
            state_sync_config: StateSyncConfig::default(),
            heterogeneity_config: HeterogeneityConfig::default(),
            instance_id: "default".to_string(),
        }
    }
}

// ---------------------------------------------------------------------------
// Region descriptor
// ---------------------------------------------------------------------------

/// Describes a registered brain region.
#[derive(Debug, Clone)]
pub struct RegionDescriptor {
    /// Unique region name (e.g. "thalamus", "amygdala").
    pub name: String,

    /// Human-readable description.
    pub description: String,

    /// Topics this region publishes to.
    pub publishes: Vec<String>,

    /// Topic prefixes this region subscribes to.
    pub subscribes: Vec<String>,

    /// Whether the region is currently enabled.
    pub enabled: bool,

    /// Number of ticks this region has been unhealthy.
    pub consecutive_unhealthy: u64,

    /// Total ticks processed by this region.
    pub ticks_processed: u64,

    /// Total errors reported by this region.
    pub errors: u64,
}

impl RegionDescriptor {
    /// Create a new region descriptor.
    pub fn new(name: impl Into<String>, description: impl Into<String>) -> Self {
        Self {
            name: name.into(),
            description: description.into(),
            publishes: Vec::new(),
            subscribes: Vec::new(),
            enabled: true,
            consecutive_unhealthy: 0,
            ticks_processed: 0,
            errors: 0,
        }
    }

    /// Add a topic this region publishes to.
    pub fn with_publish(mut self, topic: impl Into<String>) -> Self {
        self.publishes.push(topic.into());
        self
    }

    /// Add a topic prefix this region subscribes to.
    pub fn with_subscribe(mut self, prefix: impl Into<String>) -> Self {
        self.subscribes.push(prefix.into());
        self
    }
}

// ---------------------------------------------------------------------------
// Coordinator lifecycle state
// ---------------------------------------------------------------------------

/// Current lifecycle phase of the coordinator.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum CoordinatorPhase {
    /// Constructed but not yet initialized.
    Created,
    /// Initialized and ready to run.
    Initialized,
    /// Actively processing ticks.
    Running,
    /// Temporarily paused (regions not ticked).
    Paused,
    /// Shutting down gracefully.
    ShuttingDown,
    /// Fully stopped.
    Stopped,
}

impl std::fmt::Display for CoordinatorPhase {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        match self {
            CoordinatorPhase::Created => write!(f, "Created"),
            CoordinatorPhase::Initialized => write!(f, "Initialized"),
            CoordinatorPhase::Running => write!(f, "Running"),
            CoordinatorPhase::Paused => write!(f, "Paused"),
            CoordinatorPhase::ShuttingDown => write!(f, "ShuttingDown"),
            CoordinatorPhase::Stopped => write!(f, "Stopped"),
        }
    }
}

// ---------------------------------------------------------------------------
// Windowed tick snapshot
// ---------------------------------------------------------------------------

#[derive(Debug, Clone)]
#[allow(dead_code)]
struct TickSnapshot {
    regions_ticked: usize,
    regions_healthy: usize,
    total_regions: usize,
    messages_sent: u64,
    events_dispatched: u64,
}

// ---------------------------------------------------------------------------
// Coordinator statistics
// ---------------------------------------------------------------------------

/// Operational statistics for the coordinator.
#[derive(Debug, Clone, Default)]
pub struct CoordinatorStats {
    /// Total ticks processed.
    pub total_ticks: u64,

    /// Total regions registered over lifetime.
    pub total_regions_registered: u64,

    /// Total regions removed over lifetime.
    pub total_regions_removed: u64,

    /// Total health checks performed.
    pub total_health_checks: u64,

    /// Total regions degraded (cumulative).
    pub total_degradations: u64,

    /// Total regions marked unhealthy (cumulative).
    pub total_unhealthy_marks: u64,

    /// Total signals routed through the bus.
    pub total_signals_routed: u64,

    /// EMA of healthy-region ratio.
    pub ema_health_ratio: f64,

    /// EMA of messages per tick.
    pub ema_messages_per_tick: f64,
}

// ---------------------------------------------------------------------------
// BrainCoordinator
// ---------------------------------------------------------------------------

/// Central coordination hub for all neuromorphic brain regions.
///
/// See [module documentation](self) for architecture and lifecycle details.
pub struct BrainCoordinator {
    config: BrainCoordinatorConfig,

    /// Current lifecycle phase.
    phase: CoordinatorPhase,

    /// Registered brain regions keyed by name.
    regions: HashMap<String, RegionDescriptor>,

    /// Insertion-ordered region names (determines tick order).
    region_order: Vec<String>,

    /// Inter-region message bus.
    message_bus: MessageBus,

    /// Shared global state.
    global_state: GlobalState,

    /// Priority-based event dispatcher.
    event_dispatcher: EventDispatcher,

    /// Cross-region state synchronization.
    state_sync: StateSync,

    /// Heterogeneity profile for this instance, providing per-region
    /// parameter perturbations for crowding resistance.
    heterogeneity_profile: HeterogeneityProfile,

    /// Current tick counter.
    tick: u64,

    /// Whether EMA values have been initialized.
    ema_initialized: bool,

    /// Rolling window of recent tick snapshots.
    recent: VecDeque<TickSnapshot>,

    /// Accumulated statistics.
    stats: CoordinatorStats,
}

impl Default for BrainCoordinator {
    fn default() -> Self {
        Self::new()
    }
}

impl BrainCoordinator {
    // -------------------------------------------------------------------
    // Construction
    // -------------------------------------------------------------------

    /// Create a new coordinator with default configuration.
    pub fn new() -> Self {
        Self::with_config(BrainCoordinatorConfig::default())
    }

    /// Create a new coordinator with custom configuration.
    pub fn with_config(config: BrainCoordinatorConfig) -> Self {
        let message_bus = MessageBus::with_config(config.message_bus_config.clone());
        let global_state = GlobalState::with_config(config.global_state_config.clone());
        let event_dispatcher = EventDispatcher::with_config(config.event_dispatcher_config.clone());
        let state_sync = StateSync::with_config(config.state_sync_config.clone());
        let window_size = config.window_size;

        // Generate heterogeneity profile from instance ID
        let heterogeneity_profile = HeterogeneityProfile::from_instance_id(
            &config.instance_id,
            &config.heterogeneity_config,
        );

        Self {
            config,
            phase: CoordinatorPhase::Created,
            regions: HashMap::new(),
            region_order: Vec::new(),
            message_bus,
            global_state,
            event_dispatcher,
            state_sync,
            heterogeneity_profile,
            tick: 0,
            ema_initialized: false,
            recent: VecDeque::with_capacity(window_size + 1),
            stats: CoordinatorStats::default(),
        }
    }

    // -------------------------------------------------------------------
    // Region management
    // -------------------------------------------------------------------

    /// Register a brain region with the coordinator.
    ///
    /// Returns `true` if the region was newly registered, `false` if it
    /// was already registered or the registry is at capacity.
    pub fn register_region(&mut self, descriptor: RegionDescriptor) -> bool {
        if self.regions.len() >= self.config.max_regions {
            return false;
        }
        if self.regions.contains_key(&descriptor.name) {
            return false;
        }

        let name = descriptor.name.clone();

        // Register with global state for health tracking
        self.global_state.register_region(&name);

        // Register with state sync
        self.state_sync.register_region(&name);

        // Create message bus channels for the region's subscriptions
        for prefix in &descriptor.subscribes {
            let channel_name = format!("{}_{}", name, prefix.trim_end_matches('.'));
            self.message_bus.create_channel(&channel_name, prefix);
        }

        self.region_order.push(name.clone());
        self.regions.insert(name, descriptor);
        self.stats.total_regions_registered += 1;

        true
    }

    /// Remove a registered brain region.
    ///
    /// Returns `true` if the region existed and was removed.
    pub fn remove_region(&mut self, name: &str) -> bool {
        if self.regions.remove(name).is_none() {
            return false;
        }

        self.region_order.retain(|n| n != name);
        self.stats.total_regions_removed += 1;
        true
    }

    /// Get a reference to a registered region.
    pub fn region(&self, name: &str) -> Option<&RegionDescriptor> {
        self.regions.get(name)
    }

    /// Get a mutable reference to a registered region.
    pub fn region_mut(&mut self, name: &str) -> Option<&mut RegionDescriptor> {
        self.regions.get_mut(name)
    }

    /// Number of currently registered regions.
    pub fn region_count(&self) -> usize {
        self.regions.len()
    }

    /// Ordered list of region names (determines tick processing order).
    pub fn region_names(&self) -> &[String] {
        &self.region_order
    }

    /// Check if a region is registered.
    pub fn has_region(&self, name: &str) -> bool {
        self.regions.contains_key(name)
    }

    /// Enable a previously disabled region.
    pub fn enable_region(&mut self, name: &str) -> bool {
        if let Some(region) = self.regions.get_mut(name) {
            region.enabled = true;
            true
        } else {
            false
        }
    }

    /// Disable a region (it will be skipped during tick processing).
    pub fn disable_region(&mut self, name: &str) -> bool {
        if let Some(region) = self.regions.get_mut(name) {
            region.enabled = false;
            true
        } else {
            false
        }
    }

    // -------------------------------------------------------------------
    // Lifecycle management
    // -------------------------------------------------------------------

    /// Initialize the coordinator, setting up default channels and state.
    ///
    /// Transitions phase from `Created` to `Initialized`.
    pub fn initialize(&mut self) -> bool {
        if self.phase != CoordinatorPhase::Created {
            return false;
        }

        if self.config.auto_create_channels {
            self.create_default_channels();
        }

        // Seed global state with standard keys
        // Phase encoding: 0.0=created, 1.0=initialized, 2.0=running, 3.0=paused, 4.0=shutting_down, 5.0=stopped
        self.global_state.set("coordinator.phase", 1.0);
        self.global_state.set("coordinator.tick", 0.0);

        // Publish heterogeneity profile parameters to global state so that
        // individual brain regions can read their perturbations during init.
        let hp = &self.heterogeneity_profile;
        self.global_state.set("heterogeneity.seed", hp.seed as f64);
        self.global_state.set(
            "heterogeneity.thalamus.orderbook_weight",
            hp.thalamus.orderbook_weight,
        );
        self.global_state.set(
            "heterogeneity.thalamus.price_weight",
            hp.thalamus.price_weight,
        );
        self.global_state.set(
            "heterogeneity.thalamus.volume_weight",
            hp.thalamus.volume_weight,
        );
        self.global_state.set(
            "heterogeneity.thalamus.sentiment_weight",
            hp.thalamus.sentiment_weight,
        );
        self.global_state.set(
            "heterogeneity.thalamus.attention_temperature",
            hp.thalamus.attention_temperature,
        );
        self.global_state.set(
            "heterogeneity.hypothalamus.risk_appetite",
            hp.hypothalamus.risk_appetite_setpoint,
        );
        self.global_state.set(
            "heterogeneity.hypothalamus.position_sizing",
            hp.hypothalamus.position_sizing_multiplier,
        );
        self.global_state.set(
            "heterogeneity.hypothalamus.drawdown_threshold",
            hp.hypothalamus.drawdown_threshold,
        );
        self.global_state.set(
            "heterogeneity.basal_ganglia.dopamine_sensitivity",
            hp.basal_ganglia.dopamine_sensitivity,
        );
        self.global_state.set(
            "heterogeneity.basal_ganglia.go_threshold",
            hp.basal_ganglia.go_threshold,
        );
        self.global_state.set(
            "heterogeneity.basal_ganglia.discount_factor",
            hp.basal_ganglia.discount_factor,
        );
        self.global_state.set(
            "heterogeneity.basal_ganglia.selection_temperature",
            hp.basal_ganglia.selection_temperature,
        );
        self.global_state.set(
            "heterogeneity.prefrontal.risk_limit_weight",
            hp.prefrontal.risk_limit_weight,
        );
        self.global_state.set(
            "heterogeneity.prefrontal.violation_threshold",
            hp.prefrontal.violation_threshold,
        );
        self.global_state.set(
            "heterogeneity.hippocampus.replay_alpha",
            hp.hippocampus.replay_alpha,
        );
        self.global_state.set(
            "heterogeneity.hippocampus.consolidation_threshold",
            hp.hippocampus.consolidation_threshold,
        );
        self.global_state.set(
            "heterogeneity.cerebellum.ac_lambda",
            hp.cerebellum.ac_lambda,
        );
        self.global_state
            .set("heterogeneity.cerebellum.ac_eta", hp.cerebellum.ac_eta);

        self.phase = CoordinatorPhase::Initialized;
        true
    }

    /// Transition to the Running phase.
    pub fn start(&mut self) -> bool {
        if self.phase != CoordinatorPhase::Initialized && self.phase != CoordinatorPhase::Paused {
            return false;
        }
        self.phase = CoordinatorPhase::Running;
        self.global_state.set("coordinator.phase", 2.0);
        true
    }

    /// Pause the coordinator (regions stop being ticked).
    pub fn pause(&mut self) -> bool {
        if self.phase != CoordinatorPhase::Running {
            return false;
        }
        self.phase = CoordinatorPhase::Paused;
        self.global_state.set("coordinator.phase", 3.0);
        true
    }

    /// Resume from paused state.
    pub fn resume(&mut self) -> bool {
        if self.phase != CoordinatorPhase::Paused {
            return false;
        }
        self.phase = CoordinatorPhase::Running;
        self.global_state.set("coordinator.phase", 2.0);
        true
    }

    /// Initiate graceful shutdown.
    pub fn shutdown(&mut self) -> bool {
        if self.phase == CoordinatorPhase::Stopped || self.phase == CoordinatorPhase::ShuttingDown {
            return false;
        }
        self.phase = CoordinatorPhase::ShuttingDown;
        self.global_state.set("coordinator.phase", 4.0);

        // Notify all regions via the message bus
        self.message_bus.emit("system.shutdown", "coordinator", 1.0);

        self.phase = CoordinatorPhase::Stopped;
        self.global_state.set("coordinator.phase", 5.0);
        true
    }

    /// Get the current lifecycle phase.
    pub fn phase(&self) -> CoordinatorPhase {
        self.phase
    }

    /// Whether the coordinator is actively running (accepting ticks).
    pub fn running(&self) -> bool {
        self.phase == CoordinatorPhase::Running
    }

    // -------------------------------------------------------------------
    // Tick processing
    // -------------------------------------------------------------------

    /// Process a single coordination tick.
    ///
    /// This drives per-tick processing across all enabled regions, updates
    /// health tracking, routes messages, and updates statistics.
    ///
    /// Returns the number of regions that were successfully ticked.
    pub fn tick(&mut self) -> usize {
        if self.phase != CoordinatorPhase::Running {
            return 0;
        }

        self.tick += 1;

        // Update global state with current tick
        self.global_state.set("coordinator.tick", self.tick as f64);

        // Tick the message bus and global state
        self.message_bus.tick();
        self.global_state.tick();
        self.event_dispatcher.tick();

        // Count enabled regions and perform health checks
        let mut regions_ticked = 0usize;
        let mut regions_healthy = 0usize;

        let region_names: Vec<String> = self.region_order.clone();
        for name in &region_names {
            if let Some(region) = self.regions.get_mut(name) {
                if !region.enabled {
                    continue;
                }

                region.ticks_processed += 1;
                regions_ticked += 1;

                // Heartbeat to global state
                self.global_state.heartbeat(name);
            }
        }

        // Periodic health check
        if self.tick % self.config.health_check_interval == 0 {
            self.perform_health_check();
        }

        // Count healthy regions
        for name in &region_names {
            if let Some(region) = self.regions.get(name) {
                if region.enabled && region.consecutive_unhealthy == 0 {
                    regions_healthy += 1;
                }
            }
        }

        let total_regions = self.regions.len();
        let bus_stats = self.message_bus.stats();
        let dispatcher_stats = self.event_dispatcher.stats();

        // Record tick snapshot
        let snapshot = TickSnapshot {
            regions_ticked,
            regions_healthy,
            total_regions,
            messages_sent: bus_stats.total_sent,
            events_dispatched: dispatcher_stats.total_dispatched,
        };
        self.recent.push_back(snapshot);
        if self.recent.len() > self.config.window_size {
            self.recent.pop_front();
        }

        // Update EMAs
        let health_ratio = if total_regions > 0 {
            regions_healthy as f64 / total_regions as f64
        } else {
            1.0
        };

        if !self.ema_initialized {
            self.stats.ema_health_ratio = health_ratio;
            self.stats.ema_messages_per_tick = bus_stats.total_sent as f64;
            self.ema_initialized = true;
        } else {
            let alpha = self.config.ema_decay;
            self.stats.ema_health_ratio =
                alpha * health_ratio + (1.0 - alpha) * self.stats.ema_health_ratio;
            self.stats.ema_messages_per_tick = alpha * (bus_stats.total_sent as f64)
                + (1.0 - alpha) * self.stats.ema_messages_per_tick;
        }

        self.stats.total_ticks += 1;
        self.stats.total_signals_routed = bus_stats.total_sent;

        regions_ticked
    }

    /// Process a tick (alias for compatibility with `process` pattern).
    pub fn process(&mut self) -> usize {
        self.tick()
    }

    /// Get the current tick count.
    pub fn current_tick(&self) -> u64 {
        self.tick
    }

    // -------------------------------------------------------------------
    // Health monitoring
    // -------------------------------------------------------------------

    /// Perform a health check across all registered regions.
    fn perform_health_check(&mut self) {
        self.stats.total_health_checks += 1;

        let region_names: Vec<String> = self.region_order.clone();
        for name in &region_names {
            let health = self
                .global_state
                .region_health(&name)
                .map(|rec| rec.health.clone());

            if let Some(region) = self.regions.get_mut(name) {
                match health {
                    Some(RegionHealth::Healthy) => {
                        region.consecutive_unhealthy = 0;
                    }
                    Some(RegionHealth::Degraded) | Some(RegionHealth::Stale) => {
                        region.consecutive_unhealthy += 1;
                        if region.consecutive_unhealthy >= self.config.unhealthy_threshold {
                            self.global_state
                                .mark_unhealthy(&name, "consecutive unhealthy threshold exceeded");
                            self.stats.total_unhealthy_marks += 1;
                        } else if region.consecutive_unhealthy >= self.config.degraded_threshold {
                            self.global_state
                                .mark_degraded(&name, "consecutive degraded threshold exceeded");
                            self.stats.total_degradations += 1;
                        }
                    }
                    Some(RegionHealth::Unhealthy) => {
                        region.consecutive_unhealthy += 1;
                        self.stats.total_unhealthy_marks += 1;
                    }
                    None => {
                        // Region not tracked in global state — skip
                    }
                }
            }
        }
    }

    /// Report an error from a specific region.
    pub fn report_region_error(&mut self, region_name: &str) {
        if let Some(region) = self.regions.get_mut(region_name) {
            region.errors += 1;
            region.consecutive_unhealthy += 1;

            if region.consecutive_unhealthy >= self.config.unhealthy_threshold {
                self.global_state
                    .mark_unhealthy(region_name, "region reported error");
            } else if region.consecutive_unhealthy >= self.config.degraded_threshold {
                self.global_state
                    .mark_degraded(region_name, "region reported error");
            }
        }
    }

    /// Report a successful heartbeat from a region (resets unhealthy counter).
    pub fn report_region_healthy(&mut self, region_name: &str) {
        if let Some(region) = self.regions.get_mut(region_name) {
            region.consecutive_unhealthy = 0;
        }
        self.global_state.heartbeat(region_name);
    }

    /// Get the health ratio (healthy / total regions).
    pub fn health_ratio(&self) -> f64 {
        self.global_state.health_ratio()
    }

    /// Get the smoothed (EMA) health ratio.
    pub fn smoothed_health_ratio(&self) -> f64 {
        self.stats.ema_health_ratio
    }

    /// Get the number of healthy regions.
    pub fn healthy_region_count(&self) -> usize {
        self.regions
            .values()
            .filter(|r| r.enabled && r.consecutive_unhealthy == 0)
            .count()
    }

    /// Get names of unhealthy regions.
    pub fn unhealthy_regions(&self) -> Vec<String> {
        self.regions
            .values()
            .filter(|r| r.enabled && r.consecutive_unhealthy > 0)
            .map(|r| r.name.clone())
            .collect()
    }

    // -------------------------------------------------------------------
    // Signal routing (convenience wrappers around message bus)
    // -------------------------------------------------------------------

    /// Send a signal through the message bus.
    ///
    /// Returns the message ID.
    pub fn send_signal(
        &mut self,
        topic: impl Into<String>,
        sender: impl Into<String>,
        payload: f64,
    ) -> u64 {
        self.message_bus.emit(topic, sender, payload)
    }

    /// Send a signal with an attached body.
    pub fn send_signal_with_body(
        &mut self,
        topic: impl Into<String>,
        sender: impl Into<String>,
        payload: f64,
        body: String,
    ) -> u64 {
        self.message_bus.send(topic, sender, payload, Some(body))
    }

    /// Drain all messages from a named channel.
    pub fn drain_channel(&mut self, channel: &str) -> Vec<super::state::message_bus::Message> {
        self.message_bus.drain(channel)
    }

    /// Drain all messages from all channels.
    pub fn drain_all(&mut self) -> HashMap<String, Vec<super::state::message_bus::Message>> {
        self.message_bus.drain_all()
    }

    // -------------------------------------------------------------------
    // Global state access
    // -------------------------------------------------------------------

    /// Set a value in the global shared state.
    pub fn set_state(&mut self, key: impl Into<String>, value: f64) {
        self.global_state.set(key, value);
    }

    /// Get a value from the global shared state.
    pub fn get_state(&mut self, key: &str) -> Option<f64> {
        self.global_state.get(key)
    }

    /// Check if a key exists in the global state.
    pub fn has_state(&self, key: &str) -> bool {
        self.global_state.contains_key(key)
    }

    // -------------------------------------------------------------------
    // Component accessors
    // -------------------------------------------------------------------

    /// Get a reference to the message bus.
    pub fn message_bus(&self) -> &MessageBus {
        &self.message_bus
    }

    /// Get a mutable reference to the message bus.
    pub fn message_bus_mut(&mut self) -> &mut MessageBus {
        &mut self.message_bus
    }

    /// Get a reference to the global state.
    pub fn global_state(&self) -> &GlobalState {
        &self.global_state
    }

    /// Get a mutable reference to the global state.
    pub fn global_state_mut(&mut self) -> &mut GlobalState {
        &mut self.global_state
    }

    /// Get a reference to the event dispatcher.
    pub fn event_dispatcher(&self) -> &EventDispatcher {
        &self.event_dispatcher
    }

    /// Get a mutable reference to the event dispatcher.
    pub fn event_dispatcher_mut(&mut self) -> &mut EventDispatcher {
        &mut self.event_dispatcher
    }

    /// Get a reference to the state sync.
    pub fn state_sync(&self) -> &StateSync {
        &self.state_sync
    }

    /// Get a mutable reference to the state sync.
    pub fn state_sync_mut(&mut self) -> &mut StateSync {
        &mut self.state_sync
    }

    /// Get the coordinator configuration.
    pub fn config(&self) -> &BrainCoordinatorConfig {
        &self.config
    }

    /// Get operational statistics.
    pub fn stats(&self) -> &CoordinatorStats {
        &self.stats
    }

    // -------------------------------------------------------------------
    // Heterogeneity access
    // -------------------------------------------------------------------

    /// Get the heterogeneity profile for this coordinator instance.
    ///
    /// The profile contains per-region parameter perturbations that should
    /// be applied when initializing each brain region, producing idiosyncratic
    /// behavior across JANUS deployments.
    pub fn heterogeneity_profile(&self) -> &HeterogeneityProfile {
        &self.heterogeneity_profile
    }

    /// Get the instance ID used for heterogeneity generation.
    pub fn instance_id(&self) -> &str {
        &self.config.instance_id
    }

    // -------------------------------------------------------------------
    // Windowed analytics
    // -------------------------------------------------------------------

    /// Average healthy-region ratio over the recent window.
    pub fn windowed_health_ratio(&self) -> f64 {
        if self.recent.is_empty() {
            return 1.0;
        }
        let sum: f64 = self
            .recent
            .iter()
            .map(|s| {
                if s.total_regions > 0 {
                    s.regions_healthy as f64 / s.total_regions as f64
                } else {
                    1.0
                }
            })
            .sum();
        sum / self.recent.len() as f64
    }

    /// Average regions ticked per tick over the recent window.
    pub fn windowed_regions_ticked(&self) -> f64 {
        if self.recent.is_empty() {
            return 0.0;
        }
        let sum: f64 = self.recent.iter().map(|s| s.regions_ticked as f64).sum();
        sum / self.recent.len() as f64
    }

    /// Check if health is declining (linear regression over window).
    pub fn is_health_declining(&self) -> bool {
        if self.recent.len() < 4 {
            return false;
        }

        let n = self.recent.len() as f64;
        let mut sum_x = 0.0f64;
        let mut sum_y = 0.0f64;
        let mut sum_xy = 0.0f64;
        let mut sum_xx = 0.0f64;

        for (i, s) in self.recent.iter().enumerate() {
            let x = i as f64;
            let y = if s.total_regions > 0 {
                s.regions_healthy as f64 / s.total_regions as f64
            } else {
                1.0
            };
            sum_x += x;
            sum_y += y;
            sum_xy += x * y;
            sum_xx += x * x;
        }

        let denom = n * sum_xx - sum_x * sum_x;
        if denom.abs() < 1e-12 {
            return false;
        }
        let slope = (n * sum_xy - sum_x * sum_y) / denom;
        slope < -0.01
    }

    // -------------------------------------------------------------------
    // Reset
    // -------------------------------------------------------------------

    /// Reset the coordinator to its initial state.
    ///
    /// Clears all regions, resets state and statistics, and transitions
    /// back to the `Created` phase.
    pub fn reset(&mut self) {
        self.regions.clear();
        self.region_order.clear();
        self.message_bus.reset();
        self.global_state.reset();
        self.event_dispatcher.reset();
        self.state_sync.reset();
        // Re-derive heterogeneity profile (deterministic — same instance_id, same profile)
        self.heterogeneity_profile = HeterogeneityProfile::from_instance_id(
            &self.config.instance_id,
            &self.config.heterogeneity_config,
        );
        self.tick = 0;
        self.ema_initialized = false;
        self.recent.clear();
        self.stats = CoordinatorStats::default();
        self.phase = CoordinatorPhase::Created;
    }

    // -------------------------------------------------------------------
    // Private helpers
    // -------------------------------------------------------------------

    /// Create the standard set of message bus channels.
    fn create_default_channels(&mut self) {
        // System-wide events
        self.message_bus.create_channel("system", "system.");

        // Risk alerts from amygdala / hypothalamus
        self.message_bus.create_channel("risk", "risk.");

        // Market data events from thalamus
        self.message_bus.create_channel("market", "market.");

        // Trading signals from cortex / basal ganglia
        self.message_bus.create_channel("signal", "signal.");

        // Memory events from hippocampus
        self.message_bus.create_channel("memory", "memory.");

        // Training events
        self.message_bus.create_channel("training", "training.");

        // Catch-all channel for monitoring / debugging
        self.message_bus.create_channel("monitor", "");
    }
}

// ---------------------------------------------------------------------------
// Tests
// ---------------------------------------------------------------------------

#[cfg(test)]
mod tests {
    use super::*;

    fn small_config() -> BrainCoordinatorConfig {
        BrainCoordinatorConfig {
            max_regions: 4,
            health_check_interval: 2,
            degraded_threshold: 2,
            unhealthy_threshold: 5,
            window_size: 8,
            ..Default::default()
        }
    }

    fn thalamus_descriptor() -> RegionDescriptor {
        RegionDescriptor::new("thalamus", "Sensory relay and attention gating")
            .with_publish("market.tick")
            .with_publish("market.regime")
            .with_subscribe("risk.")
    }

    fn amygdala_descriptor() -> RegionDescriptor {
        RegionDescriptor::new("amygdala", "Threat detection and circuit breakers")
            .with_publish("risk.alert")
            .with_subscribe("market.")
    }

    // ---------------------------------------------------------------
    // Construction
    // ---------------------------------------------------------------

    #[test]
    fn test_new_default() {
        let coord = BrainCoordinator::new();
        assert_eq!(coord.phase(), CoordinatorPhase::Created);
        assert_eq!(coord.region_count(), 0);
        assert_eq!(coord.current_tick(), 0);
    }

    #[test]
    fn test_with_config() {
        let coord = BrainCoordinator::with_config(small_config());
        assert_eq!(coord.config().max_regions, 4);
        assert_eq!(coord.config().health_check_interval, 2);
    }

    // ---------------------------------------------------------------
    // Region management
    // ---------------------------------------------------------------

    #[test]
    fn test_register_region() {
        let mut coord = BrainCoordinator::new();
        assert!(coord.register_region(thalamus_descriptor()));
        assert_eq!(coord.region_count(), 1);
        assert!(coord.has_region("thalamus"));
        assert_eq!(coord.stats().total_regions_registered, 1);
    }

    #[test]
    fn test_register_duplicate() {
        let mut coord = BrainCoordinator::new();
        assert!(coord.register_region(thalamus_descriptor()));
        assert!(!coord.register_region(thalamus_descriptor()));
        assert_eq!(coord.region_count(), 1);
    }

    #[test]
    fn test_register_at_capacity() {
        let mut coord = BrainCoordinator::with_config(small_config());
        for i in 0..4 {
            assert!(coord.register_region(RegionDescriptor::new(format!("region_{}", i), "test")));
        }
        assert!(!coord.register_region(RegionDescriptor::new("overflow", "test")));
        assert_eq!(coord.region_count(), 4);
    }

    #[test]
    fn test_remove_region() {
        let mut coord = BrainCoordinator::new();
        coord.register_region(thalamus_descriptor());
        assert!(coord.remove_region("thalamus"));
        assert!(!coord.has_region("thalamus"));
        assert_eq!(coord.region_count(), 0);
        assert_eq!(coord.stats().total_regions_removed, 1);
    }

    #[test]
    fn test_remove_nonexistent() {
        let mut coord = BrainCoordinator::new();
        assert!(!coord.remove_region("ghost"));
    }

    #[test]
    fn test_region_names_order() {
        let mut coord = BrainCoordinator::new();
        coord.register_region(thalamus_descriptor());
        coord.register_region(amygdala_descriptor());
        assert_eq!(coord.region_names(), &["thalamus", "amygdala"]);
    }

    #[test]
    fn test_enable_disable_region() {
        let mut coord = BrainCoordinator::new();
        coord.register_region(thalamus_descriptor());
        assert!(coord.region("thalamus").unwrap().enabled);

        assert!(coord.disable_region("thalamus"));
        assert!(!coord.region("thalamus").unwrap().enabled);

        assert!(coord.enable_region("thalamus"));
        assert!(coord.region("thalamus").unwrap().enabled);
    }

    #[test]
    fn test_enable_nonexistent() {
        let mut coord = BrainCoordinator::new();
        assert!(!coord.enable_region("ghost"));
        assert!(!coord.disable_region("ghost"));
    }

    // ---------------------------------------------------------------
    // Lifecycle
    // ---------------------------------------------------------------

    #[test]
    fn test_lifecycle_happy_path() {
        let mut coord = BrainCoordinator::new();
        assert_eq!(coord.phase(), CoordinatorPhase::Created);
        assert!(!coord.running());

        assert!(coord.initialize());
        assert_eq!(coord.phase(), CoordinatorPhase::Initialized);

        assert!(coord.start());
        assert_eq!(coord.phase(), CoordinatorPhase::Running);
        assert!(coord.running());

        assert!(coord.pause());
        assert_eq!(coord.phase(), CoordinatorPhase::Paused);
        assert!(!coord.running());

        assert!(coord.resume());
        assert_eq!(coord.phase(), CoordinatorPhase::Running);

        assert!(coord.shutdown());
        assert_eq!(coord.phase(), CoordinatorPhase::Stopped);
    }

    #[test]
    fn test_cannot_initialize_twice() {
        let mut coord = BrainCoordinator::new();
        assert!(coord.initialize());
        assert!(!coord.initialize());
    }

    #[test]
    fn test_cannot_start_from_created() {
        let mut coord = BrainCoordinator::new();
        assert!(!coord.start());
    }

    #[test]
    fn test_cannot_pause_if_not_running() {
        let mut coord = BrainCoordinator::new();
        assert!(!coord.pause());
    }

    #[test]
    fn test_cannot_shutdown_twice() {
        let mut coord = BrainCoordinator::new();
        coord.initialize();
        coord.start();
        assert!(coord.shutdown());
        assert!(!coord.shutdown());
    }

    #[test]
    fn test_start_from_paused() {
        let mut coord = BrainCoordinator::new();
        coord.initialize();
        coord.start();
        coord.pause();
        assert!(coord.start());
        assert!(coord.running());
    }

    // ---------------------------------------------------------------
    // Tick processing
    // ---------------------------------------------------------------

    #[test]
    fn test_tick_requires_running() {
        let mut coord = BrainCoordinator::new();
        assert_eq!(coord.tick(), 0);

        coord.initialize();
        assert_eq!(coord.tick(), 0);
    }

    #[test]
    fn test_tick_increments() {
        let mut coord = BrainCoordinator::new();
        coord.initialize();
        coord.start();

        let ticked = coord.tick();
        assert_eq!(ticked, 0); // no regions registered
        assert_eq!(coord.current_tick(), 1);
        assert_eq!(coord.stats().total_ticks, 1);
    }

    #[test]
    fn test_tick_with_regions() {
        let mut coord = BrainCoordinator::new();
        coord.register_region(thalamus_descriptor());
        coord.register_region(amygdala_descriptor());
        coord.initialize();
        coord.start();

        let ticked = coord.tick();
        assert_eq!(ticked, 2);
        assert_eq!(coord.region("thalamus").unwrap().ticks_processed, 1);
        assert_eq!(coord.region("amygdala").unwrap().ticks_processed, 1);
    }

    #[test]
    fn test_tick_skips_disabled() {
        let mut coord = BrainCoordinator::new();
        coord.register_region(thalamus_descriptor());
        coord.register_region(amygdala_descriptor());
        coord.initialize();
        coord.start();

        coord.disable_region("amygdala");
        let ticked = coord.tick();
        assert_eq!(ticked, 1);
        assert_eq!(coord.region("thalamus").unwrap().ticks_processed, 1);
        assert_eq!(coord.region("amygdala").unwrap().ticks_processed, 0);
    }

    #[test]
    fn test_process_alias() {
        let mut coord = BrainCoordinator::new();
        coord.register_region(thalamus_descriptor());
        coord.initialize();
        coord.start();
        assert_eq!(coord.process(), 1);
    }

    // ---------------------------------------------------------------
    // Health monitoring
    // ---------------------------------------------------------------

    #[test]
    fn test_health_ratio_no_regions() {
        let coord = BrainCoordinator::new();
        // GlobalState returns 1.0 when no regions
        assert!((coord.health_ratio() - 1.0).abs() < 1e-10);
    }

    #[test]
    fn test_health_ratio_all_healthy() {
        let mut coord = BrainCoordinator::new();
        coord.register_region(thalamus_descriptor());
        coord.register_region(amygdala_descriptor());
        assert_eq!(coord.healthy_region_count(), 2);
    }

    #[test]
    fn test_report_region_error() {
        let mut coord = BrainCoordinator::with_config(small_config());
        coord.register_region(thalamus_descriptor());
        coord.report_region_error("thalamus");
        assert_eq!(coord.region("thalamus").unwrap().errors, 1);
        assert_eq!(coord.region("thalamus").unwrap().consecutive_unhealthy, 1);
    }

    #[test]
    fn test_report_region_healthy_resets() {
        let mut coord = BrainCoordinator::new();
        coord.register_region(thalamus_descriptor());
        coord.report_region_error("thalamus");
        coord.report_region_error("thalamus");
        assert_eq!(coord.region("thalamus").unwrap().consecutive_unhealthy, 2);

        coord.report_region_healthy("thalamus");
        assert_eq!(coord.region("thalamus").unwrap().consecutive_unhealthy, 0);
    }

    #[test]
    fn test_unhealthy_regions_list() {
        let mut coord = BrainCoordinator::new();
        coord.register_region(thalamus_descriptor());
        coord.register_region(amygdala_descriptor());

        coord.report_region_error("amygdala");
        let unhealthy = coord.unhealthy_regions();
        assert_eq!(unhealthy, vec!["amygdala"]);
    }

    // ---------------------------------------------------------------
    // Signal routing
    // ---------------------------------------------------------------

    #[test]
    fn test_send_and_drain_signal() {
        let mut coord = BrainCoordinator::new();
        coord.initialize();

        // "risk." channel is created by default
        let id = coord.send_signal("risk.alert", "amygdala", 0.95);
        assert!(id > 0);

        let msgs = coord.drain_channel("risk");
        assert_eq!(msgs.len(), 1);
        assert_eq!(msgs[0].topic, "risk.alert");
        assert!((msgs[0].payload - 0.95).abs() < 1e-10);
    }

    #[test]
    fn test_send_signal_with_body() {
        let mut coord = BrainCoordinator::new();
        coord.initialize();

        coord.send_signal_with_body(
            "market.regime_change",
            "thalamus",
            1.0,
            "bull->bear".to_string(),
        );

        let msgs = coord.drain_channel("market");
        assert_eq!(msgs.len(), 1);
        assert_eq!(msgs[0].body.as_deref(), Some("bull->bear"));
    }

    #[test]
    fn test_drain_all() {
        let mut coord = BrainCoordinator::new();
        coord.initialize();

        coord.send_signal("risk.alert", "amygdala", 0.9);
        coord.send_signal("market.tick", "thalamus", 100.0);

        let all = coord.drain_all();
        // "monitor" channel catches everything with "" prefix
        assert!(all.contains_key("monitor"));
    }

    // ---------------------------------------------------------------
    // Global state
    // ---------------------------------------------------------------

    #[test]
    fn test_global_state_access() {
        let mut coord = BrainCoordinator::new();
        coord.set_state("regime", 1.0);
        assert_eq!(coord.get_state("regime"), Some(1.0));
        assert!(coord.has_state("regime"));
    }

    // ---------------------------------------------------------------
    // Component accessors
    // ---------------------------------------------------------------

    #[test]
    fn test_component_accessors() {
        let mut coord = BrainCoordinator::new();
        coord.initialize();

        // Verify components are accessible
        assert_eq!(coord.message_bus().channel_count(), 7); // 7 default channels
        let _ = coord.global_state();
        let _ = coord.event_dispatcher();
        let _ = coord.state_sync();
    }

    #[test]
    fn test_mutable_component_access() {
        let mut coord = BrainCoordinator::new();
        coord.message_bus_mut().create_channel("custom", "custom.");
        assert_eq!(coord.message_bus().channel_count(), 1);
    }

    // ---------------------------------------------------------------
    // Windowed analytics
    // ---------------------------------------------------------------

    #[test]
    fn test_windowed_health_ratio_empty() {
        let coord = BrainCoordinator::new();
        assert!((coord.windowed_health_ratio() - 1.0).abs() < 1e-10);
    }

    #[test]
    fn test_windowed_regions_ticked_empty() {
        let coord = BrainCoordinator::new();
        assert!((coord.windowed_regions_ticked() - 0.0).abs() < 1e-10);
    }

    #[test]
    fn test_is_health_declining_insufficient_data() {
        let coord = BrainCoordinator::new();
        assert!(!coord.is_health_declining());
    }

    #[test]
    fn test_windowed_analytics_with_ticks() {
        let mut coord = BrainCoordinator::with_config(small_config());
        coord.register_region(thalamus_descriptor());
        coord.initialize();
        coord.start();

        for _ in 0..5 {
            coord.tick();
        }

        assert!(coord.windowed_regions_ticked() > 0.0);
        assert!(!coord.is_health_declining());
    }

    // ---------------------------------------------------------------
    // Reset
    // ---------------------------------------------------------------

    #[test]
    fn test_reset() {
        let mut coord = BrainCoordinator::new();
        coord.register_region(thalamus_descriptor());
        coord.initialize();
        coord.start();
        coord.tick();
        coord.tick();

        coord.reset();

        assert_eq!(coord.phase(), CoordinatorPhase::Created);
        assert_eq!(coord.region_count(), 0);
        assert_eq!(coord.current_tick(), 0);
        assert_eq!(coord.stats().total_ticks, 0);
    }

    // ---------------------------------------------------------------
    // Phase display
    // ---------------------------------------------------------------

    #[test]
    fn test_phase_display() {
        assert_eq!(format!("{}", CoordinatorPhase::Created), "Created");
        assert_eq!(format!("{}", CoordinatorPhase::Running), "Running");
        assert_eq!(format!("{}", CoordinatorPhase::Stopped), "Stopped");
    }

    // ---------------------------------------------------------------
    // Full lifecycle integration test
    // ---------------------------------------------------------------

    #[test]
    fn test_full_lifecycle() {
        let mut coord = BrainCoordinator::with_config(small_config());

        // Register regions
        coord.register_region(thalamus_descriptor());
        coord.register_region(amygdala_descriptor());

        // Initialize and start
        coord.initialize();
        coord.start();

        // Run several ticks
        for i in 0..10 {
            let ticked = coord.tick();
            assert_eq!(ticked, 2, "tick {} should process 2 regions", i);
        }

        assert_eq!(coord.current_tick(), 10);
        assert_eq!(coord.stats().total_ticks, 10);
        assert_eq!(coord.region("thalamus").unwrap().ticks_processed, 10);
        assert_eq!(coord.region("amygdala").unwrap().ticks_processed, 10);

        // Verify health checks ran (interval = 2, so 5 checks at ticks 2,4,6,8,10)
        assert_eq!(coord.stats().total_health_checks, 5);

        // Send some signals
        coord.send_signal("risk.flash_crash", "amygdala", 0.99);
        coord.send_signal("market.regime", "thalamus", 0.5);

        // Verify routing
        let risk_msgs = coord.drain_channel("risk");
        assert_eq!(risk_msgs.len(), 1);
        assert_eq!(risk_msgs[0].topic, "risk.flash_crash");

        // Pause, resume, shutdown
        coord.pause();
        assert_eq!(coord.tick(), 0); // no ticks while paused
        coord.resume();
        assert_eq!(coord.tick(), 2); // back to ticking

        coord.shutdown();
        assert_eq!(coord.phase(), CoordinatorPhase::Stopped);
    }

    // ---------------------------------------------------------------
    // Heterogeneity integration
    // ---------------------------------------------------------------

    #[test]
    fn test_heterogeneity_profile_generated() {
        let config = BrainCoordinatorConfig {
            instance_id: "test-prod-42".to_string(),
            ..Default::default()
        };
        let coord = BrainCoordinator::with_config(config);

        let profile = coord.heterogeneity_profile();
        assert_eq!(profile.instance_id, "test-prod-42");
        assert_ne!(profile.seed, 0);
    }

    #[test]
    fn test_heterogeneity_deterministic_across_restarts() {
        let config = BrainCoordinatorConfig {
            instance_id: "stable-instance".to_string(),
            ..Default::default()
        };

        let coord1 = BrainCoordinator::with_config(config.clone());
        let coord2 = BrainCoordinator::with_config(config);

        assert_eq!(
            coord1.heterogeneity_profile().seed,
            coord2.heterogeneity_profile().seed,
        );
        assert_eq!(
            coord1.heterogeneity_profile().thalamus.orderbook_weight,
            coord2.heterogeneity_profile().thalamus.orderbook_weight,
        );
    }

    #[test]
    fn test_heterogeneity_different_instances() {
        let config_a = BrainCoordinatorConfig {
            instance_id: "instance-alpha".to_string(),
            ..Default::default()
        };
        let config_b = BrainCoordinatorConfig {
            instance_id: "instance-beta".to_string(),
            ..Default::default()
        };

        let coord_a = BrainCoordinator::with_config(config_a);
        let coord_b = BrainCoordinator::with_config(config_b);

        let dist = coord_a
            .heterogeneity_profile()
            .distance(coord_b.heterogeneity_profile());
        assert!(
            dist > 0.0,
            "different instances should have different profiles"
        );
    }

    #[test]
    fn test_heterogeneity_published_to_global_state() {
        let config = BrainCoordinatorConfig {
            instance_id: "state-test".to_string(),
            ..Default::default()
        };
        let mut coord = BrainCoordinator::with_config(config);
        coord.initialize();

        // Verify heterogeneity parameters are in global state
        assert!(coord.has_state("heterogeneity.seed"));
        assert!(coord.has_state("heterogeneity.thalamus.orderbook_weight"));
        assert!(coord.has_state("heterogeneity.basal_ganglia.dopamine_sensitivity"));
        assert!(coord.has_state("heterogeneity.cerebellum.ac_lambda"));

        // Values should match the profile
        let profile = coord.heterogeneity_profile().clone();
        assert_eq!(
            coord.get_state("heterogeneity.thalamus.orderbook_weight"),
            Some(profile.thalamus.orderbook_weight),
        );
    }

    #[test]
    fn test_heterogeneity_preserved_after_reset() {
        let config = BrainCoordinatorConfig {
            instance_id: "reset-test".to_string(),
            ..Default::default()
        };
        let mut coord = BrainCoordinator::with_config(config);
        let seed_before = coord.heterogeneity_profile().seed;

        coord.initialize();
        coord.reset();

        // Same instance_id → same profile after reset
        assert_eq!(coord.heterogeneity_profile().seed, seed_before);
    }

    #[test]
    fn test_instance_id_accessor() {
        let config = BrainCoordinatorConfig {
            instance_id: "my-inst".to_string(),
            ..Default::default()
        };
        let coord = BrainCoordinator::with_config(config);
        assert_eq!(coord.instance_id(), "my-inst");
    }
}
