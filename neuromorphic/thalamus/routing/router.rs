//! Dynamic routing logic
//!
//! Part of the Thalamus region - handles intelligent routing of market signals
//! to appropriate processing components based on signal characteristics.

use crate::common::Result;
use std::collections::HashMap;
use std::sync::Arc;
use tokio::sync::RwLock;

/// Route identifier
pub type RouteId = String;

/// Signal type classification for routing decisions
#[derive(Debug, Clone, PartialEq, Eq, Hash)]
pub enum SignalType {
    /// Price-related signals
    Price,
    /// Volume-related signals
    Volume,
    /// Order book signals
    OrderBook,
    /// Sentiment signals from news/social
    Sentiment,
    /// Risk/threat signals
    Risk,
    /// External data signals (weather, celestial)
    External,
    /// Composite/fused signals
    Composite,
}

/// Priority level for routing
#[derive(Debug, Clone, Copy, PartialEq, Eq, PartialOrd, Ord, Hash)]
pub enum RoutePriority {
    /// Lowest priority - can be delayed
    Low = 1,
    /// Normal processing priority
    Normal = 2,
    /// Higher priority - should be processed soon
    High = 3,
    /// Critical - must be processed immediately
    Critical = 4,
    /// Emergency - bypass normal queuing
    Emergency = 5,
}

impl Default for RoutePriority {
    fn default() -> Self {
        Self::Normal
    }
}

/// Routing destination
#[derive(Debug, Clone)]
pub struct RouteDestination {
    /// Unique identifier for this destination
    pub id: RouteId,
    /// Human-readable name
    pub name: String,
    /// Target brain region
    pub region: String,
    /// Whether this destination is active
    pub active: bool,
    /// Maximum queue depth before backpressure
    pub max_queue_depth: usize,
    /// Current queue depth
    pub current_queue_depth: usize,
}

impl RouteDestination {
    /// Create a new route destination
    pub fn new(id: impl Into<String>, name: impl Into<String>, region: impl Into<String>) -> Self {
        Self {
            id: id.into(),
            name: name.into(),
            region: region.into(),
            active: true,
            max_queue_depth: 1000,
            current_queue_depth: 0,
        }
    }

    /// Check if destination can accept more signals
    pub fn can_accept(&self) -> bool {
        self.active && self.current_queue_depth < self.max_queue_depth
    }

    /// Get load factor (0.0 - 1.0)
    pub fn load_factor(&self) -> f64 {
        if self.max_queue_depth == 0 {
            return 1.0;
        }
        self.current_queue_depth as f64 / self.max_queue_depth as f64
    }
}

/// Routing rule for signal dispatch
#[derive(Debug, Clone)]
pub struct RoutingRule {
    /// Rule identifier
    pub id: String,
    /// Signal types this rule applies to
    pub signal_types: Vec<SignalType>,
    /// Minimum priority for this rule to activate
    pub min_priority: RoutePriority,
    /// Destination IDs to route to
    pub destinations: Vec<RouteId>,
    /// Whether rule is enabled
    pub enabled: bool,
    /// Rule weight for conflict resolution
    pub weight: f64,
}

impl RoutingRule {
    /// Create a new routing rule
    pub fn new(id: impl Into<String>, destinations: Vec<RouteId>) -> Self {
        Self {
            id: id.into(),
            signal_types: vec![],
            min_priority: RoutePriority::Low,
            destinations,
            enabled: true,
            weight: 1.0,
        }
    }

    /// Check if rule matches given signal type and priority
    pub fn matches(&self, signal_type: &SignalType, priority: RoutePriority) -> bool {
        if !self.enabled {
            return false;
        }
        if priority < self.min_priority {
            return false;
        }
        if self.signal_types.is_empty() {
            return true; // Match all if no specific types
        }
        self.signal_types.contains(signal_type)
    }

    /// Add signal type filter
    pub fn with_signal_type(mut self, signal_type: SignalType) -> Self {
        self.signal_types.push(signal_type);
        self
    }

    /// Set minimum priority
    pub fn with_min_priority(mut self, priority: RoutePriority) -> Self {
        self.min_priority = priority;
        self
    }

    /// Set rule weight
    pub fn with_weight(mut self, weight: f64) -> Self {
        self.weight = weight;
        self
    }
}

/// Signal to be routed
#[derive(Debug, Clone)]
pub struct RoutableSignal {
    /// Signal identifier
    pub id: String,
    /// Signal type
    pub signal_type: SignalType,
    /// Priority level
    pub priority: RoutePriority,
    /// Signal payload (serialized data)
    pub payload: Vec<u8>,
    /// Signal timestamp
    pub timestamp: u64,
    /// Source identifier
    pub source: String,
    /// Symbol (if applicable)
    pub symbol: Option<String>,
}

impl RoutableSignal {
    /// Create a new routable signal
    pub fn new(
        id: impl Into<String>,
        signal_type: SignalType,
        priority: RoutePriority,
        payload: Vec<u8>,
    ) -> Self {
        Self {
            id: id.into(),
            signal_type,
            priority,
            payload,
            timestamp: std::time::SystemTime::now()
                .duration_since(std::time::UNIX_EPOCH)
                .unwrap_or_default()
                .as_millis() as u64,
            source: String::new(),
            symbol: None,
        }
    }

    /// Set signal source
    pub fn with_source(mut self, source: impl Into<String>) -> Self {
        self.source = source.into();
        self
    }

    /// Set symbol
    pub fn with_symbol(mut self, symbol: impl Into<String>) -> Self {
        self.symbol = Some(symbol.into());
        self
    }
}

/// Routing decision result
#[derive(Debug, Clone)]
pub struct RoutingDecision {
    /// Signal that was routed
    pub signal_id: String,
    /// Selected destinations
    pub destinations: Vec<RouteId>,
    /// Rules that matched
    pub matched_rules: Vec<String>,
    /// Routing latency in microseconds
    pub latency_us: u64,
    /// Whether any fallback was used
    pub used_fallback: bool,
}

/// Router statistics
#[derive(Debug, Clone, Default)]
pub struct RouterStats {
    /// Total signals routed
    pub signals_routed: u64,
    /// Signals dropped due to backpressure
    pub signals_dropped: u64,
    /// Average routing latency in microseconds
    pub avg_latency_us: f64,
    /// Routing decisions by signal type
    pub by_signal_type: HashMap<String, u64>,
    /// Routing decisions by priority
    pub by_priority: HashMap<u8, u64>,
}

/// Dynamic routing logic for the thalamus
pub struct Router {
    /// Registered destinations
    destinations: Arc<RwLock<HashMap<RouteId, RouteDestination>>>,
    /// Routing rules
    rules: Arc<RwLock<Vec<RoutingRule>>>,
    /// Default destinations for each signal type
    default_routes: Arc<RwLock<HashMap<SignalType, Vec<RouteId>>>>,
    /// Statistics
    stats: Arc<RwLock<RouterStats>>,
    /// Whether router is active
    active: Arc<RwLock<bool>>,
    /// Load balancing enabled
    load_balance: bool,
}

impl Default for Router {
    fn default() -> Self {
        Self::new()
    }
}

impl Router {
    /// Create a new router
    pub fn new() -> Self {
        let router = Self {
            destinations: Arc::new(RwLock::new(HashMap::new())),
            rules: Arc::new(RwLock::new(Vec::new())),
            default_routes: Arc::new(RwLock::new(HashMap::new())),
            stats: Arc::new(RwLock::new(RouterStats::default())),
            active: Arc::new(RwLock::new(true)),
            load_balance: true,
        };

        // Initialize with default configuration
        router
    }

    /// Create router with load balancing setting
    pub fn with_load_balance(mut self, enabled: bool) -> Self {
        self.load_balance = enabled;
        self
    }

    /// Register a destination
    pub async fn register_destination(&self, destination: RouteDestination) {
        let mut destinations = self.destinations.write().await;
        destinations.insert(destination.id.clone(), destination);
    }

    /// Unregister a destination
    pub async fn unregister_destination(&self, id: &str) -> Option<RouteDestination> {
        let mut destinations = self.destinations.write().await;
        destinations.remove(id)
    }

    /// Add a routing rule
    pub async fn add_rule(&self, rule: RoutingRule) {
        let mut rules = self.rules.write().await;
        rules.push(rule);
        // Sort by weight (higher weight first)
        rules.sort_by(|a, b| {
            b.weight
                .partial_cmp(&a.weight)
                .unwrap_or(std::cmp::Ordering::Equal)
        });
    }

    /// Set default routes for a signal type
    pub async fn set_default_routes(&self, signal_type: SignalType, routes: Vec<RouteId>) {
        let mut defaults = self.default_routes.write().await;
        defaults.insert(signal_type, routes);
    }

    /// Route a signal to appropriate destinations
    pub async fn route(&self, signal: &RoutableSignal) -> Result<RoutingDecision> {
        let start = std::time::Instant::now();

        // Check if router is active
        if !*self.active.read().await {
            return Err(anyhow::anyhow!("Router is not active").into());
        }

        let mut decision = RoutingDecision {
            signal_id: signal.id.clone(),
            destinations: Vec::new(),
            matched_rules: Vec::new(),
            latency_us: 0,
            used_fallback: false,
        };

        // Find matching rules
        let rules = self.rules.read().await;
        for rule in rules.iter() {
            if rule.matches(&signal.signal_type, signal.priority) {
                decision.matched_rules.push(rule.id.clone());
                for dest_id in &rule.destinations {
                    if !decision.destinations.contains(dest_id) {
                        decision.destinations.push(dest_id.clone());
                    }
                }
            }
        }

        // If no rules matched, use default routes
        if decision.destinations.is_empty() {
            let defaults = self.default_routes.read().await;
            if let Some(routes) = defaults.get(&signal.signal_type) {
                decision.destinations = routes.clone();
                decision.used_fallback = true;
            }
        }

        // Filter destinations based on availability
        let destinations = self.destinations.read().await;
        decision.destinations.retain(|id| {
            destinations
                .get(id)
                .map(|d| d.can_accept())
                .unwrap_or(false)
        });

        // Apply load balancing if enabled
        if self.load_balance && decision.destinations.len() > 1 {
            self.apply_load_balancing(&mut decision.destinations, &destinations);
        }

        // Update statistics
        decision.latency_us = start.elapsed().as_micros() as u64;
        self.update_stats(&signal, &decision).await;

        Ok(decision)
    }

    /// Apply load balancing to destination selection
    fn apply_load_balancing(
        &self,
        destinations: &mut Vec<RouteId>,
        dest_map: &HashMap<RouteId, RouteDestination>,
    ) {
        // Sort destinations by load factor (least loaded first)
        destinations.sort_by(|a, b| {
            let load_a = dest_map.get(a).map(|d| d.load_factor()).unwrap_or(1.0);
            let load_b = dest_map.get(b).map(|d| d.load_factor()).unwrap_or(1.0);
            load_a
                .partial_cmp(&load_b)
                .unwrap_or(std::cmp::Ordering::Equal)
        });
    }

    /// Update statistics after routing decision
    async fn update_stats(&self, signal: &RoutableSignal, decision: &RoutingDecision) {
        let mut stats = self.stats.write().await;
        stats.signals_routed += 1;

        // Update by signal type
        let type_key = format!("{:?}", signal.signal_type);
        *stats.by_signal_type.entry(type_key).or_insert(0) += 1;

        // Update by priority
        *stats.by_priority.entry(signal.priority as u8).or_insert(0) += 1;

        // Update average latency (exponential moving average)
        let alpha = 0.1;
        stats.avg_latency_us =
            stats.avg_latency_us * (1.0 - alpha) + decision.latency_us as f64 * alpha;

        // Track dropped signals
        if decision.destinations.is_empty() {
            stats.signals_dropped += 1;
        }
    }

    /// Get router statistics
    pub async fn stats(&self) -> RouterStats {
        self.stats.read().await.clone()
    }

    /// Set router active state
    pub async fn set_active(&self, active: bool) {
        *self.active.write().await = active;
    }

    /// Check if router is active
    pub async fn is_active(&self) -> bool {
        *self.active.read().await
    }

    /// Get all registered destinations
    pub async fn get_destinations(&self) -> Vec<RouteDestination> {
        self.destinations.read().await.values().cloned().collect()
    }

    /// Get all routing rules
    pub async fn get_rules(&self) -> Vec<RoutingRule> {
        self.rules.read().await.clone()
    }

    /// Update destination queue depth
    pub async fn update_queue_depth(&self, dest_id: &str, depth: usize) {
        let mut destinations = self.destinations.write().await;
        if let Some(dest) = destinations.get_mut(dest_id) {
            dest.current_queue_depth = depth;
        }
    }

    /// Main processing function (for compatibility)
    pub fn process(&self) -> Result<()> {
        Ok(())
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[tokio::test]
    async fn test_router_creation() {
        let router = Router::new();
        assert!(router.is_active().await);
    }

    #[tokio::test]
    async fn test_register_destination() {
        let router = Router::new();

        let dest = RouteDestination::new("amygdala", "Amygdala", "amygdala");
        router.register_destination(dest).await;

        let destinations = router.get_destinations().await;
        assert_eq!(destinations.len(), 1);
        assert_eq!(destinations[0].id, "amygdala");
    }

    #[tokio::test]
    async fn test_add_rule() {
        let router = Router::new();

        let rule = RoutingRule::new("risk_rule", vec!["amygdala".to_string()])
            .with_signal_type(SignalType::Risk)
            .with_min_priority(RoutePriority::High);

        router.add_rule(rule).await;

        let rules = router.get_rules().await;
        assert_eq!(rules.len(), 1);
        assert_eq!(rules[0].id, "risk_rule");
    }

    #[tokio::test]
    async fn test_route_signal() {
        let router = Router::new();

        // Register destination
        let dest = RouteDestination::new("prefrontal", "Prefrontal Cortex", "prefrontal");
        router.register_destination(dest).await;

        // Add rule
        let rule = RoutingRule::new("price_rule", vec!["prefrontal".to_string()])
            .with_signal_type(SignalType::Price);
        router.add_rule(rule).await;

        // Create and route signal
        let signal = RoutableSignal::new(
            "sig_001",
            SignalType::Price,
            RoutePriority::Normal,
            vec![1, 2, 3],
        );

        let decision = router.route(&signal).await.unwrap();
        assert_eq!(decision.destinations.len(), 1);
        assert_eq!(decision.destinations[0], "prefrontal");
    }

    #[tokio::test]
    async fn test_default_routes() {
        let router = Router::new();

        // Register destination
        let dest = RouteDestination::new("hippocampus", "Hippocampus", "hippocampus");
        router.register_destination(dest).await;

        // Set default route
        router
            .set_default_routes(SignalType::Sentiment, vec!["hippocampus".to_string()])
            .await;

        // Route signal (no rules match, should use default)
        let signal = RoutableSignal::new(
            "sig_002",
            SignalType::Sentiment,
            RoutePriority::Normal,
            vec![],
        );

        let decision = router.route(&signal).await.unwrap();
        assert!(decision.used_fallback);
        assert_eq!(decision.destinations.len(), 1);
    }

    #[tokio::test]
    async fn test_load_balancing() {
        let router = Router::new();

        // Register two destinations with different loads
        let mut dest1 = RouteDestination::new("dest1", "Destination 1", "region1");
        dest1.current_queue_depth = 100;
        dest1.max_queue_depth = 1000;

        let mut dest2 = RouteDestination::new("dest2", "Destination 2", "region2");
        dest2.current_queue_depth = 10;
        dest2.max_queue_depth = 1000;

        router.register_destination(dest1).await;
        router.register_destination(dest2).await;

        // Add rule routing to both
        let rule = RoutingRule::new("multi_rule", vec!["dest1".to_string(), "dest2".to_string()]);
        router.add_rule(rule).await;

        // Route signal
        let signal =
            RoutableSignal::new("sig_003", SignalType::Price, RoutePriority::Normal, vec![]);

        let decision = router.route(&signal).await.unwrap();

        // dest2 should be first (lower load)
        assert_eq!(decision.destinations.len(), 2);
        assert_eq!(decision.destinations[0], "dest2");
    }

    #[tokio::test]
    async fn test_backpressure() {
        let router = Router::new();

        // Register destination at capacity
        let mut dest = RouteDestination::new("full", "Full Destination", "region");
        dest.current_queue_depth = 1000;
        dest.max_queue_depth = 1000;
        router.register_destination(dest).await;

        // Add rule
        let rule = RoutingRule::new("full_rule", vec!["full".to_string()]);
        router.add_rule(rule).await;

        // Route signal
        let signal =
            RoutableSignal::new("sig_004", SignalType::Price, RoutePriority::Normal, vec![]);

        let decision = router.route(&signal).await.unwrap();

        // Should be filtered out due to backpressure
        assert!(decision.destinations.is_empty());
    }

    #[tokio::test]
    async fn test_statistics() {
        let router = Router::new();

        let dest = RouteDestination::new("stats_dest", "Stats Dest", "region");
        router.register_destination(dest).await;

        let rule = RoutingRule::new("stats_rule", vec!["stats_dest".to_string()]);
        router.add_rule(rule).await;

        // Route multiple signals
        for i in 0..5 {
            let signal = RoutableSignal::new(
                format!("sig_{}", i),
                SignalType::Price,
                RoutePriority::Normal,
                vec![],
            );
            router.route(&signal).await.unwrap();
        }

        let stats = router.stats().await;
        assert_eq!(stats.signals_routed, 5);
    }

    #[test]
    fn test_route_destination_load_factor() {
        let mut dest = RouteDestination::new("test", "Test", "region");
        dest.current_queue_depth = 250;
        dest.max_queue_depth = 1000;

        assert!((dest.load_factor() - 0.25).abs() < 0.001);
    }

    #[test]
    fn test_routing_rule_matches() {
        let rule = RoutingRule::new("test", vec![])
            .with_signal_type(SignalType::Risk)
            .with_min_priority(RoutePriority::High);

        // Should match
        assert!(rule.matches(&SignalType::Risk, RoutePriority::High));
        assert!(rule.matches(&SignalType::Risk, RoutePriority::Critical));

        // Should not match
        assert!(!rule.matches(&SignalType::Price, RoutePriority::High));
        assert!(!rule.matches(&SignalType::Risk, RoutePriority::Normal));
    }

    #[test]
    fn test_process_compatibility() {
        let router = Router::new();
        assert!(router.process().is_ok());
    }
}
