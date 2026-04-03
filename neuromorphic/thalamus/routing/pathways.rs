//! Information pathways to cortex
//!
//! Part of the Thalamus region - manages information flow pathways
//! from sensory processing to cortical regions for higher-level analysis.

use crate::common::Result;
use std::collections::{HashMap, HashSet};
use std::sync::Arc;
use tokio::sync::RwLock;

/// Pathway identifier
pub type PathwayId = String;

/// Cortical region identifier
pub type RegionId = String;

/// Pathway state
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum PathwayState {
    /// Pathway is active and transmitting
    Active,
    /// Pathway is temporarily inhibited
    Inhibited,
    /// Pathway is potentiated (enhanced transmission)
    Potentiated,
    /// Pathway is dormant (not in use)
    Dormant,
    /// Pathway is damaged/unavailable
    Damaged,
}

impl Default for PathwayState {
    fn default() -> Self {
        Self::Active
    }
}

/// Pathway transmission mode
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum TransmissionMode {
    /// Direct transmission - immediate delivery
    Direct,
    /// Buffered transmission - batched delivery
    Buffered,
    /// Filtered transmission - selective delivery
    Filtered,
    /// Amplified transmission - enhanced signal strength
    Amplified,
    /// Attenuated transmission - reduced signal strength
    Attenuated,
}

impl Default for TransmissionMode {
    fn default() -> Self {
        Self::Direct
    }
}

/// Pathway connection to a cortical region
#[derive(Debug, Clone)]
pub struct PathwayConnection {
    /// Target region identifier
    pub target_region: RegionId,
    /// Connection strength (0.0 - 1.0)
    pub strength: f64,
    /// Transmission delay in milliseconds
    pub delay_ms: u64,
    /// Whether connection is excitatory (true) or inhibitory (false)
    pub excitatory: bool,
    /// Connection state
    pub state: PathwayState,
    /// Transmission mode
    pub mode: TransmissionMode,
    /// Bandwidth capacity (signals per second)
    pub bandwidth: u32,
    /// Current utilization (0.0 - 1.0)
    pub utilization: f64,
}

impl PathwayConnection {
    /// Create a new excitatory connection
    pub fn excitatory(target: impl Into<RegionId>, strength: f64) -> Self {
        Self {
            target_region: target.into(),
            strength: strength.clamp(0.0, 1.0),
            delay_ms: 1,
            excitatory: true,
            state: PathwayState::Active,
            mode: TransmissionMode::Direct,
            bandwidth: 1000,
            utilization: 0.0,
        }
    }

    /// Create a new inhibitory connection
    pub fn inhibitory(target: impl Into<RegionId>, strength: f64) -> Self {
        Self {
            target_region: target.into(),
            strength: strength.clamp(0.0, 1.0),
            delay_ms: 1,
            excitatory: false,
            state: PathwayState::Active,
            mode: TransmissionMode::Direct,
            bandwidth: 1000,
            utilization: 0.0,
        }
    }

    /// Set transmission delay
    pub fn with_delay(mut self, delay_ms: u64) -> Self {
        self.delay_ms = delay_ms;
        self
    }

    /// Set transmission mode
    pub fn with_mode(mut self, mode: TransmissionMode) -> Self {
        self.mode = mode;
        self
    }

    /// Set bandwidth capacity
    pub fn with_bandwidth(mut self, bandwidth: u32) -> Self {
        self.bandwidth = bandwidth;
        self
    }

    /// Check if connection can accept more traffic
    pub fn has_capacity(&self) -> bool {
        self.state == PathwayState::Active && self.utilization < 0.95
    }

    /// Calculate effective transmission strength
    pub fn effective_strength(&self) -> f64 {
        let state_modifier = match self.state {
            PathwayState::Active => 1.0,
            PathwayState::Potentiated => 1.5,
            PathwayState::Inhibited => 0.3,
            PathwayState::Dormant => 0.0,
            PathwayState::Damaged => 0.0,
        };

        let mode_modifier = match self.mode {
            TransmissionMode::Direct => 1.0,
            TransmissionMode::Buffered => 0.9,
            TransmissionMode::Filtered => 0.8,
            TransmissionMode::Amplified => 1.3,
            TransmissionMode::Attenuated => 0.5,
        };

        self.strength * state_modifier * mode_modifier
    }
}

/// Information pathway definition
#[derive(Debug, Clone)]
pub struct Pathway {
    /// Unique pathway identifier
    pub id: PathwayId,
    /// Human-readable name
    pub name: String,
    /// Source region/component
    pub source: String,
    /// Connections to target regions
    pub connections: Vec<PathwayConnection>,
    /// Pathway state
    pub state: PathwayState,
    /// Signal types this pathway handles
    pub signal_types: HashSet<String>,
    /// Whether pathway is bidirectional
    pub bidirectional: bool,
    /// Priority level for this pathway
    pub priority: u8,
    /// Total signals transmitted
    pub signals_transmitted: u64,
    /// Signals dropped due to capacity
    pub signals_dropped: u64,
}

impl Pathway {
    /// Create a new pathway
    pub fn new(id: impl Into<String>, name: impl Into<String>, source: impl Into<String>) -> Self {
        Self {
            id: id.into(),
            name: name.into(),
            source: source.into(),
            connections: Vec::new(),
            state: PathwayState::Active,
            signal_types: HashSet::new(),
            bidirectional: false,
            priority: 5,
            signals_transmitted: 0,
            signals_dropped: 0,
        }
    }

    /// Add a connection to this pathway
    pub fn add_connection(&mut self, connection: PathwayConnection) {
        self.connections.push(connection);
    }

    /// Add connection with builder pattern
    pub fn with_connection(mut self, connection: PathwayConnection) -> Self {
        self.connections.push(connection);
        self
    }

    /// Set signal types handled
    pub fn with_signal_types(mut self, types: Vec<&str>) -> Self {
        self.signal_types = types.into_iter().map(String::from).collect();
        self
    }

    /// Set as bidirectional
    pub fn bidirectional(mut self) -> Self {
        self.bidirectional = true;
        self
    }

    /// Set priority
    pub fn with_priority(mut self, priority: u8) -> Self {
        self.priority = priority;
        self
    }

    /// Check if pathway is available for transmission
    pub fn is_available(&self) -> bool {
        matches!(self.state, PathwayState::Active | PathwayState::Potentiated)
    }

    /// Get all target regions
    pub fn target_regions(&self) -> Vec<&RegionId> {
        self.connections.iter().map(|c| &c.target_region).collect()
    }

    /// Get connections that can accept traffic
    pub fn available_connections(&self) -> Vec<&PathwayConnection> {
        self.connections
            .iter()
            .filter(|c| c.has_capacity())
            .collect()
    }

    /// Calculate average connection strength
    pub fn average_strength(&self) -> f64 {
        if self.connections.is_empty() {
            return 0.0;
        }
        let total: f64 = self
            .connections
            .iter()
            .map(|c| c.effective_strength())
            .sum();
        total / self.connections.len() as f64
    }

    /// Get transmission success rate
    pub fn success_rate(&self) -> f64 {
        let total = self.signals_transmitted + self.signals_dropped;
        if total == 0 {
            return 1.0;
        }
        self.signals_transmitted as f64 / total as f64
    }
}

/// Transmission request
#[derive(Debug, Clone)]
pub struct TransmissionRequest {
    /// Source identifier
    pub source: String,
    /// Target regions (empty for broadcast)
    pub targets: Vec<RegionId>,
    /// Signal type
    pub signal_type: String,
    /// Signal data
    pub data: Vec<u8>,
    /// Priority (1-10)
    pub priority: u8,
    /// Whether to require acknowledgment
    pub require_ack: bool,
}

impl TransmissionRequest {
    /// Create a new transmission request
    pub fn new(source: impl Into<String>, signal_type: impl Into<String>, data: Vec<u8>) -> Self {
        Self {
            source: source.into(),
            targets: Vec::new(),
            signal_type: signal_type.into(),
            data,
            priority: 5,
            require_ack: false,
        }
    }

    /// Set target regions
    pub fn to_regions(mut self, regions: Vec<RegionId>) -> Self {
        self.targets = regions;
        self
    }

    /// Set priority
    pub fn with_priority(mut self, priority: u8) -> Self {
        self.priority = priority.clamp(1, 10);
        self
    }

    /// Require acknowledgment
    pub fn require_ack(mut self) -> Self {
        self.require_ack = true;
        self
    }
}

/// Transmission result
#[derive(Debug, Clone)]
pub struct TransmissionResult {
    /// Pathways used for transmission
    pub pathways_used: Vec<PathwayId>,
    /// Regions successfully reached
    pub regions_reached: Vec<RegionId>,
    /// Regions that failed
    pub regions_failed: Vec<RegionId>,
    /// Total transmission time in microseconds
    pub transmission_time_us: u64,
    /// Whether all targets were reached
    pub success: bool,
}

/// Pathways statistics
#[derive(Debug, Clone, Default)]
pub struct PathwaysStats {
    /// Total pathways registered
    pub total_pathways: usize,
    /// Active pathways
    pub active_pathways: usize,
    /// Total transmissions
    pub total_transmissions: u64,
    /// Successful transmissions
    pub successful_transmissions: u64,
    /// Failed transmissions
    pub failed_transmissions: u64,
    /// Average transmission latency in microseconds
    pub avg_latency_us: f64,
    /// Transmissions by signal type
    pub by_signal_type: HashMap<String, u64>,
    /// Transmissions by target region
    pub by_region: HashMap<String, u64>,
}

/// Information pathways manager
pub struct Pathways {
    /// Registered pathways
    pathways: Arc<RwLock<HashMap<PathwayId, Pathway>>>,
    /// Region to pathways mapping for efficient lookup
    region_pathways: Arc<RwLock<HashMap<RegionId, Vec<PathwayId>>>>,
    /// Signal type to pathways mapping
    signal_pathways: Arc<RwLock<HashMap<String, Vec<PathwayId>>>>,
    /// Statistics
    stats: Arc<RwLock<PathwaysStats>>,
    /// Whether system is active
    active: Arc<RwLock<bool>>,
    /// Default pathways for unknown signal types
    default_pathways: Arc<RwLock<Vec<PathwayId>>>,
}

impl Default for Pathways {
    fn default() -> Self {
        Self::new()
    }
}

impl Pathways {
    /// Create a new pathways manager
    pub fn new() -> Self {
        Self {
            pathways: Arc::new(RwLock::new(HashMap::new())),
            region_pathways: Arc::new(RwLock::new(HashMap::new())),
            signal_pathways: Arc::new(RwLock::new(HashMap::new())),
            stats: Arc::new(RwLock::new(PathwaysStats::default())),
            active: Arc::new(RwLock::new(true)),
            default_pathways: Arc::new(RwLock::new(Vec::new())),
        }
    }

    /// Register a new pathway
    pub async fn register_pathway(&self, pathway: Pathway) {
        let pathway_id = pathway.id.clone();

        // Update region mappings
        {
            let mut region_map = self.region_pathways.write().await;
            for region in pathway.target_regions() {
                region_map
                    .entry(region.clone())
                    .or_insert_with(Vec::new)
                    .push(pathway_id.clone());
            }
        }

        // Update signal type mappings
        {
            let mut signal_map = self.signal_pathways.write().await;
            for signal_type in &pathway.signal_types {
                signal_map
                    .entry(signal_type.clone())
                    .or_insert_with(Vec::new)
                    .push(pathway_id.clone());
            }
        }

        // Store pathway
        {
            let mut pathways = self.pathways.write().await;
            pathways.insert(pathway_id, pathway);
        }

        // Update stats
        {
            let pathways = self.pathways.read().await;
            let mut stats = self.stats.write().await;
            stats.total_pathways = pathways.len();
            stats.active_pathways = pathways.values().filter(|p| p.is_available()).count();
        }
    }

    /// Unregister a pathway
    pub async fn unregister_pathway(&self, pathway_id: &str) -> Option<Pathway> {
        let pathway = {
            let mut pathways = self.pathways.write().await;
            pathways.remove(pathway_id)
        };

        if let Some(ref p) = pathway {
            // Clean up region mappings
            let mut region_map = self.region_pathways.write().await;
            for region in p.target_regions() {
                if let Some(ids) = region_map.get_mut(region) {
                    ids.retain(|id| id != pathway_id);
                }
            }

            // Clean up signal mappings
            let mut signal_map = self.signal_pathways.write().await;
            for signal_type in &p.signal_types {
                if let Some(ids) = signal_map.get_mut(signal_type) {
                    ids.retain(|id| id != pathway_id);
                }
            }
        }

        pathway
    }

    /// Set default pathways for unknown signal types
    pub async fn set_default_pathways(&self, pathway_ids: Vec<PathwayId>) {
        *self.default_pathways.write().await = pathway_ids;
    }

    /// Transmit a signal through pathways
    pub async fn transmit(&self, request: TransmissionRequest) -> Result<TransmissionResult> {
        let start = std::time::Instant::now();

        if !*self.active.read().await {
            return Err(anyhow::anyhow!("Pathways system is not active").into());
        }

        let mut result = TransmissionResult {
            pathways_used: Vec::new(),
            regions_reached: Vec::new(),
            regions_failed: Vec::new(),
            transmission_time_us: 0,
            success: false,
        };

        // Find appropriate pathways
        let pathway_ids = self.find_pathways(&request).await;

        if pathway_ids.is_empty() {
            result.regions_failed = request.targets.clone();
            self.update_stats(&request, &result).await;
            return Ok(result);
        }

        // Transmit through each pathway
        let mut pathways = self.pathways.write().await;
        let target_set: HashSet<_> = request.targets.iter().cloned().collect();

        for pathway_id in pathway_ids {
            if let Some(pathway) = pathways.get_mut(&pathway_id) {
                if !pathway.is_available() {
                    continue;
                }

                // Check connections
                for connection in &pathway.connections {
                    // If no specific targets, send to all; otherwise check if target matches
                    if target_set.is_empty() || target_set.contains(&connection.target_region) {
                        if connection.has_capacity() {
                            result
                                .regions_reached
                                .push(connection.target_region.clone());
                            if !result.pathways_used.contains(&pathway_id) {
                                result.pathways_used.push(pathway_id.clone());
                            }
                        } else {
                            result.regions_failed.push(connection.target_region.clone());
                        }
                    }
                }

                pathway.signals_transmitted += 1;
            }
        }

        // Determine success
        if request.targets.is_empty() {
            result.success = !result.regions_reached.is_empty();
        } else {
            result.success = result.regions_failed.is_empty();
        }

        result.transmission_time_us = start.elapsed().as_micros() as u64;

        drop(pathways);
        self.update_stats(&request, &result).await;

        Ok(result)
    }

    /// Find pathways for a transmission request
    async fn find_pathways(&self, request: &TransmissionRequest) -> Vec<PathwayId> {
        let mut pathway_ids = Vec::new();

        // First try signal type mapping
        {
            let signal_map = self.signal_pathways.read().await;
            if let Some(ids) = signal_map.get(&request.signal_type) {
                pathway_ids.extend(ids.clone());
            }
        }

        // If specific targets, also check region mappings
        if !request.targets.is_empty() {
            let region_map = self.region_pathways.read().await;
            for target in &request.targets {
                if let Some(ids) = region_map.get(target) {
                    for id in ids {
                        if !pathway_ids.contains(id) {
                            pathway_ids.push(id.clone());
                        }
                    }
                }
            }
        }

        // Fall back to defaults if no pathways found
        if pathway_ids.is_empty() {
            let defaults = self.default_pathways.read().await;
            pathway_ids = defaults.clone();
        }

        pathway_ids
    }

    /// Update statistics after transmission
    async fn update_stats(&self, request: &TransmissionRequest, result: &TransmissionResult) {
        let mut stats = self.stats.write().await;

        stats.total_transmissions += 1;
        if result.success {
            stats.successful_transmissions += 1;
        } else {
            stats.failed_transmissions += 1;
        }

        // Update latency (EMA)
        let alpha = 0.1;
        stats.avg_latency_us =
            stats.avg_latency_us * (1.0 - alpha) + result.transmission_time_us as f64 * alpha;

        // Update by signal type
        *stats
            .by_signal_type
            .entry(request.signal_type.clone())
            .or_insert(0) += 1;

        // Update by region
        for region in &result.regions_reached {
            *stats.by_region.entry(region.clone()).or_insert(0) += 1;
        }
    }

    /// Get pathway by ID
    pub async fn get_pathway(&self, id: &str) -> Option<Pathway> {
        self.pathways.read().await.get(id).cloned()
    }

    /// Get all pathways
    pub async fn get_all_pathways(&self) -> Vec<Pathway> {
        self.pathways.read().await.values().cloned().collect()
    }

    /// Get pathways to a specific region
    pub async fn get_pathways_to_region(&self, region: &str) -> Vec<Pathway> {
        let region_map = self.region_pathways.read().await;
        let pathways = self.pathways.read().await;

        region_map
            .get(region)
            .map(|ids| {
                ids.iter()
                    .filter_map(|id| pathways.get(id).cloned())
                    .collect()
            })
            .unwrap_or_default()
    }

    /// Update pathway state
    pub async fn set_pathway_state(&self, pathway_id: &str, state: PathwayState) -> Result<()> {
        let mut pathways = self.pathways.write().await;
        if let Some(pathway) = pathways.get_mut(pathway_id) {
            pathway.state = state;
            Ok(())
        } else {
            Err(anyhow::anyhow!("Pathway not found: {}", pathway_id).into())
        }
    }

    /// Update connection utilization
    pub async fn update_connection_utilization(
        &self,
        pathway_id: &str,
        target_region: &str,
        utilization: f64,
    ) -> Result<()> {
        let mut pathways = self.pathways.write().await;
        if let Some(pathway) = pathways.get_mut(pathway_id) {
            for connection in &mut pathway.connections {
                if connection.target_region == target_region {
                    connection.utilization = utilization.clamp(0.0, 1.0);
                    return Ok(());
                }
            }
            Err(anyhow::anyhow!("Connection not found to region: {}", target_region).into())
        } else {
            Err(anyhow::anyhow!("Pathway not found: {}", pathway_id).into())
        }
    }

    /// Get statistics
    pub async fn stats(&self) -> PathwaysStats {
        // Update pathway counts
        let pathways = self.pathways.read().await;
        let mut stats = self.stats.write().await;
        stats.total_pathways = pathways.len();
        stats.active_pathways = pathways.values().filter(|p| p.is_available()).count();
        stats.clone()
    }

    /// Check if system is active
    pub async fn is_active(&self) -> bool {
        *self.active.read().await
    }

    /// Set active state
    pub async fn set_active(&self, active: bool) {
        *self.active.write().await = active;
    }

    /// Main processing function (for compatibility)
    pub fn process(&self) -> Result<()> {
        Ok(())
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_pathway_connection_strength() {
        let conn = PathwayConnection::excitatory("cortex", 0.8);
        assert!((conn.effective_strength() - 0.8).abs() < 0.001);

        let mut inhibited_conn = PathwayConnection::excitatory("cortex", 0.8);
        inhibited_conn.state = PathwayState::Inhibited;
        assert!(inhibited_conn.effective_strength() < 0.5);

        let mut potentiated_conn = PathwayConnection::excitatory("cortex", 0.8);
        potentiated_conn.state = PathwayState::Potentiated;
        assert!(potentiated_conn.effective_strength() > 1.0);
    }

    #[test]
    fn test_pathway_creation() {
        let pathway = Pathway::new("visual_pathway", "Visual Pathway", "eyes")
            .with_connection(PathwayConnection::excitatory("visual_cortex", 0.9))
            .with_connection(PathwayConnection::excitatory("prefrontal", 0.5))
            .with_signal_types(vec!["price", "volume"])
            .with_priority(8);

        assert_eq!(pathway.connections.len(), 2);
        assert!(pathway.signal_types.contains("price"));
        assert_eq!(pathway.priority, 8);
    }

    #[test]
    fn test_pathway_availability() {
        let mut pathway = Pathway::new("test", "Test", "source");

        assert!(pathway.is_available());

        pathway.state = PathwayState::Inhibited;
        assert!(!pathway.is_available());

        pathway.state = PathwayState::Potentiated;
        assert!(pathway.is_available());
    }

    #[tokio::test]
    async fn test_register_pathway() {
        let pathways = Pathways::new();

        let pathway = Pathway::new("amygdala_path", "Amygdala Pathway", "thalamus")
            .with_connection(PathwayConnection::excitatory("amygdala", 0.9))
            .with_signal_types(vec!["risk", "threat"]);

        pathways.register_pathway(pathway).await;

        let retrieved = pathways.get_pathway("amygdala_path").await;
        assert!(retrieved.is_some());
        assert_eq!(retrieved.unwrap().name, "Amygdala Pathway");
    }

    #[tokio::test]
    async fn test_transmit_signal() {
        let pathways = Pathways::new();

        let pathway = Pathway::new("price_path", "Price Pathway", "market_data")
            .with_connection(PathwayConnection::excitatory("prefrontal", 0.8))
            .with_signal_types(vec!["price"]);

        pathways.register_pathway(pathway).await;

        let request = TransmissionRequest::new("market_data", "price", vec![1, 2, 3]);

        let result = pathways.transmit(request).await.unwrap();
        assert!(result.success);
        assert!(!result.pathways_used.is_empty());
        assert!(result.regions_reached.contains(&"prefrontal".to_string()));
    }

    #[tokio::test]
    async fn test_transmit_to_specific_region() {
        let pathways = Pathways::new();

        let pathway = Pathway::new("multi_path", "Multi Pathway", "source")
            .with_connection(PathwayConnection::excitatory("region_a", 0.8))
            .with_connection(PathwayConnection::excitatory("region_b", 0.8));

        pathways.register_pathway(pathway).await;

        let request = TransmissionRequest::new("source", "signal", vec![])
            .to_regions(vec!["region_a".to_string()]);

        let result = pathways.transmit(request).await.unwrap();
        assert!(result.success);
        assert!(result.regions_reached.contains(&"region_a".to_string()));
    }

    #[tokio::test]
    async fn test_pathway_state_management() {
        let pathways = Pathways::new();

        let pathway = Pathway::new("test_path", "Test", "source")
            .with_connection(PathwayConnection::excitatory("target", 0.8));

        pathways.register_pathway(pathway).await;

        // Inhibit pathway
        pathways
            .set_pathway_state("test_path", PathwayState::Inhibited)
            .await
            .unwrap();

        let retrieved = pathways.get_pathway("test_path").await.unwrap();
        assert_eq!(retrieved.state, PathwayState::Inhibited);
        assert!(!retrieved.is_available());
    }

    #[tokio::test]
    async fn test_statistics() {
        let pathways = Pathways::new();

        let pathway = Pathway::new("stats_path", "Stats", "source")
            .with_connection(PathwayConnection::excitatory("target", 0.8))
            .with_signal_types(vec!["test"]);

        pathways.register_pathway(pathway).await;

        for _ in 0..5 {
            let request = TransmissionRequest::new("source", "test", vec![]);
            pathways.transmit(request).await.unwrap();
        }

        let stats = pathways.stats().await;
        assert_eq!(stats.total_transmissions, 5);
        assert_eq!(stats.successful_transmissions, 5);
    }

    #[tokio::test]
    async fn test_default_pathways() {
        let pathways = Pathways::new();

        let default_pathway = Pathway::new("default", "Default Pathway", "source")
            .with_connection(PathwayConnection::excitatory("fallback", 0.5));

        pathways.register_pathway(default_pathway).await;
        pathways
            .set_default_pathways(vec!["default".to_string()])
            .await;

        // Request with unknown signal type
        let request = TransmissionRequest::new("source", "unknown_type", vec![]);
        let result = pathways.transmit(request).await.unwrap();

        assert!(result.success);
        assert!(result.pathways_used.contains(&"default".to_string()));
    }

    #[tokio::test]
    async fn test_get_pathways_to_region() {
        let pathways = Pathways::new();

        let pathway1 = Pathway::new("path1", "Path 1", "source")
            .with_connection(PathwayConnection::excitatory("amygdala", 0.8));

        let pathway2 = Pathway::new("path2", "Path 2", "source")
            .with_connection(PathwayConnection::excitatory("amygdala", 0.6))
            .with_connection(PathwayConnection::excitatory("hippocampus", 0.7));

        pathways.register_pathway(pathway1).await;
        pathways.register_pathway(pathway2).await;

        let amygdala_pathways = pathways.get_pathways_to_region("amygdala").await;
        assert_eq!(amygdala_pathways.len(), 2);
    }

    #[test]
    fn test_process_compatibility() {
        let pathways = Pathways::new();
        assert!(pathways.process().is_ok());
    }
}
