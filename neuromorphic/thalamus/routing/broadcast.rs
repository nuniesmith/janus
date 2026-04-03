//! Broadcast to multiple regions
//!
//! Part of the Thalamus region - handles broadcasting signals to multiple
//! brain regions simultaneously for coordinated processing.

use crate::common::Result;
use std::collections::{HashMap, HashSet};
use std::sync::Arc;
use std::time::Instant;
use tokio::sync::RwLock;

/// Broadcast group identifier
pub type BroadcastGroupId = String;

/// Region identifier
pub type RegionId = String;

/// Broadcast delivery mode
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum DeliveryMode {
    /// Fire and forget - no confirmation needed
    FireAndForget,
    /// At least once delivery - retry until acknowledged
    AtLeastOnce,
    /// Exactly once delivery - deduplicated
    ExactlyOnce,
    /// Best effort - try once, log failures
    BestEffort,
}

impl Default for DeliveryMode {
    fn default() -> Self {
        Self::BestEffort
    }
}

/// Broadcast acknowledgment requirement
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum AckRequirement {
    /// No acknowledgment needed
    None,
    /// At least one region must acknowledge
    Any,
    /// Majority of regions must acknowledge
    Majority,
    /// All regions must acknowledge
    All,
}

impl Default for AckRequirement {
    fn default() -> Self {
        Self::None
    }
}

/// Broadcast message
#[derive(Debug, Clone)]
pub struct BroadcastMessage {
    /// Unique message identifier
    pub id: String,
    /// Message type/category
    pub message_type: String,
    /// Message payload
    pub payload: Vec<u8>,
    /// Source identifier
    pub source: String,
    /// Priority (1-10)
    pub priority: u8,
    /// Timestamp when message was created
    pub created_at: u64,
    /// Time-to-live in milliseconds (0 = no expiry)
    pub ttl_ms: u64,
    /// Delivery mode
    pub delivery_mode: DeliveryMode,
    /// Acknowledgment requirement
    pub ack_requirement: AckRequirement,
    /// Custom metadata
    pub metadata: HashMap<String, String>,
}

impl BroadcastMessage {
    /// Create a new broadcast message
    pub fn new(id: impl Into<String>, message_type: impl Into<String>, payload: Vec<u8>) -> Self {
        Self {
            id: id.into(),
            message_type: message_type.into(),
            payload,
            source: String::new(),
            priority: 5,
            created_at: std::time::SystemTime::now()
                .duration_since(std::time::UNIX_EPOCH)
                .unwrap_or_default()
                .as_millis() as u64,
            ttl_ms: 0,
            delivery_mode: DeliveryMode::default(),
            ack_requirement: AckRequirement::default(),
            metadata: HashMap::new(),
        }
    }

    /// Set source
    pub fn from_source(mut self, source: impl Into<String>) -> Self {
        self.source = source.into();
        self
    }

    /// Set priority
    pub fn with_priority(mut self, priority: u8) -> Self {
        self.priority = priority.clamp(1, 10);
        self
    }

    /// Set TTL
    pub fn with_ttl(mut self, ttl_ms: u64) -> Self {
        self.ttl_ms = ttl_ms;
        self
    }

    /// Set delivery mode
    pub fn with_delivery_mode(mut self, mode: DeliveryMode) -> Self {
        self.delivery_mode = mode;
        self
    }

    /// Set acknowledgment requirement
    pub fn with_ack_requirement(mut self, requirement: AckRequirement) -> Self {
        self.ack_requirement = requirement;
        self
    }

    /// Add metadata
    pub fn with_metadata(mut self, key: impl Into<String>, value: impl Into<String>) -> Self {
        self.metadata.insert(key.into(), value.into());
        self
    }

    /// Check if message has expired
    pub fn is_expired(&self) -> bool {
        if self.ttl_ms == 0 {
            return false;
        }
        let now = std::time::SystemTime::now()
            .duration_since(std::time::UNIX_EPOCH)
            .unwrap_or_default()
            .as_millis() as u64;
        now > self.created_at + self.ttl_ms
    }

    /// Get message age in milliseconds
    pub fn age_ms(&self) -> u64 {
        let now = std::time::SystemTime::now()
            .duration_since(std::time::UNIX_EPOCH)
            .unwrap_or_default()
            .as_millis() as u64;
        now.saturating_sub(self.created_at)
    }
}

/// Broadcast group definition
#[derive(Debug, Clone)]
pub struct BroadcastGroup {
    /// Group identifier
    pub id: BroadcastGroupId,
    /// Human-readable name
    pub name: String,
    /// Member regions
    pub members: HashSet<RegionId>,
    /// Whether group is active
    pub active: bool,
    /// Default delivery mode for this group
    pub default_delivery_mode: DeliveryMode,
    /// Default acknowledgment requirement
    pub default_ack_requirement: AckRequirement,
    /// Message types this group subscribes to
    pub subscribed_types: HashSet<String>,
    /// Minimum priority for messages (1-10)
    pub min_priority: u8,
    /// Total broadcasts sent to this group
    pub broadcasts_sent: u64,
    /// Total broadcasts failed
    pub broadcasts_failed: u64,
}

impl BroadcastGroup {
    /// Create a new broadcast group
    pub fn new(id: impl Into<String>, name: impl Into<String>) -> Self {
        Self {
            id: id.into(),
            name: name.into(),
            members: HashSet::new(),
            active: true,
            default_delivery_mode: DeliveryMode::default(),
            default_ack_requirement: AckRequirement::default(),
            subscribed_types: HashSet::new(),
            min_priority: 1,
            broadcasts_sent: 0,
            broadcasts_failed: 0,
        }
    }

    /// Add a member region
    pub fn add_member(&mut self, region: impl Into<RegionId>) {
        self.members.insert(region.into());
    }

    /// Add member with builder pattern
    pub fn with_member(mut self, region: impl Into<RegionId>) -> Self {
        self.members.insert(region.into());
        self
    }

    /// Add multiple members
    pub fn with_members(mut self, regions: Vec<&str>) -> Self {
        for region in regions {
            self.members.insert(region.to_string());
        }
        self
    }

    /// Subscribe to message types
    pub fn with_subscriptions(mut self, types: Vec<&str>) -> Self {
        self.subscribed_types = types.into_iter().map(String::from).collect();
        self
    }

    /// Set minimum priority
    pub fn with_min_priority(mut self, priority: u8) -> Self {
        self.min_priority = priority.clamp(1, 10);
        self
    }

    /// Set default delivery mode
    pub fn with_delivery_mode(mut self, mode: DeliveryMode) -> Self {
        self.default_delivery_mode = mode;
        self
    }

    /// Remove a member
    pub fn remove_member(&mut self, region: &str) -> bool {
        self.members.remove(region)
    }

    /// Check if group accepts a message
    pub fn accepts_message(&self, message: &BroadcastMessage) -> bool {
        if !self.active {
            return false;
        }
        if message.priority < self.min_priority {
            return false;
        }
        if self.subscribed_types.is_empty() {
            return true; // Accept all types if none specified
        }
        self.subscribed_types.contains(&message.message_type)
    }

    /// Get member count
    pub fn member_count(&self) -> usize {
        self.members.len()
    }

    /// Calculate success rate
    pub fn success_rate(&self) -> f64 {
        let total = self.broadcasts_sent;
        if total == 0 {
            return 1.0;
        }
        (total - self.broadcasts_failed) as f64 / total as f64
    }
}

/// Broadcast result for a single region
#[derive(Debug, Clone)]
pub struct RegionBroadcastResult {
    /// Region identifier
    pub region: RegionId,
    /// Whether broadcast was successful
    pub success: bool,
    /// Delivery time in microseconds
    pub delivery_time_us: u64,
    /// Whether acknowledgment was received
    pub acknowledged: bool,
    /// Error message if failed
    pub error: Option<String>,
}

/// Broadcast result for a complete broadcast operation
#[derive(Debug, Clone)]
pub struct BroadcastResult {
    /// Message that was broadcast
    pub message_id: String,
    /// Groups that received the broadcast
    pub groups: Vec<BroadcastGroupId>,
    /// Per-region results
    pub region_results: Vec<RegionBroadcastResult>,
    /// Total broadcast time in microseconds
    pub total_time_us: u64,
    /// Number of successful deliveries
    pub successful_deliveries: usize,
    /// Number of failed deliveries
    pub failed_deliveries: usize,
    /// Whether overall broadcast met requirements
    pub success: bool,
}

impl BroadcastResult {
    /// Get success rate
    pub fn success_rate(&self) -> f64 {
        let total = self.successful_deliveries + self.failed_deliveries;
        if total == 0 {
            return 1.0;
        }
        self.successful_deliveries as f64 / total as f64
    }

    /// Get regions that failed
    pub fn failed_regions(&self) -> Vec<&RegionId> {
        self.region_results
            .iter()
            .filter(|r| !r.success)
            .map(|r| &r.region)
            .collect()
    }

    /// Get regions that succeeded
    pub fn successful_regions(&self) -> Vec<&RegionId> {
        self.region_results
            .iter()
            .filter(|r| r.success)
            .map(|r| &r.region)
            .collect()
    }
}

/// Broadcast statistics
#[derive(Debug, Clone, Default)]
pub struct BroadcastStats {
    /// Total broadcasts initiated
    pub total_broadcasts: u64,
    /// Successful broadcasts
    pub successful_broadcasts: u64,
    /// Failed broadcasts
    pub failed_broadcasts: u64,
    /// Total messages delivered
    pub total_deliveries: u64,
    /// Failed deliveries
    pub failed_deliveries: u64,
    /// Average broadcast time in microseconds
    pub avg_broadcast_time_us: f64,
    /// Broadcasts by message type
    pub by_message_type: HashMap<String, u64>,
    /// Broadcasts by group
    pub by_group: HashMap<String, u64>,
    /// Messages dropped due to expiry
    pub messages_expired: u64,
    /// Active broadcast groups
    pub active_groups: usize,
}

/// Broadcast configuration
#[derive(Debug, Clone)]
pub struct BroadcastConfig {
    /// Maximum concurrent broadcasts
    pub max_concurrent: usize,
    /// Default retry count for at-least-once delivery
    pub default_retries: u32,
    /// Retry delay in milliseconds
    pub retry_delay_ms: u64,
    /// Enable deduplication for exactly-once delivery
    pub enable_deduplication: bool,
    /// Deduplication window in milliseconds
    pub dedup_window_ms: u64,
    /// Enable broadcast statistics
    pub enable_stats: bool,
}

impl Default for BroadcastConfig {
    fn default() -> Self {
        Self {
            max_concurrent: 100,
            default_retries: 3,
            retry_delay_ms: 100,
            enable_deduplication: true,
            dedup_window_ms: 60000, // 1 minute
            enable_stats: true,
        }
    }
}

/// Broadcast system for multi-region signal distribution
pub struct Broadcast {
    /// Configuration
    config: BroadcastConfig,
    /// Broadcast groups
    groups: Arc<RwLock<HashMap<BroadcastGroupId, BroadcastGroup>>>,
    /// Message type to groups mapping
    type_groups: Arc<RwLock<HashMap<String, Vec<BroadcastGroupId>>>>,
    /// Statistics
    stats: Arc<RwLock<BroadcastStats>>,
    /// Whether system is active
    active: Arc<RwLock<bool>>,
    /// Seen message IDs for deduplication
    seen_messages: Arc<RwLock<HashMap<String, u64>>>,
    /// Default groups for untyped messages
    default_groups: Arc<RwLock<Vec<BroadcastGroupId>>>,
}

impl Default for Broadcast {
    fn default() -> Self {
        Self::new()
    }
}

impl Broadcast {
    /// Create a new broadcast system
    pub fn new() -> Self {
        Self::with_config(BroadcastConfig::default())
    }

    /// Create with custom configuration
    pub fn with_config(config: BroadcastConfig) -> Self {
        Self {
            config,
            groups: Arc::new(RwLock::new(HashMap::new())),
            type_groups: Arc::new(RwLock::new(HashMap::new())),
            stats: Arc::new(RwLock::new(BroadcastStats::default())),
            active: Arc::new(RwLock::new(true)),
            seen_messages: Arc::new(RwLock::new(HashMap::new())),
            default_groups: Arc::new(RwLock::new(Vec::new())),
        }
    }

    /// Register a broadcast group
    pub async fn register_group(&self, group: BroadcastGroup) {
        let group_id = group.id.clone();

        // Update type mappings
        {
            let mut type_map = self.type_groups.write().await;
            for msg_type in &group.subscribed_types {
                type_map
                    .entry(msg_type.clone())
                    .or_insert_with(Vec::new)
                    .push(group_id.clone());
            }
        }

        // Store group
        {
            let mut groups = self.groups.write().await;
            groups.insert(group_id, group);
        }

        // Update stats
        self.update_group_count().await;
    }

    /// Unregister a broadcast group
    pub async fn unregister_group(&self, group_id: &str) -> Option<BroadcastGroup> {
        let group = {
            let mut groups = self.groups.write().await;
            groups.remove(group_id)
        };

        if let Some(ref g) = group {
            // Clean up type mappings
            let mut type_map = self.type_groups.write().await;
            for msg_type in &g.subscribed_types {
                if let Some(ids) = type_map.get_mut(msg_type) {
                    ids.retain(|id| id != group_id);
                }
            }
        }

        self.update_group_count().await;
        group
    }

    /// Set default groups for untyped messages
    pub async fn set_default_groups(&self, group_ids: Vec<BroadcastGroupId>) {
        *self.default_groups.write().await = group_ids;
    }

    /// Broadcast a message to all applicable groups
    pub async fn broadcast(&self, message: BroadcastMessage) -> Result<BroadcastResult> {
        let start = Instant::now();

        if !*self.active.read().await {
            return Err(anyhow::anyhow!("Broadcast system is not active").into());
        }

        // Check for expiry
        if message.is_expired() {
            let mut stats = self.stats.write().await;
            stats.messages_expired += 1;
            return Err(anyhow::anyhow!("Message has expired").into());
        }

        // Check for duplicates if exactly-once delivery
        if message.delivery_mode == DeliveryMode::ExactlyOnce && self.config.enable_deduplication {
            if self.is_duplicate(&message.id).await {
                return Err(anyhow::anyhow!("Duplicate message detected").into());
            }
            self.mark_seen(&message.id).await;
        }

        let mut result = BroadcastResult {
            message_id: message.id.clone(),
            groups: Vec::new(),
            region_results: Vec::new(),
            total_time_us: 0,
            successful_deliveries: 0,
            failed_deliveries: 0,
            success: false,
        };

        // Find applicable groups
        let group_ids = self.find_groups(&message).await;

        if group_ids.is_empty() {
            result.total_time_us = start.elapsed().as_micros() as u64;
            self.update_stats(&message, &result).await;
            return Ok(result);
        }

        // Broadcast to each group
        let mut groups = self.groups.write().await;
        for group_id in group_ids {
            if let Some(group) = groups.get_mut(&group_id) {
                if !group.accepts_message(&message) {
                    continue;
                }

                result.groups.push(group_id.clone());

                // Deliver to each member
                for region in &group.members {
                    let region_start = Instant::now();
                    let region_result = self.deliver_to_region(region, &message).await;

                    let delivery_result = RegionBroadcastResult {
                        region: region.clone(),
                        success: region_result.is_ok(),
                        delivery_time_us: region_start.elapsed().as_micros() as u64,
                        acknowledged: region_result.is_ok(),
                        error: region_result.err().map(|e| e.to_string()),
                    };

                    if delivery_result.success {
                        result.successful_deliveries += 1;
                    } else {
                        result.failed_deliveries += 1;
                    }

                    result.region_results.push(delivery_result);
                }

                group.broadcasts_sent += 1;
                if result.failed_deliveries > 0 {
                    group.broadcasts_failed += 1;
                }
            }
        }

        // Determine overall success based on ack requirements
        result.success = self.check_ack_requirement(&message, &result);
        result.total_time_us = start.elapsed().as_micros() as u64;

        drop(groups);
        self.update_stats(&message, &result).await;

        Ok(result)
    }

    /// Broadcast to specific groups
    pub async fn broadcast_to_groups(
        &self,
        message: BroadcastMessage,
        group_ids: Vec<BroadcastGroupId>,
    ) -> Result<BroadcastResult> {
        let start = Instant::now();

        if !*self.active.read().await {
            return Err(anyhow::anyhow!("Broadcast system is not active").into());
        }

        let mut result = BroadcastResult {
            message_id: message.id.clone(),
            groups: Vec::new(),
            region_results: Vec::new(),
            total_time_us: 0,
            successful_deliveries: 0,
            failed_deliveries: 0,
            success: false,
        };

        let mut groups = self.groups.write().await;
        for group_id in group_ids {
            if let Some(group) = groups.get_mut(&group_id) {
                if !group.active {
                    continue;
                }

                result.groups.push(group_id.clone());

                for region in &group.members {
                    let region_start = Instant::now();
                    let region_result = self.deliver_to_region(region, &message).await;

                    let delivery_result = RegionBroadcastResult {
                        region: region.clone(),
                        success: region_result.is_ok(),
                        delivery_time_us: region_start.elapsed().as_micros() as u64,
                        acknowledged: region_result.is_ok(),
                        error: region_result.err().map(|e| e.to_string()),
                    };

                    if delivery_result.success {
                        result.successful_deliveries += 1;
                    } else {
                        result.failed_deliveries += 1;
                    }

                    result.region_results.push(delivery_result);
                }

                group.broadcasts_sent += 1;
            }
        }

        result.success = self.check_ack_requirement(&message, &result);
        result.total_time_us = start.elapsed().as_micros() as u64;

        drop(groups);
        self.update_stats(&message, &result).await;

        Ok(result)
    }

    /// Find groups applicable for a message
    async fn find_groups(&self, message: &BroadcastMessage) -> Vec<BroadcastGroupId> {
        let mut group_ids = Vec::new();

        // Check type-based groups
        {
            let type_map = self.type_groups.read().await;
            if let Some(ids) = type_map.get(&message.message_type) {
                group_ids.extend(ids.clone());
            }
        }

        // If no specific groups, use defaults
        if group_ids.is_empty() {
            let defaults = self.default_groups.read().await;
            group_ids = defaults.clone();
        }

        group_ids
    }

    /// Deliver message to a specific region
    async fn deliver_to_region(&self, _region: &str, _message: &BroadcastMessage) -> Result<()> {
        // In a real implementation, this would send to the actual region handler
        // For now, simulate successful delivery
        tokio::time::sleep(tokio::time::Duration::from_micros(10)).await;
        Ok(())
    }

    /// Check if acknowledgment requirement is met
    fn check_ack_requirement(&self, message: &BroadcastMessage, result: &BroadcastResult) -> bool {
        let total = result.successful_deliveries + result.failed_deliveries;
        if total == 0 {
            return true;
        }

        match message.ack_requirement {
            AckRequirement::None => true,
            AckRequirement::Any => result.successful_deliveries > 0,
            AckRequirement::Majority => result.successful_deliveries > total / 2,
            AckRequirement::All => result.failed_deliveries == 0,
        }
    }

    /// Check if a message ID has been seen (for deduplication)
    async fn is_duplicate(&self, message_id: &str) -> bool {
        let seen = self.seen_messages.read().await;
        if let Some(timestamp) = seen.get(message_id) {
            let now = std::time::SystemTime::now()
                .duration_since(std::time::UNIX_EPOCH)
                .unwrap_or_default()
                .as_millis() as u64;
            now < timestamp + self.config.dedup_window_ms
        } else {
            false
        }
    }

    /// Mark a message ID as seen
    async fn mark_seen(&self, message_id: &str) {
        let now = std::time::SystemTime::now()
            .duration_since(std::time::UNIX_EPOCH)
            .unwrap_or_default()
            .as_millis() as u64;
        let mut seen = self.seen_messages.write().await;
        seen.insert(message_id.to_string(), now);

        // Clean up old entries
        let cutoff = now.saturating_sub(self.config.dedup_window_ms);
        seen.retain(|_, ts| *ts > cutoff);
    }

    /// Update statistics after broadcast
    async fn update_stats(&self, message: &BroadcastMessage, result: &BroadcastResult) {
        if !self.config.enable_stats {
            return;
        }

        let mut stats = self.stats.write().await;

        stats.total_broadcasts += 1;
        if result.success {
            stats.successful_broadcasts += 1;
        } else {
            stats.failed_broadcasts += 1;
        }

        stats.total_deliveries += result.successful_deliveries as u64;
        stats.failed_deliveries += result.failed_deliveries as u64;

        // Update average time (EMA)
        let alpha = 0.1;
        stats.avg_broadcast_time_us =
            stats.avg_broadcast_time_us * (1.0 - alpha) + result.total_time_us as f64 * alpha;

        // Update by message type
        *stats
            .by_message_type
            .entry(message.message_type.clone())
            .or_insert(0) += 1;

        // Update by group
        for group_id in &result.groups {
            *stats.by_group.entry(group_id.clone()).or_insert(0) += 1;
        }
    }

    /// Update active group count in stats
    async fn update_group_count(&self) {
        let groups = self.groups.read().await;
        let mut stats = self.stats.write().await;
        stats.active_groups = groups.values().filter(|g| g.active).count();
    }

    /// Get a broadcast group by ID
    pub async fn get_group(&self, group_id: &str) -> Option<BroadcastGroup> {
        self.groups.read().await.get(group_id).cloned()
    }

    /// Get all broadcast groups
    pub async fn get_all_groups(&self) -> Vec<BroadcastGroup> {
        self.groups.read().await.values().cloned().collect()
    }

    /// Add a member to a group
    pub async fn add_member_to_group(&self, group_id: &str, region: &str) -> Result<()> {
        let mut groups = self.groups.write().await;
        if let Some(group) = groups.get_mut(group_id) {
            group.add_member(region);
            Ok(())
        } else {
            Err(anyhow::anyhow!("Group not found: {}", group_id).into())
        }
    }

    /// Remove a member from a group
    pub async fn remove_member_from_group(&self, group_id: &str, region: &str) -> Result<bool> {
        let mut groups = self.groups.write().await;
        if let Some(group) = groups.get_mut(group_id) {
            Ok(group.remove_member(region))
        } else {
            Err(anyhow::anyhow!("Group not found: {}", group_id).into())
        }
    }

    /// Set group active state
    pub async fn set_group_active(&self, group_id: &str, active: bool) -> Result<()> {
        let mut groups = self.groups.write().await;
        if let Some(group) = groups.get_mut(group_id) {
            group.active = active;
            Ok(())
        } else {
            Err(anyhow::anyhow!("Group not found: {}", group_id).into())
        }
    }

    /// Get statistics
    pub async fn stats(&self) -> BroadcastStats {
        self.update_group_count().await;
        self.stats.read().await.clone()
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
    fn test_broadcast_message_creation() {
        let msg = BroadcastMessage::new("msg_001", "price_update", vec![1, 2, 3])
            .from_source("market_data")
            .with_priority(8)
            .with_ttl(5000);

        assert_eq!(msg.id, "msg_001");
        assert_eq!(msg.message_type, "price_update");
        assert_eq!(msg.priority, 8);
        assert_eq!(msg.ttl_ms, 5000);
    }

    #[test]
    fn test_message_expiry() {
        let msg = BroadcastMessage::new("msg", "type", vec![]).with_ttl(0);
        assert!(!msg.is_expired());

        // Can't easily test expiry without sleeping, so we test the logic
        let msg_with_old_timestamp = BroadcastMessage {
            id: "old".to_string(),
            message_type: "type".to_string(),
            payload: vec![],
            source: String::new(),
            priority: 5,
            created_at: 0, // Very old timestamp
            ttl_ms: 1000,
            delivery_mode: DeliveryMode::default(),
            ack_requirement: AckRequirement::default(),
            metadata: HashMap::new(),
        };
        assert!(msg_with_old_timestamp.is_expired());
    }

    #[test]
    fn test_broadcast_group_creation() {
        let group = BroadcastGroup::new("risk_group", "Risk Alert Group")
            .with_members(vec!["amygdala", "prefrontal", "hypothalamus"])
            .with_subscriptions(vec!["risk", "threat", "anomaly"])
            .with_min_priority(7);

        assert_eq!(group.member_count(), 3);
        assert!(group.subscribed_types.contains("risk"));
        assert_eq!(group.min_priority, 7);
    }

    #[test]
    fn test_group_accepts_message() {
        let group = BroadcastGroup::new("test", "Test")
            .with_subscriptions(vec!["price"])
            .with_min_priority(5);

        let valid_msg = BroadcastMessage::new("1", "price", vec![]).with_priority(6);
        assert!(group.accepts_message(&valid_msg));

        let wrong_type = BroadcastMessage::new("2", "volume", vec![]).with_priority(6);
        assert!(!group.accepts_message(&wrong_type));

        let low_priority = BroadcastMessage::new("3", "price", vec![]).with_priority(3);
        assert!(!group.accepts_message(&low_priority));
    }

    #[tokio::test]
    async fn test_register_group() {
        let broadcast = Broadcast::new();

        let group = BroadcastGroup::new("test_group", "Test Group")
            .with_members(vec!["region1", "region2"]);

        broadcast.register_group(group).await;

        let retrieved = broadcast.get_group("test_group").await;
        assert!(retrieved.is_some());
        assert_eq!(retrieved.unwrap().member_count(), 2);
    }

    #[tokio::test]
    async fn test_broadcast_message() {
        let broadcast = Broadcast::new();

        let group = BroadcastGroup::new("alert_group", "Alert Group")
            .with_members(vec!["amygdala", "hippocampus"])
            .with_subscriptions(vec!["alert"]);

        broadcast.register_group(group).await;

        let msg = BroadcastMessage::new("alert_001", "alert", vec![1, 2, 3]);

        let result = broadcast.broadcast(msg).await.unwrap();

        assert!(result.success);
        assert_eq!(result.successful_deliveries, 2);
        assert_eq!(result.failed_deliveries, 0);
    }

    #[tokio::test]
    async fn test_broadcast_to_specific_groups() {
        let broadcast = Broadcast::new();

        let group1 = BroadcastGroup::new("group1", "Group 1").with_member("region1");
        let group2 = BroadcastGroup::new("group2", "Group 2").with_member("region2");

        broadcast.register_group(group1).await;
        broadcast.register_group(group2).await;

        let msg = BroadcastMessage::new("msg", "type", vec![]);

        let result = broadcast
            .broadcast_to_groups(msg, vec!["group1".to_string()])
            .await
            .unwrap();

        assert!(result.success);
        assert_eq!(result.groups.len(), 1);
        assert!(result.groups.contains(&"group1".to_string()));
    }

    #[tokio::test]
    async fn test_ack_requirements() {
        let broadcast = Broadcast::new();

        let group = BroadcastGroup::new("ack_group", "Ack Group")
            .with_members(vec!["r1", "r2", "r3"])
            .with_subscriptions(vec!["test"]);

        broadcast.register_group(group).await;

        // Test different ack requirements
        let msg_any =
            BroadcastMessage::new("1", "test", vec![]).with_ack_requirement(AckRequirement::Any);

        let result = broadcast.broadcast(msg_any).await.unwrap();
        assert!(result.success);

        let msg_all =
            BroadcastMessage::new("2", "test", vec![]).with_ack_requirement(AckRequirement::All);

        let result = broadcast.broadcast(msg_all).await.unwrap();
        assert!(result.success);
    }

    #[tokio::test]
    async fn test_default_groups() {
        let broadcast = Broadcast::new();

        let default_group =
            BroadcastGroup::new("default", "Default Group").with_member("fallback_region");

        broadcast.register_group(default_group).await;
        broadcast
            .set_default_groups(vec!["default".to_string()])
            .await;

        // Message with unknown type should use default group
        let msg = BroadcastMessage::new("unknown_msg", "unknown_type", vec![]);

        let result = broadcast.broadcast(msg).await.unwrap();
        assert!(result.success);
        assert!(result.groups.contains(&"default".to_string()));
    }

    #[tokio::test]
    async fn test_statistics() {
        let broadcast = Broadcast::new();

        let group = BroadcastGroup::new("stats_group", "Stats Group")
            .with_member("region")
            .with_subscriptions(vec!["metric"]);

        broadcast.register_group(group).await;

        for i in 0..5 {
            let msg = BroadcastMessage::new(format!("msg_{}", i), "metric", vec![]);
            broadcast.broadcast(msg).await.unwrap();
        }

        let stats = broadcast.stats().await;
        assert_eq!(stats.total_broadcasts, 5);
        assert_eq!(stats.successful_broadcasts, 5);
        assert_eq!(stats.total_deliveries, 5);
    }

    #[tokio::test]
    async fn test_member_management() {
        let broadcast = Broadcast::new();

        let group = BroadcastGroup::new("managed", "Managed Group").with_member("initial");

        broadcast.register_group(group).await;

        // Add member
        broadcast
            .add_member_to_group("managed", "new_member")
            .await
            .unwrap();

        let group = broadcast.get_group("managed").await.unwrap();
        assert_eq!(group.member_count(), 2);

        // Remove member
        broadcast
            .remove_member_from_group("managed", "initial")
            .await
            .unwrap();

        let group = broadcast.get_group("managed").await.unwrap();
        assert_eq!(group.member_count(), 1);
    }

    #[tokio::test]
    async fn test_group_activation() {
        let broadcast = Broadcast::new();

        let group = BroadcastGroup::new("toggleable", "Toggleable Group")
            .with_member("region")
            .with_subscriptions(vec!["toggle"]);

        broadcast.register_group(group).await;

        // Deactivate group
        broadcast
            .set_group_active("toggleable", false)
            .await
            .unwrap();

        let msg = BroadcastMessage::new("msg", "toggle", vec![]);
        let result = broadcast.broadcast(msg).await.unwrap();

        // Should not deliver to deactivated group
        assert_eq!(result.groups.len(), 0);
    }

    #[test]
    fn test_broadcast_result_helpers() {
        let result = BroadcastResult {
            message_id: "test".to_string(),
            groups: vec!["g1".to_string()],
            region_results: vec![
                RegionBroadcastResult {
                    region: "r1".to_string(),
                    success: true,
                    delivery_time_us: 100,
                    acknowledged: true,
                    error: None,
                },
                RegionBroadcastResult {
                    region: "r2".to_string(),
                    success: false,
                    delivery_time_us: 50,
                    acknowledged: false,
                    error: Some("timeout".to_string()),
                },
            ],
            total_time_us: 150,
            successful_deliveries: 1,
            failed_deliveries: 1,
            success: false,
        };

        assert!((result.success_rate() - 0.5).abs() < 0.001);
        assert_eq!(result.failed_regions().len(), 1);
        assert_eq!(result.successful_regions().len(), 1);
    }

    #[test]
    fn test_process_compatibility() {
        let broadcast = Broadcast::new();
        assert!(broadcast.process().is_ok());
    }
}
