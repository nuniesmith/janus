//! Route priority management
//!
//! Part of the Thalamus region - manages priority queues and dynamic
//! priority adjustment for signal routing.

use crate::common::Result;
use std::collections::{BinaryHeap, HashMap};
use std::sync::Arc;
use std::time::{Duration, Instant};
use tokio::sync::RwLock;

/// Priority level enumeration
#[derive(Debug, Clone, Copy, PartialEq, Eq, PartialOrd, Ord, Hash)]
pub enum PriorityLevel {
    /// Background processing - can be significantly delayed
    Background = 0,
    /// Low priority - process when resources available
    Low = 1,
    /// Normal priority - standard processing
    Normal = 2,
    /// High priority - prioritize over normal
    High = 3,
    /// Critical priority - process immediately
    Critical = 4,
    /// Emergency - bypass queues if possible
    Emergency = 5,
}

impl Default for PriorityLevel {
    fn default() -> Self {
        Self::Normal
    }
}

impl PriorityLevel {
    /// Get numeric value for calculations
    pub fn value(&self) -> u8 {
        *self as u8
    }

    /// Create from numeric value
    pub fn from_value(value: u8) -> Self {
        match value {
            0 => Self::Background,
            1 => Self::Low,
            2 => Self::Normal,
            3 => Self::High,
            4 => Self::Critical,
            _ => Self::Emergency,
        }
    }

    /// Get base delay multiplier for this priority
    pub fn delay_multiplier(&self) -> f64 {
        match self {
            Self::Background => 4.0,
            Self::Low => 2.0,
            Self::Normal => 1.0,
            Self::High => 0.5,
            Self::Critical => 0.1,
            Self::Emergency => 0.0,
        }
    }
}

/// Priority boost factors based on signal characteristics
#[derive(Debug, Clone, Default)]
pub struct PriorityBoost {
    /// Boost from signal strength
    pub strength_boost: f64,
    /// Boost from signal confidence
    pub confidence_boost: f64,
    /// Boost from market volatility
    pub volatility_boost: f64,
    /// Boost from signal age (aging priority)
    pub age_boost: f64,
    /// Custom boost factor
    pub custom_boost: f64,
}

impl PriorityBoost {
    /// Calculate total boost
    pub fn total(&self) -> f64 {
        self.strength_boost
            + self.confidence_boost
            + self.volatility_boost
            + self.age_boost
            + self.custom_boost
    }

    /// Create from signal characteristics
    pub fn from_signal_characteristics(
        strength: f64,
        confidence: f64,
        volatility: f64,
        age_secs: f64,
    ) -> Self {
        Self {
            strength_boost: (strength - 0.5).max(0.0) * 0.5,
            confidence_boost: (confidence - 0.5).max(0.0) * 0.3,
            volatility_boost: (volatility - 0.2).max(0.0) * 0.4,
            age_boost: (age_secs / 10.0).min(0.3), // Cap age boost
            custom_boost: 0.0,
        }
    }
}

/// Queued item with priority
#[derive(Debug, Clone)]
pub struct PrioritizedItem<T: Clone> {
    /// The item
    pub item: T,
    /// Base priority level
    pub base_priority: PriorityLevel,
    /// Effective priority (with boosts)
    pub effective_priority: f64,
    /// When the item was queued
    pub queued_at: Instant,
    /// Priority boost factors
    pub boost: PriorityBoost,
    /// Item identifier
    pub id: String,
}

impl<T: Clone> PartialEq for PrioritizedItem<T> {
    fn eq(&self, other: &Self) -> bool {
        self.id == other.id
    }
}

impl<T: Clone> Eq for PrioritizedItem<T> {}

impl<T: Clone> PartialOrd for PrioritizedItem<T> {
    fn partial_cmp(&self, other: &Self) -> Option<std::cmp::Ordering> {
        Some(self.cmp(other))
    }
}

impl<T: Clone> Ord for PrioritizedItem<T> {
    fn cmp(&self, other: &Self) -> std::cmp::Ordering {
        // Higher effective priority comes first
        self.effective_priority
            .partial_cmp(&other.effective_priority)
            .unwrap_or(std::cmp::Ordering::Equal)
    }
}

impl<T: Clone> PrioritizedItem<T> {
    /// Create a new prioritized item
    pub fn new(item: T, id: impl Into<String>, priority: PriorityLevel) -> Self {
        Self {
            item,
            base_priority: priority,
            effective_priority: priority.value() as f64,
            queued_at: Instant::now(),
            boost: PriorityBoost::default(),
            id: id.into(),
        }
    }

    /// Apply priority boost
    pub fn with_boost(mut self, boost: PriorityBoost) -> Self {
        self.boost = boost;
        self.recalculate_priority();
        self
    }

    /// Recalculate effective priority
    pub fn recalculate_priority(&mut self) {
        let base = self.base_priority.value() as f64;
        let boost_total = self.boost.total();
        self.effective_priority = (base + boost_total).clamp(0.0, 6.0);
    }

    /// Update age boost based on time in queue
    pub fn update_age_boost(&mut self, max_age_secs: f64) {
        let age_secs = self.queued_at.elapsed().as_secs_f64();
        self.boost.age_boost = (age_secs / max_age_secs).min(0.5);
        self.recalculate_priority();
    }

    /// Get time in queue
    pub fn queue_time(&self) -> Duration {
        self.queued_at.elapsed()
    }
}

/// Priority queue configuration
#[derive(Debug, Clone)]
pub struct PriorityConfig {
    /// Maximum queue size per priority level
    pub max_queue_size: usize,
    /// Enable age-based priority boosting
    pub enable_age_boost: bool,
    /// Maximum age before forced processing (seconds)
    pub max_age_secs: f64,
    /// Priority update interval (milliseconds)
    pub update_interval_ms: u64,
    /// Enable fair scheduling between priority levels
    pub fair_scheduling: bool,
    /// Fair scheduling weight for lower priorities
    pub fairness_weight: f64,
}

impl Default for PriorityConfig {
    fn default() -> Self {
        Self {
            max_queue_size: 10000,
            enable_age_boost: true,
            max_age_secs: 30.0,
            update_interval_ms: 100,
            fair_scheduling: true,
            fairness_weight: 0.1,
        }
    }
}

/// Priority queue statistics
#[derive(Debug, Clone, Default)]
pub struct PriorityStats {
    /// Items processed by priority level
    pub processed_by_priority: HashMap<u8, u64>,
    /// Items dropped by priority level
    pub dropped_by_priority: HashMap<u8, u64>,
    /// Average queue time by priority (milliseconds)
    pub avg_queue_time_ms: HashMap<u8, f64>,
    /// Current queue depths by priority
    pub queue_depths: HashMap<u8, usize>,
    /// Total items processed
    pub total_processed: u64,
    /// Total items dropped
    pub total_dropped: u64,
    /// Priority boosts applied
    pub boosts_applied: u64,
}

/// Route priority management system
pub struct Priority {
    /// Configuration
    config: PriorityConfig,
    /// Priority queues by level
    queues: Arc<RwLock<HashMap<u8, BinaryHeap<PrioritizedItem<Vec<u8>>>>>>,
    /// Statistics
    stats: Arc<RwLock<PriorityStats>>,
    /// Whether system is active
    active: Arc<RwLock<bool>>,
    /// Last update timestamp
    last_update: Arc<RwLock<Instant>>,
    /// Priority adjustments by signal type
    type_adjustments: Arc<RwLock<HashMap<String, i8>>>,
}

impl Default for Priority {
    fn default() -> Self {
        Self::new()
    }
}

impl Priority {
    /// Create a new priority manager
    pub fn new() -> Self {
        Self::with_config(PriorityConfig::default())
    }

    /// Create with custom configuration
    pub fn with_config(config: PriorityConfig) -> Self {
        let mut queues = HashMap::new();
        for i in 0..=5 {
            queues.insert(i, BinaryHeap::new());
        }

        Self {
            config,
            queues: Arc::new(RwLock::new(queues)),
            stats: Arc::new(RwLock::new(PriorityStats::default())),
            active: Arc::new(RwLock::new(true)),
            last_update: Arc::new(RwLock::new(Instant::now())),
            type_adjustments: Arc::new(RwLock::new(HashMap::new())),
        }
    }

    /// Enqueue an item with priority
    pub async fn enqueue(
        &self,
        id: impl Into<String>,
        data: Vec<u8>,
        priority: PriorityLevel,
        boost: Option<PriorityBoost>,
    ) -> Result<()> {
        let id = id.into();
        let priority_value = priority.value();

        let mut queues = self.queues.write().await;
        let queue = queues.entry(priority_value).or_insert_with(BinaryHeap::new);

        // Check queue capacity
        if queue.len() >= self.config.max_queue_size {
            let mut stats = self.stats.write().await;
            *stats.dropped_by_priority.entry(priority_value).or_insert(0) += 1;
            stats.total_dropped += 1;
            return Err(anyhow::anyhow!("Queue full for priority level {}", priority_value).into());
        }

        let mut item = PrioritizedItem::new(data, id, priority);
        if let Some(b) = boost {
            item = item.with_boost(b);
            let mut stats = self.stats.write().await;
            stats.boosts_applied += 1;
        }

        queue.push(item);

        // Update queue depth stats
        let mut stats = self.stats.write().await;
        stats.queue_depths.insert(priority_value, queue.len());

        Ok(())
    }

    /// Dequeue the highest priority item
    pub async fn dequeue(&self) -> Option<PrioritizedItem<Vec<u8>>> {
        let mut queues = self.queues.write().await;

        // Find the item with highest effective_priority across all queues
        let mut best_priority_level: Option<u8> = None;
        let mut best_effective_priority: f64 = f64::NEG_INFINITY;

        for (&priority_level, queue) in queues.iter() {
            if let Some(item) = queue.peek() {
                if item.effective_priority > best_effective_priority {
                    best_effective_priority = item.effective_priority;
                    best_priority_level = Some(priority_level);
                }
            }
        }

        // Pop from the queue with highest effective priority
        if let Some(priority) = best_priority_level {
            if let Some(queue) = queues.get_mut(&priority) {
                if let Some(item) = queue.pop() {
                    // Update statistics
                    let queue_time_ms = item.queue_time().as_millis() as f64;

                    drop(queues); // Release lock before getting stats lock

                    let mut stats = self.stats.write().await;
                    *stats.processed_by_priority.entry(priority).or_insert(0) += 1;
                    stats.total_processed += 1;

                    // Update average queue time (EMA)
                    let current_avg = *stats.avg_queue_time_ms.get(&priority).unwrap_or(&0.0);
                    let new_avg = current_avg * 0.9 + queue_time_ms * 0.1;
                    stats.avg_queue_time_ms.insert(priority, new_avg);

                    return Some(item);
                }
            }
        }

        None
    }

    /// Dequeue with fair scheduling (gives some chance to lower priorities)
    pub async fn dequeue_fair(&self) -> Option<PrioritizedItem<Vec<u8>>> {
        if !self.config.fair_scheduling {
            return self.dequeue().await;
        }

        let mut queues = self.queues.write().await;

        // Calculate weighted selection
        let mut weighted_priorities: Vec<(u8, usize)> = Vec::new();
        for priority in 0..=5 {
            if let Some(queue) = queues.get(&priority) {
                if !queue.is_empty() {
                    // Weight = priority_value + fairness_weight * queue_size
                    let weight = priority as usize * 10
                        + (self.config.fairness_weight * queue.len() as f64) as usize;
                    weighted_priorities.push((priority, weight));
                }
            }
        }

        if weighted_priorities.is_empty() {
            return None;
        }

        // Select based on weighted probability (simplified: highest weighted wins)
        weighted_priorities.sort_by(|a, b| b.1.cmp(&a.1));
        let selected_priority = weighted_priorities[0].0;

        if let Some(queue) = queues.get_mut(&selected_priority) {
            if let Some(item) = queue.pop() {
                let queue_time_ms = item.queue_time().as_millis() as f64;

                drop(queues);

                let mut stats = self.stats.write().await;
                *stats
                    .processed_by_priority
                    .entry(selected_priority)
                    .or_insert(0) += 1;
                stats.total_processed += 1;

                let current_avg = *stats
                    .avg_queue_time_ms
                    .get(&selected_priority)
                    .unwrap_or(&0.0);
                let new_avg = current_avg * 0.9 + queue_time_ms * 0.1;
                stats.avg_queue_time_ms.insert(selected_priority, new_avg);

                return Some(item);
            }
        }

        None
    }

    /// Update age boosts for all queued items
    pub async fn update_age_boosts(&self) {
        if !self.config.enable_age_boost {
            return;
        }

        let mut queues = self.queues.write().await;

        for queue in queues.values_mut() {
            let items: Vec<_> = std::mem::take(queue).into_vec();
            let updated: Vec<_> = items
                .into_iter()
                .map(|mut item| {
                    item.update_age_boost(self.config.max_age_secs);
                    item
                })
                .collect();
            *queue = BinaryHeap::from(updated);
        }

        *self.last_update.write().await = Instant::now();
    }

    /// Set priority adjustment for a signal type
    pub async fn set_type_adjustment(&self, signal_type: impl Into<String>, adjustment: i8) {
        let mut adjustments = self.type_adjustments.write().await;
        adjustments.insert(signal_type.into(), adjustment);
    }

    /// Get adjusted priority for a signal type
    pub async fn get_adjusted_priority(
        &self,
        base_priority: PriorityLevel,
        signal_type: &str,
    ) -> PriorityLevel {
        let adjustments = self.type_adjustments.read().await;
        let adjustment = adjustments.get(signal_type).copied().unwrap_or(0);

        let new_value = (base_priority.value() as i8 + adjustment).clamp(0, 5) as u8;
        PriorityLevel::from_value(new_value)
    }

    /// Get total queue depth across all priorities
    pub async fn total_queue_depth(&self) -> usize {
        let queues = self.queues.read().await;
        queues.values().map(|q| q.len()).sum()
    }

    /// Get queue depth for a specific priority
    pub async fn queue_depth(&self, priority: PriorityLevel) -> usize {
        let queues = self.queues.read().await;
        queues.get(&priority.value()).map(|q| q.len()).unwrap_or(0)
    }

    /// Get statistics
    pub async fn stats(&self) -> PriorityStats {
        // Update queue depths before returning
        let queues = self.queues.read().await;
        let mut stats = self.stats.write().await;

        for (priority, queue) in queues.iter() {
            stats.queue_depths.insert(*priority, queue.len());
        }

        stats.clone()
    }

    /// Clear all queues
    pub async fn clear(&self) {
        let mut queues = self.queues.write().await;
        for queue in queues.values_mut() {
            queue.clear();
        }
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
    fn test_priority_level_ordering() {
        assert!(PriorityLevel::Emergency > PriorityLevel::Critical);
        assert!(PriorityLevel::Critical > PriorityLevel::High);
        assert!(PriorityLevel::High > PriorityLevel::Normal);
        assert!(PriorityLevel::Normal > PriorityLevel::Low);
        assert!(PriorityLevel::Low > PriorityLevel::Background);
    }

    #[test]
    fn test_priority_level_from_value() {
        assert_eq!(PriorityLevel::from_value(0), PriorityLevel::Background);
        assert_eq!(PriorityLevel::from_value(3), PriorityLevel::High);
        assert_eq!(PriorityLevel::from_value(10), PriorityLevel::Emergency);
    }

    #[test]
    fn test_priority_boost() {
        let boost = PriorityBoost::from_signal_characteristics(0.8, 0.9, 0.3, 5.0);

        assert!(boost.strength_boost > 0.0);
        assert!(boost.confidence_boost > 0.0);
        assert!(boost.volatility_boost > 0.0);
        assert!(boost.age_boost > 0.0);
        assert!(boost.total() > 0.0);
    }

    #[tokio::test]
    async fn test_enqueue_dequeue() {
        let priority = Priority::new();

        priority
            .enqueue("item1", vec![1, 2, 3], PriorityLevel::Normal, None)
            .await
            .unwrap();
        priority
            .enqueue("item2", vec![4, 5, 6], PriorityLevel::High, None)
            .await
            .unwrap();

        // High priority should come first
        let item = priority.dequeue().await.unwrap();
        assert_eq!(item.id, "item2");

        let item = priority.dequeue().await.unwrap();
        assert_eq!(item.id, "item1");

        assert!(priority.dequeue().await.is_none());
    }

    #[tokio::test]
    async fn test_priority_boost_effect() {
        let priority = Priority::new();

        // Low priority with high boost
        let boost = PriorityBoost {
            strength_boost: 2.0,
            confidence_boost: 1.0,
            volatility_boost: 0.5,
            age_boost: 0.0,
            custom_boost: 0.0,
        };

        priority
            .enqueue("boosted", vec![1], PriorityLevel::Low, Some(boost))
            .await
            .unwrap();
        priority
            .enqueue("normal", vec![2], PriorityLevel::Normal, None)
            .await
            .unwrap();

        // Boosted item should come first even though base priority is lower
        let item = priority.dequeue().await.unwrap();
        assert_eq!(item.id, "boosted");
    }

    #[tokio::test]
    async fn test_queue_capacity() {
        let config = PriorityConfig {
            max_queue_size: 2,
            ..Default::default()
        };
        let priority = Priority::with_config(config);

        priority
            .enqueue("item1", vec![], PriorityLevel::Normal, None)
            .await
            .unwrap();
        priority
            .enqueue("item2", vec![], PriorityLevel::Normal, None)
            .await
            .unwrap();

        // Third item should fail
        let result = priority
            .enqueue("item3", vec![], PriorityLevel::Normal, None)
            .await;
        assert!(result.is_err());
    }

    #[tokio::test]
    async fn test_type_adjustment() {
        let priority = Priority::new();

        priority.set_type_adjustment("risk", 2).await;

        let adjusted = priority
            .get_adjusted_priority(PriorityLevel::Normal, "risk")
            .await;
        assert_eq!(adjusted, PriorityLevel::Critical);

        // Test clamping
        let adjusted = priority
            .get_adjusted_priority(PriorityLevel::Emergency, "risk")
            .await;
        assert_eq!(adjusted, PriorityLevel::Emergency); // Should stay at max
    }

    #[tokio::test]
    async fn test_statistics() {
        let priority = Priority::new();

        for i in 0..5 {
            priority
                .enqueue(format!("item{}", i), vec![], PriorityLevel::Normal, None)
                .await
                .unwrap();
        }

        for _ in 0..3 {
            priority.dequeue().await;
        }

        let stats = priority.stats().await;
        assert_eq!(stats.total_processed, 3);
        assert_eq!(stats.queue_depths.get(&2).copied().unwrap_or(0), 2);
    }

    #[tokio::test]
    async fn test_clear() {
        let priority = Priority::new();

        priority
            .enqueue("item1", vec![], PriorityLevel::Normal, None)
            .await
            .unwrap();
        priority
            .enqueue("item2", vec![], PriorityLevel::High, None)
            .await
            .unwrap();

        assert_eq!(priority.total_queue_depth().await, 2);

        priority.clear().await;

        assert_eq!(priority.total_queue_depth().await, 0);
    }

    #[test]
    fn test_process_compatibility() {
        let priority = Priority::new();
        assert!(priority.process().is_ok());
    }
}
