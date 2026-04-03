//! Cached action patterns (habits)
//!
//! Part of the Basal Ganglia region
//! Component: selection
//!
//! Implements an LRU-like cache for frequently-used action patterns,
//! enabling fast habitual decision-making. When a state-action mapping
//! is used repeatedly with positive outcomes, it becomes a "habit" that
//! can be retrieved in O(1) without running the full deliberative pipeline.
//!
//! Features:
//! - LRU eviction with frequency-weighted retention
//! - Habit strength that grows with repeated successful use
//! - Decay of unused habits over time
//! - Context-dependent habit retrieval (state → action mapping)
//! - Configurable promotion/demotion thresholds

use crate::common::{Error, Result};
use std::collections::{HashMap, VecDeque};

/// A cached habit entry mapping a state pattern to an action
#[derive(Debug, Clone)]
pub struct HabitEntry {
    /// The state key that triggers this habit
    pub state_key: String,
    /// The action index associated with this state
    pub action: usize,
    /// Strength of the habit (0.0 - 1.0), grows with repeated success
    pub strength: f64,
    /// Number of times this habit has been used
    pub use_count: u64,
    /// Number of times using this habit led to a positive outcome
    pub success_count: u64,
    /// Cumulative reward from using this habit
    pub cumulative_reward: f64,
    /// Time step when this habit was last accessed
    pub last_access: u64,
    /// Time step when this habit was created
    pub created_at: u64,
    /// Whether this habit is currently active (above strength threshold)
    pub active: bool,
}

impl HabitEntry {
    /// Create a new habit entry
    fn new(state_key: String, action: usize, created_at: u64) -> Self {
        Self {
            state_key,
            action,
            strength: 0.1, // start weak
            use_count: 0,
            success_count: 0,
            cumulative_reward: 0.0,
            last_access: created_at,
            created_at,
            active: false,
        }
    }

    /// Success rate of this habit (0.0 - 1.0)
    pub fn success_rate(&self) -> f64 {
        if self.use_count == 0 {
            return 0.0;
        }
        self.success_count as f64 / self.use_count as f64
    }

    /// Average reward per use
    pub fn average_reward(&self) -> f64 {
        if self.use_count == 0 {
            return 0.0;
        }
        self.cumulative_reward / self.use_count as f64
    }

    /// Age of the habit in time steps
    pub fn age(&self, current_step: u64) -> u64 {
        current_step.saturating_sub(self.created_at)
    }

    /// Staleness: time steps since last access
    pub fn staleness(&self, current_step: u64) -> u64 {
        current_step.saturating_sub(self.last_access)
    }
}

/// Result of a habit cache lookup
#[derive(Debug, Clone)]
pub enum HabitLookup {
    /// A strong habit was found — use this action directly
    Hit {
        action: usize,
        strength: f64,
        success_rate: f64,
    },
    /// A weak habit exists but isn't strong enough for automatic use
    WeakHit { action: usize, strength: f64 },
    /// No habit exists for this state
    Miss,
}

/// Configuration for the habit cache
#[derive(Debug, Clone)]
pub struct HabitCacheConfig {
    /// Maximum number of habits to store
    pub max_capacity: usize,
    /// Minimum strength for a habit to be considered "active" and returned as a Hit
    pub activation_threshold: f64,
    /// Strength increment on each successful use
    pub reinforcement_rate: f64,
    /// Strength decrement on each failed use
    pub punishment_rate: f64,
    /// Decay rate applied to all habits per tick (0.0 = no decay)
    pub decay_rate: f64,
    /// Minimum strength before a habit is eligible for eviction
    pub eviction_threshold: f64,
    /// Number of successful uses required before a habit can become active
    pub min_successes_to_activate: u64,
    /// Maximum staleness (time steps since last access) before forced eviction
    pub max_staleness: u64,
    /// Whether to use frequency-weighted eviction (vs pure LRU)
    pub frequency_weighted_eviction: bool,
}

impl Default for HabitCacheConfig {
    fn default() -> Self {
        Self {
            max_capacity: 256,
            activation_threshold: 0.5,
            reinforcement_rate: 0.05,
            punishment_rate: 0.10,
            decay_rate: 0.001,
            eviction_threshold: 0.05,
            min_successes_to_activate: 3,
            max_staleness: 10_000,
            frequency_weighted_eviction: true,
        }
    }
}

/// Cache statistics
#[derive(Debug, Clone, Default)]
pub struct HabitCacheStats {
    /// Total lookups performed
    pub total_lookups: u64,
    /// Number of cache hits (strong habits returned)
    pub hits: u64,
    /// Number of weak hits
    pub weak_hits: u64,
    /// Number of cache misses
    pub misses: u64,
    /// Number of habits currently stored
    pub size: usize,
    /// Number of active habits (above activation threshold)
    pub active_count: usize,
    /// Number of evictions performed
    pub evictions: u64,
    /// Number of habits promoted to active
    pub promotions: u64,
    /// Number of habits demoted from active
    pub demotions: u64,
}

impl HabitCacheStats {
    /// Cache hit rate (strong hits / total lookups)
    pub fn hit_rate(&self) -> f64 {
        if self.total_lookups == 0 {
            return 0.0;
        }
        self.hits as f64 / self.total_lookups as f64
    }

    /// Combined hit rate (strong + weak hits / total lookups)
    pub fn combined_hit_rate(&self) -> f64 {
        if self.total_lookups == 0 {
            return 0.0;
        }
        (self.hits + self.weak_hits) as f64 / self.total_lookups as f64
    }
}

/// Cached action patterns (habits) for fast habitual decision-making
pub struct HabitCache {
    /// Configuration parameters
    config: HabitCacheConfig,
    /// Primary storage: state_key → habit entry
    habits: HashMap<String, HabitEntry>,
    /// LRU ordering: most recently used state keys at the back
    lru_order: VecDeque<String>,
    /// Current time step
    current_step: u64,
    /// Accumulated statistics
    stats: HabitCacheStats,
}

impl Default for HabitCache {
    fn default() -> Self {
        Self::new()
    }
}

impl HabitCache {
    /// Create a new instance with default configuration
    pub fn new() -> Self {
        Self::with_config(HabitCacheConfig::default())
    }

    /// Create a new instance with custom configuration
    pub fn with_config(config: HabitCacheConfig) -> Self {
        Self {
            habits: HashMap::with_capacity(config.max_capacity),
            lru_order: VecDeque::with_capacity(config.max_capacity),
            current_step: 0,
            stats: HabitCacheStats::default(),
            config,
        }
    }

    /// Main processing function — validates state and runs maintenance
    pub fn process(&self) -> Result<()> {
        if self.config.max_capacity == 0 {
            return Err(Error::InvalidInput(
                "HabitCache max_capacity must be > 0".into(),
            ));
        }
        Ok(())
    }

    /// Look up a habit for the given state key
    pub fn lookup(&mut self, state_key: &str) -> HabitLookup {
        self.stats.total_lookups += 1;

        if let Some(entry) = self.habits.get_mut(state_key) {
            entry.last_access = self.current_step;

            // Move to back of LRU
            self.lru_order.retain(|k| k != state_key);
            self.lru_order.push_back(state_key.to_string());

            if entry.active && entry.strength >= self.config.activation_threshold {
                self.stats.hits += 1;
                HabitLookup::Hit {
                    action: entry.action,
                    strength: entry.strength,
                    success_rate: entry.success_rate(),
                }
            } else {
                self.stats.weak_hits += 1;
                HabitLookup::WeakHit {
                    action: entry.action,
                    strength: entry.strength,
                }
            }
        } else {
            self.stats.misses += 1;
            HabitLookup::Miss
        }
    }

    /// Record a state-action association. Creates or updates a habit entry.
    ///
    /// Call this when an action is selected (even by the deliberative system)
    /// to begin forming a habit.
    pub fn record(&mut self, state_key: &str, action: usize) {
        if let Some(entry) = self.habits.get_mut(state_key) {
            // Update existing habit
            if entry.action == action {
                entry.use_count += 1;
                entry.last_access = self.current_step;
            } else {
                // Different action for same state — habit conflict
                // If the existing habit is weak, override it
                if entry.strength < self.config.activation_threshold * 0.5 {
                    entry.action = action;
                    entry.strength = 0.1;
                    entry.use_count = 1;
                    entry.success_count = 0;
                    entry.cumulative_reward = 0.0;
                    entry.last_access = self.current_step;
                    entry.active = false;
                }
                // Otherwise, ignore the conflicting action (habit persists)
            }

            // Update LRU order
            self.lru_order.retain(|k| k != state_key);
            self.lru_order.push_back(state_key.to_string());
        } else {
            // Create new habit entry
            self.maybe_evict();

            let entry = HabitEntry::new(state_key.to_string(), action, self.current_step);
            self.habits.insert(state_key.to_string(), entry);
            self.lru_order.push_back(state_key.to_string());
        }
    }

    /// Reinforce a habit after a positive outcome
    pub fn reinforce(&mut self, state_key: &str, reward: f64) {
        if let Some(entry) = self.habits.get_mut(state_key) {
            entry.success_count += 1;
            entry.cumulative_reward += reward;

            // Increase strength, capped at 1.0
            let reward_bonus = (reward.max(0.0) * 0.01).min(0.05);
            entry.strength =
                (entry.strength + self.config.reinforcement_rate + reward_bonus).min(1.0);

            // Check for promotion
            if !entry.active
                && entry.strength >= self.config.activation_threshold
                && entry.success_count >= self.config.min_successes_to_activate
            {
                entry.active = true;
                self.stats.promotions += 1;
            }
        }
    }

    /// Punish a habit after a negative outcome
    pub fn punish(&mut self, state_key: &str, penalty: f64) {
        if let Some(entry) = self.habits.get_mut(state_key) {
            let effective_penalty = self.config.punishment_rate + (penalty.abs() * 0.01).min(0.05);
            entry.strength = (entry.strength - effective_penalty).max(0.0);
            entry.cumulative_reward -= penalty.abs();

            // Check for demotion
            if entry.active && entry.strength < self.config.activation_threshold {
                entry.active = false;
                self.stats.demotions += 1;
            }
        }
    }

    /// Advance one time step — applies decay and evicts stale habits
    pub fn tick(&mut self) {
        self.current_step += 1;

        if self.config.decay_rate > 0.0 {
            let decay = self.config.decay_rate;
            let threshold = self.config.activation_threshold;
            let mut demoted = 0u64;

            for entry in self.habits.values_mut() {
                entry.strength = (entry.strength - decay).max(0.0);

                // Check for demotion due to decay
                if entry.active && entry.strength < threshold {
                    entry.active = false;
                    demoted += 1;
                }
            }

            self.stats.demotions += demoted;
        }

        // Evict stale habits
        self.evict_stale();
    }

    /// Get a reference to a specific habit entry
    pub fn get(&self, state_key: &str) -> Option<&HabitEntry> {
        self.habits.get(state_key)
    }

    /// Get the number of habits currently stored
    pub fn size(&self) -> usize {
        self.habits.len()
    }

    /// Get the number of active habits
    pub fn active_count(&self) -> usize {
        self.habits.values().filter(|h| h.active).count()
    }

    /// Check if the cache is full
    pub fn is_full(&self) -> bool {
        self.habits.len() >= self.config.max_capacity
    }

    /// Get cache statistics
    pub fn stats(&self) -> HabitCacheStats {
        HabitCacheStats {
            size: self.habits.len(),
            active_count: self.active_count(),
            ..self.stats.clone()
        }
    }

    /// Get the current time step
    pub fn current_step(&self) -> u64 {
        self.current_step
    }

    /// Get the top N strongest habits
    pub fn strongest_habits(&self, n: usize) -> Vec<&HabitEntry> {
        let mut entries: Vec<&HabitEntry> = self.habits.values().collect();
        entries.sort_by(|a, b| {
            b.strength
                .partial_cmp(&a.strength)
                .unwrap_or(std::cmp::Ordering::Equal)
        });
        entries.truncate(n);
        entries
    }

    /// Get all active habits
    pub fn active_habits(&self) -> Vec<&HabitEntry> {
        self.habits.values().filter(|h| h.active).collect()
    }

    /// Remove a specific habit
    pub fn remove(&mut self, state_key: &str) -> Option<HabitEntry> {
        self.lru_order.retain(|k| k != state_key);
        self.habits.remove(state_key)
    }

    /// Clear all habits
    pub fn clear(&mut self) {
        self.habits.clear();
        self.lru_order.clear();
    }

    /// Reset everything including statistics
    pub fn reset(&mut self) {
        self.clear();
        self.current_step = 0;
        self.stats = HabitCacheStats::default();
    }

    // ── internal ──

    /// Evict one entry to make room, if at capacity
    fn maybe_evict(&mut self) {
        if self.habits.len() < self.config.max_capacity {
            return;
        }

        let victim_key = if self.config.frequency_weighted_eviction {
            self.find_frequency_weighted_victim()
        } else {
            self.find_lru_victim()
        };

        if let Some(key) = victim_key {
            self.habits.remove(&key);
            self.lru_order.retain(|k| k != &key);
            self.stats.evictions += 1;
        }
    }

    /// Find the LRU victim (front of the LRU queue, skipping active habits if possible)
    fn find_lru_victim(&self) -> Option<String> {
        // Prefer evicting inactive habits first
        for key in &self.lru_order {
            if let Some(entry) = self.habits.get(key) {
                if !entry.active {
                    return Some(key.clone());
                }
            }
        }
        // If all are active, evict the true LRU
        self.lru_order.front().cloned()
    }

    /// Find the victim using frequency-weighted scoring
    /// Score = strength * success_rate / (staleness + 1)
    /// Lower score → more likely to be evicted
    fn find_frequency_weighted_victim(&self) -> Option<String> {
        self.habits
            .iter()
            .filter(|(_, entry)| entry.strength < self.config.eviction_threshold || !entry.active)
            .min_by(|(_, a), (_, b)| {
                let score_a = self.eviction_score(a);
                let score_b = self.eviction_score(b);
                score_a
                    .partial_cmp(&score_b)
                    .unwrap_or(std::cmp::Ordering::Equal)
            })
            .map(|(key, _)| key.clone())
            .or_else(|| {
                // Fallback: evict the weakest entry regardless
                self.habits
                    .iter()
                    .min_by(|(_, a), (_, b)| {
                        a.strength
                            .partial_cmp(&b.strength)
                            .unwrap_or(std::cmp::Ordering::Equal)
                    })
                    .map(|(key, _)| key.clone())
            })
    }

    /// Compute an eviction score (higher = more valuable, less likely to evict)
    fn eviction_score(&self, entry: &HabitEntry) -> f64 {
        let staleness = entry.staleness(self.current_step) as f64;
        let recency_factor = 1.0 / (staleness + 1.0);
        let frequency_factor = (entry.use_count as f64).ln_1p();

        entry.strength * entry.success_rate() * recency_factor * (1.0 + frequency_factor)
    }

    /// Evict habits that have been stale for too long
    fn evict_stale(&mut self) {
        let max_staleness = self.config.max_staleness;
        let current = self.current_step;

        let stale_keys: Vec<String> = self
            .habits
            .iter()
            .filter(|(_, entry)| entry.staleness(current) > max_staleness)
            .map(|(key, _)| key.clone())
            .collect();

        for key in &stale_keys {
            self.habits.remove(key);
            self.lru_order.retain(|k| k != key);
            self.stats.evictions += 1;
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_basic() {
        let instance = HabitCache::new();
        assert!(instance.process().is_ok());
    }

    #[test]
    fn test_record_and_lookup_miss() {
        let mut cache = HabitCache::new();
        let result = cache.lookup("state_abc");
        assert!(matches!(result, HabitLookup::Miss));
    }

    #[test]
    fn test_record_and_lookup_weak_hit() {
        let mut cache = HabitCache::new();
        cache.record("state_abc", 2);

        let result = cache.lookup("state_abc");
        assert!(matches!(result, HabitLookup::WeakHit { action: 2, .. }));
    }

    #[test]
    fn test_reinforcement_promotes_to_active() {
        let mut cache = HabitCache::with_config(HabitCacheConfig {
            activation_threshold: 0.4,
            reinforcement_rate: 0.15,
            min_successes_to_activate: 2,
            decay_rate: 0.0,
            ..Default::default()
        });

        cache.record("state_x", 1);

        // Reinforce multiple times until promoted
        cache.reinforce("state_x", 1.0);
        cache.reinforce("state_x", 1.0);
        cache.reinforce("state_x", 1.0);

        let entry = cache.get("state_x").unwrap();
        assert!(entry.active, "habit should be active after reinforcement");
        assert!(entry.strength >= 0.4);

        let result = cache.lookup("state_x");
        assert!(
            matches!(result, HabitLookup::Hit { action: 1, .. }),
            "active habit should produce a Hit"
        );
    }

    #[test]
    fn test_punishment_demotes_habit() {
        let mut cache = HabitCache::with_config(HabitCacheConfig {
            activation_threshold: 0.3,
            reinforcement_rate: 0.15,
            punishment_rate: 0.30,
            min_successes_to_activate: 1,
            decay_rate: 0.0,
            ..Default::default()
        });

        cache.record("s", 0);
        cache.reinforce("s", 1.0);
        cache.reinforce("s", 1.0);

        assert!(cache.get("s").unwrap().active, "should be promoted");

        cache.punish("s", 5.0);
        cache.punish("s", 5.0);

        assert!(!cache.get("s").unwrap().active, "should be demoted");
    }

    #[test]
    fn test_decay_reduces_strength() {
        let mut cache = HabitCache::with_config(HabitCacheConfig {
            decay_rate: 0.05,
            activation_threshold: 0.3,
            reinforcement_rate: 0.20,
            min_successes_to_activate: 1,
            ..Default::default()
        });

        cache.record("s1", 0);
        cache.reinforce("s1", 1.0);
        cache.reinforce("s1", 1.0);

        let before = cache.get("s1").unwrap().strength;

        for _ in 0..5 {
            cache.tick();
        }

        let after = cache.get("s1").unwrap().strength;
        assert!(
            after < before,
            "strength should decay: {} >= {}",
            after,
            before
        );
    }

    #[test]
    fn test_capacity_eviction() {
        let mut cache = HabitCache::with_config(HabitCacheConfig {
            max_capacity: 3,
            frequency_weighted_eviction: false,
            max_staleness: u64::MAX,
            decay_rate: 0.0,
            ..Default::default()
        });

        cache.record("a", 0);
        cache.record("b", 1);
        cache.record("c", 2);

        assert_eq!(cache.size(), 3);

        // Adding a 4th should evict the LRU entry ("a")
        cache.record("d", 3);

        assert_eq!(cache.size(), 3);
        assert!(cache.get("a").is_none(), "LRU entry 'a' should be evicted");
        assert!(cache.get("d").is_some(), "new entry 'd' should exist");
    }

    #[test]
    fn test_stale_eviction() {
        let mut cache = HabitCache::with_config(HabitCacheConfig {
            max_staleness: 5,
            decay_rate: 0.0,
            ..Default::default()
        });

        cache.record("stale_habit", 0);

        for _ in 0..10 {
            cache.tick();
        }

        assert!(
            cache.get("stale_habit").is_none(),
            "stale habit should be evicted"
        );
    }

    #[test]
    fn test_success_rate() {
        let mut cache = HabitCache::new();
        cache.record("s", 0);

        // Simulate 3 uses: 2 reinforced, 1 punished
        cache.record("s", 0);
        cache.reinforce("s", 1.0);
        cache.record("s", 0);
        cache.reinforce("s", 1.0);
        cache.record("s", 0);
        cache.punish("s", 0.5);

        let entry = cache.get("s").unwrap();
        assert_eq!(entry.use_count, 3);
        assert_eq!(entry.success_count, 2);
        assert!((entry.success_rate() - 2.0 / 3.0).abs() < 1e-9);
    }

    #[test]
    fn test_stats_tracking() {
        let mut cache = HabitCache::with_config(HabitCacheConfig {
            decay_rate: 0.0,
            ..Default::default()
        });

        // Miss
        cache.lookup("unknown");
        // Record and weak hit
        cache.record("known", 0);
        cache.lookup("known");

        let stats = cache.stats();
        assert_eq!(stats.total_lookups, 2);
        assert_eq!(stats.misses, 1);
        assert_eq!(stats.weak_hits, 1);
        assert_eq!(stats.size, 1);
    }

    #[test]
    fn test_strongest_habits() {
        let mut cache = HabitCache::with_config(HabitCacheConfig {
            decay_rate: 0.0,
            reinforcement_rate: 0.20,
            min_successes_to_activate: 1,
            ..Default::default()
        });

        cache.record("weak", 0);
        cache.record("strong", 1);
        cache.reinforce("strong", 5.0);
        cache.reinforce("strong", 5.0);
        cache.record("medium", 2);
        cache.reinforce("medium", 1.0);

        let top = cache.strongest_habits(2);
        assert_eq!(top.len(), 2);
        assert_eq!(
            top[0].state_key, "strong",
            "strongest habit should be first"
        );
    }

    #[test]
    fn test_remove_and_clear() {
        let mut cache = HabitCache::new();
        cache.record("a", 0);
        cache.record("b", 1);

        let removed = cache.remove("a");
        assert!(removed.is_some());
        assert_eq!(cache.size(), 1);

        cache.clear();
        assert_eq!(cache.size(), 0);
    }

    #[test]
    fn test_reset() {
        let mut cache = HabitCache::new();
        cache.record("a", 0);
        cache.lookup("a");
        cache.tick();

        cache.reset();
        assert_eq!(cache.size(), 0);
        assert_eq!(cache.current_step(), 0);
        assert_eq!(cache.stats().total_lookups, 0);
    }

    #[test]
    fn test_habit_conflict_weak_override() {
        let mut cache = HabitCache::with_config(HabitCacheConfig {
            activation_threshold: 0.5,
            decay_rate: 0.0,
            ..Default::default()
        });

        cache.record("state", 0);
        assert_eq!(cache.get("state").unwrap().action, 0);

        // The habit is weak (strength 0.1), so a different action should override
        cache.record("state", 5);
        assert_eq!(
            cache.get("state").unwrap().action,
            5,
            "weak habit should be overridden"
        );
    }
}
