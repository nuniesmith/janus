//! Shared state across all brain regions
//!
//! Part of the Integration region — State component.
//!
//! `GlobalState` provides a centralised key-value store for cross-region data,
//! tracks region health, maintains system-wide tick counters and performance
//! metrics, and exposes EMA-smoothed and windowed diagnostics.

use crate::common::Result;
use std::collections::{HashMap, VecDeque};

// ---------------------------------------------------------------------------
// Configuration
// ---------------------------------------------------------------------------

/// Configuration for the global state manager.
#[derive(Debug, Clone)]
pub struct GlobalStateConfig {
    /// Maximum number of tracked key-value entries.
    pub max_entries: usize,
    /// Maximum number of tracked regions.
    pub max_regions: usize,
    /// EMA decay factor (0 < α < 1). Higher → faster adaptation.
    pub ema_decay: f64,
    /// Window size for rolling diagnostics.
    pub window_size: usize,
    /// Number of ticks after which a region is considered stale.
    pub region_staleness_ticks: u64,
}

impl Default for GlobalStateConfig {
    fn default() -> Self {
        Self {
            max_entries: 4096,
            max_regions: 64,
            ema_decay: 0.1,
            window_size: 50,
            region_staleness_ticks: 100,
        }
    }
}

// ---------------------------------------------------------------------------
// Region health
// ---------------------------------------------------------------------------

/// Health status of a brain region.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum RegionHealth {
    /// Normal operation.
    Healthy,
    /// Operating but with degraded performance.
    Degraded,
    /// Not responding or errored out.
    Unhealthy,
    /// Region has not reported within the staleness window.
    Stale,
}

/// Per-region health record.
#[derive(Debug, Clone)]
pub struct RegionRecord {
    /// Human-readable region name.
    pub name: String,
    /// Current health status.
    pub health: RegionHealth,
    /// Last tick at which the region reported in.
    pub last_heartbeat: u64,
    /// Cumulative number of heartbeats received.
    pub heartbeat_count: u64,
    /// Optional error message from the most recent failure.
    pub last_error: Option<String>,
}

// ---------------------------------------------------------------------------
// Snapshot record (for windowed diagnostics)
// ---------------------------------------------------------------------------

#[derive(Debug, Clone)]
#[allow(dead_code)]
struct TickSnapshot {
    entry_count: usize,
    healthy_regions: usize,
    total_regions: usize,
    /// Fraction of regions that are healthy.
    health_ratio: f64,
}

// ---------------------------------------------------------------------------
// Statistics
// ---------------------------------------------------------------------------

/// Cumulative statistics for `GlobalState`.
#[derive(Debug, Clone)]
pub struct GlobalStateStats {
    /// Total number of `set` operations.
    pub total_sets: u64,
    /// Total number of `get` operations.
    pub total_gets: u64,
    /// Total number of `get` hits (key existed).
    pub total_hits: u64,
    /// Total number of `get` misses.
    pub total_misses: u64,
    /// Total number of entries evicted.
    pub total_evictions: u64,
    /// Total heartbeats received across all regions.
    pub total_heartbeats: u64,
    /// EMA-smoothed hit rate.
    pub ema_hit_rate: f64,
    /// EMA-smoothed health ratio (fraction of healthy regions).
    pub ema_health_ratio: f64,
    /// EMA-smoothed entry count.
    pub ema_entry_count: f64,
}

impl Default for GlobalStateStats {
    fn default() -> Self {
        Self {
            total_sets: 0,
            total_gets: 0,
            total_hits: 0,
            total_misses: 0,
            total_evictions: 0,
            total_heartbeats: 0,
            ema_hit_rate: 0.0,
            ema_health_ratio: 1.0,
            ema_entry_count: 0.0,
        }
    }
}

// ---------------------------------------------------------------------------
// GlobalState
// ---------------------------------------------------------------------------

/// Shared state container for cross-region communication.
///
/// Provides:
/// * A string-keyed value store (`f64` values) with optional eviction.
/// * Region health tracking with heartbeats and staleness detection.
/// * Tick-based lifecycle with per-tick EMA and windowed diagnostics.
pub struct GlobalState {
    config: GlobalStateConfig,
    /// Key → value store.
    entries: HashMap<String, f64>,
    /// Region name → record.
    regions: HashMap<String, RegionRecord>,
    /// Monotonically increasing tick counter.
    tick: u64,
    /// Whether EMA has been initialised (first tick).
    ema_initialized: bool,
    /// Rolling window of tick snapshots.
    recent: VecDeque<TickSnapshot>,
    /// Cumulative stats.
    stats: GlobalStateStats,
}

impl Default for GlobalState {
    fn default() -> Self {
        Self::new()
    }
}

impl GlobalState {
    // -------------------------------------------------------------------
    // Construction
    // -------------------------------------------------------------------

    /// Create with default configuration.
    pub fn new() -> Self {
        Self::with_config(GlobalStateConfig::default())
    }

    /// Create with explicit configuration.
    pub fn with_config(config: GlobalStateConfig) -> Self {
        Self {
            entries: HashMap::with_capacity(config.max_entries.min(256)),
            regions: HashMap::with_capacity(config.max_regions.min(16)),
            tick: 0,
            ema_initialized: false,
            recent: VecDeque::with_capacity(config.window_size),
            stats: GlobalStateStats::default(),
            config,
        }
    }

    // -------------------------------------------------------------------
    // Key-value store
    // -------------------------------------------------------------------

    /// Store a value. If the store is at capacity, the oldest entry is
    /// evicted (FIFO by insertion order is approximated by picking an
    /// arbitrary key from the map — good enough for diagnostics).
    pub fn set(&mut self, key: impl Into<String>, value: f64) {
        let key = key.into();
        self.stats.total_sets += 1;

        // Evict if at capacity and key is new
        if self.entries.len() >= self.config.max_entries && !self.entries.contains_key(&key) {
            if let Some(victim) = self.entries.keys().next().cloned() {
                self.entries.remove(&victim);
                self.stats.total_evictions += 1;
            }
        }

        self.entries.insert(key, value);
    }

    /// Retrieve a value by key.
    pub fn get(&mut self, key: &str) -> Option<f64> {
        self.stats.total_gets += 1;
        let result = self.entries.get(key).copied();
        if result.is_some() {
            self.stats.total_hits += 1;
        } else {
            self.stats.total_misses += 1;
        }
        result
    }

    /// Retrieve a value without updating stats (read-only peek).
    pub fn peek(&self, key: &str) -> Option<f64> {
        self.entries.get(key).copied()
    }

    /// Remove a key.
    pub fn remove(&mut self, key: &str) -> Option<f64> {
        self.entries.remove(key)
    }

    /// Check if a key exists.
    pub fn contains_key(&self, key: &str) -> bool {
        self.entries.contains_key(key)
    }

    /// Number of entries currently stored.
    pub fn entry_count(&self) -> usize {
        self.entries.len()
    }

    /// All stored keys (unordered).
    pub fn keys(&self) -> Vec<String> {
        self.entries.keys().cloned().collect()
    }

    /// Clear all entries.
    pub fn clear_entries(&mut self) {
        self.entries.clear();
    }

    // -------------------------------------------------------------------
    // Region health
    // -------------------------------------------------------------------

    /// Register a new region. No-op if already registered.
    pub fn register_region(&mut self, name: impl Into<String>) {
        let name = name.into();
        if self.regions.len() >= self.config.max_regions && !self.regions.contains_key(&name) {
            return; // silently ignore if at capacity
        }
        self.regions
            .entry(name.clone())
            .or_insert_with(|| RegionRecord {
                name,
                health: RegionHealth::Healthy,
                last_heartbeat: self.tick,
                heartbeat_count: 0,
                last_error: None,
            });
    }

    /// Record a heartbeat from a region, resetting its health to `Healthy`.
    pub fn heartbeat(&mut self, region: &str) {
        if let Some(rec) = self.regions.get_mut(region) {
            rec.last_heartbeat = self.tick;
            rec.heartbeat_count += 1;
            rec.health = RegionHealth::Healthy;
            rec.last_error = None;
            self.stats.total_heartbeats += 1;
        }
    }

    /// Mark a region as degraded.
    pub fn mark_degraded(&mut self, region: &str, reason: impl Into<String>) {
        if let Some(rec) = self.regions.get_mut(region) {
            rec.health = RegionHealth::Degraded;
            rec.last_error = Some(reason.into());
        }
    }

    /// Mark a region as unhealthy.
    pub fn mark_unhealthy(&mut self, region: &str, reason: impl Into<String>) {
        if let Some(rec) = self.regions.get_mut(region) {
            rec.health = RegionHealth::Unhealthy;
            rec.last_error = Some(reason.into());
        }
    }

    /// Get the health record for a region.
    pub fn region_health(&self, region: &str) -> Option<&RegionRecord> {
        self.regions.get(region)
    }

    /// Number of registered regions.
    pub fn region_count(&self) -> usize {
        self.regions.len()
    }

    /// Number of regions currently `Healthy`.
    pub fn healthy_region_count(&self) -> usize {
        self.regions
            .values()
            .filter(|r| r.health == RegionHealth::Healthy)
            .count()
    }

    /// Fraction of regions that are healthy (0.0 .. 1.0).
    pub fn health_ratio(&self) -> f64 {
        if self.regions.is_empty() {
            return 1.0;
        }
        self.healthy_region_count() as f64 / self.regions.len() as f64
    }

    /// Names of all unhealthy or stale regions.
    pub fn unhealthy_regions(&self) -> Vec<String> {
        self.regions
            .values()
            .filter(|r| r.health == RegionHealth::Unhealthy || r.health == RegionHealth::Stale)
            .map(|r| r.name.clone())
            .collect()
    }

    // -------------------------------------------------------------------
    // Tick lifecycle
    // -------------------------------------------------------------------

    /// Advance the global tick counter and update diagnostics.
    ///
    /// This should be called once per system tick. It:
    /// 1. Increments the tick counter.
    /// 2. Marks stale regions.
    /// 3. Captures a snapshot for windowed stats.
    /// 4. Updates EMA-smoothed metrics.
    pub fn tick(&mut self) {
        self.tick += 1;
        self.mark_stale_regions();

        let snapshot = TickSnapshot {
            entry_count: self.entries.len(),
            healthy_regions: self.healthy_region_count(),
            total_regions: self.regions.len(),
            health_ratio: self.health_ratio(),
        };

        // Update EMA
        let alpha = self.config.ema_decay;
        let hit_rate = if self.stats.total_gets > 0 {
            self.stats.total_hits as f64 / self.stats.total_gets as f64
        } else {
            0.0
        };

        if !self.ema_initialized {
            self.stats.ema_hit_rate = hit_rate;
            self.stats.ema_health_ratio = snapshot.health_ratio;
            self.stats.ema_entry_count = snapshot.entry_count as f64;
            self.ema_initialized = true;
        } else {
            self.stats.ema_hit_rate = alpha * hit_rate + (1.0 - alpha) * self.stats.ema_hit_rate;
            self.stats.ema_health_ratio =
                alpha * snapshot.health_ratio + (1.0 - alpha) * self.stats.ema_health_ratio;
            self.stats.ema_entry_count =
                alpha * snapshot.entry_count as f64 + (1.0 - alpha) * self.stats.ema_entry_count;
        }

        // Windowed
        if self.recent.len() >= self.config.window_size {
            self.recent.pop_front();
        }
        self.recent.push_back(snapshot);
    }

    /// Current tick value.
    pub fn current_tick(&self) -> u64 {
        self.tick
    }

    /// Main processing function (alias for `tick`).
    pub fn process(&mut self) -> Result<()> {
        self.tick();
        Ok(())
    }

    // -------------------------------------------------------------------
    // Staleness detection
    // -------------------------------------------------------------------

    fn mark_stale_regions(&mut self) {
        let threshold = self.config.region_staleness_ticks;
        let current = self.tick;
        for rec in self.regions.values_mut() {
            if rec.health != RegionHealth::Unhealthy
                && current.saturating_sub(rec.last_heartbeat) > threshold
            {
                rec.health = RegionHealth::Stale;
            }
        }
    }

    // -------------------------------------------------------------------
    // Statistics & diagnostics
    // -------------------------------------------------------------------

    /// Reference to cumulative statistics.
    pub fn stats(&self) -> &GlobalStateStats {
        &self.stats
    }

    /// Reference to the configuration.
    pub fn config(&self) -> &GlobalStateConfig {
        &self.config
    }

    /// EMA-smoothed hit rate.
    pub fn smoothed_hit_rate(&self) -> f64 {
        self.stats.ema_hit_rate
    }

    /// EMA-smoothed health ratio.
    pub fn smoothed_health_ratio(&self) -> f64 {
        self.stats.ema_health_ratio
    }

    /// EMA-smoothed entry count.
    pub fn smoothed_entry_count(&self) -> f64 {
        self.stats.ema_entry_count
    }

    /// Windowed average health ratio over the recent window.
    pub fn windowed_health_ratio(&self) -> f64 {
        if self.recent.is_empty() {
            return 1.0;
        }
        let sum: f64 = self.recent.iter().map(|s| s.health_ratio).sum();
        sum / self.recent.len() as f64
    }

    /// Windowed average entry count over the recent window.
    pub fn windowed_entry_count(&self) -> f64 {
        if self.recent.is_empty() {
            return 0.0;
        }
        let sum: f64 = self.recent.iter().map(|s| s.entry_count as f64).sum();
        sum / self.recent.len() as f64
    }

    /// Whether health is trending downward over the window.
    pub fn is_health_declining(&self) -> bool {
        if self.recent.len() < 4 {
            return false;
        }
        let n = self.recent.len();
        let half = n / 2;
        let first_avg: f64 = self
            .recent
            .iter()
            .take(half)
            .map(|s| s.health_ratio)
            .sum::<f64>()
            / half as f64;
        let second_avg: f64 = self
            .recent
            .iter()
            .skip(half)
            .map(|s| s.health_ratio)
            .sum::<f64>()
            / (n - half) as f64;
        second_avg < first_avg
    }

    /// Reset all state (entries, regions, stats, ticks).
    pub fn reset(&mut self) {
        self.entries.clear();
        self.regions.clear();
        self.tick = 0;
        self.ema_initialized = false;
        self.recent.clear();
        self.stats = GlobalStateStats::default();
    }
}

// ===========================================================================
// Tests
// ===========================================================================

#[cfg(test)]
mod tests {
    use super::*;

    fn small_config() -> GlobalStateConfig {
        GlobalStateConfig {
            max_entries: 8,
            max_regions: 4,
            ema_decay: 0.5,
            window_size: 5,
            region_staleness_ticks: 3,
        }
    }

    // -------------------------------------------------------------------
    // Construction
    // -------------------------------------------------------------------

    #[test]
    fn test_new_default() {
        let gs = GlobalState::new();
        assert_eq!(gs.entry_count(), 0);
        assert_eq!(gs.region_count(), 0);
        assert_eq!(gs.current_tick(), 0);
    }

    #[test]
    fn test_with_config() {
        let gs = GlobalState::with_config(small_config());
        assert_eq!(gs.config().max_entries, 8);
        assert_eq!(gs.config().max_regions, 4);
    }

    // -------------------------------------------------------------------
    // Key-value store
    // -------------------------------------------------------------------

    #[test]
    fn test_set_and_get() {
        let mut gs = GlobalState::new();
        gs.set("price", 100.0);
        assert_eq!(gs.get("price"), Some(100.0));
    }

    #[test]
    fn test_get_miss() {
        let mut gs = GlobalState::new();
        assert_eq!(gs.get("nonexistent"), None);
        assert_eq!(gs.stats().total_misses, 1);
    }

    #[test]
    fn test_peek_no_stats() {
        let mut gs = GlobalState::new();
        gs.set("x", 42.0);
        let total_before = gs.stats().total_gets;
        assert_eq!(gs.peek("x"), Some(42.0));
        assert_eq!(gs.stats().total_gets, total_before);
    }

    #[test]
    fn test_remove() {
        let mut gs = GlobalState::new();
        gs.set("a", 1.0);
        assert_eq!(gs.remove("a"), Some(1.0));
        assert_eq!(gs.peek("a"), None);
    }

    #[test]
    fn test_contains_key() {
        let mut gs = GlobalState::new();
        gs.set("k", 0.0);
        assert!(gs.contains_key("k"));
        assert!(!gs.contains_key("z"));
    }

    #[test]
    fn test_keys() {
        let mut gs = GlobalState::new();
        gs.set("a", 1.0);
        gs.set("b", 2.0);
        let mut keys = gs.keys();
        keys.sort();
        assert_eq!(keys, vec!["a", "b"]);
    }

    #[test]
    fn test_clear_entries() {
        let mut gs = GlobalState::new();
        gs.set("a", 1.0);
        gs.set("b", 2.0);
        gs.clear_entries();
        assert_eq!(gs.entry_count(), 0);
    }

    #[test]
    fn test_eviction_at_capacity() {
        let mut gs = GlobalState::with_config(small_config());
        for i in 0..10 {
            gs.set(format!("k{}", i), i as f64);
        }
        // Should have at most max_entries
        assert!(gs.entry_count() <= 8);
        assert!(gs.stats().total_evictions > 0);
    }

    #[test]
    fn test_overwrite_existing_no_eviction() {
        let mut gs = GlobalState::with_config(small_config());
        for i in 0..8 {
            gs.set(format!("k{}", i), i as f64);
        }
        let evictions_before = gs.stats().total_evictions;
        gs.set("k0", 999.0);
        assert_eq!(gs.stats().total_evictions, evictions_before);
        assert_eq!(gs.peek("k0"), Some(999.0));
    }

    // -------------------------------------------------------------------
    // Region health
    // -------------------------------------------------------------------

    #[test]
    fn test_register_region() {
        let mut gs = GlobalState::with_config(small_config());
        gs.register_region("cortex");
        assert_eq!(gs.region_count(), 1);
        assert_eq!(
            gs.region_health("cortex").unwrap().health,
            RegionHealth::Healthy
        );
    }

    #[test]
    fn test_register_duplicate_noop() {
        let mut gs = GlobalState::with_config(small_config());
        gs.register_region("cortex");
        gs.register_region("cortex");
        assert_eq!(gs.region_count(), 1);
    }

    #[test]
    fn test_register_region_at_capacity() {
        let mut gs = GlobalState::with_config(small_config()); // max_regions = 4
        for i in 0..5 {
            gs.register_region(format!("r{}", i));
        }
        assert_eq!(gs.region_count(), 4);
    }

    #[test]
    fn test_heartbeat() {
        let mut gs = GlobalState::with_config(small_config());
        gs.register_region("cortex");
        gs.heartbeat("cortex");
        let rec = gs.region_health("cortex").unwrap();
        assert_eq!(rec.heartbeat_count, 1);
        assert_eq!(rec.health, RegionHealth::Healthy);
        assert_eq!(gs.stats().total_heartbeats, 1);
    }

    #[test]
    fn test_heartbeat_unknown_region() {
        let mut gs = GlobalState::with_config(small_config());
        gs.heartbeat("unknown"); // should not panic
        assert_eq!(gs.stats().total_heartbeats, 0);
    }

    #[test]
    fn test_mark_degraded() {
        let mut gs = GlobalState::with_config(small_config());
        gs.register_region("thalamus");
        gs.mark_degraded("thalamus", "high latency");
        let rec = gs.region_health("thalamus").unwrap();
        assert_eq!(rec.health, RegionHealth::Degraded);
        assert_eq!(rec.last_error.as_deref(), Some("high latency"));
    }

    #[test]
    fn test_mark_unhealthy() {
        let mut gs = GlobalState::with_config(small_config());
        gs.register_region("amygdala");
        gs.mark_unhealthy("amygdala", "crash");
        assert_eq!(
            gs.region_health("amygdala").unwrap().health,
            RegionHealth::Unhealthy
        );
    }

    #[test]
    fn test_healthy_region_count() {
        let mut gs = GlobalState::with_config(small_config());
        gs.register_region("a");
        gs.register_region("b");
        gs.register_region("c");
        gs.mark_unhealthy("b", "err");
        assert_eq!(gs.healthy_region_count(), 2);
    }

    #[test]
    fn test_health_ratio() {
        let mut gs = GlobalState::with_config(small_config());
        gs.register_region("a");
        gs.register_region("b");
        gs.mark_unhealthy("b", "err");
        assert!((gs.health_ratio() - 0.5).abs() < 1e-10);
    }

    #[test]
    fn test_health_ratio_no_regions() {
        let gs = GlobalState::new();
        assert!((gs.health_ratio() - 1.0).abs() < 1e-10);
    }

    #[test]
    fn test_unhealthy_regions_list() {
        let mut gs = GlobalState::with_config(small_config());
        gs.register_region("a");
        gs.register_region("b");
        gs.mark_unhealthy("b", "err");
        let list = gs.unhealthy_regions();
        assert_eq!(list, vec!["b"]);
    }

    // -------------------------------------------------------------------
    // Staleness detection
    // -------------------------------------------------------------------

    #[test]
    fn test_stale_detection() {
        let mut gs = GlobalState::with_config(small_config()); // staleness = 3
        gs.register_region("cortex");
        gs.heartbeat("cortex");
        // Advance 4 ticks without heartbeat
        for _ in 0..4 {
            gs.tick();
        }
        assert_eq!(
            gs.region_health("cortex").unwrap().health,
            RegionHealth::Stale
        );
    }

    #[test]
    fn test_heartbeat_resets_stale() {
        let mut gs = GlobalState::with_config(small_config());
        gs.register_region("cortex");
        for _ in 0..5 {
            gs.tick();
        }
        assert_eq!(
            gs.region_health("cortex").unwrap().health,
            RegionHealth::Stale
        );
        gs.heartbeat("cortex");
        assert_eq!(
            gs.region_health("cortex").unwrap().health,
            RegionHealth::Healthy
        );
    }

    #[test]
    fn test_unhealthy_not_overwritten_by_stale() {
        let mut gs = GlobalState::with_config(small_config());
        gs.register_region("cortex");
        gs.mark_unhealthy("cortex", "crash");
        for _ in 0..5 {
            gs.tick();
        }
        // Should remain Unhealthy, not switch to Stale
        assert_eq!(
            gs.region_health("cortex").unwrap().health,
            RegionHealth::Unhealthy
        );
    }

    // -------------------------------------------------------------------
    // Tick & process
    // -------------------------------------------------------------------

    #[test]
    fn test_tick_increments() {
        let mut gs = GlobalState::new();
        gs.tick();
        gs.tick();
        assert_eq!(gs.current_tick(), 2);
    }

    #[test]
    fn test_process() {
        let mut gs = GlobalState::new();
        assert!(gs.process().is_ok());
        assert_eq!(gs.current_tick(), 1);
    }

    // -------------------------------------------------------------------
    // EMA diagnostics
    // -------------------------------------------------------------------

    #[test]
    fn test_ema_initialises_on_first_tick() {
        let mut gs = GlobalState::with_config(small_config());
        gs.set("a", 1.0);
        gs.get("a");
        gs.tick();
        // After first tick, EMA should be initialised
        assert!(gs.smoothed_hit_rate() > 0.0);
        assert!(gs.smoothed_entry_count() > 0.0);
    }

    #[test]
    fn test_ema_blends_on_subsequent_ticks() {
        let mut gs = GlobalState::with_config(GlobalStateConfig {
            ema_decay: 0.5,
            ..GlobalStateConfig::default()
        });
        gs.register_region("a");
        gs.tick(); // health_ratio = 1.0, ema = 1.0
        gs.mark_unhealthy("a", "err");
        gs.tick(); // health_ratio = 0.0, ema = 0.5 * 0.0 + 0.5 * 1.0 = 0.5
        assert!((gs.smoothed_health_ratio() - 0.5).abs() < 1e-10);
    }

    // -------------------------------------------------------------------
    // Windowed diagnostics
    // -------------------------------------------------------------------

    #[test]
    fn test_windowed_health_ratio() {
        let mut gs = GlobalState::with_config(small_config()); // window_size = 5
        gs.register_region("a");
        for _ in 0..3 {
            gs.tick();
        }
        // All ticks healthy → avg should be 1.0
        assert!((gs.windowed_health_ratio() - 1.0).abs() < 1e-10);
    }

    #[test]
    fn test_windowed_entry_count() {
        let mut gs = GlobalState::with_config(small_config());
        gs.set("x", 1.0);
        gs.tick();
        gs.set("y", 2.0);
        gs.tick();
        // Should have some positive average
        assert!(gs.windowed_entry_count() > 0.0);
    }

    #[test]
    fn test_windowed_empty() {
        let gs = GlobalState::new();
        assert!((gs.windowed_health_ratio() - 1.0).abs() < 1e-10);
        assert!((gs.windowed_entry_count()).abs() < 1e-10);
    }

    #[test]
    fn test_is_health_declining() {
        let mut gs = GlobalState::with_config(small_config());
        gs.register_region("a");
        // First few ticks healthy
        for _ in 0..3 {
            gs.heartbeat("a");
            gs.tick();
        }
        // Then mark unhealthy
        gs.mark_unhealthy("a", "err");
        for _ in 0..3 {
            gs.tick();
        }
        assert!(gs.is_health_declining());
    }

    #[test]
    fn test_is_health_declining_insufficient_data() {
        let mut gs = GlobalState::with_config(small_config());
        gs.tick();
        assert!(!gs.is_health_declining());
    }

    // -------------------------------------------------------------------
    // Hit rate stats
    // -------------------------------------------------------------------

    #[test]
    fn test_hit_rate_tracking() {
        let mut gs = GlobalState::new();
        gs.set("a", 1.0);
        gs.get("a");
        gs.get("b"); // miss
        assert_eq!(gs.stats().total_hits, 1);
        assert_eq!(gs.stats().total_misses, 1);
        assert_eq!(gs.stats().total_gets, 2);
    }

    #[test]
    fn test_set_stats() {
        let mut gs = GlobalState::new();
        gs.set("a", 1.0);
        gs.set("b", 2.0);
        assert_eq!(gs.stats().total_sets, 2);
    }

    // -------------------------------------------------------------------
    // Reset
    // -------------------------------------------------------------------

    #[test]
    fn test_reset() {
        let mut gs = GlobalState::with_config(small_config());
        gs.set("a", 1.0);
        gs.register_region("cortex");
        gs.tick();
        gs.tick();
        gs.reset();
        assert_eq!(gs.entry_count(), 0);
        assert_eq!(gs.region_count(), 0);
        assert_eq!(gs.current_tick(), 0);
        assert_eq!(gs.stats().total_sets, 0);
    }

    // -------------------------------------------------------------------
    // Full lifecycle
    // -------------------------------------------------------------------

    #[test]
    fn test_full_lifecycle() {
        let mut gs = GlobalState::with_config(small_config());

        // Register regions
        gs.register_region("cortex");
        gs.register_region("thalamus");
        gs.register_region("cerebellum");

        // Store some data
        gs.set("btc_price", 50000.0);
        gs.set("eth_price", 3000.0);

        // Run several ticks with heartbeats
        for i in 0..5 {
            gs.heartbeat("cortex");
            if i % 2 == 0 {
                gs.heartbeat("thalamus");
            }
            gs.heartbeat("cerebellum");
            gs.tick();
        }

        assert_eq!(gs.current_tick(), 5);
        assert!(gs.healthy_region_count() >= 2);
        assert_eq!(gs.entry_count(), 2);
        assert!(gs.smoothed_health_ratio() > 0.0);
        assert!(gs.stats().total_heartbeats > 0);

        // Query
        assert_eq!(gs.get("btc_price"), Some(50000.0));

        // Mark degradation
        gs.mark_degraded("cortex", "latency spike");
        assert_eq!(
            gs.region_health("cortex").unwrap().health,
            RegionHealth::Degraded
        );
    }

    #[test]
    fn test_window_rolls() {
        let mut gs = GlobalState::with_config(small_config()); // window_size = 5
        for _ in 0..10 {
            gs.tick();
        }
        assert!(gs.recent.len() <= gs.config.window_size);
    }
}
