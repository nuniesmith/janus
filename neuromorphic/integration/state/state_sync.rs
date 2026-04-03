//! State synchronization across brain regions
//!
//! Part of the Integration region — State component.
//!
//! `StateSync` provides version-tracked state synchronization between brain
//! regions. It maintains per-region version vectors, supports snapshot and
//! diff-based synchronization, detects conflicts when concurrent writes
//! occur, and exposes EMA-smoothed and windowed diagnostics.

use crate::common::Result;
use std::collections::{HashMap, VecDeque};

// ---------------------------------------------------------------------------
// Configuration
// ---------------------------------------------------------------------------

/// Configuration for the state synchronizer.
#[derive(Debug, Clone)]
pub struct StateSyncConfig {
    /// Maximum number of regions that can participate in sync.
    pub max_regions: usize,
    /// Maximum number of key-value pairs tracked per region.
    pub max_keys_per_region: usize,
    /// Maximum number of diffs retained in history.
    pub max_diff_history: usize,
    /// EMA decay factor (0 < α < 1). Higher → faster adaptation.
    pub ema_decay: f64,
    /// Window size for rolling diagnostics.
    pub window_size: usize,
}

impl Default for StateSyncConfig {
    fn default() -> Self {
        Self {
            max_regions: 64,
            max_keys_per_region: 1024,
            max_diff_history: 200,
            ema_decay: 0.1,
            window_size: 50,
        }
    }
}

// ---------------------------------------------------------------------------
// Sync types
// ---------------------------------------------------------------------------

/// A versioned value stored in a region's state.
#[derive(Debug, Clone)]
pub struct VersionedValue {
    /// The stored value.
    pub value: f64,
    /// Monotonically increasing version for this key within the owning region.
    pub version: u64,
    /// Tick at which this value was last written.
    pub updated_at: u64,
}

/// A single key-value change in a diff.
#[derive(Debug, Clone)]
pub struct DiffEntry {
    /// The key that changed.
    pub key: String,
    /// The new value.
    pub value: f64,
    /// The version after the change.
    pub version: u64,
}

/// A diff representing changes in a region since a given base version.
#[derive(Debug, Clone)]
pub struct StateDiff {
    /// Region that produced the diff.
    pub region: String,
    /// The base version vector snapshot before the changes.
    pub base_version: u64,
    /// The new version after the changes.
    pub new_version: u64,
    /// Individual key-value changes.
    pub entries: Vec<DiffEntry>,
    /// Tick at which the diff was produced.
    pub tick: u64,
}

/// A full snapshot of a region's state.
#[derive(Debug, Clone)]
pub struct StateSnapshot {
    /// Region name.
    pub region: String,
    /// Current logical version for the region.
    pub version: u64,
    /// All key-value pairs.
    pub data: HashMap<String, f64>,
    /// Tick at which the snapshot was taken.
    pub tick: u64,
}

/// Outcome of applying a diff or snapshot.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum SyncOutcome {
    /// Applied cleanly, no conflicts.
    Applied,
    /// Applied with conflict resolution (last-writer-wins).
    ConflictResolved,
    /// Rejected (e.g. stale diff, unknown region).
    Rejected,
}

/// Result of a sync operation.
#[derive(Debug, Clone)]
pub struct SyncResult {
    /// Outcome of the sync.
    pub outcome: SyncOutcome,
    /// Number of keys updated.
    pub keys_updated: usize,
    /// Number of conflicts detected.
    pub conflicts: usize,
}

// ---------------------------------------------------------------------------
// Per-region state
// ---------------------------------------------------------------------------

/// Internal per-region state tracker.
#[derive(Debug, Clone)]
struct RegionState {
    /// Key → versioned value.
    data: HashMap<String, VersionedValue>,
    /// Logical version counter for the region (increments on each write batch).
    version: u64,
    /// Last tick at which this region was synced.
    last_sync_tick: u64,
}

impl RegionState {
    fn new() -> Self {
        Self {
            data: HashMap::new(),
            version: 0,
            last_sync_tick: 0,
        }
    }
}

// ---------------------------------------------------------------------------
// Tick snapshot (windowed diagnostics)
// ---------------------------------------------------------------------------

#[derive(Debug, Clone)]
#[allow(dead_code)]
struct TickSnapshot {
    syncs_performed: u64,
    keys_synced: u64,
    conflicts_detected: u64,
    diffs_applied: u64,
    snapshots_applied: u64,
}

// ---------------------------------------------------------------------------
// Statistics
// ---------------------------------------------------------------------------

/// Cumulative statistics for `StateSync`.
#[derive(Debug, Clone)]
pub struct StateSyncStats {
    /// Total number of sync operations performed.
    pub total_syncs: u64,
    /// Total number of individual keys synced.
    pub total_keys_synced: u64,
    /// Total number of conflicts detected.
    pub total_conflicts: u64,
    /// Total number of diffs applied.
    pub total_diffs_applied: u64,
    /// Total number of full snapshots applied.
    pub total_snapshots_applied: u64,
    /// Total number of rejected sync operations.
    pub total_rejections: u64,
    /// Total number of writes (set operations).
    pub total_writes: u64,
    /// EMA-smoothed syncs per tick.
    pub ema_sync_rate: f64,
    /// EMA-smoothed keys synced per tick.
    pub ema_keys_rate: f64,
    /// EMA-smoothed conflict rate (conflicts / syncs).
    pub ema_conflict_rate: f64,
}

impl Default for StateSyncStats {
    fn default() -> Self {
        Self {
            total_syncs: 0,
            total_keys_synced: 0,
            total_conflicts: 0,
            total_diffs_applied: 0,
            total_snapshots_applied: 0,
            total_rejections: 0,
            total_writes: 0,
            ema_sync_rate: 0.0,
            ema_keys_rate: 0.0,
            ema_conflict_rate: 0.0,
        }
    }
}

// ---------------------------------------------------------------------------
// StateSync
// ---------------------------------------------------------------------------

/// Version-tracked state synchronization engine.
///
/// Provides:
/// * Per-region versioned key-value stores.
/// * `set` / `get` for local writes and reads.
/// * `snapshot` for capturing a region's full state.
/// * `apply_snapshot` for full-state replacement.
/// * `diff` for computing changes since a base version.
/// * `apply_diff` for incremental synchronization.
/// * Conflict detection (concurrent writes to the same key from different
///   regions) resolved via last-writer-wins.
/// * Tick-based lifecycle with EMA and windowed diagnostics.
pub struct StateSync {
    config: StateSyncConfig,
    /// Region name → state.
    regions: HashMap<String, RegionState>,
    /// History of applied diffs (most recent at back).
    diff_history: VecDeque<StateDiff>,
    /// Current tick.
    tick: u64,
    /// Per-tick counters (reset each tick).
    tick_syncs: u64,
    tick_keys: u64,
    tick_conflicts: u64,
    tick_diffs: u64,
    tick_snapshots: u64,
    /// EMA initialisation flag.
    ema_initialized: bool,
    /// Rolling window of tick snapshots.
    recent: VecDeque<TickSnapshot>,
    /// Cumulative statistics.
    stats: StateSyncStats,
}

impl Default for StateSync {
    fn default() -> Self {
        Self::new()
    }
}

impl StateSync {
    // -------------------------------------------------------------------
    // Construction
    // -------------------------------------------------------------------

    /// Create with default configuration.
    pub fn new() -> Self {
        Self::with_config(StateSyncConfig::default())
    }

    /// Create with explicit configuration.
    pub fn with_config(config: StateSyncConfig) -> Self {
        Self {
            regions: HashMap::with_capacity(config.max_regions.min(16)),
            diff_history: VecDeque::with_capacity(config.max_diff_history.min(64)),
            tick: 0,
            tick_syncs: 0,
            tick_keys: 0,
            tick_conflicts: 0,
            tick_diffs: 0,
            tick_snapshots: 0,
            ema_initialized: false,
            recent: VecDeque::with_capacity(config.window_size),
            stats: StateSyncStats::default(),
            config,
        }
    }

    // -------------------------------------------------------------------
    // Region management
    // -------------------------------------------------------------------

    /// Register a region for synchronization. No-op if already registered.
    pub fn register_region(&mut self, name: impl Into<String>) {
        let name = name.into();
        if self.regions.len() >= self.config.max_regions && !self.regions.contains_key(&name) {
            return;
        }
        self.regions.entry(name).or_insert_with(RegionState::new);
    }

    /// Number of registered regions.
    pub fn region_count(&self) -> usize {
        self.regions.len()
    }

    /// Whether a region is registered.
    pub fn has_region(&self, name: &str) -> bool {
        self.regions.contains_key(name)
    }

    /// Current logical version for a region.
    pub fn region_version(&self, name: &str) -> Option<u64> {
        self.regions.get(name).map(|r| r.version)
    }

    /// Number of keys stored for a region.
    pub fn region_key_count(&self, name: &str) -> usize {
        self.regions.get(name).map(|r| r.data.len()).unwrap_or(0)
    }

    // -------------------------------------------------------------------
    // Local read/write
    // -------------------------------------------------------------------

    /// Write a value into a region's state. Increments the region's version.
    pub fn set(&mut self, region: &str, key: impl Into<String>, value: f64) {
        if let Some(rs) = self.regions.get_mut(region) {
            let key = key.into();
            if rs.data.len() >= self.config.max_keys_per_region && !rs.data.contains_key(&key) {
                // At capacity for this region — evict an arbitrary key
                if let Some(victim) = rs.data.keys().next().cloned() {
                    rs.data.remove(&victim);
                }
            }
            rs.version += 1;
            rs.data.insert(
                key,
                VersionedValue {
                    value,
                    version: rs.version,
                    updated_at: self.tick,
                },
            );
            self.stats.total_writes += 1;
        }
    }

    /// Read a value from a region's state.
    pub fn get(&self, region: &str, key: &str) -> Option<f64> {
        self.regions
            .get(region)
            .and_then(|rs| rs.data.get(key))
            .map(|v| v.value)
    }

    /// Read a versioned value from a region's state.
    pub fn get_versioned(&self, region: &str, key: &str) -> Option<&VersionedValue> {
        self.regions.get(region).and_then(|rs| rs.data.get(key))
    }

    // -------------------------------------------------------------------
    // Snapshot operations
    // -------------------------------------------------------------------

    /// Take a full snapshot of a region's current state.
    pub fn snapshot(&self, region: &str) -> Option<StateSnapshot> {
        self.regions.get(region).map(|rs| StateSnapshot {
            region: region.to_string(),
            version: rs.version,
            data: rs.data.iter().map(|(k, v)| (k.clone(), v.value)).collect(),
            tick: self.tick,
        })
    }

    /// Apply a full snapshot to a region, replacing all its state.
    ///
    /// If the region does not exist, it is registered. Conflicts are detected
    /// if the target region already has keys with higher versions.
    pub fn apply_snapshot(&mut self, snapshot: StateSnapshot) -> SyncResult {
        self.register_region(&snapshot.region);

        let rs = self.regions.get_mut(&snapshot.region).unwrap();
        let mut conflicts = 0usize;
        let mut keys_updated = 0usize;

        // Detect conflicts: keys that exist locally with a higher version
        for (key, &new_val) in &snapshot.data {
            if let Some(existing) = rs.data.get(key) {
                if existing.version > snapshot.version {
                    conflicts += 1;
                }
            }
            rs.data.insert(
                key.clone(),
                VersionedValue {
                    value: new_val,
                    version: snapshot.version,
                    updated_at: self.tick,
                },
            );
            keys_updated += 1;
        }

        // Remove keys not present in the snapshot
        let snap_keys: std::collections::HashSet<&String> = snapshot.data.keys().collect();
        rs.data.retain(|k, _| snap_keys.contains(k));

        rs.version = snapshot.version;
        rs.last_sync_tick = self.tick;

        // Update stats
        self.stats.total_syncs += 1;
        self.stats.total_snapshots_applied += 1;
        self.stats.total_keys_synced += keys_updated as u64;
        self.stats.total_conflicts += conflicts as u64;
        self.tick_syncs += 1;
        self.tick_keys += keys_updated as u64;
        self.tick_conflicts += conflicts as u64;
        self.tick_snapshots += 1;

        let outcome = if conflicts > 0 {
            SyncOutcome::ConflictResolved
        } else {
            SyncOutcome::Applied
        };

        SyncResult {
            outcome,
            keys_updated,
            conflicts,
        }
    }

    // -------------------------------------------------------------------
    // Diff operations
    // -------------------------------------------------------------------

    /// Compute a diff of all changes in a region since `base_version`.
    ///
    /// Returns `None` if the region is not registered.
    pub fn diff(&self, region: &str, base_version: u64) -> Option<StateDiff> {
        let rs = self.regions.get(region)?;
        let entries: Vec<DiffEntry> = rs
            .data
            .iter()
            .filter(|(_, v)| v.version > base_version)
            .map(|(k, v)| DiffEntry {
                key: k.clone(),
                value: v.value,
                version: v.version,
            })
            .collect();

        Some(StateDiff {
            region: region.to_string(),
            base_version,
            new_version: rs.version,
            entries,
            tick: self.tick,
        })
    }

    /// Apply a diff to synchronize a region's state.
    ///
    /// Returns `Rejected` if the region does not exist or the diff's base
    /// version is ahead of the region's current version (stale local state
    /// detected — the caller should use a full snapshot instead).
    pub fn apply_diff(&mut self, diff: StateDiff) -> SyncResult {
        if !self.regions.contains_key(&diff.region) {
            self.stats.total_rejections += 1;
            return SyncResult {
                outcome: SyncOutcome::Rejected,
                keys_updated: 0,
                conflicts: 0,
            };
        }

        let rs = self.regions.get_mut(&diff.region).unwrap();

        // If the diff's base version is ahead of our version, reject
        // (the diff was created from a future state we haven't seen).
        if diff.base_version > rs.version {
            self.stats.total_rejections += 1;
            return SyncResult {
                outcome: SyncOutcome::Rejected,
                keys_updated: 0,
                conflicts: 0,
            };
        }

        let mut conflicts = 0usize;
        let mut keys_updated = 0usize;

        for entry in &diff.entries {
            // Conflict: local key has a version newer than the diff's base
            // (i.e. a concurrent local write happened after the diff was created)
            if let Some(existing) = rs.data.get(&entry.key) {
                if existing.version > diff.base_version {
                    conflicts += 1;
                }
            }

            // Apply (last-writer-wins)
            if rs.data.len() >= self.config.max_keys_per_region && !rs.data.contains_key(&entry.key)
            {
                // At capacity — skip new keys
                continue;
            }

            rs.data.insert(
                entry.key.clone(),
                VersionedValue {
                    value: entry.value,
                    version: entry.version,
                    updated_at: self.tick,
                },
            );
            keys_updated += 1;
        }

        if diff.new_version > rs.version {
            rs.version = diff.new_version;
        }
        rs.last_sync_tick = self.tick;

        // Record in history
        if self.diff_history.len() >= self.config.max_diff_history {
            self.diff_history.pop_front();
        }
        self.diff_history.push_back(diff);

        // Update stats
        self.stats.total_syncs += 1;
        self.stats.total_diffs_applied += 1;
        self.stats.total_keys_synced += keys_updated as u64;
        self.stats.total_conflicts += conflicts as u64;
        self.tick_syncs += 1;
        self.tick_keys += keys_updated as u64;
        self.tick_conflicts += conflicts as u64;
        self.tick_diffs += 1;

        let outcome = if conflicts > 0 {
            SyncOutcome::ConflictResolved
        } else {
            SyncOutcome::Applied
        };

        SyncResult {
            outcome,
            keys_updated,
            conflicts,
        }
    }

    // -------------------------------------------------------------------
    // Cross-region sync helpers
    // -------------------------------------------------------------------

    /// Synchronize region `src` → region `dst` by computing a diff from `src`
    /// and applying it to `dst`.
    ///
    /// `base_version` is the version the destination last synced from the
    /// source.
    pub fn sync_regions(&mut self, src: &str, dst: &str, base_version: u64) -> SyncResult {
        let diff = match self.diff(src, base_version) {
            Some(d) => StateDiff {
                region: dst.to_string(),
                ..d
            },
            None => {
                self.stats.total_rejections += 1;
                return SyncResult {
                    outcome: SyncOutcome::Rejected,
                    keys_updated: 0,
                    conflicts: 0,
                };
            }
        };
        self.apply_diff(diff)
    }

    /// Number of diffs in history.
    pub fn diff_history_len(&self) -> usize {
        self.diff_history.len()
    }

    // -------------------------------------------------------------------
    // Tick lifecycle
    // -------------------------------------------------------------------

    /// Advance the sync tick and update diagnostics.
    pub fn tick(&mut self) {
        self.tick += 1;

        let snapshot = TickSnapshot {
            syncs_performed: self.tick_syncs,
            keys_synced: self.tick_keys,
            conflicts_detected: self.tick_conflicts,
            diffs_applied: self.tick_diffs,
            snapshots_applied: self.tick_snapshots,
        };

        // EMA update
        let alpha = self.config.ema_decay;
        let conflict_rate = if self.tick_syncs > 0 {
            self.tick_conflicts as f64 / self.tick_syncs as f64
        } else {
            0.0
        };

        if !self.ema_initialized {
            self.stats.ema_sync_rate = snapshot.syncs_performed as f64;
            self.stats.ema_keys_rate = snapshot.keys_synced as f64;
            self.stats.ema_conflict_rate = conflict_rate;
            self.ema_initialized = true;
        } else {
            self.stats.ema_sync_rate =
                alpha * snapshot.syncs_performed as f64 + (1.0 - alpha) * self.stats.ema_sync_rate;
            self.stats.ema_keys_rate =
                alpha * snapshot.keys_synced as f64 + (1.0 - alpha) * self.stats.ema_keys_rate;
            self.stats.ema_conflict_rate =
                alpha * conflict_rate + (1.0 - alpha) * self.stats.ema_conflict_rate;
        }

        // Windowed
        if self.recent.len() >= self.config.window_size {
            self.recent.pop_front();
        }
        self.recent.push_back(snapshot);

        // Reset per-tick counters
        self.tick_syncs = 0;
        self.tick_keys = 0;
        self.tick_conflicts = 0;
        self.tick_diffs = 0;
        self.tick_snapshots = 0;
    }

    /// Current tick value.
    pub fn current_tick(&self) -> u64 {
        self.tick
    }

    /// Main processing function: advance tick.
    pub fn process(&mut self) -> Result<()> {
        self.tick();
        Ok(())
    }

    // -------------------------------------------------------------------
    // Statistics & diagnostics
    // -------------------------------------------------------------------

    /// Reference to cumulative statistics.
    pub fn stats(&self) -> &StateSyncStats {
        &self.stats
    }

    /// Reference to configuration.
    pub fn config(&self) -> &StateSyncConfig {
        &self.config
    }

    /// EMA-smoothed sync rate (syncs per tick).
    pub fn smoothed_sync_rate(&self) -> f64 {
        self.stats.ema_sync_rate
    }

    /// EMA-smoothed keys synced per tick.
    pub fn smoothed_keys_rate(&self) -> f64 {
        self.stats.ema_keys_rate
    }

    /// EMA-smoothed conflict rate.
    pub fn smoothed_conflict_rate(&self) -> f64 {
        self.stats.ema_conflict_rate
    }

    /// Windowed average sync rate.
    pub fn windowed_sync_rate(&self) -> f64 {
        if self.recent.is_empty() {
            return 0.0;
        }
        let sum: f64 = self.recent.iter().map(|s| s.syncs_performed as f64).sum();
        sum / self.recent.len() as f64
    }

    /// Windowed average keys synced per tick.
    pub fn windowed_keys_rate(&self) -> f64 {
        if self.recent.is_empty() {
            return 0.0;
        }
        let sum: f64 = self.recent.iter().map(|s| s.keys_synced as f64).sum();
        sum / self.recent.len() as f64
    }

    /// Windowed average conflict rate.
    pub fn windowed_conflict_rate(&self) -> f64 {
        if self.recent.is_empty() {
            return 0.0;
        }
        let total_syncs: u64 = self.recent.iter().map(|s| s.syncs_performed).sum();
        let total_conflicts: u64 = self.recent.iter().map(|s| s.conflicts_detected).sum();
        if total_syncs == 0 {
            return 0.0;
        }
        total_conflicts as f64 / total_syncs as f64
    }

    /// Whether conflict rate is trending upward over the window.
    pub fn is_conflict_rate_increasing(&self) -> bool {
        if self.recent.len() < 4 {
            return false;
        }
        let n = self.recent.len();
        let half = n / 2;

        let first_syncs: u64 = self
            .recent
            .iter()
            .take(half)
            .map(|s| s.syncs_performed)
            .sum();
        let first_conflicts: u64 = self
            .recent
            .iter()
            .take(half)
            .map(|s| s.conflicts_detected)
            .sum();
        let second_syncs: u64 = self
            .recent
            .iter()
            .skip(half)
            .map(|s| s.syncs_performed)
            .sum();
        let second_conflicts: u64 = self
            .recent
            .iter()
            .skip(half)
            .map(|s| s.conflicts_detected)
            .sum();

        let first_rate = if first_syncs > 0 {
            first_conflicts as f64 / first_syncs as f64
        } else {
            0.0
        };
        let second_rate = if second_syncs > 0 {
            second_conflicts as f64 / second_syncs as f64
        } else {
            0.0
        };

        second_rate > first_rate
    }

    /// Reset all state.
    pub fn reset(&mut self) {
        self.regions.clear();
        self.diff_history.clear();
        self.tick = 0;
        self.tick_syncs = 0;
        self.tick_keys = 0;
        self.tick_conflicts = 0;
        self.tick_diffs = 0;
        self.tick_snapshots = 0;
        self.ema_initialized = false;
        self.recent.clear();
        self.stats = StateSyncStats::default();
    }
}

// ===========================================================================
// Tests
// ===========================================================================

#[cfg(test)]
mod tests {
    use super::*;

    fn small_config() -> StateSyncConfig {
        StateSyncConfig {
            max_regions: 4,
            max_keys_per_region: 8,
            max_diff_history: 5,
            ema_decay: 0.5,
            window_size: 5,
        }
    }

    // -------------------------------------------------------------------
    // Construction
    // -------------------------------------------------------------------

    #[test]
    fn test_new_default() {
        let ss = StateSync::new();
        assert_eq!(ss.region_count(), 0);
        assert_eq!(ss.current_tick(), 0);
    }

    #[test]
    fn test_with_config() {
        let ss = StateSync::with_config(small_config());
        assert_eq!(ss.config().max_regions, 4);
        assert_eq!(ss.config().max_keys_per_region, 8);
    }

    // -------------------------------------------------------------------
    // Region management
    // -------------------------------------------------------------------

    #[test]
    fn test_register_region() {
        let mut ss = StateSync::with_config(small_config());
        ss.register_region("cortex");
        assert!(ss.has_region("cortex"));
        assert_eq!(ss.region_count(), 1);
        assert_eq!(ss.region_version("cortex"), Some(0));
    }

    #[test]
    fn test_register_duplicate_noop() {
        let mut ss = StateSync::with_config(small_config());
        ss.register_region("cortex");
        ss.register_region("cortex");
        assert_eq!(ss.region_count(), 1);
    }

    #[test]
    fn test_register_at_capacity() {
        let mut ss = StateSync::with_config(small_config()); // max = 4
        for i in 0..5 {
            ss.register_region(format!("r{}", i));
        }
        assert_eq!(ss.region_count(), 4);
    }

    // -------------------------------------------------------------------
    // Local read/write
    // -------------------------------------------------------------------

    #[test]
    fn test_set_and_get() {
        let mut ss = StateSync::with_config(small_config());
        ss.register_region("cortex");
        ss.set("cortex", "price", 100.0);
        assert_eq!(ss.get("cortex", "price"), Some(100.0));
    }

    #[test]
    fn test_set_increments_version() {
        let mut ss = StateSync::with_config(small_config());
        ss.register_region("cortex");
        ss.set("cortex", "a", 1.0);
        assert_eq!(ss.region_version("cortex"), Some(1));
        ss.set("cortex", "b", 2.0);
        assert_eq!(ss.region_version("cortex"), Some(2));
    }

    #[test]
    fn test_set_unknown_region() {
        let mut ss = StateSync::with_config(small_config());
        ss.set("unknown", "key", 1.0); // should not panic
        assert_eq!(ss.get("unknown", "key"), None);
    }

    #[test]
    fn test_get_versioned() {
        let mut ss = StateSync::with_config(small_config());
        ss.register_region("cortex");
        ss.set("cortex", "price", 100.0);
        let vv = ss.get_versioned("cortex", "price").unwrap();
        assert!((vv.value - 100.0).abs() < 1e-10);
        assert_eq!(vv.version, 1);
    }

    #[test]
    fn test_get_missing_key() {
        let mut ss = StateSync::with_config(small_config());
        ss.register_region("cortex");
        assert_eq!(ss.get("cortex", "nonexistent"), None);
    }

    #[test]
    fn test_region_key_count() {
        let mut ss = StateSync::with_config(small_config());
        ss.register_region("cortex");
        ss.set("cortex", "a", 1.0);
        ss.set("cortex", "b", 2.0);
        assert_eq!(ss.region_key_count("cortex"), 2);
    }

    #[test]
    fn test_set_evicts_at_capacity() {
        let mut ss = StateSync::with_config(small_config()); // max_keys = 8
        ss.register_region("cortex");
        for i in 0..10 {
            ss.set("cortex", format!("k{}", i), i as f64);
        }
        assert!(ss.region_key_count("cortex") <= 8);
    }

    #[test]
    fn test_overwrite_existing_key() {
        let mut ss = StateSync::with_config(small_config());
        ss.register_region("cortex");
        ss.set("cortex", "price", 100.0);
        ss.set("cortex", "price", 200.0);
        assert_eq!(ss.get("cortex", "price"), Some(200.0));
        assert_eq!(ss.region_key_count("cortex"), 1);
    }

    #[test]
    fn test_write_stats() {
        let mut ss = StateSync::with_config(small_config());
        ss.register_region("cortex");
        ss.set("cortex", "a", 1.0);
        ss.set("cortex", "b", 2.0);
        assert_eq!(ss.stats().total_writes, 2);
    }

    // -------------------------------------------------------------------
    // Snapshot operations
    // -------------------------------------------------------------------

    #[test]
    fn test_snapshot() {
        let mut ss = StateSync::with_config(small_config());
        ss.register_region("cortex");
        ss.set("cortex", "a", 1.0);
        ss.set("cortex", "b", 2.0);

        let snap = ss.snapshot("cortex").unwrap();
        assert_eq!(snap.region, "cortex");
        assert_eq!(snap.version, 2);
        assert_eq!(snap.data.len(), 2);
        assert_eq!(snap.data.get("a"), Some(&1.0));
        assert_eq!(snap.data.get("b"), Some(&2.0));
    }

    #[test]
    fn test_snapshot_unknown_region() {
        let ss = StateSync::with_config(small_config());
        assert!(ss.snapshot("unknown").is_none());
    }

    #[test]
    fn test_apply_snapshot_clean() {
        let mut ss = StateSync::with_config(small_config());
        ss.register_region("thalamus");

        let mut data = HashMap::new();
        data.insert("x".into(), 10.0);
        data.insert("y".into(), 20.0);

        let snap = StateSnapshot {
            region: "thalamus".into(),
            version: 5,
            data,
            tick: 0,
        };

        let result = ss.apply_snapshot(snap);
        assert_eq!(result.outcome, SyncOutcome::Applied);
        assert_eq!(result.keys_updated, 2);
        assert_eq!(result.conflicts, 0);
        assert_eq!(ss.get("thalamus", "x"), Some(10.0));
        assert_eq!(ss.region_version("thalamus"), Some(5));
        assert_eq!(ss.stats().total_snapshots_applied, 1);
    }

    #[test]
    fn test_apply_snapshot_with_conflict() {
        let mut ss = StateSync::with_config(small_config());
        ss.register_region("cortex");
        // Write a value with version 1
        ss.set("cortex", "price", 100.0);

        // Apply snapshot that is at version 0 (older), which means the local
        // key at version 1 is newer → conflict
        let mut data = HashMap::new();
        data.insert("price".into(), 200.0);
        let snap = StateSnapshot {
            region: "cortex".into(),
            version: 0,
            data,
            tick: 0,
        };

        let result = ss.apply_snapshot(snap);
        assert_eq!(result.outcome, SyncOutcome::ConflictResolved);
        assert_eq!(result.conflicts, 1);
        // Last-writer-wins: snapshot value applied
        assert_eq!(ss.get("cortex", "price"), Some(200.0));
    }

    #[test]
    fn test_apply_snapshot_creates_region() {
        let mut ss = StateSync::with_config(small_config());
        let mut data = HashMap::new();
        data.insert("k".into(), 1.0);
        let snap = StateSnapshot {
            region: "new_region".into(),
            version: 1,
            data,
            tick: 0,
        };
        let result = ss.apply_snapshot(snap);
        assert_eq!(result.outcome, SyncOutcome::Applied);
        assert!(ss.has_region("new_region"));
        assert_eq!(ss.get("new_region", "k"), Some(1.0));
    }

    #[test]
    fn test_apply_snapshot_removes_old_keys() {
        let mut ss = StateSync::with_config(small_config());
        ss.register_region("cortex");
        ss.set("cortex", "a", 1.0);
        ss.set("cortex", "b", 2.0);

        // Snapshot only has key "a"
        let mut data = HashMap::new();
        data.insert("a".into(), 10.0);
        let snap = StateSnapshot {
            region: "cortex".into(),
            version: 5,
            data,
            tick: 0,
        };
        ss.apply_snapshot(snap);
        assert_eq!(ss.get("cortex", "a"), Some(10.0));
        assert_eq!(ss.get("cortex", "b"), None); // removed
    }

    // -------------------------------------------------------------------
    // Diff operations
    // -------------------------------------------------------------------

    #[test]
    fn test_diff_all_changes() {
        let mut ss = StateSync::with_config(small_config());
        ss.register_region("cortex");
        ss.set("cortex", "a", 1.0);
        ss.set("cortex", "b", 2.0);

        let diff = ss.diff("cortex", 0).unwrap();
        assert_eq!(diff.region, "cortex");
        assert_eq!(diff.base_version, 0);
        assert_eq!(diff.new_version, 2);
        assert_eq!(diff.entries.len(), 2);
    }

    #[test]
    fn test_diff_partial() {
        let mut ss = StateSync::with_config(small_config());
        ss.register_region("cortex");
        ss.set("cortex", "a", 1.0); // version 1
        ss.set("cortex", "b", 2.0); // version 2
        ss.set("cortex", "c", 3.0); // version 3

        // Only changes after version 2
        let diff = ss.diff("cortex", 2).unwrap();
        assert_eq!(diff.entries.len(), 1);
        assert_eq!(diff.entries[0].key, "c");
    }

    #[test]
    fn test_diff_no_changes() {
        let mut ss = StateSync::with_config(small_config());
        ss.register_region("cortex");
        ss.set("cortex", "a", 1.0);

        let diff = ss.diff("cortex", 1).unwrap();
        assert_eq!(diff.entries.len(), 0);
    }

    #[test]
    fn test_diff_unknown_region() {
        let ss = StateSync::with_config(small_config());
        assert!(ss.diff("unknown", 0).is_none());
    }

    #[test]
    fn test_apply_diff_clean() {
        let mut ss = StateSync::with_config(small_config());
        ss.register_region("thalamus");

        let diff = StateDiff {
            region: "thalamus".into(),
            base_version: 0,
            new_version: 2,
            entries: vec![
                DiffEntry {
                    key: "x".into(),
                    value: 10.0,
                    version: 1,
                },
                DiffEntry {
                    key: "y".into(),
                    value: 20.0,
                    version: 2,
                },
            ],
            tick: 0,
        };

        let result = ss.apply_diff(diff);
        assert_eq!(result.outcome, SyncOutcome::Applied);
        assert_eq!(result.keys_updated, 2);
        assert_eq!(result.conflicts, 0);
        assert_eq!(ss.get("thalamus", "x"), Some(10.0));
        assert_eq!(ss.get("thalamus", "y"), Some(20.0));
        assert_eq!(ss.region_version("thalamus"), Some(2));
        assert_eq!(ss.stats().total_diffs_applied, 1);
    }

    #[test]
    fn test_apply_diff_with_conflict() {
        let mut ss = StateSync::with_config(small_config());
        ss.register_region("cortex");
        ss.set("cortex", "price", 100.0); // version 1

        // Diff based on version 0, modifying "price" which already changed
        let diff = StateDiff {
            region: "cortex".into(),
            base_version: 0,
            new_version: 1,
            entries: vec![DiffEntry {
                key: "price".into(),
                value: 200.0,
                version: 1,
            }],
            tick: 0,
        };

        let result = ss.apply_diff(diff);
        assert_eq!(result.outcome, SyncOutcome::ConflictResolved);
        assert_eq!(result.conflicts, 1);
        // Last-writer-wins
        assert_eq!(ss.get("cortex", "price"), Some(200.0));
    }

    #[test]
    fn test_apply_diff_rejected_unknown_region() {
        let mut ss = StateSync::with_config(small_config());
        let diff = StateDiff {
            region: "unknown".into(),
            base_version: 0,
            new_version: 1,
            entries: vec![],
            tick: 0,
        };
        let result = ss.apply_diff(diff);
        assert_eq!(result.outcome, SyncOutcome::Rejected);
        assert_eq!(ss.stats().total_rejections, 1);
    }

    #[test]
    fn test_apply_diff_rejected_future_base() {
        let mut ss = StateSync::with_config(small_config());
        ss.register_region("cortex"); // version 0

        let diff = StateDiff {
            region: "cortex".into(),
            base_version: 10, // ahead of current version 0
            new_version: 11,
            entries: vec![DiffEntry {
                key: "x".into(),
                value: 1.0,
                version: 11,
            }],
            tick: 0,
        };

        let result = ss.apply_diff(diff);
        assert_eq!(result.outcome, SyncOutcome::Rejected);
    }

    #[test]
    fn test_diff_history() {
        let mut ss = StateSync::with_config(small_config()); // max_diff_history = 5
        ss.register_region("cortex");

        for i in 0..7 {
            let diff = StateDiff {
                region: "cortex".into(),
                base_version: i,
                new_version: i + 1,
                entries: vec![DiffEntry {
                    key: format!("k{}", i),
                    value: i as f64,
                    version: i + 1,
                }],
                tick: i,
            };
            ss.apply_diff(diff);
        }

        assert!(ss.diff_history_len() <= 5);
    }

    // -------------------------------------------------------------------
    // Cross-region sync
    // -------------------------------------------------------------------

    #[test]
    fn test_sync_regions() {
        let mut ss = StateSync::with_config(small_config());
        ss.register_region("cortex");
        ss.register_region("thalamus");

        ss.set("cortex", "signal", 42.0);
        ss.set("cortex", "confidence", 0.9);

        let result = ss.sync_regions("cortex", "thalamus", 0);
        assert_eq!(result.outcome, SyncOutcome::Applied);
        assert_eq!(result.keys_updated, 2);
        assert_eq!(ss.get("thalamus", "signal"), Some(42.0));
        assert_eq!(ss.get("thalamus", "confidence"), Some(0.9));
    }

    #[test]
    fn test_sync_regions_unknown_src() {
        let mut ss = StateSync::with_config(small_config());
        ss.register_region("thalamus");
        let result = ss.sync_regions("unknown", "thalamus", 0);
        assert_eq!(result.outcome, SyncOutcome::Rejected);
    }

    // -------------------------------------------------------------------
    // Tick & process
    // -------------------------------------------------------------------

    #[test]
    fn test_tick_increments() {
        let mut ss = StateSync::new();
        ss.tick();
        ss.tick();
        assert_eq!(ss.current_tick(), 2);
    }

    #[test]
    fn test_process() {
        let mut ss = StateSync::new();
        assert!(ss.process().is_ok());
        assert_eq!(ss.current_tick(), 1);
    }

    // -------------------------------------------------------------------
    // EMA diagnostics
    // -------------------------------------------------------------------

    #[test]
    fn test_ema_initialises_on_first_tick() {
        let mut ss = StateSync::with_config(small_config());
        ss.register_region("cortex");
        ss.register_region("thalamus");
        ss.set("cortex", "x", 1.0);
        ss.sync_regions("cortex", "thalamus", 0);
        ss.tick();
        assert!(ss.smoothed_sync_rate() > 0.0);
        assert!(ss.smoothed_keys_rate() > 0.0);
    }

    #[test]
    fn test_ema_blends_on_subsequent_ticks() {
        let mut ss = StateSync::with_config(StateSyncConfig {
            ema_decay: 0.5,
            ..StateSyncConfig::default()
        });
        ss.register_region("cortex");
        ss.register_region("thalamus");

        // Tick 1: 1 sync
        ss.set("cortex", "x", 1.0);
        ss.sync_regions("cortex", "thalamus", 0);
        ss.tick(); // ema_sync = 1.0

        // Tick 2: 0 syncs
        ss.tick(); // ema_sync = 0.5 * 0 + 0.5 * 1.0 = 0.5

        assert!((ss.smoothed_sync_rate() - 0.5).abs() < 1e-10);
    }

    // -------------------------------------------------------------------
    // Windowed diagnostics
    // -------------------------------------------------------------------

    #[test]
    fn test_windowed_sync_rate() {
        let mut ss = StateSync::with_config(small_config());
        ss.register_region("cortex");
        ss.register_region("thalamus");

        for _ in 0..3 {
            ss.set("cortex", "x", 1.0);
            ss.sync_regions("cortex", "thalamus", 0);
            ss.tick();
        }
        assert!(ss.windowed_sync_rate() > 0.0);
    }

    #[test]
    fn test_windowed_empty() {
        let ss = StateSync::new();
        assert!((ss.windowed_sync_rate()).abs() < 1e-10);
        assert!((ss.windowed_keys_rate()).abs() < 1e-10);
        assert!((ss.windowed_conflict_rate()).abs() < 1e-10);
    }

    #[test]
    fn test_windowed_conflict_rate() {
        let mut ss = StateSync::with_config(small_config());
        ss.register_region("cortex");

        // Create a conflict scenario
        ss.set("cortex", "price", 100.0); // version 1

        let diff = StateDiff {
            region: "cortex".into(),
            base_version: 0,
            new_version: 1,
            entries: vec![DiffEntry {
                key: "price".into(),
                value: 200.0,
                version: 1,
            }],
            tick: 0,
        };
        ss.apply_diff(diff);
        ss.tick();

        assert!(ss.windowed_conflict_rate() > 0.0);
    }

    #[test]
    fn test_is_conflict_rate_increasing_insufficient_data() {
        let mut ss = StateSync::new();
        ss.tick();
        assert!(!ss.is_conflict_rate_increasing());
    }

    // -------------------------------------------------------------------
    // Reset
    // -------------------------------------------------------------------

    #[test]
    fn test_reset() {
        let mut ss = StateSync::with_config(small_config());
        ss.register_region("cortex");
        ss.set("cortex", "x", 1.0);
        ss.tick();

        ss.reset();
        assert_eq!(ss.region_count(), 0);
        assert_eq!(ss.current_tick(), 0);
        assert_eq!(ss.stats().total_writes, 0);
        assert_eq!(ss.diff_history_len(), 0);
    }

    // -------------------------------------------------------------------
    // Full lifecycle
    // -------------------------------------------------------------------

    #[test]
    fn test_full_lifecycle() {
        let mut ss = StateSync::with_config(small_config());

        // Register regions
        ss.register_region("cortex");
        ss.register_region("thalamus");
        ss.register_region("cerebellum");

        // Cortex writes data
        ss.set("cortex", "signal_strength", 0.85);
        ss.set("cortex", "confidence", 0.9);
        ss.set("cortex", "position_size", 1.5);

        // Sync cortex → thalamus
        let result = ss.sync_regions("cortex", "thalamus", 0);
        assert_eq!(result.outcome, SyncOutcome::Applied);
        assert_eq!(result.keys_updated, 3);

        // Thalamus adds its own data
        ss.set("thalamus", "attention_score", 0.7);

        // Tick
        ss.tick();

        // Take a snapshot of thalamus and apply to cerebellum
        let snap = ss.snapshot("thalamus").unwrap();
        let snap_for_cerebellum = StateSnapshot {
            region: "cerebellum".into(),
            ..snap
        };
        let result = ss.apply_snapshot(snap_for_cerebellum);
        assert_eq!(result.outcome, SyncOutcome::Applied);

        ss.tick();

        // Verify state
        assert_eq!(ss.get("cerebellum", "signal_strength"), Some(0.85));
        assert_eq!(ss.get("cerebellum", "attention_score"), Some(0.7));
        assert_eq!(ss.current_tick(), 2);
        assert!(ss.stats().total_syncs > 0);
        assert!(ss.stats().total_keys_synced > 0);
        assert!(ss.smoothed_sync_rate() > 0.0);
    }

    #[test]
    fn test_window_rolls() {
        let mut ss = StateSync::with_config(small_config()); // window = 5
        for _ in 0..10 {
            ss.tick();
        }
        assert!(ss.recent.len() <= ss.config.window_size);
    }
}
