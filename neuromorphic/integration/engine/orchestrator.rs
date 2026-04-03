//! Main orchestration loop
//!
//! Part of the Integration region — Engine component.
//!
//! `Orchestrator` coordinates the execution of brain regions in dependency-aware
//! order, schedules processing rounds, tracks per-region latencies, detects
//! stalls, and exposes EMA-smoothed and windowed diagnostics.

use crate::common::{Error, Result};
use std::collections::{HashMap, HashSet, VecDeque};

// ---------------------------------------------------------------------------
// Configuration
// ---------------------------------------------------------------------------

/// Configuration for the orchestrator.
#[derive(Debug, Clone)]
pub struct OrchestratorConfig {
    /// Maximum number of regions that can be registered.
    pub max_regions: usize,
    /// Maximum number of rounds to retain in history.
    pub max_round_history: usize,
    /// Latency (in simulated µs) above which a region is considered slow.
    pub slow_region_threshold_us: f64,
    /// Number of consecutive stalls before a region is marked degraded.
    pub stall_tolerance: u32,
    /// EMA decay factor (0 < α < 1). Higher → faster adaptation.
    pub ema_decay: f64,
    /// Window size for rolling diagnostics.
    pub window_size: usize,
}

impl Default for OrchestratorConfig {
    fn default() -> Self {
        Self {
            max_regions: 64,
            max_round_history: 200,
            slow_region_threshold_us: 5000.0,
            stall_tolerance: 3,
            ema_decay: 0.1,
            window_size: 50,
        }
    }
}

// ---------------------------------------------------------------------------
// Region descriptor
// ---------------------------------------------------------------------------

/// Execution priority tier — higher-priority regions execute first within
/// each dependency layer.
#[derive(Debug, Clone, Copy, PartialEq, Eq, PartialOrd, Ord, Hash)]
pub enum Priority {
    /// Critical-path regions (e.g. thalamus, risk).
    Critical = 0,
    /// Standard processing regions.
    Normal = 1,
    /// Background / housekeeping regions.
    Low = 2,
}

/// Status of a registered region within the orchestrator.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum RegionStatus {
    /// Region is idle, waiting to be scheduled.
    Idle,
    /// Region is currently executing within a round.
    Running,
    /// Region completed its most recent execution successfully.
    Completed,
    /// Region is stalled (exceeded stall tolerance).
    Stalled,
    /// Region has been disabled and will be skipped.
    Disabled,
}

/// Descriptor for a registered brain region.
#[derive(Debug, Clone)]
pub struct RegionDescriptor {
    /// Unique region name.
    pub name: String,
    /// Execution priority.
    pub priority: Priority,
    /// Names of regions that must execute before this one.
    pub dependencies: Vec<String>,
    /// Current status.
    pub status: RegionStatus,
    /// Last recorded latency in simulated µs.
    pub last_latency_us: f64,
    /// Cumulative number of successful executions.
    pub executions: u64,
    /// Cumulative number of failures / stalls.
    pub failures: u64,
    /// Consecutive stall counter (resets on success).
    pub consecutive_stalls: u32,
    /// EMA-smoothed latency.
    pub ema_latency_us: f64,
}

// ---------------------------------------------------------------------------
// Round record
// ---------------------------------------------------------------------------

/// Record of a single orchestration round.
#[derive(Debug, Clone)]
pub struct RoundRecord {
    /// Round number (monotonically increasing).
    pub round: u64,
    /// Tick at which the round was executed.
    pub tick: u64,
    /// Ordered list of region names as they were executed.
    pub execution_order: Vec<String>,
    /// Per-region latency for this round.
    pub latencies: HashMap<String, f64>,
    /// Total round latency (sum of all region latencies).
    pub total_latency_us: f64,
    /// Number of regions that were skipped (disabled / stalled).
    pub skipped: usize,
}

// ---------------------------------------------------------------------------
// Tick snapshot (windowed diagnostics)
// ---------------------------------------------------------------------------

#[derive(Debug, Clone)]
struct TickSnapshot {
    round_latency_us: f64,
    regions_executed: usize,
    regions_skipped: usize,
    slow_region_count: usize,
}

// ---------------------------------------------------------------------------
// Statistics
// ---------------------------------------------------------------------------

/// Cumulative statistics for the orchestrator.
#[derive(Debug, Clone)]
pub struct OrchestratorStats {
    /// Total rounds executed.
    pub total_rounds: u64,
    /// Total region executions across all rounds.
    pub total_region_executions: u64,
    /// Total region failures across all rounds.
    pub total_region_failures: u64,
    /// Total regions skipped across all rounds.
    pub total_skipped: u64,
    /// EMA-smoothed round latency.
    pub ema_round_latency_us: f64,
    /// EMA-smoothed count of regions executed per round.
    pub ema_regions_per_round: f64,
    /// EMA-smoothed fraction of slow regions per round.
    pub ema_slow_fraction: f64,
}

impl Default for OrchestratorStats {
    fn default() -> Self {
        Self {
            total_rounds: 0,
            total_region_executions: 0,
            total_region_failures: 0,
            total_skipped: 0,
            ema_round_latency_us: 0.0,
            ema_regions_per_round: 0.0,
            ema_slow_fraction: 0.0,
        }
    }
}

// ---------------------------------------------------------------------------
// Orchestrator
// ---------------------------------------------------------------------------

/// Main orchestration loop coordinating region execution.
#[derive(Debug, Clone)]
pub struct Orchestrator {
    config: OrchestratorConfig,
    /// Registered regions keyed by name.
    regions: HashMap<String, RegionDescriptor>,
    /// Insertion-order tracking for deterministic iteration.
    region_order: Vec<String>,
    /// History of executed rounds.
    round_history: VecDeque<RoundRecord>,
    /// Current tick counter.
    tick: u64,
    /// Current round counter.
    round: u64,
    /// Whether EMA has been initialised.
    ema_initialized: bool,
    /// Rolling window of per-tick snapshots.
    recent: VecDeque<TickSnapshot>,
    /// Cumulative stats.
    stats: OrchestratorStats,
}

impl Default for Orchestrator {
    fn default() -> Self {
        Self::new()
    }
}

impl Orchestrator {
    // -- Construction -------------------------------------------------------

    /// Create a new orchestrator with default configuration.
    pub fn new() -> Self {
        Self::with_config(OrchestratorConfig::default())
    }

    /// Create a new orchestrator with the given configuration.
    pub fn with_config(config: OrchestratorConfig) -> Self {
        Self {
            regions: HashMap::new(),
            region_order: Vec::new(),
            round_history: VecDeque::with_capacity(config.max_round_history.min(256)),
            tick: 0,
            round: 0,
            ema_initialized: false,
            recent: VecDeque::with_capacity(config.window_size.min(256)),
            stats: OrchestratorStats::default(),
            config,
        }
    }

    // -- Region management --------------------------------------------------

    /// Register a new region. Returns an error if the region already exists or
    /// the maximum number of regions has been reached.
    pub fn register_region(
        &mut self,
        name: impl Into<String>,
        priority: Priority,
        dependencies: Vec<String>,
    ) -> Result<()> {
        let name = name.into();
        if self.regions.contains_key(&name) {
            return Err(Error::Configuration(format!(
                "Region '{}' is already registered",
                name
            )));
        }
        if self.regions.len() >= self.config.max_regions {
            return Err(Error::Configuration(format!(
                "Maximum region count ({}) reached",
                self.config.max_regions
            )));
        }

        self.regions.insert(
            name.clone(),
            RegionDescriptor {
                name: name.clone(),
                priority,
                dependencies,
                status: RegionStatus::Idle,
                last_latency_us: 0.0,
                executions: 0,
                failures: 0,
                consecutive_stalls: 0,
                ema_latency_us: 0.0,
            },
        );
        self.region_order.push(name);
        Ok(())
    }

    /// Disable a region so it is skipped during rounds.
    pub fn disable_region(&mut self, name: &str) -> Result<()> {
        let region = self
            .regions
            .get_mut(name)
            .ok_or_else(|| Error::Configuration(format!("Unknown region '{}'", name)))?;
        region.status = RegionStatus::Disabled;
        Ok(())
    }

    /// Re-enable a previously disabled region.
    pub fn enable_region(&mut self, name: &str) -> Result<()> {
        let region = self
            .regions
            .get_mut(name)
            .ok_or_else(|| Error::Configuration(format!("Unknown region '{}'", name)))?;
        if region.status == RegionStatus::Disabled || region.status == RegionStatus::Stalled {
            region.status = RegionStatus::Idle;
            region.consecutive_stalls = 0;
        }
        Ok(())
    }

    /// Look up a region descriptor by name.
    pub fn region(&self, name: &str) -> Option<&RegionDescriptor> {
        self.regions.get(name)
    }

    /// Return the number of registered regions.
    pub fn region_count(&self) -> usize {
        self.regions.len()
    }

    /// Return the names of all registered regions.
    pub fn region_names(&self) -> Vec<&str> {
        self.region_order.iter().map(|s| s.as_str()).collect()
    }

    // -- Dependency-aware ordering ------------------------------------------

    /// Compute a topologically sorted execution order that respects
    /// dependencies and sorts within each layer by priority then name.
    fn compute_execution_order(&self) -> Result<Vec<String>> {
        // Kahn's algorithm for topological sort.
        let active: HashSet<&str> = self
            .regions
            .values()
            .filter(|r| r.status != RegionStatus::Disabled && r.status != RegionStatus::Stalled)
            .map(|r| r.name.as_str())
            .collect();

        // Build in-degree map restricted to active regions.
        let mut in_degree: HashMap<&str, usize> = HashMap::new();
        let mut dependents: HashMap<&str, Vec<&str>> = HashMap::new();

        for name in &active {
            in_degree.entry(name).or_insert(0);
        }

        for name in &active {
            let region = &self.regions[*name];
            for dep in &region.dependencies {
                if active.contains(dep.as_str()) {
                    *in_degree.entry(name).or_insert(0) += 1;
                    dependents.entry(dep.as_str()).or_default().push(name);
                }
                // If the dependency is not active we ignore it (already ran or disabled).
            }
        }

        let mut result: Vec<String> = Vec::with_capacity(active.len());
        let mut queue: Vec<&str> = in_degree
            .iter()
            .filter(|(_, &deg)| deg == 0)
            .map(|(&name, _)| name)
            .collect();

        // Sort the initial queue by (priority, name) for determinism.
        queue.sort_by(|a, b| {
            let pa = self.regions[*a].priority;
            let pb = self.regions[*b].priority;
            pa.cmp(&pb).then_with(|| a.cmp(b))
        });

        while let Some(current) = queue.first().copied() {
            queue.remove(0);
            result.push(current.to_string());

            if let Some(deps) = dependents.get(current) {
                let mut next_layer: Vec<&str> = Vec::new();
                for &dep in deps {
                    let deg = in_degree.get_mut(dep).unwrap();
                    *deg -= 1;
                    if *deg == 0 {
                        next_layer.push(dep);
                    }
                }
                // Sort new entries by priority then name.
                next_layer.sort_by(|a, b| {
                    let pa = self.regions[*a].priority;
                    let pb = self.regions[*b].priority;
                    pa.cmp(&pb).then_with(|| a.cmp(b))
                });
                // Insert at front so BFS layers stay contiguous.
                for (i, item) in next_layer.into_iter().enumerate() {
                    queue.insert(i, item);
                }
            }
        }

        if result.len() != active.len() {
            return Err(Error::Configuration(
                "Cycle detected in region dependencies".to_string(),
            ));
        }

        Ok(result)
    }

    // -- Execution ----------------------------------------------------------

    /// Execute a single orchestration round. Each active region is "executed"
    /// by calling the provided closure with the region name. The closure
    /// returns either the simulated latency in µs (`Ok(f64)`) or an error
    /// indicating a stall/failure.
    pub fn execute_round<F>(&mut self, mut exec_fn: F) -> Result<&RoundRecord>
    where
        F: FnMut(&str) -> Result<f64>,
    {
        let order = self.compute_execution_order()?;
        let mut latencies: HashMap<String, f64> = HashMap::new();
        let mut total_latency = 0.0;
        let mut skipped = 0usize;
        let threshold = self.config.slow_region_threshold_us;
        let stall_tolerance = self.config.stall_tolerance;
        let decay = self.config.ema_decay;

        for name in &order {
            // Mark running.
            if let Some(r) = self.regions.get_mut(name.as_str()) {
                r.status = RegionStatus::Running;
            }

            match exec_fn(name.as_str()) {
                Ok(latency_us) => {
                    let region = self.regions.get_mut(name.as_str()).unwrap();
                    region.last_latency_us = latency_us;
                    region.executions += 1;
                    region.consecutive_stalls = 0;
                    region.status = RegionStatus::Completed;

                    // Update per-region EMA latency.
                    if region.executions == 1 {
                        region.ema_latency_us = latency_us;
                    } else {
                        region.ema_latency_us =
                            decay * latency_us + (1.0 - decay) * region.ema_latency_us;
                    }

                    latencies.insert(name.clone(), latency_us);
                    total_latency += latency_us;
                    self.stats.total_region_executions += 1;
                }
                Err(_) => {
                    let region = self.regions.get_mut(name.as_str()).unwrap();
                    region.failures += 1;
                    region.consecutive_stalls += 1;
                    self.stats.total_region_failures += 1;

                    if region.consecutive_stalls >= stall_tolerance {
                        region.status = RegionStatus::Stalled;
                    } else {
                        region.status = RegionStatus::Idle;
                    }

                    skipped += 1;
                }
            }
        }

        let slow_count = latencies
            .values()
            .filter(|&&lat| lat > threshold)
            .count();
        let executed = latencies.len();

        self.round += 1;
        self.stats.total_rounds += 1;
        self.stats.total_skipped += skipped as u64;

        let record = RoundRecord {
            round: self.round,
            tick: self.tick,
            execution_order: order,
            latencies,
            total_latency_us: total_latency,
            skipped,
        };

        // Windowed snapshot.
        let snapshot = TickSnapshot {
            round_latency_us: total_latency,
            regions_executed: executed,
            regions_skipped: skipped,
            slow_region_count: slow_count,
        };

        if self.recent.len() >= self.config.window_size {
            self.recent.pop_front();
        }
        self.recent.push_back(snapshot);

        // EMA update.
        let slow_frac = if executed > 0 {
            slow_count as f64 / executed as f64
        } else {
            0.0
        };

        if !self.ema_initialized {
            self.stats.ema_round_latency_us = total_latency;
            self.stats.ema_regions_per_round = executed as f64;
            self.stats.ema_slow_fraction = slow_frac;
            self.ema_initialized = true;
        } else {
            self.stats.ema_round_latency_us =
                decay * total_latency + (1.0 - decay) * self.stats.ema_round_latency_us;
            self.stats.ema_regions_per_round =
                decay * executed as f64 + (1.0 - decay) * self.stats.ema_regions_per_round;
            self.stats.ema_slow_fraction =
                decay * slow_frac + (1.0 - decay) * self.stats.ema_slow_fraction;
        }

        // Store round.
        if self.round_history.len() >= self.config.max_round_history {
            self.round_history.pop_front();
        }
        self.round_history.push_back(record);

        Ok(self.round_history.back().unwrap())
    }

    // -- Tick ---------------------------------------------------------------

    /// Advance the internal tick counter. Unlike `execute_round`, this only
    /// bumps the tick — call `execute_round` separately to run a round.
    pub fn tick(&mut self) {
        self.tick += 1;
        // Reset completed regions back to idle.
        for region in self.regions.values_mut() {
            if region.status == RegionStatus::Completed {
                region.status = RegionStatus::Idle;
            }
        }
    }

    /// Current tick value.
    pub fn current_tick(&self) -> u64 {
        self.tick
    }

    /// Current round value.
    pub fn current_round(&self) -> u64 {
        self.round
    }

    /// Convenience: advance tick and execute a round in one call.
    pub fn process<F>(&mut self, exec_fn: F) -> Result<()>
    where
        F: FnMut(&str) -> Result<f64>,
    {
        self.tick();
        self.execute_round(exec_fn)?;
        Ok(())
    }

    // -- Queries ------------------------------------------------------------

    /// Return the most recent round record, if any.
    pub fn last_round(&self) -> Option<&RoundRecord> {
        self.round_history.back()
    }

    /// Return all stored round records.
    pub fn round_history(&self) -> &VecDeque<RoundRecord> {
        &self.round_history
    }

    /// Return the names of regions currently marked as stalled.
    pub fn stalled_regions(&self) -> Vec<&str> {
        self.regions
            .values()
            .filter(|r| r.status == RegionStatus::Stalled)
            .map(|r| r.name.as_str())
            .collect()
    }

    /// Return the names of regions whose EMA latency exceeds the slow
    /// threshold.
    pub fn slow_regions(&self) -> Vec<&str> {
        self.regions
            .values()
            .filter(|r| {
                r.status != RegionStatus::Disabled
                    && r.ema_latency_us > self.config.slow_region_threshold_us
            })
            .map(|r| r.name.as_str())
            .collect()
    }

    // -- Diagnostics --------------------------------------------------------

    /// Get a reference to the cumulative statistics.
    pub fn stats(&self) -> &OrchestratorStats {
        &self.stats
    }

    /// Get a reference to the configuration.
    pub fn config(&self) -> &OrchestratorConfig {
        &self.config
    }

    /// EMA-smoothed round latency.
    pub fn smoothed_round_latency(&self) -> f64 {
        self.stats.ema_round_latency_us
    }

    /// EMA-smoothed regions executed per round.
    pub fn smoothed_regions_per_round(&self) -> f64 {
        self.stats.ema_regions_per_round
    }

    /// EMA-smoothed slow-region fraction.
    pub fn smoothed_slow_fraction(&self) -> f64 {
        self.stats.ema_slow_fraction
    }

    /// Windowed average round latency.
    pub fn windowed_round_latency(&self) -> Option<f64> {
        if self.recent.is_empty() {
            return None;
        }
        let sum: f64 = self.recent.iter().map(|s| s.round_latency_us).sum();
        Some(sum / self.recent.len() as f64)
    }

    /// Windowed average regions executed per round.
    pub fn windowed_regions_per_round(&self) -> Option<f64> {
        if self.recent.is_empty() {
            return None;
        }
        let sum: f64 = self.recent.iter().map(|s| s.regions_executed as f64).sum();
        Some(sum / self.recent.len() as f64)
    }

    /// Windowed average slow-region count.
    pub fn windowed_slow_count(&self) -> Option<f64> {
        if self.recent.is_empty() {
            return None;
        }
        let sum: f64 = self
            .recent
            .iter()
            .map(|s| s.slow_region_count as f64)
            .sum();
        Some(sum / self.recent.len() as f64)
    }

    /// Whether the average round latency is trending upward over the window
    /// (compares the second half to the first half).
    pub fn is_latency_increasing(&self) -> bool {
        if self.recent.len() < 4 {
            return false;
        }
        let mid = self.recent.len() / 2;
        let first_half: f64 = self.recent.iter().take(mid).map(|s| s.round_latency_us).sum::<f64>()
            / mid as f64;
        let second_half: f64 = self.recent.iter().skip(mid).map(|s| s.round_latency_us).sum::<f64>()
            / (self.recent.len() - mid) as f64;
        second_half > first_half * 1.05
    }

    // -- Reset --------------------------------------------------------------

    /// Reset all state while preserving configuration and registered regions.
    pub fn reset(&mut self) {
        self.tick = 0;
        self.round = 0;
        self.ema_initialized = false;
        self.recent.clear();
        self.round_history.clear();
        self.stats = OrchestratorStats::default();
        for region in self.regions.values_mut() {
            region.status = RegionStatus::Idle;
            region.last_latency_us = 0.0;
            region.executions = 0;
            region.failures = 0;
            region.consecutive_stalls = 0;
            region.ema_latency_us = 0.0;
        }
    }
}

// ===========================================================================
// Tests
// ===========================================================================

#[cfg(test)]
mod tests {
    use super::*;

    fn small_config() -> OrchestratorConfig {
        OrchestratorConfig {
            max_regions: 8,
            max_round_history: 10,
            slow_region_threshold_us: 100.0,
            stall_tolerance: 2,
            ema_decay: 0.5,
            window_size: 5,
        }
    }

    // -- Construction -------------------------------------------------------

    #[test]
    fn test_new_default() {
        let o = Orchestrator::new();
        assert_eq!(o.region_count(), 0);
        assert_eq!(o.current_tick(), 0);
        assert_eq!(o.current_round(), 0);
    }

    #[test]
    fn test_with_config() {
        let o = Orchestrator::with_config(small_config());
        assert_eq!(o.config().max_regions, 8);
    }

    // -- Region management --------------------------------------------------

    #[test]
    fn test_register_region() {
        let mut o = Orchestrator::with_config(small_config());
        o.register_region("cortex", Priority::Normal, vec![]).unwrap();
        assert_eq!(o.region_count(), 1);
        assert!(o.region("cortex").is_some());
    }

    #[test]
    fn test_register_duplicate_fails() {
        let mut o = Orchestrator::with_config(small_config());
        o.register_region("cortex", Priority::Normal, vec![]).unwrap();
        assert!(o.register_region("cortex", Priority::Normal, vec![]).is_err());
    }

    #[test]
    fn test_register_at_capacity() {
        let mut o = Orchestrator::with_config(small_config());
        for i in 0..8 {
            o.register_region(format!("r{}", i), Priority::Normal, vec![])
                .unwrap();
        }
        assert!(o.register_region("overflow", Priority::Normal, vec![]).is_err());
    }

    #[test]
    fn test_disable_enable() {
        let mut o = Orchestrator::with_config(small_config());
        o.register_region("a", Priority::Normal, vec![]).unwrap();
        o.disable_region("a").unwrap();
        assert_eq!(o.region("a").unwrap().status, RegionStatus::Disabled);
        o.enable_region("a").unwrap();
        assert_eq!(o.region("a").unwrap().status, RegionStatus::Idle);
    }

    #[test]
    fn test_disable_unknown_fails() {
        let mut o = Orchestrator::with_config(small_config());
        assert!(o.disable_region("nope").is_err());
    }

    #[test]
    fn test_region_names() {
        let mut o = Orchestrator::with_config(small_config());
        o.register_region("b", Priority::Normal, vec![]).unwrap();
        o.register_region("a", Priority::Normal, vec![]).unwrap();
        let names = o.region_names();
        assert_eq!(names, vec!["b", "a"]); // insertion order
    }

    // -- Dependency ordering ------------------------------------------------

    #[test]
    fn test_execution_order_no_deps() {
        let mut o = Orchestrator::with_config(small_config());
        o.register_region("zeta", Priority::Low, vec![]).unwrap();
        o.register_region("alpha", Priority::Critical, vec![]).unwrap();
        o.register_region("beta", Priority::Normal, vec![]).unwrap();

        let order = o.compute_execution_order().unwrap();
        // Critical first, then Normal, then Low. Alphabetic within tier.
        assert_eq!(order, vec!["alpha", "beta", "zeta"]);
    }

    #[test]
    fn test_execution_order_with_deps() {
        let mut o = Orchestrator::with_config(small_config());
        o.register_region("output", Priority::Normal, vec!["middle".into()])
            .unwrap();
        o.register_region("middle", Priority::Normal, vec!["input".into()])
            .unwrap();
        o.register_region("input", Priority::Normal, vec![])
            .unwrap();

        let order = o.compute_execution_order().unwrap();
        assert_eq!(order, vec!["input", "middle", "output"]);
    }

    #[test]
    fn test_execution_order_skips_disabled() {
        let mut o = Orchestrator::with_config(small_config());
        o.register_region("a", Priority::Normal, vec![]).unwrap();
        o.register_region("b", Priority::Normal, vec![]).unwrap();
        o.disable_region("b").unwrap();
        let order = o.compute_execution_order().unwrap();
        assert_eq!(order, vec!["a"]);
    }

    #[test]
    fn test_cycle_detection() {
        let mut o = Orchestrator::with_config(small_config());
        o.register_region("a", Priority::Normal, vec!["b".into()])
            .unwrap();
        o.register_region("b", Priority::Normal, vec!["a".into()])
            .unwrap();
        assert!(o.compute_execution_order().is_err());
    }

    // -- Execution ----------------------------------------------------------

    #[test]
    fn test_execute_round_basic() {
        let mut o = Orchestrator::with_config(small_config());
        o.register_region("a", Priority::Normal, vec![]).unwrap();
        o.register_region("b", Priority::Normal, vec![]).unwrap();

        let record = o
            .execute_round(|name| match name {
                "a" => Ok(50.0),
                "b" => Ok(80.0),
                _ => unreachable!(),
            })
            .unwrap();

        assert_eq!(record.round, 1);
        assert_eq!(record.execution_order.len(), 2);
        assert!((record.total_latency_us - 130.0).abs() < 1e-9);
        assert_eq!(record.skipped, 0);
        assert_eq!(o.stats().total_rounds, 1);
        assert_eq!(o.stats().total_region_executions, 2);
    }

    #[test]
    fn test_execute_round_with_failure() {
        let mut o = Orchestrator::with_config(small_config());
        o.register_region("a", Priority::Normal, vec![]).unwrap();

        // First failure — not stalled yet (tolerance = 2).
        o.execute_round(|_| Err(Error::Configuration("fail".into())))
            .unwrap();
        assert_eq!(o.region("a").unwrap().consecutive_stalls, 1);
        assert_eq!(o.region("a").unwrap().status, RegionStatus::Idle);

        // Second failure — now stalled.
        o.execute_round(|_| Err(Error::Configuration("fail".into())))
            .unwrap();
        assert_eq!(o.region("a").unwrap().status, RegionStatus::Stalled);
        assert_eq!(o.stalled_regions(), vec!["a"]);
    }

    #[test]
    fn test_stall_reset_on_success() {
        let mut o = Orchestrator::with_config(small_config());
        o.register_region("a", Priority::Normal, vec![]).unwrap();

        // One failure then a success.
        o.execute_round(|_| Err(Error::Configuration("fail".into())))
            .unwrap();
        o.execute_round(|_| Ok(10.0)).unwrap();
        assert_eq!(o.region("a").unwrap().consecutive_stalls, 0);
        assert_eq!(o.region("a").unwrap().status, RegionStatus::Completed);
    }

    // -- Tick ---------------------------------------------------------------

    #[test]
    fn test_tick_increments() {
        let mut o = Orchestrator::new();
        o.tick();
        o.tick();
        assert_eq!(o.current_tick(), 2);
    }

    #[test]
    fn test_tick_resets_completed_to_idle() {
        let mut o = Orchestrator::with_config(small_config());
        o.register_region("a", Priority::Normal, vec![]).unwrap();
        o.execute_round(|_| Ok(10.0)).unwrap();
        assert_eq!(o.region("a").unwrap().status, RegionStatus::Completed);
        o.tick();
        assert_eq!(o.region("a").unwrap().status, RegionStatus::Idle);
    }

    #[test]
    fn test_process_convenience() {
        let mut o = Orchestrator::with_config(small_config());
        o.register_region("a", Priority::Normal, vec![]).unwrap();
        o.process(|_| Ok(10.0)).unwrap();
        assert_eq!(o.current_tick(), 1);
        assert_eq!(o.current_round(), 1);
    }

    // -- Queries ------------------------------------------------------------

    #[test]
    fn test_last_round() {
        let mut o = Orchestrator::with_config(small_config());
        assert!(o.last_round().is_none());
        o.register_region("a", Priority::Normal, vec![]).unwrap();
        o.execute_round(|_| Ok(10.0)).unwrap();
        assert!(o.last_round().is_some());
        assert_eq!(o.last_round().unwrap().round, 1);
    }

    #[test]
    fn test_round_history_capped() {
        let mut o = Orchestrator::with_config(small_config());
        o.register_region("a", Priority::Normal, vec![]).unwrap();
        for _ in 0..15 {
            o.execute_round(|_| Ok(10.0)).unwrap();
        }
        assert!(o.round_history().len() <= 10);
    }

    #[test]
    fn test_slow_regions() {
        let mut o = Orchestrator::with_config(small_config());
        o.register_region("fast", Priority::Normal, vec![]).unwrap();
        o.register_region("slow", Priority::Normal, vec![]).unwrap();
        o.execute_round(|name| match name {
            "fast" => Ok(10.0),
            "slow" => Ok(200.0), // > threshold of 100
            _ => unreachable!(),
        })
        .unwrap();
        let slow = o.slow_regions();
        assert!(slow.contains(&"slow"));
        assert!(!slow.contains(&"fast"));
    }

    // -- EMA diagnostics ----------------------------------------------------

    #[test]
    fn test_ema_initialises_on_first_round() {
        let mut o = Orchestrator::with_config(small_config());
        o.register_region("a", Priority::Normal, vec![]).unwrap();
        o.execute_round(|_| Ok(60.0)).unwrap();
        assert!((o.smoothed_round_latency() - 60.0).abs() < 1e-9);
        assert!((o.smoothed_regions_per_round() - 1.0).abs() < 1e-9);
    }

    #[test]
    fn test_ema_blends_subsequent() {
        let mut o = Orchestrator::with_config(small_config());
        o.register_region("a", Priority::Normal, vec![]).unwrap();
        o.execute_round(|_| Ok(100.0)).unwrap();
        o.execute_round(|_| Ok(200.0)).unwrap();
        // With decay=0.5: EMA = 0.5*200 + 0.5*100 = 150
        assert!((o.smoothed_round_latency() - 150.0).abs() < 1e-9);
    }

    #[test]
    fn test_per_region_ema_latency() {
        let mut o = Orchestrator::with_config(small_config());
        o.register_region("a", Priority::Normal, vec![]).unwrap();
        o.execute_round(|_| Ok(100.0)).unwrap();
        o.execute_round(|_| Ok(200.0)).unwrap();
        let ema = o.region("a").unwrap().ema_latency_us;
        // decay=0.5: 0.5*200 + 0.5*100 = 150
        assert!((ema - 150.0).abs() < 1e-9);
    }

    // -- Windowed diagnostics -----------------------------------------------

    #[test]
    fn test_windowed_round_latency() {
        let mut o = Orchestrator::with_config(small_config());
        o.register_region("a", Priority::Normal, vec![]).unwrap();
        o.execute_round(|_| Ok(100.0)).unwrap();
        o.execute_round(|_| Ok(200.0)).unwrap();
        let avg = o.windowed_round_latency().unwrap();
        assert!((avg - 150.0).abs() < 1e-9);
    }

    #[test]
    fn test_windowed_empty() {
        let o = Orchestrator::with_config(small_config());
        assert!(o.windowed_round_latency().is_none());
        assert!(o.windowed_regions_per_round().is_none());
        assert!(o.windowed_slow_count().is_none());
    }

    #[test]
    fn test_windowed_slow_count() {
        let mut o = Orchestrator::with_config(small_config());
        o.register_region("a", Priority::Normal, vec![]).unwrap();
        // All below threshold.
        o.execute_round(|_| Ok(10.0)).unwrap();
        assert!((o.windowed_slow_count().unwrap() - 0.0).abs() < 1e-9);
        // Above threshold.
        o.execute_round(|_| Ok(200.0)).unwrap();
        assert!(o.windowed_slow_count().unwrap() > 0.0);
    }

    #[test]
    fn test_is_latency_increasing() {
        let mut o = Orchestrator::with_config(small_config());
        o.register_region("a", Priority::Normal, vec![]).unwrap();
        // Not enough data.
        assert!(!o.is_latency_increasing());
        // Feed increasing latencies.
        for lat in &[10.0, 20.0, 100.0, 200.0, 300.0] {
            let l = *lat;
            o.execute_round(move |_| Ok(l)).unwrap();
        }
        assert!(o.is_latency_increasing());
    }

    #[test]
    fn test_is_latency_not_increasing() {
        let mut o = Orchestrator::with_config(small_config());
        o.register_region("a", Priority::Normal, vec![]).unwrap();
        for lat in &[100.0, 100.0, 100.0, 100.0, 100.0] {
            let l = *lat;
            o.execute_round(move |_| Ok(l)).unwrap();
        }
        assert!(!o.is_latency_increasing());
    }

    // -- Reset --------------------------------------------------------------

    #[test]
    fn test_reset() {
        let mut o = Orchestrator::with_config(small_config());
        o.register_region("a", Priority::Normal, vec![]).unwrap();
        o.process(|_| Ok(50.0)).unwrap();
        o.process(|_| Ok(60.0)).unwrap();
        assert_eq!(o.current_tick(), 2);
        assert_eq!(o.current_round(), 2);

        o.reset();

        assert_eq!(o.current_tick(), 0);
        assert_eq!(o.current_round(), 0);
        assert_eq!(o.stats().total_rounds, 0);
        assert!(o.round_history().is_empty());
        assert!(o.last_round().is_none());
        // Regions are preserved but reset.
        assert_eq!(o.region_count(), 1);
        assert_eq!(o.region("a").unwrap().executions, 0);
        assert_eq!(o.region("a").unwrap().status, RegionStatus::Idle);
    }

    // -- Full lifecycle -----------------------------------------------------

    #[test]
    fn test_full_lifecycle() {
        let mut o = Orchestrator::with_config(small_config());
        o.register_region("thalamus", Priority::Critical, vec![])
            .unwrap();
        o.register_region("cortex", Priority::Normal, vec!["thalamus".into()])
            .unwrap();
        o.register_region("hippocampus", Priority::Normal, vec!["cortex".into()])
            .unwrap();
        o.register_region("output", Priority::Low, vec!["hippocampus".into()])
            .unwrap();

        // Run several rounds.
        for i in 0..5 {
            o.process(move |name| {
                let base = match name {
                    "thalamus" => 10.0,
                    "cortex" => 30.0,
                    "hippocampus" => 20.0,
                    "output" => 5.0,
                    _ => 1.0,
                };
                Ok(base + i as f64)
            })
            .unwrap();
        }

        assert_eq!(o.current_tick(), 5);
        assert_eq!(o.current_round(), 5);
        assert_eq!(o.stats().total_region_executions, 20);
        assert_eq!(o.stats().total_region_failures, 0);

        // Execution order should respect dependencies.
        let last = o.last_round().unwrap();
        let thal_pos = last.execution_order.iter().position(|n| n == "thalamus").unwrap();
        let cort_pos = last.execution_order.iter().position(|n| n == "cortex").unwrap();
        let hipp_pos = last.execution_order.iter().position(|n| n == "hippocampus").unwrap();
        let out_pos = last.execution_order.iter().position(|n| n == "output").unwrap();
        assert!(thal_pos < cort_pos);
        assert!(cort_pos < hipp_pos);
        assert!(hipp_pos < out_pos);

        // Diagnostics should be populated.
        assert!(o.smoothed_round_latency() > 0.0);
        assert!(o.windowed_round_latency().is_some());
    }

    #[test]
    fn test_window_rolls() {
        let mut o = Orchestrator::with_config(small_config());
        o.register_region("a", Priority::Normal, vec![]).unwrap();
        for _ in 0..20 {
            o.execute_round(|_| Ok(10.0)).unwrap();
        }
        assert!(o.recent.len() <= o.config.window_size);
    }
}
