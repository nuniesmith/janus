//! DAG-based workflow execution engine
//!
//! Part of the Integration region — Workflow component.
//!
//! `GraphExecutor` manages a directed acyclic graph (DAG) of processing nodes,
//! computes topological execution order, detects parallel layers, tracks
//! per-node latencies, and exposes EMA-smoothed and windowed diagnostics.

use crate::common::{Error, Result};
use std::collections::{HashMap, HashSet, VecDeque};

// ---------------------------------------------------------------------------
// Configuration
// ---------------------------------------------------------------------------

/// Configuration for the graph executor.
#[derive(Debug, Clone)]
pub struct GraphExecutorConfig {
    /// Maximum number of nodes allowed in the graph.
    pub max_nodes: usize,
    /// Maximum number of edges allowed in the graph.
    pub max_edges: usize,
    /// Maximum execution history to retain.
    pub max_history: usize,
    /// Latency threshold (µs) above which a node is considered slow.
    pub slow_node_threshold_us: f64,
    /// EMA decay factor (0 < α < 1). Higher → faster adaptation.
    pub ema_decay: f64,
    /// Window size for rolling diagnostics.
    pub window_size: usize,
}

impl Default for GraphExecutorConfig {
    fn default() -> Self {
        Self {
            max_nodes: 256,
            max_edges: 1024,
            max_history: 100,
            slow_node_threshold_us: 5000.0,
            ema_decay: 0.1,
            window_size: 50,
        }
    }
}

// ---------------------------------------------------------------------------
// Node status
// ---------------------------------------------------------------------------

/// Execution status of a node.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum NodeStatus {
    /// Waiting to be scheduled.
    Pending,
    /// Currently executing.
    Running,
    /// Completed successfully.
    Completed,
    /// Failed during execution.
    Failed,
    /// Skipped because a dependency failed.
    Skipped,
}

// ---------------------------------------------------------------------------
// Graph node
// ---------------------------------------------------------------------------

/// A processing node within the execution graph.
#[derive(Debug, Clone)]
pub struct GraphNode {
    /// Unique node identifier.
    pub id: String,
    /// Human-readable label.
    pub label: String,
    /// IDs of nodes that must complete before this node can execute.
    pub dependencies: Vec<String>,
    /// Current execution status.
    pub status: NodeStatus,
    /// Last recorded latency in µs.
    pub last_latency_us: f64,
    /// EMA-smoothed latency.
    pub ema_latency_us: f64,
    /// Total successful executions.
    pub executions: u64,
    /// Total failures.
    pub failures: u64,
    /// Topological layer (0 = root nodes with no dependencies).
    pub layer: usize,
}

// ---------------------------------------------------------------------------
// Execution layer
// ---------------------------------------------------------------------------

/// A layer of nodes that can be executed in parallel (all dependencies in
/// earlier layers have already completed).
#[derive(Debug, Clone)]
pub struct ExecutionLayer {
    /// Layer index (0-based).
    pub index: usize,
    /// Node IDs in this layer.
    pub node_ids: Vec<String>,
}

// ---------------------------------------------------------------------------
// Execution record
// ---------------------------------------------------------------------------

/// Record of a complete graph execution run.
#[derive(Debug, Clone)]
pub struct ExecutionRecord {
    /// Run number (monotonically increasing).
    pub run: u64,
    /// Tick at which the run occurred.
    pub tick: u64,
    /// Ordered list of node IDs as executed.
    pub execution_order: Vec<String>,
    /// Per-node latencies for this run.
    pub latencies: HashMap<String, f64>,
    /// Total run latency (critical-path sum across layers).
    pub total_latency_us: f64,
    /// Number of nodes that completed.
    pub completed: usize,
    /// Number of nodes that failed.
    pub failed: usize,
    /// Number of nodes skipped.
    pub skipped: usize,
    /// Number of parallel layers traversed.
    pub layers: usize,
}

// ---------------------------------------------------------------------------
// Tick snapshot (windowed diagnostics)
// ---------------------------------------------------------------------------

#[derive(Debug, Clone)]
struct TickSnapshot {
    run_latency_us: f64,
    nodes_completed: usize,
    nodes_failed: usize,
    slow_node_count: usize,
    layer_count: usize,
}

// ---------------------------------------------------------------------------
// Statistics
// ---------------------------------------------------------------------------

/// Cumulative statistics for the graph executor.
#[derive(Debug, Clone)]
pub struct GraphExecutorStats {
    /// Total runs completed.
    pub total_runs: u64,
    /// Total node executions across all runs.
    pub total_node_executions: u64,
    /// Total node failures across all runs.
    pub total_node_failures: u64,
    /// Total nodes skipped across all runs.
    pub total_skipped: u64,
    /// EMA-smoothed run latency (critical path).
    pub ema_run_latency_us: f64,
    /// EMA-smoothed completed-node count per run.
    pub ema_completed_per_run: f64,
    /// EMA-smoothed slow-node fraction per run.
    pub ema_slow_fraction: f64,
}

impl Default for GraphExecutorStats {
    fn default() -> Self {
        Self {
            total_runs: 0,
            total_node_executions: 0,
            total_node_failures: 0,
            total_skipped: 0,
            ema_run_latency_us: 0.0,
            ema_completed_per_run: 0.0,
            ema_slow_fraction: 0.0,
        }
    }
}

// ---------------------------------------------------------------------------
// GraphExecutor
// ---------------------------------------------------------------------------

/// DAG-based workflow execution engine.
///
/// Manages a directed acyclic graph of processing nodes, computes topological
/// execution order with parallel-layer detection, tracks per-node latencies,
/// and provides EMA + windowed diagnostics.
pub struct GraphExecutor {
    config: GraphExecutorConfig,
    nodes: HashMap<String, GraphNode>,
    /// Insertion-order tracking for deterministic iteration.
    node_order: Vec<String>,
    /// Total number of edges in the graph.
    edge_count: usize,
    /// Cached execution layers (invalidated when graph changes).
    cached_layers: Option<Vec<ExecutionLayer>>,
    /// Execution history.
    history: VecDeque<ExecutionRecord>,
    /// Current tick.
    tick: u64,
    /// Current run counter.
    run: u64,
    /// Whether EMA has been initialised.
    ema_initialized: bool,
    /// Rolling window of tick snapshots.
    recent: VecDeque<TickSnapshot>,
    /// Cumulative statistics.
    stats: GraphExecutorStats,
}

impl Default for GraphExecutor {
    fn default() -> Self {
        Self::new()
    }
}

impl GraphExecutor {
    // -- Construction -------------------------------------------------------

    /// Create a new graph executor with default configuration.
    pub fn new() -> Self {
        Self::with_config(GraphExecutorConfig::default())
    }

    /// Create a new graph executor with the given configuration.
    pub fn with_config(config: GraphExecutorConfig) -> Self {
        Self {
            nodes: HashMap::new(),
            node_order: Vec::new(),
            edge_count: 0,
            cached_layers: None,
            history: VecDeque::with_capacity(config.max_history.min(256)),
            tick: 0,
            run: 0,
            ema_initialized: false,
            recent: VecDeque::with_capacity(config.window_size.min(256)),
            stats: GraphExecutorStats::default(),
            config,
        }
    }

    // -- Graph construction -------------------------------------------------

    /// Add a node to the graph. Returns an error if the node already exists,
    /// the node limit is reached, or adding its dependencies would exceed the
    /// edge limit.
    pub fn add_node(
        &mut self,
        id: impl Into<String>,
        label: impl Into<String>,
        dependencies: Vec<String>,
    ) -> Result<()> {
        let id = id.into();
        if self.nodes.contains_key(&id) {
            return Err(Error::Configuration(format!(
                "Node '{}' already exists",
                id
            )));
        }
        if self.nodes.len() >= self.config.max_nodes {
            return Err(Error::Configuration(format!(
                "Maximum node count ({}) reached",
                self.config.max_nodes
            )));
        }
        let new_edges = dependencies.len();
        if self.edge_count + new_edges > self.config.max_edges {
            return Err(Error::Configuration(format!(
                "Adding {} edges would exceed the edge limit ({})",
                new_edges, self.config.max_edges
            )));
        }

        self.edge_count += new_edges;
        self.cached_layers = None; // Invalidate cache.

        self.nodes.insert(
            id.clone(),
            GraphNode {
                id: id.clone(),
                label: label.into(),
                dependencies,
                status: NodeStatus::Pending,
                last_latency_us: 0.0,
                ema_latency_us: 0.0,
                executions: 0,
                failures: 0,
                layer: 0,
            },
        );
        self.node_order.push(id);
        Ok(())
    }

    /// Remove a node from the graph. Also removes it from other nodes'
    /// dependency lists.
    pub fn remove_node(&mut self, id: &str) -> Result<()> {
        let node = self
            .nodes
            .remove(id)
            .ok_or_else(|| Error::Configuration(format!("Unknown node '{}'", id)))?;

        self.edge_count = self.edge_count.saturating_sub(node.dependencies.len());
        self.node_order.retain(|n| n != id);

        // Remove from other nodes' dependency lists.
        for other in self.nodes.values_mut() {
            let before = other.dependencies.len();
            other.dependencies.retain(|d| d != id);
            let removed = before - other.dependencies.len();
            self.edge_count = self.edge_count.saturating_sub(removed);
        }

        self.cached_layers = None;
        Ok(())
    }

    /// Return a reference to a node by id.
    pub fn node(&self, id: &str) -> Option<&GraphNode> {
        self.nodes.get(id)
    }

    /// Number of nodes in the graph.
    pub fn node_count(&self) -> usize {
        self.nodes.len()
    }

    /// Number of edges in the graph.
    pub fn edge_count(&self) -> usize {
        self.edge_count
    }

    /// Node IDs in insertion order.
    pub fn node_ids(&self) -> &[String] {
        &self.node_order
    }

    // -- Topological ordering -----------------------------------------------

    /// Compute topological layers (Kahn's algorithm). Each layer contains
    /// nodes whose dependencies are all in earlier layers — meaning nodes
    /// within the same layer can theoretically execute in parallel.
    pub fn compute_layers(&mut self) -> Result<Vec<ExecutionLayer>> {
        if let Some(ref layers) = self.cached_layers {
            return Ok(layers.clone());
        }

        let active_ids: HashSet<&str> = self.nodes.keys().map(|s| s.as_str()).collect();

        // Build in-degree map.
        let mut in_degree: HashMap<&str, usize> = HashMap::new();
        let mut dependents: HashMap<&str, Vec<&str>> = HashMap::new();

        for id in &active_ids {
            in_degree.entry(id).or_insert(0);
        }

        for id in &active_ids {
            let node = &self.nodes[*id];
            for dep in &node.dependencies {
                if active_ids.contains(dep.as_str()) {
                    *in_degree.entry(id).or_insert(0) += 1;
                    dependents.entry(dep.as_str()).or_default().push(id);
                }
            }
        }

        let mut layers: Vec<ExecutionLayer> = Vec::new();
        let mut remaining = active_ids.len();

        // Collect initial roots.
        let mut current_layer: Vec<&str> = in_degree
            .iter()
            .filter(|(_, &deg)| deg == 0)
            .map(|(&id, _)| id)
            .collect();
        current_layer.sort();

        while !current_layer.is_empty() {
            let layer_index = layers.len();
            let layer_ids: Vec<String> = current_layer.iter().map(|s| s.to_string()).collect();

            // Assign layer to nodes.
            for id in &layer_ids {
                if let Some(n) = self.nodes.get_mut(id.as_str()) {
                    n.layer = layer_index;
                }
            }

            remaining -= current_layer.len();

            // Find next layer.
            let mut next_layer: Vec<&str> = Vec::new();
            for &id in &current_layer {
                if let Some(deps) = dependents.get(id) {
                    for &dep in deps {
                        let deg = in_degree.get_mut(dep).unwrap();
                        *deg -= 1;
                        if *deg == 0 {
                            next_layer.push(dep);
                        }
                    }
                }
            }
            next_layer.sort();
            next_layer.dedup();

            layers.push(ExecutionLayer {
                index: layer_index,
                node_ids: layer_ids,
            });

            current_layer = next_layer;
        }

        if remaining > 0 {
            return Err(Error::Configuration(
                "Cycle detected in the execution graph".to_string(),
            ));
        }

        self.cached_layers = Some(layers.clone());
        Ok(layers)
    }

    /// Return the topological execution order (flattened layers).
    pub fn execution_order(&mut self) -> Result<Vec<String>> {
        let layers = self.compute_layers()?;
        Ok(layers
            .iter()
            .flat_map(|l| l.node_ids.iter().cloned())
            .collect())
    }

    /// Number of parallel layers in the graph.
    pub fn layer_count(&mut self) -> Result<usize> {
        let layers = self.compute_layers()?;
        Ok(layers.len())
    }

    // -- Execution ----------------------------------------------------------

    /// Execute the full graph once. The provided closure is called for each
    /// node in topological order and must return the node's latency in µs
    /// (`Ok(f64)`) or an error. If a node fails, its dependents are skipped.
    pub fn execute<F>(&mut self, mut exec_fn: F) -> Result<&ExecutionRecord>
    where
        F: FnMut(&str) -> Result<f64>,
    {
        let layers = self.compute_layers()?;
        let mut latencies: HashMap<String, f64> = HashMap::new();
        let mut execution_order: Vec<String> = Vec::new();
        let mut failed_nodes: HashSet<String> = HashSet::new();
        let mut completed = 0usize;
        let mut failed = 0usize;
        let mut skipped = 0usize;
        let mut critical_path_latency = 0.0f64;
        let threshold = self.config.slow_node_threshold_us;
        let decay = self.config.ema_decay;

        for layer in &layers {
            let mut layer_max_latency = 0.0f64;

            for node_id in &layer.node_ids {
                // Check if any dependency failed → skip.
                let node = &self.nodes[node_id.as_str()];
                let dep_failed = node
                    .dependencies
                    .iter()
                    .any(|d| failed_nodes.contains(d));

                if dep_failed {
                    let n = self.nodes.get_mut(node_id.as_str()).unwrap();
                    n.status = NodeStatus::Skipped;
                    skipped += 1;
                    self.stats.total_skipped += 1;
                    continue;
                }

                // Mark running.
                if let Some(n) = self.nodes.get_mut(node_id.as_str()) {
                    n.status = NodeStatus::Running;
                }

                match exec_fn(node_id.as_str()) {
                    Ok(latency_us) => {
                        let n = self.nodes.get_mut(node_id.as_str()).unwrap();
                        n.status = NodeStatus::Completed;
                        n.last_latency_us = latency_us;
                        n.executions += 1;

                        if n.executions == 1 {
                            n.ema_latency_us = latency_us;
                        } else {
                            n.ema_latency_us =
                                decay * latency_us + (1.0 - decay) * n.ema_latency_us;
                        }

                        latencies.insert(node_id.clone(), latency_us);
                        execution_order.push(node_id.clone());
                        completed += 1;
                        self.stats.total_node_executions += 1;
                        layer_max_latency = layer_max_latency.max(latency_us);
                    }
                    Err(_) => {
                        let n = self.nodes.get_mut(node_id.as_str()).unwrap();
                        n.status = NodeStatus::Failed;
                        n.failures += 1;
                        failed_nodes.insert(node_id.clone());
                        failed += 1;
                        self.stats.total_node_failures += 1;
                    }
                }
            }

            critical_path_latency += layer_max_latency;
        }

        let slow_count = latencies
            .values()
            .filter(|&&lat| lat > threshold)
            .count();

        self.run += 1;
        self.stats.total_runs += 1;

        let record = ExecutionRecord {
            run: self.run,
            tick: self.tick,
            execution_order,
            latencies,
            total_latency_us: critical_path_latency,
            completed,
            failed,
            skipped,
            layers: layers.len(),
        };

        // Windowed snapshot.
        let snapshot = TickSnapshot {
            run_latency_us: critical_path_latency,
            nodes_completed: completed,
            nodes_failed: failed,
            slow_node_count: slow_count,
            layer_count: layers.len(),
        };
        if self.recent.len() >= self.config.window_size {
            self.recent.pop_front();
        }
        self.recent.push_back(snapshot);

        // EMA update.
        let slow_frac = if completed > 0 {
            slow_count as f64 / completed as f64
        } else {
            0.0
        };

        if !self.ema_initialized {
            self.stats.ema_run_latency_us = critical_path_latency;
            self.stats.ema_completed_per_run = completed as f64;
            self.stats.ema_slow_fraction = slow_frac;
            self.ema_initialized = true;
        } else {
            self.stats.ema_run_latency_us =
                decay * critical_path_latency + (1.0 - decay) * self.stats.ema_run_latency_us;
            self.stats.ema_completed_per_run =
                decay * completed as f64 + (1.0 - decay) * self.stats.ema_completed_per_run;
            self.stats.ema_slow_fraction =
                decay * slow_frac + (1.0 - decay) * self.stats.ema_slow_fraction;
        }

        // Store record.
        if self.history.len() >= self.config.max_history {
            self.history.pop_front();
        }
        self.history.push_back(record);

        Ok(self.history.back().unwrap())
    }

    // -- Tick ---------------------------------------------------------------

    /// Advance the internal tick counter and reset node statuses to Pending.
    pub fn tick(&mut self) {
        self.tick += 1;
        for node in self.nodes.values_mut() {
            if node.status == NodeStatus::Completed || node.status == NodeStatus::Skipped {
                node.status = NodeStatus::Pending;
            }
        }
    }

    /// Current tick value.
    pub fn current_tick(&self) -> u64 {
        self.tick
    }

    /// Current run counter.
    pub fn current_run(&self) -> u64 {
        self.run
    }

    /// Convenience: tick + execute.
    pub fn process<F>(&mut self, exec_fn: F) -> Result<()>
    where
        F: FnMut(&str) -> Result<f64>,
    {
        self.tick();
        self.execute(exec_fn)?;
        Ok(())
    }

    // -- Queries ------------------------------------------------------------

    /// Return the most recent execution record.
    pub fn last_execution(&self) -> Option<&ExecutionRecord> {
        self.history.back()
    }

    /// Return all stored execution records.
    pub fn execution_history(&self) -> &VecDeque<ExecutionRecord> {
        &self.history
    }

    /// Return node IDs whose EMA latency exceeds the slow threshold.
    pub fn slow_nodes(&self) -> Vec<&str> {
        self.nodes
            .values()
            .filter(|n| n.ema_latency_us > self.config.slow_node_threshold_us)
            .map(|n| n.id.as_str())
            .collect()
    }

    /// Return node IDs that are currently in `Failed` status.
    pub fn failed_nodes(&self) -> Vec<&str> {
        self.nodes
            .values()
            .filter(|n| n.status == NodeStatus::Failed)
            .map(|n| n.id.as_str())
            .collect()
    }

    // -- Diagnostics --------------------------------------------------------

    /// Cumulative statistics.
    pub fn stats(&self) -> &GraphExecutorStats {
        &self.stats
    }

    /// Configuration.
    pub fn config(&self) -> &GraphExecutorConfig {
        &self.config
    }

    /// EMA-smoothed run latency.
    pub fn smoothed_run_latency(&self) -> f64 {
        self.stats.ema_run_latency_us
    }

    /// EMA-smoothed completed nodes per run.
    pub fn smoothed_completed_per_run(&self) -> f64 {
        self.stats.ema_completed_per_run
    }

    /// EMA-smoothed slow-node fraction.
    pub fn smoothed_slow_fraction(&self) -> f64 {
        self.stats.ema_slow_fraction
    }

    /// Windowed average run latency.
    pub fn windowed_run_latency(&self) -> Option<f64> {
        if self.recent.is_empty() {
            return None;
        }
        let sum: f64 = self.recent.iter().map(|s| s.run_latency_us).sum();
        Some(sum / self.recent.len() as f64)
    }

    /// Windowed average completed-node count.
    pub fn windowed_completed_per_run(&self) -> Option<f64> {
        if self.recent.is_empty() {
            return None;
        }
        let sum: f64 = self.recent.iter().map(|s| s.nodes_completed as f64).sum();
        Some(sum / self.recent.len() as f64)
    }

    /// Windowed average slow-node count.
    pub fn windowed_slow_count(&self) -> Option<f64> {
        if self.recent.is_empty() {
            return None;
        }
        let sum: f64 = self
            .recent
            .iter()
            .map(|s| s.slow_node_count as f64)
            .sum();
        Some(sum / self.recent.len() as f64)
    }

    /// Windowed average layer count.
    pub fn windowed_layer_count(&self) -> Option<f64> {
        if self.recent.is_empty() {
            return None;
        }
        let sum: f64 = self.recent.iter().map(|s| s.layer_count as f64).sum();
        Some(sum / self.recent.len() as f64)
    }

    /// Whether run latency is trending upward (second half > first half).
    pub fn is_latency_increasing(&self) -> bool {
        if self.recent.len() < 4 {
            return false;
        }
        let mid = self.recent.len() / 2;
        let first_half: f64 =
            self.recent.iter().take(mid).map(|s| s.run_latency_us).sum::<f64>() / mid as f64;
        let second_half: f64 = self
            .recent
            .iter()
            .skip(mid)
            .map(|s| s.run_latency_us)
            .sum::<f64>()
            / (self.recent.len() - mid) as f64;
        second_half > first_half * 1.05
    }

    // -- Reset --------------------------------------------------------------

    /// Reset execution state while preserving the graph topology and
    /// configuration.
    pub fn reset(&mut self) {
        self.tick = 0;
        self.run = 0;
        self.ema_initialized = false;
        self.recent.clear();
        self.history.clear();
        self.stats = GraphExecutorStats::default();
        for node in self.nodes.values_mut() {
            node.status = NodeStatus::Pending;
            node.last_latency_us = 0.0;
            node.ema_latency_us = 0.0;
            node.executions = 0;
            node.failures = 0;
        }
    }

    /// Reset everything including the graph topology.
    pub fn clear(&mut self) {
        self.nodes.clear();
        self.node_order.clear();
        self.edge_count = 0;
        self.cached_layers = None;
        self.reset();
    }
}

// ===========================================================================
// Tests
// ===========================================================================

#[cfg(test)]
mod tests {
    use super::*;

    fn small_config() -> GraphExecutorConfig {
        GraphExecutorConfig {
            max_nodes: 16,
            max_edges: 32,
            max_history: 5,
            slow_node_threshold_us: 100.0,
            ema_decay: 0.5,
            window_size: 5,
        }
    }

    // -- Construction -------------------------------------------------------

    #[test]
    fn test_new_default() {
        let ge = GraphExecutor::new();
        assert_eq!(ge.node_count(), 0);
        assert_eq!(ge.edge_count(), 0);
        assert_eq!(ge.current_tick(), 0);
        assert_eq!(ge.current_run(), 0);
    }

    #[test]
    fn test_with_config() {
        let ge = GraphExecutor::with_config(small_config());
        assert_eq!(ge.config().max_nodes, 16);
    }

    // -- Graph construction -------------------------------------------------

    #[test]
    fn test_add_node() {
        let mut ge = GraphExecutor::with_config(small_config());
        ge.add_node("a", "Node A", vec![]).unwrap();
        assert_eq!(ge.node_count(), 1);
        assert_eq!(ge.edge_count(), 0);
        assert!(ge.node("a").is_some());
    }

    #[test]
    fn test_add_node_with_deps() {
        let mut ge = GraphExecutor::with_config(small_config());
        ge.add_node("a", "A", vec![]).unwrap();
        ge.add_node("b", "B", vec!["a".into()]).unwrap();
        assert_eq!(ge.node_count(), 2);
        assert_eq!(ge.edge_count(), 1);
    }

    #[test]
    fn test_add_duplicate_fails() {
        let mut ge = GraphExecutor::with_config(small_config());
        ge.add_node("a", "A", vec![]).unwrap();
        assert!(ge.add_node("a", "A2", vec![]).is_err());
    }

    #[test]
    fn test_add_node_at_capacity() {
        let cfg = GraphExecutorConfig {
            max_nodes: 2,
            ..small_config()
        };
        let mut ge = GraphExecutor::with_config(cfg);
        ge.add_node("a", "A", vec![]).unwrap();
        ge.add_node("b", "B", vec![]).unwrap();
        assert!(ge.add_node("c", "C", vec![]).is_err());
    }

    #[test]
    fn test_add_node_edge_limit() {
        let cfg = GraphExecutorConfig {
            max_edges: 1,
            ..small_config()
        };
        let mut ge = GraphExecutor::with_config(cfg);
        ge.add_node("a", "A", vec![]).unwrap();
        ge.add_node("b", "B", vec!["a".into()]).unwrap(); // 1 edge
        assert!(ge.add_node("c", "C", vec!["a".into()]).is_err()); // would be 2
    }

    #[test]
    fn test_remove_node() {
        let mut ge = GraphExecutor::with_config(small_config());
        ge.add_node("a", "A", vec![]).unwrap();
        ge.add_node("b", "B", vec!["a".into()]).unwrap();
        assert_eq!(ge.edge_count(), 1);

        ge.remove_node("a").unwrap();
        assert_eq!(ge.node_count(), 1);
        assert!(ge.node("a").is_none());
        // b's dependency on a should be removed
        assert!(ge.node("b").unwrap().dependencies.is_empty());
        assert_eq!(ge.edge_count(), 0);
    }

    #[test]
    fn test_remove_unknown_fails() {
        let mut ge = GraphExecutor::with_config(small_config());
        assert!(ge.remove_node("nope").is_err());
    }

    #[test]
    fn test_node_ids_insertion_order() {
        let mut ge = GraphExecutor::with_config(small_config());
        ge.add_node("z", "Z", vec![]).unwrap();
        ge.add_node("a", "A", vec![]).unwrap();
        ge.add_node("m", "M", vec![]).unwrap();
        assert_eq!(ge.node_ids(), &["z", "a", "m"]);
    }

    // -- Topological ordering -----------------------------------------------

    #[test]
    fn test_compute_layers_simple() {
        let mut ge = GraphExecutor::with_config(small_config());
        ge.add_node("a", "A", vec![]).unwrap();
        ge.add_node("b", "B", vec!["a".into()]).unwrap();
        ge.add_node("c", "C", vec!["b".into()]).unwrap();

        let layers = ge.compute_layers().unwrap();
        assert_eq!(layers.len(), 3);
        assert_eq!(layers[0].node_ids, vec!["a"]);
        assert_eq!(layers[1].node_ids, vec!["b"]);
        assert_eq!(layers[2].node_ids, vec!["c"]);
    }

    #[test]
    fn test_compute_layers_parallel() {
        let mut ge = GraphExecutor::with_config(small_config());
        ge.add_node("root", "Root", vec![]).unwrap();
        ge.add_node("b", "B", vec!["root".into()]).unwrap();
        ge.add_node("c", "C", vec!["root".into()]).unwrap();
        ge.add_node("d", "D", vec!["b".into(), "c".into()]).unwrap();

        let layers = ge.compute_layers().unwrap();
        assert_eq!(layers.len(), 3);
        assert_eq!(layers[0].node_ids, vec!["root"]);
        // b and c should be in the same layer (parallel)
        let mut l1 = layers[1].node_ids.clone();
        l1.sort();
        assert_eq!(l1, vec!["b", "c"]);
        assert_eq!(layers[2].node_ids, vec!["d"]);
    }

    #[test]
    fn test_compute_layers_no_deps() {
        let mut ge = GraphExecutor::with_config(small_config());
        ge.add_node("a", "A", vec![]).unwrap();
        ge.add_node("b", "B", vec![]).unwrap();

        let layers = ge.compute_layers().unwrap();
        assert_eq!(layers.len(), 1);
        assert_eq!(layers[0].node_ids.len(), 2);
    }

    #[test]
    fn test_cycle_detection() {
        let mut ge = GraphExecutor::with_config(small_config());
        ge.add_node("a", "A", vec!["b".into()]).unwrap();
        ge.add_node("b", "B", vec!["a".into()]).unwrap();
        assert!(ge.compute_layers().is_err());
    }

    #[test]
    fn test_execution_order() {
        let mut ge = GraphExecutor::with_config(small_config());
        ge.add_node("c", "C", vec!["b".into()]).unwrap();
        ge.add_node("b", "B", vec!["a".into()]).unwrap();
        ge.add_node("a", "A", vec![]).unwrap();

        let order = ge.execution_order().unwrap();
        assert_eq!(order, vec!["a", "b", "c"]);
    }

    #[test]
    fn test_layer_count() {
        let mut ge = GraphExecutor::with_config(small_config());
        ge.add_node("a", "A", vec![]).unwrap();
        ge.add_node("b", "B", vec!["a".into()]).unwrap();
        assert_eq!(ge.layer_count().unwrap(), 2);
    }

    #[test]
    fn test_node_layer_assignment() {
        let mut ge = GraphExecutor::with_config(small_config());
        ge.add_node("a", "A", vec![]).unwrap();
        ge.add_node("b", "B", vec!["a".into()]).unwrap();
        ge.compute_layers().unwrap();
        assert_eq!(ge.node("a").unwrap().layer, 0);
        assert_eq!(ge.node("b").unwrap().layer, 1);
    }

    #[test]
    fn test_layer_cache_invalidation() {
        let mut ge = GraphExecutor::with_config(small_config());
        ge.add_node("a", "A", vec![]).unwrap();
        ge.compute_layers().unwrap();
        assert!(ge.cached_layers.is_some());
        ge.add_node("b", "B", vec![]).unwrap();
        assert!(ge.cached_layers.is_none()); // Invalidated.
    }

    // -- Execution ----------------------------------------------------------

    #[test]
    fn test_execute_basic() {
        let mut ge = GraphExecutor::with_config(small_config());
        ge.add_node("a", "A", vec![]).unwrap();
        ge.add_node("b", "B", vec!["a".into()]).unwrap();

        let record = ge
            .execute(|name| match name {
                "a" => Ok(30.0),
                "b" => Ok(50.0),
                _ => unreachable!(),
            })
            .unwrap();

        assert_eq!(record.run, 1);
        assert_eq!(record.completed, 2);
        assert_eq!(record.failed, 0);
        assert_eq!(record.skipped, 0);
        assert_eq!(record.layers, 2);
        // Critical path: max(a)=30 + max(b)=50 = 80
        assert!((record.total_latency_us - 80.0).abs() < 1e-9);
    }

    #[test]
    fn test_execute_failure_skips_dependents() {
        let mut ge = GraphExecutor::with_config(small_config());
        ge.add_node("a", "A", vec![]).unwrap();
        ge.add_node("b", "B", vec!["a".into()]).unwrap();
        ge.add_node("c", "C", vec![]).unwrap();

        let record = ge
            .execute(|name| match name {
                "a" => Err(Error::Configuration("fail".into())),
                "c" => Ok(10.0),
                _ => unreachable!(), // b should be skipped
            })
            .unwrap();

        assert_eq!(record.completed, 1); // only c
        assert_eq!(record.failed, 1); // a
        assert_eq!(record.skipped, 1); // b
        assert_eq!(ge.node("a").unwrap().status, NodeStatus::Failed);
        assert_eq!(ge.node("b").unwrap().status, NodeStatus::Skipped);
        assert_eq!(ge.node("c").unwrap().status, NodeStatus::Completed);
    }

    #[test]
    fn test_execute_parallel_critical_path() {
        let mut ge = GraphExecutor::with_config(small_config());
        ge.add_node("root", "Root", vec![]).unwrap();
        ge.add_node("fast", "Fast", vec!["root".into()]).unwrap();
        ge.add_node("slow", "Slow", vec!["root".into()]).unwrap();
        ge.add_node("end", "End", vec!["fast".into(), "slow".into()])
            .unwrap();

        let record = ge
            .execute(|name| match name {
                "root" => Ok(10.0),
                "fast" => Ok(20.0),
                "slow" => Ok(80.0),
                "end" => Ok(5.0),
                _ => unreachable!(),
            })
            .unwrap();

        // Critical path: root(10) + max(fast=20, slow=80) + end(5) = 95
        assert!((record.total_latency_us - 95.0).abs() < 1e-9);
    }

    #[test]
    fn test_per_node_ema_latency() {
        let mut ge = GraphExecutor::with_config(small_config());
        ge.add_node("a", "A", vec![]).unwrap();

        ge.execute(|_| Ok(100.0)).unwrap();
        ge.tick();
        ge.execute(|_| Ok(200.0)).unwrap();

        // decay=0.5: 0.5*200 + 0.5*100 = 150
        assert!((ge.node("a").unwrap().ema_latency_us - 150.0).abs() < 1e-9);
    }

    #[test]
    fn test_node_execution_counters() {
        let mut ge = GraphExecutor::with_config(small_config());
        ge.add_node("a", "A", vec![]).unwrap();

        ge.execute(|_| Ok(10.0)).unwrap();
        ge.execute(|_| Err(Error::Configuration("fail".into())))
            .unwrap();

        assert_eq!(ge.node("a").unwrap().executions, 1);
        assert_eq!(ge.node("a").unwrap().failures, 1);
    }

    // -- Tick ---------------------------------------------------------------

    #[test]
    fn test_tick_increments() {
        let mut ge = GraphExecutor::new();
        ge.tick();
        ge.tick();
        assert_eq!(ge.current_tick(), 2);
    }

    #[test]
    fn test_tick_resets_status() {
        let mut ge = GraphExecutor::with_config(small_config());
        ge.add_node("a", "A", vec![]).unwrap();
        ge.execute(|_| Ok(10.0)).unwrap();
        assert_eq!(ge.node("a").unwrap().status, NodeStatus::Completed);
        ge.tick();
        assert_eq!(ge.node("a").unwrap().status, NodeStatus::Pending);
    }

    #[test]
    fn test_tick_does_not_reset_failed() {
        let mut ge = GraphExecutor::with_config(small_config());
        ge.add_node("a", "A", vec![]).unwrap();
        ge.execute(|_| Err(Error::Configuration("fail".into())))
            .unwrap();
        ge.tick();
        assert_eq!(ge.node("a").unwrap().status, NodeStatus::Failed);
    }

    #[test]
    fn test_process_convenience() {
        let mut ge = GraphExecutor::with_config(small_config());
        ge.add_node("a", "A", vec![]).unwrap();
        ge.process(|_| Ok(10.0)).unwrap();
        assert_eq!(ge.current_tick(), 1);
        assert_eq!(ge.current_run(), 1);
    }

    // -- Queries ------------------------------------------------------------

    #[test]
    fn test_last_execution() {
        let mut ge = GraphExecutor::with_config(small_config());
        assert!(ge.last_execution().is_none());
        ge.add_node("a", "A", vec![]).unwrap();
        ge.execute(|_| Ok(10.0)).unwrap();
        assert_eq!(ge.last_execution().unwrap().run, 1);
    }

    #[test]
    fn test_history_capped() {
        let mut ge = GraphExecutor::with_config(small_config()); // max_history=5
        ge.add_node("a", "A", vec![]).unwrap();
        for _ in 0..10 {
            ge.execute(|_| Ok(10.0)).unwrap();
        }
        assert!(ge.execution_history().len() <= 5);
    }

    #[test]
    fn test_slow_nodes() {
        let mut ge = GraphExecutor::with_config(small_config()); // threshold=100
        ge.add_node("fast", "Fast", vec![]).unwrap();
        ge.add_node("slow", "Slow", vec![]).unwrap();
        ge.execute(|name| match name {
            "fast" => Ok(10.0),
            "slow" => Ok(200.0),
            _ => unreachable!(),
        })
        .unwrap();
        let slow = ge.slow_nodes();
        assert!(slow.contains(&"slow"));
        assert!(!slow.contains(&"fast"));
    }

    #[test]
    fn test_failed_nodes() {
        let mut ge = GraphExecutor::with_config(small_config());
        ge.add_node("a", "A", vec![]).unwrap();
        ge.add_node("b", "B", vec![]).unwrap();
        ge.execute(|name| match name {
            "a" => Err(Error::Configuration("fail".into())),
            "b" => Ok(10.0),
            _ => unreachable!(),
        })
        .unwrap();
        let f = ge.failed_nodes();
        assert!(f.contains(&"a"));
        assert!(!f.contains(&"b"));
    }

    // -- EMA diagnostics ----------------------------------------------------

    #[test]
    fn test_ema_initialises_on_first_run() {
        let mut ge = GraphExecutor::with_config(small_config());
        ge.add_node("a", "A", vec![]).unwrap();
        ge.execute(|_| Ok(60.0)).unwrap();
        assert!((ge.smoothed_run_latency() - 60.0).abs() < 1e-9);
        assert!((ge.smoothed_completed_per_run() - 1.0).abs() < 1e-9);
    }

    #[test]
    fn test_ema_blends() {
        let mut ge = GraphExecutor::with_config(small_config()); // decay=0.5
        ge.add_node("a", "A", vec![]).unwrap();
        ge.execute(|_| Ok(100.0)).unwrap();
        ge.tick();
        ge.execute(|_| Ok(200.0)).unwrap();
        // 0.5*200 + 0.5*100 = 150
        assert!((ge.smoothed_run_latency() - 150.0).abs() < 1e-9);
    }

    // -- Windowed diagnostics -----------------------------------------------

    #[test]
    fn test_windowed_run_latency() {
        let mut ge = GraphExecutor::with_config(small_config());
        ge.add_node("a", "A", vec![]).unwrap();
        ge.execute(|_| Ok(100.0)).unwrap();
        ge.execute(|_| Ok(200.0)).unwrap();
        let avg = ge.windowed_run_latency().unwrap();
        assert!((avg - 150.0).abs() < 1e-9);
    }

    #[test]
    fn test_windowed_empty() {
        let ge = GraphExecutor::with_config(small_config());
        assert!(ge.windowed_run_latency().is_none());
        assert!(ge.windowed_completed_per_run().is_none());
        assert!(ge.windowed_slow_count().is_none());
        assert!(ge.windowed_layer_count().is_none());
    }

    #[test]
    fn test_is_latency_increasing() {
        let mut ge = GraphExecutor::with_config(small_config());
        ge.add_node("a", "A", vec![]).unwrap();
        assert!(!ge.is_latency_increasing());
        for lat in &[10.0, 20.0, 100.0, 200.0, 300.0] {
            let l = *lat;
            ge.execute(move |_| Ok(l)).unwrap();
        }
        assert!(ge.is_latency_increasing());
    }

    #[test]
    fn test_is_latency_not_increasing() {
        let mut ge = GraphExecutor::with_config(small_config());
        ge.add_node("a", "A", vec![]).unwrap();
        for _ in 0..5 {
            ge.execute(|_| Ok(50.0)).unwrap();
        }
        assert!(!ge.is_latency_increasing());
    }

    // -- Reset & clear ------------------------------------------------------

    #[test]
    fn test_reset_preserves_graph() {
        let mut ge = GraphExecutor::with_config(small_config());
        ge.add_node("a", "A", vec![]).unwrap();
        ge.process(|_| Ok(10.0)).unwrap();
        ge.reset();

        assert_eq!(ge.current_tick(), 0);
        assert_eq!(ge.current_run(), 0);
        assert_eq!(ge.node_count(), 1); // graph preserved
        assert_eq!(ge.node("a").unwrap().executions, 0);
        assert_eq!(ge.node("a").unwrap().status, NodeStatus::Pending);
        assert!(ge.last_execution().is_none());
        assert_eq!(ge.stats().total_runs, 0);
    }

    #[test]
    fn test_clear() {
        let mut ge = GraphExecutor::with_config(small_config());
        ge.add_node("a", "A", vec![]).unwrap();
        ge.process(|_| Ok(10.0)).unwrap();
        ge.clear();

        assert_eq!(ge.node_count(), 0);
        assert_eq!(ge.edge_count(), 0);
        assert_eq!(ge.current_tick(), 0);
    }

    // -- Full lifecycle -----------------------------------------------------

    #[test]
    fn test_full_lifecycle() {
        let mut ge = GraphExecutor::with_config(small_config());

        ge.add_node("input", "Input", vec![]).unwrap();
        ge.add_node("process_a", "Process A", vec!["input".into()])
            .unwrap();
        ge.add_node("process_b", "Process B", vec!["input".into()])
            .unwrap();
        ge.add_node("merge", "Merge", vec!["process_a".into(), "process_b".into()])
            .unwrap();
        ge.add_node("output", "Output", vec!["merge".into()])
            .unwrap();

        for i in 0..3 {
            ge.process(move |name| {
                let base = match name {
                    "input" => 10.0,
                    "process_a" => 30.0,
                    "process_b" => 50.0,
                    "merge" => 20.0,
                    "output" => 5.0,
                    _ => 1.0,
                };
                Ok(base + i as f64)
            })
            .unwrap();
        }

        assert_eq!(ge.current_tick(), 3);
        assert_eq!(ge.current_run(), 3);
        assert_eq!(ge.stats().total_node_executions, 15); // 5 nodes * 3 runs
        assert_eq!(ge.stats().total_node_failures, 0);

        // Check execution order respects dependencies
        let last = ge.last_execution().unwrap();
        let pos = |name: &str| -> usize {
            last.execution_order.iter().position(|n| n == name).unwrap()
        };
        assert!(pos("input") < pos("process_a"));
        assert!(pos("input") < pos("process_b"));
        assert!(pos("process_a") < pos("merge"));
        assert!(pos("process_b") < pos("merge"));
        assert!(pos("merge") < pos("output"));

        assert!(ge.smoothed_run_latency() > 0.0);
        assert!(ge.windowed_run_latency().is_some());
    }

    #[test]
    fn test_window_rolls() {
        let mut ge = GraphExecutor::with_config(small_config()); // window_size=5
        ge.add_node("a", "A", vec![]).unwrap();
        for _ in 0..20 {
            ge.execute(|_| Ok(10.0)).unwrap();
        }
        assert!(ge.recent.len() <= 5);
    }
}
