//! Register processing nodes
//!
//! Part of the Integration region — Workflow component.
//!
//! `NodeRegistry` provides a typed registry for processing nodes within the
//! neuromorphic workflow engine. Each node has a category, optional
//! dependencies, activation state, and execution metadata. The registry
//! supports lookup by name or category, dependency validation, activation /
//! deactivation, and exposes EMA-smoothed and windowed diagnostics.

use crate::common::{Error, Result};
use std::collections::{HashMap, HashSet, VecDeque};

// ---------------------------------------------------------------------------
// Configuration
// ---------------------------------------------------------------------------

/// Configuration for the node registry.
#[derive(Debug, Clone)]
pub struct NodeRegistryConfig {
    /// Maximum number of nodes that can be registered.
    pub max_nodes: usize,
    /// Maximum number of categories tracked.
    pub max_categories: usize,
    /// EMA decay factor (0 < α < 1). Higher → faster adaptation.
    pub ema_decay: f64,
    /// Window size for rolling diagnostics.
    pub window_size: usize,
}

impl Default for NodeRegistryConfig {
    fn default() -> Self {
        Self {
            max_nodes: 256,
            max_categories: 32,
            ema_decay: 0.1,
            window_size: 50,
        }
    }
}

// ---------------------------------------------------------------------------
// Node category
// ---------------------------------------------------------------------------

/// Category of a processing node.
#[derive(Debug, Clone, PartialEq, Eq, Hash)]
pub enum NodeCategory {
    /// Preprocessing / data ingestion nodes.
    Input,
    /// Core computation / transformation nodes.
    Transform,
    /// Aggregation / reduction nodes.
    Aggregate,
    /// Output / action nodes (e.g. signal emission).
    Output,
    /// Utility / helper nodes (logging, metrics, etc.).
    Utility,
    /// Custom user-defined category.
    Custom(String),
}

impl std::fmt::Display for NodeCategory {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        match self {
            NodeCategory::Input => write!(f, "Input"),
            NodeCategory::Transform => write!(f, "Transform"),
            NodeCategory::Aggregate => write!(f, "Aggregate"),
            NodeCategory::Output => write!(f, "Output"),
            NodeCategory::Utility => write!(f, "Utility"),
            NodeCategory::Custom(s) => write!(f, "Custom({})", s),
        }
    }
}

// ---------------------------------------------------------------------------
// Node status
// ---------------------------------------------------------------------------

/// Activation status of a registered node.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum NodeStatus {
    /// Node is registered and active — eligible for execution.
    Active,
    /// Node is registered but inactive — skipped during execution.
    Inactive,
    /// Node encountered a critical error and has been disabled.
    Error,
}

impl std::fmt::Display for NodeStatus {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        match self {
            NodeStatus::Active => write!(f, "Active"),
            NodeStatus::Inactive => write!(f, "Inactive"),
            NodeStatus::Error => write!(f, "Error"),
        }
    }
}

// ---------------------------------------------------------------------------
// Node descriptor
// ---------------------------------------------------------------------------

/// Descriptor for a registered processing node.
#[derive(Debug, Clone)]
pub struct NodeDescriptor {
    /// Unique node name.
    pub name: String,
    /// Category this node belongs to.
    pub category: NodeCategory,
    /// Human-readable description.
    pub description: String,
    /// Names of nodes that must execute before this one.
    pub dependencies: Vec<String>,
    /// Current activation status.
    pub status: NodeStatus,
    /// Version string for the node implementation.
    pub version: String,
    /// Total number of times this node has been executed.
    pub execution_count: u64,
    /// Total number of errors encountered during execution.
    pub error_count: u64,
    /// Last recorded execution latency in simulated µs.
    pub last_latency_us: f64,
    /// EMA-smoothed execution latency.
    pub ema_latency_us: f64,
    /// Tick at which the node was registered.
    pub registered_at_tick: u64,
    /// Tick of the most recent execution.
    pub last_executed_tick: u64,
}

// ---------------------------------------------------------------------------
// Tick snapshot (windowed diagnostics)
// ---------------------------------------------------------------------------

#[derive(Debug, Clone)]
struct TickSnapshot {
    total_nodes: usize,
    active_nodes: usize,
    inactive_nodes: usize,
    error_nodes: usize,
    category_count: usize,
    active_ratio: f64,
}

// ---------------------------------------------------------------------------
// Statistics
// ---------------------------------------------------------------------------

/// Cumulative statistics for the node registry.
#[derive(Debug, Clone)]
pub struct NodeRegistryStats {
    /// Total nodes ever registered (including removed).
    pub total_registered: u64,
    /// Total nodes removed.
    pub total_removed: u64,
    /// Total activations performed.
    pub total_activations: u64,
    /// Total deactivations performed.
    pub total_deactivations: u64,
    /// Total node executions recorded.
    pub total_executions: u64,
    /// Total node errors recorded.
    pub total_errors: u64,
    /// Current active-node ratio.
    pub active_ratio: f64,
    /// EMA-smoothed active ratio.
    pub ema_active_ratio: f64,
    /// EMA-smoothed total node count.
    pub ema_node_count: f64,
}

impl Default for NodeRegistryStats {
    fn default() -> Self {
        Self {
            total_registered: 0,
            total_removed: 0,
            total_activations: 0,
            total_deactivations: 0,
            total_executions: 0,
            total_errors: 0,
            active_ratio: 0.0,
            ema_active_ratio: 0.0,
            ema_node_count: 0.0,
        }
    }
}

// ---------------------------------------------------------------------------
// NodeRegistry
// ---------------------------------------------------------------------------

/// Processing node registry with categorised lookup and diagnostics.
pub struct NodeRegistry {
    config: NodeRegistryConfig,
    /// Registered nodes keyed by name.
    nodes: HashMap<String, NodeDescriptor>,
    /// Insertion-order tracking for deterministic iteration.
    insertion_order: Vec<String>,
    /// Tick counter.
    tick: u64,
    /// Whether EMA has been initialised.
    ema_initialized: bool,
    /// Rolling window of tick snapshots.
    recent: VecDeque<TickSnapshot>,
    /// Cumulative statistics.
    stats: NodeRegistryStats,
}

impl Default for NodeRegistry {
    fn default() -> Self {
        Self::new()
    }
}

impl NodeRegistry {
    // -------------------------------------------------------------------
    // Construction
    // -------------------------------------------------------------------

    /// Create a new node registry with default configuration.
    pub fn new() -> Self {
        Self::with_config(NodeRegistryConfig::default())
    }

    /// Create a new node registry with the given configuration.
    pub fn with_config(config: NodeRegistryConfig) -> Self {
        Self {
            nodes: HashMap::new(),
            insertion_order: Vec::new(),
            tick: 0,
            ema_initialized: false,
            recent: VecDeque::with_capacity(config.window_size.min(256)),
            stats: NodeRegistryStats::default(),
            config,
        }
    }

    // -------------------------------------------------------------------
    // Registration
    // -------------------------------------------------------------------

    /// Register a new processing node.
    ///
    /// Returns an error if the node already exists or the maximum capacity
    /// has been reached.
    pub fn register(
        &mut self,
        name: impl Into<String>,
        category: NodeCategory,
        description: impl Into<String>,
        dependencies: Vec<String>,
        version: impl Into<String>,
    ) -> Result<()> {
        let name = name.into();
        if self.nodes.contains_key(&name) {
            return Err(Error::Configuration(format!(
                "Node '{}' is already registered",
                name
            )));
        }
        if self.nodes.len() >= self.config.max_nodes {
            return Err(Error::Configuration(format!(
                "Maximum node count ({}) reached",
                self.config.max_nodes
            )));
        }

        let descriptor = NodeDescriptor {
            name: name.clone(),
            category,
            description: description.into(),
            dependencies,
            status: NodeStatus::Active,
            version: version.into(),
            execution_count: 0,
            error_count: 0,
            last_latency_us: 0.0,
            ema_latency_us: 0.0,
            registered_at_tick: self.tick,
            last_executed_tick: 0,
        };

        self.nodes.insert(name.clone(), descriptor);
        self.insertion_order.push(name);
        self.stats.total_registered += 1;
        self.update_active_ratio();
        Ok(())
    }

    /// Remove a node from the registry.
    pub fn remove(&mut self, name: &str) -> Result<NodeDescriptor> {
        let descriptor = self
            .nodes
            .remove(name)
            .ok_or_else(|| Error::Configuration(format!("Unknown node '{}'", name)))?;
        self.insertion_order.retain(|n| n != name);
        self.stats.total_removed += 1;
        self.update_active_ratio();
        Ok(descriptor)
    }

    /// Check whether a node with the given name exists.
    pub fn contains(&self, name: &str) -> bool {
        self.nodes.contains_key(name)
    }

    /// Look up a node descriptor by name.
    pub fn get(&self, name: &str) -> Option<&NodeDescriptor> {
        self.nodes.get(name)
    }

    /// Look up a mutable node descriptor by name.
    pub fn get_mut(&mut self, name: &str) -> Option<&mut NodeDescriptor> {
        self.nodes.get_mut(name)
    }

    /// Return the number of currently registered nodes.
    pub fn node_count(&self) -> usize {
        self.nodes.len()
    }

    /// Return the names of all registered nodes in insertion order.
    pub fn node_names(&self) -> Vec<&str> {
        self.insertion_order.iter().map(|s| s.as_str()).collect()
    }

    // -------------------------------------------------------------------
    // Category-based lookup
    // -------------------------------------------------------------------

    /// Return all nodes belonging to a specific category.
    pub fn nodes_by_category(&self, category: &NodeCategory) -> Vec<&NodeDescriptor> {
        self.insertion_order
            .iter()
            .filter_map(|name| {
                let node = self.nodes.get(name)?;
                if &node.category == category {
                    Some(node)
                } else {
                    None
                }
            })
            .collect()
    }

    /// Return the distinct categories currently in use.
    pub fn categories(&self) -> Vec<&NodeCategory> {
        let mut seen = HashSet::new();
        let mut result = Vec::new();
        for node in self.nodes.values() {
            if seen.insert(&node.category) {
                result.push(&node.category);
            }
        }
        result
    }

    /// Return the number of distinct categories.
    pub fn category_count(&self) -> usize {
        let set: HashSet<&NodeCategory> = self.nodes.values().map(|n| &n.category).collect();
        set.len()
    }

    // -------------------------------------------------------------------
    // Activation / deactivation
    // -------------------------------------------------------------------

    /// Activate a node (set status to `Active`).
    pub fn activate(&mut self, name: &str) -> Result<()> {
        let node = self
            .nodes
            .get_mut(name)
            .ok_or_else(|| Error::Configuration(format!("Unknown node '{}'", name)))?;
        node.status = NodeStatus::Active;
        self.stats.total_activations += 1;
        self.update_active_ratio();
        Ok(())
    }

    /// Deactivate a node (set status to `Inactive`).
    pub fn deactivate(&mut self, name: &str) -> Result<()> {
        let node = self
            .nodes
            .get_mut(name)
            .ok_or_else(|| Error::Configuration(format!("Unknown node '{}'", name)))?;
        node.status = NodeStatus::Inactive;
        self.stats.total_deactivations += 1;
        self.update_active_ratio();
        Ok(())
    }

    /// Mark a node as errored (set status to `Error`).
    pub fn mark_error(&mut self, name: &str) -> Result<()> {
        let node = self
            .nodes
            .get_mut(name)
            .ok_or_else(|| Error::Configuration(format!("Unknown node '{}'", name)))?;
        node.status = NodeStatus::Error;
        self.update_active_ratio();
        Ok(())
    }

    /// Return all nodes with the given status.
    pub fn nodes_with_status(&self, status: NodeStatus) -> Vec<&NodeDescriptor> {
        self.insertion_order
            .iter()
            .filter_map(|name| {
                let node = self.nodes.get(name)?;
                if node.status == status {
                    Some(node)
                } else {
                    None
                }
            })
            .collect()
    }

    /// Return the number of active nodes.
    pub fn active_count(&self) -> usize {
        self.nodes.values().filter(|n| n.status == NodeStatus::Active).count()
    }

    /// Return the number of inactive nodes.
    pub fn inactive_count(&self) -> usize {
        self.nodes.values().filter(|n| n.status == NodeStatus::Inactive).count()
    }

    /// Return the number of error nodes.
    pub fn error_count(&self) -> usize {
        self.nodes.values().filter(|n| n.status == NodeStatus::Error).count()
    }

    // -------------------------------------------------------------------
    // Execution recording
    // -------------------------------------------------------------------

    /// Record a successful execution of a node.
    pub fn record_execution(&mut self, name: &str, latency_us: f64) -> Result<()> {
        let decay = self.config.ema_decay;
        let tick = self.tick;
        let node = self
            .nodes
            .get_mut(name)
            .ok_or_else(|| Error::Configuration(format!("Unknown node '{}'", name)))?;
        node.execution_count += 1;
        node.last_latency_us = latency_us;
        node.last_executed_tick = tick;

        if node.execution_count == 1 {
            node.ema_latency_us = latency_us;
        } else {
            node.ema_latency_us = decay * latency_us + (1.0 - decay) * node.ema_latency_us;
        }

        self.stats.total_executions += 1;
        Ok(())
    }

    /// Record an error for a node.
    pub fn record_error(&mut self, name: &str) -> Result<()> {
        let node = self
            .nodes
            .get_mut(name)
            .ok_or_else(|| Error::Configuration(format!("Unknown node '{}'", name)))?;
        node.error_count += 1;
        self.stats.total_errors += 1;
        Ok(())
    }

    // -------------------------------------------------------------------
    // Dependency validation
    // -------------------------------------------------------------------

    /// Validate that all declared dependencies reference existing nodes.
    ///
    /// Returns a list of (node_name, missing_dependency_name) pairs for
    /// any broken references.
    pub fn validate_dependencies(&self) -> Vec<(String, String)> {
        let mut broken = Vec::new();
        for node in self.nodes.values() {
            for dep in &node.dependencies {
                if !self.nodes.contains_key(dep) {
                    broken.push((node.name.clone(), dep.clone()));
                }
            }
        }
        broken.sort();
        broken
    }

    /// Check whether all dependencies are satisfied (no broken references).
    pub fn all_dependencies_satisfied(&self) -> bool {
        self.validate_dependencies().is_empty()
    }

    /// Return the transitive dependency set for a node (all ancestors).
    pub fn transitive_dependencies(&self, name: &str) -> Result<HashSet<String>> {
        if !self.nodes.contains_key(name) {
            return Err(Error::Configuration(format!("Unknown node '{}'", name)));
        }

        let mut visited = HashSet::new();
        let mut stack = vec![name.to_string()];
        while let Some(current) = stack.pop() {
            if let Some(node) = self.nodes.get(&current) {
                for dep in &node.dependencies {
                    if visited.insert(dep.clone()) {
                        stack.push(dep.clone());
                    }
                }
            }
        }
        Ok(visited)
    }

    /// Return the dependents of a node (nodes that depend on it).
    pub fn dependents_of(&self, name: &str) -> Vec<&str> {
        self.nodes
            .values()
            .filter(|n| n.dependencies.iter().any(|d| d == name))
            .map(|n| n.name.as_str())
            .collect()
    }

    // -------------------------------------------------------------------
    // Tick
    // -------------------------------------------------------------------

    /// Advance the registry by one tick, updating EMA and windowed
    /// diagnostics.
    pub fn tick(&mut self) {
        self.tick += 1;

        let total = self.nodes.len();
        let active = self.active_count();
        let inactive = self.inactive_count();
        let error = self.error_count();
        let cats = self.category_count();
        let ratio = if total > 0 {
            active as f64 / total as f64
        } else {
            0.0
        };

        self.stats.active_ratio = ratio;

        let alpha = self.config.ema_decay;
        if !self.ema_initialized {
            self.stats.ema_active_ratio = ratio;
            self.stats.ema_node_count = total as f64;
            self.ema_initialized = true;
        } else {
            self.stats.ema_active_ratio =
                alpha * ratio + (1.0 - alpha) * self.stats.ema_active_ratio;
            self.stats.ema_node_count =
                alpha * total as f64 + (1.0 - alpha) * self.stats.ema_node_count;
        }

        let snapshot = TickSnapshot {
            total_nodes: total,
            active_nodes: active,
            inactive_nodes: inactive,
            error_nodes: error,
            category_count: cats,
            active_ratio: ratio,
        };
        if self.recent.len() >= self.config.window_size {
            self.recent.pop_front();
        }
        self.recent.push_back(snapshot);
    }

    /// Current tick counter.
    pub fn current_tick(&self) -> u64 {
        self.tick
    }

    /// Alias for `tick()`.
    pub fn process(&mut self) {
        self.tick();
    }

    // -------------------------------------------------------------------
    // Internal helpers
    // -------------------------------------------------------------------

    fn update_active_ratio(&mut self) {
        let total = self.nodes.len();
        self.stats.active_ratio = if total > 0 {
            self.active_count() as f64 / total as f64
        } else {
            0.0
        };
    }

    // -------------------------------------------------------------------
    // Statistics & diagnostics
    // -------------------------------------------------------------------

    /// Returns a reference to the cumulative statistics.
    pub fn stats(&self) -> &NodeRegistryStats {
        &self.stats
    }

    /// Returns a reference to the configuration.
    pub fn config(&self) -> &NodeRegistryConfig {
        &self.config
    }

    /// Current active ratio (fraction of nodes that are active).
    pub fn active_ratio(&self) -> f64 {
        self.stats.active_ratio
    }

    /// EMA-smoothed active ratio.
    pub fn smoothed_active_ratio(&self) -> f64 {
        self.stats.ema_active_ratio
    }

    /// EMA-smoothed node count.
    pub fn smoothed_node_count(&self) -> f64 {
        self.stats.ema_node_count
    }

    /// Windowed average active ratio.
    pub fn windowed_active_ratio(&self) -> Option<f64> {
        if self.recent.is_empty() {
            return None;
        }
        let sum: f64 = self.recent.iter().map(|s| s.active_ratio).sum();
        Some(sum / self.recent.len() as f64)
    }

    /// Windowed average node count.
    pub fn windowed_node_count(&self) -> Option<f64> {
        if self.recent.is_empty() {
            return None;
        }
        let sum: f64 = self.recent.iter().map(|s| s.total_nodes as f64).sum();
        Some(sum / self.recent.len() as f64)
    }

    /// Windowed average error-node count.
    pub fn windowed_error_count(&self) -> Option<f64> {
        if self.recent.is_empty() {
            return None;
        }
        let sum: f64 = self.recent.iter().map(|s| s.error_nodes as f64).sum();
        Some(sum / self.recent.len() as f64)
    }

    /// Whether the active ratio appears to be declining over the window.
    pub fn is_active_ratio_declining(&self) -> bool {
        if self.recent.len() < 4 {
            return false;
        }
        let mid = self.recent.len() / 2;
        let first_half: f64 =
            self.recent.iter().take(mid).map(|s| s.active_ratio).sum::<f64>() / mid as f64;
        let second_half: f64 = self
            .recent
            .iter()
            .skip(mid)
            .map(|s| s.active_ratio)
            .sum::<f64>()
            / (self.recent.len() - mid) as f64;
        second_half < first_half - 0.05
    }

    // -------------------------------------------------------------------
    // Reset
    // -------------------------------------------------------------------

    /// Reset all internal state, preserving configuration. All nodes are
    /// removed.
    pub fn reset(&mut self) {
        self.nodes.clear();
        self.insertion_order.clear();
        self.tick = 0;
        self.ema_initialized = false;
        self.recent.clear();
        self.stats = NodeRegistryStats::default();
    }
}

// ===========================================================================
// Tests
// ===========================================================================

#[cfg(test)]
mod tests {
    use super::*;

    fn small_config() -> NodeRegistryConfig {
        NodeRegistryConfig {
            max_nodes: 8,
            max_categories: 4,
            ema_decay: 0.5,
            window_size: 5,
        }
    }

    // -------------------------------------------------------------------
    // Construction
    // -------------------------------------------------------------------

    #[test]
    fn test_new_default() {
        let reg = NodeRegistry::new();
        assert_eq!(reg.node_count(), 0);
        assert_eq!(reg.current_tick(), 0);
    }

    #[test]
    fn test_with_config() {
        let reg = NodeRegistry::with_config(small_config());
        assert_eq!(reg.config().max_nodes, 8);
    }

    // -------------------------------------------------------------------
    // Registration
    // -------------------------------------------------------------------

    #[test]
    fn test_register() {
        let mut reg = NodeRegistry::with_config(small_config());
        reg.register("parser", NodeCategory::Input, "Parse input", vec![], "1.0")
            .unwrap();
        assert_eq!(reg.node_count(), 1);
        assert!(reg.contains("parser"));
        let node = reg.get("parser").unwrap();
        assert_eq!(node.category, NodeCategory::Input);
        assert_eq!(node.status, NodeStatus::Active);
        assert_eq!(node.version, "1.0");
    }

    #[test]
    fn test_register_duplicate_fails() {
        let mut reg = NodeRegistry::with_config(small_config());
        reg.register("a", NodeCategory::Input, "", vec![], "1.0").unwrap();
        assert!(reg.register("a", NodeCategory::Input, "", vec![], "1.0").is_err());
    }

    #[test]
    fn test_register_at_capacity() {
        let mut reg = NodeRegistry::with_config(small_config());
        for i in 0..8 {
            reg.register(format!("n{}", i), NodeCategory::Transform, "", vec![], "1.0")
                .unwrap();
        }
        assert!(reg
            .register("overflow", NodeCategory::Transform, "", vec![], "1.0")
            .is_err());
    }

    #[test]
    fn test_remove() {
        let mut reg = NodeRegistry::with_config(small_config());
        reg.register("a", NodeCategory::Input, "desc", vec![], "1.0")
            .unwrap();
        let removed = reg.remove("a").unwrap();
        assert_eq!(removed.name, "a");
        assert!(!reg.contains("a"));
        assert_eq!(reg.node_count(), 0);
        assert_eq!(reg.stats().total_removed, 1);
    }

    #[test]
    fn test_remove_unknown_fails() {
        let mut reg = NodeRegistry::with_config(small_config());
        assert!(reg.remove("nope").is_err());
    }

    #[test]
    fn test_node_names_insertion_order() {
        let mut reg = NodeRegistry::with_config(small_config());
        reg.register("b", NodeCategory::Input, "", vec![], "1.0").unwrap();
        reg.register("a", NodeCategory::Input, "", vec![], "1.0").unwrap();
        reg.register("c", NodeCategory::Input, "", vec![], "1.0").unwrap();
        assert_eq!(reg.node_names(), vec!["b", "a", "c"]);
    }

    // -------------------------------------------------------------------
    // Category-based lookup
    // -------------------------------------------------------------------

    #[test]
    fn test_nodes_by_category() {
        let mut reg = NodeRegistry::with_config(small_config());
        reg.register("in1", NodeCategory::Input, "", vec![], "1.0").unwrap();
        reg.register("in2", NodeCategory::Input, "", vec![], "1.0").unwrap();
        reg.register("tx1", NodeCategory::Transform, "", vec![], "1.0")
            .unwrap();

        let inputs = reg.nodes_by_category(&NodeCategory::Input);
        assert_eq!(inputs.len(), 2);
        let transforms = reg.nodes_by_category(&NodeCategory::Transform);
        assert_eq!(transforms.len(), 1);
        let outputs = reg.nodes_by_category(&NodeCategory::Output);
        assert!(outputs.is_empty());
    }

    #[test]
    fn test_category_count() {
        let mut reg = NodeRegistry::with_config(small_config());
        reg.register("a", NodeCategory::Input, "", vec![], "1.0").unwrap();
        reg.register("b", NodeCategory::Transform, "", vec![], "1.0")
            .unwrap();
        reg.register("c", NodeCategory::Input, "", vec![], "1.0").unwrap();
        assert_eq!(reg.category_count(), 2);
    }

    #[test]
    fn test_custom_category() {
        let mut reg = NodeRegistry::with_config(small_config());
        reg.register(
            "special",
            NodeCategory::Custom("MyType".into()),
            "",
            vec![],
            "1.0",
        )
        .unwrap();
        let nodes = reg.nodes_by_category(&NodeCategory::Custom("MyType".into()));
        assert_eq!(nodes.len(), 1);
    }

    // -------------------------------------------------------------------
    // Activation / deactivation
    // -------------------------------------------------------------------

    #[test]
    fn test_activate_deactivate() {
        let mut reg = NodeRegistry::with_config(small_config());
        reg.register("a", NodeCategory::Input, "", vec![], "1.0").unwrap();
        assert_eq!(reg.active_count(), 1);

        reg.deactivate("a").unwrap();
        assert_eq!(reg.get("a").unwrap().status, NodeStatus::Inactive);
        assert_eq!(reg.active_count(), 0);
        assert_eq!(reg.inactive_count(), 1);

        reg.activate("a").unwrap();
        assert_eq!(reg.get("a").unwrap().status, NodeStatus::Active);
        assert_eq!(reg.active_count(), 1);
    }

    #[test]
    fn test_mark_error() {
        let mut reg = NodeRegistry::with_config(small_config());
        reg.register("a", NodeCategory::Input, "", vec![], "1.0").unwrap();
        reg.mark_error("a").unwrap();
        assert_eq!(reg.get("a").unwrap().status, NodeStatus::Error);
        assert_eq!(reg.error_count(), 1);
    }

    #[test]
    fn test_activate_unknown_fails() {
        let mut reg = NodeRegistry::with_config(small_config());
        assert!(reg.activate("nope").is_err());
        assert!(reg.deactivate("nope").is_err());
        assert!(reg.mark_error("nope").is_err());
    }

    #[test]
    fn test_nodes_with_status() {
        let mut reg = NodeRegistry::with_config(small_config());
        reg.register("a", NodeCategory::Input, "", vec![], "1.0").unwrap();
        reg.register("b", NodeCategory::Input, "", vec![], "1.0").unwrap();
        reg.deactivate("b").unwrap();

        let active = reg.nodes_with_status(NodeStatus::Active);
        assert_eq!(active.len(), 1);
        assert_eq!(active[0].name, "a");

        let inactive = reg.nodes_with_status(NodeStatus::Inactive);
        assert_eq!(inactive.len(), 1);
        assert_eq!(inactive[0].name, "b");
    }

    #[test]
    fn test_active_ratio() {
        let mut reg = NodeRegistry::with_config(small_config());
        reg.register("a", NodeCategory::Input, "", vec![], "1.0").unwrap();
        reg.register("b", NodeCategory::Input, "", vec![], "1.0").unwrap();
        assert!((reg.active_ratio() - 1.0).abs() < 1e-9);

        reg.deactivate("b").unwrap();
        assert!((reg.active_ratio() - 0.5).abs() < 1e-9);
    }

    // -------------------------------------------------------------------
    // Execution recording
    // -------------------------------------------------------------------

    #[test]
    fn test_record_execution() {
        let mut reg = NodeRegistry::with_config(small_config());
        reg.register("a", NodeCategory::Input, "", vec![], "1.0").unwrap();
        reg.record_execution("a", 100.0).unwrap();
        let node = reg.get("a").unwrap();
        assert_eq!(node.execution_count, 1);
        assert!((node.last_latency_us - 100.0).abs() < 1e-9);
        assert!((node.ema_latency_us - 100.0).abs() < 1e-9);
        assert_eq!(reg.stats().total_executions, 1);
    }

    #[test]
    fn test_record_execution_ema() {
        let mut reg = NodeRegistry::with_config(small_config()); // ema_decay = 0.5
        reg.register("a", NodeCategory::Input, "", vec![], "1.0").unwrap();
        reg.record_execution("a", 100.0).unwrap(); // ema = 100
        reg.record_execution("a", 200.0).unwrap(); // ema = 0.5*200 + 0.5*100 = 150
        assert!((reg.get("a").unwrap().ema_latency_us - 150.0).abs() < 1e-9);
    }

    #[test]
    fn test_record_error() {
        let mut reg = NodeRegistry::with_config(small_config());
        reg.register("a", NodeCategory::Input, "", vec![], "1.0").unwrap();
        reg.record_error("a").unwrap();
        assert_eq!(reg.get("a").unwrap().error_count, 1);
        assert_eq!(reg.stats().total_errors, 1);
    }

    #[test]
    fn test_record_execution_unknown_fails() {
        let mut reg = NodeRegistry::with_config(small_config());
        assert!(reg.record_execution("nope", 10.0).is_err());
        assert!(reg.record_error("nope").is_err());
    }

    // -------------------------------------------------------------------
    // Dependency validation
    // -------------------------------------------------------------------

    #[test]
    fn test_all_dependencies_satisfied() {
        let mut reg = NodeRegistry::with_config(small_config());
        reg.register("a", NodeCategory::Input, "", vec![], "1.0").unwrap();
        reg.register("b", NodeCategory::Transform, "", vec!["a".into()], "1.0")
            .unwrap();
        assert!(reg.all_dependencies_satisfied());
    }

    #[test]
    fn test_broken_dependencies() {
        let mut reg = NodeRegistry::with_config(small_config());
        reg.register("b", NodeCategory::Transform, "", vec!["missing".into()], "1.0")
            .unwrap();
        let broken = reg.validate_dependencies();
        assert_eq!(broken.len(), 1);
        assert_eq!(broken[0], ("b".to_string(), "missing".to_string()));
        assert!(!reg.all_dependencies_satisfied());
    }

    #[test]
    fn test_transitive_dependencies() {
        let mut reg = NodeRegistry::with_config(small_config());
        reg.register("a", NodeCategory::Input, "", vec![], "1.0").unwrap();
        reg.register("b", NodeCategory::Transform, "", vec!["a".into()], "1.0")
            .unwrap();
        reg.register("c", NodeCategory::Output, "", vec!["b".into()], "1.0")
            .unwrap();

        let deps = reg.transitive_dependencies("c").unwrap();
        assert!(deps.contains("a"));
        assert!(deps.contains("b"));
        assert!(!deps.contains("c"));
    }

    #[test]
    fn test_transitive_dependencies_unknown_fails() {
        let reg = NodeRegistry::with_config(small_config());
        assert!(reg.transitive_dependencies("nope").is_err());
    }

    #[test]
    fn test_dependents_of() {
        let mut reg = NodeRegistry::with_config(small_config());
        reg.register("a", NodeCategory::Input, "", vec![], "1.0").unwrap();
        reg.register("b", NodeCategory::Transform, "", vec!["a".into()], "1.0")
            .unwrap();
        reg.register("c", NodeCategory::Output, "", vec!["a".into()], "1.0")
            .unwrap();
        let mut deps = reg.dependents_of("a");
        deps.sort();
        assert_eq!(deps, vec!["b", "c"]);
    }

    // -------------------------------------------------------------------
    // Tick & EMA
    // -------------------------------------------------------------------

    #[test]
    fn test_tick_increments() {
        let mut reg = NodeRegistry::with_config(small_config());
        reg.tick();
        reg.tick();
        assert_eq!(reg.current_tick(), 2);
    }

    #[test]
    fn test_process_alias() {
        let mut reg = NodeRegistry::with_config(small_config());
        reg.process();
        assert_eq!(reg.current_tick(), 1);
    }

    #[test]
    fn test_ema_initialises_on_first_tick() {
        let mut reg = NodeRegistry::with_config(small_config());
        reg.register("a", NodeCategory::Input, "", vec![], "1.0").unwrap();
        reg.tick();
        assert!((reg.smoothed_active_ratio() - 1.0).abs() < 1e-9);
        assert!((reg.smoothed_node_count() - 1.0).abs() < 1e-9);
    }

    #[test]
    fn test_ema_blends_over_ticks() {
        let mut reg = NodeRegistry::with_config(small_config()); // decay=0.5
        reg.register("a", NodeCategory::Input, "", vec![], "1.0").unwrap();
        reg.register("b", NodeCategory::Input, "", vec![], "1.0").unwrap();
        reg.tick(); // ema_active_ratio = 1.0 (init)

        reg.deactivate("b").unwrap();
        reg.tick(); // ema = 0.5*0.5 + 0.5*1.0 = 0.75
        assert!((reg.smoothed_active_ratio() - 0.75).abs() < 1e-9);
    }

    // -------------------------------------------------------------------
    // Windowed diagnostics
    // -------------------------------------------------------------------

    #[test]
    fn test_windowed_active_ratio_empty() {
        let reg = NodeRegistry::with_config(small_config());
        assert!(reg.windowed_active_ratio().is_none());
    }

    #[test]
    fn test_windowed_active_ratio() {
        let mut reg = NodeRegistry::with_config(small_config());
        reg.register("a", NodeCategory::Input, "", vec![], "1.0").unwrap();
        reg.tick();
        reg.tick();
        let ratio = reg.windowed_active_ratio().unwrap();
        assert!((ratio - 1.0).abs() < 1e-9);
    }

    #[test]
    fn test_windowed_node_count() {
        let mut reg = NodeRegistry::with_config(small_config());
        reg.register("a", NodeCategory::Input, "", vec![], "1.0").unwrap();
        reg.tick();
        let count = reg.windowed_node_count().unwrap();
        assert!((count - 1.0).abs() < 1e-9);
    }

    #[test]
    fn test_windowed_error_count() {
        let mut reg = NodeRegistry::with_config(small_config());
        reg.register("a", NodeCategory::Input, "", vec![], "1.0").unwrap();
        reg.mark_error("a").unwrap();
        reg.tick();
        let count = reg.windowed_error_count().unwrap();
        assert!((count - 1.0).abs() < 1e-9);
    }

    #[test]
    fn test_is_active_ratio_declining() {
        let mut reg = NodeRegistry::with_config(small_config());
        reg.register("a", NodeCategory::Input, "", vec![], "1.0").unwrap();
        reg.register("b", NodeCategory::Input, "", vec![], "1.0").unwrap();
        reg.register("c", NodeCategory::Input, "", vec![], "1.0").unwrap();
        // First half: all active
        reg.tick();
        reg.tick();
        // Second half: deactivate two
        reg.deactivate("b").unwrap();
        reg.deactivate("c").unwrap();
        reg.tick();
        reg.tick();
        assert!(reg.is_active_ratio_declining());
    }

    #[test]
    fn test_is_active_ratio_declining_insufficient_data() {
        let mut reg = NodeRegistry::with_config(small_config());
        reg.tick();
        assert!(!reg.is_active_ratio_declining());
    }

    #[test]
    fn test_window_rolls() {
        let mut reg = NodeRegistry::with_config(small_config()); // window_size=5
        reg.register("a", NodeCategory::Input, "", vec![], "1.0").unwrap();
        for _ in 0..20 {
            reg.tick();
        }
        assert!(reg.recent.len() <= 5);
    }

    // -------------------------------------------------------------------
    // Reset
    // -------------------------------------------------------------------

    #[test]
    fn test_reset() {
        let mut reg = NodeRegistry::with_config(small_config());
        reg.register("a", NodeCategory::Input, "", vec![], "1.0").unwrap();
        reg.record_execution("a", 50.0).unwrap();
        reg.tick();
        reg.tick();

        reg.reset();
        assert_eq!(reg.node_count(), 0);
        assert_eq!(reg.current_tick(), 0);
        assert_eq!(reg.stats().total_registered, 0);
        assert_eq!(reg.stats().total_executions, 0);
        assert!(reg.windowed_active_ratio().is_none());
    }

    // -------------------------------------------------------------------
    // Full lifecycle
    // -------------------------------------------------------------------

    #[test]
    fn test_full_lifecycle() {
        let mut reg = NodeRegistry::with_config(small_config());

        // Register nodes
        reg.register("input", NodeCategory::Input, "Data ingestion", vec![], "1.0")
            .unwrap();
        reg.register(
            "transform",
            NodeCategory::Transform,
            "Feature extraction",
            vec!["input".into()],
            "2.0",
        )
        .unwrap();
        reg.register(
            "output",
            NodeCategory::Output,
            "Signal emission",
            vec!["transform".into()],
            "1.0",
        )
        .unwrap();

        // Validate deps
        assert!(reg.all_dependencies_satisfied());
        let trans_deps = reg.transitive_dependencies("output").unwrap();
        assert!(trans_deps.contains("input"));
        assert!(trans_deps.contains("transform"));

        // Execute
        reg.tick();
        reg.record_execution("input", 10.0).unwrap();
        reg.record_execution("transform", 50.0).unwrap();
        reg.record_execution("output", 5.0).unwrap();

        // Deactivate one
        reg.deactivate("output").unwrap();
        reg.tick();
        assert!((reg.active_ratio() - 2.0 / 3.0).abs() < 1e-9);

        // Re-activate
        reg.activate("output").unwrap();
        reg.tick();

        // Stats
        assert_eq!(reg.stats().total_registered, 3);
        assert_eq!(reg.stats().total_executions, 3);
        assert_eq!(reg.stats().total_activations, 1);
        assert_eq!(reg.stats().total_deactivations, 1);
        assert!(reg.windowed_active_ratio().is_some());
    }

    // -------------------------------------------------------------------
    // Display impls
    // -------------------------------------------------------------------

    #[test]
    fn test_display() {
        assert_eq!(format!("{}", NodeCategory::Input), "Input");
        assert_eq!(format!("{}", NodeCategory::Custom("X".into())), "Custom(X)");
        assert_eq!(format!("{}", NodeStatus::Active), "Active");
        assert_eq!(format!("{}", NodeStatus::Inactive), "Inactive");
        assert_eq!(format!("{}", NodeStatus::Error), "Error");
    }
}
