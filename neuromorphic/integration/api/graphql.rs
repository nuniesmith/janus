//! GraphQL API
//!
//! Part of the Integration region — API component.
//!
//! `Graphql` manages a simulated GraphQL schema with registered query and
//! mutation fields, tracks per-field resolution statistics, computes query
//! complexity scores, monitors per-query latency, and exposes EMA-smoothed
//! and windowed diagnostics.

use crate::common::{Error, Result};
use std::collections::{HashMap, VecDeque};

// ---------------------------------------------------------------------------
// Configuration
// ---------------------------------------------------------------------------

/// Configuration for the GraphQL API tracker.
#[derive(Debug, Clone)]
pub struct GraphqlConfig {
    /// Maximum number of fields that can be registered in the schema.
    pub max_fields: usize,
    /// Maximum query history to retain.
    pub max_query_history: usize,
    /// Complexity threshold — queries above this score trigger a warning.
    pub complexity_threshold: f64,
    /// Latency threshold (µs) above which a query is considered slow.
    pub slow_query_threshold_us: f64,
    /// EMA decay factor (0 < α < 1). Higher → faster adaptation.
    pub ema_decay: f64,
    /// Window size for rolling diagnostics.
    pub window_size: usize,
}

impl Default for GraphqlConfig {
    fn default() -> Self {
        Self {
            max_fields: 256,
            max_query_history: 200,
            complexity_threshold: 100.0,
            slow_query_threshold_us: 5000.0,
            ema_decay: 0.1,
            window_size: 50,
        }
    }
}

// ---------------------------------------------------------------------------
// Field kind
// ---------------------------------------------------------------------------

/// The kind of a GraphQL field.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash)]
pub enum FieldKind {
    /// A read-only query field.
    Query,
    /// A write / side-effect mutation field.
    Mutation,
    /// A real-time subscription field.
    Subscription,
}

impl std::fmt::Display for FieldKind {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        match self {
            FieldKind::Query => write!(f, "Query"),
            FieldKind::Mutation => write!(f, "Mutation"),
            FieldKind::Subscription => write!(f, "Subscription"),
        }
    }
}

// ---------------------------------------------------------------------------
// Schema field descriptor
// ---------------------------------------------------------------------------

/// Descriptor for a registered GraphQL field.
#[derive(Debug, Clone)]
pub struct FieldDescriptor {
    /// Field name (unique within its kind).
    pub name: String,
    /// Kind of field.
    pub kind: FieldKind,
    /// Human-readable description.
    pub description: String,
    /// Complexity weight — contributes to the overall query complexity score.
    pub complexity_weight: f64,
    /// Whether the field is deprecated.
    pub deprecated: bool,
    /// Total number of times this field has been resolved.
    pub resolution_count: u64,
    /// Total errors encountered during resolution.
    pub error_count: u64,
    /// Last recorded resolution latency in µs.
    pub last_latency_us: f64,
    /// EMA-smoothed resolution latency.
    pub ema_latency_us: f64,
}

// ---------------------------------------------------------------------------
// Query record
// ---------------------------------------------------------------------------

/// Record of a single GraphQL query execution.
#[derive(Debug, Clone)]
pub struct QueryRecord {
    /// Monotonically increasing query identifier.
    pub id: u64,
    /// Tick at which the query was executed.
    pub tick: u64,
    /// Names of the fields resolved in this query.
    pub fields: Vec<String>,
    /// Computed complexity score (sum of field weights × depth).
    pub complexity: f64,
    /// Total latency for the query in µs.
    pub latency_us: f64,
    /// Whether the query exceeded the complexity threshold.
    pub exceeded_complexity: bool,
    /// Whether the query exceeded the slow-query threshold.
    pub was_slow: bool,
    /// Number of field-level errors during resolution.
    pub errors: usize,
}

// ---------------------------------------------------------------------------
// Tick snapshot (windowed diagnostics)
// ---------------------------------------------------------------------------

#[derive(Debug, Clone)]
struct TickSnapshot {
    queries_executed: u64,
    avg_latency_us: f64,
    avg_complexity: f64,
    slow_queries: u64,
    error_queries: u64,
}

// ---------------------------------------------------------------------------
// Statistics
// ---------------------------------------------------------------------------

/// Cumulative statistics for the GraphQL API tracker.
#[derive(Debug, Clone)]
pub struct GraphqlStats {
    /// Total queries executed.
    pub total_queries: u64,
    /// Total field resolutions across all queries.
    pub total_resolutions: u64,
    /// Total field-level errors across all queries.
    pub total_field_errors: u64,
    /// Total queries that exceeded the complexity threshold.
    pub total_complex_queries: u64,
    /// Total queries that exceeded the slow-query threshold.
    pub total_slow_queries: u64,
    /// EMA-smoothed query latency.
    pub ema_query_latency_us: f64,
    /// EMA-smoothed query complexity.
    pub ema_query_complexity: f64,
    /// EMA-smoothed queries per tick.
    pub ema_queries_per_tick: f64,
}

impl Default for GraphqlStats {
    fn default() -> Self {
        Self {
            total_queries: 0,
            total_resolutions: 0,
            total_field_errors: 0,
            total_complex_queries: 0,
            total_slow_queries: 0,
            ema_query_latency_us: 0.0,
            ema_query_complexity: 0.0,
            ema_queries_per_tick: 0.0,
        }
    }
}

// ---------------------------------------------------------------------------
// Graphql
// ---------------------------------------------------------------------------

/// GraphQL API schema and query tracker.
///
/// Manages registered fields (query, mutation, subscription), tracks
/// per-field resolution statistics, computes query complexity, monitors
/// latency, and provides EMA + windowed diagnostics.
pub struct Graphql {
    config: GraphqlConfig,
    /// Registered fields keyed by name.
    fields: HashMap<String, FieldDescriptor>,
    /// Insertion-order tracking.
    field_order: Vec<String>,
    /// Query history.
    query_history: VecDeque<QueryRecord>,
    /// Next query id.
    next_query_id: u64,
    /// Current tick.
    tick: u64,
    /// Queries executed in the current tick (for per-tick EMA).
    queries_this_tick: u64,
    /// Latency accumulator for current tick.
    latency_sum_this_tick: f64,
    /// Complexity accumulator for current tick.
    complexity_sum_this_tick: f64,
    /// Slow queries in the current tick.
    slow_this_tick: u64,
    /// Error queries in the current tick.
    errors_this_tick: u64,
    /// Whether EMA has been initialised.
    ema_initialized: bool,
    /// Rolling window of tick snapshots.
    recent: VecDeque<TickSnapshot>,
    /// Cumulative statistics.
    stats: GraphqlStats,
}

impl Default for Graphql {
    fn default() -> Self {
        Self::new()
    }
}

impl Graphql {
    // -------------------------------------------------------------------
    // Construction
    // -------------------------------------------------------------------

    /// Create a new GraphQL tracker with default configuration.
    pub fn new() -> Self {
        Self::with_config(GraphqlConfig::default())
    }

    /// Create a new GraphQL tracker with the given configuration.
    pub fn with_config(config: GraphqlConfig) -> Self {
        Self {
            fields: HashMap::new(),
            field_order: Vec::new(),
            query_history: VecDeque::with_capacity(config.max_query_history.min(256)),
            next_query_id: 0,
            tick: 0,
            queries_this_tick: 0,
            latency_sum_this_tick: 0.0,
            complexity_sum_this_tick: 0.0,
            slow_this_tick: 0,
            errors_this_tick: 0,
            ema_initialized: false,
            recent: VecDeque::with_capacity(config.window_size.min(256)),
            stats: GraphqlStats::default(),
            config,
        }
    }

    // -------------------------------------------------------------------
    // Schema management
    // -------------------------------------------------------------------

    /// Register a new field in the schema.
    pub fn register_field(
        &mut self,
        name: impl Into<String>,
        kind: FieldKind,
        description: impl Into<String>,
        complexity_weight: f64,
    ) -> Result<()> {
        let name = name.into();
        if self.fields.contains_key(&name) {
            return Err(Error::Configuration(format!(
                "Field '{}' is already registered",
                name
            )));
        }
        if self.fields.len() >= self.config.max_fields {
            return Err(Error::Configuration(format!(
                "Maximum field count ({}) reached",
                self.config.max_fields
            )));
        }

        self.fields.insert(
            name.clone(),
            FieldDescriptor {
                name: name.clone(),
                kind,
                description: description.into(),
                complexity_weight,
                deprecated: false,
                resolution_count: 0,
                error_count: 0,
                last_latency_us: 0.0,
                ema_latency_us: 0.0,
            },
        );
        self.field_order.push(name);
        Ok(())
    }

    /// Mark a field as deprecated.
    pub fn deprecate_field(&mut self, name: &str) -> Result<()> {
        let field = self
            .fields
            .get_mut(name)
            .ok_or_else(|| Error::Configuration(format!("Unknown field '{}'", name)))?;
        field.deprecated = true;
        Ok(())
    }

    /// Remove a field from the schema.
    pub fn remove_field(&mut self, name: &str) -> Result<()> {
        self.fields
            .remove(name)
            .ok_or_else(|| Error::Configuration(format!("Unknown field '{}'", name)))?;
        self.field_order.retain(|n| n != name);
        Ok(())
    }

    /// Look up a field descriptor by name.
    pub fn field(&self, name: &str) -> Option<&FieldDescriptor> {
        self.fields.get(name)
    }

    /// Return the number of registered fields.
    pub fn field_count(&self) -> usize {
        self.fields.len()
    }

    /// Return the names of all registered fields in insertion order.
    pub fn field_names(&self) -> Vec<&str> {
        self.field_order.iter().map(|s| s.as_str()).collect()
    }

    /// Return all fields of a given kind.
    pub fn fields_by_kind(&self, kind: FieldKind) -> Vec<&FieldDescriptor> {
        self.field_order
            .iter()
            .filter_map(|name| {
                let f = self.fields.get(name)?;
                if f.kind == kind {
                    Some(f)
                } else {
                    None
                }
            })
            .collect()
    }

    /// Return all deprecated fields.
    pub fn deprecated_fields(&self) -> Vec<&FieldDescriptor> {
        self.field_order
            .iter()
            .filter_map(|name| {
                let f = self.fields.get(name)?;
                if f.deprecated {
                    Some(f)
                } else {
                    None
                }
            })
            .collect()
    }

    // -------------------------------------------------------------------
    // Complexity scoring
    // -------------------------------------------------------------------

    /// Compute the complexity score for a set of field names and a depth
    /// multiplier. Complexity = Σ(field_weight × depth).
    pub fn compute_complexity(&self, field_names: &[&str], depth: f64) -> f64 {
        field_names
            .iter()
            .filter_map(|name| self.fields.get(*name))
            .map(|f| f.complexity_weight * depth)
            .sum()
    }

    /// Whether the given complexity score exceeds the configured threshold.
    pub fn is_complex(&self, complexity: f64) -> bool {
        complexity > self.config.complexity_threshold
    }

    // -------------------------------------------------------------------
    // Query execution
    // -------------------------------------------------------------------

    /// Execute a simulated query. The provided closure is called for each
    /// field name and must return `Ok(latency_us)` on success or `Err` on
    /// failure.
    ///
    /// Fields not registered in the schema are silently skipped.
    pub fn execute_query<F>(
        &mut self,
        field_names: &[&str],
        depth: f64,
        mut resolve_fn: F,
    ) -> Result<QueryRecord>
    where
        F: FnMut(&str) -> Result<f64>,
    {
        let complexity = self.compute_complexity(field_names, depth);
        let exceeded_complexity = self.is_complex(complexity);
        let mut total_latency = 0.0;
        let mut resolved_fields: Vec<String> = Vec::new();
        let mut errors = 0usize;
        let decay = self.config.ema_decay;

        for &name in field_names {
            if !self.fields.contains_key(name) {
                continue;
            }

            match resolve_fn(name) {
                Ok(latency_us) => {
                    let field = self.fields.get_mut(name).unwrap();
                    field.resolution_count += 1;
                    field.last_latency_us = latency_us;
                    if field.resolution_count == 1 {
                        field.ema_latency_us = latency_us;
                    } else {
                        field.ema_latency_us =
                            decay * latency_us + (1.0 - decay) * field.ema_latency_us;
                    }
                    total_latency += latency_us;
                    resolved_fields.push(name.to_string());
                    self.stats.total_resolutions += 1;
                }
                Err(_) => {
                    let field = self.fields.get_mut(name).unwrap();
                    field.error_count += 1;
                    errors += 1;
                    self.stats.total_field_errors += 1;
                }
            }
        }

        let was_slow = total_latency > self.config.slow_query_threshold_us;

        self.next_query_id += 1;
        self.stats.total_queries += 1;
        if exceeded_complexity {
            self.stats.total_complex_queries += 1;
        }
        if was_slow {
            self.stats.total_slow_queries += 1;
            self.slow_this_tick += 1;
        }
        if errors > 0 {
            self.errors_this_tick += 1;
        }

        self.queries_this_tick += 1;
        self.latency_sum_this_tick += total_latency;
        self.complexity_sum_this_tick += complexity;

        let record = QueryRecord {
            id: self.next_query_id,
            tick: self.tick,
            fields: resolved_fields,
            complexity,
            latency_us: total_latency,
            exceeded_complexity,
            was_slow,
            errors,
        };

        if self.query_history.len() >= self.config.max_query_history {
            self.query_history.pop_front();
        }
        self.query_history.push_back(record.clone());

        Ok(record)
    }

    // -------------------------------------------------------------------
    // Query history
    // -------------------------------------------------------------------

    /// Return the most recent query record.
    pub fn last_query(&self) -> Option<&QueryRecord> {
        self.query_history.back()
    }

    /// Return all stored query records.
    pub fn query_history(&self) -> &VecDeque<QueryRecord> {
        &self.query_history
    }

    // -------------------------------------------------------------------
    // Tick
    // -------------------------------------------------------------------

    /// Advance the tracker by one tick, updating EMA and windowed
    /// diagnostics.
    pub fn tick(&mut self) {
        self.tick += 1;

        let avg_latency = if self.queries_this_tick > 0 {
            self.latency_sum_this_tick / self.queries_this_tick as f64
        } else {
            0.0
        };
        let avg_complexity = if self.queries_this_tick > 0 {
            self.complexity_sum_this_tick / self.queries_this_tick as f64
        } else {
            0.0
        };

        let snapshot = TickSnapshot {
            queries_executed: self.queries_this_tick,
            avg_latency_us: avg_latency,
            avg_complexity,
            slow_queries: self.slow_this_tick,
            error_queries: self.errors_this_tick,
        };

        // EMA update.
        let alpha = self.config.ema_decay;
        if !self.ema_initialized && self.queries_this_tick > 0 {
            self.stats.ema_query_latency_us = avg_latency;
            self.stats.ema_query_complexity = avg_complexity;
            self.stats.ema_queries_per_tick = self.queries_this_tick as f64;
            self.ema_initialized = true;
        } else if self.ema_initialized {
            self.stats.ema_query_latency_us =
                alpha * avg_latency + (1.0 - alpha) * self.stats.ema_query_latency_us;
            self.stats.ema_query_complexity =
                alpha * avg_complexity + (1.0 - alpha) * self.stats.ema_query_complexity;
            self.stats.ema_queries_per_tick =
                alpha * self.queries_this_tick as f64
                    + (1.0 - alpha) * self.stats.ema_queries_per_tick;
        }

        // Window update.
        if self.recent.len() >= self.config.window_size {
            self.recent.pop_front();
        }
        self.recent.push_back(snapshot);

        // Reset per-tick counters.
        self.queries_this_tick = 0;
        self.latency_sum_this_tick = 0.0;
        self.complexity_sum_this_tick = 0.0;
        self.slow_this_tick = 0;
        self.errors_this_tick = 0;
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
    // Statistics & diagnostics
    // -------------------------------------------------------------------

    /// Returns a reference to cumulative statistics.
    pub fn stats(&self) -> &GraphqlStats {
        &self.stats
    }

    /// Returns a reference to the configuration.
    pub fn config(&self) -> &GraphqlConfig {
        &self.config
    }

    /// EMA-smoothed query latency.
    pub fn smoothed_query_latency(&self) -> f64 {
        self.stats.ema_query_latency_us
    }

    /// EMA-smoothed query complexity.
    pub fn smoothed_query_complexity(&self) -> f64 {
        self.stats.ema_query_complexity
    }

    /// EMA-smoothed queries per tick.
    pub fn smoothed_queries_per_tick(&self) -> f64 {
        self.stats.ema_queries_per_tick
    }

    /// Windowed average query latency.
    pub fn windowed_query_latency(&self) -> Option<f64> {
        if self.recent.is_empty() {
            return None;
        }
        let sum: f64 = self.recent.iter().map(|s| s.avg_latency_us).sum();
        Some(sum / self.recent.len() as f64)
    }

    /// Windowed average query complexity.
    pub fn windowed_query_complexity(&self) -> Option<f64> {
        if self.recent.is_empty() {
            return None;
        }
        let sum: f64 = self.recent.iter().map(|s| s.avg_complexity).sum();
        Some(sum / self.recent.len() as f64)
    }

    /// Windowed average queries per tick.
    pub fn windowed_queries_per_tick(&self) -> Option<f64> {
        if self.recent.is_empty() {
            return None;
        }
        let sum: f64 = self
            .recent
            .iter()
            .map(|s| s.queries_executed as f64)
            .sum();
        Some(sum / self.recent.len() as f64)
    }

    /// Windowed average slow-query count per tick.
    pub fn windowed_slow_queries(&self) -> Option<f64> {
        if self.recent.is_empty() {
            return None;
        }
        let sum: f64 = self.recent.iter().map(|s| s.slow_queries as f64).sum();
        Some(sum / self.recent.len() as f64)
    }

    /// Whether query latency is trending upward over the window.
    pub fn is_latency_increasing(&self) -> bool {
        if self.recent.len() < 4 {
            return false;
        }
        let mid = self.recent.len() / 2;
        let first_half: f64 =
            self.recent.iter().take(mid).map(|s| s.avg_latency_us).sum::<f64>() / mid as f64;
        let second_half: f64 = self
            .recent
            .iter()
            .skip(mid)
            .map(|s| s.avg_latency_us)
            .sum::<f64>()
            / (self.recent.len() - mid) as f64;
        second_half > first_half * 1.05
    }

    /// Fraction of all queries that exceeded the complexity threshold.
    pub fn complex_query_ratio(&self) -> f64 {
        if self.stats.total_queries == 0 {
            return 0.0;
        }
        self.stats.total_complex_queries as f64 / self.stats.total_queries as f64
    }

    /// Fraction of all queries that were slow.
    pub fn slow_query_ratio(&self) -> f64 {
        if self.stats.total_queries == 0 {
            return 0.0;
        }
        self.stats.total_slow_queries as f64 / self.stats.total_queries as f64
    }

    // -------------------------------------------------------------------
    // Reset
    // -------------------------------------------------------------------

    /// Reset all state, preserving configuration and registered fields
    /// (which are also reset individually).
    pub fn reset(&mut self) {
        self.tick = 0;
        self.next_query_id = 0;
        self.queries_this_tick = 0;
        self.latency_sum_this_tick = 0.0;
        self.complexity_sum_this_tick = 0.0;
        self.slow_this_tick = 0;
        self.errors_this_tick = 0;
        self.ema_initialized = false;
        self.recent.clear();
        self.query_history.clear();
        self.stats = GraphqlStats::default();

        for field in self.fields.values_mut() {
            field.resolution_count = 0;
            field.error_count = 0;
            field.last_latency_us = 0.0;
            field.ema_latency_us = 0.0;
        }
    }
}

// ===========================================================================
// Tests
// ===========================================================================

#[cfg(test)]
mod tests {
    use super::*;

    fn small_config() -> GraphqlConfig {
        GraphqlConfig {
            max_fields: 8,
            max_query_history: 5,
            complexity_threshold: 50.0,
            slow_query_threshold_us: 100.0,
            ema_decay: 0.5,
            window_size: 5,
        }
    }

    // -------------------------------------------------------------------
    // Construction
    // -------------------------------------------------------------------

    #[test]
    fn test_new_default() {
        let gql = Graphql::new();
        assert_eq!(gql.field_count(), 0);
        assert_eq!(gql.current_tick(), 0);
        assert_eq!(gql.stats().total_queries, 0);
    }

    #[test]
    fn test_with_config() {
        let gql = Graphql::with_config(small_config());
        assert_eq!(gql.config().max_fields, 8);
        assert!((gql.config().complexity_threshold - 50.0).abs() < 1e-9);
    }

    // -------------------------------------------------------------------
    // Schema management
    // -------------------------------------------------------------------

    #[test]
    fn test_register_field() {
        let mut gql = Graphql::with_config(small_config());
        gql.register_field("user", FieldKind::Query, "Get user", 1.0)
            .unwrap();
        assert_eq!(gql.field_count(), 1);
        assert!(gql.field("user").is_some());
        let f = gql.field("user").unwrap();
        assert_eq!(f.kind, FieldKind::Query);
        assert!(!f.deprecated);
    }

    #[test]
    fn test_register_duplicate_fails() {
        let mut gql = Graphql::with_config(small_config());
        gql.register_field("user", FieldKind::Query, "", 1.0)
            .unwrap();
        assert!(gql.register_field("user", FieldKind::Query, "", 1.0).is_err());
    }

    #[test]
    fn test_register_at_capacity() {
        let mut gql = Graphql::with_config(small_config());
        for i in 0..8 {
            gql.register_field(format!("f{}", i), FieldKind::Query, "", 1.0)
                .unwrap();
        }
        assert!(gql
            .register_field("overflow", FieldKind::Query, "", 1.0)
            .is_err());
    }

    #[test]
    fn test_deprecate_field() {
        let mut gql = Graphql::with_config(small_config());
        gql.register_field("old", FieldKind::Query, "", 1.0)
            .unwrap();
        gql.deprecate_field("old").unwrap();
        assert!(gql.field("old").unwrap().deprecated);
    }

    #[test]
    fn test_deprecate_unknown_fails() {
        let mut gql = Graphql::with_config(small_config());
        assert!(gql.deprecate_field("nope").is_err());
    }

    #[test]
    fn test_remove_field() {
        let mut gql = Graphql::with_config(small_config());
        gql.register_field("tmp", FieldKind::Query, "", 1.0)
            .unwrap();
        gql.remove_field("tmp").unwrap();
        assert_eq!(gql.field_count(), 0);
        assert!(gql.field("tmp").is_none());
    }

    #[test]
    fn test_remove_unknown_fails() {
        let mut gql = Graphql::with_config(small_config());
        assert!(gql.remove_field("nope").is_err());
    }

    #[test]
    fn test_field_names_insertion_order() {
        let mut gql = Graphql::with_config(small_config());
        gql.register_field("b", FieldKind::Query, "", 1.0).unwrap();
        gql.register_field("a", FieldKind::Query, "", 1.0).unwrap();
        assert_eq!(gql.field_names(), vec!["b", "a"]);
    }

    #[test]
    fn test_fields_by_kind() {
        let mut gql = Graphql::with_config(small_config());
        gql.register_field("q1", FieldKind::Query, "", 1.0)
            .unwrap();
        gql.register_field("q2", FieldKind::Query, "", 1.0)
            .unwrap();
        gql.register_field("m1", FieldKind::Mutation, "", 1.0)
            .unwrap();
        assert_eq!(gql.fields_by_kind(FieldKind::Query).len(), 2);
        assert_eq!(gql.fields_by_kind(FieldKind::Mutation).len(), 1);
        assert_eq!(gql.fields_by_kind(FieldKind::Subscription).len(), 0);
    }

    #[test]
    fn test_deprecated_fields() {
        let mut gql = Graphql::with_config(small_config());
        gql.register_field("a", FieldKind::Query, "", 1.0).unwrap();
        gql.register_field("b", FieldKind::Query, "", 1.0).unwrap();
        gql.deprecate_field("b").unwrap();
        let dep = gql.deprecated_fields();
        assert_eq!(dep.len(), 1);
        assert_eq!(dep[0].name, "b");
    }

    // -------------------------------------------------------------------
    // Complexity scoring
    // -------------------------------------------------------------------

    #[test]
    fn test_compute_complexity() {
        let mut gql = Graphql::with_config(small_config());
        gql.register_field("a", FieldKind::Query, "", 10.0)
            .unwrap();
        gql.register_field("b", FieldKind::Query, "", 20.0)
            .unwrap();
        let c = gql.compute_complexity(&["a", "b"], 2.0);
        // (10 * 2) + (20 * 2) = 60
        assert!((c - 60.0).abs() < 1e-9);
    }

    #[test]
    fn test_compute_complexity_unknown_field_skipped() {
        let mut gql = Graphql::with_config(small_config());
        gql.register_field("a", FieldKind::Query, "", 10.0)
            .unwrap();
        let c = gql.compute_complexity(&["a", "missing"], 1.0);
        assert!((c - 10.0).abs() < 1e-9);
    }

    #[test]
    fn test_is_complex() {
        let gql = Graphql::with_config(small_config()); // threshold=50
        assert!(!gql.is_complex(49.0));
        assert!(gql.is_complex(51.0));
    }

    // -------------------------------------------------------------------
    // Query execution
    // -------------------------------------------------------------------

    #[test]
    fn test_execute_query_basic() {
        let mut gql = Graphql::with_config(small_config());
        gql.register_field("a", FieldKind::Query, "", 10.0)
            .unwrap();
        gql.register_field("b", FieldKind::Query, "", 5.0).unwrap();

        let record = gql
            .execute_query(&["a", "b"], 1.0, |name| match name {
                "a" => Ok(30.0),
                "b" => Ok(20.0),
                _ => unreachable!(),
            })
            .unwrap();

        assert_eq!(record.id, 1);
        assert_eq!(record.fields.len(), 2);
        assert!((record.latency_us - 50.0).abs() < 1e-9);
        assert!((record.complexity - 15.0).abs() < 1e-9); // 10+5
        assert!(!record.exceeded_complexity); // 15 < 50
        assert!(!record.was_slow); // 50 < 100
        assert_eq!(record.errors, 0);
        assert_eq!(gql.stats().total_queries, 1);
        assert_eq!(gql.stats().total_resolutions, 2);
    }

    #[test]
    fn test_execute_query_with_errors() {
        let mut gql = Graphql::with_config(small_config());
        gql.register_field("a", FieldKind::Query, "", 1.0).unwrap();
        gql.register_field("b", FieldKind::Query, "", 1.0).unwrap();

        let record = gql
            .execute_query(&["a", "b"], 1.0, |name| match name {
                "a" => Ok(10.0),
                "b" => Err(Error::Configuration("fail".into())),
                _ => unreachable!(),
            })
            .unwrap();

        assert_eq!(record.errors, 1);
        assert_eq!(record.fields.len(), 1); // only a succeeded
        assert_eq!(gql.stats().total_field_errors, 1);
        assert_eq!(gql.field("b").unwrap().error_count, 1);
    }

    #[test]
    fn test_execute_query_slow() {
        let mut gql = Graphql::with_config(small_config()); // slow threshold=100
        gql.register_field("a", FieldKind::Query, "", 1.0).unwrap();

        let record = gql
            .execute_query(&["a"], 1.0, |_| Ok(200.0))
            .unwrap();

        assert!(record.was_slow);
        assert_eq!(gql.stats().total_slow_queries, 1);
    }

    #[test]
    fn test_execute_query_complex() {
        let mut gql = Graphql::with_config(small_config()); // complexity_threshold=50
        gql.register_field("heavy", FieldKind::Query, "", 100.0)
            .unwrap();

        let record = gql
            .execute_query(&["heavy"], 1.0, |_| Ok(10.0))
            .unwrap();

        assert!(record.exceeded_complexity);
        assert_eq!(gql.stats().total_complex_queries, 1);
    }

    #[test]
    fn test_execute_query_unknown_field_skipped() {
        let mut gql = Graphql::with_config(small_config());
        gql.register_field("a", FieldKind::Query, "", 1.0).unwrap();

        let record = gql
            .execute_query(&["a", "missing"], 1.0, |_| Ok(10.0))
            .unwrap();

        assert_eq!(record.fields.len(), 1);
        assert_eq!(record.fields[0], "a");
    }

    #[test]
    fn test_field_resolution_stats() {
        let mut gql = Graphql::with_config(small_config()); // ema_decay=0.5
        gql.register_field("a", FieldKind::Query, "", 1.0).unwrap();

        gql.execute_query(&["a"], 1.0, |_| Ok(100.0)).unwrap();
        assert_eq!(gql.field("a").unwrap().resolution_count, 1);
        assert!((gql.field("a").unwrap().ema_latency_us - 100.0).abs() < 1e-9);

        gql.execute_query(&["a"], 1.0, |_| Ok(200.0)).unwrap();
        // EMA = 0.5*200 + 0.5*100 = 150
        assert!((gql.field("a").unwrap().ema_latency_us - 150.0).abs() < 1e-9);
    }

    // -------------------------------------------------------------------
    // Query history
    // -------------------------------------------------------------------

    #[test]
    fn test_last_query() {
        let mut gql = Graphql::with_config(small_config());
        assert!(gql.last_query().is_none());
        gql.register_field("a", FieldKind::Query, "", 1.0).unwrap();
        gql.execute_query(&["a"], 1.0, |_| Ok(10.0)).unwrap();
        assert_eq!(gql.last_query().unwrap().id, 1);
    }

    #[test]
    fn test_query_history_capped() {
        let mut gql = Graphql::with_config(small_config()); // max_query_history=5
        gql.register_field("a", FieldKind::Query, "", 1.0).unwrap();
        for _ in 0..10 {
            gql.execute_query(&["a"], 1.0, |_| Ok(10.0)).unwrap();
        }
        assert!(gql.query_history().len() <= 5);
    }

    // -------------------------------------------------------------------
    // Tick & EMA
    // -------------------------------------------------------------------

    #[test]
    fn test_tick_increments() {
        let mut gql = Graphql::with_config(small_config());
        gql.tick();
        gql.tick();
        assert_eq!(gql.current_tick(), 2);
    }

    #[test]
    fn test_process_alias() {
        let mut gql = Graphql::with_config(small_config());
        gql.process();
        assert_eq!(gql.current_tick(), 1);
    }

    #[test]
    fn test_ema_initialises_on_first_active_tick() {
        let mut gql = Graphql::with_config(small_config());
        gql.register_field("a", FieldKind::Query, "", 1.0).unwrap();

        // Tick with no queries → no EMA init.
        gql.tick();
        assert!((gql.smoothed_query_latency() - 0.0).abs() < 1e-9);

        // Tick with a query → EMA init.
        gql.execute_query(&["a"], 1.0, |_| Ok(60.0)).unwrap();
        gql.tick();
        assert!((gql.smoothed_query_latency() - 60.0).abs() < 1e-9);
        assert!((gql.smoothed_queries_per_tick() - 1.0).abs() < 1e-9);
    }

    #[test]
    fn test_ema_blends_subsequent() {
        let mut gql = Graphql::with_config(small_config()); // decay=0.5
        gql.register_field("a", FieldKind::Query, "", 1.0).unwrap();

        gql.execute_query(&["a"], 1.0, |_| Ok(100.0)).unwrap();
        gql.tick(); // ema_latency = 100

        gql.execute_query(&["a"], 1.0, |_| Ok(200.0)).unwrap();
        gql.tick(); // ema_latency = 0.5*200 + 0.5*100 = 150

        assert!((gql.smoothed_query_latency() - 150.0).abs() < 1e-9);
    }

    // -------------------------------------------------------------------
    // Windowed diagnostics
    // -------------------------------------------------------------------

    #[test]
    fn test_windowed_empty() {
        let gql = Graphql::with_config(small_config());
        assert!(gql.windowed_query_latency().is_none());
        assert!(gql.windowed_query_complexity().is_none());
        assert!(gql.windowed_queries_per_tick().is_none());
        assert!(gql.windowed_slow_queries().is_none());
    }

    #[test]
    fn test_windowed_query_latency() {
        let mut gql = Graphql::with_config(small_config());
        gql.register_field("a", FieldKind::Query, "", 1.0).unwrap();

        gql.execute_query(&["a"], 1.0, |_| Ok(100.0)).unwrap();
        gql.tick();
        gql.execute_query(&["a"], 1.0, |_| Ok(200.0)).unwrap();
        gql.tick();

        let avg = gql.windowed_query_latency().unwrap();
        assert!((avg - 150.0).abs() < 1e-9);
    }

    #[test]
    fn test_windowed_queries_per_tick() {
        let mut gql = Graphql::with_config(small_config());
        gql.register_field("a", FieldKind::Query, "", 1.0).unwrap();

        gql.execute_query(&["a"], 1.0, |_| Ok(10.0)).unwrap();
        gql.execute_query(&["a"], 1.0, |_| Ok(10.0)).unwrap();
        gql.tick(); // 2 queries this tick

        let avg = gql.windowed_queries_per_tick().unwrap();
        assert!((avg - 2.0).abs() < 1e-9);
    }

    #[test]
    fn test_is_latency_increasing() {
        let mut gql = Graphql::with_config(small_config());
        gql.register_field("a", FieldKind::Query, "", 1.0).unwrap();

        assert!(!gql.is_latency_increasing());

        // Feed increasing latencies.
        for lat in &[10.0, 20.0, 100.0, 200.0, 300.0] {
            let l = *lat;
            gql.execute_query(&["a"], 1.0, move |_| Ok(l)).unwrap();
            gql.tick();
        }
        assert!(gql.is_latency_increasing());
    }

    #[test]
    fn test_is_latency_not_increasing() {
        let mut gql = Graphql::with_config(small_config());
        gql.register_field("a", FieldKind::Query, "", 1.0).unwrap();

        for _ in 0..5 {
            gql.execute_query(&["a"], 1.0, |_| Ok(50.0)).unwrap();
            gql.tick();
        }
        assert!(!gql.is_latency_increasing());
    }

    #[test]
    fn test_window_rolls() {
        let mut gql = Graphql::with_config(small_config()); // window_size=5
        gql.register_field("a", FieldKind::Query, "", 1.0).unwrap();
        for _ in 0..20 {
            gql.execute_query(&["a"], 1.0, |_| Ok(10.0)).unwrap();
            gql.tick();
        }
        assert!(gql.recent.len() <= 5);
    }

    // -------------------------------------------------------------------
    // Ratios
    // -------------------------------------------------------------------

    #[test]
    fn test_complex_query_ratio() {
        let mut gql = Graphql::with_config(small_config()); // threshold=50
        gql.register_field("heavy", FieldKind::Query, "", 100.0)
            .unwrap();
        gql.register_field("light", FieldKind::Query, "", 1.0)
            .unwrap();

        gql.execute_query(&["heavy"], 1.0, |_| Ok(10.0)).unwrap(); // complex
        gql.execute_query(&["light"], 1.0, |_| Ok(10.0)).unwrap(); // not complex

        assert!((gql.complex_query_ratio() - 0.5).abs() < 1e-9);
    }

    #[test]
    fn test_slow_query_ratio() {
        let mut gql = Graphql::with_config(small_config()); // slow threshold=100
        gql.register_field("a", FieldKind::Query, "", 1.0).unwrap();

        gql.execute_query(&["a"], 1.0, |_| Ok(200.0)).unwrap(); // slow
        gql.execute_query(&["a"], 1.0, |_| Ok(10.0)).unwrap(); // not slow

        assert!((gql.slow_query_ratio() - 0.5).abs() < 1e-9);
    }

    #[test]
    fn test_ratios_empty() {
        let gql = Graphql::with_config(small_config());
        assert!((gql.complex_query_ratio() - 0.0).abs() < 1e-9);
        assert!((gql.slow_query_ratio() - 0.0).abs() < 1e-9);
    }

    // -------------------------------------------------------------------
    // Reset
    // -------------------------------------------------------------------

    #[test]
    fn test_reset() {
        let mut gql = Graphql::with_config(small_config());
        gql.register_field("a", FieldKind::Query, "", 1.0).unwrap();
        gql.execute_query(&["a"], 1.0, |_| Ok(50.0)).unwrap();
        gql.tick();

        gql.reset();

        assert_eq!(gql.current_tick(), 0);
        assert_eq!(gql.stats().total_queries, 0);
        assert_eq!(gql.stats().total_resolutions, 0);
        assert!(gql.last_query().is_none());
        assert!(gql.windowed_query_latency().is_none());
        // Fields preserved but counters reset.
        assert_eq!(gql.field_count(), 1);
        assert_eq!(gql.field("a").unwrap().resolution_count, 0);
        assert!((gql.field("a").unwrap().ema_latency_us - 0.0).abs() < 1e-9);
    }

    // -------------------------------------------------------------------
    // Full lifecycle
    // -------------------------------------------------------------------

    #[test]
    fn test_full_lifecycle() {
        let mut gql = Graphql::with_config(small_config());

        // Register schema.
        gql.register_field("user", FieldKind::Query, "Get user by ID", 5.0)
            .unwrap();
        gql.register_field("users", FieldKind::Query, "List users", 20.0)
            .unwrap();
        gql.register_field("createUser", FieldKind::Mutation, "Create a user", 10.0)
            .unwrap();
        gql.register_field("userUpdated", FieldKind::Subscription, "User change stream", 2.0)
            .unwrap();

        assert_eq!(gql.field_count(), 4);
        assert_eq!(gql.fields_by_kind(FieldKind::Query).len(), 2);
        assert_eq!(gql.fields_by_kind(FieldKind::Mutation).len(), 1);
        assert_eq!(gql.fields_by_kind(FieldKind::Subscription).len(), 1);

        // Execute some queries.
        gql.execute_query(&["user"], 1.0, |_| Ok(30.0)).unwrap();
        gql.execute_query(&["users"], 3.0, |_| Ok(90.0)).unwrap(); // complexity=60 > 50
        gql.tick();

        gql.execute_query(&["createUser"], 1.0, |_| Ok(50.0))
            .unwrap();
        gql.tick();

        // Deprecate a field.
        gql.deprecate_field("userUpdated").unwrap();
        assert_eq!(gql.deprecated_fields().len(), 1);

        // Stats.
        assert_eq!(gql.stats().total_queries, 3);
        assert_eq!(gql.stats().total_resolutions, 3);
        assert_eq!(gql.stats().total_complex_queries, 1);
        assert!(gql.smoothed_query_latency() > 0.0);
        assert!(gql.windowed_query_latency().is_some());
    }

    // -------------------------------------------------------------------
    // Display impls
    // -------------------------------------------------------------------

    #[test]
    fn test_field_kind_display() {
        assert_eq!(format!("{}", FieldKind::Query), "Query");
        assert_eq!(format!("{}", FieldKind::Mutation), "Mutation");
        assert_eq!(format!("{}", FieldKind::Subscription), "Subscription");
    }
}
