//! Execution Metrics for Prometheus
//!
//! This module provides Prometheus-compatible metrics for tracking the health
//! and performance of the execution subsystem including:
//! - Retry operations and circuit breaker states
//! - Arbitrage opportunity detection and monitoring
//! - Best execution analysis and recommendations
//! - Signal flow coordination statistics
//!
//! # Metrics Exported
//!
//! ## Retry Metrics
//! | Metric | Type | Description |
//! |--------|------|-------------|
//! | execution_retry_total | Counter | Total retry operations |
//! | execution_retry_success | Counter | Successful operations |
//! | execution_retry_failure | Counter | Failed operations |
//! | execution_retry_attempts | Histogram | Retry attempts per operation |
//! | execution_circuit_breaker_state | Gauge | Circuit breaker state (0=closed, 1=open, 2=half-open) |
//! | execution_circuit_breaker_failures | Counter | Circuit breaker failure count |
//!
//! ## Arbitrage Metrics
//! | Metric | Type | Description |
//! |--------|------|-------------|
//! | arbitrage_opportunities_total | Counter | Total opportunities detected |
//! | arbitrage_opportunities_actionable | Counter | Actionable opportunities |
//! | arbitrage_spread_bps | Gauge | Current spread in basis points |
//! | arbitrage_best_spread_bps | Gauge | Best spread seen |
//! | arbitrage_price_updates | Counter | Price updates received |
//! | arbitrage_kraken_deviation_pct | Gauge | Kraken price deviation percentage |
//!
//! ## Best Execution Metrics
//! | Metric | Type | Description |
//! |--------|------|-------------|
//! | best_execution_analyses_total | Counter | Total analyses performed |
//! | best_execution_recommendation | Gauge | Current recommendation (0-4) |
//! | best_execution_score | Gauge | Current execution score |
//! | best_execution_slippage_bps | Gauge | Estimated slippage in bps |
//! | best_execution_quality_score | Gauge | Execution quality score |
//! | best_execution_executions_total | Counter | Total executions recorded |
//!
//! ## Signal Flow Metrics
//! | Metric | Type | Description |
//! |--------|------|-------------|
//! | signal_flow_signals_received | Counter | Total signals received |
//! | signal_flow_signals_executed | Counter | Signals executed |
//! | signal_flow_signals_rejected | Counter | Signals rejected |
//! | signal_flow_orders_submitted | Counter | Orders submitted |
//! | signal_flow_orders_filled | Counter | Orders filled |
//! | signal_flow_fills_received | Counter | Fills received |

use parking_lot::RwLock;
use rust_decimal::Decimal;
use std::collections::HashMap;
use std::sync::Arc;
use std::sync::LazyLock;
use std::sync::atomic::{AtomicU64, Ordering};
use std::time::Instant;

// ============================================================================
// Retry Metrics
// ============================================================================

/// Metrics for retry operations and circuit breakers
#[derive(Debug)]
#[allow(dead_code)]
pub struct RetryMetrics {
    /// Operation name for labeling
    operation: String,

    /// Total operations attempted
    total_operations: AtomicU64,

    /// Successful operations on first try
    first_try_successes: AtomicU64,

    /// Successful operations after retries
    retry_successes: AtomicU64,

    /// Total failed operations
    total_failures: AtomicU64,

    /// Total retry attempts
    total_retries: AtomicU64,

    /// Rate limit errors
    rate_limit_errors: AtomicU64,

    /// Network errors
    network_errors: AtomicU64,

    /// Timeout errors
    timeout_errors: AtomicU64,

    /// Last operation duration in ms
    last_duration_ms: AtomicU64,

    /// Circuit breaker state (0=closed, 1=open, 2=half-open)
    circuit_breaker_state: AtomicU64,

    /// Circuit breaker failure count
    circuit_breaker_failures: AtomicU64,

    /// Operation start time for uptime tracking
    start_time: Instant,
}

impl RetryMetrics {
    /// Create new retry metrics
    pub fn new(operation: &str) -> Self {
        Self {
            operation: operation.to_string(),
            total_operations: AtomicU64::new(0),
            first_try_successes: AtomicU64::new(0),
            retry_successes: AtomicU64::new(0),
            total_failures: AtomicU64::new(0),
            total_retries: AtomicU64::new(0),
            rate_limit_errors: AtomicU64::new(0),
            network_errors: AtomicU64::new(0),
            timeout_errors: AtomicU64::new(0),
            last_duration_ms: AtomicU64::new(0),
            circuit_breaker_state: AtomicU64::new(0),
            circuit_breaker_failures: AtomicU64::new(0),
            start_time: Instant::now(),
        }
    }

    /// Record a successful operation
    pub fn record_success(&self, attempts: u32, duration_ms: u64) {
        self.total_operations.fetch_add(1, Ordering::Relaxed);
        self.last_duration_ms.store(duration_ms, Ordering::Relaxed);

        if attempts == 1 {
            self.first_try_successes.fetch_add(1, Ordering::Relaxed);
        } else {
            self.retry_successes.fetch_add(1, Ordering::Relaxed);
            self.total_retries
                .fetch_add((attempts - 1) as u64, Ordering::Relaxed);
        }
    }

    /// Record a failed operation
    pub fn record_failure(&self, attempts: u32, duration_ms: u64) {
        self.total_operations.fetch_add(1, Ordering::Relaxed);
        self.total_failures.fetch_add(1, Ordering::Relaxed);
        self.total_retries
            .fetch_add(attempts as u64, Ordering::Relaxed);
        self.last_duration_ms.store(duration_ms, Ordering::Relaxed);
    }

    /// Record a rate limit error
    pub fn record_rate_limit(&self) {
        self.rate_limit_errors.fetch_add(1, Ordering::Relaxed);
    }

    /// Record a network error
    pub fn record_network_error(&self) {
        self.network_errors.fetch_add(1, Ordering::Relaxed);
    }

    /// Record a timeout error
    pub fn record_timeout(&self) {
        self.timeout_errors.fetch_add(1, Ordering::Relaxed);
    }

    /// Update circuit breaker state
    pub fn set_circuit_breaker_state(&self, state: CircuitBreakerState) {
        self.circuit_breaker_state
            .store(state as u64, Ordering::Relaxed);
    }

    /// Record circuit breaker failure
    pub fn record_circuit_breaker_failure(&self) {
        self.circuit_breaker_failures
            .fetch_add(1, Ordering::Relaxed);
    }

    /// Get success rate
    pub fn success_rate(&self) -> f64 {
        let total = self.total_operations.load(Ordering::Relaxed);
        if total == 0 {
            return 1.0;
        }
        let successes = self.first_try_successes.load(Ordering::Relaxed)
            + self.retry_successes.load(Ordering::Relaxed);
        successes as f64 / total as f64
    }

    /// Get first try success rate
    pub fn first_try_rate(&self) -> f64 {
        let total = self.total_operations.load(Ordering::Relaxed);
        if total == 0 {
            return 1.0;
        }
        self.first_try_successes.load(Ordering::Relaxed) as f64 / total as f64
    }

    /// Generate Prometheus metrics output
    pub fn to_prometheus(&self) -> String {
        let op = &self.operation;
        let mut output = String::new();

        // Total operations
        output.push_str(&format!(
            "# HELP execution_retry_total Total retry operations\n\
             # TYPE execution_retry_total counter\n\
             execution_retry_total{{operation=\"{}\"}} {}\n",
            op,
            self.total_operations.load(Ordering::Relaxed)
        ));

        // Successes
        output.push_str(&format!(
            "# HELP execution_retry_success Successful operations\n\
             # TYPE execution_retry_success counter\n\
             execution_retry_success{{operation=\"{}\",type=\"first_try\"}} {}\n\
             execution_retry_success{{operation=\"{}\",type=\"after_retry\"}} {}\n",
            op,
            self.first_try_successes.load(Ordering::Relaxed),
            op,
            self.retry_successes.load(Ordering::Relaxed)
        ));

        // Failures
        output.push_str(&format!(
            "# HELP execution_retry_failure Failed operations\n\
             # TYPE execution_retry_failure counter\n\
             execution_retry_failure{{operation=\"{}\"}} {}\n",
            op,
            self.total_failures.load(Ordering::Relaxed)
        ));

        // Retries
        output.push_str(&format!(
            "# HELP execution_retry_attempts Total retry attempts\n\
             # TYPE execution_retry_attempts counter\n\
             execution_retry_attempts{{operation=\"{}\"}} {}\n",
            op,
            self.total_retries.load(Ordering::Relaxed)
        ));

        // Error types
        output.push_str(&format!(
            "# HELP execution_retry_errors_by_type Errors by type\n\
             # TYPE execution_retry_errors_by_type counter\n\
             execution_retry_errors_by_type{{operation=\"{}\",type=\"rate_limit\"}} {}\n\
             execution_retry_errors_by_type{{operation=\"{}\",type=\"network\"}} {}\n\
             execution_retry_errors_by_type{{operation=\"{}\",type=\"timeout\"}} {}\n",
            op,
            self.rate_limit_errors.load(Ordering::Relaxed),
            op,
            self.network_errors.load(Ordering::Relaxed),
            op,
            self.timeout_errors.load(Ordering::Relaxed)
        ));

        // Duration
        output.push_str(&format!(
            "# HELP execution_retry_duration_ms Last operation duration in milliseconds\n\
             # TYPE execution_retry_duration_ms gauge\n\
             execution_retry_duration_ms{{operation=\"{}\"}} {}\n",
            op,
            self.last_duration_ms.load(Ordering::Relaxed)
        ));

        // Circuit breaker
        output.push_str(&format!(
            "# HELP execution_circuit_breaker_state Circuit breaker state (0=closed, 1=open, 2=half-open)\n\
             # TYPE execution_circuit_breaker_state gauge\n\
             execution_circuit_breaker_state{{operation=\"{}\"}} {}\n",
            op,
            self.circuit_breaker_state.load(Ordering::Relaxed)
        ));

        output.push_str(&format!(
            "# HELP execution_circuit_breaker_failures Circuit breaker failure count\n\
             # TYPE execution_circuit_breaker_failures counter\n\
             execution_circuit_breaker_failures{{operation=\"{}\"}} {}\n",
            op,
            self.circuit_breaker_failures.load(Ordering::Relaxed)
        ));

        // Success rate
        output.push_str(&format!(
            "# HELP execution_retry_success_rate Success rate (0-1)\n\
             # TYPE execution_retry_success_rate gauge\n\
             execution_retry_success_rate{{operation=\"{}\"}} {:.4}\n",
            op,
            self.success_rate()
        ));

        output
    }

    /// Reset all metrics
    pub fn reset(&self) {
        self.total_operations.store(0, Ordering::Relaxed);
        self.first_try_successes.store(0, Ordering::Relaxed);
        self.retry_successes.store(0, Ordering::Relaxed);
        self.total_failures.store(0, Ordering::Relaxed);
        self.total_retries.store(0, Ordering::Relaxed);
        self.rate_limit_errors.store(0, Ordering::Relaxed);
        self.network_errors.store(0, Ordering::Relaxed);
        self.timeout_errors.store(0, Ordering::Relaxed);
        self.circuit_breaker_failures.store(0, Ordering::Relaxed);
    }
}

/// Circuit breaker state enumeration
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
#[repr(u64)]
pub enum CircuitBreakerState {
    Closed = 0,
    Open = 1,
    HalfOpen = 2,
}

// ============================================================================
// Arbitrage Metrics
// ============================================================================

/// Metrics for arbitrage monitoring
#[derive(Debug)]
pub struct ArbitrageMetrics {
    /// Total opportunities detected
    opportunities_total: AtomicU64,

    /// Actionable opportunities (above threshold)
    opportunities_actionable: AtomicU64,

    /// Opportunities by symbol
    opportunities_by_symbol: RwLock<HashMap<String, u64>>,

    /// Current spread by symbol (stored as bps * 100)
    current_spread_bps: RwLock<HashMap<String, i64>>,

    /// Best spread seen by symbol (stored as bps * 100)
    best_spread_bps: RwLock<HashMap<String, i64>>,

    /// Price updates received
    price_updates: AtomicU64,

    /// Price updates by exchange
    price_updates_by_exchange: RwLock<HashMap<String, u64>>,

    /// Kraken deviation percentage by symbol (stored as pct * 10000)
    kraken_deviation_pct: RwLock<HashMap<String, i64>>,

    /// Alerts triggered
    alerts_triggered: AtomicU64,

    /// Start time
    start_time: Instant,

    /// Last update timestamp (unix millis)
    last_update_ms: AtomicU64,
}

impl ArbitrageMetrics {
    /// Create new arbitrage metrics
    pub fn new() -> Self {
        Self {
            opportunities_total: AtomicU64::new(0),
            opportunities_actionable: AtomicU64::new(0),
            opportunities_by_symbol: RwLock::new(HashMap::new()),
            current_spread_bps: RwLock::new(HashMap::new()),
            best_spread_bps: RwLock::new(HashMap::new()),
            price_updates: AtomicU64::new(0),
            price_updates_by_exchange: RwLock::new(HashMap::new()),
            kraken_deviation_pct: RwLock::new(HashMap::new()),
            alerts_triggered: AtomicU64::new(0),
            start_time: Instant::now(),
            last_update_ms: AtomicU64::new(0),
        }
    }

    /// Record an opportunity detected
    pub fn record_opportunity(&self, symbol: &str, spread_bps: Decimal, actionable: bool) {
        self.opportunities_total.fetch_add(1, Ordering::Relaxed);

        if actionable {
            self.opportunities_actionable
                .fetch_add(1, Ordering::Relaxed);
        }

        // Update by symbol
        {
            let mut by_symbol = self.opportunities_by_symbol.write();
            *by_symbol.entry(symbol.to_string()).or_insert(0) += 1;
        }

        // Update current spread
        let spread_x100 = decimal_to_i64_x100(spread_bps);
        {
            let mut spreads = self.current_spread_bps.write();
            spreads.insert(symbol.to_string(), spread_x100);
        }

        // Update best spread
        {
            let mut best = self.best_spread_bps.write();
            let entry = best.entry(symbol.to_string()).or_insert(0);
            if spread_x100 > *entry {
                *entry = spread_x100;
            }
        }

        self.last_update_ms
            .store(current_timestamp_ms(), Ordering::Relaxed);
    }

    /// Record a price update
    pub fn record_price_update(&self, exchange: &str) {
        self.price_updates.fetch_add(1, Ordering::Relaxed);

        {
            let mut by_exchange = self.price_updates_by_exchange.write();
            *by_exchange.entry(exchange.to_string()).or_insert(0) += 1;
        }

        self.last_update_ms
            .store(current_timestamp_ms(), Ordering::Relaxed);
    }

    /// Update Kraken deviation
    pub fn update_kraken_deviation(&self, symbol: &str, deviation_pct: Decimal) {
        let deviation_x10000 = decimal_to_i64_x10000(deviation_pct);
        let mut deviations = self.kraken_deviation_pct.write();
        deviations.insert(symbol.to_string(), deviation_x10000);
    }

    /// Record an alert triggered
    pub fn record_alert(&self) {
        self.alerts_triggered.fetch_add(1, Ordering::Relaxed);
    }

    /// Get current spread for a symbol
    pub fn get_spread_bps(&self, symbol: &str) -> Option<Decimal> {
        self.current_spread_bps
            .read()
            .get(symbol)
            .map(|&v| i64_x100_to_decimal(v))
    }

    /// Get best spread for a symbol
    pub fn get_best_spread_bps(&self, symbol: &str) -> Option<Decimal> {
        self.best_spread_bps
            .read()
            .get(symbol)
            .map(|&v| i64_x100_to_decimal(v))
    }

    /// Generate Prometheus metrics output
    pub fn to_prometheus(&self) -> String {
        let mut output = String::new();

        // Total opportunities
        output.push_str(&format!(
            "# HELP arbitrage_opportunities_total Total arbitrage opportunities detected\n\
             # TYPE arbitrage_opportunities_total counter\n\
             arbitrage_opportunities_total {}\n",
            self.opportunities_total.load(Ordering::Relaxed)
        ));

        // Actionable opportunities
        output.push_str(&format!(
            "# HELP arbitrage_opportunities_actionable Actionable arbitrage opportunities\n\
             # TYPE arbitrage_opportunities_actionable counter\n\
             arbitrage_opportunities_actionable {}\n",
            self.opportunities_actionable.load(Ordering::Relaxed)
        ));

        // Opportunities by symbol
        output.push_str(
            "# HELP arbitrage_opportunities_by_symbol Opportunities by trading symbol\n\
             # TYPE arbitrage_opportunities_by_symbol counter\n",
        );
        for (symbol, count) in self.opportunities_by_symbol.read().iter() {
            output.push_str(&format!(
                "arbitrage_opportunities_by_symbol{{symbol=\"{}\"}} {}\n",
                symbol, count
            ));
        }

        // Current spread
        output.push_str(
            "# HELP arbitrage_spread_bps Current spread in basis points\n\
             # TYPE arbitrage_spread_bps gauge\n",
        );
        for (symbol, spread_x100) in self.current_spread_bps.read().iter() {
            output.push_str(&format!(
                "arbitrage_spread_bps{{symbol=\"{}\"}} {:.2}\n",
                symbol,
                *spread_x100 as f64 / 100.0
            ));
        }

        // Best spread
        output.push_str(
            "# HELP arbitrage_best_spread_bps Best spread seen in basis points\n\
             # TYPE arbitrage_best_spread_bps gauge\n",
        );
        for (symbol, spread_x100) in self.best_spread_bps.read().iter() {
            output.push_str(&format!(
                "arbitrage_best_spread_bps{{symbol=\"{}\"}} {:.2}\n",
                symbol,
                *spread_x100 as f64 / 100.0
            ));
        }

        // Price updates
        output.push_str(&format!(
            "# HELP arbitrage_price_updates Total price updates received\n\
             # TYPE arbitrage_price_updates counter\n\
             arbitrage_price_updates {}\n",
            self.price_updates.load(Ordering::Relaxed)
        ));

        // Price updates by exchange
        output.push_str(
            "# HELP arbitrage_price_updates_by_exchange Price updates by exchange\n\
             # TYPE arbitrage_price_updates_by_exchange counter\n",
        );
        for (exchange, count) in self.price_updates_by_exchange.read().iter() {
            output.push_str(&format!(
                "arbitrage_price_updates_by_exchange{{exchange=\"{}\"}} {}\n",
                exchange, count
            ));
        }

        // Kraken deviation
        output.push_str(
            "# HELP arbitrage_kraken_deviation_pct Kraken price deviation percentage\n\
             # TYPE arbitrage_kraken_deviation_pct gauge\n",
        );
        for (symbol, deviation_x10000) in self.kraken_deviation_pct.read().iter() {
            output.push_str(&format!(
                "arbitrage_kraken_deviation_pct{{symbol=\"{}\"}} {:.4}\n",
                symbol,
                *deviation_x10000 as f64 / 10000.0
            ));
        }

        // Alerts
        output.push_str(&format!(
            "# HELP arbitrage_alerts_triggered Spread alerts triggered\n\
             # TYPE arbitrage_alerts_triggered counter\n\
             arbitrage_alerts_triggered {}\n",
            self.alerts_triggered.load(Ordering::Relaxed)
        ));

        // Uptime
        output.push_str(&format!(
            "# HELP arbitrage_uptime_seconds Time since metrics started\n\
             # TYPE arbitrage_uptime_seconds gauge\n\
             arbitrage_uptime_seconds {}\n",
            self.start_time.elapsed().as_secs()
        ));

        output
    }

    /// Reset all metrics
    pub fn reset(&self) {
        self.opportunities_total.store(0, Ordering::Relaxed);
        self.opportunities_actionable.store(0, Ordering::Relaxed);
        self.opportunities_by_symbol.write().clear();
        self.current_spread_bps.write().clear();
        self.best_spread_bps.write().clear();
        self.price_updates.store(0, Ordering::Relaxed);
        self.price_updates_by_exchange.write().clear();
        self.kraken_deviation_pct.write().clear();
        self.alerts_triggered.store(0, Ordering::Relaxed);
    }
}

impl Default for ArbitrageMetrics {
    fn default() -> Self {
        Self::new()
    }
}

// ============================================================================
// Best Execution Metrics
// ============================================================================

/// Recommendation type for metrics (matches ExecutionRecommendation)
#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash)]
#[repr(u64)]
pub enum RecommendationType {
    ExecuteNow = 0,
    Wait = 1,
    Acceptable = 2,
    UseLimitOrder = 3,
    Avoid = 4,
}

impl RecommendationType {
    #[allow(clippy::should_implement_trait)]
    pub fn from_str(s: &str) -> Self {
        match s.to_lowercase().as_str() {
            "executenow" | "execute_now" | "execute now" => Self::ExecuteNow,
            "wait" => Self::Wait,
            "acceptable" => Self::Acceptable,
            "uselimitorder" | "use_limit_order" | "use limit order" => Self::UseLimitOrder,
            "avoid" => Self::Avoid,
            _ => Self::Acceptable,
        }
    }

    pub fn as_str(&self) -> &'static str {
        match self {
            Self::ExecuteNow => "execute_now",
            Self::Wait => "wait",
            Self::Acceptable => "acceptable",
            Self::UseLimitOrder => "use_limit_order",
            Self::Avoid => "avoid",
        }
    }
}

/// Metrics for best execution analysis
#[derive(Debug)]
pub struct BestExecutionMetrics {
    /// Total analyses performed
    analyses_total: AtomicU64,

    /// Analyses by symbol
    analyses_by_symbol: RwLock<HashMap<String, u64>>,

    /// Current recommendation by symbol
    current_recommendation: RwLock<HashMap<String, RecommendationType>>,

    /// Recommendation counts
    recommendation_counts: RwLock<HashMap<RecommendationType, u64>>,

    /// Current execution score by symbol (stored as score * 10000)
    current_score: RwLock<HashMap<String, u64>>,

    /// Current slippage estimate by symbol (stored as bps * 100)
    current_slippage_bps: RwLock<HashMap<String, i64>>,

    /// Total executions recorded
    executions_total: AtomicU64,

    /// Execution quality scores by symbol (stored as score * 10000)
    quality_scores: RwLock<HashMap<String, u64>>,

    /// Sum of quality scores for averaging (stored as score * 10000)
    quality_score_sum: RwLock<HashMap<String, u64>>,

    /// Count for quality score averaging
    quality_score_count: RwLock<HashMap<String, u64>>,

    /// Slippage exceeded count
    slippage_exceeded_count: AtomicU64,

    /// Start time
    start_time: Instant,

    /// Last analysis timestamp
    last_analysis_ms: AtomicU64,
}

impl BestExecutionMetrics {
    /// Create new best execution metrics
    pub fn new() -> Self {
        Self {
            analyses_total: AtomicU64::new(0),
            analyses_by_symbol: RwLock::new(HashMap::new()),
            current_recommendation: RwLock::new(HashMap::new()),
            recommendation_counts: RwLock::new(HashMap::new()),
            current_score: RwLock::new(HashMap::new()),
            current_slippage_bps: RwLock::new(HashMap::new()),
            executions_total: AtomicU64::new(0),
            quality_scores: RwLock::new(HashMap::new()),
            quality_score_sum: RwLock::new(HashMap::new()),
            quality_score_count: RwLock::new(HashMap::new()),
            slippage_exceeded_count: AtomicU64::new(0),
            start_time: Instant::now(),
            last_analysis_ms: AtomicU64::new(0),
        }
    }

    /// Record an analysis
    pub fn record_analysis(
        &self,
        symbol: &str,
        recommendation: RecommendationType,
        score: Decimal,
        slippage_bps: Decimal,
    ) {
        self.analyses_total.fetch_add(1, Ordering::Relaxed);

        // By symbol
        {
            let mut by_symbol = self.analyses_by_symbol.write();
            *by_symbol.entry(symbol.to_string()).or_insert(0) += 1;
        }

        // Current recommendation
        {
            let mut recs = self.current_recommendation.write();
            recs.insert(symbol.to_string(), recommendation);
        }

        // Recommendation counts
        {
            let mut counts = self.recommendation_counts.write();
            *counts.entry(recommendation).or_insert(0) += 1;
        }

        // Current score
        {
            let score_x10000 = decimal_to_u64_x10000(score);
            let mut scores = self.current_score.write();
            scores.insert(symbol.to_string(), score_x10000);
        }

        // Current slippage
        {
            let slippage_x100 = decimal_to_i64_x100(slippage_bps);
            let mut slippages = self.current_slippage_bps.write();
            slippages.insert(symbol.to_string(), slippage_x100);
        }

        self.last_analysis_ms
            .store(current_timestamp_ms(), Ordering::Relaxed);
    }

    /// Record an execution
    pub fn record_execution(&self, symbol: &str, quality_score: Decimal, slippage_exceeded: bool) {
        self.executions_total.fetch_add(1, Ordering::Relaxed);

        if slippage_exceeded {
            self.slippage_exceeded_count.fetch_add(1, Ordering::Relaxed);
        }

        let score_x10000 = decimal_to_u64_x10000(quality_score);

        // Update quality score
        {
            let mut scores = self.quality_scores.write();
            scores.insert(symbol.to_string(), score_x10000);
        }

        // Update sum and count for averaging
        {
            let mut sum = self.quality_score_sum.write();
            *sum.entry(symbol.to_string()).or_insert(0) += score_x10000;
        }
        {
            let mut count = self.quality_score_count.write();
            *count.entry(symbol.to_string()).or_insert(0) += 1;
        }
    }

    /// Get average quality score for a symbol
    pub fn get_avg_quality_score(&self, symbol: &str) -> Option<Decimal> {
        let sum = self.quality_score_sum.read().get(symbol).copied()?;
        let count = self.quality_score_count.read().get(symbol).copied()?;
        if count == 0 {
            return None;
        }
        Some(u64_x10000_to_decimal(sum / count))
    }

    /// Generate Prometheus metrics output
    pub fn to_prometheus(&self) -> String {
        let mut output = String::new();

        // Total analyses
        output.push_str(&format!(
            "# HELP best_execution_analyses_total Total execution analyses performed\n\
             # TYPE best_execution_analyses_total counter\n\
             best_execution_analyses_total {}\n",
            self.analyses_total.load(Ordering::Relaxed)
        ));

        // Analyses by symbol
        output.push_str(
            "# HELP best_execution_analyses_by_symbol Analyses by trading symbol\n\
             # TYPE best_execution_analyses_by_symbol counter\n",
        );
        for (symbol, count) in self.analyses_by_symbol.read().iter() {
            output.push_str(&format!(
                "best_execution_analyses_by_symbol{{symbol=\"{}\"}} {}\n",
                symbol, count
            ));
        }

        // Current recommendation
        output.push_str(
            "# HELP best_execution_recommendation Current recommendation (0=execute_now, 1=wait, 2=acceptable, 3=use_limit, 4=avoid)\n\
             # TYPE best_execution_recommendation gauge\n",
        );
        for (symbol, rec) in self.current_recommendation.read().iter() {
            output.push_str(&format!(
                "best_execution_recommendation{{symbol=\"{}\",recommendation=\"{}\"}} {}\n",
                symbol,
                rec.as_str(),
                *rec as u64
            ));
        }

        // Recommendation counts
        output.push_str(
            "# HELP best_execution_recommendation_count Count of each recommendation type\n\
             # TYPE best_execution_recommendation_count counter\n",
        );
        for (rec, count) in self.recommendation_counts.read().iter() {
            output.push_str(&format!(
                "best_execution_recommendation_count{{recommendation=\"{}\"}} {}\n",
                rec.as_str(),
                count
            ));
        }

        // Current score
        output.push_str(
            "# HELP best_execution_score Current execution score (0-1)\n\
             # TYPE best_execution_score gauge\n",
        );
        for (symbol, score_x10000) in self.current_score.read().iter() {
            output.push_str(&format!(
                "best_execution_score{{symbol=\"{}\"}} {:.4}\n",
                symbol,
                *score_x10000 as f64 / 10000.0
            ));
        }

        // Current slippage
        output.push_str(
            "# HELP best_execution_slippage_bps Estimated slippage in basis points\n\
             # TYPE best_execution_slippage_bps gauge\n",
        );
        for (symbol, slippage_x100) in self.current_slippage_bps.read().iter() {
            output.push_str(&format!(
                "best_execution_slippage_bps{{symbol=\"{}\"}} {:.2}\n",
                symbol,
                *slippage_x100 as f64 / 100.0
            ));
        }

        // Total executions
        output.push_str(&format!(
            "# HELP best_execution_executions_total Total executions recorded\n\
             # TYPE best_execution_executions_total counter\n\
             best_execution_executions_total {}\n",
            self.executions_total.load(Ordering::Relaxed)
        ));

        // Quality scores
        output.push_str(
            "# HELP best_execution_quality_score Last execution quality score (0-1)\n\
             # TYPE best_execution_quality_score gauge\n",
        );
        for (symbol, score_x10000) in self.quality_scores.read().iter() {
            output.push_str(&format!(
                "best_execution_quality_score{{symbol=\"{}\"}} {:.4}\n",
                symbol,
                *score_x10000 as f64 / 10000.0
            ));
        }

        // Average quality scores
        output.push_str(
            "# HELP best_execution_avg_quality_score Average execution quality score (0-1)\n\
             # TYPE best_execution_avg_quality_score gauge\n",
        );
        let sum_map = self.quality_score_sum.read();
        let count_map = self.quality_score_count.read();
        for (symbol, &sum) in sum_map.iter() {
            if let Some(&count) = count_map.get(symbol) {
                if count > 0 {
                    output.push_str(&format!(
                        "best_execution_avg_quality_score{{symbol=\"{}\"}} {:.4}\n",
                        symbol,
                        (sum as f64 / count as f64) / 10000.0
                    ));
                }
            }
        }

        // Slippage exceeded
        output.push_str(&format!(
            "# HELP best_execution_slippage_exceeded_total Executions where slippage exceeded estimate\n\
             # TYPE best_execution_slippage_exceeded_total counter\n\
             best_execution_slippage_exceeded_total {}\n",
            self.slippage_exceeded_count.load(Ordering::Relaxed)
        ));

        // Uptime
        output.push_str(&format!(
            "# HELP best_execution_uptime_seconds Time since metrics started\n\
             # TYPE best_execution_uptime_seconds gauge\n\
             best_execution_uptime_seconds {}\n",
            self.start_time.elapsed().as_secs()
        ));

        output
    }

    /// Reset all metrics
    pub fn reset(&self) {
        self.analyses_total.store(0, Ordering::Relaxed);
        self.analyses_by_symbol.write().clear();
        self.current_recommendation.write().clear();
        self.recommendation_counts.write().clear();
        self.current_score.write().clear();
        self.current_slippage_bps.write().clear();
        self.executions_total.store(0, Ordering::Relaxed);
        self.quality_scores.write().clear();
        self.quality_score_sum.write().clear();
        self.quality_score_count.write().clear();
        self.slippage_exceeded_count.store(0, Ordering::Relaxed);
    }
}

impl Default for BestExecutionMetrics {
    fn default() -> Self {
        Self::new()
    }
}

// ============================================================================
// Signal Flow Metrics
// ============================================================================

/// Metrics for signal flow coordination
#[derive(Debug)]
pub struct SignalFlowMetrics {
    /// Signals received
    signals_received: AtomicU64,

    /// Signals executed
    signals_executed: AtomicU64,

    /// Signals rejected
    signals_rejected: AtomicU64,

    /// Signals by type
    signals_by_type: RwLock<HashMap<String, u64>>,

    /// Orders submitted
    orders_submitted: AtomicU64,

    /// Orders filled
    orders_filled: AtomicU64,

    /// Orders cancelled
    orders_cancelled: AtomicU64,

    /// Orders rejected
    orders_rejected: AtomicU64,

    /// Fills received
    fills_received: AtomicU64,

    /// Position updates
    position_updates: AtomicU64,

    /// Balance updates
    balance_updates: AtomicU64,

    /// Total volume (stored as value * 100)
    total_volume: AtomicU64,

    /// Start time
    start_time: Instant,

    /// Last signal timestamp
    last_signal_ms: AtomicU64,
}

impl SignalFlowMetrics {
    /// Create new signal flow metrics
    pub fn new() -> Self {
        Self {
            signals_received: AtomicU64::new(0),
            signals_executed: AtomicU64::new(0),
            signals_rejected: AtomicU64::new(0),
            signals_by_type: RwLock::new(HashMap::new()),
            orders_submitted: AtomicU64::new(0),
            orders_filled: AtomicU64::new(0),
            orders_cancelled: AtomicU64::new(0),
            orders_rejected: AtomicU64::new(0),
            fills_received: AtomicU64::new(0),
            position_updates: AtomicU64::new(0),
            balance_updates: AtomicU64::new(0),
            total_volume: AtomicU64::new(0),
            start_time: Instant::now(),
            last_signal_ms: AtomicU64::new(0),
        }
    }

    /// Record a signal received
    pub fn record_signal_received(&self, signal_type: &str) {
        self.signals_received.fetch_add(1, Ordering::Relaxed);

        {
            let mut by_type = self.signals_by_type.write();
            *by_type.entry(signal_type.to_string()).or_insert(0) += 1;
        }

        self.last_signal_ms
            .store(current_timestamp_ms(), Ordering::Relaxed);
    }

    /// Record a signal executed
    pub fn record_signal_executed(&self) {
        self.signals_executed.fetch_add(1, Ordering::Relaxed);
    }

    /// Record a signal rejected
    pub fn record_signal_rejected(&self) {
        self.signals_rejected.fetch_add(1, Ordering::Relaxed);
    }

    /// Record an order submitted
    pub fn record_order_submitted(&self) {
        self.orders_submitted.fetch_add(1, Ordering::Relaxed);
    }

    /// Record an order filled
    pub fn record_order_filled(&self, volume: Decimal) {
        self.orders_filled.fetch_add(1, Ordering::Relaxed);
        let volume_x100 = decimal_to_u64_x100(volume);
        self.total_volume.fetch_add(volume_x100, Ordering::Relaxed);
    }

    /// Record an order cancelled
    pub fn record_order_cancelled(&self) {
        self.orders_cancelled.fetch_add(1, Ordering::Relaxed);
    }

    /// Record an order rejected
    pub fn record_order_rejected(&self) {
        self.orders_rejected.fetch_add(1, Ordering::Relaxed);
    }

    /// Record a fill received
    pub fn record_fill_received(&self) {
        self.fills_received.fetch_add(1, Ordering::Relaxed);
    }

    /// Record a position update
    pub fn record_position_update(&self) {
        self.position_updates.fetch_add(1, Ordering::Relaxed);
    }

    /// Record a balance update
    pub fn record_balance_update(&self) {
        self.balance_updates.fetch_add(1, Ordering::Relaxed);
    }

    /// Get acceptance rate
    pub fn acceptance_rate(&self) -> f64 {
        let received = self.signals_received.load(Ordering::Relaxed);
        if received == 0 {
            return 1.0;
        }
        self.signals_executed.load(Ordering::Relaxed) as f64 / received as f64
    }

    /// Get fill rate
    pub fn fill_rate(&self) -> f64 {
        let submitted = self.orders_submitted.load(Ordering::Relaxed);
        if submitted == 0 {
            return 1.0;
        }
        self.orders_filled.load(Ordering::Relaxed) as f64 / submitted as f64
    }

    /// Generate Prometheus metrics output
    pub fn to_prometheus(&self) -> String {
        let mut output = String::new();

        // Signals
        output.push_str(&format!(
            "# HELP signal_flow_signals_received Total trading signals received\n\
             # TYPE signal_flow_signals_received counter\n\
             signal_flow_signals_received {}\n",
            self.signals_received.load(Ordering::Relaxed)
        ));

        output.push_str(&format!(
            "# HELP signal_flow_signals_executed Signals successfully executed\n\
             # TYPE signal_flow_signals_executed counter\n\
             signal_flow_signals_executed {}\n",
            self.signals_executed.load(Ordering::Relaxed)
        ));

        output.push_str(&format!(
            "# HELP signal_flow_signals_rejected Signals rejected\n\
             # TYPE signal_flow_signals_rejected counter\n\
             signal_flow_signals_rejected {}\n",
            self.signals_rejected.load(Ordering::Relaxed)
        ));

        // Signals by type
        output.push_str(
            "# HELP signal_flow_signals_by_type Signals by signal type\n\
             # TYPE signal_flow_signals_by_type counter\n",
        );
        for (signal_type, count) in self.signals_by_type.read().iter() {
            output.push_str(&format!(
                "signal_flow_signals_by_type{{type=\"{}\"}} {}\n",
                signal_type, count
            ));
        }

        // Orders
        output.push_str(&format!(
            "# HELP signal_flow_orders_submitted Orders submitted to exchange\n\
             # TYPE signal_flow_orders_submitted counter\n\
             signal_flow_orders_submitted {}\n",
            self.orders_submitted.load(Ordering::Relaxed)
        ));

        output.push_str(&format!(
            "# HELP signal_flow_orders_filled Orders filled\n\
             # TYPE signal_flow_orders_filled counter\n\
             signal_flow_orders_filled {}\n",
            self.orders_filled.load(Ordering::Relaxed)
        ));

        output.push_str(&format!(
            "# HELP signal_flow_orders_cancelled Orders cancelled\n\
             # TYPE signal_flow_orders_cancelled counter\n\
             signal_flow_orders_cancelled {}\n",
            self.orders_cancelled.load(Ordering::Relaxed)
        ));

        output.push_str(&format!(
            "# HELP signal_flow_orders_rejected Orders rejected by exchange\n\
             # TYPE signal_flow_orders_rejected counter\n\
             signal_flow_orders_rejected {}\n",
            self.orders_rejected.load(Ordering::Relaxed)
        ));

        // Fills
        output.push_str(&format!(
            "# HELP signal_flow_fills_received Fill events received\n\
             # TYPE signal_flow_fills_received counter\n\
             signal_flow_fills_received {}\n",
            self.fills_received.load(Ordering::Relaxed)
        ));

        // Updates
        output.push_str(&format!(
            "# HELP signal_flow_position_updates Position updates received\n\
             # TYPE signal_flow_position_updates counter\n\
             signal_flow_position_updates {}\n",
            self.position_updates.load(Ordering::Relaxed)
        ));

        output.push_str(&format!(
            "# HELP signal_flow_balance_updates Balance updates received\n\
             # TYPE signal_flow_balance_updates counter\n\
             signal_flow_balance_updates {}\n",
            self.balance_updates.load(Ordering::Relaxed)
        ));

        // Volume
        output.push_str(&format!(
            "# HELP signal_flow_total_volume Total traded volume\n\
             # TYPE signal_flow_total_volume counter\n\
             signal_flow_total_volume {:.2}\n",
            self.total_volume.load(Ordering::Relaxed) as f64 / 100.0
        ));

        // Rates
        output.push_str(&format!(
            "# HELP signal_flow_acceptance_rate Signal acceptance rate (0-1)\n\
             # TYPE signal_flow_acceptance_rate gauge\n\
             signal_flow_acceptance_rate {:.4}\n",
            self.acceptance_rate()
        ));

        output.push_str(&format!(
            "# HELP signal_flow_fill_rate Order fill rate (0-1)\n\
             # TYPE signal_flow_fill_rate gauge\n\
             signal_flow_fill_rate {:.4}\n",
            self.fill_rate()
        ));

        // Uptime
        output.push_str(&format!(
            "# HELP signal_flow_uptime_seconds Time since metrics started\n\
             # TYPE signal_flow_uptime_seconds gauge\n\
             signal_flow_uptime_seconds {}\n",
            self.start_time.elapsed().as_secs()
        ));

        output
    }

    /// Reset all metrics
    pub fn reset(&self) {
        self.signals_received.store(0, Ordering::Relaxed);
        self.signals_executed.store(0, Ordering::Relaxed);
        self.signals_rejected.store(0, Ordering::Relaxed);
        self.signals_by_type.write().clear();
        self.orders_submitted.store(0, Ordering::Relaxed);
        self.orders_filled.store(0, Ordering::Relaxed);
        self.orders_cancelled.store(0, Ordering::Relaxed);
        self.orders_rejected.store(0, Ordering::Relaxed);
        self.fills_received.store(0, Ordering::Relaxed);
        self.position_updates.store(0, Ordering::Relaxed);
        self.balance_updates.store(0, Ordering::Relaxed);
        self.total_volume.store(0, Ordering::Relaxed);
    }
}

impl Default for SignalFlowMetrics {
    fn default() -> Self {
        Self::new()
    }
}

// ============================================================================
// Global Execution Metrics Registry
// ============================================================================

/// Global registry for all execution metrics
#[derive(Debug)]
pub struct ExecutionMetricsRegistry {
    /// Retry metrics by operation name
    retry_metrics: RwLock<HashMap<String, Arc<RetryMetrics>>>,

    /// Arbitrage metrics
    arbitrage: Arc<ArbitrageMetrics>,

    /// Best execution metrics
    best_execution: Arc<BestExecutionMetrics>,

    /// Signal flow metrics
    signal_flow: Arc<SignalFlowMetrics>,
}

impl ExecutionMetricsRegistry {
    /// Create a new registry
    pub fn new() -> Self {
        Self {
            retry_metrics: RwLock::new(HashMap::new()),
            arbitrage: Arc::new(ArbitrageMetrics::new()),
            best_execution: Arc::new(BestExecutionMetrics::new()),
            signal_flow: Arc::new(SignalFlowMetrics::new()),
        }
    }

    /// Get or create retry metrics for an operation
    pub fn retry_metrics(&self, operation: &str) -> Arc<RetryMetrics> {
        let mut metrics = self.retry_metrics.write();
        metrics
            .entry(operation.to_string())
            .or_insert_with(|| Arc::new(RetryMetrics::new(operation)))
            .clone()
    }

    /// Get arbitrage metrics
    pub fn arbitrage(&self) -> Arc<ArbitrageMetrics> {
        self.arbitrage.clone()
    }

    /// Get best execution metrics
    pub fn best_execution(&self) -> Arc<BestExecutionMetrics> {
        self.best_execution.clone()
    }

    /// Get signal flow metrics
    pub fn signal_flow(&self) -> Arc<SignalFlowMetrics> {
        self.signal_flow.clone()
    }

    /// Generate combined Prometheus metrics output
    pub fn to_prometheus(&self) -> String {
        let mut output = String::new();

        // Retry metrics
        for (_, metrics) in self.retry_metrics.read().iter() {
            output.push_str(&metrics.to_prometheus());
            output.push('\n');
        }

        // Arbitrage metrics
        output.push_str(&self.arbitrage.to_prometheus());
        output.push('\n');

        // Best execution metrics
        output.push_str(&self.best_execution.to_prometheus());
        output.push('\n');

        // Signal flow metrics
        output.push_str(&self.signal_flow.to_prometheus());

        output
    }

    /// Reset all metrics
    pub fn reset_all(&self) {
        for (_, metrics) in self.retry_metrics.read().iter() {
            metrics.reset();
        }
        self.arbitrage.reset();
        self.best_execution.reset();
        self.signal_flow.reset();
    }
}

impl Default for ExecutionMetricsRegistry {
    fn default() -> Self {
        Self::new()
    }
}

// ============================================================================
// Global Instance
// ============================================================================

/// Global execution metrics registry
static GLOBAL_EXECUTION_METRICS: LazyLock<ExecutionMetricsRegistry> =
    LazyLock::new(ExecutionMetricsRegistry::new);

/// Get the global execution metrics registry
pub fn global_execution_metrics() -> &'static ExecutionMetricsRegistry {
    &GLOBAL_EXECUTION_METRICS
}

/// Get retry metrics for an operation (convenience function)
pub fn retry_metrics(operation: &str) -> Arc<RetryMetrics> {
    GLOBAL_EXECUTION_METRICS.retry_metrics(operation)
}

/// Get arbitrage metrics (convenience function)
pub fn arbitrage_metrics() -> Arc<ArbitrageMetrics> {
    GLOBAL_EXECUTION_METRICS.arbitrage()
}

/// Get best execution metrics (convenience function)
pub fn best_execution_metrics() -> Arc<BestExecutionMetrics> {
    GLOBAL_EXECUTION_METRICS.best_execution()
}

/// Get signal flow metrics (convenience function)
pub fn signal_flow_metrics() -> Arc<SignalFlowMetrics> {
    GLOBAL_EXECUTION_METRICS.signal_flow()
}

/// Get all execution metrics as Prometheus format (convenience function)
pub fn execution_prometheus_metrics() -> String {
    GLOBAL_EXECUTION_METRICS.to_prometheus()
}

/// Get unified Prometheus metrics combining execution and exchange metrics
///
/// This function provides a comprehensive view of all system metrics including:
/// - Retry operations and circuit breaker states
/// - Arbitrage opportunity detection
/// - Best execution analysis and recommendations
/// - Signal flow coordination
/// - Exchange WebSocket connection health
///
/// # Example
///
/// ```rust,ignore
/// use janus_execution::execution::metrics::unified_prometheus_metrics;
///
/// // Serve from HTTP endpoint
/// let metrics = unified_prometheus_metrics();
/// println!("{}", metrics);
/// ```
pub fn unified_prometheus_metrics() -> String {
    let mut output = String::new();

    // Header
    output.push_str("# FKS Unified Execution Metrics\n");
    output.push_str(&format!(
        "# Generated at: {}\n\n",
        chrono::Utc::now().to_rfc3339()
    ));

    // Execution metrics (retry, arbitrage, best-execution, signal-flow)
    output.push_str("# ==============================================\n");
    output.push_str("# Execution Subsystem Metrics\n");
    output.push_str("# ==============================================\n\n");
    output.push_str(&GLOBAL_EXECUTION_METRICS.to_prometheus());

    // Exchange WebSocket metrics
    output.push_str("\n# ==============================================\n");
    output.push_str("# Exchange WebSocket Metrics\n");
    output.push_str("# ==============================================\n\n");
    output.push_str(&crate::exchanges::metrics::prometheus_metrics());

    // Latency histogram metrics
    output.push_str("\n# ==============================================\n");
    output.push_str("# Latency Histogram Metrics\n");
    output.push_str("# ==============================================\n\n");
    output.push_str(&crate::execution::histogram::latency_prometheus_metrics());

    output
}

// ============================================================================
// Helper Functions
// ============================================================================

/// Get current timestamp in milliseconds
fn current_timestamp_ms() -> u64 {
    use std::time::{SystemTime, UNIX_EPOCH};
    SystemTime::now()
        .duration_since(UNIX_EPOCH)
        .map(|d| d.as_millis() as u64)
        .unwrap_or(0)
}

/// Convert Decimal to i64 with 100x multiplier
fn decimal_to_i64_x100(d: Decimal) -> i64 {
    use rust_decimal::prelude::ToPrimitive;
    (d * Decimal::from(100)).trunc().to_i64().unwrap_or(0)
}

/// Convert i64 with 100x multiplier back to Decimal
fn i64_x100_to_decimal(v: i64) -> Decimal {
    Decimal::new(v, 2)
}

/// Convert Decimal to u64 with 100x multiplier
fn decimal_to_u64_x100(d: Decimal) -> u64 {
    use rust_decimal::prelude::ToPrimitive;
    (d * Decimal::from(100)).trunc().to_u64().unwrap_or(0)
}

/// Convert Decimal to i64 with 10000x multiplier
fn decimal_to_i64_x10000(d: Decimal) -> i64 {
    use rust_decimal::prelude::ToPrimitive;
    (d * Decimal::from(10000)).trunc().to_i64().unwrap_or(0)
}

/// Convert Decimal to u64 with 10000x multiplier
fn decimal_to_u64_x10000(d: Decimal) -> u64 {
    use rust_decimal::prelude::ToPrimitive;
    (d * Decimal::from(10000)).trunc().to_u64().unwrap_or(0)
}

/// Convert u64 with 10000x multiplier back to Decimal
fn u64_x10000_to_decimal(v: u64) -> Decimal {
    Decimal::new(v as i64, 4)
}

// ============================================================================
// Tests
// ============================================================================

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_retry_metrics_creation() {
        let metrics = RetryMetrics::new("test_op");
        assert_eq!(metrics.operation, "test_op");
        assert_eq!(metrics.success_rate(), 1.0);
    }

    #[test]
    fn test_retry_metrics_success() {
        let metrics = RetryMetrics::new("test_op");

        metrics.record_success(1, 100);
        assert_eq!(metrics.total_operations.load(Ordering::Relaxed), 1);
        assert_eq!(metrics.first_try_successes.load(Ordering::Relaxed), 1);
        assert_eq!(metrics.success_rate(), 1.0);
        assert_eq!(metrics.first_try_rate(), 1.0);

        metrics.record_success(3, 200);
        assert_eq!(metrics.total_operations.load(Ordering::Relaxed), 2);
        assert_eq!(metrics.retry_successes.load(Ordering::Relaxed), 1);
        assert_eq!(metrics.total_retries.load(Ordering::Relaxed), 2);
    }

    #[test]
    fn test_retry_metrics_failure() {
        let metrics = RetryMetrics::new("test_op");

        metrics.record_failure(3, 500);
        assert_eq!(metrics.total_operations.load(Ordering::Relaxed), 1);
        assert_eq!(metrics.total_failures.load(Ordering::Relaxed), 1);
        assert_eq!(metrics.success_rate(), 0.0);
    }

    #[test]
    fn test_retry_metrics_errors() {
        let metrics = RetryMetrics::new("test_op");

        metrics.record_rate_limit();
        metrics.record_network_error();
        metrics.record_timeout();

        assert_eq!(metrics.rate_limit_errors.load(Ordering::Relaxed), 1);
        assert_eq!(metrics.network_errors.load(Ordering::Relaxed), 1);
        assert_eq!(metrics.timeout_errors.load(Ordering::Relaxed), 1);
    }

    #[test]
    fn test_retry_metrics_circuit_breaker() {
        let metrics = RetryMetrics::new("test_op");

        metrics.set_circuit_breaker_state(CircuitBreakerState::Open);
        assert_eq!(metrics.circuit_breaker_state.load(Ordering::Relaxed), 1);

        metrics.record_circuit_breaker_failure();
        assert_eq!(metrics.circuit_breaker_failures.load(Ordering::Relaxed), 1);
    }

    #[test]
    fn test_retry_metrics_prometheus() {
        let metrics = RetryMetrics::new("test_op");
        metrics.record_success(1, 100);

        let output = metrics.to_prometheus();
        assert!(output.contains("execution_retry_total"));
        assert!(output.contains("execution_retry_success"));
        assert!(output.contains("operation=\"test_op\""));
    }

    #[test]
    fn test_arbitrage_metrics_creation() {
        let metrics = ArbitrageMetrics::new();
        assert_eq!(metrics.opportunities_total.load(Ordering::Relaxed), 0);
    }

    #[test]
    fn test_arbitrage_metrics_opportunity() {
        let metrics = ArbitrageMetrics::new();

        metrics.record_opportunity("BTC/USDT", Decimal::from(50), true);

        assert_eq!(metrics.opportunities_total.load(Ordering::Relaxed), 1);
        assert_eq!(metrics.opportunities_actionable.load(Ordering::Relaxed), 1);
        assert_eq!(metrics.get_spread_bps("BTC/USDT"), Some(Decimal::from(50)));
    }

    #[test]
    fn test_arbitrage_metrics_price_update() {
        let metrics = ArbitrageMetrics::new();

        metrics.record_price_update("kraken");
        metrics.record_price_update("binance");
        metrics.record_price_update("kraken");

        assert_eq!(metrics.price_updates.load(Ordering::Relaxed), 3);

        let by_exchange = metrics.price_updates_by_exchange.read();
        assert_eq!(by_exchange.get("kraken"), Some(&2));
        assert_eq!(by_exchange.get("binance"), Some(&1));
    }

    #[test]
    fn test_arbitrage_metrics_prometheus() {
        let metrics = ArbitrageMetrics::new();
        metrics.record_opportunity("BTC/USDT", Decimal::from(50), true);

        let output = metrics.to_prometheus();
        assert!(output.contains("arbitrage_opportunities_total"));
        assert!(output.contains("arbitrage_spread_bps"));
    }

    #[test]
    fn test_best_execution_metrics_creation() {
        let metrics = BestExecutionMetrics::new();
        assert_eq!(metrics.analyses_total.load(Ordering::Relaxed), 0);
    }

    #[test]
    fn test_best_execution_metrics_analysis() {
        let metrics = BestExecutionMetrics::new();

        metrics.record_analysis(
            "BTC/USDT",
            RecommendationType::ExecuteNow,
            Decimal::new(85, 2), // 0.85
            Decimal::from(5),
        );

        assert_eq!(metrics.analyses_total.load(Ordering::Relaxed), 1);

        let recs = metrics.current_recommendation.read();
        assert_eq!(recs.get("BTC/USDT"), Some(&RecommendationType::ExecuteNow));
    }

    #[test]
    fn test_best_execution_metrics_execution() {
        let metrics = BestExecutionMetrics::new();

        metrics.record_execution("BTC/USDT", Decimal::new(90, 2), false);
        metrics.record_execution("BTC/USDT", Decimal::new(80, 2), true);

        assert_eq!(metrics.executions_total.load(Ordering::Relaxed), 2);
        assert_eq!(metrics.slippage_exceeded_count.load(Ordering::Relaxed), 1);
    }

    #[test]
    fn test_best_execution_metrics_prometheus() {
        let metrics = BestExecutionMetrics::new();
        metrics.record_analysis(
            "BTC/USDT",
            RecommendationType::Wait,
            Decimal::new(65, 2),
            Decimal::from(10),
        );

        let output = metrics.to_prometheus();
        assert!(output.contains("best_execution_analyses_total"));
        assert!(output.contains("best_execution_recommendation"));
    }

    #[test]
    fn test_signal_flow_metrics_creation() {
        let metrics = SignalFlowMetrics::new();
        assert_eq!(metrics.signals_received.load(Ordering::Relaxed), 0);
        assert_eq!(metrics.acceptance_rate(), 1.0);
    }

    #[test]
    fn test_signal_flow_metrics_signals() {
        let metrics = SignalFlowMetrics::new();

        metrics.record_signal_received("entry");
        metrics.record_signal_executed();
        metrics.record_signal_received("exit");
        metrics.record_signal_rejected();

        assert_eq!(metrics.signals_received.load(Ordering::Relaxed), 2);
        assert_eq!(metrics.signals_executed.load(Ordering::Relaxed), 1);
        assert_eq!(metrics.signals_rejected.load(Ordering::Relaxed), 1);
        assert_eq!(metrics.acceptance_rate(), 0.5);
    }

    #[test]
    fn test_signal_flow_metrics_orders() {
        let metrics = SignalFlowMetrics::new();

        metrics.record_order_submitted();
        metrics.record_order_submitted();
        metrics.record_order_filled(Decimal::from(100));
        metrics.record_order_cancelled();

        assert_eq!(metrics.orders_submitted.load(Ordering::Relaxed), 2);
        assert_eq!(metrics.orders_filled.load(Ordering::Relaxed), 1);
        assert_eq!(metrics.orders_cancelled.load(Ordering::Relaxed), 1);
        assert_eq!(metrics.fill_rate(), 0.5);
    }

    #[test]
    fn test_signal_flow_metrics_prometheus() {
        let metrics = SignalFlowMetrics::new();
        metrics.record_signal_received("entry");
        metrics.record_order_submitted();

        let output = metrics.to_prometheus();
        assert!(output.contains("signal_flow_signals_received"));
        assert!(output.contains("signal_flow_orders_submitted"));
    }

    #[test]
    fn test_registry_creation() {
        let registry = ExecutionMetricsRegistry::new();

        let retry = registry.retry_metrics("test_op");
        assert_eq!(retry.operation, "test_op");

        let arb = registry.arbitrage();
        assert_eq!(arb.opportunities_total.load(Ordering::Relaxed), 0);

        let best = registry.best_execution();
        assert_eq!(best.analyses_total.load(Ordering::Relaxed), 0);

        let flow = registry.signal_flow();
        assert_eq!(flow.signals_received.load(Ordering::Relaxed), 0);
    }

    #[test]
    fn test_registry_prometheus() {
        let registry = ExecutionMetricsRegistry::new();

        registry.retry_metrics("order_submit").record_success(1, 50);
        registry
            .arbitrage()
            .record_opportunity("BTC/USDT", Decimal::from(25), false);
        registry.best_execution().record_analysis(
            "ETH/USDT",
            RecommendationType::Acceptable,
            Decimal::new(75, 2),
            Decimal::from(3),
        );
        registry.signal_flow().record_signal_received("entry");

        let output = registry.to_prometheus();
        assert!(output.contains("execution_retry_total"));
        assert!(output.contains("arbitrage_opportunities_total"));
        assert!(output.contains("best_execution_analyses_total"));
        assert!(output.contains("signal_flow_signals_received"));
    }

    #[test]
    fn test_global_metrics() {
        let retry = retry_metrics("global_test");
        retry.record_success(1, 10);

        let arb = arbitrage_metrics();
        arb.record_price_update("test_exchange");

        let best = best_execution_metrics();
        best.record_analysis(
            "TEST/USDT",
            RecommendationType::Wait,
            Decimal::new(50, 2),
            Decimal::from(8),
        );

        let flow = signal_flow_metrics();
        flow.record_signal_received("test");

        let output = execution_prometheus_metrics();
        assert!(!output.is_empty());
    }

    #[test]
    fn test_recommendation_type() {
        assert_eq!(
            RecommendationType::from_str("execute_now"),
            RecommendationType::ExecuteNow
        );
        assert_eq!(
            RecommendationType::from_str("wait"),
            RecommendationType::Wait
        );
        assert_eq!(
            RecommendationType::from_str("avoid"),
            RecommendationType::Avoid
        );
        assert_eq!(RecommendationType::ExecuteNow.as_str(), "execute_now");
    }

    #[test]
    fn test_decimal_conversions() {
        let d = Decimal::new(1234, 2); // 12.34
        let x100 = decimal_to_i64_x100(d);
        assert_eq!(x100, 1234);

        let back = i64_x100_to_decimal(x100);
        assert_eq!(back, d);
    }

    #[test]
    fn test_metrics_reset() {
        let retry = RetryMetrics::new("reset_test");
        retry.record_success(1, 100);
        retry.record_failure(2, 200);
        retry.reset();

        assert_eq!(retry.total_operations.load(Ordering::Relaxed), 0);
        assert_eq!(retry.total_failures.load(Ordering::Relaxed), 0);

        let arb = ArbitrageMetrics::new();
        arb.record_opportunity("BTC/USDT", Decimal::from(50), true);
        arb.reset();

        assert_eq!(arb.opportunities_total.load(Ordering::Relaxed), 0);
        assert!(arb.opportunities_by_symbol.read().is_empty());
    }

    #[test]
    fn test_unified_prometheus_metrics() {
        // Record some data across different metric types
        let retry = retry_metrics("unified_test");
        retry.record_success(1, 15);

        let arb = arbitrage_metrics();
        arb.record_price_update("unified_exchange");

        let best = best_execution_metrics();
        best.record_analysis(
            "UNIFIED/USDT",
            RecommendationType::ExecuteNow,
            Decimal::new(75, 2),
            Decimal::from(3),
        );

        let flow = signal_flow_metrics();
        flow.record_signal_received("unified_signal");

        // Get unified metrics output
        let output = unified_prometheus_metrics();

        // Verify it contains the header
        assert!(output.contains("FKS Unified Execution Metrics"));
        assert!(output.contains("Generated at:"));

        // Verify it contains execution subsystem metrics section
        assert!(output.contains("Execution Subsystem Metrics"));

        // Verify it contains exchange websocket metrics section
        assert!(output.contains("Exchange WebSocket Metrics"));

        // Verify actual metrics are present
        assert!(output.contains("execution_retry_"));
        assert!(output.contains("arbitrage_"));
        assert!(output.contains("best_execution_"));
        assert!(output.contains("signal_flow_"));
    }
}
