//! Liquidity Crisis Detection
//!
//! Monitors order book health and detects liquidity crises that could
//! lead to severe slippage or inability to execute orders.
//!
//! Detection methods:
//! - Order book depth analysis
//! - Bid-ask spread monitoring
//! - Volume availability tracking
//! - Market maker withdrawal detection
//! - Depth imbalance analysis

use crate::common::Result;
use std::collections::VecDeque;

/// Configuration for liquidity crisis detection
#[derive(Debug, Clone)]
pub struct LiquidityCrisisConfig {
    /// Window size for historical analysis
    pub window_size: usize,
    /// Minimum samples for reliable detection
    pub min_samples: usize,
    /// Depth drop threshold for warning (percentage, e.g., 0.5 = 50% drop)
    pub depth_warning_threshold: f64,
    /// Depth drop threshold for critical (percentage)
    pub depth_critical_threshold: f64,
    /// Spread widening threshold (multiple of baseline)
    pub spread_warning_threshold: f64,
    /// Critical spread threshold
    pub spread_critical_threshold: f64,
    /// Imbalance threshold (ratio of bid to ask depth)
    pub imbalance_threshold: f64,
    /// EMA decay factor for smoothing
    pub ema_decay: f64,
    /// Number of price levels to monitor
    pub depth_levels: usize,
}

impl Default for LiquidityCrisisConfig {
    fn default() -> Self {
        Self {
            window_size: 100,
            min_samples: 20,
            depth_warning_threshold: 0.3,   // 30% depth drop
            depth_critical_threshold: 0.5,  // 50% depth drop
            spread_warning_threshold: 2.0,  // 2x normal spread
            spread_critical_threshold: 5.0, // 5x normal spread
            imbalance_threshold: 3.0,       // 3:1 ratio
            ema_decay: 0.94,
            depth_levels: 10,
        }
    }
}

/// Liquidity crisis severity levels
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum LiquiditySeverity {
    Normal,
    Stressed,
    Warning,
    Critical,
    Crisis,
}

impl LiquiditySeverity {
    pub fn as_str(&self) -> &'static str {
        match self {
            Self::Normal => "normal",
            Self::Stressed => "stressed",
            Self::Warning => "warning",
            Self::Critical => "critical",
            Self::Crisis => "crisis",
        }
    }

    pub fn is_actionable(&self) -> bool {
        matches!(self, Self::Warning | Self::Critical | Self::Crisis)
    }
}

/// Order book level data
#[derive(Debug, Clone)]
pub struct OrderBookLevel {
    pub price: f64,
    pub quantity: f64,
}

/// Order book snapshot for analysis
#[derive(Debug, Clone)]
pub struct OrderBookSnapshot {
    pub timestamp: i64,
    pub bids: Vec<OrderBookLevel>,
    pub asks: Vec<OrderBookLevel>,
    pub best_bid: f64,
    pub best_ask: f64,
}

impl OrderBookSnapshot {
    /// Calculate spread
    pub fn spread(&self) -> f64 {
        self.best_ask - self.best_bid
    }

    /// Calculate spread as percentage of mid price
    pub fn spread_bps(&self) -> f64 {
        let mid = (self.best_bid + self.best_ask) / 2.0;
        if mid > 0.0 {
            (self.spread() / mid) * 10000.0 // basis points
        } else {
            0.0
        }
    }

    /// Calculate total bid depth
    pub fn total_bid_depth(&self) -> f64 {
        self.bids.iter().map(|l| l.quantity).sum()
    }

    /// Calculate total ask depth
    pub fn total_ask_depth(&self) -> f64 {
        self.asks.iter().map(|l| l.quantity).sum()
    }

    /// Calculate depth at a given price distance (percentage from mid)
    pub fn depth_at_distance(&self, distance_pct: f64) -> (f64, f64) {
        let mid = (self.best_bid + self.best_ask) / 2.0;
        let bid_threshold = mid * (1.0 - distance_pct);
        let ask_threshold = mid * (1.0 + distance_pct);

        let bid_depth: f64 = self
            .bids
            .iter()
            .filter(|l| l.price >= bid_threshold)
            .map(|l| l.quantity)
            .sum();

        let ask_depth: f64 = self
            .asks
            .iter()
            .filter(|l| l.price <= ask_threshold)
            .map(|l| l.quantity)
            .sum();

        (bid_depth, ask_depth)
    }

    /// Calculate imbalance ratio (bid depth / ask depth)
    pub fn imbalance_ratio(&self) -> f64 {
        let bid_depth = self.total_bid_depth();
        let ask_depth = self.total_ask_depth();

        if ask_depth > 0.0 {
            bid_depth / ask_depth
        } else if bid_depth > 0.0 {
            f64::INFINITY
        } else {
            1.0
        }
    }
}

/// Liquidity metrics for analysis
#[derive(Debug, Clone, Default)]
pub struct LiquidityMetrics {
    /// Current total depth (bid + ask)
    pub total_depth: f64,
    /// Baseline total depth (historical average)
    pub baseline_depth: f64,
    /// Depth ratio (current / baseline)
    pub depth_ratio: f64,
    /// Current spread in basis points
    pub spread_bps: f64,
    /// Baseline spread
    pub baseline_spread_bps: f64,
    /// Spread ratio (current / baseline)
    pub spread_ratio: f64,
    /// Bid-ask imbalance ratio
    pub imbalance_ratio: f64,
    /// Depth at 1% from mid (bid, ask)
    pub depth_1pct: (f64, f64),
    /// Depth at 2% from mid (bid, ask)
    pub depth_2pct: (f64, f64),
    /// Market maker presence score (0-1)
    pub mm_presence: f64,
    /// Overall liquidity score (0-1, higher = better)
    pub liquidity_score: f64,
}

/// Liquidity crisis alert
#[derive(Debug, Clone)]
pub struct LiquidityAlert {
    pub severity: LiquiditySeverity,
    pub metrics: LiquidityMetrics,
    pub timestamp: i64,
    pub message: String,
    pub recommended_action: LiquidityAction,
}

/// Recommended actions for liquidity conditions
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum LiquidityAction {
    None,
    ReduceOrderSize,
    UseLimitOrders,
    DelayExecution,
    HaltTrading,
    EmergencyOnly,
}

impl LiquidityAction {
    pub fn as_str(&self) -> &'static str {
        match self {
            Self::None => "none",
            Self::ReduceOrderSize => "reduce_order_size",
            Self::UseLimitOrders => "use_limit_orders",
            Self::DelayExecution => "delay_execution",
            Self::HaltTrading => "halt_trading",
            Self::EmergencyOnly => "emergency_only",
        }
    }
}

/// Internal state for tracking liquidity
#[derive(Debug, Clone, Default)]
struct LiquidityState {
    /// Historical total depth
    depth_history: VecDeque<f64>,
    /// Historical spread (bps)
    spread_history: VecDeque<f64>,
    /// Historical imbalance
    imbalance_history: VecDeque<f64>,
    /// Running sums for efficient calculation
    depth_sum: f64,
    spread_sum: f64,
}

/// Liquidity crisis detection system
pub struct LiquidityCrisis {
    config: LiquidityCrisisConfig,
    state: LiquidityState,
    /// EMA of liquidity score
    ema_liquidity: f64,
    /// Last alert
    last_alert: Option<LiquidityAlert>,
    /// Alert count
    alert_count: u64,
    /// Current severity
    current_severity: LiquiditySeverity,
}

impl Default for LiquidityCrisis {
    fn default() -> Self {
        Self::new()
    }
}

impl LiquidityCrisis {
    /// Create a new LiquidityCrisis detector with default config
    pub fn new() -> Self {
        Self::with_config(LiquidityCrisisConfig::default())
    }

    /// Create with custom configuration
    pub fn with_config(config: LiquidityCrisisConfig) -> Self {
        let window_size = config.window_size;
        Self {
            config,
            state: LiquidityState {
                depth_history: VecDeque::with_capacity(window_size),
                spread_history: VecDeque::with_capacity(window_size),
                imbalance_history: VecDeque::with_capacity(window_size),
                ..Default::default()
            },
            ema_liquidity: 1.0, // Start assuming normal liquidity
            last_alert: None,
            alert_count: 0,
            current_severity: LiquiditySeverity::Normal,
        }
    }

    /// Update with new order book snapshot
    pub fn update(&mut self, snapshot: &OrderBookSnapshot) {
        let total_depth = snapshot.total_bid_depth() + snapshot.total_ask_depth();
        let spread_bps = snapshot.spread_bps();
        let imbalance = snapshot.imbalance_ratio();

        // Update depth history
        self.add_depth(total_depth);

        // Update spread history
        self.add_spread(spread_bps);

        // Update imbalance history
        self.add_imbalance(imbalance);
    }

    /// Analyze current liquidity conditions
    pub fn analyze(&mut self, snapshot: &OrderBookSnapshot) -> LiquidityAlert {
        let metrics = self.calculate_metrics(snapshot);
        let severity = self.determine_severity(&metrics);

        // Update EMA liquidity score
        self.ema_liquidity = self.config.ema_decay * self.ema_liquidity
            + (1.0 - self.config.ema_decay) * metrics.liquidity_score;

        // Determine action
        let recommended_action = self.determine_action(severity, &metrics);

        // Create message
        let message = self.create_message(severity, &metrics);

        let alert = LiquidityAlert {
            severity,
            metrics,
            timestamp: chrono::Utc::now().timestamp_millis(),
            message,
            recommended_action,
        };

        // Update state
        if severity.is_actionable() {
            self.alert_count += 1;
        }
        self.current_severity = severity;
        self.last_alert = Some(alert.clone());

        alert
    }

    /// Calculate comprehensive liquidity metrics
    fn calculate_metrics(&self, snapshot: &OrderBookSnapshot) -> LiquidityMetrics {
        let total_depth = snapshot.total_bid_depth() + snapshot.total_ask_depth();
        let spread_bps = snapshot.spread_bps();
        let imbalance_ratio = snapshot.imbalance_ratio();

        // Calculate baselines from history
        let baseline_depth = if self.state.depth_history.is_empty() {
            total_depth
        } else {
            self.state.depth_sum / self.state.depth_history.len() as f64
        };

        let baseline_spread_bps = if self.state.spread_history.is_empty() {
            spread_bps
        } else {
            self.state.spread_sum / self.state.spread_history.len() as f64
        };

        // Calculate ratios
        let depth_ratio = if baseline_depth > 0.0 {
            total_depth / baseline_depth
        } else {
            1.0
        };

        let spread_ratio = if baseline_spread_bps > 0.0 {
            spread_bps / baseline_spread_bps
        } else {
            1.0
        };

        // Calculate depth at distances
        let depth_1pct = snapshot.depth_at_distance(0.01);
        let depth_2pct = snapshot.depth_at_distance(0.02);

        // Estimate market maker presence
        // Based on tight spread, good depth, and balanced book
        let mm_presence = self.estimate_mm_presence(spread_ratio, depth_ratio, imbalance_ratio);

        // Calculate overall liquidity score
        let liquidity_score =
            self.calculate_liquidity_score(depth_ratio, spread_ratio, imbalance_ratio, mm_presence);

        LiquidityMetrics {
            total_depth,
            baseline_depth,
            depth_ratio,
            spread_bps,
            baseline_spread_bps,
            spread_ratio,
            imbalance_ratio,
            depth_1pct,
            depth_2pct,
            mm_presence,
            liquidity_score,
        }
    }

    /// Estimate market maker presence score
    fn estimate_mm_presence(
        &self,
        spread_ratio: f64,
        depth_ratio: f64,
        imbalance_ratio: f64,
    ) -> f64 {
        let mut score = 1.0;

        // Wide spreads suggest MM withdrawal
        if spread_ratio > 1.5 {
            score *= 0.7;
        }
        if spread_ratio > 2.0 {
            score *= 0.5;
        }

        // Low depth suggests MM withdrawal
        if depth_ratio < 0.7 {
            score *= 0.8;
        }
        if depth_ratio < 0.5 {
            score *= 0.6;
        }

        // High imbalance suggests one-sided market
        if imbalance_ratio > 2.0 || imbalance_ratio < 0.5 {
            score *= 0.8;
        }
        if imbalance_ratio > 3.0 || imbalance_ratio < 0.33 {
            score *= 0.6;
        }

        score.clamp(0.0, 1.0)
    }

    /// Calculate overall liquidity score (0-1, higher = better)
    fn calculate_liquidity_score(
        &self,
        depth_ratio: f64,
        spread_ratio: f64,
        imbalance_ratio: f64,
        mm_presence: f64,
    ) -> f64 {
        // Depth contribution (40% weight)
        let depth_score = depth_ratio.min(1.0) * 0.4;

        // Spread contribution (30% weight) - inverse relationship
        let spread_score = (1.0 / spread_ratio).min(1.0) * 0.3;

        // Balance contribution (15% weight)
        let balance_score = if imbalance_ratio > 0.0 {
            let deviation = (imbalance_ratio - 1.0).abs();
            (1.0 - deviation / 2.0).max(0.0) * 0.15
        } else {
            0.0
        };

        // MM presence contribution (15% weight)
        let mm_score = mm_presence * 0.15;

        (depth_score + spread_score + balance_score + mm_score).clamp(0.0, 1.0)
    }

    /// Determine severity based on metrics
    fn determine_severity(&self, metrics: &LiquidityMetrics) -> LiquiditySeverity {
        // Crisis: Multiple critical indicators
        if metrics.depth_ratio < 1.0 - self.config.depth_critical_threshold
            && metrics.spread_ratio > self.config.spread_critical_threshold
        {
            return LiquiditySeverity::Crisis;
        }

        // Critical: Single critical indicator
        if metrics.depth_ratio < 1.0 - self.config.depth_critical_threshold
            || metrics.spread_ratio > self.config.spread_critical_threshold
        {
            return LiquiditySeverity::Critical;
        }

        // Warning: Warning level indicators
        if metrics.depth_ratio < 1.0 - self.config.depth_warning_threshold
            || metrics.spread_ratio > self.config.spread_warning_threshold
        {
            return LiquiditySeverity::Warning;
        }

        // Stressed: Minor issues
        if metrics.liquidity_score < 0.6
            || metrics.imbalance_ratio > self.config.imbalance_threshold
            || metrics.imbalance_ratio < 1.0 / self.config.imbalance_threshold
        {
            return LiquiditySeverity::Stressed;
        }

        LiquiditySeverity::Normal
    }

    /// Determine recommended action
    fn determine_action(
        &self,
        severity: LiquiditySeverity,
        metrics: &LiquidityMetrics,
    ) -> LiquidityAction {
        match severity {
            LiquiditySeverity::Crisis => LiquidityAction::EmergencyOnly,
            LiquiditySeverity::Critical => LiquidityAction::HaltTrading,
            LiquiditySeverity::Warning => {
                if metrics.spread_ratio > self.config.spread_warning_threshold {
                    LiquidityAction::UseLimitOrders
                } else {
                    LiquidityAction::DelayExecution
                }
            }
            LiquiditySeverity::Stressed => LiquidityAction::ReduceOrderSize,
            LiquiditySeverity::Normal => LiquidityAction::None,
        }
    }

    /// Create human-readable message
    fn create_message(&self, severity: LiquiditySeverity, metrics: &LiquidityMetrics) -> String {
        match severity {
            LiquiditySeverity::Crisis => {
                format!(
                    "🚨 LIQUIDITY CRISIS: Depth at {:.0}% of normal, spread {:.1}x wider. \
                     EMERGENCY ORDERS ONLY.",
                    metrics.depth_ratio * 100.0,
                    metrics.spread_ratio
                )
            }
            LiquiditySeverity::Critical => {
                format!(
                    "⛔ CRITICAL LIQUIDITY: Depth {:.0}%, spread {:.1}x normal. \
                     Halt non-essential trading.",
                    metrics.depth_ratio * 100.0,
                    metrics.spread_ratio
                )
            }
            LiquiditySeverity::Warning => {
                format!(
                    "⚠️ LIQUIDITY WARNING: Depth {:.0}%, spread {:.1}x normal. \
                     Exercise caution with market orders.",
                    metrics.depth_ratio * 100.0,
                    metrics.spread_ratio
                )
            }
            LiquiditySeverity::Stressed => {
                format!(
                    "📉 LIQUIDITY STRESSED: Score {:.0}%, imbalance ratio {:.2}. \
                     Consider reducing order sizes.",
                    metrics.liquidity_score * 100.0,
                    metrics.imbalance_ratio
                )
            }
            LiquiditySeverity::Normal => {
                format!(
                    "✅ LIQUIDITY NORMAL: Score {:.0}%, depth stable.",
                    metrics.liquidity_score * 100.0
                )
            }
        }
    }

    /// Add depth to history
    fn add_depth(&mut self, depth: f64) {
        if self.state.depth_history.len() >= self.config.window_size {
            if let Some(old) = self.state.depth_history.pop_front() {
                self.state.depth_sum -= old;
            }
        }
        self.state.depth_history.push_back(depth);
        self.state.depth_sum += depth;
    }

    /// Add spread to history
    fn add_spread(&mut self, spread: f64) {
        if self.state.spread_history.len() >= self.config.window_size {
            if let Some(old) = self.state.spread_history.pop_front() {
                self.state.spread_sum -= old;
            }
        }
        self.state.spread_history.push_back(spread);
        self.state.spread_sum += spread;
    }

    /// Add imbalance to history
    fn add_imbalance(&mut self, imbalance: f64) {
        if self.state.imbalance_history.len() >= self.config.window_size {
            self.state.imbalance_history.pop_front();
        }
        self.state.imbalance_history.push_back(imbalance);
    }

    /// Get last alert
    pub fn last_alert(&self) -> Option<&LiquidityAlert> {
        self.last_alert.as_ref()
    }

    /// Get alert count
    pub fn alert_count(&self) -> u64 {
        self.alert_count
    }

    /// Get current severity
    pub fn current_severity(&self) -> LiquiditySeverity {
        self.current_severity
    }

    /// Get EMA liquidity score
    pub fn liquidity_score(&self) -> f64 {
        self.ema_liquidity
    }

    /// Check if in crisis
    pub fn is_crisis(&self) -> bool {
        matches!(
            self.current_severity,
            LiquiditySeverity::Crisis | LiquiditySeverity::Critical
        )
    }

    /// Check if actionable (warning or worse)
    pub fn is_actionable(&self) -> bool {
        self.current_severity.is_actionable()
    }

    /// Reset detector
    pub fn reset(&mut self) {
        self.state = LiquidityState::default();
        self.ema_liquidity = 1.0;
        self.last_alert = None;
        self.current_severity = LiquiditySeverity::Normal;
    }

    /// Main processing function (compatibility with neuromorphic interface)
    pub fn process(&self) -> Result<()> {
        Ok(())
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    fn create_snapshot(
        best_bid: f64,
        best_ask: f64,
        bid_depth: f64,
        ask_depth: f64,
    ) -> OrderBookSnapshot {
        // Create simple order book with single level
        OrderBookSnapshot {
            timestamp: chrono::Utc::now().timestamp_millis(),
            bids: vec![OrderBookLevel {
                price: best_bid,
                quantity: bid_depth,
            }],
            asks: vec![OrderBookLevel {
                price: best_ask,
                quantity: ask_depth,
            }],
            best_bid,
            best_ask,
        }
    }

    fn create_deep_snapshot(
        mid_price: f64,
        spread_bps: f64,
        depth_per_level: f64,
        levels: usize,
    ) -> OrderBookSnapshot {
        let spread = mid_price * spread_bps / 10000.0;
        let best_bid = mid_price - spread / 2.0;
        let best_ask = mid_price + spread / 2.0;

        let bids: Vec<_> = (0..levels)
            .map(|i| OrderBookLevel {
                price: best_bid - (i as f64 * 0.01 * mid_price),
                quantity: depth_per_level,
            })
            .collect();

        let asks: Vec<_> = (0..levels)
            .map(|i| OrderBookLevel {
                price: best_ask + (i as f64 * 0.01 * mid_price),
                quantity: depth_per_level,
            })
            .collect();

        OrderBookSnapshot {
            timestamp: chrono::Utc::now().timestamp_millis(),
            bids,
            asks,
            best_bid,
            best_ask,
        }
    }

    #[test]
    fn test_basic_creation() {
        let detector = LiquidityCrisis::new();
        assert_eq!(detector.current_severity(), LiquiditySeverity::Normal);
        assert!(!detector.is_crisis());
    }

    #[test]
    fn test_normal_liquidity() {
        let mut detector = LiquidityCrisis::new();

        // Build up normal baseline
        for _ in 0..30 {
            let snapshot = create_snapshot(99.95, 100.05, 10000.0, 10000.0);
            detector.update(&snapshot);
        }

        let snapshot = create_snapshot(99.95, 100.05, 10000.0, 10000.0);
        let alert = detector.analyze(&snapshot);

        assert_eq!(alert.severity, LiquiditySeverity::Normal);
        assert_eq!(alert.recommended_action, LiquidityAction::None);
    }

    #[test]
    fn test_depth_crisis() {
        let mut detector = LiquidityCrisis::new();

        // Build up normal baseline
        for _ in 0..30 {
            let snapshot = create_snapshot(99.95, 100.05, 10000.0, 10000.0);
            detector.update(&snapshot);
        }

        // Sudden depth drop (80% reduction)
        let crisis_snapshot = create_snapshot(99.95, 100.05, 2000.0, 2000.0);
        let alert = detector.analyze(&crisis_snapshot);

        assert!(alert.severity.is_actionable());
        assert!(alert.metrics.depth_ratio < 0.5);
    }

    #[test]
    fn test_spread_crisis() {
        let mut detector = LiquidityCrisis::new();

        // Build up normal baseline (tight spread)
        for _ in 0..30 {
            let snapshot = create_snapshot(99.99, 100.01, 10000.0, 10000.0);
            detector.update(&snapshot);
        }

        // Spread widens 10x
        let crisis_snapshot = create_snapshot(99.90, 100.10, 10000.0, 10000.0);
        let alert = detector.analyze(&crisis_snapshot);

        assert!(alert.metrics.spread_ratio > 2.0);
    }

    #[test]
    fn test_imbalanced_book() {
        let mut detector = LiquidityCrisis::new();

        // Build up balanced baseline
        for _ in 0..30 {
            let snapshot = create_snapshot(99.95, 100.05, 10000.0, 10000.0);
            detector.update(&snapshot);
        }

        // Heavy imbalance (5:1 bid/ask ratio)
        let imbalanced = create_snapshot(99.95, 100.05, 50000.0, 10000.0);
        let alert = detector.analyze(&imbalanced);

        assert!(alert.metrics.imbalance_ratio > 3.0);
    }

    #[test]
    fn test_order_book_snapshot_methods() {
        let snapshot = create_snapshot(99.95, 100.05, 10000.0, 10000.0);

        assert!((snapshot.spread() - 0.10).abs() < 0.001);
        assert!(snapshot.spread_bps() > 0.0);
        assert_eq!(snapshot.total_bid_depth(), 10000.0);
        assert_eq!(snapshot.total_ask_depth(), 10000.0);
        assert!((snapshot.imbalance_ratio() - 1.0).abs() < 0.001);
    }

    #[test]
    fn test_severity_levels() {
        assert!(!LiquiditySeverity::Normal.is_actionable());
        assert!(!LiquiditySeverity::Stressed.is_actionable());
        assert!(LiquiditySeverity::Warning.is_actionable());
        assert!(LiquiditySeverity::Critical.is_actionable());
        assert!(LiquiditySeverity::Crisis.is_actionable());
    }

    #[test]
    fn test_recommended_actions() {
        let detector = LiquidityCrisis::new();
        let metrics = LiquidityMetrics::default();

        let action = detector.determine_action(LiquiditySeverity::Crisis, &metrics);
        assert_eq!(action, LiquidityAction::EmergencyOnly);

        let action = detector.determine_action(LiquiditySeverity::Critical, &metrics);
        assert_eq!(action, LiquidityAction::HaltTrading);

        let action = detector.determine_action(LiquiditySeverity::Normal, &metrics);
        assert_eq!(action, LiquidityAction::None);
    }

    #[test]
    fn test_reset() {
        let mut detector = LiquidityCrisis::new();

        // Add some data
        for _ in 0..30 {
            let snapshot = create_snapshot(99.95, 100.05, 10000.0, 10000.0);
            detector.update(&snapshot);
            detector.analyze(&snapshot);
        }

        detector.reset();

        assert_eq!(detector.current_severity(), LiquiditySeverity::Normal);
        assert!(!detector.is_crisis());
    }

    #[test]
    fn test_mm_presence_estimation() {
        let detector = LiquidityCrisis::new();

        // Good conditions = high MM presence
        let high_mm = detector.estimate_mm_presence(1.0, 1.0, 1.0);
        assert!(high_mm > 0.8);

        // Bad conditions = low MM presence
        let low_mm = detector.estimate_mm_presence(3.0, 0.3, 4.0);
        assert!(low_mm < 0.5);
    }

    #[test]
    fn test_liquidity_score() {
        let detector = LiquidityCrisis::new();

        // Perfect conditions
        let high_score = detector.calculate_liquidity_score(1.0, 1.0, 1.0, 1.0);
        assert!(high_score > 0.8);

        // Poor conditions
        let low_score = detector.calculate_liquidity_score(0.3, 3.0, 5.0, 0.2);
        assert!(low_score < 0.4);
    }

    #[test]
    fn test_deep_order_book() {
        let snapshot = create_deep_snapshot(100.0, 10.0, 1000.0, 10);

        assert!(snapshot.total_bid_depth() > 0.0);
        assert!(snapshot.total_ask_depth() > 0.0);

        let (bid_1pct, ask_1pct) = snapshot.depth_at_distance(0.01);
        assert!(bid_1pct > 0.0 || ask_1pct > 0.0);
    }

    #[test]
    fn test_alert_count() {
        let mut detector = LiquidityCrisis::new();

        // Build baseline
        for _ in 0..30 {
            let snapshot = create_snapshot(99.99, 100.01, 10000.0, 10000.0);
            detector.update(&snapshot);
        }

        // Trigger alerts
        for _ in 0..5 {
            let crisis = create_snapshot(99.90, 100.10, 2000.0, 2000.0);
            detector.update(&crisis);
            detector.analyze(&crisis);
        }

        assert!(detector.alert_count() > 0);
    }

    #[test]
    fn test_process() {
        let detector = LiquidityCrisis::new();
        assert!(detector.process().is_ok());
    }
}
