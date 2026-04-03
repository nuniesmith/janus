//! Adverse Selection Model
//!
//! Part of the Cerebellum region
//! Component: Market microstructure modeling
//!
//! This module detects and quantifies adverse selection risk - the probability
//! that a counterparty has superior information. This is crucial for:
//!
//! - Market making spread determination
//! - Order toxicity classification
//! - Inventory management
//! - Fill quality assessment
//!
//! Key metrics implemented:
//! - VPIN (Volume-Synchronized Probability of Informed Trading)
//! - Kyle's Lambda (price impact coefficient)
//! - Order flow toxicity indicators
//! - Fill quality analysis

use crate::common::Result;
use std::collections::VecDeque;

/// Configuration for adverse selection model
#[derive(Debug, Clone)]
pub struct AdverseSelectionConfig {
    /// Number of buckets for VPIN calculation
    pub vpin_buckets: usize,
    /// Volume per bucket for VPIN
    pub volume_bucket_size: f64,
    /// Window size for Kyle's lambda estimation
    pub kyle_window: usize,
    /// Decay factor for exponential moving averages
    pub ema_decay: f64,
    /// Threshold for high toxicity classification
    pub high_toxicity_threshold: f64,
    /// Threshold for medium toxicity classification
    pub medium_toxicity_threshold: f64,
    /// Window for fill quality analysis (seconds)
    pub fill_quality_window: f64,
    /// Number of price levels for depth analysis
    pub depth_levels: usize,
}

impl Default for AdverseSelectionConfig {
    fn default() -> Self {
        Self {
            vpin_buckets: 50,
            volume_bucket_size: 100.0,
            kyle_window: 100,
            ema_decay: 0.94,
            high_toxicity_threshold: 0.7,
            medium_toxicity_threshold: 0.4,
            fill_quality_window: 60.0,
            depth_levels: 10,
        }
    }
}

/// A trade event for adverse selection analysis
#[derive(Debug, Clone)]
pub struct TradeEvent {
    /// Trade price
    pub price: f64,
    /// Trade quantity
    pub quantity: f64,
    /// True if buyer initiated (aggressive buy)
    pub is_buy: bool,
    /// Timestamp in microseconds
    pub timestamp: i64,
    /// Mid price at time of trade
    pub mid_price: f64,
}

impl TradeEvent {
    /// Create a new trade event
    pub fn new(price: f64, quantity: f64, is_buy: bool, timestamp: i64, mid_price: f64) -> Self {
        Self {
            price,
            quantity,
            is_buy,
            timestamp,
            mid_price,
        }
    }

    /// Calculate signed order flow (positive for buys, negative for sells)
    pub fn signed_volume(&self) -> f64 {
        if self.is_buy {
            self.quantity
        } else {
            -self.quantity
        }
    }

    /// Calculate price impact (trade price vs mid)
    pub fn price_impact(&self) -> f64 {
        if self.is_buy {
            self.price - self.mid_price
        } else {
            self.mid_price - self.price
        }
    }
}

/// Fill information for quality analysis
#[derive(Debug, Clone)]
pub struct FillInfo {
    /// Fill price
    pub price: f64,
    /// Fill quantity
    pub quantity: f64,
    /// Whether we were the maker
    pub is_maker: bool,
    /// Side of our order (true = bid/buy)
    pub is_bid: bool,
    /// Mid price at fill time
    pub mid_at_fill: f64,
    /// Mid price shortly after fill
    pub mid_after: Option<f64>,
    /// Timestamp
    pub timestamp: i64,
}

impl FillInfo {
    /// Calculate immediate adverse selection (price movement against us)
    pub fn adverse_selection(&self) -> Option<f64> {
        self.mid_after.map(|mid_after| {
            if self.is_bid {
                // We bought - adverse selection if price drops
                self.price - mid_after
            } else {
                // We sold - adverse selection if price rises
                mid_after - self.price
            }
        })
    }

    /// Calculate realized spread
    pub fn realized_spread(&self) -> Option<f64> {
        self.mid_after.map(|mid_after| {
            if self.is_bid {
                // We bought at price, can sell at mid_after
                mid_after - self.price
            } else {
                // We sold at price, can buy back at mid_after
                self.price - mid_after
            }
        })
    }
}

/// VPIN bucket for volume-synchronized analysis
#[derive(Debug, Clone, Default)]
struct VpinBucket {
    /// Buy volume in bucket
    buy_volume: f64,
    /// Sell volume in bucket
    sell_volume: f64,
    /// Total volume (should equal volume_bucket_size when complete)
    total_volume: f64,
    /// Volume-weighted average price
    vwap: f64,
    /// Start timestamp (used for bucket timing analysis)
    #[allow(dead_code)]
    start_time: i64,
    /// End timestamp
    end_time: i64,
}

impl VpinBucket {
    fn new(start_time: i64) -> Self {
        Self {
            start_time,
            end_time: start_time,
            ..Default::default()
        }
    }

    fn add_trade(&mut self, trade: &TradeEvent) {
        let volume = trade.quantity;
        if trade.is_buy {
            self.buy_volume += volume;
        } else {
            self.sell_volume += volume;
        }

        // Update VWAP
        let old_weight = self.total_volume;
        self.total_volume += volume;
        if self.total_volume > 0.0 {
            self.vwap = (self.vwap * old_weight + trade.price * volume) / self.total_volume;
        }

        self.end_time = trade.timestamp;
    }

    fn order_imbalance(&self) -> f64 {
        if self.total_volume == 0.0 {
            return 0.0;
        }
        (self.buy_volume - self.sell_volume).abs() / self.total_volume
    }
}

/// Toxicity classification
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum ToxicityLevel {
    /// Low toxicity - safe to provide liquidity
    Low,
    /// Medium toxicity - proceed with caution
    Medium,
    /// High toxicity - likely informed flow
    High,
    /// Unknown - insufficient data
    Unknown,
}

impl ToxicityLevel {
    pub fn as_str(&self) -> &'static str {
        match self {
            ToxicityLevel::Low => "low",
            ToxicityLevel::Medium => "medium",
            ToxicityLevel::High => "high",
            ToxicityLevel::Unknown => "unknown",
        }
    }
}

/// Adverse selection metrics
#[derive(Debug, Clone, Default)]
pub struct AdverseSelectionMetrics {
    /// VPIN (Volume-Synchronized Probability of Informed Trading)
    pub vpin: f64,
    /// Kyle's lambda (price impact coefficient)
    pub kyle_lambda: f64,
    /// Order flow toxicity score (0-1)
    pub toxicity_score: f64,
    /// Toxicity classification
    pub toxicity_level: Option<ToxicityLevel>,
    /// Average adverse selection per fill (in price units)
    pub avg_adverse_selection: f64,
    /// Average realized spread
    pub avg_realized_spread: f64,
    /// Proportion of fills with adverse selection
    pub adverse_fill_rate: f64,
    /// Recent order flow imbalance
    pub order_flow_imbalance: f64,
    /// Trade intensity (trades per second)
    pub trade_intensity: f64,
}

/// Adverse selection model
#[derive(Debug)]
pub struct AdverseSelectionModel {
    /// Configuration
    pub config: AdverseSelectionConfig,
    /// VPIN buckets
    vpin_buckets: VecDeque<VpinBucket>,
    /// Current bucket being filled
    current_bucket: VpinBucket,
    /// Trade history for Kyle's lambda
    trade_history: VecDeque<TradeEvent>,
    /// Fill history for quality analysis
    fill_history: VecDeque<FillInfo>,
    /// Running VPIN estimate
    vpin_ema: f64,
    /// Running toxicity estimate
    toxicity_ema: f64,
    /// Cumulative signed volume for Kyle's lambda
    cumulative_signed_volume: f64,
    /// Price at start of Kyle's lambda window
    reference_price: f64,
    /// Total trades processed
    total_trades: u64,
    /// Total volume processed
    total_volume: f64,
    /// First trade timestamp
    first_trade_time: Option<i64>,
    /// Last trade timestamp
    last_trade_time: Option<i64>,
}

impl Default for AdverseSelectionModel {
    fn default() -> Self {
        Self::new(AdverseSelectionConfig::default())
    }
}

impl AdverseSelectionModel {
    /// Create a new adverse selection model
    pub fn new(config: AdverseSelectionConfig) -> Self {
        Self {
            vpin_buckets: VecDeque::with_capacity(config.vpin_buckets),
            current_bucket: VpinBucket::new(0),
            trade_history: VecDeque::with_capacity(config.kyle_window),
            fill_history: VecDeque::with_capacity(1000),
            vpin_ema: 0.5,
            toxicity_ema: 0.3,
            cumulative_signed_volume: 0.0,
            reference_price: 0.0,
            total_trades: 0,
            total_volume: 0.0,
            first_trade_time: None,
            last_trade_time: None,
            config,
        }
    }

    /// Process a new trade
    pub fn on_trade(&mut self, trade: TradeEvent) {
        // Initialize reference price if needed
        if self.first_trade_time.is_none() {
            self.first_trade_time = Some(trade.timestamp);
            self.reference_price = trade.mid_price;
            self.current_bucket = VpinBucket::new(trade.timestamp);
        }
        self.last_trade_time = Some(trade.timestamp);
        self.total_trades += 1;
        self.total_volume += trade.quantity;

        // Add to current VPIN bucket
        self.current_bucket.add_trade(&trade);

        // Check if bucket is complete
        if self.current_bucket.total_volume >= self.config.volume_bucket_size {
            self.complete_vpin_bucket(trade.timestamp);
        }

        // Update trade history for Kyle's lambda
        self.trade_history.push_back(trade.clone());
        if self.trade_history.len() > self.config.kyle_window {
            let removed = self.trade_history.pop_front().unwrap();
            self.cumulative_signed_volume -= removed.signed_volume();
        }
        self.cumulative_signed_volume += trade.signed_volume();

        // Update toxicity EMA
        let instant_toxicity = self.calculate_instant_toxicity(&trade);
        self.toxicity_ema = self.config.ema_decay * self.toxicity_ema
            + (1.0 - self.config.ema_decay) * instant_toxicity;
    }

    /// Complete a VPIN bucket and start a new one
    fn complete_vpin_bucket(&mut self, timestamp: i64) {
        // Add completed bucket
        self.vpin_buckets.push_back(self.current_bucket.clone());

        // Maintain fixed window size
        if self.vpin_buckets.len() > self.config.vpin_buckets {
            self.vpin_buckets.pop_front();
        }

        // Update VPIN EMA
        let vpin = self.calculate_vpin();
        self.vpin_ema =
            self.config.ema_decay * self.vpin_ema + (1.0 - self.config.ema_decay) * vpin;

        // Start new bucket
        self.current_bucket = VpinBucket::new(timestamp);
    }

    /// Calculate current VPIN
    fn calculate_vpin(&self) -> f64 {
        if self.vpin_buckets.is_empty() {
            return 0.5; // Neutral prior
        }

        let total_imbalance: f64 = self.vpin_buckets.iter().map(|b| b.order_imbalance()).sum();

        total_imbalance / self.vpin_buckets.len() as f64
    }

    /// Calculate Kyle's lambda (price impact coefficient)
    fn calculate_kyle_lambda(&self) -> f64 {
        if self.trade_history.len() < 10 {
            return 0.0;
        }

        // Kyle's lambda = Cov(ΔP, SignedVolume) / Var(SignedVolume)
        let n = self.trade_history.len() as f64;

        // Calculate means
        let mean_signed_vol = self.cumulative_signed_volume / n;
        let first_price = self
            .trade_history
            .front()
            .map(|t| t.mid_price)
            .unwrap_or(0.0);
        let last_price = self
            .trade_history
            .back()
            .map(|t| t.mid_price)
            .unwrap_or(0.0);
        let _price_change = last_price - first_price;

        // Calculate variance of signed volume
        let mut var_signed_vol = 0.0;
        let mut covariance = 0.0;
        let mut running_price = first_price;

        for trade in &self.trade_history {
            let signed_vol = trade.signed_volume();
            let price_delta = trade.mid_price - running_price;

            var_signed_vol += (signed_vol - mean_signed_vol).powi(2);
            covariance += (signed_vol - mean_signed_vol) * price_delta;

            running_price = trade.mid_price;
        }

        if var_signed_vol.abs() < 1e-10 {
            return 0.0;
        }

        (covariance / var_signed_vol).abs()
    }

    /// Calculate instant toxicity from a single trade
    fn calculate_instant_toxicity(&self, trade: &TradeEvent) -> f64 {
        let mut toxicity = 0.0;

        // Factor 1: Trade size relative to recent average
        let avg_size = if self.total_trades > 0 {
            self.total_volume / self.total_trades as f64
        } else {
            trade.quantity
        };
        let size_ratio = trade.quantity / avg_size;
        toxicity += (size_ratio - 1.0).clamp(0.0, 1.0) * 0.3;

        // Factor 2: Price impact
        let impact = trade.price_impact().abs();
        let normalized_impact = (impact / trade.mid_price * 10000.0).min(1.0); // In bps, capped
        toxicity += normalized_impact * 0.3;

        // Factor 3: Alignment with recent flow
        let flow_alignment = if self.cumulative_signed_volume != 0.0 {
            let trade_direction = if trade.is_buy { 1.0 } else { -1.0 };
            let flow_direction = self.cumulative_signed_volume.signum();
            if trade_direction == flow_direction {
                0.3 // Trade aligns with recent flow
            } else {
                0.0
            }
        } else {
            0.15
        };
        toxicity += flow_alignment;

        // Factor 4: Current VPIN level
        toxicity += self.vpin_ema * 0.1;

        toxicity.min(1.0)
    }

    /// Record a fill for quality analysis
    pub fn record_fill(&mut self, fill: FillInfo) {
        self.fill_history.push_back(fill);

        // Maintain reasonable history size
        if self.fill_history.len() > 1000 {
            self.fill_history.pop_front();
        }
    }

    /// Update fill with post-fill mid price
    pub fn update_fill_outcome(&mut self, fill_timestamp: i64, mid_after: f64) {
        // Find the fill and update it
        for fill in self.fill_history.iter_mut().rev() {
            if fill.timestamp == fill_timestamp && fill.mid_after.is_none() {
                fill.mid_after = Some(mid_after);
                break;
            }
        }
    }

    /// Get current adverse selection metrics
    pub fn metrics(&self) -> AdverseSelectionMetrics {
        let vpin = self.calculate_vpin();
        let kyle_lambda = self.calculate_kyle_lambda();

        // Combine into toxicity score
        let toxicity_score =
            0.5 * vpin + 0.3 * self.toxicity_ema + 0.2 * (kyle_lambda * 100.0).min(1.0);

        // Classify toxicity level
        let toxicity_level = if self.total_trades < 100 {
            Some(ToxicityLevel::Unknown)
        } else if toxicity_score > self.config.high_toxicity_threshold {
            Some(ToxicityLevel::High)
        } else if toxicity_score > self.config.medium_toxicity_threshold {
            Some(ToxicityLevel::Medium)
        } else {
            Some(ToxicityLevel::Low)
        };

        // Calculate fill quality metrics
        let (avg_adverse, avg_realized, adverse_rate) = self.calculate_fill_quality();

        // Calculate order flow imbalance
        let order_flow_imbalance = if self.total_volume > 0.0 {
            self.cumulative_signed_volume / self.total_volume
        } else {
            0.0
        };

        // Calculate trade intensity
        let trade_intensity = match (self.first_trade_time, self.last_trade_time) {
            (Some(first), Some(last)) if last > first => {
                let duration_secs = (last - first) as f64 / 1_000_000.0;
                if duration_secs > 0.0 {
                    self.total_trades as f64 / duration_secs
                } else {
                    0.0
                }
            }
            _ => 0.0,
        };

        AdverseSelectionMetrics {
            vpin,
            kyle_lambda,
            toxicity_score,
            toxicity_level,
            avg_adverse_selection: avg_adverse,
            avg_realized_spread: avg_realized,
            adverse_fill_rate: adverse_rate,
            order_flow_imbalance,
            trade_intensity,
        }
    }

    /// Calculate fill quality metrics from history
    fn calculate_fill_quality(&self) -> (f64, f64, f64) {
        let fills_with_outcome: Vec<_> = self
            .fill_history
            .iter()
            .filter(|f| f.mid_after.is_some())
            .collect();

        if fills_with_outcome.is_empty() {
            return (0.0, 0.0, 0.0);
        }

        let mut total_adverse = 0.0;
        let mut total_realized = 0.0;
        let mut adverse_count = 0;

        for fill in &fills_with_outcome {
            if let Some(adverse) = fill.adverse_selection() {
                total_adverse += adverse;
                if adverse > 0.0 {
                    adverse_count += 1;
                }
            }
            if let Some(realized) = fill.realized_spread() {
                total_realized += realized;
            }
        }

        let n = fills_with_outcome.len() as f64;
        (
            total_adverse / n,
            total_realized / n,
            adverse_count as f64 / n,
        )
    }

    /// Get VPIN value
    pub fn vpin(&self) -> f64 {
        self.vpin_ema
    }

    /// Get toxicity score
    pub fn toxicity(&self) -> f64 {
        self.toxicity_ema
    }

    /// Check if current flow is toxic
    pub fn is_toxic(&self) -> bool {
        self.toxicity_ema > self.config.high_toxicity_threshold
    }

    /// Get recommended spread adjustment factor
    pub fn spread_adjustment(&self) -> f64 {
        // Widen spread when toxicity is high
        let base_adjustment = 1.0;
        let toxicity_factor = self.toxicity_ema * 2.0; // Up to 2x wider at max toxicity
        let vpin_factor = self.vpin_ema * 0.5; // Up to 0.5x from VPIN

        base_adjustment + toxicity_factor + vpin_factor
    }

    /// Reset the model
    pub fn reset(&mut self) {
        self.vpin_buckets.clear();
        self.current_bucket = VpinBucket::new(0);
        self.trade_history.clear();
        self.fill_history.clear();
        self.vpin_ema = 0.5;
        self.toxicity_ema = 0.3;
        self.cumulative_signed_volume = 0.0;
        self.reference_price = 0.0;
        self.total_trades = 0;
        self.total_volume = 0.0;
        self.first_trade_time = None;
        self.last_trade_time = None;
    }

    /// Get model statistics
    pub fn stats(&self) -> AdverseSelectionStats {
        AdverseSelectionStats {
            total_trades: self.total_trades,
            total_volume: self.total_volume,
            vpin_buckets_filled: self.vpin_buckets.len(),
            fill_history_size: self.fill_history.len(),
            current_vpin: self.vpin_ema,
            current_toxicity: self.toxicity_ema,
        }
    }

    /// Main processing function (for interface compatibility)
    pub fn process(&self) -> Result<()> {
        Ok(())
    }
}

/// Statistics about the adverse selection model
#[derive(Debug, Clone)]
pub struct AdverseSelectionStats {
    pub total_trades: u64,
    pub total_volume: f64,
    pub vpin_buckets_filled: usize,
    pub fill_history_size: usize,
    pub current_vpin: f64,
    pub current_toxicity: f64,
}

/// Builder for AdverseSelectionModel
#[derive(Debug, Clone)]
pub struct AdverseSelectionModelBuilder {
    config: AdverseSelectionConfig,
}

impl AdverseSelectionModelBuilder {
    pub fn new() -> Self {
        Self {
            config: AdverseSelectionConfig::default(),
        }
    }

    pub fn vpin_buckets(mut self, buckets: usize) -> Self {
        self.config.vpin_buckets = buckets;
        self
    }

    pub fn volume_bucket_size(mut self, size: f64) -> Self {
        self.config.volume_bucket_size = size;
        self
    }

    pub fn kyle_window(mut self, window: usize) -> Self {
        self.config.kyle_window = window;
        self
    }

    pub fn ema_decay(mut self, decay: f64) -> Self {
        self.config.ema_decay = decay;
        self
    }

    pub fn high_toxicity_threshold(mut self, threshold: f64) -> Self {
        self.config.high_toxicity_threshold = threshold;
        self
    }

    pub fn medium_toxicity_threshold(mut self, threshold: f64) -> Self {
        self.config.medium_toxicity_threshold = threshold;
        self
    }

    pub fn fill_quality_window(mut self, window: f64) -> Self {
        self.config.fill_quality_window = window;
        self
    }

    pub fn build(self) -> AdverseSelectionModel {
        AdverseSelectionModel::new(self.config)
    }
}

impl Default for AdverseSelectionModelBuilder {
    fn default() -> Self {
        Self::new()
    }
}

/// Presets for common configurations
pub struct AdverseSelectionPresets;

impl AdverseSelectionPresets {
    /// Configuration for high-frequency market making
    pub fn hft_market_making() -> AdverseSelectionModel {
        AdverseSelectionModelBuilder::new()
            .vpin_buckets(30)
            .volume_bucket_size(50.0)
            .kyle_window(50)
            .ema_decay(0.9)
            .high_toxicity_threshold(0.6)
            .build()
    }

    /// Configuration for slower execution
    pub fn execution() -> AdverseSelectionModel {
        AdverseSelectionModelBuilder::new()
            .vpin_buckets(100)
            .volume_bucket_size(200.0)
            .kyle_window(200)
            .ema_decay(0.98)
            .high_toxicity_threshold(0.75)
            .build()
    }

    /// Conservative configuration (more sensitive to toxicity)
    pub fn conservative() -> AdverseSelectionModel {
        AdverseSelectionModelBuilder::new()
            .vpin_buckets(50)
            .volume_bucket_size(100.0)
            .high_toxicity_threshold(0.5)
            .medium_toxicity_threshold(0.3)
            .build()
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    fn create_test_trade(is_buy: bool, quantity: f64, price: f64) -> TradeEvent {
        TradeEvent::new(price, quantity, is_buy, 1000000, 100.0)
    }

    #[test]
    fn test_model_creation() {
        let model = AdverseSelectionModel::default();
        assert_eq!(model.config.vpin_buckets, 50);
        assert_eq!(model.total_trades, 0);
    }

    #[test]
    fn test_trade_processing() {
        let mut model = AdverseSelectionModel::default();

        model.on_trade(create_test_trade(true, 10.0, 100.01));
        assert_eq!(model.total_trades, 1);
        assert_eq!(model.total_volume, 10.0);

        model.on_trade(create_test_trade(false, 5.0, 99.99));
        assert_eq!(model.total_trades, 2);
        assert_eq!(model.total_volume, 15.0);
    }

    #[test]
    fn test_vpin_calculation() {
        let mut model = AdverseSelectionModelBuilder::new()
            .volume_bucket_size(10.0)
            .vpin_buckets(5)
            .build();

        // Fill a bucket with all buys
        for i in 0..10 {
            model.on_trade(TradeEvent::new(
                100.0 + i as f64 * 0.01,
                1.0,
                true,
                i * 1000,
                100.0,
            ));
        }

        // Fill a bucket with all sells
        for i in 0..10 {
            model.on_trade(TradeEvent::new(
                100.0 - i as f64 * 0.01,
                1.0,
                false,
                (10 + i) * 1000,
                100.0,
            ));
        }

        let vpin = model.vpin();
        assert!(vpin > 0.0, "VPIN should be positive with imbalanced flow");
    }

    #[test]
    fn test_metrics() {
        let mut model = AdverseSelectionModel::default();

        // Add some trades
        for i in 0..150 {
            let is_buy = i % 3 != 0; // More buys than sells
            model.on_trade(TradeEvent::new(
                100.0 + (if is_buy { 0.01 } else { -0.01 }),
                10.0,
                is_buy,
                i * 1000,
                100.0,
            ));
        }

        let metrics = model.metrics();

        assert!(metrics.vpin >= 0.0 && metrics.vpin <= 1.0);
        assert!(metrics.toxicity_score >= 0.0);
        assert!(metrics.toxicity_level.is_some());
    }

    #[test]
    fn test_fill_recording() {
        let mut model = AdverseSelectionModel::default();

        let fill = FillInfo {
            price: 100.0,
            quantity: 10.0,
            is_maker: true,
            is_bid: true,
            mid_at_fill: 100.005,
            mid_after: None,
            timestamp: 1000000,
        };

        model.record_fill(fill);
        assert_eq!(model.fill_history.len(), 1);

        // Update with outcome
        model.update_fill_outcome(1000000, 99.99);

        let fill_ref = model.fill_history.back().unwrap();
        assert!(fill_ref.mid_after.is_some());
    }

    #[test]
    fn test_adverse_selection_calculation() {
        // Test fill with adverse selection
        let fill_adverse = FillInfo {
            price: 100.0,
            quantity: 10.0,
            is_maker: true,
            is_bid: true, // We bought
            mid_at_fill: 100.0,
            mid_after: Some(99.95), // Price dropped
            timestamp: 1000000,
        };

        let adverse = fill_adverse.adverse_selection().unwrap();
        assert!(
            adverse > 0.0,
            "Should show adverse selection when price moves against us"
        );

        // Test fill with favorable movement
        let fill_favorable = FillInfo {
            price: 100.0,
            quantity: 10.0,
            is_maker: true,
            is_bid: true, // We bought
            mid_at_fill: 100.0,
            mid_after: Some(100.05), // Price rose
            timestamp: 1000000,
        };

        let adverse = fill_favorable.adverse_selection().unwrap();
        assert!(
            adverse < 0.0,
            "Should show negative adverse selection when price moves for us"
        );
    }

    #[test]
    fn test_spread_adjustment() {
        let mut model = AdverseSelectionModel::default();

        let initial_adjustment = model.spread_adjustment();
        assert!(initial_adjustment >= 1.0);

        // Simulate toxic flow (all large buys)
        for i in 0..200 {
            model.on_trade(TradeEvent::new(
                100.0 + 0.05, // Aggressive
                50.0,         // Large
                true,
                i * 1000,
                100.0,
            ));
        }

        let toxic_adjustment = model.spread_adjustment();
        assert!(
            toxic_adjustment > initial_adjustment,
            "Spread should widen with toxic flow"
        );
    }

    #[test]
    fn test_toxicity_levels() {
        let mut model = AdverseSelectionModelBuilder::new()
            .high_toxicity_threshold(0.7)
            .medium_toxicity_threshold(0.4)
            .build();

        // Need enough trades to not be "Unknown"
        for _i in 0..100 {
            model.on_trade(create_test_trade(true, 1.0, 100.0));
        }

        let metrics = model.metrics();
        assert_ne!(metrics.toxicity_level, Some(ToxicityLevel::Unknown));
    }

    #[test]
    fn test_builder() {
        let model = AdverseSelectionModelBuilder::new()
            .vpin_buckets(30)
            .volume_bucket_size(50.0)
            .kyle_window(50)
            .ema_decay(0.9)
            .high_toxicity_threshold(0.6)
            .medium_toxicity_threshold(0.3)
            .build();

        assert_eq!(model.config.vpin_buckets, 30);
        assert_eq!(model.config.volume_bucket_size, 50.0);
        assert_eq!(model.config.kyle_window, 50);
        assert_eq!(model.config.ema_decay, 0.9);
    }

    #[test]
    fn test_presets() {
        let hft = AdverseSelectionPresets::hft_market_making();
        assert_eq!(hft.config.vpin_buckets, 30);

        let exec = AdverseSelectionPresets::execution();
        assert_eq!(exec.config.vpin_buckets, 100);

        let conservative = AdverseSelectionPresets::conservative();
        assert_eq!(conservative.config.high_toxicity_threshold, 0.5);
    }

    #[test]
    fn test_reset() {
        let mut model = AdverseSelectionModel::default();

        // Add some data
        for _i in 0..50 {
            model.on_trade(create_test_trade(true, 10.0, 100.0));
        }

        assert!(model.total_trades > 0);

        model.reset();

        assert_eq!(model.total_trades, 0);
        assert_eq!(model.total_volume, 0.0);
        assert!(model.trade_history.is_empty());
    }

    #[test]
    fn test_stats() {
        let mut model = AdverseSelectionModel::default();

        model.on_trade(create_test_trade(true, 10.0, 100.01));
        model.on_trade(create_test_trade(false, 5.0, 99.99));

        let stats = model.stats();
        assert_eq!(stats.total_trades, 2);
        assert_eq!(stats.total_volume, 15.0);
    }

    #[test]
    fn test_basic() {
        let model = AdverseSelectionModel::default();
        assert!(model.process().is_ok());
    }
}
