//! Fill Probability Model
//!
//! Part of the Cerebellum region
//! Component: Market microstructure modeling
//!
//! This module predicts the probability that a limit order will be filled
//! within a given time horizon. This is crucial for:
//!
//! - Optimal order placement
//! - Execution quality improvement
//! - Queue position management
//! - Passive vs aggressive trade-off decisions
//!
//! The model considers multiple factors:
//! - Distance from mid-price (in ticks)
//! - Order book imbalance
//! - Recent trade flow
//! - Volatility regime
//! - Time of day effects
//! - Queue position

use crate::common::Result;
use std::collections::VecDeque;

/// Configuration for the fill probability model
#[derive(Debug, Clone)]
pub struct FillProbabilityConfig {
    /// Number of price levels to consider for book imbalance
    pub book_depth_levels: usize,
    /// Time horizons for fill probability (in seconds)
    pub time_horizons: Vec<f64>,
    /// Window size for historical fill rate calculation
    pub history_window: usize,
    /// Decay factor for exponential moving average
    pub ema_decay: f64,
    /// Tick size for the instrument
    pub tick_size: f64,
    /// Number of features for the model
    pub num_features: usize,
}

impl Default for FillProbabilityConfig {
    fn default() -> Self {
        Self {
            book_depth_levels: 10,
            time_horizons: vec![1.0, 5.0, 10.0, 30.0, 60.0],
            history_window: 1000,
            ema_decay: 0.99,
            tick_size: 0.01,
            num_features: 16,
        }
    }
}

/// Order book state snapshot
#[derive(Debug, Clone)]
pub struct OrderBookSnapshot {
    /// Bid prices (sorted descending)
    pub bid_prices: Vec<f64>,
    /// Bid quantities at each level
    pub bid_quantities: Vec<f64>,
    /// Ask prices (sorted ascending)
    pub ask_prices: Vec<f64>,
    /// Ask quantities at each level
    pub ask_quantities: Vec<f64>,
    /// Timestamp (microseconds)
    pub timestamp: i64,
}

impl OrderBookSnapshot {
    /// Create a new order book snapshot
    pub fn new(
        bid_prices: Vec<f64>,
        bid_quantities: Vec<f64>,
        ask_prices: Vec<f64>,
        ask_quantities: Vec<f64>,
        timestamp: i64,
    ) -> Self {
        Self {
            bid_prices,
            bid_quantities,
            ask_prices,
            ask_quantities,
            timestamp,
        }
    }

    /// Get the mid price
    pub fn mid_price(&self) -> Option<f64> {
        if self.bid_prices.is_empty() || self.ask_prices.is_empty() {
            return None;
        }
        Some((self.bid_prices[0] + self.ask_prices[0]) / 2.0)
    }

    /// Get the spread
    pub fn spread(&self) -> Option<f64> {
        if self.bid_prices.is_empty() || self.ask_prices.is_empty() {
            return None;
        }
        Some(self.ask_prices[0] - self.bid_prices[0])
    }

    /// Calculate book imbalance at top level
    pub fn top_imbalance(&self) -> f64 {
        if self.bid_quantities.is_empty() || self.ask_quantities.is_empty() {
            return 0.0;
        }
        let bid_qty = self.bid_quantities[0];
        let ask_qty = self.ask_quantities[0];
        let total = bid_qty + ask_qty;
        if total == 0.0 {
            return 0.0;
        }
        (bid_qty - ask_qty) / total
    }

    /// Calculate weighted book imbalance across levels
    pub fn weighted_imbalance(&self, levels: usize) -> f64 {
        let levels = levels.min(self.bid_prices.len()).min(self.ask_prices.len());
        if levels == 0 {
            return 0.0;
        }

        let mut weighted_bid = 0.0;
        let mut weighted_ask = 0.0;

        for i in 0..levels {
            let weight = 1.0 / (i + 1) as f64; // Inverse distance weighting
            weighted_bid += self.bid_quantities.get(i).unwrap_or(&0.0) * weight;
            weighted_ask += self.ask_quantities.get(i).unwrap_or(&0.0) * weight;
        }

        let total = weighted_bid + weighted_ask;
        if total == 0.0 {
            return 0.0;
        }
        (weighted_bid - weighted_ask) / total
    }

    /// Get total bid quantity up to a number of levels
    pub fn total_bid_quantity(&self, levels: usize) -> f64 {
        self.bid_quantities.iter().take(levels).sum()
    }

    /// Get total ask quantity up to a number of levels
    pub fn total_ask_quantity(&self, levels: usize) -> f64 {
        self.ask_quantities.iter().take(levels).sum()
    }
}

/// Recent trade information
#[derive(Debug, Clone)]
pub struct RecentTrade {
    /// Price of the trade
    pub price: f64,
    /// Quantity traded
    pub quantity: f64,
    /// True if buyer was the taker (aggressive buy)
    pub is_buy: bool,
    /// Timestamp (microseconds)
    pub timestamp: i64,
}

/// Historical fill record
#[derive(Debug, Clone)]
pub struct FillRecord {
    /// Distance from mid in ticks when order was placed
    pub distance_ticks: f64,
    /// Time to fill (seconds), None if not filled
    pub time_to_fill: Option<f64>,
    /// Book imbalance when order was placed
    pub imbalance: f64,
    /// Whether order was on bid side
    pub is_bid: bool,
    /// Timestamp when order was placed
    pub placed_at: i64,
}

/// Features for fill probability prediction
#[derive(Debug, Clone, Default)]
pub struct FillFeatures {
    /// Distance from mid-price in ticks (negative = inside, positive = outside)
    pub distance_ticks: f64,
    /// Order book imbalance at top level (-1 to 1)
    pub top_imbalance: f64,
    /// Weighted order book imbalance
    pub weighted_imbalance: f64,
    /// Spread in ticks
    pub spread_ticks: f64,
    /// Recent trade flow imbalance (buy - sell volume)
    pub trade_flow_imbalance: f64,
    /// Estimated queue position (0-1, 0 = front)
    pub queue_position: f64,
    /// Recent volatility (price std dev)
    pub volatility: f64,
    /// Bid quantity at our level
    pub same_side_quantity: f64,
    /// Ask quantity at our level
    pub opposite_side_quantity: f64,
    /// Time since last trade (seconds)
    pub time_since_last_trade: f64,
    /// Historical fill rate at this distance
    pub historical_fill_rate: f64,
    /// Is this a bid order
    pub is_bid: bool,
}

impl FillFeatures {
    /// Convert to feature vector for model input
    pub fn to_vec(&self) -> Vec<f32> {
        vec![
            self.distance_ticks as f32,
            self.top_imbalance as f32,
            self.weighted_imbalance as f32,
            self.spread_ticks as f32,
            self.trade_flow_imbalance as f32,
            self.queue_position as f32,
            self.volatility as f32,
            self.same_side_quantity as f32,
            self.opposite_side_quantity as f32,
            self.time_since_last_trade as f32,
            self.historical_fill_rate as f32,
            if self.is_bid { 1.0 } else { -1.0 },
        ]
    }
}

/// Fill probability model
#[derive(Debug)]
pub struct FillProbabilityModel {
    /// Configuration
    pub config: FillProbabilityConfig,
    /// Historical fill records
    fill_history: VecDeque<FillRecord>,
    /// Recent trades
    recent_trades: VecDeque<RecentTrade>,
    /// Fill rates by distance (tick distance -> fill count, total count)
    fill_rates_by_distance: Vec<(u64, u64)>,
    /// EMA of trade flow
    trade_flow_ema: f64,
    /// EMA of volatility
    volatility_ema: f64,
    /// Last trade timestamp
    last_trade_time: i64,
    /// Model weights (simple logistic regression)
    weights: Vec<f64>,
    /// Bias term
    bias: f64,
}

impl Default for FillProbabilityModel {
    fn default() -> Self {
        Self::new(FillProbabilityConfig::default())
    }
}

impl FillProbabilityModel {
    /// Create a new fill probability model
    pub fn new(config: FillProbabilityConfig) -> Self {
        // Initialize with reasonable default weights
        // These would typically be learned from historical data
        let weights = vec![
            -0.5,  // distance_ticks (further = lower prob)
            0.3,   // top_imbalance (favorable imbalance = higher prob)
            0.2,   // weighted_imbalance
            -0.1,  // spread_ticks (wider spread = lower prob)
            0.2,   // trade_flow_imbalance
            -0.4,  // queue_position (back of queue = lower prob)
            0.1,   // volatility (more vol = higher prob for limits)
            -0.05, // same_side_quantity
            0.05,  // opposite_side_quantity
            -0.1,  // time_since_last_trade
            0.5,   // historical_fill_rate
            0.0,   // is_bid (neutral by default)
        ];

        Self {
            fill_rates_by_distance: vec![(0, 0); 100], // Support up to 100 ticks
            config,
            fill_history: VecDeque::with_capacity(1000),
            recent_trades: VecDeque::with_capacity(100),
            trade_flow_ema: 0.0,
            volatility_ema: 0.0,
            last_trade_time: 0,
            weights,
            bias: 0.0,
        }
    }

    /// Update model with a new trade
    pub fn on_trade(&mut self, trade: RecentTrade) {
        // Update trade flow EMA
        let flow = if trade.is_buy {
            trade.quantity
        } else {
            -trade.quantity
        };
        self.trade_flow_ema =
            self.config.ema_decay * self.trade_flow_ema + (1.0 - self.config.ema_decay) * flow;

        // Update volatility if we have recent trades
        if let Some(last_trade) = self.recent_trades.back() {
            let price_change = (trade.price - last_trade.price).abs();
            self.volatility_ema = self.config.ema_decay * self.volatility_ema
                + (1.0 - self.config.ema_decay) * price_change;
        }

        self.last_trade_time = trade.timestamp;

        // Add to recent trades
        self.recent_trades.push_back(trade);
        if self.recent_trades.len() > 100 {
            self.recent_trades.pop_front();
        }
    }

    /// Record a fill outcome for model learning
    pub fn record_fill_outcome(&mut self, record: FillRecord) {
        // Update fill rate statistics
        let tick_idx = record.distance_ticks.abs() as usize;
        if tick_idx < self.fill_rates_by_distance.len() {
            self.fill_rates_by_distance[tick_idx].1 += 1; // Total count
            if record.time_to_fill.is_some() {
                self.fill_rates_by_distance[tick_idx].0 += 1; // Fill count
            }
        }

        // Add to history
        self.fill_history.push_back(record);
        if self.fill_history.len() > self.config.history_window {
            self.fill_history.pop_front();
        }
    }

    /// Get historical fill rate at a given distance
    pub fn historical_fill_rate(&self, distance_ticks: f64) -> f64 {
        let tick_idx = distance_ticks.abs() as usize;
        if tick_idx >= self.fill_rates_by_distance.len() {
            return 0.0;
        }

        let (fills, total) = self.fill_rates_by_distance[tick_idx];
        if total == 0 {
            // Use a prior based on distance
            // Roughly: at-the-touch has ~50% fill rate, decreasing with distance
            return 0.5 * (-0.1 * distance_ticks.abs()).exp();
        }

        fills as f64 / total as f64
    }

    /// Extract features for fill probability prediction
    pub fn extract_features(
        &self,
        order_price: f64,
        is_bid: bool,
        order_book: &OrderBookSnapshot,
        current_time: i64,
    ) -> FillFeatures {
        let mid = order_book.mid_price().unwrap_or(order_price);
        let spread = order_book.spread().unwrap_or(self.config.tick_size);

        // Calculate distance in ticks
        let distance = if is_bid {
            (mid - order_price) / self.config.tick_size
        } else {
            (order_price - mid) / self.config.tick_size
        };

        // Estimate queue position
        let queue_position = self.estimate_queue_position(order_price, is_bid, order_book);

        // Get quantities at our price level
        let (same_side_qty, opposite_side_qty) =
            self.get_quantities_at_price(order_price, is_bid, order_book);

        // Time since last trade
        let time_since_trade = if self.last_trade_time > 0 {
            (current_time - self.last_trade_time) as f64 / 1_000_000.0 // Convert from micros to secs
        } else {
            0.0
        };

        FillFeatures {
            distance_ticks: distance,
            top_imbalance: order_book.top_imbalance(),
            weighted_imbalance: order_book.weighted_imbalance(self.config.book_depth_levels),
            spread_ticks: spread / self.config.tick_size,
            trade_flow_imbalance: self.trade_flow_ema,
            queue_position,
            volatility: self.volatility_ema,
            same_side_quantity: same_side_qty,
            opposite_side_quantity: opposite_side_qty,
            time_since_last_trade: time_since_trade,
            historical_fill_rate: self.historical_fill_rate(distance),
            is_bid,
        }
    }

    /// Estimate queue position (0 = front, 1 = back)
    fn estimate_queue_position(
        &self,
        order_price: f64,
        is_bid: bool,
        order_book: &OrderBookSnapshot,
    ) -> f64 {
        // Find the price level
        let (prices, _quantities) = if is_bid {
            (&order_book.bid_prices, &order_book.bid_quantities)
        } else {
            (&order_book.ask_prices, &order_book.ask_quantities)
        };

        // Find matching price level
        for (_i, &price) in prices.iter().enumerate() {
            if (price - order_price).abs() < self.config.tick_size / 2.0 {
                // Found the level - assume we're at the back of existing queue
                return 1.0; // Conservative: assume back of queue for new orders
            }
            if (is_bid && price < order_price) || (!is_bid && price > order_price) {
                // Our price is better than this level
                return 0.0; // We'd be at front of a new level
            }
        }

        // Price is worse than all visible levels
        1.0
    }

    /// Get quantities at a specific price level
    fn get_quantities_at_price(
        &self,
        order_price: f64,
        is_bid: bool,
        order_book: &OrderBookSnapshot,
    ) -> (f64, f64) {
        let same_side_prices = if is_bid {
            &order_book.bid_prices
        } else {
            &order_book.ask_prices
        };
        let same_side_qtys = if is_bid {
            &order_book.bid_quantities
        } else {
            &order_book.ask_quantities
        };
        let opposite_side_qtys = if is_bid {
            &order_book.ask_quantities
        } else {
            &order_book.bid_quantities
        };

        let mut same_qty = 0.0;

        // Find same side quantity
        for (i, &price) in same_side_prices.iter().enumerate() {
            if (price - order_price).abs() < self.config.tick_size / 2.0 {
                same_qty = *same_side_qtys.get(i).unwrap_or(&0.0);
                break;
            }
        }

        // Get top of opposite side
        let opposite_qty = *opposite_side_qtys.first().unwrap_or(&0.0);

        (same_qty, opposite_qty)
    }

    /// Sigmoid function
    fn sigmoid(x: f64) -> f64 {
        1.0 / (1.0 + (-x).exp())
    }

    /// Predict fill probability
    pub fn predict(&self, features: &FillFeatures) -> f64 {
        let feature_vec = features.to_vec();

        // Linear combination
        let mut logit = self.bias;
        for (i, &w) in self.weights.iter().enumerate() {
            if i < feature_vec.len() {
                logit += w * feature_vec[i] as f64;
            }
        }

        // Apply sigmoid
        Self::sigmoid(logit)
    }

    /// Predict fill probability for multiple time horizons
    pub fn predict_by_horizon(&self, features: &FillFeatures) -> Vec<(f64, f64)> {
        let base_prob = self.predict(features);

        // Scale probability by time horizon
        // Longer horizons have higher fill probability
        self.config
            .time_horizons
            .iter()
            .map(|&horizon| {
                // Simple scaling: prob increases with sqrt of time
                let time_factor = (horizon / self.config.time_horizons[0]).sqrt();
                let scaled_prob = (base_prob * time_factor).min(1.0);
                (horizon, scaled_prob)
            })
            .collect()
    }

    /// Get expected time to fill (in seconds)
    pub fn expected_time_to_fill(&self, features: &FillFeatures) -> Option<f64> {
        let prob = self.predict(features);

        if prob < 0.01 {
            return None; // Unlikely to fill
        }

        // Use historical data to estimate time to fill
        let base_time = self.historical_time_to_fill(features.distance_ticks);

        // Adjust based on current conditions
        let imbalance_factor = if features.is_bid {
            1.0 - features.top_imbalance * 0.2
        } else {
            1.0 + features.top_imbalance * 0.2
        };

        let volatility_factor = 1.0 / (1.0 + features.volatility);
        let queue_factor = 1.0 + features.queue_position;

        Some(base_time * imbalance_factor * volatility_factor * queue_factor)
    }

    /// Get historical average time to fill at a given distance
    fn historical_time_to_fill(&self, distance_ticks: f64) -> f64 {
        // Simple heuristic: time increases with distance squared
        let base_time = 5.0; // 5 seconds base for at-the-touch
        base_time * (1.0 + distance_ticks.abs().powi(2) * 0.1)
    }

    /// Update model weights using simple online learning
    pub fn update_weights(
        &mut self,
        features: &FillFeatures,
        actual_filled: bool,
        learning_rate: f64,
    ) {
        let predicted = self.predict(features);
        let target = if actual_filled { 1.0 } else { 0.0 };
        let error = target - predicted;

        let feature_vec = features.to_vec();
        let gradient_scale = error * predicted * (1.0 - predicted); // Sigmoid derivative

        // Update weights
        for (i, weight) in self.weights.iter_mut().enumerate() {
            if i < feature_vec.len() {
                *weight += learning_rate * gradient_scale * feature_vec[i] as f64;
            }
        }
        self.bias += learning_rate * gradient_scale;
    }

    /// Get model statistics
    pub fn stats(&self) -> FillProbabilityStats {
        let total_fills: u64 = self.fill_rates_by_distance.iter().map(|(f, _)| f).sum();
        let total_orders: u64 = self.fill_rates_by_distance.iter().map(|(_, t)| t).sum();

        FillProbabilityStats {
            total_orders,
            total_fills,
            overall_fill_rate: if total_orders > 0 {
                total_fills as f64 / total_orders as f64
            } else {
                0.0
            },
            history_size: self.fill_history.len(),
            recent_trades_count: self.recent_trades.len(),
            trade_flow_ema: self.trade_flow_ema,
            volatility_ema: self.volatility_ema,
        }
    }

    /// Main processing function (for interface compatibility)
    pub fn process(&self) -> Result<()> {
        Ok(())
    }
}

/// Statistics about the fill probability model
#[derive(Debug, Clone)]
pub struct FillProbabilityStats {
    pub total_orders: u64,
    pub total_fills: u64,
    pub overall_fill_rate: f64,
    pub history_size: usize,
    pub recent_trades_count: usize,
    pub trade_flow_ema: f64,
    pub volatility_ema: f64,
}

/// Builder for FillProbabilityModel
#[derive(Debug, Clone)]
pub struct FillProbabilityModelBuilder {
    config: FillProbabilityConfig,
}

impl FillProbabilityModelBuilder {
    pub fn new() -> Self {
        Self {
            config: FillProbabilityConfig::default(),
        }
    }

    pub fn book_depth_levels(mut self, levels: usize) -> Self {
        self.config.book_depth_levels = levels;
        self
    }

    pub fn time_horizons(mut self, horizons: Vec<f64>) -> Self {
        self.config.time_horizons = horizons;
        self
    }

    pub fn history_window(mut self, window: usize) -> Self {
        self.config.history_window = window;
        self
    }

    pub fn ema_decay(mut self, decay: f64) -> Self {
        self.config.ema_decay = decay;
        self
    }

    pub fn tick_size(mut self, tick: f64) -> Self {
        self.config.tick_size = tick;
        self
    }

    pub fn build(self) -> FillProbabilityModel {
        FillProbabilityModel::new(self.config)
    }
}

impl Default for FillProbabilityModelBuilder {
    fn default() -> Self {
        Self::new()
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    fn create_test_order_book() -> OrderBookSnapshot {
        OrderBookSnapshot::new(
            vec![100.0, 99.99, 99.98, 99.97, 99.96],
            vec![10.0, 20.0, 30.0, 40.0, 50.0],
            vec![100.01, 100.02, 100.03, 100.04, 100.05],
            vec![15.0, 25.0, 35.0, 45.0, 55.0],
            1000000,
        )
    }

    #[test]
    fn test_order_book_snapshot() {
        let book = create_test_order_book();

        assert!((book.mid_price().unwrap() - 100.005).abs() < 0.001);
        assert!((book.spread().unwrap() - 0.01).abs() < 0.001);

        // Imbalance: (10 - 15) / (10 + 15) = -0.2
        assert!((book.top_imbalance() - (-0.2)).abs() < 0.001);
    }

    #[test]
    fn test_fill_probability_model_creation() {
        let model = FillProbabilityModel::default();
        assert_eq!(model.config.book_depth_levels, 10);
        assert_eq!(model.config.time_horizons.len(), 5);
    }

    #[test]
    fn test_feature_extraction() {
        let model = FillProbabilityModelBuilder::new().tick_size(0.01).build();

        let book = create_test_order_book();

        // Order at best bid
        let features = model.extract_features(100.0, true, &book, 1000000);

        assert!(features.distance_ticks.abs() < 1.0);
        assert!(features.top_imbalance.abs() < 1.0);
    }

    #[test]
    fn test_fill_prediction() {
        let model = FillProbabilityModel::default();
        let book = create_test_order_book();

        // At-the-touch bid
        let features_touch = model.extract_features(100.0, true, &book, 1000000);
        let prob_touch = model.predict(&features_touch);

        // Far from touch bid
        let features_far = model.extract_features(99.90, true, &book, 1000000);
        let prob_far = model.predict(&features_far);

        // Probability should decrease with distance
        assert!(prob_touch > prob_far, "Touch should have higher fill prob");
    }

    #[test]
    fn test_on_trade() {
        let mut model = FillProbabilityModel::default();

        // Add some trades
        model.on_trade(RecentTrade {
            price: 100.0,
            quantity: 10.0,
            is_buy: true,
            timestamp: 1000000,
        });

        model.on_trade(RecentTrade {
            price: 100.01,
            quantity: 5.0,
            is_buy: false,
            timestamp: 1000100,
        });

        assert_eq!(model.recent_trades.len(), 2);
        assert!(model.trade_flow_ema != 0.0);
    }

    #[test]
    fn test_record_fill_outcome() {
        let mut model = FillProbabilityModel::default();

        // Record some fill outcomes
        model.record_fill_outcome(FillRecord {
            distance_ticks: 0.0,
            time_to_fill: Some(1.5),
            imbalance: 0.0,
            is_bid: true,
            placed_at: 1000000,
        });

        model.record_fill_outcome(FillRecord {
            distance_ticks: 0.0,
            time_to_fill: None,
            imbalance: 0.0,
            is_bid: true,
            placed_at: 1000100,
        });

        assert_eq!(model.fill_history.len(), 2);

        let fill_rate = model.historical_fill_rate(0.0);
        assert!((fill_rate - 0.5).abs() < 0.001);
    }

    #[test]
    fn test_predict_by_horizon() {
        let model = FillProbabilityModel::default();

        let features = FillFeatures {
            distance_ticks: 1.0,
            ..Default::default()
        };

        let predictions = model.predict_by_horizon(&features);

        assert_eq!(predictions.len(), 5);

        // Longer horizons should have higher probability
        for i in 1..predictions.len() {
            assert!(
                predictions[i].1 >= predictions[i - 1].1,
                "Fill prob should increase with horizon"
            );
        }
    }

    #[test]
    fn test_weight_update() {
        let mut model = FillProbabilityModel::default();

        let features = FillFeatures {
            distance_ticks: 0.0,
            historical_fill_rate: 0.5,
            ..Default::default()
        };

        let initial_pred = model.predict(&features);

        // Train towards filled
        for _ in 0..100 {
            model.update_weights(&features, true, 0.1);
        }

        let final_pred = model.predict(&features);

        assert!(
            final_pred > initial_pred,
            "Prediction should increase after training on positive examples"
        );
    }

    #[test]
    fn test_expected_time_to_fill() {
        let model = FillProbabilityModel::default();

        let features_close = FillFeatures {
            distance_ticks: 0.0,
            historical_fill_rate: 0.8,
            ..Default::default()
        };

        let features_far = FillFeatures {
            distance_ticks: 5.0,
            historical_fill_rate: 0.2,
            ..Default::default()
        };

        let time_close = model.expected_time_to_fill(&features_close);
        let time_far = model.expected_time_to_fill(&features_far);

        assert!(time_close.is_some());
        assert!(time_far.is_some());
        assert!(
            time_far.unwrap() > time_close.unwrap(),
            "Far orders should take longer to fill"
        );
    }

    #[test]
    fn test_stats() {
        let mut model = FillProbabilityModel::default();

        model.record_fill_outcome(FillRecord {
            distance_ticks: 0.0,
            time_to_fill: Some(1.0),
            imbalance: 0.0,
            is_bid: true,
            placed_at: 1000000,
        });

        let stats = model.stats();
        assert_eq!(stats.history_size, 1);
        assert!(stats.total_orders > 0);
    }

    #[test]
    fn test_builder() {
        let model = FillProbabilityModelBuilder::new()
            .book_depth_levels(5)
            .tick_size(0.001)
            .ema_decay(0.95)
            .history_window(500)
            .time_horizons(vec![1.0, 5.0, 10.0])
            .build();

        assert_eq!(model.config.book_depth_levels, 5);
        assert_eq!(model.config.tick_size, 0.001);
        assert_eq!(model.config.ema_decay, 0.95);
    }

    #[test]
    fn test_basic() {
        let model = FillProbabilityModel::default();
        assert!(model.process().is_ok());
    }
}
