//! Best Execution Analyzer
//!
//! This module provides intelligent execution timing analysis for Kraken orders
//! based on cross-exchange price data from Bybit and Binance.
//!
//! # Overview
//!
//! Since Kraken is the only tradeable exchange (Canada), this analyzer helps
//! determine the optimal timing for order execution by:
//! - Monitoring price deviations across exchanges
//! - Identifying favorable entry/exit points on Kraken
//! - Providing execution recommendations with confidence scores
//! - Tracking historical execution quality
//!
//! # Features
//!
//! - Real-time execution scoring
//! - Price impact estimation
//! - Optimal timing recommendations
//! - Execution quality metrics
//! - Slippage prediction
//!
//! # Example
//!
//! ```rust,ignore
//! use janus_execution::execution::best_execution::{BestExecutionAnalyzer, ExecutionConfig};
//!
//! let config = ExecutionConfig::default();
//! let analyzer = BestExecutionAnalyzer::new(config);
//!
//! // Check if now is a good time to buy on Kraken
//! let analysis = analyzer.analyze_buy("BTC/USDT", quantity).await;
//! println!("Execution score: {}, recommendation: {:?}",
//!          analysis.score, analysis.recommendation);
//! ```

use crate::execution::arbitrage::{Exchange, ExchangePrice, PriceComparison};
use crate::execution::histogram::global_latency_histograms;
use crate::execution::metrics::{RecommendationType, best_execution_metrics};
use chrono::{DateTime, Utc};
use rust_decimal::Decimal;
use rust_decimal::prelude::*;
use serde::{Deserialize, Serialize};
use std::collections::{HashMap, VecDeque};
use std::sync::Arc;
use std::time::Instant;
use tokio::sync::RwLock;

// ============================================================================
// Configuration
// ============================================================================

/// Best execution analyzer configuration
#[derive(Debug, Clone)]
pub struct ExecutionConfig {
    /// Minimum score to recommend execution (0-100)
    pub min_execution_score: Decimal,
    /// Maximum acceptable slippage in basis points
    pub max_slippage_bps: Decimal,
    /// Price history window for analysis
    pub analysis_window: std::time::Duration,
    /// Number of historical data points to keep
    pub history_size: usize,
    /// Weight for price deviation in score calculation
    pub price_deviation_weight: Decimal,
    /// Weight for spread in score calculation
    pub spread_weight: Decimal,
    /// Weight for volatility in score calculation
    pub volatility_weight: Decimal,
    /// Weight for trend in score calculation
    pub trend_weight: Decimal,
    /// Kraken fee rate in basis points
    pub kraken_fee_bps: Decimal,
}

impl Default for ExecutionConfig {
    fn default() -> Self {
        Self {
            min_execution_score: Decimal::from(60),
            max_slippage_bps: Decimal::from(50),
            analysis_window: std::time::Duration::from_secs(300), // 5 minutes
            history_size: 1000,
            price_deviation_weight: Decimal::new(40, 0),
            spread_weight: Decimal::new(25, 0),
            volatility_weight: Decimal::new(20, 0),
            trend_weight: Decimal::new(15, 0),
            kraken_fee_bps: Decimal::from(26), // Kraken taker fee
        }
    }
}

impl ExecutionConfig {
    /// Create config from environment variables
    pub fn from_env() -> Self {
        let min_execution_score = std::env::var("EXEC_MIN_SCORE")
            .ok()
            .and_then(|s| Decimal::from_str(&s).ok())
            .unwrap_or(Decimal::from(60));

        let max_slippage_bps = std::env::var("EXEC_MAX_SLIPPAGE_BPS")
            .ok()
            .and_then(|s| Decimal::from_str(&s).ok())
            .unwrap_or(Decimal::from(50));

        let kraken_fee_bps = std::env::var("KRAKEN_FEE_BPS")
            .ok()
            .and_then(|s| Decimal::from_str(&s).ok())
            .unwrap_or(Decimal::from(26));

        Self {
            min_execution_score,
            max_slippage_bps,
            kraken_fee_bps,
            ..Default::default()
        }
    }
}

// ============================================================================
// Execution Types
// ============================================================================

/// Side of the order
#[derive(Debug, Clone, Copy, PartialEq, Eq, Serialize, Deserialize)]
pub enum OrderSide {
    Buy,
    Sell,
}

impl std::fmt::Display for OrderSide {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        match self {
            OrderSide::Buy => write!(f, "BUY"),
            OrderSide::Sell => write!(f, "SELL"),
        }
    }
}

/// Execution timing recommendation
#[derive(Debug, Clone, Copy, PartialEq, Eq, Serialize, Deserialize)]
pub enum ExecutionRecommendation {
    /// Execute immediately - conditions are favorable
    ExecuteNow,
    /// Wait for better conditions
    Wait,
    /// Conditions are acceptable but not optimal
    Acceptable,
    /// Use limit order instead of market
    UseLimitOrder,
    /// Avoid execution - conditions are unfavorable
    Avoid,
}

impl std::fmt::Display for ExecutionRecommendation {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        match self {
            ExecutionRecommendation::ExecuteNow => write!(f, "EXECUTE_NOW"),
            ExecutionRecommendation::Wait => write!(f, "WAIT"),
            ExecutionRecommendation::Acceptable => write!(f, "ACCEPTABLE"),
            ExecutionRecommendation::UseLimitOrder => write!(f, "USE_LIMIT_ORDER"),
            ExecutionRecommendation::Avoid => write!(f, "AVOID"),
        }
    }
}

/// Price trend direction
#[derive(Debug, Clone, Copy, PartialEq, Eq, Serialize, Deserialize)]
pub enum PriceTrend {
    /// Price is increasing
    Up,
    /// Price is decreasing
    Down,
    /// Price is stable
    Stable,
}

/// Execution analysis result
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct ExecutionAnalysis {
    /// Trading symbol
    pub symbol: String,
    /// Order side
    pub side: OrderSide,
    /// Requested quantity
    pub quantity: Decimal,
    /// Overall execution score (0-100)
    pub score: Decimal,
    /// Execution recommendation
    pub recommendation: ExecutionRecommendation,
    /// Kraken current price (bid for sell, ask for buy)
    pub kraken_price: Decimal,
    /// Market average price
    pub market_avg_price: Decimal,
    /// Kraken deviation from market (%)
    pub kraken_deviation_pct: Decimal,
    /// Current Kraken spread (bps)
    pub kraken_spread_bps: Decimal,
    /// Estimated slippage (bps)
    pub estimated_slippage_bps: Decimal,
    /// Estimated total cost including fees (bps)
    pub estimated_cost_bps: Decimal,
    /// Price volatility (standard deviation as %)
    pub volatility_pct: Decimal,
    /// Current price trend
    pub trend: PriceTrend,
    /// Confidence level (0-1)
    pub confidence: Decimal,
    /// Suggested limit price (if using limit order)
    pub suggested_limit_price: Option<Decimal>,
    /// Reasons for recommendation
    pub reasons: Vec<String>,
    /// Analysis timestamp
    pub timestamp: DateTime<Utc>,
    /// Data freshness (oldest price age in ms)
    pub data_age_ms: i64,
}

impl ExecutionAnalysis {
    /// Check if execution is recommended
    pub fn should_execute(&self) -> bool {
        matches!(
            self.recommendation,
            ExecutionRecommendation::ExecuteNow | ExecutionRecommendation::Acceptable
        )
    }

    /// Get expected fill price
    pub fn expected_fill_price(&self) -> Decimal {
        let slippage_factor = self.estimated_slippage_bps / Decimal::from(10000);
        match self.side {
            OrderSide::Buy => self.kraken_price * (Decimal::ONE + slippage_factor),
            OrderSide::Sell => self.kraken_price * (Decimal::ONE - slippage_factor),
        }
    }

    /// Get expected total cost in quote currency
    pub fn expected_total_cost(&self) -> Decimal {
        self.expected_fill_price() * self.quantity
    }
}

/// Historical execution record
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct ExecutionRecord {
    /// Symbol
    pub symbol: String,
    /// Side
    pub side: OrderSide,
    /// Quantity
    pub quantity: Decimal,
    /// Analysis score at time of execution
    pub analysis_score: Decimal,
    /// Expected fill price
    pub expected_price: Decimal,
    /// Actual fill price
    pub actual_price: Decimal,
    /// Expected slippage (bps)
    pub expected_slippage_bps: Decimal,
    /// Actual slippage (bps)
    pub actual_slippage_bps: Decimal,
    /// Execution timestamp
    pub timestamp: DateTime<Utc>,
}

impl ExecutionRecord {
    /// Calculate execution quality score
    pub fn quality_score(&self) -> Decimal {
        // Score based on how actual slippage compares to expected
        if self.expected_slippage_bps == Decimal::ZERO {
            if self.actual_slippage_bps <= Decimal::ZERO {
                return Decimal::from(100);
            }
            return Decimal::from(50);
        }

        let ratio = self.actual_slippage_bps / self.expected_slippage_bps;
        if ratio <= Decimal::ONE {
            // Better than expected
            Decimal::from(100)
        } else if ratio <= Decimal::from(2) {
            // Up to 2x expected - acceptable
            Decimal::from(100) - (ratio - Decimal::ONE) * Decimal::from(50)
        } else {
            // Worse than 2x expected
            Decimal::ZERO.max(Decimal::from(50) - (ratio - Decimal::from(2)) * Decimal::from(25))
        }
    }
}

/// Execution quality metrics
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct ExecutionQualityMetrics {
    /// Symbol
    pub symbol: String,
    /// Total executions tracked
    pub total_executions: usize,
    /// Average quality score
    pub avg_quality_score: Decimal,
    /// Average slippage (bps)
    pub avg_slippage_bps: Decimal,
    /// Max slippage (bps)
    pub max_slippage_bps: Decimal,
    /// Times slippage exceeded expected
    pub slippage_exceeded_count: usize,
    /// Average analysis score for successful executions
    pub avg_analysis_score: Decimal,
    /// Recommendation accuracy (% of good recommendations)
    pub recommendation_accuracy: Decimal,
}

// ============================================================================
// Price History
// ============================================================================

/// Price history entry
#[derive(Debug, Clone)]
#[allow(dead_code)]
struct PriceHistoryEntry {
    timestamp: DateTime<Utc>,
    kraken_mid: Decimal,
    market_avg_mid: Decimal,
    kraken_spread_bps: Decimal,
}

/// Price history tracker
struct PriceHistory {
    entries: VecDeque<PriceHistoryEntry>,
    max_size: usize,
}

impl PriceHistory {
    fn new(max_size: usize) -> Self {
        Self {
            entries: VecDeque::with_capacity(max_size),
            max_size,
        }
    }

    fn add(&mut self, entry: PriceHistoryEntry) {
        if self.entries.len() >= self.max_size {
            self.entries.pop_front();
        }
        self.entries.push_back(entry);
    }

    fn get_volatility(&self) -> Option<Decimal> {
        if self.entries.len() < 2 {
            return None;
        }

        let prices: Vec<Decimal> = self.entries.iter().map(|e| e.kraken_mid).collect();
        let avg: Decimal = prices.iter().sum::<Decimal>() / Decimal::from(prices.len());

        if avg == Decimal::ZERO {
            return None;
        }

        let variance: Decimal = prices
            .iter()
            .map(|p| {
                let diff = *p - avg;
                diff * diff
            })
            .sum::<Decimal>()
            / Decimal::from(prices.len());

        // Standard deviation as percentage of average
        let std_dev = variance
            .to_f64()
            .map(|v| v.sqrt())
            .and_then(Decimal::from_f64)?;

        Some(std_dev / avg * Decimal::from(100))
    }

    fn get_trend(&self) -> PriceTrend {
        if self.entries.len() < 10 {
            return PriceTrend::Stable;
        }

        let recent: Vec<_> = self.entries.iter().rev().take(10).collect();
        let oldest_price = recent.last().map(|e| e.kraken_mid).unwrap_or(Decimal::ZERO);
        let newest_price = recent
            .first()
            .map(|e| e.kraken_mid)
            .unwrap_or(Decimal::ZERO);

        if oldest_price == Decimal::ZERO {
            return PriceTrend::Stable;
        }

        let change_pct = (newest_price - oldest_price) / oldest_price * Decimal::from(100);

        if change_pct > Decimal::new(5, 2) {
            // > 0.05%
            PriceTrend::Up
        } else if change_pct < Decimal::new(-5, 2) {
            // < -0.05%
            PriceTrend::Down
        } else {
            PriceTrend::Stable
        }
    }

    fn get_avg_spread(&self) -> Option<Decimal> {
        if self.entries.is_empty() {
            return None;
        }

        let sum: Decimal = self.entries.iter().map(|e| e.kraken_spread_bps).sum();
        Some(sum / Decimal::from(self.entries.len()))
    }
}

// ============================================================================
// Best Execution Analyzer
// ============================================================================

/// Best execution analyzer for Kraken orders
pub struct BestExecutionAnalyzer {
    /// Configuration
    config: ExecutionConfig,
    /// Price history by symbol
    price_history: Arc<RwLock<HashMap<String, PriceHistory>>>,
    /// Execution records by symbol
    execution_records: Arc<RwLock<HashMap<String, VecDeque<ExecutionRecord>>>>,
    /// Current price comparisons (injected from ArbitrageMonitor)
    current_prices: Arc<RwLock<HashMap<String, PriceComparison>>>,
}

impl BestExecutionAnalyzer {
    /// Create a new best execution analyzer
    pub fn new(config: ExecutionConfig) -> Self {
        Self {
            config,
            price_history: Arc::new(RwLock::new(HashMap::new())),
            execution_records: Arc::new(RwLock::new(HashMap::new())),
            current_prices: Arc::new(RwLock::new(HashMap::new())),
        }
    }

    /// Create from environment
    pub fn from_env() -> Self {
        Self::new(ExecutionConfig::from_env())
    }

    /// Update price data from arbitrage monitor
    pub async fn update_prices(&self, comparison: PriceComparison) {
        let symbol = comparison.symbol.clone();

        // Update current prices
        self.current_prices
            .write()
            .await
            .insert(symbol.clone(), comparison.clone());

        // Update price history
        if let Some(kraken_price) = comparison.prices.get(&Exchange::Kraken) {
            let market_avg: Decimal = comparison
                .prices
                .iter()
                .filter(|(e, _)| **e != Exchange::Kraken)
                .map(|(_, p)| p.mid)
                .sum::<Decimal>();

            let other_count = comparison
                .prices
                .iter()
                .filter(|(e, _)| **e != Exchange::Kraken)
                .count();

            let market_avg_mid = if other_count > 0 {
                market_avg / Decimal::from(other_count)
            } else {
                kraken_price.mid
            };

            let entry = PriceHistoryEntry {
                timestamp: Utc::now(),
                kraken_mid: kraken_price.mid,
                market_avg_mid,
                kraken_spread_bps: kraken_price.spread_bps,
            };

            let mut history = self.price_history.write().await;
            history
                .entry(symbol)
                .or_insert_with(|| PriceHistory::new(self.config.history_size))
                .add(entry);
        }
    }

    /// Analyze execution for a buy order
    pub async fn analyze_buy(&self, symbol: &str, quantity: Decimal) -> Option<ExecutionAnalysis> {
        self.analyze(symbol, OrderSide::Buy, quantity).await
    }

    /// Analyze execution for a sell order
    pub async fn analyze_sell(&self, symbol: &str, quantity: Decimal) -> Option<ExecutionAnalysis> {
        self.analyze(symbol, OrderSide::Sell, quantity).await
    }

    /// Analyze execution for an order
    pub async fn analyze(
        &self,
        symbol: &str,
        side: OrderSide,
        quantity: Decimal,
    ) -> Option<ExecutionAnalysis> {
        let start = Instant::now();
        let prices = self.current_prices.read().await;
        let comparison = prices.get(symbol)?;

        let kraken_price = comparison.prices.get(&Exchange::Kraken)?;

        // Get relevant price based on side
        let execution_price = match side {
            OrderSide::Buy => kraken_price.ask,
            OrderSide::Sell => kraken_price.bid,
        };

        // Calculate market average
        let other_mids: Vec<Decimal> = comparison
            .prices
            .iter()
            .filter(|(e, _)| **e != Exchange::Kraken)
            .map(|(_, p)| p.mid)
            .collect();

        let market_avg_price = if !other_mids.is_empty() {
            other_mids.iter().sum::<Decimal>() / Decimal::from(other_mids.len())
        } else {
            kraken_price.mid
        };

        // Calculate deviation
        let kraken_deviation_pct = if market_avg_price > Decimal::ZERO {
            (kraken_price.mid - market_avg_price) / market_avg_price * Decimal::from(100)
        } else {
            Decimal::ZERO
        };

        // Get historical data - extract all values before dropping the lock
        let (volatility_pct, trend, avg_spread, has_history) = {
            let history = self.price_history.read().await;
            match history.get(symbol) {
                Some(symbol_history) => {
                    let volatility_pct = symbol_history.get_volatility().unwrap_or(Decimal::ZERO);

                    let trend = symbol_history.get_trend();

                    let avg_spread = symbol_history
                        .get_avg_spread()
                        .unwrap_or(kraken_price.spread_bps);

                    (volatility_pct, trend, avg_spread, true)
                }
                None => (
                    Decimal::ZERO,
                    PriceTrend::Stable,
                    kraken_price.spread_bps,
                    false,
                ),
            }
        };

        // Estimate slippage based on quantity and spread
        let estimated_slippage_bps = self.estimate_slippage(quantity, kraken_price, avg_spread);

        // Calculate total cost (spread + slippage + fees)
        let estimated_cost_bps = kraken_price.spread_bps / Decimal::from(2) + // Half spread
            estimated_slippage_bps +
            self.config.kraken_fee_bps;

        // Calculate data freshness
        let data_age_ms = comparison
            .prices
            .values()
            .map(|p| {
                Utc::now()
                    .signed_duration_since(p.timestamp)
                    .num_milliseconds()
            })
            .max()
            .unwrap_or(0);

        // Calculate execution score
        let (score, reasons) = self.calculate_score(
            side,
            kraken_deviation_pct,
            kraken_price.spread_bps,
            volatility_pct,
            trend,
            estimated_cost_bps,
            data_age_ms,
        );

        // Determine recommendation
        let recommendation =
            self.determine_recommendation(score, estimated_cost_bps, volatility_pct, data_age_ms);

        // Calculate suggested limit price
        let suggested_limit_price =
            self.calculate_limit_price(side, kraken_price, market_avg_price, kraken_deviation_pct);

        // Calculate confidence based on data quality
        let confidence = self.calculate_confidence(comparison, data_age_ms, has_history);

        let analysis = ExecutionAnalysis {
            symbol: symbol.to_string(),
            side,
            quantity,
            score,
            recommendation,
            kraken_price: execution_price,
            market_avg_price,
            kraken_deviation_pct,
            kraken_spread_bps: kraken_price.spread_bps,
            estimated_slippage_bps,
            estimated_cost_bps,
            volatility_pct,
            trend,
            confidence,
            suggested_limit_price,
            reasons,
            timestamp: Utc::now(),
            data_age_ms,
        };

        // Record analysis latency in histogram
        let duration_ms = start.elapsed().as_secs_f64() * 1000.0;
        global_latency_histograms().record_best_execution_analysis(symbol, duration_ms);

        // Emit metrics directly from analyzer for better instrumentation coverage
        let metrics = best_execution_metrics();
        let rec_type = match recommendation {
            ExecutionRecommendation::ExecuteNow => RecommendationType::ExecuteNow,
            ExecutionRecommendation::Wait => RecommendationType::Wait,
            ExecutionRecommendation::Acceptable => RecommendationType::Acceptable,
            ExecutionRecommendation::UseLimitOrder => RecommendationType::UseLimitOrder,
            ExecutionRecommendation::Avoid => RecommendationType::Avoid,
        };
        metrics.record_analysis(symbol, rec_type, score, estimated_slippage_bps);

        Some(analysis)
    }

    /// Estimate slippage based on order size
    fn estimate_slippage(
        &self,
        quantity: Decimal,
        price: &ExchangePrice,
        avg_spread: Decimal,
    ) -> Decimal {
        // Base slippage from spread
        let base_slippage = avg_spread / Decimal::from(4);

        // Size impact - larger orders have more slippage
        let available_qty = price.bid_qty.min(price.ask_qty);
        let size_ratio = if available_qty > Decimal::ZERO {
            quantity / available_qty
        } else {
            Decimal::from(10) // Assume high impact if no depth data
        };

        // Size multiplier: 1x for small orders, up to 5x for large orders
        let size_multiplier = Decimal::ONE + (size_ratio * Decimal::from(4)).min(Decimal::from(4));

        base_slippage * size_multiplier
    }

    /// Calculate execution score
    #[allow(clippy::too_many_arguments)]
    fn calculate_score(
        &self,
        side: OrderSide,
        deviation_pct: Decimal,
        spread_bps: Decimal,
        volatility_pct: Decimal,
        trend: PriceTrend,
        estimated_cost_bps: Decimal,
        data_age_ms: i64,
    ) -> (Decimal, Vec<String>) {
        let mut score = Decimal::from(50); // Start at neutral
        let mut reasons = Vec::new();

        // 1. Price deviation score (40% weight)
        // For buys: negative deviation is good (Kraken cheaper)
        // For sells: positive deviation is good (Kraken higher)
        let deviation_score = match side {
            OrderSide::Buy => {
                if deviation_pct < Decimal::new(-10, 2) {
                    // > 0.10% cheaper
                    reasons.push(format!(
                        "Kraken is {:.2}% cheaper than market average",
                        deviation_pct.abs()
                    ));
                    Decimal::from(100)
                } else if deviation_pct < Decimal::ZERO {
                    reasons.push("Kraken slightly below market".to_string());
                    Decimal::from(70)
                } else if deviation_pct < Decimal::new(10, 2) {
                    Decimal::from(50)
                } else {
                    reasons.push(format!(
                        "Kraken is {:.2}% more expensive than market",
                        deviation_pct
                    ));
                    Decimal::from(20)
                }
            }
            OrderSide::Sell => {
                if deviation_pct > Decimal::new(10, 2) {
                    reasons.push(format!(
                        "Kraken is {:.2}% higher than market average",
                        deviation_pct
                    ));
                    Decimal::from(100)
                } else if deviation_pct > Decimal::ZERO {
                    reasons.push("Kraken slightly above market".to_string());
                    Decimal::from(70)
                } else if deviation_pct > Decimal::new(-10, 2) {
                    Decimal::from(50)
                } else {
                    reasons.push(format!(
                        "Kraken is {:.2}% cheaper than market",
                        deviation_pct.abs()
                    ));
                    Decimal::from(20)
                }
            }
        };
        score += (deviation_score - Decimal::from(50)) * self.config.price_deviation_weight
            / Decimal::from(100);

        // 2. Spread score (25% weight)
        let spread_score = if spread_bps < Decimal::from(5) {
            reasons.push("Very tight spread".to_string());
            Decimal::from(100)
        } else if spread_bps < Decimal::from(15) {
            Decimal::from(80)
        } else if spread_bps < Decimal::from(30) {
            Decimal::from(60)
        } else {
            reasons.push(format!("Wide spread: {:.1} bps", spread_bps));
            Decimal::from(30)
        };
        score +=
            (spread_score - Decimal::from(50)) * self.config.spread_weight / Decimal::from(100);

        // 3. Volatility score (20% weight)
        let volatility_score = if volatility_pct < Decimal::new(5, 2) {
            Decimal::from(100)
        } else if volatility_pct < Decimal::new(15, 2) {
            Decimal::from(70)
        } else if volatility_pct < Decimal::new(30, 2) {
            reasons.push("Moderate volatility".to_string());
            Decimal::from(50)
        } else {
            reasons.push(format!("High volatility: {:.2}%", volatility_pct));
            Decimal::from(20)
        };
        score += (volatility_score - Decimal::from(50)) * self.config.volatility_weight
            / Decimal::from(100);

        // 4. Trend score (15% weight)
        let trend_score = match (side, trend) {
            (OrderSide::Buy, PriceTrend::Down) => {
                reasons.push("Price trending down - good for buying".to_string());
                Decimal::from(80)
            }
            (OrderSide::Sell, PriceTrend::Up) => {
                reasons.push("Price trending up - good for selling".to_string());
                Decimal::from(80)
            }
            (OrderSide::Buy, PriceTrend::Up) => {
                reasons.push("Price trending up - consider waiting".to_string());
                Decimal::from(30)
            }
            (OrderSide::Sell, PriceTrend::Down) => {
                reasons.push("Price trending down - consider waiting".to_string());
                Decimal::from(30)
            }
            (_, PriceTrend::Stable) => Decimal::from(50),
        };
        score += (trend_score - Decimal::from(50)) * self.config.trend_weight / Decimal::from(100);

        // 5. Data freshness penalty
        if data_age_ms > 5000 {
            score -= Decimal::from(10);
            reasons.push("Stale price data".to_string());
        } else if data_age_ms > 2000 {
            score -= Decimal::from(5);
        }

        // 6. Total cost adjustment
        if estimated_cost_bps > self.config.max_slippage_bps {
            score -= Decimal::from(15);
            reasons.push(format!(
                "High estimated cost: {:.1} bps",
                estimated_cost_bps
            ));
        }

        // Clamp score to 0-100
        score = score.max(Decimal::ZERO).min(Decimal::from(100));

        (score, reasons)
    }

    /// Determine execution recommendation
    fn determine_recommendation(
        &self,
        score: Decimal,
        estimated_cost_bps: Decimal,
        volatility_pct: Decimal,
        data_age_ms: i64,
    ) -> ExecutionRecommendation {
        // Reject if data is too stale
        if data_age_ms > 10000 {
            return ExecutionRecommendation::Wait;
        }

        // High score = execute now
        if score >= Decimal::from(80) {
            return ExecutionRecommendation::ExecuteNow;
        }

        // Very low score = avoid
        if score < Decimal::from(30) {
            return ExecutionRecommendation::Avoid;
        }

        // Cost too high = use limit order
        if estimated_cost_bps > self.config.max_slippage_bps {
            return ExecutionRecommendation::UseLimitOrder;
        }

        // High volatility = wait
        if volatility_pct > Decimal::new(30, 2) {
            return ExecutionRecommendation::Wait;
        }

        // Medium score = acceptable
        if score >= self.config.min_execution_score {
            ExecutionRecommendation::Acceptable
        } else {
            ExecutionRecommendation::Wait
        }
    }

    /// Calculate suggested limit price
    fn calculate_limit_price(
        &self,
        side: OrderSide,
        kraken_price: &ExchangePrice,
        market_avg: Decimal,
        deviation_pct: Decimal,
    ) -> Option<Decimal> {
        // Suggest a limit price between Kraken and market average
        let improvement_factor = Decimal::new(3, 1); // Aim for 30% of the gap

        match side {
            OrderSide::Buy => {
                if deviation_pct > Decimal::ZERO {
                    // Kraken is more expensive, try to buy at market avg
                    let gap = kraken_price.ask - market_avg;
                    Some(kraken_price.ask - gap * improvement_factor)
                } else {
                    // Kraken is cheaper, bid slightly below ask
                    Some(
                        kraken_price.ask
                            - kraken_price.spread_bps / Decimal::from(10000)
                                * kraken_price.ask
                                * Decimal::new(3, 1),
                    )
                }
            }
            OrderSide::Sell => {
                if deviation_pct < Decimal::ZERO {
                    // Kraken is cheaper, try to sell at market avg
                    let gap = market_avg - kraken_price.bid;
                    Some(kraken_price.bid + gap * improvement_factor)
                } else {
                    // Kraken is higher, ask slightly above bid
                    Some(
                        kraken_price.bid
                            + kraken_price.spread_bps / Decimal::from(10000)
                                * kraken_price.bid
                                * Decimal::new(3, 1),
                    )
                }
            }
        }
    }

    /// Calculate confidence based on data quality
    fn calculate_confidence(
        &self,
        comparison: &PriceComparison,
        data_age_ms: i64,
        has_history: bool,
    ) -> Decimal {
        let mut confidence = Decimal::from(100);

        // Reduce confidence for stale data
        if data_age_ms > 5000 {
            confidence -= Decimal::from(30);
        } else if data_age_ms > 2000 {
            confidence -= Decimal::from(15);
        } else if data_age_ms > 1000 {
            confidence -= Decimal::from(5);
        }

        // Reduce confidence for fewer exchanges
        let exchange_count = comparison.prices.len();
        if exchange_count < 2 {
            confidence -= Decimal::from(40);
        } else if exchange_count < 3 {
            confidence -= Decimal::from(15);
        }

        // Reduce confidence without history
        if !has_history {
            confidence -= Decimal::from(20);
        }

        confidence.max(Decimal::ZERO) / Decimal::from(100)
    }

    /// Record an execution for quality tracking
    pub async fn record_execution(
        &self,
        symbol: &str,
        side: OrderSide,
        quantity: Decimal,
        analysis_score: Decimal,
        expected_price: Decimal,
        actual_price: Decimal,
    ) {
        let expected_slippage_bps = if expected_price > Decimal::ZERO {
            match side {
                OrderSide::Buy => {
                    (actual_price - expected_price) / expected_price * Decimal::from(10000)
                }
                OrderSide::Sell => {
                    (expected_price - actual_price) / expected_price * Decimal::from(10000)
                }
            }
        } else {
            Decimal::ZERO
        };

        let actual_slippage_bps = expected_slippage_bps; // In this case they're the same calculation

        let record = ExecutionRecord {
            symbol: symbol.to_string(),
            side,
            quantity,
            analysis_score,
            expected_price,
            actual_price,
            expected_slippage_bps,
            actual_slippage_bps,
            timestamp: Utc::now(),
        };

        let mut records = self.execution_records.write().await;
        let symbol_records = records
            .entry(symbol.to_string())
            .or_insert_with(|| VecDeque::with_capacity(1000));

        if symbol_records.len() >= 1000 {
            symbol_records.pop_front();
        }
        symbol_records.push_back(record.clone());

        // Emit metrics for execution quality tracking
        let quality_score = record.quality_score();
        let slippage_exceeded = record.actual_slippage_bps > self.config.max_slippage_bps;
        let metrics = best_execution_metrics();
        metrics.record_execution(symbol, quality_score, slippage_exceeded);
    }

    /// Get execution quality metrics for a symbol
    pub async fn get_quality_metrics(&self, symbol: &str) -> Option<ExecutionQualityMetrics> {
        let records = self.execution_records.read().await;
        let symbol_records = records.get(symbol)?;

        if symbol_records.is_empty() {
            return None;
        }

        let total = symbol_records.len();
        let scores: Vec<Decimal> = symbol_records.iter().map(|r| r.quality_score()).collect();
        let avg_quality_score = scores.iter().sum::<Decimal>() / Decimal::from(total);

        let slippages: Vec<Decimal> = symbol_records
            .iter()
            .map(|r| r.actual_slippage_bps)
            .collect();
        let avg_slippage_bps = slippages.iter().sum::<Decimal>() / Decimal::from(total);
        let max_slippage_bps = slippages.iter().max().copied().unwrap_or(Decimal::ZERO);

        let slippage_exceeded = symbol_records
            .iter()
            .filter(|r| r.actual_slippage_bps > r.expected_slippage_bps)
            .count();

        let avg_analysis_score = symbol_records
            .iter()
            .map(|r| r.analysis_score)
            .sum::<Decimal>()
            / Decimal::from(total);

        // Recommendation accuracy: % of executions where quality score >= 50
        let good_executions = scores.iter().filter(|s| **s >= Decimal::from(50)).count();
        let recommendation_accuracy =
            Decimal::from(good_executions) / Decimal::from(total) * Decimal::from(100);

        Some(ExecutionQualityMetrics {
            symbol: symbol.to_string(),
            total_executions: total,
            avg_quality_score,
            avg_slippage_bps,
            max_slippage_bps,
            slippage_exceeded_count: slippage_exceeded,
            avg_analysis_score,
            recommendation_accuracy,
        })
    }

    /// Get all quality metrics
    pub async fn get_all_quality_metrics(&self) -> HashMap<String, ExecutionQualityMetrics> {
        // First, collect all the symbols we need to query
        let symbols: Vec<String> = {
            let records = self.execution_records.read().await;
            records.keys().cloned().collect()
        };

        let mut metrics = HashMap::new();

        for symbol in symbols {
            if let Some(m) = self.get_quality_metrics(&symbol).await {
                metrics.insert(symbol, m);
            }
        }

        metrics
    }

    /// Clear all data
    pub async fn clear(&self) {
        self.price_history.write().await.clear();
        self.execution_records.write().await.clear();
        self.current_prices.write().await.clear();
    }
}

// ============================================================================
// Tests
// ============================================================================

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_config_default() {
        let config = ExecutionConfig::default();
        assert_eq!(config.min_execution_score, Decimal::from(60));
        assert_eq!(config.kraken_fee_bps, Decimal::from(26));
    }

    #[test]
    fn test_execution_record_quality_score() {
        let record = ExecutionRecord {
            symbol: "BTC/USDT".to_string(),
            side: OrderSide::Buy,
            quantity: Decimal::from(1),
            analysis_score: Decimal::from(80),
            expected_price: Decimal::from(50000),
            actual_price: Decimal::from(50010),
            expected_slippage_bps: Decimal::from(10),
            actual_slippage_bps: Decimal::from(5),
            timestamp: Utc::now(),
        };

        // Better than expected = 100
        assert_eq!(record.quality_score(), Decimal::from(100));
    }

    #[test]
    fn test_execution_record_quality_score_worse() {
        let record = ExecutionRecord {
            symbol: "BTC/USDT".to_string(),
            side: OrderSide::Buy,
            quantity: Decimal::from(1),
            analysis_score: Decimal::from(80),
            expected_price: Decimal::from(50000),
            actual_price: Decimal::from(50050),
            expected_slippage_bps: Decimal::from(5),
            actual_slippage_bps: Decimal::from(10), // 2x expected
            timestamp: Utc::now(),
        };

        let score = record.quality_score();
        assert!(score < Decimal::from(100));
        assert!(score > Decimal::ZERO);
    }

    #[test]
    fn test_execution_analysis_should_execute() {
        let analysis = ExecutionAnalysis {
            symbol: "BTC/USDT".to_string(),
            side: OrderSide::Buy,
            quantity: Decimal::from(1),
            score: Decimal::from(85),
            recommendation: ExecutionRecommendation::ExecuteNow,
            kraken_price: Decimal::from(50000),
            market_avg_price: Decimal::from(50010),
            kraken_deviation_pct: Decimal::new(-2, 2),
            kraken_spread_bps: Decimal::from(10),
            estimated_slippage_bps: Decimal::from(5),
            estimated_cost_bps: Decimal::from(40),
            volatility_pct: Decimal::new(5, 2),
            trend: PriceTrend::Stable,
            confidence: Decimal::new(9, 1),
            suggested_limit_price: Some(Decimal::from(49995)),
            reasons: vec!["Good conditions".to_string()],
            timestamp: Utc::now(),
            data_age_ms: 100,
        };

        assert!(analysis.should_execute());
    }

    #[test]
    fn test_execution_analysis_expected_fill() {
        let analysis = ExecutionAnalysis {
            symbol: "BTC/USDT".to_string(),
            side: OrderSide::Buy,
            quantity: Decimal::from(1),
            score: Decimal::from(80),
            recommendation: ExecutionRecommendation::ExecuteNow,
            kraken_price: Decimal::from(50000),
            market_avg_price: Decimal::from(50000),
            kraken_deviation_pct: Decimal::ZERO,
            kraken_spread_bps: Decimal::from(10),
            estimated_slippage_bps: Decimal::from(10), // 10 bps = 0.10%
            estimated_cost_bps: Decimal::from(40),
            volatility_pct: Decimal::new(5, 2),
            trend: PriceTrend::Stable,
            confidence: Decimal::ONE,
            suggested_limit_price: None,
            reasons: vec![],
            timestamp: Utc::now(),
            data_age_ms: 100,
        };

        // Expected fill = 50000 * (1 + 0.001) = 50050
        let expected = analysis.expected_fill_price();
        assert_eq!(expected, Decimal::from(50050));
    }

    #[test]
    fn test_order_side_display() {
        assert_eq!(format!("{}", OrderSide::Buy), "BUY");
        assert_eq!(format!("{}", OrderSide::Sell), "SELL");
    }

    #[test]
    fn test_recommendation_display() {
        assert_eq!(
            format!("{}", ExecutionRecommendation::ExecuteNow),
            "EXECUTE_NOW"
        );
        assert_eq!(format!("{}", ExecutionRecommendation::Wait), "WAIT");
    }

    #[tokio::test]
    async fn test_analyzer_creation() {
        let config = ExecutionConfig::default();
        let analyzer = BestExecutionAnalyzer::new(config);

        // Should start empty
        let prices = analyzer.current_prices.read().await;
        assert!(prices.is_empty());
    }

    #[tokio::test]
    async fn test_analyzer_update_prices() {
        let config = ExecutionConfig::default();
        let analyzer = BestExecutionAnalyzer::new(config);

        let mut comparison = PriceComparison::new("BTC/USDT".to_string());
        comparison.add_price(ExchangePrice::new(
            Exchange::Kraken,
            "BTC/USDT".to_string(),
            Decimal::from(50000),
            Decimal::from(50010),
            Decimal::from(1),
            Decimal::from(1),
        ));

        analyzer.update_prices(comparison).await;

        let prices = analyzer.current_prices.read().await;
        assert!(prices.contains_key("BTC/USDT"));
    }

    #[tokio::test]
    async fn test_analyzer_analyze_buy() {
        let config = ExecutionConfig::default();
        let analyzer = BestExecutionAnalyzer::new(config);

        // Add price data
        let mut comparison = PriceComparison::new("BTC/USDT".to_string());
        comparison.add_price(ExchangePrice::new(
            Exchange::Kraken,
            "BTC/USDT".to_string(),
            Decimal::from(50000),
            Decimal::from(50010),
            Decimal::from(1),
            Decimal::from(1),
        ));
        comparison.add_price(ExchangePrice::new(
            Exchange::Binance,
            "BTC/USDT".to_string(),
            Decimal::from(50005),
            Decimal::from(50015),
            Decimal::from(1),
            Decimal::from(1),
        ));

        analyzer.update_prices(comparison).await;

        let analysis = analyzer.analyze_buy("BTC/USDT", Decimal::from(1)).await;

        assert!(analysis.is_some());
        let analysis = analysis.unwrap();
        assert_eq!(analysis.symbol, "BTC/USDT");
        assert_eq!(analysis.side, OrderSide::Buy);
        assert!(analysis.score >= Decimal::ZERO);
        assert!(analysis.score <= Decimal::from(100));
    }

    #[tokio::test]
    async fn test_analyzer_record_execution() {
        let config = ExecutionConfig::default();
        let analyzer = BestExecutionAnalyzer::new(config);

        analyzer
            .record_execution(
                "BTC/USDT",
                OrderSide::Buy,
                Decimal::from(1),
                Decimal::from(80),
                Decimal::from(50000),
                Decimal::from(50005),
            )
            .await;

        let metrics = analyzer.get_quality_metrics("BTC/USDT").await;
        assert!(metrics.is_some());

        let metrics = metrics.unwrap();
        assert_eq!(metrics.total_executions, 1);
    }

    #[test]
    fn test_price_trend() {
        // Just verify enum works
        let trend = PriceTrend::Up;
        assert_eq!(trend, PriceTrend::Up);
    }
}
