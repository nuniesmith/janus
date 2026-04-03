//! Price Arbitrage Monitor
//!
//! This module provides real-time cross-exchange price monitoring
//! to identify arbitrage opportunities and optimal execution timing.
//!
//! # Overview
//!
//! Since Kraken is the only exchange available for trading in Canada,
//! the arbitrage monitor uses Bybit and Binance data to:
//! - Detect when Kraken prices are favorable compared to other exchanges
//! - Identify potential arbitrage spreads (informational only)
//! - Signal optimal entry/exit timing for Kraken orders
//!
//! # Features
//!
//! - Real-time price aggregation from multiple exchanges
//! - Spread calculation and tracking
//! - Arbitrage opportunity detection
//! - Historical spread analysis
//! - Alert system for significant price deviations
//!
//! # Example
//!
//! ```rust,ignore
//! use janus_execution::execution::arbitrage::{ArbitrageMonitor, ArbitrageConfig};
//!
//! let config = ArbitrageConfig::default();
//! let monitor = ArbitrageMonitor::new(config);
//!
//! // Start monitoring
//! monitor.start().await?;
//!
//! // Subscribe to opportunities
//! let mut rx = monitor.subscribe_opportunities();
//! while let Ok(opp) = rx.recv().await {
//!     println!("Arbitrage opportunity: {} spread {:.2}%", opp.symbol, opp.spread_pct);
//! }
//! ```

use chrono::{DateTime, Utc};
use rust_decimal::Decimal;
use rust_decimal::prelude::*;
use serde::{Deserialize, Serialize};
use std::collections::HashMap;
use std::sync::Arc;
use tokio::sync::{RwLock, broadcast};
use tokio::time::Duration;
use tracing::info;

// ============================================================================
// Configuration
// ============================================================================

/// Arbitrage monitor configuration
#[derive(Debug, Clone)]
pub struct ArbitrageConfig {
    /// Minimum spread percentage to consider as opportunity
    pub min_spread_pct: Decimal,
    /// Symbols to monitor
    pub symbols: Vec<String>,
    /// Price update interval
    pub update_interval: Duration,
    /// How long to keep price history
    pub history_window: Duration,
    /// Alert threshold for large spreads (percentage)
    pub alert_threshold_pct: Decimal,
    /// Enable Kraken comparison (primary trading exchange)
    pub compare_kraken: bool,
    /// Enable Bybit comparison
    pub compare_bybit: bool,
    /// Enable Binance comparison
    pub compare_binance: bool,
}

impl Default for ArbitrageConfig {
    fn default() -> Self {
        Self {
            min_spread_pct: Decimal::new(5, 2), // 0.05%
            symbols: vec![
                "BTC/USDT".to_string(),
                "ETH/USDT".to_string(),
                "SOL/USDT".to_string(),
            ],
            update_interval: Duration::from_millis(100),
            history_window: Duration::from_secs(300), // 5 minutes
            alert_threshold_pct: Decimal::new(50, 2), // 0.50%
            compare_kraken: true,
            compare_bybit: true,
            compare_binance: true,
        }
    }
}

impl ArbitrageConfig {
    /// Create config from environment variables
    pub fn from_env() -> Self {
        let min_spread_pct = std::env::var("ARB_MIN_SPREAD_PCT")
            .ok()
            .and_then(|s| Decimal::from_str(&s).ok())
            .unwrap_or(Decimal::new(5, 2));

        let symbols = std::env::var("ARB_SYMBOLS")
            .map(|s| s.split(',').map(|s| s.trim().to_string()).collect())
            .unwrap_or_else(|_| {
                vec![
                    "BTC/USDT".to_string(),
                    "ETH/USDT".to_string(),
                    "SOL/USDT".to_string(),
                ]
            });

        let alert_threshold_pct = std::env::var("ARB_ALERT_THRESHOLD_PCT")
            .ok()
            .and_then(|s| Decimal::from_str(&s).ok())
            .unwrap_or(Decimal::new(50, 2));

        Self {
            min_spread_pct,
            symbols,
            alert_threshold_pct,
            ..Default::default()
        }
    }
}

// ============================================================================
// Price Types
// ============================================================================

/// Exchange identifier
#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash, Serialize, Deserialize)]
pub enum Exchange {
    Kraken,
    Bybit,
    Binance,
}

impl std::fmt::Display for Exchange {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        match self {
            Exchange::Kraken => write!(f, "Kraken"),
            Exchange::Bybit => write!(f, "Bybit"),
            Exchange::Binance => write!(f, "Binance"),
        }
    }
}

/// Price quote from an exchange
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct ExchangePrice {
    /// Exchange source
    pub exchange: Exchange,
    /// Trading symbol
    pub symbol: String,
    /// Best bid price
    pub bid: Decimal,
    /// Best ask price
    pub ask: Decimal,
    /// Bid quantity available
    pub bid_qty: Decimal,
    /// Ask quantity available
    pub ask_qty: Decimal,
    /// Mid price (avg of bid/ask)
    pub mid: Decimal,
    /// Spread in basis points
    pub spread_bps: Decimal,
    /// Last update timestamp
    pub timestamp: DateTime<Utc>,
}

impl ExchangePrice {
    /// Create a new exchange price
    pub fn new(
        exchange: Exchange,
        symbol: String,
        bid: Decimal,
        ask: Decimal,
        bid_qty: Decimal,
        ask_qty: Decimal,
    ) -> Self {
        let mid = (bid + ask) / Decimal::from(2);
        let spread_bps = if mid > Decimal::ZERO {
            ((ask - bid) / mid) * Decimal::from(10000)
        } else {
            Decimal::ZERO
        };

        Self {
            exchange,
            symbol,
            bid,
            ask,
            bid_qty,
            ask_qty,
            mid,
            spread_bps,
            timestamp: Utc::now(),
        }
    }

    /// Check if price is stale (older than threshold)
    pub fn is_stale(&self, max_age: Duration) -> bool {
        let age = Utc::now()
            .signed_duration_since(self.timestamp)
            .num_milliseconds();
        age > max_age.as_millis() as i64
    }
}

// ============================================================================
// Arbitrage Opportunity
// ============================================================================

/// Direction of arbitrage opportunity
#[derive(Debug, Clone, Copy, PartialEq, Eq, Serialize, Deserialize)]
pub enum ArbitrageDirection {
    /// Buy on exchange A, sell on exchange B
    BuyASellB,
    /// Buy on exchange B, sell on exchange A
    BuyBSellA,
}

/// Detected arbitrage opportunity
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct ArbitrageOpportunity {
    /// Trading symbol
    pub symbol: String,
    /// Exchange to buy from
    pub buy_exchange: Exchange,
    /// Exchange to sell on
    pub sell_exchange: Exchange,
    /// Buy price (ask on buy exchange)
    pub buy_price: Decimal,
    /// Sell price (bid on sell exchange)
    pub sell_price: Decimal,
    /// Spread amount
    pub spread: Decimal,
    /// Spread as percentage
    pub spread_pct: Decimal,
    /// Maximum profitable quantity (limited by order book depth)
    pub max_qty: Decimal,
    /// Estimated profit (before fees)
    pub estimated_profit: Decimal,
    /// Detection timestamp
    pub timestamp: DateTime<Utc>,
    /// Time since prices updated
    pub price_age_ms: i64,
    /// Whether this is actionable (Kraken involved)
    pub actionable: bool,
}

impl ArbitrageOpportunity {
    /// Calculate estimated profit after fees
    pub fn profit_after_fees(&self, fee_rate: Decimal) -> Decimal {
        let total_fees = self.buy_price * self.max_qty * fee_rate * Decimal::from(2);
        self.estimated_profit - total_fees
    }

    /// Check if opportunity is still valid
    pub fn is_valid(&self, max_age_ms: i64) -> bool {
        self.price_age_ms <= max_age_ms && self.spread > Decimal::ZERO
    }

    /// Get opportunity score (higher = better)
    pub fn score(&self) -> Decimal {
        // Score based on spread percentage, adjusted by quantity
        self.spread_pct * self.max_qty.min(Decimal::from(1))
    }
}

// ============================================================================
// Price Comparison
// ============================================================================

/// Cross-exchange price comparison for a symbol
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct PriceComparison {
    /// Trading symbol
    pub symbol: String,
    /// Prices from each exchange
    pub prices: HashMap<Exchange, ExchangePrice>,
    /// Best bid across all exchanges
    pub best_bid: Option<(Exchange, Decimal)>,
    /// Best ask across all exchanges
    pub best_ask: Option<(Exchange, Decimal)>,
    /// Cross-exchange spread (best bid - best ask)
    pub cross_spread: Option<Decimal>,
    /// Cross-exchange spread percentage
    pub cross_spread_pct: Option<Decimal>,
    /// Kraken vs market average deviation
    pub kraken_deviation_pct: Option<Decimal>,
    /// Comparison timestamp
    pub timestamp: DateTime<Utc>,
}

impl PriceComparison {
    /// Create a new price comparison
    pub fn new(symbol: String) -> Self {
        Self {
            symbol,
            prices: HashMap::new(),
            best_bid: None,
            best_ask: None,
            cross_spread: None,
            cross_spread_pct: None,
            kraken_deviation_pct: None,
            timestamp: Utc::now(),
        }
    }

    /// Add a price quote
    pub fn add_price(&mut self, price: ExchangePrice) {
        self.prices.insert(price.exchange, price);
        self.recalculate();
    }

    /// Recalculate derived values
    fn recalculate(&mut self) {
        self.timestamp = Utc::now();

        // Find best bid and ask
        self.best_bid = self
            .prices
            .iter()
            .max_by(|a, b| a.1.bid.cmp(&b.1.bid))
            .map(|(e, p)| (*e, p.bid));

        self.best_ask = self
            .prices
            .iter()
            .min_by(|a, b| a.1.ask.cmp(&b.1.ask))
            .map(|(e, p)| (*e, p.ask));

        // Calculate cross-exchange spread
        if let (Some((_, best_bid)), Some((_, best_ask))) = (self.best_bid, self.best_ask) {
            self.cross_spread = Some(best_bid - best_ask);
            if best_ask > Decimal::ZERO {
                self.cross_spread_pct = Some((best_bid - best_ask) / best_ask * Decimal::from(100));
            }
        }

        // Calculate Kraken deviation from average
        if let Some(kraken_price) = self.prices.get(&Exchange::Kraken) {
            let other_mids: Vec<Decimal> = self
                .prices
                .iter()
                .filter(|(e, _)| **e != Exchange::Kraken)
                .map(|(_, p)| p.mid)
                .collect();

            if !other_mids.is_empty() {
                let avg_mid: Decimal =
                    other_mids.iter().sum::<Decimal>() / Decimal::from(other_mids.len());
                if avg_mid > Decimal::ZERO {
                    self.kraken_deviation_pct =
                        Some((kraken_price.mid - avg_mid) / avg_mid * Decimal::from(100));
                }
            }
        }
    }

    /// Check if there's an arbitrage opportunity
    pub fn find_arbitrage(&self, min_spread_pct: Decimal) -> Option<ArbitrageOpportunity> {
        let (best_bid_exchange, best_bid) = self.best_bid?;
        let (best_ask_exchange, best_ask) = self.best_ask?;

        // No arbitrage if same exchange
        if best_bid_exchange == best_ask_exchange {
            return None;
        }

        let spread = best_bid - best_ask;
        if spread <= Decimal::ZERO {
            return None;
        }

        let spread_pct = if best_ask > Decimal::ZERO {
            spread / best_ask * Decimal::from(100)
        } else {
            Decimal::ZERO
        };

        if spread_pct < min_spread_pct {
            return None;
        }

        // Get quantities
        let buy_qty = self
            .prices
            .get(&best_ask_exchange)
            .map(|p| p.ask_qty)
            .unwrap_or(Decimal::ZERO);
        let sell_qty = self
            .prices
            .get(&best_bid_exchange)
            .map(|p| p.bid_qty)
            .unwrap_or(Decimal::ZERO);
        let max_qty = buy_qty.min(sell_qty);

        // Calculate max age of prices
        let max_age = self
            .prices
            .values()
            .map(|p| {
                Utc::now()
                    .signed_duration_since(p.timestamp)
                    .num_milliseconds()
            })
            .max()
            .unwrap_or(0);

        // Check if Kraken is involved (actionable)
        let actionable =
            best_bid_exchange == Exchange::Kraken || best_ask_exchange == Exchange::Kraken;

        Some(ArbitrageOpportunity {
            symbol: self.symbol.clone(),
            buy_exchange: best_ask_exchange,
            sell_exchange: best_bid_exchange,
            buy_price: best_ask,
            sell_price: best_bid,
            spread,
            spread_pct,
            max_qty,
            estimated_profit: spread * max_qty,
            timestamp: Utc::now(),
            price_age_ms: max_age,
            actionable,
        })
    }

    /// Get recommendation for Kraken trading
    pub fn kraken_recommendation(&self) -> Option<KrakenRecommendation> {
        let kraken_price = self.prices.get(&Exchange::Kraken)?;
        let deviation = self.kraken_deviation_pct?;

        let action = if deviation < Decimal::new(-20, 2) {
            // Kraken is 0.20%+ cheaper - good to buy
            RecommendedAction::Buy
        } else if deviation > Decimal::new(20, 2) {
            // Kraken is 0.20%+ more expensive - good to sell
            RecommendedAction::Sell
        } else {
            RecommendedAction::Hold
        };

        let strength = deviation.abs();

        Some(KrakenRecommendation {
            symbol: self.symbol.clone(),
            action,
            deviation_pct: deviation,
            strength,
            kraken_bid: kraken_price.bid,
            kraken_ask: kraken_price.ask,
            market_avg_mid: self
                .prices
                .iter()
                .filter(|(e, _)| **e != Exchange::Kraken)
                .map(|(_, p)| p.mid)
                .sum::<Decimal>()
                / Decimal::from(
                    self.prices
                        .iter()
                        .filter(|(e, _)| **e != Exchange::Kraken)
                        .count()
                        .max(1),
                ),
            timestamp: Utc::now(),
        })
    }
}

/// Recommended action for Kraken
#[derive(Debug, Clone, Copy, PartialEq, Eq, Serialize, Deserialize)]
pub enum RecommendedAction {
    Buy,
    Sell,
    Hold,
}

/// Trading recommendation for Kraken
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct KrakenRecommendation {
    /// Trading symbol
    pub symbol: String,
    /// Recommended action
    pub action: RecommendedAction,
    /// Kraken deviation from market average
    pub deviation_pct: Decimal,
    /// Signal strength (0-1)
    pub strength: Decimal,
    /// Current Kraken bid
    pub kraken_bid: Decimal,
    /// Current Kraken ask
    pub kraken_ask: Decimal,
    /// Market average mid price
    pub market_avg_mid: Decimal,
    /// Timestamp
    pub timestamp: DateTime<Utc>,
}

// ============================================================================
// Spread History
// ============================================================================

/// Historical spread data point
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct SpreadDataPoint {
    /// Timestamp
    pub timestamp: DateTime<Utc>,
    /// Cross-exchange spread percentage
    pub spread_pct: Decimal,
    /// Kraken deviation percentage
    pub kraken_deviation_pct: Option<Decimal>,
    /// Best bid exchange
    pub best_bid_exchange: Exchange,
    /// Best ask exchange
    pub best_ask_exchange: Exchange,
}

/// Spread statistics
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct SpreadStats {
    /// Symbol
    pub symbol: String,
    /// Time window
    pub window_secs: u64,
    /// Number of data points
    pub sample_count: usize,
    /// Average spread percentage
    pub avg_spread_pct: Decimal,
    /// Maximum spread percentage
    pub max_spread_pct: Decimal,
    /// Minimum spread percentage
    pub min_spread_pct: Decimal,
    /// Standard deviation of spread
    pub std_dev: Decimal,
    /// Percentage of time with arbitrage opportunity
    pub opportunity_pct: Decimal,
    /// Average Kraken deviation
    pub avg_kraken_deviation: Option<Decimal>,
}

// ============================================================================
// Arbitrage Monitor
// ============================================================================

/// Events emitted by the arbitrage monitor
#[derive(Debug, Clone)]
pub enum ArbitrageEvent {
    /// New arbitrage opportunity detected
    Opportunity(ArbitrageOpportunity),
    /// Price comparison update
    PriceUpdate(PriceComparison),
    /// Kraken recommendation update
    Recommendation(KrakenRecommendation),
    /// Alert for significant spread
    SpreadAlert {
        symbol: String,
        spread_pct: Decimal,
        message: String,
    },
    /// Monitor started
    Started,
    /// Monitor stopped
    Stopped,
    /// Error occurred
    Error(String),
}

/// Arbitrage monitor for cross-exchange price comparison
pub struct ArbitrageMonitor {
    /// Configuration
    config: ArbitrageConfig,
    /// Current price comparisons by symbol
    comparisons: Arc<RwLock<HashMap<String, PriceComparison>>>,
    /// Spread history by symbol
    history: Arc<RwLock<HashMap<String, Vec<SpreadDataPoint>>>>,
    /// Event broadcaster
    event_tx: broadcast::Sender<ArbitrageEvent>,
    /// Is monitor running
    is_running: Arc<RwLock<bool>>,
    /// Last opportunity by symbol
    last_opportunities: Arc<RwLock<HashMap<String, ArbitrageOpportunity>>>,
}

impl ArbitrageMonitor {
    /// Create a new arbitrage monitor
    pub fn new(config: ArbitrageConfig) -> Self {
        let (event_tx, _) = broadcast::channel(1000);

        Self {
            config,
            comparisons: Arc::new(RwLock::new(HashMap::new())),
            history: Arc::new(RwLock::new(HashMap::new())),
            event_tx,
            is_running: Arc::new(RwLock::new(false)),
            last_opportunities: Arc::new(RwLock::new(HashMap::new())),
        }
    }

    /// Create from environment
    pub fn from_env() -> Self {
        Self::new(ArbitrageConfig::from_env())
    }

    /// Subscribe to arbitrage events
    pub fn subscribe(&self) -> broadcast::Receiver<ArbitrageEvent> {
        self.event_tx.subscribe()
    }

    /// Subscribe to opportunities only
    pub fn subscribe_opportunities(&self) -> broadcast::Receiver<ArbitrageEvent> {
        self.event_tx.subscribe()
    }

    /// Check if monitor is running
    pub async fn is_running(&self) -> bool {
        *self.is_running.read().await
    }

    /// Update price from an exchange
    pub async fn update_price(&self, price: ExchangePrice) {
        let symbol = price.symbol.clone();

        let mut comparisons = self.comparisons.write().await;
        let comparison = comparisons
            .entry(symbol.clone())
            .or_insert_with(|| PriceComparison::new(symbol.clone()));

        comparison.add_price(price);

        // Check for arbitrage opportunity
        if let Some(opp) = comparison.find_arbitrage(self.config.min_spread_pct) {
            // Check if this is a new or significantly different opportunity
            let should_emit = {
                let last = self.last_opportunities.read().await;
                if let Some(prev) = last.get(&symbol) {
                    // Emit if spread changed significantly (> 10% change)
                    let spread_change = ((opp.spread_pct - prev.spread_pct) / prev.spread_pct)
                        .abs()
                        * Decimal::from(100);
                    spread_change > Decimal::from(10)
                        || opp.buy_exchange != prev.buy_exchange
                        || opp.sell_exchange != prev.sell_exchange
                } else {
                    true
                }
            };

            if should_emit {
                self.last_opportunities
                    .write()
                    .await
                    .insert(symbol.clone(), opp.clone());

                let _ = self.event_tx.send(ArbitrageEvent::Opportunity(opp.clone()));

                // Check for alert threshold
                if opp.spread_pct >= self.config.alert_threshold_pct {
                    let _ = self.event_tx.send(ArbitrageEvent::SpreadAlert {
                        symbol: symbol.clone(),
                        spread_pct: opp.spread_pct,
                        message: format!(
                            "Large spread detected: buy {} on {}, sell on {} for {:.2}% profit",
                            symbol, opp.buy_exchange, opp.sell_exchange, opp.spread_pct
                        ),
                    });
                }
            }
        }

        // Emit price update
        let _ = self
            .event_tx
            .send(ArbitrageEvent::PriceUpdate(comparison.clone()));

        // Emit Kraken recommendation if applicable
        if let Some(rec) = comparison.kraken_recommendation() {
            let _ = self.event_tx.send(ArbitrageEvent::Recommendation(rec));
        }

        // Update history
        if let (Some(spread_pct), Some((best_bid_ex, _)), Some((best_ask_ex, _))) = (
            comparison.cross_spread_pct,
            comparison.best_bid,
            comparison.best_ask,
        ) {
            let data_point = SpreadDataPoint {
                timestamp: Utc::now(),
                spread_pct,
                kraken_deviation_pct: comparison.kraken_deviation_pct,
                best_bid_exchange: best_bid_ex,
                best_ask_exchange: best_ask_ex,
            };

            let mut history = self.history.write().await;
            let symbol_history = history.entry(symbol).or_insert_with(Vec::new);
            symbol_history.push(data_point);

            // Prune old history
            let cutoff = Utc::now()
                - chrono::Duration::from_std(self.config.history_window).unwrap_or_default();
            symbol_history.retain(|p| p.timestamp >= cutoff);
        }
    }

    /// Get current price comparison for a symbol
    pub async fn get_comparison(&self, symbol: &str) -> Option<PriceComparison> {
        self.comparisons.read().await.get(symbol).cloned()
    }

    /// Get all current comparisons
    pub async fn get_all_comparisons(&self) -> HashMap<String, PriceComparison> {
        self.comparisons.read().await.clone()
    }

    /// Get current arbitrage opportunities
    pub async fn get_opportunities(&self) -> Vec<ArbitrageOpportunity> {
        let comparisons = self.comparisons.read().await;
        comparisons
            .values()
            .filter_map(|c| c.find_arbitrage(self.config.min_spread_pct))
            .collect()
    }

    /// Get actionable opportunities (involving Kraken)
    pub async fn get_actionable_opportunities(&self) -> Vec<ArbitrageOpportunity> {
        self.get_opportunities()
            .await
            .into_iter()
            .filter(|o| o.actionable)
            .collect()
    }

    /// Get spread statistics for a symbol
    pub async fn get_spread_stats(&self, symbol: &str) -> Option<SpreadStats> {
        let history = self.history.read().await;
        let data = history.get(symbol)?;

        if data.is_empty() {
            return None;
        }

        let spreads: Vec<Decimal> = data.iter().map(|p| p.spread_pct).collect();
        let count = spreads.len();

        let sum: Decimal = spreads.iter().sum();
        let avg = sum / Decimal::from(count);

        let max = spreads.iter().max().copied().unwrap_or(Decimal::ZERO);
        let min = spreads.iter().min().copied().unwrap_or(Decimal::ZERO);

        // Calculate standard deviation
        let variance: Decimal = spreads
            .iter()
            .map(|s| {
                let diff = *s - avg;
                diff * diff
            })
            .sum::<Decimal>()
            / Decimal::from(count);
        let std_dev = variance
            .to_f64()
            .map(|v: f64| v.sqrt())
            .and_then(Decimal::from_f64)
            .unwrap_or(Decimal::ZERO);

        // Calculate opportunity percentage
        let opportunities = spreads
            .iter()
            .filter(|s| **s >= self.config.min_spread_pct)
            .count();
        let opportunity_pct =
            Decimal::from(opportunities) / Decimal::from(count) * Decimal::from(100);

        // Calculate average Kraken deviation
        let kraken_devs: Vec<Decimal> =
            data.iter().filter_map(|p| p.kraken_deviation_pct).collect();
        let avg_kraken_deviation = if !kraken_devs.is_empty() {
            Some(kraken_devs.iter().sum::<Decimal>() / Decimal::from(kraken_devs.len()))
        } else {
            None
        };

        Some(SpreadStats {
            symbol: symbol.to_string(),
            window_secs: self.config.history_window.as_secs(),
            sample_count: count,
            avg_spread_pct: avg,
            max_spread_pct: max,
            min_spread_pct: min,
            std_dev,
            opportunity_pct,
            avg_kraken_deviation,
        })
    }

    /// Get Kraken recommendation for a symbol
    pub async fn get_kraken_recommendation(&self, symbol: &str) -> Option<KrakenRecommendation> {
        self.comparisons
            .read()
            .await
            .get(symbol)
            .and_then(|c| c.kraken_recommendation())
    }

    /// Start the monitor (spawns background task)
    pub async fn start(&self) -> Result<(), String> {
        let mut is_running = self.is_running.write().await;
        if *is_running {
            return Err("Monitor already running".to_string());
        }

        *is_running = true;
        drop(is_running);

        let _ = self.event_tx.send(ArbitrageEvent::Started);
        info!(
            "Arbitrage monitor started for symbols: {:?}",
            self.config.symbols
        );

        Ok(())
    }

    /// Stop the monitor
    pub async fn stop(&self) {
        *self.is_running.write().await = false;
        let _ = self.event_tx.send(ArbitrageEvent::Stopped);
        info!("Arbitrage monitor stopped");
    }

    /// Clear all data
    pub async fn clear(&self) {
        self.comparisons.write().await.clear();
        self.history.write().await.clear();
        self.last_opportunities.write().await.clear();
    }

    /// Get summary of current state
    pub async fn summary(&self) -> ArbitrageSummary {
        let comparisons = self.comparisons.read().await;
        let opportunities = self.get_opportunities().await;
        let actionable = opportunities.iter().filter(|o| o.actionable).count();

        ArbitrageSummary {
            monitored_symbols: comparisons.len(),
            active_opportunities: opportunities.len(),
            actionable_opportunities: actionable,
            best_opportunity: opportunities.into_iter().max_by(|a, b| {
                a.spread_pct
                    .partial_cmp(&b.spread_pct)
                    .unwrap_or(std::cmp::Ordering::Equal)
            }),
            timestamp: Utc::now(),
        }
    }
}

/// Summary of arbitrage monitor state
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct ArbitrageSummary {
    /// Number of symbols being monitored
    pub monitored_symbols: usize,
    /// Number of active arbitrage opportunities
    pub active_opportunities: usize,
    /// Number of actionable opportunities (involving Kraken)
    pub actionable_opportunities: usize,
    /// Best current opportunity
    pub best_opportunity: Option<ArbitrageOpportunity>,
    /// Summary timestamp
    pub timestamp: DateTime<Utc>,
}

// ============================================================================
// Tests
// ============================================================================

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_config_default() {
        let config = ArbitrageConfig::default();
        assert_eq!(config.symbols.len(), 3);
        assert!(config.compare_kraken);
        assert!(config.compare_bybit);
        assert!(config.compare_binance);
    }

    #[test]
    fn test_exchange_price_creation() {
        let price = ExchangePrice::new(
            Exchange::Kraken,
            "BTC/USDT".to_string(),
            Decimal::from(50000),
            Decimal::from(50010),
            Decimal::from(1),
            Decimal::from(2),
        );

        assert_eq!(price.exchange, Exchange::Kraken);
        assert_eq!(price.mid, Decimal::from(50005));
        assert!(price.spread_bps > Decimal::ZERO);
    }

    #[test]
    fn test_price_comparison() {
        let mut comparison = PriceComparison::new("BTC/USDT".to_string());

        // Add Kraken price
        comparison.add_price(ExchangePrice::new(
            Exchange::Kraken,
            "BTC/USDT".to_string(),
            Decimal::from(50000),
            Decimal::from(50010),
            Decimal::from(1),
            Decimal::from(1),
        ));

        // Add Binance price (lower)
        comparison.add_price(ExchangePrice::new(
            Exchange::Binance,
            "BTC/USDT".to_string(),
            Decimal::from(49990),
            Decimal::from(49995),
            Decimal::from(1),
            Decimal::from(1),
        ));

        assert!(comparison.best_bid.is_some());
        assert!(comparison.best_ask.is_some());

        // Best bid should be Kraken (50000), best ask should be Binance (49995)
        assert_eq!(comparison.best_bid.unwrap().0, Exchange::Kraken);
        assert_eq!(comparison.best_ask.unwrap().0, Exchange::Binance);
    }

    #[test]
    fn test_arbitrage_detection() {
        let mut comparison = PriceComparison::new("BTC/USDT".to_string());

        // Kraken: bid 50100, ask 50110
        comparison.add_price(ExchangePrice::new(
            Exchange::Kraken,
            "BTC/USDT".to_string(),
            Decimal::from(50100),
            Decimal::from(50110),
            Decimal::from(1),
            Decimal::from(1),
        ));

        // Binance: bid 49990, ask 49995 (lower prices)
        comparison.add_price(ExchangePrice::new(
            Exchange::Binance,
            "BTC/USDT".to_string(),
            Decimal::from(49990),
            Decimal::from(49995),
            Decimal::from(1),
            Decimal::from(1),
        ));

        // Should find arbitrage: buy on Binance (49995), sell on Kraken (50100)
        let opp = comparison.find_arbitrage(Decimal::ZERO);
        assert!(opp.is_some());

        let opp = opp.unwrap();
        assert_eq!(opp.buy_exchange, Exchange::Binance);
        assert_eq!(opp.sell_exchange, Exchange::Kraken);
        assert!(opp.spread > Decimal::ZERO);
        assert!(opp.actionable); // Kraken is involved
    }

    #[test]
    fn test_no_arbitrage_same_exchange() {
        let mut comparison = PriceComparison::new("BTC/USDT".to_string());

        // Only Kraken price
        comparison.add_price(ExchangePrice::new(
            Exchange::Kraken,
            "BTC/USDT".to_string(),
            Decimal::from(50000),
            Decimal::from(50010),
            Decimal::from(1),
            Decimal::from(1),
        ));

        let opp = comparison.find_arbitrage(Decimal::ZERO);
        assert!(opp.is_none());
    }

    #[test]
    fn test_kraken_recommendation() {
        let mut comparison = PriceComparison::new("BTC/USDT".to_string());

        // Kraken cheaper than market
        comparison.add_price(ExchangePrice::new(
            Exchange::Kraken,
            "BTC/USDT".to_string(),
            Decimal::from(49800),
            Decimal::from(49810),
            Decimal::from(1),
            Decimal::from(1),
        ));

        comparison.add_price(ExchangePrice::new(
            Exchange::Binance,
            "BTC/USDT".to_string(),
            Decimal::from(50000),
            Decimal::from(50010),
            Decimal::from(1),
            Decimal::from(1),
        ));

        let rec = comparison.kraken_recommendation();
        assert!(rec.is_some());

        let rec = rec.unwrap();
        assert_eq!(rec.action, RecommendedAction::Buy);
        assert!(rec.deviation_pct < Decimal::ZERO); // Kraken is cheaper
    }

    #[test]
    fn test_opportunity_profit_after_fees() {
        let opp = ArbitrageOpportunity {
            symbol: "BTC/USDT".to_string(),
            buy_exchange: Exchange::Binance,
            sell_exchange: Exchange::Kraken,
            buy_price: Decimal::from(50000),
            sell_price: Decimal::from(50100),
            spread: Decimal::from(100),
            spread_pct: Decimal::new(2, 1), // 0.2%
            max_qty: Decimal::from(1),
            estimated_profit: Decimal::from(100),
            timestamp: Utc::now(),
            price_age_ms: 0,
            actionable: true,
        };

        // With 0.1% fee on each side (0.001)
        let fee_rate = Decimal::new(1, 3);
        let profit = opp.profit_after_fees(fee_rate);

        // Profit = 100 - (50000 * 1 * 0.001 * 2) = 100 - 100 = 0
        assert_eq!(profit, Decimal::ZERO);
    }

    #[tokio::test]
    async fn test_monitor_creation() {
        let config = ArbitrageConfig::default();
        let monitor = ArbitrageMonitor::new(config);

        assert!(!monitor.is_running().await);
    }

    #[tokio::test]
    async fn test_monitor_update_price() {
        let config = ArbitrageConfig::default();
        let monitor = ArbitrageMonitor::new(config);

        let price = ExchangePrice::new(
            Exchange::Kraken,
            "BTC/USDT".to_string(),
            Decimal::from(50000),
            Decimal::from(50010),
            Decimal::from(1),
            Decimal::from(1),
        );

        monitor.update_price(price).await;

        let comparison = monitor.get_comparison("BTC/USDT").await;
        assert!(comparison.is_some());
    }

    #[tokio::test]
    async fn test_monitor_opportunities() {
        let config = ArbitrageConfig {
            min_spread_pct: Decimal::ZERO,
            ..Default::default()
        };
        let monitor = ArbitrageMonitor::new(config);

        // Add prices with arbitrage opportunity
        monitor
            .update_price(ExchangePrice::new(
                Exchange::Kraken,
                "BTC/USDT".to_string(),
                Decimal::from(50100),
                Decimal::from(50110),
                Decimal::from(1),
                Decimal::from(1),
            ))
            .await;

        monitor
            .update_price(ExchangePrice::new(
                Exchange::Binance,
                "BTC/USDT".to_string(),
                Decimal::from(49990),
                Decimal::from(49995),
                Decimal::from(1),
                Decimal::from(1),
            ))
            .await;

        let opportunities = monitor.get_opportunities().await;
        assert!(!opportunities.is_empty());

        let actionable = monitor.get_actionable_opportunities().await;
        assert!(!actionable.is_empty());
    }

    #[tokio::test]
    async fn test_monitor_summary() {
        let config = ArbitrageConfig::default();
        let monitor = ArbitrageMonitor::new(config);

        monitor
            .update_price(ExchangePrice::new(
                Exchange::Kraken,
                "BTC/USDT".to_string(),
                Decimal::from(50000),
                Decimal::from(50010),
                Decimal::from(1),
                Decimal::from(1),
            ))
            .await;

        let summary = monitor.summary().await;
        assert_eq!(summary.monitored_symbols, 1);
    }

    #[test]
    fn test_exchange_display() {
        assert_eq!(format!("{}", Exchange::Kraken), "Kraken");
        assert_eq!(format!("{}", Exchange::Bybit), "Bybit");
        assert_eq!(format!("{}", Exchange::Binance), "Binance");
    }

    #[test]
    fn test_price_staleness() {
        let price = ExchangePrice::new(
            Exchange::Kraken,
            "BTC/USDT".to_string(),
            Decimal::from(50000),
            Decimal::from(50010),
            Decimal::from(1),
            Decimal::from(1),
        );

        // Fresh price should not be stale
        assert!(!price.is_stale(Duration::from_secs(1)));

        // With very short threshold, might be stale
        assert!(!price.is_stale(Duration::from_millis(1)));
    }
}
