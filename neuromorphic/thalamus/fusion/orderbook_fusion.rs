//! Order book fusion
//!
//! Part of the Thalamus region
//! Component: fusion
//!
//! Consolidates order book snapshots from multiple venues into a unified
//! view of market depth, computing a fair mid-price, detecting imbalances,
//! and tracking liquidity distribution.
//!
//! ## Features
//!
//! - **Multi-venue consolidation**: Merges top-of-book and depth snapshots
//!   from an arbitrary number of named venues into a single composite book.
//! - **Weighted mid-price**: Computes a size-weighted mid-price across all
//!   active venues so that venues with tighter spreads and deeper liquidity
//!   contribute more to the fair-value estimate.
//! - **Imbalance detection**: Measures bid/ask volume imbalance at
//!   configurable depth levels and flags significant directional pressure.
//! - **Spread monitoring**: Tracks consolidated best-bid/best-ask spread
//!   and flags abnormal widenings.
//! - **Depth aggregation**: Aggregates liquidity across venues at each
//!   price level to form a consolidated depth view.
//! - **EMA smoothing**: Smooths the mid-price and imbalance signals to
//!   reduce tick-level noise.
//! - **Per-venue statistics**: Tracks each venue's contribution to the
//!   consolidated book (spread, depth share, staleness).

use std::collections::{BTreeMap, HashMap, VecDeque};

use crate::common::{Error, Result};

// ---------------------------------------------------------------------------
// Configuration
// ---------------------------------------------------------------------------

/// Configuration for the order book fusion engine.
#[derive(Debug, Clone)]
pub struct OrderbookFusionConfig {
    /// EMA decay factor for smoothed mid-price and imbalance (0, 1).
    /// Closer to 1 → more smoothing / slower reaction.
    pub ema_decay: f64,
    /// Number of top price levels (per side) to consider when computing
    /// imbalance.  For example, 5 means the top-5 bid levels vs top-5 ask
    /// levels.
    pub imbalance_depth: usize,
    /// Imbalance threshold in (0, 1] above which `FusedOrderbook::is_imbalanced`
    /// returns `true`.  Imbalance is measured as
    /// `|bid_depth - ask_depth| / (bid_depth + ask_depth)`.
    pub imbalance_threshold: f64,
    /// Maximum age (seconds) before a venue's snapshot is considered stale
    /// and excluded from the consolidated book.
    pub max_staleness_secs: f64,
    /// Minimum number of active (non-stale) venues required to produce a
    /// fused snapshot.
    pub min_venues: usize,
    /// Maximum number of recent fused snapshots kept for windowed statistics.
    pub window_size: usize,
    /// Spread-widening alert threshold expressed as a multiple of the
    /// EMA-smoothed spread.  For example, 3.0 means the current spread
    /// must exceed 3× the smoothed spread to trigger `is_wide_spread`.
    pub spread_alert_multiplier: f64,
    /// Price tick size for aggregating depth across venues.  All prices are
    /// rounded to the nearest multiple of this value before merging.
    /// Set to 0.0 to disable rounding (exact price matching only).
    pub price_tick: f64,
    /// Maximum number of consolidated depth levels to retain per side.
    pub max_depth_levels: usize,
}

impl Default for OrderbookFusionConfig {
    fn default() -> Self {
        Self {
            ema_decay: 0.85,
            imbalance_depth: 5,
            imbalance_threshold: 0.4,
            max_staleness_secs: 5.0,
            min_venues: 1,
            window_size: 500,
            spread_alert_multiplier: 3.0,
            price_tick: 0.01,
            max_depth_levels: 50,
        }
    }
}

// ---------------------------------------------------------------------------
// Input / Output types
// ---------------------------------------------------------------------------

/// A single price level in an order book.
#[derive(Debug, Clone, Copy)]
pub struct PriceLevel {
    /// Price at this level.
    pub price: f64,
    /// Total size (quantity) resting at this price.
    pub size: f64,
}

/// An order book snapshot from a single venue.
#[derive(Debug, Clone)]
pub struct OrderbookSnapshot {
    /// Venue identifier (e.g. "binance", "coinbase").
    pub venue_id: String,
    /// Bid levels sorted by price descending (best bid first).
    pub bids: Vec<PriceLevel>,
    /// Ask levels sorted by price ascending (best ask first).
    pub asks: Vec<PriceLevel>,
    /// Timestamp of the snapshot (seconds since epoch or monotonic ref).
    pub timestamp: f64,
}

/// The fused (consolidated) order book output.
#[derive(Debug, Clone)]
pub struct FusedOrderbook {
    /// Size-weighted mid-price across all active venues.
    pub mid_price: f64,
    /// EMA-smoothed mid-price.
    pub smoothed_mid_price: f64,
    /// Consolidated best bid price.
    pub best_bid: f64,
    /// Consolidated best ask price.
    pub best_ask: f64,
    /// Consolidated spread (best_ask - best_bid).
    pub spread: f64,
    /// EMA-smoothed spread.
    pub smoothed_spread: f64,
    /// Whether the current spread is abnormally wide.
    pub is_wide_spread: bool,
    /// Bid/ask imbalance in [-1, 1].
    /// Positive → bid-heavy (buying pressure), negative → ask-heavy.
    pub imbalance: f64,
    /// EMA-smoothed imbalance.
    pub smoothed_imbalance: f64,
    /// Whether the imbalance exceeds the configured threshold.
    pub is_imbalanced: bool,
    /// Total bid depth (sum of all bid sizes across venues).
    pub total_bid_depth: f64,
    /// Total ask depth (sum of all ask sizes across venues).
    pub total_ask_depth: f64,
    /// Number of active (non-stale) venues contributing.
    pub active_venues: usize,
    /// Timestamp of the most recent snapshot used.
    pub latest_timestamp: f64,
    /// Consolidated bid levels (aggregated across venues, sorted descending).
    pub consolidated_bids: Vec<PriceLevel>,
    /// Consolidated ask levels (aggregated across venues, sorted ascending).
    pub consolidated_asks: Vec<PriceLevel>,
}

/// Per-venue statistics.
#[derive(Debug, Clone)]
pub struct VenueBookStats {
    /// Total snapshots received from this venue.
    pub snapshots: usize,
    /// Most recent best bid.
    pub last_best_bid: f64,
    /// Most recent best ask.
    pub last_best_ask: f64,
    /// Most recent spread.
    pub last_spread: f64,
    /// Most recent timestamp.
    pub last_timestamp: f64,
    /// Total bid depth in the most recent snapshot.
    pub last_bid_depth: f64,
    /// Total ask depth in the most recent snapshot.
    pub last_ask_depth: f64,
}

/// Aggregate statistics for the fusion engine.
#[derive(Debug, Clone, Default)]
pub struct OrderbookFusionStats {
    /// Total snapshots ingested.
    pub total_snapshots: usize,
    /// Total fused outputs produced.
    pub total_fusions: usize,
    /// Number of fusions where imbalance was detected.
    pub imbalance_count: usize,
    /// Number of fusions where wide spread was detected.
    pub wide_spread_count: usize,
    /// Number of fusions skipped due to insufficient venues.
    pub insufficient_venue_count: usize,
    /// Number of distinct venues seen.
    pub distinct_venues: usize,
}

// ---------------------------------------------------------------------------
// Internal types
// ---------------------------------------------------------------------------

#[derive(Debug, Clone)]
struct VenueRecord {
    bids: Vec<PriceLevel>,
    asks: Vec<PriceLevel>,
    timestamp: f64,
    snapshot_count: usize,
}

// ---------------------------------------------------------------------------
// OrderbookFusion
// ---------------------------------------------------------------------------

/// Multi-venue order book fusion engine.
///
/// Call [`ingest`] to feed snapshots from individual venues, then call
/// [`fuse`] to obtain the consolidated order book at a given point in time.
pub struct OrderbookFusion {
    config: OrderbookFusionConfig,
    /// Most recent snapshot per venue.
    venues: HashMap<String, VenueRecord>,
    /// EMA state for mid-price.
    ema_mid: f64,
    ema_mid_initialized: bool,
    /// EMA state for spread.
    ema_spread: f64,
    ema_spread_initialized: bool,
    /// EMA state for imbalance.
    ema_imbalance: f64,
    ema_imbalance_initialized: bool,
    /// Windowed history of mid-prices.
    history: VecDeque<f64>,
    /// Running statistics.
    stats: OrderbookFusionStats,
}

impl Default for OrderbookFusion {
    fn default() -> Self {
        Self::new()
    }
}

impl OrderbookFusion {
    /// Create a new instance with default configuration.
    pub fn new() -> Self {
        Self::with_config(OrderbookFusionConfig::default())
    }

    /// Create a new instance with the given configuration.
    pub fn with_config(config: OrderbookFusionConfig) -> Self {
        Self {
            venues: HashMap::new(),
            ema_mid: 0.0,
            ema_mid_initialized: false,
            ema_spread: 0.0,
            ema_spread_initialized: false,
            ema_imbalance: 0.0,
            ema_imbalance_initialized: false,
            history: VecDeque::with_capacity(config.window_size),
            stats: OrderbookFusionStats::default(),
            config,
        }
    }

    /// Validate configuration parameters.
    pub fn process(&self) -> Result<()> {
        if self.config.ema_decay <= 0.0 || self.config.ema_decay >= 1.0 {
            return Err(Error::InvalidInput("ema_decay must be in (0, 1)".into()));
        }
        if self.config.imbalance_depth == 0 {
            return Err(Error::InvalidInput("imbalance_depth must be > 0".into()));
        }
        if self.config.imbalance_threshold <= 0.0 || self.config.imbalance_threshold > 1.0 {
            return Err(Error::InvalidInput(
                "imbalance_threshold must be in (0, 1]".into(),
            ));
        }
        if self.config.max_staleness_secs <= 0.0 {
            return Err(Error::InvalidInput("max_staleness_secs must be > 0".into()));
        }
        if self.config.window_size == 0 {
            return Err(Error::InvalidInput("window_size must be > 0".into()));
        }
        if self.config.spread_alert_multiplier <= 0.0 {
            return Err(Error::InvalidInput(
                "spread_alert_multiplier must be > 0".into(),
            ));
        }
        if self.config.price_tick < 0.0 {
            return Err(Error::InvalidInput("price_tick must be >= 0".into()));
        }
        if self.config.max_depth_levels == 0 {
            return Err(Error::InvalidInput("max_depth_levels must be > 0".into()));
        }
        Ok(())
    }

    // -- Ingestion ---------------------------------------------------------

    /// Ingest a single order book snapshot from a venue.
    pub fn ingest(&mut self, snapshot: &OrderbookSnapshot) -> Result<()> {
        // Validate bids are sorted descending
        for i in 1..snapshot.bids.len() {
            if snapshot.bids[i].price > snapshot.bids[i - 1].price {
                return Err(Error::InvalidInput(
                    "bids must be sorted by price descending".into(),
                ));
            }
        }
        // Validate asks are sorted ascending
        for i in 1..snapshot.asks.len() {
            if snapshot.asks[i].price < snapshot.asks[i - 1].price {
                return Err(Error::InvalidInput(
                    "asks must be sorted by price ascending".into(),
                ));
            }
        }
        // Validate no negative sizes or prices
        for level in snapshot.bids.iter().chain(snapshot.asks.iter()) {
            if level.price < 0.0 {
                return Err(Error::InvalidInput("price must be >= 0".into()));
            }
            if level.size < 0.0 {
                return Err(Error::InvalidInput("size must be >= 0".into()));
            }
        }

        self.stats.total_snapshots += 1;

        if !self.venues.contains_key(&snapshot.venue_id) {
            self.stats.distinct_venues += 1;
        }

        let prev_count = self
            .venues
            .get(&snapshot.venue_id)
            .map(|r| r.snapshot_count)
            .unwrap_or(0);

        self.venues.insert(
            snapshot.venue_id.clone(),
            VenueRecord {
                bids: snapshot.bids.clone(),
                asks: snapshot.asks.clone(),
                timestamp: snapshot.timestamp,
                snapshot_count: prev_count + 1,
            },
        );

        Ok(())
    }

    // -- Fusion ------------------------------------------------------------

    /// Produce the fused (consolidated) order book at time `now`.
    ///
    /// Returns `None` if fewer than `min_venues` non-stale venues are
    /// available.
    pub fn fuse(&mut self, now: f64) -> Option<FusedOrderbook> {
        // Collect active venues
        let active_ids: Vec<String> = self
            .venues
            .iter()
            .filter(|(_, rec)| (now - rec.timestamp).max(0.0) <= self.config.max_staleness_secs)
            .map(|(id, _)| id.clone())
            .collect();

        if active_ids.len() < self.config.min_venues {
            self.stats.insufficient_venue_count += 1;
            return None;
        }

        let mut latest_ts = f64::NEG_INFINITY;

        // Build consolidated depth using BTreeMap for automatic sorting
        // Bids: keyed by negative price for descending order
        let mut bid_map: BTreeMap<i64, f64> = BTreeMap::new();
        let mut ask_map: BTreeMap<i64, f64> = BTreeMap::new();

        // For weighted mid-price computation
        let mut weighted_mid_sum = 0.0_f64;
        let mut weight_sum = 0.0_f64;

        for id in &active_ids {
            let rec = &self.venues[id];
            if rec.timestamp > latest_ts {
                latest_ts = rec.timestamp;
            }

            // Compute this venue's best bid/ask for mid-price weighting
            let venue_best_bid = rec.bids.first().map(|l| l.price).unwrap_or(0.0);
            let venue_best_ask = rec.asks.first().map(|l| l.price).unwrap_or(0.0);
            let venue_best_bid_size = rec.bids.first().map(|l| l.size).unwrap_or(0.0);
            let venue_best_ask_size = rec.asks.first().map(|l| l.size).unwrap_or(0.0);

            if venue_best_bid > 0.0 && venue_best_ask > 0.0 {
                let venue_mid = (venue_best_bid + venue_best_ask) / 2.0;
                // Weight by the harmonic mean of top-of-book sizes (rewards
                // venues with depth on both sides) divided by spread
                let venue_spread = venue_best_ask - venue_best_bid;
                let depth_weight = if venue_best_bid_size > 0.0 && venue_best_ask_size > 0.0 {
                    2.0 * venue_best_bid_size * venue_best_ask_size
                        / (venue_best_bid_size + venue_best_ask_size)
                } else {
                    (venue_best_bid_size + venue_best_ask_size) / 2.0
                };
                let spread_factor = if venue_spread > 0.0 {
                    1.0 / venue_spread
                } else {
                    1.0
                };
                let w = depth_weight * spread_factor;
                weighted_mid_sum += w * venue_mid;
                weight_sum += w;
            }

            // Merge bids
            for level in &rec.bids {
                let tick_price = self.round_to_tick(level.price);
                let key = self.price_to_key(tick_price);
                *bid_map.entry(key).or_insert(0.0) += level.size;
            }

            // Merge asks
            for level in &rec.asks {
                let tick_price = self.round_to_tick(level.price);
                let key = self.price_to_key(tick_price);
                *ask_map.entry(key).or_insert(0.0) += level.size;
            }
        }

        // Build consolidated levels
        // Bids: descending by price → iterate BTreeMap in reverse
        let mut consolidated_bids: Vec<PriceLevel> = bid_map
            .iter()
            .rev()
            .take(self.config.max_depth_levels)
            .map(|(&key, &size)| PriceLevel {
                price: self.key_to_price(key),
                size,
            })
            .collect();

        // Asks: ascending by price → iterate BTreeMap in order
        let mut consolidated_asks: Vec<PriceLevel> = ask_map
            .iter()
            .take(self.config.max_depth_levels)
            .map(|(&key, &size)| PriceLevel {
                price: self.key_to_price(key),
                size,
            })
            .collect();

        // Ensure correct sort order
        consolidated_bids.sort_by(|a, b| b.price.partial_cmp(&a.price).unwrap());
        consolidated_asks.sort_by(|a, b| a.price.partial_cmp(&b.price).unwrap());

        // Best bid / best ask
        let best_bid = consolidated_bids.first().map(|l| l.price).unwrap_or(0.0);
        let best_ask = consolidated_asks.first().map(|l| l.price).unwrap_or(0.0);
        let spread = if best_ask > 0.0 && best_bid > 0.0 {
            best_ask - best_bid
        } else {
            0.0
        };

        // Mid-price
        let mid_price = if weight_sum > 0.0 {
            weighted_mid_sum / weight_sum
        } else if best_bid > 0.0 && best_ask > 0.0 {
            (best_bid + best_ask) / 2.0
        } else {
            0.0
        };

        // Total depths
        let total_bid_depth: f64 = consolidated_bids.iter().map(|l| l.size).sum();
        let total_ask_depth: f64 = consolidated_asks.iter().map(|l| l.size).sum();

        // Imbalance at configured depth
        let bid_depth_at_level: f64 = consolidated_bids
            .iter()
            .take(self.config.imbalance_depth)
            .map(|l| l.size)
            .sum();
        let ask_depth_at_level: f64 = consolidated_asks
            .iter()
            .take(self.config.imbalance_depth)
            .map(|l| l.size)
            .sum();
        let total_at_level = bid_depth_at_level + ask_depth_at_level;
        let imbalance = if total_at_level > 0.0 {
            (bid_depth_at_level - ask_depth_at_level) / total_at_level
        } else {
            0.0
        };

        // EMA smoothing — mid-price
        let smoothed_mid = if self.ema_mid_initialized {
            let s =
                self.config.ema_decay * self.ema_mid + (1.0 - self.config.ema_decay) * mid_price;
            self.ema_mid = s;
            s
        } else {
            self.ema_mid = mid_price;
            self.ema_mid_initialized = true;
            mid_price
        };

        // EMA smoothing — spread
        let smoothed_spread = if self.ema_spread_initialized {
            let s =
                self.config.ema_decay * self.ema_spread + (1.0 - self.config.ema_decay) * spread;
            self.ema_spread = s;
            s
        } else {
            self.ema_spread = spread;
            self.ema_spread_initialized = true;
            spread
        };

        // EMA smoothing — imbalance
        let smoothed_imbalance = if self.ema_imbalance_initialized {
            let s = self.config.ema_decay * self.ema_imbalance
                + (1.0 - self.config.ema_decay) * imbalance;
            self.ema_imbalance = s;
            s
        } else {
            self.ema_imbalance = imbalance;
            self.ema_imbalance_initialized = true;
            imbalance
        };

        // Alerts
        let is_wide_spread = self.ema_spread_initialized
            && smoothed_spread > 0.0
            && spread > self.config.spread_alert_multiplier * smoothed_spread;
        let is_imbalanced = imbalance.abs() > self.config.imbalance_threshold;

        // History
        self.history.push_back(mid_price);
        while self.history.len() > self.config.window_size {
            self.history.pop_front();
        }

        // Stats
        self.stats.total_fusions += 1;
        if is_imbalanced {
            self.stats.imbalance_count += 1;
        }
        if is_wide_spread {
            self.stats.wide_spread_count += 1;
        }

        Some(FusedOrderbook {
            mid_price,
            smoothed_mid_price: smoothed_mid,
            best_bid,
            best_ask,
            spread,
            smoothed_spread,
            is_wide_spread,
            imbalance,
            smoothed_imbalance,
            is_imbalanced,
            total_bid_depth,
            total_ask_depth,
            active_venues: active_ids.len(),
            latest_timestamp: latest_ts,
            consolidated_bids,
            consolidated_asks,
        })
    }

    // -- Price tick helpers -------------------------------------------------

    fn round_to_tick(&self, price: f64) -> f64 {
        if self.config.price_tick <= 0.0 {
            return price;
        }
        (price / self.config.price_tick).round() * self.config.price_tick
    }

    /// Convert a price to an integer key for BTreeMap ordering.
    /// We multiply by a large factor to preserve precision.
    fn price_to_key(&self, price: f64) -> i64 {
        // Use enough precision to handle typical tick sizes
        (price * 1_000_000.0).round() as i64
    }

    fn key_to_price(&self, key: i64) -> f64 {
        key as f64 / 1_000_000.0
    }

    // -- Accessors ---------------------------------------------------------

    /// Get per-venue statistics.
    pub fn venue_stats(&self) -> HashMap<String, VenueBookStats> {
        self.venues
            .iter()
            .map(|(id, rec)| {
                let best_bid = rec.bids.first().map(|l| l.price).unwrap_or(0.0);
                let best_ask = rec.asks.first().map(|l| l.price).unwrap_or(0.0);
                let bid_depth: f64 = rec.bids.iter().map(|l| l.size).sum();
                let ask_depth: f64 = rec.asks.iter().map(|l| l.size).sum();
                (
                    id.clone(),
                    VenueBookStats {
                        snapshots: rec.snapshot_count,
                        last_best_bid: best_bid,
                        last_best_ask: best_ask,
                        last_spread: if best_ask > 0.0 && best_bid > 0.0 {
                            best_ask - best_bid
                        } else {
                            0.0
                        },
                        last_timestamp: rec.timestamp,
                        last_bid_depth: bid_depth,
                        last_ask_depth: ask_depth,
                    },
                )
            })
            .collect()
    }

    /// Get aggregate statistics.
    pub fn stats(&self) -> &OrderbookFusionStats {
        &self.stats
    }

    /// Current number of distinct venues seen.
    pub fn venue_count(&self) -> usize {
        self.venues.len()
    }

    /// Windowed mean of recent mid-prices.
    pub fn windowed_mean(&self) -> Option<f64> {
        if self.history.is_empty() {
            return None;
        }
        let sum: f64 = self.history.iter().sum();
        Some(sum / self.history.len() as f64)
    }

    /// Windowed standard deviation of recent mid-prices.
    pub fn windowed_std(&self) -> Option<f64> {
        if self.history.len() < 2 {
            return None;
        }
        let mean = self.windowed_mean().unwrap();
        let var: f64 = self.history.iter().map(|v| (v - mean).powi(2)).sum::<f64>()
            / (self.history.len() - 1) as f64;
        Some(var.sqrt())
    }

    /// Reset all state (venues, EMA, history, stats).
    pub fn reset(&mut self) {
        self.venues.clear();
        self.ema_mid = 0.0;
        self.ema_mid_initialized = false;
        self.ema_spread = 0.0;
        self.ema_spread_initialized = false;
        self.ema_imbalance = 0.0;
        self.ema_imbalance_initialized = false;
        self.history.clear();
        self.stats = OrderbookFusionStats::default();
    }
}

// ---------------------------------------------------------------------------
// Tests
// ---------------------------------------------------------------------------

#[cfg(test)]
mod tests {
    use super::*;

    fn level(price: f64, size: f64) -> PriceLevel {
        PriceLevel { price, size }
    }

    fn simple_snapshot(venue: &str, best_bid: f64, best_ask: f64, ts: f64) -> OrderbookSnapshot {
        OrderbookSnapshot {
            venue_id: venue.to_string(),
            bids: vec![
                level(best_bid, 10.0),
                level(best_bid - 1.0, 20.0),
                level(best_bid - 2.0, 30.0),
            ],
            asks: vec![
                level(best_ask, 10.0),
                level(best_ask + 1.0, 20.0),
                level(best_ask + 2.0, 30.0),
            ],
            timestamp: ts,
        }
    }

    fn asymmetric_snapshot(
        venue: &str,
        best_bid: f64,
        best_ask: f64,
        bid_sizes: Vec<f64>,
        ask_sizes: Vec<f64>,
        ts: f64,
    ) -> OrderbookSnapshot {
        let bids: Vec<PriceLevel> = bid_sizes
            .iter()
            .enumerate()
            .map(|(i, &s)| level(best_bid - i as f64, s))
            .collect();
        let asks: Vec<PriceLevel> = ask_sizes
            .iter()
            .enumerate()
            .map(|(i, &s)| level(best_ask + i as f64, s))
            .collect();
        OrderbookSnapshot {
            venue_id: venue.to_string(),
            bids,
            asks,
            timestamp: ts,
        }
    }

    fn default_config() -> OrderbookFusionConfig {
        OrderbookFusionConfig {
            ema_decay: 0.5,
            imbalance_depth: 3,
            imbalance_threshold: 0.3,
            max_staleness_secs: 10.0,
            min_venues: 1,
            window_size: 100,
            spread_alert_multiplier: 3.0,
            price_tick: 1.0,
            max_depth_levels: 50,
        }
    }

    fn default_fusion() -> OrderbookFusion {
        OrderbookFusion::with_config(default_config())
    }

    // -- Config validation -------------------------------------------------

    #[test]
    fn test_basic() {
        let instance = OrderbookFusion::new();
        assert!(instance.process().is_ok());
    }

    #[test]
    fn test_process_invalid_ema_decay() {
        let ob = OrderbookFusion::with_config(OrderbookFusionConfig {
            ema_decay: 0.0,
            ..Default::default()
        });
        assert!(ob.process().is_err());
    }

    #[test]
    fn test_process_invalid_ema_decay_high() {
        let ob = OrderbookFusion::with_config(OrderbookFusionConfig {
            ema_decay: 1.0,
            ..Default::default()
        });
        assert!(ob.process().is_err());
    }

    #[test]
    fn test_process_invalid_imbalance_depth() {
        let ob = OrderbookFusion::with_config(OrderbookFusionConfig {
            imbalance_depth: 0,
            ..Default::default()
        });
        assert!(ob.process().is_err());
    }

    #[test]
    fn test_process_invalid_imbalance_threshold_zero() {
        let ob = OrderbookFusion::with_config(OrderbookFusionConfig {
            imbalance_threshold: 0.0,
            ..Default::default()
        });
        assert!(ob.process().is_err());
    }

    #[test]
    fn test_process_invalid_imbalance_threshold_high() {
        let ob = OrderbookFusion::with_config(OrderbookFusionConfig {
            imbalance_threshold: 1.5,
            ..Default::default()
        });
        assert!(ob.process().is_err());
    }

    #[test]
    fn test_process_invalid_max_staleness() {
        let ob = OrderbookFusion::with_config(OrderbookFusionConfig {
            max_staleness_secs: 0.0,
            ..Default::default()
        });
        assert!(ob.process().is_err());
    }

    #[test]
    fn test_process_invalid_window_size() {
        let ob = OrderbookFusion::with_config(OrderbookFusionConfig {
            window_size: 0,
            ..Default::default()
        });
        assert!(ob.process().is_err());
    }

    #[test]
    fn test_process_invalid_spread_alert_multiplier() {
        let ob = OrderbookFusion::with_config(OrderbookFusionConfig {
            spread_alert_multiplier: 0.0,
            ..Default::default()
        });
        assert!(ob.process().is_err());
    }

    #[test]
    fn test_process_invalid_price_tick() {
        let ob = OrderbookFusion::with_config(OrderbookFusionConfig {
            price_tick: -1.0,
            ..Default::default()
        });
        assert!(ob.process().is_err());
    }

    #[test]
    fn test_process_invalid_max_depth_levels() {
        let ob = OrderbookFusion::with_config(OrderbookFusionConfig {
            max_depth_levels: 0,
            ..Default::default()
        });
        assert!(ob.process().is_err());
    }

    #[test]
    fn test_process_valid_threshold_at_one() {
        let ob = OrderbookFusion::with_config(OrderbookFusionConfig {
            imbalance_threshold: 1.0,
            ..Default::default()
        });
        assert!(ob.process().is_ok());
    }

    // -- Ingestion ---------------------------------------------------------

    #[test]
    fn test_ingest_valid() {
        let mut ob = default_fusion();
        assert!(
            ob.ingest(&simple_snapshot("binance", 100.0, 101.0, 100.0))
                .is_ok()
        );
        assert_eq!(ob.venue_count(), 1);
        assert_eq!(ob.stats().total_snapshots, 1);
    }

    #[test]
    fn test_ingest_multiple_venues() {
        let mut ob = default_fusion();
        ob.ingest(&simple_snapshot("binance", 100.0, 101.0, 100.0))
            .unwrap();
        ob.ingest(&simple_snapshot("coinbase", 100.0, 101.0, 100.0))
            .unwrap();
        ob.ingest(&simple_snapshot("kraken", 100.0, 101.0, 100.0))
            .unwrap();
        assert_eq!(ob.venue_count(), 3);
        assert_eq!(ob.stats().distinct_venues, 3);
    }

    #[test]
    fn test_ingest_replaces_old_snapshot() {
        let mut ob = default_fusion();
        ob.ingest(&simple_snapshot("binance", 100.0, 101.0, 100.0))
            .unwrap();
        ob.ingest(&simple_snapshot("binance", 200.0, 201.0, 110.0))
            .unwrap();
        assert_eq!(ob.venue_count(), 1);
        assert_eq!(ob.stats().total_snapshots, 2);

        let vs = ob.venue_stats();
        assert!((vs["binance"].last_best_bid - 200.0).abs() < 1e-12);
    }

    #[test]
    fn test_ingest_bids_wrong_order() {
        let mut ob = default_fusion();
        let snapshot = OrderbookSnapshot {
            venue_id: "bad".to_string(),
            bids: vec![level(99.0, 10.0), level(100.0, 10.0)], // ascending = wrong
            asks: vec![level(101.0, 10.0)],
            timestamp: 100.0,
        };
        assert!(ob.ingest(&snapshot).is_err());
    }

    #[test]
    fn test_ingest_asks_wrong_order() {
        let mut ob = default_fusion();
        let snapshot = OrderbookSnapshot {
            venue_id: "bad".to_string(),
            bids: vec![level(100.0, 10.0)],
            asks: vec![level(102.0, 10.0), level(101.0, 10.0)], // descending = wrong
            timestamp: 100.0,
        };
        assert!(ob.ingest(&snapshot).is_err());
    }

    #[test]
    fn test_ingest_negative_price() {
        let mut ob = default_fusion();
        let snapshot = OrderbookSnapshot {
            venue_id: "bad".to_string(),
            bids: vec![level(-1.0, 10.0)],
            asks: vec![level(101.0, 10.0)],
            timestamp: 100.0,
        };
        assert!(ob.ingest(&snapshot).is_err());
    }

    #[test]
    fn test_ingest_negative_size() {
        let mut ob = default_fusion();
        let snapshot = OrderbookSnapshot {
            venue_id: "bad".to_string(),
            bids: vec![level(100.0, -5.0)],
            asks: vec![level(101.0, 10.0)],
            timestamp: 100.0,
        };
        assert!(ob.ingest(&snapshot).is_err());
    }

    #[test]
    fn test_ingest_empty_book() {
        let mut ob = default_fusion();
        let snapshot = OrderbookSnapshot {
            venue_id: "empty".to_string(),
            bids: vec![],
            asks: vec![],
            timestamp: 100.0,
        };
        assert!(ob.ingest(&snapshot).is_ok());
    }

    #[test]
    fn test_ingest_equal_adjacent_prices_bids() {
        let mut ob = default_fusion();
        // Equal adjacent prices should be accepted (not strictly descending)
        let snapshot = OrderbookSnapshot {
            venue_id: "ok".to_string(),
            bids: vec![level(100.0, 10.0), level(100.0, 5.0)],
            asks: vec![level(101.0, 10.0)],
            timestamp: 100.0,
        };
        assert!(ob.ingest(&snapshot).is_ok());
    }

    // -- Fusion basics -----------------------------------------------------

    #[test]
    fn test_fuse_single_venue() {
        let mut ob = default_fusion();
        ob.ingest(&simple_snapshot("binance", 100.0, 102.0, 100.0))
            .unwrap();
        let fused = ob.fuse(100.0).unwrap();

        assert!((fused.best_bid - 100.0).abs() < 1e-12);
        assert!((fused.best_ask - 102.0).abs() < 1e-12);
        assert!((fused.spread - 2.0).abs() < 1e-12);
        assert_eq!(fused.active_venues, 1);
        // Mid should be 101
        assert!(
            (fused.mid_price - 101.0).abs() < 0.5,
            "expected mid ~101, got {}",
            fused.mid_price
        );
    }

    #[test]
    fn test_fuse_two_venues_consolidates_depth() {
        let mut ob = default_fusion();
        ob.ingest(&simple_snapshot("a", 100.0, 102.0, 100.0))
            .unwrap();
        ob.ingest(&simple_snapshot("b", 100.0, 102.0, 100.0))
            .unwrap();
        let fused = ob.fuse(100.0).unwrap();

        // At the same price levels, sizes should be aggregated
        // Best bid at 100 should have 10+10 = 20
        assert_eq!(fused.active_venues, 2);
        let best_bid_level = &fused.consolidated_bids[0];
        assert!(
            (best_bid_level.size - 20.0).abs() < 1e-12,
            "consolidated best bid size should be 20, got {}",
            best_bid_level.size
        );
    }

    #[test]
    fn test_fuse_two_venues_different_prices() {
        let mut ob = OrderbookFusion::with_config(OrderbookFusionConfig {
            price_tick: 0.0, // exact matching
            ..default_config()
        });
        // Venue a: bid=100, ask=102
        // Venue b: bid=101, ask=103 (better bid, worse ask)
        ob.ingest(&OrderbookSnapshot {
            venue_id: "a".to_string(),
            bids: vec![level(100.0, 10.0)],
            asks: vec![level(102.0, 10.0)],
            timestamp: 100.0,
        })
        .unwrap();
        ob.ingest(&OrderbookSnapshot {
            venue_id: "b".to_string(),
            bids: vec![level(101.0, 10.0)],
            asks: vec![level(103.0, 10.0)],
            timestamp: 100.0,
        })
        .unwrap();
        let fused = ob.fuse(100.0).unwrap();

        // Best bid should be 101 (from venue b), best ask 102 (from venue a)
        assert!(
            (fused.best_bid - 101.0).abs() < 1e-6,
            "best bid should be 101, got {}",
            fused.best_bid
        );
        assert!(
            (fused.best_ask - 102.0).abs() < 1e-6,
            "best ask should be 102, got {}",
            fused.best_ask
        );
        assert!(
            (fused.spread - 1.0).abs() < 1e-6,
            "spread should be 1, got {}",
            fused.spread
        );
    }

    #[test]
    fn test_fuse_insufficient_venues() {
        let mut ob = OrderbookFusion::with_config(OrderbookFusionConfig {
            min_venues: 3,
            ..default_config()
        });
        ob.ingest(&simple_snapshot("a", 100.0, 101.0, 100.0))
            .unwrap();
        ob.ingest(&simple_snapshot("b", 100.0, 101.0, 100.0))
            .unwrap();
        assert!(ob.fuse(100.0).is_none());
        assert_eq!(ob.stats().insufficient_venue_count, 1);
    }

    #[test]
    fn test_fuse_latest_timestamp() {
        let mut ob = default_fusion();
        ob.ingest(&simple_snapshot("a", 100.0, 101.0, 90.0))
            .unwrap();
        ob.ingest(&simple_snapshot("b", 100.0, 101.0, 95.0))
            .unwrap();
        let fused = ob.fuse(100.0).unwrap();
        assert!((fused.latest_timestamp - 95.0).abs() < 1e-12);
    }

    // -- Staleness ---------------------------------------------------------

    #[test]
    fn test_stale_venue_excluded() {
        let mut ob = OrderbookFusion::with_config(OrderbookFusionConfig {
            max_staleness_secs: 5.0,
            min_venues: 1,
            ..default_config()
        });
        ob.ingest(&simple_snapshot("old", 80.0, 85.0, 10.0))
            .unwrap();
        ob.ingest(&simple_snapshot("fresh", 100.0, 101.0, 99.0))
            .unwrap();
        let fused = ob.fuse(100.0).unwrap();
        assert_eq!(fused.active_venues, 1);
        assert!(
            (fused.best_bid - 100.0).abs() < 1e-12,
            "only fresh venue should contribute"
        );
    }

    #[test]
    fn test_all_stale_returns_none() {
        let mut ob = OrderbookFusion::with_config(OrderbookFusionConfig {
            max_staleness_secs: 5.0,
            min_venues: 1,
            ..default_config()
        });
        ob.ingest(&simple_snapshot("a", 100.0, 101.0, 10.0))
            .unwrap();
        assert!(ob.fuse(100.0).is_none());
    }

    // -- Imbalance detection -----------------------------------------------

    #[test]
    fn test_bid_heavy_imbalance() {
        let mut ob = OrderbookFusion::with_config(OrderbookFusionConfig {
            imbalance_depth: 3,
            imbalance_threshold: 0.3,
            ..default_config()
        });
        // Heavy bids, light asks
        ob.ingest(&asymmetric_snapshot(
            "a",
            100.0,
            101.0,
            vec![100.0, 100.0, 100.0],
            vec![10.0, 10.0, 10.0],
            100.0,
        ))
        .unwrap();
        let fused = ob.fuse(100.0).unwrap();
        assert!(
            fused.imbalance > 0.0,
            "bid-heavy book should have positive imbalance, got {}",
            fused.imbalance
        );
        assert!(
            fused.is_imbalanced,
            "large imbalance should trigger flag, imbalance={}",
            fused.imbalance
        );
    }

    #[test]
    fn test_ask_heavy_imbalance() {
        let mut ob = OrderbookFusion::with_config(OrderbookFusionConfig {
            imbalance_depth: 3,
            imbalance_threshold: 0.3,
            ..default_config()
        });
        // Light bids, heavy asks
        ob.ingest(&asymmetric_snapshot(
            "a",
            100.0,
            101.0,
            vec![10.0, 10.0, 10.0],
            vec![100.0, 100.0, 100.0],
            100.0,
        ))
        .unwrap();
        let fused = ob.fuse(100.0).unwrap();
        assert!(
            fused.imbalance < 0.0,
            "ask-heavy book should have negative imbalance, got {}",
            fused.imbalance
        );
        assert!(fused.is_imbalanced);
    }

    #[test]
    fn test_balanced_book() {
        let mut ob = OrderbookFusion::with_config(OrderbookFusionConfig {
            imbalance_depth: 3,
            imbalance_threshold: 0.3,
            ..default_config()
        });
        ob.ingest(&simple_snapshot("a", 100.0, 101.0, 100.0))
            .unwrap();
        let fused = ob.fuse(100.0).unwrap();
        assert!(
            fused.imbalance.abs() < 1e-12,
            "symmetric book should have zero imbalance, got {}",
            fused.imbalance
        );
        assert!(!fused.is_imbalanced);
    }

    // -- Spread monitoring -------------------------------------------------

    #[test]
    fn test_spread_tracking() {
        let mut ob = default_fusion();
        ob.ingest(&simple_snapshot("a", 100.0, 102.0, 100.0))
            .unwrap();
        let fused = ob.fuse(100.0).unwrap();
        assert!((fused.spread - 2.0).abs() < 1e-12);
    }

    #[test]
    fn test_wide_spread_detection() {
        let mut ob = OrderbookFusion::with_config(OrderbookFusionConfig {
            ema_decay: 0.9,
            spread_alert_multiplier: 2.0,
            ..default_config()
        });

        // Build baseline with tight spread
        for i in 0..20 {
            ob.ingest(&simple_snapshot("a", 100.0, 101.0, i as f64))
                .unwrap();
            ob.fuse(i as f64);
        }

        // Now widen the spread dramatically
        ob.ingest(&simple_snapshot("a", 90.0, 110.0, 20.0)).unwrap();
        let fused = ob.fuse(20.0).unwrap();
        assert!(
            fused.is_wide_spread,
            "dramatically wider spread should trigger alert, spread={}, smoothed={}",
            fused.spread, fused.smoothed_spread
        );
    }

    #[test]
    fn test_normal_spread_no_alert() {
        let mut ob = default_fusion();
        ob.ingest(&simple_snapshot("a", 100.0, 101.0, 100.0))
            .unwrap();
        let fused = ob.fuse(100.0).unwrap();
        assert!(!fused.is_wide_spread);
    }

    // -- EMA smoothing -----------------------------------------------------

    #[test]
    fn test_ema_mid_initialization() {
        let mut ob = default_fusion();
        ob.ingest(&simple_snapshot("a", 100.0, 102.0, 100.0))
            .unwrap();
        let f1 = ob.fuse(100.0).unwrap();
        // First fusion: EMA initialises to raw value
        assert!(
            (f1.smoothed_mid_price - f1.mid_price).abs() < 1e-12,
            "first fusion should initialise EMA"
        );
    }

    #[test]
    fn test_ema_mid_lags() {
        let mut ob = OrderbookFusion::with_config(OrderbookFusionConfig {
            ema_decay: 0.8,
            ..default_config()
        });
        ob.ingest(&simple_snapshot("a", 100.0, 102.0, 100.0))
            .unwrap();
        let f1 = ob.fuse(100.0).unwrap();
        let mid1 = f1.mid_price;

        ob.ingest(&simple_snapshot("a", 110.0, 112.0, 101.0))
            .unwrap();
        let f2 = ob.fuse(101.0).unwrap();
        let mid2 = f2.mid_price;

        // Smoothed mid should lag between mid1 and mid2
        assert!(
            f2.smoothed_mid_price > mid1.min(mid2) && f2.smoothed_mid_price < mid1.max(mid2),
            "smoothed mid should lag, smooth={}, mid1={}, mid2={}",
            f2.smoothed_mid_price,
            mid1,
            mid2
        );
    }

    #[test]
    fn test_ema_imbalance_smoothing() {
        let mut ob = OrderbookFusion::with_config(OrderbookFusionConfig {
            ema_decay: 0.8,
            imbalance_depth: 3,
            ..default_config()
        });
        // First: balanced book → imbalance = 0
        ob.ingest(&simple_snapshot("a", 100.0, 101.0, 100.0))
            .unwrap();
        ob.fuse(100.0).unwrap();

        // Second: bid-heavy → positive imbalance
        ob.ingest(&asymmetric_snapshot(
            "a",
            100.0,
            101.0,
            vec![100.0, 100.0, 100.0],
            vec![10.0, 10.0, 10.0],
            101.0,
        ))
        .unwrap();
        let f2 = ob.fuse(101.0).unwrap();

        // Smoothed imbalance should be positive but less than raw (lagging)
        assert!(
            f2.smoothed_imbalance > 0.0,
            "smoothed imbalance should be positive"
        );
        assert!(
            f2.smoothed_imbalance <= f2.imbalance,
            "smoothed should lag behind raw, smooth={}, raw={}",
            f2.smoothed_imbalance,
            f2.imbalance
        );
    }

    // -- Depth consolidation -----------------------------------------------

    #[test]
    fn test_depth_aggregation_at_same_level() {
        let mut ob = default_fusion();
        // Two venues both have bids at 100 with size 15 each
        ob.ingest(&OrderbookSnapshot {
            venue_id: "a".to_string(),
            bids: vec![level(100.0, 15.0)],
            asks: vec![level(102.0, 10.0)],
            timestamp: 100.0,
        })
        .unwrap();
        ob.ingest(&OrderbookSnapshot {
            venue_id: "b".to_string(),
            bids: vec![level(100.0, 25.0)],
            asks: vec![level(102.0, 20.0)],
            timestamp: 100.0,
        })
        .unwrap();
        let fused = ob.fuse(100.0).unwrap();

        assert!(
            (fused.consolidated_bids[0].size - 40.0).abs() < 1e-12,
            "bid depth at 100 should be 40, got {}",
            fused.consolidated_bids[0].size
        );
        assert!(
            (fused.consolidated_asks[0].size - 30.0).abs() < 1e-12,
            "ask depth at 102 should be 30, got {}",
            fused.consolidated_asks[0].size
        );
    }

    #[test]
    fn test_total_depth() {
        let mut ob = default_fusion();
        ob.ingest(&simple_snapshot("a", 100.0, 101.0, 100.0))
            .unwrap();
        let fused = ob.fuse(100.0).unwrap();
        // bids: 10 + 20 + 30 = 60
        assert!(
            (fused.total_bid_depth - 60.0).abs() < 1e-12,
            "total bid depth should be 60, got {}",
            fused.total_bid_depth
        );
        assert!(
            (fused.total_ask_depth - 60.0).abs() < 1e-12,
            "total ask depth should be 60, got {}",
            fused.total_ask_depth
        );
    }

    #[test]
    fn test_consolidated_bids_sorted_descending() {
        let mut ob = OrderbookFusion::with_config(OrderbookFusionConfig {
            price_tick: 0.0,
            ..default_config()
        });
        ob.ingest(&OrderbookSnapshot {
            venue_id: "a".to_string(),
            bids: vec![level(100.0, 10.0), level(98.0, 10.0)],
            asks: vec![level(101.0, 10.0)],
            timestamp: 100.0,
        })
        .unwrap();
        ob.ingest(&OrderbookSnapshot {
            venue_id: "b".to_string(),
            bids: vec![level(99.0, 10.0), level(97.0, 10.0)],
            asks: vec![level(102.0, 10.0)],
            timestamp: 100.0,
        })
        .unwrap();
        let fused = ob.fuse(100.0).unwrap();

        for i in 1..fused.consolidated_bids.len() {
            assert!(
                fused.consolidated_bids[i].price <= fused.consolidated_bids[i - 1].price,
                "bids should be sorted descending: {} > {}",
                fused.consolidated_bids[i].price,
                fused.consolidated_bids[i - 1].price
            );
        }
    }

    #[test]
    fn test_consolidated_asks_sorted_ascending() {
        let mut ob = OrderbookFusion::with_config(OrderbookFusionConfig {
            price_tick: 0.0,
            ..default_config()
        });
        ob.ingest(&OrderbookSnapshot {
            venue_id: "a".to_string(),
            bids: vec![level(100.0, 10.0)],
            asks: vec![level(101.0, 10.0), level(103.0, 10.0)],
            timestamp: 100.0,
        })
        .unwrap();
        ob.ingest(&OrderbookSnapshot {
            venue_id: "b".to_string(),
            bids: vec![level(100.0, 10.0)],
            asks: vec![level(102.0, 10.0), level(104.0, 10.0)],
            timestamp: 100.0,
        })
        .unwrap();
        let fused = ob.fuse(100.0).unwrap();

        for i in 1..fused.consolidated_asks.len() {
            assert!(
                fused.consolidated_asks[i].price >= fused.consolidated_asks[i - 1].price,
                "asks should be sorted ascending: {} < {}",
                fused.consolidated_asks[i].price,
                fused.consolidated_asks[i - 1].price
            );
        }
    }

    #[test]
    fn test_max_depth_levels_cap() {
        let mut ob = OrderbookFusion::with_config(OrderbookFusionConfig {
            max_depth_levels: 2,
            price_tick: 0.0,
            ..default_config()
        });
        ob.ingest(&OrderbookSnapshot {
            venue_id: "a".to_string(),
            bids: vec![
                level(100.0, 10.0),
                level(99.0, 10.0),
                level(98.0, 10.0),
                level(97.0, 10.0),
            ],
            asks: vec![level(101.0, 10.0), level(102.0, 10.0), level(103.0, 10.0)],
            timestamp: 100.0,
        })
        .unwrap();
        let fused = ob.fuse(100.0).unwrap();
        assert!(
            fused.consolidated_bids.len() <= 2,
            "bids should be capped at 2, got {}",
            fused.consolidated_bids.len()
        );
        assert!(
            fused.consolidated_asks.len() <= 2,
            "asks should be capped at 2, got {}",
            fused.consolidated_asks.len()
        );
    }

    // -- Price tick rounding ------------------------------------------------

    #[test]
    fn test_price_tick_aggregation() {
        let mut ob = OrderbookFusion::with_config(OrderbookFusionConfig {
            price_tick: 1.0, // round to nearest 1.0
            ..default_config()
        });
        // Two bids at 100.3 and 99.7 should both round to 100.0
        ob.ingest(&OrderbookSnapshot {
            venue_id: "a".to_string(),
            bids: vec![level(100.3, 10.0)],
            asks: vec![level(101.0, 10.0)],
            timestamp: 100.0,
        })
        .unwrap();
        ob.ingest(&OrderbookSnapshot {
            venue_id: "b".to_string(),
            bids: vec![level(99.7, 15.0)],
            asks: vec![level(101.0, 10.0)],
            timestamp: 100.0,
        })
        .unwrap();
        let fused = ob.fuse(100.0).unwrap();

        // Both should aggregate into the 100.0 bucket
        let best_bid = &fused.consolidated_bids[0];
        assert!(
            (best_bid.price - 100.0).abs() < 1e-6,
            "expected rounded to 100.0, got {}",
            best_bid.price
        );
        assert!(
            (best_bid.size - 25.0).abs() < 1e-6,
            "expected aggregated size 25, got {}",
            best_bid.size
        );
    }

    #[test]
    fn test_no_tick_rounding() {
        let mut ob = OrderbookFusion::with_config(OrderbookFusionConfig {
            price_tick: 0.0, // disabled
            ..default_config()
        });
        ob.ingest(&OrderbookSnapshot {
            venue_id: "a".to_string(),
            bids: vec![level(100.3, 10.0)],
            asks: vec![level(101.0, 10.0)],
            timestamp: 100.0,
        })
        .unwrap();
        ob.ingest(&OrderbookSnapshot {
            venue_id: "b".to_string(),
            bids: vec![level(100.1, 15.0)],
            asks: vec![level(101.0, 10.0)],
            timestamp: 100.0,
        })
        .unwrap();
        let fused = ob.fuse(100.0).unwrap();

        // Different prices should NOT aggregate
        assert!(
            fused.consolidated_bids.len() >= 2,
            "without tick rounding, distinct prices should remain separate, got {} levels",
            fused.consolidated_bids.len()
        );
    }

    // -- Weighted mid-price ------------------------------------------------

    #[test]
    fn test_weighted_mid_price_tighter_spread_wins() {
        let mut ob = OrderbookFusion::with_config(OrderbookFusionConfig {
            price_tick: 0.0,
            ..default_config()
        });
        // Venue a: tight spread (100-101), mid=100.5
        ob.ingest(&OrderbookSnapshot {
            venue_id: "a".to_string(),
            bids: vec![level(100.0, 10.0)],
            asks: vec![level(101.0, 10.0)],
            timestamp: 100.0,
        })
        .unwrap();
        // Venue b: wide spread (95-110), mid=102.5
        ob.ingest(&OrderbookSnapshot {
            venue_id: "b".to_string(),
            bids: vec![level(95.0, 10.0)],
            asks: vec![level(110.0, 10.0)],
            timestamp: 100.0,
        })
        .unwrap();
        let fused = ob.fuse(100.0).unwrap();

        // Mid should be closer to venue a's mid (100.5) than venue b's (102.5)
        assert!(
            (fused.mid_price - 100.5).abs() < (fused.mid_price - 102.5).abs(),
            "tighter-spread venue should dominate mid, got {}",
            fused.mid_price
        );
    }

    #[test]
    fn test_weighted_mid_price_deeper_book_wins() {
        let mut ob = OrderbookFusion::with_config(OrderbookFusionConfig {
            price_tick: 0.0,
            ..default_config()
        });
        // Venue a: deep book, mid=100.5
        ob.ingest(&OrderbookSnapshot {
            venue_id: "a".to_string(),
            bids: vec![level(100.0, 1000.0)],
            asks: vec![level(101.0, 1000.0)],
            timestamp: 100.0,
        })
        .unwrap();
        // Venue b: thin book, same spread, mid=200.5
        ob.ingest(&OrderbookSnapshot {
            venue_id: "b".to_string(),
            bids: vec![level(200.0, 1.0)],
            asks: vec![level(201.0, 1.0)],
            timestamp: 100.0,
        })
        .unwrap();
        let fused = ob.fuse(100.0).unwrap();

        // Mid should be much closer to 100.5 (deep venue)
        assert!(
            (fused.mid_price - 100.5).abs() < (fused.mid_price - 200.5).abs(),
            "deeper venue should dominate mid, got {}",
            fused.mid_price
        );
    }

    // -- Windowed statistics -----------------------------------------------

    #[test]
    fn test_windowed_mean() {
        let mut ob = default_fusion();
        ob.ingest(&simple_snapshot("a", 100.0, 102.0, 100.0))
            .unwrap();
        ob.fuse(100.0);
        ob.ingest(&simple_snapshot("a", 110.0, 112.0, 101.0))
            .unwrap();
        ob.fuse(101.0);

        let mean = ob.windowed_mean().unwrap();
        assert!(mean > 0.0, "windowed mean should be positive");
    }

    #[test]
    fn test_windowed_std_constant() {
        let mut ob = default_fusion();
        for i in 0..5 {
            ob.ingest(&simple_snapshot("a", 100.0, 102.0, 100.0 + i as f64))
                .unwrap();
            ob.fuse(100.0 + i as f64);
        }
        let std = ob.windowed_std().unwrap();
        assert!(
            std < 1e-6,
            "constant mid-prices should have near-zero std, got {}",
            std
        );
    }

    #[test]
    fn test_windowed_mean_empty() {
        let ob = default_fusion();
        assert!(ob.windowed_mean().is_none());
    }

    #[test]
    fn test_windowed_std_insufficient() {
        let mut ob = default_fusion();
        ob.ingest(&simple_snapshot("a", 100.0, 102.0, 100.0))
            .unwrap();
        ob.fuse(100.0);
        assert!(ob.windowed_std().is_none());
    }

    // -- Reset -------------------------------------------------------------

    #[test]
    fn test_reset() {
        let mut ob = default_fusion();
        for i in 0..10 {
            ob.ingest(&simple_snapshot("a", 100.0, 101.0, i as f64))
                .unwrap();
            ob.fuse(i as f64);
        }
        assert!(ob.venue_count() > 0);
        assert!(ob.stats().total_fusions > 0);

        ob.reset();

        assert_eq!(ob.venue_count(), 0);
        assert_eq!(ob.stats().total_fusions, 0);
        assert_eq!(ob.stats().total_snapshots, 0);
        assert!(ob.windowed_mean().is_none());
    }

    // -- Stats tracking ----------------------------------------------------

    #[test]
    fn test_stats_tracking() {
        let mut ob = default_fusion();
        ob.ingest(&simple_snapshot("a", 100.0, 101.0, 100.0))
            .unwrap();
        ob.ingest(&simple_snapshot("b", 100.0, 101.0, 100.0))
            .unwrap();
        ob.fuse(100.0);
        ob.fuse(100.0);

        let s = ob.stats();
        assert_eq!(s.total_snapshots, 2);
        assert_eq!(s.total_fusions, 2);
        assert_eq!(s.distinct_venues, 2);
    }

    #[test]
    fn test_imbalance_count_in_stats() {
        let mut ob = OrderbookFusion::with_config(OrderbookFusionConfig {
            imbalance_depth: 3,
            imbalance_threshold: 0.1,
            ..default_config()
        });
        ob.ingest(&asymmetric_snapshot(
            "a",
            100.0,
            101.0,
            vec![100.0, 100.0, 100.0],
            vec![1.0, 1.0, 1.0],
            100.0,
        ))
        .unwrap();
        ob.fuse(100.0);
        assert!(ob.stats().imbalance_count >= 1);
    }

    #[test]
    fn test_wide_spread_count_in_stats() {
        let mut ob = OrderbookFusion::with_config(OrderbookFusionConfig {
            ema_decay: 0.95,
            spread_alert_multiplier: 2.0,
            ..default_config()
        });
        // Build baseline with tight spread
        for i in 0..20 {
            ob.ingest(&simple_snapshot("a", 100.0, 100.5, i as f64))
                .unwrap();
            ob.fuse(i as f64);
        }
        // Spike spread
        ob.ingest(&simple_snapshot("a", 90.0, 110.0, 20.0)).unwrap();
        ob.fuse(20.0);
        assert!(ob.stats().wide_spread_count >= 1);
    }

    // -- Venue stats -------------------------------------------------------

    #[test]
    fn test_venue_stats() {
        let mut ob = default_fusion();
        ob.ingest(&simple_snapshot("binance", 100.0, 102.0, 100.0))
            .unwrap();
        ob.ingest(&simple_snapshot("binance", 101.0, 103.0, 110.0))
            .unwrap();

        let vs = ob.venue_stats();
        let bs = &vs["binance"];
        assert_eq!(bs.snapshots, 2);
        assert!((bs.last_best_bid - 101.0).abs() < 1e-12);
        assert!((bs.last_best_ask - 103.0).abs() < 1e-12);
        assert!((bs.last_spread - 2.0).abs() < 1e-12);
        assert!((bs.last_timestamp - 110.0).abs() < 1e-12);
    }

    // -- Window eviction ---------------------------------------------------

    #[test]
    fn test_window_eviction() {
        let mut ob = OrderbookFusion::with_config(OrderbookFusionConfig {
            window_size: 3,
            ..default_config()
        });
        for i in 0..10 {
            ob.ingest(&simple_snapshot("a", 100.0, 101.0, i as f64))
                .unwrap();
            ob.fuse(i as f64);
        }
        assert_eq!(ob.history.len(), 3);
    }

    // -- Empty book --------------------------------------------------------

    #[test]
    fn test_fuse_with_empty_book() {
        let mut ob = default_fusion();
        ob.ingest(&OrderbookSnapshot {
            venue_id: "empty".to_string(),
            bids: vec![],
            asks: vec![],
            timestamp: 100.0,
        })
        .unwrap();
        let fused = ob.fuse(100.0).unwrap();
        assert_eq!(fused.active_venues, 1);
        assert!(fused.consolidated_bids.is_empty());
        assert!(fused.consolidated_asks.is_empty());
    }

    // -- One-sided book ----------------------------------------------------

    #[test]
    fn test_fuse_bids_only() {
        let mut ob = default_fusion();
        ob.ingest(&OrderbookSnapshot {
            venue_id: "a".to_string(),
            bids: vec![level(100.0, 10.0)],
            asks: vec![],
            timestamp: 100.0,
        })
        .unwrap();
        let fused = ob.fuse(100.0).unwrap();
        assert!(fused.consolidated_asks.is_empty());
        assert!(!fused.consolidated_bids.is_empty());
    }

    #[test]
    fn test_fuse_asks_only() {
        let mut ob = default_fusion();
        ob.ingest(&OrderbookSnapshot {
            venue_id: "a".to_string(),
            bids: vec![],
            asks: vec![level(101.0, 10.0)],
            timestamp: 100.0,
        })
        .unwrap();
        let fused = ob.fuse(100.0).unwrap();
        assert!(fused.consolidated_bids.is_empty());
        assert!(!fused.consolidated_asks.is_empty());
    }
}
