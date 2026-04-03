//! # Fill Model
//!
//! Models for estimating fill probability and queue position for resting
//! limit orders in the LOB simulator. These models answer the question:
//! "Given my order's position in the queue, how likely is it to be filled?"
//!
//! # Models
//!
//! - **QueuePosition**: Tracks estimated queue position based on order
//!   arrival time and observed activity at the price level.
//! - **FillProbability**: Computes the probability that a resting limit
//!   order will be filled within a given time horizon.
//! - **FillModel**: Top-level orchestrator that combines queue position,
//!   fill probability, and partial fill estimation.
//!
//! # Queue Position Estimation
//!
//! When a limit order rests on the book, it joins the back of the FIFO
//! queue at its price level. Its position advances as:
//! 1. Orders ahead of it are filled by incoming market orders.
//! 2. Orders ahead of it are cancelled.
//!
//! Without L3 data, queue position must be estimated. We support:
//! - **Uniform**: Assume our order is at a random position in the queue.
//! - **BackOfQueue**: Pessimistic — assume we are last in line.
//! - **ProRata**: Fills are distributed proportionally across all orders
//!   at the price level (used by some futures exchanges).
//! - **Empirical**: Use historical cancel/fill rates to model queue
//!   advancement dynamically.
//!
//! # Usage
//!
//! ```rust,ignore
//! use janus_lob::fill_model::*;
//! use rust_decimal_macros::dec;
//!
//! // Create a fill model with back-of-queue estimation
//! let model = FillModel::new(QueuePositionModel::BackOfQueue);
//!
//! // Estimate fill probability for a resting bid
//! let prob = model.estimate_fill_probability(
//!     dec!(1.5),   // our order size
//!     dec!(10.0),  // total quantity at our price level
//!     dec!(3.0),   // quantity ahead of us in queue
//!     dec!(50.0),  // volume traded through this level recently
//! );
//!
//! println!("Fill probability: {:.1}%", prob.probability * 100.0);
//! ```

use rust_decimal::Decimal;
use serde::{Deserialize, Serialize};
use std::fmt;
use std::time::Duration;
use tracing::trace;

// ---------------------------------------------------------------------------
// Queue Position
// ---------------------------------------------------------------------------

/// Model for estimating queue position when L3 data is unavailable.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Serialize, Deserialize, Default)]
pub enum QueuePositionModel {
    /// Assume our order is at a uniformly random position in the queue.
    /// Expected position = total_depth / 2.
    Uniform,

    /// Pessimistic: assume our order is at the back of the queue.
    /// Position = total_depth (worst case).
    #[default]
    BackOfQueue,

    /// Optimistic: assume our order is at the front of the queue.
    /// Position = 0 (best case). Rarely realistic.
    FrontOfQueue,

    /// Pro-rata allocation: fills are distributed proportionally across
    /// all resting orders at the price level. Used by some futures
    /// exchanges (e.g., CME Eurodollars).
    ProRata,

    /// Use a configurable fractile of the queue depth.
    /// `fractile` should be in [0, 1] where 0 = front, 1 = back.
    Fractile,
}

impl fmt::Display for QueuePositionModel {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        match self {
            Self::Uniform => write!(f, "Uniform"),
            Self::BackOfQueue => write!(f, "BackOfQueue"),
            Self::FrontOfQueue => write!(f, "FrontOfQueue"),
            Self::ProRata => write!(f, "ProRata"),
            Self::Fractile => write!(f, "Fractile"),
        }
    }
}

/// Estimated queue position for a resting limit order.
#[derive(Debug, Clone, Copy, Serialize, Deserialize)]
pub struct QueuePosition {
    /// Estimated quantity ahead of our order in the FIFO queue.
    pub quantity_ahead: Decimal,

    /// Our order's quantity.
    pub our_quantity: Decimal,

    /// Total quantity at the price level (including our order).
    pub level_quantity: Decimal,

    /// Fractional position in queue: 0.0 = front, 1.0 = back.
    pub fractile: f64,

    /// The model used to estimate this position.
    pub model: QueuePositionModel,
}

impl QueuePosition {
    /// Fraction of the level that must be consumed before our order is reached.
    pub fn depth_fraction(&self) -> f64 {
        if self.level_quantity.is_zero() {
            return 0.0;
        }
        let ahead: f64 = self.quantity_ahead.try_into().unwrap_or(0.0);
        let total: f64 = self.level_quantity.try_into().unwrap_or(1.0);
        (ahead / total).clamp(0.0, 1.0)
    }

    /// Whether our order is at the front of the queue.
    pub fn is_front(&self) -> bool {
        self.quantity_ahead.is_zero()
    }

    /// Whether our order would be fully filled if `volume` trades through this level.
    pub fn would_fill(&self, volume: Decimal) -> bool {
        volume >= self.quantity_ahead + self.our_quantity
    }

    /// Estimate the partial fill quantity given `volume` traded through this level.
    pub fn partial_fill(&self, volume: Decimal) -> Decimal {
        if volume <= self.quantity_ahead {
            // Volume didn't reach our position.
            Decimal::ZERO
        } else {
            let available = volume - self.quantity_ahead;
            available.min(self.our_quantity)
        }
    }
}

impl fmt::Display for QueuePosition {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        write!(
            f,
            "QueuePos(ahead={}, ours={}, level={}, frac={:.1}%, model={})",
            self.quantity_ahead,
            self.our_quantity,
            self.level_quantity,
            self.fractile * 100.0,
            self.model,
        )
    }
}

// ---------------------------------------------------------------------------
// Fill Probability
// ---------------------------------------------------------------------------

/// Estimated fill probability for a resting limit order.
#[derive(Debug, Clone, Copy, Serialize, Deserialize)]
pub struct FillProbability {
    /// Probability of being filled (0.0 to 1.0).
    pub probability: f64,

    /// Expected fill quantity (probability × order quantity).
    pub expected_fill: Decimal,

    /// Expected time to fill (if the probability model supports it).
    pub expected_time: Option<Duration>,

    /// Queue position used in the estimate.
    pub queue_position: QueuePosition,
}

impl FillProbability {
    /// Whether the fill is likely (probability > 50%).
    pub fn is_likely(&self) -> bool {
        self.probability > 0.5
    }

    /// Whether the fill is unlikely (probability < 10%).
    pub fn is_unlikely(&self) -> bool {
        self.probability < 0.1
    }

    /// Confidence level description.
    pub fn confidence_label(&self) -> &'static str {
        if self.probability >= 0.9 {
            "very likely"
        } else if self.probability >= 0.7 {
            "likely"
        } else if self.probability >= 0.4 {
            "moderate"
        } else if self.probability >= 0.1 {
            "unlikely"
        } else {
            "very unlikely"
        }
    }
}

impl fmt::Display for FillProbability {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        write!(
            f,
            "FillProb({:.1}% [{}], expected_fill={}, {})",
            self.probability * 100.0,
            self.confidence_label(),
            self.expected_fill,
            self.queue_position,
        )
    }
}

// ---------------------------------------------------------------------------
// Fill Model Configuration
// ---------------------------------------------------------------------------

/// Configuration for the fill model.
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct FillModelConfig {
    /// Queue position estimation model.
    pub queue_model: QueuePositionModel,

    /// Fractile for the `Fractile` queue model (0.0 = front, 1.0 = back).
    /// Only used when `queue_model == QueuePositionModel::Fractile`.
    pub queue_fractile: f64,

    /// Estimated cancel rate: fraction of queue depth that cancels per time unit.
    /// Cancellations advance our queue position. Typical values: 0.3–0.7.
    pub cancel_rate: f64,

    /// Minimum fill probability to report (below this we clamp to 0).
    pub min_probability: f64,

    /// Whether to model partial fills (pro-rata) or only full fills (FIFO).
    pub allow_partial: bool,

    /// Volume lookback window for fill probability estimation.
    /// If `None`, use all available volume data.
    pub volume_lookback: Option<Duration>,
}

impl Default for FillModelConfig {
    fn default() -> Self {
        Self {
            queue_model: QueuePositionModel::BackOfQueue,
            queue_fractile: 0.75,
            cancel_rate: 0.5,
            min_probability: 0.001,
            allow_partial: true,
            volume_lookback: None,
        }
    }
}

impl FillModelConfig {
    /// Set the queue position model.
    pub fn with_queue_model(mut self, model: QueuePositionModel) -> Self {
        self.queue_model = model;
        self
    }

    /// Set the queue fractile (for `Fractile` model).
    pub fn with_queue_fractile(mut self, fractile: f64) -> Self {
        self.queue_fractile = fractile.clamp(0.0, 1.0);
        self
    }

    /// Set the cancel rate.
    pub fn with_cancel_rate(mut self, rate: f64) -> Self {
        self.cancel_rate = rate.clamp(0.0, 1.0);
        self
    }

    /// Set whether partial fills are allowed.
    pub fn with_partial_fills(mut self, allow: bool) -> Self {
        self.allow_partial = allow;
        self
    }

    /// Set the volume lookback window.
    pub fn with_volume_lookback(mut self, lookback: Duration) -> Self {
        self.volume_lookback = Some(lookback);
        self
    }
}

// ---------------------------------------------------------------------------
// Fill Model
// ---------------------------------------------------------------------------

/// Fill probability and queue position estimation model.
///
/// This model estimates the likelihood that a resting limit order will be
/// filled based on its queue position, the depth at its price level, and
/// observed trading volume.
#[derive(Debug, Clone)]
pub struct FillModel {
    /// Configuration.
    config: FillModelConfig,

    /// Running statistics.
    stats: FillModelStats,
}

/// Running statistics for the fill model.
#[derive(Debug, Clone, Default, Serialize, Deserialize)]
pub struct FillModelStats {
    /// Number of fill probability estimates computed.
    pub estimates_computed: u64,

    /// Number of times a fill was predicted as likely (>50%).
    pub predicted_likely: u64,

    /// Number of times a fill was predicted as unlikely (<10%).
    pub predicted_unlikely: u64,

    /// Sum of predicted probabilities (for computing mean predicted probability).
    pub sum_probability: f64,

    /// Number of queue position estimates computed.
    pub position_estimates: u64,
}

impl FillModelStats {
    /// Mean predicted fill probability.
    pub fn mean_probability(&self) -> f64 {
        if self.estimates_computed == 0 {
            0.0
        } else {
            self.sum_probability / self.estimates_computed as f64
        }
    }

    /// Reset all statistics.
    pub fn reset(&mut self) {
        *self = Self::default();
    }
}

impl fmt::Display for FillModelStats {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        write!(
            f,
            "FillModelStats(estimates={}, mean_prob={:.1}%, likely={}, unlikely={})",
            self.estimates_computed,
            self.mean_probability() * 100.0,
            self.predicted_likely,
            self.predicted_unlikely,
        )
    }
}

impl FillModel {
    /// Create a new fill model with the given queue position model.
    pub fn new(queue_model: QueuePositionModel) -> Self {
        Self {
            config: FillModelConfig::default().with_queue_model(queue_model),
            stats: FillModelStats::default(),
        }
    }

    /// Create a new fill model with full configuration.
    pub fn with_config(config: FillModelConfig) -> Self {
        Self {
            config,
            stats: FillModelStats::default(),
        }
    }

    /// Get the configuration.
    pub fn config(&self) -> &FillModelConfig {
        &self.config
    }

    /// Get the running statistics.
    pub fn stats(&self) -> &FillModelStats {
        &self.stats
    }

    /// Reset statistics.
    pub fn reset_stats(&mut self) {
        self.stats.reset();
    }

    // ── Queue Position Estimation ──────────────────────────────────────

    /// Estimate queue position for a resting order.
    ///
    /// # Arguments
    /// - `our_quantity`: Our order's quantity.
    /// - `level_quantity`: Total quantity at the price level (including ours).
    /// - `known_ahead`: If L3 data is available, the exact quantity ahead.
    ///   If `None`, the model estimates based on `queue_model`.
    pub fn estimate_queue_position(
        &mut self,
        our_quantity: Decimal,
        level_quantity: Decimal,
        known_ahead: Option<Decimal>,
    ) -> QueuePosition {
        self.stats.position_estimates += 1;

        let level_f64: f64 = level_quantity.try_into().unwrap_or(0.0);
        let our_f64: f64 = our_quantity.try_into().unwrap_or(0.0);

        let (quantity_ahead, fractile) = if let Some(ahead) = known_ahead {
            // L3 data available — use exact position.
            let frac = if level_f64 > 0.0 {
                let a: f64 = ahead.try_into().unwrap_or(0.0);
                a / level_f64
            } else {
                0.0
            };
            (ahead, frac)
        } else {
            // Estimate based on model.
            let others = level_quantity - our_quantity;
            let others = if others < Decimal::ZERO {
                Decimal::ZERO
            } else {
                others
            };

            match self.config.queue_model {
                QueuePositionModel::FrontOfQueue => (Decimal::ZERO, 0.0),

                QueuePositionModel::BackOfQueue => {
                    let frac = if level_f64 > 0.0 {
                        let o: f64 = others.try_into().unwrap_or(0.0);
                        o / level_f64
                    } else {
                        1.0
                    };
                    (others, frac)
                }

                QueuePositionModel::Uniform => {
                    // Expected position = others / 2
                    let ahead = others / Decimal::from(2);
                    let frac = if level_f64 > 0.0 {
                        let a: f64 = ahead.try_into().unwrap_or(0.0);
                        a / level_f64
                    } else {
                        0.5
                    };
                    (ahead, frac)
                }

                QueuePositionModel::ProRata => {
                    // Pro-rata: position doesn't matter, fill is proportional.
                    // We represent this as "no queue" and handle it in fill probability.
                    (Decimal::ZERO, 0.0)
                }

                QueuePositionModel::Fractile => {
                    let frac = self.config.queue_fractile;
                    let ahead = Decimal::try_from(level_f64 * frac - our_f64)
                        .unwrap_or(Decimal::ZERO)
                        .max(Decimal::ZERO);
                    (ahead, frac)
                }
            }
        };

        let pos = QueuePosition {
            quantity_ahead,
            our_quantity,
            level_quantity,
            fractile,
            model: self.config.queue_model,
        };

        trace!(
            queue_model = %self.config.queue_model,
            ahead = %quantity_ahead,
            our = %our_quantity,
            level = %level_quantity,
            fractile = fractile,
            "Queue position estimated"
        );

        pos
    }

    // ── Fill Probability Estimation ────────────────────────────────────

    /// Estimate the probability that a resting order will be filled.
    ///
    /// # Arguments
    /// - `our_quantity`: Our order's quantity.
    /// - `level_quantity`: Total quantity at the price level.
    /// - `quantity_ahead`: Known or estimated quantity ahead of us.
    /// - `expected_volume`: Expected volume that will trade through this level.
    ///   This can come from historical volume analysis, or from observing
    ///   the rate of trades at this price level.
    ///
    /// # Returns
    /// A `FillProbability` struct with the estimated probability and metadata.
    pub fn estimate_fill_probability(
        &mut self,
        our_quantity: Decimal,
        level_quantity: Decimal,
        quantity_ahead: Option<Decimal>,
        expected_volume: Decimal,
    ) -> FillProbability {
        let pos = self.estimate_queue_position(our_quantity, level_quantity, quantity_ahead);
        self.compute_fill_probability(pos, expected_volume)
    }

    /// Compute fill probability from an existing queue position estimate.
    pub fn compute_fill_probability(
        &mut self,
        position: QueuePosition,
        expected_volume: Decimal,
    ) -> FillProbability {
        self.stats.estimates_computed += 1;

        let volume_f64: f64 = expected_volume.try_into().unwrap_or(0.0);
        let our_f64: f64 = position.our_quantity.try_into().unwrap_or(0.0);
        let ahead_f64: f64 = position.quantity_ahead.try_into().unwrap_or(0.0);
        let level_f64: f64 = position.level_quantity.try_into().unwrap_or(1.0);

        let probability = if volume_f64 <= 0.0 || our_f64 <= 0.0 {
            0.0
        } else if position.model == QueuePositionModel::ProRata {
            // Pro-rata: our share = our_qty / level_qty.
            // Fill probability = min(1, volume * share / our_qty).
            if level_f64 <= 0.0 {
                0.0
            } else {
                let share = our_f64 / level_f64;
                let our_fill = volume_f64 * share;
                (our_fill / our_f64).min(1.0)
            }
        } else {
            // FIFO model: need volume > quantity_ahead + our_quantity for full fill.
            // Adjust for cancellation rate: effective_ahead = ahead * (1 - cancel_rate).
            let effective_ahead = ahead_f64 * (1.0 - self.config.cancel_rate);
            let need_for_full = effective_ahead + our_f64;

            if volume_f64 >= need_for_full {
                // Full fill certain.
                1.0
            } else if volume_f64 <= effective_ahead {
                // Volume doesn't even reach our position.
                if self.config.allow_partial {
                    // Small chance due to cancellation uncertainty.
                    // Use a sigmoid-like probability near the boundary.
                    let ratio = volume_f64 / (effective_ahead + 1.0);
                    let cancel_uncertainty = self.config.cancel_rate * 0.3;
                    (ratio * cancel_uncertainty).min(0.3)
                } else {
                    0.0
                }
            } else {
                // Volume reaches our position but may not fill us completely.
                let penetration = volume_f64 - effective_ahead;
                if self.config.allow_partial {
                    // Probability of at least partial fill is high.
                    // Probability of full fill scales linearly with penetration.
                    let full_fill_prob = penetration / our_f64;
                    full_fill_prob.min(1.0)
                } else {
                    // Only count full fills.
                    let full_fill_prob = (penetration / our_f64).min(1.0);
                    // Apply a penalty for uncertainty.
                    full_fill_prob * (1.0 - (1.0 - full_fill_prob).powi(2))
                }
            }
        };

        // Clamp to valid range and apply minimum threshold.
        let probability = if probability < self.config.min_probability {
            0.0
        } else {
            probability.min(1.0)
        };

        let expected_fill = if probability > 0.0 {
            Decimal::try_from(our_f64 * probability).unwrap_or(Decimal::ZERO)
        } else {
            Decimal::ZERO
        };

        // Update stats.
        self.stats.sum_probability += probability;
        if probability > 0.5 {
            self.stats.predicted_likely += 1;
        }
        if probability < 0.1 {
            self.stats.predicted_unlikely += 1;
        }

        // Estimate expected time to fill based on volume rate and queue position.
        // If we know the expected volume that will trade through this level,
        // and we know how much volume needs to trade before our order is reached,
        // we can estimate the time as: t = quantity_needed / volume_rate.
        //
        // We use the volume_lookback config as the observation window for
        // the expected_volume, giving us an implied volume rate (units/sec).
        let expected_time = if probability > 0.0 && volume_f64 > 0.0 {
            let lookback_secs = self
                .config
                .volume_lookback
                .map(|d| d.as_secs_f64())
                .unwrap_or(300.0); // Default to 5 minutes if unset
            if lookback_secs > 0.0 {
                // Volume rate = expected_volume / lookback_window (units per second)
                let volume_rate = volume_f64 / lookback_secs;
                if volume_rate > 1e-12 {
                    // Quantity that must trade before we are (at least partially) filled:
                    // effective_ahead adjusted for cancellation + a fraction of our order.
                    let effective_ahead = ahead_f64 * (1.0 - self.config.cancel_rate);
                    // For partial fills we need at least effective_ahead to be consumed;
                    // for full fills we need effective_ahead + our_quantity.
                    let quantity_needed = effective_ahead + our_f64 * 0.5; // midpoint estimate
                    let secs = quantity_needed / volume_rate;
                    // Clamp to a reasonable range (1ms .. 24 hours)
                    let clamped = secs.clamp(0.001, 86_400.0);
                    Some(Duration::from_secs_f64(clamped))
                } else {
                    None
                }
            } else {
                None
            }
        } else {
            None
        };

        let result = FillProbability {
            probability,
            expected_fill,
            expected_time,
            queue_position: position,
        };

        trace!(
            probability = probability,
            expected_fill = %expected_fill,
            volume = %expected_volume,
            model = %position.model,
            "Fill probability estimated"
        );

        result
    }

    // ── Convenience Methods ────────────────────────────────────────────

    /// Quick estimate: will this order likely fill?
    ///
    /// A simplified interface for common use cases. Returns the probability
    /// as a simple float. For detailed analysis, use `estimate_fill_probability`.
    pub fn quick_probability(
        &mut self,
        our_quantity: Decimal,
        level_quantity: Decimal,
        expected_volume: Decimal,
    ) -> f64 {
        let result =
            self.estimate_fill_probability(our_quantity, level_quantity, None, expected_volume);
        result.probability
    }

    /// Estimate the expected partial fill quantity.
    ///
    /// For a given volume traded through the level, estimate how much of
    /// our order would be filled based on queue position.
    pub fn expected_partial_fill(
        &mut self,
        our_quantity: Decimal,
        level_quantity: Decimal,
        volume_through_level: Decimal,
    ) -> Decimal {
        let pos = self.estimate_queue_position(our_quantity, level_quantity, None);
        pos.partial_fill(volume_through_level)
    }
}

impl fmt::Display for FillModel {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        write!(
            f,
            "FillModel(queue={}, cancel_rate={:.0}%, partial={})",
            self.config.queue_model,
            self.config.cancel_rate * 100.0,
            self.config.allow_partial,
        )
    }
}

// ===========================================================================
// TESTS
// ===========================================================================

#[cfg(test)]
mod tests {
    use super::*;
    use rust_decimal_macros::dec;

    // ── QueuePositionModel ─────────────────────────────────────────────

    #[test]
    fn test_queue_model_default() {
        let model = QueuePositionModel::default();
        assert_eq!(model, QueuePositionModel::BackOfQueue);
    }

    #[test]
    fn test_queue_model_display() {
        assert_eq!(format!("{}", QueuePositionModel::Uniform), "Uniform");
        assert_eq!(
            format!("{}", QueuePositionModel::BackOfQueue),
            "BackOfQueue"
        );
        assert_eq!(
            format!("{}", QueuePositionModel::FrontOfQueue),
            "FrontOfQueue"
        );
        assert_eq!(format!("{}", QueuePositionModel::ProRata), "ProRata");
        assert_eq!(format!("{}", QueuePositionModel::Fractile), "Fractile");
    }

    // ── QueuePosition ──────────────────────────────────────────────────

    #[test]
    fn test_queue_position_depth_fraction() {
        let pos = QueuePosition {
            quantity_ahead: dec!(3.0),
            our_quantity: dec!(1.0),
            level_quantity: dec!(10.0),
            fractile: 0.3,
            model: QueuePositionModel::BackOfQueue,
        };
        let frac = pos.depth_fraction();
        assert!((frac - 0.3).abs() < 0.01);
    }

    #[test]
    fn test_queue_position_depth_fraction_empty() {
        let pos = QueuePosition {
            quantity_ahead: dec!(0),
            our_quantity: dec!(1.0),
            level_quantity: dec!(0),
            fractile: 0.0,
            model: QueuePositionModel::FrontOfQueue,
        };
        assert_eq!(pos.depth_fraction(), 0.0);
    }

    #[test]
    fn test_queue_position_is_front() {
        let front = QueuePosition {
            quantity_ahead: dec!(0),
            our_quantity: dec!(1.0),
            level_quantity: dec!(5.0),
            fractile: 0.0,
            model: QueuePositionModel::FrontOfQueue,
        };
        assert!(front.is_front());

        let back = QueuePosition {
            quantity_ahead: dec!(4.0),
            our_quantity: dec!(1.0),
            level_quantity: dec!(5.0),
            fractile: 0.8,
            model: QueuePositionModel::BackOfQueue,
        };
        assert!(!back.is_front());
    }

    #[test]
    fn test_queue_position_would_fill() {
        let pos = QueuePosition {
            quantity_ahead: dec!(3.0),
            our_quantity: dec!(2.0),
            level_quantity: dec!(10.0),
            fractile: 0.3,
            model: QueuePositionModel::BackOfQueue,
        };

        assert!(!pos.would_fill(dec!(3.0))); // Only reaches us.
        assert!(!pos.would_fill(dec!(4.0))); // Partial fill.
        assert!(pos.would_fill(dec!(5.0))); // Exactly fills us.
        assert!(pos.would_fill(dec!(10.0))); // Overshoot.
    }

    #[test]
    fn test_queue_position_partial_fill() {
        let pos = QueuePosition {
            quantity_ahead: dec!(3.0),
            our_quantity: dec!(2.0),
            level_quantity: dec!(10.0),
            fractile: 0.3,
            model: QueuePositionModel::BackOfQueue,
        };

        assert_eq!(pos.partial_fill(dec!(2.0)), dec!(0)); // Doesn't reach us.
        assert_eq!(pos.partial_fill(dec!(3.0)), dec!(0)); // Exactly at our position.
        assert_eq!(pos.partial_fill(dec!(4.0)), dec!(1.0)); // Partial fill.
        assert_eq!(pos.partial_fill(dec!(5.0)), dec!(2.0)); // Full fill.
        assert_eq!(pos.partial_fill(dec!(8.0)), dec!(2.0)); // Capped at our quantity.
    }

    #[test]
    fn test_queue_position_display() {
        let pos = QueuePosition {
            quantity_ahead: dec!(3.0),
            our_quantity: dec!(1.0),
            level_quantity: dec!(10.0),
            fractile: 0.3,
            model: QueuePositionModel::BackOfQueue,
        };
        let s = format!("{}", pos);
        assert!(s.contains("QueuePos"));
        assert!(s.contains("BackOfQueue"));
    }

    // ── FillProbability ────────────────────────────────────────────────

    #[test]
    fn test_fill_probability_is_likely() {
        let prob = FillProbability {
            probability: 0.8,
            expected_fill: dec!(0.8),
            expected_time: None,
            queue_position: QueuePosition {
                quantity_ahead: dec!(0),
                our_quantity: dec!(1.0),
                level_quantity: dec!(1.0),
                fractile: 0.0,
                model: QueuePositionModel::FrontOfQueue,
            },
        };
        assert!(prob.is_likely());
        assert!(!prob.is_unlikely());
        assert_eq!(prob.confidence_label(), "likely");
    }

    #[test]
    fn test_fill_probability_is_unlikely() {
        let prob = FillProbability {
            probability: 0.05,
            expected_fill: dec!(0.05),
            expected_time: None,
            queue_position: QueuePosition {
                quantity_ahead: dec!(9.0),
                our_quantity: dec!(1.0),
                level_quantity: dec!(10.0),
                fractile: 0.9,
                model: QueuePositionModel::BackOfQueue,
            },
        };
        assert!(!prob.is_likely());
        assert!(prob.is_unlikely());
        assert_eq!(prob.confidence_label(), "very unlikely");
    }

    #[test]
    fn test_fill_probability_confidence_labels() {
        let make = |p: f64| FillProbability {
            probability: p,
            expected_fill: Decimal::ZERO,
            expected_time: None,
            queue_position: QueuePosition {
                quantity_ahead: Decimal::ZERO,
                our_quantity: Decimal::ZERO,
                level_quantity: Decimal::ZERO,
                fractile: 0.0,
                model: QueuePositionModel::Uniform,
            },
        };

        assert_eq!(make(0.95).confidence_label(), "very likely");
        assert_eq!(make(0.75).confidence_label(), "likely");
        assert_eq!(make(0.5).confidence_label(), "moderate");
        assert_eq!(make(0.2).confidence_label(), "unlikely");
        assert_eq!(make(0.05).confidence_label(), "very unlikely");
    }

    #[test]
    fn test_fill_probability_display() {
        let prob = FillProbability {
            probability: 0.75,
            expected_fill: dec!(0.75),
            expected_time: None,
            queue_position: QueuePosition {
                quantity_ahead: dec!(1.0),
                our_quantity: dec!(1.0),
                level_quantity: dec!(5.0),
                fractile: 0.2,
                model: QueuePositionModel::Uniform,
            },
        };
        let s = format!("{}", prob);
        assert!(s.contains("75.0%"));
        assert!(s.contains("likely"));
    }

    // ── FillModelConfig ────────────────────────────────────────────────

    #[test]
    fn test_config_default() {
        let config = FillModelConfig::default();
        assert_eq!(config.queue_model, QueuePositionModel::BackOfQueue);
        assert!((config.cancel_rate - 0.5).abs() < 0.01);
        assert!(config.allow_partial);
    }

    #[test]
    fn test_config_builder() {
        let config = FillModelConfig::default()
            .with_queue_model(QueuePositionModel::Uniform)
            .with_cancel_rate(0.3)
            .with_queue_fractile(0.6)
            .with_partial_fills(false)
            .with_volume_lookback(Duration::from_secs(60));

        assert_eq!(config.queue_model, QueuePositionModel::Uniform);
        assert!((config.cancel_rate - 0.3).abs() < 0.01);
        assert!((config.queue_fractile - 0.6).abs() < 0.01);
        assert!(!config.allow_partial);
        assert_eq!(config.volume_lookback, Some(Duration::from_secs(60)));
    }

    #[test]
    fn test_config_clamp_cancel_rate() {
        let config = FillModelConfig::default().with_cancel_rate(1.5);
        assert!((config.cancel_rate - 1.0).abs() < 0.01);

        let config = FillModelConfig::default().with_cancel_rate(-0.5);
        assert!((config.cancel_rate - 0.0).abs() < 0.01);
    }

    #[test]
    fn test_config_clamp_fractile() {
        let config = FillModelConfig::default().with_queue_fractile(2.0);
        assert!((config.queue_fractile - 1.0).abs() < 0.01);
    }

    // ── FillModelStats ─────────────────────────────────────────────────

    #[test]
    fn test_stats_default() {
        let stats = FillModelStats::default();
        assert_eq!(stats.estimates_computed, 0);
        assert_eq!(stats.mean_probability(), 0.0);
    }

    #[test]
    fn test_stats_mean_probability() {
        let stats = FillModelStats {
            estimates_computed: 4,
            predicted_likely: 2,
            predicted_unlikely: 1,
            sum_probability: 2.0,
            position_estimates: 4,
        };
        assert!((stats.mean_probability() - 0.5).abs() < 0.01);
    }

    #[test]
    fn test_stats_reset() {
        let mut stats = FillModelStats {
            estimates_computed: 10,
            predicted_likely: 5,
            predicted_unlikely: 3,
            sum_probability: 5.0,
            position_estimates: 10,
        };
        stats.reset();
        assert_eq!(stats.estimates_computed, 0);
        assert_eq!(stats.sum_probability, 0.0);
    }

    #[test]
    fn test_stats_display() {
        let stats = FillModelStats {
            estimates_computed: 100,
            predicted_likely: 60,
            predicted_unlikely: 10,
            sum_probability: 55.0,
            position_estimates: 100,
        };
        let s = format!("{}", stats);
        assert!(s.contains("estimates=100"));
        assert!(s.contains("55.0%"));
    }

    // ── FillModel Queue Position ───────────────────────────────────────

    #[test]
    fn test_queue_position_back_of_queue() {
        let mut model = FillModel::new(QueuePositionModel::BackOfQueue);
        let pos = model.estimate_queue_position(dec!(1.0), dec!(10.0), None);

        // Back of queue: ahead = level - our = 9.0
        assert_eq!(pos.quantity_ahead, dec!(9.0));
        assert_eq!(pos.our_quantity, dec!(1.0));
        assert_eq!(pos.model, QueuePositionModel::BackOfQueue);
    }

    #[test]
    fn test_queue_position_front_of_queue() {
        let mut model = FillModel::new(QueuePositionModel::FrontOfQueue);
        let pos = model.estimate_queue_position(dec!(1.0), dec!(10.0), None);

        assert_eq!(pos.quantity_ahead, dec!(0));
        assert!(pos.is_front());
    }

    #[test]
    fn test_queue_position_uniform() {
        let mut model = FillModel::new(QueuePositionModel::Uniform);
        let pos = model.estimate_queue_position(dec!(1.0), dec!(10.0), None);

        // Uniform: ahead = (level - our) / 2 = 4.5
        assert_eq!(pos.quantity_ahead, dec!(4.5));
    }

    #[test]
    fn test_queue_position_pro_rata() {
        let mut model = FillModel::new(QueuePositionModel::ProRata);
        let pos = model.estimate_queue_position(dec!(1.0), dec!(10.0), None);

        // Pro-rata: position doesn't matter.
        assert_eq!(pos.quantity_ahead, dec!(0));
    }

    #[test]
    fn test_queue_position_known_ahead() {
        let mut model = FillModel::new(QueuePositionModel::BackOfQueue);
        let pos = model.estimate_queue_position(dec!(1.0), dec!(10.0), Some(dec!(5.0)));

        // Known ahead overrides model.
        assert_eq!(pos.quantity_ahead, dec!(5.0));
    }

    #[test]
    fn test_queue_position_stats_counted() {
        let mut model = FillModel::new(QueuePositionModel::Uniform);
        model.estimate_queue_position(dec!(1.0), dec!(10.0), None);
        model.estimate_queue_position(dec!(2.0), dec!(10.0), None);

        assert_eq!(model.stats().position_estimates, 2);
    }

    // ── FillModel Fill Probability ─────────────────────────────────────

    #[test]
    fn test_fill_probability_full_volume() {
        let mut model = FillModel::new(QueuePositionModel::FrontOfQueue);
        let prob = model.estimate_fill_probability(
            dec!(1.0),
            dec!(10.0),
            Some(dec!(0)),
            dec!(5.0), // More than enough volume.
        );

        assert!((prob.probability - 1.0).abs() < 0.01);
    }

    #[test]
    fn test_fill_probability_no_volume() {
        let mut model = FillModel::new(QueuePositionModel::BackOfQueue);
        let prob = model.estimate_fill_probability(
            dec!(1.0),
            dec!(10.0),
            None,
            dec!(0), // No volume → no fills.
        );

        assert_eq!(prob.probability, 0.0);
    }

    #[test]
    fn test_fill_probability_insufficient_volume() {
        let mut model = FillModel::new(QueuePositionModel::BackOfQueue);
        // Our order is at back, 9 units ahead, volume = 5 (doesn't reach us).
        let prob = model.estimate_fill_probability(dec!(1.0), dec!(10.0), None, dec!(3.0));

        // With cancel_rate = 0.5, effective_ahead = 4.5, volume = 3.
        // Volume < effective_ahead → very low probability.
        assert!(prob.probability < 0.3);
    }

    #[test]
    fn test_fill_probability_partial_penetration() {
        let mut model = FillModel::new(QueuePositionModel::FrontOfQueue);
        // Front of queue, our qty = 2.0, volume = 1.0
        let prob = model.estimate_fill_probability(dec!(2.0), dec!(5.0), Some(dec!(0)), dec!(1.0));

        // Volume penetrates our position partially.
        assert!(prob.probability > 0.0);
        assert!(prob.probability < 1.0);
    }

    #[test]
    fn test_fill_probability_pro_rata() {
        let mut model = FillModel::new(QueuePositionModel::ProRata);
        // Our share = 2/10 = 20%. Volume = 10 → expected fill = 2.0 = 100%.
        let prob = model.estimate_fill_probability(dec!(2.0), dec!(10.0), None, dec!(10.0));

        assert!((prob.probability - 1.0).abs() < 0.01);
    }

    #[test]
    fn test_fill_probability_pro_rata_partial() {
        let mut model = FillModel::new(QueuePositionModel::ProRata);
        // Our share = 2/10 = 20%. Volume = 5 → expected fill = 1.0 → 50%.
        let prob = model.estimate_fill_probability(dec!(2.0), dec!(10.0), None, dec!(5.0));

        assert!((prob.probability - 0.5).abs() < 0.01);
    }

    #[test]
    fn test_fill_probability_stats_updated() {
        let mut model = FillModel::new(QueuePositionModel::FrontOfQueue);

        // A likely fill.
        model.estimate_fill_probability(dec!(1.0), dec!(5.0), Some(dec!(0)), dec!(10.0));

        // An unlikely fill.
        model.estimate_fill_probability(dec!(1.0), dec!(100.0), None, dec!(0.001));

        assert_eq!(model.stats().estimates_computed, 2);
        assert!(model.stats().predicted_likely >= 1);
    }

    #[test]
    fn test_fill_probability_expected_fill() {
        let mut model = FillModel::new(QueuePositionModel::FrontOfQueue);
        let prob = model.estimate_fill_probability(
            dec!(2.0),
            dec!(5.0),
            Some(dec!(0)),
            dec!(1.0), // Partial fill.
        );

        // expected_fill = probability * our_quantity
        let expected: f64 = prob.expected_fill.try_into().unwrap_or(0.0);
        let prob_times_qty = prob.probability * 2.0;
        assert!((expected - prob_times_qty).abs() < 0.1);
    }

    // ── FillModel Convenience ──────────────────────────────────────────

    #[test]
    fn test_quick_probability() {
        let mut model = FillModel::new(QueuePositionModel::FrontOfQueue);
        let prob = model.quick_probability(dec!(1.0), dec!(5.0), dec!(10.0));
        assert!(prob > 0.5);
    }

    #[test]
    fn test_expected_partial_fill() {
        let mut model = FillModel::new(QueuePositionModel::FrontOfQueue);
        let fill = model.expected_partial_fill(dec!(2.0), dec!(5.0), dec!(1.0));
        // Front of queue, 1.0 volume → partial fill of 1.0.
        assert_eq!(fill, dec!(1.0));
    }

    #[test]
    fn test_expected_partial_fill_back_of_queue() {
        let mut model = FillModel::new(QueuePositionModel::BackOfQueue);
        // Back of queue with 1 unit, level = 5. Ahead = 4.
        // Volume = 3 → doesn't reach us.
        let fill = model.expected_partial_fill(dec!(1.0), dec!(5.0), dec!(3.0));
        assert_eq!(fill, dec!(0));
    }

    // ── FillModel Display ──────────────────────────────────────────────

    #[test]
    fn test_fill_model_display() {
        let model = FillModel::new(QueuePositionModel::Uniform);
        let s = format!("{}", model);
        assert!(s.contains("Uniform"));
        assert!(s.contains("50%")); // cancel_rate = 0.5
    }

    // ── FillModel Reset ────────────────────────────────────────────────

    #[test]
    fn test_fill_model_reset_stats() {
        let mut model = FillModel::new(QueuePositionModel::BackOfQueue);
        model.quick_probability(dec!(1.0), dec!(5.0), dec!(10.0));
        assert!(model.stats().estimates_computed > 0);

        model.reset_stats();
        assert_eq!(model.stats().estimates_computed, 0);
    }

    // ── Edge Cases ─────────────────────────────────────────────────────

    #[test]
    fn test_zero_level_quantity() {
        let mut model = FillModel::new(QueuePositionModel::BackOfQueue);
        let pos = model.estimate_queue_position(dec!(0), dec!(0), None);
        assert_eq!(pos.quantity_ahead, dec!(0));
    }

    #[test]
    fn test_our_quantity_exceeds_level() {
        let mut model = FillModel::new(QueuePositionModel::BackOfQueue);
        // Edge case: our quantity > level quantity (shouldn't happen, but handle gracefully).
        let pos = model.estimate_queue_position(dec!(10.0), dec!(5.0), None);
        // others = level - our = -5 → clamped to 0.
        assert_eq!(pos.quantity_ahead, dec!(0));
    }

    #[test]
    fn test_fill_probability_zero_quantity() {
        let mut model = FillModel::new(QueuePositionModel::FrontOfQueue);
        let prob = model.estimate_fill_probability(dec!(0), dec!(10.0), Some(dec!(0)), dec!(5.0));
        assert_eq!(prob.probability, 0.0);
    }

    #[test]
    fn test_fill_probability_negative_volume() {
        let mut model = FillModel::new(QueuePositionModel::FrontOfQueue);
        let prob =
            model.estimate_fill_probability(dec!(1.0), dec!(10.0), Some(dec!(0)), dec!(-5.0));
        assert_eq!(prob.probability, 0.0);
    }

    #[test]
    fn test_fill_model_with_config() {
        let config = FillModelConfig::default()
            .with_queue_model(QueuePositionModel::Uniform)
            .with_cancel_rate(0.3);
        let model = FillModel::with_config(config);
        assert_eq!(model.config().queue_model, QueuePositionModel::Uniform);
        assert!((model.config().cancel_rate - 0.3).abs() < 0.01);
    }
}
