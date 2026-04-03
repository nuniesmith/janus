//! # Market Impact Models
//!
//! Models for estimating the price impact of executing orders against the
//! limit order book. Market impact is the adverse price movement caused by
//! consuming liquidity from the book.
//!
//! # Models
//!
//! | Model           | Formula                          | Best For                    |
//! |-----------------|----------------------------------|-----------------------------|
//! | Linear          | Δp = λ × q                       | Small orders, liquid markets|
//! | SquareRoot      | Δp = λ × √q × σ                 | Large orders (Almgren)      |
//! | PowerLaw        | Δp = λ × q^α                     | Empirical calibration       |
//! | AlmgrenChriss   | Δp = γ×q + η×(q/T)              | Optimal execution (TWAP)    |
//! | OrderBookBased  | Δp = f(book_depth, q)            | Real-time book analysis     |
//!
//! # Usage
//!
//! ```rust,ignore
//! use janus_lob::market_impact::*;
//!
//! // Simple linear model: 0.1 bps per unit traded
//! let model = MarketImpactModel::linear(0.1);
//! let impact = model.estimate_impact_bps(dec!(2.0), dec!(67000.0));
//! println!("Impact: {:.2} bps", impact);
//!
//! // Square root model (Almgren-style)
//! let model = MarketImpactModel::square_root(0.5, 0.02); // lambda=0.5, sigma=2%
//! let impact = model.estimate_impact_bps(dec!(10.0), dec!(67000.0));
//!
//! // Order-book-aware model
//! let model = MarketImpactModel::order_book_based();
//! let impact = model.estimate_from_book(&book, Side::Buy, dec!(5.0));
//! ```

use rust_decimal::Decimal;
use serde::{Deserialize, Serialize};
use std::fmt;
use std::time::Duration;
use tracing::trace;

// ---------------------------------------------------------------------------
// Impact Estimate
// ---------------------------------------------------------------------------

/// The result of an impact estimation.
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct ImpactEstimate {
    /// Estimated price impact in basis points.
    pub impact_bps: f64,

    /// Estimated price impact in absolute price units.
    pub impact_price: f64,

    /// The model that produced this estimate.
    pub model: String,

    /// Input quantity.
    pub quantity: f64,

    /// Input reference price.
    pub reference_price: f64,

    /// Effective execution price after impact.
    pub effective_price: f64,

    /// Cost of impact in quote currency (impact_price × quantity).
    pub impact_cost: f64,
}

impl ImpactEstimate {
    /// Create an estimate from basis points.
    fn from_bps(impact_bps: f64, quantity: f64, reference_price: f64, model: &str) -> Self {
        let impact_price = reference_price * impact_bps / 10_000.0;
        let effective_price = reference_price + impact_price;
        let impact_cost = impact_price * quantity;

        Self {
            impact_bps,
            impact_price,
            model: model.to_string(),
            quantity,
            reference_price,
            effective_price,
            impact_cost,
        }
    }
}

impl fmt::Display for ImpactEstimate {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        write!(
            f,
            "Impact[{}]: {:.2} bps ({:.4} price units), effective={:.2}, cost={:.4}",
            self.model, self.impact_bps, self.impact_price, self.effective_price, self.impact_cost,
        )
    }
}

// ---------------------------------------------------------------------------
// Market Impact Model
// ---------------------------------------------------------------------------

/// Market impact model for estimating the price effect of order execution.
///
/// Multiple model variants are supported, from simple linear models to
/// more sophisticated empirical and order-book-based approaches.
#[derive(Debug, Clone, Serialize, Deserialize)]
pub enum MarketImpactModel {
    /// No market impact (fills at exact book price).
    None,

    /// Linear impact: Δp = λ × q.
    ///
    /// Simple proportional impact. Good for small orders in liquid markets.
    /// - `lambda_bps` — impact in basis points per unit quantity.
    Linear {
        /// Impact coefficient (bps per unit).
        lambda_bps: f64,
    },

    /// Square-root impact (Almgren 2005): Δp = λ × σ × √q.
    ///
    /// Widely used empirical model. Impact grows sub-linearly with size,
    /// reflecting the concave relationship observed in real markets.
    /// - `lambda` — dimensionless impact coefficient.
    /// - `sigma` — daily volatility (as a fraction, e.g. 0.02 for 2%).
    SquareRoot {
        /// Dimensionless impact coefficient.
        lambda: f64,
        /// Daily volatility as a fraction.
        sigma: f64,
    },

    /// Power-law impact: Δp = λ × q^α.
    ///
    /// Generalisation of linear (α=1) and square-root (α=0.5) models.
    /// The exponent α can be calibrated from historical data.
    PowerLaw {
        /// Impact coefficient.
        lambda: f64,
        /// Power-law exponent (0 < α ≤ 1, typically 0.4–0.7).
        alpha: f64,
    },

    /// Almgren-Chriss (2001) temporary + permanent impact model.
    ///
    /// Splits impact into:
    /// - **Permanent impact**: γ × q — moves the equilibrium price permanently.
    /// - **Temporary impact**: η × (q / T) — additional cost from execution speed.
    ///
    /// Used for optimal execution (TWAP/VWAP) scheduling.
    AlmgrenChriss {
        /// Permanent impact coefficient (bps per unit).
        gamma: f64,
        /// Temporary impact coefficient (bps per unit per time).
        eta: f64,
        /// Execution horizon (seconds).
        horizon_secs: f64,
        /// Daily volatility.
        sigma: f64,
    },

    /// Order-book-based impact estimation.
    ///
    /// Uses the actual book depth to estimate impact by computing
    /// the VWAP if the given quantity were swept from the book. This is
    /// the most accurate model but requires a live book reference.
    OrderBookBased {
        /// Additional fixed slippage (bps) on top of book-derived impact.
        additional_slippage_bps: f64,
    },
}

impl MarketImpactModel {
    // ── Constructors ───────────────────────────────────────────────────

    /// No impact model (fills at exact book price).
    pub fn none() -> Self {
        MarketImpactModel::None
    }

    /// Linear impact model.
    ///
    /// `lambda_bps` — impact in basis points per unit quantity.
    pub fn linear(lambda_bps: f64) -> Self {
        MarketImpactModel::Linear { lambda_bps }
    }

    /// Square-root impact model (Almgren 2005).
    ///
    /// - `lambda` — dimensionless impact coefficient (typically 0.1–1.0).
    /// - `sigma` — daily volatility as a fraction (e.g. 0.02 for 2%).
    pub fn square_root(lambda: f64, sigma: f64) -> Self {
        MarketImpactModel::SquareRoot { lambda, sigma }
    }

    /// Power-law impact model.
    ///
    /// - `lambda` — impact coefficient.
    /// - `alpha` — power-law exponent (0 < α ≤ 1).
    pub fn power_law(lambda: f64, alpha: f64) -> Self {
        MarketImpactModel::PowerLaw {
            lambda,
            alpha: alpha.clamp(0.01, 1.0),
        }
    }

    /// Almgren-Chriss temporary + permanent impact model.
    ///
    /// - `gamma` — permanent impact coefficient.
    /// - `eta` — temporary impact coefficient.
    /// - `horizon` — execution horizon (time to complete the order).
    /// - `sigma` — daily volatility.
    pub fn almgren_chriss(gamma: f64, eta: f64, horizon: Duration, sigma: f64) -> Self {
        MarketImpactModel::AlmgrenChriss {
            gamma,
            eta,
            horizon_secs: horizon.as_secs_f64(),
            sigma,
        }
    }

    /// Order-book-based impact model.
    ///
    /// Uses the actual book depth to compute impact. Optionally adds
    /// a fixed slippage on top.
    pub fn order_book_based() -> Self {
        MarketImpactModel::OrderBookBased {
            additional_slippage_bps: 0.0,
        }
    }

    /// Order-book-based with additional fixed slippage.
    pub fn order_book_based_with_slippage(additional_bps: f64) -> Self {
        MarketImpactModel::OrderBookBased {
            additional_slippage_bps: additional_bps,
        }
    }

    // ── Estimation ─────────────────────────────────────────────────────

    /// Estimate market impact in basis points for a given quantity and price.
    ///
    /// This method works for all model variants except `OrderBookBased`,
    /// which requires a book reference (use `estimate_from_book` instead).
    pub fn estimate_impact_bps(&self, quantity: Decimal, reference_price: Decimal) -> f64 {
        let q: f64 = quantity.try_into().unwrap_or(0.0);
        let p: f64 = reference_price.try_into().unwrap_or(1.0);

        if q <= 0.0 || p <= 0.0 {
            return 0.0;
        }

        let bps = match self {
            MarketImpactModel::None => 0.0,

            MarketImpactModel::Linear { lambda_bps } => lambda_bps * q,

            MarketImpactModel::SquareRoot { lambda, sigma } => {
                // Δp/p = λ × σ × √q → in bps: λ × σ × √q × 10000
                lambda * sigma * q.sqrt() * 10_000.0
            }

            MarketImpactModel::PowerLaw { lambda, alpha } => {
                // Δp = λ × q^α → in bps relative to price
                lambda * q.powf(*alpha) * 10_000.0 / p
            }

            MarketImpactModel::AlmgrenChriss {
                gamma,
                eta,
                horizon_secs,
                sigma,
            } => {
                // Permanent: γ × q (bps)
                // Temporary: η × (q / T) (bps)
                let permanent = gamma * q;
                let temporary = if *horizon_secs > 0.0 {
                    eta * q / horizon_secs
                } else {
                    eta * q // Instantaneous execution
                };

                // Scale by volatility
                (permanent + temporary) * sigma * 10_000.0
            }

            MarketImpactModel::OrderBookBased {
                additional_slippage_bps,
            } => {
                // Without a book reference, fall back to the additional slippage.
                // For accurate estimation, use `estimate_from_book()`.
                *additional_slippage_bps
            }
        };

        trace!(
            model = %self,
            quantity = q,
            price = p,
            impact_bps = bps,
            "Market impact estimated"
        );

        bps
    }

    /// Produce a full impact estimate.
    pub fn estimate(&self, quantity: Decimal, reference_price: Decimal) -> ImpactEstimate {
        let q: f64 = quantity.try_into().unwrap_or(0.0);
        let p: f64 = reference_price.try_into().unwrap_or(1.0);
        let bps = self.estimate_impact_bps(quantity, reference_price);

        ImpactEstimate::from_bps(bps, q, p, &self.to_string())
    }

    /// Estimate market impact using the actual order book depth.
    ///
    /// For the `OrderBookBased` variant this computes the VWAP for sweeping
    /// `qty` from the opposite side of the book, calculates the deviation
    /// from mid-price as impact in basis points, and adds any configured
    /// `additional_slippage_bps` on top.
    ///
    /// For other model variants this falls back to the parametric
    /// `estimate()` method using the book's mid-price as reference.
    pub fn estimate_from_book(
        &self,
        book: &crate::orderbook::OrderBook,
        side: crate::order_types::Side,
        qty: Decimal,
    ) -> ImpactEstimate {
        let mid = match book.mid_price() {
            Some(m) => m,
            None => {
                // No mid-price available — return zero-impact estimate.
                return ImpactEstimate::from_bps(
                    0.0,
                    qty.try_into().unwrap_or(0.0),
                    1.0,
                    self.model_name(),
                );
            }
        };

        let mid_f64: f64 = mid.try_into().unwrap_or(1.0);
        let q_f64: f64 = qty.try_into().unwrap_or(0.0);

        if q_f64 <= 0.0 || mid_f64 <= 0.0 {
            return ImpactEstimate::from_bps(0.0, q_f64, mid_f64, self.model_name());
        }

        match self {
            MarketImpactModel::OrderBookBased {
                additional_slippage_bps,
            } => {
                // Compute VWAP by sweeping the opposite side of the book.
                let vwap_opt = match side {
                    crate::order_types::Side::Buy => book.buy_vwap(qty),
                    crate::order_types::Side::Sell => book.sell_vwap(qty),
                };

                let book_impact_bps = match vwap_opt {
                    Some(vwap) => {
                        let vwap_f64: f64 = vwap.try_into().unwrap_or(mid_f64);
                        // Impact = deviation of VWAP from mid, in bps.
                        // For buys: (vwap − mid) / mid × 10_000  (positive = adverse)
                        // For sells: (mid − vwap) / mid × 10_000  (positive = adverse)
                        match side {
                            crate::order_types::Side::Buy => {
                                (vwap_f64 - mid_f64) / mid_f64 * 10_000.0
                            }
                            crate::order_types::Side::Sell => {
                                (mid_f64 - vwap_f64) / mid_f64 * 10_000.0
                            }
                        }
                    }
                    None => 0.0,
                };

                let total_bps = (book_impact_bps + additional_slippage_bps).max(0.0);

                trace!(
                    model = "OrderBookBased",
                    side = %side,
                    quantity = q_f64,
                    mid = mid_f64,
                    book_impact_bps = book_impact_bps,
                    additional_bps = additional_slippage_bps,
                    total_bps = total_bps,
                    "Book-based impact estimated"
                );

                ImpactEstimate::from_bps(total_bps, q_f64, mid_f64, "OrderBookBased")
            }
            // For all other model variants, fall back to the parametric estimate
            // using the book's mid-price as the reference price.
            _ => self.estimate(qty, mid),
        }
    }

    /// Get the model name for display purposes.
    pub fn model_name(&self) -> &'static str {
        match self {
            MarketImpactModel::None => "None",
            MarketImpactModel::Linear { .. } => "Linear",
            MarketImpactModel::SquareRoot { .. } => "SquareRoot",
            MarketImpactModel::PowerLaw { .. } => "PowerLaw",
            MarketImpactModel::AlmgrenChriss { .. } => "AlmgrenChriss",
            MarketImpactModel::OrderBookBased { .. } => "OrderBookBased",
        }
    }
}

impl Default for MarketImpactModel {
    fn default() -> Self {
        // Default: linear 0.5 bps per unit — conservative for crypto.
        MarketImpactModel::linear(0.5)
    }
}

impl fmt::Display for MarketImpactModel {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        match self {
            MarketImpactModel::None => write!(f, "None"),
            MarketImpactModel::Linear { lambda_bps } => {
                write!(f, "Linear(λ={:.2}bps/unit)", lambda_bps)
            }
            MarketImpactModel::SquareRoot { lambda, sigma } => {
                write!(f, "SqrtImpact(λ={:.3}, σ={:.4})", lambda, sigma)
            }
            MarketImpactModel::PowerLaw { lambda, alpha } => {
                write!(f, "PowerLaw(λ={:.3}, α={:.2})", lambda, alpha)
            }
            MarketImpactModel::AlmgrenChriss {
                gamma,
                eta,
                horizon_secs,
                sigma,
            } => {
                write!(
                    f,
                    "AlmgrenChriss(γ={:.3}, η={:.3}, T={:.0}s, σ={:.4})",
                    gamma, eta, horizon_secs, sigma,
                )
            }
            MarketImpactModel::OrderBookBased {
                additional_slippage_bps,
            } => {
                write!(f, "OrderBookBased(+{:.1}bps)", additional_slippage_bps)
            }
        }
    }
}

// ---------------------------------------------------------------------------
// Tests
// ---------------------------------------------------------------------------

#[cfg(test)]
mod tests {
    use super::*;
    use rust_decimal_macros::dec;

    // ── No impact ──────────────────────────────────────────────────────

    #[test]
    fn test_no_impact() {
        let model = MarketImpactModel::none();
        let bps = model.estimate_impact_bps(dec!(10.0), dec!(50000.0));
        assert_eq!(bps, 0.0);
    }

    #[test]
    fn test_no_impact_name() {
        assert_eq!(MarketImpactModel::none().model_name(), "None");
    }

    // ── Linear impact ──────────────────────────────────────────────────

    #[test]
    fn test_linear_impact() {
        let model = MarketImpactModel::linear(1.0); // 1 bps per unit
        let bps = model.estimate_impact_bps(dec!(5.0), dec!(50000.0));
        assert!((bps - 5.0).abs() < 0.001); // 5 units × 1 bps = 5 bps
    }

    #[test]
    fn test_linear_impact_zero_quantity() {
        let model = MarketImpactModel::linear(1.0);
        let bps = model.estimate_impact_bps(dec!(0.0), dec!(50000.0));
        assert_eq!(bps, 0.0);
    }

    #[test]
    fn test_linear_impact_display() {
        let model = MarketImpactModel::linear(0.5);
        let display = format!("{}", model);
        assert!(display.contains("Linear"));
        assert!(display.contains("0.50"));
    }

    // ── Square-root impact ─────────────────────────────────────────────

    #[test]
    fn test_sqrt_impact() {
        let model = MarketImpactModel::square_root(1.0, 0.02);
        let bps = model.estimate_impact_bps(dec!(4.0), dec!(50000.0));
        // λ × σ × √q × 10000 = 1.0 × 0.02 × 2.0 × 10000 = 400 bps
        assert!((bps - 400.0).abs() < 0.1);
    }

    #[test]
    fn test_sqrt_impact_grows_sublinearly() {
        let model = MarketImpactModel::square_root(0.5, 0.02);
        let impact_1 = model.estimate_impact_bps(dec!(1.0), dec!(50000.0));
        let impact_4 = model.estimate_impact_bps(dec!(4.0), dec!(50000.0));
        let impact_16 = model.estimate_impact_bps(dec!(16.0), dec!(50000.0));

        // √4 = 2×√1, so impact should double
        assert!((impact_4 / impact_1 - 2.0).abs() < 0.01);
        // √16 = 4×√1, so impact should quadruple
        assert!((impact_16 / impact_1 - 4.0).abs() < 0.01);
    }

    #[test]
    fn test_sqrt_impact_display() {
        let model = MarketImpactModel::square_root(0.5, 0.02);
        let display = format!("{}", model);
        assert!(display.contains("SqrtImpact"));
    }

    // ── Power-law impact ───────────────────────────────────────────────

    #[test]
    fn test_power_law_impact() {
        let model = MarketImpactModel::power_law(0.001, 0.5);
        let bps = model.estimate_impact_bps(dec!(4.0), dec!(100.0));
        // λ × q^α × 10000 / p = 0.001 × 4^0.5 × 10000 / 100 = 0.001 × 2 × 100 = 0.2
        assert!((bps - 0.2).abs() < 0.01);
    }

    #[test]
    fn test_power_law_alpha_clamped() {
        let model = MarketImpactModel::power_law(1.0, 5.0);
        match model {
            MarketImpactModel::PowerLaw { alpha, .. } => {
                assert_eq!(alpha, 1.0); // Clamped to max
            }
            _ => panic!("Expected PowerLaw variant"),
        }
    }

    #[test]
    fn test_power_law_display() {
        let model = MarketImpactModel::power_law(0.5, 0.6);
        let display = format!("{}", model);
        assert!(display.contains("PowerLaw"));
        assert!(display.contains("0.60"));
    }

    // ── Almgren-Chriss impact ──────────────────────────────────────────

    #[test]
    fn test_almgren_chriss_impact() {
        let model = MarketImpactModel::almgren_chriss(
            0.1,                       // gamma
            0.05,                      // eta
            Duration::from_secs(3600), // 1 hour horizon
            0.02,                      // 2% daily vol
        );
        let bps = model.estimate_impact_bps(dec!(10.0), dec!(50000.0));
        // permanent = 0.1 × 10 = 1.0
        // temporary = 0.05 × 10 / 3600 = 0.000139
        // total = (1.0 + 0.000139) × 0.02 × 10000 ≈ 200
        assert!(bps > 0.0);
    }

    #[test]
    fn test_almgren_chriss_instantaneous() {
        let model = MarketImpactModel::almgren_chriss(
            0.1,
            0.05,
            Duration::ZERO, // Instantaneous
            0.02,
        );
        let bps = model.estimate_impact_bps(dec!(10.0), dec!(50000.0));
        // With zero horizon, temporary impact = η × q (not divided by T)
        assert!(bps > 0.0);
    }

    #[test]
    fn test_almgren_chriss_display() {
        let model = MarketImpactModel::almgren_chriss(0.1, 0.05, Duration::from_secs(300), 0.02);
        let display = format!("{}", model);
        assert!(display.contains("AlmgrenChriss"));
        assert!(display.contains("300s"));
    }

    // ── Order book based ───────────────────────────────────────────────

    #[test]
    fn test_order_book_based_fallback() {
        let model = MarketImpactModel::order_book_based();
        // Without a book, should return additional_slippage_bps (0)
        let bps = model.estimate_impact_bps(dec!(10.0), dec!(50000.0));
        assert_eq!(bps, 0.0);
    }

    #[test]
    fn test_order_book_based_with_slippage() {
        let model = MarketImpactModel::order_book_based_with_slippage(3.0);
        let bps = model.estimate_impact_bps(dec!(10.0), dec!(50000.0));
        assert_eq!(bps, 3.0);
    }

    #[test]
    fn test_order_book_based_display() {
        let model = MarketImpactModel::order_book_based_with_slippage(2.5);
        let display = format!("{}", model);
        assert!(display.contains("OrderBookBased"));
        assert!(display.contains("2.5"));
    }

    // ── Default ────────────────────────────────────────────────────────

    #[test]
    fn test_default_model() {
        let model = MarketImpactModel::default();
        assert_eq!(model.model_name(), "Linear");
    }

    // ── Impact estimate ────────────────────────────────────────────────

    #[test]
    fn test_impact_estimate_full() {
        let model = MarketImpactModel::linear(1.0);
        let est = model.estimate(dec!(5.0), dec!(10000.0));

        assert!((est.impact_bps - 5.0).abs() < 0.01);
        assert!((est.quantity - 5.0).abs() < 0.01);
        assert!((est.reference_price - 10000.0).abs() < 0.01);
        // impact_price = 10000 × 5/10000 = 5.0 price units
        assert!((est.impact_price - 5.0).abs() < 0.01);
        // effective = 10000 + 5 = 10005
        assert!((est.effective_price - 10005.0).abs() < 0.1);
        // cost = 5.0 × 5.0 = 25.0
        assert!((est.impact_cost - 25.0).abs() < 0.1);
    }

    #[test]
    fn test_impact_estimate_display() {
        let model = MarketImpactModel::linear(1.0);
        let est = model.estimate(dec!(1.0), dec!(50000.0));
        let display = format!("{}", est);
        assert!(display.contains("Impact"));
        assert!(display.contains("bps"));
    }

    // ── Model name ─────────────────────────────────────────────────────

    #[test]
    fn test_model_names() {
        assert_eq!(MarketImpactModel::none().model_name(), "None");
        assert_eq!(MarketImpactModel::linear(1.0).model_name(), "Linear");
        assert_eq!(
            MarketImpactModel::square_root(1.0, 0.02).model_name(),
            "SquareRoot"
        );
        assert_eq!(
            MarketImpactModel::power_law(1.0, 0.5).model_name(),
            "PowerLaw"
        );
        assert_eq!(
            MarketImpactModel::almgren_chriss(0.1, 0.05, Duration::from_secs(60), 0.02)
                .model_name(),
            "AlmgrenChriss"
        );
        assert_eq!(
            MarketImpactModel::order_book_based().model_name(),
            "OrderBookBased"
        );
    }

    // ── Edge cases ─────────────────────────────────────────────────────

    #[test]
    fn test_negative_quantity_returns_zero() {
        let model = MarketImpactModel::linear(1.0);
        let bps = model.estimate_impact_bps(dec!(-1.0), dec!(50000.0));
        assert_eq!(bps, 0.0);
    }

    #[test]
    fn test_zero_price_returns_zero() {
        let model = MarketImpactModel::linear(1.0);
        let bps = model.estimate_impact_bps(dec!(1.0), dec!(0.0));
        assert_eq!(bps, 0.0);
    }

    // ── Book-based impact estimation ───────────────────────────────────

    #[test]
    fn test_estimate_from_book_order_book_based() {
        use crate::order_types::Side;
        use crate::orderbook::{OrderBook, OrderBookSnapshot, PriceLevel};

        let model = MarketImpactModel::order_book_based();

        let mut book = OrderBook::new("BTC/USDT");
        book.apply_snapshot(OrderBookSnapshot {
            symbol: "BTC/USDT".into(),
            bids: vec![
                PriceLevel::new(dec!(50000.0), dec!(2.0)),
                PriceLevel::new(dec!(49999.0), dec!(3.0)),
            ],
            asks: vec![
                PriceLevel::new(dec!(50001.0), dec!(1.0)),
                PriceLevel::new(dec!(50002.0), dec!(2.0)),
            ],
            timestamp: chrono::Utc::now(),
            sequence: 1,
        })
        .unwrap();

        // Mid = (50000 + 50001) / 2 = 50000.5
        // Buy 1.0: VWAP = 50001 (fills entirely at best ask)
        // impact = (50001 - 50000.5) / 50000.5 * 10000 ≈ 0.1 bps
        let est = model.estimate_from_book(&book, Side::Buy, dec!(1.0));
        assert!(
            est.impact_bps > 0.0,
            "Expected positive impact for buy, got {}",
            est.impact_bps
        );
        assert!(est.impact_bps < 2.0, "Impact too high: {}", est.impact_bps);

        // Sell 1.0: VWAP = 50000 (fills at best bid)
        // impact = (50000.5 - 50000) / 50000.5 * 10000 ≈ 0.1 bps
        let est_sell = model.estimate_from_book(&book, Side::Sell, dec!(1.0));
        assert!(
            est_sell.impact_bps > 0.0,
            "Expected positive impact for sell"
        );
        assert!(est_sell.impact_bps < 2.0);
    }

    #[test]
    fn test_estimate_from_book_sweeps_multiple_levels() {
        use crate::order_types::Side;
        use crate::orderbook::{OrderBook, OrderBookSnapshot, PriceLevel};

        let model = MarketImpactModel::order_book_based_with_slippage(1.0);

        let mut book = OrderBook::new("TEST");
        book.apply_snapshot(OrderBookSnapshot {
            symbol: "TEST".into(),
            bids: vec![
                PriceLevel::new(dec!(100.0), dec!(5.0)),
                PriceLevel::new(dec!(99.0), dec!(5.0)),
            ],
            asks: vec![
                PriceLevel::new(dec!(101.0), dec!(5.0)),
                PriceLevel::new(dec!(102.0), dec!(5.0)),
            ],
            timestamp: chrono::Utc::now(),
            sequence: 1,
        })
        .unwrap();

        // Mid = (100 + 101) / 2 = 100.5
        // Buy 8.0: VWAP = (5*101 + 3*102) / 8 = (505 + 306) / 8 = 101.375
        // book_impact = (101.375 - 100.5) / 100.5 * 10000 ≈ 87 bps + 1.0 additional
        let est = model.estimate_from_book(&book, Side::Buy, dec!(8.0));
        assert!(
            est.impact_bps > 80.0,
            "Expected substantial impact, got {}",
            est.impact_bps
        );
        assert!(
            est.impact_bps < 100.0,
            "Impact too high: {}",
            est.impact_bps
        );
    }

    #[test]
    fn test_estimate_from_book_fallback_for_linear() {
        use crate::order_types::Side;
        use crate::orderbook::{OrderBook, OrderBookSnapshot, PriceLevel};

        // Non-OrderBookBased variant should fall back to parametric estimate.
        let model = MarketImpactModel::linear(1.0);

        let mut book = OrderBook::new("TEST");
        book.apply_snapshot(OrderBookSnapshot {
            symbol: "TEST".into(),
            bids: vec![PriceLevel::new(dec!(100.0), dec!(5.0))],
            asks: vec![PriceLevel::new(dec!(101.0), dec!(5.0))],
            timestamp: chrono::Utc::now(),
            sequence: 1,
        })
        .unwrap();

        let est = model.estimate_from_book(&book, Side::Buy, dec!(3.0));
        // Linear: 1.0 bps/unit × 3 = 3.0 bps, using mid as reference price
        assert!(
            (est.impact_bps - 3.0).abs() < 0.01,
            "Expected ~3 bps, got {}",
            est.impact_bps
        );
    }

    #[test]
    fn test_estimate_from_book_empty_book() {
        use crate::order_types::Side;
        use crate::orderbook::OrderBook;

        let model = MarketImpactModel::order_book_based();
        let book = OrderBook::new("EMPTY");

        // No mid-price → zero impact
        let est = model.estimate_from_book(&book, Side::Buy, dec!(1.0));
        assert_eq!(est.impact_bps, 0.0);
    }

    // ── Almgren-Chriss horizon sensitivity ─────────────────────────────

    #[test]
    fn test_almgren_chriss_longer_horizon_reduces_temporary_impact() {
        let short_horizon =
            MarketImpactModel::almgren_chriss(0.1, 0.05, Duration::from_secs(60), 0.02);
        let long_horizon =
            MarketImpactModel::almgren_chriss(0.1, 0.05, Duration::from_secs(3600), 0.02);

        let qty = dec!(10.0);
        let price = dec!(50000.0);

        let short_bps = short_horizon.estimate_impact_bps(qty, price);
        let long_bps = long_horizon.estimate_impact_bps(qty, price);

        // Longer horizon should reduce temporary impact → lower total
        assert!(
            long_bps < short_bps,
            "Longer horizon should reduce impact: long={}, short={}",
            long_bps,
            short_bps
        );

        // Permanent component is the same, so long_bps should still be > 0
        assert!(long_bps > 0.0);
    }
}
