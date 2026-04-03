//! # janus-regime — Market Regime Detection for JANUS
//!
//! This crate provides market regime classification to guide strategy selection
//! in the JANUS trading system. It detects whether the market is trending,
//! mean-reverting, volatile, or uncertain, and recommends the appropriate
//! trading approach.
//!
//! ## Detection Methods
//!
//! Three detection approaches are available, each with different trade-offs:
//!
//! 1. **Indicator-Based** ([`RegimeDetector`]) — Fast, rule-based classification
//!    using ADX, Bollinger Bands, ATR, and EMA. Best for real-time decision-making
//!    with minimal warmup.
//!
//! 2. **Hidden Markov Model** ([`HMMRegimeDetector`]) — Statistical learning from
//!    return distributions. Adapts to changing market dynamics over time.
//!
//! 3. **Ensemble** ([`EnsembleRegimeDetector`]) — Combines both methods for
//!    robustness. Boosts confidence when methods agree, reduces it when they
//!    disagree. **Recommended for production use.**
//!
//! ## Strategy Router
//!
//! The [`router::EnhancedRouter`] wraps any detection method and provides
//! per-asset regime tracking with strategy recommendations. It manages
//! detector lifecycle, regime change logging, and emits [`router::RoutedSignal`]s
//! that downstream services can act on.
//!
//! ## Quick Start
//!
//! ```rust
//! use janus_regime::{EnsembleRegimeDetector, MarketRegime};
//!
//! let mut detector = EnsembleRegimeDetector::default_config();
//!
//! // Feed OHLC bars
//! for i in 0..300 {
//!     let price = 100.0 + i as f64 * 0.5;
//!     let result = detector.update(price + 1.0, price - 1.0, price);
//!     if detector.is_ready() {
//!         match result.regime {
//!             MarketRegime::Trending(_) => { /* use trend-following */ }
//!             MarketRegime::MeanReverting => { /* use mean reversion */ }
//!             MarketRegime::Volatile => { /* reduce exposure */ }
//!             MarketRegime::Uncertain => { /* stay cash */ }
//!         }
//!     }
//! }
//! ```
//!
//! ## Architecture
//!
//! This crate is designed to complement — not replace — the JANUS neuromorphic
//! brain. It provides a fast, classical market state classification that can
//! run alongside the deeper neuromorphic signals:
//!
//! - **Regime says Volatile + Amygdala fear high** → strong confirmation to reduce exposure
//! - **Regime says Trending + Prefrontal agrees** → high-confidence trend signal
//! - **Regime says Uncertain** → defer entirely to neuromorphic output
//! - **Regime and brain disagree** → lower position size, flag for review
//!
//! ## Origin
//!
//! Ported from the `kraken` trading bot's regime detection system (24 tests passing),
//! adapted for the JANUS type system and workspace conventions.

// Module declarations
mod detector;
mod ensemble;
mod hmm;
pub mod indicators;
pub mod router;
pub mod types;

// ============================================================================
// Primary Re-exports
// ============================================================================

// Detector implementations
pub use detector::RegimeDetector;
pub use ensemble::{EnsembleConfig, EnsembleRegimeDetector, EnsembleResult, EnsembleStatus};
pub use hmm::{HMMConfig, HMMRegimeDetector};

// Core types
pub use types::{
    MarketRegime, RecommendedStrategy, RegimeConfidence, RegimeConfig, TrendDirection,
};

// Indicator types (re-exported for consumers that need raw indicator access)
pub use indicators::{ADX, ATR, BollingerBands, BollingerBandsValues, EMA, RSI};

// Router types
pub use router::{
    ActiveStrategy, AssetSummary, DetectionMethod, EnhancedRouter, EnhancedRouterConfig,
    RoutedSignal,
};
