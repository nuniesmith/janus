//! Trading Predicates
//!
//! Logic Tensor Network predicates for trading rules and compliance.

pub type PredicateFn = Box<dyn Fn(&MarketState) -> f32 + Send + Sync>;

pub struct TradingPredicate {
    pub name: String,
    pub lukasiewicz_fn: PredicateFn,
}

impl TradingPredicate {
    pub fn new(name: impl Into<String>, lukasiewicz_fn: PredicateFn) -> Self {
        Self {
            name: name.into(),
            lukasiewicz_fn,
        }
    }

    /// Evaluate predicate on market state
    /// Returns value in [0, 1] where 1.0 = fully satisfied
    pub fn evaluate(&self, state: &MarketState) -> f32 {
        (self.lukasiewicz_fn)(state).clamp(0.0, 1.0)
    }
}

// Placeholder market state
#[derive(Clone)]
pub struct MarketState {
    pub price: f32,
    pub volume: f32,
    pub position_size: f32,
    pub portfolio_value: f32,
    pub hour: u8,
}

/// Common trading predicates
pub mod common_predicates {
    use super::*;

    /// Price is below maximum buy threshold
    pub fn price_below_max(max_price: f32) -> TradingPredicate {
        TradingPredicate::new(
            "price_below_max",
            Box::new(move |state| {
                if state.price <= max_price {
                    1.0
                } else {
                    // Fuzzy logic: partial satisfaction
                    (max_price / state.price).min(1.0)
                }
            }),
        )
    }

    /// Position size is within limits
    pub fn position_within_limits(max_position: f32) -> TradingPredicate {
        TradingPredicate::new(
            "position_within_limits",
            Box::new(move |state| {
                let ratio = state.position_size.abs() / max_position;
                if ratio <= 1.0 {
                    1.0 - ratio // More satisfied when position is smaller
                } else {
                    0.0 // Violated
                }
            }),
        )
    }

    /// Market hours are active
    pub fn market_hours_active() -> TradingPredicate {
        TradingPredicate::new(
            "market_hours",
            Box::new(|state| {
                // NYSE: 9:30 AM - 4:00 PM ET (approx 9-16 in 24h)
                if (9..16).contains(&state.hour) {
                    1.0
                } else {
                    0.0
                }
            }),
        )
    }

    /// Sufficient liquidity
    pub fn sufficient_liquidity(min_volume: f32) -> TradingPredicate {
        TradingPredicate::new(
            "sufficient_liquidity",
            Box::new(move |state| {
                if state.volume >= min_volume {
                    1.0
                } else {
                    state.volume / min_volume
                }
            }),
        )
    }
}

#[cfg(test)]
mod tests {
    use super::common_predicates::*;
    use super::*;

    #[test]
    fn test_price_below_max() {
        let predicate = price_below_max(100.0);

        let state = MarketState {
            price: 95.0,
            volume: 1000.0,
            position_size: 10.0,
            portfolio_value: 10000.0,
            hour: 10,
        };

        let satisfaction = predicate.evaluate(&state);
        assert_eq!(satisfaction, 1.0);
    }

    #[test]
    fn test_market_hours() {
        let predicate = market_hours_active();

        let mut state = MarketState {
            price: 100.0,
            volume: 1000.0,
            position_size: 10.0,
            portfolio_value: 10000.0,
            hour: 10,
        };

        assert_eq!(predicate.evaluate(&state), 1.0);

        state.hour = 20;
        assert_eq!(predicate.evaluate(&state), 0.0);
    }
}
