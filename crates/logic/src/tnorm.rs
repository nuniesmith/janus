//! Fuzzy Logic operations using Lukasiewicz T-Norms.
//!
//! These operations allow the Forward service to compute a "Compliance Score"
//! that gates neural network outputs against logical rules.

use common::{JanusError, Order, Result};
use ndarray::Array1;

/// Fuzzy logic operations for Logic Tensor Networks
pub struct FuzzyLogic;

impl FuzzyLogic {
    /// Conjunction (AND) using Lukasiewicz T-Norm
    /// A ∧ B = max(0, A + B - 1)
    pub fn and(a: f64, b: f64) -> f64 {
        (a + b - 1.0).max(0.0)
    }

    /// Disjunction (OR) using Lukasiewicz T-Conorm
    /// A ∨ B = min(1, A + B)
    pub fn or(a: f64, b: f64) -> f64 {
        (a + b).min(1.0)
    }

    /// Implication (IF-THEN) using Lukasiewicz
    /// A → B = min(1, 1 - A + B)
    pub fn implies(a: f64, b: f64) -> f64 {
        (1.0 - a + b).min(1.0)
    }

    /// Negation
    /// ¬A = 1 - A
    pub fn not(a: f64) -> f64 {
        1.0 - a
    }

    /// Vectorized AND operation
    pub fn and_vec(a: &Array1<f64>, b: &Array1<f64>) -> Result<Array1<f64>> {
        if a.len() != b.len() {
            return Err(JanusError::Internal(
                "Vector length mismatch in fuzzy AND".to_string(),
            ));
        }
        Ok(Array1::from_iter(
            a.iter().zip(b.iter()).map(|(&ai, &bi)| Self::and(ai, bi)),
        ))
    }

    /// Vectorized OR operation
    pub fn or_vec(a: &Array1<f64>, b: &Array1<f64>) -> Result<Array1<f64>> {
        if a.len() != b.len() {
            return Err(JanusError::Internal(
                "Vector length mismatch in fuzzy OR".to_string(),
            ));
        }
        Ok(Array1::from_iter(
            a.iter().zip(b.iter()).map(|(&ai, &bi)| Self::or(ai, bi)),
        ))
    }

    /// Vectorized IMPLIES operation
    pub fn implies_vec(a: &Array1<f64>, b: &Array1<f64>) -> Result<Array1<f64>> {
        if a.len() != b.len() {
            return Err(JanusError::Internal(
                "Vector length mismatch in fuzzy IMPLIES".to_string(),
            ));
        }
        Ok(Array1::from_iter(
            a.iter()
                .zip(b.iter())
                .map(|(&ai, &bi)| Self::implies(ai, bi)),
        ))
    }
}

/// Logic Tensor Network - evaluates logical constraints
pub struct LogicTensorNetwork {
    rules: Vec<LogicalRule>,
}

/// A logical rule that can be evaluated
pub struct LogicalRule {
    pub name: String,
    pub condition: Box<dyn Fn(&Order) -> f64 + Send + Sync>,
}

impl LogicTensorNetwork {
    /// Create a new LTN with default rules
    pub fn new() -> Self {
        let mut ltn = Self { rules: Vec::new() };
        ltn.add_default_rules();
        ltn
    }

    /// Add a custom rule
    pub fn add_rule(&mut self, name: String, condition: Box<dyn Fn(&Order) -> f64 + Send + Sync>) {
        self.rules.push(LogicalRule { name, condition });
    }

    /// Evaluate all rules and return compliance score
    /// Returns: 0.0 (reject) to 1.0 (fully compliant)
    pub fn evaluate(&self, order: &Order) -> f64 {
        if self.rules.is_empty() {
            return 1.0; // No rules = always compliant
        }

        // Combine all rules using AND (all must be satisfied)
        self.rules
            .iter()
            .map(|rule| (rule.condition)(order))
            .fold(1.0, |acc, score| FuzzyLogic::and(acc, score))
    }

    /// Add default risk and regulatory rules
    fn add_default_rules(&mut self) {
        // Rule: Position size must be reasonable (placeholder)
        self.add_rule(
            "position_size".to_string(),
            Box::new(|order| {
                // Normalize quantity to [0, 1] compliance score
                // This is a placeholder - should check against actual risk limits
                let max_quantity = 10000.0; // Example limit
                let compliance = 1.0 - (order.quantity.value() / max_quantity).min(1.0);
                compliance.max(0.0)
            }),
        );

        // Rule: Price must be within reasonable bounds (for limit orders)
        self.add_rule(
            "price_bounds".to_string(),
            Box::new(|order| {
                if let Some(price) = order.price {
                    // Placeholder: check if price is positive and reasonable
                    if price.value() > 0.0 && price.value() < 1e10 {
                        1.0
                    } else {
                        0.0
                    }
                } else {
                    1.0 // Market orders don't have price constraints
                }
            }),
        );
    }
}

impl Default for LogicTensorNetwork {
    fn default() -> Self {
        Self::new()
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use common::{Order, OrderSide, OrderType, Volume};

    #[test]
    fn test_fuzzy_and() {
        assert_eq!(FuzzyLogic::and(0.8, 0.7), 0.5);
        assert_eq!(FuzzyLogic::and(0.3, 0.4), 0.0);
        assert_eq!(FuzzyLogic::and(1.0, 1.0), 1.0);
    }

    #[test]
    fn test_fuzzy_implies() {
        assert_eq!(FuzzyLogic::implies(1.0, 1.0), 1.0);
        assert_eq!(FuzzyLogic::implies(0.0, 1.0), 1.0);
        assert_eq!(FuzzyLogic::implies(1.0, 0.0), 0.0);
    }

    #[test]
    fn test_ltn_evaluation() {
        let ltn = LogicTensorNetwork::new();
        let order = Order::new(
            "BTC/USD".to_string(),
            OrderSide::Buy,
            OrderType::Market,
            Volume(100.0),
            None,
        );
        let score = ltn.evaluate(&order);
        assert!(score >= 0.0 && score <= 1.0);
    }
}
