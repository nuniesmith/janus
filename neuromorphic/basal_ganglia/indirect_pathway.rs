//! Indirect Pathway (NoGo Signal)
//!
//! Inhibits action execution - the "NoGo" pathway for risk management.

pub struct IndirectPathway {
    threshold: f32,
}

impl IndirectPathway {
    pub fn new(threshold: f32) -> Self {
        Self { threshold }
    }

    /// Compute NoGo signal strength (higher = more inhibition)
    pub fn compute_nogo_signal(&self, risk_level: f32) -> f32 {
        if risk_level > self.threshold {
            risk_level
        } else {
            0.0
        }
    }

    /// Check if action should be inhibited
    pub fn should_inhibit(&self, risk_level: f32) -> bool {
        risk_level > self.threshold
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_nogo_signal() {
        let pathway = IndirectPathway::new(0.7);

        assert!(pathway.should_inhibit(0.9));
        assert!(!pathway.should_inhibit(0.5));
    }
}
