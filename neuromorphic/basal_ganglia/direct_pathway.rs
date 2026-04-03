//! Direct Pathway (Go Signal)
//!
//! Promotes action execution - the "Go" pathway.

pub struct DirectPathway {
    threshold: f32,
}

impl DirectPathway {
    pub fn new(threshold: f32) -> Self {
        Self { threshold }
    }

    /// Compute Go signal strength
    pub fn compute_go_signal(&self, action_value: f32) -> f32 {
        if action_value > self.threshold {
            action_value
        } else {
            0.0
        }
    }

    /// Check if action should be executed
    pub fn should_go(&self, action_value: f32) -> bool {
        action_value > self.threshold
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_go_signal() {
        let pathway = DirectPathway::new(0.5);

        assert!(pathway.should_go(0.8));
        assert!(!pathway.should_go(0.3));
    }
}
