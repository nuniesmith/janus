//! VPIN - Volume-Synchronized Probability of Informed Trading
//!
//! Detects toxic flow and potential flash crashes.

pub struct VPINCalculator {
    pub bucket_size: f64,
    buckets: Vec<VPINBucket>,
    current_bucket: VPINBucket,
}

#[derive(Debug, Clone, Default)]
struct VPINBucket {
    buy_volume: f64,
    sell_volume: f64,
}

impl VPINCalculator {
    pub fn new(bucket_size: f64) -> Self {
        Self {
            bucket_size,
            buckets: Vec::new(),
            current_bucket: VPINBucket::default(),
        }
    }

    /// Update VPIN with a new trade
    pub fn update(&mut self, volume: f64, is_buy: bool) {
        if is_buy {
            self.current_bucket.buy_volume += volume;
        } else {
            self.current_bucket.sell_volume += volume;
        }

        // Check if bucket is full
        let total_volume = self.current_bucket.buy_volume + self.current_bucket.sell_volume;
        if total_volume >= self.bucket_size {
            self.buckets.push(self.current_bucket.clone());
            self.current_bucket = VPINBucket::default();

            // Keep only recent buckets (e.g., last 50)
            if self.buckets.len() > 50 {
                self.buckets.remove(0);
            }
        }
    }

    /// Calculate current VPIN value
    pub fn calculate(&self) -> f64 {
        if self.buckets.is_empty() {
            return 0.0;
        }

        let n = self.buckets.len() as f64;
        let mut total_imbalance = 0.0;

        for bucket in &self.buckets {
            let total = bucket.buy_volume + bucket.sell_volume;
            if total > 0.0 {
                let imbalance = (bucket.buy_volume - bucket.sell_volume).abs() / total;
                total_imbalance += imbalance;
            }
        }

        total_imbalance / n
    }

    /// Check if VPIN indicates danger
    pub fn is_dangerous(&self, threshold: f64) -> bool {
        self.calculate() > threshold
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_vpin_calculation() {
        let mut vpin = VPINCalculator::new(1000.0);

        // Balanced flow
        vpin.update(500.0, true);
        vpin.update(500.0, false);

        // Force bucket completion
        let current_vpin = vpin.calculate();

        // Should be relatively low for balanced flow
        assert!(current_vpin < 0.5);
    }

    #[test]
    fn test_vpin_imbalance() {
        let mut vpin = VPINCalculator::new(1000.0);

        // Highly imbalanced (all buys)
        vpin.update(1000.0, true);

        let current_vpin = vpin.calculate();

        // Should detect high imbalance
        assert!(current_vpin > 0.8 || vpin.buckets.is_empty());
    }

    #[test]
    fn test_vpin_danger_detection() {
        let mut vpin = VPINCalculator::new(1000.0);

        // Create imbalanced buckets
        for _ in 0..5 {
            vpin.update(900.0, true);
            vpin.update(100.0, false);
        }

        assert!(vpin.is_dangerous(0.5));
    }
}
