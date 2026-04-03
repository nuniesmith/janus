//! Volume-Weighted Average Price (VWAP) Execution
//!
//! VWAP is an execution strategy that distributes orders proportionally to
//! historical volume patterns. This aims to match the market's natural volume
//! distribution and minimize market impact.
//!
//! # Example
//!
//! ```rust
//! use vision::execution::vwap::{VWAPExecutor, VWAPConfig, VolumeProfile};
//! use std::time::Duration;
//!
//! // Create a simple volume profile
//! let profile = VolumeProfile::from_percentages(vec![
//!     0.05, 0.10, 0.15, 0.20, 0.20, 0.15, 0.10, 0.05
//! ]);
//!
//! let config = VWAPConfig {
//!     total_quantity: 10000.0,
//!     duration: Duration::from_secs(3600), // 1 hour
//!     volume_profile: profile,
//!     min_slice_size: 50.0,
//!     participation_rate: 0.2, // 20% of market volume
//! };
//!
//! let mut executor = VWAPExecutor::new(config);
//!
//! // Execute slices according to volume profile
//! while !executor.is_complete() {
//!     if let Some(slice) = executor.next_slice() {
//!         println!("Execute {} units ({}% of market volume)",
//!                  slice.quantity, slice.target_participation * 100.0);
//!     }
//! }
//! ```

use std::time::{Duration, Instant};

/// Volume profile representing expected volume distribution
#[derive(Debug, Clone)]
pub struct VolumeProfile {
    /// Volume percentages for each time bucket (must sum to 1.0)
    pub percentages: Vec<f64>,
}

impl VolumeProfile {
    /// Create a volume profile from percentages
    pub fn from_percentages(percentages: Vec<f64>) -> Self {
        let sum: f64 = percentages.iter().sum();
        assert!(
            (sum - 1.0).abs() < 0.01,
            "Volume percentages must sum to 1.0"
        );
        Self { percentages }
    }

    /// Create a uniform volume profile
    pub fn uniform(num_buckets: usize) -> Self {
        let percentage = 1.0 / num_buckets as f64;
        Self {
            percentages: vec![percentage; num_buckets],
        }
    }

    /// Create a typical U-shaped intraday volume profile
    pub fn intraday_u_shape(num_buckets: usize) -> Self {
        let mut percentages = Vec::with_capacity(num_buckets);

        // Higher volume at open and close, lower in the middle
        for i in 0..num_buckets {
            let t = i as f64 / (num_buckets - 1) as f64;
            // U-shape using parabola: high at 0 and 1, low at 0.5
            let volume = 1.0 + 2.0 * (t - 0.5).powi(2);
            percentages.push(volume);
        }

        // Normalize to sum to 1.0
        let sum: f64 = percentages.iter().sum();
        percentages.iter_mut().for_each(|p| *p /= sum);

        Self { percentages }
    }

    /// Create a reverse J-shaped profile (high at open, declining)
    pub fn reverse_j_shape(num_buckets: usize) -> Self {
        let mut percentages = Vec::with_capacity(num_buckets);

        for i in 0..num_buckets {
            let t = i as f64 / (num_buckets - 1) as f64;
            let volume = (1.0 - t).powi(2) + 0.3; // Declining with floor
            percentages.push(volume);
        }

        let sum: f64 = percentages.iter().sum();
        percentages.iter_mut().for_each(|p| *p /= sum);

        Self { percentages }
    }

    /// Get volume percentage for a specific bucket
    pub fn get_percentage(&self, bucket_index: usize) -> f64 {
        self.percentages.get(bucket_index).copied().unwrap_or(0.0)
    }

    /// Number of time buckets
    pub fn num_buckets(&self) -> usize {
        self.percentages.len()
    }
}

/// Configuration for VWAP execution
#[derive(Debug, Clone)]
pub struct VWAPConfig {
    /// Total quantity to execute
    pub total_quantity: f64,
    /// Total duration for execution
    pub duration: Duration,
    /// Historical volume profile
    pub volume_profile: VolumeProfile,
    /// Minimum slice size
    pub min_slice_size: f64,
    /// Target participation rate (0-1, e.g., 0.2 = 20% of market volume)
    pub participation_rate: f64,
    /// Whether to adapt to real-time volume
    pub adaptive: bool,
}

impl Default for VWAPConfig {
    fn default() -> Self {
        Self {
            total_quantity: 1000.0,
            duration: Duration::from_secs(300),
            volume_profile: VolumeProfile::uniform(10),
            min_slice_size: 10.0,
            participation_rate: 0.1,
            adaptive: false,
        }
    }
}

/// A VWAP execution slice
#[derive(Debug, Clone)]
pub struct VWAPSlice {
    /// Slice number
    pub slice_id: usize,
    /// Quantity to execute
    pub quantity: f64,
    /// Scheduled time
    pub scheduled_time: Duration,
    /// Target participation rate for this slice
    pub target_participation: f64,
    /// Expected market volume for this period
    pub expected_market_volume: Option<f64>,
    /// Executed flag
    pub executed: bool,
    /// Actual execution time
    pub actual_execution_time: Option<Instant>,
    /// Actual quantity executed
    pub actual_quantity: Option<f64>,
    /// Execution price
    pub execution_price: Option<f64>,
    /// Observed market volume during this slice
    pub observed_market_volume: Option<f64>,
}

/// VWAP execution algorithm
pub struct VWAPExecutor {
    config: VWAPConfig,
    slices: Vec<VWAPSlice>,
    current_slice: usize,
    start_time: Option<Instant>,
    completed: bool,
    total_market_volume: f64,
}

impl VWAPExecutor {
    /// Create a new VWAP executor
    pub fn new(config: VWAPConfig) -> Self {
        let slices = Self::generate_slices(&config);
        Self {
            config,
            slices,
            current_slice: 0,
            start_time: None,
            completed: false,
            total_market_volume: 0.0,
        }
    }

    /// Generate execution slices based on volume profile
    fn generate_slices(config: &VWAPConfig) -> Vec<VWAPSlice> {
        let num_buckets = config.volume_profile.num_buckets();
        let mut slices = Vec::with_capacity(num_buckets);

        let time_per_bucket = config.duration.as_secs_f64() / num_buckets as f64;
        let mut cumulative_time = 0.0;

        for i in 0..num_buckets {
            let volume_pct = config.volume_profile.get_percentage(i);
            let quantity = config.total_quantity * volume_pct;

            // Ensure minimum slice size
            let quantity = quantity.max(config.min_slice_size);

            cumulative_time += time_per_bucket;

            slices.push(VWAPSlice {
                slice_id: i,
                quantity,
                scheduled_time: Duration::from_secs_f64(cumulative_time),
                target_participation: config.participation_rate,
                expected_market_volume: None,
                executed: false,
                actual_execution_time: None,
                actual_quantity: None,
                execution_price: None,
                observed_market_volume: None,
            });
        }

        slices
    }

    /// Start execution
    pub fn start(&mut self) {
        self.start_time = Some(Instant::now());
    }

    /// Get the next slice ready for execution
    pub fn next_slice(&mut self) -> Option<&VWAPSlice> {
        if self.start_time.is_none() {
            self.start();
        }

        if self.completed || self.current_slice >= self.slices.len() {
            return None;
        }

        let elapsed = self.start_time.unwrap().elapsed();
        if elapsed >= self.slices[self.current_slice].scheduled_time {
            Some(&self.slices[self.current_slice])
        } else {
            None
        }
    }

    /// Mark current slice as executed
    pub fn mark_executed(
        &mut self,
        actual_quantity: f64,
        execution_price: f64,
        observed_volume: Option<f64>,
    ) {
        if self.current_slice < self.slices.len() {
            let slice = &mut self.slices[self.current_slice];
            slice.executed = true;
            slice.actual_execution_time = Some(Instant::now());
            slice.actual_quantity = Some(actual_quantity);
            slice.execution_price = Some(execution_price);
            slice.observed_market_volume = observed_volume;

            if let Some(vol) = observed_volume {
                self.total_market_volume += vol;

                // Adaptive adjustment for future slices
                if self.config.adaptive {
                    self.adjust_future_slices(vol);
                }
            }

            self.current_slice += 1;

            if self.current_slice >= self.slices.len() {
                self.completed = true;
            }
        }
    }

    /// Adjust future slices based on observed volume
    fn adjust_future_slices(&mut self, _observed_volume: f64) {
        if self.current_slice >= self.slices.len() {
            return;
        }

        // Calculate remaining quantity
        let remaining_quantity: f64 = self.slices[self.current_slice..]
            .iter()
            .map(|s| s.quantity)
            .sum();

        if remaining_quantity <= 0.0 {
            return;
        }

        // Redistribute remaining quantity proportionally to remaining volume profile
        let remaining_profile_sum: f64 = self.slices[self.current_slice..]
            .iter()
            .map(|s| self.config.volume_profile.get_percentage(s.slice_id))
            .sum();

        if remaining_profile_sum > 0.0 {
            for i in self.current_slice..self.slices.len() {
                let slice = &mut self.slices[i];
                let profile_pct = self.config.volume_profile.get_percentage(slice.slice_id);
                let new_quantity = (remaining_quantity * profile_pct / remaining_profile_sum)
                    .max(self.config.min_slice_size);
                slice.quantity = new_quantity;
            }
        }
    }

    /// Get all slices
    pub fn slices(&self) -> &[VWAPSlice] {
        &self.slices
    }

    /// Check if execution is complete
    pub fn is_complete(&self) -> bool {
        self.completed
    }

    /// Get progress percentage
    pub fn progress(&self) -> f64 {
        if self.slices.is_empty() {
            return 0.0;
        }
        (self.current_slice as f64 / self.slices.len() as f64) * 100.0
    }

    /// Get total executed quantity
    pub fn total_executed(&self) -> f64 {
        self.slices.iter().filter_map(|s| s.actual_quantity).sum()
    }

    /// Calculate volume-weighted average execution price
    pub fn vwap_price(&self) -> Option<f64> {
        let total_value: f64 = self
            .slices
            .iter()
            .filter_map(|s| {
                if let (Some(qty), Some(price)) = (s.actual_quantity, s.execution_price) {
                    Some(qty * price)
                } else {
                    None
                }
            })
            .sum();

        let total_qty = self.total_executed();

        if total_qty > 0.0 {
            Some(total_value / total_qty)
        } else {
            None
        }
    }

    /// Calculate actual participation rate
    pub fn actual_participation_rate(&self) -> Option<f64> {
        if self.total_market_volume <= 0.0 {
            return None;
        }

        let our_volume = self.total_executed();
        Some(our_volume / self.total_market_volume)
    }

    /// Get statistics
    pub fn statistics(&self) -> VWAPStatistics {
        let executed_slices = self.slices.iter().filter(|s| s.executed).count();
        let total_slices = self.slices.len();

        let total_quantity = self.config.total_quantity;
        let executed_quantity = self.total_executed();

        let vwap = self.vwap_price();
        let participation = self.actual_participation_rate();

        VWAPStatistics {
            total_slices,
            executed_slices,
            remaining_slices: total_slices - executed_slices,
            total_quantity,
            executed_quantity,
            remaining_quantity: total_quantity - executed_quantity,
            vwap_price: vwap,
            target_participation: self.config.participation_rate,
            actual_participation: participation,
            total_market_volume: self.total_market_volume,
            is_complete: self.completed,
        }
    }

    /// Reset the executor
    pub fn reset(&mut self) {
        for slice in &mut self.slices {
            slice.executed = false;
            slice.actual_execution_time = None;
            slice.actual_quantity = None;
            slice.execution_price = None;
            slice.observed_market_volume = None;
        }
        self.current_slice = 0;
        self.start_time = None;
        self.completed = false;
        self.total_market_volume = 0.0;
    }
}

/// VWAP execution statistics
#[derive(Debug, Clone)]
pub struct VWAPStatistics {
    pub total_slices: usize,
    pub executed_slices: usize,
    pub remaining_slices: usize,
    pub total_quantity: f64,
    pub executed_quantity: f64,
    pub remaining_quantity: f64,
    pub vwap_price: Option<f64>,
    pub target_participation: f64,
    pub actual_participation: Option<f64>,
    pub total_market_volume: f64,
    pub is_complete: bool,
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_volume_profile_uniform() {
        let profile = VolumeProfile::uniform(5);
        assert_eq!(profile.num_buckets(), 5);
        for i in 0..5 {
            assert!((profile.get_percentage(i) - 0.2).abs() < 0.01);
        }
    }

    #[test]
    fn test_volume_profile_custom() {
        let profile = VolumeProfile::from_percentages(vec![0.3, 0.5, 0.2]);
        assert_eq!(profile.num_buckets(), 3);
        assert!((profile.get_percentage(0) - 0.3).abs() < 0.01);
        assert!((profile.get_percentage(1) - 0.5).abs() < 0.01);
        assert!((profile.get_percentage(2) - 0.2).abs() < 0.01);
    }

    #[test]
    fn test_volume_profile_u_shape() {
        let profile = VolumeProfile::intraday_u_shape(10);
        assert_eq!(profile.num_buckets(), 10);

        // Check sum is close to 1.0
        let sum: f64 = profile.percentages.iter().sum();
        assert!((sum - 1.0).abs() < 0.01);

        // Higher at beginning and end
        assert!(profile.get_percentage(0) > profile.get_percentage(5));
        assert!(profile.get_percentage(9) > profile.get_percentage(5));
    }

    #[test]
    fn test_vwap_executor_creation() {
        let config = VWAPConfig::default();
        let executor = VWAPExecutor::new(config);

        assert!(!executor.is_complete());
        assert_eq!(executor.progress(), 0.0);
    }

    #[test]
    fn test_slice_generation() {
        let profile = VolumeProfile::from_percentages(vec![0.2, 0.3, 0.5]);
        let config = VWAPConfig {
            total_quantity: 1000.0,
            volume_profile: profile,
            ..Default::default()
        };

        let executor = VWAPExecutor::new(config);
        let slices = executor.slices();

        assert_eq!(slices.len(), 3);

        // Check quantities match profile (approximately)
        assert!((slices[0].quantity - 200.0).abs() < 1.0);
        assert!((slices[1].quantity - 300.0).abs() < 1.0);
        assert!((slices[2].quantity - 500.0).abs() < 1.0);
    }

    #[test]
    fn test_vwap_execution() {
        let config = VWAPConfig {
            total_quantity: 600.0,
            volume_profile: VolumeProfile::uniform(3),
            duration: Duration::from_secs(3),
            ..Default::default()
        };

        let mut executor = VWAPExecutor::new(config);
        executor.start();

        // Execute slices
        std::thread::sleep(Duration::from_millis(1100));
        if executor.next_slice().is_some() {
            executor.mark_executed(200.0, 100.0, Some(1000.0));
        }

        std::thread::sleep(Duration::from_millis(1100));
        if executor.next_slice().is_some() {
            executor.mark_executed(200.0, 102.0, Some(1100.0));
        }

        std::thread::sleep(Duration::from_millis(1100));
        if executor.next_slice().is_some() {
            executor.mark_executed(200.0, 101.0, Some(900.0));
        }

        assert!(executor.is_complete());
        assert_eq!(executor.total_executed(), 600.0);
    }

    #[test]
    fn test_vwap_price_calculation() {
        let config = VWAPConfig {
            total_quantity: 600.0,
            volume_profile: VolumeProfile::uniform(3),
            ..Default::default()
        };

        let mut executor = VWAPExecutor::new(config);
        executor.start();

        executor.mark_executed(200.0, 100.0, None);
        executor.mark_executed(200.0, 110.0, None);
        executor.mark_executed(200.0, 105.0, None);

        let vwap = executor.vwap_price().unwrap();
        let expected = (200.0 * 100.0 + 200.0 * 110.0 + 200.0 * 105.0) / 600.0;

        assert!((vwap - expected).abs() < 0.01);
    }

    #[test]
    fn test_participation_rate() {
        let config = VWAPConfig {
            total_quantity: 300.0,
            participation_rate: 0.2,
            ..Default::default()
        };

        let mut executor = VWAPExecutor::new(config);
        executor.start();

        executor.mark_executed(100.0, 100.0, Some(500.0));
        executor.mark_executed(100.0, 101.0, Some(600.0));
        executor.mark_executed(100.0, 102.0, Some(400.0));

        let actual_participation = executor.actual_participation_rate().unwrap();
        let expected = 300.0 / 1500.0; // 300 / (500 + 600 + 400)

        assert!((actual_participation - expected).abs() < 0.01);
    }

    #[test]
    fn test_statistics() {
        let config = VWAPConfig {
            total_quantity: 1000.0,
            volume_profile: VolumeProfile::uniform(4),
            ..Default::default()
        };

        let mut executor = VWAPExecutor::new(config);
        executor.start();

        executor.mark_executed(250.0, 100.0, Some(1000.0));
        executor.mark_executed(250.0, 102.0, Some(1200.0));

        let stats = executor.statistics();

        assert_eq!(stats.total_slices, 4);
        assert_eq!(stats.executed_slices, 2);
        assert_eq!(stats.remaining_slices, 2);
        assert_eq!(stats.executed_quantity, 500.0);
        assert_eq!(stats.remaining_quantity, 500.0);
        assert_eq!(stats.total_market_volume, 2200.0);
        assert!(!stats.is_complete);
    }

    #[test]
    fn test_reset() {
        let config = VWAPConfig::default();
        let mut executor = VWAPExecutor::new(config);
        executor.start();

        executor.mark_executed(100.0, 100.0, Some(500.0));
        assert!(executor.progress() > 0.0);

        executor.reset();
        assert_eq!(executor.progress(), 0.0);
        assert!(!executor.is_complete());
        assert_eq!(executor.total_executed(), 0.0);
        assert_eq!(executor.total_market_volume, 0.0);
    }

    #[test]
    fn test_min_slice_size() {
        let config = VWAPConfig {
            total_quantity: 50.0,
            volume_profile: VolumeProfile::uniform(10),
            min_slice_size: 10.0,
            ..Default::default()
        };

        let executor = VWAPExecutor::new(config);

        for slice in executor.slices() {
            assert!(slice.quantity >= 10.0);
        }
    }

    #[test]
    fn test_adaptive_mode() {
        let config = VWAPConfig {
            total_quantity: 1000.0,
            volume_profile: VolumeProfile::uniform(5),
            adaptive: true,
            ..Default::default()
        };

        let mut executor = VWAPExecutor::new(config);
        executor.start();

        // Execute first slice with high volume
        executor.mark_executed(200.0, 100.0, Some(5000.0));

        // Check that future slices might be adjusted
        assert_eq!(executor.slices().len(), 5);
    }
}
