//! Time-Weighted Average Price (TWAP) Execution
//!
//! TWAP is an algorithmic execution strategy that splits large orders into smaller
//! slices executed evenly across a specified time period. This minimizes market impact
//! by avoiding large immediate orders.
//!
//! # Example
//!
//! ```rust
//! use vision::execution::twap::{TWAPExecutor, TWAPConfig};
//! use std::time::Duration;
//!
//! let config = TWAPConfig {
//!     total_quantity: 10000.0,
//!     duration: Duration::from_secs(300), // 5 minutes
//!     num_slices: 10,
//!     min_slice_size: 100.0,
//!     randomize_timing: false,
//!     randomize_size: false,
//! };
//!
//! let mut executor = TWAPExecutor::new(config);
//!
//! // Get next slice to execute
//! if let Some(slice) = executor.next_slice() {
//!     println!("Execute {} units at {}", slice.quantity, slice.scheduled_time);
//! }
//! ```

use std::time::{Duration, Instant};

/// Configuration for TWAP execution
#[derive(Debug, Clone)]
pub struct TWAPConfig {
    /// Total quantity to execute
    pub total_quantity: f64,
    /// Total duration over which to execute
    pub duration: Duration,
    /// Number of slices to split the order into
    pub num_slices: usize,
    /// Minimum size for each slice
    pub min_slice_size: f64,
    /// Add randomness to timing to avoid predictability
    pub randomize_timing: bool,
    /// Add randomness to slice sizes
    pub randomize_size: bool,
    /// Maximum timing randomness as fraction of interval (0-1)
    pub timing_randomness: f64,
    /// Maximum size randomness as fraction of base size (0-1)
    pub size_randomness: f64,
}

impl Default for TWAPConfig {
    fn default() -> Self {
        Self {
            total_quantity: 1000.0,
            duration: Duration::from_secs(60),
            num_slices: 6,
            min_slice_size: 10.0,
            randomize_timing: false,
            randomize_size: false,
            timing_randomness: 0.2,
            size_randomness: 0.1,
        }
    }
}

/// A single execution slice
#[derive(Debug, Clone)]
pub struct ExecutionSlice {
    /// Slice number (0-indexed)
    pub slice_id: usize,
    /// Quantity to execute in this slice
    pub quantity: f64,
    /// Scheduled execution time (relative to start)
    pub scheduled_time: Duration,
    /// Whether this slice has been executed
    pub executed: bool,
    /// Actual execution time (if executed)
    pub actual_execution_time: Option<Instant>,
    /// Actual executed quantity
    pub actual_quantity: Option<f64>,
    /// Execution price
    pub execution_price: Option<f64>,
}

/// TWAP execution algorithm
pub struct TWAPExecutor {
    config: TWAPConfig,
    slices: Vec<ExecutionSlice>,
    current_slice: usize,
    start_time: Option<Instant>,
    completed: bool,
}

impl TWAPExecutor {
    /// Create a new TWAP executor
    pub fn new(config: TWAPConfig) -> Self {
        let slices = Self::generate_slices(&config);
        Self {
            config,
            slices,
            current_slice: 0,
            start_time: None,
            completed: false,
        }
    }

    /// Generate execution slices based on configuration
    fn generate_slices(config: &TWAPConfig) -> Vec<ExecutionSlice> {
        let mut slices = Vec::with_capacity(config.num_slices);

        // Calculate base slice size
        let base_slice_size = config.total_quantity / config.num_slices as f64;
        let base_interval = config.duration.as_secs_f64() / config.num_slices as f64;

        let mut remaining_quantity = config.total_quantity;
        let mut cumulative_time = 0.0;

        for i in 0..config.num_slices {
            // Calculate quantity for this slice
            let mut quantity = if i == config.num_slices - 1 {
                // Last slice gets remaining quantity
                remaining_quantity
            } else {
                base_slice_size
            };

            // Apply size randomization
            if config.randomize_size && i < config.num_slices - 1 {
                let randomness = Self::generate_random_factor(config.size_randomness);
                quantity *= randomness;
                quantity = quantity.max(config.min_slice_size);
            }

            // Ensure minimum slice size
            if quantity < config.min_slice_size {
                quantity = config.min_slice_size;
            }

            // Apply timing randomization
            let mut interval = base_interval;
            if config.randomize_timing && i > 0 {
                let randomness = Self::generate_random_factor(config.timing_randomness);
                interval *= randomness;
            }

            cumulative_time += interval;

            slices.push(ExecutionSlice {
                slice_id: i,
                quantity,
                scheduled_time: Duration::from_secs_f64(cumulative_time),
                executed: false,
                actual_execution_time: None,
                actual_quantity: None,
                execution_price: None,
            });

            remaining_quantity -= quantity;
        }

        slices
    }

    /// Generate a random factor around 1.0
    fn generate_random_factor(max_deviation: f64) -> f64 {
        // Simple deterministic "random" for reproducibility
        // In production, use a proper RNG
        let pseudo_random = (std::f64::consts::PI * max_deviation).sin().abs();
        1.0 + (pseudo_random - 0.5) * max_deviation * 2.0
    }

    /// Start the execution
    pub fn start(&mut self) {
        self.start_time = Some(Instant::now());
    }

    /// Get the next slice ready for execution
    pub fn next_slice(&mut self) -> Option<&ExecutionSlice> {
        if self.start_time.is_none() {
            self.start();
        }

        if self.completed || self.current_slice >= self.slices.len() {
            return None;
        }

        // Check if it's time for the current slice
        let elapsed = self.start_time.unwrap().elapsed();
        if elapsed >= self.slices[self.current_slice].scheduled_time {
            Some(&self.slices[self.current_slice])
        } else {
            None
        }
    }

    /// Mark the current slice as executed
    pub fn mark_executed(&mut self, actual_quantity: f64, execution_price: f64) {
        if self.current_slice < self.slices.len() {
            let slice = &mut self.slices[self.current_slice];
            slice.executed = true;
            slice.actual_execution_time = Some(Instant::now());
            slice.actual_quantity = Some(actual_quantity);
            slice.execution_price = Some(execution_price);

            self.current_slice += 1;

            if self.current_slice >= self.slices.len() {
                self.completed = true;
            }
        }
    }

    /// Get all slices
    pub fn slices(&self) -> &[ExecutionSlice] {
        &self.slices
    }

    /// Get total executed quantity
    pub fn total_executed(&self) -> f64 {
        self.slices.iter().filter_map(|s| s.actual_quantity).sum()
    }

    /// Get average execution price
    pub fn average_price(&self) -> Option<f64> {
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

    /// Check if execution is complete
    pub fn is_complete(&self) -> bool {
        self.completed
    }

    /// Get progress as percentage (0-100)
    pub fn progress(&self) -> f64 {
        if self.slices.is_empty() {
            return 0.0;
        }
        (self.current_slice as f64 / self.slices.len() as f64) * 100.0
    }

    /// Get time until next slice
    pub fn time_until_next(&self) -> Option<Duration> {
        if self.completed || self.current_slice >= self.slices.len() {
            return None;
        }

        let start = self.start_time?;
        let elapsed = start.elapsed();
        let scheduled = self.slices[self.current_slice].scheduled_time;

        if scheduled > elapsed {
            Some(scheduled - elapsed)
        } else {
            Some(Duration::from_secs(0))
        }
    }

    /// Get execution statistics
    pub fn statistics(&self) -> TWAPStatistics {
        let executed_slices = self.slices.iter().filter(|s| s.executed).count();
        let total_slices = self.slices.len();

        let total_quantity = self.config.total_quantity;
        let executed_quantity = self.total_executed();

        let avg_price = self.average_price();

        let completion_rate = if total_slices > 0 {
            executed_slices as f64 / total_slices as f64
        } else {
            0.0
        };

        let fill_rate = if total_quantity > 0.0 {
            executed_quantity / total_quantity
        } else {
            0.0
        };

        TWAPStatistics {
            total_slices,
            executed_slices,
            remaining_slices: total_slices - executed_slices,
            total_quantity,
            executed_quantity,
            remaining_quantity: total_quantity - executed_quantity,
            average_execution_price: avg_price,
            completion_rate,
            fill_rate,
            is_complete: self.completed,
        }
    }

    /// Reset the executor to start over
    pub fn reset(&mut self) {
        for slice in &mut self.slices {
            slice.executed = false;
            slice.actual_execution_time = None;
            slice.actual_quantity = None;
            slice.execution_price = None;
        }
        self.current_slice = 0;
        self.start_time = None;
        self.completed = false;
    }
}

/// Statistics for TWAP execution
#[derive(Debug, Clone)]
pub struct TWAPStatistics {
    /// Total number of slices
    pub total_slices: usize,
    /// Number of executed slices
    pub executed_slices: usize,
    /// Number of remaining slices
    pub remaining_slices: usize,
    /// Total quantity to execute
    pub total_quantity: f64,
    /// Quantity executed so far
    pub executed_quantity: f64,
    /// Remaining quantity
    pub remaining_quantity: f64,
    /// Average execution price
    pub average_execution_price: Option<f64>,
    /// Completion rate (0-1)
    pub completion_rate: f64,
    /// Fill rate (0-1)
    pub fill_rate: f64,
    /// Whether execution is complete
    pub is_complete: bool,
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_twap_executor_creation() {
        let config = TWAPConfig::default();
        let executor = TWAPExecutor::new(config.clone());

        assert_eq!(executor.slices().len(), config.num_slices);
        assert!(!executor.is_complete());
        assert_eq!(executor.progress(), 0.0);
    }

    #[test]
    fn test_slice_generation() {
        let config = TWAPConfig {
            total_quantity: 1000.0,
            num_slices: 5,
            duration: Duration::from_secs(100),
            ..Default::default()
        };

        let executor = TWAPExecutor::new(config);
        let slices = executor.slices();

        assert_eq!(slices.len(), 5);

        // Check total quantity is preserved
        let total: f64 = slices.iter().map(|s| s.quantity).sum();
        assert!((total - 1000.0).abs() < 0.01);
    }

    #[test]
    fn test_execution_flow() {
        let config = TWAPConfig {
            total_quantity: 1000.0,
            num_slices: 3,
            duration: Duration::from_secs(3),
            ..Default::default()
        };

        let mut executor = TWAPExecutor::new(config);
        executor.start();

        // Execute all slices
        for i in 0..3 {
            std::thread::sleep(Duration::from_millis(1100));

            if let Some(_slice) = executor.next_slice() {
                executor.mark_executed(333.33, 100.0 + i as f64);
            }
        }

        assert!(executor.is_complete());
        assert_eq!(executor.progress(), 100.0);
    }

    #[test]
    fn test_average_price_calculation() {
        let config = TWAPConfig {
            total_quantity: 600.0,
            num_slices: 3,
            ..Default::default()
        };

        let mut executor = TWAPExecutor::new(config);
        executor.start();

        // Execute slices at different prices
        executor.mark_executed(200.0, 100.0); // 200 @ 100
        executor.mark_executed(200.0, 110.0); // 200 @ 110
        executor.mark_executed(200.0, 105.0); // 200 @ 105

        let avg_price = executor.average_price().unwrap();
        let expected = (200.0 * 100.0 + 200.0 * 110.0 + 200.0 * 105.0) / 600.0;

        assert!((avg_price - expected).abs() < 0.01);
    }

    #[test]
    fn test_statistics() {
        let config = TWAPConfig {
            total_quantity: 1000.0,
            num_slices: 4,
            ..Default::default()
        };

        let mut executor = TWAPExecutor::new(config);
        executor.start();

        // Execute 2 out of 4 slices
        executor.mark_executed(250.0, 100.0);
        executor.mark_executed(250.0, 102.0);

        let stats = executor.statistics();

        assert_eq!(stats.total_slices, 4);
        assert_eq!(stats.executed_slices, 2);
        assert_eq!(stats.remaining_slices, 2);
        assert_eq!(stats.executed_quantity, 500.0);
        assert_eq!(stats.remaining_quantity, 500.0);
        assert_eq!(stats.completion_rate, 0.5);
        assert_eq!(stats.fill_rate, 0.5);
        assert!(!stats.is_complete);
    }

    #[test]
    fn test_min_slice_size() {
        let config = TWAPConfig {
            total_quantity: 100.0,
            num_slices: 20,
            min_slice_size: 10.0,
            ..Default::default()
        };

        let executor = TWAPExecutor::new(config);

        for slice in executor.slices() {
            assert!(slice.quantity >= 10.0);
        }
    }

    #[test]
    fn test_reset() {
        let config = TWAPConfig {
            total_quantity: 300.0,
            num_slices: 3,
            ..Default::default()
        };

        let mut executor = TWAPExecutor::new(config);
        executor.start();

        executor.mark_executed(100.0, 100.0);
        assert!((executor.progress() - 100.0 / 3.0).abs() < 0.01);

        executor.reset();
        assert_eq!(executor.progress(), 0.0);
        assert!(!executor.is_complete());
        assert_eq!(executor.total_executed(), 0.0);
    }

    #[test]
    fn test_time_until_next() {
        let config = TWAPConfig {
            total_quantity: 100.0,
            num_slices: 2,
            duration: Duration::from_secs(10),
            ..Default::default()
        };

        let mut executor = TWAPExecutor::new(config);
        executor.start();

        let time_until = executor.time_until_next();
        assert!(time_until.is_some());
    }

    #[test]
    fn test_randomization_flags() {
        let config = TWAPConfig {
            total_quantity: 1000.0,
            num_slices: 5,
            duration: Duration::from_secs(100),
            randomize_timing: true,
            randomize_size: true,
            timing_randomness: 0.2,
            size_randomness: 0.1,
            ..Default::default()
        };

        let executor = TWAPExecutor::new(config);
        assert_eq!(executor.slices().len(), 5);
    }

    #[test]
    fn test_total_executed() {
        let config = TWAPConfig::default();
        let mut executor = TWAPExecutor::new(config);
        executor.start();

        assert_eq!(executor.total_executed(), 0.0);

        executor.mark_executed(100.0, 100.0);
        assert_eq!(executor.total_executed(), 100.0);

        executor.mark_executed(150.0, 101.0);
        assert_eq!(executor.total_executed(), 250.0);
    }
}
