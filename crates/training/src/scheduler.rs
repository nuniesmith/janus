//! Learning Rate Schedulers for Training.
//!
//! This module provides various learning rate scheduling strategies to improve
//! training convergence and performance.
//!
//! # Supported Schedulers
//!
//! - **Constant**: Fixed learning rate throughout training
//! - **StepLR**: Decay learning rate by gamma every N steps
//! - **CosineAnnealing**: Cosine annealing schedule
//! - **WarmupCosine**: Warmup followed by cosine annealing (recommended)
//! - **LinearWarmup**: Linear warmup to target learning rate
//!
//! # Example
//!
//! ```ignore
//! use training::scheduler::{LRScheduler, LRSchedulerConfig};
//!
//! let config = LRSchedulerConfig::warmup_cosine()
//!     .warmup_steps(1000)
//!     .total_steps(100_000)
//!     .min_lr(1e-6)
//!     .build();
//!
//! let mut scheduler = LRScheduler::from_config(config, 1e-4);
//!
//! for step in 0..100_000 {
//!     let lr = scheduler.get_lr(step);
//!     optimizer.set_learning_rate(lr);
//!     // ... training step ...
//! }
//! ```

use serde::{Deserialize, Serialize};
use std::f64::consts::PI;

/// Type of learning rate scheduler.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Serialize, Deserialize, Default)]
pub enum SchedulerType {
    /// Constant learning rate
    Constant,

    /// Step decay: lr = initial_lr * gamma^(step / step_size)
    StepLR,

    /// Cosine annealing: lr = min_lr + 0.5 * (max_lr - min_lr) * (1 + cos(π * t / T))
    CosineAnnealing,

    /// Linear warmup followed by cosine annealing (recommended)
    #[default]
    WarmupCosine,

    /// Linear warmup to target learning rate
    LinearWarmup,

    /// Exponential decay: lr = initial_lr * decay_rate^step
    ExponentialDecay,
}

/// Configuration for learning rate schedulers.
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct LRSchedulerConfig {
    /// Type of scheduler
    pub scheduler_type: SchedulerType,

    /// Number of warmup steps (for warmup schedulers)
    pub warmup_steps: usize,

    /// Total number of training steps (for cosine annealing)
    pub total_steps: usize,

    /// Minimum learning rate (for cosine annealing)
    pub min_lr: f64,

    /// Step size for StepLR (decay every N steps)
    pub step_size: usize,

    /// Gamma for StepLR (multiplicative factor)
    pub gamma: f64,

    /// Decay rate for exponential decay
    pub decay_rate: f64,

    /// Number of cycles for cosine annealing (default: 1)
    pub num_cycles: f64,
}

impl Default for LRSchedulerConfig {
    fn default() -> Self {
        Self::warmup_cosine()
    }
}

impl LRSchedulerConfig {
    /// Create a constant learning rate scheduler.
    pub fn constant() -> Self {
        Self {
            scheduler_type: SchedulerType::Constant,
            warmup_steps: 0,
            total_steps: 100_000,
            min_lr: 0.0,
            step_size: 10_000,
            gamma: 0.1,
            decay_rate: 0.96,
            num_cycles: 1.0,
        }
    }

    /// Create a step decay scheduler.
    pub fn step_lr() -> Self {
        Self {
            scheduler_type: SchedulerType::StepLR,
            warmup_steps: 0,
            total_steps: 100_000,
            min_lr: 0.0,
            step_size: 10_000,
            gamma: 0.1,
            decay_rate: 0.96,
            num_cycles: 1.0,
        }
    }

    /// Create a cosine annealing scheduler.
    pub fn cosine_annealing() -> Self {
        Self {
            scheduler_type: SchedulerType::CosineAnnealing,
            warmup_steps: 0,
            total_steps: 100_000,
            min_lr: 1e-6,
            step_size: 10_000,
            gamma: 0.1,
            decay_rate: 0.96,
            num_cycles: 1.0,
        }
    }

    /// Create a warmup + cosine annealing scheduler (recommended).
    pub fn warmup_cosine() -> Self {
        Self {
            scheduler_type: SchedulerType::WarmupCosine,
            warmup_steps: 1000,
            total_steps: 100_000,
            min_lr: 1e-6,
            step_size: 10_000,
            gamma: 0.1,
            decay_rate: 0.96,
            num_cycles: 1.0,
        }
    }

    /// Create a linear warmup scheduler.
    pub fn linear_warmup() -> Self {
        Self {
            scheduler_type: SchedulerType::LinearWarmup,
            warmup_steps: 1000,
            total_steps: 100_000,
            min_lr: 0.0,
            step_size: 10_000,
            gamma: 0.1,
            decay_rate: 0.96,
            num_cycles: 1.0,
        }
    }

    /// Create an exponential decay scheduler.
    pub fn exponential_decay() -> Self {
        Self {
            scheduler_type: SchedulerType::ExponentialDecay,
            warmup_steps: 0,
            total_steps: 100_000,
            min_lr: 1e-7,
            step_size: 10_000,
            gamma: 0.1,
            decay_rate: 0.9999,
            num_cycles: 1.0,
        }
    }

    /// Set the number of warmup steps.
    pub fn warmup_steps(mut self, steps: usize) -> Self {
        self.warmup_steps = steps;
        self
    }

    /// Set the total number of training steps.
    pub fn total_steps(mut self, steps: usize) -> Self {
        self.total_steps = steps;
        self
    }

    /// Set the minimum learning rate.
    pub fn min_lr(mut self, lr: f64) -> Self {
        self.min_lr = lr;
        self
    }

    /// Set the step size (for StepLR).
    pub fn step_size(mut self, size: usize) -> Self {
        self.step_size = size;
        self
    }

    /// Set the gamma (for StepLR).
    pub fn gamma(mut self, gamma: f64) -> Self {
        self.gamma = gamma;
        self
    }

    /// Set the decay rate (for ExponentialDecay).
    pub fn decay_rate(mut self, rate: f64) -> Self {
        self.decay_rate = rate;
        self
    }

    /// Set the number of cosine cycles.
    pub fn num_cycles(mut self, cycles: f64) -> Self {
        self.num_cycles = cycles;
        self
    }

    /// Build the configuration.
    pub fn build(self) -> Self {
        self
    }
}

/// Learning rate scheduler.
pub struct LRScheduler {
    config: LRSchedulerConfig,
    initial_lr: f64,
}

impl LRScheduler {
    /// Create a new scheduler from configuration.
    pub fn from_config(config: LRSchedulerConfig, initial_lr: f64) -> Self {
        Self { config, initial_lr }
    }

    /// Get the learning rate for the given step.
    pub fn get_lr(&self, step: usize) -> f64 {
        match self.config.scheduler_type {
            SchedulerType::Constant => self.constant_lr(),
            SchedulerType::StepLR => self.step_lr(step),
            SchedulerType::CosineAnnealing => self.cosine_annealing(step),
            SchedulerType::WarmupCosine => self.warmup_cosine(step),
            SchedulerType::LinearWarmup => self.linear_warmup(step),
            SchedulerType::ExponentialDecay => self.exponential_decay(step),
        }
    }

    /// Get the configuration.
    pub fn config(&self) -> &LRSchedulerConfig {
        &self.config
    }

    /// Get the initial learning rate.
    pub fn initial_lr(&self) -> f64 {
        self.initial_lr
    }

    // =========================================================================
    // Scheduler implementations
    // =========================================================================

    /// Constant learning rate.
    fn constant_lr(&self) -> f64 {
        self.initial_lr
    }

    /// Step decay: lr = initial_lr * gamma^(step / step_size)
    fn step_lr(&self, step: usize) -> f64 {
        let decay_count = step / self.config.step_size;
        self.initial_lr * self.config.gamma.powi(decay_count as i32)
    }

    /// Cosine annealing: lr = min_lr + 0.5 * (max_lr - min_lr) * (1 + cos(π * t / T))
    fn cosine_annealing(&self, step: usize) -> f64 {
        let progress = (step as f64 / self.config.total_steps as f64).min(1.0);
        let cosine_decay = 0.5 * (1.0 + (PI * progress * self.config.num_cycles).cos());

        self.config.min_lr + (self.initial_lr - self.config.min_lr) * cosine_decay
    }

    /// Warmup + cosine annealing.
    fn warmup_cosine(&self, step: usize) -> f64 {
        if step < self.config.warmup_steps {
            // Linear warmup
            let warmup_progress = step as f64 / self.config.warmup_steps as f64;
            self.initial_lr * warmup_progress
        } else {
            // Cosine annealing after warmup
            let annealing_step = step - self.config.warmup_steps;
            let annealing_total = self.config.total_steps - self.config.warmup_steps;
            let progress = (annealing_step as f64 / annealing_total as f64).min(1.0);
            let cosine_decay = 0.5 * (1.0 + (PI * progress * self.config.num_cycles).cos());

            self.config.min_lr + (self.initial_lr - self.config.min_lr) * cosine_decay
        }
    }

    /// Linear warmup to target learning rate.
    fn linear_warmup(&self, step: usize) -> f64 {
        if step < self.config.warmup_steps {
            let warmup_progress = step as f64 / self.config.warmup_steps as f64;
            self.initial_lr * warmup_progress
        } else {
            self.initial_lr
        }
    }

    /// Exponential decay: lr = initial_lr * decay_rate^step
    fn exponential_decay(&self, step: usize) -> f64 {
        let lr = self.initial_lr * self.config.decay_rate.powi(step as i32);
        lr.max(self.config.min_lr)
    }
}

// =============================================================================
// Utility Functions
// =============================================================================

/// Create a warmup + cosine annealing schedule (common default).
pub fn warmup_cosine_schedule(
    initial_lr: f64,
    warmup_steps: usize,
    total_steps: usize,
    min_lr: f64,
) -> LRScheduler {
    let config = LRSchedulerConfig::warmup_cosine()
        .warmup_steps(warmup_steps)
        .total_steps(total_steps)
        .min_lr(min_lr)
        .build();

    LRScheduler::from_config(config, initial_lr)
}

/// Create a step decay schedule.
pub fn step_decay_schedule(initial_lr: f64, step_size: usize, gamma: f64) -> LRScheduler {
    let config = LRSchedulerConfig::step_lr()
        .step_size(step_size)
        .gamma(gamma)
        .build();

    LRScheduler::from_config(config, initial_lr)
}

// =============================================================================
// Tests
// =============================================================================

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_constant_scheduler() {
        let scheduler = LRScheduler::from_config(LRSchedulerConfig::constant(), 1e-3);

        assert_eq!(scheduler.get_lr(0), 1e-3);
        assert_eq!(scheduler.get_lr(1000), 1e-3);
        assert_eq!(scheduler.get_lr(10000), 1e-3);
    }

    #[test]
    fn test_step_lr() {
        let config = LRSchedulerConfig::step_lr()
            .step_size(100)
            .gamma(0.1)
            .build();

        let scheduler = LRScheduler::from_config(config, 1.0);

        assert_eq!(scheduler.get_lr(0), 1.0);
        assert_eq!(scheduler.get_lr(99), 1.0);
        assert!((scheduler.get_lr(100) - 0.1).abs() < 1e-10);
        assert!((scheduler.get_lr(200) - 0.01).abs() < 1e-10);
    }

    #[test]
    fn test_cosine_annealing() {
        let config = LRSchedulerConfig::cosine_annealing()
            .total_steps(1000)
            .min_lr(0.0)
            .build();

        let scheduler = LRScheduler::from_config(config, 1.0);

        // At step 0, should be close to initial_lr
        let lr_start = scheduler.get_lr(0);
        assert!((lr_start - 1.0).abs() < 0.01);

        // At step 500 (halfway), cosine gives 0.5 of range
        // lr = 0.0 + (1.0 - 0.0) * 0.5 = 0.5
        let lr_mid = scheduler.get_lr(500);
        assert!((lr_mid - 0.5).abs() < 0.1);

        // At step 1000 (end), should be at min_lr
        let lr_end = scheduler.get_lr(1000);
        assert!(lr_end < 0.01);
    }

    #[test]
    fn test_warmup_cosine() {
        let config = LRSchedulerConfig::warmup_cosine()
            .warmup_steps(100)
            .total_steps(1000)
            .min_lr(1e-6)
            .build();

        let scheduler = LRScheduler::from_config(config, 1.0);

        // During warmup, should increase linearly
        let lr_warmup_start = scheduler.get_lr(0);
        assert!(lr_warmup_start < 0.1);

        let lr_warmup_mid = scheduler.get_lr(50);
        assert!((lr_warmup_mid - 0.5).abs() < 0.1);

        let lr_warmup_end = scheduler.get_lr(100);
        assert!((lr_warmup_end - 1.0).abs() < 0.1);

        // After warmup, should follow cosine annealing
        let lr_after_warmup = scheduler.get_lr(500);
        assert!(lr_after_warmup < 1.0);
        assert!(lr_after_warmup > 0.0);
    }

    #[test]
    fn test_linear_warmup() {
        let config = LRSchedulerConfig::linear_warmup().warmup_steps(100).build();

        let scheduler = LRScheduler::from_config(config, 1.0);

        // Should increase linearly during warmup
        let lr_start = scheduler.get_lr(0);
        assert_eq!(lr_start, 0.0);

        let lr_mid = scheduler.get_lr(50);
        assert!((lr_mid - 0.5).abs() < 1e-10);

        // After warmup, should stay constant
        let lr_end = scheduler.get_lr(100);
        assert!((lr_end - 1.0).abs() < 1e-10);

        let lr_after = scheduler.get_lr(1000);
        assert_eq!(lr_after, 1.0);
    }

    #[test]
    fn test_exponential_decay() {
        let config = LRSchedulerConfig::exponential_decay()
            .decay_rate(0.99)
            .min_lr(1e-5)
            .build();

        let scheduler = LRScheduler::from_config(config, 1.0);

        let lr_start = scheduler.get_lr(0);
        assert_eq!(lr_start, 1.0);

        let lr_step_1 = scheduler.get_lr(1);
        assert!((lr_step_1 - 0.99).abs() < 1e-10);

        let lr_step_100 = scheduler.get_lr(100);
        assert!(lr_step_100 < 1.0);
        assert!(lr_step_100 > 0.0);

        // Should never go below min_lr
        let lr_step_10000 = scheduler.get_lr(10000);
        assert!(lr_step_10000 >= 1e-5);
    }

    #[test]
    fn test_warmup_cosine_schedule_helper() {
        let scheduler = warmup_cosine_schedule(1e-4, 1000, 100_000, 1e-6);

        assert_eq!(scheduler.initial_lr(), 1e-4);
        assert_eq!(scheduler.config().warmup_steps, 1000);
        assert_eq!(scheduler.config().total_steps, 100_000);
    }

    #[test]
    fn test_step_decay_schedule_helper() {
        let scheduler = step_decay_schedule(1e-3, 10_000, 0.5);

        assert_eq!(scheduler.initial_lr(), 1e-3);
        assert_eq!(scheduler.config().step_size, 10_000);
        assert_eq!(scheduler.config().gamma, 0.5);
    }

    #[test]
    fn test_scheduler_monotonicity() {
        // Warmup phase should be monotonically increasing
        let scheduler = LRScheduler::from_config(
            LRSchedulerConfig::warmup_cosine()
                .warmup_steps(100)
                .total_steps(1000)
                .build(),
            1.0,
        );

        let mut prev_lr = 0.0;
        for step in 0..100 {
            let lr = scheduler.get_lr(step);
            assert!(lr >= prev_lr, "Warmup should be monotonically increasing");
            prev_lr = lr;
        }
    }

    #[test]
    fn test_cosine_annealing_bounds() {
        let config = LRSchedulerConfig::cosine_annealing()
            .total_steps(1000)
            .min_lr(1e-5)
            .build();

        let scheduler = LRScheduler::from_config(config, 1e-3);

        // Check that lr stays within bounds
        for step in 0..=1000 {
            let lr = scheduler.get_lr(step);
            assert!(lr >= 1e-5, "LR should not go below min_lr");
            assert!(lr <= 1e-3, "LR should not exceed initial_lr");
        }
    }
}
