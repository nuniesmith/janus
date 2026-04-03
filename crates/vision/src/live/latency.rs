//! Latency tracking and optimization utilities for performance monitoring.
//!
//! This module provides comprehensive latency tracking and analysis tools:
//! - Real-time latency measurement with high precision
//! - Percentile calculations (P50, P95, P99, P999)
//! - Latency budget tracking and alerting
//! - Performance profiling and bottleneck detection
//! - SLA monitoring and violation tracking

use std::collections::VecDeque;
#[cfg_attr(not(test), allow(unused_imports))]
use std::time::Duration;
use std::time::Instant;

/// High-precision latency measurement.
#[derive(Debug, Clone, Copy)]
pub struct LatencyMeasurement {
    pub start: Instant,
    pub end: Option<Instant>,
}

impl LatencyMeasurement {
    /// Start a new latency measurement.
    pub fn start() -> Self {
        Self {
            start: Instant::now(),
            end: None,
        }
    }

    /// Stop the measurement and return duration in microseconds.
    pub fn stop(&mut self) -> u64 {
        self.end = Some(Instant::now());
        self.duration_us()
    }

    /// Get the duration in microseconds.
    pub fn duration_us(&self) -> u64 {
        let end = self.end.unwrap_or_else(Instant::now);
        end.duration_since(self.start).as_micros() as u64
    }

    /// Get the duration in milliseconds.
    pub fn duration_ms(&self) -> f64 {
        self.duration_us() as f64 / 1000.0
    }

    /// Get the duration in nanoseconds.
    pub fn duration_ns(&self) -> u128 {
        let end = self.end.unwrap_or_else(Instant::now);
        end.duration_since(self.start).as_nanos()
    }
}

/// Latency tracker with rolling window statistics.
///
/// Maintains a sliding window of latency measurements and computes
/// real-time statistics including percentiles.
#[derive(Debug, Clone)]
pub struct LatencyTracker {
    measurements: VecDeque<u64>,
    window_size: usize,
    total_measurements: usize,
    total_latency_us: u64,
}

impl LatencyTracker {
    /// Create a new latency tracker with the given window size.
    pub fn new(window_size: usize) -> Self {
        Self {
            measurements: VecDeque::with_capacity(window_size),
            window_size,
            total_measurements: 0,
            total_latency_us: 0,
        }
    }

    /// Record a latency measurement in microseconds.
    pub fn record(&mut self, latency_us: u64) {
        if self.measurements.len() >= self.window_size {
            if let Some(old) = self.measurements.pop_front() {
                self.total_latency_us -= old;
            }
        }

        self.measurements.push_back(latency_us);
        self.total_latency_us += latency_us;
        self.total_measurements += 1;
    }

    /// Get the mean latency in microseconds.
    pub fn mean_us(&self) -> f64 {
        if self.measurements.is_empty() {
            0.0
        } else {
            self.total_latency_us as f64 / self.measurements.len() as f64
        }
    }

    /// Get the mean latency in milliseconds.
    pub fn mean_ms(&self) -> f64 {
        self.mean_us() / 1000.0
    }

    /// Get the minimum latency in microseconds.
    pub fn min_us(&self) -> u64 {
        self.measurements.iter().min().copied().unwrap_or(0)
    }

    /// Get the maximum latency in microseconds.
    pub fn max_us(&self) -> u64 {
        self.measurements.iter().max().copied().unwrap_or(0)
    }

    /// Get a percentile value in microseconds.
    pub fn percentile_us(&self, percentile: f64) -> u64 {
        if self.measurements.is_empty() {
            return 0;
        }

        let mut sorted: Vec<u64> = self.measurements.iter().copied().collect();
        sorted.sort_unstable();

        let index = ((percentile / 100.0) * sorted.len() as f64) as usize;
        let index = index.min(sorted.len() - 1);
        sorted[index]
    }

    /// Get P50 (median) latency in microseconds.
    pub fn p50_us(&self) -> u64 {
        self.percentile_us(50.0)
    }

    /// Get P95 latency in microseconds.
    pub fn p95_us(&self) -> u64 {
        self.percentile_us(95.0)
    }

    /// Get P99 latency in microseconds.
    pub fn p99_us(&self) -> u64 {
        self.percentile_us(99.0)
    }

    /// Get P999 latency in microseconds.
    pub fn p999_us(&self) -> u64 {
        self.percentile_us(99.9)
    }

    /// Get the total number of measurements recorded.
    pub fn total_measurements(&self) -> usize {
        self.total_measurements
    }

    /// Get the current window size.
    pub fn len(&self) -> usize {
        self.measurements.len()
    }

    /// Check if the tracker is empty.
    pub fn is_empty(&self) -> bool {
        self.measurements.is_empty()
    }

    /// Reset all measurements.
    pub fn reset(&mut self) {
        self.measurements.clear();
        self.total_measurements = 0;
        self.total_latency_us = 0;
    }

    /// Get a summary of latency statistics.
    pub fn summary(&self) -> LatencySummary {
        LatencySummary {
            count: self.len(),
            mean_us: self.mean_us(),
            min_us: self.min_us(),
            max_us: self.max_us(),
            p50_us: self.p50_us(),
            p95_us: self.p95_us(),
            p99_us: self.p99_us(),
            p999_us: self.p999_us(),
        }
    }
}

/// Summary of latency statistics.
#[derive(Debug, Clone)]
pub struct LatencySummary {
    pub count: usize,
    pub mean_us: f64,
    pub min_us: u64,
    pub max_us: u64,
    pub p50_us: u64,
    pub p95_us: u64,
    pub p99_us: u64,
    pub p999_us: u64,
}

impl LatencySummary {
    /// Print a formatted summary.
    pub fn print(&self) {
        println!("=== Latency Summary ===");
        println!("Count: {}", self.count);
        println!(
            "Mean: {:.2} μs ({:.2} ms)",
            self.mean_us,
            self.mean_us / 1000.0
        );
        println!(
            "Min: {} μs ({:.2} ms)",
            self.min_us,
            self.min_us as f64 / 1000.0
        );
        println!(
            "Max: {} μs ({:.2} ms)",
            self.max_us,
            self.max_us as f64 / 1000.0
        );
        println!(
            "P50: {} μs ({:.2} ms)",
            self.p50_us,
            self.p50_us as f64 / 1000.0
        );
        println!(
            "P95: {} μs ({:.2} ms)",
            self.p95_us,
            self.p95_us as f64 / 1000.0
        );
        println!(
            "P99: {} μs ({:.2} ms)",
            self.p99_us,
            self.p99_us as f64 / 1000.0
        );
        println!(
            "P999: {} μs ({:.2} ms)",
            self.p999_us,
            self.p999_us as f64 / 1000.0
        );
    }
}

/// Latency budget tracker with SLA monitoring.
///
/// Tracks whether operations complete within specified time budgets
/// and alerts on violations.
#[derive(Debug, Clone)]
pub struct LatencyBudget {
    name: String,
    budget_us: u64,
    violations: usize,
    total_checks: usize,
    tracker: LatencyTracker,
}

impl LatencyBudget {
    /// Create a new latency budget.
    pub fn new(name: String, budget_us: u64) -> Self {
        Self {
            name,
            budget_us,
            violations: 0,
            total_checks: 0,
            tracker: LatencyTracker::new(1000),
        }
    }

    /// Create a latency budget from milliseconds.
    pub fn from_ms(name: String, budget_ms: f64) -> Self {
        Self::new(name, (budget_ms * 1000.0) as u64)
    }

    /// Check if a latency meets the budget.
    pub fn check(&mut self, latency_us: u64) -> bool {
        self.total_checks += 1;
        self.tracker.record(latency_us);

        if latency_us > self.budget_us {
            self.violations += 1;
            false
        } else {
            true
        }
    }

    /// Get the violation rate.
    pub fn violation_rate(&self) -> f64 {
        if self.total_checks == 0 {
            0.0
        } else {
            self.violations as f64 / self.total_checks as f64
        }
    }

    /// Get the compliance rate (1 - violation_rate).
    pub fn compliance_rate(&self) -> f64 {
        1.0 - self.violation_rate()
    }

    /// Check if the budget meets a target compliance rate.
    pub fn meets_sla(&self, target_compliance: f64) -> bool {
        self.compliance_rate() >= target_compliance
    }

    /// Get the budget in microseconds.
    pub fn budget_us(&self) -> u64 {
        self.budget_us
    }

    /// Get the budget in milliseconds.
    pub fn budget_ms(&self) -> f64 {
        self.budget_us as f64 / 1000.0
    }

    /// Get the number of violations.
    pub fn violations(&self) -> usize {
        self.violations
    }

    /// Get the total number of checks.
    pub fn total_checks(&self) -> usize {
        self.total_checks
    }

    /// Get the latency tracker.
    pub fn tracker(&self) -> &LatencyTracker {
        &self.tracker
    }

    /// Reset all statistics.
    pub fn reset(&mut self) {
        self.violations = 0;
        self.total_checks = 0;
        self.tracker.reset();
    }

    /// Print a budget report.
    pub fn report(&self) {
        println!("=== Latency Budget Report: {} ===", self.name);
        println!("Budget: {} μs ({:.2} ms)", self.budget_us, self.budget_ms());
        println!("Total checks: {}", self.total_checks);
        println!("Violations: {}", self.violations);
        println!("Violation rate: {:.2}%", self.violation_rate() * 100.0);
        println!("Compliance rate: {:.2}%", self.compliance_rate() * 100.0);
        println!("\nLatency Statistics:");
        self.tracker.summary().print();
    }
}

/// Multi-stage latency profiler for identifying bottlenecks.
///
/// Tracks latency across multiple stages of a pipeline to identify
/// which stages are taking the most time.
#[derive(Debug)]
pub struct LatencyProfiler {
    stages: Vec<(String, LatencyTracker)>,
    current_stage: Option<(String, LatencyMeasurement)>,
}

impl LatencyProfiler {
    /// Create a new latency profiler.
    pub fn new() -> Self {
        Self {
            stages: Vec::new(),
            current_stage: None,
        }
    }

    /// Start timing a named stage.
    pub fn start_stage(&mut self, name: &str) {
        // Stop previous stage if any
        if let Some((stage_name, mut measurement)) = self.current_stage.take() {
            let latency = measurement.stop();
            self.record_stage_latency(&stage_name, latency);
        }

        self.current_stage = Some((name.to_string(), LatencyMeasurement::start()));
    }

    /// Stop the current stage.
    pub fn stop_stage(&mut self) {
        if let Some((stage_name, mut measurement)) = self.current_stage.take() {
            let latency = measurement.stop();
            self.record_stage_latency(&stage_name, latency);
        }
    }

    /// Record latency for a specific stage.
    fn record_stage_latency(&mut self, name: &str, latency_us: u64) {
        if let Some((_, tracker)) = self.stages.iter_mut().find(|(n, _)| n == name) {
            tracker.record(latency_us);
        } else {
            let mut tracker = LatencyTracker::new(1000);
            tracker.record(latency_us);
            self.stages.push((name.to_string(), tracker));
        }
    }

    /// Get the tracker for a specific stage.
    pub fn stage_tracker(&self, name: &str) -> Option<&LatencyTracker> {
        self.stages.iter().find(|(n, _)| n == name).map(|(_, t)| t)
    }

    /// Get all stage names.
    pub fn stage_names(&self) -> Vec<String> {
        self.stages.iter().map(|(n, _)| n.clone()).collect()
    }

    /// Get total latency across all stages.
    pub fn total_latency_us(&self) -> f64 {
        self.stages.iter().map(|(_, t)| t.mean_us()).sum()
    }

    /// Get the slowest stage.
    pub fn slowest_stage(&self) -> Option<(String, f64)> {
        self.stages
            .iter()
            .max_by(|(_, t1), (_, t2)| {
                t1.mean_us()
                    .partial_cmp(&t2.mean_us())
                    .unwrap_or(std::cmp::Ordering::Equal)
            })
            .map(|(n, t)| (n.clone(), t.mean_us()))
    }

    /// Reset all stage trackers.
    pub fn reset(&mut self) {
        for (_, tracker) in &mut self.stages {
            tracker.reset();
        }
        self.current_stage = None;
    }

    /// Print a profiling report.
    pub fn report(&self) {
        println!("=== Latency Profiling Report ===");
        println!(
            "Total pipeline latency: {:.2} μs ({:.2} ms)\n",
            self.total_latency_us(),
            self.total_latency_us() / 1000.0
        );

        for (name, tracker) in &self.stages {
            let summary = tracker.summary();
            let percentage = (summary.mean_us / self.total_latency_us()) * 100.0;
            println!("Stage: {}", name);
            println!(
                "  Mean: {:.2} μs ({:.2} ms) - {:.1}%",
                summary.mean_us,
                summary.mean_us / 1000.0,
                percentage
            );
            println!(
                "  P95: {} μs ({:.2} ms)",
                summary.p95_us,
                summary.p95_us as f64 / 1000.0
            );
            println!(
                "  P99: {} μs ({:.2} ms)",
                summary.p99_us,
                summary.p99_us as f64 / 1000.0
            );
            println!();
        }

        if let Some((stage, latency)) = self.slowest_stage() {
            println!("Bottleneck: {} ({:.2} μs)", stage, latency);
        }
    }
}

impl Default for LatencyProfiler {
    fn default() -> Self {
        Self::new()
    }
}

/// Scoped latency measurement helper.
///
/// Automatically measures latency when created and records it when dropped.
pub struct LatencyScope<'a> {
    tracker: &'a mut LatencyTracker,
    measurement: LatencyMeasurement,
}

impl<'a> LatencyScope<'a> {
    /// Create a new latency scope.
    pub fn new(tracker: &'a mut LatencyTracker) -> Self {
        Self {
            tracker,
            measurement: LatencyMeasurement::start(),
        }
    }
}

impl<'a> Drop for LatencyScope<'a> {
    fn drop(&mut self) {
        let latency = self.measurement.duration_us();
        self.tracker.record(latency);
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_latency_measurement() {
        let mut measurement = LatencyMeasurement::start();
        std::thread::sleep(Duration::from_millis(1));
        let latency = measurement.stop();

        assert!(latency >= 1000); // At least 1ms in microseconds
        assert_eq!(latency, measurement.duration_us());
    }

    #[test]
    fn test_latency_tracker_basic() {
        let mut tracker = LatencyTracker::new(100);
        tracker.record(100);
        tracker.record(200);
        tracker.record(300);

        assert_eq!(tracker.len(), 3);
        assert_eq!(tracker.mean_us(), 200.0);
        assert_eq!(tracker.min_us(), 100);
        assert_eq!(tracker.max_us(), 300);
    }

    #[test]
    fn test_latency_tracker_window() {
        let mut tracker = LatencyTracker::new(3);
        tracker.record(100);
        tracker.record(200);
        tracker.record(300);
        tracker.record(400); // Should evict 100

        assert_eq!(tracker.len(), 3);
        assert_eq!(tracker.min_us(), 200);
        assert_eq!(tracker.max_us(), 400);
    }

    #[test]
    fn test_percentiles() {
        let mut tracker = LatencyTracker::new(100);
        for i in 1..=100 {
            tracker.record(i * 10);
        }

        assert_eq!(tracker.p50_us(), 510);
        assert_eq!(tracker.p95_us(), 960);
        assert_eq!(tracker.p99_us(), 1000);
    }

    #[test]
    fn test_latency_budget() {
        let mut budget = LatencyBudget::new("test".to_string(), 1000);

        assert!(budget.check(500)); // Within budget
        assert!(budget.check(800)); // Within budget
        assert!(!budget.check(1500)); // Violation

        assert_eq!(budget.violations(), 1);
        assert_eq!(budget.total_checks(), 3);
        assert!((budget.violation_rate() - 1.0 / 3.0).abs() < 1e-10);
    }

    #[test]
    fn test_latency_budget_sla() {
        let mut budget = LatencyBudget::from_ms("test".to_string(), 10.0);

        for _ in 0..95 {
            budget.check(5000); // 5ms - within budget
        }
        for _ in 0..5 {
            budget.check(15000); // 15ms - violation
        }

        assert_eq!(budget.compliance_rate(), 0.95);
        assert!(budget.meets_sla(0.95));
        assert!(!budget.meets_sla(0.96));
    }

    #[test]
    fn test_latency_profiler() {
        let mut profiler = LatencyProfiler::new();

        profiler.start_stage("stage1");
        std::thread::sleep(Duration::from_micros(100));

        profiler.start_stage("stage2");
        std::thread::sleep(Duration::from_micros(200));

        profiler.stop_stage();

        assert_eq!(profiler.stage_names().len(), 2);
        assert!(profiler.total_latency_us() > 0.0);
    }

    #[test]
    fn test_latency_scope() {
        let mut tracker = LatencyTracker::new(100);

        {
            let _scope = LatencyScope::new(&mut tracker);
            std::thread::sleep(Duration::from_micros(100));
        }

        assert_eq!(tracker.len(), 1);
        assert!(tracker.mean_us() >= 100.0);
    }

    #[test]
    fn test_tracker_reset() {
        let mut tracker = LatencyTracker::new(100);
        tracker.record(100);
        tracker.record(200);

        tracker.reset();
        assert_eq!(tracker.len(), 0);
        assert_eq!(tracker.total_measurements(), 0);
    }

    #[test]
    fn test_empty_tracker() {
        let tracker = LatencyTracker::new(100);
        assert_eq!(tracker.mean_us(), 0.0);
        assert_eq!(tracker.min_us(), 0);
        assert_eq!(tracker.max_us(), 0);
        assert_eq!(tracker.p50_us(), 0);
    }

    #[test]
    fn test_profiler_slowest_stage() {
        let mut profiler = LatencyProfiler::new();

        profiler.start_stage("fast");
        std::thread::sleep(Duration::from_micros(50));

        profiler.start_stage("slow");
        std::thread::sleep(Duration::from_micros(200));

        profiler.stop_stage();

        let (name, _) = profiler.slowest_stage().unwrap();
        assert_eq!(name, "slow");
    }
}
