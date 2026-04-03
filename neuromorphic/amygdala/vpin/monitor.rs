//! VPIN Monitor - Real-time flash crash detection with adaptive thresholds
//!
//! Enhanced monitoring with:
//! - Adaptive threshold calibration
//! - Historical VPIN analytics
//! - Multi-timeframe analysis
//! - Alert level system (info, warning, critical)

use super::calculator::VPINCalculator;
use crate::amygdala::kill_switch::KillSwitch;
use std::collections::VecDeque;
use std::sync::Arc;

#[derive(Debug, Clone)]
pub struct Trade {
    pub volume: f64,
    pub is_buy: bool,
    pub price: f64,
    pub timestamp: i64,
}

/// Alert severity levels
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum AlertLevel {
    Info,
    Warning,
    Critical,
}

/// VPIN statistics for a time window
#[derive(Debug, Clone, Default)]
pub struct VPINStats {
    pub mean: f64,
    pub std_dev: f64,
    pub max: f64,
    pub min: f64,
    pub current: f64,
}

/// Configuration for VPIN monitoring
#[derive(Debug, Clone)]
pub struct VPINMonitorConfig {
    /// Base threshold (can be adapted)
    pub base_threshold: f64,
    /// Warning threshold (before critical)
    pub warning_threshold: f64,
    /// Enable adaptive thresholds
    pub adaptive: bool,
    /// Historical window size for calibration
    pub calibration_window: usize,
    /// Minimum samples before adaptive kicks in
    pub min_samples: usize,
}

impl Default for VPINMonitorConfig {
    fn default() -> Self {
        Self {
            base_threshold: 0.7,
            warning_threshold: 0.5,
            adaptive: true,
            calibration_window: 1000,
            min_samples: 100,
        }
    }
}

pub struct VPINMonitor {
    calculator: VPINCalculator,
    kill_switch: Arc<KillSwitch>,
    config: VPINMonitorConfig,
    last_vpin: f64,
    /// Historical VPIN values for calibration
    history: VecDeque<f64>,
    /// Number of alerts triggered
    alert_count: u64,
    /// Last alert level
    last_alert: Option<AlertLevel>,
}

impl VPINMonitor {
    /// Create a new VPIN monitor with default config
    pub fn new(bucket_size: f64, kill_switch: Arc<KillSwitch>, threshold: f64) -> Self {
        let config = VPINMonitorConfig {
            base_threshold: threshold,
            ..Default::default()
        };
        Self::with_config(bucket_size, kill_switch, config)
    }

    /// Create a new VPIN monitor with custom configuration
    pub fn with_config(
        bucket_size: f64,
        kill_switch: Arc<KillSwitch>,
        config: VPINMonitorConfig,
    ) -> Self {
        let calibration_window = config.calibration_window;
        Self {
            calculator: VPINCalculator::new(bucket_size),
            kill_switch,
            config,
            last_vpin: 0.0,
            history: VecDeque::with_capacity(calibration_window),
            alert_count: 0,
            last_alert: None,
        }
    }

    /// Process a new trade and check for threats with enhanced detection
    pub async fn check_trade(&mut self, trade: &Trade) -> Result<f64, String> {
        // Update VPIN with trade data
        self.calculator.update(trade.volume, trade.is_buy);

        // Calculate current VPIN
        let vpin = self.calculator.calculate();
        self.last_vpin = vpin;

        // Update history for calibration
        self.update_history(vpin);

        // Get current threshold (adaptive or static)
        let threshold = self.current_threshold();
        let warning_threshold = self.config.warning_threshold;

        // Determine alert level
        let alert_level = if vpin >= threshold {
            Some(AlertLevel::Critical)
        } else if vpin >= warning_threshold {
            Some(AlertLevel::Warning)
        } else {
            None
        };

        // Handle alerts
        if let Some(level) = alert_level {
            self.handle_alert(level, vpin, threshold).await?;
        } else if self.last_alert.is_some() {
            // Alert cleared
            tracing::info!("✓ VPIN returned to normal: {:.4}", vpin);
            self.last_alert = None;
        }

        Ok(vpin)
    }

    /// Handle alert at specified level
    async fn handle_alert(
        &mut self,
        level: AlertLevel,
        vpin: f64,
        threshold: f64,
    ) -> Result<(), String> {
        // Only increment if this is a new alert or escalation
        if self.last_alert.is_none() || self.last_alert.unwrap() != level {
            self.alert_count += 1;
        }
        self.last_alert = Some(level);

        match level {
            AlertLevel::Info => {
                tracing::info!("ℹ️  VPIN elevated: {:.4}", vpin);
            }
            AlertLevel::Warning => {
                tracing::warn!(
                    "⚠️  VPIN warning level: {:.4} (threshold: {:.4})",
                    vpin,
                    self.config.warning_threshold
                );
            }
            AlertLevel::Critical => {
                tracing::error!(
                    "🚨 VPIN CRITICAL: {:.4} > {:.4} - Triggering kill switch!",
                    vpin,
                    threshold
                );

                // Trigger kill switch
                self.kill_switch
                    .trigger(
                        vpin as f32,
                        &format!(
                            "VPIN critical threshold exceeded - toxic flow detected (VPIN: {:.4}, threshold: {:.4})",
                            vpin, threshold
                        ),
                    )
                    .await
                    .map_err(|e: crate::common::Error| {
                        format!("Failed to trigger kill switch: {:?}", e)
                    })?;
            }
        }

        Ok(())
    }

    /// Update historical VPIN values
    fn update_history(&mut self, vpin: f64) {
        if self.history.len() >= self.config.calibration_window {
            self.history.pop_front();
        }
        self.history.push_back(vpin);
    }

    /// Get current adaptive or static threshold
    fn current_threshold(&self) -> f64 {
        if !self.config.adaptive || self.history.len() < self.config.min_samples {
            return self.config.base_threshold;
        }

        // Calculate adaptive threshold based on historical statistics
        let stats = self.calculate_stats();

        // Adaptive threshold: mean + 2*std_dev (captures ~95% of normal behavior)
        let adaptive = stats.mean + 2.0 * stats.std_dev;

        // Don't let adaptive threshold go below base threshold
        adaptive.max(self.config.base_threshold)
    }

    /// Calculate statistics from historical data
    pub fn calculate_stats(&self) -> VPINStats {
        if self.history.is_empty() {
            return VPINStats::default();
        }

        let n = self.history.len() as f64;
        let sum: f64 = self.history.iter().sum();
        let mean = sum / n;

        let variance: f64 = self
            .history
            .iter()
            .map(|v| {
                let diff = v - mean;
                diff * diff
            })
            .sum::<f64>()
            / n;

        let std_dev = variance.sqrt();
        let max = self
            .history
            .iter()
            .copied()
            .fold(f64::NEG_INFINITY, f64::max);
        let min = self.history.iter().copied().fold(f64::INFINITY, f64::min);

        VPINStats {
            mean,
            std_dev,
            max,
            min,
            current: self.last_vpin,
        }
    }

    /// Calibrate threshold based on historical data
    pub fn calibrate(&mut self, percentile: f64) -> f64 {
        if self.history.is_empty() {
            return self.config.base_threshold;
        }

        // Sort history to find percentile
        let mut sorted: Vec<f64> = self.history.iter().copied().collect();
        sorted.sort_by(|a, b| a.partial_cmp(b).unwrap());

        let index = ((sorted.len() - 1) as f64 * percentile) as usize;
        let threshold = sorted[index];

        tracing::info!(
            "📊 VPIN calibration: {:.0}th percentile = {:.4} (n={})",
            percentile * 100.0,
            threshold,
            sorted.len()
        );

        threshold
    }

    /// Get alert statistics
    pub fn alert_stats(&self) -> (u64, Option<AlertLevel>) {
        (self.alert_count, self.last_alert)
    }

    /// Get current VPIN value
    pub fn current_vpin(&self) -> f64 {
        self.last_vpin
    }

    /// Check if currently dangerous
    pub fn is_dangerous(&self) -> bool {
        let threshold = self.current_threshold();
        self.last_vpin >= threshold
    }

    /// Update base threshold
    pub fn set_threshold(&mut self, threshold: f64) {
        self.config.base_threshold = threshold;
        tracing::info!("Updated VPIN base threshold to: {:.4}", threshold);
    }

    /// Get current effective threshold (may be adaptive)
    pub fn threshold(&self) -> f64 {
        self.current_threshold()
    }

    /// Get base threshold
    pub fn base_threshold(&self) -> f64 {
        self.config.base_threshold
    }

    /// Get number of historical samples
    pub fn sample_count(&self) -> usize {
        self.history.len()
    }

    /// Check if adaptive mode is enabled and active
    pub fn is_adaptive(&self) -> bool {
        self.config.adaptive && self.history.len() >= self.config.min_samples
    }

    /// Reset calibration history
    pub fn reset_calibration(&mut self) {
        self.history.clear();
        self.alert_count = 0;
        self.last_alert = None;
        tracing::info!("🔄 VPIN calibration reset");
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::amygdala::kill_switch::KillSwitch;

    #[tokio::test]
    async fn test_vpin_monitor_normal_flow() {
        let kill_switch = Arc::new(KillSwitch::new(0.8));
        let mut monitor = VPINMonitor::new(1000.0, kill_switch.clone(), 0.7);

        // Balanced flow - should not trigger
        let trade1 = Trade {
            volume: 500.0,
            is_buy: true,
            price: 100.0,
            timestamp: 1000,
        };

        let trade2 = Trade {
            volume: 500.0,
            is_buy: false,
            price: 100.0,
            timestamp: 1001,
        };

        monitor.check_trade(&trade1).await.unwrap();
        monitor.check_trade(&trade2).await.unwrap();

        assert!(!kill_switch.is_triggered());
        assert!(!monitor.is_dangerous());
    }

    #[tokio::test]
    async fn test_vpin_monitor_imbalanced_flow() {
        let kill_switch = Arc::new(KillSwitch::new(0.5));
        let mut monitor = VPINMonitor::new(1000.0, kill_switch.clone(), 0.6);

        // Highly imbalanced flow
        for _ in 0..10 {
            let trade = Trade {
                volume: 100.0,
                is_buy: true,
                price: 100.0,
                timestamp: 1000,
            };
            let _ = monitor.check_trade(&trade).await;
        }

        // Should detect high imbalance
        let vpin = monitor.current_vpin();
        assert!(vpin > 0.0);
    }

    #[test]
    fn test_threshold_update() {
        let kill_switch = Arc::new(KillSwitch::new(0.8));
        let mut monitor = VPINMonitor::new(1000.0, kill_switch, 0.7);

        assert_eq!(monitor.base_threshold(), 0.7);

        monitor.set_threshold(0.5);
        assert_eq!(monitor.base_threshold(), 0.5);
    }

    #[tokio::test]
    async fn test_adaptive_threshold() {
        let kill_switch = Arc::new(KillSwitch::new(0.8));
        let config = VPINMonitorConfig {
            base_threshold: 0.5,
            warning_threshold: 0.3,
            adaptive: true,
            calibration_window: 100,
            min_samples: 10,
        };
        let mut monitor = VPINMonitor::with_config(1000.0, kill_switch, config);

        // Build up history with low VPIN values
        for i in 0..50 {
            let trade = Trade {
                volume: 100.0,
                is_buy: i % 2 == 0,
                price: 100.0,
                timestamp: i,
            };
            let _ = monitor.check_trade(&trade).await;
        }

        assert!(monitor.is_adaptive());
        assert!(monitor.sample_count() > 10);

        let stats = monitor.calculate_stats();
        assert!(stats.mean >= 0.0);
        assert!(stats.std_dev >= 0.0);
    }

    #[test]
    fn test_calibration() {
        let kill_switch = Arc::new(KillSwitch::new(0.8));
        let mut monitor = VPINMonitor::new(1000.0, kill_switch, 0.7);

        // Add some history
        for i in 0..100 {
            monitor.update_history(0.1 + (i as f64 * 0.005));
        }

        // Calibrate at 95th percentile
        let p95 = monitor.calibrate(0.95);
        assert!(p95 > 0.1 && p95 < 0.7);
    }

    #[tokio::test]
    async fn test_alert_levels() {
        let kill_switch = Arc::new(KillSwitch::new(0.8));
        let config = VPINMonitorConfig {
            base_threshold: 0.8,
            warning_threshold: 0.5,
            adaptive: false,
            ..Default::default()
        };
        let mut monitor = VPINMonitor::with_config(1000.0, kill_switch.clone(), config);

        // Normal trade
        let normal_trade = Trade {
            volume: 100.0,
            is_buy: true,
            price: 100.0,
            timestamp: 1000,
        };
        let _ = monitor.check_trade(&normal_trade).await;

        let (count, level) = monitor.alert_stats();
        assert_eq!(count, 0);
        assert!(level.is_none());
    }

    #[test]
    fn test_reset_calibration() {
        let kill_switch = Arc::new(KillSwitch::new(0.8));
        let mut monitor = VPINMonitor::new(1000.0, kill_switch, 0.7);

        // Add history
        for i in 0..50 {
            monitor.update_history(i as f64);
        }

        assert_eq!(monitor.sample_count(), 50);

        monitor.reset_calibration();
        assert_eq!(monitor.sample_count(), 0);

        let (count, level) = monitor.alert_stats();
        assert_eq!(count, 0);
        assert!(level.is_none());
    }
}
