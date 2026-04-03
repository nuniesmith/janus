//! Detect high-priority replay events
//!
//! Part of the Hippocampus region
//! Component: swr
//!
//! Sharp-Wave Ripples (SWR) are neural events that trigger memory replay.
//! This module detects significant trading events that warrant replay,
//! including large PnL swings, novel patterns, and high-impact outcomes.

use crate::common::Result;
use std::collections::{HashMap, VecDeque};

/// Type of ripple event detected
#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash)]
pub enum RippleType {
    /// Large profit event
    LargeProfit,
    /// Large loss event
    LargeLoss,
    /// Novel pattern detected
    NovelPattern,
    /// Anomalous market behavior
    MarketAnomaly,
    /// Strategy breakthrough
    StrategyBreakthrough,
    /// Risk event (near miss or violation)
    RiskEvent,
    /// Regime change detection
    RegimeChange,
    /// Correlation breakdown
    CorrelationBreakdown,
    /// Volatility spike
    VolatilitySpike,
    /// Liquidity event
    LiquidityEvent,
    /// Time-based (periodic consolidation)
    Periodic,
}

impl RippleType {
    /// Get base priority for this ripple type
    pub fn base_priority(&self) -> f64 {
        match self {
            RippleType::LargeLoss => 1.0,    // Highest priority - learn from losses
            RippleType::RiskEvent => 0.95,   // Near misses are critical
            RippleType::LargeProfit => 0.85, // Success patterns
            RippleType::RegimeChange => 0.80, // Market structure changes
            RippleType::MarketAnomaly => 0.75, // Unusual events
            RippleType::CorrelationBreakdown => 0.70,
            RippleType::StrategyBreakthrough => 0.65,
            RippleType::NovelPattern => 0.60,
            RippleType::VolatilitySpike => 0.55,
            RippleType::LiquidityEvent => 0.50,
            RippleType::Periodic => 0.30, // Routine consolidation
        }
    }

    /// Get description
    pub fn description(&self) -> &'static str {
        match self {
            RippleType::LargeProfit => "Significant profitable outcome",
            RippleType::LargeLoss => "Significant losing outcome",
            RippleType::NovelPattern => "Novel or rare pattern detected",
            RippleType::MarketAnomaly => "Anomalous market behavior",
            RippleType::StrategyBreakthrough => "Strategy performance breakthrough",
            RippleType::RiskEvent => "Risk limit event or near-miss",
            RippleType::RegimeChange => "Market regime transition",
            RippleType::CorrelationBreakdown => "Correlation structure breakdown",
            RippleType::VolatilitySpike => "Volatility spike detected",
            RippleType::LiquidityEvent => "Liquidity event (gap, slippage)",
            RippleType::Periodic => "Scheduled consolidation trigger",
        }
    }

    /// Check if this type requires immediate processing
    pub fn requires_immediate_processing(&self) -> bool {
        matches!(
            self,
            RippleType::LargeLoss
                | RippleType::RiskEvent
                | RippleType::MarketAnomaly
                | RippleType::RegimeChange
        )
    }
}

/// Significance level of the ripple
#[derive(Debug, Clone, Copy, PartialEq, Eq, PartialOrd, Ord)]
pub enum SignificanceLevel {
    /// Low significance - queue for batch processing
    Low,
    /// Moderate significance - process in next cycle
    Moderate,
    /// High significance - prioritize processing
    High,
    /// Critical significance - process immediately
    Critical,
}

impl Default for SignificanceLevel {
    fn default() -> Self {
        SignificanceLevel::Low
    }
}

impl SignificanceLevel {
    /// Get priority multiplier
    pub fn multiplier(&self) -> f64 {
        match self {
            SignificanceLevel::Low => 1.0,
            SignificanceLevel::Moderate => 1.5,
            SignificanceLevel::High => 2.0,
            SignificanceLevel::Critical => 3.0,
        }
    }
}

/// A detected ripple event
#[derive(Debug, Clone)]
pub struct RippleEvent {
    /// Unique event ID
    pub id: u64,
    /// Timestamp of detection
    pub timestamp: u64,
    /// Type of ripple
    pub ripple_type: RippleType,
    /// Significance level
    pub significance: SignificanceLevel,
    /// Priority score (0-1)
    pub priority: f64,
    /// Associated symbol (if applicable)
    pub symbol: Option<String>,
    /// Associated trade/episode ID (if applicable)
    pub episode_id: Option<String>,
    /// Magnitude of the event (context-dependent)
    pub magnitude: f64,
    /// Associated PnL impact (if applicable)
    pub pnl_impact: Option<f64>,
    /// Metadata/context
    pub metadata: HashMap<String, String>,
    /// Whether this event has been processed
    pub processed: bool,
    /// Number of times replayed
    pub replay_count: usize,
    /// Related events (by ID)
    pub related_events: Vec<u64>,
}

impl RippleEvent {
    pub fn new(id: u64, timestamp: u64, ripple_type: RippleType, magnitude: f64) -> Self {
        let base_priority = ripple_type.base_priority();
        let significance = Self::calculate_significance(ripple_type, magnitude);
        let priority = base_priority * significance.multiplier() * magnitude.abs().min(1.0);

        Self {
            id,
            timestamp,
            ripple_type,
            significance,
            priority: priority.min(1.0),
            symbol: None,
            episode_id: None,
            magnitude,
            pnl_impact: None,
            metadata: HashMap::new(),
            processed: false,
            replay_count: 0,
            related_events: Vec::new(),
        }
    }

    /// Calculate significance based on type and magnitude
    fn calculate_significance(ripple_type: RippleType, magnitude: f64) -> SignificanceLevel {
        let abs_magnitude = magnitude.abs();

        match ripple_type {
            RippleType::LargeLoss | RippleType::RiskEvent => {
                if abs_magnitude > 0.8 {
                    SignificanceLevel::Critical
                } else if abs_magnitude > 0.5 {
                    SignificanceLevel::High
                } else {
                    SignificanceLevel::Moderate
                }
            }
            RippleType::RegimeChange | RippleType::MarketAnomaly => {
                if abs_magnitude > 0.7 {
                    SignificanceLevel::High
                } else {
                    SignificanceLevel::Moderate
                }
            }
            _ => {
                if abs_magnitude > 0.9 {
                    SignificanceLevel::High
                } else if abs_magnitude > 0.5 {
                    SignificanceLevel::Moderate
                } else {
                    SignificanceLevel::Low
                }
            }
        }
    }

    /// Add metadata
    pub fn with_metadata(mut self, key: &str, value: &str) -> Self {
        self.metadata.insert(key.to_string(), value.to_string());
        self
    }

    /// Set symbol
    pub fn with_symbol(mut self, symbol: &str) -> Self {
        self.symbol = Some(symbol.to_string());
        self
    }

    /// Set episode ID
    pub fn with_episode(mut self, episode_id: &str) -> Self {
        self.episode_id = Some(episode_id.to_string());
        self
    }

    /// Set PnL impact
    pub fn with_pnl(mut self, pnl: f64) -> Self {
        self.pnl_impact = Some(pnl);
        self
    }

    /// Mark as processed
    pub fn mark_processed(&mut self) {
        self.processed = true;
        self.replay_count += 1;
    }

    /// Check if requires immediate processing
    pub fn requires_immediate(&self) -> bool {
        self.ripple_type.requires_immediate_processing()
            || self.significance >= SignificanceLevel::Critical
    }
}

/// Configuration for ripple detection
#[derive(Debug, Clone)]
pub struct RippleDetectionConfig {
    /// PnL threshold for large profit (as multiple of average)
    pub profit_threshold: f64,
    /// PnL threshold for large loss (as multiple of average)
    pub loss_threshold: f64,
    /// Volatility spike threshold (multiple of average)
    pub volatility_spike_threshold: f64,
    /// Novelty threshold for pattern detection
    pub novelty_threshold: f64,
    /// Correlation breakdown threshold
    pub correlation_threshold: f64,
    /// Minimum time between similar ripples (seconds)
    pub dedup_window_secs: u64,
    /// Maximum ripple queue size
    pub max_queue_size: usize,
    /// Enable automatic periodic ripples
    pub enable_periodic: bool,
    /// Periodic ripple interval (seconds)
    pub periodic_interval_secs: u64,
    /// Priority decay rate (per second)
    pub priority_decay_rate: f64,
    /// Maximum replay count before discarding
    pub max_replay_count: usize,
}

impl Default for RippleDetectionConfig {
    fn default() -> Self {
        Self {
            profit_threshold: 2.0,
            loss_threshold: 2.0,
            volatility_spike_threshold: 2.5,
            novelty_threshold: 0.7,
            correlation_threshold: 0.5,
            dedup_window_secs: 60,
            max_queue_size: 1000,
            enable_periodic: true,
            periodic_interval_secs: 3600, // 1 hour
            priority_decay_rate: 0.001,
            max_replay_count: 10,
        }
    }
}

/// Running statistics for detection thresholds
#[derive(Debug, Clone)]
pub struct DetectionStats {
    /// Average PnL
    pub avg_pnl: f64,
    /// PnL standard deviation
    pub pnl_std: f64,
    /// Average volatility
    pub avg_volatility: f64,
    /// Volatility standard deviation
    pub volatility_std: f64,
    /// Sample count
    pub sample_count: usize,
    /// EWMA lambda for updating stats
    ewma_lambda: f64,
}

impl Default for DetectionStats {
    fn default() -> Self {
        Self {
            avg_pnl: 0.0,
            pnl_std: 100.0, // Default assumption
            avg_volatility: 0.15,
            volatility_std: 0.05,
            sample_count: 0,
            ewma_lambda: 0.95,
        }
    }
}

impl DetectionStats {
    /// Update statistics with new PnL observation
    pub fn update_pnl(&mut self, pnl: f64) {
        self.sample_count += 1;

        if self.sample_count == 1 {
            self.avg_pnl = pnl;
            self.pnl_std = pnl.abs().max(1.0);
        } else {
            let lambda = self.ewma_lambda;
            self.avg_pnl = lambda * self.avg_pnl + (1.0 - lambda) * pnl;

            // Update variance estimate
            let deviation = (pnl - self.avg_pnl).abs();
            self.pnl_std = lambda * self.pnl_std + (1.0 - lambda) * deviation;
        }
    }

    /// Update statistics with new volatility observation
    pub fn update_volatility(&mut self, vol: f64) {
        let lambda = self.ewma_lambda;
        self.avg_volatility = lambda * self.avg_volatility + (1.0 - lambda) * vol;

        let deviation = (vol - self.avg_volatility).abs();
        self.volatility_std = lambda * self.volatility_std + (1.0 - lambda) * deviation;
    }

    /// Check if PnL is significant
    pub fn is_significant_pnl(&self, pnl: f64, threshold: f64) -> bool {
        if self.pnl_std <= 0.0 {
            return pnl.abs() > 0.0;
        }
        (pnl - self.avg_pnl).abs() / self.pnl_std > threshold
    }

    /// Check if volatility is a spike
    pub fn is_volatility_spike(&self, vol: f64, threshold: f64) -> bool {
        if self.volatility_std <= 0.0 {
            return vol > self.avg_volatility * 1.5;
        }
        (vol - self.avg_volatility) / self.volatility_std > threshold
    }

    /// Get PnL z-score
    pub fn pnl_zscore(&self, pnl: f64) -> f64 {
        if self.pnl_std <= 0.0 {
            return 0.0;
        }
        (pnl - self.avg_pnl) / self.pnl_std
    }
}

/// Ripple detection result
#[derive(Debug, Clone)]
pub struct DetectionResult {
    /// Detected ripples
    pub ripples: Vec<RippleEvent>,
    /// Total ripples in queue
    pub queue_size: usize,
    /// High priority count
    pub high_priority_count: usize,
    /// Events requiring immediate processing
    pub immediate_count: usize,
}

/// Detect high-priority replay events
pub struct RippleDetection {
    /// Configuration
    config: RippleDetectionConfig,
    /// Running statistics
    stats: DetectionStats,
    /// Ripple event queue
    ripple_queue: VecDeque<RippleEvent>,
    /// Event ID counter
    event_counter: u64,
    /// Last periodic ripple time
    last_periodic_time: u64,
    /// Recent ripple types for deduplication
    recent_ripples: HashMap<RippleType, u64>,
    /// Symbol-specific recent events
    symbol_recent: HashMap<String, Vec<(RippleType, u64)>>,
    /// Total ripples detected
    total_detected: usize,
    /// Total ripples processed
    total_processed: usize,
}

impl Default for RippleDetection {
    fn default() -> Self {
        Self::new()
    }
}

impl RippleDetection {
    /// Create a new instance
    pub fn new() -> Self {
        Self {
            config: RippleDetectionConfig::default(),
            stats: DetectionStats::default(),
            ripple_queue: VecDeque::new(),
            event_counter: 0,
            last_periodic_time: 0,
            recent_ripples: HashMap::new(),
            symbol_recent: HashMap::new(),
            total_detected: 0,
            total_processed: 0,
        }
    }

    /// Create with custom configuration
    pub fn with_config(config: RippleDetectionConfig) -> Self {
        Self {
            config,
            stats: DetectionStats::default(),
            ripple_queue: VecDeque::new(),
            event_counter: 0,
            last_periodic_time: 0,
            recent_ripples: HashMap::new(),
            symbol_recent: HashMap::new(),
            total_detected: 0,
            total_processed: 0,
        }
    }

    /// Generate next event ID
    fn next_id(&mut self) -> u64 {
        self.event_counter += 1;
        self.event_counter
    }

    /// Check if ripple should be deduplicated
    fn should_dedupe(&self, ripple_type: RippleType, timestamp: u64, symbol: Option<&str>) -> bool {
        // Check global recent
        if let Some(&last_time) = self.recent_ripples.get(&ripple_type) {
            if timestamp - last_time < self.config.dedup_window_secs {
                return true;
            }
        }

        // Check symbol-specific
        if let Some(sym) = symbol {
            if let Some(recent) = self.symbol_recent.get(sym) {
                for (rt, ts) in recent {
                    if *rt == ripple_type && timestamp - ts < self.config.dedup_window_secs {
                        return true;
                    }
                }
            }
        }

        false
    }

    /// Record ripple for deduplication
    fn record_for_dedupe(&mut self, ripple_type: RippleType, timestamp: u64, symbol: Option<&str>) {
        self.recent_ripples.insert(ripple_type, timestamp);

        if let Some(sym) = symbol {
            let recent = self
                .symbol_recent
                .entry(sym.to_string())
                .or_insert_with(Vec::new);
            recent.push((ripple_type, timestamp));

            // Keep only recent entries
            recent.retain(|(_, ts)| timestamp - ts < self.config.dedup_window_secs * 2);
        }
    }

    /// Detect ripple from PnL event
    pub fn detect_pnl_event(
        &mut self,
        pnl: f64,
        symbol: Option<&str>,
        episode_id: Option<&str>,
        timestamp: u64,
    ) -> Option<RippleEvent> {
        // Update stats
        self.stats.update_pnl(pnl);

        // Check significance
        let zscore = self.stats.pnl_zscore(pnl);

        let ripple_type = if pnl > 0.0 && zscore > self.config.profit_threshold {
            RippleType::LargeProfit
        } else if pnl < 0.0 && zscore.abs() > self.config.loss_threshold {
            RippleType::LargeLoss
        } else {
            return None;
        };

        // Check deduplication
        if self.should_dedupe(ripple_type, timestamp, symbol) {
            return None;
        }

        // Create ripple
        let magnitude = (zscore.abs() / 5.0).min(1.0); // Normalize z-score to 0-1
        let id = self.next_id();
        let mut ripple = RippleEvent::new(id, timestamp, ripple_type, magnitude).with_pnl(pnl);

        if let Some(sym) = symbol {
            ripple = ripple.with_symbol(sym);
        }
        if let Some(ep_id) = episode_id {
            ripple = ripple.with_episode(ep_id);
        }

        ripple = ripple.with_metadata("zscore", &format!("{:.2}", zscore));

        self.add_ripple(ripple.clone(), symbol, timestamp);
        Some(ripple)
    }

    /// Detect ripple from volatility event
    pub fn detect_volatility_event(
        &mut self,
        volatility: f64,
        symbol: Option<&str>,
        timestamp: u64,
    ) -> Option<RippleEvent> {
        self.stats.update_volatility(volatility);

        if !self
            .stats
            .is_volatility_spike(volatility, self.config.volatility_spike_threshold)
        {
            return None;
        }

        if self.should_dedupe(RippleType::VolatilitySpike, timestamp, symbol) {
            return None;
        }

        let magnitude =
            ((volatility - self.stats.avg_volatility) / self.stats.volatility_std / 5.0)
                .abs()
                .min(1.0);

        let id = self.next_id();
        let mut ripple = RippleEvent::new(id, timestamp, RippleType::VolatilitySpike, magnitude);

        if let Some(sym) = symbol {
            ripple = ripple.with_symbol(sym);
        }

        ripple = ripple.with_metadata("volatility", &format!("{:.4}", volatility));

        self.add_ripple(ripple.clone(), symbol, timestamp);
        Some(ripple)
    }

    /// Detect ripple from regime change
    pub fn detect_regime_change(
        &mut self,
        from_regime: &str,
        to_regime: &str,
        confidence: f64,
        timestamp: u64,
    ) -> Option<RippleEvent> {
        if self.should_dedupe(RippleType::RegimeChange, timestamp, None) {
            return None;
        }

        let id = self.next_id();
        let ripple = RippleEvent::new(id, timestamp, RippleType::RegimeChange, confidence)
            .with_metadata("from_regime", from_regime)
            .with_metadata("to_regime", to_regime);

        self.add_ripple(ripple.clone(), None, timestamp);
        Some(ripple)
    }

    /// Detect ripple from risk event
    pub fn detect_risk_event(
        &mut self,
        risk_type: &str,
        severity: f64,
        symbol: Option<&str>,
        timestamp: u64,
    ) -> Option<RippleEvent> {
        if self.should_dedupe(RippleType::RiskEvent, timestamp, symbol) {
            return None;
        }

        let id = self.next_id();
        let mut ripple = RippleEvent::new(id, timestamp, RippleType::RiskEvent, severity)
            .with_metadata("risk_type", risk_type);

        if let Some(sym) = symbol {
            ripple = ripple.with_symbol(sym);
        }

        self.add_ripple(ripple.clone(), symbol, timestamp);
        Some(ripple)
    }

    /// Detect ripple from novel pattern
    pub fn detect_novel_pattern(
        &mut self,
        pattern_name: &str,
        novelty_score: f64,
        symbol: Option<&str>,
        timestamp: u64,
    ) -> Option<RippleEvent> {
        if novelty_score < self.config.novelty_threshold {
            return None;
        }

        if self.should_dedupe(RippleType::NovelPattern, timestamp, symbol) {
            return None;
        }

        let id = self.next_id();
        let mut ripple = RippleEvent::new(id, timestamp, RippleType::NovelPattern, novelty_score)
            .with_metadata("pattern", pattern_name);

        if let Some(sym) = symbol {
            ripple = ripple.with_symbol(sym);
        }

        self.add_ripple(ripple.clone(), symbol, timestamp);
        Some(ripple)
    }

    /// Detect ripple from correlation breakdown
    pub fn detect_correlation_breakdown(
        &mut self,
        pair: (&str, &str),
        expected_corr: f64,
        actual_corr: f64,
        timestamp: u64,
    ) -> Option<RippleEvent> {
        let deviation = (expected_corr - actual_corr).abs();
        if deviation < self.config.correlation_threshold {
            return None;
        }

        if self.should_dedupe(RippleType::CorrelationBreakdown, timestamp, None) {
            return None;
        }

        let magnitude = (deviation / 1.0).min(1.0);
        let id = self.next_id();
        let ripple = RippleEvent::new(id, timestamp, RippleType::CorrelationBreakdown, magnitude)
            .with_metadata("pair", &format!("{}:{}", pair.0, pair.1))
            .with_metadata("expected", &format!("{:.3}", expected_corr))
            .with_metadata("actual", &format!("{:.3}", actual_corr));

        self.add_ripple(ripple.clone(), None, timestamp);
        Some(ripple)
    }

    /// Detect ripple from market anomaly
    pub fn detect_market_anomaly(
        &mut self,
        anomaly_type: &str,
        severity: f64,
        symbol: Option<&str>,
        timestamp: u64,
    ) -> Option<RippleEvent> {
        if self.should_dedupe(RippleType::MarketAnomaly, timestamp, symbol) {
            return None;
        }

        let id = self.next_id();
        let mut ripple = RippleEvent::new(id, timestamp, RippleType::MarketAnomaly, severity)
            .with_metadata("anomaly_type", anomaly_type);

        if let Some(sym) = symbol {
            ripple = ripple.with_symbol(sym);
        }

        self.add_ripple(ripple.clone(), symbol, timestamp);
        Some(ripple)
    }

    /// Check and generate periodic ripple
    pub fn check_periodic(&mut self, timestamp: u64) -> Option<RippleEvent> {
        if !self.config.enable_periodic {
            return None;
        }

        if timestamp - self.last_periodic_time >= self.config.periodic_interval_secs {
            self.last_periodic_time = timestamp;

            let id = self.next_id();
            let ripple = RippleEvent::new(id, timestamp, RippleType::Periodic, 0.3);

            self.add_ripple(ripple.clone(), None, timestamp);
            return Some(ripple);
        }

        None
    }

    /// Add ripple to queue
    fn add_ripple(&mut self, ripple: RippleEvent, symbol: Option<&str>, timestamp: u64) {
        self.record_for_dedupe(ripple.ripple_type, timestamp, symbol);
        self.ripple_queue.push_back(ripple);
        self.total_detected += 1;

        // Enforce queue size limit
        while self.ripple_queue.len() > self.config.max_queue_size {
            self.ripple_queue.pop_front();
        }
    }

    /// Get next ripple for processing (highest priority)
    pub fn pop_highest_priority(&mut self) -> Option<RippleEvent> {
        if self.ripple_queue.is_empty() {
            return None;
        }

        // Find highest priority
        let mut best_idx = 0;
        let mut best_priority = 0.0;

        for (idx, ripple) in self.ripple_queue.iter().enumerate() {
            if !ripple.processed && ripple.priority > best_priority {
                best_priority = ripple.priority;
                best_idx = idx;
            }
        }

        self.ripple_queue.remove(best_idx).map(|mut r| {
            r.mark_processed();
            self.total_processed += 1;
            r
        })
    }

    /// Get all ripples requiring immediate processing
    pub fn get_immediate(&mut self) -> Vec<RippleEvent> {
        let mut immediate = Vec::new();

        // Use drain_filter-like behavior
        let mut i = 0;
        while i < self.ripple_queue.len() {
            if self.ripple_queue[i].requires_immediate() && !self.ripple_queue[i].processed {
                if let Some(mut ripple) = self.ripple_queue.remove(i) {
                    ripple.mark_processed();
                    self.total_processed += 1;
                    immediate.push(ripple);
                }
            } else {
                i += 1;
            }
        }

        // Sort by priority
        immediate.sort_by(|a, b| b.priority.partial_cmp(&a.priority).unwrap());

        immediate
    }

    /// Get batch of ripples for processing
    pub fn get_batch(&mut self, batch_size: usize) -> Vec<RippleEvent> {
        let mut batch = Vec::with_capacity(batch_size);

        // Collect indices of unprocessed ripples
        let mut candidates: Vec<(usize, f64)> = self
            .ripple_queue
            .iter()
            .enumerate()
            .filter(|(_, r)| !r.processed)
            .map(|(idx, r)| (idx, r.priority))
            .collect();

        // Sort by priority (descending)
        candidates.sort_by(|a, b| b.1.partial_cmp(&a.1).unwrap());

        // Take top batch_size
        let indices_to_remove: Vec<usize> = candidates
            .iter()
            .take(batch_size)
            .map(|(idx, _)| *idx)
            .collect();

        // Remove in reverse order to maintain indices
        let mut sorted_indices = indices_to_remove.clone();
        sorted_indices.sort_by(|a, b| b.cmp(a));

        for idx in sorted_indices {
            if let Some(mut ripple) = self.ripple_queue.remove(idx) {
                ripple.mark_processed();
                self.total_processed += 1;
                batch.push(ripple);
            }
        }

        // Sort batch by priority
        batch.sort_by(|a, b| b.priority.partial_cmp(&a.priority).unwrap());

        batch
    }

    /// Apply priority decay to all queued ripples
    pub fn apply_decay(&mut self, elapsed_secs: u64) {
        let decay = 1.0 - (self.config.priority_decay_rate * elapsed_secs as f64);
        let decay = decay.max(0.1);

        for ripple in &mut self.ripple_queue {
            ripple.priority *= decay;
        }
    }

    /// Remove ripples exceeding max replay count
    pub fn cleanup_exhausted(&mut self) {
        self.ripple_queue
            .retain(|r| r.replay_count < self.config.max_replay_count);
    }

    /// Get queue statistics
    pub fn get_stats(&self) -> DetectionResult {
        let high_priority_count = self
            .ripple_queue
            .iter()
            .filter(|r| r.significance >= SignificanceLevel::High && !r.processed)
            .count();

        let immediate_count = self
            .ripple_queue
            .iter()
            .filter(|r| r.requires_immediate() && !r.processed)
            .count();

        DetectionResult {
            ripples: self.ripple_queue.iter().cloned().collect(),
            queue_size: self.ripple_queue.len(),
            high_priority_count,
            immediate_count,
        }
    }

    /// Get queue size
    pub fn queue_size(&self) -> usize {
        self.ripple_queue.len()
    }

    /// Get total detected count
    pub fn total_detected(&self) -> usize {
        self.total_detected
    }

    /// Get total processed count
    pub fn total_processed(&self) -> usize {
        self.total_processed
    }

    /// Check if queue is empty
    pub fn is_empty(&self) -> bool {
        self.ripple_queue.is_empty()
    }

    /// Get running statistics
    pub fn detection_stats(&self) -> &DetectionStats {
        &self.stats
    }

    /// Clear the queue
    pub fn clear(&mut self) {
        self.ripple_queue.clear();
    }

    /// Main processing function
    pub fn process(&self) -> Result<()> {
        // Processing is done on-demand via detection methods
        Ok(())
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_basic() {
        let instance = RippleDetection::new();
        assert!(instance.process().is_ok());
        assert!(instance.is_empty());
    }

    #[test]
    fn test_pnl_ripple_detection() {
        let mut detector = RippleDetection::new();
        detector.config.profit_threshold = 2.0;
        detector.config.loss_threshold = 2.0;

        // Build up some baseline statistics
        for i in 0..20 {
            let pnl = if i % 2 == 0 { 100.0 } else { -100.0 };
            detector.detect_pnl_event(pnl, None, None, i);
        }

        // Large profit should trigger
        let ripple = detector.detect_pnl_event(500.0, Some("AAPL"), None, 100);
        assert!(ripple.is_some());

        let r = ripple.unwrap();
        assert_eq!(r.ripple_type, RippleType::LargeProfit);
        assert!(r.pnl_impact.is_some());
    }

    #[test]
    fn test_loss_ripple_detection() {
        let mut detector = RippleDetection::new();

        // Build baseline
        for i in 0..20 {
            detector.detect_pnl_event(50.0, None, None, i);
        }

        // Large loss
        let ripple = detector.detect_pnl_event(-300.0, Some("GOOG"), None, 100);
        assert!(ripple.is_some());

        let r = ripple.unwrap();
        assert_eq!(r.ripple_type, RippleType::LargeLoss);
        assert!(r.significance >= SignificanceLevel::Moderate);
    }

    #[test]
    fn test_volatility_spike_detection() {
        let mut detector = RippleDetection::new();

        // Normal volatility
        for _i in 0..10 {
            detector.stats.update_volatility(0.15);
        }

        // Spike
        let ripple = detector.detect_volatility_event(0.45, Some("SPY"), 100);
        assert!(ripple.is_some());

        let r = ripple.unwrap();
        assert_eq!(r.ripple_type, RippleType::VolatilitySpike);
    }

    #[test]
    fn test_regime_change_detection() {
        let mut detector = RippleDetection::new();

        let ripple = detector.detect_regime_change("Bullish", "Bearish", 0.8, 100);
        assert!(ripple.is_some());

        let r = ripple.unwrap();
        assert_eq!(r.ripple_type, RippleType::RegimeChange);
        assert!(r.requires_immediate());
    }

    #[test]
    fn test_risk_event_detection() {
        let mut detector = RippleDetection::new();

        let ripple = detector.detect_risk_event("margin_warning", 0.9, Some("TSLA"), 100);
        assert!(ripple.is_some());

        let r = ripple.unwrap();
        assert_eq!(r.ripple_type, RippleType::RiskEvent);
        assert!(r.requires_immediate());
    }

    #[test]
    fn test_novel_pattern_detection() {
        let mut detector = RippleDetection::new();
        detector.config.novelty_threshold = 0.5;

        // Below threshold - no ripple
        let ripple = detector.detect_novel_pattern("head_shoulders", 0.3, None, 100);
        assert!(ripple.is_none());

        // Above threshold
        let ripple = detector.detect_novel_pattern("rare_divergence", 0.8, Some("QQQ"), 200);
        assert!(ripple.is_some());
    }

    #[test]
    fn test_correlation_breakdown_detection() {
        let mut detector = RippleDetection::new();
        detector.config.correlation_threshold = 0.3;

        let ripple = detector.detect_correlation_breakdown(("SPY", "QQQ"), 0.9, 0.3, 100);
        assert!(ripple.is_some());

        let r = ripple.unwrap();
        assert_eq!(r.ripple_type, RippleType::CorrelationBreakdown);
    }

    #[test]
    fn test_market_anomaly_detection() {
        let mut detector = RippleDetection::new();

        let ripple = detector.detect_market_anomaly("flash_crash", 0.95, Some("ES"), 100);
        assert!(ripple.is_some());

        let r = ripple.unwrap();
        assert!(r.requires_immediate());
    }

    #[test]
    fn test_periodic_ripple() {
        let mut detector = RippleDetection::new();
        detector.config.enable_periodic = true;
        detector.config.periodic_interval_secs = 100;

        // First check - should trigger
        let ripple = detector.check_periodic(100);
        assert!(ripple.is_some());

        // Too soon - should not trigger
        let ripple = detector.check_periodic(150);
        assert!(ripple.is_none());

        // After interval - should trigger
        let ripple = detector.check_periodic(250);
        assert!(ripple.is_some());
    }

    #[test]
    fn test_deduplication() {
        let mut detector = RippleDetection::new();
        detector.config.dedup_window_secs = 60;

        // First event
        let r1 = detector.detect_risk_event("test", 0.8, Some("AAPL"), 100);
        assert!(r1.is_some());

        // Duplicate within window
        let r2 = detector.detect_risk_event("test", 0.8, Some("AAPL"), 120);
        assert!(r2.is_none());

        // After window
        let r3 = detector.detect_risk_event("test", 0.8, Some("AAPL"), 200);
        assert!(r3.is_some());
    }

    #[test]
    fn test_priority_ordering() {
        let mut detector = RippleDetection::new();

        // Add various ripples
        detector.detect_novel_pattern("pattern1", 0.8, None, 100);
        detector.detect_risk_event("critical", 0.95, None, 101);
        detector.detect_market_anomaly("anomaly", 0.7, None, 102);

        // Pop should return highest priority first
        let first = detector.pop_highest_priority();
        assert!(first.is_some());

        // Risk events have higher base priority
        assert!(matches!(
            first.unwrap().ripple_type,
            RippleType::RiskEvent | RippleType::MarketAnomaly
        ));
    }

    #[test]
    fn test_get_immediate() {
        let mut detector = RippleDetection::new();

        // Non-immediate
        detector.detect_novel_pattern("pattern", 0.8, None, 100);

        // Immediate events
        detector.detect_risk_event("margin", 0.9, None, 101);
        detector.detect_regime_change("bull", "bear", 0.8, 102);

        let immediate = detector.get_immediate();
        assert!(immediate.len() >= 2);

        // All should require immediate processing
        for r in &immediate {
            assert!(
                r.ripple_type.requires_immediate_processing()
                    || r.significance >= SignificanceLevel::Critical
            );
        }
    }

    #[test]
    fn test_get_batch() {
        let mut detector = RippleDetection::new();

        // Add multiple ripples
        for i in 0..10 {
            detector.detect_market_anomaly(
                &format!("anomaly_{}", i),
                0.5 + i as f64 * 0.05,
                None,
                i as u64 * 100,
            );
        }

        let batch = detector.get_batch(3);
        assert_eq!(batch.len(), 3);

        // Should be sorted by priority (descending)
        for i in 1..batch.len() {
            assert!(batch[i - 1].priority >= batch[i].priority);
        }
    }

    #[test]
    fn test_priority_decay() {
        let mut detector = RippleDetection::new();
        detector.config.priority_decay_rate = 0.01;

        detector.detect_risk_event("test", 0.8, None, 100);

        let initial_priority = detector.ripple_queue.front().unwrap().priority;

        detector.apply_decay(10);

        let decayed_priority = detector.ripple_queue.front().unwrap().priority;
        assert!(decayed_priority < initial_priority);
    }

    #[test]
    fn test_queue_size_limit() {
        let mut detector = RippleDetection::new();
        detector.config.max_queue_size = 5;
        detector.config.dedup_window_secs = 0; // Disable dedup for test

        for i in 0..10 {
            detector.detect_market_anomaly(&format!("anomaly_{}", i), 0.5, None, i as u64 * 100);
        }

        assert!(detector.queue_size() <= 5);
    }

    #[test]
    fn test_ripple_type_properties() {
        assert!(RippleType::LargeLoss.base_priority() > RippleType::NovelPattern.base_priority());
        assert!(RippleType::RiskEvent.requires_immediate_processing());
        assert!(!RippleType::Periodic.requires_immediate_processing());
    }

    #[test]
    fn test_significance_levels() {
        assert!(SignificanceLevel::Critical > SignificanceLevel::High);
        assert!(SignificanceLevel::High > SignificanceLevel::Moderate);
        assert!(SignificanceLevel::Moderate > SignificanceLevel::Low);

        assert!(SignificanceLevel::Critical.multiplier() > SignificanceLevel::Low.multiplier());
    }

    #[test]
    fn test_detection_stats() {
        let mut stats = DetectionStats::default();

        // Add observations
        for i in 0..10 {
            stats.update_pnl(100.0 * (i % 3 - 1) as f64);
        }

        assert!(stats.sample_count == 10);
        assert!(stats.pnl_std > 0.0);
    }

    #[test]
    fn test_ripple_event_builders() {
        let ripple = RippleEvent::new(1, 100, RippleType::LargeProfit, 0.8)
            .with_symbol("AAPL")
            .with_episode("ep_123")
            .with_pnl(1000.0)
            .with_metadata("strategy", "momentum");

        assert_eq!(ripple.symbol, Some("AAPL".to_string()));
        assert_eq!(ripple.episode_id, Some("ep_123".to_string()));
        assert_eq!(ripple.pnl_impact, Some(1000.0));
        assert_eq!(
            ripple.metadata.get("strategy"),
            Some(&"momentum".to_string())
        );
    }

    #[test]
    fn test_stats_report() {
        let mut detector = RippleDetection::new();

        detector.detect_risk_event("test1", 0.8, None, 100);
        detector.detect_risk_event("test2", 0.9, None, 200);
        detector.detect_novel_pattern("pattern", 0.75, None, 300);

        let stats = detector.get_stats();
        assert_eq!(stats.queue_size, 3);
        assert!(stats.high_priority_count > 0);
    }
}
