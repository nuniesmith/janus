//! Significant market events detection and storage
//!
//! Part of the Hippocampus region
//! Component: episodes
//!
//! This module detects, classifies, and stores significant market events
//! such as flash crashes, circuit breakers, large moves, and unusual volume.

use crate::common::{Error, Result};
use std::collections::{HashMap, VecDeque};

/// Type of market event
#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash)]
pub enum EventType {
    /// Flash crash - rapid price decline
    FlashCrash,
    /// Flash rally - rapid price increase
    FlashRally,
    /// Circuit breaker triggered
    CircuitBreaker,
    /// Unusually high volume
    VolumeSpike,
    /// Large single trade
    LargeTrade,
    /// Gap up at open
    GapUp,
    /// Gap down at open
    GapDown,
    /// Volatility spike
    VolatilitySpike,
    /// Liquidity crisis
    LiquidityCrisis,
    /// News-driven move
    NewsEvent,
    /// Exchange outage
    ExchangeOutage,
    /// New all-time high
    AllTimeHigh,
    /// New all-time low
    AllTimeLow,
    /// Correlation breakdown
    CorrelationBreak,
    /// Regime change detected
    RegimeChange,
    /// Custom/Other event
    Custom,
}

impl EventType {
    /// Get the default severity for this event type
    pub fn default_severity(&self) -> EventSeverity {
        match self {
            EventType::FlashCrash | EventType::CircuitBreaker | EventType::LiquidityCrisis => {
                EventSeverity::Critical
            }
            EventType::FlashRally
            | EventType::ExchangeOutage
            | EventType::VolatilitySpike
            | EventType::RegimeChange => EventSeverity::High,
            EventType::VolumeSpike
            | EventType::LargeTrade
            | EventType::GapUp
            | EventType::GapDown
            | EventType::CorrelationBreak => EventSeverity::Medium,
            EventType::AllTimeHigh
            | EventType::AllTimeLow
            | EventType::NewsEvent
            | EventType::Custom => EventSeverity::Low,
        }
    }
}

/// Severity level of an event
#[derive(Debug, Clone, Copy, PartialEq, Eq, PartialOrd, Ord, Hash)]
pub enum EventSeverity {
    /// Informational - no action needed
    Info,
    /// Low impact event
    Low,
    /// Medium impact event
    Medium,
    /// High impact event
    High,
    /// Critical - immediate attention required
    Critical,
}

impl EventSeverity {
    /// Get numeric value for severity (for calculations)
    pub fn value(&self) -> f64 {
        match self {
            EventSeverity::Info => 0.0,
            EventSeverity::Low => 0.25,
            EventSeverity::Medium => 0.5,
            EventSeverity::High => 0.75,
            EventSeverity::Critical => 1.0,
        }
    }
}

/// A significant market event
#[derive(Debug, Clone)]
pub struct MarketEvent {
    /// Unique event ID
    pub id: u64,
    /// Event type
    pub event_type: EventType,
    /// Severity level
    pub severity: EventSeverity,
    /// Symbol(s) affected
    pub symbols: Vec<String>,
    /// Timestamp when event started (epoch millis)
    pub start_time: u64,
    /// Timestamp when event ended (None if ongoing)
    pub end_time: Option<u64>,
    /// Price at event start
    pub start_price: f64,
    /// Price at event end (or current if ongoing)
    pub end_price: f64,
    /// Price change percentage
    pub price_change_pct: f64,
    /// Volume during event
    pub volume: f64,
    /// Average volume for comparison
    pub avg_volume: f64,
    /// Volume ratio (volume / avg_volume)
    pub volume_ratio: f64,
    /// Description of the event
    pub description: String,
    /// Additional metadata
    pub metadata: HashMap<String, String>,
    /// Was this event predicted/expected?
    pub was_predicted: bool,
    /// Actions taken in response
    pub actions_taken: Vec<String>,
    /// Impact assessment score (0.0 - 1.0)
    pub impact_score: f64,
}

impl MarketEvent {
    /// Create a new market event
    pub fn new(
        id: u64,
        event_type: EventType,
        symbols: Vec<String>,
        start_time: u64,
        start_price: f64,
    ) -> Self {
        Self {
            id,
            event_type,
            severity: event_type.default_severity(),
            symbols,
            start_time,
            end_time: None,
            start_price,
            end_price: start_price,
            price_change_pct: 0.0,
            volume: 0.0,
            avg_volume: 0.0,
            volume_ratio: 1.0,
            description: String::new(),
            metadata: HashMap::new(),
            was_predicted: false,
            actions_taken: Vec::new(),
            impact_score: 0.0,
        }
    }

    /// Update event with new price
    pub fn update_price(&mut self, price: f64) {
        self.end_price = price;
        if self.start_price != 0.0 {
            self.price_change_pct = (price - self.start_price) / self.start_price * 100.0;
        }
    }

    /// Close the event
    pub fn close(&mut self, end_time: u64, end_price: f64) {
        self.end_time = Some(end_time);
        self.update_price(end_price);
    }

    /// Duration of the event in milliseconds
    pub fn duration_ms(&self) -> Option<u64> {
        self.end_time.map(|end| end - self.start_time)
    }

    /// Check if event is ongoing
    pub fn is_ongoing(&self) -> bool {
        self.end_time.is_none()
    }

    /// Calculate impact score based on various factors
    pub fn calculate_impact(&mut self) {
        let severity_weight = self.severity.value();
        let price_impact = (self.price_change_pct.abs() / 10.0).min(1.0); // 10% = max
        let volume_impact = ((self.volume_ratio - 1.0) / 5.0).clamp(0.0, 1.0); // 6x vol = max

        self.impact_score =
            (severity_weight * 0.4 + price_impact * 0.4 + volume_impact * 0.2).clamp(0.0, 1.0);
    }
}

/// Detection thresholds for market events
#[derive(Debug, Clone)]
pub struct DetectionThresholds {
    /// Price change threshold for flash crash/rally (percentage)
    pub flash_move_threshold: f64,
    /// Time window for flash detection (milliseconds)
    pub flash_time_window: u64,
    /// Volume spike threshold (multiple of average)
    pub volume_spike_threshold: f64,
    /// Gap threshold for gap up/down (percentage)
    pub gap_threshold: f64,
    /// Large trade threshold (multiple of average)
    pub large_trade_threshold: f64,
    /// Volatility spike threshold (multiple of average)
    pub volatility_spike_threshold: f64,
    /// Minimum duration to consider event significant (milliseconds)
    pub min_event_duration: u64,
}

impl Default for DetectionThresholds {
    fn default() -> Self {
        Self {
            flash_move_threshold: 5.0,       // 5% move
            flash_time_window: 300_000,      // 5 minutes
            volume_spike_threshold: 3.0,     // 3x average volume
            gap_threshold: 2.0,              // 2% gap
            large_trade_threshold: 10.0,     // 10x average trade size
            volatility_spike_threshold: 2.5, // 2.5x average volatility
            min_event_duration: 1000,        // 1 second minimum
        }
    }
}

/// Configuration for market events module
#[derive(Debug, Clone)]
pub struct MarketEventsConfig {
    /// Detection thresholds
    pub thresholds: DetectionThresholds,
    /// Maximum events to store in memory
    pub max_events: usize,
    /// Maximum events per symbol to track
    pub max_events_per_symbol: usize,
    /// Window for calculating averages (milliseconds)
    pub average_window: u64,
    /// Enable automatic detection
    pub auto_detect: bool,
    /// Cooldown between similar events (milliseconds)
    pub event_cooldown: u64,
}

impl Default for MarketEventsConfig {
    fn default() -> Self {
        Self {
            thresholds: DetectionThresholds::default(),
            max_events: 10_000,
            max_events_per_symbol: 500,
            average_window: 24 * 60 * 60 * 1000, // 24 hours
            auto_detect: true,
            event_cooldown: 60_000, // 1 minute
        }
    }
}

/// Price/volume data point for analysis
#[derive(Debug, Clone)]
pub struct DataPoint {
    pub timestamp: u64,
    pub price: f64,
    pub volume: f64,
    pub high: f64,
    pub low: f64,
}

/// Symbol statistics for detection
#[derive(Debug, Clone)]
pub struct SymbolStats {
    /// Symbol
    pub symbol: String,
    /// Average volume
    pub avg_volume: f64,
    /// Average trade size
    pub avg_trade_size: f64,
    /// Average volatility (as percentage)
    pub avg_volatility: f64,
    /// Last close price
    pub last_close: f64,
    /// All-time high
    pub all_time_high: f64,
    /// All-time low
    pub all_time_low: f64,
    /// Recent data points
    pub recent_data: VecDeque<DataPoint>,
    /// Last event time by type
    pub last_event_times: HashMap<EventType, u64>,
}

impl SymbolStats {
    pub fn new(symbol: &str) -> Self {
        Self {
            symbol: symbol.to_string(),
            avg_volume: 0.0,
            avg_trade_size: 0.0,
            avg_volatility: 0.0,
            last_close: 0.0,
            all_time_high: 0.0,
            all_time_low: f64::MAX,
            recent_data: VecDeque::new(),
            last_event_times: HashMap::new(),
        }
    }

    /// Update statistics with new data point
    pub fn update(&mut self, data: &DataPoint) {
        self.recent_data.push_back(data.clone());

        // Keep only recent data
        while self.recent_data.len() > 1000 {
            self.recent_data.pop_front();
        }

        // Update all-time high/low
        if data.high > self.all_time_high {
            self.all_time_high = data.high;
        }
        if data.low < self.all_time_low && data.low > 0.0 {
            self.all_time_low = data.low;
        }

        // Recalculate averages
        if !self.recent_data.is_empty() {
            let sum_volume: f64 = self.recent_data.iter().map(|d| d.volume).sum();
            self.avg_volume = sum_volume / self.recent_data.len() as f64;

            // Calculate average volatility (high-low range as percentage of close)
            let sum_vol: f64 = self
                .recent_data
                .iter()
                .filter(|d| d.low > 0.0)
                .map(|d| (d.high - d.low) / d.low * 100.0)
                .sum();
            let count = self.recent_data.iter().filter(|d| d.low > 0.0).count();
            if count > 0 {
                self.avg_volatility = sum_vol / count as f64;
            }
        }

        self.last_close = data.price;
    }
}

/// Market events tracker and detector
pub struct MarketEvents {
    /// Configuration
    config: MarketEventsConfig,
    /// All stored events
    events: VecDeque<MarketEvent>,
    /// Events by symbol
    events_by_symbol: HashMap<String, VecDeque<u64>>,
    /// Ongoing events by ID
    ongoing_events: HashMap<u64, MarketEvent>,
    /// Symbol statistics
    symbol_stats: HashMap<String, SymbolStats>,
    /// Next event ID
    next_event_id: u64,
    /// Event counts by type
    event_counts: HashMap<EventType, u64>,
}

impl Default for MarketEvents {
    fn default() -> Self {
        Self::new()
    }
}

impl MarketEvents {
    /// Create a new instance
    pub fn new() -> Self {
        Self::with_config(MarketEventsConfig::default())
    }

    /// Create with custom configuration
    pub fn with_config(config: MarketEventsConfig) -> Self {
        Self {
            config,
            events: VecDeque::new(),
            events_by_symbol: HashMap::new(),
            ongoing_events: HashMap::new(),
            symbol_stats: HashMap::new(),
            next_event_id: 1,
            event_counts: HashMap::new(),
        }
    }

    /// Register a symbol for tracking
    pub fn register_symbol(&mut self, symbol: &str) {
        if !self.symbol_stats.contains_key(symbol) {
            self.symbol_stats
                .insert(symbol.to_string(), SymbolStats::new(symbol));
            self.events_by_symbol
                .insert(symbol.to_string(), VecDeque::new());
        }
    }

    /// Update symbol statistics with new data
    pub fn update_symbol_data(&mut self, symbol: &str, data: DataPoint) -> Vec<MarketEvent> {
        self.register_symbol(symbol);

        let mut detected_events = Vec::new();

        // Clone stats for detection to avoid borrow issues
        let stats_clone = self.symbol_stats.get(symbol).cloned();

        if self.config.auto_detect {
            if let Some(stats) = stats_clone {
                detected_events = self.detect_events(symbol, &data, &stats);
            }
        }

        // Update stats with mutable borrow
        if let Some(stats) = self.symbol_stats.get_mut(symbol) {
            stats.update(&data);
        }

        detected_events
    }

    /// Detect potential events based on new data
    fn detect_events(
        &mut self,
        symbol: &str,
        data: &DataPoint,
        stats: &SymbolStats,
    ) -> Vec<MarketEvent> {
        let mut events = Vec::new();
        let thresholds = self.config.thresholds.clone();

        // Flash crash/rally detection
        if let Some(flash_event) = self.detect_flash_move(symbol, data, stats, &thresholds) {
            events.push(flash_event);
        }

        // Volume spike detection
        if let Some(volume_event) = self.detect_volume_spike(symbol, data, stats, &thresholds) {
            events.push(volume_event);
        }

        // All-time high/low detection
        if let Some(ath_event) = self.detect_all_time_levels(symbol, data, stats) {
            events.push(ath_event);
        }

        // Gap detection
        if let Some(gap_event) = self.detect_gap(symbol, data, stats, &thresholds) {
            events.push(gap_event);
        }

        // Volatility spike detection
        if let Some(vol_event) = self.detect_volatility_spike(symbol, data, stats, &thresholds) {
            events.push(vol_event);
        }

        events
    }

    /// Detect flash crash or rally
    fn detect_flash_move(
        &self,
        symbol: &str,
        data: &DataPoint,
        stats: &SymbolStats,
        thresholds: &DetectionThresholds,
    ) -> Option<MarketEvent> {
        // Check cooldown
        if let Some(&last_time) = stats.last_event_times.get(&EventType::FlashCrash) {
            if data.timestamp - last_time < self.config.event_cooldown {
                return None;
            }
        }
        if let Some(&last_time) = stats.last_event_times.get(&EventType::FlashRally) {
            if data.timestamp - last_time < self.config.event_cooldown {
                return None;
            }
        }

        // Look back in recent data for rapid price change
        let lookback_time = data.timestamp.saturating_sub(thresholds.flash_time_window);

        for old_data in stats.recent_data.iter().rev() {
            if old_data.timestamp < lookback_time {
                break;
            }

            if old_data.price <= 0.0 {
                continue;
            }

            let price_change_pct = (data.price - old_data.price) / old_data.price * 100.0;

            if price_change_pct <= -thresholds.flash_move_threshold {
                let mut event = MarketEvent::new(
                    self.next_event_id,
                    EventType::FlashCrash,
                    vec![symbol.to_string()],
                    old_data.timestamp,
                    old_data.price,
                );
                event.update_price(data.price);
                event.description = format!(
                    "Flash crash detected: {:.2}% drop in {}ms",
                    price_change_pct,
                    data.timestamp - old_data.timestamp
                );
                event.calculate_impact();
                return Some(event);
            } else if price_change_pct >= thresholds.flash_move_threshold {
                let mut event = MarketEvent::new(
                    self.next_event_id,
                    EventType::FlashRally,
                    vec![symbol.to_string()],
                    old_data.timestamp,
                    old_data.price,
                );
                event.update_price(data.price);
                event.description = format!(
                    "Flash rally detected: +{:.2}% in {}ms",
                    price_change_pct,
                    data.timestamp - old_data.timestamp
                );
                event.calculate_impact();
                return Some(event);
            }
        }

        None
    }

    /// Detect volume spike
    fn detect_volume_spike(
        &self,
        symbol: &str,
        data: &DataPoint,
        stats: &SymbolStats,
        thresholds: &DetectionThresholds,
    ) -> Option<MarketEvent> {
        // Check cooldown
        if let Some(&last_time) = stats.last_event_times.get(&EventType::VolumeSpike) {
            if data.timestamp - last_time < self.config.event_cooldown {
                return None;
            }
        }

        if stats.avg_volume <= 0.0 {
            return None;
        }

        let volume_ratio = data.volume / stats.avg_volume;

        if volume_ratio >= thresholds.volume_spike_threshold {
            let mut event = MarketEvent::new(
                self.next_event_id,
                EventType::VolumeSpike,
                vec![symbol.to_string()],
                data.timestamp,
                data.price,
            );
            event.volume = data.volume;
            event.avg_volume = stats.avg_volume;
            event.volume_ratio = volume_ratio;
            event.description = format!("Volume spike: {:.1}x average volume", volume_ratio);
            event.calculate_impact();
            return Some(event);
        }

        None
    }

    /// Detect all-time high or low
    fn detect_all_time_levels(
        &self,
        symbol: &str,
        data: &DataPoint,
        stats: &SymbolStats,
    ) -> Option<MarketEvent> {
        // Check cooldown
        if let Some(&last_time) = stats.last_event_times.get(&EventType::AllTimeHigh) {
            if data.timestamp - last_time < self.config.event_cooldown * 10 {
                return None;
            }
        }
        if let Some(&last_time) = stats.last_event_times.get(&EventType::AllTimeLow) {
            if data.timestamp - last_time < self.config.event_cooldown * 10 {
                return None;
            }
        }

        if data.high > stats.all_time_high && stats.all_time_high > 0.0 {
            let mut event = MarketEvent::new(
                self.next_event_id,
                EventType::AllTimeHigh,
                vec![symbol.to_string()],
                data.timestamp,
                data.high,
            );
            event.description = format!(
                "New all-time high: {:.2} (previous: {:.2})",
                data.high, stats.all_time_high
            );
            event.calculate_impact();
            return Some(event);
        }

        if data.low < stats.all_time_low && data.low > 0.0 && stats.all_time_low < f64::MAX {
            let mut event = MarketEvent::new(
                self.next_event_id,
                EventType::AllTimeLow,
                vec![symbol.to_string()],
                data.timestamp,
                data.low,
            );
            event.description = format!(
                "New all-time low: {:.2} (previous: {:.2})",
                data.low, stats.all_time_low
            );
            event.calculate_impact();
            return Some(event);
        }

        None
    }

    /// Detect gap up or down
    fn detect_gap(
        &self,
        symbol: &str,
        data: &DataPoint,
        stats: &SymbolStats,
        thresholds: &DetectionThresholds,
    ) -> Option<MarketEvent> {
        if stats.last_close <= 0.0 {
            return None;
        }

        // Check cooldown
        if let Some(&last_time) = stats.last_event_times.get(&EventType::GapUp) {
            if data.timestamp - last_time < self.config.event_cooldown {
                return None;
            }
        }
        if let Some(&last_time) = stats.last_event_times.get(&EventType::GapDown) {
            if data.timestamp - last_time < self.config.event_cooldown {
                return None;
            }
        }

        let gap_pct = (data.price - stats.last_close) / stats.last_close * 100.0;

        if gap_pct >= thresholds.gap_threshold {
            let mut event = MarketEvent::new(
                self.next_event_id,
                EventType::GapUp,
                vec![symbol.to_string()],
                data.timestamp,
                stats.last_close,
            );
            event.update_price(data.price);
            event.description = format!("Gap up: +{:.2}% from previous close", gap_pct);
            event.calculate_impact();
            return Some(event);
        } else if gap_pct <= -thresholds.gap_threshold {
            let mut event = MarketEvent::new(
                self.next_event_id,
                EventType::GapDown,
                vec![symbol.to_string()],
                data.timestamp,
                stats.last_close,
            );
            event.update_price(data.price);
            event.description = format!("Gap down: {:.2}% from previous close", gap_pct);
            event.calculate_impact();
            return Some(event);
        }

        None
    }

    /// Detect volatility spike
    fn detect_volatility_spike(
        &self,
        symbol: &str,
        data: &DataPoint,
        stats: &SymbolStats,
        thresholds: &DetectionThresholds,
    ) -> Option<MarketEvent> {
        if stats.avg_volatility <= 0.0 || data.low <= 0.0 {
            return None;
        }

        // Check cooldown
        if let Some(&last_time) = stats.last_event_times.get(&EventType::VolatilitySpike) {
            if data.timestamp - last_time < self.config.event_cooldown {
                return None;
            }
        }

        let current_vol = (data.high - data.low) / data.low * 100.0;
        let vol_ratio = current_vol / stats.avg_volatility;

        if vol_ratio >= thresholds.volatility_spike_threshold {
            let mut event = MarketEvent::new(
                self.next_event_id,
                EventType::VolatilitySpike,
                vec![symbol.to_string()],
                data.timestamp,
                data.price,
            );
            event
                .metadata
                .insert("current_volatility".to_string(), current_vol.to_string());
            event.metadata.insert(
                "average_volatility".to_string(),
                stats.avg_volatility.to_string(),
            );
            event
                .metadata
                .insert("volatility_ratio".to_string(), vol_ratio.to_string());
            event.description = format!(
                "Volatility spike: {:.1}x average ({:.2}% vs {:.2}%)",
                vol_ratio, current_vol, stats.avg_volatility
            );
            event.calculate_impact();
            return Some(event);
        }

        None
    }

    /// Record an event (internal, updates tracking structures)
    fn record_event_internal(&mut self, mut event: MarketEvent) {
        event.id = self.next_event_id;
        self.next_event_id += 1;

        // Update event counts
        *self.event_counts.entry(event.event_type).or_insert(0) += 1;

        // Update last event time for symbols
        for symbol in &event.symbols {
            if let Some(stats) = self.symbol_stats.get_mut(symbol) {
                stats
                    .last_event_times
                    .insert(event.event_type, event.start_time);
            }

            // Track event by symbol
            if let Some(symbol_events) = self.events_by_symbol.get_mut(symbol) {
                symbol_events.push_back(event.id);
                while symbol_events.len() > self.config.max_events_per_symbol {
                    symbol_events.pop_front();
                }
            }
        }

        // Store event
        if event.is_ongoing() {
            self.ongoing_events.insert(event.id, event);
        } else {
            self.events.push_back(event);
        }

        // Trim events if needed
        while self.events.len() > self.config.max_events {
            self.events.pop_front();
        }
    }

    /// Manually record a custom event
    pub fn record_event(&mut self, event: MarketEvent) {
        self.record_event_internal(event);
    }

    /// Create and record a custom event
    pub fn create_event(
        &mut self,
        event_type: EventType,
        symbols: Vec<String>,
        timestamp: u64,
        price: f64,
        description: &str,
    ) -> u64 {
        let mut event = MarketEvent::new(self.next_event_id, event_type, symbols, timestamp, price);
        event.description = description.to_string();
        // Close the event immediately so it's stored in events, not ongoing_events
        event.end_time = Some(timestamp);
        event.calculate_impact();
        let id = event.id;
        self.record_event_internal(event);
        id
    }

    /// Close an ongoing event
    pub fn close_event(&mut self, event_id: u64, end_time: u64, end_price: f64) -> Result<()> {
        if let Some(mut event) = self.ongoing_events.remove(&event_id) {
            event.close(end_time, end_price);
            event.calculate_impact();
            self.events.push_back(event);
            Ok(())
        } else {
            Err(Error::NotFound(format!("Event {} not found", event_id)))
        }
    }

    /// Get recent events
    pub fn get_recent_events(&self, count: usize) -> Vec<&MarketEvent> {
        self.events.iter().rev().take(count).collect()
    }

    /// Get events for a symbol
    pub fn get_events_for_symbol(&self, symbol: &str) -> Vec<&MarketEvent> {
        if let Some(event_ids) = self.events_by_symbol.get(symbol) {
            event_ids
                .iter()
                .filter_map(|id| {
                    self.events
                        .iter()
                        .find(|e| e.id == *id)
                        .or_else(|| self.ongoing_events.get(id))
                })
                .collect()
        } else {
            Vec::new()
        }
    }

    /// Get events by type
    pub fn get_events_by_type(&self, event_type: EventType) -> Vec<&MarketEvent> {
        self.events
            .iter()
            .filter(|e| e.event_type == event_type)
            .collect()
    }

    /// Get events above severity threshold
    pub fn get_events_by_severity(&self, min_severity: EventSeverity) -> Vec<&MarketEvent> {
        self.events
            .iter()
            .filter(|e| e.severity >= min_severity)
            .collect()
    }

    /// Get ongoing events
    pub fn get_ongoing_events(&self) -> Vec<&MarketEvent> {
        self.ongoing_events.values().collect()
    }

    /// Get event statistics
    pub fn get_stats(&self) -> MarketEventsStats {
        let critical_count = self
            .events
            .iter()
            .filter(|e| e.severity == EventSeverity::Critical)
            .count();
        let high_count = self
            .events
            .iter()
            .filter(|e| e.severity == EventSeverity::High)
            .count();

        let avg_impact = if self.events.is_empty() {
            0.0
        } else {
            self.events.iter().map(|e| e.impact_score).sum::<f64>() / self.events.len() as f64
        };

        MarketEventsStats {
            total_events: self.events.len() + self.ongoing_events.len(),
            ongoing_events: self.ongoing_events.len(),
            critical_events: critical_count,
            high_severity_events: high_count,
            symbols_tracked: self.symbol_stats.len(),
            event_counts: self.event_counts.clone(),
            average_impact_score: avg_impact,
        }
    }

    /// Main processing function
    pub fn process(&self) -> Result<()> {
        Ok(())
    }
}

/// Statistics for market events
#[derive(Debug, Clone)]
pub struct MarketEventsStats {
    pub total_events: usize,
    pub ongoing_events: usize,
    pub critical_events: usize,
    pub high_severity_events: usize,
    pub symbols_tracked: usize,
    pub event_counts: HashMap<EventType, u64>,
    pub average_impact_score: f64,
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_basic() {
        let instance = MarketEvents::new();
        assert!(instance.process().is_ok());
    }

    #[test]
    fn test_register_symbol() {
        let mut events = MarketEvents::new();
        events.register_symbol("BTC-USD");
        assert!(events.symbol_stats.contains_key("BTC-USD"));
    }

    #[test]
    fn test_create_event() {
        let mut events = MarketEvents::new();
        let id = events.create_event(
            EventType::Custom,
            vec!["BTC-USD".to_string()],
            1000,
            50000.0,
            "Test event",
        );
        assert!(id > 0);
        assert_eq!(events.events.len(), 1);
    }

    #[test]
    fn test_flash_crash_detection() {
        let config = MarketEventsConfig {
            thresholds: DetectionThresholds {
                flash_move_threshold: 5.0,
                flash_time_window: 60000,
                ..Default::default()
            },
            ..Default::default()
        };
        let mut events = MarketEvents::with_config(config);
        events.register_symbol("BTC-USD");

        // Add initial data points
        for i in 0..10 {
            let data = DataPoint {
                timestamp: i * 1000,
                price: 50000.0,
                volume: 100.0,
                high: 50100.0,
                low: 49900.0,
            };
            events.update_symbol_data("BTC-USD", data);
        }

        // Trigger flash crash (6% drop)
        let crash_data = DataPoint {
            timestamp: 15000,
            price: 47000.0,
            volume: 500.0,
            high: 50000.0,
            low: 46500.0,
        };
        let detected = events.update_symbol_data("BTC-USD", crash_data);

        assert!(!detected.is_empty());
        assert!(
            detected
                .iter()
                .any(|e| e.event_type == EventType::FlashCrash)
        );
    }

    #[test]
    fn test_volume_spike_detection() {
        let config = MarketEventsConfig {
            thresholds: DetectionThresholds {
                volume_spike_threshold: 3.0,
                ..Default::default()
            },
            event_cooldown: 0, // No cooldown for test
            ..Default::default()
        };
        let mut events = MarketEvents::with_config(config);
        events.register_symbol("ETH-USD");

        // Build up average volume
        for i in 0..20 {
            let data = DataPoint {
                timestamp: i * 1000,
                price: 3000.0,
                volume: 100.0,
                high: 3010.0,
                low: 2990.0,
            };
            events.update_symbol_data("ETH-USD", data);
        }

        // Volume spike (5x average)
        let spike_data = DataPoint {
            timestamp: 25000,
            price: 3050.0,
            volume: 500.0,
            high: 3100.0,
            low: 3000.0,
        };
        let detected = events.update_symbol_data("ETH-USD", spike_data);

        assert!(!detected.is_empty());
        assert!(
            detected
                .iter()
                .any(|e| e.event_type == EventType::VolumeSpike)
        );
    }

    #[test]
    fn test_get_events_by_severity() {
        let mut events = MarketEvents::new();

        // Create events of different severities
        let mut critical_event = MarketEvent::new(
            1,
            EventType::FlashCrash,
            vec!["BTC-USD".to_string()],
            1000,
            50000.0,
        );
        critical_event.severity = EventSeverity::Critical;
        critical_event.end_time = Some(1000); // Close the event so it's stored in events
        events.record_event(critical_event);

        let mut low_event = MarketEvent::new(
            2,
            EventType::AllTimeHigh,
            vec!["ETH-USD".to_string()],
            2000,
            3000.0,
        );
        low_event.severity = EventSeverity::Low;
        low_event.end_time = Some(2000); // Close the event so it's stored in events
        events.record_event(low_event);

        let high_severity = events.get_events_by_severity(EventSeverity::High);
        assert_eq!(high_severity.len(), 1);
    }

    #[test]
    fn test_ongoing_events() {
        let mut events = MarketEvents::new();

        let ongoing = MarketEvent::new(
            1,
            EventType::VolatilitySpike,
            vec!["BTC-USD".to_string()],
            1000,
            50000.0,
        );
        // Don't close it - it's ongoing
        events.ongoing_events.insert(ongoing.id, ongoing);

        assert_eq!(events.get_ongoing_events().len(), 1);

        // Close it
        events.close_event(1, 5000, 48000.0).unwrap();
        assert_eq!(events.get_ongoing_events().len(), 0);
        assert_eq!(events.events.len(), 1);
    }

    #[test]
    fn test_event_stats() {
        let mut events = MarketEvents::new();

        for i in 0..5 {
            events.create_event(
                EventType::VolumeSpike,
                vec!["BTC-USD".to_string()],
                i * 1000,
                50000.0 + i as f64 * 100.0,
                "Test event",
            );
        }

        let stats = events.get_stats();
        assert_eq!(stats.total_events, 5);
        assert_eq!(stats.event_counts.get(&EventType::VolumeSpike), Some(&5));
    }

    #[test]
    fn test_event_impact_calculation() {
        let mut event = MarketEvent::new(
            1,
            EventType::FlashCrash,
            vec!["BTC-USD".to_string()],
            1000,
            50000.0,
        );
        event.update_price(45000.0); // 10% drop
        event.volume = 1000.0;
        event.avg_volume = 200.0;
        event.volume_ratio = 5.0;
        event.severity = EventSeverity::Critical;
        event.calculate_impact();

        assert!(event.impact_score > 0.5);
    }

    #[test]
    fn test_get_events_for_symbol() {
        let mut events = MarketEvents::new();
        events.register_symbol("BTC-USD");
        events.register_symbol("ETH-USD");

        events.create_event(
            EventType::VolumeSpike,
            vec!["BTC-USD".to_string()],
            1000,
            50000.0,
            "BTC event",
        );

        events.create_event(
            EventType::GapUp,
            vec!["ETH-USD".to_string()],
            2000,
            3000.0,
            "ETH event",
        );

        let btc_events = events.get_events_for_symbol("BTC-USD");
        assert_eq!(btc_events.len(), 1);
        assert_eq!(btc_events[0].symbols[0], "BTC-USD");
    }

    #[test]
    fn test_symbol_stats_update() {
        let mut stats = SymbolStats::new("BTC-USD");

        for i in 0..10 {
            stats.update(&DataPoint {
                timestamp: i * 1000,
                price: 50000.0 + i as f64 * 100.0,
                volume: 100.0 + i as f64 * 10.0,
                high: 50100.0 + i as f64 * 100.0,
                low: 49900.0 + i as f64 * 100.0,
            });
        }

        assert!(stats.avg_volume > 0.0);
        assert!(stats.all_time_high > stats.all_time_low);
        assert_eq!(stats.recent_data.len(), 10);
    }

    #[test]
    fn test_event_severity_ordering() {
        assert!(EventSeverity::Critical > EventSeverity::High);
        assert!(EventSeverity::High > EventSeverity::Medium);
        assert!(EventSeverity::Medium > EventSeverity::Low);
        assert!(EventSeverity::Low > EventSeverity::Info);
    }

    #[test]
    fn test_event_type_default_severity() {
        assert_eq!(
            EventType::FlashCrash.default_severity(),
            EventSeverity::Critical
        );
        assert_eq!(
            EventType::VolumeSpike.default_severity(),
            EventSeverity::Medium
        );
        assert_eq!(
            EventType::AllTimeHigh.default_severity(),
            EventSeverity::Low
        );
    }
}
