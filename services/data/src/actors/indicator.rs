//! Indicator Actor - Real-time Technical Indicator Calculator
//!
//! This actor receives candle data from the Router and calculates technical
//! indicators in real-time using incremental algorithms for efficiency.
//!
//! ## Supported Indicators:
//! - EMA (8, 21, 50, 200 periods)
//! - RSI (14 period)
//! - MACD (12/26/9)
//! - ATR (14 period)
//!
//! ## Architecture:
//! - Maintains per-symbol, per-timeframe indicator state
//! - Uses O(1) incremental updates where possible
//! - Outputs to QuestDB `indicators_crypto` table
//! - Exposes current values via API

use anyhow::{Context, Result};
use std::collections::HashMap;
use std::sync::Arc;
use tokio::sync::{RwLock, broadcast, mpsc};
use tracing::{debug, error, info};

use crate::config::Config;
use crate::metrics::prometheus_exporter::{
    INDICATOR_CALCULATION_DURATION, INDICATOR_PAIRS_TRACKED, INDICATORS_CALCULATED,
};
use crate::storage::StorageManager;

/// Indicator calculation result
#[derive(Debug, Clone)]
pub struct IndicatorData {
    /// Symbol (e.g., "BTC-USDT")
    pub symbol: String,
    /// Exchange source
    pub exchange: String,
    /// Timeframe (e.g., "1m", "5m", "1h")
    pub timeframe: String,
    /// Candle timestamp (open time)
    pub timestamp: i64,
    /// Close price used for calculation
    pub close: f64,
    /// EMA 8
    pub ema_8: Option<f64>,
    /// EMA 21
    pub ema_21: Option<f64>,
    /// EMA 50
    pub ema_50: Option<f64>,
    /// EMA 200
    pub ema_200: Option<f64>,
    /// RSI 14
    pub rsi_14: Option<f64>,
    /// MACD Line (12-26)
    pub macd_line: Option<f64>,
    /// MACD Signal (9-period EMA of MACD)
    pub macd_signal: Option<f64>,
    /// MACD Histogram
    pub macd_histogram: Option<f64>,
    /// ATR 14
    pub atr_14: Option<f64>,
    /// Calculation timestamp
    pub calculated_at: i64,
}

/// Message types for the indicator actor
#[derive(Debug, Clone)]
pub enum IndicatorMessage {
    /// New candle data to process
    Candle(CandleInput),
    /// Request current indicator values
    GetIndicators {
        symbol: String,
        timeframe: String,
        response_tx: mpsc::Sender<Option<IndicatorData>>,
    },
    /// Reset indicators for a symbol
    Reset { symbol: String, timeframe: String },
}

/// Input candle for indicator calculation
#[derive(Debug, Clone)]
pub struct CandleInput {
    pub symbol: String,
    pub exchange: String,
    pub timeframe: String,
    pub timestamp: i64,
    pub open: f64,
    pub high: f64,
    pub low: f64,
    pub close: f64,
    pub volume: f64,
}

/// Incremental EMA calculator with proper warmup
#[derive(Debug, Clone)]
struct IncrementalEMA {
    period: usize,
    alpha: f64,
    value: Option<f64>,
    warmup_prices: Vec<f64>,
    initialized: bool,
}

impl IncrementalEMA {
    fn new(period: usize) -> Self {
        let alpha = 2.0 / (period as f64 + 1.0);
        Self {
            period,
            alpha,
            value: None,
            warmup_prices: Vec::with_capacity(period),
            initialized: false,
        }
    }

    fn update(&mut self, price: f64) -> Option<f64> {
        if !self.initialized {
            self.warmup_prices.push(price);
            if self.warmup_prices.len() >= self.period {
                // Initialize with SMA
                let sma: f64 = self.warmup_prices.iter().sum::<f64>() / self.period as f64;
                self.value = Some(sma);
                self.initialized = true;
                self.warmup_prices.clear();
            }
        } else if let Some(prev) = self.value {
            self.value = Some(price * self.alpha + prev * (1.0 - self.alpha));
        }
        self.value
    }

    fn value(&self) -> Option<f64> {
        self.value
    }

    fn is_ready(&self) -> bool {
        self.initialized
    }

    fn reset(&mut self) {
        self.value = None;
        self.warmup_prices.clear();
        self.initialized = false;
    }
}

/// Incremental RSI calculator
#[derive(Debug, Clone)]
struct IncrementalRSI {
    period: usize,
    avg_gain: f64,
    avg_loss: f64,
    prev_close: Option<f64>,
    count: usize,
    initialized: bool,
    gains: Vec<f64>,
    losses: Vec<f64>,
}

impl IncrementalRSI {
    fn new(period: usize) -> Self {
        Self {
            period,
            avg_gain: 0.0,
            avg_loss: 0.0,
            prev_close: None,
            count: 0,
            initialized: false,
            gains: Vec::with_capacity(period),
            losses: Vec::with_capacity(period),
        }
    }

    fn update(&mut self, close: f64) -> Option<f64> {
        if let Some(prev) = self.prev_close {
            let change = close - prev;
            let gain = if change > 0.0 { change } else { 0.0 };
            let loss = if change < 0.0 { -change } else { 0.0 };

            if !self.initialized {
                self.gains.push(gain);
                self.losses.push(loss);
                self.count += 1;

                if self.count >= self.period {
                    // Initialize with simple average
                    self.avg_gain = self.gains.iter().sum::<f64>() / self.period as f64;
                    self.avg_loss = self.losses.iter().sum::<f64>() / self.period as f64;
                    self.initialized = true;
                    self.gains.clear();
                    self.losses.clear();
                }
            } else {
                // Smoothed moving average (Wilder's smoothing)
                self.avg_gain =
                    (self.avg_gain * (self.period - 1) as f64 + gain) / self.period as f64;
                self.avg_loss =
                    (self.avg_loss * (self.period - 1) as f64 + loss) / self.period as f64;
            }
        }

        self.prev_close = Some(close);

        if self.initialized {
            let rsi = if self.avg_loss == 0.0 {
                100.0
            } else {
                let rs = self.avg_gain / self.avg_loss;
                100.0 - (100.0 / (1.0 + rs))
            };
            Some(rsi)
        } else {
            None
        }
    }

    fn value(&self) -> Option<f64> {
        if self.initialized {
            let rsi = if self.avg_loss == 0.0 {
                100.0
            } else {
                let rs = self.avg_gain / self.avg_loss;
                100.0 - (100.0 / (1.0 + rs))
            };
            Some(rsi)
        } else {
            None
        }
    }

    fn reset(&mut self) {
        self.avg_gain = 0.0;
        self.avg_loss = 0.0;
        self.prev_close = None;
        self.count = 0;
        self.initialized = false;
        self.gains.clear();
        self.losses.clear();
    }
}

/// Incremental MACD calculator
#[derive(Debug, Clone)]
struct IncrementalMACD {
    fast_ema: IncrementalEMA,
    slow_ema: IncrementalEMA,
    signal_ema: IncrementalEMA,
}

impl IncrementalMACD {
    fn new(fast_period: usize, slow_period: usize, signal_period: usize) -> Self {
        Self {
            fast_ema: IncrementalEMA::new(fast_period),
            slow_ema: IncrementalEMA::new(slow_period),
            signal_ema: IncrementalEMA::new(signal_period),
        }
    }

    fn update(&mut self, price: f64) -> (Option<f64>, Option<f64>, Option<f64>) {
        let fast = self.fast_ema.update(price);
        let slow = self.slow_ema.update(price);

        let macd_line = match (fast, slow) {
            (Some(f), Some(s)) => Some(f - s),
            _ => None,
        };

        let signal = if let Some(macd) = macd_line {
            self.signal_ema.update(macd)
        } else {
            None
        };

        let histogram = match (macd_line, signal) {
            (Some(m), Some(s)) => Some(m - s),
            _ => None,
        };

        (macd_line, signal, histogram)
    }

    fn values(&self) -> (Option<f64>, Option<f64>, Option<f64>) {
        let fast = self.fast_ema.value();
        let slow = self.slow_ema.value();

        let macd_line = match (fast, slow) {
            (Some(f), Some(s)) => Some(f - s),
            _ => None,
        };

        let signal = self.signal_ema.value();

        let histogram = match (macd_line, signal) {
            (Some(m), Some(s)) => Some(m - s),
            _ => None,
        };

        (macd_line, signal, histogram)
    }

    fn reset(&mut self) {
        self.fast_ema.reset();
        self.slow_ema.reset();
        self.signal_ema.reset();
    }
}

/// Incremental ATR calculator
#[derive(Debug, Clone)]
struct IncrementalATR {
    period: usize,
    ema: IncrementalEMA,
    prev_close: Option<f64>,
}

impl IncrementalATR {
    fn new(period: usize) -> Self {
        Self {
            period,
            ema: IncrementalEMA::new(period),
            prev_close: None,
        }
    }

    fn update(&mut self, high: f64, low: f64, close: f64) -> Option<f64> {
        let tr = if let Some(prev_close) = self.prev_close {
            let hl = high - low;
            let hc = (high - prev_close).abs();
            let lc = (low - prev_close).abs();
            hl.max(hc).max(lc)
        } else {
            high - low
        };

        self.prev_close = Some(close);
        self.ema.update(tr)
    }

    fn value(&self) -> Option<f64> {
        self.ema.value()
    }

    fn reset(&mut self) {
        self.ema.reset();
        self.prev_close = None;
    }
}

/// Per-symbol, per-timeframe indicator state
#[derive(Debug, Clone)]
struct IndicatorState {
    symbol: String,
    exchange: String,
    timeframe: String,
    ema_8: IncrementalEMA,
    ema_21: IncrementalEMA,
    ema_50: IncrementalEMA,
    ema_200: IncrementalEMA,
    rsi_14: IncrementalRSI,
    macd: IncrementalMACD,
    atr_14: IncrementalATR,
    last_update: i64,
    candle_count: u64,
}

impl IndicatorState {
    fn new(symbol: String, exchange: String, timeframe: String) -> Self {
        Self {
            symbol,
            exchange,
            timeframe,
            ema_8: IncrementalEMA::new(8),
            ema_21: IncrementalEMA::new(21),
            ema_50: IncrementalEMA::new(50),
            ema_200: IncrementalEMA::new(200),
            rsi_14: IncrementalRSI::new(14),
            macd: IncrementalMACD::new(12, 26, 9),
            atr_14: IncrementalATR::new(14),
            last_update: 0,
            candle_count: 0,
        }
    }

    fn update(&mut self, candle: &CandleInput) -> IndicatorData {
        // Update all indicators with new candle data
        let ema_8 = self.ema_8.update(candle.close);
        let ema_21 = self.ema_21.update(candle.close);
        let ema_50 = self.ema_50.update(candle.close);
        let ema_200 = self.ema_200.update(candle.close);
        let rsi_14 = self.rsi_14.update(candle.close);
        let (macd_line, macd_signal, macd_histogram) = self.macd.update(candle.close);
        let atr_14 = self.atr_14.update(candle.high, candle.low, candle.close);

        self.last_update = candle.timestamp;
        self.candle_count += 1;

        IndicatorData {
            symbol: candle.symbol.clone(),
            exchange: candle.exchange.clone(),
            timeframe: candle.timeframe.clone(),
            timestamp: candle.timestamp,
            close: candle.close,
            ema_8,
            ema_21,
            ema_50,
            ema_200,
            rsi_14,
            macd_line,
            macd_signal,
            macd_histogram,
            atr_14,
            calculated_at: chrono::Utc::now().timestamp_millis(),
        }
    }

    fn current_values(&self) -> IndicatorData {
        let (macd_line, macd_signal, macd_histogram) = self.macd.values();

        IndicatorData {
            symbol: self.symbol.clone(),
            exchange: self.exchange.clone(),
            timeframe: self.timeframe.clone(),
            timestamp: self.last_update,
            close: 0.0, // Not available without last candle
            ema_8: self.ema_8.value(),
            ema_21: self.ema_21.value(),
            ema_50: self.ema_50.value(),
            ema_200: self.ema_200.value(),
            rsi_14: self.rsi_14.value(),
            macd_line,
            macd_signal,
            macd_histogram,
            atr_14: self.atr_14.value(),
            calculated_at: chrono::Utc::now().timestamp_millis(),
        }
    }

    fn reset(&mut self) {
        self.ema_8.reset();
        self.ema_21.reset();
        self.ema_50.reset();
        self.ema_200.reset();
        self.rsi_14.reset();
        self.macd.reset();
        self.atr_14.reset();
        self.last_update = 0;
        self.candle_count = 0;
    }

    fn warmup_status(&self) -> WarmupStatus {
        WarmupStatus {
            ema_8_ready: self.ema_8.is_ready(),
            ema_21_ready: self.ema_21.is_ready(),
            ema_50_ready: self.ema_50.is_ready(),
            ema_200_ready: self.ema_200.is_ready(),
            rsi_ready: self.rsi_14.value().is_some(),
            macd_ready: self.macd.values().0.is_some(),
            atr_ready: self.atr_14.value().is_some(),
            candles_processed: self.candle_count,
        }
    }
}

/// Warmup status for indicators
#[derive(Debug, Clone)]
pub struct WarmupStatus {
    pub ema_8_ready: bool,
    pub ema_21_ready: bool,
    pub ema_50_ready: bool,
    pub ema_200_ready: bool,
    pub rsi_ready: bool,
    pub macd_ready: bool,
    pub atr_ready: bool,
    pub candles_processed: u64,
}

impl WarmupStatus {
    pub fn all_ready(&self) -> bool {
        self.ema_8_ready
            && self.ema_21_ready
            && self.ema_50_ready
            && self.ema_200_ready
            && self.rsi_ready
            && self.macd_ready
            && self.atr_ready
    }

    pub fn min_candles_needed(&self) -> u64 {
        // EMA 200 needs 200 candles, RSI needs 15 (period + 1), MACD needs 35 (26 + 9)
        200
    }
}

/// Key for indicator state map
#[derive(Debug, Clone, Hash, Eq, PartialEq)]
struct StateKey {
    symbol: String,
    timeframe: String,
}

impl StateKey {
    fn new(symbol: &str, timeframe: &str) -> Self {
        Self {
            symbol: symbol.to_string(),
            timeframe: timeframe.to_string(),
        }
    }
}

/// Indicator Actor - calculates technical indicators in real-time
pub struct IndicatorActor {
    #[allow(dead_code)]
    config: Arc<Config>,
    storage: Arc<StorageManager>,

    /// Indicator state per symbol/timeframe
    state: Arc<RwLock<HashMap<StateKey, IndicatorState>>>,

    /// Sender for incoming messages
    tx: mpsc::UnboundedSender<IndicatorMessage>,

    /// Broadcast sender for indicator updates
    pub indicator_tx: broadcast::Sender<IndicatorData>,
}

impl IndicatorActor {
    /// Create a new IndicatorActor
    pub async fn new(
        config: Arc<Config>,
        storage: Arc<StorageManager>,
        mut shutdown_rx: broadcast::Receiver<()>,
    ) -> Result<Arc<Self>> {
        let (tx, mut rx) = mpsc::unbounded_channel();
        let (indicator_tx, _) = broadcast::channel(1000);

        let state = Arc::new(RwLock::new(HashMap::new()));

        let actor = Arc::new(Self {
            config: config.clone(),
            storage: storage.clone(),
            state: state.clone(),
            tx,
            indicator_tx: indicator_tx.clone(),
        });

        // Spawn the main processing loop
        let storage_clone = storage.clone();
        let state_clone = state.clone();
        let indicator_tx_clone = indicator_tx.clone();

        tokio::spawn(async move {
            if let Err(e) = Self::run_loop(
                storage_clone,
                state_clone,
                indicator_tx_clone,
                &mut rx,
                &mut shutdown_rx,
            )
            .await
            {
                error!("IndicatorActor: Task failed: {}", e);
            }
        });

        info!("IndicatorActor: Initialized and running");
        Ok(actor)
    }

    /// Get sender handle for this actor
    pub fn get_sender(&self) -> mpsc::UnboundedSender<IndicatorMessage> {
        self.tx.clone()
    }

    /// Get current indicator values for a symbol/timeframe
    pub async fn get_indicators(&self, symbol: &str, timeframe: &str) -> Option<IndicatorData> {
        let state = self.state.read().await;
        let key = StateKey::new(symbol, timeframe);
        state.get(&key).map(|s| s.current_values())
    }

    /// Get warmup status for a symbol/timeframe
    pub async fn get_warmup_status(&self, symbol: &str, timeframe: &str) -> Option<WarmupStatus> {
        let state = self.state.read().await;
        let key = StateKey::new(symbol, timeframe);
        state.get(&key).map(|s| s.warmup_status())
    }

    /// Get all tracked symbol/timeframe pairs
    pub async fn get_tracked_pairs(&self) -> Vec<(String, String)> {
        let state = self.state.read().await;
        state
            .keys()
            .map(|k| (k.symbol.clone(), k.timeframe.clone()))
            .collect()
    }

    /// Main processing loop
    async fn run_loop(
        storage: Arc<StorageManager>,
        state: Arc<RwLock<HashMap<StateKey, IndicatorState>>>,
        indicator_tx: broadcast::Sender<IndicatorData>,
        rx: &mut mpsc::UnboundedReceiver<IndicatorMessage>,
        shutdown_rx: &mut broadcast::Receiver<()>,
    ) -> Result<()> {
        info!("IndicatorActor: Starting processing loop");

        let mut stats_interval = tokio::time::interval(tokio::time::Duration::from_secs(60));
        let mut candles_processed: u64 = 0;
        let mut indicators_calculated: u64 = 0;

        loop {
            tokio::select! {
                Some(msg) = rx.recv() => {
                    match msg {
                        IndicatorMessage::Candle(candle) => {
                            match Self::process_candle(
                                &storage,
                                &state,
                                &indicator_tx,
                                candle,
                            ).await {
                                Ok(_) => {
                                    candles_processed += 1;
                                    indicators_calculated += 1;
                                }
                                Err(e) => {
                                    error!("IndicatorActor: Failed to process candle: {}", e);
                                }
                            }
                        }

                        IndicatorMessage::GetIndicators { symbol, timeframe, response_tx } => {
                            let state_read = state.read().await;
                            let key = StateKey::new(&symbol, &timeframe);
                            let result = state_read.get(&key).map(|s| s.current_values());
                            let _ = response_tx.send(result).await;
                        }

                        IndicatorMessage::Reset { symbol, timeframe } => {
                            let mut state_write = state.write().await;
                            let key = StateKey::new(&symbol, &timeframe);
                            if let Some(indicator_state) = state_write.get_mut(&key) {
                                indicator_state.reset();
                                info!("IndicatorActor: Reset indicators for {}:{}", symbol, timeframe);
                            }
                        }
                    }
                }

                _ = stats_interval.tick() => {
                    let state_read = state.read().await;
                    let tracked_count = state_read.len();
                    info!(
                        "IndicatorActor Stats - Tracked: {} pairs, Candles: {}, Calculations: {}",
                        tracked_count, candles_processed, indicators_calculated
                    );

                    // Log warmup status for each pair
                    for (key, indicator_state) in state_read.iter() {
                        let status = indicator_state.warmup_status();
                        if !status.all_ready() {
                            debug!(
                                "IndicatorActor: {}:{} warmup progress: {}/{} candles",
                                key.symbol, key.timeframe,
                                status.candles_processed,
                                status.min_candles_needed()
                            );
                        }
                    }
                }

                _ = shutdown_rx.recv() => {
                    info!("IndicatorActor: Shutdown signal received");
                    break;
                }
            }
        }

        info!("IndicatorActor: Stopped");
        Ok(())
    }

    /// Process a single candle and update indicators
    async fn process_candle(
        storage: &Arc<StorageManager>,
        state: &Arc<RwLock<HashMap<StateKey, IndicatorState>>>,
        indicator_tx: &broadcast::Sender<IndicatorData>,
        candle: CandleInput,
    ) -> Result<()> {
        let start_time = std::time::Instant::now();

        let key = StateKey::new(&candle.symbol, &candle.timeframe);

        // Get or create indicator state
        let indicator_data = {
            let mut state_write = state.write().await;

            let indicator_state = state_write.entry(key.clone()).or_insert_with(|| {
                info!(
                    "IndicatorActor: Creating new indicator state for {}:{}",
                    candle.symbol, candle.timeframe
                );
                IndicatorState::new(
                    candle.symbol.clone(),
                    candle.exchange.clone(),
                    candle.timeframe.clone(),
                )
            });

            indicator_state.update(&candle)
        };

        // Update Prometheus metrics
        let calc_duration = start_time.elapsed();
        INDICATOR_CALCULATION_DURATION
            .with_label_values(&[&candle.symbol, &candle.timeframe])
            .observe(calc_duration.as_secs_f64());

        INDICATORS_CALCULATED
            .with_label_values(&[&candle.symbol, &candle.timeframe])
            .inc();

        // Update pairs tracked count
        let state_read = state.read().await;
        INDICATOR_PAIRS_TRACKED.set(state_read.len() as i64);
        drop(state_read);

        // Log calculated values at debug level
        debug!(
            "IndicatorActor: {}:{} @ {} - EMA8={:.2?}, EMA21={:.2?}, RSI={:.2?}, MACD={:.4?}",
            candle.symbol,
            candle.timeframe,
            candle.timestamp,
            indicator_data.ema_8,
            indicator_data.ema_21,
            indicator_data.rsi_14,
            indicator_data.macd_line,
        );

        // Broadcast to subscribers
        let _ = indicator_tx.send(indicator_data.clone());

        // Store to QuestDB
        Self::store_indicators(storage, &indicator_data).await?;

        Ok(())
    }

    /// Store indicator data to QuestDB
    async fn store_indicators(storage: &Arc<StorageManager>, data: &IndicatorData) -> Result<()> {
        let mut writer = storage.ilp_writer.lock().await;
        writer
            .write_indicator(data)
            .await
            .context("Failed to write indicator data to QuestDB")?;
        Ok(())
    }
}

/// Builder for creating IndicatorActor with custom configuration
pub struct IndicatorActorBuilder {
    config: Option<Arc<Config>>,
    storage: Option<Arc<StorageManager>>,
}

impl IndicatorActorBuilder {
    pub fn new() -> Self {
        Self {
            config: None,
            storage: None,
        }
    }

    pub fn config(mut self, config: Arc<Config>) -> Self {
        self.config = Some(config);
        self
    }

    pub fn storage(mut self, storage: Arc<StorageManager>) -> Self {
        self.storage = Some(storage);
        self
    }

    pub async fn build(self, shutdown_rx: broadcast::Receiver<()>) -> Result<Arc<IndicatorActor>> {
        let config = self.config.context("Config is required")?;
        let storage = self.storage.context("Storage is required")?;

        IndicatorActor::new(config, storage, shutdown_rx).await
    }
}

impl Default for IndicatorActorBuilder {
    fn default() -> Self {
        Self::new()
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_incremental_ema() {
        let mut ema = IncrementalEMA::new(3);

        // First 3 values: warmup
        assert!(ema.update(10.0).is_none());
        assert!(ema.update(11.0).is_none());

        // Third value initializes with SMA
        let v = ema.update(12.0);
        assert!(v.is_some());
        assert!((v.unwrap() - 11.0).abs() < 0.001); // SMA of 10, 11, 12

        // Fourth value uses EMA formula
        let v = ema.update(13.0);
        assert!(v.is_some());
        // EMA = 13 * 0.5 + 11 * 0.5 = 12
        assert!((v.unwrap() - 12.0).abs() < 0.001);
    }

    #[test]
    fn test_incremental_rsi() {
        let mut rsi = IncrementalRSI::new(14);

        // Feed some prices - RSI won't be ready until we have enough data
        let prices = vec![
            44.0, 44.34, 44.09, 43.61, 44.33, 44.83, 45.10, 45.42, 45.84, 46.08, 45.89, 46.03,
            45.61, 46.28, 46.28, 46.00, 46.03, 46.41,
        ];

        let mut last_rsi = None;
        for price in prices {
            last_rsi = rsi.update(price);
        }

        // RSI should be calculated after enough data
        assert!(last_rsi.is_some());
        let rsi_value = last_rsi.unwrap();
        assert!((0.0..=100.0).contains(&rsi_value));
    }

    #[test]
    fn test_incremental_atr() {
        let mut atr = IncrementalATR::new(3);

        // First candle: TR = high - low
        let v1 = atr.update(50.0, 48.0, 49.0);
        assert!(v1.is_none()); // Not ready yet

        // Second candle
        let v2 = atr.update(52.0, 49.0, 51.0);
        assert!(v2.is_none()); // Still warming up

        // Third candle - should initialize
        let v3 = atr.update(53.0, 50.0, 52.0);
        assert!(v3.is_some());
    }

    #[test]
    fn test_incremental_macd() {
        let mut macd = IncrementalMACD::new(3, 5, 2);

        // Feed prices until MACD is ready
        let prices = vec![10.0, 11.0, 12.0, 11.5, 12.5, 13.0, 12.0, 13.5];

        let mut last_macd = (None, None, None);
        for price in prices {
            last_macd = macd.update(price);
        }

        // MACD should have values after slow EMA (5 periods) + signal (2 periods)
        assert!(last_macd.0.is_some()); // MACD line
    }

    #[test]
    fn test_state_key() {
        let key1 = StateKey::new("BTC-USDT", "1m");
        let key2 = StateKey::new("BTC-USDT", "1m");
        let key3 = StateKey::new("ETH-USDT", "1m");

        assert_eq!(key1, key2);
        assert_ne!(key1, key3);
    }

    #[test]
    fn test_warmup_status() {
        let state = IndicatorState::new(
            "BTC-USDT".to_string(),
            "binance".to_string(),
            "1m".to_string(),
        );

        let status = state.warmup_status();
        assert!(!status.all_ready());
        assert_eq!(status.candles_processed, 0);
        assert_eq!(status.min_candles_needed(), 200);
    }
}
