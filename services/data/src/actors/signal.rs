//! Signal Generation Actor
//!
//! Generates trading signals based on technical indicator values.
//! Subscribes to indicator updates and emits signals for:
//!
//! - **EMA Crossovers**: Golden Cross (bullish), Death Cross (bearish)
//! - **RSI Thresholds**: Overbought (>70), Oversold (<30), Exit zones
//! - **MACD Crossovers**: Bullish/bearish signal line crossovers
//! - **Trend Signals**: Multi-indicator confluence signals
//!
//! ## Architecture
//!
//! The SignalActor subscribes to the IndicatorActor's broadcast channel
//! and processes each indicator update to detect signal conditions.
//! Generated signals are:
//! 1. Broadcast via a channel for real-time consumers
//! 2. Stored to QuestDB for historical analysis
//! 3. Exposed via Prometheus metrics
//!
//! ## Usage
//!
//! ```rust,ignore
//! let signal_actor = SignalActor::new(
//!     config,
//!     storage,
//!     indicator_rx,
//!     shutdown_rx,
//! ).await?;
//!
//! // Subscribe to signals
//! let mut signal_rx = signal_actor.subscribe();
//! while let Ok(signal) = signal_rx.recv().await {
//!     println!("Signal: {:?}", signal);
//! }
//! ```

use anyhow::Result;
use chrono::Utc;
use serde::{Deserialize, Serialize};
use std::collections::HashMap;
use std::sync::Arc;
use tokio::sync::{RwLock, broadcast};
use tracing::{debug, error, info, warn};

use crate::actors::indicator::IndicatorData;
use crate::config::Config;
use crate::storage::StorageManager;

/// Signal type enumeration
#[derive(Debug, Clone, Copy, PartialEq, Eq, Serialize, Deserialize)]
#[serde(rename_all = "snake_case")]
pub enum SignalType {
    /// EMA Golden Cross (short EMA crosses above long EMA)
    EmaGoldenCross,
    /// EMA Death Cross (short EMA crosses below long EMA)
    EmaDeathCross,
    /// RSI Overbought (RSI > 70)
    RsiOverbought,
    /// RSI Oversold (RSI < 30)
    RsiOversold,
    /// RSI Exit Overbought (RSI drops below 70)
    RsiExitOverbought,
    /// RSI Exit Oversold (RSI rises above 30)
    RsiExitOversold,
    /// MACD Bullish Crossover (MACD crosses above signal)
    MacdBullishCross,
    /// MACD Bearish Crossover (MACD crosses below signal)
    MacdBearishCross,
    /// Bullish Trend Confluence (multiple bullish indicators)
    BullishConfluence,
    /// Bearish Trend Confluence (multiple bearish indicators)
    BearishConfluence,
}

impl SignalType {
    /// Get the signal direction
    pub fn direction(&self) -> SignalDirection {
        match self {
            SignalType::EmaGoldenCross
            | SignalType::RsiOversold
            | SignalType::RsiExitOversold
            | SignalType::MacdBullishCross
            | SignalType::BullishConfluence => SignalDirection::Bullish,

            SignalType::EmaDeathCross
            | SignalType::RsiOverbought
            | SignalType::RsiExitOverbought
            | SignalType::MacdBearishCross
            | SignalType::BearishConfluence => SignalDirection::Bearish,
        }
    }

    /// Get signal strength (1-5)
    pub fn strength(&self) -> u8 {
        match self {
            SignalType::BullishConfluence | SignalType::BearishConfluence => 5,
            SignalType::EmaGoldenCross | SignalType::EmaDeathCross => 4,
            SignalType::MacdBullishCross | SignalType::MacdBearishCross => 3,
            SignalType::RsiOverbought | SignalType::RsiOversold => 2,
            SignalType::RsiExitOverbought | SignalType::RsiExitOversold => 1,
        }
    }

    /// Get signal name as string
    pub fn as_str(&self) -> &'static str {
        match self {
            SignalType::EmaGoldenCross => "ema_golden_cross",
            SignalType::EmaDeathCross => "ema_death_cross",
            SignalType::RsiOverbought => "rsi_overbought",
            SignalType::RsiOversold => "rsi_oversold",
            SignalType::RsiExitOverbought => "rsi_exit_overbought",
            SignalType::RsiExitOversold => "rsi_exit_oversold",
            SignalType::MacdBullishCross => "macd_bullish_cross",
            SignalType::MacdBearishCross => "macd_bearish_cross",
            SignalType::BullishConfluence => "bullish_confluence",
            SignalType::BearishConfluence => "bearish_confluence",
        }
    }
}

/// Signal direction
#[derive(Debug, Clone, Copy, PartialEq, Eq, Serialize, Deserialize)]
#[serde(rename_all = "lowercase")]
pub enum SignalDirection {
    Bullish,
    Bearish,
}

impl SignalDirection {
    pub fn as_str(&self) -> &'static str {
        match self {
            SignalDirection::Bullish => "bullish",
            SignalDirection::Bearish => "bearish",
        }
    }
}

/// A trading signal
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct Signal {
    /// Symbol (e.g., "BTCUSD")
    pub symbol: String,
    /// Exchange (e.g., "binance")
    pub exchange: String,
    /// Timeframe (e.g., "1m", "5m")
    pub timeframe: String,
    /// Signal type
    pub signal_type: SignalType,
    /// Signal direction
    pub direction: SignalDirection,
    /// Signal strength (1-5)
    pub strength: u8,
    /// Current price when signal was generated
    pub price: f64,
    /// Indicator value that triggered the signal
    pub trigger_value: f64,
    /// Optional secondary trigger value
    pub trigger_value_2: Option<f64>,
    /// Timestamp when signal was generated (ms)
    pub timestamp: i64,
    /// Additional context/description
    pub description: String,
}

impl Signal {
    /// Create a new signal
    pub fn new(
        symbol: String,
        exchange: String,
        timeframe: String,
        signal_type: SignalType,
        price: f64,
        trigger_value: f64,
        description: String,
    ) -> Self {
        Self {
            symbol,
            exchange,
            timeframe,
            direction: signal_type.direction(),
            strength: signal_type.strength(),
            signal_type,
            price,
            trigger_value,
            trigger_value_2: None,
            timestamp: Utc::now().timestamp_millis(),
            description,
        }
    }

    /// Add a secondary trigger value
    pub fn with_trigger_value_2(mut self, value: f64) -> Self {
        self.trigger_value_2 = Some(value);
        self
    }
}

/// Signal generation configuration
#[derive(Debug, Clone)]
pub struct SignalConfig {
    /// RSI overbought threshold (default: 70)
    pub rsi_overbought: f64,
    /// RSI oversold threshold (default: 30)
    pub rsi_oversold: f64,
    /// RSI hysteresis (default: 2) - prevents signal flapping
    pub rsi_hysteresis: f64,
    /// Minimum EMA separation for crossover (percentage, default: 0.01%)
    pub ema_min_separation_pct: f64,
    /// Enable confluence signals
    pub enable_confluence: bool,
    /// Minimum indicators for confluence (default: 3)
    pub confluence_min_indicators: usize,
}

impl Default for SignalConfig {
    fn default() -> Self {
        Self {
            rsi_overbought: 70.0,
            rsi_oversold: 30.0,
            rsi_hysteresis: 2.0,
            ema_min_separation_pct: 0.01,
            enable_confluence: true,
            confluence_min_indicators: 3,
        }
    }
}

/// Previous indicator state for crossover detection
#[derive(Debug, Clone, Default)]
struct PreviousState {
    ema_8: Option<f64>,
    ema_21: Option<f64>,
    ema_50: Option<f64>,
    rsi_14: Option<f64>,
    macd_line: Option<f64>,
    macd_signal: Option<f64>,
    /// Track active RSI zones for exit signals
    in_overbought: bool,
    in_oversold: bool,
}

/// State key for tracking per-symbol/timeframe
#[derive(Debug, Clone, Hash, PartialEq, Eq)]
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

/// Signal generation actor
pub struct SignalActor {
    #[allow(dead_code)]
    config: Arc<Config>,
    signal_config: SignalConfig,
    storage: Arc<StorageManager>,
    /// Previous state for each symbol/timeframe
    state: Arc<RwLock<HashMap<StateKey, PreviousState>>>,
    /// Broadcast sender for signal updates
    pub signal_tx: broadcast::Sender<Signal>,
    /// Statistics
    signals_generated: Arc<RwLock<u64>>,
}

impl SignalActor {
    /// Create a new SignalActor
    pub async fn new(
        config: Arc<Config>,
        storage: Arc<StorageManager>,
        mut indicator_rx: broadcast::Receiver<IndicatorData>,
        mut shutdown_rx: broadcast::Receiver<()>,
    ) -> Result<Arc<Self>> {
        let (signal_tx, _) = broadcast::channel(1000);
        let state = Arc::new(RwLock::new(HashMap::new()));
        let signals_generated = Arc::new(RwLock::new(0u64));

        let actor = Arc::new(Self {
            config,
            signal_config: SignalConfig::default(),
            storage,
            state: state.clone(),
            signal_tx: signal_tx.clone(),
            signals_generated: signals_generated.clone(),
        });

        // Spawn processing loop
        let state_clone = state.clone();
        let signal_tx_clone = signal_tx.clone();
        let storage_clone = actor.storage.clone();
        let signal_config = actor.signal_config.clone();
        let signals_generated_clone = signals_generated.clone();

        tokio::spawn(async move {
            info!("SignalActor: Starting processing loop");

            let mut stats_interval = tokio::time::interval(std::time::Duration::from_secs(60));

            loop {
                tokio::select! {
                    _ = shutdown_rx.recv() => {
                        info!("SignalActor: Shutdown signal received");
                        break;
                    }

                    Ok(indicator_data) = indicator_rx.recv() => {
                        if let Err(e) = Self::process_indicator(
                            &indicator_data,
                            &signal_config,
                            &state_clone,
                            &signal_tx_clone,
                            &storage_clone,
                            &signals_generated_clone,
                        ).await {
                            error!("SignalActor: Error processing indicator: {}", e);
                        }
                    }

                    _ = stats_interval.tick() => {
                        let count = *signals_generated_clone.read().await;
                        let pairs = state_clone.read().await.len();
                        info!("SignalActor Stats - Tracked: {} pairs, Signals generated: {}", pairs, count);
                    }
                }
            }

            info!("SignalActor: Processing loop ended");
        });

        info!("SignalActor: Initialized and running");
        Ok(actor)
    }

    /// Subscribe to signal updates
    pub fn subscribe(&self) -> broadcast::Receiver<Signal> {
        self.signal_tx.subscribe()
    }

    /// Get total signals generated
    pub async fn get_signals_count(&self) -> u64 {
        *self.signals_generated.read().await
    }

    /// Process an indicator update and generate signals
    async fn process_indicator(
        indicator: &IndicatorData,
        config: &SignalConfig,
        state: &Arc<RwLock<HashMap<StateKey, PreviousState>>>,
        signal_tx: &broadcast::Sender<Signal>,
        storage: &Arc<StorageManager>,
        signals_generated: &Arc<RwLock<u64>>,
    ) -> Result<()> {
        let key = StateKey::new(&indicator.symbol, &indicator.timeframe);
        let mut state_guard = state.write().await;

        let prev = state_guard.entry(key.clone()).or_default();
        let mut signals = Vec::new();

        // Check EMA crossovers (8/21)
        if let (Some(ema_8), Some(ema_21)) = (indicator.ema_8, indicator.ema_21) {
            if let (Some(prev_ema_8), Some(prev_ema_21)) = (prev.ema_8, prev.ema_21) {
                // Golden Cross: EMA-8 crosses above EMA-21
                if prev_ema_8 <= prev_ema_21 && ema_8 > ema_21 {
                    let separation_pct = ((ema_8 - ema_21) / ema_21 * 100.0).abs();
                    if separation_pct >= config.ema_min_separation_pct {
                        signals.push(
                            Signal::new(
                                indicator.symbol.clone(),
                                indicator.exchange.clone(),
                                indicator.timeframe.clone(),
                                SignalType::EmaGoldenCross,
                                indicator.close,
                                ema_8,
                                format!(
                                    "EMA-8 ({:.2}) crossed above EMA-21 ({:.2})",
                                    ema_8, ema_21
                                ),
                            )
                            .with_trigger_value_2(ema_21),
                        );
                    }
                }

                // Death Cross: EMA-8 crosses below EMA-21
                if prev_ema_8 >= prev_ema_21 && ema_8 < ema_21 {
                    let separation_pct = ((ema_21 - ema_8) / ema_21 * 100.0).abs();
                    if separation_pct >= config.ema_min_separation_pct {
                        signals.push(
                            Signal::new(
                                indicator.symbol.clone(),
                                indicator.exchange.clone(),
                                indicator.timeframe.clone(),
                                SignalType::EmaDeathCross,
                                indicator.close,
                                ema_8,
                                format!(
                                    "EMA-8 ({:.2}) crossed below EMA-21 ({:.2})",
                                    ema_8, ema_21
                                ),
                            )
                            .with_trigger_value_2(ema_21),
                        );
                    }
                }
            }

            prev.ema_8 = Some(ema_8);
            prev.ema_21 = Some(ema_21);
        }

        // Check RSI signals
        if let Some(rsi) = indicator.rsi_14 {
            // Overbought entry
            if rsi > config.rsi_overbought && !prev.in_overbought {
                signals.push(Signal::new(
                    indicator.symbol.clone(),
                    indicator.exchange.clone(),
                    indicator.timeframe.clone(),
                    SignalType::RsiOverbought,
                    indicator.close,
                    rsi,
                    format!(
                        "RSI ({:.1}) entered overbought zone (>{:.0})",
                        rsi, config.rsi_overbought
                    ),
                ));
                prev.in_overbought = true;
            }

            // Overbought exit (with hysteresis)
            if prev.in_overbought && rsi < (config.rsi_overbought - config.rsi_hysteresis) {
                signals.push(Signal::new(
                    indicator.symbol.clone(),
                    indicator.exchange.clone(),
                    indicator.timeframe.clone(),
                    SignalType::RsiExitOverbought,
                    indicator.close,
                    rsi,
                    format!("RSI ({:.1}) exited overbought zone", rsi),
                ));
                prev.in_overbought = false;
            }

            // Oversold entry
            if rsi < config.rsi_oversold && !prev.in_oversold {
                signals.push(Signal::new(
                    indicator.symbol.clone(),
                    indicator.exchange.clone(),
                    indicator.timeframe.clone(),
                    SignalType::RsiOversold,
                    indicator.close,
                    rsi,
                    format!(
                        "RSI ({:.1}) entered oversold zone (<{:.0})",
                        rsi, config.rsi_oversold
                    ),
                ));
                prev.in_oversold = true;
            }

            // Oversold exit (with hysteresis)
            if prev.in_oversold && rsi > (config.rsi_oversold + config.rsi_hysteresis) {
                signals.push(Signal::new(
                    indicator.symbol.clone(),
                    indicator.exchange.clone(),
                    indicator.timeframe.clone(),
                    SignalType::RsiExitOversold,
                    indicator.close,
                    rsi,
                    format!("RSI ({:.1}) exited oversold zone", rsi),
                ));
                prev.in_oversold = false;
            }

            prev.rsi_14 = Some(rsi);
        }

        // Check MACD crossovers
        if let (Some(macd_line), Some(macd_signal)) = (indicator.macd_line, indicator.macd_signal) {
            if let (Some(prev_macd_line), Some(prev_macd_signal)) =
                (prev.macd_line, prev.macd_signal)
            {
                // Bullish crossover: MACD crosses above signal
                if prev_macd_line <= prev_macd_signal && macd_line > macd_signal {
                    signals.push(
                        Signal::new(
                            indicator.symbol.clone(),
                            indicator.exchange.clone(),
                            indicator.timeframe.clone(),
                            SignalType::MacdBullishCross,
                            indicator.close,
                            macd_line,
                            format!(
                                "MACD ({:.4}) crossed above signal ({:.4})",
                                macd_line, macd_signal
                            ),
                        )
                        .with_trigger_value_2(macd_signal),
                    );
                }

                // Bearish crossover: MACD crosses below signal
                if prev_macd_line >= prev_macd_signal && macd_line < macd_signal {
                    signals.push(
                        Signal::new(
                            indicator.symbol.clone(),
                            indicator.exchange.clone(),
                            indicator.timeframe.clone(),
                            SignalType::MacdBearishCross,
                            indicator.close,
                            macd_line,
                            format!(
                                "MACD ({:.4}) crossed below signal ({:.4})",
                                macd_line, macd_signal
                            ),
                        )
                        .with_trigger_value_2(macd_signal),
                    );
                }
            }

            prev.macd_line = Some(macd_line);
            prev.macd_signal = Some(macd_signal);
        }

        // Update EMA-50 state
        if let Some(ema_50) = indicator.ema_50 {
            prev.ema_50 = Some(ema_50);
        }

        // Check for confluence signals
        if config.enable_confluence && !signals.is_empty() {
            let bullish_count = signals
                .iter()
                .filter(|s| s.direction == SignalDirection::Bullish)
                .count();

            let bearish_count = signals
                .iter()
                .filter(|s| s.direction == SignalDirection::Bearish)
                .count();

            // Collect bullish signal types before pushing new signals
            if bullish_count >= config.confluence_min_indicators {
                let descriptions: Vec<_> = signals
                    .iter()
                    .filter(|s| s.direction == SignalDirection::Bullish)
                    .map(|s| s.signal_type.as_str())
                    .collect();

                signals.push(Signal::new(
                    indicator.symbol.clone(),
                    indicator.exchange.clone(),
                    indicator.timeframe.clone(),
                    SignalType::BullishConfluence,
                    indicator.close,
                    bullish_count as f64,
                    format!(
                        "Bullish confluence: {} signals ({})",
                        bullish_count,
                        descriptions.join(", ")
                    ),
                ));
            }

            // Collect bearish signal types before pushing new signals
            if bearish_count >= config.confluence_min_indicators {
                let descriptions: Vec<_> = signals
                    .iter()
                    .filter(|s| s.direction == SignalDirection::Bearish)
                    .map(|s| s.signal_type.as_str())
                    .collect();

                signals.push(Signal::new(
                    indicator.symbol.clone(),
                    indicator.exchange.clone(),
                    indicator.timeframe.clone(),
                    SignalType::BearishConfluence,
                    indicator.close,
                    bearish_count as f64,
                    format!(
                        "Bearish confluence: {} signals ({})",
                        bearish_count,
                        descriptions.join(", ")
                    ),
                ));
            }
        }

        // Broadcast and store signals
        for signal in signals {
            debug!(
                "SignalActor: Generated {} signal for {}:{} @ {}",
                signal.signal_type.as_str(),
                signal.symbol,
                signal.timeframe,
                signal.price
            );

            // Update metrics
            crate::metrics::prometheus_exporter::record_signal_generated(
                &signal.symbol,
                &signal.timeframe,
                signal.signal_type.as_str(),
                signal.direction.as_str(),
            );

            // Broadcast signal
            if signal_tx.receiver_count() > 0 {
                let _ = signal_tx.send(signal.clone());
            }

            // Store to QuestDB
            if let Err(e) = Self::store_signal(storage, &signal).await {
                warn!("SignalActor: Failed to store signal: {}", e);
            }

            // Update counter
            let mut count = signals_generated.write().await;
            *count += 1;
        }

        Ok(())
    }

    /// Store a signal to QuestDB
    async fn store_signal(storage: &Arc<StorageManager>, signal: &Signal) -> Result<()> {
        // Format as ILP line
        let line = format!(
            "signals_crypto,symbol={},exchange={},timeframe={},signal_type={},direction={} \
             strength={}i,price={},trigger_value={}{} {}000000\n",
            signal.symbol,
            signal.exchange,
            signal.timeframe,
            signal.signal_type.as_str(),
            signal.direction.as_str(),
            signal.strength,
            signal.price,
            signal.trigger_value,
            signal
                .trigger_value_2
                .map(|v| format!(",trigger_value_2={}", v))
                .unwrap_or_default(),
            signal.timestamp,
        );

        storage.write_raw_ilp(&line).await
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_signal_type_direction() {
        assert_eq!(
            SignalType::EmaGoldenCross.direction(),
            SignalDirection::Bullish
        );
        assert_eq!(
            SignalType::EmaDeathCross.direction(),
            SignalDirection::Bearish
        );
        assert_eq!(
            SignalType::RsiOversold.direction(),
            SignalDirection::Bullish
        );
        assert_eq!(
            SignalType::RsiOverbought.direction(),
            SignalDirection::Bearish
        );
    }

    #[test]
    fn test_signal_type_strength() {
        assert_eq!(SignalType::BullishConfluence.strength(), 5);
        assert_eq!(SignalType::EmaGoldenCross.strength(), 4);
        assert_eq!(SignalType::MacdBullishCross.strength(), 3);
        assert_eq!(SignalType::RsiOverbought.strength(), 2);
        assert_eq!(SignalType::RsiExitOverbought.strength(), 1);
    }

    #[test]
    fn test_signal_creation() {
        let signal = Signal::new(
            "BTCUSD".to_string(),
            "binance".to_string(),
            "1m".to_string(),
            SignalType::EmaGoldenCross,
            50000.0,
            50100.0,
            "Test signal".to_string(),
        );

        assert_eq!(signal.symbol, "BTCUSD");
        assert_eq!(signal.direction, SignalDirection::Bullish);
        assert_eq!(signal.strength, 4);
        assert!(signal.trigger_value_2.is_none());
    }

    #[test]
    fn test_signal_with_trigger_value_2() {
        let signal = Signal::new(
            "BTCUSD".to_string(),
            "binance".to_string(),
            "1m".to_string(),
            SignalType::EmaGoldenCross,
            50000.0,
            50100.0,
            "Test signal".to_string(),
        )
        .with_trigger_value_2(49900.0);

        assert_eq!(signal.trigger_value_2, Some(49900.0));
    }

    #[test]
    fn test_signal_config_default() {
        let config = SignalConfig::default();
        assert_eq!(config.rsi_overbought, 70.0);
        assert_eq!(config.rsi_oversold, 30.0);
        assert!(config.enable_confluence);
    }

    #[test]
    fn test_state_key() {
        let key1 = StateKey::new("BTCUSD", "1m");
        let key2 = StateKey::new("BTCUSD", "1m");
        let key3 = StateKey::new("ETHUSDT", "1m");

        assert_eq!(key1, key2);
        assert_ne!(key1, key3);
    }

    #[test]
    fn test_signal_serialization() {
        let signal = Signal::new(
            "BTCUSD".to_string(),
            "binance".to_string(),
            "1m".to_string(),
            SignalType::RsiOverbought,
            50000.0,
            75.5,
            "RSI overbought".to_string(),
        );

        let json = serde_json::to_string(&signal).unwrap();
        assert!(json.contains("rsi_overbought"));
        assert!(json.contains("bearish"));
    }
}
