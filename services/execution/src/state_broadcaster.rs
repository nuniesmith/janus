//! State Broadcaster for Execution Service
//!
//! Broadcasts execution state (equity, positions, volatility) to Redis
//! for consumption by other services (Brain, UI, etc.) without requiring
//! blocking HTTP calls.
//!
//! This eliminates the "HTTP blocking in event loop" issue identified in
//! the 168-hour soak test audit.

use redis::aio::ConnectionManager;
use serde::{Deserialize, Serialize};
use std::sync::Arc;
use tokio::sync::RwLock;
use tokio::time::{Duration, interval};
use tracing::{debug, error, info};

/// State that gets broadcast to Redis
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct ExecutionState {
    /// Current account equity
    pub equity: f64,

    /// Available balance (equity - used margin)
    pub available_balance: f64,

    /// Current volatility estimate
    pub volatility: VolatilityEstimate,

    /// Number of open positions
    pub open_positions: usize,

    /// Total unrealized P&L
    pub unrealized_pnl: f64,

    /// Timestamp (UTC milliseconds)
    pub timestamp: i64,
}

/// Volatility estimate with regime classification
#[derive(Debug, Clone, Copy, Serialize, Deserialize, PartialEq)]
pub struct VolatilityEstimate {
    /// ATR percentage (0-100)
    pub value: f64,

    /// Volatility regime
    pub regime: VolatilityRegime,

    /// Confidence (0.0-1.0)
    pub confidence: f64,
}

/// Volatility regime classification
#[derive(Debug, Clone, Copy, Serialize, Deserialize, PartialEq, Eq)]
pub enum VolatilityRegime {
    /// ATR < 1.0%
    Low,

    /// 1.0% <= ATR < 2.5%
    Normal,

    /// 2.5% <= ATR < 5.0%
    High,

    /// ATR >= 5.0%
    Extreme,
}

impl VolatilityRegime {
    /// Classify volatility from ATR percentage
    pub fn from_atr_pct(atr_pct: f64) -> Self {
        match atr_pct {
            x if x < 1.0 => Self::Low,
            x if x < 2.5 => Self::Normal,
            x if x < 5.0 => Self::High,
            _ => Self::Extreme,
        }
    }
}

/// Shared state container (thread-safe)
#[derive(Debug, Clone)]
pub struct SharedExecutionState {
    inner: Arc<RwLock<ExecutionState>>,
}

impl SharedExecutionState {
    /// Create new shared state with default values
    pub fn new(initial_equity: f64) -> Self {
        let state = ExecutionState {
            equity: initial_equity,
            available_balance: initial_equity,
            volatility: VolatilityEstimate {
                value: 1.5,
                regime: VolatilityRegime::Normal,
                confidence: 0.5,
            },
            open_positions: 0,
            unrealized_pnl: 0.0,
            timestamp: chrono::Utc::now().timestamp_millis(),
        };

        Self {
            inner: Arc::new(RwLock::new(state)),
        }
    }

    /// Update equity
    pub async fn set_equity(&self, equity: f64, available: f64) {
        let mut state = self.inner.write().await;
        state.equity = equity;
        state.available_balance = available;
        state.timestamp = chrono::Utc::now().timestamp_millis();
    }

    /// Update volatility
    pub async fn set_volatility(&self, estimate: VolatilityEstimate) {
        let mut state = self.inner.write().await;
        state.volatility = estimate;
        state.timestamp = chrono::Utc::now().timestamp_millis();
    }

    /// Update position count and P&L
    pub async fn set_positions(&self, count: usize, unrealized_pnl: f64) {
        let mut state = self.inner.write().await;
        state.open_positions = count;
        state.unrealized_pnl = unrealized_pnl;
        state.timestamp = chrono::Utc::now().timestamp_millis();
    }

    /// Get current state snapshot
    pub async fn get(&self) -> ExecutionState {
        self.inner.read().await.clone()
    }
}

/// State broadcaster configuration
#[derive(Debug, Clone)]
pub struct BroadcasterConfig {
    /// Broadcast interval (default: 100ms = 10Hz)
    pub interval: Duration,

    /// Redis channel prefix
    pub channel_prefix: String,

    /// Enable detailed logging
    pub verbose: bool,
}

impl Default for BroadcasterConfig {
    fn default() -> Self {
        Self {
            interval: Duration::from_millis(100), // 10Hz
            channel_prefix: "janus.state".to_string(),
            verbose: false,
        }
    }
}

/// State broadcaster - publishes execution state to Redis
pub struct StateBroadcaster {
    redis: ConnectionManager,
    state: SharedExecutionState,
    config: BroadcasterConfig,
}

impl StateBroadcaster {
    /// Create new broadcaster
    pub fn new(
        redis: ConnectionManager,
        state: SharedExecutionState,
        config: BroadcasterConfig,
    ) -> Self {
        Self {
            redis,
            state,
            config,
        }
    }

    /// Start broadcasting (runs forever)
    pub async fn run(mut self) -> Result<(), Box<dyn std::error::Error>> {
        info!(
            "Starting state broadcaster (interval: {:?})",
            self.config.interval
        );

        let mut ticker = interval(self.config.interval);

        loop {
            ticker.tick().await;

            // Get current state
            let state = self.state.get().await;

            // Broadcast full state
            if let Err(e) = self.broadcast_state(&state).await {
                error!("Failed to broadcast state: {}", e);
                continue;
            }

            // Also broadcast individual components for targeted subscriptions
            if let Err(e) = self.broadcast_equity(&state).await {
                error!("Failed to broadcast equity: {}", e);
            }

            if let Err(e) = self.broadcast_volatility(&state).await {
                error!("Failed to broadcast volatility: {}", e);
            }

            if self.config.verbose {
                debug!(
                    "Broadcast state: equity={:.2}, volatility={:.2}% ({}), positions={}",
                    state.equity,
                    state.volatility.value,
                    format!("{:?}", state.volatility.regime).to_uppercase(),
                    state.open_positions
                );
            }
        }
    }

    /// Broadcast full state
    async fn broadcast_state(&mut self, state: &ExecutionState) -> Result<(), redis::RedisError> {
        let channel = format!("{}.full", self.config.channel_prefix);
        let payload = serde_json::to_string(state).map_err(|e| {
            redis::RedisError::from((
                redis::ErrorKind::Io,
                "JSON serialization failed",
                e.to_string(),
            ))
        })?;

        redis::cmd("PUBLISH")
            .arg(&channel)
            .arg(payload)
            .query_async(&mut self.redis)
            .await
    }

    /// Broadcast equity only
    async fn broadcast_equity(&mut self, state: &ExecutionState) -> Result<(), redis::RedisError> {
        let channel = format!("{}.equity", self.config.channel_prefix);
        let payload = serde_json::json!({
            "equity": state.equity,
            "available_balance": state.available_balance,
            "unrealized_pnl": state.unrealized_pnl,
            "timestamp": state.timestamp,
        });

        redis::cmd("PUBLISH")
            .arg(&channel)
            .arg(payload.to_string())
            .query_async(&mut self.redis)
            .await
    }

    /// Broadcast volatility only
    async fn broadcast_volatility(
        &mut self,
        state: &ExecutionState,
    ) -> Result<(), redis::RedisError> {
        let channel = format!("{}.volatility", self.config.channel_prefix);
        let payload = serde_json::to_string(&state.volatility).map_err(|e| {
            redis::RedisError::from((
                redis::ErrorKind::Io,
                "JSON serialization failed",
                e.to_string(),
            ))
        })?;

        redis::cmd("PUBLISH")
            .arg(&channel)
            .arg(payload)
            .query_async(&mut self.redis)
            .await
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[tokio::test]
    async fn test_shared_state_creation() {
        let state = SharedExecutionState::new(10000.0);
        let snapshot = state.get().await;

        assert_eq!(snapshot.equity, 10000.0);
        assert_eq!(snapshot.available_balance, 10000.0);
        assert_eq!(snapshot.volatility.regime, VolatilityRegime::Normal);
    }

    #[tokio::test]
    async fn test_equity_update() {
        let state = SharedExecutionState::new(10000.0);
        state.set_equity(12000.0, 11000.0).await;

        let snapshot = state.get().await;
        assert_eq!(snapshot.equity, 12000.0);
        assert_eq!(snapshot.available_balance, 11000.0);
    }

    #[tokio::test]
    async fn test_volatility_update() {
        let state = SharedExecutionState::new(10000.0);

        let vol = VolatilityEstimate {
            value: 3.5,
            regime: VolatilityRegime::High,
            confidence: 0.9,
        };

        state.set_volatility(vol).await;

        let snapshot = state.get().await;
        assert_eq!(snapshot.volatility.value, 3.5);
        assert_eq!(snapshot.volatility.regime, VolatilityRegime::High);
    }

    #[test]
    fn test_volatility_regime_classification() {
        assert_eq!(VolatilityRegime::from_atr_pct(0.5), VolatilityRegime::Low);
        assert_eq!(
            VolatilityRegime::from_atr_pct(1.5),
            VolatilityRegime::Normal
        );
        assert_eq!(VolatilityRegime::from_atr_pct(3.5), VolatilityRegime::High);
        assert_eq!(
            VolatilityRegime::from_atr_pct(6.0),
            VolatilityRegime::Extreme
        );
    }

    #[test]
    fn test_serialization() {
        let state = ExecutionState {
            equity: 10000.0,
            available_balance: 9500.0,
            volatility: VolatilityEstimate {
                value: 2.1,
                regime: VolatilityRegime::Normal,
                confidence: 0.85,
            },
            open_positions: 2,
            unrealized_pnl: 150.0,
            timestamp: 1234567890,
        };

        let json = serde_json::to_string(&state).unwrap();
        let deserialized: ExecutionState = serde_json::from_str(&json).unwrap();

        assert_eq!(deserialized.equity, state.equity);
        assert_eq!(deserialized.volatility.regime, state.volatility.regime);
    }
}
