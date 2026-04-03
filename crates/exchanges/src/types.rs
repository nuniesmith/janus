//! Common Types for Exchange Adapters
//!
//! This module defines shared types used across all exchange adapters.

use janus_core::{Exchange, Symbol};
use serde::{Deserialize, Serialize};
use std::time::Duration;

/// WebSocket connection configuration
#[derive(Debug, Clone)]
pub struct ConnectionConfig {
    /// WebSocket URL
    pub url: String,
    /// Connection timeout
    pub timeout: Duration,
    /// Reconnection attempts before giving up
    pub max_reconnect_attempts: u32,
    /// Initial reconnection delay
    pub reconnect_delay: Duration,
    /// Maximum reconnection delay (for exponential backoff)
    pub max_reconnect_delay: Duration,
    /// Ping interval for keepalive
    pub ping_interval: Duration,
    /// Pong timeout (if no pong received)
    pub pong_timeout: Duration,
}

impl ConnectionConfig {
    /// Create a new connection config with defaults
    pub fn new(url: String) -> Self {
        Self {
            url,
            timeout: Duration::from_secs(10),
            max_reconnect_attempts: 10,
            reconnect_delay: Duration::from_secs(1),
            max_reconnect_delay: Duration::from_secs(60),
            ping_interval: Duration::from_secs(30),
            pong_timeout: Duration::from_secs(10),
        }
    }

    /// Builder method for timeout
    pub fn with_timeout(mut self, timeout: Duration) -> Self {
        self.timeout = timeout;
        self
    }

    /// Builder method for reconnect attempts
    pub fn with_max_reconnect_attempts(mut self, attempts: u32) -> Self {
        self.max_reconnect_attempts = attempts;
        self
    }

    /// Builder method for ping interval
    pub fn with_ping_interval(mut self, interval: Duration) -> Self {
        self.ping_interval = interval;
        self
    }
}

impl Default for ConnectionConfig {
    fn default() -> Self {
        Self::new("wss://localhost:8443".to_string())
    }
}

/// Subscription request for market data
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct SubscriptionRequest {
    /// Exchange to subscribe to
    pub exchange: Exchange,
    /// Symbol to subscribe to
    pub symbol: Symbol,
    /// Channels to subscribe to
    pub channels: Vec<ChannelType>,
}

impl SubscriptionRequest {
    /// Create a new subscription request
    pub fn new(exchange: Exchange, symbol: Symbol, channels: Vec<ChannelType>) -> Self {
        Self {
            exchange,
            symbol,
            channels,
        }
    }

    /// Create request for trade channel only
    pub fn trades_only(exchange: Exchange, symbol: Symbol) -> Self {
        Self::new(exchange, symbol, vec![ChannelType::Trade])
    }

    /// Create request for all available channels
    pub fn all_channels(exchange: Exchange, symbol: Symbol) -> Self {
        Self::new(
            exchange,
            symbol,
            vec![
                ChannelType::Trade,
                ChannelType::Ticker,
                ChannelType::OrderBook,
            ],
        )
    }
}

/// Generic channel types across exchanges
#[derive(Debug, Clone, Copy, PartialEq, Eq, Serialize, Deserialize)]
#[serde(rename_all = "snake_case")]
pub enum ChannelType {
    /// Real-time trade executions
    Trade,
    /// 24-hour ticker statistics
    Ticker,
    /// Order book snapshots and updates
    OrderBook,
    /// Best bid/offer updates
    BestBidOffer,
    /// OHLC candlestick data
    Kline,
    /// Funding rate (for futures)
    FundingRate,
    /// Liquidation orders (for futures)
    Liquidation,
}

impl std::fmt::Display for ChannelType {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        match self {
            ChannelType::Trade => write!(f, "trade"),
            ChannelType::Ticker => write!(f, "ticker"),
            ChannelType::OrderBook => write!(f, "orderbook"),
            ChannelType::BestBidOffer => write!(f, "bbo"),
            ChannelType::Kline => write!(f, "kline"),
            ChannelType::FundingRate => write!(f, "funding_rate"),
            ChannelType::Liquidation => write!(f, "liquidation"),
        }
    }
}

/// Reconnection strategy
#[derive(Debug, Clone, Copy, PartialEq)]
pub enum ReconnectStrategy {
    /// Constant delay between reconnection attempts
    Constant(Duration),
    /// Exponential backoff (initial delay, multiplier, max delay)
    ExponentialBackoff {
        initial: Duration,
        multiplier: f64,
        max: Duration,
    },
    /// Linear backoff (initial delay, increment, max delay)
    LinearBackoff {
        initial: Duration,
        increment: Duration,
        max: Duration,
    },
}

impl ReconnectStrategy {
    /// Calculate delay for a given attempt number
    pub fn delay_for_attempt(&self, attempt: u32) -> Duration {
        match self {
            ReconnectStrategy::Constant(delay) => *delay,
            ReconnectStrategy::ExponentialBackoff {
                initial,
                multiplier,
                max,
            } => {
                let delay_secs =
                    initial.as_secs_f64() * multiplier.powi(attempt.saturating_sub(1) as i32);
                let delay = Duration::from_secs_f64(delay_secs);
                delay.min(*max)
            }
            ReconnectStrategy::LinearBackoff {
                initial,
                increment,
                max,
            } => {
                let delay = *initial + (*increment * attempt.saturating_sub(1));
                delay.min(*max)
            }
        }
    }
}

impl Default for ReconnectStrategy {
    fn default() -> Self {
        ReconnectStrategy::ExponentialBackoff {
            initial: Duration::from_secs(1),
            multiplier: 2.0,
            max: Duration::from_secs(60),
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_connection_config() {
        let config = ConnectionConfig::new("wss://example.com".to_string())
            .with_timeout(Duration::from_secs(5))
            .with_max_reconnect_attempts(5);

        assert_eq!(config.timeout, Duration::from_secs(5));
        assert_eq!(config.max_reconnect_attempts, 5);
    }

    #[test]
    fn test_subscription_request() {
        let symbol = Symbol::new("BTC", "USDT");
        let request = SubscriptionRequest::trades_only(Exchange::Binance, symbol);

        assert_eq!(request.exchange, Exchange::Binance);
        assert_eq!(request.channels.len(), 1);
        assert_eq!(request.channels[0], ChannelType::Trade);
    }

    #[test]
    fn test_constant_reconnect_strategy() {
        let strategy = ReconnectStrategy::Constant(Duration::from_secs(5));

        assert_eq!(strategy.delay_for_attempt(1), Duration::from_secs(5));
        assert_eq!(strategy.delay_for_attempt(10), Duration::from_secs(5));
    }

    #[test]
    fn test_exponential_backoff() {
        let strategy = ReconnectStrategy::ExponentialBackoff {
            initial: Duration::from_secs(1),
            multiplier: 2.0,
            max: Duration::from_secs(30),
        };

        assert_eq!(strategy.delay_for_attempt(1), Duration::from_secs(1));
        assert_eq!(strategy.delay_for_attempt(2), Duration::from_secs(2));
        assert_eq!(strategy.delay_for_attempt(3), Duration::from_secs(4));
        assert_eq!(strategy.delay_for_attempt(4), Duration::from_secs(8));
        assert_eq!(strategy.delay_for_attempt(5), Duration::from_secs(16));
        assert_eq!(strategy.delay_for_attempt(6), Duration::from_secs(30)); // Capped at max
    }

    #[test]
    fn test_linear_backoff() {
        let strategy = ReconnectStrategy::LinearBackoff {
            initial: Duration::from_secs(1),
            increment: Duration::from_secs(2),
            max: Duration::from_secs(10),
        };

        assert_eq!(strategy.delay_for_attempt(1), Duration::from_secs(1));
        assert_eq!(strategy.delay_for_attempt(2), Duration::from_secs(3));
        assert_eq!(strategy.delay_for_attempt(3), Duration::from_secs(5));
        assert_eq!(strategy.delay_for_attempt(4), Duration::from_secs(7));
        assert_eq!(strategy.delay_for_attempt(5), Duration::from_secs(9));
        assert_eq!(strategy.delay_for_attempt(6), Duration::from_secs(10)); // Capped
    }

    #[test]
    fn test_channel_type_display() {
        assert_eq!(ChannelType::Trade.to_string(), "trade");
        assert_eq!(ChannelType::Ticker.to_string(), "ticker");
        assert_eq!(ChannelType::OrderBook.to_string(), "orderbook");
    }
}
