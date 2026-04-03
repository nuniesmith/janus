# JANUS Exchanges

Unified cryptocurrency exchange adapters for market data ingestion.

## Overview

This crate provides a consistent interface for connecting to multiple cryptocurrency exchanges via WebSocket, normalizing their disparate message formats into JANUS's unified `MarketDataEvent` types.

## Supported Exchanges

| Exchange | Feature Flag | Markets | WebSocket API Version | Status |
|----------|--------------|---------|----------------------|---------|
| Coinbase | `coinbase`   | Spot | Advanced Trade | ✅ Complete |
| Kraken   | `kraken`     | Spot, Futures | v2 | ✅ Complete |
| OKX      | `okx`        | Spot, Swap, Futures | v5 | ✅ Complete |
| Binance  | `binance`    | Spot, Futures | v3 | 📋 Planned (migration) |
| Bybit    | `bybit`      | Spot, Derivatives | V5 | 📋 Planned (migration) |
| Kucoin   | `kucoin`     | Spot | v2 | 📋 Planned (migration) |

## Features

- **Unified API**: All exchanges expose the same interface
- **Type Safety**: Exchange-specific messages converted to type-safe Rust structs
- **Event Normalization**: All data flows through `MarketDataEvent` enum
- **Health Monitoring**: Built-in health checking and metrics
- **Extensible**: Easy to add new exchanges
- **Feature Flags**: Enable only the exchanges you need

## Installation

Add to your `Cargo.toml`:

```toml
[dependencies]
janus-exchanges = { path = "../exchanges" }
janus-core = { path = "../../lib/janus-core" }
```

Or enable specific exchanges:

```toml
[dependencies]
janus-exchanges = { path = "../exchanges", features = ["coinbase", "kraken"] }
```

## Quick Start

### Basic Usage

```rust
use janus_exchanges::adapters::coinbase::CoinbaseAdapter;
use janus_core::{Symbol, MarketDataEvent};

#[tokio::main]
async fn main() -> anyhow::Result<()> {
    // Create adapter
    let adapter = CoinbaseAdapter::new();
    
    // Build subscription message
    let symbol = Symbol::new("BTC", "USD");
    let subscribe_msg = adapter.default_subscribe(&symbol);
    
    // Connect to WebSocket (using tokio-tungstenite)
    let (mut ws_stream, _) = tokio_tungstenite::connect_async(adapter.ws_url()).await?;
    
    // Send subscription
    use futures_util::SinkExt;
    ws_stream.send(subscribe_msg.into()).await?;
    
    // Process incoming messages
    use futures_util::StreamExt;
    while let Some(msg) = ws_stream.next().await {
        let msg = msg?;
        if let Ok(text) = msg.to_text() {
            let events = adapter.parse_message(text)?;
            
            for event in events {
                match event {
                    MarketDataEvent::Trade(trade) => {
                        println!("Trade: {} @ {}", trade.quantity, trade.price);
                    }
                    MarketDataEvent::Ticker(ticker) => {
                        println!("Ticker: {}", ticker.last_price);
                    }
                    _ => {}
                }
            }
        }
    }
    
    Ok(())
}
```

### All Exchanges Example

```rust
use janus_exchanges::adapters::*;
use janus_core::{Exchange, Symbol};

fn get_adapter(exchange: Exchange) -> Box<dyn ExchangeAdapter> {
    match exchange {
        Exchange::Binance => Box::new(binance::BinanceAdapter::new()),
        Exchange::Bybit => Box::new(bybit::BybitAdapter::new()),
        Exchange::Coinbase => Box::new(coinbase::CoinbaseAdapter::new()),
        Exchange::Kraken => Box::new(kraken::KrakenAdapter::new()),
        Exchange::Okx => Box::new(okx::OkxAdapter::new()),
        Exchange::Kucoin => Box::new(kucoin::KucoinAdapter::new()),
    }
}
```

### Health Monitoring

```rust
use janus_exchanges::HealthChecker;
use std::time::Duration;

#[tokio::main]
async fn main() {
    let health_checker = HealthChecker::new();
    
    // Record successful message
    health_checker.record_message("binance", Some(5.2)).await;
    
    // Check health
    if let Some(health) = health_checker.get_health("binance").await {
        println!("Status: {}", health.status);
        println!("Latency: {:.2}ms", health.avg_latency_ms);
        println!("Messages: {}", health.total_messages);
    }
    
    // Get overall system health
    let score = health_checker.system_health_score().await;
    println!("System health: {:.2}", score);
}
```

## Architecture

```
┌─────────────────────────────────────────────────────────┐
│                   Exchange Adapters                      │
├──────────┬──────────┬──────────┬──────────┬──────────────┤
│ Binance  │  Bybit   │ Coinbase │  Kraken  │  OKX  Kucoin │
└────┬─────┴────┬─────┴────┬─────┴────┬─────┴────┬────┬───┘
     │          │          │          │          │    │
     └──────────┴──────────┴──────────┴──────────┴────┘
                          │
                   Unified Events
                          │
     ┌────────────────────┴────────────────────┐
     │         MarketDataEvent                 │
     ├─────────────────────────────────────────┤
     │ • Trade                                 │
     │ • OrderBook                             │
     │ • Ticker                                │
     │ • Liquidation                           │
     │ • FundingRate                           │
     │ • Kline                                 │
     └─────────────────────────────────────────┘
```

## Event Types

### TradeEvent
Real-time trade executions with price, quantity, side, and timestamp.

```rust
pub struct TradeEvent {
    pub exchange: Exchange,
    pub symbol: Symbol,
    pub timestamp: i64,          // Unix microseconds
    pub received_at: i64,        // Reception timestamp
    pub price: Decimal,
    pub quantity: Decimal,
    pub side: Side,
    pub trade_id: String,
    pub buyer_is_maker: Option<bool>,
}
```

### OrderBookEvent
Full order book snapshots or incremental updates.

```rust
pub struct OrderBookEvent {
    pub exchange: Exchange,
    pub symbol: Symbol,
    pub timestamp: i64,
    pub sequence: u64,
    pub is_snapshot: bool,
    pub bids: Vec<PriceLevel>,
    pub asks: Vec<PriceLevel>,
}
```

### TickerEvent
24-hour rolling statistics.

```rust
pub struct TickerEvent {
    pub exchange: Exchange,
    pub symbol: Symbol,
    pub timestamp: i64,
    pub last_price: Decimal,
    pub best_bid: Option<Decimal>,
    pub best_ask: Option<Decimal>,
    pub volume_24h: Decimal,
    pub quote_volume_24h: Decimal,
    pub price_change_24h: Option<Decimal>,
    pub price_change_pct_24h: Option<Decimal>,
    pub high_24h: Option<Decimal>,
    pub low_24h: Option<Decimal>,
}
```

### LiquidationEvent
Liquidation orders (futures markets).

```rust
pub struct LiquidationEvent {
    pub exchange: Exchange,
    pub symbol: Symbol,
    pub timestamp: i64,
    pub side: Side,
    pub price: Decimal,
    pub quantity: Decimal,
    pub order_id: Option<String>,
}
```

### FundingRateEvent
Funding rate updates (perpetual futures).

```rust
pub struct FundingRateEvent {
    pub exchange: Exchange,
    pub symbol: Symbol,
    pub timestamp: i64,
    pub rate: Decimal,
    pub next_funding_time: i64,
}
```

## Symbol Format

Each exchange uses a different symbol format. The `Symbol` struct provides conversions:

```rust
use janus_core::{Symbol, Exchange};

let symbol = Symbol::new("BTC", "USDT");

// Convert to exchange-specific format
assert_eq!(symbol.to_exchange_format(Exchange::Binance), "BTCUSD");
assert_eq!(symbol.to_exchange_format(Exchange::Coinbase), "BTC-USD");
assert_eq!(symbol.to_exchange_format(Exchange::Kraken), "BTC/USDT");
assert_eq!(symbol.to_exchange_format(Exchange::Okx), "BTC-USDT");

// Parse from exchange format
let sym = Symbol::from_exchange_format("BTCUSD", Exchange::Binance).unwrap();
assert_eq!(sym.base, "BTC");
assert_eq!(sym.quote, "USDT");
```

## Adding a New Exchange

1. Create `src/adapters/your_exchange.rs`
2. Implement the adapter struct:

```rust
pub struct YourExchangeAdapter {
    ws_url: String,
}

impl YourExchangeAdapter {
    pub fn new() -> Self {
        Self {
            ws_url: "wss://your-exchange.com/ws".to_string(),
        }
    }
    
    pub fn ws_url(&self) -> &str {
        &self.ws_url
    }
    
    pub fn subscribe_message(&self, symbol: &Symbol) -> String {
        // Build subscription JSON
    }
    
    pub fn parse_message(&self, raw: &str) -> Result<Vec<MarketDataEvent>> {
        // Parse exchange messages to MarketDataEvent
    }
}
```

3. Add to `Cargo.toml`:

```toml
[features]
your-exchange = []
```

4. Export from `src/adapters/mod.rs`:

```rust
#[cfg(feature = "your-exchange")]
pub mod your_exchange;

#[cfg(feature = "your-exchange")]
pub use your_exchange::YourExchangeAdapter;
```

5. Add tests in `src/adapters/your_exchange.rs`:

```rust
#[cfg(test)]
mod tests {
    use super::*;
    
    #[test]
    fn test_parse_trade() {
        let adapter = YourExchangeAdapter::new();
        let raw = r#"{"type":"trade",...}"#;
        let events = adapter.parse_message(raw).unwrap();
        assert_eq!(events.len(), 1);
    }
}
```

## Testing

Run all tests:

```bash
cargo test --all-features
```

Test specific exchange:

```bash
cargo test --features coinbase
```

## Performance

- Zero-copy message parsing where possible
- Efficient decimal handling with `rust_decimal`
- Minimal allocations in hot path
- Async-first design with Tokio

## License

MIT OR Apache-2.0

## Contributing

Contributions are welcome! Please ensure:

1. All tests pass
2. New exchanges include comprehensive tests
3. Code follows Rust style guidelines
4. Documentation is updated

## Week 1 Status: ✅ COMPLETE

✅ **Completed:**
- [x] Unified `MarketDataEvent` types in `janus-core`
- [x] Coinbase adapter with trade, ticker, and **L2 order book** support
- [x] Kraken adapter with trade, ticker, and **L2 order book** support
- [x] OKX adapter with trade, ticker, **L2 order book**, **BBO**, funding rate, and liquidation support
- [x] Health monitoring system (`HealthChecker`)
- [x] Price/volume normalization utilities
- [x] Comprehensive test coverage (36 passing tests)
- [x] Complete documentation and examples

📋 **Week 1 Extension (Optional):**
- [ ] CNS metrics integration (exchange health → Prometheus)
- [ ] Integration with `services/data` (ConnectorManager refactor)
- [ ] Binance/Bybit/Kucoin adapter migration

📋 **Week 2 & Beyond:**
- [ ] WebSocket auto-reconnection utilities
- [ ] Rate limiting per exchange
- [ ] Message buffering and replay
- [ ] Cross-exchange arbitrage detection utilities
- [ ] Kline/candlestick parsing (if needed)