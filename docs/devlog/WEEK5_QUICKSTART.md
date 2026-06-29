# Week 5 Quick Start Guide

## Overview

Week 5 adds real-time capabilities:
- **WebSocket** - Live exchange updates
- **Position Tracking** - P&L calculations
- **Account Management** - Balance & margin monitoring

## Quick Examples

### 1. WebSocket Setup

```rust
use fks_execution::exchanges::bybit::{BybitWebSocket, BybitWsConfig, BybitEvent};

#[tokio::main]
async fn main() -> Result<(), Box<dyn std::error::Error>> {
    // Configure WebSocket
    let config = BybitWsConfig {
        api_key: std::env::var("BYBIT_API_KEY")?,
        api_secret: std::env::var("BYBIT_API_SECRET")?,
        testnet: true,
        subscribe_orders: true,
        subscribe_positions: true,
        subscribe_wallet: true,
    };

    // Create WebSocket client
    let ws = BybitWebSocket::new(config);
    let mut rx = ws.subscribe();

    // Start streaming
    ws.start().await?;

    // Process events
    while let Ok(event) = rx.recv().await {
        match event {
            BybitEvent::OrderUpdate(update) => {
                println!("📝 Order: {} {} {} @ {}",
                    update.symbol,
                    update.side,
                    update.qty,
                    update.price
                );
            }
            BybitEvent::PositionUpdate(update) => {
                println!("📊 Position: {} size={} PnL={}",
                    update.symbol,
                    update.size,
                    update.unrealised_pnl
                );
            }
            BybitEvent::WalletUpdate(update) => {
                println!("💰 Wallet: equity={} available={}",
                    update.total_equity,
                    update.total_available_balance
                );
            }
            BybitEvent::Connected => println!("✅ Connected"),
            BybitEvent::Disconnected => println!("❌ Disconnected"),
            BybitEvent::Error(e) => eprintln!("⚠️  Error: {}", e),
        }
    }

    Ok(())
}
```

### 2. Position Tracking

```rust
use fks_execution::positions::PositionTracker;
use fks_execution::types::OrderSide;
use rust_decimal::Decimal;

#[tokio::main]
async fn main() -> Result<(), Box<dyn std::error::Error>> {
    let tracker = PositionTracker::new();

    // Open long position: Buy 1 BTC at $50,000
    tracker.apply_fill(
        "bybit",
        "BTCUSD".to_string(),
        OrderSide::Buy,
        Decimal::ONE,
        Decimal::from(50000),
    ).await?;

    // Add to position: Buy 1 more BTC at $51,000
    tracker.apply_fill(
        "bybit",
        "BTCUSD".to_string(),
        OrderSide::Buy,
        Decimal::ONE,
        Decimal::from(51000),
    ).await?;

    // Get position
    let pos = tracker.get_position("bybit", "BTCUSD".to_string()).await;
    println!("Size: {}", pos.size);                    // 2.0
    println!("Entry: ${}", pos.entry_price);           // $50,500 (average)
    println!("Side: {}", pos.direction());             // LONG

    // Update mark price
    tracker.update_mark_price(
        "bybit",
        "BTCUSD".to_string(),
        Decimal::from(52000),
    ).await?;

    // Check P&L
    let pos = tracker.get_position("bybit", "BTCUSD".to_string()).await;
    println!("Unrealized P&L: ${}", pos.unrealized_pnl);  // $3,000
    println!("P&L %: {}%", pos.unrealized_pnl_pct);       // 2.94%

    // Partial close: Sell 1 BTC at $52,000
    tracker.apply_fill(
        "bybit",
        "BTCUSD".to_string(),
        OrderSide::Sell,
        Decimal::ONE,
        Decimal::from(52000),
    ).await?;

    let pos = tracker.get_position("bybit", "BTCUSD".to_string()).await;
    println!("Size: {}", pos.size);                    // 1.0
    println!("Realized P&L: ${}", pos.realized_pnl);   // $1,500
    println!("Unrealized P&L: ${}", pos.unrealized_pnl); // $1,500

    // Get stats across all positions
    let stats = tracker.get_stats().await;
    println!("\n📊 Portfolio Stats:");
    println!("Total positions: {}", stats.total_positions);
    println!("Long positions: {}", stats.long_positions);
    println!("Total P&L: ${}", stats.total_pnl);
    println!("Position value: ${}", stats.total_position_value);

    Ok(())
}
```

### 3. Account Management

```rust
use fks_execution::positions::AccountManager;
use rust_decimal::Decimal;

#[tokio::main]
async fn main() -> Result<(), Box<dyn std::error::Error>> {
    let manager = AccountManager::new();

    // Update balance
    manager.update_balance(
        "bybit",
        "USDT".to_string(),
        Decimal::from(10000),  // total
        Decimal::from(9000),   // available
        Decimal::from(1000),   // locked
    ).await?;

    // Update margin metrics
    manager.update_margin_metrics(
        "bybit",
        Decimal::from(10500),  // equity (with unrealized P&L)
        Decimal::from(10000),  // wallet balance
        Decimal::from(8000),   // available
        Decimal::from(10000),  // margin balance
        Decimal::from(500),    // unrealized P&L
        Decimal::from(2000),   // initial margin required
        Decimal::from(1000),   // maintenance margin required
    ).await?;

    // Get account
    let account = manager.get_account("bybit").await.unwrap();
    println!("💰 Account: {}", account.exchange);
    println!("Equity: ${}", account.total_equity);
    println!("Available: ${}", account.total_available_balance);
    println!("Margin ratio: {:.2}", account.margin_ratio);
    println!("Health ratio: {:.2}", account.health_ratio);
    println!("Status: {}", if account.is_healthy() {
        "✅ Healthy"
    } else if account.is_at_risk() {
        "⚠️  At Risk"
    } else {
        "🔴 Danger"
    });

    // Check buying power
    let leverage = Decimal::from(10);
    let buying_power = account.buying_power(leverage);
    println!("Buying power (10x): ${}", buying_power);  // $80,000

    // Check risk across all exchanges
    let at_risk = manager.check_risk().await;
    if !at_risk.is_empty() {
        println!("\n⚠️  Accounts at risk:");
        for account in at_risk {
            println!("  - {}", account);
        }
    }

    // Global stats
    let stats = manager.get_stats().await;
    println!("\n🌍 Global Stats:");
    println!("Exchanges: {}", stats.num_exchanges);
    println!("Currencies: {}", stats.num_currencies);
    println!("Total equity: ${}", stats.total_equity);
    println!("Global health: {:.2}", stats.global_health_ratio);

    Ok(())
}
```

### 4. Integrated Example (WebSocket + Position + Account)

```rust
use fks_execution::exchanges::bybit::{BybitWebSocket, BybitWsConfig, BybitEvent};
use fks_execution::positions::{PositionTracker, AccountManager};
use rust_decimal::Decimal;
use std::str::FromStr;

#[tokio::main]
async fn main() -> Result<(), Box<dyn std::error::Error>> {
    // Initialize components
    let position_tracker = PositionTracker::new();
    let account_manager = AccountManager::new();

    // Setup WebSocket
    let config = BybitWsConfig {
        api_key: std::env::var("BYBIT_API_KEY")?,
        api_secret: std::env::var("BYBIT_API_SECRET")?,
        testnet: true,
        subscribe_orders: true,
        subscribe_positions: true,
        subscribe_wallet: true,
    };

    let ws = BybitWebSocket::new(config);
    let mut rx = ws.subscribe();
    ws.start().await?;

    // Process real-time updates
    while let Ok(event) = rx.recv().await {
        match event {
            BybitEvent::PositionUpdate(update) => {
                // Update position with real-time data
                let size = Decimal::from_str(&update.size)?;
                let mark_price = Decimal::from_str(&update.mark_price)?;
                
                position_tracker.update_mark_price(
                    "bybit",
                    update.symbol.clone(),
                    mark_price,
                ).await?;

                let pos = position_tracker
                    .get_position("bybit", update.symbol.clone())
                    .await;
                
                println!("📊 {} P&L: ${} ({}%)",
                    update.symbol,
                    pos.unrealized_pnl,
                    pos.unrealized_pnl_pct
                );
            }

            BybitEvent::WalletUpdate(update) => {
                // Update account balances
                for coin in &update.coin {
                    let total = Decimal::from_str(&coin.wallet_balance)?;
                    let available = Decimal::from_str(&coin.available_balance)?;
                    let locked = Decimal::from_str(&coin.locked)?;

                    account_manager.update_balance(
                        "bybit",
                        coin.coin.clone(),
                        total,
                        available,
                        locked,
                    ).await?;
                }

                // Check account health
                let stats = account_manager.get_stats().await;
                println!("💰 Equity: ${} | Health: {:.2}",
                    stats.total_equity,
                    stats.global_health_ratio
                );

                // Alert if at risk
                let at_risk = account_manager.check_risk().await;
                if !at_risk.is_empty() {
                    eprintln!("⚠️  RISK ALERT: {:?}", at_risk);
                }
            }

            BybitEvent::OrderUpdate(update) => {
                println!("📝 Order {}: {} {} @ {}",
                    update.order_status,
                    update.side,
                    update.qty,
                    update.price
                );
            }

            _ => {}
        }
    }

    Ok(())
}
```

## Testing

```bash
# Run all tests
cargo test --lib

# Run position tests
cargo test --lib positions::

# Run WebSocket tests
cargo test --lib websocket::

# Run account tests
cargo test --lib account::

# With output
cargo test --lib -- --nocapture
```

## Common Patterns

### 1. Position P&L Monitoring

```rust
// Periodic P&L check
let mut interval = tokio::time::interval(Duration::from_secs(60));
loop {
    interval.tick().await;
    
    let stats = position_tracker.get_stats().await;
    println!("Total P&L: ${}", stats.total_pnl);
    
    if stats.total_unrealized_pnl < Decimal::from(-1000) {
        eprintln!("⚠️  Drawdown alert: ${}", stats.total_unrealized_pnl);
    }
}
```

### 2. Risk Monitoring

```rust
// Continuous risk checks
let mut interval = tokio::time::interval(Duration::from_secs(30));
loop {
    interval.tick().await;
    
    let at_risk = account_manager.check_risk().await;
    if !at_risk.is_empty() {
        // Send alert, reduce positions, etc.
        eprintln!("🚨 RISK ALERT: {:?}", at_risk);
    }
}
```

### 3. Position Rebalancing

```rust
// Check if position exceeds target
let pos = position_tracker.get_position("bybit", "BTCUSD".to_string()).await;
let target_size = Decimal::from(5);

if pos.size.abs() > target_size {
    let excess = pos.size.abs() - target_size;
    println!("Need to reduce position by {}", excess);
    // Execute reduce order...
}
```

## Environment Variables

```bash
# Required for WebSocket
export BYBIT_API_KEY="your_api_key"
export BYBIT_API_SECRET="your_api_secret"

# Optional
export BYBIT_TESTNET="true"
export LOG_LEVEL="debug"
```

## Key Metrics

### Position Health
- **Size**: Current position (+ long, - short)
- **Entry Price**: Average entry price
- **Mark Price**: Current market price
- **Unrealized P&L**: Mark-to-market profit/loss
- **Realized P&L**: Locked-in profit/loss from closes

### Account Health
- **Equity**: Total value including unrealized P&L
- **Margin Ratio**: Equity / Initial Margin (higher = better)
- **Health Ratio**: Equity / Maintenance Margin (>1.2 = safe)
- **Buying Power**: Available × Leverage

## Troubleshooting

### WebSocket disconnects
- Check API credentials
- Verify network connectivity
- Monitor reconnection attempts (max 10)
- Check exchange rate limits

### P&L calculations seem wrong
- Ensure mark prices are updating
- Verify fill prices and quantities
- Check for position flips vs reduces
- Review entry price averaging

### Margin ratio issues
- Update margin metrics regularly
- Sync with exchange position data
- Check leverage settings
- Verify maintenance margin calculations

## Next Steps

See `WEEK5_COMPLETE.md` for:
- Detailed architecture
- All test cases
- Integration patterns
- Performance characteristics