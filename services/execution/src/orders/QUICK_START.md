# Order Management Quick Start

## 30-Second Setup

```rust
use fks_execution::orders::OrderManager;

let manager = OrderManager::new("localhost:9009").await?;
```

## Common Operations

### Submit Order
```rust
let order_id = manager.submit_order(order).await?;
```

### Get Order Status
```rust
let order = manager.get_order(&order_id)?;
println!("Status: {:?}, Filled: {}/{}", 
    order.status, order.filled_quantity, order.quantity);
```

### Cancel Order
```rust
manager.cancel_order(&order_id).await?;
```

### Handle Fill
```rust
manager.handle_fill(&order_id, fill).await?;
```

### List Active Orders
```rust
let active = manager.get_active_orders();
for order in active {
    println!("{}: {:?}", order.id, order.status);
}
```

## Order Creation

### Market Order
```rust
use fks_execution::types::{Order, OrderSide, OrderTypeEnum};
use rust_decimal::Decimal;

let order = Order::new(
    "signal-123".to_string(),      // signal_id
    "BTCUSD".to_string(),          // symbol
    "bybit".to_string(),            // exchange
    OrderSide::Buy,                 // side
    OrderTypeEnum::Market,          // type
    Decimal::from(1),               // quantity
);
```

### Limit Order
```rust
let mut order = Order::new(
    "signal-123".to_string(),
    "BTCUSD".to_string(),
    "bybit".to_string(),
    OrderSide::Buy,
    OrderTypeEnum::Limit,
    Decimal::from(1),
);
order.price = Some(Decimal::from(50000));
```

### Stop-Limit Order
```rust
let mut order = Order::new(
    "signal-123".to_string(),
    "BTCUSD".to_string(),
    "bybit".to_string(),
    OrderSide::Sell,
    OrderTypeEnum::StopLimit,
    Decimal::from(1),
);
order.price = Some(Decimal::from(48000));      // limit price
order.stop_price = Some(Decimal::from(49000)); // trigger price
```

## Custom Validation

### Basic Configuration
```rust
use fks_execution::orders::ValidationConfig;
use rust_decimal::Decimal;

let config = ValidationConfig {
    max_order_value: Decimal::from(50_000),
    min_order_value: Decimal::from(100),
    max_orders_per_second: 5,
    ..Default::default()
};

let manager = OrderManager::with_config(
    "localhost:9009".to_string(),
    config
).await?;
```

### Symbol-Specific Rules
```rust
use fks_execution::orders::SymbolValidationConfig;
use std::collections::HashMap;

let mut config = ValidationConfig::default();
let mut overrides = HashMap::new();

// Higher limits for BTC
overrides.insert("BTCUSD".to_string(), SymbolValidationConfig {
    max_order_value: Some(Decimal::from(200_000)),
    allow_trading: true,
    ..Default::default()
});

// Disable risky symbol
overrides.insert("SHIBAINU".to_string(), SymbolValidationConfig {
    allow_trading: false,
    ..Default::default()
});

config.symbol_overrides = overrides;
```

## State Tracking

### Get State History
```rust
let history = manager.get_order_state_history(&order_id)?;
for transition in history {
    println!("{:?} -> {:?} at {} ({})",
        transition.from_state,
        transition.to_state,
        transition.timestamp,
        transition.reason.unwrap_or_default()
    );
}
```

### Check Order State
```rust
if order.is_active() {
    // Can be filled or cancelled
}

if order.is_terminal() {
    // Completed (Filled/Cancelled/Rejected)
}
```

## Error Handling

```rust
use fks_execution::error::ExecutionError;

match manager.submit_order(order).await {
    Ok(order_id) => println!("Order submitted: {}", order_id),
    
    Err(ExecutionError::Validation(msg)) => {
        eprintln!("Validation failed: {}", msg);
    },
    
    Err(ExecutionError::Exchange { exchange, message, .. }) => {
        eprintln!("Exchange {} error: {}", exchange, message);
    },
    
    Err(ExecutionError::RateLimitExceeded(exch)) => {
        eprintln!("Rate limited on {}", exch);
    },
    
    Err(e) => eprintln!("Error: {}", e),
}
```

## Statistics

```rust
let stats = manager.get_statistics();
println!("Active: {}, Completed: {}, Total: {}",
    stats.active_orders,
    stats.completed_orders,
    stats.total_orders_tracked
);
```

## Order Reconciliation

After crash/restart:

```rust
// Sync with exchange
let count = manager.reconcile_orders().await?;
println!("Reconciled {} orders", count);
```

## Complete Example

```rust
use fks_execution::{
    orders::{OrderManager, ValidationConfig},
    types::{Order, OrderSide, OrderTypeEnum, Fill},
};
use rust_decimal::Decimal;
use chrono::Utc;

#[tokio::main]
async fn main() -> Result<(), Box<dyn std::error::Error>> {
    // 1. Create manager
    let manager = OrderManager::new("localhost:9009").await?;
    
    // 2. Create order
    let mut order = Order::new(
        "sig-001".to_string(),
        "BTCUSD".to_string(),
        "bybit".to_string(),
        OrderSide::Buy,
        OrderTypeEnum::Limit,
        Decimal::from(1),
    );
    order.price = Some(Decimal::from(50000));
    
    // 3. Submit
    let order_id = manager.submit_order(order).await?;
    println!("Order submitted: {}", order_id);
    
    // 4. Simulate fill
    let fill = Fill {
        id: uuid::Uuid::new_v4().to_string(),
        order_id: order_id.clone(),
        quantity: Decimal::from(1),
        price: Decimal::from(50000),
        fee: Decimal::from(50),
        fee_currency: "USDT".to_string(),
        side: OrderSide::Buy,
        timestamp: Utc::now(),
        is_maker: true,
    };
    
    manager.handle_fill(&order_id, fill).await?;
    
    // 5. Check status
    let order = manager.get_order(&order_id)?;
    println!("Order filled: {} @ {}", 
        order.filled_quantity,
        order.average_fill_price.unwrap()
    );
    
    Ok(())
}
```

## Validation Result

```rust
use fks_execution::orders::OrderValidator;

let validator = OrderValidator::new(ValidationConfig::default());
let result = validator.validate_order(&order).await?;

if result.is_valid {
    println!("✓ Order valid");
} else {
    println!("✗ Rejected:");
    for reason in &result.rejection_reasons {
        println!("  - {}", reason);
    }
}

for warning in &result.warnings {
    println!("⚠ {}", warning);
}

for suggestion in &result.suggestions {
    println!("💡 {}", suggestion);
}
```

## State Machine

```
Valid Transitions:

New          → Submitted, Rejected, Cancelled
Submitted    → PartiallyFilled, Filled, PendingCancel, Cancelled, Rejected
PartiallyFilled → Filled, PendingCancel, Cancelled
PendingCancel   → Cancelled, Filled, PartiallyFilled

Terminal States (no further transitions):
- Filled
- Cancelled
- Rejected
- Expired
```

## QuestDB Tables

### Query Orders
```sql
-- Latest order status
SELECT * FROM execution_orders 
WHERE order_id = 'order-123' 
ORDER BY timestamp DESC 
LIMIT 1;

-- Orders by symbol
SELECT * FROM execution_orders 
WHERE symbol = 'BTCUSD' 
AND timestamp > dateadd('h', -1, now());
```

### Query Fills
```sql
-- Fills for an order
SELECT * FROM execution_fills 
WHERE order_id = 'order-123' 
ORDER BY timestamp;

-- Recent fills
SELECT * FROM execution_fills 
WHERE timestamp > dateadd('m', -5, now());
```

## Environment Variables

```bash
# Required
QUESTDB_HOST=localhost:9009

# Optional validation limits
MAX_ORDER_VALUE=100000
MIN_ORDER_VALUE=10
MAX_POSITION_VALUE=500000
MAX_ORDERS_PER_SECOND=10
MIN_ORDER_INTERVAL_MS=100
```

## Need More Help?

- Full docs: `src/orders/README.md`
- Completion report: `docs/execution/WEEK3_COMPLETE.md`
- Examples: `src/orders/mod.rs` tests
- API reference: `cargo doc --open`
