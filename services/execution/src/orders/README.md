# Order Management Module

Comprehensive order management system with pre-trade validation, lifecycle tracking, and persistence.

## Overview

The orders module provides production-ready order management with:
- **Pre-trade validation** - Risk checks before order submission
- **State machine** - Robust order lifecycle tracking
- **QuestDB persistence** - Time-series storage for compliance
- **Partial fills** - Automatic fill aggregation and tracking

## Components

### OrderManager (`mod.rs`)

Central coordinator for all order operations.

```rust
use fks_execution::orders::OrderManager;

// Initialize with QuestDB connection
let manager = OrderManager::new("localhost:9009").await?;

// Submit order (validates, tracks, persists)
let order_id = manager.submit_order(order).await?;

// Get order status
let order = manager.get_order(&order_id)?;

// Handle fill from exchange
manager.handle_fill(&order_id, fill).await?;

// Cancel order
manager.cancel_order(&order_id).await?;

// Get active orders
let active_orders = manager.get_active_orders();

// Reconcile with exchange (recover from crashes)
let reconciled_count = manager.reconcile_orders().await?;
```

### OrderValidator (`validation.rs`)

Pre-trade validation with configurable risk rules.

```rust
use fks_execution::orders::{OrderValidator, ValidationConfig};
use rust_decimal::Decimal;

// Configure validation rules
let config = ValidationConfig {
    max_order_value: Decimal::from(100_000),
    min_order_value: Decimal::from(10),
    max_position_value: Decimal::from(500_000),
    max_orders_per_second: 10,
    check_balance: true,
    check_price_bounds: true,
    allow_short_selling: true,
    ..Default::default()
};

let validator = OrderValidator::new(config);

// Validate before submission
let result = validator.validate_order(&order).await?;
if !result.is_valid {
    println!("Validation failed: {:?}", result.rejection_reasons);
}
```

**Validation Rules**:
- Order size limits (min/max value)
- Position size limits
- Price reasonableness checks
- Rate limiting (orders per second)
- Symbol-specific overrides
- Required fields per order type
- Short selling controls

### OrderTracker (`tracking.rs`)

State machine for order lifecycle management.

```rust
use fks_execution::orders::OrderTracker;
use fks_execution::types::OrderStatusEnum;

let tracker = OrderTracker::new();

// Add order to tracking
tracker.add_order(&order)?;

// Transition states (validates transitions)
tracker.transition(
    "order-123",
    OrderStatusEnum::New,
    OrderStatusEnum::Submitted,
    Some("Sent to exchange".to_string())
)?;

// Query state
let current_state = tracker.get_current_state("order-123")?;
let history = tracker.get_state_history("order-123")?;

// Get orders by state
let active = tracker.get_active_orders();
let filled = tracker.get_orders_in_state(OrderStatusEnum::Filled);
```

**State Machine**:
```
New → Submitted → PartiallyFilled → Filled
 ↓         ↓            ↓
 ↓    PendingCancel ────→ Cancelled
 ↓         ↓
 └────→ Rejected

Terminal states: Filled, Cancelled, Rejected, Expired
```

### OrderHistory (`history.rs`)

QuestDB persistence using InfluxDB Line Protocol.

```rust
use fks_execution::orders::OrderHistory;

// Initialize
let history = OrderHistory::new("localhost:9009").await?;

// Record order
history.record_order(&order).await?;

// Record fill
history.record_fill(&fill).await?;

// Batch operations
history.batch_record_orders(&orders).await?;
history.batch_record_fills(&fills).await?;
```

**Tables**:
- `execution_orders` - Order state changes with tags and metrics
- `execution_fills` - Individual fill records

## Usage Examples

### Basic Order Submission

```rust
use fks_execution::{
    orders::OrderManager,
    types::{Order, OrderSide, OrderTypeEnum},
};
use rust_decimal::Decimal;

#[tokio::main]
async fn main() -> Result<(), Box<dyn std::error::Error>> {
    // Create manager
    let manager = OrderManager::new("localhost:9009").await?;
    
    // Create order
    let mut order = Order::new(
        "signal-123".to_string(),
        "BTCUSD".to_string(),
        "bybit".to_string(),
        OrderSide::Buy,
        OrderTypeEnum::Limit,
        Decimal::from(1),
    );
    order.price = Some(Decimal::from(50000));
    
    // Submit (validates, tracks, persists)
    match manager.submit_order(order).await {
        Ok(order_id) => {
            println!("Order submitted: {}", order_id);
        }
        Err(e) => {
            eprintln!("Order rejected: {}", e);
        }
    }
    
    Ok(())
}
```

### Handling Partial Fills

```rust
use fks_execution::types::{Fill, OrderSide};
use rust_decimal::Decimal;
use uuid::Uuid;

// First fill: 30% of order
let fill1 = Fill {
    id: Uuid::new_v4().to_string(),
    order_id: order_id.clone(),
    quantity: Decimal::new(3, 1), // 0.3
    price: Decimal::from(49950),
    fee: Decimal::from(15),
    fee_currency: "USDT".to_string(),
    side: OrderSide::Buy,
    timestamp: Utc::now(),
    is_maker: true,
};

manager.handle_fill(&order_id, fill1).await?;

// Check order status
let order = manager.get_order(&order_id)?;
println!("Filled: {} / {}", order.filled_quantity, order.quantity);
println!("Status: {:?}", order.status); // PartiallyFilled
println!("Avg price: {:?}", order.average_fill_price);

// Second fill: remaining 70%
let fill2 = Fill {
    id: Uuid::new_v4().to_string(),
    order_id: order_id.clone(),
    quantity: Decimal::new(7, 1), // 0.7
    price: Decimal::from(50050),
    fee: Decimal::from(35),
    fee_currency: "USDT".to_string(),
    side: OrderSide::Buy,
    timestamp: Utc::now(),
    is_maker: false,
};

manager.handle_fill(&order_id, fill2).await?;

let order = manager.get_order(&order_id)?;
println!("Status: {:?}", order.status); // Filled
```

### Custom Validation Rules

```rust
use fks_execution::orders::{ValidationConfig, SymbolValidationConfig};
use std::collections::HashMap;

let mut config = ValidationConfig::default();

// Add symbol-specific rules
let mut overrides = HashMap::new();
overrides.insert(
    "BTCUSD".to_string(),
    SymbolValidationConfig {
        max_order_value: Some(Decimal::from(200_000)), // Higher limit
        min_order_value: Some(Decimal::from(100)),
        max_position_value: None,
        max_leverage: None,
        allow_trading: true,
    },
);

// Disable high-risk symbol
overrides.insert(
    "DOGEUSDT".to_string(),
    SymbolValidationConfig {
        max_order_value: None,
        min_order_value: None,
        max_position_value: None,
        max_leverage: None,
        allow_trading: false, // Blocked
    },
);

config.symbol_overrides = overrides;

let manager = OrderManager::with_config(
    "localhost:9009".to_string(),
    config,
).await?;
```

### Order Reconciliation

After a crash or network outage, reconcile with exchange:

```rust
// Reconcile all active orders with exchange
let reconciled = manager.reconcile_orders().await?;
println!("Reconciled {} orders", reconciled);

// This will:
// 1. Get all active orders from manager
// 2. Query exchange for each order's current status
// 3. Update local state if status changed
// 4. Persist updates to QuestDB
```

## Architecture

### Data Flow

```
Signal
  ↓
OrderManager.submit_order()
  ↓
OrderValidator.validate_order() ← Pre-trade checks
  ↓
OrderTracker.add_order() ← Initialize tracking
  ↓
ExchangeRouter.place_order() ← Submit to exchange
  ↓
OrderTracker.transition() ← Update to Submitted
  ↓
OrderHistory.record_order() ← Persist to QuestDB
  ↓
[Wait for fill...]
  ↓
OrderManager.handle_fill()
  ↓
Order.add_fill() ← Calculate avg price, update quantities
  ↓
OrderTracker.transition() ← Update state if needed
  ↓
OrderHistory.record_fill() ← Persist fill
```

### Concurrency Model

- **Read-Write Locks**: Used for shared state (active orders, state history)
- **Concurrent Reads**: Multiple threads can read simultaneously
- **Serialized Writes**: Writes acquire exclusive locks
- **Lock-Free Lookups**: Order cache uses `RwLock<HashMap>`

### Error Handling

```rust
use fks_execution::error::ExecutionError;

match manager.submit_order(order).await {
    Ok(order_id) => { /* success */ },
    Err(ExecutionError::Validation(msg)) => {
        // Pre-trade validation failed
        eprintln!("Validation: {}", msg);
    },
    Err(ExecutionError::Exchange { exchange, message, .. }) => {
        // Exchange rejected order
        eprintln!("Exchange {}: {}", exchange, message);
    },
    Err(ExecutionError::Storage(err)) => {
        // QuestDB persistence failed
        eprintln!("Storage: {}", err);
    },
    Err(e) => {
        // Other errors
        eprintln!("Error: {}", e);
    }
}
```

## Configuration

### Environment Variables

```bash
# QuestDB connection
QUESTDB_HOST=localhost:9009

# Validation limits
MAX_ORDER_VALUE=100000
MIN_ORDER_VALUE=10
MAX_POSITION_VALUE=500000

# Rate limiting
MAX_ORDERS_PER_SECOND=10
MIN_ORDER_INTERVAL_MS=100
```

### Runtime Configuration

```rust
// Get current config
let current_config = validator.get_config();

// Update config
validator.update_config(new_config);

// Clear rate limit history
validator.clear_history();
```

## QuestDB Schema

### execution_orders

**Tags** (indexed):
- `order_id` - Internal order ID
- `symbol` - Trading symbol (e.g., BTCUSD)
- `exchange` - Exchange name (e.g., bybit)
- `side` - Buy/Sell
- `order_type` - Market/Limit/etc
- `status` - Order status
- `exchange_order_id` - Exchange's order ID
- `signal_id` - Original signal ID

**Fields**:
- `quantity` - Order size
- `filled_quantity` - Amount filled
- `remaining_quantity` - Amount remaining
- `price` - Limit price (if applicable)
- `stop_price` - Stop price (if applicable)
- `average_fill_price` - Weighted average fill price
- `time_in_force` - GTC/IOC/FOK/etc
- `strategy` - Execution strategy
- `fills_count` - Number of fills

**Timestamp**: Order update time (nanosecond precision)

### execution_fills

**Tags** (indexed):
- `fill_id` - Unique fill ID
- `order_id` - Parent order ID
- `side` - Buy/Sell
- `fee_currency` - Fee currency

**Fields**:
- `quantity` - Fill size
- `price` - Fill price
- `fee` - Fee amount
- `is_maker` - Maker vs taker

**Timestamp**: Fill time (nanosecond precision)

## Testing

Run tests:
```bash
cargo test --lib
```

Test specific module:
```bash
cargo test --lib orders::validation
cargo test --lib orders::tracking
cargo test --lib orders::history
```

## Performance

### Benchmarks (Typical)

- Order validation: ~50 μs
- State transition: ~10 μs
- QuestDB write: ~10-20 ms (network)
- Order lookup: ~1 μs (hash map)

### Optimization Tips

1. **Batch operations** when possible
2. **Cache active orders** in memory
3. **Use async** for I/O operations
4. **Rate limit** to prevent exchange bans

## Troubleshooting

### QuestDB Connection Failed

```
Error: Storage(Connection("Cannot connect to QuestDB at localhost:9009"))
```

**Solution**: Ensure QuestDB is running and ILP is enabled:
```bash
# Check QuestDB is running
curl http://localhost:9000

# Check ILP port is open
nc -zv localhost 9009
```

### Validation Failed

```
Error: Validation("Order validation failed: Order value 150000 exceeds maximum 100000")
```

**Solution**: Adjust validation config or order size.

### Invalid State Transition

```
Error: InvalidOrderState("Invalid transition from Filled to Submitted")
```

**Solution**: Check order is in correct state before attempting transition.

## See Also

- [Week 3 Completion Report](../../../docs/execution/WEEK3_COMPLETE.md)
- [Exchange Integration](../exchanges/README.md)
- [Type Definitions](../types.rs)
- [Error Handling](../error.rs)