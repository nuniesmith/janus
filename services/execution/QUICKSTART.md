# JANUS Execution Service - Quick Start Guide

## Overview

The JANUS Execution Service handles order execution, position management, and risk controls for the JANUS trading system. It supports simulated, paper, and live trading modes with Redis-based state broadcasting.

## Quick Start

### 1. Prerequisites

```bash
# Install Rust (if not already installed)
curl --proto '=https' --tlsv1.2 -sSf https://sh.rustup.rs | sh

# Start Redis (required for state broadcasting)
redis-server

# Optional: Start QuestDB (for order history)
docker run -p 9000:9000 -p 9009:9009 questdb/questdb
```

### 2. Build

```bash
# From JANUS workspace
cd src/janus
cargo build -p janus-execution

# Or from FKS root
cd fks
cargo build -p janus-execution
```

### 3. Configure

Set environment variables:

```bash
# Required
export REDIS_URL=redis://localhost:6379
export INITIAL_EQUITY=10000.0
export EXECUTION_MODE=paper

# Optional
export QUESTDB_HOST=localhost:9009
export DISCORD_WEBHOOK_GENERAL=https://discord.com/api/webhooks/...
export DISCORD_ENABLE_NOTIFICATIONS=true
```

Or create a `.env` file in the project root:

```env
REDIS_URL=redis://localhost:6379
INITIAL_EQUITY=10000.0
EXECUTION_MODE=paper
QUESTDB_HOST=localhost:9009
```

### 4. Run

```bash
# From JANUS workspace
cd src/janus
cargo run -p janus-execution

# Or use the binary directly after building
./target/debug/janus-execution
```

## Execution Modes

### Simulated Mode
In-memory execution for backtesting and development. No real exchange connectivity.

```bash
export EXECUTION_MODE=simulated
cargo run -p janus-execution
```

### Paper Trading Mode
Live market data with simulated execution. Safe for testing strategies.

```bash
export EXECUTION_MODE=paper
export BYBIT_API_KEY=your_testnet_key
export BYBIT_API_SECRET=your_testnet_secret
cargo run -p janus-execution
```

### Live Trading Mode
**⚠️ WARNING: Real money trading!**

```bash
export EXECUTION_MODE=live
export BYBIT_API_KEY=your_live_key
export BYBIT_API_SECRET=your_live_secret
cargo run -p janus-execution
```

## State Broadcasting

The execution service broadcasts state updates to Redis at 10Hz:

### Channels

- `janus.state.full` - Complete execution state (equity, positions, volatility, etc.)
- `janus.state.equity` - Equity and P&L updates only
- `janus.state.volatility` - Volatility regime updates only

### Subscribe to Updates

```bash
# Monitor equity updates
redis-cli SUBSCRIBE janus.state.equity

# Monitor volatility updates
redis-cli SUBSCRIBE janus.state.volatility

# Monitor full state
redis-cli SUBSCRIBE janus.state.full
```

### Example Output

```json
// janus.state.equity
{
  "equity": 10250.50,
  "available_balance": 9500.00,
  "unrealized_pnl": 250.50,
  "timestamp": "2024-01-15T10:30:45.123Z"
}

// janus.state.volatility
{
  "current": 0.025,
  "regime": "Medium",
  "ewma_short": 0.023,
  "ewma_long": 0.027,
  "timestamp": "2024-01-15T10:30:45.123Z"
}
```

## API Endpoints

### gRPC Service

Default port: `50051`

```bash
# Example using grpcurl
grpcurl -plaintext localhost:50051 list
grpcurl -plaintext localhost:50051 fks.execution.v1.ExecutionService/GetAccountInfo
```

### HTTP API

Default port: `8080`

```bash
# Health check
curl http://localhost:8080/health

# Get account info
curl http://localhost:8080/api/v1/account

# Get positions
curl http://localhost:8080/api/v1/positions

# Get order history
curl http://localhost:8080/api/v1/orders

# Submit order
curl -X POST http://localhost:8080/api/v1/orders \
  -H "Content-Type: application/json" \
  -d '{
    "symbol": "BTCUSDT",
    "side": "Buy",
    "order_type": "Limit",
    "quantity": 0.01,
    "price": 42000.0
  }'
```

## Examples

### Paper Trading

```bash
cd src/janus
cargo run --package janus-execution --example paper_trading
```

### Simulated Environment

```bash
cargo run --package janus-execution --example sim_environment
```

### Walk-Forward Backtest

```bash
cargo run --package janus-execution --example walk_forward_backtest
```

### QuestDB Walk-Forward

```bash
cargo run --package janus-execution --example questdb_walkforward
```

### Benchmark Optimization

```bash
cargo run --package janus-execution --example benchmark_optimization
```

## Testing

### Run all tests

```bash
cd src/janus
cargo test -p janus-execution
```

### Run specific test

```bash
cargo test -p janus-execution test_order_submission
```

### Run with logs

```bash
RUST_LOG=debug cargo test -p janus-execution -- --nocapture
```

### Run benchmarks

```bash
cargo bench -p janus-execution
```

## Monitoring

### Logs

Logs are output to stdout/stderr. Control verbosity with `RUST_LOG`:

```bash
# Info level (default)
RUST_LOG=info cargo run -p janus-execution

# Debug level
RUST_LOG=debug cargo run -p janus-execution

# Trace level (very verbose)
RUST_LOG=trace cargo run -p janus-execution

# Filter by module
RUST_LOG=janus_execution::orders=debug cargo run -p janus-execution
```

### Redis Monitoring

```bash
# Monitor all Redis activity
redis-cli MONITOR

# Check channel subscribers
redis-cli PUBSUB CHANNELS janus.state.*

# Check number of subscribers
redis-cli PUBSUB NUMSUB janus.state.equity
```

### QuestDB Queries

```sql
-- View recent orders
SELECT * FROM orders ORDER BY timestamp DESC LIMIT 10;

-- View today's P&L
SELECT 
  symbol,
  SUM(realized_pnl) as total_pnl
FROM trades
WHERE timestamp > today()
GROUP BY symbol;

-- View execution latency stats
SELECT 
  percentile_cont(0.50, latency_ms) as p50,
  percentile_cont(0.95, latency_ms) as p95,
  percentile_cont(0.99, latency_ms) as p99
FROM order_events
WHERE timestamp > today();
```

## Integration with Brain

The Brain (Python) should subscribe to state updates instead of making blocking HTTP calls:

```python
import redis
import json

# Connect to Redis
r = redis.Redis(host='localhost', port=6379, decode_responses=True)
pubsub = r.pubsub()

# Subscribe to equity updates
pubsub.subscribe('janus.state.equity')

# Non-blocking message handler
for message in pubsub.listen():
    if message['type'] == 'message':
        data = json.loads(message['data'])
        equity = data['equity']
        # Use equity in sizing decisions...
```

## Troubleshooting

### Service won't start

1. Check Redis is running: `redis-cli PING`
2. Verify environment variables: `env | grep REDIS`
3. Check logs: `RUST_LOG=debug cargo run -p janus-execution`

### No state updates in Redis

1. Verify Redis connection: `redis-cli MONITOR`
2. Check broadcaster logs for errors
3. Ensure `REDIS_URL` is correct

### Orders not executing

1. Check execution mode: `echo $EXECUTION_MODE`
2. Verify exchange credentials (paper/live mode)
3. Check risk limits and compliance rules
4. Review logs for rejection reasons

### QuestDB connection failed

1. Verify QuestDB is running: `curl http://localhost:9000`
2. Check `QUESTDB_HOST` variable
3. Service will continue without QuestDB (with warning)

## Configuration Files

### Config Structure

The service uses environment variables and optional config files:

```
src/janus/services/execution/
├── config/
│   ├── development.toml
│   ├── production.toml
│   └── staging.toml
```

### Example config.toml

```toml
[service]
grpc_port = 50051
http_port = 8080

[execution_mode]
mode = "paper"

[redis]
url = "redis://localhost:6379"

[exchanges.bybit]
api_key = "${BYBIT_API_KEY}"
api_secret = "${BYBIT_API_SECRET}"
testnet = true

[risk]
max_position_size = 1000.0
max_total_exposure = 10000.0
max_order_value = 5000.0

[compliance]
enable_wash_trading_detection = true
enable_market_manipulation_detection = true
```

## Performance Tuning

### State Broadcast Frequency

Adjust in code (`src/main.rs`):

```rust
let broadcaster_config = BroadcasterConfig {
    interval: std::time::Duration::from_millis(100), // 10Hz
    channel_prefix: "janus.state".to_string(),
    verbose: false,
};
```

### Redis Connection Pool

Redis connection manager is automatically managed by the `redis` crate with connection pooling.

### Async Runtime Tuning

```bash
# Increase worker threads for high-throughput scenarios
TOKIO_WORKER_THREADS=8 cargo run -p janus-execution
```

## Production Deployment

See `deploy/` directory for:

- Docker configuration
- Kubernetes manifests
- systemd service files
- Monitoring dashboards

Quick Docker deployment:

```bash
cd src/janus/services/execution/deploy
docker-compose up -d
```

## Development

### Code Structure

```
src/
├── api/              # gRPC and HTTP interfaces
├── exchanges/        # Exchange connectors
├── execution/        # Core execution engine
├── notifications/    # Discord/alerts
├── orders/          # Order management
├── positions/       # Position tracking
├── sim/             # Simulation engine
├── strategies/      # Strategy execution
├── config.rs        # Configuration
├── error.rs         # Error types
├── lib.rs           # Library exports
├── main.rs          # Service entry point
├── state_broadcaster.rs  # Redis pub/sub
└── types.rs         # Common types
```

### Adding a New Exchange

1. Implement the `Exchange` trait in `src/exchanges/`
2. Add exchange-specific configuration
3. Register in `ExecutionEngineFactory`
4. Add tests and examples

### Adding New Strategies

1. Implement the `Strategy` trait in `src/strategies/`
2. Add strategy configuration
3. Register in strategy factory
4. Add backtesting examples

## Resources

- **Migration Guide**: `MIGRATION.md` - Details about the migration to JANUS
- **Implementation Guide**: `../../docs/IMPLEMENTATION_GUIDE.md` - Step-by-step fixes
- **Critical Fixes**: `../../docs/CRITICAL_FIXES.md` - 168hr audit fixes
- **API Documentation**: `cargo doc --open -p janus-execution`

## Support

For issues:

1. Check diagnostics: `cargo check -p janus-execution`
2. Review logs with `RUST_LOG=debug`
3. Verify all dependencies are running (Redis, QuestDB)
4. Consult the examples directory for working code

---

**Ready to trade!** 🚀

For more details, see the full documentation in `docs/` or run `cargo doc --open`.