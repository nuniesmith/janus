# FKS Execution Service

**Standalone trade execution and position management microservice**

---

## Overview

The Execution Service is responsible for:
- **Trade execution** across multiple cryptocurrency exchanges
- **Order management** (placement, cancellation, status tracking)
- **Position tracking** with real-time P&L calculation
- **Account management** (balance, margin, exposure)
- **Simulated trading** for testing and backtesting
- **gRPC API** for receiving signals from Janus Service

This service is part of a microservices architecture where:
- **Data Service** handles all market data ingestion
- **Janus Service** generates trading signals
- **Execution Service** (this) executes trades and manages positions

---

## Architecture

```
┌─────────────────────────────────────────────────────────┐
│                  Execution Service                      │
├─────────────────────────────────────────────────────────┤
│                                                         │
│  ┌─────────────┐   ┌─────────────┐   ┌─────────────┐  │
│  │   API       │   │   Order     │   │  Position   │  │
│  │   Layer     │   │  Management │   │  Tracking   │  │
│  │             │   │             │   │             │  │
│  │ • gRPC      │──▶│ • Placement │──▶│ • Open Pos  │  │
│  │ • Health    │   │ • Cancel    │   │ • P&L Calc  │  │
│  │ • Metrics   │   │ • Status    │   │ • Risk      │  │
│  └─────────────┘   └─────────────┘   └─────────────┘  │
│                                                         │
│  ┌─────────────┐   ┌─────────────┐   ┌─────────────┐  │
│  │  Exchange   │   │ Simulation  │   │   Audit     │  │
│  │  Adapters   │   │   Engine    │   │   Trail     │  │
│  │             │   │             │   │             │  │
│  │ • Bybit     │   │ • Matching  │   │ • QuestDB   │  │
│  │ • Binance   │   │ • Fills     │   │ • Logs      │  │
│  │ • Kucoin    │   │ • Slippage  │   │ • Reports   │  │
│  └─────────────┘   └─────────────┘   └─────────────┘  │
└─────────────────────────────────────────────────────────┘
```

---

## Features

### 🎯 Multi-Exchange Support
- **Bybit**: Spot & derivatives trading
- **Binance**: Spot & futures trading
- **Kucoin**: Spot trading
- **Exchange abstraction**: Easy to add new exchanges
- **Automatic failover**: Switch exchanges on errors

### 🔄 Execution Modes

#### 1. Simulated Mode
- Internal order matching engine
- No real API calls
- Perfect for testing strategies
- Configurable slippage and fees
- Instant fills or realistic delays

#### 2. Paper Trading Mode
- Real exchange APIs
- Demo/testnet accounts only
- Full API integration testing
- No real money at risk

#### 3. Live Trading Mode
- Real exchange APIs
- Real accounts with real money
- Full risk management
- Compliance checks
- Audit logging

### 📊 Order Management
- **Order Types**: Market, Limit, Stop Loss, Take Profit
- **Order Status**: Real-time tracking
- **Order Fills**: Partial and complete fills
- **Order Cancellation**: Individual or bulk
- **Order History**: Complete audit trail

### 💼 Position Management
- **Real-time P&L**: Mark-to-market
- **Position Sizing**: Automatic calculation
- **Risk Limits**: Per-position and portfolio
- **Margin Tracking**: Available and used margin
- **Position Closing**: Market or limit orders

### 🛡️ Risk Management
- **Pre-trade validation**: Size, margin, exposure
- **Position limits**: Max positions per symbol
- **Portfolio limits**: Total exposure caps
- **Drawdown limits**: Stop trading on losses
- **Compliance checks**: Regulatory requirements

### 📡 APIs

#### gRPC API (Port 50052)
```protobuf
service ExecutionService {
  rpc SubmitSignal(TradingSignal) returns (ExecutionResponse);
  rpc GetPosition(PositionRequest) returns (Position);
  rpc GetOpenPositions(OpenPositionsRequest) returns (OpenPositionsResponse);
  rpc GetOpenOrders(OrderRequest) returns (OrderList);
  rpc CancelOrder(CancelRequest) returns (CancelResponse);
  rpc GetAccountBalance(BalanceRequest) returns (BalanceResponse);
  rpc StreamExecutions(ExecutionStreamRequest) returns (stream ExecutionUpdate);
  rpc HealthCheck(HealthCheckRequest) returns (HealthCheckResponse);
}
```

---

## Quick Start

### Prerequisites

- Rust 1.75+ (2021 edition)
- Redis 7.0+ running
- QuestDB 7.0+ running (for audit trail)
- Exchange API keys (for live/paper trading)

### Installation

1. **Build the service:**
```bash
cd src/execution
cargo build --release
```

2. **Configure environment:**
```bash
cp .env.example .env
# Edit .env with your settings
```

3. **Run the service:**
```bash
cargo run --release
```

### Docker

```bash
# Build image
docker build -f ../../docker/rust/Dockerfile \
  --target standalone \
  --build-arg SERVICE_NAME=execution-service \
  -t fks-execution:latest \
  ../..

# Run container (simulated mode)
docker run -d \
  --name fks-execution \
  -p 50052:50052 \
  -e EXECUTION_MODE=simulated \
  -e REDIS_URL=redis://redis:6379 \
  fks-execution:latest
```

### Docker Compose

```bash
# Start with dependencies
docker compose up -d execution redis questdb

# View logs
docker compose logs -f execution
```

---

## Configuration

### Environment Variables

```bash
# Service Configuration
RUST_LOG=info,fks_execution=debug
GRPC_PORT=50052
HTTP_PORT=8083

# Execution Mode
EXECUTION_MODE=simulated  # simulated, paper, live

# Redis Configuration (for state)
REDIS_URL=redis://localhost:6379

# QuestDB Configuration (for audit trail)
QUESTDB_HOST=localhost
QUESTDB_ILP_PORT=9009

# Exchange API Keys (for paper/live modes)
BYBIT_API_KEY=your_key_here
BYBIT_API_SECRET=your_secret_here
BYBIT_TESTNET=true

BINANCE_API_KEY=your_key_here
BINANCE_API_SECRET=your_secret_here
BINANCE_TESTNET=true

# Risk Management
MAX_POSITION_SIZE_USD=10000
MAX_PORTFOLIO_EXPOSURE_USD=50000
MAX_OPEN_POSITIONS=5
MAX_DAILY_LOSS_USD=1000

# Simulated Mode Settings
SIMULATED_INITIAL_BALANCE=100000
SIMULATED_SLIPPAGE_BPS=5
SIMULATED_FEE_BPS=10
SIMULATED_FILL_DELAY_MS=100

# Performance
ORDER_CACHE_TTL_SECONDS=3600
POSITION_UPDATE_INTERVAL_MS=1000
```

### Config File (config/execution.toml)

```toml
[service]
name = "execution-service"
version = "0.1.0"

[grpc]
host = "0.0.0.0"
port = 50052

[http]
host = "0.0.0.0"
port = 8083

[execution]
mode = "simulated"  # simulated, paper, live

[redis]
url = "redis://localhost:6379"

[questdb]
host = "localhost"
ilp_port = 9009

[risk]
max_position_size_usd = 10000.0
max_portfolio_exposure_usd = 50000.0
max_open_positions = 5
max_daily_loss_usd = 1000.0

[simulation]
initial_balance = 100000.0
slippage_bps = 5
fee_bps = 10
fill_delay_ms = 100

[exchanges.bybit]
enabled = true
testnet = true

[exchanges.binance]
enabled = false
testnet = true
```

---

## Development

### Project Structure

```
src/execution/
├── Cargo.toml
├── build.rs              # Protobuf compilation
├── proto/                # Protobuf definitions
├── src/
│   ├── main.rs           # Entry point
│   ├── lib.rs            # Library exports
│   ├── config/           # Configuration management
│   ├── exchanges/        # Exchange adapters
│   │   ├── traits.rs     # Exchange trait definition
│   │   ├── bybit/
│   │   │   ├── rest.rs
│   │   │   └── websocket.rs
│   │   └── binance/
│   │       └── rest.rs
│   ├── orders/           # Order management
│   │   ├── manager.rs
│   │   └── state.rs
│   ├── positions/        # Position tracking
│   │   ├── tracker.rs
│   │   └── pnl.rs
│   ├── simulation/       # Simulated exchange
│   │   ├── engine.rs
│   │   └── matcher.rs
│   ├── api/              # API layer
│   │   └── grpc.rs       # gRPC server
│   ├── validation/       # Pre-trade validation
│   │   └── rules.rs
│   └── audit/            # Audit logging
│       └── logger.rs
├── tests/
│   ├── integration_tests.rs
│   └── simulation_tests.rs
└── examples/
    └── submit_signal.rs
```

### Building

```bash
# Debug build
cargo build

# Release build
cargo build --release

# Run tests
cargo test

# Run with specific mode
EXECUTION_MODE=simulated cargo run

# Check code
cargo clippy
cargo fmt --check
```

### Testing

```bash
# Unit tests
cargo test --lib

# Integration tests
cargo test --test '*'

# Test simulated mode
cargo test --test simulation_tests

# Test with logging
RUST_LOG=debug cargo test -- --nocapture
```

---

## API Examples

### gRPC Client (Rust)

```rust
use tonic::Request;
use fks_execution::fks::v1::execution_service_client::ExecutionServiceClient;
use fks_execution::fks::v1::{TradingSignal, SignalAction};

#[tokio::main]
async fn main() -> Result<(), Box<dyn std::error::Error>> {
    let mut client = ExecutionServiceClient::connect("http://localhost:50052").await?;

    let signal = TradingSignal {
        signal_id: "sig-001".to_string(),
        symbol: "BTCUSD".to_string(),
        action: SignalAction::Buy as i32,
        quantity: 0.01,
        price_limit: Some(50000.0),
        stop_loss: Some(49000.0),
        take_profit: Some(52000.0),
        timestamp: chrono::Utc::now().timestamp(),
        strategy_id: "momentum-v1".to_string(),
        confidence: 0.85,
        max_slippage: Some(0.001),
        expiry_timestamp: None,
        mode: 1, // SIMULATED
    };

    let response = client.submit_signal(Request::new(signal)).await?;
    println!("Execution response: {:?}", response.into_inner());

    Ok(())
}
```

### Get Position

```rust
use fks_execution::fks::v1::PositionRequest;

let request = PositionRequest {
    symbol: "BTCUSD".to_string(),
};

let position = client.get_position(Request::new(request)).await?.into_inner();
println!("Position: {:?}", position);
```

### Stream Execution Updates

```rust
use fks_execution::fks::v1::ExecutionStreamRequest;

let request = ExecutionStreamRequest {
    strategy_id: Some("momentum-v1".to_string()),
    symbol: None,
};

let mut stream = client.stream_executions(Request::new(request)).await?.into_inner();

while let Some(update) = stream.message().await? {
    println!("Execution update: {:?}", update);
}
```

---

## Execution Modes

### Simulated Mode

**Use Case**: Strategy testing, backtesting, development

**Features**:
- No real API calls
- Internal order matching
- Configurable slippage and fees
- Instant or realistic fill delays
- Perfect for testing

**Configuration**:
```bash
EXECUTION_MODE=simulated
SIMULATED_INITIAL_BALANCE=100000
SIMULATED_SLIPPAGE_BPS=5
SIMULATED_FEE_BPS=10
```

**Limitations**:
- No real market depth
- Simplified matching logic
- No exchange-specific quirks

### Paper Trading Mode

**Use Case**: Testing with real exchange APIs, no real money

**Features**:
- Real exchange API calls
- Testnet/demo accounts
- Real market data
- Real order book depth
- Exchange-specific behavior

**Configuration**:
```bash
EXECUTION_MODE=paper
BYBIT_TESTNET=true
BYBIT_API_KEY=testnet_key
BYBIT_API_SECRET=testnet_secret
```

**Requirements**:
- Exchange testnet account
- Testnet API keys

### Live Trading Mode

**Use Case**: Production trading with real money

**Features**:
- Real exchange API calls
- Real accounts
- Real money
- Full audit trail
- Compliance checks
- Enhanced risk management

**Configuration**:
```bash
EXECUTION_MODE=live
BYBIT_TESTNET=false
BYBIT_API_KEY=live_key
BYBIT_API_SECRET=live_secret
```

**Safety**:
- Requires explicit confirmation
- Enhanced validation
- Strict risk limits
- Complete audit logging

---

## Risk Management

### Pre-Trade Validation

Every signal goes through validation before execution:

1. **Position Size Check**: Does not exceed max per symbol
2. **Portfolio Exposure Check**: Total exposure within limits
3. **Balance Check**: Sufficient margin available
4. **Drawdown Check**: Not in daily loss limit
5. **Compliance Check**: Meets regulatory requirements

### Position Limits

```toml
[risk]
max_position_size_usd = 10000.0      # Max per position
max_portfolio_exposure_usd = 50000.0  # Max total exposure
max_open_positions = 5                # Max concurrent positions
max_daily_loss_usd = 1000.0           # Daily loss limit
max_position_hold_hours = 24          # Max hold time
```

### Circuit Breakers

Automatic trading halt when:
- Daily loss limit reached
- Too many consecutive losses
- Exchange connectivity issues
- Abnormal price movements

---

## Monitoring

### Prometheus Metrics

Available at `http://localhost:8083/metrics`:

```
# Execution metrics
fks_execution_signals_received_total{strategy="momentum-v1"}
fks_execution_orders_placed_total{exchange="bybit",symbol="BTCUSD"}
fks_execution_orders_filled_total{exchange="bybit",symbol="BTCUSD"}
fks_execution_orders_rejected_total{reason="insufficient_margin"}

# Position metrics
fks_execution_positions_open{symbol="BTCUSD"}
fks_execution_position_pnl_usd{symbol="BTCUSD"}

# Performance metrics
fks_execution_latency_seconds{operation="submit_signal",quantile="0.95"}
fks_execution_latency_seconds{operation="place_order",quantile="0.99"}

# Risk metrics
fks_execution_portfolio_exposure_usd
fks_execution_daily_pnl_usd
fks_execution_drawdown_usd
```

### Health Checks

```bash
# gRPC health check
grpcurl -plaintext localhost:50052 fks.v1.ExecutionService/HealthCheck

# HTTP health check
curl http://localhost:8083/health
```

---

## Deployment

### Production Checklist

- [ ] Set `EXECUTION_MODE=live` only after testing
- [ ] Configure proper risk limits
- [ ] Set up Prometheus scraping
- [ ] Configure alerting for circuit breakers
- [ ] Enable TLS for gRPC
- [ ] Rotate API keys regularly
- [ ] Set up automated backups for audit logs
- [ ] Configure Redis persistence
- [ ] Set up monitoring dashboards
- [ ] Document runbooks for emergencies
- [ ] Test emergency stop procedures

### Scaling

**Vertical Scaling** (Recommended):
- Single instance for order consistency
- Increase CPU for faster validation
- Increase memory for order cache

**Horizontal Scaling** (Advanced):
- Partition by symbol or strategy
- Shared Redis for coordination
- Leader election for risk checks

---

## Troubleshooting

### Common Issues

**Issue: Orders rejected with "insufficient margin"**
```
Solution: Check account balance, reduce position size, check margin requirements
```

**Issue: High latency in order placement**
```
Solution: Check exchange API status, verify network latency, optimize validation
```

**Issue: Position P&L not updating**
```
Solution: Verify data service connection, check position tracker, restart service
```

**Issue: Simulated fills not matching expected**
```
Solution: Review slippage settings, check fill delay configuration, verify matching logic
```

### Debug Mode

```bash
# Enable verbose logging
RUST_LOG=debug,fks_execution=trace cargo run

# Test signal submission
cargo run --example submit_signal

# Validate configuration
cargo run -- --validate-config
```

---

## Security

### API Key Management

**DO NOT**:
- ❌ Hardcode API keys
- ❌ Commit keys to Git
- ❌ Share keys in logs
- ❌ Use production keys in development

**DO**:
- ✅ Use environment variables
- ✅ Use Docker secrets (production)
- ✅ Rotate keys regularly
- ✅ Use testnet keys for testing
- ✅ Monitor API key usage

### Audit Trail

Every execution is logged to QuestDB:
- Signal received
- Validation results
- Order placement
- Order fills
- Position updates
- Errors and rejections

---

## Contributing

1. Fork the repository
2. Create a feature branch (`git checkout -b feature/amazing-feature`)
3. Add tests for new features
4. Ensure all tests pass (`cargo test`)
5. Run clippy and fmt (`cargo clippy && cargo fmt`)
6. Commit your changes
7. Push to the branch
8. Open a Pull Request

---

## License

MIT License - see LICENSE file for details

---

## Related Services

- **Data Service**: Market data ingestion and storage
- **Janus Service**: Trading algorithm and signal generation
- **Monitor Service**: System monitoring and alerting
- **Audit Service**: Compliance and audit logging

---

## Support

- **Documentation**: `docs/MICROSERVICES_RESTRUCTURE.md`
- **Issues**: GitHub Issues
- **Discussions**: GitHub Discussions

---

**Status**: ✅ Ready for Implementation  
**Version**: 0.1.0  
**Last Updated**: 2024