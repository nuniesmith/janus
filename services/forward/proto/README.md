# Forward Service Protocol Buffers

This directory contains the Protocol Buffer definitions specific to the JANUS Forward Service (Real-time Trading Engine).

## Overview

The Forward Service is the "wake state" of JANUS - it executes live trading strategies in real-time, processes market data, generates trading signals, and manages active positions.

## Package Structure

```
proto/
├── janus/v1/
│   └── janus.proto          # Service-specific signal generation API
├── buf.yaml                  # Buf configuration
└── README.md
```

## Proto Organization

### Local Protos (this directory)

**Package:** `janus.v1`

The `janus.v1` proto defines the signal generation gRPC service specific to the Forward Service:

- `JanusService` - Trading signal generation service
- `TradingSignal`, `IndicatorAnalysis` - Signal types
- Health check and model management endpoints

### Centralized Protos (from `fks-proto` crate)

Common proto types are imported from the centralized `fks-proto` crate:

```rust
// Import execution types from centralized proto crate
use fks_proto::execution::{
    ExecutionServiceClient,
    SignalRequest,
    OrderType,
    Side,
};

// Import forward service types
use fks_proto::forward::{
    ForwardServiceClient,
    StartRequest,
    ExecutionMode,
};
```

See `proto/fks/` in the root repository for centralized proto definitions.

## JanusService Definition

The `janus.v1.JanusService` provides trading signal generation capabilities:

| RPC | Description | Request | Response |
|-----|-------------|---------|----------|
| `GenerateSignal` | Generate a single trading signal | `GenerateSignalRequest` | `GenerateSignalResponse` |
| `GenerateSignalBatch` | Generate signals for multiple symbols | `GenerateSignalBatchRequest` | `GenerateSignalBatchResponse` |
| `GetHealth` | Get service health status | `HealthRequest` | `HealthResponse` |
| `GetMetrics` | Get service metrics | `MetricsRequest` | `MetricsResponse` |
| `LoadModel` | Load an ML model | `LoadModelRequest` | `LoadModelResponse` |
| `StreamSignals` | Stream real-time signals | `StreamSignalsRequest` | `SignalUpdate` (stream) |

## Key Message Types

### GenerateSignalRequest

```protobuf
message GenerateSignalRequest {
  string symbol = 1;              // Trading symbol (e.g., "BTCUSD")
  int32 timeframe = 2;            // Timeframe enum value
  double current_price = 3;       // Current market price
  IndicatorAnalysis analysis = 4; // Technical indicator data
}
```

### TradingSignal

```protobuf
message TradingSignal {
  string signal_id = 1;
  string symbol = 2;
  SignalType signal_type = 3;     // BUY, SELL, HOLD, etc.
  int32 timeframe = 4;
  double strength = 5;            // 0.0 to 1.0
  double confidence = 6;          // 0.0 to 1.0
  SignalSource source = 7;
  int64 timestamp = 8;
  double stop_loss = 10;
  double take_profit = 11;
  double entry_price = 13;
  double position_size = 14;
  string reasoning = 15;
  IndicatorAnalysis indicator_analysis = 16;
}
```

### IndicatorAnalysis

```protobuf
message IndicatorAnalysis {
  double ema_fast = 1;
  double ema_slow = 2;
  double ema_cross = 3;
  double rsi = 4;
  double rsi_signal = 5;
  double macd_line = 6;
  double macd_signal = 7;
  double macd_histogram = 8;
  double bb_upper = 10;
  double bb_middle = 11;
  double bb_lower = 12;
  double atr = 14;
  double trend_strength = 15;
  double volatility = 16;
}
```

## Enums

### SignalType

| Value | Description |
|-------|-------------|
| `STRONG_BUY` | Strong buy signal |
| `BUY` | Buy signal |
| `HOLD` | Hold position |
| `SELL` | Sell signal |
| `STRONG_SELL` | Strong sell signal |

### Timeframe

| Value | Description |
|-------|-------------|
| `TIMEFRAME_1M` | 1 minute |
| `TIMEFRAME_5M` | 5 minutes |
| `TIMEFRAME_15M` | 15 minutes |
| `TIMEFRAME_1H` | 1 hour |
| `TIMEFRAME_4H` | 4 hours |
| `TIMEFRAME_1D` | 1 day |

## Usage Examples

### Generate a Trading Signal

```rust
// Import from local proto (compiled via build.rs)
mod proto {
    tonic::include_proto!("janus.v1");
}

use proto::{
    janus_service_client::JanusServiceClient,
    GenerateSignalRequest,
    IndicatorAnalysis,
    Timeframe,
};

let mut client = JanusServiceClient::connect("http://localhost:50053").await?;

let request = GenerateSignalRequest {
    symbol: "BTCUSD".to_string(),
    timeframe: Timeframe::Timeframe5m as i32,
    current_price: 65000.0,
    analysis: Some(IndicatorAnalysis {
        ema_fast: 64800.0,
        ema_slow: 64500.0,
        rsi: 55.0,
        macd_line: 150.0,
        macd_signal: 120.0,
        ..Default::default()
    }),
};

let response = client.generate_signal(request).await?;
```

### Use Execution Types from fks-proto

```rust
// Import execution types from centralized proto crate
use fks_proto::execution::{
    execution_service_client::ExecutionServiceClient,
    SignalRequest,
    OrderType,
    Side,
    TimeInForce,
    ExecutionStrategy,
};

let signal_request = SignalRequest {
    symbol: "BTCUSD".to_string(),
    side: Side::Buy as i32,
    quantity: 0.1,
    order_type: OrderType::Market as i32,
    ..Default::default()
};
```

## Building

The proto files are compiled during `cargo build`:

```bash
cd src/janus/services/forward
cargo build
```

The `build.rs` script compiles `proto/janus/v1/janus.proto` using `tonic-prost-build`.

## Buf Configuration

This proto follows buf.build conventions:
- Package `janus.v1` matches directory `janus/v1/`
- Linting with DEFAULT rules

Lint the proto:

```bash
cd src/janus/services/forward/proto
buf lint
```

## Integration

The Forward Service integrates with:
- **Execution Service** (`fks-proto::execution`) - Submits orders for execution
- **Data Service** (`fks-proto::data`) - Receives real-time market data
- **CNS Service** (`fks-proto::cns`) - Reports health metrics
- **Backward Service** - Sends experiences for memory consolidation

## Related Documentation

- [Centralized Proto Definitions](../../../../../../proto/README.md) - All FKS proto definitions
- [fks-proto Crate](../../../../../proto/README.md) - Rust proto crate
- [Proto Migration Guide](../../../../../../proto/MIGRATION_GUIDE.md) - Migration instructions

---

**Local Package:** `janus.v1`  
**Centralized Package:** `fks.forward.v1` (via `fks-proto`)  
**Status:** ✅ Production Ready  
**Buf Compliant:** ✅ Yes