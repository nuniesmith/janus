# Forward → Execution Integration

This module provides gRPC client integration between the JANUS Forward service (signal generation) and the FKS Execution service (order execution).

## Overview

The execution client allows the forward service to automatically submit trading signals for execution, enabling end-to-end automated trading.

```
Signal Generation → Execution Client → Execution Service → Exchange
```

## Components

### `client.rs`

**ExecutionClient** - gRPC client for submitting signals to the execution service.

**Features:**
- Automatic retry with exponential backoff
- Configurable timeouts and connection settings
- Signal-to-order conversion
- Health check integration
- Comprehensive error handling

**ExecutionClientConfig** - Configuration for the client:
```rust
ExecutionClientConfig {
    endpoint: String,              // e.g., "http://execution:50052"
    connect_timeout_secs: u64,     // Connection timeout
    request_timeout_secs: u64,     // Request timeout
    enable_tls: bool,              // Enable TLS
    max_retries: u32,              // Max retry attempts
    retry_backoff_ms: u64,         // Backoff delay in ms
}
```

## Usage

### Basic Example

```rust
use janus_forward::execution::{ExecutionClient, ExecutionClientConfig};
use janus_forward::signal::TradingSignal;

#[tokio::main]
async fn main() -> Result<(), Box<dyn std::error::Error>> {
    // Create client configuration
    let config = ExecutionClientConfig {
        endpoint: "http://execution:50052".to_string(),
        connect_timeout_secs: 10,
        request_timeout_secs: 30,
        enable_tls: false,
        max_retries: 3,
        retry_backoff_ms: 100,
    };

    // Connect to execution service
    let mut client = ExecutionClient::new(config).await?;

    // Submit a signal
    let signal = TradingSignal {
        signal_id: "sig-123".to_string(),
        symbol: "BTCUSD".to_string(),
        // ... other fields ...
    };

    let response = client.submit_signal(&signal).await?;
    
    println!("Order ID: {:?}", response.order_id);
    println!("Success: {}", response.success);

    Ok(())
}
```

### Integration with Signal Generator

```rust
use janus_forward::signal::SignalGenerator;
use janus_forward::execution::{ExecutionClient, ExecutionClientConfig};
use std::sync::Arc;
use tokio::sync::Mutex;

pub struct SignalGeneratorWithExecution {
    generator: SignalGenerator,
    execution_client: Option<Arc<Mutex<ExecutionClient>>>,
}

impl SignalGeneratorWithExecution {
    pub async fn new(
        generator_config: SignalGeneratorConfig,
        execution_config: Option<ExecutionClientConfig>,
    ) -> Result<Self> {
        let generator = SignalGenerator::new(generator_config);
        
        let execution_client = if let Some(config) = execution_config {
            let client = ExecutionClient::new(config).await?;
            Some(Arc::new(Mutex::new(client)))
        } else {
            None
        };

        Ok(Self {
            generator,
            execution_client,
        })
    }

    async fn submit_signal(&self, signal: &TradingSignal) -> Result<()> {
        if let Some(client) = &self.execution_client {
            let mut client = client.lock().await;
            client.submit_signal(signal).await?;
        }
        Ok(())
    }
}
```

## Environment Variables

When using with the forward service, configure via environment variables:

```bash
# Enable execution client
ENABLE_EXECUTION=true

# Execution service endpoint
EXECUTION_ENDPOINT=http://execution:50052

# Timeouts (seconds)
EXECUTION_CONNECT_TIMEOUT=10
EXECUTION_REQUEST_TIMEOUT=30

# Retry configuration
EXECUTION_MAX_RETRIES=3
EXECUTION_RETRY_BACKOFF_MS=100

# TLS (optional)
EXECUTION_ENABLE_TLS=false
```

## Signal to Order Conversion

The client automatically converts TradingSignal to execution SignalRequest:

| Signal Field | Order Field | Notes |
|--------------|-------------|-------|
| `signal_id` | `signal_id` | Preserved for tracking |
| `symbol` | `symbol` | Trading pair |
| `exchange` | `exchange` | Default: "bybit" |
| `signal_type` | `side` | Buy/Sell/Long/Short → BUY/SELL |
| `position_size.quantity` | `quantity` | Required |
| `entry_price` | `price` | Optional (limit orders) |
| `stop_loss` | `stop_price` | Optional |
| `confidence` | `metadata` | Included for reference |
| `strength` | `metadata` | Included for reference |
| `strategy` | `metadata` | Included for reference |

## Error Handling

The client provides comprehensive error handling:

- **Connection Errors**: Automatic retry with backoff
- **Timeout Errors**: Configurable timeouts
- **Validation Errors**: Signal validation before submission
- **Execution Errors**: Detailed error messages from service

Example:
```rust
match client.submit_signal(&signal).await {
    Ok(response) => {
        if response.success {
            println!("Order created: {}", response.order_id.unwrap());
        } else {
            eprintln!("Order rejected: {}", response.message);
        }
    }
    Err(e) => {
        eprintln!("Failed to submit signal: {}", e);
        // Handle error (log, retry, alert, etc.)
    }
}
```

## Health Checks

Check if the execution service is available:

```rust
if client.health_check().await? {
    println!("Execution service is healthy");
} else {
    eprintln!("Execution service is down");
}
```

## Testing

Run tests:
```bash
cargo test --package janus-forward --lib execution
```

## Monitoring

The client logs all operations:
- Signal submissions (DEBUG level)
- Successful orders (INFO level)
- Errors and retries (WARN/ERROR level)

Example log output:
```
[INFO] Connected to execution service
[DEBUG] Submitting signal sig-123 for BTCUSD (Buy)
[INFO] ✅ Signal sig-123 submitted successfully (order_id: ord-456)
```

## Performance

- **Latency**: ~5-20ms per submission (local network)
- **Throughput**: Handles 100+ signals/second
- **Retries**: Automatic with exponential backoff
- **Connection**: Persistent gRPC channel

## Security

- TLS support for encrypted communication
- No credentials stored in client
- API keys managed by execution service
- All operations logged for audit

## Documentation

For more details, see:
- **Implementation Guide**: `FORWARD_EXECUTION_INTEGRATION.md`
- **System Overview**: `TRADING_SYSTEM_IMPLEMENTATION_GUIDE.md`
- **Testing Guide**: `NEXT_STEPS_CHECKLIST.md`

## Support

For issues or questions:
1. Check execution service logs: `docker compose logs execution`
2. Verify connectivity: `curl http://execution:50052`
3. Review health status: `curl http://localhost:8081/health`
4. See troubleshooting guide in `FORWARD_EXECUTION_INTEGRATION.md`
