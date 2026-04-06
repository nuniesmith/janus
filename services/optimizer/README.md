# JANUS Optimizer Service (Rust)

A high-performance Rust-native parameter optimization service that replaces the Python optimizer. Collects OHLC data from Kraken, runs hyperparameter optimization using TPE sampling, and publishes optimized parameters to Redis for hot-reload by the Forward trading service.

## Quick Start

```bash
# Run optimization for a single asset
janus-optimizer optimize --asset BTC

# Run optimization for multiple assets with quick mode (fewer trials)
janus-optimizer optimize --assets BTC,ETH,SOL --quick

# Start the scheduler daemon (runs every 6 hours)
janus-optimizer run --interval 6h

# Run a single optimization cycle and exit
janus-optimizer run-once --assets BTC,ETH

# Check status
janus-optimizer status -A

# Collect OHLC data without optimization
janus-optimizer collect --days 30
```

## Deployment Model

The optimizer binary (`janus-optimizer`) is **bundled inside the janus Docker container** alongside the main `janus` binary. This allows:

- Single container deployment for all janus services
- Shared data volume for OHLC storage and optimized parameters
- Unified logging and metrics collection
- Simplified orchestration in docker-compose/kubernetes

The optimizer can be run as:
1. A background process within the janus container
2. A separate one-shot job via `docker exec`
3. Manually for testing/debugging

## Overview

The optimizer service provides:

- **Kraken REST API Integration**: Fetches historical and real-time OHLC candle data
- **SQLite Storage**: Efficient local storage for backtesting data
- **TPE Optimization**: Tree-structured Parzen Estimator for intelligent parameter search
- **Redis Hot-Reload**: Publishes optimized params via pub/sub for live updates
- **Prometheus Metrics**: Full observability with `/metrics` endpoint (port 9091)
- **Scheduled Runs**: Configurable optimization intervals with backoff on failure

## Architecture

```
┌─────────────────────────────────────────────────────────────────┐
│                    Kraken REST API                              │
│                  (OHLC Data Collection)                         │
└────────────────────────────┬────────────────────────────────────┘
                             │
                             ▼
┌─────────────────────────────────────────────────────────────────┐
│              OPTIMIZER SERVICE (Rust)                           │
│  ┌─────────────────┐  ┌─────────────────┐  ┌─────────────────┐  │
│  │ Data Collector  │  │   Backtester    │  │   TPE Sampler   │  │
│  │  (Kraken API)   │→ │ (janus-backtest)│← │(janus-optimizer)│  │
│  └────────┬────────┘  └────────┬────────┘  └──────┬──────────┘  │
│           │                    │                   │             │
│           ▼                    ▼                   ▼             │
│  ┌─────────────────┐  ┌─────────────────┐  ┌─────────────────┐  │
│  │    SQLite DB    │  │ Backtest Results│  │ Best Params     │  │
│  │   (OHLC Data)   │  │   (Polars DF)   │  │                 │  │
│  └─────────────────┘  └─────────────────┘  └──────┬──────────┘  │
│                                                    │             │
│  ┌────────────────────────────────────────────────┼───────────┐ │
│  │           Prometheus Metrics (:9092)           │           │ │
│  └────────────────────────────────────────────────┼───────────┘ │
└───────────────────────────────────────────────────┼─────────────┘
                                                    │
                    ┌───────────────────────────────┴──────────┐
                    │              Redis Pub/Sub               │
                    │  fks:{instance}:optimized_params         │
                    │  fks:{instance}:param_updates            │
                    └───────────────────────────────┬──────────┘
                                                    │
                                                    ▼
                    ┌─────────────────────────────────────────┐
                    │           JANUS Forward Service         │
                    │    (Subscribes to param_updates)        │
                    │    (Hot-reloads without restart)        │
                    └─────────────────────────────────────────┘
```

## CLI Commands

The optimizer provides a comprehensive CLI for running optimizations in various modes.

### `optimize` - Run optimization for specific assets

```bash
# Optimize a single asset
janus-optimizer optimize --asset BTC

# Optimize multiple assets
janus-optimizer optimize --assets BTC,ETH,SOL

# Quick mode with fewer trials (20 instead of 100)
janus-optimizer optimize --asset BTC --quick

# Specify number of trials
janus-optimizer optimize --asset BTC --trials 50

# Dry run (don't publish to Redis)
janus-optimizer optimize --asset BTC --dry-run

# Save results to file
janus-optimizer optimize --asset BTC --save --output results.json

# Use specific OHLC interval for backtesting
janus-optimizer optimize --asset BTC --interval 15
```

### `run` - Start the scheduler daemon

```bash
# Run with default 6-hour interval
janus-optimizer run

# Custom interval
janus-optimizer run --interval 2h

# Specify assets and run initial optimization
janus-optimizer run --assets BTC,ETH --run-on-start

# Enable data collection loop
janus-optimizer run --collect-data --collect-interval 5

# Custom metrics port
janus-optimizer run --metrics-port 9093
```

### `run-once` - Run single optimization cycle

```bash
# Run once with default assets
janus-optimizer run-once

# Specific assets with quick mode
janus-optimizer run-once --assets BTC,ETH --quick

# Update data before optimization
janus-optimizer run-once --update-data

# Strict mode (exit with error if any fails)
janus-optimizer run-once --strict

# Fail fast on first error
janus-optimizer run-once --fail-fast
```

### `status` - Check service status

```bash
# Show all status information
janus-optimizer status -A

# Check specific asset
janus-optimizer status --asset BTC

# Check Redis connection only
janus-optimizer status --check-redis

# Check data availability
janus-optimizer status --check-data

# Detailed output
janus-optimizer status -d
```

### `collect` - Collect OHLC data

```bash
# Collect 30 days of historical data
janus-optimizer collect --days 30

# Specific assets
janus-optimizer collect --assets BTC,ETH --days 60

# Specific intervals
janus-optimizer collect --intervals 1,5,15,60

# Force re-download
janus-optimizer collect --force

# Show progress
janus-optimizer collect --progress
```

### `list-assets` - Show configured assets

```bash
# List all assets
janus-optimizer list-assets

# Detailed view
janus-optimizer list-assets -d

# Filter by category
janus-optimizer list-assets --category major

# Show only enabled assets
janus-optimizer list-assets --enabled-only
```

### `history` - View optimization history

```bash
# Show last 10 entries
janus-optimizer history

# Show more entries
janus-optimizer history -n 20

# Filter by asset
janus-optimizer history --asset BTC

# Show only failures
janus-optimizer history --failures-only

# Clear history
janus-optimizer history --clear
```

### Global Options

```bash
# Custom Redis URL
janus-optimizer --redis-url redis://custom:6379 optimize --asset BTC

# Custom instance ID
janus-optimizer --instance-id production optimize --asset BTC

# Verbose output
janus-optimizer -v optimize --asset BTC      # Debug level
janus-optimizer -vv optimize --asset BTC     # Trace level

# Quiet mode (errors only)
janus-optimizer -q run-once

# JSON output
janus-optimizer --format json status
janus-optimizer --format json-compact run-once

# Custom data directory
janus-optimizer --data-dir /custom/path optimize --asset BTC
```

## Building

### From Workspace Root

```bash
# Build both janus and janus-optimizer binaries
cargo build --release -p janus-optimizer-service

# Or build the entire workspace (includes optimizer)
cargo build --release
```

### Docker (Part of Janus Image)

The optimizer is automatically built and included in the janus Docker image:

```bash
# Build the janus image (includes optimizer binary)
docker build -f infrastructure/docker/base/rust/Dockerfile \
  --target workspace \
  --build-arg SERVICE_NAME=janus \
  -t nuniesmith/fks:janus .
```

### Running the Optimizer in Docker

```bash
# Run optimizer inside the janus container
docker exec -it fks_janus /app/janus-optimizer

# Or start it as a background process
docker exec -d fks_janus /app/janus-optimizer

# Run a one-shot optimization
docker exec fks_janus /app/janus-optimizer --run-once
```

### Docker Compose Configuration

The optimizer is configured via environment variables in the janus service:

```yaml
# infrastructure/compose/docker-compose.yml
services:
  janus:
    environment:
      # Optimizer configuration
      - OPTIMIZER_ENABLED=${OPTIMIZER_ENABLED:-true}
      - OPTIMIZER_DATA_DIR=/app/data
      - OPTIMIZER_METRICS_PORT=9091
      - OPTIMIZE_INTERVAL=${OPTIMIZE_INTERVAL:-6h}
      - OPTIMIZE_ASSETS=${OPTIMIZE_ASSETS:-BTC,ETH,SOL}
      - OPTIMIZE_TRIALS=${OPTIMIZE_TRIALS:-100}
    ports:
      - "9091:9091"  # Optimizer metrics
    volumes:
      - optimizer_data:/app/data
```

## Configuration

### Environment Variables

| Variable | Default | Description |
|----------|---------|-------------|
| `DATA_DIR` | `/data` | Directory for OHLC data, params, and results |
| `OPTIMIZE_ASSETS` | `BTC,ETH,SOL` | Comma-separated list of assets to optimize |
| `OPTUNA_TRIALS` | `100` | Number of optimization trials per asset |
| `OPTIMIZATION_INTERVAL` | `6h` | How often to run optimization (e.g., `6h`, `30m`, `1d`) |
| `MIN_DATA_DAYS` | `7` | Minimum days of data required for optimization |
| `PREFERRED_INTERVAL_MINUTES` | `60` | Preferred OHLC interval for backtesting |
| `COLLECTION_INTERVALS` | `1,5,15,60,240,1440` | OHLC intervals to collect |
| `DATA_COLLECTION_ENABLED` | `true` | Enable/disable automatic data collection |
| `DATA_COLLECTION_INTERVAL_MINUTES` | `5` | How often to fetch new data |
| `REDIS_URL` | `redis://localhost:6379` | Redis connection URL |
| `REDIS_INSTANCE_ID` | `default` | Instance ID for Redis key prefixes |
| `OPTIMIZER_METRICS_PORT` | `9091` | Prometheus metrics port |
| `RUN_ON_START` | `true` | Run optimization immediately on startup |
| `N_JOBS` | `<cpu_count>` | Number of parallel optimization jobs |
| `HISTORICAL_DAYS` | `30` | Days of historical data to fetch initially |
| `RUST_LOG` | `info` | Log level (`debug`, `info`, `warn`, `error`) |

## API Endpoints

### Health & Metrics

| Endpoint | Description |
|----------|-------------|
| `GET /health` | Basic health check (200 if healthy) |
| `GET /health/detailed` | Detailed health with component status |
| `GET /metrics` | Prometheus metrics endpoint |
| `GET /status` | Service status with last optimization info |

### Example Responses

**GET /health**
```json
{
  "status": "ok",
  "healthy": true,
  "uptime_seconds": 3600.5,
  "version": "0.1.0"
}
```

**GET /health/detailed**
```json
{
  "status": "ok",
  "healthy": true,
  "uptime_seconds": 3600.5,
  "version": "0.1.0",
  "components": {
    "database": { "status": "ok", "healthy": true },
    "redis": { "status": "ok", "healthy": true },
    "scheduler": { "status": "ok", "healthy": true }
  },
  "config": {
    "assets": ["BTC", "ETH", "SOL"],
    "optimization_interval": "6h",
    "n_trials": 100,
    "data_collection_enabled": true,
    "metrics_port": 9092
  },
  "last_optimization": {
    "started_at": "2025-01-15T12:00:00Z",
    "completed_at": "2025-01-15T12:05:30Z",
    "assets_optimized": ["BTC", "ETH", "SOL"],
    "assets_failed": [],
    "duration_seconds": 330.5
  }
}
```

## Prometheus Metrics

Metrics are exposed on port 9092 (configurable via `OPTIMIZER_METRICS_PORT`) and scraped by Prometheus:

```yaml
# infrastructure/config/prometheus/prometheus.yml
- job_name: "fks-optimizer"
  scrape_interval: 30s
  metrics_path: "/metrics"
  static_configs:
    - targets: ["janus:9092"]
      labels:
        service: "optimizer"
```

Available metrics:

```prometheus
# Optimization metrics
optimizer_optimization_duration_seconds{asset="BTC"}
optimizer_best_score{asset="BTC"}
optimizer_best_return_pct{asset="BTC"}
optimizer_best_win_rate{asset="BTC"}
optimizer_best_max_drawdown{asset="BTC"}
optimizer_trials_total{asset="BTC"}

# Data collection metrics
optimizer_collection_duration_seconds
optimizer_collection_success_total
optimizer_collection_failure_total
optimizer_candles_collected_total{asset="BTC", interval="60"}

# Service metrics
optimizer_healthy
optimizer_uptime_seconds
optimizer_scheduled_runs_total
optimizer_scheduled_runs_success_total
optimizer_scheduled_runs_failure_total
optimizer_last_scheduled_duration_seconds
```

A Grafana dashboard is available at `infrastructure/config/grafana/dashboards/optimizer.json`.

## Redis Integration

### Data Structure

Parameters are stored in a Redis hash:

```
fks:{instance}:optimized_params
├── BTC: { json params }
├── ETH: { json params }
├── SOL: { json params }
└── _last_updated: "2025-01-15T12:00:00Z"
```

### Pub/Sub Notifications

The optimizer publishes to channel `fks:{instance}:param_updates`:

```json
// param_update - New params for a single asset
{
  "type": "param_update",
  "asset": "BTC",
  "timestamp": "2025-01-15T12:05:30Z",
  "params": { ... }
}

// optimization_started - Cycle beginning
{
  "type": "optimization_started",
  "timestamp": "2025-01-15T12:00:00Z",
  "assets": ["BTC", "ETH", "SOL"]
}

// optimization_complete - Cycle finished
{
  "type": "optimization_complete",
  "timestamp": "2025-01-15T12:05:30Z",
  "successful": 3,
  "failed": 0,
  "assets": ["BTC", "ETH", "SOL"]
}

// optimization_failed - Asset optimization failed
{
  "type": "optimization_failed",
  "timestamp": "2025-01-15T12:03:00Z",
  "asset": "SOL",
  "error": "Insufficient data"
}
```

## Output Files

### JSON Parameters

```
/data/optimized_params/
├── btc_params.json      # Individual asset params
├── eth_params.json
└── sol_params.json
```

### Environment Format

```
/data/optimized_params/
├── btc_params.env       # Copy-paste ready env vars
├── eth_params.env
└── sol_params.env
```

## Asset Categories

Different asset classes have different parameter constraints:

| Category | Assets | Min EMA Spread | Min Hold Time |
|----------|--------|----------------|---------------|
| Major | BTC, ETH, SOL | 0.15% | 15 min |
| L1/L2 | AVAX, DOT, ATOM | 0.18% | 18 min |
| DeFi | UNI, AAVE, LINK | 0.20% | 20 min |
| Meme | DOGE, PEPE, SHIB | 0.30% | 30 min |

## Comparison with Python Optimizer

| Feature | Python | Rust |
|---------|--------|------|
| Runtime | ~5 min/asset | ~30 sec/asset |
| Memory | ~500 MB | ~50 MB |
| Dependencies | Optuna, Pandas, NumPy | Native Rust |
| Data Processing | Pandas DataFrame | Polars (Apache Arrow) |
| Parallelization | GIL-limited | Rayon (true parallel) |
| Binary Size | ~200 MB (container) | ~15 MB (container) |
| Startup Time | ~5 sec | ~100 ms |

## Development

### Running Locally

```bash
# Set environment variables
export DATA_DIR=/tmp/optimizer-data
export REDIS_URL=redis://localhost:6379
export RUST_LOG=debug

# Run in daemon mode (original behavior)
cargo run -p janus-optimizer-service

# Run CLI commands
cargo run -p janus-optimizer-service -- optimize --asset BTC --quick
cargo run -p janus-optimizer-service -- status -A
cargo run -p janus-optimizer-service -- run-once --assets BTC,ETH
```

### Running Tests

```bash
# Run optimizer service tests (42 tests)
cargo test -p janus-optimizer-service

# Run optimizer crate tests (100 tests)
cargo test -p janus-optimizer

# Run forward service tests (216 tests)
cargo test -p janus-forward --lib

# Run integration tests (requires Redis)
cargo test -p janus-forward --test param_reload_integration
```

### Running with Docker Compose (Development)

```bash
# Start all services including janus with optimizer
docker compose -f infrastructure/compose/docker-compose.yml up -d

# View optimizer logs
docker logs -f fks_janus 2>&1 | grep optimizer

# Run optimizer manually
docker exec -it fks_janus /app/janus-optimizer --run-once
```

### Logs

The service uses `tracing` for structured logging:

```
2025-01-15T12:00:00Z INFO  janus_optimizer::main  - JANUS Optimizer Service Starting
2025-01-15T12:00:01Z INFO  janus_optimizer::service - ✅ Connected to Redis at redis://localhost:6379
2025-01-15T12:00:02Z INFO  janus_optimizer::collector - Fetching 30 days of historical data for BTC interval=60
2025-01-15T12:00:05Z DEBUG janus_optimizer::collector - Fetched 720 candles for BTC interval=60
2025-01-15T12:00:10Z INFO  janus_optimizer::service - Starting optimization for BTC
2025-01-15T12:00:40Z INFO  janus_optimizer::service - Optimization complete for BTC: score=42.50, return=15.30%, win_rate=65.0%
2025-01-15T12:00:40Z INFO  janus_optimizer::service - 📡 Published BTC params to Redis (subscribers: 1)
```

## Troubleshooting

### No data collected

1. Check Kraken API connectivity: `curl https://api.kraken.com/0/public/Time`
2. Verify asset symbols are valid Kraken pairs
3. Check rate limiting (1 request/second by default)
4. Review logs for API errors

### Optimization skipped

1. Check minimum data requirement (`MIN_DATA_DAYS`)
2. Verify data exists: `sqlite3 /data/db/ohlc.db "SELECT COUNT(*) FROM ohlc_candles"`
3. Check logs for "Insufficient data" messages

### Redis connection failed

1. Verify Redis is running: `redis-cli ping`
2. Check `REDIS_URL` format
3. Service will continue without Redis (file-only mode)

### Forward service not receiving updates

1. Verify Redis pub/sub: `redis-cli SUBSCRIBE fks:default:param_updates`
2. Check `REDIS_INSTANCE_ID` matches between optimizer and Forward
3. Review Forward service logs for subscription errors

## Migration from Python Optimizer

The Python optimizer at `src/optimizer/` has been removed. To migrate:

1. **No code changes required** - The Rust optimizer uses the same Redis keys and pub/sub channels
2. **Environment variables** - Most are compatible; see the table above for any renamed variables
3. **Docker** - The optimizer is now part of the janus image, no separate container needed
4. **Data** - OHLC data is stored in SQLite at `$DATA_DIR/db/ohlc.db`

## Integration Tests

The project includes comprehensive integration tests for the hot-reload flow:

```bash
# Run integration tests (requires Redis at redis://127.0.0.1:6379)
cargo test -p janus-forward --test param_reload_integration

# Run with custom Redis URL
REDIS_URL=redis://custom:6379 cargo test -p janus-forward --test param_reload_integration
```

### Test Coverage

| Test Category | Tests | Description |
|---------------|-------|-------------|
| Unit Tests | 10 | Applier conversions, handle creation, component wiring |
| Integration | 14 | Pub/sub flow, stats tracking, concurrent updates |
| Edge Cases | Included | Rapid updates, isolation, position limits, constraints |
| HA Scenarios | Included | Multiple managers subscribing to same instance |

### Key Integration Tests

- `test_publish_and_receive_params` - Full pub/sub cycle
- `test_full_hot_reload_flow` - End-to-end: publish → subscribe → apply → use
- `test_concurrent_param_updates` - Multiple assets updated simultaneously
- `test_rapid_param_version_updates` - Rapid version changes
- `test_multiple_managers_same_instance` - HA setup with multiple subscribers

## License

MIT License - See LICENSE file in project root