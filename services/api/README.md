# JANUS Rust Gateway

High-performance API gateway for Project JANUS, replacing the Python FastAPI gateway.

## Overview

The Rust Gateway (`janus-gateway`) is a drop-in replacement for the Python gateway service. It provides:

- **REST API endpoints** for signal management, health checks, and monitoring
- **Redis Pub/Sub** for signal dispatch to the Forward service
- **Dead Man's Switch** heartbeat mechanism
- **Prometheus metrics** at `/metrics` for observability
- **Rate limiting** with configurable limits per endpoint
- **gRPC-Web proxy** for Kotlin/JS clients (via tonic-web)
- **Sub-millisecond latency** compared to Python's ~10ms overhead

## Architecture

```
┌─────────────────┐     ┌─────────────────┐
│  Web Client     │     │  Mobile Client  │
│  (Kotlin/JS)    │     │  (KMP)          │
└────────┬────────┘     └────────┬────────┘
         │                       │
         │  HTTP/gRPC-Web        │
         ▼                       ▼
┌─────────────────────────────────────────┐
│         janus-gateway (Rust)            │
│  • REST API (Axum)                      │
│  • gRPC-Web proxy (tonic-web)           │
│  • Redis Pub/Sub (signal dispatch)      │
│  • Dead Man's Switch (heartbeat)        │
│  • Prometheus metrics                   │
│  • Rate limiting (governor)             │
└────────┬───────────────────┬────────────┘
         │                   │
         │ gRPC              │ Redis Pub/Sub
         ▼                   ▼
┌─────────────────┐   ┌─────────────────┐
│  janus-forward  │   │  janus-backward │
│  (Live Trading) │   │  (Training)     │
└─────────────────┘   └─────────────────┘
```

## Quick Start

### Build

```bash
cd src/janus
cargo build --release --package janus-gateway
```

### Run

```bash
# With defaults (port 8000)
cargo run --release --package janus-gateway

# With custom configuration
PORT=8001 REDIS_SIGNAL_URL=redis://localhost:6379/0 cargo run --release --package janus-gateway
```

### Docker

```bash
docker build -t janus-gateway -f docker/rust/Dockerfile --target workspace --build-arg SERVICE_NAME=gateway .
docker run -p 8000:8000 janus-gateway
```

## Configuration

Configuration is loaded from environment variables:

| Variable | Default | Description |
|----------|---------|-------------|
| `SERVICE_NAME` | `janus-gateway` | Service identifier |
| `PORT` | `8000` | HTTP server port |
| `ENVIRONMENT` | `development` | Environment (development/staging/production) |
| `JANUS_FORWARD_URL` | `localhost:50051` | Forward service gRPC URL |
| `JANUS_BACKWARD_URL` | `localhost:50052` | Backward service gRPC URL |
| `REDIS_SIGNAL_URL` | `redis://localhost:6379/0` | Redis URL for Pub/Sub |
| `CORS_ORIGINS` | `*` | Comma-separated CORS origins |
| `HEARTBEAT_INTERVAL_SECS` | `2` | Dead Man's Switch interval |

## API Endpoints

### Health

| Method | Path | Description |
|--------|------|-------------|
| GET | `/` | Root endpoint with service info |
| GET | `/health` | Alias for health check (Docker) |
| GET | `/api/v1/health` | Full health check with components |
| GET | `/api/v1/health/ready` | Readiness probe |
| GET | `/api/v1/health/live` | Liveness probe |
| GET | `/api/v1/test/hello` | Simple connectivity test |

### Signals

| Method | Path | Description |
|--------|------|-------------|
| GET | `/api/signals/generate` | Trigger signal generation |
| POST | `/api/signals/dispatch` | Manually dispatch a signal |
| GET | `/api/signals/from-files` | Load signals from JSON files |
| GET | `/api/signals/by-id/{id}` | Get signal by ID |
| GET | `/api/signals/by-symbol/{symbol}` | Get signals for a symbol |
| GET | `/api/signals/summary` | Get signal statistics |
| GET | `/api/signals/categories` | List trade categories |

### Metrics

| Method | Path | Description |
|--------|------|-------------|
| GET | `/metrics` | Prometheus metrics (text format) |
| GET | `/api/v1/metrics` | Alias for metrics endpoint |

### Example: Dispatch a Signal

```bash
curl -X POST http://localhost:8000/api/signals/dispatch \
  -H "Content-Type: application/json" \
  -d '{
    "symbol": "BTCUSD",
    "side": "Buy",
    "strength": 0.8,
    "confidence": 0.9,
    "entry_price": 50000.0,
    "stop_loss": 49000.0,
    "take_profit": 52000.0
  }'
```

## Prometheus Metrics

The gateway exposes the following metrics at `/metrics`:

### HTTP Metrics
- `janus_gateway_http_requests_total{method, path, status}` - Total HTTP requests
- `janus_gateway_http_request_duration_seconds{method, path}` - Request duration histogram
- `janus_gateway_http_requests_in_flight` - Currently active requests

### Signal Metrics
- `janus_gateway_signals_dispatched_total{symbol, side}` - Signals sent
- `janus_gateway_signal_dispatch_errors_total` - Signal dispatch errors

### Connection Metrics
- `janus_gateway_redis_connected` - Redis connection status (1/0)
- `janus_gateway_grpc_connected` - gRPC connection status (1/0)

### Heartbeat Metrics
- `janus_gateway_heartbeats_sent_total` - Dead Man's Switch heartbeats sent
- `janus_gateway_heartbeat_errors_total` - Heartbeat errors

### System Metrics
- `janus_gateway_uptime_seconds` - Service uptime

### Example Prometheus Query

```promql
# Request rate by endpoint
rate(janus_gateway_http_requests_total[5m])

# 99th percentile latency
histogram_quantile(0.99, rate(janus_gateway_http_request_duration_seconds_bucket[5m]))

# Signal dispatch rate by symbol
rate(janus_gateway_signals_dispatched_total[1m])
```

## Rate Limiting

The gateway implements global and endpoint-specific rate limiting:

### Global Limits
- **Production**: 100 requests/second with burst of 50
- **Development**: 1000 requests/second with burst of 500

### Endpoint-Specific Limits
- **Signal dispatch**: 10 requests/second with burst of 5
- **Signal generation**: 1 request/second with burst of 2

### Rate Limit Response

When rate limited, the gateway returns:

```json
{
  "error": "rate_limit_exceeded",
  "message": "Too many requests. Please slow down.",
  "retry_after_secs": 1
}
```

With HTTP status `429 Too Many Requests` and `Retry-After` header.

## Migration from Python Gateway

The Rust gateway is API-compatible with the Python gateway. To migrate:

1. **Update docker-compose.yml**: Change the gateway service image
2. **Update environment variables**: Same variables are supported
3. **No client changes needed**: API endpoints are identical

### Feature Parity

| Feature | Python | Rust |
|---------|--------|------|
| Health endpoints | ✅ | ✅ |
| Signal dispatch | ✅ | ✅ |
| Redis Pub/Sub | ✅ | ✅ |
| Heartbeat (DMS) | ✅ | ✅ |
| CORS | ✅ | ✅ |
| Prometheus metrics | ❌ | ✅ |
| Rate limiting | ❌ | ✅ |
| gRPC-Web proxy | ❌ (Envoy) | ✅ (native) |
| Celery tasks | ✅ | ❌ (use Redis jobs) |

## Development

### Run Tests

```bash
cargo test --package janus-gateway
```

### Code Structure

```
services/gateway-rs/
├── Cargo.toml              # Dependencies
├── README.md               # This file
└── src/
    ├── main.rs             # Entry point, server setup
    ├── lib.rs              # Library exports
    ├── config.rs           # Configuration from environment
    ├── state.rs            # Shared application state
    ├── redis_dispatcher.rs # Redis Pub/Sub signal dispatch
    ├── metrics.rs          # Prometheus metrics collection
    ├── rate_limit.rs       # Request rate limiting
    ├── grpc.rs             # gRPC-Web support (tonic-web)
    └── routes/
        ├── mod.rs          # Route module index
        ├── health.rs       # Health check endpoints
        └── signals.rs      # Signal API endpoints
```

## Performance

| Metric | Python (FastAPI) | Rust (Axum) | Improvement |
|--------|------------------|-------------|-------------|
| Request latency (p50) | ~10ms | <1ms | **10x** |
| Request latency (p99) | ~50ms | <5ms | **10x** |
| Memory usage | ~150MB | ~15MB | **10x** |
| Startup time | ~3s | <100ms | **30x** |
| Signal dispatch | ~5ms | <100µs | **50x** |
| Requests/sec | ~5K | ~100K | **20x** |

## Roadmap

### Phase 1 ✅
- [x] Basic Axum server
- [x] Health endpoints
- [x] Signal dispatch via Redis
- [x] Heartbeat task
- [x] CORS middleware

### Phase 1.1 ✅
- [x] Prometheus metrics endpoint
- [x] Request duration tracking
- [x] Connection status metrics
- [x] Rate limiting (global)
- [x] Rate limiting (per-endpoint)
- [x] gRPC-Web layer setup

### Phase 2 (Planned)
- [ ] Full gRPC client to Forward/Backward services
- [ ] Manual trading endpoints
- [ ] WebSocket streaming
- [ ] JWT authentication middleware
- [ ] Structured JSON logging

### Phase 3 (Planned)
- [ ] OpenTelemetry tracing
- [ ] Circuit breaker patterns
- [ ] Request caching
- [ ] API versioning

## Troubleshooting

### Gateway won't start

1. Check if port 8000 is available: `lsof -i :8000`
2. Verify Redis is running: `redis-cli ping`
3. Check logs: `RUST_LOG=debug cargo run --package janus-gateway`

### Rate limiting too aggressive

Adjust limits via environment or modify `RateLimitConfig` in code:

```bash
# In development, limits are automatically higher
ENVIRONMENT=development cargo run --package janus-gateway
```

### Metrics not showing

1. Ensure you're hitting `/metrics` (not `/metric`)
2. Check response headers for `text/plain; version=0.0.4`
3. Verify Prometheus can scrape: `curl http://localhost:8000/metrics`

### Redis connection failing

```bash
# Test Redis connectivity
redis-cli -u redis://localhost:6379/0 ping

# Check gateway logs for connection errors
RUST_LOG=info cargo run --package janus-gateway 2>&1 | grep -i redis
```

## License

MIT License - see [LICENSE](../../../../LICENSE) for details.