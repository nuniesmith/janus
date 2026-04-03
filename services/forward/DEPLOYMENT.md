# JANUS Forward Service — Deployment Guide

This document covers deploying the JANUS Forward Service, the Regime Bridge
Consumer, and the supporting infrastructure (Redis, Prometheus, Grafana).

---

## Table of Contents

1. [Architecture Overview](#architecture-overview)
2. [Environment Variables Reference](#environment-variables-reference)
3. [Building](#building)
4. [Systemd Deployment](#systemd-deployment)
5. [Docker / Container Deployment](#docker--container-deployment)
6. [Prometheus & Grafana](#prometheus--grafana)
7. [Operational Runbook](#operational-runbook)

---

## Architecture Overview

```text
                         ┌──────────────────────────┐
                         │  Bybit WebSocket (V5)    │
                         └────────────┬─────────────┘
                                      │ ticks
                                      ▼
┌─────────────────────────────────────────────────────────────────────┐
│                   janus-forward-production                          │
│                                                                     │
│  Tick → CandleAggregator → EnhancedRouter → RoutedSignal           │
│           │                       │                                  │
│           │ volume                │ regime                           │
│           ▼                       ▼                                  │
│  RelativeVolume calc       RegimeBridge                              │
│                              │        │                              │
│                    broadcast │        │ Prometheus metrics           │
│                              ▼        ▼                              │
│                    ┌──────────────┐  :9090/metrics                   │
│                    │ bridge_task  │                                   │
│                    └──────┬───────┘                                   │
│                           │ XADD (optional, if REDIS_URL set)       │
└───────────────────────────┼─────────────────────────────────────────┘
                            │
                            ▼
                   ┌─────────────────┐
                   │  Redis Stream   │
                   │  janus:regime:  │
                   │  bridge         │
                   └────────┬────────┘
                            │ XREADGROUP
                            ▼
               ┌─────────────────────────┐
               │ regime-bridge-consumer  │
               │                         │
               │  LoggingHandler (default)│
               │  — or —                 │
               │  GrpcForwarderHandler   │──► Remote neuromorphic gRPC
               └─────────────────────────┘
```

---

## Environment Variables Reference

### janus-forward-production

| Variable | Default | Description |
|----------|---------|-------------|
| `SYMBOL` | `BTCUSD` | Trading symbol |
| `BYBIT_TESTNET` | `true` | Use Bybit testnet (`true`) or mainnet (`false`) |
| `BYBIT_API_KEY` | *(required)* | Bybit API key |
| `BYBIT_API_SECRET` | *(required)* | Bybit API secret |
| `QUESTDB_HOST` | `127.0.0.1` | QuestDB ILP host |
| `QUESTDB_PORT` | `9009` | QuestDB ILP port |
| `ACCOUNT_SIZE` | `10000.0` | Account capital (USD) |
| `CHALLENGE_TYPE` | `OneStep` | Prop firm challenge type (`OneStep`, `TwoStep`, `Funded`) |
| `TRADING_ENABLED` | `false` | Enable live order execution |
| `MAX_RISK_PER_TRADE` | `0.01` | Max risk per trade (fraction, e.g. `0.01` = 1%) |
| `SESSION_START_HOURS_UTC` | `0,8,13` | Comma-separated session start hours (UTC) |
| `REGIME_TOML_PATH` | *(unset)* | Path to `regime.toml` config file. When set, also enables the hot-reload file watcher |
| `REGIME_RELOAD_INTERVAL_SECS` | `30` | How often (seconds) the config watcher checks `regime.toml` for changes. Only active when `REGIME_TOML_PATH` is set |
| `REDIS_URL` | *(unset)* | Redis URL for cross-process regime bridge publishing |
| `REGIME_BRIDGE_STREAM` | `janus:regime:bridge` | Redis stream key for bridged regime states |
| `REGIME_GRPC_PORT` | *(unset)* | TCP port for regime bridge gRPC server (e.g. `50052`). When set, starts `RegimeBridgeService` serving `StreamRegimeUpdates`, `GetCurrentRegime`, and `PushRegimeState` |
| `REGIME_GRPC_AUTH_TOKEN` | *(unset)* | Shared secret for gRPC bearer-token authentication. When set, every RPC must include `authorization: Bearer <token>` metadata. When unset, auth is disabled (backward-compatible) |
| `REGIME_GRPC_TLS_CERT` | *(unset)* | Path to PEM-encoded server certificate for TLS/mTLS (requires `tls` feature) |
| `REGIME_GRPC_TLS_KEY` | *(unset)* | Path to PEM-encoded server private key for TLS/mTLS (requires `tls` feature) |
| `REGIME_GRPC_TLS_CA` | *(unset)* | Path to PEM-encoded CA certificate for client verification — enables mutual TLS (requires `tls` feature) |
| `REGIME_ARCHIVE_ENABLED` | `true` | Persist every bridged regime state to QuestDB `regime_states` table for historical replay and model training. Set to `false` to disable |
| `METRICS_PORT` | `9090` | Prometheus metrics HTTP port |
| `RUST_LOG` | `info,janus_forward=debug` | Tracing log level filter |

### regime-bridge-consumer

| Variable | Default | Description |
|----------|---------|-------------|
| `REDIS_URL` | `redis://127.0.0.1:6379` | Redis connection URL |
| `REGIME_BRIDGE_STREAM` | `janus:regime:bridge` | Redis stream key to read from |
| `CONSUMER_GROUP` | `neuromorphic` | Consumer group name |
| `CONSUMER_NAME` | `consumer-{pid}` | Unique consumer name within the group |
| `BLOCK_MS` | `5000` | XREADGROUP block timeout (ms) |
| `BATCH_SIZE` | `10` | Max messages per XREADGROUP call |
| `PEL_SWEEP_INTERVAL_SECS` | `60` | Seconds between periodic PEL (Pending Entries List) sweeps. Reclaims stale entries from crashed consumers via `XAUTOCLAIM`. Set to `0` to disable periodic sweeps (startup drain still runs). |
| `PEL_MIN_IDLE_MS` | `30000` | Minimum idle time (ms) before a pending entry is eligible for reclamation from another consumer. Entries idle for less than this are assumed to still be actively processed. |
| `PEL_BATCH_SIZE` | `100` | Max entries to reclaim per `XAUTOCLAIM` call. |
| `GRPC_TARGET` | *(unset)* | gRPC endpoint for forwarding (e.g. `http://host:50051`). When set, uses `GrpcForwarderHandler` |
| `GRPC_SOURCE_ID` | `regime-bridge-consumer-{pid}` | Source identifier in gRPC push requests |
| `SERVE_PORT` | *(unset)* | TCP port for an embedded gRPC server (e.g. `50051`). When set, uses `ServerHandler` — neuromorphic consumers connect here via `StreamRegimeUpdates` / `GetCurrentRegime`. Takes precedence over `LoggingHandler` but not `GRPC_TARGET`. |
| `RUST_LOG` | `info` | Tracing log level filter |

#### PEL (Pending Entries List) Reclamation

The consumer automatically recovers from crashes using Redis Stream consumer group semantics:

1. **Startup drain**: On launch, the consumer re-reads and processes any entries that were delivered to it but never ACKed (e.g. due to a crash between `XREADGROUP` and `XACK`). This uses `XREADGROUP` with start ID `"0"` to fetch all pending entries assigned to this consumer.

2. **Periodic sweep**: A background sweep runs every `PEL_SWEEP_INTERVAL_SECS` seconds and uses `XAUTOCLAIM` to reclaim entries that have been idle for longer than `PEL_MIN_IDLE_MS` from **other** consumers in the same group that may have crashed. This ensures no messages are permanently lost even if a consumer dies without restarting.

> **Note**: `XAUTOCLAIM` requires Redis >= 6.2. On older Redis versions, the periodic sweep is silently skipped (the startup drain still works on all versions).

```bash
# Example: aggressive reclamation for low-latency pipelines
PEL_SWEEP_INTERVAL_SECS=15 \
  PEL_MIN_IDLE_MS=10000 \
  PEL_BATCH_SIZE=200 \
  REDIS_URL=redis://127.0.0.1:6379 \
  SERVE_PORT=50051 \
  cargo run --bin regime-bridge-consumer

# Example: disable periodic sweeps (startup drain only)
PEL_SWEEP_INTERVAL_SECS=0 \
  REDIS_URL=redis://127.0.0.1:6379 \
  cargo run --bin regime-bridge-consumer
```

#### Handler selection priority

The consumer selects a handler based on which environment variables are set:

| Priority | Env Var | Handler | Description |
|----------|---------|---------|-------------|
| 1 | `GRPC_TARGET` | `GrpcForwarderHandler` | Forwards each state to a remote gRPC server via `PushRegimeState` |
| 2 | `SERVE_PORT` | `ServerHandler` | Starts an embedded `RegimeBridgeServer` on the given port. Neuromorphic subsystems connect directly to the consumer and call `StreamRegimeUpdates`, `GetCurrentRegime`, etc. |
| 3 | *(neither)* | `LoggingHandler` | Logs each state to stdout (development / debugging) |

```bash
# Mode 1: Forward to a remote neuromorphic gRPC server
REDIS_URL=redis://127.0.0.1:6379 \
  GRPC_TARGET=http://hypothalamus-svc:50051 \
  cargo run --bin regime-bridge-consumer

# Mode 2: Serve neuromorphic consumers directly (no external server needed)
REDIS_URL=redis://127.0.0.1:6379 \
  SERVE_PORT=50051 \
  cargo run --bin regime-bridge-consumer

# Mode 3: Logging only (default)
REDIS_URL=redis://127.0.0.1:6379 \
  cargo run --bin regime-bridge-consumer
```

### Regime Bridge gRPC Server (embedded in forward service)

#### Authentication

The gRPC server supports two layers of authentication, configured via environment variables:

1. **Bearer token** (`REGIME_GRPC_AUTH_TOKEN`): When set, every RPC must include an `authorization: Bearer <token>` metadata header. Uses constant-time comparison to prevent timing attacks. When unset, all requests pass through (no-op interceptor).

2. **TLS / mTLS** (`REGIME_GRPC_TLS_CERT` + `REGIME_GRPC_TLS_KEY`, optionally `REGIME_GRPC_TLS_CA`): Requires the `tls` crate feature (`cargo build --features tls`). When a CA certificate is also provided, the server requires clients to present a valid certificate signed by that CA (mutual TLS).

Both layers can be enabled simultaneously — token **and** TLS/mTLS are enforced together.

##### Multi-Token Support & Zero-Downtime Rotation

`REGIME_GRPC_AUTH_TOKEN` accepts a **comma-separated** list of tokens for zero-downtime rotation:

```bash
# Multiple tokens — any one is accepted
REGIME_GRPC_AUTH_TOKEN=old-secret,new-secret
```

**Zero-downtime rotation workflow:**

1. Add the new token alongside the old: `REGIME_GRPC_AUTH_TOKEN=old-token,new-token` → restart
2. Migrate all clients to use `new-token`
3. Remove the old token: `REGIME_GRPC_AUTH_TOKEN=new-token` → restart

**Runtime rotation** (no restart required):

Tokens can also be added/revoked/replaced at runtime via the `AuthInterceptor` API:

```rust
// Add a new token (old tokens remain valid)
interceptor.add_token("new-secret-v2").await;

// Revoke an old token
interceptor.revoke_token("old-secret-v1").await;

// Atomically replace all tokens
interceptor.replace_tokens(vec!["fresh-token".into()]).await;
```

##### Example: bearer token only (recommended for internal networks)

```bash
REGIME_GRPC_PORT=50052 \
  REGIME_GRPC_AUTH_TOKEN=my-secret-token-42 \
  cargo run --bin janus-forward-production
```

##### Example: connecting with grpcurl (authenticated)

```bash
grpcurl -plaintext \
  -H 'authorization: Bearer my-secret-token-42' \
  -d '{"client_id":"monitor-1"}' \
  localhost:50052 janus.v1.bridge.RegimeBridgeService/StreamRegimeUpdates
```

##### Example: injecting auth in Rust client code

```rust
use janus_forward::regime_bridge_auth::inject_auth_metadata;

let mut request = tonic::Request::new(payload);
inject_auth_metadata(&mut request, "my-secret-token-42").unwrap();
let response = client.push_regime_state(request).await?;
```

#### Persistent Archiving (QuestDB)

When `REGIME_ARCHIVE_ENABLED=true` (the default), the forward service's in-process regime bridge consumer writes every `BridgedRegimeState` to the QuestDB `regime_states` table via ILP. This table uses the following schema:

- **Tags (indexed):** `symbol`, `hypothalamus` (regime name), `amygdala` (regime name)
- **Fields:** `position_scale`, `is_high_risk`, `confidence`, `trend`, `trend_strength`, `volatility`, `volatility_percentile`, `momentum`, `relative_volume`, `liquidity_score`, `is_transition`
- **Timestamp:** nanosecond-precision from the write time

Query examples (QuestDB SQL):

```sql
-- Last 100 regime states for BTCUSD
SELECT * FROM regime_states WHERE symbol = 'BTCUSD' ORDER BY timestamp DESC LIMIT 100;

-- Transition frequency over the last 24 hours
SELECT symbol, count(*) FROM regime_states
WHERE is_transition = true AND timestamp > dateadd('d', -1, now())
GROUP BY symbol;

-- Average confidence by regime type
SELECT hypothalamus, avg(confidence), count(*)
FROM regime_states WHERE timestamp > dateadd('h', -6, now())
GROUP BY hypothalamus ORDER BY count DESC;
```

#### Hot-Reload (regime.toml)

When `REGIME_TOML_PATH` is set, the forward service spawns a background file watcher that checks the file's last-modified time every `REGIME_RELOAD_INTERVAL_SECS` seconds (default: 30). When a change is detected, it reloads the TOML file and applies updated parameters to the running `RegimeManager`:

**Hot-reloadable parameters:**
- `volume_lookback` (global default)
- `volume_lookback_overrides` (per-asset)
- `min_confidence`
- `volatile_position_factor`
- `log_regime_changes`

**Not hot-reloadable (restart required):**
- `ticks_per_candle`
- `detection_method`

Changes to non-reloadable fields are logged as warnings but do not take effect until the service is restarted.

#### gRPC Server Prometheus Metrics

When the forward service has Prometheus metrics enabled (default), the gRPC server registers additional gauges on the same `/metrics` endpoint:

| Metric | Type | Description |
|--------|------|-------------|
| `janus_grpc_bridge_active_streams` | Gauge | Number of active `StreamRegimeUpdates` subscribers |
| `janus_grpc_bridge_stream_states_delivered_total` | Counter | Total regime states delivered to stream subscribers |
| `janus_grpc_bridge_push_requests_total` | Counter | Total push RPC requests received (unary + batch) |
| `janus_grpc_bridge_push_accepted_total` | Counter | Total states accepted via push RPCs |
| `janus_grpc_bridge_push_rejected_total` | Counter | Total states rejected (validation failures) |
| `janus_grpc_bridge_push_latency_seconds` | Histogram | Push RPC processing latency (sub-millisecond buckets) |
| `janus_grpc_bridge_get_current_queries_total` | Counter | Total `GetCurrentRegime` queries |
| `janus_grpc_bridge_config_reloads_total` | Counter | Total regime config hot-reloads |


When `REGIME_GRPC_PORT` is set, the forward service starts a gRPC server
implementing `RegimeBridgeService`. This allows neuromorphic consumers to
subscribe to regime updates **directly** via `StreamRegimeUpdates` — no
Redis hop required. The server also exposes `GetCurrentRegime` for
point-in-time queries and `PushRegimeState` / `PushRegimeStateBatch` for
accepting external enrichment.

| RPC | Direction | Description |
|-----|-----------|-------------|
| `StreamRegimeUpdates` | server → client | Server-side stream wired into the event loop's broadcast channel. Supports symbol filters, `transitions_only` mode, and `min_confidence` threshold. |
| `GetCurrentRegime` | unary | Returns the latest regime state per symbol from an in-memory snapshot map. |
| `PushRegimeState` | client → server | Accepts a single pushed regime state, stores it, and re-broadcasts it on the channel. |
| `PushRegimeStateBatch` | client → server | Batch variant of `PushRegimeState`. |

#### Example: connect with `grpcurl`

```bash
# Stream all regime updates
grpcurl -plaintext -d '{"client_id":"monitor-1"}' \
  localhost:50052 janus.v1.bridge.RegimeBridgeService/StreamRegimeUpdates

# Stream only BTCUSD transitions with ≥70% confidence
grpcurl -plaintext -d '{"symbols":["BTCUSD"],"transitions_only":true,"min_confidence":0.7,"client_id":"dash-1"}' \
  localhost:50052 janus.v1.bridge.RegimeBridgeService/StreamRegimeUpdates

# Point-in-time query
grpcurl -plaintext -d '{"symbols":["BTCUSD","ETHUSD"]}' \
  localhost:50052 janus.v1.bridge.RegimeBridgeService/GetCurrentRegime
```

#### Example: Rust client (tonic)

```rust
use janus_forward::regime_bridge_proto::regime_bridge_service_client::RegimeBridgeServiceClient;
use janus_forward::regime_bridge_proto::StreamRegimeUpdatesRequest;

let mut client = RegimeBridgeServiceClient::connect("http://127.0.0.1:50052").await?;

let request = tonic::Request::new(StreamRegimeUpdatesRequest {
    symbols: vec!["BTCUSD".into()],
    transitions_only: false,
    min_confidence: 0.0,
    client_id: "my-consumer".into(),
});

let mut stream = client.stream_regime_updates(request).await?.into_inner();

while let Some(state) = stream.message().await? {
    println!("regime: {} hypo={} amyg={} scale={:.0}%",
        state.symbol,
        state.hypothalamus_regime,
        state.amygdala_regime,
        state.position_scale * 100.0,
    );
}
```

#### Standalone neuromorphic server

To embed the server in a neuromorphic service (receiving pushes from
`regime-bridge-consumer`):

```rust
use janus_forward::regime_bridge_server::RegimeBridgeServer;
use janus_forward::regime_bridge_proto::regime_bridge_service_server::RegimeBridgeServiceServer;

// Create a standalone server with its own broadcast channel
let (server, mut rx) = RegimeBridgeServer::standalone(64);

// Spawn a task to process received states
tokio::spawn(async move {
    while let Ok(state) = rx.recv().await {
        // Feed into hypothalamus / amygdala
        println!("Received: {}", state);
    }
});

// Start gRPC server on port 50051
tonic::transport::Server::builder()
    .add_service(RegimeBridgeServiceServer::new(server))
    .serve("0.0.0.0:50051".parse()?)
    .await?;
```

### regime.toml — Volume Lookback

The `volume_lookback` setting in the `[manager]` section controls the rolling
window size (in candles) used for relative volume calculation:

```toml
[manager]
volume_lookback = 20   # default; set to 10 for fast, 50 for smooth
```

---

## Building

### Prerequisites

- Rust toolchain (stable, 1.75+)
- Protocol Buffers compiler (`protoc`)
- Redis 7+ (for cross-process bridge)
- QuestDB 7+ (for tick/trade persistence)

### Build with TLS support (optional)

To enable mTLS for the gRPC server, add the `tls` feature:

```bash
cargo build --release -p janus-forward --features tls
```

### Build release binaries

```bash
cd fks/src/janus/services/forward

# Build both binaries in release mode
cargo build --release --bin janus-forward-production --bin regime-bridge-consumer

# Binaries are at:
#   target/release/janus-forward-production
#   target/release/regime-bridge-consumer
```

### Build without Redis feature (logging-only bridge)

```bash
cargo build --release --no-default-features --bin janus-forward-production
```

---

## Systemd Deployment

### 1. Create a service user

```bash
sudo useradd --system --shell /usr/sbin/nologin --home-dir /opt/janus janus
sudo mkdir -p /opt/janus/{bin,config,logs}
sudo chown -R janus:janus /opt/janus
```

### 2. Install binaries and config

```bash
sudo cp target/release/janus-forward-production /opt/janus/bin/
sudo cp target/release/regime-bridge-consumer    /opt/janus/bin/
sudo cp config/regime.toml                       /opt/janus/config/regime.toml
```

### 3. Create environment file

```bash
sudo tee /opt/janus/config/forward.env << 'EOF'
# ── Trading ──────────────────────────────────
SYMBOL=BTCUSD
BYBIT_TESTNET=true
BYBIT_API_KEY=your-api-key-here
BYBIT_API_SECRET=your-api-secret-here
TRADING_ENABLED=false
ACCOUNT_SIZE=10000.0
CHALLENGE_TYPE=OneStep
MAX_RISK_PER_TRADE=0.01
SESSION_START_HOURS_UTC=0,8,13

# ── Infrastructure ──────────────────────────
QUESTDB_HOST=127.0.0.1
QUESTDB_PORT=9009
REDIS_URL=redis://127.0.0.1:6379
REGIME_BRIDGE_STREAM=janus:regime:bridge
METRICS_PORT=9090

# ── Regime Detection ────────────────────────
REGIME_TOML_PATH=/opt/janus/config/regime.toml

# ── Logging ─────────────────────────────────
RUST_LOG=info,janus_forward=debug
EOF

sudo chmod 600 /opt/janus/config/forward.env
sudo chown janus:janus /opt/janus/config/forward.env
```

### 4. Forward Service unit file

```ini
# /etc/systemd/system/janus-forward.service
[Unit]
Description=JANUS Forward Trading Service
After=network-online.target redis.service questdb.service
Wants=network-online.target
Requires=redis.service

[Service]
Type=simple
User=janus
Group=janus
WorkingDirectory=/opt/janus
EnvironmentFile=/opt/janus/config/forward.env
ExecStart=/opt/janus/bin/janus-forward-production
Restart=on-failure
RestartSec=5
StartLimitBurst=5
StartLimitIntervalSec=60

# Security hardening
NoNewPrivileges=yes
ProtectSystem=strict
ProtectHome=yes
ReadWritePaths=/opt/janus/logs
PrivateTmp=yes

# Resource limits
LimitNOFILE=65536
MemoryMax=2G

# Logging
StandardOutput=journal
StandardError=journal
SyslogIdentifier=janus-forward

[Install]
WantedBy=multi-user.target
```

### 5. Regime Bridge Consumer unit file

```ini
# /etc/systemd/system/janus-bridge-consumer.service
[Unit]
Description=JANUS Regime Bridge Redis Consumer
After=network-online.target redis.service janus-forward.service
Wants=janus-forward.service
Requires=redis.service

[Service]
Type=simple
User=janus
Group=janus
WorkingDirectory=/opt/janus

Environment=REDIS_URL=redis://127.0.0.1:6379
Environment=REGIME_BRIDGE_STREAM=janus:regime:bridge
Environment=CONSUMER_GROUP=neuromorphic
Environment=CONSUMER_NAME=hypothalamus-1
Environment=BLOCK_MS=5000
Environment=BATCH_SIZE=10
Environment=RUST_LOG=info,regime_bridge_consumer=debug

# Uncomment to enable gRPC forwarding:
# Environment=GRPC_TARGET=http://127.0.0.1:50051
# Environment=GRPC_SOURCE_ID=bridge-consumer-prod

ExecStart=/opt/janus/bin/regime-bridge-consumer
Restart=on-failure
RestartSec=3

# Security
NoNewPrivileges=yes
ProtectSystem=strict
ProtectHome=yes
PrivateTmp=yes

StandardOutput=journal
StandardError=journal
SyslogIdentifier=janus-bridge-consumer

[Install]
WantedBy=multi-user.target
```

### 6. Enable and start

```bash
sudo systemctl daemon-reload
sudo systemctl enable --now janus-forward.service
sudo systemctl enable --now janus-bridge-consumer.service

# Check status
sudo systemctl status janus-forward.service
sudo systemctl status janus-bridge-consumer.service

# View logs
sudo journalctl -u janus-forward -f
sudo journalctl -u janus-bridge-consumer -f
```

---

## Docker / Container Deployment

### Dockerfile

```dockerfile
# ── Build stage ──────────────────────────────
FROM rust:1.82-bookworm AS builder

RUN apt-get update && apt-get install -y protobuf-compiler && rm -rf /var/lib/apt/lists/*

WORKDIR /build
COPY . .

RUN cargo build --release \
    --bin janus-forward-production \
    --bin regime-bridge-consumer

# ── Runtime stage ────────────────────────────
FROM debian:bookworm-slim

RUN apt-get update && \
    apt-get install -y --no-install-recommends ca-certificates && \
    rm -rf /var/lib/apt/lists/*

COPY --from=builder /build/target/release/janus-forward-production /usr/local/bin/
COPY --from=builder /build/target/release/regime-bridge-consumer   /usr/local/bin/
COPY src/janus/config/regime.toml /etc/janus/regime.toml

ENV REGIME_TOML_PATH=/etc/janus/regime.toml
ENV RUST_LOG=info,janus_forward=debug

EXPOSE 9090

ENTRYPOINT ["janus-forward-production"]
```

### Docker Compose

```yaml
# docker-compose.yml
version: "3.9"

services:
  # ── Redis ──────────────────────────────────
  redis:
    image: redis:7-alpine
    ports:
      - "6379:6379"
    volumes:
      - redis-data:/data
    command: redis-server --appendonly yes
    healthcheck:
      test: ["CMD", "redis-cli", "ping"]
      interval: 10s
      retries: 3

  # ── QuestDB ────────────────────────────────
  questdb:
    image: questdb/questdb:7.4.0
    ports:
      - "9000:9000"   # web console
      - "9009:9009"   # ILP
      - "8812:8812"   # postgres wire
    volumes:
      - questdb-data:/var/lib/questdb
    healthcheck:
      test: ["CMD", "wget", "-q", "--spider", "http://localhost:9000"]
      interval: 15s
      retries: 3

  # ── Forward Service ────────────────────────
  forward:
    build:
      context: ../../../..          # repo root (fks/)
      dockerfile: src/janus/services/forward/Dockerfile
    depends_on:
      redis:
        condition: service_healthy
      questdb:
        condition: service_healthy
    environment:
      SYMBOL: BTCUSD
      BYBIT_TESTNET: "true"
      BYBIT_API_KEY: ${BYBIT_API_KEY}
      BYBIT_API_SECRET: ${BYBIT_API_SECRET}
      TRADING_ENABLED: "false"
      ACCOUNT_SIZE: "10000.0"
      CHALLENGE_TYPE: OneStep
      MAX_RISK_PER_TRADE: "0.01"
      SESSION_START_HOURS_UTC: "0,8,13"
      QUESTDB_HOST: questdb
      QUESTDB_PORT: "9009"
      REDIS_URL: redis://redis:6379
      REGIME_BRIDGE_STREAM: "janus:regime:bridge"
      REGIME_TOML_PATH: /etc/janus/regime.toml
      REGIME_GRPC_PORT: "50052"
      METRICS_PORT: "9090"
      RUST_LOG: "info,janus_forward=debug"
    ports:
      - "9090:9090"
      - "50052:50052"
    restart: unless-stopped

  # ── Bridge Consumer ────────────────────────
  bridge-consumer:
    build:
      context: ../../../..
      dockerfile: src/janus/services/forward/Dockerfile
    entrypoint: ["regime-bridge-consumer"]
    depends_on:
      redis:
        condition: service_healthy
      forward:
        condition: service_started
    environment:
      REDIS_URL: redis://redis:6379
      REGIME_BRIDGE_STREAM: "janus:regime:bridge"
      CONSUMER_GROUP: neuromorphic
      CONSUMER_NAME: hypothalamus-1
      BLOCK_MS: "5000"
      BATCH_SIZE: "10"
      RUST_LOG: "info,regime_bridge_consumer=debug"
      # Uncomment to forward to a gRPC neuromorphic service:
      # GRPC_TARGET: "http://neuromorphic-svc:50051"
    restart: unless-stopped

  # ── Prometheus ─────────────────────────────
  prometheus:
    image: prom/prometheus:v2.51.0
    ports:
      - "9091:9090"
    volumes:
      - ./prometheus.yml:/etc/prometheus/prometheus.yml:ro
      - ./grafana/alerts/brain-alerts.yml:/etc/prometheus/rules/brain-alerts.yml:ro
      - prometheus-data:/prometheus
    command:
      - "--config.file=/etc/prometheus/prometheus.yml"
      - "--storage.tsdb.retention.time=30d"

  alertmanager:
    image: prom/alertmanager:v0.27.0
    ports:
      - "9093:9093"
    volumes:
      - ./grafana/alerts/alertmanager.yml:/etc/alertmanager/alertmanager.yml:ro
    command:
      - "--config.file=/etc/alertmanager/alertmanager.yml"
      - "--storage.path=/alertmanager"
    restart: unless-stopped

  # ── Grafana ────────────────────────────────
  # Auto-provisions the JANUS Regime Monitor + Brain Pipeline dashboards on startup.
  # Dashboard JSON: grafana/janus-regime-dashboard.json
  #                 grafana/brain-dashboard.json
  # Provisioning:   grafana/provisioning-dashboards.yml
  grafana:
    image: grafana/grafana:10.4.0
    ports:
      - "3000:3000"
    volumes:
      - ./grafana/provisioning-dashboards.yml:/etc/grafana/provisioning/dashboards/janus.yml:ro
      - ./grafana:/var/lib/grafana/dashboards/janus:ro
      - grafana-data:/var/lib/grafana
    environment:
      GF_SECURITY_ADMIN_PASSWORD: ${GRAFANA_PASSWORD:-admin}
      GF_UNIFIED_ALERTING_ENABLED: "true"

volumes:
  redis-data:
  questdb-data:
  prometheus-data:
  grafana-data:
  alertmanager-data:
```

### Prometheus scrape config

```yaml
# prometheus.yml
global:
  scrape_interval: 15s
  evaluation_interval: 15s

# Alertmanager endpoint for firing alerts
alerting:
  alertmanagers:
    - static_configs:
        - targets: ["alertmanager:9093"]

# Alert rules for brain pipeline, regime detection, etc.
rule_files:
  - /etc/prometheus/rules/brain-alerts.yml

scrape_configs:
  - job_name: "janus-forward"
    static_configs:
      - targets: ["forward:9090"]
    metrics_path: /metrics
    scrape_interval: 5s
```

> **Quick-start:** Copy `grafana/alerts/alertmanager.yml` and replace the
> Slack webhook URLs / PagerDuty routing keys with your own values before
> deploying. The alert rules in `grafana/alerts/brain-alerts.yml` are
> loaded automatically by Prometheus via the `rule_files` directive above.

---

## Prometheus, Grafana & Alerting

### Key Metrics

#### Regime Detection

| Metric | Type | Labels | Description |
|--------|------|--------|-------------|
| `janus_regime_current` | IntGaugeVec | `asset` | Current regime (int-encoded) |
| `janus_regime_confidence` | GaugeVec | `asset` | Detection confidence (0–1) |
| `janus_regime_transitions_total` | IntCounterVec | `asset`, `from`, `to` | Total regime transitions |
| `janus_regime_candles_processed_total` | IntCounterVec | `asset` | Candles processed |
| `janus_regime_detector_ready` | IntGaugeVec | `asset` | 1 if detector is warmed up |

#### Regime Bridge

| Metric | Type | Labels | Description |
|--------|------|--------|-------------|
| `janus_regime_relative_volume` | GaugeVec | `asset` | Current relative volume |
| `janus_bridge_hypothalamus_regime` | IntGaugeVec | `asset` | Hypothalamus regime (int) |
| `janus_bridge_amygdala_regime` | IntGaugeVec | `asset` | Amygdala regime (int) |
| `janus_bridge_position_scale` | GaugeVec | `asset` | Recommended position scale |

#### Strategy

| Metric | Type | Labels | Description |
|--------|------|--------|-------------|
| `janus_regime_strategy_current` | IntGaugeVec | `asset` | Active strategy (int) |
| `janus_regime_position_factor` | GaugeVec | `asset` | Current position factor |
| `janus_regime_methods_agree` | IntGaugeVec | `asset` | 1 if indicator+HMM agree |
| `janus_regime_expected_duration` | GaugeVec | `asset` | Expected regime duration |

### Grafana Dashboard JSON Snippets

#### Regime State Panel (Stat)

```json
{
  "title": "Current Regime",
  "type": "stat",
  "targets": [
    {
      "expr": "janus_bridge_hypothalamus_regime{asset=\"BTCUSD\"}",
      "legendFormat": "Hypothalamus"
    }
  ],
  "fieldConfig": {
    "defaults": {
      "mappings": [
        { "type": "value", "options": { "1": { "text": "StrongBullish", "color": "dark-green" } } },
        { "type": "value", "options": { "2": { "text": "Bullish", "color": "green" } } },
        { "type": "value", "options": { "3": { "text": "Neutral", "color": "yellow" } } },
        { "type": "value", "options": { "4": { "text": "Bearish", "color": "orange" } } },
        { "type": "value", "options": { "5": { "text": "StrongBearish", "color": "red" } } },
        { "type": "value", "options": { "6": { "text": "HighVol", "color": "dark-red" } } },
        { "type": "value", "options": { "7": { "text": "LowVol", "color": "blue" } } },
        { "type": "value", "options": { "8": { "text": "Transitional", "color": "purple" } } },
        { "type": "value", "options": { "9": { "text": "Crisis", "color": "dark-red" } } },
        { "type": "value", "options": { "10": { "text": "Unknown", "color": "gray" } } }
      ]
    }
  }
}
```

#### Relative Volume Time Series

```json
{
  "title": "Relative Volume",
  "type": "timeseries",
  "targets": [
    {
      "expr": "janus_regime_relative_volume{asset=\"BTCUSD\"}",
      "legendFormat": "{{asset}}"
    }
  ],
  "fieldConfig": {
    "defaults": {
      "custom": {
        "thresholdsStyle": { "mode": "line" }
      },
      "thresholds": {
        "steps": [
          { "value": 0, "color": "blue" },
          { "value": 1.5, "color": "yellow" },
          { "value": 2.5, "color": "red" }
        ]
      }
    }
  }
}
```

#### Position Scale Gauge

```json
{
  "title": "Position Scale",
  "type": "gauge",
  "targets": [
    {
      "expr": "janus_bridge_position_scale{asset=\"BTCUSD\"}",
      "legendFormat": "{{asset}}"
    }
  ],
  "fieldConfig": {
    "defaults": {
      "min": 0,
      "max": 1.5,
      "thresholds": {
        "steps": [
          { "value": 0, "color": "red" },
          { "value": 0.5, "color": "yellow" },
          { "value": 0.8, "color": "green" },
          { "value": 1.2, "color": "dark-green" }
        ]
      },
      "unit": "percentunit"
    }
  }
}
```

#### Regime Transitions Rate

```json
{
  "title": "Regime Transitions / 5m",
  "type": "timeseries",
  "targets": [
    {
      "expr": "rate(janus_regime_transitions_total{asset=\"BTCUSD\"}[5m])",
      "legendFormat": "{{from}} → {{to}}"
    }
  ]
}
```

#### Confidence + Methods Agreement Panel

```json
{
  "title": "Detection Confidence & Agreement",
  "type": "timeseries",
  "targets": [
    {
      "expr": "janus_regime_confidence{asset=\"BTCUSD\"}",
      "legendFormat": "Confidence"
    },
    {
      "expr": "janus_regime_methods_agree{asset=\"BTCUSD\"}",
      "legendFormat": "Methods Agree"
    }
  ],
  "fieldConfig": {
    "defaults": {
      "min": 0,
      "max": 1
    }
  }
}
```

### Useful PromQL Queries

```promql
# Regime transition rate (per minute)
rate(janus_regime_transitions_total[5m]) * 60

# Average confidence over last hour
avg_over_time(janus_regime_confidence{asset="BTCUSD"}[1h])

# Volume spike detection (>2x average)
janus_regime_relative_volume > 2.0

# Position scale reduction (regime risk active)
janus_bridge_position_scale{asset="BTCUSD"} < 0.5

# Detector warmup status
janus_regime_detector_ready == 0

# Candle processing rate (per second)
rate(janus_regime_candles_processed_total[1m])

# High-risk regime active
janus_bridge_amygdala_regime{asset="BTCUSD"} >= 3
```

### Alert Rules (Regime Detection — Prometheus)

> **Brain pipeline alerts** are maintained separately in
> `grafana/alerts/brain-alerts.yml`. See the dedicated file for kill-switch,
> block-rate, watchdog, latency, reduce-only, and scale-distribution alerts.
> Import it via the `rule_files` directive shown in the Prometheus config above.


```yaml
groups:
  - name: janus-regime
    rules:
      - alert: RegimeDetectorNotReady
        expr: janus_regime_detector_ready == 0
        for: 10m
        labels:
          severity: warning
        annotations:
          summary: "Regime detector not ready for {{ $labels.asset }}"

      - alert: CrisisRegimeDetected
        expr: janus_bridge_hypothalamus_regime == 9
        for: 1m
        labels:
          severity: critical
        annotations:
          summary: "Crisis regime detected for {{ $labels.asset }}"

      - alert: ExtremeVolumeSpike
        expr: janus_regime_relative_volume > 5.0
        for: 30s
        labels:
          severity: warning
        annotations:
          summary: "Extreme volume spike ({{ $value }}x) for {{ $labels.asset }}"

      - alert: NoRegimeTransitions
        expr: rate(janus_regime_transitions_total[1h]) == 0
        for: 2h
        labels:
          severity: info
        annotations:
          summary: "No regime transitions in 2h for {{ $labels.asset }} — may indicate stale data"

      - alert: LowDetectionConfidence
        expr: janus_regime_confidence < 0.3
        for: 15m
        labels:
          severity: warning
        annotations:
          summary: "Low regime detection confidence ({{ $value }}) for {{ $labels.asset }}"
```

---

## Operational Runbook

### Initial Deployment Checklist (24-Hour Validation)

1. **Start infrastructure first:**
   ```bash
   docker compose up -d redis questdb prometheus grafana
   ```

2. **Verify Redis is healthy:**
   ```bash
   redis-cli ping  # → PONG
   ```

3. **Set API credentials** (never commit to source control):
   ```bash
   export BYBIT_API_KEY="..."
   export BYBIT_API_SECRET="..."
   ```

4. **Start forward service with trading DISABLED:**
   ```bash
   TRADING_ENABLED=false docker compose up -d forward
   ```

5. **Verify metrics are being scraped:**
   ```bash
   curl -s http://localhost:9090/metrics | grep janus_regime
   ```

6. **Start bridge consumer:**
   ```bash
   docker compose up -d bridge-consumer
   ```

7. **Verify gRPC auth (if enabled):**
   ```bash
   # Should succeed (200 / stream opens):
   grpcurl -plaintext \
     -H 'authorization: Bearer <your-token>' \
     -d '{"symbol":"BTCUSD"}' \
     localhost:50052 janus.v1.bridge.RegimeBridgeService/GetCurrentRegime

   # Should fail with UNAUTHENTICATED:
   grpcurl -plaintext \
     -d '{"symbol":"BTCUSD"}' \
     localhost:50052 janus.v1.bridge.RegimeBridgeService/GetCurrentRegime

   # If multi-token, verify each token works:
   grpcurl -plaintext \
     -H 'authorization: Bearer <token-2>' \
     -d '{"symbol":"BTCUSD"}' \
     localhost:50052 janus.v1.bridge.RegimeBridgeService/GetCurrentRegime
   ```

8. **Verify PEL reclamation is active:**
   ```bash
   # Check consumer pending entries (should be 0 or near-0 under normal operation)
   redis-cli XPENDING janus:regime:bridge neuromorphic

   # Check per-consumer pending breakdown
   redis-cli XPENDING janus:regime:bridge neuromorphic - + 10

   # Look for PEL drain/sweep log lines in consumer output
   docker compose logs bridge-consumer | grep -E "PEL|Drained|reclaimed"
   ```

9. **Verify QuestDB archiving:**
   ```bash
   # Check regime_states table has rows
   curl -G 'http://localhost:9000/exec' \
     --data-urlencode "query=SELECT count() FROM regime_states"

   # Check recent entries
   curl -G 'http://localhost:9000/exec' \
     --data-urlencode "query=SELECT * FROM regime_states ORDER BY timestamp DESC LIMIT 5"
   ```

10. **Monitor for 24 hours** before enabling trading.

### Monitoring Health

```bash
# Forward service logs
docker compose logs -f forward

# Bridge consumer logs
docker compose logs -f bridge-consumer

# Redis stream info
redis-cli XINFO STREAM janus:regime:bridge
redis-cli XINFO GROUPS janus:regime:bridge

# Redis stream length (should stay bounded ≤ ~10000 via MAXLEN)
redis-cli XLEN janus:regime:bridge

# PEL health — total pending entries across all consumers
redis-cli XPENDING janus:regime:bridge neuromorphic

# PEL per-consumer breakdown (detect stuck/dead consumers)
redis-cli XPENDING janus:regime:bridge neuromorphic - + 20

# Prometheus targets
curl http://localhost:9091/api/v1/targets

# gRPC bridge metrics (if Prometheus metrics enabled)
curl -s http://localhost:9090/metrics | grep janus_grpc_bridge

# Auth rejection rate (should be 0 in normal operation)
curl -s http://localhost:9090/metrics | grep push_rejected_total
```

### 24-Hour Validation Checklist

Before enabling live trading, verify all of the following during the 24-hour dry run:

- [ ] Forward service running without crashes or panics
- [ ] Regime detection producing transitions with reasonable confidence (check `janus_regime_confidence`)
- [ ] Prometheus metrics appearing at `/metrics` (including `janus_grpc_bridge_*` metrics)
- [ ] Grafana dashboard panels populating: relative volume, bridged regimes, position scale
- [ ] gRPC auth: valid tokens accepted, invalid/missing tokens rejected (check `push_rejected_total == 0` in normal operation)
- [ ] QuestDB `regime_states` table accumulating rows with correct data
- [ ] Redis stream length bounded (≤ ~10,000 entries via MAXLEN)
- [ ] PEL (pending entries) near zero — no stuck messages (`redis-cli XPENDING`)
- [ ] Hot-reload: modify `regime.toml` and confirm log shows "Regime config reloaded" within poll interval
- [ ] Bridge consumer startup drain: restart consumer and confirm "Drained N pending entries" or "No pending entries — clean startup" in logs
- [ ] No sustained `push_rejected_total` spikes or `active_streams` drops
- [ ] Memory and CPU stable over 24 hours (no leaks)

### Enabling Live Trading

> **⚠️ WARNING:** Only enable after completing the 24-hour validation checklist above.

1. Verify regime detection is stable and confidence is consistent.
2. Verify position sizing and risk parameters are correct.
3. Verify PEL is clean (`redis-cli XPENDING janus:regime:bridge neuromorphic` shows 0 pending).
4. Set `TRADING_ENABLED=true` and restart:
   ```bash
   docker compose up -d forward
   ```

### Troubleshooting

#### PEL entries accumulating (pending count growing)

If `redis-cli XPENDING janus:regime:bridge neuromorphic` shows a growing count:

1. **Consumer alive but slow**: Increase `BATCH_SIZE` or reduce processing latency.
2. **Consumer crashed**: Restart it — the startup drain will recover pending entries. Other running consumers will also reclaim stale entries via periodic PEL sweep.
3. **Dead consumer name lingering**: If a consumer name is permanently gone (e.g. scaled down), another consumer's periodic sweep will reclaim its entries after `PEL_MIN_IDLE_MS`. To reclaim immediately:
   ```bash
   # Manual reclaim: transfer all entries from dead consumer to a live one
   redis-cli XCLAIM janus:regime:bridge neuromorphic <live-consumer-name> 0 <entry-id-1> <entry-id-2> ...
   ```

#### gRPC auth rejections in production

If `push_rejected_total` is non-zero:

1. Check client is sending `authorization: Bearer <token>` header (case-insensitive `bearer` prefix is accepted).
2. Verify the token matches one of the configured tokens in `REGIME_GRPC_AUTH_TOKEN` (comma-separated).
3. If rotating tokens: ensure both old and new tokens are in the comma-separated list during the overlap window.
4. Check for clock skew or encoding issues in the token value (no trailing whitespace, etc.).


| Symptom | Possible Cause | Resolution |
|---------|---------------|------------|
| No regime transitions | Detector not warmed up | Wait for sufficient candles (depends on `ticks_per_candle` and HMM `min_observations`) |
| `janus_regime_detector_ready == 0` | Insufficient data | Feed more ticks; check WebSocket connection to Bybit |
| Bridge consumer not receiving messages | Redis stream empty | Verify `REDIS_URL` is set on the forward service; check `XLEN` |
| Consumer group BUSYGROUP error | Group already exists | This is normal — the consumer reuses existing groups |
| gRPC push failures | Target unreachable | Check `GRPC_TARGET` URL; verify the neuromorphic service is running |
| High `relative_volume` spikes | Market event | Expected behavior — regime bridge will flag as high risk |
| Metrics not appearing in Prometheus | Scrape config wrong | Verify `prometheus.yml` targets match the forward service's `METRICS_PORT` |

### Tuning Guide

| Parameter | Effect of Increasing | Effect of Decreasing |
|-----------|---------------------|---------------------|
| `ticks_per_candle` | Smoother regime detection, slower reaction | Faster reaction, more noise |
| `min_confidence` | Fewer trades, higher conviction | More trades, more noise |
| `volatile_position_factor` | Larger positions in volatile regimes | Smaller positions (more conservative) |
| `volume_lookback` | Smoother volume baseline | Faster reaction to volume changes |
| `adx_trending_threshold` | Fewer trending classifications | More trending classifications |
| `regime_stability_bars` | Fewer regime transitions (anti-whipsaw) | More responsive but whippier |
| `agreement_confidence_boost` | Higher confidence when methods agree | Less reward for agreement |

### Backup and Recovery

```bash
# Backup regime config
cp /opt/janus/config/regime.toml /opt/janus/config/regime.toml.bak

# Export Redis stream to file (for replay)
redis-cli --rdb /tmp/redis-backup.rdb BGSAVE

# Restore from config backup
cp /opt/janus/config/regime.toml.bak /opt/janus/config/regime.toml
sudo systemctl restart janus-forward
```

### Scaling Multiple Symbols

To run regime detection for multiple symbols, launch separate forward service
instances per symbol (each with its own `SYMBOL` env var). The bridge consumer
can read from a shared Redis stream — all symbols' states flow through the
same `janus:regime:bridge` stream, differentiated by the `symbol` field.

```bash
# Instance 1
SYMBOL=BTCUSD METRICS_PORT=9090 janus-forward-production &

# Instance 2
SYMBOL=ETHUSD METRICS_PORT=9091 janus-forward-production &

# Single consumer reads both
regime-bridge-consumer
```
