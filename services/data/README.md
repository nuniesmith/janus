# FKS Data Service

**High-Performance Market Data Ingestion Service**

The FKS Data Service is a standalone Rust-based microservice designed for high-throughput ingestion of cryptocurrency market data from multiple exchanges and alternative data sources. This service was extracted from the JANUS trading system to operate independently.

---

## 🎯 Overview

The Data Service implements an **Actor Model** architecture using Tokio for concurrent data processing. It handles:

- **Real-time WebSocket streams** from Binance, Bybit, and Kucoin
- **Alternative metrics polling** (Fear & Greed, ETF flows, volatility)
- **High-speed persistence** to QuestDB via Influx Line Protocol (ILP)
- **State management** using Redis for rate limiting and deduplication
- **Automatic failover** between exchanges for resilience

### Performance

- 🚀 **100K+ ticks/second** ingestion to QuestDB
- ⚡ **Sub-millisecond** WebSocket message processing
- 📦 **Batched writes** with configurable flush intervals
- 🔄 **Zero-downtime** reconnection and failover

---

## 🏗️ Architecture

```
┌─────────────────────────────────────────────────────────────┐
│                      FKS Data Service                        │
├─────────────────────────────────────────────────────────────┤
│                                                               │
│  ┌───────────────┐  ┌───────────────┐  ┌──────────────┐    │
│  │   Binance WS  │  │   Bybit WS    │  │  Kucoin WS   │    │
│  │   (Primary)   │  │  (Secondary)  │  │  (Tertiary)  │    │
│  └───────┬───────┘  └───────┬───────┘  └──────┬───────┘    │
│          │                  │                   │             │
│          └──────────────────┴───────────────────┘             │
│                             │                                 │
│                      ┌──────▼──────┐                         │
│                      │    Router    │                         │
│                      │   (Actor)    │                         │
│                      └──────┬───────┘                         │
│                             │                                 │
│          ┌──────────────────┴───────────────────┐            │
│          │                                       │            │
│  ┌───────▼────────┐                    ┌────────▼────────┐  │
│  │  QuestDB ILP   │                    │  Redis Manager  │  │
│  │  (Batched)     │                    │  (State/Cache)  │  │
│  └────────────────┘                    └─────────────────┘  │
│                                                               │
│  ┌───────────────────────────────────────────────────────┐  │
│  │              Metrics Pollers (REST)                   │  │
│  ├───────────────┬───────────────┬──────────────────────┤  │
│  │ Fear & Greed  │  ETF Flows    │  Volatility (DVOL)   │  │
│  └───────────────┴───────────────┴──────────────────────┘  │
└─────────────────────────────────────────────────────────────┘
```

### Components

#### 1. **Actors** (`src/actors/`)
- **Router**: Central message dispatcher
- **WebSocketActor**: Manages persistent WebSocket connections
- **PollerActor**: Handles REST API polling with rate limiting

#### 2. **Connectors** (`src/connectors/`)
- **Binance**: Trade stream (`<symbol>@trade`)
- **Bybit**: V5 API (`publicTrade.<symbol>`)
- **Kucoin**: Token-based WebSocket with ping/pong

#### 3. **Metrics** (`src/metrics/`)
- **Fear & Greed Index**: Daily sentiment indicator (Alternative.me)
- **ETF Net Flows**: Institutional demand (Farside Investors)
- **DVOL Index**: Implied volatility (Deribit)

#### 4. **Storage** (`src/storage/`)
- **IlpWriter**: Batched writes to QuestDB using ILP
- **RedisManager**: State management, rate limiting, deduplication

---

## 📊 Data Schema

### Tables in QuestDB

#### `trades_crypto`
```sql
CREATE TABLE trades_crypto (
    ts TIMESTAMP,
    symbol SYMBOL,
    exchange SYMBOL,
    side SYMBOL,
    price DOUBLE,
    amount DOUBLE,
    trade_id STRING,
    latency_ms LONG
) TIMESTAMP(ts) PARTITION BY DAY;
```

#### `candles_crypto`
```sql
CREATE TABLE candles_crypto (
    ts TIMESTAMP,
    symbol SYMBOL,
    exchange SYMBOL,
    interval SYMBOL,
    open DOUBLE,
    high DOUBLE,
    low DOUBLE,
    close DOUBLE,
    volume DOUBLE
) TIMESTAMP(ts) PARTITION BY DAY;
```

#### `market_metrics`
```sql
CREATE TABLE market_metrics (
    ts TIMESTAMP,
    metric_type SYMBOL,
    asset SYMBOL,
    source SYMBOL,
    value DOUBLE,
    meta STRING
) TIMESTAMP(ts) PARTITION BY MONTH;
```

---

## 🚀 Configuration

### Environment Variables

```bash
# Assets to track
ASSETS=BTC,ETH,SOL

# Exchange Configuration
PRIMARY_EXCHANGE=binance
SECONDARY_EXCHANGE=bybit
TERTIARY_EXCHANGE=kucoin

BINANCE_WS_URL=wss://stream.binance.com:9443/ws
BYBIT_WS_URL=wss://stream.bybit.com/v5/public/spot
KUCOIN_REST_URL=https://api.kucoin.com

# QuestDB Configuration
QUESTDB_HOST=questdb
QUESTDB_ILP_PORT=9009
QUESTDB_HTTP_PORT=9000
QUESTDB_BUFFER_SIZE=1000
QUESTDB_FLUSH_INTERVAL_MS=100

# Redis Configuration
REDIS_URL=redis://redis:6379
REDIS_KEY_PREFIX=data_factory
REDIS_POOL_SIZE=10

# Metrics Configuration
ENABLE_FEAR_GREED=true
ENABLE_ETF_FLOWS=true
ENABLE_VOLATILITY=true
ENABLE_ALTCOIN_SEASON=false

FEAR_GREED_URL=https://api.alternative.me/fng/
ETF_FLOWS_URL=https://farside.co.uk/btc/
DVOL_URL=https://www.deribit.com/api/v2/public/get_volatility_index_data
METRICS_POLL_INTERVAL_SECS=300

# Operational Settings
ENABLE_BACKFILL=true
MAX_BACKFILL_HOURS=24
ENABLE_FAILOVER=true
FAILOVER_LATENCY_THRESHOLD_MS=500
FAILOVER_ERROR_COUNT=10
HEALTH_CHECK_INTERVAL_SECS=30

# Logging
RUST_LOG=info,janus_data_factory=debug
```

---

## 🐳 Docker Deployment

### Build

```bash
docker build -f docker/Dockerfile.rust --build-arg SERVICE_PATH=src/janus/services/data-factory -t janus-data-factory:latest .
```

### Run

```bash
docker-compose up data-factory
```

### docker-compose.yml Entry

```yaml
data-factory:
  build:
    context: .
    dockerfile: docker/Dockerfile.rust
    args:
      SERVICE_PATH: src/janus/services/data-factory
  image: janus-data-factory:latest
  container_name: janus_data_factory
  environment:
    - ASSETS=BTC,ETH,SOL
    - QUESTDB_HOST=questdb
    - REDIS_URL=redis://redis:6379
  depends_on:
    - questdb
    - redis
  restart: unless-stopped
```

---

## 🔧 Development

### Build

```bash
cd src/janus/services/data-factory
cargo build --release
```

### Run

```bash
cargo run --release
```

### Test

```bash
cargo test
```

### Lint

```bash
cargo clippy -- -D warnings
```

---

## 📈 Monitoring

### Redis Keys

The Data Factory uses the following Redis key patterns:

- `fks_ruby:health:{component}` - Component health status
- `fks_ruby:ratelimit:{service}:{window}` - Rate limit counters
- `fks_ruby:metric_hash:{type}:{asset}` - Metric deduplication hashes
- `fks_ruby:heartbeat:{component}` - Component heartbeat timestamps
- `fks_ruby:gaps:{symbol}` - Detected data gaps for backfilling
- `fks_ruby:config:active_exchange:{asset}` - Active exchange per asset

### QuestDB Queries

**Check ingestion rate:**
```sql
SELECT symbol, exchange, count(*) as trades, 
       avg(latency_ms) as avg_latency_ms
FROM trades_crypto
WHERE ts > dateadd('m', -5, now())
GROUP BY symbol, exchange
ORDER BY trades DESC;
```

**Monitor Fear & Greed Index:**
```sql
SELECT ts, value, meta
FROM market_metrics
WHERE metric_type = 'fear_greed'
ORDER BY ts DESC
LIMIT 10;
```

**Check data gaps:**
```sql
SELECT symbol, exchange, 
       sample_by(1h, max(ts)) as last_trade
FROM trades_crypto
WHERE ts > dateadd('d', -1, now())
SAMPLE BY 1h;
```

---

## 🛡️ Error Handling

### Automatic Reconnection

WebSocket connections implement exponential backoff:
- Initial delay: 5 seconds
- Maximum delay: 160 seconds (5s * 2^5)
- Maximum attempts: 10

### Failover Logic

The system automatically switches exchanges when:
1. Latency exceeds `FAILOVER_LATENCY_THRESHOLD_MS` (default: 500ms)
2. Consecutive errors exceed `FAILOVER_ERROR_COUNT` (default: 10)
3. Connection is lost for more than 30 seconds

### Data Integrity

- **Deduplication**: Metrics are hashed and checked against Redis before storage
- **Gap Detection**: Missing data is logged to Redis for backfilling
- **Idempotency**: Backfill operations check for existing data before insertion

---

## 🔍 Troubleshooting

### No data in QuestDB

1. Check ILP connection:
   ```bash
   telnet questdb 9009
   ```

2. Verify Redis connectivity:
   ```bash
   redis-cli -h redis ping
   ```

3. Check logs:
   ```bash
   docker logs janus_data_factory
   ```

### High latency

1. Check QuestDB buffer settings:
   - Increase `QUESTDB_BUFFER_SIZE` if flush errors
   - Decrease `QUESTDB_FLUSH_INTERVAL_MS` for lower latency

2. Monitor Redis memory:
   ```bash
   redis-cli -h redis info memory
   ```

### Exchange connection failures

1. Verify exchange URLs are accessible
2. Check rate limits (especially for Kucoin token endpoint)
3. Review WebSocket actor logs for specific errors

---

## 📝 Future Enhancements

- [ ] Add support for more exchanges (Coinbase, Kraken, OKX)
- [ ] Implement orderbook snapshots and deltas
- [ ] Add funding rate ingestion for perpetual futures
- [ ] Real-time realized volatility calculation
- [ ] Prometheus metrics endpoint
- [ ] Grafana dashboard templates
- [ ] Automatic gap detection and backfilling scheduler
- [ ] Multi-region deployment support
- [ ] WebSocket compression (gzip/deflate)

---

## 📚 References

- [Binance WebSocket Streams](https://binance-docs.github.io/apidocs/spot/en/#websocket-market-streams)
- [Bybit V5 WebSocket API](https://bybit-exchange.github.io/docs/v5/ws/connect)
- [Kucoin WebSocket API](https://docs.kucoin.com/#websocket-feed)
- [QuestDB ILP Documentation](https://questdb.io/docs/reference/api/ilp/overview/)
- [Alternative.me Fear & Greed API](https://alternative.me/crypto/fear-and-greed-index/)
- [Farside Investors ETF Data](https://farside.co.uk/)
- [Deribit DVOL Index](https://www.deribit.com/statistics/BTC/volatility-index)

---

## 📄 License

MIT License - see [LICENSE](../../../../LICENSE) for details

---

**Built with ❤️ by the FKS team**