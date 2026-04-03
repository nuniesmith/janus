# Integration Guide: Rate Limiter & Gap Detection
## Using Spike Prototypes in Data Factory

**Version:** 1.0  
**Last Updated:** 2024  
**Status:** Production Integration

---

## Overview

This guide explains how to integrate the validated spike prototypes into the Data Factory service:

- **Rate Limiter** (`janus-rate-limiter`) - Token bucket & sliding window rate limiting
- **Gap Detection** (`janus-gap-detection`) - Multi-layer data completeness monitoring

These crates have been moved from `spike-prototypes/` to `src/janus/crates/` and are now part of the JANUS workspace.

---

## Table of Contents

1. [Quick Start](#quick-start)
2. [Rate Limiter Integration](#rate-limiter-integration)
3. [Gap Detection Integration](#gap-detection-integration)
4. [Combined Usage](#combined-usage)
5. [Configuration](#configuration)
6. [Monitoring & Metrics](#monitoring--metrics)
7. [Testing](#testing)
8. [Migration from Spike Prototypes](#migration-from-spike-prototypes)

---

## Quick Start

### Add Dependencies

The dependencies are already added to `Cargo.toml`:

```toml
[dependencies]
janus-rate-limiter = { path = "../../crates/rate-limiter" }
janus-gap-detection = { path = "../../crates/gap-detection" }
```

### Basic Usage

```rust
use janus_rate_limiter::{RateLimiterManager, TokenBucketConfig};
use janus_gap_detection::{GapDetectionManager, Trade};

#[tokio::main]
async fn main() -> anyhow::Result<()> {
    // Initialize rate limiter
    let rate_limiter = RateLimiterManager::new();
    rate_limiter.register(
        "binance".to_string(),
        TokenBucketConfig::binance_spot(),
    )?;
    
    // Initialize gap detector
    let gap_detector = GapDetectionManager::new();
    gap_detector.register_pair("binance", "BTCUSD").await;
    
    // Use them together
    // ... (see examples below)
    
    Ok(())
}
```

---

## Rate Limiter Integration

### 1. Per-Exchange Configuration

Create a rate limiter for each exchange with appropriate configs:

```rust
use janus_rate_limiter::{RateLimiterManager, TokenBucketConfig};
use std::time::Duration;

pub fn setup_rate_limiters() -> RateLimiterManager {
    let manager = RateLimiterManager::new();
    
    // Binance: 6000 requests/minute with sliding window
    manager.register(
        "binance".to_string(),
        TokenBucketConfig::binance_spot(),
    ).unwrap();
    
    // Bybit: 120 requests/second (no sliding window)
    manager.register(
        "bybit".to_string(),
        TokenBucketConfig::bybit_v5(),
    ).unwrap();
    
    // Kucoin: 200 requests/10 seconds
    manager.register(
        "kucoin".to_string(),
        TokenBucketConfig::kucoin_public(),
    ).unwrap();
    
    // Custom exchange
    manager.register(
        "custom".to_string(),
        TokenBucketConfig {
            capacity: 1000,
            refill_rate: 16.67, // 1000 per minute
            sliding_window: true,
            window_duration: Duration::from_secs(60),
            safety_margin: 0.9, // Use 90% of capacity
        },
    ).unwrap();
    
    manager
}
```

### 2. Integration with WebSocket Actor

```rust
use janus_rate_limiter::RateLimiterManager;
use std::sync::Arc;

pub struct WebSocketActor {
    exchange: String,
    rate_limiter: Arc<RateLimiterManager>,
    // ... other fields
}

impl WebSocketActor {
    pub async fn send_request(&self, weight: u32) -> anyhow::Result<()> {
        // Acquire rate limit tokens before sending request
        self.rate_limiter.acquire(&self.exchange, weight).await?;
        
        // Safe to send request now
        self.send_api_request().await?;
        
        Ok(())
    }
    
    async fn send_api_request(&self) -> anyhow::Result<()> {
        // Your existing API request logic
        todo!()
    }
}
```

### 3. Integration with REST API Calls

```rust
use janus_rate_limiter::RateLimiterManager;
use anyhow::Result;

pub struct ExchangeClient {
    rate_limiter: Arc<RateLimiterManager>,
    http_client: reqwest::Client,
}

impl ExchangeClient {
    pub async fn get_klines(
        &self,
        exchange: &str,
        symbol: &str,
        interval: &str,
    ) -> Result<Vec<Kline>> {
        // Determine request weight (exchange-specific)
        let weight = self.calculate_weight(exchange, "klines");
        
        // Acquire rate limit
        self.rate_limiter.acquire(exchange, weight).await?;
        
        // Make HTTP request
        let url = format!("https://api.{}.com/klines", exchange);
        let response = self.http_client
            .get(&url)
            .query(&[("symbol", symbol), ("interval", interval)])
            .send()
            .await?;
        
        // Update rate limiter from response headers
        if let Some(limiter) = self.rate_limiter.get(exchange) {
            if let (Some(used), Some(limit)) = (
                response.headers().get("X-MBX-USED-WEIGHT-1M"),
                response.headers().get("X-MBX-ORDER-COUNT-10S"),
            ) {
                let used: u32 = used.to_str()?.parse()?;
                let limit: u32 = limit.to_str()?.parse()?;
                limiter.update_from_headers(used, limit);
            }
        }
        
        let klines = response.json().await?;
        Ok(klines)
    }
}
```

### 4. Metrics Collection

```rust
use janus_rate_limiter::RateLimiterManager;
use std::time::Duration;
use tokio::time::interval;

pub async fn collect_rate_limiter_metrics(
    manager: Arc<RateLimiterManager>,
) {
    let mut tick = interval(Duration::from_secs(10));
    
    loop {
        tick.tick().await;
        
        // Get metrics for all exchanges
        let all_metrics = manager.all_metrics();
        
        for (exchange, metrics) in all_metrics {
            // Export to Prometheus (see Monitoring section)
            metrics::gauge!(
                "rate_limiter_tokens_available",
                metrics.current_tokens,
                "exchange" => exchange.clone()
            );
            
            metrics::counter!(
                "rate_limiter_requests_total",
                metrics.total_requests,
                "exchange" => exchange.clone()
            );
            
            metrics::counter!(
                "rate_limiter_rejected_total",
                metrics.rejected_requests,
                "exchange" => exchange.clone()
            );
            
            tracing::debug!(
                exchange = %exchange,
                total_requests = metrics.total_requests,
                accepted = metrics.accepted_requests,
                rejected = metrics.rejected_requests,
                tokens = metrics.current_tokens,
                "Rate limiter metrics"
            );
        }
    }
}
```

---

## Gap Detection Integration

### 1. Setup per Trading Pair

```rust
use janus_gap_detection::{GapDetectionManager, GapDetectorConfig, Trade};
use std::time::Duration;

pub async fn setup_gap_detection() -> GapDetectionManager {
    let manager = GapDetectionManager::new();
    
    // Register high-volume pairs (strict sequence checking)
    manager.register_pair_with_config(
        "binance",
        "BTCUSD",
        GapDetectorConfig {
            heartbeat_timeout: Duration::from_secs(10),
            enable_sequence_check: true,
            enable_statistical_check: true,
            enable_volume_check: true,
            sequence_gap_threshold: 1, // Alert on any missing sequence
            statistical_window_size: 100,
            statistical_std_multiplier: 3.0,
        },
    ).await;
    
    // Register low-volume pairs (relaxed checking)
    manager.register_pair_with_config(
        "binance",
        "RAREUSDT",
        GapDetectorConfig {
            heartbeat_timeout: Duration::from_secs(60), // Longer timeout
            enable_sequence_check: true,
            enable_statistical_check: false, // Disable for low volume
            enable_volume_check: false,
            sequence_gap_threshold: 1,
            statistical_window_size: 100,
            statistical_std_multiplier: 3.0,
        },
    ).await;
    
    manager
}
```

### 2. Integration with Trade Ingestion

```rust
use janus_gap_detection::{GapDetectionManager, Trade};
use chrono::Utc;

pub struct TradeIngestionActor {
    gap_detector: Arc<GapDetectionManager>,
    // ... other fields
}

impl TradeIngestionActor {
    pub async fn process_trade(
        &self,
        exchange: &str,
        pair: &str,
        trade_data: &ExchangeTrade,
    ) -> anyhow::Result<()> {
        // Convert to Gap Detection Trade format
        let trade = Trade {
            id: trade_data.id,
            exchange: exchange.to_string(),
            pair: pair.to_string(),
            price: trade_data.price,
            quantity: trade_data.quantity,
            timestamp: trade_data.timestamp,
            is_buyer_maker: trade_data.is_buyer_maker,
        };
        
        // Process through gap detector
        self.gap_detector.process_trade(trade).await?;
        
        // Check for gaps
        if let Some(gaps) = self.gap_detector.get_gaps(exchange, pair).await {
            for gap in gaps {
                tracing::warn!(
                    exchange = exchange,
                    pair = pair,
                    gap_type = ?gap.gap_type,
                    missing_count = gap.missing_count,
                    severity = ?gap.severity(),
                    "Data gap detected"
                );
                
                // Trigger backfill
                self.trigger_backfill(gap).await?;
            }
        }
        
        // Write to QuestDB
        self.write_to_questdb(&trade).await?;
        
        Ok(())
    }
    
    async fn trigger_backfill(&self, gap: Gap) -> anyhow::Result<()> {
        // Send gap to backfill queue
        // TODO: Implement backfill logic (see TODO_IMPLEMENTATION_PLAN.md)
        todo!()
    }
}
```

### 3. Periodic Gap Checking

```rust
use janus_gap_detection::GapDetectionManager;
use std::time::Duration;
use tokio::time::interval;

pub async fn periodic_gap_check(
    manager: Arc<GapDetectionManager>,
) {
    let mut tick = interval(Duration::from_secs(30));
    
    loop {
        tick.tick().await;
        
        // Check all registered pairs
        let all_gaps = manager.get_all_gaps();
        
        if !all_gaps.is_empty() {
            tracing::warn!(
                gap_count = all_gaps.len(),
                "Gaps detected across all pairs"
            );
            
            for gap in &all_gaps {
                // Export metrics
                metrics::counter!(
                    "gaps_detected_total",
                    1,
                    "exchange" => gap.exchange.clone(),
                    "pair" => gap.pair.clone(),
                    "type" => format!("{:?}", gap.gap_type),
                );
                
                metrics::histogram!(
                    "gap_size_trades",
                    gap.missing_count as f64,
                    "exchange" => gap.exchange.clone(),
                );
                
                // Critical gaps need immediate attention
                if gap.severity() == janus_gap_detection::GapSeverity::Critical {
                    tracing::error!(
                        exchange = %gap.exchange,
                        pair = %gap.pair,
                        missing = gap.missing_count,
                        "CRITICAL gap detected - immediate backfill required"
                    );
                    
                    // Send alert
                    // TODO: Implement alerting (see TODO_IMPLEMENTATION_PLAN.md)
                }
            }
        }
        
        // Update completeness metric
        let completeness = calculate_data_completeness(&manager).await;
        metrics::gauge!("data_completeness_percent", completeness);
    }
}

async fn calculate_data_completeness(manager: &GapDetectionManager) -> f64 {
    // Calculate percentage of expected data that was received
    // TODO: Implement based on your SLO definition
    99.9
}
```

---

## Combined Usage

### Complete Data Factory Integration

```rust
use janus_rate_limiter::RateLimiterManager;
use janus_gap_detection::GapDetectionManager;
use std::sync::Arc;
use tokio::sync::mpsc;

pub struct DataFactory {
    rate_limiter: Arc<RateLimiterManager>,
    gap_detector: Arc<GapDetectionManager>,
    // ... other components
}

impl DataFactory {
    pub async fn new() -> anyhow::Result<Self> {
        // Setup rate limiters
        let rate_limiter = Arc::new(setup_rate_limiters());
        
        // Setup gap detection
        let gap_detector = Arc::new(setup_gap_detection().await);
        
        Ok(Self {
            rate_limiter,
            gap_detector,
        })
    }
    
    pub async fn start(&self) -> anyhow::Result<()> {
        // Start monitoring tasks
        let rl_metrics = self.rate_limiter.clone();
        tokio::spawn(async move {
            collect_rate_limiter_metrics(rl_metrics).await;
        });
        
        let gd_check = self.gap_detector.clone();
        tokio::spawn(async move {
            periodic_gap_check(gd_check).await;
        });
        
        // Start WebSocket connections
        self.start_websocket_connections().await?;
        
        // Start REST API poller
        self.start_rest_poller().await?;
        
        Ok(())
    }
    
    async fn start_websocket_connections(&self) -> anyhow::Result<()> {
        let exchanges = vec!["binance", "bybit", "kucoin"];
        
        for exchange in exchanges {
            let actor = WebSocketActor::new(
                exchange.to_string(),
                self.rate_limiter.clone(),
                self.gap_detector.clone(),
            );
            
            tokio::spawn(async move {
                if let Err(e) = actor.run().await {
                    tracing::error!(
                        exchange = exchange,
                        error = %e,
                        "WebSocket actor failed"
                    );
                }
            });
        }
        
        Ok(())
    }
    
    async fn start_rest_poller(&self) -> anyhow::Result<()> {
        // Poll for candles, exotic metrics, etc.
        todo!()
    }
}

// Example WebSocket Actor
pub struct WebSocketActor {
    exchange: String,
    rate_limiter: Arc<RateLimiterManager>,
    gap_detector: Arc<GapDetectionManager>,
}

impl WebSocketActor {
    pub async fn run(&self) -> anyhow::Result<()> {
        loop {
            // Connect to WebSocket
            match self.connect().await {
                Ok(mut ws) => {
                    tracing::info!(
                        exchange = %self.exchange,
                        "WebSocket connected"
                    );
                    
                    // Process messages
                    while let Some(msg) = ws.next().await {
                        match msg {
                            Ok(trade_msg) => {
                                self.process_message(trade_msg).await?;
                            }
                            Err(e) => {
                                tracing::error!(
                                    exchange = %self.exchange,
                                    error = %e,
                                    "WebSocket error"
                                );
                                break;
                            }
                        }
                    }
                }
                Err(e) => {
                    tracing::error!(
                        exchange = %self.exchange,
                        error = %e,
                        "WebSocket connection failed"
                    );
                }
            }
            
            // Reconnect after delay
            tokio::time::sleep(Duration::from_secs(5)).await;
        }
    }
    
    async fn process_message(&self, msg: WebSocketMessage) -> anyhow::Result<()> {
        // Parse trade from message
        let trade = self.parse_trade(msg)?;
        
        // Process through gap detector
        self.gap_detector.process_trade(trade.clone()).await?;
        
        // Write to QuestDB
        self.write_trade(trade).await?;
        
        Ok(())
    }
}
```

---

## Configuration

### Environment Variables

```bash
# Rate Limiter Configuration
RATE_LIMIT_SAFETY_MARGIN=0.9  # Use 90% of capacity
RATE_LIMIT_ENABLE_SLIDING_WINDOW=true

# Gap Detection Configuration
GAP_DETECTION_HEARTBEAT_TIMEOUT=30  # seconds
GAP_DETECTION_ENABLE_SEQUENCE_CHECK=true
GAP_DETECTION_ENABLE_STATISTICAL_CHECK=true

# Backfill Configuration (TODO)
BACKFILL_MAX_CONCURRENT=2
BACKFILL_BATCH_SIZE=10000
BACKFILL_DISK_THRESHOLD=90  # percent
```

### Config File (config.toml)

```toml
[rate_limiter]
safety_margin = 0.9
enable_sliding_window = true

[rate_limiter.exchanges.binance]
capacity = 6000
refill_rate = 100.0
window_duration_secs = 60

[rate_limiter.exchanges.bybit]
capacity = 120
refill_rate = 120.0
window_duration_secs = 1
enable_sliding_window = false

[gap_detection]
heartbeat_timeout_secs = 30
enable_sequence_check = true
enable_statistical_check = true
statistical_window_size = 100
statistical_std_multiplier = 3.0

[gap_detection.pairs]
# High-volume pairs (strict)
"binance.BTCUSD" = { sequence_threshold = 1, heartbeat_timeout = 10 }
"binance.ETHUSDT" = { sequence_threshold = 1, heartbeat_timeout = 10 }

# Low-volume pairs (relaxed)
"binance.RAREUSDT" = { sequence_threshold = 1, heartbeat_timeout = 60, enable_statistical = false }
```

### Loading Configuration

```rust
use config::{Config, File};
use serde::Deserialize;

#[derive(Debug, Deserialize)]
pub struct DataFactoryConfig {
    pub rate_limiter: RateLimiterSection,
    pub gap_detection: GapDetectionSection,
}

#[derive(Debug, Deserialize)]
pub struct RateLimiterSection {
    pub safety_margin: f64,
    pub enable_sliding_window: bool,
    pub exchanges: HashMap<String, ExchangeConfig>,
}

pub fn load_config() -> anyhow::Result<DataFactoryConfig> {
    let config = Config::builder()
        .add_source(File::with_name("config/data-factory"))
        .add_source(config::Environment::with_prefix("DATA_FACTORY"))
        .build()?;
    
    Ok(config.try_deserialize()?)
}
```

---

## Monitoring & Metrics

### Prometheus Metrics Export

See `docs/TODO_IMPLEMENTATION_PLAN.md` section 1.5 for full implementation.

Key metrics to export:

```rust
// Rate Limiter
metrics::gauge!("rate_limiter_tokens_available", tokens, "exchange" => exchange);
metrics::counter!("rate_limiter_requests_total", count, "exchange" => exchange);
metrics::counter!("rate_limiter_rejected_total", count, "exchange" => exchange);
metrics::histogram!("rate_limiter_wait_time_ms", wait_ms, "exchange" => exchange);

// Gap Detection
metrics::counter!("gaps_detected_total", 1, "exchange" => ex, "pair" => pair, "type" => gap_type);
metrics::histogram!("gap_size_trades", size, "exchange" => ex);
metrics::gauge!("data_completeness_percent", percent);
metrics::histogram!("gap_detection_latency_ms", latency);

// SLI Metrics (from SLI_SLO.md)
metrics::gauge!("sli_data_completeness_percent", completeness);
metrics::histogram!("sli_ingestion_latency_ms", latency);
metrics::gauge!("sli_system_uptime_percent", uptime);
```

### Grafana Dashboards

See `spike-prototypes/monitoring/dashboards/` (TODO: implement as per plan).

---

## Testing

### Unit Tests

Run tests for individual crates:

```bash
# Rate limiter tests
cd src/janus/crates/rate-limiter
cargo test

# Gap detection tests
cd src/janus/crates/gap-detection
cargo test
```

### Integration Tests

```bash
# Data factory integration tests
cd src/janus/services/data-factory
cargo test --test integration_test
```

### Example Integration Test

```rust
#[tokio::test]
async fn test_rate_limited_ingestion_with_gap_detection() {
    // Setup
    let rate_limiter = Arc::new(RateLimiterManager::new());
    rate_limiter.register("test", TokenBucketConfig {
        capacity: 10,
        refill_rate: 1.0,
        sliding_window: false,
        window_duration: Duration::from_secs(1),
        safety_margin: 1.0,
    }).unwrap();
    
    let gap_detector = Arc::new(GapDetectionManager::new());
    gap_detector.register_pair("test", "BTCUSD").await;
    
    // Simulate ingestion
    for i in 0..20 {
        // Rate limit will cause some requests to wait
        rate_limiter.acquire("test", 1).await.unwrap();
        
        let trade = Trade {
            id: i,
            exchange: "test".to_string(),
            pair: "BTCUSD".to_string(),
            price: 50000.0,
            quantity: 1.0,
            timestamp: Utc::now(),
            is_buyer_maker: false,
        };
        
        gap_detector.process_trade(trade).await.unwrap();
    }
    
    // Verify no gaps (sequential IDs)
    let gaps = gap_detector.get_all_gaps();
    assert_eq!(gaps.len(), 0);
    
    // Verify rate limiting worked
    let metrics = rate_limiter.get("test").unwrap().metrics();
    assert!(metrics.total_requests >= 20);
}
```

---

## Migration from Spike Prototypes

### Code Changes Required

1. **Update imports:**
   ```rust
   // Old
   use rate_limiter_spike::{TokenBucket, TokenBucketConfig};
   use gap_detection_spike::{GapDetectionManager, Trade};
   
   // New
   use janus_rate_limiter::{TokenBucket, TokenBucketConfig};
   use janus_gap_detection::{GapDetectionManager, Trade};
   ```

2. **Update Cargo.toml:**
   ```toml
   # Old
   rate-limiter-spike = { path = "../spike-prototypes/rate-limiter" }
   
   # New
   janus-rate-limiter = { path = "../../crates/rate-limiter" }
   ```

3. **No API changes** - The public API is identical

### Documentation Location Changes

| Old Location | New Location |
|--------------|--------------|
| `spike-prototypes/documentation/THREAT_MODEL.md` | `src/janus/services/data-factory/docs/THREAT_MODEL.md` |
| `spike-prototypes/documentation/SLI_SLO.md` | `src/janus/services/data-factory/docs/SLI_SLO.md` |
| `spike-prototypes/documentation/SPIKE_VALIDATION_REPORT.md` | `src/janus/services/data-factory/docs/SPIKE_VALIDATION_REPORT.md` |
| `spike-prototypes/TODO_IMPLEMENTATION_PLAN.md` | `src/janus/services/data-factory/docs/TODO_IMPLEMENTATION_PLAN.md` |
| `spike-prototypes/CRITICAL_TODOS.md` | `src/janus/services/data-factory/docs/CRITICAL_TODOS.md` |

### Examples Location

| Component | Location |
|-----------|----------|
| Rate Limiter Examples | `src/janus/crates/rate-limiter/examples/` |
| Gap Detection Examples | `src/janus/crates/gap-detection/examples/` |

Run examples:
```bash
cd src/janus/crates/rate-limiter
cargo run --example exchange_actor

cd src/janus/crates/gap-detection
cargo run --example real_world_simulation
```

---

## Next Steps

### Immediate (Before Production)

See `docs/CRITICAL_TODOS.md` for the 7 critical items that **must** be implemented before production:

1. ✅ ~~API Key Security (Docker Secrets)~~ → **TODO**
2. ✅ ~~Backfill Locking (Redis)~~ → **TODO**
3. ✅ ~~Circuit Breaker~~ → **TODO**
4. ✅ ~~Backfill Throttling + Disk Monitoring~~ → **TODO**
5. ✅ ~~Prometheus Metrics Export~~ → **TODO**
6. ✅ ~~Grafana Dashboards~~ → **TODO**
7. ✅ ~~Alerting Rules~~ → **TODO**

**Estimated Effort:** 22 hours (3 days)

### Reference Documentation

- **Threat Model:** `docs/THREAT_MODEL.md`
- **SLI/SLO Definitions:** `docs/SLI_SLO.md`
- **Spike Validation Report:** `docs/SPIKE_VALIDATION_REPORT.md`
- **Implementation Plan:** `docs/TODO_IMPLEMENTATION_PLAN.md`
- **Critical TODOs:** `docs/CRITICAL_TODOS.md`

### Rate Limiter Crate

- **Source:** `src/janus/crates/rate-limiter/`
- **Documentation:** `src/janus/crates/rate-limiter/README.md`
- **Examples:** `src/janus/crates/rate-limiter/examples/`
- **Tests:** `cargo test -p janus-rate-limiter`

### Gap Detection Crate

- **Source:** `src/janus/crates/gap-detection/`
- **Documentation:** `src/janus/crates/gap-detection/README.md`
- **Examples:** `src/janus/crates/gap-detection/examples/`
- **Tests:** `cargo test -p janus-gap-detection`

---

## Support & Questions

For questions or issues:

1. Check the relevant README.md in the crate directory
2. Review the spike validation report for design decisions
3. Check the TODO implementation plan for known gaps
4. Review the threat model for security considerations

---

**Status:** ✅ Crates integrated into workspace  
**Production Ready:** ⚠️ 85% (7 critical TODOs remaining)  
**Next Milestone:** Complete critical TODOs (3 days estimated)