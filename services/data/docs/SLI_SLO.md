# Service Level Indicators (SLIs) & Objectives (SLOs)
# Crypto Data Factory - Production Metrics Framework

**Document Version:** 1.0  
**Last Updated:** 2024  
**Review Frequency:** Quarterly  
**System:** Rust Data Factory for BTC/ETH/SOL Market Data Ingestion

---

## Executive Summary

This document defines measurable Service Level Indicators (SLIs) and target Service Level Objectives (SLOs) for the Data Factory. These metrics are critical for:

1. **Operational Excellence** - Ensuring the system meets business requirements
2. **Incident Response** - Defining "healthy" vs "degraded" states
3. **Capacity Planning** - Identifying when to scale
4. **SLA Commitments** - Supporting downstream service guarantees

**Key Targets:**
- **Data Freshness:** 99% of trades ingested within 500ms
- **Data Completeness:** 99.9% of trades captured (< 0.1% data loss)
- **System Availability:** 99.5% uptime (43.8 minutes downtime/month allowed)
- **Backfill Latency:** 95% of gaps filled within 10 minutes

---

## Table of Contents

1. [SLI Categories](#sli-categories)
2. [Ingestion Metrics](#ingestion-metrics)
3. [Data Quality Metrics](#data-quality-metrics)
4. [Availability Metrics](#availability-metrics)
5. [Performance Metrics](#performance-metrics)
6. [Dependency Metrics](#dependency-metrics)
7. [Error Budget](#error-budget)
8. [Alerting Thresholds](#alerting-thresholds)
9. [Measurement Implementation](#measurement-implementation)
10. [Dashboard Requirements](#dashboard-requirements)

---

## SLI Categories

We track 5 categories of SLIs:

| Category | Purpose | Example SLI |
|----------|---------|-------------|
| **Availability** | Is the system up? | % of successful health checks |
| **Latency** | How fast is data processed? | P99 ingestion latency |
| **Correctness** | Is data accurate? | % of trades passing validation |
| **Completeness** | Are we missing data? | Gap detection rate |
| **Freshness** | How stale is data? | Time from exchange → database |

---

## Ingestion Metrics

### SLI-I1: Trade Ingestion Latency

**Definition:** Time from trade timestamp at exchange to QuestDB commit

**Measurement:**
```rust
let ingestion_latency = questdb_commit_time - exchange_timestamp;
histogram!(
    "ingestion_latency_seconds",
    ingestion_latency.as_secs_f64(),
    "exchange" => exchange_name,
    "pair" => pair_name
);
```

**SLO Targets:**

| Percentile | Target | Rationale |
|------------|--------|-----------|
| P50 | < 200ms | Median should be near real-time |
| P95 | < 500ms | Acceptable for most use cases |
| **P99** | **< 1000ms** | **PRIMARY SLO** - Critical for HFT |
| P99.9 | < 5000ms | Tolerate occasional spikes |

**Alert Thresholds:**
- **Warning:** P95 > 750ms for 5 minutes
- **Critical:** P99 > 2000ms for 2 minutes

**Error Budget:** 1% of trades can exceed P99 target

---

### SLI-I2: Throughput (Trades per Second)

**Definition:** Number of trades successfully written to QuestDB per second

**Measurement:**
```rust
counter!("trades_ingested_total", "exchange" => exchange, "pair" => pair);

// Rate calculation (in PromQL):
// rate(trades_ingested_total[1m])
```

**SLO Targets:**

| Metric | Target | Peak Capacity |
|--------|--------|---------------|
| Sustained | 10,000 trades/sec | 20,000 trades/sec |
| Burst (1 min) | 50,000 trades/sec | During BTC flash crash |
| Per Exchange | 5,000 trades/sec | Binance alone |

**Alert Thresholds:**
- **Warning:** Throughput drops below 1,000 trades/sec during market hours
- **Critical:** Throughput = 0 for 30 seconds

**Measurement Window:** 1-minute rolling average

---

### SLI-I3: WebSocket Connection Uptime

**Definition:** Percentage of time WebSocket connections are active and receiving data

**Measurement:**
```rust
// Record connection state changes
gauge!(
    "websocket_connected",
    if connected { 1.0 } else { 0.0 },
    "exchange" => exchange,
    "pair" => pair
);
```

**SLO Target:** 99.5% uptime per exchange

**Calculation:**
```
Uptime % = (Total Time - Disconnected Time) / Total Time * 100

Example:
- Month = 30 days = 43,200 minutes
- Allowed downtime = 0.5% = 216 minutes (3.6 hours)
```

**Alert Thresholds:**
- **Warning:** Connection down for 5 minutes
- **Critical:** Connection down for 15 minutes OR 3 reconnects in 10 minutes

**Exclusions:** Scheduled maintenance windows (announced 24h in advance)

---

### SLI-I4: Reconnection Success Rate

**Definition:** Percentage of WebSocket reconnection attempts that succeed

**Measurement:**
```rust
counter!("websocket_reconnect_attempts_total", "exchange" => exchange);
counter!("websocket_reconnect_success_total", "exchange" => exchange);

// Rate = success / attempts * 100
```

**SLO Target:** 95% of reconnects succeed within 30 seconds

**Alert Thresholds:**
- **Warning:** Success rate < 90% over 1 hour
- **Critical:** 3 consecutive reconnect failures

---

## Data Quality Metrics

### SLI-Q1: Data Completeness (Gap Detection Rate)

**Definition:** Percentage of expected trades successfully captured (no gaps)

**Measurement:**
```rust
// Sequence gap detection
let gap_rate = detected_gaps / total_expected_trades * 100;

gauge!("data_completeness_percent", 100.0 - gap_rate);
```

**SLO Target:** 99.9% completeness (< 0.1% data loss)

**Calculation Example:**
```
Expected trades (based on sequence IDs): 1,000,000
Detected gaps: 500 trades missing
Completeness = (1,000,000 - 500) / 1,000,000 = 99.95% ✓
```

**Alert Thresholds:**
- **Warning:** Completeness < 99.5% over 10 minutes
- **Critical:** Completeness < 99.0% OR single gap > 1000 trades

**Measurement Window:** 1-hour rolling window

---

### SLI-Q2: Cross-Exchange Price Deviation

**Definition:** Percentage of trades where price deviates >5% from other exchanges

**Measurement:**
```rust
fn validate_price(price: f64, exchange: &str, pair: &str) -> bool {
    let reference_price = get_median_price_across_exchanges(pair);
    let deviation = (price - reference_price).abs() / reference_price;
    
    histogram!(
        "price_deviation_percent",
        deviation * 100.0,
        "exchange" => exchange,
        "pair" => pair
    );
    
    deviation < 0.05 // 5% threshold
}
```

**SLO Target:** < 0.01% of trades have suspicious deviations

**Alert Thresholds:**
- **Warning:** Single trade deviates > 5%
- **Critical:** > 10 trades deviate > 5% in 1 minute (likely exchange compromise)

**Actions on Violation:**
1. Flag suspicious trades in database
2. Switch to backup exchange
3. Trigger manual review

---

### SLI-Q3: Duplicate Detection Rate

**Definition:** Percentage of duplicate trades detected and filtered

**Measurement:**
```rust
counter!("duplicate_trades_detected_total", "exchange" => exchange);
counter!("unique_trades_ingested_total", "exchange" => exchange);

// Deduplication rate = duplicates / (duplicates + unique) * 100
```

**SLO Target:** < 0.5% duplicate rate

**Alert Thresholds:**
- **Warning:** Duplicate rate > 1% over 10 minutes
- **Critical:** Duplicate rate > 5% (indicates reconnection overlap issue)

---

### SLI-Q4: Data Validation Success Rate

**Definition:** Percentage of trades passing all validation checks

**Measurement:**
```rust
// Validation checks:
// 1. Price > 0
// 2. Amount > 0
// 3. Timestamp within reasonable range
// 4. Trade ID is monotonic

counter!("trades_validated_total", "result" => "pass");
counter!("trades_validated_total", "result" => "fail", "reason" => reason);
```

**SLO Target:** 99.99% pass validation

**Alert Thresholds:**
- **Warning:** Validation failure rate > 0.1%
- **Critical:** Validation failure rate > 1% (likely exchange API change)

---

## Availability Metrics

### SLI-A1: Health Check Success Rate

**Definition:** Percentage of successful health check responses

**Measurement:**
```rust
// HTTP endpoint: /health
// Returns 200 if:
// - At least 1 exchange connected
// - QuestDB reachable
// - Redis reachable
// - No critical errors in last 5 minutes

counter!("health_check_total", "status" => if healthy { "success" } else { "failure" });
```

**SLO Target:** 99.5% success rate

**Alert Thresholds:**
- **Warning:** 3 consecutive failures
- **Critical:** 5 consecutive failures OR health check timeout

**Health Check Frequency:** Every 10 seconds

---

### SLI-A2: Service Uptime

**Definition:** Percentage of time the Data Factory container is running

**Measurement:**
```
Uptime = (Total Time - Crash Time - Restart Time) / Total Time * 100
```

**SLO Target:** 99.5% uptime

**Allowed Downtime:**
- Per month: 3.6 hours
- Per week: 50 minutes
- Per day: 7.2 minutes

**Exclusions:**
- Planned deployments (max 2 per week, < 5 min each)
- Force majeure (cloud provider outages)

**Tracking:**
```prometheus
# PromQL
(time() - process_start_time_seconds) / time() * 100
```

---

### SLI-A3: Dependency Availability

**Definition:** Percentage of time each dependency is available

**Measurement:**
```rust
// Track each dependency separately
gauge!("dependency_available", if available { 1.0 } else { 0.0 }, "service" => "questdb");
gauge!("dependency_available", if available { 1.0 } else { 0.0 }, "service" => "redis");
gauge!("dependency_available", if available { 1.0 } else { 0.0 }, "service" => "binance");
```

**SLO Targets:**

| Dependency | Target Uptime | Blast Radius |
|------------|---------------|--------------|
| QuestDB | 99.9% | Critical - no persistence |
| Redis | 99.5% | High - stale data in Forward service |
| Binance | 99.0% | Medium - failover to Bybit |
| Bybit | 99.0% | Medium - failover to Kucoin |
| CoinMarketCap | 95.0% | Low - Fear/Greed only |

**Alert Thresholds:**
- **Critical:** QuestDB down for 1 minute
- **Warning:** Any exchange down for 5 minutes

---

## Performance Metrics

### SLI-P1: Rate Limiter Efficiency

**Definition:** Percentage of requests that pass rate limiter without waiting

**Measurement:**
```rust
counter!("rate_limit_requests_total", "result" => "immediate_pass");
counter!("rate_limit_requests_total", "result" => "waited");
counter!("rate_limit_requests_total", "result" => "rejected");

// Efficiency = immediate_pass / total * 100
```

**SLO Target:** > 95% requests pass immediately

**Alert Thresholds:**
- **Warning:** Efficiency < 90% (rate limiter too conservative)
- **Critical:** > 10 requests rejected per minute (approaching IP ban)

---

### SLI-P2: Backfill Latency

**Definition:** Time from gap detection to backfill completion

**Measurement:**
```rust
histogram!(
    "backfill_duration_seconds",
    duration.as_secs_f64(),
    "exchange" => exchange,
    "gap_size" => gap_size_bucket(gap.missing_count)
);
```

**SLO Targets:**

| Gap Size | P95 Target | P99 Target |
|----------|------------|------------|
| < 100 trades | 30 seconds | 1 minute |
| 100-1000 | 5 minutes | 10 minutes |
| 1000-10000 | 30 minutes | 1 hour |
| > 10000 | Manual intervention | N/A |

**Alert Thresholds:**
- **Warning:** Backfill queue depth > 10
- **Critical:** Backfill job running > 1 hour

---

### SLI-P3: Memory Usage

**Definition:** Percentage of container memory limit used

**Measurement:**
```rust
gauge!("memory_usage_bytes", current_usage);
gauge!("memory_limit_bytes", container_limit);

// Usage % = current / limit * 100
```

**SLO Target:** < 80% memory usage during normal operation

**Alert Thresholds:**
- **Warning:** Memory usage > 80% for 5 minutes
- **Critical:** Memory usage > 95% (risk of OOM kill)

**Capacity Planning Trigger:** If P95 memory > 70%, scale up

---

### SLI-P4: CPU Usage

**Definition:** Percentage of CPU time used

**Measurement:**
```rust
gauge!("cpu_usage_percent", cpu_usage);
```

**SLO Target:** < 60% CPU usage during normal operation

**Alert Thresholds:**
- **Warning:** CPU > 80% for 10 minutes
- **Critical:** CPU > 95% for 5 minutes

**Expected Profile:**
- Idle: 5-10% (heartbeats, health checks)
- Normal: 30-50% (real-time ingestion)
- Peak: 70-80% (volatility spikes, backfills)

---

## Dependency Metrics

### SLI-D1: QuestDB Write Latency

**Definition:** Time to write a batch of trades to QuestDB via ILP

**Measurement:**
```rust
histogram!(
    "questdb_write_latency_ms",
    duration.as_millis() as f64,
    "batch_size" => batch_size_bucket(batch.len())
);
```

**SLO Targets:**

| Batch Size | P95 | P99 |
|------------|-----|-----|
| 1-100 | 10ms | 50ms |
| 100-1000 | 50ms | 200ms |
| 1000-10000 | 200ms | 500ms |

**Alert Thresholds:**
- **Warning:** P95 > 100ms
- **Critical:** P99 > 1000ms (QuestDB overloaded)

---

### SLI-D2: Redis Cache Hit Rate

**Definition:** Percentage of Redis reads that return cached data

**Measurement:**
```rust
counter!("redis_requests_total", "result" => "hit");
counter!("redis_requests_total", "result" => "miss");

// Hit rate = hits / (hits + misses) * 100
```

**SLO Target:** > 90% cache hit rate

**Alert Thresholds:**
- **Warning:** Hit rate < 80% (cache eviction issues)
- **Informational:** Hit rate < 50% (cold start or cache clear)

---

### SLI-D3: Exchange API Response Time

**Definition:** Time for exchange REST API to respond

**Measurement:**
```rust
histogram!(
    "exchange_api_latency_ms",
    duration.as_millis() as f64,
    "exchange" => exchange,
    "endpoint" => endpoint
);
```

**SLO Targets:**

| Exchange | P95 | P99 |
|----------|-----|-----|
| Binance | 200ms | 500ms |
| Bybit | 300ms | 800ms |
| Kucoin | 500ms | 1500ms |

**Alert Thresholds:**
- **Warning:** Exchange P95 > 2x normal
- **Critical:** Exchange P99 > 5 seconds (likely degraded)

---

## Error Budget

### Monthly Error Budget Calculation

**SLO:** 99.9% data completeness

**Error Budget:** 0.1% data loss allowed per month

**Calculation:**
```
Trades per month (estimated): 100,000,000
Error budget: 100,000,000 * 0.001 = 100,000 trades

If we lose 50,000 trades in Week 1:
- Budget consumed: 50%
- Remaining budget: 50,000 trades for rest of month
```

### Error Budget Policy

| Budget Remaining | Policy |
|------------------|--------|
| > 75% | **GREEN** - Aggressive features, experimentation allowed |
| 50-75% | **YELLOW** - Cautious deployments, increase monitoring |
| 25-50% | **ORANGE** - Freeze non-critical changes, focus on stability |
| < 25% | **RED** - Emergency mode, only bug fixes, post-mortem required |
| 0% (exhausted) | **BLACK** - Full freeze, executive escalation |

### Budget Tracking

```prometheus
# PromQL - Data Loss Rate
(
  sum(rate(gaps_detected_trades_total[30d])) /
  sum(rate(trades_ingested_total[30d]))
) * 100

# Alert if > 0.1%
```

---

## Alerting Thresholds

### Alert Priority Levels

| Level | Response Time | Escalation | Examples |
|-------|---------------|------------|----------|
| **P0 - Critical** | 15 minutes | Page on-call immediately | Data loss > 1%, all exchanges down |
| **P1 - High** | 1 hour | Alert during business hours | Single exchange down, QuestDB slow |
| **P2 - Medium** | 4 hours | Email + Slack | High latency, cache misses |
| **P3 - Low** | Next business day | Ticket only | Minor config drift |

### Alert Routing

```yaml
# Prometheus Alertmanager Config
route:
  group_by: ['alertname', 'severity']
  group_wait: 10s
  group_interval: 5m
  repeat_interval: 4h
  
  routes:
    - match:
        severity: critical
      receiver: pagerduty
      
    - match:
        severity: warning
      receiver: slack
      
    - match:
        severity: info
      receiver: email
```

### Critical Alerts (P0)

```yaml
# 1. Data Completeness SLO Violation
- alert: DataCompletenessViolation
  expr: (1 - rate(gaps_detected_trades_total[1h]) / rate(trades_ingested_total[1h])) < 0.999
  for: 5m
  labels:
    severity: critical
  annotations:
    summary: "Data completeness below SLO (< 99.9%)"
    
# 2. All Exchanges Disconnected
- alert: AllExchangesDown
  expr: sum(websocket_connected) == 0
  for: 1m
  labels:
    severity: critical
  annotations:
    summary: "No active exchange connections"
    
# 3. QuestDB Unreachable
- alert: QuestDBDown
  expr: up{job="questdb"} == 0
  for: 1m
  labels:
    severity: critical
  annotations:
    summary: "QuestDB is unreachable"
    
# 4. Ingestion Completely Stopped
- alert: IngestionStopped
  expr: rate(trades_ingested_total[5m]) == 0
  for: 2m
  labels:
    severity: critical
  annotations:
    summary: "No trades ingested in 5 minutes"
```

---

## Measurement Implementation

### Prometheus Metrics Exporter

```rust
use prometheus::{
    Counter, Gauge, Histogram, HistogramOpts, IntCounter, IntGauge, Opts, Registry,
};

pub struct MetricsCollector {
    // Ingestion
    pub trades_ingested: IntCounter,
    pub ingestion_latency: Histogram,
    
    // Quality
    pub gaps_detected: IntCounter,
    pub duplicates_filtered: IntCounter,
    
    // Availability
    pub websocket_connected: IntGauge,
    pub health_check_success: IntCounter,
    
    // Performance
    pub memory_usage_bytes: Gauge,
    pub cpu_usage_percent: Gauge,
}

impl MetricsCollector {
    pub fn new(registry: &Registry) -> Result<Self, prometheus::Error> {
        let trades_ingested = IntCounter::with_opts(
            Opts::new("trades_ingested_total", "Total trades successfully ingested")
        )?;
        registry.register(Box::new(trades_ingested.clone()))?;
        
        let ingestion_latency = Histogram::with_opts(
            HistogramOpts::new("ingestion_latency_seconds", "Time from exchange to database")
                .buckets(vec![0.01, 0.05, 0.1, 0.25, 0.5, 1.0, 2.5, 5.0, 10.0])
        )?;
        registry.register(Box::new(ingestion_latency.clone()))?;
        
        // ... register all metrics
        
        Ok(Self {
            trades_ingested,
            ingestion_latency,
            // ...
        })
    }
    
    pub fn record_trade_ingested(&self, latency: Duration) {
        self.trades_ingested.inc();
        self.ingestion_latency.observe(latency.as_secs_f64());
    }
}
```

### HTTP Metrics Endpoint

```rust
use axum::{routing::get, Router};

async fn metrics_handler(registry: Registry) -> String {
    use prometheus::Encoder;
    let encoder = prometheus::TextEncoder::new();
    
    let metric_families = registry.gather();
    let mut buffer = vec![];
    encoder.encode(&metric_families, &mut buffer).unwrap();
    
    String::from_utf8(buffer).unwrap()
}

let app = Router::new()
    .route("/metrics", get(metrics_handler))
    .route("/health", get(health_handler));
```

---

## Dashboard Requirements

### Primary Dashboard: Real-Time Operations

**Panels:**

1. **Ingestion Overview**
   - Trades/second (per exchange)
   - Ingestion latency (P50, P95, P99)
   - WebSocket connection status (green/red)

2. **Data Quality**
   - Completeness % (rolling 1h)
   - Gaps detected (count)
   - Duplicate rate

3. **System Health**
   - Memory usage (%)
   - CPU usage (%)
   - QuestDB write latency
   - Redis cache hit rate

4. **Alerts**
   - Active alerts (severity-coded)
   - Recent incidents (last 24h)

**Refresh Rate:** 10 seconds

---

### Secondary Dashboard: SLO Compliance

**Panels:**

1. **SLO Burndown**
   - Monthly error budget consumption
   - Projected budget exhaustion date

2. **SLO Trends (30 days)**
   - Data completeness: 99.9% target line
   - Ingestion latency P99: 1000ms target line
   - Uptime: 99.5% target line

3. **SLO Violations**
   - Count of violations per SLO
   - Time to recovery per incident

**Refresh Rate:** 1 minute

---

### Grafana Dashboard JSON

```json
{
  "dashboard": {
    "title": "Data Factory - Real-Time Operations",
    "panels": [
      {
        "title": "Trades Ingested (per second)",
        "targets": [
          {
            "expr": "sum(rate(trades_ingested_total[1m])) by (exchange)",
            "legendFormat": "{{exchange}}"
          }
        ],
        "type": "graph"
      },
      {
        "title": "Ingestion Latency (P99)",
        "targets": [
          {
            "expr": "histogram_quantile(0.99, rate(ingestion_latency_seconds_bucket[5m]))",
            "legendFormat": "P99"
          }
        ],
        "type": "graph",
        "alert": {
          "conditions": [
            {
              "evaluator": { "params": [1.0], "type": "gt" },
              "operator": { "type": "and" },
              "query": { "params": ["A", "5m", "now"] },
              "reducer": { "params": [], "type": "avg" },
              "type": "query"
            }
          ],
          "name": "P99 Latency > 1s"
        }
      }
    ]
  }
}
```

---

## SLO Review Process

### Weekly Review (Ops Team)
- Review SLO compliance for past week
- Identify trending violations
- Update error budget tracking
- Plan remediations for violations

### Monthly Review (Engineering + Product)
- Full SLO compliance report
- Error budget retrospective
- Adjust SLOs if consistently over/under target
- Prioritize reliability work vs features

### Quarterly Review (Executive)
- SLO alignment with business objectives
- Cost vs reliability trade-offs
- Major architecture changes needed
- Update this document

---

## Appendix A: PromQL Queries

### Data Completeness
```promql
# Current completeness %
(1 - (
  sum(rate(gaps_detected_trades_total[1h])) /
  sum(rate(trades_ingested_total[1h]))
)) * 100

# Trades lost per hour
sum(rate(gaps_detected_trades_total[1h])) * 3600
```

### Ingestion Latency Percentiles
```promql
# P50
histogram_quantile(0.50, rate(ingestion_latency_seconds_bucket[5m]))

# P95
histogram_quantile(0.95, rate(ingestion_latency_seconds_bucket[5m]))

# P99
histogram_quantile(0.99, rate(ingestion_latency_seconds_bucket[5m]))
```

### System Uptime
```promql
# Uptime in days
(time() - process_start_time_seconds{job="data-factory"}) / 86400

# Uptime percentage (last 30 days)
avg_over_time(up{job="data-factory"}[30d]) * 100
```

---

## Appendix B: SLO Summary Table

| ID | SLI | Target | Alert Threshold | Priority |
|----|-----|--------|-----------------|----------|
| I1 | Ingestion Latency P99 | < 1000ms | > 2000ms | P0 |
| I2 | Throughput | 10k trades/sec | < 1k trades/sec | P0 |
| I3 | WebSocket Uptime | 99.5% | Down > 15min | P0 |
| I4 | Reconnect Success | 95% | < 90% | P1 |
| Q1 | Data Completeness | 99.9% | < 99.5% | P0 |
| Q2 | Price Deviation | < 0.01% | > 5% deviation | P0 |
| Q3 | Duplicate Rate | < 0.5% | > 5% | P1 |
| Q4 | Validation Success | 99.99% | < 99% | P1 |
| A1 | Health Check Success | 99.5% | 5 failures | P0 |
| A2 | Service Uptime | 99.5% | Down > 5min | P0 |
| A3 | QuestDB Uptime | 99.9% | Down > 1min | P0 |
| P1 | Rate Limiter Efficiency | > 95% | < 90% | P2 |
| P2 | Backfill Latency (P95) | < 10min | > 1 hour | P1 |
| P3 | Memory Usage | < 80% | > 95% | P0 |
| P4 | CPU Usage | < 60% | > 95% | P1 |

---

## Document History

| Version | Date | Author | Changes |
|---------|------|--------|---------|
| 1.0 | 2024 | Engineering Team | Initial SLI/SLO definitions |

**Next Review:** Q1 2025