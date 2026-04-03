# Execution Metrics Quick Reference

**Quick guide for using Vision execution metrics and monitoring**

---

## Table of Contents

1. [Basic Usage](#basic-usage)
2. [Metric Reference](#metric-reference)
3. [Common Queries](#common-queries)
4. [Alerting Examples](#alerting-examples)
5. [Troubleshooting](#troubleshooting)

---

## Basic Usage

### Simple Integration

```rust
use vision::execution::{InstrumentedExecutionManager, OrderRequest, Side, ExecutionStrategy};
use std::time::Duration;

// Create instrumented manager (metrics enabled by default)
let mut exec_manager = InstrumentedExecutionManager::new();

// Submit orders - metrics recorded automatically
let order_id = exec_manager.submit_order(OrderRequest {
    symbol: "BTCUSD".to_string(),
    quantity: 1.0,
    side: Side::Buy,
    strategy: ExecutionStrategy::TWAP {
        duration: Duration::from_secs(60),
        num_slices: 6,
    },
    limit_price: None,
    venues: None,
});

// Process executions - metrics updated automatically
exec_manager.process();

// Check health
if !exec_manager.is_healthy() {
    eprintln!("Warning: Execution manager unhealthy!");
}

// Export metrics for Prometheus
let metrics = exec_manager.export_metrics()?;
println!("{}", metrics);
```

### Run Metrics Server

```bash
# Terminal 1: Start metrics server
cargo run --example metrics_server --release

# Terminal 2: Access endpoints
curl http://localhost:9090/metrics     # Prometheus format
curl http://localhost:9090/health      # JSON health check
open http://localhost:9090/status      # HTML dashboard

# Terminal 3: Run live trading
cargo run --example live_pipeline_with_execution --release
```

---

## Metric Reference

### Volume Metrics

| Metric | Type | Labels | Description |
|--------|------|--------|-------------|
| `vision_execution_total` | Counter | `side`, `venue` | Total number of executions |
| `vision_execution_quantity_total` | Gauge | `side`, `venue` | Total quantity executed |
| `vision_execution_cost_total` | Gauge | `side`, `venue` | Total execution cost ($) |

**Example Values**:
```
vision_execution_total{side="buy",venue="binance"} 1247
vision_execution_total{side="sell",venue="binance"} 892
vision_execution_quantity_total{side="buy",venue="binance"} 15234.5
```

### Slippage Metrics

| Metric | Type | Labels | Description |
|--------|------|--------|-------------|
| `vision_execution_slippage_bps` | Histogram | `side`, `venue` | Slippage distribution (bps) |
| `vision_execution_avg_slippage_bps` | Gauge | `venue` | Average slippage per venue |
| `vision_execution_vwap_slippage_bps` | Gauge | - | Volume-weighted avg slippage |

**Buckets**: -50, -25, -10, -5, -2, -1, 0, 1, 2, 5, 10, 25, 50 bps

### Quality Metrics

| Metric | Type | Labels | Description |
|--------|------|--------|-------------|
| `vision_execution_quality_score` | Gauge | - | Overall quality score (0-100) |
| `vision_execution_fill_rate_pct` | Gauge | `order_type` | Fill rate by order type |
| `vision_execution_implementation_shortfall_pct` | Gauge | - | Implementation shortfall |

**Quality Score Formula**:
```
quality_score = 100 - (abs(avg_slippage_bps) * 2) - (failure_rate * 50)
```

### Latency Metrics

| Metric | Type | Labels | Description |
|--------|------|--------|-------------|
| `vision_execution_latency_us` | Histogram | - | Processing latency (μs) |
| `vision_execution_venue_latency_us` | Histogram | `venue` | Venue-specific latency |
| `vision_execution_time_to_fill_seconds` | Histogram | - | Time from submit to fill |

**Latency Buckets**: 10, 50, 100, 500, 1000, 5000, 10000, 50000 μs

### Order Type Metrics

| Metric | Type | Labels | Description |
|--------|------|--------|-------------|
| `vision_execution_market_total` | Counter | - | Market order count |
| `vision_execution_limit_total` | Counter | - | Limit order count |
| `vision_execution_twap_total` | Counter | - | TWAP order count |
| `vision_execution_vwap_total` | Counter | - | VWAP order count |

### Error Metrics

| Metric | Type | Labels | Description |
|--------|------|--------|-------------|
| `vision_execution_failed_total` | Counter | `reason` | Failed execution count |
| `vision_execution_rejected_total` | Counter | `reason` | Rejected order count |
| `vision_execution_partial_fills_total` | Counter | - | Partial fill count |

---

## Common Queries

### Throughput & Volume

```promql
# Executions per second (1-minute average)
rate(vision_execution_total[1m])

# Executions per second by venue
sum by (venue) (rate(vision_execution_total[1m]))

# Total daily volume
sum(increase(vision_execution_quantity_total[24h]))

# Buy vs Sell ratio
sum(rate(vision_execution_total{side="buy"}[5m])) /
sum(rate(vision_execution_total{side="sell"}[5m]))
```

### Slippage Analysis

```promql
# Current average slippage
avg(vision_execution_avg_slippage_bps)

# 95th percentile slippage (5-minute window)
histogram_quantile(0.95, 
  rate(vision_execution_slippage_bps_bucket[5m]))

# 99th percentile slippage
histogram_quantile(0.99,
  rate(vision_execution_slippage_bps_bucket[5m]))

# Slippage by venue (sorted)
topk(5, vision_execution_avg_slippage_bps)
```

### Performance Metrics

```promql
# Current quality score
vision_execution_quality_score

# Quality score trend (1-hour)
avg_over_time(vision_execution_quality_score[1h])

# Median processing latency
histogram_quantile(0.5, 
  rate(vision_execution_latency_us_bucket[5m]))

# p99 latency
histogram_quantile(0.99,
  rate(vision_execution_latency_us_bucket[5m]))
```

### Error Rates

```promql
# Overall failure rate
rate(vision_execution_failed_total[5m]) /
rate(vision_execution_total[5m])

# Failure rate by reason
sum by (reason) (rate(vision_execution_failed_total[5m]))

# Rejection rate
rate(vision_execution_rejected_total[5m]) /
(rate(vision_execution_total[5m]) + 
 rate(vision_execution_rejected_total[5m]))
```

### Strategy Distribution

```promql
# TWAP usage percentage
rate(vision_execution_twap_total[1h]) /
(rate(vision_execution_twap_total[1h]) +
 rate(vision_execution_vwap_total[1h]) +
 rate(vision_execution_market_total[1h]) +
 rate(vision_execution_limit_total[1h]))

# Strategy counts (last hour)
sum(increase(vision_execution_twap_total[1h]))
sum(increase(vision_execution_vwap_total[1h]))
sum(increase(vision_execution_market_total[1h]))
sum(increase(vision_execution_limit_total[1h]))
```

---

## Alerting Examples

### Prometheus Alert Rules

```yaml
groups:
  - name: vision_execution_alerts
    interval: 30s
    rules:
      # Critical: Execution manager down
      - alert: ExecutionManagerDown
        expr: up{job="vision-execution"} == 0
        for: 1m
        labels:
          severity: critical
          component: execution
        annotations:
          summary: "Execution manager is down"
          description: "Cannot scrape metrics from {{ $labels.instance }}"

      # Critical: High failure rate
      - alert: HighExecutionFailureRate
        expr: |
          rate(vision_execution_failed_total[5m]) /
          rate(vision_execution_total[5m]) > 0.1
        for: 3m
        labels:
          severity: critical
          component: execution
        annotations:
          summary: "High execution failure rate"
          description: "Failure rate is {{ $value | humanizePercentage }} (threshold: 10%)"

      # Warning: Elevated slippage
      - alert: HighExecutionSlippage
        expr: avg(vision_execution_avg_slippage_bps) > 10
        for: 5m
        labels:
          severity: warning
          component: execution
        annotations:
          summary: "High execution slippage"
          description: "Average slippage is {{ $value | printf \"%.2f\" }} bps (threshold: 10 bps)"

      # Warning: Quality degradation
      - alert: LowExecutionQuality
        expr: vision_execution_quality_score < 80
        for: 10m
        labels:
          severity: warning
          component: execution
        annotations:
          summary: "Execution quality degraded"
          description: "Quality score is {{ $value | printf \"%.1f\" }}/100 (threshold: 80)"

      # Warning: High latency
      - alert: HighExecutionLatency
        expr: |
          histogram_quantile(0.99,
            rate(vision_execution_latency_us_bucket[5m])) > 5000
        for: 5m
        labels:
          severity: warning
          component: execution
        annotations:
          summary: "High execution latency"
          description: "p99 latency is {{ $value | printf \"%.0f\" }} μs (threshold: 5000 μs)"

      # Info: Unusual rejection rate
      - alert: UnusualRejectionRate
        expr: |
          rate(vision_execution_rejected_total[10m]) /
          (rate(vision_execution_total[10m]) +
           rate(vision_execution_rejected_total[10m])) > 0.05
        for: 15m
        labels:
          severity: info
          component: execution
        annotations:
          summary: "Unusual order rejection rate"
          description: "Rejection rate is {{ $value | humanizePercentage }} (threshold: 5%)"
```

### Grafana Alert Conditions

```json
{
  "conditions": [
    {
      "evaluator": {
        "params": [80],
        "type": "lt"
      },
      "operator": {
        "type": "and"
      },
      "query": {
        "params": ["A", "5m", "now"]
      },
      "reducer": {
        "params": [],
        "type": "avg"
      },
      "type": "query"
    }
  ],
  "frequency": "60s",
  "handler": 1,
  "message": "Execution quality below threshold",
  "name": "Low Execution Quality",
  "noDataState": "no_data",
  "notifications": [
    {"uid": "pagerduty"}
  ]
}
```

---

## Troubleshooting

### Health Check Returns Unhealthy

**Symptoms**:
```bash
$ curl http://localhost:9090/health
{"status": "unhealthy", "healthy": false, "failed_orders": 50, ...}
```

**Diagnosis**:
```rust
let health = exec_manager.health_status();
println!("Total orders: {}", health.total_orders);
println!("Failed: {}", health.failed_orders);
println!("Failure rate: {:.1}%", 
    100.0 * health.failed_orders as f64 / health.total_orders as f64);

// Check recent errors
for error in &health.recent_errors {
    println!("Error: {}", error);
}
```

**Common Causes**:
- Failure rate > 10%
- Network connectivity issues
- Invalid order parameters
- Venue API errors

**Resolution**:
1. Check recent error messages
2. Verify venue connectivity
3. Review order parameters
4. Check rate limits
5. Reset if transient: `exec_manager.reset()`

### High Slippage

**Symptoms**:
```promql
vision_execution_avg_slippage_bps > 10
```

**Diagnosis**:
```rust
let report = exec_manager.execution_report();
println!("Average slippage: {:.2} bps", report.average_slippage_bps);
println!("VWAP slippage: {:.2} bps", report.vwap_slippage_bps);

// Check per-venue
for venue_stat in report.venue_stats {
    println!("Venue {}: {:.2} bps", 
        venue_stat.venue, venue_stat.average_slippage_bps);
}
```

**Common Causes**:
- Market volatility
- Large order sizes
- Poor timing (market open/close)
- Venue selection

**Resolution**:
1. Use VWAP for large orders
2. Increase TWAP duration
3. Split across multiple venues
4. Avoid high-volatility periods

### Metrics Not Updating

**Symptoms**:
- Stale metrics in Prometheus
- `/metrics` returns old data

**Diagnosis**:
```rust
// Check if processing is running
exec_manager.process();

// Verify metrics export
let metrics = exec_manager.export_metrics()?;
println!("Metrics length: {}", metrics.len());
println!("Last update: {:?}", exec_manager.health_status().last_update.elapsed());
```

**Common Causes**:
- Forgot to call `process()`
- No active orders
- Metrics export error

**Resolution**:
1. Ensure `process()` is called regularly
2. Check for active orders
3. Verify metrics export doesn't error

### High Latency

**Symptoms**:
```promql
histogram_quantile(0.99, 
  rate(vision_execution_latency_us_bucket[5m])) > 5000
```

**Diagnosis**:
```rust
// Profile execution
let start = Instant::now();
exec_manager.process();
let elapsed = start.elapsed();
println!("Process took: {:?}", elapsed);

// Check metrics overhead
let start = Instant::now();
exec_manager.export_metrics()?;
let elapsed = start.elapsed();
println!("Metrics export took: {:?}", elapsed);
```

**Common Causes**:
- Too many active orders
- Metrics export overhead
- Lock contention
- CPU saturation

**Resolution**:
1. Limit concurrent orders
2. Reduce scrape frequency
3. Optimize hot path
4. Scale horizontally

---

## Best Practices

### DO ✅

- Call `process()` regularly (every 100ms - 1s)
- Monitor health status continuously
- Set up alerts on critical metrics
- Export metrics for long-term storage
- Use VWAP/TWAP for large orders
- Track quality score trends

### DON'T ❌

- Call `process()` too frequently (< 10ms)
- Ignore health check failures
- Run without monitoring
- Use market orders for large sizes
- Ignore slippage trends
- Skip metrics export

### Performance Tips

1. **Batch processing**: Process multiple orders in one `process()` call
2. **Lazy metrics**: Only export when scraped
3. **Lock-free paths**: Metrics use atomic operations where possible
4. **Async processing**: Use tokio for concurrent order management
5. **Caching**: Results cached between `process()` calls

---

## Quick Deployment

### Docker Compose

```yaml
version: '3.8'
services:
  vision-execution:
    image: vision-metrics:latest
    ports:
      - "9090:9090"
    environment:
      - RUST_LOG=info
    healthcheck:
      test: ["CMD", "curl", "-f", "http://localhost:9090/health"]
      interval: 30s
      timeout: 10s
      retries: 3

  prometheus:
    image: prom/prometheus:latest
    ports:
      - "9091:9090"
    volumes:
      - ./prometheus.yml:/etc/prometheus/prometheus.yml
    command:
      - '--config.file=/etc/prometheus/prometheus.yml'

  grafana:
    image: grafana/grafana:latest
    ports:
      - "3000:3000"
    environment:
      - GF_SECURITY_ADMIN_PASSWORD=admin
```

```bash
docker-compose up -d
```

---

## Support & Resources

- **Examples**: `examples/live_pipeline_with_execution.rs`, `examples/metrics_server.rs`
- **Documentation**: `docs/week8_day6_summary.md`
- **Tests**: `cargo test --lib execution`
- **Source**: `src/execution/metrics.rs`, `src/execution/instrumented.rs`

---

**Quick Reference Version**: 1.0  
**Last Updated**: Week 8 Day 6  
**Compatibility**: Vision 0.1.0+