# TODO Implementation Plan
## Spike Prototypes & Production Readiness

**Last Updated:** 2024  
**Status:** Active Development  
**Priority:** High - Pre-Production Blockers Identified

---

## Overview

This document tracks all identified implementation gaps from the spike prototypes and threat model analysis. Items are prioritized based on production readiness requirements and security criticality.

---

## Table of Contents

1. [Critical (P0) - Blocking Production](#critical-p0---blocking-production)
2. [High Priority (P1) - Security & Stability](#high-priority-p1---security--stability)
3. [Medium Priority (P2) - Operational Excellence](#medium-priority-p2---operational-excellence)
4. [Low Priority (P3) - Enhancements](#low-priority-p3---enhancements)
5. [Investigation Required](#investigation-required)
6. [Tracking & Progress](#tracking--progress)

---

## Critical (P0) - Blocking Production

**Total Effort:** ~22 hours (3 days)  
**Deadline:** Before MVP deployment  
**Owner:** TBD

### 1.1 Security: API Key Management

**Status:** ✅ IMPLEMENTED  
**Effort:** 4 hours (completed)  
**Location:** `src/config.rs` - `read_secret()` and `load_credentials()` functions

**Requirements:**
- [x] Implement Docker Secrets for all exchange API keys
- [x] Remove hardcoded/env-var API keys from all services (env vars only as fallback with warnings)
- [ ] Add secrets rotation capability (future enhancement)
- [x] Update docker-compose.yml with secrets configuration

**Implementation (in `src/config.rs`):**

The `read_secret()` function implements the Docker Secrets pattern:
1. First attempts to read from file path specified by `*_FILE` environment variable
2. Falls back to direct environment variable ONLY in development (with warning log)
3. Never logs actual secret values
4. Returns None for optional secrets

```rust
// src/config.rs - ALREADY IMPLEMENTED
fn read_secret(file_env_var: &str, fallback_env_var: Option<&str>) -> Option<String> {
    // Try reading from file path specified in environment variable
    if let Ok(secret_path) = env::var(file_env_var) {
        if let Ok(content) = fs::read_to_string(&secret_path) {
            let trimmed = content.trim().to_string();
            if !trimmed.is_empty() {
                tracing::debug!(file_path = %secret_path, "Successfully loaded secret from file");
                return Some(trimmed);
            }
        }
    }
    // Fallback to direct environment variable (development mode) with WARNING
    if let Some(env_var) = fallback_env_var {
        if let Ok(value) = env::var(env_var) {
            tracing::warn!(env_var, "Loading secret from env var (NOT RECOMMENDED for production)");
            return Some(value.trim().to_string());
        }
    }
    None
}
```

**Docker Compose Configuration:**
```yaml
# docker-compose.yml
services:
  data-factory:
    secrets:
      - binance_api_key
      - binance_api_secret
      - bybit_api_key
      - bybit_api_secret
    environment:
      BINANCE_API_KEY_FILE: /run/secrets/binance_api_key
      BINANCE_API_SECRET_FILE: /run/secrets/binance_api_secret
      
secrets:
  binance_api_key:
    external: true
  binance_api_secret:
    external: true
```

**Validation:**
- [x] Docker Secrets supported via `*_FILE` environment variables
- [x] Secrets readable only by container user (Docker manages permissions)
- [x] ApiKeyPair and KucoinCredentials implement Debug trait that redacts secrets
- [x] Warnings logged when falling back to environment variables
- [ ] Rotation test passes (manual rotation supported, automated rotation future enhancement)

**References:**
- Threat Model: T1.2 (API Key Theft)
- SPIKE_VALIDATION_REPORT.md: Priority 1 Mitigations

---

### 1.2 Concurrency: Backfill Locking (Race Condition Prevention)

**Status:** ❌ NOT IMPLEMENTED  
**Effort:** 4 hours  
**Location:** `spike-prototypes/gap-detection/` + backfill scheduler

**Requirements:**
- [ ] Implement Redis-based distributed lock for backfill jobs
- [ ] Prevent duplicate backfills of the same gap
- [ ] Add lock timeout/expiration (prevent deadlocks)
- [ ] Implement lock monitoring and metrics

**Implementation Template:**
```rust
// src/backfill/lock.rs
use redis::AsyncCommands;

pub struct BackfillLock {
    redis: redis::Client,
    ttl: Duration,
}

impl BackfillLock {
    pub async fn acquire(&self, gap_id: &str) -> Result<Option<LockGuard>> {
        let key = format!("backfill:lock:{}", gap_id);
        let lock_id = Uuid::new_v4().to_string();
        
        // SET key value NX EX ttl
        let acquired: bool = self.redis
            .get_async_connection()
            .await?
            .set_nx(&key, &lock_id, self.ttl.as_secs())
            .await?;
            
        if acquired {
            Ok(Some(LockGuard::new(self.redis.clone(), key, lock_id)))
        } else {
            Ok(None)
        }
    }
}

// Usage:
async fn backfill_gap(gap: Gap, lock: &BackfillLock) -> Result<()> {
    let _guard = lock.acquire(&gap.id).await?
        .ok_or_else(|| anyhow!("Gap already being backfilled"))?;
    
    // Perform backfill (lock released on drop)
    todo!("Implement backfill logic")
}
```

**Validation:**
- [ ] Two concurrent backfill attempts: only one succeeds
- [ ] Lock expires after TTL if holder crashes
- [ ] Metrics show lock contention rate
- [ ] Integration test passes

**References:**
- Threat Model: T2.1 (Race Condition in Gap Backfilling)
- SPIKE_VALIDATION_REPORT.md: Priority 1 Mitigations

---

### 1.3 Resilience: Circuit Breaker for Rate Limiter

**Status:** ⚠️ PARTIALLY IMPLEMENTED (has rate limiter, needs circuit breaker)  
**Effort:** 4 hours  
**Location:** `spike-prototypes/rate-limiter/src/lib.rs`

**Requirements:**
- [ ] Implement circuit breaker pattern around exchange API calls
- [ ] Track consecutive failures (threshold: 5)
- [ ] Open circuit on threshold, close after cooldown
- [ ] Emit metrics on state transitions
- [ ] Add half-open state for gradual recovery

**Implementation Template:**
```rust
// spike-prototypes/rate-limiter/src/circuit_breaker.rs
use std::sync::atomic::{AtomicU32, AtomicU8, Ordering};

#[derive(Debug, Clone, Copy, PartialEq)]
enum CircuitState {
    Closed = 0,
    Open = 1,
    HalfOpen = 2,
}

pub struct CircuitBreaker {
    failures: AtomicU32,
    state: AtomicU8,
    threshold: u32,
    timeout: Duration,
    last_failure: Arc<Mutex<Instant>>,
}

impl CircuitBreaker {
    pub fn new(threshold: u32, timeout: Duration) -> Self {
        Self {
            failures: AtomicU32::new(0),
            state: AtomicU8::new(CircuitState::Closed as u8),
            threshold,
            timeout,
            last_failure: Arc::new(Mutex::new(Instant::now())),
        }
    }
    
    pub async fn call_api<F, T>(&self, f: F) -> Result<T>
    where
        F: Future<Output = Result<T>>,
    {
        // Check circuit state
        match self.current_state() {
            CircuitState::Open => {
                // Check if timeout elapsed
                if self.should_attempt_reset() {
                    self.half_open();
                } else {
                    return Err(anyhow!("Circuit breaker is OPEN"));
                }
            }
            CircuitState::HalfOpen => {
                // Allow one request through
            }
            CircuitState::Closed => {
                // Normal operation
            }
        }
        
        // Execute the API call
        match f.await {
            Ok(result) => {
                self.on_success();
                Ok(result)
            }
            Err(e) => {
                self.on_failure();
                Err(e)
            }
        }
    }
    
    fn on_failure(&self) {
        let failures = self.failures.fetch_add(1, Ordering::SeqCst) + 1;
        *self.last_failure.lock() = Instant::now();
        
        if failures >= self.threshold {
            self.open();
            warn!("Circuit breaker OPENED after {} failures", failures);
        }
    }
    
    fn on_success(&self) {
        self.failures.store(0, Ordering::SeqCst);
        if self.current_state() == CircuitState::HalfOpen {
            self.close();
            info!("Circuit breaker CLOSED after successful recovery");
        }
    }
    
    // TODO: Implement state transition methods
    fn open(&self) { todo!() }
    fn close(&self) { todo!() }
    fn half_open(&self) { todo!() }
    fn should_attempt_reset(&self) -> bool { todo!() }
    fn current_state(&self) -> CircuitState { todo!() }
}
```

**Validation:**
- [ ] After 5 consecutive 429s, circuit opens
- [ ] Requests fail fast while open
- [ ] Circuit closes after successful test request
- [ ] Metrics show state transitions
- [ ] Load test with simulated failures passes

**References:**
- Threat Model: T5.1 (Rate Limit Exhaustion Attack)
- SPIKE_VALIDATION_REPORT.md: Priority 1 Mitigations

---

### 1.4 Resource Management: Backfill Throttling & Disk Monitoring

**Status:** ❌ NOT IMPLEMENTED  
**Effort:** 6 hours  
**Location:** `spike-prototypes/gap-detection/` + monitoring service

**Requirements:**
- [ ] Implement semaphore to limit concurrent backfills (max: 2)
- [ ] Add disk space monitoring (alert at 80%, stop backfill at 90%)
- [ ] Implement backfill batch size limits (prevent OOO overflow)
- [ ] Add QuestDB write rate monitoring

**Implementation Template:**
```rust
// src/backfill/throttle.rs
use tokio::sync::Semaphore;
use lazy_static::lazy_static;

lazy_static! {
    static ref BACKFILL_SEMAPHORE: Semaphore = Semaphore::new(2);
}

const MAX_OOO_ROWS: usize = 1_000_000;
const BACKFILL_BATCH_SIZE: usize = 10_000;

pub async fn backfill_gap(gap: Gap) -> Result<()> {
    // Acquire semaphore permit
    let _permit = BACKFILL_SEMAPHORE.acquire().await?;
    
    // Check disk space before starting
    let disk_usage = check_disk_usage().await?;
    if disk_usage > 0.90 {
        return Err(anyhow!("Disk usage {}% - aborting backfill", disk_usage * 100.0));
    }
    
    // Process in batches to prevent OOO overflow
    for batch in gap.chunk(BACKFILL_BATCH_SIZE) {
        write_batch_to_questdb(batch).await?;
        
        // Brief pause between batches to allow QuestDB to process
        tokio::time::sleep(Duration::from_millis(100)).await;
    }
    
    Ok(())
}

async fn check_disk_usage() -> Result<f64> {
    todo!("Implement using statvfs or similar")
}

// Monitoring task
async fn monitor_disk_space() {
    loop {
        let usage = check_disk_usage().await.unwrap_or(0.0);
        
        metrics::gauge!("questdb.disk_usage_percent", usage * 100.0);
        
        if usage > 0.80 {
            warn!("Disk usage high: {:.1}%", usage * 100.0);
        }
        
        tokio::time::sleep(Duration::from_secs(60)).await;
    }
}
```

**Validation:**
- [ ] Only 2 backfills run concurrently
- [ ] Backfill stops if disk >90% full
- [ ] Alert fires at 80% disk usage
- [ ] Batch size prevents QuestDB OOO overflow
- [ ] Stress test with 10 gaps passes

**References:**
- Threat Model: T2.2 (QuestDB OOO Overflow), T5.2 (Backfill Amplification)
- SPIKE_VALIDATION_REPORT.md: Priority 1 Mitigations

---

### 1.5 Observability: Prometheus Metrics Export

**Status:** ❌ NOT IMPLEMENTED  
**Effort:** 6 hours  
**Location:** All spike prototypes + main services

**Requirements:**
- [ ] Implement Prometheus metrics exporter for rate limiter
- [ ] Implement Prometheus metrics exporter for gap detector
- [ ] Export SLI metrics as defined in SLI_SLO.md
- [ ] Add /metrics HTTP endpoint to all services
- [ ] Implement authentication for metrics endpoint

**Implementation Template:**
```rust
// src/metrics/exporter.rs
use prometheus::{Encoder, TextEncoder, Registry, Counter, Gauge, Histogram};
use warp::Filter;

lazy_static! {
    pub static ref REGISTRY: Registry = Registry::new();
    
    // Rate Limiter Metrics
    pub static ref RATE_LIMIT_REQUESTS: Counter = Counter::new(
        "rate_limiter_requests_total",
        "Total rate limiter requests"
    ).unwrap();
    
    pub static ref RATE_LIMIT_ACCEPTED: Counter = Counter::new(
        "rate_limiter_accepted_total",
        "Accepted requests"
    ).unwrap();
    
    pub static ref RATE_LIMIT_REJECTED: Counter = Counter::new(
        "rate_limiter_rejected_total",
        "Rejected requests"
    ).unwrap();
    
    pub static ref RATE_LIMIT_TOKENS: Gauge = Gauge::new(
        "rate_limiter_tokens_available",
        "Available tokens"
    ).unwrap();
    
    // Gap Detection Metrics
    pub static ref GAPS_DETECTED: Counter = Counter::new(
        "gaps_detected_total",
        "Total gaps detected"
    ).unwrap();
    
    pub static ref GAP_SIZE: Histogram = Histogram::with_opts(
        prometheus::HistogramOpts::new("gap_size_trades", "Gap size distribution")
            .buckets(vec![1.0, 10.0, 100.0, 1000.0, 10000.0])
    ).unwrap();
    
    // SLI Metrics (from SLI_SLO.md)
    pub static ref DATA_COMPLETENESS: Gauge = Gauge::new(
        "data_completeness_percent",
        "Data completeness percentage"
    ).unwrap();
    
    pub static ref INGESTION_LATENCY: Histogram = Histogram::with_opts(
        prometheus::HistogramOpts::new("ingestion_latency_ms", "Ingestion latency")
            .buckets(prometheus::exponential_buckets(10.0, 2.0, 10).unwrap())
    ).unwrap();
}

pub async fn metrics_handler() -> Result<impl warp::Reply, warp::Rejection> {
    let encoder = TextEncoder::new();
    let metric_families = REGISTRY.gather();
    let mut buffer = vec![];
    encoder.encode(&metric_families, &mut buffer).unwrap();
    
    Ok(warp::reply::with_header(
        buffer,
        "Content-Type",
        encoder.format_type(),
    ))
}

// Protected metrics endpoint
pub fn metrics_route() -> impl Filter<Extract = impl warp::Reply, Error = warp::Rejection> + Clone {
    warp::path!("metrics")
        .and(warp::get())
        .and(authenticate_metrics())
        .and_then(metrics_handler)
}

async fn authenticate_metrics() -> Result<(), warp::Rejection> {
    // TODO: Implement token-based auth
    todo!()
}
```

**Validation:**
- [ ] /metrics endpoint returns Prometheus format
- [ ] All SLI metrics are exported
- [ ] Authentication works (401 for invalid token)
- [ ] Grafana can scrape metrics
- [ ] Load test shows accurate counters

**References:**
- SPIKE_VALIDATION_REPORT.md: Gaps Identified (SLI/SLO section)
- Threat Model: T4.1 (Metrics need authentication)

---

### 1.6 Observability: Grafana Dashboards

**Status:** ❌ NOT IMPLEMENTED  
**Effort:** 2 hours  
**Location:** `spike-prototypes/monitoring/dashboards/`

**Requirements:**
- [ ] Create Grafana dashboard for rate limiter metrics
- [ ] Create Grafana dashboard for gap detection
- [ ] Create SLO dashboard with error budget tracking
- [ ] Create operational overview dashboard
- [ ] Export as JSON for version control

**Dashboard Requirements:**

**Rate Limiter Dashboard:**
- Request rate (accepted vs rejected)
- Token availability over time
- Wait time distribution
- Circuit breaker state
- Per-exchange breakdown

**Gap Detection Dashboard:**
- Gaps detected (count & rate)
- Gap size distribution
- Detection latency
- Backfill queue depth
- Data completeness percentage

**SLO Dashboard:**
- Data completeness (target: 99.9%)
- Ingestion latency P99 (target: <1000ms)
- System uptime (target: 99.5%)
- Error budget remaining
- SLO violation alerts

**Files to Create:**
- [ ] `spike-prototypes/monitoring/dashboards/rate-limiter.json`
- [ ] `spike-prototypes/monitoring/dashboards/gap-detection.json`
- [ ] `spike-prototypes/monitoring/dashboards/slo.json`
- [ ] `spike-prototypes/monitoring/dashboards/overview.json`

**References:**
- SLI_SLO.md: All defined SLIs/SLOs

---

### 1.7 Observability: Alertmanager Rules

**Status:** ❌ NOT IMPLEMENTED  
**Effort:** 2 hours  
**Location:** `spike-prototypes/monitoring/alerts/`

**Requirements:**
- [ ] Create Alertmanager configuration
- [ ] Define alert rules for SLO violations
- [ ] Define alert rules for critical errors
- [ ] Configure notification channels (Slack/PagerDuty)
- [ ] Add runbook links to alerts

**Alert Rules Template:**
```yaml
# spike-prototypes/monitoring/alerts/data-factory.yml
groups:
  - name: slo_violations
    interval: 30s
    rules:
      # Data Completeness SLO
      - alert: DataCompletenessLow
        expr: data_completeness_percent < 99.9
        for: 5m
        labels:
          severity: critical
          slo: data_completeness
        annotations:
          summary: "Data completeness below SLO ({{ $value }}%)"
          description: "Data completeness is {{ $value }}%, below 99.9% SLO"
          runbook: "https://wiki/runbooks/data-completeness"
      
      # Ingestion Latency SLO
      - alert: IngestionLatencyHigh
        expr: histogram_quantile(0.99, ingestion_latency_ms) > 1000
        for: 5m
        labels:
          severity: warning
          slo: ingestion_latency
        annotations:
          summary: "P99 ingestion latency above 1000ms"
          runbook: "https://wiki/runbooks/ingestion-latency"
      
      # Rate Limit Violations
      - alert: RateLimitRejectionHigh
        expr: rate(rate_limiter_rejected_total[5m]) > 10
        for: 2m
        labels:
          severity: warning
        annotations:
          summary: "High rate limit rejection rate"
          description: "{{ $value }} requests/sec being rejected"
          runbook: "https://wiki/runbooks/rate-limit"
      
      # Gap Detection
      - alert: LargeGapDetected
        expr: gap_size_trades > 10000
        labels:
          severity: critical
        annotations:
          summary: "Large data gap detected ({{ $value }} trades)"
          runbook: "https://wiki/runbooks/gap-backfill"
      
      # Disk Space
      - alert: DiskSpaceHigh
        expr: questdb_disk_usage_percent > 80
        for: 5m
        labels:
          severity: warning
        annotations:
          summary: "QuestDB disk usage at {{ $value }}%"
          runbook: "https://wiki/runbooks/disk-space"
      
      # Circuit Breaker
      - alert: CircuitBreakerOpen
        expr: circuit_breaker_state == 1
        for: 1m
        labels:
          severity: critical
        annotations:
          summary: "Circuit breaker is OPEN for {{ $labels.exchange }}"
          runbook: "https://wiki/runbooks/circuit-breaker"
```

**Validation:**
- [ ] Alerts fire correctly in test environment
- [ ] Notifications reach configured channels
- [ ] Runbook links are accessible
- [ ] Alert fatigue is minimal (<5 false positives/day)

**References:**
- SLI_SLO.md: Alert threshold definitions
- THREAT_MODEL.md: Incident Response playbooks

---

## High Priority (P1) - Security & Stability

**Total Effort:** ~20 hours  
**Deadline:** Within 2 weeks of MVP deployment  
**Owner:** TBD

### 2.1 Rate Limiter: Max Burst Limiter

**Status:** ❌ NOT IMPLEMENTED  
**Effort:** 4 hours  
**Location:** `spike-prototypes/rate-limiter/src/lib.rs`

**Current State:** Rate limiter allows full capacity to be consumed instantly  
**Target State:** Limit burst rate to 2x refill rate to prevent token exhaustion

**Requirements:**
- [ ] Add `max_burst_rate` config parameter
- [ ] Implement burst detection logic
- [ ] Reject requests exceeding burst rate
- [ ] Add metrics for burst rate tracking

**Implementation:**
```rust
// In TokenBucketConfig
pub struct TokenBucketConfig {
    // ... existing fields
    pub max_burst_rate: Option<f64>, // requests per second
}

// In acquire()
impl TokenBucket {
    pub fn acquire(&self, weight: u32) -> Result<()> {
        // ... existing logic
        
        // Check burst rate
        if let Some(max_burst) = self.config.max_burst_rate {
            let recent_rate = self.calculate_recent_rate(Duration::from_secs(1));
            if recent_rate > max_burst {
                return Err(RateLimitError::BurstExceeded {
                    current_rate: recent_rate,
                    max_rate: max_burst,
                });
            }
        }
        
        // ... rest of logic
    }
}
```

**References:**
- SPIKE_VALIDATION_REPORT.md: Rate Limiter Gaps

---

### 2.2 Rate Limiter: Distributed State Coordination

**Status:** ❌ NOT IMPLEMENTED  
**Effort:** 8 hours  
**Location:** `spike-prototypes/rate-limiter/src/lib.rs`

**Current State:** Each instance has independent rate limit state  
**Target State:** Shared state via Redis for multi-instance deployments

**Requirements:**
- [ ] Store token count in Redis
- [ ] Use Redis for sliding window request history
- [ ] Implement optimistic locking for token updates
- [ ] Add fallback to local state if Redis unavailable

**Implementation:**
```rust
pub struct DistributedTokenBucket {
    local: TokenBucket,
    redis: Option<redis::Client>,
}

impl DistributedTokenBucket {
    pub async fn acquire(&self, weight: u32) -> Result<()> {
        if let Some(redis) = &self.redis {
            self.acquire_distributed(weight).await
        } else {
            self.local.acquire(weight)
        }
    }
    
    async fn acquire_distributed(&self, weight: u32) -> Result<()> {
        // TODO: Implement Redis-based token bucket
        // Use Lua script for atomic operations
        todo!()
    }
}
```

**References:**
- SPIKE_VALIDATION_REPORT.md: Rate Limiter Gaps

---

### 2.3 Rate Limiter: State Persistence

**Status:** ❌ NOT IMPLEMENTED  
**Effort:** 2 hours  
**Location:** `spike-prototypes/rate-limiter/src/lib.rs`

**Current State:** State lost on restart  
**Target State:** Persist to Redis, restore on startup

**Requirements:**
- [ ] Periodically save state to Redis (every 10s)
- [ ] Restore state on startup
- [ ] Handle missing/corrupted state gracefully

**References:**
- SPIKE_VALIDATION_REPORT.md: Rate Limiter Gaps

---

### 2.4 Gap Detection: Deduplication Logic

**Status:** ❌ NOT IMPLEMENTED  
**Effort:** 4 hours  
**Location:** `spike-prototypes/gap-detection/src/lib.rs`

**Current State:** Reconnect may cause duplicate writes  
**Target State:** Track last 1000 trade IDs in ring buffer

**Requirements:**
- [ ] Implement ring buffer for recent trade IDs
- [ ] Check for duplicates before writing
- [ ] Add metrics for duplicate detection rate
- [ ] Persist dedup state to Redis

**Implementation:**
```rust
use std::collections::VecDeque;

pub struct DeduplicationFilter {
    recent_ids: Mutex<VecDeque<u64>>,
    capacity: usize,
    redis: Option<redis::Client>, // For persistence
}

impl DeduplicationFilter {
    pub fn new(capacity: usize) -> Self {
        Self {
            recent_ids: Mutex::new(VecDeque::with_capacity(capacity)),
            capacity,
            redis: None,
        }
    }
    
    pub fn is_duplicate(&self, trade_id: u64) -> bool {
        let mut recent = self.recent_ids.lock();
        
        if recent.contains(&trade_id) {
            return true;
        }
        
        if recent.len() >= self.capacity {
            recent.pop_front();
        }
        recent.push_back(trade_id);
        
        false
    }
}
```

**References:**
- SPIKE_VALIDATION_REPORT.md: Gap Detection Gaps

---

### 2.5 Gap Detection: Persistent Gap Queue

**Status:** ❌ NOT IMPLEMENTED  
**Effort:** 4 hours  
**Location:** `spike-prototypes/gap-detection/src/lib.rs`

**Current State:** Detected gaps stored in memory  
**Target State:** Store in QuestDB table, survives restart

**Requirements:**
- [ ] Create `gaps` table in QuestDB
- [ ] Write detected gaps to table
- [ ] Implement gap queue reader for backfill scheduler
- [ ] Add gap lifecycle tracking (detected → backfilling → filled → verified)

**Schema:**
```sql
CREATE TABLE IF NOT EXISTS gaps (
    id SYMBOL,
    exchange SYMBOL,
    pair SYMBOL,
    start_id LONG,
    end_id LONG,
    estimated_count INT,
    detected_at TIMESTAMP,
    status SYMBOL, -- 'pending', 'backfilling', 'filled', 'verified'
    backfill_started_at TIMESTAMP,
    backfill_completed_at TIMESTAMP,
    actual_count INT
) timestamp(detected_at) PARTITION BY DAY;
```

**References:**
- SPIKE_VALIDATION_REPORT.md: Gap Detection Gaps

---

### 2.6 Gap Detection: Concurrency Improvements

**Status:** ❌ NOT IMPLEMENTED  
**Effort:** 3 hours  
**Location:** `spike-prototypes/gap-detection/src/lib.rs`

**Current State:** Single-threaded HashMap  
**Target State:** DashMap for lock-free concurrent access

**Requirements:**
- [ ] Replace HashMap with DashMap
- [ ] Remove unnecessary locks
- [ ] Benchmark performance improvement

**References:**
- SPIKE_VALIDATION_REPORT.md: Gap Detection Gaps

---

### 2.7 Security: Cross-Exchange Price Validation

**Status:** ❌ NOT IMPLEMENTED  
**Effort:** 6 hours  
**Location:** New module in data-factory

**Requirements:**
- [ ] Track prices from multiple exchanges for same pair
- [ ] Detect price deviations >5%
- [ ] Alert on suspected data poisoning
- [ ] Log validation failures to audit trail

**Implementation:**
```rust
pub struct PriceValidator {
    exchanges: Vec<String>,
    deviation_threshold: f64, // 0.05 = 5%
}

impl PriceValidator {
    pub fn validate_price(&self, pair: &str, exchange: &str, price: f64) -> Result<bool> {
        let other_prices = self.get_recent_prices(pair, exchange)?;
        
        if other_prices.is_empty() {
            // Not enough data to validate
            return Ok(true);
        }
        
        let median = calculate_median(&other_prices);
        let deviation = (price - median).abs() / median;
        
        if deviation > self.deviation_threshold {
            warn!(
                pair = pair,
                exchange = exchange,
                price = price,
                median = median,
                deviation = deviation,
                "Price deviation detected - possible data poisoning"
            );
            
            // Log to audit trail
            self.log_anomaly(pair, exchange, price, median, deviation).await?;
            
            return Ok(false);
        }
        
        Ok(true)
    }
}
```

**References:**
- THREAT_MODEL.md: T1.1 (MitM Attack)
- SPIKE_VALIDATION_REPORT.md: Attack Scenario Validation

---

## Medium Priority (P2) - Operational Excellence

**Total Effort:** ~16 hours  
**Deadline:** Within 1 month of production  
**Owner:** TBD

### 3.1 Security: Container Hardening

**Status:** ❌ NOT IMPLEMENTED  
**Effort:** 4 hours

**Requirements:**
- [ ] Run containers as non-root user
- [ ] Drop all capabilities, add only required ones
- [ ] Enable read-only root filesystem
- [ ] Scan images with Trivy/Grype
- [ ] Pin base image versions

**Template:**
```dockerfile
FROM rust:1.75-alpine AS builder
# ... build

FROM alpine:3.19
RUN addgroup -g 1001 appuser && \
    adduser -D -u 1001 -G appuser appuser

USER appuser
COPY --from=builder --chown=appuser:appuser /app /app

# Will be configured in docker-compose
```

```yaml
# docker-compose.yml
services:
  questdb:
    user: "1001:1001"
    cap_drop:
      - ALL
    cap_add:
      - NET_BIND_SERVICE
    read_only: true
    tmpfs:
      - /tmp
      - /var/run
```

**References:**
- THREAT_MODEL.md: T6.1 (Container Escape)

---

### 3.2 Security: Dependency Scanning

**Status:** ❌ NOT IMPLEMENTED  
**Effort:** 2 hours

**Requirements:**
- [ ] Add cargo-audit to CI pipeline
- [ ] Add Trivy scan to CI pipeline
- [ ] Configure automated dependency updates
- [ ] Set up security advisory notifications

**CI Integration:**
```yaml
# .github/workflows/security.yml
name: Security Scan
on: [push, pull_request]

jobs:
  security:
    runs-on: ubuntu-latest
    steps:
      - uses: actions/checkout@v4
      
      - name: Cargo Audit
        run: |
          cargo install cargo-audit
          cargo audit
      
      - name: Trivy Scan
        uses: aquasecurity/trivy-action@master
        with:
          scan-type: 'fs'
          scan-ref: '.'
          format: 'sarif'
          output: 'trivy-results.sarif'
      
      - name: Upload to GitHub Security
        uses: github/codeql-action/upload-sarif@v2
        with:
          sarif_file: 'trivy-results.sarif'
```

**References:**
- SPIKE_VALIDATION_REPORT.md: Medium-term Actions

---

### 3.3 Monitoring: Distributed Tracing

**Status:** ❌ NOT IMPLEMENTED  
**Effort:** 8 hours

**Requirements:**
- [ ] Integrate OpenTelemetry
- [ ] Add trace IDs to all log entries
- [ ] Export traces to Jaeger
- [ ] Create trace views for critical paths

**Implementation:**
```rust
use opentelemetry::{global, trace::Tracer};
use tracing_opentelemetry::OpenTelemetryLayer;

pub fn init_tracing() {
    let tracer = opentelemetry_jaeger::new_pipeline()
        .with_service_name("data-factory")
        .install_simple()
        .unwrap();
    
    let telemetry = OpenTelemetryLayer::new(tracer);
    
    tracing_subscriber::registry()
        .with(telemetry)
        .with(tracing_subscriber::fmt::layer())
        .init();
}

// Usage:
#[tracing::instrument]
async fn ingest_trade(trade: Trade) -> Result<()> {
    // Automatically traced with context propagation
    todo!()
}
```

**References:**
- SPIKE_VALIDATION_REPORT.md: Medium-term Actions

---

### 3.4 Testing: Load & Stress Tests

**Status:** ❌ NOT IMPLEMENTED  
**Effort:** 4 hours

**Requirements:**
- [ ] Create load test scenarios (10x expected traffic)
- [ ] Test QuestDB OOO behavior under load
- [ ] Test Redis under load
- [ ] Measure P99 latency under stress
- [ ] Identify breaking points

**Tools:**
- k6 for HTTP load testing
- Custom Rust harness for WebSocket load

**References:**
- SPIKE_VALIDATION_REPORT.md: Long-term Actions

---

## Low Priority (P3) - Enhancements

**Total Effort:** ~12 hours  
**Deadline:** Post-production, as needed

### 4.1 Rate Limiter: Adaptive Rate Adjustment

**Effort:** 6 hours

**Description:** Automatically adjust refill rate based on observed server behavior

---

### 4.2 Gap Detection: ML-based Anomaly Detection

**Effort:** 8 hours

**Description:** Use statistical models to predict gaps before they occur

---

### 4.3 Observability: Custom Grafana Plugins

**Effort:** 4 hours

**Description:** Build custom visualizations for trading-specific metrics

---

## Investigation Required

### I.1 Test Hang: `test_sliding_window`

**Status:** ⚠️ REQUIRES INVESTIGATION  
**Effort:** 2-4 hours  
**Location:** `spike-prototypes/rate-limiter/src/lib.rs:515`

**Issue:** Test hangs indefinitely  
**Symptoms:**
- Works in isolation sometimes
- Hangs when run with full test suite
- No error message, just infinite wait

**Hypotheses:**
1. Parking_lot Mutex deadlock
2. Timing race condition in test
3. Test framework interaction issue

**Investigation Steps:**
- [ ] Add extensive tracing to test
- [ ] Try with different test frameworks
- [ ] Test with std::sync::Mutex instead of parking_lot
- [ ] Rewrite as async test with timeout
- [ ] Check for interaction with other tests

**Temporary Mitigation:** Test is currently `#[ignore]`'d

**References:**
- BUGFIXES.md: Section 8

---

## Tracking & Progress

### Completion Metrics

| Priority | Total Items | Completed | In Progress | Not Started | Completion % |
|----------|-------------|-----------|-------------|-------------|--------------|
| P0       | 7           | 0         | 0           | 7           | 0%           |
| P1       | 7           | 0         | 0           | 7           | 0%           |
| P2       | 4           | 0         | 0           | 4           | 0%           |
| P3       | 3           | 0         | 0           | 3           | 0%           |
| **Total**| **21**      | **0**     | **0**       | **21**      | **0%**       |

### Effort Summary

| Priority | Estimated Hours | % of Total |
|----------|-----------------|------------|
| P0       | 22              | 38%        |
| P1       | 20              | 34%        |
| P2       | 16              | 27%        |
| P3       | 12              | 21%        |
| **Total**| **70**          | **100%**   |

### Timeline

```
Week 1-2 (P0 - Critical):
├─ Day 1-2: API Key Management + Backfill Locking
├─ Day 3: Circuit Breaker
├─ Day 4-5: Backfill Throttling + Disk Monitoring
└─ Day 6-7: Prometheus + Grafana + Alerting

Week 3-4 (P1 - High Priority):
├─ Day 8-9: Rate Limiter Enhancements
├─ Day 10-11: Gap Detection Enhancements
└─ Day 12-14: Security (Price Validation, etc.)

Week 5-6 (P2 - Medium Priority):
├─ Container Hardening
├─ Dependency Scanning
├─ Distributed Tracing
└─ Load Testing

Post-Production (P3 - Low Priority):
└─ As needed based on operational feedback
```

---

## Appendix A: Quick Reference

### File Locations for Implementation

```
fks/
├── spike-prototypes/
│   ├── rate-limiter/
│   │   ├── src/
│   │   │   ├── lib.rs                    # P0.3, P1.1, P1.2, P1.3
│   │   │   ├── circuit_breaker.rs        # P0.3 (NEW FILE)
│   │   │   └── distributed.rs            # P1.2 (NEW FILE)
│   │   └── tests/
│   │       └── integration_test.rs       # I.1
│   ├── gap-detection/
│   │   ├── src/
│   │   │   ├── lib.rs                    # P1.4, P1.5, P1.6
│   │   │   ├── deduplication.rs          # P1.4 (NEW FILE)
│   │   │   └── persistent_queue.rs       # P1.5 (NEW FILE)
│   ├── monitoring/
│   │   ├── dashboards/                   # P0.6 (NEW DIR)
│   │   │   ├── rate-limiter.json
│   │   │   ├── gap-detection.json
│   │   │   ├── slo.json
│   │   │   └── overview.json
│   │   └── alerts/                       # P0.7 (NEW DIR)
│   │       └── data-factory.yml
│   └── documentation/
│       └── IMPLEMENTATION_CHECKLIST.md   # Track progress
├── src/
│   └── janus/
│       └── services/
│           └── data-factory/
│               ├── src/
│               │   ├── config.rs         # P0.1
│               │   ├── metrics/
│               │   │   └── exporter.rs   # P0.5 (NEW FILE)
│               │   ├── backfill/
│               │   │   ├── lock.rs       # P0.2 (NEW FILE)
│               │   │   └── throttle.rs   # P0.4 (NEW FILE)
│               │   └── validation/
│               │       └── price.rs      # P1.7 (NEW FILE)
│               └── Dockerfile            # P2.1
└── docker-compose.yml                     # P0.1, P2.1
```

---

## Appendix B: Dependencies to Add

```toml
# Cargo.toml additions

[dependencies]
# P0.2: Backfill locking
redis = { version = "0.24", features = ["tokio-comp", "connection-manager"] }
uuid = { version = "1.6", features = ["v4"] }

# P0.5: Metrics
prometheus = "0.13"
warp = "0.3" # For /metrics endpoint

# P1.2: Distributed rate limiting
# (redis already added)

# P1.4: Deduplication
# (no new deps, uses std)

# P2.3: Distributed tracing
opentelemetry = "0.21"
opentelemetry-jaeger = "0.20"
tracing-opentelemetry = "0.22"

# P2.4: Load testing
# Use k6 externally, no Rust deps
```

---

## Document History

| Version | Date | Author | Changes |
|---------|------|--------|---------|
| 1.0     | 2024 | AI     | Initial creation from spike validation report |

---

**Status Legend:**
- ✅ Complete
- ⚠️ Partially implemented / needs investigation
- ❌ Not implemented
- 🔄 In progress

**Priority Legend:**
- P0: Critical - Blocks production
- P1: High - Security/stability risk
- P2: Medium - Operational excellence
- P3: Low - Nice to have
