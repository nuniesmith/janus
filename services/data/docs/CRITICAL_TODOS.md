# Critical TODOs - Executive Summary
## Spike Prototypes → Production Readiness

**Last Updated:** 2024  
**Status:** 🔴 BLOCKING PRODUCTION  
**Estimated Effort:** 22 hours (3 days)  
**Owner:** Engineering Team

---

## 🚨 CRITICAL PATH (Must complete before production)

### Overview
The spike prototypes have validated our approach, but **7 critical items** must be implemented before production deployment. These address security vulnerabilities, data integrity risks, and operational blindspots.

---

## The 7 Critical Items

### 1. 🔐 API Key Security (4 hours)
**Current Risk:** API keys in environment variables → container escape = total compromise  
**Required:** Docker Secrets integration

```bash
# Block production until:
✓ All API keys moved to Docker Secrets
✓ No keys in env vars or config files  
✓ Secrets rotation tested
```

**Files to modify:**
- `docker-compose.yml`
- `src/janus/services/data-factory/src/config.rs`

**Validation:** `grep -r "API_KEY" src/` returns no matches

---

### 2. 🔒 Backfill Race Condition Prevention (4 hours)
**Current Risk:** Two instances backfill same gap → duplicate data + wasted resources  
**Required:** Redis-based distributed locking

```rust
// Implement:
pub async fn backfill_gap(gap: Gap, lock: &BackfillLock) -> Result<()> {
    let _guard = lock.acquire(&gap.id).await?
        .ok_or_else(|| anyhow!("Already backfilling"))?;
    
    // Safe to backfill - lock released on drop
}
```

**Files to create:**
- `src/janus/services/data-factory/src/backfill/lock.rs`

**Validation:** Concurrent backfill test - only one succeeds

---

### 3. ⚡ Circuit Breaker (4 hours)
**Current Risk:** Rate limit hit → keep retrying → IP ban → 24hr outage  
**Required:** Fail-fast pattern with auto-recovery

```rust
// After 5 consecutive 429s, circuit opens
// Requests fail immediately (no API calls)
// Test recovery after cooldown period
```

**Files to create:**
- `spike-prototypes/rate-limiter/src/circuit_breaker.rs`

**Validation:** Simulate 429s → circuit opens → fast fail → recovers

---

### 4. 💾 Backfill Resource Protection (6 hours)
**Current Risk:** Unlimited backfills → QuestDB OOO overflow OR disk full → crash  
**Required:** Semaphore + disk monitoring

```rust
// Max 2 concurrent backfills
static BACKFILL_SEMAPHORE: Semaphore = Semaphore::new(2);

// Stop backfill if disk > 90% full
// Alert at 80%
```

**Files to create:**
- `src/janus/services/data-factory/src/backfill/throttle.rs`

**Validation:** 
- ✓ Only 2 backfills run concurrently
- ✓ Stops at 90% disk usage

---

### 5. 📊 Prometheus Metrics Export (6 hours)
**Current Risk:** Zero visibility into system health in production  
**Required:** /metrics endpoint with SLI tracking

Must export:
- `data_completeness_percent` (SLO: 99.9%)
- `ingestion_latency_ms` (P99 < 1000ms)
- `rate_limiter_rejected_total`
- `gaps_detected_total`
- `circuit_breaker_state`

**Files to create:**
- `src/janus/services/data-factory/src/metrics/exporter.rs`

**Validation:** 
```bash
curl http://localhost:9090/metrics | grep data_completeness
```

---

### 6. 📈 Grafana Dashboards (2 hours)
**Current Risk:** Metrics exist but not visualized → can't detect issues  
**Required:** 4 dashboards (JSON export to git)

**Dashboards:**
1. **Rate Limiter** - Request rate, token availability, circuit breaker state
2. **Gap Detection** - Gaps found, backfill queue, data completeness %
3. **SLO Dashboard** - Error budget tracking, SLO violations
4. **Operations Overview** - System health at-a-glance

**Files to create:**
- `spike-prototypes/monitoring/dashboards/*.json` (4 files)

**Validation:** Import to Grafana, verify data flows

---

### 7. 🚨 Alerting Rules (2 hours)
**Current Risk:** System degrading but no one notified  
**Required:** Alertmanager config with runbooks

**Critical Alerts:**
- Data completeness < 99.9% for 5 minutes
- P99 latency > 1000ms for 5 minutes  
- Circuit breaker OPEN for 1 minute
- Disk usage > 80% for 5 minutes
- Large gap detected (>10,000 trades)

**Files to create:**
- `spike-prototypes/monitoring/alerts/data-factory.yml`

**Validation:** Trigger test alert → notification received

---

## Implementation Timeline

```
Day 1 (8 hours):
  Morning:   #1 API Key Security (4h)
  Afternoon: #2 Backfill Locking (4h)

Day 2 (8 hours):
  Morning:   #3 Circuit Breaker (4h)
  Afternoon: #4 Backfill Throttling (4h)

Day 3 (6 hours):
  Morning:   #5 Prometheus Metrics (6h)
  Afternoon: #6 Grafana + #7 Alerting (2h + 2h)
```

**Total:** 22 hours = 3 business days with one person full-time

---

## Acceptance Criteria

Before deploying to production, ALL of these must pass:

- [ ] **Security Audit**: No API keys in code/env, only Docker Secrets
- [ ] **Concurrency Test**: Two pods try to backfill same gap → one succeeds, one fails gracefully
- [ ] **Rate Limit Test**: Simulate 429s → circuit opens → 0 requests sent → recovers after cooldown
- [ ] **Resource Test**: 10 concurrent backfills attempted → only 2 run, 90% disk stops backfill
- [ ] **Observability Test**: Prometheus scrapes metrics, Grafana shows dashboards, alerts fire correctly
- [ ] **Integration Test**: End-to-end with all components → metrics accurate, no data loss

---

## Risk Matrix (If NOT Implemented)

| Item | Impact if Skipped | Probability | Severity |
|------|------------------|-------------|----------|
| #1 API Keys | API keys leaked → financial loss | Medium | 🔴 Critical |
| #2 Locking | Duplicate data → corrupted analytics | High | 🟠 High |
| #3 Circuit Breaker | IP ban → 24hr outage | Medium | 🔴 Critical |
| #4 Throttling | Disk full → system crash | Medium | 🟠 High |
| #5 Metrics | Can't debug production issues | High | 🟠 High |
| #6 Dashboards | Slow incident response | High | 🟡 Medium |
| #7 Alerts | Incidents go unnoticed | High | 🔴 Critical |

**Combined Risk if None Implemented:** 🔴 **UNACCEPTABLE FOR PRODUCTION**

---

## Go/No-Go Decision

### ✅ GO if:
- All 7 items implemented ✓
- Acceptance criteria pass ✓
- Security review approved ✓
- Load test passed ✓

### ❌ NO-GO if:
- Any P0 item incomplete
- Security concerns unresolved
- No observability (blind deployment)

---

## Next Steps

1. **Assign Owner**: Designate engineer responsible for each item
2. **Create Issues**: Track in GitHub/Jira with links to this doc
3. **Daily Standup**: Review progress, blockers
4. **Integration Test**: Run full suite before production deploy
5. **Production Deploy**: Only after ALL items complete

---

## Additional Context

### Why These 7?

These were identified through:
- **Spike validation** (SPIKE_VALIDATION_REPORT.md)
- **Threat modeling** (THREAT_MODEL.md - STRIDE analysis)
- **SLO definition** (SLI_SLO.md - operational requirements)

**Production Readiness:** Currently 85% → Need 100%

The spike prototypes prove the **approach works**. These 7 items make it **production-safe**.

---

## References

- 📄 **Full TODO List**: `spike-prototypes/TODO_IMPLEMENTATION_PLAN.md` (70 hours total, P0-P3)
- 📊 **Spike Validation**: `spike-prototypes/documentation/SPIKE_VALIDATION_REPORT.md`
- 🛡️ **Threat Model**: `spike-prototypes/documentation/THREAT_MODEL.md`
- 📈 **SLI/SLO**: `spike-prototypes/documentation/SLI_SLO.md`
- 🐛 **Recent Fixes**: `spike-prototypes/rate-limiter/BUGFIXES.md`

---

## Questions?

Contact the team lead or review the comprehensive TODO plan for implementation details, code templates, and validation steps.

**Remember:** These 7 items are not optional. They are the difference between a working prototype and a production-ready system.

---

**Status Legend:**
- 🔴 Critical - Blocks production
- 🟠 High - Significant risk
- 🟡 Medium - Important but not blocking
- 🟢 Low - Nice to have

**Current Production Readiness:** 85% → **Target:** 100%
**Time to Production Ready:** 3 days (22 hours)