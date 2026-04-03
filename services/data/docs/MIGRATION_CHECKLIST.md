# Migration Checklist - Spike Prototypes Integration
## Verification & Next Steps

---

## ✅ Completed

- [x] Rate limiter crate copied to `src/janus/crates/rate-limiter/`
- [x] Gap detection crate copied to `src/janus/crates/gap-detection/`
- [x] Package names updated (`janus-rate-limiter`, `janus-gap-detection`)
- [x] Workspace `Cargo.toml` updated with new crates
- [x] Data Factory `Cargo.toml` updated with dependencies
- [x] Import statements updated in examples
- [x] Documentation migrated to `docs/` directory
- [x] Build verification successful (all crates compile)
- [x] Integration guide created
- [x] Migration summary created

---

## 🔍 Verification Commands

Run these to confirm everything works:

```bash
cd src/janus

# Check individual crates
cargo check -p janus-rate-limiter
cargo check -p janus-gap-detection
cargo check -p janus-data-factory

# Run tests
cargo test -p janus-rate-limiter
cargo test -p janus-gap-detection

# Build entire workspace
cargo build --workspace

# Run examples
cd crates/rate-limiter && cargo run --example exchange_actor --features examples
cd ../gap-detection && cargo run --example real_world_simulation
```

---

## 📚 Documentation Locations

All spike documentation is now in `src/janus/services/data-factory/docs/`:

- ✅ `INTEGRATION_GUIDE.md` - How to use the crates
- ✅ `SPIKE_INTEGRATION_SUMMARY.md` - This migration summary
- ✅ `CRITICAL_TODOS.md` - 7 blocking items (22 hours)
- ✅ `TODO_IMPLEMENTATION_PLAN.md` - Complete implementation plan (70 hours)
- ✅ `THREAT_MODEL.md` - Security analysis (STRIDE)
- ✅ `SLI_SLO.md` - Operational metrics
- ✅ `SPIKE_VALIDATION_REPORT.md` - Validation evidence

---

## 🚨 NEXT STEPS - CRITICAL (Before Production)

### 7 Blocking Items (22 hours total)

See `docs/CRITICAL_TODOS.md` for full details.

1. **API Key Security** (4h)
   - Implement Docker Secrets
   - Remove hardcoded keys from env vars

2. **Backfill Locking** (4h)
   - Redis distributed locks
   - Prevent duplicate backfills

3. **Circuit Breaker** (4h)
   - Fail-fast on rate limit exhaustion
   - Auto-recovery after cooldown

4. **Backfill Throttling** (6h)
   - Max 2 concurrent backfills
   - Disk space monitoring (alert at 80%, stop at 90%)

5. **Prometheus Metrics** (6h)
   - /metrics endpoint
   - Export all SLIs

6. **Grafana Dashboards** (2h)
   - 4 dashboards (rate-limiter, gap-detection, SLO, overview)

7. **Alerting Rules** (2h)
   - Alertmanager config
   - Notification channels

**Timeline:** 3 business days (1 engineer full-time)

---

## 📝 Integration Workflow

### Step 1: Update Your Code

```rust
// Old imports
use rate_limiter_spike::{TokenBucket, TokenBucketConfig};
use gap_detection_spike::{GapDetectionManager, Trade};

// New imports
use janus_rate_limiter::{TokenBucket, TokenBucketConfig};
use janus_gap_detection::{GapDetectionManager, Trade};
```

### Step 2: Add to Your Actors

See `docs/INTEGRATION_GUIDE.md` for complete examples:
- WebSocket actor integration
- REST API client integration
- Metrics collection
- Gap detection workflows

### Step 3: Configure

```toml
# config.toml
[rate_limiter.exchanges.binance]
capacity = 6000
refill_rate = 100.0
safety_margin = 0.9

[gap_detection]
heartbeat_timeout_secs = 30
enable_sequence_check = true
```

---

## ✅ Acceptance Criteria

Before production deploy, verify:

- [ ] All 7 P0 items implemented
- [ ] Security audit passed (no API keys in code)
- [ ] Concurrency test passed (backfill locking works)
- [ ] Rate limit test passed (circuit breaker works)
- [ ] Resource test passed (semaphore + disk monitoring works)
- [ ] Observability test passed (Prometheus scrapes, Grafana shows data, alerts fire)
- [ ] Integration test passed (end-to-end with real data)
- [ ] Load test passed (10x expected traffic)

---

## 📞 Support

Questions? Check:
1. `docs/INTEGRATION_GUIDE.md` - Usage examples
2. `crates/rate-limiter/README.md` - Rate limiter docs
3. `crates/gap-detection/README.md` - Gap detection docs
4. `docs/TODO_IMPLEMENTATION_PLAN.md` - Implementation details

---

**Status:** ✅ Migration Complete  
**Production Ready:** 85% (need 7 P0 items)  
**Estimated Time to Production:** 3 days + testing
