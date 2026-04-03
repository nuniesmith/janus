# Backfill Lock Quick Start Guide

## Installation

The backfill locking module is already included in `janus-data-factory`. No additional dependencies needed.

## Basic Usage

```rust
use janus_data_factory::backfill::{BackfillLock, LockConfig, LockMetrics};
use prometheus::Registry;
use std::sync::Arc;

// Initialize Redis client
let redis = redis::Client::open("redis://127.0.0.1:6379/")?;

// Create metrics registry
let registry = Registry::new();
let metrics = Arc::new(LockMetrics::new(&registry)?);

// Create lock manager with default config
let config = LockConfig::default();
let lock = BackfillLock::new(redis, config, metrics);

// Try to acquire lock for a gap
let gap_id = "gap_12345";
match lock.acquire(gap_id).await? {
    Some(guard) => {
        println!("Lock acquired! Processing gap {}", gap_id);
        
        // Perform backfill work here
        backfill_gap(gap_id).await?;
        
        // Lock automatically released when guard drops
    }
    None => {
        println!("Gap {} is already being processed by another instance", gap_id);
    }
}
```

## Running Tests

```bash
# Run all backfill tests
cargo test backfill

# Run only unit tests
cargo test --lib backfill

# Run only integration tests  
cargo test --test backfill_lock_integration

# Run specific test with output
cargo test test_realistic_backfill_scenario -- --nocapture
```

## Requirements

- **Redis:** Must be running on localhost:6379 (or configure connection string)
- Tests will skip gracefully if Redis is not available

## Configuration

```rust
use std::time::Duration;

let mut config = LockConfig::default();
config.ttl = Duration::from_secs(300);              // 5 minutes (default)
config.key_prefix = "backfill:lock:".to_string();   // Redis key prefix
config.extension_interval = Duration::from_secs(150); // Extend every 2.5 minutes
```

## Metrics

Access via Prometheus:

- `backfill_locks_acquired_total` - Total successful acquisitions
- `backfill_locks_contended_total` - Total contentions (already locked)
- `backfill_locks_released_total` - Total releases
- `backfill_locks_extended_total` - Total TTL extensions
- `backfill_locks_held` - Current number of held locks
- `backfill_lock_release_errors_total` - Errors during release

## See Also

- **P0.2_BACKFILL_LOCK_SUMMARY.md** - Complete implementation details
- **IMPLEMENTATION_PROGRESS.md** - Overall progress tracker
- **TODO_IMPLEMENTATION_PLAN.md** - Full implementation plan
