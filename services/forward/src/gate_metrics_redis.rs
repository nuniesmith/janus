//! Syncs the execution gate ↔ Redis: mirrors the gate's block/eval counters for
//! Grafana **and** persists the consecutive-loss breaker state across restarts
//! (Track C observability + durability).
//!
//! Best-effort + opt-in (`JANUS_GATE_METRICS_REDIS=1`): a periodic task, spawned
//! at forward startup, snapshots the shared [`ForwardGate`] and writes, under
//! keys wire-compatible with the Python gate's scheme:
//!
//! - `{prefix}gate_evals:{asset}`            — total evaluations
//! - `{prefix}gate_blocks:{asset}:{reason}`  — per-reason block count
//! - `{prefix}gate_breaker_state`            — JSON breaker snapshot (durability)
//!
//! On startup the task restores the persisted breaker state — a tripped breaker
//! should survive a restart, not silently reset. The pure key/value mapping
//! ([`redis_kv`]) is unit-tested; the Redis I/O is a thin wrapper that never
//! blocks trading — if Redis is unreachable the task simply doesn't start
//! (logged once), and per-SET errors are logged + skipped.

use std::sync::Arc;
use std::time::Duration;

use redis::AsyncCommands;
use tokio::sync::RwLock;
use tokio::task::JoinHandle;
use tracing::{info, warn};

use crate::gate_integration::{BreakerSnapshotEntry, ForwardGate, GateMetricsSnapshot};

/// Default Redis key prefix.
const DEFAULT_PREFIX: &str = "janus:";
/// Default flush interval.
const DEFAULT_INTERVAL_SECS: u64 = 15;

/// Configuration for the gate-metrics Redis exporter.
#[derive(Debug, Clone)]
pub struct GateMetricsRedisConfig {
    /// Whether the exporter runs (`JANUS_GATE_METRICS_REDIS`). Off by default.
    pub enabled: bool,
    /// Redis connection URL (`REDIS_URL`).
    pub redis_url: String,
    /// Key prefix (`JANUS_GATE_METRICS_PREFIX`).
    pub prefix: String,
    /// Flush interval (`JANUS_GATE_METRICS_INTERVAL_SECS`).
    pub interval: Duration,
}

impl Default for GateMetricsRedisConfig {
    fn default() -> Self {
        Self {
            enabled: false,
            redis_url: "redis://127.0.0.1:6379".to_string(),
            prefix: DEFAULT_PREFIX.to_string(),
            interval: Duration::from_secs(DEFAULT_INTERVAL_SECS),
        }
    }
}

impl GateMetricsRedisConfig {
    /// Build from the environment.
    pub fn from_env() -> Self {
        let enabled = std::env::var("JANUS_GATE_METRICS_REDIS")
            .map(|v| v == "1" || v.eq_ignore_ascii_case("true"))
            .unwrap_or(false);
        let redis_url =
            std::env::var("REDIS_URL").unwrap_or_else(|_| "redis://127.0.0.1:6379".to_string());
        let prefix = std::env::var("JANUS_GATE_METRICS_PREFIX")
            .unwrap_or_else(|_| DEFAULT_PREFIX.to_string());
        let interval_secs: u64 = std::env::var("JANUS_GATE_METRICS_INTERVAL_SECS")
            .ok()
            .and_then(|v| v.parse().ok())
            .unwrap_or(DEFAULT_INTERVAL_SECS);
        Self {
            enabled,
            redis_url,
            prefix,
            interval: Duration::from_secs(interval_secs.max(1)),
        }
    }
}

/// Map a metrics snapshot to `(redis_key, value)` pairs. Pure — unit-tested.
pub fn redis_kv(snapshot: &GateMetricsSnapshot, prefix: &str) -> Vec<(String, u64)> {
    let mut kv = Vec::with_capacity(snapshot.evals.len() + snapshot.blocks.len());
    for (asset, evals) in &snapshot.evals {
        kv.push((format!("{prefix}gate_evals:{asset}"), *evals));
    }
    for (asset, reason, count) in &snapshot.blocks {
        kv.push((format!("{prefix}gate_blocks:{asset}:{reason}"), *count));
    }
    kv
}

/// Spawn the periodic exporter.
///
/// Returns `None` (and starts nothing) when disabled or when the initial Redis
/// connect fails — observability is best-effort and never blocks trading.
pub async fn spawn(
    gate: Arc<RwLock<ForwardGate>>,
    config: GateMetricsRedisConfig,
) -> Option<JoinHandle<()>> {
    if !config.enabled {
        info!(
            "gate metrics → Redis: disabled (set JANUS_GATE_METRICS_REDIS=1 to mirror gate counters)"
        );
        return None;
    }

    let client = match redis::Client::open(config.redis_url.as_str()) {
        Ok(c) => c,
        Err(e) => {
            warn!(
                "gate metrics → Redis: invalid URL {} ({e}); exporter not started",
                config.redis_url
            );
            return None;
        }
    };

    let mut conn = match tokio::time::timeout(
        Duration::from_secs(5),
        redis::aio::ConnectionManager::new(client),
    )
    .await
    {
        Ok(Ok(c)) => c,
        _ => {
            warn!(
                "gate metrics → Redis: could not connect to {}; exporter not started",
                config.redis_url
            );
            return None;
        }
    };

    let breaker_key = format!("{}gate_breaker_state", config.prefix);

    // Restore persisted breaker state before the first flush — a tripped breaker
    // should survive a restart, not silently reset.
    if let Ok(Some(json)) = conn.get::<_, Option<String>>(&breaker_key).await
        && let Ok(entries) = serde_json::from_str::<Vec<BreakerSnapshotEntry>>(&json)
        && !entries.is_empty()
    {
        let n = entries.len();
        gate.write().await.import_breaker(entries);
        info!("gate → Redis: restored breaker state for {n} asset(s)");
    }

    info!(
        "✅ gate → Redis sync started (prefix={}, every {:?})",
        config.prefix, config.interval
    );

    Some(tokio::spawn(async move {
        loop {
            tokio::time::sleep(config.interval).await;
            let (counters, breaker) = {
                let g = gate.read().await;
                (
                    redis_kv(&g.metrics_snapshot(), &config.prefix),
                    g.export_breaker(),
                )
            };
            for (key, val) in counters {
                let res: redis::RedisResult<()> = conn.set(&key, val).await;
                if let Err(e) = res {
                    warn!("gate → Redis: SET {key} failed: {e}");
                }
            }
            if let Ok(json) = serde_json::to_string(&breaker) {
                let res: redis::RedisResult<()> = conn.set(&breaker_key, json).await;
                if let Err(e) = res {
                    warn!("gate → Redis: persist breaker state failed: {e}");
                }
            }
        }
    }))
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn redis_kv_maps_snapshot_to_wire_keys() {
        let snap = GateMetricsSnapshot {
            evals: vec![("btc".to_string(), 7), ("eth".to_string(), 3)],
            blocks: vec![
                ("btc".to_string(), "block_risk".to_string(), 2),
                ("btc".to_string(), "block_correlation".to_string(), 1),
            ],
        };
        let kv = redis_kv(&snap, "janus:");
        assert!(kv.contains(&("janus:gate_evals:btc".to_string(), 7)));
        assert!(kv.contains(&("janus:gate_evals:eth".to_string(), 3)));
        assert!(kv.contains(&("janus:gate_blocks:btc:block_risk".to_string(), 2)));
        assert!(kv.contains(&("janus:gate_blocks:btc:block_correlation".to_string(), 1)));
        assert_eq!(kv.len(), 4);
    }

    #[test]
    fn redis_kv_empty_snapshot_is_empty() {
        assert!(redis_kv(&GateMetricsSnapshot::default(), "janus:").is_empty());
    }

    #[test]
    fn config_from_env_defaults_disabled() {
        // No env set in this test process ⇒ disabled with sane defaults.
        let c = GateMetricsRedisConfig::default();
        assert!(!c.enabled);
        assert_eq!(c.prefix, "janus:");
    }
}
