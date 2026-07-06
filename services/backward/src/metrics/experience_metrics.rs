//! # Experience Ingest Metrics Collector
//!
//! Prometheus counters for the experience-pipeline intake worker
//! (spool → `handle_ingest` → Qdrant). Exported on the same global
//! registry that backs the backward service's `/metrics` endpoint
//! (see [`crate::http`], which encodes `cns::metrics::METRICS_REGISTRY`).

use prometheus::{IntCounter, Opts, Registry};
use std::sync::LazyLock;
use tracing::warn;

/// Global collector instance, registered once on the CNS metrics registry
/// served by the backward `/metrics` endpoint.
static GLOBAL_EXPERIENCE_METRICS: LazyLock<ExperienceMetricsCollector> = LazyLock::new(|| {
    let collector = ExperienceMetricsCollector::detached();
    let registry = cns::metrics::CNSMetrics::registry();
    if let Err(e) = collector.register(&registry.registry) {
        // Registration can only fail on duplicate registration, which cannot
        // happen for a LazyLock-initialised singleton — but never panic in a
        // metrics path.
        warn!(error = %e, "Failed to register experience ingest metrics");
    }
    collector
});

/// Prometheus counters for experience-batch intake (design §6).
#[derive(Clone)]
pub struct ExperienceMetricsCollector {
    /// Batches successfully ingested (spool file read + persisted).
    pub batches_ingested: IntCounter,
    /// Rows successfully validated across all ingested batches.
    pub rows_ingested: IntCounter,
    /// Rows that failed validation and were skipped.
    pub rows_skipped: IntCounter,
    /// Batches that exhausted their retry budget and were parked in `failed/`.
    pub batches_failed: IntCounter,
    /// Batches recovered by the spool sweep (startup / periodic / failed-retry)
    /// rather than the Redis queue.
    pub sweep_recovered: IntCounter,
    /// Batches re-queued to Redis after a transient ingest failure.
    pub batches_requeued: IntCounter,
}

impl ExperienceMetricsCollector {
    /// Create a collector without registering it anywhere (tests, or callers
    /// that want to register on a custom registry via [`Self::register`]).
    pub fn detached() -> Self {
        let counter = |name: &str, help: &str| {
            IntCounter::with_opts(Opts::new(name, help)).expect("valid counter opts")
        };

        Self {
            batches_ingested: counter(
                "janus_experience_batches_ingested_total",
                "Total experience batches successfully ingested from the spool",
            ),
            rows_ingested: counter(
                "janus_experience_rows_ingested_total",
                "Total experience rows validated during ingest",
            ),
            rows_skipped: counter(
                "janus_experience_rows_skipped_total",
                "Total experience rows that failed validation and were skipped",
            ),
            batches_failed: counter(
                "janus_experience_batches_failed_total",
                "Total experience batches parked in failed/ after exhausting retries",
            ),
            sweep_recovered: counter(
                "janus_experience_sweep_recovered_total",
                "Total experience batches recovered by the spool sweep",
            ),
            batches_requeued: counter(
                "janus_experience_batches_requeued_total",
                "Total experience batches re-queued to Redis after a transient failure",
            ),
        }
    }

    /// Register all counters on the given registry.
    pub fn register(&self, registry: &Registry) -> Result<(), prometheus::Error> {
        registry.register(Box::new(self.batches_ingested.clone()))?;
        registry.register(Box::new(self.rows_ingested.clone()))?;
        registry.register(Box::new(self.rows_skipped.clone()))?;
        registry.register(Box::new(self.batches_failed.clone()))?;
        registry.register(Box::new(self.sweep_recovered.clone()))?;
        registry.register(Box::new(self.batches_requeued.clone()))?;
        Ok(())
    }

    /// The process-wide collector, registered on the registry served by the
    /// backward `/metrics` endpoint.
    pub fn global() -> &'static Self {
        &GLOBAL_EXPERIENCE_METRICS
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_detached_collector_counts() {
        let m = ExperienceMetricsCollector::detached();
        m.batches_ingested.inc();
        m.rows_ingested.inc_by(42);
        assert_eq!(m.batches_ingested.get(), 1);
        assert_eq!(m.rows_ingested.get(), 42);
        assert_eq!(m.batches_failed.get(), 0);
    }

    #[test]
    fn test_register_on_custom_registry() {
        let m = ExperienceMetricsCollector::detached();
        let registry = Registry::new();
        m.register(&registry).unwrap();
        m.sweep_recovered.inc();
        let families = registry.gather();
        assert!(
            families
                .iter()
                .any(|f| f.name() == "janus_experience_sweep_recovered_total")
        );
    }

    #[test]
    fn test_global_registered_on_cns_registry() {
        let m = ExperienceMetricsCollector::global();
        m.batches_requeued.inc();
        let text = cns::metrics::CNSMetrics::encode_text().unwrap();
        assert!(text.contains("janus_experience_batches_requeued_total"));
    }
}
