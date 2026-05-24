//! JanusAI session metrics push client (JFLOW-A).
//!
//! Forward signal pipeline rolls up per-cycle metrics (signal counts, win
//! rate, average confidence, latency) and pushes them to JanusAI via
//! `POST {JANUS_AI_URL}/api/janus-ai/sessions/{session_id}/metrics`.
//!
//! The client is a no-op when `JANUS_AI_URL` is unset so the binary can boot
//! and the signal pipeline can run without a JanusAI sidecar (dev / backtest
//! / standalone compose).

use serde::{Deserialize, Serialize};
use std::time::Duration;
use tracing::{debug, warn};

/// Per-session metrics snapshot pushed to JanusAI.
///
/// The shape is intentionally narrow — the JanusAI service owns the wider
/// schema and is free to ignore unknown fields. Add fields here only when
/// JanusAI is ready to consume them.
#[derive(Debug, Clone, Default, Serialize, Deserialize)]
pub struct SessionMetrics {
    /// Signals generated since the last push.
    pub signals_generated: u64,
    /// Signals filtered out by gates / risk before publish.
    pub signals_filtered: u64,
    /// Mean confidence of generated signals (0.0..1.0).
    pub avg_confidence: f64,
    /// P50 end-to-end signal generation latency in microseconds.
    pub p50_latency_us: u64,
    /// P99 end-to-end signal generation latency in microseconds.
    pub p99_latency_us: u64,
    /// Optional human-readable regime label at snapshot time.
    pub regime: Option<String>,
}

/// HTTP client for pushing session metrics to JanusAI.
///
/// Construct once at startup with `from_env()` and share via `Arc`.
#[derive(Debug, Clone)]
pub struct SessionMetricsClient {
    inner: Option<ActiveClient>,
}

#[derive(Debug, Clone)]
struct ActiveClient {
    base_url: String,
    http: reqwest::Client,
}

impl SessionMetricsClient {
    /// Build a client from environment variables.
    ///
    /// Reads `JANUS_AI_URL` (e.g. `http://fks_janus_ai:8000`). When unset,
    /// returns a disabled client whose `push()` is a debug-logged no-op.
    pub fn from_env() -> Self {
        let url = std::env::var("JANUS_AI_URL").ok().filter(|s| !s.is_empty());

        let inner = url.map(|base_url| {
            let timeout_secs = std::env::var("JANUS_AI_TIMEOUT_SECS")
                .ok()
                .and_then(|v| v.parse().ok())
                .unwrap_or(2);
            let http = reqwest::Client::builder()
                .timeout(Duration::from_secs(timeout_secs))
                .build()
                .unwrap_or_else(|_| reqwest::Client::new());
            ActiveClient {
                base_url: base_url.trim_end_matches('/').to_string(),
                http,
            }
        });

        Self { inner }
    }

    /// True when the client will actually attempt a network call.
    pub fn is_enabled(&self) -> bool {
        self.inner.is_some()
    }

    /// Push a metrics snapshot for `session_id`. Errors are logged and
    /// swallowed — metric push must never break the signal pipeline.
    pub async fn push(&self, session_id: &str, metrics: &SessionMetrics) {
        let Some(client) = &self.inner else {
            debug!(
                session_id,
                "session metrics push skipped — JANUS_AI_URL not set"
            );
            return;
        };

        let url = format!(
            "{base}/api/janus-ai/sessions/{session_id}/metrics",
            base = client.base_url
        );

        match client.http.post(&url).json(metrics).send().await {
            Ok(resp) if resp.status().is_success() => {
                debug!(session_id, status = %resp.status(), "session metrics pushed");
            }
            Ok(resp) => {
                warn!(
                    session_id,
                    status = %resp.status(),
                    "session metrics push: non-success response"
                );
            }
            Err(e) => {
                warn!(session_id, error = %e, "session metrics push failed");
            }
        }
    }
}

impl Default for SessionMetricsClient {
    fn default() -> Self {
        Self::from_env()
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn disabled_when_env_unset() {
        // Defensive: snapshot any pre-existing value so other tests are not
        // perturbed by removal.
        let prev = std::env::var("JANUS_AI_URL").ok();
        // SAFETY: tests in this module are not run in parallel against each
        // other for this env var.
        unsafe { std::env::remove_var("JANUS_AI_URL") };
        let client = SessionMetricsClient::from_env();
        assert!(!client.is_enabled());
        if let Some(v) = prev {
            unsafe { std::env::set_var("JANUS_AI_URL", v) };
        }
    }

    #[tokio::test]
    async fn push_no_ops_when_disabled() {
        let prev = std::env::var("JANUS_AI_URL").ok();
        unsafe { std::env::remove_var("JANUS_AI_URL") };
        let client = SessionMetricsClient::from_env();
        // Should return without error / panic even with no JanusAI running.
        client.push("session-abc", &SessionMetrics::default()).await;
        if let Some(v) = prev {
            unsafe { std::env::set_var("JANUS_AI_URL", v) };
        }
    }
}
