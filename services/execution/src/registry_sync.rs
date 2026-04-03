//! Registry routing sync module
//!
//! Periodically fetches the routing map from the JANUS Registry Service
//! and applies it to the [`ExchangeRouter`] so that symbol→exchange mappings
//! stay up-to-date without manual configuration.

use crate::exchanges::router::ExchangeRouter;
use serde::Deserialize;
use std::collections::HashMap;
use std::sync::Arc;
use tracing::{info, warn};

/// Response shape returned by `GET /api/v1/routing` on the Registry Service.
#[derive(Debug, Deserialize)]
struct RoutingMapResponse {
    routes: HashMap<String, String>,
    #[allow(dead_code)]
    exchange_symbols: HashMap<String, String>,
}

/// Background task that periodically syncs routing rules from the Registry
/// Service into the given [`ExchangeRouter`].
///
/// # Environment variables
///
/// | Variable                     | Default                        | Description                  |
/// |------------------------------|--------------------------------|------------------------------|
/// | `REGISTRY_SERVICE_URL`       | `http://fks_registry:8085`     | Base URL of registry service |
/// | `ROUTING_SYNC_INTERVAL_SECS` | `120`                          | Polling interval in seconds  |
pub async fn registry_routing_sync_loop(router: Arc<ExchangeRouter>) {
    let base_url = std::env::var("REGISTRY_SERVICE_URL")
        .unwrap_or_else(|_| "http://fks_registry:8085".to_string());

    let interval_secs: u64 = std::env::var("ROUTING_SYNC_INTERVAL_SECS")
        .ok()
        .and_then(|v| v.parse().ok())
        .unwrap_or(120);

    let client = reqwest::Client::builder()
        .timeout(std::time::Duration::from_secs(30))
        .build()
        .unwrap_or_default();

    // Initial delay — give the registry service time to start and populate assets.
    tokio::time::sleep(std::time::Duration::from_secs(15)).await;
    info!(
        "Registry routing sync loop started (url={}, interval={}s)",
        base_url, interval_secs
    );

    loop {
        let url = format!("{}/api/v1/routing", base_url);
        match client.get(&url).send().await {
            Ok(resp) => {
                if resp.status().is_success() {
                    match resp.json::<RoutingMapResponse>().await {
                        Ok(routing_map) => {
                            let count = router
                                .sync_routing_from_registry(routing_map.routes)
                                .await;
                            info!(
                                count,
                                "Registry routing sync complete — applied routing rules"
                            );
                        }
                        Err(e) => {
                            warn!(error = %e, "Failed to parse routing map response from registry");
                        }
                    }
                } else {
                    warn!(
                        status = %resp.status(),
                        "Registry routing endpoint returned non-success status"
                    );
                }
            }
            Err(e) => {
                warn!(error = %e, "Failed to reach registry service for routing sync");
            }
        }

        tokio::time::sleep(std::time::Duration::from_secs(interval_secs)).await;
    }
}
