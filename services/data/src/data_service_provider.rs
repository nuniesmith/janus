//! Data Service Provider
//!
//! High-level interface for Janus modules to pull data from the Python data
//! service (`fks_ruby`).  The data service is the single source of truth for
//! bar data within the platform — it resolves candles through a priority chain:
//!
//!   Redis cache → PostgreSQL (ruby_db) → External APIs
//!
//! External API backends (managed by the Python service):
//!   - **Massive** — CME/futures historical bars (MGC, MES, SIL, MNQ, M2K, MYM)
//!   - **Kraken** — Crypto spot (BTC, ETH, SOL, etc.)
//!   - **Binance** — Crypto spot/futures
//!   - **Bybit** — Crypto derivatives
//!   - **Rithmic** — CME live streaming (once available)
//!
//! ## Usage
//!
//! ```rust,ignore
//! use janus_data::data_service_provider::DataServiceProvider;
//!
//! let provider = DataServiceProvider::from_env().unwrap();
//!
//! // Pre-flight: check connectivity and cache asset list
//! provider.preflight().await;
//!
//! // Fetch candles (routed through the Python service's resolution chain)
//! let candles = provider.get_candles("MGC", "1m", 500, 30).await?;
//!
//! // Discover available symbols
//! let symbols = provider.get_symbols().await?;
//!
//! // Check data freshness
//! let gaps = provider.get_gaps("MGC", 7, "1m").await?;
//! ```
//!
//! ## Configuration
//!
//! Inherits all `PythonDataClient` env vars:
//!
//! | Variable                     | Default                | Description                            |
//! |------------------------------|------------------------|----------------------------------------|
//! | `PYTHON_DATA_SERVICE_URL`    | `http://fks_ruby:8000` | Base URL of the data service           |
//! | `DATA_SERVICE_API_KEY`       | *(empty)*              | Bearer token                           |
//! | `DATA_CLIENT_TIMEOUT_SECS`   | `30`                   | Per-request HTTP timeout               |
//! | `DATA_CLIENT_MAX_RETRIES`    | `3`                    | Retries on transient errors            |
//! | `DATA_CLIENT_RETRY_DELAY_MS` | `500`                  | Base delay between retries             |

use anyhow::{Context, Result};
use std::sync::Arc;
use tokio::sync::RwLock;
use tracing::{debug, info, warn};

use crate::actors::indicator::CandleInput;
use crate::backfill::python_data_client::{
    AssetInfo, BarsStatusResponse, CandleFetchRequest, FillJobResponse, GapReportResponse,
    LiveFeedStatus, PythonDataClient, PythonDataClientConfig, to_data_service_symbol,
};

// ============================================================================
// Provider state
// ============================================================================

/// Cached discovery data — refreshed periodically or on demand.
#[derive(Debug, Clone, Default)]
struct CachedState {
    /// Available symbols (lightweight list).
    symbols: Vec<String>,
    /// Full asset descriptors with bar counts.
    assets: Vec<AssetInfo>,
    /// Timestamp of last cache refresh.
    last_refresh: Option<std::time::Instant>,
    /// Whether the service was reachable at last check.
    is_healthy: bool,
}

// ============================================================================
// DataServiceProvider
// ============================================================================

/// High-level data provider that routes all data requests through the Python
/// data service.  Wraps `PythonDataClient` with caching and discovery.
pub struct DataServiceProvider {
    client: PythonDataClient,
    cache: Arc<RwLock<CachedState>>,
    /// How long cached symbols/assets remain valid before auto-refreshing.
    cache_ttl: std::time::Duration,
}

impl DataServiceProvider {
    /// Create a new provider from an existing client.
    pub fn new(client: PythonDataClient) -> Self {
        Self {
            client,
            cache: Arc::new(RwLock::new(CachedState::default())),
            cache_ttl: std::time::Duration::from_secs(300), // 5 minutes
        }
    }

    /// Create a provider from environment variables.
    pub fn from_env() -> Result<Self> {
        let client = PythonDataClient::from_env()
            .context("DataServiceProvider: failed to create PythonDataClient from env")?;
        Ok(Self::new(client))
    }

    /// Create a provider with custom configuration.
    pub fn with_config(config: PythonDataClientConfig) -> Result<Self> {
        let client = PythonDataClient::new(config)
            .context("DataServiceProvider: failed to create PythonDataClient")?;
        Ok(Self::new(client))
    }

    /// Set the cache TTL (how long symbol/asset lists are cached).
    pub fn with_cache_ttl(mut self, ttl: std::time::Duration) -> Self {
        self.cache_ttl = ttl;
        self
    }

    // =========================================================================
    // Pre-flight & health
    // =========================================================================

    /// Run a pre-flight check: verify the data service is reachable and
    /// cache the symbol/asset lists.
    ///
    /// This is non-fatal — if the service is down, the provider logs a
    /// warning and continues.  Callers should check `is_healthy()` to
    /// decide whether to fall back to alternative data sources.
    pub async fn preflight(&self) {
        info!(
            "DataServiceProvider: running pre-flight check against {}",
            self.client.base_url()
        );

        let healthy = self.client.health_check().await;

        if healthy {
            info!("DataServiceProvider: ✅ data service is healthy");

            // Cache symbols and assets
            if let Err(e) = self.refresh_cache().await {
                warn!(
                    "DataServiceProvider: pre-flight cache refresh failed — {}",
                    e
                );
            }
        } else {
            warn!(
                "DataServiceProvider: ⚠️ data service is unreachable at {}",
                self.client.base_url()
            );
        }

        let mut state = self.cache.write().await;
        state.is_healthy = healthy;
    }

    /// Check if the data service was healthy at the last check.
    pub async fn is_healthy(&self) -> bool {
        self.cache.read().await.is_healthy
    }

    /// Run a live health check (hits the service, not cached).
    pub async fn health_check(&self) -> bool {
        let healthy = self.client.health_check().await;
        self.cache.write().await.is_healthy = healthy;
        healthy
    }

    // =========================================================================
    // Symbol & asset discovery
    // =========================================================================

    /// Get the list of available symbols, using cache when fresh.
    pub async fn get_symbols(&self) -> Result<Vec<String>> {
        self.maybe_refresh_cache().await;
        let state = self.cache.read().await;
        Ok(state.symbols.clone())
    }

    /// Get the full asset list with bar counts, using cache when fresh.
    pub async fn get_assets(&self) -> Result<Vec<AssetInfo>> {
        self.maybe_refresh_cache().await;
        let state = self.cache.read().await;
        Ok(state.assets.clone())
    }

    /// Force a cache refresh.
    pub async fn refresh_cache(&self) -> Result<()> {
        debug!("DataServiceProvider: refreshing symbol/asset cache");

        let symbols = match self.client.fetch_symbols().await {
            Ok(resp) => {
                info!(
                    "DataServiceProvider: cached {} symbols from data service",
                    resp.symbols.len()
                );
                resp.symbols
            }
            Err(e) => {
                warn!("DataServiceProvider: failed to fetch symbols — {}", e);
                Vec::new()
            }
        };

        let assets = match self.client.fetch_assets().await {
            Ok(resp) => {
                info!(
                    "DataServiceProvider: cached {} assets from data service",
                    resp.assets.len()
                );
                resp.assets
            }
            Err(e) => {
                warn!("DataServiceProvider: failed to fetch assets — {}", e);
                Vec::new()
            }
        };

        let mut state = self.cache.write().await;
        state.symbols = symbols;
        state.assets = assets;
        state.last_refresh = Some(std::time::Instant::now());
        Ok(())
    }

    /// Refresh cache if it's expired.
    async fn maybe_refresh_cache(&self) {
        let needs_refresh = {
            let state = self.cache.read().await;
            match state.last_refresh {
                None => true,
                Some(last) => last.elapsed() > self.cache_ttl,
            }
        };

        if needs_refresh && let Err(e) = self.refresh_cache().await {
            warn!("DataServiceProvider: cache refresh failed — {}", e);
        }
    }

    // =========================================================================
    // Candle data (the main interface)
    // =========================================================================

    /// Fetch candles for a symbol through the data service's resolution chain:
    /// Redis → Postgres → Massive/Kraken/Binance/Bybit.
    ///
    /// This is the primary method Janus should use to get historical candle
    /// data.  The Python data service handles all source resolution internally.
    pub async fn get_candles(
        &self,
        symbol: &str,
        interval: &str,
        limit: usize,
        days_back: u32,
    ) -> Result<Vec<CandleInput>> {
        let ds_symbol = to_data_service_symbol(symbol);

        let request = CandleFetchRequest {
            symbol: ds_symbol.clone(),
            interval: interval.to_string(),
            limit: limit.min(10_000) as u32,
            days_back,
            auto_fill: true,
        };

        let result = self.client.fetch_candles(&request).await.with_context(|| {
            format!(
                "DataServiceProvider: failed to fetch candles for {}:{}",
                symbol, interval
            )
        })?;

        debug!(
            "DataServiceProvider: got {} candles for {}:{} from data service",
            result.candles.len(),
            symbol,
            interval
        );

        Ok(result.candles)
    }

    /// Fetch candles for multiple symbols.  Failures for individual symbols
    /// are logged but don't abort the batch.
    pub async fn get_candles_multi(
        &self,
        requests: &[(String, String, usize)], // (symbol, interval, limit)
        days_back: u32,
    ) -> Vec<(String, Result<Vec<CandleInput>>)> {
        let mut results = Vec::with_capacity(requests.len());

        for (symbol, interval, limit) in requests {
            let result = self.get_candles(symbol, interval, *limit, days_back).await;
            results.push((symbol.clone(), result));
        }

        results
    }

    // =========================================================================
    // Gap detection & fill management
    // =========================================================================

    /// Get a gap report for a specific symbol.
    pub async fn get_gaps(
        &self,
        symbol: &str,
        days_back: u32,
        interval: &str,
    ) -> Result<GapReportResponse> {
        self.client
            .fetch_gaps(symbol, days_back, interval)
            .await
            .with_context(|| format!("DataServiceProvider: failed to get gaps for {}", symbol))
    }

    /// Get bar status for all symbols.
    pub async fn get_bars_status(&self) -> Result<BarsStatusResponse> {
        self.client
            .fetch_bars_status()
            .await
            .context("DataServiceProvider: failed to get bars status")
    }

    /// Trigger a fill for a specific symbol.
    pub async fn trigger_fill(
        &self,
        symbol: &str,
        days_back: u32,
        interval: &str,
    ) -> Result<FillJobResponse> {
        info!(
            "DataServiceProvider: triggering fill for {} ({} days, {})",
            symbol, days_back, interval
        );
        self.client
            .trigger_fill(symbol, days_back, interval)
            .await
            .with_context(|| format!("DataServiceProvider: failed to trigger fill for {}", symbol))
    }

    /// Trigger a fill for all assets.
    pub async fn trigger_fill_all(&self, days_back: u32) -> Result<FillJobResponse> {
        info!(
            "DataServiceProvider: triggering fill-all ({} days)",
            days_back
        );
        self.client
            .trigger_fill_all(days_back)
            .await
            .context("DataServiceProvider: failed to trigger fill-all")
    }

    /// Poll the status of a fill job.
    pub async fn poll_fill_status(&self, job_id: Option<&str>) -> Result<FillJobResponse> {
        self.client
            .poll_fill_status(job_id)
            .await
            .context("DataServiceProvider: failed to poll fill status")
    }

    // =========================================================================
    // Live feed & data source info
    // =========================================================================

    /// Query which data source the engine is currently using.
    pub async fn get_data_source(&self) -> Result<String> {
        self.client
            .fetch_data_source()
            .await
            .context("DataServiceProvider: failed to get data source")
    }

    /// Get the live feed connection status.
    pub async fn get_live_feed_status(&self) -> Result<LiveFeedStatus> {
        self.client
            .fetch_live_feed_status()
            .await
            .context("DataServiceProvider: failed to get live feed status")
    }

    /// Return a reference to the underlying client for advanced use.
    pub fn client(&self) -> &PythonDataClient {
        &self.client
    }
}

// ============================================================================
// Tests
// ============================================================================

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_provider_from_env() {
        let provider = DataServiceProvider::from_env();
        assert!(
            provider.is_ok(),
            "DataServiceProvider::from_env() should not fail"
        );
    }

    #[test]
    fn test_provider_with_custom_ttl() {
        let provider = DataServiceProvider::from_env()
            .unwrap()
            .with_cache_ttl(std::time::Duration::from_secs(60));
        assert_eq!(provider.cache_ttl, std::time::Duration::from_secs(60));
    }

    #[tokio::test]
    async fn test_cached_state_default() {
        let provider = DataServiceProvider::from_env().unwrap();
        let state = provider.cache.read().await;
        assert!(state.symbols.is_empty());
        assert!(state.assets.is_empty());
        assert!(state.last_refresh.is_none());
        assert!(!state.is_healthy);
    }
}
