//! # JANUS Exchanges — ingestion support utilities
//!
//! Janus-side support code for the market-data ingestion pipeline:
//!
//! - [`CNSReporter`] — CNS metrics (message counts, latency, parse errors).
//! - [`HealthChecker`] / [`ExchangeHealth`] — per-exchange connection health.
//! - [`PriceNormalizer`] — cross-exchange price normalization.
//! - [`ConnectionConfig`] / [`SubscriptionRequest`] — connection/subscription types.
//!
//! > **Exchange connectivity now lives in the published `exchange-apiws` crate.**
//! > The in-house WebSocket adapters that used to live here (Coinbase, Kraken,
//! > OKX, …) were retired once `exchange-apiws` covered those venues; the data
//! > service's bridges wrap `exchange_apiws::*Connector` directly and layer the
//! > CNS/health utilities below on top. See `services/data/src/connectors/bridge.rs`.

pub mod cns;
pub mod health;
pub mod normalizer;
pub mod types;

// Re-export commonly used types
pub use cns::CNSReporter;
pub use health::{ExchangeHealth, HealthChecker};
pub use normalizer::PriceNormalizer;
pub use types::{ConnectionConfig, SubscriptionRequest};

/// Library version
pub const VERSION: &str = env!("CARGO_PKG_VERSION");

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_version() {
        assert!(!VERSION.is_empty());
    }
}
