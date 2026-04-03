//! Integration tests
//!
//! End-to-end tests with real exchange APIs and infrastructure services.
//!
//! - **Kraken public WebSocket**: FREE market data (ticker, trades, candles) — no API key required
//! - **Kraken authenticated REST/WS**: Balance, orders, fills — requires `KRAKEN_API_KEY` + `KRAKEN_API_SECRET`
//! - **Redis + QuestDB**: Data service integration tests with real infrastructure
//!
//! Set `RUN_INTEGRATION_TESTS=1` to enable exchange integration tests.

#[path = "integration/config.rs"]
mod config;
#[path = "integration/scenarios.rs"]
mod scenarios;

#[cfg(test)]
mod tests {
    use super::config::TestnetConfig;

    #[test]
    fn test_config_loading() {
        let config = TestnetConfig::from_env();
        assert!(
            config.is_ok(),
            "Config should load from env or use defaults"
        );
    }
}
