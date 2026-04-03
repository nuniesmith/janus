//! Integration tests for external API clients using wiremock
//!
//! These tests mock the external APIs (NewsAPI, CryptoPanic, OpenWeatherMap, SpaceWeather)
//! to enable deterministic CI testing without actual API keys or network access.
//!
//! ## Test Modules
//!
//! - `test_clients` - Unit/integration tests for individual API clients
//! - `test_bridge` - Tests for DataSourceBridge functionality
//! - `test_aggregator` - Tests for AggregatorService functionality
//!
//! ## Running Tests
//!
//! ```bash
//! # Run all integration tests
//! cargo test -p janus-neuromorphic --test integration_tests
//!
//! # Run specific test module
//! cargo test -p janus-neuromorphic test_clients
//!
//! # Run with output
//! cargo test -p janus-neuromorphic -- --nocapture
//! ```

pub mod test_aggregator;
pub mod test_bridge;
pub mod test_clients;
