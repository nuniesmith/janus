//! JANUS Asset Registry and Service Discovery
//!
//! This crate provides centralized management of:
//! - Tradable assets and their metadata
//! - Service registration and discovery
//! - Configuration management
//! - Redis-based persistence and caching
//!
//! ## Features
//!
//! - **Asset Registry**: Define and manage tradable assets (crypto, forex, equities)
//! - **Service Discovery**: Register and discover JANUS services
//! - **Configuration**: Centralized configuration for trading parameters
//! - **Redis Cache**: Persistent caching layer for registry data
//!
//! ## Architecture
//!
//! ```text
//! ┌─────────────────────────────────────────────────────────────┐
//! │                    JANUS REGISTRY                           │
//! ├─────────────────────────────────────────────────────────────┤
//! │                                                              │
//! │  ┌──────────────┐  ┌──────────────┐  ┌──────────────┐      │
//! │  │    Asset     │  │   Service    │  │    Config    │      │
//! │  │   Registry   │  │  Discovery   │  │   Registry   │      │
//! │  └──────────────┘  └──────────────┘  └──────────────┘      │
//! │                                                              │
//! │  ┌────────────────────────────────────────────────────┐     │
//! │  │              Registry Manager                       │     │
//! │  └────────────────────────────────────────────────────┘     │
//! │                                                              │
//! └─────────────────────────────────────────────────────────────┘
//! ```

pub mod cache;

use chrono::{DateTime, Utc};
use serde::{Deserialize, Serialize};
use std::collections::HashMap;
use std::sync::Arc;
use tokio::sync::RwLock;
use tracing::{debug, info, warn};

// Re-export cache types
pub use cache::{CacheConfig, CacheError, CacheStats, CacheStatsSnapshot, RegistryCache};

/// Registry error types
#[derive(Debug, thiserror::Error)]
pub enum RegistryError {
    #[error("Asset not found: {0}")]
    AssetNotFound(String),

    #[error("Service not found: {0}")]
    ServiceNotFound(String),

    #[error("Duplicate asset: {0}")]
    DuplicateAsset(String),

    #[error("Duplicate service: {0}")]
    DuplicateService(String),

    #[error("Invalid asset configuration: {0}")]
    InvalidAsset(String),

    #[error("Invalid service configuration: {0}")]
    InvalidService(String),

    #[error("Registry locked: {0}")]
    Locked(String),
}

/// Result type for registry operations
pub type Result<T> = std::result::Result<T, RegistryError>;

// ============================================================================
// Asset Registry
// ============================================================================

/// Asset type classification
#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash, Serialize, Deserialize, Default)]
pub enum AssetType {
    /// Cryptocurrency
    #[default]
    Crypto,
    /// Foreign exchange
    Forex,
    /// Equities/Stocks
    Equity,
    /// Futures contracts
    Futures,
    /// Options
    Options,
    /// Commodities
    Commodity,
    /// Indices
    Index,
    /// Other/Custom
    Other,
}

/// Asset status
#[derive(Debug, Clone, Copy, PartialEq, Eq, Serialize, Deserialize, Default)]
pub enum AssetStatus {
    /// Asset is active and tradable
    #[default]
    Active,
    /// Asset is inactive (not tradable)
    Inactive,
    /// Asset is suspended (temporarily not tradable)
    Suspended,
    /// Asset is delisted
    Delisted,
}

/// Trading pair definition
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct TradingPair {
    /// Base asset (e.g., BTC in BTC/USDT)
    pub base: String,

    /// Quote asset (e.g., USDT in BTC/USDT)
    pub quote: String,

    /// Combined symbol (e.g., "BTC/USDT")
    pub symbol: String,

    /// Exchange-specific symbol format
    pub exchange_symbol: Option<String>,
}

impl TradingPair {
    /// Create a new trading pair
    pub fn new(base: &str, quote: &str) -> Self {
        Self {
            base: base.to_uppercase(),
            quote: quote.to_uppercase(),
            symbol: format!("{}/{}", base.to_uppercase(), quote.to_uppercase()),
            exchange_symbol: None,
        }
    }

    /// Set exchange-specific symbol
    pub fn with_exchange_symbol(mut self, symbol: &str) -> Self {
        self.exchange_symbol = Some(symbol.to_string());
        self
    }
}

/// Asset definition with trading parameters
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct Asset {
    /// Unique asset identifier
    pub id: String,

    /// Human-readable name
    pub name: String,

    /// Asset symbol (e.g., "BTC", "ETH")
    pub symbol: String,

    /// Asset type
    pub asset_type: AssetType,

    /// Current status
    pub status: AssetStatus,

    /// Trading pairs for this asset
    pub trading_pairs: Vec<TradingPair>,

    /// Decimal precision for price
    pub price_precision: u8,

    /// Decimal precision for quantity
    pub quantity_precision: u8,

    /// Minimum order size
    pub min_order_size: f64,

    /// Maximum order size
    pub max_order_size: Option<f64>,

    /// Minimum notional value
    pub min_notional: Option<f64>,

    /// Tick size (minimum price movement)
    pub tick_size: f64,

    /// Lot size (minimum quantity movement)
    pub lot_size: f64,

    /// Trading fees (maker/taker)
    pub maker_fee: f64,
    pub taker_fee: f64,

    /// Supported exchanges
    pub exchanges: Vec<String>,

    /// Additional metadata
    pub metadata: HashMap<String, String>,

    /// Creation timestamp
    pub created_at: DateTime<Utc>,

    /// Last update timestamp
    pub updated_at: DateTime<Utc>,
}

impl Asset {
    /// Create a new asset with minimal required fields
    pub fn new(id: &str, name: &str, symbol: &str, asset_type: AssetType) -> Self {
        let now = Utc::now();
        Self {
            id: id.to_string(),
            name: name.to_string(),
            symbol: symbol.to_uppercase(),
            asset_type,
            status: AssetStatus::Active,
            trading_pairs: Vec::new(),
            price_precision: 8,
            quantity_precision: 8,
            min_order_size: 0.0001,
            max_order_size: None,
            min_notional: None,
            tick_size: 0.00000001,
            lot_size: 0.00000001,
            maker_fee: 0.001,
            taker_fee: 0.001,
            exchanges: Vec::new(),
            metadata: HashMap::new(),
            created_at: now,
            updated_at: now,
        }
    }

    /// Add a trading pair
    pub fn add_trading_pair(&mut self, pair: TradingPair) {
        self.trading_pairs.push(pair);
        self.updated_at = Utc::now();
    }

    /// Add an exchange
    pub fn add_exchange(&mut self, exchange: &str) {
        if !self.exchanges.contains(&exchange.to_string()) {
            self.exchanges.push(exchange.to_string());
            self.updated_at = Utc::now();
        }
    }

    /// Check if asset is tradable
    pub fn is_tradable(&self) -> bool {
        self.status == AssetStatus::Active
    }

    /// Validate order size
    pub fn validate_order_size(&self, size: f64) -> bool {
        if size < self.min_order_size {
            return false;
        }
        if let Some(max) = self.max_order_size
            && size > max
        {
            return false;
        }
        true
    }

    /// Round price to tick size
    pub fn round_price(&self, price: f64) -> f64 {
        (price / self.tick_size).round() * self.tick_size
    }

    /// Round quantity to lot size
    pub fn round_quantity(&self, quantity: f64) -> f64 {
        (quantity / self.lot_size).round() * self.lot_size
    }
}

/// Asset registry for managing tradable assets
pub struct AssetRegistry {
    assets: RwLock<HashMap<String, Asset>>,
    symbols_index: RwLock<HashMap<String, String>>, // symbol -> id mapping
}

impl AssetRegistry {
    /// Create a new asset registry
    pub fn new() -> Self {
        Self {
            assets: RwLock::new(HashMap::new()),
            symbols_index: RwLock::new(HashMap::new()),
        }
    }

    /// Register a new asset
    pub async fn register(&self, asset: Asset) -> Result<()> {
        let mut assets = self.assets.write().await;
        let mut index = self.symbols_index.write().await;

        if assets.contains_key(&asset.id) {
            return Err(RegistryError::DuplicateAsset(asset.id));
        }

        if index.contains_key(&asset.symbol) {
            return Err(RegistryError::DuplicateAsset(asset.symbol));
        }

        info!("Registering asset: {} ({})", asset.name, asset.symbol);
        index.insert(asset.symbol.clone(), asset.id.clone());
        assets.insert(asset.id.clone(), asset);

        Ok(())
    }

    /// Get asset by ID
    pub async fn get(&self, id: &str) -> Result<Asset> {
        let assets = self.assets.read().await;
        assets
            .get(id)
            .cloned()
            .ok_or_else(|| RegistryError::AssetNotFound(id.to_string()))
    }

    /// Get asset by symbol
    pub async fn get_by_symbol(&self, symbol: &str) -> Result<Asset> {
        let index = self.symbols_index.read().await;
        let id = index
            .get(&symbol.to_uppercase())
            .ok_or_else(|| RegistryError::AssetNotFound(symbol.to_string()))?;

        self.get(id).await
    }

    /// List all assets
    pub async fn list(&self) -> Vec<Asset> {
        let assets = self.assets.read().await;
        assets.values().cloned().collect()
    }

    /// List assets by type
    pub async fn list_by_type(&self, asset_type: AssetType) -> Vec<Asset> {
        let assets = self.assets.read().await;
        assets
            .values()
            .filter(|a| a.asset_type == asset_type)
            .cloned()
            .collect()
    }

    /// List tradable assets
    pub async fn list_tradable(&self) -> Vec<Asset> {
        let assets = self.assets.read().await;
        assets
            .values()
            .filter(|a| a.is_tradable())
            .cloned()
            .collect()
    }

    /// Update asset status
    pub async fn update_status(&self, id: &str, status: AssetStatus) -> Result<()> {
        let mut assets = self.assets.write().await;
        let asset = assets
            .get_mut(id)
            .ok_or_else(|| RegistryError::AssetNotFound(id.to_string()))?;

        asset.status = status;
        asset.updated_at = Utc::now();

        info!("Updated asset status: {} -> {:?}", id, status);
        Ok(())
    }

    /// Remove an asset
    pub async fn remove(&self, id: &str) -> Result<Asset> {
        let mut assets = self.assets.write().await;
        let mut index = self.symbols_index.write().await;

        let asset = assets
            .remove(id)
            .ok_or_else(|| RegistryError::AssetNotFound(id.to_string()))?;

        index.remove(&asset.symbol);

        info!("Removed asset: {} ({})", asset.name, asset.symbol);
        Ok(asset)
    }

    /// Get asset count
    pub async fn count(&self) -> usize {
        let assets = self.assets.read().await;
        assets.len()
    }
}

impl Default for AssetRegistry {
    fn default() -> Self {
        Self::new()
    }
}

// ============================================================================
// Service Discovery
// ============================================================================

/// Service type
#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash, Serialize, Deserialize)]
pub enum ServiceType {
    /// Forward service (signal generation)
    Forward,
    /// Backward service (historical analysis)
    Backward,
    /// CNS service (monitoring)
    Cns,
    /// Data service (market data)
    Data,
    /// API service (external API)
    Api,
    /// Execution service
    Execution,
    /// Custom service
    Custom,
}

/// Service health status
#[derive(Debug, Clone, Copy, PartialEq, Eq, Serialize, Deserialize, Default)]
pub enum ServiceHealth {
    Healthy,
    Degraded,
    Unhealthy,
    #[default]
    Unknown,
}

/// Service endpoint
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct ServiceEndpoint {
    /// Protocol (http, https, grpc, ws)
    pub protocol: String,

    /// Host
    pub host: String,

    /// Port
    pub port: u16,

    /// Path (optional)
    pub path: Option<String>,
}

impl ServiceEndpoint {
    /// Create a new endpoint
    pub fn new(protocol: &str, host: &str, port: u16) -> Self {
        Self {
            protocol: protocol.to_string(),
            host: host.to_string(),
            port,
            path: None,
        }
    }

    /// Get full URL
    pub fn url(&self) -> String {
        let base = format!("{}://{}:{}", self.protocol, self.host, self.port);
        match &self.path {
            Some(path) => format!("{}{}", base, path),
            None => base,
        }
    }
}

/// Service instance definition
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct ServiceInstance {
    /// Unique instance ID
    pub id: String,

    /// Service name
    pub name: String,

    /// Service type
    pub service_type: ServiceType,

    /// Service version
    pub version: String,

    /// Primary endpoint
    pub endpoint: ServiceEndpoint,

    /// Additional endpoints (e.g., metrics, health)
    pub additional_endpoints: HashMap<String, ServiceEndpoint>,

    /// Current health status
    pub health: ServiceHealth,

    /// Last health check timestamp
    pub last_health_check: Option<DateTime<Utc>>,

    /// Service metadata
    pub metadata: HashMap<String, String>,

    /// Registration timestamp
    pub registered_at: DateTime<Utc>,

    /// Last heartbeat timestamp
    pub last_heartbeat: DateTime<Utc>,
}

impl ServiceInstance {
    /// Create a new service instance
    pub fn new(
        name: &str,
        service_type: ServiceType,
        version: &str,
        endpoint: ServiceEndpoint,
    ) -> Self {
        let now = Utc::now();
        Self {
            id: uuid::Uuid::new_v4().to_string(),
            name: name.to_string(),
            service_type,
            version: version.to_string(),
            endpoint,
            additional_endpoints: HashMap::new(),
            health: ServiceHealth::Unknown,
            last_health_check: None,
            metadata: HashMap::new(),
            registered_at: now,
            last_heartbeat: now,
        }
    }

    /// Add additional endpoint
    pub fn add_endpoint(&mut self, name: &str, endpoint: ServiceEndpoint) {
        self.additional_endpoints.insert(name.to_string(), endpoint);
    }

    /// Update health status
    pub fn update_health(&mut self, health: ServiceHealth) {
        self.health = health;
        self.last_health_check = Some(Utc::now());
    }

    /// Update heartbeat
    pub fn heartbeat(&mut self) {
        self.last_heartbeat = Utc::now();
    }

    /// Check if service is stale (no heartbeat in given duration)
    pub fn is_stale(&self, max_age_seconds: i64) -> bool {
        let age = Utc::now() - self.last_heartbeat;
        age.num_seconds() > max_age_seconds
    }
}

/// Service discovery registry
pub struct ServiceRegistry {
    services: RwLock<HashMap<String, ServiceInstance>>,
    by_type: RwLock<HashMap<ServiceType, Vec<String>>>,
    heartbeat_timeout: i64, // seconds
}

impl ServiceRegistry {
    /// Create a new service registry
    pub fn new() -> Self {
        Self {
            services: RwLock::new(HashMap::new()),
            by_type: RwLock::new(HashMap::new()),
            heartbeat_timeout: 60, // 60 seconds default
        }
    }

    /// Create with custom heartbeat timeout
    pub fn with_timeout(heartbeat_timeout: i64) -> Self {
        Self {
            services: RwLock::new(HashMap::new()),
            by_type: RwLock::new(HashMap::new()),
            heartbeat_timeout,
        }
    }

    /// Register a service instance
    pub async fn register(&self, service: ServiceInstance) -> Result<String> {
        let mut services = self.services.write().await;
        let mut by_type = self.by_type.write().await;

        let id = service.id.clone();
        let service_type = service.service_type;

        info!(
            "Registering service: {} ({:?}) at {}",
            service.name,
            service.service_type,
            service.endpoint.url()
        );

        services.insert(id.clone(), service);

        by_type
            .entry(service_type)
            .or_insert_with(Vec::new)
            .push(id.clone());

        Ok(id)
    }

    /// Deregister a service instance
    pub async fn deregister(&self, id: &str) -> Result<ServiceInstance> {
        let mut services = self.services.write().await;
        let mut by_type = self.by_type.write().await;

        let service = services
            .remove(id)
            .ok_or_else(|| RegistryError::ServiceNotFound(id.to_string()))?;

        if let Some(ids) = by_type.get_mut(&service.service_type) {
            ids.retain(|i| i != id);
        }

        info!(
            "Deregistered service: {} ({:?})",
            service.name, service.service_type
        );
        Ok(service)
    }

    /// Get service by ID
    pub async fn get(&self, id: &str) -> Result<ServiceInstance> {
        let services = self.services.read().await;
        services
            .get(id)
            .cloned()
            .ok_or_else(|| RegistryError::ServiceNotFound(id.to_string()))
    }

    /// Get services by type
    pub async fn get_by_type(&self, service_type: ServiceType) -> Vec<ServiceInstance> {
        let services = self.services.read().await;
        let by_type = self.by_type.read().await;

        by_type
            .get(&service_type)
            .map(|ids| {
                ids.iter()
                    .filter_map(|id| services.get(id).cloned())
                    .collect()
            })
            .unwrap_or_default()
    }

    /// Get healthy services by type
    pub async fn get_healthy_by_type(&self, service_type: ServiceType) -> Vec<ServiceInstance> {
        self.get_by_type(service_type)
            .await
            .into_iter()
            .filter(|s| s.health == ServiceHealth::Healthy && !s.is_stale(self.heartbeat_timeout))
            .collect()
    }

    /// Update service heartbeat
    pub async fn heartbeat(&self, id: &str) -> Result<()> {
        let mut services = self.services.write().await;
        let service = services
            .get_mut(id)
            .ok_or_else(|| RegistryError::ServiceNotFound(id.to_string()))?;

        service.heartbeat();
        debug!("Heartbeat received from service: {}", id);
        Ok(())
    }

    /// Update service health
    pub async fn update_health(&self, id: &str, health: ServiceHealth) -> Result<()> {
        let mut services = self.services.write().await;
        let service = services
            .get_mut(id)
            .ok_or_else(|| RegistryError::ServiceNotFound(id.to_string()))?;

        service.update_health(health);
        info!("Updated health for service {}: {:?}", id, health);
        Ok(())
    }

    /// List all services
    pub async fn list(&self) -> Vec<ServiceInstance> {
        let services = self.services.read().await;
        services.values().cloned().collect()
    }

    /// Cleanup stale services
    pub async fn cleanup_stale(&self) -> Vec<String> {
        let mut services = self.services.write().await;
        let mut by_type = self.by_type.write().await;

        let stale_ids: Vec<String> = services
            .iter()
            .filter(|(_, s)| s.is_stale(self.heartbeat_timeout))
            .map(|(id, _)| id.clone())
            .collect();

        for id in &stale_ids {
            if let Some(service) = services.remove(id) {
                if let Some(ids) = by_type.get_mut(&service.service_type) {
                    ids.retain(|i| i != id);
                }
                warn!(
                    "Cleaned up stale service: {} ({:?})",
                    service.name, service.service_type
                );
            }
        }

        stale_ids
    }

    /// Get service count
    pub async fn count(&self) -> usize {
        let services = self.services.read().await;
        services.len()
    }
}

impl Default for ServiceRegistry {
    fn default() -> Self {
        Self::new()
    }
}

// ============================================================================
// Registry Manager
// ============================================================================

/// Unified registry manager
pub struct RegistryManager {
    /// Asset registry
    pub assets: Arc<AssetRegistry>,

    /// Service registry
    pub services: Arc<ServiceRegistry>,
}

impl RegistryManager {
    /// Create a new registry manager
    pub fn new() -> Self {
        Self {
            assets: Arc::new(AssetRegistry::new()),
            services: Arc::new(ServiceRegistry::new()),
        }
    }

    /// Create with custom service heartbeat timeout
    pub fn with_heartbeat_timeout(heartbeat_timeout: i64) -> Self {
        Self {
            assets: Arc::new(AssetRegistry::new()),
            services: Arc::new(ServiceRegistry::with_timeout(heartbeat_timeout)),
        }
    }

    /// Initialize with default crypto assets
    pub async fn init_default_crypto_assets(&self) -> Result<()> {
        let btc = Asset::new("btc", "Bitcoin", "BTC", AssetType::Crypto);
        let eth = Asset::new("eth", "Ethereum", "ETH", AssetType::Crypto);
        let sol = Asset::new("sol", "Solana", "SOL", AssetType::Crypto);
        let bnb = Asset::new("bnb", "Binance Coin", "BNB", AssetType::Crypto);
        let xrp = Asset::new("xrp", "Ripple", "XRP", AssetType::Crypto);

        self.assets.register(btc).await?;
        self.assets.register(eth).await?;
        self.assets.register(sol).await?;
        self.assets.register(bnb).await?;
        self.assets.register(xrp).await?;

        info!("Initialized default crypto assets");
        Ok(())
    }
}

impl Default for RegistryManager {
    fn default() -> Self {
        Self::new()
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_trading_pair() {
        let pair = TradingPair::new("btc", "usdt");
        assert_eq!(pair.symbol, "BTC/USDT");
        assert_eq!(pair.base, "BTC");
        assert_eq!(pair.quote, "USDT");
    }

    #[test]
    fn test_asset_creation() {
        let asset = Asset::new("btc", "Bitcoin", "BTC", AssetType::Crypto);
        assert_eq!(asset.symbol, "BTC");
        assert!(asset.is_tradable());
    }

    #[test]
    fn test_asset_order_validation() {
        let mut asset = Asset::new("btc", "Bitcoin", "BTC", AssetType::Crypto);
        asset.min_order_size = 0.001;
        asset.max_order_size = Some(100.0);

        assert!(asset.validate_order_size(0.01));
        assert!(!asset.validate_order_size(0.0001));
        assert!(!asset.validate_order_size(200.0));
    }

    #[test]
    fn test_price_rounding() {
        let mut asset = Asset::new("btc", "Bitcoin", "BTC", AssetType::Crypto);
        asset.tick_size = 0.01;

        let rounded = asset.round_price(50000.123);
        assert!((rounded - 50000.12).abs() < 0.001);
    }

    #[tokio::test]
    async fn test_asset_registry() {
        let registry = AssetRegistry::new();

        let asset = Asset::new("btc", "Bitcoin", "BTC", AssetType::Crypto);
        registry.register(asset).await.unwrap();

        let retrieved = registry.get("btc").await.unwrap();
        assert_eq!(retrieved.symbol, "BTC");

        let by_symbol = registry.get_by_symbol("BTC").await.unwrap();
        assert_eq!(by_symbol.id, "btc");

        assert_eq!(registry.count().await, 1);
    }

    #[test]
    fn test_service_endpoint() {
        let endpoint = ServiceEndpoint::new("http", "localhost", 8080);
        assert_eq!(endpoint.url(), "http://localhost:8080");
    }

    #[test]
    fn test_service_instance() {
        let endpoint = ServiceEndpoint::new("http", "localhost", 8080);
        let mut service = ServiceInstance::new("forward", ServiceType::Forward, "1.0.0", endpoint);

        assert_eq!(service.health, ServiceHealth::Unknown);

        service.update_health(ServiceHealth::Healthy);
        assert_eq!(service.health, ServiceHealth::Healthy);
    }

    #[tokio::test]
    async fn test_service_registry() {
        let registry = ServiceRegistry::new();

        let endpoint = ServiceEndpoint::new("http", "localhost", 8080);
        let service = ServiceInstance::new("forward", ServiceType::Forward, "1.0.0", endpoint);

        let id = registry.register(service).await.unwrap();

        let retrieved = registry.get(&id).await.unwrap();
        assert_eq!(retrieved.name, "forward");

        let by_type = registry.get_by_type(ServiceType::Forward).await;
        assert_eq!(by_type.len(), 1);

        assert_eq!(registry.count().await, 1);
    }

    #[tokio::test]
    async fn test_registry_manager() {
        let manager = RegistryManager::new();
        manager.init_default_crypto_assets().await.unwrap();

        let assets = manager.assets.list().await;
        assert_eq!(assets.len(), 5);
    }
}
