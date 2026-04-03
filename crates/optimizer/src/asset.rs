//! Asset Configuration and Registry
//!
//! Defines asset categories and their trading parameters based on volatility profiles.
//! Used to enforce per-asset constraints during optimization to prevent whipsaw trades.
//!
//! Asset categories are derived from `kraken_assets.md` and include:
//! - Major coins (BTC, ETH, SOL) - Lower volatility, tighter spreads
//! - Altcoins (L1/L2 chains) - Medium volatility
//! - DeFi tokens - Medium-high volatility, correlated with ETH
//! - Meme coins - Extreme volatility, require wide stops
//! - AI/Compute - High volatility, narrative-driven
//! - Gaming/Metaverse - High volatility
//! - Forex - Low volatility

use serde::{Deserialize, Serialize};
use std::collections::HashMap;

/// Asset category based on volatility and trading characteristics
#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash, Serialize, Deserialize, Default)]
pub enum AssetCategory {
    /// Blue chip cryptocurrencies (BTC, ETH, SOL, XRP, etc.)
    Major,

    /// Layer 1 and Layer 2 infrastructure tokens
    L1L2,

    /// DeFi protocol tokens
    DeFi,

    /// Meme coins with extreme volatility
    Meme,

    /// AI and compute-related tokens
    AiCompute,

    /// Gaming and metaverse tokens
    Gaming,

    /// Forex currency pairs
    Forex,

    /// Other/uncategorized assets
    #[default]
    Other,
}

impl AssetCategory {
    /// Get the minimum EMA spread percentage for this category
    pub fn min_ema_spread_pct(&self) -> f64 {
        match self {
            AssetCategory::Major => 0.15,
            AssetCategory::L1L2 => 0.18,
            AssetCategory::DeFi => 0.20,
            AssetCategory::Meme => 0.30,
            AssetCategory::AiCompute => 0.25,
            AssetCategory::Gaming => 0.25,
            AssetCategory::Forex => 0.10,
            AssetCategory::Other => 0.20,
        }
    }

    /// Get the minimum hold time in minutes for this category
    pub fn min_hold_minutes(&self) -> u32 {
        match self {
            AssetCategory::Major => 15,
            AssetCategory::L1L2 => 20,
            AssetCategory::DeFi => 20,
            AssetCategory::Meme => 30,
            AssetCategory::AiCompute => 25,
            AssetCategory::Gaming => 25,
            AssetCategory::Forex => 60,
            AssetCategory::Other => 20,
        }
    }

    /// Get typical volatility level description
    pub fn volatility_level(&self) -> &'static str {
        match self {
            AssetCategory::Major => "low-medium",
            AssetCategory::L1L2 => "medium",
            AssetCategory::DeFi => "medium-high",
            AssetCategory::Meme => "extreme",
            AssetCategory::AiCompute => "high",
            AssetCategory::Gaming => "high",
            AssetCategory::Forex => "low",
            AssetCategory::Other => "medium",
        }
    }

    /// Get recommended ATR multiplier range
    pub fn atr_multiplier_range(&self) -> (f64, f64) {
        match self {
            AssetCategory::Major => (1.5, 2.5),
            AssetCategory::L1L2 => (1.5, 3.0),
            AssetCategory::DeFi => (2.0, 3.5),
            AssetCategory::Meme => (2.5, 4.0),
            AssetCategory::AiCompute => (2.0, 3.5),
            AssetCategory::Gaming => (2.0, 3.5),
            AssetCategory::Forex => (1.0, 2.0),
            AssetCategory::Other => (1.5, 3.0),
        }
    }

    /// Get recommended take profit percentage range
    pub fn take_profit_range(&self) -> (f64, f64) {
        match self {
            AssetCategory::Major => (2.0, 8.0),
            AssetCategory::L1L2 => (3.0, 10.0),
            AssetCategory::DeFi => (3.0, 12.0),
            AssetCategory::Meme => (5.0, 20.0),
            AssetCategory::AiCompute => (4.0, 15.0),
            AssetCategory::Gaming => (4.0, 15.0),
            AssetCategory::Forex => (1.0, 5.0),
            AssetCategory::Other => (3.0, 10.0),
        }
    }

    /// Get minimum profit percentage floor
    pub fn min_profit_floor(&self) -> f64 {
        match self {
            AssetCategory::Major => 0.40,
            AssetCategory::L1L2 => 0.50,
            AssetCategory::DeFi => 0.50,
            AssetCategory::Meme => 0.75,
            AssetCategory::AiCompute => 0.60,
            AssetCategory::Gaming => 0.60,
            AssetCategory::Forex => 0.20,
            AssetCategory::Other => 0.50,
        }
    }
}

impl std::fmt::Display for AssetCategory {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        let name = match self {
            AssetCategory::Major => "Major",
            AssetCategory::L1L2 => "L1/L2",
            AssetCategory::DeFi => "DeFi",
            AssetCategory::Meme => "Meme",
            AssetCategory::AiCompute => "AI/Compute",
            AssetCategory::Gaming => "Gaming",
            AssetCategory::Forex => "Forex",
            AssetCategory::Other => "Other",
        };
        write!(f, "{}", name)
    }
}

/// Configuration for a specific asset
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct AssetConfig {
    /// Asset symbol (e.g., "BTC", "ETH")
    pub symbol: String,

    /// Kraken trading pair (e.g., "XXBTZUSD")
    pub kraken_pair: Option<String>,

    /// Asset category
    pub category: AssetCategory,

    /// Whether this asset is enabled for trading
    pub enabled: bool,

    /// Liquidity tier (1 = highest, 4 = lowest)
    pub liquidity_tier: u8,

    /// Custom minimum EMA spread (overrides category default)
    pub custom_min_ema_spread: Option<f64>,

    /// Custom minimum hold time (overrides category default)
    pub custom_min_hold_minutes: Option<u32>,

    /// Maximum position size in USD
    pub max_position_usd: f64,

    /// Additional notes
    pub notes: Option<String>,
}

impl AssetConfig {
    /// Create a new asset config with default values
    pub fn new(symbol: impl Into<String>, category: AssetCategory) -> Self {
        Self {
            symbol: symbol.into(),
            kraken_pair: None,
            category,
            enabled: true,
            liquidity_tier: 3, // Default to medium liquidity
            custom_min_ema_spread: None,
            custom_min_hold_minutes: None,
            max_position_usd: 20.0,
            notes: None,
        }
    }

    /// Set the Kraken trading pair
    pub fn with_kraken_pair(mut self, pair: impl Into<String>) -> Self {
        self.kraken_pair = Some(pair.into());
        self
    }

    /// Set the liquidity tier
    pub fn with_liquidity_tier(mut self, tier: u8) -> Self {
        self.liquidity_tier = tier.clamp(1, 4);
        self
    }

    /// Set custom minimum EMA spread
    pub fn with_min_ema_spread(mut self, spread: f64) -> Self {
        self.custom_min_ema_spread = Some(spread);
        self
    }

    /// Set custom minimum hold time
    pub fn with_min_hold_minutes(mut self, minutes: u32) -> Self {
        self.custom_min_hold_minutes = Some(minutes);
        self
    }

    /// Set maximum position size
    pub fn with_max_position(mut self, max_usd: f64) -> Self {
        self.max_position_usd = max_usd;
        self
    }

    /// Disable this asset
    pub fn disabled(mut self) -> Self {
        self.enabled = false;
        self
    }

    /// Get effective minimum EMA spread
    pub fn min_ema_spread_pct(&self) -> f64 {
        self.custom_min_ema_spread
            .unwrap_or_else(|| self.category.min_ema_spread_pct())
    }

    /// Get effective minimum hold time
    pub fn min_hold_minutes(&self) -> u32 {
        self.custom_min_hold_minutes
            .unwrap_or_else(|| self.category.min_hold_minutes())
    }
}

impl Default for AssetConfig {
    fn default() -> Self {
        Self::new("UNKNOWN", AssetCategory::Other)
    }
}

/// Registry of all known assets with their configurations
#[derive(Debug, Clone)]
pub struct AssetRegistry {
    /// Map of asset symbol to config
    assets: HashMap<String, AssetConfig>,

    /// Default config for unknown assets
    default_config: AssetConfig,
}

impl AssetRegistry {
    /// Create a new empty registry
    pub fn new() -> Self {
        Self {
            assets: HashMap::new(),
            default_config: AssetConfig::default(),
        }
    }

    /// Create registry with default Kraken assets
    pub fn with_kraken_defaults() -> Self {
        let mut registry = Self::new();
        registry.register_kraken_assets();
        registry
    }

    /// Register an asset
    pub fn register(&mut self, config: AssetConfig) {
        self.assets.insert(config.symbol.to_uppercase(), config);
    }

    /// Get config for an asset (returns default if not found)
    pub fn get(&self, symbol: &str) -> AssetConfig {
        self.assets
            .get(&symbol.to_uppercase())
            .cloned()
            .unwrap_or_else(|| {
                let mut config = self.default_config.clone();
                config.symbol = symbol.to_uppercase();
                config
            })
    }

    /// Check if an asset is registered
    pub fn contains(&self, symbol: &str) -> bool {
        self.assets.contains_key(&symbol.to_uppercase())
    }

    /// Get all registered assets
    pub fn all_assets(&self) -> Vec<&AssetConfig> {
        self.assets.values().collect()
    }

    /// Get all enabled assets
    pub fn enabled_assets(&self) -> Vec<&AssetConfig> {
        self.assets.values().filter(|a| a.enabled).collect()
    }

    /// Get assets by category
    pub fn by_category(&self, category: AssetCategory) -> Vec<&AssetConfig> {
        self.assets
            .values()
            .filter(|a| a.category == category)
            .collect()
    }

    /// Get assets by liquidity tier
    pub fn by_liquidity_tier(&self, tier: u8) -> Vec<&AssetConfig> {
        self.assets
            .values()
            .filter(|a| a.liquidity_tier == tier)
            .collect()
    }

    /// Get the number of registered assets
    pub fn len(&self) -> usize {
        self.assets.len()
    }

    /// Check if the registry is empty
    pub fn is_empty(&self) -> bool {
        self.assets.is_empty()
    }

    /// Register all default Kraken assets from kraken_assets.md
    fn register_kraken_assets(&mut self) {
        // ============================================
        // Major Coins (Tier 1-2 Liquidity)
        // ============================================
        self.register(
            AssetConfig::new("BTC", AssetCategory::Major)
                .with_kraken_pair("XXBTZUSD")
                .with_liquidity_tier(1)
                .with_max_position(100.0),
        );
        self.register(
            AssetConfig::new("ETH", AssetCategory::Major)
                .with_kraken_pair("XETHZUSD")
                .with_liquidity_tier(1)
                .with_max_position(100.0),
        );
        self.register(
            AssetConfig::new("SOL", AssetCategory::Major)
                .with_kraken_pair("SOLUSD")
                .with_liquidity_tier(1)
                .with_max_position(50.0),
        );
        self.register(
            AssetConfig::new("XRP", AssetCategory::Major)
                .with_kraken_pair("XXRPZUSD")
                .with_liquidity_tier(1)
                .with_max_position(50.0),
        );
        self.register(
            AssetConfig::new("ADA", AssetCategory::Major)
                .with_kraken_pair("ADAUSD")
                .with_liquidity_tier(2),
        );
        self.register(
            AssetConfig::new("AVAX", AssetCategory::Major)
                .with_kraken_pair("AVAXUSD")
                .with_liquidity_tier(2),
        );
        self.register(
            AssetConfig::new("DOT", AssetCategory::Major)
                .with_kraken_pair("DOTUSD")
                .with_liquidity_tier(2),
        );
        self.register(
            AssetConfig::new("LINK", AssetCategory::Major)
                .with_kraken_pair("LINKUSD")
                .with_liquidity_tier(2),
        );
        self.register(
            AssetConfig::new("LTC", AssetCategory::Major)
                .with_kraken_pair("XLTCZUSD")
                .with_liquidity_tier(2),
        );
        self.register(
            AssetConfig::new("BCH", AssetCategory::Major)
                .with_kraken_pair("BCHUSD")
                .with_liquidity_tier(2),
        );
        self.register(
            AssetConfig::new("XLM", AssetCategory::Major)
                .with_kraken_pair("XXLMZUSD")
                .with_liquidity_tier(2),
        );
        self.register(
            AssetConfig::new("ATOM", AssetCategory::Major)
                .with_kraken_pair("ATOMUSD")
                .with_liquidity_tier(2),
        );
        self.register(
            AssetConfig::new("ALGO", AssetCategory::Major)
                .with_kraken_pair("ALGOUSD")
                .with_liquidity_tier(2),
        );
        self.register(
            AssetConfig::new("HBAR", AssetCategory::Major)
                .with_kraken_pair("HBARUSD")
                .with_liquidity_tier(2),
        );
        self.register(
            AssetConfig::new("XMR", AssetCategory::Major)
                .with_kraken_pair("XXMRZUSD")
                .with_liquidity_tier(2),
        );

        // ============================================
        // Layer 1 & Layer 2
        // ============================================
        self.register(
            AssetConfig::new("NEAR", AssetCategory::L1L2)
                .with_kraken_pair("NEARUSD")
                .with_liquidity_tier(2),
        );
        self.register(
            AssetConfig::new("FIL", AssetCategory::L1L2)
                .with_kraken_pair("FILUSD")
                .with_liquidity_tier(3),
        );
        self.register(
            AssetConfig::new("ICP", AssetCategory::L1L2)
                .with_kraken_pair("ICPUSD")
                .with_liquidity_tier(3),
        );
        self.register(
            AssetConfig::new("APT", AssetCategory::L1L2)
                .with_kraken_pair("APTUSD")
                .with_liquidity_tier(3),
        );
        self.register(
            AssetConfig::new("SUI", AssetCategory::L1L2)
                .with_kraken_pair("SUIUSD")
                .with_liquidity_tier(3),
        );
        self.register(
            AssetConfig::new("SEI", AssetCategory::L1L2)
                .with_kraken_pair("SEIUSD")
                .with_liquidity_tier(3),
        );
        self.register(
            AssetConfig::new("INJ", AssetCategory::L1L2)
                .with_kraken_pair("INJUSD")
                .with_liquidity_tier(3),
        );
        self.register(
            AssetConfig::new("TIA", AssetCategory::L1L2)
                .with_kraken_pair("TIAUSD")
                .with_liquidity_tier(3),
        );
        self.register(
            AssetConfig::new("ARB", AssetCategory::L1L2)
                .with_kraken_pair("ARBUSD")
                .with_liquidity_tier(2),
        );
        self.register(
            AssetConfig::new("OP", AssetCategory::L1L2)
                .with_kraken_pair("OPUSD")
                .with_liquidity_tier(2),
        );
        self.register(
            AssetConfig::new("POL", AssetCategory::L1L2)
                .with_kraken_pair("POLUSD")
                .with_liquidity_tier(2),
        );
        self.register(
            AssetConfig::new("STRK", AssetCategory::L1L2)
                .with_kraken_pair("STRKUSD")
                .with_liquidity_tier(3),
        );
        self.register(
            AssetConfig::new("IMX", AssetCategory::L1L2)
                .with_kraken_pair("IMXUSD")
                .with_liquidity_tier(3),
        );
        self.register(
            AssetConfig::new("STX", AssetCategory::L1L2)
                .with_kraken_pair("STXUSD")
                .with_liquidity_tier(3),
        );

        // ============================================
        // DeFi Tokens
        // ============================================
        self.register(
            AssetConfig::new("UNI", AssetCategory::DeFi)
                .with_kraken_pair("UNIUSD")
                .with_liquidity_tier(2),
        );
        self.register(
            AssetConfig::new("AAVE", AssetCategory::DeFi)
                .with_kraken_pair("AAVEUSD")
                .with_liquidity_tier(3),
        );
        self.register(
            AssetConfig::new("MKR", AssetCategory::DeFi)
                .with_kraken_pair("MKRUSD")
                .with_liquidity_tier(3),
        );
        self.register(
            AssetConfig::new("CRV", AssetCategory::DeFi)
                .with_kraken_pair("CRVUSD")
                .with_liquidity_tier(3),
        );
        self.register(
            AssetConfig::new("LDO", AssetCategory::DeFi)
                .with_kraken_pair("LDOUSD")
                .with_liquidity_tier(3),
        );
        self.register(
            AssetConfig::new("COMP", AssetCategory::DeFi)
                .with_kraken_pair("COMPUSD")
                .with_liquidity_tier(3),
        );
        self.register(
            AssetConfig::new("SNX", AssetCategory::DeFi)
                .with_kraken_pair("SNXUSD")
                .with_liquidity_tier(3),
        );
        self.register(
            AssetConfig::new("SUSHI", AssetCategory::DeFi)
                .with_kraken_pair("SUSHIUSD")
                .with_liquidity_tier(3),
        );
        self.register(
            AssetConfig::new("1INCH", AssetCategory::DeFi)
                .with_kraken_pair("1INCHUSD")
                .with_liquidity_tier(3),
        );
        self.register(
            AssetConfig::new("GRT", AssetCategory::DeFi)
                .with_kraken_pair("GRTUSD")
                .with_liquidity_tier(3),
        );
        self.register(
            AssetConfig::new("ENS", AssetCategory::DeFi)
                .with_kraken_pair("ENSUSD")
                .with_liquidity_tier(3),
        );
        self.register(
            AssetConfig::new("DYDX", AssetCategory::DeFi)
                .with_kraken_pair("DYDXUSD")
                .with_liquidity_tier(3),
        );
        self.register(
            AssetConfig::new("GMX", AssetCategory::DeFi)
                .with_kraken_pair("GMXUSD")
                .with_liquidity_tier(3),
        );
        self.register(
            AssetConfig::new("JUP", AssetCategory::DeFi)
                .with_kraken_pair("JUPUSD")
                .with_liquidity_tier(3),
        );
        self.register(
            AssetConfig::new("RAY", AssetCategory::DeFi)
                .with_kraken_pair("RAYUSD")
                .with_liquidity_tier(3),
        );
        self.register(
            AssetConfig::new("ONDO", AssetCategory::DeFi)
                .with_kraken_pair("ONDOUSD")
                .with_liquidity_tier(3),
        );
        self.register(
            AssetConfig::new("ENA", AssetCategory::DeFi)
                .with_kraken_pair("ENAUSD")
                .with_liquidity_tier(3),
        );

        // ============================================
        // Meme Coins
        // ============================================
        self.register(
            AssetConfig::new("DOGE", AssetCategory::Meme)
                .with_kraken_pair("XDGUSD")
                .with_liquidity_tier(1)
                .with_max_position(30.0),
        );
        self.register(
            AssetConfig::new("SHIB", AssetCategory::Meme)
                .with_kraken_pair("SHIBUSD")
                .with_liquidity_tier(2),
        );
        self.register(
            AssetConfig::new("PEPE", AssetCategory::Meme)
                .with_kraken_pair("PEPEUSD")
                .with_liquidity_tier(2),
        );
        self.register(
            AssetConfig::new("BONK", AssetCategory::Meme)
                .with_kraken_pair("BONKUSD")
                .with_liquidity_tier(3),
        );
        self.register(
            AssetConfig::new("WIF", AssetCategory::Meme)
                .with_kraken_pair("WIFUSD")
                .with_liquidity_tier(3),
        );
        self.register(
            AssetConfig::new("FLOKI", AssetCategory::Meme)
                .with_kraken_pair("FLOKIUSD")
                .with_liquidity_tier(3),
        );
        self.register(
            AssetConfig::new("TURBO", AssetCategory::Meme)
                .with_kraken_pair("TURBOUSD")
                .with_liquidity_tier(4),
        );
        self.register(
            AssetConfig::new("MEW", AssetCategory::Meme)
                .with_kraken_pair("MEWUSD")
                .with_liquidity_tier(4),
        );
        self.register(
            AssetConfig::new("PONKE", AssetCategory::Meme)
                .with_kraken_pair("PONKEUSD")
                .with_liquidity_tier(4),
        );
        self.register(
            AssetConfig::new("FWOG", AssetCategory::Meme)
                .with_kraken_pair("FWOGUSD")
                .with_liquidity_tier(4),
        );
        self.register(
            AssetConfig::new("POPCAT", AssetCategory::Meme)
                .with_kraken_pair("POPCATUSD")
                .with_liquidity_tier(4),
        );
        self.register(
            AssetConfig::new("FARTCOIN", AssetCategory::Meme)
                .with_kraken_pair("FARTCOINUSD")
                .with_liquidity_tier(4),
        );
        self.register(
            AssetConfig::new("TRUMP", AssetCategory::Meme)
                .with_kraken_pair("TRUMPUSD")
                .with_liquidity_tier(4),
        );
        self.register(
            AssetConfig::new("PUMP", AssetCategory::Meme)
                .with_kraken_pair("PUMPUSD")
                .with_liquidity_tier(4),
        );

        // ============================================
        // AI & Compute
        // ============================================
        self.register(
            AssetConfig::new("RENDER", AssetCategory::AiCompute)
                .with_kraken_pair("RENDERUSD")
                .with_liquidity_tier(3),
        );
        self.register(
            AssetConfig::new("FET", AssetCategory::AiCompute)
                .with_kraken_pair("FETUSD")
                .with_liquidity_tier(3),
        );
        self.register(
            AssetConfig::new("TAO", AssetCategory::AiCompute)
                .with_kraken_pair("TAOUSD")
                .with_liquidity_tier(3),
        );
        self.register(
            AssetConfig::new("VIRTUAL", AssetCategory::AiCompute)
                .with_kraken_pair("VIRTUALUSD")
                .with_liquidity_tier(4),
        );
        self.register(
            AssetConfig::new("AGIX", AssetCategory::AiCompute)
                .with_kraken_pair("AGIXUSD")
                .with_liquidity_tier(3),
        );
        self.register(
            AssetConfig::new("OCEAN", AssetCategory::AiCompute)
                .with_kraken_pair("OCEANUSD")
                .with_liquidity_tier(3),
        );
        self.register(
            AssetConfig::new("AKT", AssetCategory::AiCompute)
                .with_kraken_pair("AKTUSD")
                .with_liquidity_tier(4),
        );
        self.register(
            AssetConfig::new("AR", AssetCategory::AiCompute)
                .with_kraken_pair("ARUSD")
                .with_liquidity_tier(3),
        );

        // ============================================
        // Gaming & Metaverse
        // ============================================
        self.register(
            AssetConfig::new("AXS", AssetCategory::Gaming)
                .with_kraken_pair("AXSUSD")
                .with_liquidity_tier(3),
        );
        self.register(
            AssetConfig::new("SAND", AssetCategory::Gaming)
                .with_kraken_pair("SANDUSD")
                .with_liquidity_tier(3),
        );
        self.register(
            AssetConfig::new("MANA", AssetCategory::Gaming)
                .with_kraken_pair("MANAUSD")
                .with_liquidity_tier(3),
        );
        self.register(
            AssetConfig::new("APE", AssetCategory::Gaming)
                .with_kraken_pair("APEUSD")
                .with_liquidity_tier(3),
        );
        self.register(
            AssetConfig::new("GALA", AssetCategory::Gaming)
                .with_kraken_pair("GALAUSD")
                .with_liquidity_tier(3),
        );
        self.register(
            AssetConfig::new("ILV", AssetCategory::Gaming)
                .with_kraken_pair("ILVUSD")
                .with_liquidity_tier(4),
        );
        self.register(
            AssetConfig::new("PRIME", AssetCategory::Gaming)
                .with_kraken_pair("PRIMEUSD")
                .with_liquidity_tier(4),
        );

        // ============================================
        // Other Notable Assets
        // ============================================
        self.register(
            AssetConfig::new("TON", AssetCategory::Other)
                .with_kraken_pair("TONUSD")
                .with_liquidity_tier(2),
        );
        self.register(
            AssetConfig::new("TRX", AssetCategory::Other)
                .with_kraken_pair("TRXUSD")
                .with_liquidity_tier(2),
        );
        self.register(
            AssetConfig::new("ETC", AssetCategory::Other)
                .with_kraken_pair("XETCZUSD")
                .with_liquidity_tier(2),
        );
        self.register(
            AssetConfig::new("ZEC", AssetCategory::Other)
                .with_kraken_pair("XZECZUSD")
                .with_liquidity_tier(3),
        );
        self.register(
            AssetConfig::new("DASH", AssetCategory::Other)
                .with_kraken_pair("DASHUSD")
                .with_liquidity_tier(3),
        );
        self.register(
            AssetConfig::new("EOS", AssetCategory::Other)
                .with_kraken_pair("EOSUSD")
                .with_liquidity_tier(3),
        );
        self.register(
            AssetConfig::new("XTZ", AssetCategory::Other)
                .with_kraken_pair("XTZUSD")
                .with_liquidity_tier(3),
        );
        self.register(
            AssetConfig::new("KAVA", AssetCategory::Other)
                .with_kraken_pair("KAVAUSD")
                .with_liquidity_tier(3),
        );
        self.register(
            AssetConfig::new("RUNE", AssetCategory::Other)
                .with_kraken_pair("RUNEUSD")
                .with_liquidity_tier(3),
        );
        self.register(
            AssetConfig::new("CHZ", AssetCategory::Other)
                .with_kraken_pair("CHZUSD")
                .with_liquidity_tier(3),
        );
        self.register(
            AssetConfig::new("FLOW", AssetCategory::Other)
                .with_kraken_pair("FLOWUSD")
                .with_liquidity_tier(3),
        );
        self.register(
            AssetConfig::new("KSM", AssetCategory::Other)
                .with_kraken_pair("KSMUSD")
                .with_liquidity_tier(3),
        );
        self.register(
            AssetConfig::new("MINA", AssetCategory::Other)
                .with_kraken_pair("MINAUSD")
                .with_liquidity_tier(3),
        );
        self.register(
            AssetConfig::new("ROSE", AssetCategory::Other)
                .with_kraken_pair("ROSEUSD")
                .with_liquidity_tier(4),
        );

        // ============================================
        // Forex Pairs
        // ============================================
        self.register(
            AssetConfig::new("EURUSD", AssetCategory::Forex)
                .with_kraken_pair("EURUSD")
                .with_liquidity_tier(1)
                .with_max_position(1000.0),
        );
        self.register(
            AssetConfig::new("GBPUSD", AssetCategory::Forex)
                .with_kraken_pair("GBPUSD")
                .with_liquidity_tier(1)
                .with_max_position(1000.0),
        );
        self.register(
            AssetConfig::new("USDJPY", AssetCategory::Forex)
                .with_kraken_pair("USDJPY")
                .with_liquidity_tier(1)
                .with_max_position(1000.0),
        );
        self.register(
            AssetConfig::new("USDCHF", AssetCategory::Forex)
                .with_kraken_pair("USDCHF")
                .with_liquidity_tier(2),
        );
        self.register(
            AssetConfig::new("USDCAD", AssetCategory::Forex)
                .with_kraken_pair("USDCAD")
                .with_liquidity_tier(2),
        );
        self.register(
            AssetConfig::new("AUDUSD", AssetCategory::Forex)
                .with_kraken_pair("AUDUSD")
                .with_liquidity_tier(2),
        );
        self.register(
            AssetConfig::new("EURGBP", AssetCategory::Forex)
                .with_kraken_pair("EURGBP")
                .with_liquidity_tier(2),
        );
        self.register(
            AssetConfig::new("EURCHF", AssetCategory::Forex)
                .with_kraken_pair("EURCHF")
                .with_liquidity_tier(2),
        );
        self.register(
            AssetConfig::new("EURJPY", AssetCategory::Forex)
                .with_kraken_pair("EURJPY")
                .with_liquidity_tier(2),
        );
        self.register(
            AssetConfig::new("EURCAD", AssetCategory::Forex)
                .with_kraken_pair("EURCAD")
                .with_liquidity_tier(3),
        );
        self.register(
            AssetConfig::new("GBPJPY", AssetCategory::Forex)
                .with_kraken_pair("GBPJPY")
                .with_liquidity_tier(2),
        );
        self.register(
            AssetConfig::new("AUDJPY", AssetCategory::Forex)
                .with_kraken_pair("AUDJPY")
                .with_liquidity_tier(3),
        );
        self.register(
            AssetConfig::new("NZDUSD", AssetCategory::Forex)
                .with_kraken_pair("NZDUSD")
                .with_liquidity_tier(3),
        );
        self.register(
            AssetConfig::new("CADJPY", AssetCategory::Forex)
                .with_kraken_pair("CADJPY")
                .with_liquidity_tier(3),
        );
    }
}

impl Default for AssetRegistry {
    fn default() -> Self {
        Self::with_kraken_defaults()
    }
}

/// Predefined sets of assets for quick access
pub mod asset_sets {

    /// Primary assets (always enabled) - highest liquidity
    pub const PRIMARY: &[&str] = &[
        "BTC", "ETH", "SOL", "XRP", "ADA", "AVAX", "DOT", "LINK", "DOGE",
    ];

    /// Secondary assets (high priority)
    pub const SECONDARY: &[&str] = &[
        "LTC", "BCH", "ATOM", "NEAR", "FIL", "UNI", "AAVE", "ARB", "OP", "POL",
    ];

    /// Tertiary assets (medium priority)
    pub const TERTIARY: &[&str] = &[
        "SHIB", "PEPE", "BONK", "WIF", "RENDER", "FET", "TAO", "APT", "SUI", "INJ",
    ];

    /// Watch list (enable selectively)
    pub const WATCH_LIST: &[&str] = &[
        "FLOKI", "TURBO", "MEW", "PONKE", "VIRTUAL", "TIA", "STRK", "LDO", "CRV", "MKR",
    ];

    /// Get all primary and secondary assets
    pub fn core_assets() -> Vec<&'static str> {
        PRIMARY.iter().chain(SECONDARY.iter()).copied().collect()
    }

    /// Get all assets
    pub fn all_assets() -> Vec<&'static str> {
        PRIMARY
            .iter()
            .chain(SECONDARY.iter())
            .chain(TERTIARY.iter())
            .chain(WATCH_LIST.iter())
            .copied()
            .collect()
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_asset_category_defaults() {
        assert_eq!(AssetCategory::Major.min_ema_spread_pct(), 0.15);
        assert_eq!(AssetCategory::Meme.min_ema_spread_pct(), 0.30);
        assert_eq!(AssetCategory::Major.min_hold_minutes(), 15);
        assert_eq!(AssetCategory::Meme.min_hold_minutes(), 30);
    }

    #[test]
    fn test_asset_config_creation() {
        let config = AssetConfig::new("BTC", AssetCategory::Major)
            .with_kraken_pair("XXBTZUSD")
            .with_liquidity_tier(1);

        assert_eq!(config.symbol, "BTC");
        assert_eq!(config.category, AssetCategory::Major);
        assert_eq!(config.kraken_pair, Some("XXBTZUSD".to_string()));
        assert_eq!(config.liquidity_tier, 1);
        assert!(config.enabled);
    }

    #[test]
    fn test_asset_config_custom_overrides() {
        let config = AssetConfig::new("CUSTOM", AssetCategory::Major)
            .with_min_ema_spread(0.50)
            .with_min_hold_minutes(60);

        assert_eq!(config.min_ema_spread_pct(), 0.50);
        assert_eq!(config.min_hold_minutes(), 60);
    }

    #[test]
    fn test_asset_config_uses_category_defaults() {
        let config = AssetConfig::new("BTC", AssetCategory::Major);
        assert_eq!(config.min_ema_spread_pct(), 0.15); // From Major category
        assert_eq!(config.min_hold_minutes(), 15); // From Major category
    }

    #[test]
    fn test_asset_registry_creation() {
        let registry = AssetRegistry::with_kraken_defaults();
        assert!(!registry.is_empty());
        assert!(registry.contains("BTC"));
        assert!(registry.contains("btc")); // Case insensitive
        assert!(registry.contains("ETH"));
        assert!(registry.contains("DOGE"));
    }

    #[test]
    fn test_asset_registry_get() {
        let registry = AssetRegistry::with_kraken_defaults();

        let btc = registry.get("BTC");
        assert_eq!(btc.symbol, "BTC");
        assert_eq!(btc.category, AssetCategory::Major);

        let doge = registry.get("DOGE");
        assert_eq!(doge.category, AssetCategory::Meme);

        // Unknown asset returns default
        let unknown = registry.get("UNKNOWN123");
        assert_eq!(unknown.symbol, "UNKNOWN123");
        assert_eq!(unknown.category, AssetCategory::Other);
    }

    #[test]
    fn test_asset_registry_by_category() {
        let registry = AssetRegistry::with_kraken_defaults();

        let majors = registry.by_category(AssetCategory::Major);
        assert!(!majors.is_empty());
        assert!(majors.iter().any(|a| a.symbol == "BTC"));

        let memes = registry.by_category(AssetCategory::Meme);
        assert!(!memes.is_empty());
        assert!(memes.iter().any(|a| a.symbol == "DOGE"));
    }

    #[test]
    fn test_asset_registry_by_liquidity() {
        let registry = AssetRegistry::with_kraken_defaults();

        let tier1 = registry.by_liquidity_tier(1);
        assert!(!tier1.is_empty());
        assert!(tier1.iter().any(|a| a.symbol == "BTC"));
        assert!(tier1.iter().any(|a| a.symbol == "ETH"));
    }

    #[test]
    fn test_asset_category_display() {
        assert_eq!(AssetCategory::Major.to_string(), "Major");
        assert_eq!(AssetCategory::Meme.to_string(), "Meme");
        assert_eq!(AssetCategory::L1L2.to_string(), "L1/L2");
    }

    #[test]
    fn test_asset_sets() {
        use asset_sets::*;

        assert!(PRIMARY.contains(&"BTC"));
        assert!(PRIMARY.contains(&"ETH"));
        assert!(SECONDARY.contains(&"LTC"));
        assert!(TERTIARY.contains(&"PEPE"));

        let core = core_assets();
        assert!(core.contains(&"BTC"));
        assert!(core.contains(&"LTC"));
    }

    #[test]
    fn test_category_atr_ranges() {
        let (min, max) = AssetCategory::Major.atr_multiplier_range();
        assert!(min < max);
        assert!(min >= 1.0);

        let (min, _max) = AssetCategory::Meme.atr_multiplier_range();
        assert!(min >= 2.0); // Meme coins need wider stops
    }

    #[test]
    fn test_category_take_profit_ranges() {
        let (min, max) = AssetCategory::Major.take_profit_range();
        assert!(min < max);

        let (meme_min, _) = AssetCategory::Meme.take_profit_range();
        let (major_min, _) = AssetCategory::Major.take_profit_range();
        assert!(meme_min > major_min); // Meme coins have higher TP targets
    }
}
