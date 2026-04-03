//! Declarative memory for factual market knowledge
//!
//! Part of the Cortex region
//! Component: memory
//!
//! Stores and retrieves factual knowledge about markets: asset characteristics
//! (sector, asset class, liquidity tier), pairwise correlation structures,
//! historical event catalogs, and regime labels. Provides query interfaces
//! for upstream planners and strategy modules to look up market facts without
//! re-deriving them each cycle.
//!
//! Key features:
//! - Asset fact registry (sector, class, liquidity, typical vol, beta)
//! - Pairwise correlation matrix storage with age-based decay
//! - Historical event catalog (crashes, rallies, regime shifts) with tagging
//! - Regime label store (bull, bear, ranging, crisis) with timestamps
//! - Fact versioning: each update is timestamped so stale facts can be pruned
//! - Query-by-tag and query-by-asset interfaces
//! - EMA-smoothed tracking of fact freshness and query rates
//! - Sliding window of recent fact updates for audit
//! - Running statistics with hit/miss rates and staleness tracking

use crate::common::{Error, Result};
use std::collections::VecDeque;

// ---------------------------------------------------------------------------
// Configuration
// ---------------------------------------------------------------------------

/// Configuration for the declarative memory store
#[derive(Debug, Clone)]
pub struct DeclarativeConfig {
    /// Maximum number of asset facts to store
    pub max_assets: usize,
    /// Maximum number of events in the catalog
    pub max_events: usize,
    /// Maximum number of regime labels to retain (history)
    pub max_regime_history: usize,
    /// Staleness threshold: facts older than this many steps are considered stale
    pub staleness_threshold: u64,
    /// EMA decay factor for tracking metrics (0 < decay < 1)
    pub ema_decay: f64,
    /// Sliding window size for recent updates
    pub window_size: usize,
    /// Maximum number of correlation pairs to store
    pub max_correlation_pairs: usize,
    /// Whether to automatically prune stale facts
    pub auto_prune_stale: bool,
}

impl Default for DeclarativeConfig {
    fn default() -> Self {
        Self {
            max_assets: 500,
            max_events: 1000,
            max_regime_history: 200,
            staleness_threshold: 5000,
            ema_decay: 0.1,
            window_size: 100,
            max_correlation_pairs: 10000,
            auto_prune_stale: false,
        }
    }
}

// ---------------------------------------------------------------------------
// Domain types
// ---------------------------------------------------------------------------

/// Asset class classification
#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash)]
pub enum AssetClass {
    /// Equities / stocks
    Equity,
    /// Fixed income / bonds
    FixedIncome,
    /// Commodities (metals, energy, agriculture)
    Commodity,
    /// Foreign exchange
    Forex,
    /// Cryptocurrency
    Crypto,
    /// Derivatives (options, futures)
    Derivative,
    /// Other / unclassified
    Other,
}

/// Liquidity tier classification
#[derive(Debug, Clone, Copy, PartialEq, Eq, PartialOrd, Ord, Hash)]
pub enum LiquidityTier {
    /// Highly liquid: tight spreads, deep books
    High,
    /// Moderate liquidity
    Medium,
    /// Low liquidity: wide spreads, potential slippage
    Low,
    /// Illiquid: may not be tradable at desired size
    VeryLow,
}

impl LiquidityTier {
    /// Numeric weight (higher = more liquid)
    pub fn weight(&self) -> f64 {
        match self {
            LiquidityTier::High => 4.0,
            LiquidityTier::Medium => 3.0,
            LiquidityTier::Low => 2.0,
            LiquidityTier::VeryLow => 1.0,
        }
    }
}

/// Sector classification for equities
#[derive(Debug, Clone, PartialEq, Eq, Hash)]
pub enum Sector {
    Technology,
    Healthcare,
    Financials,
    ConsumerDiscretionary,
    ConsumerStaples,
    Industrials,
    Energy,
    Materials,
    Utilities,
    RealEstate,
    Communication,
    Other(String),
}

/// Factual record about a single asset
#[derive(Debug, Clone)]
pub struct AssetFact {
    /// Unique asset identifier (ticker / symbol)
    pub asset_id: String,
    /// Human-readable name
    pub name: String,
    /// Asset class
    pub asset_class: AssetClass,
    /// Sector (primarily for equities)
    pub sector: Sector,
    /// Liquidity tier
    pub liquidity: LiquidityTier,
    /// Typical annualised volatility
    pub typical_volatility: f64,
    /// Beta relative to a broad market benchmark
    pub beta: f64,
    /// Average daily volume (notional, approximate)
    pub avg_daily_volume: f64,
    /// Typical bid-ask spread as fraction of mid price
    pub typical_spread: f64,
    /// Market capitalisation or notional outstanding (approximate)
    pub market_cap: f64,
    /// Whether the asset is currently tradable
    pub tradable: bool,
    /// Step at which this fact was last updated
    pub last_updated: u64,
    /// Tags for flexible categorisation
    pub tags: Vec<String>,
}

impl AssetFact {
    /// Create a minimal asset fact
    pub fn new(asset_id: impl Into<String>, asset_class: AssetClass) -> Self {
        Self {
            asset_id: asset_id.into(),
            name: String::new(),
            asset_class,
            sector: Sector::Other("unknown".into()),
            liquidity: LiquidityTier::Medium,
            typical_volatility: 0.20,
            beta: 1.0,
            avg_daily_volume: 0.0,
            typical_spread: 0.001,
            market_cap: 0.0,
            tradable: true,
            last_updated: 0,
            tags: Vec::new(),
        }
    }

    /// Whether this fact is stale relative to the given current step
    pub fn is_stale(&self, current_step: u64, threshold: u64) -> bool {
        current_step.saturating_sub(self.last_updated) > threshold
    }
}

/// A pairwise correlation record between two assets
#[derive(Debug, Clone)]
pub struct CorrelationRecord {
    /// First asset ID
    pub asset_a: String,
    /// Second asset ID
    pub asset_b: String,
    /// Estimated correlation coefficient (-1 to 1)
    pub correlation: f64,
    /// Lookback period in trading days used to compute this estimate
    pub lookback_days: u32,
    /// Step at which this correlation was last updated
    pub last_updated: u64,
    /// Confidence score (0–1): higher = more data points, more stable
    pub confidence: f64,
}

impl CorrelationRecord {
    /// Canonical key for this pair (sorted alphabetically)
    pub fn key(&self) -> (String, String) {
        if self.asset_a <= self.asset_b {
            (self.asset_a.clone(), self.asset_b.clone())
        } else {
            (self.asset_b.clone(), self.asset_a.clone())
        }
    }

    /// Whether this record is stale
    pub fn is_stale(&self, current_step: u64, threshold: u64) -> bool {
        current_step.saturating_sub(self.last_updated) > threshold
    }
}

/// Type of historical market event
#[derive(Debug, Clone, PartialEq, Eq, Hash)]
pub enum EventType {
    /// Market crash / sharp drawdown
    Crash,
    /// Strong rally
    Rally,
    /// Regime shift (e.g., from bull to bear)
    RegimeShift,
    /// Central bank action (rate decision, QE announcement)
    CentralBankAction,
    /// Geopolitical event
    Geopolitical,
    /// Earnings / corporate event
    CorporateEvent,
    /// Liquidity event (flash crash, circuit breaker)
    LiquidityEvent,
    /// Custom event type
    Custom(String),
}

/// Impact severity of an event
#[derive(Debug, Clone, Copy, PartialEq, Eq, PartialOrd, Ord, Hash)]
pub enum EventSeverity {
    /// Minor impact
    Low,
    /// Moderate impact
    Medium,
    /// Significant impact
    High,
    /// Extreme / tail-event impact
    Extreme,
}

impl EventSeverity {
    /// Numeric weight
    pub fn weight(&self) -> f64 {
        match self {
            EventSeverity::Low => 1.0,
            EventSeverity::Medium => 2.0,
            EventSeverity::High => 3.0,
            EventSeverity::Extreme => 4.0,
        }
    }
}

/// A historical market event record
#[derive(Debug, Clone)]
pub struct MarketEvent {
    /// Unique event identifier
    pub id: String,
    /// Event type
    pub event_type: EventType,
    /// Severity classification
    pub severity: EventSeverity,
    /// Human-readable description
    pub description: String,
    /// Step at which the event occurred
    pub step: u64,
    /// Affected asset IDs (empty = market-wide)
    pub affected_assets: Vec<String>,
    /// Estimated return impact (negative for crashes)
    pub return_impact: f64,
    /// Estimated volatility impact (positive = vol increase)
    pub volatility_impact: f64,
    /// Duration in steps (0 = instantaneous)
    pub duration_steps: u64,
    /// Tags for flexible search
    pub tags: Vec<String>,
}

/// Market regime label
#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash)]
pub enum RegimeLabel {
    Bull,
    Bear,
    Ranging,
    Crisis,
    Recovery,
}

impl RegimeLabel {
    /// Numeric index for array-based tracking
    pub fn index(&self) -> usize {
        match self {
            RegimeLabel::Bull => 0,
            RegimeLabel::Bear => 1,
            RegimeLabel::Ranging => 2,
            RegimeLabel::Crisis => 3,
            RegimeLabel::Recovery => 4,
        }
    }

    /// From numeric index
    pub fn from_index(idx: usize) -> Self {
        match idx {
            0 => RegimeLabel::Bull,
            1 => RegimeLabel::Bear,
            2 => RegimeLabel::Ranging,
            3 => RegimeLabel::Crisis,
            4 => RegimeLabel::Recovery,
            _ => RegimeLabel::Ranging,
        }
    }

    /// Number of regime variants
    pub const COUNT: usize = 5;
}

/// A timestamped regime label entry
#[derive(Debug, Clone)]
pub struct RegimeEntry {
    /// Regime label
    pub regime: RegimeLabel,
    /// Step at which this regime was recorded
    pub step: u64,
    /// Confidence in the classification (0–1)
    pub confidence: f64,
    /// Optional notes
    pub notes: String,
}

/// A lightweight record of a fact update for the sliding window
#[derive(Debug, Clone)]
pub struct UpdateRecord {
    /// What was updated: "asset", "correlation", "event", "regime"
    pub category: String,
    /// ID of the entity updated
    pub entity_id: String,
    /// Step at which the update occurred
    pub step: u64,
}

/// Result of a query operation
#[derive(Debug, Clone)]
pub struct QueryResult<T: Clone> {
    /// Items matching the query
    pub items: Vec<T>,
    /// Number of items scanned
    pub scanned: usize,
    /// Whether any items were stale (and thus potentially unreliable)
    pub has_stale: bool,
}

// ---------------------------------------------------------------------------
// Running statistics
// ---------------------------------------------------------------------------

/// Running statistics for the declarative memory store
#[derive(Debug, Clone)]
pub struct DeclarativeStats {
    /// Total updates across all categories
    pub total_updates: u64,
    /// Total queries processed
    pub total_queries: u64,
    /// Total cache hits (query found results)
    pub query_hits: u64,
    /// Total cache misses (query found nothing)
    pub query_misses: u64,
    /// Total stale facts pruned
    pub total_pruned: u64,
    /// EMA of query hit rate
    pub ema_hit_rate: f64,
    /// EMA of fact freshness (average age of facts in steps)
    pub ema_avg_freshness: f64,
    /// EMA of updates per evaluation cycle
    pub ema_update_rate: f64,
    /// Current asset count
    pub asset_count: usize,
    /// Current correlation pair count
    pub correlation_count: usize,
    /// Current event count
    pub event_count: usize,
    /// Current regime history length
    pub regime_history_length: usize,
}

impl Default for DeclarativeStats {
    fn default() -> Self {
        Self {
            total_updates: 0,
            total_queries: 0,
            query_hits: 0,
            query_misses: 0,
            total_pruned: 0,
            ema_hit_rate: 0.5,
            ema_avg_freshness: 0.0,
            ema_update_rate: 0.0,
            asset_count: 0,
            correlation_count: 0,
            event_count: 0,
            regime_history_length: 0,
        }
    }
}

impl DeclarativeStats {
    /// Overall query hit rate
    pub fn hit_rate(&self) -> f64 {
        let total = self.query_hits + self.query_misses;
        if total == 0 {
            return 0.0;
        }
        self.query_hits as f64 / total as f64
    }

    /// Total fact count across all categories
    pub fn total_facts(&self) -> usize {
        self.asset_count + self.correlation_count + self.event_count + self.regime_history_length
    }
}

// ---------------------------------------------------------------------------
// Declarative Memory Engine
// ---------------------------------------------------------------------------

/// Declarative memory store.
///
/// Holds factual knowledge about the market: asset characteristics,
/// pairwise correlations, event catalogs, and regime history. Provides
/// insert, update, query, and pruning operations with statistics tracking.
pub struct Declarative {
    config: DeclarativeConfig,
    /// Asset fact registry
    assets: Vec<AssetFact>,
    /// Pairwise correlation records
    correlations: Vec<CorrelationRecord>,
    /// Historical event catalog
    events: Vec<MarketEvent>,
    /// Regime history (most recent last)
    regime_history: VecDeque<RegimeEntry>,
    /// Current step counter
    step: u64,
    /// EMA initialized flag
    ema_initialized: bool,
    /// Sliding window of recent updates
    recent_updates: VecDeque<UpdateRecord>,
    /// Running statistics
    stats: DeclarativeStats,
    /// Updates in the current evaluation cycle (for EMA tracking)
    cycle_updates: u64,
}

impl Default for Declarative {
    fn default() -> Self {
        Self::new()
    }
}

impl Declarative {
    /// Create a new declarative memory store with default configuration
    pub fn new() -> Self {
        Self {
            config: DeclarativeConfig::default(),
            assets: Vec::new(),
            correlations: Vec::new(),
            events: Vec::new(),
            regime_history: VecDeque::new(),
            step: 0,
            ema_initialized: false,
            recent_updates: VecDeque::new(),
            stats: DeclarativeStats::default(),
            cycle_updates: 0,
        }
    }

    /// Create with explicit configuration
    pub fn with_config(config: DeclarativeConfig) -> Result<Self> {
        if config.max_assets == 0 {
            return Err(Error::InvalidInput("max_assets must be > 0".into()));
        }
        if config.max_events == 0 {
            return Err(Error::InvalidInput("max_events must be > 0".into()));
        }
        if config.max_regime_history == 0 {
            return Err(Error::InvalidInput("max_regime_history must be > 0".into()));
        }
        if config.ema_decay <= 0.0 || config.ema_decay >= 1.0 {
            return Err(Error::InvalidInput("ema_decay must be in (0, 1)".into()));
        }
        if config.window_size == 0 {
            return Err(Error::InvalidInput("window_size must be > 0".into()));
        }
        if config.max_correlation_pairs == 0 {
            return Err(Error::InvalidInput(
                "max_correlation_pairs must be > 0".into(),
            ));
        }
        Ok(Self {
            config,
            assets: Vec::new(),
            correlations: Vec::new(),
            events: Vec::new(),
            regime_history: VecDeque::new(),
            step: 0,
            ema_initialized: false,
            recent_updates: VecDeque::new(),
            stats: DeclarativeStats::default(),
            cycle_updates: 0,
        })
    }

    /// Main processing function (trait conformance entry point)
    pub fn process(&self) -> Result<()> {
        Ok(())
    }

    /// Advance the internal step counter and update cycle metrics.
    ///
    /// Call this once per evaluation cycle to track freshness and update rates.
    pub fn tick(&mut self) {
        self.step += 1;

        // EMA update for update rate
        let alpha = self.config.ema_decay;
        let rate = self.cycle_updates as f64;
        if !self.ema_initialized {
            self.stats.ema_update_rate = rate;
            self.ema_initialized = true;
        } else {
            self.stats.ema_update_rate = alpha * rate + (1.0 - alpha) * self.stats.ema_update_rate;
        }

        // Update freshness EMA
        let avg_freshness = self.compute_avg_freshness();
        if self.ema_initialized {
            self.stats.ema_avg_freshness =
                alpha * avg_freshness + (1.0 - alpha) * self.stats.ema_avg_freshness;
        } else {
            self.stats.ema_avg_freshness = avg_freshness;
        }

        self.cycle_updates = 0;

        // Auto-prune if configured
        if self.config.auto_prune_stale {
            self.prune_stale();
        }

        self.sync_counts();
    }

    fn sync_counts(&mut self) {
        self.stats.asset_count = self.assets.len();
        self.stats.correlation_count = self.correlations.len();
        self.stats.event_count = self.events.len();
        self.stats.regime_history_length = self.regime_history.len();
    }

    fn compute_avg_freshness(&self) -> f64 {
        let mut total_age = 0u64;
        let mut count = 0u64;
        for a in &self.assets {
            total_age += self.step.saturating_sub(a.last_updated);
            count += 1;
        }
        for c in &self.correlations {
            total_age += self.step.saturating_sub(c.last_updated);
            count += 1;
        }
        if count == 0 {
            return 0.0;
        }
        total_age as f64 / count as f64
    }

    fn record_update(&mut self, category: &str, entity_id: &str) {
        self.stats.total_updates += 1;
        self.cycle_updates += 1;

        let record = UpdateRecord {
            category: category.into(),
            entity_id: entity_id.into(),
            step: self.step,
        };
        if self.recent_updates.len() >= self.config.window_size {
            self.recent_updates.pop_front();
        }
        self.recent_updates.push_back(record);
    }

    fn record_query_hit(&mut self) {
        self.stats.total_queries += 1;
        self.stats.query_hits += 1;

        let alpha = self.config.ema_decay;
        self.stats.ema_hit_rate = alpha * 1.0 + (1.0 - alpha) * self.stats.ema_hit_rate;
    }

    fn record_query_miss(&mut self) {
        self.stats.total_queries += 1;
        self.stats.query_misses += 1;

        let alpha = self.config.ema_decay;
        self.stats.ema_hit_rate = alpha * 0.0 + (1.0 - alpha) * self.stats.ema_hit_rate;
    }

    // -----------------------------------------------------------------------
    // Asset facts
    // -----------------------------------------------------------------------

    /// Store or update an asset fact. If the asset ID already exists, it is replaced.
    pub fn store_asset(&mut self, mut fact: AssetFact) -> Result<()> {
        fact.last_updated = self.step;

        if let Some(existing) = self.assets.iter_mut().find(|a| a.asset_id == fact.asset_id) {
            *existing = fact.clone();
            self.record_update("asset", &fact.asset_id);
            return Ok(());
        }

        if self.assets.len() >= self.config.max_assets {
            return Err(Error::ResourceExhausted(format!(
                "max asset count ({}) reached",
                self.config.max_assets
            )));
        }

        let id = fact.asset_id.clone();
        self.assets.push(fact);
        self.record_update("asset", &id);
        self.sync_counts();
        Ok(())
    }

    /// Remove an asset fact by ID
    pub fn remove_asset(&mut self, asset_id: &str) -> Result<()> {
        let idx = self
            .assets
            .iter()
            .position(|a| a.asset_id == asset_id)
            .ok_or_else(|| Error::NotFound(format!("asset '{}' not found", asset_id)))?;
        self.assets.remove(idx);
        self.sync_counts();
        Ok(())
    }

    /// Query an asset fact by ID
    pub fn query_asset(&mut self, asset_id: &str) -> Option<&AssetFact> {
        let found = self.assets.iter().any(|a| a.asset_id == asset_id);
        if found {
            self.record_query_hit();
        } else {
            self.record_query_miss();
        }
        self.assets.iter().find(|a| a.asset_id == asset_id)
    }

    /// Query all assets of a given asset class
    pub fn query_assets_by_class(&mut self, class: AssetClass) -> QueryResult<AssetFact> {
        let threshold = self.config.staleness_threshold;
        let step = self.step;
        let items: Vec<AssetFact> = self
            .assets
            .iter()
            .filter(|a| a.asset_class == class)
            .cloned()
            .collect();
        let has_stale = items.iter().any(|a| a.is_stale(step, threshold));
        let scanned = self.assets.len();

        if items.is_empty() {
            self.record_query_miss();
        } else {
            self.record_query_hit();
        }

        QueryResult {
            items,
            scanned,
            has_stale,
        }
    }

    /// Query all assets matching a given tag
    pub fn query_assets_by_tag(&mut self, tag: &str) -> QueryResult<AssetFact> {
        let threshold = self.config.staleness_threshold;
        let step = self.step;
        let items: Vec<AssetFact> = self
            .assets
            .iter()
            .filter(|a| a.tags.iter().any(|t| t == tag))
            .cloned()
            .collect();
        let has_stale = items.iter().any(|a| a.is_stale(step, threshold));
        let scanned = self.assets.len();

        if items.is_empty() {
            self.record_query_miss();
        } else {
            self.record_query_hit();
        }

        QueryResult {
            items,
            scanned,
            has_stale,
        }
    }

    /// Query assets by sector
    pub fn query_assets_by_sector(&mut self, sector: &Sector) -> QueryResult<AssetFact> {
        let threshold = self.config.staleness_threshold;
        let step = self.step;
        let items: Vec<AssetFact> = self
            .assets
            .iter()
            .filter(|a| a.sector == *sector)
            .cloned()
            .collect();
        let has_stale = items.iter().any(|a| a.is_stale(step, threshold));
        let scanned = self.assets.len();

        if items.is_empty() {
            self.record_query_miss();
        } else {
            self.record_query_hit();
        }

        QueryResult {
            items,
            scanned,
            has_stale,
        }
    }

    /// Query assets by liquidity tier (returns all assets at or above the given tier)
    pub fn query_assets_by_min_liquidity(
        &mut self,
        min_tier: LiquidityTier,
    ) -> QueryResult<AssetFact> {
        let threshold = self.config.staleness_threshold;
        let step = self.step;
        let items: Vec<AssetFact> = self
            .assets
            .iter()
            .filter(|a| a.liquidity <= min_tier) // Ord: High < Medium < Low < VeryLow
            .cloned()
            .collect();
        let has_stale = items.iter().any(|a| a.is_stale(step, threshold));
        let scanned = self.assets.len();

        if items.is_empty() {
            self.record_query_miss();
        } else {
            self.record_query_hit();
        }

        QueryResult {
            items,
            scanned,
            has_stale,
        }
    }

    /// Get all stored assets
    pub fn all_assets(&self) -> &[AssetFact] {
        &self.assets
    }

    /// Number of stored asset facts
    pub fn asset_count(&self) -> usize {
        self.assets.len()
    }

    // -----------------------------------------------------------------------
    // Correlations
    // -----------------------------------------------------------------------

    /// Store or update a correlation record. Uses canonical key (sorted pair).
    pub fn store_correlation(&mut self, mut record: CorrelationRecord) -> Result<()> {
        if record.correlation < -1.0 || record.correlation > 1.0 {
            return Err(Error::InvalidInput("correlation must be in [-1, 1]".into()));
        }
        record.last_updated = self.step;
        let key = record.key();

        if let Some(existing) = self.correlations.iter_mut().find(|c| c.key() == key) {
            *existing = record;
            self.record_update("correlation", &format!("{}:{}", key.0, key.1));
            return Ok(());
        }

        if self.correlations.len() >= self.config.max_correlation_pairs {
            return Err(Error::ResourceExhausted(format!(
                "max correlation pairs ({}) reached",
                self.config.max_correlation_pairs
            )));
        }

        let label = format!("{}:{}", key.0, key.1);
        self.correlations.push(record);
        self.record_update("correlation", &label);
        self.sync_counts();
        Ok(())
    }

    /// Query correlation between two assets
    pub fn query_correlation(
        &mut self,
        asset_a: &str,
        asset_b: &str,
    ) -> Option<&CorrelationRecord> {
        let key = if asset_a <= asset_b {
            (asset_a.to_string(), asset_b.to_string())
        } else {
            (asset_b.to_string(), asset_a.to_string())
        };

        let found = self.correlations.iter().any(|c| c.key() == key);
        if found {
            self.record_query_hit();
        } else {
            self.record_query_miss();
        }
        self.correlations.iter().find(|c| c.key() == key)
    }

    /// Query all correlations involving a given asset
    pub fn query_correlations_for_asset(
        &mut self,
        asset_id: &str,
    ) -> QueryResult<CorrelationRecord> {
        let threshold = self.config.staleness_threshold;
        let step = self.step;
        let items: Vec<CorrelationRecord> = self
            .correlations
            .iter()
            .filter(|c| c.asset_a == asset_id || c.asset_b == asset_id)
            .cloned()
            .collect();
        let has_stale = items.iter().any(|c| c.is_stale(step, threshold));
        let scanned = self.correlations.len();

        if items.is_empty() {
            self.record_query_miss();
        } else {
            self.record_query_hit();
        }

        QueryResult {
            items,
            scanned,
            has_stale,
        }
    }

    /// Query all high-correlation pairs above a given threshold
    pub fn query_high_correlations(&mut self, threshold: f64) -> QueryResult<CorrelationRecord> {
        let stale_threshold = self.config.staleness_threshold;
        let step = self.step;
        let items: Vec<CorrelationRecord> = self
            .correlations
            .iter()
            .filter(|c| c.correlation.abs() >= threshold)
            .cloned()
            .collect();
        let has_stale = items.iter().any(|c| c.is_stale(step, stale_threshold));
        let scanned = self.correlations.len();

        if items.is_empty() {
            self.record_query_miss();
        } else {
            self.record_query_hit();
        }

        QueryResult {
            items,
            scanned,
            has_stale,
        }
    }

    /// Number of stored correlation records
    pub fn correlation_count(&self) -> usize {
        self.correlations.len()
    }

    /// Average absolute correlation across all pairs
    pub fn avg_abs_correlation(&self) -> f64 {
        if self.correlations.is_empty() {
            return 0.0;
        }
        let sum: f64 = self.correlations.iter().map(|c| c.correlation.abs()).sum();
        sum / self.correlations.len() as f64
    }

    // -----------------------------------------------------------------------
    // Event catalog
    // -----------------------------------------------------------------------

    /// Record a new market event
    pub fn record_event(&mut self, mut event: MarketEvent) -> Result<()> {
        if event.id.is_empty() {
            return Err(Error::InvalidInput("event id must not be empty".into()));
        }
        // Check for duplicate ID
        if self.events.iter().any(|e| e.id == event.id) {
            return Err(Error::InvalidInput(format!(
                "event with id '{}' already exists",
                event.id
            )));
        }

        event.step = self.step;

        // Evict oldest if at capacity
        if self.events.len() >= self.config.max_events {
            self.events.remove(0);
        }

        let id = event.id.clone();
        self.events.push(event);
        self.record_update("event", &id);
        self.sync_counts();
        Ok(())
    }

    /// Query events by type
    pub fn query_events_by_type(&mut self, event_type: &EventType) -> QueryResult<MarketEvent> {
        let items: Vec<MarketEvent> = self
            .events
            .iter()
            .filter(|e| e.event_type == *event_type)
            .cloned()
            .collect();
        let scanned = self.events.len();

        if items.is_empty() {
            self.record_query_miss();
        } else {
            self.record_query_hit();
        }

        QueryResult {
            items,
            scanned,
            has_stale: false,
        }
    }

    /// Query events by severity (at or above given level)
    pub fn query_events_by_min_severity(
        &mut self,
        min_severity: EventSeverity,
    ) -> QueryResult<MarketEvent> {
        let items: Vec<MarketEvent> = self
            .events
            .iter()
            .filter(|e| e.severity >= min_severity)
            .cloned()
            .collect();
        let scanned = self.events.len();

        if items.is_empty() {
            self.record_query_miss();
        } else {
            self.record_query_hit();
        }

        QueryResult {
            items,
            scanned,
            has_stale: false,
        }
    }

    /// Query events affecting a specific asset
    pub fn query_events_for_asset(&mut self, asset_id: &str) -> QueryResult<MarketEvent> {
        let items: Vec<MarketEvent> = self
            .events
            .iter()
            .filter(|e| {
                e.affected_assets.is_empty() || e.affected_assets.iter().any(|a| a == asset_id)
            })
            .cloned()
            .collect();
        let scanned = self.events.len();

        if items.is_empty() {
            self.record_query_miss();
        } else {
            self.record_query_hit();
        }

        QueryResult {
            items,
            scanned,
            has_stale: false,
        }
    }

    /// Query events by tag
    pub fn query_events_by_tag(&mut self, tag: &str) -> QueryResult<MarketEvent> {
        let items: Vec<MarketEvent> = self
            .events
            .iter()
            .filter(|e| e.tags.iter().any(|t| t == tag))
            .cloned()
            .collect();
        let scanned = self.events.len();

        if items.is_empty() {
            self.record_query_miss();
        } else {
            self.record_query_hit();
        }

        QueryResult {
            items,
            scanned,
            has_stale: false,
        }
    }

    /// Query recent events (last N by step)
    pub fn recent_events(&self, count: usize) -> Vec<&MarketEvent> {
        let n = self.events.len();
        let start = n.saturating_sub(count);
        self.events[start..].iter().collect()
    }

    /// Number of stored events
    pub fn event_count(&self) -> usize {
        self.events.len()
    }

    /// Count events by severity
    pub fn event_count_by_severity(&self, severity: EventSeverity) -> usize {
        self.events
            .iter()
            .filter(|e| e.severity == severity)
            .count()
    }

    // -----------------------------------------------------------------------
    // Regime history
    // -----------------------------------------------------------------------

    /// Record a regime label
    pub fn record_regime(
        &mut self,
        regime: RegimeLabel,
        confidence: f64,
        notes: impl Into<String>,
    ) {
        let entry = RegimeEntry {
            regime,
            step: self.step,
            confidence: confidence.clamp(0.0, 1.0),
            notes: notes.into(),
        };

        if self.regime_history.len() >= self.config.max_regime_history {
            self.regime_history.pop_front();
        }
        self.regime_history.push_back(entry);
        self.record_update("regime", &format!("{:?}", regime));
        self.sync_counts();
    }

    /// Get the current (most recent) regime
    pub fn current_regime(&self) -> Option<&RegimeEntry> {
        self.regime_history.back()
    }

    /// Get the full regime history
    pub fn regime_history(&self) -> &VecDeque<RegimeEntry> {
        &self.regime_history
    }

    /// Count of regime transitions (entries where regime differs from previous)
    pub fn regime_transition_count(&self) -> usize {
        if self.regime_history.len() < 2 {
            return 0;
        }
        self.regime_history
            .iter()
            .zip(self.regime_history.iter().skip(1))
            .filter(|(a, b)| a.regime != b.regime)
            .count()
    }

    /// Fraction of regime history spent in each regime
    pub fn regime_distribution(&self) -> [f64; RegimeLabel::COUNT] {
        let mut counts = [0u64; RegimeLabel::COUNT];
        for entry in &self.regime_history {
            counts[entry.regime.index()] += 1;
        }
        let total = self.regime_history.len() as f64;
        let mut dist = [0.0; RegimeLabel::COUNT];
        if total > 0.0 {
            for (i, c) in counts.iter().enumerate() {
                dist[i] = *c as f64 / total;
            }
        }
        dist
    }

    /// Most common regime in history
    pub fn dominant_regime(&self) -> Option<RegimeLabel> {
        if self.regime_history.is_empty() {
            return None;
        }
        let mut counts = [0u64; RegimeLabel::COUNT];
        for entry in &self.regime_history {
            counts[entry.regime.index()] += 1;
        }
        let max_idx = counts
            .iter()
            .enumerate()
            .max_by_key(|(_, c)| *c)
            .map(|(i, _)| i)
            .unwrap_or(2);
        Some(RegimeLabel::from_index(max_idx))
    }

    // -----------------------------------------------------------------------
    // Pruning
    // -----------------------------------------------------------------------

    /// Prune all stale facts (assets and correlations older than staleness_threshold)
    pub fn prune_stale(&mut self) -> usize {
        let threshold = self.config.staleness_threshold;
        let step = self.step;
        let before = self.assets.len() + self.correlations.len();

        self.assets.retain(|a| !a.is_stale(step, threshold));
        self.correlations.retain(|c| !c.is_stale(step, threshold));

        let after = self.assets.len() + self.correlations.len();
        let pruned = before - after;
        self.stats.total_pruned += pruned as u64;
        self.sync_counts();
        pruned
    }

    /// Count of stale facts at the current step
    pub fn stale_count(&self) -> usize {
        let threshold = self.config.staleness_threshold;
        let step = self.step;
        let stale_assets = self
            .assets
            .iter()
            .filter(|a| a.is_stale(step, threshold))
            .count();
        let stale_corrs = self
            .correlations
            .iter()
            .filter(|c| c.is_stale(step, threshold))
            .count();
        stale_assets + stale_corrs
    }

    // -----------------------------------------------------------------------
    // Accessors
    // -----------------------------------------------------------------------

    /// Running statistics
    pub fn stats(&self) -> &DeclarativeStats {
        &self.stats
    }

    /// Configuration
    pub fn config(&self) -> &DeclarativeConfig {
        &self.config
    }

    /// Current step
    pub fn current_step(&self) -> u64 {
        self.step
    }

    /// Recent update records (sliding window)
    pub fn recent_update_records(&self) -> &VecDeque<UpdateRecord> {
        &self.recent_updates
    }

    /// EMA-smoothed query hit rate
    pub fn smoothed_hit_rate(&self) -> f64 {
        self.stats.ema_hit_rate
    }

    /// EMA-smoothed fact freshness (lower = fresher)
    pub fn smoothed_freshness(&self) -> f64 {
        self.stats.ema_avg_freshness
    }

    /// EMA-smoothed update rate
    pub fn smoothed_update_rate(&self) -> f64 {
        self.stats.ema_update_rate
    }

    /// Windowed update rate (updates in window / window size)
    pub fn windowed_update_rate(&self) -> f64 {
        if self.recent_updates.is_empty() {
            return 0.0;
        }
        self.recent_updates.len() as f64 / self.config.window_size as f64
    }

    /// Reset state (clears all data and statistics)
    pub fn reset(&mut self) {
        self.assets.clear();
        self.correlations.clear();
        self.events.clear();
        self.regime_history.clear();
        self.step = 0;
        self.ema_initialized = false;
        self.recent_updates.clear();
        self.stats = DeclarativeStats::default();
        self.cycle_updates = 0;
    }
}

// ---------------------------------------------------------------------------
// Preset fact builders
// ---------------------------------------------------------------------------

/// Create a sample equity asset fact
pub fn preset_equity_fact(
    ticker: &str,
    name: &str,
    sector: Sector,
    vol: f64,
    beta: f64,
) -> AssetFact {
    let mut fact = AssetFact::new(ticker, AssetClass::Equity);
    fact.name = name.into();
    fact.sector = sector;
    fact.typical_volatility = vol;
    fact.beta = beta;
    fact.liquidity = LiquidityTier::High;
    fact.tradable = true;
    fact
}

/// Create a sample crypto asset fact
pub fn preset_crypto_fact(ticker: &str, name: &str, vol: f64) -> AssetFact {
    let mut fact = AssetFact::new(ticker, AssetClass::Crypto);
    fact.name = name.into();
    fact.sector = Sector::Other("crypto".into());
    fact.typical_volatility = vol;
    fact.beta = 1.5;
    fact.liquidity = LiquidityTier::Medium;
    fact.typical_spread = 0.003;
    fact.tradable = true;
    fact
}

/// Create a standard market crash event
pub fn preset_crash_event(id: &str, description: &str, return_impact: f64) -> MarketEvent {
    MarketEvent {
        id: id.into(),
        event_type: EventType::Crash,
        severity: if return_impact.abs() > 0.20 {
            EventSeverity::Extreme
        } else if return_impact.abs() > 0.10 {
            EventSeverity::High
        } else {
            EventSeverity::Medium
        },
        description: description.into(),
        step: 0,
        affected_assets: Vec::new(),
        return_impact,
        volatility_impact: return_impact.abs() * 2.0,
        duration_steps: 20,
        tags: vec!["crash".into(), "drawdown".into()],
    }
}

// ---------------------------------------------------------------------------
// Tests
// ---------------------------------------------------------------------------

#[cfg(test)]
mod tests {
    use super::*;

    // -- Helpers --

    fn sample_asset(id: &str) -> AssetFact {
        AssetFact::new(id, AssetClass::Equity)
    }

    fn sample_crypto(id: &str) -> AssetFact {
        let mut a = AssetFact::new(id, AssetClass::Crypto);
        a.sector = Sector::Other("crypto".into());
        a.typical_volatility = 0.80;
        a
    }

    fn sample_correlation(a: &str, b: &str, corr: f64) -> CorrelationRecord {
        CorrelationRecord {
            asset_a: a.into(),
            asset_b: b.into(),
            correlation: corr,
            lookback_days: 60,
            last_updated: 0,
            confidence: 0.8,
        }
    }

    fn sample_event(id: &str, event_type: EventType, severity: EventSeverity) -> MarketEvent {
        MarketEvent {
            id: id.into(),
            event_type,
            severity,
            description: format!("Test event {}", id),
            step: 0,
            affected_assets: Vec::new(),
            return_impact: -0.05,
            volatility_impact: 0.10,
            duration_steps: 5,
            tags: vec!["test".into()],
        }
    }

    // -- Construction --

    #[test]
    fn test_new_default() {
        let d = Declarative::new();
        assert_eq!(d.asset_count(), 0);
        assert_eq!(d.correlation_count(), 0);
        assert_eq!(d.event_count(), 0);
        assert_eq!(d.current_step(), 0);
        assert!(d.process().is_ok());
    }

    #[test]
    fn test_with_config() {
        let d = Declarative::with_config(DeclarativeConfig::default());
        assert!(d.is_ok());
    }

    #[test]
    fn test_invalid_config_zero_max_assets() {
        let mut cfg = DeclarativeConfig::default();
        cfg.max_assets = 0;
        assert!(Declarative::with_config(cfg).is_err());
    }

    #[test]
    fn test_invalid_config_zero_max_events() {
        let mut cfg = DeclarativeConfig::default();
        cfg.max_events = 0;
        assert!(Declarative::with_config(cfg).is_err());
    }

    #[test]
    fn test_invalid_config_zero_regime_history() {
        let mut cfg = DeclarativeConfig::default();
        cfg.max_regime_history = 0;
        assert!(Declarative::with_config(cfg).is_err());
    }

    #[test]
    fn test_invalid_config_ema_decay_zero() {
        let mut cfg = DeclarativeConfig::default();
        cfg.ema_decay = 0.0;
        assert!(Declarative::with_config(cfg).is_err());
    }

    #[test]
    fn test_invalid_config_ema_decay_one() {
        let mut cfg = DeclarativeConfig::default();
        cfg.ema_decay = 1.0;
        assert!(Declarative::with_config(cfg).is_err());
    }

    #[test]
    fn test_invalid_config_zero_window() {
        let mut cfg = DeclarativeConfig::default();
        cfg.window_size = 0;
        assert!(Declarative::with_config(cfg).is_err());
    }

    #[test]
    fn test_invalid_config_zero_corr_pairs() {
        let mut cfg = DeclarativeConfig::default();
        cfg.max_correlation_pairs = 0;
        assert!(Declarative::with_config(cfg).is_err());
    }

    // -- Asset facts --

    #[test]
    fn test_store_asset() {
        let mut d = Declarative::new();
        assert!(d.store_asset(sample_asset("AAPL")).is_ok());
        assert_eq!(d.asset_count(), 1);
    }

    #[test]
    fn test_store_asset_updates_existing() {
        let mut d = Declarative::new();
        let mut a1 = sample_asset("AAPL");
        a1.typical_volatility = 0.20;
        d.store_asset(a1).unwrap();

        let mut a2 = sample_asset("AAPL");
        a2.typical_volatility = 0.30;
        d.store_asset(a2).unwrap();

        assert_eq!(d.asset_count(), 1); // should not duplicate
    }

    #[test]
    fn test_store_asset_exceeds_max() {
        let mut cfg = DeclarativeConfig::default();
        cfg.max_assets = 2;
        let mut d = Declarative::with_config(cfg).unwrap();
        d.store_asset(sample_asset("A")).unwrap();
        d.store_asset(sample_asset("B")).unwrap();
        assert!(d.store_asset(sample_asset("C")).is_err());
    }

    #[test]
    fn test_remove_asset() {
        let mut d = Declarative::new();
        d.store_asset(sample_asset("AAPL")).unwrap();
        assert!(d.remove_asset("AAPL").is_ok());
        assert_eq!(d.asset_count(), 0);
    }

    #[test]
    fn test_remove_asset_not_found() {
        let mut d = Declarative::new();
        assert!(d.remove_asset("nope").is_err());
    }

    #[test]
    fn test_query_asset() {
        let mut d = Declarative::new();
        d.store_asset(sample_asset("AAPL")).unwrap();
        let a = d.query_asset("AAPL");
        assert!(a.is_some());
        assert_eq!(a.unwrap().asset_id, "AAPL");
    }

    #[test]
    fn test_query_asset_miss() {
        let mut d = Declarative::new();
        let a = d.query_asset("NOPE");
        assert!(a.is_none());
    }

    #[test]
    fn test_query_assets_by_class() {
        let mut d = Declarative::new();
        d.store_asset(sample_asset("AAPL")).unwrap();
        d.store_asset(sample_crypto("BTC")).unwrap();

        let result = d.query_assets_by_class(AssetClass::Crypto);
        assert_eq!(result.items.len(), 1);
        assert_eq!(result.items[0].asset_id, "BTC");
    }

    #[test]
    fn test_query_assets_by_class_miss() {
        let mut d = Declarative::new();
        d.store_asset(sample_asset("AAPL")).unwrap();
        let result = d.query_assets_by_class(AssetClass::Commodity);
        assert_eq!(result.items.len(), 0);
    }

    #[test]
    fn test_query_assets_by_tag() {
        let mut d = Declarative::new();
        let mut a = sample_asset("AAPL");
        a.tags = vec!["tech".into(), "mega_cap".into()];
        d.store_asset(a).unwrap();

        let result = d.query_assets_by_tag("tech");
        assert_eq!(result.items.len(), 1);

        let result2 = d.query_assets_by_tag("small_cap");
        assert_eq!(result2.items.len(), 0);
    }

    #[test]
    fn test_query_assets_by_sector() {
        let mut d = Declarative::new();
        let mut a = sample_asset("AAPL");
        a.sector = Sector::Technology;
        d.store_asset(a).unwrap();

        let result = d.query_assets_by_sector(&Sector::Technology);
        assert_eq!(result.items.len(), 1);

        let result2 = d.query_assets_by_sector(&Sector::Healthcare);
        assert_eq!(result2.items.len(), 0);
    }

    #[test]
    fn test_query_assets_by_min_liquidity() {
        let mut d = Declarative::new();
        let mut a = sample_asset("AAPL");
        a.liquidity = LiquidityTier::High;
        d.store_asset(a).unwrap();

        let mut b = sample_asset("PENNY");
        b.liquidity = LiquidityTier::VeryLow;
        d.store_asset(b).unwrap();

        let result = d.query_assets_by_min_liquidity(LiquidityTier::Medium);
        // High <= Medium so AAPL should be included
        assert!(result.items.iter().any(|a| a.asset_id == "AAPL"));
    }

    #[test]
    fn test_asset_staleness() {
        let mut d = Declarative::new();
        d.store_asset(sample_asset("AAPL")).unwrap();

        // Advance step past staleness threshold
        for _ in 0..5001 {
            d.tick();
        }

        assert!(d.stale_count() > 0);
    }

    // -- Correlations --

    #[test]
    fn test_store_correlation() {
        let mut d = Declarative::new();
        let corr = sample_correlation("AAPL", "MSFT", 0.75);
        assert!(d.store_correlation(corr).is_ok());
        assert_eq!(d.correlation_count(), 1);
    }

    #[test]
    fn test_store_correlation_updates_existing() {
        let mut d = Declarative::new();
        let c1 = sample_correlation("AAPL", "MSFT", 0.75);
        d.store_correlation(c1).unwrap();

        let c2 = sample_correlation("MSFT", "AAPL", 0.80); // reversed order, same canonical pair
        d.store_correlation(c2).unwrap();

        assert_eq!(d.correlation_count(), 1);
    }

    #[test]
    fn test_store_correlation_invalid() {
        let mut d = Declarative::new();
        let c = sample_correlation("A", "B", 1.5); // out of range
        assert!(d.store_correlation(c).is_err());
    }

    #[test]
    fn test_store_correlation_exceeds_max() {
        let mut cfg = DeclarativeConfig::default();
        cfg.max_correlation_pairs = 1;
        let mut d = Declarative::with_config(cfg).unwrap();
        d.store_correlation(sample_correlation("A", "B", 0.5))
            .unwrap();
        assert!(
            d.store_correlation(sample_correlation("C", "D", 0.3))
                .is_err()
        );
    }

    #[test]
    fn test_query_correlation() {
        let mut d = Declarative::new();
        d.store_correlation(sample_correlation("AAPL", "MSFT", 0.75))
            .unwrap();

        let r = d.query_correlation("AAPL", "MSFT");
        assert!(r.is_some());
        assert!((r.unwrap().correlation - 0.75).abs() < 1e-10);

        // Reversed order should also work
        let r2 = d.query_correlation("MSFT", "AAPL");
        assert!(r2.is_some());
    }

    #[test]
    fn test_query_correlation_miss() {
        let mut d = Declarative::new();
        let r = d.query_correlation("X", "Y");
        assert!(r.is_none());
    }

    #[test]
    fn test_query_correlations_for_asset() {
        let mut d = Declarative::new();
        d.store_correlation(sample_correlation("AAPL", "MSFT", 0.75))
            .unwrap();
        d.store_correlation(sample_correlation("AAPL", "GOOG", 0.60))
            .unwrap();
        d.store_correlation(sample_correlation("MSFT", "GOOG", 0.50))
            .unwrap();

        let result = d.query_correlations_for_asset("AAPL");
        assert_eq!(result.items.len(), 2);
    }

    #[test]
    fn test_query_high_correlations() {
        let mut d = Declarative::new();
        d.store_correlation(sample_correlation("A", "B", 0.90))
            .unwrap();
        d.store_correlation(sample_correlation("C", "D", 0.30))
            .unwrap();
        d.store_correlation(sample_correlation("E", "F", -0.85))
            .unwrap();

        let result = d.query_high_correlations(0.80);
        assert_eq!(result.items.len(), 2); // 0.90 and |-0.85| >= 0.80
    }

    #[test]
    fn test_avg_abs_correlation() {
        let mut d = Declarative::new();
        d.store_correlation(sample_correlation("A", "B", 0.80))
            .unwrap();
        d.store_correlation(sample_correlation("C", "D", -0.60))
            .unwrap();

        let avg = d.avg_abs_correlation();
        assert!((avg - 0.70).abs() < 1e-10);
    }

    #[test]
    fn test_avg_abs_correlation_empty() {
        let d = Declarative::new();
        assert_eq!(d.avg_abs_correlation(), 0.0);
    }

    // -- Events --

    #[test]
    fn test_record_event() {
        let mut d = Declarative::new();
        let e = sample_event("crash_2020", EventType::Crash, EventSeverity::Extreme);
        assert!(d.record_event(e).is_ok());
        assert_eq!(d.event_count(), 1);
    }

    #[test]
    fn test_record_event_empty_id() {
        let mut d = Declarative::new();
        let e = sample_event("", EventType::Crash, EventSeverity::High);
        assert!(d.record_event(e).is_err());
    }

    #[test]
    fn test_record_event_duplicate_id() {
        let mut d = Declarative::new();
        let e1 = sample_event("e1", EventType::Crash, EventSeverity::High);
        let e2 = sample_event("e1", EventType::Rally, EventSeverity::Low);
        d.record_event(e1).unwrap();
        assert!(d.record_event(e2).is_err());
    }

    #[test]
    fn test_record_event_evicts_oldest() {
        let mut cfg = DeclarativeConfig::default();
        cfg.max_events = 2;
        let mut d = Declarative::with_config(cfg).unwrap();

        d.record_event(sample_event("e1", EventType::Crash, EventSeverity::High))
            .unwrap();
        d.record_event(sample_event("e2", EventType::Rally, EventSeverity::Low))
            .unwrap();
        d.record_event(sample_event("e3", EventType::Crash, EventSeverity::Medium))
            .unwrap();

        assert_eq!(d.event_count(), 2);
        // e1 should have been evicted
        let crashes = d.query_events_by_type(&EventType::Crash);
        assert!(!crashes.items.iter().any(|e| e.id == "e1"));
    }

    #[test]
    fn test_query_events_by_type() {
        let mut d = Declarative::new();
        d.record_event(sample_event("e1", EventType::Crash, EventSeverity::High))
            .unwrap();
        d.record_event(sample_event("e2", EventType::Rally, EventSeverity::Low))
            .unwrap();

        let result = d.query_events_by_type(&EventType::Crash);
        assert_eq!(result.items.len(), 1);
        assert_eq!(result.items[0].id, "e1");
    }

    #[test]
    fn test_query_events_by_min_severity() {
        let mut d = Declarative::new();
        d.record_event(sample_event("e1", EventType::Crash, EventSeverity::Extreme))
            .unwrap();
        d.record_event(sample_event("e2", EventType::Rally, EventSeverity::Low))
            .unwrap();
        d.record_event(sample_event("e3", EventType::Crash, EventSeverity::High))
            .unwrap();

        let result = d.query_events_by_min_severity(EventSeverity::High);
        assert_eq!(result.items.len(), 2); // Extreme and High
    }

    #[test]
    fn test_query_events_for_asset() {
        let mut d = Declarative::new();
        let mut e = sample_event("e1", EventType::Crash, EventSeverity::High);
        e.affected_assets = vec!["AAPL".into()];
        d.record_event(e).unwrap();

        // Market-wide event (empty affected_assets)
        d.record_event(sample_event("e2", EventType::Crash, EventSeverity::Extreme))
            .unwrap();

        let result = d.query_events_for_asset("AAPL");
        assert_eq!(result.items.len(), 2); // specific + market-wide
    }

    #[test]
    fn test_query_events_by_tag() {
        let mut d = Declarative::new();
        d.record_event(sample_event("e1", EventType::Crash, EventSeverity::High))
            .unwrap();

        let result = d.query_events_by_tag("test");
        assert_eq!(result.items.len(), 1);

        let result2 = d.query_events_by_tag("nonexistent");
        assert_eq!(result2.items.len(), 0);
    }

    #[test]
    fn test_recent_events() {
        let mut d = Declarative::new();
        for i in 0..5 {
            d.record_event(sample_event(
                &format!("e{}", i),
                EventType::Crash,
                EventSeverity::Medium,
            ))
            .unwrap();
        }

        let recent = d.recent_events(3);
        assert_eq!(recent.len(), 3);
        assert_eq!(recent[0].id, "e2");
        assert_eq!(recent[2].id, "e4");
    }

    #[test]
    fn test_event_count_by_severity() {
        let mut d = Declarative::new();
        d.record_event(sample_event("e1", EventType::Crash, EventSeverity::Extreme))
            .unwrap();
        d.record_event(sample_event("e2", EventType::Rally, EventSeverity::Extreme))
            .unwrap();
        d.record_event(sample_event("e3", EventType::Crash, EventSeverity::Low))
            .unwrap();

        assert_eq!(d.event_count_by_severity(EventSeverity::Extreme), 2);
        assert_eq!(d.event_count_by_severity(EventSeverity::Low), 1);
        assert_eq!(d.event_count_by_severity(EventSeverity::Medium), 0);
    }

    // -- Regime history --

    #[test]
    fn test_record_regime() {
        let mut d = Declarative::new();
        d.record_regime(RegimeLabel::Bull, 0.9, "strong uptrend");
        assert!(d.current_regime().is_some());
        assert_eq!(d.current_regime().unwrap().regime, RegimeLabel::Bull);
    }

    #[test]
    fn test_regime_history_stored() {
        let mut d = Declarative::new();
        d.record_regime(RegimeLabel::Bull, 0.9, "");
        d.record_regime(RegimeLabel::Bear, 0.7, "");
        d.record_regime(RegimeLabel::Crisis, 0.95, "");

        assert_eq!(d.regime_history().len(), 3);
        assert_eq!(d.current_regime().unwrap().regime, RegimeLabel::Crisis);
    }

    #[test]
    fn test_regime_history_capped() {
        let mut cfg = DeclarativeConfig::default();
        cfg.max_regime_history = 3;
        let mut d = Declarative::with_config(cfg).unwrap();

        for _ in 0..5 {
            d.record_regime(RegimeLabel::Bull, 0.8, "");
        }

        assert_eq!(d.regime_history().len(), 3);
    }

    #[test]
    fn test_regime_transition_count() {
        let mut d = Declarative::new();
        d.record_regime(RegimeLabel::Bull, 0.9, "");
        d.record_regime(RegimeLabel::Bull, 0.8, "");
        d.record_regime(RegimeLabel::Bear, 0.7, "");
        d.record_regime(RegimeLabel::Crisis, 0.9, "");
        d.record_regime(RegimeLabel::Crisis, 0.85, "");

        assert_eq!(d.regime_transition_count(), 2); // Bull→Bear, Bear→Crisis
    }

    #[test]
    fn test_regime_distribution() {
        let mut d = Declarative::new();
        d.record_regime(RegimeLabel::Bull, 0.9, "");
        d.record_regime(RegimeLabel::Bull, 0.8, "");
        d.record_regime(RegimeLabel::Bear, 0.7, "");
        d.record_regime(RegimeLabel::Bear, 0.6, "");

        let dist = d.regime_distribution();
        assert!((dist[RegimeLabel::Bull.index()] - 0.5).abs() < 1e-10);
        assert!((dist[RegimeLabel::Bear.index()] - 0.5).abs() < 1e-10);
        assert!((dist[RegimeLabel::Ranging.index()] - 0.0).abs() < 1e-10);
    }

    #[test]
    fn test_dominant_regime() {
        let mut d = Declarative::new();
        d.record_regime(RegimeLabel::Bull, 0.9, "");
        d.record_regime(RegimeLabel::Bull, 0.8, "");
        d.record_regime(RegimeLabel::Bear, 0.7, "");

        assert_eq!(d.dominant_regime(), Some(RegimeLabel::Bull));
    }

    #[test]
    fn test_dominant_regime_empty() {
        let d = Declarative::new();
        assert_eq!(d.dominant_regime(), None);
    }

    #[test]
    fn test_current_regime_empty() {
        let d = Declarative::new();
        assert!(d.current_regime().is_none());
    }

    #[test]
    fn test_regime_confidence_clamped() {
        let mut d = Declarative::new();
        d.record_regime(RegimeLabel::Bull, 2.0, "");
        assert!((d.current_regime().unwrap().confidence - 1.0).abs() < 1e-10);

        d.record_regime(RegimeLabel::Bear, -0.5, "");
        assert!((d.current_regime().unwrap().confidence - 0.0).abs() < 1e-10);
    }

    // -- Pruning --

    #[test]
    fn test_prune_stale() {
        let mut cfg = DeclarativeConfig::default();
        cfg.staleness_threshold = 10;
        let mut d = Declarative::with_config(cfg).unwrap();

        d.store_asset(sample_asset("AAPL")).unwrap();
        d.store_correlation(sample_correlation("AAPL", "MSFT", 0.5))
            .unwrap();

        // Advance past staleness
        for _ in 0..15 {
            d.tick();
        }

        let pruned = d.prune_stale();
        assert_eq!(pruned, 2);
        assert_eq!(d.asset_count(), 0);
        assert_eq!(d.correlation_count(), 0);
    }

    #[test]
    fn test_prune_stale_none_stale() {
        let mut d = Declarative::new();
        d.store_asset(sample_asset("AAPL")).unwrap();

        let pruned = d.prune_stale();
        assert_eq!(pruned, 0);
        assert_eq!(d.asset_count(), 1);
    }

    #[test]
    fn test_stale_count() {
        let mut cfg = DeclarativeConfig::default();
        cfg.staleness_threshold = 5;
        let mut d = Declarative::with_config(cfg).unwrap();

        d.store_asset(sample_asset("AAPL")).unwrap();

        assert_eq!(d.stale_count(), 0);

        for _ in 0..10 {
            d.tick();
        }

        assert!(d.stale_count() > 0);
    }

    #[test]
    fn test_auto_prune_on_tick() {
        let mut cfg = DeclarativeConfig::default();
        cfg.staleness_threshold = 3;
        cfg.auto_prune_stale = true;
        let mut d = Declarative::with_config(cfg).unwrap();

        d.store_asset(sample_asset("AAPL")).unwrap();

        for _ in 0..10 {
            d.tick();
        }

        assert_eq!(d.asset_count(), 0);
        assert!(d.stats().total_pruned > 0);
    }

    // -- Statistics --

    #[test]
    fn test_stats_initial() {
        let d = Declarative::new();
        assert_eq!(d.stats().total_updates, 0);
        assert_eq!(d.stats().total_queries, 0);
    }

    #[test]
    fn test_stats_updates_tracked() {
        let mut d = Declarative::new();
        d.store_asset(sample_asset("AAPL")).unwrap();
        d.store_correlation(sample_correlation("A", "B", 0.5))
            .unwrap();
        d.record_event(sample_event("e1", EventType::Crash, EventSeverity::High))
            .unwrap();
        d.record_regime(RegimeLabel::Bull, 0.9, "");

        assert_eq!(d.stats().total_updates, 4);
    }

    #[test]
    fn test_stats_queries_tracked() {
        let mut d = Declarative::new();
        d.store_asset(sample_asset("AAPL")).unwrap();

        d.query_asset("AAPL");
        d.query_asset("NOPE");

        assert_eq!(d.stats().total_queries, 2);
        assert_eq!(d.stats().query_hits, 1);
        assert_eq!(d.stats().query_misses, 1);
    }

    #[test]
    fn test_stats_hit_rate() {
        let stats = DeclarativeStats {
            query_hits: 7,
            query_misses: 3,
            ..Default::default()
        };
        assert!((stats.hit_rate() - 0.7).abs() < 1e-10);
    }

    #[test]
    fn test_stats_hit_rate_zero_queries() {
        let stats = DeclarativeStats::default();
        assert_eq!(stats.hit_rate(), 0.0);
    }

    #[test]
    fn test_stats_total_facts() {
        let stats = DeclarativeStats {
            asset_count: 10,
            correlation_count: 5,
            event_count: 3,
            regime_history_length: 2,
            ..Default::default()
        };
        assert_eq!(stats.total_facts(), 20);
    }

    #[test]
    fn test_stats_counts_synced() {
        let mut d = Declarative::new();
        d.store_asset(sample_asset("AAPL")).unwrap();
        d.store_correlation(sample_correlation("A", "B", 0.5))
            .unwrap();
        d.record_event(sample_event("e1", EventType::Crash, EventSeverity::High))
            .unwrap();
        d.record_regime(RegimeLabel::Bull, 0.9, "");

        d.tick();

        assert_eq!(d.stats().asset_count, 1);
        assert_eq!(d.stats().correlation_count, 1);
        assert_eq!(d.stats().event_count, 1);
        assert_eq!(d.stats().regime_history_length, 1);
    }

    // -- EMA tracking --

    #[test]
    fn test_smoothed_hit_rate() {
        let mut d = Declarative::new();
        d.store_asset(sample_asset("AAPL")).unwrap();

        d.query_asset("AAPL"); // hit
        d.query_asset("AAPL"); // hit
        d.query_asset("NOPE"); // miss

        // EMA should be between 0 and 1
        assert!(d.smoothed_hit_rate() >= 0.0);
        assert!(d.smoothed_hit_rate() <= 1.0);
    }

    #[test]
    fn test_smoothed_update_rate() {
        let mut d = Declarative::new();
        d.store_asset(sample_asset("AAPL")).unwrap();
        d.store_asset(sample_asset("MSFT")).unwrap();
        d.tick();

        assert!(d.smoothed_update_rate() > 0.0);
    }

    #[test]
    fn test_smoothed_freshness() {
        let mut d = Declarative::new();
        d.store_asset(sample_asset("AAPL")).unwrap();
        d.tick();
        d.tick();
        d.tick();

        // Facts are aging → freshness should increase
        assert!(d.smoothed_freshness() >= 0.0);
    }

    // -- Sliding window --

    #[test]
    fn test_recent_updates_stored() {
        let mut d = Declarative::new();
        d.store_asset(sample_asset("AAPL")).unwrap();

        assert_eq!(d.recent_update_records().len(), 1);
        assert_eq!(d.recent_update_records()[0].category, "asset");
    }

    #[test]
    fn test_recent_updates_windowed() {
        let mut cfg = DeclarativeConfig::default();
        cfg.window_size = 3;
        let mut d = Declarative::with_config(cfg).unwrap();

        for i in 0..10 {
            d.store_asset(sample_asset(&format!("A{}", i))).unwrap();
        }

        assert!(d.recent_update_records().len() <= 3);
    }

    #[test]
    fn test_windowed_update_rate() {
        let mut d = Declarative::new();
        d.store_asset(sample_asset("AAPL")).unwrap();

        let rate = d.windowed_update_rate();
        assert!(rate > 0.0);
    }

    #[test]
    fn test_windowed_update_rate_empty() {
        let d = Declarative::new();
        assert_eq!(d.windowed_update_rate(), 0.0);
    }

    // -- Tick --

    #[test]
    fn test_tick_increments_step() {
        let mut d = Declarative::new();
        assert_eq!(d.current_step(), 0);
        d.tick();
        assert_eq!(d.current_step(), 1);
        d.tick();
        assert_eq!(d.current_step(), 2);
    }

    // -- Reset --

    #[test]
    fn test_reset() {
        let mut d = Declarative::new();
        d.store_asset(sample_asset("AAPL")).unwrap();
        d.store_correlation(sample_correlation("A", "B", 0.5))
            .unwrap();
        d.record_event(sample_event("e1", EventType::Crash, EventSeverity::High))
            .unwrap();
        d.record_regime(RegimeLabel::Bull, 0.9, "");
        d.tick();

        assert!(d.asset_count() > 0);
        assert!(d.current_step() > 0);

        d.reset();

        assert_eq!(d.asset_count(), 0);
        assert_eq!(d.correlation_count(), 0);
        assert_eq!(d.event_count(), 0);
        assert_eq!(d.regime_history().len(), 0);
        assert_eq!(d.current_step(), 0);
        assert_eq!(d.stats().total_updates, 0);
    }

    // -- Presets --

    #[test]
    fn test_preset_equity_fact() {
        let f = preset_equity_fact("AAPL", "Apple Inc", Sector::Technology, 0.25, 1.1);
        assert_eq!(f.asset_id, "AAPL");
        assert_eq!(f.asset_class, AssetClass::Equity);
        assert_eq!(f.sector, Sector::Technology);
        assert!((f.typical_volatility - 0.25).abs() < 1e-10);
    }

    #[test]
    fn test_preset_crypto_fact() {
        let f = preset_crypto_fact("BTC", "Bitcoin", 0.80);
        assert_eq!(f.asset_class, AssetClass::Crypto);
        assert!((f.typical_volatility - 0.80).abs() < 1e-10);
    }

    #[test]
    fn test_preset_crash_event() {
        let e = preset_crash_event("covid_crash", "COVID-19 market crash", -0.35);
        assert_eq!(e.severity, EventSeverity::Extreme);
        assert!(e.tags.contains(&"crash".to_string()));
    }

    #[test]
    fn test_preset_crash_event_severity_levels() {
        let extreme = preset_crash_event("e1", "extreme", -0.25);
        assert_eq!(extreme.severity, EventSeverity::Extreme);

        let high = preset_crash_event("e2", "high", -0.15);
        assert_eq!(high.severity, EventSeverity::High);

        let medium = preset_crash_event("e3", "medium", -0.05);
        assert_eq!(medium.severity, EventSeverity::Medium);
    }

    #[test]
    fn test_presets_can_be_stored() {
        let mut d = Declarative::new();
        d.store_asset(preset_equity_fact(
            "AAPL",
            "Apple",
            Sector::Technology,
            0.25,
            1.1,
        ))
        .unwrap();
        d.store_asset(preset_crypto_fact("BTC", "Bitcoin", 0.80))
            .unwrap();
        d.record_event(preset_crash_event("crash", "test crash", -0.20))
            .unwrap();

        assert_eq!(d.asset_count(), 2);
        assert_eq!(d.event_count(), 1);
    }

    // -- Correlation record key --

    #[test]
    fn test_correlation_key_canonical() {
        let c1 = sample_correlation("B", "A", 0.5);
        let c2 = sample_correlation("A", "B", 0.5);
        assert_eq!(c1.key(), c2.key());
        assert_eq!(c1.key(), ("A".to_string(), "B".to_string()));
    }

    // -- AssetFact --

    #[test]
    fn test_asset_fact_new() {
        let f = AssetFact::new("TEST", AssetClass::Commodity);
        assert_eq!(f.asset_id, "TEST");
        assert_eq!(f.asset_class, AssetClass::Commodity);
        assert!(f.tradable);
    }

    #[test]
    fn test_asset_fact_staleness() {
        let mut f = AssetFact::new("TEST", AssetClass::Equity);
        f.last_updated = 100;
        assert!(!f.is_stale(105, 10));
        assert!(f.is_stale(115, 10));
    }

    // -- LiquidityTier --

    #[test]
    fn test_liquidity_tier_ordering() {
        assert!(LiquidityTier::High < LiquidityTier::Medium);
        assert!(LiquidityTier::Medium < LiquidityTier::Low);
        assert!(LiquidityTier::Low < LiquidityTier::VeryLow);
    }

    #[test]
    fn test_liquidity_tier_weight() {
        assert!(LiquidityTier::High.weight() > LiquidityTier::VeryLow.weight());
    }

    // -- EventSeverity --

    #[test]
    fn test_event_severity_ordering() {
        assert!(EventSeverity::Low < EventSeverity::Medium);
        assert!(EventSeverity::Medium < EventSeverity::High);
        assert!(EventSeverity::High < EventSeverity::Extreme);
    }

    #[test]
    fn test_event_severity_weight() {
        assert!(EventSeverity::Extreme.weight() > EventSeverity::Low.weight());
    }

    // -- RegimeLabel --

    #[test]
    fn test_regime_label_index_roundtrip() {
        for i in 0..RegimeLabel::COUNT {
            let r = RegimeLabel::from_index(i);
            assert_eq!(r.index(), i);
        }
    }

    // -- Integration test --

    #[test]
    fn test_full_lifecycle() {
        let mut d = Declarative::new();

        // Populate with assets
        d.store_asset(preset_equity_fact(
            "AAPL",
            "Apple",
            Sector::Technology,
            0.25,
            1.1,
        ))
        .unwrap();
        d.store_asset(preset_equity_fact(
            "MSFT",
            "Microsoft",
            Sector::Technology,
            0.22,
            1.0,
        ))
        .unwrap();
        d.store_asset(preset_crypto_fact("BTC", "Bitcoin", 0.80))
            .unwrap();

        // Store correlations
        d.store_correlation(sample_correlation("AAPL", "MSFT", 0.85))
            .unwrap();
        d.store_correlation(sample_correlation("AAPL", "BTC", 0.20))
            .unwrap();

        // Record events
        d.record_event(preset_crash_event("covid", "COVID crash", -0.35))
            .unwrap();

        // Record regimes
        d.record_regime(RegimeLabel::Bull, 0.9, "pre-covid");
        d.record_regime(RegimeLabel::Crisis, 0.95, "covid crash");
        d.record_regime(RegimeLabel::Recovery, 0.7, "post-covid");

        // Tick forward
        d.tick();
        d.tick();

        // Query
        let aapl = d.query_asset("AAPL");
        assert!(aapl.is_some());

        let tech = d.query_assets_by_class(AssetClass::Equity);
        assert_eq!(tech.items.len(), 2);

        let high_corr = d.query_high_correlations(0.80);
        assert_eq!(high_corr.items.len(), 1);

        let crashes = d.query_events_by_type(&EventType::Crash);
        assert_eq!(crashes.items.len(), 1);

        assert_eq!(d.regime_transition_count(), 2);
        assert_eq!(d.current_regime().unwrap().regime, RegimeLabel::Recovery);

        // Verify stats
        assert!(d.stats().total_updates > 0);
        assert!(d.stats().total_queries > 0);
        assert!(d.stats().query_hits > 0);
        assert_eq!(d.stats().asset_count, 3);
        assert_eq!(d.stats().correlation_count, 2);
        assert_eq!(d.stats().event_count, 1);
        assert_eq!(d.stats().regime_history_length, 3);
    }
}
