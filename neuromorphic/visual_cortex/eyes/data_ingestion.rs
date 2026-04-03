//! Raw market data ingestion
//!
//! Part of the Visual Cortex region
//! Component: eyes
//!
//! Provides a raw market data ingestion pipeline that accepts heterogeneous
//! data points from multiple sources, validates them against a declared
//! schema, deduplicates by (source, symbol, timestamp) key, normalises
//! numeric fields, and forwards clean records to downstream consumers.
//!
//! ## Features
//!
//! - **Multi-source registration**: Register named data sources (exchanges,
//!   feeds) with independent schemas and quality tracking
//! - **Schema validation**: Declare required and optional fields per source;
//!   incoming records are validated before acceptance
//! - **Deduplication**: Rolling dedup window catches duplicate records
//!   within a configurable time horizon
//! - **Field normalisation**: Min-max or z-score normalisation per field
//!   using running statistics
//! - **EMA throughput tracking**: Exponentially weighted moving average
//!   of ingestion rate per tick
//! - **Quality scoring**: Per-source completeness and validity metrics
//! - **Running statistics**: Per-field online mean/variance for normalisation
//!   and anomaly detection
//! - **Batch ingestion**: Accept multiple records atomically

use crate::common::{Error, Result};
use std::collections::{HashMap, HashSet, VecDeque};

// ---------------------------------------------------------------------------
// Configuration
// ---------------------------------------------------------------------------

/// Configuration for the data ingestion engine.
#[derive(Debug, Clone)]
pub struct DataIngestionConfig {
    /// Maximum number of data sources that can be registered.
    pub max_sources: usize,
    /// Maximum number of fields per schema.
    pub max_fields: usize,
    /// Deduplication window size (number of recent keys to remember).
    pub dedup_window: usize,
    /// EMA decay factor for throughput tracking (0 < decay < 1).
    pub ema_decay: f64,
    /// Minimum ticks before EMA is considered initialised.
    pub min_samples: usize,
    /// Maximum records held in the staging buffer before drain.
    pub max_staging: usize,
    /// Whether to enable deduplication.
    pub dedup_enabled: bool,
}

impl Default for DataIngestionConfig {
    fn default() -> Self {
        Self {
            max_sources: 64,
            max_fields: 128,
            dedup_window: 10_000,
            ema_decay: 0.05,
            min_samples: 10,
            max_staging: 100_000,
            dedup_enabled: true,
        }
    }
}

// ---------------------------------------------------------------------------
// Normalisation
// ---------------------------------------------------------------------------

/// Normalisation strategy for numeric fields.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum NormStrategy {
    /// No normalisation — pass through raw values.
    None,
    /// Min-max scaling to [0, 1].
    MinMax,
    /// Z-score standardisation (subtract mean, divide by std dev).
    ZScore,
}

impl Default for NormStrategy {
    fn default() -> Self {
        Self::None
    }
}

// ---------------------------------------------------------------------------
// Field Schema
// ---------------------------------------------------------------------------

/// Specification for a single field in a data source schema.
#[derive(Debug, Clone)]
pub struct FieldSpec {
    /// Field name.
    pub name: String,
    /// Whether this field is required (vs optional).
    pub required: bool,
    /// Normalisation strategy.
    pub norm: NormStrategy,
}

impl FieldSpec {
    /// Create a required field with no normalisation.
    pub fn required(name: impl Into<String>) -> Self {
        Self {
            name: name.into(),
            required: true,
            norm: NormStrategy::None,
        }
    }

    /// Create an optional field with no normalisation.
    pub fn optional(name: impl Into<String>) -> Self {
        Self {
            name: name.into(),
            required: false,
            norm: NormStrategy::None,
        }
    }

    /// Set the normalisation strategy (builder pattern).
    pub fn with_norm(mut self, norm: NormStrategy) -> Self {
        self.norm = norm;
        self
    }
}

// ---------------------------------------------------------------------------
// Data Record
// ---------------------------------------------------------------------------

/// A single raw data record from a source.
#[derive(Debug, Clone)]
pub struct DataRecord {
    /// Source name.
    pub source: String,
    /// Symbol / instrument identifier.
    pub symbol: String,
    /// Timestamp in milliseconds since epoch.
    pub timestamp_ms: i64,
    /// Named numeric fields.
    pub fields: HashMap<String, f64>,
}

impl DataRecord {
    /// Create a new data record.
    pub fn new(source: impl Into<String>, symbol: impl Into<String>, timestamp_ms: i64) -> Self {
        Self {
            source: source.into(),
            symbol: symbol.into(),
            timestamp_ms,
            fields: HashMap::new(),
        }
    }

    /// Add a field value (builder pattern).
    pub fn with_field(mut self, name: impl Into<String>, value: f64) -> Self {
        self.fields.insert(name.into(), value);
        self
    }

    /// Set a field value.
    pub fn set_field(&mut self, name: impl Into<String>, value: f64) {
        self.fields.insert(name.into(), value);
    }

    /// Get a field value.
    pub fn field(&self, name: &str) -> Option<f64> {
        self.fields.get(name).copied()
    }

    /// Deduplication key: (source, symbol, timestamp).
    fn dedup_key(&self) -> (String, String, i64) {
        (self.source.clone(), self.symbol.clone(), self.timestamp_ms)
    }
}

// ---------------------------------------------------------------------------
// Validated (normalised) record
// ---------------------------------------------------------------------------

/// A validated and optionally normalised data record.
#[derive(Debug, Clone)]
pub struct ValidatedRecord {
    /// Source name.
    pub source: String,
    /// Symbol / instrument identifier.
    pub symbol: String,
    /// Timestamp in milliseconds since epoch.
    pub timestamp_ms: i64,
    /// Normalised numeric fields.
    pub fields: HashMap<String, f64>,
    /// Raw (pre-normalisation) numeric fields.
    pub raw_fields: HashMap<String, f64>,
    /// Tick at which the record was ingested.
    pub ingested_tick: u64,
}

// ---------------------------------------------------------------------------
// Running field statistics (for normalisation)
// ---------------------------------------------------------------------------

/// Online running statistics for a single field (Welford's algorithm).
#[derive(Debug, Clone)]
struct FieldStats {
    count: u64,
    mean: f64,
    m2: f64,
    min: f64,
    max: f64,
}

impl FieldStats {
    fn new() -> Self {
        Self {
            count: 0,
            mean: 0.0,
            m2: 0.0,
            min: f64::INFINITY,
            max: f64::NEG_INFINITY,
        }
    }

    fn update(&mut self, value: f64) {
        self.count += 1;
        let delta = value - self.mean;
        self.mean += delta / self.count as f64;
        let delta2 = value - self.mean;
        self.m2 += delta * delta2;
        if value < self.min {
            self.min = value;
        }
        if value > self.max {
            self.max = value;
        }
    }

    fn variance(&self) -> f64 {
        if self.count < 2 {
            0.0
        } else {
            self.m2 / (self.count - 1) as f64
        }
    }

    fn std_dev(&self) -> f64 {
        self.variance().sqrt()
    }

    fn normalise(&self, value: f64, strategy: NormStrategy) -> f64 {
        match strategy {
            NormStrategy::None => value,
            NormStrategy::MinMax => {
                let range = self.max - self.min;
                if range.abs() < 1e-15 {
                    0.5
                } else {
                    ((value - self.min) / range).clamp(0.0, 1.0)
                }
            }
            NormStrategy::ZScore => {
                let sd = self.std_dev();
                if sd.abs() < 1e-15 {
                    0.0
                } else {
                    (value - self.mean) / sd
                }
            }
        }
    }

    fn reset(&mut self) {
        self.count = 0;
        self.mean = 0.0;
        self.m2 = 0.0;
        self.min = f64::INFINITY;
        self.max = f64::NEG_INFINITY;
    }
}

// ---------------------------------------------------------------------------
// Source
// ---------------------------------------------------------------------------

/// Per-source registration: schema + quality tracking.
#[derive(Debug, Clone)]
struct Source {
    /// Field schemas in declaration order.
    fields: Vec<FieldSpec>,
    /// Quick lookup of field names.
    field_names: HashSet<String>,
    /// Required field names.
    required_fields: HashSet<String>,
    /// Running statistics per field.
    field_stats: HashMap<String, FieldStats>,
    /// Number of records ingested.
    total_ingested: u64,
    /// Number of records rejected (validation failure).
    total_rejected: u64,
    /// Number of records deduplicated.
    total_deduped: u64,
}

impl Source {
    fn new(fields: Vec<FieldSpec>) -> Self {
        let field_names: HashSet<String> = fields.iter().map(|f| f.name.clone()).collect();
        let required_fields: HashSet<String> = fields
            .iter()
            .filter(|f| f.required)
            .map(|f| f.name.clone())
            .collect();
        let field_stats: HashMap<String, FieldStats> = fields
            .iter()
            .map(|f| (f.name.clone(), FieldStats::new()))
            .collect();
        Self {
            fields,
            field_names,
            required_fields,
            field_stats,
            total_ingested: 0,
            total_rejected: 0,
            total_deduped: 0,
        }
    }

    /// Validate a record against this source's schema.
    /// Returns a list of validation error messages (empty = valid).
    fn validate(&self, record: &DataRecord) -> Vec<String> {
        let mut errors = Vec::new();
        for req in &self.required_fields {
            if !record.fields.contains_key(req) {
                errors.push(format!("missing required field '{}'", req));
            }
        }
        errors
    }

    /// Normalise a record's fields using running statistics.
    fn normalise(&self, record: &DataRecord) -> HashMap<String, f64> {
        let mut normalised = HashMap::new();
        for spec in &self.fields {
            if let Some(&value) = record.fields.get(&spec.name) {
                if let Some(stats) = self.field_stats.get(&spec.name) {
                    normalised.insert(spec.name.clone(), stats.normalise(value, spec.norm));
                } else {
                    normalised.insert(spec.name.clone(), value);
                }
            }
        }
        // Include any extra fields not in schema (pass through raw)
        for (k, &v) in &record.fields {
            if !normalised.contains_key(k) {
                normalised.insert(k.clone(), v);
            }
        }
        normalised
    }

    fn quality_score(&self) -> f64 {
        let total = self.total_ingested + self.total_rejected;
        if total == 0 {
            1.0
        } else {
            self.total_ingested as f64 / total as f64
        }
    }

    fn reset_stats(&mut self) {
        self.total_ingested = 0;
        self.total_rejected = 0;
        self.total_deduped = 0;
        for stats in self.field_stats.values_mut() {
            stats.reset();
        }
    }
}

// ---------------------------------------------------------------------------
// Ingestion result
// ---------------------------------------------------------------------------

/// Result of ingesting a single record.
#[derive(Debug, Clone)]
pub struct IngestionResult {
    /// Whether the record was accepted.
    pub accepted: bool,
    /// Whether the record was a duplicate (and thus skipped).
    pub duplicate: bool,
    /// Validation errors (if rejected).
    pub errors: Vec<String>,
}

/// Result of a batch ingestion.
#[derive(Debug, Clone)]
pub struct BatchIngestionResult {
    /// Number of records accepted.
    pub accepted: usize,
    /// Number of records rejected.
    pub rejected: usize,
    /// Number of records deduplicated.
    pub deduplicated: usize,
    /// Per-record results.
    pub results: Vec<IngestionResult>,
}

// ---------------------------------------------------------------------------
// Source stats (public query)
// ---------------------------------------------------------------------------

/// Statistics for a single data source.
#[derive(Debug, Clone)]
pub struct SourceStats {
    /// Source name.
    pub name: String,
    /// Number of fields in schema.
    pub field_count: usize,
    /// Number of required fields.
    pub required_count: usize,
    /// Total records ingested.
    pub total_ingested: u64,
    /// Total records rejected.
    pub total_rejected: u64,
    /// Total records deduplicated.
    pub total_deduped: u64,
    /// Quality score (acceptance rate).
    pub quality_score: f64,
}

/// Per-field public statistics.
#[derive(Debug, Clone)]
pub struct FieldStatsSummary {
    /// Field name.
    pub name: String,
    /// Number of observations.
    pub count: u64,
    /// Running mean.
    pub mean: f64,
    /// Running standard deviation.
    pub std_dev: f64,
    /// All-time minimum.
    pub min: f64,
    /// All-time maximum.
    pub max: f64,
}

// ---------------------------------------------------------------------------
// Global stats
// ---------------------------------------------------------------------------

/// Global ingestion statistics.
#[derive(Debug, Clone)]
pub struct IngestionStats {
    /// Total records ingested across all sources.
    pub total_ingested: u64,
    /// Total records rejected across all sources.
    pub total_rejected: u64,
    /// Total records deduplicated.
    pub total_deduped: u64,
    /// Number of registered sources.
    pub source_count: usize,
    /// Current staging buffer size.
    pub staging_size: usize,
    /// EMA of ingestion rate (records per tick).
    pub ema_throughput: f64,
    /// Number of ticks processed.
    pub ticks: u64,
    /// Total records drained from staging.
    pub total_drained: u64,
}

impl Default for IngestionStats {
    fn default() -> Self {
        Self {
            total_ingested: 0,
            total_rejected: 0,
            total_deduped: 0,
            source_count: 0,
            staging_size: 0,
            ema_throughput: 0.0,
            ticks: 0,
            total_drained: 0,
        }
    }
}

// ---------------------------------------------------------------------------
// DataIngestion
// ---------------------------------------------------------------------------

/// Raw market data ingestion pipeline.
///
/// Multi-source ingestion with schema validation, deduplication,
/// normalisation, and throughput tracking. Validated records are placed
/// into a staging buffer that can be drained by downstream consumers.
pub struct DataIngestion {
    config: DataIngestionConfig,
    sources: HashMap<String, Source>,
    source_order: Vec<String>,

    // Deduplication
    dedup_set: HashSet<(String, String, i64)>,
    dedup_queue: VecDeque<(String, String, i64)>,

    // Staging buffer for validated records
    staging: VecDeque<ValidatedRecord>,

    // EMA throughput state
    ema_throughput: f64,
    ema_initialized: bool,
    ingestions_this_tick: u64,

    current_tick: u64,
    total_drained: u64,
}

impl Default for DataIngestion {
    fn default() -> Self {
        Self::new()
    }
}

impl DataIngestion {
    /// Create a new data ingestion engine with default configuration.
    pub fn new() -> Self {
        Self::with_config(DataIngestionConfig::default()).unwrap()
    }

    /// Create with explicit configuration.
    pub fn with_config(config: DataIngestionConfig) -> Result<Self> {
        if config.max_sources == 0 {
            return Err(Error::InvalidInput("max_sources must be > 0".into()));
        }
        if config.max_fields == 0 {
            return Err(Error::InvalidInput("max_fields must be > 0".into()));
        }
        if config.ema_decay <= 0.0 || config.ema_decay >= 1.0 {
            return Err(Error::InvalidInput("ema_decay must be in (0, 1)".into()));
        }
        if config.max_staging == 0 {
            return Err(Error::InvalidInput("max_staging must be > 0".into()));
        }
        Ok(Self {
            config,
            sources: HashMap::new(),
            source_order: Vec::new(),
            dedup_set: HashSet::new(),
            dedup_queue: VecDeque::new(),
            staging: VecDeque::new(),
            ema_throughput: 0.0,
            ema_initialized: false,
            ingestions_this_tick: 0,
            current_tick: 0,
            total_drained: 0,
        })
    }

    // -----------------------------------------------------------------------
    // Source management
    // -----------------------------------------------------------------------

    /// Register a data source with a field schema.
    pub fn register_source(
        &mut self,
        name: impl Into<String>,
        fields: Vec<FieldSpec>,
    ) -> Result<()> {
        let name = name.into();
        if self.sources.contains_key(&name) {
            return Err(Error::InvalidInput(format!(
                "source '{}' already registered",
                name
            )));
        }
        if self.sources.len() >= self.config.max_sources {
            return Err(Error::ResourceExhausted(format!(
                "maximum sources ({}) reached",
                self.config.max_sources
            )));
        }
        if fields.len() > self.config.max_fields {
            return Err(Error::InvalidInput(format!(
                "schema has {} fields, maximum is {}",
                fields.len(),
                self.config.max_fields
            )));
        }
        self.sources.insert(name.clone(), Source::new(fields));
        self.source_order.push(name);
        Ok(())
    }

    /// Deregister a data source.
    pub fn deregister_source(&mut self, name: &str) -> Result<()> {
        if self.sources.remove(name).is_none() {
            return Err(Error::NotFound(format!("source '{}' not found", name)));
        }
        self.source_order.retain(|n| n != name);
        Ok(())
    }

    /// Get the list of registered source names (in registration order).
    pub fn source_names(&self) -> &[String] {
        &self.source_order
    }

    /// Number of registered sources.
    pub fn source_count(&self) -> usize {
        self.sources.len()
    }

    // -----------------------------------------------------------------------
    // Ingest
    // -----------------------------------------------------------------------

    /// Ingest a single data record.
    pub fn ingest(&mut self, record: DataRecord) -> Result<IngestionResult> {
        let source = self
            .sources
            .get_mut(&record.source)
            .ok_or_else(|| Error::NotFound(format!("source '{}' not registered", record.source)))?;

        // Validate
        let errors = source.validate(&record);
        if !errors.is_empty() {
            source.total_rejected += 1;
            return Ok(IngestionResult {
                accepted: false,
                duplicate: false,
                errors,
            });
        }

        // Deduplication
        if self.config.dedup_enabled {
            let key = record.dedup_key();
            if self.dedup_set.contains(&key) {
                // Re-fetch source as mutable (we need to work around borrow)
                self.sources.get_mut(&record.source).unwrap().total_deduped += 1;
                return Ok(IngestionResult {
                    accepted: false,
                    duplicate: true,
                    errors: Vec::new(),
                });
            }
            // Add to dedup window
            self.dedup_set.insert(key.clone());
            self.dedup_queue.push_back(key);
            // Evict oldest if window exceeded
            while self.dedup_queue.len() > self.config.dedup_window {
                if let Some(old) = self.dedup_queue.pop_front() {
                    self.dedup_set.remove(&old);
                }
            }
        }

        // Update running statistics before normalisation
        // (re-borrow source)
        let source = self.sources.get_mut(&record.source).unwrap();
        for (field_name, &value) in &record.fields {
            if let Some(stats) = source.field_stats.get_mut(field_name) {
                stats.update(value);
            }
        }

        // Normalise
        let normalised_fields = source.normalise(&record);
        source.total_ingested += 1;

        let validated = ValidatedRecord {
            source: record.source.clone(),
            symbol: record.symbol.clone(),
            timestamp_ms: record.timestamp_ms,
            fields: normalised_fields,
            raw_fields: record.fields,
            ingested_tick: self.current_tick,
        };

        // Add to staging (evict oldest if at capacity)
        if self.staging.len() >= self.config.max_staging {
            self.staging.pop_front();
        }
        self.staging.push_back(validated);
        self.ingestions_this_tick += 1;

        Ok(IngestionResult {
            accepted: true,
            duplicate: false,
            errors: Vec::new(),
        })
    }

    /// Ingest a batch of records.
    pub fn ingest_batch(&mut self, records: Vec<DataRecord>) -> Result<BatchIngestionResult> {
        let mut accepted = 0;
        let mut rejected = 0;
        let mut deduplicated = 0;
        let mut results = Vec::with_capacity(records.len());

        for record in records {
            let result = self.ingest(record)?;
            if result.accepted {
                accepted += 1;
            } else if result.duplicate {
                deduplicated += 1;
            } else {
                rejected += 1;
            }
            results.push(result);
        }

        Ok(BatchIngestionResult {
            accepted,
            rejected,
            deduplicated,
            results,
        })
    }

    // -----------------------------------------------------------------------
    // Staging buffer
    // -----------------------------------------------------------------------

    /// Number of validated records in the staging buffer.
    pub fn staging_len(&self) -> usize {
        self.staging.len()
    }

    /// Whether the staging buffer is empty.
    pub fn staging_is_empty(&self) -> bool {
        self.staging.is_empty()
    }

    /// Drain up to `max` validated records from the staging buffer.
    pub fn drain(&mut self, max: usize) -> Vec<ValidatedRecord> {
        let n = max.min(self.staging.len());
        let drained: Vec<ValidatedRecord> = self.staging.drain(..n).collect();
        self.total_drained += drained.len() as u64;
        drained
    }

    /// Drain all validated records from the staging buffer.
    pub fn drain_all(&mut self) -> Vec<ValidatedRecord> {
        let drained: Vec<ValidatedRecord> = self.staging.drain(..).collect();
        self.total_drained += drained.len() as u64;
        drained
    }

    /// Peek at the next record in the staging buffer without removing it.
    pub fn peek(&self) -> Option<&ValidatedRecord> {
        self.staging.front()
    }

    // -----------------------------------------------------------------------
    // Queries
    // -----------------------------------------------------------------------

    /// Get statistics for a specific source.
    pub fn source_stats(&self, name: &str) -> Result<SourceStats> {
        let source = self
            .sources
            .get(name)
            .ok_or_else(|| Error::NotFound(format!("source '{}' not found", name)))?;
        Ok(SourceStats {
            name: name.to_string(),
            field_count: source.fields.len(),
            required_count: source.required_fields.len(),
            total_ingested: source.total_ingested,
            total_rejected: source.total_rejected,
            total_deduped: source.total_deduped,
            quality_score: source.quality_score(),
        })
    }

    /// Get running statistics for a specific field on a source.
    pub fn field_stats(&self, source_name: &str, field_name: &str) -> Result<FieldStatsSummary> {
        let source = self
            .sources
            .get(source_name)
            .ok_or_else(|| Error::NotFound(format!("source '{}' not found", source_name)))?;
        let stats = source.field_stats.get(field_name).ok_or_else(|| {
            Error::NotFound(format!(
                "field '{}' not found in source '{}'",
                field_name, source_name
            ))
        })?;
        Ok(FieldStatsSummary {
            name: field_name.to_string(),
            count: stats.count,
            mean: stats.mean,
            std_dev: stats.std_dev(),
            min: stats.min,
            max: stats.max,
        })
    }

    /// Whether the EMA throughput tracker has been initialised.
    pub fn is_warmed_up(&self) -> bool {
        self.current_tick >= self.config.min_samples as u64
    }

    /// Current EMA throughput (records ingested per tick, smoothed).
    pub fn ema_throughput(&self) -> f64 {
        self.ema_throughput
    }

    /// Current tick.
    pub fn current_tick(&self) -> u64 {
        self.current_tick
    }

    // -----------------------------------------------------------------------
    // Tick / Process
    // -----------------------------------------------------------------------

    /// Advance one tick. Updates the EMA throughput.
    pub fn tick(&mut self) {
        let rate = self.ingestions_this_tick as f64;
        if !self.ema_initialized {
            self.ema_throughput = rate;
            self.ema_initialized = true;
        } else {
            self.ema_throughput =
                self.config.ema_decay * rate + (1.0 - self.config.ema_decay) * self.ema_throughput;
        }
        self.ingestions_this_tick = 0;
        self.current_tick += 1;
    }

    /// Aggregate global statistics.
    pub fn stats(&self) -> IngestionStats {
        let mut total_ingested = 0u64;
        let mut total_rejected = 0u64;
        let mut total_deduped = 0u64;

        for source in self.sources.values() {
            total_ingested += source.total_ingested;
            total_rejected += source.total_rejected;
            total_deduped += source.total_deduped;
        }

        IngestionStats {
            total_ingested,
            total_rejected,
            total_deduped,
            source_count: self.sources.len(),
            staging_size: self.staging.len(),
            ema_throughput: self.ema_throughput,
            ticks: self.current_tick,
            total_drained: self.total_drained,
        }
    }

    // -----------------------------------------------------------------------
    // Clear / Reset
    // -----------------------------------------------------------------------

    /// Clear the staging buffer.
    pub fn clear_staging(&mut self) {
        self.staging.clear();
    }

    /// Clear the deduplication window.
    pub fn clear_dedup(&mut self) {
        self.dedup_set.clear();
        self.dedup_queue.clear();
    }

    /// Reset all statistics but keep source registrations.
    pub fn reset_stats(&mut self) {
        for source in self.sources.values_mut() {
            source.reset_stats();
        }
        self.ema_throughput = 0.0;
        self.ema_initialized = false;
        self.total_drained = 0;
    }

    /// Full reset — clears everything including source registrations.
    pub fn reset(&mut self) {
        self.sources.clear();
        self.source_order.clear();
        self.dedup_set.clear();
        self.dedup_queue.clear();
        self.staging.clear();
        self.ema_throughput = 0.0;
        self.ema_initialized = false;
        self.ingestions_this_tick = 0;
        self.current_tick = 0;
        self.total_drained = 0;
    }

    /// Main processing function (tick alias).
    pub fn process(&mut self) -> Result<()> {
        self.tick();
        Ok(())
    }
}

// ===========================================================================
// Tests
// ===========================================================================

#[cfg(test)]
mod tests {
    use super::*;

    fn trade_schema() -> Vec<FieldSpec> {
        vec![
            FieldSpec::required("price"),
            FieldSpec::required("volume"),
            FieldSpec::optional("side"),
        ]
    }

    fn make_record(source: &str, symbol: &str, ts: i64, price: f64, volume: f64) -> DataRecord {
        DataRecord::new(source, symbol, ts)
            .with_field("price", price)
            .with_field("volume", volume)
    }

    // -----------------------------------------------------------------------
    // Construction
    // -----------------------------------------------------------------------

    #[test]
    fn test_basic() {
        let di = DataIngestion::new();
        assert_eq!(di.source_count(), 0);
        assert_eq!(di.current_tick(), 0);
    }

    #[test]
    fn test_default() {
        let di = DataIngestion::default();
        assert_eq!(di.source_count(), 0);
    }

    #[test]
    fn test_with_config() {
        let cfg = DataIngestionConfig {
            max_sources: 4,
            max_fields: 16,
            dedup_window: 100,
            ema_decay: 0.1,
            min_samples: 5,
            max_staging: 500,
            dedup_enabled: true,
        };
        let di = DataIngestion::with_config(cfg).unwrap();
        assert_eq!(di.source_count(), 0);
    }

    #[test]
    fn test_invalid_config_max_sources() {
        let mut cfg = DataIngestionConfig::default();
        cfg.max_sources = 0;
        assert!(DataIngestion::with_config(cfg).is_err());
    }

    #[test]
    fn test_invalid_config_max_fields() {
        let mut cfg = DataIngestionConfig::default();
        cfg.max_fields = 0;
        assert!(DataIngestion::with_config(cfg).is_err());
    }

    #[test]
    fn test_invalid_config_ema_decay_zero() {
        let mut cfg = DataIngestionConfig::default();
        cfg.ema_decay = 0.0;
        assert!(DataIngestion::with_config(cfg).is_err());
    }

    #[test]
    fn test_invalid_config_ema_decay_one() {
        let mut cfg = DataIngestionConfig::default();
        cfg.ema_decay = 1.0;
        assert!(DataIngestion::with_config(cfg).is_err());
    }

    #[test]
    fn test_invalid_config_max_staging() {
        let mut cfg = DataIngestionConfig::default();
        cfg.max_staging = 0;
        assert!(DataIngestion::with_config(cfg).is_err());
    }

    // -----------------------------------------------------------------------
    // Source management
    // -----------------------------------------------------------------------

    #[test]
    fn test_register_source() {
        let mut di = DataIngestion::new();
        di.register_source("binance", trade_schema()).unwrap();
        assert_eq!(di.source_count(), 1);
        assert_eq!(di.source_names(), &["binance"]);
    }

    #[test]
    fn test_register_duplicate() {
        let mut di = DataIngestion::new();
        di.register_source("binance", trade_schema()).unwrap();
        assert!(di.register_source("binance", trade_schema()).is_err());
    }

    #[test]
    fn test_register_max_capacity() {
        let cfg = DataIngestionConfig {
            max_sources: 2,
            ..Default::default()
        };
        let mut di = DataIngestion::with_config(cfg).unwrap();
        di.register_source("a", trade_schema()).unwrap();
        di.register_source("b", trade_schema()).unwrap();
        assert!(di.register_source("c", trade_schema()).is_err());
    }

    #[test]
    fn test_register_too_many_fields() {
        let cfg = DataIngestionConfig {
            max_fields: 2,
            ..Default::default()
        };
        let mut di = DataIngestion::with_config(cfg).unwrap();
        // trade_schema() has 3 fields
        assert!(di.register_source("x", trade_schema()).is_err());
    }

    #[test]
    fn test_deregister_source() {
        let mut di = DataIngestion::new();
        di.register_source("binance", trade_schema()).unwrap();
        di.deregister_source("binance").unwrap();
        assert_eq!(di.source_count(), 0);
    }

    #[test]
    fn test_deregister_nonexistent() {
        let mut di = DataIngestion::new();
        assert!(di.deregister_source("nope").is_err());
    }

    // -----------------------------------------------------------------------
    // Ingest
    // -----------------------------------------------------------------------

    #[test]
    fn test_ingest_valid() {
        let mut di = DataIngestion::new();
        di.register_source("binance", trade_schema()).unwrap();
        let record = make_record("binance", "BTCUSDT", 1000, 50000.0, 1.5);
        let result = di.ingest(record).unwrap();
        assert!(result.accepted);
        assert!(!result.duplicate);
        assert!(result.errors.is_empty());
        assert_eq!(di.staging_len(), 1);
    }

    #[test]
    fn test_ingest_missing_required_field() {
        let mut di = DataIngestion::new();
        di.register_source("binance", trade_schema()).unwrap();
        // Missing "volume"
        let record = DataRecord::new("binance", "BTCUSDT", 1000).with_field("price", 50000.0);
        let result = di.ingest(record).unwrap();
        assert!(!result.accepted);
        assert!(!result.duplicate);
        assert!(!result.errors.is_empty());
        assert_eq!(di.staging_len(), 0);
    }

    #[test]
    fn test_ingest_unregistered_source() {
        let mut di = DataIngestion::new();
        let record = make_record("unknown", "BTCUSDT", 1000, 50000.0, 1.5);
        assert!(di.ingest(record).is_err());
    }

    #[test]
    fn test_ingest_with_optional_field() {
        let mut di = DataIngestion::new();
        di.register_source("binance", trade_schema()).unwrap();
        let record = make_record("binance", "BTCUSDT", 1000, 50000.0, 1.5).with_field("side", 1.0);
        let result = di.ingest(record).unwrap();
        assert!(result.accepted);
    }

    // -----------------------------------------------------------------------
    // Deduplication
    // -----------------------------------------------------------------------

    #[test]
    fn test_dedup_catches_duplicate() {
        let mut di = DataIngestion::new();
        di.register_source("binance", trade_schema()).unwrap();
        let r1 = make_record("binance", "BTCUSDT", 1000, 50000.0, 1.5);
        let r2 = make_record("binance", "BTCUSDT", 1000, 50000.0, 1.5); // same key
        di.ingest(r1).unwrap();
        let result = di.ingest(r2).unwrap();
        assert!(!result.accepted);
        assert!(result.duplicate);
        assert_eq!(di.staging_len(), 1);
    }

    #[test]
    fn test_dedup_different_timestamps() {
        let mut di = DataIngestion::new();
        di.register_source("binance", trade_schema()).unwrap();
        let r1 = make_record("binance", "BTCUSDT", 1000, 50000.0, 1.5);
        let r2 = make_record("binance", "BTCUSDT", 1001, 50000.0, 1.5); // different ts
        di.ingest(r1).unwrap();
        let result = di.ingest(r2).unwrap();
        assert!(result.accepted);
        assert_eq!(di.staging_len(), 2);
    }

    #[test]
    fn test_dedup_different_symbols() {
        let mut di = DataIngestion::new();
        di.register_source("binance", trade_schema()).unwrap();
        let r1 = make_record("binance", "BTCUSDT", 1000, 50000.0, 1.5);
        let r2 = make_record("binance", "ETHUSDT", 1000, 3000.0, 10.0); // different symbol
        di.ingest(r1).unwrap();
        let result = di.ingest(r2).unwrap();
        assert!(result.accepted);
    }

    #[test]
    fn test_dedup_window_eviction() {
        let cfg = DataIngestionConfig {
            dedup_window: 3,
            ..Default::default()
        };
        let mut di = DataIngestion::with_config(cfg).unwrap();
        di.register_source("binance", trade_schema()).unwrap();
        // Fill dedup window
        for i in 0..4 {
            let r = make_record("binance", "BTCUSDT", i, 100.0, 1.0);
            di.ingest(r).unwrap();
        }
        // First key (ts=0) should have been evicted, so re-ingesting ts=0 should succeed
        let r = make_record("binance", "BTCUSDT", 0, 100.0, 1.0);
        let result = di.ingest(r).unwrap();
        assert!(result.accepted);
    }

    #[test]
    fn test_dedup_disabled() {
        let cfg = DataIngestionConfig {
            dedup_enabled: false,
            ..Default::default()
        };
        let mut di = DataIngestion::with_config(cfg).unwrap();
        di.register_source("binance", trade_schema()).unwrap();
        let r1 = make_record("binance", "BTCUSDT", 1000, 50000.0, 1.5);
        let r2 = make_record("binance", "BTCUSDT", 1000, 50000.0, 1.5);
        di.ingest(r1).unwrap();
        let result = di.ingest(r2).unwrap();
        assert!(result.accepted); // duplicates allowed
        assert_eq!(di.staging_len(), 2);
    }

    // -----------------------------------------------------------------------
    // Batch ingestion
    // -----------------------------------------------------------------------

    #[test]
    fn test_ingest_batch() {
        let mut di = DataIngestion::new();
        di.register_source("binance", trade_schema()).unwrap();
        let records = vec![
            make_record("binance", "BTCUSDT", 1, 50000.0, 1.0),
            make_record("binance", "BTCUSDT", 2, 50001.0, 2.0),
            make_record("binance", "BTCUSDT", 3, 50002.0, 3.0),
        ];
        let result = di.ingest_batch(records).unwrap();
        assert_eq!(result.accepted, 3);
        assert_eq!(result.rejected, 0);
        assert_eq!(result.deduplicated, 0);
        assert_eq!(result.results.len(), 3);
    }

    #[test]
    fn test_ingest_batch_mixed() {
        let mut di = DataIngestion::new();
        di.register_source("binance", trade_schema()).unwrap();
        let records = vec![
            make_record("binance", "BTCUSDT", 1, 50000.0, 1.0),
            DataRecord::new("binance", "BTCUSDT", 2).with_field("price", 50001.0), // missing volume
            make_record("binance", "BTCUSDT", 1, 50000.0, 1.0),                    // duplicate
        ];
        let result = di.ingest_batch(records).unwrap();
        assert_eq!(result.accepted, 1);
        assert_eq!(result.rejected, 1);
        assert_eq!(result.deduplicated, 1);
    }

    // -----------------------------------------------------------------------
    // Staging buffer
    // -----------------------------------------------------------------------

    #[test]
    fn test_drain() {
        let mut di = DataIngestion::new();
        di.register_source("binance", trade_schema()).unwrap();
        for i in 0..5 {
            di.ingest(make_record("binance", "BTCUSDT", i, 100.0, 1.0))
                .unwrap();
        }
        let drained = di.drain(3);
        assert_eq!(drained.len(), 3);
        assert_eq!(di.staging_len(), 2);
    }

    #[test]
    fn test_drain_all() {
        let mut di = DataIngestion::new();
        di.register_source("binance", trade_schema()).unwrap();
        for i in 0..5 {
            di.ingest(make_record("binance", "BTCUSDT", i, 100.0, 1.0))
                .unwrap();
        }
        let drained = di.drain_all();
        assert_eq!(drained.len(), 5);
        assert!(di.staging_is_empty());
    }

    #[test]
    fn test_drain_more_than_available() {
        let mut di = DataIngestion::new();
        di.register_source("binance", trade_schema()).unwrap();
        di.ingest(make_record("binance", "BTCUSDT", 0, 100.0, 1.0))
            .unwrap();
        let drained = di.drain(100);
        assert_eq!(drained.len(), 1);
    }

    #[test]
    fn test_peek() {
        let mut di = DataIngestion::new();
        di.register_source("binance", trade_schema()).unwrap();
        assert!(di.peek().is_none());
        di.ingest(make_record("binance", "BTCUSDT", 42, 100.0, 1.0))
            .unwrap();
        let peeked = di.peek().unwrap();
        assert_eq!(peeked.timestamp_ms, 42);
        // peek does not remove
        assert_eq!(di.staging_len(), 1);
    }

    #[test]
    fn test_staging_overflow() {
        let cfg = DataIngestionConfig {
            max_staging: 3,
            ..Default::default()
        };
        let mut di = DataIngestion::with_config(cfg).unwrap();
        di.register_source("binance", trade_schema()).unwrap();
        for i in 0..5 {
            di.ingest(make_record("binance", "BTCUSDT", i, 100.0, 1.0))
                .unwrap();
        }
        assert_eq!(di.staging_len(), 3);
        // Oldest should have been evicted; remaining should be ts 2,3,4
        let peeked = di.peek().unwrap();
        assert_eq!(peeked.timestamp_ms, 2);
    }

    // -----------------------------------------------------------------------
    // Normalisation
    // -----------------------------------------------------------------------

    #[test]
    fn test_normalisation_min_max() {
        let mut di = DataIngestion::new();
        let schema = vec![
            FieldSpec::required("price").with_norm(NormStrategy::MinMax),
            FieldSpec::required("volume"),
        ];
        di.register_source("binance", schema).unwrap();

        // Push several records to build up min/max stats
        for i in 0..10 {
            di.ingest(make_record(
                "binance",
                "BTCUSDT",
                i,
                (i + 1) as f64 * 10.0,
                1.0,
            ))
            .unwrap();
        }

        let records = di.drain_all();
        // After 10 records, min=10, max=100
        // The last record (price=100) should normalise to 1.0
        let last = records.last().unwrap();
        assert!((last.fields["price"] - 1.0).abs() < 1e-10);
        // raw_fields should still have the original value
        assert!((last.raw_fields["price"] - 100.0).abs() < 1e-10);
    }

    #[test]
    fn test_normalisation_zscore() {
        let mut di = DataIngestion::new();
        let schema = vec![
            FieldSpec::required("price").with_norm(NormStrategy::ZScore),
            FieldSpec::required("volume"),
        ];
        di.register_source("binance", schema).unwrap();

        for i in 0..100 {
            di.ingest(make_record("binance", "BTCUSDT", i, i as f64, 1.0))
                .unwrap();
        }

        let stats = di.field_stats("binance", "price").unwrap();
        assert!(stats.count == 100);
        // Mean should be ~49.5
        assert!((stats.mean - 49.5).abs() < 0.1);
        assert!(stats.std_dev > 0.0);
    }

    #[test]
    fn test_normalisation_none() {
        let mut di = DataIngestion::new();
        let schema = vec![
            FieldSpec::required("price").with_norm(NormStrategy::None),
            FieldSpec::required("volume"),
        ];
        di.register_source("binance", schema).unwrap();
        di.ingest(make_record("binance", "BTCUSDT", 0, 50000.0, 1.0))
            .unwrap();
        let records = di.drain_all();
        assert!((records[0].fields["price"] - 50000.0).abs() < 1e-10);
    }

    // -----------------------------------------------------------------------
    // Validated record
    // -----------------------------------------------------------------------

    #[test]
    fn test_validated_record_tick() {
        let mut di = DataIngestion::new();
        di.register_source("binance", trade_schema()).unwrap();
        di.tick();
        di.tick();
        di.ingest(make_record("binance", "BTCUSDT", 0, 100.0, 1.0))
            .unwrap();
        let records = di.drain_all();
        assert_eq!(records[0].ingested_tick, 2);
    }

    #[test]
    fn test_validated_record_has_raw_fields() {
        let mut di = DataIngestion::new();
        di.register_source("binance", trade_schema()).unwrap();
        di.ingest(make_record("binance", "BTCUSDT", 0, 42.0, 7.0))
            .unwrap();
        let records = di.drain_all();
        assert!((records[0].raw_fields["price"] - 42.0).abs() < 1e-10);
        assert!((records[0].raw_fields["volume"] - 7.0).abs() < 1e-10);
    }

    // -----------------------------------------------------------------------
    // Source stats
    // -----------------------------------------------------------------------

    #[test]
    fn test_source_stats() {
        let mut di = DataIngestion::new();
        di.register_source("binance", trade_schema()).unwrap();
        di.ingest(make_record("binance", "BTCUSDT", 0, 100.0, 1.0))
            .unwrap();
        // Inject a rejection
        let bad = DataRecord::new("binance", "BTCUSDT", 1).with_field("price", 100.0); // missing volume
        di.ingest(bad).unwrap();

        let stats = di.source_stats("binance").unwrap();
        assert_eq!(stats.total_ingested, 1);
        assert_eq!(stats.total_rejected, 1);
        assert_eq!(stats.field_count, 3);
        assert_eq!(stats.required_count, 2);
        assert!((stats.quality_score - 0.5).abs() < 1e-10);
    }

    #[test]
    fn test_source_stats_nonexistent() {
        let di = DataIngestion::new();
        assert!(di.source_stats("nope").is_err());
    }

    // -----------------------------------------------------------------------
    // Field stats
    // -----------------------------------------------------------------------

    #[test]
    fn test_field_stats() {
        let mut di = DataIngestion::new();
        di.register_source("binance", trade_schema()).unwrap();
        di.ingest(make_record("binance", "BTCUSDT", 0, 10.0, 1.0))
            .unwrap();
        di.ingest(make_record("binance", "BTCUSDT", 1, 20.0, 2.0))
            .unwrap();
        di.ingest(make_record("binance", "BTCUSDT", 2, 30.0, 3.0))
            .unwrap();

        let stats = di.field_stats("binance", "price").unwrap();
        assert_eq!(stats.count, 3);
        assert!((stats.mean - 20.0).abs() < 1e-10);
        assert!((stats.min - 10.0).abs() < 1e-10);
        assert!((stats.max - 30.0).abs() < 1e-10);
        assert!(stats.std_dev > 0.0);
    }

    #[test]
    fn test_field_stats_nonexistent_source() {
        let di = DataIngestion::new();
        assert!(di.field_stats("nope", "price").is_err());
    }

    #[test]
    fn test_field_stats_nonexistent_field() {
        let mut di = DataIngestion::new();
        di.register_source("binance", trade_schema()).unwrap();
        assert!(di.field_stats("binance", "nonexistent").is_err());
    }

    // -----------------------------------------------------------------------
    // EMA throughput
    // -----------------------------------------------------------------------

    #[test]
    fn test_ema_throughput_initial() {
        let mut di = DataIngestion::new();
        di.register_source("binance", trade_schema()).unwrap();
        for i in 0..5 {
            di.ingest(make_record("binance", "BTCUSDT", i, 100.0, 1.0))
                .unwrap();
        }
        di.tick();
        assert!((di.ema_throughput() - 5.0).abs() < 1e-10);
    }

    #[test]
    fn test_ema_throughput_smoothing() {
        let mut di = DataIngestion::new();
        di.register_source("binance", trade_schema()).unwrap();
        // Tick 1: 10 ingestions
        for i in 0..10 {
            di.ingest(make_record("binance", "BTCUSDT", i, 100.0, 1.0))
                .unwrap();
        }
        di.tick();
        assert!((di.ema_throughput() - 10.0).abs() < 1e-10);
        // Tick 2: 0 ingestions — should decay
        di.tick();
        assert!(di.ema_throughput() < 10.0);
        assert!(di.ema_throughput() > 0.0);
    }

    #[test]
    fn test_is_warmed_up() {
        let cfg = DataIngestionConfig {
            min_samples: 3,
            ..Default::default()
        };
        let mut di = DataIngestion::with_config(cfg).unwrap();
        assert!(!di.is_warmed_up());
        di.tick();
        di.tick();
        assert!(!di.is_warmed_up());
        di.tick();
        assert!(di.is_warmed_up());
    }

    // -----------------------------------------------------------------------
    // Global stats
    // -----------------------------------------------------------------------

    #[test]
    fn test_global_stats() {
        let mut di = DataIngestion::new();
        di.register_source("binance", trade_schema()).unwrap();
        di.register_source("kraken", trade_schema()).unwrap();

        di.ingest(make_record("binance", "BTCUSDT", 0, 100.0, 1.0))
            .unwrap();
        di.ingest(make_record("kraken", "BTCUSD", 0, 100.0, 1.0))
            .unwrap();
        di.tick();

        let stats = di.stats();
        assert_eq!(stats.total_ingested, 2);
        assert_eq!(stats.source_count, 2);
        assert_eq!(stats.staging_size, 2);
        assert_eq!(stats.ticks, 1);
    }

    #[test]
    fn test_stats_drained_count() {
        let mut di = DataIngestion::new();
        di.register_source("binance", trade_schema()).unwrap();
        for i in 0..5 {
            di.ingest(make_record("binance", "BTCUSDT", i, 100.0, 1.0))
                .unwrap();
        }
        di.drain(3);
        assert_eq!(di.stats().total_drained, 3);
    }

    // -----------------------------------------------------------------------
    // Tick / Process
    // -----------------------------------------------------------------------

    #[test]
    fn test_tick_increments() {
        let mut di = DataIngestion::new();
        assert_eq!(di.current_tick(), 0);
        di.tick();
        assert_eq!(di.current_tick(), 1);
    }

    #[test]
    fn test_process() {
        let mut di = DataIngestion::new();
        assert!(di.process().is_ok());
        assert_eq!(di.current_tick(), 1);
    }

    // -----------------------------------------------------------------------
    // Clear / Reset
    // -----------------------------------------------------------------------

    #[test]
    fn test_clear_staging() {
        let mut di = DataIngestion::new();
        di.register_source("binance", trade_schema()).unwrap();
        di.ingest(make_record("binance", "BTCUSDT", 0, 100.0, 1.0))
            .unwrap();
        di.clear_staging();
        assert!(di.staging_is_empty());
    }

    #[test]
    fn test_clear_dedup() {
        let mut di = DataIngestion::new();
        di.register_source("binance", trade_schema()).unwrap();
        di.ingest(make_record("binance", "BTCUSDT", 0, 100.0, 1.0))
            .unwrap();
        di.clear_dedup();
        // Now the same key should be accepted again
        let result = di
            .ingest(make_record("binance", "BTCUSDT", 0, 100.0, 1.0))
            .unwrap();
        assert!(result.accepted);
    }

    #[test]
    fn test_reset_stats() {
        let mut di = DataIngestion::new();
        di.register_source("binance", trade_schema()).unwrap();
        di.ingest(make_record("binance", "BTCUSDT", 0, 100.0, 1.0))
            .unwrap();
        di.tick();
        di.reset_stats();
        let stats = di.source_stats("binance").unwrap();
        assert_eq!(stats.total_ingested, 0);
        assert!((di.ema_throughput() - 0.0).abs() < 1e-10);
        // Source still registered
        assert_eq!(di.source_count(), 1);
    }

    #[test]
    fn test_reset() {
        let mut di = DataIngestion::new();
        di.register_source("binance", trade_schema()).unwrap();
        di.ingest(make_record("binance", "BTCUSDT", 0, 100.0, 1.0))
            .unwrap();
        di.tick();
        di.reset();
        assert_eq!(di.source_count(), 0);
        assert_eq!(di.current_tick(), 0);
        assert!(di.staging_is_empty());
    }

    // -----------------------------------------------------------------------
    // DataRecord API
    // -----------------------------------------------------------------------

    #[test]
    fn test_data_record_field() {
        let r = DataRecord::new("src", "sym", 100)
            .with_field("price", 42.0)
            .with_field("volume", 7.0);
        assert!((r.field("price").unwrap() - 42.0).abs() < 1e-10);
        assert!(r.field("nonexistent").is_none());
    }

    #[test]
    fn test_data_record_set_field() {
        let mut r = DataRecord::new("src", "sym", 100);
        r.set_field("price", 99.0);
        assert!((r.field("price").unwrap() - 99.0).abs() < 1e-10);
    }

    // -----------------------------------------------------------------------
    // FieldSpec API
    // -----------------------------------------------------------------------

    #[test]
    fn test_field_spec_required() {
        let f = FieldSpec::required("price");
        assert!(f.required);
        assert_eq!(f.norm, NormStrategy::None);
    }

    #[test]
    fn test_field_spec_optional() {
        let f = FieldSpec::optional("side");
        assert!(!f.required);
    }

    #[test]
    fn test_field_spec_with_norm() {
        let f = FieldSpec::required("price").with_norm(NormStrategy::MinMax);
        assert_eq!(f.norm, NormStrategy::MinMax);
    }

    // -----------------------------------------------------------------------
    // Quality score
    // -----------------------------------------------------------------------

    #[test]
    fn test_quality_score_perfect() {
        let mut di = DataIngestion::new();
        di.register_source("binance", trade_schema()).unwrap();
        for i in 0..10 {
            di.ingest(make_record("binance", "BTCUSDT", i, 100.0, 1.0))
                .unwrap();
        }
        let stats = di.source_stats("binance").unwrap();
        assert!((stats.quality_score - 1.0).abs() < 1e-10);
    }

    #[test]
    fn test_quality_score_empty() {
        let mut di = DataIngestion::new();
        di.register_source("binance", trade_schema()).unwrap();
        let stats = di.source_stats("binance").unwrap();
        assert!((stats.quality_score - 1.0).abs() < 1e-10);
    }

    // -----------------------------------------------------------------------
    // Extra fields not in schema
    // -----------------------------------------------------------------------

    #[test]
    fn test_extra_fields_passed_through() {
        let mut di = DataIngestion::new();
        di.register_source("binance", trade_schema()).unwrap();
        let record =
            make_record("binance", "BTCUSDT", 0, 100.0, 1.0).with_field("extra_metric", 42.0);
        di.ingest(record).unwrap();
        let records = di.drain_all();
        assert!((records[0].fields["extra_metric"] - 42.0).abs() < 1e-10);
    }

    // -----------------------------------------------------------------------
    // Source dedup tracking
    // -----------------------------------------------------------------------

    #[test]
    fn test_source_dedup_count() {
        let mut di = DataIngestion::new();
        di.register_source("binance", trade_schema()).unwrap();
        di.ingest(make_record("binance", "BTCUSDT", 0, 100.0, 1.0))
            .unwrap();
        di.ingest(make_record("binance", "BTCUSDT", 0, 100.0, 1.0))
            .unwrap(); // duplicate
        let stats = di.source_stats("binance").unwrap();
        assert_eq!(stats.total_deduped, 1);
    }

    // -----------------------------------------------------------------------
    // Source order preserved
    // -----------------------------------------------------------------------

    #[test]
    fn test_source_order() {
        let mut di = DataIngestion::new();
        di.register_source("c", trade_schema()).unwrap();
        di.register_source("a", trade_schema()).unwrap();
        di.register_source("b", trade_schema()).unwrap();
        assert_eq!(di.source_names(), &["c", "a", "b"]);
    }

    #[test]
    fn test_deregister_preserves_order() {
        let mut di = DataIngestion::new();
        di.register_source("a", trade_schema()).unwrap();
        di.register_source("b", trade_schema()).unwrap();
        di.register_source("c", trade_schema()).unwrap();
        di.deregister_source("b").unwrap();
        assert_eq!(di.source_names(), &["a", "c"]);
    }

    // -----------------------------------------------------------------------
    // Running statistics internals
    // -----------------------------------------------------------------------

    #[test]
    fn test_field_stats_single_value() {
        let mut stats = FieldStats::new();
        stats.update(5.0);
        assert!((stats.mean - 5.0).abs() < 1e-10);
        assert!((stats.std_dev() - 0.0).abs() < 1e-10);
        assert!((stats.min - 5.0).abs() < 1e-10);
        assert!((stats.max - 5.0).abs() < 1e-10);
    }

    #[test]
    fn test_field_stats_reset() {
        let mut stats = FieldStats::new();
        stats.update(10.0);
        stats.update(20.0);
        stats.reset();
        assert_eq!(stats.count, 0);
        assert!((stats.mean - 0.0).abs() < 1e-10);
        assert_eq!(stats.min, f64::INFINITY);
    }

    #[test]
    fn test_normalise_minmax_constant() {
        let mut stats = FieldStats::new();
        stats.update(5.0);
        stats.update(5.0);
        // constant value → normalises to 0.5
        let n = stats.normalise(5.0, NormStrategy::MinMax);
        assert!((n - 0.5).abs() < 1e-10);
    }

    #[test]
    fn test_normalise_zscore_constant() {
        let mut stats = FieldStats::new();
        stats.update(5.0);
        stats.update(5.0);
        // constant value → z-score is 0
        let n = stats.normalise(5.0, NormStrategy::ZScore);
        assert!((n - 0.0).abs() < 1e-10);
    }
}

