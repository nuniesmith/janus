//! Qdrant Production Client — Real vector database integration
//!
//! Replaces the mock storage layer in `vector_db.rs` with a full production
//! Qdrant client using the `qdrant-client` crate. Provides reliable vector
//! storage and similarity search for market regime embeddings, sentiment
//! embeddings, and episodic memory vectors.
//!
//! # Architecture
//!
//! ```text
//! ┌─────────────────────────────────────────────────────────────┐
//! │                  Qdrant Production Client                    │
//! ├─────────────────────────────────────────────────────────────┤
//! │                                                              │
//! │  ┌──────────────┐   ┌──────────────┐   ┌──────────────┐    │
//! │  │  Hippocampus │   │   Cortex     │   │  Thalamus    │    │
//! │  │  (Episodes)  │   │  (Schemas)   │   │ (Sentiment)  │    │
//! │  └──────┬───────┘   └──────┬───────┘   └──────┬───────┘    │
//! │         │                  │                   │            │
//! │         └──────────────────┼───────────────────┘            │
//! │                            ▼                                │
//! │                   ┌────────────────┐                        │
//! │                   │ QdrantClient   │                        │
//! │                   │ • upsert()     │                        │
//! │                   │ • search()     │                        │
//! │                   │ • delete()     │                        │
//! │                   │ • scroll()     │                        │
//! │                   └────────┬───────┘                        │
//! │                            │                                │
//! │                   ┌────────▼───────┐                        │
//! │                   │ Connection     │                        │
//! │                   │ Pool + Retry   │                        │
//! │                   └────────┬───────┘                        │
//! │                            │                                │
//! └────────────────────────────┼────────────────────────────────┘
//!                              │ gRPC / HTTP
//!                     ┌────────▼───────┐
//!                     │  Qdrant Server │
//!                     │  (Docker/K8s)  │
//!                     └────────────────┘
//! ```
//!
//! # Collections
//!
//! The client manages several Qdrant collections:
//!
//! | Collection            | Vector Dim | Use Case                        |
//! |-----------------------|------------|---------------------------------|
//! | `market_regimes`      | 64         | Regime embedding similarity     |
//! | `episodic_memory`     | 128        | Experience replay retrieval     |
//! | `sentiment_embeddings`| 768        | News/text embedding similarity  |
//! | `schema_prototypes`   | 64         | Regime schema cluster centres   |
//!
//! # Integration Points
//!
//! - **Hippocampus**: Stores and retrieves episodic memories for experience replay.
//!   During consolidation (backward service), similar past experiences are fetched
//!   to prioritise replay of relevant episodes.
//! - **Cortex/Schemas**: Stores regime schema prototypes. During classification,
//!   the nearest schema is retrieved to determine the current market regime.
//! - **Thalamus/Sentiment**: Stores BERT [CLS] embeddings from news articles.
//!   Enables "find similar news" queries for regime-aware sentiment aggregation.
//! - **Visual Cortex/UMAP**: After parametric UMAP projection, low-dimensional
//!   embeddings are stored for real-time schema monitoring.
//!
//! # Retry & Resilience
//!
//! The client implements:
//! - Exponential backoff with jitter on transient failures
//! - Circuit breaker pattern (open after N consecutive failures)
//! - Automatic fallback to mock storage when Qdrant is unreachable
//! - Health check probes for CNS monitoring integration

use common::{JanusError, MarketRegime, Result};
use qdrant_client::Qdrant;
use qdrant_client::qdrant::{
    CreateCollectionBuilder, DeletePointsBuilder, Distance, Filter, GetPointsBuilder, PointId,
    PointStruct, ScrollPointsBuilder, SearchPointsBuilder, UpsertPointsBuilder,
    Value as QdrantValue, VectorParamsBuilder, point_id::PointIdOptions,
    value::Kind as QdrantValueKind,
};
use serde::{Deserialize, Serialize};
use std::collections::HashMap;
use std::sync::Arc;
use std::time::Duration;
use tokio::sync::RwLock;
use tracing::{debug, error, info, warn};

// ---------------------------------------------------------------------------
// Configuration
// ---------------------------------------------------------------------------

/// Configuration for the production Qdrant client.
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct QdrantProductionConfig {
    /// Qdrant server URL (gRPC endpoint). Example: `http://localhost:6334`
    pub url: String,

    /// Optional API key for Qdrant Cloud or authenticated deployments.
    pub api_key: Option<String>,

    /// Connection timeout in seconds.
    pub connect_timeout_secs: u64,

    /// Request timeout in seconds (per-operation).
    pub request_timeout_secs: u64,

    /// Maximum number of retry attempts for transient failures.
    pub max_retries: u32,

    /// Base delay for exponential backoff (milliseconds).
    pub retry_base_delay_ms: u64,

    /// Maximum backoff delay (milliseconds).
    pub retry_max_delay_ms: u64,

    /// Whether to add jitter to retry delays.
    pub retry_jitter: bool,

    /// Circuit breaker: number of consecutive failures before opening the circuit.
    pub circuit_breaker_threshold: u32,

    /// Circuit breaker: duration to keep the circuit open before half-opening (seconds).
    pub circuit_breaker_reset_secs: u64,

    /// Whether to fall back to in-memory mock storage when Qdrant is unreachable.
    pub fallback_to_mock: bool,

    /// Whether to auto-create collections on first use if they don't exist.
    pub auto_create_collections: bool,

    /// Default distance metric for new collections.
    pub default_distance: DistanceMetric,

    /// Write consistency factor (number of replicas that must acknowledge a write).
    /// Set to `None` for default Qdrant behaviour.
    pub write_consistency: Option<u32>,

    /// Prefix for collection names. Useful for multi-tenant or staging environments.
    /// Example: `"janus_prod_"` → collection `"janus_prod_market_regimes"`.
    pub collection_prefix: String,

    /// Whether to enable TLS for the Qdrant connection.
    pub tls_enabled: bool,

    /// Path to a custom CA certificate file (PEM) for verifying the server.
    /// If `None`, the system's default root certificates are used.
    pub tls_ca_cert_path: Option<String>,

    /// Path to a client certificate file (PEM) for mutual TLS authentication.
    pub tls_client_cert_path: Option<String>,

    /// Path to a client private key file (PEM) for mutual TLS authentication.
    pub tls_client_key_path: Option<String>,

    /// Override the TLS server name used for certificate verification.
    /// Useful when connecting via IP address or a load balancer with a
    /// different hostname than the certificate's CN/SAN.
    pub tls_server_name: Option<String>,
}

/// Distance metric for vector similarity.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Serialize, Deserialize)]
pub enum DistanceMetric {
    /// Cosine similarity (most common for normalised embeddings).
    Cosine,
    /// Euclidean (L2) distance.
    Euclid,
    /// Dot product (for unnormalised vectors where magnitude matters).
    Dot,
    /// Manhattan (L1) distance.
    Manhattan,
}

impl DistanceMetric {
    /// Convert to the Qdrant protobuf Distance enum.
    fn to_qdrant(self) -> Distance {
        match self {
            Self::Cosine => Distance::Cosine,
            Self::Euclid => Distance::Euclid,
            Self::Dot => Distance::Dot,
            Self::Manhattan => Distance::Manhattan,
        }
    }
}

impl Default for QdrantProductionConfig {
    fn default() -> Self {
        Self {
            url: "http://localhost:6334".to_string(),
            api_key: None,
            connect_timeout_secs: 10,
            request_timeout_secs: 30,
            max_retries: 3,
            retry_base_delay_ms: 100,
            retry_max_delay_ms: 5000,
            retry_jitter: true,
            circuit_breaker_threshold: 5,
            circuit_breaker_reset_secs: 60,
            fallback_to_mock: true,
            auto_create_collections: true,
            default_distance: DistanceMetric::Cosine,
            write_consistency: None,
            collection_prefix: String::new(),
            tls_enabled: false,
            tls_ca_cert_path: None,
            tls_client_cert_path: None,
            tls_client_key_path: None,
            tls_server_name: None,
        }
    }
}

impl QdrantProductionConfig {
    /// Configuration for local Docker development.
    pub fn local() -> Self {
        Self::default()
    }

    /// Configuration for Qdrant Cloud.
    pub fn cloud(url: &str, api_key: &str) -> Self {
        Self {
            url: url.to_string(),
            api_key: Some(api_key.to_string()),
            connect_timeout_secs: 15,
            request_timeout_secs: 60,
            max_retries: 5,
            fallback_to_mock: false,
            ..Default::default()
        }
    }

    /// Get the full collection name with prefix applied.
    pub fn collection_name(&self, base_name: &str) -> String {
        if self.collection_prefix.is_empty() {
            base_name.to_string()
        } else {
            format!("{}{}", self.collection_prefix, base_name)
        }
    }
}

// ---------------------------------------------------------------------------
// Collection Definitions
// ---------------------------------------------------------------------------

/// Well-known collection names used by the JANUS system.
pub mod collections {
    /// Market regime embeddings (from cortex/schemas and regime detector).
    pub const MARKET_REGIMES: &str = "market_regimes";

    /// Episodic memory vectors (from hippocampus experience buffer).
    pub const EPISODIC_MEMORY: &str = "episodic_memory";

    /// Sentiment text embeddings (from thalamus BERT pipeline).
    pub const SENTIMENT_EMBEDDINGS: &str = "sentiment_embeddings";

    /// Schema prototype vectors (from cortex/memory/schemas).
    pub const SCHEMA_PROTOTYPES: &str = "schema_prototypes";
}

/// Specification for a Qdrant collection.
#[derive(Debug, Clone)]
pub struct CollectionSpec {
    /// Base name of the collection (before prefix).
    pub name: String,
    /// Dimensionality of vectors in this collection.
    pub vector_dim: u64,
    /// Distance metric for similarity search.
    pub distance: DistanceMetric,
    /// Whether to create an on-disk payload index for filtering.
    pub payload_index_fields: Vec<String>,
}

impl CollectionSpec {
    pub fn new(name: &str, vector_dim: u64, distance: DistanceMetric) -> Self {
        Self {
            name: name.to_string(),
            vector_dim,
            distance,
            payload_index_fields: Vec::new(),
        }
    }

    pub fn with_payload_index(mut self, field: &str) -> Self {
        self.payload_index_fields.push(field.to_string());
        self
    }
}

/// Default collection specifications for the JANUS system.
pub fn default_collection_specs() -> Vec<CollectionSpec> {
    vec![
        CollectionSpec::new(collections::MARKET_REGIMES, 64, DistanceMetric::Cosine)
            .with_payload_index("regime_name")
            .with_payload_index("timestamp"),
        CollectionSpec::new(collections::EPISODIC_MEMORY, 128, DistanceMetric::Cosine)
            .with_payload_index("reward")
            .with_payload_index("timestamp")
            .with_payload_index("regime"),
        CollectionSpec::new(
            collections::SENTIMENT_EMBEDDINGS,
            768,
            DistanceMetric::Cosine,
        )
        .with_payload_index("source")
        .with_payload_index("timestamp")
        .with_payload_index("sentiment_score"),
        CollectionSpec::new(collections::SCHEMA_PROTOTYPES, 64, DistanceMetric::Cosine)
            .with_payload_index("regime_id")
            .with_payload_index("priority"),
    ]
}

// ---------------------------------------------------------------------------
// Circuit Breaker
// ---------------------------------------------------------------------------

/// Circuit breaker state for resilient Qdrant connectivity.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum CircuitState {
    /// Normal operation — requests go through to Qdrant.
    Closed,
    /// Too many failures — requests are rejected immediately (or fall back to mock).
    Open,
    /// Testing recovery — a single request is allowed through to check if Qdrant is back.
    HalfOpen,
}

impl std::fmt::Display for CircuitState {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        match self {
            Self::Closed => write!(f, "Closed"),
            Self::Open => write!(f, "Open"),
            Self::HalfOpen => write!(f, "HalfOpen"),
        }
    }
}

/// Internal circuit breaker tracker.
struct CircuitBreaker {
    state: CircuitState,
    consecutive_failures: u32,
    threshold: u32,
    last_failure_time: Option<std::time::Instant>,
    reset_duration: Duration,
    total_trips: u64,
}

impl CircuitBreaker {
    fn new(threshold: u32, reset_secs: u64) -> Self {
        Self {
            state: CircuitState::Closed,
            consecutive_failures: 0,
            threshold,
            last_failure_time: None,
            reset_duration: Duration::from_secs(reset_secs),
            total_trips: 0,
        }
    }

    /// Check if a request should be allowed through.
    fn should_allow(&mut self) -> bool {
        match self.state {
            CircuitState::Closed => true,
            CircuitState::Open => {
                // Check if enough time has passed to try again
                if let Some(last_fail) = self.last_failure_time {
                    if last_fail.elapsed() >= self.reset_duration {
                        self.state = CircuitState::HalfOpen;
                        debug!("Circuit breaker transitioning to HalfOpen");
                        true
                    } else {
                        false
                    }
                } else {
                    // No recorded failure time — shouldn't happen, but allow
                    self.state = CircuitState::HalfOpen;
                    true
                }
            }
            CircuitState::HalfOpen => {
                // Allow one request through to test
                true
            }
        }
    }

    /// Record a successful operation.
    fn record_success(&mut self) {
        self.consecutive_failures = 0;
        if self.state != CircuitState::Closed {
            info!("Circuit breaker closing — Qdrant connectivity restored");
        }
        self.state = CircuitState::Closed;
    }

    /// Record a failed operation.
    fn record_failure(&mut self) {
        self.consecutive_failures += 1;
        self.last_failure_time = Some(std::time::Instant::now());

        if self.consecutive_failures >= self.threshold && self.state != CircuitState::Open {
            self.state = CircuitState::Open;
            self.total_trips += 1;
            warn!(
                "Circuit breaker OPEN after {} consecutive failures (trip #{})",
                self.consecutive_failures, self.total_trips
            );
        }
    }

    fn state(&self) -> CircuitState {
        self.state
    }
}

// ---------------------------------------------------------------------------
// Search Result Types
// ---------------------------------------------------------------------------

/// A single search result from Qdrant with score and payload.
#[derive(Debug, Clone)]
pub struct ScoredPoint {
    /// Point ID (UUID string).
    pub id: String,
    /// Similarity score (higher = more similar for cosine/dot, lower for euclid).
    pub score: f64,
    /// The vector itself (if requested).
    pub vector: Option<Vec<f64>>,
    /// Payload fields as key-value pairs.
    pub payload: HashMap<String, PayloadValue>,
}

/// A payload value that can be stored in Qdrant.
#[derive(Debug, Clone, Serialize, Deserialize)]
pub enum PayloadValue {
    /// String value.
    String(String),
    /// Integer value.
    Integer(i64),
    /// Float value.
    Float(f64),
    /// Boolean value.
    Bool(bool),
    /// List of strings.
    StringList(Vec<String>),
    /// Null / missing value.
    Null,
}

impl PayloadValue {
    /// Convert to a Qdrant protobuf Value.
    pub fn to_qdrant_value(&self) -> QdrantValue {
        match self {
            Self::String(s) => QdrantValue {
                kind: Some(QdrantValueKind::StringValue(s.clone())),
            },
            Self::Integer(i) => QdrantValue {
                kind: Some(QdrantValueKind::IntegerValue(*i)),
            },
            Self::Float(f) => QdrantValue {
                kind: Some(QdrantValueKind::DoubleValue(*f)),
            },
            Self::Bool(b) => QdrantValue {
                kind: Some(QdrantValueKind::BoolValue(*b)),
            },
            Self::StringList(items) => {
                let values: Vec<QdrantValue> = items
                    .iter()
                    .map(|s| QdrantValue {
                        kind: Some(QdrantValueKind::StringValue(s.clone())),
                    })
                    .collect();
                QdrantValue {
                    kind: Some(QdrantValueKind::ListValue(
                        qdrant_client::qdrant::ListValue { values },
                    )),
                }
            }
            Self::Null => QdrantValue {
                kind: Some(QdrantValueKind::NullValue(0)),
            },
        }
    }

    /// Try to extract from a Qdrant protobuf Value.
    pub fn from_qdrant_value(value: &QdrantValue) -> Self {
        match &value.kind {
            Some(QdrantValueKind::StringValue(s)) => Self::String(s.clone()),
            Some(QdrantValueKind::IntegerValue(i)) => Self::Integer(*i),
            Some(QdrantValueKind::DoubleValue(f)) => Self::Float(*f),
            Some(QdrantValueKind::BoolValue(b)) => Self::Bool(*b),
            Some(QdrantValueKind::NullValue(_)) | None => Self::Null,
            Some(QdrantValueKind::ListValue(list)) => {
                // Attempt to extract as a list of strings. If any element is
                // not a string we fall back to Null (we only model StringList).
                let strings: Vec<String> = list
                    .values
                    .iter()
                    .filter_map(|v| match &v.kind {
                        Some(QdrantValueKind::StringValue(s)) => Some(s.clone()),
                        _ => None,
                    })
                    .collect();
                if strings.len() == list.values.len() {
                    Self::StringList(strings)
                } else {
                    // Mixed-type list — degrade gracefully to Null.
                    Self::Null
                }
            }
            _ => {
                // StructValue and other complex nested payloads are not yet
                // modelled in PayloadValue. Fall back to Null.
                Self::Null
            }
        }
    }

    /// Get as string, if applicable.
    pub fn as_str(&self) -> Option<&str> {
        match self {
            Self::String(s) => Some(s.as_str()),
            _ => None,
        }
    }

    /// Get as f64, if applicable.
    pub fn as_f64(&self) -> Option<f64> {
        match self {
            Self::Float(f) => Some(*f),
            Self::Integer(i) => Some(*i as f64),
            _ => None,
        }
    }

    /// Get as i64, if applicable.
    pub fn as_i64(&self) -> Option<i64> {
        match self {
            Self::Integer(i) => Some(*i),
            _ => None,
        }
    }

    /// Get as bool, if applicable.
    pub fn as_bool(&self) -> Option<bool> {
        match self {
            Self::Bool(b) => Some(*b),
            _ => None,
        }
    }
}

/// A point to upsert into Qdrant.
#[derive(Debug, Clone)]
pub struct UpsertPoint {
    /// Point ID (UUID string). If empty, a new UUID will be generated.
    pub id: String,
    /// The vector to store.
    pub vector: Vec<f64>,
    /// Payload key-value pairs.
    pub payload: HashMap<String, PayloadValue>,
}

impl UpsertPoint {
    /// Create a new upsert point with an auto-generated UUID.
    pub fn new(vector: Vec<f64>) -> Self {
        Self {
            id: uuid::Uuid::new_v4().to_string(),
            vector,
            payload: HashMap::new(),
        }
    }

    /// Create with a specific ID.
    pub fn with_id(id: &str, vector: Vec<f64>) -> Self {
        Self {
            id: id.to_string(),
            vector,
            payload: HashMap::new(),
        }
    }

    /// Add a string payload field.
    pub fn with_string(mut self, key: &str, value: &str) -> Self {
        self.payload
            .insert(key.to_string(), PayloadValue::String(value.to_string()));
        self
    }

    /// Add a float payload field.
    pub fn with_float(mut self, key: &str, value: f64) -> Self {
        self.payload
            .insert(key.to_string(), PayloadValue::Float(value));
        self
    }

    /// Add an integer payload field.
    pub fn with_integer(mut self, key: &str, value: i64) -> Self {
        self.payload
            .insert(key.to_string(), PayloadValue::Integer(value));
        self
    }

    /// Add a boolean payload field.
    pub fn with_bool(mut self, key: &str, value: bool) -> Self {
        self.payload
            .insert(key.to_string(), PayloadValue::Bool(value));
        self
    }

    /// Convert to a Qdrant PointStruct.
    fn to_qdrant_point(&self) -> PointStruct {
        let payload: HashMap<String, QdrantValue> = self
            .payload
            .iter()
            .map(|(k, v)| (k.clone(), v.to_qdrant_value()))
            .collect();

        let vector_f32: Vec<f32> = self.vector.iter().map(|&v| v as f32).collect();

        PointStruct::new(self.id.clone(), vector_f32, payload)
    }
}

// ---------------------------------------------------------------------------
// Statistics
// ---------------------------------------------------------------------------

/// Operational statistics for the Qdrant production client.
#[derive(Debug, Clone, Default)]
pub struct QdrantClientStats {
    /// Total upsert operations attempted.
    pub upserts_attempted: u64,
    /// Total upsert operations succeeded.
    pub upserts_succeeded: u64,
    /// Total points upserted.
    pub points_upserted: u64,
    /// Total search operations attempted.
    pub searches_attempted: u64,
    /// Total search operations succeeded.
    pub searches_succeeded: u64,
    /// Total delete operations attempted.
    pub deletes_attempted: u64,
    /// Total delete operations succeeded.
    pub deletes_succeeded: u64,
    /// Total retries across all operations.
    pub total_retries: u64,
    /// Total circuit breaker trips.
    pub circuit_breaker_trips: u64,
    /// Total operations that fell back to mock storage.
    pub mock_fallbacks: u64,
    /// Total errors (after all retries exhausted).
    pub total_errors: u64,
    /// Cumulative search latency in milliseconds (for averaging).
    pub cumulative_search_latency_ms: f64,
    /// Cumulative upsert latency in milliseconds.
    pub cumulative_upsert_latency_ms: f64,
    /// Number of collections created.
    pub collections_created: u64,
}

impl QdrantClientStats {
    /// Average search latency in milliseconds.
    pub fn avg_search_latency_ms(&self) -> f64 {
        if self.searches_succeeded == 0 {
            return 0.0;
        }
        self.cumulative_search_latency_ms / self.searches_succeeded as f64
    }

    /// Average upsert latency in milliseconds.
    pub fn avg_upsert_latency_ms(&self) -> f64 {
        if self.upserts_succeeded == 0 {
            return 0.0;
        }
        self.cumulative_upsert_latency_ms / self.upserts_succeeded as f64
    }

    /// Upsert success rate.
    pub fn upsert_success_rate(&self) -> f64 {
        if self.upserts_attempted == 0 {
            return 1.0;
        }
        self.upserts_succeeded as f64 / self.upserts_attempted as f64
    }

    /// Search success rate.
    pub fn search_success_rate(&self) -> f64 {
        if self.searches_attempted == 0 {
            return 1.0;
        }
        self.searches_succeeded as f64 / self.searches_attempted as f64
    }

    /// Error rate across all operations.
    pub fn error_rate(&self) -> f64 {
        let total = self.upserts_attempted + self.searches_attempted + self.deletes_attempted;
        if total == 0 {
            return 0.0;
        }
        self.total_errors as f64 / total as f64
    }
}

// ---------------------------------------------------------------------------
// Production Client
// ---------------------------------------------------------------------------

/// Production Qdrant vector database client.
///
/// This client wraps the `qdrant-client` crate with:
/// - Automatic retry with exponential backoff
/// - Circuit breaker pattern for resilience
/// - Optional mock fallback when Qdrant is unreachable
/// - Statistics tracking for CNS monitoring
/// - Collection auto-creation with correct schemas
///
/// # Usage
///
/// ```rust,ignore
/// use memory::qdrant_client::*;
///
/// let config = QdrantProductionConfig::local();
/// let client = QdrantProductionClient::connect(config).await?;
///
/// // Ensure collections exist
/// client.ensure_collections(&default_collection_specs()).await?;
///
/// // Upsert a vector
/// let point = UpsertPoint::with_id("regime_001", vec![0.1; 64])
///     .with_string("regime_name", "bull")
///     .with_float("volatility", 0.02);
/// client.upsert(collections::MARKET_REGIMES, &[point]).await?;
///
/// // Search for similar
/// let results = client.search(
///     collections::MARKET_REGIMES,
///     &vec![0.1; 64],
///     10,
///     None,
/// ).await?;
/// ```
pub struct QdrantProductionClient {
    config: QdrantProductionConfig,
    /// The underlying Qdrant gRPC client, wrapped in `Arc<RwLock<..>>` to
    /// allow hot-swap reconnection without restarting the service.
    /// `None` if connection failed and we're operating in mock-fallback mode.
    client: Arc<RwLock<Option<Qdrant>>>,
    /// Circuit breaker for connection resilience.
    circuit_breaker: Arc<RwLock<CircuitBreaker>>,
    /// Operational statistics.
    stats: Arc<RwLock<QdrantClientStats>>,
    /// Whether we're currently in mock-fallback mode.
    using_mock: Arc<RwLock<bool>>,
    /// In-memory fallback storage (used when Qdrant is unreachable).
    /// Maps collection_name → Vec<(id, vector, payload)>.
    mock_storage: Arc<RwLock<HashMap<String, Vec<MockPoint>>>>,
}

/// Internal mock point for fallback storage.
#[derive(Debug, Clone)]
struct MockPoint {
    id: String,
    vector: Vec<f64>,
    payload: HashMap<String, PayloadValue>,
}

impl QdrantProductionClient {
    /// Connect to a Qdrant server with the given configuration.
    ///
    /// If connection fails and `fallback_to_mock` is enabled, the client will
    /// operate in mock mode until Qdrant becomes available again.
    pub async fn connect(config: QdrantProductionConfig) -> Result<Self> {
        let circuit_breaker = Arc::new(RwLock::new(CircuitBreaker::new(
            config.circuit_breaker_threshold,
            config.circuit_breaker_reset_secs,
        )));

        let client_result = Self::create_qdrant_client(&config).await;

        let (client, using_mock) = match client_result {
            Ok(c) => {
                info!("Connected to Qdrant at {}", config.url);
                (Some(c), false)
            }
            Err(e) => {
                if config.fallback_to_mock {
                    warn!(
                        "Failed to connect to Qdrant at {}: {}. Using mock fallback.",
                        config.url, e
                    );
                    (None, true)
                } else {
                    return Err(JanusError::Memory(format!(
                        "Failed to connect to Qdrant at {}: {}",
                        config.url, e
                    )));
                }
            }
        };

        Ok(Self {
            config,
            client: Arc::new(RwLock::new(client)),
            circuit_breaker,
            stats: Arc::new(RwLock::new(QdrantClientStats::default())),
            using_mock: Arc::new(RwLock::new(using_mock)),
            mock_storage: Arc::new(RwLock::new(HashMap::new())),
        })
    }

    /// Create a mock-only client for testing (no Qdrant connection).
    pub fn mock() -> Self {
        Self {
            config: QdrantProductionConfig::default(),
            client: Arc::new(RwLock::new(None)),
            circuit_breaker: Arc::new(RwLock::new(CircuitBreaker::new(5, 60))),
            stats: Arc::new(RwLock::new(QdrantClientStats::default())),
            using_mock: Arc::new(RwLock::new(true)),
            mock_storage: Arc::new(RwLock::new(HashMap::new())),
        }
    }

    /// Create the underlying Qdrant gRPC client.
    ///
    /// When `tls_enabled` is set in the config, the URL scheme is
    /// automatically upgraded to `https://` if it was `http://`. The
    /// qdrant-client crate uses tonic under the hood which negotiates
    /// TLS via the URL scheme. Custom CA certificates and mutual TLS
    /// require a tonic-level configuration that the qdrant-client
    /// builder does not expose; if those fields are set in the config
    /// a warning is logged.
    async fn create_qdrant_client(config: &QdrantProductionConfig) -> Result<Qdrant> {
        // Resolve URL — upgrade to https if TLS is enabled.
        let url = if config.tls_enabled && config.url.starts_with("http://") {
            let upgraded = config.url.replacen("http://", "https://", 1);
            info!(
                "TLS enabled — upgraded Qdrant URL from {} to {}",
                config.url, upgraded
            );
            upgraded
        } else {
            config.url.clone()
        };

        // Warn about unsupported custom TLS options (qdrant-client 1.x
        // does not expose tonic TLS config through its builder).
        if config.tls_ca_cert_path.is_some()
            || config.tls_client_cert_path.is_some()
            || config.tls_client_key_path.is_some()
        {
            warn!(
                "Custom TLS certificates (CA / client cert / client key) are configured but \
                 the qdrant-client crate does not expose tonic TLS configuration. \
                 Only system-default root CAs will be used for verification. \
                 To use custom certificates, build a tonic Channel directly and pass it \
                 to the Qdrant client."
            );
        }

        let mut builder =
            Qdrant::from_url(&url).timeout(Duration::from_secs(config.request_timeout_secs));

        if let Some(ref api_key) = config.api_key {
            builder = builder.api_key(api_key.clone());
        }

        let client = builder
            .build()
            .map_err(|e| JanusError::Memory(format!("Failed to build Qdrant client: {}", e)))?;

        // Health-check: list collections to verify the connection is alive.
        client
            .list_collections()
            .await
            .map_err(|e| JanusError::Memory(format!("Qdrant health check failed: {}", e)))?;

        info!("Qdrant client built and health-checked successfully");
        Ok(client)
    }

    // -----------------------------------------------------------------------
    // Collection Management
    // -----------------------------------------------------------------------

    /// Ensure a collection exists, creating it if necessary.
    pub async fn ensure_collection(&self, spec: &CollectionSpec) -> Result<()> {
        let collection_name = self.config.collection_name(&spec.name);

        {
            let client_guard = self.client.read().await;
            if let Some(ref client) = *client_guard
                && !*self.using_mock.read().await
            {
                // 1. Check if collection already exists
                let exists = client
                    .collection_exists(&collection_name)
                    .await
                    .unwrap_or(false);

                if !exists {
                    // 2. Create collection with vector config
                    let vectors_config =
                        VectorParamsBuilder::new(spec.vector_dim, spec.distance.to_qdrant());
                    client
                        .create_collection(
                            CreateCollectionBuilder::new(&collection_name)
                                .vectors_config(vectors_config),
                        )
                        .await
                        .map_err(|e| {
                            JanusError::Memory(format!(
                                "Failed to create collection '{}': {}",
                                collection_name, e
                            ))
                        })?;

                    info!("Created Qdrant collection '{}'", collection_name);

                    // 3. Update stats
                    let mut stats = self.stats.write().await;
                    stats.collections_created += 1;
                } else {
                    debug!("Collection '{}' already exists", collection_name);
                }

                return Ok(());
            }
        }

        // Mock mode: just ensure the map entry exists
        let mut storage = self.mock_storage.write().await;
        storage.entry(collection_name).or_insert_with(Vec::new);
        Ok(())
    }

    /// Ensure all default JANUS collections exist.
    pub async fn ensure_collections(&self, specs: &[CollectionSpec]) -> Result<()> {
        for spec in specs {
            self.ensure_collection(spec).await?;
        }
        Ok(())
    }

    // -----------------------------------------------------------------------
    // Upsert
    // -----------------------------------------------------------------------

    /// Upsert one or more points into a collection.
    ///
    /// This operation is idempotent: re-upserting a point with the same ID
    /// will overwrite the previous vector and payload.
    pub async fn upsert(&self, collection: &str, points: &[UpsertPoint]) -> Result<()> {
        if points.is_empty() {
            return Ok(());
        }

        let collection_name = self.config.collection_name(collection);
        let mut stats = self.stats.write().await;
        stats.upserts_attempted += 1;
        let point_count = points.len() as u64;
        drop(stats);

        let start = std::time::Instant::now();

        // Check circuit breaker
        {
            let mut cb = self.circuit_breaker.write().await;
            if !cb.should_allow() {
                if self.config.fallback_to_mock {
                    let mut stats = self.stats.write().await;
                    stats.mock_fallbacks += 1;
                    drop(stats);
                    return self.mock_upsert(&collection_name, points).await;
                }
                return Err(JanusError::Memory(
                    "Qdrant circuit breaker is open — upsert rejected".into(),
                ));
            }
        }

        // If we're in mock mode, go straight to mock
        if *self.using_mock.read().await || self.client.read().await.is_none() {
            return self.mock_upsert(&collection_name, points).await;
        }

        // Real Qdrant upsert with retry
        let result = self
            .with_retry(|| async {
                let client_guard = self.client.read().await;
                let client = client_guard
                    .as_ref()
                    .ok_or_else(|| JanusError::Memory("No Qdrant client".into()))?;

                let qdrant_points: Vec<PointStruct> =
                    points.iter().map(|p| p.to_qdrant_point()).collect();

                client
                    .upsert_points(
                        UpsertPointsBuilder::new(&collection_name, qdrant_points).wait(true),
                    )
                    .await
                    .map_err(|e| JanusError::Memory(format!("Qdrant upsert failed: {}", e)))?;

                Ok(())
            })
            .await;

        let elapsed_ms = start.elapsed().as_secs_f64() * 1000.0;

        match &result {
            Ok(()) => {
                let mut cb = self.circuit_breaker.write().await;
                cb.record_success();
                drop(cb);

                let mut stats = self.stats.write().await;
                stats.upserts_succeeded += 1;
                stats.points_upserted += point_count;
                stats.cumulative_upsert_latency_ms += elapsed_ms;
            }
            Err(_) => {
                let mut cb = self.circuit_breaker.write().await;
                cb.record_failure();
                drop(cb);

                let mut stats = self.stats.write().await;
                stats.total_errors += 1;

                // Fall back to mock if configured
                if self.config.fallback_to_mock {
                    stats.mock_fallbacks += 1;
                    drop(stats);
                    return self.mock_upsert(&collection_name, points).await;
                }
            }
        }

        result
    }

    /// Upsert into mock storage.
    async fn mock_upsert(&self, collection: &str, points: &[UpsertPoint]) -> Result<()> {
        let mut storage = self.mock_storage.write().await;
        let coll = storage
            .entry(collection.to_string())
            .or_insert_with(Vec::new);

        for point in points {
            // Remove existing point with same ID
            coll.retain(|p| p.id != point.id);
            coll.push(MockPoint {
                id: point.id.clone(),
                vector: point.vector.clone(),
                payload: point.payload.clone(),
            });
        }

        let mut stats = self.stats.write().await;
        stats.upserts_succeeded += 1;
        stats.points_upserted += points.len() as u64;

        Ok(())
    }

    // -----------------------------------------------------------------------
    // Search
    // -----------------------------------------------------------------------

    /// Search for the nearest vectors in a collection.
    ///
    /// Returns up to `limit` results, sorted by decreasing similarity.
    ///
    /// # Arguments
    /// * `collection` — Base collection name (prefix is applied automatically).
    /// * `query_vector` — The query vector (must match collection dimensionality).
    /// * `limit` — Maximum number of results to return.
    /// * `score_threshold` — Optional minimum similarity score filter.
    pub async fn search(
        &self,
        collection: &str,
        query_vector: &[f64],
        limit: u64,
        score_threshold: Option<f64>,
    ) -> Result<Vec<ScoredPoint>> {
        let collection_name = self.config.collection_name(collection);
        let mut stats = self.stats.write().await;
        stats.searches_attempted += 1;
        drop(stats);

        let start = std::time::Instant::now();

        // Check circuit breaker
        {
            let mut cb = self.circuit_breaker.write().await;
            if !cb.should_allow() {
                if self.config.fallback_to_mock {
                    let mut stats = self.stats.write().await;
                    stats.mock_fallbacks += 1;
                    drop(stats);
                    return self
                        .mock_search(&collection_name, query_vector, limit, score_threshold)
                        .await;
                }
                return Err(JanusError::Memory(
                    "Qdrant circuit breaker is open — search rejected".into(),
                ));
            }
        }

        // Mock mode
        if *self.using_mock.read().await || self.client.read().await.is_none() {
            return self
                .mock_search(&collection_name, query_vector, limit, score_threshold)
                .await;
        }

        // Real Qdrant search with retry
        let result = self
            .with_retry(|| async {
                let client_guard = self.client.read().await;
                let client = client_guard
                    .as_ref()
                    .ok_or_else(|| JanusError::Memory("No Qdrant client".into()))?;

                let query_f32: Vec<f32> = query_vector.iter().map(|&v| v as f32).collect();

                let mut search_builder =
                    SearchPointsBuilder::new(&collection_name, query_f32, limit)
                        .with_payload(true)
                        .with_vectors(true);

                if let Some(threshold) = score_threshold {
                    search_builder = search_builder.score_threshold(threshold as f32);
                }

                let response = client
                    .search_points(search_builder)
                    .await
                    .map_err(|e| JanusError::Memory(format!("Qdrant search failed: {}", e)))?;

                let results: Vec<ScoredPoint> = response
                    .result
                    .iter()
                    .map(|sp| {
                        let id = match &sp.id {
                            Some(pid) => match &pid.point_id_options {
                                Some(PointIdOptions::Uuid(uuid)) => uuid.clone(),
                                Some(PointIdOptions::Num(n)) => n.to_string(),
                                None => String::new(),
                            },
                            None => String::new(),
                        };

                        let payload: HashMap<String, PayloadValue> = sp
                            .payload
                            .iter()
                            .map(|(k, v)| (k.clone(), PayloadValue::from_qdrant_value(v)))
                            .collect();

                        #[allow(deprecated)]
                        let vector: Option<Vec<f64>> = sp
                            .vectors
                            .as_ref()
                            .and_then(|v| v.get_vector())
                            .and_then(|v| {
                                use qdrant_client::qdrant::vector_output::Vector;
                                match v {
                                    Vector::Dense(dense) => {
                                        Some(dense.data.iter().map(|&f| f as f64).collect())
                                    }
                                    _ => None,
                                }
                            });

                        ScoredPoint {
                            id,
                            score: sp.score as f64,
                            vector,
                            payload,
                        }
                    })
                    .collect();

                Ok(results)
            })
            .await;

        let elapsed_ms = start.elapsed().as_secs_f64() * 1000.0;

        match &result {
            Ok(_) => {
                let mut cb = self.circuit_breaker.write().await;
                cb.record_success();
                drop(cb);

                let mut stats = self.stats.write().await;
                stats.searches_succeeded += 1;
                stats.cumulative_search_latency_ms += elapsed_ms;
            }
            Err(_) => {
                let mut cb = self.circuit_breaker.write().await;
                cb.record_failure();
                drop(cb);

                let mut stats = self.stats.write().await;
                stats.total_errors += 1;

                if self.config.fallback_to_mock {
                    stats.mock_fallbacks += 1;
                    drop(stats);
                    return self
                        .mock_search(&collection_name, query_vector, limit, score_threshold)
                        .await;
                }
            }
        }

        result
    }

    /// Search in mock storage using cosine similarity.
    async fn mock_search(
        &self,
        collection: &str,
        query_vector: &[f64],
        limit: u64,
        score_threshold: Option<f64>,
    ) -> Result<Vec<ScoredPoint>> {
        let storage = self.mock_storage.read().await;
        let points = match storage.get(collection) {
            Some(p) => p,
            None => return Ok(Vec::new()),
        };

        let mut scored: Vec<(f64, &MockPoint)> = points
            .iter()
            .map(|p| {
                let score = cosine_similarity(query_vector, &p.vector);
                (score, p)
            })
            .collect();

        // Filter by threshold
        if let Some(threshold) = score_threshold {
            scored.retain(|(score, _)| *score >= threshold);
        }

        // Sort descending by score
        scored.sort_by(|a, b| b.0.partial_cmp(&a.0).unwrap_or(std::cmp::Ordering::Equal));

        let results: Vec<ScoredPoint> = scored
            .into_iter()
            .take(limit as usize)
            .map(|(score, point)| ScoredPoint {
                id: point.id.clone(),
                score,
                vector: Some(point.vector.clone()),
                payload: point.payload.clone(),
            })
            .collect();

        let mut stats = self.stats.write().await;
        stats.searches_succeeded += 1;

        Ok(results)
    }

    // -----------------------------------------------------------------------
    // Delete
    // -----------------------------------------------------------------------

    /// Delete points by IDs from a collection.
    pub async fn delete(&self, collection: &str, ids: &[String]) -> Result<u64> {
        if ids.is_empty() {
            return Ok(0);
        }

        let collection_name = self.config.collection_name(collection);
        let mut stats = self.stats.write().await;
        stats.deletes_attempted += 1;
        drop(stats);

        // Mock mode
        if *self.using_mock.read().await || self.client.read().await.is_none() {
            return self.mock_delete(&collection_name, ids).await;
        }

        let client_guard = self.client.read().await;
        let client = client_guard
            .as_ref()
            .ok_or_else(|| JanusError::Memory("No Qdrant client".into()))?;

        let point_ids: Vec<PointId> = ids
            .iter()
            .map(|id| PointId {
                point_id_options: Some(PointIdOptions::Uuid(id.clone())),
            })
            .collect();

        let count = point_ids.len() as u64;

        client
            .delete_points(
                DeletePointsBuilder::new(&collection_name)
                    .points(point_ids)
                    .wait(true),
            )
            .await
            .map_err(|e| JanusError::Memory(format!("Qdrant delete failed: {}", e)))?;

        let mut stats = self.stats.write().await;
        stats.deletes_succeeded += 1;

        Ok(count)
    }

    /// Delete from mock storage.
    async fn mock_delete(&self, collection: &str, ids: &[String]) -> Result<u64> {
        let mut storage = self.mock_storage.write().await;
        let coll = match storage.get_mut(collection) {
            Some(c) => c,
            None => return Ok(0),
        };

        let before = coll.len();
        coll.retain(|p| !ids.contains(&p.id));
        let deleted = (before - coll.len()) as u64;

        let mut stats = self.stats.write().await;
        stats.deletes_succeeded += 1;

        Ok(deleted)
    }

    // -----------------------------------------------------------------------
    // Scroll (pagination)
    // -----------------------------------------------------------------------

    /// Scroll through all points in a collection, yielding pages of results.
    ///
    /// This is useful for batch processing or migration. Each call returns
    /// a page of points and an optional `next_offset` for the next page.
    pub async fn scroll(
        &self,
        collection: &str,
        limit: u64,
        offset: Option<String>,
    ) -> Result<(Vec<ScoredPoint>, Option<String>)> {
        let collection_name = self.config.collection_name(collection);

        // Real Qdrant scroll
        let client_guard = self.client.read().await;
        if let Some(ref client) = *client_guard
            && !*self.using_mock.read().await
        {
            let mut scroll_builder = ScrollPointsBuilder::new(&collection_name)
                .limit(limit as u32)
                .with_payload(true)
                .with_vectors(true);

            if let Some(ref offset_id) = offset {
                scroll_builder = scroll_builder.offset(PointId {
                    point_id_options: Some(PointIdOptions::Uuid(offset_id.clone())),
                });
            }

            let response = client
                .scroll(scroll_builder)
                .await
                .map_err(|e| JanusError::Memory(format!("Qdrant scroll failed: {}", e)))?;

            let points: Vec<ScoredPoint> = response
                .result
                .iter()
                .map(|rp| {
                    let id = match &rp.id {
                        Some(pid) => match &pid.point_id_options {
                            Some(PointIdOptions::Uuid(uuid)) => uuid.clone(),
                            Some(PointIdOptions::Num(n)) => n.to_string(),
                            None => String::new(),
                        },
                        None => String::new(),
                    };

                    let payload: HashMap<String, PayloadValue> = rp
                        .payload
                        .iter()
                        .map(|(k, v)| (k.clone(), PayloadValue::from_qdrant_value(v)))
                        .collect();

                    #[allow(deprecated)]
                    let vector: Option<Vec<f64>> = rp
                        .vectors
                        .as_ref()
                        .and_then(|v| v.get_vector())
                        .and_then(|v| {
                            use qdrant_client::qdrant::vector_output::Vector;
                            match v {
                                Vector::Dense(dense) => {
                                    Some(dense.data.iter().map(|&f| f as f64).collect())
                                }
                                _ => None,
                            }
                        });

                    ScoredPoint {
                        id,
                        score: 1.0,
                        vector,
                        payload,
                    }
                })
                .collect();

            let next_offset =
                response
                    .next_page_offset
                    .as_ref()
                    .and_then(|pid| match &pid.point_id_options {
                        Some(PointIdOptions::Uuid(uuid)) => Some(uuid.clone()),
                        Some(PointIdOptions::Num(n)) => Some(n.to_string()),
                        None => None,
                    });

            return Ok((points, next_offset));
        }

        // Mock implementation
        let storage = self.mock_storage.read().await;
        let points = match storage.get(&collection_name) {
            Some(p) => p,
            None => return Ok((Vec::new(), None)),
        };

        let start_idx = offset
            .as_ref()
            .and_then(|o| o.parse::<usize>().ok())
            .unwrap_or(0);

        let page: Vec<ScoredPoint> = points
            .iter()
            .skip(start_idx)
            .take(limit as usize)
            .map(|p| ScoredPoint {
                id: p.id.clone(),
                score: 1.0,
                vector: Some(p.vector.clone()),
                payload: p.payload.clone(),
            })
            .collect();

        let next_offset = if start_idx + page.len() < points.len() {
            Some((start_idx + page.len()).to_string())
        } else {
            None
        };

        Ok((page, next_offset))
    }

    // -----------------------------------------------------------------------
    // Get by ID
    // -----------------------------------------------------------------------

    /// Get specific points by their IDs.
    pub async fn get(&self, collection: &str, ids: &[String]) -> Result<Vec<ScoredPoint>> {
        let collection_name = self.config.collection_name(collection);

        // Real Qdrant get
        let client_guard = self.client.read().await;
        if let Some(ref client) = *client_guard
            && !*self.using_mock.read().await
        {
            let point_ids: Vec<PointId> = ids
                .iter()
                .map(|id| PointId {
                    point_id_options: Some(PointIdOptions::Uuid(id.clone())),
                })
                .collect();

            let response = client
                .get_points(
                    GetPointsBuilder::new(&collection_name, point_ids)
                        .with_payload(true)
                        .with_vectors(true),
                )
                .await
                .map_err(|e| JanusError::Memory(format!("Qdrant get failed: {}", e)))?;

            let results: Vec<ScoredPoint> = response
                .result
                .iter()
                .map(|rp| {
                    let id = match &rp.id {
                        Some(pid) => match &pid.point_id_options {
                            Some(PointIdOptions::Uuid(uuid)) => uuid.clone(),
                            Some(PointIdOptions::Num(n)) => n.to_string(),
                            None => String::new(),
                        },
                        None => String::new(),
                    };

                    let payload: HashMap<String, PayloadValue> = rp
                        .payload
                        .iter()
                        .map(|(k, v)| (k.clone(), PayloadValue::from_qdrant_value(v)))
                        .collect();

                    #[allow(deprecated)]
                    let vector: Option<Vec<f64>> = rp
                        .vectors
                        .as_ref()
                        .and_then(|v| v.get_vector())
                        .and_then(|v| {
                            use qdrant_client::qdrant::vector_output::Vector;
                            match v {
                                Vector::Dense(dense) => {
                                    Some(dense.data.iter().map(|&f| f as f64).collect())
                                }
                                _ => None,
                            }
                        });

                    ScoredPoint {
                        id,
                        score: 1.0,
                        vector,
                        payload,
                    }
                })
                .collect();

            return Ok(results);
        }

        // Mock implementation
        let storage = self.mock_storage.read().await;
        let points = match storage.get(&collection_name) {
            Some(p) => p,
            None => return Ok(Vec::new()),
        };

        let results: Vec<ScoredPoint> = points
            .iter()
            .filter(|p| ids.contains(&p.id))
            .map(|p| ScoredPoint {
                id: p.id.clone(),
                score: 1.0,
                vector: Some(p.vector.clone()),
                payload: p.payload.clone(),
            })
            .collect();

        Ok(results)
    }

    // -----------------------------------------------------------------------
    // Count
    // -----------------------------------------------------------------------

    /// Get the number of points in a collection.
    pub async fn count(&self, collection: &str) -> Result<u64> {
        let collection_name = self.config.collection_name(collection);

        // Real Qdrant count
        let client_guard = self.client.read().await;
        if let Some(ref client) = *client_guard
            && !*self.using_mock.read().await
        {
            let info = client
                .collection_info(&collection_name)
                .await
                .map_err(|e| JanusError::Memory(format!("Qdrant collection_info failed: {}", e)))?;

            let count = info
                .result
                .map(|r| r.points_count.unwrap_or(0))
                .unwrap_or(0);

            return Ok(count);
        }

        // Mock implementation
        let storage = self.mock_storage.read().await;
        let count = storage
            .get(&collection_name)
            .map(|p| p.len() as u64)
            .unwrap_or(0);
        Ok(count)
    }

    // -----------------------------------------------------------------------
    // MarketRegime convenience methods (backward-compatible with VectorDb)
    // -----------------------------------------------------------------------

    /// Store a market regime (convenience wrapper matching VectorDb API).
    pub async fn store_regime(&self, regime: &MarketRegime) -> Result<()> {
        let point = UpsertPoint::with_id(&regime.id, regime.features.clone())
            .with_string("regime_name", &regime.name)
            .with_float("volatility", regime.volatility)
            .with_float("trend", regime.trend);
        self.upsert(collections::MARKET_REGIMES, &[point]).await
    }

    /// Store multiple regimes in a batch.
    pub async fn store_regimes(&self, regimes: &[MarketRegime]) -> Result<usize> {
        let points: Vec<UpsertPoint> = regimes
            .iter()
            .map(|r| {
                UpsertPoint::with_id(&r.id, r.features.clone())
                    .with_string("regime_name", &r.name)
                    .with_float("volatility", r.volatility)
                    .with_float("trend", r.trend)
            })
            .collect();

        self.upsert(collections::MARKET_REGIMES, &points).await?;
        Ok(points.len())
    }

    /// Search for similar market regimes.
    pub async fn search_similar_regimes(
        &self,
        query_vector: &[f64],
        limit: u64,
    ) -> Result<Vec<MarketRegime>> {
        let results = self
            .search(collections::MARKET_REGIMES, query_vector, limit, None)
            .await?;

        let regimes: Vec<MarketRegime> = results
            .into_iter()
            .map(|sp| {
                let name = sp
                    .payload
                    .get("regime_name")
                    .and_then(|v| v.as_str())
                    .unwrap_or("unknown")
                    .to_string();
                let volatility = sp
                    .payload
                    .get("volatility")
                    .and_then(|v| v.as_f64())
                    .unwrap_or(0.0);
                let trend = sp
                    .payload
                    .get("trend")
                    .and_then(|v| v.as_f64())
                    .unwrap_or(0.0);
                let features = sp.vector.unwrap_or_default();

                MarketRegime {
                    id: sp.id,
                    name,
                    features,
                    volatility,
                    trend,
                }
            })
            .collect();

        Ok(regimes)
    }

    /// Delete a regime by ID.
    pub async fn delete_regime(&self, id: &str) -> Result<bool> {
        let deleted = self
            .delete(collections::MARKET_REGIMES, &[id.to_string()])
            .await?;
        Ok(deleted > 0)
    }

    // -----------------------------------------------------------------------
    // Retry logic
    // -----------------------------------------------------------------------

    /// Execute an async operation with exponential backoff retry.
    async fn with_retry<F, Fut, T>(&self, operation: F) -> Result<T>
    where
        F: Fn() -> Fut,
        Fut: std::future::Future<Output = Result<T>>,
    {
        let mut attempt = 0u32;
        let mut delay = self.config.retry_base_delay_ms;

        loop {
            match operation().await {
                Ok(result) => return Ok(result),
                Err(e) => {
                    attempt += 1;
                    if attempt > self.config.max_retries {
                        error!("Qdrant operation failed after {} attempts: {}", attempt, e);
                        return Err(e);
                    }

                    let mut stats = self.stats.write().await;
                    stats.total_retries += 1;
                    drop(stats);

                    // Compute delay with optional jitter
                    let actual_delay = if self.config.retry_jitter {
                        // Simple jitter: ±25% of the delay
                        let jitter_range = delay / 4;
                        let jitter = (std::time::SystemTime::now()
                            .duration_since(std::time::UNIX_EPOCH)
                            .unwrap_or_default()
                            .subsec_nanos()
                            % (2 * jitter_range as u32 + 1))
                            as u64;
                        delay.saturating_sub(jitter_range) + jitter
                    } else {
                        delay
                    };

                    warn!(
                        "Qdrant operation failed (attempt {}/{}), retrying in {}ms: {}",
                        attempt,
                        self.config.max_retries + 1,
                        actual_delay,
                        e
                    );

                    tokio::time::sleep(Duration::from_millis(actual_delay)).await;

                    // Exponential backoff (capped)
                    delay = (delay * 2).min(self.config.retry_max_delay_ms);
                }
            }
        }
    }

    // -----------------------------------------------------------------------
    // Health & Status
    // -----------------------------------------------------------------------

    /// Check if the client is connected to a real Qdrant instance.
    pub async fn is_connected(&self) -> bool {
        !*self.using_mock.read().await && self.client.read().await.is_some()
    }

    /// Check if the client is using mock storage.
    pub async fn is_using_mock(&self) -> bool {
        *self.using_mock.read().await
    }

    /// Get the current circuit breaker state.
    pub async fn circuit_state(&self) -> CircuitState {
        self.circuit_breaker.read().await.state()
    }

    /// Get operational statistics.
    pub async fn stats(&self) -> QdrantClientStats {
        self.stats.read().await.clone()
    }

    /// Get the configuration.
    pub fn config(&self) -> &QdrantProductionConfig {
        &self.config
    }

    /// Attempt to reconnect to Qdrant (useful after circuit breaker recovery).
    ///
    /// If a client already exists, verifies connectivity and resets the
    /// circuit breaker. If no client exists (e.g. initial connection failed
    /// and we fell back to mock), creates a brand-new client and hot-swaps
    /// it in behind the `Arc<RwLock<..>>`.
    pub async fn try_reconnect(&self) -> Result<bool> {
        // First, check if we already have a client that can reach Qdrant.
        {
            let client_guard = self.client.read().await;
            if let Some(ref client) = *client_guard {
                match client.list_collections().await {
                    Ok(_) => {
                        let mut cb = self.circuit_breaker.write().await;
                        cb.record_success();
                        drop(cb);

                        let mut mock = self.using_mock.write().await;
                        *mock = false;

                        info!("Reconnected to Qdrant successfully (existing client)");
                        return Ok(true);
                    }
                    Err(e) => {
                        warn!("Qdrant reconnect attempt failed on existing client: {}", e);
                        // Fall through to try creating a new client
                    }
                }
            }
        }

        // No working client — attempt to create a fresh one and hot-swap it in.
        match Self::create_qdrant_client(&self.config).await {
            Ok(new_client) => {
                let mut client_guard = self.client.write().await;
                *client_guard = Some(new_client);
                drop(client_guard);

                let mut cb = self.circuit_breaker.write().await;
                cb.record_success();
                drop(cb);

                let mut mock = self.using_mock.write().await;
                *mock = false;

                info!("Hot-swapped new Qdrant client — reconnection successful");
                Ok(true)
            }
            Err(e) => {
                warn!("Qdrant reconnect attempt failed (new client): {}", e);
                Ok(false)
            }
        }
    }

    /// Perform a health check against Qdrant.
    ///
    /// Returns Ok(true) if Qdrant is healthy, Ok(false) if unreachable,
    /// or Err if the check itself failed unexpectedly.
    pub async fn health_check(&self) -> Result<bool> {
        let client_guard = self.client.read().await;
        if let Some(ref client) = *client_guard {
            match client.list_collections().await {
                Ok(_) => {
                    let mut cb = self.circuit_breaker.write().await;
                    cb.record_success();
                    Ok(true)
                }
                Err(e) => {
                    let mut cb = self.circuit_breaker.write().await;
                    cb.record_failure();
                    debug!("Qdrant health check failed: {}", e);
                    Ok(false)
                }
            }
        } else {
            Ok(false)
        }
    }

    // -----------------------------------------------------------------------
    // Clear / Drop
    // -----------------------------------------------------------------------

    /// Clear all points from a collection (but keep the collection).
    pub async fn clear_collection(&self, collection: &str) -> Result<()> {
        let collection_name = self.config.collection_name(collection);

        // Real Qdrant clear: delete all points matching an empty filter
        let client_guard = self.client.read().await;
        if let Some(ref client) = *client_guard
            && !*self.using_mock.read().await
        {
            client
                .delete_points(
                    DeletePointsBuilder::new(&collection_name)
                        .points(Filter::default())
                        .wait(true),
                )
                .await
                .map_err(|e| {
                    JanusError::Memory(format!("Qdrant clear_collection failed: {}", e))
                })?;

            info!("Cleared all points from '{}'", collection_name);
            return Ok(());
        }

        // Mock implementation
        let mut storage = self.mock_storage.write().await;
        if let Some(coll) = storage.get_mut(&collection_name) {
            coll.clear();
        }
        Ok(())
    }

    /// Drop a collection entirely.
    pub async fn drop_collection(&self, collection: &str) -> Result<()> {
        let collection_name = self.config.collection_name(collection);

        // Real Qdrant drop
        let client_guard = self.client.read().await;
        if let Some(ref client) = *client_guard
            && !*self.using_mock.read().await
        {
            client
                .delete_collection(&collection_name)
                .await
                .map_err(|e| JanusError::Memory(format!("Qdrant drop_collection failed: {}", e)))?;

            info!("Dropped collection '{}'", collection_name);
            return Ok(());
        }

        // Mock implementation
        let mut storage = self.mock_storage.write().await;
        storage.remove(&collection_name);
        Ok(())
    }
}

// ---------------------------------------------------------------------------
// Utility Functions
// ---------------------------------------------------------------------------

/// Compute cosine similarity between two vectors.
fn cosine_similarity(a: &[f64], b: &[f64]) -> f64 {
    if a.len() != b.len() || a.is_empty() {
        return 0.0;
    }

    let dot: f64 = a.iter().zip(b.iter()).map(|(x, y)| x * y).sum();
    let norm_a: f64 = a.iter().map(|x| x * x).sum::<f64>().sqrt();
    let norm_b: f64 = b.iter().map(|x| x * x).sum::<f64>().sqrt();

    if norm_a > 0.0 && norm_b > 0.0 {
        dot / (norm_a * norm_b)
    } else {
        0.0
    }
}

// ---------------------------------------------------------------------------
// Tests
// ---------------------------------------------------------------------------

#[cfg(test)]
mod tests {
    use super::*;

    fn mock_client() -> QdrantProductionClient {
        QdrantProductionClient::mock()
    }

    fn test_regime(id: &str, features: Vec<f64>) -> MarketRegime {
        MarketRegime {
            id: id.to_string(),
            name: format!("Regime {}", id),
            features,
            volatility: 0.02,
            trend: 0.5,
        }
    }

    // -----------------------------------------------------------------------
    // Config tests
    // -----------------------------------------------------------------------

    #[test]
    fn test_default_config() {
        let config = QdrantProductionConfig::default();
        assert_eq!(config.url, "http://localhost:6334");
        assert!(config.api_key.is_none());
        assert!(config.fallback_to_mock);
        assert!(config.auto_create_collections);
        assert_eq!(config.default_distance, DistanceMetric::Cosine);
        assert!(config.collection_prefix.is_empty());
    }

    #[test]
    fn test_cloud_config() {
        let config = QdrantProductionConfig::cloud("https://cloud.qdrant.io:6334", "my_key");
        assert_eq!(config.url, "https://cloud.qdrant.io:6334");
        assert_eq!(config.api_key.as_deref(), Some("my_key"));
        assert!(!config.fallback_to_mock);
        assert_eq!(config.max_retries, 5);
    }

    #[test]
    fn test_collection_name_no_prefix() {
        let config = QdrantProductionConfig::default();
        assert_eq!(config.collection_name("market_regimes"), "market_regimes");
    }

    #[test]
    fn test_collection_name_with_prefix() {
        let config = QdrantProductionConfig {
            collection_prefix: "janus_prod_".to_string(),
            ..QdrantProductionConfig::default()
        };
        assert_eq!(
            config.collection_name("market_regimes"),
            "janus_prod_market_regimes"
        );
    }

    #[test]
    fn test_distance_metric_to_qdrant() {
        assert_eq!(DistanceMetric::Cosine.to_qdrant(), Distance::Cosine);
        assert_eq!(DistanceMetric::Euclid.to_qdrant(), Distance::Euclid);
        assert_eq!(DistanceMetric::Dot.to_qdrant(), Distance::Dot);
        assert_eq!(DistanceMetric::Manhattan.to_qdrant(), Distance::Manhattan);
    }

    // -----------------------------------------------------------------------
    // Collection specs
    // -----------------------------------------------------------------------

    #[test]
    fn test_default_collection_specs() {
        let specs = default_collection_specs();
        assert_eq!(specs.len(), 4);
        assert_eq!(specs[0].name, "market_regimes");
        assert_eq!(specs[0].vector_dim, 64);
        assert_eq!(specs[1].name, "episodic_memory");
        assert_eq!(specs[1].vector_dim, 128);
        assert_eq!(specs[2].name, "sentiment_embeddings");
        assert_eq!(specs[2].vector_dim, 768);
        assert_eq!(specs[3].name, "schema_prototypes");
    }

    #[test]
    fn test_collection_spec_with_payload_index() {
        let spec = CollectionSpec::new("test", 64, DistanceMetric::Cosine)
            .with_payload_index("field_a")
            .with_payload_index("field_b");
        assert_eq!(spec.payload_index_fields.len(), 2);
        assert_eq!(spec.payload_index_fields[0], "field_a");
    }

    // -----------------------------------------------------------------------
    // Circuit breaker
    // -----------------------------------------------------------------------

    #[test]
    fn test_circuit_breaker_starts_closed() {
        let cb = CircuitBreaker::new(3, 60);
        assert_eq!(cb.state(), CircuitState::Closed);
    }

    #[test]
    fn test_circuit_breaker_opens_after_threshold() {
        let mut cb = CircuitBreaker::new(3, 60);
        assert!(cb.should_allow());
        cb.record_failure();
        cb.record_failure();
        assert_eq!(cb.state(), CircuitState::Closed);
        cb.record_failure(); // Third failure = threshold
        assert_eq!(cb.state(), CircuitState::Open);
    }

    #[test]
    fn test_circuit_breaker_resets_on_success() {
        let mut cb = CircuitBreaker::new(3, 60);
        cb.record_failure();
        cb.record_failure();
        cb.record_success();
        assert_eq!(cb.state(), CircuitState::Closed);
        assert_eq!(cb.consecutive_failures, 0);
    }

    #[test]
    fn test_circuit_breaker_open_rejects() {
        let mut cb = CircuitBreaker::new(1, 3600); // Long reset to keep it open
        cb.record_failure(); // Opens immediately (threshold = 1)
        assert_eq!(cb.state(), CircuitState::Open);
        assert!(!cb.should_allow());
    }

    #[test]
    fn test_circuit_breaker_total_trips() {
        let mut cb = CircuitBreaker::new(1, 60);
        cb.record_failure();
        assert_eq!(cb.total_trips, 1);
        cb.record_success(); // Close it
        cb.record_failure(); // Open again
        assert_eq!(cb.total_trips, 2);
    }

    #[test]
    fn test_circuit_state_display() {
        assert_eq!(format!("{}", CircuitState::Closed), "Closed");
        assert_eq!(format!("{}", CircuitState::Open), "Open");
        assert_eq!(format!("{}", CircuitState::HalfOpen), "HalfOpen");
    }

    // -----------------------------------------------------------------------
    // Payload values
    // -----------------------------------------------------------------------

    #[test]
    fn test_payload_value_string() {
        let v = PayloadValue::String("hello".to_string());
        assert_eq!(v.as_str(), Some("hello"));
        assert!(v.as_f64().is_none());
    }

    #[test]
    fn test_payload_value_float() {
        let v = PayloadValue::Float(3.125);
        assert!((v.as_f64().unwrap() - 3.125).abs() < f64::EPSILON);
        assert!(v.as_str().is_none());
    }

    #[test]
    fn test_payload_value_integer() {
        let v = PayloadValue::Integer(42);
        assert_eq!(v.as_i64(), Some(42));
        assert_eq!(v.as_f64(), Some(42.0));
    }

    #[test]
    fn test_payload_value_bool() {
        let v = PayloadValue::Bool(true);
        assert_eq!(v.as_bool(), Some(true));
    }

    #[test]
    fn test_payload_value_null() {
        let v = PayloadValue::Null;
        assert!(v.as_str().is_none());
        assert!(v.as_f64().is_none());
        assert!(v.as_i64().is_none());
        assert!(v.as_bool().is_none());
    }

    #[test]
    fn test_payload_roundtrip() {
        let original = PayloadValue::String("test".to_string());
        let qdrant_val = original.to_qdrant_value();
        let recovered = PayloadValue::from_qdrant_value(&qdrant_val);
        assert_eq!(recovered.as_str(), Some("test"));
    }

    #[test]
    fn test_payload_roundtrip_float() {
        let original = PayloadValue::Float(2.625);
        let qdrant_val = original.to_qdrant_value();
        let recovered = PayloadValue::from_qdrant_value(&qdrant_val);
        assert!((recovered.as_f64().unwrap() - 2.625).abs() < f64::EPSILON);
    }

    #[test]
    fn test_payload_roundtrip_integer() {
        let original = PayloadValue::Integer(-99);
        let qdrant_val = original.to_qdrant_value();
        let recovered = PayloadValue::from_qdrant_value(&qdrant_val);
        assert_eq!(recovered.as_i64(), Some(-99));
    }

    #[test]
    fn test_payload_roundtrip_bool() {
        let original = PayloadValue::Bool(false);
        let qdrant_val = original.to_qdrant_value();
        let recovered = PayloadValue::from_qdrant_value(&qdrant_val);
        assert_eq!(recovered.as_bool(), Some(false));
    }

    // -----------------------------------------------------------------------
    // UpsertPoint
    // -----------------------------------------------------------------------

    #[test]
    fn test_upsert_point_new() {
        let point = UpsertPoint::new(vec![1.0, 2.0, 3.0]);
        assert!(!point.id.is_empty());
        assert_eq!(point.vector.len(), 3);
        assert!(point.payload.is_empty());
    }

    #[test]
    fn test_upsert_point_with_id() {
        let point = UpsertPoint::with_id("my_id", vec![1.0]);
        assert_eq!(point.id, "my_id");
    }

    #[test]
    fn test_upsert_point_builder() {
        let point = UpsertPoint::with_id("p1", vec![0.1, 0.2])
            .with_string("name", "test")
            .with_float("score", 0.95)
            .with_integer("count", 7)
            .with_bool("active", true);

        assert_eq!(point.payload.len(), 4);
        assert_eq!(
            point.payload.get("name").and_then(|v| v.as_str()),
            Some("test")
        );
        assert_eq!(
            point.payload.get("score").and_then(|v| v.as_f64()),
            Some(0.95)
        );
        assert_eq!(point.payload.get("count").and_then(|v| v.as_i64()), Some(7));
        assert_eq!(
            point.payload.get("active").and_then(|v| v.as_bool()),
            Some(true)
        );
    }

    #[test]
    fn test_upsert_point_to_qdrant() {
        let point = UpsertPoint::with_id("p1", vec![0.5, 0.5]).with_string("key", "val");
        let qdrant_point = point.to_qdrant_point();
        assert!(!qdrant_point.payload.is_empty());
    }

    // -----------------------------------------------------------------------
    // Statistics
    // -----------------------------------------------------------------------

    #[test]
    fn test_stats_defaults() {
        let stats = QdrantClientStats::default();
        assert_eq!(stats.upserts_attempted, 0);
        assert!((stats.avg_search_latency_ms()).abs() < f64::EPSILON);
        assert!((stats.avg_upsert_latency_ms()).abs() < f64::EPSILON);
        assert!((stats.upsert_success_rate() - 1.0).abs() < f64::EPSILON);
        assert!((stats.search_success_rate() - 1.0).abs() < f64::EPSILON);
        assert!((stats.error_rate()).abs() < f64::EPSILON);
    }

    #[test]
    fn test_stats_search_latency() {
        let stats = QdrantClientStats {
            searches_succeeded: 10,
            cumulative_search_latency_ms: 100.0,
            ..QdrantClientStats::default()
        };
        assert!((stats.avg_search_latency_ms() - 10.0).abs() < f64::EPSILON);
    }

    #[test]
    fn test_stats_success_rates() {
        let stats = QdrantClientStats {
            upserts_attempted: 10,
            upserts_succeeded: 8,
            ..QdrantClientStats::default()
        };
        assert!((stats.upsert_success_rate() - 0.8).abs() < f64::EPSILON);
    }

    #[test]
    fn test_stats_error_rate() {
        let stats = QdrantClientStats {
            upserts_attempted: 10,
            searches_attempted: 10,
            total_errors: 4,
            ..QdrantClientStats::default()
        };
        assert!((stats.error_rate() - 0.2).abs() < f64::EPSILON);
    }

    // -----------------------------------------------------------------------
    // Mock client operations
    // -----------------------------------------------------------------------

    #[tokio::test]
    async fn test_mock_creation() {
        let client = mock_client();
        assert!(client.is_using_mock().await);
        assert!(!client.is_connected().await);
    }

    #[tokio::test]
    async fn test_mock_ensure_collection() {
        let client = mock_client();
        let spec = CollectionSpec::new("test_coll", 64, DistanceMetric::Cosine);
        let result = client.ensure_collection(&spec).await;
        assert!(result.is_ok());
    }

    #[tokio::test]
    async fn test_mock_upsert_and_search() {
        let client = mock_client();

        // Ensure collection exists
        let spec = CollectionSpec::new(collections::MARKET_REGIMES, 3, DistanceMetric::Cosine);
        client.ensure_collection(&spec).await.unwrap();

        // Upsert some points
        let points = vec![
            UpsertPoint::with_id("p1", vec![1.0, 0.0, 0.0]).with_string("name", "point_1"),
            UpsertPoint::with_id("p2", vec![0.9, 0.1, 0.0]).with_string("name", "point_2"),
            UpsertPoint::with_id("p3", vec![0.0, 1.0, 0.0]).with_string("name", "point_3"),
        ];
        client
            .upsert(collections::MARKET_REGIMES, &points)
            .await
            .unwrap();

        // Search
        let results = client
            .search(collections::MARKET_REGIMES, &[1.0, 0.0, 0.0], 2, None)
            .await
            .unwrap();

        assert_eq!(results.len(), 2);
        assert_eq!(results[0].id, "p1"); // Exact match
        assert!(results[0].score > 0.99);
        assert_eq!(results[1].id, "p2"); // Close match
    }

    #[tokio::test]
    async fn test_mock_search_with_threshold() {
        let client = mock_client();
        let spec = CollectionSpec::new("test", 2, DistanceMetric::Cosine);
        client.ensure_collection(&spec).await.unwrap();

        let points = vec![
            UpsertPoint::with_id("p1", vec![1.0, 0.0]),
            UpsertPoint::with_id("p2", vec![0.0, 1.0]), // Orthogonal — cosine sim = 0.0
        ];
        client.upsert("test", &points).await.unwrap();

        // With high threshold, orthogonal point should be filtered out
        let results = client
            .search("test", &[1.0, 0.0], 10, Some(0.5))
            .await
            .unwrap();
        assert_eq!(results.len(), 1);
        assert_eq!(results[0].id, "p1");
    }

    #[tokio::test]
    async fn test_mock_upsert_overwrites() {
        let client = mock_client();
        let spec = CollectionSpec::new("test", 2, DistanceMetric::Cosine);
        client.ensure_collection(&spec).await.unwrap();

        // Insert
        let point1 = vec![UpsertPoint::with_id("p1", vec![1.0, 0.0]).with_string("v", "first")];
        client.upsert("test", &point1).await.unwrap();

        // Overwrite
        let point2 = vec![UpsertPoint::with_id("p1", vec![0.0, 1.0]).with_string("v", "second")];
        client.upsert("test", &point2).await.unwrap();

        // Should have only 1 point
        let count = client.count("test").await.unwrap();
        assert_eq!(count, 1);

        // And it should have the new payload
        let results = client.get("test", &["p1".to_string()]).await.unwrap();
        assert_eq!(results.len(), 1);
        assert_eq!(
            results[0].payload.get("v").and_then(|v| v.as_str()),
            Some("second")
        );
    }

    #[tokio::test]
    async fn test_mock_delete() {
        let client = mock_client();
        let spec = CollectionSpec::new("test", 2, DistanceMetric::Cosine);
        client.ensure_collection(&spec).await.unwrap();

        let points = vec![
            UpsertPoint::with_id("p1", vec![1.0, 0.0]),
            UpsertPoint::with_id("p2", vec![0.0, 1.0]),
        ];
        client.upsert("test", &points).await.unwrap();
        assert_eq!(client.count("test").await.unwrap(), 2);

        let deleted = client.delete("test", &["p1".to_string()]).await.unwrap();
        assert_eq!(deleted, 1);
        assert_eq!(client.count("test").await.unwrap(), 1);
    }

    #[tokio::test]
    async fn test_mock_delete_nonexistent() {
        let client = mock_client();
        let deleted = client
            .delete("nonexistent", &["xxx".to_string()])
            .await
            .unwrap();
        assert_eq!(deleted, 0);
    }

    #[tokio::test]
    async fn test_mock_get() {
        let client = mock_client();
        let spec = CollectionSpec::new("test", 2, DistanceMetric::Cosine);
        client.ensure_collection(&spec).await.unwrap();

        let points = vec![
            UpsertPoint::with_id("p1", vec![1.0, 0.0]).with_string("name", "one"),
            UpsertPoint::with_id("p2", vec![0.0, 1.0]).with_string("name", "two"),
        ];
        client.upsert("test", &points).await.unwrap();

        let results = client.get("test", &["p2".to_string()]).await.unwrap();
        assert_eq!(results.len(), 1);
        assert_eq!(results[0].id, "p2");
        assert_eq!(
            results[0].payload.get("name").and_then(|v| v.as_str()),
            Some("two")
        );
    }

    #[tokio::test]
    async fn test_mock_get_nonexistent() {
        let client = mock_client();
        let results = client
            .get("nonexistent", &["xxx".to_string()])
            .await
            .unwrap();
        assert!(results.is_empty());
    }

    #[tokio::test]
    async fn test_mock_count() {
        let client = mock_client();
        assert_eq!(client.count("test").await.unwrap(), 0);

        let spec = CollectionSpec::new("test", 2, DistanceMetric::Cosine);
        client.ensure_collection(&spec).await.unwrap();

        let points: Vec<UpsertPoint> = (0..5)
            .map(|i| UpsertPoint::with_id(&format!("p{}", i), vec![i as f64, 0.0]))
            .collect();
        client.upsert("test", &points).await.unwrap();

        assert_eq!(client.count("test").await.unwrap(), 5);
    }

    #[tokio::test]
    async fn test_mock_scroll() {
        let client = mock_client();
        let spec = CollectionSpec::new("test", 2, DistanceMetric::Cosine);
        client.ensure_collection(&spec).await.unwrap();

        let points: Vec<UpsertPoint> = (0..10)
            .map(|i| UpsertPoint::with_id(&format!("p{}", i), vec![i as f64, 0.0]))
            .collect();
        client.upsert("test", &points).await.unwrap();

        // First page
        let (page1, next) = client.scroll("test", 3, None).await.unwrap();
        assert_eq!(page1.len(), 3);
        assert!(next.is_some());

        // Second page
        let (page2, next2) = client.scroll("test", 3, next).await.unwrap();
        assert_eq!(page2.len(), 3);
        assert!(next2.is_some());

        // Last page (remaining 4)
        let (page3, next3) = client.scroll("test", 10, next2).await.unwrap();
        assert_eq!(page3.len(), 4);
        assert!(next3.is_none());
    }

    #[tokio::test]
    async fn test_mock_clear_collection() {
        let client = mock_client();
        let spec = CollectionSpec::new("test", 2, DistanceMetric::Cosine);
        client.ensure_collection(&spec).await.unwrap();

        let points = vec![UpsertPoint::with_id("p1", vec![1.0, 0.0])];
        client.upsert("test", &points).await.unwrap();
        assert_eq!(client.count("test").await.unwrap(), 1);

        client.clear_collection("test").await.unwrap();
        assert_eq!(client.count("test").await.unwrap(), 0);
    }

    #[tokio::test]
    async fn test_mock_drop_collection() {
        let client = mock_client();
        let spec = CollectionSpec::new("test", 2, DistanceMetric::Cosine);
        client.ensure_collection(&spec).await.unwrap();

        let points = vec![UpsertPoint::with_id("p1", vec![1.0, 0.0])];
        client.upsert("test", &points).await.unwrap();

        client.drop_collection("test").await.unwrap();
        assert_eq!(client.count("test").await.unwrap(), 0);
    }

    #[tokio::test]
    async fn test_upsert_empty() {
        let client = mock_client();
        let result = client.upsert("test", &[]).await;
        assert!(result.is_ok());
    }

    #[tokio::test]
    async fn test_delete_empty() {
        let client = mock_client();
        let result = client.delete("test", &[]).await;
        assert!(result.is_ok());
        assert_eq!(result.unwrap(), 0);
    }

    #[tokio::test]
    async fn test_search_empty_collection() {
        let client = mock_client();
        let results = client
            .search("nonexistent", &[1.0, 0.0], 10, None)
            .await
            .unwrap();
        assert!(results.is_empty());
    }

    // -----------------------------------------------------------------------
    // MarketRegime convenience methods
    // -----------------------------------------------------------------------

    #[tokio::test]
    async fn test_store_and_search_regime() {
        let client = mock_client();
        let spec = CollectionSpec::new(collections::MARKET_REGIMES, 3, DistanceMetric::Cosine);
        client.ensure_collection(&spec).await.unwrap();

        let regime = test_regime("r1", vec![1.0, 0.0, 0.0]);
        client.store_regime(&regime).await.unwrap();

        let results = client
            .search_similar_regimes(&[0.9, 0.1, 0.0], 5)
            .await
            .unwrap();
        assert_eq!(results.len(), 1);
        assert_eq!(results[0].id, "r1");
        assert_eq!(results[0].name, "Regime r1");
    }

    #[tokio::test]
    async fn test_store_regimes_batch() {
        let client = mock_client();
        let spec = CollectionSpec::new(collections::MARKET_REGIMES, 3, DistanceMetric::Cosine);
        client.ensure_collection(&spec).await.unwrap();

        let regimes = vec![
            test_regime("r1", vec![1.0, 0.0, 0.0]),
            test_regime("r2", vec![0.0, 1.0, 0.0]),
            test_regime("r3", vec![0.0, 0.0, 1.0]),
        ];

        let count = client.store_regimes(&regimes).await.unwrap();
        assert_eq!(count, 3);
        assert_eq!(client.count(collections::MARKET_REGIMES).await.unwrap(), 3);
    }

    #[tokio::test]
    async fn test_delete_regime() {
        let client = mock_client();
        let spec = CollectionSpec::new(collections::MARKET_REGIMES, 3, DistanceMetric::Cosine);
        client.ensure_collection(&spec).await.unwrap();

        let regime = test_regime("r1", vec![1.0, 0.0, 0.0]);
        client.store_regime(&regime).await.unwrap();

        let deleted = client.delete_regime("r1").await.unwrap();
        assert!(deleted);

        let deleted_again = client.delete_regime("r1").await.unwrap();
        assert!(!deleted_again);
    }

    // -----------------------------------------------------------------------
    // Stats tracking
    // -----------------------------------------------------------------------

    #[tokio::test]
    async fn test_stats_tracked_on_operations() {
        let client = mock_client();
        let spec = CollectionSpec::new("test", 2, DistanceMetric::Cosine);
        client.ensure_collection(&spec).await.unwrap();

        let points = vec![
            UpsertPoint::with_id("p1", vec![1.0, 0.0]),
            UpsertPoint::with_id("p2", vec![0.0, 1.0]),
        ];
        client.upsert("test", &points).await.unwrap();

        client.search("test", &[1.0, 0.0], 5, None).await.unwrap();

        client.delete("test", &["p1".to_string()]).await.unwrap();

        let stats = client.stats().await;
        assert_eq!(stats.upserts_succeeded, 1);
        assert_eq!(stats.points_upserted, 2);
        assert_eq!(stats.searches_succeeded, 1);
        assert_eq!(stats.deletes_succeeded, 1);
        assert_eq!(stats.total_errors, 0);
    }

    #[tokio::test]
    async fn test_circuit_state_accessible() {
        let client = mock_client();
        assert_eq!(client.circuit_state().await, CircuitState::Closed);
    }

    #[tokio::test]
    async fn test_health_check_mock() {
        let client = mock_client();
        let healthy = client.health_check().await.unwrap();
        assert!(!healthy); // Mock mode returns false
    }

    // -----------------------------------------------------------------------
    // Cosine similarity
    // -----------------------------------------------------------------------

    #[test]
    fn test_cosine_similarity_identical() {
        let sim = cosine_similarity(&[1.0, 0.0, 0.0], &[1.0, 0.0, 0.0]);
        assert!((sim - 1.0).abs() < 0.001);
    }

    #[test]
    fn test_cosine_similarity_orthogonal() {
        let sim = cosine_similarity(&[1.0, 0.0, 0.0], &[0.0, 1.0, 0.0]);
        assert!(sim.abs() < 0.001);
    }

    #[test]
    fn test_cosine_similarity_opposite() {
        let sim = cosine_similarity(&[1.0, 0.0, 0.0], &[-1.0, 0.0, 0.0]);
        assert!((sim - (-1.0)).abs() < 0.001);
    }

    #[test]
    fn test_cosine_similarity_different_lengths() {
        let sim = cosine_similarity(&[1.0, 0.0], &[1.0, 0.0, 0.0]);
        assert_eq!(sim, 0.0);
    }

    #[test]
    fn test_cosine_similarity_empty() {
        let sim = cosine_similarity(&[], &[]);
        assert_eq!(sim, 0.0);
    }

    #[test]
    fn test_cosine_similarity_zero_vector() {
        let sim = cosine_similarity(&[0.0, 0.0], &[1.0, 0.0]);
        assert_eq!(sim, 0.0);
    }
}
