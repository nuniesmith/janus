//! # Experience Vector Store
//!
//! Persists validated experience data from Arrow IPC batches into Qdrant for
//! similarity-based experience retrieval during training.
//!
//! ## Architecture
//!
//! ```text
//! Arrow RecordBatch (validated)
//!       │
//!       ▼
//! ┌─────────────────────┐
//! │  ExperienceStore    │
//! │  - extract vectors  │
//! │  - build payloads   │
//! │  - upsert to Qdrant│
//! └─────────────────────┘
//!       │
//!       ▼
//! ┌─────────────────────┐
//! │  Qdrant Collection  │
//! │  "experiences"      │
//! └─────────────────────┘
//! ```
//!
//! Each experience row is stored as a Qdrant point where:
//! - **Vector**: The `state_gaf` binary field decoded as `f32` little-endian values
//! - **Payload**: Metadata fields (action, reward, symbol, timestamp, done flag)
//! - **ID**: A deterministic UUID v5 of `(batch_id, row_index)` so that
//!   at-least-once delivery (Redis queue + spool sweep) re-upserts the same
//!   point instead of duplicating it (design §5)

use anyhow::{Context, Result};
use arrow::array::{
    Array, BinaryArray, BooleanArray, Float32Array, Int64Array, StringArray, UInt8Array,
};
use arrow::record_batch::RecordBatch;
use qdrant_client::Qdrant;
use qdrant_client::qdrant::{
    CreateCollectionBuilder, Distance, PointStruct, ScalarQuantizationBuilder, UpsertPointsBuilder,
    VectorParamsBuilder,
};
use serde::{Deserialize, Serialize};
use std::collections::HashMap;
use std::sync::Arc;
use tokio::sync::RwLock;
use tracing::{debug, error, info, warn};

// ─── Configuration ────────────────────────────────────────────────────────────

/// Configuration for the experience vector store.
#[derive(Debug, Clone)]
pub struct ExperienceStoreConfig {
    /// Qdrant gRPC endpoint URL (e.g. `http://localhost:6334`).
    pub qdrant_url: String,

    /// Name of the Qdrant collection for experience vectors.
    pub collection_name: String,

    /// Dimensionality of the GAF state vector.
    ///
    /// This must match the number of `f32` values encoded in the `state_gaf`
    /// binary column of the Arrow IPC file.  The default (9) corresponds to
    /// a 3×3 GAF image flattened to a vector.
    pub vector_dim: u64,

    /// Maximum number of points to upsert in a single Qdrant request.
    pub upsert_batch_size: usize,

    /// Whether to use mock mode (in-memory, no real Qdrant connection).
    pub use_mock: bool,

    /// Connection timeout in seconds.
    pub timeout_secs: u64,

    /// Skip the qdrant-client's client/server version-compatibility check.
    ///
    /// The check `println!`s its warning **to stdout** (qdrant-client 1.17,
    /// `qdrant_client/mod.rs`), which corrupts any consumer that treats
    /// stdout as machine-readable output — the `gate_a_verdict` binary sets
    /// this so its JSON report stays parseable. Default `false`: the
    /// long-running service keeps the check (its stdout is logs anyway).
    pub skip_compatibility_check: bool,
}

impl Default for ExperienceStoreConfig {
    fn default() -> Self {
        Self {
            qdrant_url: "http://localhost:6334".to_string(),
            collection_name: "experiences".to_string(),
            vector_dim: 9,
            upsert_batch_size: 256,
            use_mock: true,
            timeout_secs: 10,
            skip_compatibility_check: false,
        }
    }
}

impl ExperienceStoreConfig {
    /// Load configuration from environment variables, falling back to defaults.
    ///
    /// Note: `QDRANT_USE_MOCK` defaults to **false** — mock mode must be
    /// requested explicitly (`QDRANT_USE_MOCK=true`). A misconfigured or
    /// unreachable Qdrant is a hard error at store construction, never a
    /// silent fallback to in-memory storage (design §5).
    pub fn from_env() -> Self {
        Self {
            qdrant_url: std::env::var("QDRANT_URL")
                .unwrap_or_else(|_| "http://localhost:6334".to_string()),
            collection_name: std::env::var("QDRANT_EXPERIENCE_COLLECTION")
                .unwrap_or_else(|_| "experiences".to_string()),
            vector_dim: std::env::var("QDRANT_EXPERIENCE_DIM")
                .ok()
                .and_then(|v| v.parse().ok())
                .unwrap_or(9),
            upsert_batch_size: std::env::var("QDRANT_UPSERT_BATCH_SIZE")
                .ok()
                .and_then(|v| v.parse().ok())
                .unwrap_or(256),
            use_mock: std::env::var("QDRANT_USE_MOCK")
                .unwrap_or_else(|_| "false".to_string())
                .parse()
                .unwrap_or(false),
            timeout_secs: std::env::var("QDRANT_TIMEOUT_SECS")
                .ok()
                .and_then(|v| v.parse().ok())
                .unwrap_or(10),
            skip_compatibility_check: false,
        }
    }
}

// ─── Metrics ──────────────────────────────────────────────────────────────────

/// Counters for experience persistence operations.
#[derive(Debug, Clone, Default)]
pub struct PersistenceMetrics {
    /// Total number of points successfully upserted.
    pub points_upserted: usize,
    /// Total number of rows that could not be converted to points.
    pub rows_failed: usize,
    /// Number of upsert batches sent to Qdrant.
    pub upsert_calls: usize,
    /// Number of upsert calls that failed.
    pub upsert_errors: usize,
}

// ─── Extracted experience row ─────────────────────────────────────────────────

/// A single experience extracted from an Arrow record batch, ready for
/// Qdrant upsert.
/// Per-row side-channel parsed from the producer's `rows_meta` file metadata
/// (EXPERIENCE_PIPELINE.md §3): fields that don't fit the frozen Arrow schema
/// but belong in the Qdrant payload for the UMAP view. All optional — absent
/// for older producers, and absence must never fail an ingest.
#[derive(Debug, Clone, Default, Serialize, serde::Deserialize)]
pub struct RowSideMeta {
    #[serde(default)]
    pub interval: Option<String>,
    #[serde(default)]
    pub confidence: Option<f64>,
    #[serde(default)]
    pub blocked: Option<String>,
    #[serde(default)]
    pub regime: Option<String>,
}

#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct ExperienceRow {
    /// The GAF state vector (f32 values decoded from binary).
    pub state_vector: Vec<f32>,
    /// Raw supplementary state features (optional).
    pub state_raw: Option<Vec<f32>>,
    /// Action type index (0=Buy, 1=Sell, 2=Hold, 3=Close).
    pub action_type: u8,
    /// Trading symbol.
    pub action_symbol: String,
    /// Order quantity.
    pub action_qty: f32,
    /// Scalar reward.
    pub reward: f32,
    /// Next-state GAF vector.
    pub next_state_vector: Vec<f32>,
    /// Next-state raw features (optional).
    pub next_state_raw: Option<Vec<f32>>,
    /// Episode termination flag.
    pub done: bool,
    /// Unix epoch milliseconds.
    pub timestamp_ms: i64,
    /// Side-channel payload fields (rows_meta), when the producer sent them.
    pub meta: Option<RowSideMeta>,
}

// ─── Mock in-memory store ─────────────────────────────────────────────────────

/// In-memory mock storage used during testing or when Qdrant is unavailable.
#[derive(Debug, Default)]
struct MockStore {
    points: Vec<(String, Vec<f32>, HashMap<String, serde_json::Value>)>,
}

impl MockStore {
    /// Upsert points by ID: an existing point with the same ID is replaced
    /// (mirrors Qdrant upsert semantics so idempotency is testable).
    fn upsert(&mut self, points: &[(String, ExperienceRow)]) -> usize {
        let mut count = 0;
        for (id, row) in points {
            let mut payload: HashMap<String, serde_json::Value> = HashMap::new();
            payload.insert("action_type".into(), serde_json::json!(row.action_type));
            payload.insert("action_symbol".into(), serde_json::json!(row.action_symbol));
            payload.insert("action_qty".into(), serde_json::json!(row.action_qty));
            payload.insert("reward".into(), serde_json::json!(row.reward));
            payload.insert("done".into(), serde_json::json!(row.done));
            payload.insert("timestamp_ms".into(), serde_json::json!(row.timestamp_ms));

            let entry = (id.clone(), row.state_vector.clone(), payload);
            if let Some(existing) = self.points.iter_mut().find(|(pid, _, _)| pid == id) {
                *existing = entry;
            } else {
                self.points.push(entry);
            }
            count += 1;
        }
        count
    }

    fn len(&self) -> usize {
        self.points.len()
    }

    fn clear(&mut self) {
        self.points.clear();
    }
}

// ─── ExperienceStore ──────────────────────────────────────────────────────────

/// Client for persisting experience vectors to Qdrant.
///
/// Supports both a real Qdrant backend and an in-memory mock for testing /
/// environments where Qdrant is unavailable.
pub struct ExperienceStore {
    config: ExperienceStoreConfig,
    client: Option<Qdrant>,
    mock: Arc<RwLock<MockStore>>,
    is_mock: bool,
    metrics: Arc<RwLock<PersistenceMetrics>>,
}

impl ExperienceStore {
    /// Create a new experience store.
    ///
    /// If `config.use_mock` is `true`, no real Qdrant connection is attempted.
    /// Otherwise the constructor connects to Qdrant and verifies it is
    /// reachable via a health check. A failed connection is an **error** —
    /// there is no silent fallback to mock mode (design §5); mock mode is
    /// only available via explicit `use_mock=true` / `QDRANT_USE_MOCK=true`.
    /// Callers that need startup resilience should retry with backoff (see
    /// [`Self::connect_with_retry`]).
    pub async fn new(config: ExperienceStoreConfig) -> Result<Self> {
        if config.use_mock {
            info!("ExperienceStore running in mock mode (no Qdrant connection)");
            return Ok(Self {
                config,
                client: None,
                mock: Arc::new(RwLock::new(MockStore::default())),
                is_mock: true,
                metrics: Arc::new(RwLock::new(PersistenceMetrics::default())),
            });
        }

        let mut builder = Qdrant::from_url(&config.qdrant_url)
            .timeout(std::time::Duration::from_secs(config.timeout_secs));
        if config.skip_compatibility_check {
            // The version check println!s to STDOUT on mismatch/unreachable —
            // callers with machine-readable stdout (gate_a_verdict) opt out.
            builder = builder.skip_compatibility_check();
        }
        let client = builder
            .build()
            .with_context(|| {
                format!(
                    "Failed to build Qdrant client for '{}' (set QDRANT_USE_MOCK=true only for explicit mock mode)",
                    config.qdrant_url
                )
            })?;

        client.health_check().await.with_context(|| {
            format!(
                "Qdrant health check failed at '{}' — refusing to fall back to mock mode",
                config.qdrant_url
            )
        })?;

        info!(url = %config.qdrant_url, "Connected to Qdrant for experience persistence");
        Ok(Self {
            config,
            client: Some(client),
            mock: Arc::new(RwLock::new(MockStore::default())),
            is_mock: false,
            metrics: Arc::new(RwLock::new(PersistenceMetrics::default())),
        })
    }

    /// Create a store, retrying with exponential backoff while Qdrant is
    /// unreachable (real mode only — mock mode never fails).
    ///
    /// Returns `None` if `cancel` fires before a connection is established.
    pub async fn connect_with_retry(
        config: ExperienceStoreConfig,
        cancel: &tokio_util::sync::CancellationToken,
    ) -> Option<Self> {
        let mut backoff = std::time::Duration::from_secs(5);
        const MAX_BACKOFF: std::time::Duration = std::time::Duration::from_secs(300);

        loop {
            match Self::new(config.clone()).await {
                Ok(store) => return Some(store),
                Err(e) => {
                    error!(
                        error = %e,
                        retry_in_secs = backoff.as_secs(),
                        "ExperienceStore connection failed — retrying"
                    );
                }
            }

            tokio::select! {
                _ = cancel.cancelled() => return None,
                _ = tokio::time::sleep(backoff) => {}
            }
            backoff = (backoff * 2).min(MAX_BACKOFF);
        }
    }

    /// Create a mock-only experience store (convenience for tests).
    pub fn mock() -> Self {
        Self {
            config: ExperienceStoreConfig {
                use_mock: true,
                ..Default::default()
            },
            client: None,
            mock: Arc::new(RwLock::new(MockStore::default())),
            is_mock: true,
            metrics: Arc::new(RwLock::new(PersistenceMetrics::default())),
        }
    }

    /// Whether this store is operating in mock mode.
    pub fn is_mock(&self) -> bool {
        self.is_mock
    }

    /// Returns a snapshot of the current persistence metrics.
    pub async fn metrics(&self) -> PersistenceMetrics {
        self.metrics.read().await.clone()
    }

    /// Number of points currently stored (mock mode only — returns 0 for
    /// real Qdrant since we'd need a collection info call).
    pub async fn mock_point_count(&self) -> usize {
        self.mock.read().await.len()
    }

    /// Clear all mock storage.
    pub async fn mock_clear(&self) {
        self.mock.write().await.clear();
    }

    // ── Collection management ─────────────────────────────────────────────

    /// Ensure the Qdrant collection exists, creating it if necessary.
    pub async fn ensure_collection(&self) -> Result<()> {
        if self.is_mock {
            debug!("Mock mode — collection always exists");
            return Ok(());
        }

        let client = self
            .client
            .as_ref()
            .context("Qdrant client not initialised")?;

        let exists = client
            .collection_exists(&self.config.collection_name)
            .await
            .context("Failed to check if collection exists")?;

        if exists {
            debug!(
                collection = %self.config.collection_name,
                "Experience collection already exists"
            );
            self.ensure_payload_indexes(client).await;
            return Ok(());
        }

        info!(
            collection = %self.config.collection_name,
            dim = self.config.vector_dim,
            "Creating experience collection in Qdrant"
        );

        client
            .create_collection(
                CreateCollectionBuilder::new(&self.config.collection_name)
                    .vectors_config(VectorParamsBuilder::new(
                        self.config.vector_dim,
                        Distance::Cosine,
                    ))
                    .quantization_config(ScalarQuantizationBuilder::default()),
            )
            .await
            .context("Failed to create Qdrant collection")?;

        info!(
            collection = %self.config.collection_name,
            "Experience collection created"
        );

        self.ensure_payload_indexes(client).await;
        Ok(())
    }

    /// Payload indexes for filtered/ordered sampling (design §5 Phase 1.5).
    /// Best-effort and idempotent: Qdrant treats re-creation as a no-op, and
    /// an index failure must never block ingestion — sampling just degrades.
    async fn ensure_payload_indexes(&self, client: &Qdrant) {
        use qdrant_client::qdrant::{CreateFieldIndexCollectionBuilder, FieldType};
        for (field, ftype) in [
            ("action_symbol", FieldType::Keyword),
            ("action_type", FieldType::Integer),
            ("timestamp_ms", FieldType::Integer),
        ] {
            if let Err(e) = client
                .create_field_index(CreateFieldIndexCollectionBuilder::new(
                    &self.config.collection_name,
                    field,
                    ftype,
                ))
                .await
            {
                debug!(field, "payload index create skipped/failed: {e}");
            }
        }
    }

    // ── Sampling (UMAP view / read API — EXPERIENCE_PIPELINE.md §8) ───────

    /// Sample recent experience points (newest-first by `timestamp_ms`) with
    /// optional payload filters. Returns `(points, total_in_collection)`.
    /// Mock mode returns an honest empty result.
    pub async fn sample(
        &self,
        limit: usize,
        symbol: Option<&str>,
        action: Option<u8>,
        since_ms: Option<i64>,
    ) -> Result<(Vec<SamplePoint>, u64)> {
        if self.is_mock {
            return Ok((Vec::new(), 0));
        }
        let client = self
            .client
            .as_ref()
            .context("Qdrant client not initialised")?;

        use qdrant_client::qdrant::{
            Condition, CountPointsBuilder, Direction, Filter, OrderBy, Range, ScrollPointsBuilder,
            r#match::MatchValue,
        };

        let mut conditions: Vec<Condition> = Vec::new();
        if let Some(sym) = symbol {
            conditions.push(Condition::matches(
                "action_symbol",
                MatchValue::Keyword(sym.to_string()),
            ));
        }
        if let Some(a) = action {
            conditions.push(Condition::matches("action_type", a as i64));
        }
        if let Some(since) = since_ms {
            conditions.push(Condition::range(
                "timestamp_ms",
                Range {
                    gte: Some(since as f64),
                    ..Default::default()
                },
            ));
        }

        let mut scroll = ScrollPointsBuilder::new(&self.config.collection_name)
            .limit(limit as u32)
            .with_payload(true)
            .with_vectors(true)
            .order_by(OrderBy {
                key: "timestamp_ms".to_string(),
                direction: Some(Direction::Desc.into()),
                ..Default::default()
            });
        if !conditions.is_empty() {
            scroll = scroll.filter(Filter::must(conditions));
        }

        let scrolled = client
            .scroll(scroll)
            .await
            .context("Qdrant scroll failed")?;

        let total = client
            .count(CountPointsBuilder::new(&self.config.collection_name).exact(false))
            .await
            .map(|r| r.result.map(|c| c.count).unwrap_or(0))
            .unwrap_or(0);

        let points = scrolled
            .result
            .into_iter()
            .filter_map(|p| {
                let id = p.id.as_ref().map(point_id_string)??;
                let vector = extract_vector(p.vectors.as_ref())?;
                let payload = p
                    .payload
                    .into_iter()
                    .map(|(k, v)| (k, qdrant_value_to_json(v)))
                    .collect();
                Some(SamplePoint {
                    id,
                    vector,
                    payload,
                })
            })
            .collect();

        Ok((points, total))
    }

    /// Approximate total number of points in the collection (mock mode
    /// reports the mock store's size).
    pub async fn collection_count(&self) -> Result<u64> {
        if self.is_mock {
            return Ok(self.mock.read().await.len() as u64);
        }
        let client = self
            .client
            .as_ref()
            .context("Qdrant client not initialised")?;
        use qdrant_client::qdrant::CountPointsBuilder;
        let count = client
            .count(CountPointsBuilder::new(&self.config.collection_name).exact(false))
            .await
            .context("Qdrant count failed")?
            .result
            .map(|c| c.count)
            .unwrap_or(0);
        Ok(count)
    }

    /// Scroll EVERY point whose `timestamp_ms` payload falls inside
    /// `[start_ms, end_ms]`, paginating with Qdrant's `next_page_offset`
    /// cursor until the window is exhausted (or `max_points` is reached as a
    /// runaway guard).
    ///
    /// This is the full-window read path for the Gate-A verdict script: the
    /// HTTP sample endpoint caps at 5000 points, and [`Self::sample`]'s
    /// single `order_by` scroll cannot paginate (Qdrant does not return a
    /// page cursor for ordered scrolls) — a 2-week window at ~10 rows/min is
    /// ~200k points, so this scrolls unordered pages and lets the caller sort.
    /// Vectors are not fetched (payload-only; `SamplePoint::vector` is empty)
    /// since window statistics only need the payload.
    ///
    /// Mock mode returns an honest empty result, mirroring [`Self::sample`].
    pub async fn scroll_window(
        &self,
        start_ms: i64,
        end_ms: i64,
        page_size: usize,
        max_points: usize,
    ) -> Result<Vec<SamplePoint>> {
        if self.is_mock {
            return Ok(Vec::new());
        }
        let client = self
            .client
            .as_ref()
            .context("Qdrant client not initialised")?;

        use qdrant_client::qdrant::{Condition, Filter, PointId, Range, ScrollPointsBuilder};

        let filter = Filter::must([Condition::range(
            "timestamp_ms",
            Range {
                gte: Some(start_ms as f64),
                lte: Some(end_ms as f64),
                ..Default::default()
            },
        )]);

        let mut out: Vec<SamplePoint> = Vec::new();
        let mut offset: Option<PointId> = None;
        loop {
            let mut scroll = ScrollPointsBuilder::new(&self.config.collection_name)
                .limit(page_size.clamp(1, 10_000) as u32)
                .with_payload(true)
                .with_vectors(false)
                .filter(filter.clone());
            if let Some(off) = offset.take() {
                scroll = scroll.offset(off);
            }
            let scrolled = client
                .scroll(scroll)
                .await
                .context("Qdrant window scroll failed")?;

            let page_len = scrolled.result.len();
            for p in scrolled.result {
                let Some(id) = p.id.as_ref().and_then(point_id_string) else {
                    continue;
                };
                let payload = p
                    .payload
                    .into_iter()
                    .map(|(k, v)| (k, qdrant_value_to_json(v)))
                    .collect();
                out.push(SamplePoint {
                    id,
                    vector: Vec::new(),
                    payload,
                });
            }

            offset = scrolled.next_page_offset;
            if offset.is_none() || page_len == 0 || out.len() >= max_points {
                break;
            }
        }
        Ok(out)
    }

    // ── Training read (EXPERIENCE_PIPELINE.md §7 Phase 2) ────────────────

    /// Sample live-collected experiences shaped for the DQN replay buffer.
    ///
    /// Reuses the newest-first [`Self::sample`] scroll path (with vectors +
    /// payload) and maps each point into a [`TrainingSample`] — exactly the
    /// `(state_vector, action, reward, next_state_vector, done)` tuple the
    /// replay buffer needs. Points that predate the Phase 1.5 payload contract
    /// (missing `reward`/`action_type`/`done`/`next_state_vector`) are skipped
    /// rather than silently defaulted, so a partial upgrade never injects
    /// zero-reward garbage into training.
    ///
    /// Mock mode returns an empty vec (the underlying `sample` is honest about
    /// having no Qdrant to scroll), so this never fabricates training data.
    pub async fn sample_for_training(
        &self,
        limit: usize,
        since_ms: Option<i64>,
    ) -> Result<Vec<TrainingSample>> {
        let (points, _total) = self.sample(limit, None, None, since_ms).await?;
        Ok(points
            .iter()
            .filter_map(training_sample_from_point)
            .collect())
    }

    // ── Batch persistence ─────────────────────────────────────────────────

    /// Persist a validated Arrow record batch to the vector store.
    ///
    /// `batch_id` and `row_offset` (the number of Arrow rows in preceding
    /// record batches of the same IPC file) derive the deterministic UUIDv5
    /// point IDs, so re-ingesting the same file upserts the same points.
    ///
    /// Hard-fails (returns `Err`) when:
    /// - any row's `state_gaf` vector length ≠ the configured collection dim
    ///   (the whole batch is rejected so the caller can park the file, rather
    ///   than letting Qdrant reject rows one by one — design §5), or
    /// - any upsert chunk fails (e.g. Qdrant down), so the caller can retry.
    ///
    /// Returns the number of rows successfully persisted.
    pub async fn persist_batch(
        &self,
        batch: &RecordBatch,
        batch_id: &str,
        row_offset: usize,
    ) -> Result<usize> {
        self.persist_batch_with_meta(batch, batch_id, row_offset, None)
            .await
    }

    /// [`Self::persist_batch`] with the producer's per-row side-channel
    /// (`rows_meta` file metadata, keyed by absolute row index within the
    /// file) threaded into the Qdrant payload.
    pub async fn persist_batch_with_meta(
        &self,
        batch: &RecordBatch,
        batch_id: &str,
        row_offset: usize,
        rows_meta: Option<&HashMap<usize, RowSideMeta>>,
    ) -> Result<usize> {
        let rows = self.extract_rows(batch)?;

        if rows.is_empty() {
            debug!("No valid rows extracted from record batch — nothing to persist");
            return Ok(0);
        }

        // ── Dimension hard-check (design §5) ──────────────────────────────
        let expected_dim = self.config.vector_dim as usize;
        if let Some((row_idx, row)) = rows
            .iter()
            .find(|(_, r)| r.state_vector.len() != expected_dim)
        {
            anyhow::bail!(
                "Batch '{}' has state_gaf vector of length {} at row {} but the \
                 configured collection dim is {} — rejecting the whole batch",
                batch_id,
                row.state_vector.len(),
                row_idx,
                expected_dim
            );
        }

        let total = rows.len();
        let points: Vec<(String, ExperienceRow)> = rows
            .into_iter()
            .map(|(row_idx, mut row)| {
                row.meta = rows_meta
                    .and_then(|m| m.get(&(row_offset + row_idx)))
                    .cloned();
                (experience_point_id(batch_id, row_offset + row_idx), row)
            })
            .collect();

        let mut persisted = 0;
        let mut chunk_errors = 0usize;
        let mut last_error = String::new();

        // Chunk into upsert batches
        for chunk in points.chunks(self.config.upsert_batch_size) {
            match self.upsert_chunk(chunk).await {
                Ok(n) => {
                    persisted += n;
                    let mut m = self.metrics.write().await;
                    m.points_upserted += n;
                    m.upsert_calls += 1;
                }
                Err(e) => {
                    error!(error = %e, chunk_size = chunk.len(), "Failed to upsert experience chunk");
                    last_error = format!("{e:#}");
                    chunk_errors += 1;
                    let mut m = self.metrics.write().await;
                    m.upsert_errors += 1;
                    m.rows_failed += chunk.len();
                }
            }
        }

        if chunk_errors > 0 {
            anyhow::bail!(
                "Batch '{}': {} of {} upsert chunk(s) failed (persisted {}/{} rows); last error: {}",
                batch_id,
                chunk_errors,
                points.len().div_ceil(self.config.upsert_batch_size),
                persisted,
                total,
                last_error
            );
        }

        debug!(persisted, total, "Persisted experience batch");
        Ok(persisted)
    }

    /// Persist pre-extracted experience rows directly (best-effort).
    ///
    /// Point IDs are random UUIDv4 — this path is **not** idempotent and is
    /// intended for seeding / testing, not for the at-least-once ingest path
    /// (use [`Self::persist_batch`] there).
    pub async fn persist_rows(&self, rows: &[ExperienceRow]) -> Result<usize> {
        if rows.is_empty() {
            return Ok(0);
        }

        let points: Vec<(String, ExperienceRow)> = rows
            .iter()
            .map(|row| (uuid::Uuid::new_v4().to_string(), row.clone()))
            .collect();

        let mut persisted = 0;
        for chunk in points.chunks(self.config.upsert_batch_size) {
            match self.upsert_chunk(chunk).await {
                Ok(n) => {
                    persisted += n;
                    let mut m = self.metrics.write().await;
                    m.points_upserted += n;
                    m.upsert_calls += 1;
                }
                Err(e) => {
                    error!(error = %e, "Failed to upsert experience rows");
                    let mut m = self.metrics.write().await;
                    m.upsert_errors += 1;
                    m.rows_failed += chunk.len();
                }
            }
        }
        Ok(persisted)
    }

    // ── Internal helpers ──────────────────────────────────────────────────

    /// Upsert a chunk of `(point_id, row)` pairs to Qdrant (or mock).
    async fn upsert_chunk(&self, points: &[(String, ExperienceRow)]) -> Result<usize> {
        if self.is_mock {
            let mut store = self.mock.write().await;
            let n = store.upsert(points);
            return Ok(n);
        }

        let client = self
            .client
            .as_ref()
            .context("Qdrant client not initialised")?;

        let points: Vec<PointStruct> = points
            .iter()
            .map(|(id, row)| self.row_to_point(id.clone(), row))
            .collect();
        let count = points.len();

        client
            .upsert_points(
                UpsertPointsBuilder::new(&self.config.collection_name, points).wait(true),
            )
            .await
            .context("Qdrant upsert failed")?;

        Ok(count)
    }

    /// Convert a single `ExperienceRow` into a Qdrant `PointStruct` with the
    /// given (deterministic) point ID.
    fn row_to_point(&self, id: String, row: &ExperienceRow) -> PointStruct {
        use qdrant_client::qdrant::Value;
        use qdrant_client::qdrant::value::Kind;

        let mut payload: HashMap<String, Value> = HashMap::new();

        payload.insert(
            "action_type".into(),
            Value {
                kind: Some(Kind::IntegerValue(row.action_type as i64)),
            },
        );
        payload.insert(
            "action_symbol".into(),
            Value {
                kind: Some(Kind::StringValue(row.action_symbol.clone())),
            },
        );
        payload.insert(
            "action_qty".into(),
            Value {
                kind: Some(Kind::DoubleValue(row.action_qty as f64)),
            },
        );
        payload.insert(
            "reward".into(),
            Value {
                kind: Some(Kind::DoubleValue(row.reward as f64)),
            },
        );
        payload.insert(
            "done".into(),
            Value {
                kind: Some(Kind::BoolValue(row.done)),
            },
        );
        payload.insert(
            "timestamp_ms".into(),
            Value {
                kind: Some(Kind::IntegerValue(row.timestamp_ms)),
            },
        );

        // Side-channel payload (UMAP view coloring — design §5 Phase 1.5).
        // Absent fields are simply omitted; readers get null.
        if let Some(meta) = &row.meta {
            if let Some(c) = meta.confidence {
                payload.insert(
                    "confidence".into(),
                    Value {
                        kind: Some(Kind::DoubleValue(c)),
                    },
                );
            }
            payload.insert(
                "blocked".into(),
                Value {
                    kind: Some(Kind::BoolValue(meta.blocked.is_some())),
                },
            );
            if let Some(b) = &meta.blocked {
                payload.insert(
                    "blocked_reason".into(),
                    Value {
                        kind: Some(Kind::StringValue(b.clone())),
                    },
                );
            }
            if let Some(iv) = &meta.interval {
                payload.insert(
                    "interval".into(),
                    Value {
                        kind: Some(Kind::StringValue(iv.clone())),
                    },
                );
            }
            if let Some(rg) = &meta.regime {
                payload.insert(
                    "regime".into(),
                    Value {
                        kind: Some(Kind::StringValue(rg.clone())),
                    },
                );
            }
        }

        // Encode next_state_vector as a JSON string in the payload so it can
        // be recovered during training without a separate lookup.
        let next_state_json =
            serde_json::to_string(&row.next_state_vector).unwrap_or_else(|_| "[]".to_string());
        payload.insert(
            "next_state_vector".into(),
            Value {
                kind: Some(Kind::StringValue(next_state_json)),
            },
        );

        PointStruct::new(id, row.state_vector.clone(), payload)
    }

    /// Extract `ExperienceRow`s from a validated Arrow record batch, paired
    /// with their original Arrow row index (used for deterministic point IDs).
    ///
    /// Rows that cannot be fully decoded (missing required columns, bad binary
    /// length, etc.) are skipped with a warning and counted in the metrics.
    pub fn extract_rows(&self, batch: &RecordBatch) -> Result<Vec<(usize, ExperienceRow)>> {
        let num_rows = batch.num_rows();
        if num_rows == 0 {
            return Ok(Vec::new());
        }

        let schema = batch.schema();

        // ── Resolve column indices ────────────────────────────────────────
        let col_state_gaf = schema
            .index_of("state_gaf")
            .context("Missing required column: state_gaf")?;
        let col_state_raw = schema.index_of("state_raw").ok();
        let col_action_type = schema
            .index_of("action_type")
            .context("Missing required column: action_type")?;
        let col_action_symbol = schema
            .index_of("action_symbol")
            .context("Missing required column: action_symbol")?;
        let col_action_qty = schema
            .index_of("action_qty")
            .context("Missing required column: action_qty")?;
        let col_reward = schema
            .index_of("reward")
            .context("Missing required column: reward")?;
        let col_next_state_gaf = schema
            .index_of("next_state_gaf")
            .context("Missing required column: next_state_gaf")?;
        let col_next_state_raw = schema.index_of("next_state_raw").ok();
        let col_done = schema
            .index_of("done")
            .context("Missing required column: done")?;
        let col_timestamp = schema
            .index_of("timestamp_ms")
            .context("Missing required column: timestamp_ms")?;

        // ── Downcast columns ──────────────────────────────────────────────
        let state_gaf_arr = batch
            .column(col_state_gaf)
            .as_any()
            .downcast_ref::<BinaryArray>()
            .context("state_gaf column is not Binary")?;

        let action_type_arr = batch
            .column(col_action_type)
            .as_any()
            .downcast_ref::<UInt8Array>()
            .context("action_type column is not UInt8")?;

        let action_symbol_arr = batch
            .column(col_action_symbol)
            .as_any()
            .downcast_ref::<StringArray>()
            .context("action_symbol column is not Utf8")?;

        let action_qty_arr = batch
            .column(col_action_qty)
            .as_any()
            .downcast_ref::<Float32Array>()
            .context("action_qty column is not Float32")?;

        let reward_arr = batch
            .column(col_reward)
            .as_any()
            .downcast_ref::<Float32Array>()
            .context("reward column is not Float32")?;

        let next_state_gaf_arr = batch
            .column(col_next_state_gaf)
            .as_any()
            .downcast_ref::<BinaryArray>()
            .context("next_state_gaf column is not Binary")?;

        let done_arr = batch
            .column(col_done)
            .as_any()
            .downcast_ref::<BooleanArray>()
            .context("done column is not Boolean")?;

        let timestamp_arr = batch
            .column(col_timestamp)
            .as_any()
            .downcast_ref::<Int64Array>()
            .context("timestamp_ms column is not Int64")?;

        // Optional columns
        let state_raw_arr =
            col_state_raw.and_then(|idx| batch.column(idx).as_any().downcast_ref::<BinaryArray>());

        let next_state_raw_arr = col_next_state_raw
            .and_then(|idx| batch.column(idx).as_any().downcast_ref::<BinaryArray>());

        // ── Extract rows ──────────────────────────────────────────────────
        let mut rows = Vec::with_capacity(num_rows);
        let mut failed = 0usize;

        for i in 0..num_rows {
            // Skip rows with null required fields
            if state_gaf_arr.is_null(i)
                || action_type_arr.is_null(i)
                || action_symbol_arr.is_null(i)
                || action_qty_arr.is_null(i)
                || reward_arr.is_null(i)
                || next_state_gaf_arr.is_null(i)
                || done_arr.is_null(i)
                || timestamp_arr.is_null(i)
            {
                debug!(row = i, "Skipping row with null required field");
                failed += 1;
                continue;
            }

            let state_vector = match decode_f32_le(state_gaf_arr.value(i)) {
                Some(v) => v,
                None => {
                    debug!(row = i, "Failed to decode state_gaf binary to f32 vector");
                    failed += 1;
                    continue;
                }
            };

            let next_state_vector = match decode_f32_le(next_state_gaf_arr.value(i)) {
                Some(v) => v,
                None => {
                    debug!(
                        row = i,
                        "Failed to decode next_state_gaf binary to f32 vector"
                    );
                    failed += 1;
                    continue;
                }
            };

            let state_raw = state_raw_arr.and_then(|arr| {
                if arr.is_null(i) {
                    None
                } else {
                    decode_f32_le(arr.value(i))
                }
            });

            let next_state_raw = next_state_raw_arr.and_then(|arr| {
                if arr.is_null(i) {
                    None
                } else {
                    decode_f32_le(arr.value(i))
                }
            });

            rows.push((
                i,
                ExperienceRow {
                    state_vector,
                    state_raw,
                    action_type: action_type_arr.value(i),
                    action_symbol: action_symbol_arr.value(i).to_string(),
                    action_qty: action_qty_arr.value(i),
                    reward: reward_arr.value(i),
                    next_state_vector,
                    next_state_raw,
                    done: done_arr.value(i),
                    timestamp_ms: timestamp_arr.value(i),
                    meta: None,
                },
            ));
        }

        if failed > 0 {
            let mut m_guard = self.metrics.try_write().unwrap_or_else(|_| {
                // If the lock is poisoned (shouldn't happen with tokio RwLock),
                // just skip updating metrics — not worth panicking.
                unreachable!("tokio RwLock cannot be poisoned")
            });
            m_guard.rows_failed += failed;
            warn!(
                failed,
                extracted = rows.len(),
                "Some rows failed extraction"
            );
        }

        Ok(rows)
    }
}

// ─── Utility functions ────────────────────────────────────────────────────────

/// One sampled experience point for the read API.
#[derive(Debug, Serialize)]
pub struct SamplePoint {
    pub id: String,
    pub vector: Vec<f32>,
    pub payload: serde_json::Map<String, serde_json::Value>,
}

/// A replay-ready experience recovered from a Qdrant point, containing exactly
/// the fields the DQN replay buffer needs (EXPERIENCE_PIPELINE.md §7 Phase 2).
///
/// `action` is the canonical `0=Buy, 1=Sell, 2=Hold, 3=Close` index (design
/// §3.2). `state_vector` / `next_state_vector` are the flat GAF vectors at `t`
/// and `t+1`; `reward` is the signed next-bar return (design §3.3); `done` is
/// the episode-termination flag (design §3.4).
#[derive(Debug, Clone, PartialEq)]
pub struct TrainingSample {
    pub state_vector: Vec<f32>,
    pub action: u8,
    pub reward: f32,
    pub next_state_vector: Vec<f32>,
    pub done: bool,
}

/// Map a sampled Qdrant point to a [`TrainingSample`].
///
/// Returns `None` when a required field is missing or malformed — the point's
/// vector is empty, `action_type`/`reward`/`done` are absent, or the
/// `next_state_vector` payload (a JSON-encoded `Vec<f32>` string written by
/// [`ExperienceStore::row_to_point`]) cannot be parsed. This keeps pre-Phase-2
/// points out of the replay buffer instead of defaulting them.
pub fn training_sample_from_point(point: &SamplePoint) -> Option<TrainingSample> {
    if point.vector.is_empty() {
        return None;
    }
    let payload = &point.payload;
    let action = payload.get("action_type")?.as_i64()?;
    if !(0..=3).contains(&action) {
        return None;
    }
    let reward = payload.get("reward")?.as_f64()? as f32;
    let done = payload.get("done")?.as_bool()?;
    let next_state_vector: Vec<f32> = payload
        .get("next_state_vector")
        .and_then(|v| v.as_str())
        .and_then(|s| serde_json::from_str(s).ok())?;
    if next_state_vector.is_empty() {
        return None;
    }
    Some(TrainingSample {
        state_vector: point.vector.clone(),
        action: action as u8,
        reward,
        next_state_vector,
        done,
    })
}

fn point_id_string(id: &qdrant_client::qdrant::PointId) -> Option<String> {
    use qdrant_client::qdrant::point_id::PointIdOptions;
    match id.point_id_options.as_ref()? {
        PointIdOptions::Uuid(u) => Some(u.clone()),
        PointIdOptions::Num(n) => Some(n.to_string()),
    }
}

fn extract_vector(vectors: Option<&qdrant_client::qdrant::VectorsOutput>) -> Option<Vec<f32>> {
    use qdrant_client::qdrant::vectors_output::VectorsOptions;
    match vectors?.vectors_options.as_ref()? {
        VectorsOptions::Vector(v) => Some(v.data.clone()),
        VectorsOptions::Vectors(_) => None, // named vectors are not used here
    }
}

/// Minimal qdrant→JSON conversion for the payload kinds this store writes.
fn qdrant_value_to_json(v: qdrant_client::qdrant::Value) -> serde_json::Value {
    use qdrant_client::qdrant::value::Kind;
    match v.kind {
        Some(Kind::IntegerValue(i)) => serde_json::json!(i),
        Some(Kind::DoubleValue(d)) => serde_json::json!(d),
        Some(Kind::StringValue(s)) => serde_json::json!(s),
        Some(Kind::BoolValue(b)) => serde_json::json!(b),
        _ => serde_json::Value::Null,
    }
}

/// Deterministic Qdrant point ID: UUIDv5 of `(batch_id, row_index)` under a
/// janus-experience namespace. At-least-once delivery (queue entry + sweep)
/// therefore re-upserts the same point instead of duplicating it (design §5).
pub fn experience_point_id(batch_id: &str, row_index: usize) -> String {
    let namespace = uuid::Uuid::new_v5(&uuid::Uuid::NAMESPACE_OID, b"janus:experience");
    uuid::Uuid::new_v5(&namespace, format!("{batch_id}:{row_index}").as_bytes()).to_string()
}

/// Decode a byte slice as a sequence of little-endian `f32` values.
///
/// Returns `None` if the byte length is not a multiple of 4 (i.e. not
/// aligned to `f32` boundaries).
fn decode_f32_le(bytes: &[u8]) -> Option<Vec<f32>> {
    if !bytes.len().is_multiple_of(4) {
        return None;
    }
    let values: Vec<f32> = bytes
        .chunks_exact(4)
        .map(|chunk| f32::from_le_bytes([chunk[0], chunk[1], chunk[2], chunk[3]]))
        .collect();
    Some(values)
}

/// Encode a slice of `f32` values as little-endian bytes.
#[cfg(test)]
fn encode_f32_le(values: &[f32]) -> Vec<u8> {
    values.iter().flat_map(|v| v.to_le_bytes()).collect()
}

// ─── Tests ────────────────────────────────────────────────────────────────────

#[cfg(test)]
mod tests {
    use super::*;
    use arrow::array::{
        BinaryArray, BooleanArray, Float32Array, Int64Array, StringArray, UInt8Array,
    };
    use arrow::datatypes::{DataType, Field, Schema};
    use arrow::record_batch::RecordBatch;
    use std::sync::Arc as StdArc;

    /// Build the expected experience schema.
    fn experience_schema() -> Schema {
        Schema::new(vec![
            Field::new("state_gaf", DataType::Binary, false),
            Field::new("state_raw", DataType::Binary, true),
            Field::new("action_type", DataType::UInt8, false),
            Field::new("action_symbol", DataType::Utf8, false),
            Field::new("action_qty", DataType::Float32, false),
            Field::new("reward", DataType::Float32, false),
            Field::new("next_state_gaf", DataType::Binary, false),
            Field::new("next_state_raw", DataType::Binary, true),
            Field::new("done", DataType::Boolean, false),
            Field::new("timestamp_ms", DataType::Int64, false),
        ])
    }

    /// Create a test record batch with the given number of rows.
    fn create_test_batch(num_rows: usize) -> RecordBatch {
        let schema = StdArc::new(experience_schema());

        // 9 floats × 4 bytes = 36 bytes per GAF vector
        let gaf_bytes: Vec<u8> = encode_f32_le(&[1.0, 2.0, 3.0, 4.0, 5.0, 6.0, 7.0, 8.0, 9.0]);
        let raw_bytes: Vec<u8> = encode_f32_le(&[0.1, 0.2, 0.3]);

        let state_gaf: Vec<&[u8]> = (0..num_rows).map(|_| gaf_bytes.as_slice()).collect();
        let state_raw: Vec<Option<&[u8]>> =
            (0..num_rows).map(|_| Some(raw_bytes.as_slice())).collect();
        let next_state_gaf: Vec<&[u8]> = (0..num_rows).map(|_| gaf_bytes.as_slice()).collect();
        let next_state_raw: Vec<Option<&[u8]>> =
            (0..num_rows).map(|_| Some(raw_bytes.as_slice())).collect();

        let action_types: Vec<u8> = (0..num_rows).map(|i| (i % 4) as u8).collect();
        let symbols: Vec<&str> = (0..num_rows).map(|_| "BTCUSD").collect();
        let quantities: Vec<f32> = (0..num_rows).map(|i| (i + 1) as f32 * 0.1).collect();
        let rewards: Vec<f32> = (0..num_rows).map(|i| (i as f32) * 0.01 - 0.05).collect();
        let dones: Vec<bool> = (0..num_rows).map(|i| i == num_rows - 1).collect();
        let timestamps: Vec<i64> = (0..num_rows)
            .map(|i| 1_700_000_000_000_i64 + (i as i64) * 1000)
            .collect();

        RecordBatch::try_new(
            schema,
            vec![
                StdArc::new(BinaryArray::from_vec(state_gaf)),
                StdArc::new(BinaryArray::from(state_raw)),
                StdArc::new(UInt8Array::from(action_types)),
                StdArc::new(StringArray::from(symbols)),
                StdArc::new(Float32Array::from(quantities)),
                StdArc::new(Float32Array::from(rewards)),
                StdArc::new(BinaryArray::from_vec(next_state_gaf)),
                StdArc::new(BinaryArray::from(next_state_raw)),
                StdArc::new(BooleanArray::from(dones)),
                StdArc::new(Int64Array::from(timestamps)),
            ],
        )
        .expect("Failed to create test RecordBatch")
    }

    // ── Unit tests ────────────────────────────────────────────────────────

    #[test]
    fn test_decode_f32_le_round_trip() {
        let values = vec![1.0_f32, 2.5, -std::f32::consts::PI, 0.0];
        let encoded = encode_f32_le(&values);
        let decoded = decode_f32_le(&encoded).unwrap();
        assert_eq!(values.len(), decoded.len());
        for (a, b) in values.iter().zip(decoded.iter()) {
            assert!((a - b).abs() < 1e-6);
        }
    }

    #[test]
    fn test_decode_f32_le_bad_length() {
        assert!(decode_f32_le(&[0u8; 5]).is_none());
        assert!(decode_f32_le(&[0u8; 3]).is_none());
        assert!(decode_f32_le(&[0u8; 1]).is_none());
    }

    #[test]
    fn test_decode_f32_le_empty() {
        let decoded = decode_f32_le(&[]).unwrap();
        assert!(decoded.is_empty());
    }

    #[test]
    fn test_default_config() {
        let config = ExperienceStoreConfig::default();
        assert_eq!(config.collection_name, "experiences");
        assert_eq!(config.vector_dim, 9);
        assert!(config.use_mock);
    }

    #[test]
    fn test_extract_rows_from_batch() {
        let store = ExperienceStore::mock();
        let batch = create_test_batch(5);
        let rows = store.extract_rows(&batch).unwrap();

        assert_eq!(rows.len(), 5);

        // Check first row (rows are paired with their Arrow row index)
        let (idx, first) = &rows[0];
        assert_eq!(*idx, 0);
        assert_eq!(first.state_vector.len(), 9);
        assert!((first.state_vector[0] - 1.0).abs() < 1e-6);
        assert_eq!(first.action_type, 0);
        assert_eq!(first.action_symbol, "BTCUSD");
        assert!(!first.done);
        assert_eq!(first.timestamp_ms, 1_700_000_000_000);

        // Last row should be terminal, with its original index
        assert_eq!(rows[4].0, 4);
        assert!(rows[4].1.done);
    }

    #[test]
    fn test_extract_rows_empty_batch() {
        let store = ExperienceStore::mock();
        let batch = create_test_batch(0);
        let rows = store.extract_rows(&batch).unwrap();
        assert!(rows.is_empty());
    }

    #[tokio::test]
    async fn test_mock_persist_batch() {
        let store = ExperienceStore::mock();
        let batch = create_test_batch(10);

        let persisted = store.persist_batch(&batch, "batch-a", 0).await.unwrap();
        assert_eq!(persisted, 10);

        assert_eq!(store.mock_point_count().await, 10);

        let m = store.metrics().await;
        assert_eq!(m.points_upserted, 10);
        assert_eq!(m.upsert_calls, 1);
        assert_eq!(m.upsert_errors, 0);
    }

    #[tokio::test]
    async fn test_persist_batch_is_idempotent() {
        let store = ExperienceStore::mock();
        let batch = create_test_batch(10);

        store.persist_batch(&batch, "batch-a", 0).await.unwrap();
        // Re-ingesting the same batch (sweep + queue race) must not duplicate
        store.persist_batch(&batch, "batch-a", 0).await.unwrap();
        assert_eq!(store.mock_point_count().await, 10);

        // A different batch_id produces distinct points
        store.persist_batch(&batch, "batch-b", 0).await.unwrap();
        assert_eq!(store.mock_point_count().await, 20);

        // Same batch_id at a different row offset also produces distinct points
        store.persist_batch(&batch, "batch-a", 10).await.unwrap();
        assert_eq!(store.mock_point_count().await, 30);
    }

    #[tokio::test]
    async fn test_persist_batch_rejects_dim_mismatch() {
        let config = ExperienceStoreConfig {
            vector_dim: 4, // test batches carry 9-dim vectors
            ..Default::default()
        };
        let store = ExperienceStore::new(config).await.unwrap();
        let batch = create_test_batch(3);

        let result = store.persist_batch(&batch, "batch-dim", 0).await;
        assert!(result.is_err(), "expected dim mismatch to hard-fail");
        let msg = format!("{:#}", result.unwrap_err());
        assert!(msg.contains("configured collection dim"), "got: {msg}");

        // Nothing partially persisted
        assert_eq!(store.mock_point_count().await, 0);
    }

    #[test]
    fn test_experience_point_id_deterministic() {
        let a = experience_point_id("batch-1", 0);
        let b = experience_point_id("batch-1", 0);
        assert_eq!(a, b);

        assert_ne!(
            experience_point_id("batch-1", 0),
            experience_point_id("batch-1", 1)
        );
        assert_ne!(
            experience_point_id("batch-1", 0),
            experience_point_id("batch-2", 0)
        );

        // Valid UUID, version 5
        let parsed = uuid::Uuid::parse_str(&a).unwrap();
        assert_eq!(parsed.get_version_num(), 5);
    }

    #[tokio::test]
    async fn test_mock_persist_rows() {
        let store = ExperienceStore::mock();

        let rows = vec![
            ExperienceRow {
                state_vector: vec![1.0; 9],
                state_raw: None,
                action_type: 0,
                action_symbol: "ETHUSD".into(),
                action_qty: 1.0,
                reward: 0.5,
                next_state_vector: vec![2.0; 9],
                next_state_raw: None,
                done: false,
                timestamp_ms: 1_700_000_000_000,
                meta: None,
            },
            ExperienceRow {
                state_vector: vec![3.0; 9],
                state_raw: Some(vec![0.1, 0.2]),
                action_type: 1,
                action_symbol: "BTCUSD".into(),
                action_qty: 0.5,
                reward: -0.2,
                next_state_vector: vec![4.0; 9],
                next_state_raw: Some(vec![0.3, 0.4]),
                done: true,
                timestamp_ms: 1_700_000_001_000,
                meta: None,
            },
        ];

        let persisted = store.persist_rows(&rows).await.unwrap();
        assert_eq!(persisted, 2);
        assert_eq!(store.mock_point_count().await, 2);
    }

    #[tokio::test]
    async fn test_mock_persist_batch_chunked() {
        let config = ExperienceStoreConfig {
            upsert_batch_size: 3, // force chunking
            ..Default::default()
        };

        let store = ExperienceStore::new(config).await.unwrap();
        let batch = create_test_batch(10);

        let persisted = store
            .persist_batch(&batch, "batch-chunked", 0)
            .await
            .unwrap();
        assert_eq!(persisted, 10);
        assert_eq!(store.mock_point_count().await, 10);

        let m = store.metrics().await;
        // 10 rows / 3 per chunk = 4 upsert calls (3+3+3+1)
        assert_eq!(m.upsert_calls, 4);
    }

    #[tokio::test]
    async fn test_mock_clear() {
        let store = ExperienceStore::mock();
        let batch = create_test_batch(5);
        store.persist_batch(&batch, "batch-clear", 0).await.unwrap();
        assert_eq!(store.mock_point_count().await, 5);

        store.mock_clear().await;
        assert_eq!(store.mock_point_count().await, 0);
    }

    #[tokio::test]
    async fn test_ensure_collection_mock() {
        let store = ExperienceStore::mock();
        // Should not error in mock mode
        store.ensure_collection().await.unwrap();
    }

    #[test]
    fn test_row_to_point() {
        let store = ExperienceStore::mock();
        let row = ExperienceRow {
            state_vector: vec![1.0, 2.0, 3.0],
            state_raw: None,
            action_type: 2,
            action_symbol: "BTCUSD".into(),
            action_qty: 0.5,
            reward: 1.23,
            next_state_vector: vec![4.0, 5.0, 6.0],
            next_state_raw: None,
            done: false,
            timestamp_ms: 1_700_000_000_000,
            meta: None,
        };

        let point = store.row_to_point(experience_point_id("batch-p", 0), &row);

        // Should have a UUID-format ID
        assert!(point.id.is_some());

        // Check payload fields exist
        let payload = &point.payload;
        assert!(payload.contains_key("action_type"));
        assert!(payload.contains_key("action_symbol"));
        assert!(payload.contains_key("reward"));
        assert!(payload.contains_key("done"));
        assert!(payload.contains_key("timestamp_ms"));
        assert!(payload.contains_key("next_state_vector"));
    }

    #[test]
    fn test_experience_row_serde() {
        let row = ExperienceRow {
            state_vector: vec![1.0, 2.0],
            state_raw: Some(vec![0.5]),
            action_type: 1,
            action_symbol: "ETHUSD".into(),
            action_qty: 2.0,
            reward: -0.1,
            next_state_vector: vec![3.0, 4.0],
            next_state_raw: None,
            done: true,
            timestamp_ms: 123_456_789,
            meta: None,
        };

        let json = serde_json::to_string(&row).unwrap();
        let deserialized: ExperienceRow = serde_json::from_str(&json).unwrap();

        assert_eq!(deserialized.action_symbol, "ETHUSD");
        assert_eq!(deserialized.state_vector.len(), 2);
        assert!(deserialized.done);
    }

    // ── Phase 2: training read mapping (payload → replay tuple) ───────────

    /// Build a `SamplePoint` whose payload mirrors what `sample()` returns
    /// after `qdrant_value_to_json` (integers/doubles/bools/strings), including
    /// the JSON-string-encoded `next_state_vector` written by `row_to_point`.
    fn sample_point_with(
        vector: Vec<f32>,
        action_type: i64,
        reward: f64,
        done: bool,
        next_state: &[f32],
    ) -> SamplePoint {
        let mut payload = serde_json::Map::new();
        payload.insert("action_type".into(), serde_json::json!(action_type));
        payload.insert("reward".into(), serde_json::json!(reward));
        payload.insert("done".into(), serde_json::json!(done));
        payload.insert(
            "next_state_vector".into(),
            serde_json::json!(serde_json::to_string(next_state).unwrap()),
        );
        SamplePoint {
            id: "00000000-0000-0000-0000-000000000001".into(),
            vector,
            payload,
        }
    }

    #[test]
    fn test_training_sample_from_point_maps_all_fields() {
        let point = sample_point_with(vec![0.1, 0.2, 0.3], 1, -0.0007, true, &[0.4, 0.5, 0.6]);
        let sample = training_sample_from_point(&point).expect("should map");
        assert_eq!(sample.state_vector, vec![0.1, 0.2, 0.3]);
        assert_eq!(sample.action, 1);
        assert!((sample.reward - -0.0007).abs() < 1e-9);
        assert_eq!(sample.next_state_vector, vec![0.4, 0.5, 0.6]);
        assert!(sample.done);
    }

    #[test]
    fn test_training_sample_from_point_round_trips_row_to_point_payload() {
        // Full contract check: persist path (row_to_point) → scroll decode
        // (qdrant_value_to_json) → training mapping recovers the replay tuple.
        let store = ExperienceStore::mock();
        let row = ExperienceRow {
            state_vector: vec![1.0, 2.0, 3.0],
            state_raw: None,
            action_type: 0,
            action_symbol: "BTCUSDT".into(),
            action_qty: 0.0,
            reward: 0.0031,
            next_state_vector: vec![1.1, 2.1, 3.1],
            next_state_raw: None,
            done: false,
            timestamp_ms: 1_782_000_000_000,
            meta: None,
        };
        let point = store.row_to_point(experience_point_id("b", 0), &row);
        // Decode the qdrant payload exactly as sample() would.
        let payload: serde_json::Map<String, serde_json::Value> = point
            .payload
            .into_iter()
            .map(|(k, v)| (k, qdrant_value_to_json(v)))
            .collect();
        let sp = SamplePoint {
            id: "x".into(),
            vector: row.state_vector.clone(),
            payload,
        };
        let sample = training_sample_from_point(&sp).expect("should map");
        assert_eq!(sample.state_vector, row.state_vector);
        assert_eq!(sample.action, row.action_type);
        assert!((sample.reward - row.reward).abs() < 1e-6);
        assert_eq!(sample.next_state_vector, row.next_state_vector);
        assert_eq!(sample.done, row.done);
    }

    #[test]
    fn test_training_sample_from_point_skips_incomplete() {
        // Empty vector → skipped.
        let mut p = sample_point_with(vec![], 2, 0.0, false, &[0.1]);
        assert!(training_sample_from_point(&p).is_none());

        // Missing reward → skipped (pre-Phase-2 point).
        p = sample_point_with(vec![0.1], 2, 0.0, false, &[0.1]);
        p.payload.remove("reward");
        assert!(training_sample_from_point(&p).is_none());

        // Missing next_state_vector → skipped.
        p = sample_point_with(vec![0.1], 2, 0.0, false, &[0.1]);
        p.payload.remove("next_state_vector");
        assert!(training_sample_from_point(&p).is_none());

        // Empty next_state_vector JSON → skipped.
        p = sample_point_with(vec![0.1], 2, 0.0, false, &[]);
        assert!(training_sample_from_point(&p).is_none());

        // Out-of-range action index → skipped.
        p = sample_point_with(vec![0.1], 7, 0.0, false, &[0.1]);
        assert!(training_sample_from_point(&p).is_none());
    }

    #[tokio::test]
    async fn test_sample_for_training_mock_is_empty() {
        // Mock mode has no Qdrant to scroll; it must return an honest empty
        // result rather than fabricating training data.
        let store = ExperienceStore::mock();
        let samples = store.sample_for_training(100, None).await.unwrap();
        assert!(samples.is_empty());
    }
}
